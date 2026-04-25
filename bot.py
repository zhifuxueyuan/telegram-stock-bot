#!/usr/bin/env python3
"""
Telegram Bot AI Assistant for Hong Kong and US Stock Investment Analysis
Uses Qwen API (OpenAI-compatible) for intelligent responses based on YouTube channel knowledge base
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Bot configuration from environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
BOT_NAME = "致富學院投資助手"

# Initialize Qwen client
def create_client():
    """Create OpenAI-compatible client for Qwen API."""
    if not QWEN_API_KEY:
        logger.error("QWEN_API_KEY is empty!")
        return None
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

ai_client = None

# Knowledge base cache
_kb_cache = None
_sp_cache = None

def load_knowledge_base() -> str:
    """Load and cache the knowledge base."""
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    kb_path = Path(__file__).parent / "youtube_knowledge_base.md"
    if kb_path.exists():
        with open(kb_path, 'r', encoding='utf-8') as f:
            _kb_cache = f.read()
        logger.info(f"Knowledge base loaded: {len(_kb_cache)} chars")
    else:
        _kb_cache = ""
        logger.warning("Knowledge base file not found")
    return _kb_cache

def load_system_prompt() -> str:
    """Load and cache the system prompt."""
    global _sp_cache
    if _sp_cache is not None:
        return _sp_cache
    prompt_path = Path(__file__).parent / "system_prompt.txt"
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            _sp_cache = f.read()
    else:
        _sp_cache = (
            "你是致富學院的專業AI投資助手，由陳Sir和Amy聯合打造，專注於港股和美股投資分析。"
            "請用專業但友好的語氣回答用戶的投資相關問題。"
            "回答時優先參考知識庫中的內容。如果知識庫沒有相關信息，可以使用你的金融知識回答，但要保持專業和客觀。"
        )
    return _sp_cache

def get_relevant_knowledge(query: str) -> str:
    """Extract relevant knowledge based on user query. Limited to 1500 chars."""
    kb = load_knowledge_base()
    if not kb:
        return ""

    query_lower = query.lower()
    lines = kb.split('\n')
    scored_lines = []

    keywords = [
        '港股', '美股', '恆指', '恒指', '納指', '纳指', '股票', '投資', '投资',
        '分析', '展望', '熱門', '热门', '基本面', '交易', '選股', '选股',
        'K線', 'K线', '抄底', '均線', '均线', '量價', '量价', '牛股',
        'Amy', '陳sir', '陈sir', '致富學院', '致富学院',
        'TSLA', 'NVDA', 'AAPL', 'GOOG', 'META', 'AMD', 'AVGO', 'PLTR', 'INTC',
        '騰訊', '腾讯', '阿里', '巴巴', '比亞迪', '比亚迪', '吉利', '蘋果', '苹果',
        '特斯拉', '英偉達', '英伟达', '谷歌'
    ]

    query_words = [w for w in query_lower.split() if len(w) > 1]

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('---'):
            continue
        score = 0
        for kw in keywords:
            if kw in line:
                score += 1
        for w in query_words:
            if w in line.lower():
                score += 3
        if score > 0:
            scored_lines.append((score, line_stripped))

    scored_lines.sort(key=lambda x: x[0], reverse=True)
    result_lines = [line for _, line in scored_lines[:25]]
    result = '\n'.join(result_lines)

    if len(result) > 1500:
        result = result[:1500]
    if not result:
        result = kb[:1000]
    return result

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = (
        f"歡迎使用 {BOT_NAME}！👋\n\n"
        "我是一個專業的港美股投資分析助手，基於致富學院的投資知識庫。\n\n"
        "我可以幫助您：\n"
        "✅ 回答港美股投資相關問題\n"
        "✅ 分析熱門股票\n"
        "✅ 討論投資策略\n"
        "✅ 提供市場分析\n\n"
        "請直接輸入您的問題，我會為您提供專業的回答。"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_message = (
        "使用說明：\n\n"
        "/start - 開始使用\n"
        "/help - 顯示幫助信息\n"
        "/about - 關於此助手\n\n"
        "直接輸入您的投資問題，我會根據致富學院的知識庫為您解答。"
    )
    await update.message.reply_text(help_message)

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    about_message = (
        f"{BOT_NAME}\n\n"
        "這是一個基於致富學院內容的AI投資助手。\n"
        "由陳Sir和Amy聯合打造，專注於港美股投資分析、財經自媒體內容和金融咨詢。\n\n"
        "使用AI模型提供專業、準確的投資分析。\n"
        "所有回答均基於致富學院的教學內容和投資理念。"
    )
    await update.message.reply_text(about_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages and generate AI responses."""
    global ai_client

    user_message = update.message.text
    user_id = update.message.from_user.id
    logger.info(f"User {user_id}: {user_message}")

    await update.message.chat.send_action("typing")

    try:
        if ai_client is None:
            ai_client = create_client()
        if ai_client is None:
            await update.message.reply_text("系統配置錯誤，請聯繫管理員。")
            return

        system_prompt = load_system_prompt()
        relevant_knowledge = get_relevant_knowledge(user_message)

        logger.info(f"Knowledge snippet length: {len(relevant_knowledge)}")

        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n參考資料：\n{relevant_knowledge}"
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        response = ai_client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        ai_response = response.choices[0].message.content
        logger.info(f"AI response length: {len(ai_response)}")

        if len(ai_response) > 4096:
            for i in range(0, len(ai_response), 4096):
                await update.message.reply_text(ai_response[i:i+4096])
        else:
            await update.message.reply_text(ai_response)

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Error for user {user_id}: {error_detail}")
        await update.message.reply_text("抱歉，我遇到了一個問題。請稍後再試。")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update {update} caused error {context.error}")

def main() -> None:
    """Start the bot."""
    logger.info("=" * 50)
    logger.info("Starting Telegram Bot...")
    logger.info(f"TELEGRAM_BOT_TOKEN set: {bool(TELEGRAM_BOT_TOKEN)}")
    logger.info(f"QWEN_API_KEY set: {bool(QWEN_API_KEY)}")

    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set! Exiting.")
        sys.exit(1)
    if not QWEN_API_KEY:
        logger.error("QWEN_API_KEY is not set! Exiting.")
        sys.exit(1)

    # Pre-load caches
    load_knowledge_base()
    load_system_prompt()

    # Test Qwen API connection
    try:
        test_client = create_client()
        test_resp = test_client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10
        )
        logger.info(f"Qwen API test OK: {test_resp.choices[0].message.content}")
    except Exception as e:
        logger.error(f"Qwen API test FAILED: {e}")
        logger.error("Bot will start but AI responses may not work.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    logger.info("Bot is now polling for updates...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
