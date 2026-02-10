# coding: utf-8
import os
import aiohttp
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from telegram.error import TimedOut, NetworkError

from memory import init_db, save_message, load_context

# ================== НАСТРОЙКИ ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")

MODEL_NAME = "local-model"
MAX_TOKENS = 1500
TIMEOUT = 40

if not TELEGRAM_TOKEN or ":" not in TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не найден или неверный")

# ================== ТРИГГЕРЫ ==================
SEARCH_TRIGGERS = [
    "сегодня", "сейчас", "курс", "цена",
    "новости", "актуально", "доллар",
    "евро", "биткоин", "погода"
]

UNCERTAINTY_PHRASES = [
    "не знаю", "не уверен", "нет информации",
    "неизвестно"
]

# ================== LM STUDIO ==================
async def ask_lm_studio(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Ты полезный ИИ ассистент."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": MAX_TOKENS,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LM_STUDIO_URL}/v1/chat/completions",
                json=payload,
                timeout=TIMEOUT
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Ошибка LM Studio: {e}"

# ================== ИНТЕРНЕТ ПОИСК ==================
async def internet_search(query: str) -> str:
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=20) as resp:
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        snippets = soup.select(".result__snippet")
        results = [s.get_text(strip=True) for s in snippets[:5]]

        return "\n".join(results) if results else "Актуальная информация не найдена."
    except Exception:
        return "Ошибка получения данных из интернета."

# ================== ЛОГИКА ==================
def needs_search(user_text: str, answer: str) -> bool:
    t = user_text.lower()
    if any(k in t for k in SEARCH_TRIGGERS):
        return True
    if any(p in answer.lower() for p in UNCERTAINTY_PHRASES):
        return True
    return False

# ================== SAFE SEND ==================
async def safe_send(update: Update, text: str):
    try:
        await update.message.reply_text(
            text,
            disable_web_page_preview=True
        )
    except (TimedOut, NetworkError):
        pass
    except Exception:
        pass

# ================== HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(
        update,
        "🤖 AI бот с памятью\n\n"
        "• LM Studio\n"
        "• Интернет при необходимости\n"
        "• Память между сессиями\n\n"
        "Задай вопрос 👇"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    if not user_text:
        return

    user_id = update.message.from_user.id

    save_message(user_id, "user", user_text)

    try:
        await update.message.chat.send_action("typing")
    except Exception:
        pass

    past_context = load_context(user_id)

    prompt = (
        f"Контекст диалога:\n{past_context}\n\n"
        f"Текущий вопрос:\n{user_text}"
    )

    answer = await ask_lm_studio(prompt)

    if needs_search(user_text, answer):
        search_data = await internet_search(user_text)
        prompt = (
            f"Контекст диалога:\n{past_context}\n\n"
            f"Актуальные данные:\n{search_data}\n\n"
            f"Вопрос:\n{user_text}"
        )
        answer = await ask_lm_studio(prompt)

    save_message(user_id, "assistant", answer)

    for part in split_text(answer):
        await safe_send(update, part)

# ================== ERROR HANDLER ==================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    if isinstance(context.error, (TimedOut, NetworkError)):
        return

# ================== UTILS ==================
def split_text(text: str, limit: int = 4096):
    return [text[i:i + limit] for i in range(0, len(text), limit)]

# ================== MAIN ==================
def main():
    print("✅ TELEGRAM TOKEN OK")
    print("🤖 Запуск бота...")

    init_db()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🚀 Бот успешно запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
