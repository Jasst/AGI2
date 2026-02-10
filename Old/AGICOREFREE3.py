# coding: utf-8
import os
import aiohttp

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

from memory import (
    init_db, save_message, load_recent,
    load_summary, should_summarize, replace_with_summary
)
from reasoning import need_internet
from websearch import search

# ================== CONFIG ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
MODEL_NAME = "local-model"

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM TOKEN missing")

# ================== MODEL ==================
async def ask_model(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Ты полезный и точный ИИ."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 1200
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json=payload,
            timeout=40
        ) as r:
            data = await r.json()
            return data["choices"][0]["message"]["content"]

# ================== HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI Core v2\n"
        "• память\n"
        "• интернет\n"
        "• reasoning\n\n"
        "Задай вопрос 👇"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    save_message(user_id, "user", text)

    summary = load_summary(user_id)
    recent = load_recent(user_id)

    base_context = ""
    if summary:
        base_context += f"SUMMARY:\n{summary}\n\n"
    base_context += f"DIALOG:\n{recent}\n"

    need_web = await need_internet(ask_model, text)

    if need_web:
        web_data = await search(text)
        prompt = f"{base_context}\nWEB:\n{web_data}\n\nQUESTION:\n{text}"
    else:
        prompt = f"{base_context}\nQUESTION:\n{text}"

    answer = await ask_model(prompt)

    save_message(user_id, "assistant", answer)

    if should_summarize(user_id):
        summary_prompt = (
            "Сожми диалог в краткое, полезное резюме:\n\n" +
            load_recent(user_id, 20)
        )
        summary_text = await ask_model(summary_prompt)
        replace_with_summary(user_id, summary_text)

    try:
        await update.message.reply_text(answer)
    except (TimedOut, NetworkError):
        pass

# ================== MAIN ==================
def main():
    init_db()
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 AI Core v2 запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
