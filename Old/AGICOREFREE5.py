# coding: utf-8
import os
import aiohttp
import sqlite3
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pytz

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from telegram.error import TimedOut, NetworkError

# ================== ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ==================
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")

MODEL_NAME = "local-model"
MAX_TOKENS = 1500
TIMEOUT = 40
SUMMARY_TRIGGER = 20
CACHE = {}

if not TELEGRAM_TOKEN or ":" not in TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не найден или неверный")

# ================== ТРИГГЕРЫ ==================
SEARCH_TRIGGERS = ["сегодня", "сейчас", "курс", "цена", "новости", "актуально", "доллар", "евро", "биткоин", "погода"]
UNCERTAINTY_PHRASES = ["не знаю", "не уверен", "нет информации", "неизвестно"]
CRYPTO_KEYWORDS = ["биткоин", "bitcoin", "btc", "курс биткоина", "цена биткоина"]
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
DB_NAME = "memory.db"

# ================== ИНИЦИАЛИЗАЦИЯ БД ==================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS summary (
            user_id INTEGER PRIMARY KEY,
            content TEXT,
            updated TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS metaknowledge (
            user_id INTEGER,
            fact TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================== ПАМЯТЬ ==================
def save_message(user_id: int, role: str, content: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def load_recent(user_id: int, limit: int = 6) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    return "\n".join(f"{r}: {c}" for r, c in rows)

def load_summary(user_id: int) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT content FROM summary WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

def should_summarize(user_id: int) -> bool:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,))
    count = c.fetchone()[0]
    conn.close()
    return count >= SUMMARY_TRIGGER

def replace_with_summary(user_id: int, summary_text: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    c.execute("""
        INSERT OR REPLACE INTO summary (user_id, content, updated)
        VALUES (?, ?, ?)
    """, (user_id, summary_text, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# ================== МЕТАПАМЯТЬ ==================
def save_metafact(user_id: int, fact: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO metaknowledge (user_id, fact, timestamp) VALUES (?, ?, ?)",
        (user_id, fact, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def load_metaknowledge(user_id: int) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT fact FROM metaknowledge WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return "\n".join(r[0] for r in rows)

# ================== УТИЛИТЫ ==================
def split_text(text: str, limit: int = 4096):
    return [text[i:i + limit] for i in range(0, len(text), limit)]

def is_crypto_query(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CRYPTO_KEYWORDS)

def must_use_internet(text: str) -> bool:
    if is_crypto_query(text):
        return True
    t = text.lower()
    return any(k in t for k in SEARCH_TRIGGERS)

def needs_search(user_text: str, answer: str) -> bool:
    t = user_text.lower()
    if any(k in t for k in SEARCH_TRIGGERS):
        return True
    if any(p in answer.lower() for p in UNCERTAINTY_PHRASES):
        return True
    return False

def current_time_info(tz_name="Europe/Moscow"):
    tz = pytz.timezone(tz_name)
    now_local = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(tz)
    return now_local.strftime("%d.%m.%Y %H:%M:%S %Z")

# ================== LM STUDIO ==================
async def ask_model(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Ты полезный когнитивный ИИ ассистент."},
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
    if query in CACHE:
        return CACHE[query]
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=20) as resp:
                html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        snippets = soup.select(".result__snippet")
        result = "\n".join(s.get_text(strip=True) for s in snippets[:5])
        CACHE[query] = result or "Актуальная информация не найдена."
        return CACHE[query]
    except Exception:
        return "Ошибка интернет-поиска."

async def get_bitcoin_price() -> str:
    params = {"ids": "bitcoin", "vs_currencies": "usd,eur"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(COINGECKO_URL, params=params, timeout=15) as resp:
                data = await resp.json()
        btc = data.get("bitcoin", {})
        usd = btc.get("usd")
        eur = btc.get("eur")
        if usd:
            return f"₿ Bitcoin:\nUSD: ${usd}\nEUR: €{eur}"
        else:
            return "Не удалось получить цену биткоина."
    except Exception:
        return "Ошибка получения данных о биткоине."

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

# ================== АВТОНОМНОЕ УТОЧНЕНИЕ ФАКТОВ ==================
async def refine_answer(user_id: int, user_text: str, answer: str) -> str:
    if any(p in answer.lower() for p in UNCERTAINTY_PHRASES) or needs_search(user_text, answer):
        web_data = await internet_search(user_text)
        meta_context = load_metaknowledge(user_id)
        prompt = (
            "Используй данные из интернета и свои прошлые выводы (метапамять).\n"
            f"META:\n{meta_context}\nWEB:\n{web_data}\n\n"
            f"Пользователь спросил: {user_text}\n"
            f"Первый ответ: {answer}\n"
            "Составь уточнённый, точный и полный ответ без слов 'не знаю' или 'не уверен'."
        )
        refined = await ask_model(prompt)
        meta_prompt = f"Выдели ключевые выводы из уточнённого ответа:\n{refined}"
        meta_fact = await ask_model(meta_prompt)
        save_metafact(user_id, meta_fact)
        return refined
    return answer

# ================== HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(update, f"🤖 Когнитивный AI бот\nДата/время: {current_time_info()}\n\nЗадай любой вопрос 👇")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()
    save_message(user_id, "user", text)

    # 🔥 КРИПТО — СРАЗУ API
    if is_crypto_query(text):
        btc_price = await get_bitcoin_price()
        save_message(user_id, "assistant", btc_price)
        await safe_send(update, btc_price)
        return

    # 🧠 Загружаем контекст и метапознание
    summary = load_summary(user_id)
    recent = load_recent(user_id)
    meta_context = load_metaknowledge(user_id)
    context_text = ""
    if summary:
        context_text += f"SUMMARY:\n{summary}\n\n"
    if meta_context:
        context_text += f"META-KNOWLEDGE:\n{meta_context}\n\n"
    context_text += f"DIALOG:\n{recent}\n\n"

    # 🌐 Интернет-данные
    if must_use_internet(text):
        web_data = await internet_search(text)
        prompt = (
            "Используй данные ниже и НЕ упоминай ограничения доступа.\n\n"
            f"{context_text}\nWEB:\n{web_data}\n\nQUESTION:\n{text}"
        )
    else:
        prompt = f"{context_text}\nQUESTION:\n{text}"

    # 📝 Запрос к модели
    answer = await ask_model(prompt)

    # 🔍 Уточнение фактов
    answer = await refine_answer(user_id, text, answer)
    save_message(user_id, "assistant", answer)

    # 💾 Сжатие summary
    if should_summarize(user_id):
        summary_prompt = f"Сжать диалог для короткой когнитивной памяти:\n{load_recent(user_id, limit=SUMMARY_TRIGGER)}"
        summary_text = await ask_model(summary_prompt)
        replace_with_summary(user_id, summary_text)

    # 📤 Отправка ответа
    await safe_send(update, f"🕒 {current_time_info()}\n\n{answer}")

# ================== ERROR HANDLER ==================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    if isinstance(context.error, (TimedOut, NetworkError)):
        return

# ================== MAIN ==================
def main():
    print("✅ TELEGRAM TOKEN OK")
    print("🤖 Запуск когнитивного бота...")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🚀 Бот успешно запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
