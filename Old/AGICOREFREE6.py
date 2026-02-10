# coding: utf-8
import os
import aiohttp
import sqlite3
import pytz
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv

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
TIMEOUT = 60
SUMMARY_TRIGGER = 20  # сообщений до сжатия
CACHE = {}  # кеш интернет-поиска

if not TELEGRAM_TOKEN or ":" not in TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не найден или неверный")

# ================== НАСТРОЙКИ ТРИГГЕРОВ ==================
SEARCH_TRIGGERS = [
    "сегодня", "сейчас", "курс", "цена",
    "новости", "актуально", "доллар",
    "евро", "биткоин", "погода"
]

UNCERTAINTY_PHRASES = [
    "не знаю", "не уверен", "нет информации",
    "неизвестно"
]

CRYPTO_KEYWORDS = [
    "биткоин", "bitcoin", "btc",
    "курс биткоина", "цена биткоина"
]

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
            user_id INTEGER PRIMARY KEY,
            content TEXT,
            updated TEXT
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

def load_metaknowledge(user_id: int) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT content FROM metaknowledge WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

def update_metaknowledge(user_id: int, content: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO metaknowledge (user_id, content, updated)
        VALUES (?, ?, ?)
    """, (user_id, content, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

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
            async with session.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT) as resp:
                if resp.status != 200:
                    return f"Ошибка LM Studio: HTTP {resp.status}"
                data = await resp.json()
                if "choices" not in data or not data["choices"]:
                    return "Ошибка LM Studio: пустой ответ"
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Ошибка LM Studio: {e}"

# ================== ИНТЕРНЕТ ПОИСК ==================
async def internet_search(query: str, max_results=5) -> str:
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
        result = "\n".join(f"- {s.get_text(strip=True)}" for s in snippets[:max_results])
        CACHE[query] = result or "Актуальная информация не найдена."
        return CACHE[query]
    except Exception:
        return "Ошибка получения новостей из интернета."

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

# ================== ВРЕМЯ ==================
def current_time_info(tz_name="Europe/Moscow") -> str:
    tz = pytz.timezone(tz_name)
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    now_local = now_utc.astimezone(tz)
    return now_local.strftime("%d.%m.%Y %H:%M:%S %Z")

def get_time_for_all_regions() -> str:
    regions = {
        "Москва": "Europe/Moscow",
        "Нью-Йорк": "America/New_York",
        "Берлин": "Europe/Berlin",
        "Пекин": "Asia/Shanghai",
        "Тегеран": "Asia/Tehran"
    }
    result = ""
    for city, tz in regions.items():
        result += f"- **{city}:** {current_time_info(tz)}\n"
    return result

# ================== SAFE SEND ==================
async def safe_send(update: Update, text: str):
    try:
        await update.message.reply_text(text, disable_web_page_preview=True)
    except (TimedOut, NetworkError):
        pass
    except Exception:
        pass

# ================== HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    times = get_time_for_all_regions()
    await safe_send(update,
        f"🕒 {current_time_info()} MSK\n\n"
        f"**Привет!**\n\n"
        f"Текущая дата и время по регионам:\n{times}\n\n"
        "Задай любой вопрос или запроси новости 👇"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.message.from_user.id
    save_message(user_id, "user", text)

    # 🕒 Ответ на дату/время
    if "дата" in text.lower() or "время" in text.lower():
        times = get_time_for_all_regions()
        await safe_send(update, f"🕒 {current_time_info()} MSK\n\nТекущая дата и время:\n{times}")
        return

    # 🔥 КРИПТО — прямой API
    if is_crypto_query(text):
        btc_price = await get_bitcoin_price()
        save_message(user_id, "assistant", btc_price)
        await safe_send(update, btc_price)
        return

    # 🧠 Загружаем контекст и метапамять
    summary = load_summary(user_id)
    recent = load_recent(user_id)
    metaknow = load_metaknowledge(user_id)
    context_text = ""
    if summary:
        context_text += f"SUMMARY:\n{summary}\n\n"
    if metaknow:
        context_text += f"METAKNOWLEDGE:\n{metaknow}\n\n"
    context_text += f"DIALOG:\n{recent}\n\n"

    # 🌐 Определяем необходимость интернет-данных
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

    # 🔍 Автономная проверка: если ответ сомнительный, уточняем через интернет
    if needs_search(text, answer):
        web_data = await internet_search(text)
        refine_prompt = f"Ответ может быть неточным. Используй данные ниже и исправь ответ:\nWEB:\n{web_data}\nORIGINAL_ANSWER:\n{answer}"
        refined_answer = await ask_model(refine_prompt)
        answer = refined_answer or answer

    save_message(user_id, "assistant", answer)

    # 💾 Обновляем метапознание
    update_metaknowledge(user_id, answer)

    # 💾 Проверка необходимости summary
    if should_summarize(user_id):
        summary_prompt = f"Сжать диалог для краткой когнитивной памяти без потери смысла:\n{load_recent(user_id, limit=SUMMARY_TRIGGER)}"
        summary_text = await ask_model(summary_prompt)
        replace_with_summary(user_id, summary_text)

    # 📤 Отправка ответа
    await safe_send(update, answer)

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
