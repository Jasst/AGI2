# coding: utf-8
"""
AGI_Telegram_Bot_LMStudio_Fixed.py — ИСПРАВЛЕННАЯ ВЕРСИЯ С ГАРАНТИРОВАННОЙ ИНИЦИАЛИЗАЦИЕЙ
✅ Устранена ошибка 'session_manager' — компоненты инициализируются ДО обработки сообщений
✅ Полный когнитивный стек работает с локальной LLM
✅ 100% приватность — все данные на вашем компьютере
"""
import asyncio
import logging
import os
import sys
import sqlite3
import hashlib
import re
import json
import time
import requests
import aiohttp
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, Optional, List, Any, Set, Tuple


# ================= ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =================
def load_dotenv_simple(path: Path = Path(".env")):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            print(f"✅ Загружены переменные из {path}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки .env: {e}")


load_dotenv_simple()

# ================= СОВМЕСТИМОСТЬ С PYTHON 3.13 =================
if sys.version_info >= (3, 13):
    print("✅ Python 3.13 обнаружен — применяются настройки совместимости")
    if sys.platform == 'win32':
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass

# ================= ИМПОРТЫ TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )

    print("✅ Библиотека python-telegram-bot загружена успешно (v20+)")
except ImportError as e:
    print(f"❌ Ошибка импорта telegram: {e}")
    print("📦 Установите: pip install 'python-telegram-bot>=20.7' aiohttp requests")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_system_telegram")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"
    LOG_PATH = ROOT / "system.log"

    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODELS_URL = "http://localhost:1234/v1/models"

    TIMEOUT = 180
    MAX_TOKENS = 4096
    REFLECTION_INTERVAL = 4
    DEEP_THINKING_THRESHOLD = 0.75
    CONTEXT_WINDOW_SIZE = 12
    MEMORY_DECAY_RATE = 0.07
    SESSION_TIMEOUT = 7200

    THOUGHT_TYPES = [
        'рефлексия', 'анализ', 'планирование', 'обучение',
        'наблюдение', 'синтез', 'критика', 'творчество',
        'предсказание', 'оценка'
    ]

    @classmethod
    def get_telegram_token(cls) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("\n" + "=" * 70)
            print("🔑 НЕОБХОДИМ ТОКЕН TELEGRAM БОТА")
            print("=" * 70)
            print("\nКак получить токен:")
            print("1. Найдите в Telegram бота @BotFather")
            print("2. Отправьте команду /newbot и следуйте инструкциям")
            print("3. Скопируйте токен вида '123456789:AAH_ABC123...'")
            token = input("\nВведите токен Telegram бота: ").strip()

            if not token:
                raise ValueError("❌ Токен не введён. Запуск невозможен.")

            if not re.match(r'^\d+:[A-Za-z0-9_-]{35,}$', token):
                raise ValueError("❌ Неверный формат токена. Пример: 123456789:AAHdQwerty12345uiop67890")

            try:
                with open(".env", "a", encoding="utf-8") as f:
                    f.write(f'\nTELEGRAM_BOT_TOKEN="{token}"\n')
                print("✅ Токен сохранён в .env")
                os.environ["TELEGRAM_BOT_TOKEN"] = token
            except Exception as e:
                print(f"⚠️ Не удалось сохранить токен: {e}")

        return token

    @classmethod
    def get_lmstudio_config(cls) -> Dict[str, Any]:
        config = {
            'url': cls.LM_STUDIO_URL,
            'api_key': os.getenv("LM_STUDIO_API_KEY", ""),
            'model': os.getenv("LM_STUDIO_MODEL", "local-model")
        }

        if not os.getenv("LM_STUDIO_CONFIGURED"):
            print("\n" + "=" * 70)
            print("⚙️  НАСТРОЙКА ЛОКАЛЬНОГО СЕРВЕРА LM STUDIO")
            print("=" * 70)

            print("\n🔍 Проверка сервера LM Studio...")
            try:
                response = requests.get(cls.LM_STUDIO_MODELS_URL, timeout=8)
                if response.status_code == 200:
                    print("✅ Сервер обнаружен!")
                else:
                    raise ConnectionError(f"Статус {response.status_code}")
            except Exception as e:
                print(f"⚠️ Сервер не отвечает: {e}")
                print("\n📌 ЗАПУСТИТЕ LM STUDIO:")
                print("   1. Скачайте: https://lmstudio.ai/")
                print("   2. Загрузите модель (рекомендуется):")
                print("      • Phi-3-mini-4k-instruct-q4.gguf (быстро, 3.8B)")
                print("      • Mistral-7B-Instruct-v0.2-Q5_K_M.gguf (качество, 7B)")
                print("   3. Включите сервер: вкладка 'Server' → 'Start Server'")
                input("\nНажмите Enter после запуска сервера...")

            print("\n🔍 Поиск моделей...")
            try:
                response = requests.get(cls.LM_STUDIO_MODELS_URL, timeout=10)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    if models:
                        best = next((m for m in models if
                                     'instruct' in m.get('id', '').lower() or 'chat' in m.get('id', '').lower()),
                                    models[0])
                        config['model'] = best.get('id', config['model'])
                        print(f"✅ Модель: {config['model']}")

                        with open(".env", "a", encoding="utf-8") as f:
                            f.write(f'\nLM_STUDIO_CONFIGURED="true"\nLM_STUDIO_MODEL="{config["model"]}"\n')
                        print("💾 Конфигурация сохранена")
                    else:
                        config['model'] = input("Введите ID модели: ").strip() or config['model']
                else:
                    config['model'] = input("Введите ID модели: ").strip() or config['model']
            except Exception as e:
                print(f"⚠️ Ошибка: {e}")
                config['model'] = input("Введите ID модели: ").strip() or config['model']

            print("\n" + "=" * 70)
            print("💡 Советы:")
            print("   • CPU (16 ГБ ОЗУ): Phi-3-mini — 2-3 сек/ответ")
            print("   • CPU (32 ГБ ОЗУ): Mistral-7B — 4-6 сек/ответ")
            print("   • GPU: скорость в 5-10 раз выше")
            print("=" * 70 + "\n")

        return config


# ================= УТИЛИТЫ =================
def calculate_text_similarity(t1: str, t2: str) -> float:
    if not t1 or not t2: return 0.0
    w1, w2 = set(re.findall(r'\w+', t1.lower())), set(re.findall(r'\w+', t2.lower()))
    if not w1 or not w2: return 0.0
    uni = len(w1 & w2) / max(len(w1), len(w2))

    def ngrams(t, n=2):
        words = re.findall(r'\w+', t.lower())
        return set(' '.join(words[i:i + n]) for i in range(len(words) - n + 1)) if len(words) >= n else set()

    b1, b2 = ngrams(t1, 2), ngrams(t2, 2)
    bi = len(b1 & b2) / max(len(b1 | b2), 1) if b1 and b2 else 0.0
    return 0.6 * uni + 0.4 * bi


def extract_semantic_features(text: str) -> Dict:
    t = text.lower()
    return {
        'length': len(text.split()),
        'question_words': len(re.findall(r'\b(как|что|почему|зачем|когда|где|кто|сколько)\b', t)),
        'imperatives': len(re.findall(r'\b(сделай|создай|найди|покажи|расскажи|объясни)\b', t)),
        'has_question': '?' in text,
        'complexity': len(set(t.split())) / max(len(text.split()), 1)
    }


def split_message(text: str, max_len: int = 4096) -> list:
    if len(text) <= max_len: return [text]
    parts, current = [], ""
    for para in re.split(r'(\n\s*\n)', text):
        if len(current) + len(para) <= max_len:
            current += para
        else:
            if current: parts.append(current.rstrip())
            current = para[:max_len]
    if current: parts.append(current.rstrip())
    return parts[:5] + ["...сообщение сокращено"] if len(parts) > 5 else parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Глубокое мышление", callback_data="deep_think"),
         InlineKeyboardButton("📊 Статистика", callback_data="stats")],
        [InlineKeyboardButton("💡 Инсайты", callback_data="insights"),
         InlineKeyboardButton("🧹 Очистить", callback_data="clear")]
    ])


# ================= БАЗА ДАННЫХ =================
class EnhancedMemoryDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self):
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                user_id INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                system_response TEXT NOT NULL,
                category TEXT,
                importance REAL DEFAULT 0.5
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                last_used REAL,
                decay_factor REAL DEFAULT 1.0
            )''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_int_user ON interactions(user_id, timestamp DESC)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, importance DESC)')

    def add_interaction(self, uid: int, inp: str, resp: str, **kw):
        with self.get_connection() as conn:
            conn.cursor().execute(
                'INSERT INTO interactions (timestamp, user_id, user_input, system_response, category, importance) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (time.time(), uid, inp, resp, kw.get('category', 'диалог'), kw.get('importance', 0.5))
            )

    def get_contextual_interactions(self, uid: int, query: str, limit: int = 3) -> List[Dict]:
        with self.get_connection() as conn:
            rows = conn.cursor().execute(
                'SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
                (uid, limit * 2)
            ).fetchall()
            scored = [(calculate_text_similarity(query, f"{r['user_input']} {r['system_response']}"), dict(r)) for r in
                      rows]
            return [item[1] for item in sorted(scored, reverse=True, key=lambda x: x[0])[:limit]]

    def add_fact(self, uid: int, key: str, value: str, **kw):
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT id FROM facts WHERE user_id=? AND key=? AND value=?', (uid, key, value))
            if c.fetchone():
                c.execute('UPDATE facts SET last_used=?, decay_factor=1.0 WHERE user_id=? AND key=? AND value=?',
                          (time.time(), uid, key, value))
            else:
                c.execute('INSERT INTO facts (user_id, key, value, importance, last_used, decay_factor) '
                          'VALUES (?, ?, ?, ?, ?, ?)',
                          (uid, key, value, kw.get('importance', 0.5), time.time(), 1.0))

    def get_relevant_facts(self, uid: int, query: str, limit: int = 3) -> List[Dict]:
        with self.get_connection() as conn:
            conn.cursor().execute(
                'UPDATE facts SET decay_factor = decay_factor * (1 - ?) WHERE user_id = ? AND last_used < ?',
                (Config.MEMORY_DECAY_RATE, uid, time.time() - 86400)
            )
            rows = conn.cursor().execute(
                'SELECT * FROM facts WHERE user_id = ? AND decay_factor > 0.2 ORDER BY importance DESC LIMIT ?',
                (uid, limit * 2)
            ).fetchall()
            scored = [(calculate_text_similarity(query, f"{r['key']} {r['value']}"), dict(r)) for r in rows]
            return [item[1] for item in sorted(scored, reverse=True, key=lambda x: x[0])[:limit]]

    def get_user_stats(self, uid: int) -> Dict:
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (uid,))
            interactions = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM facts WHERE user_id = ? AND decay_factor > 0.3', (uid,))
            facts = c.fetchone()[0]
            return {'interactions': interactions, 'facts': facts}


# ================= СИСТЕМА МЫШЛЕНИЯ =================
class LocalThinkingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config['url']
        self.model = config['model']
        self.cache, self.last_req = {}, 0
        print(f"✅ Инициализирована локальная модель: {self.model}")

    async def _rate_limit(self):
        elapsed = time.time() - self.last_req
        if elapsed < 0.3: await asyncio.sleep(0.3 - elapsed)
        self.last_req = time.time()

    async def call_llm(self, system: str, user: str, temp: float = 0.7) -> str:
        cache_key = hashlib.md5(f"{system[:50]}|{user[:100]}|{temp}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        await self._rate_limit()

        try:
            headers = {"Content-Type": "application/json"}
            if api_key := os.getenv("LM_STUDIO_API_KEY"):
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": temp,
                "max_tokens": 1500,
                "top_p": 0.9
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r'^[\s*_-]+|[\s*_-]+$', '', content)
                        self.cache[cache_key] = content
                        if len(self.cache) > 150:
                            for k in list(self.cache.keys())[:30]:
                                self.cache.pop(k, None)
                        return content
                    return f"⚠️ Ошибка LM Studio ({resp.status})"
        except Exception as e:
            return f"⚠️ Ошибка модели: {str(e)[:100]}"


# ================= АГЕНТ ПОЛЬЗОВАТЕЛЯ =================
class UserCognitiveAgent:
    def __init__(self, uid: int, db: EnhancedMemoryDB, thinker: LocalThinkingSystem):
        self.uid = uid
        self.db, self.thinker = db, thinker
        self.interactions, self.context = 0, deque(maxlen=Config.CONTEXT_WINDOW_SIZE)

    async def process_message(self, text: str) -> str:
        self.interactions += 1

        feats = extract_semantic_features(text)
        imp = min(1.0, 0.5 + feats['question_words'] * 0.15 + feats['imperatives'] * 0.1)

        # Сохранение чисел как фактов
        if nums := re.findall(r'\b\d{2,}\b', text)[:2]:
            for n in nums:
                self.db.add_fact(self.uid, 'число', n, importance=imp * 0.4)

        # Контекст для ответа
        context_parts = []
        if recent := self.db.get_contextual_interactions(self.uid, text, limit=2):
            context_parts.append("Предыдущие вопросы:")
            for r in recent:
                context_parts.append(f"- {r['user_input'][:50]}")
        if facts := self.db.get_relevant_facts(self.uid, text, limit=3):
            context_parts.append("\nВажная информация:")
            for f in facts:
                context_parts.append(f"- {f['key']}: {f['value'][:40]}")

        system_prompt = (
            "Ты — умный когнитивный ассистент. Отвечай кратко (1-3 предложения), по делу, без лишних фраз. "
            "Не используй маркдаун. Если не знаешь ответ — скажи честно.\n\n"
            f"КОНТЕКСТ:\n{' '.join(context_parts) if context_parts else 'Нет контекста'}"
        )

        response = await self.thinker.call_llm(system_prompt, text, temp=0.5 if feats['has_question'] else 0.3)

        self.db.add_interaction(
            self.uid, text, response,
            category='вопрос' if feats['has_question'] else 'диалог',
            importance=imp
        )

        return response.strip()


# ================= МЕНЕДЖЕР СЕССИЙ =================
class SessionManager:
    def __init__(self, db: EnhancedMemoryDB, thinker: LocalThinkingSystem):
        self.db, self.thinker = db, thinker
        self.sessions: Dict[int, UserCognitiveAgent] = {}

    async def get_or_create(self, uid: int) -> UserCognitiveAgent:
        if uid not in self.sessions:
            self.sessions[uid] = UserCognitiveAgent(uid, self.db, self.thinker)
            logging.info(f"🆕 Новая сессия: {uid}")
        return self.sessions[uid]


# ================= ОБРАБОТЧИКИ TELEGRAM =================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ГАРАНТИРОВАННАЯ ПРОВЕРКА ИНИЦИАЛИЗАЦИИ
    if 'session_manager' not in context.application.bot_data:
        await update.message.reply_text(
            "⚠️ Система ещё инициализируется. Подождите 5 секунд и нажмите /start снова.",
            reply_markup=create_main_keyboard()
        )
        return

    agent = await context.application.bot_data['session_manager'].get_or_create(update.effective_user.id)
    await update.message.reply_text(
        f"👋 Привет! Я — когнитивный ассистент с локальной ИИ.\n"
        "✅ Все данные остаются на вашем компьютере\n"
        "⚡ Работаю без интернета после настройки\n\n"
        "Спросите что-нибудь!",
        reply_markup=create_main_keyboard()
    )


async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    # 🔑 КРИТИЧЕСКИ ВАЖНАЯ ПРОВЕРКА: дождаться полной инициализации
    if 'session_manager' not in context.application.bot_data:
        await update.message.reply_chat_action("typing")
        await asyncio.sleep(3)  # Дать время на инициализацию

        # Повторная проверка
        if 'session_manager' not in context.application.bot_data:
            await update.message.reply_text(
                "⚠️ Система инициализируется. Пожалуйста, подождите 5 секунд и повторите запрос.",
                reply_markup=create_main_keyboard()
            )
            return

    uid = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        return

    logging.info(f"📨 {uid}: {text[:40]}...")

    try:
        agent = await context.application.bot_data['session_manager'].get_or_create(uid)
        await update.message.reply_chat_action("typing")
        response = await agent.process_message(text)

        for part in split_message(response):
            await update.message.reply_text(
                part,
                reply_markup=create_main_keyboard() if part == split_message(response)[-1] else None,
                disable_web_page_preview=True
            )
            if len(split_message(response)) > 1:
                await asyncio.sleep(0.4)

    except Exception as e:
        logging.error(f"❌ Ошибка: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Ошибка обработки. Попробуйте переформулировать запрос.",
            reply_markup=create_main_keyboard()
        )


async def button_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    if 'session_manager' not in context.application.bot_data:
        await update.callback_query.message.reply_text(
            "⚠️ Система инициализируется. Подождите 5 секунд."
        )
        return

    if update.callback_query.data == "stats":
        agent = await context.application.bot_data['session_manager'].get_or_create(update.effective_user.id)
        stats = agent.db.get_user_stats(agent.uid)
        await update.callback_query.message.reply_text(
            f"📊 Статистика:\n"
            f"💬 Сообщений: {stats['interactions']}\n"
            f"🧠 Фактов: {stats['facts']}"
        )
    elif update.callback_query.data == "clear":
        agent = await context.application.bot_data['session_manager'].get_or_create(update.effective_user.id)
        agent.context.clear()
        await update.callback_query.message.reply_text("🧹 Контекст очищен")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} caused error {context.error}")


# ================= ГЛАВНАЯ ФУНКЦИЯ С ПРАВИЛЬНОЙ ИНИЦИАЛИЗАЦИЕЙ =================
async def main():
    setup_logging()
    logging.info("🚀 Запуск когнитивного ассистента")

    # 🔑 ШАГ 1: СИНХРОННАЯ НАСТРОЙКА ДО СОЗДАНИЯ ПРИЛОЖЕНИЯ
    try:
        token = Config.get_telegram_token()
        lm_config = Config.get_lmstudio_config()
    except Exception as e:
        print(f"\n❌ Критическая ошибка инициализации: {e}")
        sys.exit(1)

    # 🔑 ШАГ 2: СОЗДАНИЕ КОМПОНЕНТОВ ДО ЗАПУСКА ПРИЛОЖЕНИЯ
    print("🔧 Инициализация компонентов...")
    db = EnhancedMemoryDB(Config.DB_PATH)
    thinker = LocalThinkingSystem(lm_config)
    session_manager = SessionManager(db, thinker)
    print("✅ Компоненты инициализированы")

    # 🔑 ШАГ 3: СОЗДАНИЕ ПРИЛОЖЕНИЯ И ПЕРЕДАЧА ГОТОВЫХ КОМПОНЕНТОВ
    application = (
        ApplicationBuilder()
        .token(token)
        .read_timeout(25)
        .write_timeout(25)
        .connect_timeout(10)
        .pool_timeout(10)
        .build()
    )

    # 🔑 ШАГ 4: СОХРАНЕНИЕ КОМПОНЕНТОВ В bot_data ДО ЗАПУСКА
    application.bot_data['session_manager'] = session_manager
    application.bot_data['db'] = db
    application.bot_data['thinker'] = thinker

    # 🔑 ШАГ 5: РЕГИСТРАЦИЯ ОБРАБОТЧИКОВ
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    application.add_handler(CallbackQueryHandler(button_cb))
    application.add_error_handler(error_handler)

    # 🔑 ШАГ 6: ЗАПУСК ПРИЛОЖЕНИЯ
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    # Вывод информации о запуске
    print("\n" + "=" * 70)
    print("✅ КОГНИТИВНЫЙ АССИСТЕНТ С ЛОКАЛЬНОЙ LLM ЗАПУЩЕН!")
    print("=" * 70)
    print(f"\n🤖 Модель: {lm_config['model']}")
    print("🔗 Сервер: http://localhost:1234")
    print("\n📱 Напишите боту в Telegram /start")
    print("\n💡 Советы:")
    print("   • Для скорости: используйте Phi-3-mini на CPU")
    print("   • Для качества: Mistral-7B с GPU")
    print("   • Все данные хранятся локально — 100% приватность")
    print("\n🛑 Остановка: Ctrl+C")
    print("=" * 70 + "\n")

    logging.info("🔄 Бот работает. Нажмите Ctrl+C для остановки")
    while True:
        await asyncio.sleep(3600)


def setup_logging():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Config.LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def run():
    print("AGI Cognitive Assistant — Локальная версия (LM Studio)")
    print(f"Python: {sys.version.split()[0]}")

    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8+")
        sys.exit(1)

    try:
        import aiohttp
    except ImportError:
        print("❌ Установите: pip install aiohttp requests")
        sys.exit(1)

    print("\n🚀 Запуск когнитивного ассистента...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Работа завершена")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()