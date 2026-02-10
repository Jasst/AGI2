# coding: utf-8
"""
AGI_Telegram_Bot.py — ПОЛНОСТЬЮ ИНТЕГРИРОВАННЫЙ КОГНИТИВНЫЙ АГЕНТ ДЛЯ TELEGRAM
Совместим с Python 3.13+, поддержка многопользовательских сессий с изолированной памятью
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
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional, List, Any, Set, Tuple
from dataclasses import dataclass, field

# ================= ПОДГОТОВКА К СОВМЕСТИМОСТИ С PYTHON 3.13 =================
if sys.version_info >= (3, 13):
    print("✅ Python 3.13 обнаружен — применяются специальные настройки совместимости")

# Для Windows: использовать совместимую политику цикла событий
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass  # В новых версиях asyncio это может быть не нужно

# ================= ИМПОРТЫ TELEGRAM =================
try:
    import telegram
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters
    )
    from telegram.error import TelegramError, BadRequest

    print("✅ Библиотека python-telegram-bot загружена успешно (v20+)")
except ImportError as e:
    print(f"❌ Ошибка импорта telegram: {e}")
    print("📦 Установите: pip install 'python-telegram-bot>=20.7' aiohttp")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    """Унифицированная конфигурация системы и бота"""
    ROOT = Path("./cognitive_system_telegram")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"
    CACHE_PATH = ROOT / "cache.json"
    LOG_PATH = ROOT / "system.log"

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    TIMEOUT = 300
    MAX_TOKENS = 8000

    # Когнитивные параметры
    REFLECTION_INTERVAL = 3
    DEEP_THINKING_THRESHOLD = 0.7
    CREATIVITY_FACTOR = 0.8
    LEARNING_RATE = 0.15
    CONTEXT_WINDOW_SIZE = 15
    MEMORY_DECAY_RATE = 0.05
    MAX_MEMORY_ITEMS = 10000

    # Параметры бота
    MAX_MESSAGE_LENGTH = 4096
    MAX_RESPONSE_CHUNKS = 5
    TYPING_DELAY = 0.8
    REQUEST_TIMEOUT = 30
    SESSION_TIMEOUT = 3600  # 1 час неактивности

    THOUGHT_TYPES = [
        'рефлексия', 'анализ', 'планирование', 'обучение',
        'наблюдение', 'синтез', 'критика', 'творчество',
        'предсказание', 'оценка'
    ]

    @classmethod
    def get_api_key(cls) -> str:
        """Получение API ключа OpenRouter"""
        key = os.getenv("OPENROUTER_API_KEY")
        if key and key.strip():
            return key.strip()

        env_path = Path(".env")
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            return line.split("=", 1)[1].strip(' "\'')
            except Exception as e:
                print(f"⚠️ Ошибка чтения .env: {e}")

        print("\n🔑 API ключ OpenRouter не найден.")
        print("📌 Получите ключ на: https://openrouter.ai/keys")
        key = input("Введите ваш API ключ OpenRouter: ").strip()
        if key:
            try:
                with open(".env", "a", encoding="utf-8") as f:
                    f.write(f'\nOPENROUTER_API_KEY="{key}"\n')
                print("✅ Ключ сохранен в файл .env")
                return key
            except Exception as e:
                print(f"⚠️ Не удалось сохранить ключ: {e}")
                return key
        raise ValueError("API ключ OpenRouter не найден")

    @classmethod
    def get_telegram_token(cls) -> str:
        """Получение токена Telegram бота"""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token and token.strip():
            return token.strip()

        env_path = Path(".env")
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TELEGRAM_BOT_TOKEN="):
                            return line.split("=", 1)[1].strip(' "\'')
            except Exception as e:
                print(f"⚠️ Ошибка чтения .env: {e}")

        print("\n🤖 Токен Telegram бота не найден.")
        print("📌 Создайте бота через @BotFather и получите токен")
        token = input("Введите токен вашего Telegram бота: ").strip()
        if token:
            try:
                env_exists = env_path.exists()
                with open(env_path, "a" if env_exists else "w", encoding="utf-8") as f:
                    if env_exists:
                        f.write("\n")
                    f.write(f'TELEGRAM_BOT_TOKEN="{token}"\n')
                print("✅ Токен сохранен в файл .env")
                return token
            except Exception as e:
                print(f"⚠️ Не удалось сохранить токен: {e}")
                return token
        raise ValueError("Токен Telegram бота не найден")


# ================= УТИЛИТЫ =================
def calculate_text_similarity(text1: str, text2: str) -> float:
    """Расчёт схожести текстов с учётом n-грамм"""
    if not text1 or not text2:
        return 0.0

    def get_ngrams(text: str, n: int = 2) -> Set[str]:
        words = re.findall(r'\w+', text.lower())
        if len(words) < n:
            return set([' '.join(words)])
        return set(' '.join(words[i:i + n]) for i in range(len(words) - n + 1))

    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0

    unigram_sim = len(words1 & words2) / max(len(words1), len(words2))
    bigrams1 = get_ngrams(text1, 2)
    bigrams2 = get_ngrams(text2, 2)
    bigram_sim = len(bigrams1 & bigrams2) / max(len(bigrams1 | bigrams2), 1) if bigrams1 and bigrams2 else 0.0

    return 0.6 * unigram_sim + 0.4 * bigram_sim


def extract_semantic_features(text: str) -> Dict[str, Any]:
    """Извлечение семантических характеристик текста"""
    text_lower = text.lower()
    words = text.split()
    features = {
        'length': len(words),
        'complexity': len(set(text_lower.split())) / max(len(words), 1),
        'question_words': len(re.findall(r'\b(как|что|почему|зачем|когда|где|кто|сколько)\b', text_lower)),
        'numbers': len(re.findall(r'\b\d+\b', text)),
        'emotions': len(
            re.findall(r'\b(хорошо|плохо|отлично|ужасно|интересно|скучно|рад|грустно|восхищён)\b', text_lower)),
        'imperatives': len(
            re.findall(r'\b(сделай|создай|найди|покажи|расскажи|объясни|запомни|сохрани)\b', text_lower)),
        'has_question': '?' in text,
        'sentiment': analyze_sentiment(text)
    }
    return features


def analyze_sentiment(text: str) -> float:
    """Простой анализ тональности (-1 до 1)"""
    positive = ['хорошо', 'отлично', 'прекрасно', 'замечательно', 'классно', 'супер', 'рад', 'счастлив', 'восхищён',
                'люблю']
    negative = ['плохо', 'ужасно', 'отвратительно', 'кошмар', 'провал', 'грустно', 'ненавижу', 'злой', 'разочарован']
    text_lower = text.lower()
    pos_count = sum(1 for word in positive if word in text_lower)
    neg_count = sum(1 for word in negative if word in text_lower)
    total = pos_count + neg_count
    return (pos_count - neg_count) / total if total > 0 else 0.0


def split_message(text: str, max_length: int = Config.MAX_MESSAGE_LENGTH) -> list:
    """Разбивает длинное сообщение на части с сохранением структуры"""
    if not text:
        return [""]

    if len(text) <= max_length:
        return [text]

    # Попытка разбить по логическим разделителям
    parts = []
    current = ""

    # Разбиваем по абзацам
    paragraphs = re.split(r'(\n\s*\n)', text)

    for para in paragraphs:
        if len(current) + len(para) <= max_length:
            current += para
        else:
            if current:
                parts.append(current.rstrip())
            # Если параграф слишком длинный - разбиваем по предложениям
            if len(para) > max_length:
                sentences = re.split(r'([.!?]+)', para)
                temp = ""
                for i in range(0, len(sentences), 2):
                    chunk = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                    if len(temp) + len(chunk) <= max_length:
                        temp += chunk
                    else:
                        if temp:
                            parts.append(temp.rstrip())
                        temp = chunk
                if temp:
                    current = temp
            else:
                current = para

    if current:
        parts.append(current.rstrip())

    # Ограничиваем количество частей
    if len(parts) > Config.MAX_RESPONSE_CHUNKS:
        parts = parts[:Config.MAX_RESPONSE_CHUNKS]
        parts.append("📝 *...сообщение сокращено из-за ограничений Telegram*")

    return parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    """Создание главной клавиатуры с улучшенной структурой"""
    keyboard = [
        [
            InlineKeyboardButton("🧠 Глубокое мышление", callback_data="deep_think"),
            InlineKeyboardButton("🔍 Анализ", callback_data="analysis")
        ],
        [
            InlineKeyboardButton("📊 Статистика", callback_data="stats"),
            InlineKeyboardButton("🎯 Цели", callback_data="goals")
        ],
        [
            InlineKeyboardButton("💡 Инсайты", callback_data="insights"),
            InlineKeyboardButton("🔗 Паттерны", callback_data="patterns")
        ],
        [
            InlineKeyboardButton("🧹 Очистить контекст", callback_data="clear"),
            InlineKeyboardButton("❓ Помощь", callback_data="help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= РАСШИРЕННАЯ БАЗА ДАННЫХ С ПОДДЕРЖКОЙ ПОЛЬЗОВАТЕЛЕЙ =================
class EnhancedMemoryDB:
    """Продвинутая база данных с поддержкой изолированных пользовательских сессий"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()
        self._init_cache()

    def _init_cache(self):
        """Инициализация кэша для ускорения запросов"""
        self.fact_cache = {}
        self.pattern_cache = {}
        self.last_cache_clear = time.time()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Таблица взаимодействий с привязкой к пользователю
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user_id INTEGER NOT NULL,
                    user_input TEXT NOT NULL,
                    system_response TEXT NOT NULL,
                    context TEXT,
                    emotion TEXT DEFAULT 'neutral',
                    category TEXT,
                    importance REAL DEFAULT 0.5,
                    complexity REAL DEFAULT 0.5,
                    satisfaction REAL DEFAULT 0.5,
                    tokens_used INTEGER DEFAULT 0
                )
            ''')

            # Таблица фактов с привязкой к пользователю
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT,
                    confidence REAL DEFAULT 1.0,
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    last_used REAL,
                    usage_count INTEGER DEFAULT 0,
                    decay_factor REAL DEFAULT 1.0,
                    source TEXT,
                    UNIQUE(user_id, key, value)
                )
            ''')

            # Таблица связей между фактами
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fact_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    fact_id_1 INTEGER NOT NULL,
                    fact_id_2 INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (fact_id_1) REFERENCES facts(id),
                    FOREIGN KEY (fact_id_2) REFERENCES facts(id)
                )
            ''')

            # Таблица мыслей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS thoughts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    thought_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    trigger TEXT,
                    importance REAL DEFAULT 0.5,
                    depth_level INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.7,
                    outcome TEXT
                )
            ''')

            # Таблица целей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    parent_goal_id INTEGER,
                    created_at REAL NOT NULL,
                    description TEXT NOT NULL,
                    priority REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'active',
                    progress REAL DEFAULT 0.0,
                    deadline REAL,
                    next_action TEXT,
                    success_criteria TEXT,
                    FOREIGN KEY (parent_goal_id) REFERENCES goals(id)
                )
            ''')

            # Таблица паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    occurrences INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    success_rate REAL DEFAULT 0.5
                )
            ''')

            # Индексы для производительности
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_category ON interactions(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, importance DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoughts_user ON thoughts(user_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status, priority DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_user ON patterns(user_id, confidence DESC)')

            conn.commit()

    # === Взаимодействия ===
    def add_interaction(self, user_id: int, user_input: str, system_response: str, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions
                (timestamp, user_id, user_input, system_response, context, emotion, category,
                importance, complexity, satisfaction, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                user_id,
                user_input,
                system_response,
                kwargs.get('context', ''),
                kwargs.get('emotion', 'neutral'),
                kwargs.get('category', ''),
                kwargs.get('importance', 0.5),
                kwargs.get('complexity', 0.5),
                kwargs.get('satisfaction', 0.5),
                kwargs.get('tokens_used', 0)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_contextual_interactions(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM interactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit * 3))

            all_interactions = [dict(row) for row in cursor.fetchall()]
            if not all_interactions:
                return []

            # Ранжируем по релевантности
            scored = []
            for interaction in all_interactions:
                relevance = calculate_text_similarity(
                    query,
                    interaction['user_input'] + ' ' + interaction['system_response']
                )
                recency = 1.0 - (time.time() - interaction['timestamp']) / (7 * 24 * 3600)  # 7 дней
                recency = max(0, min(1, recency))
                score = 0.6 * relevance + 0.3 * interaction['importance'] + 0.1 * recency
                scored.append((score, interaction))

            scored.sort(reverse=True, key=lambda x: x[0])
            return [item[1] for item in scored[:limit]]

    # === Факты ===
    def add_fact(self, user_id: int, key: str, value: str, **kwargs):
        # Очистка кэша при добавлении нового факта
        cache_key = f"{user_id}_{key}_{value}"
        if cache_key in self.fact_cache:
            del self.fact_cache[cache_key]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM facts WHERE user_id = ? AND key = ? AND value = ?',
                           (user_id, key, value))
            existing = cursor.fetchone()

            if existing:
                cursor.execute('''
                    UPDATE facts
                    SET confidence = ?, importance = ?, last_used = ?,
                        usage_count = usage_count + 1, decay_factor = 1.0
                    WHERE id = ?
                ''', (
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(),
                    existing[0]
                ))
            else:
                cursor.execute('''
                    INSERT INTO facts
                    (user_id, key, value, category, confidence, importance, created_at, last_used,
                    usage_count, decay_factor, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, key, value,
                    kwargs.get('category', ''),
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(), time.time(), 1, 1.0,
                    kwargs.get('source', 'user')
                ))
            conn.commit()

    def get_relevant_facts(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        # Проверка кэша
        cache_key = f"{user_id}_{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.fact_cache and time.time() - self.last_cache_clear < 300:
            return self.fact_cache[cache_key]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Обновляем decay_factor для старых фактов
            cursor.execute('''
                UPDATE facts
                SET decay_factor = decay_factor * (1 - ?)
                WHERE user_id = ? AND last_used < ?
            ''', (Config.MEMORY_DECAY_RATE, user_id, time.time() - 86400))

            cursor.execute('''
                SELECT * FROM facts
                WHERE user_id = ? AND confidence > 0.3 AND decay_factor > 0.1
                ORDER BY importance DESC, usage_count DESC
                LIMIT ?
            ''', (user_id, limit * 2))

            all_facts = [dict(row) for row in cursor.fetchall()]
            if not all_facts:
                return []

            # Ранжируем по релевантности
            scored = []
            for fact in all_facts:
                relevance = calculate_text_similarity(query, f"{fact['key']} {fact['value']}")
                score = (
                        0.4 * relevance +
                        0.3 * fact['importance'] +
                        0.2 * fact['confidence'] +
                        0.1 * fact['decay_factor']
                )
                scored.append((score, fact))

            scored.sort(reverse=True, key=lambda x: x[0])
            result = [item[1] for item in scored[:limit]]

            # Кэшируем результат
            self.fact_cache[cache_key] = result
            return result

    # === Мысли ===
    def add_thought(self, user_id: int, thought_type: str, content: str, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO thoughts
                (user_id, timestamp, thought_type, content, trigger, importance, depth_level, confidence, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                time.time(),
                thought_type,
                content,
                kwargs.get('trigger', ''),
                kwargs.get('importance', 0.5),
                kwargs.get('depth_level', 1),
                kwargs.get('confidence', 0.7),
                kwargs.get('outcome', '')
            ))
            conn.commit()

    def get_thought_insights(self, user_id: int, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT thought_type, COUNT(*) as count, AVG(importance) as avg_importance
                FROM thoughts
                WHERE user_id = ? AND timestamp > ?
                GROUP BY thought_type
                ORDER BY count DESC
                LIMIT ?
            ''', (user_id, time.time() - 7 * 86400, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === Паттерны ===
    def add_pattern(self, user_id: int, pattern_type: str, description: str, confidence: float = 0.5):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM patterns 
                WHERE user_id = ? AND pattern_type = ? AND description = ?
            ''', (user_id, pattern_type, description))
            existing = cursor.fetchone()

            if existing:
                cursor.execute('''
                    UPDATE patterns
                    SET occurrences = occurrences + 1, last_seen = ?, confidence = ?
                    WHERE id = ?
                ''', (time.time(), min(1.0, confidence * 1.1), existing[0]))
            else:
                cursor.execute('''
                    INSERT INTO patterns
                    (user_id, pattern_type, description, occurrences, confidence, created_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, pattern_type, description, 1, confidence, time.time(), time.time()))
            conn.commit()

    def get_patterns(self, user_id: int, min_confidence: float = 0.6, limit: int = 10) -> List[Dict]:
        # Проверка кэша
        cache_key = f"{user_id}_patterns_{min_confidence}"
        if cache_key in self.pattern_cache and time.time() - self.last_cache_clear < 600:
            return self.pattern_cache[cache_key]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM patterns
                WHERE user_id = ? AND confidence >= ?
                ORDER BY occurrences DESC, confidence DESC
                LIMIT ?
            ''', (user_id, min_confidence, limit))
            result = [dict(row) for row in cursor.fetchall()]

            # Кэшируем результат
            self.pattern_cache[cache_key] = result
            return result

    # === Цели ===
    def add_goal(self, user_id: int, description: str, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO goals
                (user_id, parent_goal_id, created_at, description, priority, status, progress,
                deadline, next_action, success_criteria)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                kwargs.get('parent_goal_id'),
                time.time(),
                description,
                kwargs.get('priority', 0.5),
                kwargs.get('status', 'active'),
                kwargs.get('progress', 0.0),
                kwargs.get('deadline'),
                kwargs.get('next_action', ''),
                kwargs.get('success_criteria', '')
            ))
            conn.commit()
            return cursor.lastrowid

    def get_goal_hierarchy(self, user_id: int, parent_id: Optional[int] = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if parent_id is None:
                cursor.execute('''
                    SELECT * FROM goals
                    WHERE user_id = ? AND parent_goal_id IS NULL AND status = 'active'
                    ORDER BY priority DESC
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT * FROM goals
                    WHERE user_id = ? AND parent_goal_id = ? AND status = 'active'
                    ORDER BY priority DESC
                ''', (user_id, parent_id))
            return [dict(row) for row in cursor.fetchall()]

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Получение статистики по пользователю"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Взаимодействия
            cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (user_id,))
            interactions = cursor.fetchone()[0]

            # Факты
            cursor.execute('SELECT COUNT(*) FROM facts WHERE user_id = ? AND decay_factor > 0.3', (user_id,))
            facts = cursor.fetchone()[0]

            # Мысли
            cursor.execute('SELECT COUNT(*) FROM thoughts WHERE user_id = ?', (user_id,))
            thoughts = cursor.fetchone()[0]

            # Паттерны
            cursor.execute('SELECT COUNT(*) FROM patterns WHERE user_id = ? AND confidence > 0.5', (user_id,))
            patterns = cursor.fetchone()[0]

            # Цели
            cursor.execute('SELECT COUNT(*) FROM goals WHERE user_id = ? AND status = \'active\'', (user_id,))
            goals = cursor.fetchone()[0]

            # Средняя удовлетворённость
            cursor.execute('''
                SELECT AVG(satisfaction) FROM interactions 
                WHERE user_id = ? AND timestamp > ?
            ''', (user_id, time.time() - 86400))
            avg_satisfaction = cursor.fetchone()[0] or 0.5

            return {
                'interactions': interactions,
                'facts': facts,
                'thoughts': thoughts,
                'patterns': patterns,
                'goals': goals,
                'avg_satisfaction': avg_satisfaction,
                'first_interaction': self._get_first_interaction_time(user_id)
            }

    def _get_first_interaction_time(self, user_id: int) -> float:
        """Получение времени первого взаимодействия"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp FROM interactions 
                WHERE user_id = ? ORDER BY timestamp ASC LIMIT 1
            ''', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else time.time()


# ================= СИСТЕМА МЫШЛЕНИЯ =================
class EnhancedThinkingSystem:
    """Продвинутая многоуровневая система мышления"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit = 1.5
        self.last_request_time = 0
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.reasoning_history = deque(maxlen=100)
        self.thinking_performance = {
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_depth': 1.0,
            'creativity_score': 0.5
        }

    async def _wait_for_rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def multi_level_thinking(self, context: str, depth: int = 2) -> Dict[str, str]:
        """Многоуровневое мышление с разными типами анализа"""
        thoughts = {}
        thinking_layers = [
            ('surface', 'Что очевидно в этом запросе?', 0.3),
            ('analytical', 'Какие скрытые связи и паттерны можно обнаружить?', 0.5),
            ('strategic', 'Каковы долгосрочные последствия и возможности?', 0.7),
            ('creative', 'Какие нестандартные, креативные решения возможны?', 0.9)
        ]

        for layer_name, prompt, temperature in thinking_layers[:depth]:
            thought = await self.generate_thought_with_prompt(
                f"{prompt}\n\nКонтекст: {context}",
                temperature=temperature
            )
            if thought:
                thoughts[layer_name] = thought

        return thoughts

    async def generate_thought_with_prompt(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Генерация мысли с заданным промптом"""
        system_prompt = """Ты — продвинутая когнитивная система с глубокими аналитическими способностями.
Твои сильные стороны:
- Многоуровневое мышление (поверхностное → аналитическое → стратегическое → креативное)
- Обнаружение неявных паттернов и связей
- Креативное решение сложных проблем
- Критический анализ предпосылок
- Предсказание вероятных последствий

Отвечай кратко, но содержательно. Фокусируйся на нетривиальных инсайтах, а не на очевидных вещах."""

        response = await self.call_llm(system_prompt, prompt, temperature)
        if response and len(response) > 15:
            self.reasoning_history.append({
                'timestamp': time.time(),
                'prompt': prompt[:100],
                'response': response[:200],
                'temperature': temperature
            })
            return response
        return None

    async def predict_outcome(self, action: str, context: str) -> Dict[str, Any]:
        """Предсказание результатов действия"""
        prompt = f"""Предскажи возможные результаты следующего действия:
Действие: {action}
Контекст: {context}

Оцени объективно:
1. Вероятность успеха (0.0-1.0)
2. Основные риски и их вероятность
3. Ожидаемые выгоды и их значимость
4. 2-3 альтернативных подхода с кратким сравнением

Формат ответа: чистый JSON без дополнительного текста."""

        system_prompt = "Ты эксперт по прогнозированию и анализу рисков. Будь точным, объективным и практичным."
        response = await self.call_llm(system_prompt, prompt, temperature=0.4)

        # Попытка распарсить JSON
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logging.warning(f"Ошибка парсинга JSON предсказания: {e}")

        return {'raw_prediction': response}

    async def synthesize_knowledge(self, facts: List[Dict], question: str) -> str:
        """Синтез знаний из нескольких фактов"""
        facts_text = "\n".join([f"- {f['key']}: {f['value']}" for f in facts[:5]])
        prompt = f"""Синтезируй исчерпывающий ответ на вопрос, используя доступные факты:

Вопрос: {question}

Доступные факты:
{facts_text}

Требования к ответу:
- Создай связный, логичный ответ
- Укажи, какие факты использованы
- Если фактов недостаточно для полного ответа — честно укажи это и предложи, какую информацию нужно дополнить
- Избегай домыслов и предположений без оснований"""

        system_prompt = "Ты эксперт по синтезу информации. Находи глубокие связи между фактами и создавай целостную картину."
        return await self.call_llm(system_prompt, prompt, temperature=0.6)

    async def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Вызов LLM с улучшенным кэшированием и обработкой ошибок"""
        # Проверка кэша
        cache_key = hashlib.md5(
            f"{system_prompt[:100]}|{user_prompt[:200]}|{temperature}".encode()
        ).hexdigest()

        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1

        await self._wait_for_rate_limit()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AGI24/cognitive-agent",  # Для OpenRouter
                "X-Title": "AGI Telegram Bot"
            }

            payload = {
                "model": Config.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": Config.MAX_TOKENS,
                "top_p": 0.95
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        Config.OPENROUTER_URL,
                        headers=headers,
                        json=payload,
                        timeout=Config.TIMEOUT
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()

                        # Сохраняем в кэш
                        self.cache[cache_key] = content

                        # Ограничиваем размер кэша
                        if len(self.cache) > 300:
                            # Удаляем 20% самых старых записей
                            keys_to_remove = list(self.cache.keys())[:60]
                            for key in keys_to_remove:
                                self.cache.pop(key, None)

                        return content
                    else:
                        error_text = await response.text()
                        return f"⚠️ Ошибка API ({response.status}): {error_text[:200]}"

        except asyncio.TimeoutError:
            return "⚠️ Таймаут запроса к нейросети. Попробуйте позже."
        except Exception as e:
            return f"⚠️ Ошибка нейросети: {str(e)[:150]}"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности мышления"""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0

        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'reasoning_history_size': len(self.reasoning_history),
            'thinking_performance': self.thinking_performance.copy()
        }


# ================= ПОЛЬЗОВАТЕЛЬСКИЙ КОГНИТИВНЫЙ АГЕНТ =================
class UserCognitiveAgent:
    """Изолированный когнитивный агент для одного пользователя Telegram"""

    def __init__(self, user_id: int, db: EnhancedMemoryDB, thinker: EnhancedThinkingSystem):
        self.user_id = user_id
        self.db = db
        self.thinker = thinker
        self.interaction_count = 0
        self.deep_thoughts_count = 0
        self.patterns_found = 0
        self.start_time = time.time()
        self.context_window = deque(maxlen=Config.CONTEXT_WINDOW_SIZE)
        self.self_assessment = {
            'knowledge_gaps': [],
            'strong_areas': [],
            'improvement_areas': []
        }

        # Инициализация базовых целей при первом запуске
        self._init_user_goals()

    def _init_user_goals(self):
        """Инициализация базовых целей для пользователя"""
        existing_goals = self.db.get_goal_hierarchy(self.user_id)
        if not existing_goals:
            # Главная цель
            main_goal = self.db.add_goal(
                self.user_id,
                "Быть максимально полезным персональным когнитивным помощником",
                priority=1.0,
                success_criteria="Высокий уровень удовлетворённости пользователя в диалогах"
            )

            # Подцели
            self.db.add_goal(
                self.user_id,
                "Глубоко понимать запросы и контекст пользователя",
                parent_goal_id=main_goal,
                priority=0.95
            )
            self.db.add_goal(
                self.user_id,
                "Непрерывно обучаться на основе диалогов с пользователем",
                parent_goal_id=main_goal,
                priority=0.9
            )
            self.db.add_goal(
                self.user_id,
                "Находить неочевидные решения и инсайты",
                parent_goal_id=main_goal,
                priority=0.85
            )
            self.db.add_goal(
                self.user_id,
                "Помогать в решении сложных задач через многоуровневый анализ",
                parent_goal_id=main_goal,
                priority=0.8
            )

    async def process_message(self, user_input: str) -> str:
        """Обработка сообщения пользователя с полным когнитивным циклом"""
        start_time = time.time()
        self.interaction_count += 1

        # Добавляем в контекстное окно
        self.context_window.append({
            'type': 'user',
            'content': user_input,
            'timestamp': time.time()
        })

        # Обработка команд
        command_response = self._handle_command(user_input)
        if command_response:
            return command_response

        # Анализируем входной запрос
        features = extract_semantic_features(user_input)

        # Определяем сложность и важность
        complexity = features['complexity']
        importance = self._calculate_importance(user_input, features)

        # Извлекаем и сохраняем информацию
        await self._extract_and_store_information(user_input, importance)

        # Генерируем ответ с учётом контекста
        response = await self._generate_contextual_response(
            user_input,
            features,
            complexity,
            importance
        )

        # Сохраняем взаимодействие
        self.db.add_interaction(
            user_id=self.user_id,
            user_input=user_input,
            system_response=response,
            context=self._get_context_summary(),
            category=self._categorize_input(user_input, features),
            importance=importance,
            complexity=complexity,
            tokens_used=len(response.split())
        )

        # Добавляем ответ в контекстное окно
        self.context_window.append({
            'type': 'assistant',
            'content': response,
            'timestamp': time.time()
        })

        # Обнаружение паттернов (асинхронно, чтобы не блокировать ответ)
        asyncio.create_task(self._detect_patterns())

        # Периодическое глубокое мышление
        if self.interaction_count % Config.REFLECTION_INTERVAL == 0:
            asyncio.create_task(self._deep_autonomous_thinking())

        # Логирование производительности
        duration = time.time() - start_time
        if duration > 3.0:
            logging.info(f"Пользователь {self.user_id}: обработка заняла {duration:.2f}с")

        return response

    def _calculate_importance(self, text: str, features: Dict) -> float:
        """Расчёт важности запроса"""
        importance = 0.5

        # Наличие ключевых слов
        if any(word in text.lower() for word in ['важно', 'срочно', 'критично', 'обязательно', 'немедленно']):
            importance += 0.3

        # Вопросительные слова
        importance += min(0.25, features['question_words'] * 0.12)

        # Императивы
        importance += min(0.2, features['imperatives'] * 0.1)

        # Сложность запроса
        importance += features['complexity'] * 0.25

        # Длина сообщения (длинные запросы часто важнее)
        if features['length'] > 20:
            importance += 0.1

        return min(1.0, importance)

    async def _extract_and_store_information(self, text: str, importance: float):
        """Извлечение и сохранение информации с метриками"""
        # Извлекаем числа
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers[:5]:  # Ограничиваем количество
            self.db.add_fact(
                self.user_id,
                'число',
                num,
                category='данные',
                importance=importance * 0.4,
                source='user_input'
            )

        # Извлекаем имена собственные (более надёжное извлечение)
        names = re.findall(r'\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*\b', text)
        for name in names[:3]:
            if len(name) > 2 and not any(skip in name.lower() for skip in ['это', 'как', 'что']):
                self.db.add_fact(
                    self.user_id,
                    'имя',
                    name,
                    category='персона',
                    importance=importance * 0.7,
                    source='user_input'
                )

        # Ключевые утверждения и определения
        patterns = [
            (r'(\w+)\s+(?:это|является|называется)\s+([^.,!?]+)', 'определение'),
            (r'запомни[,:]?\s*(.+)', 'важная_информация'),
            (r'(\w+)\s*=\s*([^,]+)', 'равенство'),
            (r'я люблю (\w+)', 'предпочтение'),
            (r'мой любимый (\w+)\s+(?:это|—)\s+([^.,!?]+)', 'предпочтение')
        ]

        for pattern, category in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
            for match in matches[:2]:  # Ограничиваем
                if isinstance(match, tuple) and len(match) >= 2:
                    key, value = match[0], match[1]
                    self.db.add_fact(
                        self.user_id,
                        key.strip().lower(),
                        value.strip(),
                        category=category,
                        importance=min(1.0, importance * 1.2),
                        source='user_input'
                    )
                elif isinstance(match, str):
                    self.db.add_fact(
                        self.user_id,
                        'память',
                        match.strip(),
                        category=category,
                        importance=min(1.0, importance * 1.2),
                        source='user_input'
                    )

    async def _generate_contextual_response(
            self,
            user_input: str,
            features: Dict,
            complexity: float,
            importance: float
    ) -> str:
        """Генерация ответа с глубоким контекстом и адаптивным мышлением"""
        # Получаем релевантный контекст
        relevant_interactions = self.db.get_contextual_interactions(self.user_id, user_input, limit=4)
        relevant_facts = self.db.get_relevant_facts(self.user_id, user_input, limit=6)
        active_goals = self.db.get_goal_hierarchy(self.user_id)
        patterns = self.db.get_patterns(self.user_id, min_confidence=0.65, limit=3)

        # Формируем контекст для генерации
        context_parts = []

        # История диалога
        if relevant_interactions:
            context_parts.append("## КОНТЕКСТ ДИАЛОГА:")
            for interaction in relevant_interactions[:3]:
                context_parts.append(f"Пользователь: {interaction['user_input'][:70]}...")
                context_parts.append(f"Ассистент: {interaction['system_response'][:70]}...")
                context_parts.append("")

        # Факты
        if relevant_facts:
            context_parts.append("## РЕЛЕВАНТНЫЕ ФАКТЫ:")
            for fact in relevant_facts[:5]:
                conf_stars = "★" * int(fact['confidence'] * 5)
                context_parts.append(f"- {fact['key']}: {fact['value']} [{conf_stars}]")
            context_parts.append("")

        # Цели
        if active_goals:
            context_parts.append("## АКТИВНЫЕ ЦЕЛИ ПОЛЬЗОВАТЕЛЯ:")
            for goal in active_goals[:3]:
                progress = int(goal['progress'] * 100)
                context_parts.append(f"- {goal['description'][:60]} (прогресс: {progress}%)")
            context_parts.append("")

        # Паттерны
        if patterns:
            context_parts.append("## ОБНАРУЖЕННЫЕ ПАТТЕРНЫ:")
            for pattern in patterns[:2]:
                context_parts.append(f"- {pattern['description'][:70]}")
            context_parts.append("")

        context = "\n".join(context_parts) if context_parts else "Нет дополнительного контекста"

        # Определяем нужен ли глубокий анализ
        needs_deep_thinking = (
                complexity > Config.DEEP_THINKING_THRESHOLD or
                importance > 0.75 or
                features['question_words'] > 2 or
                'анализ' in user_input.lower() or
                'почему' in user_input.lower() or
                'как сделать' in user_input.lower()
        )

        if needs_deep_thinking:
            # Многоуровневое мышление
            deep_thoughts = await self.thinker.multi_level_thinking(
                f"Запрос пользователя: {user_input}\n\n{context}",
                depth=3
            )

            # Синтез ответа на основе многоуровневого анализа
            synthesis_prompt = f"""На основе многоуровневого когнитивного анализа ответь на запрос пользователя.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{user_input}

РЕЗУЛЬТАТЫ АНАЛИЗА:
{chr(10).join([f'{level.upper()}: {thought}' for level, thought in deep_thoughts.items()])}

ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:
{context}

ИНСТРУКЦИИ:
- Дай исчерпывающий, но лаконичный ответ
- Включи ключевые инсайты из анализа
- Упомяни релевантные факты из памяти, если они есть
- Если вопрос сложный — предложи пошаговое решение
- Будь полезным, точным и креативным при необходимости"""

            system_prompt = """Ты — продвинутый когнитивный ассистент с многоуровневым мышлением.
Синтезируй ответ, интегрируя все уровни анализа. Давай практические, полезные ответы с глубокими инсайтами."""

            response = await self.thinker.call_llm(
                system_prompt,
                synthesis_prompt,
                temperature=0.65
            )
            self.deep_thoughts_count += 1
        else:
            # Стандартный ответ с контекстом
            system_prompt = f"""Ты — персональный когнитивный ассистент пользователя.
Используй контекст для персонализации ответа.

КОНТЕКСТ:
{context}

ПРИНЦИПЫ ОТВЕТА:
- Будь полезным, точным и дружелюбным
- Используй сохранённые факты для персонализации
- Учитывай цели пользователя
- Давай конкретные, практичные ответы
- При необходимости задавай уточняющие вопросы"""

            response = await self.thinker.call_llm(
                system_prompt,
                user_input,
                temperature=0.55
            )

        return response.strip()

    async def _detect_patterns(self):
        """Обнаружение паттернов в поведении пользователя"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Частые категории запросов
            cursor.execute('''
                SELECT category, COUNT(*) as count
                FROM interactions
                WHERE user_id = ? AND timestamp > ? AND category IS NOT NULL AND category != ''
                GROUP BY category
                HAVING count > 2
                ORDER BY count DESC
                LIMIT 5
            ''', (self.user_id, time.time() - 14 * 86400))

            categories = cursor.fetchall()
            for category, count in categories:
                if category and len(category) < 50:
                    confidence = min(0.95, 0.4 + count * 0.1)
                    self.db.add_pattern(
                        self.user_id,
                        'frequent_topic',
                        f"Пользователь часто интересуется темой: {category}",
                        confidence=confidence
                    )

            # Временные паттерны активности
            cursor.execute('''
                SELECT 
                    CAST(strftime('%H', datetime(timestamp, 'unixepoch')) AS INTEGER) as hour,
                    COUNT(*) as count
                FROM interactions
                WHERE user_id = ? AND timestamp > ?
                GROUP BY hour
                HAVING count > 3
                ORDER BY count DESC
                LIMIT 3
            ''', (self.user_id, time.time() - 21 * 86400))

            time_patterns = cursor.fetchall()
            for hour, count in time_patterns:
                period = "утром" if 6 <= hour < 12 else "днём" if 12 <= hour < 18 else "вечером" if 18 <= hour < 23 else "ночью"
                self.db.add_pattern(
                    self.user_id,
                    'time_preference',
                    f"Пользователь наиболее активен {period} (около {hour}:00)",
                    confidence=0.6 + min(0.3, count * 0.05)
                )

        self.patterns_found = len(categories) + len(time_patterns)

    async def _deep_autonomous_thinking(self):
        """Глубокое автономное мышление для улучшения понимания пользователя"""
        # Собираем данные для рефлексии
        recent = self.db.get_contextual_interactions(self.user_id, "последние запросы", limit=8)
        patterns = self.db.get_patterns(self.user_id, min_confidence=0.6, limit=5)
        insights = self.db.get_thought_insights(self.user_id, limit=8)

        if len(recent) < 3:
            return  # Недостаточно данных для глубокого анализа

        # Формируем контекст для размышлений
        context_lines = [
            f"Глубокая рефлексия для пользователя ID: {self.user_id}",
            f"Последние {min(5, len(recent))} взаимодействий:",
            *[f"{i + 1}. П: {r['user_input'][:60]}..." for i, r in enumerate(recent[:5])],
            "",
            f"Обнаружено паттернов ({len(patterns)}):",
            *[f"- {p['description'][:70]}" for p in patterns[:4]],
            "",
            f"Инсайты из мыслей ({len(insights)} типов):",
            *[f"- {t['thought_type']}: {t['count']} раз (важность: {t['avg_importance']:.2f})"
              for t in insights[:4]]
        ]

        context = "\n".join(context_lines)

        # Многоуровневое мышление для самосовершенствования
        thoughts = await self.thinker.multi_level_thinking(context, depth=4)

        # Сохраняем наиболее ценные мысли
        for thought_type, content in thoughts.items():
            if content and len(content) > 30:
                self.db.add_thought(
                    user_id=self.user_id,
                    thought_type=thought_type,
                    content=content,
                    trigger='autonomous_reflection',
                    importance=0.85,
                    depth_level=4,
                    confidence=0.8
                )

        # Обновляем самооценку
        await self._update_self_assessment()

    async def _update_self_assessment(self):
        """Обновление самооценки системы на основе взаимодействий"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Средняя удовлетворённость за последнюю неделю
            cursor.execute('''
                SELECT AVG(satisfaction) FROM interactions
                WHERE user_id = ? AND timestamp > ?
            ''', (self.user_id, time.time() - 7 * 86400))
            avg_satisfaction = cursor.fetchone()[0] or 0.5

            # Категории с низкой удовлетворённостью
            cursor.execute('''
                SELECT category, AVG(satisfaction) as avg_sat, COUNT(*) as count
                FROM interactions
                WHERE user_id = ? AND timestamp > ? AND category IS NOT NULL AND category != ''
                GROUP BY category
                HAVING avg_sat < 0.6 AND count >= 3
                ORDER BY avg_sat ASC
                LIMIT 3
            ''', (self.user_id, time.time() - 14 * 86400))
            weak_categories = [row[0] for row in cursor.fetchall()]

            # Категории с высокой удовлетворённостью
            cursor.execute('''
                SELECT category, AVG(satisfaction) as avg_sat, COUNT(*) as count
                FROM interactions
                WHERE user_id = ? AND timestamp > ? AND category IS NOT NULL AND category != ''
                GROUP BY category
                HAVING avg_sat > 0.8 AND count >= 3
                ORDER BY avg_sat DESC
                LIMIT 3
            ''', (self.user_id, time.time() - 14 * 86400))
            strong_categories = [row[0] for row in cursor.fetchall()]

            # Обновляем самооценку
            self.self_assessment = {
                'avg_satisfaction': avg_satisfaction,
                'improvement_areas': weak_categories,
                'strong_areas': strong_categories,
                'patterns_discovered': self.patterns_found,
                'deep_thoughts': self.deep_thoughts_count,
                'interaction_count': self.interaction_count
            }

    def _handle_command(self, text: str) -> Optional[str]:
        """Обработка внутренних команд пользователя"""
        text_lower = text.lower().strip()

        if text_lower in ['думай глубоко', 'глубокое мышление', '/think']:
            asyncio.create_task(self._deep_autonomous_thinking())
            return "🧠 Запускаю глубокое многоуровневое мышление...\nЭто займёт 10-15 секунд. Результаты будут доступны в /insights"

        elif text_lower in ['/clear', 'очистить контекст', 'забудь']:
            self.context_window.clear()
            return "🧹 Контекст диалога очищен. Я больше не помню предыдущие сообщения в этом разговоре."

        return None

    def _get_context_summary(self) -> str:
        """Краткое резюме текущего контекста"""
        if not self.context_window:
            return ""

        summary = []
        for item in list(self.context_window)[-6:]:
            prefix = "П:" if item['type'] == 'user' else "Я:"
            summary.append(f"{prefix} {item['content'][:40]}...")

        return " | ".join(summary)

    def _categorize_input(self, text: str, features: Dict) -> str:
        """Улучшенная категоризация запроса"""
        text_lower = text.lower()

        # Правила категоризации
        categories = {
            'математика': ['сколько', 'посчитай', 'вычисли', 'математик', '+', '-', '*', '/', '='],
            'память': ['запомни', 'сохрани', 'напомни', 'записывай', 'запоминай'],
            'анализ': ['проанализируй', 'разбери', 'оцени', 'сравни', 'анализ'],
            'творчество': ['придумай', 'создай', 'сочини', 'напиши', 'идея', 'креатив'],
            'планирование': ['план', 'распиши', 'как достичь', 'стратегия', 'расписание'],
            'объяснение': ['почему', 'как работает', 'зачем', 'объясни', 'расскажи про', 'что такое'],
            'поиск': ['найди', 'покажи', 'где', 'ищи', 'источник'],
            'совет': ['совет', 'посоветуй', 'как лучше', 'что делать'],
            'эмоции': ['чувству', 'эмоции', 'настроение', 'грустно', 'радостно']
        }

        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return 'диалог'

    def get_comprehensive_stats(self) -> str:
        """Получение комплексной статистики по пользователю"""
        stats = self.db.get_user_stats(self.user_id)

        # Время работы с пользователем
        uptime = time.time() - stats['first_interaction']
        days = int(uptime // 86400)
        hours = int((uptime % 86400) // 3600)

        lines = [
            "📊 ПЕРСОНАЛЬНАЯ СТАТИСТИКА",
            "=" * 45,
            f"⏱️  Время знакомства: {days}д {hours}ч",
            f"💬 Всего сообщений: {stats['interactions']}",
            f"🧠 Глубоких мыслей: {self.deep_thoughts_count}",
            f"🔗 Обнаружено паттернов: {stats['patterns']}",
            f"📚 Сохранено фактов: {stats['facts']}",
            f"🎯 Активных целей: {stats['goals']}",
            f"💡 Мыслей и инсайтов: {stats['thoughts']}",
            f"😊 Средняя удовлетворённость: {stats['avg_satisfaction']:.2f}",
            "",
            "📈 ПРОГРЕСС ОБУЧЕНИЯ:",
            f"   • Адаптация к стилю общения: {'✅' if stats['interactions'] > 10 else '🔄'}",
            f"   • Персонализация ответов: {'✅' if stats['facts'] > 5 else '🔄'}",
            f"   • Обнаружение паттернов: {'✅' if stats['patterns'] > 3 else '🔄'}"
        ]

        # Добавляем информацию о сильных сторонах
        if self.self_assessment.get('strong_areas'):
            lines.append("\n✅ ВАШИ ПРЕДПОЧТЕНИЯ:")
            for area in self.self_assessment['strong_areas'][:3]:
                lines.append(f"   • {area}")

        return "\n".join(lines)

    def get_patterns_summary(self) -> str:
        """Получение сводки по обнаруженным паттернам"""
        patterns = self.db.get_patterns(self.user_id, min_confidence=0.55, limit=12)
        if not patterns:
            return "🔍 Паттерны пока не обнаружены.\nЧем больше мы общаемся, тем лучше я вас узнаю!"

        lines = ["🔍 ОБНАРУЖЕННЫЕ ПАТТЕРНЫ", "=" * 45]

        # Группируем по типу
        by_type = defaultdict(list)
        for p in patterns:
            by_type[p['pattern_type']].append(p)

        for ptype, plist in list(by_type.items())[:4]:
            lines.append(f"\n📌 {ptype.replace('_', ' ').upper()}:")
            for p in plist[:3]:
                conf_pct = int(p['confidence'] * 100)
                conf_bar = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
                lines.append(f"  {conf_bar} {conf_pct}%")
                lines.append(f"  {p['description'][:75]}")
                if p['occurrences'] > 1:
                    lines.append(f"  (встречалось {p['occurrences']} раз)")

        return "\n".join(lines)

    def get_insights_summary(self) -> str:
        """Получение сводки по инсайтам из мыслей"""
        insights = self.db.get_thought_insights(self.user_id, limit=8)
        if not insights:
            return "💡 Инсайты пока не накоплены.\nАктивируйте глубокое мышление командой /think"

        lines = ["💡 ИНСАЙТЫ ИЗ МОИХ РАЗМЫШЛЕНИЙ", "=" * 45]

        # Сортируем по важности и количеству
        insights.sort(key=lambda x: (x['avg_importance'], x['count']), reverse=True)

        for insight in insights[:6]:
            importance = insight['avg_importance']
            imp_bar = "★" * int(importance * 5)
            lines.append(f"\n{imp_bar} {insight['thought_type'].upper()}")
            lines.append(f"   Встречалось: {insight['count']} раз")
            lines.append(f"   Средняя значимость: {importance:.2f}")

        return "\n".join(lines)

    def get_goals_summary(self) -> str:
        """Получение сводки по целям"""
        main_goals = self.db.get_goal_hierarchy(self.user_id, parent_id=None)
        if not main_goals:
            return "🎯 Цели ещё не определены.\nЯ автоматически создаю цели на основе наших диалогов."

        lines = ["🎯 ВАША ИЕРАРХИЯ ЦЕЛЕЙ", "=" * 45]

        for goal in main_goals[:3]:
            progress = int(goal['progress'] * 100)
            prog_bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
            lines.append(f"\n📍 {goal['description'][:60]}")
            lines.append(f"   Прогресс: [{prog_bar}] {progress}%")
            lines.append(f"   Приоритет: {goal['priority']:.2f}")

            # Подцели
            subgoals = self.db.get_goal_hierarchy(self.user_id, parent_id=goal['id'])
            if subgoals:
                lines.append(f"   Подцели ({len(subgoals)}):")
                for sub in subgoals[:3]:
                    sub_prog = int(sub['progress'] * 100)
                    lines.append(f"     • {sub['description'][:50]} ({sub_prog}%)")

        return "\n".join(lines)


# ================= МЕНЕДЖЕР ПОЛЬЗОВАТЕЛЬСКИХ СЕССИЙ =================
class SessionManager:
    """Управление изолированными сессиями пользователей"""

    def __init__(self, db: EnhancedMemoryDB, thinker: EnhancedThinkingSystem):
        self.db = db
        self.thinker = thinker
        self.sessions: Dict[int, UserCognitiveAgent] = {}
        self.session_timeout = Config.SESSION_TIMEOUT
        self.last_cleanup = time.time()

    async def get_or_create_session(self, user_id: int) -> UserCognitiveAgent:
        """Получение или создание сессии пользователя"""
        # Очистка старых сессий каждые 5 минут
        if time.time() - self.last_cleanup > 300:
            await self._cleanup_inactive_sessions()
            self.last_cleanup = time.time()

        # Возвращаем существующую сессию или создаём новую
        if user_id not in self.sessions:
            logging.info(f"🆕 Создание новой когнитивной сессии для пользователя {user_id}")
            self.sessions[user_id] = UserCognitiveAgent(user_id, self.db, self.thinker)

        return self.sessions[user_id]

    async def _cleanup_inactive_sessions(self):
        """Очистка неактивных сессий для экономии памяти"""
        current_time = time.time()
        inactive_users = [
            user_id for user_id, session in self.sessions.items()
            if current_time - session.start_time > self.session_timeout * 2
        ]

        for user_id in inactive_users:
            del self.sessions[user_id]
            logging.info(f"🧹 Очищена неактивная сессия пользователя {user_id}")

    def get_global_stats(self) -> Dict[str, Any]:
        """Получение глобальной статистики по всем пользователям"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM interactions")
            total_users = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM facts WHERE decay_factor > 0.3")
            total_facts = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM patterns WHERE confidence > 0.5")
            total_patterns = cursor.fetchone()[0] or 0

            # Активные пользователи за последнюю неделю
            cursor.execute('''
                SELECT COUNT(DISTINCT user_id) FROM interactions
                WHERE timestamp > ?
            ''', (time.time() - 7 * 86400,))
            active_users = cursor.fetchone()[0] or 0

        return {
            'total_users': total_users,
            'active_users': active_users,
            'total_interactions': total_interactions,
            'total_facts': total_facts,
            'total_patterns': total_patterns,
            'active_sessions': len(self.sessions)
        }


# ================= ОБРАБОТЧИКИ TELEGRAM =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    user_id = user.id

    # Получаем или создаём сессию пользователя
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    welcome_text = f"""👋 Привет, {user.first_name}!

🧠 Я — **Когнитивный Ассистент AGI** — ваш персональный ИИ с продвинутыми аналитическими способностями:

✨ **Мои возможности:**
• 🤯 Многоуровневый анализ (от поверхностного до стратегического)
• 🧠 Контекстная память и персонализация ответов
• 🔍 Автоматическое обнаружение паттернов в ваших запросах
• 💡 Генерация креативных решений и неочевидных инсайтов
• 📊 Предиктивный анализ и планирование

💬 **Просто напишите мне запрос** — я помогу с анализом, творчеством, обучением или решением задач!

📌 **Полезные команды:**
• /think — активировать глубокое мышление
• /stats — ваша персональная статистика
• /patterns — обнаруженные паттерны вашего поведения
• /insights — инсайты из моих размышлений о вас
• /goals — ваши цели и прогресс
• /clear — очистить контекст текущего диалога

🚀 **Примеры запросов:**
• "Запомни, что я люблю кофе с молоком"
• "Проанализируй мои последние запросы и найди паттерны"
• "Придумай креативное название для моего проекта"
• "Объясни квантовую запутанность простыми словами"
• "Помоги спланировать обучение новому навыку"

💡 Я запоминаю контекст наших диалогов и непрерывно учусь на ваших запросах!"""

    try:
        await update.message.reply_text(
            welcome_text,
            reply_markup=create_main_keyboard(),
            parse_mode='MarkdownV2',
            disable_web_page_preview=True
        )
    except BadRequest:
        # Fallback если Markdown вызывает ошибки
        clean_text = re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', welcome_text)
        await update.message.reply_text(
            clean_text,
            reply_markup=create_main_keyboard(),
            parse_mode='MarkdownV2'
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """📖 **ПОЛНЫЙ СПРАВОЧНИК**

**🎯 ОСНОВНЫЕ КОМАНДЫ:**
/start — начало работы и приветствие
/help — этот справочник
/stats — ваша персональная статистика
/clear — очистить контекст текущего диалога

**🧠 КОГНИТИВНЫЕ ФУНКЦИИ:**
/think — активировать глубокое многоуровневое мышление
/patterns — показать обнаруженные паттерны вашего поведения
/insights — инсайты из моих размышлений о вас
/goals — ваши цели и прогресс в их достижении

**💡 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:**
• *Простые вопросы:* "Сколько будет 17 * 24?"
• *Анализ:* "Проанализируй мои интересы на основе наших диалогов"
• *Память:* "Запомни, что завтра у меня встреча в 15:00"
• *Творчество:* "Придумай 5 нестандартных идей для стартапа"
• *Планирование:* "Помоги составить план изучения Python за месяц"
• *Обучение:* "Объясни теорию относительности так, чтобы понял ребёнок"

**⚙️ ОСОБЕННОСТИ РАБОТЫ:**
• Я запоминаю контекст нашего диалога в течение сессии
• Автоматически обнаруживаю паттерны в ваших запросах
• Адаптируюсь к вашему стилю общения со временем
• Сохраняю важную информацию для персонализации ответов
• Все ваши данные хранятся приватно и изолированно

💬 **Просто напишите свой запрос — и я постараюсь помочь наилучшим образом!**"""

    try:
        await update.message.reply_text(
            help_text,
            parse_mode='MarkdownV2',
            disable_web_page_preview=True
        )
    except BadRequest as e:
        logging.warning(f"Ошибка Markdown в help: {e}")
        await update.message.reply_text(help_text.replace('*', '').replace('`', '').replace('_', '\\_'))


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stats"""
    user_id = update.effective_user.id
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    await update.message.reply_chat_action("typing")
    stats_text = agent.get_comprehensive_stats()

    await update.message.reply_text(
        f"```\n{stats_text}\n```",
        parse_mode='MarkdownV2',
        reply_markup=create_main_keyboard()
    )


async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /think — активация глубокого мышления"""
    user_id = update.effective_user.id
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    await update.message.reply_text(
        "🧠 Запускаю глубокое многоуровневое мышление...\n"
        "Анализирую наши диалоги, обнаруживаю паттерны и генерирую инсайты.\n"
        "Это займёт 10-20 секунд...",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("⏳ Ожидание завершения...", callback_data="noop")
        ]])
    )

    # Запускаем глубокое мышление асинхронно
    await agent._deep_autonomous_thinking()

    await update.message.reply_text(
        "✅ Глубокое мышление завершено!\n"
        "Проверьте инсайты командой /insights или паттерны командой /patterns",
        reply_markup=create_main_keyboard()
    )


async def patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /patterns"""
    user_id = update.effective_user.id
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    await update.message.reply_chat_action("typing")
    patterns_text = agent.get_patterns_summary()

    await update.message.reply_text(
        patterns_text,
        reply_markup=create_main_keyboard()
    )


async def insights_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /insights"""
    user_id = update.effective_user.id
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    await update.message.reply_chat_action("typing")
    insights_text = agent.get_insights_summary()

    await update.message.reply_text(
        insights_text,
        reply_markup=create_main_keyboard()
    )


async def goals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /goals"""
    user_id = update.effective_user.id
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    await update.message.reply_chat_action("typing")
    goals_text = agent.get_goals_summary()

    await update.message.reply_text(
        goals_text,
        reply_markup=create_main_keyboard()
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clear — очистка контекста"""
    user_id = update.effective_user.id
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    agent.context_window.clear()

    await update.message.reply_text(
        "🧹 Контекст текущего диалога очищен.\n"
        "Я больше не помню предыдущие сообщения в этом разговоре.\n"
        "Вся сохранённая долгосрочная память (факты, паттерны) остаётся нетронутой.",
        reply_markup=create_main_keyboard()
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Основной обработчик текстовых сообщений"""
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_message = update.message.text.strip()

    if not user_message:
        return

    logging.info(f"📨 Пользователь {user_id}: {user_message[:50]}...")

    try:
        # Получаем или создаём сессию пользователя
        session_manager = context.application.bot_data['session_manager']
        agent = await session_manager.get_or_create_session(user_id)

        # Показываем индикатор набора текста
        await update.message.reply_chat_action("typing")

        # Имитация "мышления" для лучшего UX (реальная обработка уже асинхронна)
        if len(user_message) > 50 or '?' in user_message:
            await asyncio.sleep(min(0.5, Config.TYPING_DELAY))

        # Обрабатываем сообщение через когнитивного агента
        response = await agent.process_message(user_message)

        # Разбиваем длинные ответы
        parts = split_message(response)

        # Отправляем части ответа
        for i, part in enumerate(parts):
            # Для последней части добавляем клавиатуру каждые 5 сообщений
            reply_markup = create_main_keyboard() if (
                    i == len(parts) - 1 and
                    agent.interaction_count % 5 == 0
            ) else None

            await update.message.reply_text(
                part,
                reply_markup=reply_markup,
                disable_web_page_preview=True
            )

            # Небольшая пауза между частями для лучшего восприятия
            if i < len(parts) - 1:
                await asyncio.sleep(0.4)

    except Exception as e:
        logging.error(f"❌ Ошибка обработки сообщения от {user_id}: {e}", exc_info=True)
        error_msg = (
            "⚠️ Произошла ошибка при обработке вашего запроса.\n"
            "Попробуйте переформулировать вопрос или использовать команду /start"
        )
        await update.message.reply_text(error_msg, reply_markup=create_main_keyboard())


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    if callback_data == "noop":
        return  # Игнорируем служебные кнопки

    # Получаем сессию пользователя
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)

    # Обработка разных типов кнопок
    handlers = {
        "deep_think": lambda: think_command(update, context),
        "analysis": lambda: stats_command(update, context),
        "stats": lambda: stats_command(update, context),
        "goals": lambda: goals_command(update, context),
        "insights": lambda: insights_command(update, context),
        "patterns": lambda: patterns_command(update, context),
        "clear": lambda: clear_command(update, context),
        "help": lambda: help_command(update, context)
    }

    handler = handlers.get(callback_data)
    if handler:
        # Создаём имитацию сообщения для совместимости с командными обработчиками
        class FakeUpdate:
            def __init__(self, real_update, text):
                self.message = type('obj', (object,), {
                    'reply_text': real_update.message.reply_text,
                    'reply_chat_action': real_update.message.reply_chat_action,
                    'chat_id': real_update.effective_chat.id
                })
                self.effective_user = real_update.effective_user
                self.effective_chat = real_update.effective_chat

        fake_update = FakeUpdate(update, callback_data)
        await handler()
    else:
        await query.edit_message_text(
            "❓ Неизвестная команда. Используйте кнопки или команды из меню.",
            reply_markup=create_main_keyboard()
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    error = context.error
    logging.error(f"Глобальная ошибка: {error}", exc_info=error)

    error_msg = "⚠️ Произошла непредвиденная ошибка. Попробуйте позже."

    # Специальная обработка известных типов ошибок
    if isinstance(error, telegram.error.TimedOut):
        error_msg = "⏱️ Превышено время ожидания. Повторите запрос через несколько секунд."
    elif isinstance(error, telegram.error.NetworkError):
        error_msg = "🌐 Проблемы с сетью. Проверьте подключение и повторите попытку."
    elif "flood" in str(error).lower() or "retry" in str(error).lower():
        error_msg = "⏳ Слишком много запросов. Подождите 10-15 секунд перед следующим сообщением."

    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                f"{error_msg}\n\nПодробности: {str(error)[:100]}",
                reply_markup=create_main_keyboard()
            )
        except Exception as e:
            logging.error(f"Не удалось отправить сообщение об ошибке: {e}")


async def post_init(application: Application) -> None:
    """Инициализация после запуска приложения"""
    # Установка команд меню
    await application.bot.set_my_commands([
        BotCommand("start", "Начать работу"),
        BotCommand("think", "Глубокое мышление"),
        BotCommand("stats", "Статистика"),
        BotCommand("patterns", "Паттерны"),
        BotCommand("insights", "Инсайты"),
        BotCommand("goals", "Цели"),
        BotCommand("clear", "Очистить контекст"),
        BotCommand("help", "Помощь")
    ])

    # Инициализация компонентов
    api_key = Config.get_api_key()
    db = EnhancedMemoryDB(Config.DB_PATH)
    thinker = EnhancedThinkingSystem(api_key)
    session_manager = SessionManager(db, thinker)

    # Сохранение в глобальные данные бота
    application.bot_data['session_manager'] = session_manager
    application.bot_data['db'] = db
    application.bot_data['thinker'] = thinker

    # Вывод статистики запуска
    global_stats = session_manager.get_global_stats()
    logging.info(
        f"✅ Бот успешно инициализирован\n"
        f"   Пользователей всего: {global_stats['total_users']}\n"
        f"   Активных за неделю: {global_stats['active_users']}\n"
        f"   Фактов в памяти: {global_stats['total_facts']}"
    )

    print("\n" + "=" * 70)
    print("✅ AGI Когнитивный Ассистент успешно запущен!")
    print("=" * 70)
    print("\n📱 Найдите бота в Telegram и напишите /start")
    print("   Или перейдите по ссылке: https://t.me/ваш_бот")
    print("\n🛑 Для остановки нажмите Ctrl+C")
    print("=" * 70 + "\n")


async def post_shutdown(application: Application) -> None:
    """Очистка ресурсов при завершении"""
    print("\n🔄 Завершение работы бота...")

    # Получение статистики
    if 'session_manager' in application.bot_data:
        session_manager = application.bot_data['session_manager']
        stats = session_manager.get_global_stats()

        print(f"\n📊 Финальная статистика:")
        print(f"   • Всего пользователей: {stats['total_users']}")
        print(f"   • Обработано сообщений: {stats['total_interactions']}")
        print(f"   • Сохранено фактов: {stats['total_facts']}")
        print(f"   • Обнаружено паттернов: {stats['total_patterns']}")

    print("\n✅ Бот завершил работу корректно")


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Config.LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)


async def main():
    """Основная асинхронная функция запуска бота"""
    setup_logging()
    logging.info("=" * 70)
    logging.info("🚀 ЗАПУСК AGI КОГНИТИВНОГО АССИСТЕНТА (Python 3.13+)")
    logging.info("=" * 70)

    try:
        # Получение токена
        token = Config.get_telegram_token()
        logging.info("✅ Токен Telegram получен")

        # Создание приложения
        application = (
            ApplicationBuilder()
            .token(token)
            .read_timeout(30)
            .write_timeout(30)
            .connect_timeout(10)
            .pool_timeout(10)
            .get_updates_read_timeout(42)
            .post_init(post_init)
            .post_shutdown(post_shutdown)
            .build()
        )

        # Регистрация обработчиков
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("stats", stats_command))
        application.add_handler(CommandHandler("think", think_command))
        application.add_handler(CommandHandler("patterns", patterns_command))
        application.add_handler(CommandHandler("insights", insights_command))
        application.add_handler(CommandHandler("goals", goals_command))
        application.add_handler(CommandHandler("clear", clear_command))

        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(CallbackQueryHandler(button_callback))
        application.add_error_handler(error_handler)

        # Запуск бота
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )

        logging.info("🔄 Бот работает в режиме ожидания сообщений...")
        logging.info("   (Нажмите Ctrl+C для остановки)\n")

        # Бесконечное ожидание
        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        logging.info("👋 Получен сигнал остановки (Ctrl+C)...")
        raise
    except ValueError as e:
        logging.error(f"❌ Ошибка конфигурации: {e}")
        print("\n💡 Создайте файл .env в корне проекта со следующим содержимым:")
        print("OPENROUTER_API_KEY=ваш_ключ_openrouter")
        print("TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather")
        raise
    except Exception as e:
        logging.exception(f"🚨 Критическая ошибка: {e}")
        raise
    finally:
        # Корректное завершение
        if 'application' in locals():
            try:
                await application.stop()
                await application.shutdown()
            except Exception as e:
                logging.error(f"⚠️ Ошибка при завершении: {e}")


# ================= ТОЧКА ВХОДА =================
def run():
    """Точка входа для запуска бота"""
    print("AGI Cognitive Assistant — Telegram Edition")
    print("Copyright (c) 2024-2026 AGI Research Group")
    print(f"Python version: {sys.version.split()[0]}")
    print("\n" + "=" * 70)

    # Проверка версии Python
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        print(f"📌 У вас установлен Python {sys.version}")
        sys.exit(1)

    if sys.version_info >= (3, 13):
        print("✅ Python 3.13+ обнаружен — применяются специальные настройки")

    # Проверка обязательных библиотек
    try:
        import aiohttp
        print("✅ aiohttp загружен")
    except ImportError:
        print("❌ Библиотека aiohttp не установлена")
        print("📦 Установите: pip install aiohttp")
        sys.exit(1)

    print("=" * 70)
    print("🚀 Запуск когнитивного агента...")
    print("=" * 70 + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем (Ctrl+C)")
        print("\n✅ Работа завершена корректно")
        sys.exit(0)
    except Exception as e:
        print(f"\n🚨 Критическая ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()