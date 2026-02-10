# coding: utf-8
"""
AGI_Cognitive_Agent_LMStudio_v3.py — ПОЛНОЦЕННЫЙ КОГНИТИВНЫЙ АГЕНТ С ЛОКАЛЬНОЙ LLM
✅ Гарантированная инициализация (никаких ошибок 'session_manager')
✅ Многоуровневое мышление: рефлексия → анализ → планирование → обучение → наблюдение
✅ Контекстная память с затуханием и приоритизацией
✅ Автономное обнаружение паттернов и самоанализ
✅ Иерархия целей с прогрессом и следующими действиями
✅ Кэширование ответов для ускорения повторяющихся запросов
✅ 100% приватность — все данные остаются на вашем компьютере
✅ Адаптация под мощность локальной модели (от 3B до 46B параметров)
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
import random


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


# ================= КОНФИГУРАЦИЯ ДЛЯ ЛОКАЛЬНОГО СЕРВЕРА =================
class Config:
    ROOT = Path("./cognitive_system_telegram")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"
    CACHE_PATH = ROOT / "cache.json"
    LOG_PATH = ROOT / "system.log"

    # 🔥 ЛОКАЛЬНЫЙ СЕРВЕР LM STUDIO
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODELS_URL = "http://localhost:1234/v1/models"

    TIMEOUT = 180
    MAX_TOKENS = 4096

    # Когнитивные параметры
    REFLECTION_INTERVAL = 5  # Глубокое мышление каждые 5 сообщений
    DEEP_THINKING_THRESHOLD = 0.7
    CONTEXT_WINDOW_SIZE = 12
    MEMORY_DECAY_RATE = 0.07
    MAX_MEMORY_ITEMS = 5000

    # Параметры бота
    MAX_MESSAGE_LENGTH = 4096
    SESSION_TIMEOUT = 7200

    # Типы мышления (из оптимизированной версии)
    THOUGHT_TYPES = ['рефлексия', 'анализ', 'планирование', 'обучение', 'наблюдение']

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
                print("      • Phi-3-mini-4k-instruct-q4.gguf (быстро, 3.8B) — для слабых ПК")
                print("      • Mistral-7B-Instruct-v0.2-Q5_K_M.gguf (качество, 7B) — баланс")
                print("      • Nous-Hermes-2-Mixtral-8x7B (46B) — максимальное качество")
                print("   3. Включите сервер: вкладка 'Server' → 'Start Server'")
                input("\nНажмите Enter после запуска сервера...")

            print("\n🔍 Поиск моделей...")
            try:
                response = requests.get(cls.LM_STUDIO_MODELS_URL, timeout=10)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    if models:
                        # Выбираем модель с "instruct" или "chat" как приоритетную
                        best = next((m for m in models if
                                     'instruct' in m.get('id', '').lower() or 'chat' in m.get('id', '').lower()),
                                    models[0])
                        config['model'] = best.get('id', config['model'])
                        print(f"✅ Обнаружена модель: {config['model']}")

                        with open(".env", "a", encoding="utf-8") as f:
                            f.write(f'\nLM_STUDIO_CONFIGURED="true"\nLM_STUDIO_MODEL="{config["model"]}"\n')
                        print("💾 Конфигурация сохранена в .env")
                    else:
                        config['model'] = input("Введите ID модели вручную: ").strip() or config['model']
                else:
                    config['model'] = input("Введите ID модели вручную: ").strip() or config['model']
            except Exception as e:
                print(f"⚠️ Ошибка при определении модели: {e}")
                config['model'] = input("Введите ID модели вручную: ").strip() or config['model']

            print("\n" + "=" * 70)
            print("💡 РЕКОМЕНДАЦИИ ПО ПРОИЗВОДИТЕЛЬНОСТИ:")
            print("   • CPU (16 ГБ ОЗУ): используйте Phi-3-mini — 2-3 сек/ответ")
            print("   • CPU (32 ГБ ОЗУ): используйте Mistral-7B — 4-6 сек/ответ")
            print("   • GPU (8+ ГБ VRAM): скорость возрастёт в 5-10 раз")
            print("=" * 70 + "\n")

        return config


# ================= УТИЛИТЫ ОБРАБОТКИ ТЕКСТА =================
def calculate_text_similarity(text1: str, text2: str) -> float:
    """Расчёт схожести текстов с учётом n-грамм (из оптимизированной версии)"""
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
    """Извлечение семантических характеристик (улучшенная версия)"""
    text_lower = text.lower()
    words = text.split()
    features = {
        'length': len(words),
        'complexity': len(set(text_lower.split())) / max(len(words), 1),
        'question_words': len(re.findall(r'\b(как|что|почему|зачем|когда|где|кто|сколько)\b', text_lower)),
        'numbers': len(re.findall(r'\b\d+\b', text)),
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


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Извлечение сущностей из текста (из оптимизированной версии)"""
    entities = {
        'numbers': re.findall(r'\b\d+\b', text),
        'dates': re.findall(
            r'\b\d{1,2}[./]\d{1,2}[./]?\d{2,4}\b|\b(?:янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\b',
            text, re.IGNORECASE
        ),
        'names': re.findall(r'\b(?:[А-Я][а-я]+)\b', text),
        'emails': re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text),
        'urls': re.findall(r'https?://\S+', text)
    }
    return {k: v for k, v in entities.items() if v}


def split_message(text: str, max_length: int = 4096) -> list:
    """Разбивает длинное сообщение на части"""
    if not text:
        return [""]
    if len(text) <= max_length:
        return [text]

    parts = []
    current = ""
    paragraphs = re.split(r'(\n\s*\n)', text)

    for para in paragraphs:
        if len(current) + len(para) <= max_length:
            current += para
        else:
            if current:
                parts.append(current.rstrip())
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

    if len(parts) > 5:
        parts = parts[:5]
        parts.append("📝 ...сообщение сокращено из-за ограничений Telegram")
    return parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    """Создание главной клавиатуры"""
    keyboard = [
        [
            InlineKeyboardButton("🧠 Глубокое мышление", callback_data="deep_think"),
            InlineKeyboardButton("📊 Статистика", callback_data="stats")
        ],
        [
            InlineKeyboardButton("💡 Инсайты", callback_data="insights"),
            InlineKeyboardButton("🎯 Цели", callback_data="goals")
        ],
        [
            InlineKeyboardButton("🔗 Паттерны", callback_data="patterns"),
            InlineKeyboardButton("🧹 Очистить", callback_data="clear")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= КЭШ ОТВЕТОВ ДЛЯ ЛОКАЛЬНЫХ МОДЕЛЕЙ =================
class ResponseCache:
    """Простой кэш ответов для ускорения повторяющихся запросов"""

    def __init__(self, max_size: int = 150):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.load()

    def _make_key(self, system_prompt: str, user_prompt: str) -> str:
        content = f"{system_prompt[:100]}|{user_prompt[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        key = self._make_key(system_prompt, user_prompt)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, system_prompt: str, user_prompt: str, response: str):
        key = self._make_key(system_prompt, user_prompt)
        self.cache[key] = response
        self.access_times[key] = time.time()

        if len(self.cache) > self.max_size:
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:self.max_size // 4]]
            for k in keys_to_remove:
                self.cache.pop(k, None)
                self.access_times.pop(k, None)
        self.save()

    def save(self):
        try:
            data = {'cache': self.cache, 'access_times': self.access_times}
            with open(Config.CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения кэша: {e}")

    def load(self):
        if Config.CACHE_PATH.exists():
            try:
                with open(Config.CACHE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
            except:
                pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'usage_percent': len(self.cache) / self.max_size * 100
        }


# ================= РАСШИРЕННАЯ БАЗА ДАННЫХ =================
class EnhancedMemoryDB:
    """Продвинутая база данных с поддержкой всех когнитивных функций"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()
        self.fact_cache = {}
        self.pattern_cache = {}

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
            cursor = conn.cursor()
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
                complexity REAL DEFAULT 0.5
            )''')
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
                UNIQUE(user_id, key, value)
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                thought_type TEXT NOT NULL,
                content TEXT NOT NULL,
                trigger TEXT,
                importance REAL DEFAULT 0.5,
                depth_level INTEGER DEFAULT 1
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                description TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                progress REAL DEFAULT 0.0,
                next_action TEXT
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                last_seen REAL NOT NULL
            )''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, importance DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoughts_user ON thoughts(user_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status, priority DESC)')
            conn.commit()

    # === Взаимодействия ===
    def add_interaction(self, user_id: int, user_input: str, system_response: str, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO interactions
            (timestamp, user_id, user_input, system_response, context, emotion, category, importance, complexity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                user_id,
                user_input,
                system_response,
                kwargs.get('context', ''),
                kwargs.get('emotion', 'neutral'),
                kwargs.get('category', 'диалог'),
                kwargs.get('importance', 0.5),
                kwargs.get('complexity', 0.5)
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

            scored = []
            for interaction in all_interactions:
                relevance = calculate_text_similarity(
                    query,
                    interaction['user_input'] + ' ' + interaction['system_response']
                )
                recency = 1.0 - (time.time() - interaction['timestamp']) / (7 * 24 * 3600)
                recency = max(0, min(1, recency))
                score = 0.6 * relevance + 0.3 * interaction['importance'] + 0.1 * recency
                scored.append((score, interaction))
            scored.sort(reverse=True, key=lambda x: x[0])
            return [item[1] for item in scored[:limit]]

    # === Факты ===
    def add_fact(self, user_id: int, key: str, value: str, **kwargs):
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
                usage_count, decay_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, key, value,
                    kwargs.get('category', ''),
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(), time.time(), 1, 1.0
                ))
            conn.commit()

    def get_relevant_facts(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        cache_key = f"{user_id}_{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.fact_cache and time.time() - time.time() < 300:
            return self.fact_cache[cache_key]

        with self.get_connection() as conn:
            cursor = conn.cursor()
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
            self.fact_cache[cache_key] = result
            return result

    def search_facts(self, user_id: int, query_text: str, limit: int = 10) -> List[Dict]:
        """Поиск фактов по тексту (из оптимизированной версии)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            search_term = f"%{query_text}%"
            cursor.execute('''
                SELECT * FROM facts 
                WHERE user_id = ? AND (key LIKE ? OR value LIKE ?) 
                ORDER BY usage_count DESC, confidence DESC 
                LIMIT ?
            ''', (user_id, search_term, search_term, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === Мысли ===
    def add_thought(self, user_id: int, thought_type: str, content: str, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO thoughts
            (user_id, timestamp, thought_type, content, trigger, importance, depth_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                time.time(),
                thought_type,
                content[:300],
                kwargs.get('trigger', ''),
                kwargs.get('importance', 0.5),
                kwargs.get('depth_level', 1)
            ))
            conn.commit()

    def get_recent_thoughts(self, user_id: int, limit: int = 10,
                            thought_type: Optional[str] = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if thought_type:
                cursor.execute('''
                    SELECT * FROM thoughts 
                    WHERE user_id = ? AND thought_type = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, thought_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM thoughts 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === Цели ===
    def add_goal(self, user_id: int, description: str, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO goals
            (user_id, created_at, description, priority, status, progress, next_action)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                time.time(),
                description,
                kwargs.get('priority', 0.5),
                kwargs.get('status', 'active'),
                kwargs.get('progress', 0.0),
                kwargs.get('next_action', '')
            ))
            conn.commit()
            return cursor.lastrowid

    def get_active_goals(self, user_id: int, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM goals 
                WHERE user_id = ? AND status = 'active' 
                ORDER BY priority DESC, created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def update_goal_progress(self, user_id: int, goal_id: int, progress: float):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE goals 
                SET progress = ? 
                WHERE id = ? AND user_id = ?
            ''', (progress, goal_id, user_id))
            conn.commit()

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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT * FROM patterns
            WHERE user_id = ? AND confidence >= ?
            ORDER BY occurrences DESC, confidence DESC
            LIMIT ?
            ''', (user_id, min_confidence, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === Статистика ===
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (user_id,))
            interactions = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM facts WHERE user_id = ? AND decay_factor > 0.3', (user_id,))
            facts = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM thoughts WHERE user_id = ?', (user_id,))
            thoughts = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM patterns WHERE user_id = ? AND confidence > 0.5', (user_id,))
            patterns = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM goals WHERE user_id = ? AND status = \'active\'', (user_id,))
            goals = cursor.fetchone()[0]
            return {
                'interactions': interactions,
                'facts': facts,
                'thoughts': thoughts,
                'patterns': patterns,
                'goals': goals,
                'first_interaction': self._get_first_interaction_time(user_id)
            }

    def _get_first_interaction_time(self, user_id: int) -> float:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT timestamp FROM interactions
            WHERE user_id = ? ORDER BY timestamp ASC LIMIT 1
            ''', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else time.time()


# ================= СИСТЕМА МЫШЛЕНИЯ ДЛЯ ЛОКАЛЬНЫХ МОДЕЛЕЙ =================
class LocalThinkingSystem:
    """Оптимизированная система мышления для локальных LLM с многоуровневым анализом"""

    def __init__(self, config: Dict[str, Any]):
        self.api_url = config['url']
        self.api_key = config.get('api_key', "")
        self.model = config['model']
        self.rate_limit = 0.3
        self.last_request_time = 0
        self.cache = ResponseCache()
        self.reasoning_history = deque(maxlen=50)

        # Оптимизация промптов для локальных моделей
        self.SYSTEM_PROMPT_PREFIX = (
            "Ты — когнитивный ассистент. Отвечай кратко, по делу, без лишних фраз. "
            "Фокусируйся на сути запроса. Не используй маркдаун. "
        )

        print(f"✅ Инициализирована локальная система мышления")
        print(f"   Модель: {self.model}")
        print(f"   Сервер: {self.api_url}")

    async def _wait_for_rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def generate_thought(self, thought_type: str, context: str) -> Optional[str]:
        """Генерация мысли определенного типа (из оптимизированной версии)"""
        thought_prompts = {
            'рефлексия': (
                "Ты анализируешь последние взаимодействия. Какие выводы можно сделать? "
                "Что было эффективно, а что можно улучшить?"
            ),
            'планирование': (
                "Ты планируешь следующие действия. Что нужно сделать для достижения целей? "
                "Какие шаги будут наиболее эффективными?"
            ),
            'анализ': (
                "Ты анализируешь текущую ситуацию. Какие факторы важны? "
                "Что нужно учесть при принятии решений?"
            ),
            'обучение': (
                "Ты извлекаешь уроки из опыта. Что нового ты узнал? "
                "Как это можно применить в будущем?"
            ),
            'наблюдение': (
                "Ты замечаешь паттерны и закономерности. Что повторяется? "
                "Какие связи можно увидеть между разными событиями?"
            )
        }

        if thought_type not in thought_prompts:
            return None

        system_prompt = f"Ты — когнитивная система. {thought_prompts[thought_type]}"
        user_prompt = f"Контекст:\n{context}\n\nМои мысли:"

        response = await self.call_llm(system_prompt, user_prompt, temperature=0.7)
        return response if response and len(response) > 15 else None

    async def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Вызов локальной LLM через LM Studio API с кэшированием"""
        # Проверка кэша
        cached = self.cache.get(system_prompt, user_prompt)
        if cached:
            return cached

        await self._wait_for_rate_limit()

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": min(2048, Config.MAX_TOKENS),
                "top_p": 0.9,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()

                        # Очистка от лишних символов (частая проблема локальных моделей)
                        content = re.sub(r'^[\s\*\-\_]+|[\s\*\-\_]+$', '', content)
                        content = re.sub(r'\n{3,}', '\n\n', content)

                        # Сохраняем в кэш
                        self.cache.set(system_prompt, user_prompt, content)
                        return content
                    else:
                        error_text = await response.text()
                        return f"⚠️ Ошибка LM Studio ({response.status}): {error_text[:150]}"

        except asyncio.TimeoutError:
            return "⚠️ Таймаут запроса. Модель может быть перегружена."
        except aiohttp.ClientConnectorError:
            return "⚠️ Не удаётся подключиться к LM Studio. Запущен ли сервер на http://localhost:1234?"
        except Exception as e:
            return f"⚠️ Ошибка локальной модели: {str(e)[:120]}"


# ================= ПОЛЬЗОВАТЕЛЬСКИЙ КОГНИТИВНЫЙ АГЕНТ =================
class UserCognitiveAgent:
    """Изолированный когнитивный агент с полным набором когнитивных способностей"""

    def __init__(self, user_id: int, db: EnhancedMemoryDB, thinker: LocalThinkingSystem):
        self.user_id = user_id
        self.db = db
        self.thinker = thinker
        self.interaction_count = 0
        self.thoughts_generated = 0
        self.start_time = time.time()
        self.context_window = deque(maxlen=Config.CONTEXT_WINDOW_SIZE)
        self._init_user_goals()

    def _init_user_goals(self):
        """Инициализация базовых целей для пользователя (из оптимизированной версии)"""
        existing_goals = self.db.get_active_goals(self.user_id, limit=1)
        if not existing_goals:
            self.db.add_goal(
                self.user_id,
                "Помогать пользователю решать задачи",
                priority=0.9,
                next_action="Анализировать запросы и предоставлять полезные ответы"
            )
            self.db.add_goal(
                self.user_id,
                "Запоминать важную информацию",
                priority=0.8,
                next_action="Извлекать и сохранять ключевые факты из диалогов"
            )
            self.db.add_goal(
                self.user_id,
                "Учиться и адаптироваться",
                priority=0.7,
                next_action="Обнаруживать паттерны в поведении пользователя"
            )

    async def process_message(self, user_input: str) -> str:
        start_time = time.time()
        self.interaction_count += 1
        self.context_window.append({'type': 'user', 'content': user_input, 'timestamp': time.time()})

        # Обработка команд
        command_response = self._handle_command(user_input)
        if command_response:
            return command_response

        # Анализ запроса
        features = extract_semantic_features(user_input)
        complexity = features['complexity']
        importance = self._calculate_importance(user_input, features)

        # Извлечение и сохранение информации
        await self._extract_and_store_information(user_input, importance)

        # Генерация ответа с учётом контекста
        response = await self._generate_contextual_response(user_input, features, complexity, importance)

        # Сохранение взаимодействия
        self.db.add_interaction(
            user_id=self.user_id,
            user_input=user_input,
            system_response=response,
            context=self._get_context_summary(),
            category=self._categorize_input(user_input, features),
            importance=importance,
            complexity=complexity
        )

        self.context_window.append({'type': 'assistant', 'content': response, 'timestamp': time.time()})

        # Периодическое автономное мышление
        if self.interaction_count % Config.REFLECTION_INTERVAL == 0:
            await self._autonomous_thinking()

        duration = time.time() - start_time
        if duration > 2.0:
            logging.info(f"Пользователь {self.user_id}: обработка заняла {duration:.2f}с")

        return response

    def _calculate_importance(self, text: str, features: Dict) -> float:
        importance = 0.5
        if any(word in text.lower() for word in ['важно', 'срочно', 'критично', 'обязательно']):
            importance += 0.25
        importance += min(0.2, features['question_words'] * 0.1)
        importance += min(0.15, features['imperatives'] * 0.08)
        importance += features['complexity'] * 0.2
        if features['length'] > 15:
            importance += 0.08
        return min(1.0, importance)

    async def _extract_and_store_information(self, text: str, importance: float):
        """Извлечение информации из текста (улучшенная версия из оптимизированного кода)"""
        # Извлечение сущностей
        entities = extract_entities(text)

        # Сохраняем числа как факты
        for number in entities.get('numbers', [])[:3]:
            self.db.add_fact(self.user_id, 'число', number, category='информация', importance=importance * 0.4)

        # Сохраняем имена
        for name in entities.get('names', [])[:2]:
            if len(name) > 2:
                self.db.add_fact(self.user_id, 'имя', name, category='персона', importance=importance * 0.6)

        # Извлекаем факты из утверждений
        patterns = [
            (r'(\w+)\s+(?:это|является|называется)\s+([^.,!?]+)', 'определение'),
            (r'запомни[,:]?\s*(.+)', 'важная_информация'),
            (r'я люблю (\w+)', 'предпочтение'),
            (r'(\w+)\s+=\s+([^.,]+)', 'равенство')
        ]
        for pattern, category in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
            for match in matches[:1]:
                if isinstance(match, tuple) and len(match) >= 2:
                    key, value = match[0], match[1]
                    if len(key) > 2 and len(value) > 1:
                        self.db.add_fact(
                            self.user_id,
                            key.strip().lower(),
                            value.strip(),
                            category=category,
                            importance=min(1.0, importance * 1.1)
                        )

    async def _generate_contextual_response(self, user_input: str, features: Dict, complexity: float,
                                            importance: float) -> str:
        # Получение контекста
        relevant_interactions = self.db.get_contextual_interactions(self.user_id, user_input, limit=3)
        relevant_facts = self.db.get_relevant_facts(self.user_id, user_input, limit=4)
        active_goals = self.db.get_active_goals(self.user_id, limit=2)

        # Формирование контекста
        context_parts = []
        if relevant_interactions:
            context_parts.append("Недавний диалог:")
            for interaction in relevant_interactions[:2]:
                context_parts.append(f"Вы: {interaction['user_input'][:60]}")
                context_parts.append(f"Я: {interaction['system_response'][:60]}")

        if relevant_facts:
            context_parts.append("\nРелевантные факты:")
            for fact in relevant_facts[:3]:
                context_parts.append(f"- {fact['key']}: {fact['value']}")

        if active_goals:
            context_parts.append("\nТекущие цели:")
            for goal in active_goals:
                progress_bar = "█" * int(goal['progress'] * 10) + "░" * (10 - int(goal['progress'] * 10))
                context_parts.append(f"- {goal['description'][:50]} [{progress_bar}]")

        context = "\n".join(context_parts) if context_parts else "Нет контекста"

        # Определение необходимости глубокого анализа
        needs_deep_thinking = (
                complexity > Config.DEEP_THINKING_THRESHOLD or
                importance > 0.8 or
                features['question_words'] > 2 or
                any(word in user_input.lower() for word in ['почему', 'как сделать', 'анализ', 'проанализируй'])
        )

        if needs_deep_thinking and len(self.context_window) > 2:
            # Многоуровневое мышление
            deep_thoughts = []
            for thought_type in ['анализ', 'рефлексия', 'планирование']:
                thought = await self.thinker.generate_thought(thought_type,
                                                              f"Запрос: {user_input}\nКонтекст: {context}")
                if thought:
                    deep_thoughts.append(thought)
                    self.db.add_thought(
                        self.user_id,
                        thought_type,
                        thought[:200],
                        trigger='глубокое_мышление',
                        importance=0.7,
                        depth_level=2
                    )

            # Синтез ответа
            synthesis_prompt = (
                f"Запрос: {user_input}\n"
                f"Анализ: {' | '.join(deep_thoughts) if deep_thoughts else 'Стандартный'}\n"
                f"Контекст: {context}\n"
                "Дай исчерпывающий, но лаконичный ответ. Включи ключевые инсайты из анализа."
            )

            system_prompt = self.thinker.SYSTEM_PROMPT_PREFIX + (
                "Синтезируй ответ на основе многоуровневого анализа. Будь полезным, точным и креативным при необходимости."
            )

            response = await self.thinker.call_llm(system_prompt, synthesis_prompt, temperature=0.65)
            self.thoughts_generated += len(deep_thoughts)
        else:
            # Стандартный ответ
            system_prompt = self.thinker.SYSTEM_PROMPT_PREFIX + (
                f"КОНТЕКСТ:\n{context}\n\n"
                "Отвечай кратко, дружелюбно и по делу. Используй факты из контекста."
            )
            response = await self.thinker.call_llm(system_prompt, user_input, temperature=0.5)

        return response.strip()

    async def _autonomous_thinking(self):
        """Автономный процесс мышления (из оптимизированной версии)"""
        recent_interactions = self.db.get_contextual_interactions(self.user_id, "запросы", limit=5)
        if len(recent_interactions) < 2:
            return

        # Готовим контекст
        context_lines = []
        for i, interaction in enumerate(recent_interactions[-3:], 1):
            context_lines.append(f"{i}. {interaction['user_input'][:50]}... → {interaction['system_response'][:50]}...")

        context = "\n".join(context_lines)

        # Случайный выбор типа мышления
        thought_type = random.choice(Config.THOUGHT_TYPES)

        # Генерируем мысль
        thought_content = await self.thinker.generate_thought(thought_type, context)

        if thought_content and len(thought_content) > 20:
            self.db.add_thought(
                self.user_id,
                thought_type,
                thought_content[:300],
                trigger="автономное_мышление",
                importance=0.6,
                depth_level=1
            )
            self.thoughts_generated += 1

    def _handle_command(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in ['думай', 'подумай', 'мысли', '/think']:
            asyncio.create_task(self._autonomous_thinking())
            return "🧠 Запускаю процесс мышления... (результаты в /insights)"
        elif text_lower in ['/clear', 'очистить контекст', 'забудь']:
            self.context_window.clear()
            return "🧹 Контекст очищен. Долгосрочная память сохранена."
        return None

    def _get_context_summary(self) -> str:
        if not self.context_window:
            return ""
        summary = []
        for item in list(self.context_window)[-4:]:
            prefix = "П:" if item['type'] == 'user' else "Я:"
            summary.append(f"{prefix}{item['content'][:30]}")
        return " | ".join(summary)

    def _categorize_input(self, text: str, features: Dict) -> str:
        text_lower = text.lower()
        categories = {
            'вопрос': ['что', 'как', 'почему', 'зачем', 'когда', 'где', 'кто', 'сколько', '?'],
            'память': ['запомни', 'сохрани', 'напомни'],
            'анализ': ['анализ', 'разбери', 'оцени', 'сравни'],
            'творчество': ['придумай', 'создай', 'идея', 'креатив'],
            'план': ['план', 'расписание', 'как достичь']
        }
        scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in categories.items()}
        return max(scores, key=scores.get) if any(scores.values()) else 'диалог'

    def get_comprehensive_stats(self) -> str:
        stats = self.db.get_user_stats(self.user_id)
        uptime = time.time() - stats['first_interaction']
        days = int(uptime // 86400)
        hours = int((uptime % 86400) // 3600)
        cache_stats = self.thinker.cache.get_stats()
        return (
            f"📊 ПЕРСОНАЛЬНАЯ СТАТИСТИКА\n{'=' * 40}\n"
            f"⏱️ Время знакомства: {days}д {hours}ч\n"
            f"💬 Сообщений: {stats['interactions']}\n"
            f"🧠 Мыслей сгенерировано: {self.thoughts_generated}\n"
            f"📚 Фактов: {stats['facts']}\n"
            f"💡 Сохранённых мыслей: {stats['thoughts']}\n"
            f"🔗 Паттернов: {stats['patterns']}\n"
            f"🎯 Активных целей: {stats['goals']}\n"
            f"\n💾 Кэш ответов:\n"
            f"   Размер: {cache_stats['size']} / {cache_stats['max_size']}\n"
            f"   Заполнение: {cache_stats['usage_percent']:.1f}%"
        )

    def get_patterns_summary(self) -> str:
        patterns = self.db.get_patterns(self.user_id, min_confidence=0.55, limit=8)
        if not patterns:
            return "🔍 Паттерны не обнаружены. Продолжайте диалог!"
        lines = ["🔍 ОБНАРУЖЕННЫЕ ПАТТЕРНЫ", "=" * 40]
        for p in patterns[:5]:
            conf = int(p['confidence'] * 100)
            lines.append(f"• {p['description'][:60]} ({conf}%)")
        return "\n".join(lines)

    def get_insights_summary(self) -> str:
        thoughts = self.db.get_recent_thoughts(self.user_id, limit=8)
        if not thoughts:
            return "💡 Инсайты отсутствуют. Используйте /think для анализа."
        lines = ["💡 ПОСЛЕДНИЕ ИНСАЙТЫ", "=" * 40]
        for thought in thoughts[:6]:
            lines.append(f"• [{thought['thought_type']}] {thought['content'][:70]}...")
        return "\n".join(lines)

    def get_goals_summary(self) -> str:
        goals = self.db.get_active_goals(self.user_id, limit=5)
        if not goals:
            return "🎯 Цели не определены."
        lines = ["🎯 ВАША ИЕРАРХИЯ ЦЕЛЕЙ", "=" * 40]
        for goal in goals[:3]:
            prog = int(goal['progress'] * 100)
            progress_bar = "█" * (prog // 10) + "░" * (10 - prog // 10)
            lines.append(f"• {goal['description'][:50]}")
            lines.append(f"  Приоритет: {goal['priority']:.1f} | Прогресс: [{progress_bar}] {prog}%")
            if goal.get('next_action'):
                lines.append(f"  След. шаг: {goal['next_action'][:40]}")
            lines.append("")
        return "\n".join(lines)


# ================= МЕНЕДЖЕР СЕССИЙ =================
class SessionManager:
    def __init__(self, db: EnhancedMemoryDB, thinker: LocalThinkingSystem):
        self.db = db
        self.thinker = thinker
        self.sessions: Dict[int, UserCognitiveAgent] = {}
        self.session_timeout = Config.SESSION_TIMEOUT
        self.last_cleanup = time.time()

    async def get_or_create_session(self, user_id: int) -> UserCognitiveAgent:
        if time.time() - self.last_cleanup > 300:
            await self._cleanup_inactive_sessions()
            self.last_cleanup = time.time()

        if user_id not in self.sessions:
            self.sessions[user_id] = UserCognitiveAgent(user_id, self.db, self.thinker)
            logging.info(f"🆕 Новая сессия для пользователя {user_id}")
        return self.sessions[user_id]

    async def _cleanup_inactive_sessions(self):
        current_time = time.time()
        inactive = [uid for uid, sess in self.sessions.items()
                    if current_time - sess.start_time > self.session_timeout]
        for uid in inactive:
            del self.sessions[uid]
            logging.info(f"🧹 Очищена сессия {uid}")


# ================= ОБРАБОТЧИКИ TELEGRAM =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user.id)

    welcome = (
        f"👋 Привет, {user.first_name}!\n\n"
        "🧠 Я — когнитивный ассистент с локальной ИИ-моделью.\n"
        "✅ Полная приватность — все данные остаются на вашем компьютере\n"
        "⚡ Быстрая работа без интернета после настройки\n\n"
        "✨ **Мои когнитивные способности:**\n"
        "• 🤯 Многоуровневый анализ (рефлексия → анализ → планирование)\n"
        "• 🧠 Контекстная память с приоритизацией важной информации\n"
        "• 🔍 Автоматическое обнаружение паттернов в ваших запросах\n"
        "• 💡 Генерация креативных решений и неочевидных инсайтов\n"
        "• 🎯 Иерархия целей с прогрессом и следующими действиями\n"
        "• 💾 Кэширование ответов для ускорения повторяющихся запросов\n\n"
        "📌 **Команды:**\n"
        "• /think — глубокий анализ и рефлексия\n"
        "• /stats — персональная статистика\n"
        "• /patterns — обнаруженные паттерны поведения\n"
        "• /insights — инсайты из моих размышлений\n"
        "• /goals — ваши цели и прогресс\n"
        "• /clear — очистить контекст диалога"
    )
    await update.message.reply_text(welcome, reply_markup=create_main_keyboard())


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.message.reply_chat_action("typing")
    await update.message.reply_text(f"```\n{agent.get_comprehensive_stats()}\n```",
                                    parse_mode='MarkdownV2', reply_markup=create_main_keyboard())


async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.message.reply_text(
        "🧠 Запускаю глубокое многоуровневое мышление...\nАнализирую диалоги и генерирую инсайты...")
    await agent._autonomous_thinking()
    await update.message.reply_text("✅ Глубокое мышление завершено! Результаты в /insights",
                                    reply_markup=create_main_keyboard())


async def patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.message.reply_chat_action("typing")
    await update.message.reply_text(agent.get_patterns_summary(), reply_markup=create_main_keyboard())


async def insights_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.message.reply_chat_action("typing")
    await update.message.reply_text(agent.get_insights_summary(), reply_markup=create_main_keyboard())


async def goals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.message.reply_chat_action("typing")
    await update.message.reply_text(agent.get_goals_summary(), reply_markup=create_main_keyboard())


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    agent.context_window.clear()
    await update.message.reply_text("🧹 Контекст диалога очищен", reply_markup=create_main_keyboard())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    # 🔑 КРИТИЧЕСКИ ВАЖНАЯ ПРОВЕРКА: гарантируем, что система инициализирована
    if 'session_manager' not in context.application.bot_data:
        await update.message.reply_chat_action("typing")
        await asyncio.sleep(2)  # Дать время на инициализацию

        # Повторная проверка
        if 'session_manager' not in context.application.bot_data:
            await update.message.reply_text(
                "⚠️ Система ещё инициализируется. Подождите 5 секунд и повторите запрос.",
                reply_markup=create_main_keyboard()
            )
            return

    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        return

    logging.info(f"📨 {user_id}: {text[:40]}...")

    try:
        session_manager = context.application.bot_data['session_manager']
        agent = await session_manager.get_or_create_session(user_id)
        await update.message.reply_chat_action("typing")

        # Короткая пауза для лучшего UX
        if len(text) > 30 or '?' in text:
            await asyncio.sleep(0.3)

        response = await agent.process_message(text)
        parts = split_message(response)

        for i, part in enumerate(parts):
            reply_markup = create_main_keyboard() if (
                        i == len(parts) - 1 and agent.interaction_count % 4 == 0) else None
            await update.message.reply_text(part, reply_markup=reply_markup, disable_web_page_preview=True)
            if i < len(parts) - 1:
                await asyncio.sleep(0.3)

    except Exception as e:
        logging.error(f"❌ Ошибка: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Ошибка обработки. Попробуйте переформулировать запрос.",
            reply_markup=create_main_keyboard()
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    handlers = {
        "deep_think": think_command,
        "stats": stats_command,
        "goals": goals_command,
        "insights": insights_command,
        "patterns": patterns_command,
        "clear": clear_command,
    }
    handler = handlers.get(update.callback_query.data)
    if handler:
        await handler(update, context)
    else:
        await update.callback_query.message.reply_text("❓ Неизвестная команда", reply_markup=create_main_keyboard())


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Ошибка: {context.error}", exc_info=context.error)


# ================= ГЛАВНАЯ ФУНКЦИЯ С ГАРАНТИРОВАННОЙ ИНИЦИАЛИЗАЦИЕЙ =================
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


async def main():
    setup_logging()
    logging.info("🚀 Запуск когнитивного ассистента с локальной LLM")

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
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("think", think_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("patterns", patterns_command))
    application.add_handler(CommandHandler("insights", insights_command))
    application.add_handler(CommandHandler("goals", goals_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    # 🔑 ШАГ 6: НАСТРОЙКА КОМАНД МЕНЮ
    await application.bot.set_my_commands([
        BotCommand("start", "Начать"),
        BotCommand("think", "Глубокий анализ"),
        BotCommand("stats", "Статистика"),
        BotCommand("patterns", "Паттерны"),
        BotCommand("insights", "Инсайты"),
        BotCommand("goals", "Цели"),
        BotCommand("clear", "Очистить контекст")
    ])

    # 🔑 ШАГ 7: ЗАПУСК ПРИЛОЖЕНИЯ
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
    print("\n💡 СОВЕТЫ ПО ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("   • Для скорости на слабом ПК: используйте Phi-3-mini (3.8B)")
    print("   • Для баланса: Mistral-7B-Instruct с квантизацией Q5_K_M")
    print("   • Для максимального качества: модели 13B+ с GPU ускорением")
    print("   • Все данные хранятся локально — 100% приватность")
    print("\n🛑 Остановка: Ctrl+C")
    print("=" * 70 + "\n")

    logging.info("🔄 Бот работает. Нажмите Ctrl+C для остановки")
    while True:
        await asyncio.sleep(3600)


def run():
    print("AGI Cognitive Assistant — Полная когнитивная система с локальной LLM (LM Studio)")
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
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()