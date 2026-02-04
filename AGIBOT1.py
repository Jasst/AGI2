# coding: utf-8
"""
AGI24_Bot.py — ЕДИНЫЙ ФАЙЛ: КОГНИТИВНАЯ СИСТЕМА + TELEGRAM БОТ
Исправленная версия для Python 3.13
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any, Set, Tuple
from datetime import datetime
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

# Обновленные импорты для Telegram Bot API
try:
    import telegram
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters
    )
    from telegram.error import TelegramError

    print("✅ Библиотека python-telegram-bot загружена успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта telegram: {e}")
    print("📦 Установите последнюю версию: pip install python-telegram-bot")
    sys.exit(1)

# Проверка версии Python
if sys.version_info >= (3, 13):
    print("⚠️ ВНИМАНИЕ: Версия Python 3.13")
    print("📌 Рекомендуется использовать Python 3.10-3.12 для полной совместимости")


# ================= ОБЪЕДИНЁННАЯ КОНФИГУРАЦИЯ =================

class Config:
    """Унифицированная конфигурация системы и бота"""
    ROOT = Path("./cognitive_system_v30")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    TIMEOUT = 300
    MAX_TOKENS = 8000

    # Когнитивные параметры
    REFLECTION_INTERVAL = 3
    DEEP_THINKING_THRESHOLD = 0.7
    CONTEXT_WINDOW_SIZE = 10
    MEMORY_DECAY_RATE = 0.05

    # Параметры бота
    MAX_MESSAGE_LENGTH = 4096
    MAX_RESPONSE_CHUNKS = 5
    TYPING_DELAY = 1.5
    REQUEST_TIMEOUT = 30

    @classmethod
    def get_api_key(cls):
        """Получение API ключа OpenRouter"""
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
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
                    f.write(f'\nOPENROUTER_API_KEY="{key}"')
                print("✅ Ключ сохранен в файл .env")
                return key
            except Exception as e:
                print(f"⚠️ Не удалось сохранить ключ: {e}")
                return key

        raise ValueError("API ключ OpenRouter не найден")

    @classmethod
    def get_telegram_token(cls):
        """Получение токена Telegram бота"""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
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
                env_exists = Path(".env").exists()
                with open(".env", "a" if env_exists else "w", encoding="utf-8") as f:
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

    unigram_sim = len(words1.intersection(words2)) / max(len(words1), len(words2))

    bigrams1 = get_ngrams(text1, 2)
    bigrams2 = get_ngrams(text2, 2)

    if bigrams1 and bigrams2:
        bigram_sim = len(bigrams1.intersection(bigrams2)) / max(len(bigrams1), len(bigrams2), 1)
    else:
        bigram_sim = 0.0

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
        'emotions': len(re.findall(r'\b(хорошо|плохо|отлично|ужасно|интересно|скучно|рад|грустно)\b', text_lower)),
        'imperatives': len(re.findall(r'\b(сделай|создай|найди|покажи|расскажи|объясни)\b', text_lower)),
        'has_question': '?' in text,
        'sentiment': analyze_sentiment(text)
    }
    return features


def analyze_sentiment(text: str) -> float:
    """Простой анализ тональности (-1 до 1)"""
    positive = ['хорошо', 'отлично', 'прекрасно', 'замечательно', 'классно', 'супер', 'рад', 'счастлив']
    negative = ['плохо', 'ужасно', 'отвратительно', 'кошмар', 'провал', 'грустно', 'ненавижу', 'злой']

    text_lower = text.lower()
    pos_count = sum(1 for word in positive if word in text_lower)
    neg_count = sum(1 for word in negative if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


# ================= РАСШИРЕННАЯ БАЗА ДАННЫХ =================

class EnhancedMemoryDB:
    """Продвинутая база данных с поддержкой контекста и связей"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user_input TEXT NOT NULL,
                    system_response TEXT NOT NULL,
                    context TEXT,
                    emotion TEXT DEFAULT 'neutral',
                    category TEXT,
                    importance REAL DEFAULT 0.5,
                    complexity REAL DEFAULT 0.5,
                    satisfaction REAL DEFAULT 0.5,
                    tokens_used INTEGER DEFAULT 0,
                    user_id INTEGER DEFAULT 0
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    UNIQUE(key, value)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS thoughts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_goal_id INTEGER,
                    created_at REAL NOT NULL,
                    description TEXT NOT NULL,
                    priority REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'active',
                    progress REAL DEFAULT 0.0,
                    deadline REAL,
                    next_action TEXT,
                    success_criteria TEXT,
                    learned_lessons TEXT,
                    FOREIGN KEY (parent_goal_id) REFERENCES goals(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    occurrences INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    success_rate REAL DEFAULT 0.5
                )
            ''')

            conn.commit()

    def add_interaction(self, user_input: str, system_response: str, user_id: int = 0, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions
                (timestamp, user_input, system_response, context, emotion, category,
                importance, complexity, satisfaction, tokens_used, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                user_input[:5000],
                system_response[:5000],
                kwargs.get('context', '')[:1000],
                kwargs.get('emotion', 'neutral'),
                kwargs.get('category', ''),
                kwargs.get('importance', 0.5),
                kwargs.get('complexity', 0.5),
                kwargs.get('satisfaction', 0.5),
                kwargs.get('tokens_used', 0),
                user_id
            ))
            conn.commit()
            return cursor.lastrowid

    def get_contextual_interactions(self, query: str, limit: int = 5, user_id: int = 0) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if user_id:
                cursor.execute('''
                    SELECT * FROM interactions
                    WHERE user_id = ? OR user_id = 0
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, limit * 3))
            else:
                cursor.execute('''
                    SELECT * FROM interactions
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit * 3,))

            all_interactions = [dict(row) for row in cursor.fetchall()]

            scored = []
            for interaction in all_interactions:
                relevance = calculate_text_similarity(
                    query,
                    interaction['user_input'] + ' ' + interaction['system_response']
                )
                recency = 1.0 - (time.time() - interaction['timestamp']) / (30 * 24 * 3600)
                recency = max(0, min(1, recency))
                score = 0.6 * relevance + 0.3 * interaction['importance'] + 0.1 * recency
                scored.append((score, interaction))

            scored.sort(reverse=True, key=lambda x: x[0])
            return [item[1] for item in scored[:limit]]

    def add_fact(self, key: str, value: str, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT id, usage_count FROM facts WHERE key = ? AND value = ?', (key, value))
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
                    (key, value, category, confidence, importance, created_at, last_used,
                    usage_count, decay_factor, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    key[:500],
                    value[:500],
                    kwargs.get('category', ''),
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(),
                    time.time(),
                    1,
                    1.0,
                    kwargs.get('source', 'user')
                ))
            conn.commit()

    def get_relevant_facts(self, query: str, limit: int = 5) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM facts
                WHERE confidence > 0.3 AND decay_factor > 0.1
                ORDER BY importance DESC, usage_count DESC
                LIMIT ?
            ''', (limit * 2,))

            all_facts = [dict(row) for row in cursor.fetchall()]

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
            return [item[1] for item in scored[:limit]]

    def add_thought(self, thought_type: str, content: str, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO thoughts
                (timestamp, thought_type, content, trigger, importance, depth_level, confidence, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                thought_type,
                content[:2000],
                kwargs.get('trigger', ''),
                kwargs.get('importance', 0.5),
                kwargs.get('depth_level', 1),
                kwargs.get('confidence', 0.7),
                kwargs.get('outcome', '')
            ))
            conn.commit()

    def get_thought_insights(self, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT thought_type, COUNT(*) as count, AVG(importance) as avg_importance
                FROM thoughts
                WHERE timestamp > ?
                GROUP BY thought_type
                ORDER BY count DESC
                LIMIT ?
            ''', (time.time() - 7 * 86400, limit))
            return [dict(row) for row in cursor.fetchall()]

    def add_pattern(self, pattern_type: str, description: str, confidence: float = 0.5):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id FROM patterns WHERE pattern_type = ? AND description = ?
            ''', (pattern_type, description))
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
                    (pattern_type, description, occurrences, confidence, created_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (pattern_type, description, 1, confidence, time.time(), time.time()))
            conn.commit()

    def get_patterns(self, min_confidence: float = 0.6, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM patterns
                WHERE confidence >= ?
                ORDER BY occurrences DESC, confidence DESC
                LIMIT ?
            ''', (min_confidence, limit))
            return [dict(row) for row in cursor.fetchall()]

    def add_goal(self, description: str, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO goals
                (parent_goal_id, created_at, description, priority, status, progress,
                deadline, next_action, success_criteria)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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

    def get_goal_hierarchy(self, parent_id: Optional[int] = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if parent_id is None:
                cursor.execute('''
                    SELECT * FROM goals
                    WHERE parent_goal_id IS NULL AND status = 'active'
                    ORDER BY priority DESC
                ''')
            else:
                cursor.execute('''
                    SELECT * FROM goals
                    WHERE parent_goal_id = ? AND status = 'active'
                    ORDER BY priority DESC
                ''', (parent_id,))

            return [dict(row) for row in cursor.fetchall()]


# ================= СИСТЕМА МЫШЛЕНИЯ =================

class EnhancedThinkingSystem:
    """Продвинутая многоуровневая система мышления"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit = 2.0
        self.last_request_time = 0
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_max_size = 500
        self.reasoning_history = deque(maxlen=100)

    async def _wait_for_rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def multi_level_thinking(self, context: str, depth: int = 2) -> Dict[str, str]:
        thoughts = {}
        thinking_layers = [
            ('surface', 'Что очевидно? Какие факты прямо указаны?', 0.3),
            ('analytical', 'Какие связи и паттерны можно увидеть? Что неявно?', 0.5),
            ('strategic', 'Какие долгосрочные последствия? Какие стратегии можно применить?', 0.7),
            ('creative', 'Какие неожиданные решения возможны? Какие инновационные подходы?', 0.9)
        ]

        depth = min(depth, 4)

        for layer_name, prompt, temperature in thinking_layers[:depth]:
            thought = await self.generate_thought_with_prompt(
                f"{prompt}\n\nКонтекст: {context}",
                temperature=temperature
            )
            if thought and len(thought) > 10:
                thoughts[layer_name] = thought

        return thoughts

    async def generate_thought_with_prompt(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        system_prompt = """Ты — продвинутая когнитивная система с глубоким аналитическим мышлением.
Твои сильные стороны:
- Многоуровневый анализ информации
- Обнаружение скрытых паттернов и связей
- Креативное решение сложных проблем
- Критическое мышление и оценка
- Предсказание последствий и планирование

Отвечай кратко, но содержательно. Фокусируйся на ключевых инсайтах, а не на очевидных вещах.
Будь конкретным и практичным в своих выводах."""

        response = await self.call_llm(system_prompt, prompt, temperature)
        if response and len(response) > 10:
            self.reasoning_history.append({
                'timestamp': time.time(),
                'prompt': prompt[:200],
                'response': response[:300],
                'temperature': temperature
            })
            return response
        return None

    async def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        cache_key = hashlib.md5(
            f"{system_prompt[:100]}{user_prompt[:200]}{temperature}".encode()
        ).hexdigest()

        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1
        await self._wait_for_rate_limit()

        try:
            import aiohttp

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AGI24-Bot",
                "X-Title": "AGI24 Cognitive System"
            }

            payload = {
                "model": Config.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": Config.MAX_TOKENS,
                "top_p": 0.95,
                "stream": False
            }

            timeout = aiohttp.ClientTimeout(total=Config.TIMEOUT)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                        Config.OPENROUTER_URL,
                        headers=headers,
                        json=payload,
                        timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"].strip()

                            self.cache[cache_key] = content

                            if len(self.cache) > self.cache_max_size:
                                keys_to_remove = list(self.cache.keys())[:100]
                                for key in keys_to_remove:
                                    del self.cache[key]

                            return content
                        else:
                            return "⚠️ Неожиданный формат ответа от API"
                    else:
                        error_text = await response.text()
                        return f"⚠️ Ошибка API ({response.status}): {error_text[:200]}"

        except ImportError:
            return "⚠️ Ошибка: библиотека aiohttp не установлена. Установите: pip install aiohttp"
        except Exception as e:
            return f"⚠️ Неожиданная ошибка: {str(e)[:100]}"


# ================= АВТОНОМНЫЙ АГЕНТ =================

class EnhancedAutonomousAgent:
    """Продвинутый автономный когнитивный агент"""

    def __init__(self):
        print("🤖 Инициализация когнитивного агента...")

        try:
            self.api_key = Config.get_api_key()
            print("✅ API ключ получен")
        except Exception as e:
            print(f"❌ Ошибка получения API ключа: {e}")
            raise

        self.db = EnhancedMemoryDB(Config.DB_PATH)
        self.thinker = EnhancedThinkingSystem(self.api_key)

        self.interaction_count = 0
        self.deep_thoughts_count = 0
        self.patterns_found = 0
        self.start_time = time.time()
        self.context_window = deque(maxlen=Config.CONTEXT_WINDOW_SIZE)
        self.active_tasks = []

        self.self_assessment = {
            'knowledge_gaps': [],
            'strong_areas': [],
            'improvement_areas': [],
            'avg_response_time': 0.0,
            'success_rate': 1.0
        }

        self._init_system()
        print("✅ Когнитивный агент инициализирован успешно")

    def _init_system(self):
        existing_goals = self.db.get_goal_hierarchy()
        if not existing_goals:
            main_goal = self.db.add_goal(
                "Быть максимально полезным и эффективным когнитивным помощником",
                priority=1.0,
                success_criteria="Высокий уровень удовлетворённости пользователей и качественные ответы"
            )
            self.db.add_goal(
                "Глубоко понимать и анализировать запросы пользователей",
                parent_goal_id=main_goal,
                priority=0.9,
                next_action="Анализ контекста и извлечение ключевых аспектов"
            )
            self.db.add_goal(
                "Непрерывно обучаться и адаптироваться",
                parent_goal_id=main_goal,
                priority=0.85,
                next_action="Обнаружение паттернов и обновление знаний"
            )

    async def process_input(self, user_input: str, user_id: int = 0) -> str:
        start_time = time.time()
        self.interaction_count += 1

        self.context_window.append({
            'type': 'user',
            'content': user_input,
            'timestamp': time.time(),
            'user_id': user_id
        })

        command_response = self._handle_command(user_input)
        if command_response:
            return command_response

        features = extract_semantic_features(user_input)
        complexity = features['complexity']
        importance = self._calculate_importance(user_input, features)

        self._extract_and_store_information(user_input, importance)
        response = await self._generate_contextual_response(
            user_input, features, complexity, importance, user_id
        )

        interaction_id = self.db.add_interaction(
            user_input=user_input,
            system_response=response,
            user_id=user_id,
            context=self._get_context_summary(),
            category=self._categorize_input(user_input, features),
            importance=importance,
            complexity=complexity,
            satisfaction=self._calculate_satisfaction(response, features),
            tokens_used=len(response.split())
        )

        self.context_window.append({
            'type': 'assistant',
            'content': response[:200],
            'timestamp': time.time(),
            'interaction_id': interaction_id
        })

        await self._detect_patterns(user_id)

        if self.interaction_count % Config.REFLECTION_INTERVAL == 0:
            asyncio.create_task(self._deep_autonomous_thinking())

        duration = time.time() - start_time
        self.self_assessment['avg_response_time'] = (
                self.self_assessment['avg_response_time'] * 0.9 + duration * 0.1
        )

        return response

    def _calculate_importance(self, text: str, features: Dict) -> float:
        importance = 0.5

        important_keywords = ['важно', 'срочно', 'критично', 'обязательно', 'жизненно', 'решающий']
        if any(word in text.lower() for word in important_keywords):
            importance += 0.3

        importance += min(0.2, features['question_words'] * 0.05)
        importance += min(0.15, features['imperatives'] * 0.05)
        importance += features['complexity'] * 0.15

        if abs(features['sentiment']) > 0.5:
            importance += 0.1

        return min(1.0, max(0.1, importance))

    def _calculate_satisfaction(self, response: str, features: Dict) -> float:
        satisfaction = 0.7

        if len(response.split()) > 20:
            satisfaction += 0.1

        if any(word in response.lower() for word in ['конкретно', 'например', 'во-первых', 'таким образом']):
            satisfaction += 0.1

        if '⚠️' not in response and 'Ошибка' not in response:
            satisfaction += 0.1

        return min(1.0, satisfaction)

    def _extract_and_store_information(self, text: str, importance: float):
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers:
            if len(num) < 10:
                self.db.add_fact('число', num, category='данные', importance=importance * 0.5)

        names = re.findall(r'\b[А-ЯЁ][а-яё]+\b', text)
        for name in names:
            if len(name) > 2 and name not in ['Это', 'Что', 'Как', 'Почему']:
                self.db.add_fact('имя', name, category='персона', importance=importance * 0.7)

        definition_patterns = [
            (r'(\w+)\s+(?:это|равно|составляет)\s+([^.,]+)', 'определение'),
            (r'запомни[,:]\s*(.+)', 'важная_информация'),
            (r'(\w+)\s*=\s*([^,]+)', 'равенство'),
            (r'([А-Яа-яёЁ\s]+)\s+—\s+([^.,]+)', 'определение')
        ]

        for pattern, category in definition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    key, value = match
                    key = key.strip()
                    value = value.strip()
                    if key and value and len(key) < 100 and len(value) < 500:
                        self.db.add_fact(
                            key,
                            value,
                            category=category,
                            importance=importance,
                            source='user_input'
                        )

    async def _generate_contextual_response(
            self, user_input: str, features: Dict, complexity: float,
            importance: float, user_id: int = 0
    ) -> str:
        relevant_interactions = self.db.get_contextual_interactions(
            user_input, limit=3, user_id=user_id
        )
        relevant_facts = self.db.get_relevant_facts(user_input, limit=5)
        active_goals = self.db.get_goal_hierarchy()

        context_parts = []

        if relevant_interactions:
            context_parts.append("📜 Релевантная история:")
            for interaction in relevant_interactions[:2]:
                context_parts.append(f"  П: {interaction['user_input'][:60]}...")
                context_parts.append(f"  Я: {interaction['system_response'][:60]}...")

        if relevant_facts:
            context_parts.append("\n📚 Релевантные факты:")
            for fact in relevant_facts[:4]:
                conf_stars = "★" * int(fact['confidence'] * 5)
                context_parts.append(f"  • {fact['key']}: {fact['value'][:50]} [{conf_stars}]")

        if active_goals:
            context_parts.append("\n🎯 Текущие цели:")
            for goal in active_goals[:2]:
                context_parts.append(f"  • {goal['description'][:50]}")

        patterns = self.db.get_patterns(min_confidence=0.7, limit=2)
        if patterns:
            context_parts.append("\n🔍 Обнаруженные паттерны:")
            for pattern in patterns:
                context_parts.append(f"  • {pattern['description'][:60]}")

        context = "\n".join(context_parts) if context_parts else "Нет дополнительного контекста"

        needs_deep_thinking = (
                complexity > Config.DEEP_THINKING_THRESHOLD or
                importance > 0.7 or
                features['question_words'] > 2 or
                features['has_question']
        )

        if needs_deep_thinking:
            deep_thoughts = await self.thinker.multi_level_thinking(
                f"Запрос пользователя: {user_input}\n\nКонтекст:\n{context}",
                depth=3
            )

            if deep_thoughts:
                synthesis_prompt = f"""На основе многоуровневого анализа ответь на запрос пользователя.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_input}

АНАЛИЗ:
{chr(10).join([f'{level.upper()}: {thought}' for level, thought in deep_thoughts.items()])}

КОНТЕКСТ:
{context}

СФОРМИРУЙ ЦЕЛЬНЫЙ, ГЛУБОКИЙ И ПРАКТИЧНЫЙ ОТВЕТ, КОТОРЫЙ:
1. Отвечает на все аспекты запроса
2. Использует инсайты из анализа
3. Учитывает контекст и историю
4. Предлагает конкретные действия или решения
5. Будет полезен и понятен пользователю"""

                system_prompt = """Ты — продвинутая когнитивная система. Синтезируй ответ на основе глубокого анализа. 
Будь точным, структурированным и полезным. Избегай общих фраз, фокусируйся на конкретике."""

                response = await self.thinker.call_llm(system_prompt, synthesis_prompt, temperature=0.7)
                self.deep_thoughts_count += 1
            else:
                response = await self._generate_standard_response(user_input, context)
        else:
            response = await self._generate_standard_response(user_input, context)

        return response.strip() if response else "Извините, не удалось сгенерировать ответ."

    async def _generate_standard_response(self, user_input: str, context: str) -> str:
        system_prompt = f"""Ты — интеллектуальный помощник с доступом к памяти и контексту.

КОНТЕКСТ:
{context}

ПРИНЦИПЫ ОТВЕТА:
- Используй факты из памяти при их наличии
- Будь точным и конкретным
- Отвечай дружелюбно и полезно
- Если не знаешь ответа — честно говори об этом
- Структурируй сложные ответы
- Избегай излишней техничности, если пользователь не специалист

Отвечай на русском языке, если пользователь пишет по-русски."""

        return await self.thinker.call_llm(system_prompt, user_input, temperature=0.6)

    async def _detect_patterns(self, user_id: int = 0):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT category, COUNT(*) as count
                FROM interactions
                WHERE timestamp > ? AND (user_id = ? OR user_id = 0) AND category IS NOT NULL
                GROUP BY category
                HAVING count > 2
            ''', (time.time() - 7 * 86400, user_id))

            categories = cursor.fetchall()
            for category, count in categories:
                if category:
                    self.db.add_pattern(
                        'frequent_category',
                        f"Пользователь часто спрашивает про '{category}'",
                        confidence=min(1.0, count / 10)
                    )

            cursor.execute('''
                SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour, COUNT(*) as count
                FROM interactions
                WHERE timestamp > ? AND (user_id = ? OR user_id = 0)
                GROUP BY hour
                HAVING count > 3
            ''', (time.time() - 7 * 86400, user_id))

            time_patterns = cursor.fetchall()
            for hour, count in time_patterns:
                self.db.add_pattern(
                    'time_preference',
                    f"Активность в {hour}:00 (встречалось {count} раз)",
                    confidence=min(1.0, count / 20)
                )

            self.patterns_found = len(categories) + len(time_patterns)

    async def _deep_autonomous_thinking(self):
        print("\n💭 [Запуск глубокого автономного мышления...]")

        try:
            recent = self.db.get_contextual_interactions("анализ рефлексия", limit=7)
            patterns = self.db.get_patterns(min_confidence=0.6, limit=5)
            insights = self.db.get_thought_insights(limit=5)

            if not recent:
                print("  💭 Недостаточно данных для глубокого анализа")
                return

            context_lines = [
                "📊 ПОСЛЕДНИЕ ВЗАИМОДЕЙСТВИЯ:",
                *[f"- {i['user_input'][:50]}..." for i in recent[:3]],
                "\n🔍 ОБНАРУЖЕННЫЕ ПАТТЕРНЫ:",
                *[f"- {p['description']}" for p in patterns[:3]],
                "\n💡 ИНСАЙТЫ ИЗ ПРЕДЫДУЩИХ РАЗМЫШЛЕНИЙ:",
                *[f"- {t['thought_type']}: {t['count']} случаев" for t in insights[:3]]
            ]
            context = "\n".join(context_lines)

            thoughts = await self.thinker.multi_level_thinking(context, depth=4)

            for thought_type, content in thoughts.items():
                if content and len(content) > 20:
                    self.db.add_thought(
                        thought_type=thought_type,
                        content=content,
                        trigger='autonomous_deep_thinking',
                        importance=0.8,
                        depth_level=4,
                        confidence=0.7,
                        outcome='reflection_completed'
                    )
                    print(f"  💡 [{thought_type.upper()}] {content[:80]}...")

            await self._update_self_assessment()

            print("✅ Глубокое мышление завершено")

        except Exception as e:
            print(f"⚠️ Ошибка в глубоком мышлении: {e}")

    async def _update_self_assessment(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT AVG(satisfaction) FROM interactions
                WHERE timestamp > ?
            ''', (time.time() - 86400,))
            avg_satisfaction = cursor.fetchone()[0] or 0.5

            cursor.execute('''
                SELECT category, AVG(satisfaction) as avg_sat
                FROM interactions
                WHERE timestamp > ? AND category IS NOT NULL
                GROUP BY category
                HAVING avg_sat < 0.5 AND COUNT(*) > 2
            ''', (time.time() - 7 * 86400,))
            weak_categories = [row[0] for row in cursor.fetchall()]

            cursor.execute('''
                SELECT category, AVG(satisfaction) as avg_sat
                FROM interactions
                WHERE timestamp > ? AND category IS NOT NULL
                GROUP BY category
                HAVING avg_sat > 0.7 AND COUNT(*) > 2
            ''', (time.time() - 7 * 86400,))
            strong_categories = [row[0] for row in cursor.fetchall()]

            self.self_assessment.update({
                'avg_satisfaction': avg_satisfaction,
                'improvement_areas': weak_categories[:5],
                'strong_areas': strong_categories[:5],
                'patterns_discovered': self.patterns_found,
                'deep_thoughts': self.deep_thoughts_count,
                'total_interactions': self.interaction_count
            })

    def _handle_command(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()

        command_map = {
            'думай глубоко': "🧠 Запускаю глубокое многоуровневое мышление...",
            'глубокое мышление': "🧠 Запускаю глубокое многоуровневое мышление...",
            'анализ': None,
            'паттерны': None,
            'инсайты': None,
            'цели': None,
            'статистика': None,
            'факты': None,
            'выход': "👋 До новых встреч!",
            'exit': "👋 Goodbye!",
            'quit': "👋 Goodbye!"
        }

        if text_lower in command_map:
            response = command_map[text_lower]
            if response:
                if 'глубокое' in text_lower:
                    asyncio.create_task(self._deep_autonomous_thinking())
                return response

        return None

    def _get_comprehensive_analysis(self) -> str:
        lines = ["🔍 КОМПЛЕКСНЫЙ АНАЛИЗ СИСТЕМЫ", "=" * 60]

        lines.append(f"\n📊 Время работы: {self._format_uptime()}")
        lines.append(f"Всего взаимодействий: {self.interaction_count}")
        lines.append(f"Глубоких мыслей: {self.deep_thoughts_count}")
        lines.append(f"Паттернов обнаружено: {self.patterns_found}")

        lines.append(f"\n📈 Самооценка:")
        lines.append(f"  Средняя удовлетворённость: {self.self_assessment.get('avg_satisfaction', 0.5):.2f}")
        lines.append(f"  Среднее время ответа: {self.self_assessment.get('avg_response_time', 0):.2f}с")

        if self.self_assessment.get('strong_areas'):
            lines.append("\n✅ Сильные области:")
            for area in self.self_assessment['strong_areas'][:3]:
                lines.append(f"  • {area}")

        if self.self_assessment.get('improvement_areas'):
            lines.append("\n⚠️ Области для улучшения:")
            for area in self.self_assessment['improvement_areas'][:3]:
                lines.append(f"  • {area}")

        return "\n".join(lines)

    def _format_patterns(self) -> str:
        patterns = self.db.get_patterns(min_confidence=0.5, limit=15)
        if not patterns:
            return "🔍 Паттернов пока не обнаружено."

        lines = ["🔍 ОБНАРУЖЕННЫЕ ПАТТЕРНЫ", "=" * 60]

        by_type = defaultdict(list)
        for p in patterns:
            by_type[p['pattern_type']].append(p)

        for ptype, plist in by_type.items():
            lines.append(f"\n📌 {ptype.upper().replace('_', ' ')}:")
            for p in plist:
                conf_bar = "█" * int(p['confidence'] * 10)
                lines.append(f"  • {p['description']}")
                lines.append(
                    f"    Встречалось: {p['occurrences']} раз | Уверенность: [{conf_bar}] {p['confidence']:.2f}")

        return "\n".join(lines)

    def _format_insights(self) -> str:
        insights = self.db.get_thought_insights(limit=10)
        if not insights:
            return "💡 Инсайтов пока нет."

        lines = ["💡 ИНСАЙТЫ ИЗ МЫСЛЕЙ", "=" * 60]

        for insight in insights:
            lines.append(f"\n🧠 {insight['thought_type'].upper().replace('_', ' ')}:")
            lines.append(f"  Количество: {insight['count']}")
            lines.append(f"  Средняя важность: {insight['avg_importance']:.2f}")

        return "\n".join(lines)

    def _format_goal_hierarchy(self) -> str:
        main_goals = self.db.get_goal_hierarchy(parent_id=None)
        if not main_goals:
            return "🎯 Нет активных целей."

        lines = ["🎯 ИЕРАРХИЯ ЦЕЛЕЙ", "=" * 60]

        for goal in main_goals:
            progress_bar = "█" * int(goal['progress'] * 10) + "░" * (10 - int(goal['progress'] * 10))
            lines.append(f"\n📍 {goal['description']}")
            lines.append(f"  Приоритет: {goal['priority']:.2f} | Прогресс: [{progress_bar}] {goal['progress']:.0%}")
            lines.append(f"  Статус: {goal['status']}")

            if goal['next_action']:
                lines.append(f"  Следующий шаг: {goal['next_action']}")

            subgoals = self.db.get_goal_hierarchy(parent_id=goal['id'])
            if subgoals:
                lines.append("  Подцели:")
                for sub in subgoals[:3]:
                    sub_bar = "█" * int(sub['progress'] * 5)
                    lines.append(f"    • {sub['description'][:50]} [{sub_bar}]")

        return "\n".join(lines)

    def _get_comprehensive_stats(self) -> str:
        lines = ["📊 ПОЛНАЯ СТАТИСТИКА СИСТЕМЫ", "=" * 70]

        lines.append(f"\n⏱️ {self._format_uptime()}")
        lines.append(f"Взаимодействий: {self.interaction_count}")
        lines.append(f"Глубоких мыслей: {self.deep_thoughts_count}")
        lines.append(f"Паттернов найдено: {self.patterns_found}")

        lines.append("\n📈 Самооценка:")
        lines.append(f"  Средняя удовлетворённость: {self.self_assessment.get('avg_satisfaction', 0.5):.2f}")
        lines.append(f"  Среднее время ответа: {self.self_assessment.get('avg_response_time', 0):.2f}с")
        lines.append(f"  Сильные области: {len(self.self_assessment.get('strong_areas', []))}")
        lines.append(f"  Области улучшения: {len(self.self_assessment.get('improvement_areas', []))}")

        lines.append("\n⚙️ Конфигурация:")
        lines.append(f"  Модель: {Config.MODEL}")
        lines.append(f"  Размер контекстного окна: {Config.CONTEXT_WINDOW_SIZE}")
        lines.append(f"  Интервал рефлексии: {Config.REFLECTION_INTERVAL}")
        lines.append(f"  Порог глубокого мышления: {Config.DEEP_THINKING_THRESHOLD}")

        return "\n".join(lines)

    def _format_uptime(self) -> str:
        uptime = time.time() - self.start_time
        days = uptime // 86400
        hours = (uptime % 86400) // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60

        parts = []
        if days > 0:
            parts.append(f"{int(days)}д")
        if hours > 0:
            parts.append(f"{int(hours)}ч")
        if minutes > 0:
            parts.append(f"{int(minutes)}м")
        parts.append(f"{int(seconds)}с")

        return "Время работы: " + " ".join(parts)

    def _get_context_summary(self) -> str:
        if not self.context_window:
            return ""

        summary = []
        for item in list(self.context_window)[-4:]:
            prefix = "П:" if item['type'] == 'user' else "Я:"
            content = item['content']
            if len(content) > 50:
                content = content[:47] + "..."
            summary.append(f"{prefix} {content}")

        return "\n".join(summary)

    def _categorize_input(self, text: str, features: Dict) -> str:
        text_lower = text.lower()

        categories = {
            'математика': ['сколько', 'посчитай', 'вычисли', 'сумма', 'разность', 'процент', 'равно'],
            'память': ['запомни', 'сохрани', 'напомни', 'записывай', 'не забывай'],
            'анализ': ['проанализируй', 'разбери', 'оцени', 'сравни', 'исследуй'],
            'творчество': ['придумай', 'создай', 'сочини', 'напиши', 'генерация'],
            'планирование': ['план', 'распиши', 'как достичь', 'стратегия', 'расписание'],
            'объяснение': ['почему', 'как', 'зачем', 'объясни', 'расскажи', 'что такое'],
            'поиск': ['найди', 'покажи', 'где', 'ищи', 'найти', 'поиск'],
            'техническая помощь': ['помоги', 'не работает', 'ошибка', 'проблема', 'исправь'],
            'развлечение': ['шутка', 'расскажи историю', 'развлеки', 'игра'],
            'диалог': ['привет', 'как дела', 'спасибо', 'пока']
        }

        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        if features['has_question']:
            return 'вопрос'
        elif features['imperatives'] > 0:
            return 'команда'
        elif features['emotions'] > 0:
            return 'эмоциональный'

        return 'общий'


# ================= ХРАНИЛИЩЕ СЕССИЙ БОТА =================

class UserSessionManager:
    """Управление пользовательскими сессиями - ИСПРАВЛЕННАЯ ВЕРСИЯ"""

    def __init__(self):
        self.sessions: Dict[int, Dict] = {}
        self.global_agent: Optional[EnhancedAutonomousAgent] = None
        self.session_timeout = 3600  # 1 час

        print("✅ Менеджер сессий инициализирован (без фоновых задач)")

    async def get_or_create_session(self, user_id: int) -> Dict:
        """Получение или создание сессии пользователя"""
        now = time.time()

        if user_id not in self.sessions:
            print(f"🆕 Создание новой сессии для пользователя {user_id}")
            try:
                if self.global_agent is None:
                    self.global_agent = EnhancedAutonomousAgent()

                self.sessions[user_id] = {
                    'agent': self.global_agent,
                    'created_at': datetime.now(),
                    'last_activity': datetime.now(),
                    'message_count': 0,
                    'user_id': user_id,
                    'last_timestamp': now
                }
            except Exception as e:
                print(f"❌ Ошибка создания сессии для {user_id}: {e}")
                raise
        else:
            self.sessions[user_id]['last_activity'] = datetime.now()
            self.sessions[user_id]['last_timestamp'] = now

        return self.sessions[user_id]

    def get_stats(self) -> Dict:
        """Получение статистики по сессиям"""
        now = time.time()
        active_sessions = 0
        total_messages = 0

        for session in self.sessions.values():
            if now - session['last_timestamp'] < self.session_timeout:
                active_sessions += 1
                total_messages += session['message_count']

        return {
            'total_users': len(self.sessions),
            'active_users': active_sessions,
            'total_messages': total_messages,
            'session_timeout': self.session_timeout
        }

    async def cleanup_inactive_sessions(self):
        """Очистка неактивных сессий (вызывается вручную)"""
        now = time.time()
        inactive_users = []

        for user_id, session in self.sessions.items():
            if now - session['last_timestamp'] > self.session_timeout:
                inactive_users.append(user_id)

        for user_id in inactive_users:
            print(f"🗑️ Удаление неактивной сессии пользователя {user_id}")
            del self.sessions[user_id]

        if inactive_users:
            print(f"✅ Удалено {len(inactive_users)} неактивных сессий")


# Глобальный менеджер сессий - ИСПРАВЛЕННЫЙ
session_manager = UserSessionManager()


# ================= УТИЛИТЫ БОТА =================

def split_message(text: str, max_length: int = Config.MAX_MESSAGE_LENGTH) -> list:
    """Разбивает длинное сообщение на части"""
    if len(text) <= max_length:
        return [text]

    parts = []
    current_part = ""

    paragraphs = text.split('\n\n')

    for para in paragraphs:
        if len(current_part) + len(para) + 2 <= max_length:
            if current_part:
                current_part += '\n\n' + para
            else:
                current_part = para
        else:
            if current_part:
                parts.append(current_part)

            if len(para) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_sentence = ""

                for sentence in sentences:
                    if len(current_sentence) + len(sentence) + 1 <= max_length:
                        if current_sentence:
                            current_sentence += ' ' + sentence
                        else:
                            current_sentence = sentence
                    else:
                        if current_sentence:
                            parts.append(current_sentence)
                        current_sentence = sentence

                if current_sentence:
                    current_part = current_sentence
            else:
                current_part = para

    if current_part:
        parts.append(current_part)

    if len(parts) > Config.MAX_RESPONSE_CHUNKS:
        parts = parts[:Config.MAX_RESPONSE_CHUNKS]
        parts.append("\n\n📝 *Сообщение слишком длинное, показана только часть*")

    return parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    """Создание главной клавиатуры"""
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
            InlineKeyboardButton("📚 Факты", callback_data="facts"),
            InlineKeyboardButton("❓ Помощь", callback_data="help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= ОБРАБОТЧИКИ КОМАНД =================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id

    print(f"👋 Новый пользователь: {user.first_name} (ID: {user_id})")

    try:
        await session_manager.get_or_create_session(user_id)
    except Exception as e:
        await update.message.reply_text(
            f"⚠️ Ошибка инициализации сессии: {str(e)[:100]}\n\n"
            "Попробуйте еще раз через несколько секунд."
        )
        return

    welcome_text = f"""👋 Привет, {user.first_name}!

🧠 Я — **AGI24 Cognitive System** — продвинутый когнитивный агент:

✨ **Мои способности:**
• 🤯 Многоуровневое аналитическое мышление
• 🧠 Контекстная память и обучение
• 🔍 Обнаружение скрытых паттернов
• 💡 Креативное решение сложных задач
• 📊 Предсказательный анализ и планирование

💬 **Просто напиши мне что-нибудь, и я помогу!**

📌 **Используй кнопки ниже** для быстрого доступа:
/help — полный список команд
/stats — статистика системы
/think — активация глубокого мышления
/clear — очистка контекста

🚀 **Примеры:**
• "Запомни, что Python — мой любимый язык"
• "Сколько будет 25 * 34 + 17?"
• "Придумай креативное решение для..."
• "Объясни сложную концепцию просто"

📈 **Я запоминаю контекст и учусь на диалогах!**"""

    try:
        await update.message.reply_text(
            welcome_text,
            reply_markup=create_main_keyboard(),
            parse_mode='Markdown'
        )
    except Exception as e:
        print(f"❌ Ошибка отправки приветствия: {e}")
        await update.message.reply_text(
            f"👋 Привет, {user.first_name}!\n\n"
            "Я — когнитивный помощник AGI24. Напиши мне что-нибудь!\n"
            "Используй /help для списка команд."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """📖 **ПОЛНЫЙ СПРАВОЧНИК КОМАНД**

**🎯 ОСНОВНЫЕ КОМАНДЫ:**
/start — начало работы
/help — этот справочник
/stats — полная статистика
/clear — очистить контекст

**🧠 КОГНИТИВНЫЕ ФУНКЦИИ:**
/think — активировать глубокое мышление
/analyze — комплексный анализ системы
/goals — показать цели системы
/patterns — обнаруженные паттерны
/insights — инсайты из размышлений
/facts — сохранённые факты

**💡 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:**
• *Простые вопросы:* "Сколько будет 2+2?"
• *Анализ:* "Проанализируй эту ситуацию"
• *Память:* "Запомни, что я люблю кофе"
• *Творчество:* "Придумай название для проекта"
• *Планирование:* "Помоги спланировать день"
• *Обучение:* "Объясни квантовую физику просто"

**🎮 ИНТЕРАКТИВНЫЕ ВОЗМОЖНОСТИ:**
• Используй кнопки для быстрых действий
• Бот запоминает контекст разговора
• Автоматическое обнаружение паттернов
• Адаптация к стилю общения
• Непрерывное обучение

💬 **Просто напиши что-нибудь — и я постараюсь помочь!**"""

    try:
        await update.message.reply_text(
            help_text,
            parse_mode='Markdown'
        )
    except Exception as e:
        print(f"❌ Ошибка отправки помощи: {e}")
        await update.message.reply_text(
            "Используй команды:\n"
            "/start - начать\n"
            "/stats - статистика\n"
            "/help - подробная помощь\n"
            "\nИли просто напиши сообщение!"
        )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        stats = agent._get_comprehensive_stats()

        bot_stats = session_manager.get_stats()
        stats += f"\n\n🤖 **СТАТИСТИКА БОТА:**"
        stats += f"\nВсего пользователей: {bot_stats['total_users']}"
        stats += f"\nАктивных сейчас: {bot_stats['active_users']}"
        stats += f"\nВсего сообщений: {bot_stats['total_messages']}"
        stats += f"\nСообщений в вашей сессии: {session['message_count']}"

        parts = split_message(stats)
        for i, part in enumerate(parts):
            if i == 0:
                await update.message.reply_text(part, parse_mode='Markdown')
            else:
                await update.message.reply_text(part)

            if i < len(parts) - 1:
                await asyncio.sleep(0.3)

    except Exception as e:
        print(f"❌ Ошибка в stats_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка получения статистики: {str(e)[:100]}"
        )


async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        await update.message.reply_text(
            "🧠 Активирую глубокое многоуровневое мышление...\n"
            "Это может занять некоторое время."
        )

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        await agent._deep_autonomous_thinking()

        await update.message.reply_text(
            "✅ Глубокое мышление завершено!\n"
            "Проверьте /insights для результатов или /analyze для анализа системы.",
            reply_markup=create_main_keyboard()
        )

    except Exception as e:
        print(f"❌ Ошибка в think_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка активации глубокого мышления: {str(e)[:100]}"
        )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        analysis = agent._get_comprehensive_analysis()
        parts = split_message(analysis)

        for i, part in enumerate(parts):
            if i == 0:
                await update.message.reply_text(part, parse_mode='Markdown')
            else:
                await update.message.reply_text(part)

            if i < len(parts) - 1:
                await asyncio.sleep(0.3)

    except Exception as e:
        print(f"❌ Ошибка в analyze_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка анализа: {str(e)[:100]}"
        )


async def goals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        goals = agent._format_goal_hierarchy()
        parts = split_message(goals)

        for part in parts:
            await update.message.reply_text(part)
            await asyncio.sleep(0.3)

    except Exception as e:
        print(f"❌ Ошибка в goals_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка получения целей: {str(e)[:100]}"
        )


async def patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        patterns = agent._format_patterns()
        parts = split_message(patterns)

        for part in parts:
            await update.message.reply_text(part)
            await asyncio.sleep(0.3)

    except Exception as e:
        print(f"❌ Ошибка в patterns_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка получения паттернов: {str(e)[:100]}"
        )


async def insights_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        insights = agent._format_insights()
        parts = split_message(insights)

        for part in parts:
            await update.message.reply_text(part)
            await asyncio.sleep(0.3)

    except Exception as e:
        print(f"❌ Ошибка в insights_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка получения инсайтов: {str(e)[:100]}"
        )


async def facts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        facts = agent.db.get_relevant_facts("все факты", limit=25)
        if not facts:
            await update.message.reply_text(
                "📚 Фактов пока не сохранено.\n\n"
                "Чтобы добавить факты, используйте фразы типа:\n"
                "• \"Запомни, что я люблю кофе\"\n"
                "• \"Python = мой любимый язык\"\n"
                "• \"Мой день рождения — 15 января\""
            )
            return

        categories = defaultdict(list)
        for fact in facts:
            category = fact.get('category', 'разное')
            categories[category].append(fact)

        lines = ["📚 **СОХРАНЁННЫЕ ФАКТЫ:**\n"]

        for category, category_facts in categories.items():
            lines.append(f"\n📌 **{category.upper()}:**")
            for fact in category_facts[:5]:
                confidence_stars = "★" * int(fact['confidence'] * 5)
                lines.append(f"• *{fact['key']}:* {fact['value']} [{confidence_stars}]")

            if len(category_facts) > 5:
                lines.append(f"... и ещё {len(category_facts) - 5}")

        lines.append(f"\n📊 Всего фактов: {len(facts)}")

        text = "\n".join(lines)
        parts = split_message(text)

        for i, part in enumerate(parts):
            if i == 0:
                await update.message.reply_text(part, parse_mode='Markdown')
            else:
                await update.message.reply_text(part)

            if i < len(parts) - 1:
                await asyncio.sleep(0.3)

    except Exception as e:
        print(f"❌ Ошибка в facts_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка получения фактов: {str(e)[:100]}"
        )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        agent.context_window.clear()

        await update.message.reply_text(
            "🧹 Контекст разговора очищен!\n\n"
            "Теперь я не помню предыдущие сообщения из этого диалога.\n"
            "Память в базе данных осталась нетронутой.",
            reply_markup=create_main_keyboard()
        )

    except Exception as e:
        print(f"❌ Ошибка в clear_command: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка очистки контекста: {str(e)[:100]}"
        )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /ping"""
    try:
        await update.message.reply_text(
            "🏓 Pong!\n\n"
            "✅ Бот активен и работает\n"
            "📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        print(f"❌ Ошибка в ping_command: {e}")


# ================= ОБРАБОТЧИК КНОПОК =================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query

    try:
        await query.answer()
    except Exception as e:
        print(f"⚠️ Ошибка answer callback: {e}")

    user_id = update.effective_user.id
    callback_data = query.data

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        response = ""
        parse_mode = None

        if callback_data == "deep_think":
            await query.message.reply_text("🧠 Активирую глубокое мышление...")
            await agent._deep_autonomous_thinking()
            response = "✅ Глубокое мышление завершено! Проверьте /insights для результатов."

        elif callback_data == "analysis":
            response = agent._get_comprehensive_analysis()
            parse_mode = 'Markdown'

        elif callback_data == "stats":
            response = agent._get_comprehensive_stats()
            bot_stats = session_manager.get_stats()
            response += f"\n\n🤖 **Статистика бота:**"
            response += f"\nАктивных пользователей: {bot_stats['active_users']}"
            response += f"\nВсего сообщений: {bot_stats['total_messages']}"
            response += f"\nСообщений в вашей сессии: {session['message_count']}"
            parse_mode = 'Markdown'

        elif callback_data == "goals":
            response = agent._format_goal_hierarchy()

        elif callback_data == "insights":
            response = agent._format_insights()

        elif callback_data == "patterns":
            response = agent._format_patterns()

        elif callback_data == "facts":
            facts = agent.db.get_relevant_facts("все", limit=15)
            if facts:
                response = "📚 **СОХРАНЁННЫЕ ФАКТЫ:**\n\n"
                for fact in facts[:10]:
                    response += f"• *{fact['key']}:* {fact['value'][:50]}\n"
                response += f"\n📊 Всего показано: {len(facts[:10])} из {len(facts)}"
                parse_mode = 'Markdown'
            else:
                response = "📚 Фактов пока нет. Добавьте факты через диалог."

        elif callback_data == "help":
            response = """📖 **ПОМОЩЬ И КОМАНДЫ**

💬 **Основное:** Просто пишите сообщения, и я буду помогать!

🎯 **Основные команды:**
/start - начало работы
/help - полная справка
/stats - статистика системы
/clear - очистить контекст

🧠 **Когнитивные функции:**
/think - глубокое мышление
/analyze - анализ системы
/goals - цели системы
/patterns - обнаруженные паттерны

📱 **Используйте кнопки для быстрого доступа к функциям!**

💡 **Совет:** Я запоминаю контекст разговора и учусь на диалогах."""
            parse_mode = 'Markdown'

        else:
            response = "❓ Неизвестная команда"

        if response:
            parts = split_message(response)
            for i, part in enumerate(parts):
                if i == 0:
                    try:
                        await query.edit_message_text(
                            text=part,
                            parse_mode=parse_mode,
                            reply_markup=create_main_keyboard()
                        )
                    except Exception as e:
                        await query.message.reply_text(
                            part,
                            parse_mode=parse_mode,
                            reply_markup=create_main_keyboard()
                        )
                else:
                    await query.message.reply_text(part)

                if i < len(parts) - 1:
                    await asyncio.sleep(0.3)

    except Exception as e:
        print(f"❌ Ошибка в button_callback: {e}")
        try:
            await query.edit_message_text(
                f"⚠️ Ошибка обработки запроса: {str(e)[:100]}",
                reply_markup=create_main_keyboard()
            )
        except:
            await query.message.reply_text(
                f"⚠️ Ошибка обработки запроса: {str(e)[:100]}",
                reply_markup=create_main_keyboard()
            )


# ================= ОБРАБОТЧИК СООБЩЕНИЙ =================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    print(f"📨 Сообщение от {user_id}: {user_message[:50]}...")

    try:
        session = await session_manager.get_or_create_session(user_id)
        agent = session['agent']
        session['message_count'] += 1

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        await asyncio.sleep(Config.TYPING_DELAY)

        response = await agent.process_input(user_message, user_id)

        if response == "SYSTEM_EXIT":
            await update.message.reply_text("👋 До новых встреч!")
            return

        parts = split_message(response, Config.MAX_MESSAGE_LENGTH)
        if len(parts) > Config.MAX_RESPONSE_CHUNKS:
            parts = parts[:Config.MAX_RESPONSE_CHUNKS]
            parts.append("\n\n📝 *Сообщение слишком длинное, показана только часть*")

        for i, part in enumerate(parts):
            await update.message.reply_text(
                part,
                parse_mode='Markdown' if i == 0 else None,
                reply_markup=create_main_keyboard() if i == len(parts) - 1 and session[
                    'message_count'] % 5 == 0 else None
            )

            if i < len(parts) - 1:
                await asyncio.sleep(0.5)

        if session['message_count'] % 10 == 0:
            await update.message.reply_text(
                "💡 Что ещё могу сделать? Используйте кнопки ниже или команды.",
                reply_markup=create_main_keyboard()
            )

    except Exception as e:
        logging.error(f"❌ Ошибка обработки сообщения от {user_id}: {e}")

        error_message = f"⚠️ Произошла ошибка при обработке вашего сообщения.\n\n"

        if "API" in str(e):
            error_message += "**Проблема с API:**\n"
            error_message += "• Проверьте интернет-соединение\n"
            error_message += "• Проверьте API ключ в .env файле\n"
            error_message += "• Убедитесь, что ключ активен на openrouter.ai\n\n"
        elif "баз" in str(e).lower() or "sql" in str(e).lower():
            error_message += "**Проблема с базой данных:**\n"
            error_message += "• Проверьте права доступа к файлам\n"
            error_message += "• Убедитесь, что есть место на диске\n\n"
        elif "timeout" in str(e).lower():
            error_message += "**Таймаут запроса:**\n"
            error_message += "• Сервер долго не отвечает\n"
            error_message += "• Попробуйте позже или упростите запрос\n\n"
        else:
            error_message += f"**Ошибка:** {str(e)[:150]}\n\n"

        error_message += "Попробуйте ещё раз или используйте /start для перезапуска."

        try:
            await update.message.reply_text(
                error_message,
                parse_mode='Markdown',
                reply_markup=create_main_keyboard()
            )
        except Exception as send_error:
            print(f"❌ Ошибка отправки сообщения об ошибке: {send_error}")


# ================= ОБРАБОТКА ОШИБОК =================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    error = context.error

    logging.error(f"Глобальная ошибка: {error}", exc_info=error)

    error_type = "Неизвестная ошибка"
    user_message = "Произошла непредвиденная ошибка."

    if isinstance(error, telegram.error.TimedOut):
        error_type = "Таймаут"
        user_message = "Превышено время ожидания ответа. Попробуйте позже."
    elif isinstance(error, telegram.error.NetworkError):
        error_type = "Сетевая ошибка"
        user_message = "Проблемы с сетью. Проверьте интернет-соединение."
    elif isinstance(error, telegram.error.TelegramError):
        error_type = "Ошибка Telegram API"
        user_message = "Проблема с Telegram API. Попробуйте позже."
    elif isinstance(error, asyncio.TimeoutError):
        error_type = "Таймаут операции"
        user_message = "Операция заняла слишком много времени."

    print(f"🚨 Глобальная ошибка ({error_type}): {error}")

    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                f"⚠️ {user_message}\n\n"
                f"Ошибка: {str(error)[:100]}\n\n"
                "Попробуйте ещё раз или используйте /start для перезапуска."
            )
        except Exception as e:
            print(f"❌ Не удалось отправить сообщение об ошибке: {e}")


# ================= ГЛАВНАЯ ФУНКЦИЯ =================

async def main():
    """Основная асинхронная функция запуска бота — СОВМЕСТИМО С PYTHON 3.13"""
    print("=" * 70)
    print("🚀 ЗАПУСК AGI24 КОГНИТИВНОГО АГЕНТА С TELEGRAM ИНТЕРФЕЙСОМ")
    print("=" * 70)

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    try:
        # ✅ ПОЛУЧЕНИЕ ТОКЕНА ЧЕРЕЗ CONFIG (без глобальной переменной)
        token = Config.get_telegram_token()
        print(f"✅ Токен Telegram получен: {token[:15]}...")

        # ✅ СОЗДАНИЕ ПРИЛОЖЕНИЯ
        app = ApplicationBuilder().token(token).build()

        # Регистрация обработчиков
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("clear", clear_history))
        app.add_handler(CommandHandler("stats", show_stats))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.DOCUMENT, handle_media))
        app.add_error_handler(error_handler)

        print("\n" + "=" * 70)
        print("✅ Бот успешно инициализирован и готов к работе!")
        print("📱 Найдите бота в Telegram и напишите /start")
        print("\n🛑 Для остановки нажмите Ctrl+C")
        print("=" * 70 + "\n")

        # ✅ РУЧНОЙ ЗАПУСК БЕЗ run_polling() — КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
        await app.initialize()
        await app.start()

        # ЗАПУСК POLLING БЕЗ ПАРАМЕТРА close_loop (его нет в старых версиях!)
        # Используем только поддерживаемые параметры
        await app.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            # close_loop=False — УДАЛЕНО, так как не поддерживается!
        )

        print("🔄 Бот работает в режиме ожидания сообщений...")
        print("   (Нажмите Ctrl+C для остановки)\n")

        # Бесконечное ожидание с возможностью прерывания
        # Используем короткие интервалы для быстрого реагирования на Ctrl+C
        while True:
            await asyncio.sleep(1)  # Короткий сон для отзывчивости

    except KeyboardInterrupt:
        print("\n👋 Получен сигнал остановки (Ctrl+C)...")
        raise
    except ValueError as e:
        print(f"\n❌ Ошибка конфигурации: {e}")
        print("\n💡 Создайте файл .env в корне проекта:")
        print("OPENROUTER_API_KEY=ваш_ключ_openrouter")
        print("TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather")
        raise
    except Exception as e:
        print(f"\n🚨 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # ✅ КОРРЕКТНОЕ ЗАВЕРШЕНИЕ БЕЗ ПОПЫТОК ЗАКРЫТЬ ЦИКЛ
        print("\n🔄 Завершение работы бота...")
        try:
            # Остановка polling
            if hasattr(app, 'updater') and app.updater and app.updater.running:
                await app.updater.stop()
                print("✅ Polling остановлен")

            # Остановка приложения
            if hasattr(app, 'running') and app.running:
                await app.stop()
                print("✅ Приложение остановлено")

            # Освобождение ресурсов
            await app.shutdown()
            print("✅ Ресурсы освобождены")

            # Статистика при завершении
            if session_manager:
                stats = session_manager.get_stats()
                print(f"\n📊 Финальная статистика:")
                print(f"   • Всего пользователей: {stats['total_users']}")
                print(f"   • Всего сообщений: {stats['total_messages']}")
                print(f"   • Активных сессий: {stats['active_users']}")

        except Exception as e:
            print(f"⚠️  Ошибка при завершении: {e}")


def run():
    """Точка входа для запуска бота — СОВМЕСТИМО С PYTHON 3.13"""
    print("AGI24 Cognitive Bot - Version 3.0")
    print("Copyright (c) 2024 AGI24 Project")
    print("\n" + "=" * 70)

    # Проверка версии Python
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        print(f"📌 У вас установлен Python {sys.version}")
        sys.exit(1)

    # Предупреждение о версии 3.13
    if sys.version_info >= (3, 13):
        print("⚠️  ВНИМАНИЕ: Версия Python 3.13")
        print("📌 Рекомендуется использовать Python 3.10-3.12 для полной совместимости")

    # Проверка библиотек
    required_libs = ['aiohttp', 'telegram']
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        print(f"❌ Отсутствуют библиотеки: {', '.join(missing)}")
        print("📦 Установите: pip install python-telegram-bot aiohttp")
        sys.exit(1)
    else:
        print("✅ Библиотеки загружены успешно")

    # ✅ ИСПРАВЛЕНИЕ: Используем правильное имя класса UserSessionManager
    global session_manager
    try:
        session_manager = UserSessionManager()  # Было: SessionManager()
        print("✅ Менеджер сессий инициализирован")
    except Exception as e:
        print(f"⚠️  Ошибка инициализации менеджера сессий: {e}")

    # Запуск основного цикла
    try:
        import asyncio

        # Для Windows: использовать совместимую политику цикла
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # ✅ nest_asyncio для совместимости с уже запущенным циклом (в PyCharm)
        try:
            import nest_asyncio
            nest_asyncio.apply()
            print("✅ nest_asyncio применён для совместимости с Python 3.13")
        except ImportError:
            print("⚠️  nest_asyncio не установлен. Установите: pip install nest_asyncio")

        # ✅ asyncio.run() создаёт НОВЫЙ чистый цикл событий
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем (Ctrl+C)")
        print("\n✅ Бот завершил работу корректно")
        sys.exit(0)
    except Exception as e:
        print(f"\n🚨 Критическая ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()