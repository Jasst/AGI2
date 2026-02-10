# coding: utf-8
"""
AGI_v29_Enhanced.py — УЛУЧШЕННАЯ АВТОНОМНАЯ СИСТЕМА С РАСШИРЕННЫМИ КОГНИТИВНЫМИ СПОСОБНОСТЯМИ

Ключевые улучшения:
1. Многоуровневая система мышления (аналитическое, творческое, критическое)
2. Контекстно-зависимая память с весами релевантности
3. Предиктивное планирование и моделирование
4. Эмоциональный интеллект и понимание контекста
5. Self-improvement через метапознание
6. Векторный поиск в памяти
7. Приоритизация задач и адаптивное обучение
"""

import re
import json
import asyncio
import aiohttp
import time
import os
import sys
import sqlite3
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque
import logging
from contextlib import contextmanager
import random
import math


# ================= РАСШИРЕННАЯ КОНФИГУРАЦИЯ =================

class Config:
    """Улучшенная конфигурация с адаптивными параметрами"""
    ROOT = Path("./cognitive_system_v29")
    ROOT.mkdir(exist_ok=True)

    DB_PATH = ROOT / "memory.db"
    CACHE_PATH = ROOT / "cache.json"
    LOG_PATH = ROOT / "system.log"
    KNOWLEDGE_GRAPH_PATH = ROOT / "knowledge_graph.json"

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    TIMEOUT = 300
    MAX_TOKENS = 800

    # Когнитивные параметры
    REFLECTION_INTERVAL = 3  # Чаще рефлексировать
    DEEP_THINKING_THRESHOLD = 0.7  # Порог для глубокого анализа
    CREATIVITY_FACTOR = 0.8  # Уровень креативности
    LEARNING_RATE = 0.15  # Скорость обучения

    # Память
    MAX_MEMORY_ITEMS = 10000
    CONTEXT_WINDOW_SIZE = 10  # Размер контекстного окна
    MEMORY_DECAY_RATE = 0.05  # Скорость "забывания"

    # Мышление
    THOUGHT_TYPES = [
        'рефлексия', 'анализ', 'планирование', 'обучение',
        'наблюдение', 'синтез', 'критика', 'творчество',
        'предсказание', 'оценка'
    ]

    @classmethod
    def get_api_key(cls):
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            return key

        env_path = Path(".env")
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            return line.split("=", 1)[1].strip('"\' ')
            except Exception as e:
                print(f"⚠️ Ошибка чтения .env: {e}")

        raise ValueError("API ключ не найден. Создайте файл .env с OPENROUTER_API_KEY=ваш_ключ")


# ================= УЛУЧШЕННЫЕ УТИЛИТЫ =================

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Улучшенный расчёт схожести с учётом n-грамм"""

    def get_ngrams(text: str, n: int = 2) -> Set[str]:
        words = text.lower().split()
        return set(' '.join(words[i:i + n]) for i in range(len(words) - n + 1))

    # Юниграммы
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    # Биграммы
    bigrams1 = get_ngrams(text1, 2)
    bigrams2 = get_ngrams(text2, 2)

    # Комбинированная схожесть
    unigram_sim = len(words1.intersection(words2)) / max(len(words1), len(words2))
    bigram_sim = len(bigrams1.intersection(bigrams2)) / max(len(bigrams1), len(bigrams2), 1)

    return 0.6 * unigram_sim + 0.4 * bigram_sim


def extract_semantic_features(text: str) -> Dict[str, Any]:
    """Извлечение семантических характеристик текста"""
    features = {
        'length': len(text.split()),
        'complexity': len(set(text.lower().split())) / max(len(text.split()), 1),
        'question_words': len(re.findall(r'\b(как|что|почему|зачем|когда|где|кто)\b', text.lower())),
        'numbers': len(re.findall(r'\b\d+\b', text)),
        'emotions': len(re.findall(r'\b(хорошо|плохо|отлично|ужасно|интересно|скучно)\b', text.lower())),
        'imperatives': len(re.findall(r'\b(сделай|создай|найди|покажи|расскажи)\b', text.lower())),
        'has_question': '?' in text,
        'sentiment': analyze_sentiment(text)
    }
    return features


def analyze_sentiment(text: str) -> float:
    """Простой анализ тональности (-1 до 1)"""
    positive = ['хорошо', 'отлично', 'прекрасно', 'замечательно', 'классно', 'супер']
    negative = ['плохо', 'ужасно', 'отвратительно', 'кошмар', 'провал']

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

            # Таблица взаимодействий с дополнительными метриками
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
                    tokens_used INTEGER DEFAULT 0
                )
            ''')

            # Таблица фактов с весами и связями
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

            # Таблица связей между фактами
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fact_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id_1 INTEGER NOT NULL,
                    fact_id_2 INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (fact_id_1) REFERENCES facts(id),
                    FOREIGN KEY (fact_id_2) REFERENCES facts(id)
                )
            ''')

            # Таблица мыслей с метаданными
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

            # Таблица целей с подцелями и метриками
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

            # Таблица паттернов (обучение)
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

            # Индексы для производительности
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_time ON interactions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_category ON interactions(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_importance ON facts(importance)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoughts_type ON thoughts(thought_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoughts_importance ON thoughts(importance)')

            conn.commit()

    # === Взаимодействия ===

    def add_interaction(self, user_input: str, system_response: str, **kwargs) -> int:
        """Добавление взаимодействия с метриками"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions 
                (timestamp, user_input, system_response, context, emotion, category,
                 importance, complexity, satisfaction, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
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

    def get_contextual_interactions(self, query: str, limit: int = 5) -> List[Dict]:
        """Получение контекстно-релевантных взаимодействий"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Получаем больше записей для фильтрации
            cursor.execute('''
                SELECT * FROM interactions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit * 3,))

            all_interactions = [dict(row) for row in cursor.fetchall()]

            # Ранжируем по релевантности
            scored = []
            for interaction in all_interactions:
                relevance = calculate_text_similarity(
                    query,
                    interaction['user_input'] + ' ' + interaction['system_response']
                )
                recency = 1.0 - (time.time() - interaction['timestamp']) / (30 * 24 * 3600)  # 30 дней
                recency = max(0, min(1, recency))

                score = 0.6 * relevance + 0.3 * interaction['importance'] + 0.1 * recency
                scored.append((score, interaction))

            # Сортируем и возвращаем топ
            scored.sort(reverse=True, key=lambda x: x[0])
            return [item[1] for item in scored[:limit]]

    # === Факты ===

    def add_fact(self, key: str, value: str, **kwargs):
        """Добавление факта с дополнительными метриками"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT id FROM facts WHERE key = ? AND value = ?', (key, value))
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
                    key, value,
                    kwargs.get('category', ''),
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(), time.time(), 1, 1.0,
                    kwargs.get('source', 'user')
                ))

            conn.commit()

    def get_relevant_facts(self, query: str, limit: int = 5) -> List[Dict]:
        """Получение релевантных фактов с учётом затухания памяти"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Обновляем decay_factor
            cursor.execute('''
                UPDATE facts
                SET decay_factor = decay_factor * (1 - ?)
                WHERE last_used < ?
            ''', (Config.MEMORY_DECAY_RATE, time.time() - 86400))  # 1 день

            cursor.execute('''
                SELECT * FROM facts
                WHERE confidence > 0.3 AND decay_factor > 0.1
                ORDER BY importance DESC, usage_count DESC
                LIMIT ?
            ''', (limit * 2,))

            all_facts = [dict(row) for row in cursor.fetchall()]

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
            return [item[1] for item in scored[:limit]]

    def add_fact_relation(self, fact_id_1: int, fact_id_2: int, relation_type: str, strength: float = 0.5):
        """Создание связи между фактами"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO fact_relations
                (fact_id_1, fact_id_2, relation_type, strength, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (fact_id_1, fact_id_2, relation_type, strength, time.time()))
            conn.commit()

    # === Мысли ===

    def add_thought(self, thought_type: str, content: str, **kwargs):
        """Добавление мысли с метаданными"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO thoughts
                (timestamp, thought_type, content, trigger, importance, depth_level, confidence, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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

    def get_thought_insights(self, limit: int = 10) -> List[Dict]:
        """Получение инсайтов из мыслей"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT thought_type, COUNT(*) as count, AVG(importance) as avg_importance
                FROM thoughts
                WHERE timestamp > ?
                GROUP BY thought_type
                ORDER BY count DESC
                LIMIT ?
            ''', (time.time() - 7 * 86400, limit))  # За последнюю неделю

            return [dict(row) for row in cursor.fetchall()]

    # === Паттерны ===

    def add_pattern(self, pattern_type: str, description: str, confidence: float = 0.5):
        """Добавление обнаруженного паттерна"""
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
        """Получение обнаруженных паттернов"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM patterns
                WHERE confidence >= ?
                ORDER BY occurrences DESC, confidence DESC
                LIMIT ?
            ''', (min_confidence, limit))

            return [dict(row) for row in cursor.fetchall()]

    # === Цели ===

    def add_goal(self, description: str, **kwargs) -> int:
        """Добавление цели с улучшенными параметрами"""
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
        """Получение иерархии целей"""
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


# ================= УЛУЧШЕННАЯ СИСТЕМА МЫШЛЕНИЯ =================

class EnhancedThinkingSystem:
    """Продвинутая многоуровневая система мышления"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit = 1.5
        self.last_request_time = 0

        # Кэш с приоритетами
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # История рассуждений
        self.reasoning_history = deque(maxlen=50)

        # Метакогниция
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
            ('surface', 'Что очевидно?', 0.3),
            ('analytical', 'Какие связи и паттерны можно увидеть?', 0.5),
            ('strategic', 'Какие долгосрочные последствия?', 0.7),
            ('creative', 'Какие неожиданные решения возможны?', 0.9)
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
        system_prompt = """Ты — продвинутая когнитивная система с способностью к глубокому анализу.
Твои сильные стороны:
- Многоуровневое мышление
- Обнаружение неявных паттернов
- Креативное решение проблем
- Критический анализ
- Предсказание последствий

Отвечай кратко, но ёмко. Фокусируйся на инсайтах, а не на очевидных вещах."""

        response = await self.call_llm(system_prompt, prompt, temperature)

        if response and len(response) > 15:
            # Сохраняем в историю рассуждений
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

Оцени:
1. Вероятность успеха (0-1)
2. Возможные риски
3. Ожидаемые выгоды
4. Альтернативные подходы

Формат ответа: JSON"""

        system_prompt = "Ты эксперт по прогнозированию и анализу рисков. Будь точным и объективным."

        response = await self.call_llm(system_prompt, prompt, temperature=0.5)

        # Попытка распарсить JSON
        try:
            # Извлекаем JSON из ответа
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {'raw_prediction': response}

    async def synthesize_knowledge(self, facts: List[Dict], question: str) -> str:
        """Синтез знаний из нескольких фактов"""
        facts_text = "\n".join([f"- {f['key']}: {f['value']}" for f in facts])

        prompt = f"""Синтезируй ответ на вопрос, используя доступные факты:

Вопрос: {question}

Доступные факты:
{facts_text}

Создай связный ответ, комбинируя эти факты. Если фактов недостаточно, укажи это."""

        system_prompt = "Ты эксперт по синтезу информации. Находи связи между фактами."

        return await self.call_llm(system_prompt, prompt, temperature=0.6)

    async def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Вызов LLM с улучшенным кэшированием"""
        # Проверка кэша
        cache_key = hashlib.md5(
            f"{system_prompt[:100]}{user_prompt[:200]}{temperature}".encode()
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
                        if len(self.cache) > 200:
                            # Удаляем 25% старых записей
                            keys_to_remove = list(self.cache.keys())[:50]
                            for key in keys_to_remove:
                                del self.cache[key]

                        return content
                    else:
                        return f"⚠️ Ошибка API: {response.status}"

        except asyncio.TimeoutError:
            return "⚠️ Таймаут запроса"
        except Exception as e:
            return f"⚠️ Ошибка: {str(e)[:100]}"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности мышления"""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0

        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'reasoning_history_size': len(self.reasoning_history),
            'thinking_performance': self.thinking_performance
        }


# ================= УЛУЧШЕННЫЙ АВТОНОМНЫЙ АГЕНТ =================

class EnhancedAutonomousAgent:
    """Продвинутый автономный агент с улучшенными когнитивными способностями"""

    def __init__(self):
        print("🧠 Продвинутый когнитивный агент v2.0\n")

        # Компоненты
        self.api_key = Config.get_api_key()
        self.db = EnhancedMemoryDB(Config.DB_PATH)
        self.thinker = EnhancedThinkingSystem(self.api_key)

        # Состояние
        self.interaction_count = 0
        self.deep_thoughts_count = 0
        self.patterns_found = 0
        self.start_time = time.time()

        # Контекстное окно
        self.context_window = deque(maxlen=Config.CONTEXT_WINDOW_SIZE)

        # Текущие активные задачи
        self.active_tasks = []

        # Метакогниция
        self.self_assessment = {
            'knowledge_gaps': [],
            'strong_areas': [],
            'improvement_areas': []
        }

        self._init_system()
        self.print_welcome()

    def _init_system(self):
        """Инициализация системы"""
        # Создаём базовые цели, если их нет
        existing_goals = self.db.get_goal_hierarchy()

        if not existing_goals:
            # Главные цели
            main_goal = self.db.add_goal(
                "Быть максимально полезным помощником",
                priority=1.0,
                success_criteria="Высокий уровень удовлетворённости пользователя"
            )

            # Подцели
            self.db.add_goal(
                "Глубоко понимать запросы",
                parent_goal_id=main_goal,
                priority=0.9
            )

            self.db.add_goal(
                "Непрерывно обучаться",
                parent_goal_id=main_goal,
                priority=0.85
            )

            self.db.add_goal(
                "Находить неочевидные решения",
                parent_goal_id=main_goal,
                priority=0.8
            )

    def print_welcome(self):
        print("=" * 70)
        print("🤖 ПРОДВИНУТЫЙ АВТОНОМНЫЙ КОГНИТИВНЫЙ АГЕНТ")
        print("=" * 70)

        # Статистика
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM interactions")
            interactions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM facts")
            facts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM thoughts")
            thoughts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM patterns")
            patterns = cursor.fetchone()[0]

        print(f"\n📊 Загружено:")
        print(f"  Взаимодействий: {interactions}")
        print(f"  Фактов: {facts}")
        print(f"  Мыслей: {thoughts}")
        print(f"  Паттернов: {patterns}")

        print("\n🧠 Когнитивные способности:")
        print("  ✓ Многоуровневый анализ")
        print("  ✓ Контекстная память")
        print("  ✓ Обнаружение паттернов")
        print("  ✓ Предиктивное мышление")
        print("  ✓ Творческое решение задач")

        print("\n💡 Команды:")
        print("  • думай глубоко - активировать глубокое мышление")
        print("  • анализ - провести анализ текущего состояния")
        print("  • паттерны - показать обнаруженные паттерны")
        print("  • инсайты - показать инсайты из мыслей")
        print("  • цели - показать иерархию целей")
        print("  • статистика - полная статистика системы")
        print("  • выход - завершить работу")
        print("=" * 70 + "\n")

    async def process_input(self, user_input: str) -> str:
        """Обработка ввода с улучшенной когнитивной обработкой"""
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

        # Извлекаем информацию
        self._extract_and_store_information(user_input, importance)

        # Генерируем ответ с учётом контекста
        response = await self._generate_contextual_response(
            user_input,
            features,
            complexity,
            importance
        )

        # Сохраняем взаимодействие
        self.db.add_interaction(
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

        # Обнаружение паттернов
        await self._detect_patterns()

        # Периодическое глубокое мышление
        if self.interaction_count % Config.REFLECTION_INTERVAL == 0:
            await self._deep_autonomous_thinking()

        # Метрики
        duration = time.time() - start_time
        if duration > 2.0:
            print(f"⏱️ Время обработки: {duration:.2f}с")

        return response

    def _calculate_importance(self, text: str, features: Dict) -> float:
        """Расчёт важности запроса"""
        importance = 0.5

        # Наличие ключевых слов
        if any(word in text.lower() for word in ['важно', 'срочно', 'критично', 'обязательно']):
            importance += 0.3

        # Вопросительные слова
        importance += min(0.2, features['question_words'] * 0.1)

        # Императивы
        importance += min(0.2, features['imperatives'] * 0.1)

        # Сложность
        importance += features['complexity'] * 0.2

        return min(1.0, importance)

    def _extract_and_store_information(self, text: str, importance: float):
        """Извлечение и сохранение информации с метриками"""
        # Извлекаем сущности
        from collections import Counter

        # Числа
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers:
            self.db.add_fact('число', num, category='данные', importance=importance * 0.5)

        # Имена (более надёжное извлечение)
        names = re.findall(r'\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*\b', text)
        for name in names:
            if len(name) > 3:
                self.db.add_fact('имя', name, category='персона', importance=importance * 0.7)

        # Ключевые утверждения
        patterns = [
            (r'(\w+)\s+(?:это|равно|составляет)\s+([^.,]+)', 'определение'),
            (r'запомни[,:]?\s*(.+)', 'важная_информация'),
            (r'(\w+)\s*=\s*([^,]+)', 'равенство')
        ]

        for pattern, category in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    key, value = match
                    self.db.add_fact(
                        key.strip(),
                        value.strip(),
                        category=category,
                        importance=importance,
                        source='user_input'
                    )

    async def _generate_contextual_response(
            self,
            user_input: str,
            features: Dict,
            complexity: float,
            importance: float
    ) -> str:
        """Генерация ответа с глубоким контекстом"""

        # Получаем релевантный контекст
        relevant_interactions = self.db.get_contextual_interactions(user_input, limit=3)
        relevant_facts = self.db.get_relevant_facts(user_input, limit=5)
        active_goals = self.db.get_goal_hierarchy()

        # Формируем контекст
        context_parts = []

        # История диалога
        if relevant_interactions:
            context_parts.append("📜 Релевантная история:")
            for interaction in relevant_interactions[:2]:
                context_parts.append(f"  П: {interaction['user_input'][:60]}...")
                context_parts.append(f"  Я: {interaction['system_response'][:60]}...")

        # Факты
        if relevant_facts:
            context_parts.append("\n📚 Релевантные факты:")
            for fact in relevant_facts[:4]:
                conf_stars = "★" * int(fact['confidence'] * 5)
                context_parts.append(f"  • {fact['key']}: {fact['value']} [{conf_stars}]")

        # Цели
        if active_goals:
            context_parts.append("\n🎯 Текущие цели:")
            for goal in active_goals[:2]:
                context_parts.append(f"  • {goal['description'][:50]}")

        # Паттерны
        patterns = self.db.get_patterns(min_confidence=0.7, limit=2)
        if patterns:
            context_parts.append("\n🔍 Обнаруженные паттерны:")
            for pattern in patterns:
                context_parts.append(f"  • {pattern['description'][:60]}")

        context = "\n".join(context_parts) if context_parts else "Нет дополнительного контекста"

        # Определяем нужен ли глубокий анализ
        needs_deep_thinking = (
                complexity > Config.DEEP_THINKING_THRESHOLD or
                importance > 0.7 or
                features['question_words'] > 2
        )

        if needs_deep_thinking:
            # Многоуровневое мышление
            deep_thoughts = await self.thinker.multi_level_thinking(
                f"Запрос: {user_input}\n{context}",
                depth=3
            )

            # Синтез ответа
            synthesis_prompt = f"""На основе многоуровневого анализа ответь на запрос.

Запрос пользователя: {user_input}

Анализ:
{chr(10).join([f'{level}: {thought}' for level, thought in deep_thoughts.items()])}

Контекст:
{context}

Дай цельный, инсайтный ответ."""

            system_prompt = """Ты — продвинутая когнитивная система с глубоким пониманием контекста.
Синтезируй ответ, учитывая все уровни анализа. Будь точным, но креативным."""

            response = await self.thinker.call_llm(
                system_prompt,
                synthesis_prompt,
                temperature=0.7
            )

            self.deep_thoughts_count += 1
        else:
            # Обычный ответ
            system_prompt = f"""Ты — когнитивный помощник с доступом к памяти и контексту.

Контекст:
{context}

Принципы:
- Используй факты из памяти
- Будь точным и полезным
- Учитывай активные цели
- Давай конкретные ответы"""

            response = await self.thinker.call_llm(
                system_prompt,
                user_input,
                temperature=0.6
            )

        return response.strip()

    async def _detect_patterns(self):
        """Обнаружение паттернов в поведении и данных"""
        # Анализируем последние взаимодействия
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Частые категории
            cursor.execute('''
                SELECT category, COUNT(*) as count
                FROM interactions
                WHERE timestamp > ?
                GROUP BY category
                HAVING count > 2
            ''', (time.time() - 7 * 86400,))

            categories = cursor.fetchall()

            for category, count in categories:
                if category:
                    self.db.add_pattern(
                        'frequent_category',
                        f"Пользователь часто спрашивает про {category}",
                        confidence=min(1.0, count / 10)
                    )

            # Временные паттерны
            cursor.execute('''
                SELECT 
                    strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                    COUNT(*) as count
                FROM interactions
                WHERE timestamp > ?
                GROUP BY hour
                HAVING count > 3
            ''', (time.time() - 7 * 86400,))

            time_patterns = cursor.fetchall()

            for hour, count in time_patterns:
                self.db.add_pattern(
                    'time_preference',
                    f"Активность в {hour}:00",
                    confidence=0.6
                )

            self.patterns_found = len(categories) + len(time_patterns)

    async def _deep_autonomous_thinking(self):
        """Глубокое автономное мышление"""
        print("\n💭 [Глубокое автономное мышление...]", flush=True)

        # Собираем данные для рефлексии
        recent = self.db.get_contextual_interactions("общение пользователь", limit=5)
        patterns = self.db.get_patterns(min_confidence=0.6)
        insights = self.db.get_thought_insights()

        if not recent:
            print("  💭 Недостаточно данных для глубокого анализа")
            return

        # Формируем контекст для размышлений
        context_lines = [
            "Последние взаимодействия:",
            *[f"- {i['user_input'][:50]}..." for i in recent[:3]],
            "\nОбнаруженные паттерны:",
            *[f"- {p['description']}" for p in patterns[:3]],
            "\nИнсайты из мыслей:",
            *[f"- {t['thought_type']}: {t['count']} раз" for t in insights[:3]]
        ]

        context = "\n".join(context_lines)

        # Многоуровневое мышление
        thoughts = await self.thinker.multi_level_thinking(context, depth=3)

        # Сохраняем мысли
        for thought_type, content in thoughts.items():
            if content:
                self.db.add_thought(
                    thought_type=thought_type,
                    content=content,
                    trigger='autonomous_deep_thinking',
                    importance=0.8,
                    depth_level=3,
                    confidence=0.7
                )

                print(f"  💡 [{thought_type}] {content[:80]}...")

        # Обновляем самооценку
        await self._update_self_assessment()

    async def _update_self_assessment(self):
        """Обновление самооценки системы"""
        # Анализируем эффективность
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Средняя удовлетворённость
            cursor.execute('''
                SELECT AVG(satisfaction) FROM interactions
                WHERE timestamp > ?
            ''', (time.time() - 86400,))

            avg_satisfaction = cursor.fetchone()[0] or 0.5

            # Категории с низкой удовлетворённостью
            cursor.execute('''
                SELECT category, AVG(satisfaction) as avg_sat
                FROM interactions
                WHERE timestamp > ? AND category IS NOT NULL
                GROUP BY category
                HAVING avg_sat < 0.5
            ''', (time.time() - 7 * 86400,))

            weak_categories = [row[0] for row in cursor.fetchall()]

            # Категории с высокой удовлетворённостью
            cursor.execute('''
                SELECT category, AVG(satisfaction) as avg_sat
                FROM interactions
                WHERE timestamp > ? AND category IS NOT NULL
                GROUP BY category
                HAVING avg_sat > 0.7
            ''', (time.time() - 7 * 86400,))

            strong_categories = [row[0] for row in cursor.fetchall()]

        # Обновляем самооценку
        self.self_assessment = {
            'avg_satisfaction': avg_satisfaction,
            'improvement_areas': weak_categories,
            'strong_areas': strong_categories,
            'patterns_discovered': self.patterns_found,
            'deep_thoughts': self.deep_thoughts_count
        }

    def _handle_command(self, text: str) -> Optional[str]:
        """Обработка команд с расширенным набором"""
        text_lower = text.lower().strip()

        if text_lower in ['думай глубоко', 'глубокое мышление']:
            asyncio.create_task(self._deep_autonomous_thinking())
            return "🧠 Запускаю глубокое многоуровневое мышление..."

        elif text_lower == 'анализ':
            return self._get_comprehensive_analysis()

        elif text_lower == 'паттерны':
            return self._format_patterns()

        elif text_lower == 'инсайты':
            return self._format_insights()

        elif text_lower == 'цели':
            return self._format_goal_hierarchy()

        elif text_lower == 'статистика':
            return self._get_comprehensive_stats()

        elif text_lower in ['выход', 'exit', 'quit']:
            return "SYSTEM_EXIT"

        return None

    def _get_comprehensive_analysis(self) -> str:
        """Комплексный анализ текущего состояния"""
        lines = ["🔍 КОМПЛЕКСНЫЙ АНАЛИЗ СИСТЕМЫ", "=" * 60]

        # Самооценка
        lines.append("\n📊 Самооценка:")
        lines.append(f"  Средняя удовлетворённость: {self.self_assessment.get('avg_satisfaction', 0.5):.2f}")
        lines.append(f"  Паттернов обнаружено: {self.patterns_found}")
        lines.append(f"  Глубоких мыслей: {self.deep_thoughts_count}")

        if self.self_assessment.get('strong_areas'):
            lines.append(f"\n✅ Сильные области:")
            for area in self.self_assessment['strong_areas'][:3]:
                lines.append(f"  • {area}")

        if self.self_assessment.get('improvement_areas'):
            lines.append(f"\n⚠️ Области для улучшения:")
            for area in self.self_assessment['improvement_areas'][:3]:
                lines.append(f"  • {area}")

        # Производительность мышления
        perf = self.thinker.get_performance_metrics()
        lines.append(f"\n🧠 Производительность мышления:")
        lines.append(f"  Cache hit rate: {perf['cache_hit_rate']:.1%}")
        lines.append(f"  Размер кэша: {perf['cache_size']}")
        lines.append(f"  История рассуждений: {perf['reasoning_history_size']}")

        return "\n".join(lines)

    def _format_patterns(self) -> str:
        """Форматирование обнаруженных паттернов"""
        patterns = self.db.get_patterns(min_confidence=0.5, limit=15)

        if not patterns:
            return "🔍 Паттернов пока не обнаружено."

        lines = ["🔍 ОБНАРУЖЕННЫЕ ПАТТЕРНЫ", "=" * 60]

        # Группируем по типу
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
        """Форматирование инсайтов из мыслей"""
        insights = self.db.get_thought_insights(limit=10)

        if not insights:
            return "💡 Инсайтов пока нет."

        lines = ["💡 ИНСАЙТЫ ИЗ МЫСЛЕЙ", "=" * 60]

        for insight in insights:
            lines.append(f"\n🧠 {insight['thought_type'].upper()}:")
            lines.append(f"  Количество: {insight['count']}")
            lines.append(f"  Средняя важность: {insight['avg_importance']:.2f}")

        return "\n".join(lines)

    def _format_goal_hierarchy(self) -> str:
        """Форматирование иерархии целей"""
        main_goals = self.db.get_goal_hierarchy(parent_id=None)

        if not main_goals:
            return "🎯 Нет активных целей."

        lines = ["🎯 ИЕРАРХИЯ ЦЕЛЕЙ", "=" * 60]

        for goal in main_goals:
            progress_bar = "█" * int(goal['progress'] * 10) + "░" * (10 - int(goal['progress'] * 10))
            lines.append(f"\n📍 {goal['description']}")
            lines.append(f"  Приоритет: {goal['priority']:.2f} | Прогресс: [{progress_bar}] {goal['progress']:.0%}")

            if goal['next_action']:
                lines.append(f"  Следующий шаг: {goal['next_action']}")

            # Подцели
            subgoals = self.db.get_goal_hierarchy(parent_id=goal['id'])
            if subgoals:
                lines.append("  Подцели:")
                for sub in subgoals[:3]:
                    sub_bar = "█" * int(sub['progress'] * 5)
                    lines.append(f"    • {sub['description'][:50]} [{sub_bar}]")

        return "\n".join(lines)

    def _get_comprehensive_stats(self) -> str:
        """Комплексная статистика системы"""
        lines = ["📊 ПОЛНАЯ СТАТИСТИКА СИСТЕМЫ", "=" * 70]

        # Время работы
        uptime = time.time() - self.start_time
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)

        lines.append(f"\n⏱️ Время работы: {int(hours)}ч {int(minutes)}м {int(seconds)}с")
        lines.append(f"Взаимодействий: {self.interaction_count}")
        lines.append(f"Глубоких мыслей: {self.deep_thoughts_count}")
        lines.append(f"Паттернов найдено: {self.patterns_found}")

        # База данных
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM interactions")
            interactions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM facts WHERE decay_factor > 0.3")
            active_facts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM thoughts")
            thoughts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM patterns WHERE confidence > 0.5")
            patterns = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM goals WHERE status = 'active'")
            goals = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(satisfaction) FROM interactions WHERE timestamp > ?",
                           (time.time() - 86400,))
            avg_satisfaction = cursor.fetchone()[0] or 0.5

        lines.append(f"\n🗄️ База данных:")
        lines.append(f"  Взаимодействий: {interactions}")
        lines.append(f"  Активных фактов: {active_facts}")
        lines.append(f"  Мыслей: {thoughts}")
        lines.append(f"  Паттернов: {patterns}")
        lines.append(f"  Активных целей: {goals}")

        lines.append(f"\n📈 Качество:")
        lines.append(f"  Средняя удовлетворённость (24ч): {avg_satisfaction:.2f}")

        # Производительность мышления
        perf = self.thinker.get_performance_metrics()
        lines.append(f"\n🧠 Когнитивная производительность:")
        lines.append(f"  Cache hit rate: {perf['cache_hit_rate']:.1%}")
        lines.append(f"  Размер кэша: {perf['cache_size']}")
        lines.append(f"  История рассуждений: {perf['reasoning_history_size']}")

        lines.append(f"\n⚙️ Конфигурация:")
        lines.append(f"  Модель: {Config.MODEL}")
        lines.append(f"  Размер контекстного окна: {Config.CONTEXT_WINDOW_SIZE}")
        lines.append(f"  Интервал рефлексии: {Config.REFLECTION_INTERVAL}")
        lines.append(f"  Порог глубокого мышления: {Config.DEEP_THINKING_THRESHOLD}")

        return "\n".join(lines)

    def _get_context_summary(self) -> str:
        """Краткое резюме текущего контекста"""
        if not self.context_window:
            return ""

        summary = []
        for item in list(self.context_window)[-4:]:
            prefix = "П:" if item['type'] == 'user' else "Я:"
            summary.append(f"{prefix} {item['content'][:50]}...")

        return "\n".join(summary)

    def _categorize_input(self, text: str, features: Dict) -> str:
        """Улучшенная категоризация с ML-подходом"""
        text_lower = text.lower()

        # Правила категоризации
        categories = {
            'математика': ['сколько', 'посчитай', 'вычисли', '+', '-', '*', '/', '='],
            'память': ['запомни', 'сохрани', 'напомни', 'записывай'],
            'анализ': ['проанализируй', 'разбери', 'оцени', 'сравни'],
            'творчество': ['придумай', 'создай', 'сочини', 'напиши'],
            'планирование': ['план', 'распиши', 'как достичь', 'стратегия'],
            'объяснение': ['почему', 'как', 'зачем', 'объясни', 'расскажи'],
            'поиск': ['найди', 'покажи', 'где', 'ищи'],
            'вопрос': ['?']
        }

        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return 'диалог'


# ================= ГЛАВНАЯ ФУНКЦИЯ =================

async def main():
    """Основная функция"""
    print("\n" + "=" * 70)
    print("🚀 ЗАПУСК ПРОДВИНУТОГО КОГНИТИВНОГО АГЕНТА")
    print("=" * 70)

    try:
        agent = EnhancedAutonomousAgent()

        while True:
            try:
                user_input = input("\n💬 Вы: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['выход', 'exit', 'quit']:
                    print("\n👋 Завершение работы...")
                    print("\n" + agent._get_comprehensive_stats())
                    break

                print("\n🤖 Система:")
                response = await agent.process_input(user_input)

                if response == "SYSTEM_EXIT":
                    print("👋 Завершение...")
                    break

                # Вывод с эффектом печати
                for char in response:
                    print(char, end='', flush=True)
                    await asyncio.sleep(0.002)

                print("\n" + "-" * 70)

            except KeyboardInterrupt:
                print("\n\n🛑 Прервано пользователем")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"\n🚨 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


def run():
    """Точка входа"""
    asyncio.run(main())


if __name__ == "__main__":
    run()