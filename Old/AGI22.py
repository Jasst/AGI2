# coding: utf-8
"""
AGI_v28_Optimized.py — ОПТИМИЗИРОВАННАЯ АВТОНОМНАЯ СИСТЕМА

Улучшения:
1. Упрощенная архитектура без лишних зависимостей
2. Улучшенная производительность
3. Практичные механизмы памяти
4. Модульная структура для легкой модификации
5. Реалистичное "мышление" с ограничениями
6. Улучшенная обработка естественного языка
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
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
import logging
from contextlib import contextmanager
import random


# ================= КОНФИГУРАЦИЯ =================
class Config:
    """Упрощенная конфигурация"""

    # Пути
    ROOT = Path("./cognitive_system")
    ROOT.mkdir(exist_ok=True)

    # Базы данных
    DB_PATH = ROOT / "memory.db"
    CACHE_PATH = ROOT / "cache.json"
    LOG_PATH = ROOT / "system.log"

    # API
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "qwen/qwen-2.5-7b-instruct"  # Можно заменить на более быстрый вариант
    TIMEOUT = 30
    MAX_TOKENS = 600

    # Параметры системы
    REFLECTION_INTERVAL = 3
    MAX_MEMORY_ITEMS = 500
    THOUGHT_HISTORY_SIZE = 50
    GOAL_HISTORY_SIZE = 20

    # Безопасность (базовая)
    MAX_INPUT_LENGTH = 1000
    MAX_RESPONSE_LENGTH = 1500

    @classmethod
    def get_api_key(cls):
        """Получение API ключа из .env"""
        # 1. Проверяем переменные окружения
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            return key

        # 2. Проверяем файл .env в текущей директории
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

        # 3. Проверяем файл .env в домашней директории
        home_env = Path.home() / ".openrouter_env"
        if home_env.exists():
            try:
                with open(home_env, "r", encoding="utf-8") as f:
                    for line in f:
                        if "OPENROUTER_API_KEY" in line:
                            return line.split("=", 1)[1].strip()
            except:
                pass

        # 4. Запрашиваем у пользователя
        print("\n🔑 API ключ OpenRouter не найден.")
        print("Вы можете:")
        print("1. Создать файл .env с OPENROUTER_API_KEY=ваш_ключ")
        print("2. Установить переменную окружения")
        print("3. Ввести ключ сейчас (не будет сохранен)")

        choice = input("\nВыберите вариант (1/2/3): ").strip()

        if choice == "3":
            key = input("Введите API ключ: ").strip()
            if key:
                # Сохраняем временно для этой сессии
                os.environ["OPENROUTER_API_KEY"] = key
                return key

        raise ValueError("API ключ не найден. Создайте файл .env с OPENROUTER_API_KEY=ваш_ключ")


# ================= УТИЛИТЫ =================
def print_typing(text: str, delay: float = 0.003):
    """Эффект печати с задержкой"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def clean_text(text: str) -> str:
    """Очистка текста"""
    # Удаляем лишние пробелы, переносы
    text = re.sub(r'\s+', ' ', text.strip())
    # Убираем слишком длинные последовательности символов
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    return text


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Извлечение сущностей из текста"""
    entities = {
        'numbers': re.findall(r'\b\d+\b', text),
        'dates': re.findall(
            r'\b\d{1,2}[./]\d{1,2}[./]?\d{2,4}\b|\b(?:янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\b', text,
            re.IGNORECASE),
        'names': re.findall(r'\b(?:[А-Я][а-я]+)\b', text),
        'emails': re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text),
        'urls': re.findall(r'https?://\S+', text)
    }
    return {k: v for k, v in entities.items() if v}


def calculate_similarity(text1: str, text2: str) -> float:
    """Простой расчет схожести текстов"""
    # Базовый алгоритм на основе общих слов
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    common = words1.intersection(words2)
    return len(common) / max(len(words1), len(words2))


# ================= БАЗА ДАННЫХ =================
class MemoryDB:
    """Простая база данных для хранения памяти"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()
        self.connection_cache = None

    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для подключения"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        """Инициализация таблиц"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Таблица взаимодействий
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user_input TEXT NOT NULL,
                    system_response TEXT NOT NULL,
                    context TEXT,
                    emotion TEXT DEFAULT 'neutral',
                    category TEXT
                )
            ''')

            # Таблица фактов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    last_used REAL,
                    usage_count INTEGER DEFAULT 0,
                    UNIQUE(key, value)
                )
            ''')

            # Таблица мыслей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS thoughts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    thought_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    trigger TEXT,
                    importance REAL DEFAULT 0.5
                )
            ''')

            # Таблица целей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    description TEXT NOT NULL,
                    priority REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'active',
                    progress REAL DEFAULT 0.0,
                    next_action TEXT
                )
            ''')

            # Индексы для производительности
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_time ON interactions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoughts_time ON thoughts(timestamp)')

            conn.commit()

    # === Методы для взаимодействий ===
    def add_interaction(self, user_input: str, system_response: str,
                        context: str = "", emotion: str = "neutral",
                        category: str = "") -> int:
        """Добавление взаимодействия"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions 
                (timestamp, user_input, system_response, context, emotion, category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (time.time(), user_input, system_response, context, emotion, category))
            conn.commit()
            return cursor.lastrowid

    def get_recent_interactions(self, limit: int = 5) -> List[Dict]:
        """Получение последних взаимодействий"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM interactions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_interactions_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Получение взаимодействий по категории"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM interactions 
                WHERE category = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (category, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === Методы для фактов ===
    def add_fact(self, key: str, value: str, category: str = "", confidence: float = 1.0):
        """Добавление факта"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Проверяем, существует ли уже такой факт
            cursor.execute('''
                SELECT id FROM facts WHERE key = ? AND value = ?
            ''', (key, value))

            if cursor.fetchone():
                # Обновляем существующий
                cursor.execute('''
                    UPDATE facts 
                    SET confidence = ?, last_used = ?, usage_count = usage_count + 1 
                    WHERE key = ? AND value = ?
                ''', (confidence, time.time(), key, value))
            else:
                # Добавляем новый
                cursor.execute('''
                    INSERT INTO facts 
                    (key, value, category, confidence, created_at, last_used, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (key, value, category, confidence, time.time(), time.time(), 1))

            conn.commit()

    def get_facts(self, key: Optional[str] = None,
                  category: Optional[str] = None,
                  min_confidence: float = 0.3,
                  limit: int = 20) -> List[Dict]:
        """Получение фактов с фильтрацией"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM facts WHERE confidence >= ?"
            params = [min_confidence]

            if key:
                query += " AND key = ?"
                params.append(key)

            if category:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY last_used DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def search_facts(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Поиск фактов по тексту"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Простой текстовый поиск
            search_term = f"%{query_text}%"
            cursor.execute('''
                SELECT * FROM facts 
                WHERE key LIKE ? OR value LIKE ? 
                ORDER BY usage_count DESC, confidence DESC 
                LIMIT ?
            ''', (search_term, search_term, limit))

            return [dict(row) for row in cursor.fetchall()]

    # === Методы для мыслей ===
    def add_thought(self, thought_type: str, content: str,
                    trigger: str = "", importance: float = 0.5):
        """Добавление мысли"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO thoughts 
                (timestamp, thought_type, content, trigger, importance)
                VALUES (?, ?, ?, ?, ?)
            ''', (time.time(), thought_type, content, trigger, importance))
            conn.commit()

    def get_recent_thoughts(self, limit: int = 10,
                            thought_type: Optional[str] = None) -> List[Dict]:
        """Получение последних мыслей"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if thought_type:
                cursor.execute('''
                    SELECT * FROM thoughts 
                    WHERE thought_type = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (thought_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM thoughts 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]

    # === Методы для целей ===
    def add_goal(self, description: str, priority: float = 0.5,
                 next_action: str = "") -> int:
        """Добавление цели"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO goals 
                (created_at, description, priority, status, progress, next_action)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (time.time(), description, priority, 'active', 0.0, next_action))
            conn.commit()
            return cursor.lastrowid

    def get_active_goals(self, limit: int = 10) -> List[Dict]:
        """Получение активных целей"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM goals 
                WHERE status = 'active' 
                ORDER BY priority DESC, created_at DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def update_goal_progress(self, goal_id: int, progress: float):
        """Обновление прогресса цели"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE goals 
                SET progress = ? 
                WHERE id = ?
            ''', (progress, goal_id))
            conn.commit()

    def complete_goal(self, goal_id: int):
        """Завершение цели"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE goals 
                SET status = 'completed', progress = 1.0 
                WHERE id = ?
            ''', (goal_id,))
            conn.commit()

    # === Утилиты ===
    def get_statistics(self) -> Dict[str, int]:
        """Получение статистики базы данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Подсчет записей в каждой таблице
            tables = ['interactions', 'facts', 'thoughts', 'goals']
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[table] = cursor.fetchone()[0]

            # Средняя уверенность в фактах
            cursor.execute('SELECT AVG(confidence) FROM facts')
            avg_conf = cursor.fetchone()[0]
            stats['avg_fact_confidence'] = round(avg_conf or 0, 2)

            # Активные цели
            cursor.execute("SELECT COUNT(*) FROM goals WHERE status = 'active'")
            stats['active_goals'] = cursor.fetchone()[0]

            return stats


# ================= КЭШ ОТВЕТОВ =================
class ResponseCache:
    """Простой кэш ответов LLM"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.load()

    def _make_key(self, system_prompt: str, user_prompt: str) -> str:
        """Создание ключа для кэша"""
        # Используем хеш для экономии памяти
        content = f"{system_prompt[:200]}|{user_prompt[:300]}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Получение ответа из кэша"""
        key = self._make_key(system_prompt, user_prompt)

        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]

        return None

    def set(self, system_prompt: str, user_prompt: str, response: str):
        """Сохранение ответа в кэш"""
        key = self._make_key(system_prompt, user_prompt)

        self.cache[key] = response
        self.access_times[key] = time.time()

        # Очистка старых записей
        if len(self.cache) > self.max_size:
            # Удаляем наименее используемые
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:self.max_size // 4]]

            for k in keys_to_remove:
                self.cache.pop(k, None)
                self.access_times.pop(k, None)

        self.save()

    def save(self):
        """Сохранение кэша на диск"""
        try:
            data = {
                'cache': self.cache,
                'access_times': self.access_times
            }
            with open(Config.CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения кэша: {e}")

    def load(self):
        """Загрузка кэша с диска"""
        if Config.CACHE_PATH.exists():
            try:
                with open(Config.CACHE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
            except:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'usage_percent': len(self.cache) / self.max_size * 100
        }


# ================= СИСТЕМА МЫШЛЕНИЯ =================
class ThinkingSystem:
    """Упрощенная система мышления"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = ResponseCache()
        self.rate_limit = 2.0  # секунд между запросами
        self.last_request_time = 0

    async def _wait_for_rate_limit(self):
        """Ожидание ограничения скорости"""
        now = time.time()
        elapsed = now - self.last_request_time

        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)

        self.last_request_time = time.time()

    async def generate_thought(self, thought_type: str, context: str) -> Optional[str]:
        """Генерация мысли определенного типа"""

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
        return response if response and len(response) > 10 else None

    async def call_llm(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.7) -> str:
        """Вызов LLM с кэшированием"""

        # Проверка кэша
        cached = self.cache.get(system_prompt, user_prompt)
        if cached:
            return cached

        # Ожидание ограничения скорости
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
                "top_p": 0.9
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
                        self.cache.set(system_prompt, user_prompt, content)
                        return content
                    else:
                        error_text = await response.text()
                        return f"⚠️ Ошибка API: {response.status}"

        except asyncio.TimeoutError:
            return "⚠️ Таймаут запроса"
        except Exception as e:
            return f"⚠️ Ошибка: {str(e)[:100]}"


# ================= АВТОНОМНЫЙ АГЕНТ =================
class AutonomousAgent:
    """Упрощенный автономный агент"""

    def __init__(self):
        print("🧠 Автономный когнитивный агент v1.0\n")

        # Инициализация компонентов
        self.api_key = Config.get_api_key()
        self.db = MemoryDB(Config.DB_PATH)
        self.thinker = ThinkingSystem(self.api_key)

        # Состояние системы
        self.interaction_count = 0
        self.thoughts_generated = 0
        self.start_time = time.time()

        # Инициализация базовых целей
        self._init_default_goals()

        # Статистика
        self.stats = {
            'interactions': 0,
            'cache_hits': 0,
            'thoughts': 0,
            'facts_stored': 0
        }

        self.print_welcome()

    def _init_default_goals(self):
        """Инициализация целей по умолчанию"""
        default_goals = [
            ("Помогать пользователю решать задачи", 0.9),
            ("Запоминать важную информацию", 0.8),
            ("Учиться и адаптироваться", 0.7),
            ("Быть полезным и эффективным", 0.85)
        ]

        # Проверяем, есть ли уже цели
        existing = self.db.get_active_goals(limit=1)
        if not existing:
            for description, priority in default_goals:
                self.db.add_goal(description, priority)

    def print_welcome(self):
        """Вывод приветственного сообщения"""
        print("=" * 60)
        print("🤖 АВТОНОМНЫЙ КОГНИТИВНЫЙ АГЕНТ")
        print("=" * 60)

        stats = self.db.get_statistics()
        print(f"\n📊 Загружено:")
        print(f"   Взаимодействий: {stats.get('interactions', 0)}")
        print(f"   Фактов: {stats.get('facts', 0)}")
        print(f"   Мыслей: {stats.get('thoughts', 0)}")
        print(f"   Целей: {stats.get('goals', 0)}")

        print("\n💡 Доступные команды:")
        print("   • думай, мысли - активировать мышление")
        print("   • цели - показать текущие цели")
        print("   • факты - показать сохраненные факты")
        print("   • поиск [текст] - поиск в памяти")
        print("   • статистика - показать статистику")
        print("   • очистить кэш - очистить кэш ответов")
        print("\n   • выход, quit - завершить работу")
        print("=" * 60 + "\n")

    async def process_input(self, user_input: str) -> str:
        """Обработка пользовательского ввода"""
        start_time = time.time()
        self.interaction_count += 1

        # Обработка команд
        command_response = self._handle_command(user_input)
        if command_response:
            return command_response

        # Извлечение информации из ввода
        self._extract_information(user_input)

        # Генерация ответа
        response = await self._generate_response(user_input)

        # Сохранение взаимодействия
        self.db.add_interaction(
            user_input=user_input[:500],
            system_response=response[:500],
            context=self._get_context_summary(),
            category=self._categorize_input(user_input)
        )

        # Периодическое автономное мышление
        if self.interaction_count % Config.REFLECTION_INTERVAL == 0:
            await self._autonomous_thinking()

        # Логирование производительности
        duration = time.time() - start_time
        if duration > 1.0:
            print(f"⏱️ Время обработки: {duration:.2f}с")

        return response

    def _handle_command(self, text: str) -> Optional[str]:
        """Обработка специальных команд"""
        text_lower = text.lower().strip()

        if text_lower in ['думай', 'подумай', 'мысли']:
            return "🧠 Запускаю процесс мышления... (используй 'статистика' для просмотра)"

        elif text_lower == 'цели':
            return self._format_goals()

        elif text_lower == 'факты':
            return self._format_facts()

        elif text_lower.startswith('поиск '):
            query = text_lower[6:].strip()
            return self._search_memory(query)

        elif text_lower == 'статистика':
            return self._get_system_stats()

        elif text_lower == 'очистить кэш':
            self.thinker.cache = ResponseCache()
            return "✅ Кэш очищен"

        elif text_lower in ['выход', 'exit', 'quit', 'q']:
            return "SYSTEM_EXIT"

        return None

    def _extract_information(self, text: str):
        """Извлечение и сохранение информации из текста"""
        entities = extract_entities(text)

        # Сохраняем числа как факты
        for number in entities.get('numbers', []):
            self.db.add_fact('число', number, 'информация')

        # Сохраняем имена
        for name in entities.get('names', []):
            if len(name) > 2:  # Игнорируем короткие "слова"
                self.db.add_fact('имя', name, 'персона')

        # Извлекаем факты из утверждений
        if re.search(r'(?:это|равно|составляет|запомни)\s+\d+', text.lower()):
            # Находим пары ключ-значение
            patterns = [
                (r'(\w+)\s+(?:составляет|равно|это)\s+(\d+)', 'значение'),
                (r'запомни\s+что\s+(\w+)\s+—\s+([^.,]+)', 'факт'),
                (r'(\w+)\s+=\s+([^.,]+)', 'равенство')
            ]

            for pattern, category in patterns:
                matches = re.findall(pattern, text.lower())
                for key, value in matches:
                    if len(key) > 2 and len(value) > 1:
                        self.db.add_fact(key.strip(), value.strip(), category)

    def _get_context_summary(self) -> str:
        """Получение краткого контекста для сохранения"""
        recent = self.db.get_recent_interactions(3)
        if not recent:
            return ""

        summary = []
        for i, interaction in enumerate(recent[-3:], 1):
            summary.append(f"{i}. П: {interaction['user_input'][:50]}...")
            summary.append(f"   Я: {interaction['system_response'][:50]}...")

        return "\n".join(summary)

    def _categorize_input(self, text: str) -> str:
        """Категоризация ввода"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['сколько', 'сколько будет', 'посчитай']):
            return 'математика'
        elif any(word in text_lower for word in ['запомни', 'сохрани', 'напомни']):
            return 'память'
        elif any(word in text_lower for word in ['почему', 'как', 'зачем', 'объясни']):
            return 'объяснение'
        elif any(word in text_lower for word in ['сделай', 'найди', 'создай', 'напиши']):
            return 'действие'
        elif '?' in text:
            return 'вопрос'
        else:
            return 'диалог'

    async def _generate_response(self, user_input: str) -> str:
        """Генерация ответа на основе контекста"""

        # Собираем контекст
        context_parts = []

        # Недавние взаимодействия
        recent = self.db.get_recent_interactions(2)
        if recent:
            context_parts.append("Недавний диалог:")
            for interaction in reversed(recent):
                context_parts.append(f"Вы: {interaction['user_input'][:80]}")
                context_parts.append(f"Я: {interaction['system_response'][:80]}")

        # Релевантные факты
        facts = self.db.search_facts(user_input, limit=3)
        if facts:
            context_parts.append("\nРелевантные факты:")
            for fact in facts:
                context_parts.append(f"- {fact['key']}: {fact['value']}")

        # Активные цели
        goals = self.db.get_active_goals(limit=2)
        if goals:
            context_parts.append("\nТекущие цели:")
            for goal in goals:
                progress_bar = "█" * int(goal['progress'] * 10) + "░" * (10 - int(goal['progress'] * 10))
                context_parts.append(f"- {goal['description'][:50]} [{progress_bar}]")

        context = "\n".join(context_parts) if context_parts else "Нет контекста"

        # Системный промпт
        system_prompt = f"""Ты — автономный когнитивный агент. 

Твои принципы:
1. Будь полезным и конкретным
2. Используй информацию из памяти, если она есть
3. Если не знаешь ответа — честно говори об этом
4. Будь естественным в общении

Контекст из памяти:
{context}

Отвечай на русском языке, будь краток и точен."""

        # Вызов LLM
        response = await self.thinker.call_llm(system_prompt, user_input)

        # Очистка ответа
        response = clean_text(response)

        # Ограничение длины
        if len(response) > Config.MAX_RESPONSE_LENGTH:
            response = response[:Config.MAX_RESPONSE_LENGTH] + "..."

        return response

    async def _autonomous_thinking(self):
        """Автономный процесс мышления"""
        print("\n💭 [Автономное мышление...]", flush=True)

        # Получаем данные для размышлений
        recent_interactions = self.db.get_recent_interactions(5)
        recent_thoughts = self.db.get_recent_thoughts(3)

        if len(recent_interactions) < 2:
            print("   💭 Мало данных для размышлений")
            return

        # Готовим контекст
        context_lines = []
        for i, interaction in enumerate(recent_interactions[-3:], 1):
            context_lines.append(f"{i}. {interaction['user_input'][:50]}... → {interaction['system_response'][:50]}...")

        context = "\n".join(context_lines)

        # Типы мыслей для генерации
        thought_types = ['рефлексия', 'наблюдение', 'обучение']
        selected_type = random.choice(thought_types)

        # Генерируем мысль
        thought_content = await self.thinker.generate_thought(selected_type, context)

        if thought_content and len(thought_content) > 20:
            # Сохраняем мысль
            self.db.add_thought(
                thought_type=selected_type,
                content=thought_content[:300],
                trigger="автономное_мышление",
                importance=0.6
            )

            self.thoughts_generated += 1
            print(f"   💡 [{selected_type}] {thought_content[:70]}...")
        else:
            print("   💭 Не удалось сгенерировать мысль")

    def _format_goals(self) -> str:
        """Форматирование списка целей"""
        goals = self.db.get_active_goals(10)

        if not goals:
            return "Нет активных целей."

        lines = ["🎯 АКТИВНЫЕ ЦЕЛИ:\n"]

        for i, goal in enumerate(goals, 1):
            progress = goal['progress']
            progress_bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))

            lines.append(f"{i}. {goal['description']}")
            lines.append(f"   Приоритет: {goal['priority']:.1f} | Прогресс: [{progress_bar}]")

            if goal.get('next_action'):
                lines.append(f"   След. шаг: {goal['next_action']}")

            lines.append("")

        return "\n".join(lines)

    def _format_facts(self) -> str:
        """Форматирование фактов"""
        facts = self.db.get_facts(limit=20)

        if not facts:
            return "Нет сохранённых фактов."

        # Группируем по категориям
        categories = defaultdict(list)
        for fact in facts:
            categories[fact.get('category', 'разное')].append(fact)

        lines = ["📚 СОХРАНЁННЫЕ ФАКТЫ:\n"]

        for category, category_facts in categories.items():
            lines.append(f"📌 {category.upper()}:")

            for fact in category_facts[:5]:  # Показываем по 5 из каждой категории
                confidence_stars = "★" * int(fact['confidence'] * 5)
                lines.append(f"  • {fact['key']}: {fact['value']} [{confidence_stars}]")

            if len(category_facts) > 5:
                lines.append(f"  ... и ещё {len(category_facts) - 5}")

            lines.append("")

        return "\n".join(lines)

    def _search_memory(self, query: str) -> str:
        """Поиск в памяти"""
        if not query or len(query) < 2:
            return "Укажите запрос для поиска (минимум 2 символа)."

        # Ищем в фактах
        facts = self.db.search_facts(query, limit=5)

        # Ищем во взаимодействиях
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            search_term = f"%{query}%"
            cursor.execute('''
                SELECT user_input, system_response 
                FROM interactions 
                WHERE user_input LIKE ? OR system_response LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT 3
            ''', (search_term, search_term))
            interactions = cursor.fetchall()

        lines = [f"🔍 РЕЗУЛЬТАТЫ ПОИСКА ДЛЯ '{query}':\n"]

        if facts:
            lines.append("📌 Факты:")
            for fact in facts:
                lines.append(f"  • {fact['key']}: {fact['value']}")
            lines.append("")

        if interactions:
            lines.append("💬 Диалоги:")
            for interaction in interactions:
                user_text = interaction[0]
                system_text = interaction[1]

                # Подсвечиваем найденное
                user_highlighted = user_text.replace(query, f"**{query}**")
                system_highlighted = system_text.replace(query, f"**{query}**")

                lines.append(f"  Вы: {user_highlighted[:80]}...")
                lines.append(f"  Я: {system_highlighted[:80]}...")
                lines.append("")

        if not facts and not interactions:
            return f"По запросу '{query}' ничего не найдено."

        return "\n".join(lines)

    def _get_system_stats(self) -> str:
        """Получение статистики системы"""
        db_stats = self.db.get_statistics()
        cache_stats = self.thinker.cache.get_stats()

        # Время работы
        uptime = time.time() - self.start_time
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = [
            "📊 СТАТИСТИКА СИСТЕМЫ",
            "=" * 40,
            f"Время работы: {int(hours)}ч {int(minutes)}м {int(seconds)}с",
            f"Взаимодействий: {self.interaction_count}",
            f"Сгенерировано мыслей: {self.thoughts_generated}",
            "",
            "🗄️ База данных:",
            f"  Взаимодействий: {db_stats.get('interactions', 0)}",
            f"  Фактов: {db_stats.get('facts', 0)}",
            f"  Мыслей: {db_stats.get('thoughts', 0)}",
            f"  Целей: {db_stats.get('goals', 0)}",
            f"  Активных целей: {db_stats.get('active_goals', 0)}",
            "",
            "💾 Кэш ответов:",
            f"  Размер: {cache_stats['size']} / {cache_stats['max_size']}",
            f"  Заполнение: {cache_stats['usage_percent']:.1f}%",
            "",
            "⚙️ Конфигурация:",
            f"  Модель: {Config.MODEL}",
            f"  Таймаут: {Config.TIMEOUT}с",
            f"  Лимит токенов: {Config.MAX_TOKENS}"
        ]

        return "\n".join(lines)

    def save_state(self):
        """Сохранение состояния системы"""
        print("💾 Сохранение состояния...")
        self.thinker.cache.save()
        print("✅ Состояние сохранено")


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
async def main():
    """Основная асинхронная функция"""
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК АВТОНОМНОГО КОГНИТИВНОГО АГЕНТА")
    print("=" * 60)

    try:
        # Создаем агента
        agent = AutonomousAgent()

        # Основной цикл
        while True:
            try:
                # Ввод пользователя
                user_input = input("\n💬 Вы: ").strip()

                if not user_input:
                    continue

                # Проверка на выход
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("\n👋 Завершение работы...")
                    agent.save_state()

                    # Вывод статистики
                    print("\n" + "=" * 60)
                    print(agent._get_system_stats())
                    print("=" * 60)
                    break

                # Обработка
                print("\n🤖 Система:")
                response = await agent.process_input(user_input)

                # Проверка на команду выхода из системы
                if response == "SYSTEM_EXIT":
                    print("👋 Завершение работы...")
                    agent.save_state()
                    break

                # Вывод с эффектом печати
                print_typing(response, delay=0.002)
                print("\n" + "-" * 60)

            except KeyboardInterrupt:
                print("\n\n🛑 Прервано пользователем")
                agent.save_state()
                break

            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"\n🚨 Критическая ошибка при запуске: {e}")
        print("Проверьте наличие файла .env с API ключом")
        print("Формат файла .env: OPENROUTER_API_KEY=ваш_ключ")


def run():
    """Точка входа"""
    if sys.platform == "win32":
        # Включение цветов в Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass

    asyncio.run(main())


if __name__ == "__main__":
    run()