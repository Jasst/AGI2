#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v8.0 - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ
✅ РАБОЧАЯ краткосрочная и долгосрочная память с автосохранением
✅ РАБОЧИЙ веб-поиск с валидацией и синтезом результатов
✅ Автоматический перенос важных данных в долгосрочную память
✅ Анализ когнитивных паттернов и адаптация
✅ Мета-когнитивный цикл с саморефлексией
"""
import os
import json
import re
import asyncio
import aiohttp
import traceback
import hashlib

import time
from datetime import datetime, timedelta
from pathlib import Path  # ← ДОБАВИТЬ ЭТУ СТРОКУ
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Импорт для парсинга веб-страниц
try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️ BeautifulSoup не установлен. Парсинг веб-страниц будет ограничен.")

# DuckDuckGo Search (поддержка Python 3.13)
DDGS_AVAILABLE = False
DDGS_ASYNC = False
try:
    # Пробуем асинхронную версию (для Python 3.11, 3.12)
    from duckduckgo_search import AsyncDDGS

    DDGS_AVAILABLE = True
    DDGS_ASYNC = True
    print("✅ Async DuckDuckGo Search доступен")
except ImportError:
    try:
        # Fallback: синхронная версия (для Python 3.13)
        from duckduckgo_search import DDGS

        DDGS_AVAILABLE = True
        DDGS_ASYNC = False
        print("✅ DuckDuckGo Search доступен (синхронный режим)")
        print("   ⚠️ Для лучшей производительности используйте Python 3.12")
    except ImportError:
        print(f"⚠️ DuckDuckGo Search не установлен")
        print(f"   Установите: pip install duckduckgo-search")

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
if not TELEGRAM_TOKEN:
    raise ValueError("❌ ОШИБКА: Не найден TELEGRAM_TOKEN в .env!")

LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

# Директории
CORES_DIR = "dynamic_cores"
MEMORY_DIR = "brain_memory"
USER_FILES_DIR = "user_files"
CACHE_DIR = "cache"

for directory in [CORES_DIR, MEMORY_DIR, USER_FILES_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Файлы
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
CORE_PERFORMANCE_LOG = os.path.join(MEMORY_DIR, "core_performance.json")
WEB_CACHE_FILE = os.path.join(CACHE_DIR, "web_search_cache.json")


# ==================== СИСТЕМА АВТОМАТИЧЕСКОЙ ПАМЯТИ v2.0 ====================
class AutoMemorySystem:
    """
    🧠 АВТОМАТИЧЕСКАЯ СИСТЕМА ПАМЯТИ С КОГНИТИВНЫМ АНАЛИЗОМ
    - Автоматический анализ каждого сообщения
    - Извлечение фактов, предпочтений, паттернов
    - Автоматический перенос из краткосрочной в долгосрочную
    - Адаптация к стилю пользователя
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.memory_dir, exist_ok=True)

        # Файлы памяти
        self.short_term_file = os.path.join(self.memory_dir, "short_term.json")
        self.long_term_file = os.path.join(self.memory_dir, "long_term.json")
        self.patterns_file = os.path.join(self.memory_dir, "patterns.json")
        self.preferences_file = os.path.join(self.memory_dir, "preferences.json")
        self.metadata_file = os.path.join(self.memory_dir, "metadata.json")

        # Загрузка данных
        self.short_term = self._load_json(self.short_term_file, [])
        self.long_term = self._load_json(self.long_term_file, [])
        self.patterns = self._load_json(self.patterns_file, {
            'communication_style': 'neutral',
            'frequent_topics': [],
            'emotional_markers': {},
            'time_preferences': {},
            'interaction_patterns': {},
            'last_pattern_update': datetime.now().isoformat()
        })
        self.preferences = self._load_json(self.preferences_file, {
            'explicit': {},  # Явно указанные пользователем
            'inferred': {},  # Выведенные из поведения
            'confirmed': {}  # Подтверждённые
        })
        self.metadata = self._load_json(self.metadata_file, {
            'total_messages': 0,
            'facts_extracted': 0,
            'patterns_identified': 0,
            'auto_transfers': 0,  # Счётчик автопереносов
            'first_interaction': datetime.now().isoformat(),
            'last_interaction': datetime.now().isoformat(),
            'session_count': 0
        })

        # Счётчик для автосохранения
        self._changes_count = 0
        self._autosave_threshold = 3  # Сохранять каждые 3 изменения

        print(f"🧠 Память загружена для {user_id}: "
              f"ST={len(self.short_term)}, LT={len(self.long_term)}, "
              f"фактов={self.metadata['facts_extracted']}")

    # ==================== АВТОМАТИЧЕСКИЙ АНАЛИЗ СООБЩЕНИЙ ====================
    async def analyze_and_store_message(self, role: str, content: str,
                                        llm_caller: callable = None) -> Dict[str, Any]:
        """
        Автоматически анализирует сообщение и извлекает:
        - Факты о пользователе
        - Предпочтения
        - Эмоциональные маркеры
        - Паттерны поведения
        """
        timestamp = datetime.now()

        # 1. Добавляем в краткосрочную память
        entry = {
            'role': role,
            'content': content,
            'timestamp': timestamp.isoformat(),
            'word_count': len(content.split()),
            'has_question': '?' in content,
            'has_numbers': bool(re.search(r'\d', content)),
            'extracted_facts': [],
            'detected_emotions': [],
            'topic_tags': []
        }

        # 2. Базовый анализ (быстрый, без LLM)
        analysis = self._quick_analysis(content, role)
        entry.update(analysis)

        # 3. Глубокий анализ через LLM (если доступен и это пользователь)
        if llm_caller and role == 'user' and len(content) > 10:
            deep_analysis = await self._deep_analysis(content, llm_caller)
            entry['extracted_facts'].extend(deep_analysis.get('facts', []))
            entry['detected_emotions'].extend(deep_analysis.get('emotions', []))
            entry['topic_tags'].extend(deep_analysis.get('topics', []))

            # Сохраняем важные факты в долгосрочную память
            for fact in deep_analysis.get('important_facts', []):
                await self.add_to_long_term(
                    fact['content'],
                    category=fact.get('category', 'general'),
                    importance=fact.get('importance', 0.7),
                    source='auto_extraction'
                )

        self.short_term.append(entry)

        # 4. Ограничиваем размер краткосрочной памяти
        if len(self.short_term) > 50:
            # Переносим старые важные сообщения в долгосрочную
            await self._auto_transfer_to_long_term()
            self.short_term = self.short_term[-30:]

        # 5. Обновляем паттерны
        await self._update_patterns(entry)

        # 6. Метаданные
        self.metadata['total_messages'] += 1
        self.metadata['last_interaction'] = timestamp.isoformat()

        # 7. Автосохранение
        await self._auto_save()

        return entry

    def _quick_analysis(self, content: str, role: str) -> Dict[str, Any]:
        """Быстрый анализ без LLM"""
        result = {
            'sentiment': 'neutral',
            'urgency': 'normal',
            'topics': [],
            'contains_personal_info': False
        }

        content_lower = content.lower()

        # Эмоциональный тон
        positive_words = ['спасибо', 'отлично', 'хорошо', 'замечательно', 'супер', 'класс']
        negative_words = ['плохо', 'ужасно', 'не нравится', 'проблема', 'ошибка']

        if any(word in content_lower for word in positive_words):
            result['sentiment'] = 'positive'
        elif any(word in content_lower for word in negative_words):
            result['sentiment'] = 'negative'

        # Срочность
        urgent_markers = ['срочно', 'быстро', 'немедленно', 'сейчас же', 'прямо сейчас']
        if any(marker in content_lower for marker in urgent_markers):
            result['urgency'] = 'high'

        # Топики
        topic_keywords = {
            'технологии': ['код', 'программ', 'компьютер', 'api', 'данные'],
            'финансы': ['деньги', 'курс', 'доллар', 'евро', 'биткоин', 'рубл'],
            'погода': ['погода', 'температура', 'дождь', 'снег'],
            'время': ['время', 'дата', 'день', 'час', 'минут'],
            'личное': ['я', 'мне', 'мой', 'моя', 'моё']
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                result['topics'].append(topic)

        # Персональная информация
        personal_markers = ['я живу', 'мне нравится', 'я предпочитаю', 'мой любимый']
        if any(marker in content_lower for marker in personal_markers):
            result['contains_personal_info'] = True

        return result

    async def _deep_analysis(self, content: str, llm_caller: callable) -> Dict[str, Any]:
        """Глубокий анализ через LLM для извлечения фактов"""
        prompt = f"""Проанализируй сообщение пользователя и извлеки структурированную информацию.

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:
{content}

ЗАДАЧА: Верни ТОЛЬКО JSON (без markdown, без пояснений) со следующей структурой:
{{
  "facts": ["факт1", "факт2"],
  "important_facts": [
    {{"content": "важный факт", "category": "personal|preference|habit|knowledge", "importance": 0.8}}
  ],
  "emotions": ["радость", "любопытство"],
  "topics": ["технологии", "спорт"],
  "preferences": {{
    "explicit": {{"тип": "значение"}},
    "inferred": {{"тип": "предположение"}}
  }},
  "requires_long_term_storage": true|false
}}

ПРАВИЛА:
1. facts - любые упомянутые факты
2. important_facts - только важная информация о пользователе (имя, увлечения, предпочтения, цели)
3. emotions - обнаруженные эмоции
4. topics - темы разговора
5. preferences - явные и выведенные предпочтения
6. requires_long_term_storage - true если есть что запомнить надолго

JSON:"""

        try:
            response = await llm_caller(prompt, temperature=0.2, max_tokens=500)

            # Извлекаем JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
        except Exception as e:
            print(f"⚠️ Ошибка глубокого анализа: {e}")

        return {'facts': [], 'important_facts': [], 'emotions': [], 'topics': [], 'preferences': {}}

    async def _auto_transfer_to_long_term(self):
        """Автоматический перенос важных сообщений в долгосрочную память"""
        transferred = 0

        # Анализируем старые сообщения (за пределами последних 30)
        old_messages = self.short_term[:-30] if len(self.short_term) > 30 else []

        for msg in old_messages:
            # Критерии важности
            is_important = (
                    msg.get('contains_personal_info', False) or
                    len(msg.get('extracted_facts', [])) > 0 or
                    msg.get('urgency') == 'high' or
                    msg.get('word_count', 0) > 50
            )

            if is_important and msg['role'] == 'user':
                # Создаём запись в долгосрочной памяти
                summary = msg['content'][:200] + ('...' if len(msg['content']) > 200 else '')

                await self.add_to_long_term(
                    content=summary,
                    category='conversation_history',
                    importance=0.6,
                    source='auto_transfer',
                    metadata={
                        'original_timestamp': msg['timestamp'],
                        'sentiment': msg.get('sentiment'),
                        'topics': msg.get('topic_tags', [])
                    }
                )
                transferred += 1

        if transferred > 0:
            self.metadata['auto_transfers'] += transferred
            print(f"📤 Автоматически перенесено в долгосрочную память: {transferred} записей")

    async def _update_patterns(self, entry: Dict[str, Any]):
        """Обновление паттернов поведения"""
        if entry['role'] != 'user':
            return

        # 1. Обновляем частые топики
        for topic in entry.get('topic_tags', []):
            if topic not in self.patterns['frequent_topics']:
                self.patterns['frequent_topics'].append(topic)

        # Ограничиваем до топ-10
        if len(self.patterns['frequent_topics']) > 10:
            self.patterns['frequent_topics'] = self.patterns['frequent_topics'][:10]

        # 2. Время активности
        hour = datetime.fromisoformat(entry['timestamp']).hour
        time_slot = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening' if 18 <= hour < 23 else 'night'

        if 'time_preferences' not in self.patterns:
            self.patterns['time_preferences'] = {}

        self.patterns['time_preferences'][time_slot] = self.patterns['time_preferences'].get(time_slot, 0) + 1

        # 3. Эмоциональные маркеры
        sentiment = entry.get('sentiment', 'neutral')
        if sentiment not in self.patterns['emotional_markers']:
            self.patterns['emotional_markers'][sentiment] = 0
        self.patterns['emotional_markers'][sentiment] += 1

        # 4. Стиль общения (на основе длины сообщений)
        avg_length = entry.get('word_count', 0)
        if avg_length < 10:
            style = 'concise'
        elif avg_length < 30:
            style = 'balanced'
        else:
            style = 'detailed'

        self.patterns['communication_style'] = style
        self.patterns['last_pattern_update'] = datetime.now().isoformat()

        self.metadata['patterns_identified'] += 1

    # ==================== ДОЛГОСРОЧНАЯ ПАМЯТЬ ====================
    async def add_to_long_term(self, content: str, category: str = 'general',
                               importance: float = 0.7, source: str = 'manual',
                               metadata: Optional[Dict] = None):
        """Добавление в долгосрочную память с дедупликацией"""
        # Проверка на дубликаты
        content_normalized = self._normalize_text(content)

        for existing in self.long_term:
            existing_normalized = self._normalize_text(existing['content'])
            similarity = self._text_similarity(content_normalized, existing_normalized)

            if similarity > 0.85:
                # Обновляем существующую запись
                existing['importance'] = max(existing.get('importance', 0.5), importance)
                existing['last_updated'] = datetime.now().isoformat()
                existing['access_count'] = existing.get('access_count', 0) + 1
                await self._auto_save()
                return

        # Новая запись
        entry = {
            'id': hashlib.md5(content.encode()).hexdigest()[:12],
            'content': content,
            'category': category,
            'importance': importance,
            'source': source,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'access_count': 1,
            'metadata': metadata or {}
        }

        self.long_term.append(entry)

        # Сортируем по важности
        self.long_term.sort(key=lambda x: x['importance'], reverse=True)

        # Ограничиваем размер (топ-200)
        if len(self.long_term) > 200:
            self.long_term = self.long_term[:200]

        self.metadata['facts_extracted'] += 1
        await self._auto_save()

    async def search_long_term(self, query: str, limit: int = 5,
                               min_similarity: float = 0.3) -> List[Dict]:
        """Поиск в долгосрочной памяти с релевантностью"""
        if not self.long_term:
            return []

        query_normalized = self._normalize_text(query)
        scored_results = []

        for memory in self.long_term:
            content_normalized = self._normalize_text(memory['content'])

            # Вычисляем релевантность
            similarity = self._text_similarity(query_normalized, content_normalized)
            importance = memory.get('importance', 0.5)

            # Свежесть (новые записи приоритетнее)
            try:
                created = datetime.fromisoformat(memory['created_at'])
                days_old = (datetime.now() - created).days
                freshness = max(0.1, 1.0 - days_old / 180)  # 6 месяцев
            except:
                freshness = 0.5

            # Итоговый score
            score = (
                    similarity * 0.5 +
                    importance * 0.3 +
                    freshness * 0.2
            )

            if score >= min_similarity:
                scored_results.append((score, memory))
                # Увеличиваем счётчик доступа
                memory['access_count'] = memory.get('access_count', 0) + 1
                memory['last_accessed'] = datetime.now().isoformat()

        # Сортируем и возвращаем
        scored_results.sort(key=lambda x: x[0], reverse=True)

        await self._auto_save()
        return [mem for _, mem in scored_results[:limit]]

    # ==================== КОНТЕКСТ ДЛЯ LLM ====================
    async def get_enhanced_context(self, query: str, short_term_limit: int = 10,
                                   long_term_limit: int = 5) -> str:
        """Формирует расширенный контекст для LLM"""
        # 1. Релевантные факты из долгосрочной памяти
        relevant_facts = await self.search_long_term(query, limit=long_term_limit)

        facts_text = ""
        if relevant_facts:
            facts_text = "# ВАЖНЫЕ ФАКТЫ О ПОЛЬЗОВАТЕЛЕ (из долгосрочной памяти):\n"
            for i, mem in enumerate(relevant_facts, 1):
                category = mem.get('category', 'general')
                importance = mem.get('importance', 0.5)
                stars = "⭐" * min(5, int(importance * 5))
                facts_text += f"{i}. [{category.upper()}] {stars}\n   {mem['content']}\n"

        # 2. Недавняя история
        recent = self.short_term[-short_term_limit:] if self.short_term else []
        history_text = ""
        if recent:
            history_text = "\n# НЕДАВНЯЯ ИСТОРИЯ ДИАЛОГА:\n"
            for msg in recent:
                time = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M')
                role = "Пользователь" if msg['role'] == 'user' else "Бот"
                history_text += f"[{time}] {role}: {msg['content'][:150]}\n"

        # 3. Паттерны и предпочтения
        patterns_text = "\n# ПАТТЕРНЫ ПОВЕДЕНИЯ ПОЛЬЗОВАТЕЛЯ:\n"
        patterns_text += f"• Стиль общения: {self.patterns.get('communication_style', 'neutral')}\n"

        if self.patterns.get('frequent_topics'):
            topics = ", ".join(self.patterns['frequent_topics'][:5])
            patterns_text += f"• Частые темы: {topics}\n"

        if self.patterns.get('time_preferences'):
            most_active = max(self.patterns['time_preferences'].items(), key=lambda x: x[1])[0]
            patterns_text += f"• Наиболее активен: {most_active}\n"

        if self.preferences.get('confirmed'):
            prefs = ", ".join([f"{k}={v}" for k, v in list(self.preferences['confirmed'].items())[:3]])
            patterns_text += f"• Подтверждённые предпочтения: {prefs}\n"

        # 4. Текущий запрос
        query_text = f"\n# ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}\n"

        return facts_text + history_text + patterns_text + query_text

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для сравнения"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Вычисление схожести текстов (Jaccard + бонусы)"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union)

        # Бонус за общие числа
        nums1 = set(re.findall(r'\d+', text1))
        nums2 = set(re.findall(r'\d+', text2))
        if nums1 & nums2:
            jaccard += 0.1

        return min(jaccard, 1.0)

    async def _auto_save(self):
        """Автоматическое сохранение при изменениях"""
        self._changes_count += 1

        if self._changes_count >= self._autosave_threshold:
            self._save_all()
            self._changes_count = 0

    def _save_all(self):
        """Сохранение всех данных на диск"""
        try:
            self._save_json(self.short_term_file, self.short_term)
            self._save_json(self.long_term_file, self.long_term)
            self._save_json(self.patterns_file, self.patterns)
            self._save_json(self.preferences_file, self.preferences)
            self._save_json(self.metadata_file, self.metadata)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения памяти: {e}")

    def _load_json(self, filepath: str, default: Any) -> Any:
        """Загрузка JSON с защитой от повреждений"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)

        except Exception as e:
            print(f"⚠️ Ошибка загрузки {filepath}: {e}")

            # 🔥 backup повреждённого файла
            try:
                os.rename(filepath, filepath + ".broken")
            except:
                pass

        return default

    def _save_json(self, filepath: str, data: Any):
        """НАДЁЖНОЕ сохранение JSON (без потерь)"""
        try:
            tmp_file = filepath + ".tmp"

            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # 🔥 гарантирует запись на диск

            # атомарная замена
            os.replace(tmp_file, filepath)

        except Exception as e:
            print(f"⚠️ Ошибка записи {filepath}: {e}")

    def get_stats(self) -> Dict:
        """Статистика памяти"""
        return {
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'total_messages': self.metadata['total_messages'],
            'facts_extracted': self.metadata['facts_extracted'],
            'patterns_identified': self.metadata['patterns_identified'],
            'auto_transfers': self.metadata['auto_transfers'],
            'frequent_topics': self.patterns.get('frequent_topics', []),
            'communication_style': self.patterns.get('communication_style', 'neutral')
        }

# ==================== УЛУЧШЕННЫЙ ВЕБ-ПОИСК С ВАЛИДАЦИЕЙ ====================
class ValidatedWebSearchCore:
    """
    🔥 ВЕБ-ПОИСК v2 (БЕЗ API, УСИЛЕННЫЙ)

    Возможности:
    - Multi-search (несколько запросов)
    - Перекрёстная проверка фактов
    - Умная фильтрация мусора
    - Дожим контента через загрузку страницы
    - Кэширование
    """

    def __init__(self):
        self.ddgs = AsyncDDGS() if DDGS_ASYNC else DDGS() if DDGS_AVAILABLE else None
        self.session = None
        self.cache = {}

        self.domain_trust = {
            "gov": 0.95,
            "edu": 0.95,
            "wikipedia.org": 0.9,
            "bbc.com": 0.85,
            "reuters.com": 0.9,
            "bloomberg.com": 0.9,
            "cbr.ru": 0.98,
            "gismeteo.ru": 0.9,
            "yandex.ru": 0.85,
        }

    async def initialize(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=aiohttp.ClientTimeout(total=20)
            )

    async def search_and_validate(self, query: str, llm_caller):
        if not self.ddgs:
            return {"success": False, "answer": "DDG не установлен", "confidence": 0.0}

        await self.initialize()

        # 🔥 multi search
        queries = self._expand_query(query)

        all_results = []
        for q in queries:
            results = await self._search(q)
            all_results.extend(results)

        # удаляем дубли
        unique = {}
        for r in all_results:
            unique[r["url"]] = r

        results = list(unique.values())

        # валидируем
        validated = await self._validate(results)

        if not validated:
            # fallback
            fallback_q = query + " что это подробности"
            results = await self._search(fallback_q)
            validated = await self._validate(results)

        if not validated:
            return {"success": False, "answer": "Ничего не найдено", "confidence": 0.0}

        # 🔥 cross-check
        validated = self._cross_check(validated)

        # синтез
        answer = await self._synthesize(query, validated, llm_caller)

        return {
            "success": True,
            "answer": answer,
            "confidence": 0.8,  # 👈 ДОБАВЬ ЭТО
            "sources": validated[:3]
        }

    # =============================

    def _expand_query(self, query):
        q = query.lower()

        variations = [q]

        # убираем мусор
        q_clean = re.sub(r"(найди|что|такое|покажи|расскажи)", "", q).strip()

        variations.append(q_clean)
        variations.append(q_clean + " факты")
        variations.append(q_clean + " объяснение")

        if "курс" in q:
            variations.append(q_clean + " официальный")
        if "погода" in q:
            variations.append(q_clean + " прогноз")

        return list(set(variations))

    # =============================

    async def _search(self, query):
        results = []

        try:
            if DDGS_ASYNC:
                raw = await self.ddgs.text(query, max_results=5)
            else:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None, lambda: list(self.ddgs.text(query, max_results=5))
                )

            for r in raw:
                url = r.get("href") or r.get("link") or ""
                title = r.get("title") or ""
                body = r.get("body") or ""

                if not url:
                    continue

                url = self._normalize_url(url)

                # дожим контента
                if len(body) < 100:
                    body = await self._fetch(url) or body

                results.append({
                    "url": url,
                    "title": title,
                    "text": body[:2000]
                })

        except Exception as e:
            print("search error:", e)

        return results

    # =============================

    async def _fetch(self, url):
        try:
            if url in self.cache:
                return self.cache[url]

            async with self.session.get(url) as r:
                if r.status != 200:
                    return None

                html = await r.text()

                if BEAUTIFULSOUP_AVAILABLE:
                    soup = BeautifulSoup(html, "html.parser")
                    for s in soup(["script", "style"]):
                        s.decompose()
                    text = soup.get_text(" ")
                else:
                    text = re.sub("<.*?>", " ", html)

                text = re.sub(r"\s+", " ", text)

                self.cache[url] = text[:4000]
                return self.cache[url]

        except:
            return None

    # =============================

    async def _validate(self, results):
        valid = []

        for r in results:
            url = r["url"]

            domain = re.search(r"https?://([^/]+)", url)
            domain = domain.group(1) if domain else ""

            trust = 0.5

            for d, score in self.domain_trust.items():
                if d in domain:
                    trust = score

            # фильтр мусора
            if any(x in domain for x in ["forum", "pinterest", "yahoo"]):
                continue

            if len(r["text"]) < 50:
                continue

            r["trust"] = trust
            r["domain"] = domain

            valid.append(r)

        valid.sort(key=lambda x: x["trust"], reverse=True)

        return valid

    # =============================

    def _cross_check(self, results):
        # ищем совпадения текста
        for r in results:
            r["score"] = r["trust"]

            for other in results:
                if r == other:
                    continue

                common = self._similarity(r["text"], other["text"])

                if common > 0.2:
                    r["score"] += 0.1

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _similarity(self, t1, t2):
        words1 = set(t1.lower().split())
        words2 = set(t2.lower().split())

        if not words1 or not words2:
            return 0

        return len(words1 & words2) / len(words1 | words2)

    # =============================

    async def _synthesize(self, query, sources, llm):
        text = ""

        for i, s in enumerate(sources[:5], 1):
            text += f"[{i}] {s['text'][:500]}\n"

        prompt = f"""
Ответь кратко и точно на основе источников.

Вопрос:
{query}

Источники:
{text}

Ответ:
"""

        try:
            return await llm(prompt, temperature=0.2, max_tokens=500)
        except:
            return "Не удалось сгенерировать ответ"

    # =============================

    def _normalize_url(self, url):
        url = url.split("#")[0]
        url = url.split("?")[0]
        return url

# ==================== ПРОСТЫЕ ЯДРА ====================
class DateTimeCore:
    """Точная информация о дате и времени"""

    async def can_handle(self, query: str) -> Tuple[bool, float]:
        q = query.lower()

        # ПРЯМЫЕ вопросы о дате/времени (высокий приоритет)
        direct_questions = [
            'какой день', 'какое число', 'который час', 'сколько времени',
            'текущая дата', 'какая дата', 'какое сегодня число',
            'день недели', 'какой сегодня день недели'
        ]

        # Проверяем прямые вопросы
        if any(q.startswith(kw) or f' {kw}' in q for kw in direct_questions):
            return True, 0.95

        # НЕ срабатываем если есть другие важные слова
        web_priority_words = ['курс', 'погода', 'новост', 'произошло', 'случилось', 'событи']
        if any(word in q for word in web_priority_words):
            return False, 0.0  # Пропускаем, пусть обработает веб-поиск

        # Только "сегодня"/"завтра"/"вчера" без контекста - низкий приоритет
        temporal_words = ['сегодня', 'завтра', 'вчера']
        if any(word in q for word in temporal_words) and len(q.split()) <= 3:
            # "какое сегодня" - да, "что сегодня произошло" - нет
            return True, 0.60  # ПОНИЖЕН приоритет

        return False, 0.0

    async def execute(self, query: str) -> Dict[str, Any]:
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
        months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                  'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']

        result = (
            f"📅 **Сегодня:**\n"
            f"• Дата: {now.day} {months[now.month - 1]} {now.year} года\n"
            f"• День недели: {weekdays[now.weekday()]}\n"
            f"• Время: {now.strftime('%H:%M:%S')}"
        )

        return {'success': True, 'answer': result, 'confidence': 0.99}


class CalculatorCore:
    """Безопасный калькулятор"""

    async def can_handle(self, query: str) -> Tuple[bool, float]:
        q = query.lower()
        has_math = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', q))
        has_words = any(w in q for w in ['сколько будет', 'посчитай', 'вычисли', 'раздели', 'умножь'])

        if has_math:
            return True, 0.95
        if has_words and any(c.isdigit() for c in q):
            return True, 0.85
        return False, 0.0

    async def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Извлекаем выражение
            expr_match = re.search(r'([\d\.\+\-\*\/\(\)\s]+)', query)
            if not expr_match:
                return {'success': False, 'answer': 'Не найдено математическое выражение'}

            expr = expr_match.group(1).replace(' ', '')

            # Безопасное вычисление через eval с ограничениями
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expr):
                return {'success': False, 'answer': 'Недопустимые символы в выражении'}

            result = eval(expr, {"__builtins__": {}}, {})

            answer = f"🧮 **Результат:**\n`{expr} = {result}`"
            return {'success': True, 'answer': answer, 'confidence': 0.99}

        except Exception as e:
            return {'success': False, 'answer': f'Ошибка вычисления: {str(e)}'}


class FileSaveCore:
    """Ядро для сохранения файлов по запросу пользователя"""

    def __init__(self, base_dir: str = "user_files"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.allowed_extensions = {'.txt', '.json', '.py'}

    def _sanitize_filename(self, filename: str) -> str:
        filename = Path(filename).name
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'^\.+', '', filename)
        if not filename or filename.startswith('.'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"file_{timestamp}.txt"
        return filename

    async def can_handle(self, query: str) -> Tuple[bool, float]:
        q = query.lower()
        save_keywords = ['сохрани', 'запиши', 'сохранить', 'записать', 'в файл', 'как файл']
        format_keywords = ['.txt', '.json', '.py', 'текстовый', 'json', 'пайтон', 'питон']
        if any(kw in q for kw in save_keywords) and any(fkw in q for fkw in format_keywords):
            if not any(x in q for x in ['как сделать', 'как сохранить', 'умеешь ли', 'можешь ли']):
                return True, 0.88
        return False, 0.0

    async def execute(self, query: str, user_id: str, llm_caller: callable) -> Dict[str, Any]:
        prompt = f"""Проанализируй запрос и извлеки данные для сохранения.
ЗАПРОС: "{query}"
ВЕРНИ ТОЛЬКО ВАЛИДНЫЙ JSON:
{{
  "filename": "имя.расширение",
  "content": "содержимое",
  "format": "txt|json|py",
  "reason": "почему сохранить"
}}
Если нет данных — верни {{"error": "no_data"}}"""

        try:
            response = await llm_caller(prompt, temperature=0.1, max_tokens=300)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {'success': False, 'answer': '❌ Не распознаны данные для сохранения', 'confidence': 0.0}

            data = json.loads(json_match.group(0))
            if 'error' in data:
                return {'success': False, 'answer': 'ℹ️ Нет данных для сохранения', 'confidence': 0.0}

            filename = self._sanitize_filename(data['filename'])
            content = data['content'].strip()
            fmt = data.get('format', 'txt').lower()

            if not content:
                return {'success': False, 'answer': '❌ Пустое содержимое', 'confidence': 0.0}

            if '.' not in filename:
                ext = '.txt' if fmt == 'txt' else '.json' if fmt == 'json' else '.py'
                filename += ext

            user_dir = self.base_dir / f"user_{user_id}"
            user_dir.mkdir(exist_ok=True)

            filepath = user_dir / filename
            counter = 1
            while filepath.exists():
                stem = filepath.stem
                suffix = filepath.suffix
                filepath = user_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            try:
                if filepath.suffix.lower() == '.json':
                    parsed = json.loads(content)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
            except json.JSONDecodeError as e:
                return {'success': False, 'answer': f'❌ Невалидный JSON: {str(e)[:80]}', 'confidence': 0.0}

            size_kb = filepath.stat().st_size / 1024
            answer = (
                f"✅ **Файл сохранён**\n"
                f"📁 `{filename}`\n"
                f"📊 {size_kb:.1f} КБ\n"
                f"💡 {data.get('reason', 'По запросу')}"
            )

            return {
                'success': True,
                'answer': answer,
                'confidence': 0.95,
                'strategy': 'file_save',
                'metadata': {
                    'filepath': str(filepath.relative_to(self.base_dir)),
                    'format': filepath.suffix.lower()[1:]
                }
            }

        except Exception as e:
            return {
                'success': False,
                'answer': f'❌ Ошибка: {str(e)[:120]}',
                'confidence': 0.0,
                'strategy': 'file_save'
            }

# ==================== МЕТА-КОГНИТИВНЫЙ МЕНЕДЖЕР ====================
class MetaCognitiveManager:
    """
    🧠 МЕТА-КОГНИТИВНЫЙ МЕНЕДЖЕР v8.0
    - Автоматическая память
    - Валидированный веб-поиск
    - Адаптивные стратегии
    """

    def __init__(self, user_id: str, llm_caller: callable):
        self.user_id = user_id
        self.llm_caller = llm_caller

        # Системы
        self.memory = AutoMemorySystem(user_id)
        self.web_search = ValidatedWebSearchCore()
        self.datetime_core = DateTimeCore()
        self.calculator_core = CalculatorCore()
        # Добавить после инициализации других ядер:
        self.file_save_core = FileSaveCore()

    async def initialize(self):
        """Инициализация асинхронных компонентов"""
        await self.web_search.initialize()

    async def close(self):
        """Закрытие ресурсов"""
        await self.web_search.close()

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Полный цикл обработки запроса:
        1. Анализ и сохранение в память
        2. Выбор стратегии
        3. Выполнение
        4. Рефлексия
        5. Обучение
        """
        print(f"\n{'=' * 70}")
        print(f"🧠 ОБРАБОТКА: {query[:60]}...")
        print(f"{'=' * 70}")

        # 1. АВТОМАТИЧЕСКОЕ СОХРАНЕНИЕ В ПАМЯТЬ
        await self.memory.analyze_and_store_message('user', query, self.llm_caller)

        # 2. ВЫБОР СТРАТЕГИИ
        strategy = await self._select_strategy(query)
        print(f"📋 СТРАТЕГИЯ: {strategy['name']}")

        # 3. ВЫПОЛНЕНИЕ
        result = await self._execute_strategy(query, strategy)

        # 4. СОХРАНЕНИЕ ОТВЕТА В ПАМЯТЬ
        await self.memory.analyze_and_store_message('assistant', result['answer'])

        # 5. ПРИНУДИТЕЛЬНОЕ СОХРАНЕНИЕ (для надёжности)
        self.memory._save_all()

        return result

    async def _select_strategy(self, query: str) -> Dict[str, Any]:
        """Выбор оптимальной стратегии обработки"""
        q = query.lower()

        # 1. ПРИОРИТЕТ: Проверяем веб-ключевые слова (курс, погода и т.д.)
        web_priority_keywords = ['курс', 'погода', 'новост', 'произошло', 'случилось', 'событи']
        has_web_priority = any(kw in q for kw in web_priority_keywords)

        # 2. Проверяем калькулятор (высокий приоритет для математики)
        can_calc, conf_calc = await self.calculator_core.can_handle(query)
        if can_calc and conf_calc > 0.8:
            return {'name': 'calculator', 'confidence': conf_calc}

        # 3. Если есть веб-приоритетные слова - СРАЗУ на веб-поиск
        if has_web_priority and DDGS_AVAILABLE:
            # Проверяем что это не мета-вопрос
            meta_patterns = [
                r'кто ты\b', r'как тебя зовут', r'ты бот\b',
                r'что ты умеешь', r'как ты работаешь', r'твои команды',
                r'что ты можешь\b', r'расскажи о себе'
            ]
            is_meta = any(re.search(p, q) for p in meta_patterns)

            if not is_meta:
                print(f"   🌐 Веб-поиск (приоритет): {q[:30]}...")
                return {'name': 'web_search', 'confidence': 0.90}

        # 4. ТОЛЬКО ПОТОМ проверяем datetime (если нет веб-приоритета)
        can_datetime, conf_dt = await self.datetime_core.can_handle(query)

        # Понижаем приоритет datetime если есть другие ключевые слова
        if can_datetime and conf_dt > 0.8:
            # НО если есть "курс", "погода" и т.д. - НЕ используем datetime
            if not has_web_priority:
                return {'name': 'datetime', 'confidence': conf_dt}

        # 5. Общая проверка веб-поиска (для других ключевых слов)
        web_keywords = ['актуальн', 'последн', 'свежи', 'найди', 'поищи', 'сейчас']
        needs_web = any(kw in q for kw in web_keywords)

        # Избегаем веб-поиска ТОЛЬКО для вопросов О САМОМ БОТЕ
        meta_patterns = [
            r'кто ты\b', r'как тебя зовут', r'ты бот\b',
            r'что ты умеешь', r'как ты работаешь', r'твои команды',
            r'что ты можешь\b', r'расскажи о себе'
        ]
        is_meta = any(re.search(p, q) for p in meta_patterns)

        # Специальная обработка для запросов к памяти
        memory_queries = [r'что ты знаешь обо мне', r'моя память', r'что помнишь']
        is_memory_query = any(re.search(p, q) for p in memory_queries)

        # Если это запрос к памяти - сразу ищем в памяти
        if is_memory_query:
            memory_facts = await self.memory.search_long_term(query, limit=5, min_similarity=0.3)
            if memory_facts:
                return {'name': 'memory_direct', 'confidence': 0.9, 'facts': memory_facts}

        if needs_web and not is_meta and DDGS_AVAILABLE:
            print(f"   🌐 Требуется веб-поиск: needs_web={needs_web}, is_meta={is_meta}, DDGS={DDGS_AVAILABLE}")
            return {'name': 'web_search', 'confidence': 0.85}
        elif needs_web and not DDGS_AVAILABLE:
            print(f"   ⚠️ Веб-поиск нужен, но DDGS недоступен!")
        elif needs_web and is_meta:
            print(f"   ℹ️ Пропускаем веб-поиск: это мета-вопрос о боте")

        # Проверяем память на прямой ответ
        memory_facts = await self.memory.search_long_term(query, limit=3, min_similarity=0.6)
        if memory_facts and memory_facts[0].get('importance', 0) > 0.8:
            return {'name': 'memory_direct', 'confidence': 0.8, 'facts': memory_facts}
        # === ПРОВЕРКА ЯДРА СОХРАНЕНИЯ ФАЙЛОВ ===
        can_save, conf_save = await self.file_save_core.can_handle(query)
        if can_save and conf_save > 0.85:
            return {'name': 'file_save', 'confidence': conf_save}
        # По умолчанию - общий LLM с контекстом памяти
        return {'name': 'llm_with_memory', 'confidence': 0.7}

    async def _execute_strategy(self, query: str, strategy: Dict) -> Dict[str, Any]:
        """Выполнение выбранной стратегии"""
        strategy_name = strategy['name']

        try:
            # === СПЕЦИАЛИЗИРОВАННЫЕ ЯДРА ===
            if strategy_name == 'datetime':
                return await self.datetime_core.execute(query)

            if strategy_name == 'calculator':
                return await self.calculator_core.execute(query)

            # === ВЕБ-ПОИСК С ВАЛИДАЦИЕЙ ===
            if strategy_name == 'web_search':
                search_result = await self.web_search.search_and_validate(query, self.llm_caller)

                if search_result['success']:
                    # Сохраняем найденную информацию в долгосрочную память
                    if search_result.get('confidence', 0) > 0.7:
                        await self.memory.add_to_long_term(
                            content=search_result['answer'][:300],
                            category='web_fact',
                            importance=0.6,
                            source='web_search_validated',
                            metadata={'query': query, 'confidence': search_result['confidence']}
                        )

                    return {
                        'success': True,
                        'answer': search_result['answer'],
                        'confidence': search_result['confidence'],
                        'strategy': 'web_search'
                    }
                else:
                    # Fallback на LLM
                    return await self._llm_response_with_memory(query)

            # === ПРЯМОЙ ОТВЕТ ИЗ ПАМЯТИ ===
            if strategy_name == 'memory_direct':
                facts = strategy.get('facts', [])
                answer = "🧠 **Из моей памяти о вас:**\n\n"
                for i, fact in enumerate(facts[:3], 1):
                    answer += f"{i}. {fact['content']}\n"

                return {
                    'success': True,
                    'answer': answer,
                    'confidence': 0.85,
                    'strategy': 'memory_direct'
                }
            # Добавить ПЕРЕД блоком "=== ОБЩИЙ LLM С КОНТЕКСТОМ ПАМЯТИ ===":
            # === СОХРАНЕНИЕ ФАЙЛА ===
            # === СОХРАНЕНИЕ ФАЙЛА ЧЕРЕЗ LLM-АНАЛИЗ ===
            if strategy_name == 'file_save':
                result = await self.file_save_core.execute(query, self.user_id, self.llm_caller)
                # Сохраняем факт в долгосрочную память
                if result['success'] and 'metadata' in result:
                    await self.memory.add_to_long_term(
                        content=f"Сохранён файл: {result['metadata']['filepath']} ({result['metadata']['format']})",
                        category='user_activity',
                        importance=0.6,
                        source='file_save'
                    )
                return result
            # === ОБЩИЙ LLM С КОНТЕКСТОМ ПАМЯТИ ===
            return await self._llm_response_with_memory(query)

        except Exception as e:
            print(f"❌ Ошибка выполнения стратегии: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'answer': f'Произошла ошибка при обработке запроса: {str(e)[:150]}',
                'confidence': 0.0
            }

    async def _llm_response_with_memory(self, query: str) -> Dict[str, Any]:
        """Общий ответ LLM с полным контекстом памяти"""
        # Получаем расширенный контекст
        context = await self.memory.get_enhanced_context(query,
                                                         short_term_limit=12,
                                                         long_term_limit=6)

        # Формируем промпт
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        system_prompt = f"""Ты — интеллектуальный ассистент с долгосрочной памятью о пользователе.

ТЕКУЩЕЕ ВРЕМЯ: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})

{context}

ИНСТРУКЦИЯ:
1. Используй информацию из ДОЛГОСРОЧНОЙ ПАМЯТИ о пользователе
2. Учитывай ПАТТЕРНЫ ПОВЕДЕНИЯ и адаптируй стиль ответа
3. Ссылайся на ПРЕДЫДУЩИЕ РАЗГОВОРЫ если это релевантно
4. Если не уверен - честно признай это
5. Для актуальных данных (курсы, погода, новости) - предложи использовать веб-поиск
6. Будь полезным, точным и дружелюбным

ОТВЕТ:"""

        try:
            response = await self.llm_caller(system_prompt, temperature=0.6, max_tokens=1000)

            return {
                'success': True,
                'answer': response,
                'confidence': 0.75,
                'strategy': 'llm_with_memory'
            }

        except Exception as e:
            return {
                'success': False,
                'answer': f'Ошибка LLM: {str(e)[:150]}',
                'confidence': 0.0
            }


# ==================== ТЕЛЕГРАМ БОТ ====================
class TelegramBot:
    def __init__(self):
        self.user_managers: Dict[str, MetaCognitiveManager] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Инициализация HTTP сессии"""
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            )

    async def close(self):
        """Закрытие ресурсов"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

        for manager in self.user_managers.values():
            await manager.close()

    def get_manager(self, user_id: str) -> MetaCognitiveManager:
        """Получить/создать менеджер для пользователя"""
        if user_id not in self.user_managers:
            self.user_managers[user_id] = MetaCognitiveManager(user_id, self.get_llm_response)
            print(f"🧠 Создан менеджер для пользователя {user_id}")
        return self.user_managers[user_id]

    async def get_llm_response(self, prompt: str, temperature: float = 0.5,
                               max_tokens: int = 1500) -> str:
        """Асинхронный запрос к LLM"""
        try:
            async with self.http_session.post(
                    LM_STUDIO_API_URL,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {LM_STUDIO_API_KEY}'
                    },
                    json={
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'stream': False
                    }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    error = await resp.text()
                    return f"❌ Ошибка LLM ({resp.status}): {error[:100]}"

        except asyncio.TimeoutError:
            return "❌ Таймаут ответа от LLM (120 сек)"
        except Exception as e:
            return f"❌ Ошибка LLM: {str(e)[:150]}"

    # ==================== КОМАНДЫ БОТА ====================
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        web_status = "✅ Доступен" if DDGS_AVAILABLE else "❌ Недоступен"

        await update.message.reply_text(
            f"🧠 **МИНИ-МОЗГ БОТ v8.0**\n"
            f"\n✨ **Новые возможности:**\n"
            f"• ✅ РАБОЧАЯ долгосрочная память с автосохранением\n"
            f"• 🌐 Веб-поиск с валидацией и синтезом: {web_status}\n"
            f"• 🧠 Автоматический анализ каждого сообщения\n"
            f"• 📊 Выявление паттернов поведения\n"
            f"• 🎯 Адаптация под ваш стиль общения\n"
            f"\n⚙️ **Команды:**\n"
            f"/memory - статистика памяти\n"
            f"/search_memory [запрос] - поиск в памяти\n"
            f"/clear - очистить краткосрочную память\n"
            
            f"/help - справка",
            parse_mode='Markdown'
        )

    async def memory_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        stats = manager.memory.get_stats()

        msg = (
            f"🧠 **СТАТИСТИКА ПАМЯТИ**\n"
            f"\n📝 Краткосрочная память: {stats['short_term_count']} сообщений\n"
            f"📚 Долгосрочная память: {stats['long_term_count']} фактов\n"
            f"📊 Всего сообщений: {stats['total_messages']}\n"
            f"🔍 Извлечено фактов: {stats['facts_extracted']}\n"
            f"🎯 Выявлено паттернов: {stats['patterns_identified']}\n"
            f"📤 Автопереносов в LT: {stats['auto_transfers']}\n"
            f"\n💬 **Ваш стиль:** {stats['communication_style']}\n"
        )

        if stats['frequent_topics']:
            topics = ", ".join(stats['frequent_topics'][:5])
            msg += f"📌 **Частые темы:** {topics}\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def search_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Поиск в памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        query = ' '.join(context.args) if context.args else ''

        if not query:
            await update.message.reply_text(
                "🔍 Использование: `/search_memory ваш запрос`",
                parse_mode='Markdown'
            )
            return

        results = await manager.memory.search_long_term(query, limit=5)

        if not results:
            await update.message.reply_text(f"❌ Ничего не найдено по запросу: `{query}`", parse_mode='Markdown')
            return

        msg = f"🔍 **Найдено в памяти:** `{query}`\n\n"
        for i, mem in enumerate(results, 1):
            category = mem.get('category', 'general')
            importance = "⭐" * min(5, int(mem.get('importance', 0.5) * 5))
            msg += f"{i}. [{category}] {importance}\n"
            msg += f"   {mem['content'][:150]}\n\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистка краткосрочной памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)

        manager.memory.short_term = []
        manager.memory._save_all()

        await update.message.reply_text(
            "🧹 **Краткосрочная память очищена!**\n"
            "Долгосрочная память и паттерны сохранены.",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Справка"""
        await update.message.reply_text(
            "📖 **СПРАВКА**\n"
            "\n**Как работает бот:**\n"
            "1️⃣ Автоматически анализирует каждое ваше сообщение\n"
            "2️⃣ Извлекает факты, предпочтения, эмоции\n"
            "3️⃣ Сохраняет важное в долгосрочную память\n"
            "4️⃣ Адаптируется под ваш стиль общения\n"
            "5️⃣ Использует веб-поиск для актуальной информации\n"
            "\n**Примеры запросов:**\n"
            "• `курс доллара` - веб-поиск с валидацией\n"
            "• `сколько будет 125 * 8` - калькулятор\n"
            "• `какой сегодня день` - дата/время\n"
            "• `что ты знаешь обо мне` - поиск в памяти\n"
            "• `сохрани заметка.txt купить молоко` - сохранение файла\n",
            "\n**Команды:**\n"
            "/memory - статистика\n"
            "/search_memory - поиск\n"
            "/clear - очистка\n"
            "/help - эта справка",
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            manager = self.get_manager(user_id)

            # Инициализируем если нужно
            if not hasattr(manager, '_initialized'):
                await manager.initialize()
                manager._initialized = True

            # Обрабатываем запрос
            result = await manager.process_query(text)

            # Отправляем ответ
            answer = result.get('answer', 'Не удалось сформировать ответ')

            # Ограничиваем длину для Telegram
            if len(answer) > 4000:
                answer = answer[:3950] + "\n\n...(усечено)"

            await update.message.reply_text(
                answer,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )

            # Логируем
            confidence = result.get('confidence', 0)
            strategy = result.get('strategy', 'unknown')
            print(f"✅ Ответ отправлен | Стратегия: {strategy} | Уверенность: {confidence:.2f}")

        except Exception as e:
            error_msg = f"❌ **Ошибка:** {str(e)[:200]}"
            print(f"ERROR: {e}")
            traceback.print_exc()

            await update.message.reply_text(error_msg, parse_mode='Markdown')


# ==================== ЗАПУСК ====================
async def main_async():
    """Асинхронная главная функция"""
    print("\n" + "=" * 70)
    print("🚀 МИНИ-МОЗГ БОТ v8.0 - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("=" * 70)
    print(f"⏰ Запуск: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"🧠 Память: {MEMORY_DIR}/")
    print(f"🌐 Веб-поиск: {'✅ Включён (с валидацией)' if DDGS_AVAILABLE else '❌ Выключен'}")
    print("=" * 70)

    # Проверка LM Studio
    try:
        async with aiohttp.ClientSession() as session:
            test_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '/v1/models')
            async with session.get(test_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    print(f"✅ LM Studio доступна")
                else:
                    print(f"⚠️ LM Studio код: {resp.status}")
    except Exception as e:
        print(f"⚠️ LM Studio недоступна: {e}")

    print("=" * 70)
    print("\n🔄 Инициализация...")

    bot = TelegramBot()
    await bot.initialize()

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("memory", bot.memory_stats))
    application.add_handler(CommandHandler("search_memory", bot.search_memory))
    application.add_handler(CommandHandler("clear", bot.clear))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот готов!")
    print("=" * 70)
    print("💬 Ключевые функции:")
    print("   • ✅ Автоматическая долгосрочная память")
    print("   • 🌐 Веб-поиск с валидацией источников")
    print("   • 🧠 Анализ каждого сообщения")
    print("   • 📊 Выявление паттернов поведения")
    print("   • 🎯 Адаптация под пользователя")
    print("=" * 70)
    print("\nCtrl+C для остановки\n")

    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)

        # Бесконечное ожидание
        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()
    finally:
        await application.stop()
        await application.shutdown()
        await bot.close()
        print("✅ Бот остановлен")


def main():
    """Точка входа"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Выход")
    except Exception as e:
        print(f"\n❌ Фатальная ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()