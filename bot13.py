#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v9.0 - УЛУЧШЕННАЯ ВЕРСИЯ
✅ Продвинутая краткосрочная и долгосрочная память
✅ Ассоциативная память и воображение
✅ Интеллектуальный анализ и обучение
✅ Когнитивные паттерны и мета-обучение
✅ Улучшенная архитектура и логика
"""

import os
import json
import re
import asyncio
import aiohttp
import traceback
import hashlib
import math
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
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
    from duckduckgo_search import AsyncDDGS

    DDGS_AVAILABLE = True
    DDGS_ASYNC = True
    print("✅ Async DuckDuckGo Search доступен")
except ImportError:
    try:
        from duckduckgo_search import DDGS

        DDGS_AVAILABLE = True
        DDGS_ASYNC = False
        print("✅ DuckDuckGo Search доступен (синхронный режим)")
    except ImportError:
        print("⚠️ DuckDuckGo Search не установлен")

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


# ==================== БАЗОВЫЕ УТИЛИТЫ ====================
class MemoryUtils:
    """Утилиты для работы с памятью"""

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Вычисление схожести текстов (упрощённый cosine similarity)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def extract_keywords(text: str, top_n: int = 5) -> List[str]:
        """Извлечение ключевых слов"""
        # Удаляем стоп-слова
        stop_words = {'в', 'и', 'на', 'с', 'по', 'для', 'от', 'к', 'о', 'у', 'из', 'за', 'что', 'это', 'как', 'то', 'а',
                      'но', 'или'}
        words = [w.lower() for w in re.findall(r'\b\w{3,}\b', text)]
        filtered = [w for w in words if w not in stop_words]

        # Считаем частоту
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]

    @staticmethod
    def calculate_importance(entry: Dict[str, Any]) -> float:
        """Вычисление важности записи памяти"""
        importance = 0.5  # базовая важность

        # Факторы важности
        if entry.get('has_question'):
            importance += 0.1
        if entry.get('has_numbers'):
            importance += 0.1
        if len(entry.get('extracted_facts', [])) > 0:
            importance += 0.2
        if len(entry.get('detected_emotions', [])) > 0:
            importance += 0.1
        if entry.get('word_count', 0) > 50:
            importance += 0.1

        return min(1.0, importance)


# ==================== АССОЦИАТИВНАЯ ПАМЯТЬ И ВООБРАЖЕНИЕ ====================
class AssociativeMemory:
    """
    🌐 АССОЦИАТИВНАЯ ПАМЯТЬ
    - Связывает концепции и идеи
    - Формирует ассоциативные цепочки
    - Генерирует новые связи (воображение)
    """

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        self.associations_file = os.path.join(memory_dir, "associations.json")
        self.imagination_file = os.path.join(memory_dir, "imagination.json")

        # Граф ассоциаций: concept -> {связанные концепты с весами}
        self.associations: Dict[str, Dict[str, float]] = self._load_json(
            self.associations_file, {}
        )

        # Воображаемые концепты (синтезированные из ассоциаций)
        self.imaginations: List[Dict[str, Any]] = self._load_json(
            self.imagination_file, []
        )

    def _load_json(self, filepath: str, default: Any) -> Any:
        """Загрузка JSON"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {filepath}: {e}")
        return default

    def _save_json(self, filepath: str, data: Any):
        """Сохранение JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения {filepath}: {e}")

    def add_association(self, concept1: str, concept2: str, strength: float = 0.5):
        """Добавление ассоциации между концептами"""
        concept1 = concept1.lower().strip()
        concept2 = concept2.lower().strip()

        if concept1 == concept2:
            return

        # Двунаправленная связь
        if concept1 not in self.associations:
            self.associations[concept1] = {}
        if concept2 not in self.associations:
            self.associations[concept2] = {}

        # Усиливаем существующую связь или создаём новую
        current_strength = self.associations[concept1].get(concept2, 0.0)
        self.associations[concept1][concept2] = min(1.0, current_strength + strength)

        current_strength = self.associations[concept2].get(concept1, 0.0)
        self.associations[concept2][concept1] = min(1.0, current_strength + strength)

    def get_associations(self, concept: str, min_strength: float = 0.3, limit: int = 10) -> List[Tuple[str, float]]:
        """Получение ассоциаций для концепта"""
        concept = concept.lower().strip()

        if concept not in self.associations:
            return []

        # Сортируем по силе связи
        associations = [
            (related, strength)
            for related, strength in self.associations[concept].items()
            if strength >= min_strength
        ]

        associations.sort(key=lambda x: x[1], reverse=True)
        return associations[:limit]

    def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Поиск ассоциативного пути между концептами (BFS)"""
        start = start.lower().strip()
        end = end.lower().strip()

        if start == end:
            return [start]

        if start not in self.associations or end not in self.associations:
            return None

        # BFS
        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor, strength in self.associations.get(current, {}).items():
                if neighbor in visited or strength < 0.3:
                    continue

                new_path = path + [neighbor]

                if neighbor == end:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None

    def imagine(self, base_concepts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Воображение: синтез нового концепта из базовых
        Находит общие ассоциации и создаёт новую идею
        """
        if len(base_concepts) < 2:
            return None

        # Собираем все ассоциации
        all_associations = defaultdict(float)

        for concept in base_concepts:
            associations = self.get_associations(concept, min_strength=0.4)
            for related, strength in associations:
                all_associations[related] += strength

        if not all_associations:
            return None

        # Находим самую сильную общую ассоциацию
        common = max(all_associations.items(), key=lambda x: x[1])

        imagination = {
            'id': hashlib.md5(f"{'_'.join(base_concepts)}_{common[0]}".encode()).hexdigest()[:12],
            'base_concepts': base_concepts,
            'imagined_concept': common[0],
            'strength': common[1] / len(base_concepts),
            'timestamp': datetime.now().isoformat(),
            'description': f"Синтез из {', '.join(base_concepts)} через {common[0]}"
        }

        self.imaginations.append(imagination)
        return imagination

    def save(self):
        """Сохранение ассоциаций и воображения"""
        self._save_json(self.associations_file, self.associations)
        self._save_json(self.imagination_file, self.imaginations)


# ==================== УЛУЧШЕННАЯ СИСТЕМА ПАМЯТИ v3.0 ====================
class EnhancedMemorySystem:
    """
    🧠 УЛУЧШЕННАЯ СИСТЕМА ПАМЯТИ
    - Краткосрочная память (последние N сообщений)
    - Долгосрочная память (важные факты и знания)
    - Ассоциативная память (связи между концептами)
    - Рабочая память (текущий контекст разговора)
    - Эпизодическая память (воспоминания о событиях)
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.memory_dir, exist_ok=True)

        # Файлы памяти
        self.short_term_file = os.path.join(self.memory_dir, "short_term.json")
        self.long_term_file = os.path.join(self.memory_dir, "long_term.json")
        self.working_file = os.path.join(self.memory_dir, "working.json")
        self.episodic_file = os.path.join(self.memory_dir, "episodic.json")
        self.patterns_file = os.path.join(self.memory_dir, "patterns.json")
        self.preferences_file = os.path.join(self.memory_dir, "preferences.json")
        self.metadata_file = os.path.join(self.memory_dir, "metadata.json")

        # Загрузка данных
        self.short_term: List[Dict] = self._load_json(self.short_term_file, [])
        self.long_term: List[Dict] = self._load_json(self.long_term_file, [])
        self.working_memory: Dict = self._load_json(self.working_file, {
            'current_topic': None,
            'conversation_context': [],
            'active_goals': [],
            'timestamp': datetime.now().isoformat()
        })
        self.episodic_memory: List[Dict] = self._load_json(self.episodic_file, [])

        self.patterns: Dict = self._load_json(self.patterns_file, {
            'communication_style': 'neutral',
            'frequent_topics': [],
            'emotional_markers': {},
            'time_preferences': {},
            'interaction_patterns': {},
            'cognitive_patterns': {},  # новое
            'learning_speed': 1.0,  # скорость обучения
            'last_pattern_update': datetime.now().isoformat()
        })

        self.preferences: Dict = self._load_json(self.preferences_file, {
            'explicit': {},
            'inferred': {},
            'confirmed': {}
        })

        self.metadata: Dict = self._load_json(self.metadata_file, {
            'total_messages': 0,
            'facts_extracted': 0,
            'patterns_identified': 0,
            'auto_transfers': 0,
            'associations_created': 0,
            'imaginations_generated': 0,
            'first_interaction': datetime.now().isoformat(),
            'last_interaction': datetime.now().isoformat(),
            'session_count': 0,
            'learning_cycles': 0
        })

        # Ассоциативная память
        self.associative = AssociativeMemory(self.memory_dir)

        # Счётчик для автосохранения
        self._changes_count = 0
        self._autosave_threshold = 3

        print(f"🧠 Память загружена для {user_id}:")
        print(f"   ST={len(self.short_term)}, LT={len(self.long_term)}, "
              f"Episodes={len(self.episodic_memory)}")
        print(f"   Ассоциации={len(self.associative.associations)}, "
              f"Воображения={len(self.associative.imaginations)}")

    def _load_json(self, filepath: str, default: Any) -> Any:
        """Загрузка JSON"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {filepath}: {e}")
        return default

    def _save_json(self, filepath: str, data: Any):
        """Сохранение JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения {filepath}: {e}")

    # ==================== АНАЛИЗ И ХРАНЕНИЕ ====================
    async def analyze_and_store_message(
            self,
            role: str,
            content: str,
            llm_caller: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Глубокий анализ сообщения с извлечением:
        - Фактов и знаний
        - Эмоций и настроения
        - Концептов и ассоциаций
        - Паттернов мышления
        """
        timestamp = datetime.now()

        # 1. Базовый анализ
        entry = {
            'role': role,
            'content': content,
            'timestamp': timestamp.isoformat(),
            'word_count': len(content.split()),
            'has_question': '?' in content,
            'has_numbers': bool(re.search(r'\d', content)),
            'keywords': MemoryUtils.extract_keywords(content),
            'extracted_facts': [],
            'detected_emotions': [],
            'topic_tags': [],
            'concepts': []
        }

        # 2. Быстрый анализ
        quick_analysis = self._quick_analysis(content, role)
        entry.update(quick_analysis)

        # 3. Глубокий анализ через LLM (если доступен)
        if llm_caller and role == 'user' and len(content) > 10:
            deep_analysis = await self._deep_analysis(content, llm_caller)

            entry['extracted_facts'].extend(deep_analysis.get('facts', []))
            entry['detected_emotions'].extend(deep_analysis.get('emotions', []))
            entry['topic_tags'].extend(deep_analysis.get('topics', []))
            entry['concepts'].extend(deep_analysis.get('concepts', []))

            # Сохраняем важные факты в долгосрочную память
            for fact in deep_analysis.get('important_facts', []):
                await self.add_to_long_term(
                    fact['content'],
                    category=fact.get('category', 'general'),
                    importance=fact.get('importance', 0.7),
                    source='auto_extraction'
                )

            # Создаём ассоциации между концептами
            concepts = entry['concepts']
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i + 1:]:
                    self.associative.add_association(concept1, concept2, strength=0.4)
                    self.metadata['associations_created'] += 1

        # 4. Добавляем в краткосрочную память
        self.short_term.append(entry)

        # 5. Обновляем рабочую память
        self._update_working_memory(entry)

        # 6. Проверка на создание эпизода
        if self._should_create_episode():
            await self._create_episode()

        # 7. Ограничиваем краткосрочную память
        if len(self.short_term) > 50:
            await self._intelligent_transfer_to_long_term()
            self.short_term = self.short_term[-30:]

        # 8. Обновляем паттерны и обучаемся
        await self._update_patterns(entry)
        await self._learn_from_interaction(entry)

        # 9. Метаданные
        self.metadata['total_messages'] += 1
        self.metadata['last_interaction'] = timestamp.isoformat()

        # 10. Автосохранение
        await self._auto_save()

        return entry

    def _quick_analysis(self, content: str, role: str) -> Dict[str, Any]:
        """Быстрый анализ без LLM"""
        result = {
            'sentiment': 'neutral',
            'urgency': 'normal',
            'length_category': 'medium'
        }

        # Определение длины
        word_count = len(content.split())
        if word_count < 10:
            result['length_category'] = 'short'
        elif word_count > 50:
            result['length_category'] = 'long'

        # Примитивный анализ настроения
        positive_words = ['хорошо', 'отлично', 'супер', 'класс', 'спасибо', 'thanks', 'great', 'good']
        negative_words = ['плохо', 'ужасно', 'не', 'нет', 'bad', 'terrible', 'wrong']

        content_lower = content.lower()
        positive_count = sum(1 for w in positive_words if w in content_lower)
        negative_count = sum(1 for w in negative_words if w in content_lower)

        if positive_count > negative_count:
            result['sentiment'] = 'positive'
        elif negative_count > positive_count:
            result['sentiment'] = 'negative'

        # Срочность
        urgent_markers = ['срочно', 'быстро', 'важно', 'urgent', 'asap', '!!!']
        if any(marker in content_lower for marker in urgent_markers):
            result['urgency'] = 'high'

        return result

    async def _deep_analysis(self, content: str, llm_caller: callable) -> Dict[str, Any]:
        """Глубокий анализ через LLM"""
        prompt = f"""Проанализируй следующее сообщение и извлеки:
1. Факты (конкретная информация)
2. Эмоции (sentiment)
3. Темы (категории)
4. Концепты (ключевые идеи)
5. Важность (0.0-1.0)

Сообщение: "{content}"

Ответь в JSON формате:
{{
    "facts": ["факт1", "факт2"],
    "emotions": ["эмоция1"],
    "topics": ["тема1"],
    "concepts": ["концепт1", "концепт2"],
    "important_facts": [
        {{"content": "важный факт", "category": "категория", "importance": 0.8}}
    ]
}}"""

        try:
            response = await llm_caller(prompt, temperature=0.3, max_tokens=500)

            # Пытаемся извлечь JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️ Ошибка глубокого анализа: {e}")

        return {
            'facts': [],
            'emotions': [],
            'topics': [],
            'concepts': [],
            'important_facts': []
        }

    def _update_working_memory(self, entry: Dict[str, Any]):
        """Обновление рабочей памяти"""
        # Добавляем в контекст разговора
        context_entry = {
            'role': entry['role'],
            'content': entry['content'][:200],  # укороченная версия
            'timestamp': entry['timestamp'],
            'keywords': entry.get('keywords', [])
        }

        self.working_memory['conversation_context'].append(context_entry)

        # Ограничиваем размер контекста
        if len(self.working_memory['conversation_context']) > 10:
            self.working_memory['conversation_context'] = \
                self.working_memory['conversation_context'][-10:]

        # Определяем текущую тему
        if entry.get('topic_tags'):
            self.working_memory['current_topic'] = entry['topic_tags'][0]

        self.working_memory['timestamp'] = datetime.now().isoformat()

    def _should_create_episode(self) -> bool:
        """Проверка необходимости создания эпизода"""
        # Создаём эпизод каждые 20 сообщений или при смене темы
        if len(self.short_term) >= 20:
            return True

        if len(self.short_term) >= 5:
            recent_topics = set()
            for msg in self.short_term[-5:]:
                recent_topics.update(msg.get('topic_tags', []))

            # Если появилась новая тема
            if len(recent_topics) > 2:
                return True

        return False

    async def _create_episode(self):
        """Создание эпизодической памяти"""
        if len(self.short_term) < 3:
            return

        # Берём последние N сообщений
        messages = self.short_term[-20:]

        # Суммаризация эпизода
        summary = self._summarize_episode(messages)

        episode = {
            'id': hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            'start_time': messages[0]['timestamp'],
            'end_time': messages[-1]['timestamp'],
            'message_count': len(messages),
            'summary': summary,
            'main_topics': self._extract_main_topics(messages),
            'emotional_tone': self._determine_emotional_tone(messages),
            'importance': self._calculate_episode_importance(messages),
            'concepts': self._extract_episode_concepts(messages)
        }

        self.episodic_memory.append(episode)

        # Ограничиваем количество эпизодов
        if len(self.episodic_memory) > 50:
            self.episodic_memory = self.episodic_memory[-50:]

        print(f"📖 Создан эпизод: {summary[:50]}...")

    def _summarize_episode(self, messages: List[Dict]) -> str:
        """Суммаризация эпизода"""
        topics = []
        for msg in messages:
            topics.extend(msg.get('topic_tags', []))

        topic_counts = Counter(topics)
        main_topics = [topic for topic, _ in topic_counts.most_common(3)]

        if main_topics:
            return f"Обсуждение: {', '.join(main_topics)}"
        else:
            return f"Разговор из {len(messages)} сообщений"

    def _extract_main_topics(self, messages: List[Dict]) -> List[str]:
        """Извлечение главных тем"""
        topics = []
        for msg in messages:
            topics.extend(msg.get('topic_tags', []))

        topic_counts = Counter(topics)
        return [topic for topic, _ in topic_counts.most_common(5)]

    def _determine_emotional_tone(self, messages: List[Dict]) -> str:
        """Определение эмоционального тона эпизода"""
        sentiments = [msg.get('sentiment', 'neutral') for msg in messages]
        sentiment_counts = Counter(sentiments)

        most_common = sentiment_counts.most_common(1)
        return most_common[0][0] if most_common else 'neutral'

    def _calculate_episode_importance(self, messages: List[Dict]) -> float:
        """Вычисление важности эпизода"""
        total_importance = 0.0

        for msg in messages:
            msg_importance = MemoryUtils.calculate_importance(msg)
            total_importance += msg_importance

        return total_importance / len(messages) if messages else 0.5

    def _extract_episode_concepts(self, messages: List[Dict]) -> List[str]:
        """Извлечение концептов из эпизода"""
        concepts = set()
        for msg in messages:
            concepts.update(msg.get('concepts', []))

        return list(concepts)[:10]

    # ==================== ИНТЕЛЛЕКТУАЛЬНЫЙ ПЕРЕНОС В ДОЛГОВРЕМЕННУЮ ПАМЯТЬ ====================
    async def _intelligent_transfer_to_long_term(self):
        """
        Умный перенос: анализирует важность, релевантность и связи
        """
        if len(self.short_term) < 10:
            return

        transferred = 0

        for entry in self.short_term:
            importance = MemoryUtils.calculate_importance(entry)

            # Критерии для переноса
            should_transfer = False

            # 1. Высокая важность
            if importance >= 0.7:
                should_transfer = True

            # 2. Содержит факты
            if len(entry.get('extracted_facts', [])) > 0:
                should_transfer = True

            # 3. Эмоционально значимо
            if entry.get('detected_emotions') and importance >= 0.6:
                should_transfer = True

            # 4. Часто упоминаемые концепты
            concepts = entry.get('concepts', [])
            if concepts:
                # Проверяем, часто ли встречается этот концепт
                concept_frequency = sum(
                    1 for msg in self.short_term
                    if any(c in msg.get('concepts', []) for c in concepts)
                )
                if concept_frequency >= 3:
                    should_transfer = True

            if should_transfer:
                # Переносим в долгосрочную память
                await self.add_to_long_term(
                    entry['content'],
                    category=entry.get('topic_tags', ['general'])[0] if entry.get('topic_tags') else 'general',
                    importance=importance,
                    source='intelligent_transfer',
                    metadata={
                        'keywords': entry.get('keywords', []),
                        'concepts': entry.get('concepts', []),
                        'emotions': entry.get('detected_emotions', []),
                        'original_timestamp': entry['timestamp']
                    }
                )
                transferred += 1
                self.metadata['auto_transfers'] += 1

        if transferred > 0:
            print(f"🔄 Интеллектуальный перенос: {transferred} записей в LT память")

    async def add_to_long_term(
            self,
            content: str,
            category: str = 'general',
            importance: float = 0.5,
            source: str = 'manual',
            metadata: Optional[Dict] = None
    ):
        """Добавление в долгосрочную память"""
        # Проверяем дубликаты
        for existing in self.long_term:
            similarity = MemoryUtils.calculate_similarity(content, existing['content'])
            if similarity > 0.8:
                # Усиливаем важность существующей записи
                existing['importance'] = min(1.0, existing['importance'] + 0.1)
                existing['access_count'] += 1
                existing['last_accessed'] = datetime.now().isoformat()
                return

        # Добавляем новую запись
        entry = {
            'id': hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            'content': content,
            'category': category,
            'importance': importance,
            'source': source,
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 0,
            'keywords': MemoryUtils.extract_keywords(content),
            'metadata': metadata or {}
        }

        self.long_term.append(entry)
        self.metadata['facts_extracted'] += 1

        # Создаём ассоциации из keywords
        keywords = entry['keywords']
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i + 1:]:
                self.associative.add_association(kw1, kw2, strength=0.3)

        # Ограничиваем размер
        if len(self.long_term) > 500:
            # Удаляем наименее важные и редко используемые
            self.long_term.sort(
                key=lambda x: (x['importance'], x['access_count']),
                reverse=True
            )
            self.long_term = self.long_term[:400]

    # ==================== ОБУЧЕНИЕ И АДАПТАЦИЯ ====================
    async def _update_patterns(self, entry: Dict[str, Any]):
        """Обновление паттернов поведения"""
        if entry['role'] != 'user':
            return

        # 1. Стиль общения
        word_count = entry['word_count']
        if word_count < 10:
            style = 'краткий'
        elif word_count < 30:
            style = 'средний'
        else:
            style = 'подробный'

        current_style = self.patterns.get('communication_style', 'neutral')
        if current_style != style:
            # Медленно адаптируем стиль
            self.patterns['communication_style'] = style

        # 2. Частые темы
        topics = entry.get('topic_tags', [])
        for topic in topics:
            if topic not in self.patterns['frequent_topics']:
                self.patterns['frequent_topics'].append(topic)

        # Ограничиваем количество тем
        if len(self.patterns['frequent_topics']) > 20:
            self.patterns['frequent_topics'] = self.patterns['frequent_topics'][-20:]

        # 3. Эмоциональные маркеры
        emotions = entry.get('detected_emotions', [])
        for emotion in emotions:
            if emotion not in self.patterns['emotional_markers']:
                self.patterns['emotional_markers'][emotion] = 0
            self.patterns['emotional_markers'][emotion] += 1

        # 4. Когнитивные паттерны
        await self._identify_cognitive_patterns(entry)

        self.patterns['last_pattern_update'] = datetime.now().isoformat()
        self.metadata['patterns_identified'] += 1

    async def _identify_cognitive_patterns(self, entry: Dict[str, Any]):
        """Выявление когнитивных паттернов мышления"""
        content = entry['content'].lower()

        # Паттерны мышления
        patterns = {
            'analytical': ['потому что', 'следовательно', 'анализ', 'данные', 'факты'],
            'creative': ['идея', 'представь', 'возможно', 'креативно', 'воображение'],
            'practical': ['сделать', 'как', 'практически', 'применить', 'использовать'],
            'emotional': ['чувствую', 'переживаю', 'радость', 'грусть', 'волнуюсь'],
            'questioning': ['почему', 'как', 'что', 'зачем', 'когда']
        }

        if 'cognitive_patterns' not in self.patterns:
            self.patterns['cognitive_patterns'] = {}

        for pattern_type, markers in patterns.items():
            count = sum(1 for marker in markers if marker in content)
            if count > 0:
                if pattern_type not in self.patterns['cognitive_patterns']:
                    self.patterns['cognitive_patterns'][pattern_type] = 0
                self.patterns['cognitive_patterns'][pattern_type] += count

    async def _learn_from_interaction(self, entry: Dict[str, Any]):
        """Обучение на основе взаимодействия"""
        # Увеличиваем счётчик циклов обучения
        self.metadata['learning_cycles'] += 1

        # Адаптация скорости обучения
        # Чем больше взаимодействий, тем медленнее меняется скорость
        total_messages = self.metadata['total_messages']
        if total_messages > 100:
            self.patterns['learning_speed'] = max(0.5, 1.0 - (total_messages / 10000))

        # Анализ успешности обучения
        # (можно расширить, добавив feedback от пользователя)

    # ==================== ПОИСК И ВООБРАЖЕНИЕ ====================
    async def search_memory(
            self,
            query: str,
            search_in: str = 'all',
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Поиск в памяти с поддержкой:
        - Точного совпадения
        - Семантического поиска (по схожести)
        - Ассоциативного поиска
        """
        results = []
        query_lower = query.lower()

        # 1. Поиск в долгосрочной памяти
        if search_in in ['all', 'long_term']:
            for mem in self.long_term:
                # Точное совпадение
                if query_lower in mem['content'].lower():
                    score = 1.0
                # Схожесть по keywords
                elif any(kw in query_lower for kw in mem.get('keywords', [])):
                    score = 0.8
                # Семантическая схожесть
                else:
                    score = MemoryUtils.calculate_similarity(query, mem['content'])

                if score > 0.3:
                    results.append({
                        'source': 'long_term',
                        'score': score,
                        'content': mem['content'],
                        'category': mem.get('category', 'general'),
                        'importance': mem.get('importance', 0.5),
                        'metadata': mem.get('metadata', {})
                    })

        # 2. Поиск в эпизодической памяти
        if search_in in ['all', 'episodic']:
            for episode in self.episodic_memory:
                if query_lower in episode['summary'].lower():
                    results.append({
                        'source': 'episodic',
                        'score': 0.9,
                        'content': episode['summary'],
                        'topics': episode.get('main_topics', []),
                        'importance': episode.get('importance', 0.5)
                    })

        # 3. Ассоциативный поиск
        keywords = MemoryUtils.extract_keywords(query)
        for keyword in keywords:
            associations = self.associative.get_associations(keyword, min_strength=0.4, limit=5)
            for related, strength in associations:
                results.append({
                    'source': 'associative',
                    'score': strength,
                    'content': f"Ассоциация: {keyword} → {related}",
                    'related_concept': related,
                    'strength': strength
                })

        # Сортировка по релевантности
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:limit]

    async def imagine_from_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Воображение: создание новых идей на основе запроса
        """
        # Извлекаем концепты из запроса
        keywords = MemoryUtils.extract_keywords(query, top_n=3)

        if len(keywords) < 2:
            return None

        # Используем ассоциативную память для воображения
        imagination = self.associative.imagine(keywords)

        if imagination:
            self.metadata['imaginations_generated'] += 1
            print(f"💭 Воображение: {imagination['description']}")

        return imagination

    # ==================== КОНТЕКСТ ДЛЯ LLM ====================
    def get_context_for_llm(self, max_tokens: int = 2000) -> str:
        """
        Формирование контекста для LLM из памяти
        """
        context_parts = []

        # 1. Рабочая память (текущий разговор)
        if self.working_memory.get('conversation_context'):
            context_parts.append("=== ТЕКУЩИЙ РАЗГОВОР ===")
            for msg in self.working_memory['conversation_context'][-5:]:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        # 2. Релевантные факты из долгосрочной памяти
        if self.long_term:
            # Берём самые важные и недавно использованные
            relevant_facts = sorted(
                self.long_term,
                key=lambda x: (x['importance'], x['access_count']),
                reverse=True
            )[:10]

            if relevant_facts:
                context_parts.append("\n=== ИЗВЕСТНЫЕ ФАКТЫ ===")
                for fact in relevant_facts:
                    context_parts.append(f"• {fact['content']}")

        # 3. Паттерны пользователя
        context_parts.append(f"\n=== ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ ===")
        context_parts.append(f"Стиль: {self.patterns.get('communication_style', 'neutral')}")

        if self.patterns.get('frequent_topics'):
            topics = ', '.join(self.patterns['frequent_topics'][:5])
            context_parts.append(f"Интересы: {topics}")

        # 4. Недавние эпизоды
        if self.episodic_memory:
            recent_episodes = self.episodic_memory[-3:]
            context_parts.append("\n=== НЕДАВНИЕ РАЗГОВОРЫ ===")
            for ep in recent_episodes:
                context_parts.append(f"• {ep['summary']}")

        context = "\n".join(context_parts)

        # Ограничиваем размер
        if len(context) > max_tokens * 4:  # примерно 4 символа на токен
            context = context[:max_tokens * 4]

        return context

    # ==================== СОХРАНЕНИЕ ====================
    async def _auto_save(self):
        """Автосохранение при накоплении изменений"""
        self._changes_count += 1

        if self._changes_count >= self._autosave_threshold:
            self._save_all()
            self._changes_count = 0

    def _save_all(self):
        """Сохранение всех данных"""
        self._save_json(self.short_term_file, self.short_term)
        self._save_json(self.long_term_file, self.long_term)
        self._save_json(self.working_file, self.working_memory)
        self._save_json(self.episodic_file, self.episodic_memory)
        self._save_json(self.patterns_file, self.patterns)
        self._save_json(self.preferences_file, self.preferences)
        self._save_json(self.metadata_file, self.metadata)

        # Сохраняем ассоциации
        self.associative.save()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика памяти"""
        return {
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'episodic_count': len(self.episodic_memory),
            'total_messages': self.metadata['total_messages'],
            'facts_extracted': self.metadata['facts_extracted'],
            'patterns_identified': self.metadata['patterns_identified'],
            'auto_transfers': self.metadata['auto_transfers'],
            'associations': len(self.associative.associations),
            'imaginations': len(self.associative.imaginations),
            'learning_cycles': self.metadata['learning_cycles'],
            'communication_style': self.patterns['communication_style'],
            'frequent_topics': self.patterns['frequent_topics'][:10],
            'cognitive_patterns': self.patterns.get('cognitive_patterns', {}),
            'learning_speed': self.patterns.get('learning_speed', 1.0)
        }


# ==================== LLM ИНТЕРФЕЙС ====================
class LLMInterface:
    """Интерфейс для работы с LM Studio"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Инициализация сессии"""
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()

    async def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000
    ) -> str:
        """Генерация ответа"""
        if not self.session:
            await self.initialize()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with self.session.post(
                    self.api_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    print(f"⚠️ LLM ошибка: {response.status}")
                    return "Не удалось получить ответ от LLM"
        except Exception as e:
            print(f"⚠️ LLM исключение: {e}")
            return f"Ошибка LLM: {str(e)}"


# ==================== ВЕБ-ПОИСК ====================
class WebSearcher:
    """Поиск информации в интернете"""

    def __init__(self):
        self.cache_file = WEB_CACHE_FILE
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Загрузка кеша"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки кеша: {e}")
        return {}

    def _save_cache(self):
        """Сохранение кеша"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения кеша: {e}")

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Поиск в интернете"""
        # Проверка кеша
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            # Проверка свежести (1 час)
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=1):
                print(f"📦 Использован кеш для: {query}")
                return cached['results']

        if not DDGS_AVAILABLE:
            return []

        try:
            results = []

            if DDGS_ASYNC:
                async with AsyncDDGS() as ddgs:
                    search_results = ddgs.text(query, max_results=max_results)
                    async for r in search_results:
                        results.append({
                            'title': r.get('title', ''),
                            'url': r.get('href', ''),
                            'snippet': r.get('body', ''),
                            'source': 'duckduckgo'
                        })
            else:
                # Синхронный режим
                ddgs = DDGS()
                search_results = ddgs.text(query, max_results=max_results)
                for r in search_results:
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'source': 'duckduckgo'
                    })

            # Сохраняем в кеш
            self.cache[cache_key] = {
                'query': query,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()

            print(f"🔍 Найдено {len(results)} результатов для: {query}")
            return results

        except Exception as e:
            print(f"⚠️ Ошибка веб-поиска: {e}")
            return []


# ==================== МЕНЕДЖЕР ОБРАБОТКИ ЗАПРОСОВ ====================
class QueryManager:
    """Управление обработкой запросов пользователя"""

    def __init__(self, user_id: str, llm: LLMInterface, web_searcher: WebSearcher):
        self.user_id = user_id
        self.llm = llm
        self.web_searcher = web_searcher
        self.memory = EnhancedMemorySystem(user_id)

    async def initialize(self):
        """Инициализация"""
        print(f"🎯 QueryManager инициализирован для user_{self.user_id}")

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Обработка запроса пользователя
        """
        # 1. Анализируем и сохраняем запрос
        await self.memory.analyze_and_store_message(
            role='user',
            content=query,
            llm_caller=self.llm.generate
        )

        # 2. Определяем стратегию ответа
        strategy = await self._determine_strategy(query)

        # 3. Обрабатываем в зависимости от стратегии
        result = {'strategy': strategy, 'confidence': 0.0, 'answer': ''}

        if strategy == 'direct_answer':
            result = await self._handle_direct_answer(query)
        elif strategy == 'web_search':
            result = await self._handle_web_search(query)
        elif strategy == 'memory_search':
            result = await self._handle_memory_search(query)
        elif strategy == 'imagination':
            result = await self._handle_imagination(query)
        else:
            result = await self._handle_llm_generation(query)

        # 4. Сохраняем ответ
        await self.memory.analyze_and_store_message(
            role='assistant',
            content=result['answer']
        )

        return result

    async def _determine_strategy(self, query: str) -> str:
        """Определение стратегии обработки"""
        query_lower = query.lower()

        # Поиск в интернете
        web_triggers = ['курс', 'погода', 'новости', 'цена', 'купить', 'найди', 'поищи']
        if any(trigger in query_lower for trigger in web_triggers):
            return 'web_search'

        # Поиск в памяти
        memory_triggers = ['помнишь', 'ты знаешь', 'что я говорил', 'расскажи обо мне']
        if any(trigger in query_lower for trigger in memory_triggers):
            return 'memory_search'

        # Воображение
        imagination_triggers = ['представь', 'вообрази', 'придумай', 'создай идею']
        if any(trigger in query_lower for trigger in imagination_triggers):
            return 'imagination'

        # Прямой ответ (вычисления, дата/время)
        if re.search(r'\d+\s*[\+\-\*/]\s*\d+', query) or 'сколько' in query_lower:
            return 'direct_answer'

        return 'llm_generation'

    async def _handle_direct_answer(self, query: str) -> Dict[str, Any]:
        """Прямой ответ (вычисления, факты)"""
        # Простой калькулятор
        math_match = re.search(r'(\d+)\s*([\+\-\*/])\s*(\d+)', query)
        if math_match:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            operations = {'+': a + b, '-': a - b, '*': a * b, '/': a / b if b != 0 else 'ошибка деления на ноль'}
            result = operations.get(op, 'неизвестная операция')
            return {
                'strategy': 'direct_answer',
                'confidence': 1.0,
                'answer': f"Результат: {result}"
            }

        # Дата/время
        if 'дата' in query.lower() or 'время' in query.lower() or 'день' in query.lower():
            now = datetime.now()
            return {
                'strategy': 'direct_answer',
                'confidence': 1.0,
                'answer': f"Сейчас: {now.strftime('%d.%m.%Y %H:%M:%S')}, {now.strftime('%A')}"
            }

        return {
            'strategy': 'direct_answer',
            'confidence': 0.3,
            'answer': 'Не удалось дать прямой ответ'
        }

    async def _handle_web_search(self, query: str) -> Dict[str, Any]:
        """Поиск в интернете"""
        results = await self.web_searcher.search(query, max_results=5)

        if not results:
            return {
                'strategy': 'web_search',
                'confidence': 0.0,
                'answer': 'Не удалось найти информацию в интернете'
            }

        # Формируем ответ из результатов
        answer = f"🔍 **Результаты поиска:** `{query}`\n\n"

        for i, res in enumerate(results[:3], 1):
            answer += f"{i}. **{res['title']}**\n"
            answer += f"   {res['snippet'][:150]}...\n"
            answer += f"   {res['url']}\n\n"

        return {
            'strategy': 'web_search',
            'confidence': 0.9,
            'answer': answer,
            'sources': results
        }

    async def _handle_memory_search(self, query: str) -> Dict[str, Any]:
        """Поиск в памяти"""
        results = await self.memory.search_memory(query, search_in='all', limit=5)

        if not results:
            return {
                'strategy': 'memory_search',
                'confidence': 0.0,
                'answer': 'Ничего не найдено в моей памяти'
            }

        answer = f"🧠 **Найдено в памяти:** `{query}`\n\n"

        for i, mem in enumerate(results, 1):
            source = mem.get('source', 'unknown')
            score = mem.get('score', 0.0)
            content = mem.get('content', '')[:200]

            answer += f"{i}. [{source}] 📊 {score:.2f}\n"
            answer += f"   {content}\n\n"

        return {
            'strategy': 'memory_search',
            'confidence': 0.8,
            'answer': answer,
            'results': results
        }

    async def _handle_imagination(self, query: str) -> Dict[str, Any]:
        """Генерация воображаемых идей"""
        imagination = await self.memory.imagine_from_query(query)

        if not imagination:
            # Fallback к обычной генерации
            return await self._handle_llm_generation(query)

        answer = f"💭 **Воображение:**\n\n"
        answer += f"Основа: {', '.join(imagination['base_concepts'])}\n"
        answer += f"Идея: {imagination['imagined_concept']}\n"
        answer += f"Сила связи: {imagination['strength']:.2f}\n\n"
        answer += f"{imagination['description']}"

        return {
            'strategy': 'imagination',
            'confidence': imagination['strength'],
            'answer': answer,
            'imagination': imagination
        }

    async def _handle_llm_generation(self, query: str) -> Dict[str, Any]:
        """Генерация ответа через LLM"""
        # Получаем контекст из памяти
        context = self.memory.get_context_for_llm(max_tokens=1500)

        # Формируем промпт
        system_prompt = f"""Ты - умный ассистент с памятью.

{context}

Отвечай на основе контекста и своих знаний. Будь полезным и естественным."""

        # Генерируем ответ
        response = await self.llm.generate(
            prompt=query,
            system=system_prompt,
            temperature=0.7,
            max_tokens=2000
        )

        return {
            'strategy': 'llm_generation',
            'confidence': 0.7,
            'answer': response
        }


# ==================== TELEGRAM БОТ ====================
class TelegramBot:
    """Telegram-интерфейс бота"""

    def __init__(self):
        self.llm = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.web_searcher = WebSearcher()
        self.managers: Dict[str, QueryManager] = {}

    async def initialize(self):
        """Инициализация"""
        await self.llm.initialize()
        print("✅ TelegramBot инициализирован")

    async def close(self):
        """Закрытие"""
        await self.llm.close()

    def get_manager(self, user_id: str) -> QueryManager:
        """Получение или создание менеджера для пользователя"""
        if user_id not in self.managers:
            self.managers[user_id] = QueryManager(user_id, self.llm, self.web_searcher)
        return self.managers[user_id]

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        await update.message.reply_text(
            "🧠 **Мини-Мозг Бот v9.0**\n\n"
            "Улучшенная система памяти и обучения!\n\n"
            "✨ **Новые возможности:**\n"
            "• 📚 Эпизодическая память\n"
            "• 🌐 Ассоциативная память\n"
            "• 💭 Воображение через ассоциации\n"
            "• 🧠 Когнитивные паттерны\n"
            "• 🎯 Интеллектуальное обучение\n\n"
            "Используй /help для справки",
            parse_mode='Markdown'
        )

    async def memory_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)

        stats = manager.memory.get_stats()

        msg = (
            f"🧠 **СТАТИСТИКА ПАМЯТИ**\n\n"
            f"📝 Краткосрочная: {stats['short_term_count']} сообщений\n"
            f"📚 Долгосрочная: {stats['long_term_count']} фактов\n"
            f"📖 Эпизодическая: {stats['episodic_count']} эпизодов\n"
            f"🌐 Ассоциации: {stats['associations']} связей\n"
            f"💭 Воображения: {stats['imaginations']}\n\n"
            f"📊 **Статистика:**\n"
            f"• Всего сообщений: {stats['total_messages']}\n"
            f"• Извлечено фактов: {stats['facts_extracted']}\n"
            f"• Выявлено паттернов: {stats['patterns_identified']}\n"
            f"• Автопереносов в LT: {stats['auto_transfers']}\n"
            f"• Циклов обучения: {stats['learning_cycles']}\n\n"
            f"💬 **Ваш профиль:**\n"
            f"• Стиль: {stats['communication_style']}\n"
            f"• Скорость обучения: {stats['learning_speed']:.2f}\n"
        )

        if stats['frequent_topics']:
            topics = ", ".join(stats['frequent_topics'][:5])
            msg += f"• Частые темы: {topics}\n"

        if stats['cognitive_patterns']:
            patterns = ", ".join([f"{k}({v})" for k, v in list(stats['cognitive_patterns'].items())[:3]])
            msg += f"• Когнитивные паттерны: {patterns}\n"

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

        results = await manager.memory.search_memory(query, search_in='all', limit=5)

        if not results:
            await update.message.reply_text(
                f"❌ Ничего не найдено по запросу: `{query}`",
                parse_mode='Markdown'
            )
            return

        msg = f"🔍 **Найдено:** `{query}`\n\n"
        for i, mem in enumerate(results, 1):
            source = mem.get('source', 'unknown')
            score = mem.get('score', 0.0)
            msg += f"{i}. [{source}] 📊 {score:.2f}\n"
            msg += f"   {mem.get('content', '')[:150]}\n\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def imagine(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда воображения"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        query = ' '.join(context.args) if context.args else ''

        if not query:
            await update.message.reply_text(
                "💭 Использование: `/imagine концепт1 концепт2`",
                parse_mode='Markdown'
            )
            return

        imagination = await manager.memory.imagine_from_query(query)

        if not imagination:
            await update.message.reply_text(
                "❌ Не удалось создать воображение. Нужно больше концептов в памяти.",
                parse_mode='Markdown'
            )
            return

        msg = (
            f"💭 **ВООБРАЖЕНИЕ**\n\n"
            f"Основа: {', '.join(imagination['base_concepts'])}\n"
            f"Идея: {imagination['imagined_concept']}\n"
            f"Сила: {'⭐' * min(5, int(imagination['strength'] * 5))}\n\n"
            f"{imagination['description']}"
        )

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистка краткосрочной памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)

        manager.memory.short_term = []
        manager.memory.working_memory['conversation_context'] = []
        manager.memory._save_all()

        await update.message.reply_text(
            "🧹 **Краткосрочная память очищена!**\n"
            "Долгосрочная, эпизодическая и ассоциативная память сохранены.",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Справка"""
        await update.message.reply_text(
            "📖 **СПРАВКА - МИНИ-МОЗГ БОТ v9.0**\n\n"
            "🧠 **Система памяти:**\n"
            "• Краткосрочная - последние сообщения\n"
            "• Долгосрочная - важные факты и знания\n"
            "• Эпизодическая - воспоминания о разговорах\n"
            "• Ассоциативная - связи между идеями\n"
            "• Рабочая - текущий контекст\n\n"
            "💭 **Воображение:**\n"
            "Бот создаёт новые идеи, комбинируя\n"
            "известные концепты через ассоциации!\n\n"
            "📝 **Команды:**\n"
            "/memory - статистика памяти\n"
            "/search_memory <запрос> - поиск\n"
            "/imagine <концепты> - воображение\n"
            "/clear - очистка ST памяти\n"
            "/help - эта справка\n\n"
            "💬 **Просто пиши мне, и я:**\n"
            "• Запомню важное\n"
            "• Найду информацию\n"
            "• Создам ассоциации\n"
            "• Буду учиться и адаптироваться!",
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            manager = self.get_manager(user_id)

            # Инициализация
            if not hasattr(manager, '_initialized'):
                await manager.initialize()
                manager._initialized = True

            # Обработка запроса
            result = await manager.process_query(text)

            # Отправка ответа
            answer = result.get('answer', 'Не удалось сформировать ответ')

            # Ограничение длины для Telegram
            if len(answer) > 4000:
                answer = answer[:3950] + "\n\n...(усечено)"

            await update.message.reply_text(
                answer,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )

            # Логирование
            strategy = result.get('strategy', 'unknown')
            confidence = result.get('confidence', 0)
            print(f"✅ Ответ | Стратегия: {strategy} | Уверенность: {confidence:.2f}")

        except Exception as e:
            error_msg = f"❌ **Ошибка:** {str(e)[:200]}"
            print(f"ERROR: {e}")
            traceback.print_exc()

            await update.message.reply_text(error_msg, parse_mode='Markdown')


# ==================== ЗАПУСК ====================
async def main_async():
    """Асинхронная главная функция"""
    print("\n" + "=" * 70)
    print("🚀 МИНИ-МОЗГ БОТ v9.0 - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("=" * 70)
    print(f"⏰ Запуск: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"🧠 Память: {MEMORY_DIR}/")
    print(f"🌐 Веб-поиск: {'✅' if DDGS_AVAILABLE else '❌'}")
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
    application.add_handler(CommandHandler("imagine", bot.imagine))
    application.add_handler(CommandHandler("clear", bot.clear))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот готов!")
    print("=" * 70)
    print("💡 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:")
    print("   • 🧠 5 типов памяти (ST, LT, Episodic, Associative, Working)")
    print("   • 💭 Воображение через ассоциативные связи")
    print("   • 🎯 Интеллектуальный перенос в LT память")
    print("   • 📊 Когнитивные паттерны и мета-обучение")
    print("   • 🔗 Граф ассоциаций между концептами")
    print("   • 📖 Эпизодическая память разговоров")
    print("   • ⚡ Улучшенная архитектура кода")
    print("=" * 70)
    print("\nCtrl+C для остановки\n")

    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)

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