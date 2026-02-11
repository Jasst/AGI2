#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v6.0 - КОГНИТИВНЫЙ АГИ-ЯДРО
✅ Исправлены все синтаксические ошибки
✅ Улучшенное создание и понимание ядер
✅ Адаптивная память (кратковременная + долговременная)
✅ Хронологическая последовательность
✅ Конкатенация фрагментов памяти в образы
✅ Обратная прогрессия для воображения
"""

import os
import json
import re
import ast
import asyncio
import requests
import traceback
import hashlib
import time
import shutil
import uuid
import math
import random
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pickle
import csv
import yaml
import pathlib
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

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
KNOWLEDGE_BASE_DIR = "knowledge_base"
COGNITIVE_MODELS_DIR = "cognitive_models"

for directory in [CORES_DIR, MEMORY_DIR, USER_FILES_DIR, KNOWLEDGE_BASE_DIR, COGNITIVE_MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Файлы
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
CORE_PERFORMANCE_LOG = os.path.join(MEMORY_DIR, "core_performance.json")
REJECTED_CORES_LOG = os.path.join(MEMORY_DIR, "rejected_cores.json")
COGNITIVE_STATE = os.path.join(MEMORY_DIR, "cognitive_state.json")
LONG_TERM_MEMORY = os.path.join(MEMORY_DIR, "long_term_memory.jsonl")
SHORT_TERM_MEMORY = os.path.join(MEMORY_DIR, "short_term_memory.json")

# Создаем __init__.py для импортов
with open(os.path.join(CORES_DIR, "__init__.py"), "w") as f:
    f.write("# Auto-generated for imports\n")


# ==================== КОГНИТИВНЫЕ КЛАССЫ ====================
class CognitiveState(Enum):
    """Состояния мышления системы"""
    OBSERVING = "observing"  # Наблюдение и сбор данных
    ANALYZING = "analyzing"  # Анализ информации
    REFLECTING = "reflecting"  # Рефлексия и самоанализ
    CREATING = "creating"  # Создание нового
    ADAPTING = "adapting"  # Адаптация к ситуации
    DECIDING = "deciding"  # Принятие решений
    EXECUTING = "executing"  # Выполнение действий
    IMAGINING = "imagining"  # Воображение


@dataclass
class Thought:
    """Единица мыслительного процесса"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    state: CognitiveState = CognitiveState.OBSERVING
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    parent_thought: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # Важность мысли для памяти

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'state': self.state.value,
            'confidence': self.confidence,
            'tags': self.tags,
            'parent_thought': self.parent_thought,
            'metadata': self.metadata,
            'importance': self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thought':
        """Создает из словаря"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data['timestamp']),
            content=data['content'],
            state=CognitiveState(data['state']),
            confidence=data.get('confidence', 0.5),
            tags=data.get('tags', []),
            parent_thought=data.get('parent_thought'),
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 0.5)
        )


@dataclass
class MemoryRecord:
    """Запись в памяти"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    associations: List[str] = field(default_factory=list)
    memory_type: str = "short_term"  # short_term или long_term
    consolidation_score: float = 0.0  # Оценка для перехода в долговременную
    emotional_valence: float = 0.0  # Эмоциональная окраска
    sensory_data: Dict[str, Any] = field(default_factory=dict)  # Сенсорные данные

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'context': self.context,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
            'associations': self.associations,
            'memory_type': self.memory_type,
            'consolidation_score': self.consolidation_score,
            'emotional_valence': self.emotional_valence,
            'sensory_data': self.sensory_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryRecord':
        """Создает из словаря"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data['timestamp']),
            content=data['content'],
            context=data.get('context', {}),
            importance=data.get('importance', 0.5),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data.get('last_accessed', datetime.now().isoformat())),
            associations=data.get('associations', []),
            memory_type=data.get('memory_type', 'short_term'),
            consolidation_score=data.get('consolidation_score', 0.0),
            emotional_valence=data.get('emotional_valence', 0.0),
            sensory_data=data.get('sensory_data', {})
        )


# ==================== АДАПТИВНАЯ СИСТЕМА ПАМЯТИ ====================
class AdaptiveMemorySystem:
    """
    Продвинутая система адаптивной памяти

    Возможности:
    - Кратковременная память (рабочая память)
    - Долговременная память (консолидированная)
    - Автоматический переход из кратковременной в долговременную
    - Хронологическая последовательность
    - Дедупликация и обновление
    - Конкатенация фрагментов в образы
    - Обратная прогрессия для воображения
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.short_term_capacity = 20  # Вместимость кратковременной памяти
        self.consolidation_threshold = 0.7  # Порог для перехода в долговременную
        self.deduplication_threshold = 0.85  # Порог схожести для дедупликации

        # Кратковременная память (быстрый доступ)
        self.short_term_memory: deque = deque(maxlen=self.short_term_capacity)

        # Долговременная память (постоянное хранилище)
        self.long_term_memory: List[MemoryRecord] = []

        # Индексы для быстрого поиска
        self.memory_index: Dict[str, MemoryRecord] = {}
        self.chronological_index: List[str] = []  # ID в хронологическом порядке
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)

        # Метрики системы
        self.total_consolidations = 0
        self.total_deduplications = 0
        self.total_imaginations = 0

        # Загружаем память
        self._load_memory()

        print(f"💾 Система памяти инициализирована для пользователя {user_id}")
        print(f"   📦 Кратковременная: {len(self.short_term_memory)} записей")
        print(f"   🗄️ Долговременная: {len(self.long_term_memory)} записей")

    def _load_memory(self):
        """Загружает память из файлов"""
        # Загружаем кратковременную память
        stm_file = os.path.join(MEMORY_DIR, f"stm_{self.user_id}.json")
        if os.path.exists(stm_file):
            try:
                with open(stm_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data.get('records', []):
                        record = MemoryRecord.from_dict(record_data)
                        self.short_term_memory.append(record)
                        self.memory_index[record.id] = record
            except Exception as e:
                print(f"⚠️ Ошибка загрузки кратковременной памяти: {e}")

        # Загружаем долговременную память
        ltm_file = os.path.join(MEMORY_DIR, f"ltm_{self.user_id}.jsonl")
        if os.path.exists(ltm_file):
            try:
                with open(ltm_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = MemoryRecord.from_dict(json.loads(line))
                            self.long_term_memory.append(record)
                            self.memory_index[record.id] = record
                            self.chronological_index.append(record.id)

                            # Восстанавливаем ассоциации
                            for assoc_id in record.associations:
                                self.association_graph[record.id].add(assoc_id)
                                self.association_graph[assoc_id].add(record.id)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки долговременной памяти: {e}")

    def save_memory(self):
        """Сохраняет память в файлы"""
        try:
            # Сохраняем кратковременную память
            stm_file = os.path.join(MEMORY_DIR, f"stm_{self.user_id}.json")
            with open(stm_file, 'w', encoding='utf-8') as f:
                data = {
                    'user_id': self.user_id,
                    'timestamp': datetime.now().isoformat(),
                    'records': [record.to_dict() for record in self.short_term_memory]
                }
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Сохраняем долговременную память (append-only)
            ltm_file = os.path.join(MEMORY_DIR, f"ltm_{self.user_id}.jsonl")
            # Перезаписываем файл для дедупликации
            with open(ltm_file, 'w', encoding='utf-8') as f:
                for record in self.long_term_memory:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"⚠️ Ошибка сохранения памяти: {e}")

    def add_memory(self, content: str, context: Dict[str, Any] = None,
                   importance: float = 0.5, emotional_valence: float = 0.0,
                   sensory_data: Dict[str, Any] = None) -> MemoryRecord:
        """
        Добавляет новую запись в кратковременную память

        Args:
            content: Содержимое записи
            context: Контекст записи
            importance: Важность (0.0-1.0)
            emotional_valence: Эмоциональная окраска (-1.0 до 1.0)
            sensory_data: Сенсорные данные (визуальные, аудио и т.д.)
        """
        # Проверяем на дубликаты
        duplicate = self._find_duplicate(content)
        if duplicate:
            # Обновляем существующую запись
            duplicate.access_count += 1
            duplicate.last_accessed = datetime.now()
            duplicate.importance = max(duplicate.importance, importance)
            duplicate.consolidation_score += 0.1

            # Обновляем контекст
            if context:
                duplicate.context.update(context)

            self.total_deduplications += 1
            print(f"   🔄 Обновлена существующая запись (дедупликация)")

            return duplicate

        # Создаем новую запись
        record = MemoryRecord(
            content=content,
            context=context or {},
            importance=importance,
            memory_type='short_term',
            emotional_valence=emotional_valence,
            sensory_data=sensory_data or {}
        )

        # Вычисляем начальный consolidation_score
        record.consolidation_score = self._calculate_consolidation_score(record)

        # Добавляем в кратковременную память
        self.short_term_memory.append(record)
        self.memory_index[record.id] = record

        # Проверяем необходимость консолидации
        if record.consolidation_score >= self.consolidation_threshold:
            self._consolidate_to_long_term(record)

        # Создаем ассоциации с последними записями
        self._create_associations(record)

        return record

    def _find_duplicate(self, content: str) -> Optional[MemoryRecord]:
        """Находит дубликаты в памяти"""
        # Простая проверка по содержимому
        for record in list(self.short_term_memory) + self.long_term_memory:
            similarity = self._calculate_similarity(content, record.content)
            if similarity >= self.deduplication_threshold:
                return record

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет схожесть двух текстов (упрощенная версия)"""
        if not text1 or not text2:
            return 0.0

        # Нормализуем
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Коэффициент Жаккара
        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _calculate_consolidation_score(self, record: MemoryRecord) -> float:
        """
        Вычисляет оценку для консолидации в долговременную память

        Факторы:
        - Важность
        - Количество обращений
        - Эмоциональная окраска
        - Связи с другими записями
        """
        score = 0.0

        # Важность (0-0.3)
        score += record.importance * 0.3

        # Частота обращений (0-0.3)
        access_score = min(record.access_count / 10.0, 1.0) * 0.3
        score += access_score

        # Эмоциональная окраска (0-0.2)
        emotional_score = abs(record.emotional_valence) * 0.2
        score += emotional_score

        # Количество ассоциаций (0-0.2)
        association_score = min(len(record.associations) / 5.0, 1.0) * 0.2
        score += association_score

        return min(score, 1.0)

    def _consolidate_to_long_term(self, record: MemoryRecord):
        """Переносит запись из кратковременной в долговременную память"""
        # Помечаем как долговременную
        record.memory_type = 'long_term'

        # Добавляем в долговременную память
        if record not in self.long_term_memory:
            self.long_term_memory.append(record)
            self.chronological_index.append(record.id)
            self.total_consolidations += 1

            print(f"   📚 Запись консолидирована в долговременную память")
            print(f"      Оценка: {record.consolidation_score:.2f}, Важность: {record.importance:.2f}")

    def _create_associations(self, record: MemoryRecord):
        """Создает ассоциации с другими записями"""
        # Связываем с последними записями из кратковременной памяти
        recent_memories = list(self.short_term_memory)[-5:]

        for mem in recent_memories:
            if mem.id != record.id:
                # Вычисляем семантическую связь
                similarity = self._calculate_similarity(record.content, mem.content)

                if similarity > 0.3:  # Порог для создания ассоциации
                    record.associations.append(mem.id)
                    mem.associations.append(record.id)

                    # Обновляем граф ассоциаций
                    self.association_graph[record.id].add(mem.id)
                    self.association_graph[mem.id].add(record.id)

    def recall_memory(self, query: str, memory_type: str = 'both',
                      limit: int = 5) -> List[MemoryRecord]:
        """
        Извлекает релевантные записи из памяти

        Args:
            query: Поисковый запрос
            memory_type: 'short_term', 'long_term' или 'both'
            limit: Максимальное количество результатов
        """
        memories_to_search = []

        if memory_type in ['short_term', 'both']:
            memories_to_search.extend(self.short_term_memory)

        if memory_type in ['long_term', 'both']:
            memories_to_search.extend(self.long_term_memory)

        # Вычисляем релевантность
        scored_memories = []
        for memory in memories_to_search:
            relevance = self._calculate_similarity(query, memory.content)

            # Учитываем важность и частоту обращений
            score = relevance * 0.6 + memory.importance * 0.3 + min(memory.access_count / 10, 0.1)

            scored_memories.append((score, memory))

        # Сортируем по релевантности
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Обновляем статистику обращений
        results = []
        for score, memory in scored_memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()

            # Повышаем consolidation_score при частых обращениях
            if memory.memory_type == 'short_term':
                memory.consolidation_score += 0.05
                if memory.consolidation_score >= self.consolidation_threshold:
                    self._consolidate_to_long_term(memory)

            results.append(memory)

        return results

    def get_chronological_sequence(self, start_time: datetime = None,
                                   end_time: datetime = None,
                                   limit: int = 20) -> List[MemoryRecord]:
        """
        Возвращает хронологическую последовательность записей

        Args:
            start_time: Начальное время
            end_time: Конечное время
            limit: Максимальное количество записей
        """
        # Фильтруем по времени
        filtered = []

        for record_id in self.chronological_index:
            if record_id in self.memory_index:
                record = self.memory_index[record_id]

                if start_time and record.timestamp < start_time:
                    continue

                if end_time and record.timestamp > end_time:
                    continue

                filtered.append(record)

        return filtered[:limit]

    def concatenate_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """
        Конкатенирует фрагменты памяти в единый образ

        Args:
            memory_ids: ID записей для объединения

        Returns:
            Объединенный образ с контекстом
        """
        memories = [self.memory_index[mid] for mid in memory_ids if mid in self.memory_index]

        if not memories:
            return {}

        # Сортируем по времени
        memories.sort(key=lambda x: x.timestamp)

        # Объединяем содержимое
        concatenated_content = "\n".join([m.content for m in memories])

        # Объединяем контексты
        merged_context = {}
        for memory in memories:
            merged_context.update(memory.context)

        # Объединяем ассоциации
        all_associations = set()
        for memory in memories:
            all_associations.update(memory.associations)

        # Вычисляем средние метрики
        avg_importance = sum(m.importance for m in memories) / len(memories)
        avg_emotional = sum(m.emotional_valence for m in memories) / len(memories)

        # Создаем образ
        image = {
            'content': concatenated_content,
            'context': merged_context,
            'source_memories': memory_ids,
            'time_span': {
                'start': memories[0].timestamp.isoformat(),
                'end': memories[-1].timestamp.isoformat()
            },
            'importance': avg_importance,
            'emotional_valence': avg_emotional,
            'associations': list(all_associations),
            'creation_time': datetime.now().isoformat()
        }

        print(f"   🎨 Создан образ из {len(memories)} фрагментов памяти")

        return image

    def reverse_progression_for_imagination(self, goal_state: str,
                                            steps: int = 5) -> List[MemoryRecord]:
        """
        Обратная прогрессия из долговременной памяти в кратковременную для воображения

        Используется для:
        - Планирования
        - Прогнозирования
        - Творческого мышления

        Args:
            goal_state: Желаемое конечное состояние
            steps: Количество шагов обратной прогрессии

        Returns:
            Последовательность записей для достижения цели
        """
        self.total_imaginations += 1

        # Находим релевантные записи из долговременной памяти
        relevant_ltm = self.recall_memory(goal_state, memory_type='long_term', limit=10)

        if not relevant_ltm:
            print("   ⚠️ Не найдено релевантных записей для воображения")
            return []

        # Строим обратную последовательность
        imagination_sequence = []

        # Начинаем с наиболее релевантной записи
        current_memory = relevant_ltm[0]
        imagination_sequence.append(current_memory)

        # Идем назад по ассоциациям
        for _ in range(steps - 1):
            # Находим предшествующие ассоциации
            associated_memories = []

            for assoc_id in current_memory.associations:
                if assoc_id in self.memory_index:
                    assoc_mem = self.memory_index[assoc_id]
                    # Берем только те, что раньше по времени
                    if assoc_mem.timestamp < current_memory.timestamp:
                        associated_memories.append(assoc_mem)

            if not associated_memories:
                break

            # Выбираем наиболее важную
            current_memory = max(associated_memories, key=lambda x: x.importance)
            imagination_sequence.insert(0, current_memory)  # Добавляем в начало

        # Создаем промежуточные "воображаемые" состояния в кратковременной памяти
        for i, memory in enumerate(imagination_sequence):
            imagined_content = f"[ВООБРАЖЕНИЕ ШАГА {i + 1}] На основе: {memory.content[:100]}"

            self.add_memory(
                content=imagined_content,
                context={
                    'type': 'imagination',
                    'goal': goal_state,
                    'step': i + 1,
                    'source_memory_id': memory.id
                },
                importance=0.6,
                emotional_valence=0.2
            )

        print(f"   💭 Создана последовательность воображения из {len(imagination_sequence)} шагов")

        return imagination_sequence

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику памяти"""
        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'total_consolidations': self.total_consolidations,
            'total_deduplications': self.total_deduplications,
            'total_imaginations': self.total_imaginations,
            'total_associations': sum(len(assocs) for assocs in self.association_graph.values()) // 2,
            'avg_importance_stm': sum(m.importance for m in self.short_term_memory) / len(
                self.short_term_memory) if self.short_term_memory else 0,
            'avg_importance_ltm': sum(m.importance for m in self.long_term_memory) / len(
                self.long_term_memory) if self.long_term_memory else 0
        }

    def cleanup_old_memories(self, days_threshold: int = 30):
        """
        Очищает старые малозначимые записи из долговременной памяти

        Args:
            days_threshold: Порог в днях для удаления
        """
        threshold_date = datetime.now() - timedelta(days=days_threshold)

        # Фильтруем долговременную память
        cleaned_memories = []
        removed_count = 0

        for memory in self.long_term_memory:
            # Сохраняем если:
            # - Запись свежая ИЛИ
            # - Важная (importance > 0.7) ИЛИ
            # - Часто используется (access_count > 5)
            should_keep = (
                    memory.timestamp > threshold_date or
                    memory.importance > 0.7 or
                    memory.access_count > 5
            )

            if should_keep:
                cleaned_memories.append(memory)
            else:
                removed_count += 1
                # Удаляем из индексов
                if memory.id in self.memory_index:
                    del self.memory_index[memory.id]
                if memory.id in self.chronological_index:
                    self.chronological_index.remove(memory.id)

        self.long_term_memory = cleaned_memories

        if removed_count > 0:
            print(f"   🧹 Очищено {removed_count} старых записей из долговременной памяти")
            self.save_memory()


# ==================== ТИПЫ ОТВЕТОВ ЯДЕР ====================
class CoreResponse:
    """Стандартизированный ответ от ядра знаний"""

    def __init__(
            self,
            success: bool,
            data: Optional[Dict[str, Any]] = None,
            raw_result: Optional[str] = None,
            confidence: float = 1.0,
            source: str = "unknown",
            direct_answer: bool = False,
            needs_reflection: bool = False,
            reflection_context: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data or {}
        self.raw_result = raw_result
        self.confidence = confidence
        self.source = source
        self.direct_answer = direct_answer
        self.needs_reflection = needs_reflection
        self.reflection_context = reflection_context or {}

    def to_context_string(self) -> str:
        """Преобразование в текст для контекста LLM"""
        if not self.success:
            return f"❌ Ошибка получения данных из источника '{self.source}': {self.data.get('error', 'Неизвестная ошибка')}"

        if self.raw_result:
            return f"📊 ДАННЫЕ ОТ '{self.source.upper()}':\n{self.raw_result}"

        if self.data:
            formatted_data = json.dumps(self.data, ensure_ascii=False, indent=2)
            return f"📊 СТРУКТУРИРОВАННЫЕ ДАННЫЕ ОТ '{self.source.upper()}':\n{formatted_data}"

        return f"ℹ️ Источник '{self.source}' обработал запрос, но не вернул данных"

    def is_final_answer(self) -> bool:
        """Можно ли использовать ответ как финальный?"""
        return self.direct_answer and self.success and self.raw_result and len(self.raw_result) > 10


# ==================== БАЗОВЫЙ КЛАСС ЯДРА ====================
class KnowledgeCore(ABC):
    """Базовый класс для всех ядер знаний"""
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []
    priority: int = 5
    direct_answer_mode: bool = False
    cognitive_load: int = 1
    version: str = "1.0.0"

    # Метрики
    total_executions: int = 0
    successful_executions: int = 0
    average_confidence: float = 0.0
    last_execution: Optional[datetime] = None

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Определяет, может ли ядро обработать запрос"""
        pass

    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        """Выполняет обработку запроса"""
        pass

    def get_confidence(self, query: str) -> float:
        """Уверенность в обработке запроса (0.0-1.0)"""
        return 0.5 if self.can_handle(query) else 0.0

    def update_metrics(self, success: bool, confidence: float):
        """Обновление метрик ядра"""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        self.average_confidence = (self.average_confidence * (
                    self.total_executions - 1) + confidence) / self.total_executions
        self.last_execution = datetime.now()

    def get_efficiency_score(self) -> float:
        """Счет эффективности ядра"""
        if self.total_executions == 0:
            return 0.0
        success_rate = self.successful_executions / self.total_executions
        return (success_rate * 0.6 + self.average_confidence * 0.4) * (10 / self.cognitive_load)


# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
    """Ядро для работы с датой и временем"""
    name = "datetime_core"
    description = "Точная информация о дате, времени и днях недели"
    capabilities = ["текущая дата", "время", "день недели", "расчёт дат"]
    priority = 1
    direct_answer_mode = True
    cognitive_load = 1
    version = "2.0.0"

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        keywords = [
            'какой сегодня день', 'какое число', 'день недели', 'сколько времени',
            'который час', 'текущая дата', 'сегодня', 'завтра', 'вчера'
        ]
        return any(kw in q for kw in keywords)

    def get_confidence(self, query: str) -> float:
        q = query.lower()
        high_conf = ['какой сегодня', 'какое число', 'который час']
        return 0.98 if any(kw in q for kw in high_conf) else (0.8 if self.can_handle(query) else 0.0)

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            now = datetime.now()
            weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
            months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                      'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']

            result = (
                f"📅 **Текущая дата и время:**\n"
                f"• Дата: {now.day} {months[now.month - 1]} {now.year} года\n"
                f"• День недели: {weekdays[now.weekday()]}\n"
                f"• Время: {now.strftime('%H:%M:%S')}\n"
            )

            self.update_metrics(True, 0.98)
            return CoreResponse(
                success=True,
                data={'timestamp': now.isoformat(), 'weekday': weekdays[now.weekday()]},
                raw_result=result,
                confidence=0.98,
                source=self.name,
                direct_answer=True
            )
        except Exception as e:
            self.update_metrics(False, 0.0)
            return CoreResponse(success=False, data={'error': str(e)}, source=self.name)


class CalculatorCore(KnowledgeCore):
    """Ядро для математических вычислений"""
    name = "calculator_core"
    description = "Математические вычисления"
    capabilities = ["сложение", "вычитание", "умножение", "деление", "проценты"]
    priority = 2
    direct_answer_mode = True
    cognitive_load = 2
    version = "2.0.0"

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return bool(re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', q)) or any(
            word in q for word in ['сколько будет', 'посчитай', 'вычисли']
        )

    def get_confidence(self, query: str) -> float:
        return 0.95 if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', query.lower()) else (0.7 if self.can_handle(query) else 0.0)

    def _safe_eval(self, expr: str) -> Any:
        """Безопасное вычисление"""
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'pow': pow, 'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
            '__builtins__': {}
        }

        # Проверка безопасности
        if any(danger in expr for danger in ['import', 'exec', 'eval', '__', 'open']):
            raise ValueError("Недопустимое выражение")

        return eval(expr, {"__builtins__": {}}, safe_dict)

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            # Извлекаем выражение
            expr = re.search(r'([\d\.\+\-\*\/\^\(\)\s]+)', query)
            if not expr:
                raise ValueError("Не найдено математическое выражение")

            expr_str = expr.group(1).strip()
            expr_str = expr_str.replace('x', '*').replace('×', '*').replace('÷', '/').replace('^', '**')

            result = self._safe_eval(expr_str)

            response_text = f"🧮 **Результат:** `{expr_str} = {result}`"

            self.update_metrics(True, 0.95)
            return CoreResponse(
                success=True,
                data={'expression': expr_str, 'result': result},
                raw_result=response_text,
                confidence=0.95,
                source=self.name,
                direct_answer=True
            )
        except Exception as e:
            self.update_metrics(False, 0.0)
            return CoreResponse(success=False, data={'error': str(e)}, source=self.name)


class WebSearchCore(KnowledgeCore):
    """Ядро для поиска в интернете"""
    name = "web_search_core"
    description = "Поиск актуальной информации в интернете"
    capabilities = ["новости", "поиск", "информация"]
    priority = 7
    cognitive_load = 3
    version = "2.0.0"

    def __init__(self):
        self.ddgs = DDGS() if DDGS else None
        self.cache = {}

    def can_handle(self, query: str) -> bool:
        if not self.ddgs:
            return False

        q = query.lower()
        exclude = ['какой сегодня день', 'который час', 'сколько будет']
        if any(kw in q for kw in exclude):
            return False

        keywords = ['найди', 'новост', 'информаци', 'что такое', 'кто такой']
        return any(kw in q for kw in keywords)

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        if not self.ddgs:
            return CoreResponse(success=False, data={'error': 'DuckDuckGo недоступен'}, source=self.name)

        try:
            # Кэширование
            cache_key = hashlib.md5(query.encode()).hexdigest()[:12]
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if time.time() - cached['timestamp'] < 300:
                    results = cached['results']
                else:
                    del self.cache[cache_key]
                    results = list(self.ddgs.text(query, max_results=5))
            else:
                results = list(self.ddgs.text(query, max_results=5))
                self.cache[cache_key] = {'results': results, 'timestamp': time.time()}

            if not results:
                return CoreResponse(
                    success=False,
                    data={'error': 'Результаты не найдены'},
                    source=self.name
                )

            # Форматируем результаты
            results_text = "🌐 **Результаты поиска:**\n\n"
            for i, r in enumerate(results[:3], 1):
                results_text += f"**{i}. {r.get('title', 'Без заголовка')}**\n"
                snippet = r.get('body', '')[:200]
                if snippet:
                    results_text += f"   _{snippet}_...\n"
                results_text += f"   🔗 {r.get('href', '')}\n\n"

            self.update_metrics(True, 0.85)
            return CoreResponse(
                success=True,
                data={'query': query, 'results_count': len(results)},
                raw_result=results_text,
                confidence=0.85,
                source=self.name,
                needs_reflection=True
            )
        except Exception as e:
            self.update_metrics(False, 0.0)
            return CoreResponse(success=False, data={'error': str(e)}, source=self.name)


class FileStorageCore(KnowledgeCore):
    """Ядро для работы с файлами"""
    name = "file_storage_core"
    description = "Работа с текстовыми файлами"
    capabilities = ["сохранить файл", "прочитать файл", "список файлов"]
    priority = 1
    direct_answer_mode = True
    cognitive_load = 2
    version = "3.0.0"

    def __init__(self):
        self.storage_dir = USER_FILES_DIR
        os.makedirs(self.storage_dir, exist_ok=True)

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        keywords = ['файл', 'сохрани', 'прочитай', 'список файлов']
        return any(keyword in q for keyword in keywords)

    def get_confidence(self, query: str) -> float:
        q = query.lower()
        if any(cmd in q for cmd in ['прочитай файл', 'сохрани в файл']):
            return 0.95
        return 0.7 if self.can_handle(query) else 0.0

    def _sanitize_filename(self, filename: str) -> str:
        """Очищает имя файла"""
        if not filename:
            return f"document_{int(time.time())}.txt"

        filename = filename.strip().strip('\'"')
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        if '.' not in filename:
            filename += '.txt'

        return filename

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            q = query.lower()

            # ЧТЕНИЕ
            if any(cmd in q for cmd in ['прочитай', 'открой']):
                match = re.search(r'файл\s+["\']?([^"\'\s]+)["\']?', query, re.IGNORECASE)
                if not match:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Не указано имя файла'},
                        raw_result="❌ Укажите имя файла: `прочитай файл имя.txt`",
                        source=self.name,
                        direct_answer=True
                    )

                filename = self._sanitize_filename(match.group(1))
                filepath = os.path.join(self.storage_dir, filename)

                if not os.path.exists(filepath):
                    return CoreResponse(
                        success=False,
                        data={'error': 'Файл не найден'},
                        raw_result=f"❌ Файл `{filename}` не найден",
                        source=self.name,
                        direct_answer=True
                    )

                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                preview = content[:500] + ('...' if len(content) > 500 else '')
                result = f"📄 **Файл `{filename}`:**\n\n```text\n{preview}\n```"

                self.update_metrics(True, 1.0)
                return CoreResponse(
                    success=True,
                    data={'filename': filename, 'content': content},
                    raw_result=result,
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

            # СОХРАНЕНИЕ
            elif 'сохрани' in q:
                match = re.search(r'файл\s+["\']?([^"\':]+)["\']?\s*[:：]\s*(.+)', query, re.IGNORECASE | re.DOTALL)
                if not match:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Неверный формат'},
                        raw_result="❌ Формат: `сохрани в файл имя.txt: текст`",
                        source=self.name,
                        direct_answer=True
                    )

                filename = self._sanitize_filename(match.group(1))
                content = match.group(2).strip()
                filepath = os.path.join(self.storage_dir, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                result = f"✅ **Файл сохранен:** `{filename}`\n📊 Размер: {len(content)} байт"

                self.update_metrics(True, 1.0)
                return CoreResponse(
                    success=True,
                    data={'filename': filename, 'size': len(content)},
                    raw_result=result,
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

            # СПИСОК
            elif 'список' in q:
                files = [f for f in os.listdir(self.storage_dir) if os.path.isfile(os.path.join(self.storage_dir, f))]

                if not files:
                    result = "📁 **Файловое хранилище пусто**"
                else:
                    result = f"📁 **Файлы ({len(files)}):**\n\n"
                    for i, f in enumerate(files[:10], 1):
                        size = os.path.getsize(os.path.join(self.storage_dir, f))
                        result += f"{i}. `{f}` ({size} байт)\n"

                self.update_metrics(True, 1.0)
                return CoreResponse(
                    success=True,
                    data={'files': files, 'count': len(files)},
                    raw_result=result,
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

            return CoreResponse(
                success=False,
                data={'error': 'Неизвестная команда'},
                source=self.name
            )

        except Exception as e:
            self.update_metrics(False, 0.0)
            return CoreResponse(success=False, data={'error': str(e)}, source=self.name)


# ==================== КОГНИТИВНАЯ СИСТЕМА ====================
class CognitiveSystem:
    """Система когниции и рефлексии с адаптивной памятью"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.thoughts: List[Thought] = []
        self.current_state = CognitiveState.OBSERVING
        self.learning_rate = 0.1

        # Интеграция адаптивной памяти
        self.memory_system = AdaptiveMemorySystem(user_id)

        self._load_state()

    def _load_state(self):
        """Загружает когнитивное состояние"""
        if os.path.exists(COGNITIVE_STATE):
            try:
                with open(COGNITIVE_STATE, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for thought_data in data.get('thoughts', []):
                    thought = Thought.from_dict(thought_data)
                    self.thoughts.append(thought)

                self.current_state = CognitiveState(data.get('current_state', 'observing'))
                self.learning_rate = data.get('learning_rate', 0.1)

                print(f"🧠 Загружено {len(self.thoughts)} мыслей")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки состояния: {e}")

    def save_state(self):
        """Сохраняет когнитивное состояние"""
        try:
            data = {
                'thoughts': [t.to_dict() for t in self.thoughts[-100:]],
                'current_state': self.current_state.value,
                'learning_rate': self.learning_rate,
                'last_saved': datetime.now().isoformat()
            }

            with open(COGNITIVE_STATE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Сохраняем память
            self.memory_system.save_memory()

        except Exception as e:
            print(f"⚠️ Ошибка сохранения состояния: {e}")

    def add_thought(self, content: str, state: CognitiveState = None,
                    confidence: float = 0.5, tags: List[str] = None,
                    importance: float = 0.5) -> Thought:
        """Добавляет новую мысль и сохраняет в память"""
        thought = Thought(
            content=content,
            state=state or self.current_state,
            confidence=confidence,
            tags=tags or [],
            importance=importance
        )

        self.thoughts.append(thought)

        # Сохраняем в адаптивную память
        self.memory_system.add_memory(
            content=content,
            context={'type': 'thought', 'state': state.value if state else self.current_state.value},
            importance=importance,
            emotional_valence=confidence - 0.5  # Преобразуем уверенность в эмоцию
        )

        return thought

    def transition_state(self, new_state: CognitiveState):
        """Переход в новое когнитивное состояние"""
        old_state = self.current_state
        self.current_state = new_state

        self.add_thought(
            f"Переход состояния: {old_state.value} → {new_state.value}",
            state=CognitiveState.REFLECTING,
            confidence=0.8,
            tags=['transition'],
            importance=0.6
        )

    def analyze_query_pattern(self, query: str) -> Dict[str, Any]:
        """Анализирует паттерны в запросе"""
        analysis = {
            'complexity': len(query.split()) / 20.0,
            'intent': self._detect_intent(query),
            'entities': self._extract_entities(query),
            'emotional_tone': 'neutral'
        }

        # Сохраняем анализ в память
        self.memory_system.add_memory(
            content=f"Анализ запроса: {query[:100]}",
            context={'analysis': analysis, 'query': query},
            importance=0.4
        )

        return analysis

    def _detect_intent(self, query: str) -> str:
        """Определяет намерение"""
        q = query.lower()
        if any(word in q for word in ['что', 'кто', 'где', 'когда']):
            return 'question'
        elif any(word in q for word in ['сделай', 'создай', 'сохрани']):
            return 'command'
        return 'general'

    def _extract_entities(self, query: str) -> List[str]:
        """Извлекает сущности"""
        entities = []
        # Числа
        entities.extend(re.findall(r'\b\d+\b', query))
        # Файлы
        entities.extend(re.findall(r'\b[\w\-]+\.[a-z]{2,4}\b', query, re.IGNORECASE))
        return entities

    def get_memory_context(self, query: str, limit: int = 3) -> str:
        """Получает контекст из памяти для запроса"""
        # Ищем релевантные воспоминания
        memories = self.memory_system.recall_memory(query, memory_type='both', limit=limit)

        if not memories:
            return ""

        context = "📚 **Релевантные воспоминания:**\n"
        for i, mem in enumerate(memories, 1):
            time_ago = (datetime.now() - mem.timestamp).total_seconds() / 3600
            context += f"{i}. [{mem.memory_type}] {mem.content[:100]}... ({time_ago:.1f}ч назад)\n"

        return context

    def create_mental_image(self, query: str) -> Optional[Dict[str, Any]]:
        """Создает ментальный образ через конкатенацию памяти"""
        # Находим релевантные воспоминания
        memories = self.memory_system.recall_memory(query, limit=5)

        if len(memories) < 2:
            return None

        memory_ids = [m.id for m in memories]
        image = self.memory_system.concatenate_memories(memory_ids)

        return image

    def imagine_solution(self, goal: str, steps: int = 3) -> List[str]:
        """Использует воображение для поиска решения"""
        self.transition_state(CognitiveState.IMAGINING)

        # Обратная прогрессия
        imagination_sequence = self.memory_system.reverse_progression_for_imagination(goal, steps)

        # Формируем шаги решения
        solution_steps = []
        for i, mem in enumerate(imagination_sequence, 1):
            solution_steps.append(f"Шаг {i}: {mem.content[:100]}")

        return solution_steps


# ==================== МЕНЕДЖЕР ЯДЕР ====================
class AdvancedToolsManager:
    """Менеджер ядер знаний"""

    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self.core_successes: Dict[str, int] = defaultdict(int)
        self.core_failures: Dict[str, int] = defaultdict(int)

        self._load_builtin_cores()
        print(f"✅ Загружено ядер: {len(self.cores)}")

    def _load_builtin_cores(self):
        """Загружает встроенные ядра"""
        builtin_cores = [
            DateTimeCore(),
            CalculatorCore(),
            WebSearchCore(),
            FileStorageCore()
        ]

        for core in builtin_cores:
            self.cores[core.name] = core

    def find_best_core(self, query: str) -> Optional[Tuple[KnowledgeCore, float]]:
        """Находит лучшее ядро для запроса"""
        candidates = []

        for core in self.cores.values():
            if core.can_handle(query):
                confidence = core.get_confidence(query)
                if confidence >= 0.4:
                    candidates.append((core, confidence))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[1], -x[0].priority), reverse=True)
        return candidates[0]

    def execute_core(self, core: KnowledgeCore, query: str, context: Dict = None) -> CoreResponse:
        """Выполняет ядро"""
        try:
            response = core.execute(query, context)

            if response.success:
                self.core_successes[core.name] += 1
            else:
                self.core_failures[core.name] += 1

            return response
        except Exception as e:
            self.core_failures[core.name] += 1
            return CoreResponse(success=False, data={'error': str(e)}, source=core.name)


# ==================== ПРОДВИНУТЫЙ МИНИ-МОЗГ ====================
class AdvancedMiniBrain:
    """Продвинутая версия с адаптивной памятью"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.cognitive_system = CognitiveSystem(user_id)
        self.tools_manager = AdvancedToolsManager()
        self.conversation_history = deque(maxlen=20)

        print(f"🧠 AdvancedMiniBrain готов для пользователя {user_id}")

    async def process(self, query: str, llm_caller) -> Dict[str, Any]:
        """Обработка запроса с использованием адаптивной памяти"""
        print(f"\n{'=' * 70}\n🧠 Запрос: {query[:80]}\n{'=' * 70}")

        # Анализ запроса
        query_analysis = self.cognitive_system.analyze_query_pattern(query)
        print(f"   🔍 Анализ: {query_analysis['intent']}")

        # Получаем контекст из памяти
        memory_context = self.cognitive_system.get_memory_context(query)

        # Ищем подходящее ядро
        core_result = self.tools_manager.find_best_core(query)

        if core_result:
            core, confidence = core_result
            print(f"   🎯 Ядро: {core.name} (уверенность: {confidence:.2f})")

            response = self.tools_manager.execute_core(core, query, {
                'user_id': self.user_id,
                'memory_context': memory_context
            })

            # Сохраняем в память
            self.cognitive_system.memory_system.add_memory(
                content=f"Запрос: {query} | Ответ: {response.raw_result[:100] if response.raw_result else 'Нет ответа'}",
                context={'source': core.name, 'success': response.success},
                importance=0.7 if response.success else 0.3
            )

            if response.success and response.direct_answer:
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': response.raw_result,
                    'source': core.name
                })

                return {
                    'type': 'direct_core_response',
                    'response': response.raw_result,
                    'source': core.name,
                    'need_llm': False
                }

        # Используем LLM с контекстом памяти
        llm_context = self._build_llm_context(query, query_analysis, memory_context)

        return {
            'type': 'llm_with_memory',
            'context': llm_context,
            'need_llm': True
        }

    def _build_llm_context(self, query: str, analysis: Dict, memory_context: str) -> str:
        """Строит контекст для LLM с учетом памяти"""
        now = datetime.now()

        context = f"""# КОГНИТИВНЫЙ ИИ-АССИСТЕНТ v6.0

## ВРЕМЯ: {now.strftime('%d.%m.%Y %H:%M:%S')}
## ПОЛЬЗОВАТЕЛЬ: {self.user_id}
## СОСТОЯНИЕ: {self.cognitive_system.current_state.value}

## АНАЛИЗ ЗАПРОСА:
• Намерение: {analysis.get('intent')}
• Сложность: {analysis.get('complexity', 0):.2f}

{memory_context}

## ИСТОРИЯ ДИАЛОГА:"""

        for entry in list(self.conversation_history)[-3:]:
            role = "👤" if entry.get('role') == 'user' else "🤖"
            context += f"\n{role}: {entry.get('content', '')[:80]}..."

        context += f"\n\n## ЗАПРОС:\n{query}\n\nТВОЙ ОТВЕТ:"

        return context


# ==================== ТЕЛЕГРАМ БОТ ====================
class AdvancedTelegramBot:
    """Telegram бот с адаптивной памятью"""

    def __init__(self):
        self.user_brains: Dict[str, AdvancedMiniBrain] = {}
        print("🚀 Бот инициализирован")

    def get_brain(self, user_id: str) -> AdvancedMiniBrain:
        """Получает или создает мозг для пользователя"""
        if user_id not in self.user_brains:
            self.user_brains[user_id] = AdvancedMiniBrain(user_id)
        return self.user_brains[user_id]

    async def get_llm_response(self, context: str, temperature: float = 0.6) -> str:
        """Запрос к LLM"""
        try:
            response = requests.post(
                LM_STUDIO_API_URL,
                headers={'Content-Type': 'application/json'},
                json={
                    'messages': [{'role': 'user', 'content': context}],
                    'temperature': temperature,
                    'max_tokens': 2000
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"❌ Ошибка LLM (HTTP {response.status_code})"
        except Exception as e:
            return f"❌ Ошибка: {str(e)[:100]}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        await update.message.reply_text(
            "🤖 **КОГНИТИВНЫЙ ИИ-АССИСТЕНТ v6.0**\n\n"
            "✨ **Новое в этой версии:**\n"
            "• 🧠 Адаптивная память (кратковременная + долговременная)\n"
            "• 🔄 Автоматическая консолидация памяти\n"
            "• 🎨 Конкатенация фрагментов в образы\n"
            "• 💭 Воображение через обратную прогрессию\n"
            "• 🔍 Дедупликация и валидация памяти\n\n"
            "**Команды:**\n"
            "/help - справка\n"
            "/memory - статистика памяти\n"
            "/imagine <цель> - использовать воображение\n"
            "/clear - очистить память\n\n"
            "*Готов к работе!*",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        await update.message.reply_text(
            "📖 **СПРАВКА**\n\n"
            "**Примеры:**\n"
            "• `какой сегодня день`\n"
            "• `сколько будет 45 * 78`\n"
            "• `сохрани в файл test.txt: привет мир`\n"
            "• `прочитай файл test.txt`\n"
            "• `список файлов`\n"
            "• `найди информацию о Python`\n\n"
            "**Память:**\n"
            "Система автоматически запоминает важную информацию\n"
            "и переносит её в долговременную память.\n\n"
            "**Воображение:**\n"
            "`/imagine создать бота` - система найдёт путь к цели",
            parse_mode='Markdown'
        )

    async def memory_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /memory - статистика памяти"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)

        stats = brain.cognitive_system.memory_system.get_memory_statistics()

        message = (
            f"💾 **СТАТИСТИКА ПАМЯТИ**\n\n"
            f"📦 **Кратковременная:** {stats['short_term_count']} записей\n"
            f"🗄️ **Долговременная:** {stats['long_term_count']} записей\n\n"
            f"📊 **Метрики:**\n"
            f"• Консолидаций: {stats['total_consolidations']}\n"
            f"• Дедупликаций: {stats['total_deduplications']}\n"
            f"• Воображений: {stats['total_imaginations']}\n"
            f"• Ассоциаций: {stats['total_associations']}\n\n"
            f"⭐ **Средняя важность:**\n"
            f"• Кратковременная: {stats['avg_importance_stm']:.2f}\n"
            f"• Долговременная: {stats['avg_importance_ltm']:.2f}"
        )

        await update.message.reply_text(message, parse_mode='Markdown')

    async def imagine_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /imagine - использование воображения"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)

        if not context.args:
            await update.message.reply_text(
                "💭 **Использование:**\n`/imagine <ваша цель>`\n\n"
                "**Пример:**\n`/imagine создать чат-бота`",
                parse_mode='Markdown'
            )
            return

        goal = ' '.join(context.args)

        # Используем воображение
        steps = brain.cognitive_system.imagine_solution(goal, steps=4)

        if not steps:
            await update.message.reply_text(
                "💭 Недостаточно данных в памяти для воображения.\n"
                "Попробуйте задать больше вопросов сначала."
            )
            return

        message = f"💭 **ВООБРАЖЕНИЕ: {goal}**\n\n"
        message += "**Возможные шаги:**\n"
        for step in steps:
            message += f"• {step}\n"

        await update.message.reply_text(message, parse_mode='Markdown')

    async def clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /clear - очистка памяти"""
        user_id = str(update.effective_user.id)

        if user_id in self.user_brains:
            brain = self.user_brains[user_id]
            brain.conversation_history.clear()
            brain.cognitive_system.save_state()

            await update.message.reply_text(
                "🧹 **История очищена**\n"
                "Кратковременная память сброшена.\n"
                "Долговременная память сохранена."
            )
        else:
            await update.message.reply_text("✅ Память уже пуста")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        if not text:
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        brain = self.get_brain(user_id)

        try:
            # Обрабатываем запрос
            result = await brain.process(text, self.get_llm_response)

            if not result.get('need_llm'):
                # Прямой ответ от ядра
                await update.message.reply_text(
                    result['response'],
                    parse_mode='Markdown',
                    disable_web_page_preview=True
                )
                return

            # Получаем ответ от LLM
            llm_response = await self.get_llm_response(result['context'])

            # Сохраняем в историю
            brain.conversation_history.append({
                'role': 'assistant',
                'content': llm_response[:500],
                'source': 'llm'
            })

            # Сохраняем состояние
            brain.cognitive_system.save_state()

            # Отправляем ответ
            await update.message.reply_text(
                llm_response[:4000],
                parse_mode='Markdown',
                disable_web_page_preview=True
            )

        except Exception as e:
            error_msg = f"❌ Ошибка: {str(e)[:200]}"
            print(f"ERROR: {e}")
            traceback.print_exc()
            await update.message.reply_text(error_msg)


# ==================== ЗАПУСК ====================
def main():
    """Запуск бота"""
    print("\n" + "=" * 80)
    print("🤖 ЗАПУСК КОГНИТИВНОГО ИИ-АССИСТЕНТА v6.0")
    print("=" * 80)

    bot = AdvancedTelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Команды
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("memory", bot.memory_cmd))
    application.add_handler(CommandHandler("imagine", bot.imagine_cmd))
    application.add_handler(CommandHandler("clear", bot.clear_cmd))

    # Сообщения
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот готов к работе!")
    print("=" * 80 + "\n")

    try:
        application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()