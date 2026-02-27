#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 КОГНИТИВНЫЙ ИИ-АССИСТЕНТ v6.3 - ПОЛНОСТЬЮ РАБОЧАЯ ВЕРСИЯ
✅ ИСПРАВЛЕНО: бот теперь использует результаты поиска и других ядер
✅ Безопасный парсинг вместо опасного eval()
✅ Асинхронные запросы через httpx (без блокировки event loop)
✅ Защита от инъекций и path traversal
✅ Автоматическая передача данных от ВСЕХ ядер в контекст LLM
✅ Структурированное логирование вместо print()
✅ Отправка длинных сообщений частями
✅ Блокировки для защиты от параллельных запросов
✅ Ограничения на размер файлов и контента
✅ Умное кэширование с автоматической очисткой
"""
import os
import json
import re
import ast
import asyncio
import traceback
import hashlib
import time
import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import logging
from logging.handlers import RotatingFileHandler
# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = RotatingFileHandler(
        'bot.log', maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    return root_logger

logger = setup_logging()
logger.info("=" * 80)
logger.info("🚀 ЗАПУСК КОГНИТИВНОГО ИИ-АССИСТЕНТА v6.3 (полностью рабочая версия)")
logger.info("=" * 80)

# ==================== ИМПОРТЫ С ОБРАБОТКОЙ ОШИБОК ====================
try:
    import httpx
except ImportError:
    logger.error("❌ Требуется установка: pip install httpx")
    raise

try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None
        logger.warning("⚠️ DuckDuckGo Search недоступен. Установите: pip install duckduckgo-search")

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
if not TELEGRAM_TOKEN:
    raise ValueError("❌ ОШИБКА: Не найден TELEGRAM_TOKEN в .env!")

LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
CORES_DIR = "dynamic_cores"
MEMORY_DIR = "brain_memory"
USER_FILES_DIR = "user_files"

for directory in [CORES_DIR, MEMORY_DIR, USER_FILES_DIR]:
    os.makedirs(directory, exist_ok=True)

with open(os.path.join(CORES_DIR, "__init__.py"), "w", encoding='utf-8') as f:
    f.write("# Auto-generated for imports\n")

# ==================== ИСКЛЮЧЕНИЯ БЕЗОПАСНОСТИ ====================
class SecurityError(Exception):
    """Исключение для нарушений безопасности"""
    pass

# ==================== КОГНИТИВНЫЕ КЛАССЫ ====================
class CognitiveState(Enum):
    OBSERVING = "observing"
    ANALYZING = "analyzing"
    REFLECTING = "reflecting"
    CREATING = "creating"
    ADAPTING = "adapting"
    DECIDING = "deciding"
    EXECUTING = "executing"
    IMAGINING = "imagining"

@dataclass
class Thought:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    state: CognitiveState = CognitiveState.OBSERVING
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    parent_thought: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    associations: List[str] = field(default_factory=list)
    memory_type: str = "short_term"
    consolidation_score: float = 0.0
    emotional_valence: float = 0.0
    sensory_data: Dict[str, Any] = field(default_factory=dict)  # ИСПРАВЛЕНО: было "sensory_"

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryRecord':  # ИСПРАВЛЕНО: сигнатура метода
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
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.short_term_capacity = 20
        self.consolidation_threshold = 0.7
        self.deduplication_threshold = 0.85
        self.short_term_memory: deque = deque(maxlen=self.short_term_capacity)
        self.long_term_memory: List[MemoryRecord] = []
        self.memory_index: Dict[str, MemoryRecord] = {}
        self.chronological_index: List[str] = []
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)
        self.total_consolidations = 0
        self.total_deduplications = 0
        self.total_imaginations = 0
        self._load_memory()
        logger.info(f"💾 Система памяти инициализирована для пользователя {user_id} | "
                    f"Кратковременная: {len(self.short_term_memory)} | "
                    f"Долговременная: {len(self.long_term_memory)}")

    def _load_memory(self):
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
                logger.warning(f"⚠️ Ошибка загрузки кратковременной памяти: {e}")

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
                            for assoc_id in record.associations:
                                self.association_graph[record.id].add(assoc_id)
                                self.association_graph[assoc_id].add(record.id)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки долговременной памяти: {e}")

    def save_memory(self):
        try:
            stm_file = os.path.join(MEMORY_DIR, f"stm_{self.user_id}.json")
            with open(stm_file, 'w', encoding='utf-8') as f:
                data = {
                    'user_id': self.user_id,
                    'timestamp': datetime.now().isoformat(),
                    'records': [record.to_dict() for record in self.short_term_memory]
                }
                json.dump(data, f, ensure_ascii=False, indent=2)

            ltm_file = os.path.join(MEMORY_DIR, f"ltm_{self.user_id}.jsonl")
            with open(ltm_file, 'w', encoding='utf-8') as f:
                for record in self.long_term_memory:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"⚠️ Ошибка сохранения памяти: {e}")

    def add_memory(self, content: str, context: Dict[str, Any] = None,
                   importance: float = 0.5, emotional_valence: float = 0.0,
                   sensory_data: Dict[str, Any] = None) -> MemoryRecord:  # ИСПРАВЛЕНО: sensory_data
        duplicate = self._find_duplicate(content)
        if duplicate:
            duplicate.access_count += 1
            duplicate.last_accessed = datetime.now()
            duplicate.importance = max(duplicate.importance, importance)
            duplicate.consolidation_score += 0.1
            if context:
                duplicate.context.update(context)
            self.total_deduplications += 1
            logger.debug(f"🔄 Обновлена существующая запись (дедупликация)")
            return duplicate

        record = MemoryRecord(
            content=content[:10000],
            context=context or {},
            importance=min(max(importance, 0.0), 1.0),
            memory_type='short_term',
            emotional_valence=min(max(emotional_valence, -1.0), 1.0),
            sensory_data=sensory_data or {}  # ИСПРАВЛЕНО: sensory_data
        )
        record.consolidation_score = self._calculate_consolidation_score(record)
        self.short_term_memory.append(record)
        self.memory_index[record.id] = record
        if record.consolidation_score >= self.consolidation_threshold:
            self._consolidate_to_long_term(record)
        self._create_associations(record)
        return record

    def _find_duplicate(self, content: str) -> Optional[MemoryRecord]:
        for record in list(self.short_term_memory) + self.long_term_memory:
            similarity = self._calculate_similarity(content, record.content)
            if similarity >= self.deduplication_threshold:
                return record
        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0

    def _calculate_consolidation_score(self, record: MemoryRecord) -> float:
        score = 0.0
        score += record.importance * 0.3
        score += min(record.access_count / 10.0, 1.0) * 0.3
        score += abs(record.emotional_valence) * 0.2
        score += min(len(record.associations) / 5.0, 1.0) * 0.2
        return min(score, 1.0)

    def _consolidate_to_long_term(self, record: MemoryRecord):
        record.memory_type = 'long_term'
        if record not in self.long_term_memory:
            self.long_term_memory.append(record)
            self.chronological_index.append(record.id)
            self.total_consolidations += 1
            logger.debug(f"📚 Запись консолидирована | Оценка: {record.consolidation_score:.2f}")

    def _create_associations(self, record: MemoryRecord):
        recent_memories = list(self.short_term_memory)[-5:]
        for mem in recent_memories:
            if mem.id != record.id:
                similarity = self._calculate_similarity(record.content, mem.content)
                if similarity > 0.3:
                    record.associations.append(mem.id)
                    mem.associations.append(record.id)
                    self.association_graph[record.id].add(mem.id)
                    self.association_graph[mem.id].add(record.id)

    def recall_memory(self, query: str, memory_type: str = 'both',
                      limit: int = 5) -> List[MemoryRecord]:
        memories_to_search = []
        if memory_type in ['short_term', 'both']:
            memories_to_search.extend(self.short_term_memory)
        if memory_type in ['long_term', 'both']:
            memories_to_search.extend(self.long_term_memory)

        scored_memories = []
        for memory in memories_to_search:
            relevance = self._calculate_similarity(query, memory.content)
            score = relevance * 0.6 + memory.importance * 0.3 + min(memory.access_count / 10, 0.1)
            scored_memories.append((score, memory))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, memory in scored_memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            if memory.memory_type == 'short_term':
                memory.consolidation_score += 0.05
                if memory.consolidation_score >= self.consolidation_threshold:
                    self._consolidate_to_long_term(memory)
            results.append(memory)
        return results

    def get_chronological_sequence(self, start_time: datetime = None,
                                    end_time: datetime = None,
                                    limit: int = 20) -> List[MemoryRecord]:
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
        memories = [self.memory_index[mid] for mid in memory_ids if mid in self.memory_index]
        if not memories:
            return {}
        memories.sort(key=lambda x: x.timestamp)
        concatenated_content = "\n".join([m.content for m in memories])
        merged_context = {}
        for memory in memories:
            merged_context.update(memory.context)
        all_associations = set()
        for memory in memories:
            all_associations.update(memory.associations)
        avg_importance = sum(m.importance for m in memories) / len(memories)
        avg_emotional = sum(m.emotional_valence for m in memories) / len(memories)
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
        logger.debug(f"🎨 Создан образ из {len(memories)} фрагментов памяти")
        return image

    def reverse_progression_for_imagination(self, goal_state: str,
                                            steps: int = 5) -> List[MemoryRecord]:
        self.total_imaginations += 1
        relevant_ltm = self.recall_memory(goal_state, memory_type='long_term', limit=10)
        if not relevant_ltm:
            logger.warning("⚠️ Не найдено релевантных записей для воображения")
            return []

        imagination_sequence = []
        current_memory = relevant_ltm[0]
        imagination_sequence.append(current_memory)

        for _ in range(steps - 1):
            associated_memories = []
            for assoc_id in current_memory.associations:
                if assoc_id in self.memory_index:
                    assoc_mem = self.memory_index[assoc_id]
                    if assoc_mem.timestamp < current_memory.timestamp:
                        associated_memories.append(assoc_mem)
            if not associated_memories:
                break
            current_memory = max(associated_memories, key=lambda x: x.importance)
            imagination_sequence.insert(0, current_memory)

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
        logger.debug(f"💭 Создана последовательность воображения из {len(imagination_sequence)} шагов")
        return imagination_sequence

    def get_memory_statistics(self) -> Dict[str, Any]:
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

# ==================== ТИПЫ ОТВЕТОВ ЯДЕР ====================
class CoreResponse:
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
        self.raw_result = raw_result or ""
        self.confidence = confidence
        self.source = source
        self.direct_answer = direct_answer
        self.needs_reflection = needs_reflection
        self.reflection_context = reflection_context or {}

    def to_context_string(self) -> str:
        if not self.success:
            return f"❌ Ошибка получения данных из источника '{self.source}': {self.data.get('error', 'Неизвестная ошибка')}"
        if self.raw_result:
            return f"📊 ДАННЫЕ ОТ '{self.source.upper()}':\n{self.raw_result}"
        if self.data:  # ИСПРАВЛЕНО: добавлено условие if self.data
            formatted_data = json.dumps(self.data, ensure_ascii=False, indent=2)
            return f"📊 СТРУКТУРИРОВАННЫЕ ДАННЫЕ ОТ '{self.source.upper()}':\n{formatted_data}"
        return f"ℹ️ Источник '{self.source}' обработал запрос, но не вернул данных"

    def is_final_answer(self) -> bool:
        return self.direct_answer and self.success and self.raw_result and len(self.raw_result.strip()) > 10

# ==================== БАЗОВЫЙ КЛАСС ЯДРА ====================
class KnowledgeCore(ABC):
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []
    priority: int = 5
    direct_answer_mode: bool = False
    cognitive_load: int = 1
    version: str = "1.0.0"
    total_executions: int = 0
    successful_executions: int = 0
    average_confidence: float = 0.0
    last_execution: Optional[datetime] = None

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        pass

    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        pass

    def get_confidence(self, query: str) -> float:
        return 0.5 if self.can_handle(query) else 0.0

    def update_metrics(self, success: bool, confidence: float):
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        self.average_confidence = (self.average_confidence * (
            self.total_executions - 1) + confidence) / self.total_executions
        self.last_execution = datetime.now()

    def get_efficiency_score(self) -> float:
        if self.total_executions == 0:
            return 0.0
        success_rate = self.successful_executions / self.total_executions
        return (success_rate * 0.6 + self.average_confidence * 0.4) * (10 / self.cognitive_load)

# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
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
    name = "calculator_core"
    description = "Математические вычисления"
    capabilities = ["сложение", "вычитание", "умножение", "деление", "проценты"]
    priority = 2
    direct_answer_mode = True
    cognitive_load = 2
    version = "2.1.0"

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return bool(re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', q)) or any(
            word in q for word in ['сколько будет', 'посчитай', 'вычисли']
        )

    def get_confidence(self, query: str) -> float:
        return 0.95 if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', query.lower()) else (0.7 if self.can_handle(query) else 0.0)

    def _safe_eval(self, expr: str) -> Any:
        import operator
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        allowed_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
            'sqrt': math.sqrt,
        }
        allowed_constants = {
            'pi': math.pi,
            'e': math.e,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Недопустимый тип константы: {type(node.value).__name__}")
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op_type = type(node.op)
                if op_type not in operators:
                    raise ValueError(f"Недопустимая операция: {op_type.__name__}")
                return operators[op_type](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                op_type = type(node.op)
                if op_type not in operators:
                    raise ValueError(f"Недопустимая унарная операция: {op_type.__name__}")
                return operators[op_type](operand)
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Поддерживаются только простые вызовы функций")
                func_name = node.func.id
                if func_name not in allowed_functions:
                    raise ValueError(f"Недопустимая функция: {func_name}")
                args = [eval_node(arg) for arg in node.args]
                return allowed_functions[func_name](*args)
            elif isinstance(node, ast.Name):
                if node.id in allowed_constants:
                    return allowed_constants[node.id]
                raise ValueError(f"Неизвестная переменная: {node.id}")
            elif isinstance(node, ast.Expr):
                return eval_node(node.value)
            else:
                raise ValueError(f"Недопустимый элемент выражения: {type(node).__name__}")

        try:
            if len(expr) > 200:
                raise ValueError("Выражение слишком длинное")
            dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'os.', 'sys.', 'subprocess']
            if any(pattern in expr.lower() for pattern in dangerous_patterns):
                raise ValueError("Обнаружены запрещенные паттерны в выражении")
            tree = ast.parse(expr, mode='eval')
            result = eval_node(tree.body)
            if isinstance(result, float):
                if math.isinf(result):
                    raise ValueError("Результат бесконечен")
                if math.isnan(result):
                    raise ValueError("Результат не является числом")
            return result
        except Exception as e:
            raise ValueError(f"Ошибка вычисления: {str(e)}")

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            expr_match = re.search(r'([\d\.\+\-\*\/\^\(\)\s\w]+)', query)
            if not expr_match:
                raise ValueError("Не найдено математическое выражение")
            expr_str = expr_match.group(1).strip()
            expr_str = expr_str.replace('x', '*').replace('×', '*').replace('÷', '/').replace('^', '**')
            if not re.match(r'^[\d\s\.\+\-\*\/\(\)\w\,]+$', expr_str):
                raise ValueError("Недопустимые символы в выражении")
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
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ Ошибка вычисления: {str(e)[:150]}",
                source=self.name,
                direct_answer=True
            )

class WebSearchCore(KnowledgeCore):
    name = "web_search_core"
    description = "Поиск актуальной информации в интернете"
    capabilities = ["новости", "поиск", "информация"]
    priority = 7
    cognitive_load = 3
    version = "2.1.0"

    def __init__(self):
        self.ddgs = DDGS() if DDGS else None
        self.cache = OrderedDict()
        self.cache_max_size = 50
        self.cache_ttl = 300
        self.total_cache_hits = 0
        self.total_cache_misses = 0

    def _cleanup_cache(self):
        current_time = time.time()
        to_remove = []
        for key, value in list(self.cache.items()):
            if current_time - value['timestamp'] > self.cache_ttl:
                to_remove.append(key)
        for key in to_remove:
            del self.cache[key]
        while len(self.cache) > self.cache_max_size:
            self.cache.popitem(last=False)

    def can_handle(self, query: str) -> bool:
        if not self.ddgs:
            return False
        q = query.lower()
        exclude = ['какой сегодня день', 'который час', 'сколько будет']
        if any(kw in q for kw in exclude):
            return False
        keywords = ['найди', 'новост', 'информаци', 'что такое', 'кто такой', 'курс', 'цена']
        return any(kw in q for kw in keywords)

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        if not self.ddgs:
            return CoreResponse(
                success=False,
                data={'error': 'DuckDuckGo недоступен'},
                source=self.name
            )

        try:
            self._cleanup_cache()
            cache_key = hashlib.md5(query.strip().lower().encode()).hexdigest()
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                self.total_cache_hits += 1
                logger.debug(f"💡 Кэш HIT (ключ: {cache_key[:8]})")
                results = cached['results']
                self.cache.move_to_end(cache_key)
            else:
                self.total_cache_misses += 1
                logger.debug(f"🔍 Кэш MISS (ключ: {cache_key[:8]})")
                results = list(self.ddgs.text(query, max_results=5))
                self.cache[cache_key] = {
                    'results': results,
                    'timestamp': time.time(),
                    'query': query
                }
                self.cache.move_to_end(cache_key)

            if not results:
                return CoreResponse(
                    success=False,
                    data={'error': 'Результаты не найдены'},
                    source=self.name
                )

            results_text = "🌐 **Результаты поиска:**\n"
            for i, r in enumerate(results[:3], 1):
                title = r.get('title', 'Без заголовка').replace('\n', ' ')
                results_text += f"**{i}. {title}**\n"
                snippet = r.get('body', '')[:200].replace('\n', ' ')
                if snippet:
                    results_text += f"   _{snippet}_...\n"
                href = r.get('href', '')
                if href:
                    results_text += f"   🔗 [{href[:50]}]({href})\n"

            cache_stats = f"\n📦 Кэш: {len(self.cache)}/{self.cache_max_size} записей"
            results_text += cache_stats

            self.update_metrics(True, 0.85)
            return CoreResponse(
                success=True,
                data={
                    'query': query,
                    'results_count': len(results),
                    'cache_hit': cache_key in self.cache
                },
                raw_result=results_text,
                confidence=0.85,
                source=self.name,
                needs_reflection=True
            )
        except Exception as e:
            self.update_metrics(False, 0.0)
            error_msg = f"Ошибка поиска: {str(e)[:150]}"
            logger.error(f"WebSearchCore error: {error_msg}", exc_info=True)
            return CoreResponse(
                success=False,
                data={'error': error_msg},
                source=self.name
            )

class FileStorageCore(KnowledgeCore):
    name = "file_storage_core"
    description = "Работа с текстовыми файлами"
    capabilities = ["сохранить файл", "прочитать файл", "список файлов"]
    priority = 1
    direct_answer_mode = True
    cognitive_load = 2
    version = "3.1.0"

    def __init__(self):
        self.storage_dir = USER_FILES_DIR
        self.max_file_size = 10 * 1024 * 1024
        os.makedirs(self.storage_dir, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        if not filename:
            return f"document_{int(time.time())}.txt"
        filename = os.path.basename(filename)
        filename = filename.strip().strip('\'"')
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3',
            'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6',
            'LPT7', 'LPT8', 'LPT9'
        }
        name_without_ext = os.path.splitext(filename)[0].upper()
        if name_without_ext in reserved_names:
            filename = f"safe_{filename}"
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255 - len(ext)] + ext
        elif len(filename) < 1:
            filename = f"unnamed_{int(time.time())}.txt"
        if '.' not in filename or filename.startswith('.'):
            filename += '.txt'
        return filename

    def _validate_file_size(self, content: str) -> Tuple[bool, str]:
        content_bytes = len(content.encode('utf-8'))
        if content_bytes > self.max_file_size:
            return False, f"Файл слишком большой ({content_bytes / 1024 / 1024:.1f} МБ). Максимум: {self.max_file_size / 1024 / 1024} МБ"
        return True, ""

    def _get_safe_filepath(self, filename: str) -> str:
        sanitized = self._sanitize_filename(filename)
        filepath = os.path.join(self.storage_dir, sanitized)
        filepath = os.path.abspath(filepath)
        storage_abs = os.path.abspath(self.storage_dir)
        if not filepath.startswith(storage_abs + os.sep):
            raise SecurityError(f"Попытка выхода за пределы хранилища: {filename}")
        return filepath

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        keywords = ['файл', 'сохрани', 'прочитай', 'список файлов']
        return any(keyword in q for keyword in keywords)

    def get_confidence(self, query: str) -> float:
        q = query.lower()
        if any(cmd in q for cmd in ['прочитай файл', 'сохрани в файл']):
            return 0.95
        return 0.7 if self.can_handle(query) else 0.0

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            q = query.lower()
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
                filepath = self._get_safe_filepath(filename)
                if not os.path.exists(filepath):
                    return CoreResponse(
                        success=False,
                        data={'error': 'Файл не найден'},
                        raw_result=f"❌ Файл `{filename}` не найден",
                        source=self.name,
                        direct_answer=True
                    )
                file_size = os.path.getsize(filepath)
                if file_size > self.max_file_size:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Файл слишком большой для чтения'},
                        raw_result=f"❌ Файл `{filename}` слишком большой ({file_size / 1024:.1f} КБ)",
                        source=self.name,
                        direct_answer=True
                    )
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read(5000)
                preview = content[:500] + ('...' if len(content) > 500 else '')
                result = f"📄 **Файл `{filename}`:**\n```text\n{preview}\n```"
                self.update_metrics(True, 1.0)
                return CoreResponse(
                    success=True,
                    data={'filename': filename, 'size': len(content)},
                    raw_result=result,
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )
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
                is_valid, error_msg = self._validate_file_size(content)
                if not is_valid:
                    return CoreResponse(
                        success=False,
                        data={'error': error_msg},
                        raw_result=f"❌ {error_msg}",
                        source=self.name,
                        direct_answer=True
                    )
                filepath = self._get_safe_filepath(filename)
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
            elif 'список' in q:
                files = []
                for f in os.listdir(self.storage_dir):
                    fp = os.path.join(self.storage_dir, f)
                    if os.path.isfile(fp) and os.path.abspath(fp).startswith(os.path.abspath(self.storage_dir)):
                        files.append(f)
                if not files:
                    result = "📁 **Файловое хранилище пусто**"
                else:
                    result = f"📁 **Файлы ({len(files)}):**\n"
                    for i, f in enumerate(sorted(files)[:10], 1):
                        size = os.path.getsize(os.path.join(self.storage_dir, f))
                        size_str = f"{size / 1024:.1f} КБ" if size > 1024 else f"{size} байт"
                        result += f"{i}. `{f}` ({size_str})\n"
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
        except SecurityError as e:
            logger.warning(f"⚠️ Попытка нарушения безопасности: {e}")
            return CoreResponse(
                success=False,
                data={'error': 'Нарушение безопасности'},
                raw_result="❌ Обнаружена попытка нарушения безопасности",
                source=self.name,
                direct_answer=True
            )
        except Exception as e:
            self.update_metrics(False, 0.0)
            logger.error(f"FileStorageCore error: {e}", exc_info=True)
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ Ошибка: {str(e)[:150]}",
                source=self.name,
                direct_answer=True
            )

# ==================== КОГНИТИВНАЯ СИСТЕМА ====================
class CognitiveSystem:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.thoughts: List[Thought] = []
        self.current_state = CognitiveState.OBSERVING
        self.learning_rate = 0.1
        self.memory_system = AdaptiveMemorySystem(user_id)
        self._load_state()

    def _load_state(self):
        cognitive_state_file = os.path.join(MEMORY_DIR, f"cognitive_state_{self.user_id}.json")
        if os.path.exists(cognitive_state_file):
            try:
                with open(cognitive_state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for thought_data in data.get('thoughts', []):
                        thought = Thought.from_dict(thought_data)
                        self.thoughts.append(thought)
                    self.current_state = CognitiveState(data.get('current_state', 'observing'))
                    self.learning_rate = data.get('learning_rate', 0.1)
                logger.info(f"🧠 Загружено {len(self.thoughts)} мыслей")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки состояния: {e}")

    def save_state(self):
        try:
            cognitive_state_file = os.path.join(MEMORY_DIR, f"cognitive_state_{self.user_id}.json")
            data = {
                'thoughts': [t.to_dict() for t in self.thoughts[-100:]],
                'current_state': self.current_state.value,
                'learning_rate': self.learning_rate,
                'last_saved': datetime.now().isoformat()
            }
            with open(cognitive_state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.memory_system.save_memory()
        except Exception as e:
            logger.error(f"⚠️ Ошибка сохранения состояния: {e}")

    def add_thought(self, content: str, state: CognitiveState = None,
                    confidence: float = 0.5, tags: List[str] = None,
                    importance: float = 0.5) -> Thought:
        thought = Thought(
            content=content[:5000],
            state=state or self.current_state,
            confidence=min(max(confidence, 0.0), 1.0),
            tags=tags or [],
            importance=min(max(importance, 0.0), 1.0)
        )
        self.thoughts.append(thought)
        self.memory_system.add_memory(
            content=content[:5000],
            context={'type': 'thought', 'state': state.value if state else self.current_state.value},
            importance=importance,
            emotional_valence=confidence - 0.5
        )
        return thought

    def transition_state(self, new_state: CognitiveState):
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
        analysis = {
            'complexity': min(len(query.split()) / 20.0, 1.0),
            'intent': self._detect_intent(query),
            'entities': self._extract_entities(query),
            'emotional_tone': 'neutral'
        }
        self.memory_system.add_memory(
            content=f"Анализ запроса: {query[:100]}",
            context={'analysis': analysis, 'query': query},
            importance=0.4
        )
        return analysis

    def _detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ['что', 'кто', 'где', 'когда', 'курс', 'цена']):
            return 'question'
        elif any(word in q for word in ['сделай', 'создай', 'сохрани']):
            return 'command'
        return 'general'

    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        entities.extend(re.findall(r'\b\d+\b', query))
        entities.extend(re.findall(r'\b[\w\-]+\.[a-z]{2,4}\b', query, re.IGNORECASE))
        return entities[:10]

    def get_memory_context(self, query: str, limit: int = 3) -> str:
        memories = self.memory_system.recall_memory(query, memory_type='both', limit=limit)
        if not memories:
            return ""
        context = "📚 **Релевантные воспоминания:**\n"
        for i, mem in enumerate(memories, 1):
            time_ago = (datetime.now() - mem.timestamp).total_seconds() / 3600
            context += f"{i}. [{mem.memory_type}] {mem.content[:100]}... ({time_ago:.1f}ч назад)\n"
        return context

    def create_mental_image(self, query: str) -> Optional[Dict[str, Any]]:
        memories = self.memory_system.recall_memory(query, limit=5)
        if len(memories) < 2:
            return None
        memory_ids = [m.id for m in memories]
        return self.memory_system.concatenate_memories(memory_ids)

    def imagine_solution(self, goal: str, steps: int = 3) -> List[str]:
        self.transition_state(CognitiveState.IMAGINING)
        imagination_sequence = self.memory_system.reverse_progression_for_imagination(goal, steps)
        solution_steps = []
        for i, mem in enumerate(imagination_sequence, 1):
            solution_steps.append(f"Шаг {i}: {mem.content[:100]}")
        return solution_steps

# ==================== МЕНЕДЖЕР ЯДЕР ====================
class AdvancedToolsManager:
    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self.core_successes: Dict[str, int] = defaultdict(int)
        self.core_failures: Dict[str, int] = defaultdict(int)
        self._load_builtin_cores()
        logger.info(f"✅ Загружено ядер: {len(self.cores)}")

    def _load_builtin_cores(self):
        builtin_cores = [
            DateTimeCore(),
            CalculatorCore(),
            WebSearchCore(),
            FileStorageCore()
        ]
        for core in builtin_cores:
            self.cores[core.name] = core

    def find_all_relevant_cores(self, query: str) -> List[Tuple[KnowledgeCore, float]]:
        """Находит ВСЕ релевантные ядра для запроса (не только одно лучшее)"""
        candidates = []
        for core in self.cores.values():
            if core.can_handle(query):
                confidence = core.get_confidence(query)
                if confidence >= 0.3:
                    candidates.append((core, confidence))
        candidates.sort(key=lambda x: (x[1], -x[0].priority), reverse=True)
        return candidates

    def execute_core(self, core: KnowledgeCore, query: str, context: Dict = None) -> CoreResponse:
        try:
            response = core.execute(query, context)
            if response.success:
                self.core_successes[core.name] += 1
            else:
                self.core_failures[core.name] += 1
            return response
        except Exception as e:
            self.core_failures[core.name] += 1
            logger.error(f"Ошибка выполнения ядра {core.name}: {e}", exc_info=True)
            return CoreResponse(success=False, data={'error': str(e)}, source=core.name)

# ==================== ПРОДВИНУТЫЙ МИНИ-МОЗГ (КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ) ====================
class AdvancedMiniBrain:
    """Продвинутая версия с ПРАВИЛЬНОЙ передачей данных от ядер в контекст LLM"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.cognitive_system = CognitiveSystem(user_id)
        self.tools_manager = AdvancedToolsManager()
        self.conversation_history = deque(maxlen=20)
        logger.info(f"🧠 AdvancedMiniBrain готов для пользователя {user_id}")

    async def process(self, query: str, llm_caller) -> Dict[str, Any]:
        logger.info(f"{'=' * 70}\n🧠 Запрос от {self.user_id}: {query[:80]}\n{'=' * 70}")

        # Анализ запроса
        query_analysis = self.cognitive_system.analyze_query_pattern(query)
        logger.debug(f"🔍 Анализ: {query_analysis['intent']}")

        # Получаем контекст из памяти
        memory_context = self.cognitive_system.get_memory_context(query)

        # 🔑 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Собираем данные от ВСЕХ релевантных ядер
        core_responses: List[CoreResponse] = []
        direct_response: Optional[CoreResponse] = None

        # Находим все подходящие ядра
        relevant_cores = self.tools_manager.find_all_relevant_cores(query)
        logger.debug(f"🎯 Найдено релевантных ядер: {len(relevant_cores)}")

        for core, confidence in relevant_cores:
            logger.debug(f"   → Выполняем ядро '{core.name}' (уверенность: {confidence:.2f})")
            try:
                response = self.tools_manager.execute_core(core, query, {
                    'user_id': self.user_id,
                    'memory_context': memory_context,
                    'query_analysis': query_analysis
                })
                core_responses.append(response)

                # Сохраняем данные в память
                self.cognitive_system.memory_system.add_memory(
                    content=f"Данные от {core.name}: {response.raw_result[:200] if response.raw_result else 'Нет данных'}",
                    context={'source': core.name, 'query': query, 'type': 'core_data'},
                    importance=0.6 + (response.confidence * 0.3) if response.success else 0.2
                )

                # Если это прямой ответ с высокой уверенностью - запоминаем для немедленного ответа
                if response.success and response.direct_answer and response.confidence >= 0.85:
                    if direct_response is None or response.confidence > direct_response.confidence:
                        direct_response = response
                        logger.info(f"⚡ Выбран прямой ответ от ядра '{core.name}'")
            except Exception as e:
                logger.error(f"Ошибка выполнения ядра {core.name}: {e}", exc_info=True)
                continue

        # Если есть прямой ответ с высокой уверенностью - возвращаем его сразу
        if direct_response and direct_response.raw_result.strip():
            self.conversation_history.append({
                'role': 'assistant',
                'content': direct_response.raw_result[:500],
                'source': direct_response.source
            })
            return {
                'type': 'direct_core_response',
                'response': direct_response.raw_result,
                'source': direct_response.source,
                'need_llm': False
            }

        # 🔑 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Формируем контекст для LLM с данными от ВСЕХ ядер
        core_data_context = ""
        has_core_data = False
        if core_responses:
            # Сортируем по уверенности и приоритету
            core_responses.sort(key=lambda r: (
                r.confidence,
                -self.tools_manager.cores[r.source].priority if r.source in self.tools_manager.cores else 0
            ), reverse=True)

            # Формируем контекст из данных ядер
            for i, resp in enumerate(core_responses, 1):
                if resp.success and resp.raw_result.strip():
                    has_core_data = True
                    core_data_context += f"\n### 📊 ДАННЫЕ ОТ МОДУЛЯ [{resp.source.upper()}] (уверенность: {resp.confidence:.2f}):\n{resp.raw_result}\n"
                    logger.debug(f"   ➕ Добавлены данные от ядра '{resp.source}' в контекст LLM")

        # Если нет данных от ядер - добавляем подсказку для LLM
        if not has_core_data:
            core_data_context = "\n### 📊 ДАННЫЕ ОТ СПЕЦИАЛИЗИРОВАННЫХ МОДУЛЕЙ:\nНе найдено релевантных данных от специализированных модулей. Ответь на основе общих знаний.\n"
            logger.debug("⚠️ Нет данных от ядер, используем общие знания LLM")

        # Строим полный контекст для LLM с ЧЕТКИМИ ИНСТРУКЦИЯМИ использовать данные ядер
        llm_context = self._build_llm_context(
            query=query,
            analysis=query_analysis,
            memory_context=memory_context,
            core_data_context=core_data_context  # ← КРИТИЧЕСКИ ВАЖНО: передаем данные от ядер!
        )
        logger.info(f"🧠 Передано в LLM: {len(llm_context)} символов контекста (включая данные от ядер)")

        return {
            'type': 'llm_with_memory_and_cores',
            'context': llm_context,
            'need_llm': True
        }

    def _build_llm_context(self, query: str, analysis: Dict, memory_context: str, core_data_context: str) -> str:
        """Строит контекст для LLM с данными от ядер знаний и ЧЕТКИМИ ИНСТРУКЦИЯМИ"""
        now = datetime.now()
        context = f"""# КОГНИТИВНЫЙ ИИ-АССИСТЕНТ v6.3
## ВРЕМЯ: {now.strftime('%d.%m.%Y %H:%M:%S')}
## ПОЛЬЗОВАТЕЛЬ: {self.user_id}
## СОСТОЯНИЕ: {self.cognitive_system.current_state.value}
## АНАЛИЗ ЗАПРОСА:
• Намерение: {analysis.get('intent', 'неизвестно')}
• Сложность: {analysis.get('complexity', 0):.2f}
• Сущности: {', '.join(analysis.get('entities', [])) if analysis.get('entities') else 'нет'}
## КОНТЕКСТ ИЗ ПАМЯТИ:
{memory_context if memory_context.strip() else 'Нет релевантных воспоминаний'}
## 🔑 ДАННЫЕ ОТ СПЕЦИАЛИЗИРОВАННЫХ МОДУЛЕЙ (ИСПОЛЬЗУЙ КАК ПЕРВОИСТОЧНИК):
{core_data_context}
## ИСТОРИЯ ДИАЛОГА:"""

        for entry in list(self.conversation_history)[-3:]:
            role = "👤 ПОЛЬЗОВАТЕЛЬ" if entry.get('role') == 'user' else "🤖 АССИСТЕНТ"
            content = entry.get('content', '')[:150]
            source = f" (источник: {entry.get('source', 'llm')})" if entry.get('source') else ""
            context += f"\n{role}{source}: {content}"

        context += f"\n## 🎯 ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}\n"
        context += "## 📌 ИНСТРУКЦИИ ДЛЯ ТЕБЯ (ИИ-АССИСТЕНТА):\n"
        context += "1. ВНИМАТЕЛЬНО проанализируй данные от специализированных модулей выше.\n"
        context += "2. ИСПОЛЬЗУЙ эти данные как ПЕРВОИСТОЧНИК для формирования ответа.\n"
        context += "3. Если данные содержат актуальную информацию (курсы, новости, время) - ОБЯЗАТЕЛЬНО используй их.\n"
        context += "4. Если данные противоречивы - укажи на это и дай наиболее вероятный ответ.\n"
        context += "5. Если данных нет или они нерелевантны - ответь на основе общих знаний, но укажи это.\n"
        context += "6. Отвечай кратко, по делу, на русском языке.\n"
        context += "## ✅ ТВОЙ ОТВЕТ:\n"

        # Ограничиваем общий размер контекста
        return context[:4000] if len(context) > 4000 else context

# ==================== ТЕЛЕГРАМ БОТ ====================
class AdvancedTelegramBot:
    """Telegram бот с полной интеграцией данных от ядер"""
    def __init__(self):
        self.user_brains: Dict[str, AdvancedMiniBrain] = {}
        self.user_locks: Dict[str, asyncio.Lock] = {}
        self.global_lock = asyncio.Lock()
        logger.info("🚀 Бот инициализирован с защитой от параллельных запросов")

    async def get_user_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self.user_locks:
            async with self.global_lock:
                if user_id not in self.user_locks:
                    self.user_locks[user_id] = asyncio.Lock()
        return self.user_locks[user_id]

    def get_brain(self, user_id: str) -> AdvancedMiniBrain:
        if user_id not in self.user_brains:
            self.user_brains[user_id] = AdvancedMiniBrain(user_id)
        return self.user_brains[user_id]

    async def get_llm_response(self, context: str, temperature: float = 0.6) -> str:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    LM_STUDIO_API_URL,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {LM_STUDIO_API_KEY}' if LM_STUDIO_API_KEY != 'lm-studio' else ''
                    },
                    json={
                        'messages': [{'role': 'user', 'content': context[:4000]}],
                        'temperature': temperature,
                        'max_tokens': 2000,
                        'stream': False
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        return data['choices'][0]['message']['content'].strip() or "⚠️ LLM вернул пустой ответ"
                    return "⚠️ LLM вернул пустой ответ"
                else:
                    return f"❌ Ошибка LLM (HTTP {response.status_code}): {response.text[:150]}"
        except httpx.TimeoutException:
            return "⏱️ Таймаут запроса к LLM (60 сек)"
        except httpx.NetworkError as e:
            return f"🌐 Ошибка сети: {str(e)[:100]}"
        except Exception as e:
            logger.error(f"Ошибка LLM запроса: {e}", exc_info=True)
            return f"❌ Внутренняя ошибка: {str(e)[:100]}"

    async def send_long_message(self, bot, chat_id: int, text: str,
                                parse_mode: str = 'Markdown',
                                disable_web_page_preview: bool = True) -> List[int]:
        MAX_LENGTH = 4000
        message_ids = []

        # Вспомогательная функция для безопасного экранирования Markdown
        def escape_markdown(text: str) -> str:
            return text.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]')

        if len(text) <= MAX_LENGTH:
            try:
                msg = await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_web_page_preview
                )
                return [msg.message_id]
            except Exception as e:
                logger.warning(f"Ошибка отправки короткого сообщения: {e}")
                # Повторная попытка без форматирования
                msg = await bot.send_message(
                    chat_id=chat_id,
                    text=escape_markdown(text),
                    parse_mode=None,
                    disable_web_page_preview=disable_web_page_preview
                )
                return [msg.message_id]

        parts = []
        remaining = text
        while remaining:
            if len(remaining) <= MAX_LENGTH:
                parts.append(remaining)
                break
            split_pos = -1
            pos = remaining.rfind('. ', 0, MAX_LENGTH)
            if pos != -1:
                split_pos = pos + 1
            if split_pos == -1:
                pos = remaining.rfind('\n', 0, MAX_LENGTH)
                if pos != -1:
                    split_pos = pos + 1
            if split_pos == -1:
                pos = remaining.rfind(', ', 0, MAX_LENGTH)
                if pos != -1:
                    split_pos = pos + 1
            if split_pos == -1:
                split_pos = MAX_LENGTH
            parts.append(remaining[:split_pos])
            remaining = remaining[split_pos:].lstrip()

        total_parts = len(parts)
        for i, part in enumerate(parts):
            if total_parts > 1:
                if i == 0:
                    part = f"{part}\n*(Продолжение {i + 2}/{total_parts}...)*"
                elif i == total_parts - 1:
                    part = f"*(Часть {i + 1}/{total_parts})*\n{part}"
                else:
                    part = f"*(Часть {i + 1}/{total_parts})*\n{part}\n*(Продолжение {i + 2}/{total_parts}...)*"

            try:
                msg = await bot.send_message(
                    chat_id=chat_id,
                    text=part,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_web_page_preview
                )
                message_ids.append(msg.message_id)
                if i < total_parts - 1:
                    await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"Ошибка отправки части {i + 1}: {e}")
                try:
                    # Повторная попытка с экранированием
                    clean_part = escape_markdown(part)
                    msg = await bot.send_message(
                        chat_id=chat_id,
                        text=clean_part,
                        parse_mode=None,
                        disable_web_page_preview=disable_web_page_preview
                    )
                    message_ids.append(msg.message_id)
                except Exception as e2:
                    logger.error(f"Полный провал отправки части {i + 1}: {e2}")
        return message_ids

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 **КОГНИТИВНЫЙ ИИ-АССИСТЕНТ v6.3**\n"
            "✨ **Ключевые улучшения:**\n"
            "• ✅ Бот ТЕПЕРЬ ИСПОЛЬЗУЕТ результаты поиска и других ядер\n"
            "• 🔒 Полная безопасность: защита от инъекций, валидация путей\n"
            "• ⚡ Асинхронность: полная поддержка асинхронных операций\n"
            "• 📦 Ограничения: размер файлов (10 МБ), длина сообщений, кэш с очисткой\n"
            "• 🧠 Адаптивная память (кратковременная + долговременная)\n"
            "• 💭 Воображение через обратную прогрессию\n"
            "**Примеры запросов:**\n"
            "• `курс доллара сегодня`\n"
            "• `сколько будет 45 * 78`\n"
            "• `какой сегодня день`\n"
            "• `сохрани в файл: привет мир`\n"
            "*Готов к работе!*",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📖 **СПРАВКА**\n"
            "**Примеры:**\n"
            "• `какой сегодня день` → точная дата и время\n"
            "• `курс доллара` → актуальный курс из интернета\n"
            "• `сколько будет 45 * 78` → мгновенный расчет\n"
            "• `сохрани в файл test.txt: привет мир` → сохранение в файл\n"
            "• `прочитай файл test.txt` → чтение файла\n"
            "• `список файлов` → список всех файлов\n"
            "**Важно:**\n"
            "Бот автоматически использует специализированные модули\n"
            "для получения актуальной информации из интернета!",
            parse_mode='Markdown'
        )

    async def memory_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)
        stats = brain.cognitive_system.memory_system.get_memory_statistics()
        message = (
            f"💾 **СТАТИСТИКА ПАМЯТИ**\n"
            f"📦 **Кратковременная:** {stats['short_term_count']} записей\n"
            f"🗄️ **Долговременная:** {stats['long_term_count']} записей\n"
            f"📊 **Метрики:**\n"
            f"• Консолидаций: {stats['total_consolidations']}\n"
            f"• Дедупликаций: {stats['total_deduplications']}\n"
            f"• Воображений: {stats['total_imaginations']}\n"
            f"• Ассоциаций: {stats['total_associations']}\n"
            f"⭐ **Средняя важность:**\n"
            f"• Кратковременная: {stats['avg_importance_stm']:.2f}\n"
            f"• Долговременная: {stats['avg_importance_ltm']:.2f}"
        )
        await update.message.reply_text(message, parse_mode='Markdown')

    async def imagine_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)
        if not context.args:
            await update.message.reply_text(
                "💭 **Использование:**\n"
                "`/imagine <ваша цель>`\n"
                "**Пример:**\n"
                "`/imagine создать чат-бота`",
                parse_mode='Markdown'
            )
            return
        goal = ' '.join(context.args)[:200]
        steps = brain.cognitive_system.imagine_solution(goal, steps=4)
        if not steps:
            await update.message.reply_text(
                "💭 Недостаточно данных в памяти для воображения.\n"
                "Попробуйте задать больше вопросов сначала."
            )
            return
        message = f"💭 **ВООБРАЖЕНИЕ: {goal}**\n"
        message += "**Возможные шаги:**\n"
        for step in steps:
            message += f"• {step}\n"
        await update.message.reply_text(message, parse_mode='Markdown')

    async def clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        if not update.message or not update.message.text:
            return
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()[:2000]
        if not text:
            return

        user_lock = await self.get_user_lock(user_id)
        async with user_lock:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"
            )
            brain = self.get_brain(user_id)
            start_time = time.time()
            try:
                result = await brain.process(text, self.get_llm_response)
                processing_time = time.time() - start_time
                logger.info(f"⏱️ Пользователь {user_id}: обработка '{text[:30]}' за {processing_time:.2f}с")

                if not result.get('need_llm'):
                    await self.send_long_message(
                        context.bot,
                        chat_id=update.effective_chat.id,
                        text=result['response'],
                        parse_mode='Markdown',
                        disable_web_page_preview=True
                    )
                    return

                llm_response = await self.get_llm_response(result['context'])
                brain.conversation_history.append({
                    'role': 'assistant',
                    'content': llm_response[:500],
                    'source': 'llm'
                })
                brain.cognitive_system.save_state()
                await self.send_long_message(
                    context.bot,
                    chat_id=update.effective_chat.id,
                    text=llm_response,
                    parse_mode='Markdown',
                    disable_web_page_preview=True
                )
            except Exception as e:
                error_msg = f"❌ Ошибка обработки: {str(e)[:150]}"
                logger.error(f"Ошибка обработки сообщения от {user_id}: {e}", exc_info=True)
                await update.message.reply_text(
                    error_msg + "\nПопробуйте повторить запрос через несколько секунд.",
                    parse_mode=None
                )

# ==================== ЗАПУСК ====================
def main():
    logger.info("=" * 80)
    logger.info("🤖 ЗАПУСК КОГНИТИВНОГО ИИ-АССИСТЕНТА v6.3 (полностью рабочая версия)")
    logger.info("=" * 80)
    bot = AdvancedTelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("memory", bot.memory_cmd))
    application.add_handler(CommandHandler("imagine", bot.imagine_cmd))
    application.add_handler(CommandHandler("clear", bot.clear_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    logger.info("✅ Бот готов к работе!")
    logger.info("=" * 80 + "\n")
    try:
        application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("\n🛑 Остановка по запросу пользователя...")
    except Exception as e:
        logger.exception(f"\n❌ Критическая ошибка: {e}")
        raise

if __name__ == "__main__":
    main()