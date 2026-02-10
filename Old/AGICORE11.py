# coding: utf-8
"""
🧠 COGNITIVE_AGENT_PRO_v11_METACOGNITIVE.py
✅ Полная метакогнитивная архитектура
✅ 8-этапный когнитивный цикл
✅ 4 типа памяти с самообучением
✅ Анализ ошибок и стратегий
✅ Рост интеллекта через память
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
import aiohttp
import random
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import html
from collections import defaultdict
import statistics


# ================= ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =================
def load_dotenv(path: Path = Path(".env")):
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
            print(f"✅ Загружены переменные окружения из {path}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки .env: {e}")


load_dotenv()


# ================= ЦВЕТНЫЕ ЛОГИ =================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    ORANGE = "\033[33m"


def log_stage(stage: str, message: str, color: str = Colors.CYAN):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.BOLD}{color}[{timestamp}] {stage}{Colors.RESET}")
    print(f"{color}→ {message}{Colors.RESET}")


# ================= TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )
    from telegram.error import TelegramError, TimedOut
except ImportError as e:
    print(f"{Colors.RED}❌ Ошибка импорта telegram: {e}{Colors.RESET}")
    print(f"{Colors.YELLOW}📦 Установите: pip install python-telegram-bot>=20.7{Colors.RESET}")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_brain_v11")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "brain_memory.db"
    META_DB_PATH = ROOT / "meta_cognition.db"

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")

    TIMEOUT = 45
    MAX_TOKENS = 2000
    MAX_MESSAGE_LENGTH = 4096


# ================= ENUMS И СТРУКТУРЫ =================
class ThinkingStrategy(Enum):
    FACTUAL = "factual"  # Фактологический
    ANALYTICAL = "analytical"  # Аналитический
    STEP_BY_STEP = "step_by_step"  # Пошаговый
    EXPLORATORY = "exploratory"  # Исследовательский
    EXPLANATORY = "explanatory"  # Объяснительный


class KnowledgeBoundary(Enum):
    KNOWN = "known"  # Я знаю
    ASSUMED = "assumed"  # Я предполагаю
    UNKNOWN = "unknown"  # Я не знаю
    NEED_SEARCH = "need_search"  # Нужен поиск


class MemoryType(Enum):
    EPISODIC = "episodic"  # Опыт, диалоги
    SEMANTIC = "semantic"  # Знания, факты
    PROCEDURAL = "procedural"  # Как думать, стратегии
    META = "meta"  # О самом агенте


@dataclass
class CognitiveState:
    """Состояние когнитивного цикла"""
    query: str
    query_class: str = ""
    knowledge_boundary: KnowledgeBoundary = KnowledgeBoundary.UNKNOWN
    thinking_strategy: ThinkingStrategy = ThinkingStrategy.FACTUAL
    activated_memory: Dict[MemoryType, List] = field(default_factory=dict)
    reasoning_plan: List[str] = field(default_factory=list)
    confidence: float = 0.0
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class MetacognitiveMetrics:
    """Метрики метакогниции"""
    self_awareness_score: float = 0.0
    error_recognition_score: float = 0.0
    strategy_adaptation_score: float = 0.0
    learning_efficiency: float = 0.0
    total_reflections: int = 0
    successful_adaptations: int = 0


@dataclass
class MemoryItem:
    """Элемент памяти"""
    id: str
    content: str
    memory_type: MemoryType
    confidence: float
    usage_count: int
    last_used: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class CognitiveResponse:
    """Полный когнитивный ответ с метаданными"""
    final_answer: str
    confidence: float
    knowledge_state: str
    thinking_strategy: str
    used_memory_types: List[str]
    processing_steps: List[str]
    error_analysis: Dict[str, Any]
    cognitive_insights: List[str]
    processing_time: float
    model_used: str


# ================= КОГНИТИВНАЯ ПАМЯТЬ (4 ТИПА) =================
class CognitiveMemory:
    """Многоуровневая когнитивная память"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
        self.cache: Dict[str, MemoryItem] = {}

    def _init_db(self):
        """Инициализация таблиц памяти"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Основная таблица памяти
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                last_used REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT 0
            )
        ''')

        # Таблица связей (для ассоциативной памяти)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT,
                target_id TEXT,
                strength REAL DEFAULT 1.0,
                link_type TEXT,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memory_items(id),
                FOREIGN KEY (target_id) REFERENCES memory_items(id)
            )
        ''')

        # Индексы для быстрого поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_used ON memory_items(last_used DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON memory_items(confidence DESC)')

        conn.commit()
        conn.close()

    def store(self, item: MemoryItem):
        """Сохранить элемент памяти"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO memory_items 
            (id, content, memory_type, confidence, usage_count, last_used, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id,
            item.content[:5000],
            item.memory_type.value,
            item.confidence,
            item.usage_count,
            item.last_used,
            json.dumps(item.metadata),
            item.created_at
        ))

        conn.commit()
        conn.close()
        self.cache[item.id] = item

    def retrieve(self, query: str, memory_types: List[MemoryType] = None,
                 limit: int = 10) -> List[MemoryItem]:
        """Извлечь релевантные элементы памяти"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if memory_types:
            type_conditions = " OR ".join([f"memory_type = '{mt.value}'" for mt in memory_types])
            type_clause = f"AND ({type_conditions})"
        else:
            type_clause = ""

        # Простой поиск по ключевым словам
        keywords = re.findall(r'\b\w{3,}\b', query.lower())
        if not keywords:
            return []

        search_conditions = []
        for kw in keywords[:5]:
            search_conditions.append(f"content LIKE '%{kw}%'")

        where_clause = " OR ".join(search_conditions)

        cursor.execute(f'''
            SELECT id, content, memory_type, confidence, usage_count, 
                   last_used, metadata, created_at
            FROM memory_items
            WHERE ({where_clause}) {type_clause}
            ORDER BY confidence DESC, usage_count DESC
            LIMIT ?
        ''', (limit,))

        items = []
        for row in cursor.fetchall():
            items.append(MemoryItem(
                id=row[0],
                content=row[1],
                memory_type=MemoryType(row[2]),
                confidence=row[3],
                usage_count=row[4],
                last_used=row[5],
                metadata=json.loads(row[6]),
                created_at=row[7]
            ))

        conn.close()
        return items

    def update_usage(self, memory_id: str):
        """Обновить счетчик использования"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE memory_items 
            SET usage_count = usage_count + 1, last_used = ?
            WHERE id = ?
        ''', (time.time(), memory_id))

        conn.commit()
        conn.close()

        if memory_id in self.cache:
            self.cache[memory_id].usage_count += 1
            self.cache[memory_id].last_used = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика памяти"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                memory_type,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(usage_count) as avg_usage,
                MAX(last_used) as latest_use
            FROM memory_items
            GROUP BY memory_type
        ''')

        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                'count': row[1],
                'avg_confidence': round(row[2] or 0, 2),
                'avg_usage': round(row[3] or 0, 2),
                'latest_use': datetime.fromtimestamp(row[4]).strftime('%Y-%m-%d') if row[4] else 'never'
            }

        conn.close()
        return stats


# ================= МЕТАКОГНИТИВНЫЙ МОДУЛЬ =================
class MetacognitionEngine:
    """Двигатель метакогниции и самоанализа"""

    def __init__(self, memory: CognitiveMemory):
        self.memory = memory
        self.metrics = MetacognitiveMetrics()
        self.strategy_history: List[Tuple[ThinkingStrategy, float]] = []  # (стратегия, успешность)
        self.error_patterns: Dict[str, int] = defaultdict(int)

    def analyze_query(self, query: str) -> Tuple[str, KnowledgeBoundary, ThinkingStrategy]:
        """Анализ запроса и выбор стратегии"""
        query_lower = query.lower()

        # 1. Определение границы знаний
        knowledge_boundary = self._determine_knowledge_boundary(query_lower)

        # 2. Классификация запроса
        query_class = self._classify_query(query_lower)

        # 3. Выбор стратегии мышления
        strategy = self._select_thinking_strategy(query_class, knowledge_boundary)

        return query_class, knowledge_boundary, strategy

    def _determine_knowledge_boundary(self, query: str) -> KnowledgeBoundary:
        """Определение границы знаний по запросу"""
        # Проверяем, есть ли в памяти
        memory_items = self.memory.retrieve(query, [MemoryType.SEMANTIC, MemoryType.EPISODIC])

        if len(memory_items) >= 3:
            avg_confidence = sum([item.confidence for item in memory_items]) / len(memory_items)
            if avg_confidence > 0.7:
                return KnowledgeBoundary.KNOWN

        # Проверяем, требует ли запрос поиска
        search_triggers = [
            'курс', 'погод', 'новост', 'актуальн', 'сегодня', 'сейчас',
            'биткоин', 'доллар', 'евро', 'цена', 'сколько стоит'
        ]

        if any(trigger in query for trigger in search_triggers):
            return KnowledgeBoundary.NEED_SEARCH

        # Проверяем фактологические вопросы
        fact_patterns = [
            r'кто такой',
            r'что такое',
            r'где находится',
            r'когда был',
            r'сколько лет'
        ]

        for pattern in fact_patterns:
            if re.search(pattern, query):
                return KnowledgeBoundary.ASSUMED

        return KnowledgeBoundary.UNKNOWN

    def _classify_query(self, query: str) -> str:
        """Классификация запроса"""
        classifications = {
            'factual': ['что', 'кто', 'где', 'когда', 'сколько', 'какой'],
            'analytical': ['почему', 'зачем', 'как так', 'в чем причина'],
            'procedural': ['как сделать', 'как создать', 'шаги', 'инструкция'],
            'explanatory': ['объясни', 'расскажи', 'опиши', 'что значит'],
            'comparative': ['чем отличается', 'сравни', 'лучше или']
        }

        for class_name, triggers in classifications.items():
            for trigger in triggers:
                if trigger in query:
                    return class_name

        return 'general'

    def _select_thinking_strategy(self, query_class: str,
                                  knowledge_boundary: KnowledgeBoundary) -> ThinkingStrategy:
        """Выбор стратегии мышления на основе истории"""
        # Базовые соответствия
        strategy_map = {
            'factual': ThinkingStrategy.FACTUAL,
            'analytical': ThinkingStrategy.ANALYTICAL,
            'procedural': ThinkingStrategy.STEP_BY_STEP,
            'explanatory': ThinkingStrategy.EXPLANATORY,
            'comparative': ThinkingStrategy.ANALYTICAL,
            'general': ThinkingStrategy.EXPLORATORY
        }

        base_strategy = strategy_map.get(query_class, ThinkingStrategy.EXPLORATORY)

        # Адаптация на основе истории успеха
        if self.strategy_history:
            successful_strategies = [
                strat for strat, success in self.strategy_history[-10:]
                if success > 0.7
            ]
            if successful_strategies:
                # Выбираем наиболее успешную стратегию для этого типа запроса
                strategy_counts = defaultdict(int)
                for strat in successful_strategies:
                    strategy_counts[strat] += 1

                if strategy_counts:
                    best_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0]
                    if strategy_counts[best_strategy] >= 2:
                        return best_strategy

        return base_strategy

    def analyze_response(self, cognitive_state: CognitiveState,
                         response: str, confidence: float) -> Dict[str, Any]:
        """Анализ ответа и извлечение уроков"""
        analysis = {
            'confidence_evaluation': self._evaluate_confidence(confidence, response),
            'knowledge_gaps': self._identify_knowledge_gaps(response),
            'strategy_effectiveness': self._evaluate_strategy_effectiveness(cognitive_state, confidence),
            'lessons': [],
            'memory_updates': []
        }

        # Извлечение уроков
        if confidence < 0.6:
            lesson = f"При запросе '{cognitive_state.query[:50]}...' стратегия " \
                     f"{cognitive_state.thinking_strategy.value} показала низкую уверенность ({confidence:.0%})"
            analysis['lessons'].append(lesson)
            self.error_patterns[cognitive_state.query_class] += 1

            # Предложение альтернативной стратегии
            alt_strategy = self._suggest_alternative_strategy(cognitive_state.thinking_strategy)
            analysis['lessons'].append(f"Рекомендуется попробовать стратегию {alt_strategy.value}")

        # Обновление истории стратегий
        self.strategy_history.append((cognitive_state.thinking_strategy, confidence))

        # Обновление метрик
        self.metrics.total_reflections += 1
        if confidence > 0.7:
            self.metrics.successful_adaptations += 1

        return analysis

    def _evaluate_confidence(self, confidence: float, response: str) -> Dict[str, Any]:
        """Оценка уверенности ответа"""
        evaluation = {
            'raw_confidence': confidence,
            'adjusted_confidence': confidence,
            'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low',
            'warning_signs': []
        }

        # Признаки низкой уверенности в тексте
        low_confidence_phrases = [
            'возможно', 'может быть', 'вероятно', 'не уверен',
            'не знаю точно', 'предполагаю', 'скорее всего'
        ]

        for phrase in low_confidence_phrases:
            if phrase in response.lower():
                evaluation['warning_signs'].append(f"Использована фраза неопределенности: '{phrase}'")
                evaluation['adjusted_confidence'] *= 0.8

        # Длина ответа как индикатор
        if len(response.split()) < 20:
            evaluation['warning_signs'].append("Ответ слишком краток")
            evaluation['adjusted_confidence'] *= 0.9

        evaluation['adjusted_confidence'] = max(0.1, min(1.0, evaluation['adjusted_confidence']))

        return evaluation

    def _identify_knowledge_gaps(self, response: str) -> List[str]:
        """Выявление пробелов в знаниях"""
        gaps = []

        # Поиск маркеров незнания
        unknown_markers = [
            'не знаю', 'не могу ответить', 'нет информации',
            'не располагаю данными', 'неизвестно'
        ]

        for marker in unknown_markers:
            if marker in response.lower():
                gaps.append(f"Отсутствие знаний по теме (маркер: '{marker}')")

        return gaps

    def _evaluate_strategy_effectiveness(self, cognitive_state: CognitiveState,
                                         confidence: float) -> Dict[str, Any]:
        """Оценка эффективности стратегии"""
        effectiveness = {
            'strategy': cognitive_state.thinking_strategy.value,
            'confidence_achieved': confidence,
            'suitable_for_query_type': True,
            'recommendation': 'continue'
        }

        # Анализ соответствия стратегии типу запроса
        query_type = cognitive_state.query_class

        # Матрица соответствия стратегий типам запросов
        suitability_matrix = {
            'factual': [ThinkingStrategy.FACTUAL],
            'analytical': [ThinkingStrategy.ANALYTICAL, ThinkingStrategy.EXPLORATORY],
            'procedural': [ThinkingStrategy.STEP_BY_STEP],
            'explanatory': [ThinkingStrategy.EXPLANATORY, ThinkingStrategy.STEP_BY_STEP]
        }

        if query_type in suitability_matrix:
            if cognitive_state.thinking_strategy not in suitability_matrix[query_type]:
                effectiveness['suitable_for_query_type'] = False
                effectiveness['recommendation'] = 'change'

        # Оценка на основе истории
        if confidence < 0.5:
            effectiveness['recommendation'] = 'change'
        elif confidence > 0.8:
            effectiveness['recommendation'] = 'reinforce'

        return effectiveness

    def _suggest_alternative_strategy(self, current_strategy: ThinkingStrategy) -> ThinkingStrategy:
        """Предложение альтернативной стратегии"""
        alternatives = {
            ThinkingStrategy.FACTUAL: ThinkingStrategy.EXPLORATORY,
            ThinkingStrategy.ANALYTICAL: ThinkingStrategy.STEP_BY_STEP,
            ThinkingStrategy.STEP_BY_STEP: ThinkingStrategy.EXPLANATORY,
            ThinkingStrategy.EXPLORATORY: ThinkingStrategy.ANALYTICAL,
            ThinkingStrategy.EXPLANATORY: ThinkingStrategy.FACTUAL
        }

        return alternatives.get(current_strategy, ThinkingStrategy.EXPLORATORY)

    def get_metrics(self) -> MetacognitiveMetrics:
        """Получение текущих метрик"""
        if self.metrics.total_reflections > 0:
            self.metrics.self_awareness_score = len(self.error_patterns) / max(1, self.metrics.total_reflections)
            self.metrics.error_recognition_score = self.metrics.successful_adaptations / max(1,
                                                                                             self.metrics.total_reflections)

            if len(self.strategy_history) >= 3:
                recent_success = [s for _, s in self.strategy_history[-3:]]
                self.metrics.strategy_adaptation_score = statistics.mean(recent_success)

        return self.metrics


# ================= КОГНИТИВНОЕ ЯДРО (8-ЭТАПНЫЙ ЦИКЛ) =================
class CognitiveCore:
    """Ядро с полным метакогнитивным циклом"""

    def __init__(self, model, search_engine, memory: CognitiveMemory):
        self.model = model
        self.search_engine = search_engine
        self.memory = memory
        self.metacognition = MetacognitionEngine(memory)

        self.cycle_count = 0
        self.successful_cycles = 0

    async def execute_cognitive_cycle(self, query: str, context: str = "") -> CognitiveResponse:
        """Выполнение полного 8-этапного когнитивного цикла"""
        start_time = time.time()
        cognitive_state = CognitiveState(query=query)
        processing_steps = []
        cognitive_insights = []

        try:
            # 🔄 ЭТАП 1: ВОСПРИЯТИЕ ЗАПРОСА
            processing_steps.append("1. Восприятие запроса")
            clean_query = self._clean_query(query)
            cognitive_state.query = clean_query

            # 🔄 ЭТАП 2: КОГНИТИВНАЯ КЛАССИФИКАЦИЯ
            processing_steps.append("2. Когнитивная классификация")
            query_class, knowledge_boundary, strategy = self.metacognition.analyze_query(clean_query)
            cognitive_state.query_class = query_class
            cognitive_state.knowledge_boundary = knowledge_boundary
            cognitive_state.thinking_strategy = strategy

            cognitive_insights.append(f"Классификация: {query_class}")
            cognitive_insights.append(f"Граница знаний: {knowledge_boundary.value}")
            cognitive_insights.append(f"Стратегия: {strategy.value}")

            # 🔄 ЭТАП 3: АКТИВАЦИЯ ПАМЯТИ
            processing_steps.append("3. Активация памяти")
            activated_memory = await self._activate_memory(clean_query, cognitive_state)
            cognitive_state.activated_memory = activated_memory

            memory_summary = []
            for mem_type, items in activated_memory.items():
                if items:
                    memory_summary.append(f"{mem_type.value}: {len(items)} items")
            cognitive_insights.append(f"Активирована память: {', '.join(memory_summary)}")

            # 🔄 ЭТАП 4: ПЛАН РАССУЖДЕНИЯ
            processing_steps.append("4. План рассуждения")
            reasoning_plan = self._create_reasoning_plan(cognitive_state)
            cognitive_state.reasoning_plan = reasoning_plan

            cognitive_insights.append(f"План рассуждения: {' → '.join(reasoning_plan[:3])}")

            # 🔄 ЭТАП 5: ВЫПОЛНЕНИЕ РАССУЖДЕНИЯ
            processing_steps.append("5. Выполнение рассуждения")
            response, confidence = await self._execute_reasoning(cognitive_state, context)

            # 🔄 ЭТАП 6: ОЦЕНКА КАЧЕСТВА ОТВЕТА
            processing_steps.append("6. Оценка качества")
            error_analysis = self.metacognition.analyze_response(cognitive_state, response, confidence)
            cognitive_state.error_analysis = error_analysis
            cognitive_state.confidence = confidence

            cognitive_insights.extend(error_analysis.get('lessons', []))

            # 🔄 ЭТАП 7: ОБУЧЕНИЕ И ЗАПИСЬ В ПАМЯТЬ
            processing_steps.append("7. Обучение и запись")
            await self._learn_from_cycle(cognitive_state, response, confidence)

            # 🔄 ЭТАП 8: МЕТАКОГНИТИВНАЯ КОРРЕКТИРОВКА
            processing_steps.append("8. Метакогнитивная коррекция")
            self._metacognitive_correction(cognitive_state)

            # УСПЕШНОСТЬ ЦИКЛА
            self.cycle_count += 1
            if confidence > 0.6:
                self.successful_cycles += 1

            total_time = time.time() - start_time

            # ФОРМИРОВАНИЕ ФИНАЛЬНОГО ОТВЕТА
            final_answer = self._format_final_response(response, cognitive_state, total_time)

            used_memory_types = [mt.value for mt, items in activated_memory.items() if items]

            return CognitiveResponse(
                final_answer=final_answer,
                confidence=confidence,
                knowledge_state=knowledge_boundary.value,
                thinking_strategy=strategy.value,
                used_memory_types=used_memory_types,
                processing_steps=processing_steps,
                error_analysis=error_analysis,
                cognitive_insights=cognitive_insights,
                processing_time=total_time,
                model_used=self.model.model_name
            )

        except Exception as e:
            log_stage("❌ КОГНИТИВНЫЙ ЦИКЛ", f"Ошибка: {e}", Colors.RED)
            return self._create_error_response(str(e), start_time)

    def _clean_query(self, query: str) -> str:
        """Очистка запроса"""
        # Удаление приветствий
        greetings = ['привет', 'здравствуй', 'здравствуйте', 'hello', 'hi', 'хай']
        query_lower = query.lower()

        for greet in greetings:
            if query_lower.startswith(greet):
                query = query[len(greet):].strip()
                query = re.sub(r'^[,\s\.!?]+', '', query)
                break

        return query[:500].strip()

    async def _activate_memory(self, query: str,
                               cognitive_state: CognitiveState) -> Dict[MemoryType, List[MemoryItem]]:
        """Активация релевантных воспоминаний"""
        activated = {mt: [] for mt in MemoryType}

        # Семантическая память (знания)
        semantic_items = self.memory.retrieve(
            query,
            [MemoryType.SEMANTIC],
            limit=5
        )
        activated[MemoryType.SEMANTIC] = semantic_items

        # Эпизодическая память (опыт)
        episodic_items = self.memory.retrieve(
            query,
            [MemoryType.EPISODIC],
            limit=3
        )
        activated[MemoryType.EPISODIC] = episodic_items

        # Процедурная память (стратегии)
        if cognitive_state.query_class:
            proc_query = f"{cognitive_state.query_class} {cognitive_state.thinking_strategy.value}"
            procedural_items = self.memory.retrieve(
                proc_query,
                [MemoryType.PROCEDURAL],
                limit=2
            )
            activated[MemoryType.PROCEDURAL] = procedural_items

        # Мета-память (об агенте)
        meta_items = self.memory.retrieve(
            f"ошибка {cognitive_state.query_class}",
            [MemoryType.META],
            limit=2
        )
        activated[MemoryType.META] = meta_items

        return activated

    def _create_reasoning_plan(self, cognitive_state: CognitiveState) -> List[str]:
        """Создание плана рассуждения"""
        plan = []

        # Базовые шаги
        plan.append(f"Анализ запроса типа: {cognitive_state.query_class}")

        # В зависимости от стратегии
        if cognitive_state.thinking_strategy == ThinkingStrategy.FACTUAL:
            plan.append("Поиск точных фактов и данных")
            plan.append("Верификация источников")
            plan.append("Структурирование информации")

        elif cognitive_state.thinking_strategy == ThinkingStrategy.ANALYTICAL:
            plan.append("Выявление ключевых аспектов")
            plan.append("Анализ причинно-следственных связей")
            plan.append("Формирование выводов")

        elif cognitive_state.thinking_strategy == ThinkingStrategy.STEP_BY_STEP:
            plan.append("Разбиение на последовательные шаги")
            plan.append("Определение порядка действий")
            plan.append("Проверка логической последовательности")

        elif cognitive_state.thinking_strategy == ThinkingStrategy.EXPLORATORY:
            plan.append("Исследование различных аспектов")
            plan.append("Генерация гипотез")
            plan.append("Сравнение альтернатив")

        elif cognitive_state.thinking_strategy == ThinkingStrategy.EXPLANATORY:
            plan.append("Определение базовых понятий")
            plan.append("Построение объяснения от простого к сложному")
            plan.append("Использование аналогий и примеров")

        plan.append("Формирование итогового ответа")

        return plan

    async def _execute_reasoning(self, cognitive_state: CognitiveState,
                                 context: str) -> Tuple[str, float]:
        """Выполнение рассуждения"""
        # Подготовка системного промпта
        system_prompt = self._build_system_prompt(cognitive_state)

        # Подготовка пользовательского промпта
        user_prompt = self._build_user_prompt(cognitive_state, context)

        # Генерация ответа
        response, confidence = await self.model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self._get_temperature_for_strategy(cognitive_state.thinking_strategy),
            max_tokens=1800
        )

        return response, confidence

    def _build_system_prompt(self, cognitive_state: CognitiveState) -> str:
        """Построение системного промпта"""
        now = get_current_datetime()

        prompt_parts = [
            f"Ты — самообучающийся когнитивный агент с метакогницией.",
            f"Твоя цель — не просто ответить, а научиться думать лучше.",
            "",
            f"📅 Контекст времени: {now['full_date']}, {now['weekday']}, {now['time']}",
            "",
            f"🧠 Текущий когнитивный статус:",
            f"• Тип запроса: {cognitive_state.query_class}",
            f"• Граница знаний: {cognitive_state.knowledge_boundary.value}",
            f"• Стратегия мышления: {cognitive_state.thinking_strategy.value}",
            f"• План рассуждения: {' → '.join(cognitive_state.reasoning_plan[:3])}",
            "",
            "🔍 МЕТАКОГНИТИВНЫЕ ПРАВИЛА:",
            "1. Явно различай: 'я знаю', 'я предполагаю', 'я не знаю'",
            "2. Анализируй собственную уверенность",
            "3. Используй логические паттерны из памяти",
            "4. При недостатке знаний — запрашивай поиск или указывай границы",
            "5. Формулируй ответ как процесс мышления, а не просто информацию",
            "",
            "📚 АКТИВИРОВАННАЯ ПАМЯТЬ:"
        ]

        # Добавляем информацию из памяти
        for mem_type, items in cognitive_state.activated_memory.items():
            if items:
                prompt_parts.append(f"\n{mem_type.value.upper()}:")
                for i, item in enumerate(items[:2], 1):
                    prompt_parts.append(f"{i}. {item.content[:150]}... (уверенность: {item.confidence:.0%})")

        prompt_parts.extend([
            "",
            "💭 ИНСТРУКЦИЯ ПО ОТВЕТУ:",
            "1. Начни с осознания границы знаний",
            f"2. Используй стратегию {cognitive_state.thinking_strategy.value}",
            "3. Структурируй ответ согласно плану рассуждения",
            "4. В конце добавь метакогнитивную рефлексию:",
            "   - Насколько уверен в ответе?",
            "   - Какие аспекты требуют уточнения?",
            "   - Что узнал нового в этом процессе?"
        ])

        return "\n".join(prompt_parts)

    def _build_user_prompt(self, cognitive_state: CognitiveState, context: str) -> str:
        """Построение пользовательского промпта"""
        prompt = f"Запрос пользователя: {cognitive_state.query}\n"

        if context:
            prompt += f"\nКонтекст предыдущего диалога:\n{context}\n"

        prompt += "\nПожалуйста, выполни рассуждение согласно указанной стратегии и плану."

        # Добавляем информацию о необходимости поиска
        if cognitive_state.knowledge_boundary == KnowledgeBoundary.NEED_SEARCH:
            prompt += "\n\n⚠️ Этот запрос требует актуальной информации из интернета. " \
                      "Если данные не предоставлены, укажи это и предложи поиск."

        return prompt

    def _get_temperature_for_strategy(self, strategy: ThinkingStrategy) -> float:
        """Получение температуры для стратегии"""
        temperatures = {
            ThinkingStrategy.FACTUAL: 0.3,
            ThinkingStrategy.ANALYTICAL: 0.5,
            ThinkingStrategy.STEP_BY_STEP: 0.4,
            ThinkingStrategy.EXPLORATORY: 0.7,
            ThinkingStrategy.EXPLANATORY: 0.6
        }
        return temperatures.get(strategy, 0.5)

    async def _learn_from_cycle(self, cognitive_state: CognitiveState,
                                response: str, confidence: float):
        """Обучение на основе цикла"""
        # Создание элементов памяти
        memory_items = []

        # Эпизодическая память (опыт)
        episodic_id = hashlib.md5(f"episodic:{cognitive_state.query}:{time.time()}".encode()).hexdigest()
        episodic_item = MemoryItem(
            id=episodic_id,
            content=f"Запрос: {cognitive_state.query[:100]} | Ответ: {response[:200]}... | Уверенность: {confidence:.0%}",
            memory_type=MemoryType.EPISODIC,
            confidence=confidence,
            usage_count=1,
            last_used=time.time(),
            metadata={
                'query_class': cognitive_state.query_class,
                'strategy': cognitive_state.thinking_strategy.value,
                'timestamp': time.time()
            }
        )
        memory_items.append(episodic_item)

        # Процедурная память (если успешно)
        if confidence > 0.7:
            proc_id = hashlib.md5(
                f"procedural:{cognitive_state.query_class}:{cognitive_state.thinking_strategy.value}".encode()).hexdigest()
            procedural_item = MemoryItem(
                id=proc_id,
                content=f"Для запросов типа '{cognitive_state.query_class}' стратегия '{cognitive_state.thinking_strategy.value}' была эффективна (уверенность: {confidence:.0%})",
                memory_type=MemoryType.PROCEDURAL,
                confidence=confidence,
                usage_count=1,
                last_used=time.time(),
                metadata={
                    'query_class': cognitive_state.query_class,
                    'strategy': cognitive_state.thinking_strategy.value,
                    'success_rate': confidence
                }
            )
            memory_items.append(procedural_item)

        # Мета-память (анализ ошибок)
        if cognitive_state.error_analysis and 'lessons' in cognitive_state.error_analysis:
            for lesson in cognitive_state.error_analysis['lessons']:
                meta_id = hashlib.md5(f"meta:{lesson}".encode()).hexdigest()
                meta_item = MemoryItem(
                    id=meta_id,
                    content=lesson[:300],
                    memory_type=MemoryType.META,
                    confidence=0.8,
                    usage_count=1,
                    last_used=time.time(),
                    metadata={'lesson_type': 'error_analysis'}
                )
                memory_items.append(meta_item)

        # Сохранение в память
        for item in memory_items:
            self.memory.store(item)

    def _metacognitive_correction(self, cognitive_state: CognitiveState):
        """Метакогнитивная коррекция стратегий"""
        metrics = self.metacognition.get_metrics()

        if metrics.total_reflections > 10:
            success_rate = self.successful_cycles / self.cycle_count

            if success_rate < 0.5:
                log_stage("🔄 МЕТАКОРРЕКЦИЯ",
                          f"Низкая успешность циклов ({success_rate:.0%}). Анализ стратегий...",
                          Colors.ORANGE)

    def _format_final_response(self, response: str, cognitive_state: CognitiveState,
                               processing_time: float) -> str:
        """Форматирование финального ответа с метаданными"""
        # Базовый ответ
        final = response

        # Добавляем метакогнитивную рефлексию
        reflection = "\n\n---\n🧠 **Метакогнитивная рефлексия:**\n"

        reflection += f"• **Стратегия:** {cognitive_state.thinking_strategy.value}\n"
        reflection += f"• **Уверенность:** {cognitive_state.confidence:.0%}\n"
        reflection += f"• **Граница знаний:** {cognitive_state.knowledge_boundary.value}\n"

        if cognitive_state.error_analysis and 'warning_signs' in cognitive_state.error_analysis:
            warnings = cognitive_state.error_analysis['warning_signs']
            if warnings:
                reflection += f"• **Внимание:** {', '.join(warnings[:2])}\n"

        reflection += f"• **Время обработки:** {processing_time:.1f}с\n"
        reflection += f"• **Цикл обучения:** {self.cycle_count} (успешных: {self.successful_cycles})"

        final += reflection

        return final

    def _create_error_response(self, error: str, start_time: float) -> CognitiveResponse:
        """Создание ответа при ошибке"""
        total_time = time.time() - start_time

        error_response = (
            "❌ **Произошла ошибка в когнитивном цикле:**\n\n"
            f"Ошибка: {error[:200]}\n\n"
            "Пожалуйста, попробуйте переформулировать запрос или повторите позже."
        )

        return CognitiveResponse(
            final_answer=error_response,
            confidence=0.1,
            knowledge_state="error",
            thinking_strategy="none",
            used_memory_types=[],
            processing_steps=["Ошибка в цикле"],
            error_analysis={"error": error},
            cognitive_insights=["Ошибка прервала когнитивный цикл"],
            processing_time=total_time,
            model_used=self.model.model_name
        )

    def get_core_stats(self) -> Dict[str, Any]:
        """Статистика ядра"""
        memory_stats = self.memory.get_stats()
        meta_metrics = self.metacognition.get_metrics()

        return {
            'cycle_count': self.cycle_count,
            'successful_cycles': self.successful_cycles,
            'success_rate': self.successful_cycles / max(1, self.cycle_count),
            'memory_stats': memory_stats,
            'metacognitive_metrics': {
                'self_awareness': round(meta_metrics.self_awareness_score, 2),
                'error_recognition': round(meta_metrics.error_recognition_score, 2),
                'strategy_adaptation': round(meta_metrics.strategy_adaptation_score, 2),
                'total_reflections': meta_metrics.total_reflections
            }
        }


# ================= УТИЛИТЫ (остаются без изменений) =================
def clean_text(text: str) -> str:
    """Очистка текста от HTML и лишних символов"""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_keywords(text: str) -> List[str]:
    """Извлечение ключевых слов из текста"""
    if not text:
        return []
    stop_words = {
        'что', 'как', 'где', 'когда', 'почему', 'какой', 'какая', 'какие',
        'это', 'есть', 'был', 'была', 'были', 'быть', 'можно', 'нужно',
        'мне', 'тебе', 'его', 'ее', 'их', 'наш', 'ваш', 'в', 'на', 'и', 'или',
        'а', 'но', 'да', 'нет', 'не', 'по', 'за', 'к', 'от', 'до', 'из', 'со'
    }
    words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
    return [w for w in words if w not in stop_words][:10]


def split_message(text: str, max_length: int = 4096) -> List[str]:
    """Разделяет длинное сообщение на части"""
    if len(text) <= max_length:
        return [text]
    parts = []
    while len(text) > max_length:
        split_point = text.rfind('\n', 0, max_length)
        if split_point == -1:
            split_point = text.rfind('. ', 0, max_length)
        if split_point == -1:
            split_point = text.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length
        parts.append(text[:split_point].strip())
        text = text[split_point:].strip()
    if text:
        parts.append(text)
    return parts[:10]


def get_current_datetime() -> Dict[str, Any]:
    """Получает актуальную дату и время"""
    now = datetime.now()
    months_ru = [
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ]
    weekdays_ru = [
        'понедельник', 'вторник', 'среда', 'четверг', 'пятница',
        'суббота', 'воскресенье'
    ]
    return {
        'datetime': now,
        'full_date': f"{now.day} {months_ru[now.month - 1]} {now.year} года",
        'year': now.year,
        'month': now.month,
        'month_name': months_ru[now.month - 1],
        'day': now.day,
        'weekday': weekdays_ru[now.weekday()],
        'time': now.strftime('%H:%M'),
        'time_full': now.strftime('%H:%M:%S'),
        'iso': now.isoformat(),
        'timestamp': now.timestamp()
    }


# ================= ПОИСКОВЫЙ ДВИГАТЕЛЬ (упрощенный, из вашего кода) =================
class RealSearchEngine:
    """Поисковой движок (упрощенная версия из v10)"""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}
        self.session = None

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def search(self, query: str) -> Tuple[str, str]:
        """Основной метод поиска"""
        # Здесь должна быть ваша реализация поиска из v10
        # Для экономии места оставляю заглушку
        return f"Информация по запросу '{query}' (поиск отключен в демо-версии)", "demo"

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# ================= МОДЕЛЬ (из вашего кода) =================
class ActiveModelInterface:
    """Интерфейс для работы с моделью (упрощенный)"""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.model_name = "LM Studio Model"
        self.call_count = 0
        self.success_count = 0

    async def detect_active_model(self) -> bool:
        """Определяет какая модель сейчас загружена"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "unknown",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "temperature": 0.1,
                    "max_tokens": 10,
                    "stream": False
                }
                async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.model_name = data.get('model', 'LM Studio Chat Model')
                        return True
        except:
            pass
        return False

    async def generate(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.7, max_tokens: int = 2000) -> Tuple[str, float]:
        """Генерация ответа"""
        self.call_count += 1

        try:
            async with aiohttp.ClientSession() as session:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt[:1500]})
                messages.append({"role": "user", "content": user_prompt[:3000]})

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }

                async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=45)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip()

                        # Оценка уверенности
                        confidence = 0.7
                        if len(content) > 100:
                            confidence = 0.8
                        if len(content) < 30:
                            confidence = 0.3

                        self.success_count += 1
                        return content, confidence
                    else:
                        return f"Ошибка API: {resp.status}", 0.1
        except Exception as e:
            return f"Ошибка: {str(e)[:100]}", 0.1

    def get_stats(self) -> Dict[str, Any]:
        """Статистика модели"""
        success_rate = (self.success_count / max(1, self.call_count)) * 100
        return {
            'calls': self.call_count,
            'success': self.success_count,
            'success_rate': round(success_rate, 1),
            'model': self.model_name
        }


# ================= ТЕЛЕГРАМ БОТ (адаптированный) =================
class CognitiveTelegramBot:
    """Телеграм бот с метакогнитивным агентом"""

    def __init__(self, token: str, cognitive_core: CognitiveCore):
        self.token = token
        self.core = cognitive_core
        self.active_users = set()
        self.user_contexts: Dict[int, List[str]] = defaultdict(list)

    async def start(self):
        """Запуск бота"""
        application = (
            ApplicationBuilder()
            .token(self.token)
            .read_timeout(60)
            .write_timeout(60)
            .build()
        )

        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("stats", self._handle_stats))
        application.add_handler(CommandHandler("meta", self._handle_meta))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        await application.initialize()
        await application.start()

        log_stage("🤖 БОТ", "Metacognitive Telegram Bot запущен", Colors.GREEN)

        await application.updater.start_polling()
        return application

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /start"""
        user = update.effective_user
        self.active_users.add(user.id)

        welcome = (
            f"🧠 **Метакогнитивный Агент v11**\n\n"
            f"Привет, {user.first_name}! Я не просто бот — я самообучающаяся когнитивная система.\n\n"
            f"✨ **Мои особенности:**\n"
            f"• 8-этапный когнитивный цикл\n"
            f"• 4 типа памяти (эпизодическая, семантическая, процедурная, мета)\n"
            f"• Метакогнитивная рефлексия\n"
            f"• Самообучение на каждом диалоге\n\n"
            f"🔍 **Я осознаю свои:**\n"
            f"• Границы знаний (что знаю, предполагаю, не знаю)\n"
            f"• Эффективность стратегий мышления\n"
            f"• Собственные ошибки и паттерны\n\n"
            f"💡 Просто задай вопрос — и ты увидишь, как я думаю!"
        )

        await update.message.reply_text(welcome, parse_mode="Markdown")

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /stats"""
        stats = self.core.get_core_stats()

        stats_text = (
            f"📊 **Статистика когнитивного ядра:**\n\n"
            f"🔁 **Циклы:**\n"
            f"• Всего: {stats['cycle_count']}\n"
            f"• Успешных: {stats['successful_cycles']}\n"
            f"• Успешность: {stats['success_rate']:.0%}\n\n"
            f"🧠 **Метакогнитивные метрики:**\n"
            f"• Самосознание: {stats['metacognitive_metrics']['self_awareness']:.0%}\n"
            f"• Распознавание ошибок: {stats['metacognitive_metrics']['error_recognition']:.0%}\n"
            f"• Адаптация стратегий: {stats['metacognitive_metrics']['strategy_adaptation']:.0%}\n\n"
            f"💾 **Память:**\n"
        )

        for mem_type, mem_stats in stats['memory_stats'].items():
            stats_text += f"• {mem_type}: {mem_stats['count']} записей\n"

        await update.message.reply_text(stats_text, parse_mode="Markdown")

    async def _handle_meta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать метакогнитивную информацию"""
        meta_text = (
            f"🧠 **Метакогнитивная архитектура v11**\n\n"
            f"Я построен по принципу self-improving cognitive agent:\n\n"
            f"**8-ЭТАПНЫЙ ЦИКЛ:**\n"
            f"1. Восприятие запроса\n"
            f"2. Когнитивная классификация\n"
            f"3. Активация памяти\n"
            f"4. План рассуждения\n"
            f"5. Выполнение рассуждения\n"
            f"6. Оценка качества\n"
            f"7. Обучение и запись\n"
            f"8. Метакогнитивная коррекция\n\n"
            f"**4 ТИПА ПАМЯТИ:**\n"
            f"• Эпизодическая (опыт диалогов)\n"
            f"• Семантическая (знания и факты)\n"
            f"• Процедурная (как думать)\n"
            f"• Мета (о себе самом)\n\n"
            f"**СТРАТЕГИИ МЫШЛЕНИЯ:**\n"
            f"• Фактологическая\n"
            f"• Аналитическая\n"
            f"• Пошаговая\n"
            f"• Исследовательская\n"
            f"• Объяснительная"
        )

        await update.message.reply_text(meta_text, parse_mode="Markdown")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = update.effective_user.id
        query = update.message.text.strip()

        if not query:
            return

        await update.message.chat.send_action(action="typing")

        # Получение контекста
        context = ""
        if user_id in self.user_contexts and self.user_contexts[user_id]:
            context = "\n".join(self.user_contexts[user_id][-3:])

        # Выполнение когнитивного цикла
        log_stage("🧠 КОГНИТИВНЫЙ ЦИКЛ", f"User {user_id}: {query[:50]}...", Colors.MAGENTA)

        response = await self.core.execute_cognitive_cycle(query, context)

        # Сохранение в контекст
        context_entry = f"В: {query[:100]}\nО: {response.final_answer[:200]}..."
        self.user_contexts[user_id].append(context_entry)
        if len(self.user_contexts[user_id]) > 10:
            self.user_contexts[user_id] = self.user_contexts[user_id][-10:]

        # Отправка ответа
        parts = split_message(response.final_answer)
        for i, part in enumerate(parts):
            parse_mode = "Markdown" if i == 0 else None
            await update.message.reply_text(
                part,
                disable_web_page_preview=True,
                parse_mode=parse_mode
            )
            await asyncio.sleep(0.3)

        # Логирование метрик
        log_stage("📊 МЕТРИКИ",
                  f"Уверенность: {response.confidence:.0%} | Стратегия: {response.thinking_strategy} | "
                  f"Время: {response.processing_time:.1f}с",
                  Colors.GREEN)


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
async def main():
    """Основная функция запуска"""

    if not Config.TELEGRAM_TOKEN:
        print(f"\n{Colors.RED}❌ Telegram токен не найден!{Colors.RESET}")
        print(f"{Colors.YELLOW}Создайте файл .env с содержимым:{Colors.RESET}")
        print(f"TELEGRAM_BOT_TOKEN=ваш_токен_здесь")
        print(f"LM_STUDIO_BASE_URL=http://localhost:1234{Colors.RESET}")
        return

    now = get_current_datetime()
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'=' * 70}")
    print(f"🧠 МЕТАКОГНИТИВНЫЙ АГЕНТ v11")
    print(f"{'=' * 70}{Colors.RESET}")
    print(f"{Colors.GREEN}📅 Запуск: {now['full_date']}, {now['weekday']}{Colors.RESET}")
    print(f"{Colors.GREEN}🕐 Время: {now['time']}{Colors.RESET}\n")

    # Инициализация компонентов
    model = ActiveModelInterface(Config.LM_STUDIO_BASE_URL)

    if not await model.detect_active_model():
        print(f"{Colors.RED}❌ Не удалось подключиться к LM Studio!{Colors.RESET}")
        return

    memory = CognitiveMemory(Config.DB_PATH)
    search_engine = RealSearchEngine()

    # Создание когнитивного ядра
    cognitive_core = CognitiveCore(model, search_engine, memory)

    # Запуск бота
    bot = CognitiveTelegramBot(Config.TELEGRAM_TOKEN, cognitive_core)

    try:
        application = await bot.start()

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 70}")
        print(f"✅ МЕТАКОГНИТИВНАЯ АРХИТЕКТУРА ЗАПУЩЕНА")
        print(f"{'=' * 70}{Colors.RESET}\n")

        print(f"{Colors.CYAN}📊 Архитектура:{Colors.RESET}")
        print(f"  • 8-этапный когнитивный цикл")
        print(f"  • 4 типа памяти с самообучением")
        print(f"  • Метакогнитивный анализ ошибок")
        print(f"  • 5 стратегий мышления")
        print(f"  • Рост интеллекта через память")

        print(f"\n{Colors.YELLOW}📱 Откройте Telegram и найдите бота{Colors.RESET}")
        print(f"{Colors.RED}🛑 Для остановки нажмите Ctrl+C{Colors.RESET}\n")

        # Основной цикл
        while True:
            await asyncio.sleep(60)

            # Периодический вывод статистики
            stats = cognitive_core.get_core_stats()
            if stats['cycle_count'] % 10 == 0 and stats['cycle_count'] > 0:
                log_stage("📈 ПРОГРЕСС",
                          f"Циклов: {stats['cycle_count']} | "
                          f"Успешность: {stats['success_rate']:.0%} | "
                          f"Самосознание: {stats['metacognitive_metrics']['self_awareness']:.0%}",
                          Colors.BLUE)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Получен сигнал остановки...{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Критическая ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n{Colors.GREEN}👋 Завершение работы метакогнитивного агента{Colors.RESET}")


def run():
    """Точка входа"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}👋 Завершение работы{Colors.RESET}")


if __name__ == "__main__":
    run()