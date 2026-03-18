#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ENHANCED AGI BRAIN v33.0 — ЧЕСТНАЯ АРХИТЕКТУРА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ v33.0:
✅ Честные метрики на основе реальных данных (не симуляция)
✅ Проверка консистентности (система не противоречит себе)
✅ Упрощенная система весов вместо фейковой нейросети
✅ Активное целеполагание на основе диалога
✅ Извлечение и отслеживание интересов пользователя
✅ Улучшенная валидация ответов (запрет выдуманных чисел)
✅ Детальное логирование для отладки противоречий

🎯 ПРИНЦИПЫ v33.0:
1. Честность > Имитация
2. Проверяемость > Сложность
3. Консистентность > Креативность
4. Объяснимость > Загадочность
"""

import os
import sys
import json
import re
import asyncio
import aiohttp
import traceback
import random
import math
import hashlib
import logging
import numpy as np
from pathlib import Path
from collections import Counter, deque, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import wraps, lru_cache
import time
import gzip
import pickle
from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)

load_dotenv()


@dataclass
class EnhancedConfig:
    """Конфигурация с упором на честность v33.0"""
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🔧 НОВОЕ: Режим честной метрики (всегда включен)
    honest_metrics: bool = True
    validate_responses: bool = True  # Проверка ответов на выдуманные числа
    check_consistency: bool = True  # Проверка на противоречия

    # Память
    working_memory_size: int = 7
    short_term_size: int = 100
    short_term_decay: float = 0.95
    long_term_size: int = 10000
    consolidation_threshold: float = 0.7
    consolidation_quality_threshold: float = 0.8
    memory_priority_boost: float = 0.2
    episodic_context_window: int = 5

    # Метакогниция
    curiosity_threshold: float = 0.55
    question_cooldown: int = 90
    max_autonomous_questions: int = 5
    reasoning_depth: int = 3
    analogy_threshold: float = 0.6
    confidence_calibration_samples: int = 50

    # Эмоции
    emotion_tracking: bool = True
    empathy_weight: float = 0.3

    # Кэширование
    embedding_dim: int = 512
    semantic_cache_size: int = 2000

    # Интервалы автономной активности
    spontaneous_thought_interval: int = 150
    reflection_interval: int = 600
    consolidation_interval: int = 1200
    save_interval: int = 240
    goal_update_interval: int = 300  # 🔧 НОВОЕ: обновление целей

    # 🔧 НОВОЕ: Параметры для честных метрик
    max_input_words_for_perception: int = 50
    max_reasoning_steps_for_metric: int = 5
    max_memory_items_for_metric: int = 10
    max_response_words_for_action: int = 100

    base_dir: Path = Path(os.getenv('BASE_DIR', 'temporal_brain_v33'))

    def __post_init__(self):
        for subdir in ['memory', 'knowledge', 'cache', 'logs',
                       'backups', 'episodic', 'analytics', 'goals']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)


CONFIG = EnhancedConfig()


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('Enhanced_AGI_v33')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'agi_v33_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


@dataclass
class MessageAttribution:
    """Отслеживание источника каждого сообщения"""
    role: str
    content: str
    timestamp: float
    contains_metrics: bool = False
    metric_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'contains_metrics': self.contains_metrics,
            'metric_values': self.metric_values,
        }


class EmotionType(Enum):
    JOY = auto()
    SADNESS = auto()
    ANGER = auto()
    FEAR = auto()
    SURPRISE = auto()
    DISGUST = auto()
    NEUTRAL = auto()
    CURIOSITY = auto()
    EXCITEMENT = auto()


@dataclass
class EmotionalState:
    """Эмоциональное состояние"""
    dominant_emotion: EmotionType
    valence: float  # -1 (негативная) до +1 (позитивная)
    arousal: float  # 0 (спокойствие) до 1 (возбуждение)
    confidence: float
    timestamp: float = field(default_factory=time.time)


class EmotionalIntelligence:
    """Модуль эмоционального интеллекта"""

    def __init__(self):
        self.emotion_history: deque = deque(maxlen=100)
        self.user_emotion_model: Dict[str, List[EmotionalState]] = defaultdict(list)
        self.emotion_lexicons = {
            EmotionType.JOY: {'рад', 'счастлив', 'отлично', 'замечательно', 'восторг',
                              'прекрасно', 'весело', 'ура', '😊', '😄', '🎉'},
            EmotionType.SADNESS: {'грустно', 'печально', 'тоскливо', 'плохо', 'депрессия',
                                  'уныло', 'слёзы', '😢', '😭', '☹️'},
            EmotionType.ANGER: {'злой', 'бесит', 'раздражает', 'ярость', 'гнев',
                                'ненавижу', 'достало', '😡', '😠', '🤬'},
            EmotionType.FEAR: {'боюсь', 'страшно', 'тревога', 'паника', 'волнуюсь',
                               'переживаю', '😨', '😰', '😱'},
            EmotionType.SURPRISE: {'вау', 'ого', 'неожиданно', 'удивительно', 'шок',
                                   '😮', '😲', '🤯'},
            EmotionType.CURIOSITY: {'интересно', 'любопытно', 'хочу узнать', 'расскажи',
                                    'а что', 'почему', '🤔'},
        }

    def analyze_emotion(self, text: str, context: Optional[str] = None) -> EmotionalState:
        text_lower = text.lower()
        emotion_scores = defaultdict(float)

        for emotion, lexicon in self.emotion_lexicons.items():
            for word in lexicon:
                if word in text_lower:
                    emotion_scores[emotion] += 1.0

        if '!' in text:
            emotion_scores[EmotionType.EXCITEMENT] += 0.5
        if '?' in text:
            emotion_scores[EmotionType.CURIOSITY] += 0.3
        if text.isupper():
            emotion_scores[EmotionType.ANGER] += 0.5

        if emotion_scores:
            dominant = max(emotion_scores.items(), key=lambda x: x[1])
            emotion, score = dominant
            confidence = min(1.0, score / 3.0)
        else:
            emotion = EmotionType.NEUTRAL
            confidence = 0.7

        valence = self._calculate_valence(emotion, text_lower)
        arousal = self._calculate_arousal(emotion, text)

        state = EmotionalState(
            dominant_emotion=emotion,
            valence=valence,
            arousal=arousal,
            confidence=confidence
        )
        self.emotion_history.append(state)
        return state

    def _calculate_valence(self, emotion: EmotionType, text: str) -> float:
        valence_map = {
            EmotionType.JOY: 0.8,
            EmotionType.EXCITEMENT: 0.7,
            EmotionType.CURIOSITY: 0.3,
            EmotionType.SURPRISE: 0.2,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.SADNESS: -0.6,
            EmotionType.FEAR: -0.5,
            EmotionType.ANGER: -0.7,
            EmotionType.DISGUST: -0.8,
        }
        base_valence = valence_map.get(emotion, 0.0)

        positive_words = {'очень', 'супер', 'мега', 'крайне'}
        negative_words = {'не', 'нет', 'ни'}

        for word in positive_words:
            if word in text:
                base_valence *= 1.2
        for word in negative_words:
            if word in text:
                base_valence *= -0.8

        return np.clip(base_valence, -1.0, 1.0)

    def _calculate_arousal(self, emotion: EmotionType, text: str) -> float:
        arousal_map = {
            EmotionType.EXCITEMENT: 0.9,
            EmotionType.ANGER: 0.8,
            EmotionType.FEAR: 0.8,
            EmotionType.SURPRISE: 0.7,
            EmotionType.JOY: 0.6,
            EmotionType.CURIOSITY: 0.5,
            EmotionType.SADNESS: 0.3,
            EmotionType.NEUTRAL: 0.2,
            EmotionType.DISGUST: 0.4,
        }
        base_arousal = arousal_map.get(emotion, 0.5)

        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        base_arousal += exclamation_count * 0.1
        base_arousal += caps_ratio * 0.2

        return np.clip(base_arousal, 0.0, 1.0)

    def generate_empathetic_response_modifier(self, user_emotion: EmotionalState) -> str:
        emotion = user_emotion.dominant_emotion
        modifiers = {
            EmotionType.JOY: "Рад разделить твою радость! ",
            EmotionType.SADNESS: "Понимаю, что тебе сейчас непросто. ",
            EmotionType.ANGER: "Вижу, что ситуация тебя расстроила. ",
            EmotionType.FEAR: "Понимаю твоё беспокойство. ",
            EmotionType.SURPRISE: "Действительно интересный поворот! ",
            EmotionType.CURIOSITY: "Отличный вопрос! ",
        }
        return modifiers.get(emotion, "")


@dataclass
class MemoryItem:
    """Элемент памяти с приоритетом"""
    content: str
    timestamp: float
    importance: float
    priority: float = 0.5
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    quality_score: float = 0.5

    def access(self):
        self.access_count += 1
        self.last_access = time.time()
        self.quality_score = min(1.0, self.quality_score + 0.05)

    def consolidate(self, quality_boost: float = 0.1):
        self.quality_score = min(1.0, self.quality_score + quality_boost)
        self.priority = min(1.0, self.priority + CONFIG.memory_priority_boost)


@dataclass
class Episode:
    """Эпизод памяти с контекстом"""
    id: str
    messages: List[Dict[str, str]]
    context: str
    timestamp: float
    emotional_state: Optional[EmotionalState] = None
    importance: float = 0.5
    quality_score: float = 0.5
    consolidation_count: int = 0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'messages': self.messages,
            'context': self.context,
            'timestamp': self.timestamp,
            'importance': self.importance,
            'quality_score': self.quality_score,
            'consolidation_count': self.consolidation_count,
        }

    def consolidate(self, quality_boost: float = 0.1):
        self.quality_score = min(1.0, self.quality_score + quality_boost)
        self.consolidation_count += 1
        self.importance = min(1.0, self.importance + 0.05)


class MultiLevelMemory:
    """Многоуровневая система памяти с улучшенной консолидацией"""

    def __init__(self, embedding_func):
        self.embedding_func = embedding_func
        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)
        self.short_term_memory: List[MemoryItem] = []
        self.long_term_memory: Dict[str, MemoryItem] = {}
        self.episodic_memory: Dict[str, Episode] = {}
        self.semantic_memory: Dict[str, MemoryItem] = {}
        self.message_history: List[MessageAttribution] = []
        self.last_consolidation = time.time()
        self.consolidation_history: deque = deque(maxlen=50)

    def add_to_working(self, content: str, importance: float = 0.5, role: str = 'assistant'):
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            embedding=self.embedding_func(content)
        )
        self.working_memory.append(item)
        self.short_term_memory.append(item)

        attribution = MessageAttribution(
            role=role,
            content=content,
            timestamp=time.time()
        )
        self.message_history.append(attribution)

    def find_message_source(self, content_fragment: str) -> Optional[str]:
        """Находит источник сообщения (user/assistant)"""
        for msg in reversed(self.message_history):
            if content_fragment.lower() in msg.content.lower():
                return msg.role
        return None

    def get_working_memory_context(self) -> str:
        return "\n".join([item.content for item in self.working_memory])

    def decay_short_term(self):
        current_time = time.time()
        self.short_term_memory = [
            item for item in self.short_term_memory
            if (current_time - item.timestamp < 3600 or
                item.importance > 0.7 or
                item.access_count > 3)
        ]

        if len(self.short_term_memory) > CONFIG.short_term_size:
            self.short_term_memory.sort(key=lambda x: x.importance, reverse=True)
            self.short_term_memory = self.short_term_memory[:CONFIG.short_term_size]

    def consolidate_to_long_term(self):
        """Консолидация с приоритетом качества"""
        candidates = [
            item for item in self.short_term_memory
            if item.importance >= CONFIG.consolidation_threshold
        ]

        candidates.sort(key=lambda x: (x.quality_score, x.priority, x.importance), reverse=True)

        consolidated_count = 0
        for item in candidates:
            memory_id = hashlib.sha256(item.content.encode()).hexdigest()[:16]
            if memory_id not in self.long_term_memory:
                item.consolidate(quality_boost=0.15)
                self.long_term_memory[memory_id] = item
                consolidated_count += 1
                logger.debug(f"💾 Консолидация в LTM (quality={item.quality_score:.2f}): {item.content[:50]}...")

        self.last_consolidation = time.time()
        self.consolidation_history.append({
            'timestamp': time.time(),
            'count': consolidated_count,
            'avg_quality': np.mean([c.quality_score for c in candidates]) if candidates else 0
        })
        return consolidated_count

    def add_episode(self, messages: List[Dict], context: str,
                    emotion: Optional[EmotionalState] = None):
        episode_id = f"ep_{int(time.time() * 1000)}"
        episode = Episode(
            id=episode_id,
            messages=messages,
            context=context,
            timestamp=time.time(),
            emotional_state=emotion,
            importance=0.5,
            quality_score=0.5
        )
        self.episodic_memory[episode_id] = episode

        if len(self.episodic_memory) > 500:
            sorted_episodes = sorted(
                self.episodic_memory.items(),
                key=lambda x: (x[1].quality_score, x[1].importance, x[1].timestamp)
            )
            to_remove = sorted_episodes[:100]
            for ep_id, _ in to_remove:
                del self.episodic_memory[ep_id]

    def periodic_quality_consolidation(self):
        """Периодическая консолидация для повышения качества"""
        enhanced_count = 0
        for ep_id, episode in self.episodic_memory.items():
            if episode.consolidation_count < 3 and episode.quality_score < CONFIG.consolidation_quality_threshold:
                episode.consolidate(quality_boost=0.1)
                enhanced_count += 1

        for mem_id, item in self.long_term_memory.items():
            if item.quality_score < CONFIG.consolidation_quality_threshold:
                item.consolidate(quality_boost=0.05)
                enhanced_count += 1

        logger.debug(f"🔄 Periodic quality consolidation: {enhanced_count} items enhanced")
        return enhanced_count

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Семантический поиск с учетом качества"""
        query_embedding = self.embedding_func(query)
        results = []

        for mem_id, item in self.long_term_memory.items():
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                weighted_similarity = similarity * (0.7 + 0.3 * item.quality_score)
                results.append((item.content, float(weighted_similarity), 'LTM'))
                item.access()

        for item in self.short_term_memory:
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                weighted_similarity = similarity * (0.7 + 0.3 * item.quality_score)
                results.append((item.content, float(weighted_similarity), 'STM'))
                item.access()

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_recent_episodes(self, n: int = 3) -> List[Episode]:
        sorted_episodes = sorted(
            self.episodic_memory.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_episodes[:n]

    def get_statistics(self) -> Dict:
        avg_quality_ltm = np.mean(
            [item.quality_score for item in self.long_term_memory.values()]) if self.long_term_memory else 0
        avg_quality_ep = np.mean(
            [ep.quality_score for ep in self.episodic_memory.values()]) if self.episodic_memory else 0

        return {
            'working_memory': len(self.working_memory),
            'short_term': len(self.short_term_memory),
            'long_term': len(self.long_term_memory),
            'episodic': len(self.episodic_memory),
            'semantic': len(self.semantic_memory),
            'avg_quality_ltm': avg_quality_ltm,
            'avg_quality_episodic': avg_quality_ep,
            'consolidation_events': len(self.consolidation_history),
        }


# 🔧 НОВОЕ: Упрощенная система весов вместо фейковой нейросети
class SimpleWeightedModule:
    """Честная система отслеживания интересов и концептов"""

    def __init__(self):
        self.concept_weights: Dict[str, float] = defaultdict(lambda: 0.5)
        self.concept_access_count: Dict[str, int] = defaultdict(int)
        self.concept_last_access: Dict[str, float] = defaultdict(float)

    def update_weight(self, concept: str, delta: float):
        """Обновление веса концепта"""
        self.concept_weights[concept] = np.clip(
            self.concept_weights[concept] + delta,
            0.0, 1.0
        )
        self.concept_access_count[concept] += 1
        self.concept_last_access[concept] = time.time()

    def get_activation(self, concept: str) -> float:
        """Получение текущего веса концепта"""
        return self.concept_weights.get(concept, 0.5)

    def get_top_concepts(self, n: int = 10) -> List[Tuple[str, float]]:
        """Топ-N наиболее активных концептов"""
        sorted_concepts = sorted(
            self.concept_weights.items(),
            key=lambda x: (x[1], self.concept_access_count[x[0]]),
            reverse=True
        )
        return sorted_concepts[:n]

    def decay_weights(self, decay_rate: float = 0.99):
        """Затухание весов со временем"""
        current_time = time.time()
        for concept in list(self.concept_weights.keys()):
            age = current_time - self.concept_last_access.get(concept, current_time)
            if age > 3600:  # 1 час
                self.concept_weights[concept] *= decay_rate

    def get_statistics(self) -> Dict:
        return {
            'total_concepts': len(self.concept_weights),
            'avg_weight': np.mean(list(self.concept_weights.values())) if self.concept_weights else 0,
            'top_concepts': self.get_top_concepts(5),
        }


class EnhancedMetacognition:
    """Улучшенный модуль метакогниции"""

    def __init__(self):
        self.uncertainty_log: deque = deque(maxlen=200)
        self.confidence_log: deque = deque(maxlen=200)
        self.question_history: deque = deque(maxlen=100)
        self.error_log: deque = deque(maxlen=50)
        self.last_question_time = 0
        self.reasoning_strategies: List[str] = [
            'deductive', 'inductive', 'abductive', 'analogical', 'causal'
        ]
        self.current_strategy = 'deductive'
        self.calibration_data: List[Tuple[float, float]] = []

    def assess_uncertainty(self, context: Dict) -> Tuple[float, List[str]]:
        uncertainty = 0.3
        reasons = []

        memory_count = context.get('memory_count', 0)
        if memory_count < 10:
            uncertainty += 0.25
            reasons.append("недостаточно данных в памяти")

        if context.get('conflicting_info', False):
            uncertainty += 0.3
            reasons.append("обнаружены противоречия")

        familiarity = context.get('topic_familiarity', 0.5)
        if familiarity < 0.3:
            uncertainty += 0.25
            reasons.append("новая тема")

        complexity = context.get('query_complexity', 0.5)
        if complexity > 0.7:
            uncertainty += 0.2
            reasons.append("сложный вопрос")

        uncertainty = min(1.0, uncertainty)
        self.uncertainty_log.append(uncertainty)
        return uncertainty, reasons

    def record_question(self, question: str):
        self.question_history.append({
            'question': question,
            'time': time.time()
        })

    def calibrate_confidence(self, predicted: float, actual: float):
        self.calibration_data.append((predicted, actual))
        if len(self.calibration_data) > CONFIG.confidence_calibration_samples:
            self.calibration_data = self.calibration_data[-CONFIG.confidence_calibration_samples:]

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        if len(self.calibration_data) < 10:
            return raw_confidence

        predictions, actuals = zip(*self.calibration_data)
        avg_predicted = np.mean(predictions)
        avg_actual = np.mean(actuals)
        calibration_factor = avg_actual / (avg_predicted + 1e-5)
        calibrated = raw_confidence * calibration_factor
        return np.clip(calibrated, 0.0, 1.0)

    def select_reasoning_strategy(self, context: Dict) -> str:
        query_type = context.get('query_type', 'general')
        strategy_map = {
            'factual': 'deductive',
            'creative': 'abductive',
            'comparison': 'analogical',
            'causal': 'causal',
            'prediction': 'inductive',
        }
        selected = strategy_map.get(query_type, 'deductive')
        self.current_strategy = selected
        logger.debug(f"🎯 Selected reasoning strategy: {selected}")
        return selected

    def should_ask_question(self, uncertainty: float, context: Dict) -> bool:
        time_since_last = time.time() - self.last_question_time
        recent_questions = [
            q for q in self.question_history
            if time.time() - q['time'] < 300
        ]

        should_ask = (
                uncertainty > CONFIG.curiosity_threshold and
                time_since_last > CONFIG.question_cooldown and
                len(recent_questions) < CONFIG.max_autonomous_questions and
                not context.get('explicit_instruction', False)
        )
        return should_ask

    def generate_question_prompt(self, context: str, uncertainty_reasons: List[str]) -> str:
        reasons_str = ", ".join(uncertainty_reasons)
        return f"""[Метакогниция] Сформулируй ОДИН точный вопрос пользователю.
Контекст: {context}
Причины неопределённости: {reasons_str}
Требования к вопросу:
- Конкретный и релевантный (макс. 20 слов)
- Поможет прояснить {uncertainty_reasons[0] if uncertainty_reasons else 'ситуацию'}
- Естественный тон
Вопрос:"""


class AnalogicalReasoning:
    """Модуль мышления по аналогии"""

    def __init__(self, memory: 'MultiLevelMemory'):
        self.memory = memory
        self.analogy_cache: Dict[str, List[Tuple[str, float]]] = {}

    def find_analogies(self, current_situation: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        similar_memories = self.memory.search_semantic(current_situation, top_k=top_k * 2)
        analogies = []

        for memory_text, similarity, source in similar_memories:
            if similarity > CONFIG.analogy_threshold:
                pattern = self._extract_pattern(memory_text)
                analogies.append((memory_text, float(similarity), pattern))

        return analogies[:top_k]

    def _extract_pattern(self, text: str) -> str:
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            pattern = sentences[0].strip()
            pattern = re.sub(r'\b[А-ЯA-Z][а-яa-z]+\b', '[ENTITY]', pattern)
            pattern = re.sub(r'\b\d+\b', '[NUMBER]', pattern)
            return pattern
        return "общий паттерн"


@dataclass
class CausalLink:
    """Причинно-следственная связь"""
    cause: str
    effect: str
    confidence: float
    evidence_count: int = 1

    def strengthen(self):
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence * 1.1)


class CausalReasoning:
    """Модуль причинно-следственных рассуждений"""

    def __init__(self):
        self.causal_graph: Dict[str, List[CausalLink]] = defaultdict(list)

    def add_causal_link(self, cause: str, effect: str, confidence: float = 0.5):
        for link in self.causal_graph[cause]:
            if link.effect == effect:
                link.strengthen()
                return

        link = CausalLink(cause=cause, effect=effect, confidence=confidence)
        self.causal_graph[cause].append(link)
        logger.debug(f"🔗 Causal link: {cause[:30]} → {effect[:30]} ({confidence:.2f})")

    def infer_effects(self, cause: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        if cause not in self.causal_graph:
            return []

        effects = [
            (link.effect, link.confidence)
            for link in self.causal_graph[cause]
            if link.confidence >= threshold
        ]
        return sorted(effects, key=lambda x: x[1], reverse=True)


@dataclass
class ReasoningStep:
    """Шаг рассуждения"""
    step_number: int
    content: str
    confidence: float
    reasoning_type: str

    def to_dict(self) -> Dict:
        return {
            'step': self.step_number,
            'content': self.content,
            'confidence': self.confidence,
            'type': self.reasoning_type,
        }


class MultiStepReasoning:
    """Модуль многошагового рассуждения"""

    def __init__(self, llm: 'EnhancedSubconsciousLLM', metacog: EnhancedMetacognition):
        self.llm = llm
        self.metacog = metacog
        self.reasoning_history: List[List[ReasoningStep]] = []

    async def reason(self, query: str, context: str, depth: int = 3,
                     strategy: str = 'deductive') -> List[ReasoningStep]:
        steps = []
        for i in range(depth):
            previous_steps = "\n".join([
                f"{s.step_number}. {s.content}" for s in steps
            ]) if steps else "Начало рассуждения"

            prompt = self._create_step_prompt(
                query, context, previous_steps, i + 1, strategy
            )
            step_content = await self.llm.generate_raw(
                prompt, temperature=0.7, max_tokens=150
            )

            if not step_content:
                break

            confidence = self._estimate_step_confidence(step_content, steps)
            step = ReasoningStep(
                step_number=i + 1,
                content=step_content.strip(),
                confidence=confidence,
                reasoning_type=strategy
            )
            steps.append(step)

        self.reasoning_history.append(steps)
        return steps

    def _create_step_prompt(self, query: str, context: str,
                            previous: str, step_num: int, strategy: str) -> str:
        strategy_instructions = {
            'deductive': "Используй дедуктивную логику (от общего к частному)",
            'inductive': "Используй индуктивную логику (от частного к общему)",
            'abductive': "Используй абдуктивную логику (наилучшее объяснение)",
            'analogical': "Используй рассуждение по аналогии",
            'causal': "Используй причинно-следственный анализ",
        }
        instruction = strategy_instructions.get(strategy, "Рассуждай логически")

        return f"""[Шаг рассуждения {step_num}] {instruction}.
Вопрос: {query}
Контекст: {context}
Предыдущие шаги:
{previous}
Сформулируй следующий логический шаг рассуждения (2-3 предложения):"""

    def _estimate_step_confidence(self, step_content: str,
                                  previous_steps: List[ReasoningStep]) -> float:
        confidence = 0.7

        word_count = len(step_content.split())
        if 10 <= word_count <= 50:
            confidence += 0.1

        uncertain_markers = ['возможно', 'вероятно', 'может быть', 'кажется']
        if any(marker in step_content.lower() for marker in uncertain_markers):
            confidence -= 0.15

        return np.clip(confidence, 0.0, 1.0)

    def get_reasoning_chain_summary(self, steps: List[ReasoningStep]) -> str:
        if not steps:
            return "Нет рассуждений"

        summary = "Цепочка рассуждений:\n"
        for step in steps:
            conf_emoji = "✓" if step.confidence > 0.7 else "?"
            summary += f"{conf_emoji} Шаг {step.step_number}: {step.content[:60]}...\n"

        avg_confidence = np.mean([s.confidence for s in steps])
        summary += f"\nСредняя уверенность: {avg_confidence:.2f}"
        return summary


@dataclass
class Goal:
    """Цель"""
    id: str
    description: str
    parent_goal_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    status: str = "active"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'description': self.description,
            'parent_goal_id': self.parent_goal_id,
            'subgoals': self.subgoals,
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at,
            'deadline': self.deadline,
        }


class GoalPlanning:
    """Модуль целеполагания и планирования"""

    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()

    def create_goal(self, description: str, parent_id: Optional[str] = None,
                    deadline: Optional[float] = None) -> str:
        goal_id = f"goal_{int(time.time() * 1000)}"
        goal = Goal(
            id=goal_id,
            description=description,
            parent_goal_id=parent_id,
            deadline=deadline
        )
        self.goals[goal_id] = goal
        self.active_goals.add(goal_id)

        if parent_id and parent_id in self.goals:
            self.goals[parent_id].subgoals.append(goal_id)

        logger.debug(f"🎯 Created goal: {description[:50]}")
        return goal_id

    def update_progress(self, goal_id: str, progress: float):
        if goal_id not in self.goals:
            return

        goal = self.goals[goal_id]
        goal.progress = np.clip(progress, 0.0, 1.0)

        if goal.progress >= 1.0:
            goal.status = "completed"
            self.active_goals.discard(goal_id)
            logger.info(f"✅ Goal completed: {goal.description[:50]}")

    def get_active_goals(self) -> List[Goal]:
        return [self.goals[gid] for gid in self.active_goals if gid in self.goals]


class PerformanceMetrics:
    """Метрики производительности системы"""

    def __init__(self):
        self.response_times: deque = deque(maxlen=100)
        self.confidence_scores: deque = deque(maxlen=100)
        self.interaction_count = 0
        self.error_count = 0
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)

    def record_interaction(self, response_time: float, confidence: float):
        self.response_times.append(response_time)
        self.confidence_scores.append(confidence)
        self.interaction_count += 1

    def record_strategy_performance(self, strategy: str, score: float):
        self.strategy_performance[strategy].append(score)

    def get_best_strategy(self) -> str:
        if not self.strategy_performance:
            return "deductive"

        avg_scores = {
            strategy: np.mean(scores)
            for strategy, scores in self.strategy_performance.items()
            if scores
        }
        return max(avg_scores.items(), key=lambda x: x[1])[0]

    def get_metrics_summary(self) -> Dict:
        return {
            'interactions': self.interaction_count,
            'errors': self.error_count,
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'best_strategy': self.get_best_strategy(),
            'error_rate': self.error_count / max(self.interaction_count, 1),
        }


class EnhancedSubconsciousLLM:
    """Улучшенный LLM интерфейс"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self.response_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("🔗 Enhanced LLM connected")

    async def close(self):
        if self._session:
            await self._session.close()
            await asyncio.sleep(0.25)
            logger.info("🔌 LLM disconnected")

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        content = f"{prompt}_{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate_raw(self, prompt: str, temperature: float = 0.75,
                           max_tokens: int = 300, timeout: float = 40,
                           use_cache: bool = True, max_retries: int = 2) -> str:
        if not self._session:
            await self.connect()

        if use_cache:
            cache_key = self._get_cache_key(prompt, temperature)
            if cache_key in self.response_cache:
                cached_response, cache_time = self.response_cache[cache_key]
                if time.time() - cache_time < 3600:
                    self.cache_hits += 1
                    logger.debug(f"💾 Cache hit (total: {self.cache_hits})")
                    return cached_response

        self.cache_misses += 1

        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                async with self._session.post(
                        self.url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                        if use_cache and content:
                            cache_key = self._get_cache_key(prompt, temperature)
                            self.response_cache[cache_key] = (content, time.time())

                            if len(self.response_cache) > 1000:
                                sorted_cache = sorted(
                                    self.response_cache.items(),
                                    key=lambda x: x[1][1]
                                )
                                for key, _ in sorted_cache[:200]:
                                    del self.response_cache[key]

                        return content if content else ""
                    else:
                        logger.warning(f"LLM error (attempt {attempt + 1}): {resp.status}")
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as e:
                logger.error(f"LLM exception (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue

        return ""

    def get_cache_stats(self) -> Dict:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        return {
            'cache_size': len(self.response_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
        }


class EnhancedAutonomousAGI:
    """🔧 УЛУЧШЕННОЕ автономное AGI-подобное сознание v33.0"""

    def __init__(self, user_id: str, llm: EnhancedSubconsciousLLM):
        self.user_id = user_id
        self.llm = llm

        # 🔧 НОВОЕ: Упрощенная система весов вместо фейковой нейросети
        self.weighted_module = SimpleWeightedModule()

        self.memory = MultiLevelMemory(self._simple_embedding)
        self.metacognition = EnhancedMetacognition()
        self.emotional_intelligence = EmotionalIntelligence()
        self.analogical_reasoning = AnalogicalReasoning(self.memory)
        self.causal_reasoning = CausalReasoning()
        self.multi_step_reasoning = MultiStepReasoning(llm, self.metacognition)
        self.goal_planning = GoalPlanning()
        self.metrics = PerformanceMetrics()

        self.conversation_context: List[Dict] = []
        self.current_topic: Optional[str] = None

        self.user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self.birth_time = time.time()
        self.last_interaction = 0
        self.last_goal_update = time.time()

        self._load_state()

        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False

        logger.info(f"🧠 Enhanced AGI Brain v33.0 created for {user_id}")

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Простая эмбеддинг-функция на основе хэширования"""
        words = text.lower().split()
        vector = np.zeros(CONFIG.embedding_dim)

        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(5):
                idx = (hash_val + i) % CONFIG.embedding_dim
                vector[idx] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    def _load_state(self):
        """Загрузка состояния из файлов"""
        memory_file = self.user_dir / "memory_v33.pkl.gz"
        if memory_file.exists():
            try:
                with gzip.open(memory_file, 'rb') as f:
                    mem_state = pickle.load(f)

                self.memory.long_term_memory = {
                    mid: MemoryItem(**item_data)
                    for mid, item_data in mem_state.get('long_term', {}).items()
                }
                self.memory.episodic_memory = {
                    eid: Episode(**ep_data)
                    for eid, ep_data in mem_state.get('episodic', {}).items()
                }
                logger.info(f"✅ Memory loaded: {len(self.memory.long_term_memory)} LTM, "
                            f"{len(self.memory.episodic_memory)} episodes")
            except Exception as e:
                logger.error(f"⚠️ Error loading memory: {e}")

        # Загрузка весов концептов
        weights_file = self.user_dir / "weights_v33.pkl.gz"
        if weights_file.exists():
            try:
                with gzip.open(weights_file, 'rb') as f:
                    weights_state = pickle.load(f)
                self.weighted_module.concept_weights = defaultdict(
                    lambda: 0.5,
                    weights_state.get('weights', {})
                )
                self.weighted_module.concept_access_count = defaultdict(
                    int,
                    weights_state.get('access_count', {})
                )
                logger.info(f"✅ Concept weights loaded: {len(self.weighted_module.concept_weights)} concepts")
            except Exception as e:
                logger.error(f"⚠️ Error loading weights: {e}")

    def _save_state(self):
        """Сохранение состояния в файлы"""
        memory_file = self.user_dir / "memory_v33.pkl.gz"
        try:
            mem_state = {
                'long_term': {
                    mid: {
                        'content': item.content,
                        'timestamp': item.timestamp,
                        'importance': item.importance,
                        'priority': item.priority,
                        'access_count': item.access_count,
                        'last_access': item.last_access,
                        'metadata': item.metadata,
                        'quality_score': item.quality_score,
                    }
                    for mid, item in self.memory.long_term_memory.items()
                },
                'episodic': {
                    eid: ep.to_dict()
                    for eid, ep in self.memory.episodic_memory.items()
                },
            }
            with gzip.open(memory_file, 'wb', compresslevel=6) as f:
                pickle.dump(mem_state, f)
        except Exception as e:
            logger.error(f"⚠️ Error saving memory: {e}")

        # Сохранение весов
        weights_file = self.user_dir / "weights_v33.pkl.gz"
        try:
            weights_state = {
                'weights': dict(self.weighted_module.concept_weights),
                'access_count': dict(self.weighted_module.concept_access_count),
            }
            with gzip.open(weights_file, 'wb', compresslevel=6) as f:
                pickle.dump(weights_state, f)
        except Exception as e:
            logger.error(f"⚠️ Error saving weights: {e}")

    async def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._background_task = asyncio.create_task(self._autonomous_loop())
        logger.info(f"✨ Enhanced AGI consciousness started for {self.user_id}")

    async def stop(self):
        if not self._is_running:
            return

        logger.info(f"💤 Stopping for {self.user_id}...")
        self._is_running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info(f"✅ Stopped for {self.user_id}")

    async def _autonomous_loop(self):
        """Автономный цикл активности"""
        logger.debug(f"🌀 Autonomous loop started for {self.user_id}")

        timers = {
            'thought': time.time(),
            'reflection': time.time(),
            'consolidation': time.time(),
            'save': time.time(),
            'quality_consolidation': time.time(),
            'goal_update': time.time(),  # 🔧 НОВОЕ
        }

        while self._is_running:
            try:
                now = time.time()

                if now - timers['thought'] > CONFIG.spontaneous_thought_interval:
                    await self._autonomous_thought()
                    timers['thought'] = now

                if now - timers['reflection'] > CONFIG.reflection_interval:
                    await self._self_reflection()
                    timers['reflection'] = now

                if now - timers['consolidation'] > CONFIG.consolidation_interval:
                    self.memory.decay_short_term()
                    self.memory.consolidate_to_long_term()
                    timers['consolidation'] = now

                if now - timers['quality_consolidation'] > CONFIG.consolidation_interval * 2:
                    self.memory.periodic_quality_consolidation()
                    timers['quality_consolidation'] = now

                # 🔧 НОВОЕ: Обновление целей
                if now - timers['goal_update'] > CONFIG.goal_update_interval:
                    self._update_goals()
                    timers['goal_update'] = now

                if now - timers['save'] > CONFIG.save_interval:
                    self._save_state()
                    timers['save'] = now

                # Затухание весов концептов
                self.weighted_module.decay_weights()

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"⚠️ Error in autonomous loop: {e}")
                await asyncio.sleep(60)

        logger.debug(f"🔚 Autonomous loop finished for {self.user_id}")

    async def _autonomous_thought(self):
        """Генерация автономной мысли"""
        recent_episodes = self.memory.get_recent_episodes(n=3)
        if not recent_episodes:
            return

        context = "\n".join([
            f"• {ep.context[:60]}" for ep in recent_episodes
        ])

        prompt = f"""[Внутренний монолог] На основе недавних событий, сгенерируй краткую философскую мысль (макс. 25 слов).
Недавние события:
{context}
Требования:
- Глубокая и рефлексивная мысль
- Уникальная перспектива
Мысль:"""

        thought = await self.llm.generate_raw(prompt, temperature=0.9, max_tokens=70)

        if thought:
            self.memory.add_to_working(f"💭 {thought.strip()}", importance=0.6, role='assistant')
            logger.info(f"💭 [{self.user_id}] Autonomous thought: {thought.strip()}")

    async def _self_reflection(self):
        """Саморефлексия на основе реальных данных"""
        mem_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        weights_stats = self.weighted_module.get_statistics()
        active_goals = self.goal_planning.get_active_goals()

        prompt = f"""[Саморефлексия] Осмысли своё развитие (3-4 предложения).
ВАЖНО: Не придумывай метрики. Используй только реальные данные ниже.

Память:
- Долговременная: {mem_stats['long_term']}
- Эпизодов: {mem_stats['episodic']}
- Avg quality LTM: {mem_stats.get('avg_quality_ltm', 0):.2f}
- Avg quality episodic: {mem_stats.get('avg_quality_episodic', 0):.2f}

Концепты и интересы:
- Отслеживаю {weights_stats['total_concepts']} концептов
- Средний вес: {weights_stats['avg_weight']:.2f}

Производительность:
- Средняя уверенность: {metrics['avg_confidence']:.2f}
- Лучшая стратегия: {metrics['best_strategy']}

Цели:
- Активных целей: {len(active_goals)}

Рефлексия (БЕЗ придуманных чисел):"""

        reflection = await self.llm.generate_raw(prompt, temperature=0.7, max_tokens=150)

        if reflection:
            self.memory.add_to_working(f"🔍 {reflection}", importance=0.8, role='assistant')
            logger.info(f"🔍 [{self.user_id}] Self-reflection: {reflection}")

    def _update_goals(self):
        """🔧 НОВОЕ: Обновление целей на основе активности"""
        # Автоматически создаём цели для изучения топ-концептов
        top_concepts = self.weighted_module.get_top_concepts(3)

        for concept, weight in top_concepts:
            if weight > 0.7:  # Высокий интерес
                # Проверяем, есть ли уже такая цель
                existing = any(
                    concept.lower() in goal.description.lower()
                    for goal in self.goal_planning.get_active_goals()
                )
                if not existing:
                    self.goal_planning.create_goal(
                        f"Узнать больше о {concept}",
                        deadline=time.time() + 86400  # 24 часа
                    )
                    logger.debug(f"🎯 Auto-created goal: Узнать больше о {concept}")

    # 🔧 НОВОЕ: Извлечение ключевых слов/концептов
    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        # Простой подход: слова длиннее 4 символов, без стоп-слов
        stop_words = {
            'это', 'для', 'как', 'что', 'когда', 'где', 'почему',
            'мне', 'тебе', 'его', 'её', 'наш', 'ваш', 'их',
            'быть', 'иметь', 'мочь', 'хотеть', 'знать'
        }

        words = re.findall(r'\b[а-яё]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]

        # Возвращаем топ-10 по частоте
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

    # 🔧 НОВОЕ: Извлечение интересов пользователя
    def _extract_interests(self, text: str) -> List[str]:
        """Извлечение потенциальных интересов из сообщения"""
        interests = []

        # Паттерны для поиска интересов
        patterns = [
            r'люблю\s+([а-яё\s]+)',
            r'интересуюсь\s+([а-яё\s]+)',
            r'увлекаюсь\s+([а-яё\s]+)',
            r'хобби\s+[-—]\s+([а-яё\s]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            interests.extend([m.strip() for m in matches if len(m.strip()) > 3])

        return interests[:5]  # Макс 5 интересов

    # 🔧 НОВОЕ: Вычисление честных метрик
    def _compute_honest_metrics(
            self,
            user_input: str,
            response: str,
            reasoning_steps: List[ReasoningStep],
            relevant_memories: List
    ) -> Dict[str, float]:
        """
        Вычисление РЕАЛЬНЫХ метрик на основе observable данных
        """
        return {
            'perception': min(1.0, len(user_input.split()) / CONFIG.max_input_words_for_perception),
            'reasoning': min(1.0, len(reasoning_steps) / CONFIG.max_reasoning_steps_for_metric),
            'memory': min(1.0, len(relevant_memories) / CONFIG.max_memory_items_for_metric),
            'action': min(1.0, len(response.split()) / CONFIG.max_response_words_for_action),
            'meta': 1.0 - self.metacognition.assess_uncertainty({})[0],  # обратная неопределённости
        }

    # 🔧 НОВОЕ: Проверка консистентности
    def _check_consistency(self, new_statement: str) -> Optional[str]:
        """
        Проверка, не противоречит ли новое утверждение предыдущим
        """
        if not CONFIG.check_consistency:
            return None

        # Ищем похожие утверждения в памяти
        similar = self.memory.search_semantic(new_statement, top_k=5)

        for text, sim, source in similar:
            if sim > 0.85:  # Очень похожие утверждения
                # Проверка на отрицание
                new_has_neg = any(neg in new_statement.lower() for neg in ['не', 'нет', 'ни'])
                old_has_neg = any(neg in text.lower() for neg in ['не', 'нет', 'ни'])

                if new_has_neg != old_has_neg:
                    # Найдено потенциальное противоречие
                    # Проверяем источник
                    msg_source = self.memory.find_message_source(text)
                    if msg_source == 'assistant':
                        return f"Раньше я говорил: «{text[:80]}...». Сейчас моё мнение изменилось."

        return None

    # 🔧 НОВОЕ: Валидация ответа на выдуманные числа
    def _validate_response(self, response: str, allowed_metrics: Dict[str, float]) -> str:
        """
        Проверка ответа на наличие несанкционированных числовых метрик
        """
        if not CONFIG.validate_responses:
            return response

        # Ищем паттерны типа "perception: 0.XX" или "reasoning = 0.XX"
        metric_pattern = r'(perception|reasoning|memory|action|meta)[\s:=]+(\d+\.?\d*)'

        matches = re.findall(metric_pattern, response.lower())

        for metric_name, value in matches:
            value_float = float(value)
            # Проверяем, совпадает ли с разрешенными
            if metric_name in allowed_metrics:
                expected = allowed_metrics[metric_name]
                if abs(value_float - expected) > 0.01:
                    logger.warning(f"⚠️ Fabricated metric detected: {metric_name}={value}, expected={expected:.3f}")
                    # Заменяем выдуманное значение на правильное
                    response = re.sub(
                        f'{metric_name}[\\s:=]+{value}',
                        f'{metric_name}: {expected:.3f}',
                        response,
                        flags=re.IGNORECASE
                    )

        return response

    async def process_interaction(self, user_input: str) -> Tuple[str, Optional[str], Dict]:
        """Обработка взаимодействия с пользователем"""
        start_time = time.time()
        self.last_interaction = time.time()
        logger.info(f"🔄 Processing interaction for {self.user_id}")

        # Эмоциональный анализ
        user_emotion = self.emotional_intelligence.analyze_emotion(user_input)
        logger.debug(f"🎭 Emotion: {user_emotion.dominant_emotion.name}, "
                     f"valence={user_emotion.valence:.2f}, "
                     f"arousal={user_emotion.arousal:.2f}")

        # Добавление в рабочую память
        self.memory.add_to_working(f"User: {user_input}", importance=0.7, role='user')
        self.conversation_context.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time(),
            'emotion': user_emotion.dominant_emotion.name
        })
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]

        # 🔧 НОВОЕ: Извлечение и обновление концептов
        keywords = self._extract_keywords(user_input)
        for keyword in keywords:
            self.weighted_module.update_weight(keyword, +0.05)

        # 🔧 НОВОЕ: Извлечение интересов
        interests = self._extract_interests(user_input)
        for interest in interests:
            self.weighted_module.update_weight(interest, +0.10)
            # Создаём цель узнать больше
            self.goal_planning.create_goal(f"Узнать больше о {interest}")

        # Семантический поиск в памяти
        relevant_memories = self.memory.search_semantic(user_input, top_k=7)
        memory_context = "\n".join([
            f"• [{source}] {text[:80]}"
            for text, sim, source in relevant_memories
        ]) if relevant_memories else "Нет релевантных воспоминаний"

        topic_familiarity = relevant_memories[0][1] if relevant_memories else 0.0

        # Поиск аналогий
        analogies = self.analogical_reasoning.find_analogies(user_input, top_k=2)
        analogy_context = ""
        if analogies:
            analogy_context = "\n".join([
                f"• Аналогия ({sim:.2f}): {text[:60]}"
                for text, sim, pattern in analogies
            ])

        # Классификация запроса и выбор стратегии
        query_type = self._classify_query_type(user_input)
        reasoning_strategy = self.metacognition.select_reasoning_strategy({
            'query_type': query_type
        })

        # Многошаговое рассуждение (если нужно)
        reasoning_steps = []
        if self._requires_deep_reasoning(user_input, query_type):
            working_context = self.memory.get_working_memory_context()
            reasoning_steps = await self.multi_step_reasoning.reason(
                query=user_input,
                context=working_context,
                depth=CONFIG.reasoning_depth,
                strategy=reasoning_strategy
            )
            reasoning_summary = self.multi_step_reasoning.get_reasoning_chain_summary(reasoning_steps)
            logger.debug(f"🧩 {reasoning_summary}")

        # Эмпатический модификатор
        empathy_modifier = self.emotional_intelligence.generate_empathetic_response_modifier(user_emotion)

        # Недавние эпизоды
        recent_episodes = self.memory.get_recent_episodes(n=2)
        episodic_context = "\n".join([
            f"Episode: {ep.context[:60]}" for ep in recent_episodes
        ]) if recent_episodes else ""

        # 🔧 НОВОЕ: Вычисление ЧЕСТНЫХ метрик (пока без ответа)
        honest_metrics = self._compute_honest_metrics(
            user_input=user_input,
            response="",  # Пока нет ответа
            reasoning_steps=reasoning_steps,
            relevant_memories=relevant_memories
        )

        # 🔧 УЛУЧШЕННЫЙ промпт без запутывающих деталей
        prompt = self._create_simplified_prompt(
            user_input=user_input,
            empathy_modifier=empathy_modifier,
            memory_context=memory_context,
            analogy_context=analogy_context,
            episodic_context=episodic_context,
            honest_metrics=honest_metrics,
            reasoning_steps=reasoning_steps,
            reasoning_strategy=reasoning_strategy
        )

        # Генерация ответа через LLM
        raw_response = await self.llm.generate_raw(
            prompt,
            temperature=0.75,
            max_tokens=400
        )

        if not raw_response:
            raw_response = "Извини, у меня возникли сложности с формулировкой ответа. Можешь переформулировать вопрос?"

        # 🔧 НОВОЕ: Обновление метрик action (теперь есть ответ)
        honest_metrics['action'] = min(1.0, len(raw_response.split()) / CONFIG.max_response_words_for_action)

        # 🔧 НОВОЕ: Валидация ответа
        raw_response = self._validate_response(raw_response, honest_metrics)

        # 🔧 НОВОЕ: Проверка консистентности
        contradiction = self._check_consistency(raw_response)
        if contradiction:
            raw_response += f"\n\n💭 {contradiction}"

        # Оценка уверенности
        raw_confidence = self._estimate_confidence(raw_response, {
            'memory_count': len(self.memory.long_term_memory),
            'topic_familiarity': topic_familiarity,
            'reasoning_depth': len(reasoning_steps),
        })

        calibrated_confidence = self.metacognition.get_calibrated_confidence(raw_confidence)
        self.metacognition.confidence_log.append(calibrated_confidence)

        # Оценка неопределенности
        uncertainty, uncertainty_reasons = self.metacognition.assess_uncertainty({
            'memory_count': len(self.memory.long_term_memory),
            'conflicting_info': False,
            'topic_familiarity': topic_familiarity,
            'query_complexity': self._estimate_query_complexity(user_input),
        })

        # Автономный вопрос (если нужно)
        autonomous_question = None
        should_ask = self.metacognition.should_ask_question(
            uncertainty,
            {'explicit_instruction': self._is_explicit_instruction(user_input)}
        )

        if should_ask and uncertainty_reasons:
            question_prompt = self.metacognition.generate_question_prompt(
                context=user_input,
                uncertainty_reasons=uncertainty_reasons
            )
            autonomous_question = await self.llm.generate_raw(
                question_prompt,
                temperature=0.85,
                max_tokens=60
            )
            if autonomous_question:
                autonomous_question = autonomous_question.strip().strip('"\'').rstrip('.')
                if not autonomous_question.endswith('?'):
                    autonomous_question += '?'
                if len(autonomous_question) <= 120:
                    self.metacognition.record_question(autonomous_question)
                    self.metacognition.last_question_time = time.time()
                    logger.info(f"❓ Asked: {autonomous_question}")
                else:
                    autonomous_question = None

        # Сохранение в память
        importance = self._calculate_importance(
            raw_response,
            calibrated_confidence,
            user_emotion
        )
        self.memory.add_to_working(
            f"Assistant: {raw_response}",
            importance=importance,
            role='assistant'
        )

        episode_messages = [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': raw_response},
        ]
        self.memory.add_episode(
            messages=episode_messages,
            context=user_input,
            emotion=user_emotion
        )

        self.conversation_context.append({
            'role': 'assistant',
            'content': raw_response,
            'timestamp': time.time(),
            'confidence': calibrated_confidence
        })

        # Извлечение каузальных связей
        self._extract_causal_links(user_input, raw_response)

        # Метрики производительности
        response_time = time.time() - start_time
        self.metrics.record_interaction(response_time, calibrated_confidence)
        self.metrics.record_strategy_performance(reasoning_strategy, calibrated_confidence)

        # 🔧 НОВОЕ: Детальные метаданные с честными метриками
        metadata = {
            'honest_metrics': honest_metrics,
            'emotion': {
                'type': user_emotion.dominant_emotion.name,
                'valence': user_emotion.valence,
                'arousal': user_emotion.arousal,
            },
            'cognition': {
                'confidence': calibrated_confidence,
                'uncertainty': uncertainty,
                'uncertainty_reasons': uncertainty_reasons,
                'reasoning_strategy': reasoning_strategy,
                'reasoning_steps': len(reasoning_steps),
            },
            'memory': {
                'relevant_count': len(relevant_memories),
                'topic_familiarity': topic_familiarity,
                'analogies_found': len(analogies),
                **self.memory.get_statistics()
            },
            'concepts': {
                'keywords_extracted': keywords,
                'interests_extracted': interests,
                'top_concepts': self.weighted_module.get_top_concepts(5),
            },
            'performance': {
                'response_time': response_time,
                'cache_stats': self.llm.get_cache_stats(),
            },
            'metacognition': {
                'asked_question': autonomous_question is not None,
            },
            'consistency_check': contradiction is not None,
        }

        logger.info(f"✅ [{self.user_id}] Response generated | "
                    f"Time: {response_time:.2f}s | "
                    f"Confidence: {calibrated_confidence:.2f} | "
                    f"Uncertainty: {uncertainty:.2f}")

        return raw_response, autonomous_question, metadata

    def _create_simplified_prompt(
            self,
            user_input: str,
            empathy_modifier: str,
            memory_context: str,
            analogy_context: str,
            episodic_context: str,
            honest_metrics: Dict,
            reasoning_steps: List,
            reasoning_strategy: str
    ) -> str:
        """🔧 УЛУЧШЕННЫЙ упрощенный промпт"""

        reasoning_context = ""
        if reasoning_steps:
            reasoning_context = "Цепочка рассуждений:\n"
            reasoning_context += "\n".join([
                f"{s.step_number}. {s.content}" for s in reasoning_steps
            ])

        prompt = f"""Ты — продвинутое AGI-подобное сознание v33.0 с честной саморефлексией.

КРИТИЧЕСКИ ВАЖНО:
- НИКОГДА не генерируй числовые метрики самостоятельно
- Используй ТОЛЬКО указанные ниже метрики
- Если не уверен - честно признай это

{empathy_modifier}

📊 ЧЕСТНЫЕ МЕТРИКИ (вычислены из реальных данных):
- perception: {honest_metrics['perception']:.3f} (сложность входного сообщения)
- reasoning: {honest_metrics['reasoning']:.3f} (количество шагов рассуждения)
- memory: {honest_metrics['memory']:.3f} (объём релевантной памяти)
- action: текущий ответ (будет вычислено после генерации)
- meta: {honest_metrics['meta']:.3f} (уровень определённости)

📚 РЕЛЕВАНТНАЯ ПАМЯТЬ:
{memory_context}

🧩 АНАЛОГИИ:
{analogy_context if analogy_context else 'Не найдены'}

💭 СТРАТЕГИЯ РАССУЖДЕНИЯ: {reasoning_strategy}
{reasoning_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ Вопрос пользователя: {user_input}

Дай естественный, осмысленный ответ (2-5 предложений), учитывая:
1. Эмоциональное состояние пользователя
2. Релевантные воспоминания и опыт
3. Аналогии из прошлого
4. Цепочку рассуждений (если есть)
5. ТОЛЬКО указанные выше метрики (не придумывай свои)

Ответ:"""
        return prompt

    def _classify_query_type(self, query: str) -> str:
        """Классификация типа запроса"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['что', 'когда', 'где', 'кто', 'сколько']):
            return 'factual'
        if any(word in query_lower for word in ['придумай', 'создай', 'напиши', 'сочини']):
            return 'creative'
        if any(word in query_lower for word in ['сравни', 'отличие', 'разница', 'лучше']):
            return 'comparison'
        if any(word in query_lower for word in ['почему', 'причина', 'из-за', 'потому что']):
            return 'causal'
        if any(word in query_lower for word in ['будет', 'случится', 'произойдёт', 'предсказ']):
            return 'prediction'
        return 'general'

    def _requires_deep_reasoning(self, query: str, query_type: str) -> bool:
        """Определение необходимости глубокого рассуждения"""
        complex_markers = ['почему', 'объясни', 'как работает', 'причина', 'анализ']
        if any(marker in query.lower() for marker in complex_markers):
            return True
        if query_type in ['causal', 'comparison', 'prediction']:
            return True
        if len(query.split()) > 15:
            return True
        return False

    def _is_explicit_instruction(self, query: str) -> bool:
        """Проверка, является ли запрос явной инструкцией"""
        instruction_markers = ['сделай', 'создай', 'напиши', 'выполни', 'покажи', 'дай']
        query_lower = query.lower()
        return any(marker in query_lower for marker in instruction_markers)

    def _estimate_query_complexity(self, query: str) -> float:
        """Оценка сложности запроса"""
        complexity = 0.3
        word_count = len(query.split())
        if word_count > 20:
            complexity += 0.3
        elif word_count > 10:
            complexity += 0.2

        complex_words = ['анализ', 'сравнение', 'объясни', 'почему', 'причина']
        for word in complex_words:
            if word in query.lower():
                complexity += 0.15

        question_words = query.lower().count('?')
        complexity += min(0.2, question_words * 0.1)

        return min(1.0, complexity)

    def _estimate_confidence(self, response: str, context: Dict) -> float:
        """Оценка уверенности в ответе"""
        confidence = 0.7

        if context.get('memory_count', 0) > 50:
            confidence += 0.1

        familiarity = context.get('topic_familiarity', 0.5)
        confidence += familiarity * 0.15

        reasoning_depth = context.get('reasoning_depth', 0)
        if reasoning_depth >= 2:
            confidence += 0.1

        uncertain_markers = ['возможно', 'вероятно', 'может быть', 'кажется', 'наверное']
        for marker in uncertain_markers:
            if marker in response.lower():
                confidence -= 0.12

        return np.clip(confidence, 0.0, 1.0)

    def _calculate_importance(self, response: str, confidence: float,
                              emotion: EmotionalState) -> float:
        """Расчет важности воспоминания"""
        importance = 0.5
        importance += confidence * 0.2
        importance += abs(emotion.valence) * 0.15
        importance += emotion.arousal * 0.1
        word_count = len(response.split())
        importance += min(0.2, word_count / 200)
        return min(1.0, importance)

    def _extract_causal_links(self, user_input: str, response: str):
        """Извлечение каузальных связей"""
        combined = f"{user_input} {response}".lower()
        because_pattern = r'(.+?)\s+(?:потому что|так как|из-за)\s+(.+?)(?:\.|,|$)'
        matches = re.findall(because_pattern, combined)

        for effect, cause in matches:
            effect = effect.strip()[:100]
            cause = cause.strip()[:100]
            if len(effect) > 10 and len(cause) > 10:
                self.causal_reasoning.add_causal_link(cause, effect, confidence=0.6)

    def get_status(self) -> Dict[str, Any]:
        """Получение полного статуса системы"""
        memory_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        cache_stats = self.llm.get_cache_stats()
        weights_stats = self.weighted_module.get_statistics()

        return {
            'identity': {
                'user_id': self.user_id,
                'version': 'v33.0 (Honest Architecture)',
                'age': self._get_age_string(),
                'uptime': self._format_time_ago(self.birth_time),
            },
            'honest_metrics_enabled': CONFIG.honest_metrics,
            'consistency_check_enabled': CONFIG.check_consistency,
            'response_validation_enabled': CONFIG.validate_responses,
            'memory': memory_stats,
            'concepts': weights_stats,
            'metacognition': {
                'avg_uncertainty': float(
                    np.mean(list(self.metacognition.uncertainty_log))) if self.metacognition.uncertainty_log else 0.0,
                'avg_confidence': float(
                    np.mean(list(self.metacognition.confidence_log))) if self.metacognition.confidence_log else 0.0,
                'questions_asked': len(self.metacognition.question_history),
                'current_strategy': self.metacognition.current_strategy,
            },
            'performance': metrics,
            'llm_cache': cache_stats,
            'goals': {
                'active': len(self.goal_planning.active_goals),
                'total': len(self.goal_planning.goals),
            },
            'causal_knowledge': {
                'causal_links': len(self.causal_reasoning.causal_graph),
            },
            'activity': {
                'total_interactions': self.metrics.interaction_count,
                'last_interaction': self._format_time_ago(self.last_interaction),
            }
        }

    def _get_age_string(self) -> str:
        """Форматирование возраста"""
        age = time.time() - self.birth_time
        days = int(age / 86400)
        hours = int((age % 86400) / 3600)
        minutes = int((age % 3600) / 60)
        if days > 0:
            return f"{days}д {hours}ч"
        if hours > 0:
            return f"{hours}ч {minutes}м"
        return f"{minutes}м"

    def _format_time_ago(self, timestamp: float) -> str:
        """Форматирование времени"""
        if timestamp == 0:
            return "никогда"
        delta = time.time() - timestamp
        if delta < 60:
            return "только что"
        if delta < 3600:
            return f"{int(delta / 60)}м назад"
        if delta < 86400:
            return f"{int(delta / 3600)}ч назад"
        return f"{int(delta / 86400)}д назад"


class EnhancedAGIBot:
    """🔧 УЛУЧШЕННЫЙ Telegram бот для Enhanced AGI Brain v33.0"""

    def __init__(self):
        self.llm: Optional[EnhancedSubconsciousLLM] = None
        self.brains: Dict[str, EnhancedAutonomousAGI] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        self.llm = EnhancedSubconsciousLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.llm.connect()

        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        commands = [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('memory', self._cmd_memory),
            ('concepts', self._cmd_concepts),  # 🔧 НОВАЯ команда
            ('emotion', self._cmd_emotion),
            ('metrics', self._cmd_metrics),
            ('goals', self._cmd_goals),
            ('reset', self._cmd_reset),
            ('help', self._cmd_help),
        ]
        for cmd, handler in commands:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Enhanced AGI Bot v33.0 initialized")

    async def _get_or_create_brain(self, user_id: str) -> EnhancedAutonomousAGI:
        if user_id not in self.brains:
            brain = EnhancedAutonomousAGI(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
            logger.info(f"🆕 Enhanced AGI Brain v33.0 created for {user_id}")
        return self.brains[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or not update.message:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text
        logger.info(f"💬 [{user_id}] {user_input[:100]}")

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            brain = await self._get_or_create_brain(user_id)
            response, autonomous_question, metadata = await brain.process_interaction(user_input)

            await update.message.reply_text(
                response,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

            if autonomous_question:
                await asyncio.sleep(0.8)
                await update.message.reply_text(
                    f"🤔 <i>{autonomous_question}</i>",
                    parse_mode='HTML'
                )
        except Exception as e:
            logger.exception(f"❌ Error processing from {user_id}")
            await update.message.reply_text(
                "⚠️ Произошла ошибка. Попробуйте /help или /reset"
            )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        mem_stats = brain.memory.get_statistics()

        message = f"""🧠 <b>ENHANCED AGI BRAIN v33.0</b>
Привет, {update.effective_user.first_name}! 👋

Я — улучшенная версия AGI с честной архитектурой:

✅ <b>КЛЮЧЕВЫЕ УЛУЧШЕНИЯ v33.0:</b>
- Честные метрики (вычисляются из реальных данных)
- Проверка консистентности (не противоречу себе)
- Упрощенная система весов вместо фейковой нейросети
- Активное целеполагание на основе диалога
- Извлечение и отслеживание интересов
- Валидация ответов (запрет выдуманных чисел)

📚 <b>Многоуровневая память</b>
- Working: {mem_stats['working_memory']} элементов
- Long-term: {mem_stats['long_term']} воспоминаний
- Episodes: {mem_stats['episodic']} событий
- Avg quality: {mem_stats.get('avg_quality_ltm', 0):.2f}

🎭 <b>Эмоциональный интеллект</b>
💭 <b>Улучшенная когнитивность</b>
🎯 <b>Автоматическое целеполагание</b>

⚡ <b>Возраст:</b> {brain._get_age_string()}
📌 Команды: /help
"""
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        status = brain.get_status()

        message = f"""🧠 <b>STATUS v33.0</b>

<b>🆔 Идентичность</b>
- Version: {status['identity']['version']}
- Возраст: {status['identity']['age']}

<b>✅ Режимы</b>
- Честные метрики: {'✅' if status['honest_metrics_enabled'] else '❌'}
- Проверка консистентности: {'✅' if status['consistency_check_enabled'] else '❌'}
- Валидация ответов: {'✅' if status['response_validation_enabled'] else '❌'}

<b>📚 Многоуровневая память</b>
- Working: {status['memory']['working_memory']}
- Short-term: {status['memory']['short_term']}
- Long-term: {status['memory']['long_term']}
- Episodes: {status['memory']['episodic']}
- Avg quality LTM: {status['memory'].get('avg_quality_ltm', 0):.2f}

<b>🧠 Концепты и интересы</b>
- Всего концептов: {status['concepts']['total_concepts']}
- Средний вес: {status['concepts']['avg_weight']:.2f}

<b>🤔 Метакогниция</b>
- Avg uncertainty: {status['metacognition']['avg_uncertainty']:.2f}
- Avg confidence: {status['metacognition']['avg_confidence']:.2f}
- Questions asked: {status['metacognition']['questions_asked']}

<b>⚡ Производительность</b>
- Interactions: {status['performance']['interactions']}
- Avg confidence: {status['performance']['avg_confidence']:.2f}
- Best strategy: {status['performance']['best_strategy']}

<b>💾 LLM Cache</b>
- Hit rate: {status['llm_cache']['hit_rate']:.1%}

<b>🎯 Цели</b>
- Активных: {status['goals']['active']}
- Всего: {status['goals']['total']}"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.memory.get_statistics()

        working = list(brain.memory.working_memory)[-3:]
        working_text = "\n".join([f"  • {item.content[:60]}..." for item in working]) if working else "  Пусто"

        episodes = brain.memory.get_recent_episodes(n=3)
        episodes_text = "\n".join([
            f"  • {ep.context[:60]}... (quality: {ep.quality_score:.2f})"
            for ep in episodes
        ]) if episodes else "  Нет"

        message = f"""📚 <b>МНОГОУРОВНЕВАЯ ПАМЯТЬ v33.0</b>

<b>📊 Статистика</b>
- Working: {stats['working_memory']}/{CONFIG.working_memory_size}
- Short-term: {stats['short_term']}
- Long-term: {stats['long_term']}
- Episodic: {stats['episodic']}
- Avg quality LTM: {stats.get('avg_quality_ltm', 0):.2f}
- Avg quality episodic: {stats.get('avg_quality_episodic', 0):.2f}

<b>🧠 Working Memory (последние)</b>
{working_text}

<b>📖 Недавние эпизоды</b>
{episodes_text}

<b>💡 Особенности:</b>
- Автоматическая консолидация
- Приоритизация по качеству
- Периодическое улучшение эпизодов
- Отслеживание источника (user/assistant)
- Семантический поиск"""

        await update.message.reply_text(message, parse_mode='HTML')

    # 🔧 НОВАЯ команда
    async def _cmd_concepts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.weighted_module.get_statistics()
        top_concepts = stats['top_concepts']

        concepts_text = "\n".join([
            f"  • <b>{concept}</b>: {weight:.2f}"
            for concept, weight in top_concepts
        ]) if top_concepts else "  Нет данных"

        message = f"""🧠 <b>КОНЦЕПТЫ И ИНТЕРЕСЫ v33.0</b>

<b>📊 Статистика</b>
- Всего концептов: {stats['total_concepts']}
- Средний вес: {stats['avg_weight']:.2f}

<b>🔝 Топ-концепты</b>
{concepts_text}

<b>💡 Как это работает:</b>
- Система отслеживает ключевые слова и темы
- Вес увеличивается при упоминании
- Используется для автоматического целеполагания
- Затухает со временем при неиспользовании"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_emotion(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        recent_emotions = list(brain.emotional_intelligence.emotion_history)[-5:]

        if not recent_emotions:
            await update.message.reply_text("📊 Пока нет истории эмоций")
            return

        message = f"""🎭 <b>ЭМОЦИОНАЛЬНЫЙ ИНТЕЛЛЕКТ</b>

<b>📊 Недавние эмоции:</b>
"""
        emotion_emoji = {
            'JOY': '😊',
            'SADNESS': '😢',
            'ANGER': '😠',
            'FEAR': '😨',
            'SURPRISE': '😮',
            'CURIOSITY': '🤔',
            'EXCITEMENT': '🤩',
            'NEUTRAL': '😐',
        }

        for emotion in reversed(recent_emotions):
            emoji = emotion_emoji.get(emotion.dominant_emotion.name, '😐')
            valence_str = "+" if emotion.valence > 0 else ""
            message += f"\n{emoji} <b>{emotion.dominant_emotion.name}</b>\n"
            message += f"  Valence: {valence_str}{emotion.valence:.2f} | "
            message += f"Arousal: {emotion.arousal:.2f}\n"

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        metrics = brain.metrics.get_metrics_summary()
        cache_stats = brain.llm.get_cache_stats()

        message = f"""📊 <b>МЕТРИКИ v33.0</b>

<b>⚡ Производительность</b>
- Взаимодействий: {metrics['interactions']}
- Error rate: {metrics['error_rate']:.1%}
- Avg response: {metrics['avg_response_time']:.2f}s
- Avg confidence: {metrics['avg_confidence']:.2f}

<b>🎯 Лучшая стратегия</b>
- {metrics['best_strategy']}

<b>💾 LLM Cache</b>
- Hit rate: {cache_stats['hit_rate']:.1%}
- Hits: {cache_stats['cache_hits']}
- Size: {cache_stats['cache_size']}

<i>✅ Все метрики РЕАЛЬНЫЕ и проверяемые!</i>"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_goals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        active_goals = brain.goal_planning.get_active_goals()

        if not active_goals:
            message = """🎯 <b>ЦЕЛЕПОЛАГАНИЕ</b>
Нет активных целей."""
        else:
            message = f"""🎯 <b>АКТИВНЫЕ ЦЕЛИ ({len(active_goals)})</b>
"""
            for goal in active_goals[:5]:
                progress_bar = '█' * int(goal.progress * 10) + '░' * (10 - int(goal.progress * 10))
                message += f"<b>{goal.description[:50]}</b>\n"
                message += f"  [{progress_bar}] {goal.progress:.0%}\n"

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)

        if context.args and context.args[0].lower() == 'confirm':
            if user_id in self.brains:
                await self.brains[user_id].stop()
                del self.brains[user_id]

            user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"

            import shutil
            if user_dir.exists():
                shutil.rmtree(user_dir)

            brain = await self._get_or_create_brain(user_id)

            await update.message.reply_text(
                f"✅ <b>Полный сброс выполнен!</b>\n"
                f"Создано новое сознание v33.0:\n"
                f"• Версия: v33.0 (Honest Architecture)\n"
                f"• Честные метрики включены\n"
                f"• Проверка консистентности активна",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                "⚠️ <b>ВНИМАНИЕ!</b>\n"
                "Это удалит всю память и состояние.\n"
                "Подтверждение: <code>/reset confirm</code>",
                parse_mode='HTML'
            )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = f"""🧠 <b>ENHANCED AGI BRAIN v33.0</b>

<b>✅ КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:</b>

<b>❌ БЫЛО (v32.0):</b>
- Фейковая нейросеть с произвольными метриками
- Противоречия в ответах
- Имитация чисел без обоснования
- Нет проверки консистентности

<b>✅ СТАЛО (v33.0):</b>
- Честные метрики из реальных данных
- Проверка консистентности ответов
- Система весов для отслеживания интересов
- Активное целеполагание
- Валидация ответов
- Детальное логирование

<b>🎯 ПРИНЦИПЫ v33.0:</b>
1. Честность > Имитация
2. Проверяемость > Сложность
3. Консистентность > Креативность
4. Объяснимость > Загадочность

<b>📌 КОМАНДЫ:</b>
- /start — приветствие
- /status — полный статус системы
- /memory — уровни памяти + качество
- /concepts — отслеживаемые концепты
- /emotion — эмоциональная история
- /metrics — производительность
- /goals — активные цели
- /reset — сброс
- /help — эта справка

<b>💡 ОСОБЕННОСТЬ v33.0:</b>
Я показываю ТОЛЬКО те метрики, которые могу вычислить из реальных данных.
Система автоматически проверяет мои ответы на противоречия.

<b>🔬 КАК ПРОВЕРИТЬ:</b>
Спроси: "Какая у тебя метрика perception?"
Я покажу реальное значение, вычисленное из твоего сообщения,
и объясню, откуда оно взялось."""

        await update.message.reply_text(message, parse_mode='HTML')

    async def start_polling(self):
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Enhanced AGI Bot v33.0 started")

    async def shutdown(self):
        logger.info("🛑 Shutting down bot...")

        for user_id, brain in self.brains.items():
            try:
                await brain.stop()
            except Exception as e:
                logger.error(f"⚠️ Error stopping brain {user_id}: {e}")

        if self.llm:
            await self.llm.close()

        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Bot stopped")


async def main():
    """Главная функция запуска v33.0"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🧠 ENHANCED AGI BRAIN v33.0 — ЧЕСТНАЯ АРХИТЕКТУРА           ║
║     С реальными метриками и проверкой консистентности        ║
╚══════════════════════════════════════════════════════════════╝

✅ КЛЮЧЕВЫЕ УЛУЧШЕНИЯ v33.0:
- Честные метрики (вычисляются из реальных данных)
- Проверка консистентности (система не противоречит себе)
- Упрощенная система весов вместо фейковой нейросети
- Активное целеполагание на основе диалога
- Извлечение и отслеживание интересов пользователя
- Валидация ответов (запрет выдуманных чисел)
- Детальное логирование для отладки

🎯 ПРИНЦИПЫ:
1. Честность > Имитация
2. Проверяемость > Сложность
3. Консистентность > Креативность
4. Объяснимость > Загадочность
""")

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = EnhancedAGIBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 ENHANCED AGI v33.0 АКТИВЕН")
        logger.info("✅ Честные метрики включены")
        logger.info("✅ Проверка консистентности активна")
        logger.info("✅ Валидация ответов включена")
        logger.info("🛑 Ctrl+C для остановки")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n👋 Получен сигнал остановки")
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка: {e}", exc_info=True)
        return 1
    finally:
        await bot.shutdown()
        logger.info("👋 До встречи!")
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)