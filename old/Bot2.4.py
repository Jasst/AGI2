#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 TEMPORAL COGNITIVE BRAIN v32.0 — РАСШИРЕННАЯ ОПТИМИЗАЦИЯ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 НОВЫЕ ВОЗМОЖНОСТИ v32.0:

✨ ДИНАМИЧЕСКОЕ РАСШИРЕНИЕ НЕЙРОНОВ:
• Автоматический рост при плато метрик
• Адаптивные пороги на основе истории
• Предотвращение переобучения

🎨 ВАРИАТИВНОСТЬ АВТОНОМНЫХ МЫСЛЕЙ:
• Динамическая температура (0.7-1.1)
• Diversity penalty для разнообразия
• Топ-K и Top-P сэмплирование

🔗 КРОСС-МОДУЛЬНАЯ СИНЕРГИЯ:
• Взаимное усиление модулей
• Синергетические бонусы
• Каскадная активация

💾 УЛУЧШЕННАЯ КОНСОЛИДАЦИЯ ПАМЯТИ:
• Семантическое слияние похожих эпизодов
• Приоритизация важных воспоминаний
• Архивация старых данных

🆕 ДОПОЛНИТЕЛЬНЫЕ ОПТИМИЗАЦИИ:
• Adaptive Learning Rate с планировщиком
• Gradient clipping для стабильности
• Batch normalization для нейронов
• Dropout scheduling
• Периодическая дефрагментация памяти
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
from scipy.special import softmax

from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)

load_dotenv()


@dataclass
class EnhancedConfig:
    """Расширенная конфигурация v32.0"""
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # Честность метрик
    honest_metrics: bool = True
    show_uncertainty: bool = True

    # Нейросеть
    initial_neurons: int = 150
    max_neurons: int = 20000
    neurogenesis_threshold: float = 0.75
    pruning_threshold: float = 0.015
    learning_rate: float = 0.12
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.3
    attention_heads: int = 4
    dropout_rate: float = 0.1
    dropout_rate_min: float = 0.05
    dropout_rate_max: float = 0.3

    # 🆕 Динамическое расширение
    plateau_detection_window: int = 10
    plateau_threshold: float = 0.02
    expansion_factor: float = 1.15
    max_expansion_per_cycle: int = 50

    # 🆕 Кросс-модульная синергия
    synergy_enabled: bool = True
    synergy_threshold: float = 0.7
    synergy_bonus: float = 0.2

    # 🆕 Вариативность мыслей
    thought_temperature_min: float = 0.7
    thought_temperature_max: float = 1.1
    thought_diversity_penalty: float = 0.3
    thought_top_k: int = 50
    thought_top_p: float = 0.9

    # Память
    working_memory_size: int = 7
    short_term_size: int = 100
    short_term_decay: float = 0.95
    long_term_size: int = 10000
    consolidation_threshold: float = 0.7
    episodic_context_window: int = 5

    # 🆕 Улучшенная консолидация
    semantic_merge_threshold: float = 0.85
    memory_archive_days: int = 30
    consolidation_batch_size: int = 20

    # Метакогниция
    curiosity_threshold: float = 0.55
    question_cooldown: int = 90
    max_autonomous_questions: int = 5
    reasoning_depth: int = 3
    analogy_threshold: float = 0.6
    confidence_calibration_samples: int = 50
    error_correction_threshold: float = 0.3

    # Эмоции
    emotion_tracking: bool = True
    empathy_weight: float = 0.3

    # Вспомогательное
    embedding_dim: int = 512
    semantic_cache_size: int = 2000

    # Интервалы (секунды)
    spontaneous_thought_interval: int = 150
    reflection_interval: int = 600
    consolidation_interval: int = 900
    save_interval: int = 240
    neural_optimization_interval: int = 450
    metrics_update_interval: int = 180
    synergy_check_interval: int = 300
    memory_defrag_interval: int = 1800

    base_dir: Path = Path(os.getenv('BASE_DIR', 'temporal_brain_v32'))

    def __post_init__(self):
        for subdir in ['memory', 'neural_nets', 'knowledge', 'cache', 'logs',
                       'backups', 'episodic', 'analytics', 'goals', 'archives']:
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
    logger = logging.getLogger('Enhanced_AGI_v32')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'agi_v32_{datetime.now():%Y%m%d}.log'
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
    valence: float
    arousal: float
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
        """Анализ эмоций в тексте"""
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
    """Элемент памяти"""
    content: str
    timestamp: float
    importance: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def access(self):
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class Episode:
    """Эпизод памяти"""
    id: str
    messages: List[Dict[str, str]]
    context: str
    timestamp: float
    emotional_state: Optional[EmotionalState] = None
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'messages': self.messages,
            'context': self.context,
            'timestamp': self.timestamp,
            'importance': self.importance,
        }


class MultiLevelMemory:
    """🆕 Улучшенная многоуровневая система памяти с консолидацией"""

    def __init__(self, embedding_func):
        self.embedding_func = embedding_func

        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)
        self.short_term_memory: List[MemoryItem] = []
        self.long_term_memory: Dict[str, MemoryItem] = {}
        self.episodic_memory: Dict[str, Episode] = {}
        self.archived_memory: Dict[str, Episode] = {}
        self.current_episode_buffer: List[Dict] = []
        self.semantic_memory: Dict[str, MemoryItem] = {}

        self.message_history: List[MessageAttribution] = []

        self.last_consolidation = time.time()
        self.last_defragmentation = time.time()

        self.consolidation_stats = {
            'merges': 0,
            'archives': 0,
            'promotions': 0,
        }

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
        """🆕 Улучшенная консолидация с приоритизацией"""
        candidates = [
            item for item in self.short_term_memory
            if item.importance >= CONFIG.consolidation_threshold
        ]

        candidates.sort(
            key=lambda x: (x.importance * 0.7 + (x.access_count / 10) * 0.3),
            reverse=True
        )

        promoted = 0
        for item in candidates[:CONFIG.consolidation_batch_size]:
            memory_id = hashlib.sha256(item.content.encode()).hexdigest()[:16]

            if memory_id not in self.long_term_memory:
                self.long_term_memory[memory_id] = item
                promoted += 1
                logger.debug(f"💾 Consolidation: {item.content[:50]}...")

        if promoted > 0:
            self.consolidation_stats['promotions'] += promoted
            logger.info(f"📊 Promoted {promoted} memories to LTM")

        self.last_consolidation = time.time()

    def add_episode(self, messages: List[Dict], context: str,
                    emotion: Optional[EmotionalState] = None):
        episode_id = f"ep_{int(time.time() * 1000)}"

        episode_text = f"{context} {' '.join([m.get('content', '') for m in messages])}"
        episode_embedding = self.embedding_func(episode_text)

        episode = Episode(
            id=episode_id,
            messages=messages,
            context=context,
            timestamp=time.time(),
            emotional_state=emotion,
            importance=0.5,
            embedding=episode_embedding
        )

        self.episodic_memory[episode_id] = episode

        if len(self.episodic_memory) > 500:
            self._archive_old_episodes()

    def _archive_old_episodes(self):
        """🆕 Архивация старых эпизодов"""
        current_time = time.time()
        archive_threshold = current_time - (CONFIG.memory_archive_days * 86400)

        to_archive = []
        for ep_id, episode in self.episodic_memory.items():
            if episode.timestamp < archive_threshold and episode.importance < 0.6:
                to_archive.append(ep_id)

        for ep_id in to_archive:
            self.archived_memory[ep_id] = self.episodic_memory.pop(ep_id)

        if to_archive:
            self.consolidation_stats['archives'] += len(to_archive)
            logger.info(f"📦 Archived {len(to_archive)} old episodes")

    def semantic_merge_episodes(self):
        """🆕 Семантическое слияние похожих эпизодов"""
        if len(self.episodic_memory) < 10:
            return

        episodes = list(self.episodic_memory.values())
        merged_count = 0

        to_remove = set()
        for i, ep1 in enumerate(episodes):
            if ep1.id in to_remove or ep1.embedding is None:
                continue

            for ep2 in episodes[i + 1:]:
                if ep2.id in to_remove or ep2.embedding is None:
                    continue

                similarity = np.dot(ep1.embedding, ep2.embedding)

                if similarity > CONFIG.semantic_merge_threshold:
                    merged_messages = ep1.messages + ep2.messages
                    merged_context = f"{ep1.context}; {ep2.context}"
                    merged_importance = max(ep1.importance, ep2.importance)

                    ep1.messages = merged_messages
                    ep1.context = merged_context
                    ep1.importance = merged_importance

                    to_remove.add(ep2.id)
                    merged_count += 1

        for ep_id in to_remove:
            del self.episodic_memory[ep_id]

        if merged_count > 0:
            self.consolidation_stats['merges'] += merged_count
            logger.info(f"🔗 Merged {merged_count} similar episodes")

    def defragment_memory(self):
        """🆕 Дефрагментация памяти для оптимизации"""
        unique_contents = {}
        duplicates = []

        for mem_id, item in self.long_term_memory.items():
            content_hash = hashlib.md5(item.content.encode()).hexdigest()

            if content_hash in unique_contents:
                existing_id = unique_contents[content_hash]
                if item.importance > self.long_term_memory[existing_id].importance:
                    duplicates.append(existing_id)
                    unique_contents[content_hash] = mem_id
                else:
                    duplicates.append(mem_id)
            else:
                unique_contents[content_hash] = mem_id

        for dup_id in duplicates:
            del self.long_term_memory[dup_id]

        if duplicates:
            logger.info(f"🗑️ Removed {len(duplicates)} duplicate memories")

        self.last_defragmentation = time.time()

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = self.embedding_func(query)
        results = []

        for mem_id, item in self.long_term_memory.items():
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                results.append((item.content, float(similarity), 'LTM'))
                item.access()

        for item in self.short_term_memory:
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                results.append((item.content, float(similarity), 'STM'))
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
        return {
            'working_memory': len(self.working_memory),
            'short_term': len(self.short_term_memory),
            'long_term': len(self.long_term_memory),
            'episodic': len(self.episodic_memory),
            'archived': len(self.archived_memory),
            'semantic': len(self.semantic_memory),
            'consolidation_stats': self.consolidation_stats,
        }


@dataclass
class EnhancedNeuron:
    """Улучшенный нейрон с batch normalization"""
    id: str
    layer: int
    module: str
    activation: float = 0.0
    bias: float = 0.0
    neuron_type: str = "general"
    created_at: float = field(default_factory=time.time)
    activation_count: int = 0
    specialization: Optional[str] = None
    importance_score: float = 0.5

    running_mean: float = 0.0
    running_var: float = 1.0
    bn_momentum: float = 0.1

    def activate(self, input_sum: float, dropout: float = 0.0, use_bn: bool = True) -> float:
        """🆕 Активация с batch normalization и dropout"""
        if random.random() < dropout:
            self.activation = 0.0
        else:
            if use_bn:
                normalized = (input_sum - self.running_mean) / (np.sqrt(self.running_var) + 1e-5)
                input_sum = normalized

                self.running_mean = (1 - self.bn_momentum) * self.running_mean + self.bn_momentum * input_sum
                self.running_var = (1 - self.bn_momentum) * self.running_var + self.bn_momentum * (input_sum ** 2)

            self.activation = 1 / (1 + math.exp(-input_sum - self.bias))

        self.activation_count += 1
        return self.activation

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'layer': self.layer,
            'module': self.module,
            'bias': self.bias,
            'neuron_type': self.neuron_type,
            'created_at': self.created_at,
            'activation_count': self.activation_count,
            'specialization': self.specialization,
            'importance_score': self.importance_score,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedNeuron':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class EnhancedSynapse:
    """Улучшенный синапс с gradient clipping"""
    source_id: str
    target_id: str
    weight: float = 0.0
    strength: float = 1.0
    plasticity: float = 1.0
    activation_count: int = 0
    created_at: float = field(default_factory=time.time)
    attention_weight: float = 1.0

    gradient_clip_value: float = 1.0

    def hebbian_update(self, source_activation: float, target_activation: float,
                       learning_rate: float, attention: float = 1.0):
        """🆕 Hebbian learning с gradient clipping"""
        delta = learning_rate * source_activation * target_activation * attention

        delta = np.clip(delta, -self.gradient_clip_value, self.gradient_clip_value)

        self.weight += delta * self.plasticity
        self.strength = 0.9 * self.strength + 0.1 * abs(self.weight)
        self.activation_count += 1
        self.plasticity *= 0.9995

    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'weight': self.weight,
            'strength': self.strength,
            'plasticity': self.plasticity,
            'activation_count': self.activation_count,
            'created_at': self.created_at,
            'attention_weight': self.attention_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedSynapse':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class AdaptiveLearningRateScheduler:
    """🆕 Адаптивный планировщик learning rate"""

    def __init__(self, initial_lr: float, min_lr: float, max_lr: float):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.performance_history: deque = deque(maxlen=20)

    def step(self, performance: float) -> float:
        """Обновление learning rate на основе производительности"""
        self.performance_history.append(performance)

        if len(self.performance_history) < 5:
            return self.current_lr

        recent_perf = np.mean(list(self.performance_history)[-5:])
        older_perf = np.mean(list(self.performance_history)[-10:-5]) if len(
            self.performance_history) >= 10 else recent_perf

        trend = recent_perf - older_perf

        if trend > 0.05:
            self.current_lr = min(self.current_lr * 1.05, self.max_lr)
        elif trend < -0.05:
            self.current_lr = max(self.current_lr * 0.95, self.min_lr)

        return self.current_lr

    def get_lr(self) -> float:
        return self.current_lr


class DropoutScheduler:
    """🆕 Планировщик dropout rate"""

    def __init__(self, initial_dropout: float, min_dropout: float, max_dropout: float):
        self.current_dropout = initial_dropout
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.step_count = 0

    def step(self, overfitting_detected: bool = False):
        """Обновление dropout"""
        self.step_count += 1

        if overfitting_detected:
            self.current_dropout = min(self.current_dropout * 1.1, self.max_dropout)
        else:
            self.current_dropout = max(self.current_dropout * 0.999, self.min_dropout)

    def get_dropout(self) -> float:
        return self.current_dropout


class EnhancedNeuralNetwork:
    """🆕 Улучшенная нейросеть с динамическим расширением и синергией"""

    def __init__(self, initial_neurons: int = 150):
        self.neurons: Dict[str, EnhancedNeuron] = {}
        self.synapses: Dict[Tuple[str, str], EnhancedSynapse] = {}
        self.layers: Dict[int, Set[str]] = defaultdict(set)
        self.modules: Dict[str, Set[str]] = defaultdict(set)

        self.attention_heads = CONFIG.attention_heads

        self.total_activations = 0
        self.neurogenesis_events = 0
        self.pruning_events = 0
        self.meta_learning_score = 0.5

        self.last_real_metrics: Dict[str, float] = {}
        self.metrics_timestamp: float = 0.0

        self.performance_history: deque = deque(maxlen=CONFIG.plateau_detection_window)
        self.expansion_count = 0

        self.synergy_matrix: Dict[Tuple[str, str], float] = {}
        self.last_synergy_check = time.time()

        self.lr_scheduler = AdaptiveLearningRateScheduler(
            CONFIG.learning_rate,
            CONFIG.learning_rate_min,
            CONFIG.learning_rate_max
        )
        self.dropout_scheduler = DropoutScheduler(
            CONFIG.dropout_rate,
            CONFIG.dropout_rate_min,
            CONFIG.dropout_rate_max
        )

        self._initialize_modular_network(initial_neurons)
        self._initialize_synergy_matrix()

        logger.info(f"🧬 Enhanced Neural Network v32: {len(self.neurons)} neurons")

    def _initialize_modular_network(self, n: int):
        """Инициализация модульной архитектуры"""
        modules_config = {
            'perception': n // 5,
            'reasoning': n // 5,
            'memory': n // 5,
            'action': n // 5,
            'meta': n // 5,
        }

        neuron_counter = 0

        for module_name, module_size in modules_config.items():
            for layer in range(4):
                neurons_in_layer = module_size // 4

                for i in range(neurons_in_layer):
                    neuron_id = f"{module_name}_L{layer}_{neuron_counter}"

                    neuron = EnhancedNeuron(
                        id=neuron_id,
                        layer=layer,
                        module=module_name,
                        neuron_type=self._get_neuron_type(layer),
                        bias=random.gauss(0, 0.1)
                    )

                    self.neurons[neuron_id] = neuron
                    self.layers[layer].add(neuron_id)
                    self.modules[module_name].add(neuron_id)
                    neuron_counter += 1

        for module_name, module_neurons in self.modules.items():
            self._connect_within_module(module_neurons)

        self._create_inter_module_connections()

    def _initialize_synergy_matrix(self):
        """🆕 Инициализация матрицы синергии"""
        module_pairs = [
            ('perception', 'reasoning'),
            ('reasoning', 'memory'),
            ('reasoning', 'action'),
            ('memory', 'reasoning'),
            ('meta', 'reasoning'),
            ('perception', 'memory'),
            ('action', 'meta'),
        ]

        for m1, m2 in module_pairs:
            self.synergy_matrix[(m1, m2)] = 0.0
            self.synergy_matrix[(m2, m1)] = 0.0

    def _get_neuron_type(self, layer: int) -> str:
        types = {0: "sensory", 1: "hidden", 2: "hidden", 3: "motor"}
        return types.get(layer, "general")

    def _connect_within_module(self, module_neurons: Set[str]):
        neurons_by_layer = defaultdict(list)

        for nid in module_neurons:
            layer = self.neurons[nid].layer
            neurons_by_layer[layer].append(nid)

        for layer in range(3):
            source_neurons = neurons_by_layer[layer]
            target_neurons = neurons_by_layer[layer + 1]

            for source_id in source_neurons:
                num_targets = max(1, int(len(target_neurons) * random.uniform(0.3, 0.5)))
                targets = random.sample(target_neurons, min(num_targets, len(target_neurons)))

                for target_id in targets:
                    synapse = EnhancedSynapse(
                        source_id=source_id,
                        target_id=target_id,
                        weight=random.gauss(0, 0.5)
                    )
                    self.synapses[(source_id, target_id)] = synapse

    def _create_inter_module_connections(self):
        module_pairs = [
            ('perception', 'reasoning'),
            ('reasoning', 'memory'),
            ('reasoning', 'action'),
            ('memory', 'reasoning'),
            ('meta', 'reasoning'),
        ]

        for source_module, target_module in module_pairs:
            source_neurons = list(self.modules[source_module])
            target_neurons = list(self.modules[target_module])

            num_connections = max(1, int(len(source_neurons) * 0.08))

            for _ in range(num_connections):
                source_id = random.choice(source_neurons)
                target_id = random.choice(target_neurons)

                if (source_id, target_id) not in self.synapses:
                    synapse = EnhancedSynapse(
                        source_id=source_id,
                        target_id=target_id,
                        weight=random.gauss(0, 0.3)
                    )
                    self.synapses[(source_id, target_id)] = synapse

    def detect_plateau(self) -> bool:
        """🆕 Обнаружение плато в метриках"""
        if len(self.performance_history) < CONFIG.plateau_detection_window:
            return False

        performances = list(self.performance_history)
        recent = np.mean(performances[-5:])
        older = np.mean(performances[-10:-5])

        improvement = recent - older

        return abs(improvement) < CONFIG.plateau_threshold

    def expand_neurons_dynamic(self):
        """🆕 Динамическое расширение нейронов"""
        if not self.detect_plateau():
            return 0

        if len(self.neurons) >= CONFIG.max_neurons:
            logger.warning("⚠️ Max neurons reached")
            return 0

        module_activations = {
            module: self.get_module_activation(module)
            for module in self.modules.keys()
        }

        neurons_added = 0
        for module, activation in sorted(module_activations.items(), key=lambda x: x[1], reverse=True):
            if activation > CONFIG.neurogenesis_threshold:
                num_new = int(len(self.modules[module]) * (CONFIG.expansion_factor - 1.0))
                num_new = min(num_new, CONFIG.max_expansion_per_cycle)

                if neurons_added + num_new > CONFIG.max_expansion_per_cycle:
                    num_new = CONFIG.max_expansion_per_cycle - neurons_added

                if num_new <= 0:
                    continue

                for _ in range(num_new):
                    layer = random.randint(1, 2)
                    neuron_id = f"{module}_L{layer}_exp_{int(time.time() * 1000)}_{random.randint(0, 9999)}"

                    neuron = EnhancedNeuron(
                        id=neuron_id,
                        layer=layer,
                        module=module,
                        neuron_type="general",
                        bias=random.gauss(0, 0.1)
                    )

                    self.neurons[neuron_id] = neuron
                    self.layers[layer].add(neuron_id)
                    self.modules[module].add(neuron_id)

                    self._create_connections_for_new_neuron(neuron_id, module, layer)

                    neurons_added += 1

                if neurons_added >= CONFIG.max_expansion_per_cycle:
                    break

        if neurons_added > 0:
            self.expansion_count += 1
            self.neurogenesis_events += neurons_added
            logger.info(f"🌱 Dynamic expansion: +{neurons_added} neurons")

        return neurons_added

    def _create_connections_for_new_neuron(self, neuron_id: str, module: str, layer: int):
        """Создание связей для нового нейрона"""
        module_neurons = [nid for nid in self.modules[module] if nid != neuron_id]

        if layer > 0:
            prev_layer_neurons = [nid for nid in self.layers[layer - 1]
                                  if self.neurons[nid].module == module]

            num_inputs = min(5, len(prev_layer_neurons))
            if prev_layer_neurons:
                sources = random.sample(prev_layer_neurons, num_inputs)
                for source_id in sources:
                    synapse = EnhancedSynapse(
                        source_id=source_id,
                        target_id=neuron_id,
                        weight=random.gauss(0, 0.3)
                    )
                    self.synapses[(source_id, neuron_id)] = synapse

        if layer < 3:
            next_layer_neurons = [nid for nid in self.layers[layer + 1]
                                  if self.neurons[nid].module == module]

            num_outputs = min(5, len(next_layer_neurons))
            if next_layer_neurons:
                targets = random.sample(next_layer_neurons, num_outputs)
                for target_id in targets:
                    synapse = EnhancedSynapse(
                        source_id=neuron_id,
                        target_id=target_id,
                        weight=random.gauss(0, 0.3)
                    )
                    self.synapses[(neuron_id, target_id)] = synapse

    def check_and_update_synergy(self):
        """🆕 Проверка и обновление синергии"""
        if not CONFIG.synergy_enabled:
            return

        current_time = time.time()
        if current_time - self.last_synergy_check < CONFIG.synergy_check_interval:
            return

        module_activations = {
            module: self.get_module_activation(module)
            for module in self.modules.keys()
        }

        for (m1, m2), _ in self.synergy_matrix.items():
            act1 = module_activations.get(m1, 0.0)
            act2 = module_activations.get(m2, 0.0)

            if act1 > CONFIG.synergy_threshold and act2 > CONFIG.synergy_threshold:
                self.synergy_matrix[(m1, m2)] = min(1.0, self.synergy_matrix[(m1, m2)] + 0.1)
                self._apply_synergy_bonus(m1, m2)
            else:
                self.synergy_matrix[(m1, m2)] = max(0.0, self.synergy_matrix[(m1, m2)] * 0.95)

        self.last_synergy_check = current_time

    def _apply_synergy_bonus(self, module1: str, module2: str):
        """🆕 Применение синергетического бонуса"""
        synergy_strength = self.synergy_matrix.get((module1, module2), 0.0)

        if synergy_strength < 0.3:
            return

        m1_neurons = self.modules[module1]
        m2_neurons = self.modules[module2]

        enhanced_count = 0
        for (source_id, target_id), synapse in self.synapses.items():
            source_module = self.neurons[source_id].module
            target_module = self.neurons[target_id].module

            if (source_module == module1 and target_module == module2) or \
                    (source_module == module2 and target_module == module1):
                synapse.attention_weight = min(2.0, synapse.attention_weight * (1.0 + CONFIG.synergy_bonus))
                enhanced_count += 1

        if enhanced_count > 0:
            logger.debug(f"🔗 Synergy: {module1} ↔ {module2} ({enhanced_count} synapses)")

    def forward_pass_with_attention(self, input_vector: np.ndarray,
                                    target_module: str = None) -> np.ndarray:
        """Прямой проход с attention и синергией"""
        for neuron in self.neurons.values():
            neuron.activation = 0.0

        input_neurons = [nid for nid in self.layers[0]
                         if target_module is None or self.neurons[nid].module == target_module]

        for i, neuron_id in enumerate(input_neurons):
            if i < len(input_vector):
                self.neurons[neuron_id].activation = input_vector[i]

        max_layer = max(self.layers.keys())
        current_dropout = self.dropout_scheduler.get_dropout()

        for layer in range(1, max_layer + 1):
            layer_neurons = self.layers[layer]

            for target_id in layer_neurons:
                inputs = []
                attention_weights = []

                for (source_id, tid), synapse in self.synapses.items():
                    if tid == target_id:
                        source = self.neurons[source_id]
                        inputs.append(source.activation * synapse.weight)
                        attention_weights.append(synapse.attention_weight * source.importance_score)

                if inputs:
                    if len(attention_weights) > 1:
                        attention_weights = softmax(attention_weights)
                        weighted_input = sum(i * a for i, a in zip(inputs, attention_weights))
                    else:
                        weighted_input = sum(inputs)

                    self.neurons[target_id].activate(weighted_input, current_dropout, use_bn=True)

        output_neurons = list(self.layers[max_layer])
        output = np.array([self.neurons[nid].activation for nid in output_neurons])

        self.total_activations += 1

        self._update_real_metrics()
        self.check_and_update_synergy()

        return output

    def _update_real_metrics(self):
        """Вычисление реальных метрик"""
        self.last_real_metrics = {
            'perception': self.get_module_activation('perception'),
            'reasoning': self.get_module_activation('reasoning'),
            'memory': self.get_module_activation('memory'),
            'action': self.get_module_activation('action'),
            'meta': self.get_module_activation('meta'),
            'total_activation': np.mean([n.activation for n in self.neurons.values()]),
            'active_neurons_ratio': len(self.get_active_neurons(0.3)) / len(self.neurons),
        }
        self.metrics_timestamp = time.time()

        avg_performance = np.mean([
            self.last_real_metrics['perception'],
            self.last_real_metrics['reasoning'],
            self.last_real_metrics['memory'],
        ])
        self.performance_history.append(avg_performance)

    def get_real_metrics(self) -> Dict[str, float]:
        if not CONFIG.honest_metrics:
            return {}

        age = time.time() - self.metrics_timestamp

        return {
            **self.last_real_metrics,
            'metrics_age_seconds': age,
        }

    def meta_learning_update(self, performance_score: float):
        """Meta-learning с адаптивным LR"""
        new_lr = self.lr_scheduler.step(performance_score)
        CONFIG.learning_rate = new_lr

        self.meta_learning_score = 0.9 * self.meta_learning_score + 0.1 * performance_score

        overfitting = False
        if len(self.performance_history) >= 10:
            recent_variance = np.var(list(self.performance_history)[-5:])
            if recent_variance > 0.05:
                overfitting = True

        self.dropout_scheduler.step(overfitting)

        logger.debug(f"🎓 Meta: LR={new_lr:.4f}, Dropout={self.dropout_scheduler.get_dropout():.3f}")

    def get_module_activation(self, module_name: str) -> float:
        module_neurons = self.modules[module_name]
        activations = [self.neurons[nid].activation for nid in module_neurons]
        return np.mean(activations) if activations else 0.0

    def get_active_neurons(self, threshold: float = 0.3) -> List[str]:
        return [nid for nid, neuron in self.neurons.items()
                if neuron.activation > threshold]

    def get_synergy_statistics(self) -> Dict[str, Any]:
        """Статистика синергии"""
        if not CONFIG.synergy_enabled:
            return {}

        active_synergies = {
            f"{m1}-{m2}": score
            for (m1, m2), score in self.synergy_matrix.items()
            if score > 0.3
        }

        return {
            'active_synergies': active_synergies,
            'total_synergy_score': sum(self.synergy_matrix.values()),
            'avg_synergy': np.mean(list(self.synergy_matrix.values())) if self.synergy_matrix else 0.0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        neuron_types = Counter(n.neuron_type for n in self.neurons.values())
        module_sizes = {m: len(neurons) for m, neurons in self.modules.items()}

        return {
            'neurons': {
                'total': len(self.neurons),
                'by_type': dict(neuron_types),
                'by_module': module_sizes,
                'by_layer': {l: len(neurons) for l, neurons in self.layers.items()},
            },
            'synapses': {
                'total': len(self.synapses),
                'avg_strength': np.mean([s.strength for s in self.synapses.values()]),
                'avg_plasticity': np.mean([s.plasticity for s in self.synapses.values()]),
            },
            'activity': {
                'total_activations': self.total_activations,
                'neurogenesis_events': self.neurogenesis_events,
                'pruning_events': self.pruning_events,
                'meta_learning_score': self.meta_learning_score,
                'expansion_count': self.expansion_count,
            },
            'modules': {
                m: self.get_module_activation(m) for m in self.modules.keys()
            },
            'adaptive': {
                'learning_rate': self.lr_scheduler.get_lr(),
                'dropout_rate': self.dropout_scheduler.get_dropout(),
                'plateau_detected': self.detect_plateau(),
            },
            'synergy': self.get_synergy_statistics(),
        }

    def save(self, path: Path):
        state = {
            'neurons': [n.to_dict() for n in self.neurons.values()],
            'synapses': [s.to_dict() for s in self.synapses.values()],
            'layers': {l: list(neurons) for l, neurons in self.layers.items()},
            'modules': {m: list(neurons) for m, neurons in self.modules.items()},
            'meta': {
                'total_activations': self.total_activations,
                'neurogenesis_events': self.neurogenesis_events,
                'pruning_events': self.pruning_events,
                'meta_learning_score': self.meta_learning_score,
                'expansion_count': self.expansion_count,
            },
            'synergy_matrix': {f"{m1}_{m2}": score for (m1, m2), score in self.synergy_matrix.items()},
            'lr_history': list(self.lr_scheduler.performance_history),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb', compresslevel=6) as f:
            pickle.dump(state, f)

        logger.info(f"💾 Neural network saved: {len(self.neurons)} neurons")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False

        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)

            self.neurons = {}
            for n_dict in state.get('neurons', []):
                neuron = EnhancedNeuron.from_dict(n_dict)
                self.neurons[neuron.id] = neuron

            self.synapses = {}
            for s_dict in state.get('synapses', []):
                synapse = EnhancedSynapse.from_dict(s_dict)
                self.synapses[(synapse.source_id, synapse.target_id)] = synapse

            self.layers = {int(l): set(neurons) for l, neurons in state.get('layers', {}).items()}
            self.modules = {m: set(neurons) for m, neurons in state.get('modules', {}).items()}

            meta = state.get('meta', {})
            self.total_activations = meta.get('total_activations', 0)
            self.neurogenesis_events = meta.get('neurogenesis_events', 0)
            self.pruning_events = meta.get('pruning_events', 0)
            self.meta_learning_score = meta.get('meta_learning_score', 0.5)
            self.expansion_count = meta.get('expansion_count', 0)

            synergy_data = state.get('synergy_matrix', {})
            for key, score in synergy_data.items():
                m1, m2 = key.split('_')
                self.synergy_matrix[(m1, m2)] = score

            logger.info(f"✅ Neural network loaded: {len(self.neurons)} neurons")
            return True

        except Exception as e:
            logger.error(f"⚠️ Error loading: {e}")
            return False


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
            reasons.append("недостаточно данных")

        if context.get('conflicting_info', False):
            uncertainty += 0.3
            reasons.append("противоречия")

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

        return f"""[Метакогниция] Сформулируй ОДИН точный вопрос (макс. 20 слов).

Контекст: {context}
Причины: {reasons_str}

Вопрос:"""


class AnalogicalReasoning:
    """Модуль аналогического мышления"""

    def __init__(self, memory: 'MultiLevelMemory'):
        self.memory = memory

    def find_analogies(self, current_situation: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        similar_memories = self.memory.search_semantic(current_situation, top_k=top_k * 2)
        analogies = []

        for memory_text, similarity, source in similar_memories:
            if similarity > CONFIG.analogy_threshold:
                pattern = "общий паттерн"
                analogies.append((memory_text, float(similarity), pattern))

        return analogies[:top_k]


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


@dataclass
class ReasoningStep:
    """Шаг рассуждения"""
    step_number: int
    content: str
    confidence: float
    reasoning_type: str


class MultiStepReasoning:
    """Модуль многошагового рассуждения"""

    def __init__(self, llm: 'EnhancedSubconsciousLLM', metacog: EnhancedMetacognition):
        self.llm = llm
        self.metacog = metacog

    async def reason(self, query: str, context: str, depth: int = 3,
                     strategy: str = 'deductive') -> List[ReasoningStep]:
        steps = []

        for i in range(depth):
            previous_steps = "\n".join([
                f"{s.step_number}. {s.content}" for s in steps
            ]) if steps else "Начало"

            prompt = f"""[Рассуждение {i + 1}]

Вопрос: {query}
Контекст: {context}
Предыдущие шаги: {previous_steps}

Следующий шаг (2-3 предложения):"""

            step_content = await self.llm.generate_raw(
                prompt, temperature=0.7, max_tokens=150
            )

            if not step_content:
                break

            step = ReasoningStep(
                step_number=i + 1,
                content=step_content.strip(),
                confidence=0.7,
                reasoning_type=strategy
            )

            steps.append(step)

        return steps


@dataclass
class Goal:
    """Цель"""
    id: str
    description: str
    status: str = "active"
    progress: float = 0.0


class GoalPlanning:
    """Модуль целеполагания"""

    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()


class PerformanceMetrics:
    """Метрики производительности"""

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

        return max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else "deductive"

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
    """🆕 Улучшенный LLM с вариативностью"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

        self.response_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        self.recent_outputs: deque = deque(maxlen=50)

    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("🔗 Enhanced LLM v32 connected")

    async def close(self):
        if self._session:
            await self._session.close()
            await asyncio.sleep(0.25)

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        content = f"{prompt}_{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _calculate_diversity_penalty(self, new_text: str) -> float:
        """🆕 Вычисление штрафа за повторяемость"""
        if not self.recent_outputs:
            return 0.0

        new_words = set(new_text.lower().split())

        max_overlap = 0.0
        for old_text in self.recent_outputs:
            old_words = set(old_text.lower().split())

            if not new_words or not old_words:
                continue

            overlap = len(new_words & old_words) / len(new_words)
            max_overlap = max(max_overlap, overlap)

        return max_overlap

    async def generate_raw(self, prompt: str, temperature: float = 0.75,
                           max_tokens: int = 300, timeout: float = 40,
                           use_cache: bool = True, max_retries: int = 2,
                           top_k: Optional[int] = None, top_p: Optional[float] = None) -> str:
        """🆕 Генерация с вариативностью"""
        if not self._session:
            await self.connect()

        if use_cache:
            cache_key = self._get_cache_key(prompt, temperature)
            if cache_key in self.response_cache:
                cached_response, cache_time = self.response_cache[cache_key]
                if time.time() - cache_time < 3600:
                    self.cache_hits += 1
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

                if top_k is not None:
                    payload["top_k"] = top_k
                if top_p is not None:
                    payload["top_p"] = top_p

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

                        if content:
                            diversity_penalty = self._calculate_diversity_penalty(content)

                            if diversity_penalty > 0.7:
                                logger.debug(f"⚠️ High diversity penalty: {diversity_penalty:.2f}")
                                if attempt < max_retries:
                                    temperature = min(1.2, temperature * 1.1)
                                    continue

                            self.recent_outputs.append(content)

                            if use_cache:
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
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)
                        continue

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                continue

            except Exception as e:
                logger.error(f"LLM error: {e}")
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
    """🚀 Главное автономное AGI-сознание v32.0"""

    def __init__(self, user_id: str, llm: EnhancedSubconsciousLLM):
        self.user_id = user_id
        self.llm = llm

        self.neural_net = EnhancedNeuralNetwork(CONFIG.initial_neurons)
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
        self.neural_path = CONFIG.base_dir / 'neural_nets' / f"{user_id}_v32.pkl.gz"

        self.birth_time = time.time()
        self.last_interaction = 0

        self._load_state()

        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False

        logger.info(f"🧠 Enhanced AGI v32 created for {user_id}")

    def _simple_embedding(self, text: str) -> np.ndarray:
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
        """Загрузка состояния"""
        if self.neural_path.exists():
            self.neural_net.load(self.neural_path)

        memory_file = self.user_dir / "memory_v32.pkl.gz"
        if memory_file.exists():
            try:
                with gzip.open(memory_file, 'rb') as f:
                    mem_state = pickle.load(f)

                self.memory.long_term_memory = {
                    mid: MemoryItem(**item_data)
                    for mid, item_data in mem_state.get('long_term', {}).items()
                }

                self.memory.episodic_memory = {
                    eid: Episode(**{k: v for k, v in ep_data.items() if k in Episode.__annotations__})
                    for eid, ep_data in mem_state.get('episodic', {}).items()
                }

                self.memory.archived_memory = {
                    eid: Episode(**{k: v for k, v in ep_data.items() if k in Episode.__annotations__})
                    for eid, ep_data in mem_state.get('archived', {}).items()
                }

                logger.info(f"✅ Memory loaded for {self.user_id}")

            except Exception as e:
                logger.error(f"⚠️ Error loading memory: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        try:
            self.neural_net.save(self.neural_path)
        except Exception as e:
            logger.error(f"⚠️ Error saving neural: {e}")

        memory_file = self.user_dir / "memory_v32.pkl.gz"
        try:
            mem_state = {
                'long_term': {
                    mid: {
                        'content': item.content,
                        'timestamp': item.timestamp,
                        'importance': item.importance,
                        'access_count': item.access_count,
                        'last_access': item.last_access,
                        'metadata': item.metadata,
                    }
                    for mid, item in self.memory.long_term_memory.items()
                },
                'episodic': {
                    eid: ep.to_dict()
                    for eid, ep in self.memory.episodic_memory.items()
                },
                'archived': {
                    eid: ep.to_dict()
                    for eid, ep in self.memory.archived_memory.items()
                },
            }

            with gzip.open(memory_file, 'wb', compresslevel=6) as f:
                pickle.dump(mem_state, f)

        except Exception as e:
            logger.error(f"⚠️ Error saving memory: {e}")

    async def start(self):
        """Запуск автономного существования"""
        if self._is_running:
            return

        self._is_running = True
        self._background_task = asyncio.create_task(self._autonomous_loop())
        logger.info(f"✨ AGI v32 started for {self.user_id}")

    async def stop(self):
        """Остановка"""
        if not self._is_running:
            return

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
        """🆕 Автономный цикл с новыми процессами"""
        timers = {
            'thought': time.time(),
            'reflection': time.time(),
            'consolidation': time.time(),
            'optimization': time.time(),
            'save': time.time(),
            'metrics': time.time(),
            'synergy': time.time(),
            'defrag': time.time(),
        }

        while self._is_running:
            try:
                now = time.time()

                if now - timers['thought'] > CONFIG.spontaneous_thought_interval:
                    await self._autonomous_thought_variative()
                    timers['thought'] = now

                if now - timers['reflection'] > CONFIG.reflection_interval:
                    await self._self_reflection()
                    timers['reflection'] = now

                if now - timers['consolidation'] > CONFIG.consolidation_interval:
                    self.memory.decay_short_term()
                    self.memory.consolidate_to_long_term()
                    self.memory.semantic_merge_episodes()
                    timers['consolidation'] = now

                if now - timers['optimization'] > CONFIG.neural_optimization_interval:
                    await self._optimize_neural_network_enhanced()
                    timers['optimization'] = now

                if now - timers['metrics'] > CONFIG.metrics_update_interval:
                    self._update_metrics()
                    timers['metrics'] = now

                if now - timers['synergy'] > CONFIG.synergy_check_interval:
                    self.neural_net.check_and_update_synergy()
                    timers['synergy'] = now

                if now - timers['defrag'] > CONFIG.memory_defrag_interval:
                    self.memory.defragment_memory()
                    timers['defrag'] = now

                if now - timers['save'] > CONFIG.save_interval:
                    self._save_state()
                    timers['save'] = now

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"⚠️ Error in loop: {e}")
                await asyncio.sleep(60)

    async def _autonomous_thought_variative(self):
        """🆕 Автономная мысль с вариативностью"""
        recent_episodes = self.memory.get_recent_episodes(n=3)

        if not recent_episodes:
            return

        context = "\n".join([f"• {ep.context[:60]}" for ep in recent_episodes])

        base_temp = random.uniform(CONFIG.thought_temperature_min, CONFIG.thought_temperature_max)

        recent_thoughts = [
            item.content for item in list(self.memory.working_memory)[-5:]
            if "💭" in item.content
        ]

        if len(recent_thoughts) >= 3:
            base_temp = min(1.2, base_temp * 1.2)

        prompt = f"""[Внутренний монолог] Философская мысль (макс. 25 слов).

События:
{context}

ВАЖНО: Мысль должна быть ОРИГИНАЛЬНОЙ.

Мысль:"""

        thought = await self.llm.generate_raw(
            prompt,
            temperature=base_temp,
            max_tokens=70,
            top_k=CONFIG.thought_top_k,
            top_p=CONFIG.thought_top_p
        )

        if thought:
            self.memory.add_to_working(f"💭 {thought}", importance=0.6, role='assistant')
            logger.info(f"💭 [{self.user_id}] Thought (T={base_temp:.2f}): {thought}")

    async def _self_reflection(self):
        """Саморефлексия"""
        stats = self.neural_net.get_statistics()
        mem_stats = self.memory.get_statistics()
        real_metrics = self.neural_net.get_real_metrics()

        prompt = f"""[Саморефлексия v32] (3-4 предложения).

🧬 Нейроны: {stats['neurons']['total']}
Расширений: {stats['activity']['expansion_count']}
LR: {stats['adaptive']['learning_rate']:.4f}

📚 LTM: {mem_stats['long_term']}
Слияний: {mem_stats.get('consolidation_stats', {}).get('merges', 0)}

Рефлексия:"""

        reflection = await self.llm.generate_raw(prompt, temperature=0.7, max_tokens=150)

        if reflection:
            self.memory.add_to_working(f"🔍 {reflection}", importance=0.8, role='assistant')
            logger.info(f"🔍 [{self.user_id}] Reflection: {reflection}")

    async def _optimize_neural_network_enhanced(self):
        """🆕 Улучшенная оптимизация"""
        pruned = 0
        to_prune = []

        for key, synapse in self.neural_net.synapses.items():
            age_days = (time.time() - synapse.created_at) / 86400

            if (synapse.strength < CONFIG.pruning_threshold and
                    synapse.activation_count < 15 and
                    age_days > 1):
                to_prune.append(key)

        for key in to_prune:
            del self.neural_net.synapses[key]
            pruned += 1

        if pruned > 0:
            self.neural_net.pruning_events += pruned

        expanded = self.neural_net.expand_neurons_dynamic()

        if pruned > 0 or expanded > 0:
            logger.info(f"⚙️ Optimization: pruned={pruned}, expanded={expanded}")

    def _update_metrics(self):
        """Обновление метрик"""
        avg_confidence = np.mean(
            list(self.metacognition.confidence_log)) if self.metacognition.confidence_log else 0.7

        self.neural_net.meta_learning_update(avg_confidence)

    async def process_interaction(self, user_input: str) -> Tuple[str, Optional[str], Dict]:
        """Обработка взаимодействия v32"""
        start_time = time.time()
        self.last_interaction = time.time()

        user_emotion = self.emotional_intelligence.analyze_emotion(user_input)

        self.memory.add_to_working(f"User: {user_input}", importance=0.7, role='user')

        self.conversation_context.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time(),
        })

        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]

        relevant_memories = self.memory.search_semantic(user_input, top_k=7)

        memory_context = "\n".join([
            f"• [{source}] {text[:80]}"
            for text, sim, source in relevant_memories
        ]) if relevant_memories else "Нет"

        topic_familiarity = relevant_memories[0][1] if relevant_memories else 0.0

        query_type = self._classify_query_type(user_input)
        target_module = self._map_query_to_module(query_type)

        input_vector = self._simple_embedding(user_input)[:100]
        input_vector = np.pad(input_vector, (0, max(0, 100 - len(input_vector))))

        neural_output = self.neural_net.forward_pass_with_attention(
            input_vector,
            target_module=target_module
        )

        current_lr = self.neural_net.lr_scheduler.get_lr()
        for synapse in self.neural_net.synapses.values():
            source = self.neural_net.neurons[synapse.source_id]
            target = self.neural_net.neurons[synapse.target_id]

            if source.activation > 0.1 and target.activation > 0.1:
                synapse.hebbian_update(
                    source.activation,
                    target.activation,
                    current_lr,
                    synapse.attention_weight
                )

        real_metrics = self.neural_net.get_real_metrics()

        empathy_modifier = self.emotional_intelligence.generate_empathetic_response_modifier(user_emotion)

        prompt = f"""AGI v32.0

{empathy_modifier}

📚 ПАМЯТЬ:
{memory_context}

❓ {user_input}

Ответ (2-5 предложений):"""

        raw_response = await self.llm.generate_raw(
            prompt,
            temperature=0.75,
            max_tokens=400
        )

        if not raw_response:
            raw_response = "Извини, возникли сложности с ответом."

        raw_confidence = 0.7
        calibrated_confidence = self.metacognition.get_calibrated_confidence(raw_confidence)
        self.metacognition.confidence_log.append(calibrated_confidence)

        uncertainty, uncertainty_reasons = self.metacognition.assess_uncertainty({
            'memory_count': len(self.memory.long_term_memory),
            'topic_familiarity': topic_familiarity,
        })

        autonomous_question = None

        should_ask = self.metacognition.should_ask_question(
            uncertainty,
            {'explicit_instruction': False}
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
                autonomous_question = autonomous_question.strip().strip('"\'')
                if not autonomous_question.endswith('?'):
                    autonomous_question += '?'
                if len(autonomous_question) <= 120:
                    self.metacognition.record_question(autonomous_question)
                    self.metacognition.last_question_time = time.time()
                else:
                    autonomous_question = None

        importance = 0.5 + calibrated_confidence * 0.3

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
        })

        self._extract_causal_links(user_input, raw_response)

        response_time = time.time() - start_time

        self.metrics.record_interaction(response_time, calibrated_confidence)

        metadata = {
            'real_metrics': real_metrics if CONFIG.honest_metrics else None,
            'response_time': response_time,
        }

        logger.info(f"✅ [{self.user_id}] Response v32 | Time: {response_time:.2f}s")

        return raw_response, autonomous_question, metadata

    def _classify_query_type(self, query: str) -> str:
        query_lower = query.lower()

        if any(word in query_lower for word in ['что', 'когда', 'где']):
            return 'factual'
        if any(word in query_lower for word in ['придумай', 'создай']):
            return 'creative'
        if any(word in query_lower for word in ['почему', 'причина']):
            return 'causal'

        return 'general'

    def _map_query_to_module(self, query_type: str) -> str:
        module_map = {
            'factual': 'memory',
            'creative': 'reasoning',
            'causal': 'reasoning',
            'general': 'perception',
        }
        return module_map.get(query_type, 'perception')

    def _extract_causal_links(self, user_input: str, response: str):
        """Извлечение причинных связей"""
        combined = f"{user_input} {response}".lower()
        pattern = r'(.+?)\s+(?:потому что|так как)\s+(.+?)(?:\.|,|$)'
        matches = re.findall(pattern, combined)

        for effect, cause in matches:
            effect = effect.strip()[:100]
            cause = cause.strip()[:100]

            if len(effect) > 10 and len(cause) > 10:
                self.causal_reasoning.add_causal_link(cause, effect, confidence=0.6)

    def get_status(self) -> Dict[str, Any]:
        """Статус системы v32"""
        neural_stats = self.neural_net.get_statistics()
        memory_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        real_metrics = self.neural_net.get_real_metrics() if CONFIG.honest_metrics else {}

        return {
            'identity': {
                'user_id': self.user_id,
                'version': 'v32.0',
                'age': self._get_age_string(),
            },
            'neural_network': neural_stats,
            'real_metrics': real_metrics,
            'memory': memory_stats,
            'performance': metrics,
        }

    def _get_age_string(self) -> str:
        age = time.time() - self.birth_time
        hours = int(age / 3600)
        minutes = int((age % 3600) / 60)

        if hours > 0:
            return f"{hours}ч {minutes}м"
        return f"{minutes}м"


class EnhancedAGIBot:
    """Telegram бот для Enhanced AGI v32.0"""

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
            ('neural', self._cmd_neural),
            ('memory', self._cmd_memory),
            ('synergy', self._cmd_synergy),
            ('help', self._cmd_help),
        ]

        for cmd, handler in commands:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Enhanced AGI Bot v32.0 initialized")

    async def _get_or_create_brain(self, user_id: str) -> EnhancedAutonomousAGI:
        if user_id not in self.brains:
            brain = EnhancedAutonomousAGI(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
        return self.brains[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or not update.message:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text

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
            logger.exception(f"❌ Error: {e}")
            await update.message.reply_text("⚠️ Ошибка. /help")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        stats = brain.neural_net.get_statistics()
        mem_stats = brain.memory.get_statistics()

        message = f"""🧠 <b>ENHANCED AGI v32.0</b>

🚀 <b>НОВОЕ:</b>
✨ Динамическое расширение ({stats['activity']['expansion_count']} расширений)
🎨 Вариативность мыслей
🔗 Кросс-модульная синергия
💾 Улучшенная консолидация ({mem_stats.get('consolidation_stats', {}).get('merges', 0)} слияний)

📊 Нейронов: {stats['neurons']['total']}
📚 LTM: {mem_stats['long_term']}

/help — справка"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        status = brain.get_status()

        message = f"""🧠 <b>STATUS v32</b>

🧬 Нейронов: {status['neural_network']['neurons']['total']}
Расширений: {status['neural_network']['activity']['expansion_count']}

📚 LTM: {status['memory']['long_term']}
Эпизодов: {status['memory']['episodic']}
Архивов: {status['memory']['archived']}"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_neural(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.neural_net.get_statistics()

        message = f"""🧬 <b>НЕЙРОСЕТЬ v32</b>

• Нейронов: {stats['neurons']['total']}
• Расширений: {stats['activity']['expansion_count']}
• LR: {stats['adaptive']['learning_rate']:.4f}
• Dropout: {stats['adaptive']['dropout_rate']:.3f}
• Плато: {'Да' if stats['adaptive']['plateau_detected'] else 'Нет'}"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.memory.get_statistics()

        message = f"""📚 <b>ПАМЯТЬ v32</b>

• LTM: {stats['long_term']}
• Эпизодов: {stats['episodic']}
• Архивов: {stats['archived']}
• Слияний: {stats.get('consolidation_stats', {}).get('merges', 0)}"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_synergy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.neural_net.get_statistics()
        synergy = stats.get('synergy', {})

        active = synergy.get('active_synergies', {})

        message = f"""🔗 <b>СИНЕРГИЯ</b>

Активных связей: {len(active)}
Средняя: {synergy.get('avg_synergy', 0.0):.2f}"""

        if active:
            message += "\n\n<b>Активные:</b>"
            for pair, score in list(active.items())[:5]:
                message += f"\n{pair}: {score:.2f}"

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🧠 <b>ENHANCED AGI v32.0</b>

<b>КОМАНДЫ:</b>
/start — приветствие
/status — статус
/neural — нейросеть
/memory — память
/synergy — синергия
/help — справка

<b>ВОЗМОЖНОСТИ:</b>
✨ Динамическое расширение
🎨 Вариативность мыслей
🔗 Кросс-модульная синергия
💾 Умная консолидация"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def start_polling(self):
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot v32.0 started")

    async def shutdown(self):
        logger.info("🛑 Shutting down...")

        for user_id, brain in self.brains.items():
            try:
                await brain.stop()
            except Exception as e:
                logger.error(f"⚠️ Error stopping {user_id}: {e}")

        if self.llm:
            await self.llm.close()

        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Stopped")


async def main():
    """Главная функция"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🧠 ENHANCED AGI BRAIN v32.0 — РАСШИРЕННАЯ ОПТИМИЗАЦИЯ      ║
    ╚══════════════════════════════════════════════════════════════╝

    🚀 НОВЫЕ ВОЗМОЖНОСТИ:

    ✨ ДИНАМИЧЕСКОЕ РАСШИРЕНИЕ НЕЙРОНОВ
    🎨 ВАРИАТИВНОСТЬ АВТОНОМНЫХ МЫСЛЕЙ
    🔗 КРОСС-МОДУЛЬНАЯ СИНЕРГИЯ
    💾 УЛУЧШЕННАЯ КОНСОЛИДАЦИЯ ПАМЯТИ

    🆕 ДОПОЛНИТЕЛЬНО:
    • Adaptive Learning Rate
    • Gradient Clipping
    • Batch Normalization
    • Dropout Scheduling
    • Memory Defragmentation
    """)

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = EnhancedAGIBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 ENHANCED AGI v32.0 АКТИВЕН")
        logger.info("🛑 Ctrl+C для остановки\n")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n👋 Остановка")
    except Exception as e:
        logger.critical(f"❌ Ошибка: {e}", exc_info=True)
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
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)