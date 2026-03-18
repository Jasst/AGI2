#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 TEMPORAL COGNITIVE BRAIN v31.1 — ИСПРАВЛЕННАЯ ВЕРСИЯ С ЧЕСТНОЙ САМОРЕФЛЕКСИЕЙ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ v31.1:

❌ ПРОБЛЕМА: ИИ генерировал псевдометрики (0.46, 0.49) без реальных вычислений
✅ РЕШЕНИЕ: Реальные метрики из нейросети или честное признание их отсутствия

❌ ПРОБЛЕМА: Путаница self/other (кто сгенерировал числа)
✅ РЕШЕНИЕ: Явное отслеживание источника каждого сообщения

❌ ПРОБЛЕМА: Цикл "признание → повтор ошибки"
✅ РЕШЕНИЕ: Проверка перед выводом любых метрик

🎯 КЛЮЧЕВЫЕ ПРИНЦИПЫ v31.1:
1. Честность > Правдоподобие
2. Если метрика не вычислена — не выводить
3. Явная атрибуция источника данных
4. Проверяемые утверждения
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
    """Расширенная конфигурация v31.1"""
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🔧 НОВОЕ: Режим честной метрики
    honest_metrics: bool = True  # Выводить только реальные метрики
    show_uncertainty: bool = True  # Показывать неуверенность явно

    initial_neurons: int = 150
    max_neurons: int = 15000
    neurogenesis_threshold: float = 0.75
    pruning_threshold: float = 0.015
    learning_rate: float = 0.12
    attention_heads: int = 4
    dropout_rate: float = 0.1

    working_memory_size: int = 7
    short_term_size: int = 100
    short_term_decay: float = 0.95
    long_term_size: int = 10000
    consolidation_threshold: float = 0.7

    episodic_context_window: int = 5

    curiosity_threshold: float = 0.55
    question_cooldown: int = 90
    max_autonomous_questions: int = 5
    reasoning_depth: int = 3
    analogy_threshold: float = 0.6

    confidence_calibration_samples: int = 50
    error_correction_threshold: float = 0.3

    emotion_tracking: bool = True
    empathy_weight: float = 0.3

    embedding_dim: int = 512
    semantic_cache_size: int = 2000

    spontaneous_thought_interval: int = 150
    reflection_interval: int = 600
    consolidation_interval: int = 1200
    save_interval: int = 240
    neural_optimization_interval: int = 450
    metrics_update_interval: int = 180

    base_dir: Path = Path(os.getenv('BASE_DIR', 'temporal_brain_v31'))

    def __post_init__(self):
        for subdir in ['memory', 'neural_nets', 'knowledge', 'cache', 'logs',
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
    logger = logging.getLogger('Enhanced_AGI')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'agi_v31_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# 🔧 НОВЫЙ КЛАСС: Отслеживание атрибуции сообщений
@dataclass
class MessageAttribution:
    """Отслеживание источника каждого сообщения"""
    role: str  # 'user' или 'assistant'
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
    valence: float  # -1.0 до 1.0
    arousal: float  # 0.0 до 1.0
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
        """Расчёт эмоциональной валентности"""
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
        """Расчёт уровня возбуждения"""
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
        """Генерация модификатора для эмпатичного ответа"""
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
        """Обновление при доступе"""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class Episode:
    """Эпизод памяти (событие с контекстом)"""
    id: str
    messages: List[Dict[str, str]]
    context: str
    timestamp: float
    emotional_state: Optional[EmotionalState] = None
    importance: float = 0.5

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'messages': self.messages,
            'context': self.context,
            'timestamp': self.timestamp,
            'importance': self.importance,
        }


class MultiLevelMemory:
    """Многоуровневая система памяти"""

    def __init__(self, embedding_func):
        self.embedding_func = embedding_func

        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)
        self.short_term_memory: List[MemoryItem] = []
        self.long_term_memory: Dict[str, MemoryItem] = {}
        self.episodic_memory: Dict[str, Episode] = {}
        self.current_episode_buffer: List[Dict] = []
        self.semantic_memory: Dict[str, MemoryItem] = {}

        # 🔧 НОВОЕ: Отслеживание атрибуции
        self.message_history: List[MessageAttribution] = []

        self.last_consolidation = time.time()

    def add_to_working(self, content: str, importance: float = 0.5, role: str = 'assistant'):
        """Добавление в оперативную память с атрибуцией"""
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            embedding=self.embedding_func(content)
        )
        self.working_memory.append(item)
        self.short_term_memory.append(item)

        # 🔧 НОВОЕ: Сохранение атрибуции
        attribution = MessageAttribution(
            role=role,
            content=content,
            timestamp=time.time()
        )
        self.message_history.append(attribution)

    def find_message_source(self, content_fragment: str) -> Optional[str]:
        """🔧 НОВОЕ: Поиск источника сообщения"""
        for msg in reversed(self.message_history):
            if content_fragment.lower() in msg.content.lower():
                return msg.role
        return None

    def get_working_memory_context(self) -> str:
        """Получение контекста из оперативной памяти"""
        return "\n".join([item.content for item in self.working_memory])

    def decay_short_term(self):
        """Применение decay к кратковременной памяти"""
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
        """Консолидация важных воспоминаний в долговременную память"""
        candidates = [
            item for item in self.short_term_memory
            if item.importance >= CONFIG.consolidation_threshold
        ]

        for item in candidates:
            memory_id = hashlib.sha256(item.content.encode()).hexdigest()[:16]

            if memory_id not in self.long_term_memory:
                self.long_term_memory[memory_id] = item
                logger.debug(f"💾 Консолидация в LTM: {item.content[:50]}...")

        self.last_consolidation = time.time()

    def add_episode(self, messages: List[Dict], context: str,
                    emotion: Optional[EmotionalState] = None):
        """Добавление эпизода"""
        episode_id = f"ep_{int(time.time() * 1000)}"

        episode = Episode(
            id=episode_id,
            messages=messages,
            context=context,
            timestamp=time.time(),
            emotional_state=emotion,
            importance=0.5
        )

        self.episodic_memory[episode_id] = episode

        if len(self.episodic_memory) > 500:
            sorted_episodes = sorted(
                self.episodic_memory.items(),
                key=lambda x: (x[1].importance, x[1].timestamp)
            )
            to_remove = sorted_episodes[:100]
            for ep_id, _ in to_remove:
                del self.episodic_memory[ep_id]

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Семантический поиск по всем уровням памяти"""
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
        """Получение недавних эпизодов"""
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
            'semantic': len(self.semantic_memory),
        }


@dataclass
class EnhancedNeuron:
    """Улучшенный нейрон с модульностью"""
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

    def activate(self, input_sum: float, dropout: float = 0.0) -> float:
        """Активация с dropout"""
        if random.random() < dropout:
            self.activation = 0.0
        else:
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
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedNeuron':
        return cls(**data)


@dataclass
class EnhancedSynapse:
    """Улучшенный синапс с attention"""
    source_id: str
    target_id: str
    weight: float = 0.0
    strength: float = 1.0
    plasticity: float = 1.0
    activation_count: int = 0
    created_at: float = field(default_factory=time.time)
    attention_weight: float = 1.0

    def hebbian_update(self, source_activation: float, target_activation: float,
                       learning_rate: float, attention: float = 1.0):
        """Hebbian learning с attention"""
        delta = learning_rate * source_activation * target_activation * attention
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
        return cls(**data)


class EnhancedNeuralNetwork:
    """Улучшенная нейросеть с РЕАЛЬНЫМИ метриками"""

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

        # 🔧 НОВОЕ: Кеш последних реальных метрик
        self.last_real_metrics: Dict[str, float] = {}
        self.metrics_timestamp: float = 0.0

        self._initialize_modular_network(initial_neurons)
        logger.info(f"🧬 Enhanced Neural Network: {len(self.neurons)} neurons, "
                    f"{len(self.modules)} modules")

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

    def _get_neuron_type(self, layer: int) -> str:
        types = {0: "sensory", 1: "hidden", 2: "hidden", 3: "motor"}
        return types.get(layer, "general")

    def _connect_within_module(self, module_neurons: Set[str]):
        """Создание связей внутри модуля"""
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
        """Межмодульные связи"""
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

    def forward_pass_with_attention(self, input_vector: np.ndarray,
                                    target_module: str = None) -> np.ndarray:
        """Прямой проход с attention mechanism + СОХРАНЕНИЕ РЕАЛЬНЫХ МЕТРИК"""
        for neuron in self.neurons.values():
            neuron.activation = 0.0

        input_neurons = [nid for nid in self.layers[0]
                         if target_module is None or self.neurons[nid].module == target_module]

        for i, neuron_id in enumerate(input_neurons):
            if i < len(input_vector):
                self.neurons[neuron_id].activation = input_vector[i]

        max_layer = max(self.layers.keys())

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

                    self.neurons[target_id].activate(weighted_input, CONFIG.dropout_rate)

        output_neurons = list(self.layers[max_layer])
        output = np.array([self.neurons[nid].activation for nid in output_neurons])

        self.total_activations += 1

        # 🔧 НОВОЕ: Сохранение РЕАЛЬНЫХ метрик
        self._update_real_metrics()

        return output

    def _update_real_metrics(self):
        """🔧 НОВОЕ: Вычисление и сохранение реальных метрик"""
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

        logger.debug(f"📊 Real metrics updated: {self.last_real_metrics}")

    def get_real_metrics(self) -> Dict[str, float]:
        """🔧 НОВОЕ: Получение ТОЛЬКО реальных метрик"""
        if not CONFIG.honest_metrics:
            return {}

        age = time.time() - self.metrics_timestamp

        if age > 60:
            logger.warning(f"⚠️ Metrics are {age:.0f}s old - may be stale")

        return {
            **self.last_real_metrics,
            'metrics_age_seconds': age,
        }

    def meta_learning_update(self, performance_score: float):
        """Meta-learning: обучение обучаться"""
        if performance_score > 0.8:
            CONFIG.learning_rate *= 1.05
        elif performance_score < 0.5:
            CONFIG.learning_rate *= 0.95

        CONFIG.learning_rate = np.clip(CONFIG.learning_rate, 0.01, 0.3)

        self.meta_learning_score = 0.9 * self.meta_learning_score + 0.1 * performance_score

        logger.debug(f"🎓 Meta-learning: LR={CONFIG.learning_rate:.4f}, "
                     f"Score={self.meta_learning_score:.3f}")

    def transfer_learning(self, source_module: str, target_module: str, strength: float = 0.5):
        """Transfer learning между модулями"""
        source_neurons = self.modules[source_module]
        target_neurons = self.modules[target_module]

        num_transfers = max(1, int(len(source_neurons) * 0.1))

        for _ in range(num_transfers):
            source_id = random.choice(list(source_neurons))
            target_id = random.choice(list(target_neurons))

            if (source_id, target_id) not in self.synapses:
                synapse = EnhancedSynapse(
                    source_id=source_id,
                    target_id=target_id,
                    weight=random.gauss(0, 0.2) * strength
                )
                self.synapses[(source_id, target_id)] = synapse
                logger.debug(f"🔄 Transfer learning: {source_module} → {target_module}")

    def get_module_activation(self, module_name: str) -> float:
        """Средняя активация модуля"""
        module_neurons = self.modules[module_name]
        activations = [self.neurons[nid].activation for nid in module_neurons]
        return np.mean(activations) if activations else 0.0

    def get_active_neurons(self, threshold: float = 0.3) -> List[str]:
        """Получить ID активных нейронов выше порога"""
        return [nid for nid, neuron in self.neurons.items()
                if neuron.activation > threshold]

    def get_statistics(self) -> Dict[str, Any]:
        """Расширенная статистика"""
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
            },
            'modules': {
                m: self.get_module_activation(m) for m in self.modules.keys()
            }
        }

    def save(self, path: Path):
        """Сохранение сети"""
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
            }
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb', compresslevel=6) as f:
            pickle.dump(state, f)

        logger.info(f"💾 Neural network saved: {len(self.neurons)} neurons")

    def load(self, path: Path) -> bool:
        """Загрузка сети"""
        if not path.exists():
            return False

        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)

            self.neurons = {}
            for n_dict in state.get('neurons', []):
                neuron = EnhancedNeuron(**{k: v for k, v in n_dict.items() if k != 'activation'})
                self.neurons[neuron.id] = neuron

            self.synapses = {}
            for s_dict in state.get('synapses', []):
                synapse = EnhancedSynapse(**s_dict)
                self.synapses[(synapse.source_id, synapse.target_id)] = synapse

            self.layers = {int(l): set(neurons) for l, neurons in state.get('layers', {}).items()}
            self.modules = {m: set(neurons) for m, neurons in state.get('modules', {}).items()}

            meta = state.get('meta', {})
            self.total_activations = meta.get('total_activations', 0)
            self.neurogenesis_events = meta.get('neurogenesis_events', 0)
            self.pruning_events = meta.get('pruning_events', 0)
            self.meta_learning_score = meta.get('meta_learning_score', 0.5)

            logger.info(f"✅ Neural network loaded: {len(self.neurons)} neurons, "
                        f"{len(self.synapses)} synapses")
            return True

        except Exception as e:
            logger.error(f"⚠️ Error loading neural network: {e}")
            return False

# Остальные классы (EnhancedMetacognition, AnalogicalReasoning, и т.д.)
# остаются без изменений, так как проблема была специфична для метрик нейросети

# Продолжение следует в следующей части...