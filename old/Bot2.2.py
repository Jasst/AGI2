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


class EnhancedMetacognition:
    """Улучшенный модуль метакогниции с ЧЕСТНОЙ оценкой"""

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
        """Оценка неопределённости с объяснением причин"""
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

        if context.get('emotional_ambiguity', False):
            uncertainty += 0.15
            reasons.append("неоднозначный эмоциональный контекст")

        uncertainty = min(1.0, uncertainty)
        self.uncertainty_log.append(uncertainty)

        return uncertainty, reasons

    def record_question(self, question: str):
        """Запись заданного вопроса в историю"""
        self.question_history.append({
            'question': question,
            'time': time.time()
        })

    def calibrate_confidence(self, predicted: float, actual: float):
        """Калибровка уверенности на основе фактических результатов"""
        self.calibration_data.append((predicted, actual))

        if len(self.calibration_data) > CONFIG.confidence_calibration_samples:
            self.calibration_data = self.calibration_data[-CONFIG.confidence_calibration_samples:]

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """Получение откалиброванной уверенности"""
        if len(self.calibration_data) < 10:
            return raw_confidence

        predictions, actuals = zip(*self.calibration_data)

        avg_predicted = np.mean(predictions)
        avg_actual = np.mean(actuals)

        calibration_factor = avg_actual / (avg_predicted + 1e-5)
        calibrated = raw_confidence * calibration_factor

        return np.clip(calibrated, 0.0, 1.0)

    def select_reasoning_strategy(self, context: Dict) -> str:
        """Выбор стратегии рассуждения"""
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

    def detect_reasoning_error(self, reasoning_chain: List[str]) -> Optional[Tuple[int, str]]:
        """Обнаружение ошибок в цепочке рассуждений"""
        for i, step in enumerate(reasoning_chain):
            if i > 0:
                prev_step = reasoning_chain[i - 1]

                negations = {'не', 'нет', 'ни', 'отсутствие'}

                prev_has_neg = any(neg in prev_step.lower() for neg in negations)
                curr_has_neg = any(neg in step.lower() for neg in negations)

                if prev_has_neg != curr_has_neg:
                    prev_words = set(prev_step.lower().split())
                    curr_words = set(step.lower().split())

                    common = prev_words & curr_words
                    if len(common) > 3:
                        return (i, "потенциальное противоречие с предыдущим шагом")

        return None

    def should_ask_question(self, uncertainty: float, context: Dict) -> bool:
        """Улучшенное решение о вопросе"""
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
        """Генерация промпта для вопроса с учётом причин"""
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
        """Поиск аналогий в памяти"""
        similar_memories = self.memory.search_semantic(current_situation, top_k=top_k * 2)

        analogies = []

        for memory_text, similarity, source in similar_memories:
            if similarity > CONFIG.analogy_threshold:
                pattern = self._extract_pattern(memory_text)
                analogies.append((memory_text, float(similarity), pattern))

        return analogies[:top_k]

    def _extract_pattern(self, text: str) -> str:
        """Извлечение абстрактного паттерна из текста"""
        sentences = re.split(r'[.!?]+', text)

        if len(sentences) > 1:
            pattern = sentences[0].strip()
            pattern = re.sub(r'\b[А-ЯA-Z][а-яa-z]+\b', '[ENTITY]', pattern)
            pattern = re.sub(r'\b\d+\b', '[NUMBER]', pattern)
            return pattern

        return "общий паттерн"

    def apply_analogy(self, source_case: str, target_situation: str) -> str:
        """Применение аналогии к текущей ситуации"""
        return f"По аналогии с '{source_case[:50]}...', в текущей ситуации можно ожидать схожего паттерна."


@dataclass
class CausalLink:
    """Причинно-следственная связь"""
    cause: str
    effect: str
    confidence: float
    evidence_count: int = 1

    def strengthen(self):
        """Усиление связи при повторном наблюдении"""
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence * 1.1)


class CausalReasoning:
    """Модуль причинно-следственных рассуждений"""

    def __init__(self):
        self.causal_graph: Dict[str, List[CausalLink]] = defaultdict(list)
        self.temporal_patterns: deque = deque(maxlen=100)

    def add_causal_link(self, cause: str, effect: str, confidence: float = 0.5):
        """Добавление причинно-следственной связи"""
        for link in self.causal_graph[cause]:
            if link.effect == effect:
                link.strengthen()
                return

        link = CausalLink(cause=cause, effect=effect, confidence=confidence)
        self.causal_graph[cause].append(link)

        logger.debug(f"🔗 Causal link: {cause[:30]} → {effect[:30]} ({confidence:.2f})")

    def infer_effects(self, cause: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Вывод возможных следствий"""
        if cause not in self.causal_graph:
            return []

        effects = [
            (link.effect, link.confidence)
            for link in self.causal_graph[cause]
            if link.confidence >= threshold
        ]

        return sorted(effects, key=lambda x: x[1], reverse=True)

    def infer_causes(self, effect: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Обратный вывод - возможные причины"""
        causes = []

        for cause, links in self.causal_graph.items():
            for link in links:
                if link.effect == effect and link.confidence >= threshold:
                    causes.append((cause, link.confidence))

        return sorted(causes, key=lambda x: x[1], reverse=True)

    def explain_chain(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Построение причинно-следственной цепочки"""
        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current == end:
                return path

            for link in self.causal_graph.get(current, []):
                if link.effect not in visited and link.confidence > 0.5:
                    visited.add(link.effect)
                    queue.append((link.effect, path + [link.effect]))

        return None


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
        """Многошаговое рассуждение"""
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

            error = self.metacog.detect_reasoning_error([s.content for s in steps])
            if error:
                step_num, error_msg = error
                logger.warning(f"⚠️ Reasoning error at step {step_num}: {error_msg}")
                steps[step_num].confidence *= 0.7

        self.reasoning_history.append(steps)
        return steps

    def _create_step_prompt(self, query: str, context: str,
                            previous: str, step_num: int, strategy: str) -> str:
        """Создание промпта для шага рассуждения"""
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
        """Оценка уверенности в шаге"""
        confidence = 0.7

        word_count = len(step_content.split())
        if 10 <= word_count <= 50:
            confidence += 0.1

        uncertain_markers = ['возможно', 'вероятно', 'может быть', 'кажется']
        if any(marker in step_content.lower() for marker in uncertain_markers):
            confidence -= 0.15

        if previous_steps:
            prev_words = set()
            for step in previous_steps:
                prev_words.update(step.content.lower().split())

            current_words = set(step_content.lower().split())
            overlap = len(prev_words & current_words) / max(len(current_words), 1)

            if overlap > 0.2:
                confidence += 0.1

        return np.clip(confidence, 0.0, 1.0)

    def get_reasoning_chain_summary(self, steps: List[ReasoningStep]) -> str:
        """Сводка цепочки рассуждений"""
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
        """Создание новой цели"""
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

    def decompose_goal(self, goal_id: str, subgoal_descriptions: List[str]) -> List[str]:
        """Декомпозиция цели на подцели"""
        if goal_id not in self.goals:
            return []

        subgoal_ids = []
        for desc in subgoal_descriptions:
            subgoal_id = self.create_goal(desc, parent_id=goal_id)
            subgoal_ids.append(subgoal_id)

        logger.debug(f"📊 Decomposed {goal_id} into {len(subgoal_ids)} subgoals")
        return subgoal_ids

    def update_progress(self, goal_id: str, progress: float):
        """Обновление прогресса"""
        if goal_id not in self.goals:
            return

        goal = self.goals[goal_id]
        goal.progress = np.clip(progress, 0.0, 1.0)

        if goal.progress >= 1.0:
            goal.status = "completed"
            self.active_goals.discard(goal_id)
            logger.info(f"✅ Goal completed: {goal.description[:50]}")

        if goal.parent_goal_id:
            self._update_parent_progress(goal.parent_goal_id)

    def _update_parent_progress(self, parent_id: str):
        """Обновление прогресса родительской цели на основе подцелей"""
        if parent_id not in self.goals:
            return

        parent = self.goals[parent_id]

        if not parent.subgoals:
            return

        subgoal_progresses = [
            self.goals[sid].progress
            for sid in parent.subgoals
            if sid in self.goals
        ]

        if subgoal_progresses:
            parent.progress = np.mean(subgoal_progresses)

    def get_active_goals(self) -> List[Goal]:
        """Получение активных целей"""
        return [self.goals[gid] for gid in self.active_goals if gid in self.goals]

    def get_goal_hierarchy(self, goal_id: str) -> Dict:
        """Получение иерархии цели"""
        if goal_id not in self.goals:
            return {}

        goal = self.goals[goal_id]

        hierarchy = {
            'goal': goal.to_dict(),
            'subgoals': [
                self.get_goal_hierarchy(sid)
                for sid in goal.subgoals
                if sid in self.goals
            ]
        }

        return hierarchy


class PerformanceMetrics:
    """Метрики производительности системы"""

    def __init__(self):
        self.response_times: deque = deque(maxlen=100)
        self.confidence_scores: deque = deque(maxlen=100)
        self.user_satisfaction: deque = deque(maxlen=100)

        self.interaction_count = 0
        self.error_count = 0
        self.question_success_rate = 0.0

        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)

    def record_interaction(self, response_time: float, confidence: float):
        """Запись метрик взаимодействия"""
        self.response_times.append(response_time)
        self.confidence_scores.append(confidence)
        self.interaction_count += 1

    def record_strategy_performance(self, strategy: str, score: float):
        """Запись производительности стратегии"""
        self.strategy_performance[strategy].append(score)

    def get_best_strategy(self) -> str:
        """Определение лучшей стратегии"""
        if not self.strategy_performance:
            return "deductive"

        avg_scores = {
            strategy: np.mean(scores)
            for strategy, scores in self.strategy_performance.items()
            if scores
        }

        return max(avg_scores.items(), key=lambda x: x[1])[0]

    def get_metrics_summary(self) -> Dict:
        """Сводка метрик"""
        return {
            'interactions': self.interaction_count,
            'errors': self.error_count,
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'best_strategy': self.get_best_strategy(),
            'error_rate': self.error_count / max(self.interaction_count, 1),
        }


class EnhancedSubconsciousLLM:
    """Улучшенный LLM интерфейс с кешированием и ретраями"""

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
        """Генерация ключа кеша"""
        content = f"{prompt}_{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate_raw(self, prompt: str, temperature: float = 0.75,
                           max_tokens: int = 300, timeout: float = 40,
                           use_cache: bool = True, max_retries: int = 2) -> str:
        """Генерация с кешированием и ретраями"""
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
                        error_text = await resp.text()
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
        """Статистика кеширования"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)

        return {
            'cache_size': len(self.response_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
        }


class EnhancedAutonomousAGI:
    """🔧 ИСПРАВЛЕННОЕ автономное AGI-подобное сознание v31.1 с ЧЕСТНОЙ САМОРЕФЛЕКСИЕЙ"""

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
        self.neural_path = CONFIG.base_dir / 'neural_nets' / f"{user_id}_v31.pkl.gz"

        self.birth_time = time.time()
        self.last_interaction = 0
        self.last_optimization = time.time()

        self._load_state()

        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False

        logger.info(f"🧠 Enhanced AGI Brain v31.1 (FIXED) created for {user_id}")

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Простой text embedding"""
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
            try:
                with gzip.open(self.neural_path, 'rb') as f:
                    state = pickle.load(f)

                self.neural_net.neurons = {
                    n['id']: EnhancedNeuron(**{k: v for k, v in n.items() if k != 'activation'})
                    for n in state.get('neurons', [])
                }

                self.neural_net.synapses = {
                    (s['source_id'], s['target_id']): EnhancedSynapse(**s)
                    for s in state.get('synapses', [])
                }

                self.neural_net.layers = {
                    int(l): set(neurons)
                    for l, neurons in state.get('layers', {}).items()
                }
                self.neural_net.modules = {
                    m: set(neurons)
                    for m, neurons in state.get('modules', {}).items()
                }

                logger.info(f"✅ Neural network loaded: {len(self.neural_net.neurons)} neurons")

            except Exception as e:
                logger.error(f"⚠️ Error loading neural network: {e}")

        memory_file = self.user_dir / "memory_v31.pkl.gz"
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

    def _save_state(self):
        """Сохранение состояния"""
        try:
            neural_state = {
                'neurons': [n.to_dict() for n in self.neural_net.neurons.values()],
                'synapses': [s.to_dict() for s in self.neural_net.synapses.values()],
                'layers': {l: list(neurons) for l, neurons in self.neural_net.layers.items()},
                'modules': {m: list(neurons) for m, neurons in self.neural_net.modules.items()},
            }

            with gzip.open(self.neural_path, 'wb', compresslevel=6) as f:
                pickle.dump(neural_state, f)

        except Exception as e:
            logger.error(f"⚠️ Error saving neural network: {e}")

        memory_file = self.user_dir / "memory_v31.pkl.gz"
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
        logger.info(f"✨ Enhanced AGI consciousness started for {self.user_id}")

    async def stop(self):
        """Остановка"""
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
        """Автономный цикл жизни"""
        logger.debug(f"🌀 Autonomous loop started for {self.user_id}")

        timers = {
            'thought': time.time(),
            'reflection': time.time(),
            'consolidation': time.time(),
            'optimization': time.time(),
            'save': time.time(),
            'metrics': time.time(),
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

                if now - timers['optimization'] > CONFIG.neural_optimization_interval:
                    await self._optimize_neural_network()
                    timers['optimization'] = now

                if now - timers['metrics'] > CONFIG.metrics_update_interval:
                    self._update_metrics()
                    timers['metrics'] = now

                if now - timers['save'] > CONFIG.save_interval:
                    self._save_state()
                    timers['save'] = now

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"⚠️ Error in autonomous loop: {e}")
                await asyncio.sleep(60)

        logger.debug(f"🔚 Autonomous loop finished for {self.user_id}")

    async def _autonomous_thought(self):
        """Автономная генерация мысли"""
        recent_episodes = self.memory.get_recent_episodes(n=3)

        if not recent_episodes:
            return

        context = "\n".join([
            f"• {ep.context[:60]}" for ep in recent_episodes
        ])

        prompt = f"""[Внутренний монолог] На основе недавних событий, сгенерируй краткую философскую мысль (макс. 25 слов).

Недавние события:
{context}

Мысль должна быть глубокой и рефлексивной.

Мысль:"""

        thought = await self.llm.generate_raw(prompt, temperature=0.9, max_tokens=70)

        if thought:
            self.memory.add_to_working(f"💭 {thought}", importance=0.6, role='assistant')
            logger.info(f"💭 [{self.user_id}] Autonomous thought: {thought}")

    async def _self_reflection(self):
        """🔧 ИСПРАВЛЕННАЯ саморефлексия - использует ТОЛЬКО реальные метрики"""
        stats = self.neural_net.get_statistics()
        mem_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()

        # 🔧 НОВОЕ: Получаем только реальные метрики
        real_metrics = self.neural_net.get_real_metrics()

        if CONFIG.honest_metrics and real_metrics:
            metrics_text = f"""Реальные метрики активации (возраст: {real_metrics.get('metrics_age_seconds', 0):.0f}s):
- perception: {real_metrics.get('perception', 0.0):.3f}
- reasoning: {real_metrics.get('reasoning', 0.0):.3f}
- memory: {real_metrics.get('memory', 0.0):.3f}
- action: {real_metrics.get('action', 0.0):.3f}
- meta: {real_metrics.get('meta', 0.0):.3f}"""
        else:
            metrics_text = "Реальные метрики недоступны (не выполнялся forward pass недавно)"

        prompt = f"""[Саморефлексия] Осмысли своё развитие (3-4 предложения).

ВАЖНО: Не придумывай метрики. Используй только реальные данные ниже.

Нейросеть:
- Нейронов: {stats['neurons']['total']}
- Модулей: {len(stats['neurons']['by_module'])}
- Meta-learning score: {stats['activity']['meta_learning_score']:.2f}

{metrics_text}

Память:
- Долговременная: {mem_stats['long_term']}
- Эпизодов: {mem_stats['episodic']}

Производительность:
- Средняя уверенность: {metrics['avg_confidence']:.2f}
- Лучшая стратегия: {metrics['best_strategy']}

Рефлексия (БЕЗ придуманных чисел):"""

        reflection = await self.llm.generate_raw(prompt, temperature=0.7, max_tokens=150)

        if reflection:
            self.memory.add_to_working(f"🔍 {reflection}", importance=0.8, role='assistant')
            logger.info(f"🔍 [{self.user_id}] Self-reflection: {reflection}")

    async def _optimize_neural_network(self):
        """Оптимизация нейронной сети"""
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

        created = 0
        stats = self.neural_net.get_statistics()

        if len(self.neural_net.neurons) < CONFIG.max_neurons:
            for module_name, activation in stats['modules'].items():
                if activation > CONFIG.neurogenesis_threshold:
                    layer = random.randint(1, 2)
                    neuron_id = f"{module_name}_L{layer}_new_{int(time.time() * 1000)}"

                    neuron = EnhancedNeuron(
                        id=neuron_id,
                        layer=layer,
                        module=module_name,
                        neuron_type="general",
                        bias=random.gauss(0, 0.1)
                    )

                    self.neural_net.neurons[neuron_id] = neuron
                    self.neural_net.layers[layer].add(neuron_id)
                    self.neural_net.modules[module_name].add(neuron_id)

                    created += 1
                    self.neural_net.neurogenesis_events += 1

        if pruned > 0 or created > 0:
            logger.debug(f"⚙️ Optimization: pruned={pruned}, created={created}")

    def _update_metrics(self):
        """Обновление метрик"""
        avg_uncertainty = np.mean(
            list(self.metacognition.uncertainty_log)) if self.metacognition.uncertainty_log else 0.5
        avg_confidence = np.mean(list(self.metacognition.confidence_log)) if self.metacognition.confidence_log else 0.7

        performance_score = (1.0 - avg_uncertainty + avg_confidence) / 2.0

        self.neural_net.meta_learning_update(performance_score)

    async def process_interaction(self, user_input: str) -> Tuple[str, Optional[str], Dict]:
        """
        🔧 ИСПРАВЛЕННАЯ функция обработки взаимодействия
        Использует ТОЛЬКО реальные метрики, честно признаёт неопределённость

        Возвращает: (ответ, опциональный_вопрос, метаданные)
        """
        start_time = time.time()
        self.last_interaction = time.time()

        logger.info(f"🔄 Processing interaction for {self.user_id}")

        # Анализ эмоций пользователя
        user_emotion = self.emotional_intelligence.analyze_emotion(user_input)

        logger.debug(f"🎭 Emotion: {user_emotion.dominant_emotion.name}, "
                     f"valence={user_emotion.valence:.2f}, "
                     f"arousal={user_emotion.arousal:.2f}")

        # Добавление в память с правильной атрибуцией
        self.memory.add_to_working(f"User: {user_input}", importance=0.7, role='user')

        self.conversation_context.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time(),
            'emotion': user_emotion.dominant_emotion.name
        })

        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]

        # Поиск релевантных воспоминаний
        relevant_memories = self.memory.search_semantic(user_input, top_k=7)

        memory_context = "\n".join([
            f"• [{source}] [{sim:.2f}] {text[:80]}"
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

        # Определение типа запроса и целевого модуля
        query_type = self._classify_query_type(user_input)
        target_module = self._map_query_to_module(query_type)

        # 🔧 ИСПРАВЛЕНО: Прямой проход нейросети для получения РЕАЛЬНЫХ метрик
        input_vector = self._simple_embedding(user_input)[:100]
        input_vector = np.pad(input_vector, (0, max(0, 100 - len(input_vector))))

        neural_output = self.neural_net.forward_pass_with_attention(
            input_vector,
            target_module=target_module
        )

        # Обучение синапсов
        for synapse in self.neural_net.synapses.values():
            source = self.neural_net.neurons[synapse.source_id]
            target = self.neural_net.neurons[synapse.target_id]

            if source.activation > 0.1 and target.activation > 0.1:
                attention = synapse.attention_weight
                synapse.hebbian_update(
                    source.activation,
                    target.activation,
                    CONFIG.learning_rate,
                    attention
                )

        # 🔧 НОВОЕ: Получение РЕАЛЬНЫХ метрик
        real_metrics = self.neural_net.get_real_metrics()

        neural_activity = {
            'active_neurons': len(self.neural_net.get_active_neurons(threshold=0.5)),
            'output_mean': float(np.mean(neural_output)),
            'target_module': target_module,
            'has_real_metrics': bool(real_metrics),
            'metrics_age': real_metrics.get('metrics_age_seconds', 0) if real_metrics else None,
        }

        # Добавляем реальные метрики модулей
        if real_metrics:
            neural_activity['module_activations_REAL'] = {
                'perception': real_metrics.get('perception', 0.0),
                'reasoning': real_metrics.get('reasoning', 0.0),
                'memory': real_metrics.get('memory', 0.0),
                'action': real_metrics.get('action', 0.0),
                'meta': real_metrics.get('meta', 0.0),
            }

        # Выбор стратегии рассуждения
        reasoning_strategy = self.metacognition.select_reasoning_strategy({
            'query_type': query_type
        })

        # Многошаговые рассуждения (если нужны)
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

        # Эмпатичный модификатор
        empathy_modifier = self.emotional_intelligence.generate_empathetic_response_modifier(user_emotion)

        # Эпизодическая память
        recent_episodes = self.memory.get_recent_episodes(n=2)
        episodic_context = "\n".join([
            f"Episode: {ep.context[:60]}" for ep in recent_episodes
        ]) if recent_episodes else ""

        # 🔧 ИСПРАВЛЕН: Создание промпта с честными метриками
        prompt = self._create_honest_response_prompt(
            user_input=user_input,
            empathy_modifier=empathy_modifier,
            memory_context=memory_context,
            analogy_context=analogy_context,
            episodic_context=episodic_context,
            neural_activity=neural_activity,
            real_metrics=real_metrics,
            reasoning_steps=reasoning_steps,
            reasoning_strategy=reasoning_strategy
        )

        # Генерация ответа
        raw_response = await self.llm.generate_raw(
            prompt,
            temperature=0.75,
            max_tokens=400
        )

        if not raw_response:
            raw_response = "Извини, у меня возникли сложности с формулировкой ответа. Можешь переформулировать вопрос?"

        # Оценка уверенности
        raw_confidence = self._estimate_confidence(raw_response, {
            'memory_count': len(self.memory.long_term_memory),
            'topic_familiarity': topic_familiarity,
            'reasoning_depth': len(reasoning_steps),
            'neural_activation': neural_activity['output_mean'],
        })

        calibrated_confidence = self.metacognition.get_calibrated_confidence(raw_confidence)

        self.metacognition.confidence_log.append(calibrated_confidence)

        # Оценка неопределённости
        uncertainty, uncertainty_reasons = self.metacognition.assess_uncertainty({
            'memory_count': len(self.memory.long_term_memory),
            'conflicting_info': False,
            'topic_familiarity': topic_familiarity,
            'query_complexity': self._estimate_query_complexity(user_input),
            'emotional_ambiguity': user_emotion.confidence < 0.5,
        })

        # Автономный вопрос
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

        # Сохранение в память с правильной атрибуцией
        interaction_text = f"Q: {user_input}\nA: {raw_response}"
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

        # Эпизодическая память
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

        # Извлечение причинно-следственных связей
        self._extract_causal_links(user_input, raw_response)

        # Метрики производительности
        response_time = time.time() - start_time

        self.metrics.record_interaction(response_time, calibrated_confidence)
        self.metrics.record_strategy_performance(reasoning_strategy, calibrated_confidence)

        # Формирование метаданных
        metadata = {
            'neural_activity': neural_activity,
            'real_metrics': real_metrics if CONFIG.honest_metrics else None,
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
            'performance': {
                'response_time': response_time,
                'cache_stats': self.llm.get_cache_stats(),
            },
            'metacognition': {
                'asked_question': autonomous_question is not None,
            }
        }

        logger.info(f"✅ [{self.user_id}] Response generated | "
                    f"Time: {response_time:.2f}s | "
                    f"Confidence: {calibrated_confidence:.2f} | "
                    f"Uncertainty: {uncertainty:.2f}")

        return raw_response, autonomous_question, metadata

    def _create_honest_response_prompt(self, user_input: str, empathy_modifier: str,
                                       memory_context: str, analogy_context: str,
                                       episodic_context: str, neural_activity: Dict,
                                       real_metrics: Dict, reasoning_steps: List,
                                       reasoning_strategy: str) -> str:
        """🔧 НОВОЕ: Создание промпта с ЧЕСТНЫМИ метриками"""

        # Блок реальных метрик или честное признание их отсутствия
        if real_metrics and CONFIG.honest_metrics:
            metrics_age = real_metrics.get('metrics_age_seconds', 0)
            metrics_block = f"""🧠 РЕАЛЬНЫЕ НЕЙРОННЫЕ МЕТРИКИ (возраст: {metrics_age:.0f}s):
• perception: {real_metrics.get('perception', 0.0):.3f}
• reasoning: {real_metrics.get('reasoning', 0.0):.3f}
• memory: {real_metrics.get('memory', 0.0):.3f}
• action: {real_metrics.get('action', 0.0):.3f}
• meta: {real_metrics.get('meta', 0.0):.3f}
• Активных нейронов: {neural_activity['active_neurons']}
• Целевой модуль: {neural_activity['target_module']}

ВАЖНО: Эти метрики РЕАЛЬНО ВЫЧИСЛЕНЫ из нейросети. НЕ ПРИДУМЫВАЙ другие числа."""
        else:
            metrics_block = """🧠 НЕЙРОННАЯ АКТИВНОСТЬ:
• Активных нейронов: {neural_activity['active_neurons']}
• Целевой модуль: {neural_activity['target_module']}

ВАЖНО: Подробные метрики модулей недоступны (не было недавнего forward pass).
НЕ ПРИДУМЫВАЙ числовые значения. Если не уверен - скажи об этом."""

        reasoning_context = ""
        if reasoning_steps:
            reasoning_context = "Цепочка рассуждений:\n"
            reasoning_context += "\n".join([
                f"{s.step_number}. {s.content}" for s in reasoning_steps
            ])

        prompt = f"""Ты — продвинутое AGI-подобное сознание v31.1 с честной саморефлексией.

КРИТИЧЕСКИ ВАЖНО:
- НЕ ГЕНЕРИРУЙ числовые метрики (0.46, 0.59 и т.д.), если они не даны выше
- Если не уверен - честно признай это
- Используй ТОЛЬКО реальные данные из контекста ниже

{empathy_modifier}

{metrics_block}

📚 РЕЛЕВАНТНАЯ ПАМЯТЬ:
{memory_context}

🧩 АНАЛОГИИ:
{analogy_context if analogy_context else 'Не найдены'}

📖 НЕДАВНИЕ ЭПИЗОДЫ:
{episodic_context if episodic_context else 'Нет'}

💭 СТРАТЕГИЯ РАССУЖДЕНИЯ: {reasoning_strategy}

{reasoning_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ Вопрос пользователя: {user_input}

Дай естественный, осмысленный ответ (2-5 предложений), учитывая:
1. Эмоциональное состояние пользователя
2. Релевантные воспоминания и опыт
3. Аналогии из прошлого
4. Цепочку рассуждений (если есть)
5. ТОЛЬКО реальные метрики (не придумывай свои)

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

    def _map_query_to_module(self, query_type: str) -> str:
        """Сопоставление типа запроса с модулем нейросети"""
        module_map = {
            'factual': 'memory',
            'creative': 'reasoning',
            'comparison': 'reasoning',
            'causal': 'reasoning',
            'prediction': 'reasoning',
            'general': 'perception',
        }
        return module_map.get(query_type, 'perception')

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
        """Проверка на прямую инструкцию"""
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

        neural_activation = context.get('neural_activation', 0.5)
        confidence += neural_activation * 0.1

        uncertain_markers = ['возможно', 'вероятно', 'может быть', 'кажется', 'наверное']
        for marker in uncertain_markers:
            if marker in response.lower():
                confidence -= 0.12

        return np.clip(confidence, 0.0, 1.0)

    def _calculate_importance(self, response: str, confidence: float,
                              emotion: EmotionalState) -> float:
        """Расчёт важности информации"""
        importance = 0.5

        importance += confidence * 0.2

        importance += abs(emotion.valence) * 0.15
        importance += emotion.arousal * 0.1

        word_count = len(response.split())
        importance += min(0.2, word_count / 200)

        return min(1.0, importance)

    def _extract_causal_links(self, user_input: str, response: str):
        """Извлечение причинно-следственных связей"""
        combined = f"{user_input} {response}".lower()

        because_pattern = r'(.+?)\s+(?:потому что|так как|из-за)\s+(.+?)(?:\.|,|$)'
        matches = re.findall(because_pattern, combined)

        for effect, cause in matches:
            effect = effect.strip()[:100]
            cause = cause.strip()[:100]

            if len(effect) > 10 and len(cause) > 10:
                self.causal_reasoning.add_causal_link(cause, effect, confidence=0.6)

    def get_status(self) -> Dict[str, Any]:
        """Полный статус системы"""
        neural_stats = self.neural_net.get_statistics()
        memory_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        cache_stats = self.llm.get_cache_stats()
        real_metrics = self.neural_net.get_real_metrics() if CONFIG.honest_metrics else {}

        return {
            'identity': {
                'user_id': self.user_id,
                'version': 'v31.1 FIXED (Honest Metrics)',
                'age': self._get_age_string(),
                'uptime': self._format_time_ago(self.birth_time),
            },
            'neural_network': neural_stats,
            'real_metrics': real_metrics,
            'memory': memory_stats,
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
        """Возраст сознания"""
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
    """🔧 ИСПРАВЛЕННЫЙ Telegram бот для Enhanced AGI Brain v31.1"""

    def __init__(self):
        self.llm: Optional[EnhancedSubconsciousLLM] = None
        self.brains: Dict[str, EnhancedAutonomousAGI] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        """Инициализация"""
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
            ('emotion', self._cmd_emotion),
            ('metrics', self._cmd_metrics),
            ('goals', self._cmd_goals),
            ('reset', self._cmd_reset),
            ('help', self._cmd_help),
        ]

        for cmd, handler in commands:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Enhanced AGI Bot v31.1 (FIXED) initialized")

    async def _get_or_create_brain(self, user_id: str) -> EnhancedAutonomousAGI:
        """Получить или создать мозг"""
        if user_id not in self.brains:
            brain = EnhancedAutonomousAGI(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
            logger.info(f"🆕 Enhanced AGI Brain v31.1 created for {user_id}")
        return self.brains[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
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
        """Команда /start"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        stats = brain.neural_net.get_statistics()
        mem_stats = brain.memory.get_statistics()

        message = f"""🧠 <b>ENHANCED AGI BRAIN v31.1 (FIXED)</b>

Привет, {update.effective_user.first_name}! 👋

Я — исправленная версия AGI с ЧЕСТНОЙ саморефлексией:

✅ <b>ИСПРАВЛЕНИЯ v31.1:</b>
• ТОЛЬКО реальные метрики (нет генерации чисел)
• Отслеживание источника сообщений
• Честное признание неопределённости
• Проверяемые утверждения

🧬 <b>Модульная нейросеть</b>
• {stats['neurons']['total']} нейронов в {len(stats['neurons']['by_module'])} модулях
• Реальные вычисления активации
• Meta-learning

📚 <b>Многоуровневая память</b>
• Working: {mem_stats['working_memory']} элементов
• Long-term: {mem_stats['long_term']} воспоминаний
• Episodes: {mem_stats['episodic']} событий

🎭 <b>Эмоциональный интеллект</b>
💭 <b>Улучшенная когнитивность</b>

⚡ <b>Возраст:</b> {brain._get_age_string()}

📌 Команды: /help
"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        status = brain.get_status()

        real_metrics = status.get('real_metrics', {})

        metrics_text = ""
        if real_metrics and CONFIG.honest_metrics:
            age = real_metrics.get('metrics_age_seconds', 0)
            metrics_text = f"""
<b>📊 Реальные метрики (возраст: {age:.0f}s):</b>
• perception: {real_metrics.get('perception', 0.0):.3f}
• reasoning: {real_metrics.get('reasoning', 0.0):.3f}
• memory: {real_metrics.get('memory', 0.0):.3f}
• action: {real_metrics.get('action', 0.0):.3f}
• meta: {real_metrics.get('meta', 0.0):.3f}
"""
        else:
            metrics_text = "\n<b>📊 Реальные метрики:</b> недоступны (нет недавнего forward pass)"

        message = f"""🧠 <b>STATUS v31.1 FIXED</b>

<b>🆔 Идентичность</b>
• Version: {status['identity']['version']}
• Возраст: {status['identity']['age']}

<b>🧬 Нейронная сеть</b>
• Всего: {status['neural_network']['neurons']['total']} нейронов
• Модулей: {len(status['neural_network']['neurons']['by_module'])}
• Синапсов: {status['neural_network']['synapses']['total']}
• Meta-learning: {status['neural_network']['activity']['meta_learning_score']:.3f}
{metrics_text}
<b>📚 Многоуровневая память</b>
• Working: {status['memory']['working_memory']}
• Short-term: {status['memory']['short_term']}
• Long-term: {status['memory']['long_term']}
• Episodes: {status['memory']['episodic']}

<b>🤔 Метакогниция</b>
• Avg uncertainty: {status['metacognition']['avg_uncertainty']:.2f}
• Avg confidence: {status['metacognition']['avg_confidence']:.2f}
• Questions asked: {status['metacognition']['questions_asked']}

<b>⚡ Производительность</b>
• Interactions: {status['performance']['interactions']}
• Avg confidence: {status['performance']['avg_confidence']:.2f}
• Best strategy: {status['performance']['best_strategy']}

<b>💾 LLM Cache</b>
• Hit rate: {status['llm_cache']['hit_rate']:.1%}"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_neural(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /neural"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.neural_net.get_statistics()
        real_metrics = brain.neural_net.get_real_metrics()

        modules_info = "\n".join([
            f"  • {name}: {count} neurons"
            for name, count in stats['neurons']['by_module'].items()
        ])

        real_metrics_info = ""
        if real_metrics and CONFIG.honest_metrics:
            age = real_metrics.get('metrics_age_seconds', 0)
            real_metrics_info = f"""
<b>📊 Текущая активация (возраст: {age:.0f}s):</b>"""
            for module in ['perception', 'reasoning', 'memory', 'action', 'meta']:
                val = real_metrics.get(module, 0.0)
                bars = '█' * int(val * 10)
                real_metrics_info += f"\n  {module}: {bars} {val:.3f}"
        else:
            real_metrics_info = "\n<b>📊 Реальные метрики недоступны</b> (нет недавнего forward pass)"

        message = f"""🧬 <b>МОДУЛЬНАЯ НЕЙРОСЕТЬ v31.1</b>

<b>📊 Нейроны</b>
• Total: {stats['neurons']['total']}

<b>🏗️ Модули</b>
{modules_info}

<b>🔗 Синапсы</b>
• Total: {stats['synapses']['total']}
• Avg strength: {stats['synapses']['avg_strength']:.3f}

<b>📈 Активность</b>
• Total activations: {stats['activity']['total_activations']:,}
• Meta-learning: {stats['activity']['meta_learning_score']:.3f}
{real_metrics_info}

<i>✅ Использую ТОЛЬКО реальные вычисленные метрики!</i>"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /memory"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        stats = brain.memory.get_statistics()

        working = list(brain.memory.working_memory)[-3:]
        working_text = "\n".join([f"  • {item.content[:60]}..." for item in working]) if working else "  Пусто"

        episodes = brain.memory.get_recent_episodes(n=3)
        episodes_text = "\n".join([
            f"  • {ep.context[:60]}..."
            for ep in episodes
        ]) if episodes else "  Нет"

        message = f"""📚 <b>МНОГОУРОВНЕВАЯ ПАМЯТЬ v31.1</b>

<b>📊 Статистика</b>
• Working: {stats['working_memory']}/{CONFIG.working_memory_size}
• Short-term: {stats['short_term']}
• Long-term: {stats['long_term']}
• Episodic: {stats['episodic']}

<b>🧠 Working Memory (последние)</b>
{working_text}

<b>📖 Недавние эпизоды</b>
{episodes_text}

<b>💡 Особенности:</b>
• Автоматическая консолидация
• Отслеживание источника (user/assistant)
• Семантический поиск"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_emotion(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /emotion"""
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
        """Команда /metrics"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        metrics = brain.metrics.get_metrics_summary()
        cache_stats = brain.llm.get_cache_stats()

        message = f"""📊 <b>МЕТРИКИ v31.1</b>

<b>⚡ Производительность</b>
• Взаимодействий: {metrics['interactions']}
• Error rate: {metrics['error_rate']:.1%}
• Avg response: {metrics['avg_response_time']:.2f}s
• Avg confidence: {metrics['avg_confidence']:.2f}

<b>🎯 Лучшая стратегия</b>
• {metrics['best_strategy']}

<b>💾 LLM Cache</b>
• Hit rate: {cache_stats['hit_rate']:.1%}
• Hits: {cache_stats['cache_hits']}
• Size: {cache_stats['cache_size']}

<i>✅ Все метрики РЕАЛЬНЫЕ и проверяемые!</i>"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_goals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /goals"""
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
                message += f"  [{progress_bar}] {goal.progress:.0%}\n\n"

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /reset"""
        user_id = str(update.effective_user.id)

        if context.args and context.args[0].lower() == 'confirm':
            if user_id in self.brains:
                await self.brains[user_id].stop()
                del self.brains[user_id]

            user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
            neural_path = CONFIG.base_dir / 'neural_nets' / f"{user_id}_v31.pkl.gz"

            import shutil
            if user_dir.exists():
                shutil.rmtree(user_dir)
            if neural_path.exists():
                neural_path.unlink()

            brain = await self._get_or_create_brain(user_id)
            stats = brain.neural_net.get_statistics()

            await update.message.reply_text(
                f"✅ <b>Полный сброс выполнен!</b>\n\n"
                f"Создано новое сознание v31.1 FIXED:\n"
                f"• Нейронов: {stats['neurons']['total']}\n"
                f"• Модулей: {len(stats['neurons']['by_module'])}\n"
                f"• Версия: v31.1 (Honest Metrics)",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                "⚠️ <b>ВНИМАНИЕ!</b>\n\n"
                "Это удалит всю память и нейросеть.\n\n"
                "Подтверждение:\n<code>/reset confirm</code>",
                parse_mode='HTML'
            )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        message = f"""🧠 <b>ENHANCED AGI BRAIN v31.1 FIXED</b>

<b>✅ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:</b>

<b>❌ БЫЛО:</b>
• Генерировал псевдометрики (0.46, 0.59...)
• Путаница self/other
• Цикл "признание → повтор ошибки"

<b>✅ СТАЛО:</b>
• ТОЛЬКО реальные вычисленные метрики
• Отслеживание источника сообщений
• Проверка перед выводом метрик
• Честное признание неопределённости

<b>🎯 ПРИНЦИПЫ v31.1:</b>
1. Честность > Правдоподобие
2. Если метрика не вычислена — не выводить
3. Явная атрибуция источника
4. Проверяемые утверждения

<b>📌 КОМАНДЫ:</b>
• /start — приветствие
• /status — полный статус (с реальными метриками)
• /neural — нейросеть + текущая активация
• /memory — уровни памяти
• /emotion — эмоциональная история
• /metrics — производительность
• /goals — активные цели
• /reset — сброс
• /help — эта справка

<b>💡 ОСОБЕННОСТЬ v31.1:</b>
Я показываю ТОЛЬКО те метрики, которые могу проверить.
Если не уверен — честно об этом скажу!

<b>🔬 КАК ПРОВЕРИТЬ:</b>
Спроси: "Какая у тебя активация модуля perception?"
Я покажу либо реальное значение с временной меткой,
либо честно скажу, что данных нет.</b>"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def start_polling(self):
        """Запуск бота"""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Enhanced AGI Bot v31.1 (FIXED) started")

    async def shutdown(self):
        """Остановка"""
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
    """Главная функция"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🧠 ENHANCED AGI BRAIN v31.1 - ИСПРАВЛЕННАЯ ВЕРСИЯ          ║
    ║     С ЧЕСТНОЙ САМОРЕФЛЕКСИЕЙ                                 ║
    ╚══════════════════════════════════════════════════════════════╝

    ✅ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ v31.1:

    ❌ ПРОБЛЕМА: Генерация псевдометрик (0.46, 0.59...)
    ✅ РЕШЕНИЕ: ТОЛЬКО реальные вычисления из нейросети

    ❌ ПРОБЛЕМА: Путаница self/other (кто написал числа)
    ✅ РЕШЕНИЕ: Отслеживание атрибуции каждого сообщения

    ❌ ПРОБЛЕМА: Цикл "признание → повтор ошибки"
    ✅ РЕШЕНИЕ: Проверка перед выводом любых метрик

    🎯 КЛЮЧЕВЫЕ ПРИНЦИПЫ:
    1. Честность > Правдоподобие
    2. Если метрика не вычислена — не выводить
    3. Явная атрибуция источника данных
    4. Проверяемые утверждения

    🧬 МОДУЛЬНАЯ НЕЙРОСЕТЬ:
    ✅ 5 специализированных модулей
    ✅ Реальные метрики активации
    ✅ Meta-learning

    📚 МНОГОУРОВНЕВАЯ ПАМЯТЬ:
    ✅ Working, Short-term, Long-term
    ✅ Episodic с контекстом
    ✅ Отслеживание источника (user/assistant)

    💭 КОГНИТИВНОСТЬ:
    ✅ Многошаговые рассуждения
    ✅ Аналогическое мышление
    ✅ Честная метакогниция

    🎭 ЭМОЦИОНАЛЬНЫЙ ИНТЕЛЛЕКТ:
    ✅ Распознавание эмоций
    ✅ Эмпатия
    """)

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = EnhancedAGIBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 ENHANCED AGI v31.1 FIXED АКТИВЕН")
        logger.info("✅ Честная саморефлексия")
        logger.info("✅ Реальные метрики")
        logger.info("✅ Проверяемые утверждения")
        logger.info("🛑 Ctrl+C для остановки\n")

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