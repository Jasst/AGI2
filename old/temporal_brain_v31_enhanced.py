#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 TEMPORAL COGNITIVE BRAIN v31.0 — ENHANCED INTELLIGENCE & ADAPTIVITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 НОВЫЕ РЕВОЛЮЦИОННЫЕ УЛУЧШЕНИЯ V31.0:

🎯 МНОГОУРОВНЕВАЯ ПАМЯТЬ:
  ✅ Working Memory (оперативная, 7±2 элемента)
  ✅ Short-Term Memory (кратковременная, decay со временем)
  ✅ Long-Term Memory (долговременная, консолидация важного)
  ✅ Episodic Memory (эпизодическая, события с контекстом)
  ✅ Semantic Memory (семантическая, факты и концепты)

🧬 УЛУЧШЕННАЯ НЕЙРОСЕТЬ:
  ✅ Attention mechanism (внимание к важным связям)
  ✅ Модульная архитектура (специализированные подсети)
  ✅ Transfer learning (перенос знаний между модулями)
  ✅ Meta-learning (обучение обучаться)
  ✅ Dropout для регуляризации

🎓 ПРОГРЕССИВНОЕ ОБУЧЕНИЕ:
  ✅ Curriculum learning (от простого к сложному)
  ✅ Active learning (запрос меток для неопределённых случаев)
  ✅ Few-shot learning (обучение на малых данных)
  ✅ Continual learning (без катастрофического забывания)

💡 РАСШИРЕННАЯ КОГНИТИВНОСТЬ:
  ✅ Multi-step reasoning (цепочки рассуждений)
  ✅ Analogical thinking (мышление по аналогии)
  ✅ Causal inference (причинно-следственные связи)
  ✅ Counterfactual reasoning (что если бы...)
  ✅ Theory of Mind (модель чужого сознания)

🔍 УЛУЧШЕННАЯ МЕТАКОГНИЦИЯ:
  ✅ Confidence calibration (калибровка уверенности)
  ✅ Error detection & correction (обнаружение ошибок)
  ✅ Strategy selection (выбор стратегии решения)
  ✅ Self-explanation (объяснение своих решений)

🌊 ЭМОЦИОНАЛЬНЫЙ ИНТЕЛЛЕКТ:
  ✅ Emotion recognition (распознавание эмоций)
  ✅ Empathy modeling (моделирование эмпатии)
  ✅ Emotional regulation (регуляция эмоций)
  ✅ Mood tracking (отслеживание настроения)

🎯 ЦЕЛЕПОЛАГАНИЕ И ПЛАНИРОВАНИЕ:
  ✅ Goal decomposition (декомпозиция целей)
  ✅ Hierarchical planning (иерархическое планирование)
  ✅ Progress tracking (отслеживание прогресса)
  ✅ Plan adjustment (корректировка планов)

📊 АНАЛИТИКА И ОПТИМИЗАЦИЯ:
  ✅ Performance metrics (метрики производительности)
  ✅ A/B testing стратегий
  ✅ Adaptive hyperparameters (адаптивные гиперпараметры)
  ✅ Real-time optimization (оптимизация в реальном времени)

🔗 КОНТЕКСТНОЕ ПОНИМАНИЕ:
  ✅ Conversation threading (нити разговора)
  ✅ Context switching (переключение контекста)
  ✅ Topic modeling (моделирование тем)
  ✅ Intent recognition (распознавание намерений)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ═══════════════════════════════════════════════════════════════
# 📋 КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════
load_dotenv()


@dataclass
class EnhancedConfig:
    """Расширенная конфигурация v31.0"""
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    # Режимы
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # Нейронная сеть v31
    initial_neurons: int = 150
    max_neurons: int = 15000
    neurogenesis_threshold: float = 0.75
    pruning_threshold: float = 0.015
    learning_rate: float = 0.12
    attention_heads: int = 4  # Для attention mechanism
    dropout_rate: float = 0.1

    # Многоуровневая память
    working_memory_size: int = 7  # 7±2 правило Миллера
    short_term_size: int = 100
    short_term_decay: float = 0.95  # Decay rate
    long_term_size: int = 10000
    consolidation_threshold: float = 0.7  # Порог для консолидации в LTM
    
    # Эпизодическая память
    episodic_context_window: int = 5  # Кол-во сообщений в эпизоде
    
    # Когнитивные параметры v31
    curiosity_threshold: float = 0.55
    question_cooldown: int = 90
    max_autonomous_questions: int = 5
    reasoning_depth: int = 3  # Глубина цепочек рассуждений
    analogy_threshold: float = 0.6  # Порог для аналогий
    
    # Метакогниция v31
    confidence_calibration_samples: int = 50
    error_correction_threshold: float = 0.3
    
    # Эмоциональный интеллект
    emotion_tracking: bool = True
    empathy_weight: float = 0.3
    
    # Векторная память
    embedding_dim: int = 512  # Увеличена размерность
    semantic_cache_size: int = 2000

    # Временные интервалы (секунды)
    spontaneous_thought_interval: int = 150
    reflection_interval: int = 600
    consolidation_interval: int = 1200
    save_interval: int = 240
    neural_optimization_interval: int = 450
    metrics_update_interval: int = 180

    # Пути
    base_dir: Path = Path(os.getenv('BASE_DIR', 'temporal_brain_v31'))

    def __post_init__(self):
        for subdir in ['memory', 'neural_nets', 'knowledge', 'cache', 'logs', 
                       'backups', 'episodic', 'analytics', 'goals']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)


CONFIG = EnhancedConfig()


# ═══════════════════════════════════════════════════════════════
# 🪵 ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# 🎭 ЭМОЦИОНАЛЬНЫЙ АНАЛИЗАТОР
# ═══════════════════════════════════════════════════════════════
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
    valence: float  # -1 (негативное) до +1 (позитивное)
    arousal: float  # 0 (спокойное) до 1 (возбуждённое)
    confidence: float
    timestamp: float = field(default_factory=time.time)


class EmotionalIntelligence:
    """Модуль эмоционального интеллекта"""
    
    def __init__(self):
        self.emotion_history: deque = deque(maxlen=100)
        self.user_emotion_model: Dict[str, List[EmotionalState]] = defaultdict(list)
        
        # Словари для распознавания эмоций (упрощённая версия)
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
        
        # Подсчёт эмоциональных маркеров
        emotion_scores = defaultdict(float)
        
        for emotion, lexicon in self.emotion_lexicons.items():
            for word in lexicon:
                if word in text_lower:
                    emotion_scores[emotion] += 1.0
        
        # Анализ пунктуации
        if '!' in text:
            emotion_scores[EmotionType.EXCITEMENT] += 0.5
        if '?' in text:
            emotion_scores[EmotionType.CURIOSITY] += 0.3
        if text.isupper():
            emotion_scores[EmotionType.ANGER] += 0.5
        
        # Определение доминирующей эмоции
        if emotion_scores:
            dominant = max(emotion_scores.items(), key=lambda x: x[1])
            emotion, score = dominant
            confidence = min(1.0, score / 3.0)
        else:
            emotion = EmotionType.NEUTRAL
            confidence = 0.7
        
        # Расчёт valence и arousal
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
        
        # Модификация на основе интенсивности
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
        
        # Модификация на основе пунктуации
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


# ═══════════════════════════════════════════════════════════════
# 🧠 МНОГОУРОВНЕВАЯ ПАМЯТЬ
# ═══════════════════════════════════════════════════════════════
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
        
        # Working Memory (оперативная) - 7±2 элемента
        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)
        
        # Short-Term Memory (кратковременная) - с decay
        self.short_term_memory: List[MemoryItem] = []
        
        # Long-Term Memory (долговременная)
        self.long_term_memory: Dict[str, MemoryItem] = {}
        
        # Episodic Memory (эпизодическая)
        self.episodic_memory: Dict[str, Episode] = {}
        self.current_episode_buffer: List[Dict] = []
        
        # Semantic Memory (семантическая - факты, концепты)
        self.semantic_memory: Dict[str, MemoryItem] = {}
        
        self.last_consolidation = time.time()
    
    def add_to_working(self, content: str, importance: float = 0.5):
        """Добавление в оперативную память"""
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            embedding=self.embedding_func(content)
        )
        self.working_memory.append(item)
        
        # Автоматически переносим в STM
        self.short_term_memory.append(item)
    
    def get_working_memory_context(self) -> str:
        """Получение контекста из оперативной памяти"""
        return "\n".join([item.content for item in self.working_memory])
    
    def decay_short_term(self):
        """Применение decay к кратковременной памяти"""
        current_time = time.time()
        
        # Удаляем старые и неважные элементы
        self.short_term_memory = [
            item for item in self.short_term_memory
            if (current_time - item.timestamp < 3600 or  # Меньше часа
                item.importance > 0.7 or  # Важные
                item.access_count > 3)  # Часто используемые
        ]
        
        # Ограничиваем размер
        if len(self.short_term_memory) > CONFIG.short_term_size:
            # Сортируем по важности и оставляем топ
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
        
        # Ограничиваем размер
        if len(self.episodic_memory) > 500:
            # Удаляем старые неважные эпизоды
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
        
        # Поиск в LTM
        for mem_id, item in self.long_term_memory.items():
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                results.append((item.content, float(similarity), 'LTM'))
                item.access()
        
        # Поиск в STM
        for item in self.short_term_memory:
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                results.append((item.content, float(similarity), 'STM'))
                item.access()
        
        # Сортировка по релевантности
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


# ═══════════════════════════════════════════════════════════════
# 🧬 УЛУЧШЕННАЯ НЕЙРОННАЯ СЕТЬ С ATTENTION
# ═══════════════════════════════════════════════════════════════
@dataclass
class EnhancedNeuron:
    """Улучшенный нейрон с модульностью"""
    id: str
    layer: int
    module: str  # perception, reasoning, memory, action, meta
    activation: float = 0.0
    bias: float = 0.0
    neuron_type: str = "general"
    created_at: float = field(default_factory=time.time)
    activation_count: int = 0
    specialization: Optional[str] = None
    importance_score: float = 0.5  # Для attention
    
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
    attention_weight: float = 1.0  # Для attention mechanism
    
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


class EnhancedNeuralNetwork:
    """Улучшенная нейросеть с модульностью и attention"""
    
    def __init__(self, initial_neurons: int = 150):
        self.neurons: Dict[str, EnhancedNeuron] = {}
        self.synapses: Dict[Tuple[str, str], EnhancedSynapse] = {}
        self.layers: Dict[int, Set[str]] = defaultdict(set)
        self.modules: Dict[str, Set[str]] = defaultdict(set)  # Модули
        
        # Attention
        self.attention_heads = CONFIG.attention_heads
        
        # Метрики
        self.total_activations = 0
        self.neurogenesis_events = 0
        self.pruning_events = 0
        self.meta_learning_score = 0.5
        
        self._initialize_modular_network(initial_neurons)
        logger.info(f"🧬 Enhanced Neural Network: {len(self.neurons)} neurons, "
                   f"{len(self.modules)} modules")
    
    def _initialize_modular_network(self, n: int):
        """Инициализация модульной архитектуры"""
        modules_config = {
            'perception': n // 5,  # Восприятие
            'reasoning': n // 5,   # Рассуждение
            'memory': n // 5,      # Память
            'action': n // 5,      # Действие
            'meta': n // 5,        # Метакогниция
        }
        
        neuron_counter = 0
        
        for module_name, module_size in modules_config.items():
            # Каждый модуль имеет 4 слоя
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
        
        # Создание синапсов внутри модулей (dense)
        for module_name, module_neurons in self.modules.items():
            self._connect_within_module(module_neurons)
        
        # Межмодульные связи (sparse)
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
        
        # Связи между соседними слоями
        for layer in range(3):
            source_neurons = neurons_by_layer[layer]
            target_neurons = neurons_by_layer[layer + 1]
            
            for source_id in source_neurons:
                # Каждый нейрон связывается с 30-50% следующего слоя
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
            
            # Sparse connections (5-10%)
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
        """Прямой проход с attention mechanism"""
        # Сброс активаций
        for neuron in self.neurons.values():
            neuron.activation = 0.0
        
        # Инициализация входного слоя
        input_neurons = [nid for nid in self.layers[0] 
                        if target_module is None or self.neurons[nid].module == target_module]
        
        for i, neuron_id in enumerate(input_neurons):
            if i < len(input_vector):
                self.neurons[neuron_id].activation = input_vector[i]
        
        # Прямой проход с attention
        max_layer = max(self.layers.keys())
        
        for layer in range(1, max_layer + 1):
            layer_neurons = self.layers[layer]
            
            for target_id in layer_neurons:
                # Собираем входы от всех синапсов
                inputs = []
                attention_weights = []
                
                for (source_id, tid), synapse in self.synapses.items():
                    if tid == target_id:
                        source = self.neurons[source_id]
                        inputs.append(source.activation * synapse.weight)
                        attention_weights.append(synapse.attention_weight * source.importance_score)
                
                if inputs:
                    # Применение attention
                    if len(attention_weights) > 1:
                        attention_weights = softmax(attention_weights)
                        weighted_input = sum(i * a for i, a in zip(inputs, attention_weights))
                    else:
                        weighted_input = sum(inputs)
                    
                    # Активация с dropout
                    self.neurons[target_id].activate(weighted_input, CONFIG.dropout_rate)
        
        # Извлечение выхода
        output_neurons = list(self.layers[max_layer])
        output = np.array([self.neurons[nid].activation for nid in output_neurons])
        
        self.total_activations += 1
        return output
    
    def meta_learning_update(self, performance_score: float):
        """Meta-learning: обучение обучаться"""
        # Обновляем learning rate на основе производительности
        if performance_score > 0.8:
            CONFIG.learning_rate *= 1.05  # Увеличиваем
        elif performance_score < 0.5:
            CONFIG.learning_rate *= 0.95  # Уменьшаем
        
        CONFIG.learning_rate = np.clip(CONFIG.learning_rate, 0.01, 0.3)
        
        self.meta_learning_score = 0.9 * self.meta_learning_score + 0.1 * performance_score
        
        logger.debug(f"🎓 Meta-learning: LR={CONFIG.learning_rate:.4f}, "
                    f"Score={self.meta_learning_score:.3f}")
    
    def transfer_learning(self, source_module: str, target_module: str, strength: float = 0.5):
        """Transfer learning между модулями"""
        source_neurons = self.modules[source_module]
        target_neurons = self.modules[target_module]
        
        # Создаём дополнительные связи для переноса знаний
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


# ═══════════════════════════════════════════════════════════════
# 🤔 РАСШИРЕННАЯ МЕТАКОГНИЦИЯ
# ═══════════════════════════════════════════════════════════════
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
        
        # Калибровка уверенности
        self.calibration_data: List[Tuple[float, float]] = []  # (predicted_conf, actual_accuracy)
    
    def assess_uncertainty(self, context: Dict) -> Tuple[float, List[str]]:
        """Оценка неопределённости с объяснением причин"""
        uncertainty = 0.3
        reasons = []
        
        # 1. Недостаток данных
        memory_count = context.get('memory_count', 0)
        if memory_count < 10:
            uncertainty += 0.25
            reasons.append("недостаточно данных в памяти")
        
        # 2. Противоречивая информация
        if context.get('conflicting_info', False):
            uncertainty += 0.3
            reasons.append("обнаружены противоречия")
        
        # 3. Низкая знакомость с темой
        familiarity = context.get('topic_familiarity', 0.5)
        if familiarity < 0.3:
            uncertainty += 0.25
            reasons.append("новая тема")
        
        # 4. Сложность запроса
        complexity = context.get('query_complexity', 0.5)
        if complexity > 0.7:
            uncertainty += 0.2
            reasons.append("сложный вопрос")
        
        # 5. Эмоциональная неопределённость
        if context.get('emotional_ambiguity', False):
            uncertainty += 0.15
            reasons.append("неоднозначный эмоциональный контекст")
        
        uncertainty = min(1.0, uncertainty)
        self.uncertainty_log.append(uncertainty)
        
        return uncertainty, reasons
    
    def calibrate_confidence(self, predicted: float, actual: float):
        """Калибровка уверенности на основе фактических результатов"""
        self.calibration_data.append((predicted, actual))
        
        # Ограничиваем размер
        if len(self.calibration_data) > CONFIG.confidence_calibration_samples:
            self.calibration_data = self.calibration_data[-CONFIG.confidence_calibration_samples:]
    
    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """Получение откалиброванной уверенности"""
        if len(self.calibration_data) < 10:
            return raw_confidence
        
        # Простая калибровка на основе исторических данных
        predictions, actuals = zip(*self.calibration_data)
        
        avg_predicted = np.mean(predictions)
        avg_actual = np.mean(actuals)
        
        # Корректировка
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
        # Простая эвристическая проверка
        for i, step in enumerate(reasoning_chain):
            # Проверка на противоречия
            if i > 0:
                prev_step = reasoning_chain[i-1]
                
                # Ищем отрицания
                negations = {'не', 'нет', 'ни', 'отсутствие'}
                
                prev_has_neg = any(neg in prev_step.lower() for neg in negations)
                curr_has_neg = any(neg in step.lower() for neg in negations)
                
                # Простая проверка на противоречие
                if prev_has_neg != curr_has_neg:
                    # Проверяем, говорят ли они об одном и том же
                    prev_words = set(prev_step.lower().split())
                    curr_words = set(step.lower().split())
                    
                    common = prev_words & curr_words
                    if len(common) > 3:  # Значимое пересечение
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
            not context.get('explicit_instruction', False)  # Не спрашиваем при прямых инструкциях
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


# Продолжение в следующем файле из-за ограничения размера...