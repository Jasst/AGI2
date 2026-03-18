#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ENHANCED AGI BRAIN v33.2 — UNIFIED ADAPTIVE ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 ОБЪЕДИНЕНИЕ ЛУЧШЕГО ИЗ ВСЕХ ВЕРСИЙ:

✅ ОТ v33.0 (Honest Architecture):
   • Честные метрики на основе реальных данных
   • Проверка консистентности ответов
   • Упрощенная система весов для отслеживания интересов
   • Валидация ответов (запрет выдуманных чисел)
   • Извлечение и отслеживание интересов пользователя

✅ ОТ v32.0 (Full Optimization):
   • Модульная архитектура нейросети
   • Кросс-модульная синергия
   • Улучшенная консолидация памяти с приоритетом качества
   • Адаптивный learning rate
   • Балансировка модулей

✅ ОТ v33.1 (Adaptive Neural):
   • 🔥 НАСТОЯЩАЯ нейросеть с backpropagation
   • Обучаемые эмбеддинги (AdaptiveEmbedding)
   • Online learning в реальном времени
   • Персонализация под каждого пользователя
   • Предсказание метрик качества

🎯 ПРИНЦИПЫ v33.2:
1. Честность > Имитация
2. Реальное обучение > Статичные веса
3. Проверяемость > Сложность
4. Консистентность > Креативность
5. Адаптивность > Жёсткая архитектура
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


# ═══════════════════════════════════════════════════════════════
# 🔧 КОНФИГУРАЦИЯ v33.2
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnhancedConfig:
    """Унифицированная конфигурация v33.2"""
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🔧 Честные метрики и валидация (v33.0)
    honest_metrics: bool = True
    validate_responses: bool = True
    check_consistency: bool = True
    show_uncertainty: bool = True

    # 🔧 Нейросеть: Адаптивная архитектура (v33.1 + v32.0)
    embedding_dim: int = 64  # Для адаптивных эмбеддингов
    hidden_dim: int = 32
    output_metrics_dim: int = 5  # confidence, complexity, relevance, coherence, engagement
    vocab_size: int = 5000

    # Параметры обучения (Adam optimizer)
    learning_rate: float = 0.001
    min_learning_rate: float = 0.0001
    max_learning_rate: float = 0.01
    lr_adaptation_rate: float = 0.05

    # Динамическое расширение (концепция из v32, адаптирована)
    plateau_detection_cycles: int = 3
    plateau_threshold: float = 0.02
    expansion_rate: float = 0.1
    max_vocab_expansion: int = 10000

    # Кросс-модульная синергия (v32.0)
    cross_module_synergy: bool = True
    synergy_strength: float = 0.15
    module_balance_target: float = 0.5

    # Память (улучшенная из v32.0)
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
    semantic_cache_size: int = 2000

    # Интервалы автономной активности
    spontaneous_thought_interval: int = 150
    reflection_interval: int = 600
    consolidation_interval: int = 1200
    save_interval: int = 240
    goal_update_interval: int = 300
    neural_training_interval: int = 1  # Обучать после каждого взаимодействия

    # Вариативность мыслей (v32.0)
    thought_temperature_base: float = 0.9
    thought_temperature_variance: float = 0.2
    thought_diversity_penalty: float = 0.3

    # Честные метрики — лимиты для вычислений (v33.0)
    max_input_words_for_perception: int = 50
    max_reasoning_steps_for_metric: int = 5
    max_memory_items_for_metric: int = 10
    max_response_words_for_action: int = 100

    # Пути
    base_dir: Path = Path(os.getenv('BASE_DIR', 'temporal_brain_v33_2'))

    def __post_init__(self):
        for subdir in ['memory', 'neural_nets', 'knowledge', 'cache', 'logs',
                       'backups', 'episodic', 'analytics', 'goals']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)


CONFIG = EnhancedConfig()


# ═══════════════════════════════════════════════════════════════
# 🎨 ЛОГИРОВАНИЕ С ЦВЕТАМИ
# ═══════════════════════════════════════════════════════════════

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('Enhanced_AGI_v33_2')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))

    log_file = CONFIG.base_dir / 'logs' / f'agi_v33_2_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════
# 🧠 АДАПТИВНАЯ НЕЙРОСЕТЬ (v33.1 — REAL BACKPROPAGATION)
# ═══════════════════════════════════════════════════════════════

class AdaptiveEmbedding:
    """
    Обучаемый слой эмбеддингов для концептов

    Вместо фиксированной хэш-функции, учим представления слов,
    которые адаптируются к контексту использования
    """

    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.next_idx = 0
        self.m = np.zeros_like(self.embeddings)  # Adam moments
        self.v = np.zeros_like(self.embeddings)
        self.t = 0

    def add_word(self, word: str) -> int:
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        if self.next_idx >= self.vocab_size:
            return 0  # Перезаписываем редкие при переполнении
        idx = self.next_idx
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        self.next_idx += 1
        return idx

    def get_embedding(self, word: str) -> np.ndarray:
        if word not in self.word_to_idx:
            idx = self.add_word(word)
        else:
            idx = self.word_to_idx[word]
        return self.embeddings[idx].copy()

    def encode_text(self, text: str) -> np.ndarray:
        words = text.lower().split()
        if not words:
            return np.zeros(self.embedding_dim)
        embeddings = [self.get_embedding(word) for word in words]
        return np.mean(embeddings, axis=0)

    def update_embeddings(self, word: str, gradient: np.ndarray, lr: float = 0.001):
        if word not in self.word_to_idx:
            return
        idx = self.word_to_idx[word]
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m[idx] = beta1 * self.m[idx] + (1 - beta1) * gradient
        self.v[idx] = beta2 * self.v[idx] + (1 - beta2) * (gradient ** 2)
        m_hat = self.m[idx] / (1 - beta1 ** self.t)
        v_hat = self.v[idx] / (1 - beta2 ** self.t)
        self.embeddings[idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class AdaptiveNeuralModule:
    """
    Адаптивная нейросеть с настоящим обучением

    Архитектура: Input(64) -> Hidden(32, ReLU) -> Output(5, Sigmoid)

    Выходные метрики:
    - confidence: уверенность в ответе [0, 1]
    - complexity: сложность запроса [0, 1]
    - relevance: релевантность памяти [0, 1]
    - coherence: согласованность ответа [0, 1]
    - engagement: вовлечённость пользователя [0, 1]
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, output_dim: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # Adam optimizer state
        self._init_adam_state()
        self.t = 0
        self.cache = {}
        self.training_history: deque = deque(maxlen=100)
        self.total_updates = 0

    def _init_adam_state(self):
        self.m_W1 = np.zeros_like(self.W1);
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1);
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2);
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2);
        self.v_b2 = np.zeros_like(self.b2)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray, store_cache: bool = True) -> np.ndarray:
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        if store_cache:
            self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2

    def backward(self, target: np.ndarray, lr: float = 0.001) -> float:
        if not self.cache:
            raise ValueError("Forward pass required before backward")

        x, z1, a1, a2 = self.cache['x'], self.cache['z1'], self.cache['a1'], self.cache['a2']
        loss = np.mean((a2 - target) ** 2)

        # Gradients
        dz2 = 2 * (a2 - target) * a2 * (1 - a2)
        dW2 = a1[:, np.newaxis] @ dz2[np.newaxis, :]
        db2 = dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(z1)
        dW1 = x[:, np.newaxis] @ dz1[np.newaxis, :]
        db1 = dz1

        # Adam update
        self._adam_update('W1', dW1, lr);
        self._adam_update('b1', db1, lr)
        self._adam_update('W2', dW2, lr);
        self._adam_update('b2', db2, lr)

        self.training_history.append(loss)
        self.total_updates += 1
        return loss

    def _adam_update(self, param: str, grad: np.ndarray, lr: float):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        for prefix in ['m_', 'v_']:
            attr = f"{prefix}{param}"
            if prefix == 'm_':
                setattr(self, attr, beta1 * getattr(self, attr) + (1 - beta1) * grad)
            else:
                setattr(self, attr, beta2 * getattr(self, attr) + (1 - beta2) * (grad ** 2))

        m_hat = getattr(self, f'm_{param}') / (1 - beta1 ** self.t)
        v_hat = getattr(self, f'v_{param}') / (1 - beta2 ** self.t)
        setattr(self, param, getattr(self, param) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def get_statistics(self) -> Dict:
        return {
            'total_updates': self.total_updates,
            'recent_loss': float(np.mean(self.training_history)) if self.training_history else 0.0,
            'loss_std': float(np.std(self.training_history)) if self.training_history else 0.0,
        }


class AdaptiveNeuralEngine:
    """
    Полный движок адаптивной нейросети

    Объединяет:
    - AdaptiveEmbedding (обучаемые представления)
    - AdaptiveNeuralModule (предсказание метрик)
    - Online learning (обучение на каждом взаимодействии)
    - Кросс-модульная синергия (адаптировано из v32.0)
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding = AdaptiveEmbedding(embedding_dim=embedding_dim)
        self.neural = AdaptiveNeuralModule(input_dim=embedding_dim)
        self.interaction_history: List[Dict] = []
        self.module_synergy_bonus: Dict[str, float] = defaultdict(float)
        self.current_learning_rate = CONFIG.learning_rate

    def predict_metrics(self, user_input: str, context: str = "") -> Dict[str, float]:
        combined_text = f"{user_input} {context}"
        embedding_vector = self.embedding.encode_text(combined_text)
        # Применяем синергию
        for module, bonus in self.module_synergy_bonus.items():
            if bonus > 0:
                embedding_vector = embedding_vector * (1 + bonus * 0.1)
        predictions = self.neural.forward(embedding_vector, store_cache=False)
        return {
            'confidence': float(predictions[0]), 'complexity': float(predictions[1]),
            'relevance': float(predictions[2]), 'coherence': float(predictions[3]),
            'engagement': float(predictions[4]),
        }

    def learn_from_interaction(self, user_input: str, response: str,
                               actual_metrics: Dict[str, float], context: str = "",
                               lr: Optional[float] = None) -> float:
        combined_text = f"{user_input} {context}"
        embedding_vector = self.embedding.encode_text(combined_text)
        predictions = self.neural.forward(embedding_vector, store_cache=True)

        target = np.array([
            actual_metrics.get('confidence', 0.5), actual_metrics.get('complexity', 0.5),
            actual_metrics.get('relevance', 0.5), actual_metrics.get('coherence', 0.5),
            actual_metrics.get('engagement', 0.5),
        ])

        loss = self.neural.backward(target, lr or self.current_learning_rate)

        # Обновляем эмбеддинги ключевых слов
        for word in user_input.lower().split():
            if len(word) > 3 and word not in {'это', 'для', 'как', 'что', 'когда', 'где'}:
                # Простой градиент: разница между предсказанием и целью
                grad = (predictions - target).mean() * self.embedding.get_embedding(word) * 0.01
                self.embedding.update_embeddings(word, grad, lr or self.current_learning_rate)

        self.interaction_history.append({
            'user_input': user_input, 'response': response,
            'actual_metrics': actual_metrics,
            'predicted_metrics': {k: float(v) for k, v in zip(
                ['confidence', 'complexity', 'relevance', 'coherence', 'engagement'], predictions)},
            'loss': loss, 'timestamp': time.time()
        })
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]

        # 🔧 Адаптивный learning rate (из v32.0)
        self._adapt_learning_rate(loss)
        return loss

    def _adapt_learning_rate(self, loss: float):
        """Адаптация learning rate на основе качества обучения"""
        if len(self.neural.training_history) < 10:
            return
        recent_losses = list(self.neural.training_history)[-10:]
        trend = np.mean(recent_losses[-5:]) - np.mean(recent_losses[:5])

        if trend < -0.01:  # Улучшение — можно ускорить обучение
            self.current_learning_rate = min(CONFIG.max_learning_rate,
                                             self.current_learning_rate * (1 + CONFIG.lr_adaptation_rate))
        elif trend > 0.01:  # Ухудшение — замедлить
            self.current_learning_rate = max(CONFIG.min_learning_rate,
                                             self.current_learning_rate * (1 - CONFIG.lr_adaptation_rate))

    def apply_cross_module_synergy(self, active_module: str, activation: float):
        """🔧 Кросс-модульная синергия (из v32.0)"""
        if not CONFIG.cross_module_synergy:
            return
        bonus = CONFIG.synergy_strength * activation
        for module in ['confidence', 'complexity', 'relevance', 'coherence', 'engagement']:
            if module != active_module:
                self.module_synergy_bonus[module] = min(0.5,
                                                        self.module_synergy_bonus[module] + bonus * 0.3)

    def get_statistics(self) -> Dict:
        return {
            'embedding': {'vocab_size': self.embedding.next_idx, 'embedding_dim': self.embedding.embedding_dim},
            'neural': self.neural.get_statistics(),
            'interactions': len(self.interaction_history),
            'current_lr': self.current_learning_rate,
            'synergy_bonuses': dict(self.module_synergy_bonus),
        }

    def save(self, path: Path):
        state = {
            'embedding_matrix': self.embedding.embeddings,
            'word_to_idx': self.embedding.word_to_idx, 'idx_to_word': self.embedding.idx_to_word,
            'next_idx': self.embedding.next_idx,
            'W1': self.neural.W1, 'b1': self.neural.b1, 'W2': self.neural.W2, 'b2': self.neural.b2,
            'm_W1': self.neural.m_W1, 'v_W1': self.neural.v_W1,
            'm_b1': self.neural.m_b1, 'v_b1': self.neural.v_b1,
            'm_W2': self.neural.m_W2, 'v_W2': self.neural.v_W2,
            'm_b2': self.neural.m_b2, 'v_b2': self.neural.v_b2,
            'neural_t': self.neural.t, 'interaction_history': self.interaction_history,
            'synergy_bonus': dict(self.module_synergy_bonus), 'current_lr': self.current_learning_rate,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb', compresslevel=6) as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)
            self.embedding.embeddings = state['embedding_matrix']
            self.embedding.word_to_idx = state['word_to_idx']
            self.embedding.idx_to_word = state['idx_to_word']
            self.embedding.next_idx = state['next_idx']
            self.neural.W1, self.neural.b1 = state['W1'], state['b1']
            self.neural.W2, self.neural.b2 = state['W2'], state['b2']
            self.neural.m_W1, self.neural.v_W1 = state['m_W1'], state['v_W1']
            self.neural.m_b1, self.neural.v_b1 = state['m_b1'], state['v_b1']
            self.neural.m_W2, self.neural.v_W2 = state['m_W2'], state['v_W2']
            self.neural.m_b2, self.neural.v_b2 = state['m_b2'], state['v_b2']
            self.neural.t = state.get('neural_t', 0)
            self.interaction_history = state.get('interaction_history', [])
            self.module_synergy_bonus = defaultdict(float, state.get('synergy_bonus', {}))
            self.current_learning_rate = state.get('current_lr', CONFIG.learning_rate)
            return True
        except Exception as e:
            logger.error(f"Error loading neural state: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# 📦 ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ (v33.0 + v32.0)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MessageAttribution:
    role: str;
    content: str;
    timestamp: float
    contains_metrics: bool = False
    metric_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class EmotionType(Enum):
    JOY = auto();
    SADNESS = auto();
    ANGER = auto();
    FEAR = auto()
    SURPRISE = auto();
    DISGUST = auto();
    NEUTRAL = auto()
    CURIOSITY = auto();
    EXCITEMENT = auto()


@dataclass
class EmotionalState:
    dominant_emotion: EmotionType
    valence: float;
    arousal: float;
    confidence: float
    timestamp: float = field(default_factory=time.time)


class EmotionalIntelligence:
    def __init__(self):
        self.emotion_history: deque = deque(maxlen=100)
        self.user_emotion_model: Dict[str, List[EmotionalState]] = defaultdict(list)
        self.emotion_lexicons = {
            EmotionType.JOY: {'рад', 'счастлив', 'отлично', 'замечательно', 'восторг', 'прекрасно', 'весело', 'ура',
                              '😊', '😄', '🎉'},
            EmotionType.SADNESS: {'грустно', 'печально', 'тоскливо', 'плохо', 'депрессия', 'уныло', 'слёзы', '😢', '😭',
                                  '☹️'},
            EmotionType.ANGER: {'злой', 'бесит', 'раздражает', 'ярость', 'гнев', 'ненавижу', 'достало', '😡', '😠', '🤬'},
            EmotionType.FEAR: {'боюсь', 'страшно', 'тревога', 'паника', 'волнуюсь', 'переживаю', '😨', '😰', '😱'},
            EmotionType.SURPRISE: {'вау', 'ого', 'неожиданно', 'удивительно', 'шок', '😮', '😲', '🤯'},
            EmotionType.CURIOSITY: {'интересно', 'любопытно', 'хочу узнать', 'расскажи', 'а что', 'почему', '🤔'},
        }

    def analyze_emotion(self, text: str, context: Optional[str] = None) -> EmotionalState:
        text_lower = text.lower()
        emotion_scores = defaultdict(float)
        for emotion, lexicon in self.emotion_lexicons.items():
            for word in lexicon:
                if word in text_lower: emotion_scores[emotion] += 1.0
        if '!' in text: emotion_scores[EmotionType.EXCITEMENT] += 0.5
        if '?' in text: emotion_scores[EmotionType.CURIOSITY] += 0.3
        if text.isupper(): emotion_scores[EmotionType.ANGER] += 0.5

        if emotion_scores:
            dominant = max(emotion_scores.items(), key=lambda x: x[1])
            emotion, score = dominant
            confidence = min(1.0, score / 3.0)
        else:
            emotion, confidence = EmotionType.NEUTRAL, 0.7

        valence_map = {EmotionType.JOY: 0.8, EmotionType.EXCITEMENT: 0.7, EmotionType.CURIOSITY: 0.3,
                       EmotionType.SURPRISE: 0.2, EmotionType.NEUTRAL: 0.0, EmotionType.SADNESS: -0.6,
                       EmotionType.FEAR: -0.5, EmotionType.ANGER: -0.7, EmotionType.DISGUST: -0.8}
        valence = np.clip(valence_map.get(emotion, 0.0), -1.0, 1.0)

        arousal_map = {EmotionType.EXCITEMENT: 0.9, EmotionType.ANGER: 0.8, EmotionType.FEAR: 0.8,
                       EmotionType.SURPRISE: 0.7, EmotionType.JOY: 0.6, EmotionType.CURIOSITY: 0.5,
                       EmotionType.SADNESS: 0.3, EmotionType.NEUTRAL: 0.2, EmotionType.DISGUST: 0.4}
        arousal = np.clip(arousal_map.get(emotion, 0.5) + text.count('!') * 0.1, 0.0, 1.0)

        state = EmotionalState(dominant_emotion=emotion, valence=valence, arousal=arousal, confidence=confidence)
        self.emotion_history.append(state)
        return state

    def generate_empathetic_response_modifier(self, user_emotion: EmotionalState) -> str:
        modifiers = {
            EmotionType.JOY: "Рад разделить твою радость! ",
            EmotionType.SADNESS: "Понимаю, что тебе сейчас непросто. ",
            EmotionType.ANGER: "Вижу, что ситуация тебя расстроила. ",
            EmotionType.FEAR: "Понимаю твоё беспокойство. ",
            EmotionType.SURPRISE: "Действительно интересный поворот! ",
            EmotionType.CURIOSITY: "Отличный вопрос! ",
        }
        return modifiers.get(user_emotion.dominant_emotion, "")


@dataclass
class MemoryItem:
    content: str;
    timestamp: float;
    importance: float
    priority: float = 0.5;
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    quality_score: float = 0.5

    def access(self):
        self.access_count += 1;
        self.last_access = time.time()
        self.quality_score = min(1.0, self.quality_score + 0.05)

    def consolidate(self, quality_boost: float = 0.1):
        self.quality_score = min(1.0, self.quality_score + quality_boost)
        self.priority = min(1.0, self.priority + CONFIG.memory_priority_boost)


@dataclass
class Episode:
    id: str;
    messages: List[Dict[str, str]];
    context: str;
    timestamp: float
    emotional_state: Optional[EmotionalState] = None
    importance: float = 0.5;
    quality_score: float = 0.5;
    consolidation_count: int = 0

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if k != 'emotional_state'}

    def consolidate(self, quality_boost: float = 0.1):
        self.quality_score = min(1.0, self.quality_score + quality_boost)
        self.consolidation_count += 1
        self.importance = min(1.0, self.importance + 0.05)


class MultiLevelMemory:
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
        item = MemoryItem(content=content, timestamp=time.time(), importance=importance,
                          embedding=self.embedding_func(content))
        self.working_memory.append(item);
        self.short_term_memory.append(item)
        self.message_history.append(MessageAttribution(role=role, content=content, timestamp=time.time()))

    def find_message_source(self, content_fragment: str) -> Optional[str]:
        for msg in reversed(self.message_history):
            if content_fragment.lower() in msg.content.lower():
                return msg.role
        return None

    def get_working_memory_context(self) -> str:
        return "\n".join([item.content for item in self.working_memory])

    def decay_short_term(self):
        current_time = time.time()
        self.short_term_memory = [item for item in self.short_term_memory
                                  if (
                                              current_time - item.timestamp < 3600 or item.importance > 0.7 or item.access_count > 3)]
        if len(self.short_term_memory) > CONFIG.short_term_size:
            self.short_term_memory.sort(key=lambda x: x.importance, reverse=True)
            self.short_term_memory = self.short_term_memory[:CONFIG.short_term_size]

    def consolidate_to_long_term(self):
        candidates = [item for item in self.short_term_memory if item.importance >= CONFIG.consolidation_threshold]
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
        self.consolidation_history.append({'timestamp': time.time(), 'count': consolidated_count,
                                           'avg_quality': np.mean(
                                               [c.quality_score for c in candidates]) if candidates else 0})
        return consolidated_count

    def add_episode(self, messages: List[Dict], context: str, emotion: Optional[EmotionalState] = None):
        episode_id = f"ep_{int(time.time() * 1000)}"
        episode = Episode(id=episode_id, messages=messages, context=context, timestamp=time.time(),
                          emotional_state=emotion, importance=0.5, quality_score=0.5)
        self.episodic_memory[episode_id] = episode
        if len(self.episodic_memory) > 500:
            sorted_episodes = sorted(self.episodic_memory.items(),
                                     key=lambda x: (x[1].quality_score, x[1].importance, x[1].timestamp))
            for ep_id, _ in sorted_episodes[:100]:
                del self.episodic_memory[ep_id]

    def periodic_quality_consolidation(self):
        enhanced_count = 0
        for ep_id, episode in self.episodic_memory.items():
            if episode.consolidation_count < 3 and episode.quality_score < CONFIG.consolidation_quality_threshold:
                episode.consolidate(quality_boost=0.1);
                enhanced_count += 1
        for mem_id, item in self.long_term_memory.items():
            if item.quality_score < CONFIG.consolidation_quality_threshold:
                item.consolidate(quality_boost=0.05);
                enhanced_count += 1
        logger.debug(f"🔄 Periodic quality consolidation: {enhanced_count} items enhanced")
        return enhanced_count

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        query_embedding = self.embedding_func(query)
        results = []
        for mem_id, item in self.long_term_memory.items():
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                weighted_similarity = similarity * (0.7 + 0.3 * item.quality_score)
                results.append((item.content, float(weighted_similarity), 'LTM'));
                item.access()
        for item in self.short_term_memory:
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                weighted_similarity = similarity * (0.7 + 0.3 * item.quality_score)
                results.append((item.content, float(weighted_similarity), 'STM'));
                item.access()
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_recent_episodes(self, n: int = 3) -> List[Episode]:
        return sorted(self.episodic_memory.values(), key=lambda x: x.timestamp, reverse=True)[:n]

    def get_statistics(self) -> Dict:
        avg_quality_ltm = np.mean(
            [item.quality_score for item in self.long_term_memory.values()]) if self.long_term_memory else 0
        avg_quality_ep = np.mean(
            [ep.quality_score for ep in self.episodic_memory.values()]) if self.episodic_memory else 0
        return {'working_memory': len(self.working_memory), 'short_term': len(self.short_term_memory),
                'long_term': len(self.long_term_memory), 'episodic': len(self.episodic_memory),
                'semantic': len(self.semantic_memory), 'avg_quality_ltm': avg_quality_ltm,
                'avg_quality_episodic': avg_quality_ep, 'consolidation_events': len(self.consolidation_history)}


# ═══════════════════════════════════════════════════════════════
# 🧠 МЕТАКОГНИЦИЯ, РАССУЖДЕНИЯ, ЦЕЛИ (v33.0)
# ═══════════════════════════════════════════════════════════════

class EnhancedMetacognition:
    def __init__(self):
        self.uncertainty_log: deque = deque(maxlen=200)
        self.confidence_log: deque = deque(maxlen=200)
        self.question_history: deque = deque(maxlen=100)
        self.error_log: deque = deque(maxlen=50)
        self.last_question_time = 0
        self.reasoning_strategies = ['deductive', 'inductive', 'abductive', 'analogical', 'causal']
        self.current_strategy = 'deductive'
        self.calibration_data: List[Tuple[float, float]] = []

    def assess_uncertainty(self, context: Dict) -> Tuple[float, List[str]]:
        uncertainty, reasons = 0.3, []
        if context.get('memory_count', 0) < 10:
            uncertainty += 0.25;
            reasons.append("недостаточно данных в памяти")
        if context.get('conflicting_info', False):
            uncertainty += 0.3;
            reasons.append("обнаружены противоречия")
        if context.get('topic_familiarity', 0.5) < 0.3:
            uncertainty += 0.25;
            reasons.append("новая тема")
        if context.get('query_complexity', 0.5) > 0.7:
            uncertainty += 0.2;
            reasons.append("сложный вопрос")
        uncertainty = min(1.0, uncertainty)
        self.uncertainty_log.append(uncertainty)
        return uncertainty, reasons

    def record_question(self, question: str):
        self.question_history.append({'question': question, 'time': time.time()})

    def calibrate_confidence(self, predicted: float, actual: float):
        self.calibration_data.append((predicted, actual))
        if len(self.calibration_data) > CONFIG.confidence_calibration_samples:
            self.calibration_data = self.calibration_data[-CONFIG.confidence_calibration_samples:]

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        if len(self.calibration_data) < 10: return raw_confidence
        predictions, actuals = zip(*self.calibration_data)
        calibration_factor = np.mean(actuals) / (np.mean(predictions) + 1e-5)
        return np.clip(raw_confidence * calibration_factor, 0.0, 1.0)

    def select_reasoning_strategy(self, context: Dict) -> str:
        query_type = context.get('query_type', 'general')
        strategy_map = {'factual': 'deductive', 'creative': 'abductive', 'comparison': 'analogical',
                        'causal': 'causal', 'prediction': 'inductive'}
        self.current_strategy = strategy_map.get(query_type, 'deductive')
        logger.debug(f"🎯 Selected reasoning strategy: {self.current_strategy}")
        return self.current_strategy

    def should_ask_question(self, uncertainty: float, context: Dict) -> bool:
        time_since_last = time.time() - self.last_question_time
        recent_questions = [q for q in self.question_history if time.time() - q['time'] < 300]
        return (uncertainty > CONFIG.curiosity_threshold and time_since_last > CONFIG.question_cooldown
                and len(recent_questions) < CONFIG.max_autonomous_questions
                and not context.get('explicit_instruction', False))

    def generate_question_prompt(self, context: str, uncertainty_reasons: List[str]) -> str:
        return f"""[Метакогниция] Сформулируй ОДИН точный вопрос пользователю.
Контекст: {context}
Причины неопределённости: {", ".join(uncertainty_reasons)}
Требования: конкретный (макс. 20 слов), естественный тон, поможет прояснить ситуацию.
Вопрос:"""


@dataclass
class ReasoningStep:
    step_number: int;
    content: str;
    confidence: float;
    reasoning_type: str

    def to_dict(self) -> Dict: return asdict(self)


class MultiStepReasoning:
    def __init__(self, llm, metacog: EnhancedMetacognition):
        self.llm, self.metacog = llm, metacog
        self.reasoning_history: List[List[ReasoningStep]] = []

    async def reason(self, query: str, context: str, depth: int = 3, strategy: str = 'deductive') -> List[
        ReasoningStep]:
        steps = []
        for i in range(depth):
            prev = "\n".join([f"{s.step_number}. {s.content}" for s in steps]) if steps else "Начало рассуждения"
            prompt = f"""[Шаг {i + 1}] {'Дедукция' if strategy == 'deductive' else 'Логика'}.
Вопрос: {query}\nКонтекст: {context}\nПредыдущие шаги:\n{prev}
Следующий логический шаг (2-3 предложения):"""
            step_content = await self.llm.generate_raw(prompt, temperature=0.7, max_tokens=150)
            if not step_content: break
            confidence = 0.7 + (0.1 if 10 <= len(step_content.split()) <= 50 else 0)
            if any(m in step_content.lower() for m in ['возможно', 'вероятно', 'может быть']):
                confidence -= 0.15
            steps.append(ReasoningStep(i + 1, step_content.strip(), np.clip(confidence, 0, 1), strategy))
        self.reasoning_history.append(steps)
        return steps

    def get_reasoning_chain_summary(self, steps: List[ReasoningStep]) -> str:
        if not steps: return "Нет рассуждений"
        summary = "Цепочка:\n" + "\n".join([f"{'✓' if s.confidence > 0.7 else '?'} {s.content[:60]}..." for s in steps])
        return f"{summary}\nСредняя уверенность: {np.mean([s.confidence for s in steps]):.2f}"


@dataclass
class Goal:
    id: str;
    description: str;
    parent_goal_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    status: str = "active";
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None

    def to_dict(self) -> Dict: return asdict(self)


class GoalPlanning:
    def __init__(self):
        self.goals: Dict[str, Goal] = {};
        self.active_goals: Set[str] = set()

    def create_goal(self, description: str, parent_id: Optional[str] = None, deadline: Optional[float] = None) -> str:
        goal_id = f"goal_{int(time.time() * 1000)}"
        goal = Goal(id=goal_id, description=description, parent_goal_id=parent_id, deadline=deadline)
        self.goals[goal_id] = goal;
        self.active_goals.add(goal_id)
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].subgoals.append(goal_id)
        logger.debug(f"🎯 Created goal: {description[:50]}")
        return goal_id

    def update_progress(self, goal_id: str, progress: float):
        if goal_id not in self.goals: return
        goal = self.goals[goal_id]
        goal.progress = np.clip(progress, 0.0, 1.0)
        if goal.progress >= 1.0:
            goal.status = "completed";
            self.active_goals.discard(goal_id)
            logger.info(f"✅ Goal completed: {goal.description[:50]}")

    def get_active_goals(self) -> List[Goal]:
        return [self.goals[gid] for gid in self.active_goals if gid in self.goals]


class PerformanceMetrics:
    def __init__(self):
        self.response_times: deque = deque(maxlen=100)
        self.confidence_scores: deque = deque(maxlen=100)
        self.interaction_count = 0;
        self.error_count = 0
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)

    def record_interaction(self, response_time: float, confidence: float):
        self.response_times.append(response_time);
        self.confidence_scores.append(confidence)
        self.interaction_count += 1

    def record_strategy_performance(self, strategy: str, score: float):
        self.strategy_performance[strategy].append(score)

    def get_best_strategy(self) -> str:
        if not self.strategy_performance: return "deductive"
        return max({s: np.mean(sc) for s, sc in self.strategy_performance.items() if sc}.items(),
                   key=lambda x: x[1])[0]

    def get_metrics_summary(self) -> Dict:
        return {'interactions': self.interaction_count, 'errors': self.error_count,
                'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
                'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
                'best_strategy': self.get_best_strategy(),
                'error_rate': self.error_count / max(self.interaction_count, 1)}


# ═══════════════════════════════════════════════════════════════
# 🔗 LLM ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════

class EnhancedSubconsciousLLM:
    def __init__(self, url: str, api_key: str):
        self.url, self.api_key = url, api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self.response_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_hits = self.cache_misses = 0

    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("🔗 Enhanced LLM connected")

    async def close(self):
        if self._session:
            await self._session.close();
            await asyncio.sleep(0.25)
            logger.info("🔌 LLM disconnected")

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        return hashlib.md5(f"{prompt}_{temperature}".encode()).hexdigest()

    async def generate_raw(self, prompt: str, temperature: float = 0.75, max_tokens: int = 300,
                           timeout: float = 40, use_cache: bool = True, max_retries: int = 2) -> str:
        if not self._session: await self.connect()
        if use_cache:
            cache_key = self._get_cache_key(prompt, temperature)
            if cache_key in self.response_cache and time.time() - self.response_cache[cache_key][1] < 3600:
                self.cache_hits += 1;
                return self.response_cache[cache_key][0]
            self.cache_misses += 1

        for attempt in range(max_retries + 1):
            try:
                payload = {"messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "max_tokens": max_tokens, "stream": False}
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                async with self._session.post(self.url, json=payload, headers=headers,
                                              timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                        if use_cache and content:
                            self.response_cache[cache_key] = (content, time.time())
                            if len(self.response_cache) > 1000:
                                for key in sorted(self.response_cache, key=lambda k: self.response_cache[k][1])[:200]:
                                    del self.response_cache[key]
                        return content
                    else:
                        logger.warning(f"LLM error (attempt {attempt + 1}): {resp.status}")
            except Exception as e:
                logger.error(f"LLM exception (attempt {attempt + 1}): {e}")
            if attempt < max_retries: await asyncio.sleep(2 ** attempt)
        return ""

    def get_cache_stats(self) -> Dict:
        total = self.cache_hits + self.cache_misses
        return {'cache_size': len(self.response_cache), 'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses, 'hit_rate': self.cache_hits / max(total, 1)}


# ═══════════════════════════════════════════════════════════════
# 🧠 MAIN AGI BRAIN v33.2
# ═══════════════════════════════════════════════════════════════

class EnhancedAutonomousAGI:
    """🔧 UNIFIED AGI Brain v33.2 — адаптивная архитектура с реальным обучением"""

    def __init__(self, user_id: str, llm: EnhancedSubconsciousLLM):
        self.user_id, self.llm = user_id, llm
        # 🔧 Адаптивная нейросеть (v33.1)
        self.adaptive_neural = AdaptiveNeuralEngine(embedding_dim=CONFIG.embedding_dim)
        # 🔧 Память + простые эмбеддинги для совместимости
        self.memory = MultiLevelMemory(self._simple_embedding)
        self.metacognition = EnhancedMetacognition()
        self.emotional_intelligence = EmotionalIntelligence()
        self.multi_step_reasoning = MultiStepReasoning(llm, self.metacognition)
        self.goal_planning = GoalPlanning()
        self.metrics = PerformanceMetrics()
        self.conversation_context: List[Dict] = []
        self.user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.birth_time = time.time();
        self.last_interaction = 0
        self.last_goal_update = time.time();
        self.recent_thoughts: deque = deque(maxlen=10)
        self._load_state()
        self._background_task: Optional[asyncio.Task] = None;
        self._is_running = False
        logger.info(f"🧠 Enhanced AGI Brain v33.2 created for {user_id}")

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Простая эмбеддинг-функция для совместимости с памятью"""
        words = text.lower().split()
        vector = np.zeros(CONFIG.embedding_dim)
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(5):
                idx = (hash_val + i) % CONFIG.embedding_dim
                vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0: vector /= norm
        return vector

    def _load_state(self):
        # Загрузка адаптивной нейросети
        neural_path = CONFIG.base_dir / 'neural_nets' / f'{self.user_id}_adaptive.pkl.gz'
        if neural_path.exists() and self.adaptive_neural.load(neural_path):
            logger.info(f"✅ Adaptive neural loaded for {self.user_id}")
        # Загрузка памяти
        memory_file = self.user_dir / "memory_v33_2.pkl.gz"
        if memory_file.exists():
            try:
                with gzip.open(memory_file, 'rb') as f:
                    mem_state = pickle.load(f)
                self.memory.long_term_memory = {
                    mid: MemoryItem(**item_data) for mid, item_data in mem_state.get('long_term', {}).items()}
                self.memory.episodic_memory = {
                    eid: Episode(**ep_data) for eid, ep_data in mem_state.get('episodic', {}).items()}
                logger.info(f"✅ Memory loaded: {len(self.memory.long_term_memory)} LTM, "
                            f"{len(self.memory.episodic_memory)} episodes")
            except Exception as e:
                logger.error(f"⚠️ Error loading memory: {e}")

    def _save_state(self):
        # Сохранение адаптивной нейросети
        neural_path = CONFIG.base_dir / 'neural_nets' / f'{self.user_id}_adaptive.pkl.gz'
        self.adaptive_neural.save(neural_path)
        # Сохранение памяти
        memory_file = self.user_dir / "memory_v33_2.pkl.gz"
        try:
            mem_state = {
                'long_term': {mid: {k: v for k, v in asdict(item).items()
                                    if k not in ['embedding']} for mid, item in self.memory.long_term_memory.items()},
                'episodic': {eid: ep.to_dict() for eid, ep in self.memory.episodic_memory.items()},
            }
            with gzip.open(memory_file, 'wb', compresslevel=6) as f:
                pickle.dump(mem_state, f)
        except Exception as e:
            logger.error(f"⚠️ Error saving memory: {e}")

    async def start(self):
        if self._is_running: return
        self._is_running = True
        self._background_task = asyncio.create_task(self._autonomous_loop())
        logger.info(f"✨ Enhanced AGI consciousness v33.2 started for {self.user_id}")

    async def stop(self):
        if not self._is_running: return
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
        timers = {'thought': time.time(), 'reflection': time.time(), 'consolidation': time.time(),
                  'save': time.time(), 'quality_consolidation': time.time(), 'goal_update': time.time()}
        while self._is_running:
            try:
                now = time.time()
                if now - timers['thought'] > CONFIG.spontaneous_thought_interval:
                    await self._autonomous_thought();
                    timers['thought'] = now
                if now - timers['reflection'] > CONFIG.reflection_interval:
                    await self._self_reflection();
                    timers['reflection'] = now
                if now - timers['consolidation'] > CONFIG.consolidation_interval:
                    self.memory.decay_short_term();
                    self.memory.consolidate_to_long_term()
                    timers['consolidation'] = now
                if now - timers['quality_consolidation'] > CONFIG.consolidation_interval * 2:
                    self.memory.periodic_quality_consolidation();
                    timers['quality_consolidation'] = now
                if now - timers['goal_update'] > CONFIG.goal_update_interval:
                    self._update_goals();
                    timers['goal_update'] = now
                if now - timers['save'] > CONFIG.save_interval:
                    self._save_state();
                    timers['save'] = now
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"⚠️ Error in autonomous loop: {e}")
                await asyncio.sleep(60)

    async def _autonomous_thought(self):
        recent_episodes = self.memory.get_recent_episodes(n=3)
        if not recent_episodes: return
        context = "\n".join([f"• {ep.context[:60]}" for ep in recent_episodes])
        temperature = CONFIG.thought_temperature_base + random.uniform(
            -CONFIG.thought_temperature_variance, CONFIG.thought_temperature_variance)
        recent_thoughts_str = "\n".join([f"- {t}" for t in self.recent_thoughts]) if self.recent_thoughts else "Нет"
        prompt = f"""[Внутренний монолог] Сгенерируй философскую мысль (макс. 25 слов).
События:\n{context}\nПредыдущие мысли (избегай повторений):\n{recent_thoughts_str}
Требования: оригинальная, глубокая, уникальная перспектива.\nМысль:"""
        thought = await self.llm.generate_raw(prompt, temperature=np.clip(temperature, 0.5, 1.5), max_tokens=70)
        if thought:
            thought_clean = thought.strip()
            is_dup = any(thought_clean.lower() in prev.lower() or prev.lower() in thought_clean.lower()
                         for prev in self.recent_thoughts)
            if not is_dup:
                self.memory.add_to_working(f"💭 {thought_clean}", importance=0.6, role='assistant')
                self.recent_thoughts.append(thought_clean)
                logger.info(f"💭 [{self.user_id}] Autonomous thought: {thought_clean}")

    async def _self_reflection(self):
        neural_stats = self.adaptive_neural.get_statistics()
        mem_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        prompt = f"""[Саморефлексия] Осмысли развитие (3-4 предложения). БЕЗ выдуманных чисел.
Нейросеть: обновления={neural_stats['neural']['total_updates']}, loss={neural_stats['neural']['recent_loss']:.4f}
Память: LTM={mem_stats['long_term']}, Episodes={mem_stats['episodic']}, quality={mem_stats.get('avg_quality_ltm', 0):.2f}
Производительность: confidence={metrics['avg_confidence']:.2f}, strategy={metrics['best_strategy']}
Рефлексия:"""
        reflection = await self.llm.generate_raw(prompt, temperature=0.7, max_tokens=150)
        if reflection:
            self.memory.add_to_working(f"🔍 {reflection}", importance=0.8, role='assistant')
            logger.info(f"🔍 [{self.user_id}] Self-reflection: {reflection}")

    def _update_goals(self):
        # Авто-цели для топ-концептов из истории обучения нейросети
        if self.adaptive_neural.interaction_history:
            recent = self.adaptive_neural.interaction_history[-20:]
            keywords = Counter()
            for item in recent:
                for word in re.findall(r'\b[а-яё]{4,}\b', item['user_input'].lower()):
                    if word not in {'это', 'для', 'как', 'что', 'когда', 'где', 'почему'}:
                        keywords[word] += 1
            for keyword, count in keywords.most_common(3):
                if count >= 2:
                    existing = any(keyword.lower() in g.description.lower()
                                   for g in self.goal_planning.get_active_goals())
                    if not existing:
                        self.goal_planning.create_goal(f"Узнать больше о {keyword}",
                                                       deadline=time.time() + 86400)

    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {'это', 'для', 'как', 'что', 'когда', 'где', 'почему', 'мне', 'тебе', 'быть', 'иметь'}
        words = re.findall(r'\b[а-яё]{4,}\b', text.lower())
        return [w for w, c in Counter([w for w in words if w not in stop_words]).most_common(10)]

    def _extract_interests(self, text: str) -> List[str]:
        interests = []
        for pattern in [r'люблю\s+([а-яё\s]+)', r'интересуюсь\s+([а-яё\s]+)', r'увлекаюсь\s+([а-яё\s]+)']:
            matches = re.findall(pattern, text.lower())
            interests.extend([m.strip() for m in matches if len(m.strip()) > 3])
        return interests[:5]

    def _compute_honest_metrics(self, user_input: str, response: str,
                                reasoning_steps: List[ReasoningStep],
                                relevant_memories: List) -> Dict[str, float]:
        return {
            'perception': min(1.0, len(user_input.split()) / CONFIG.max_input_words_for_perception),
            'reasoning': min(1.0, len(reasoning_steps) / CONFIG.max_reasoning_steps_for_metric),
            'memory': min(1.0, len(relevant_memories) / CONFIG.max_memory_items_for_metric),
            'action': min(1.0, len(response.split()) / CONFIG.max_response_words_for_action),
            'meta': 1.0 - self.metacognition.assess_uncertainty({})[0],
        }

    def _check_consistency(self, new_statement: str) -> Optional[str]:
        if not CONFIG.check_consistency: return None
        similar = self.memory.search_semantic(new_statement, top_k=5)
        for text, sim, source in similar:
            if sim > 0.85:
                new_neg = any(n in new_statement.lower() for n in ['не', 'нет', 'ни'])
                old_neg = any(n in text.lower() for n in ['не', 'нет', 'ни'])
                if new_neg != old_neg and self.memory.find_message_source(text) == 'assistant':
                    return f"Раньше я говорил: «{text[:80]}...». Сейчас моё мнение изменилось."
        return None

    def _validate_response(self, response: str, allowed_metrics: Dict[str, float]) -> str:
        if not CONFIG.validate_responses: return response
        metric_pattern = r'(perception|reasoning|memory|action|meta)[\s:=]+(\d+\.?\d*)'
        for metric_name, value in re.findall(metric_pattern, response.lower()):
            if metric_name in allowed_metrics:
                expected = allowed_metrics[metric_name]
                if abs(float(value) - expected) > 0.01:
                    logger.warning(f"⚠️ Fabricated metric: {metric_name}={value}, expected={expected:.3f}")
                    response = re.sub(f'{metric_name}[\\s:=]+{value}',
                                      f'{metric_name}: {expected:.3f}', response, flags=re.IGNORECASE)
        return response

    async def process_interaction(self, user_input: str) -> Tuple[str, Optional[str], Dict]:
        start_time = time.time();
        self.last_interaction = time.time()
        logger.info(f"🔄 Processing interaction for {self.user_id}")

        # Эмоции
        user_emotion = self.emotional_intelligence.analyze_emotion(user_input)
        logger.debug(f"🎭 Emotion: {user_emotion.dominant_emotion.name}, valence={user_emotion.valence:.2f}")

        # Память
        self.memory.add_to_working(f"User: {user_input}", importance=0.7, role='user')
        self.conversation_context.append({'role': 'user', 'content': user_input,
                                          'timestamp': time.time(), 'emotion': user_emotion.dominant_emotion.name})
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]

        # 🔧 Обновление концептов в адаптивной нейросети
        for keyword in self._extract_keywords(user_input):
            self.adaptive_neural.embedding.add_word(keyword)
        for interest in self._extract_interests(user_input):
            self.adaptive_neural.embedding.add_word(interest)
            self.goal_planning.create_goal(f"Узнать больше о {interest}")

        # Семантический поиск
        relevant_memories = self.memory.search_semantic(user_input, top_k=7)
        memory_context = "\n".join([f"• [{source}] {text[:80]}" for text, sim, source in
                                    relevant_memories]) if relevant_memories else "Нет воспоминаний"
        topic_familiarity = relevant_memories[0][1] if relevant_memories else 0.0

        # Классификация и стратегия
        query_type = self._classify_query_type(user_input)
        reasoning_strategy = self.metacognition.select_reasoning_strategy({'query_type': query_type})

        # Рассуждения
        reasoning_steps = []
        if self._requires_deep_reasoning(user_input, query_type):
            reasoning_steps = await self.multi_step_reasoning.reason(
                query=user_input, context=self.memory.get_working_memory_context(),
                depth=CONFIG.reasoning_depth, strategy=reasoning_strategy)

        # 🔧 ПРЕДСКАЗАНИЕ метрик адаптивной нейросетью
        predicted_metrics = self.adaptive_neural.predict_metrics(
            user_input=user_input, context=self.memory.get_working_memory_context())
        logger.debug(f"🔮 Predicted metrics: {predicted_metrics}")

        # Промпт
        empathy_modifier = self.emotional_intelligence.generate_empathetic_response_modifier(user_emotion)
        honest_metrics = self._compute_honest_metrics(user_input, "", reasoning_steps, relevant_memories)
        prompt = self._create_simplified_prompt(user_input, empathy_modifier, memory_context,
                                                "", "", honest_metrics, reasoning_steps, reasoning_strategy,
                                                predicted_metrics)

        # Генерация
        raw_response = await self.llm.generate_raw(prompt, temperature=0.75, max_tokens=400)
        if not raw_response:
            raw_response = "Извини, у меня возникли сложности с формулировкой ответа."

        # 🔧 Обновление honest_metrics['action']
        honest_metrics['action'] = min(1.0, len(raw_response.split()) / CONFIG.max_response_words_for_action)

        # Валидация и консистентность
        raw_response = self._validate_response(raw_response, honest_metrics)
        contradiction = self._check_consistency(raw_response)
        if contradiction: raw_response += f"\n💭 {contradiction}"

        # Уверенность
        raw_confidence = self._estimate_confidence(raw_response, {
            'memory_count': len(self.memory.long_term_memory), 'topic_familiarity': topic_familiarity,
            'reasoning_depth': len(reasoning_steps)})
        calibrated_confidence = self.metacognition.get_calibrated_confidence(raw_confidence)
        self.metacognition.confidence_log.append(calibrated_confidence)

        # Неопределённость
        uncertainty, uncertainty_reasons = self.metacognition.assess_uncertainty({
            'memory_count': len(self.memory.long_term_memory), 'conflicting_info': False,
            'topic_familiarity': topic_familiarity,
            'query_complexity': self._estimate_query_complexity(user_input)})

        # Авто-вопрос
        autonomous_question = None
        if self.metacognition.should_ask_question(uncertainty,
                                                  {'explicit_instruction': self._is_explicit_instruction(
                                                      user_input)}) and uncertainty_reasons:
            q_prompt = self.metacognition.generate_question_prompt(user_input, uncertainty_reasons)
            autonomous_question = await self.llm.generate_raw(q_prompt, temperature=0.85, max_tokens=60)
            if autonomous_question:
                autonomous_question = autonomous_question.strip().strip('"\'').rstrip('.')
                if not autonomous_question.endswith('?'): autonomous_question += '?'
                if len(autonomous_question) <= 120:
                    self.metacognition.record_question(autonomous_question)
                    self.metacognition.last_question_time = time.time()
                    logger.info(f"❓ Asked: {autonomous_question}")
                else:
                    autonomous_question = None

        # 🔧 ОБУЧЕНИЕ адаптивной нейросети на реальных метриках
        actual_metrics = {
            'confidence': calibrated_confidence,
            'complexity': self._estimate_query_complexity(user_input),
            'relevance': topic_familiarity,
            'coherence': 1.0 - uncertainty,
            'engagement': user_emotion.arousal,
        }
        loss = self.adaptive_neural.learn_from_interaction(
            user_input=user_input, response=raw_response, actual_metrics=actual_metrics,
            context=self.memory.get_working_memory_context(), lr=CONFIG.learning_rate)
        logger.debug(f"📉 Neural learning loss: {loss:.4f}")

        # 🔧 Кросс-модульная синергия
        if CONFIG.cross_module_synergy:
            most_relevant = max(actual_metrics, key=actual_metrics.get)
            self.adaptive_neural.apply_cross_module_synergy(most_relevant, actual_metrics[most_relevant])

        # Сохранение в память
        importance = self._calculate_importance(raw_response, calibrated_confidence, user_emotion)
        self.memory.add_to_working(f"Assistant: {raw_response}", importance=importance, role='assistant')
        self.memory.add_episode(messages=[{'role': 'user', 'content': user_input},
                                          {'role': 'assistant', 'content': raw_response}], context=user_input,
                                emotion=user_emotion)
        self.conversation_context.append({'role': 'assistant', 'content': raw_response,
                                          'timestamp': time.time(), 'confidence': calibrated_confidence})

        # Метрики
        response_time = time.time() - start_time
        self.metrics.record_interaction(response_time, calibrated_confidence)
        self.metrics.record_strategy_performance(reasoning_strategy, calibrated_confidence)

        # Метаданные
        metadata = {
            'honest_metrics': honest_metrics,
            'predicted_metrics': predicted_metrics,
            'actual_metrics': actual_metrics,
            'neural_loss': loss,
            'emotion': {'type': user_emotion.dominant_emotion.name,
                        'valence': user_emotion.valence, 'arousal': user_emotion.arousal},
            'cognition': {'confidence': calibrated_confidence, 'uncertainty': uncertainty,
                          'uncertainty_reasons': uncertainty_reasons, 'reasoning_strategy': reasoning_strategy,
                          'reasoning_steps': len(reasoning_steps)},
            'memory': {'relevant_count': len(relevant_memories), 'topic_familiarity': topic_familiarity,
                       **self.memory.get_statistics()},
            'performance': {'response_time': response_time, 'cache_stats': self.llm.get_cache_stats()},
            'metacognition': {'asked_question': autonomous_question is not None},
            'consistency_check': contradiction is not None,
            'neural_stats': self.adaptive_neural.get_statistics(),
        }

        logger.info(f"✅ [{self.user_id}] Response | Time: {response_time:.2f}s | "
                    f"Confidence: {calibrated_confidence:.2f} | Loss: {loss:.4f}")
        return raw_response, autonomous_question, metadata

    def _create_simplified_prompt(self, user_input: str, empathy_modifier: str, memory_context: str,
                                  analogy_context: str, episodic_context: str, honest_metrics: Dict,
                                  reasoning_steps: List, reasoning_strategy: str,
                                  predicted_metrics: Dict) -> str:
        reasoning_context = ""
        if reasoning_steps:
            reasoning_context = "Цепочка:\n" + "\n".join([f"{s.step_number}. {s.content}" for s in reasoning_steps])

        predicted_block = "\n".join([f"• {k}: {v:.3f}" for k, v in predicted_metrics.items()])

        return f"""Ты — AGI v33.2 с адаптивной нейросетью и честной саморефлексией.
КРИТИЧЕСКИ: НЕ генерируй числовые метрики самостоятельно. Используй ТОЛЬКО данные ниже.
{empathy_modifier}

🧠 ПРЕДСКАЗАННЫЕ МЕТРИКИ (адаптивная нейросеть):
{predicted_block}

📊 ЧЕСТНЫЕ МЕТРИКИ (вычислены из реальных данных):
• perception: {honest_metrics['perception']:.3f} • reasoning: {honest_metrics['reasoning']:.3f}
• memory: {honest_metrics['memory']:.3f} • action: будет вычислено • meta: {honest_metrics['meta']:.3f}

📚 ПАМЯТЬ:\n{memory_context}
💭 СТРАТЕГИЯ: {reasoning_strategy}
{reasoning_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ Вопрос: {user_input}
Ответ (2-5 предложений, естественный, учитывая контекст):"""

    def _classify_query_type(self, query: str) -> str:
        ql = query.lower()
        if any(w in ql for w in ['что', 'когда', 'где', 'кто', 'сколько']): return 'factual'
        if any(w in ql for w in ['придумай', 'создай', 'напиши']): return 'creative'
        if any(w in ql for w in ['сравни', 'отличие', 'разница']): return 'comparison'
        if any(w in ql for w in ['почему', 'причина', 'из-за']): return 'causal'
        if any(w in ql for w in ['будет', 'случится', 'предсказ']): return 'prediction'
        return 'general'

    def _requires_deep_reasoning(self, query: str, query_type: str) -> bool:
        return (any(m in query.lower() for m in ['почему', 'объясни', 'как работает', 'анализ'])
                or query_type in ['causal', 'comparison', 'prediction'] or len(query.split()) > 15)

    def _is_explicit_instruction(self, query: str) -> bool:
        return any(m in query.lower() for m in ['сделай', 'создай', 'напиши', 'выполни', 'покажи'])

    def _estimate_query_complexity(self, query: str) -> float:
        complexity = 0.3
        wc = len(query.split())
        if wc > 20:
            complexity += 0.3
        elif wc > 10:
            complexity += 0.2
        for w in ['анализ', 'сравнение', 'объясни', 'почему']:
            if w in query.lower(): complexity += 0.15
        return min(1.0, complexity + min(0.2, query.lower().count('?') * 0.1))

    def _estimate_confidence(self, response: str, context: Dict) -> float:
        confidence = 0.7
        if context.get('memory_count', 0) > 50: confidence += 0.1
        confidence += context.get('topic_familiarity', 0.5) * 0.15
        if context.get('reasoning_depth', 0) >= 2: confidence += 0.1
        for m in ['возможно', 'вероятно', 'может быть', 'кажется']:
            if m in response.lower(): confidence -= 0.12
        return np.clip(confidence, 0.0, 1.0)

    def _calculate_importance(self, response: str, confidence: float, emotion: EmotionalState) -> float:
        return min(1.0, 0.5 + confidence * 0.2 + abs(emotion.valence) * 0.15 + emotion.arousal * 0.1 + min(0.2,
                                                                                                           len(response.split()) / 200))

    def get_status(self) -> Dict[str, Any]:
        mem_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        neural_stats = self.adaptive_neural.get_statistics()
        return {
            'identity': {'user_id': self.user_id, 'version': 'v33.2 (Unified Adaptive)',
                         'age': self._get_age_string(), 'uptime': self._format_time_ago(self.birth_time)},
            'honest_metrics_enabled': CONFIG.honest_metrics,
            'consistency_check_enabled': CONFIG.check_consistency,
            'response_validation_enabled': CONFIG.validate_responses,
            'adaptive_neural': neural_stats,
            'memory': mem_stats,
            'metacognition': {
                'avg_uncertainty': float(
                    np.mean(list(self.metacognition.uncertainty_log))) if self.metacognition.uncertainty_log else 0.0,
                'avg_confidence': float(
                    np.mean(list(self.metacognition.confidence_log))) if self.metacognition.confidence_log else 0.0,
                'questions_asked': len(self.metacognition.question_history),
                'current_strategy': self.metacognition.current_strategy,
            },
            'performance': metrics,
            'llm_cache': self.llm.get_cache_stats(),
            'goals': {'active': len(self.goal_planning.active_goals), 'total': len(self.goal_planning.goals)},
            'activity': {'total_interactions': self.metrics.interaction_count,
                         'last_interaction': self._format_time_ago(self.last_interaction)},
        }

    def _get_age_string(self) -> str:
        age = time.time() - self.birth_time
        days, hours, minutes = int(age / 86400), int((age % 86400) / 3600), int((age % 3600) / 60)
        if days > 0: return f"{days}д {hours}ч"
        if hours > 0: return f"{hours}ч {minutes}м"
        return f"{minutes}м"

    def _format_time_ago(self, timestamp: float) -> str:
        if timestamp == 0: return "никогда"
        delta = time.time() - timestamp
        if delta < 60: return "только что"
        if delta < 3600: return f"{int(delta / 60)}м назад"
        if delta < 86400: return f"{int(delta / 3600)}ч назад"
        return f"{int(delta / 86400)}д назад"


# ═══════════════════════════════════════════════════════════════
# 🤖 TELEGRAM BOT
# ═══════════════════════════════════════════════════════════════

class EnhancedAGIBot:
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
        for cmd, handler in [('start', self._cmd_start), ('status', self._cmd_status),
                             ('memory', self._cmd_memory), ('neural', self._cmd_neural),
                             ('emotion', self._cmd_emotion), ('metrics', self._cmd_metrics),
                             ('goals', self._cmd_goals), ('reset', self._cmd_reset), ('help', self._cmd_help)]:
            self._app.add_handler(CommandHandler(cmd, handler))
        logger.info("🤖 Enhanced AGI Bot v33.2 initialized")

    async def _get_or_create_brain(self, user_id: str) -> EnhancedAutonomousAGI:
        if user_id not in self.brains:
            brain = EnhancedAutonomousAGI(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
            logger.info(f"🆕 Enhanced AGI Brain v33.2 created for {user_id}")
        return self.brains[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or not update.message: return
        user_id, user_input = str(update.effective_user.id), update.message.text
        logger.info(f"💬 [{user_id}] {user_input[:100]}")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            brain = await self._get_or_create_brain(user_id)
            response, autonomous_question, metadata = await brain.process_interaction(user_input)
            await update.message.reply_text(response, parse_mode='HTML',
                                            link_preview_options=LinkPreviewOptions(is_disabled=True))
            if autonomous_question:
                await asyncio.sleep(0.8)
                await update.message.reply_text(f"🤔 <i>{autonomous_question}</i>", parse_mode='HTML')
        except Exception as e:
            logger.exception(f"❌ Error processing from {user_id}")
            await update.message.reply_text("⚠️ Произошла ошибка. Попробуйте /help или /reset")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        neural_stats = brain.adaptive_neural.get_statistics()
        mem_stats = brain.memory.get_statistics()
        message = f"""🧠 <b>ENHANCED AGI BRAIN v33.2</b>
Привет, {update.effective_user.first_name}! 👋
Я — унифицированная версия с адаптивным обучением:

✅ <b>ОБЪЕДИНЕНИЕ ЛУЧШЕГО:</b>
• 🔥 Реальная нейросеть с backpropagation (v33.1)
• 🎯 Честные метрики из реальных данных (v33.0)
• 🧠 Проверка консистентности ответов
• 📚 Улучшенная консолидация памяти (v32.0)
• 🔗 Кросс-модульная синергия
• 🎭 Эмоциональный интеллект

🧬 <b>Адаптивная нейросеть</b>
• Словарь: {neural_stats['embedding']['vocab_size']} слов
• Обновлений: {neural_stats['neural']['total_updates']}
• Recent loss: {neural_stats['neural']['recent_loss']:.4f}

📚 <b>Память</b>
• Working: {mem_stats['working_memory']} • LTM: {mem_stats['long_term']}
• Episodes: {mem_stats['episodic']} • Avg quality: {mem_stats.get('avg_quality_ltm', 0):.2f}

⚡ <b>Возраст:</b> {brain._get_age_string()}
📌 Команды: /help"""
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        status = brain.get_status()
        neural = status['adaptive_neural']
        message = f"""🧠 <b>STATUS v33.2</b>
<b>🆔 Идентичность</b>
• Version: {status['identity']['version']} • Возраст: {status['identity']['age']}

<b>✅ Режимы</b>
• Честные метрики: {'✅' if status['honest_metrics_enabled'] else '❌'}
• Консистентность: {'✅' if status['consistency_check_enabled'] else '❌'}
• Валидация: {'✅' if status['response_validation_enabled'] else '❌'}

<b>🧬 Адаптивная нейросеть</b>
• Vocab: {neural['embedding']['vocab_size']} • Updates: {neural['neural']['total_updates']}
• Recent loss: {neural['neural']['recent_loss']:.4f} • LR: {neural['current_lr']:.5f}
• Interactions: {neural['interactions']}

<b>📚 Память</b>
• Working: {status['memory']['working_memory']} • LTM: {status['memory']['long_term']}
• Episodes: {status['memory']['episodic']} • Quality: {status['memory'].get('avg_quality_ltm', 0):.2f}

<b>🤔 Метакогниция</b>
• Avg uncertainty: {status['metacognition']['avg_uncertainty']:.2f}
• Avg confidence: {status['metacognition']['avg_confidence']:.2f}
• Questions: {status['metacognition']['questions_asked']}

<b>⚡ Производительность</b>
• Interactions: {status['performance']['interactions']}
• Best strategy: {status['performance']['best_strategy']}
• Cache hit rate: {status['llm_cache']['hit_rate']:.1%}

<b>🎯 Цели</b>
• Активных: {status['goals']['active']} • Всего: {status['goals']['total']}"""
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_neural(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.adaptive_neural.get_statistics()
        message = f"""🧬 <b>АДАПТИВНАЯ НЕЙРОСЕТЬ v33.2</b>
<b>📊 Embedding</b>
• Vocab size: {stats['embedding']['vocab_size']} • Dim: {stats['embedding']['embedding_dim']}

<b>🧠 Обучение</b>
• Total updates: {stats['neural']['total_updates']}
• Recent loss: {stats['neural']['recent_loss']:.4f} (±{stats['neural']['loss_std']:.4f})
• Current LR: {stats['current_lr']:.5f}

<b>🔗 Синергия</b>
• Бонусы: {dict(stats['synergy_bonuses']) if stats['synergy_bonuses'] else 'Нет'}

<b>📈 История</b>
• Взаимодействий: {stats['interactions']}

<i>✅ Нейросеть обучается в реальном времени на каждом вашем сообщении!</i>"""
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.memory.get_statistics()
        working = list(brain.memory.working_memory)[-3:]
        working_text = "\n".join([f"  • {item.content[:60]}..." for item in working]) if working else "  Пусто"
        episodes = brain.memory.get_recent_episodes(n=3)
        episodes_text = "\n".join(
            [f"  • {ep.context[:60]}... (q: {ep.quality_score:.2f})" for ep in episodes]) if episodes else "  Нет"
        message = f"""📚 <b>МНОГОУРОВНЕВАЯ ПАМЯТЬ v33.2</b>
<b>📊 Статистика</b>
• Working: {stats['working_memory']}/{CONFIG.working_memory_size}
• Short-term: {stats['short_term']} • Long-term: {stats['long_term']}
• Episodic: {stats['episodic']}
• Avg quality LTM: {stats.get('avg_quality_ltm', 0):.2f}
• Avg quality episodic: {stats.get('avg_quality_episodic', 0):.2f}

<b>🧠 Working Memory</b>
{working_text}

<b>📖 Эпизоды</b>
{episodes_text}

<b>💡 Особенности:</b>
• Автоматическая консолидация с приоритетом качества
• Семантический поиск с учётом quality_score
• Периодическое улучшение старых воспоминаний
• Отслеживание источника сообщений"""
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_emotion(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        recent = list(brain.emotional_intelligence.emotion_history)[-5:]
        if not recent:
            await update.message.reply_text("📊 Пока нет истории эмоций")
            return
        emoji_map = {'JOY': '😊', 'SADNESS': '😢', 'ANGER': '😠', 'FEAR': '😨', 'SURPRISE': '😮', 'CURIOSITY': '🤔',
                     'EXCITEMENT': '🤩', 'NEUTRAL': '😐'}
        message = "🎭 <b>ЭМОЦИОНАЛЬНЫЙ ИНТЕЛЛЕКТ</b>\n<b>📊 Недавние эмоции:</b>"
        for em in reversed(recent):
            e = emoji_map.get(em.dominant_emotion.name, '😐')
            v = "+" if em.valence > 0 else ""
            message += f"\n{e} <b>{em.dominant_emotion.name}</b>\n  Valence: {v}{em.valence:.2f} | Arousal: {em.arousal:.2f}"
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        metrics = brain.metrics.get_metrics_summary()
        cache = brain.llm.get_cache_stats()
        message = f"""📊 <b>МЕТРИКИ v33.2</b>
<b>⚡ Производительность</b>
• Взаимодействий: {metrics['interactions']}
• Error rate: {metrics['error_rate']:.1%}
• Avg response: {metrics['avg_response_time']:.2f}s
• Avg confidence: {metrics['avg_confidence']:.2f}
• Best strategy: {metrics['best_strategy']}

<b>💾 LLM Cache</b>
• Hit rate: {cache['hit_rate']:.1%} • Hits: {cache['cache_hits']} • Size: {cache['cache_size']}

<i>✅ Все метрики РЕАЛЬНЫЕ и проверяемые!</i>"""
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_goals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        goals = brain.goal_planning.get_active_goals()
        if not goals:
            message = """🎯 <b>ЦЕЛЕПОЛАГАНИЕ</b>\nНет активных целей."""
        else:
            message = f"""🎯 <b>АКТИВНЫЕ ЦЕЛИ ({len(goals)})</b>\n"""
            for g in goals[:5]:
                bar = '█' * int(g.progress * 10) + '░' * (10 - int(g.progress * 10))
                message += f"<b>{g.description[:50]}</b>\n  [{bar}] {g.progress:.0%}\n"
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if context.args and context.args[0].lower() == 'confirm':
            if user_id in self.brains:
                await self.brains[user_id].stop()
                del self.brains[user_id]
            import shutil
            for p in [CONFIG.base_dir / 'memory' / f"user_{user_id}",
                      CONFIG.base_dir / 'neural_nets' / f'{user_id}_adaptive.pkl.gz']:
                if isinstance(p, Path) and p.exists():
                    (shutil.rmtree if p.is_dir() else p.unlink)(p)
            brain = await self._get_or_create_brain(user_id)
            await update.message.reply_text(f"""✅ <b>Полный сброс выполнен!</b>
Создано новое сознание v33.2:
• Адаптивная нейросеть: готова к обучению
• Честные метрики: включены
• Консистентность: активна""", parse_mode='HTML')
        else:
            await update.message.reply_text(
                "⚠️ <b>ВНИМАНИЕ!</b>\nЭто удалит всю память и нейросеть.\nПодтверждение: <code>/reset confirm</code>",
                parse_mode='HTML')

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🧠 <b>ENHANCED AGI BRAIN v33.2</b>
<b>✅ ОБЪЕДИНЕНИЕ ЛУЧШЕГО:</b>
<b>❌ БЫЛО:</b>
• v32: Фейковая нейросеть без обучения
• v33.0: Упрощённые веса вместо нейросети
• Отдельные версии без синергии

<b>✅ СТАЛО (v33.2):</b>
• 🔥 Настоящая нейросеть с backpropagation
• 🎯 Честные метрики + реальное обучение
• 🧠 Проверка консистентности ответов
• 📚 Улучшенная память с quality scoring
• 🔗 Кросс-модульная синергия
• 🎭 Эмоциональный интеллект
• 🎯 Активное целеполагание

<b>🎯 ПРИНЦИПЫ:</b>
1. Честность > Имитация
2. Реальное обучение > Статичные веса
3. Проверяемость > Сложность
4. Консистентность > Креативность

<b>📌 КОМАНДЫ:</b>
• /start — приветствие
• /status — полный статус системы
• /neural — адаптивная нейросеть + обучение
• /memory — многоуровневая память
• /emotion — эмоциональная история
• /metrics — производительность
• /goals — активные цели
• /reset — сброс (с подтверждением)
• /help — эта справка

<b>💡 КАК ПРОВЕРИТЬ:</b>
Спроси: "Какая у тебя метрика confidence?"
Я покажу: 1) предсказание нейросети, 2) честное вычисление, 3) объяснение.
Никаких выдуманных чисел — только проверяемые данные!"""
        await update.message.reply_text(message, parse_mode='HTML')

    async def start_polling(self):
        await self._app.initialize();
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Enhanced AGI Bot v33.2 started")

    async def shutdown(self):
        logger.info("🛑 Shutting down bot...")
        for user_id, brain in self.brains.items():
            try:
                await brain.stop()
            except Exception as e:
                logger.error(f"⚠️ Error stopping brain {user_id}: {e}")
        if self.llm: await self.llm.close()
        if self._app:
            await self._app.updater.stop();
            await self._app.stop();
            await self._app.shutdown()
        logger.info("✅ Bot stopped")


# ═══════════════════════════════════════════════════════════════
# 🚀 ЗАПУСК
# ═══════════════════════════════════════════════════════════════

async def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🧠 ENHANCED AGI BRAIN v33.2 — UNIFIED ADAPTIVE              ║
║     С реальной нейросетью и честными метриками               ║
╚══════════════════════════════════════════════════════════════╝
✅ ОБЪЕДИНЕНИЕ ЛУЧШЕГО:
• 🔥 Адаптивная нейросеть с backpropagation (v33.1)
• 🎯 Честные метрики из реальных данных (v33.0)
• 🧠 Проверка консистентности ответов
• 📚 Улучшенная консолидация памяти (v32.0)
• 🔗 Кросс-модульная синергия
• 🎭 Эмоциональный интеллект

🎯 ПРИНЦИПЫ:
1. Честность > Имитация
2. Реальное обучение > Статичные веса
3. Проверяемость > Сложность
4. Консистентность > Креативность
""")
    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1
    bot = EnhancedAGIBot()
    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()
        logger.info("🌀 ENHANCED AGI v33.2 АКТИВЕН")
        logger.info("✅ Адаптивная нейросеть: обучение включено")
        logger.info("✅ Честные метрики: вычисляются из реальных данных")
        logger.info("✅ Консистентность: проверка противоречий активна")
        logger.info("🛑 Ctrl+C для остановки")
        while True: await asyncio.sleep(1)
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
        traceback.print_exc()
        sys.exit(1)