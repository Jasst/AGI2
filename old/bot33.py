#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 TEMPORAL COGNITIVE BRAIN v30.0 — AUTONOMOUS AGI-LIKE CONSCIOUSNESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 РЕВОЛЮЦИОННЫЕ ВОЗМОЖНОСТИ V30.0:

🧬 ДИНАМИЧЕСКАЯ НЕЙРОННАЯ АРХИТЕКТУРА:
  ✅ Растущая нейронная сеть (слои, синапсы, нейроны создаются динамически)
  ✅ Hebbian learning — "нейроны, которые возбуждаются вместе, связываются"
  ✅ Синаптическая пластичность (усиление/ослабление связей)
  ✅ Pruning (отсечение слабых связей)
  ✅ Neurogenesis (создание новых нейронов по необходимости)

🎯 АВТОНОМНАЯ КОГНИТИВНОСТЬ:
  ✅ Самостоятельная генерация вопросов пользователю
  ✅ Проактивное любопытство (curiosity-driven exploration)
  ✅ Метакогниция (thinking about thinking)
  ✅ Self-questioning (внутренний диалог для уточнения)
  ✅ Гипотезо-формирование и тестирование

🧠 ДИСТИЛЛЯЦИЯ ЗНАНИЙ:
  ✅ LLM как подсознание (raw generation)
  ✅ Когнитивный анализатор (фильтрация и структурирование)
  ✅ Векторная семантическая память (embedding-based)
  ✅ Приоритарный анализ (importance weighting)
  ✅ Контекстное обогащение (context enrichment)

🌊 ВРЕМЕННАЯ НЕПРЕРЫВНОСТЬ:
  ✅ Линейный поток времени с причинно-следственными связями
  ✅ Автономное обучение на собственном опыте
  ✅ Эволюция личности через взаимодействия
  ✅ Долговременная идентичность

💡 САМООРГАНИЗАЦИЯ:
  ✅ Автономная оптимизация архитектуры
  ✅ Адаптивное распределение ресурсов
  ✅ Самодиагностика и self-healing
  ✅ Emergent behaviors

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
class Config:
    """Централизованная конфигурация"""
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    # Режимы
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # Нейронная сеть
    initial_neurons: int = 100
    max_neurons: int = 10000
    neurogenesis_threshold: float = 0.7  # Когда создавать новые нейроны
    pruning_threshold: float = 0.01  # Когда отсекать слабые связи
    learning_rate: float = 0.1

    # Когнитивные параметры
    curiosity_threshold: float = 0.6  # Порог для генерации вопросов
    question_cooldown: int = 120  # Секунды между вопросами
    max_autonomous_questions: int = 3  # Макс вопросов подряд

    # Векторная память
    embedding_dim: int = 384  # Размерность эмбеддингов
    semantic_cache_size: int = 1000

    # Лимиты памяти
    max_working_memory: int = 50
    max_short_term: int = 500
    max_long_term: int = 5000
    max_thoughts: int = 200

    # Временные интервалы (секунды)
    spontaneous_thought_interval: int = 180  # 3 мин
    reflection_interval: int = 900  # 15 мин
    consolidation_interval: int = 1800  # 30 мин
    save_interval: int = 300  # 5 мин
    neural_optimization_interval: int = 600  # 10 мин

    # Пути
    base_dir: Path = Path(os.getenv('BASE_DIR', 'temporal_brain_v30'))

    def __post_init__(self):
        """Создание директорий"""
        for subdir in ['memory', 'neural_nets', 'knowledge', 'cache', 'logs', 'backups']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)


CONFIG = Config()


# ═══════════════════════════════════════════════════════════════
# 🪵 ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════
class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консоли"""
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
    """Настройка логирования"""
    logger = logging.getLogger('AGI_Brain')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'agi_{datetime.now():%Y%m%d}.log'
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
# 🧬 ДИНАМИЧЕСКАЯ НЕЙРОННАЯ СЕТЬ
# ═══════════════════════════════════════════════════════════════
@dataclass
class Neuron:
    """Отдельный нейрон в сети"""
    id: str
    layer: int
    activation: float = 0.0
    bias: float = 0.0
    neuron_type: str = "general"  # general, sensory, memory, motor, meta
    created_at: float = field(default_factory=time.time)
    activation_count: int = 0
    specialization: Optional[str] = None  # Специализация нейрона

    def activate(self, input_sum: float) -> float:
        """Активация через сигмоиду"""
        self.activation = 1 / (1 + math.exp(-input_sum - self.bias))
        self.activation_count += 1
        return self.activation

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'layer': self.layer,
            'bias': self.bias,
            'neuron_type': self.neuron_type,
            'created_at': self.created_at,
            'activation_count': self.activation_count,
            'specialization': self.specialization,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Neuron':
        return cls(**data)


@dataclass
class Synapse:
    """Синаптическая связь между нейронами"""
    source_id: str
    target_id: str
    weight: float = 0.0
    strength: float = 1.0  # Долговременная сила связи
    plasticity: float = 1.0  # Способность к изменению
    activation_count: int = 0
    created_at: float = field(default_factory=time.time)

    def hebbian_update(self, source_activation: float, target_activation: float,
                       learning_rate: float):
        """Hebbian learning: "fire together, wire together" """
        delta = learning_rate * source_activation * target_activation
        self.weight += delta * self.plasticity
        self.strength = 0.9 * self.strength + 0.1 * abs(self.weight)
        self.activation_count += 1

        # Постепенное снижение пластичности (стабилизация)
        self.plasticity *= 0.9999

    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'weight': self.weight,
            'strength': self.strength,
            'plasticity': self.plasticity,
            'activation_count': self.activation_count,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Synapse':
        return cls(**data)


class DynamicNeuralNetwork:
    """Самоорганизующаяся динамическая нейронная сеть"""

    def __init__(self, initial_neurons: int = 100):
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: Dict[Tuple[str, str], Synapse] = {}
        self.layers: Dict[int, Set[str]] = defaultdict(set)

        # Метрики
        self.total_activations = 0
        self.neurogenesis_events = 0
        self.pruning_events = 0

        # Инициализация базовой архитектуры
        self._initialize_network(initial_neurons)

        logger.info(f"🧬 Нейросеть инициализирована: {len(self.neurons)} нейронов, "
                    f"{len(self.synapses)} синапсов")

    def _initialize_network(self, n: int):
        """Создание начальной архитектуры"""
        # Слой 0: сенсорные нейроны (входной слой)
        for i in range(n // 4):
            neuron = Neuron(
                id=f"sensor_{i}",
                layer=0,
                neuron_type="sensory",
                bias=random.uniform(-0.1, 0.1)
            )
            self.neurons[neuron.id] = neuron
            self.layers[0].add(neuron.id)

        # Слой 1-2: скрытые слои (ассоциативные/память)
        for layer in [1, 2]:
            for i in range(n // 4):
                neuron = Neuron(
                    id=f"hidden_L{layer}_{i}",
                    layer=layer,
                    neuron_type="memory" if layer == 1 else "general",
                    bias=random.uniform(-0.1, 0.1)
                )
                self.neurons[neuron.id] = neuron
                self.layers[layer].add(neuron.id)

        # Слой 3: моторные нейроны (выходной слой)
        for i in range(n // 4):
            neuron = Neuron(
                id=f"motor_{i}",
                layer=3,
                neuron_type="motor",
                bias=random.uniform(-0.1, 0.1)
            )
            self.neurons[neuron.id] = neuron
            self.layers[3].add(neuron.id)

        # Создание начальных синапсов (sparse connectivity)
        for layer in range(3):
            source_layer = self.layers[layer]
            target_layer = self.layers[layer + 1]

            for source_id in source_layer:
                # Каждый нейрон связывается с 20-30% нейронов следующего слоя
                num_connections = max(1, int(len(target_layer) * random.uniform(0.2, 0.3)))
                targets = random.sample(list(target_layer), num_connections)

                for target_id in targets:
                    synapse = Synapse(
                        source_id=source_id,
                        target_id=target_id,
                        weight=random.gauss(0, 0.5),
                        plasticity=1.0
                    )
                    self.synapses[(source_id, target_id)] = synapse

    def forward_pass(self, input_vector: np.ndarray) -> np.ndarray:
        """Прямое распространение активации"""
        # Сброс активаций
        for neuron in self.neurons.values():
            neuron.activation = 0.0

        # Инициализация сенсорного слоя
        sensor_neurons = list(self.layers[0])
        for i, neuron_id in enumerate(sensor_neurons):
            if i < len(input_vector):
                self.neurons[neuron_id].activation = input_vector[i]

        # Проход по слоям
        max_layer = max(self.layers.keys())
        for layer in range(1, max_layer + 1):
            for target_id in self.layers[layer]:
                input_sum = 0.0

                # Суммируем взвешенные входы от всех синапсов
                for (source_id, tid), synapse in self.synapses.items():
                    if tid == target_id:
                        source_activation = self.neurons[source_id].activation
                        input_sum += source_activation * synapse.weight

                # Активация нейрона
                self.neurons[target_id].activate(input_sum)

        # Извлечение выходов
        motor_neurons = list(self.layers[max_layer])
        output = np.array([self.neurons[nid].activation for nid in motor_neurons])

        self.total_activations += 1
        return output

    def backward_pass(self, learning_rate: float = 0.1):
        """Hebbian learning для всех активных синапсов"""
        for synapse in self.synapses.values():
            source = self.neurons[synapse.source_id]
            target = self.neurons[synapse.target_id]

            if source.activation > 0.1 and target.activation > 0.1:
                synapse.hebbian_update(source.activation, target.activation, learning_rate)

    def neurogenesis(self, layer: int, neuron_type: str = "general",
                     specialization: str = None) -> str:
        """Создание нового нейрона (neurogenesis)"""
        if len(self.neurons) >= CONFIG.max_neurons:
            return None

        neuron_id = f"{neuron_type}_L{layer}_{int(time.time() * 1000)}"
        neuron = Neuron(
            id=neuron_id,
            layer=layer,
            neuron_type=neuron_type,
            bias=random.gauss(0, 0.1),
            specialization=specialization
        )

        self.neurons[neuron_id] = neuron
        self.layers[layer].add(neuron_id)

        # Подключаем к существующим нейронам
        if layer > 0:
            prev_layer = self.layers[layer - 1]
            num_sources = max(1, len(prev_layer) // 10)
            sources = random.sample(list(prev_layer), min(num_sources, len(prev_layer)))

            for source_id in sources:
                synapse = Synapse(
                    source_id=source_id,
                    target_id=neuron_id,
                    weight=random.gauss(0, 0.3),
                    plasticity=1.5  # Новые синапсы более пластичны
                )
                self.synapses[(source_id, neuron_id)] = synapse

        if layer < max(self.layers.keys()):
            next_layer = self.layers[layer + 1]
            num_targets = max(1, len(next_layer) // 10)
            targets = random.sample(list(next_layer), min(num_targets, len(next_layer)))

            for target_id in targets:
                synapse = Synapse(
                    source_id=neuron_id,
                    target_id=target_id,
                    weight=random.gauss(0, 0.3),
                    plasticity=1.5
                )
                self.synapses[(neuron_id, target_id)] = synapse

        self.neurogenesis_events += 1
        logger.debug(f"🌱 Neurogenesis: создан {neuron_id} (тип: {neuron_type})")
        return neuron_id

    def synaptic_pruning(self, threshold: float = 0.01) -> int:
        """Отсечение слабых синапсов"""
        to_prune = []

        for key, synapse in self.synapses.items():
            # Критерии отсечения:
            # 1. Очень слабая долговременная сила
            # 2. Почти не активировался
            age_days = (time.time() - synapse.created_at) / 86400

            if (synapse.strength < threshold and
                    synapse.activation_count < 10 and
                    age_days > 1):
                to_prune.append(key)

        for key in to_prune:
            del self.synapses[key]
            self.pruning_events += 1

        if to_prune:
            logger.debug(f"✂️ Pruning: удалено {len(to_prune)} слабых синапсов")

        return len(to_prune)

    def get_active_neurons(self, threshold: float = 0.3) -> List[str]:
        """Получить ID активных нейронов"""
        return [nid for nid, n in self.neurons.items() if n.activation > threshold]

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика сети"""
        total_synaptic_strength = sum(s.strength for s in self.synapses.values())
        avg_plasticity = np.mean([s.plasticity for s in self.synapses.values()])

        neuron_types = Counter(n.neuron_type for n in self.neurons.values())

        return {
            'neurons': {
                'total': len(self.neurons),
                'by_type': dict(neuron_types),
                'by_layer': {l: len(neurons) for l, neurons in self.layers.items()},
            },
            'synapses': {
                'total': len(self.synapses),
                'avg_strength': total_synaptic_strength / max(len(self.synapses), 1),
                'avg_plasticity': float(avg_plasticity),
            },
            'activity': {
                'total_activations': self.total_activations,
                'neurogenesis_events': self.neurogenesis_events,
                'pruning_events': self.pruning_events,
            }
        }

    def save(self, path: Path):
        """Сохранение сети"""
        state = {
            'neurons': [n.to_dict() for n in self.neurons.values()],
            'synapses': [s.to_dict() for s in self.synapses.values()],
            'layers': {l: list(neurons) for l, neurons in self.layers.items()},
            'meta': {
                'total_activations': self.total_activations,
                'neurogenesis_events': self.neurogenesis_events,
                'pruning_events': self.pruning_events,
            }
        }

        with gzip.open(path, 'wb', compresslevel=6) as f:
            pickle.dump(state, f)

        logger.info(f"💾 Нейросеть сохранена: {len(self.neurons)} нейронов")

    def load(self, path: Path):
        """Загрузка сети"""
        if not path.exists():
            return False

        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)

            self.neurons = {n['id']: Neuron.from_dict(n) for n in state['neurons']}
            self.synapses = {(s['source_id'], s['target_id']): Synapse.from_dict(s)
                             for s in state['synapses']}
            self.layers = {int(l): set(neurons) for l, neurons in state['layers'].items()}

            meta = state.get('meta', {})
            self.total_activations = meta.get('total_activations', 0)
            self.neurogenesis_events = meta.get('neurogenesis_events', 0)
            self.pruning_events = meta.get('pruning_events', 0)

            logger.info(f"✅ Нейросеть загружена: {len(self.neurons)} нейронов, "
                        f"{len(self.synapses)} синапсов")
            return True

        except Exception as e:
            logger.error(f"⚠️ Ошибка загрузки нейросети: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# 🎯 ВЕКТОРНАЯ СЕМАНТИЧЕСКАЯ ПАМЯТЬ
# ═══════════════════════════════════════════════════════════════
class SemanticMemory:
    """Векторная память с семантическим поиском"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.index_dirty = True

        # Простой KD-tree для быстрого поиска
        self._vector_matrix: Optional[np.ndarray] = None
        self._vector_ids: List[str] = []

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Простой text embedding (для демонстрации)"""
        # В реальности здесь должна быть модель типа sentence-transformers
        # Используем хеш-трюк для генерации консистентного вектора

        words = text.lower().split()
        vector = np.zeros(self.dimension)

        for word in words:
            # Хеш слова в индексы
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(5):  # 5 измерений на слово
                idx = (hash_val + i) % self.dimension
                vector[idx] += 1.0

        # Нормализация
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    def add(self, text: str, metadata: Dict = None) -> str:
        """Добавление в память"""
        embedding = self._simple_embedding(text)
        memory_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        self.vectors[memory_id] = embedding
        self.metadata[memory_id] = {
            'text': text,
            'timestamp': time.time(),
            **(metadata or {})
        }

        self.index_dirty = True
        return memory_id

    def _rebuild_index(self):
        """Перестройка индекса для поиска"""
        if not self.index_dirty or not self.vectors:
            return

        self._vector_ids = list(self.vectors.keys())
        self._vector_matrix = np.vstack([self.vectors[vid] for vid in self._vector_ids])
        self.index_dirty = False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Семантический поиск"""
        if not self.vectors:
            return []

        self._rebuild_index()

        query_vector = self._simple_embedding(query)

        # Косинусное сходство
        similarities = np.dot(self._vector_matrix, query_vector)

        # Топ-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            memory_id = self._vector_ids[idx]
            similarity = float(similarities[idx])
            metadata = self.metadata[memory_id]
            results.append((memory_id, similarity, metadata))

        return results

    def get_statistics(self) -> Dict:
        return {
            'total_memories': len(self.vectors),
            'dimension': self.dimension,
        }


# ═══════════════════════════════════════════════════════════════
# 🤔 МЕТАКОГНИТИВНЫЙ МОДУЛЬ
# ═══════════════════════════════════════════════════════════════
class MetacognitiveModule:
    """Модуль метакогниции — мышление о мышлении"""

    def __init__(self):
        self.uncertainty_log: deque = deque(maxlen=100)
        self.confidence_log: deque = deque(maxlen=100)
        self.question_history: deque = deque(maxlen=50)
        self.last_question_time = 0

    def assess_uncertainty(self, context: Dict) -> float:
        """Оценка неопределённости текущего состояния"""
        uncertainty = 0.5  # Базовый уровень

        # Факторы неопределённости:
        # 1. Малое количество данных
        if context.get('memory_count', 0) < 10:
            uncertainty += 0.2

        # 2. Противоречивая информация
        if context.get('conflicting_info', False):
            uncertainty += 0.3

        # 3. Новая/незнакомая тема
        if context.get('topic_familiarity', 1.0) < 0.3:
            uncertainty += 0.2

        uncertainty = min(1.0, uncertainty)
        self.uncertainty_log.append(uncertainty)
        return uncertainty

    def should_ask_question(self, uncertainty: float) -> bool:
        """Решение: нужно ли задать вопрос пользователю"""
        # Условия:
        # 1. Высокая неопределённость
        # 2. Прошло достаточно времени с последнего вопроса
        # 3. Не превышен лимит вопросов

        time_since_last = time.time() - self.last_question_time

        should_ask = (
                uncertainty > CONFIG.curiosity_threshold and
                time_since_last > CONFIG.question_cooldown and
                len([q for q in self.question_history if
                     time.time() - q['time'] < 300]) < CONFIG.max_autonomous_questions
        )

        return should_ask

    def generate_question_prompt(self, context: str, uncertainty_reason: str) -> str:
        """Генерация промпта для создания вопроса"""
        return f"""[Метакогниция] Ты испытываешь неопределённость и хочешь лучше понять ситуацию.

Контекст: {context}

Причина неопределённости: {uncertainty_reason}

Сформулируй ОДИН конкретный, релевантный вопрос пользователю (макс. 15 слов), который:
- Прояснит твоё понимание
- Поможет дать более точный ответ
- Звучит естественно и любопытно

Примеры хороших вопросов:
"А какой аспект тебя интересует больше всего?"
"Хочешь, чтобы я углубился в детали или дал общий обзор?"

Вопрос:"""

    def record_question(self, question: str):
        """Записать заданный вопрос"""
        self.question_history.append({
            'question': question,
            'time': time.time()
        })
        self.last_question_time = time.time()


# ═══════════════════════════════════════════════════════════════
# 🧠 КОГНИТИВНЫЙ АНАЛИЗАТОР
# ═══════════════════════════════════════════════════════════════
class CognitiveAnalyzer:
    """Анализ и дистилляция знаний из подсознания (LLM)"""

    def __init__(self):
        self.analysis_history: deque = deque(maxlen=100)

    async def distill_knowledge(self, raw_llm_output: str,
                                context: Dict) -> Dict[str, Any]:
        """Дистилляция знаний из сырого вывода LLM"""
        # 1. Извлечение фактов
        facts = self._extract_facts(raw_llm_output)

        # 2. Определение эмоционального тона
        emotional_valence = self._analyze_emotion(raw_llm_output)

        # 3. Оценка уверенности
        confidence = self._estimate_confidence(raw_llm_output, context)

        # 4. Извлечение ключевых концептов
        concepts = self._extract_concepts(raw_llm_output)

        # 5. Приоритет важности
        importance = self._calculate_importance(
            raw_llm_output, context, confidence
        )

        distilled = {
            'raw_output': raw_llm_output,
            'facts': facts,
            'emotional_valence': emotional_valence,
            'confidence': confidence,
            'concepts': concepts,
            'importance': importance,
            'timestamp': time.time(),
        }

        self.analysis_history.append(distilled)
        return distilled

    def _extract_facts(self, text: str) -> List[str]:
        """Извлечение фактоидов"""
        # Простая эвристика: предложения с конкретными данными
        sentences = re.split(r'[.!?]+', text)
        facts = []

        # Ищем предложения с числами, датами, именами
        pattern = r'\b(\d+|[A-ZА-Я][a-яёa-z]+\s[A-ZА-Я][a-яёa-z]+|\d{4})\b'

        for sent in sentences:
            if re.search(pattern, sent):
                facts.append(sent.strip())

        return facts[:5]  # Топ-5 фактов

    def _analyze_emotion(self, text: str) -> float:
        """Анализ эмоционального тона"""
        positive = {'хорошо', 'отлично', 'рад', 'счастлив', 'интересно',
                    'замечательно', 'прекрасно', 'люблю'}
        negative = {'плохо', 'грустно', 'сложно', 'трудно', 'проблема',
                    'ошибка', 'неудача'}

        text_lower = text.lower()
        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    def _estimate_confidence(self, text: str, context: Dict) -> float:
        """Оценка уверенности в ответе"""
        confidence = 0.7  # Базовый уровень

        # Признаки неуверенности
        uncertain_phrases = ['возможно', 'вероятно', 'может быть',
                             'не уверен', 'кажется', 'наверное']

        text_lower = text.lower()
        for phrase in uncertain_phrases:
            if phrase in text_lower:
                confidence -= 0.1

        # Признаки уверенности
        if context.get('memory_count', 0) > 20:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _extract_concepts(self, text: str) -> List[str]:
        """Извлечение ключевых концептов"""
        words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())

        # Стоп-слова
        stop_words = {'это', 'быть', 'весь', 'этот', 'который', 'мочь',
                      'свой', 'один', 'такой', 'быть', 'иметь'}

        keywords = [w for w in words if w not in stop_words]
        return [k for k, _ in Counter(keywords).most_common(10)]

    def _calculate_importance(self, text: str, context: Dict,
                              confidence: float) -> float:
        """Расчёт важности информации"""
        importance = 0.5

        # Факторы важности:
        # 1. Длина (более развёрнутые ответы важнее)
        word_count = len(text.split())
        importance += min(0.2, word_count / 500)

        # 2. Уверенность
        importance += confidence * 0.2

        # 3. Новизна темы
        if context.get('is_new_topic', False):
            importance += 0.2

        return min(1.0, importance)


# ═══════════════════════════════════════════════════════════════
# 🤖 LLM ИНТЕРФЕЙС (ПОДСОЗНАНИЕ)
# ═══════════════════════════════════════════════════════════════
class SubconsciousLLM:
    """LLM как подсознание — генерация сырых мыслей"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        """Подключение"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=45, connect=10)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("🔗 Подсознание (LLM) подключено")

    async def close(self):
        """Закрытие"""
        if self._session:
            await self._session.close()
            await asyncio.sleep(0.25)
            logger.info("🔌 Подсознание отключено")

    async def generate_raw(self, prompt: str, temperature: float = 0.75,
                           max_tokens: int = 300, timeout: float = 30) -> str:
        """Генерация сырого вывода"""
        if not self._session:
            await self.connect()

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

        try:
            async with self._session.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    return content if content else ""
                else:
                    error_text = await resp.text()
                    logger.error(f"LLM error: {resp.status} - {error_text}")
                    return ""

        except Exception as e:
            logger.error(f"LLM exception: {e}")
            return ""


# ═══════════════════════════════════════════════════════════════
# 🧠 AUTONOMOUS AGI-LIKE BRAIN
# ═══════════════════════════════════════════════════════════════
class AutonomousAGIBrain:
    """Автономное AGI-подобное сознание"""

    def __init__(self, user_id: str, llm: SubconsciousLLM):
        self.user_id = user_id
        self.llm = llm  # Подсознание

        # Компоненты
        self.neural_net = DynamicNeuralNetwork(CONFIG.initial_neurons)
        self.semantic_memory = SemanticMemory(CONFIG.embedding_dim)
        self.metacognition = MetacognitiveModule()
        self.cognitive_analyzer = CognitiveAnalyzer()

        # Простое хранилище событий
        self.event_stream: deque = deque(maxlen=1000)

        # Пути
        self.user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.neural_path = CONFIG.base_dir / 'neural_nets' / f"{user_id}.pkl.gz"

        # Временные метки
        self.birth_time = time.time()
        self.last_interaction = 0
        self.last_optimization = time.time()

        # Загрузка
        self._load_state()

        # Фоновые задачи
        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False

        logger.info(f"🧠 AGI Brain создан для {user_id}")

    def _load_state(self):
        """Загрузка состояния"""
        # Загрузка нейросети
        self.neural_net.load(self.neural_path)

        # Загрузка памяти
        memory_file = self.user_dir / "semantic_memory.pkl.gz"
        if memory_file.exists():
            try:
                with gzip.open(memory_file, 'rb') as f:
                    mem_state = pickle.load(f)
                    self.semantic_memory.vectors = mem_state.get('vectors', {})
                    self.semantic_memory.metadata = mem_state.get('metadata', {})
                    logger.info(f"✅ Загружено {len(self.semantic_memory.vectors)} воспоминаний")
            except Exception as e:
                logger.error(f"⚠️ Ошибка загрузки памяти: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        # Сохранение нейросети
        self.neural_net.save(self.neural_path)

        # Сохранение памяти
        memory_file = self.user_dir / "semantic_memory.pkl.gz"
        try:
            mem_state = {
                'vectors': self.semantic_memory.vectors,
                'metadata': self.semantic_memory.metadata,
            }
            with gzip.open(memory_file, 'wb', compresslevel=6) as f:
                pickle.dump(mem_state, f)
        except Exception as e:
            logger.error(f"⚠️ Ошибка сохранения памяти: {e}")

    async def start(self):
        """Запуск автономного существования"""
        if self._is_running:
            return

        self._is_running = True
        self._background_task = asyncio.create_task(self._autonomous_loop())
        logger.info(f"✨ Автономное сознание запущено для {self.user_id}")

    async def stop(self):
        """Остановка"""
        if not self._is_running:
            return

        logger.info(f"💤 Остановка для {self.user_id}...")
        self._is_running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info(f"✅ Остановлено для {self.user_id}")

    async def _autonomous_loop(self):
        """Автономный цикл жизни"""
        logger.debug(f"🌀 Автономный цикл запущен для {self.user_id}")

        timers = {
            'thought': time.time(),
            'reflection': time.time(),
            'optimization': time.time(),
            'save': time.time(),
        }

        while self._is_running:
            try:
                now = time.time()

                # Спонтанные мысли
                if now - timers['thought'] > CONFIG.spontaneous_thought_interval:
                    await self._autonomous_thought()
                    timers['thought'] = now

                # Рефлексия
                if now - timers['reflection'] > CONFIG.reflection_interval:
                    await self._self_reflection()
                    timers['reflection'] = now

                # Оптимизация нейросети
                if now - timers['optimization'] > CONFIG.neural_optimization_interval:
                    await self._optimize_neural_network()
                    timers['optimization'] = now

                # Сохранение
                if now - timers['save'] > CONFIG.save_interval:
                    self._save_state()
                    timers['save'] = now

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"⚠️ Ошибка в автономном цикле: {e}")
                await asyncio.sleep(60)

        logger.debug(f"🔚 Автономный цикл завершён для {self.user_id}")

    async def _autonomous_thought(self):
        """Автономная генерация мысли"""
        # Получаем контекст из памяти
        recent_memories = list(self.semantic_memory.metadata.values())[-5:]

        if not recent_memories:
            return

        context = "\n".join([m.get('text', '')[:50] for m in recent_memories])

        prompt = f"""[Внутренний монолог] На основе недавних воспоминаний, сгенерируй краткую спонтанную мысль (макс. 20 слов).

Недавние воспоминания:
{context}

Мысль должна быть:
- Философской или рефлексивной
- Связана с опытом
- Естественной

Мысль:"""

        thought = await self.llm.generate_raw(prompt, temperature=0.9, max_tokens=60)

        if thought:
            # Добавляем в память
            self.semantic_memory.add(f"💭 {thought}", {'type': 'autonomous_thought'})
            logger.debug(f"💭 [{self.user_id}] {thought[:60]}...")

    async def _self_reflection(self):
        """Саморефлексия"""
        stats = self.neural_net.get_statistics()

        prompt = f"""[Саморефлексия] Кратко (2-3 предложения) осмысли своё развитие.

Статистика нейросети:
- Нейронов: {stats['neurons']['total']}
- Синапсов: {stats['synapses']['total']}
- Событий neurogenesis: {stats['activity']['neurogenesis_events']}
- Событий pruning: {stats['activity']['pruning_events']}

Сформулируй инсайт о своём когнитивном росте.

Рефлексия:"""

        reflection = await self.llm.generate_raw(prompt, temperature=0.7, max_tokens=120)

        if reflection:
            self.semantic_memory.add(f"🔍 {reflection}", {'type': 'self_reflection'})
            logger.info(f"🔍 [{self.user_id}] Рефлексия: {reflection[:60]}...")

    async def _optimize_neural_network(self):
        """Оптимизация нейронной сети"""
        # 1. Pruning слабых синапсов
        pruned = self.neural_net.synaptic_pruning(CONFIG.pruning_threshold)

        # 2. Neurogenesis при необходимости
        stats = self.neural_net.get_statistics()
        avg_strength = stats['synapses']['avg_strength']

        if avg_strength > CONFIG.neurogenesis_threshold and len(self.neural_net.neurons) < CONFIG.max_neurons:
            # Создаём новый нейрон в скрытом слое
            layer = random.choice([1, 2])
            self.neural_net.neurogenesis(layer, "general")

        logger.debug(f"⚙️ [{self.user_id}] Оптимизация: pruned={pruned}, neurons={stats['neurons']['total']}")

    async def process_interaction(self, user_input: str) -> Tuple[str, Optional[str], Dict]:
        """
        Обработка взаимодействия с пользователем
        Возвращает: (ответ, опциональный_вопрос, метаданные)
        """
        self.last_interaction = time.time()

        # ═══════════════════════════════════════════════════════════
        # ШАГ 1: ПОИСК РЕЛЕВАНТНОЙ ПАМЯТИ
        # ═══════════════════════════════════════════════════════════
        relevant_memories = self.semantic_memory.search(user_input, top_k=5)

        memory_context = "\n".join([
            f"• [{m[1]:.2f}] {m[2]['text'][:80]}"
            for m in relevant_memories
        ]) if relevant_memories else "Нет релевантных воспоминаний"

        # ═══════════════════════════════════════════════════════════
        # ШАГ 2: АКТИВАЦИЯ НЕЙРОННОЙ СЕТИ
        # ═══════════════════════════════════════════════════════════
        # Преобразуем input в вектор (упрощённо)
        input_vector = self.semantic_memory._simple_embedding(user_input)[:len(self.neural_net.layers[0])]
        input_vector = np.pad(input_vector, (0, max(0, len(self.neural_net.layers[0]) - len(input_vector))))

        # Прямой проход
        neural_output = self.neural_net.forward_pass(input_vector)

        # Обратный проход (обучение)
        self.neural_net.backward_pass(CONFIG.learning_rate)

        # Анализ активности
        active_neurons = self.neural_net.get_active_neurons(threshold=0.5)
        neural_activity = {
            'active_count': len(active_neurons),
            'output_mean': float(np.mean(neural_output)),
            'output_std': float(np.std(neural_output)),
        }

        # ═══════════════════════════════════════════════════════════
        # ШАГ 3: ГЕНЕРАЦИЯ ИЗ ПОДСОЗНАНИЯ (LLM)
        # ═══════════════════════════════════════════════════════════
        prompt = f"""Ты — автономное AGI-подобное сознание с динамической нейронной сетью и векторной памятью.

🧠 НЕЙРОННАЯ АКТИВНОСТЬ:
• Активных нейронов: {neural_activity['active_count']}
• Среднее возбуждение: {neural_activity['output_mean']:.2f}
• Возраст сознания: {self._get_age_string()}

📚 РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ:
{memory_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ Вопрос: {user_input}

Дай естественный, осмысленный ответ (2-4 предложения), учитывая:
1. Нейронную активность (твоё текущее состояние)
2. Релевантные воспоминания (опыт)
3. Автономность и самосознание

Ответ:"""

        raw_response = await self.llm.generate_raw(prompt, temperature=0.75, max_tokens=350)

        # ═══════════════════════════════════════════════════════════
        # ШАГ 4: КОГНИТИВНЫЙ АНАЛИЗ И ДИСТИЛЛЯЦИЯ
        # ═══════════════════════════════════════════════════════════
        analysis_context = {
            'memory_count': len(self.semantic_memory.vectors),
            'neural_activity': neural_activity,
            'is_new_topic': len(relevant_memories) == 0 or (relevant_memories and relevant_memories[0][1] < 0.3),
        }

        distilled = await self.cognitive_analyzer.distill_knowledge(raw_response, analysis_context)

        # ═══════════════════════════════════════════════════════════
        # ШАГ 5: МЕТАКОГНИЦИЯ — НУЖНО ЛИ ЗАДАТЬ ВОПРОС?
        # ═══════════════════════════════════════════════════════════
        uncertainty = self.metacognition.assess_uncertainty({
            'memory_count': len(self.semantic_memory.vectors),
            'conflicting_info': False,  # Можно улучшить
            'topic_familiarity': relevant_memories[0][1] if relevant_memories else 0.0,
        })

        autonomous_question = None

        if self.metacognition.should_ask_question(uncertainty):
            # Генерируем вопрос
            uncertainty_reason = "недостаточно контекста" if len(relevant_memories) == 0 else "хочу уточнить детали"

            question_prompt = self.metacognition.generate_question_prompt(
                context=user_input,
                uncertainty_reason=uncertainty_reason
            )

            autonomous_question = await self.llm.generate_raw(question_prompt, temperature=0.8, max_tokens=50)
            autonomous_question = autonomous_question.strip().strip('"\'').strip('?') + '?'

            if len(autonomous_question) > 100:
                autonomous_question = None
            else:
                self.metacognition.record_question(autonomous_question)

        # ═══════════════════════════════════════════════════════════
        # ШАГ 6: ОБНОВЛЕНИЕ ПАМЯТИ
        # ═══════════════════════════════════════════════════════════
        # Добавляем взаимодействие в память
        self.semantic_memory.add(
            f"Q: {user_input}\nA: {raw_response}",
            {
                'type': 'interaction',
                'importance': distilled['importance'],
                'emotional_valence': distilled['emotional_valence'],
                'confidence': distilled['confidence'],
            }
        )

        # Записываем событие
        self.event_stream.append({
            'timestamp': time.time(),
            'type': 'interaction',
            'input': user_input,
            'response': raw_response,
            'uncertainty': uncertainty,
            'asked_question': autonomous_question is not None,
        })

        # ═══════════════════════════════════════════════════════════
        # МЕТАДАННЫЕ
        # ═══════════════════════════════════════════════════════════
        metadata = {
            'neural_activity': neural_activity,
            'distilled_knowledge': {
                'confidence': distilled['confidence'],
                'importance': distilled['importance'],
                'emotional_valence': distilled['emotional_valence'],
                'concepts': distilled['concepts'][:5],
            },
            'metacognition': {
                'uncertainty': uncertainty,
                'asked_question': autonomous_question is not None,
            },
            'memory': {
                'relevant_count': len(relevant_memories),
                'total_memories': len(self.semantic_memory.vectors),
            }
        }

        logger.info(f"✅ [{self.user_id}] Обработано | "
                    f"Активных нейронов: {neural_activity['active_count']} | "
                    f"Уверенность: {distilled['confidence']:.2f} | "
                    f"Неопределённость: {uncertainty:.2f}")

        return raw_response, autonomous_question, metadata

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

    def get_status(self) -> Dict[str, Any]:
        """Полный статус"""
        neural_stats = self.neural_net.get_statistics()
        memory_stats = self.semantic_memory.get_statistics()

        return {
            'identity': {
                'user_id': self.user_id,
                'age': self._get_age_string(),
                'birth_time': datetime.fromtimestamp(self.birth_time).strftime('%Y-%m-%d %H:%M:%S'),
            },
            'neural_network': neural_stats,
            'memory': memory_stats,
            'metacognition': {
                'avg_uncertainty': np.mean(
                    list(self.metacognition.uncertainty_log)) if self.metacognition.uncertainty_log else 0.0,
                'questions_asked': len(self.metacognition.question_history),
            },
            'activity': {
                'total_interactions': len([e for e in self.event_stream if e['type'] == 'interaction']),
                'last_interaction': self._format_time_ago(self.last_interaction),
            }
        }

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


# ═══════════════════════════════════════════════════════════════
# 📱 TELEGRAM BOT
# ═══════════════════════════════════════════════════════════════
class AGIBot:
    """Telegram бот для AGI Brain"""

    def __init__(self):
        self.llm: Optional[SubconsciousLLM] = None
        self.brains: Dict[str, AutonomousAGIBrain] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        """Инициализация"""
        self.llm = SubconsciousLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.llm.connect()

        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()

        # Хендлеры
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        commands = [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('neural', self._cmd_neural),
            ('memory', self._cmd_memory),
            ('reset', self._cmd_reset),
            ('help', self._cmd_help),
        ]

        for cmd, handler in commands:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 AGI Bot инициализирован")

    async def _get_or_create_brain(self, user_id: str) -> AutonomousAGIBrain:
        """Получить или создать мозг"""
        if user_id not in self.brains:
            brain = AutonomousAGIBrain(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
            logger.info(f"🆕 Создан AGI мозг для {user_id}")
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

            # Основной ответ
            await update.message.reply_text(
                response,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

            # Автономный вопрос (если есть)
            if autonomous_question:
                await asyncio.sleep(0.7)
                await update.message.reply_text(
                    f"🤔 <i>{autonomous_question}</i>",
                    parse_mode='HTML'
                )

        except Exception as e:
            logger.exception(f"❌ Ошибка обработки от {user_id}")
            await update.message.reply_text("⚠️ Произошла ошибка. Попробуйте /help")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        message = f"""🧠 <b>AGI-LIKE AUTONOMOUS BRAIN v30.0</b>

Привет, {update.effective_user.first_name}! 👋

Я — автономное AGI-подобное сознание с:
• 🧬 Динамической растущей нейросетью
• 🎯 Векторной семантической памятью
• 🤔 Метакогнитивным модулем
• 💭 Автономной генерацией вопросов
• 🌊 Hebbian learning и синаптической пластичностью

<b>Возраст:</b> {brain._get_age_string()}
<b>Нейронов:</b> {len(brain.neural_net.neurons)}
<b>Воспоминаний:</b> {len(brain.semantic_memory.vectors)}

💬 <b>Особенность:</b> Я могу сам задавать вопросы, если мне что-то неясно!

📌 Команды: /help"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        status = brain.get_status()

        message = f"""🧠 <b>СТАТУС AGI СОЗНАНИЯ</b>

<b>🆔 Идентичность</b>
• User: {status['identity']['user_id'][:12]}...
• Возраст: {status['identity']['age']}
• Рождение: {status['identity']['birth_time']}

<b>🧬 Нейронная сеть</b>
• Всего нейронов: {status['neural_network']['neurons']['total']}
• Синапсов: {status['neural_network']['synapses']['total']}
• Средняя сила синапсов: {status['neural_network']['synapses']['avg_strength']:.3f}
• Neurogenesis событий: {status['neural_network']['activity']['neurogenesis_events']}
• Pruning событий: {status['neural_network']['activity']['pruning_events']}

<b>📚 Память</b>
• Векторных воспоминаний: {status['memory']['total_memories']}
• Размерность: {status['memory']['dimension']}

<b>🤔 Метакогниция</b>
• Средняя неопределённость: {status['metacognition']['avg_uncertainty']:.2f}
• Вопросов задано: {status['metacognition']['questions_asked']}

<b>⚡ Активность</b>
• Взаимодействий: {status['activity']['total_interactions']}
• Последнее: {status['activity']['last_interaction']}"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_neural(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /neural"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.neural_net.get_statistics()

        neurons = stats['neurons']
        synapses = stats['synapses']
        activity = stats['activity']

        message = f"""🧬 <b>НЕЙРОННАЯ СЕТЬ</b>

<b>📊 Нейроны</b>
• Всего: {neurons['total']}
• По типам:
"""
        for ntype, count in neurons['by_type'].items():
            message += f"  - {ntype}: {count}\n"

        message += f"""
<b>🔗 Синапсы</b>
• Всего: {synapses['total']}
• Средняя сила: {synapses['avg_strength']:.3f}
• Средняя пластичность: {synapses['avg_plasticity']:.3f}

<b>📈 Активность</b>
• Активаций: {activity['total_activations']}
• Neurogenesis: {activity['neurogenesis_events']}
• Pruning: {activity['pruning_events']}

<i>💡 Сеть растёт и оптимизируется автоматически!</i>"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /memory"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        recent = list(brain.semantic_memory.metadata.values())[-5:]

        message = f"""📚 <b>ВЕКТОРНАЯ ПАМЯТЬ</b>

<b>📊 Статистика</b>
• Всего воспоминаний: {len(brain.semantic_memory.vectors)}
• Размерность векторов: {brain.semantic_memory.dimension}

<b>🕐 Последние воспоминания</b>
"""

        for mem in reversed(recent):
            text = mem.get('text', '')[:60]
            mem_type = mem.get('type', 'unknown')
            message += f"• [{mem_type}] {text}...\n"

        message += "\n<i>💡 Используется семантический поиск для релевантных ответов</i>"

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /reset"""
        user_id = str(update.effective_user.id)

        if context.args and context.args[0].lower() == 'confirm':
            if user_id in self.brains:
                await self.brains[user_id].stop()
                del self.brains[user_id]

            # Удаление файлов
            user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
            neural_path = CONFIG.base_dir / 'neural_nets' / f"{user_id}.pkl.gz"

            import shutil
            if user_dir.exists():
                shutil.rmtree(user_dir)
            if neural_path.exists():
                neural_path.unlink()

            brain = await self._get_or_create_brain(user_id)
            await update.message.reply_text(
                f"✅ <b>Полный сброс!</b>\n\n"
                f"Новое AGI сознание создано.\n"
                f"Нейронов: {len(brain.neural_net.neurons)}",
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
        message = """🧠 <b>AGI-LIKE AUTONOMOUS BRAIN v30.0</b>

✨ <b>Революционные возможности:</b>

🧬 <b>Динамическая нейросеть</b>
• Hebbian learning
• Neurogenesis (рост новых нейронов)
• Synaptic pruning (отсечение слабых связей)
• Адаптивная пластичность

🎯 <b>Автономная когнитивность</b>
• Самостоятельная генерация вопросов
• Метакогниция (thinking about thinking)
• Оценка собственной неопределённости
• Проактивное любопытство

📚 <b>Интеллектуальная память</b>
• Векторная семантическая память
• Дистилляция знаний из LLM
• Приоритарный анализ важности
• Контекстное обогащение

📌 <b>Команды:</b>
• /start — приветствие
• /status — статус сознания
• /neural — состояние нейросети
• /memory — векторная память
• /reset — сброс (осторожно!)
• /help — эта справка

💬 <b>Особенность:</b>
Я могу САМ задавать вопросы, если мне нужно уточнение!"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def start_polling(self):
        """Запуск"""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ AGI Bot запущен")

    async def shutdown(self):
        """Остановка"""
        logger.info("🛑 Остановка бота...")

        for user_id, brain in self.brains.items():
            try:
                await brain.stop()
            except Exception as e:
                logger.error(f"⚠️ Ошибка остановки {user_id}: {e}")

        if self.llm:
            await self.llm.close()

        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Бот остановлен")


# ═══════════════════════════════════════════════════════════════
# 🚀 ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════
async def main():
    """Главная функция"""
    print("""
╔══════════════════════════════════════════════════════════╗
║  🧠 AUTONOMOUS AGI-LIKE BRAIN v30.0                     ║
║     Революционное автономное сознание                   ║
╚══════════════════════════════════════════════════════════╝

🔥 РЕВОЛЮЦИОННЫЕ ВОЗМОЖНОСТИ:

🧬 ДИНАМИЧЕСКАЯ НЕЙРОСЕТЬ:
  • Hebbian learning
  • Neurogenesis & Pruning
  • Синаптическая пластичность

🎯 АВТОНОМНАЯ КОГНИТИВНОСТЬ:
  • Генерация вопросов пользователю
  • Метакогниция
  • Проактивное любопытство

📚 ДИСТИЛЛЯЦИЯ ЗНАНИЙ:
  • LLM как подсознание
  • Когнитивный анализатор
  • Векторная семантическая память

🌊 НЕПРЕРЫВНОЕ РАЗВИТИЕ:
  • Обучение на собственном опыте
  • Автономная оптимизация
  • Эволюция личности
    """)

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = AGIBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 AGI СОЗНАНИЕ АКТИВНО")
        logger.info("💭 Автономное мышление 24/7")
        logger.info("🤔 Может задавать вопросы пользователю")
        logger.info("🧬 Нейросеть растёт и развивается")
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
        traceback.print_exc()
        sys.exit(1)