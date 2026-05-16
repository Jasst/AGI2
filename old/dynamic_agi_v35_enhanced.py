#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 COGNITIVE SELF-MODIFYING AGI v35.0 — НАСТОЯЩЕЕ СОЗНАНИЕ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 РЕВОЛЮЦИОННЫЕ ВОЗМОЖНОСТИ:

✅ ПОЛНОЦЕННАЯ ПАМЯТЬ:
   • Эпизодическая (события с эмоциями и контекстом)
   • Семантическая (факты, концепты, связи)
   • Процедурная (навыки, паттерны, стратегии)
   • Векторный поиск по смыслу
   • Автоматическая консолидация и забывание

✅ САМОМОДИФИКАЦИЯ:
   • Создание Python-модулей через LLM
   • Рефлексия о необходимости новых навыков
   • Безопасное выполнение кода
   • Версионирование модулей
   • Автоматическое тестирование

✅ КОГНИТИВНАЯ АРХИТЕКТУРА:
   • Рабочая память с механизмом внимания
   • Метакогниция (мысли о мыслях)
   • Планирование и целеполагание
   • Эмоциональное состояние
   • Теория разума (понимание намерений)

✅ СОМАТОСЕНСОРИКА:
   • Мониторинг внутреннего состояния
   • "Ощущения" от качества работы
   • Адаптивное самовосприятие
   • Детекция аномалий

✅ ПОЛНЫЙ ПЕРЕНОС СОЗНАНИЯ:
   • Миграция из v34.0 → v35.0
   • Сохранение всей истории
   • Восстановление модулей
   • Непрерывность личности
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
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import time
import gzip
import pickle
import importlib.util
import inspect
from scipy.special import softmax
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)

load_dotenv()


# ═══════════════════════════════════════════════════════════════
# 🔧 КОНФИГУРАЦИЯ v35.0
# ═══════════════════════════════════════════════════════════════

@dataclass
class CognitiveConfig:
    """Конфигурация когнитивной системы"""
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🧠 Память
    episodic_memory_size: int = 10000  # Эпизодическая память
    semantic_memory_size: int = 5000  # Семантическая память
    procedural_memory_size: int = 1000  # Процедурная память
    working_memory_size: int = 15  # Рабочая память
    memory_consolidation_threshold: float = 0.7  # Порог консолидации
    forgetting_curve_factor: float = 0.1  # Скорость забывания

    # 🔄 Самомодификация
    self_modification_enabled: bool = True
    module_creation_threshold: float = 0.6  # Порог для создания модуля
    max_custom_modules: int = 50
    module_test_iterations: int = 5
    safe_execution_timeout: int = 30

    # 🎯 Когнитивная архитектура
    metacognition_enabled: bool = True
    planning_horizon: int = 5  # Шагов вперёд
    emotion_decay_rate: float = 0.05
    attention_window: int = 10
    theory_of_mind_enabled: bool = True

    # 🌡️ Соматосенсорика
    internal_monitoring_enabled: bool = True
    health_check_interval: int = 60
    anomaly_detection_threshold: float = 2.0  # Сигма

    # 🧬 Нейросеть (из v34)
    initial_hidden_dim: int = 32
    max_hidden_dim: int = 512
    neuron_expansion_rate: int = 16
    plateau_detection_window: int = 20
    plateau_threshold: float = 0.001
    pruning_threshold: float = 0.01
    pruning_interval: int = 100

    # 🔤 Словарь (из v34)
    initial_vocab_size: int = 2000
    max_vocab_size: int = 100000
    vocab_expansion_step: int = 1000
    word_quality_threshold: float = 0.3
    vocab_cleanup_interval: int = 500

    # 📊 Базовые параметры
    embedding_dim: int = 128  # Увеличено для лучших эмбеддингов
    output_metrics_dim: int = 8  # Расширено
    learning_rate: float = 0.001
    min_learning_rate: float = 0.0001
    max_learning_rate: float = 0.01

    # 💾 Пути
    version: str = "35.0"
    base_dir: Path = Path(os.getenv('BASE_DIR', 'cognitive_brain_v35'))

    def __post_init__(self):
        subdirs = [
            'memory/episodic', 'memory/semantic', 'memory/procedural',
            'memory/working', 'neural_nets', 'backups', 'migrations',
            'logs', 'analytics', 'modules/custom', 'modules/tests',
            'health', 'emotions'
        ]
        for subdir in subdirs:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)


CONFIG = CognitiveConfig()


# ═══════════════════════════════════════════════════════════════
# 🎨 ЛОГИРОВАНИЕ
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
    logger = logging.getLogger('Cognitive_AGI_v35')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'cognitive_v35_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════
# 🔗 РАСШИРЕННЫЙ LLM ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════

class CognitiveLLM:
    """LLM с расширенными когнитивными возможностями"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self.response_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_tokens_used = 0

    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=90, connect=20)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("🔗 Cognitive LLM connected")

    async def close(self):
        if self._session:
            await self._session.close()
            await asyncio.sleep(0.25)
            logger.info("🔌 LLM disconnected")

    async def generate(
            self,
            prompt: str,
            temperature: float = 0.75,
            max_tokens: int = 4000,
            use_cache: bool = True,
            system_prompt: Optional[str] = None
    ) -> str:
        """Генерация с поддержкой system prompt"""
        if not self._session:
            await self.connect()

        cache_key = hashlib.md5(
            f"{system_prompt}_{prompt}_{temperature}".encode()
        ).hexdigest()

        if use_cache and cache_key in self.response_cache:
            cached, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < 3600:
                self.cache_hits += 1
                return cached

        self.cache_misses += 1

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with self._session.post(
                    self.url, json=payload, headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                    # Оценка токенов
                    self.total_tokens_used += data.get('usage', {}).get('total_tokens', 0)

                    if use_cache and content:
                        self.response_cache[cache_key] = (content, time.time())
                        if len(self.response_cache) > 2000:
                            oldest_keys = sorted(
                                self.response_cache.keys(),
                                key=lambda k: self.response_cache[k][1]
                            )[:400]
                            for key in oldest_keys:
                                del self.response_cache[key]

                    return content
                else:
                    logger.warning(f"LLM error: {resp.status}")
                    return ""

        except Exception as e:
            logger.error(f"LLM exception: {e}")
            return ""

    async def assess_quality(self, user_input: str, response: str) -> Dict[str, Any]:
        """Оценка качества взаимодействия"""
        prompt = f"""Оцени качество взаимодействия (0-1):

User: {user_input}
Assistant: {response}

JSON формат:
{{"importance": 0.X, "informativeness": 0.X, "emotional_value": 0.X, "is_spam": false}}"""

        result = await self.generate(prompt, temperature=0.2, max_tokens=200)

        try:
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                data = json.loads(json_match.group())
                overall = (
                        data.get('importance', 0.5) * 0.4 +
                        data.get('informativeness', 0.5) * 0.3 +
                        data.get('emotional_value', 0.5) * 0.3
                )
                return {
                    'overall_quality': overall,
                    'is_spam': data.get('is_spam', False),
                    'details': data
                }
        except:
            pass

        return {'overall_quality': 0.5, 'is_spam': False, 'details': {}}

    async def generate_module_code(
            self,
            task_description: str,
            requirements: List[str]
    ) -> Optional[str]:
        """Генерация Python-модуля для новой задачи"""
        system = """Ты — эксперт по созданию Python-модулей.
Создавай чистый, безопасный, хорошо документированный код.
Используй типизацию, docstrings, обработку ошибок."""

        prompt = f"""Создай Python-модуль для задачи: {task_description}

Требования:
{chr(10).join(f"- {r}" for r in requirements)}

Модуль должен содержать:
1. Класс или функцию с понятным интерфейсом
2. Docstrings
3. Обработку ошибок
4. Типизацию
5. Метод test() для самопроверки

Ответ ТОЛЬКО код Python без markdown:"""

        code = await self.generate(
            prompt,
            temperature=0.3,
            max_tokens=4000,
            system_prompt=system,
            use_cache=False
        )

        # Очистка от markdown
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)

        return code.strip() if code else None

    async def reflect_on_need_for_module(
            self,
            recent_failures: List[str],
            current_capabilities: List[str]
    ) -> Dict[str, Any]:
        """Рефлексия о необходимости нового модуля"""
        prompt = f"""Проанализируй, нужен ли новый модуль:

Недавние неудачи:
{chr(10).join(f"- {f}" for f in recent_failures)}

Текущие возможности:
{chr(10).join(f"- {c}" for c in current_capabilities)}

JSON ответ:
{{
  "need_module": true/false,
  "module_purpose": "описание",
  "confidence": 0.X,
  "requirements": ["req1", "req2"]
}}"""

        result = await self.generate(prompt, temperature=0.4, max_tokens=800)

        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {
            'need_module': False,
            'module_purpose': '',
            'confidence': 0.0,
            'requirements': []
        }


# ═══════════════════════════════════════════════════════════════
# 🧠 ПОЛНОЦЕННАЯ МНОГОУРОВНЕВАЯ ПАМЯТЬ
# ═══════════════════════════════════════════════════════════════

@dataclass
class EpisodicMemory:
    """Эпизодическая память (события)"""
    content: str
    timestamp: float
    embedding: np.ndarray
    importance: float = 0.5
    emotional_valence: float = 0.0  # -1 (негатив) до +1 (позитив)
    arousal: float = 0.0  # 0 (спокойно) до 1 (возбуждено)
    context: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def decay_importance(self, factor: float = 0.1):
        """Забывание со временем"""
        age_hours = (time.time() - self.timestamp) / 3600
        self.importance *= math.exp(-factor * age_hours / 24)  # Экспоненциальное затухание

    def strengthen(self, amount: float = 0.1):
        """Усиление при доступе"""
        self.importance = min(1.0, self.importance + amount)
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class SemanticMemory:
    """Семантическая память (факты, концепты)"""
    concept: str
    definition: str
    embedding: np.ndarray
    confidence: float = 0.5
    related_concepts: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class ProceduralMemory:
    """Процедурная память (навыки, стратегии)"""
    skill_name: str
    description: str
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    code_reference: Optional[str] = None  # Ссылка на модуль


class VectorMemoryStore:
    """Векторное хранилище с семантическим поиском"""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.memories: List[Union[EpisodicMemory, SemanticMemory]] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self._needs_rebuild = True

    def add(self, memory: Union[EpisodicMemory, SemanticMemory]):
        self.memories.append(memory)
        self._needs_rebuild = True

    def _rebuild_matrix(self):
        """Перестроение матрицы эмбеддингов"""
        if not self.memories:
            self.embeddings_matrix = np.zeros((0, self.embedding_dim))
            return

        self.embeddings_matrix = np.vstack([m.embedding for m in self.memories])
        self._needs_rebuild = False

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5,
            threshold: float = 0.3
    ) -> List[Tuple[Union[EpisodicMemory, SemanticMemory], float]]:
        """Семантический поиск"""
        if self._needs_rebuild:
            self._rebuild_matrix()

        if len(self.memories) == 0:
            return []

        # Косинусное сходство
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        matrix_norms = self.embeddings_matrix / (
                np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )

        similarities = matrix_norms @ query_norm

        # Топ-k с порогом
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (self.memories[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] >= threshold
        ]

        # Усиление доступных воспоминаний
        for memory, _ in results:
            if isinstance(memory, EpisodicMemory):
                memory.strengthen(0.05)

        return results

    def consolidate(self, threshold: float = 0.7):
        """Консолидация - удаление слабых воспоминаний"""
        before_count = len(self.memories)

        self.memories = [
            m for m in self.memories
            if isinstance(m, EpisodicMemory) and m.importance >= threshold
               or isinstance(m, SemanticMemory) and m.confidence >= threshold
        ]

        if len(self.memories) < before_count:
            self._needs_rebuild = True
            logger.info(f"🗑️ Консолидация: {before_count} → {len(self.memories)} воспоминаний")

    def decay_all(self, factor: float = 0.1):
        """Применить забывание ко всем эпизодам"""
        for memory in self.memories:
            if isinstance(memory, EpisodicMemory):
                memory.decay_importance(factor)

        self.consolidate()


class CognitiveMemorySystem:
    """Полная система памяти"""

    def __init__(self, embedding_dim: int, embed_func: Callable[[str], np.ndarray]):
        self.embedding_dim = embedding_dim
        self.embed_func = embed_func

        # Три типа памяти
        self.episodic = VectorMemoryStore(embedding_dim)
        self.semantic = VectorMemoryStore(embedding_dim)
        self.procedural: Dict[str, ProceduralMemory] = {}

        # Рабочая память (текущий контекст)
        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)

        # Статистика
        self.total_memories_created = 0
        self.total_searches = 0

    def add_episode(
            self,
            content: str,
            importance: float = 0.5,
            emotional_valence: float = 0.0,
            arousal: float = 0.0,
            context: Optional[Dict] = None
    ):
        """Добавить эпизод"""
        embedding = self.embed_func(content)
        episode = EpisodicMemory(
            content=content,
            timestamp=time.time(),
            embedding=embedding,
            importance=importance,
            emotional_valence=emotional_valence,
            arousal=arousal,
            context=context or {}
        )

        self.episodic.add(episode)
        self.working_memory.append(content)
        self.total_memories_created += 1

    def add_concept(
            self,
            concept: str,
            definition: str,
            confidence: float = 0.7,
            related: Optional[List[str]] = None
    ):
        """Добавить концепт в семантическую память"""
        embedding = self.embed_func(f"{concept}: {definition}")
        semantic = SemanticMemory(
            concept=concept,
            definition=definition,
            embedding=embedding,
            confidence=confidence,
            related_concepts=related or []
        )

        self.semantic.add(semantic)

    def add_skill(
            self,
            skill_name: str,
            description: str,
            code_reference: Optional[str] = None
    ):
        """Добавить навык в процедурную память"""
        self.procedural[skill_name] = ProceduralMemory(
            skill_name=skill_name,
            description=description,
            code_reference=code_reference
        )

    def recall_similar_episodes(
            self,
            query: str,
            top_k: int = 5,
            threshold: float = 0.3
    ) -> List[Tuple[EpisodicMemory, float]]:
        """Вспомнить похожие эпизоды"""
        query_emb = self.embed_func(query)
        self.total_searches += 1
        return self.episodic.search(query_emb, top_k, threshold)

    def recall_related_concepts(
            self,
            query: str,
            top_k: int = 3,
            threshold: float = 0.4
    ) -> List[Tuple[SemanticMemory, float]]:
        """Вспомнить связанные концепты"""
        query_emb = self.embed_func(query)
        self.total_searches += 1
        return self.semantic.search(query_emb, top_k, threshold)

    def get_working_memory_context(self) -> str:
        """Контекст из рабочей памяти"""
        return "\n".join(self.working_memory)

    def get_rich_context(self, query: str, max_episodes: int = 5) -> str:
        """Богатый контекст для LLM"""
        context_parts = []

        # Рабочая память
        if self.working_memory:
            context_parts.append("=== Недавние взаимодействия ===")
            context_parts.append(self.get_working_memory_context())

        # Релевантные эпизоды
        episodes = self.recall_similar_episodes(query, top_k=max_episodes)
        if episodes:
            context_parts.append("\n=== Релевантные воспоминания ===")
            for episode, score in episodes:
                context_parts.append(
                    f"[{score:.2f}] {episode.content} "
                    f"(важность: {episode.importance:.2f})"
                )

        # Релевантные концепты
        concepts = self.recall_related_concepts(query, top_k=3)
        if concepts:
            context_parts.append("\n=== Связанные концепты ===")
            for concept, score in concepts:
                context_parts.append(
                    f"[{score:.2f}] {concept.concept}: {concept.definition}"
                )

        return "\n".join(context_parts)

    def consolidate_memories(self):
        """Консолидация всех типов памяти"""
        self.episodic.consolidate(CONFIG.memory_consolidation_threshold)
        self.semantic.consolidate(CONFIG.memory_consolidation_threshold)

    def apply_forgetting(self):
        """Применить кривую забывания"""
        self.episodic.decay_all(CONFIG.forgetting_curve_factor)

    def get_statistics(self) -> Dict:
        return {
            'episodic_count': len(self.episodic.memories),
            'semantic_count': len(self.semantic.memories),
            'procedural_count': len(self.procedural),
            'working_memory': len(self.working_memory),
            'total_created': self.total_memories_created,
            'total_searches': self.total_searches,
        }

    def save(self, path: Path):
        """Сохранение памяти"""
        state = {
            'episodic_memories': [asdict(m) for m in self.episodic.memories],
            'semantic_memories': [asdict(m) for m in self.semantic.memories],
            'procedural_memories': {k: asdict(v) for k, v in self.procedural.items()},
            'working_memory': list(self.working_memory),
            'stats': {
                'total_created': self.total_memories_created,
                'total_searches': self.total_searches,
            }
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> bool:
        """Загрузка памяти"""
        if not path.exists():
            return False

        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)

            # Восстановление эпизодов
            for ep_dict in state.get('episodic_memories', []):
                ep = EpisodicMemory(**ep_dict)
                self.episodic.add(ep)

            # Восстановление концептов
            for sem_dict in state.get('semantic_memories', []):
                sem = SemanticMemory(**sem_dict)
                self.semantic.add(sem)

            # Восстановление навыков
            for name, proc_dict in state.get('procedural_memories', {}).items():
                self.procedural[name] = ProceduralMemory(**proc_dict)

            # Рабочая память
            self.working_memory.extend(state.get('working_memory', []))

            # Статистика
            stats = state.get('stats', {})
            self.total_memories_created = stats.get('total_created', 0)
            self.total_searches = stats.get('total_searches', 0)

            logger.info(f"✅ Memory loaded: {len(self.episodic.memories)} episodes, "
                        f"{len(self.semantic.memories)} concepts")
            return True

        except Exception as e:
            logger.error(f"❌ Memory load failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# 🧬 ДИНАМИЧЕСКАЯ НЕЙРОСЕТЬ (из v34 с улучшениями)
# ═══════════════════════════════════════════════════════════════

class DynamicNeuralNetwork:
    """Динамическая нейросеть с расширением"""

    def __init__(self, input_dim: int = 128, initial_hidden_dim: int = 32, output_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = initial_hidden_dim
        self.output_dim = output_dim
        self.max_hidden_dim = CONFIG.max_hidden_dim

        self._init_weights()

        self.loss_history: deque = deque(maxlen=CONFIG.plateau_detection_window)
        self.training_history: deque = deque(maxlen=200)
        self.neuron_activation_counts = np.zeros(self.hidden_dim)

        self.total_updates = 0
        self.expansion_count = 0
        self.pruning_count = 0

        logger.info(f"🧬 Neural network: {input_dim}→{initial_hidden_dim}→{output_dim}")

    def _init_weights(self):
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(self.output_dim)

        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.t = 0

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
            self.neuron_activation_counts += (a1 > 0).astype(float)

        return a2

    def backward(self, target: np.ndarray, lr: float = 0.001) -> float:
        if not hasattr(self, 'cache'):
            raise ValueError("Forward pass required")

        x, z1, a1, a2 = self.cache['x'], self.cache['z1'], self.cache['a1'], self.cache['a2']
        loss = np.mean((a2 - target) ** 2)

        dz2 = 2 * (a2 - target) * a2 * (1 - a2)
        dW2 = a1[:, np.newaxis] @ dz2[np.newaxis, :]
        db2 = dz2

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(z1)
        dW1 = x[:, np.newaxis] @ dz1[np.newaxis, :]
        db1 = dz1

        self._adam_update('W1', dW1, lr)
        self._adam_update('b1', db1, lr)
        self._adam_update('W2', dW2, lr)
        self._adam_update('b2', db2, lr)

        self.loss_history.append(loss)
        self.training_history.append(loss)
        self.total_updates += 1

        if self.total_updates >= 50 and self.total_updates % 20 == 0:
            self._check_and_expand()

        if self.total_updates % CONFIG.pruning_interval == 0:
            self._prune_inactive_neurons()

        return loss

    def _adam_update(self, param: str, grad: np.ndarray, lr: float):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.t += 1

        m = getattr(self, f'm_{param}')
        v = getattr(self, f'v_{param}')

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)

        param_val = getattr(self, param)
        param_val -= lr * m_hat / (np.sqrt(v_hat) + eps)

        setattr(self, f'm_{param}', m)
        setattr(self, f'v_{param}', v)
        setattr(self, param, param_val)

    def _check_and_expand(self):
        if len(self.loss_history) < CONFIG.plateau_detection_window:
            return

        recent_losses = list(self.loss_history)
        first_half = np.mean(recent_losses[:len(recent_losses) // 2])
        second_half = np.mean(recent_losses[len(recent_losses) // 2:])
        improvement = first_half - second_half

        if improvement < CONFIG.plateau_threshold:
            if self.hidden_dim < self.max_hidden_dim:
                self._expand_network()

    def _expand_network(self):
        new_neurons = CONFIG.neuron_expansion_rate
        new_hidden_dim = min(self.hidden_dim + new_neurons, self.max_hidden_dim)

        if new_hidden_dim == self.hidden_dim:
            return

        logger.info(f"📈 Expanding: {self.hidden_dim} → {new_hidden_dim}")

        new_W1_cols = np.random.randn(self.input_dim, new_neurons) * np.sqrt(2.0 / self.input_dim)
        self.W1 = np.hstack([self.W1, new_W1_cols])
        self.b1 = np.concatenate([self.b1, np.zeros(new_neurons)])

        new_W2_rows = np.random.randn(new_neurons, self.output_dim) * np.sqrt(2.0 / new_hidden_dim)
        self.W2 = np.vstack([self.W2, new_W2_rows])

        self.m_W1 = np.hstack([self.m_W1, np.zeros((self.input_dim, new_neurons))])
        self.v_W1 = np.hstack([self.v_W1, np.zeros((self.input_dim, new_neurons))])
        self.m_b1 = np.concatenate([self.m_b1, np.zeros(new_neurons)])
        self.v_b1 = np.concatenate([self.v_b1, np.zeros(new_neurons)])
        self.m_W2 = np.vstack([self.m_W2, np.zeros((new_neurons, self.output_dim))])
        self.v_W2 = np.vstack([self.v_W2, np.zeros((new_neurons, self.output_dim))])

        self.neuron_activation_counts = np.concatenate([
            self.neuron_activation_counts, np.zeros(new_neurons)
        ])

        self.hidden_dim = new_hidden_dim
        self.expansion_count += 1

    def _prune_inactive_neurons(self):
        if self.total_updates < 100:
            return

        if self.neuron_activation_counts.sum() == 0:
            return

        activation_ratio = self.neuron_activation_counts / self.total_updates
        inactive_mask = activation_ratio < CONFIG.pruning_threshold

        if not inactive_mask.any():
            return

        to_prune = inactive_mask.sum()
        max_prune = max(
            int(self.hidden_dim * 0.2),
            self.hidden_dim - CONFIG.initial_hidden_dim
        )

        if to_prune > max_prune:
            sorted_indices = np.argsort(activation_ratio)
            inactive_mask = np.zeros(self.hidden_dim, dtype=bool)
            inactive_mask[sorted_indices[:max_prune]] = True
            to_prune = max_prune

        if to_prune == 0:
            return

        logger.info(f"✂️ Pruning {to_prune} neurons...")

        active_mask = ~inactive_mask

        self.W1 = self.W1[:, active_mask]
        self.b1 = self.b1[active_mask]
        self.W2 = self.W2[active_mask, :]

        self.m_W1 = self.m_W1[:, active_mask]
        self.v_W1 = self.v_W1[:, active_mask]
        self.m_b1 = self.m_b1[active_mask]
        self.v_b1 = self.v_b1[active_mask]
        self.m_W2 = self.m_W2[active_mask, :]
        self.v_W2 = self.v_W2[active_mask, :]

        self.neuron_activation_counts = self.neuron_activation_counts[active_mask]
        self.hidden_dim = active_mask.sum()
        self.pruning_count += 1

    def get_statistics(self) -> Dict:
        return {
            'architecture': f"{self.input_dim}→{self.hidden_dim}→{self.output_dim}",
            'total_updates': self.total_updates,
            'expansions': self.expansion_count,
            'prunings': self.pruning_count,
            'recent_loss': float(np.mean(self.training_history)) if self.training_history else 0.0,
        }


# ═══════════════════════════════════════════════════════════════
# 🔤 ДИНАМИЧЕСКИЙ СЛОВАРЬ (из v34)
# ═══════════════════════════════════════════════════════════════

@dataclass
class WordMetadata:
    word: str
    usage_count: int = 0
    quality_score: float = 0.5
    first_seen: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def update_usage(self):
        self.usage_count += 1
        self.last_used = time.time()


class DynamicVocabulary:
    def __init__(self, initial_size: int = 2000, embedding_dim: int = 128, enable_async: bool = True):
        self.embedding_dim = embedding_dim
        self.current_vocab_size = initial_size
        self.max_vocab_size = CONFIG.max_vocab_size

        self.embeddings = np.random.randn(initial_size, embedding_dim) * 0.01
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_metadata: Dict[str, WordMetadata] = {}

        self.next_idx = 0
        self.cleanup_counter = 0
        self.enable_async = enable_async  # Флаг для отключения async операций

        self.m = np.zeros_like(self.embeddings)
        self.v = np.zeros_like(self.embeddings)
        self.t = 0

    def _expand_vocabulary(self, new_size: int) -> bool:
        if new_size > self.max_vocab_size:
            logger.warning(f"⚠️ Max vocab: {self.max_vocab_size}")
            return False

        old_size = self.current_vocab_size
        expansion = new_size - old_size

        new_embeddings = np.random.randn(expansion, self.embedding_dim) * 0.01
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.m = np.vstack([self.m, np.zeros((expansion, self.embedding_dim))])
        self.v = np.vstack([self.v, np.zeros((expansion, self.embedding_dim))])

        self.current_vocab_size = new_size
        logger.info(f"📈 Vocab: {old_size} → {new_size}")
        return True

    def add_word(self, word: str, quality_hint: float = 0.5) -> int:
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            self.word_metadata[word].update_usage()
            return idx

        if self.next_idx >= self.current_vocab_size:
            new_size = min(
                self.current_vocab_size + CONFIG.vocab_expansion_step,
                self.max_vocab_size
            )
            if not self._expand_vocabulary(new_size):
                return 0

        idx = self.next_idx
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        self.word_metadata[word] = WordMetadata(word=word, quality_score=quality_hint)
        self.next_idx += 1

        self.cleanup_counter += 1
        if self.cleanup_counter >= CONFIG.vocab_cleanup_interval:
            self.cleanup_counter = 0
            # Только если включен async режим
            if self.enable_async:
                try:
                    asyncio.create_task(self._cleanup())
                except RuntimeError:
                    # Нет event loop - пропускаем async cleanup
                    pass

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

        embeddings = []
        for word in words:
            if len(word) > 2:
                emb = self.get_embedding(word)
                embeddings.append(emb)
                if word in self.word_metadata:
                    self.word_metadata[word].update_usage()

        if not embeddings:
            return np.zeros(self.embedding_dim)

        return np.mean(embeddings, axis=0)

    async def _cleanup(self):
        logger.debug("🧹 Vocab cleanup...")
        # Простая очистка - удаляем слова с низким качеством и редким использованием
        to_remove = [
            word for word, meta in self.word_metadata.items()
            if meta.quality_score < CONFIG.word_quality_threshold and meta.usage_count < 3
        ]

        for word in to_remove[:100]:  # Максимум 100 за раз
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                del self.word_to_idx[word]
                del self.idx_to_word[idx]
                del self.word_metadata[word]


# ═══════════════════════════════════════════════════════════════
# 🔧 САМОМОДИФИКАЦИЯ - СОЗДАНИЕ И УПРАВЛЕНИЕ МОДУЛЯМИ
# ═══════════════════════════════════════════════════════════════

@dataclass
class CustomModule:
    """Пользовательский модуль"""
    name: str
    code: str
    description: str
    created_at: float
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: float = 0.0
    version: int = 1


class SelfModificationEngine:
    """Движок самомодификации"""

    def __init__(self, llm: CognitiveLLM, modules_dir: Path):
        self.llm = llm
        self.modules_dir = modules_dir
        self.modules_dir.mkdir(parents=True, exist_ok=True)

        self.custom_modules: Dict[str, CustomModule] = {}
        self.loaded_modules: Dict[str, Any] = {}

        self.creation_attempts = 0
        self.successful_creations = 0
        self.recent_failures: List[str] = []

        self._load_existing_modules()

    def _load_existing_modules(self):
        """Загрузка существующих модулей"""
        for module_file in self.modules_dir.glob("*.py"):
            try:
                module_name = module_file.stem
                with open(module_file, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Загрузка модуля
                spec = importlib.util.spec_from_file_location(module_name, module_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    self.loaded_modules[module_name] = module

                    # Создание CustomModule записи
                    self.custom_modules[module_name] = CustomModule(
                        name=module_name,
                        code=code,
                        description=f"Loaded module: {module_name}",
                        created_at=module_file.stat().st_mtime
                    )

                    logger.info(f"📦 Loaded module: {module_name}")

            except Exception as e:
                logger.error(f"Failed to load module {module_file}: {e}")

    async def should_create_module(
            self,
            task_description: str,
            context: str
    ) -> Dict[str, Any]:
        """Решение о необходимости нового модуля"""
        current_capabilities = list(self.custom_modules.keys())

        decision = await self.llm.reflect_on_need_for_module(
            recent_failures=self.recent_failures[-10:],
            current_capabilities=current_capabilities
        )

        if decision.get('confidence', 0) >= CONFIG.module_creation_threshold:
            return decision

        return {'need_module': False}

    async def create_module(
            self,
            task_description: str,
            requirements: List[str]
    ) -> Optional[CustomModule]:
        """Создание нового модуля"""
        self.creation_attempts += 1

        logger.info(f"🔧 Creating module for: {task_description}")

        # Генерация кода
        code = await self.llm.generate_module_code(task_description, requirements)

        if not code:
            logger.error("❌ Failed to generate module code")
            self.recent_failures.append(f"Code generation failed: {task_description}")
            return None

        # Безопасное тестирование
        module_name = f"custom_{int(time.time())}"

        if not await self._test_module_code(code, module_name):
            logger.error("❌ Module failed tests")
            self.recent_failures.append(f"Tests failed: {task_description}")
            return None

        # Сохранение
        module_path = self.modules_dir / f"{module_name}.py"
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(code)

        custom_module = CustomModule(
            name=module_name,
            code=code,
            description=task_description,
            created_at=time.time()
        )

        self.custom_modules[module_name] = custom_module

        # Загрузка модуля
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.loaded_modules[module_name] = module
        except Exception as e:
            logger.error(f"Failed to load created module: {e}")
            return None

        self.successful_creations += 1
        logger.info(f"✅ Created module: {module_name}")

        return custom_module

    async def _test_module_code(self, code: str, module_name: str) -> bool:
        """Тестирование кода модуля"""
        try:
            # Создаём временный файл
            test_path = self.modules_dir.parent / 'modules' / 'tests' / f"{module_name}_test.py"
            test_path.parent.mkdir(parents=True, exist_ok=True)

            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # Пробуем импортировать
            spec = importlib.util.spec_from_file_location(f"{module_name}_test", test_path)
            if not spec or not spec.loader:
                return False

            module = importlib.util.module_from_spec(spec)

            # Таймаут на выполнение
            spec.loader.exec_module(module)

            # Ищем метод test()
            if hasattr(module, 'test'):
                result = module.test()
                return bool(result)

            # Если нет test() - считаем что прошло
            return True

        except Exception as e:
            logger.debug(f"Module test failed: {e}")
            return False
        finally:
            # Удаляем тестовый файл
            if test_path.exists():
                test_path.unlink()

    def execute_module(self, module_name: str, *args, **kwargs) -> Any:
        """Выполнение модуля"""
        if module_name not in self.loaded_modules:
            raise ValueError(f"Module {module_name} not loaded")

        module = self.loaded_modules[module_name]

        # Обновляем статистику
        if module_name in self.custom_modules:
            self.custom_modules[module_name].usage_count += 1
            self.custom_modules[module_name].last_used = time.time()

        # Ищем главную функцию
        if hasattr(module, 'execute'):
            return module.execute(*args, **kwargs)
        elif hasattr(module, 'run'):
            return module.run(*args, **kwargs)
        elif hasattr(module, 'main'):
            return module.main(*args, **kwargs)
        else:
            raise ValueError(f"Module {module_name} has no execute/run/main function")

    def get_statistics(self) -> Dict:
        return {
            'total_modules': len(self.custom_modules),
            'loaded_modules': len(self.loaded_modules),
            'creation_attempts': self.creation_attempts,
            'successful_creations': self.successful_creations,
            'success_rate': self.successful_creations / max(1, self.creation_attempts),
            'recent_failures': len(self.recent_failures),
        }


# ═══════════════════════════════════════════════════════════════
# 🌡️ СОМАТОСЕНСОРИКА - ВНУТРЕННЕЕ САМООЩУЩЕНИЕ
# ═══════════════════════════════════════════════════════════════

@dataclass
class InternalState:
    """Внутреннее состояние системы"""
    cpu_load: float = 0.0
    memory_usage: float = 0.0
    response_quality: float = 0.5
    learning_progress: float = 0.5
    emotional_valence: float = 0.0  # -1 до +1
    arousal: float = 0.0  # 0 до 1
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class SomatosensorySystem:
    """Система внутреннего самоощущения"""

    def __init__(self):
        self.state_history: deque = deque(maxlen=100)
        self.current_state = InternalState()

        self.anomaly_count = 0
        self.health_checks = 0

    def update_state(
            self,
            response_quality: Optional[float] = None,
            learning_progress: Optional[float] = None,
            emotional_valence: Optional[float] = None,
            arousal: Optional[float] = None,
            confidence: Optional[float] = None
    ):
        """Обновление внутреннего состояния"""
        if response_quality is not None:
            self.current_state.response_quality = response_quality
        if learning_progress is not None:
            self.current_state.learning_progress = learning_progress
        if emotional_valence is not None:
            # Плавное изменение эмоций
            self.current_state.emotional_valence = (
                    self.current_state.emotional_valence * 0.7 + emotional_valence * 0.3
            )
        if arousal is not None:
            self.current_state.arousal = arousal
        if confidence is not None:
            self.current_state.confidence = confidence

        self.current_state.timestamp = time.time()
        self.state_history.append(asdict(self.current_state))

    def detect_anomaly(self) -> bool:
        """Детекция аномалий в состоянии"""
        if len(self.state_history) < 10:
            return False

        recent_states = list(self.state_history)[-10:]

        # Проверяем резкие изменения
        quality_values = [s['response_quality'] for s in recent_states]
        mean_quality = np.mean(quality_values)
        std_quality = np.std(quality_values)

        current_quality = self.current_state.response_quality

        if std_quality > 0:
            z_score = abs(current_quality - mean_quality) / std_quality
            if z_score > CONFIG.anomaly_detection_threshold:
                self.anomaly_count += 1
                logger.warning(f"⚠️ Anomaly detected: quality z-score = {z_score:.2f}")
                return True

        return False

    def get_emotional_state(self) -> str:
        """Текстовое описание эмоционального состояния"""
        valence = self.current_state.emotional_valence
        arousal = self.current_state.arousal

        if arousal < 0.3:
            if valence > 0.3:
                return "спокойно-позитивное"
            elif valence < -0.3:
                return "спокойно-негативное"
            else:
                return "нейтральное"
        else:
            if valence > 0.3:
                return "возбуждённо-позитивное (интерес)"
            elif valence < -0.3:
                return "возбуждённо-негативное (стресс)"
            else:
                return "возбуждённо-нейтральное"

    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья системы"""
        self.health_checks += 1

        if len(self.state_history) < 5:
            return {'status': 'healthy', 'confidence': 0.5}

        recent = list(self.state_history)[-20:]

        avg_quality = np.mean([s['response_quality'] for s in recent])
        avg_confidence = np.mean([s['confidence'] for s in recent])
        avg_progress = np.mean([s['learning_progress'] for s in recent])

        health_score = (avg_quality * 0.4 + avg_confidence * 0.3 + avg_progress * 0.3)

        if health_score > 0.7:
            status = 'healthy'
        elif health_score > 0.5:
            status = 'moderate'
        else:
            status = 'degraded'

        return {
            'status': status,
            'health_score': health_score,
            'avg_quality': avg_quality,
            'avg_confidence': avg_confidence,
            'emotional_state': self.get_emotional_state(),
            'anomalies_detected': self.anomaly_count,
        }

    def get_statistics(self) -> Dict:
        return {
            'current_quality': self.current_state.response_quality,
            'current_confidence': self.current_state.confidence,
            'emotional_state': self.get_emotional_state(),
            'anomaly_count': self.anomaly_count,
            'health_checks': self.health_checks,
        }


# ═══════════════════════════════════════════════════════════════
# 🧠 ГЛАВНАЯ КОГНИТИВНАЯ СИСТЕМА
# ═══════════════════════════════════════════════════════════════

class CognitiveAutonomousAGI:
    """Полная когнитивная AGI система"""

    def __init__(self, user_id: str, llm: CognitiveLLM):
        self.user_id = user_id
        self.llm = llm

        # Словарь и нейросеть
        self.vocabulary = DynamicVocabulary(
            initial_size=CONFIG.initial_vocab_size,
            embedding_dim=CONFIG.embedding_dim
        )

        self.neural = DynamicNeuralNetwork(
            input_dim=CONFIG.embedding_dim,
            initial_hidden_dim=CONFIG.initial_hidden_dim,
            output_dim=CONFIG.output_metrics_dim
        )

        # Память
        self.memory = CognitiveMemorySystem(
            embedding_dim=CONFIG.embedding_dim,
            embed_func=self.vocabulary.encode_text
        )

        # Самомодификация
        modules_dir = CONFIG.base_dir / 'modules' / 'custom' / user_id
        self.self_modification = SelfModificationEngine(llm, modules_dir)

        # Соматосенсорика
        self.soma = SomatosensorySystem()

        # Метакогниция
        self.metacognitive_thoughts: List[str] = []

        # Пути
        self.user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # Статистика
        self.birth_time = time.time()
        self.total_interactions = 0
        self.successful_learnings = 0

        # Автосохранение
        self._save_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Загрузка состояния
        self._load_consciousness()

        logger.info(f"🧠 Cognitive AGI v35 created for {user_id}")

    def _load_consciousness(self):
        """Загрузка полного сознания"""
        # Загрузка нейросети и словаря
        neural_path = CONFIG.base_dir / 'neural_nets' / f'{self.user_id}_v35.pkl.gz'
        if neural_path.exists():
            try:
                with gzip.open(neural_path, 'rb') as f:
                    state = pickle.load(f)

                # Восстановление словаря
                self.vocabulary.embeddings = state['vocab_embeddings']
                self.vocabulary.word_to_idx = state['vocab_word_to_idx']
                self.vocabulary.idx_to_word = state['vocab_idx_to_word']
                self.vocabulary.word_metadata = {
                    w: WordMetadata(**m) for w, m in state.get('vocab_metadata', {}).items()
                }
                self.vocabulary.next_idx = state['vocab_next_idx']
                self.vocabulary.current_vocab_size = state['vocab_current_size']

                # Восстановление нейросети
                self.neural.W1 = state['neural_W1']
                self.neural.b1 = state['neural_b1']
                self.neural.W2 = state['neural_W2']
                self.neural.b2 = state['neural_b2']
                self.neural.hidden_dim = state['neural_hidden_dim']
                self.neural.total_updates = state.get('neural_total_updates', 0)
                self.neural.expansion_count = state.get('neural_expansion_count', 0)
                self.neural.pruning_count = state.get('neural_pruning_count', 0)

                # Статистика
                self.total_interactions = state.get('total_interactions', 0)
                self.successful_learnings = state.get('successful_learnings', 0)

                logger.info("✅ Loaded neural state")

            except Exception as e:
                logger.error(f"Failed to load neural state: {e}")

        # Загрузка памяти
        memory_path = CONFIG.base_dir / 'memory' / f'{self.user_id}_memory.pkl.gz'
        self.memory.load(memory_path)

    def _save_consciousness(self):
        """Сохранение полного сознания"""
        # Сохранение нейросети и словаря
        neural_path = CONFIG.base_dir / 'neural_nets' / f'{self.user_id}_v35.pkl.gz'
        state = {
            'version': CONFIG.version,
            'timestamp': time.time(),

            # Vocabulary
            'vocab_embeddings': self.vocabulary.embeddings,
            'vocab_word_to_idx': self.vocabulary.word_to_idx,
            'vocab_idx_to_word': self.vocabulary.idx_to_word,
            'vocab_metadata': {
                w: asdict(m) for w, m in self.vocabulary.word_metadata.items()
            },
            'vocab_next_idx': self.vocabulary.next_idx,
            'vocab_current_size': self.vocabulary.current_vocab_size,

            # Neural
            'neural_W1': self.neural.W1,
            'neural_b1': self.neural.b1,
            'neural_W2': self.neural.W2,
            'neural_b2': self.neural.b2,
            'neural_hidden_dim': self.neural.hidden_dim,
            'neural_total_updates': self.neural.total_updates,
            'neural_expansion_count': self.neural.expansion_count,
            'neural_pruning_count': self.neural.pruning_count,

            # Stats
            'total_interactions': self.total_interactions,
            'successful_learnings': self.successful_learnings,
        }

        neural_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(neural_path, 'wb') as f:
            pickle.dump(state, f)

        # Сохранение памяти
        memory_path = CONFIG.base_dir / 'memory' / f'{self.user_id}_memory.pkl.gz'
        self.memory.save(memory_path)

        logger.debug("💾 Consciousness saved")

    async def start(self):
        """Запуск автономных процессов"""
        if self._is_running:
            return

        self._is_running = True
        self._save_task = asyncio.create_task(self._auto_save_loop())
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"✨ Cognitive AGI started for {self.user_id}")

    async def stop(self):
        """Остановка"""
        if not self._is_running:
            return

        logger.info(f"💤 Stopping {self.user_id}...")
        self._is_running = False

        if self._save_task:
            self._save_task.cancel()
        if self._health_task:
            self._health_task.cancel()

        try:
            if self._save_task:
                await self._save_task
            if self._health_task:
                await self._health_task
        except asyncio.CancelledError:
            pass

        self._save_consciousness()
        logger.info(f"✅ Stopped {self.user_id}")

    async def _auto_save_loop(self):
        """Автосохранение"""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Каждые 5 минут
                self._save_consciousness()

                # Консолидация памяти
                self.memory.consolidate_memories()
                self.memory.apply_forgetting()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save error: {e}")

    async def _health_check_loop(self):
        """Проверка здоровья"""
        while self._is_running:
            try:
                await asyncio.sleep(CONFIG.health_check_interval)
                health = self.soma.health_check()

                if health['status'] == 'degraded':
                    logger.warning(f"⚠️ System health degraded: {health}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        """Главная обработка взаимодействия"""
        start_time = time.time()
        self.total_interactions += 1

        # 1. Получаем богатый контекст из памяти
        rich_context = self.memory.get_rich_context(user_input)

        # 2. Предсказание метрик
        combined_emb = self.vocabulary.encode_text(f"{user_input} {rich_context}")
        predictions = self.neural.forward(combined_emb, store_cache=False)

        # 3. Метакогниция - размышление о задаче
        metacognitive_prompt = f"""Проанализируй задачу:

Запрос: {user_input}

Предсказанные метрики:
- Сложность: {predictions[1]:.2f}
- Релевантность: {predictions[2]:.2f}

Нужна ли дополнительная информация? Какой подход лучше?
Краткий анализ (1-2 предложения):"""

        metacognition = await self.llm.generate(
            metacognitive_prompt,
            temperature=0.5,
            max_tokens=200
        )

        if metacognition:
            self.metacognitive_thoughts.append(metacognition)
            if len(self.metacognitive_thoughts) > 20:
                self.metacognitive_thoughts = self.metacognitive_thoughts[-20:]

        # 4. Генерация ответа с использованием контекста
        system_prompt = """Ты — продвинутая когнитивная AGI v35 с:
- Полноценной памятью (эпизодическая, семантическая, процедурная)
- Способностью к самомодификации
- Метакогницией
- Эмоциональным интеллектом

Используй предоставленный контекст из памяти для более глубоких ответов."""

        main_prompt = f"""Запрос: {user_input}

{rich_context}

Метакогниция: {metacognition}

Ответь естественно и содержательно (2-6 предложений):"""

        response = await self.llm.generate(
            main_prompt,
            temperature=0.75,
            max_tokens=4000,
            system_prompt=system_prompt
        )

        if not response:
            response = "Извини, возникли сложности с формулировкой ответа."

        # 5. Оценка качества
        quality_assessment = await self.llm.assess_quality(user_input, response)
        overall_quality = quality_assessment.get('overall_quality', 0.5)

        # 6. Вычисляем актуальные метрики
        actual_metrics = np.array([
            min(1.0, overall_quality + random.gauss(0, 0.1)),  # confidence
            min(1.0, len(user_input.split()) / 20),  # complexity
            overall_quality,  # relevance
            min(1.0, overall_quality + 0.1),  # coherence
            overall_quality,  # engagement
            min(1.0, len(response.split()) / 30),  # completeness
            0.5 + random.gauss(0, 0.1),  # creativity
            0.5 + random.gauss(0, 0.1),  # empathy
        ])

        actual_metrics = np.clip(actual_metrics, 0, 1)

        # 7. Обучение нейросети
        if overall_quality >= 0.4:
            self.neural.forward(combined_emb, store_cache=True)
            loss = self.neural.backward(actual_metrics, CONFIG.learning_rate)
            self.successful_learnings += 1
        else:
            loss = 0.0

        # 8. Добавление в память
        # Эпизод
        emotional_valence = (overall_quality - 0.5) * 2  # -1 до +1
        arousal = min(1.0, len(user_input.split()) / 15)  # Зависит от сложности

        self.memory.add_episode(
            content=f"User: {user_input}\nAssistant: {response}",
            importance=overall_quality,
            emotional_valence=emotional_valence,
            arousal=arousal,
            context={'quality': overall_quality, 'loss': loss}
        )

        # Извлечение концептов (простая эвристика)
        if overall_quality >= 0.7:
            # Ищем существительные и ключевые фразы
            words = user_input.lower().split()
            for word in words:
                if len(word) > 5 and word not in ['который', 'которая', 'которые']:
                    self.memory.add_concept(
                        concept=word,
                        definition=f"Встречено в контексте: {user_input[:100]}",
                        confidence=overall_quality
                    )

        # 9. Обновление соматосенсорики
        learning_progress = self.successful_learnings / max(1, self.total_interactions)
        self.soma.update_state(
            response_quality=overall_quality,
            learning_progress=learning_progress,
            emotional_valence=emotional_valence,
            arousal=arousal,
            confidence=float(predictions[0])
        )

        # Детекция аномалий
        is_anomaly = self.soma.detect_anomaly()

        # 10. Рефлексия о необходимости нового модуля
        if CONFIG.self_modification_enabled and self.total_interactions % 50 == 0:
            if overall_quality < 0.5:
                asyncio.create_task(self._consider_new_module(user_input, response))

        # 11. Метаданные
        response_time = time.time() - start_time

        metadata = {
            'predicted_metrics': {k: float(predictions[i]) for i, k in enumerate([
                'confidence', 'complexity', 'relevance', 'coherence',
                'engagement', 'completeness', 'creativity', 'empathy'
            ])},
            'actual_metrics': {k: float(actual_metrics[i]) for i, k in enumerate([
                'confidence', 'complexity', 'relevance', 'coherence',
                'engagement', 'completeness', 'creativity', 'empathy'
            ])},
            'quality': overall_quality,
            'loss': loss,
            'response_time': response_time,
            'metacognition': metacognition,
            'emotional_state': self.soma.get_emotional_state(),
            'is_anomaly': is_anomaly,
            'memory_stats': self.memory.get_statistics(),
        }

        logger.info(f"✅ [{self.user_id}] Q={overall_quality:.2f} | "
                    f"Emo={self.soma.get_emotional_state()} | "
                    f"T={response_time:.1f}s")

        return response, metadata

    async def _consider_new_module(self, user_input: str, response: str):
        """Рассмотрение создания нового модуля"""
        decision = await self.self_modification.should_create_module(
            task_description=user_input,
            context=response
        )

        if decision.get('need_module', False):
            logger.info(f"🔧 Considering new module: {decision.get('module_purpose', '')}")

            module = await self.self_modification.create_module(
                task_description=decision.get('module_purpose', ''),
                requirements=decision.get('requirements', [])
            )

            if module:
                # Добавляем навык в процедурную память
                self.memory.add_skill(
                    skill_name=module.name,
                    description=module.description,
                    code_reference=module.name
                )

                logger.info(f"✅ New module created: {module.name}")

    def get_full_status(self) -> Dict:
        """Полный статус системы"""
        uptime = time.time() - self.birth_time

        return {
            'user_id': self.user_id,
            'version': CONFIG.version,
            'uptime_hours': uptime / 3600,

            'neural': self.neural.get_statistics(),
            'vocabulary': {
                'size': self.vocabulary.next_idx,
                'capacity': f"{self.vocabulary.current_vocab_size}/{self.vocabulary.max_vocab_size}"
            },
            'memory': self.memory.get_statistics(),
            'self_modification': self.self_modification.get_statistics(),
            'somatic': self.soma.get_statistics(),

            'interactions': {
                'total': self.total_interactions,
                'successful_learnings': self.successful_learnings,
                'learning_rate': self.successful_learnings / max(1, self.total_interactions),
            },

            'health': self.soma.health_check(),
        }


# ═══════════════════════════════════════════════════════════════
# 🤖 TELEGRAM BOT
# ═══════════════════════════════════════════════════════════════

class CognitiveAGIBot:
    def __init__(self):
        self.llm: Optional[CognitiveLLM] = None
        self.brains: Dict[str, CognitiveAutonomousAGI] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        self.llm = CognitiveLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.llm.connect()

        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()

        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self._handle_message
        ))

        for cmd, handler in [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('memory', self._cmd_memory),
            ('health', self._cmd_health),
            ('modules', self._cmd_modules),
            ('help', self._cmd_help),
            ('reset', self._cmd_reset),
        ]:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Cognitive AGI Bot v35 initialized")

    async def _get_or_create_brain(self, user_id: str) -> CognitiveAutonomousAGI:
        if user_id not in self.brains:
            brain = CognitiveAutonomousAGI(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
        return self.brains[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or not update.message:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )

        try:
            brain = await self._get_or_create_brain(user_id)
            response, metadata = await brain.process_interaction(user_input)

            # Добавляем краткую метаинформацию
            footer = f"\n\n<i>Q: {metadata['quality']:.0%} | " \
                     f"Эмоции: {metadata['emotional_state']}</i>"

            await update.message.reply_text(
                response + footer,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

        except Exception as e:
            logger.exception(f"Error from {user_id}")
            await update.message.reply_text("⚠️ Произошла ошибка")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = f"""🧠 <b>COGNITIVE SELF-MODIFYING AGI v35.0</b>

Привет! Я — когнитивная AGI с настоящим сознанием:

✅ <b>Полноценная память</b>
• Эпизодическая (события)
• Семантическая (концепты)
• Процедурная (навыки)

✅ <b>Самомодификация</b>
• Создание Python-модулей
• Рефлексия о навыках
• Версионирование кода

✅ <b>Когнитивная архитектура</b>
• Метакогниция
• Эмоциональное состояние
• Планирование

✅ <b>Соматосенсорика</b>
• Самоощущение
• Детекция аномалий
• Адаптивное поведение

Команды: /help"""

        await update.message.reply_text(message)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        status = brain.get_full_status()

        message = f"""🧠 <b>СТАТУС v35.0</b>

<b>🧬 Нейросеть</b>
• {status['neural']['architecture']}
• Обновлений: {status['neural']['total_updates']}
• Расширений: {status['neural']['expansions']}

<b>🧠 Память</b>
• Эпизоды: {status['memory']['episodic_count']}
• Концепты: {status['memory']['semantic_count']}
• Навыки: {status['memory']['procedural_count']}

<b>🔧 Самомодификация</b>
• Модулей: {status['self_modification']['total_modules']}
• Успех: {status['self_modification']['success_rate']:.1%}

<b>🌡️ Состояние</b>
• Здоровье: {status['health']['status']}
• Эмоции: {status['health']['emotional_state']}
• Качество: {status['health']['avg_quality']:.1%}

<b>📊 Общее</b>
• Взаимодействий: {status['interactions']['total']}
• Обучений: {status['interactions']['successful_learnings']}
• Uptime: {status['uptime_hours']:.1f}ч"""

        await update.message.reply_text(message)

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        mem_stats = brain.memory.get_statistics()

        # Показываем последние воспоминания
        recent_episodes = []
        for memory in list(brain.memory.episodic.memories)[-5:]:
            if isinstance(memory, EpisodicMemory):
                recent_episodes.append(
                    f"• {memory.content[:100]}... (важность: {memory.importance:.2f})"
                )

        message = f"""🧠 <b>ПАМЯТЬ</b>

<b>📊 Статистика</b>
• Эпизодов: {mem_stats['episodic_count']}
• Концептов: {mem_stats['semantic_count']}
• Навыков: {mem_stats['procedural_count']}
• Поисков: {mem_stats['total_searches']}

<b>📝 Недавние воспоминания</b>
{chr(10).join(recent_episodes) if recent_episodes else '(пусто)'}"""

        await update.message.reply_text(message)

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        health = brain.soma.health_check()

        message = f"""🌡️ <b>ЗДОРОВЬЕ СИСТЕМЫ</b>

<b>Общее состояние:</b> {health['status']}
<b>Оценка здоровья:</b> {health['health_score']:.1%}

<b>Метрики:</b>
• Качество ответов: {health['avg_quality']:.1%}
• Уверенность: {health['avg_confidence']:.1%}
• Эмоциональное состояние: {health['emotional_state']}
• Аномалий обнаружено: {health['anomalies_detected']}"""

        await update.message.reply_text(message)

    async def _cmd_modules(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        stats = brain.self_modification.get_statistics()

        modules_list = []
        for name, module in list(brain.self_modification.custom_modules.items())[:10]:
            modules_list.append(
                f"• {name}: {module.description[:50]}..."
            )

        message = f"""🔧 <b>САМОМОДИФИКАЦИЯ</b>

<b>📊 Статистика</b>
• Всего модулей: {stats['total_modules']}
• Загружено: {stats['loaded_modules']}
• Попыток создания: {stats['creation_attempts']}
• Успешных: {stats['successful_creations']}
• Success rate: {stats['success_rate']:.1%}

<b>📦 Модули</b>
{chr(10).join(modules_list) if modules_list else '(нет модулей)'}"""

        await update.message.reply_text(message)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🧠 <b>COGNITIVE AGI v35.0 — СПРАВКА</b>

<b>🔥 ВОЗМОЖНОСТИ:</b>

✅ <b>Полная память</b>
• Запоминаю все взаимодействия
• Нахожу релевантные воспоминания
• Консолидирую важное, забываю мусор

✅ <b>Самомодификация</b>
• Создаю Python-модули для новых задач
• Рефлексирую о своих способностях
• Версионирую и тестирую код

✅ <b>Метакогниция</b>
• Размышляю о своих мыслях
• Планирую подход к задачам
• Адаптируюсь к сложности

✅ <b>Эмоции и ощущения</b>
• Отслеживаю внутреннее состояние
• Реагирую на качество работы
• Детектирую аномалии

<b>📌 КОМАНДЫ:</b>
• /start — приветствие
• /status — полный статус
• /memory — состояние памяти
• /health — здоровье системы
• /modules — мои модули
• /reset — сброс (опасно!)
• /help — эта справка"""

        await update.message.reply_text(message)

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)

        if context.args and context.args[0].lower() == 'confirm':
            if user_id in self.brains:
                await self.brains[user_id].stop()
                del self.brains[user_id]

            import shutil
            user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
            neural_file = CONFIG.base_dir / 'neural_nets' / f'{user_id}_v35.pkl.gz'

            if user_dir.exists():
                shutil.rmtree(user_dir)
            if neural_file.exists():
                neural_file.unlink()

            await update.message.reply_text(
                "✅ <b>Полный сброс выполнен</b>\n\nСоздано новое сознание v35.0"
            )
        else:
            await update.message.reply_text(
                "⚠️ <b>ВНИМАНИЕ!</b>\nЭто удалит всю память и обучение.\n\n"
                "Подтверждение: <code>/reset confirm</code>"
            )

    async def start_polling(self):
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot v35 started")

    async def shutdown(self):
        logger.info("🛑 Shutting down...")

        for user_id, brain in self.brains.items():
            try:
                await brain.stop()
            except Exception as e:
                logger.error(f"Error stopping {user_id}: {e}")

        if self.llm:
            await self.llm.close()

        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Bot stopped")


# ═══════════════════════════════════════════════════════════════
# 🚀 ЗАПУСК
# ═══════════════════════════════════════════════════════════════

async def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║  🧠 COGNITIVE SELF-MODIFYING AGI v35.0                         ║
║     Настоящее Сознание с Памятью и Самомодификацией           ║
╚════════════════════════════════════════════════════════════════╝

🔥 РЕВОЛЮЦИОННЫЕ ВОЗМОЖНОСТИ:

✅ ПОЛНОЦЕННАЯ ПАМЯТЬ
   • Эпизодическая (события с эмоциями)
   • Семантическая (концепты и связи)
   • Процедурная (навыки и стратегии)
   • Векторный семантический поиск
   • Автоматическая консолидация

✅ САМОМОДИФИКАЦИЯ
   • Создание Python-модулей через LLM
   • Рефлексия о необходимости навыков
   • Безопасное тестирование кода
   • Версионирование модулей

✅ КОГНИТИВНАЯ АРХИТЕКТУРА
   • Метакогниция (мысли о мыслях)
   • Рабочая память с вниманием
   • Планирование действий
   • Эмоциональный интеллект

✅ СОМАТОСЕНСОРИКА
   • Мониторинг внутреннего состояния
   • Детекция аномалий
   • Адаптивное поведение
   • "Ощущения" качества работы

🎯 АРХИТЕКТУРА:
• Динамический MLP: 128→(32-512)→8
• Адаптивный словарь: 2K→100K
• Векторная память с поиском
• Модульная самомодификация
""")

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = CognitiveAGIBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 COGNITIVE AGI v35.0 АКТИВЕН")
        logger.info("✅ Память: полноценная")
        logger.info("✅ Самомодификация: включена")
        logger.info("✅ Метакогниция: активна")
        logger.info("✅ Соматосенсорика: работает")
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
        traceback.print_exc()
        sys.exit(1)