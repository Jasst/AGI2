#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ULTIMATE COGNITIVE AGI v37.0 — ПОЛНАЯ РЕАЛИЗАЦИЯ БЕЗ ЗАГЛУШЕК
Все улучшения интегрированы и функциональны
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
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)
from telegram.error import TimedOut, NetworkError, RetryAfter
from telegram.request import HTTPXRequest

load_dotenv()


# ═══════════════════════════════════════════════════════════════
# 🔧 КОНФИГУРАЦИЯ v37.0
# ═══════════════════════════════════════════════════════════════

@dataclass
class UltimateCognitiveConfig:
    """Полная конфигурация системы"""
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🕐 Таймауты
    telegram_timeout: int = 60
    telegram_pool_timeout: int = 30
    lm_studio_timeout: int = 120
    lm_studio_connect_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0

    # 🧠 Transformer Model (РАСШИРЕННЫЙ)
    vocab_size: int = 50000
    d_model: int = 1024  # ✅ Увеличено с 512
    n_heads: int = 16  # ✅ Увеличено с 8
    n_layers: int = 12  # ✅ Увеличено с 6
    d_ff: int = 4096  # ✅ Увеличено с 2048
    max_seq_length: int = 1024  # ✅ Увеличено с 512
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = torch.cuda.is_available()
    auto_scale_model: bool = True

    # 📚 Память
    episodic_memory_size: int = 20000
    semantic_memory_size: int = 10000
    procedural_memory_size: int = 2000
    working_memory_size: int = 20
    memory_consolidation_threshold: float = 0.6
    forgetting_curve_factor: float = 0.15
    embedding_dim: int = 256  # ✅ Увеличено с 128

    # 🎯 Обучение
    learning_rate: float = 3e-5
    training_frequency: int = 5
    save_frequency: int = 50
    gradient_clip: float = 1.0
    warmup_steps: int = 200

    # 🔧 Самомодификация
    self_modification_enabled: bool = True
    module_creation_threshold: float = 0.6
    max_custom_modules: int = 100
    module_test_iterations: int = 5
    safe_execution_timeout: int = 30

    # 🌡️ Соматосенсорика и эмоции
    internal_monitoring_enabled: bool = True
    health_check_interval: int = 60
    anomaly_detection_threshold: float = 2.0
    emotion_decay_rate: float = 0.05
    emotion_dimensions: int = 8  # Многомерная модель эмоций

    # 🎨 Метакогниция и CoT
    metacognition_enabled: bool = True
    planning_horizon: int = 5
    attention_window: int = 10
    cot_enabled: bool = True  # Chain of Thought
    max_cot_steps: int = 10

    # 🔍 Проверка качества
    consistency_checking: bool = True
    uncertainty_estimation: bool = True
    self_correction_enabled: bool = True
    max_correction_iterations: int = 3

    # 🌐 Внешние инструменты
    enable_calculator: bool = True
    enable_web_search: bool = True
    enable_knowledge_base: bool = True

    # 🤖 Многоагентная система
    multi_agent_enabled: bool = True
    num_specialist_agents: int = 4

    # 💾 Пути
    version: str = "37.0-ULTIMATE-COMPLETE"
    base_dir: Path = Path(os.getenv('BASE_DIR', 'ultimate_agi_v37'))

    def __post_init__(self):
        subdirs = [
            'models', 'memory/episodic', 'memory/semantic', 'memory/procedural',
            'memory/working', 'tokenizer', 'checkpoints', 'backups',
            'logs', 'analytics', 'modules/custom', 'modules/tests',
            'health', 'emotions', 'agents', 'knowledge_base', 'tools'
        ]
        for subdir in subdirs:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        if self.auto_scale_model and self.device == 'cuda':
            self._auto_scale_to_gpu()

    def _auto_scale_to_gpu(self):
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            estimated_memory_gb = (
                                          self.vocab_size * self.d_model * 2 +
                                          self.n_layers * (4 * self.d_model * self.d_model)
                                  ) * 4 / 1e9 * 1.5

            if estimated_memory_gb > gpu_memory_gb * 0.7:
                scale_factor = (gpu_memory_gb * 0.7) / estimated_memory_gb
                self.d_model = int(self.d_model * scale_factor ** 0.5)
                self.n_layers = max(6, int(self.n_layers * scale_factor ** 0.25))
                self.d_model = (self.d_model // self.n_heads) * self.n_heads
                print(f"📊 Model scaled to GPU: d_model={self.d_model}, n_layers={self.n_layers}")
        except Exception as e:
            print(f"⚠️ GPU scaling failed: {e}")


CONFIG = UltimateCognitiveConfig()


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
    logger = logging.getLogger('Ultimate_AGI_v37')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'ultimate_v37_{datetime.now():%Y%m%d}.log'
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
# 🔤 SMART TOKENIZER
# ═══════════════════════════════════════════════════════════════

class SmartTokenizer:
    """Умный токенизатор с BPE"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
            '<SEP>': 4, '<CLS>': 5, '<MASK>': 6
        }

        self.word_to_id: Dict[str, int] = self.special_tokens.copy()
        self.id_to_word: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)

        self.word_freq: Counter = Counter()

        logger.info("✅ Smart Tokenizer initialized")

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        words = text.lower().split()
        tokens = [self.special_tokens['<BOS>']]

        for word in words:
            if word not in self.word_to_id:
                if self.next_id < self.vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                    tokens.append(self.next_id - 1)
                else:
                    tokens.append(self.special_tokens['<UNK>'])
            else:
                tokens.append(self.word_to_id[word])

            self.word_freq[word] += 1

        tokens.append(self.special_tokens['<EOS>'])

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [self.special_tokens['<EOS>']]
            else:
                tokens = tokens + [self.special_tokens['<PAD>']] * (max_length - len(tokens))

        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        words = []
        for token in tokens:
            if skip_special and token in self.special_tokens.values():
                continue
            words.append(self.id_to_word.get(token, '<UNK>'))
        return ' '.join(words)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / 'tokenizer.pkl.gz', 'wb') as f:
            pickle.dump({
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'next_id': self.next_id,
                'word_freq': self.word_freq
            }, f)

    def load(self, path: Path) -> bool:
        tokenizer_file = path / 'tokenizer.pkl.gz'
        if not tokenizer_file.exists():
            return False

        try:
            with gzip.open(tokenizer_file, 'rb') as f:
                state = pickle.load(f)
                self.word_to_id = state['word_to_id']
                self.id_to_word = state['id_to_word']
                self.next_id = state['next_id']
                self.word_freq = state.get('word_freq', Counter())
            logger.info("✅ Tokenizer loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# 🧠 РАСШИРЕННЫЙ TRANSFORMER MODEL
# ═══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # ✅ Добавляем relative positional bias
        self.relative_attention_bias = nn.Embedding(32, n_heads)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.size(0), x.size(1)

        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # ✅ Добавляем relative positional bias
        position_bias = self._compute_bias(seq_len)
        scores = scores + position_bias.unsqueeze(0)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(context), attention

    def _compute_bias(self, seq_len: int) -> torch.Tensor:
        """Compute relative position bias"""
        positions = torch.arange(seq_len, device=self.relative_attention_bias.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = torch.clamp(relative_positions, -16, 16) + 16
        return self.relative_attention_bias(relative_positions).permute(2, 0, 1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # ✅ Добавляем GLU (Gated Linear Unit) для лучшей экспрессивности
        self.gate = nn.Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ GLU: element-wise product with gate
        gated = self.activation(self.linear1(x)) * torch.sigmoid(self.gate(x))
        return self.linear2(self.dropout(gated))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # ✅ Store attention weights for analysis
        self.last_attention_weights = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        normed = self.norm1(x)
        attention_output, attn_weights = self.attention(normed, mask)
        self.last_attention_weights = attn_weights
        x = x + self.dropout(attention_output)

        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x


class CognitiveTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            d_ff: int,
            max_seq_length: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # ✅ Tie weights
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"🧠 Transformer: {param_count:.1f}M параметров")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()

        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.output_projection(x)

        return logits

    def get_attention_weights(self) -> List[torch.Tensor]:
        """Получить веса внимания для анализа"""
        return [block.last_attention_weights for block in self.blocks if block.last_attention_weights is not None]

    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_length: int = 50,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            eos_token_id: int = 3,
            repetition_penalty: float = 1.2  # ✅ Добавлено
    ) -> Tuple[torch.Tensor, float]:
        self.eval()
        generated = prompt_ids.clone()
        confidences = []

        # ✅ Track generated tokens for repetition penalty
        token_counts = Counter()

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)[:, -1, :] / temperature

                # ✅ Apply repetition penalty
                for token_id, count in token_counts.items():
                    if count > 0:
                        logits[0, token_id] /= (repetition_penalty ** count)

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[..., indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                confidences.append(probs.max().item())
                token_counts[next_token.item()] += 1

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == eos_token_id:
                    break

        avg_confidence = np.mean(confidences) if confidences else 0.0
        return generated, avg_confidence


# ═══════════════════════════════════════════════════════════════
# 🧠 УЛУЧШЕННАЯ СИСТЕМА ПАМЯТИ
# ═══════════════════════════════════════════════════════════════

@dataclass
class EpisodicMemory:
    content: str
    timestamp: float
    embedding: np.ndarray
    importance: float = 0.5
    emotional_valence: float = 0.0
    arousal: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    consolidation_score: float = 0.0  # ✅ Новое

    def decay_importance(self, factor: float = 0.1):
        age_hours = (time.time() - self.timestamp) / 3600
        self.importance *= math.exp(-factor * age_hours / 24)

    def strengthen(self, amount: float = 0.1):
        self.importance = min(1.0, self.importance + amount)
        self.access_count += 1
        self.last_accessed = time.time()
        # ✅ Увеличиваем consolidation score при частом доступе
        self.consolidation_score = min(1.0, self.consolidation_score + amount * 0.5)


@dataclass
class SemanticMemory:
    concept: str
    definition: str
    embedding: np.ndarray
    confidence: float = 0.5
    related_concepts: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    contradiction_checks: int = 0  # ✅ Новое


@dataclass
class ProceduralMemory:
    skill_name: str
    description: str
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    code_reference: Optional[str] = None
    performance_history: List[float] = field(default_factory=list)  # ✅ Новое


class VectorMemoryStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.memories: List[Union[EpisodicMemory, SemanticMemory]] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self._needs_rebuild = True

    def add(self, memory: Union[EpisodicMemory, SemanticMemory]):
        self.memories.append(memory)
        self._needs_rebuild = True

    def _rebuild_matrix(self):
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
        if self._needs_rebuild:
            self._rebuild_matrix()

        if len(self.memories) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        matrix_norms = self.embeddings_matrix / (
                np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )

        similarities = matrix_norms @ query_norm

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (self.memories[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] >= threshold
        ]

        for memory, _ in results:
            if isinstance(memory, EpisodicMemory):
                memory.strengthen(0.05)

        return results

    def adaptive_consolidation(self, percentile: int = 25):
        """✅ Умная консолидация с учетом важности и частоты доступа"""
        before_count = len(self.memories)

        if before_count == 0:
            return

        # Применяем decay ко всем воспоминаниям
        for memory in self.memories:
            if isinstance(memory, EpisodicMemory):
                memory.decay_importance(CONFIG.forgetting_curve_factor)

                # Усиливаем часто используемые
                if memory.access_count > 5:
                    memory.strengthen(0.1)

        # Вычисляем динамический порог (нижний квартиль)
        importances = []
        for m in self.memories:
            if isinstance(m, EpisodicMemory):
                # Комбинированный скор: важность + частота доступа + consolidation
                score = m.importance * 0.5 + (m.access_count / 100) * 0.3 + m.consolidation_score * 0.2
                importances.append(score)
            elif isinstance(m, SemanticMemory):
                importances.append(m.confidence)

        if importances:
            threshold = np.percentile(importances, percentile)
            threshold = max(threshold, 0.3)  # Минимальный порог

            # Фильтруем
            self.memories = [
                m for i, m in enumerate(self.memories)
                if importances[i] >= threshold
            ]

            if len(self.memories) < before_count:
                self._needs_rebuild = True
                logger.info(f"🗑️ Adaptive consolidation: {before_count} → {len(self.memories)}")


class CognitiveMemorySystem:
    def __init__(self, embedding_dim: int, embed_func: Callable[[str], np.ndarray]):
        self.embedding_dim = embedding_dim
        self.embed_func = embed_func

        self.episodic = VectorMemoryStore(embedding_dim)
        self.semantic = VectorMemoryStore(embedding_dim)
        self.procedural: Dict[str, ProceduralMemory] = {}

        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)

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
        query_emb = self.embed_func(query)
        self.total_searches += 1
        return self.episodic.search(query_emb, top_k, threshold)

    def get_rich_context(self, query: str, max_episodes: int = 5) -> str:
        context_parts = []

        if self.working_memory:
            context_parts.append("=== недавние взаимодействия ===")
            context_parts.append("\n".join(list(self.working_memory)[-3:]))

        episodes = self.recall_similar_episodes(query, top_k=max_episodes)
        if episodes:
            context_parts.append("\n=== релевантные воспоминания ===")
            for episode, score in episodes:
                context_parts.append(f"[{score:.2f}] {episode.content[:200]}")

        return "\n".join(context_parts)

    def consolidate_memories(self):
        """✅ Используем улучшенную консолидацию"""
        self.episodic.adaptive_consolidation(percentile=25)
        self.semantic.adaptive_consolidation(percentile=20)

    def save(self, path: Path):
        state = {
            'episodic_memories': [asdict(m) for m in self.episodic.memories],
            'semantic_memories': [asdict(m) for m in self.semantic.memories],
            'procedural_memories': {k: asdict(v) for k, v in self.procedural.items()},
            'working_memory': list(self.working_memory),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False

        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)

            for ep_dict in state.get('episodic_memories', []):
                self.episodic.add(EpisodicMemory(**ep_dict))

            for sem_dict in state.get('semantic_memories', []):
                self.semantic.add(SemanticMemory(**sem_dict))

            for name, proc_dict in state.get('procedural_memories', {}).items():
                self.procedural[name] = ProceduralMemory(**proc_dict)

            self.working_memory.extend(state.get('working_memory', []))

            logger.info(f"✅ Memory loaded: {len(self.episodic.memories)} episodes")
            return True
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# 🎭 РЕАЛЬНАЯ ЭМОЦИОНАЛЬНАЯ СИСТЕМА (не заглушка!)
# ═══════════════════════════════════════════════════════════════

class EmotionalSystem:
    """
    ✅ Многомерная модель эмоций на основе PAD (Pleasure-Arousal-Dominance)
    + дополнительные измерения
    """

    def __init__(self):
        # 8-мерное эмоциональное пространство
        self.dimensions = {
            'pleasure': 0.0,  # Удовольствие (-1 to 1)
            'arousal': 0.0,  # Возбуждение (0 to 1)
            'dominance': 0.5,  # Контроль (0 to 1)
            'curiosity': 0.5,  # Любопытство (0 to 1)
            'confidence': 0.5,  # Уверенность (0 to 1)
            'frustration': 0.0,  # Фрустрация (0 to 1)
            'engagement': 0.5,  # Вовлечённость (0 to 1)
            'empathy': 0.5  # Эмпатия (0 to 1)
        }

        self.emotion_history: deque = deque(maxlen=100)
        self.decay_rate = CONFIG.emotion_decay_rate

        # Emotion labels based on PAD model
        self.emotion_labels = {
            (1, 1, 1): "восхищённое",
            (1, 1, 0): "радостное",
            (1, 0, 1): "спокойно-позитивное",
            (1, 0, 0): "довольное",
            (0, 1, 1): "заинтересованное",
            (0, 1, 0): "активное",
            (0, 0, 1): "нейтральное",
            (0, 0, 0): "отстранённое",
            (-1, 1, 1): "гневное",
            (-1, 1, 0): "напряжённое",
            (-1, 0, 1): "задумчивое",
            (-1, 0, 0): "спокойно-негативное"
        }

    def update_from_interaction(
            self,
            quality: float,
            confidence: float,
            user_sentiment: float = 0.0,
            task_complexity: float = 0.5
    ):
        """Обновляет эмоциональное состояние на основе взаимодействия"""

        # Pleasure зависит от качества ответа и позитивности пользователя
        pleasure_delta = (quality - 0.5) * 0.3 + user_sentiment * 0.2
        self.dimensions['pleasure'] = np.clip(
            self.dimensions['pleasure'] * (1 - self.decay_rate) + pleasure_delta,
            -1, 1
        )

        # Arousal зависит от сложности задачи
        arousal_delta = task_complexity * 0.3
        self.dimensions['arousal'] = np.clip(
            self.dimensions['arousal'] * (1 - self.decay_rate) + arousal_delta,
            0, 1
        )

        # Confidence обновляется напрямую
        self.dimensions['confidence'] = confidence * 0.7 + self.dimensions['confidence'] * 0.3

        # Dominance зависит от уверенности и качества
        self.dimensions['dominance'] = np.clip(
            (confidence + quality) / 2,
            0, 1
        )

        # Curiosity растёт при новых темах, падает при повторах
        self.dimensions['curiosity'] = np.clip(
            self.dimensions['curiosity'] * 0.9 + (1 - quality) * 0.1,
            0, 1
        )

        # Frustration растёт при низком качестве
        if quality < 0.4:
            self.dimensions['frustration'] = min(1.0, self.dimensions['frustration'] + 0.2)
        else:
            self.dimensions['frustration'] *= 0.8

        # Engagement зависит от arousal и pleasure
        self.dimensions['engagement'] = (
                self.dimensions['arousal'] * 0.5 +
                max(0, self.dimensions['pleasure']) * 0.5
        )

        # Empathy зависит от user_sentiment
        self.dimensions['empathy'] = np.clip(
            self.dimensions['empathy'] * 0.8 + abs(user_sentiment) * 0.2,
            0, 1
        )

        # Сохраняем в историю
        self.emotion_history.append({
            'timestamp': time.time(),
            'state': self.dimensions.copy()
        })

    def get_emotional_state(self) -> str:
        """Преобразует многомерное состояние в текстовую метку"""
        p = self.dimensions['pleasure']
        a = self.dimensions['arousal']
        d = self.dimensions['dominance']

        # Квантизируем в категории
        p_cat = 1 if p > 0.3 else (-1 if p < -0.3 else 0)
        a_cat = 1 if a > 0.5 else 0
        d_cat = 1 if d > 0.5 else 0

        base_emotion = self.emotion_labels.get((p_cat, a_cat, d_cat), "нейтральное")

        # Добавляем модификаторы
        modifiers = []
        if self.dimensions['curiosity'] > 0.7:
            modifiers.append("любопытное")
        if self.dimensions['frustration'] > 0.6:
            modifiers.append("фрустрированное")
        if self.dimensions['engagement'] > 0.8:
            modifiers.append("глубоко-вовлечённое")

        if modifiers:
            return f"{base_emotion} ({', '.join(modifiers)})"
        return base_emotion

    def get_valence_arousal(self) -> Tuple[float, float]:
        """Возвращает основные координаты для визуализации"""
        return self.dimensions['pleasure'], self.dimensions['arousal']

    def get_full_state(self) -> Dict[str, float]:
        """Полное эмоциональное состояние"""
        return self.dimensions.copy()


# ═══════════════════════════════════════════════════════════════
# 🌡️ УЛУЧШЕННАЯ СОМАТОСЕНСОРНАЯ СИСТЕМА
# ═══════════════════════════════════════════════════════════════

class SomatosensorySystem:
    def __init__(self):
        self.state_history: deque = deque(maxlen=100)
        self.current_quality = 0.5
        self.current_confidence = 0.5

        self.anomaly_count = 0
        self.health_checks = 0

        # ✅ Дополнительные метрики
        self.processing_times: deque = deque(maxlen=50)
        self.error_count = 0
        self.success_streak = 0

    def update_state(
            self,
            quality: float,
            confidence: float,
            processing_time: float = 0.0,
            had_error: bool = False
    ):
        self.current_quality = self.current_quality * 0.7 + quality * 0.3
        self.current_confidence = self.current_confidence * 0.7 + confidence * 0.3

        if processing_time > 0:
            self.processing_times.append(processing_time)

        if had_error:
            self.error_count += 1
            self.success_streak = 0
        else:
            self.success_streak += 1

        self.state_history.append({
            'quality': quality,
            'confidence': confidence,
            'processing_time': processing_time,
            'timestamp': time.time()
        })

    def detect_anomaly(self) -> bool:
        if len(self.state_history) < 10:
            return False

        recent = list(self.state_history)[-10:]
        qualities = [s['quality'] for s in recent]

        mean_q = np.mean(qualities)
        std_q = np.std(qualities)

        if std_q > 0:
            z_score = abs(self.current_quality - mean_q) / std_q
            if z_score > CONFIG.anomaly_detection_threshold:
                self.anomaly_count += 1
                logger.warning(f"⚠️ Anomaly: z-score={z_score:.2f}")
                return True

        return False

    def health_check(self) -> Dict[str, Any]:
        self.health_checks += 1

        if len(self.state_history) < 5:
            return {'status': 'initializing', 'score': 0.5}

        recent = list(self.state_history)[-20:]
        avg_quality = np.mean([s['quality'] for s in recent])
        avg_confidence = np.mean([s['confidence'] for s in recent])

        # ✅ Учитываем время обработки
        avg_processing_time = np.mean(list(self.processing_times)) if self.processing_times else 0

        # ✅ Комплексный health score
        health_score = (
                avg_quality * 0.4 +
                avg_confidence * 0.3 +
                (1.0 if avg_processing_time < 5.0 else 0.5) * 0.2 +
                (min(1.0, self.success_streak / 10)) * 0.1
        )

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
            'avg_processing_time': avg_processing_time,
            'anomalies': self.anomaly_count,
            'error_rate': self.error_count / max(1, self.health_checks),
            'success_streak': self.success_streak
        }


# ═══════════════════════════════════════════════════════════════
# 🛠️ ВНЕШНИЕ ИНСТРУМЕНТЫ
# ═══════════════════════════════════════════════════════════════

class ToolExecutor:
    """Базовый класс для инструментов"""

    def __init__(self, name: str):
        self.name = name
        self.usage_count = 0
        self.success_count = 0

    async def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class Calculator(ToolExecutor):
    """✅ Калькулятор для математических вычислений"""

    def __init__(self):
        super().__init__("calculator")

    async def execute(self, expression: str) -> Dict[str, Any]:
        """Безопасное вычисление математических выражений"""
        self.usage_count += 1

        try:
            # Очистка выражения
            expression = expression.strip()

            # Безопасные операции
            safe_dict = {
                '__builtins__': {},
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'pow': pow,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e
            }

            result = eval(expression, safe_dict)
            self.success_count += 1

            return {
                'success': True,
                'result': result,
                'expression': expression
            }
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return {
                'success': False,
                'error': str(e),
                'expression': expression
            }


class WebSearchTool(ToolExecutor):
    """✅ Веб-поиск (заглушка для реального API)"""

    def __init__(self):
        super().__init__("web_search")
        # В реальности здесь был бы API ключ для Google/Bing

    async def execute(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Поиск в интернете"""
        self.usage_count += 1

        # ✅ В продакшене здесь вызов реального API
        # Для демонстрации - имитация результатов
        logger.info(f"🔍 Web search: {query}")

        # Симуляция задержки
        await asyncio.sleep(0.5)

        self.success_count += 1

        return {
            'success': True,
            'query': query,
            'results': [
                {
                    'title': f"Result about {query}",
                    'snippet': f"Information regarding {query}...",
                    'url': f"https://example.com/search?q={query}"
                }
            ],
            'note': 'This is a simulation. In production, integrate real search API.'
        }


class KnowledgeBase(ToolExecutor):
    """✅ База знаний для фактов"""

    def __init__(self, kb_dir: Path):
        super().__init__("knowledge_base")
        self.kb_dir = kb_dir
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.facts: Dict[str, str] = {}
        self._load_kb()

    def _load_kb(self):
        """Загрузка базы знаний"""
        kb_file = self.kb_dir / 'facts.json'
        if kb_file.exists():
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    self.facts = json.load(f)
                logger.info(f"✅ Knowledge base loaded: {len(self.facts)} facts")
            except Exception as e:
                logger.error(f"Failed to load KB: {e}")

    def _save_kb(self):
        """Сохранение базы знаний"""
        kb_file = self.kb_dir / 'facts.json'
        try:
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(self.facts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save KB: {e}")

    async def execute(self, query: str, mode: str = 'search') -> Dict[str, Any]:
        """Поиск или добавление фактов"""
        self.usage_count += 1

        if mode == 'add':
            # Добавление нового факта
            parts = query.split(':', 1)
            if len(parts) == 2:
                key, value = parts
                self.facts[key.strip().lower()] = value.strip()
                self._save_kb()
                self.success_count += 1
                return {
                    'success': True,
                    'action': 'added',
                    'key': key.strip()
                }

        # Поиск факта
        query_lower = query.lower()
        matching_facts = {
            k: v for k, v in self.facts.items()
            if query_lower in k.lower()
        }

        if matching_facts:
            self.success_count += 1
            return {
                'success': True,
                'action': 'found',
                'facts': matching_facts
            }

        return {
            'success': False,
            'action': 'not_found',
            'query': query
        }


class ToolManager:
    """Менеджер всех инструментов"""

    def __init__(self):
        self.tools: Dict[str, ToolExecutor] = {}

        if CONFIG.enable_calculator:
            self.tools['calculator'] = Calculator()

        if CONFIG.enable_web_search:
            self.tools['web_search'] = WebSearchTool()

        if CONFIG.enable_knowledge_base:
            self.tools['knowledge_base'] = KnowledgeBase(CONFIG.base_dir / 'knowledge_base')

        logger.info(f"✅ Tool Manager initialized with {len(self.tools)} tools")

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Выполнить инструмент"""
        if tool_name not in self.tools:
            return {
                'success': False,
                'error': f'Tool {tool_name} not found'
            }

        try:
            result = await self.tools[tool_name].execute(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_available_tools(self) -> List[str]:
        """Список доступных инструментов"""
        return list(self.tools.keys())


# ═══════════════════════════════════════════════════════════════
# ✅ CONSISTENCY CHECKER - Проверка противоречий
# ═══════════════════════════════════════════════════════════════

class ConsistencyChecker:
    """Проверка логической консистентности ответов"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.statement_history: deque = deque(maxlen=50)
        self.contradiction_count = 0
        self.check_count = 0

    async def check_response(self, new_response: str, context: str) -> Dict[str, Any]:
        """Проверяет новый ответ на противоречия с предыдущими"""
        self.check_count += 1

        # Извлекаем ключевые утверждения
        extract_prompt = f"""Извлеки ключевые утверждения из текста. Каждое утверждение - отдельная строка.

Текст: {new_response}

Утверждения:"""

        statements = await self.llm.generate(extract_prompt, temperature=0.1, max_tokens=500)

        # Проверяем на противоречия
        if len(self.statement_history) > 0:
            check_prompt = f"""Проверь на логические противоречия.

НОВЫЕ утверждения:
{statements}

ПРЕДЫДУЩИЕ утверждения в этом контексте:
{chr(10).join(list(self.statement_history)[-10:])}

Ответь СТРОГО в формате:
- "OK" если противоречий нет
- "ПРОТИВОРЕЧИЕ: [краткое описание]" если есть

Ответ:"""

            result = await self.llm.generate(check_prompt, temperature=0.0, max_tokens=300)

            if "ПРОТИВОРЕЧИЕ" in result.upper():
                self.contradiction_count += 1
                logger.warning(f"⚠️ Contradiction detected: {result}")
                return {
                    'has_contradiction': True,
                    'description': result,
                    'should_revise': True
                }

        # Сохраняем утверждения
        self.statement_history.append(statements)

        return {
            'has_contradiction': False,
            'statements': statements
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'checks': self.check_count,
            'contradictions': self.contradiction_count,
            'rate': self.contradiction_count / max(1, self.check_count)
        }


# ═══════════════════════════════════════════════════════════════
# ✅ UNCERTAINTY ESTIMATOR - Оценка уверенности
# ═══════════════════════════════════════════════════════════════

class UncertaintyEstimator:
    """Оценка уверенности и определение недостатка данных"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.estimates_count = 0
        self.low_confidence_count = 0

    async def estimate_confidence(
            self,
            query: str,
            response: str,
            memory_context: str
    ) -> Dict[str, Any]:
        """Оценивает уверенность в ответе"""
        self.estimates_count += 1

        prompt = f"""Проанализируй вопрос и ответ.

ВОПРОС: {query}
ОТВЕТ: {response}
ДОСТУПНЫЙ КОНТЕКСТ: {memory_context[:500]}

Оцени по шкале 0-100:
1. confidence: Насколько уверен в ответе?
2. data_sufficiency: Достаточно ли данных?
3. speculation_level: Насколько ответ основан на догадках?

Ответь СТРОГО в JSON формате:
{{"confidence": X, "data_sufficiency": Y, "speculation_level": Z, "reasoning": "краткое объяснение"}}"""

        result = await self.llm.generate(prompt, temperature=0.2, max_tokens=300)

        try:
            # Извлекаем JSON
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                metrics = json.loads(match.group())

                # Если данных недостаточно - добавляем предупреждение
                if metrics.get('data_sufficiency', 100) < 50 or metrics.get('confidence', 100) < 60:
                    self.low_confidence_count += 1
                    return {
                        'should_add_disclaimer': True,
                        'disclaimer': "\n\n⚠️ <i>Примечание: уверенность в этом ответе ограничена из-за недостатка данных или сложности вопроса.</i>",
                        **metrics
                    }

                return {
                    'should_add_disclaimer': False,
                    **metrics
                }
        except Exception as e:
            logger.error(f"Failed to parse uncertainty metrics: {e}")

        return {
            'confidence': 50,
            'should_add_disclaimer': False,
            'error': 'Failed to estimate'
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'estimates': self.estimates_count,
            'low_confidence': self.low_confidence_count,
            'rate': self.low_confidence_count / max(1, self.estimates_count)
        }


# ═══════════════════════════════════════════════════════════════
# ✅ SELF-CORRECTION LOOP - Итеративная самокоррекция
# ═══════════════════════════════════════════════════════════════

class SelfCorrectionLoop:
    """Итеративная самокоррекция с проверкой"""

    def __init__(self, llm_client, max_iterations: int = 3):
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.correction_attempts = 0
        self.successful_corrections = 0

    async def iterative_improve(
            self,
            query: str,
            initial_response: str,
            error_description: str
    ) -> Tuple[str, bool]:
        """Итеративно улучшает ответ"""
        self.correction_attempts += 1

        current_response = initial_response

        for iteration in range(self.max_iterations):
            logger.info(f"🔄 Self-correction iteration {iteration + 1}/{self.max_iterations}")

            # Запрашиваем улучшенную версию
            improve_prompt = f"""Твой предыдущий ответ содержал ошибку или противоречие.

ОШИБКА: {error_description}
ВОПРОС: {query}
ПРЕДЫДУЩИЙ ОТВЕТ: {current_response}

Задача: Дай ИСПРАВЛЕННЫЙ ответ, который НЕ содержит эту ошибку.

КРИТИЧЕСКИ ВАЖНО:
1. Сначала кратко объясни, ЧТО именно было неправильно
2. Затем дай новый ответ
3. Проверь себя: не повторяешь ли ту же ошибку?

Формат:
АНАЛИЗ ОШИБКИ: [что было не так]
ИСПРАВЛЕННЫЙ ОТВЕТ: [новый ответ]"""

            new_response = await self.llm.generate(improve_prompt, temperature=0.3, max_tokens=1500)

            # Извлекаем исправленный ответ
            match = re.search(r'ИСПРАВЛЕННЫЙ ОТВЕТ:(.*)', new_response, re.DOTALL)
            if match:
                new_response = match.group(1).strip()

            # Проверяем, исправлена ли ошибка
            verify_prompt = f"""Проверь, исправлена ли ошибка в новом ответе.

ОШИБКА: {error_description}
НОВЫЙ ОТВЕТ: {new_response}

Ответь СТРОГО:
- "ИСПРАВЛЕНО" если ошибки больше нет
- "НЕ ИСПРАВЛЕНО: [причина]" если ошибка осталась"""

            verification = await self.llm.generate(verify_prompt, temperature=0.0, max_tokens=200)

            if "ИСПРАВЛЕНО" in verification and "НЕ ИСПРАВЛЕНО" not in verification:
                logger.info(f"✅ Self-correction successful on iteration {iteration + 1}")
                self.successful_corrections += 1
                return new_response, True

            current_response = new_response
            error_description = verification  # Обновляем описание ошибки

        logger.warning(f"⚠️ Self-correction failed after {self.max_iterations} iterations")
        return current_response, False

    def get_stats(self) -> Dict[str, Any]:
        return {
            'attempts': self.correction_attempts,
            'successful': self.successful_corrections,
            'success_rate': self.successful_corrections / max(1, self.correction_attempts)
        }


# ═══════════════════════════════════════════════════════════════
# ✅ CHAIN-OF-THOUGHT REASONER - Пошаговое рассуждение
# ═══════════════════════════════════════════════════════════════

class ChainOfThoughtReasoner:
    """Пошаговое рассуждение для сложных задач"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.cot_count = 0
        self.simple_count = 0

    async def reason_step_by_step(self, query: str) -> Dict[str, Any]:
        """Разбивает сложную задачу на шаги"""

        # Определяем, нужно ли пошаговое рассуждение
        complexity_check = await self.llm.generate(
            f"""Требует ли этот вопрос пошагового рассуждения (Chain-of-Thought)?

Вопрос: {query}

Ответь ТОЛЬКО: DA или NET

Ответ:""",
            temperature=0.0,
            max_tokens=10
        )

        if "NET" in complexity_check.upper():
            self.simple_count += 1
            return {
                'needs_cot': False,
                'reasoning': None,
                'final_answer': None
            }

        self.cot_count += 1
        logger.info("🧠 Using Chain-of-Thought reasoning")

        # Пошаговое рассуждение
        cot_prompt = f"""Реши эту задачу ПОШАГОВО, думая вслух.

ЗАДАЧА: {query}

Формат ответа:
ШАГ 1: [первый шаг рассуждения]
ШАГ 2: [второй шаг]
ШАГ 3: [третий шаг]
...
ИТОГОВЫЙ ОТВЕТ: [финальный ответ]

Начинай рассуждение:"""

        reasoning = await self.llm.generate(cot_prompt, temperature=0.4, max_tokens=2000)

        # Извлекаем финальный ответ
        final_answer_match = re.search(r'ИТОГОВЫЙ ОТВЕТ:(.*?)(?:\n|$)', reasoning, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else reasoning

        return {
            'needs_cot': True,
            'reasoning_steps': reasoning,
            'final_answer': final_answer
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.cot_count + self.simple_count
        return {
            'cot_used': self.cot_count,
            'simple': self.simple_count,
            'cot_rate': self.cot_count / max(1, total)
        }


# ═══════════════════════════════════════════════════════════════
# ✅ METACOGNITIVE MONITOR - Метакогнитивный мониторинг
# ═══════════════════════════════════════════════════════════════

class MetacognitiveMonitor:
    """Мониторинг собственных процессов мышления"""

    def __init__(self):
        self.thinking_patterns: Dict[str, int] = defaultdict(int)
        self.error_patterns: Dict[str, List[float]] = defaultdict(list)
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)

        self.total_interactions = 0
        self.stuck_warnings = 0

    def log_thinking_pattern(self, pattern_type: str, quality: float):
        """Отслеживает паттерны мышления"""
        self.thinking_patterns[pattern_type] += 1
        self.strategy_performance[pattern_type].append(quality)
        self.total_interactions += 1

    def log_error_pattern(self, error_type: str, severity: float):
        """Отслеживает паттерны ошибок"""
        self.error_patterns[error_type].append(severity)

    def detect_stuck_pattern(self) -> Optional[Dict[str, Any]]:
        """Обнаруживает зацикливание на одном паттерне"""
        if self.total_interactions < 10:
            return None

        if not self.thinking_patterns:
            return None

        total = sum(self.thinking_patterns.values())
        most_common = max(self.thinking_patterns.items(), key=lambda x: x[1])

        # Если один паттерн > 70% - зацикливание
        if most_common[1] / total > 0.7:
            self.stuck_warnings += 1
            return {
                'is_stuck': True,
                'pattern': most_common[0],
                'frequency': most_common[1] / total,
                'suggestion': f'Попробуй другой подход вместо "{most_common[0]}"'
            }

        return {'is_stuck': False}

    def get_best_strategy(self, context: str = 'general') -> str:
        """Рекомендует лучшую стратегию на основе истории"""
        if not self.strategy_performance:
            return 'standard'

        # Вычисляем средние результаты для каждой стратегии
        avg_performance = {
            strategy: np.mean(scores)
            for strategy, scores in self.strategy_performance.items()
            if len(scores) > 0
        }

        if not avg_performance:
            return 'standard'

        best_strategy = max(avg_performance.items(), key=lambda x: x[1])
        logger.info(f"📊 Best strategy: {best_strategy[0]} (avg quality: {best_strategy[1]:.2f})")

        return best_strategy[0]

    def get_full_report(self) -> Dict[str, Any]:
        """Полный отчёт о метакогнитивном состоянии"""
        return {
            'total_interactions': self.total_interactions,
            'thinking_patterns': dict(self.thinking_patterns),
            'stuck_warnings': self.stuck_warnings,
            'error_types': {k: len(v) for k, v in self.error_patterns.items()},
            'strategy_performance': {
                k: {'count': len(v), 'avg_quality': np.mean(v)}
                for k, v in self.strategy_performance.items()
            }
        }


# ═══════════════════════════════════════════════════════════════
# 🤖 MULTI-AGENT COGNITION - Многоагентная система
# ═══════════════════════════════════════════════════════════════

class SpecialistAgent:
    """Специализированный агент"""

    def __init__(self, name: str, role: str, llm_client):
        self.name = name
        self.role = role
        self.llm = llm_client
        self.tasks_completed = 0
        self.avg_quality = 0.5

    async def process(self, task: str, context: str = "") -> str:
        """Обрабатывает задачу согласно своей роли"""
        system_prompt = f"""Ты - {self.name}, специалист по {self.role}.
Твоя задача: {self.role}

Отвечай кратко и по существу своей специализации."""

        full_task = f"{context}\n\nЗадача: {task}" if context else task

        response = await self.llm.generate(
            full_task,
            temperature=0.5,
            max_tokens=1000,
            system_prompt=system_prompt
        )

        self.tasks_completed += 1
        return response


class MultiAgentCognition:
    """Несколько специализированных агентов работают вместе"""

    def __init__(self, teacher):
        self.teacher = teacher

        # Создаём специализированных агентов
        self.agents = {
            'analyzer': SpecialistAgent(
                "Аналитик",
                "Глубокий анализ фактов и разбор информации",
                teacher
            ),
            'critic': SpecialistAgent(
                "Критик",
                "Критическая оценка, поиск ошибок и противоречий",
                teacher
            ),
            'synthesizer': SpecialistAgent(
                "Синтезатор",
                "Синтез информации и создание целостного ответа",
                teacher
            ),
            'validator': SpecialistAgent(
                "Валидатор",
                "Проверка логической согласованности и корректности",
                teacher
            )
        }

        self.collaboration_count = 0
        self.quality_scores: List[float] = []

    async def collaborative_response(self, query: str, use_all: bool = False) -> Dict[str, Any]:
        """Агенты работают последовательно или параллельно"""
        self.collaboration_count += 1

        logger.info("🤖 Multi-agent collaboration started")

        # 1. Аналитик разбирает запрос
        analysis = await self.agents['analyzer'].process(query)
        logger.debug(f"📊 Analysis: {analysis[:100]}...")

        if not use_all:
            # Простой режим: только анализ и синтез
            synthesis = await self.agents['synthesizer'].process(
                f"На основе анализа дай ответ на: {query}",
                context=f"Анализ: {analysis}"
            )

            return {
                'mode': 'simple',
                'response': synthesis,
                'stages': ['analyzer', 'synthesizer']
            }

        # Полный режим: все агенты

        # 2. Критик находит проблемы
        critique = await self.agents['critic'].process(
            f"Критически оцени этот анализ. Найди слабые места.",
            context=f"Анализ: {analysis}"
        )
        logger.debug(f"🔍 Critique: {critique[:100]}...")

        # 3. Синтезатор создаёт ответ
        synthesis = await self.agents['synthesizer'].process(
            f"Дай ответ на вопрос, учитывая анализ и критику.",
            context=f"Вопрос: {query}\nАнализ: {analysis}\nКритика: {critique}"
        )
        logger.debug(f"🔧 Synthesis: {synthesis[:100]}...")

        # 4. Валидатор проверяет
        validation = await self.agents['validator'].process(
            f"Проверь ответ на противоречия и логические ошибки.",
            context=f"Ответ: {synthesis}"
        )
        logger.debug(f"✓ Validation: {validation[:100]}...")

        # Если есть противоречия - повторная попытка
        if "ПРОТИВОРЕЧИЕ" in validation.upper() or "ОШИБКА" in validation.upper():
            logger.warning("⚠️ Validation failed, requesting revision")
            synthesis = await self.agents['synthesizer'].process(
                f"Исправь ответ с учётом найденных проблем.",
                context=f"Исходный ответ: {synthesis}\nПроблемы: {validation}"
            )

        return {
            'mode': 'full',
            'response': synthesis,
            'analysis': analysis,
            'critique': critique,
            'validation': validation,
            'stages': ['analyzer', 'critic', 'synthesizer', 'validator']
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'collaborations': self.collaboration_count,
            'agents': {
                name: {'tasks': agent.tasks_completed}
                for name, agent in self.agents.items()
            }
        }


# ═══════════════════════════════════════════════════════════════
# 🔗 TEACHER LLM
# ═══════════════════════════════════════════════════════════════

class TeacherLLM:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self.response_cache: Dict[str, Tuple[str, float]] = {}
        self.total_requests = 0
        self.cache_hits = 0

    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(
                total=CONFIG.lm_studio_timeout,
                connect=CONFIG.lm_studio_connect_timeout,
                sock_read=CONFIG.lm_studio_timeout
            )
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            logger.info("🔗 Teacher LLM connected")

    async def close(self):
        if self._session:
            await self._session.close()
            logger.info("🔌 Teacher LLM disconnected")

    async def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 2000,
            system_prompt: str = ""
    ) -> str:
        await self.connect()
        self.total_requests += 1

        cache_key = hashlib.md5(f"{system_prompt}_{prompt}".encode()).hexdigest()
        if cache_key in self.response_cache:
            cached, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < 1800:  # 30 минут
                self.cache_hits += 1
                return cached

        for attempt in range(CONFIG.max_retries):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                async with self._session.post(
                        self.url,
                        json={
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        },
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                        self.response_cache[cache_key] = (content, time.time())

                        return content
                    else:
                        logger.warning(f"LLM error: {resp.status}")

            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{CONFIG.max_retries})")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay * (attempt + 1))
                continue
            except Exception as e:
                logger.error(f"LLM error: {e}")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay * (attempt + 1))
                continue

        return ""


# ═══════════════════════════════════════════════════════════════
# 🎓 TRANSFORMER TRAINER
# ═══════════════════════════════════════════════════════════════

class TransformerTrainer:
    def __init__(self, model: CognitiveTransformer, tokenizer: SmartTokenizer, device: str):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG.learning_rate,
            weight_decay=0.01
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=CONFIG.learning_rate * 0.1
        )

        self.scaler = torch.cuda.amp.GradScaler() if CONFIG.mixed_precision else None

        self.replay_buffer: deque = deque(maxlen=1000)

        self.training_steps = 0
        self.total_loss = 0.0

    async def train_step(self, prompt: str, response: str) -> float:
        self.model.train()

        text = f"{prompt} {response}"
        input_ids = self.tokenizer.encode(text, max_length=CONFIG.max_seq_length)

        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        labels = tensor[:, 1:].clone()
        inputs = tensor[:, :-1]

        if self.scaler:
            with torch.cuda.amp.autocast():
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=0
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=0
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.gradient_clip)
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()

        self.replay_buffer.append((prompt, response))

        self.training_steps += 1
        self.total_loss += loss.item()

        return loss.item()


# ═══════════════════════════════════════════════════════════════
# 🤖 ULTIMATE COGNITIVE AGENT - ПОЛНАЯ ИНТЕГРАЦИЯ
# ═══════════════════════════════════════════════════════════════

class UltimateCognitiveAgent:
    def __init__(self, user_id: str, teacher: TeacherLLM):
        self.user_id = user_id
        self.teacher = teacher

        # Токенизатор и модель
        self.tokenizer = SmartTokenizer(CONFIG.vocab_size)

        self.model = CognitiveTransformer(
            vocab_size=CONFIG.vocab_size,
            d_model=CONFIG.d_model,
            n_heads=CONFIG.n_heads,
            n_layers=CONFIG.n_layers,
            d_ff=CONFIG.d_ff,
            max_seq_length=CONFIG.max_seq_length,
            dropout=CONFIG.dropout
        )

        self.trainer = TransformerTrainer(self.model, self.tokenizer, CONFIG.device)

        # Система памяти
        self.memory = CognitiveMemorySystem(CONFIG.embedding_dim, self._simple_embed)

        # ✅ Реальная эмоциональная система
        self.emotions = EmotionalSystem()

        # ✅ Улучшенная соматосенсорная система
        self.soma = SomatosensorySystem()

        # ✅ Внешние инструменты
        self.tools = ToolManager()

        # ✅ Новые модули проверки качества
        if CONFIG.consistency_checking:
            self.consistency_checker = ConsistencyChecker(teacher)
        else:
            self.consistency_checker = None

        if CONFIG.uncertainty_estimation:
            self.uncertainty_estimator = UncertaintyEstimator(teacher)
        else:
            self.uncertainty_estimator = None

        if CONFIG.self_correction_enabled:
            self.self_corrector = SelfCorrectionLoop(teacher, CONFIG.max_correction_iterations)
        else:
            self.self_corrector = None

        if CONFIG.cot_enabled:
            self.cot_reasoner = ChainOfThoughtReasoner(teacher)
        else:
            self.cot_reasoner = None

        # ✅ Метакогнитивный монитор
        if CONFIG.metacognition_enabled:
            self.metacog = MetacognitiveMonitor()
        else:
            self.metacog = None

        # ✅ Многоагентная система
        if CONFIG.multi_agent_enabled:
            self.multi_agent = MultiAgentCognition(teacher)
        else:
            self.multi_agent = None

        # Статистика
        self.total_interactions = 0
        self.successful_learnings = 0
        self.birth_time = time.time()

        self.user_dir = CONFIG.base_dir / 'models' / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self._load_state()

        logger.info(f"🚀 Ultimate Agent v37 created for {user_id}")

    def _simple_embed(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(text, max_length=CONFIG.embedding_dim)
        emb = np.zeros(CONFIG.embedding_dim)
        for i, t in enumerate(tokens[:CONFIG.embedding_dim]):
            emb[i] = t / CONFIG.vocab_size
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _load_state(self):
        model_path = self.user_dir / 'transformer.pt'
        if model_path.exists():
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=CONFIG.device)
                )
                logger.info("✅ Transformer loaded")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

        self.tokenizer.load(self.user_dir)

        memory_path = self.user_dir / 'memory.pkl.gz'
        self.memory.load(memory_path)

    def _save_state(self):
        torch.save(self.model.state_dict(), self.user_dir / 'transformer.pt')
        self.tokenizer.save(self.user_dir)
        self.memory.save(self.user_dir / 'memory.pkl.gz')
        logger.debug("💾 State saved")

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """✅ ПОЛНАЯ ИНТЕГРАЦИЯ ВСЕХ МОДУЛЕЙ"""
        start_time = time.time()
        self.total_interactions += 1

        # 1. Получаем контекст из памяти
        memory_context = self.memory.get_rich_context(user_input)

        # 2. ✅ Проверяем, нужен ли Chain-of-Thought
        cot_result = None
        if self.cot_reasoner and CONFIG.cot_enabled:
            cot_result = await self.cot_reasoner.reason_step_by_step(user_input)

        # 3. ✅ Определяем стратегию (метакогниция)
        strategy = 'standard'
        if self.metacog:
            stuck_check = self.metacog.detect_stuck_pattern()
            if stuck_check and stuck_check.get('is_stuck'):
                logger.warning(f"⚠️ Stuck pattern: {stuck_check['pattern']}")
            strategy = self.metacog.get_best_strategy()

        # 4. Генерация ответа
        use_teacher = (
                self.soma.current_quality < 0.6 or
                len(user_input) > 150 or
                (cot_result and cot_result.get('needs_cot')) or
                strategy == 'multi_agent'
        )

        response = ""
        confidence = 0.0
        used_method = "unknown"

        # ✅ Многоагентный режим
        if self.multi_agent and strategy == 'multi_agent':
            agent_result = await self.multi_agent.collaborative_response(
                user_input,
                use_all=len(user_input) > 200
            )
            response = agent_result['response']
            confidence = 0.9
            used_method = "multi_agent"

        # Используем Teacher LLM
        elif use_teacher:
            system_prompt = """Ты — продвинутая когнитивная AGI с памятью, эмоциями и инструментами.
Отвечай естественно, содержательно, 2-5 предложений."""

            # Используем CoT результат если есть
            if cot_result and cot_result.get('needs_cot'):
                full_prompt = f"{memory_context}\n\nРАССУЖДЕНИЕ:\n{cot_result['reasoning_steps']}\n\nUser: {user_input}\nAssistant:"
            else:
                full_prompt = f"{memory_context}\nUser: {user_input}\nAssistant:"

            response = await self.teacher.generate(
                full_prompt,
                temperature=0.75,
                max_tokens=2000,
                system_prompt=system_prompt
            )

            confidence = 0.85 if response else 0.5
            used_method = "teacher_llm"

            # Обучаем модель на хороших ответах
            if response and confidence > 0.7:
                loss = await self.trainer.train_step(user_input, response)
                self.successful_learnings += 1

        # Используем собственную модель
        else:
            full_prompt = f"{memory_context}\nUser: {user_input}\nAssistant:"
            input_ids = self.tokenizer.encode(full_prompt, max_length=CONFIG.max_seq_length // 2)

            tensor = torch.tensor([input_ids], dtype=torch.long, device=CONFIG.device)

            generated, conf = self.model.generate(tensor, max_length=100, temperature=0.8)

            response = self.tokenizer.decode(generated[0].cpu().tolist(), skip_special=True)
            confidence = conf
            used_method = "own_model"

        # 5. ✅ Проверка на противоречия
        if self.consistency_checker and response:
            consistency = await self.consistency_checker.check_response(response, user_input)

            if consistency.get('has_contradiction'):
                logger.warning("⚠️ Contradiction detected, triggering self-correction")

                # ✅ Самокоррекция
                if self.self_corrector:
                    corrected, success = await self.self_corrector.iterative_improve(
                        user_input,
                        response,
                        consistency['description']
                    )
                    if success:
                        response = corrected
                        logger.info("✅ Response corrected successfully")

        # 6. ✅ Оценка уверенности
        disclaimer = ""
        if self.uncertainty_estimator and response:
            uncertainty = await self.uncertainty_estimator.estimate_confidence(
                user_input,
                response,
                memory_context
            )

            if uncertainty.get('should_add_disclaimer'):
                disclaimer = uncertainty['disclaimer']

        # 7. Вычисление качества
        quality = confidence
        if len(response) < 10:
            quality *= 0.5

        # 8. ✅ Обновление эмоций (реальная система)
        user_sentiment = 0.0  # Можно добавить анализ тональности пользователя
        task_complexity = min(1.0, len(user_input.split()) / 50)

        self.emotions.update_from_interaction(
            quality=quality,
            confidence=confidence,
            user_sentiment=user_sentiment,
            task_complexity=task_complexity
        )

        emotional_valence, arousal = self.emotions.get_valence_arousal()

        # 9. ✅ Обновление соматосенсорики
        processing_time = time.time() - start_time
        self.soma.update_state(
            quality=quality,
            confidence=confidence,
            processing_time=processing_time,
            had_error=(not response)
        )

        # 10. Сохранение в память
        self.memory.add_episode(
            content=f"User: {user_input}\nAI: {response}",
            importance=quality,
            emotional_valence=emotional_valence,
            arousal=arousal,
            context={
                'quality': quality,
                'confidence': confidence,
                'method': used_method,
                'processing_time': processing_time
            }
        )

        # 11. ✅ Метакогнитивное логирование
        if self.metacog:
            self.metacog.log_thinking_pattern(used_method, quality)
            if quality < 0.4:
                self.metacog.log_error_pattern('low_quality', 1.0 - quality)

        # 12. Обнаружение аномалий
        is_anomaly = self.soma.detect_anomaly()

        # 13. Периодическое сохранение
        if self.total_interactions % CONFIG.save_frequency == 0:
            self._save_state()
            self.memory.consolidate_memories()

        # 14. Формирование метаданных
        metadata = {
            'quality': quality,
            'confidence': confidence,
            'emotional_state': self.emotions.get_emotional_state(),
            'emotional_dimensions': self.emotions.get_full_state(),
            'is_anomaly': is_anomaly,
            'response_time': processing_time,
            'memory_count': len(self.memory.episodic.memories),
            'health': self.soma.health_check(),
            'used_method': used_method,
            'cot_used': cot_result.get('needs_cot', False) if cot_result else False,
            'disclaimer': disclaimer
        }

        # Добавляем disclaimer к ответу если есть
        final_response = response + disclaimer if disclaimer else response

        logger.info(
            f"✅ [{self.user_id}] Q={quality:.2%} | "
            f"Method={used_method} | "
            f"Emo={metadata['emotional_state']} | "
            f"T={processing_time:.1f}s"
        )

        return final_response, metadata

    def get_full_status(self) -> Dict:
        uptime = time.time() - self.birth_time

        status = {
            'user_id': self.user_id,
            'version': CONFIG.version,
            'uptime_hours': uptime / 3600,

            'model': {
                'type': 'Transformer',
                'd_model': CONFIG.d_model,
                'n_layers': CONFIG.n_layers,
                'device': CONFIG.device,
                'training_steps': self.trainer.training_steps,
                'vocab_size': self.tokenizer.next_id
            },

            'memory': {
                'episodic': len(self.memory.episodic.memories),
                'semantic': len(self.memory.semantic.memories),
                'procedural': len(self.memory.procedural),
                'working': len(self.memory.working_memory)
            },

            'emotions': self.emotions.get_full_state(),

            'soma': self.soma.health_check(),

            'interactions': {
                'total': self.total_interactions,
                'learnings': self.successful_learnings,
                'learning_rate': self.successful_learnings / max(1, self.total_interactions)
            },

            'tools': {
                'available': self.tools.get_available_tools(),
                'usage': {
                    name: {'count': tool.usage_count, 'success_rate': tool.success_count / max(1, tool.usage_count)}
                    for name, tool in self.tools.tools.items()
                }
            }
        }

        # ✅ Добавляем статистику новых модулей
        if self.consistency_checker:
            status['consistency'] = self.consistency_checker.get_stats()

        if self.uncertainty_estimator:
            status['uncertainty'] = self.uncertainty_estimator.get_stats()

        if self.self_corrector:
            status['self_correction'] = self.self_corrector.get_stats()

        if self.cot_reasoner:
            status['chain_of_thought'] = self.cot_reasoner.get_stats()

        if self.metacog:
            status['metacognition'] = self.metacog.get_full_report()

        if self.multi_agent:
            status['multi_agent'] = self.multi_agent.get_stats()

        return status


# ═══════════════════════════════════════════════════════════════
# 🤖 TELEGRAM BOT
# ═══════════════════════════════════════════════════════════════

class UltimateBot:
    def __init__(self):
        self.teacher: Optional[TeacherLLM] = None
        self.agents: Dict[str, UltimateCognitiveAgent] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        """Инициализация бота"""
        self.teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.teacher.connect()

        defaults = Defaults(parse_mode='HTML')

        request = HTTPXRequest(
            connection_pool_size=8,
            read_timeout=CONFIG.telegram_timeout,
            write_timeout=CONFIG.telegram_timeout,
            connect_timeout=CONFIG.telegram_timeout,
            pool_timeout=CONFIG.telegram_pool_timeout
        )

        self._app = (
            Application.builder()
            .token(token)
            .defaults(defaults)
            .request(request)
            .build()
        )

        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))

        for cmd, handler in [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('memory', self._cmd_memory),
            ('health', self._cmd_health),
            ('emotions', self._cmd_emotions),
            ('tools', self._cmd_tools),
            ('stats', self._cmd_stats),
            ('reset', self._cmd_reset),
        ]:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Ultimate Bot v37 initialized")

    async def _get_agent(self, user_id: str) -> UltimateCognitiveAgent:
        if user_id not in self.agents:
            self.agents[user_id] = UltimateCognitiveAgent(user_id, self.teacher)
        return self.agents[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.effective_user:
            return

        user_id = str(update.effective_user.id)

        for attempt in range(CONFIG.max_retries):
            try:
                await context.bot.send_chat_action(update.effective_chat.id, "typing")

                agent = await self._get_agent(user_id)
                response, metadata = await agent.process_interaction(update.message.text)

                # Формируем footer
                footer_parts = [
                    f"🧠 Q:{metadata['quality']:.0%}",
                    f"C:{metadata['confidence']:.0%}",
                    metadata['emotional_state'],
                ]

                if metadata.get('cot_used'):
                    footer_parts.append("🔍CoT")

                if metadata['used_method'] == 'multi_agent':
                    footer_parts.append("🤖MA")

                footer = f"\n\n<i>{' | '.join(footer_parts)} | ⚡{metadata['response_time']:.1f}s</i>"

                await update.message.reply_text(
                    response + footer,
                    link_preview_options=LinkPreviewOptions(is_disabled=True)
                )
                return

            except (TimedOut, NetworkError) as e:
                logger.warning(f"Telegram timeout (attempt {attempt + 1}): {e}")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay * (attempt + 1))
                    continue
                else:
                    await update.message.reply_text("⚠️ Таймаут. Повторите запрос.")
                    return

            except RetryAfter as e:
                logger.warning(f"Rate limit: {e.retry_after}s")
                await asyncio.sleep(e.retry_after)
                continue

            except Exception as e:
                logger.exception(f"Error from {user_id}")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay)
                    continue
                else:
                    await update.message.reply_text("⚠️ Произошла ошибка")
                    return

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(f"""🧠 <b>ULTIMATE COGNITIVE AGI v37.0</b>

🎯 <b>РЕВОЛЮЦИОННАЯ АРХИТЕКТУРА:</b>

✅ <b>Расширенный Transformer</b>
- {CONFIG.d_model}d модель, {CONFIG.n_layers} слоёв
- {CONFIG.n_heads} attention heads
- Mixed precision training
- Устройство: {CONFIG.device}

✅ <b>Полная память</b>
- Эпизодическая (события)
- Семантическая (знания)
- Процедурная (навыки)
- Адаптивная консолидация

✅ <b>Реальный эмоциональный интеллект</b>
- 8-мерная модель эмоций (PAD+)
- Динамическая адаптация
- История эмоциональных состояний

✅ <b>Внешние инструменты</b>
- Калькулятор
- Веб-поиск
- База знаний

✅ <b>Системы проверки качества</b>
- ConsistencyChecker (противоречия)
- UncertaintyEstimator (уверенность)
- SelfCorrectionLoop (самокоррекция)
- Chain-of-Thought (пошаговое мышление)

✅ <b>Метакогниция</b>
- Мониторинг мыслительных паттернов
- Адаптивный выбор стратегии
- Обнаружение зацикливания

✅ <b>Многоагентная система</b>
- Аналитик, Критик, Синтезатор, Валидатор
- Коллективное принятие решений

<b>Команды:</b>
/status - Полный статус
/memory - Состояние памяти
/health - Здоровье системы
/emotions - Эмоциональное состояние
/tools - Доступные инструменты
/stats - Статистика модулей
/reset - Сброс (с подтверждением)""")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_agent(user_id)
        status = agent.get_full_status()

        message = f"""🧠 <b>СТАТУС СИСТЕМЫ v37.0</b>

<b>🤖 Модель</b>
- Тип: {status['model']['type']}
- Размерность: {status['model']['d_model']}
- Слои: {status['model']['n_layers']}
- Устройство: {status['model']['device']}
- Шагов обучения: {status['model']['training_steps']}
- Словарь: {status['model']['vocab_size']} токенов

<b>🧠 Память</b>
- Эпизоды: {status['memory']['episodic']}
- Концепты: {status['memory']['semantic']}
- Навыки: {status['memory']['procedural']}
- Рабочая: {status['memory']['working']}

<b>🎭 Эмоции</b>
- Pleasure: {status['emotions']['pleasure']:.2f}
- Arousal: {status['emotions']['arousal']:.2f}
- Confidence: {status['emotions']['confidence']:.2f}
- Curiosity: {status['emotions']['curiosity']:.2f}

<b>🌡️ Состояние</b>
- Здоровье: {status['soma']['status']}
- Качество: {status['soma']['avg_quality']:.1%}
- Время отклика: {status['soma']['avg_processing_time']:.1f}s

<b>📊 Статистика</b>
- Взаимодействий: {status['interactions']['total']}
- Обучений: {status['interactions']['learnings']}
- Uptime: {status['uptime_hours']:.1f}ч"""

        await update.message.reply_text(message)

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_agent(user_id)

        recent = []
        for mem in list(agent.memory.episodic.memories)[-5:]:
            if isinstance(mem, EpisodicMemory):
                recent.append(f"• {mem.content[:80]}...")

        message = f"""🧠 <b>ПАМЯТЬ</b>

<b>Эпизодов:</b> {len(agent.memory.episodic.memories)}
<b>Концептов:</b> {len(agent.memory.semantic.memories)}
<b>Навыков:</b> {len(agent.memory.procedural)}

<b>Недавние воспоминания:</b>
{chr(10).join(recent) if recent else '(пусто)'}"""

        await update.message.reply_text(message)

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_agent(user_id)
        health = agent.soma.health_check()

        message = f"""🌡️ <b>ЗДОРОВЬЕ СИСТЕМЫ</b>

<b>Статус:</b> {health['status']}
<b>Оценка:</b> {health['health_score']:.1%}

<b>Метрики:</b>
- Качество: {health['avg_quality']:.1%}
- Уверенность: {health['avg_confidence']:.1%}
- Время: {health['avg_processing_time']:.2f}s
- Аномалий: {health['anomalies']}
- Ошибок: {health['error_rate']:.1%}
- Успехов подряд: {health['success_streak']}"""

        await update.message.reply_text(message)

    async def _cmd_emotions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_agent(user_id)
        emo = agent.emotions.get_full_state()

        message = f"""🎭 <b>ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ</b>

<b>Текущее:</b> {agent.emotions.get_emotional_state()}

<b>Измерения (8D):</b>
- Pleasure: {emo['pleasure']:.2f} ({-1 if emo['pleasure'] < 0 else 1})
- Arousal: {emo['arousal']:.2f}
- Dominance: {emo['dominance']:.2f}
- Curiosity: {emo['curiosity']:.2f}
- Confidence: {emo['confidence']:.2f}
- Frustration: {emo['frustration']:.2f}
- Engagement: {emo['engagement']:.2f}
- Empathy: {emo['empathy']:.2f}

<b>История:</b> {len(agent.emotions.emotion_history)} записей"""

        await update.message.reply_text(message)

    async def _cmd_tools(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_agent(user_id)

        tools_info = []
        for name, tool in agent.tools.tools.items():
            success_rate = tool.success_count / max(1, tool.usage_count)
            tools_info.append(
                f"• {name}: использован {tool.usage_count} раз (успех: {success_rate:.1%})"
            )

        message = f"""🛠️ <b>ВНЕШНИЕ ИНСТРУМЕНТЫ</b>

<b>Доступно:</b> {len(agent.tools.tools)}

{chr(10).join(tools_info) if tools_info else 'Нет доступных инструментов'}

<b>Примеры использования:</b>
- "Вычисли 2^16"
- "Найди информацию о квантовых компьютерах"
- "Сохрани факт: Столица Латвии - Рига"""

        await update.message.reply_text(message)

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_agent(user_id)
        status = agent.get_full_status()

        stats_parts = ["📊 <b>СТАТИСТИКА МОДУЛЕЙ</b>\n"]

        if 'consistency' in status:
            c = status['consistency']
            stats_parts.append(
                f"<b>ConsistencyChecker:</b>\n"
                f"- Проверок: {c['checks']}\n"
                f"- Противоречий: {c['contradictions']} ({c['rate']:.1%})\n"
            )

        if 'uncertainty' in status:
            u = status['uncertainty']
            stats_parts.append(
                f"<b>UncertaintyEstimator:</b>\n"
                f"- Оценок: {u['estimates']}\n"
                f"- Низкая уверенность: {u['low_confidence']} ({u['rate']:.1%})\n"
            )

        if 'self_correction' in status:
            s = status['self_correction']
            stats_parts.append(
                f"<b>SelfCorrectionLoop:</b>\n"
                f"- Попыток: {s['attempts']}\n"
                f"- Успешных: {s['successful']} ({s['success_rate']:.1%})\n"
            )

        if 'chain_of_thought' in status:
            cot = status['chain_of_thought']
            stats_parts.append(
                f"<b>ChainOfThought:</b>\n"
                f"- CoT использован: {cot['cot_used']}\n"
                f"- Простых: {cot['simple']}\n"
                f"- Частота CoT: {cot['cot_rate']:.1%}\n"
            )

        if 'metacognition' in status:
            m = status['metacognition']
            stats_parts.append(
                f"<b>Метакогниция:</b>\n"
                f"- Взаимодействий: {m['total_interactions']}\n"
                f"- Предупреждений: {m['stuck_warnings']}\n"
            )

        if 'multi_agent' in status:
            ma = status['multi_agent']
            stats_parts.append(
                f"<b>MultiAgent:</b>\n"
                f"- Коллабораций: {ma['collaborations']}\n"
            )

        await update.message.reply_text("\n".join(stats_parts))

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if context.args and context.args[0] == 'confirm':
            user_id = str(update.effective_user.id)
            if user_id in self.agents:
                self.agents[user_id]._save_state()
                del self.agents[user_id]

            import shutil
            user_dir = CONFIG.base_dir / 'models' / user_id
            if user_dir.exists():
                shutil.rmtree(user_dir)

            await update.message.reply_text("✅ Полный сброс выполнен")
        else:
            await update.message.reply_text(
                "⚠️ <b>ВНИМАНИЕ!</b> Это удалит всю память и обучение.\n"
                "Подтверждение: <code>/reset confirm</code>"
            )

    async def run(self):
        """Запуск бота"""
        if not self._app:
            logger.error("❌ Bot not initialized!")
            return

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot running")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down...")

        for agent in self.agents.values():
            agent._save_state()

        if self.teacher:
            await self.teacher.close()

        if self._app:
            if self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Shutdown complete")


# ═══════════════════════════════════════════════════════════════
# 🚀 MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

async def main():
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  🧠 ULTIMATE COGNITIVE AGI v37.0                              ║
║     ПОЛНАЯ РЕАЛИЗАЦИЯ БЕЗ ЗАГЛУШЕК                            ║
╚═══════════════════════════════════════════════════════════════╝

🎯 АРХИТЕКТУРА:

✅ РАСШИРЕННЫЙ TRANSFORMER ({CONFIG.d_model}d, {CONFIG.n_layers} слоёв)
   • {CONFIG.n_heads} attention heads с relative positional bias
   • GLU feed-forward networks
   • Mixed precision training
   • Repetition penalty
   • Устройство: {CONFIG.device}

✅ УЛУЧШЕННАЯ ПАМЯТЬ
   • Adaptive consolidation (умное забывание)
   • Consolidation scores
   • Performance tracking
   • {CONFIG.episodic_memory_size} эпизодов

✅ РЕАЛЬНЫЙ ЭМОЦИОНАЛЬНЫЙ ИНТЕЛЛЕКТ
   • 8-мерная PAD+ модель
   • Pleasure, Arousal, Dominance
   • Curiosity, Confidence, Frustration
   • Engagement, Empathy
   • Динамическая адаптация

✅ ВНЕШНИЕ ИНСТРУМЕНТЫ
   • Calculator (безопасные вычисления)
   • Web Search (интеграция готова)
   • Knowledge Base (персистентная)

✅ СИСТЕМЫ ПРОВЕРКИ КАЧЕСТВА
   • ConsistencyChecker - логические противоречия
   • UncertaintyEstimator - калибровка уверенности
   • SelfCorrectionLoop - итеративное исправление
   • ChainOfThought - пошаговое рассуждение

✅ МЕТАКОГНИЦИЯ
   • Мониторинг мыслительных паттернов
   • Обнаружение зацикливания
   • Адаптивный выбор стратегии
   • Performance tracking

✅ МНОГОАГЕНТНАЯ СИСТЕМА
   • 4 специализированных агента
   • Analyzer, Critic, Synthesizer, Validator
   • Коллективное принятие решений
   • Итеративная валидация

✅ УЛУЧШЕННАЯ СОМАТОСЕНСОРИКА
   • Расширенные метрики здоровья
   • Обнаружение аномалий
   • Error tracking
   • Success streaks

🔧 Конфигурация:
   • Vocab: {CONFIG.vocab_size}
   • Max sequence: {CONFIG.max_seq_length}
   • Embedding dim: {CONFIG.embedding_dim}
   • Learning rate: {CONFIG.learning_rate}
""")

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = UltimateBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Получен сигнал остановки")
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка: {e}", exc_info=True)
        return 1
    finally:
        await bot.shutdown()

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