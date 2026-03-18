#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ULTIMATE COGNITIVE AGI v36.0 — TRANSFORMER + ПОЛНОЕ СОЗНАНИЕ (FIXED)
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
from telegram.request import HTTPXRequest  # ✅ Для настройки таймаутов

load_dotenv()


# ═══════════════════════════════════════════════════════════════
# 🔧 КОНФИГУРАЦИЯ v36.0
# ═══════════════════════════════════════════════════════════════

@dataclass
class UltimateCognitiveConfig:
    """Полная конфигурация системы"""
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🕐 Таймауты (увеличены для стабильности)
    telegram_timeout: int = 60
    telegram_pool_timeout: int = 30
    lm_studio_timeout: int = 120
    lm_studio_connect_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0

    # 🧠 Transformer Model (PyTorch)
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = torch.cuda.is_available()
    auto_scale_model: bool = True

    # 📚 Память
    episodic_memory_size: int = 10000
    semantic_memory_size: int = 5000
    procedural_memory_size: int = 1000
    working_memory_size: int = 15
    memory_consolidation_threshold: float = 0.7
    forgetting_curve_factor: float = 0.1
    embedding_dim: int = 128

    # 🎯 Обучение
    learning_rate: float = 5e-5
    training_frequency: int = 5
    save_frequency: int = 50
    gradient_clip: float = 1.0
    warmup_steps: int = 100

    # 🔧 Самомодификация
    self_modification_enabled: bool = True
    module_creation_threshold: float = 0.6
    max_custom_modules: int = 50
    module_test_iterations: int = 5
    safe_execution_timeout: int = 30

    # 🌡️ Соматосенсорика
    internal_monitoring_enabled: bool = True
    health_check_interval: int = 60
    anomaly_detection_threshold: float = 2.0

    # 🎨 Метакогниция
    metacognition_enabled: bool = True
    planning_horizon: int = 5
    emotion_decay_rate: float = 0.05
    attention_window: int = 10

    # 💾 Пути
    version: str = "36.0-ULTIMATE-FIXED"
    base_dir: Path = Path(os.getenv('BASE_DIR', 'ultimate_agi_v36'))

    def __post_init__(self):
        subdirs = [
            'models', 'memory/episodic', 'memory/semantic', 'memory/procedural',
            'memory/working', 'tokenizer', 'checkpoints', 'backups',
            'logs', 'analytics', 'modules/custom', 'modules/tests',
            'health', 'emotions'
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
                self.n_layers = max(4, int(self.n_layers * scale_factor ** 0.25))
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
    logger = logging.getLogger('Ultimate_AGI_v36')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'ultimate_v36_{datetime.now():%Y%m%d}.log'
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
    """Умный токенизатор"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
            '<SEP>': 4, '<CLS>': 5
        }

        self.word_to_id: Dict[str, int] = self.special_tokens.copy()
        self.id_to_word: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)

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
                'next_id': self.next_id
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
            logger.info("✅ Tokenizer loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# 🧠 TRANSFORMER MODEL
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)

        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attention_output)

        ff_output = self.feed_forward(self.norm2(x))
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

    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_length: int = 50,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            eos_token_id: int = 3
    ) -> Tuple[torch.Tensor, float]:
        self.eval()
        generated = prompt_ids.clone()
        confidences = []

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)[:, -1, :] / temperature

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

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == eos_token_id:
                    break

        avg_confidence = np.mean(confidences) if confidences else 0.0
        return generated, avg_confidence


# ═══════════════════════════════════════════════════════════════
# 🧠 MEMORY SYSTEM
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

    def decay_importance(self, factor: float = 0.1):
        age_hours = (time.time() - self.timestamp) / 3600
        self.importance *= math.exp(-factor * age_hours / 24)

    def strengthen(self, amount: float = 0.1):
        self.importance = min(1.0, self.importance + amount)
        self.access_count += 1
        self.last_accessed = time.time()


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


@dataclass
class ProceduralMemory:
    skill_name: str
    description: str
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    code_reference: Optional[str] = None


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

    def consolidate(self, threshold: float = 0.7):
        before_count = len(self.memories)

        self.memories = [
            m for m in self.memories
            if (isinstance(m, EpisodicMemory) and m.importance >= threshold) or
               (isinstance(m, SemanticMemory) and m.confidence >= threshold)
        ]

        if len(self.memories) < before_count:
            self._needs_rebuild = True
            logger.info(f"🗑️ Консолидация: {before_count} → {len(self.memories)}")


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
            context_parts.append("=== Недавние взаимодействия ===")
            context_parts.append("\n".join(list(self.working_memory)[-3:]))

        episodes = self.recall_similar_episodes(query, top_k=max_episodes)
        if episodes:
            context_parts.append("\n=== Релевантные воспоминания ===")
            for episode, score in episodes:
                context_parts.append(f"[{score:.2f}] {episode.content[:200]}")

        return "\n".join(context_parts)

    def consolidate_memories(self):
        self.episodic.consolidate(CONFIG.memory_consolidation_threshold)
        self.semantic.consolidate(CONFIG.memory_consolidation_threshold)

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
# 🌡️ SOMATOSENSORY
# ═══════════════════════════════════════════════════════════════

class SomatosensorySystem:
    def __init__(self):
        self.state_history: deque = deque(maxlen=100)
        self.current_quality = 0.5
        self.current_confidence = 0.5
        self.emotional_valence = 0.0
        self.arousal = 0.0

        self.anomaly_count = 0
        self.health_checks = 0

    def update_state(
            self,
            quality: float,
            confidence: float,
            emotional_valence: Optional[float] = None,
            arousal: Optional[float] = None
    ):
        self.current_quality = self.current_quality * 0.7 + quality * 0.3
        self.current_confidence = self.current_confidence * 0.7 + confidence * 0.3

        if emotional_valence is not None:
            self.emotional_valence = self.emotional_valence * 0.7 + emotional_valence * 0.3

        if arousal is not None:
            self.arousal = arousal

        self.state_history.append({
            'quality': quality,
            'confidence': confidence,
            'timestamp': time.time()
        })

    def get_emotional_state(self) -> str:
        if self.arousal < 0.3:
            if self.emotional_valence > 0.3:
                return "спокойно-позитивное"
            elif self.emotional_valence < -0.3:
                return "спокойно-негативное"
            else:
                return "нейтральное"
        else:
            if self.emotional_valence > 0.3:
                return "заинтересованное"
            elif self.emotional_valence < -0.3:
                return "напряжённое"
            else:
                return "активное"

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

        health_score = avg_quality * 0.6 + avg_confidence * 0.4

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
            'anomalies': self.anomaly_count
        }


# ═══════════════════════════════════════════════════════════════
# 🔧 SELF-MODIFICATION
# ═══════════════════════════════════════════════════════════════

class SelfModificationEngine:
    def __init__(self, llm_client, modules_dir: Path):
        self.llm = llm_client
        self.modules_dir = modules_dir
        self.modules_dir.mkdir(parents=True, exist_ok=True)

        self.custom_modules: Dict[str, Any] = {}
        self.loaded_modules: Dict[str, Any] = {}

        self.creation_attempts = 0
        self.successful_creations = 0

    async def create_module(
            self,
            task_description: str,
            requirements: List[str]
    ) -> Optional[str]:
        self.creation_attempts += 1

        logger.info(f"🔧 Creating module: {task_description}")

        prompt = f"""Создай Python-модуль для задачи: {task_description}

Требования:
{chr(10).join(f"- {r}" for r in requirements)}

Модуль должен содержать функцию execute(). Только код Python, без markdown."""

        code = await self.llm.generate(prompt, temperature=0.3)

        if not code:
            return None

        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        code = code.strip()

        module_name = f"custom_{int(time.time())}"
        module_path = self.modules_dir / f"{module_name}.py"

        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(code)

        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.loaded_modules[module_name] = module
                self.successful_creations += 1
                logger.info(f"✅ Module created: {module_name}")
                return module_name
        except Exception as e:
            logger.error(f"Failed to load module: {e}")
            return None


# ═══════════════════════════════════════════════════════════════
# 🔗 TEACHER LLM
# ═══════════════════════════════════════════════════════════════

class TeacherLLM:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self.response_cache: Dict[str, Tuple[str, float]] = {}

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

        cache_key = hashlib.md5(f"{system_prompt}_{prompt}".encode()).hexdigest()
        if cache_key in self.response_cache:
            cached, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < 1800:
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
# 🎓 TRAINER
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

        return loss.item()


# ═══════════════════════════════════════════════════════════════
# 🤖 ULTIMATE AGENT
# ═══════════════════════════════════════════════════════════════

class UltimateCognitiveAgent:
    def __init__(self, user_id: str, teacher: TeacherLLM):
        self.user_id = user_id
        self.teacher = teacher

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

        self.memory = CognitiveMemorySystem(CONFIG.embedding_dim, self._simple_embed)

        self.soma = SomatosensorySystem()

        self.self_mod = SelfModificationEngine(
            teacher,
            CONFIG.base_dir / 'modules' / 'custom' / user_id
        )

        self.total_interactions = 0
        self.successful_learnings = 0
        self.birth_time = time.time()

        self.user_dir = CONFIG.base_dir / 'models' / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self._load_state()

        logger.info(f"🚀 Ultimate Agent v36 created for {user_id}")

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
        start_time = time.time()
        self.total_interactions += 1

        memory_context = self.memory.get_rich_context(user_input)

        use_teacher = (
                self.soma.current_quality < 0.6 or
                len(user_input) > 150 or
                np.random.random() < 0.3
        )

        response = ""
        confidence = 0.0

        if not use_teacher:
            full_prompt = f"{memory_context}\nUser: {user_input}\nAssistant:"
            input_ids = self.tokenizer.encode(full_prompt, max_length=CONFIG.max_seq_length // 2)

            tensor = torch.tensor([input_ids], dtype=torch.long, device=CONFIG.device)

            generated, conf = self.model.generate(tensor, max_length=100, temperature=0.8)

            response = self.tokenizer.decode(generated[0].cpu().tolist(), skip_special=True)
            confidence = conf

        else:
            system_prompt = """Ты — продвинутая когнитивная AGI с памятью и эмоциями.
Отвечай естественно, содержательно, 2-5 предложений."""

            full_prompt = f"{memory_context}\nUser: {user_input}\nAssistant:"

            response = await self.teacher.generate(
                full_prompt,
                temperature=0.75,
                max_tokens=2000,
                system_prompt=system_prompt
            )

            confidence = 1.0 if response else 0.5

            if response:
                loss = await self.trainer.train_step(user_input, response)
                self.successful_learnings += 1

        quality = confidence
        if len(response) < 10:
            quality *= 0.5

        emotional_valence = (quality - 0.5) * 2
        arousal = min(1.0, len(user_input.split()) / 20)

        self.memory.add_episode(
            content=f"User: {user_input}\nAI: {response}",
            importance=quality,
            emotional_valence=emotional_valence,
            arousal=arousal,
            context={'quality': quality, 'confidence': confidence}
        )

        self.soma.update_state(quality, confidence, emotional_valence, arousal)

        is_anomaly = self.soma.detect_anomaly()

        if CONFIG.self_modification_enabled and quality < 0.4:
            if np.random.random() < 0.1:
                asyncio.create_task(
                    self.self_mod.create_module(
                        user_input,
                        ["Улучшить обработку подобных запросов"]
                    )
                )

        if self.total_interactions % CONFIG.save_frequency == 0:
            self._save_state()
            self.memory.consolidate_memories()

        metadata = {
            'quality': quality,
            'confidence': confidence,
            'emotional_state': self.soma.get_emotional_state(),
            'is_anomaly': is_anomaly,
            'response_time': time.time() - start_time,
            'memory_count': len(self.memory.episodic.memories),
            'health': self.soma.health_check()
        }

        logger.info(
            f"✅ [{self.user_id}] Q={quality:.2%} | "
            f"Emo={metadata['emotional_state']} | "
            f"T={metadata['response_time']:.1f}s"
        )

        return response, metadata

    def get_full_status(self) -> Dict:
        uptime = time.time() - self.birth_time

        return {
            'user_id': self.user_id,
            'version': CONFIG.version,
            'uptime_hours': uptime / 3600,

            'model': {
                'type': 'Transformer',
                'd_model': CONFIG.d_model,
                'n_layers': CONFIG.n_layers,
                'device': CONFIG.device,
                'training_steps': self.trainer.training_steps
            },

            'memory': {
                'episodic': len(self.memory.episodic.memories),
                'semantic': len(self.memory.semantic.memories),
                'procedural': len(self.memory.procedural),
                'working': len(self.memory.working_memory)
            },

            'self_modification': {
                'modules': len(self.self_mod.loaded_modules),
                'attempts': self.self_mod.creation_attempts,
                'success_rate': (
                        self.self_mod.successful_creations /
                        max(1, self.self_mod.creation_attempts)
                )
            },

            'soma': self.soma.health_check(),

            'interactions': {
                'total': self.total_interactions,
                'learnings': self.successful_learnings,
                'learning_rate': (
                        self.successful_learnings /
                        max(1, self.total_interactions)
                )
            }
        }


# ═══════════════════════════════════════════════════════════════
# 🤖 TELEGRAM BOT (FIXED)
# ═══════════════════════════════════════════════════════════════

class UltimateBot:
    def __init__(self):
        self.teacher: Optional[TeacherLLM] = None
        self.agents: Dict[str, UltimateCognitiveAgent] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        """✅ FIXED: Правильная настройка таймаутов"""
        self.teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.teacher.connect()

        defaults = Defaults(parse_mode='HTML')

        # ✅ Правильная настройка таймаутов для python-telegram-bot v20+
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
            .request(request)  # ✅ Устанавливаем таймауты через request
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
            ('reset', self._cmd_reset),
        ]:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Ultimate Bot v36 initialized")

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

                footer = (
                    f"\n\n<i>🧠 Q:{metadata['quality']:.0%} | "
                    f"{metadata['emotional_state']} | "
                    f"⚡{metadata['response_time']:.1f}s</i>"
                )

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
        await update.message.reply_text(f"""🧠 <b>ULTIMATE COGNITIVE AGI v36.0</b>

Революционная гибридная архитектура:

✅ <b>Transformer нейросеть</b>
- Multi-head attention
- {CONFIG.d_model}d модель, {CONFIG.n_layers} слоёв
- Mixed precision training

✅ <b>Полная память</b>
- Эпизодическая + Семантическая + Процедурная
- Векторный семантический поиск
- Автоматическая консолидация

✅ <b>Когнитивная архитектура</b>
- Метакогниция
- Эмоциональный интеллект
- Самомодификация

Команды: /status /memory /health""")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_agent(user_id)
        status = agent.get_full_status()

        message = f"""🧠 <b>СТАТУС v36.0</b>

<b>🤖 Модель</b>
- Тип: {status['model']['type']}
- Размерность: {status['model']['d_model']}
- Слои: {status['model']['n_layers']}
- Устройство: {status['model']['device']}
- Шагов обучения: {status['model']['training_steps']}

<b>🧠 Память</b>
- Эпизоды: {status['memory']['episodic']}
- Концепты: {status['memory']['semantic']}
- Навыки: {status['memory']['procedural']}

<b>🔧 Самомодификация</b>
- Модулей: {status['self_modification']['modules']}
- Успех: {status['self_modification']['success_rate']:.1%}

<b>🌡️ Состояние</b>
- Здоровье: {status['soma']['status']}
- Эмоции: {status['soma']['emotional_state']}

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
- Эмоции: {health['emotional_state']}
- Аномалий: {health['anomalies']}"""

        await update.message.reply_text(message)

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
                "⚠️ <b>ВНИМАНИЕ!</b> Это удалит всю память.\n"
                "Подтверждение: <code>/reset confirm</code>"
            )

    async def run(self):
        """✅ FIXED: Правильная остановка"""
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
        """✅ FIXED: Проверка перед остановкой"""
        logger.info("🛑 Shutting down...")

        for agent in self.agents.values():
            agent._save_state()

        if self.teacher:
            await self.teacher.close()

        if self._app:
            # ✅ Проверяем, запущен ли updater перед остановкой
            if self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Shutdown complete")


# ═══════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════

async def main():
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  🧠 ULTIMATE COGNITIVE AGI v36.0                              ║
║     TRANSFORMER + ПОЛНОЕ СОЗНАНИЕ                             ║
╚═══════════════════════════════════════════════════════════════╝
🔥 РЕВОЛЮЦИОННАЯ АРХИТЕКТУРА:
✅ TRANSFORMER НЕЙРОСЕТЬ (PyTorch)
   • {CONFIG.d_model}d модель, {CONFIG.n_layers} слоёв
   • Multi-head attention ({CONFIG.n_heads} heads)
   • Mixed precision training
   • Устройство: {CONFIG.device}
✅ ПОЛНОЦЕННАЯ ПАМЯТЬ
   • Эпизодическая (события с эмоциями)
   • Семантическая (концепты и связи)
   • Процедурная (навыки и код)
   • Векторный семантический поиск
✅ КОГНИТИВНАЯ АРХИТЕКТУРА
   • Метакогниция (мысли о мыслях)
   • Эмоциональный интеллект
   • Планирование и целеполагание
   • Самомодификация
✅ НАДЁЖНОСТЬ
   • Retry-логика для сети
   • Увеличенные таймауты
   • Graceful error handling
   • Connection pooling
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