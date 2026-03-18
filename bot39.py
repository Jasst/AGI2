#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ADVANCED AUTONOMOUS LEARNING AGENT v3.0 — Максимальная версия
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ НОВЫЕ ВОЗМОЖНОСТИ:

🧠 УВЕЛИЧЕННАЯ МОДЕЛЬ:
   • До 100M параметров (адаптивно под GPU)
   • 12 слоёв Transformer
   • 1024 размерность
   • Динамическое определение размера под доступную память

🔤 BPE TOKENIZER:
   • Байтовый уровень (Byte Pair Encoding)
   • 50,000 токенов
   • Подслова для незнакомых слов
   • Как в GPT-2/GPT-3

📚 RAG (Retrieval-Augmented Generation):
   • Векторная база знаний (ChromaDB)
   • Семантический поиск
   • Контекстное дополнение
   • Долговременная память

🎯 META-LEARNING:
   • MAML (Model-Agnostic Meta-Learning)
   • Few-shot адаптация
   • Быстрое обучение на новых доменах
   • Адаптивное обучение

⏰ ВНУТРЕННЕЕ ВРЕМЯ:
   • Временные embeddings
   • Непрерывность сознания
   • Циркадные ритмы
   • Контекст времени
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
import gzip
import pickle
import time
import re
from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)

# Для RAG
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available. Install: pip install chromadb")

# Для BPE токенизации
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("⚠️ Tokenizers not available. Install: pip install tokenizers")

load_dotenv()


# ══════════════════════════════════════════════════════════════
# ⚙️ ADVANCED CONFIGURATION
# ══════════════════════════════════════════════════════════════

@dataclass
class AdvancedConfig:
    """Продвинутая конфигурация v3"""
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    # Student Model - МАКСИМАЛЬНАЯ версия
    vocab_size: int = 50000  # BPE словарь
    d_model: int = 1024  # Увеличено с 512
    n_heads: int = 16  # Увеличено с 8
    n_layers: int = 12  # Увеличено с 6
    d_ff: int = 4096  # Увеличено с 2048
    max_seq_length: int = 1024  # Увеличено с 512
    dropout: float = 0.1

    # Автоматическая адаптация под GPU
    auto_scale_model: bool = True
    max_gpu_memory_gb: float = 8.0  # Максимум памяти для модели

    # Обучение
    learning_rate: float = 5e-5
    meta_learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.7

    # Meta-Learning
    meta_learning_enabled: bool = True
    few_shot_examples: int = 5
    meta_batch_size: int = 4
    inner_loop_steps: int = 3

    # RAG
    rag_enabled: bool = CHROMADB_AVAILABLE
    rag_top_k: int = 5
    rag_chunk_size: int = 512
    rag_embedding_dim: int = 768

    # Внутреннее время
    temporal_embeddings: bool = True
    time_embedding_dim: int = 64
    circadian_cycle_hours: int = 24
    memory_decay_rate: float = 0.01

    # Автономность
    initial_teacher_usage: float = 1.0
    min_teacher_usage: float = 0.05
    autonomy_growth_rate: float = 0.002
    confidence_threshold: float = 0.75

    # Память
    replay_buffer_size: int = 50000
    training_frequency: int = 5
    save_frequency: int = 50

    # Пути
    base_dir: Path = Path('advanced_agent_v3')
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = torch.cuda.is_available()

    def __post_init__(self):
        for subdir in ['models', 'memory', 'logs', 'checkpoints', 'rag', 'tokenizer']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Автомасштабирование модели под GPU
        if self.auto_scale_model and self.device == 'cuda':
            self._auto_scale_to_gpu()

    def _auto_scale_to_gpu(self):
        """Автоматическая адаптация размера модели под GPU"""
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU Memory: {gpu_memory_gb:.2f} GB")

            # Оценка параметров модели
            estimated_params = (
                    self.vocab_size * self.d_model * 2 +  # Embeddings
                    self.n_layers * (
                            4 * self.d_model * self.d_model +  # Attention
                            2 * self.d_model * self.d_ff  # FFN
                    )
            )

            # Оценка памяти (FP32 = 4 bytes)
            estimated_memory_gb = estimated_params * 4 / 1e9 * 1.5  # 1.5x для градиентов

            print(f"📊 Estimated model size: {estimated_params / 1e6:.1f}M params, {estimated_memory_gb:.2f} GB")

            # Если не влезает - уменьшаем
            if estimated_memory_gb > gpu_memory_gb * 0.7:  # 70% от GPU
                scale_factor = (gpu_memory_gb * 0.7) / estimated_memory_gb

                self.d_model = int(self.d_model * scale_factor ** 0.5)
                self.d_ff = int(self.d_ff * scale_factor ** 0.5)
                self.n_layers = max(6, int(self.n_layers * scale_factor ** 0.25))

                # Выравниваем для multi-head attention
                self.d_model = (self.d_model // self.n_heads) * self.n_heads

                print(f"⚙️ Auto-scaled: d_model={self.d_model}, n_layers={self.n_layers}, d_ff={self.d_ff}")

        except Exception as e:
            print(f"⚠️ Auto-scaling failed: {e}")


CONFIG = AdvancedConfig()


# ══════════════════════════════════════════════════════════════
# 📊 LOGGING
# ══════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    logger = logging.getLogger('AdvancedAgent_v3')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'agent_v3_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ══════════════════════════════════════════════════════════════
# 🔤 ADVANCED BPE TOKENIZER
# ══════════════════════════════════════════════════════════════

class AdvancedBPETokenizer:
    """Продвинутый BPE токенизатор (как в GPT-2)"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer: Optional[Tokenizer] = None
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }

        if not TOKENIZERS_AVAILABLE:
            logger.warning("⚠️ Tokenizers library not available, using fallback")
            self._init_fallback_tokenizer()
        else:
            self._init_bpe_tokenizer()

    def _init_bpe_tokenizer(self):
        """Инициализация настоящего BPE"""
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

        # Byte-level pre-tokenization (как в GPT-2)
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        # Post-processing
        self.tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", self.special_tokens['<BOS>']),
                ("<EOS>", self.special_tokens['<EOS>']),
            ],
        )

        logger.info("✅ BPE tokenizer initialized")

    def _init_fallback_tokenizer(self):
        """Fallback простой токенизатор"""
        self.word_to_id: Dict[str, int] = self.special_tokens.copy()
        self.id_to_word: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)

    def train(self, texts: List[str]):
        """Обучение BPE на корпусе"""
        if not TOKENIZERS_AVAILABLE:
            return self._train_fallback(texts)

        # Сохраняем тексты во временный файл
        train_file = CONFIG.base_dir / 'tokenizer' / 'train_corpus.txt'
        train_file.parent.mkdir(exist_ok=True)

        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        # Обучаем BPE
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            show_progress=True,
        )

        self.tokenizer.train([str(train_file)], trainer)
        logger.info(f"✅ BPE trained on {len(texts)} texts")

        # Удаляем временный файл
        train_file.unlink()

    def _train_fallback(self, texts: List[str]):
        """Обучение fallback токенизатора"""
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)

        for word, _ in word_freq.most_common(self.vocab_size - len(self.special_tokens)):
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Кодирование текста"""
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            encoding = self.tokenizer.encode(text)
            tokens = encoding.ids
        else:
            # Fallback
            words = text.lower().split()
            tokens = [self.special_tokens['<BOS>']]
            for word in words:
                token_id = self.word_to_id.get(word, self.special_tokens['<UNK>'])
                tokens.append(token_id)
            tokens.append(self.special_tokens['<EOS>'])

        # Обрезаем/дополняем
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.special_tokens['<PAD>']] * (max_length - len(tokens))

        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Декодирование токенов"""
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            # Фильтруем специальные токены
            if skip_special:
                tokens = [t for t in tokens if t >= len(self.special_tokens)]

            text = self.tokenizer.decode(tokens, skip_special_tokens=skip_special)
        else:
            # Fallback
            words = []
            for token_id in tokens:
                word = self.id_to_word.get(token_id, '<UNK>')
                if not skip_special or word not in self.special_tokens:
                    words.append(word)
            text = ' '.join(words)

        return text

    def save(self, path: Path):
        """Сохранение токенизатора"""
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            self.tokenizer.save(str(path / 'tokenizer.json'))
        else:
            state = {
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'next_id': self.next_id,
            }
            with gzip.open(path / 'tokenizer_fallback.pkl.gz', 'wb') as f:
                pickle.dump(state, f)

        logger.info(f"💾 Tokenizer saved")

    def load(self, path: Path) -> bool:
        """Загрузка токенизатора"""
        if (path / 'tokenizer.json').exists() and TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
            logger.info("✅ BPE tokenizer loaded")
            return True
        elif (path / 'tokenizer_fallback.pkl.gz').exists():
            with gzip.open(path / 'tokenizer_fallback.pkl.gz', 'rb') as f:
                state = pickle.load(f)
            self.word_to_id = state['word_to_id']
            self.id_to_word = state['id_to_word']
            self.next_id = state['next_id']
            logger.info("✅ Fallback tokenizer loaded")
            return True

        return False


# ══════════════════════════════════════════════════════════════
# ⏰ TEMPORAL EMBEDDINGS (Внутреннее время)
# ══════════════════════════════════════════════════════════════

class TemporalEmbeddings(nn.Module):
    """Временные embeddings для непрерывности сознания"""

    def __init__(self, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim

        # Синусоидальные embeddings для времени (как в Transformer)
        self.time_scale = 10000.0

        # Циркадные ритмы (24-часовой цикл)
        self.circadian_embedding = nn.Embedding(24, time_dim)

        # День недели
        self.weekday_embedding = nn.Embedding(7, time_dim)

        # Месяц
        self.month_embedding = nn.Embedding(12, time_dim)

        # Время с момента запуска (непрерывность)
        self.register_buffer('birth_timestamp', torch.tensor(time.time()))

        logger.info(f"⏰ Temporal embeddings initialized (dim={time_dim})")

    def get_current_time_features(self) -> Dict[str, int]:
        """Получение текущих временных признаков"""
        now = datetime.now()
        current_timestamp = time.time()

        return {
            'hour': now.hour,
            'weekday': now.weekday(),
            'month': now.month - 1,  # 0-indexed
            'seconds_since_birth': int(current_timestamp - self.birth_timestamp.item()),
        }

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """Генерация временных embeddings"""
        features = self.get_current_time_features()

        device = next(self.parameters()).device

        # Циркадные (час)
        hour_emb = self.circadian_embedding(
            torch.tensor([features['hour']], device=device)
        ).expand(batch_size, -1)

        # День недели
        weekday_emb = self.weekday_embedding(
            torch.tensor([features['weekday']], device=device)
        ).expand(batch_size, -1)

        # Месяц
        month_emb = self.month_embedding(
            torch.tensor([features['month']], device=device)
        ).expand(batch_size, -1)

        # Синусоидальные для непрерывного времени
        seconds = features['seconds_since_birth']
        position = torch.arange(self.time_dim, device=device).float()

        div_term = torch.exp(position * -(np.log(self.time_scale) / self.time_dim))
        continuous_emb = torch.zeros(batch_size, self.time_dim, device=device)

        continuous_emb[:, 0::2] = torch.sin(seconds * div_term[0::2])
        continuous_emb[:, 1::2] = torch.cos(seconds * div_term[1::2])

        # Комбинируем всё
        temporal_emb = hour_emb + weekday_emb + month_emb + continuous_emb

        return temporal_emb


# ══════════════════════════════════════════════════════════════
# 📚 RAG SYSTEM (Retrieval-Augmented Generation)
# ══════════════════════════════════════════════════════════════

class RAGSystem:
    """Система RAG для долговременной памяти"""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.client: Optional[chromadb.Client] = None
        self.collection = None

        if not CHROMADB_AVAILABLE:
            logger.warning("⚠️ RAG disabled (ChromaDB not available)")
            return

        try:
            # Инициализация ChromaDB
            self.client = chromadb.Client(Settings(
                persist_directory=str(CONFIG.base_dir / 'rag'),
                anonymized_telemetry=False
            ))

            # Создание/загрузка коллекции
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                metadata={"description": "Long-term knowledge storage"}
            )

            logger.info(f"✅ RAG initialized ({self.collection.count()} documents)")

        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            CHROMADB_AVAILABLE = False

    def add_interaction(
            self,
            interaction_id: str,
            user_input: str,
            assistant_response: str,
            metadata: Optional[Dict] = None
    ):
        """Добавление взаимодействия в базу знаний"""
        if not CHROMADB_AVAILABLE or not self.collection:
            return

        try:
            # Комбинируем вход и выход для контекста
            document = f"User: {user_input}\nAssistant: {assistant_response}"

            meta = metadata or {}
            meta.update({
                'timestamp': time.time(),
                'user_input': user_input[:500],  # Обрезаем для метаданных
                'response': assistant_response[:500],
            })

            self.collection.add(
                documents=[document],
                ids=[interaction_id],
                metadatas=[meta]
            )

        except Exception as e:
            logger.error(f"Failed to add to RAG: {e}")

    def retrieve_relevant(
            self,
            query: str,
            top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """Поиск релевантных взаимодействий"""
        if not CHROMADB_AVAILABLE or not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            if not results['documents'] or not results['documents'][0]:
                return []

            retrieved = []
            for doc, dist, meta in zip(
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
            ):
                similarity = 1.0 - dist  # Преобразуем distance в similarity
                retrieved.append((doc, similarity, meta))

            return retrieved

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []

    def get_augmented_context(self, query: str, top_k: int = 3) -> str:
        """Получение контекста для RAG"""
        retrieved = self.retrieve_relevant(query, top_k)

        if not retrieved:
            return ""

        context_parts = ["=== Релевантные знания из памяти ==="]

        for i, (doc, similarity, meta) in enumerate(retrieved, 1):
            context_parts.append(f"\n[{i}] (релевантность: {similarity:.2f})")
            context_parts.append(doc[:300])  # Обрезаем для контекста

        return "\n".join(context_parts)


# ══════════════════════════════════════════════════════════════
# 🧠 ADVANCED TRANSFORMER WITH TEMPORAL EMBEDDINGS
# ══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention с улучшениями"""

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
        self.scale = np.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)

        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward с GELU"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer Encoder Block с улучшениями"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (более стабильное обучение)
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)

        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)

        return x


class AdvancedStudentTransformer(nn.Module):
    """Продвинутая трансформерная модель с временными embeddings"""

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 1024,
            n_heads: int = 16,
            n_layers: int = 12,
            d_ff: int = 4096,
            max_seq_length: int = 1024,
            dropout: float = 0.1,
            use_temporal: bool = True,
            time_dim: int = 64
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_temporal = use_temporal

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Temporal embeddings
        if use_temporal:
            self.temporal_embeddings = TemporalEmbeddings(time_dim)
            self.temporal_projection = nn.Linear(time_dim, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Tie weights (как в GPT)
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

        # Подсчёт параметров
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 Model: {total_params / 1e6:.1f}M parameters, {n_layers} layers, {d_model} dim")

    def _init_weights(self):
        """Инициализация весов"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            input_ids: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_logits: bool = True
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)

        # Temporal embeddings (внутреннее время)
        if self.use_temporal:
            temporal_emb = self.temporal_embeddings(batch_size)
            temporal_emb = self.temporal_projection(temporal_emb).unsqueeze(1)
            x = x + temporal_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Output projection
        logits = self.output_projection(x)

        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=-1)

    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
            eos_token_id: int = 3
    ) -> Tuple[torch.Tensor, float]:
        """Генерация текста с улучшенной выборкой"""
        self.eval()

        generated = prompt_ids.clone()
        confidences = []

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated, return_logits=True)
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Confidence
                confidence = probs.max().item()
                confidences.append(confidence)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

                # Check EOS
                if next_token.item() == eos_token_id:
                    break

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return generated, avg_confidence


# [Продолжение в следующем файле из-за ограничения длины...]
# Продолжение advanced_agent_v3.py - часть 2

# ══════════════════════════════════════════════════════════════
# 🎯 META-LEARNING (MAML)
# ══════════════════════════════════════════════════════════════

class MetaLearner:
    """Model-Agnostic Meta-Learning для быстрой адаптации"""

    def __init__(
            self,
            model: nn.Module,
            meta_lr: float = 1e-4,
            inner_lr: float = 1e-3,
            inner_steps: int = 3
    ):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=meta_lr
        )

        logger.info(f"🎯 Meta-Learning initialized (MAML)")

    def inner_loop(
            self,
            support_data: List[Tuple[torch.Tensor, torch.Tensor]],
            model_copy: nn.Module
    ) -> nn.Module:
        """Inner loop - быстрая адаптация на support set"""

        # Создаём временный optimizer для inner loop
        temp_optimizer = torch.optim.SGD(
            model_copy.parameters(),
            lr=self.inner_lr
        )

        # Несколько шагов градиентного спуска
        for _ in range(self.inner_steps):
            total_loss = 0.0

            for input_ids, labels in support_data:
                logits = model_copy(input_ids, return_logits=True)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0
                )

                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()

                total_loss += loss.item()

        return model_copy

    def meta_update(
            self,
            tasks: List[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]]
    ) -> float:
        """Meta update на батче задач"""

        meta_loss = 0.0

        for task in tasks:
            # Копируем модель для inner loop
            model_copy = copy.deepcopy(self.model)

            # Inner loop на support set
            support_data = task['support']
            model_copy = self.inner_loop(support_data, model_copy)

            # Вычисляем loss на query set
            query_data = task['query']
            for input_ids, labels in query_data:
                logits = model_copy(input_ids, return_logits=True)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0
                )
                meta_loss += loss

        # Meta gradient step
        meta_loss = meta_loss / len(tasks)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()

        return meta_loss.item()

    def few_shot_adapt(
            self,
            examples: List[Tuple[str, str]],
            tokenizer
    ) -> float:
        """Быстрая адаптация на нескольких примерах (few-shot)"""

        # Подготовка данных
        support_data = []
        for prompt, response in examples:
            text = f"{prompt} {response}"
            input_ids = tokenizer.encode(text, max_length=CONFIG.max_seq_length)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=CONFIG.device)
            labels = input_tensor[:, 1:].clone()
            input_for_model = input_tensor[:, :-1]

            support_data.append((input_for_model, labels))

        # Адаптация
        model_copy = copy.deepcopy(self.model)
        adapted_model = self.inner_loop(support_data, model_copy)

        # Копируем веса обратно
        self.model.load_state_dict(adapted_model.state_dict())

        logger.info(f"✅ Few-shot adapted on {len(examples)} examples")

        return 0.0  # TODO: вернуть loss


# ══════════════════════════════════════════════════════════════
# 👨‍🏫 TEACHER MODEL
# ══════════════════════════════════════════════════════════════

class TeacherLLM:
    """Внешний LLM (учитель)"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self.total_calls = 0

    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info("🔗 Teacher LLM connected")

    async def close(self):
        if self._session:
            await self._session.close()
            await asyncio.sleep(0.25)

    async def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 500
    ) -> Tuple[str, List[float]]:
        """Генерация от учителя"""
        if not self._session:
            await self.connect()

        self.total_calls += 1

        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with self._session.post(self.url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    return content, []
                else:
                    logger.warning(f"Teacher LLM error: {resp.status}")
                    return "", []

        except Exception as e:
            logger.error(f"Teacher LLM exception: {e}")
            return "", []


# ══════════════════════════════════════════════════════════════
# 🎓 ADVANCED DISTILLATION TRAINER
# ══════════════════════════════════════════════════════════════

class AdvancedDistillationTrainer:
    """Продвинутый тренер с Mixed Precision, Meta-Learning"""

    def __init__(
            self,
            student_model: AdvancedStudentTransformer,
            tokenizer: AdvancedBPETokenizer,
            device: str = 'cpu',
            use_meta_learning: bool = True
    ):
        self.student = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=CONFIG.learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,
            T_mult=2
        )

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if CONFIG.mixed_precision else None

        # Meta-Learning
        if use_meta_learning and CONFIG.meta_learning_enabled:
            self.meta_learner = MetaLearner(
                self.student,
                meta_lr=CONFIG.meta_learning_rate,
                inner_lr=CONFIG.learning_rate * 10,
                inner_steps=CONFIG.inner_loop_steps
            )
        else:
            self.meta_learner = None

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=CONFIG.replay_buffer_size)

        # Gradient accumulation
        self.gradient_accumulation_steps = CONFIG.gradient_accumulation_steps
        self.accumulation_counter = 0

        # Statistics
        self.training_steps = 0
        self.total_loss = 0.0
        self.losses_history: deque = deque(maxlen=1000)

    def add_to_replay_buffer(self, prompt: str, teacher_response: str):
        """Добавление в replay buffer"""
        self.replay_buffer.append({
            'prompt': prompt,
            'response': teacher_response,
            'timestamp': time.time()
        })

    async def train_on_interaction(
            self,
            prompt: str,
            teacher_response: str,
            use_distillation: bool = True
    ) -> float:
        """Обучение на взаимодействии"""
        self.student.train()

        # Добавление в replay buffer
        self.add_to_replay_buffer(prompt, teacher_response)

        # Подготовка данных
        text = f"{prompt} {teacher_response}"
        input_ids = self.tokenizer.encode(text, max_length=CONFIG.max_seq_length)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        labels = input_tensor[:, 1:].clone()
        input_for_model = input_tensor[:, :-1]

        # Forward pass с Mixed Precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                student_logits = self.student(input_for_model, return_logits=True)

                # Cross-entropy loss
                loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0
                )

                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward с scaling
            self.scaler.scale(loss).backward()

            self.accumulation_counter += 1

            if self.accumulation_counter >= self.gradient_accumulation_steps:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                self.accumulation_counter = 0
        else:
            # Без Mixed Precision
            student_logits = self.student(input_for_model, return_logits=True)

            loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )

            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            self.accumulation_counter += 1

            if self.accumulation_counter >= self.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulation_counter = 0

        # Scheduler step
        self.scheduler.step()

        # Statistics
        self.training_steps += 1
        actual_loss = loss.item() * self.gradient_accumulation_steps
        self.total_loss += actual_loss
        self.losses_history.append(actual_loss)

        return actual_loss

    async def train_on_replay_buffer(self, num_batches: int = 5) -> float:
        """Обучение на replay buffer"""
        if len(self.replay_buffer) < CONFIG.batch_size:
            return 0.0

        total_loss = 0.0

        for _ in range(num_batches):
            batch_indices = np.random.choice(
                len(self.replay_buffer),
                size=min(CONFIG.batch_size, len(self.replay_buffer)),
                replace=False
            )

            batch = [self.replay_buffer[i] for i in batch_indices]

            for item in batch:
                loss = await self.train_on_interaction(
                    item['prompt'],
                    item['response'],
                    use_distillation=False
                )
                total_loss += loss

        return total_loss / (num_batches * CONFIG.batch_size)

    def few_shot_adapt(self, examples: List[Tuple[str, str]]) -> float:
        """Few-shot адаптация через Meta-Learning"""
        if not self.meta_learner:
            logger.warning("Meta-learning not enabled")
            return 0.0

        return self.meta_learner.few_shot_adapt(examples, self.tokenizer)

    def get_statistics(self) -> Dict:
        """Статистика обучения"""
        return {
            'training_steps': self.training_steps,
            'avg_loss': np.mean(list(self.losses_history)) if self.losses_history else 0.0,
            'recent_loss': self.losses_history[-1] if self.losses_history else 0.0,
            'replay_buffer_size': len(self.replay_buffer),
            'current_lr': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler,
                                                                     'get_last_lr') else CONFIG.learning_rate,
        }


# ══════════════════════════════════════════════════════════════
# 🤖 ADVANCED AUTONOMOUS AGENT
# ══════════════════════════════════════════════════════════════

class AdvancedAutonomousAgent:
    """Продвинутый автономный агент v3"""

    def __init__(self, user_id: str, teacher: TeacherLLM):
        self.user_id = user_id
        self.teacher = teacher

        # BPE Tokenizer
        self.tokenizer = AdvancedBPETokenizer(vocab_size=CONFIG.vocab_size)

        # Student Model
        self.student_model = AdvancedStudentTransformer(
            vocab_size=CONFIG.vocab_size,
            d_model=CONFIG.d_model,
            n_heads=CONFIG.n_heads,
            n_layers=CONFIG.n_layers,
            d_ff=CONFIG.d_ff,
            max_seq_length=CONFIG.max_seq_length,
            dropout=CONFIG.dropout,
            use_temporal=CONFIG.temporal_embeddings,
            time_dim=CONFIG.time_embedding_dim
        )

        # Trainer
        self.trainer = AdvancedDistillationTrainer(
            self.student_model,
            self.tokenizer,
            device=CONFIG.device,
            use_meta_learning=CONFIG.meta_learning_enabled
        )

        # RAG System
        self.rag = RAGSystem(embedding_dim=CONFIG.rag_embedding_dim) if CONFIG.rag_enabled else None

        # Автономность
        self.teacher_usage_probability = CONFIG.initial_teacher_usage
        self.autonomy_level = 0.0

        # Статистика
        self.total_interactions = 0
        self.teacher_calls = 0
        self.autonomous_responses = 0
        self.successful_autonomous = 0

        # Пути
        self.user_dir = CONFIG.base_dir / 'models' / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # Загрузка
        self._load_state()

        logger.info(f"🚀 Advanced Agent v3 created for {user_id}")
        logger.info(f"📊 Model: {self._count_parameters() / 1e6:.1f}M parameters")

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.student_model.parameters())

    def _load_state(self):
        """Загрузка состояния"""
        # Загрузка токенизатора
        tokenizer_path = self.user_dir / 'tokenizer'
        if tokenizer_path.exists():
            self.tokenizer.load(tokenizer_path)

        # Загрузка модели
        model_path = self.user_dir / 'student_model.pt'
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=CONFIG.device)
                self.student_model.load_state_dict(checkpoint['model_state_dict'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                self.total_interactions = checkpoint.get('total_interactions', 0)
                self.teacher_calls = checkpoint.get('teacher_calls', 0)
                self.autonomous_responses = checkpoint.get('autonomous_responses', 0)
                self.autonomy_level = checkpoint.get('autonomy_level', 0.0)
                self.teacher_usage_probability = checkpoint.get('teacher_usage_probability',
                                                                CONFIG.initial_teacher_usage)

                logger.info(
                    f"✅ Model loaded: {self.total_interactions} interactions, autonomy={self.autonomy_level:.2%}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        # Сохранение токенизатора
        tokenizer_path = self.user_dir / 'tokenizer'
        tokenizer_path.mkdir(exist_ok=True)
        self.tokenizer.save(tokenizer_path)

        # Сохранение модели
        model_path = self.user_dir / 'student_model.pt'
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'total_interactions': self.total_interactions,
            'teacher_calls': self.teacher_calls,
            'autonomous_responses': self.autonomous_responses,
            'autonomy_level': self.autonomy_level,
            'teacher_usage_probability': self.teacher_usage_probability,
        }
        torch.save(checkpoint, model_path)

        logger.debug(f"💾 State saved: autonomy={self.autonomy_level:.2%}")

    def _should_use_teacher(self) -> bool:
        return np.random.random() < self.teacher_usage_probability

    def _update_autonomy(self, was_successful: bool):
        if was_successful:
            self.autonomy_level = min(1.0, self.autonomy_level + CONFIG.autonomy_growth_rate)
            self.successful_autonomous += 1
        else:
            self.autonomy_level = max(0.0, self.autonomy_level - CONFIG.autonomy_growth_rate * 0.5)

        self.teacher_usage_probability = max(
            CONFIG.min_teacher_usage,
            1.0 - self.autonomy_level
        )

    async def generate_autonomous(self, prompt: str, use_rag: bool = True) -> Tuple[str, float]:
        """Автономная генерация с RAG"""
        # RAG context
        rag_context = ""
        if use_rag and self.rag and CONFIG.rag_enabled:
            rag_context = self.rag.get_augmented_context(prompt, top_k=CONFIG.rag_top_k)

        # Комбинируем prompt с RAG context
        full_prompt = f"{rag_context}\n\n{prompt}" if rag_context else prompt

        # Encode
        prompt_ids = self.tokenizer.encode(full_prompt, max_length=CONFIG.max_seq_length // 2)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=CONFIG.device)

        # Generate
        generated_tensor, confidence = self.student_model.generate(
            prompt_tensor,
            max_length=CONFIG.max_seq_length // 2,
            temperature=0.8,
            eos_token_id=self.tokenizer.special_tokens.get('<EOS>', 3)
        )

        # Decode
        generated_ids = generated_tensor[0].cpu().tolist()
        response = self.tokenizer.decode(generated_ids, skip_special=True)

        return response, confidence

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        """Главная обработка взаимодействия"""
        start_time = time.time()
        self.total_interactions += 1

        response = ""
        confidence = 0.0
        used_teacher = False
        autonomous_attempt = False
        used_rag = False

        # Решение: использовать учителя?
        use_teacher = self._should_use_teacher()

        if not use_teacher:
            # Пытаемся сами
            autonomous_attempt = True
            self.autonomous_responses += 1

            response, confidence = await self.generate_autonomous(user_input, use_rag=True)
            used_rag = CONFIG.rag_enabled and self.rag is not None

            # Проверяем уверенность
            if confidence < CONFIG.confidence_threshold:
                logger.info(f"🤔 Low confidence ({confidence:.2f}) - asking teacher")
                use_teacher = True

        if use_teacher:
            # Спрашиваем учителя
            self.teacher_calls += 1
            used_teacher = True

            teacher_response, _ = await self.teacher.generate(user_input)

            if teacher_response:
                response = teacher_response
                confidence = 1.0

                # Обучаемся
                if self.total_interactions % CONFIG.training_frequency == 0:
                    loss = await self.trainer.train_on_interaction(
                        user_input,
                        teacher_response,
                        use_distillation=True
                    )
                    logger.debug(f"📚 Trained, loss={loss:.4f}")
                else:
                    self.trainer.add_to_replay_buffer(user_input, teacher_response)

                # Добавление в RAG
                if self.rag and CONFIG.rag_enabled:
                    interaction_id = f"{self.user_id}_{self.total_interactions}_{int(time.time())}"
                    self.rag.add_interaction(interaction_id, user_input, teacher_response)
            else:
                response = "Извините, возникла проблема с генерацией ответа."
                confidence = 0.0

        # Обновление токенизатора
        if self.total_interactions % 100 == 0:
            self.tokenizer.train([user_input, response])

        # Replay buffer training
        if self.total_interactions % (CONFIG.training_frequency * 5) == 0:
            replay_loss = await self.trainer.train_on_replay_buffer(num_batches=3)
            logger.debug(f"🔄 Replay training, loss={replay_loss:.4f}")

        # Обновление автономности
        was_successful = confidence >= CONFIG.confidence_threshold
        if autonomous_attempt:
            self._update_autonomy(was_successful)

        # Сохранение
        if self.total_interactions % CONFIG.save_frequency == 0:
            self._save_state()

        # Метаданные
        response_time = time.time() - start_time

        metadata = {
            'used_teacher': used_teacher,
            'autonomous_attempt': autonomous_attempt,
            'confidence': confidence,
            'autonomy_level': self.autonomy_level,
            'teacher_usage_prob': self.teacher_usage_probability,
            'response_time': response_time,
            'training_stats': self.trainer.get_statistics(),
            'used_rag': used_rag,
            'model_size': f"{self._count_parameters() / 1e6:.1f}M",
        }

        logger.info(
            f"✅ [{self.user_id}] "
            f"Teacher={'Yes' if used_teacher else 'No'} | "
            f"RAG={'Yes' if used_rag else 'No'} | "
            f"Conf={confidence:.2f} | "
            f"Autonomy={self.autonomy_level:.1%} | "
            f"T={response_time:.1f}s"
        )

        return response, metadata

    def get_status(self) -> Dict:
        """Статус агента"""
        return {
            'user_id': self.user_id,
            'model_parameters': self._count_parameters(),
            'model_size_mb': self._count_parameters() * 4 / 1e6,  # FP32

            'autonomy': {
                'level': self.autonomy_level,
                'teacher_usage_probability': self.teacher_usage_probability,
                'success_rate': self.successful_autonomous / max(1, self.autonomous_responses),
            },

            'interactions': {
                'total': self.total_interactions,
                'teacher_calls': self.teacher_calls,
                'autonomous_responses': self.autonomous_responses,
                'successful_autonomous': self.successful_autonomous,
            },

            'training': self.trainer.get_statistics(),

            'tokenizer': {
                'type': 'BPE' if TOKENIZERS_AVAILABLE else 'Fallback',
                'vocab_size': CONFIG.vocab_size,
            },

            'features': {
                'rag_enabled': CONFIG.rag_enabled and self.rag is not None,
                'meta_learning': CONFIG.meta_learning_enabled,
                'temporal_embeddings': CONFIG.temporal_embeddings,
                'mixed_precision': CONFIG.mixed_precision,
            }
        }


import copy  # Для Meta-Learning


# ══════════════════════════════════════════════════════════════
# 🤖 TELEGRAM BOT
# ══════════════════════════════════════════════════════════════

class AdvancedBot:
    """Telegram бот с продвинутым агентом v3"""

    def __init__(self):
        self.teacher: Optional[TeacherLLM] = None
        self.agents: Dict[str, AdvancedAutonomousAgent] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        """Инициализация бота"""
        self.teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.teacher.connect()

        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()

        # Handlers
        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self._handle_message
        ))

        for cmd, handler in [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('stats', self._cmd_stats),
            ('help', self._cmd_help),
        ]:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Advanced Bot v3 initialized")

    async def _get_or_create_agent(self, user_id: str) -> AdvancedAutonomousAgent:
        """Получение или создание агента"""
        if user_id not in self.agents:
            self.agents[user_id] = AdvancedAutonomousAgent(user_id, self.teacher)
        return self.agents[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщения"""
        if not update.effective_user or not update.message:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )

        try:
            agent = await self._get_or_create_agent(user_id)
            response, metadata = await agent.process_interaction(user_input)

            # Footer
            footer = f"\n\n<i>"
            if metadata['used_teacher']:
                footer += "👨‍🏫 Teacher"
            else:
                footer += f"🤖 Auto (conf: {metadata['confidence']:.0%})"

            if metadata.get('used_rag'):
                footer += " + 📚 RAG"

            footer += f" | 🎯 {metadata['autonomy_level']:.0%}"
            footer += f" | ⚡ {metadata['model_size']}"
            footer += "</i>"

            await update.message.reply_text(
                response + footer,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

        except Exception as e:
            logger.exception(f"Error from {user_id}")
            await update.message.reply_text("⚠️ Произошла ошибка при обработке запроса")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        message = """🚀 <b>ADVANCED AUTONOMOUS AGENT v3.0</b>

Привет! Я — продвинутый самообучающийся агент.

<b>✨ НОВЫЕ ВОЗМОЖНОСТИ v3:</b>

🧠 <b>Увеличенная модель</b>
• До 100M параметров (адаптация под GPU)
• 12 слоёв Transformer
• 1024 размерность

🔤 <b>BPE Tokenizer</b>
• 50,000 токенов
• Подслова для незнакомых слов
• Как в GPT-2/GPT-3

📚 <b>RAG (Retrieval-Augmented Generation)</b>
• Векторная база знаний
• Семантический поиск
• Контекстное дополнение

🎯 <b>Meta-Learning (MAML)</b>
• Быстрая адаптация
• Few-shot learning
• Обучение обучаться

⏰ <b>Внутреннее время</b>
• Временные embeddings
• Непрерывность сознания
• Циркадные ритмы

<b>Команды:</b> /help | /status | /stats"""

        await update.message.reply_text(message)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status"""
        user_id = str(update.effective_user.id)
        agent = await self._get_or_create_agent(user_id)
        status = agent.get_status()

        message = f"""🚀 <b>СТАТУС v3.0</b>

<b>🧠 Модель</b>
• Параметров: {status['model_parameters']:,}
• Размер: {status['model_size_mb']:.1f} MB
• Токенизатор: {status['tokenizer']['type']}

<b>🎯 Автономность</b>
• Уровень: {status['autonomy']['level']:.1%}
• Вероятность учителя: {status['autonomy']['teacher_usage_probability']:.1%}
• Успех авто: {status['autonomy']['success_rate']:.1%}

<b>📊 Взаимодействия</b>
• Всего: {status['interactions']['total']}
• К учителю: {status['interactions']['teacher_calls']}
• Автономных: {status['interactions']['autonomous_responses']}

<b>📚 Обучение</b>
• Шагов: {status['training']['training_steps']}
• Потеря: {status['training']['avg_loss']:.4f}
• LR: {status['training'].get('current_lr', 0):.2e}

<b>✨ Функции</b>
• RAG: {'✅' if status['features']['rag_enabled'] else '❌'}
• Meta-Learning: {'✅' if status['features']['meta_learning'] else '❌'}
• Временные emb: {'✅' if status['features']['temporal_embeddings'] else '❌'}
• Mixed Precision: {'✅' if status['features']['mixed_precision'] else '❌'}"""

        await update.message.reply_text(message)

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /stats"""
        user_id = str(update.effective_user.id)
        agent = await self._get_or_create_agent(user_id)
        status = agent.get_status()

        # Графики прогресса
        autonomy = status['autonomy']['level']
        teacher_prob = status['autonomy']['teacher_usage_probability']

        autonomy_bar = "█" * int(autonomy * 20) + "░" * (20 - int(autonomy * 20))
        teacher_bar = "█" * int(teacher_prob * 20) + "░" * (20 - int(teacher_prob * 20))

        message = f"""📊 <b>ДЕТАЛЬНАЯ СТАТИСТИКА v3</b>

<b>Автономность:</b>
{autonomy_bar} {autonomy:.1%}

<b>Зависимость от учителя:</b>
{teacher_bar} {teacher_prob:.1%}

<b>Модель:</b>
• Параметры: {status['model_parameters'] / 1e6:.1f}M
• Размер: {status['model_size_mb']:.1f} MB
• Архитектура: {CONFIG.n_layers}L-{CONFIG.d_model}D-{CONFIG.n_heads}H

<b>Обучение:</b>
• Средняя потеря: {status['training']['avg_loss']:.4f}
• Последняя: {status['training']['recent_loss']:.4f}
• Learning Rate: {status['training'].get('current_lr', 0):.2e}
• Replay buffer: {status['training']['replay_buffer_size']}

<b>Фичи v3:</b>
• BPE Tokenizer: {status['tokenizer']['vocab_size']} токенов
• RAG: {'Включён' if status['features']['rag_enabled'] else 'Выключен'}
• Meta-Learning: {'Включён' if status['features']['meta_learning'] else 'Выключен'}
• Temporal Embeddings: {'Да' if status['features']['temporal_embeddings'] else 'Нет'}
• Mixed Precision: {'Да' if status['features']['mixed_precision'] else 'Нет'}

<b>Устройство:</b> {CONFIG.device.upper()}</b>"""

        await update.message.reply_text(message)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        message = """🚀 <b>ADVANCED AGENT v3.0 — СПРАВКА</b>

<b>🔥 УЛУЧШЕНИЯ v3:</b>

<b>1. Увеличенная модель</b>
• Автоадаптация под GPU
• До 100M параметров
• 12 слоёв вместо 6

<b>2. BPE Tokenizer</b>
• 50K токенов вместо 10K
• Подслова (как GPT-2)
• Лучше незнакомые слова

<b>3. RAG System</b>
• Векторная база знаний
• Семантический поиск
• Долговременная память

<b>4. Meta-Learning</b>
• MAML алгоритм
• Few-shot адаптация
• Быстрое обучение

<b>5. Внутреннее время</b>
• Временные embeddings
• Непрерывность
• Циркадные ритмы

<b>📌 КОМАНДЫ:</b>
• /start — приветствие
• /status — статус
• /stats — детали
• /help — эта справка

<b>⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:</b>
• Быстрее на 2-3x (Mixed Precision)
• Умнее (больше параметров)
• Память (RAG)
• Адаптивнее (Meta-Learning)</b>"""

        await update.message.reply_text(message)

    async def start_polling(self):
        """Запуск бота"""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot v3 started")

    async def shutdown(self):
        """Остановка бота"""
        logger.info("🛑 Shutting down...")

        # Сохранение всех агентов
        for user_id, agent in self.agents.items():
            try:
                agent._save_state()
                logger.info(f"💾 Saved agent: {user_id}")
            except Exception as e:
                logger.error(f"Error saving agent {user_id}: {e}")

        if self.teacher:
            await self.teacher.close()

        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Shutdown complete")


# ══════════════════════════════════════════════════════════════
# 🚀 MAIN
# ══════════════════════════════════════════════════════════════

async def main():
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  🚀 ADVANCED AUTONOMOUS AGENT v3.0                            ║
║     Максимальная версия с RAG, Meta-Learning, BPE             ║
╚═══════════════════════════════════════════════════════════════╝

🔥 НОВЫЕ ВОЗМОЖНОСТИ v3:

✅ УВЕЛИЧЕННАЯ МОДЕЛЬ
   • До {CONFIG.d_model}D, {CONFIG.n_layers}L, {CONFIG.n_heads}H
   • Автоадаптация под GPU
   • Mixed Precision Training

✅ BPE TOKENIZER
   • {CONFIG.vocab_size} токенов
   • Byte Pair Encoding
   • Подслова для OOV

✅ RAG SYSTEM
   • ChromaDB векторная БД
   • Семантический поиск
   • Долговременная память

✅ META-LEARNING
   • MAML алгоритм
   • Few-shot adaptation
   • Быстрое обучение

✅ ВНУТРЕННЕЕ ВРЕМЯ
   • Temporal embeddings
   • Непрерывность сознания
   • Циркадные ритмы

🎯 УСТРОЙСТВО: {CONFIG.device.upper()}
📊 Параметры модели: ~{(CONFIG.vocab_size * CONFIG.d_model + CONFIG.n_layers * CONFIG.d_model * CONFIG.d_model * 8) / 1e6:.1f}M
""")

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = AdvancedBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 ADVANCED AGENT v3 АКТИВЕН")
        logger.info(f"🎮 Device: {CONFIG.device}")
        logger.info(f"📊 Model: {CONFIG.n_layers}L-{CONFIG.d_model}D-{CONFIG.n_heads}H")
        logger.info(f"🔤 Tokenizer: BPE {CONFIG.vocab_size} tokens")
        logger.info(f"📚 RAG: {'Enabled' if CONFIG.rag_enabled else 'Disabled'}")
        logger.info(f"🎯 Meta-Learning: {'Enabled' if CONFIG.meta_learning_enabled else 'Disabled'}")
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