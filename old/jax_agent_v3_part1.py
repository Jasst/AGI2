#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ADVANCED AUTONOMOUS AGENT v3.0 — JAX VERSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ JAX вместо PyTorch - современная альтернатива от Google!
✅ Отличная поддержка GPU/TPU
✅ Автоматическая дифференциация
✅ JIT компиляция (очень быстро!)
✅ NumPy-like синтаксис

🎯 ПРЕИМУЩЕСТВА JAX:
   • Быстрее PyTorch на многих задачах
   • Проще для понимания (похож на NumPy)
   • Лучше для исследований
   • XLA компиляция → максимальная скорость
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
import numpy as np
from pathlib import Path
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import gzip
import pickle
import time
import re
from dotenv import load_dotenv

# JAX imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.experimental import optimizers
import flax.linen as nn
from flax.training import train_state

# Telegram
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)

# BPE Tokenizer (опционально)
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("⚠️ Tokenizers not available. Install: pip install tokenizers")

load_dotenv()

# Проверка JAX и GPU
print(f"🔥 JAX version: {jax.__version__}")
print(f"🎮 JAX devices: {jax.devices()}")
print(f"🚀 Default backend: {jax.default_backend()}")


# ══════════════════════════════════════════════════════════════
# ⚙️ CONFIGURATION
# ══════════════════════════════════════════════════════════════

@dataclass
class JAXConfig:
    """Конфигурация для JAX версии"""
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    # Model - адаптируем под память
    vocab_size: int = 50000
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    d_ff: int = 4096
    max_seq_length: int = 1024
    dropout_rate: float = 0.1

    # Training
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    batch_size: int = 4

    # Temporal
    use_temporal: bool = True
    time_dim: int = 64

    # Autonomy
    initial_teacher_usage: float = 1.0
    min_teacher_usage: float = 0.05
    autonomy_growth_rate: float = 0.002
    confidence_threshold: float = 0.75

    # Memory
    replay_buffer_size: int = 50000
    training_frequency: int = 5
    save_frequency: int = 50

    # JAX specific
    jax_platform: str = 'gpu'  # 'gpu', 'cpu', or 'tpu'

    # Paths
    base_dir: Path = Path('jax_agent_v3')

    def __post_init__(self):
        for subdir in ['models', 'memory', 'logs', 'tokenizer']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Автомасштабирование под GPU
        if self.jax_platform == 'gpu':
            self._auto_scale_for_gpu()

    def _auto_scale_for_gpu(self):
        """Автомасштабирование под доступную GPU память"""
        try:
            # Получаем информацию о GPU через JAX
            devices = jax.devices('gpu')
            if not devices:
                print("⚠️ No GPU found, using CPU")
                self.jax_platform = 'cpu'
                self.d_model = 512
                self.n_layers = 8
                return

            # Примерная оценка памяти GPU (JAX не даёт прямого API)
            # Будем консервативны
            estimated_params = (
                    self.vocab_size * self.d_model * 2 +
                    self.n_layers * (
                            4 * self.d_model * self.d_model +
                            2 * self.d_model * self.d_ff
                    )
            )

            estimated_memory_gb = estimated_params * 4 / 1e9 * 1.5

            print(f"📊 Estimated model: {estimated_params / 1e6:.1f}M params, ~{estimated_memory_gb:.2f} GB")

            # Если больше 8GB - уменьшаем
            if estimated_memory_gb > 8.0:
                scale = (8.0 / estimated_memory_gb) ** 0.5
                self.d_model = int(self.d_model * scale)
                self.d_ff = int(self.d_ff * scale)
                self.d_model = (self.d_model // self.n_heads) * self.n_heads  # Выравнивание
                print(f"⚙️ Auto-scaled: d_model={self.d_model}, d_ff={self.d_ff}")

        except Exception as e:
            print(f"⚠️ Auto-scaling failed: {e}")


CONFIG = JAXConfig()


# ══════════════════════════════════════════════════════════════
# 📊 LOGGING
# ══════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    logger = logging.getLogger('JAX_Agent_v3')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'jax_agent_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ══════════════════════════════════════════════════════════════
# 🔤 BPE TOKENIZER
# ══════════════════════════════════════════════════════════════

class BPETokenizer:
    """BPE токенизатор"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }

        if TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        else:
            # Fallback
            self.word_to_id = self.special_tokens.copy()
            self.id_to_word = {v: k for k, v in self.special_tokens.items()}
            self.next_id = len(self.special_tokens)

    def train(self, texts: List[str]):
        """Обучение BPE"""
        if not TOKENIZERS_AVAILABLE:
            # Fallback: word-level
            word_freq = Counter()
            for text in texts:
                word_freq.update(text.lower().split())

            for word, _ in word_freq.most_common(self.vocab_size - len(self.special_tokens)):
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
            return

        # BPE training
        train_file = CONFIG.base_dir / 'tokenizer' / 'train.txt'
        train_file.parent.mkdir(exist_ok=True)

        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.keys())
        )
        self.tokenizer.train([str(train_file)], trainer)
        train_file.unlink()

    def encode(self, text: str, max_length: Optional[int] = None) -> np.ndarray:
        """Кодирование в массив"""
        if TOKENIZERS_AVAILABLE:
            encoding = self.tokenizer.encode(text)
            tokens = encoding.ids
        else:
            words = text.lower().split()
            tokens = [self.special_tokens['<BOS>']]
            for word in words:
                tokens.append(self.word_to_id.get(word, self.special_tokens['<UNK>']))
            tokens.append(self.special_tokens['<EOS>'])

        # Padding/truncation
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.special_tokens['<PAD>']] * (max_length - len(tokens))

        return np.array(tokens, dtype=np.int32)

    def decode(self, tokens: np.ndarray, skip_special: bool = True) -> str:
        """Декодирование"""
        tokens = tokens.tolist() if isinstance(tokens, np.ndarray) else tokens

        if TOKENIZERS_AVAILABLE:
            if skip_special:
                tokens = [t for t in tokens if t >= len(self.special_tokens)]
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special)
        else:
            words = []
            for token_id in tokens:
                word = self.id_to_word.get(token_id, '<UNK>')
                if not skip_special or word not in self.special_tokens:
                    words.append(word)
            return ' '.join(words)


# ══════════════════════════════════════════════════════════════
# ⏰ TEMPORAL EMBEDDINGS (JAX)
# ══════════════════════════════════════════════════════════════

class TemporalEmbeddings(nn.Module):
    """Временные embeddings на JAX/Flax"""
    time_dim: int = 64

    @nn.compact
    def __call__(self, batch_size: int = 1):
        # Получаем текущее время
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        month = now.month - 1

        # Embeddings
        hour_emb = nn.Embed(24, self.time_dim)(jnp.array([hour]))
        weekday_emb = nn.Embed(7, self.time_dim)(jnp.array([weekday]))
        month_emb = nn.Embed(12, self.time_dim)(jnp.array([month]))

        # Комбинируем
        temporal = hour_emb + weekday_emb + month_emb

        # Expand для batch
        temporal = jnp.broadcast_to(temporal, (batch_size, self.time_dim))

        return temporal


# ══════════════════════════════════════════════════════════════
# 🧠 TRANSFORMER MODEL (JAX/Flax)
# ══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention на Flax"""
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, training: bool = True):
        d_k = self.d_model // self.num_heads

        # Linear projections
        q = nn.Dense(self.d_model)(x)
        k = nn.Dense(self.d_model)(x)
        v = nn.Dense(self.d_model)(x)

        # Reshape для multi-head
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        q = q.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)

        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        attention = jax.nn.softmax(scores, axis=-1)
        attention = nn.Dropout(self.dropout_rate, deterministic=not training)(attention)

        # Apply attention
        out = jnp.matmul(attention, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        out = nn.Dense(self.d_model)(out)

        return out


class FeedForward(nn.Module):
    """Feed-Forward Network"""
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(self.d_ff)(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.d_model)(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, training: bool = True):
        # Self-attention
        attn_out = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout_rate=self.dropout_rate
        )(x, mask, training)

        x = nn.LayerNorm()(x + nn.Dropout(self.dropout_rate, deterministic=not training)(attn_out))

        # Feed-forward
        ff_out = FeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )(x, training)

        x = nn.LayerNorm()(x + nn.Dropout(self.dropout_rate, deterministic=not training)(ff_out))

        return x


class TransformerModel(nn.Module):
    """Полная Transformer модель на JAX"""
    vocab_size: int
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 12
    d_ff: int = 4096
    max_seq_length: int = 1024
    dropout_rate: float = 0.1
    use_temporal: bool = True
    time_dim: int = 64

    @nn.compact
    def __call__(self, input_ids, training: bool = True):
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = nn.Embed(self.vocab_size, self.d_model)(input_ids)

        # Position embeddings
        positions = jnp.arange(seq_len)[None, :]
        pos_emb = nn.Embed(self.max_seq_length, self.d_model)(positions)
        x = x + pos_emb

        # Temporal embeddings
        if self.use_temporal:
            temporal_emb = TemporalEmbeddings(time_dim=self.time_dim)(batch_size)
            # Проецируем temporal в d_model
            temporal_proj = nn.Dense(self.d_model)(temporal_emb)
            x = x + temporal_proj[:, None, :]

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            )(x, training=training)

        # Output
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)

        return logits


# ══════════════════════════════════════════════════════════════
# 🎓 TRAINING STATE
# ══════════════════════════════════════════════════════════════

def create_train_state(rng, model, learning_rate):
    """Создание training state"""
    # Dummy input для инициализации
    dummy_input = jnp.ones((1, CONFIG.max_seq_length), dtype=jnp.int32)

    # Инициализация параметров
    params = model.init(rng, dummy_input, training=False)

    # Optimizer (AdamW)
    tx = optimizers.adam(learning_rate)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jit
def train_step(state, batch, labels):
    """JIT-compiled training step"""

    def loss_fn(params):
        logits = state.apply_fn(params, batch, training=True)
        # Cross-entropy loss
        loss = optimizers.cross_entropy_loss(logits.reshape(-1, CONFIG.vocab_size), labels.reshape(-1))
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


@jit
def generate_step(state, input_ids):
    """JIT-compiled generation step"""
    logits = state.apply_fn(state.params, input_ids, training=False)
    # Берём последний токен
    next_token_logits = logits[:, -1, :]
    # Softmax для вероятностей
    probs = jax.nn.softmax(next_token_logits, axis=-1)
    return probs


# ══════════════════════════════════════════════════════════════
# 👨‍🏫 TEACHER LLM
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

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Tuple[str, List[float]]:
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

# 🎓 JAX TRAINER
# ══════════════════════════════════════════════════════════════

class JAXTrainer:
    """Тренер на JAX"""

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer

        # Создание модели
        self.model = TransformerModel(
            vocab_size=CONFIG.vocab_size,
            d_model=CONFIG.d_model,
            num_heads=CONFIG.n_heads,
            num_layers=CONFIG.n_layers,
            d_ff=CONFIG.d_ff,
            max_seq_length=CONFIG.max_seq_length,
            dropout_rate=CONFIG.dropout_rate,
            use_temporal=CONFIG.use_temporal,
            time_dim=CONFIG.time_dim
        )

        # Инициализация
        rng = random.PRNGKey(0)
        self.state = create_train_state(rng, self.model, CONFIG.learning_rate)

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=CONFIG.replay_buffer_size)

        # Stats
        self.training_steps = 0
        self.total_loss = 0.0
        self.losses_history: deque = deque(maxlen=1000)

        # Подсчёт параметров
        params_count = sum(x.size for x in jax.tree_util.tree_leaves(self.state.params))
        logger.info(f"🧠 JAX Model: {params_count / 1e6:.1f}M parameters")

    def add_to_replay_buffer(self, prompt: str, teacher_response: str):
        """Добавление в replay buffer"""
        self.replay_buffer.append({
            'prompt': prompt,
            'response': teacher_response,
            'timestamp': time.time()
        })

    async def train_on_interaction(self, prompt: str, teacher_response: str) -> float:
        """Обучение на взаимодействии"""
        self.add_to_replay_buffer(prompt, teacher_response)

        # Подготовка данных
        text = f"{prompt} {teacher_response}"
        input_ids = self.tokenizer.encode(text, max_length=CONFIG.max_seq_length)

        # Входы и метки
        input_tensor = jnp.array([input_ids[:-1]])  # [batch=1, seq_len-1]
        labels = jnp.array([input_ids[1:]])  # [batch=1, seq_len-1]

        # Training step
        self.state, loss = train_step(self.state, input_tensor, labels)

        # Stats
        self.training_steps += 1
        loss_val = float(loss)
        self.total_loss += loss_val
        self.losses_history.append(loss_val)

        return loss_val

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
                    item['response']
                )
                total_loss += loss

        return total_loss / (num_batches * CONFIG.batch_size)

    def generate(
            self,
            prompt_ids: np.ndarray,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            eos_token_id: int = 3
    ) -> Tuple[np.ndarray, float]:
        """Генерация текста"""
        generated = list(prompt_ids)
        confidences = []

        for _ in range(max_length):
            # Подготовка входа
            input_tensor = jnp.array([generated[-CONFIG.max_seq_length:]])

            # Forward pass (JIT-compiled!)
            probs = generate_step(self.state, input_tensor)
            probs = np.array(probs[0])  # Конвертируем в NumPy

            # Temperature
            if temperature != 1.0:
                logits = np.log(probs + 1e-10)
                logits = logits / temperature
                probs = np.exp(logits) / np.sum(np.exp(logits))

            # Top-k filtering
            if top_k > 0:
                top_k_indices = np.argsort(probs)[-top_k:]
                probs_filtered = np.zeros_like(probs)
                probs_filtered[top_k_indices] = probs[top_k_indices]
                probs = probs_filtered / probs_filtered.sum()

            # Sample
            next_token = np.random.choice(len(probs), p=probs)

            # Confidence
            confidence = float(probs[next_token])
            confidences.append(confidence)

            # Append
            generated.append(int(next_token))

            # Check EOS
            if next_token == eos_token_id:
                break

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return np.array(generated), avg_confidence

    def get_statistics(self) -> Dict:
        """Статистика обучения"""
        return {
            'training_steps': self.training_steps,
            'avg_loss': np.mean(list(self.losses_history)) if self.losses_history else 0.0,
            'recent_loss': self.losses_history[-1] if self.losses_history else 0.0,
            'replay_buffer_size': len(self.replay_buffer),
        }


# ══════════════════════════════════════════════════════════════
# 🤖 JAX AUTONOMOUS AGENT
# ══════════════════════════════════════════════════════════════

class JAXAutonomousAgent:
    """Автономный агент на JAX"""

    def __init__(self, user_id: str, teacher: TeacherLLM):
        self.user_id = user_id
        self.teacher = teacher

        # Tokenizer
        self.tokenizer = BPETokenizer(vocab_size=CONFIG.vocab_size)

        # Trainer (содержит модель)
        self.trainer = JAXTrainer(self.tokenizer)

        # Autonomy
        self.teacher_usage_probability = CONFIG.initial_teacher_usage
        self.autonomy_level = 0.0

        # Stats
        self.total_interactions = 0
        self.teacher_calls = 0
        self.autonomous_responses = 0
        self.successful_autonomous = 0

        # Paths
        self.user_dir = CONFIG.base_dir / 'models' / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self._load_state()

        logger.info(f"🚀 JAX Agent created for {user_id}")

    def _count_parameters(self) -> int:
        """Подсчёт параметров"""
        return sum(x.size for x in jax.tree_util.tree_leaves(self.trainer.state.params))

    def _load_state(self):
        """Загрузка состояния"""
        # Tokenizer
        tokenizer_path = self.user_dir / 'tokenizer'
        if (tokenizer_path / 'tokenizer.json').exists() and TOKENIZERS_AVAILABLE:
            from tokenizers import Tokenizer as TokenizerLoader
            self.tokenizer.tokenizer = TokenizerLoader.from_file(str(tokenizer_path / 'tokenizer.json'))
            logger.info("✅ Tokenizer loaded")

        # Model state
        model_path = self.user_dir / 'model_state.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    saved_state = pickle.load(f)

                # Restore JAX state
                self.trainer.state = train_state.TrainState.create(
                    apply_fn=self.trainer.model.apply,
                    params=saved_state['params'],
                    tx=self.trainer.state.tx
                )

                self.total_interactions = saved_state.get('total_interactions', 0)
                self.teacher_calls = saved_state.get('teacher_calls', 0)
                self.autonomous_responses = saved_state.get('autonomous_responses', 0)
                self.autonomy_level = saved_state.get('autonomy_level', 0.0)
                self.teacher_usage_probability = saved_state.get('teacher_usage_probability',
                                                                 CONFIG.initial_teacher_usage)

                logger.info(
                    f"✅ Model loaded: {self.total_interactions} interactions, autonomy={self.autonomy_level:.2%}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        # Tokenizer
        tokenizer_path = self.user_dir / 'tokenizer'
        tokenizer_path.mkdir(exist_ok=True)

        if TOKENIZERS_AVAILABLE and hasattr(self.tokenizer, 'tokenizer'):
            self.tokenizer.tokenizer.save(str(tokenizer_path / 'tokenizer.json'))

        # Model state
        model_path = self.user_dir / 'model_state.pkl'
        saved_state = {
            'params': self.trainer.state.params,
            'total_interactions': self.total_interactions,
            'teacher_calls': self.teacher_calls,
            'autonomous_responses': self.autonomous_responses,
            'autonomy_level': self.autonomy_level,
            'teacher_usage_probability': self.teacher_usage_probability,
        }

        with open(model_path, 'wb') as f:
            pickle.dump(saved_state, f)

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

    async def generate_autonomous(self, prompt: str) -> Tuple[str, float]:
        """Автономная генерация"""
        # Encode
        prompt_ids = self.tokenizer.encode(prompt, max_length=CONFIG.max_seq_length // 2)

        # Generate
        generated_ids, confidence = self.trainer.generate(
            prompt_ids,
            max_length=CONFIG.max_seq_length // 2,
            temperature=0.8,
            eos_token_id=self.tokenizer.special_tokens.get('<EOS>', 3)
        )

        # Decode
        response = self.tokenizer.decode(generated_ids, skip_special=True)

        return response, confidence

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        """Главная обработка"""
        start_time = time.time()
        self.total_interactions += 1

        response = ""
        confidence = 0.0
        used_teacher = False
        autonomous_attempt = False

        # Decision: use teacher?
        use_teacher = self._should_use_teacher()

        if not use_teacher:
            autonomous_attempt = True
            self.autonomous_responses += 1

            response, confidence = await self.generate_autonomous(user_input)

            if confidence < CONFIG.confidence_threshold:
                logger.info(f"🤔 Low confidence ({confidence:.2f}) - asking teacher")
                use_teacher = True

        if use_teacher:
            self.teacher_calls += 1
            used_teacher = True

            teacher_response, _ = await self.teacher.generate(user_input)

            if teacher_response:
                response = teacher_response
                confidence = 1.0

                # Train
                if self.total_interactions % CONFIG.training_frequency == 0:
                    loss = await self.trainer.train_on_interaction(user_input, teacher_response)
                    logger.debug(f"📚 Trained, loss={loss:.4f}")
                else:
                    self.trainer.add_to_replay_buffer(user_input, teacher_response)
            else:
                response = "Извините, возникла проблема с генерацией ответа."
                confidence = 0.0

        # Update tokenizer
        if self.total_interactions % 100 == 0:
            self.tokenizer.train([user_input, response])

        # Replay training
        if self.total_interactions % (CONFIG.training_frequency * 5) == 0:
            replay_loss = await self.trainer.train_on_replay_buffer(num_batches=3)
            logger.debug(f"🔄 Replay training, loss={replay_loss:.4f}")

        # Update autonomy
        was_successful = confidence >= CONFIG.confidence_threshold
        if autonomous_attempt:
            self._update_autonomy(was_successful)

        # Save
        if self.total_interactions % CONFIG.save_frequency == 0:
            self._save_state()

        # Metadata
        response_time = time.time() - start_time

        metadata = {
            'used_teacher': used_teacher,
            'autonomous_attempt': autonomous_attempt,
            'confidence': confidence,
            'autonomy_level': self.autonomy_level,
            'teacher_usage_prob': self.teacher_usage_probability,
            'response_time': response_time,
            'training_stats': self.trainer.get_statistics(),
            'model_size': f"{self._count_parameters() / 1e6:.1f}M",
            'backend': jax.default_backend(),
        }

        logger.info(
            f"✅ [{self.user_id}] "
            f"Teacher={'Yes' if used_teacher else 'No'} | "
            f"Conf={confidence:.2f} | "
            f"Autonomy={self.autonomy_level:.1%} | "
            f"T={response_time:.1f}s | "
            f"Backend={jax.default_backend()}"
        )

        return response, metadata

    def get_status(self) -> Dict:
        """Статус агента"""
        return {
            'user_id': self.user_id,
            'model_parameters': self._count_parameters(),
            'jax_backend': jax.default_backend(),
            'jax_devices': str(jax.devices()),

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
            }
        }


# ══════════════════════════════════════════════════════════════
# 🤖 TELEGRAM BOT
# ══════════════════════════════════════════════════════════════

class JAXBot:
    """Telegram бот с JAX агентом"""

    def __init__(self):
        self.teacher: Optional[TeacherLLM] = None
        self.agents: Dict[str, JAXAutonomousAgent] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        self.teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.teacher.connect()

        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        for cmd, handler in [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('help', self._cmd_help),
        ]:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 JAX Bot initialized")

    async def _get_or_create_agent(self, user_id: str) -> JAXAutonomousAgent:
        if user_id not in self.agents:
            self.agents[user_id] = JAXAutonomousAgent(user_id, self.teacher)
        return self.agents[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or not update.message:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            agent = await self._get_or_create_agent(user_id)
            response, metadata = await agent.process_interaction(user_input)

            footer = f"\n\n<i>"
            if metadata['used_teacher']:
                footer += "👨‍🏫 Teacher"
            else:
                footer += f"🤖 Auto ({metadata['confidence']:.0%})"

            footer += f" | 🎯 {metadata['autonomy_level']:.0%}"
            footer += f" | ⚡ {metadata['model_size']}"
            footer += f" | 🔥 JAX-{metadata['backend']}"
            footer += "</i>"

            await update.message.reply_text(
                response + footer,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

        except Exception as e:
            logger.exception(f"Error from {user_id}")
            await update.message.reply_text("⚠️ Произошла ошибка")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🔥 <b>JAX AUTONOMOUS AGENT v3.0</b>

Привет! Я на <b>JAX</b> - современной альтернативе PyTorch от Google!

<b>✨ ПРЕИМУЩЕСТВА JAX:</b>

⚡ <b>Скорость</b>
• JIT компиляция
• XLA оптимизация
• Быстрее PyTorch на многих задачах

🎮 <b>GPU/TPU</b>
• Отличная поддержка
• Автоматическая оптимизация
• Работает на {jax.default_backend()}

🧠 <b>Модель</b>
• {CONFIG.n_layers} слоёв
• {CONFIG.d_model} размерность
• Temporal embeddings

<b>Команды:</b> /help | /status"""

        await update.message.reply_text(message)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_or_create_agent(user_id)
        status = agent.get_status()

        message = f"""🔥 <b>JAX AGENT STATUS</b>

<b>🧠 Модель</b>
• Параметров: {status['model_parameters']:,}
• Backend: {status['jax_backend']}
• Устройства: {status['jax_devices'][:50]}...

<b>🎯 Автономность</b>
• Уровень: {status['autonomy']['level']:.1%}
• Учитель: {status['autonomy']['teacher_usage_probability']:.1%}
• Успех: {status['autonomy']['success_rate']:.1%}

<b>📊 Взаимодействия</b>
• Всего: {status['interactions']['total']}
• К учителю: {status['interactions']['teacher_calls']}
• Автономных: {status['interactions']['autonomous_responses']}

<b>📚 Обучение</b>
• Шагов: {status['training']['training_steps']}
• Потеря: {status['training']['avg_loss']:.4f}

<b>🔤 Tokenizer</b>
• Тип: {status['tokenizer']['type']}
• Словарь: {status['tokenizer']['vocab_size']:,}"""

        await update.message.reply_text(message)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🔥 <b>JAX AGENT v3 — СПРАВКА</b>

<b>ЧТО ТАКОЕ JAX?</b>

JAX - это библиотека от Google для:
• Быстрых вычислений
• Автоматической дифференциации
• GPU/TPU оптимизации

<b>ПОЧЕМУ JAX vs PyTorch?</b>

✅ Работает на Python 3.8-3.11
✅ JIT компиляция → быстрее
✅ Проще для понимания
✅ Отличная поддержка GPU

<b>ВОЗМОЖНОСТИ:</b>

🧠 Transformer модель
⏰ Внутреннее время
🎓 Дистилляция знаний
📈 Рост автономности

<b>КОМАНДЫ:</b>
• /start — приветствие
• /status — статус
• /help — эта справка

<b>Backend:</b> {jax.default_backend()}"""

        await update.message.reply_text(message)

    async def start_polling(self):
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ JAX Bot started")

    async def shutdown(self):
        logger.info("🛑 Shutting down...")

        for user_id, agent in self.agents.items():
            try:
                agent._save_state()
                logger.info(f"💾 Saved: {user_id}")
            except Exception as e:
                logger.error(f"Error saving {user_id}: {e}")

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
║  🔥 JAX AUTONOMOUS AGENT v3.0                                 ║
║     Powered by Google JAX - альтернатива PyTorch              ║
╚═══════════════════════════════════════════════════════════════╝

🔥 JAX VERSION: {jax.__version__}
🎮 BACKEND: {jax.default_backend()}
🖥️  DEVICES: {jax.devices()}

✅ ПРЕИМУЩЕСТВА JAX:
   • JIT компиляция → Быстрее
   • XLA оптимизация → Эффективнее
   • Работает на Python 3.8-3.11
   • Отлично с GPU/TPU

🧠 МОДЕЛЬ:
   • {CONFIG.n_layers} слоёв
   • {CONFIG.d_model} размерность
   • {CONFIG.vocab_size:,} токенов
   • Temporal embeddings

🎯 УСТРОЙСТВО: {jax.default_backend().upper()}
""")

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = JAXBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 JAX AGENT АКТИВЕН")
        logger.info(f"🔥 Backend: {jax.default_backend()}")
        logger.info(f"🎮 Devices: {jax.devices()}")
        logger.info("🛑 Ctrl+C для остановки")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n👋 Остановка...")
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
        import traceback

        traceback.print_exc()
        sys.exit(1)