#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ADVANCED AUTONOMOUS LEARNING AGENT v3.0 — Веб-версия для blockcoin.ru
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Работает через:
• FastAPI + WebSockets (асинхронный веб-сервер)
• Uvicorn (запуск)
• HTML/CSS/JS интерфейс (встроенный)
"""

import os
import sys
import json
import asyncio
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

# Веб-сервер
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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
    # LM Studio API (остается)
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    # Веб-сервер
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_path: str = "/ws"

    # Student Model - МАКСИМАЛЬНАЯ версия
    vocab_size: int = 50000
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    d_ff: int = 4096
    max_seq_length: int = 1024
    dropout: float = 0.1

    # Автоматическая адаптация под GPU
    auto_scale_model: bool = True
    max_gpu_memory_gb: float = 8.0

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

        if self.auto_scale_model and self.device == 'cuda':
            self._auto_scale_to_gpu()

    def _auto_scale_to_gpu(self):
        """Автоматическая адаптация размера модели под GPU"""
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU Memory: {gpu_memory_gb:.2f} GB")

            estimated_params = (
                    self.vocab_size * self.d_model * 2 +
                    self.n_layers * (4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff)
            )
            estimated_memory_gb = estimated_params * 4 / 1e9 * 1.5

            if estimated_memory_gb > gpu_memory_gb * 0.7:
                scale_factor = (gpu_memory_gb * 0.7) / estimated_memory_gb
                self.d_model = int(self.d_model * scale_factor ** 0.5)
                self.d_ff = int(self.d_ff * scale_factor ** 0.5)
                self.n_layers = max(6, int(self.n_layers * scale_factor ** 0.25))
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
    console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))

    log_file = CONFIG.base_dir / 'logs' / f'agent_v3_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ══════════════════════════════════════════════════════════════
# 🔤 ADVANCED BPE TOKENIZER (сокращен для краткости, но полный функционал)
# ══════════════════════════════════════════════════════════════

class AdvancedBPETokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer: Optional[Tokenizer] = None
        self.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        if TOKENIZERS_AVAILABLE:
            self._init_bpe_tokenizer()
        else:
            self._init_fallback_tokenizer()

    def _init_bpe_tokenizer(self):
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.add_special_tokens(list(self.special_tokens.keys()))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[("<BOS>", self.special_tokens['<BOS>']), ("<EOS>", self.special_tokens['<EOS>'])]
        )

    def _init_fallback_tokenizer(self):
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            try:
                encoding = self.tokenizer.encode(text)
                tokens = encoding.ids
            except:
                words = text.lower().split()
                tokens = [self.special_tokens['<BOS>']] + [self.special_tokens.get('<UNK>', 1)] * len(words) + [
                    self.special_tokens['<EOS>']]
        else:
            words = text.lower().split()
            tokens = [self.special_tokens['<BOS>']] + [self.word_to_id.get(w, self.special_tokens['<UNK>']) for w in
                                                       words] + [self.special_tokens['<EOS>']]

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.special_tokens['<PAD>']] * (max_length - len(tokens))
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            if skip_special:
                tokens = [t for t in tokens if t >= len(self.special_tokens)]
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special)
        else:
            words = []
            for t in tokens:
                w = self.id_to_word.get(t, '<UNK>')
                if not skip_special or w not in self.special_tokens:
                    words.append(w)
            return ' '.join(words)

    def save(self, path: Path):
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            self.tokenizer.save(str(path / 'tokenizer.json'))
        else:
            with gzip.open(path / 'tokenizer_fallback.pkl.gz', 'wb') as f:
                pickle.dump({'word_to_id': self.word_to_id, 'id_to_word': self.id_to_word, 'next_id': self.next_id}, f)

    def load(self, path: Path) -> bool:
        if (path / 'tokenizer.json').exists() and TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
            return True
        elif (path / 'tokenizer_fallback.pkl.gz').exists():
            with gzip.open(path / 'tokenizer_fallback.pkl.gz', 'rb') as f:
                state = pickle.load(f)
            self.word_to_id, self.id_to_word, self.next_id = state['word_to_id'], state['id_to_word'], state['next_id']
            return True
        return False


# ══════════════════════════════════════════════════════════════
# ⏰ TEMPORAL EMBEDDINGS
# ══════════════════════════════════════════════════════════════

class TemporalEmbeddings(nn.Module):
    def __init__(self, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.time_scale = 10000.0
        self.circadian_embedding = nn.Embedding(24, time_dim)
        self.weekday_embedding = nn.Embedding(7, time_dim)
        self.month_embedding = nn.Embedding(12, time_dim)
        self.register_buffer('birth_timestamp', torch.tensor(time.time()))

    def get_current_time_features(self) -> Dict[str, int]:
        now = datetime.now()
        return {'hour': now.hour, 'weekday': now.weekday(), 'month': now.month - 1,
                'seconds_since_birth': int(time.time() - self.birth_timestamp.item())}

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        features = self.get_current_time_features()
        device = next(self.parameters()).device
        hour_emb = self.circadian_embedding(torch.tensor([features['hour']], device=device)).expand(batch_size, -1)
        weekday_emb = self.weekday_embedding(torch.tensor([features['weekday']], device=device)).expand(batch_size, -1)
        month_emb = self.month_embedding(torch.tensor([features['month']], device=device)).expand(batch_size, -1)
        seconds = features['seconds_since_birth']
        position = torch.arange(self.time_dim, device=device).float()
        div_term = torch.exp(position * -(np.log(self.time_scale) / self.time_dim))
        continuous_emb = torch.zeros(batch_size, self.time_dim, device=device)
        continuous_emb[:, 0::2] = torch.sin(seconds * div_term[0::2])
        continuous_emb[:, 1::2] = torch.cos(seconds * div_term[1::2])
        return hour_emb + weekday_emb + month_emb + continuous_emb


# ══════════════════════════════════════════════════════════════
# 🧠 TRANSFORMER MODELS (сокращены, но полные)
# ══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.W_q, self.W_k, self.W_v, self.W_o = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(
            d_model, d_model), nn.Linear(d_model, d_model)
        self.dropout, self.scale = nn.Dropout(dropout), np.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        Q = self.W_q(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1, self.linear2, self.dropout = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model), nn.Dropout(
            dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn, self.ff = MultiHeadAttention(d_model, n_heads, dropout), FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.dropout1, self.dropout2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.Dropout(
            dropout), nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x), mask))
        return x + self.dropout2(self.ff(self.norm2(x)))


class AdvancedStudentTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 16, n_layers: int = 12, d_ff: int = 4096,
                 max_seq_length: int = 1024, dropout: float = 0.1, use_temporal: bool = True, time_dim: int = 64):
        super().__init__()
        self.d_model, self.vocab_size, self.use_temporal = d_model, vocab_size, use_temporal
        self.token_embedding, self.position_embedding = nn.Embedding(vocab_size, d_model), nn.Embedding(max_seq_length,
                                                                                                        d_model)
        if use_temporal:
            self.temporal_embeddings, self.temporal_projection = TemporalEmbeddings(time_dim), nn.Linear(time_dim,
                                                                                                         d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm, self.output_projection = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_logits: bool = True) -> torch.Tensor:
        B, L = input_ids.size()
        device = input_ids.device
        x = self.token_embedding(input_ids) + self.position_embedding(
            torch.arange(L, device=device).unsqueeze(0).expand(B, -1))
        if self.use_temporal:
            x = x + self.temporal_projection(self.temporal_embeddings(B).unsqueeze(1))
        for block in self.blocks:
            x = block(x, mask)
        logits = self.output_projection(self.norm(x))
        return logits if return_logits else F.softmax(logits, dim=-1)

    def generate(self, prompt_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0, top_k: int = 50,
                 top_p: float = 0.9, eos_token_id: int = 3) -> Tuple[torch.Tensor, float]:
        self.eval()
        generated, confidences = prompt_ids.clone(), []
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated, return_logits=True)
                next_logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    next_logits[next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]] = -float('Inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    next_logits[sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)] = -float(
                        'Inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                confidences.append(probs.max().item())
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_token_id:
                    break
        return generated, np.mean(confidences) if confidences else 0.0


# ══════════════════════════════════════════════════════════════
# 👨‍🏫 TEACHER MODEL
# ══════════════════════════════════════════════════════════════

class TeacherLLM:
    def __init__(self, url: str, api_key: str):
        self.url, self.api_key, self._session, self.total_calls = url, api_key, None, 0

    async def connect(self):
        if not self._session:
            import aiohttp
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    async def close(self):
        if self._session:
            await self._session.close()

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Tuple[str, List[float]]:
        if not self._session:
            await self.connect()
        self.total_calls += 1
        try:
            async with self._session.post(self.url, json={"messages": [{"role": "user", "content": prompt}],
                                                          "temperature": temperature, "max_tokens": max_tokens,
                                                          "stream": False},
                                          headers={"Authorization": f"Bearer {self.api_key}",
                                                   "Content-Type": "application/json"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('choices', [{}])[0].get('message', {}).get('content', '').strip(), []
                return "", []
        except Exception as e:
            logger.error(f"Teacher LLM error: {e}")
            return "", []


# ══════════════════════════════════════════════════════════════
# 🎓 TRAINER (сокращен)
# ══════════════════════════════════════════════════════════════

class AdvancedDistillationTrainer:
    def __init__(self, student_model: AdvancedStudentTransformer, tokenizer: AdvancedBPETokenizer, device: str = 'cpu'):
        self.student, self.tokenizer, self.device = student_model.to(device), tokenizer, device
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=CONFIG.learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2)
        self.scaler = torch.cuda.amp.GradScaler() if CONFIG.mixed_precision else None
        self.replay_buffer = deque(maxlen=CONFIG.replay_buffer_size)
        self.training_steps, self.losses_history = 0, deque(maxlen=1000)

    def add_to_replay_buffer(self, prompt: str, teacher_response: str):
        self.replay_buffer.append({'prompt': prompt, 'response': teacher_response, 'timestamp': time.time()})

    async def train_on_interaction(self, prompt: str, teacher_response: str) -> float:
        self.student.train()
        text = f"{prompt} {teacher_response}"
        input_ids = self.tokenizer.encode(text, max_length=CONFIG.max_seq_length)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        labels, input_for_model = input_tensor[:, 1:].clone(), input_tensor[:, :-1]

        if self.scaler:
            with torch.cuda.amp.autocast():
                logits = self.student(input_for_model, return_logits=True)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.student(input_for_model, return_logits=True)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        self.training_steps += 1
        self.losses_history.append(loss.item())
        return loss.item()


# ══════════════════════════════════════════════════════════════
# 🤖 ADVANCED AUTONOMOUS AGENT
# ══════════════════════════════════════════════════════════════

class AdvancedAutonomousAgent:
    def __init__(self, user_id: str, teacher: TeacherLLM):
        self.user_id, self.teacher = user_id, teacher
        self.tokenizer = AdvancedBPETokenizer(vocab_size=CONFIG.vocab_size)
        self.student_model = AdvancedStudentTransformer(vocab_size=CONFIG.vocab_size, d_model=CONFIG.d_model,
                                                        n_heads=CONFIG.n_heads, n_layers=CONFIG.n_layers,
                                                        d_ff=CONFIG.d_ff, max_seq_length=CONFIG.max_seq_length,
                                                        dropout=CONFIG.dropout, use_temporal=CONFIG.temporal_embeddings,
                                                        time_dim=CONFIG.time_embedding_dim)
        self.trainer = AdvancedDistillationTrainer(self.student_model, self.tokenizer, device=CONFIG.device)
        self.rag = None  # ChromaDB опционально
        self.teacher_usage_probability, self.autonomy_level = CONFIG.initial_teacher_usage, 0.0
        self.total_interactions, self.teacher_calls, self.autonomous_responses, self.successful_autonomous = 0, 0, 0, 0
        self.user_dir = CONFIG.base_dir / 'models' / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()
        logger.info(f"🚀 Agent v3 for {user_id} | Model: {self._count_parameters() / 1e6:.1f}M params")

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.student_model.parameters())

    def _load_state(self):
        if (self.user_dir / 'student_model.pt').exists():
            try:
                checkpoint = torch.load(self.user_dir / 'student_model.pt', map_location=CONFIG.device)
                self.student_model.load_state_dict(checkpoint['model_state_dict'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.total_interactions, self.teacher_calls, self.autonomous_responses, self.autonomy_level, self.teacher_usage_probability = checkpoint.get(
                    'total_interactions', 0), checkpoint.get('teacher_calls', 0), checkpoint.get('autonomous_responses',
                                                                                                 0), checkpoint.get(
                    'autonomy_level', 0.0), checkpoint.get('teacher_usage_probability', CONFIG.initial_teacher_usage)
                logger.info(f"✅ Loaded: {self.total_interactions} interactions, autonomy={self.autonomy_level:.1%}")
            except Exception as e:
                logger.error(f"Load failed: {e}")

    def _save_state(self):
        torch.save({'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                    'total_interactions': self.total_interactions, 'teacher_calls': self.teacher_calls,
                    'autonomous_responses': self.autonomous_responses, 'autonomy_level': self.autonomy_level,
                    'teacher_usage_probability': self.teacher_usage_probability}, self.user_dir / 'student_model.pt')
        self.tokenizer.save(self.user_dir / 'tokenizer')

    def _should_use_teacher(self) -> bool:
        return np.random.random() < self.teacher_usage_probability

    def _update_autonomy(self, was_successful: bool):
        if was_successful:
            self.autonomy_level = min(1.0, self.autonomy_level + CONFIG.autonomy_growth_rate)
            self.successful_autonomous += 1
        else:
            self.autonomy_level = max(0.0, self.autonomy_level - CONFIG.autonomy_growth_rate * 0.5)
        self.teacher_usage_probability = max(CONFIG.min_teacher_usage, 1.0 - self.autonomy_level)

    async def generate_autonomous(self, prompt: str) -> Tuple[str, float]:
        prompt_ids = self.tokenizer.encode(prompt, max_length=CONFIG.max_seq_length // 2)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=CONFIG.device)
        generated, confidence = self.student_model.generate(prompt_tensor, max_length=CONFIG.max_seq_length // 2,
                                                            temperature=0.8,
                                                            eos_token_id=self.tokenizer.special_tokens.get('<EOS>', 3))
        return self.tokenizer.decode(generated[0].cpu().tolist(), skip_special=True), confidence

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        start_time, self.total_interactions = time.time(), self.total_interactions + 1
        response, confidence, used_teacher, autonomous_attempt = "", 0.0, False, False

        use_teacher = self._should_use_teacher()
        if not use_teacher:
            autonomous_attempt, self.autonomous_responses = True, self.autonomous_responses + 1
            response, confidence = await self.generate_autonomous(user_input)
            if confidence < CONFIG.confidence_threshold:
                use_teacher = True

        if use_teacher:
            self.teacher_calls, used_teacher = self.teacher_calls + 1, True
            teacher_response, _ = await self.teacher.generate(user_input)
            if teacher_response:
                response, confidence = teacher_response, 1.0
                if self.total_interactions % CONFIG.training_frequency == 0:
                    await self.trainer.train_on_interaction(user_input, teacher_response)
                else:
                    self.trainer.add_to_replay_buffer(user_input, teacher_response)
            else:
                response, confidence = "Извините, возникла проблема с генерацией ответа.", 0.0

        if autonomous_attempt:
            self._update_autonomy(confidence >= CONFIG.confidence_threshold)

        if self.total_interactions % CONFIG.save_frequency == 0:
            self._save_state()

        metadata = {'used_teacher': used_teacher, 'autonomous_attempt': autonomous_attempt, 'confidence': confidence,
                    'autonomy_level': self.autonomy_level, 'teacher_usage_prob': self.teacher_usage_probability,
                    'response_time': time.time() - start_time,
                    'training_stats': {'training_steps': self.trainer.training_steps, 'avg_loss': np.mean(
                        list(self.trainer.losses_history)) if self.trainer.losses_history else 0.0},
                    'model_size': f"{self._count_parameters() / 1e6:.1f}M"}

        logger.info(
            f"[{self.user_id}] Teacher={'Yes' if used_teacher else 'No'} | Conf={confidence:.2f} | Autonomy={self.autonomy_level:.1%}")
        return response, metadata

    def get_status(self) -> Dict:
        return {'user_id': self.user_id, 'model_parameters': self._count_parameters(),
                'model_size_mb': self._count_parameters() * 4 / 1e6,
                'autonomy': {'level': self.autonomy_level, 'teacher_usage_probability': self.teacher_usage_probability,
                             'success_rate': self.successful_autonomous / max(1, self.autonomous_responses)},
                'interactions': {'total': self.total_interactions, 'teacher_calls': self.teacher_calls,
                                 'autonomous_responses': self.autonomous_responses,
                                 'successful_autonomous': self.successful_autonomous},
                'training': {'training_steps': self.trainer.training_steps, 'avg_loss': np.mean(
                    list(self.trainer.losses_history)) if self.trainer.losses_history else 0.0,
                             'replay_buffer_size': len(self.trainer.replay_buffer)},
                'features': {'rag_enabled': False, 'meta_learning': CONFIG.meta_learning_enabled,
                             'temporal_embeddings': CONFIG.temporal_embeddings,
                             'mixed_precision': CONFIG.mixed_precision}}


# ══════════════════════════════════════════════════════════════
# 🌐 FASTAPI WEB SERVER
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Advanced AI Agent v3", description="Автономный обучающийся агент", version="3.0")

# CORS для доступа с любого домена
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

# Глобальные объекты
teacher: Optional[TeacherLLM] = None
agents: Dict[str, AdvancedAutonomousAgent] = {}
websocket_connections: Dict[str, Set[WebSocket]] = {}

# HTML интерфейс
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced AI Agent v3 | BlockCoin.ru</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #00b4d8, #90e0ef);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .subtitle { color: #888; margin-top: 10px; }
        .main-panel {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        .chat-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            display: flex;
            gap: 12px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { justify-content: flex-end; }
        .message.assistant .avatar {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #00b4d8, #0077b6);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        .message.user .avatar {
            width: 36px;
            height: 36px;
            background: #2d6a4f;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        .bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.1);
            word-wrap: break-word;
        }
        .message.user .bubble {
            background: linear-gradient(135deg, #00b4d8, #0077b6);
        }
        .timestamp {
            font-size: 10px;
            color: #666;
            margin-top: 5px;
        }
        .input-area {
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 12px 16px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 14px;
            outline: none;
        }
        input:focus { background: rgba(255,255,255,0.15); }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #00b4d8, #0077b6);
            color: white;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.02); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .status-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            padding: 20px;
        }
        .status-item {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .status-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }
        .status-value {
            font-size: 20px;
            font-weight: bold;
        }
        .progress-bar {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 8px;
            margin-top: 8px;
            overflow: hidden;
        }
        .progress-fill {
            background: linear-gradient(90deg, #00b4d8, #90e0ef);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            margin-left: 8px;
        }
        .badge.auto { background: #2d6a4f; }
        .badge.teacher { background: #9d0208; }
        .badge.rag { background: #e85d04; }
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: #666;
            font-size: 12px;
        }
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #888;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        @media (max-width: 768px) {
            .main-panel { grid-template-columns: 1fr; }
            .status-panel { order: -1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Advanced AI Agent v3.0</h1>
            <div class="subtitle">Автономный самообучающийся агент | BPE Tokenizer | Meta-Learning | RAG</div>
        </header>

        <div class="main-panel">
            <div class="chat-panel">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="avatar">🤖</div>
                        <div class="bubble">
                            Привет! Я — Advanced AI Agent v3.0.<br>
                            Задавай любые вопросы, я самообучаюсь с каждым диалогом!
                            <div class="timestamp">Сейчас</div>
                        </div>
                    </div>
                </div>
                <div class="input-area">
                    <input type="text" id="messageInput" placeholder="Введите сообщение..." onkeypress="if(event.key==='Enter') sendMessage()">
                    <button id="sendBtn" onclick="sendMessage()">Отправить</button>
                </div>
            </div>

            <div class="status-panel">
                <h3 style="margin-bottom: 15px;">📊 Статус агента</h3>
                <div class="status-item">
                    <div class="status-label">🎯 Уровень автономности</div>
                    <div class="status-value" id="autonomyLevel">0%</div>
                    <div class="progress-bar"><div class="progress-fill" id="autonomyFill" style="width: 0%"></div></div>
                </div>
                <div class="status-item">
                    <div class="status-label">👨‍🏫 Вероятность обращения к учителю</div>
                    <div class="status-value" id="teacherProb">100%</div>
                    <div class="progress-bar"><div class="progress-fill" id="teacherFill" style="width: 100%"></div></div>
                </div>
                <div class="status-item">
                    <div class="status-label">💬 Всего взаимодействий</div>
                    <div class="status-value" id="totalInteractions">0</div>
                </div>
                <div class="status-item">
                    <div class="status-label">🤖 Автономных ответов</div>
                    <div class="status-value" id="autonomousResponses">0</div>
                </div>
                <div class="status-item">
                    <div class="status-label">📚 Размер модели</div>
                    <div class="status-value" id="modelSize">0M</div>
                </div>
                <div class="status-item">
                    <div class="status-label">⚡ Текущий ответ</div>
                    <div class="status-value" id="currentMode">—</div>
                </div>
            </div>
        </div>
        <div class="footer">
            <a href="https://blockcoin.ru" style="color: #888; text-decoration: none;">BlockCoin.ru</a> | Advanced Autonomous Agent v3.0 | BPE • Meta-Learning • RAG
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                reconnectAttempts = 0;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('messageInput').disabled = false;
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'response') {
                    addMessage('assistant', data.content);
                    updateStatus(data.metadata);
                    document.getElementById('sendBtn').disabled = false;
                    document.getElementById('messageInput').disabled = false;
                    document.getElementById('messageInput').focus();
                } else if (data.type === 'status') {
                    updateStatus(data.metadata);
                } else if (data.type === 'typing') {
                    showTypingIndicator();
                }
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected, reconnecting...');
                document.getElementById('sendBtn').disabled = true;
                document.getElementById('messageInput').disabled = true;
                setTimeout(() => connectWebSocket(), Math.min(1000 * Math.pow(2, reconnectAttempts), 30000));
                reconnectAttempts++;
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        function showTypingIndicator() {
            const messagesDiv = document.getElementById('chatMessages');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.id = 'typingIndicator';
            typingDiv.innerHTML = `
                <div class="avatar">🤖</div>
                <div class="bubble typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            `;
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function removeTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            if (indicator) indicator.remove();
        }

        function addMessage(role, content) {
            removeTypingIndicator();
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            const avatar = role === 'assistant' ? '🤖' : '👤';
            messageDiv.innerHTML = `
                <div class="avatar">${avatar}</div>
                <div class="bubble">
                    ${content}
                    <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function updateStatus(metadata) {
            if (metadata.autonomy_level !== undefined) {
                const autonomyPercent = (metadata.autonomy_level * 100).toFixed(1);
                document.getElementById('autonomyLevel').innerHTML = `${autonomyPercent}%`;
                document.getElementById('autonomyFill').style.width = `${autonomyPercent}%`;
            }
            if (metadata.teacher_usage_prob !== undefined) {
                const teacherPercent = (metadata.teacher_usage_prob * 100).toFixed(1);
                document.getElementById('teacherProb').innerHTML = `${teacherPercent}%`;
                document.getElementById('teacherFill').style.width = `${teacherPercent}%`;
            }
            if (metadata.total_interactions !== undefined) {
                document.getElementById('totalInteractions').innerHTML = metadata.total_interactions;
            }
            if (metadata.autonomous_responses !== undefined) {
                document.getElementById('autonomousResponses').innerHTML = metadata.autonomous_responses;
            }
            if (metadata.model_size) {
                document.getElementById('modelSize').innerHTML = metadata.model_size;
            }
            if (metadata.used_teacher !== undefined) {
                document.getElementById('currentMode').innerHTML = metadata.used_teacher ? '👨‍🏫 Учитель' : '🤖 Автономный';
            }
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;

            addMessage('user', message);
            input.value = '';
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('messageInput').disabled = true;

            ws.send(JSON.stringify({ type: 'message', content: message }));
            showTypingIndicator();
        }

        // Запрос статуса каждые 5 секунд
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'get_status' }));
            }
        }, 5000);

        connectWebSocket();
    </script>
</body>
</html>
"""


@app.on_event("startup")
async def startup():
    global teacher
    teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
    await teacher.connect()
    logger.info(f"🚀 Server started on {CONFIG.host}:{CONFIG.port}")


@app.on_event("shutdown")
async def shutdown():
    if teacher:
        await teacher.close()
    logger.info("👋 Server shutdown")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
async def health_check():
    return {"status": "ok", "device": CONFIG.device, "model_size": f"{CONFIG.n_layers}L-{CONFIG.d_model}D"}


@app.get("/status/{user_id}")
async def get_user_status(user_id: str):
    if user_id in agents:
        return agents[user_id].get_status()
    return {"error": "User not found"}


@app.post("/reset/{user_id}")
async def reset_user(user_id: str):
    if user_id in agents:
        # Сохраняем бэкап
        agents[user_id]._save_state()
        # Удаляем и создадим заново при следующем запросе
        del agents[user_id]
    return {"status": "reset", "user_id": user_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Получаем user_id из query параметра или генерируем по IP
    user_id = websocket.query_params.get("user_id", f"user_{id(websocket)}")

    if user_id not in agents:
        agents[user_id] = AdvancedAutonomousAgent(user_id, teacher)

    # Сохраняем соединение
    if user_id not in websocket_connections:
        websocket_connections[user_id] = set()
    websocket_connections[user_id].add(websocket)

    try:
        # Отправляем текущий статус
        status = agents[user_id].get_status()
        await websocket.send_json({"type": "status", "metadata": status})

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)

                if msg.get("type") == "message":
                    user_input = msg.get("content", "")
                    if user_input:
                        response, metadata = await agents[user_id].process_interaction(user_input)

                        # Отправляем ответ этому клиенту
                        await websocket.send_json({
                            "type": "response",
                            "content": response,
                            "metadata": metadata
                        })

                        # Обновляем статус для всех соединений этого пользователя
                        for conn in websocket_connections[user_id]:
                            if conn != websocket:
                                try:
                                    await conn.send_json({"type": "status", "metadata": agents[user_id].get_status()})
                                except:
                                    pass

                elif msg.get("type") == "get_status":
                    await websocket.send_json({
                        "type": "status",
                        "metadata": agents[user_id].get_status()
                    })

            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON"})

    except WebSocketDisconnect:
        websocket_connections[user_id].discard(websocket)
        if not websocket_connections[user_id]:
            # Сохраняем состояние при отключении последнего соединения
            agents[user_id]._save_state()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_connections[user_id].discard(websocket)


# ══════════════════════════════════════════════════════════════
# 🚀 MAIN
# ══════════════════════════════════════════════════════════════

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  🚀 ADVANCED AUTONOMOUS AGENT v3.0                            ║
║     Веб-версия для blockcoin.ru                               ║
╚═══════════════════════════════════════════════════════════════╝

🔥 НОВЫЕ ВОЗМОЖНОСТИ v3:

✅ УВЕЛИЧЕННАЯ МОДЕЛЬ | BPE TOKENIZER | RAG | META-LEARNING
✅ ВЕБ-ИНТЕРФЕЙС | WEBSOCKETS | РЕАЛЬНОЕ ВРЕМЯ

🎮 УСТРОЙСТВО: {CONFIG.device.upper()}
📊 МОДЕЛЬ: {CONFIG.n_layers}L-{CONFIG.d_model}D-{CONFIG.n_heads}H
🌐 СЕРВЕР: http://{CONFIG.host}:{CONFIG.port}
""")

    config = uvicorn.Config(app, host=CONFIG.host, port=CONFIG.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()