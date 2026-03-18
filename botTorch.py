#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID COGNITIVE AGI v4.0 — ИСПРАВЛЕННАЯ ВЕРСИЯ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ИСПРАВЛЕНИЯ:
• Увеличенные таймауты для Telegram и LLM
• Retry-логика для сетевых запросов
• Graceful error handling
• Connection pooling оптимизация
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
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, field, asdict
import gzip
import pickle
import time
import re
import math
import hashlib
import importlib.util
import copy
import traceback
from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)
from telegram.error import TimedOut, NetworkError, RetryAfter

# ──────────────────────────────────────────────────────────────
# 📦 ЗАВИСИМОСТИ (Optional)
# ──────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available. Install: pip install chromadb")

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


# ──────────────────────────────────────────────────────────────
# ⚙️ КОНФИГУРАЦИЯ (С УВЕЛИЧЕННЫМИ ТАЙМАУТАМИ)
# ──────────────────────────────────────────────────────────────
@dataclass
class HybridConfig:
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🕐 ТАЙМАУТЫ (УВЕЛИЧЕНЫ)
    telegram_timeout: int = 60  # Было 30
    telegram_pool_timeout: int = 30
    lm_studio_timeout: int = 120  # Было 60
    lm_studio_connect_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0

    # 🧠 Модель (Transformer)
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    auto_scale_model: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = torch.cuda.is_available()

    # 📚 Память (Cognitive)
    episodic_memory_size: int = 5000
    semantic_memory_size: int = 2000
    working_memory_size: int = 15
    memory_consolidation_threshold: float = 0.6
    forgetting_curve_factor: float = 0.1
    rag_enabled: bool = CHROMADB_AVAILABLE
    embedding_dim: int = 128

    # 🎯 Обучение
    learning_rate: float = 5e-5
    meta_learning_enabled: bool = True
    distillation_temperature: float = 2.0
    training_frequency: int = 5
    save_frequency: int = 50

    # 🔧 Самомодификация
    self_modification_enabled: bool = True
    module_creation_threshold: float = 0.6
    max_custom_modules: int = 20

    # 🌡️ Соматосенсорика
    internal_monitoring_enabled: bool = True
    anomaly_detection_threshold: float = 2.0

    # 📂 Пути
    version: str = "4.0-HYBRID-FIXED"
    base_dir: Path = Path(os.getenv('BASE_DIR', 'hybrid_agi_v4'))

    def __post_init__(self):
        subdirs = [
            'models', 'memory', 'logs', 'checkpoints',
            'rag', 'tokenizer', 'modules/custom', 'modules/tests'
        ]
        for subdir in subdirs:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        if self.auto_scale_model and self.device == 'cuda':
            self._auto_scale_to_gpu()

    def _auto_scale_to_gpu(self):
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            estimated_memory_gb = (self.vocab_size * self.d_model * 2 +
                                   self.n_layers * (4 * self.d_model * self.d_model)) * 4 / 1e9 * 1.5
            if estimated_memory_gb > gpu_memory_gb * 0.7:
                scale_factor = (gpu_memory_gb * 0.7) / estimated_memory_gb
                self.d_model = int(self.d_model * scale_factor ** 0.5)
                self.n_layers = max(4, int(self.n_layers * scale_factor ** 0.25))
                self.d_model = (self.d_model // self.n_heads) * self.n_heads
        except Exception:
            pass


CONFIG = HybridConfig()


# ──────────────────────────────────────────────────────────────
# 📊 LOGGING
# ──────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger('Hybrid_AGI_v4')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    log_file = CONFIG.base_dir / 'logs' / f'agi_v4_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ──────────────────────────────────────────────────────────────
# 🔤 BPE TOKENIZER
# ──────────────────────────────────────────────────────────────
class HybridBPETokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer: Optional[Tokenizer] = None
        self.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        if not TOKENIZERS_AVAILABLE:
            self._init_fallback()
        else:
            self._init_bpe()

    def _init_bpe(self):
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.add_special_tokens(list(self.special_tokens.keys()))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        logger.info("✅ BPE Tokenizer initialized")

    def _init_fallback(self):
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)
        logger.warning("⚠️ Using fallback tokenizer")

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            try:
                encoding = self.tokenizer.encode(text)
                tokens = encoding.ids
            except:
                tokens = [self.special_tokens['<BOS>']] + [1] * len(text.split()) + [self.special_tokens['<EOS>']]
        else:
            words = text.lower().split()
            tokens = [self.special_tokens['<BOS>']]
            for word in words:
                tokens.append(self.word_to_id.get(word, 1))
            tokens.append(self.special_tokens['<EOS>'])

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            if skip_special:
                tokens = [t for t in tokens if t >= len(self.special_tokens)]
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special)
        else:
            words = [self.id_to_word.get(t, '<UNK>') for t in tokens
                     if skip_special and t not in self.special_tokens.values()]
            return ' '.join(words)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            self.tokenizer.save(str(path / 'tokenizer.json'))
        else:
            with gzip.open(path / 'tokenizer_fallback.pkl.gz', 'wb') as f:
                pickle.dump({'word_to_id': self.word_to_id, 'id_to_word': self.id_to_word}, f)

    def load(self, path: Path) -> bool:
        if (path / 'tokenizer.json').exists() and TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
            return True
        elif (path / 'tokenizer_fallback.pkl.gz').exists():
            with gzip.open(path / 'tokenizer_fallback.pkl.gz', 'rb') as f:
                state = pickle.load(f)
                self.word_to_id = state['word_to_id']
                self.id_to_word = state['id_to_word']
            return True
        return False


# ──────────────────────────────────────────────────────────────
# 🧠 TRANSFORMER MODEL
# ──────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(attention), V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class HybridTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()
        logger.info(f"🧠 Transformer: {sum(p.numel() for p in self.parameters()) / 1e6:.1f}M params")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        x = self.token_embedding(input_ids) + self.position_embedding(
            torch.arange(seq_length, device=input_ids.device)
        )
        for block in self.blocks:
            x = block(x, mask)
        return self.output_projection(self.norm(x))

    def generate(self, prompt_ids: torch.Tensor, max_length: int = 50,
                 temperature: float = 0.8, eos_token_id: int = 3) -> Tuple[torch.Tensor, float]:
        self.eval()
        generated = prompt_ids.clone()
        confidences = []
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                confidences.append(probs.max().item())
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_token_id:
                    break
        return generated, np.mean(confidences) if confidences else 0.0


# ──────────────────────────────────────────────────────────────
# 📚 COGNITIVE MEMORY
# ──────────────────────────────────────────────────────────────
@dataclass
class EpisodicMemory:
    content: str
    timestamp: float
    importance: float = 0.5
    emotional_valence: float = 0.0
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(128))


class CognitiveMemorySystem:
    def __init__(self, embedding_dim: int, embed_func: Callable[[str], np.ndarray]):
        self.embedding_dim = embedding_dim
        self.embed_func = embed_func
        self.episodic: List[EpisodicMemory] = []
        self.semantic: Dict[str, str] = {}
        self.procedural: Dict[str, str] = {}
        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)

    def add_episode(self, content: str, importance: float, emotional_valence: float):
        emb = self.embed_func(content)
        self.episodic.append(EpisodicMemory(
            content=content,
            timestamp=time.time(),
            importance=importance,
            emotional_valence=emotional_valence,
            embedding=emb
        ))
        self.working_memory.append(content)
        if len(self.episodic) > CONFIG.episodic_memory_size:
            self.episodic.sort(key=lambda x: x.importance, reverse=True)
            self.episodic = self.episodic[:CONFIG.episodic_memory_size]

    def add_concept(self, concept: str, definition: str):
        self.semantic[concept] = definition

    def add_skill(self, skill_name: str, code_ref: str):
        self.procedural[skill_name] = code_ref

    def get_context(self, query: str, top_k: int = 3) -> str:
        query_emb = self.embed_func(query)
        scored = []
        for mem in self.episodic:
            sim = np.dot(mem.embedding, query_emb) / (
                    np.linalg.norm(mem.embedding) * np.linalg.norm(query_emb) + 1e-8
            )
            scored.append((mem, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        context = "=== Memory ===\n"
        for mem, score in scored[:top_k]:
            if score > 0.3:
                context += f"[{score:.2f}] {mem.content}\n"
        return context if len(context) > 15 else ""

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb') as f:
            pickle.dump({
                'episodic': self.episodic,
                'semantic': self.semantic,
                'procedural': self.procedural
            }, f)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)
                self.episodic = state.get('episodic', [])
                self.semantic = state.get('semantic', {})
                self.procedural = state.get('procedural', {})
            return True
        except:
            return False


# ──────────────────────────────────────────────────────────────
# 🌡️ SOMATOSENSORY SYSTEM
# ──────────────────────────────────────────────────────────────
class SomatosensorySystem:
    def __init__(self):
        self.state_history: deque = deque(maxlen=50)
        self.current_quality = 0.5
        self.emotional_valence = 0.0

    def update(self, quality: float, emotion: float):
        self.current_quality = self.current_quality * 0.8 + quality * 0.2
        self.emotional_valence = self.emotional_valence * 0.9 + emotion * 0.1
        self.state_history.append({'q': quality, 't': time.time()})

    def get_state(self) -> str:
        if self.current_quality > 0.7:
            return "Уверенное"
        if self.current_quality < 0.3:
            return "Неопределенное"
        return "Нормальное"

    def detect_anomaly(self) -> bool:
        if len(self.state_history) < 10:
            return False
        vals = [s['q'] for s in self.state_history]
        return abs(vals[-1] - np.mean(vals)) > CONFIG.anomaly_detection_threshold * np.std(vals)


# ──────────────────────────────────────────────────────────────
# 🔧 SELF-MODIFICATION
# ──────────────────────────────────────────────────────────────
class SelfModificationEngine:
    def __init__(self, llm_client, modules_dir: Path):
        self.llm = llm_client
        self.modules_dir = modules_dir
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_modules: Dict[str, Any] = {}
        self._load_existing()

    def _load_existing(self):
        for f in self.modules_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(f.stem, f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.loaded_modules[f.stem] = mod
            except:
                pass

    async def create_module(self, task: str) -> Optional[str]:
        prompt = f"Создай Python модуль для: {task}. Только код, без markdown. Функция execute()."
        code = await self.llm.generate(prompt, temperature=0.3)
        code = re.sub(r'```python|```', '', code).strip()
        if not code:
            return None
        module_name = f"mod_{int(time.time())}"
        path = self.modules_dir / f"{module_name}.py"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.loaded_modules[module_name] = mod
            return module_name
        except:
            return None


# ──────────────────────────────────────────────────────────────
# 🔗 TEACHER LLM (С RETRY И ТАЙМАУТАМИ)
# ──────────────────────────────────────────────────────────────
class TeacherLLM:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        if not self._session:
            # ✅ Увеличенные таймауты
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

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """✅ С retry-логикой"""
        await self.connect()

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
                            "temperature": 0.7,
                            "max_tokens": 1000
                        },
                        headers={
                            "Authorization": f"Bearer {self.key}",
                            "Content-Type": "application/json"
                        }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    else:
                        logger.warning(f"LLM error: {resp.status}")
            except asyncio.TimeoutError as e:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{CONFIG.max_retries}): {e}")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay * (attempt + 1))
                continue
            except Exception as e:
                logger.error(f"LLM Error (attempt {attempt + 1}): {e}")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay * (attempt + 1))
                continue

        return ""  # Возвращаем пустую строку после всех попыток


# ──────────────────────────────────────────────────────────────
# 🎓 TRAINER
# ──────────────────────────────────────────────────────────────
class HybridTrainer:
    def __init__(self, model: HybridTransformer, tokenizer: HybridBPETokenizer, device: str):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG.learning_rate
        )
        self.scaler = torch.cuda.amp.GradScaler() if CONFIG.mixed_precision else None
        self.replay_buffer: deque = deque(maxlen=1000)

    async def train_step(self, prompt: str, response: str):
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
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.replay_buffer.append((prompt, response))
        return loss.item()


# ──────────────────────────────────────────────────────────────
# 🤖 HYBRID AGENT
# ──────────────────────────────────────────────────────────────
class HybridAutonomousAgent:
    def __init__(self, user_id: str, teacher: TeacherLLM):
        self.user_id = user_id
        self.teacher = teacher
        self.tokenizer = HybridBPETokenizer(CONFIG.vocab_size)
        self.model = HybridTransformer(
            CONFIG.vocab_size, CONFIG.d_model, CONFIG.n_heads,
            CONFIG.n_layers, CONFIG.d_ff, CONFIG.max_seq_length, CONFIG.dropout
        )
        self.trainer = HybridTrainer(self.model, self.tokenizer, CONFIG.device)

        # Memory & Cognition
        self.memory = CognitiveMemorySystem(CONFIG.embedding_dim, self._simple_embed)
        self.soma = SomatosensorySystem()
        self.self_mod = SelfModificationEngine(
            teacher,
            CONFIG.base_dir / 'modules' / 'custom' / user_id
        )

        self.user_dir = CONFIG.base_dir / 'models' / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()
        logger.info(f"🚀 Agent v4.0 created for {user_id}")

    def _simple_embed(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(text, max_length=50)
        emb = np.zeros(CONFIG.embedding_dim)
        for i, t in enumerate(tokens):
            if i < CONFIG.embedding_dim:
                emb[i] = t / CONFIG.vocab_size
        return emb

    def _load_state(self):
        model_path = self.user_dir / 'model.pt'
        if model_path.exists():
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=CONFIG.device)
                )
                logger.info("✅ Model loaded")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        mem_path = self.user_dir / 'memory.pkl.gz'
        self.memory.load(mem_path)
        if TOKENIZERS_AVAILABLE:
            self.tokenizer.load(self.user_dir)

    def _save_state(self):
        torch.save(self.model.state_dict(), self.user_dir / 'model.pt')
        self.memory.save(self.user_dir / 'memory.pkl.gz')
        self.tokenizer.save(self.user_dir)
        logger.debug("💾 State saved")

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        start = time.time()

        # 1. Context
        mem_context = self.memory.get_context(user_input)
        full_prompt = f"{mem_context}\nUser: {user_input}\nAssistant:" if mem_context else f"User: {user_input}\nAssistant:"

        # 2. Decision (Model vs Teacher)
        use_teacher = np.random.random() > self.soma.current_quality or len(user_input) > 100
        response = ""
        confidence = 0.0

        if not use_teacher:
            ids = self.tokenizer.encode(full_prompt, max_length=CONFIG.max_seq_length // 2)
            tensor = torch.tensor([ids], dtype=torch.long, device=CONFIG.device)
            gen, conf = self.model.generate(tensor, max_length=100)
            response = self.tokenizer.decode(gen[0].cpu().tolist())
            confidence = conf
        else:
            response = await self.teacher.generate(
                full_prompt,
                system_prompt="Ты полезный ассистент."
            )
            confidence = 1.0 if response else 0.5

        # 3. Learning
        if use_teacher and response:
            await self.trainer.train_step(user_input, response)

        # 4. Memory & Soma
        emotion = (confidence - 0.5) * 2
        self.memory.add_episode(f"User: {user_input}\nAI: {response}", confidence, emotion)
        self.soma.update(confidence, emotion)

        # 5. Self-Mod Check
        if CONFIG.self_modification_enabled and confidence < 0.4 and np.random.random() < 0.1:
            await self.self_mod.create_module(user_input)

        # 6. Save
        if len(self.memory.episodic) % CONFIG.save_frequency == 0:
            self._save_state()

        metadata = {
            'quality': confidence,
            'state': self.soma.get_state(),
            'time': time.time() - start,
            'memory_count': len(self.memory.episodic)
        }
        return response, metadata


# ──────────────────────────────────────────────────────────────
# 🤖 TELEGRAM BOT (С ОБРАБОТКОЙ ТАЙМАУТОВ)
# ──────────────────────────────────────────────────────────────
class HybridBot:
    def __init__(self):
        self.teacher: Optional[TeacherLLM] = None
        self.agents: Dict[str, HybridAutonomousAgent] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        """✅ Инициализация бота с увеличенными таймаутами"""
        self.teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.teacher.connect()

        defaults = Defaults(parse_mode='HTML')

        # ✅ Увеличенные таймауты для Telegram
        self._app = Application.builder().token(token).defaults(defaults).build()

        # Настраиваем bot с таймаутами
        self._app.bot._request = self._app.bot._request._replace(
            read_timeout=CONFIG.telegram_timeout,
            connect_timeout=CONFIG.telegram_timeout,
            write_timeout=CONFIG.telegram_timeout,
            pool_timeout=CONFIG.telegram_pool_timeout
        )

        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_msg
        ))

        for cmd, h in [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('reset', self._cmd_reset)
        ]:
            self._app.add_handler(CommandHandler(cmd, h))

        logger.info("🤖 Bot v4.0 initialized")

    async def _get_agent(self, user_id: str) -> HybridAutonomousAgent:
        if user_id not in self.agents:
            self.agents[user_id] = HybridAutonomousAgent(user_id, self.teacher)
        return self.agents[user_id]

    async def _handle_msg(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """✅ С обработкой таймаутов и retry"""
        if not update.message or not update.effective_user:
            return

        user_id = str(update.effective_user.id)

        for attempt in range(CONFIG.max_retries):
            try:
                await context.bot.send_chat_action(
                    update.effective_chat.id,
                    "typing"
                )

                agent = await self._get_agent(user_id)
                resp, meta = await agent.process_interaction(update.message.text)

                footer = f"\n<i>🧠 Q:{meta['quality']:.0%} | {meta['state']} | ⚡{meta['time']:.1f}s</i>"

                await update.message.reply_text(
                    resp + footer,
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                    read_timeout=CONFIG.telegram_timeout,
                    write_timeout=CONFIG.telegram_timeout,
                    connect_timeout=CONFIG.telegram_timeout
                )
                return  # ✅ Успех

            except (TimedOut, NetworkError) as e:
                logger.warning(f"Telegram timeout (attempt {attempt + 1}): {e}")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay * (attempt + 1))
                    continue
                else:
                    await update.message.reply_text(
                        "⚠️ <b>Таймаут соединения</b>\nПовторите запрос.",
                        parse_mode='HTML'
                    )
                    return

            except RetryAfter as e:
                logger.warning(f"Telegram rate limit: {e.retry_after}s")
                await asyncio.sleep(e.retry_after)
                continue

            except Exception as e:
                logger.exception(f"Error from {user_id}")
                if attempt < CONFIG.max_retries - 1:
                    await asyncio.sleep(CONFIG.retry_delay)
                    continue
                else:
                    await update.message.reply_text("⚠️ Ошибка обработки")
                    return

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 <b>HYBRID AGI v4.0</b>\nСинтез Трансформера и Когнитивной Архитектуры.\nПишите мне!"
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        agent = await self._get_agent(str(update.effective_user.id))
        status = (
            f"📊 <b>Статус</b>\n"
            f"Память: {len(agent.memory.episodic)}\n"
            f"Качество: {agent.soma.current_quality:.1%}\n"
            f"Модулей: {len(agent.self_mod.loaded_modules)}"
        )
        await update.message.reply_text(status)

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if context.args and context.args[0] == 'confirm':
            uid = str(update.effective_user.id)
            if uid in self.agents:
                self.agents[uid]._save_state()
                del self.agents[uid]
            await update.message.reply_text("✅ Сброшено")
        else:
            await update.message.reply_text("⚠️ Используйте /reset confirm")

    async def run(self):
        """✅ Запуск бота"""
        if not self._app:
            logger.error("❌ Bot not initialized! Call initialize() first")
            return

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot Running")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.shutdown()

    async def shutdown(self):
        """✅ Остановка бота"""
        logger.info("🛑 Shutting down...")
        for a in self.agents.values():
            a._save_state()
        if self.teacher:
            await self.teacher.close()
        if self._app:
            await self._app.stop()
        logger.info("✅ Shutdown complete")


# ──────────────────────────────────────────────────────────────
# 🚀 MAIN
# ──────────────────────────────────────────────────────────────
async def main():
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🧠 HYBRID COGNITIVE AGI v4.0 — FIXED                     ║
    ║     Transformer + Memory + Self-Mod                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN missing!")
        return 1

    bot = HybridBot()
    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Stop signal received")
    except Exception as e:
        logger.critical(f"❌ Critical error: {e}", exc_info=True)
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