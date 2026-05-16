#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID COGNITIVE AGI v4.2 — ЛОКАЛЬНАЯ ВЕРСИЯ С GUI (PyQt6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Фоновая петля мышления
✅ Проверка истинности (Truthfulness)
✅ Интенциональность (psi-уровень)
✅ Самомодификация с песочницей
✅ Полностью локальный чат
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
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import pickle
import gzip
import time
import re
import hashlib
import importlib.util
import traceback
import ast
from dotenv import load_dotenv

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QTabWidget, QPlainTextEdit
)
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QFont, QTextCursor
import qasync

load_dotenv()

# ──────────────────────────────────────────────────────────────
# ⚙️ КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────
@dataclass
class HybridConfig:
    # API
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # Таймауты
    lm_studio_timeout: int = 120
    lm_studio_connect_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0

    # Модель Transformer
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = torch.cuda.is_available()

    # Память
    episodic_memory_size: int = 5000
    semantic_memory_size: int = 2000
    working_memory_size: int = 15
    embedding_dim: int = 384  # для sentence-transformers (all-MiniLM-L6-v2)

    # Обучение
    learning_rate: float = 5e-5
    training_frequency: int = 5
    save_frequency: int = 50

    # Самомодификация
    self_modification_enabled: bool = True
    module_creation_threshold: float = 0.6
    max_custom_modules: int = 20

    # Соматосенсорика
    internal_monitoring_enabled: bool = True

    # Пути
    version: str = "4.2-LOCAL"
    base_dir: Path = Path(os.getenv('BASE_DIR', 'hybrid_agi_v4'))

    def __post_init__(self):
        subdirs = ['models', 'memory', 'logs', 'modules/custom', 'tokenizer']
        for subdir in subdirs:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

CONFIG = HybridConfig()

# ──────────────────────────────────────────────────────────────
# 📊 LOGGING
# ──────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger('Hybrid_AGI_Local')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    log_file = CONFIG.base_dir / 'logs' / f'agi_{datetime.now():%Y%m%d}.log'
    log_file.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

# ──────────────────────────────────────────────────────────────
# 🔤 BPE TOKENIZER (упрощённый fallback)
# ──────────────────────────────────────────────────────────────
class HybridBPETokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        words = text.lower().split()
        tokens = [self.special_tokens['<BOS>']]
        for word in words:
            tokens.append(self.word_to_id.get(word, self.special_tokens['<UNK>']))
        tokens.append(self.special_tokens['<EOS>'])
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.special_tokens['<PAD>']] * (max_length - len(tokens))
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        words = []
        for t in tokens:
            if skip_special and t in self.special_tokens.values():
                continue
            words.append(self.id_to_word.get(t, '<UNK>'))
        return ' '.join(words)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / 'tokenizer.pkl.gz', 'wb') as f:
            pickle.dump({'word_to_id': self.word_to_id, 'id_to_word': self.id_to_word}, f)

    def load(self, path: Path) -> bool:
        p = path / 'tokenizer.pkl.gz'
        if p.exists():
            with gzip.open(p, 'rb') as f:
                state = pickle.load(f)
                self.word_to_id = state['word_to_id']
                self.id_to_word = state['id_to_word']
            return True
        return False

# ──────────────────────────────────────────────────────────────
# 🧠 TRANSFORMER (оставлен без изменений)
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
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
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
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

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
# 📚 COGNITIVE MEMORY (с улучшенным эмбеддингом)
# ──────────────────────────────────────────────────────────────
@dataclass
class EpisodicMemory:
    content: str
    timestamp: float
    importance: float = 0.5
    emotional_valence: float = 0.0
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(CONFIG.embedding_dim))

class CognitiveMemorySystem:
    def __init__(self, embed_func: Callable[[str], np.ndarray]):
        self.embed_func = embed_func
        self.episodic: List[EpisodicMemory] = []
        self.semantic: Dict[str, str] = {}
        self.procedural: Dict[str, str] = {}
        self.working_memory: deque = deque(maxlen=CONFIG.working_memory_size)

    def add_episode(self, content: str, importance: float, emotional_valence: float):
        emb = self.embed_func(content)
        self.episodic.append(EpisodicMemory(
            content=content, timestamp=time.time(),
            importance=importance, emotional_valence=emotional_valence, embedding=emb
        ))
        self.working_memory.append(content)
        if len(self.episodic) > CONFIG.episodic_memory_size:
            self.episodic.sort(key=lambda x: x.importance, reverse=True)
            self.episodic = self.episodic[:CONFIG.episodic_memory_size]

    def add_concept(self, concept: str, definition: str):
        self.semantic[concept] = definition

    def get_context(self, query: str, top_k: int = 3) -> str:
        if not self.episodic:
            return ""
        query_emb = self.embed_func(query)
        scored = []
        for mem in self.episodic:
            sim = np.dot(mem.embedding, query_emb) / (np.linalg.norm(mem.embedding) * np.linalg.norm(query_emb) + 1e-8)
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
            pickle.dump({'episodic': self.episodic, 'semantic': self.semantic, 'procedural': self.procedural}, f)

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
# 🌡️ SOMATOSENSORY SYSTEM (с psi-уровнем)
# ──────────────────────────────────────────────────────────────
class SomatosensorySystem:
    def __init__(self):
        self.state_history: deque = deque(maxlen=50)
        self.current_quality = 0.5
        self.emotional_valence = 0.0
        self.psi_level = 0.5          # внутренняя энергия / любопытство
        self.last_action_time = time.time()

    def update(self, quality: float, emotion: float):
        self.current_quality = self.current_quality * 0.8 + quality * 0.2
        self.emotional_valence = self.emotional_valence * 0.9 + emotion * 0.1
        self.state_history.append({'q': quality, 't': time.time()})
        # Обновление psi: новизна = 1 - quality (если качество низкое, новизна высокая)
        novelty = 1.0 - quality
        self._update_psi(novelty, quality)

    def _update_psi(self, novelty: float, success: float):
        time_since = time.time() - self.last_action_time
        time_factor = 1.0 - np.exp(-time_since / 60.0)  # растёт до 1 за минуту
        self.psi_level = np.clip(
            self.psi_level + 0.1 * novelty - 0.05 * (1.0 - success) + 0.05 * time_factor,
            0.1, 1.0
        )
        self.last_action_time = time.time()

    def should_initiate_action(self) -> bool:
        """Вероятность инициировать действие (мышление/исследование)"""
        return np.random.random() < self.psi_level

    def get_state(self) -> str:
        if self.current_quality > 0.7:
            return "Уверенное"
        if self.current_quality < 0.3:
            return "Неопределенное"
        return "Нормальное"

# ──────────────────────────────────────────────────────────────
# 🔧 SELF-MODIFICATION (с безопасной песочницей)
# ──────────────────────────────────────────────────────────────
class SelfModificationEngine:
    FORBIDDEN_MODULES = {'os', 'subprocess', 'sys', 'shutil', 'importlib', 'eval', 'exec', 'compile', '__import__'}

    def __init__(self, llm_client, modules_dir: Path):
        self.llm = llm_client
        self.modules_dir = modules_dir
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_modules: Dict[str, Any] = {}

    def _is_safe_code(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in self.FORBIDDEN_MODULES:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.FORBIDDEN_MODULES:
                    return False
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {'eval', 'exec', 'compile'}:
                    return False
        return True

    async def create_module(self, task: str) -> Optional[str]:
        prompt = f"Создай Python модуль для: {task}. Только код, без markdown. Функция execute()."
        code = await self.llm.generate(prompt, temperature=0.3)
        code = re.sub(r'```python|```', '', code).strip()
        if not code or not self._is_safe_code(code):
            logger.warning("Сгенерированный код небезопасен или пуст")
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
            logger.info(f"✅ Модуль {module_name} создан")
            return module_name
        except Exception as e:
            logger.error(f"Ошибка загрузки модуля: {e}")
            return None

# ──────────────────────────────────────────────────────────────
# 🔗 TEACHER LLM (без изменений)
# ──────────────────────────────────────────────────────────────
class TeacherLLM:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=CONFIG.lm_studio_timeout)
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def close(self):
        if self._session:
            await self._session.close()

    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        await self.connect()
        for attempt in range(CONFIG.max_retries):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                async with self._session.post(
                    self.url,
                    json={"messages": messages, "temperature": temperature, "max_tokens": 1000},
                    headers={"Authorization": f"Bearer {self.key}"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['choices'][0]['message']['content'].strip()
            except Exception as e:
                logger.warning(f"LLM error (attempt {attempt+1}): {e}")
                await asyncio.sleep(CONFIG.retry_delay * (attempt+1))
        return ""

# ──────────────────────────────────────────────────────────────
# 🎓 TRAINER (упрощён)
# ──────────────────────────────────────────────────────────────
class HybridTrainer:
    def __init__(self, model: HybridTransformer, tokenizer: HybridBPETokenizer, device: str):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler() if CONFIG.mixed_precision else None

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
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

# ──────────────────────────────────────────────────────────────
# 🤖 HYBRID AGENT (с фоновой петлёй мышления)
# ──────────────────────────────────────────────────────────────
class HybridAutonomousAgent:
    def __init__(self, teacher: TeacherLLM):
        self.user_id = "local_user"
        self.teacher = teacher
        self.tokenizer = HybridBPETokenizer(CONFIG.vocab_size)
        self.model = HybridTransformer(
            CONFIG.vocab_size, CONFIG.d_model, CONFIG.n_heads,
            CONFIG.n_layers, CONFIG.d_ff, CONFIG.max_seq_length, CONFIG.dropout
        )
        self.trainer = HybridTrainer(self.model, self.tokenizer, CONFIG.device)

        # Эмбеддинг-функция (пытаемся использовать sentence-transformers, иначе fallback)
        self.embed_model = self._init_embed_model()
        def embed_func(text: str) -> np.ndarray:
            return self._compute_embedding(text)
        self.memory = CognitiveMemorySystem(embed_func)
        self.soma = SomatosensorySystem()
        self.self_mod = SelfModificationEngine(teacher, CONFIG.base_dir / 'modules' / 'custom' / self.user_id)

        self.user_dir = CONFIG.base_dir / 'models' / self.user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

        # Фоновая петля мышления
        self.is_active = True
        self.thought_task: Optional[asyncio.Task] = None
        self.new_thought_signal = None  # будет установлен из GUI

        logger.info("🚀 Агент инициализирован")

    def _init_embed_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2', device=CONFIG.device)
            logger.info("✅ SentenceTransformer загружен")
            return model
        except ImportError:
            logger.warning("⚠️ sentence-transformers не найден, используется fallback-эмбеддинг")
            return None

    def _compute_embedding(self, text: str) -> np.ndarray:
        if self.embed_model is not None:
            return self.embed_model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: bag-of-tokens
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
                self.model.load_state_dict(torch.load(model_path, map_location=CONFIG.device))
                logger.info("✅ Модель загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
        mem_path = self.user_dir / 'memory.pkl.gz'
        self.memory.load(mem_path)
        self.tokenizer.load(self.user_dir)

    def _save_state(self):
        torch.save(self.model.state_dict(), self.user_dir / 'model.pt')
        self.memory.save(self.user_dir / 'memory.pkl.gz')
        self.tokenizer.save(self.user_dir)

    async def start_thinking(self):
        """Запуск фоновой петли мышления"""
        if not self.thought_task:
            self.thought_task = asyncio.create_task(self._background_think_loop())

    async def stop_thinking(self):
        self.is_active = False
        if self.thought_task:
            await self.thought_task

    async def _background_think_loop(self):
        """Вечный внутренний монолог"""
        await asyncio.sleep(2)  # начальная пауза
        while self.is_active:
            try:
                if self.soma.should_initiate_action():
                    topic = await self._select_thought_topic()
                    thought = await self._generate_internal_thought(topic)
                    if await self._verify_truth(thought):
                        self.memory.add_episode(f"THOUGHT: {thought}", importance=0.8, emotional_valence=0.2)
                        if self.new_thought_signal:
                            self.new_thought_signal.emit(thought)
                        # Попытка создать модуль, если мысль содержит алгоритм
                        if CONFIG.self_modification_enabled and "алгоритм" in thought.lower():
                            await self.self_mod.create_module(thought)
                # Сон разума
                await asyncio.sleep(np.random.uniform(15, 45))
            except Exception as e:
                logger.error(f"Ошибка в фоновом мышлении: {e}")
                await asyncio.sleep(5)

    async def _select_thought_topic(self) -> str:
        """Выбор темы для размышления (интенциональность)"""
        if self.memory.semantic and np.random.random() < 0.3:
            concept = np.random.choice(list(self.memory.semantic.keys()))
            return f"Исследовать глубже концепт: {concept}"
        recent = list(self.memory.working_memory)[-3:]
        state_desc = f"Состояние: {self.soma.get_state()}. Последние события: {' '.join(recent)}"
        prompt = f"Ты автономный разум. Сформулируй вопрос для глубокого размышления (1 предложение), исходя из: {state_desc}"
        topic = await self.teacher.generate(prompt, system_prompt="Ты генератор внутренних вопросов для AGI.", temperature=0.9)
        return topic or "Какова природа моего существования?"

    async def _generate_internal_thought(self, topic: str) -> str:
        """Генерация внутренней мысли"""
        prompt = f"Вопрос для размышления: {topic}\nГлубокая мысль:"
        return await self.teacher.generate(prompt, system_prompt="Ты философствующий ИИ. Отвечай одной содержательной фразой.", temperature=0.8)

    async def _verify_truth(self, thought: str) -> bool:
        """Проверка истинности мысли"""
        # 1. Проверка противоречий с семантической памятью
        for concept, definition in self.memory.semantic.items():
            if concept in thought:
                check = f"Определение: {definition}\nУтверждение: {thought}\nПротиворечит? Ответь только Да или Нет."
                ans = await self.teacher.generate(check, temperature=0.1)
                if "Да" in ans:
                    return False
        # 2. Логическая оценка
        score_str = await self.teacher.generate(
            f"Оцени истинность утверждения по шкале 0.0-1.0 (только число):\n{thought}",
            temperature=0.0
        )
        try:
            score = float(re.search(r"(\d\.\d+|\d+)", score_str).group())
        except:
            score = 0.5
        return score > 0.7

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        start = time.time()
        # Добавляем контекст недавних мыслей
        recent_thoughts = [m for m in self.memory.working_memory if m.startswith("THOUGHT:")]
        thought_ctx = "\n".join(recent_thoughts[-2:]) if recent_thoughts else ""
        mem_context = self.memory.get_context(user_input)
        full_prompt = f"{mem_context}\n{thought_ctx}\nUser: {user_input}\nAssistant:"

        use_teacher = np.random.random() > self.soma.current_quality or len(user_input) > 100
        response = ""
        confidence = 0.0
        if not use_teacher:
            ids = self.tokenizer.encode(full_prompt, max_length=CONFIG.max_seq_length//2)
            tensor = torch.tensor([ids], dtype=torch.long, device=CONFIG.device)
            gen, conf = self.model.generate(tensor, max_length=100)
            response = self.tokenizer.decode(gen[0].cpu().tolist())
            confidence = conf
        else:
            response = await self.teacher.generate(full_prompt, system_prompt="Ты полезный ассистент.")
            confidence = 1.0 if response else 0.5

        if use_teacher and response:
            await self.trainer.train_step(user_input, response)

        emotion = (confidence - 0.5) * 2
        self.memory.add_episode(f"User: {user_input}\nAI: {response}", confidence, emotion)
        self.soma.update(confidence, emotion)

        if len(self.memory.episodic) % CONFIG.save_frequency == 0:
            self._save_state()

        metadata = {'quality': confidence, 'state': self.soma.get_state(), 'time': time.time() - start}
        return response, metadata

# ──────────────────────────────────────────────────────────────
# 🖥️ GUI (PyQt6) с интеграцией asyncio
# ──────────────────────────────────────────────────────────────
class AgentSignals(QObject):
    """Сигналы для обновления GUI из асинхронного кода"""
    new_message = pyqtSignal(str, str)  # role, content
    new_thought = pyqtSignal(str)
    status_update = pyqtSignal(str)

class MainWindow(QMainWindow):
    def __init__(self, agent: HybridAutonomousAgent):
        super().__init__()
        self.agent = agent
        self.signals = AgentSignals()
        self.agent.new_thought_signal = self.signals.new_thought

        self.setWindowTitle("🧠 Hybrid AGI v4.2 — Локальный когнитивный агент")
        self.setMinimumSize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Вкладки
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Вкладка чата
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Segoe UI", 11))
        chat_layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите сообщение...")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_btn = QPushButton("Отправить")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(input_layout)

        self.tabs.addTab(chat_widget, "💬 Диалог")

        # Вкладка лога разума
        thought_widget = QWidget()
        thought_layout = QVBoxLayout(thought_widget)
        self.thought_log = QPlainTextEdit()
        self.thought_log.setReadOnly(True)
        self.thought_log.setFont(QFont("Consolas", 10))
        thought_layout.addWidget(self.thought_log)
        self.tabs.addTab(thought_widget, "🧠 Лог разума")

        # Статус бар
        self.status_label = QLabel("Готов")
        self.statusBar().addWidget(self.status_label)

        # Подключаем сигналы
        self.signals.new_message.connect(self.append_message)
        self.signals.new_thought.connect(self.append_thought)
        self.signals.status_update.connect(self.status_label.setText)

        # Таймер для периодического сохранения
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.save_agent_state)
        self.save_timer.start(60000)  # каждую минуту

        self._append_system("🤖 Агент запущен. Начинаю фоновое мышление...")

    def append_message(self, role: str, content: str):
        if role == "user":
            self.chat_history.append(f"<b>Вы:</b> {content}")
        else:
            self.chat_history.append(f"<b>Агент:</b> {content}")
        self.chat_history.moveCursor(QTextCursor.MoveOperation.End)

    def append_thought(self, thought: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.thought_log.appendPlainText(f"[{timestamp}] {thought}")
        # Автопрокрутка
        self.thought_log.verticalScrollBar().setValue(self.thought_log.verticalScrollBar().maximum())

    def _append_system(self, msg: str):
        self.chat_history.append(f"<i>{msg}</i>")

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.signals.new_message.emit("user", text)
        self.status_label.setText("Думаю...")
        # Запускаем асинхронную обработку
        asyncio.create_task(self._process_and_display(text))

    async def _process_and_display(self, user_input: str):
        try:
            response, meta = await self.agent.process_interaction(user_input)
            self.signals.new_message.emit("assistant", response)
            self.signals.status_update.emit(
                f"Качество: {meta['quality']:.0%} | Состояние: {meta['state']} | Время: {meta['time']:.2f}с"
            )
        except Exception as e:
            logger.exception("Ошибка обработки")
            self.signals.new_message.emit("system", f"⚠️ Ошибка: {e}")
        finally:
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.clear()
            self.input_field.setFocus()

    def save_agent_state(self):
        try:
            self.agent._save_state()
            logger.debug("Состояние агента сохранено")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")

    def closeEvent(self, event):
        # Корректное завершение фоновых задач
        self.agent.is_active = False
        if self.agent.thought_task:
            self.agent.thought_task.cancel()
        self.save_agent_state()
        event.accept()

# ──────────────────────────────────────────────────────────────
# 🚀 MAIN
# ──────────────────────────────────────────────────────────────
async def main():
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🧠 HYBRID COGNITIVE AGI v4.2 — LOCAL EDITION              ║
    ║     Transformer + Memory + Self-Mod + Background Thinking  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Инициализация Teacher LLM
    teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
    await teacher.connect()
    try:
        # Проверка связи с LLM
        test = await teacher.generate("Привет", temperature=0.1)
        if not test:
            logger.error("Не удалось получить ответ от LLM. Проверьте LM Studio.")
            return 1
        logger.info("✅ Связь с LLM установлена")
    except Exception as e:
        logger.error(f"Ошибка подключения к LLM: {e}")
        return 1

    # Создание агента
    agent = HybridAutonomousAgent(teacher)
    await agent.start_thinking()

    # Запуск GUI
    app = QApplication(sys.argv)
    loop = asyncio.get_event_loop()
    qasync.QApplication.setLoop(loop)

    window = MainWindow(agent)
    window.show()

    # Завершение работы
    try:
        await loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop_thinking()
        await teacher.close()
        logger.info("🛑 Приложение завершено")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()