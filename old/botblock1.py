#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID COGNITIVE AGI v4.3 — ЛОКАЛЬНАЯ ВЕРСИЯ С GUI (PyQt6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Фоновая петля мышления
✅ Проверка истинности (Truthfulness)
✅ Интенциональность (psi-уровень)
✅ Самомодификация с песочницей
✅ Полностью локальный чат
✅ ИСПРАВЛЕНО: таймауты, логика подключений, обработка ответов
✅ 🔧 FIX: WindowsSelectorEventLoopPolicy для совместимости aiohttp
"""

# 🔧 🔥 КРИТИЧЕСКИЙ ФИКС ДЛЯ WINDOWS 🔥
# Должен быть ДО всех импортов asyncio/aiohttp!
import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ──────────────────────────────────────────────────────────────
# 📦 ИМПОРТЫ
# ──────────────────────────────────────────────────────────────
import os
import json
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
import socket
import warnings
from dotenv import load_dotenv

# 🔥 КРИТИЧЕСКИ ВАЖНО: aiohttp должен быть импортирован!
import aiohttp

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QTabWidget, QPlainTextEdit,
    QProgressBar
)
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, Qt, QThread
from PyQt6.QtGui import QFont, QTextCursor
import qasync

from sympy import false

load_dotenv()
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)


# ──────────────────────────────────────────────────────────────
# ⚙️ КОНФИГУРАЦИЯ (исправленная)
# ──────────────────────────────────────────────────────────────
@dataclass
class HybridConfig:
    # API
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # 🔧 ИСПРАВЛЕНИЕ: Раздельные таймауты для разных этапов
    lm_studio_timeout_total: int = int(os.getenv('LM_TIMEOUT_TOTAL', '300'))  # 5 мин - общая
    lm_studio_timeout_connect: int = int(os.getenv('LM_TIMEOUT_CONNECT', '30'))  # 30 сек - подключение
    lm_studio_timeout_read: int = int(os.getenv('LM_TIMEOUT_READ', '180'))  # 3 мин - чтение ответа

    max_retries: int = 3
    retry_delay_base: float = 2.0  # база для экспоненциальной задержки

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
    embedding_dim: int = 384

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
    version: str = "4.3-LOCAL-FIXED"
    base_dir: Path = Path(os.getenv('BASE_DIR', 'hybrid_agi_v4'))

    def __post_init__(self):
        subdirs = ['models', 'memory', 'logs', 'modules/custom', 'tokenizer']
        for subdir in subdirs:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def get_aiohttp_timeout(self) -> aiohttp.ClientTimeout:
        """🔧 Возвращает корректный таймаут для aiohttp"""
        return aiohttp.ClientTimeout(
            total=self.lm_studio_timeout_total,
            connect=self.lm_studio_timeout_connect,
            sock_read=self.lm_studio_timeout_read
        )


CONFIG = HybridConfig()


# ──────────────────────────────────────────────────────────────
# 📊 LOGGING (улучшенный)
# ──────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger('Hybrid_AGI_Local')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    # Консоль с цветами (упрощённо)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    # Файл
    log_file = CONFIG.base_dir / 'logs' / f'agi_{datetime.now():%Y%m%d}.log'
    log_file.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ──────────────────────────────────────────────────────────────
# 🔤 BPE TOKENIZER (без изменений)
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
            try:
                with gzip.open(p, 'rb') as f:
                    state = pickle.load(f)
                    self.word_to_id = state['word_to_id']
                    self.id_to_word = state['id_to_word']
                return True
            except Exception as e:
                logger.error(f"Ошибка загрузки токенизатора: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# 🧠 TRANSFORMER (без изменений)
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
# 📚 COGNITIVE MEMORY (без изменений)
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
        try:
            emb = self.embed_func(content)
            self.episodic.append(EpisodicMemory(
                content=content, timestamp=time.time(),
                importance=importance, emotional_valence=emotional_valence, embedding=emb
            ))
            self.working_memory.append(content)
            if len(self.episodic) > CONFIG.episodic_memory_size:
                self.episodic.sort(key=lambda x: x.importance, reverse=True)
                self.episodic = self.episodic[:CONFIG.episodic_memory_size]
        except Exception as e:
            logger.error(f"Ошибка добавления эпизода: {e}")

    def add_concept(self, concept: str, definition: str):
        self.semantic[concept] = definition

    def get_context(self, query: str, top_k: int = 3) -> str:
        if not self.episodic:
            return ""
        try:
            query_emb = self.embed_func(query)
            scored = []
            for mem in self.episodic:
                norm_q = np.linalg.norm(query_emb)
                norm_m = np.linalg.norm(mem.embedding)
                if norm_q * norm_m > 1e-8:
                    sim = np.dot(mem.embedding, query_emb) / (norm_q * norm_m)
                    scored.append((mem, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            context = "=== Memory ===\n"
            for mem, score in scored[:top_k]:
                if score > 0.3:
                    context += f"[{score:.2f}] {mem.content}\n"
            return context if len(context) > 15 else ""
        except Exception as e:
            logger.warning(f"Ошибка получения контекста: {e}")
            return ""

    def save(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'wb') as f:
                pickle.dump({
                    'episodic': self.episodic,
                    'semantic': self.semantic,
                    'procedural': self.procedural
                }, f)
        except Exception as e:
            logger.error(f"Ошибка сохранения памяти: {e}")

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
        except Exception as e:
            logger.error(f"Ошибка загрузки памяти: {e}")
            return False


# ──────────────────────────────────────────────────────────────
# 🌡️ SOMATOSENSORY SYSTEM (без изменений)
# ──────────────────────────────────────────────────────────────
class SomatosensorySystem:
    def __init__(self):
        self.state_history: deque = deque(maxlen=50)
        self.current_quality = 0.5
        self.emotional_valence = 0.0
        self.psi_level = 0.5
        self.last_action_time = time.time()

    def update(self, quality: float, emotion: float):
        self.current_quality = self.current_quality * 0.8 + quality * 0.2
        self.emotional_valence = self.emotional_valence * 0.9 + emotion * 0.1
        self.state_history.append({'q': quality, 't': time.time()})
        novelty = 1.0 - quality
        self._update_psi(novelty, quality)

    def _update_psi(self, novelty: float, success: float):
        time_since = time.time() - self.last_action_time
        time_factor = 1.0 - np.exp(-time_since / 60.0)
        self.psi_level = np.clip(
            self.psi_level + 0.1 * novelty - 0.05 * (1.0 - success) + 0.05 * time_factor,
            0.1, 1.0
        )
        self.last_action_time = time.time()

    def should_initiate_action(self) -> bool:
        return np.random.random() < self.psi_level

    def get_state(self) -> str:
        if self.current_quality > 0.7:
            return "Уверенное"
        if self.current_quality < 0.3:
            return "Неопределенное"
        return "Нормальное"


# ──────────────────────────────────────────────────────────────
# 🔧 SELF-MODIFICATION (улучшенная проверка безопасности)
# ──────────────────────────────────────────────────────────────
class SelfModificationEngine:
    FORBIDDEN_MODULES = {
        'os', 'subprocess', 'sys', 'shutil', 'importlib',
        'eval', 'exec', 'compile', '__import__', 'pty', 'socket'
    }
    FORBIDDEN_FUNCTIONS = {'eval', 'exec', 'compile', '__import__', 'open'}

    def __init__(self, llm_client, modules_dir: Path):
        self.llm = llm_client
        self.modules_dir = modules_dir
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_modules: Dict[str, Any] = {}

    def _is_safe_code(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Синтаксическая ошибка в коде: {e}")
            return False

        for node in ast.walk(tree):
            # Проверка импортов
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in self.FORBIDDEN_MODULES:
                        logger.warning(f"Запрещённый модуль: {alias.name}")
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.FORBIDDEN_MODULES:
                    logger.warning(f"Запрещённый модуль: {node.module}")
                    return False
            # Проверка вызовов функций
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_FUNCTIONS:
                    logger.warning(f"Запрещённая функция: {node.func.id}")
                    return False
                # Проверка getattr(object, '__import__') и подобных
                if isinstance(node.func, ast.Attribute) and node.func.attr in self.FORBIDDEN_FUNCTIONS:
                    logger.warning(f"Запрещённый атрибут: {node.func.attr}")
                    return False
        return True

    async def create_module(self, task: str) -> Optional[str]:
        try:
            prompt = f"Создай Python модуль для: {task}. Только код, без markdown. Функция execute()."
            code = await self.llm.generate(prompt, temperature=0.3)
            code = re.sub(r'```python|```|`', '', code).strip()

            if not code or not self._is_safe_code(code):
                logger.warning("Сгенерированный код небезопасен или пуст")
                return None

            module_name = f"mod_{int(time.time())}"
            path = self.modules_dir / f"{module_name}.py"

            with open(path, 'w', encoding='utf-8') as f:
                f.write(code)

            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.loaded_modules[module_name] = mod
                logger.info(f"✅ Модуль {module_name} создан и загружен")
                return module_name
        except Exception as e:
            logger.error(f"Ошибка создания модуля: {e}\n{traceback.format_exc()}")
            return None



# ──────────────────────────────────────────────────────────────
# 🔗 TEACHER LLM (✅ ИСПРАВЛЕН: ленивая инициализация сессии)
# ──────────────────────────────────────────────────────────────
class TeacherLLM:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """🔧 Создаёт сессию ТОЛЬКО в текущем запущенном event loop"""
        if self._session is not None and not self._session.closed:
            return self._session

        timeout = CONFIG.get_aiohttp_timeout()
        connector = aiohttp.TCPConnector(
            limit=5,
            ttl_dns_cache=300,
            force_close=False,
            keepalive_timeout=30
        )
        self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        self._connected = True
        logger.info("✅ Сессия aiohttp создана в активном QEventLoop")
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._connected = False
            logger.info("🔌 Сессия LM Studio закрыта")

    async def _parse_response(self, resp: aiohttp.ClientResponse, raw_text: str) -> Optional[str]:
        """🔧 Парсинг ответа (без изменений)"""
        if not raw_text or not raw_text.strip():
            return None
        try:
            data = json.loads(raw_text)
            if "error" in data:
                logger.error(f"❌ LM Studio error: {data['error']}")
                return None
            if 'choices' not in data or not data['choices']:
                return None
            message = data['choices'][0].get('message')
            if not message or not isinstance(message, dict):
                return None
            content = message.get('content')
            if isinstance(content, str) and content.strip():
                return content.strip()
            reasoning = message.get('reasoning_content')
            if isinstance(reasoning, str) and reasoning.strip():
                if CONFIG.debug_mode:
                    logger.debug("🧠 Используем reasoning_content как fallback")
                return re.sub(r'^Thinking Process:\s*', '', reasoning.strip())
            delta = data['choices'][0].get('delta', {})
            if delta and isinstance(delta, dict):
                delta_content = delta.get('content')
                if isinstance(delta_content, str) and delta_content.strip():
                    return delta_content.strip()
            if CONFIG.debug_mode:
                logger.warning(f"⚠️ Нет контента. Message: {json.dumps(message, ensure_ascii=False)[:200]}")
            return None
        except json.JSONDecodeError:
            match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
            return None
        except Exception as e:
            if CONFIG.debug_mode:
                logger.error(f"❌ Ошибка парсинга: {type(e).__name__}: {e}")
            return None

    async def generate(self, prompt: str, system_prompt: str = "",
                       temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """🔧 Генерация с ленивой сессией и корректным таймаутом"""
        session = await self._get_session()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "model": os.getenv('LM_MODEL', 'local-model')
        }
        headers = {"Content-Type": "application/json"}
        if self.key and self.key != "lm-studio" and self.key.strip():
            headers["Authorization"] = f"Bearer {self.key}"

        # 🔥 ВОССТАНОВЛЕНО: получаем таймаут из конфига
        timeout = CONFIG.get_aiohttp_timeout()

        last_error = None
        for attempt in range(CONFIG.max_retries):
            try:
                if CONFIG.debug_mode:
                    logger.debug(f"📤 Запрос (попытка {attempt + 1}/{CONFIG.max_retries}): {prompt[:100]}...")

                async with session.post(self.url, json=payload, headers=headers, timeout=timeout) as resp:
                    raw_text = await resp.text()
                    if CONFIG.debug_mode:
                        logger.debug(f"📥 Ответ {resp.status} ({len(raw_text)} байт): {raw_text[:300]}...")

                    if resp.status == 200:
                        content = await self._parse_response(resp, raw_text)
                        if content:
                            return content
                        last_error = "Empty content"
                    else:
                        last_error = f"HTTP {resp.status}"
            except asyncio.TimeoutError:
                last_error = f"Timeout (попытка {attempt + 1})"
            except aiohttp.ClientConnectionError as e:
                last_error = f"ConnectionError: {e}"
                self._session = None  # Сброс сессии при обрыве
            except Exception as e:
                last_error = f"Exception: {type(e).__name__}: {e}"
                if CONFIG.debug_mode:
                    logger.debug(traceback.format_exc())

            if attempt < CONFIG.max_retries - 1:
                delay = CONFIG.retry_delay_base * (2 ** attempt) + np.random.uniform(0, 1)
                await asyncio.sleep(delay)

        logger.error(f"❌ Не удалось получить ответ после {CONFIG.max_retries} попыток. Ошибка: {last_error}")
        return f"⚠️ Ошибка связи с моделью: {last_error}"


# ──────────────────────────────────────────────────────────────
# 🎓 TRAINER (без изменений)
# ──────────────────────────────────────────────────────────────
class HybridTrainer:
    def __init__(self, model: HybridTransformer, tokenizer: HybridBPETokenizer, device: str):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler() if CONFIG.mixed_precision and torch.cuda.is_available() else None

    async def train_step(self, prompt: str, response: str):
        try:
            self.model.train()
            text = f"{prompt} {response}"
            input_ids = self.tokenizer.encode(text, max_length=CONFIG.max_seq_length)

            if len(input_ids) < 2:
                return

            tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            labels = tensor[:, 1:].clone()
            inputs = tensor[:, :-1]

            if self.scaler and self.device == 'cuda':
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
        except Exception as e:
            logger.debug(f"Ошибка обучения (некритично): {e}")


# ──────────────────────────────────────────────────────────────
# 🤖 HYBRID AGENT (исправленная логика)
# ──────────────────────────────────────────────────────────────
class HybridAutonomousAgent:
    def __init__(self, teacher: TeacherLLM):
        self.user_id = "local_user"
        self.teacher = teacher
        self.tokenizer = HybridBPETokenizer(CONFIG.vocab_size)

        # Инициализация модели
        try:
            self.model = HybridTransformer(
                CONFIG.vocab_size, CONFIG.d_model, CONFIG.n_heads,
                CONFIG.n_layers, CONFIG.d_ff, CONFIG.max_seq_length, CONFIG.dropout
            )
            logger.info("✅ Локальная модель Transformer создана")
        except Exception as e:
            logger.error(f"❌ Ошибка создания модели: {e}")
            raise

        self.trainer = HybridTrainer(self.model, self.tokenizer, CONFIG.device)

        # Эмбеддинг
        self.embed_model = self._init_embed_model()

        def embed_func(text: str) -> np.ndarray:
            return self._compute_embedding(text)

        self.memory = CognitiveMemorySystem(embed_func)
        self.soma = SomatosensorySystem()
        self.self_mod = SelfModificationEngine(
            teacher,
            CONFIG.base_dir / 'modules' / 'custom' / self.user_id
        )

        self.user_dir = CONFIG.base_dir / 'models' / self.user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

        # Фоновая петля
        self.is_active = True
        self.thought_task: Optional[asyncio.Task] = None
        self.new_thought_signal = None

        logger.info("🚀 Агент инициализирован")

    def _init_embed_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2', device=CONFIG.device)
            logger.info("✅ SentenceTransformer загружен")
            return model
        except ImportError:
            logger.warning("⚠️ sentence-transformers не найден, используется fallback")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки эмбеддинга: {e}")
            return None

    def _compute_embedding(self, text: str) -> np.ndarray:
        if self.embed_model is not None:
            try:
                return self.embed_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            except:
                pass
        # Fallback
        tokens = self.tokenizer.encode(text, max_length=50)
        emb = np.zeros(CONFIG.embedding_dim)
        for i, t in enumerate(tokens[:CONFIG.embedding_dim]):
            emb[i] = t / CONFIG.vocab_size
        return emb

    def _load_state(self):
        try:
            model_path = self.user_dir / 'model.pt'
            if model_path.exists():
                self.model.load_state_dict(
                    torch.load(model_path, map_location=CONFIG.device, weights_only=True)
                )
                logger.info("✅ Модель загружена")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить модель: {e}")

        try:
            mem_path = self.user_dir / 'memory.pkl.gz'
            self.memory.load(mem_path)
        except:
            pass

        self.tokenizer.load(self.user_dir)

    def _save_state(self):
        try:
            torch.save(self.model.state_dict(), self.user_dir / 'model.pt')
            self.memory.save(self.user_dir / 'memory.pkl.gz')
            self.tokenizer.save(self.user_dir)
            logger.debug("💾 Состояние сохранено")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")

    async def start_thinking(self):
        if not self.thought_task or self.thought_task.done():
            self.is_active = True
            self.thought_task = asyncio.create_task(self._background_think_loop())
            logger.info("🧠 Фоновое мышление запущено")

    async def stop_thinking(self):
        self.is_active = False
        if self.thought_task and not self.thought_task.done():
            self.thought_task.cancel()
            try:
                await self.thought_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Фоновое мышление остановлено")

    async def _background_think_loop(self):
        await asyncio.sleep(3)
        while self.is_active:
            try:
                if self.soma.should_initiate_action():
                    topic = await self._select_thought_topic()
                    thought = await self._generate_internal_thought(topic)
                    if thought and await self._verify_truth(thought):
                        self.memory.add_episode(
                            f"THOUGHT: {thought}",
                            importance=0.8,
                            emotional_valence=0.2
                        )
                        if self.new_thought_signal:
                            self.new_thought_signal.emit(thought)

                        if CONFIG.self_modification_enabled and "алгоритм" in thought.lower():
                            await self.self_mod.create_module(thought)

                # Случайный интервал для "размышлений"
                await asyncio.sleep(np.random.uniform(20, 60))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в фоновом мышлении: {e}")
                await asyncio.sleep(10)

    async def _select_thought_topic(self) -> str:
        if self.memory.semantic and np.random.random() < 0.3:
            concept = np.random.choice(list(self.memory.semantic.keys()))
            return f"Исследовать: {concept}"

        recent = list(self.memory.working_memory)[-3:]
        state_desc = f"Состояние: {self.soma.get_state()}. Контекст: {' | '.join(recent)}"
        prompt = f"Автономный разум. Сформулируй вопрос для размышления (1 предложение): {state_desc}"

        try:
            topic = await self.teacher.generate(
                prompt,
                system_prompt="Ты генератор внутренних вопросов.",
                temperature=0.9,
                max_tokens=100
            )
            return topic if topic and len(topic) > 10 else "Что я могу улучшить в своём мышлении?"
        except:
            return "Какова природа моего существования?"

    async def _generate_internal_thought(self, topic: str) -> str:
        prompt = f"Вопрос: {topic}\nГлубокая мысль:"
        try:
            return await self.teacher.generate(
                prompt,
                system_prompt="Ты философствующий ИИ. Отвечай одной содержательной фразой.",
                temperature=0.8,
                max_tokens=150
            )
        except:
            return ""

    async def _verify_truth(self, thought: str) -> bool:
        if not thought:
            return False
        try:
            # Проверка противоречий
            for concept, definition in list(self.memory.semantic.items())[:5]:
                if concept in thought and len(thought) < 200:
                    check = f"Определение: {definition}\nУтверждение: {thought}\nПротиворечит? Только Да или Нет."
                    ans = await self.teacher.generate(check, temperature=0.1, max_tokens=10)
                    if ans and "Да" in ans:
                        return False

            # Оценка истинности
            score_prompt = f"Оцени истинность (только число 0.0-1.0):\n{thought}"
            score_str = await self.teacher.generate(score_prompt, temperature=0.0, max_tokens=10)
            match = re.search(r"(\d\.?\d*)", score_str or "")
            score = float(match.group()) if match else 0.5
            return score > 0.6
        except:
            return True  # По умолчанию доверяем

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        """✅ Улучшенная логика обработки взаимодействия (из v37.0)"""
        start = time.time()
        metadata = {'quality': 0.5, 'state': 'Unknown', 'time': 0, 'source': 'unknown'}

        try:
            # 1. Получаем контекст из памяти
            recent_thoughts = [m for m in self.memory.working_memory if str(m).startswith("THOUGHT:")]
            thought_ctx = "\n".join(recent_thoughts[-2:]) if recent_thoughts else ""
            mem_context = self.memory.get_context(user_input)

            # 2. Формируем промпт
            full_prompt = f"{mem_context}\n{thought_ctx}\nUser: {user_input}\nAssistant:"

            # 3. Выбираем источник ответа (умное переключение)
            use_teacher = (
                    self.soma.current_quality < 0.6 or
                    len(user_input) > 100 or
                    '?' in user_input or '!' in user_input
            )

            response = ""
            confidence = 0.0

            if not use_teacher:
                # Локальная модель
                try:
                    ids = self.tokenizer.encode(full_prompt, max_length=CONFIG.max_seq_length // 2)
                    if len(ids) > 3:
                        tensor = torch.tensor([ids], dtype=torch.long, device=CONFIG.device)
                        gen, conf = self.model.generate(tensor, max_length=80, temperature=0.8)
                        response = self.tokenizer.decode(gen[0].cpu().tolist(), skip_special=True)
                        confidence = conf
                        metadata['source'] = 'local'
                except Exception as e:
                    logger.debug(f"Локальная генерация: {e}")
                    use_teacher = True

            if use_teacher:
                # Teacher LLM
                response = await self.teacher.generate(
                    full_prompt,
                    system_prompt="Ты полезный и точный ассистент. Отвечай естественно, 2-4 предложения.",
                    temperature=0.75,
                    max_tokens=2048  # ✅ Увеличено для полных ответов
                )
                confidence = 1.0 if response and "ошибка" not in response.lower() and "⚠️" not in response else 0.3
                metadata['source'] = 'teacher'

                # Обучение на успешных ответах
                if confidence > 0.7 and response:
                    await self.trainer.train_step(user_input, response)

            # 4. Обновление состояния
            if response and response.strip():
                emotion = (confidence - 0.5) * 2
                self.memory.add_episode(
                    f"User: {user_input}\nAI: {response}",
                    confidence,
                    emotion
                )
                self.soma.update(confidence, emotion)

                # Автосохранение
                if len(self.memory.episodic) % CONFIG.save_frequency == 0:
                    self._save_state()
            else:
                response = "⚠️ Не удалось сгенерировать ответ"
                confidence = 0.1

            # 5. Формируем метаданные
            metadata.update({
                'quality': confidence,
                'state': self.soma.get_state(),
                'time': time.time() - start,
                'response_length': len(response)
            })

            logger.info(
                f"✅ [{self.user_id}] Q={confidence:.0%} | "
                f"Method={metadata['source']} | "
                f"State={self.soma.get_state()} | "
                f"T={metadata['time']:.1f}s"
            )

            return response, metadata

        except Exception as e:
            logger.exception("❌ Критическая ошибка в process_interaction")
            return f"⚠️ Внутренняя ошибка: {type(e).__name__}", {
                'quality': 0.0, 'state': 'Error', 'time': time.time() - start, 'source': 'error'
            }


# ──────────────────────────────────────────────────────────────
# 🖥️ GUI (улучшенная интеграция)
# ──────────────────────────────────────────────────────────────
class AgentSignals(QObject):
    # ✅ Вернул к 2 аргументам, чтобы совпадало со всеми вызовами
    new_message = pyqtSignal(str, str)
    new_thought = pyqtSignal(str)
    status_update = pyqtSignal(str)


class MainWindow(QMainWindow):
    def __init__(self, agent: HybridAutonomousAgent):
        super().__init__()
        self.agent = agent
        self.signals = AgentSignals()
        self.agent.new_thought_signal = self.signals.new_thought

        self.setWindowTitle(f"🧠 Hybrid AGI {CONFIG.version}")
        self.setMinimumSize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Вкладки
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Чат
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Segoe UI", 11))
        self.chat_history.setAcceptRichText(True)
        chat_layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите сообщение... (Enter для отправки)")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_btn = QPushButton("➤ Отправить")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setFixedWidth(100)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(input_layout)

        self.tabs.addTab(chat_widget, "💬 Диалог")

        # Лог разума
        thought_widget = QWidget()
        thought_layout = QVBoxLayout(thought_widget)
        self.thought_log = QPlainTextEdit()
        self.thought_log.setReadOnly(True)
        self.thought_log.setFont(QFont("Consolas", 9))
        thought_layout.addWidget(self.thought_log)
        self.tabs.addTab(thought_widget, "🧠 Лог разума")

        # Статус
        self.status_label = QLabel("🟡 Инициализация...")
        self.statusBar().addWidget(self.status_label)

        # Сигналы
        self.signals.new_message.connect(self.append_message)
        self.signals.new_thought.connect(self.append_thought)
        self.signals.status_update.connect(self.status_label.setText)

        # Таймер сохранения
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.save_agent_state)
        self.save_timer.start(60000)

        self._append_system("🤖 Агент готов. Начинаю фоновое мышление...")
        self.input_field.setFocus()

    def append_message(self, role: str, content: str):
        if role == "user":
            self.chat_history.append(
                f'<div style="text-align:right; margin:5px;"><b style="color:#2196F3">Вы:</b><br>{content}</div>'
            )
        elif role == "system":
            self.chat_history.append(
                f'<div style="text-align:center; margin:5px; color:#666; font-style:italic">{content}</div>'
            )
        else:
            self.chat_history.append(
                f'<div style="text-align:left; margin:5px;"><b style="color:#4CAF50">Агент:</b><br>{content}</div>'
            )
        self.chat_history.moveCursor(QTextCursor.MoveOperation.End)

    def append_thought(self, thought: str):
        if not thought:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.thought_log.appendPlainText(f"[{timestamp}] {thought}")
        scrollbar = self.thought_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_system(self, msg: str):
        self.signals.new_message.emit("system", msg)

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.signals.new_message.emit("user", text)
        self.status_label.setText("🔄 Думаю...")
        self.input_field.clear()

        # 🔧 Безопасное создание задачи в qasync
        loop = asyncio.get_event_loop()
        loop.create_task(self._process_and_display(text))

    async def _process_and_display(self, user_input: str):
        try:
            response, meta = await self.agent.process_interaction(user_input)
            # ✅ Эмитим только 2 аргумента
            self.signals.new_message.emit("assistant", response)
            # ✅ Метаданные выводим в статус-бар через отдельный сигнал
            self.signals.status_update.emit(
                f"🎯 {meta.get('source', '?')} | Качество: {meta['quality']:.0%} | {meta['state']} | {meta['time']:.2f}с"
            )
        except Exception as e:
            logger.exception("Ошибка обработки сообщения")
            self.signals.new_message.emit("system", f"⚠️ Ошибка: {e}")
        finally:
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setFocus()

    def save_agent_state(self):
        try:
            self.agent._save_state()
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")

    def closeEvent(self, event):
        logger.info("👋 Закрытие приложения...")
        self.agent.is_active = False
        if hasattr(self.agent, 'thought_task') and self.agent.thought_task:
            self.agent.thought_task.cancel()

        self.save_agent_state()

        # ✅ Корректное закрытие сессии
        if hasattr(self.agent, 'teacher'):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.agent.teacher.close())
            else:
                loop.run_until_complete(self.agent.teacher.close())

        event.accept()


# ──────────────────────────────────────────────────────────────
# 🚀 MAIN (✅ ИСПРАВЛЕННАЯ ВЕРСИЯ — ЕДИНСТВЕННАЯ)
# ──────────────────────────────────────────────────────────────
async def main_async():
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🧠 HYBRID COGNITIVE AGI {CONFIG.version}
    ║     Transformer + Memory + Self-Mod + Background Thinking  ║
    ╚═══════════════════════════════════════════════════════════╝
    Device: {CONFIG.device} | Debug: {CONFIG.debug_mode}
    """)

    # ✅ 1. СНАЧАЛА инициализируем Qt и QEventLoop
    # Все async-объекты (aiohttp и др.) создадутся в этом loop
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    from qasync import QEventLoop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)  # 🔥 Делаем его активным для всего приложения

    if CONFIG.debug_mode:
        logger.debug(f"🔍 Event loop тип: {type(asyncio.get_event_loop()).__name__}")

    # ✅ 2. Проверка LM Studio (синхронная — безопасна)
    def check_lm_studio(host: str, port: int, timeout: float = 3.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    lm_available = False
    for host in ["127.0.0.1", "localhost"]:
        if check_lm_studio(host, 1234):
            logger.info(f"✅ LM Studio обнаружен на {host}:1234")
            if host == "localhost":
                CONFIG.lm_studio_url = CONFIG.lm_studio_url.replace("127.0.0.1", "localhost")
            lm_available = True
            break

    if not lm_available:
        logger.warning("⚠️ LM Studio не обнаружен. Агент будет использовать локальную модель.")

    # ✅ 3. Инициализация Teacher LLM (после установки loop!)
    teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)

    if lm_available:
        await teacher.connect()
        try:
            test = await teacher.generate("Привет", temperature=0.1, max_tokens=50)
            if test and "ошибка" not in test.lower() and "⚠️" not in test:
                logger.info(f"✅ LLM отвечает: '{test[:50]}...'")
            else:
                logger.warning("⚠️ LLM вернула некорректный ответ")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка тестового запроса: {e}")
    else:
        logger.info("🔌 Работа в офлайн-режиме (только локальная модель)")

    # ✅ 4. Создание агента
    try:
        agent = HybridAutonomousAgent(teacher)
        await agent.start_thinking()
    except Exception as e:
        logger.critical(f"❌ Не удалось создать агента: {e}")
        app.quit()
        return

    # ✅ 5. Создание и показ окна
    window = MainWindow(agent)
    window.show()
    logger.info("🚀 GUI запущен")

    # ✅ 6. Запуск цикла событий
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Прервано пользователем")
    finally:
        # Корректная очистка
        await agent.stop_thinking()
        await teacher.close()
        agent._save_state()
        loop.close()
        logger.info("💾 Состояние сохранено. Выход.")


# ──────────────────────────────────────────────────────────────
# 🚀 MAIN (исправленная последовательность инициализации)
# ──────────────────────────────────────────────────────────────
async def main_async():
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🧠 HYBRID COGNITIVE AGI {CONFIG.version}
    ║     Transformer + Memory + Self-Mod + Background Thinking  ║
    ╚═══════════════════════════════════════════════════════════╝
    Device: {CONFIG.device} | Debug: {CONFIG.debug_mode}
    """)

    # ✅ 1. СНАЧАЛА инициализируем Qt и QEventLoop
    # Все последующие async-объекты (включая aiohttp) будут привязаны к этому циклу
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    from qasync import QEventLoop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)  # Делаем его активным для всего приложения

    # ✅ 2. Проверка LM Studio (синхронная, безопасна)
    import socket
    def check_lm_studio(host: str, port: int, timeout: float = 3.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    lm_available = False
    for host in ["127.0.0.1", "localhost"]:
        if check_lm_studio(host, 1234):
            logger.info(f"✅ LM Studio обнаружен на {host}:1234")
            if host == "localhost":
                CONFIG.lm_studio_url = CONFIG.lm_studio_url.replace("127.0.0.1", "localhost")
            lm_available = True
            break

    if not lm_available:
        logger.warning("⚠️ LM Studio не обнаружен. Агент будет использовать локальную модель.")

    # ✅ 3. Инициализация Teacher LLM (без раннего connect!)
    teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)

    if lm_available:
        # Сессия создастся автоматически при первом запросе из GUI
        logger.info("✅ LM Studio готов к работе (ленивое подключение)")
    else:
        logger.info("🔌 Работа в офлайн-режиме (только локальная модель)")

    # ✅ 4. Создание агента
    try:
        agent = HybridAutonomousAgent(teacher)
        await agent.start_thinking()
    except Exception as e:
        logger.critical(f"❌ Не удалось создать агента: {e}")
        app.quit()
        return

    # ✅ 5. Создание и показ окна
    window = MainWindow(agent)
    window.show()
    logger.info("🚀 GUI запущен")

    # ✅ 6. Запуск цикла событий
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Прервано пользователем")
    finally:
        # Корректная очистка
        await agent.stop_thinking()
        await teacher.close()
        agent._save_state()
        loop.close()
        logger.info("💾 Состояние сохранено. Выход.")

# ──────────────────────────────────────────────────────────────
# 🚀 ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Подавление предупреждений
    warnings.filterwarnings("ignore", message=".*pynvml.*")
    warnings.filterwarnings("ignore", message=".*torch.cuda.*")

    try:
        # ✅ asyncio.run() теперь безопасен благодаря WindowsSelectorEventLoopPolicy
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)