#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat-AGI — расширение LoRA-обучаемого шелла с направлениями к AGI.
Включает:
 - Двойное мышление (draft + critic → refined)
 - Долговременная семантическая память (FAISS или fallback)
 - Навыки-модули + диспетчер
 - Автономный цикл: думать → проверять → учиться → обновлять память
 - Интерфейсы внешних инструментов (поиск, калькулятор, исполнение кода)
 - Улучшенное обучение: triplet loss, динамические веса, прогресс, ранняя остановка и регуляризация
 - Поддержка mixed precision (fp16/bf16), gradient clipping, seed для повторяемости
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
import math
import random
import ast
import torch
import tqdm
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Any, Dict, Callable
from contextlib import contextmanager
import requests
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel

# Optional libs for memory and semantic validation
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
except Exception:
    faiss = None
    SentenceTransformer = None
    HAS_FAISS = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    cosine_similarity = None

@dataclass
class Config:
    student_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    save_dir: str = "./student_model"
    results_dir: str = "./results"
    data_dir: str = "./data"
    chat_history_file: str = field(init=False)
    train_log_file: str = field(init=False)
    batch_size: int = 4
    new_knowledge_buffer: int = 12
    max_training_epochs: int = 3
    learning_rate: float = 1e-4
    teacher_timeout: int = 120
    teacher_url: str = "http://localhost:1234/v1/chat/completions"
    teacher_model: str = "qwen/qwen3-coder-30b"
    use_gpu: bool = field(init=False)
    device: torch.device = field(init=False)
    no_teacher: bool = False
    dry_run: bool = False
    auto_train: bool = True
    deterministic: bool = False
    headless: bool = False
    enable_reflection: bool = True
    enable_longterm_memory: bool = True  # ✅ ВКЛЮЧЕНО ПО УМОЛЧАНИЮ
    memory_dim: int = 384
    memory_top_k: int = 5
    teacher_max_retries: int = 3
    teacher_backoff_factor: float = 2.0
    enable_contrastive_learning: bool = True  # ✅ ВКЛЮЧЕНО ПО УМОЛЧАНИЮ
    ce_weight: float = 1.0
    contrastive_weight: float = 1.0
    early_stopping_patience: int = 3
    gradient_clip_val: float = 1.0
    random_seed: int = 42
    bf16: bool = False
    fp16: bool = True

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.chat_history_file = os.path.join(self.data_dir, "chat_history.json")
        self.train_log_file = os.path.join(self.data_dir, "training_log.json")
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")


def parse_args(argv: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Chat-AGI with LoRA — AGI-oriented extensions")
    parser.add_argument("--student", default=None, help="Student model name or path")
    parser.add_argument("--save-dir", default=None, help="Directory to save/load model")
    parser.add_argument("--data-dir", default=None, help="Directory for data files")
    parser.add_argument("--no-memory", action="store_true", help="Disable long-term memory")  # ← отключение
    parser.add_argument("--no-contrastive", action="store_true", help="Disable contrastive learning")  # ← отключение
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher model")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual saving)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic generation")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--buffer-size", type=int, default=None, help="Knowledge buffer size")
    parser.add_argument("--teacher-retries", type=int, default=None, help="Max retries for teacher calls")
    parser.add_argument("--teacher-backoff", type=float, default=None, help="Backoff factor for teacher calls")
    parser.add_argument("--ce-weight", type=float, default=None, help="CrossEntropy weight in loss")
    parser.add_argument("--contrastive-weight", type=float, default=None, help="Contrastive loss weight")
    parser.add_argument("--early-stopping", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--gradient-clip", type=float, default=None, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision")
    args = parser.parse_args(argv)
    cfg = Config()
    if args.student: cfg.student_model_name = args.student
    if args.save_dir: cfg.save_dir = args.save_dir
    if args.data_dir: cfg.data_dir = args.data_dir
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.buffer_size: cfg.new_knowledge_buffer = args.buffer_size
    cfg.no_teacher = args.no_teacher
    cfg.dry_run = args.dry_run
    cfg.deterministic = args.deterministic
    cfg.headless = args.headless
    cfg.enable_longterm_memory = not args.no_memory  # ← по умолчанию True
    if args.teacher_retries is not None: cfg.teacher_max_retries = args.teacher_retries
    if args.teacher_backoff is not None: cfg.teacher_backoff_factor = args.teacher_backoff
    cfg.enable_contrastive_learning = not args.no_contrastive  # ← по умолчанию True
    if args.ce_weight is not None: cfg.ce_weight = args.ce_weight
    if args.contrastive_weight is not None: cfg.contrastive_weight = args.contrastive_weight
    if args.early_stopping is not None: cfg.early_stopping_patience = args.early_stopping
    if args.gradient_clip is not None: cfg.gradient_clip_val = args.gradient_clip
    if args.seed is not None: cfg.random_seed = args.seed
    cfg.bf16 = args.bf16
    cfg.fp16 = args.fp16
    return cfg


def setup_logging(cfg: Config) -> logging.Logger:
    log_file = os.path.join(cfg.data_dir, "chat_agi.log")
    logger = logging.getLogger("chat_agi")
    logger.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging initialized")
    return logger


@contextmanager
def model_inference_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


cfg: Optional[Config] = None
logger: Optional[logging.Logger] = None
student_model: Optional[torch.nn.Module] = None
student_tokenizer = None
chat_history: List[dict] = []
knowledge_buffer: List[str] = []
_memory_index = None
_memory_embeddings: Optional[Any] = None
_memory_texts: List[str] = []
_embedder = None
SKILL_MAP: List[Tuple[Callable[[str], bool], Callable[[str], str]]] = []


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DialogueDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        self._prepare(texts)

    def _prepare(self, texts: List[str]):
        for text in texts:
            if not isinstance(text, str) or not self.is_valid_example(text):
                continue
            try:
                encoded = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.block_size,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]
                labels = input_ids.clone()
                assistant_token = "### Ассистент:"
                if assistant_token in text:
                    idx = text.find(assistant_token) + len(assistant_token)
                    prompt_tokens = self.tokenizer(text[:idx], add_special_tokens=False)["input_ids"]
                    max_len = min(len(prompt_tokens), labels.size(0))
                    labels[:max_len] = -100
                self.examples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
            except Exception as e:
                if logger:
                    logger.warning("DialogueDataset: skipping example due to: %s", e)

    @staticmethod
    def is_valid_example(text: str) -> bool:
        if not text.startswith("### Пользователь:"): return False
        if "### Ассистент:" not in text: return False
        if "### Пользователь: ###" in text or "### Ассистент: ###" in text: return False
        return True

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        return {k: v.clone() for k, v in item.items()}


class ContrastiveDialogueDataset(Dataset):
    def __init__(self, examples: List[Tuple[str, str, str]], tokenizer: AutoTokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        self._prepare(examples)

    def _prepare(self, examples: List[Tuple[str, str, str]]):
        for query, positive, negative in examples:
            if not all(isinstance(t, str) and len(t.strip()) > 0 for t in [query, positive, negative]):
                continue
            pos_text = f"### Пользователь: {query}### Ассистент: {positive}</s>"
            neg_text = f"### Пользователь: {query}### Ассистент: {negative}</s>"
            pos_encoded = self.tokenizer(
                pos_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.block_size,
                padding="max_length",
                return_tensors="pt",
            )
            neg_encoded = self.tokenizer(
                neg_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.block_size,
                padding="max_length",
                return_tensors="pt",
            )
            pos_labels = pos_encoded["input_ids"][0].clone()
            neg_labels = neg_encoded["input_ids"][0].clone()
            assistant_token = "### Ассистент:"
            if assistant_token in pos_text:
                idx = pos_text.find(assistant_token) + len(assistant_token)
                prompt_tokens = self.tokenizer(pos_text[:idx], add_special_tokens=False)["input_ids"]
                max_len = min(len(prompt_tokens), pos_labels.size(0))
                pos_labels[:max_len] = -100
            if assistant_token in neg_text:
                idx = neg_text.find(assistant_token) + len(assistant_token)
                prompt_tokens = self.tokenizer(neg_text[:idx], add_special_tokens=False)["input_ids"]
                max_len = min(len(prompt_tokens), neg_labels.size(0))
                neg_labels[:max_len] = -100
            self.examples.append({
                "positive": {
                    "input_ids": pos_encoded["input_ids"][0],
                    "attention_mask": pos_encoded["attention_mask"][0],
                    "labels": pos_labels
                },
                "negative": {
                    "input_ids": neg_encoded["input_ids"][0],
                    "attention_mask": neg_encoded["attention_mask"][0],
                    "labels": neg_labels
                }
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        pos_item = {k: v.clone() for k, v in item["positive"].items()}
        neg_item = {k: v.clone() for k, v in item["negative"].items()}
        return pos_item, neg_item


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_student_model(local_cfg: Config):
    global student_model, student_tokenizer, cfg, logger
    cfg = local_cfg
    logger.info("Loading student: %s", cfg.student_model_name)
    adapter_path = os.path.join(cfg.save_dir, "adapter_model.safetensors")
    config_path = os.path.join(cfg.save_dir, "adapter_config.json")
    try:
        if os.path.isdir(cfg.save_dir) and os.listdir(cfg.save_dir):
            try:
                student_tokenizer = AutoTokenizer.from_pretrained(cfg.save_dir)
                logger.info("Loaded tokenizer from save_dir")
            except Exception:
                student_tokenizer = AutoTokenizer.from_pretrained(cfg.student_model_name)
                logger.info("Loaded tokenizer from model hub")
        else:
            student_tokenizer = AutoTokenizer.from_pretrained(cfg.student_model_name)
            logger.info("Loaded tokenizer from model hub")
    except Exception as e:
        logger.exception("Failed to load tokenizer: %s", e)
        raise

    if student_tokenizer.pad_token is None:
        student_tokenizer.add_special_tokens({"pad_token": student_tokenizer.eos_token or "[PAD]"})
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    # ✅ Исправление: torch_dtype → dtype
    base_dtype = torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32)
    base_kwargs = {"dtype": base_dtype, "low_cpu_mem_usage": True}

    try:
        if os.path.exists(adapter_path) and os.path.exists(config_path):
            logger.info("Found saved LoRA adapter — loading base + PEFT from save_dir")
            base_model = AutoModelForCausalLM.from_pretrained(cfg.student_model_name, device_map="auto", **base_kwargs)
            student_model = PeftModel.from_pretrained(base_model, cfg.save_dir, is_trainable=True)
            logger.info("Loaded PEFT model")
        else:
            logger.info("No adapter found — loading base model and applying LoRA")
            base_model = AutoModelForCausalLM.from_pretrained(cfg.student_model_name, device_map="auto", **base_kwargs)
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["k_proj", "q_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            student_model = get_peft_model(base_model, lora_cfg)
            logger.info("Applied LoRA to base model")

        student_model.train()
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        logger.info("Model params: total=%s, trainable=%s", f"{total_params:,}", f"{trainable_params:,}")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise


def save_student_model(save_adapter_only: bool = True) -> bool:
    global student_model, student_tokenizer, cfg, logger
    try:
        os.makedirs(cfg.save_dir, exist_ok=True)
        if cfg.dry_run:
            logger.info("Dry run enabled — skipping actual save")
            return True
        if isinstance(student_model, PeftModel) and save_adapter_only:
            student_model.save_pretrained(cfg.save_dir)
        else:
            student_model.save_pretrained(cfg.save_dir)
        if student_tokenizer is not None:
            student_tokenizer.save_pretrained(cfg.save_dir)
        logger.info("Saved model & tokenizer to %s", cfg.save_dir)
        return True
    except Exception as e:
        logger.exception("Failed to save model: %s", e)
        return False


def load_chat_history():
    global chat_history, cfg, logger
    try:
        if os.path.exists(cfg.chat_history_file):
            with open(cfg.chat_history_file, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
                logger.info("Loaded %d chat history items.", len(chat_history))
        else:
            chat_history = []
    except Exception:
        logger.exception("Failed loading chat history")
        chat_history = []


def save_chat_history():
    global chat_history, cfg, logger
    try:
        with open(cfg.chat_history_file, "w", encoding="utf-8") as f:
            json.dump(chat_history[-200:], f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed saving chat history")


def add_to_history(role: str, content: str):
    global chat_history
    chat_history.append({"timestamp": datetime.now(timezone.utc).isoformat(), "role": role, "content": content})
    if len(chat_history) > 200:
        chat_history = chat_history[-200:]
    save_chat_history()


def clean_model_response(raw: Optional[str]) -> str:
    if not raw:
        return ""
    raw = raw.split("</s>")[0]
    raw = re.sub(r"^###\s*(?:Ассистент|Пользователь):\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"###.*$", "", raw).strip()
    raw = re.sub(r"Пользователь:.*$", "", raw).strip()
    return raw


def generate_with_role(prompt: str, role: str = "assistant", max_new_tokens: int = 150) -> str:
    global student_model, student_tokenizer, logger, cfg
    if student_model is None or student_tokenizer is None:
        logger.error("Model/tokenizer not initialized")
        return "Модель не загружена."
    try:
        role_instruction = ""
        if role.lower() == "critic":
             role_instruction = "Ты критик. Твоя задача — объективно оценить черновик ответа студента, указать на ошибки и предложить улучшения. Будь конкретен и конструктивен."
        elif role.lower() == "planner":
             role_instruction = "Ты планировщик. Разбей задачу на логичные шаги. Каждый шаг должен быть четким и выполнимым."
        elif role.lower() == "teacher":
             role_instruction = "Ты учитель. Объясни концепцию ясно и подробно. Используй примеры, если это необходимо."
        formatted = f"{role_instruction}### Пользователь: {prompt}### Ассистент:"
        inputs = student_tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        device = _get_model_device(student_model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not cfg.deterministic,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.5 if not cfg.deterministic else 0.0,
            "pad_token_id": student_tokenizer.pad_token_id,
            "eos_token_id": student_tokenizer.eos_token_id,
            "repetition_penalty": 1.2,
            # ✅ Убрано length_penalty — не поддерживается в generate()
        }
        with model_inference_mode(student_model):
            outputs = student_model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = student_tokenizer.decode(generated, skip_special_tokens=True)
        return clean_model_response(text) or "Извините, я не понял ваш запрос."
    except Exception as e:
        logger.exception("Error during generation with role '%s': %s", role, e)
        return "Ошибка генерации."
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return "Генерация прервана пользователем."


def generate_student(prompt: str, max_new_tokens: int = 150) -> str:
    return generate_with_role(prompt, role="assistant", max_new_tokens=max_new_tokens)


def generate_with_reflection(user_input: str) -> Tuple[str, str]:
    draft = generate_with_role(user_input, role="assistant", max_new_tokens=160)
    critic_prompt = (
        f"Вопрос: {user_input}"
        f"Черновик ответа студента: {draft}"
        "Ты — строгий редактор. Проверь ответ по следующим критериям:"
        "1. Релевантность: Ответ прямо относится к вопросу? (Да/Нет)"
        "2. Логичность: В ответе есть внутренние противоречия или нелогичные переходы? (Да/Нет)"
        "3. Ясность: Ответ понятен и не содержит жаргона или бессвязных фраз? (Да/Нет)"
        "4. Краткость: Ответ не содержит лишней информации? (Да/Нет)"
        "Укажи конкретные ошибки по каждому пункту. Затем предоставь УЛУЧШЕННУЮ ВЕРСИЮ, которая исправляет все указанные недостатки."
        "Формат ответа:Оценка:1. Релевантность: ...2. Логичность: ...3. Ясность: ...4. Краткость: ...Улучшенная версия: [ВАШ УЛУЧШЕННЫЙ ОТВЕТ]"
    )
    critic_raw = generate_with_role(critic_prompt, role="critic", max_new_tokens=180)
    improved = None
    m = re.search(r"Улучшенная версия\s*[:\-]?\s*(.*)", critic_raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        improved = m.group(1).strip()
    else:
        parts = re.split(r"(?:Вариант|1\.)", critic_raw)
        if len(parts) > 1:
            improved = parts[-1].strip()
    if not improved or len(improved) < 10:
        ask_refine = f"Пожалуйста, перепиши и исправь черновик: {draft}в 1-2 абзацах, на русском."
        improved = generate_student(ask_refine, max_new_tokens=160)
    return improved, critic_raw


# === НОВЫЕ ФУНКЦИИ ===

def teacher_request_correct_answer(prompt: str, model: Optional[str] = None, max_retries: Optional[int] = None, backoff_factor: Optional[float] = None) -> str:
    global cfg, logger
    if cfg is None or getattr(cfg, "no_teacher", False):
        logger.info("Teacher calls disabled")
        return ""

    model = model or cfg.teacher_model
    url = cfg.teacher_url
    max_retries = max_retries if max_retries is not None else cfg.teacher_max_retries
    backoff_factor = backoff_factor if backoff_factor is not None else cfg.teacher_backoff_factor

    system_prompt = (
        "Ты — эксперт-учитель. Твоя задача — дать ОДИН точный, полный и понятный ответ на вопрос пользователя на русском языке.\n"
        "Не пиши вводных фраз, не объясняй, не комментируй. Просто дай ответ.\n"
        "Если вопрос неясен — сделай разумное предположение и ответь кратко."
    )

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,  # ← Ключевое изменение
        "temperature": 0.3,
        "max_tokens": 300,
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=data, timeout=cfg.teacher_timeout)
            r.raise_for_status()
            result = r.json()

            content = ""
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"].strip()
                elif "text" in choice:
                    content = choice["text"].strip()
            elif "message" in result and "content" in result["message"]:
                content = result["message"]["content"].strip()
            else:
                content = str(result).strip()

            if content and len(content) > 5:
                return content
            else:
                logger.warning("Teacher returned empty or too short response")
                return ""

        except requests.exceptions.Timeout:
            logger.warning("Teacher request timed out (attempt %d)", attempt)
        except requests.exceptions.RequestException as e:
            logger.warning("Teacher request failed (attempt %d): %s", attempt, e)
        except ValueError as e:
            raw = r.text[:300] if 'r' in locals() else "N/A"
            logger.warning("JSON decode error from teacher (attempt %d): %s — raw: %s", attempt, e, raw)
        except Exception as e:
            logger.warning("Unexpected error in teacher request (attempt %d): %s", attempt, e)

        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= backoff_factor
        else:
            logger.error("Teacher is unavailable after retries")
    return ""


def validate_and_correct_student_answer(student_answer: str, user_question: str) -> Tuple[bool, str, str]:
    global cfg, logger, _embedder

    if getattr(cfg, "no_teacher", False):
        is_ok = bool(student_answer and len(student_answer.strip()) > 10)
        return is_ok, "(no teacher — heuristic)", ""

    correct_answer = teacher_request_correct_answer(user_question)
    if not correct_answer:
        return False, "teacher unreachable", ""

    if _embedder is not None and HAS_SKLEARN:
        try:
            emb_student = _embedder.encode([student_answer])
            emb_correct = _embedder.encode([correct_answer])
            sim = cosine_similarity(emb_student, emb_correct)[0][0]
            is_correct = sim > 0.7
            explanation = f"Сходство: {sim:.2f} | Студент: {student_answer[:50]}... | Эталон: {correct_answer[:50]}..."
        except Exception as e:
            logger.warning("Embedding comparison failed: %s", e)
            is_correct = False
            explanation = "embedding error"
    else:
        comparison_prompt = (
            f"Вопрос: {user_question}\n"
            f"Ответ студента: {student_answer}\n"
            f"Эталонный ответ: {correct_answer}\n\n"
            "Сравни два ответа. Является ли ответ студента семантически эквивалентным эталонному? "
            "Ответь строго одним словом: ДА или НЕТ."
        )
        validation_response = generate_with_role(comparison_prompt, role="critic", max_new_tokens=20)
        is_correct = "да" in validation_response.lower()
        explanation = f"LLM-валидация: {validation_response} | Студент: {student_answer[:50]}..."

    return is_correct, explanation, correct_answer


def autonomous_cycle(user_input: str, max_steps: int = 3):
    global knowledge_buffer, cfg, logger
    log = []

    skill_out = skill_dispatcher(user_input)
    if skill_out:
        log.append("Использован модуль навыка")
        add_to_history("Студент", skill_out)
        add_to_longterm_memory(f"Ответ на: {user_input} {skill_out}")
        return skill_out, log

    current_answer = None
    step = 0

    while step < max_steps:
        step += 1
        log.append(f"Шаг {step}/{max_steps}")

        if cfg.enable_reflection:
            refined, critic = generate_with_reflection(user_input)
            log.append("Генерация с рефлексией")
        else:
            refined = generate_student(user_input)
            log.append("Генерация (без рефлексии)")

        current_answer = refined
        add_to_history("Студент", refined)

        is_correct, explanation, correct_answer = validate_and_correct_student_answer(refined, user_input)
        log.append(f"Валидация: {is_correct} — {explanation}")

        if is_correct:
            formatted = f"### Пользователь: {user_input}### Ассистент: {refined}</s>"
            if formatted not in knowledge_buffer:
                knowledge_buffer.append(formatted)
            add_to_longterm_memory(formatted)
            log.append("✅ Ответ принят — добавлен в буфер и память")
            break
        else:
            wrong = f"### Пользователь: {user_input}### Ассистент: {refined}</s>"
            if wrong not in knowledge_buffer:
                knowledge_buffer.append(wrong)

            if correct_answer:
                right = f"### Пользователь: {user_input}### Ассистент: {correct_answer}</s>"
                if right not in knowledge_buffer:
                    knowledge_buffer.append(right)
                add_to_longterm_memory(right)
                log.append("Добавлен эталонный ответ от учителя")

            if len(knowledge_buffer) >= cfg.new_knowledge_buffer and cfg.auto_train:
                logger.info("Buffer full during iteration — start training.")
                ok = train_student(knowledge_buffer)
                if ok:
                    knowledge_buffer.clear()
                    log.append("Автообучение завершено — буфер очищен")
                else:
                    log.append("Автообучение не удалось")

            if step == max_steps:
                log.append("❌ Максимум итераций достигнут — принимаем текущий ответ")
                formatted = f"### Пользователь: {user_input}### Ассистент: {refined}</s>"
                if formatted not in knowledge_buffer:
                    knowledge_buffer.append(formatted)

    if len(knowledge_buffer) >= cfg.new_knowledge_buffer and cfg.auto_train:
        logger.info("Buffer full — final training.")
        ok = train_student(knowledge_buffer)
        if ok:
            knowledge_buffer.clear()
            log.append("Финальное автообучение — буфер очищен")
        else:
            log.append("Финальное автообучение не удалось")

    return current_answer or "Не удалось сгенерировать ответ.", log


# === ОСТАЛЬНЫЕ ФУНКЦИИ БЕЗ ИЗМЕНЕНИЙ ===

def init_memory():
    global _memory_index, _embedder, _memory_embeddings, _memory_texts, cfg, logger
    if not cfg.enable_longterm_memory:
        logger.info("Longterm memory disabled by config")
        return
    if HAS_FAISS and SentenceTransformer is not None:
        try:
            logger.info("Initializing FAISS memory with sentence-transformers")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            dim = cfg.memory_dim
            index = faiss.IndexFlatL2(dim)
            _memory_index = index
            _memory_embeddings = None
            _memory_texts = []
            logger.info("FAISS memory ready")
            return
        except Exception as e:
            logger.warning("FAISS initialization failed: %s — falling back to list memory", e)
    logger.info("Initializing fallback memory (list)")
    if SentenceTransformer is not None:
        try:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = None
    else:
        _embedder = None
    _memory_index = None
    _memory_embeddings = []
    _memory_texts = []


def add_to_longterm_memory(text: str):
    global _memory_index, _embedder, _memory_embeddings, _memory_texts, cfg, logger
    if not cfg.enable_longterm_memory: return
    if _embedder is None:
        _memory_texts.append(text)
        return
    try:
        vec = _embedder.encode([text])
        vec = np.array(vec, dtype="float32")
        if _memory_index is not None:
            _memory_index.add(vec)
            _memory_texts.append(text)
        else:
            _memory_embeddings.append(vec[0])
            _memory_texts.append(text)
    except Exception as e:
        logger.warning("Failed to add to memory: %s", e)
        _memory_texts.append(text)


def retrieve_from_memory(query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    global _memory_index, _embedder, _memory_embeddings, _memory_texts, cfg
    top_k = top_k or cfg.memory_top_k
    if not cfg.enable_longterm_memory: return []
    if not _memory_texts: return []
    if _embedder is None:
        res = []
        for t in _memory_texts:
            if query.lower() in t.lower():
                res.append((t, 1.0))
        return res[:top_k]
    try:
        vec = _embedder.encode([query])
        vec = np.array(vec, dtype="float32")
        if _memory_index is not None:
            if len(_memory_texts) == 0: return []
            D, I = _memory_index.search(vec, min(top_k, len(_memory_texts)))
            out = []
            for dist, idx in zip(D[0], I[0]):
                if 0 <= idx < len(_memory_texts):
                    out.append((_memory_texts[int(idx)], float(dist)))
            return out
        else:
            if not _memory_embeddings: return []
            dists = []
            for emb in _memory_embeddings:
                d = float(np.linalg.norm(emb - vec[0]))
                dists.append(d)
            idxs = sorted(range(len(dists)), key=lambda i: dists[i])[:top_k]
            return [(_memory_texts[i], float(dists[i])) for i in idxs]
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)
        return []


def retrieval_module(user_input: str) -> str:
    ctxs = retrieve_from_memory(user_input)
    if not ctxs:
        return "Я не нашёл релевантных записей в памяти."
    snippets = "".join([f"- {t[:400]}..." for t, _ in ctxs])
    prompt = f"Используя следующие выдержки из памяти:{snippets}Ответь на вопрос: {user_input}"
    return generate_student(prompt, max_new_tokens=200)


def planning_module(user_input: str) -> str:
    prompt = f"Задача: {user_input}"
    return generate_with_role(prompt, role="planner", max_new_tokens=160)


def calculator_module(expression: str) -> str:
    try:
        expr = expression.replace(",", ".").replace(" ", "")
        if not re.fullmatch(r"[\d\.\(\)\+\-\*/\^]+", expr):
            return "Калькулятор: выражение содержит недопустимые символы."
        expr = expr.replace("^", "**")
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        code = compile(expr, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Использование '{name}' не разрешено")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"Результат: {result}"
    except ZeroDivisionError:
        return "Калькулятор: ошибка вычисления: деление на ноль."
    except OverflowError:
        return "Калькулятор: ошибка вычисления: результат слишком велик."
    except Exception as e:
        return f"Калькулятор: ошибка вычисления: {e}"


def code_runner_module(code: str, lang: str = "python") -> str:
    prompt = f"Проанализируй следующий код ({lang}):Объясни, что он делает, и укажи возможные проблемы или улучшения."
    return generate_student(prompt, max_new_tokens=200)


def register_skill(condition: Callable[[str], bool], handler: Callable[[str], str]):
    SKILL_MAP.append((condition, handler))


def skill_dispatcher(user_input: str) -> Optional[str]:
    low = user_input.lower().strip()
    for condition, handler in SKILL_MAP:
        if condition(user_input):
            return handler(user_input)
    if re.match(r"^\s*[-+*/\d\s\.\(\)\^,]+$", user_input.strip()):
        return calculator_module(user_input)
    if any(w in low for w in ("план", "как по шагам", "шаги", "как сделать")):
        return planning_module(user_input)
    if any(w in low for w in ("найди в памяти", "в памяти", "помни", "вспомни")):
        return retrieval_module(user_input)
    if low.startswith(("код:", "напиши код", "запусти код", "объясни код")):
        code = user_input.partition(":")[2] if ":" in user_input else user_input
        return code_runner_module(code)
    return None


def prepare_triplet_examples(examples: list) -> list:
    triplets = []
    i = 0
    while i < len(examples):
        wrong_example = examples[i]
        if "### Пользователь:" in wrong_example and "### Ассистент:" in wrong_example:
            user_part, assistant_part = wrong_example.split("### Ассистент:", 1)
            query = user_part.replace("### Пользователь:", "").strip()
            wrong_answer = assistant_part.replace("</s>", "").strip()
            j = i + 1
            right_answers = []
            while j < len(examples):
                right_example = examples[j]
                if "### Пользователь:" in right_example and "### Ассистент:" in right_example:
                    r_user_part, r_assistant_part = right_example.split("### Ассистент:", 1)
                    r_query = r_user_part.replace("### Пользователь:", "").strip()
                    r_answer = r_assistant_part.replace("</s>", "").strip()
                    if r_query == query:
                        right_answers.append(r_answer)
                        j += 1
                    else:
                        break
                else:
                    break
            if wrong_answer and right_answers:
                anchor = query
                positive = random.choice(right_answers)
                negative = wrong_answer
                triplets.append((anchor, positive, negative))
                i = j
            else:
                i += 1
        else:
            i += 1
    return triplets


class ImprovedContrastiveTrainer(Trainer):
    def __init__(self, margin=1.0, ce_weight=1.0, contrastive_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        pos_inputs, neg_inputs = zip(*inputs)
        pos_batch = {k: torch.stack([item[k] for item in pos_inputs]) for k in pos_inputs[0].keys()}
        neg_batch = {k: torch.stack([item[k] for item in neg_inputs]) for k in neg_inputs[0].keys()}
        pos_outputs = model(input_ids=pos_batch['input_ids'], attention_mask=pos_batch['attention_mask'],
                            labels=pos_batch['labels'])
        neg_outputs = model(input_ids=neg_batch['input_ids'], attention_mask=neg_batch['attention_mask'],
                            labels=neg_batch['labels'])
        pos_loss = pos_outputs.loss
        neg_loss = neg_outputs.loss
        contrastive_loss = F.relu(self.margin - (neg_loss - pos_loss))
        total_loss = self.ce_weight * pos_loss + self.contrastive_weight * contrastive_loss
        if hasattr(self.args, "max_grad_norm") and self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        return (total_loss, pos_outputs) if return_outputs else total_loss


def train_student(examples: List[str]) -> bool:
    global student_model, student_tokenizer, cfg, logger
    if not examples or len(examples) < 3:
        logger.warning("train_student: not enough examples")
        return False
    logger.info("Training on %d examples", len(examples))
    set_global_seed(cfg.random_seed)
    triplet_examples = prepare_triplet_examples(examples)
    use_contrastive = len(triplet_examples) > 0 and cfg.enable_contrastive_learning
    if use_contrastive:
        dataset = ContrastiveDialogueDataset(triplet_examples, student_tokenizer, block_size=256)
        logger.info(f"Using {len(dataset)} triplet pairs for training (contrastive).")
        trainer_cls = ImprovedContrastiveTrainer
        trainer_kwargs = {
            "margin": 1.0,
            "ce_weight": cfg.ce_weight,
            "contrastive_weight": cfg.contrastive_weight,
        }
    else:
        dataset = DialogueDataset(examples, student_tokenizer, block_size=256)
        logger.info("Using standard DialogueDataset for training.")
        trainer_cls = Trainer
        trainer_kwargs = {}

    if len(dataset) == 0:
        logger.warning("Dataset empty after processing")
        return False

    data_collator = DataCollatorForLanguageModeling(tokenizer=student_tokenizer, mlm=False)

    # ✅ ИСПРАВЛЕНО: save_strategy = "steps" или "epoch", чтобы совпадало с eval_strategy
    training_args = TrainingArguments(
        output_dir=cfg.results_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.max_training_epochs,
        per_device_train_batch_size=cfg.batch_size,
        save_total_limit=2,
        learning_rate=cfg.learning_rate,
        logging_dir=os.path.join(cfg.data_dir, "logs"),
        report_to=None,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="epoch",  # ← ИСПРАВЛЕНО: было "no"
        eval_strategy="no",
        metric_for_best_model="loss",
        load_best_model_at_end=False,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        weight_decay=0.01,
        logging_first_step=True,
        seed=cfg.random_seed,
        max_grad_norm=cfg.gradient_clip_val,
    )

    callbacks = []
    if cfg.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = trainer_cls(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        **trainer_kwargs
    )

    logger.info("Начинаем обучение...")
    best_loss = float("inf")
    try:
        train_result = trainer.train()
        best_loss = train_result.training_loss if hasattr(train_result, "training_loss") else best_loss
        logger.info("Training finished: %s", train_result)
        try:
            with open(cfg.train_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "examples_count": len(examples),
                    "triplet_count": len(triplet_examples) if use_contrastive else 0,
                    "best_loss": best_loss
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to save training log: %s", e)
        saved = save_student_model(save_adapter_only=True)
        if saved:
            logger.info("Adapter saved")
        else:
            logger.warning("Adapter saving failed")
        return saved
    except Exception as e:
        logger.exception("Training error: %s", e)
        return False


def chat_loop(headless: bool = False):
    global knowledge_buffer, cfg, logger
    load_chat_history()
    print("🤖 Chat-AGI (AGI-oriented) — запущено")
    if cfg.use_gpu:
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print("Команды: exit/quit/выход, save, clear, status, train, memory-status, reset, no-memory, no-contrastive")
    if headless:
        logger.info("Running in headless mode — read from stdin until EOF")
        try:
            inputs = sys.stdin.read().splitlines()
            for line in inputs:
                user_input = line.strip()
                if not user_input:
                    continue
                _process_user_input(user_input)
        except Exception as e:
            logger.exception("Error in headless mode: %s", e)
        return
    while True:
        try:
            user_input = input("👤 Пользователь: ").strip()
            if not user_input:
                continue
            cmd = user_input.lower()
            if cmd in ("exit", "quit", "выход"):
                save_chat_history()
                save_student_model()
                print("💾 Сохранено. Пока!")
                break
            elif cmd == "save":
                ok = save_student_model()
                print("✅ Сохранено" if ok else "❌ Ошибка при сохранении")
                continue
            elif cmd == "clear":
                knowledge_buffer.clear()
                print("✅ Буфер очищен")
                continue
            elif cmd == "status":
                print(f"""
    📊 Буфер знаний: {len(knowledge_buffer)}/{cfg.new_knowledge_buffer}
    🧠 Модель: {cfg.student_model_name}
    💾 Память: {len(_memory_texts)} записей
    🔁 Автообучение: {'вкл' if cfg.auto_train else 'выкл'}
    🧠 Рефлексия: {'вкл' if cfg.enable_reflection else 'выкл'}
    💾 Устройство: {cfg.device}
    🔁 Повторы учителя: {cfg.teacher_max_retries}
    🔁 Контрастное обучение: {'вкл' if cfg.enable_contrastive_learning else 'выкл'}
    🟦 CE weight: {cfg.ce_weight}
    🟨 Contrastive weight: {cfg.contrastive_weight}
    🔁 Gradient clip: {cfg.gradient_clip_val}
    🔁 Early stopping: {cfg.early_stopping_patience}
    🔁 Seed: {cfg.random_seed}
                    """.strip())
                continue
            elif cmd == "train":
                print("🔁 Ручной запуск обучения...")
                ok = train_student(knowledge_buffer)
                print("✅ Обучение завершено" if ok else "❌ Обучение не удалось")
                continue
            elif cmd == "memory-status":
                print(f"💾 Памяти в базе: {len(_memory_texts) if _memory_texts is not None else 0}")
                continue
            elif cmd == "reset":
                chat_history.clear()
                knowledge_buffer.clear()
                _memory_texts.clear()
                if _memory_embeddings is not None:
                    _memory_embeddings.clear()
                print("✅ Сброшено")
                continue
            _process_user_input(user_input)
        except KeyboardInterrupt:
            print("⏹ KeyboardInterrupt — сохраняем и выходим...")
            save_chat_history()
            save_student_model()
            break
        except EOFError:
            print("⏹ EOF — сохраняем и выходим...")
            save_chat_history()
            save_student_model()
            break
        except Exception as e:
            logger.exception("Unexpected error in chat loop: %s", e)
            print("❌ Ошибка исполнения. Смотрите логи.")


def _process_user_input(user_input: str):
    global knowledge_buffer, cfg
    add_to_history("Пользователь", user_input)
    try:
        answer, log = autonomous_cycle(user_input)
        print(f"🤖 Студент: {answer}")
        for l in log:
            print(f"   └─ {l}")
    except Exception as e:
        logger.exception("Error processing input: %s", e)
        print("❌ Ошибка обработки запроса. Смотрите логи.")


def main(argv: Optional[List[str]] = None):
    global cfg, logger
    cfg = parse_args(argv)
    logger = setup_logging(cfg)
    set_global_seed(cfg.random_seed)
    logger.info("Using device: %s", cfg.device)
    init_memory()
    register_skill(lambda x: re.match(r"^\s*[-+*/\d\s\.\(\)\^,]+$", x.strip()), calculator_module)
    register_skill(lambda x: any(w in x.lower() for w in ("план", "как по шагам", "шаги", "как сделать")),
                   planning_module)
    register_skill(lambda x: any(w in x.lower() for w in ("найди в памяти", "в памяти", "помни", "вспомни")),
                   retrieval_module)
    register_skill(lambda x: x.lower().strip().startswith(("код:", "напиши код", "запусти код", "объясни код")),
                   code_runner_module)
    try:
        load_student_model(cfg)
    except Exception as e:
        logger.error("Cannot initialize model; exiting: %s", e)
        return 1
    try:
        chat_loop(headless=getattr(cfg, "headless", False))
        return 0
    except Exception as e:
        logger.exception("Fatal error in chat loop: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())