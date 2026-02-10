# coding: utf-8
"""
AGI_learning_until_understands.py
Модель обучается до тех пор, пока не начнёт отвечать как Qwen.
Рефлексия + обучение + создание клеток + расширение словаря.
"""
import os
import re
import random
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST_MODEL = True
except Exception:
    _HAS_ST_MODEL = False

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================
def clean_qwen_response(text: str) -> str:
    if not isinstance(text, str):
        return "Хорошо."
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'>\s*', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[\*\.\!\?\:\-\–—\s]+', '', text)
    text = re.sub(r'[\*\.\!\?\:\-\–—\s]+$', '', text)
    words = text.split()
    if len(words) > 60:
        text = ' '.join(words[:60])
        if not text.endswith(('.', '!', '?')):
            text += '.'
    return text or "Хорошо."

def safe_cell_name(base: str) -> str:
    name = re.sub(r'[^a-zA-Zа-яА-Я0-9_]', '_', base)
    name = re.sub(r'_+', '_', name).strip('_')
    return name if name else "unknown"

def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).lower().strip()

# ======================
# ТИПЫ МЫШЛЕНИЯ
# ======================
TAGS = {"[SOC]", "[FCT]", "[CAU]", "[PRC]", "[OPN]", "[MET]", "[CRT]"}
TAG_PATTERN = re.compile(r'\[(SOC|FCT|CAU|PRC|OPN|MET|CRT)\]')

def detect_input_type(user_input: str) -> str:
    s = user_input.lower().strip()
    if re.search(r'\b(привет|здравствуй|добрый день|как дела|пока)\b', s):
        return "SOC"
    if re.search(r'\b(что такое|кто такой|где находится|какая столица|формула|определение)\b', s):
        return "FCT"
    if re.search(r'\b(почему|зачем|отчего|причина)\b', s):
        return "CAU"
    if re.search(r'\b(как сделать|как приготовить|инструкция|шаг|алгоритм)\b', s):
        return "PRC"
    if re.search(r'\b(как ты думаешь|твоё мнение|лучше ли|нравится ли)\b', s):
        return "OPN"
    if re.search(r'\b(представь|вообрази|сочини|опиши как|метафора)\b', s):
        return "CRT"
    if re.search(r'\b(почему ты|как ты понял|что ты имел в виду|объясни свой ответ)\b', s):
        return "MET"
    return "FCT"

INPUT_TYPE_TO_STAGES = {
    "SOC": ["social"],
    "FCT": ["fact", "meta"],
    "CAU": ["cause", "fact", "meta"],
    "PRC": ["procedure", "fact"],
    "OPN": ["opinion", "meta"],
    "CRT": ["creative", "metaphor", "meta"],
    "MET": ["meta", "fact"]
}

def get_allowed_stages(input_type: str) -> List[str]:
    return INPUT_TYPE_TO_STAGES.get(input_type, ["fact", "meta"])

# ======================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# ======================
class EnhancedGlobalThoughtState:
    def __init__(self, batch_size: int, hidden_size: int, device: torch.device):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.context_vector = torch.zeros(batch_size, hidden_size, device=device)
        self.active_stages: List[str] = []
        self.working_memory = {"insights": []}
        self.thinking_depth = 0

    def update(self, new_info: torch.Tensor, stage: str):
        self.context_vector = 0.8 * self.context_vector + 0.2 * new_info
        if stage not in self.active_stages:
            self.active_stages.append(stage)
        self.thinking_depth += 1

# ======================
# РЕФЛЕКСИВНАЯ КЛЕТКА
# ======================
class ReflectiveBrainCell(nn.Module):
    def __init__(self, cell_id: int, input_size: int, hidden_size: int, cell_type: str = "generic"):
        super().__init__()
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.input_adapter = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.memory = nn.LSTMCell(hidden_size, hidden_size)
        self.self_monitoring = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.reflection_cache = []

    def forward(self, x: torch.Tensor, hx=None, cx=None, global_state=None):
        x = self.input_adapter(x)
        if hx is None:
            hx = torch.zeros(x.size(0), self.memory.hidden_size, device=x.device)
        if cx is None:
            cx = torch.zeros_like(hx)
        hx, cx = self.memory(x, (hx, cx))
        assessment = torch.sigmoid(self.self_monitoring(hx))
        confidence = assessment[:, 0].mean()
        if len(self.reflection_cache) > 3:
            self.reflection_cache.pop(0)
        self.reflection_cache.append(hx.detach())
        gate = torch.sigmoid(self.gate_logit + confidence)
        return hx * gate, hx, cx, confidence.item()

# ======================
# КОГНИТИВНАЯ СЕТЬ
# ======================
class EnhancedCognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 192, hidden_size: int = 384,
                 eos_token_id: int = 1, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.proj = nn.Linear(embedding_dim, hidden_size)
        self.cells = nn.ModuleDict()
        self.cell_states = {}
        self.cell_counter = 0
        self.global_state = None

        self._init_cells()
        self.to(self.device)

    def _init_cells(self):
        base_types = ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"]
        for t in base_types:
            self._add_cell(t)

    def _add_cell(self, cell_type: str):
        cid = f"{cell_type}_{self.cell_counter}"
        cell = ReflectiveBrainCell(self.cell_counter, self.hidden_size, self.hidden_size, cell_type)
        self.cells[cid] = cell.to(self.device)
        self.cell_states[cid] = (None, None)
        self.cell_counter += 1

    def reset_states(self, batch_size: int):
        for cid in self.cells:
            self.cell_states[cid] = (
                torch.zeros(batch_size, self.hidden_size, device=self.device),
                torch.zeros(batch_size, self.hidden_size, device=self.device)
            )

    def think(self, token_emb: torch.Tensor, allowed_stages: List[str], user_input: str, max_cycles=5):
        batch_size = token_emb.size(0)
        self.global_state = EnhancedGlobalThoughtState(batch_size, self.hidden_size, self.device)
        x = token_emb
        for cycle in range(max_cycles):
            outputs, acts = [], []
            for stage in allowed_stages:
                for cid, cell in self.cells.items():
                    if stage in cell.cell_type:
                        hx, cx = self.cell_states[cid]
                        out, new_hx, new_cx, act = cell(x, hx, cx, self.global_state)
                        self.cell_states[cid] = (new_hx, new_cx)
                        outputs.append(out)
                        acts.append(act)
            if outputs:
                weights = F.softmax(torch.tensor(acts, device=self.device), dim=0)
                stacked = torch.stack(outputs)
                x = (stacked * weights.unsqueeze(1)).sum(dim=0)
                dominant = allowed_stages[0]
                self.global_state.update(x, dominant)
        return x

    def generate(self, input_tokens: List[int], allowed_stages: List[str], user_input: str, max_len=30) -> List[int]:
        self.reset_states(1)
        with torch.no_grad():
            inp = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            emb = self.embedding(inp)
            emb = self.proj(emb)
            for t in range(emb.size(1)):
                self.think(emb[:, t, :], allowed_stages, user_input)
            current = inp[:, -1:]
            out_tokens = []
            for _ in range(max_len):
                emb_step = self.embedding(current).squeeze(1)
                emb_step = self.proj(emb_step)
                thought = self.think(emb_step, allowed_stages, user_input)
                logits = nn.Linear(self.hidden_size, self.vocab_size).to(self.device)(thought)
                probs = F.softmax(logits / 0.8, dim=-1)
                next_id = torch.multinomial(probs, 1)
                val = next_id.item()
                if val == self.eos_token_id:
                    break
                out_tokens.append(val)
                current = next_id
            return out_tokens

    def expand_vocab(self, new_words: List[str], vocab: Dict[str, int]):
        old_size = self.vocab_size
        for w in new_words:
            if w not in vocab:
                vocab[w] = len(vocab)
        new_size = len(vocab)
        if new_size > old_size:
            old_emb = self.embedding.weight.data.clone()
            self.embedding = nn.Embedding(new_size, self.embedding.embedding_dim, padding_idx=0)
            self.embedding.weight.data[:old_size] = old_emb
            self.vocab_size = new_size
            self.to(self.device)

# ======================
# СЕМАНТИЧЕСКАЯ ОЦЕНКА
# ======================
class SemanticEvaluator:
    def __init__(self):
        self.model = None
        if _HAS_ST_MODEL:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                pass

    def similarity(self, a: str, b: str) -> float:
        if self.model is not None:
            emb = self.model.encode([a, b], normalize_embeddings=True)
            return float(np.dot(emb[0], emb[1]))
        a_clean = set(clean_for_similarity(a).split())
        b_clean = set(clean_for_similarity(b).split())
        if not a_clean or not b_clean:
            return 0.0
        return len(a_clean & b_clean) / len(a_clean | b_clean)

# ======================
# УЧИТЕЛЬ С ЦИКЛОМ ОБУЧЕНИЯ ДО ПОНИМАНИЯ
# ======================
class Teacher:
    def __init__(self, api_url="http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        self.evaluator = SemanticEvaluator()

    def ask_qwen(self, prompt: str) -> str:
        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.5
            }, timeout=15)
            if resp.status_code == 200:
                return clean_qwen_response(resp.json()['choices'][0]['message']['content'])
        except:
            pass
        return "Хорошо."

    def train_until_understands(self, brain: EnhancedCognitiveNetwork, user_input: str,
                                vocab: Dict[str, int], ivocab: Dict[int, str],
                                max_attempts=6) -> str:
        qwen_answer = self.ask_qwen(user_input)
        print(f"👤: {user_input}")
        print(f"🤖 Qwen: {qwen_answer}")

        allowed_stages = get_allowed_stages(detect_input_type(user_input))
        best_similarity = -1.0
        best_response = ""

        # Токенизация
        input_tokens = self.tokenize(user_input, vocab)
        target_tokens = self.tokenize(qwen_answer, vocab)
        full_input = input_tokens + [vocab.get('<EOS>', 1)]
        full_target = target_tokens + [vocab.get('<EOS>', 1)]
        ivocab.update({v: k for k, v in vocab.items()})

        optimizer = torch.optim.AdamW(brain.parameters(), lr=1e-3)

        for attempt in range(1, max_attempts + 1):
            # === ОБУЧЕНИЕ ===
            brain.train()
            optimizer.zero_grad()

            inp_tensor = torch.tensor([full_input], dtype=torch.long, device=brain.device)
            tgt_tensor = torch.tensor([full_target], dtype=torch.long, device=brain.device)

            embedded = brain.embedding(inp_tensor)
            embedded = brain.proj(embedded)
            for t in range(embedded.size(1)):
                brain.think(embedded[:, t, :], allowed_stages, user_input)

            final_hidden = brain.global_state.context_vector
            logits = nn.Linear(brain.hidden_size, brain.vocab_size).to(brain.device)(final_hidden)
            loss = F.cross_entropy(logits, tgt_tensor[:, 0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()

            # === ГЕНЕРАЦИЯ ===
            brain.eval()
            with torch.no_grad():
                brain_tokens = brain.generate(input_tokens, allowed_stages, user_input)
                brain_answer = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)
                brain_answer = TAG_PATTERN.sub('', brain_answer).strip() or "Хорошо."

            sim = self.evaluator.similarity(qwen_answer, brain_answer)
            print(f"  🔁 Попытка {attempt}: loss={loss.item():.4f}, сходство = {sim:.3f}")

            if sim > best_similarity:
                best_similarity = sim
                best_response = brain_answer

            if sim >= 0.85:
                print("✅ Модель поняла!")
                return best_response

            # Создаём клетку и расширяем словарь
            brain._add_cell(detect_input_type(user_input))
            unknown = set(clean_for_similarity(qwen_answer).split()) - set(vocab.keys())
            if unknown:
                brain.expand_vocab(list(unknown), vocab)
                ivocab.update({v: k for k, v in vocab.items()})
                print(f"  ➕ Расширен словарь: +{len(unknown)} слов")

        print(f"⚠️ Макс. попыток. Лучшее сходство: {best_similarity:.3f}")
        return best_response

    def tokenize(self, text: str, vocab: Dict[str, int]) -> List[int]:
        words = clean_for_similarity(text).split()
        tokens = []
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
            tokens.append(vocab[w])
        return tokens

# ======================
# MAIN
# ======================
def main():
    print("🧠 Обучение до полного понимания")
    vocab = {
        '<PAD>': 0, '<EOS>': 1, '<UNK>': 2,
        'привет': 10, 'что': 11, 'такое': 12, 'почему': 13, 'как': 14,
        '.': 15, ',': 16, '?': 17, '!': 18, 'хорошо': 19, 'помочь': 20, 'чем': 21
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    brain = EnhancedCognitiveNetwork(vocab_size=len(vocab) + 500, device=device)
    teacher = Teacher()

    while True:
        user_input = input("\n👤 Вы: ").strip()
        if user_input.lower() in ['выход', 'exit']:
            break
        if not user_input:
            continue

        ivocab = {v: k for k, v in vocab.items()}
        final_answer = teacher.train_until_understands(brain, user_input, vocab, ivocab)
        print(f"\n💡 Ответ модели: {final_answer}")

if __name__ == "__main__":
    main()