# coding: utf-8
import os
import re
import math
import random
import traceback
from collections import Counter, deque
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОЧИСТКИ И РАЗМЕТКИ
# ======================

def clean_qwen_response(text: str) -> str:
    """Очищает ответ Qwen от markdown, лишней пунктуации и обрезает до разумного размера."""
    if not isinstance(text, str):
        return "Хорошо."
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'>\s*', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
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
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else "unknown"

def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# ======================
# РАСШИРЕННАЯ РАЗМЕТКА ЗНАНИЙ
# ======================

def classify_and_tag_response(text: str) -> str:
    if not text or not text.strip():
        return "[SOC] Хорошо."
    text = clean_qwen_response(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    tagged_parts = []
    for sent in sentences:
        if not sent.strip():
            continue
        tag = _detect_sentence_type(sent)
        tagged_parts.append(f"[{tag}] {sent}")
    return " ".join(tagged_parts)

def _detect_sentence_type(sentence: str) -> str:
    s = sentence.lower().strip()
    if re.search(r'\b(что такое|почему|как работает|объясни|значит ли|противоречие|размышля|думаешь|что значит)\b', s):
        return "MET"
    if re.search(r'\b(чтобы|нужно|следует|шаг|сначала|потом|затем|инструкция|алгоритм|как приготовить|как сделать)\b', s):
        return "PRC"
    if re.search(r'\b(потому что|так как|из-за|следствие|причина|происходит из-за|ведёт к|обусловлено)\b', s):
        return "CAU"
    if re.search(r'.+ — это .+|.+ называется .+|.+ состоит из .+|столица .+ — .+|формула .+ — .+', s):
        return "FCT"
    if re.search(r'\b(я думаю|мне кажется|по моему мнению|я считаю|мне нравится|скучный|отличный|лучше|хуже)\b', s):
        return "OPN"
    if re.search(r'\b(представь|вообрази|как будто|словно|подобно|жизнь —|мир как|если бы|фантазия)\b', s):
        return "CRT"
    social_keywords = ["привет", "здравствуй", "добрый", "спасибо", "пожалуйста", "извини", "хорошо", "ладно", "ок", "пока"]
    if any(kw in s for kw in social_keywords) or len(s.split()) <= 3:
        return "SOC"
    return "FCT"

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

# ======================
# КОГНИТИВНАЯ КЛЕТКА
# ======================

class BrainCell(nn.Module):
    def __init__(self, cell_id: int, input_size: int, hidden_size: int, cell_type: str = "generic"):
        super().__init__()
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_adapter = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.perception = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.memory = nn.LSTMCell(hidden_size, hidden_size)
        self.association = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        # gating: scalar multiplier learned at cell level
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        # runtime stats
        self.activation_level = 0.0

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None, cx: Optional[torch.Tensor] = None):
        batch_size = x.size(0)
        device = x.device
        x = self.input_adapter(x)
        perceived = self.perception(x)
        # activation as mean absolute activation
        self.activation_level = float(torch.mean(torch.abs(perceived)).detach().cpu().item())

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=device)
        if cx is None:
            cx = torch.zeros(batch_size, self.hidden_size, device=device)

        hx, cx = self.memory(perceived, (hx, cx))
        associated = self.association(hx)
        gate = torch.sigmoid(self.gate_logit)  # scalar in (0,1)
        # gated output
        return associated * gate, hx, cx, self.activation_level

# ======================
# META-COGNITION
# ======================

class MetaCognition:
    def __init__(self, vocab: Dict[str, int], network: Optional['CognitiveNetwork'] = None):
        self.vocab = vocab
        self.unknown_concepts: Set[str] = set()
        self.reflection_log: List[Dict[str, Any]] = []
        self.network = network  # for embeddings-based similarity

    def detect_unknown_words(self, text: str) -> Set[str]:
        words = set(clean_for_similarity(text).split())
        known = set(k for k in self.vocab.keys() if clean_for_similarity(k))
        return words - known

    def log_reflection(self, user_input: str, qwen_resp: str, brain_resp: str, question: Optional[str], reason: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "qwen_response": qwen_resp,
            "brain_response": brain_resp,
            "generated_question": question,
            "reason": reason
        }
        self.reflection_log.append(entry)

    def _embed_text(self, text: str) -> np.ndarray:
        if self.network is None:
            # fallback to bag-of-words
            tokens = clean_for_similarity(text).split()
            vec = np.zeros(128, dtype=float)
            for i, w in enumerate(tokens):
                vec[i % 128] += 1.0
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-8)
        # use network embedding average
        tokens = self.network.text_to_tokens(text, self.vocab)
        if not tokens:
            return np.zeros(self.network.embed_dim, dtype=float)
        emb = self.network.embedding(torch.tensor(tokens, dtype=torch.long, device=self.network.device))
        emb = emb.detach().cpu().numpy()
        vec = emb.mean(axis=0)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _estimate_similarity_semantic(self, resp1: str, resp2: str) -> float:
        v1 = self._embed_text(resp1)
        v2 = self._embed_text(resp2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return cos

    def should_reflect(self, user_input: str, qwen_resp: str, brain_resp: str) -> Tuple[bool, str]:
        expected_type = detect_input_type(user_input)
        actual_type = "FCT"
        marker_match = re.search(r'\[(SOC|FCT|CAU|PRC|OPN|MET|CRT)\]', qwen_resp)
        if marker_match:
            actual_type = marker_match.group(1)

        type_mismatch = False
        reason = ""

        if expected_type == "FCT" and actual_type in ["SOC", "OPN"]:
            type_mismatch = True
            reason = f"ожидался факт [FCT], но Qwen дал {actual_type}"
        elif expected_type == "PRC" and actual_type not in ["PRC", "FCT"]:
            type_mismatch = True
            reason = f"ожидалась инструкция [PRC], но Qwen дал {actual_type}"
        elif expected_type == "CRT" and actual_type in ["SOC", "FCT"]:
            type_mismatch = True
            reason = f"ождалось творчество [CRT], но Qwen дал {actual_type}"

        similarity = self._estimate_similarity_semantic(brain_resp, qwen_resp)
        if type_mismatch or similarity < 0.35:
            return True, reason if type_mismatch else f"низкое сходство ({similarity:.2f})"
        return False, ""

# ======================
# REPLAY BUFFER
# ======================

class ReplayBuffer:
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, inp: List[int], target: List[int], meta: Optional[Dict] = None):
        self.buffer.append({
            "input": inp,
            "target": target,
            "meta": meta or {}
        })

    def sample(self, batch_size: int) -> List[Dict]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)

# ======================
# КОГНИТИВНАЯ СЕТЬ
# ======================

class CognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_size: int = 512, eos_token_id: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_proj = nn.Linear(embedding_dim, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=max(1, min(8, hidden_size // 64)), batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, vocab_size)

        self.cells = nn.ModuleDict()
        self.cell_states: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        self.cell_activations: Dict[str, float] = {}
        self.cell_gates: Dict[str, float] = {}

        self.cell_counter = 0
        self.activation_threshold = 0.25
        self.thought_cycles = 3

        self.meta_cog: Optional[MetaCognition] = None

        self.replay = ReplayBuffer(capacity=5000)

        # move to device
        self.to(self.device)

    # ----------------------
    # управление клетками
    # ----------------------
    def _initialize_base_cells(self):
        base_cells = [
            ("social_greeting", "social"),
            ("social_thanks", "social"),
            ("fact_definition", "fact"),
            ("cause_explanation", "cause"),
            ("procedure_step", "procedure"),
            ("opinion_expression", "opinion"),
            ("meta_question", "meta"),
            ("creative_metaphor", "creative")
        ]
        for name, ctype in base_cells:
            self.add_cell(name, self.hidden_size, self.hidden_size, ctype)

    def add_cell(self, cell_type: str, input_size: int, hidden_size: int, cell_subtype: str = "association") -> str:
        safe_type = safe_cell_name(cell_type)
        cell_id = f"{safe_type}_{self.cell_counter}"
        self.cells[cell_id] = BrainCell(self.cell_counter, input_size, hidden_size, cell_subtype)
        self.cell_states[cell_id] = (None, None)
        self.cell_activations[cell_id] = 0.0
        self.cell_gates[cell_id] = float(torch.sigmoid(self.cells[cell_id].gate_logit).detach().cpu().item())
        self.cell_counter += 1
        # ensure module moved to device
        self.cells[cell_id].to(self.device)
        return cell_id

    def reset_cell_states(self, batch_size: int, device: Optional[torch.device] = None):
        device = device or self.device
        for cell_id in self.cells:
            self.cell_states[cell_id] = (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device)
            )

    # ----------------------
    # forward / process
    # ----------------------
    def _process_step(self, token_emb: torch.Tensor):
        # token_emb: [B, hidden_size]
        batch_size = token_emb.size(0)
        device = token_emb.device
        x = token_emb
        # stages order with grouping -- flexible via substring match
        stages = ["social", "fact", "cause", "procedure", "opinion", "meta", "creative", "association"]
        for stage in stages:
            stage_outputs = []
            for cid in [c for c in self.cells if stage in c or stage == "association"]:
                cell = self.cells[cid]
                hx, cx = self.cell_states.get(cid, (None, None))
                if hx is None:
                    hx = torch.zeros(batch_size, self.hidden_size, device=device)
                    cx = torch.zeros(batch_size, self.hidden_size, device=device)
                cell_output, new_hx, new_cx, act = cell(x, hx, cx)
                self.cell_states[cid] = (new_hx, new_cx)
                self.cell_activations[cid] = act
                self.cell_gates[cid] = float(torch.sigmoid(cell.gate_logit).detach().cpu().item())
                stage_outputs.append(cell_output)
            if stage_outputs:
                # aggregation: weighted average by learned gate per cell (approximated)
                stacked = torch.stack(stage_outputs, dim=0)  # [num_cells, B, H]
                x = torch.mean(stacked, dim=0)
        logits = self.final_proj(x)
        return logits

    def process_sequence(self, input_tokens: torch.Tensor) -> torch.Tensor:
        # input_tokens: [B, seq_len]
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        self.reset_cell_states(batch_size, device)
        embedded = self.embedding(input_tokens)  # [B, L, E]
        embedded = self.embed_proj(embedded)     # [B, L, H]
        attn_out, _ = self.attention(embedded, embedded, embedded)
        combined = self.norm(embedded + attn_out)
        outputs = []
        for t in range(seq_len):
            token_emb = combined[:, t, :]
            logits = self._process_step(token_emb)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)  # [B, L, V]

    # ----------------------
    # generation
    # ----------------------
    def generate_sequence(self, input_tokens: torch.Tensor, max_length: int = 20, base_temperature: float = 0.9,
                          top_k: int = 50, top_p: float = 0.9, eos_token_id: Optional[int] = None) -> List[int]:
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        device = input_tokens.device
        batch_size = input_tokens.size(0)
        self.reset_cell_states(batch_size, device)
        with torch.no_grad():
            embedded = self.embedding(input_tokens)
            embedded = self.embed_proj(embedded)
            attn_out, _ = self.attention(embedded, embedded, embedded)
            combined = self.norm(embedded + attn_out)
            for t in range(combined.size(1)):
                token_emb = combined[:, t, :]
                _ = self._process_step(token_emb)

            generated = []
            current = input_tokens[:, -1:].clone()  # [B,1]
            cur_batch = current
            cur_token_id = cur_batch[:, 0] if current.numel() > 0 else torch.tensor([0], device=device)

            # adapt temperature by marker token if available (heuristic)
            marker_map = {3: "SOC", 4: "FCT", 5: "CAU", 6: "PRC", 7: "OPN", 8: "MET", 9: "CRT"}
            try:
                last_id = int(cur_token_id[0].item())
                marker = marker_map.get(last_id, None)
            except Exception:
                marker = None
            if marker in ["FCT", "CAU"]:
                base_temperature = min(base_temperature, 0.4)
            elif marker == "CRT":
                base_temperature = max(base_temperature, 1.0)
            elif marker == "OPN":
                base_temperature = min(base_temperature, 0.8)

            for _ in range(max_length):
                emb = self.embedding(cur_batch).squeeze(1)  # [B, E]
                emb = self.embed_proj(emb)  # [B, H]
                logits = None
                for _ in range(self.thought_cycles):
                    logits = self._process_step(emb)  # [B, V]
                if logits is None:
                    break
                logits = logits / base_temperature
                probs = F.softmax(logits, dim=-1)  # [B, V]

                # top-k/top-p sampling for each item in batch (here batch=1 typical)
                next_tokens = []
                for b in range(probs.size(0)):
                    p = probs[b].cpu().numpy()
                    # top-k
                    if top_k > 0:
                        topk_idx = np.argpartition(-p, min(top_k, len(p)-1))[:min(top_k, len(p))]
                        mask = np.zeros_like(p, dtype=bool)
                        mask[topk_idx] = True
                        p = p * mask
                    # top-p (nucleus)
                    if top_p < 1.0:
                        sorted_idx = np.argsort(-p)
                        sorted_p = p[sorted_idx]
                        cumsum = np.cumsum(sorted_p)
                        cutoff = np.searchsorted(cumsum, top_p)
                        allowed = sorted_idx[:cutoff+1] if cutoff < len(sorted_idx) else sorted_idx
                        mask = np.zeros_like(p, dtype=bool)
                        mask[allowed] = True
                        p = p * mask
                    if p.sum() <= 0:
                        # fallback to uniform
                        p = np.ones_like(p) / len(p)
                    else:
                        p = p / p.sum()
                    next_id = np.random.choice(len(p), p=p)
                    next_tokens.append(next_id)
                next_tensor = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
                next_id = int(next_tokens[0])
                if next_id == eos_token_id:
                    break
                generated.append(next_id)
                cur_batch = next_tensor
            if not generated:
                fallback = [10, 13, 33, 39]
                generated = random.choices(fallback, k=min(2, max_length))
            return generated

    # ----------------------
    # метаподдержка: расширение словаря
    # ----------------------
    def expand_vocab(self, new_vocab_size: int):
        if new_vocab_size <= self.vocab_size:
            return
        # expand embedding
        old_emb = self.embedding.weight.data.cpu()
        old_out = self.final_proj.weight.data.cpu()
        old_vocab = self.vocab_size

        new_embedding = nn.Embedding(new_vocab_size, self.embed_dim, padding_idx=0)
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
        new_embedding.weight.data[:old_vocab, :self.embed_dim] = old_emb
        self.embedding = new_embedding.to(self.device)

        new_final = nn.Linear(self.hidden_size, new_vocab_size)
        nn.init.xavier_uniform_(new_final.weight)
        new_final.bias.data.zero_()
        new_final.weight.data[:, :old_vocab] = old_out.t()
        # note: final_proj expects [H -> V], assignment above sets columns up to old_vocab with old_out.T
        self.final_proj = new_final.to(self.device)

        self.vocab_size = new_vocab_size

    # ----------------------
    # отражение и обучение
    # ----------------------
    def _check_cell_creation(self, user_input: str, qwen_resp: str, brain_resp: str):
        if self.meta_cog is None:
            return False
        unknown = self.meta_cog.detect_unknown_words(qwen_resp)
        if unknown:
            for word in list(unknown)[:2]:
                clean_word = safe_cell_name(word)
                self.meta_cog.unknown_concepts.add(word)
                self.add_cell(f"concept_{clean_word}", self.hidden_size, self.hidden_size, "association")
                print(f"🧬 Создана клетка для нового понятия: {word} → concept_{clean_word}")

        high_activation = [cid for cid, act in self.cell_activations.items() if act > self.activation_threshold]
        if len(high_activation) >= 3 and random.random() < 0.25:
            cell_types = ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"]
            new_type = random.choice(cell_types)
            new_id = self.add_cell(f"{new_type}_adaptive", self.hidden_size, self.hidden_size, new_type)
            print(f"🧬 Адаптивная клетка создана: {new_id}")
            return True
        return False

    def reflect_and_learn(self, user_input: str, qwen_response: str, vocab: Dict[str, int], ivocab: Dict[int, str]) -> Optional[str]:
        # use network to generate a local response and decide whether to ask a clarification
        input_tokens = self.text_to_tokens(user_input, vocab)
        if not input_tokens:
            return None
        with torch.no_grad():
            brain_tokens = self.generate_sequence(torch.tensor([input_tokens], dtype=torch.long, device=self.device), max_length=15)
            brain_response = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)
        tokens = brain_response.split()
        if not tokens:
            return None
        unk_ratio = brain_response.count("<UNK>") / len(tokens)
        if unk_ratio > 0.7:
            return None
        assert self.meta_cog is not None
        should_reflect, reason = self.meta_cog.should_reflect(user_input, qwen_response, brain_response)
        if should_reflect:
            question = self._formulate_deep_question(user_input, qwen_response, brain_response, vocab)
            self.meta_cog.log_reflection(user_input, qwen_response, brain_response, question, reason)
            return question
        return None

    def _formulate_deep_question(self, user_input: str, qwen_resp: str, brain_resp: str, vocab: Dict[str, int]) -> str:
        unknown = self.meta_cog.detect_unknown_words(qwen_resp) if self.meta_cog else set()
        if unknown:
            word = next(iter(unknown))
            if len(word) > 2:
                return f"что значит '{word}'?"
        if "нет" in brain_resp.lower() and "да" in qwen_resp.lower():
            return f"противоречие: ты сказал '{brain_resp}', а Qwen — '{qwen_resp[:60]}...'. Кто прав?"
        return f"объясни подробнее: {user_input}"

    def text_to_tokens(self, text: str, vocab: Dict[str, int]) -> List[int]:
        # простая токенизация по пробелам и базовой пунктуации
        text_proc = text.lower()
        text_proc = text_proc.replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('!', ' ! ')
        words = text_proc.split()
        tokens = []
        for word in words:
            if word not in vocab:
                # новый токен: добавляем в vocab внешне (caller должен обновить ivocab)
                vocab[word] = len(vocab)
            tokens.append(vocab[word])
        # ensure embedding and final_proj large enough
        if self.vocab_size < len(vocab):
            self.expand_vocab(len(vocab) + 128)
        return tokens

    def find_activated_cells_by_type(self, threshold: float = 0.15) -> Dict[str, List[str]]:
        grouped = {
            "social": [], "fact": [], "cause": [], "procedure": [],
            "opinion": [], "meta": [], "creative": [], "other": []
        }
        for cid, act in self.cell_activations.items():
            if act <= threshold:
                continue
            if any(kw in cid for kw in ["social", "greeting", "thanks"]):
                grouped["social"].append(cid)
            elif any(kw in cid for kw in ["fact", "definition"]):
                grouped["fact"].append(cid)
            elif any(kw in cid for kw in ["cause", "explanation"]):
                grouped["cause"].append(cid)
            elif any(kw in cid for kw in ["procedure", "step"]):
                grouped["procedure"].append(cid)
            elif "opinion" in cid:
                grouped["opinion"].append(cid)
            elif any(kw in cid for kw in ["meta", "question"]):
                grouped["meta"].append(cid)
            elif any(kw in cid for kw in ["creative", "metaphor"]):
                grouped["creative"].append(cid)
            else:
                grouped["other"].append(cid)
        return {k: v for k, v in grouped.items() if v}

    # ----------------------
    # сохранение / загрузка знаний
    # ----------------------
    def save_knowledge(self, filepath: str, vocab: Dict[str, int]):
        cell_configs = []
        for cell_id, cell in self.cells.items():
            config = {
                'cell_id': cell_id,
                'cell_counter': cell.cell_id,
                'input_size': cell.input_size,
                'hidden_size': cell.hidden_size,
                'cell_type': cell.cell_type,
                'gate_logit': float(cell.gate_logit.detach().cpu().item())
            }
            cell_configs.append(config)
        knowledge = {
            'model_state': self.state_dict(),
            'vocab': vocab,
            'cell_configs': cell_configs,
            'cell_counter': self.cell_counter,
            'meta_cog_log': self.meta_cog.reflection_log if self.meta_cog else [],
            'unknown_concepts': list(self.meta_cog.unknown_concepts) if self.meta_cog else [],
            'replay': list(self.replay.buffer)
        }
        torch.save(knowledge, filepath)
        print(f"💾 Сохранено {len(self.cells)} клеток, {len(vocab)} слов, {len(self.meta_cog.reflection_log) if self.meta_cog else 0} размышлений")

    def load_knowledge(self, filepath: str):
        if not os.path.exists(filepath):
            return None
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            saved_vocab = checkpoint.get('vocab', {})
            cell_configs = checkpoint.get('cell_configs', [])
            saved_state = checkpoint.get('model_state', {})
            # rebuild cells
            self.cells = nn.ModuleDict()
            self.cell_states = {}
            self.cell_activations = {}
            for config in cell_configs:
                cid = config['cell_id']
                input_size = int(config['input_size'])
                hidden_size = int(config['hidden_size'])
                cell_type = config.get('cell_type', 'generic')
                # retain original id index if present
                self.cells[cid] = BrainCell(config.get('cell_counter', 0), input_size, hidden_size, cell_type)
                # initialize gate logit from saved
                gate_val = float(config.get('gate_logit', 0.0))
                with torch.no_grad():
                    self.cells[cid].gate_logit.copy_(torch.tensor(gate_val))
                self.cell_states[cid] = (None, None)
                self.cell_activations[cid] = 0.0
            self.cell_counter = checkpoint.get('cell_counter', len(self.cells))
            # load model state (allow missing keys)
            self.load_state_dict(saved_state, strict=False)
            # meta cognition
            self.meta_cog = MetaCognition(saved_vocab, network=self)
            self.meta_cog.reflection_log = checkpoint.get('meta_cog_log', [])
            self.meta_cog.unknown_concepts = set(checkpoint.get('unknown_concepts', []))
            # replay
            try:
                saved_replay = checkpoint.get('replay', [])
                for item in saved_replay:
                    self.replay.add(item.get('input', []), item.get('target', []), item.get('meta', {}))
            except Exception:
                pass
            print(f"🧠 Загружено {len(self.cells)} клеток, словарь: {len(saved_vocab)}")
            return saved_vocab
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            traceback.print_exc()
            return None

    # ----------------------
    # консолидация / очистка
    # ----------------------
    def consolidate_knowledge(self, remove_threshold: float = 0.01, max_prune: int = 3):
        # удаляем самые слабые клетки по gate, до max_prune
        gates = [(cid, float(torch.sigmoid(self.cells[cid].gate_logit).detach().cpu().item())) for cid in self.cells]
        weak = sorted(gates, key=lambda x: x[1])[:max_prune]
        removed = 0
        for cid, g in weak:
            if g < remove_threshold and removed < max_prune and len(self.cells) > 1:
                del self.cells[cid]
                self.cell_states.pop(cid, None)
                self.cell_activations.pop(cid, None)
                self.cell_gates.pop(cid, None)
                removed += 1
                print(f"🧹 Удалена слабая клетка: {cid} (gate={g:.4f})")
        return removed

# ======================
# УЧИТЕЛЬ (LM Studio / Qwen)
# ======================

class BrainTeacher:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions", device: Optional[torch.device] = None):
        self.api_url = api_url
        self.conversation_history: List[Dict[str, Any]] = []
        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def query_qwen(self, prompt: str, timeout: int = 20) -> str:
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 120,
                    "stream": False
                },
                timeout=timeout
            )
            if response.status_code == 200:
                raw = response.json()['choices'][0]['message']['content'].strip()
                return clean_qwen_response(raw)
            else:
                return "Хорошо."
        except Exception as e:
            print(f"⚠️ Ошибка LM Studio: {e}")
            return "Хорошо."

    def teach_brain(self, brain: CognitiveNetwork, user_input: str, vocab: Dict[str, int], ivocab: Dict[int, str],
                    epochs: int = 2, batch_size: int = 4, teacher_forcing_start: float = 1.0):
        raw_qwen = self.query_qwen(user_input)
        tagged_qwen = classify_and_tag_response(raw_qwen)
        print(f"👤: {user_input}")
        print(f"🤖: {tagged_qwen}")

        # tokens
        input_tokens = brain.text_to_tokens(user_input, vocab)
        response_tokens = brain.text_to_tokens(tagged_qwen, vocab)
        if not response_tokens:
            response_tokens = [vocab.get("хорошо", 15), vocab.get(".", 39)]

        eos_id = vocab.get('<EOS>', 1)
        full_seq = input_tokens + response_tokens + [eos_id]
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]

        # add to replay (experience)
        brain.replay.add(input_seq, target_seq, meta={"qwen": tagged_qwen, "raw_qwen": raw_qwen})
        self.replay_buffer.add(input_seq, target_seq, meta={"qwen": tagged_qwen})

        # prepare training batches: current example + random samples from replay
        optimizer = torch.optim.AdamW(brain.parameters(), lr=5e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        brain.train()
        total_loss = 0.0

        for ep in range(epochs):
            # build batch
            batch_inputs = [input_seq]
            batch_targets = [target_seq]
            # sample extra
            replay_samples = brain.replay.sample(batch_size - 1)
            for s in replay_samples:
                batch_inputs.append(s['input'])
                batch_targets.append(s['target'])
            # pad sequences to same length
            max_len = max(len(x) for x in batch_inputs)
            inp_padded = [x + [0] * (max_len - len(x)) for x in batch_inputs]
            tgt_padded = [x + [0] * (max_len - len(x)) for x in batch_targets]

            input_tensor = torch.tensor(inp_padded, dtype=torch.long, device=brain.device)
            target_tensor = torch.tensor(tgt_padded, dtype=torch.long, device=brain.device)

            # scheduled sampling / teacher forcing ratio decays over epochs
            teacher_forcing_ratio = max(0.1, teacher_forcing_start * (1.0 - 0.05 * ep))

            # token dropout: randomly replace some tokens with <UNK> to regularize
            token_dropout_mask = (torch.rand_like(input_tensor.float()) < 0.05)
            input_tensor[token_dropout_mask] = vocab.get('<UNK>', 2)

            optimizer.zero_grad()
            brain.reset_cell_states(input_tensor.size(0), input_tensor.device)
            logits = brain.process_sequence(input_tensor)  # [B, L, V]
            loss = criterion(logits.view(-1, brain.vocab_size), target_tensor.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.8)
            optimizer.step()
            scheduler.step(ep + 0.1)
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, epochs)
        print(f"📚 Потеря (avg): {avg_loss:.4f}")

        brain.eval()
        with torch.no_grad():
            brain_tokens = brain.generate_sequence(torch.tensor([input_tokens], dtype=torch.long, device=brain.device),
                                                   max_length=25, base_temperature=0.85)
            brain_response_raw = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)
        brain_response_clean = re.sub(r'\[(SOC|FCT|CAU|PRC|OPN|MET|CRT)\]\s*', '', brain_response_raw)

        # save conversation history
        self.conversation_history.append({
            'input': user_input,
            'qwen_tagged': tagged_qwen,
            'brain_raw': brain_response_raw,
            'brain_clean': brain_response_clean,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        })

        print(f"🧠 Сырой: {brain_response_raw}")
        print(f"💬 Чистый: {brain_response_clean}")

        # periodic consolidation
        if random.random() < 0.12:
            pruned = brain.consolidate_knowledge(remove_threshold=0.02, max_prune=2)
            if pruned:
                print(f"🧠 Консолидация: удалено {pruned} клеток")

        return tagged_qwen, brain_response_clean

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

def create_initial_vocabulary() -> Dict[str, int]:
    # избавился от дублирования 'что'
    return {
        '<PAD>': 0, '<EOS>': 1, '<UNK>': 2,
        '[SOC]': 3, '[FCT]': 4, '[CAU]': 5,
        '[PRC]': 6, '[OPN]': 7, '[MET]': 8, '[CRT]': 9,
        'привет': 10, 'здравствуй': 11, 'добрый': 12, 'день': 13,
        'как': 14, 'ты': 15, 'дела': 16, 'что': 17, 'такое': 18,
        'это': 19, 'потому': 20, 'чтобы': 21,
        'я': 22, 'думаю': 23, 'мне': 24, 'кажется': 25,
        'представь': 26, 'вообрази': 27, 'жизнь': 28,
        'спасибо': 29, 'пожалуйста': 30, 'извини': 31,
        'хорошо': 32, 'ладно': 33, 'ок': 34,
        'нейрон': 35, 'гравитация': 36, 'вода': 37,
        '.': 38, ',': 39, '?': 40, '!': 41
    }

# ======================
# MAIN
# ======================

def main():
    print("🧠 Запуск когнитивного агента с улучшенным обучением...")
    vocab = create_initial_vocabulary()
    vocab_size = len(vocab) + 4096  # оставляем запас
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    brain = CognitiveNetwork(vocab_size=vocab_size, embedding_dim=192, hidden_size=384, eos_token_id=vocab['<EOS>'], device=device)
    teacher = BrainTeacher(device=device)

    loaded_vocab = brain.load_knowledge("brain_knowledge.pth")
    if loaded_vocab is not None:
        vocab = loaded_vocab
        print("✅ Знания загружены и словарь обновлён")
    else:
        brain._initialize_base_cells()
        brain.cell_counter = len(brain.cells)
        brain.meta_cog = MetaCognition(vocab, network=brain)
        print("📝 Создана новая сеть с метакогнитивным модулем")

    ivocab = {v: k for k, v in vocab.items()}
    print(f"🔢 Клеток: {len(brain.cells)} | 📚 Словарь: {len(vocab)} слов")
    print("💬 Готов к диалогу! (введите 'выход' для завершения)")

    conversation_count = 0
    reflection_buffer: List[str] = []
    MAX_REFLECTION_CHAIN = 4
    reflection_chain = 0

    try:
        while True:
            if reflection_buffer:
                user_input = reflection_buffer.pop(0)
                print(f"\n💭 Мозг задаёт вопрос: {user_input}")
            else:
                user_input = input("\n👤 Вы: ").strip()
                if user_input.lower() in ['выход', 'exit', 'quit']:
                    break
                if not user_input:
                    continue
                reflection_chain = 0

            qwen_resp, brain_resp = teacher.teach_brain(brain, user_input, vocab, ivocab)
            ivocab = {v: k for k, v in vocab.items()}

            new_question = brain.reflect_and_learn(user_input, qwen_resp, vocab, ivocab)
            if new_question:
                recent_inputs = {entry['input'] for entry in teacher.conversation_history[-5:]}
                if new_question not in recent_inputs and new_question not in reflection_buffer:
                    if reflection_chain < MAX_REFLECTION_CHAIN:
                        reflection_buffer.append(new_question)
                        reflection_chain += 1
                    else:
                        print("🧠 Цепочка размышлений прервана (слишком глубоко)")
                        reflection_chain = 0

            brain._check_cell_creation(user_input, qwen_resp, brain_resp)

            activated_by_type = brain.find_activated_cells_by_type(threshold=0.15)
            if activated_by_type:
                print("🔬 Активные клетки по типу:")
                for cell_type, cells in activated_by_type.items():
                    display_cells = ', '.join(cells[:2]) + ('...' if len(cells) > 2 else '')
                    print(f"  • {cell_type.upper()}: {len(cells)} клеток ({display_cells})")

            conversation_count += 1
            if conversation_count % 2 == 0:
                brain.save_knowledge("brain_knowledge.pth", vocab)
                print("💾 Сохранено!")

    except KeyboardInterrupt:
        print("\n🛑 Прервано")
    except Exception as e:
        print(f"❌ Ошибка main loop: {e}")
        traceback.print_exc()
    finally:
        brain.save_knowledge("brain_knowledge.pth", vocab)
        print(f"\n🧠 Итог: клеток = {len(brain.cells)}, словарь = {len(vocab)} слов")
        if brain.meta_cog:
            print(f"🤔 Размышлений: {len(brain.meta_cog.reflection_log)}")
            print(f"❓ Неизвестных понятий: {len(brain.meta_cog.unknown_concepts)}")

if __name__ == "__main__":
    main()