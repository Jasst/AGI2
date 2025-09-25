import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
import requests
import os
import traceback
from collections import Counter
import re


# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОЧИСТКИ И РАЗМЕТКИ
# ======================

def clean_qwen_response(text: str) -> str:
    """Очищает ответ Qwen от markdown, лишней пунктуации и обрезает до разумного размера."""
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'>\s*', '', text)
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

    return text if text else "Хорошо."


def safe_cell_name(base: str) -> str:
    name = re.sub(r'[^a-zA-Zа-яА-Я0-9_]', '_', base)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else "unknown"


def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


# ======================
# РАСШИРЕННАЯ РАЗМЕТКА ЗНАНИЙ
# ======================

def classify_and_tag_response(text: str) -> str:
    if not text.strip():
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

    social_keywords = ["привет", "здравствуй", "добрый", "спасибо", "пожалуйста", "извини", "хорошо", "ладно", "ок", "приветствую"]
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
    def __init__(self, cell_id: int, input_size: int, hidden_size: int, output_size: int, cell_type: str = "generic"):
        super().__init__()
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

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
        # УБРАНО: output_proj и emotion_bias — они не нужны внутри клетки
        self.activation_level = 0.0
        self.confidence = 0.0

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None, cx: Optional[torch.Tensor] = None):
        batch_size = x.size(0)
        device = x.device

        x = self.input_adapter(x)
        perceived = self.perception(x)
        self.activation_level = torch.mean(torch.abs(perceived)).item()

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=device)
        if cx is None:
            cx = torch.zeros(batch_size, self.hidden_size, device=device)

        hx, cx = self.memory(perceived, (hx, cx))
        associated = self.association(hx)
        # Возвращаем ТОЛЬКО скрытое состояние
        return associated, hx, cx, self.activation_level


# ======================
# МЕТАКОГНИТИВНЫЙ МОДУЛЬ
# ======================

class MetaCognition:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.unknown_concepts: Set[str] = set()
        self.reflection_log: List[Dict] = []

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

    def _estimate_similarity_semantic(self, resp1: str, resp2: str) -> float:
        words1 = resp1.split()
        words2 = resp2.split()
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        all_words = words1 + words2
        freq = Counter(all_words)
        tf1 = Counter(words1)
        tf2 = Counter(words2)

        vec1 = np.array([tf1.get(w, 0) / freq[w] for w in freq])
        vec2 = np.array([tf2.get(w, 0) / freq[w] for w in freq])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(cos_sim)

    def should_reflect(self, user_input: str, qwen_resp: str, brain_resp: str, vocab: Dict[str, int]) -> Tuple[bool, str]:
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
            reason = f"ожидалось творчество [CRT], но Qwen дал {actual_type}"

        clean_brain = clean_for_similarity(brain_resp)
        clean_qwen = clean_for_similarity(qwen_resp)
        similarity = self._estimate_similarity_semantic(clean_brain, clean_qwen)

        if type_mismatch or similarity < 0.35:
            return True, reason if type_mismatch else "низкое сходство"
        return False, ""


# ======================
# КОГНИТИВНАЯ СЕТЬ
# ======================

class CognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_size: int = 512, eos_token_id: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_proj = nn.Linear(embedding_dim, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, vocab_size)  # ← ЕДИНСТВЕННАЯ ПРОЕКЦИЯ В СЛОВАРЬ

        self.cells = nn.ModuleDict()
        self.cell_states = {}
        self.cell_activations = {}

        self.cell_counter = 0
        self.activation_threshold = 0.25
        self.thought_cycles = 3

        self.meta_cog = None

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
            # Все клетки: input=hidden, output=hidden
            self.add_cell(name, self.hidden_size, self.hidden_size, self.hidden_size, ctype)

    def add_cell(self, cell_type: str, input_size: int, hidden_size: int, output_size: int, cell_subtype: str):
        # output_size ИГНОРИРУЕТСЯ — все клетки работают в hidden_size
        safe_type = safe_cell_name(cell_type)
        cell_id = f"{safe_type}_{self.cell_counter}"
        self.cells[cell_id] = BrainCell(self.cell_counter, input_size, hidden_size, hidden_size, cell_subtype)
        self.cell_states[cell_id] = (None, None)
        self.cell_activations[cell_id] = 0.0
        self.cell_counter += 1
        return cell_id

    def reset_cell_states(self, batch_size: int, device: torch.device):
        for cell_id in self.cells:
            self.cell_states[cell_id] = (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device)
            )

    def _process_step(self, token_emb: torch.Tensor):
        batch_size = token_emb.size(0)
        device = token_emb.device
        x = token_emb  # [B, hidden_size]

        stages = ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"]
        for stage in stages:
            stage_outputs = []
            for cid in [c for c in self.cells if stage in c]:
                cell = self.cells[cid]
                hx, cx = self.cell_states[cid]
                if hx is None:
                    hx = torch.zeros(batch_size, self.hidden_size, device=device)
                    cx = torch.zeros(batch_size, self.hidden_size, device=device)
                # Получаем СКРЫТОЕ состояние
                cell_output, new_hx, new_cx, act = cell(x, hx, cx)
                self.cell_states[cid] = (new_hx, new_cx)
                self.cell_activations[cid] = act
                stage_outputs.append(cell_output)
            if stage_outputs:
                x = torch.mean(torch.stack(stage_outputs), dim=0)

        # ЕДИНСТВЕННАЯ ПРОЕКЦИЯ В СЛОВАРЬ
        return self.final_proj(x)

    def process_sequence(self, input_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        self.reset_cell_states(batch_size, device)

        embedded = self.embedding(input_tokens)
        embedded = self.embed_proj(embedded)
        attn_out, _ = self.attention(embedded, embedded, embedded)
        combined = self.norm(embedded + attn_out)

        outputs = []
        for t in range(seq_len):
            token_emb = combined[:, t, :]
            logits = self._process_step(token_emb)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)

    def generate_sequence(self, input_tokens: torch.Tensor, max_length: int = 20, base_temperature: float = 0.9) -> List[int]:
        device = input_tokens.device
        batch_size = input_tokens.size(0)

        with torch.no_grad():
            embedded = self.embedding(input_tokens)
            embedded = self.embed_proj(embedded)
            attn_out, _ = self.attention(embedded, embedded, embedded)
            combined = self.norm(embedded + attn_out)

            for t in range(combined.size(1)):
                token_emb = combined[:, t, :]
                _ = self._process_step(token_emb)

        generated = []
        current_token = input_tokens[:, -1:]

        # Адаптивная температура по маркеру
        temperature = base_temperature
        if current_token.numel() > 0:
            last_token_id = current_token.item()
            if 3 <= last_token_id <= 9:  # ID маркеров [SOC]...[CRT]
                marker_map = {3: "SOC", 4: "FCT", 5: "CAU", 6: "PRC", 7: "OPN", 8: "MET", 9: "CRT"}
                marker = marker_map.get(last_token_id, "FCT")
                if marker in ["FCT", "CAU"]:
                    temperature = 0.3
                elif marker == "CRT":
                    temperature = 1.2
                elif marker == "OPN":
                    temperature = 0.7
                else:
                    temperature = 0.6

        for _ in range(max_length):
            with torch.no_grad():
                emb = self.embedding(current_token)
                emb = self.embed_proj(emb)
                token_emb = emb.squeeze(1)

                for _ in range(self.thought_cycles):
                    logits = self._process_step(token_emb)

                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                token_id = next_token.item()

                if token_id == self.eos_token_id or len(generated) >= max_length:
                    break

                generated.append(token_id)
                current_token = next_token

        if not generated:
            fallback = [10, 13, 33, 39]
            generated = random.choices(fallback, k=min(2, max_length))
        return generated

    def _check_cell_creation(self, user_input: str, qwen_resp: str, brain_resp: str):
        unknown = self.meta_cog.detect_unknown_words(qwen_resp)
        if unknown:
            for word in list(unknown)[:2]:
                clean_word = safe_cell_name(word)
                self.meta_cog.unknown_concepts.add(word)
                self.add_cell(f"concept_{clean_word}", self.hidden_size, self.hidden_size, self.hidden_size, "association")
                print(f"🧬 Создана клетка для нового понятия: {word} → concept_{clean_word}")

        high_activation = [cid for cid, act in self.cell_activations.items() if act > self.activation_threshold]
        if len(high_activation) >= 3 and random.random() < 0.3:
            cell_types = ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"]
            new_type = random.choice(cell_types)
            new_id = self.add_cell(f"{new_type}_adaptive", self.hidden_size, self.hidden_size, self.hidden_size, new_type)
            print(f"🧬 Адаптивная клетка создана: {new_id}")
            return True
        return False

    def reflect_and_learn(self, user_input: str, qwen_response: str, vocab: Dict[str, int], ivocab: Dict[int, str]) -> Optional[str]:
        input_tokens = self.text_to_tokens(user_input, vocab)
        with torch.no_grad():
            brain_tokens = self.generate_sequence(torch.tensor([input_tokens], dtype=torch.long), max_length=15)
            brain_response = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)

        tokens = brain_response.split()
        if not tokens:
            return None
        unk_ratio = brain_response.count("<UNK>") / len(tokens)
        if unk_ratio > 0.7:
            return None

        should_reflect, reason = self.meta_cog.should_reflect(user_input, qwen_response, brain_response, vocab)
        if should_reflect:
            question = self._formulate_deep_question(user_input, qwen_response, brain_response, vocab)
            self.meta_cog.log_reflection(user_input, qwen_response, brain_response, question, reason)
            return question
        return None

    def _formulate_deep_question(self, user_input: str, qwen_resp: str, brain_resp: str, vocab: Dict[str, int]) -> str:
        unknown = self.meta_cog.detect_unknown_words(qwen_resp)
        if unknown:
            word = next(iter(unknown))
            if len(word) > 2:
                return f"что значит '{word}'?"

        if "нет" in brain_resp.lower() and "да" in qwen_resp.lower():
            return f"противоречие: ты сказал '{brain_resp}', а Qwen — '{qwen_resp[:30]}...'. Кто прав?"

        return f"объясни подробнее: {user_input}"

    def text_to_tokens(self, text: str, vocab: Dict[str, int]) -> List[int]:
        words = text.lower().replace('.', ' .').replace(',', ' ,').replace('?', ' ?').replace('!', ' !').split()
        tokens = []
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
            tokens.append(vocab[word])
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

    def save_knowledge(self, filepath: str, vocab: Dict[str, int]):
        cell_configs = []
        for cell_id, cell in self.cells.items():
            config = {
                'cell_id': cell_id,
                'input_size': cell.input_size,
                'hidden_size': cell.hidden_size,
                'output_size': cell.output_size,
                'cell_type': cell.cell_type
            }
            cell_configs.append(config)

        knowledge = {
            'model_state': self.state_dict(),
            'vocab': vocab,
            'cell_configs': cell_configs,
            'cell_counter': self.cell_counter,
            'meta_cog_log': self.meta_cog.reflection_log if self.meta_cog else [],
            'unknown_concepts': list(self.meta_cog.unknown_concepts) if self.meta_cog else []
        }
        torch.save(knowledge, filepath)
        print(f"💾 Сохранено {len(self.cells)} клеток, {len(vocab)} слов, {len(self.meta_cog.reflection_log)} размышлений")

    def load_knowledge(self, filepath: str):
        if not os.path.exists(filepath):
            return None
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            saved_vocab = checkpoint['vocab']
            cell_configs = checkpoint['cell_configs']
            saved_state_dict = checkpoint['model_state']

            self.cells = nn.ModuleDict()
            self.cell_states = {}
            self.cell_activations = {}

            for config in cell_configs:
                input_size = config['input_size']
                hidden_size = config['hidden_size']
                cell_type = config['cell_type']

                safe_type = safe_cell_name(config['cell_id'].rsplit('_', 1)[0])
                cell_id = f"{safe_type}_{config['cell_id'].rsplit('_', 1)[1]}"

                self.cells[cell_id] = BrainCell(
                    cell_id=int(config['cell_id'].split('_')[-1]),
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,  # всегда hidden_size
                    cell_type=cell_type
                )
                self.cell_states[cell_id] = (None, None)
                self.cell_activations[cell_id] = 0.0

            self.cell_counter = checkpoint.get('cell_counter', len(self.cells))

            # Загрузка состояния с учётом final_proj
            self.load_state_dict(saved_state_dict, strict=False)

            self.meta_cog = MetaCognition(saved_vocab)
            self.meta_cog.reflection_log = checkpoint.get('meta_cog_log', [])
            self.meta_cog.unknown_concepts = set(checkpoint.get('unknown_concepts', []))

            print(f"🧠 Загружено {len(self.cells)} клеток, словарь расширен до {len(saved_vocab)}")
            return saved_vocab

        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            traceback.print_exc()
            return None


# ======================
# УЧИТЕЛЬ ДЛЯ LM STUDIO
# ======================

class BrainTeacher:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        self.conversation_history = []

    def query_qwen(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 80,
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                raw = response.json()['choices'][0]['message']['content'].strip()
                return clean_qwen_response(raw)
            else:
                return "Хорошо."
        except Exception as e:
            print(f"⚠️ Ошибка LM Studio: {e}")
            return "Хорошо."

    def teach_brain(self, brain: CognitiveNetwork, user_input: str, vocab: Dict[str, int], ivocab: Dict[int, str]):
        raw_qwen = self.query_qwen(user_input)
        tagged_qwen = classify_and_tag_response(raw_qwen)

        print(f"👤: {user_input}")
        print(f"🤖: {tagged_qwen}")

        input_tokens = brain.text_to_tokens(user_input, vocab)
        response_tokens = brain.text_to_tokens(tagged_qwen, vocab)

        if not response_tokens:
            response_tokens = [vocab.get("хорошо", 15), vocab.get(".", 34)]

        eos_id = vocab.get('<EOS>', 1)
        full_seq = input_tokens + response_tokens + [eos_id]
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]

        input_tensor = torch.tensor([input_seq], dtype=torch.long)
        target_tensor = torch.tensor([target_seq], dtype=torch.long)

        optimizer = torch.optim.AdamW(brain.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        brain.train()
        total_loss = 0.0
        for _ in range(3):
            brain.reset_cell_states(input_tensor.size(0), input_tensor.device)
            optimizer.zero_grad()
            logits = brain.process_sequence(input_tensor)
            loss = criterion(logits.view(-1, brain.vocab_size), target_tensor.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / 3
        print(f"📚 Потеря: {avg_loss:.4f}")

        brain.eval()
        with torch.no_grad():
            brain_tokens = brain.generate_sequence(
                torch.tensor([input_tokens], dtype=torch.long),
                max_length=25,
                base_temperature=0.85
            )
            brain_response_raw = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)

        brain_response_clean = re.sub(r'\[(SOC|FCT|CAU|PRC|OPN|MET|CRT)\]\s*', '', brain_response_raw)

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

        return tagged_qwen, brain_response_clean


# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

def create_initial_vocabulary() -> Dict[str, int]:
    return {
        '<PAD>': 0, '<EOS>': 1, '<UNK>': 2,
        '[SOC]': 3, '[FCT]': 4, '[CAU]': 5,
        '[PRC]': 6, '[OPN]': 7, '[MET]': 8, '[CRT]': 9,
        'привет': 10, 'здравствуй': 11, 'добрый': 12, 'день': 13,
        'как': 14, 'ты': 15, 'дела': 16, 'что': 17, 'такое': 18,
        'это': 19, 'потому': 20, 'что': 21, 'чтобы': 22,
        'я': 23, 'думаю': 24, 'мне': 25, 'кажется': 26,
        'представь': 27, 'вообрази': 28, 'жизнь': 29,
        'спасибо': 30, 'пожалуйста': 31, 'извини': 32,
        'хорошо': 33, 'ладно': 34, 'ок': 35,
        'нейрон': 36, 'гравитация': 37, 'вода': 38,
        '.': 39, ',': 40, '?': 41, '!': 42
    }


# ======================
# MAIN
# ======================

def main():
    print("🧠 Запуск когнитивного агента с типизированными знаниями...")
    vocab = create_initial_vocabulary()
    vocab_size = len(vocab) + 10000  # оставляем место для новых слов
    brain = CognitiveNetwork(vocab_size=vocab_size, eos_token_id=vocab['<EOS>'])
    teacher = BrainTeacher()

    loaded_vocab = brain.load_knowledge("brain_knowledge.pth")
    if loaded_vocab is not None:
        vocab = loaded_vocab
        print("✅ Знания загружены и словарь обновлён")
    else:
        brain._initialize_base_cells()
        brain.cell_counter = len(brain.cells)
        brain.meta_cog = MetaCognition(vocab)
        print("📝 Создана новая сеть с метакогнитивным модулем")

    ivocab = {v: k for k, v in vocab.items()}
    print(f"🔢 Клеток: {len(brain.cells)} | 📚 Словарь: {len(vocab)} слов")
    print("💬 Готов к диалогу! (введите 'выход' для завершения)")

    conversation_count = 0
    reflection_buffer = []
    MAX_REFLECTION_CHAIN = 4
    reflection_chain = 0

    while True:
        try:
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

            # 🔬 Визуализация активных клеток по типу
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
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()

    brain.save_knowledge("brain_knowledge.pth", vocab)
    print(f"\n🧠 Итог: клеток = {len(brain.cells)}, словарь = {len(vocab)} слов")
    if brain.meta_cog:
        print(f"🤔 Размышлений: {len(brain.meta_cog.reflection_log)}")
        print(f"❓ Неизвестных понятий: {len(brain.meta_cog.unknown_concepts)}")


if __name__ == "__main__":
    main()