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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОЧИСТКИ
# ======================

def clean_qwen_response(text: str) -> str:
    """Очищает ответ Qwen от markdown, лишней пунктуации и обрезает до разумного размера."""
    # Удаляем markdown-разметку
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # **жирный** → жирный
    text = re.sub(r'#{1,3}\s*', '', text)  # ### Заголовок → Заголовок
    text = re.sub(r'>\s*', '', text)  # > Цитата → Цитата
    text = re.sub(r'\n+', ' ', text)  # Многострочность → пробел
    text = re.sub(r'\s+', ' ', text)  # Много пробелов → один
    text = text.strip()

    # Удаляем trailing пунктуацию и эмодзи в начале/конце
    text = re.sub(r'^[\*\.\!\?\:\-\–—\s]+', '', text)
    text = re.sub(r'[\*\.\!\?\:\-\–—\s]+$', '', text)

    # Обрезаем до ~60 слов (чтобы не перегружать обучение)
    words = text.split()
    if len(words) > 60:
        text = ' '.join(words[:60])
        if not text.endswith(('.', '!', '?')):
            text += '.'

    return text if text else "Хорошо."


def safe_cell_name(base: str) -> str:
    """Преобразует строку в безопасное имя для PyTorch ModuleDict."""
    # Разрешены: буквы, цифры, подчёркивания
    name = re.sub(r'[^a-zA-Zа-яА-Я0-9_]', '_', base)
    name = re.sub(r'_+', '_', name)  # несколько _ → один
    name = name.strip('_')
    return name if name else "unknown"


def clean_for_similarity(text: str) -> str:
    """Очищает текст для оценки сходства: удаляет пунктуацию и эмодзи."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


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
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.emotion_bias = nn.Parameter(torch.randn(output_size) * 0.01)
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
        logits = self.output_proj(associated) + self.emotion_bias
        probs = F.softmax(logits, dim=-1)
        self.confidence = torch.max(probs).item()
        return logits, hx, cx, self.activation_level


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

    def should_reflect(self, brain_resp: str, qwen_resp: str, similarity: float) -> bool:
        # Не размышлять, если Qwen дал шаблонный ответ
        if any(phrase in qwen_resp.lower() for phrase in ["ошибка", "извини", "не могу", "я - qwen"]):
            return False
        return similarity < 0.4 or len(brain_resp.strip().split()) < 2


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

        self.cells = nn.ModuleDict()
        self.cell_states = {}
        self.cell_activations = {}

        self.cell_counter = 0
        self.activation_threshold = 0.25
        self.thought_cycles = 3

        self.meta_cog = None

    def _initialize_base_cells(self):
        base_cells = [
            ("perception_word", "perception"),
            ("perception_context", "perception"),
            ("memory_short", "memory"),
            ("memory_long", "memory"),
            ("association_semantic", "association"),
            ("association_emotional", "association"),
            ("generation_response", "generation"),
            ("generation_creative", "generation")
        ]
        for name, ctype in base_cells:
            in_size = self.hidden_size
            out_size = self.vocab_size if ctype == "generation" else self.hidden_size
            self.add_cell(name, in_size, self.hidden_size, out_size, ctype)

    def add_cell(self, cell_type: str, input_size: int, hidden_size: int, output_size: int, cell_subtype: str):
        safe_type = safe_cell_name(cell_type)
        cell_id = f"{safe_type}_{self.cell_counter}"
        self.cells[cell_id] = BrainCell(self.cell_counter, input_size, hidden_size, output_size, cell_subtype)
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
        x = token_emb

        stages = ["perception", "memory", "association", "generation"]
        for stage in stages:
            stage_logits = []
            for cid in [c for c in self.cells if stage in c]:
                cell = self.cells[cid]
                hx, cx = self.cell_states[cid]
                if hx is None:
                    hx = torch.zeros(batch_size, self.hidden_size, device=device)
                    cx = torch.zeros(batch_size, self.hidden_size, device=device)
                logits, new_hx, new_cx, act = cell(x, hx, cx)
                self.cell_states[cid] = (new_hx, new_cx)
                self.cell_activations[cid] = act
                stage_logits.append(logits)
            if stage_logits:
                x = torch.mean(torch.stack(stage_logits), dim=0)

        if not any("generation" in c for c in self.cells):
            temp_layer = nn.Linear(x.size(-1), self.vocab_size).to(x.device)
            x = temp_layer(x)
        return x

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

    def generate_sequence(self, input_tokens: torch.Tensor, max_length: int = 20, temperature: float = 0.9) -> List[
        int]:
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
            fallback = [3, 4, 17, 23]
            generated = random.choices(fallback, k=min(2, max_length))
        return generated

    def _check_cell_creation(self, user_input: str, qwen_resp: str, brain_resp: str):
        unknown = self.meta_cog.detect_unknown_words(qwen_resp)
        if unknown:
            for word in list(unknown)[:2]:
                clean_word = safe_cell_name(word)
                self.meta_cog.unknown_concepts.add(word)
                self.add_cell(f"concept_{clean_word}", self.hidden_size, self.hidden_size, self.hidden_size,
                              "association")
                print(f"🧬 Создана клетка для нового понятия: {word} → concept_{clean_word}")

        high_activation = [cid for cid, act in self.cell_activations.items() if act > self.activation_threshold]
        if len(high_activation) >= 3 and random.random() < 0.3:
            cell_types = ["perception", "memory", "association", "generation"]
            new_type = random.choice(cell_types)
            in_size = self.hidden_size
            out_size = self.vocab_size if new_type == "generation" else self.hidden_size
            new_id = self.add_cell(f"{new_type}_adaptive", in_size, self.hidden_size, out_size, new_type)
            print(f"🧬 Адаптивная клетка создана: {new_id}")
            return True
        return False

    def reflect_and_learn(self, user_input: str, qwen_response: str, vocab: Dict[str, int], ivocab: Dict[int, str]) -> \
    Optional[str]:
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

        # Используем очищенные версии для сравнения
        clean_brain = clean_for_similarity(brain_response)
        clean_qwen = clean_for_similarity(qwen_response)
        similarity = self._estimate_similarity_semantic(clean_brain, clean_qwen)

        if self.meta_cog.should_reflect(brain_response, qwen_response, similarity):
            question = self._formulate_deep_question(user_input, qwen_response, brain_response, vocab)
            self.meta_cog.log_reflection(user_input, qwen_response, brain_response, question, "низкое сходство")
            return question
        return None

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

    def _formulate_deep_question(self, user_input: str, qwen_resp: str, brain_resp: str, vocab: Dict[str, int]) -> str:
        unknown = self.meta_cog.detect_unknown_words(qwen_resp)
        if unknown:
            word = next(iter(unknown))
            if len(word) > 2:
                return f"что значит '{word}'?"

        if len(qwen_resp.split()) > 6 and len(brain_resp.split()) < 3:
            return f"почему на '{user_input}' отвечают так: '{qwen_resp[:50]}...'?"

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

    def find_activated_cells(self, threshold: float = 0.2) -> List[str]:
        return [cid for cid, act in self.cell_activations.items() if act > threshold]

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
        print(
            f"💾 Сохранено {len(self.cells)} клеток, {len(vocab)} слов, {len(self.meta_cog.reflection_log)} размышлений")

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
                output_size = config['output_size']
                cell_type = config['cell_type']
                if cell_type == "generation":
                    output_size = self.vocab_size

                safe_type = safe_cell_name(config['cell_id'].rsplit('_', 1)[0])
                cell_id = f"{safe_type}_{config['cell_id'].rsplit('_', 1)[1]}"

                self.cells[cell_id] = BrainCell(
                    cell_id=int(config['cell_id'].split('_')[-1]),
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    cell_type=cell_type
                )
                self.cell_states[cell_id] = (None, None)
                self.cell_activations[cell_id] = 0.0

            self.cell_counter = checkpoint.get('cell_counter', len(self.cells))

            new_state_dict = self.state_dict()
            saved_vocab_size = len(saved_vocab)
            current_vocab_size = self.vocab_size

            for name, param in saved_state_dict.items():
                if name not in new_state_dict:
                    continue

                if name == 'embedding.weight':
                    saved_weight = param
                    current_weight = new_state_dict[name]
                    min_size = min(saved_vocab_size, current_vocab_size)
                    current_weight[:min_size] = saved_weight[:min_size]
                    if current_vocab_size > saved_vocab_size:
                        unk_idx = saved_vocab.get('<UNK>', 2)
                        if unk_idx < saved_vocab_size:
                            unk_vec = saved_weight[unk_idx]
                            current_weight[saved_vocab_size:] = unk_vec
                        else:
                            nn.init.normal_(current_weight[saved_vocab_size:], mean=0.0, std=0.02)
                    new_state_dict[name] = current_weight

                elif name.startswith('cells.') and ('output_proj.weight' in name or 'output_proj.bias' in name):
                    cell_id = name.split('.')[1]
                    if cell_id in self.cells:
                        cell = self.cells[cell_id]
                        if cell.cell_type == "generation":
                            saved_param = param
                            current_param = new_state_dict[name]
                            min_out = min(saved_param.size(0), current_param.size(0))
                            current_param[:min_out] = saved_param[:min_out]
                            if current_param.size(0) > saved_param.size(0):
                                if 'weight' in name:
                                    nn.init.normal_(current_param[min_out:], mean=0.0, std=0.02)
                                else:
                                    nn.init.zeros_(current_param[min_out:])
                            new_state_dict[name] = current_param
                        else:
                            if param.shape == new_state_dict[name].shape:
                                new_state_dict[name] = param
                    else:
                        print(f"⚠️ Пропущен параметр клетки: {name}")
                elif param.shape == new_state_dict[name].shape:
                    new_state_dict[name] = param
                else:
                    print(f"⚠️ Пропущен параметр из-за несовпадения формы: {name}")

            self.load_state_dict(new_state_dict, strict=False)

            self.meta_cog = MetaCognition(saved_vocab)
            self.meta_cog.reflection_log = checkpoint.get('meta_cog_log', [])
            self.meta_cog.unknown_concepts = set(checkpoint.get('unknown_concepts', []))

            print(
                f"🧠 Загружено {len(self.cells)} клеток, словарь расширен до {current_vocab_size} (был {saved_vocab_size})")
            return saved_vocab

        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            traceback.print_exc()
            return None


# ======================
# УЧИТЕЛЬ
# ======================

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
        qwen_response = self.query_qwen(user_input)
        print(f"👤: {user_input}")
        print(f"🤖: {qwen_response}")

        input_tokens = brain.text_to_tokens(user_input, vocab)
        response_tokens = brain.text_to_tokens(qwen_response, vocab)

        if not response_tokens:
            return qwen_response, "<пусто>"

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
                max_length=15,
                temperature=0.85
            )
            brain_response = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)

        self.conversation_history.append({
            'input': user_input,
            'response': qwen_response,
            'brain_response': brain_response,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        })

        return qwen_response, brain_response


# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

def create_initial_vocabulary() -> Dict[str, int]:
    return {
        '<PAD>': 0, '<EOS>': 1, '<UNK>': 2,
        'привет': 3, 'здравствуй': 4, 'добрый': 5, 'день': 6,
        'как': 7, 'ты': 8, 'дела': 9, 'что': 10, 'мне': 11,
        'ответь': 12, 'да': 13, 'нет': 14, 'хорошо': 15, 'плохо': 16,
        'спасибо': 17, 'пожалуйста': 18, 'извини': 19,
        'понимаю': 20, 'не': 21, 'знаю': 22, 'помоги': 23,
        'объясни': 24, 'расскажи': 25, 'человек': 26,
        'мир': 27, 'время': 28, 'жизнь': 29, 'работа': 30,
        'город': 31, 'дом': 32, 'семья': 33,
        '.': 34, ',': 35, '?': 36, '!': 37
    }


# ======================
# MAIN
# ======================

def main():
    print("🧠 Запуск когнитивного агента с метакогнитивными размышлениями...")
    vocab = create_initial_vocabulary()
    vocab_size = len(vocab) + 10000
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
            print(f"🧠 Мозг: {brain_resp}")

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

            activated = brain.find_activated_cells(0.15)
            if activated:
                print(f"🔬 Активные клетки ({len(activated)}): {', '.join(activated[:4])}")

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