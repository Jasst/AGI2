import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import requests
import os
import traceback


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
        return logits, hx, cx, self.activation_level


# ======================
# КОГНИТИВНАЯ СЕТЬ С САМОРАЗВИТИЕМ
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
        self.thought_cycles = 2
        self.reflection_depth = 3  # глубина размышлений

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
        cell_id = f"{cell_type}_{self.cell_counter}"
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

        # Perception
        perception_logits = []
        for cid in [c for c in self.cells if "perception" in c]:
            cell = self.cells[cid]
            hx, cx = self.cell_states[cid]
            if hx is None:
                hx = torch.zeros(batch_size, self.hidden_size, device=device)
                cx = torch.zeros(batch_size, self.hidden_size, device=device)
            logits, new_hx, new_cx, act = cell(token_emb, hx, cx)
            self.cell_states[cid] = (new_hx, new_cx)
            self.cell_activations[cid] = act
            perception_logits.append(logits)
        x = torch.mean(torch.stack(perception_logits), dim=0) if perception_logits else token_emb

        # Memory
        memory_logits = []
        for cid in [c for c in self.cells if "memory" in c]:
            cell = self.cells[cid]
            hx, cx = self.cell_states[cid]
            logits, new_hx, new_cx, act = cell(x, hx, cx)
            self.cell_states[cid] = (new_hx, new_cx)
            self.cell_activations[cid] = act
            memory_logits.append(logits)
        x = torch.mean(torch.stack(memory_logits), dim=0) if memory_logits else x

        # Association
        assoc_logits = []
        for cid in [c for c in self.cells if "association" in c]:
            cell = self.cells[cid]
            hx, cx = self.cell_states[cid]
            logits, new_hx, new_cx, act = cell(x, hx, cx)
            self.cell_states[cid] = (new_hx, new_cx)
            self.cell_activations[cid] = act
            assoc_logits.append(logits)
        x = torch.mean(torch.stack(assoc_logits), dim=0) if assoc_logits else x

        # Generation
        gen_logits = []
        for cid in [c for c in self.cells if "generation" in c]:
            cell = self.cells[cid]
            hx, cx = self.cell_states[cid]
            logits, new_hx, new_cx, act = cell(x, hx, cx)
            self.cell_states[cid] = (new_hx, new_cx)
            self.cell_activations[cid] = act
            gen_logits.append(logits)

        if gen_logits:
            final_logits = torch.mean(torch.stack(gen_logits), dim=0)
        else:
            temp_layer = nn.Linear(x.size(-1), self.vocab_size).to(x.device)
            final_logits = temp_layer(x)

        return final_logits

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

    def generate_sequence(self, input_tokens: torch.Tensor, max_length: int = 20, temperature: float = 0.9) -> List[int]:
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
            fallback = [2, 3, 16, 22]  # привет, здравствуй, спасибо, помоги
            generated = random.choices(fallback, k=2)
        return generated

    def _check_cell_creation(self):
        high_activation = [cid for cid, act in self.cell_activations.items() if act > self.activation_threshold]
        if len(high_activation) >= 2 and random.random() < 0.4:
            cell_types = ["perception", "memory", "association", "generation"]
            new_type = random.choice(cell_types)
            in_size = self.hidden_size
            out_size = self.vocab_size if new_type == "generation" else self.hidden_size
            new_id = self.add_cell(f"{new_type}_adaptive", in_size, self.hidden_size, out_size, new_type)
            print(f"🧬 Создана адаптивная клетка: {new_id}")
            return True
        return False

    def reflect_and_learn(self, user_input: str, qwen_response: str, vocab: Dict[str, int], ivocab: Dict[int, str]) -> Optional[str]:
        """
        Цикл размышлений: мозг пытается понять, есть ли в ответе Qwen новая информация,
        которую можно превратить в знание или вопрос.
        """
        # Генерируем свой ответ
        input_tokens = self.text_to_tokens(user_input, vocab)
        with torch.no_grad():
            brain_tokens = self.generate_sequence(torch.tensor([input_tokens], dtype=torch.long), max_length=15)
            brain_response = " ".join(ivocab.get(tid, "?") for tid in brain_tokens)

        # Если ответы сильно различаются — есть повод задуматься
        if brain_response.strip() and qwen_response.strip():
            similarity = self._estimate_similarity(brain_response, qwen_response)
            if similarity < 0.3:  # низкое сходство → новая информация
                # Попробуем сформулировать вопрос на основе расхождения
                reflection_question = self._formulate_question(user_input, qwen_response, brain_response, vocab, ivocab)
                if reflection_question:
                    print(f"🤔 Размышление: {reflection_question}")
                    return reflection_question
        return None

    def _estimate_similarity(self, resp1: str, resp2: str) -> float:
        words1 = set(resp1.lower().split())
        words2 = set(resp2.lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def _formulate_question(self, user_input: str, qwen_resp: str, brain_resp: str, vocab: Dict[str, int], ivocab: Dict[int, str]) -> Optional[str]:
        # Простая эвристика: если Qwen дал длинный ответ, а мозг — короткий, спросим "почему?"
        if len(qwen_resp.split()) > 5 and len(brain_resp.split()) < 3:
            new_question = f"почему {user_input} ?"
            return new_question
        # Или: если в ответе есть новые слова — спросим о них
        brain_words = set(brain_resp.lower().split())
        qwen_words = set(qwen_resp.lower().split())
        new_words = qwen_words - brain_words
        if new_words:
            word = next(iter(new_words))
            if len(word) > 2 and word.isalpha():
                return f"что значит {word} ?"
        return None

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
        }
        torch.save(knowledge, filepath)
        print(f"💾 Сохранено {len(self.cells)} клеток и словарь из {len(vocab)} слов")

    def load_knowledge(self, filepath: str):
        if not os.path.exists(filepath):
            return None
        try:
            knowledge = torch.load(filepath, map_location='cpu')
            vocab = knowledge['vocab']
            cell_configs = knowledge['cell_configs']

            self.cells = nn.ModuleDict()
            self.cell_states = {}
            self.cell_activations = {}

            for config in cell_configs:
                self.cells[config['cell_id']] = BrainCell(
                    cell_id=int(config['cell_id'].split('_')[-1]),
                    input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    output_size=config['output_size'],
                    cell_type=config['cell_type']
                )
                self.cell_states[config['cell_id']] = (None, None)
                self.cell_activations[config['cell_id']] = 0.0

            self.cell_counter = knowledge.get('cell_counter', len(self.cells))
            self.load_state_dict(knowledge['model_state'], strict=True)
            print(f"🧠 Загружено {len(self.cells)} клеток и словарь из {len(vocab)} слов")
            return vocab
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            traceback.print_exc()
            return None


# ======================
# УЧИТЕЛЬ С ЦИКЛОМ РАЗМЫШЛЕНИЙ
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
                    "model": "qwen2.5:0.5b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 80,
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return "Привет! (ошибка API)"
        except Exception as e:
            return f"Привет! (ошибка: {str(e)})"

    def teach_brain(self, brain: CognitiveNetwork, user_input: str, vocab: Dict[str, int], ivocab: Dict[int, str]):
        qwen_response = self.query_qwen(user_input)
        print(f"👤: {user_input}")
        print(f"🤖: {qwen_response}")

        input_tokens = brain.text_to_tokens(user_input, vocab)
        response_tokens = brain.text_to_tokens(qwen_response, vocab)

        if not response_tokens:
            return qwen_response, "?"

        eos_id = vocab.get('<EOS>', 1)
        full_seq = input_tokens + response_tokens + [eos_id]
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]

        input_tensor = torch.tensor([input_seq], dtype=torch.long)
        target_tensor = torch.tensor([target_seq], dtype=torch.long)

        optimizer = torch.optim.AdamW(brain.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        brain.train()
        for _ in range(3):
            optimizer.zero_grad()
            logits = brain.process_sequence(input_tensor)
            loss = criterion(logits.view(-1, brain.vocab_size), target_tensor.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
            optimizer.step()

        print(f"📚 Потеря: {loss.item():.4f}")

        brain.eval()
        with torch.no_grad():
            brain_tokens = brain.generate_sequence(
                torch.tensor([input_tokens], dtype=torch.long),
                max_length=15,
                temperature=0.85
            )
            brain_response = " ".join(ivocab.get(tid, "?") for tid in brain_tokens)

        self.conversation_history.append({
            'input': user_input,
            'response': qwen_response,
            'brain_response': brain_response,
            'timestamp': datetime.now().isoformat()
        })

        return qwen_response, brain_response


# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

def create_initial_vocabulary() -> Dict[str, int]:
    return {
        '<PAD>': 0, '<EOS>': 1,
        'привет': 2, 'здравствуй': 3, 'добрый': 4, 'день': 5,
        'как': 6, 'ты': 7, 'дела': 8, 'что': 9, 'мне': 10,
        'ответь': 11, 'да': 12, 'нет': 13, 'хорошо': 14, 'плохо': 15,
        'спасибо': 16, 'пожалуйста': 17, 'извини': 18,
        'понимаю': 19, 'не': 20, 'знаю': 21, 'помоги': 22,
        'объясни': 23, 'расскажи': 24, 'человек': 25,
        'мир': 26, 'время': 27, 'жизнь': 28, 'работа': 29,
        'город': 30, 'дом': 31, 'семья': 32,
        '.': 33, ',': 34, '?': 35, '!': 36
    }


# ======================
# MAIN — С ЦИКЛОМ САМОРАЗВИТИЯ
# ======================

def main():
    print("🧠 Запуск самообучающегося когнитивного агента...")
    vocab = create_initial_vocabulary()
    brain = CognitiveNetwork(vocab_size=15000, eos_token_id=vocab['<EOS>'])
    teacher = BrainTeacher()

    loaded_vocab = brain.load_knowledge("brain_knowledge.pth")
    if loaded_vocab is not None:
        vocab = loaded_vocab
        print("✅ Знания (включая клетки) загружены")
    else:
        brain._initialize_base_cells()
        brain.cell_counter = len(brain.cells)
        print("📝 Создана новая сеть с базовыми клетками")

    ivocab = {v: k for k, v in vocab.items()}
    print(f"🔢 Клеток: {len(brain.cells)} | 📚 Словарь: {len(vocab)} слов")
    print("💬 Готов к диалогу и самообучению! (введите 'выход' для завершения)")

    conversation_count = 0
    reflection_buffer = []

    while True:
        try:
            # Сначала обрабатываем отложенные вопросы от размышлений
            if reflection_buffer:
                user_input = reflection_buffer.pop(0)
                print(f"\n💭 Мозг задаёт вопрос: {user_input}")
            else:
                user_input = input("\n👤 Вы: ").strip()
                if user_input.lower() in ['выход', 'exit', 'quit']:
                    break
                if not user_input:
                    continue

            qwen_resp, brain_resp = teacher.teach_brain(brain, user_input, vocab, ivocab)
            ivocab = {v: k for k, v in vocab.items()}

            print(f"🧠 Мозг: {brain_resp}")

            # === ЦИКЛ РАЗМЫШЛЕНИЙ ===
            new_question = brain.reflect_and_learn(user_input, qwen_resp, vocab, ivocab)
            if new_question and new_question not in [q['input'] for q in teacher.conversation_history[-5:]]:
                reflection_buffer.append(new_question)

            # Адаптивное создание клеток
            if brain._check_cell_creation():
                pass  # уже печатается в _check_cell_creation

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
    print(f"💬 Диалогов: {len(teacher.conversation_history)}")


if __name__ == "__main__":
    main()