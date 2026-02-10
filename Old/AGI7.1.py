# coding: utf-8
"""
AGI_final_improved.py
Улучшенная версия: более чёткие ответы, контекстная история, устойчивое обучение,
лучше токенизация и семантическая оценка, стабильное сохранение/загрузка словаря и опыта.

Главные изменения:
 - DialogContext для хранения N последних сообщений (контекст)
 - VocabManager: корректная сериализация/десериализация, паддинг, усечение
 - CognitiveNetwork.generate: управляемая генерация (temperature, top_k, top_p), защита от повторов n-грамм
 - Teacher: переиспользуемый оптимизатор, более надёжные запросы к Qwen, параметры обучения
 - SemanticEvaluator: robust SentenceTransformer use + cosine similarity
 - Улучшено сохранение/загрузка состояний

Примечание: это учебный toy-модель, не заменяет большие трансформеры. Для лучшей семантики
рекомендуется использовать готовые модели embeddings (SentenceTransformer) и/или LLM.
"""

import os
import re
import random
import json
import time
import traceback
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional

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
# ПУТИ
# ======================
MODEL_PATH = "agi_brain_improved.pth"
VOCAB_PATH = "agi_vocab_improved.json"
EXPERIENCE_PATH = "agi_experience_improved.json"

# ======================
# УТИЛИТЫ
# ======================

def clean_qwen_response(text: str) -> str:
    if not isinstance(text, str):
        return "Хорошо."
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'[>#\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip(' \t\n\r\"\'')
    words = text.split()
    if len(words) > 80:
        text = ' '.join(words[:80])
        if not text.endswith(('.', '!', '?')):
            text += '.'
    return text or "Хорошо."


def tokenize_preserve(text: str) -> List[str]:
    # Сохраняем буквы, цифры, дефисы; приводим к нижнему регистру
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^0-9a-zа-яё\-\s]", ' ', text)
    tokens = [t for t in text.split() if t]
    return tokens

# ======================
# ВОКАБУЛЯРИЙ
# ======================
class VocabManager:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_id = 4

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.word2idx[word] = self.next_id
            self.idx2word[self.next_id] = word
            self.next_id += 1
        return self.word2idx[word]

    def add_words(self, words: List[str]):
        for w in words:
            self.add_word(w)

    def tokenize(self, text: str, max_len: int = 128) -> List[int]:
        words = tokenize_preserve(text)
        ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        # паддим/усекаем
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [0] * (max_len - len(ids))

    def detokenize(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == 0 or i == self.word2idx.get('<PAD>'):
                continue
            w = self.idx2word.get(i, '<UNK>')
            if w in ['<BOS>', '<EOS>']:
                continue
            words.append(w)
        return ' '.join(words).strip()

    def save(self, path: str):
        data = {
            'word2idx': self.word2idx,
            'next_id': self.next_id
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # keys might arrive as strings - convert where necessary
        self.word2idx = {k: int(v) if isinstance(v, (int, str)) and str(v).isdigit() else v for k, v in data.get('word2idx', {}).items()}
        # rebuild idx2word
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.next_id = data.get('next_id', max(self.idx2word.keys()) + 1 if self.idx2word else 4)

# ======================
# ПАМЯТЬ ОПЫТА
# ======================
class ExperienceBuffer:
    def __init__(self, maxlen=1000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, user: str, target: str):
        self.buffer.append((user, target))

    def sample(self, n: int) -> List[Tuple[str, str]]:
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n) if n > 0 else []

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(list(self.buffer), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.buffer = deque(data, maxlen=self.buffer.maxlen)

# ======================
# DIALOG CONTEXT
# ======================
class DialogContext:
    """Хранит N последних пар (user, assistant) и предоставляет текст-вход с контекстом."""
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)

    def add_turn(self, user: str, assistant: str):
        self.turns.append((user, assistant))

    def build_prompt(self, current_user: str) -> str:
        parts = []
        for u, a in self.turns:
            parts.append(f"User: {u}\nAssistant: {a}")
        parts.append(f"User: {current_user}\nAssistant:")
        return '\n'.join(parts)

# ======================
# COGNITIVE NETWORK (toy architecture)
# ======================
class CognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_size: int = 512, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden_size, batch_first=True, num_layers=1)
        self.decoder = nn.LSTM(emb_dim, hidden_size, batch_first=True, num_layers=1)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.2)

        self.to(self.device)

    def encode(self, x: torch.Tensor):
        emb = self.dropout(self.embedding(x))
        _, (h, c) = self.encoder(emb)
        return h, c

    def decode(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        emb = self.dropout(self.embedding(x))
        output, _ = self.decoder(emb, (h, c))
        output = self.dropout(output)
        return self.output_proj(output)

    def _prevent_repeats(self, logits: torch.Tensor, recent_tokens: List[int], ban_size: int = 3):
        # баним повторение токенов, которые уже были в последних N generated
        for t in recent_tokens[-ban_size:]:
            if 0 <= t < logits.shape[-1]:
                logits[0, t] = -1e9
        return logits

    def generate(self, input_ids: torch.Tensor, max_len: int = 40, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95) -> List[int]:
        self.eval()
        with torch.no_grad():
            h, c = self.encode(input_ids)
            batch_size = input_ids.size(0)
            current = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.long)  # <BOS>
            generated = []
            recent = []

            for step in range(max_len):
                emb = self.embedding(current)
                output, (h, c) = self.decoder(emb, (h, c))
                logits = self.output_proj(output.squeeze(1))

                # отсечение повторов
                logits = self._prevent_repeats(logits, recent, ban_size=4)

                # temperature + top-k/top-p
                logits = logits / max(temperature, 1e-6)
                # top-k
                if top_k is not None and top_k > 0:
                    values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_values = values[:, -1].unsqueeze(1)
                    logits = torch.where(logits < min_values, torch.full_like(logits, -1e9), logits)
                # top-p (nucleus)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumsum_probs = torch.cumsum(probs, dim=-1)
                    # mask tokens beyond top_p
                    mask = cumsum_probs > top_p
                    # keep first token above threshold
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits[mask] = -1e9
                    # unsort
                    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token.item())

                if token_id == 2:  # <EOS>
                    break
                generated.append(token_id)
                recent.append(token_id)
                current = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token

            return generated

# ======================
# SEMANTIC EVALUATOR
# ======================
class SemanticEvaluator:
    def __init__(self):
        self.model = None
        if _HAS_ST_MODEL:
            try:
                # lazy load - может занять время при первом вызове
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.model = None

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a, b))

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if self.model is not None:
            try:
                emb = self.model.encode([a, b], normalize_embeddings=True)
                return float(np.dot(emb[0], emb[1]))
            except Exception:
                pass
        # fallback: jaccard on token sets
        a_set = set(tokenize_preserve(a))
        b_set = set(tokenize_preserve(b))
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

# ======================
# TEACHER
# ======================
class Teacher:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions", retry: int = 2):
        self.api_url = api_url
        self.evaluator = SemanticEvaluator()
        self.experience = ExperienceBuffer()
        self.context = DialogContext(max_turns=6)
        self.optimizer = None
        self.retry = retry

    def bind_model(self, model: CognitiveNetwork, lr: float = 1e-3):
        # создаём оптимизатор один раз и держим его
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.model = model

    def ask_qwen(self, prompt: str, timeout: int = 10) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.6
        }
        headers = {'Content-Type': 'application/json'}
        for attempt in range(1, self.retry + 2):
            try:
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    # robust path to content
                    content = None
                    try:
                        content = data['choices'][0]['message']['content']
                    except Exception:
                        # older or different schema
                        try:
                            content = data['choices'][0]['text']
                        except Exception:
                            content = None
                    return clean_qwen_response(content) if content else "Хорошо."
                else:
                    time.sleep(0.5)
            except Exception as e:
                # логируем, но продолжаем
                print(f"⚠️ Qwen attempt {attempt} error: {e}")
                time.sleep(0.5)
        return "Хорошо."

    def train_step(self, model: CognitiveNetwork, vocab: VocabManager, user_input: str, qwen_answer: str, lr: float = 1e-3):
        # обновляем словарь
        vocab.add_words(tokenize_preserve(user_input))
        vocab.add_words(tokenize_preserve(qwen_answer))

        # подготовка батча (1 пример)
        inp_ids = vocab.tokenize(user_input)
        tgt_ids = vocab.tokenize(qwen_answer)

        # уберём пады в конце при вычислении loss
        enc = torch.tensor([inp_ids], device=model.device, dtype=torch.long)
        dec_in = torch.tensor([[vocab.word2idx.get('<BOS>')] + [i for i in tgt_ids if i != 0]], device=model.device, dtype=torch.long)
        dec_tgt = torch.tensor([ [i for i in tgt_ids if i != 0] + [vocab.word2idx.get('<EOS>')] ], device=model.device, dtype=torch.long)

        # forward
        model.train()
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.optimizer.zero_grad()
        h, c = model.encode(enc)
        logits = model.decode(dec_in, h, c)  # (1, seq_len, vocab)
        logits = logits.view(-1, model.output_proj.out_features)
        targets = dec_tgt.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=vocab.word2idx.get('<PAD>', 0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_until_understands(self, model: CognitiveNetwork, vocab: VocabManager, user_input: str, max_attempts: int = 6, sim_threshold: float = 0.82):
        # Генерируем контекстный prompt для Qwen, чтобы ответ был в контексте
        prompt = self.context.build_prompt(user_input)
        qwen_answer = self.ask_qwen(prompt)
        print(f"👤: {user_input}")
        print(f"🤖 Qwen: {qwen_answer}")

        best_sim = -1.0
        best_response = ""

        # bind optimizer if needed
        if not hasattr(self, 'model') or self.model is not model:
            self.bind_model(model)

        for attempt in range(1, max_attempts + 1):
            try:
                loss = self.train_step(model, vocab, user_input, qwen_answer, lr=1e-3)
            except Exception as e:
                print(f"⚠️ Train step failed: {e}")
                loss = float('nan')

            # generation: используем контекст + user_input
            model.eval()
            with torch.no_grad():
                inp_tensor = torch.tensor([vocab.tokenize(self.context.build_prompt(user_input))], device=model.device)
                gen_ids = model.generate(inp_tensor, max_len=60, temperature=0.6, top_k=40, top_p=0.92)
                brain_answer = vocab.detokenize(gen_ids)

            sim = self.evaluator.similarity(qwen_answer, brain_answer)
            loss_str = f"{loss:.4f}" if not (loss != loss) else "nan"
            print(f"  🔁 Попытка {attempt}: loss={loss_str}, сходство = {sim:.3f}")

            print(f"     💬 Модель: {brain_answer}")

            if sim > best_sim:
                best_sim = sim
                best_response = brain_answer

            if sim >= sim_threshold:
                # сохраним в опыт и контекст
                self.experience.add(user_input, qwen_answer)
                self.context.add_turn(user_input, brain_answer)
                print("✅ Поняла! Опыт сохранён.")
                return best_response

        # Дообучение на прошлом опыте (короткий цикл)
        if len(self.experience.buffer) > 0:
            print("🧠 Дообучение на прошлом опыте...")
            for u, a in self.experience.sample(4):
                try:
                    self.train_step(model, vocab, u, a, lr=5e-4)
                except Exception as e:
                    print(f"⚠️ Доп. обучение не удалось: {e}")

        # Добавим последний user->best_response в контекст
        self.context.add_turn(user_input, best_response)
        print(f"⚠️ Макс. попыток. Лучшее сходство: {best_sim:.3f}")
        return best_response

    def save_experience(self, path: str):
        self.experience.save(path)

# ======================
# MAIN
# ======================

def main():
    print("🧠 AGI (improved): обучение с контекстом и семантикой")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = VocabManager()
    model = CognitiveNetwork(vocab_size=8000, emb_dim=256, hidden_size=512, device=device)
    teacher = Teacher()
    teacher.bind_model(model, lr=1e-3)

    # загрузка
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("🧠 Модель загружена.")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель: {e}")
    if os.path.exists(VOCAB_PATH):
        try:
            vocab.load(VOCAB_PATH)
            print("📚 Словарь загружен.")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить словарь: {e}")
    teacher.experience.load(EXPERIENCE_PATH)

    try:
        while True:
            user_input = input("\n👤 Вы: ").strip()
            if user_input.lower() in ['выход', 'exit', 'quit']:
                break
            if not user_input:
                continue

            answer = teacher.train_until_understands(model, vocab, user_input)
            print(f"\n💡 Ответ модели: {answer}")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            torch.save(model.state_dict(), MODEL_PATH)
            vocab.save(VOCAB_PATH)
            teacher.save_experience(EXPERIENCE_PATH)
            print("\n💾 Состояние сохранено.")
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")


if __name__ == "__main__":
    main()
