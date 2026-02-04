# coding: utf-8
"""
AGI_final.py
Финальная версия: модель учится у Qwen, но отвечает своими словами,
сохраняет память между сессиями и избегает повторов.
"""
import os
import re
import random
import json
import traceback
from collections import deque
from typing import Dict, List, Tuple
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
# ПУТИ ДЛЯ СОХРАНЕНИЯ
# ======================
MODEL_PATH = "agi_brain.pth"
VOCAB_PATH = "agi_vocab.json"
EXPERIENCE_PATH = "agi_experience.json"

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================
def clean_qwen_response(text: str) -> str:
    if not isinstance(text, str):
        return "Хорошо."
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'[>#\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[^\wа-яА-Я]+', '', text)
    text = re.sub(r'[^\wа-яА-Я]+$', '', text)
    words = text.split()
    if len(words) > 50:
        text = ' '.join(words[:50])
        if not text.endswith(('.', '!', '?')):
            text += '.'
    return text or "Хорошо."

def tokenize_preserve(text: str) -> List[str]:
    # Сохраняем слова, цифры, дефисы — удаляем только спецсимволы
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return [w.lower() for w in text.split() if w]

# ======================
# УПРАВЛЕНИЕ СЛОВАРЁМ
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

    def tokenize(self, text: str) -> List[int]:
        words = tokenize_preserve(text)
        return [self.word2idx.get(w, 3) for w in words]  # 3 = <UNK>

    def decode(self, ids: List[int]) -> str:
        tokens = [self.idx2word.get(i, '<UNK>') for i in ids]
        return ' '.join(tokens).replace('<BOS>', '').replace('<EOS>', '').strip()

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'next_id': self.next_id
            }, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.word2idx = {k: int(v) for k, v in data['word2idx'].items()}
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            self.next_id = data['next_id']

# ======================
# ПАМЯТЬ ОПЫТА
# ======================
class ExperienceBuffer:
    def __init__(self, maxlen=500):
        self.buffer = deque(maxlen=maxlen)

    def add(self, user: str, target: str):
        self.buffer.append((user, target))

    def sample(self, n: int) -> List[Tuple[str, str]]:
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n) if n > 0 else []

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(list(self.buffer), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.buffer = deque(data, maxlen=self.buffer.maxlen)

# ======================
# КОГНИТИВНАЯ СЕТЬ
# ======================
class CognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_size: int = 512, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden_size, batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(emb_dim, hidden_size, batch_first=True, dropout=0.2)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

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

    def generate(self, input_ids: torch.Tensor, max_len: int = 30) -> List[int]:
        self.eval()
        with torch.no_grad():
            h, c = self.encode(input_ids)
            batch_size = input_ids.size(0)
            current = torch.full((batch_size, 1), 1, device=self.device)  # <BOS>
            generated = []

            for _ in range(max_len):
                emb = self.embedding(current)
                output, (h, c) = self.decoder(emb, (h, c))
                logits = self.output_proj(output.squeeze(1))

                # Запрет повторов (n=2)
                if len(generated) >= 1:
                    last_token = generated[-1]
                    logits[0, last_token] = -1e9

                probs = F.softmax(logits / 0.9, dim=-1)
                next_token = torch.multinomial(probs, 1)
                token_id = next_token.item()
                if token_id == 2:  # <EOS>
                    break
                generated.append(token_id)
                current = next_token

            return generated

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
        if not a.strip() or not b.strip():
            return 0.0
        if self.model is not None:
            emb = self.model.encode([a, b], normalize_embeddings=True)
            return float(np.dot(emb[0], emb[1]))
        a_clean = set(tokenize_preserve(a))
        b_clean = set(tokenize_preserve(b))
        if not a_clean or not b_clean:
            return 0.0
        return len(a_clean & b_clean) / len(a_clean | b_clean)

# ======================
# УЧИТЕЛЬ С ПАМЯТЬЮ
# ======================
class Teacher:
    def __init__(self, api_url="http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        self.evaluator = SemanticEvaluator()
        self.experience = ExperienceBuffer()

    def ask_qwen(self, prompt: str) -> str:
        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7
            }, timeout=15)
            if resp.status_code == 200:
                return clean_qwen_response(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"⚠️ Qwen API error: {e}")
        return "Хорошо."

    def train_step(self, model, vocab, user_input, qwen_answer, lr=1e-3):
        vocab.add_words(tokenize_preserve(user_input))
        vocab.add_words(tokenize_preserve(qwen_answer))

        inp_ids = vocab.tokenize(user_input)
        tgt_ids = vocab.tokenize(qwen_answer)

        encoder_in = torch.tensor([inp_ids], device=model.device)
        decoder_in = torch.tensor([[1] + tgt_ids], device=model.device)      # <BOS> + ответ
        decoder_tgt = torch.tensor([tgt_ids + [2]], device=model.device)     # ответ + <EOS>

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        optimizer.zero_grad()

        h, c = model.encode(encoder_in)
        logits = model.decode(decoder_in, h, c)
        logits = logits.view(-1, model.vocab_size)
        targets = decoder_tgt.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def train_until_understands(self, model, vocab, user_input, max_attempts=6):
        qwen_answer = self.ask_qwen(user_input)
        print(f"👤: {user_input}")
        print(f"🤖 Qwen: {qwen_answer}")

        best_sim = -1.0
        best_response = ""

        for attempt in range(1, max_attempts + 1):
            loss = self.train_step(model, vocab, user_input, qwen_answer, lr=1e-3)

            # Генерация собственного ответа
            model.eval()
            with torch.no_grad():
                inp_tensor = torch.tensor([vocab.tokenize(user_input)], device=model.device)
                gen_ids = model.generate(inp_tensor, max_len=30)
                brain_answer = vocab.decode(gen_ids)

            sim = self.evaluator.similarity(qwen_answer, brain_answer)
            print(f"  🔁 Попытка {attempt}: loss={loss:.4f}, сходство = {sim:.3f}")
            print(f"     💬 Модель: {brain_answer}")

            if sim > best_sim:
                best_sim = sim
                best_response = brain_answer

            if sim >= 0.80:
                self.experience.add(user_input, qwen_answer)
                print("✅ Поняла! Опыт сохранён.")
                return best_response

        # Дообучение на опыте (1 эпоха)
        if self.experience.buffer:
            print("🧠 Дообучение на прошлом опыте...")
            for u, a in self.experience.sample(3):
                self.train_step(model, vocab, u, a, lr=5e-4)

        print(f"⚠️ Макс. попыток. Лучшее сходство: {best_sim:.3f}")
        return best_response

    def save_experience(self, path: str):
        self.experience.save(path)

# ======================
# MAIN
# ======================
def main():
    print("🧠 AGI: обучение с памятью и пониманием (финальная версия)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = VocabManager()
    model = CognitiveNetwork(vocab_size=5000, emb_dim=256, hidden_size=512, device=device)
    teacher = Teacher()

    # Загрузка состояния
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("🧠 Модель загружена.")
    if os.path.exists(VOCAB_PATH):
        vocab.load(VOCAB_PATH)
        print("📚 Словарь загружен.")
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
        # Сохранение
        torch.save(model.state_dict(), MODEL_PATH)
        vocab.save(VOCAB_PATH)
        teacher.save_experience(EXPERIENCE_PATH)
        print("\n💾 Состояние сохранено. До встречи!")

if __name__ == "__main__":
    main()