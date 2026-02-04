# coding: utf-8
"""
AGI_learning_until_understands.py
Модель обучается до тех пор, пока не начнёт отвечать как Qwen.
Теперь с корректным seq2seq обучением, управлением словарём и стабильной генерацией.
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

def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).lower().strip()

# ======================
# ТИПЫ МЫШЛЕНИЯ
# ======================
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
# УПРАВЛЕНИЕ СЛОВАРЁМ
# ======================
class VocabManager:
    def __init__(self):
        self.word2idx = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
        }
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
        words = clean_for_similarity(text).split()
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.idx2word.get(i, '<UNK>') for i in ids]
        return ' '.join(tokens).replace('<BOS>', '').replace('<EOS>', '').strip()

    @property
    def size(self):
        return len(self.word2idx)

# ======================
# КОГНИТИВНАЯ СЕТЬ (УПРОЩЁННАЯ, НО РАБОЧАЯ)
# ======================
class CognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_size: int = 256, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        self.decoder_rnn = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        self.to(self.device)

    def encode(self, input_ids: torch.Tensor):
        emb = self.embedding(input_ids)  # [B, T_in, E]
        _, (h, c) = self.encoder_rnn(emb)  # h, c: [1, B, H]
        return h, c

    def decode(self, target_ids: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        emb = self.embedding(target_ids)  # [B, T_out, E]
        output, _ = self.decoder_rnn(emb, (h, c))  # [B, T_out, H]
        logits = self.output_proj(output)  # [B, T_out, V]
        return logits

    def generate(self, input_ids: torch.Tensor, max_len: int = 30) -> List[int]:
        self.eval()
        with torch.no_grad():
            h, c = self.encode(input_ids)
            batch_size = input_ids.size(0)
            current_token = torch.full((batch_size, 1), 1, device=self.device)  # <BOS> = 1
            generated = []

            for _ in range(max_len):
                emb = self.embedding(current_token)  # [B, 1, E]
                output, (h, c) = self.decoder_rnn(emb, (h, c))  # [B, 1, H]
                logits = self.output_proj(output.squeeze(1))  # [B, V]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [B, 1]
                token_id = next_token.item()
                if token_id == 2:  # <EOS>
                    break
                generated.append(token_id)
                current_token = next_token

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
        if not a or not b:
            return 0.0
        if self.model is not None:
            emb = self.model.encode([a, b], normalize_embeddings=True)
            return float(np.dot(emb[0], emb[1]))
        a_clean = set(clean_for_similarity(a).split())
        b_clean = set(clean_for_similarity(b).split())
        if not a_clean or not b_clean:
            return 0.0
        return len(a_clean & b_clean) / len(a_clean | b_clean)

# ======================
# УЧИТЕЛЬ
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
        except Exception as e:
            print(f"⚠️ Ошибка Qwen API: {e}")
        return "Хорошо."

    def train_until_understands(
        self,
        model: CognitiveNetwork,
        vocab: VocabManager,
        user_input: str,
        max_attempts: int = 6,
        lr: float = 1e-3
    ) -> str:
        qwen_answer = self.ask_qwen(user_input)
        print(f"👤: {user_input}")
        print(f"🤖 Qwen: {qwen_answer}")

        # Обновляем словарь
        vocab.add_words(clean_for_similarity(user_input).split())
        vocab.add_words(clean_for_similarity(qwen_answer).split())

        # Токенизация
        input_tokens = vocab.tokenize(user_input)
        target_tokens = vocab.tokenize(qwen_answer)

        # Формируем последовательности
        encoder_input = torch.tensor([input_tokens], device=model.device)  # [1, T_in]
        decoder_input = torch.tensor([[1] + target_tokens], device=model.device)  # <BOS> + ответ
        decoder_target = torch.tensor([target_tokens + [2]], device=model.device)  # ответ + <EOS>

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        best_sim = -1.0
        best_response = ""

        for attempt in range(1, max_attempts + 1):
            model.train()
            optimizer.zero_grad()

            # Кодируем вход
            h, c = model.encode(encoder_input)

            # Декодируем с teacher forcing
            logits = model.decode(decoder_input, h, c)  # [1, T_out, V]
            logits = logits.view(-1, model.vocab_size)  # [T_out, V]
            targets = decoder_target.view(-1)           # [T_out]

            loss = F.cross_entropy(logits, targets, ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Генерация
            model.eval()
            with torch.no_grad():
                gen_ids = model.generate(encoder_input, max_len=30)
                brain_answer = vocab.decode(gen_ids)

            sim = self.evaluator.similarity(qwen_answer, brain_answer)
            print(f"  🔁 Попытка {attempt}: loss={loss.item():.4f}, сходство = {sim:.3f}")

            if sim > best_sim:
                best_sim = sim
                best_response = brain_answer

            if sim >= 0.85:
                print("✅ Модель поняла!")
                return best_response

        print(f"⚠️ Макс. попыток. Лучшее сходство: {best_sim:.3f}")
        return best_response

# ======================
# MAIN
# ======================
def main():
    print("🧠 Обучение до полного понимания (улучшенная версия)")
    vocab = VocabManager()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CognitiveNetwork(vocab_size=1000, emb_dim=128, hidden_size=256, device=device)
    teacher = Teacher()

    # Предзаполним словарь базовыми словами
    base_words = "привет что такое почему как помочь хорошо чем день".split()
    vocab.add_words(base_words)

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
            if user_input.lower() in ['выход', 'exit', 'quit']:
                break
            if not user_input:
                continue

            final_answer = teacher.train_until_understands(model, vocab, user_input)
            print(f"\n💡 Ответ модели: {final_answer}")

        except KeyboardInterrupt:
            print("\n👋 Пока!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()