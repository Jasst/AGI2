# coding: utf-8
"""
AGI_Advanced_Learning.py
Мощная когнитивная модель с сохранением знаний, контекстом и продвинутым обучением.
"""
import os
import re
import json
import pickle
import random
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST_MODEL = True
except:
    _HAS_ST_MODEL = False


# ======================
# КОНФИГУРАЦИЯ
# ======================
class Config:
    SAVE_DIR = Path("./cognitive_model_data")
    MODEL_PATH = SAVE_DIR / "model.pt"
    VOCAB_PATH = SAVE_DIR / "vocab.pkl"
    MEMORY_PATH = SAVE_DIR / "memory.json"
    KNOWLEDGE_PATH = SAVE_DIR / "knowledge.json"

    VOCAB_SIZE = 5000
    EMB_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    MAX_SEQ_LEN = 100

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 8
    EPOCHS = 5
    MAX_ATTEMPTS = 10

    QWEN_API = "http://localhost:1234/v1/chat/completions"
    CONTEXT_SIZE = 10


# ======================
# ИНИЦИАЛИЗАЦИЯ
# ======================
Config.SAVE_DIR.mkdir(exist_ok=True)


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
    if len(words) > 100:
        text = ' '.join(words[:100])
        if not text.endswith(('.', '!', '?')):
            text += '.'
    return text or "Хорошо."


def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).lower().strip()


def detect_input_type(user_input: str) -> str:
    s = user_input.lower().strip()
    patterns = {
        "SOC": r'\b(привет|здравствуй|добрый день|как дела|пока|спасибо)\b',
        "FCT": r'\b(что такое|кто такой|где находится|какая столица|формула|определение)\b',
        "CAU": r'\b(почему|зачем|отчего|причина|как это происходит)\b',
        "PRC": r'\b(как сделать|как приготовить|инструкция|шаг|алгоритм|пошаговый)\b',
        "OPN": r'\b(как ты думаешь|твоё мнение|лучше ли|нравится ли|согласен ли)\b',
        "CRT": r'\b(представь|вообрази|сочини|опиши как|метафора|история)\b',
        "MET": r'\b(почему ты|как ты понял|что ты имел в виду|объясни)\b',
        "ANA": r'\b(проанализируй|сравни|различие|сходство|анализ)\b',
    }
    for itype, pattern in patterns.items():
        if re.search(pattern, s):
            return itype
    return "FCT"


# ======================
# УПРАВЛЕНИЕ СЛОВАРЁМ (РАСШИРЕННОЕ)
# ======================
class VocabManager:
    def __init__(self):
        self.word2idx = {
            '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3,
            '<START>': 4, '<FACT>': 5, '<REASON>': 6, '<PROC>': 7,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        self.next_id = 8

    def add_word(self, word: str) -> int:
        word_lower = word.lower()
        if word_lower not in self.word2idx:
            if self.next_id < Config.VOCAB_SIZE:
                self.word2idx[word_lower] = self.next_id
                self.idx2word[self.next_id] = word_lower
                self.next_id += 1
        self.word_freq[word_lower] += 1
        return self.word2idx.get(word_lower, self.word2idx['<UNK>'])

    def add_words(self, words: List[str]):
        for w in words:
            if w.strip():
                self.add_word(w)

    def tokenize(self, text: str) -> List[int]:
        words = clean_for_similarity(text).split()
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.idx2word.get(i, '<UNK>') for i in ids]
        text = ' '.join(tokens)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    @property
    def size(self):
        return len(self.word2idx)

    def save(self):
        with open(Config.VOCAB_PATH, 'wb') as f:
            pickle.dump((self.word2idx, self.idx2word, self.word_freq, self.next_id), f)

    def load(self):
        if Config.VOCAB_PATH.exists():
            with open(Config.VOCAB_PATH, 'rb') as f:
                self.word2idx, self.idx2word, self.word_freq, self.next_id = pickle.load(f)
            return True
        return False


# ======================
# ДАТАСЕТ
# ======================
class ConversationDataset(Dataset):
    def __init__(self, conversations: List[Tuple[List[int], List[int]]], max_len=Config.MAX_SEQ_LEN):
        self.conversations = conversations
        self.max_len = max_len

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        input_ids, target_ids = self.conversations[idx]

        input_ids = input_ids[:self.max_len]
        target_ids = target_ids[:self.max_len]

        input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        target_ids = [1] + target_ids[:self.max_len - 1]
        target_ids_out = target_ids[1:] + [2]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'target_out': torch.tensor(target_ids_out, dtype=torch.long),
        }


# ======================
# МОЩНАЯ АРХИТЕКТУРА С ATTENTION
# ======================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, hidden, context):
        Q = self.query(hidden)
        K = self.key(context)
        V = self.value(context)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights


class AdvancedCognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.vocab_size = vocab_size
        self.emb_dim = Config.EMB_DIM
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(Config.DROPOUT)

        self.encoder = nn.LSTM(
            self.emb_dim, self.hidden_size,
            self.num_layers, batch_first=True,
            dropout=Config.DROPOUT if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Decoder получает combined hidden state размером hidden_size * 2
        self.decoder = nn.LSTM(
            self.emb_dim, self.hidden_size * 2,
            self.num_layers, batch_first=True,
            dropout=Config.DROPOUT if self.num_layers > 1 else 0
        )

        self.attention = AttentionLayer(self.hidden_size * 2)

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, vocab_size)
        )

        self.knowledge_base = {}
        self.to(self.device)

    def encode(self, input_ids: torch.Tensor):
        emb = self.embedding_dropout(self.embedding(input_ids))
        _, (h, c) = self.encoder(emb)
        # h и c shape: [num_layers * num_directions, batch, hidden_size]
        # Преобразуем в формат для decoder: [num_layers, batch, hidden_size * 2]
        batch_size = h.size(1)

        # Комбинируем forward и backward
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)  # [num_layers, batch, hidden_size * 2]

        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = torch.cat([c[:, 0], c[:, 1]], dim=-1)  # [num_layers, batch, hidden_size * 2]

        return h, c

    def decode_teacher_forcing(self, target_ids: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        emb = self.embedding_dropout(self.embedding(target_ids))
        output, (h, c) = self.decoder(emb, (h, c))
        attn_out, attn_weights = self.attention(output, output)
        combined = output + attn_out
        logits = self.output_proj(combined)
        return logits

    def generate(self, input_ids: torch.Tensor, max_len: int = 50, temperature: float = 0.8) -> List[int]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            h, c = self.encode(input_ids)
            batch_size = input_ids.size(0)
            current_token = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.long)
            generated = []

            for _ in range(max_len):
                emb = self.embedding(current_token)
                output, (h, c) = self.decoder(emb, (h, c))
                attn_out, _ = self.attention(output, output)
                combined = output + attn_out
                logits = self.output_proj(combined.squeeze(1))

                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                token_id = next_token.item()

                if token_id == 2:
                    break
                generated.append(token_id)
                current_token = next_token

            if was_training:
                self.train()
            return generated

    def save(self):
        torch.save({
            'model_state': self.state_dict(),
            'knowledge_base': self.knowledge_base
        }, Config.MODEL_PATH)

    def load(self):
        if Config.MODEL_PATH.exists():
            checkpoint = torch.load(Config.MODEL_PATH, map_location=self.device)
            self.load_state_dict(checkpoint['model_state'])
            self.knowledge_base = checkpoint.get('knowledge_base', {})
            return True
        return False


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
            try:
                emb = self.model.encode([a, b], normalize_embeddings=True)
                return float(np.dot(emb[0], emb[1]))
            except:
                pass

        a_clean = set(clean_for_similarity(a).split())
        b_clean = set(clean_for_similarity(b).split())
        if not a_clean or not b_clean:
            return 0.0
        intersection = len(a_clean & b_clean)
        union = len(a_clean | b_clean)
        return intersection / union if union > 0 else 0.0


# ======================
# ДОЛГОСРОЧНАЯ ПАМЯТЬ
# ======================
class LongTermMemory:
    def __init__(self):
        self.memory = deque(maxlen=Config.CONTEXT_SIZE)
        self.knowledge = {}
        self.load()

    def add_interaction(self, user_input: str, ai_response: str, similarity: float):
        self.memory.append({
            'user': user_input,
            'ai': ai_response,
            'similarity': similarity,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def add_knowledge(self, key: str, value: str):
        self.knowledge[key] = {
            'value': value,
            'added': datetime.now().isoformat(),
            'usage_count': 0
        }
        self.save()

    def get_similar_context(self, query: str, evaluator: SemanticEvaluator, top_k: int = 3) -> List[str]:
        contexts = []
        for item in self.memory:
            sim = evaluator.similarity(query, item['user'])
            if sim > 0.3:
                contexts.append((item['ai'], sim))

        contexts.sort(key=lambda x: x[1], reverse=True)
        return [ctx[0] for ctx in contexts[:top_k]]

    def save(self):
        data = {
            'memory': list(self.memory),
            'knowledge': self.knowledge
        }
        with open(Config.MEMORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.MEMORY_PATH.exists():
            try:
                with open(Config.MEMORY_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory = deque(data.get('memory', []), maxlen=Config.CONTEXT_SIZE)
                    self.knowledge = data.get('knowledge', {})
            except:
                pass


# ======================
# УЧИТЕЛЬ С КУРАТОРСТВОМ
# ======================
class AdvancedTeacher:
    def __init__(self):
        self.api_url = Config.QWEN_API
        self.evaluator = SemanticEvaluator()
        self.memory = LongTermMemory()

    def ask_qwen(self, prompt: str, context: Optional[str] = None) -> str:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 200,
                "temperature": 0.7
            }, timeout=20)
            if resp.status_code == 200:
                return clean_qwen_response(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"⚠️  Ошибка API: {e}")
        return "Хорошо."

    def train_model(
            self,
            model: AdvancedCognitiveNetwork,
            vocab: VocabManager,
            user_input: str,
            input_type: str
    ) -> str:
        # Получаем ответ от учителя
        context = self.get_context(user_input)
        qwen_answer = self.ask_qwen(user_input, context)

        print(f"👤: {user_input}")
        print(f"🤖 Qwen: {qwen_answer}")

        # Обновляем словарь
        vocab.add_words(clean_for_similarity(user_input).split())
        vocab.add_words(clean_for_similarity(qwen_answer).split())

        # Токенизируем
        input_tokens = vocab.tokenize(user_input)
        target_tokens = vocab.tokenize(qwen_answer)

        if not input_tokens:
            input_tokens = [1, 3]
        if not target_tokens:
            target_tokens = [3]

        # Подготавливаем данные
        encoder_input = torch.tensor([input_tokens], device=model.device)
        target_ids = torch.tensor([[1] + target_tokens], device=model.device)
        target_out = torch.tensor([target_tokens + [2]], device=model.device)

        # Обучение
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        best_sim = -1.0
        best_response = ""

        for attempt in range(1, Config.MAX_ATTEMPTS + 1):
            model.train()
            optimizer.zero_grad()

            h, c = model.encode(encoder_input)
            logits = model.decode_teacher_forcing(target_ids, h, c)

            loss = F.cross_entropy(logits.view(-1, model.vocab_size), target_out.view(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Генерируем ответ
            gen_ids = model.generate(encoder_input, max_len=50)
            brain_answer = vocab.decode(gen_ids)

            sim = self.evaluator.similarity(qwen_answer, brain_answer)
            print(f"  🔁 Попытка {attempt}: loss={loss.item():.4f}, сходство={sim:.3f}")

            if sim > best_sim:
                best_sim = sim
                best_response = brain_answer

            if sim >= 0.80:
                print("✅ Модель усвоила концепцию!")
                break

            if attempt % 3 == 0:
                scheduler.step()

        # Сохраняем в память
        self.memory.add_interaction(user_input, best_response, best_sim)

        return best_response

    def get_context(self, user_input: str) -> str:
        similar = self.memory.get_similar_context(user_input, self.evaluator)
        if similar:
            return "Контекст из прошлых разговоров:\n" + "\n".join(similar[:2])
        return ""


# ======================
# ОСНОВНАЯ ПРОГРАММА
# ======================
def main():
    print("🧠 Продвинутая когнитивная система обучения v2.0")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Устройство: {device}")

    # Загружаем или создаём компоненты
    vocab = VocabManager()
    if vocab.load():
        print("✅ Словарь загружен из памяти")
    else:
        print("🔨 Создаём новый словарь")
        base_words = "привет спасибо да нет что как почему где когда кто который который".split()
        vocab.add_words(base_words)

    model = AdvancedCognitiveNetwork(vocab_size=max(Config.VOCAB_SIZE, vocab.size), device=device)
    if model.load():
        print("✅ Модель загружена из памяти")
    else:
        print("🔨 Создаём новую модель")

    teacher = AdvancedTeacher()

    print("\n💡 Команды: 'выход' - выход, 'сохранить' - сохранить, 'память' - показать память\n")

    interaction_count = 0

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("💾 Сохраняю...")
                model.save()
                vocab.save()
                print("👋 До свидания!")
                break

            if user_input.lower() in ['сохранить', 'save']:
                model.save()
                vocab.save()
                print("✅ Сохранено!")
                continue

            if user_input.lower() in ['память', 'memory']:
                print(f"\n📚 В памяти {len(teacher.memory.memory)} взаимодействий")
                for i, item in enumerate(list(teacher.memory.memory)[-3:]):
                    print(f"  {i + 1}. Q: {item['user'][:50]}...")
                print()
                continue

            if not user_input:
                continue

            input_type = detect_input_type(user_input)
            final_answer = teacher.train_model(model, vocab, user_input, input_type)
            print(f"\n💡 Ответ модели: {final_answer}")

            interaction_count += 1
            if interaction_count % 5 == 0:
                print(f"\n💾 Автосохранение (взаимодействие #{interaction_count})...")
                model.save()
                vocab.save()

        except KeyboardInterrupt:
            print("\n👋 Пока!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()