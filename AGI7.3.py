# coding: utf-8
"""
AGI_SuperIntelligent.py
Исключительно мощная когнитивная система с трансформерами и продвинутым обучением
"""
import os
import re
import json
import pickle
import random
import traceback
import math
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
    MODEL_PATH = SAVE_DIR / "model_superintel.pt"
    VOCAB_PATH = SAVE_DIR / "vocab_superintel.pkl"
    MEMORY_PATH = SAVE_DIR / "memory_superintel.json"

    VOCAB_SIZE = 10000
    EMB_DIM = 512
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 4
    NUM_HEADS = 8
    DROPOUT = 0.2
    MAX_SEQ_LEN = 150

    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 1000
    MAX_ATTEMPTS = 15

    QWEN_API = "http://localhost:1234/v1/chat/completions"
    CONTEXT_SIZE = 20


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
    text = re.sub(r'\r\n|\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[\*\.\!\?\:\-\–—\s]+', '', text)
    text = re.sub(r'[\*\.\!\?\:\-\–—\s]+$', '', text)
    words = text.split()
    if len(words) > 150:
        text = ' '.join(words[:150])
        if not text.endswith(('.', '!', '?')):
            text += '.'
    return text or "Хорошо."


def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).lower().strip()


def detect_input_type(user_input: str) -> str:
    s = user_input.lower().strip()
    patterns = {
        "SOC": r'\b(привет|здравствуй|добрый день|как дела|пока|спасибо|благодарю)\b',
        "FCT": r'\b(что такое|кто такой|где находится|какая столица|формула|определение|расскажи о)\b',
        "CAU": r'\b(почему|зачем|отчего|причина|как это происходит|объясни механизм)\b',
        "PRC": r'\b(как сделать|как приготовить|инструкция|шаг|алгоритм|пошаговый|технология)\b',
        "OPN": r'\b(как ты думаешь|твоё мнение|лучше ли|нравится ли|согласен ли|философия)\b',
        "CRT": r'\b(представь|вообрази|сочини|опиши как|метафора|история|создай|креатив)\b',
        "ANA": r'\b(проанализируй|сравни|различие|сходство|анализ|тенденция|закономерность)\b',
    }
    for itype, pattern in patterns.items():
        if re.search(pattern, s):
            return itype
    return "FCT"


# ======================
# РАСШИРЕННЫЙ VOCABULARY
# ======================
class AdvancedVocabManager:
    def __init__(self):
        self.word2idx = {
            '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3,
            '<START>': 4, '<FACT>': 5, '<REASON>': 6, '<PROC>': 7,
            '<EMOTION>': 8, '<CONCEPT>': 9, '<ENTITY>': 10,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        self.word_embeddings = {}
        self.next_id = 11
        self.semantic_tags = defaultdict(list)

    def add_word(self, word: str, semantic_tag: Optional[str] = None) -> int:
        word_lower = word.lower()
        if word_lower not in self.word2idx:
            if self.next_id < Config.VOCAB_SIZE:
                self.word2idx[word_lower] = self.next_id
                self.idx2word[self.next_id] = word_lower
                self.next_id += 1

        self.word_freq[word_lower] += 1
        if semantic_tag:
            self.semantic_tags[word_lower].append(semantic_tag)

        return self.word2idx.get(word_lower, self.word2idx['<UNK>'])

    def add_words(self, words: List[str]):
        for w in words:
            if w.strip():
                self.add_word(w)

    def tokenize(self, text: str) -> List[int]:
        words = clean_for_similarity(text).split()
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.idx2word.get(i, '<UNK>') for i in ids if i not in [0, 1, 2]]
        text = ' '.join(tokens)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    @property
    def size(self):
        return len(self.word2idx)

    def save(self):
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'next_id': self.next_id,
            'semantic_tags': dict(self.semantic_tags)
        }
        with open(Config.VOCAB_PATH, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        if Config.VOCAB_PATH.exists():
            with open(Config.VOCAB_PATH, 'rb') as f:
                data = pickle.load(f)
                self.word2idx = data['word2idx']
                self.idx2word = data['idx2word']
                self.word_freq = Counter(data['word_freq'])
                self.next_id = data['next_id']
                self.semantic_tags = defaultdict(list, data.get('semantic_tags', {}))
            return True
        return False


# ======================
# ПОЗИЦИОННОЕ КОДИРОВАНИЕ
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 5000):
        super().__init__()
        self.emb_dim = emb_dim

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ======================
# MULTI-HEAD ATTENTION
# ======================
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.fc_out = nn.Linear(emb_dim, emb_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.emb_dim)
        output = self.fc_out(context)

        return output, weights


# ======================
# ТРАНСФОРМЕР БЛОК
# ======================
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ======================
# СУПЕР ИНТЕЛЛЕКТУАЛЬНАЯ СЕТЬ
# ======================
class SuperIntelligentBrain(nn.Module):
    def __init__(self, vocab_size: int, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.vocab_size = vocab_size
        self.emb_dim = Config.EMB_DIM
        self.hidden_size = Config.HIDDEN_SIZE

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(self.emb_dim, Config.MAX_SEQ_LEN)
        self.embedding_dropout = nn.Dropout(Config.DROPOUT)

        # Encoder Transformer Stack
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, Config.NUM_HEADS, self.hidden_size, Config.DROPOUT)
            for _ in range(Config.NUM_LAYERS)
        ])

        # Decoder Transformer Stack
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, Config.NUM_HEADS, self.hidden_size, Config.DROPOUT)
            for _ in range(Config.NUM_LAYERS)
        ])

        # Cross Attention
        self.cross_attentions = nn.ModuleList([
            MultiHeadAttention(self.emb_dim, Config.NUM_HEADS)
            for _ in range(Config.NUM_LAYERS)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, vocab_size)
        )

        # Memory banks
        self.memory_bank = None
        self.concept_bank = defaultdict(list)

        self.to(self.device)

    def encode(self, input_ids: torch.Tensor):
        emb = self.embedding(input_ids)
        emb = self.pos_encoding(emb)
        emb = self.embedding_dropout(emb)

        for block in self.encoder_blocks:
            emb = block(emb)

        self.memory_bank = emb
        return emb

    def decode_with_attention(self, target_ids: torch.Tensor, encoder_output: torch.Tensor):
        emb = self.embedding(target_ids)
        emb = self.pos_encoding(emb)
        emb = self.embedding_dropout(emb)

        for i, block in enumerate(self.decoder_blocks):
            emb = block(emb)

            # Cross-attention к encoder output
            cross_out, _ = self.cross_attentions[i](emb, encoder_output, encoder_output)
            emb = emb + cross_out

        return emb

    def generate(self, input_ids: torch.Tensor, max_len: int = 80, temperature: float = 0.9) -> List[int]:
        was_training = self.training
        self.eval()

        with torch.no_grad():
            encoder_output = self.encode(input_ids)
            batch_size = input_ids.size(0)
            current_tokens = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.long)
            generated = []

            for step in range(max_len):
                decoder_output = self.decode_with_attention(current_tokens, encoder_output)
                logits = self.output_proj(decoder_output[:, -1, :])

                probs = F.softmax(logits / temperature, dim=-1)

                # Top-k sampling для разнообразия
                top_k = min(50, probs.size(-1))
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token = top_k_indices[0, torch.multinomial(top_k_probs[0], 1)]

                token_id = next_token.item()
                if token_id == 2:  # EOS
                    break

                generated.append(token_id)
                current_tokens = torch.cat([current_tokens, next_token.view(batch_size, 1)], dim=1)

            if was_training:
                self.train()

            return generated

    def save(self):
        torch.save({
            'model_state': self.state_dict(),
            'concept_bank': dict(self.concept_bank),
        }, Config.MODEL_PATH)

    def load(self):
        if Config.MODEL_PATH.exists():
            checkpoint = torch.load(Config.MODEL_PATH, map_location=self.device)
            self.load_state_dict(checkpoint['model_state'])
            self.concept_bank = defaultdict(list, checkpoint.get('concept_bank', {}))
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
        return len(a_clean & b_clean) / len(a_clean | b_clean)


# ======================
# ДОЛГОСРОЧНАЯ ПАМЯТЬ С АССОЦИАЦИЯМИ
# ======================
class AdvancedMemory:
    def __init__(self):
        self.memory = deque(maxlen=Config.CONTEXT_SIZE)
        self.associations = defaultdict(set)
        self.concepts = defaultdict(int)
        self.load()

    def add_interaction(self, user_input: str, ai_response: str, similarity: float, input_type: str):
        self.memory.append({
            'user': user_input,
            'ai': ai_response,
            'similarity': similarity,
            'type': input_type,
            'timestamp': datetime.now().isoformat()
        })

        # Извлекаем концепции
        for word in clean_for_similarity(user_input).split():
            self.concepts[word] += 1
        for word in clean_for_similarity(ai_response).split():
            self.concepts[word] += 1

        self.save()

    def get_semantic_context(self, query: str, evaluator: SemanticEvaluator, top_k: int = 5) -> List[str]:
        contexts = []
        for item in self.memory:
            sim = evaluator.similarity(query, item['user'])
            if sim > 0.25:
                contexts.append((item['ai'], sim))

        contexts.sort(key=lambda x: x[1], reverse=True)
        return [ctx[0] for ctx in contexts[:top_k]]

    def get_top_concepts(self, n: int = 10) -> List[str]:
        return sorted(self.concepts.items(), key=lambda x: x[1], reverse=True)[:n]

    def save(self):
        data = {
            'memory': list(self.memory),
            'associations': {k: list(v) for k, v in self.associations.items()},
            'concepts': dict(self.concepts)
        }
        with open(Config.MEMORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.MEMORY_PATH.exists():
            try:
                with open(Config.MEMORY_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory = deque(data.get('memory', []), maxlen=Config.CONTEXT_SIZE)
                    self.concepts = Counter(data.get('concepts', {}))
            except:
                pass


# ======================
# ПРОДВИНУТЫЙ УЧИТЕЛЬ
# ======================
class SupervisedTeacher:
    def __init__(self):
        self.api_url = Config.QWEN_API
        self.evaluator = SemanticEvaluator()
        self.memory = AdvancedMemory()
        self.step_count = 0

    def ask_qwen(self, prompt: str, context: Optional[str] = None) -> str:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 250,
                "temperature": 0.8
            }, timeout=25)
            if resp.status_code == 200:
                return clean_qwen_response(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"⚠️ API Ошибка: {e}")
        return "Понял вопрос."

    def train_step(self, model: SuperIntelligentBrain, vocab: AdvancedVocabManager,
                   user_input: str, input_type: str) -> str:
        context = self.get_context(user_input)
        qwen_answer = self.ask_qwen(user_input, context)

        print(f"👤: {user_input}")
        print(f"🤖 Учитель: {qwen_answer}")

        # Обновляем словарь
        vocab.add_words(clean_for_similarity(user_input).split())
        vocab.add_words(clean_for_similarity(qwen_answer).split())

        # Токенизация
        input_tokens = vocab.tokenize(user_input)
        target_tokens = vocab.tokenize(qwen_answer)

        if not input_tokens:
            input_tokens = [1, 3]
        if not target_tokens:
            target_tokens = [3]

        encoder_input = torch.tensor([input_tokens[:Config.MAX_SEQ_LEN]], device=model.device)
        target_ids = torch.tensor([[1] + target_tokens[:Config.MAX_SEQ_LEN - 1]], device=model.device)
        target_out = torch.tensor([target_tokens[:Config.MAX_SEQ_LEN - 1] + [2]], device=model.device)

        # Обучение с расписанием
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

        best_sim = -1.0
        best_response = ""

        for attempt in range(1, Config.MAX_ATTEMPTS + 1):
            model.train()
            optimizer.zero_grad()

            encoder_output = model.encode(encoder_input)
            decoder_output = model.decode_with_attention(target_ids, encoder_output)
            logits = model.output_proj(decoder_output)

            loss = F.cross_entropy(logits.view(-1, model.vocab_size), target_out.view(-1), ignore_index=0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            gen_ids = model.generate(encoder_input, max_len=80)
            brain_answer = vocab.decode(gen_ids)

            sim = self.evaluator.similarity(qwen_answer, brain_answer)
            print(f"  🔁 Итерация {attempt}: loss={loss.item():.4f}, сходство={sim:.3f}")

            if sim > best_sim:
                best_sim = sim
                best_response = brain_answer

            if sim >= 0.75:
                print("✅ Концепция усвоена!")
                break

            self.step_count += 1

        self.memory.add_interaction(user_input, best_response, best_sim, input_type)
        return best_response

    def get_context(self, user_input: str) -> str:
        similar = self.memory.get_semantic_context(user_input, self.evaluator, top_k=3)
        if similar:
            context = "📚 Контекст:\n" + "\n".join([f"- {s[:60]}..." for s in similar])
            return context
        return ""


# ======================
# ГЛАВНАЯ ПРОГРАММА
# ======================
def main():
    print("🧠 СУПЕР ИНТЕЛЛЕКТУАЛЬНАЯ КОГНИТИВНАЯ СИСТЕМА v3.0")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Устройство: {device}")

    vocab = AdvancedVocabManager()
    if vocab.load():
        print(f"✅ Словарь загружен ({vocab.size} слов)")
    else:
        print("🔨 Создаю новый словарь...")
        base_words = "привет спасибо да нет что как почему где когда кто который интересно понимаю узнал новое".split()
        vocab.add_words(base_words)

    model = SuperIntelligentBrain(vocab_size=max(Config.VOCAB_SIZE, vocab.size), device=device)
    if model.load():
        print("✅ Модель загружена")
    else:
        print("🔨 Инициализирую новую модель...")

    teacher = SupervisedTeacher()
    print(f"\n💡 Команды: 'выход' | 'память' | 'концепции' | 'сохранить'\n")

    interaction_count = 0

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("💾 Сохраняю...")
                model.save()
                vocab.save()
                print("✨ До встречи!")
                break

            if user_input.lower() in ['память', 'memory']:
                print(f"\n📚 Памяти: {len(teacher.memory.memory)} взаимодействий")
                for item in list(teacher.memory.memory)[-3:]:
                    print(f"  Q: {item['user'][:50]}... ({item['similarity']:.2f})")
                continue

            if user_input.lower() in ['концепции', 'concepts']:
                top = teacher.memory.get_top_concepts(10)
                print("\n🎯 Ключевые концепции:")
                for word, count in top:
                    print(f"  {word}: {count}x")
                continue

            if user_input.lower() in ['сохранить', 'save']:
                model.save()
                vocab.save()
                print("✅ Сохранено!")
                continue

            if not user_input:
                continue

            input_type = detect_input_type(user_input)
            final_answer = teacher.train_step(model, vocab, user_input, input_type)
            print(f"\n💡 Ответ: {final_answer}\n")

            interaction_count += 1
            if interaction_count % 3 == 0:
                print(f"💾 Автосохранение (#{interaction_count})...")
                model.save()
                vocab.save()

        except KeyboardInterrupt:
            print("\n✨ Пока!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()