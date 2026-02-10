# coding: utf-8
"""
AGI_CognitiveReasoning.py
Когнитивная система с мыслительными процессами, рефлексией и обучением
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
    LEARNING_PATH = SAVE_DIR / "learning_superintel.json"

    VOCAB_SIZE = 15000
    EMB_DIM = 512
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 4
    NUM_HEADS = 8
    DROPOUT = 0.2
    MAX_SEQ_LEN = 150

    LEARNING_RATE = 2e-4
    MAX_ATTEMPTS = 20
    CONTEXT_SIZE = 30

    # Параметры мышления
    CONFIDENCE_THRESHOLD = 0.75  # При какой уверенности ответ считается готовым
    REFLECTION_DEPTH = 5  # Глубина рефлексии
    QWEN_API = "http://localhost:1234/v1/chat/completions"


Config.SAVE_DIR.mkdir(exist_ok=True)


# ======================
# СИСТЕМА МЫШЛЕНИЯ И РЕФЛЕКСИИ
# ======================
class ThoughtProcess:
    """Представляет процесс мышления АИ"""

    def __init__(self):
        self.thoughts = []
        self.confidence = 0.0
        self.doubts = []
        self.reasoning_steps = []
        self.final_answer = ""
        self.learning_occurred = False

    def add_thought(self, thought: str, confidence: float = 0.5):
        """Добавить мысль в процесс"""
        self.thoughts.append({
            'text': thought,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        self.confidence = np.mean([t['confidence'] for t in self.thoughts])

    def add_doubt(self, doubt: str):
        """Выразить сомнение"""
        self.doubts.append({
            'text': doubt,
            'timestamp': datetime.now().isoformat()
        })

    def add_reasoning_step(self, step: str):
        """Добавить шаг рассуждения"""
        self.reasoning_steps.append(step)

    def is_confident(self, threshold: float = Config.CONFIDENCE_THRESHOLD) -> bool:
        """Уверен ли в ответе"""
        return self.confidence >= threshold

    def __str__(self):
        result = "🧠 ПРОЦЕСС МЫШЛЕНИЯ:\n"
        if self.thoughts:
            result += "💭 Мысли:\n"
            for t in self.thoughts[-3:]:  # Последние 3 мысли
                result += f"  • {t['text']} (уверенность: {t['confidence']:.1%})\n"
        if self.doubts:
            result += "❓ Сомнения:\n"
            for d in self.doubts[-2:]:
                result += f"  • {d['text']}\n"
        if self.reasoning_steps:
            result += "📍 Логика рассуждения:\n"
            for i, step in enumerate(self.reasoning_steps[-3:], 1):
                result += f"  {i}. {step}\n"
        result += f"📊 Общая уверенность: {self.confidence:.1%}\n"
        return result


# ======================
# МЕНЕДЖЕР ОБУЧЕНИЯ
# ======================
class LearningManager:
    """Управляет процессом обучения АИ"""

    def __init__(self):
        self.knowledge_base = {}
        self.learning_history = []
        self.skill_level = 0.1  # Уровень мастерства (0-1)
        self.asked_questions_count = 0
        self.correct_answers_count = 0
        self.load()

    def record_learning(self, topic: str, concept: str, teacher_answer: str,
                        ai_answer: str, similarity: float):
        """Записать факт обучения"""
        record = {
            'topic': topic,
            'concept': concept,
            'teacher_answer': teacher_answer,
            'ai_answer': ai_answer,
            'similarity': similarity,
            'timestamp': datetime.now().isoformat(),
            'skill_improvement': similarity
        }
        self.learning_history.append(record)

        # Обновляем базу знаний
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        self.knowledge_base[topic].append({
            'concept': concept,
            'answer': teacher_answer,
            'learned': True
        })

        # Обновляем уровень мастерства
        self.update_skill_level(similarity)
        self.save()

    def update_skill_level(self, similarity: float):
        """Обновить уровень мастерства"""
        new_level = (self.skill_level * len(self.learning_history) + similarity) / (len(self.learning_history) + 1)
        self.skill_level = min(1.0, new_level)
        self.correct_answers_count += int(similarity > 0.7)
        self.asked_questions_count += 1

    def get_known_topics(self) -> List[str]:
        """Получить известные темы"""
        return list(self.knowledge_base.keys())

    def get_topic_knowledge(self, topic: str) -> List[Dict]:
        """Получить знания по теме"""
        return self.knowledge_base.get(topic, [])

    def should_ask_teacher(self, confidence: float) -> bool:
        """Нужно ли спросить учителя"""
        return confidence < Config.CONFIDENCE_THRESHOLD

    def get_learning_progress(self) -> str:
        """Получить прогресс обучения"""
        total = len(self.learning_history)
        if total == 0:
            return "Обучение еще не началось"

        accuracy = self.correct_answers_count / self.asked_questions_count if self.asked_questions_count > 0 else 0
        return f"Уровень мастерства: {self.skill_level:.1%} | Точность: {accuracy:.1%} | Выученных концепций: {len(self.knowledge_base)}"

    def save(self):
        data = {
            'knowledge_base': self.knowledge_base,
            'learning_history': self.learning_history,
            'skill_level': self.skill_level,
            'asked_questions_count': self.asked_questions_count,
            'correct_answers_count': self.correct_answers_count
        }
        with open(Config.LEARNING_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.LEARNING_PATH.exists():
            try:
                with open(Config.LEARNING_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base = data.get('knowledge_base', {})
                    self.learning_history = data.get('learning_history', [])
                    self.skill_level = data.get('skill_level', 0.1)
                    self.asked_questions_count = data.get('asked_questions_count', 0)
                    self.correct_answers_count = data.get('correct_answers_count', 0)
            except:
                pass


# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================
def clean_text(text: str) -> str:
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
    """Определить тип входного вопроса"""
    s = user_input.lower().strip()
    patterns = {
        "SOCIAL": r'\b(привет|здравствуй|добрый день|как дела|пока|спасибо|благодарю)\b',
        "FACT": r'\b(что такое|кто такой|где находится|какая столица|формула|определение|расскажи о)\b',
        "REASON": r'\b(почему|зачем|отчего|причина|как это происходит|объясни механизм)\b',
        "PROCESS": r'\b(как сделать|как приготовить|инструкция|шаг|алгоритм|пошаговый)\b',
        "OPINION": r'\b(как ты думаешь|твоё мнение|лучше ли|нравится ли|согласен ли)\b',
        "CREATIVE": r'\b(представь|вообрази|сочини|опиши как|метафора|история|создай)\b',
        "ANALYSIS": r'\b(проанализируй|сравни|различие|сходство|анализ|тенденция)\b',
    }
    for qtype, pattern in patterns.items():
        if re.search(pattern, s):
            return qtype
    return "FACT"


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

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(self.emb_dim, Config.MAX_SEQ_LEN)
        self.embedding_dropout = nn.Dropout(Config.DROPOUT)

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, Config.NUM_HEADS, self.hidden_size, Config.DROPOUT)
            for _ in range(Config.NUM_LAYERS)
        ])

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, Config.NUM_HEADS, self.hidden_size, Config.DROPOUT)
            for _ in range(Config.NUM_LAYERS)
        ])

        self.cross_attentions = nn.ModuleList([
            MultiHeadAttention(self.emb_dim, Config.NUM_HEADS)
            for _ in range(Config.NUM_LAYERS)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, vocab_size)
        )

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

                top_k = min(50, probs.size(-1))
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token = top_k_indices[0, torch.multinomial(top_k_probs[0], 1)]

                token_id = next_token.item()
                if token_id == 2:
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
# ПРОДВИНУТЫЙ УЧИТЕЛЬ С РЕФЛЕКСИЕЙ
# ======================
class SupervisedTeacher:
    def __init__(self):
        self.api_url = Config.QWEN_API
        self.evaluator = SemanticEvaluator()
        self.learning_manager = LearningManager()
        self.step_count = 0

    def ask_teacher(self, prompt: str) -> str:
        """Спросить у старшей модели"""
        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.8
            }, timeout=25)
            if resp.status_code == 200:
                return clean_text(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"⚠️ Ошибка API: {e}")
        return "Я не знаю ответа."

    def generate_thoughts(self, user_input: str, input_type: str) -> ThoughtProcess:
        """Генерировать мысли перед ответом"""
        thought_process = ThoughtProcess()

        # Первоначальная мысль
        thought_process.add_thought(
            f"Мне задали вопрос типа '{input_type}': '{user_input}'",
            confidence=0.6
        )

        # Проверка памяти знаний
        known_topics = self.learning_manager.get_known_topics()
        skill_level = self.learning_manager.skill_level

        if skill_level < 0.3:
            thought_process.add_thought(
                "Я ещё в начале обучения, нужно быть осторожнее с ответами",
                confidence=0.8
            )
            thought_process.add_doubt("Я недостаточно опытен, возможно я ошибаюсь")
        elif skill_level > 0.7:
            thought_process.add_thought(
                "Я хорошо обучился, могу давать более уверенные ответы",
                confidence=0.85
            )

        # Извлечение ключевых слов
        key_words = [w for w in clean_for_similarity(user_input).split() if len(w) > 3]
        if key_words:
            thought_process.add_reasoning_step(
                f"Ключевые слова: {', '.join(key_words[:3])}"
            )

        return thought_process

    def train_step(self, model: SuperIntelligentBrain, vocab: AdvancedVocabManager,
                   user_input: str, input_type: str) -> str:
        """Основной шаг обучения с рефлексией"""

        print(f"\n👤 Вы: {user_input}")
        print(f"📋 Тип вопроса: {input_type}")

        # ========== ЭТАП 1: ГЕНЕРАЦИЯ МЫСЛЕЙ ==========
        print("\n🧠 Мышление...")
        thought_process = self.generate_thoughts(user_input, input_type)
        print(thought_process)

        # ========== ЭТАП 2: ПОПЫТКА ОТВЕТИТЬ САМОСТОЯТЕЛЬНО ==========
        print("\n🔄 Попытка ответить самостоятельно...")

        # Обновляем словарь
        vocab.add_words(clean_for_similarity(user_input).split())

        input_tokens = vocab.tokenize(user_input)
        if not input_tokens:
            input_tokens = [1, 3]

        encoder_input = torch.tensor([input_tokens[:Config.MAX_SEQ_LEN]], device=model.device)

        # Генерируем ответ
        gen_ids = model.generate(encoder_input, max_len=80, temperature=0.85)
        ai_answer = vocab.decode(gen_ids)

        print(f"💭 Мой ответ: {ai_answer}")

        # ========== ЭТАП 3: ПРОВЕРКА УВЕРЕННОСТИ ==========
        if thought_process.is_confident():
            print("\n✅ Я уверен в своем ответе!")
            vocab.add_words(clean_for_similarity(ai_answer).split())
            return ai_answer
        else:
            thought_process.add_doubt("Я не совсем уверен в правильности ответа")
            print("\n❓ Я не уверен. Спрашиваю у учителя...")

        # ========== ЭТАП 4: ОБУЧЕНИЕ У УЧИТЕЛЯ ==========
        print("\n👨‍🏫 Спрашиваю у старшей модели...")
        teacher_answer = self.ask_teacher(user_input)
        print(f"👨‍🏫 Учитель: {teacher_answer}")

        # Оцениваем сходство
        similarity = self.evaluator.similarity(ai_answer, teacher_answer)
        print(f"📊 Сходство ответов: {similarity:.1%}")

        # ========== ЭТАП 5: АНАЛИЗ И ОБУЧЕНИЕ ==========
        print("\n📚 Обучение...")

        target_tokens = vocab.tokenize(teacher_answer)
        if not target_tokens:
            target_tokens = [3]

        vocab.add_words(clean_for_similarity(teacher_answer).split())

        # Пересчитаем токены
        target_tokens = vocab.tokenize(teacher_answer)
        if not target_tokens:
            target_tokens = [3]

        target_ids = torch.tensor([[1] + target_tokens[:Config.MAX_SEQ_LEN - 1]], device=model.device)
        target_out = torch.tensor([target_tokens[:Config.MAX_SEQ_LEN - 1] + [2]], device=model.device)

        # Тренируем модель
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

        best_sim = similarity
        best_response = ai_answer
        improvements = []

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

            # Генерируем новый ответ
            gen_ids = model.generate(encoder_input, max_len=80, temperature=0.8)
            brain_answer = vocab.decode(gen_ids)

            new_sim = self.evaluator.similarity(teacher_answer, brain_answer)

            improvement = "📈" if new_sim > best_sim else "📉" if new_sim < best_sim else "➡️"
            print(f"  🔁 Итерация {attempt}: loss={loss.item():.4f}, сходство={new_sim:.1%} {improvement}")

            if new_sim > best_sim:
                best_sim = new_sim
                best_response = brain_answer
                improvements.append(new_sim)

            if new_sim >= Config.CONFIDENCE_THRESHOLD:
                print("✅ КОНЦЕПЦИЯ УСВОЕНА!\n")
                thought_process.learning_occurred = True
                break

            self.step_count += 1

        # ========== ЭТАП 6: СОХРАНЕНИЕ ОБУЧЕНИЯ ==========
        self.learning_manager.record_learning(
            topic=input_type,
            concept=user_input,
            teacher_answer=teacher_answer,
            ai_answer=best_response,
            similarity=best_sim
        )

        print(f"\n📚 {self.learning_manager.get_learning_progress()}")

        return best_response


# ======================
# ГЛАВНАЯ ПРОГРАММА
# ======================
def main():
    print("\n" + "=" * 70)
    print("🧠 КОГНИТИВНАЯ СИСТЕМА С РЕФЛЕКСИЕЙ И ОБУЧЕНИЕМ v4.0")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Устройство: {device}")

    # Инициализируем словарь
    vocab = AdvancedVocabManager()
    if vocab.load():
        print(f"✅ Словарь загружен ({vocab.size} слов)")
    else:
        print("🔨 Создаю новый словарь...")
        base_words = "привет спасибо да нет что как почему где когда кто который интересно понимаю узнал новое мышление рефлексия сомнение вопрос ответ мнение факт процесс анализ".split()
        vocab.add_words(base_words)
        print(f"✅ Создан словарь с {vocab.size} словами")

    # Инициализируем модель
    model = SuperIntelligentBrain(vocab_size=max(Config.VOCAB_SIZE, vocab.size), device=device)
    if model.load():
        print("✅ Модель загружена")
    else:
        print("🔨 Инициализирую новую модель...")
        print("✅ Модель создана")

    # Инициализируем учителя
    teacher = SupervisedTeacher()

    print(f"\n💡 КОМАНДЫ:")
    print(f"   'выход' - завершить программу")
    print(f"   'память' - показать историю взаимодействий")
    print(f"   'прогресс' - показать прогресс обучения")
    print(f"   'сохранить' - сохранить модель")
    print(f"   'очистить' - очистить память")

    interaction_count = 0

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("\n💾 Сохраняю модель...")
                model.save()
                vocab.save()
                teacher.learning_manager.save()
                print("✨ До встречи! Спасибо за обучение!")
                break

            if user_input.lower() in ['память', 'memory']:
                print(f"\n📚 История обучения:")
                history = teacher.learning_manager.learning_history
                if history:
                    for item in history[-5:]:
                        print(f"  📍 {item['concept'][:40]}...")
                        print(f"     Сходство: {item['similarity']:.1%}")
                else:
                    print("  (история пуста)")
                continue

            if user_input.lower() in ['прогресс', 'progress', 'stats']:
                print(f"\n📊 ПРОГРЕСС ОБУЧЕНИЯ:")
                print(f"  {teacher.learning_manager.get_learning_progress()}")
                known = teacher.learning_manager.get_known_topics()
                if known:
                    print(f"  Известные темы: {', '.join(known)}")
                continue

            if user_input.lower() in ['сохранить', 'save']:
                print("\n💾 Сохраняю...")
                model.save()
                vocab.save()
                teacher.learning_manager.save()
                print("✅ Сохранено!")
                continue

            if user_input.lower() in ['очистить', 'clear']:
                if input("Вы уверены? (да/нет): ").lower() == 'да':
                    Config.LEARNING_PATH.unlink(missing_ok=True)
                    teacher.learning_manager = LearningManager()
                    print("✅ Память очищена!")
                continue

            if not user_input:
                continue

            input_type = detect_input_type(user_input)
            final_answer = teacher.train_step(model, vocab, user_input, input_type)

            print(f"\n💡 МОЙ ОТВЕТ: {final_answer}\n")
            print("=" * 70)

            interaction_count += 1
            if interaction_count % 3 == 0:
                print(f"💾 Автосохранение...")
                model.save()
                vocab.save()
                teacher.learning_manager.save()

        except KeyboardInterrupt:
            print("\n✨ Прерывание. До встречи!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()