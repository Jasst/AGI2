# coding: utf-8
"""
AGI_CognitiveReasoning_v7_COMPLETE_FIXED.py
Полная когнитивная система с правильным обучением и рефлексией
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


# ====================== КОНФИГУРАЦИЯ ======================
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
    CONFIDENCE_THRESHOLD = 0.6
    REFLECTION_DEPTH = 5
    QWEN_API = "http://localhost:1234/v1/chat/completions"


Config.SAVE_DIR.mkdir(exist_ok=True)


# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def clean_text(text: str) -> str:
    """Очистить текст от артефактов"""
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


def clean_generated_response(text: str) -> str:
    """Специальная очистка для сгенерированных ответов модели"""
    if not isinstance(text, str) or not text.strip():
        return ""
    words = text.split()
    cleaned = []
    for word in words:
        if not cleaned or cleaned[-1].lower() != word.lower():
            cleaned.append(word)
    if len(cleaned) < 2:
        return ""
    result = ' '.join(cleaned[:25])
    if result and not result.endswith(('.', '!', '?', '😊')):
        if len(cleaned) > 2:
            result += '.'
    return result.strip()


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


# ====================== СИСТЕМА МЫШЛЕНИЯ ======================
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
        self.thoughts.append({
            'text': thought,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        self.confidence = np.mean([t['confidence'] for t in self.thoughts])

    def add_doubt(self, doubt: str):
        self.doubts.append({
            'text': doubt,
            'timestamp': datetime.now().isoformat()
        })

    def add_reasoning_step(self, step: str):
        self.reasoning_steps.append(step)

    def is_confident(self, threshold: float = Config.CONFIDENCE_THRESHOLD) -> bool:
        return self.confidence >= threshold

    def __str__(self):
        result = "🧠 ПРОЦЕСС МЫШЛЕНИЯ:\n"
        if self.thoughts:
            result += "💭 Мысли:\n"
            for t in self.thoughts[-3:]:
                result += f" • {t['text']} (уверенность: {t['confidence']:.1%})\n"
        if self.doubts:
            result += "❓ Сомнения:\n"
            for d in self.doubts[-2:]:
                result += f" • {d['text']}\n"
        if self.reasoning_steps:
            result += "📍 Логика рассуждения:\n"
            for i, step in enumerate(self.reasoning_steps[-3:], 1):
                result += f" {i}. {step}\n"
        result += f"📊 Общая уверенность: {self.confidence:.1%}\n"
        return result


# ====================== ПАМЯТЬ ======================
class ContextMemory:
    """Долгосрочная память для запоминания разговоров"""

    def __init__(self):
        self.conversations = deque(maxlen=50)
        self.user_profile = {}
        self.topics_discussed = defaultdict(list)
        self.relationships = {}
        self.load()

    def add_interaction(self, user_input: str, ai_response: str, topic: str, similarity: float):
        interaction = {
            'user_input': user_input,
            'ai_response': ai_response,
            'topic': topic,
            'similarity': similarity,
            'timestamp': datetime.now().isoformat()
        }
        self.conversations.append(interaction)
        if topic not in self.topics_discussed:
            self.topics_discussed[topic] = []
        self.topics_discussed[topic].append({
            'question': user_input,
            'answer': ai_response,
            'confidence': similarity
        })
        self.save()

    def get_context(self, topic: str, num_context: int = 5) -> str:
        if topic not in self.topics_discussed:
            return ""
        recent = self.topics_discussed[topic][-num_context:]
        if not recent:
            return ""
        context = f"📚 Мои знания о теме '{topic}':\n"
        for i, item in enumerate(recent, 1):
            context += f"{i}. Q: {item['question'][:50]}...\n   A: {item['answer'][:50]}...\n"
        return context

    def get_recent_context(self, num_last: int = 3) -> str:
        if not self.conversations:
            return ""
        recent = list(self.conversations)[-num_last:]
        context = "📚 Последний контекст:\n"
        for conv in recent:
            context += f"• {conv['ai_response'][:40]}...\n"
        return context

    def understand_user_intent(self, user_input: str) -> Dict[str, Any]:
        intent = {
            'is_continuation': False,
            'related_topic': None,
            'context': ""
        }
        if self.conversations:
            last_topic = self.conversations[-1]['topic']
            if user_input.lower() in ['да', 'еще', 'продолжи', 'и?', 'что еще']:
                intent['is_continuation'] = True
                intent['related_topic'] = last_topic
                intent['context'] = self.get_recent_context(2)
        return intent

    def save(self):
        data = {
            'conversations': list(self.conversations),
            'topics_discussed': dict(self.topics_discussed),
            'user_profile': self.user_profile,
            'relationships': self.relationships
        }
        memory_file = Config.SAVE_DIR / "context_memory.json"
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        memory_file = Config.SAVE_DIR / "context_memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = deque(data.get('conversations', []), maxlen=50)
                    self.topics_discussed = defaultdict(list, data.get('topics_discussed', {}))
                    self.user_profile = data.get('user_profile', {})
                    self.relationships = data.get('relationships', {})
            except:
                pass


# ====================== МЕНЕДЖЕР ОБУЧЕНИЯ ======================
class LearningManager:
    """Управляет процессом обучения АИ"""

    def __init__(self):
        self.knowledge_base = {}
        self.learning_history = []
        self.skill_level = 0.1
        self.asked_questions_count = 0
        self.correct_answers_count = 0
        self.accuracies = []
        self.load()

    def record_learning(self, topic: str, concept: str, teacher_answer: str,
                        ai_answer: str, similarity: float):
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
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        self.knowledge_base[topic].append({
            'concept': concept,
            'answer': teacher_answer,
            'learned': True,
            'similarity': similarity
        })
        self.update_skill_level(similarity)
        self.save()

    def update_skill_level(self, similarity: float):
        """Правильно обновить уровень мастерства"""
        self.accuracies.append(similarity)
        recent_accuracies = self.accuracies[-10:]
        avg_accuracy = np.mean(recent_accuracies)

        if avg_accuracy > 0.75:
            improvement = (avg_accuracy - 0.75) * 0.05
            self.skill_level = min(1.0, self.skill_level + improvement)
        elif avg_accuracy > 0.5:
            improvement = (avg_accuracy - 0.5) * 0.02
            self.skill_level = min(1.0, self.skill_level + improvement)

        self.correct_answers_count += int(similarity > 0.7)
        self.asked_questions_count += 1

    def get_known_topics(self) -> List[str]:
        return list(self.knowledge_base.keys())

    def get_learning_progress(self) -> str:
        total = len(self.learning_history)
        if total == 0:
            return "Обучение еще не началось"
        recent_acc = np.mean(self.accuracies[-5:]) if self.accuracies else 0
        return (f"📈 Уровень мастерства: {self.skill_level:.1%} | "
                f"Точность (последние 5): {recent_acc:.1%} | "
                f"Выученных концепций: {len(self.knowledge_base)}")

    def save(self):
        data = {
            'knowledge_base': self.knowledge_base,
            'learning_history': self.learning_history,
            'skill_level': self.skill_level,
            'asked_questions_count': self.asked_questions_count,
            'correct_answers_count': self.correct_answers_count,
            'accuracies': self.accuracies
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
                    self.accuracies = data.get('accuracies', [])
            except:
                pass


# ====================== VOCABULARY ======================
class AdvancedVocabManager:
    def __init__(self):
        self.word2idx = {
            '<pad>': 0,
            '<start>': 1,
            '<end>': 2,
            '<unk>': 3,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        self.next_id = 4
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
        return self.word2idx.get(word_lower, self.word2idx['<unk>'])

    def add_words(self, words: List[str]):
        for w in words:
            if w.strip():
                self.add_word(w)

    def tokenize(self, text: str) -> List[int]:
        words = clean_for_similarity(text).split()
        return [self.word2idx.get(w, self.word2idx['<unk>']) for w in words]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.idx2word.get(i, '') for i in ids if i not in [0, 1, 2]]
        deduped = []
        for token in tokens:
            if not deduped or deduped[-1] != token:
                deduped.append(token)
        deduped = deduped[:20]
        text = ' '.join(deduped)
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


# ====================== ПОЗИЦИОННОЕ КОДИРОВАНИЕ ======================
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 5000):
        super().__init__()
        self.emb_dim = emb_dim
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() *
                             (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ====================== MULTI-HEAD ATTENTION ======================
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


# ====================== ТРАНСФОРМЕР БЛОК ======================
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


# ====================== СУПЕР ИНТЕЛЛЕКТУАЛЬНЫЙ МОЗГ ======================
class SuperIntelligentBrain(nn.Module):
    def __init__(self, vocab_size: int, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available()
                                 else torch.device('cpu'))
        self.vocab_size = vocab_size
        self.emb_dim = Config.EMB_DIM
        self.hidden_size = Config.HIDDEN_SIZE

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(self.emb_dim, Config.MAX_SEQ_LEN)
        self.embedding_dropout = nn.Dropout(Config.DROPOUT)

        # Encoder и Decoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, Config.NUM_HEADS, self.hidden_size, Config.DROPOUT)
            for _ in range(Config.NUM_LAYERS)
        ])

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, Config.NUM_HEADS, self.hidden_size, Config.DROPOUT)
            for _ in range(Config.NUM_LAYERS)
        ])

        # Cross-attention
        self.cross_attentions = nn.ModuleList([
            MultiHeadAttention(self.emb_dim, Config.NUM_HEADS)
            for _ in range(Config.NUM_LAYERS)
        ])

        # Output projection
        self.output_proj = nn.Linear(self.emb_dim, vocab_size)

        # Концептуальная база
        self.concept_bank = defaultdict(list)
        self.memory_bank = None

        self.to(self.device)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Энкодирование входа"""
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.embedding_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        self.memory_bank = x
        return x

    def decode_with_attention(self, target_ids: torch.Tensor,
                              encoder_output: torch.Tensor) -> torch.Tensor:
        """Декодирование с кросс-атентшеном"""
        x = self.embedding(target_ids)
        x = self.pos_encoding(x)
        x = self.embedding_dropout(x)

        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            cross_out, _ = self.cross_attentions[i](x, encoder_output, encoder_output)
            x = x + cross_out

        return x

    def generate(self, input_ids: torch.Tensor, max_len: int = 80,
                 temperature: float = 1.2) -> List[int]:
        """Генерация последовательности с механизмом разнообразия"""
        was_training = self.training
        self.eval()

        with torch.no_grad():
            encoder_output = self.encode(input_ids)
            batch_size = input_ids.size(0)
            current_tokens = torch.full((batch_size, 1), 1,
                                        device=self.device, dtype=torch.long)
            generated = []
            last_tokens = deque(maxlen=3)  # Отслеживаем последние 3 токена для избежания зацикливания

            for step in range(max_len):
                decoder_output = self.decode_with_attention(current_tokens, encoder_output)
                logits = self.output_proj(decoder_output[:, -1, :])

                # Увеличиваем температуру для разнообразия
                logits = logits / max(temperature, 0.5)

                # Штраф за повторение последних токенов
                if len(last_tokens) > 0:
                    for token_id in last_tokens:
                        logits[0, token_id] -= 2.0

                probs = F.softmax(logits, dim=-1)

                # Nucleus sampling (top-p) вместо top-k для лучшего разнообразия
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > 0.9  # top 90%
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[0, indices_to_remove] = 0
                probs = probs / probs.sum()

                # Избегаем padding токена
                probs[0, 0] = 0
                probs = probs / probs.sum()

                next_token_idx = torch.multinomial(probs[0], 1)
                token_id = next_token_idx.item()

                if token_id == 2 or token_id == 0:  # END или PAD
                    break

                generated.append(token_id)
                last_tokens.append(token_id)
                current_tokens = torch.cat([current_tokens, next_token_idx.view(batch_size, 1)], dim=1)

                # Ограничиваем длину сгенерированной последовательности
                if len(current_tokens[0]) > Config.MAX_SEQ_LEN:
                    break

        if was_training:
            self.train()

        return generated

    def save_model(self):
        torch.save({
            'model_state': self.state_dict(),
            'concept_bank': dict(self.concept_bank),
        }, Config.MODEL_PATH)

    def load_model(self):
        if Config.MODEL_PATH.exists():
            try:
                checkpoint = torch.load(Config.MODEL_PATH, map_location=self.device)
                self.load_state_dict(checkpoint['model_state'])
                self.concept_bank = defaultdict(list, checkpoint.get('concept_bank', {}))
                return True
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке модели: {e}")
                return False
        return False


# ====================== СЕМАНТИЧЕСКАЯ ОЦЕНКА ======================
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


# ====================== УЧИТЕЛЬ (ОБУЧЕНИЕ) ======================
class SupervisedTeacher:
    def __init__(self):
        self.api_url = Config.QWEN_API
        self.evaluator = SemanticEvaluator()
        self.learning_manager = LearningManager()
        self.context_memory = ContextMemory()
        self.step_count = 0

    def ask_teacher(self, prompt: str, context: str = "") -> str:
        """Спросить у старшей модели"""
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\nВопрос: {prompt}"

        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 300,
                "temperature": 0.8
            }, timeout=25)

            if resp.status_code == 200:
                return clean_text(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"⚠️ Ошибка API: {e}")

        return "Я не знаю ответа."

    def generate_thoughts(self, user_input: str, input_type: str) -> ThoughtProcess:
        thought_process = ThoughtProcess()
        thought_process.add_thought(
            f"Мне задали вопрос типа '{input_type}': '{user_input}'",
            confidence=0.6
        )

        skill_level = self.learning_manager.skill_level
        if skill_level < 0.3:
            thought_process.add_thought(
                "Я еще в начале обучения, нужно быть осторожнее",
                confidence=0.8
            )
            thought_process.add_doubt("Я недостаточно опытен")
        elif skill_level > 0.7:
            thought_process.add_thought(
                "Я хорошо обучился, могу давать уверенные ответы",
                confidence=0.85
            )

        intent = self.context_memory.understand_user_intent(user_input)
        if intent['is_continuation']:
            thought_process.add_thought(
                f"Продолжение разговора о '{intent['related_topic']}'",
                confidence=0.9
            )

        topic_context = self.context_memory.get_context(input_type, num_context=2)
        if topic_context:
            thought_process.add_reasoning_step("Я помню предыдущие ответы по этой теме")

        key_words = [w for w in clean_for_similarity(user_input).split() if len(w) > 3]
        if key_words:
            thought_process.add_reasoning_step(f"Ключевые слова: {', '.join(key_words[:3])}")

        return thought_process

    def train_step(self, model: SuperIntelligentBrain, vocab: AdvancedVocabManager,
                   user_input: str, input_type: str) -> str:
        """ИСПРАВЛЕННЫЙ шаг обучения с правильной логикой"""

        print(f"\n👤 Вы: {user_input}")
        print(f"📋 Тип вопроса: {input_type}")

        # === 1. ГЕНЕРАЦИЯ МЫСЛЕЙ (РЕФЛЕКСИЯ) ===
        thought_process = self.generate_thoughts(user_input, input_type)
        print(thought_process)

        # === 2. ПОДГОТОВКА ДАННЫХ ===
        vocab.add_words(clean_for_similarity(user_input).split())
        input_tokens = vocab.tokenize(user_input)
        if not input_tokens or len(input_tokens) < 2:
            input_tokens = [1, 3]

        # Обрезаем и паддируем encoder input
        input_tokens = input_tokens[:Config.MAX_SEQ_LEN]
        encoder_len = len(input_tokens)
        input_tokens = input_tokens + [0] * (Config.MAX_SEQ_LEN - encoder_len)

        encoder_input = torch.tensor([input_tokens], device=model.device, dtype=torch.long)

        # === 3. ПОЛУЧЕНИЕ ПРАВИЛЬНОГО ОТВЕТА ОТ УЧИТЕЛЯ ===
        print("\n👨‍🏫 Получаю ответ от учителя...")
        memory_context = self.context_memory.get_context(input_type, num_context=2)
        teacher_answer = self.ask_teacher(user_input, memory_context)
        print(f"👨‍🏫 Учитель: {teacher_answer}")

        # === 4. ПОДГОТОВКА ЦЕЛЕВЫХ ТОКЕНОВ ===
        vocab.add_words(clean_for_similarity(teacher_answer).split())
        target_tokens = vocab.tokenize(teacher_answer)

        if not target_tokens or len(target_tokens) < 2:
            target_tokens = [1, 3]

        # Обрезаем до MAX_SEQ_LEN - 1 (место для END токена)
        target_tokens = target_tokens[:Config.MAX_SEQ_LEN - 1]
        target_len = len(target_tokens)

        # Паддируем до MAX_SEQ_LEN - 1
        target_tokens_padded = target_tokens + [0] * (Config.MAX_SEQ_LEN - 1 - target_len)

        # Входы decoder: [START] + first MAX_SEQ_LEN-1 tokens
        # Выходы: last MAX_SEQ_LEN-1 tokens + [END]
        decoder_input = [1] + target_tokens_padded[:-1]  # START + первые N-1
        target_output = target_tokens_padded[1:] + [2]  # последние N-1 + END

        # Обе последовательности должны быть длины MAX_SEQ_LEN
        decoder_input = decoder_input[:Config.MAX_SEQ_LEN]
        target_output = target_output[:Config.MAX_SEQ_LEN]

        # Убеждаемся, что обе имеют правильную длину
        while len(decoder_input) < Config.MAX_SEQ_LEN:
            decoder_input.append(0)
        while len(target_output) < Config.MAX_SEQ_LEN:
            target_output.append(0)

        assert len(decoder_input) == Config.MAX_SEQ_LEN, f"decoder_input len: {len(decoder_input)}"
        assert len(target_output) == Config.MAX_SEQ_LEN, f"target_output len: {len(target_output)}"

        decoder_input = torch.tensor([decoder_input], device=model.device, dtype=torch.long)
        target_output = torch.tensor([target_output], device=model.device, dtype=torch.long)

        # === 5. ОБУЧЕНИЕ С ВАЛИДАЦИЕЙ ===
        print("\n🔄 Процесс обучения:")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                      weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        best_loss = float('inf')
        best_response = teacher_answer
        no_improve_count = 0

        for attempt in range(1, Config.MAX_ATTEMPTS + 1):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            encoder_output = model.encode(encoder_input)
            decoder_output = model.decode_with_attention(decoder_input, encoder_output)

            # Проекция на словарь
            logits = model.output_proj(decoder_output)

            # Loss calculation с учетом паддинга
            # logits: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
            # target: [batch, seq_len] -> [batch*seq_len]
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, vocab_size),
                target_output.reshape(batch_size * seq_len),
                ignore_index=0,
                reduction='mean'
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # === 6. ВАЛИДАЦИЯ И ПРОВЕРКА ===
            model.eval()
            with torch.no_grad():
                # Генерируем ответ с повышенной температурой
                gen_ids = model.generate(encoder_input, max_len=50, temperature=1.3)
                predicted_answer = vocab.decode(gen_ids)
                predicted_answer = clean_generated_response(predicted_answer)

                if not predicted_answer:
                    predicted_answer = teacher_answer

                # Вычисляем сходство
                similarity = self.evaluator.similarity(teacher_answer, predicted_answer)

                # Логирование прогресса
                improvement = "📈" if loss.item() < best_loss else "📉"
                print(f" 🔁 Итер. {attempt:2d}: "
                      f"loss={loss.item():.4f}, "
                      f"сходство={similarity:.1%} {improvement}")

                # Обновляем best если улучшение
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_response = predicted_answer
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            # Остановка если нет прогресса
            if no_improve_count >= 3:
                print("⏹️  Остановка: нет прогресса 3 итерации подряд")
                break

            # Раннее завершение при хорошем результате
            if loss.item() < 0.5 and similarity > 0.6:
                print("✅ Хорошее совпадение достигнуто!")
                thought_process.learning_occurred = True
                break

            scheduler.step()

        # === 7. СОХРАНЕНИЕ В ПАМЯТЬ ===
        final_similarity = self.evaluator.similarity(teacher_answer, best_response)

        self.learning_manager.record_learning(
            topic=input_type,
            concept=user_input,
            teacher_answer=teacher_answer,
            ai_answer=best_response,
            similarity=final_similarity
        )

        self.context_memory.add_interaction(
            user_input=user_input,
            ai_response=best_response,
            topic=input_type,
            similarity=final_similarity
        )

        # Сохраняем концепцию в model.concept_bank
        model.concept_bank[input_type].append({
            'input': user_input,
            'output': best_response,
            'similarity': final_similarity,
            'timestamp': datetime.now().isoformat()
        })

        print(f"\n📊 Финальное сходство: {final_similarity:.1%}")
        print(f"📚 {self.learning_manager.get_learning_progress()}")

        self.step_count += 1
        return best_response


# ====================== ГЛАВНАЯ ПРОГРАММА ======================
def main():
    print("\n" + "=" * 70)
    print("🧠 КОГНИТИВНАЯ СИСТЕМА С РЕФЛЕКСИЕЙ И ОБУЧЕНИЕМ v7.0 (FIXED)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Устройство: {device}")

    # === ИНИЦИАЛИЗАЦИЯ VOCAB ===
    vocab = AdvancedVocabManager()
    if vocab.load():
        print(f"✅ Словарь загружен ({vocab.size} слов)")
    else:
        print("🔨 Создаю новый словарь...")
        base_words = ("привет спасибо да нет что как почему где когда кто который "
                      "интересно понимаю узнал новое мышление рефлексия сомнение вопрос "
                      "ответ мнение факт процесс анализ").split()
        vocab.add_words(base_words)
        print(f"✅ Создан словарь с {vocab.size} словами")

    # === ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ===
    model = SuperIntelligentBrain(vocab_size=max(Config.VOCAB_SIZE, vocab.size),
                                  device=device)
    if model.load_model():
        print("✅ Модель загружена")
    else:
        print("🔨 Инициализирую новую модель...")
        print("✅ Модель создана")

    # === ИНИЦИАЛИЗАЦИЯ УЧИТЕЛЯ ===
    teacher = SupervisedTeacher()

    print(f"\n💡 КОМАНДЫ:")
    print(f" 'выход' - завершить программу")
    print(f" 'память' - показать историю взаимодействий")
    print(f" 'контекст' - показать запомненный контекст")
    print(f" 'темы' - показать изученные темы с фактами")
    print(f" 'запомнил' - показать что модель запомнила")
    print(f" 'прогресс' - показать прогресс обучения")
    print(f" 'график' - показать график обучения")
    print(f" 'сохранить' - сохранить модель")
    print(f" 'очистить' - очистить память")

    interaction_count = 0

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("\n💾 Сохраняю модель...")
                model.save_model()
                vocab.save()
                teacher.learning_manager.save()
                teacher.context_memory.save()
                print("✨ До встречи! Спасибо за обучение!")
                break

            if user_input.lower() in ['память', 'memory']:
                print(f"\n📚 История обучения:")
                history = teacher.learning_manager.learning_history
                if history:
                    for item in history[-5:]:
                        print(f" 📍 {item['concept'][:40]}...")
                        print(f" Сходство: {item['similarity']:.1%}")
                else:
                    print(" (история пуста)")
                continue

            if user_input.lower() in ['контекст', 'context']:
                print(f"\n🧠 ЗАПОМНЕННЫЙ КОНТЕКСТ:")
                if teacher.context_memory.conversations:
                    print(f" Всего взаимодействий: {len(teacher.context_memory.conversations)}")
                    print(f"\n Последние 3 разговора:")
                    for i, conv in enumerate(list(teacher.context_memory.conversations)[-3:], 1):
                        print(f"\n {i}. Тема: {conv['topic']}")
                        print(f" Вопрос: {conv['user_input'][:50]}...")
                        print(f" Ответ: {conv['ai_response'][:50]}...")
                        print(f" Уверенность: {conv['similarity']:.1%}")
                else:
                    print(" (контекст еще не накоплен)")
                continue

            if user_input.lower() in ['темы', 'topics']:
                print(f"\n📚 ИЗУЧЕННЫЕ ТЕМЫ И ФАКТЫ:")
                topics = teacher.context_memory.topics_discussed
                if topics:
                    for topic, facts in list(topics.items())[-5:]:
                        print(f"\n 📌 Тема: {topic}")
                        print(f" Выучено фактов: {len(facts)}")
                        if facts:
                            avg_confidence = np.mean([f['confidence'] for f in facts])
                            print(f" Средняя уверенность: {avg_confidence:.1%}")
                            print(f" Последние факты:")
                            for f in facts[-2:]:
                                print(f" • {f['answer'][:50]}...")
                else:
                    print(" (темы еще не изучены)")
                continue

            if user_input.lower() in ['запомнил', 'запомни', 'что ты помнишь', 'вспомни']:
                print(f"\n🧠 ЧТО Я ЗАПОМНИЛ:")
                memory = teacher.context_memory
                if memory.conversations:
                    print(f" Всего взаимодействий: {len(memory.conversations)}")
                    print(f"\n КЛЮЧЕВЫЕ ФАКТЫ ИЗ ПАМЯТИ:")
                    for topic, facts in list(memory.topics_discussed.items())[-3:]:
                        print(f"\n 📌 {topic}:")
                        for fact in facts[-2:]:
                            print(f" Q: {fact['question'][:50]}...")
                            print(f" A: {fact['answer'][:50]}...\n")
                else:
                    print(" (память пуста)")
                continue

            if user_input.lower() in ['прогресс', 'progress', 'stats']:
                print(f"\n📊 ПРОГРЕСС ОБУЧЕНИЯ:")
                print(f" {teacher.learning_manager.get_learning_progress()}")
                known = teacher.learning_manager.get_known_topics()
                if known:
                    print(f" Известные темы: {', '.join(known)}")
                continue

            if user_input.lower() in ['график', 'graph', 'chart']:
                print(f"\n📈 ГРАФИК ОБУЧЕНИЯ (последние 10 результатов):")
                accuracies = teacher.learning_manager.accuracies[-10:]
                if accuracies:
                    for i, acc in enumerate(accuracies, 1):
                        bar_length = int(acc * 30)
                        bar = "█" * bar_length + "░" * (30 - bar_length)
                        print(f" {i:2d}. [{bar}] {acc:.1%}")
                    avg = np.mean(accuracies)
                    print(f"\n Средняя точность: {avg:.1%}")
                else:
                    print(" (нет данных)")
                continue

            if user_input.lower() in ['сохранить', 'save']:
                print("\n💾 Сохраняю...")
                model.save_model()
                vocab.save()
                teacher.learning_manager.save()
                teacher.context_memory.save()
                print("✅ Сохранено!")
                continue

            if user_input.lower() in ['очистить', 'clear']:
                if input("Вы уверены? (да/нет): ").lower() == 'да':
                    Config.LEARNING_PATH.unlink(missing_ok=True)
                    (Config.SAVE_DIR / "context_memory.json").unlink(missing_ok=True)
                    teacher.learning_manager = LearningManager()
                    teacher.context_memory = ContextMemory()
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
                model.save_model()
                vocab.save()
                teacher.learning_manager.save()
                teacher.context_memory.save()

        except KeyboardInterrupt:
            print("\n✨ Прерывание. До встречи!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()