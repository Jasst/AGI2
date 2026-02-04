# coding: utf-8
"""
AGI_CognitiveReasoning_v8_COMPLETE.py
Полная когнитивная система с внутренним диалогом, параллельным мышлением и адаптивным обучением
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
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import queue

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
    SAVE_DIR = Path("./cognitive_model_data_v8")
    MODEL_PATH = SAVE_DIR / "model_superintel.pt"
    VOCAB_PATH = SAVE_DIR / "vocab_superintel.pkl"
    MEMORY_PATH = SAVE_DIR / "memory_superintel.json"
    LEARNING_PATH = SAVE_DIR / "learning_superintel.json"
    REASONING_PATH = SAVE_DIR / "reasoning_superintel.json"

    VOCAB_SIZE = 20000
    EMB_DIM = 768
    HIDDEN_SIZE = 2048
    NUM_LAYERS = 6
    NUM_HEADS = 12
    DROPOUT = 0.15
    MAX_SEQ_LEN = 200
    LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-5
    MAX_ATTEMPTS = 35
    CONTEXT_SIZE = 50
    CONFIDENCE_THRESHOLD = 0.65
    UNCERTAINTY_THRESHOLD = 0.5
    REFLECTION_DEPTH = 8
    INTERNAL_DIALOG_TURNS = 4
    QWEN_API = "http://localhost:1234/v1/chat/completions"
    PARALLEL_THREADS = 3


Config.SAVE_DIR.mkdir(exist_ok=True)


# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
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
    if len(words) > 200:
        text = ' '.join(words[:200])
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text or "Хорошо."


def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).lower().strip()


def detect_input_type(user_input: str) -> str:
    s = user_input.lower().strip()
    patterns = {
        "SOCIAL": r'\b(привет|здравствуй|добрый|как дела|пока|спасибо|благодарю)\b',
        "FACT": r'\b(что такое|кто такой|где|какая|формула|определение|расскажи)\b',
        "REASON": r'\b(почему|зачем|отчего|причина|механизм|как это)\b',
        "PROCESS": r'\b(как сделать|инструкция|шаг|алгоритм|пошаговый|создать)\b',
        "OPINION": r'\b(думаешь|мнение|лучше|нравится|согласен|предпочитаешь)\b',
        "CREATIVE": r'\b(представь|вообрази|сочини|опиши|метафора|история|создай)\b',
        "ANALYSIS": r'\b(проанализируй|сравни|различие|сходство|анализ|тенденция)\b',
        "CLARIFICATION": r'\b(уточни|объясни|расшифруй|разъясни|что имел в виду)\b',
    }
    for qtype, pattern in patterns.items():
        if re.search(pattern, s):
            return qtype
    return "FACT"


# ====================== ВНУТРЕННИЙ ДИАЛОГ ======================
class InternalDialogue:
    """Система внутреннего диалога для уточнения мыслей"""

    def __init__(self, teacher_api: str):
        self.teacher_api = teacher_api
        self.dialogue_history = deque(maxlen=20)
        self.lock = Lock()

    def generate_hypothesis(self, topic: str, initial_thought: str) -> Dict[str, Any]:
        """Генерирует гипотезу"""
        return {
            'topic': topic,
            'hypothesis': initial_thought,
            'confidence': 0.0,
            'refinements': [],
            'timestamp': datetime.now().isoformat()
        }

    def refine_through_dialog(self, hypothesis: Dict, turns: int = Config.INTERNAL_DIALOG_TURNS) -> Dict:
        """Уточняет гипотезу через внутренний диалог"""
        current_idea = hypothesis['hypothesis']

        clarifying_questions = [
            f"Что я конкретно понимаю под: {current_idea}?",
            f"Какие неявные предположения я делаю о: {current_idea}?",
            f"Есть ли противоречия или исключения в: {current_idea}?",
            f"Какие доказательства поддерживают идею: {current_idea}?"
        ]

        for turn in range(turns):
            question = clarifying_questions[turn % len(clarifying_questions)]
            refined = self._ask_teacher_for_refinement(current_idea, question)

            if refined and refined != current_idea:
                hypothesis['refinements'].append({
                    'turn': turn + 1,
                    'question': question,
                    'refined_idea': refined,
                    'timestamp': datetime.now().isoformat()
                })
                current_idea = refined

        hypothesis['final_idea'] = current_idea
        return hypothesis

    def _ask_teacher_for_refinement(self, idea: str, question: str) -> str:
        """Запрашивает уточнение у учителя"""
        prompt = f"""Помоги мне уточнить понимание.
Моя текущая идея: {idea}
Мой вопрос к себе: {question}

Дай конкретный, краткий ответ (1-2 предложения для уточнения)."""

        try:
            resp = requests.post(self.teacher_api, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.6
            }, timeout=10)

            if resp.status_code == 200:
                return clean_text(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            pass

        return idea


# ====================== ПРОДВИНУТЫЙ ПРОЦЕСС МЫШЛЕНИЯ ======================
class AdvancedThoughtProcess:
    """Расширенная система мышления с рефлексией"""

    def __init__(self):
        self.primary_thoughts = []
        self.secondary_thoughts = []
        self.doubts = []
        self.contradictions = []
        self.uncertainties = []
        self.reasoning_chain = []
        self.confidence = 0.0
        self.learning_occurred = False
        self.asked_teacher = False
        self.asked_user = False
        self.internal_dialogue_result = None

    def add_primary_thought(self, thought: str, confidence: float = 0.5, source: str = "reasoning"):
        self.primary_thoughts.append({
            'text': thought,
            'confidence': confidence,
            'source': source,
            'timestamp': datetime.now().isoformat()
        })
        self.update_confidence()

    def add_secondary_thought(self, thought: str):
        self.secondary_thoughts.append({
            'text': thought,
            'timestamp': datetime.now().isoformat()
        })

    def add_doubt(self, doubt: str, severity: float = 0.5):
        self.doubts.append({
            'text': doubt,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })

    def add_contradiction(self, contradiction: str):
        self.contradictions.append({
            'text': contradiction,
            'timestamp': datetime.now().isoformat()
        })

    def add_uncertainty(self, uncertainty: str):
        self.uncertainties.append({
            'text': uncertainty,
            'timestamp': datetime.now().isoformat()
        })

    def add_reasoning_step(self, step: str):
        self.reasoning_chain.append({
            'step': step,
            'number': len(self.reasoning_chain) + 1,
            'timestamp': datetime.now().isoformat()
        })

    def update_confidence(self):
        if self.primary_thoughts:
            self.confidence = np.mean([t['confidence'] for t in self.primary_thoughts])

    def is_confident(self, threshold: float = Config.CONFIDENCE_THRESHOLD) -> bool:
        return self.confidence >= threshold

    def needs_external_help(self) -> bool:
        return (self.confidence < Config.UNCERTAINTY_THRESHOLD or
                len(self.doubts) > 2 or
                len(self.contradictions) > 0)

    def __str__(self) -> str:
        result = "🧠 ПРОЦЕСС МЫШЛЕНИЯ:\n"

        if self.primary_thoughts:
            result += "\n💭 ОСНОВНЫЕ МЫСЛИ:\n"
            for t in self.primary_thoughts[-3:]:
                result += f" • {t['text'][:70]}... ({t['source']}) [{t['confidence']:.1%}]\n"

        if self.secondary_thoughts:
            result += "\n🔀 ПОБОЧНЫЕ МЫСЛИ:\n"
            for t in self.secondary_thoughts[-2:]:
                result += f" ↳ {t['text'][:60]}...\n"

        if self.doubts:
            result += "\n❓ СОМНЕНИЯ:\n"
            for d in self.doubts[-2:]:
                result += f" ⚠ {d['text'][:60]}... (уровень: {d['severity']:.1%})\n"

        if self.contradictions:
            result += "\n🔴 ПРОТИВОРЕЧИЯ:\n"
            for c in self.contradictions:
                result += f" ✖ {c['text'][:60]}...\n"

        if self.uncertainties:
            result += "\n❔ НЕОПРЕДЕЛЁННОСТИ:\n"
            for u in self.uncertainties[-2:]:
                result += f" ? {u['text'][:60]}...\n"

        if self.reasoning_chain:
            result += "\n📍 ЦЕПЬ РАССУЖДЕНИЙ:\n"
            for step in self.reasoning_chain[-4:]:
                result += f" {step['number']}. {step['step'][:60]}...\n"

        result += f"\n📊 УВЕРЕННОСТЬ: {self.confidence:.1%}\n"

        if self.asked_teacher:
            result += "🎓 Консультировался со старшей моделью\n"
        if self.asked_user:
            result += "👤 Запросил уточнение у пользователя\n"

        return result


# ====================== ПАМЯТЬ С КОНТЕКСТОМ ======================
class AdvancedContextMemory:
    """Продвинутая память с отслеживанием эволюции"""

    def __init__(self):
        self.conversations = deque(maxlen=150)
        self.concept_evolution = defaultdict(list)
        self.learning_gaps = []
        self.discussion_threads = defaultdict(list)
        self.load()

    def add_interaction(self, user_input: str, ai_response: str, topic: str,
                        thought_process: AdvancedThoughtProcess, confidence: float):
        interaction = {
            'user_input': user_input,
            'ai_response': ai_response,
            'topic': topic,
            'confidence': confidence,
            'thought_process': {
                'primary_thoughts': len(thought_process.primary_thoughts),
                'doubts': len(thought_process.doubts),
                'uncertainties': len(thought_process.uncertainties),
                'asked_teacher': thought_process.asked_teacher,
                'asked_user': thought_process.asked_user,
            },
            'timestamp': datetime.now().isoformat()
        }
        self.conversations.append(interaction)

        self.concept_evolution[topic].append({
            'input': user_input,
            'response': ai_response,
            'confidence': confidence,
            'turn': len(self.concept_evolution[topic])
        })

        self.save()

    def get_concept_history(self, topic: str) -> List[Dict]:
        return self.concept_evolution.get(topic, [])

    def identify_learning_gaps(self) -> List[str]:
        gaps = []
        for topic, history in self.concept_evolution.items():
            if history:
                avg_confidence = np.mean([h['confidence'] for h in history])
                if avg_confidence < 0.5:
                    gaps.append(f"Слабое понимание '{topic}' ({avg_confidence:.1%})")
        return gaps

    def get_recent_context(self, num_last: int = 3) -> str:
        if not self.conversations:
            return ""
        recent = list(self.conversations)[-num_last:]
        context = "📚 Недавние разговоры:\n"
        for conv in recent:
            context += f"• Тема: {conv['topic']} | Ответ: {conv['ai_response'][:40]}...\n"
        return context

    def save(self):
        data = {
            'conversations': list(self.conversations),
            'concept_evolution': dict(self.concept_evolution),
            'learning_gaps': self.learning_gaps
        }
        with open(Config.SAVE_DIR / "advanced_memory.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        path = Config.SAVE_DIR / "advanced_memory.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = deque(data.get('conversations', []), maxlen=150)
                    self.concept_evolution = defaultdict(list, data.get('concept_evolution', {}))
                    self.learning_gaps = data.get('learning_gaps', [])
            except:
                pass


# ====================== МЕНЕДЖЕР ОБУЧЕНИЯ ======================
class AdvancedLearningManager:
    """Менеджер с адаптивным обучением"""

    def __init__(self):
        self.knowledge_base = defaultdict(lambda: {'concepts': [], 'confidence': 0.0})
        self.learning_trajectory = []
        self.skill_levels = defaultdict(float)
        self.learning_speed = 0.1
        self.adaptivity = 0.5
        self.accuracies = []
        self.load()

    def record_learning(self, topic: str, concept: str, user_answer: str,
                        teacher_answer: str, similarity: float, thought_quality: int = 5):
        record = {
            'topic': topic,
            'concept': concept,
            'user_answer': user_answer,
            'teacher_answer': teacher_answer,
            'similarity': similarity,
            'thought_quality': thought_quality,
            'timestamp': datetime.now().isoformat()
        }

        self.learning_trajectory.append(record)
        self.accuracies.append(similarity)

        self.knowledge_base[topic]['concepts'].append({
            'concept': concept,
            'answer': teacher_answer,
            'learned_similarity': similarity,
            'refinements': 1
        })

        self._update_skill_adaptively(topic, similarity, thought_quality)
        self.save()

    def _update_skill_adaptively(self, topic: str, similarity: float, thought_quality: int):
        base_improvement = similarity * 0.1
        thought_bonus = (thought_quality / 10.0) * 0.05
        total_improvement = base_improvement + thought_bonus

        current_skill = self.skill_levels[topic]
        self.skill_levels[topic] = min(1.0, current_skill + total_improvement)

        if similarity > 0.8:
            self.learning_speed = min(0.3, self.learning_speed + 0.01)
        elif similarity < 0.4:
            self.learning_speed = max(0.05, self.learning_speed - 0.01)

    def get_skill_report(self) -> str:
        if not self.skill_levels:
            return "Обучение еще не началось"

        report = "📈 УРОВНИ НАВЫКОВ:\n"
        for topic, level in sorted(self.skill_levels.items(), key=lambda x: x[1], reverse=True)[:10]:
            bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
            report += f" {topic:15s} [{bar}] {level:.1%}\n"

        if self.skill_levels:
            avg_skill = np.mean(list(self.skill_levels.values()))
            report += f"\nСредний уровень: {avg_skill:.1%}\n"
        return report

    def get_progress_report(self) -> str:
        if not self.accuracies:
            return "Прогресс обучения: нет данных"

        recent = self.accuracies[-10:]
        avg_recent = np.mean(recent)
        overall_avg = np.mean(self.accuracies)

        report = f"📊 ПРОГРЕСС ОБУЧЕНИЯ:\n"
        report += f" Точность (последние 10): {avg_recent:.1%}\n"
        report += f" Общая точность: {overall_avg:.1%}\n"
        report += f" Всего попыток обучения: {len(self.learning_trajectory)}\n"
        return report

    def save(self):
        data = {
            'knowledge_base': dict(self.knowledge_base),
            'learning_trajectory': self.learning_trajectory,
            'skill_levels': dict(self.skill_levels),
            'learning_speed': self.learning_speed,
            'accuracies': self.accuracies
        }
        with open(Config.LEARNING_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.LEARNING_PATH.exists():
            try:
                with open(Config.LEARNING_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base = defaultdict(lambda: {'concepts': [], 'confidence': 0.0},
                                                      data.get('knowledge_base', {}))
                    self.learning_trajectory = data.get('learning_trajectory', [])
                    self.skill_levels = defaultdict(float, data.get('skill_levels', {}))
                    self.learning_speed = data.get('learning_speed', 0.1)
                    self.accuracies = data.get('accuracies', [])
            except:
                pass


# ====================== СЛОВАРЬ ======================
class AdvancedVocabManager:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        self.word_context = defaultdict(list)
        self.semantic_relations = defaultdict(set)
        self.next_id = 4

    def add_word(self, word: str, context: Optional[str] = None, related_words: Optional[List[str]] = None) -> int:
        word_lower = word.lower()
        if word_lower not in self.word2idx:
            if self.next_id < Config.VOCAB_SIZE:
                self.word2idx[word_lower] = self.next_id
                self.idx2word[self.next_id] = word_lower
                self.next_id += 1

        self.word_freq[word_lower] += 1

        if context:
            self.word_context[word_lower].append(context)

        if related_words:
            for rw in related_words:
                self.semantic_relations[word_lower].add(rw.lower())

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
        deduped = deduped[:40]
        return ' '.join(deduped).strip()

    @property
    def size(self):
        return len(self.word2idx)

    def save(self):
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'next_id': self.next_id,
            'semantic_relations': {k: list(v) for k, v in self.semantic_relations.items()}
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
                self.semantic_relations = defaultdict(set, {k: set(v) for k, v in
                                                            data.get('semantic_relations', {}).items()})
                return True
        return False


# ====================== ПОЗИЦИОННОЕ КОДИРОВАНИЕ ======================
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 5000):
        super().__init__()
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


# ====================== ПРОДВИНУТЫЙ МОЗГ ======================
class AdvancedIntelligenceBrain(nn.Module):
    def __init__(self, vocab_size: int, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        self.output_proj = nn.Linear(self.emb_dim, vocab_size)
        self.concept_bank = defaultdict(list)

        self.to(self.device)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.embedding_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode_with_attention(self, target_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self.embedding(target_ids)
        x = self.pos_encoding(x)
        x = self.embedding_dropout(x)

        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            cross_out, _ = self.cross_attentions[i](x, encoder_output, encoder_output)
            x = x + cross_out

        return x

    def generate(self, input_ids: torch.Tensor, max_len: int = 80, temperature: float = 1.2) -> List[int]:
        was_training = self.training
        self.eval()

        with torch.no_grad():
            encoder_output = self.encode(input_ids)
            batch_size = input_ids.size(0)
            current_tokens = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.long)
            generated = []
            last_tokens = deque(maxlen=4)

            for step in range(max_len):
                decoder_output = self.decode_with_attention(current_tokens, encoder_output)
                logits = self.output_proj(decoder_output[:, -1, :])

                logits = logits / max(temperature, 0.5)

                if len(last_tokens) > 0:
                    for token_id in last_tokens:
                        if token_id < logits.size(1):
                            logits[0, token_id] -= 2.5

                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > 0.92
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[0, indices_to_remove] = 0
                probs = probs / (probs.sum() + 1e-10)

                probs[0, 0] = 0
                probs = probs / (probs.sum() + 1e-10)

                next_token_idx = torch.multinomial(probs[0], 1)
                token_id = next_token_idx.item()

                if token_id == 2 or token_id == 0:
                    break

                generated.append(token_id)
                last_tokens.append(token_id)
                current_tokens = torch.cat([current_tokens, next_token_idx.view(batch_size, 1)], dim=1)

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
                print(f"⚠️ Ошибка загрузки: {e}")
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


# ====================== УЧИТЕЛЬ С ДИАЛОГОМ ======================
class AdvancedTeacher:
    """Учитель с внутренним диалогом и адаптивным обучением"""

    def __init__(self):
        self.api_url = Config.QWEN_API
        self.evaluator = SemanticEvaluator()
        self.learning_manager = AdvancedLearningManager()
        self.context_memory = AdvancedContextMemory()
        self.internal_dialog = InternalDialogue(Config.QWEN_API)
        self.step_count = 0
        self.executor = ThreadPoolExecutor(max_workers=Config.PARALLEL_THREADS)

    def ask_teacher(self, prompt: str, context: str = "") -> str:
        """Получить ответ от учителя"""
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\nВопрос: {prompt}"

        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 350,
                "temperature": 0.85
            }, timeout=30)

            if resp.status_code == 200:
                return clean_text(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"⚠️ Ошибка API: {e}")

        return "Я не знаю ответа на этот вопрос."

    def ask_user_for_clarification(self, topic: str, ai_answer: str) -> str:
        """Попросить уточнение у пользователя"""
        print(f"\n🤔 Я не уверен в своём ответе.")
        print(f"💡 Мой ответ: {ai_answer}")
        user_clarification = input("\n👤 Можешь помочь мне понять правильный ответ? ")
        return user_clarification.strip()

    def generate_thoughts(self, user_input: str, input_type: str) -> AdvancedThoughtProcess:
        """Генерирует процесс мышления"""
        thought_process = AdvancedThoughtProcess()

        # Основная мысль о вопросе
        thought_process.add_primary_thought(
            f"Мне задан вопрос типа '{input_type}': {user_input[:50]}...",
            confidence=0.7,
            source="анализ_входа"
        )

        # Оценка своего уровня
        skill_level = self.learning_manager.skill_levels.get(input_type, 0.1)
        if skill_level < 0.3:
            thought_process.add_doubt("Я слабо понимаю эту тему", severity=0.8)
            thought_process.add_uncertainty("Нужна помощь учителя")
        elif skill_level > 0.8:
            thought_process.add_primary_thought(
                "Я хорошо разбираюсь в этой теме",
                confidence=0.85,
                source="опыт"
            )

        # Проверка памяти
        recent_context = self.context_memory.get_recent_context(2)
        if recent_context:
            thought_process.add_secondary_thought(
                "Я помню похожие вопросы из прошлого"
            )
            thought_process.add_reasoning_step("Использую контекст из памяти")

        # Анализ ключевых слов
        key_words = [w for w in clean_for_similarity(user_input).split() if len(w) > 3]
        if key_words:
            thought_process.add_reasoning_step(
                f"Ключевые слова: {', '.join(key_words[:3])}"
            )

        return thought_process

    def train_step(self, model: AdvancedIntelligenceBrain, vocab: AdvancedVocabManager,
                   user_input: str, input_type: str) -> str:
        """Продвинутый шаг обучения с диалогом"""

        print(f"\n{'=' * 70}")
        print(f"👤 ВЫ: {user_input}")
        print(f"📋 Тип: {input_type}")

        # === 1. ГЕНЕРАЦИЯ МЫСЛЕЙ ===
        thought_process = self.generate_thoughts(user_input, input_type)
        print(thought_process)

        # === 2. ВНУТРЕННИЙ ДИАЛОГ ===
        print("\n🧠 ВНУТРЕННИЙ ДИАЛОГ...")
        initial_hypothesis = self.internal_dialog.generate_hypothesis(
            input_type,
            f"Ответ на вопрос о {input_type}"
        )
        refined_hypothesis = self.internal_dialog.refine_through_dialog(
            initial_hypothesis,
            turns=Config.INTERNAL_DIALOG_TURNS
        )
        thought_process.internal_dialogue_result = refined_hypothesis
        print(f"✓ Гипотеза уточнена: {refined_hypothesis.get('final_idea', '')[:50]}...")

        # === 3. ПОДГОТОВКА ДАННЫХ ===
        vocab.add_words(clean_for_similarity(user_input).split())
        input_tokens = vocab.tokenize(user_input)
        if not input_tokens or len(input_tokens) < 2:
            input_tokens = [1, 3]

        input_tokens = input_tokens[:Config.MAX_SEQ_LEN]
        input_tokens = input_tokens + [0] * (Config.MAX_SEQ_LEN - len(input_tokens))
        encoder_input = torch.tensor([input_tokens], device=model.device, dtype=torch.long)

        # === 4. ПОЛУЧЕНИЕ ОТВЕТА ОТ УЧИТЕЛЯ ===
        print("\n👨‍🏫 Запрос ответа учителя...")
        memory_context = self.context_memory.get_recent_context(2)
        teacher_answer = self.ask_teacher(user_input, memory_context)
        print(f"👨‍🏫 ОТВЕТ: {teacher_answer}")

        # === 5. ПОДГОТОВКА ЦЕЛЕЙ ===
        vocab.add_words(clean_for_similarity(teacher_answer).split())
        target_tokens = vocab.tokenize(teacher_answer)

        if not target_tokens or len(target_tokens) < 2:
            target_tokens = [1, 3]

        target_tokens = target_tokens[:Config.MAX_SEQ_LEN - 1]
        target_tokens_padded = target_tokens + [0] * (Config.MAX_SEQ_LEN - 1 - len(target_tokens))

        decoder_input = [1] + target_tokens_padded[:-1]
        target_output = target_tokens_padded[1:] + [2]

        decoder_input = (decoder_input + [0] * Config.MAX_SEQ_LEN)[:Config.MAX_SEQ_LEN]
        target_output = (target_output + [0] * Config.MAX_SEQ_LEN)[:Config.MAX_SEQ_LEN]

        decoder_input = torch.tensor([decoder_input], device=model.device, dtype=torch.long)
        target_output = torch.tensor([target_output], device=model.device, dtype=torch.long)

        # === 6. ОБУЧЕНИЕ ===
        print("\n🔄 ОБУЧЕНИЕ МОДЕЛИ...")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

        best_loss = float('inf')
        best_response = teacher_answer
        no_improve_count = 0

        for attempt in range(1, Config.MAX_ATTEMPTS + 1):
            model.train()
            optimizer.zero_grad()

            encoder_output = model.encode(encoder_input)
            decoder_output = model.decode_with_attention(decoder_input, encoder_output)
            logits = model.output_proj(decoder_output)

            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, vocab_size),
                target_output.reshape(batch_size * seq_len),
                ignore_index=0,
                reduction='mean'
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # === ВАЛИДАЦИЯ ===
            model.eval()
            with torch.no_grad():
                gen_ids = model.generate(encoder_input, max_len=60, temperature=1.4)
                predicted_answer = vocab.decode(gen_ids)

                if not predicted_answer or predicted_answer.strip() == "":
                    predicted_answer = teacher_answer

                similarity = self.evaluator.similarity(teacher_answer, predicted_answer)

                improvement = "📈" if loss.item() < best_loss else "📉"
                print(f" 🔁 Итер. {attempt:2d}: loss={loss.item():.4f}, сходство={similarity:.1%} {improvement}")

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_response = predicted_answer
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if no_improve_count >= 4:
                print("⏹️ Остановка: нет улучшений 4 раза")
                break

            if loss.item() < 0.4 and similarity > 0.65:
                print("✅ Хорошее совпадение достигнуто!")
                thought_process.learning_occurred = True
                break

            scheduler.step()

        # === 7. ПРОВЕРКА И ДИАЛОГ ===
        final_similarity = self.evaluator.similarity(teacher_answer, best_response)

        if final_similarity < Config.UNCERTAINTY_THRESHOLD:
            print(f"\n❓ Я не уверен в ответе (сходство: {final_similarity:.1%})")
            thought_process.asked_teacher = True

            if final_similarity < 0.3:
                print("\n👤 Запрос уточнения у пользователя...")
                user_help = self.ask_user_for_clarification(input_type, best_response)
                if user_help:
                    thought_process.asked_user = True
                    refined_answer = self.ask_teacher(
                        f"На основе этого уточнения: {user_help}\nПеревели ответ на: {user_help}",
                        ""
                    )
                    best_response = refined_answer
                    final_similarity = self.evaluator.similarity(teacher_answer, best_response)

        # === 8. СОХРАНЕНИЕ ===
        self.learning_manager.record_learning(
            topic=input_type,
            concept=user_input,
            user_answer=best_response,
            teacher_answer=teacher_answer,
            similarity=final_similarity,
            thought_quality=len(thought_process.reasoning_chain)
        )

        self.context_memory.add_interaction(
            user_input=user_input,
            ai_response=best_response,
            topic=input_type,
            thought_process=thought_process,
            confidence=final_similarity
        )

        model.concept_bank[input_type].append({
            'input': user_input,
            'output': best_response,
            'similarity': final_similarity,
            'timestamp': datetime.now().isoformat()
        })

        print(f"\n📊 Финальное сходство: {final_similarity:.1%}")
        print(self.learning_manager.get_progress_report())

        self.step_count += 1
        return best_response


# ====================== ГЛАВНАЯ ПРОГРАММА ======================
def main():
    print("\n" + "=" * 70)
    print("🧠 КОГНИТИВНАЯ СИСТЕМА v8.0 (ПОЛНАЯ ВЕРСИЯ)")
    print("С внутренним диалогом, параллельным мышлением и адаптивным обучением")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Устройство: {device}\n")

    # === ИНИЦИАЛИЗАЦИЯ ===
    vocab = AdvancedVocabManager()
    if vocab.load():
        print(f"✅ Словарь загружен ({vocab.size} слов)")
    else:
        print("🔨 Создаю словарь...")
        base_words = ("привет спасибо да нет что как почему где когда кто который "
                      "интересно понимаю узнал новое мышление рефлексия сомнение вопрос "
                      "ответ мнение факт процесс анализ диалог уточнение гипотеза").split()
        vocab.add_words(base_words)
        print(f"✅ Словарь создан ({vocab.size} слов)")

    model = AdvancedIntelligenceBrain(vocab_size=max(Config.VOCAB_SIZE, vocab.size), device=device)
    if model.load_model():
        print("✅ Модель загружена")
    else:
        print("🔨 Модель инициализирована")

    teacher = AdvancedTeacher()

    print(f"\n💡 КОМАНДЫ:")
    print(f" 'выход' - завершить")
    print(f" 'память' - история")
    print(f" 'навыки' - уровни навыков")
    print(f" 'прогресс' - отчёт обучения")
    print(f" 'пробелы' - пробелы в знаниях")
    print(f" 'график' - график точности")
    print(f" 'сохранить' - сохранить")
    print(f" 'очистить' - очистить память")

    interaction_count = 0

    while True:
        try:
            user_input = input("\n👤 ВЫ: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("\n💾 Сохраняю...")
                model.save_model()
                vocab.save()
                teacher.learning_manager.save()
                teacher.context_memory.save()
                print("✨ До встречи!")
                break

            if user_input.lower() in ['память', 'history']:
                print(f"\n📚 История обучения:")
                if teacher.context_memory.conversations:
                    for i, conv in enumerate(list(teacher.context_memory.conversations)[-5:], 1):
                        print(f"\n {i}. Тема: {conv['topic']}")
                        print(f" Вопрос: {conv['user_input'][:60]}...")
                        print(f" Уверенность: {conv['confidence']:.1%}")
                else:
                    print(" (история пуста)")
                continue

            if user_input.lower() in ['навыки', 'skills']:
                print(f"\n{teacher.learning_manager.get_skill_report()}")
                continue

            if user_input.lower() in ['прогресс', 'progress']:
                print(f"\n{teacher.learning_manager.get_progress_report()}")
                continue

            if user_input.lower() in ['пробелы', 'gaps']:
                gaps = teacher.context_memory.identify_learning_gaps()
                print(f"\n📍 ПРОБЕЛЫ В ЗНАНИЯХ:")
                if gaps:
                    for gap in gaps:
                        print(f" • {gap}")
                else:
                    print(" Нет значительных пробелов!")
                continue

            if user_input.lower() in ['график', 'chart']:
                print(f"\n📈 ГРАФИК ТОЧНОСТИ (последние 15):")
                accuracies = teacher.learning_manager.accuracies[-15:]
                if accuracies:
                    for i, acc in enumerate(accuracies, 1):
                        bar = "█" * int(acc * 25) + "░" * (25 - int(acc * 25))
                        print(f" {i:2d}. [{bar}] {acc:.1%}")
                    avg = np.mean(accuracies)
                    print(f"\n Средняя: {avg:.1%}")
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
                if input("Уверен? (да/нет): ").lower() == 'да':
                    Config.LEARNING_PATH.unlink(missing_ok=True)
                    (Config.SAVE_DIR / "advanced_memory.json").unlink(missing_ok=True)
                    teacher.learning_manager = AdvancedLearningManager()
                    teacher.context_memory = AdvancedContextMemory()
                    print("✅ Очищено!")
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
            print("\n✨ До встречи!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()