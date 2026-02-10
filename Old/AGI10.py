# coding: utf-8
"""
AGI_MultiMind_Autonomous_v11.py
Множественные копии модели формируют коллективное сознание
Qwen только для знаний, мышление полностью автономное
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
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST_MODEL = True
except:
    _HAS_ST_MODEL = False


# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    SAVE_DIR = Path("./cognitive_multimind_v11")
    MODEL_PATH = SAVE_DIR / "model.pt"
    VOCAB_PATH = SAVE_DIR / "vocab.pkl"
    KNOWLEDGE_PATH = SAVE_DIR / "knowledge.json"
    MEMORY_PATH = SAVE_DIR / "memory.json"

    NUM_MINDS = 5  # Количество параллельных "умов"

    VOCAB_SIZE = 20000
    EMB_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    MAX_SEQ_LEN = 64
    LEARNING_RATE = 1e-3

    # Параметры автономного мышления
    THINKING_ITERATIONS = 3  # Сколько раз "подумать"
    CONSENSUS_THRESHOLD = 0.6  # Порог согласия между умами
    CONFIDENCE_TO_ANSWER = 0.65  # Уверенность для самостоятельного ответа

    QWEN_API = "http://localhost:1234/v1/chat/completions"


Config.SAVE_DIR.mkdir(exist_ok=True)


# ====================== УТИЛИТЫ ======================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return "Хорошо."
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text or "Хорошо."


def clean_for_tokenize(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text.lower()).strip()


# ====================== БАЗА ЗНАНИЙ ======================
@dataclass
class KnowledgeEntry:
    """Запись знания"""
    question: str
    answer: str
    context: str
    confidence: float
    timestamp: str
    source: str  # 'self' или 'qwen'


class KnowledgeBase:
    """База накопленных знаний"""

    def __init__(self):
        self.entries: List[KnowledgeEntry] = []
        self.index = defaultdict(list)  # слово -> список индексов
        self.load()

    def add(self, question: str, answer: str, context: str = "",
            confidence: float = 1.0, source: str = "qwen"):
        """Добавляет знание"""
        entry = KnowledgeEntry(
            question=question,
            answer=answer,
            context=context,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            source=source
        )

        self.entries.append(entry)

        # Индексируем по словам
        words = set(clean_for_tokenize(question).split())
        for word in words:
            self.index[word].append(len(self.entries) - 1)

        self.save()

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeEntry]:
        """Ищет релевантные знания"""
        words = set(clean_for_tokenize(query).split())

        # Считаем релевантность
        scores = Counter()
        for word in words:
            if word in self.index:
                for idx in self.index[word]:
                    scores[idx] += 1

        # Берём топ-k
        top_indices = [idx for idx, _ in scores.most_common(top_k)]
        return [self.entries[idx] for idx in top_indices if idx < len(self.entries)]

    def get_all_knowledge(self) -> str:
        """Возвращает всю базу знаний как текст"""
        if not self.entries:
            return ""

        knowledge_text = []
        for entry in self.entries[-20:]:  # Последние 20
            knowledge_text.append(f"Q: {entry.question}\nA: {entry.answer}")

        return "\n".join(knowledge_text)

    def save(self):
        data = {
            'entries': [
                {
                    'question': e.question,
                    'answer': e.answer,
                    'context': e.context,
                    'confidence': e.confidence,
                    'timestamp': e.timestamp,
                    'source': e.source
                }
                for e in self.entries
            ]
        }
        with open(Config.KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.KNOWLEDGE_PATH.exists():
            try:
                with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for e in data.get('entries', []):
                        entry = KnowledgeEntry(**e)
                        self.entries.append(entry)

                        words = set(clean_for_tokenize(entry.question).split())
                        for word in words:
                            self.index[word].append(len(self.entries) - 1)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки знаний: {e}")


# ====================== УЛУЧШЕННЫЙ СЛОВАРЬ ======================
class ImprovedVocab:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.next_id = 4
        self.max_vocab_size = Config.VOCAB_SIZE
        self.load()

    def add_word(self, word: str) -> int:
        word_clean = word.lower().strip()
        if not word_clean or len(word_clean) < 1:
            return 3
        if word_clean in self.word2idx:
            return self.word2idx[word_clean]
        if self.next_id >= self.max_vocab_size:
            return 3

        self.word2idx[word_clean] = self.next_id
        self.idx2word[self.next_id] = word_clean
        self.next_id += 1
        return self.word2idx[word_clean]

    def add_words_from_text(self, text: str):
        words = clean_for_tokenize(text).split()
        for word in words:
            if len(word) > 1:
                self.add_word(word)

    def encode(self, text: str) -> List[int]:
        words = clean_for_tokenize(text).split()
        ids = [1]
        for word in words[:Config.MAX_SEQ_LEN - 2]:
            word_id = self.word2idx.get(word, 3)
            ids.append(min(word_id, self.max_vocab_size - 1))
        ids.append(2)
        while len(ids) < Config.MAX_SEQ_LEN:
            ids.append(0)
        return ids[:Config.MAX_SEQ_LEN]

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx in [0, 2]:
                break
            if idx == 1:
                continue
            if idx in self.idx2word and idx < self.max_vocab_size:
                word = self.idx2word[idx]
                if word not in ['<pad>', '<start>', '<end>', '<unk>']:
                    words.append(word)

        if not words:
            return "Не знаю."

        unique_words = []
        prev = None
        for w in words[:30]:
            if w != prev:
                unique_words.append(w)
                prev = w

        return ' '.join(unique_words).capitalize() + '.'

    @property
    def size(self):
        return min(len(self.word2idx), self.max_vocab_size)

    def save(self):
        with open(Config.VOCAB_PATH, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'next_id': self.next_id
            }, f)

    def load(self):
        if Config.VOCAB_PATH.exists():
            try:
                with open(Config.VOCAB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.word2idx = data['word2idx']
                    self.idx2word = data['idx2word']
                    self.next_id = data.get('next_id', 4)
            except:
                pass


# ====================== НЕЙРОННАЯ АРХИТЕКТУРА ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MindBrain(nn.Module):
    """Одна копия 'ума' """

    def __init__(self, vocab_size: int, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, Config.EMB_DIM, padding_idx=0)
        self.pos_encoding = PositionalEncoding(Config.EMB_DIM, Config.MAX_SEQ_LEN)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=Config.EMB_DIM,
                nhead=Config.NUM_HEADS,
                dim_feedforward=Config.HIDDEN_SIZE,
                dropout=Config.DROPOUT,
                batch_first=True
            ),
            num_layers=Config.NUM_LAYERS
        )

        self.decoder = nn.Linear(Config.EMB_DIM, vocab_size)
        self.to(self.device)

    def forward(self, input_ids):
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        padding_mask = (input_ids == 0)
        encoder_out = self.encoder(x, src_key_padding_mask=padding_mask)
        logits = self.decoder(encoder_out)
        return logits

    def generate(self, input_ids, max_len: int = 40, temperature: float = 0.9) -> List[int]:
        self.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            output_ids = [1]

            for _ in range(max_len):
                current_seq = torch.tensor([output_ids], device=self.device, dtype=torch.long)
                if current_seq.size(1) < Config.MAX_SEQ_LEN:
                    padding = torch.zeros((1, Config.MAX_SEQ_LEN - current_seq.size(1)),
                                          device=self.device, dtype=torch.long)
                    current_seq = torch.cat([current_seq, padding], dim=1)

                logits = self.forward(input_ids)
                next_logits = logits[0, min(len(output_ids) - 1, Config.MAX_SEQ_LEN - 1), :]
                next_logits = next_logits / temperature
                next_logits[0] = -float('inf')
                next_logits[1] = -float('inf')

                probs = F.softmax(next_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()

                if next_token == 2:
                    break
                output_ids.append(next_token)

        return output_ids


# ====================== КОЛЛЕКТИВНЫЙ РАЗУМ ======================
class CollectiveMind:
    """Множество умов, которые думают вместе"""

    def __init__(self, vocab: ImprovedVocab, device):
        self.vocab = vocab
        self.device = device

        # Создаём несколько копий модели
        self.minds: List[MindBrain] = []
        for i in range(Config.NUM_MINDS):
            mind = MindBrain(vocab.size, device)
            self.minds.append(mind)

        print(f"🧠 Создано {Config.NUM_MINDS} умов для коллективного мышления")

    def think_collectively(self, question: str, context: str = "") -> Tuple[str, float]:
        """Коллективное мышление - все умы думают параллельно"""

        print(f"\n💭 КОЛЛЕКТИВНОЕ МЫШЛЕНИЕ ({Config.NUM_MINDS} умов):")

        # Формируем промпт с контекстом
        if context:
            full_input = f"{context}\n\nВопрос: {question}\nОтвет:"
        else:
            full_input = f"Вопрос: {question}\nОтвет:"

        input_ids = torch.tensor([self.vocab.encode(full_input)], device=self.device)

        # Каждый ум генерирует свой ответ
        answers = []
        for i, mind in enumerate(self.minds):
            try:
                generated = mind.generate(input_ids, max_len=35, temperature=0.7 + i * 0.1)
                answer = self.vocab.decode(generated)
                answers.append(answer)
                print(f"  Ум #{i + 1}: {answer[:60]}...")
            except Exception as e:
                print(f"  ⚠️ Ум #{i + 1}: ошибка - {e}")
                answers.append("Не знаю.")

        # Находим консенсус
        consensus_answer, confidence = self._find_consensus(answers)

        print(f"\n🎯 КОНСЕНСУС: {consensus_answer[:80]}...")
        print(f"📊 Уверенность: {confidence:.1%}")

        return consensus_answer, confidence

    def _find_consensus(self, answers: List[str]) -> Tuple[str, float]:
        """Находит консенсус между ответами"""

        if not answers:
            return "Не знаю.", 0.0

        # Убираем пустые ответы
        valid_answers = [a for a in answers if a and a != "Не знаю."]

        if not valid_answers:
            return "Не знаю.", 0.0

        # Считаем схожесть между ответами
        similarities = []
        for i in range(len(valid_answers)):
            for j in range(i + 1, len(valid_answers)):
                sim = self._simple_similarity(valid_answers[i], valid_answers[j])
                similarities.append(sim)

        # Средняя схожесть = уверенность
        confidence = np.mean(similarities) if similarities else 0.0

        # Если высокая согласованность, берём самый длинный ответ
        if confidence > Config.CONSENSUS_THRESHOLD:
            best_answer = max(valid_answers, key=len)
        else:
            # Иначе берём самый частый паттерн слов
            word_counter = Counter()
            for answer in valid_answers:
                words = clean_for_tokenize(answer).split()
                word_counter.update(words)

            # Формируем ответ из самых частых слов
            common_words = [w for w, c in word_counter.most_common(10)]
            best_answer = ' '.join(common_words).capitalize() + '.'

        return best_answer, confidence

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Простая метрика схожести"""
        words1 = set(clean_for_tokenize(text1).split())
        words2 = set(clean_for_tokenize(text2).split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def train_all(self, input_ids, target_ids, optimizer):
        """Обучает все умы одновременно"""
        total_loss = 0.0

        for mind in self.minds:
            mind.train()
            logits = mind(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab.size),
                target_ids.view(-1),
                ignore_index=0
            )
            loss.backward()
            total_loss += loss.item()

        return total_loss / len(self.minds)

    def save(self, path: Path):
        """Сохраняет все умы"""
        for i, mind in enumerate(self.minds):
            mind_path = path.parent / f"{path.stem}_mind{i}.pt"
            torch.save({
                'state_dict': mind.state_dict(),
                'vocab_size': mind.vocab_size
            }, mind_path)

    def load(self, path: Path):
        """Загружает все умы"""
        loaded = 0
        for i, mind in enumerate(self.minds):
            mind_path = path.parent / f"{path.stem}_mind{i}.pt"
            if mind_path.exists():
                try:
                    checkpoint = torch.load(mind_path, map_location=self.device)
                    mind.load_state_dict(checkpoint['state_dict'])
                    loaded += 1
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки ума #{i}: {e}")

        return loaded == len(self.minds)


# ====================== АВТОНОМНАЯ КОГНИТИВНАЯ СИСТЕМА ======================
class AutonomousCognitiveSystem:
    """Система с автономным мышлением"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📊 Устройство: {self.device}")

        # Инициализация
        self.vocab = ImprovedVocab()
        self.knowledge = KnowledgeBase()

        # Базовые слова
        base_words = ("привет спасибо да нет что как почему где когда кто который "
                      "интересно понимаю узнал новое хорошо думаю знаю считаю полагаю "
                      "мне кажется возможно вероятно").split()
        for word in base_words:
            self.vocab.add_word(word)

        print(f"📚 Словарь: {self.vocab.size} слов")

        # Коллективный разум
        self.collective_mind = CollectiveMind(self.vocab, self.device)

        if self.collective_mind.load(Config.MODEL_PATH):
            print("✅ Умы загружены")
        else:
            print("🆕 Новые умы созданы")

        self.vocab.save()

    def autonomous_answer(self, question: str) -> Tuple[str, bool]:
        """Автономный ответ - думает сам, без помощи Qwen"""

        print("\n" + "=" * 70)
        print(f"👤 ВОПРОС: {question}")
        print("=" * 70)

        # 1. Ищем в базе знаний
        print("\n🔍 Ищу в памяти...")
        relevant_knowledge = self.knowledge.search(question, top_k=3)

        context = ""
        if relevant_knowledge:
            print(f"✅ Найдено {len(relevant_knowledge)} релевантных знаний")
            for i, entry in enumerate(relevant_knowledge, 1):
                print(f"  {i}. {entry.question[:50]}... (уверенность: {entry.confidence:.1%})")
                context += f"{entry.question} -> {entry.answer}\n"
        else:
            print("❌ Ничего не найдено в памяти")

        # 2. Думаем коллективно несколько раз
        best_answer = None
        best_confidence = 0.0

        for iteration in range(Config.THINKING_ITERATIONS):
            print(f"\n🤔 Итерация мышления #{iteration + 1}:")

            answer, confidence = self.collective_mind.think_collectively(question, context)

            if confidence > best_confidence:
                best_answer = answer
                best_confidence = confidence

            # Если достигли высокой уверенности, можем остановиться
            if confidence >= Config.CONFIDENCE_TO_ANSWER:
                print(f"✅ Достигнута уверенность {confidence:.1%} - останавливаемся")
                break

        # 3. Решаем, можем ли ответить сами
        can_answer_autonomously = best_confidence >= Config.CONFIDENCE_TO_ANSWER

        if can_answer_autonomously:
            print(f"\n✅ МОГУ ОТВЕТИТЬ САМОСТОЯТЕЛЬНО (уверенность: {best_confidence:.1%})")
        else:
            print(f"\n❓ НЕ УВЕРЕН (уверенность: {best_confidence:.1%}) - нужна помощь Qwen")

        return best_answer, can_answer_autonomously

    def learn_from_qwen(self, question: str):
        """Получает знания от Qwen и учится"""

        print("\n👨‍🏫 Обращаюсь к Qwen за знаниями...")

        try:
            resp = requests.post(Config.QWEN_API, json={
                "messages": [
                    {"role": "user", "content": f"{question}\n\nДай короткий, точный ответ (2-3 предложения)."}],
                "max_tokens": 150,
                "temperature": 0.7
            }, timeout=20)

            if resp.status_code == 200:
                teacher_answer = clean_text(resp.json()['choices'][0]['message']['content'])
                print(f"👨‍🏫 QWEN: {teacher_answer}")

                # Добавляем в базу знаний
                self.knowledge.add(question, teacher_answer, source='qwen', confidence=1.0)

                # Обучаем все умы на этом знании
                self._train_minds(question, teacher_answer)

                return teacher_answer
        except Exception as e:
            print(f"⚠️ Ошибка Qwen: {e}")

        return "Не удалось получить ответ."

    def _train_minds(self, question: str, answer: str):
        """Обучает все умы на новом знании"""

        print("\n🔄 Обучаю все умы...")

        # Добавляем слова в словарь
        self.vocab.add_words_from_text(question)
        self.vocab.add_words_from_text(answer)

        # Обновляем размер словаря в умах если нужно
        if self.collective_mind.minds[0].vocab_size < self.vocab.size:
            print(f"📚 Словарь вырос: {self.collective_mind.minds[0].vocab_size} → {self.vocab.size}")
            for mind in self.collective_mind.minds:
                old_embedding = mind.embedding.weight.data
                mind.embedding = nn.Embedding(self.vocab.size, Config.EMB_DIM, padding_idx=0)
                mind.embedding.weight.data[:old_embedding.size(0)] = old_embedding
                mind.decoder = nn.Linear(Config.EMB_DIM, self.vocab.size)
                mind.vocab_size = self.vocab.size
                mind.to(self.device)

        # Кодируем
        input_ids = torch.tensor([self.vocab.encode(question)], device=self.device)
        target_ids = torch.tensor([self.vocab.encode(answer)], device=self.device)

        # Создаём единый оптимизатор для всех умов
        all_params = []
        for mind in self.collective_mind.minds:
            all_params.extend(mind.parameters())

        optimizer = torch.optim.AdamW(all_params, lr=Config.LEARNING_RATE)

        # Обучаем
        for epoch in range(10):
            optimizer.zero_grad()
            avg_loss = self.collective_mind.train_all(input_ids, target_ids, optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            if epoch % 3 == 0:
                print(f"  Эпоха {epoch}: loss={avg_loss:.3f}")

        # Сохраняем
        self.collective_mind.save(Config.MODEL_PATH)
        self.vocab.save()

        print("✅ Обучение завершено")

    def process(self, question: str):
        """Полный цикл обработки вопроса"""

        # 1. Пытаемся ответить автономно
        my_answer, can_answer = self.autonomous_answer(question)

        if can_answer:
            # Отвечаем сами
            print(f"\n💡 МОЙ АВТОНОМНЫЙ ОТВЕТ: {my_answer}")

            # Сохраняем как собственное знание
            self.knowledge.add(question, my_answer, source='self', confidence=0.8)

            return my_answer
        else:
            # Нужна помощь Qwen
            print(f"\n⚠️ Мой ответ: {my_answer}")
            print("❓ Недостаточно уверен - учусь у Qwen...")

            teacher_answer = self.learn_from_qwen(question)

            print(f"\n💡 ИТОГОВЫЙ ОТВЕТ: {teacher_answer}")

            return teacher_answer


# ====================== ИНТЕРФЕЙС ======================
def main():
    print("\n" + "=" * 70)
    print("🧠 АВТОНОМНАЯ КОГНИТИВНАЯ СИСТЕМА v11.0")
    print("Множественные умы • Коллективное сознание • Автономное мышление")
    print("=" * 70)

    try:
        system = AutonomousCognitiveSystem()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        traceback.print_exc()
        return

    print(f"\n💡 КОМАНДЫ:")
    print(f" 'выход' - завершить")
    print(f" 'знания' - показать базу знаний")
    print(f" 'статистика' - статистика обучения")

    while True:
        try:
            user_input = input("\n👤 ВЫ: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("\n✨ До встречи!")
                break

            if user_input.lower() == 'знания':
                entries = system.knowledge.entries[-10:]
                if entries:
                    print("\n📚 БАЗА ЗНАНИЙ (последние 10):")
                    for i, entry in enumerate(entries, 1):
                        source_icon = "🤖" if entry.source == "self" else "👨‍🏫"
                        print(f"\n{i}. {source_icon} {entry.question}")
                        print(f"   → {entry.answer[:80]}...")
                        print(f"   📊 Уверенность: {entry.confidence:.1%} | Источник: {entry.source}")
                else:
                    print("\n📚 База знаний пуста")
                continue

            if user_input.lower() == 'статистика':
                total = len(system.knowledge.entries)
                self_learned = sum(1 for e in system.knowledge.entries if e.source == 'self')
                qwen_learned = sum(1 for e in system.knowledge.entries if e.source == 'qwen')

                print("\n📊 СТАТИСТИКА ОБУЧЕНИЯ:")
                print(f"  Всего знаний: {total}")
                print(f"  🤖 Самостоятельно: {self_learned} ({self_learned / total * 100 if total else 0:.1f}%)")
                print(f"  👨‍🏫 От Qwen: {qwen_learned} ({qwen_learned / total * 100 if total else 0:.1f}%)")
                print(f"  📚 Размер словаря: {system.vocab.size} слов")
                print(f"  🧠 Количество умов: {Config.NUM_MINDS}")
                continue

            # Основная обработка вопроса
            system.process(user_input)

        except KeyboardInterrupt:
            print("\n✨ Прерывание")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()