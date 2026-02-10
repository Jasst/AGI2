# coding: utf-8
"""
AGI_CognitiveReasoning_v10_FIXED.py
Исправлена критическая ошибка с индексацией + оптимизации
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
    SAVE_DIR = Path("./cognitive_model_data_v10")
    MODEL_PATH = SAVE_DIR / "model.pt"
    VOCAB_PATH = SAVE_DIR / "vocab.pkl"
    LEARNING_PATH = SAVE_DIR / "learning.json"

    VOCAB_SIZE = 15000  # Уменьшено для стабильности
    EMB_DIM = 256  # Уменьшено для скорости
    HIDDEN_SIZE = 512  # Уменьшено
    NUM_LAYERS = 3  # Уменьшено
    NUM_HEADS = 8
    DROPOUT = 0.1
    MAX_SEQ_LEN = 64  # Уменьшено для скорости
    LEARNING_RATE = 1e-3
    MAX_ATTEMPTS = 15  # Уменьшено
    CONFIDENCE_THRESHOLD = 0.65
    UNCERTAINTY_THRESHOLD = 0.45
    QWEN_API = "http://localhost:1234/v1/chat/completions"


Config.SAVE_DIR.mkdir(exist_ok=True)


# ====================== УТИЛИТЫ ======================
def clean_text(text: str) -> str:
    """Очищает текст от артефактов"""
    if not isinstance(text, str):
        return "Хорошо."

    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'>\s*', '', text)
    text = re.sub(r'\r\n|\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    if len(words) > 100:
        text = ' '.join(words[:100])

    if text and not text.endswith(('.', '!', '?', '😊')):
        text += '.'

    return text or "Хорошо."


def clean_for_tokenize(text: str) -> str:
    """Очищает для токенизации"""
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text.lower()).strip()


def detect_input_type(user_input: str) -> str:
    """Определяет тип вопроса"""
    s = user_input.lower()
    if any(w in s for w in ['привет', 'здравствуй', 'добрый', 'как дела', 'спасибо']):
        return "GREETING"
    elif any(w in s for w in ['что', 'кто', 'где', 'какой', 'определение', 'расскажи']):
        return "DEFINITION"
    elif any(w in s for w in ['почему', 'зачем', 'отчего', 'причина']):
        return "REASON"
    elif any(w in s for w in ['как сделать', 'инструкция', 'шаг', 'алгоритм']):
        return "PROCESS"
    else:
        return "GENERAL"


# ====================== УМНЫЙ ДИАЛОГ С УЧИТЕЛЕМ ======================
class TeacherDialog:
    """Система интеллектуального диалога с учителем"""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.dialog_history = deque(maxlen=10)

    def ask_smart_question(self, user_input: str, input_type: str) -> str:
        """Умный вопрос к учителю"""

        if input_type == "GREETING":
            prompt = f"Пользователь поздоровался: '{user_input}'. Дай тёплый, короткий ответ (одна строка)."
        elif input_type == "DEFINITION":
            prompt = f"Пользователь спрашивает: '{user_input}'\nЭто запрос на определение. Дай чёткое определение (2-3 предложения max)."
        elif input_type == "REASON":
            prompt = f"Пользователь спрашивает: '{user_input}'\nОбъясни причину кратко (2-3 предложения max)."
        elif input_type == "PROCESS":
            prompt = f"Пользователь спрашивает: '{user_input}'\nДай пошаговую инструкцию (3-4 шага max)."
        else:
            prompt = f"Ответь на вопрос: {user_input}\nОтвет должен быть кратким (2-3 предложения)."

        try:
            resp = requests.post(self.api_url, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.7
            }, timeout=20)

            if resp.status_code == 200:
                answer = resp.json()['choices'][0]['message']['content']
                return clean_text(answer)
        except Exception as e:
            print(f"⚠️ API ошибка: {e}")

        return "Я не знаю ответа на этот вопрос."


# ====================== ПРОДВИНУТЫЙ ПРОЦЕСС МЫШЛЕНИЯ ======================
class ThinkingProcess:
    """Система мышления с рефлексией"""

    def __init__(self):
        self.observations = []
        self.confidence = 0.5

    def observe(self, observation: str, confidence: float = 0.5):
        self.observations.append({'text': observation, 'confidence': confidence})
        self._update_confidence()

    def _update_confidence(self):
        if self.observations:
            self.confidence = np.mean([o['confidence'] for o in self.observations])

    def __str__(self) -> str:
        result = "🧠 ДУМАЮ:\n"
        if self.observations:
            for obs in self.observations[-2:]:
                result += f"  • {obs['text'][:50]}... ({obs['confidence']:.0%})\n"
        result += f"📊 Уверенность: {self.confidence:.0%}\n"
        return result


# ====================== УЛУЧШЕННЫЙ СЛОВАРЬ С ЗАЩИТОЙ ОТ ОШИБОК ======================
class ImprovedVocab:
    """Словарь с правильной индексацией и проверками"""

    def __init__(self):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.word_freq = Counter()
        self.next_id = 4
        self.max_vocab_size = Config.VOCAB_SIZE
        self.load()

    def add_word(self, word: str) -> int:
        """Добавляет слово с проверкой границ"""
        word_clean = word.lower().strip()

        if not word_clean or len(word_clean) < 1:
            return self.word2idx['<unk>']

        if word_clean in self.word2idx:
            self.word_freq[word_clean] += 1
            return self.word2idx[word_clean]

        # КРИТИЧЕСКИ ВАЖНО: проверяем границы
        if self.next_id >= self.max_vocab_size:
            return self.word2idx['<unk>']

        self.word2idx[word_clean] = self.next_id
        self.idx2word[self.next_id] = word_clean
        self.word_freq[word_clean] = 1
        self.next_id += 1

        return self.word2idx[word_clean]

    def add_words_from_text(self, text: str):
        """Добавляет слова из текста"""
        words = clean_for_tokenize(text).split()
        for word in words:
            if len(word) > 1:
                self.add_word(word)

    def encode(self, text: str) -> List[int]:
        """Кодирует текст в ID с проверкой границ"""
        words = clean_for_tokenize(text).split()
        ids = [1]  # start token

        for word in words[:Config.MAX_SEQ_LEN - 2]:
            word_id = self.word2idx.get(word, 3)  # 3 = <unk>

            # КРИТИЧЕСКАЯ ПРОВЕРКА: индекс в пределах словаря
            if word_id < self.max_vocab_size:
                ids.append(word_id)
            else:
                ids.append(3)  # fallback to <unk>

        ids.append(2)  # end token

        # Паддинг
        while len(ids) < Config.MAX_SEQ_LEN:
            ids.append(0)  # pad token

        return ids[:Config.MAX_SEQ_LEN]

    def decode(self, ids: List[int]) -> str:
        """Декодирует ID в текст"""
        words = []
        for idx in ids:
            if idx == 0 or idx == 2:  # pad, end
                break
            if idx == 1:  # start
                continue

            # КРИТИЧЕСКАЯ ПРОВЕРКА
            if idx in self.idx2word and idx < self.max_vocab_size:
                word = self.idx2word[idx]
                if word not in ['<pad>', '<start>', '<end>', '<unk>']:
                    words.append(word)

        if not words:
            return "Хорошо."

        # Убираем дубликаты подряд
        unique_words = []
        prev = None
        for w in words[:25]:
            if w != prev:
                unique_words.append(w)
                prev = w

        return ' '.join(unique_words).capitalize() + '.'

    @property
    def size(self):
        """Реальный размер словаря"""
        return min(len(self.word2idx), self.max_vocab_size)

    def save(self):
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'next_id': self.next_id,
            'max_vocab_size': self.max_vocab_size
        }
        with open(Config.VOCAB_PATH, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        if Config.VOCAB_PATH.exists():
            try:
                with open(Config.VOCAB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.word2idx = data['word2idx']
                    self.idx2word = data['idx2word']
                    self.word_freq = Counter(data.get('word_freq', {}))
                    self.next_id = data.get('next_id', 4)
                    self.max_vocab_size = data.get('max_vocab_size', Config.VOCAB_SIZE)
                    return True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки словаря: {e}")
        return False


# ====================== УЛУЧШЕННАЯ НЕЙРОННАЯ АРХИТЕКТУРА ======================
class PositionalEncoding(nn.Module):
    """Позиционное кодирование с защитой"""

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
        seq_len = x.size(1)
        # Защита от выхода за границы
        if seq_len > self.pe.size(1):
            seq_len = self.pe.size(1)
        return x + self.pe[:, :seq_len, :]


class SimpleBrain(nn.Module):
    """Упрощённая и стабильная архитектура"""

    def __init__(self, vocab_size: int, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size

        # КРИТИЧЕСКИ ВАЖНО: embedding с правильным размером
        self.embedding = nn.Embedding(vocab_size, Config.EMB_DIM, padding_idx=0)
        self.pos_encoding = PositionalEncoding(Config.EMB_DIM, Config.MAX_SEQ_LEN)

        # Упрощённая архитектура
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

    def forward(self, input_ids, target_ids=None):
        """Forward pass с проверкой индексов"""
        # КРИТИЧЕСКАЯ ПРОВЕРКА
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)

        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # Создаём padding mask
        padding_mask = (input_ids == 0)

        encoder_out = self.encoder(x, src_key_padding_mask=padding_mask)
        logits = self.decoder(encoder_out)

        return logits

    def generate(self, input_ids, max_len: int = 40, temperature: float = 0.9):
        """Генерация с защитой"""
        self.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            batch_size = input_ids.size(0)

            output_ids = [1]  # start token

            for _ in range(max_len):
                # Создаём временную последовательность
                current_seq = torch.tensor([output_ids], device=self.device, dtype=torch.long)

                # Паддим до MAX_SEQ_LEN
                if current_seq.size(1) < Config.MAX_SEQ_LEN:
                    padding = torch.zeros((1, Config.MAX_SEQ_LEN - current_seq.size(1)),
                                          device=self.device, dtype=torch.long)
                    current_seq = torch.cat([current_seq, padding], dim=1)

                logits = self.forward(input_ids, current_seq)
                next_logits = logits[0, len(output_ids) - 1, :]

                # Применяем температуру
                next_logits = next_logits / temperature

                # Маскируем служебные токены
                next_logits[0] = -float('inf')  # pad
                next_logits[1] = -float('inf')  # start

                probs = F.softmax(next_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()

                if next_token == 2:  # end token
                    break

                output_ids.append(next_token)

        return output_ids

    def save(self):
        torch.save({
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size
        }, Config.MODEL_PATH)

    def load(self):
        if Config.MODEL_PATH.exists():
            try:
                checkpoint = torch.load(Config.MODEL_PATH, map_location=self.device)
                self.load_state_dict(checkpoint['state_dict'])
                return True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
        return False


# ====================== МЕНЕДЖЕР ОБУЧЕНИЯ ======================
class LearningManager:
    def __init__(self):
        self.history = []
        self.skill_levels = defaultdict(float)
        self.accuracies = []
        self.load()

    def record(self, topic: str, similarity: float):
        self.history.append({
            'topic': topic,
            'similarity': similarity,
            'time': datetime.now().isoformat()
        })
        self.accuracies.append(similarity)

        if similarity > 0.5:
            self.skill_levels[topic] = min(1.0, self.skill_levels[topic] + 0.05)

        self.save()

    def get_report(self) -> str:
        if not self.accuracies:
            return "Обучение не началось"

        recent = self.accuracies[-10:]
        avg = np.mean(recent)

        report = f"📊 ОБУЧЕНИЕ:\n"
        report += f" Последние 10: {avg:.1%}\n"
        report += f" Всего: {len(self.history)}\n"

        if self.skill_levels:
            best_topic = max(self.skill_levels, key=self.skill_levels.get)
            report += f" Лучшая тема: {best_topic} ({self.skill_levels[best_topic]:.1%})\n"

        return report

    def save(self):
        data = {
            'history': self.history,
            'skill_levels': dict(self.skill_levels),
            'accuracies': self.accuracies
        }
        with open(Config.LEARNING_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.LEARNING_PATH.exists():
            try:
                with open(Config.LEARNING_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.skill_levels = defaultdict(float, data.get('skill_levels', {}))
                    self.accuracies = data.get('accuracies', [])
            except:
                pass


# ====================== СЕМАНТИЧЕСКИЙ АНАЛИЗ ======================
class SemanticSimilarity:
    def __init__(self):
        self.model = None
        if _HAS_ST_MODEL:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                pass

    def similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        if self.model:
            try:
                emb = self.model.encode([text1, text2], normalize_embeddings=True)
                return float(np.dot(emb[0], emb[1]))
            except:
                pass

        # Fallback: Jaccard
        words1 = set(clean_for_tokenize(text1).split())
        words2 = set(clean_for_tokenize(text2).split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# ====================== ГЛАВНАЯ СИСТЕМА ======================
class CognitiveSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📊 Устройство: {self.device}")

        self.vocab = ImprovedVocab()

        # Инициализируем базовые слова
        base_words = "привет спасибо да нет что как почему где когда кто который интересно понимаю узнал новое хорошо".split()
        for word in base_words:
            self.vocab.add_word(word)

        print(f"📚 Словарь: {self.vocab.size} слов (max: {Config.VOCAB_SIZE})")

        # КРИТИЧЕСКИ ВАЖНО: создаём модель с правильным размером
        self.brain = SimpleBrain(self.vocab.size, self.device)

        self.teacher = TeacherDialog(Config.QWEN_API)
        self.similarity = SemanticSimilarity()
        self.learning_manager = LearningManager()

        if self.brain.load():
            print("✅ Модель загружена")
        else:
            print("🆕 Новая модель создана")

        self.vocab.save()

    def learn(self, user_input: str):
        """Полный цикл обучения с защитой от ошибок"""

        input_type = detect_input_type(user_input)

        print(f"\n👤 ВЫ: {user_input}")
        print(f"📋 Тип: {input_type}")

        thinking = ThinkingProcess()
        thinking.observe(f"Получил вопрос: {user_input[:40]}...", 0.7)
        print(thinking)

        # Получаем ответ учителя
        print("\n👨‍🏫 Спрашиваю учителя...")
        teacher_answer = self.teacher.ask_smart_question(user_input, input_type)
        print(f"👨‍🏫 ОТВЕТ: {teacher_answer}")

        # Добавляем слова в словарь ДО создания тензоров
        self.vocab.add_words_from_text(user_input)
        self.vocab.add_words_from_text(teacher_answer)

        # Обновляем модель если словарь вырос
        if self.brain.vocab_size < self.vocab.size:
            print(f"📚 Словарь вырос: {self.brain.vocab_size} → {self.vocab.size}")
            old_embedding = self.brain.embedding.weight.data
            self.brain.embedding = nn.Embedding(self.vocab.size, Config.EMB_DIM, padding_idx=0)
            self.brain.embedding.weight.data[:old_embedding.size(0)] = old_embedding
            self.brain.decoder = nn.Linear(Config.EMB_DIM, self.vocab.size)
            self.brain.vocab_size = self.vocab.size
            self.brain.to(self.device)

        # Кодируем с проверкой
        try:
            input_ids = torch.tensor([self.vocab.encode(user_input)], device=self.device)
            target_ids = torch.tensor([self.vocab.encode(teacher_answer)], device=self.device)
        except Exception as e:
            print(f"❌ Ошибка кодирования: {e}")
            return teacher_answer

        # Обучение
        print("\n🔄 ОБУЧАЮ МОДЕЛЬ...")
        self.brain.train()
        optimizer = torch.optim.AdamW(self.brain.parameters(), lr=Config.LEARNING_RATE)

        best_loss = float('inf')
        best_answer = teacher_answer

        for epoch in range(Config.MAX_ATTEMPTS):
            try:
                optimizer.zero_grad()

                logits = self.brain(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab.size),
                    target_ids.view(-1),
                    ignore_index=0
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
                optimizer.step()

                if epoch % 3 == 0:
                    self.brain.eval()
                    with torch.no_grad():
                        generated = self.brain.generate(input_ids, max_len=40, temperature=0.8)
                        my_answer = self.vocab.decode(generated)
                        similarity = self.similarity.similarity(teacher_answer, my_answer)

                        status = "✅" if similarity > 0.5 else "❌"
                        print(f" {epoch:2d}. loss={loss.item():.3f}, схож={similarity:.1%} {status}")

                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            best_answer = my_answer

                    self.brain.train()

                if loss.item() < 0.5:
                    break

            except Exception as e:
                print(f"❌ Ошибка на эпохе {epoch}: {e}")
                break

        # Финальная генерация
        self.brain.eval()
        try:
            with torch.no_grad():
                final_ids = self.brain.generate(input_ids, max_len=40, temperature=0.7)
                final_answer = self.vocab.decode(final_ids)
                final_similarity = self.similarity.similarity(teacher_answer, final_answer)
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            final_answer = best_answer
            final_similarity = 0.5

        print(f"\n💡 МОЙ ОТВЕТ: {final_answer}")
        print(f"📊 Сходство с учителем: {final_similarity:.1%}")

        self.learning_manager.record(input_type, final_similarity)
        print(self.learning_manager.get_report())

        self.brain.save()
        self.vocab.save()

        return final_answer


# ====================== ИНТЕРФЕЙС ======================
def main():
    print("\n" + "=" * 70)
    print("🧠 КОГНИТИВНАЯ СИСТЕМА v10.0 (ИСПРАВЛЕННАЯ)")
    print("Исправлена критическая ошибка индексации + оптимизации")
    print("=" * 70)

    try:
        system = CognitiveSystem()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        traceback.print_exc()
        return

    print(f"\n💡 КОМАНДЫ:")
    print(f" 'выход' - завершить")
    print(f" 'статус' - статус обучения")
    print(f" 'навыки' - уровни по темам")

    while True:
        try:
            user_input = input("\n👤 ВЫ: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("\n✨ До встречи!")
                break

            if user_input.lower() == 'статус':
                print(system.learning_manager.get_report())
                continue

            if user_input.lower() == 'навыки':
                if system.learning_manager.skill_levels:
                    print("\n📈 НАВЫКИ:")
                    for topic, level in sorted(system.learning_manager.skill_levels.items(),
                                               key=lambda x: x[1], reverse=True):
                        bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
                        print(f" {topic:15s} [{bar}] {level:.1%}")
                continue

            # Основной цикл обучения
            system.learn(user_input)

        except KeyboardInterrupt:
            print("\n✨ Прерывание")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()