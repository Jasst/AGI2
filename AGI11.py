# coding: utf-8
"""
AGI_Hybrid_MultiMind_v14.py
ГИБРИД: Множественные нейросети + Семантическая память + Коллективное мышление
Лучшее из обоих миров!
"""

import os
import re
import json
import pickle
import traceback
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass

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
    SAVE_DIR = Path("./cognitive_hybrid_v14")
    KNOWLEDGE_PATH = SAVE_DIR / "knowledge.json"
    VOCAB_PATH = SAVE_DIR / "vocab.pkl"
    MODELS_DIR = SAVE_DIR / "models"

    # Архитектура
    NUM_MINDS = 3  # Количество нейросетей
    VOCAB_SIZE = 15000
    EMB_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.1
    MAX_SEQ_LEN = 48
    LEARNING_RATE = 5e-4

    # Мышление
    THINKING_ITERATIONS = 2
    CONFIDENCE_THRESHOLD = 0.70
    SEMANTIC_SIMILARITY_THRESHOLD = 0.75

    QWEN_API = "http://localhost:1234/v1/chat/completions"


Config.SAVE_DIR.mkdir(exist_ok=True)
Config.MODELS_DIR.mkdir(exist_ok=True)


# ====================== УТИЛИТЫ ======================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_for_tokenize(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text.lower()).strip()


# ====================== СЕМАНТИЧЕСКАЯ ПАМЯТЬ ======================
@dataclass
class Memory:
    question: str
    answer: str
    embedding: np.ndarray
    source: str  # 'self' or 'qwen'
    confidence: float
    timestamp: str
    usage_count: int = 0


class SemanticMemory:
    """Семантическая память с embeddings"""

    def __init__(self):
        self.encoder = None
        self.memories: List[Memory] = []

        if _HAS_ST_MODEL:
            try:
                print("📦 Загружаю encoder...")
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Encoder готов")
            except Exception as e:
                print(f"⚠️ Ошибка: {e}")

        self.load()

    def add(self, question: str, answer: str, source: str = "qwen", confidence: float = 1.0):
        """Добавляет память"""

        # Проверяем дубликаты
        for mem in self.memories:
            if mem.question.lower().strip() == question.lower().strip():
                mem.answer = answer
                mem.confidence = max(mem.confidence, confidence)
                mem.timestamp = datetime.now().isoformat()
                self.save()
                return

        # Создаём embedding
        embedding = self._encode(question)
        if embedding is None:
            return

        memory = Memory(
            question=question,
            answer=answer,
            embedding=embedding,
            source=source,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )

        self.memories.append(memory)
        self.save()
        print(f"💾 Сохранено: '{question}' → '{answer[:40]}...'")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Memory, float]]:
        """Ищет похожие воспоминания"""

        if not self.memories or self.encoder is None:
            return []

        query_emb = self._encode(query)
        if query_emb is None:
            return []

        # Считаем схожесть
        results = []
        for mem in self.memories:
            similarity = float(np.dot(query_emb, mem.embedding))
            if similarity > 0.3:
                results.append((mem, similarity))

        # Сортируем
        results.sort(key=lambda x: x[1], reverse=True)

        # Обновляем счётчик
        for mem, _ in results[:top_k]:
            mem.usage_count += 1

        return results[:top_k]

    def _encode(self, text: str):
        """Кодирует текст"""
        if self.encoder is None:
            return None
        try:
            return self.encoder.encode(text, normalize_embeddings=True)
        except:
            return None

    def save(self):
        data = {
            'memories': [
                {
                    'question': m.question,
                    'answer': m.answer,
                    'source': m.source,
                    'confidence': m.confidence,
                    'timestamp': m.timestamp,
                    'usage_count': m.usage_count
                }
                for m in self.memories
            ]
        }
        with open(Config.KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.KNOWLEDGE_PATH.exists():
            try:
                with open(Config.KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for m in data.get('memories', []):
                        emb = self._encode(m['question'])
                        if emb is not None:
                            self.memories.append(Memory(
                                question=m['question'],
                                answer=m['answer'],
                                embedding=emb,
                                source=m['source'],
                                confidence=m['confidence'],
                                timestamp=m['timestamp'],
                                usage_count=m.get('usage_count', 0)
                            ))
                print(f"📚 Загружено {len(self.memories)} воспоминаний")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки: {e}")


# ====================== СЛОВАРЬ ======================
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.next_id = 2
        self.load()

    def add_words(self, text: str):
        words = clean_for_tokenize(text).split()
        for word in words:
            if word and word not in self.word2idx:
                if self.next_id < Config.VOCAB_SIZE:
                    self.word2idx[word] = self.next_id
                    self.idx2word[self.next_id] = word
                    self.next_id += 1

    def encode(self, text: str) -> List[int]:
        words = clean_for_tokenize(text).split()[:Config.MAX_SEQ_LEN]
        ids = [self.word2idx.get(w, 1) for w in words]
        # Паддинг
        while len(ids) < Config.MAX_SEQ_LEN:
            ids.append(0)
        return ids[:Config.MAX_SEQ_LEN]

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx == 0:  # pad
                break
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word != '<pad>' and word != '<unk>':
                    words.append(word)

        if not words:
            return ""

        # Убираем дубликаты подряд
        result = []
        prev = None
        for w in words[:30]:
            if w != prev:
                result.append(w)
                prev = w

        text = ' '.join(result)
        return text.capitalize() if text else ""

    @property
    def size(self):
        return len(self.word2idx)

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
                    self.next_id = data['next_id']
            except:
                pass


# ====================== НЕЙРОННАЯ СЕТЬ ======================
class MindNetwork(nn.Module):
    """Одна нейросеть-разум"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, Config.EMB_DIM, padding_idx=0)

        self.encoder = nn.GRU(
            Config.EMB_DIM,
            Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.decoder = nn.GRU(
            Config.EMB_DIM,
            Config.HIDDEN_SIZE * 2,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0,
            batch_first=True
        )

        self.output_layer = nn.Linear(Config.HIDDEN_SIZE * 2, vocab_size)

    def forward(self, input_ids, target_ids):
        # Encode
        input_emb = self.embedding(input_ids)
        _, hidden = self.encoder(input_emb)

        # Decode
        target_emb = self.embedding(target_ids)
        hidden = self._prepare_decoder_hidden(hidden)

        decoder_out, _ = self.decoder(target_emb, hidden)
        logits = self.output_layer(decoder_out)

        return logits

    def generate(self, input_ids, max_len: int = 30, temperature: float = 0.8):
        """Генерация ответа"""
        self.eval()
        with torch.no_grad():
            # Encode
            input_emb = self.embedding(input_ids)
            _, hidden = self.encoder(input_emb)
            hidden = self._prepare_decoder_hidden(hidden)

            # Decode
            generated = []
            current_input = torch.zeros((1, 1), dtype=torch.long, device=input_ids.device)

            for _ in range(max_len):
                emb = self.embedding(current_input)
                out, hidden = self.decoder(emb, hidden)
                logits = self.output_layer(out[:, -1, :])

                # Температура
                logits = logits / temperature
                logits[0, 0] = -float('inf')  # mask padding

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs[0], 1).item()

                if next_token == 0:  # stop at padding
                    break

                generated.append(next_token)
                current_input = torch.tensor([[next_token]], device=input_ids.device)

        return generated

    def _prepare_decoder_hidden(self, encoder_hidden):
        """Подготавливает hidden state для декодера"""
        # encoder_hidden: [num_layers*2, batch, hidden_size]
        # Объединяем forward и backward
        batch_size = encoder_hidden.size(1)
        hidden = encoder_hidden.view(Config.NUM_LAYERS, 2, batch_size, Config.HIDDEN_SIZE)
        # Конкатенируем forward и backward
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        return hidden.contiguous()


# ====================== КОЛЛЕКТИВНЫЙ РАЗУМ ======================
class CollectiveIntelligence:
    """Множественные нейросети думают вместе"""

    def __init__(self, vocab: Vocabulary, device):
        self.vocab = vocab
        self.device = device
        self.minds: List[MindNetwork] = []

        for i in range(Config.NUM_MINDS):
            mind = MindNetwork(vocab.size).to(device)
            self.minds.append(mind)

        print(f"🧠 Создано {Config.NUM_MINDS} нейросетевых разумов")
        self.load_all()

    def think(self, question: str, context: str = "") -> Tuple[List[str], float]:
        """Коллективное мышление"""

        # Формируем входной промпт
        if context:
            full_input = f"{context}\n{question}"
        else:
            full_input = question

        input_ids = torch.tensor([self.vocab.encode(full_input)], device=self.device)

        # Каждый разум генерирует ответ
        answers = []
        print(f"\n💭 Думают {Config.NUM_MINDS} разума:")

        for i, mind in enumerate(self.minds):
            try:
                temp = 0.75 + i * 0.1
                generated_ids = mind.generate(input_ids, max_len=30, temperature=temp)
                answer = self.vocab.decode(generated_ids)

                if answer and len(answer.split()) >= 2:
                    answers.append(answer)
                    print(f"  Разум #{i + 1}: {answer}")
            except Exception as e:
                print(f"  ⚠️ Разум #{i + 1}: ошибка")

        if not answers:
            return [], 0.0

        # Вычисляем консенсус
        best_answer, confidence = self._find_consensus(answers)

        return answers, confidence

    def _find_consensus(self, answers: List[str]) -> Tuple[str, float]:
        """Находит консенсус"""
        if not answers:
            return "", 0.0

        if len(answers) == 1:
            return answers[0], 0.5

        # Считаем схожесть между всеми парами
        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                sim = self._jaccard_similarity(answers[i], answers[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0.0

        # Берём самый длинный ответ
        best_answer = max(answers, key=lambda x: len(x.split()))

        return best_answer, avg_similarity

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def train_all(self, question: str, answer: str, epochs: int = 10):
        """Обучает все разумы"""

        print("\n🔄 Обучаю все разумы...")

        # Добавляем слова
        self.vocab.add_words(question)
        self.vocab.add_words(answer)

        # Обновляем размер embeddings если нужно
        self._update_embeddings()

        # Подготовка данных
        input_ids = torch.tensor([self.vocab.encode(question)], device=self.device)
        target_ids = torch.tensor([self.vocab.encode(answer)], device=self.device)

        # Обучение
        for mind in self.minds:
            mind.train()
            optimizer = torch.optim.Adam(mind.parameters(), lr=Config.LEARNING_RATE)

            for epoch in range(epochs):
                optimizer.zero_grad()
                logits = mind(input_ids, target_ids)
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab.size),
                    target_ids.view(-1),
                    ignore_index=0
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mind.parameters(), 1.0)
                optimizer.step()

                if epoch % 3 == 0:
                    print(f"  Эпоха {epoch}: loss={loss.item():.3f}")

        self.save_all()
        self.vocab.save()
        print("✅ Обучение завершено")

    def _update_embeddings(self):
        """Обновляет размер embeddings"""
        for mind in self.minds:
            if mind.vocab_size < self.vocab.size:
                old_emb = mind.embedding.weight.data
                mind.embedding = nn.Embedding(self.vocab.size, Config.EMB_DIM, padding_idx=0).to(self.device)
                mind.embedding.weight.data[:old_emb.size(0)] = old_emb

                mind.output_layer = nn.Linear(Config.HIDDEN_SIZE * 2, self.vocab.size).to(self.device)
                mind.vocab_size = self.vocab.size

    def save_all(self):
        for i, mind in enumerate(self.minds):
            path = Config.MODELS_DIR / f"mind_{i}.pt"
            torch.save(mind.state_dict(), path)

    def load_all(self):
        loaded = 0
        for i, mind in enumerate(self.minds):
            path = Config.MODELS_DIR / f"mind_{i}.pt"
            if path.exists():
                try:
                    mind.load_state_dict(torch.load(path, map_location=self.device))
                    loaded += 1
                except:
                    pass
        if loaded > 0:
            print(f"✅ Загружено {loaded} разумов")


# ====================== ГИБРИДНАЯ СИСТЕМА ======================
class HybridCognitiveSystem:
    """Гибрид: Семантика + Нейросети"""

    def __init__(self):
        print(f"\n{'=' * 70}")
        print(f"🧠 ГИБРИДНАЯ КОГНИТИВНАЯ СИСТЕМА v14.0")
        print(f"Семантическая память • Множественные нейросети • Синтез")
        print(f"{'=' * 70}\n")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📊 Устройство: {self.device}")

        # Инициализация компонентов
        self.memory = SemanticMemory()
        self.vocab = Vocabulary()

        # Базовые слова
        base_text = "привет спасибо да нет что как почему где когда кто хорошо понимаю знаю думаю"
        self.vocab.add_words(base_text)

        self.collective = CollectiveIntelligence(self.vocab, self.device)

        if not _HAS_ST_MODEL:
            print("\n⚠️ ВНИМАНИЕ: Установите sentence-transformers для полной функциональности")
            print("   pip install sentence-transformers\n")

    def process(self, question: str):
        """Обрабатывает вопрос"""

        print(f"\n{'=' * 70}")
        print(f"👤 ВОПРОС: {question}")
        print(f"{'=' * 70}")

        # ШАГ 1: Семантический поиск
        print(f"\n🔍 ШАГ 1: Поиск в семантической памяти")
        similar_memories = self.memory.search(question, top_k=3)

        if similar_memories:
            best_mem, best_sim = similar_memories[0]
            print(f"✅ Найдено {len(similar_memories)} похожих воспоминаний")
            print(f"   Лучшее: '{best_mem.question}' (схожесть: {best_sim:.1%})")

            # Если очень похоже - сразу возвращаем
            if best_sim >= Config.SEMANTIC_SIMILARITY_THRESHOLD:
                print(f"\n✅ ТОЧНОЕ СОВПАДЕНИЕ (sim={best_sim:.1%})")
                print(f"💡 ОТВЕТ: {best_mem.answer}")
                best_mem.usage_count += 1
                return best_mem.answer

            # Формируем контекст
            context = "\n".join([f"{m.question}: {m.answer}" for m, _ in similar_memories])
        else:
            print("❌ Похожих воспоминаний не найдено")
            context = ""

        # ШАГ 2: Коллективное мышление нейросетей
        print(f"\n🧠 ШАГ 2: Коллективное мышление нейросетей")

        best_answer = None
        best_confidence = 0.0

        for iteration in range(Config.THINKING_ITERATIONS):
            print(f"\n   Итерация #{iteration + 1}:")
            answers, confidence = self.collective.think(question, context)

            if answers:
                print(f"   📊 Согласованность: {confidence:.1%}")
                if confidence > best_confidence:
                    best_answer = max(answers, key=len)
                    best_confidence = confidence

                if confidence >= 0.6:
                    break

        # ШАГ 3: Принятие решения
        print(f"\n🎯 ШАГ 3: Синтез и решение")

        # Комбинированная уверенность
        semantic_confidence = best_sim if similar_memories else 0.0
        neural_confidence = best_confidence
        combined_confidence = (semantic_confidence * 0.6 + neural_confidence * 0.4)

        print(f"   Семантическая уверенность: {semantic_confidence:.1%}")
        print(f"   Нейросетевая уверенность: {neural_confidence:.1%}")
        print(f"   Общая уверенность: {combined_confidence:.1%}")

        if combined_confidence >= Config.CONFIDENCE_THRESHOLD:
            # Выбираем лучший ответ
            if semantic_confidence > neural_confidence and similar_memories:
                final_answer = similar_memories[0][0].answer
            else:
                final_answer = best_answer if best_answer else "Не знаю"

            print(f"\n✅ ОТВЕЧАЮ САМОСТОЯТЕЛЬНО")
            print(f"💡 МОЙ ОТВЕТ: {final_answer}")

            # Сохраняем как собственное знание
            self.memory.add(question, final_answer, source='self', confidence=combined_confidence)

            return final_answer
        else:
            print(f"\n❓ НЕДОСТАТОЧНО УВЕРЕН")
            print(f"   Мой вариант: {best_answer if best_answer else 'Не знаю'}")
            print(f"\n👨‍🏫 Учусь у Qwen...")

            return self._learn_from_qwen(question)

    def _learn_from_qwen(self, question: str) -> str:
        """Учится у Qwen"""

        try:
            resp = requests.post(Config.QWEN_API, json={
                "messages": [{"role": "user", "content": f"{question}\n\nДай короткий ответ (1-2 предложения)."}],
                "max_tokens": 100,
                "temperature": 0.7
            }, timeout=20)

            if resp.status_code == 200:
                answer = clean_text(resp.json()['choices'][0]['message']['content'])
                print(f"👨‍🏫 QWEN: {answer}")

                # Сохраняем в память
                self.memory.add(question, answer, source='qwen', confidence=1.0)

                # Обучаем нейросети
                self.collective.train_all(question, answer, epochs=8)

                print(f"\n💡 ИТОГОВЫЙ ОТВЕТ: {answer}")
                return answer
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")

        return "Не удалось получить ответ"

    def show_statistics(self):
        total = len(self.memory.memories)
        self_count = sum(1 for m in self.memory.memories if m.source == 'self')
        qwen_count = sum(1 for m in self.memory.memories if m.source == 'qwen')

        print(f"\n📊 СТАТИСТИКА:")
        print(f"  Воспоминаний: {total}")
        print(f"  🤖 Самостоятельных: {self_count} ({self_count / total * 100 if total else 0:.1f}%)")
        print(f"  👨‍🏫 От Qwen: {qwen_count} ({qwen_count / total * 100 if total else 0:.1f}%)")
        print(f"  📚 Словарь: {self.vocab.size} слов")
        print(f"  🧠 Нейросетей: {Config.NUM_MINDS}")

        if self.memory.memories:
            most_used = max(self.memory.memories, key=lambda m: m.usage_count)
            print(f"  🔥 Популярное: '{most_used.question[:30]}...' ({most_used.usage_count}x)")

    def show_memory(self, count: int = 10):
        recent = self.memory.memories[-count:]
        if not recent:
            print("\n📚 Память пуста")
            return

        print(f"\n📚 ПОСЛЕДНИЕ {len(recent)} ВОСПОМИНАНИЙ:")
        for i, mem in enumerate(recent, 1):
            icon = "🤖" if mem.source == 'self' else "👨‍🏫"
            print(f"\n{i}. {icon} {mem.question}")
            print(f"   → {mem.answer}")
            print(f"   📊 {mem.confidence:.0%} | Использовано: {mem.usage_count}x")


# ====================== ИНТЕРФЕЙС ======================
def main():
    try:
        system = HybridCognitiveSystem()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return

    print(f"\n💡 КОМАНДЫ: 'выход', 'память', 'статистика'")

    while True:
        try:
            user_input = input("\n👤 ВЫ: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("\n✨ До встречи!")
                break

            if user_input.lower() in ['память', 'знания']:
                system.show_memory(10)
                continue

            if user_input.lower() == 'статистика':
                system.show_statistics()
                continue

            # Обработка вопроса
            system.process(user_input)

        except KeyboardInterrupt:
            print("\n✨ Прерывание")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()