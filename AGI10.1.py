# coding: utf-8
"""
AGI_MultiMind_v13_REAL.py
РЕШЕНИЕ: Использование embeddings вместо обучения с нуля
Qwen даёт знания → мы индексируем их → отвечаем на основе семантики
"""

import os
import re
import json
import pickle
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple
from pathlib import Path

import numpy as np
import requests

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST_MODEL = True
except:
    _HAS_ST_MODEL = False


# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    SAVE_DIR = Path("./cognitive_multimind_v13")
    KNOWLEDGE_PATH = SAVE_DIR / "knowledge.json"
    THINKING_LOG_PATH = SAVE_DIR / "thinking_log.json"

    QWEN_API = "http://localhost:1234/v1/chat/completions"

    # Параметры мышления
    NUM_PERSPECTIVES = 5  # Сколько разных углов рассмотреть
    CONFIDENCE_TO_ANSWER = 0.75
    SIMILARITY_THRESHOLD = 0.6


Config.SAVE_DIR.mkdir(exist_ok=True)


# ====================== УТИЛИТЫ ======================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ====================== SEMANTIC MEMORY ======================
class SemanticMemory:
    """Семантическая память на embeddings"""

    def __init__(self):
        self.encoder = None
        if _HAS_ST_MODEL:
            try:
                print("📦 Загружаю sentence-transformers...")
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Encoder загружен")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки encoder: {e}")

        self.memories = []  # [{question, answer, embedding, source, confidence, timestamp}]
        self.load()

    def add(self, question: str, answer: str, source: str = "qwen", confidence: float = 1.0):
        """Добавляет память"""

        # Проверяем дубликаты
        for mem in self.memories:
            if mem['question'].lower() == question.lower():
                mem['answer'] = answer
                mem['confidence'] = max(mem['confidence'], confidence)
                mem['source'] = source
                mem['timestamp'] = datetime.now().isoformat()
                self.save()
                return

        # Создаём embedding
        embedding = self._encode(question)

        memory = {
            'question': question,
            'answer': answer,
            'embedding': embedding.tolist() if embedding is not None else None,
            'source': source,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'usage_count': 0
        }

        self.memories.append(memory)
        self.save()
        print(f"💾 Сохранена память: '{question}' → '{answer[:40]}...'")

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Ищет похожие воспоминания"""

        if not self.memories:
            return []

        query_embedding = self._encode(query)
        if query_embedding is None:
            return []

        # Считаем схожесть
        scores = []
        for i, mem in enumerate(self.memories):
            if mem['embedding'] is None:
                continue

            mem_embedding = np.array(mem['embedding'])
            similarity = np.dot(query_embedding, mem_embedding)
            scores.append((i, similarity))

        # Сортируем
        scores.sort(key=lambda x: x[1], reverse=True)

        # Берём топ-k
        results = []
        for idx, sim in scores[:top_k]:
            if sim > 0.3:  # Минимальный порог
                mem = self.memories[idx].copy()
                mem['similarity'] = sim
                mem['usage_count'] += 1
                results.append(mem)

        return results

    def _encode(self, text: str):
        """Кодирует текст в embedding"""
        if self.encoder is None:
            return None

        try:
            embedding = self.encoder.encode(text, normalize_embeddings=True)
            return embedding
        except:
            return None

    def save(self):
        # Сохраняем без embeddings (они большие)
        data = {
            'memories': [
                {
                    'question': m['question'],
                    'answer': m['answer'],
                    'source': m['source'],
                    'confidence': m['confidence'],
                    'timestamp': m['timestamp'],
                    'usage_count': m.get('usage_count', 0)
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
                        embedding = self._encode(m['question'])
                        self.memories.append({
                            **m,
                            'embedding': embedding.tolist() if embedding is not None else None
                        })
                print(f"📚 Загружено {len(self.memories)} воспоминаний")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки: {e}")


# ====================== THINKING PERSPECTIVES ======================
class ThinkingPerspectives:
    """Множественные перспективы мышления"""

    def __init__(self):
        self.perspectives = [
            {
                'name': 'Логик',
                'prompt': 'Подумай логически и рационально. Дай чёткий, обоснованный ответ.'
            },
            {
                'name': 'Аналитик',
                'prompt': 'Проанализируй вопрос с разных сторон. Взвесь все за и против.'
            },
            {
                'name': 'Практик',
                'prompt': 'Дай практичный, полезный ответ, который можно применить.'
            },
            {
                'name': 'Креативщик',
                'prompt': 'Посмотри на вопрос творчески. Найди нестандартные решения.'
            },
            {
                'name': 'Критик',
                'prompt': 'Критически оцени вопрос. Укажи на возможные проблемы и нюансы.'
            }
        ]

    def generate_perspectives(self, question: str, context: str = "") -> List[str]:
        """Генерирует разные перспективы на вопрос"""

        print(f"\n🧠 ГЕНЕРИРУЮ ПЕРСПЕКТИВЫ:")

        perspectives_text = []
        for p in self.perspectives[:Config.NUM_PERSPECTIVES]:
            # Формируем промпт
            if context:
                prompt = f"{context}\n\nВопрос: {question}\n\n{p['prompt']}"
            else:
                prompt = f"Вопрос: {question}\n\n{p['prompt']}"

            perspectives_text.append(f"{p['name']}: {prompt}")
            print(f"  📝 {p['name']}: сформирована перспектива")

        return perspectives_text


# ====================== COGNITIVE SYNTHESIS ======================
class CognitiveSynthesis:
    """Синтез знаний и мышления"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def synthesize_answer(self, question: str, relevant_memories: List[dict],
                          perspectives: List[str]) -> Tuple[str, float]:
        """Синтезирует ответ из памяти и перспектив"""

        print(f"\n🔮 СИНТЕЗ ОТВЕТА:")

        # Если есть очень похожая память - используем её
        if relevant_memories and relevant_memories[0]['similarity'] > 0.85:
            best_mem = relevant_memories[0]
            print(f"  ✅ Найдено точное совпадение (sim={best_mem['similarity']:.1%})")
            return best_mem['answer'], 0.95

        # Формируем контекст из памяти
        context_parts = []
        if relevant_memories:
            print(f"  📚 Использую {len(relevant_memories)} воспоминаний")
            for i, mem in enumerate(relevant_memories[:3], 1):
                context_parts.append(f"Похожий вопрос: {mem['question']}\nОтвет: {mem['answer']}")
                print(f"    {i}. {mem['question'][:40]}... (sim={mem['similarity']:.1%})")

        context = "\n\n".join(context_parts)

        # Формируем финальный промпт
        if context:
            final_prompt = f"""На основе этих знаний:

{context}

Ответь на вопрос: {question}

Дай краткий, точный ответ (1-2 предложения)."""
        else:
            final_prompt = f"Ответь на вопрос: {question}\n\nДай краткий, точный ответ (1-2 предложения)."

        # Спрашиваем API
        print(f"  🤔 Думаю над ответом...")
        answer = self._ask_api(final_prompt)

        if answer:
            # Оцениваем уверенность
            confidence = self._estimate_confidence(relevant_memories, answer)
            print(f"  📊 Уверенность: {confidence:.1%}")
            return answer, confidence

        return "Не знаю", 0.0

    def _ask_api(self, prompt: str) -> str:
        """Спрашивает API"""
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
            print(f"    ⚠️ Ошибка API: {e}")

        return ""

    def _estimate_confidence(self, memories: List[dict], answer: str) -> float:
        """Оценивает уверенность в ответе"""

        if not memories:
            return 0.5

        # Базовая уверенность от лучшего совпадения
        best_sim = memories[0]['similarity']
        base_confidence = best_sim * 0.8

        # Бонус за количество похожих воспоминаний
        count_bonus = min(len(memories) * 0.05, 0.2)

        # Бонус за длину ответа (более развёрнутые ответы = больше уверенности)
        length_bonus = min(len(answer.split()) * 0.01, 0.1)

        total = base_confidence + count_bonus + length_bonus
        return min(total, 1.0)


# ====================== АВТОНОМНАЯ СИСТЕМА ======================
class AutonomousCognitiveSystem:
    """Автономная когнитивная система на embeddings"""

    def __init__(self):
        print(f"\n{'=' * 70}")
        print(f"🧠 АВТОНОМНАЯ КОГНИТИВНАЯ СИСТЕМА v13.0")
        print(f"Семантическая память • Множественные перспективы • Синтез знаний")
        print(f"{'=' * 70}\n")

        self.memory = SemanticMemory()
        self.perspectives = ThinkingPerspectives()
        self.synthesis = CognitiveSynthesis(Config.QWEN_API)

        if not _HAS_ST_MODEL:
            print("⚠️ ВНИМАНИЕ: sentence-transformers не установлен!")
            print("   Установите: pip install sentence-transformers")
            print("   Без него система не сможет работать с семантикой\n")

    def think_and_answer(self, question: str) -> str:
        """Думает и отвечает на вопрос"""

        print(f"\n{'=' * 70}")
        print(f"👤 ВОПРОС: {question}")
        print(f"{'=' * 70}")

        # 1. Ищем в памяти
        print(f"\n🔍 ШАГ 1: Поиск в памяти")
        relevant_memories = self.memory.search(question, top_k=5)

        if relevant_memories:
            print(f"✅ Найдено {len(relevant_memories)} релевантных воспоминаний")
            for i, mem in enumerate(relevant_memories, 1):
                print(f"  {i}. {mem['question'][:50]}... (схожесть: {mem['similarity']:.1%})")
        else:
            print(f"❌ Релевантные воспоминания не найдены")

        # 2. Генерируем перспективы
        print(f"\n🧠 ШАГ 2: Генерация перспектив")
        context = "\n\n".join([f"{m['question']} → {m['answer']}" for m in relevant_memories[:2]])
        perspectives = self.perspectives.generate_perspectives(question, context)

        # 3. Синтезируем ответ
        print(f"\n🔮 ШАГ 3: Синтез ответа")
        answer, confidence = self.synthesis.synthesize_answer(question, relevant_memories, perspectives)

        # 4. Решаем, нужно ли учиться
        if confidence >= Config.CONFIDENCE_TO_ANSWER:
            print(f"\n✅ УВЕРЕН В ОТВЕТЕ (confidence={confidence:.1%})")
            print(f"💡 МОЙ ОТВЕТ: {answer}")

            # Сохраняем как собственное знание
            self.memory.add(question, answer, source='self', confidence=confidence)

            return answer
        else:
            print(f"\n❓ НЕДОСТАТОЧНО УВЕРЕН (confidence={confidence:.1%})")
            print(f"⚠️ Мой вариант: {answer}")
            print(f"\n👨‍🏫 Учусь у Qwen...")

            # Получаем правильный ответ от Qwen
            teacher_answer = self._learn_from_qwen(question)

            print(f"💡 ИТОГОВЫЙ ОТВЕТ: {teacher_answer}")

            return teacher_answer

    def _learn_from_qwen(self, question: str) -> str:
        """Учится у Qwen"""

        try:
            resp = requests.post(Config.QWEN_API, json={
                "messages": [
                    {"role": "user", "content": f"{question}\n\nДай короткий, точный ответ (1-2 предложения)."}],
                "max_tokens": 150,
                "temperature": 0.7
            }, timeout=20)

            if resp.status_code == 200:
                answer = clean_text(resp.json()['choices'][0]['message']['content'])
                print(f"👨‍🏫 QWEN: {answer}")

                # Сохраняем в память
                self.memory.add(question, answer, source='qwen', confidence=1.0)

                return answer
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")

        return "Не удалось получить ответ"

    def show_statistics(self):
        """Показывает статистику"""

        total = len(self.memory.memories)
        self_count = sum(1 for m in self.memory.memories if m['source'] == 'self')
        qwen_count = sum(1 for m in self.memory.memories if m['source'] == 'qwen')

        print(f"\n📊 СТАТИСТИКА:")
        print(f"  Всего воспоминаний: {total}")
        print(f"  🤖 Самостоятельных: {self_count} ({self_count / total * 100 if total else 0:.1f}%)")
        print(f"  👨‍🏫 От Qwen: {qwen_count} ({qwen_count / total * 100 if total else 0:.1f}%)")

        if self.memory.memories:
            most_used = max(self.memory.memories, key=lambda m: m.get('usage_count', 0))
            print(f"  🔥 Самое используемое: '{most_used['question'][:40]}...' ({most_used.get('usage_count', 0)}x)")

    def show_memories(self, count: int = 10):
        """Показывает последние воспоминания"""

        recent = self.memory.memories[-count:]

        if not recent:
            print("\n📚 Память пуста")
            return

        print(f"\n📚 ПОСЛЕДНИЕ {len(recent)} ВОСПОМИНАНИЙ:")
        for i, mem in enumerate(recent, 1):
            icon = "🤖" if mem['source'] == 'self' else "👨‍🏫"
            print(f"\n{i}. {icon} {mem['question']}")
            print(f"   → {mem['answer']}")
            print(f"   📊 {mem['confidence']:.0%} | Использовано: {mem.get('usage_count', 0)}x")


# ====================== ИНТЕРФЕЙС ======================
def main():
    try:
        system = AutonomousCognitiveSystem()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
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
                system.show_memories(10)
                continue

            if user_input.lower() == 'статистика':
                system.show_statistics()
                continue

            # Обрабатываем вопрос
            system.think_and_answer(user_input)

        except KeyboardInterrupt:
            print("\n✨ Прерывание")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()