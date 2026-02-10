# coding: utf-8
"""
AGI_Enhanced_v23.py — ADVANCED COGNITIVE ARCHITECTURE
Улучшенная архитектура с:
- Векторной семантической памятью
- Механизмом внимания
- Иерархической памятью (рабочая → эпизодическая → семантическая)
- Системой приоритетов и важности
- Контекстным обучением
- Эмоциональной окраской концептов
"""

import re
import json
import requests
import time
import os
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
from collections import defaultdict, Counter
import math


# ================= ЗАГРУЗКА КЛЮЧА =================
def load_api_key():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "OPENROUTER_API_KEY":
                                key = v.strip().strip('"').strip("'")
    return key


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_v23")
    ROOT.mkdir(exist_ok=True)

    # Файлы памяти
    SEMANTIC_DB = ROOT / "semantic_memory.json"
    EPISODIC_DB = ROOT / "episodic_memory.json"
    CAUSAL_DB = ROOT / "causal_graph.json"
    WORKING_DB = ROOT / "working_memory.json"
    META_DB = ROOT / "meta_state.json"
    VECTORS_DB = ROOT / "concept_vectors.json"
    LOG = ROOT / "system.log"

    # Параметры памяти
    WORKING_MEMORY_SIZE = 15  # Краткосрочная память
    EPISODIC_MEMORY_SIZE = 200  # Долгосрочная эпизодическая
    SEMANTIC_MEMORY_SIZE = 1000  # Семантическая память

    # Параметры обучения
    LEARNING_RATE = 0.15
    DECAY_RATE = 0.005
    MIN_CONFIDENCE = 0.1
    CONSOLIDATION_THRESHOLD = 3  # Сколько раз встретить для консолидации

    # Параметры внимания
    ATTENTION_TOP_K = 7
    CONTEXT_WINDOW = 5

    # API параметры
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = load_api_key()
    MODEL = "qwen/qwen-2.5-7b-instruct"
    TIMEOUT = 30
    MAX_TOKENS = 400


# ================= УТИЛИТЫ =================
def clean_text(text: str) -> str:
    """Нормализация текста"""
    text = text.lower()
    text = re.sub(r'[^\w\s\-абвгдеёжзийклмнопрстуфхцчшщъыьэюя]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text: str) -> List[str]:
    """Извлечение ключевых слов с улучшенной фильтрацией"""
    stop_words = {
        'что', 'как', 'почему', 'если', 'то', 'это', 'этот', 'эта', 'эти',
        'я', 'ты', 'мы', 'вы', 'они', 'он', 'она', 'оно', 'мой', 'твой',
        'в', 'на', 'и', 'с', 'по', 'для', 'от', 'к', 'о', 'из', 'у',
        'да', 'нет', 'не', 'ни', 'же', 'бы', 'ли', 'уже', 'еще',
        'быть', 'есть', 'был', 'была', 'было', 'были'
    }

    words = clean_text(text).split()
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]
    return keywords


def text_hash(text: str) -> str:
    """Хеш текста для уникальной идентификации"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Косинусное сходство между векторами"""
    if not vec1 or not vec2:
        return 0.0

    keys = set(vec1.keys()) & set(vec2.keys())
    if not keys:
        return 0.0

    dot = sum(vec1[k] * vec2[k] for k in keys)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


def print_typing(text: str, delay=0.01):
    """Эффект печатания"""
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print(flush=True)


# ================= КОНЦЕПТ =================
@dataclass
class Concept:
    """Расширенный концепт с векторным представлением"""
    name: str
    confidence: float = 0.2
    frequency: int = 0
    last_accessed: float = field(default_factory=time.time)

    # Связи
    relations: Dict[str, float] = field(default_factory=dict)  # связанные концепты
    causes: Dict[str, float] = field(default_factory=dict)  # что вызывает
    effects: Dict[str, float] = field(default_factory=dict)  # что вызывает это

    # Контекст
    contexts: List[str] = field(default_factory=list)  # контексты использования
    emotional_valence: float = 0.0  # эмоциональная окраска (-1 до 1)

    # Векторное представление (TF-IDF подобное)
    vector: Dict[str, float] = field(default_factory=dict)

    def reinforce(self, amount: float = None):
        """Усиление концепта"""
        if amount is None:
            amount = Config.LEARNING_RATE
        self.confidence = min(1.0, self.confidence + amount)
        self.frequency += 1
        self.last_accessed = time.time()

    def decay(self):
        """Затухание концепта"""
        self.confidence *= (1 - Config.DECAY_RATE)
        if self.frequency > 0:
            self.frequency -= 1

    def add_relation(self, other: str, strength: float = 0.3):
        """Добавить связь с другим концептом"""
        current = self.relations.get(other, 0.0)
        self.relations[other] = min(1.0, current + strength)

    def add_context(self, context: str):
        """Добавить контекст использования"""
        if context not in self.contexts:
            self.contexts.append(context)
            if len(self.contexts) > 10:
                self.contexts.pop(0)

    def update_vector(self, keywords: List[str]):
        """Обновить векторное представление"""
        for word in keywords:
            self.vector[word] = self.vector.get(word, 0.0) + 1.0

        # Нормализация
        total = sum(self.vector.values())
        if total > 0:
            self.vector = {k: v / total for k, v in self.vector.items()}

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'confidence': self.confidence,
            'frequency': self.frequency,
            'last_accessed': self.last_accessed,
            'relations': self.relations,
            'causes': self.causes,
            'effects': self.effects,
            'contexts': self.contexts,
            'emotional_valence': self.emotional_valence,
            'vector': self.vector
        }

    @staticmethod
    def from_dict(data: dict) -> 'Concept':
        return Concept(
            name=data['name'],
            confidence=data.get('confidence', 0.2),
            frequency=data.get('frequency', 0),
            last_accessed=data.get('last_accessed', time.time()),
            relations=data.get('relations', {}),
            causes=data.get('causes', {}),
            effects=data.get('effects', {}),
            contexts=data.get('contexts', []),
            emotional_valence=data.get('emotional_valence', 0.0),
            vector=data.get('vector', {})
        )


# ================= ЭПИЗОД =================
@dataclass
class Episode:
    """Эпизодическая память - конкретное событие"""
    id: str
    timestamp: float
    input_text: str
    response: str
    concepts: List[str]
    importance: float = 0.5
    emotional_tone: float = 0.0

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'input_text': self.input_text,
            'response': self.response,
            'concepts': self.concepts,
            'importance': self.importance,
            'emotional_tone': self.emotional_tone
        }

    @staticmethod
    def from_dict(data: dict) -> 'Episode':
        return Episode(
            id=data['id'],
            timestamp=data['timestamp'],
            input_text=data['input_text'],
            response=data['response'],
            concepts=data['concepts'],
            importance=data.get('importance', 0.5),
            emotional_tone=data.get('emotional_tone', 0.0)
        )


# ================= СЕМАНТИЧЕСКАЯ ПАМЯТЬ =================
class SemanticMemory:
    """Долговременная семантическая память с векторным поиском"""

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.load()

    def get_or_create(self, name: str) -> Concept:
        """Получить или создать концепт"""
        if name not in self.concepts:
            self.concepts[name] = Concept(name=name)
        return self.concepts[name]

    def learn_from_text(self, text: str, importance: float = 0.5):
        """Обучение из текста"""
        keywords = extract_keywords(text)

        # Создание/усиление концептов
        for word in keywords:
            concept = self.get_or_create(word)
            concept.reinforce(Config.LEARNING_RATE * importance)
            concept.update_vector(keywords)
            concept.add_context(text[:100])

        # Создание связей между соседними словами
        for i in range(len(keywords) - 1):
            c1 = self.get_or_create(keywords[i])
            c2 = self.get_or_create(keywords[i + 1])
            c1.add_relation(c2.name, 0.2)
            c2.add_relation(c1.name, 0.15)

    def find_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Найти похожие концепты через векторное сходство"""
        query_keywords = extract_keywords(query)
        query_vector = {}
        for word in query_keywords:
            query_vector[word] = query_vector.get(word, 0.0) + 1.0

        # Нормализация
        total = sum(query_vector.values())
        if total > 0:
            query_vector = {k: v / total for k, v in query_vector.items()}

        similarities = []
        for name, concept in self.concepts.items():
            if concept.confidence < Config.MIN_CONFIDENCE:
                continue

            sim = cosine_similarity(query_vector, concept.vector)
            if sim > 0:
                # Учитываем confidence и frequency
                score = sim * concept.confidence * (1 + math.log1p(concept.frequency))
                similarities.append((name, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_related_concepts(self, concept_name: str, depth: int = 2) -> Set[str]:
        """Получить связанные концепты с заданной глубиной"""
        if concept_name not in self.concepts:
            return set()

        result = {concept_name}
        current_level = {concept_name}

        for _ in range(depth):
            next_level = set()
            for name in current_level:
                if name in self.concepts:
                    concept = self.concepts[name]
                    # Добавляем связанные концепты с высокой силой связи
                    for rel_name, strength in concept.relations.items():
                        if strength > 0.3:
                            next_level.add(rel_name)

            result.update(next_level)
            current_level = next_level

            if not current_level:
                break

        return result

    def decay_all(self):
        """Затухание всех концептов"""
        for concept in self.concepts.values():
            concept.decay()

        # Удаление слабых концептов
        to_remove = [
            name for name, c in self.concepts.items()
            if c.confidence < Config.MIN_CONFIDENCE and c.frequency == 0
        ]
        for name in to_remove:
            del self.concepts[name]

    def get_statistics(self) -> dict:
        """Статистика памяти"""
        return {
            'total_concepts': len(self.concepts),
            'strong_concepts': sum(1 for c in self.concepts.values() if c.confidence > 0.5),
            'total_relations': sum(len(c.relations) for c in self.concepts.values()),
            'avg_confidence': sum(c.confidence for c in self.concepts.values()) / max(len(self.concepts), 1)
        }

    def save(self):
        """Сохранить память"""
        data = {name: concept.to_dict() for name, concept in self.concepts.items()}
        with open(Config.SEMANTIC_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """Загрузить память"""
        if Config.SEMANTIC_DB.exists():
            try:
                with open(Config.SEMANTIC_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.concepts = {
                        name: Concept.from_dict(cdata)
                        for name, cdata in data.items()
                    }
            except Exception as e:
                print(f"⚠️ Ошибка загрузки семантической памяти: {e}")


# ================= ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ =================
class EpisodicMemory:
    """Эпизодическая память - конкретные события"""

    def __init__(self):
        self.episodes: List[Episode] = []
        self.load()

    def add(self, input_text: str, response: str, concepts: List[str], importance: float = 0.5):
        """Добавить эпизод"""
        episode = Episode(
            id=text_hash(f"{input_text}{time.time()}"),
            timestamp=time.time(),
            input_text=input_text,
            response=response,
            concepts=concepts,
            importance=importance
        )

        self.episodes.append(episode)

        # Ограничение размера
        if len(self.episodes) > Config.EPISODIC_MEMORY_SIZE:
            # Удаляем наименее важные старые эпизоды
            self.episodes.sort(key=lambda e: e.importance * (1 / (time.time() - e.timestamp + 1)))
            self.episodes = self.episodes[-Config.EPISODIC_MEMORY_SIZE:]

    def recall_similar(self, query: str, top_k: int = 3) -> List[Episode]:
        """Вспомнить похожие эпизоды"""
        query_keywords = set(extract_keywords(query))

        scored_episodes = []
        for episode in self.episodes:
            episode_keywords = set(extract_keywords(episode.input_text))

            # Jaccard similarity
            intersection = len(query_keywords & episode_keywords)
            union = len(query_keywords | episode_keywords)

            if union > 0:
                similarity = intersection / union
                # Учитываем важность и свежесть
                recency = 1 / (1 + (time.time() - episode.timestamp) / 86400)  # дни
                score = similarity * episode.importance * (0.5 + 0.5 * recency)
                scored_episodes.append((episode, score))

        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored_episodes[:top_k]]

    def get_recent(self, n: int = 5) -> List[Episode]:
        """Получить последние эпизоды"""
        return sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def save(self):
        """Сохранить память"""
        data = [ep.to_dict() for ep in self.episodes]
        with open(Config.EPISODIC_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """Загрузить память"""
        if Config.EPISODIC_DB.exists():
            try:
                with open(Config.EPISODIC_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.episodes = [Episode.from_dict(ep) for ep in data]
            except Exception as e:
                print(f"⚠️ Ошибка загрузки эпизодической памяти: {e}")


# ================= ПРИЧИННАЯ ПАМЯТЬ =================
class CausalMemory:
    """Причинно-следственные связи"""

    def __init__(self):
        self.graph: Dict[str, Dict[str, float]] = {}
        self.load()

    def add_link(self, cause: str, effect: str, strength: float = 0.3):
        """Добавить причинную связь"""
        if cause not in self.graph:
            self.graph[cause] = {}

        current = self.graph[cause].get(effect, 0.0)
        self.graph[cause][effect] = min(1.0, current + strength)

    def learn_from_conditional(self, text: str):
        """Обучение из условных конструкций"""
        text = clean_text(text)

        # Если-то паттерн
        if 'если' in text and 'то' in text:
            parts = text.split('то', 1)
            condition = parts[0].replace('если', '').strip()
            consequence = parts[1].strip()

            cond_keywords = extract_keywords(condition)
            cons_keywords = extract_keywords(consequence)

            if cond_keywords and cons_keywords:
                # Связываем ключевые концепты
                for c in cond_keywords[-2:]:  # последние 2 слова из условия
                    for e in cons_keywords[:2]:  # первые 2 слова из следствия
                        self.add_link(c, e, 0.4)

        # Потому что паттерн
        elif 'потому что' in text or 'так как' in text:
            if 'потому что' in text:
                parts = text.split('потому что', 1)
            else:
                parts = text.split('так как', 1)

            effect_part = parts[0].strip()
            cause_part = parts[1].strip()

            cause_keywords = extract_keywords(cause_part)
            effect_keywords = extract_keywords(effect_part)

            if cause_keywords and effect_keywords:
                for c in cause_keywords[-2:]:
                    for e in effect_keywords[:2]:
                        self.add_link(c, e, 0.4)

    def predict_chain(self, start: str, max_steps: int = 5) -> List[str]:
        """Предсказать причинную цепочку"""
        chain = [start]
        current = start

        for _ in range(max_steps):
            if current not in self.graph or not self.graph[current]:
                break

            # Выбираем наиболее вероятное следствие
            next_concept = max(self.graph[current].items(), key=lambda x: x[1])
            if next_concept[1] < 0.2:  # слишком слабая связь
                break

            if next_concept[0] in chain:  # цикл
                break

            chain.append(next_concept[0])
            current = next_concept[0]

        return chain

    def get_all_chains(self, min_length: int = 2, max_count: int = 5) -> List[List[str]]:
        """Получить все значимые цепочки"""
        chains = []
        for start in self.graph:
            chain = self.predict_chain(start, max_steps=4)
            if len(chain) >= min_length:
                chains.append(chain)

        # Сортируем по длине и силе связей
        chains.sort(key=lambda c: len(c), reverse=True)
        return chains[:max_count]

    def decay_all(self):
        """Затухание связей"""
        for cause in list(self.graph.keys()):
            for effect in list(self.graph[cause].keys()):
                self.graph[cause][effect] *= (1 - Config.DECAY_RATE)

                if self.graph[cause][effect] < Config.MIN_CONFIDENCE:
                    del self.graph[cause][effect]

            if not self.graph[cause]:
                del self.graph[cause]

    def save(self):
        """Сохранить память"""
        with open(Config.CAUSAL_DB, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

    def load(self):
        """Загрузить память"""
        if Config.CAUSAL_DB.exists():
            try:
                with open(Config.CAUSAL_DB, 'r', encoding='utf-8') as f:
                    self.graph = json.load(f)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки причинной памяти: {e}")


# ================= РАБОЧАЯ ПАМЯТЬ =================
@dataclass
class WorkingMemoryItem:
    """Элемент рабочей памяти"""
    content: str
    timestamp: float
    importance: float
    concepts: List[str]


class WorkingMemory:
    """Краткосрочная рабочая память"""

    def __init__(self):
        self.items: List[WorkingMemoryItem] = []
        self.attention_focus: Optional[str] = None

    def add(self, content: str, importance: float = 0.5):
        """Добавить в рабочую память"""
        concepts = extract_keywords(content)

        item = WorkingMemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            concepts=concepts
        )

        self.items.append(item)

        # Обновляем фокус внимания
        if concepts:
            self.attention_focus = concepts[0]

        # Ограничиваем размер
        if len(self.items) > Config.WORKING_MEMORY_SIZE:
            # Удаляем наименее важные старые элементы
            self.items.sort(key=lambda x: x.importance * (1 / (time.time() - x.timestamp + 1)))
            self.items = self.items[-Config.WORKING_MEMORY_SIZE:]

    def get_recent_context(self, n: int = 5) -> List[str]:
        """Получить недавний контекст"""
        recent = sorted(self.items, key=lambda x: x.timestamp, reverse=True)[:n]
        return [item.content for item in recent]

    def get_relevant(self, query: str, top_k: int = 3) -> List[str]:
        """Получить релевантные элементы"""
        query_concepts = set(extract_keywords(query))

        scored = []
        for item in self.items:
            item_concepts = set(item.concepts)
            overlap = len(query_concepts & item_concepts)

            if overlap > 0:
                score = overlap * item.importance
                scored.append((item.content, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [content for content, _ in scored[:top_k]]


# ================= МЕХАНИЗМ ВНИМАНИЯ =================
class AttentionMechanism:
    """Механизм внимания для выбора релевантной информации"""

    @staticmethod
    def compute_relevance(query: str, items: List[str]) -> List[Tuple[str, float]]:
        """Вычислить релевантность элементов к запросу"""
        query_concepts = set(extract_keywords(query))

        scored = []
        for item in items:
            item_concepts = set(extract_keywords(item))

            # Jaccard similarity
            intersection = len(query_concepts & item_concepts)
            union = len(query_concepts | item_concepts)

            if union > 0:
                relevance = intersection / union
                scored.append((item, relevance))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    @staticmethod
    def select_top_k(items: List[Tuple[str, float]], k: int) -> List[str]:
        """Выбрать топ-K наиболее релевантных"""
        return [item for item, _ in items[:k]]


# ================= КОГНИТИВНАЯ СИСТЕМА =================
class CognitiveSystemV23:
    """Продвинутая когнитивная система с иерархической памятью"""

    def __init__(self):
        print("🧠 Cognitive System v23 — Advanced Memory Architecture\n")

        if not Config.OPENROUTER_API_KEY:
            print("❌ КРИТИЧЕСКАЯ ОШИБКА: Не найден OPENROUTER_API_KEY!")
            sys.exit(1)

        # Инициализация систем памяти
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.causal = CausalMemory()
        self.working = WorkingMemory()
        self.attention = AttentionMechanism()

        # Метаданные
        self.meta = self.load_meta()

        # Логирование
        self.log_file = open(Config.LOG, 'a', encoding='utf-8')
        self.log("System initialized")

        print("✅ Инициализация завершена")
        self._print_statistics()

    def log(self, message: str):
        """Логирование"""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()

    def _print_statistics(self):
        """Вывести статистику памяти"""
        stats = self.semantic.get_statistics()
        print(f"\n📊 Статистика памяти:")
        print(f"   Концепты: {stats['total_concepts']} (сильных: {stats['strong_concepts']})")
        print(f"   Связи: {stats['total_relations']}")
        print(f"   Эпизоды: {len(self.episodic.episodes)}")
        print(f"   Причинные связи: {len(self.causal.graph)}")
        print(f"   Взаимодействий: {self.meta['interactions']}")

    def build_context(self, query: str) -> str:
        """Построить контекст из всех типов памяти"""
        context_parts = []

        # 1. Рабочая память (текущий контекст)
        recent = self.working.get_recent_context(3)
        if recent:
            context_parts.append("💭 ТЕКУЩИЙ КОНТЕКСТ:")
            for i, item in enumerate(recent[::-1], 1):
                context_parts.append(f"  {i}. {item[:100]}")

        # 2. Релевантные эпизоды (прошлый опыт)
        similar_episodes = self.episodic.recall_similar(query, top_k=2)
        if similar_episodes:
            context_parts.append("\n📚 РЕЛЕВАНТНЫЙ ОПЫТ:")
            for i, ep in enumerate(similar_episodes, 1):
                context_parts.append(f"  {i}. Вопрос: {ep.input_text[:80]}")
                context_parts.append(f"     Ответ: {ep.response[:80]}")

        # 3. Семантически похожие концепты
        similar_concepts = self.semantic.find_similar(query, top_k=5)
        if similar_concepts:
            context_parts.append("\n🔗 КЛЮЧЕВЫЕ КОНЦЕПТЫ:")
            for name, score in similar_concepts:
                concept = self.semantic.concepts[name]
                context_parts.append(
                    f"  • {name} (уверенность: {concept.confidence:.2f}, "
                    f"частота: {concept.frequency})"
                )

                # Добавляем связанные концепты
                if concept.relations:
                    top_relations = sorted(
                        concept.relations.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    rel_str = ", ".join([f"{r[0]}({r[1]:.2f})" for r in top_relations])
                    context_parts.append(f"    Связи: {rel_str}")

        # 4. Причинные цепочки
        query_keywords = extract_keywords(query)
        if query_keywords:
            chains = []
            for keyword in query_keywords[:2]:
                chain = self.causal.predict_chain(keyword, max_steps=3)
                if len(chain) > 1:
                    chains.append(" → ".join(chain))

            if chains:
                context_parts.append("\n⚡ ПРИЧИННЫЕ СВЯЗИ:")
                for chain in chains[:3]:
                    context_parts.append(f"  • {chain}")

        if context_parts:
            return "\n".join(context_parts)

        return ""

    def process(self, user_input: str) -> str:
        """Основная обработка входа"""
        self.meta['interactions'] += 1

        # Логирование
        self.log(f"INPUT: {user_input}")

        # 1. Добавляем в рабочую память
        self.working.add(user_input, importance=0.7)

        # 2. Обучаемся из входа
        self.semantic.learn_from_text(user_input, importance=0.6)
        self.causal.learn_from_conditional(user_input)

        # 3. Проверяем специальные команды
        if user_input.lower() in ['статистика', 'stats', 'память']:
            return self._handle_stats_command()

        if user_input.lower().startswith('вспомни'):
            return self._handle_recall_command(user_input)

        # 4. Строим контекст
        context = self.build_context(user_input)

        # 5. Генерируем ответ через внешнюю модель
        response = self._query_llm(user_input, context)

        # 6. Обучаемся из ответа
        self.semantic.learn_from_text(response, importance=0.5)

        # 7. Сохраняем эпизод
        concepts = extract_keywords(user_input) + extract_keywords(response)
        self.episodic.add(user_input, response, list(set(concepts)), importance=0.6)

        # 8. Периодическая консолидация
        if self.meta['interactions'] % 10 == 0:
            self._consolidate_memory()

        # 9. Сохранение
        self.save_all()

        self.log(f"OUTPUT: {response[:100]}...")

        return response

    def _query_llm(self, query: str, context: str) -> str:
        """Запрос к языковой модели с контекстом"""
        try:
            # Формируем системный промпт
            system_prompt = (
                "Ты — продвинутая когнитивная система с долговременной памятью. "
                "Используй предоставленный контекст из своей памяти для ответа. "
                "Отвечай на русском языке естественно и информативно.\n\n"
            )

            if context:
                system_prompt += f"КОНТЕКСТ ИЗ ПАМЯТИ:\n{context}\n\n"
                system_prompt += (
                    "ВАЖНО: Опирайся на контекст памяти при ответе. "
                    "Если в памяти есть релевантная информация — используй её. "
                    "Не выдумывай факты, которых нет в контексте."
                )

            # Показываем что используем память
            if context:
                print(f"\n🧠 Использую контекст памяти ({len(context)} символов)")

            # Запрос к API
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "CognitiveSystemV23"
            }

            payload = {
                "model": Config.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.4,
                "max_tokens": Config.MAX_TOKENS
            }

            print("⏳ Генерирую ответ...", flush=True)

            response = requests.post(
                Config.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=Config.TIMEOUT
            )

            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            return content

        except requests.exceptions.Timeout:
            return "⚠️ Превышено время ожидания ответа от сервера."
        except requests.exceptions.RequestException as e:
            error_msg = f"⚠️ Ошибка сети: {str(e)[:100]}"
            self.log(f"API ERROR: {e}")
            return error_msg
        except Exception as e:
            error_msg = f"⚠️ Неожиданная ошибка: {str(e)[:100]}"
            self.log(f"ERROR: {e}")
            return error_msg

    def _handle_stats_command(self) -> str:
        """Обработка команды статистики"""
        stats = self.semantic.get_statistics()

        output = ["📊 СТАТИСТИКА КОГНИТИВНОЙ СИСТЕМЫ\n"]
        output.append(f"Взаимодействий: {self.meta['interactions']}")
        output.append(f"\nСЕМАНТИЧЕСКАЯ ПАМЯТЬ:")
        output.append(f"  • Всего концептов: {stats['total_concepts']}")
        output.append(f"  • Сильных концептов: {stats['strong_concepts']}")
        output.append(f"  • Связей: {stats['total_relations']}")
        output.append(f"  • Средняя уверенность: {stats['avg_confidence']:.2f}")

        output.append(f"\nЭПИЗОДИЧЕСКАЯ ПАМЯТЬ:")
        output.append(f"  • Сохранено эпизодов: {len(self.episodic.episodes)}")

        output.append(f"\nПРИЧИННАЯ ПАМЯТЬ:")
        output.append(f"  • Причинных узлов: {len(self.causal.graph)}")
        total_links = sum(len(effects) for effects in self.causal.graph.values())
        output.append(f"  • Причинных связей: {total_links}")

        # Топ концептов
        top_concepts = sorted(
            self.semantic.concepts.values(),
            key=lambda c: c.confidence * c.frequency,
            reverse=True
        )[:5]

        if top_concepts:
            output.append(f"\nТОП-5 КОНЦЕПТОВ:")
            for i, concept in enumerate(top_concepts, 1):
                output.append(
                    f"  {i}. {concept.name} "
                    f"(conf: {concept.confidence:.2f}, freq: {concept.frequency})"
                )

        return "\n".join(output)

    def _handle_recall_command(self, command: str) -> str:
        """Обработка команды воспоминания"""
        query = command.replace('вспомни', '').strip()

        if not query:
            recent = self.episodic.get_recent(5)
            if not recent:
                return "🤔 Моя эпизодическая память пуста."

            output = ["📚 ПОСЛЕДНИЕ ВОСПОМИНАНИЯ:\n"]
            for i, ep in enumerate(recent, 1):
                time_str = datetime.fromtimestamp(ep.timestamp).strftime('%Y-%m-%d %H:%M')
                output.append(f"{i}. [{time_str}]")
                output.append(f"   Вопрос: {ep.input_text[:80]}")
                output.append(f"   Ответ: {ep.response[:80]}\n")

            return "\n".join(output)

        else:
            similar = self.episodic.recall_similar(query, top_k=3)
            if not similar:
                return f"🤔 Не нашёл воспоминаний о: {query}"

            output = [f"📚 ВОСПОМИНАНИЯ О '{query}':\n"]
            for i, ep in enumerate(similar, 1):
                time_str = datetime.fromtimestamp(ep.timestamp).strftime('%Y-%m-%d %H:%M')
                output.append(f"{i}. [{time_str}]")
                output.append(f"   Вопрос: {ep.input_text}")
                output.append(f"   Ответ: {ep.response}\n")

            return "\n".join(output)

    def _consolidate_memory(self):
        """Консолидация памяти (перенос из краткосрочной в долговременную)"""
        print("🔄 Консолидация памяти...", end=" ", flush=True)

        # Затухание старой информации
        self.semantic.decay_all()
        self.causal.decay_all()

        # Усиление часто встречающихся концептов
        concept_counter = Counter()
        for item in self.working.items:
            for concept in item.concepts:
                concept_counter[concept] += 1

        for concept_name, count in concept_counter.items():
            if count >= Config.CONSOLIDATION_THRESHOLD:
                concept = self.semantic.get_or_create(concept_name)
                concept.reinforce(0.2)

        print("✓")
        self.log("Memory consolidated")

    def save_all(self):
        """Сохранить все системы памяти"""
        self.semantic.save()
        self.episodic.save()
        self.causal.save()

        with open(Config.META_DB, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load_meta(self) -> dict:
        """Загрузить метаданные"""
        if Config.META_DB.exists():
            try:
                with open(Config.META_DB, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки метаданных: {e}")

        return {
            'interactions': 0,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

    def __del__(self):
        """Деструктор"""
        if hasattr(self, 'log_file'):
            self.log_file.close()


# ================= ДИАГНОСТИКА =================
def run_diagnosis() -> bool:
    """Диагностика системы"""
    print("=" * 70)
    print("🔍 ДИАГНОСТИКА СИСТЕМЫ")
    print("=" * 70)

    if not Config.OPENROUTER_API_KEY:
        print("❌ Не найден OPENROUTER_API_KEY")
        print("\n💡 Создайте файл .env со строкой:")
        print("   OPENROUTER_API_KEY=your_key_here")
        return False

    print(f"✅ API ключ: {Config.OPENROUTER_API_KEY[:12]}...{Config.OPENROUTER_API_KEY[-4:]}")
    print(f"✅ Модель: {Config.MODEL}")
    print(f"✅ Директория памяти: {Config.ROOT}")

    # Проверка подключения
    try:
        print("\n📡 Проверка подключения к API...", end=" ", flush=True)

        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "DiagnosticTest"
        }

        payload = {
            "model": Config.MODEL,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        }

        response = requests.post(
            Config.OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            print("✅ УСПЕШНО")
            return True
        else:
            print(f"❌ ОШИБКА {response.status_code}")
            print(f"Ответ: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return False


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
def main():
    """Главная функция"""
    # Настройка консоли для Windows
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass

    print("\n" + "=" * 70)
    print("🧠 COGNITIVE SYSTEM v23")
    print("   Advanced Memory Architecture")
    print("=" * 70 + "\n")

    # Диагностика
    if not run_diagnosis():
        print("\n❌ Диагностика не пройдена. Проверьте настройки.")
        return

    print("\n" + "=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ")
    print("=" * 70 + "\n")

    # Создание системы
    system = CognitiveSystemV23()

    print("\n" + "=" * 70)
    print("💬 СИСТЕМА ГОТОВА К ДИАЛОГУ")
    print("=" * 70)
    print("\n📋 Возможности:")
    print("  • Долговременная семантическая память")
    print("  • Эпизодическая память (конкретные события)")
    print("  • Причинно-следственные связи")
    print("  • Механизм внимания и контекста")
    print("  • Векторный поиск по памяти")
    print("\n🎯 Команды:")
    print("  • 'статистика' или 'память' — показать статистику памяти")
    print("  • 'вспомни' — показать последние воспоминания")
    print("  • 'вспомни <тема>' — вспомнить о конкретной теме")
    print("  • 'выход' или 'exit' — завершить работу")
    print("=" * 70 + "\n")

    # Главный цикл
    while True:
        try:
            user_input = input("💭 Ваш вопрос: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'выход', 'quit', 'q']:
                print("\n👋 Завершение работы...")
                system.save_all()
                print("💾 Память сохранена")
                break

            print()
            response = system.process(user_input)

            print("\n🤖 Ответ:")
            print_typing(response, delay=0.01)

            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем")
            system.save_all()
            print("💾 Память сохранена")
            break

        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()