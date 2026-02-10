# coding: utf-8
"""
AGI_Enhanced_v24_FactMemory.py — С ФАКТОЛОГИЧЕСКОЙ ПАМЯТЬЮ
Исправлена проблема запоминания конкретных фактов (чисел, имён, дат)
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
from typing import Dict, List, Optional, Tuple, Set, Any
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
    ROOT = Path("./cognitive_v24")
    ROOT.mkdir(exist_ok=True)

    # Файлы памяти
    SEMANTIC_DB = ROOT / "semantic_memory.json"
    EPISODIC_DB = ROOT / "episodic_memory.json"
    CAUSAL_DB = ROOT / "causal_graph.json"
    WORKING_DB = ROOT / "working_memory.json"
    META_DB = ROOT / "meta_state.json"
    FACTUAL_DB = ROOT / "factual_memory.json"  # НОВОЕ!
    LOG = ROOT / "system.log"

    # Параметры памяти
    WORKING_MEMORY_SIZE = 15
    EPISODIC_MEMORY_SIZE = 200
    SEMANTIC_MEMORY_SIZE = 1000
    FACTUAL_MEMORY_SIZE = 500  # НОВОЕ!

    # Параметры обучения
    LEARNING_RATE = 0.15
    DECAY_RATE = 0.003  # Медленнее забываем факты
    MIN_CONFIDENCE = 0.1
    CONSOLIDATION_THRESHOLD = 3

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
    """Извлечение ключевых слов"""
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


def extract_numbers(text: str) -> List[int]:
    """Извлечение чисел из текста"""
    # Находим все числа в тексте
    numbers = re.findall(r'\b\d+\b', text)
    return [int(n) for n in numbers]


def extract_facts(text: str) -> Dict[str, Any]:
    """Извлечение фактов из текста"""
    facts = {}

    # Числа
    numbers = extract_numbers(text)
    if numbers:
        facts['numbers'] = numbers

    # Паттерны для извлечения фактов
    patterns = {
        'name': r'(?:меня зовут|я|мое имя|моё имя)\s+([А-ЯЁA-Z][а-яёa-z]+)',
        'age': r'(?:мне|возраст)\s+(\d+)\s*(?:лет|год|года)',
        'color': r'(?:любимый цвет|цвет)\s+([а-яё]+)',
        'city': r'(?:живу в|город|из города)\s+([А-ЯЁ][а-яё]+)',
    }

    for fact_type, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts[fact_type] = match.group(1)

    return facts


def text_hash(text: str) -> str:
    """Хеш текста"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Косинусное сходство"""
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


# ================= ФАКТОЛОГИЧЕСКАЯ ПАМЯТЬ (НОВОЕ!) =================
@dataclass
class Fact:
    """Конкретный факт"""
    fact_type: str  # 'number', 'name', 'date', 'color', etc.
    value: Any
    context: str  # В каком контексте упомянут
    timestamp: float
    confidence: float = 1.0
    source: str = "user"  # откуда получен факт

    def to_dict(self) -> dict:
        return {
            'fact_type': self.fact_type,
            'value': self.value,
            'context': self.context,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'source': self.source
        }

    @staticmethod
    def from_dict(data: dict) -> 'Fact':
        return Fact(
            fact_type=data['fact_type'],
            value=data['value'],
            context=data['context'],
            timestamp=data['timestamp'],
            confidence=data.get('confidence', 1.0),
            source=data.get('source', 'user')
        )


class FactualMemory:
    """Долговременная фактологическая память - хранит конкретные факты"""

    def __init__(self):
        self.facts: Dict[str, List[Fact]] = defaultdict(list)
        self.load()

    def add_fact(self, fact_type: str, value: Any, context: str, confidence: float = 1.0):
        """Добавить факт"""
        fact = Fact(
            fact_type=fact_type,
            value=value,
            context=context[:200],  # ограничиваем контекст
            timestamp=time.time(),
            confidence=confidence
        )

        # Проверяем, нет ли уже такого факта
        existing = self.facts[fact_type]
        for i, old_fact in enumerate(existing):
            if old_fact.value == value:
                # Обновляем существующий факт
                existing[i] = fact
                return

        # Добавляем новый факт
        self.facts[fact_type].append(fact)

        # Ограничиваем количество фактов одного типа
        if len(self.facts[fact_type]) > 50:
            # Удаляем самые старые с низкой уверенностью
            self.facts[fact_type].sort(key=lambda f: f.confidence * f.timestamp)
            self.facts[fact_type] = self.facts[fact_type][-50:]

    def learn_from_text(self, text: str):
        """Автоматическое извлечение и запоминание фактов"""
        # Извлекаем факты
        facts = extract_facts(text)

        # Сохраняем числа
        if 'numbers' in facts:
            for num in facts['numbers']:
                # Ищем контекст для числа
                context_match = re.search(rf'(.{{0,50}}){num}(.{{0,50}})', text)
                context = text if not context_match else context_match.group(0)
                self.add_fact('number', num, context)

        # Сохраняем другие факты
        for fact_type, value in facts.items():
            if fact_type != 'numbers':
                self.add_fact(fact_type, value, text)

    def get_facts_by_type(self, fact_type: str) -> List[Fact]:
        """Получить все факты определённого типа"""
        return sorted(
            self.facts.get(fact_type, []),
            key=lambda f: f.timestamp,
            reverse=True
        )

    def search_facts(self, query: str) -> List[Fact]:
        """Поиск фактов по запросу"""
        results = []
        query_lower = query.lower()

        # Проверяем упоминание типов фактов
        if any(word in query_lower for word in ['число', 'цифр', 'number']):
            results.extend(self.get_facts_by_type('number'))

        if any(word in query_lower for word in ['имя', 'зовут', 'name']):
            results.extend(self.get_facts_by_type('name'))

        if any(word in query_lower for word in ['возраст', 'лет', 'age']):
            results.extend(self.get_facts_by_type('age'))

        if any(word in query_lower for word in ['цвет', 'color']):
            results.extend(self.get_facts_by_type('color'))

        # Если не нашли по типу, ищем по контексту
        if not results:
            for fact_list in self.facts.values():
                for fact in fact_list:
                    if query_lower in fact.context.lower():
                        results.append(fact)

        return results[:10]  # ограничиваем результаты

    def get_all_facts(self) -> List[Fact]:
        """Получить все факты"""
        all_facts = []
        for fact_list in self.facts.values():
            all_facts.extend(fact_list)
        return sorted(all_facts, key=lambda f: f.timestamp, reverse=True)

    def format_facts_for_context(self, facts: List[Fact]) -> str:
        """Форматировать факты для контекста"""
        if not facts:
            return ""

        lines = []

        # Группируем по типам
        by_type = defaultdict(list)
        for fact in facts:
            by_type[fact.fact_type].append(fact)

        # Форматируем по типам
        if 'number' in by_type:
            numbers = [str(f.value) for f in by_type['number']]
            lines.append(f"Запомненные числа: {', '.join(numbers)}")

        if 'name' in by_type:
            names = [str(f.value) for f in by_type['name']]
            lines.append(f"Имена: {', '.join(names)}")

        if 'age' in by_type:
            ages = [str(f.value) for f in by_type['age']]
            lines.append(f"Возраст: {', '.join(ages)}")

        if 'color' in by_type:
            colors = [str(f.value) for f in by_type['color']]
            lines.append(f"Цвета: {', '.join(colors)}")

        # Остальные типы
        for fact_type, fact_list in by_type.items():
            if fact_type not in ['number', 'name', 'age', 'color']:
                values = [str(f.value) for f in fact_list]
                lines.append(f"{fact_type}: {', '.join(values)}")

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """Статистика памяти"""
        return {
            'total_facts': sum(len(facts) for facts in self.facts.values()),
            'fact_types': len(self.facts),
            'by_type': {k: len(v) for k, v in self.facts.items()}
        }

    def save(self):
        """Сохранить память"""
        data = {
            fact_type: [f.to_dict() for f in facts]
            for fact_type, facts in self.facts.items()
        }
        with open(Config.FACTUAL_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """Загрузить память"""
        if Config.FACTUAL_DB.exists():
            try:
                with open(Config.FACTUAL_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for fact_type, facts_data in data.items():
                        self.facts[fact_type] = [
                            Fact.from_dict(f) for f in facts_data
                        ]
            except Exception as e:
                print(f"⚠️ Ошибка загрузки фактологической памяти: {e}")


# ================= КОНЦЕПТ =================
@dataclass
class Concept:
    """Концепт"""
    name: str
    confidence: float = 0.2
    frequency: int = 0
    last_accessed: float = field(default_factory=time.time)
    relations: Dict[str, float] = field(default_factory=dict)
    causes: Dict[str, float] = field(default_factory=dict)
    effects: Dict[str, float] = field(default_factory=dict)
    contexts: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    vector: Dict[str, float] = field(default_factory=dict)

    def reinforce(self, amount: float = None):
        if amount is None:
            amount = Config.LEARNING_RATE
        self.confidence = min(1.0, self.confidence + amount)
        self.frequency += 1
        self.last_accessed = time.time()

    def decay(self):
        self.confidence *= (1 - Config.DECAY_RATE)
        if self.frequency > 0:
            self.frequency -= 1

    def add_relation(self, other: str, strength: float = 0.3):
        current = self.relations.get(other, 0.0)
        self.relations[other] = min(1.0, current + strength)

    def add_context(self, context: str):
        if context not in self.contexts:
            self.contexts.append(context)
            if len(self.contexts) > 10:
                self.contexts.pop(0)

    def update_vector(self, keywords: List[str]):
        for word in keywords:
            self.vector[word] = self.vector.get(word, 0.0) + 1.0
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
    """Эпизод"""
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
    """Семантическая память"""

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.load()

    def get_or_create(self, name: str) -> Concept:
        if name not in self.concepts:
            self.concepts[name] = Concept(name=name)
        return self.concepts[name]

    def learn_from_text(self, text: str, importance: float = 0.5):
        keywords = extract_keywords(text)
        for word in keywords:
            concept = self.get_or_create(word)
            concept.reinforce(Config.LEARNING_RATE * importance)
            concept.update_vector(keywords)
            concept.add_context(text[:100])

        for i in range(len(keywords) - 1):
            c1 = self.get_or_create(keywords[i])
            c2 = self.get_or_create(keywords[i + 1])
            c1.add_relation(c2.name, 0.2)
            c2.add_relation(c1.name, 0.15)

    def find_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_keywords = extract_keywords(query)
        query_vector = {}
        for word in query_keywords:
            query_vector[word] = query_vector.get(word, 0.0) + 1.0

        total = sum(query_vector.values())
        if total > 0:
            query_vector = {k: v / total for k, v in query_vector.items()}

        similarities = []
        for name, concept in self.concepts.items():
            if concept.confidence < Config.MIN_CONFIDENCE:
                continue

            sim = cosine_similarity(query_vector, concept.vector)
            if sim > 0:
                score = sim * concept.confidence * (1 + math.log1p(concept.frequency))
                similarities.append((name, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def decay_all(self):
        for concept in self.concepts.values():
            concept.decay()

        to_remove = [
            name for name, c in self.concepts.items()
            if c.confidence < Config.MIN_CONFIDENCE and c.frequency == 0
        ]
        for name in to_remove:
            del self.concepts[name]

    def get_statistics(self) -> dict:
        return {
            'total_concepts': len(self.concepts),
            'strong_concepts': sum(1 for c in self.concepts.values() if c.confidence > 0.5),
            'total_relations': sum(len(c.relations) for c in self.concepts.values()),
            'avg_confidence': sum(c.confidence for c in self.concepts.values()) / max(len(self.concepts), 1)
        }

    def save(self):
        data = {name: concept.to_dict() for name, concept in self.concepts.items()}
        with open(Config.SEMANTIC_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
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
    """Эпизодическая память"""

    def __init__(self):
        self.episodes: List[Episode] = []
        self.load()

    def add(self, input_text: str, response: str, concepts: List[str], importance: float = 0.5):
        episode = Episode(
            id=text_hash(f"{input_text}{time.time()}"),
            timestamp=time.time(),
            input_text=input_text,
            response=response,
            concepts=concepts,
            importance=importance
        )

        self.episodes.append(episode)

        if len(self.episodes) > Config.EPISODIC_MEMORY_SIZE:
            self.episodes.sort(key=lambda e: e.importance * (1 / (time.time() - e.timestamp + 1)))
            self.episodes = self.episodes[-Config.EPISODIC_MEMORY_SIZE:]

    def recall_similar(self, query: str, top_k: int = 3) -> List[Episode]:
        query_keywords = set(extract_keywords(query))

        scored_episodes = []
        for episode in self.episodes:
            episode_keywords = set(extract_keywords(episode.input_text))

            intersection = len(query_keywords & episode_keywords)
            union = len(query_keywords | episode_keywords)

            if union > 0:
                similarity = intersection / union
                recency = 1 / (1 + (time.time() - episode.timestamp) / 86400)
                score = similarity * episode.importance * (0.5 + 0.5 * recency)
                scored_episodes.append((episode, score))

        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored_episodes[:top_k]]

    def get_recent(self, n: int = 5) -> List[Episode]:
        return sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def save(self):
        data = [ep.to_dict() for ep in self.episodes]
        with open(Config.EPISODIC_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.EPISODIC_DB.exists():
            try:
                with open(Config.EPISODIC_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.episodes = [Episode.from_dict(ep) for ep in data]
            except Exception as e:
                print(f"⚠️ Ошибка загрузки эпизодической памяти: {e}")


# ================= ПРИЧИННАЯ ПАМЯТЬ =================
class CausalMemory:
    """Причинная память"""

    def __init__(self):
        self.graph: Dict[str, Dict[str, float]] = {}
        self.load()

    def add_link(self, cause: str, effect: str, strength: float = 0.3):
        if cause not in self.graph:
            self.graph[cause] = {}
        current = self.graph[cause].get(effect, 0.0)
        self.graph[cause][effect] = min(1.0, current + strength)

    def learn_from_conditional(self, text: str):
        text = clean_text(text)

        if 'если' in text and 'то' in text:
            parts = text.split('то', 1)
            condition = parts[0].replace('если', '').strip()
            consequence = parts[1].strip()

            cond_keywords = extract_keywords(condition)
            cons_keywords = extract_keywords(consequence)

            if cond_keywords and cons_keywords:
                for c in cond_keywords[-2:]:
                    for e in cons_keywords[:2]:
                        self.add_link(c, e, 0.4)

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
        chain = [start]
        current = start

        for _ in range(max_steps):
            if current not in self.graph or not self.graph[current]:
                break

            next_concept = max(self.graph[current].items(), key=lambda x: x[1])
            if next_concept[1] < 0.2:
                break

            if next_concept[0] in chain:
                break

            chain.append(next_concept[0])
            current = next_concept[0]

        return chain

    def decay_all(self):
        for cause in list(self.graph.keys()):
            for effect in list(self.graph[cause].keys()):
                self.graph[cause][effect] *= (1 - Config.DECAY_RATE)

                if self.graph[cause][effect] < Config.MIN_CONFIDENCE:
                    del self.graph[cause][effect]

            if not self.graph[cause]:
                del self.graph[cause]

    def save(self):
        with open(Config.CAUSAL_DB, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.CAUSAL_DB.exists():
            try:
                with open(Config.CAUSAL_DB, 'r', encoding='utf-8') as f:
                    self.graph = json.load(f)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки причинной памяти: {e}")


# ================= РАБОЧАЯ ПАМЯТЬ =================
@dataclass
class WorkingMemoryItem:
    content: str
    timestamp: float
    importance: float
    concepts: List[str]


class WorkingMemory:
    """Рабочая память"""

    def __init__(self):
        self.items: List[WorkingMemoryItem] = []
        self.attention_focus: Optional[str] = None

    def add(self, content: str, importance: float = 0.5):
        concepts = extract_keywords(content)

        item = WorkingMemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            concepts=concepts
        )

        self.items.append(item)

        if concepts:
            self.attention_focus = concepts[0]

        if len(self.items) > Config.WORKING_MEMORY_SIZE:
            self.items.sort(key=lambda x: x.importance * (1 / (time.time() - x.timestamp + 1)))
            self.items = self.items[-Config.WORKING_MEMORY_SIZE:]

    def get_recent_context(self, n: int = 5) -> List[str]:
        recent = sorted(self.items, key=lambda x: x.timestamp, reverse=True)[:n]
        return [item.content for item in recent]


# ================= КОГНИТИВНАЯ СИСТЕМА =================
class CognitiveSystemV24:
    """Когнитивная система с фактологической памятью"""

    def __init__(self):
        print("🧠 Cognitive System v24 — With Factual Memory\n")

        if not Config.OPENROUTER_API_KEY:
            print("❌ КРИТИЧЕСКАЯ ОШИБКА: Не найден OPENROUTER_API_KEY!")
            sys.exit(1)

        # Инициализация систем памяти
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.causal = CausalMemory()
        self.working = WorkingMemory()
        self.factual = FactualMemory()  # НОВОЕ!

        self.meta = self.load_meta()
        self.log_file = open(Config.LOG, 'a', encoding='utf-8')
        self.log("System initialized with factual memory")

        print("✅ Инициализация завершена")
        self._print_statistics()

    def log(self, message: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()

    def _print_statistics(self):
        stats = self.semantic.get_statistics()
        fact_stats = self.factual.get_statistics()
        print(f"\n📊 Статистика памяти:")
        print(f"   Концепты: {stats['total_concepts']} (сильных: {stats['strong_concepts']})")
        print(f"   Связи: {stats['total_relations']}")
        print(f"   Эпизоды: {len(self.episodic.episodes)}")
        print(f"   Причинные связи: {len(self.causal.graph)}")
        print(f"   Факты: {fact_stats['total_facts']} ({fact_stats['fact_types']} типов)")  # НОВОЕ!
        print(f"   Взаимодействий: {self.meta['interactions']}")

    def build_context(self, query: str) -> str:
        """Построить контекст с фактами"""
        context_parts = []

        # 1. ФАКТЫ (САМОЕ ВАЖНОЕ!)
        relevant_facts = self.factual.search_facts(query)
        if relevant_facts:
            context_parts.append("🎯 ЗАПОМНЕННЫЕ ФАКТЫ:")
            fact_text = self.factual.format_facts_for_context(relevant_facts)
            context_parts.append(fact_text)

        # 2. Рабочая память
        recent = self.working.get_recent_context(3)
        if recent:
            context_parts.append("\n💭 ТЕКУЩИЙ КОНТЕКСТ:")
            for i, item in enumerate(recent[::-1], 1):
                context_parts.append(f"  {i}. {item[:100]}")

        # 3. Релевантные эпизоды
        similar_episodes = self.episodic.recall_similar(query, top_k=2)
        if similar_episodes:
            context_parts.append("\n📚 РЕЛЕВАНТНЫЙ ОПЫТ:")
            for i, ep in enumerate(similar_episodes, 1):
                context_parts.append(f"  {i}. Вопрос: {ep.input_text[:80]}")
                context_parts.append(f"     Ответ: {ep.response[:80]}")

        # 4. Семантические концепты
        similar_concepts = self.semantic.find_similar(query, top_k=4)
        if similar_concepts:
            context_parts.append("\n🔗 КЛЮЧЕВЫЕ КОНЦЕПТЫ:")
            for name, score in similar_concepts[:3]:
                concept = self.semantic.concepts[name]
                context_parts.append(
                    f"  • {name} (conf: {concept.confidence:.2f}, freq: {concept.frequency})"
                )

        if context_parts:
            return "\n".join(context_parts)

        return ""

    def process(self, user_input: str) -> str:
        """Основная обработка"""
        self.meta['interactions'] += 1
        self.log(f"INPUT: {user_input}")

        # 1. Добавляем в рабочую память
        self.working.add(user_input, importance=0.7)

        # 2. ИЗВЛЕКАЕМ И СОХРАНЯЕМ ФАКТЫ!
        self.factual.learn_from_text(user_input)

        # 3. Обучаемся
        self.semantic.learn_from_text(user_input, importance=0.6)
        self.causal.learn_from_conditional(user_input)

        # 4. Специальные команды
        if user_input.lower() in ['статистика', 'stats', 'память']:
            return self._handle_stats_command()

        if user_input.lower().startswith('вспомни'):
            return self._handle_recall_command(user_input)

        if user_input.lower() in ['факты', 'facts']:
            return self._handle_facts_command()

        # 5. Строим контекст
        context = self.build_context(user_input)

        # 6. Генерируем ответ
        response = self._query_llm(user_input, context)

        # 7. ИЗВЛЕКАЕМ ФАКТЫ ИЗ ОТВЕТА
        self.factual.learn_from_text(response)

        # 8. Обучаемся из ответа
        self.semantic.learn_from_text(response, importance=0.5)

        # 9. Сохраняем эпизод
        concepts = extract_keywords(user_input) + extract_keywords(response)
        self.episodic.add(user_input, response, list(set(concepts)), importance=0.6)

        # 10. Консолидация
        if self.meta['interactions'] % 10 == 0:
            self._consolidate_memory()

        # 11. Сохранение
        self.save_all()

        self.log(f"OUTPUT: {response[:100]}...")

        return response

    def _query_llm(self, query: str, context: str) -> str:
        """Запрос к LLM с контекстом"""
        try:
            system_prompt = (
                "Ты — продвинутая когнитивная система с долговременной памятью. "
                "ВАЖНО: В разделе 'ЗАПОМНЕННЫЕ ФАКТЫ' находятся конкретные факты, которые ты ДОЛЖЕН помнить. "
                "Используй эти факты при ответе. Отвечай на русском языке естественно.\n\n"
            )

            if context:
                system_prompt += f"КОНТЕКСТ ИЗ ПАМЯТИ:\n{context}\n\n"
                system_prompt += (
                    "КРИТИЧЕСКИ ВАЖНО: Если в разделе 'ЗАПОМНЕННЫЕ ФАКТЫ' есть числа, имена или другие факты — "
                    "используй ИМЕННО их в ответе. Не придумывай новые факты."
                )

            if context:
                print(f"\n🧠 Использую контекст памяти ({len(context)} символов)")
                # Показываем факты если есть
                if "ЗАПОМНЕННЫЕ ФАКТЫ" in context:
                    print("📌 Найдены конкретные факты в памяти!")

            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "CognitiveSystemV24"
            }

            payload = {
                "model": Config.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.3,  # Ниже температура для точности фактов
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

        except Exception as e:
            error_msg = f"⚠️ Ошибка: {str(e)[:100]}"
            self.log(f"API ERROR: {e}")
            return error_msg

    def _handle_stats_command(self) -> str:
        """Статистика"""
        stats = self.semantic.get_statistics()
        fact_stats = self.factual.get_statistics()

        output = ["📊 СТАТИСТИКА КОГНИТИВНОЙ СИСТЕМЫ\n"]
        output.append(f"Взаимодействий: {self.meta['interactions']}")

        output.append(f"\nФАКТОЛОГИЧЕСКАЯ ПАМЯТЬ:")
        output.append(f"  • Всего фактов: {fact_stats['total_facts']}")
        output.append(f"  • Типов фактов: {fact_stats['fact_types']}")
        for fact_type, count in fact_stats['by_type'].items():
            output.append(f"    - {fact_type}: {count}")

        output.append(f"\nСЕМАНТИЧЕСКАЯ ПАМЯТЬ:")
        output.append(f"  • Концепты: {stats['total_concepts']}")
        output.append(f"  • Связи: {stats['total_relations']}")

        output.append(f"\nЭПИЗОДИЧЕСКАЯ ПАМЯТЬ:")
        output.append(f"  • Эпизоды: {len(self.episodic.episodes)}")

        output.append(f"\nПРИЧИННАЯ ПАМЯТЬ:")
        output.append(f"  • Причинные связи: {len(self.causal.graph)}")

        return "\n".join(output)

    def _handle_facts_command(self) -> str:
        """Показать все факты"""
        all_facts = self.factual.get_all_facts()

        if not all_facts:
            return "🤔 Фактологическая память пуста."

        output = ["📚 ВСЕ ЗАПОМНЕННЫЕ ФАКТЫ:\n"]

        # Группируем по типам
        by_type = defaultdict(list)
        for fact in all_facts:
            by_type[fact.fact_type].append(fact)

        for fact_type, facts in by_type.items():
            output.append(f"\n{fact_type.upper()}:")
            for fact in facts[:10]:  # максимум 10 на тип
                time_str = datetime.fromtimestamp(fact.timestamp).strftime('%Y-%m-%d %H:%M')
                output.append(f"  • {fact.value} [{time_str}]")
                if fact.context:
                    output.append(f"    Контекст: {fact.context[:60]}...")

        return "\n".join(output)

    def _handle_recall_command(self, command: str) -> str:
        """Воспоминания"""
        query = command.replace('вспомни', '').strip()

        if not query:
            recent = self.episodic.get_recent(5)
            if not recent:
                return "🤔 Эпизодическая память пуста."

            output = ["📚 ПОСЛЕДНИЕ ВОСПОМИНАНИЯ:\n"]
            for i, ep in enumerate(recent, 1):
                time_str = datetime.fromtimestamp(ep.timestamp).strftime('%Y-%m-%d %H:%M')
                output.append(f"{i}. [{time_str}]")
                output.append(f"   Q: {ep.input_text[:80]}")
                output.append(f"   A: {ep.response[:80]}\n")

            return "\n".join(output)
        else:
            similar = self.episodic.recall_similar(query, top_k=3)
            if not similar:
                return f"🤔 Нет воспоминаний о: {query}"

            output = [f"📚 ВОСПОМИНАНИЯ О '{query}':\n"]
            for i, ep in enumerate(similar, 1):
                time_str = datetime.fromtimestamp(ep.timestamp).strftime('%Y-%m-%d %H:%M')
                output.append(f"{i}. [{time_str}]")
                output.append(f"   Q: {ep.input_text}")
                output.append(f"   A: {ep.response}\n")

            return "\n".join(output)

    def _consolidate_memory(self):
        """Консолидация"""
        print("🔄 Консолидация памяти...", end=" ", flush=True)

        self.semantic.decay_all()
        self.causal.decay_all()

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
        """Сохранить всё"""
        self.semantic.save()
        self.episodic.save()
        self.causal.save()
        self.factual.save()  # НОВОЕ!

        with open(Config.META_DB, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load_meta(self) -> dict:
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
        if hasattr(self, 'log_file'):
            self.log_file.close()


# ================= ДИАГНОСТИКА =================
def run_diagnosis() -> bool:
    print("=" * 70)
    print("🔍 ДИАГНОСТИКА СИСТЕМЫ")
    print("=" * 70)

    if not Config.OPENROUTER_API_KEY:
        print("❌ Не найден OPENROUTER_API_KEY")
        return False

    print(f"✅ API ключ: {Config.OPENROUTER_API_KEY[:12]}...{Config.OPENROUTER_API_KEY[-4:]}")
    print(f"✅ Модель: {Config.MODEL}")
    print(f"✅ Директория: {Config.ROOT}")

    try:
        print("\n📡 Проверка API...", end=" ", flush=True)

        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": Config.MODEL,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
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
            return False

    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return False


# ================= MAIN =================
def main():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass

    print("\n" + "=" * 70)
    print("🧠 COGNITIVE SYSTEM v24")
    print("   With Factual Memory — REMEMBERS NUMBERS!")
    print("=" * 70 + "\n")

    if not run_diagnosis():
        print("\n❌ Диагностика не пройдена.")
        return

    print("\n" + "=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ")
    print("=" * 70 + "\n")

    system = CognitiveSystemV24()

    print("\n" + "=" * 70)
    print("💬 СИСТЕМА ГОТОВА")
    print("=" * 70)
    print("\n🎯 Новое в v24:")
    print("  • ФАКТОЛОГИЧЕСКАЯ ПАМЯТЬ — запоминает числа, имена, даты")
    print("  • Автоматическое извлечение фактов из текста")
    print("  • Факты приоритетны в контексте для LLM")
    print("\n📋 Команды:")
    print("  • 'статистика' — статистика памяти")
    print("  • 'факты' — показать все запомненные факты")
    print("  • 'вспомни' — последние воспоминания")
    print("  • 'выход' — завершить")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("💭 Ваш вопрос: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'выход', 'quit', 'q']:
                print("\n👋 Завершение...")
                system.save_all()
                print("💾 Память сохранена")
                break

            print()
            response = system.process(user_input)

            print("\n🤖 Ответ:")
            print_typing(response, delay=0.01)

            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Прервано")
            system.save_all()
            print("💾 Память сохранена")
            break

        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()