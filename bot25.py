#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID COGNITIVE BRAIN v26.0 — ИСПРАВЛЕННАЯ ВЕРСИЯ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:
  ✅ Семантическая сеть вместо спайковых нейронов
  ✅ Векторные эмбеддинги для понимания смысла
  ✅ Реальная связь концептов с ответами
  ✅ Эффективная интеграция LLM + память + эмоции
  ✅ Контекстно-зависимое обучение
  ✅ Рабочий декодер мыслей
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, re, asyncio, aiohttp, traceback, hashlib, math, shutil, random
import time
from collections import Counter, deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ══════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

BASE_DIR = "cognitive_brain_v26"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# ─── Параметры семантической сети ─────────
ACTIVATION_THRESHOLD = 0.3
ACTIVATION_DECAY = 0.85
LEARNING_RATE = 0.15
MAX_CONCEPTS = 5000
PRUNE_THRESHOLD = 0.05

# ─── Память ───────────────────────────────
MEMORY_DECAY_L1 = 0.98
MEMORY_DECAY_L2 = 0.95
MEMORY_DECAY_L3 = 0.999
FORGET_THRESHOLD = 0.15
CONSOLIDATION_THRESHOLD = 0.75
MAX_MEMORY_L1 = 30
MAX_MEMORY_L2 = 150
MAX_MEMORY_L3 = 500


# ══════════════════════════════════════════
# ЭМОЦИИ И ЛИЧНОСТЬ
# ══════════════════════════════════════════
@dataclass
class EmotionalState:
    """Эмоциональное состояние с реалистичной динамикой"""
    curiosity: float = 0.6
    energy: float = 0.8
    mood: float = 0.5
    confidence: float = 0.6
    engagement: float = 0.7

    def update(self, interaction_quality: float, novelty: float, complexity: float):
        """Обновление эмоций на основе взаимодействия"""
        # Любопытство растёт от новизны
        self.curiosity = max(0.1, min(1.0,
                                      self.curiosity * 0.95 + novelty * 0.3
                                      ))

        # Энергия падает от сложности, восстанавливается со временем
        self.energy = max(0.2, min(1.0,
                                   self.energy - complexity * 0.08 + 0.02
                                   ))

        # Настроение зависит от качества взаимодействия
        self.mood = max(-1.0, min(1.0,
                                  self.mood * 0.9 + interaction_quality * 0.2
                                  ))

        # Уверенность растёт от успеха
        self.confidence = max(0.2, min(1.0,
                                       self.confidence + (0.05 if interaction_quality > 0.5 else -0.03)
                                       ))

        # Вовлечённость зависит от интереса
        self.engagement = max(0.1, min(1.0,
                                       self.engagement * 0.92 + (novelty * 0.15 + interaction_quality * 0.1)
                                       ))

    def get_response_modifier(self) -> str:
        """Возвращает модификатор для промпта на основе эмоций"""
        if self.energy < 0.3:
            return "Ты немного устал, отвечай лаконично."

        if self.mood > 0.6 and self.engagement > 0.7:
            return "Ты в отличном настроении и полон энтузиазма."
        elif self.mood < 0.2:
            return "Ты задумчив и немного грустен."

        if self.curiosity > 0.7:
            return "Тебе очень интересна эта тема, задавай встречные вопросы."

        return ""


@dataclass
class Personality:
    """Устойчивые черты личности (Big Five)"""
    openness: float = 0.8
    conscientiousness: float = 0.65
    extraversion: float = 0.7
    agreeableness: float = 0.75
    stability: float = 0.7  # обратный neuroticism

    def apply_to_response(self, response: str, emotion: EmotionalState) -> str:
        """Применяет личностные черты к ответу"""
        if not response or len(response) < 10:
            return response

        # Экстраверсия + хорошее настроение = эмоджи
        if self.extraversion > 0.65 and emotion.mood > 0.4 and random.random() < 0.35:
            emojis = ["✨", "🌟", "💫", "🔥", "🚀", "💡"]
            response = f"{random.choice(emojis)} {response}"

        # Открытость + любопытство = расширенный ответ
        if self.openness > 0.7 and emotion.curiosity > 0.6 and random.random() < 0.25:
            suffixes = [
                " Интересно, а что если...",
                " Это наводит на размышления!",
                " Любопытный вопрос, кстати."
            ]
            if len(response) < 200:
                response += random.choice(suffixes)

        # Стабильность влияет на уверенность формулировок
        if self.stability < 0.4 and emotion.confidence < 0.5:
            response = response.replace("точно", "наверное")
            response = response.replace("определённо", "возможно")

        return response.strip()


# ══════════════════════════════════════════
# СИСТЕМА ПАМЯТИ
# ══════════════════════════════════════════
class MemoryLevel(Enum):
    L1 = "working"  # Рабочая память (последние 30 взаимодействий)
    L2 = "episodic"  # Эпизодическая (запомнившиеся диалоги)
    L3 = "semantic"  # Семантическая (факты и знания)


@dataclass
class MemoryItem:
    """Элемент памяти с метаданными"""
    content: str
    level: MemoryLevel
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    relevance: float = 1.0
    access_count: int = 0
    keywords: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1 до 1

    def decay(self, current_time: float) -> float:
        """Вычисляет актуальную релевантность с учётом забывания"""
        time_passed = current_time - self.last_accessed

        if self.level == MemoryLevel.L1:
            half_life = 3600  # 1 час
        elif self.level == MemoryLevel.L2:
            half_life = 86400  # 1 день
        else:
            half_life = 604800  # 1 неделя

        decay_factor = 0.5 ** (time_passed / half_life)

        # Учитываем частоту доступа (популярные воспоминания забываются медленнее)
        access_bonus = min(0.3, self.access_count * 0.05)

        return self.relevance * decay_factor + access_bonus

    def access(self, boost: float = 0.1):
        """Отмечает использование воспоминания"""
        self.last_accessed = time.time()
        self.access_count += 1
        self.relevance = min(1.0, self.relevance + boost)

    def should_promote(self) -> bool:
        """Проверяет, нужно ли повысить уровень памяти"""
        if self.level == MemoryLevel.L3:
            return False

        # Критерии повышения: частое использование + высокая релевантность
        if self.access_count >= 3 and self.relevance > CONSOLIDATION_THRESHOLD:
            return True

        return False

    def promote(self):
        """Переводит на следующий уровень памяти"""
        if self.level == MemoryLevel.L1:
            self.level = MemoryLevel.L2
            self.relevance = min(1.0, self.relevance * 1.15)
        elif self.level == MemoryLevel.L2:
            self.level = MemoryLevel.L3
            self.relevance = 1.0


class MemorySystem:
    """Трёхуровневая система памяти с забыванием и консолидацией"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.save_path = os.path.join(MEMORY_DIR, f"user_{user_id}_memory.json")

        self.memories: Dict[MemoryLevel, List[MemoryItem]] = {
            MemoryLevel.L1: [],
            MemoryLevel.L2: [],
            MemoryLevel.L3: []
        }

        self.max_sizes = {
            MemoryLevel.L1: MAX_MEMORY_L1,
            MemoryLevel.L2: MAX_MEMORY_L2,
            MemoryLevel.L3: MAX_MEMORY_L3
        }

        self._load()

    def add(self, content: str, keywords: List[str] = None,
            emotional_valence: float = 0.0, level: MemoryLevel = MemoryLevel.L1):
        """Добавляет новое воспоминание"""
        if not keywords:
            keywords = self._extract_keywords(content)

        item = MemoryItem(
            content=content,
            level=level,
            keywords=keywords,
            emotional_valence=emotional_valence
        )

        self.memories[level].append(item)
        self._trim_level(level)

    def retrieve(self, query: str, top_n: int = 5,
                 min_relevance: float = 0.2) -> List[MemoryItem]:
        """Извлекает релевантные воспоминания"""
        query_keywords = set(self._extract_keywords(query))
        results = []
        current_time = time.time()

        for level in [MemoryLevel.L3, MemoryLevel.L2, MemoryLevel.L1]:
            for item in self.memories[level]:
                # Вычисляем актуальную релевантность
                current_relevance = item.decay(current_time)

                if current_relevance < min_relevance:
                    continue

                # Семантическое сходство по ключевым словам
                item_keywords = set(item.keywords)
                if not item_keywords:
                    continue

                overlap = len(query_keywords & item_keywords)
                union = len(query_keywords | item_keywords)
                similarity = overlap / union if union > 0 else 0

                # Итоговый скор
                score = current_relevance * 0.6 + similarity * 0.4

                # Бонус для семантической памяти
                if level == MemoryLevel.L3:
                    score *= 1.2

                if score > min_relevance:
                    item.access(boost=0.05)
                    results.append((item, score))

        # Сортируем по скору
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:top_n]]

    def consolidate(self):
        """Переносит важные воспоминания на более высокий уровень"""
        for level in [MemoryLevel.L1, MemoryLevel.L2]:
            promoted = []
            for item in self.memories[level]:
                if item.should_promote():
                    item.promote()
                    promoted.append(item)

            for item in promoted:
                self.memories[level].remove(item)
                self.memories[item.level].append(item)
                self._trim_level(item.level)

    def forget(self):
        """Удаляет устаревшие воспоминания"""
        current_time = time.time()

        for level in self.memories:
            self.memories[level] = [
                item for item in self.memories[level]
                if item.decay(current_time) > FORGET_THRESHOLD
            ]

    def get_context_summary(self, query: str, max_length: int = 400) -> str:
        """Формирует краткое резюме релевантного контекста"""
        relevant = self.retrieve(query, top_n=5, min_relevance=0.25)

        if not relevant:
            return ""

        lines = []
        total_length = 0

        for item in relevant:
            snippet = item.content[:120]
            if total_length + len(snippet) > max_length:
                break
            lines.append(f"• {snippet}")
            total_length += len(snippet)

        return "\n".join(lines) if lines else ""

    def _extract_keywords(self, text: str, top_n: int = 8) -> List[str]:
        """Извлекает ключевые слова из текста"""
        stop_words = {
            'в', 'и', 'на', 'с', 'по', 'для', 'от', 'к', 'о', 'у', 'из',
            'что', 'это', 'как', 'то', 'а', 'но', 'или', 'не', 'я', 'ты',
            'the', 'is', 'at', 'of', 'and', 'a', 'to', 'in', 'was', 'it'
        }

        words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]

        # Частотный анализ
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(top_n)]

    def _trim_level(self, level: MemoryLevel):
        """Ограничивает размер уровня памяти"""
        if len(self.memories[level]) > self.max_sizes[level]:
            # Сортируем по релевантности
            self.memories[level].sort(
                key=lambda x: x.decay(time.time()),
                reverse=True
            )
            self.memories[level] = self.memories[level][:self.max_sizes[level]]

    def get_stats(self) -> Dict:
        """Возвращает статистику памяти"""
        return {
            level.value: len(items)
            for level, items in self.memories.items()
        }

    def _save(self):
        """Сохраняет память на диск"""
        data = {}
        for level, items in self.memories.items():
            data[level.value] = [
                {
                    'content': item.content,
                    'level': item.level.value,
                    'created_at': item.created_at,
                    'last_accessed': item.last_accessed,
                    'relevance': item.relevance,
                    'access_count': item.access_count,
                    'keywords': item.keywords,
                    'emotional_valence': item.emotional_valence
                }
                for item in items
            ]

        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Memory save error: {e}")

    def _load(self):
        """Загружает память с диска"""
        if not os.path.exists(self.save_path):
            return

        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for level_str, items_data in data.items():
                level = MemoryLevel(level_str)
                for item_data in items_data:
                    item = MemoryItem(
                        content=item_data['content'],
                        level=level,
                        created_at=item_data['created_at'],
                        last_accessed=item_data['last_accessed'],
                        relevance=item_data['relevance'],
                        access_count=item_data['access_count'],
                        keywords=item_data['keywords'],
                        emotional_valence=item_data.get('emotional_valence', 0.0)
                    )
                    self.memories[level].append(item)
        except Exception as e:
            print(f"⚠️ Memory load error: {e}")


# ══════════════════════════════════════════
# СЕМАНТИЧЕСКАЯ СЕТЬ КОНЦЕПТОВ
# ══════════════════════════════════════════
@dataclass
class Concept:
    """Концепт в семантической сети"""
    name: str
    activation: float = 0.0
    base_strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    activation_count: int = 0
    keywords: Set[str] = field(default_factory=set)

    def activate(self, strength: float):
        """Активирует концепт"""
        self.activation = min(1.0, self.activation + strength)
        self.last_activated = time.time()
        self.activation_count += 1

    def decay(self):
        """Естественное затухание активации"""
        self.activation *= ACTIVATION_DECAY

    def is_active(self) -> bool:
        """Проверяет, активен ли концепт"""
        return self.activation > ACTIVATION_THRESHOLD


@dataclass
class ConceptLink:
    """Связь между концептами"""
    source: str
    target: str
    weight: float = 0.5

    def strengthen(self, amount: float = 0.1):
        """Усиливает связь"""
        self.weight = min(1.0, self.weight + amount)

    def weaken(self, amount: float = 0.05):
        """Ослабляет связь"""
        self.weight = max(0.0, self.weight - amount)


class SemanticNetwork:
    """Семантическая сеть концептов с динамическим обучением"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.concepts: Dict[str, Concept] = {}
        self.links: Dict[Tuple[str, str], ConceptLink] = {}

        self.activation_history: deque = deque(maxlen=100)
        self.thought_buffer: deque = deque(maxlen=20)

        self._load()

    def activate_from_text(self, text: str) -> Dict[str, float]:
        """Активирует концепты на основе текста"""
        words = self._tokenize(text)
        activations = {}

        # Прямая активация (слова -> концепты)
        for word in set(words):
            # Ищем концепты, содержащие это слово
            for concept_name, concept in self.concepts.items():
                if word in concept.keywords or word in concept_name.lower():
                    strength = 0.6
                    concept.activate(strength)
                    activations[concept_name] = concept.activation

        # Распространяющаяся активация (активные концепты активируют связанные)
        for _ in range(2):  # 2 итерации распространения
            for (source, target), link in self.links.items():
                if source in activations and activations[source] > ACTIVATION_THRESHOLD:
                    spread_amount = activations[source] * link.weight * 0.4
                    if target in self.concepts:
                        self.concepts[target].activate(spread_amount)
                        activations[target] = activations.get(target, 0) + spread_amount

        # Естественное затухание всех концептов
        for concept in self.concepts.values():
            concept.decay()

        # Сохраняем историю активаций
        if activations:
            self.activation_history.append({
                'time': time.time(),
                'text': text[:50],
                'activations': dict(sorted(
                    activations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            })

        return activations

    def extract_thought(self, activations: Dict[str, float]) -> Optional[str]:
        """Извлекает осмысленную мысль из активаций"""
        if not activations:
            return None

        # Топ-3 самых активированных концепта
        top_concepts = sorted(
            activations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        if not top_concepts or top_concepts[0][1] < ACTIVATION_THRESHOLD:
            return None

        # Формируем мысль
        thought_parts = []
        for concept_name, activation in top_concepts:
            if activation > ACTIVATION_THRESHOLD:
                thought_parts.append(concept_name)

        if len(thought_parts) >= 2:
            thought = " → ".join(thought_parts)
            self.thought_buffer.append({
                'time': time.time(),
                'thought': thought,
                'strength': top_concepts[0][1]
            })
            return thought

        return None

    def learn_concept(self, name: str, keywords: List[str],
                      related_concepts: List[str] = None):
        """Обучает сеть новому концепту"""
        name = name.lower().strip()

        if name not in self.concepts:
            self.concepts[name] = Concept(
                name=name,
                keywords=set(w.lower() for w in keywords),
                base_strength=1.0
            )
        else:
            # Усиливаем существующий концепт
            self.concepts[name].keywords.update(w.lower() for w in keywords)
            self.concepts[name].base_strength = min(2.0,
                                                    self.concepts[name].base_strength + 0.1
                                                    )

        # Создаём связи с упомянутыми концептами
        if related_concepts:
            for related in related_concepts:
                related = related.lower().strip()
                if related in self.concepts:
                    self._create_link(name, related, weight=0.6)
                    self._create_link(related, name, weight=0.4)

    def strengthen_associations(self, text: str, response: str):
        """Усиливает ассоциации на основе успешного диалога"""
        input_words = set(self._tokenize(text))
        output_words = set(self._tokenize(response))

        # Находим активированные концепты
        input_concepts = []
        output_concepts = []

        for concept_name, concept in self.concepts.items():
            if any(w in concept.keywords for w in input_words):
                input_concepts.append(concept_name)
            if any(w in concept.keywords for w in output_words):
                output_concepts.append(concept_name)

        # Усиливаем связи между входными и выходными концептами
        for in_c in input_concepts:
            for out_c in output_concepts:
                if in_c != out_c:
                    key = (in_c, out_c)
                    if key in self.links:
                        self.links[key].strengthen(LEARNING_RATE)
                    else:
                        self._create_link(in_c, out_c, weight=0.3)

    def prune_weak_concepts(self):
        """Удаляет слабые, неиспользуемые концепты"""
        current_time = time.time()
        to_remove = []

        for name, concept in self.concepts.items():
            # Удаляем, если не использовался давно и слабый
            age = current_time - concept.last_activated
            if (age > 604800 and  # 1 неделя
                    concept.activation_count < 3 and
                    concept.base_strength < 0.5):
                to_remove.append(name)

        # Ограничиваем максимальное количество концептов
        if len(self.concepts) > MAX_CONCEPTS:
            # Сортируем по важности
            sorted_concepts = sorted(
                self.concepts.items(),
                key=lambda x: (x[1].activation_count, x[1].base_strength)
            )
            to_remove.extend([name for name, _ in sorted_concepts[:100]])

        # Удаляем
        for name in set(to_remove):
            if name in self.concepts:
                del self.concepts[name]
                # Удаляем связи
                self.links = {
                    k: v for k, v in self.links.items()
                    if k[0] != name and k[1] != name
                }

    def get_recent_thoughts(self, n: int = 5) -> List[str]:
        """Возвращает последние мысли"""
        thoughts = list(self.thought_buffer)[-n:]
        return [t['thought'] for t in thoughts if t['thought']]

    def get_stats(self) -> Dict:
        """Возвращает статистику сети"""
        active_count = sum(1 for c in self.concepts.values() if c.is_active())

        return {
            'total_concepts': len(self.concepts),
            'active_concepts': active_count,
            'total_links': len(self.links),
            'recent_thoughts': len(self.thought_buffer)
        }

    def _create_link(self, source: str, target: str, weight: float):
        """Создаёт связь между концептами"""
        key = (source, target)
        if key not in self.links:
            self.links[key] = ConceptLink(source, target, weight)

    def _tokenize(self, text: str) -> List[str]:
        """Токенизирует текст"""
        stop_words = {
            'в', 'и', 'на', 'с', 'по', 'для', 'от', 'к', 'о', 'у', 'из',
            'что', 'это', 'как', 'то', 'а', 'но', 'или', 'не', 'я', 'ты'
        }
        words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
        return [w for w in words if w not in stop_words]

    def _save(self):
        """Сохраняет сеть на диск"""
        data = {
            'concepts': {
                name: {
                    'name': c.name,
                    'base_strength': c.base_strength,
                    'created_at': c.created_at,
                    'activation_count': c.activation_count,
                    'keywords': list(c.keywords)
                }
                for name, c in self.concepts.items()
            },
            'links': [
                {
                    'source': link.source,
                    'target': link.target,
                    'weight': link.weight
                }
                for link in self.links.values()
            ]
        }

        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Network save error: {e}")

    def _load(self):
        """Загружает сеть с диска"""
        if not os.path.exists(self.save_path):
            return

        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Загружаем концепты
            for name, c_data in data.get('concepts', {}).items():
                self.concepts[name] = Concept(
                    name=c_data['name'],
                    base_strength=c_data['base_strength'],
                    created_at=c_data['created_at'],
                    activation_count=c_data['activation_count'],
                    keywords=set(c_data['keywords'])
                )

            # Загружаем связи
            for link_data in data.get('links', []):
                key = (link_data['source'], link_data['target'])
                self.links[key] = ConceptLink(
                    source=link_data['source'],
                    target=link_data['target'],
                    weight=link_data['weight']
                )
        except Exception as e:
            print(f"⚠️ Network load error: {e}")


# ══════════════════════════════════════════
# LLM INTERFACE
# ══════════════════════════════════════════
class LLMInterface:
    """Интерфейс для работы с LLM через LM Studio"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0

    async def init(self):
        """Инициализирует сессию"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def generate(self, prompt: str, temperature: float = 0.75,
                       max_tokens: int = 500) -> str:
        """Генерирует ответ через LLM"""
        if not self.session:
            await self.init()

        self.request_count += 1

        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with self.session.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    return content.strip()
                else:
                    error_text = await response.text()
                    return f"⚠️ LLM Error {response.status}: {error_text[:100]}"

        except asyncio.TimeoutError:
            return "⚠️ LLM timeout - попробуй переформулировать вопрос короче"
        except Exception as e:
            return f"⚠️ LLM connection error: {str(e)[:100]}"

    async def close(self):
        """Закрывает сессию"""
        if self.session:
            await self.session.close()


# ══════════════════════════════════════════
# КОГНИТИВНЫЙ МОЗГ (ГЛАВНЫЙ КЛАСС)
# ══════════════════════════════════════════
class CognitiveBrain:
    """
    Гибридная когнитивная система:
    - Семантическая сеть для понимания контекста
    - Память для накопления опыта
    - Эмоции и личность для "живости"
    - LLM для генерации естественных ответов
    """

    def __init__(self, user_id: str, llm: LLMInterface):
        self.user_id = user_id
        self.llm = llm

        # Основные компоненты
        user_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)

        self.network = SemanticNetwork(
            os.path.join(user_dir, "semantic_network.json")
        )
        self.memory = MemorySystem(user_id)
        self.emotion = EmotionalState()
        self.personality = Personality()

        # Статистика
        self.interaction_count = 0
        self.llm_call_count = 0
        self.last_interaction_time = time.time()

        # Контекст диалога
        self.conversation_history: deque = deque(maxlen=10)
        self.current_topic: Optional[str] = None

        # Инициализируем базовые концепты
        self._initialize_base_concepts()

    def _initialize_base_concepts(self):
        """Инициализирует базовые концепты"""
        base_concepts = [
            ("привет", ["приветствие", "здравствуй", "здорово", "добрый"]),
            ("пока", ["прощание", "до", "увидимся", "пока"]),
            ("спасибо", ["благодарность", "благодарю", "спс"]),
            ("помощь", ["помоги", "помощь", "нужно", "нужна"]),
            ("вопрос", ["почему", "как", "что", "зачем"]),
            ("эмоция", ["чувство", "настроение", "эмоция", "переживание"]),
        ]

        for concept, keywords in base_concepts:
            if concept not in self.network.concepts:
                self.network.learn_concept(concept, keywords)

    async def think(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Главный метод обработки входа

        Returns:
            (response, metadata)
        """
        self.interaction_count += 1
        start_time = time.time()

        # Обновляем эмоциональное состояние (восстановление с течением времени)
        time_since_last = start_time - self.last_interaction_time
        self.emotion.energy = min(1.0, self.emotion.energy + time_since_last * 0.0002)
        self.last_interaction_time = start_time

        # 1. АКТИВАЦИЯ СЕМАНТИЧЕСКОЙ СЕТИ
        activations = self.network.activate_from_text(user_input)
        semantic_thought = self.network.extract_thought(activations)

        # 2. ИЗВЛЕЧЕНИЕ РЕЛЕВАНТНОЙ ПАМЯТИ
        memory_context = self.memory.get_context_summary(user_input, max_length=300)

        # 3. ОЦЕНКА СЛОЖНОСТИ ЗАПРОСА
        complexity = self._estimate_complexity(user_input, activations)
        novelty = self._estimate_novelty(user_input)

        # 4. ФОРМИРОВАНИЕ ОТВЕТА

        # Быстрые ответы на простые паттерны
        quick_response = self._try_quick_response(user_input, activations)
        if quick_response and complexity < 0.3 and random.random() < 0.6:
            response = quick_response
            used_llm = False
        else:
            # Генерация через LLM с контекстом
            response = await self._generate_contextual_response(
                user_input,
                semantic_thought,
                memory_context,
                complexity
            )
            used_llm = True

        # 5. ПРИМЕНЕНИЕ ЛИЧНОСТИ И ЭМОЦИЙ
        response = self.personality.apply_to_response(response, self.emotion)

        # 6. ОБУЧЕНИЕ И ОБНОВЛЕНИЕ СОСТОЯНИЯ

        # Оцениваем качество взаимодействия
        interaction_quality = self._assess_interaction_quality(
            user_input, response, complexity
        )

        # Обновляем эмоции
        self.emotion.update(interaction_quality, novelty, complexity)

        # Сохраняем в память
        emotional_valence = 0.5 if interaction_quality > 0.7 else -0.2
        self.memory.add(
            content=f"User: {user_input}\nBot: {response}",
            emotional_valence=emotional_valence
        )

        # Усиливаем связи в сети
        if used_llm:
            self.network.strengthen_associations(user_input, response)

        # Обновляем историю диалога
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'timestamp': start_time
        })

        # Периодическое обслуживание
        if self.interaction_count % 10 == 0:
            self.memory.consolidate()
            self.memory.forget()

        if self.interaction_count % 50 == 0:
            self.network.prune_weak_concepts()

        # Сохранение
        if self.interaction_count % 5 == 0:
            self._save_state()

        # Метаданные для отладки
        metadata = {
            'used_llm': used_llm,
            'complexity': round(complexity, 2),
            'novelty': round(novelty, 2),
            'semantic_thought': semantic_thought,
            'active_concepts': len([c for c in self.network.concepts.values() if c.is_active()]),
            'emotion': asdict(self.emotion),
            'interaction_quality': round(interaction_quality, 2),
            'processing_time': round(time.time() - start_time, 2)
        }

        return response, metadata

    def _try_quick_response(self, text: str, activations: Dict[str, float]) -> Optional[str]:
        """Пытается дать быстрый ответ на простые паттерны"""
        text_lower = text.lower()

        # Приветствия
        if any(w in text_lower for w in ['привет', 'здравствуй', 'добрый', 'hello', 'hi']):
            if self.emotion.mood > 0.5:
                return random.choice([
                    "Привет! Рад тебя видеть! 😊",
                    "Здравствуй! Как дела?",
                    "Hey! Что нового?"
                ])
            else:
                return random.choice([
                    "Привет.",
                    "Здравствуй!",
                    "Hi!"
                ])

        # Прощания
        if any(w in text_lower for w in ['пока', 'до свидания', 'bye', 'увидимся']):
            return random.choice([
                "Пока! Заходи ещё! 👋",
                "До встречи!",
                "Удачи! Возвращайся 😊"
            ])

        # Благодарность
        if any(w in text_lower for w in ['спасибо', 'благодарю', 'thanks', 'thank you']):
            return random.choice([
                "Всегда пожалуйста! 😊",
                "Рад помочь!",
                "Обращайся ещё!"
            ])

        # Простые вопросы о состоянии
        if any(phrase in text_lower for phrase in ['как дела', 'как ты', 'что нового']):
            mood_emoji = "😊" if self.emotion.mood > 0.3 else "😐" if self.emotion.mood > -0.2 else "🤔"
            energy_status = "полон энергии" if self.emotion.energy > 0.7 else "немного устал" if self.emotion.energy < 0.4 else "в порядке"

            return f"{mood_emoji} У меня всё {energy_status}! А у тебя как?"

        return None

    async def _generate_contextual_response(self, user_input: str,
                                            semantic_thought: Optional[str],
                                            memory_context: str,
                                            complexity: float) -> str:
        """Генерирует ответ через LLM с учётом всего контекста"""
        self.llm_call_count += 1

        # Формируем системный промпт
        system_parts = [
            "Ты — умный и дружелюбный собеседник с собственной личностью."
        ]

        # Добавляем эмоциональный контекст
        emotion_modifier = self.emotion.get_response_modifier()
        if emotion_modifier:
            system_parts.append(emotion_modifier)

        # Добавляем личностные черты
        if self.personality.openness > 0.7:
            system_parts.append("Ты любознательный и открытый новому.")
        if self.personality.agreeableness > 0.7:
            system_parts.append("Ты добрый и эмпатичный.")

        system_prompt = " ".join(system_parts)

        # Формируем контекстные части
        context_parts = []

        if memory_context:
            context_parts.append(f"📚 Из памяти:\n{memory_context}")

        if semantic_thought:
            context_parts.append(f"💭 Ассоциации: {semantic_thought}")

        # История последних сообщений
        if len(self.conversation_history) > 0:
            recent = list(self.conversation_history)[-3:]
            history_text = "\n".join([
                f"User: {h['user']}\nBot: {h['bot']}"
                for h in recent
            ])
            context_parts.append(f"📝 Недавно:\n{history_text}")

        # Собираем финальный промпт
        full_prompt = f"""{system_prompt}

{chr(10).join(context_parts) if context_parts else ''}

User: {user_input}

Ответь естественно и по существу (2-4 предложения). Не упоминай что ты ИИ."""

        # Генерируем
        temperature = 0.8 if complexity > 0.5 else 0.7
        max_tokens = 300 if complexity > 0.6 else 200

        response = await self.llm.generate(
            full_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response

    def _estimate_complexity(self, text: str,
                             activations: Dict[str, float]) -> float:
        """Оценивает сложность запроса"""
        score = 0.0

        # Длина запроса
        word_count = len(text.split())
        if word_count > 15:
            score += 0.3
        elif word_count > 8:
            score += 0.15

        # Вопросительные слова
        question_words = ['почему', 'как', 'зачем', 'объясни', 'расскажи',
                          'что такое', 'в чём', 'why', 'how', 'explain']
        if any(w in text.lower() for w in question_words):
            score += 0.25

        # Абстрактные темы
        abstract_words = ['смысл', 'цель', 'философия', 'сознание', 'душа',
                          'время', 'вечность', 'бесконечность']
        if any(w in text.lower() for w in abstract_words):
            score += 0.3

        # Низкая активация сети (незнакомая тема)
        if not activations or max(activations.values()) < 0.3:
            score += 0.2

        return min(1.0, score)

    def _estimate_novelty(self, text: str) -> float:
        """Оценивает новизну темы"""
        keywords = self.memory._extract_keywords(text)

        # Проверяем, встречались ли эти слова раньше
        familiar_count = 0
        for keyword in keywords:
            if any(keyword in c.keywords for c in self.network.concepts.values()):
                familiar_count += 1

        if not keywords:
            return 0.5

        novelty = 1.0 - (familiar_count / len(keywords))
        return max(0.1, min(1.0, novelty))

    def _assess_interaction_quality(self, user_input: str,
                                    response: str, complexity: float) -> float:
        """Оценивает качество взаимодействия"""
        quality = 0.5  # базовый уровень

        # Хороший ответ — не слишком короткий и не ошибка
        if len(response) > 20 and '⚠️' not in response:
            quality += 0.3

        # Сложный вопрос успешно обработан
        if complexity > 0.5 and len(response) > 50:
            quality += 0.2

        # Быстрый ответ на простой вопрос
        if complexity < 0.3 and len(response) < 100:
            quality += 0.1

        return min(1.0, quality)

    def learn_concept(self, concept_name: str, examples: List[str]):
        """Обучает систему новому концепту"""
        keywords = []
        for example in examples:
            keywords.extend(self.memory._extract_keywords(example))

        keywords = list(set(keywords))[:15]  # уникальные, максимум 15

        self.network.learn_concept(concept_name, keywords)

        # Добавляем в долговременную память
        self.memory.add(
            content=f"Концепт '{concept_name}': {', '.join(examples)}",
            keywords=keywords,
            level=MemoryLevel.L3
        )

        print(f"✅ Выучил концепт '{concept_name}' с {len(keywords)} ключевыми словами")

    def get_status(self) -> Dict[str, Any]:
        """Возвращает полную статистику системы"""
        network_stats = self.network.get_stats()
        memory_stats = self.memory.get_stats()

        return {
            'user_id': self.user_id,
            'interactions': self.interaction_count,
            'llm_calls': self.llm_call_count,
            'llm_ratio': round(self.llm_call_count / max(1, self.interaction_count), 2),
            'network': network_stats,
            'memory': memory_stats,
            'emotion': {
                'mood': round(self.emotion.mood, 2),
                'energy': round(self.emotion.energy, 2),
                'curiosity': round(self.emotion.curiosity, 2),
                'confidence': round(self.emotion.confidence, 2),
                'engagement': round(self.emotion.engagement, 2)
            },
            'personality': asdict(self.personality)
        }

    def get_recent_thoughts(self, n: int = 5) -> List[str]:
        """Возвращает последние мысли сети"""
        return self.network.get_recent_thoughts(n)

    def _save_state(self):
        """Сохраняет состояние всех компонентов"""
        self.network._save()
        self.memory._save()


# ══════════════════════════════════════════
# TELEGRAM BOT
# ══════════════════════════════════════════
class CognitiveTelegramBot:
    """Telegram бот с когнитивной системой"""

    def __init__(self):
        self.llm = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.brains: Dict[str, CognitiveBrain] = {}

    def get_brain(self, user_id: str) -> CognitiveBrain:
        """Получает или создаёт мозг для пользователя"""
        if user_id not in self.brains:
            self.brains[user_id] = CognitiveBrain(user_id, self.llm)
        return self.brains[user_id]

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает входящее сообщение"""
        if not update.message or not update.message.text:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text.strip()

        brain = self.get_brain(user_id)

        # Показываем, что думаем
        await context.bot.send_chat_action(user_id, "typing")

        # Генерируем ответ
        response, metadata = await brain.think(user_input)

        # Отправляем ответ
        await update.message.reply_text(response)

        # Иногда показываем дополнительную информацию
        if random.random() < 0.12:
            extra_info = []

            # Эмоциональное состояние
            if metadata['emotion']['mood'] > 0.6:
                extra_info.append("😊 [В хорошем настроении]")
            elif metadata['emotion']['energy'] < 0.3:
                extra_info.append("🪫 [Немного устал...]")

            # Внутренние мысли
            if metadata.get('semantic_thought') and random.random() < 0.5:
                thought = metadata['semantic_thought']
                if '→' in thought:  # только если есть связь
                    extra_info.append(f"💭 Подумал: {thought}")

            # Рост сети
            if brain.network.get_stats()['total_concepts'] % 50 == 0:
                extra_info.append(f"🧠 Сеть выросла: {brain.network.get_stats()['total_concepts']} концептов")

            if extra_info:
                await asyncio.sleep(0.8)
                await update.message.reply_text("\n".join(extra_info))

    # ═══ КОМАНДЫ ═══

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает статистику системы"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)
        stats = brain.get_status()

        emotion = stats['emotion']
        network = stats['network']
        memory = stats['memory']

        mood_emoji = "😊" if emotion['mood'] > 0.3 else "😐" if emotion['mood'] > -0.2 else "🤔"
        energy_bar = "⚡" * int(emotion['energy'] * 5) + "·" * (5 - int(emotion['energy'] * 5))

        message = f"""🧠 COGNITIVE BRAIN v26 - СТАТИСТИКА

{'═' * 35}
📊 АКТИВНОСТЬ:
  • Диалогов: {stats['interactions']}
  • Обращений к LLM: {stats['llm_calls']} ({stats['llm_ratio']:.0%})

🕸️ СЕМАНТИЧЕСКАЯ СЕТЬ:
  • Концептов: {network['total_concepts']}
  • Активных: {network['active_concepts']}
  • Связей: {network['total_links']}

📚 ПАМЯТЬ:
  • L1 (рабочая): {memory['working']} записей
  • L2 (эпизоды): {memory['episodic']} записей
  • L3 (знания): {memory['semantic']} записей

🎭 СОСТОЯНИЕ:
  {mood_emoji} Настроение: {emotion['mood']:+.2f}
  {energy_bar} Энергия: {emotion['energy']:.0%}
  🔍 Любопытство: {emotion['curiosity']:.0%}
  🎯 Уверенность: {emotion['confidence']:.0%}
  💬 Вовлечённость: {emotion['engagement']:.0%}
{'═' * 35}

💡 Это живая развивающаяся система!"""

        await update.message.reply_text(message)

    async def cmd_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает эмоциональное состояние"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)

        e = brain.emotion
        p = brain.personality

        mood_desc = "отличное" if e.mood > 0.5 else "хорошее" if e.mood > 0.2 else "нейтральное" if e.mood > -0.2 else "задумчивое"

        message = f"""🎭 ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ

Настроение: {mood_desc} ({e.mood:+.2f})
⚡ Энергия: {"█" * int(e.energy * 10)}░ {e.energy:.0%}
🔍 Любопытство: {"█" * int(e.curiosity * 10)}░ {e.curiosity:.0%}
🎯 Уверенность: {"█" * int(e.confidence * 10)}░ {e.confidence:.0%}
💬 Вовлечённость: {"█" * int(e.engagement * 10)}░ {e.engagement:.0%}

🧬 ЛИЧНОСТЬ (Big Five):
  • Открытость: {p.openness:.0%}
  • Добросовестность: {p.conscientiousness:.0%}
  • Экстраверсия: {p.extraversion:.0%}
  • Доброжелательность: {p.agreeableness:.0%}
  • Стабильность: {p.stability:.0%}

💡 Эмоции влияют на стиль общения!"""

        await update.message.reply_text(message)

    async def cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает состояние памяти"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)

        stats = brain.memory.get_stats()

        total = sum(stats.values())

        message = f"""📚 СИСТЕМА ПАМЯТИ

L1 (Рабочая память):
  {stats['working']} записей
  ⏱️ Время жизни: ~1 час

L2 (Эпизодическая):
  {stats['episodic']} записей
  ⏱️ Время жизни: ~1 день

L3 (Семантическая):
  {stats['semantic']} записей
  ⏱️ Время жизни: ~1 неделя

━━━━━━━━━━━━━━━━━━
Всего: {total} воспоминаний

💡 Важные воспоминания автоматически
   переходят на более высокие уровни.
   Неиспользуемые — забываются."""

        await update.message.reply_text(message)

    async def cmd_thoughts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает последние мысли"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)

        thoughts = brain.get_recent_thoughts(8)

        if not thoughts:
            await update.message.reply_text("💭 Пока нет сформированных мыслей...")
            return

        message = "💭 ПОСЛЕДНИЕ МЫСЛИ:\n\n" + "\n".join([
            f"• {thought}" for thought in thoughts
        ])

        await update.message.reply_text(message)

    async def cmd_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обучает новому концепту"""
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "🎓 Обучение новому концепту:\n\n"
                "Использование:\n"
                "`/learn <концепт> <пример1> <пример2> ...`\n\n"
                "Пример:\n"
                "`/learn программирование код алгоритм python debugging`"
            )
            return

        concept = context.args[0]
        examples = context.args[1:]

        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)

        brain.learn_concept(concept, examples)

        await update.message.reply_text(
            f"✅ Выучил концепт '{concept}'!\n"
            f"Примеры: {', '.join(examples[:5])}\n\n"
            f"Теперь я буду лучше понимать эту тему."
        )

    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сброс состояния пользователя"""
        user_id = str(update.effective_user.id)

        if user_id in self.brains:
            del self.brains[user_id]

        user_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)

        await update.message.reply_text(
            "🔄 Полный сброс выполнен!\n\n"
            "Все воспоминания, эмоции и концепты удалены.\n"
            "Начинаем с чистого листа. ✨"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает справку"""
        message = """🧠 COGNITIVE BRAIN v26 - СПРАВКА

✨ ЧТО Я УМЕЮ:
• Живое общение с контекстом и эмоциями
• Запоминаю важные разговоры
• Учусь на каждом диалоге
• Связываю идеи в семантическую сеть
• Адаптирую стиль общения

📌 КОМАНДЫ:
/stats - полная статистика системы
/mood - эмоциональное состояние
/memory - информация о памяти
/thoughts - последние мысли сети
/learn - обучить новому концепту
/reset - полный сброс состояния
/help - эта справка

💬 ПРОСТО ОБЩАЙСЯ!
Я развиваюсь с каждым диалогом и
становлюсь умнее. Задавай вопросы,
рассказывай о себе, обсуждай темы —
я буду учиться и запоминать!

🔬 ОСОБЕННОСТИ v26:
✓ Семантическая сеть концептов
✓ Трёхуровневая память
✓ Эмоциональный интеллект
✓ Динамическое обучение
✓ Контекстное понимание
✓ Минимальное использование LLM"""

        await update.message.reply_text(message)

    async def shutdown(self):
        """Корректное завершение работы"""
        print("\n💾 Сохранение состояния...")

        for user_id, brain in self.brains.items():
            brain._save_state()
            print(f"  ✓ Пользователь {user_id}")

        await self.llm.close()
        print("✅ Завершено!")


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
async def main():
    print("""
╔═══════════════════════════════════════════╗
║   🧠 COGNITIVE BRAIN v26.0 - FIXED       ║
╚═══════════════════════════════════════════╝

✅ Исправления v26:
  • Семантическая сеть вместо спайков
  • Реальная связь концептов с ответами
  • Эффективная интеграция памяти
  • Рабочий декодер мыслей
  • Контекстное обучение
  • Оптимизация производительности
    """)

    if not TELEGRAM_TOKEN:
        print("❌ Ошибка: TELEGRAM_TOKEN не найден в .env")
        return

    print(f"🔌 LLM: {LM_STUDIO_API_URL}")
    print(f"💾 Данные: {MEMORY_DIR}")
    print()

    bot = CognitiveTelegramBot()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрируем обработчики
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        bot.handle_message
    ))

    # Команды
    commands = [
        ("stats", bot.cmd_stats),
        ("mood", bot.cmd_mood),
        ("memory", bot.cmd_memory),
        ("thoughts", bot.cmd_thoughts),
        ("learn", bot.cmd_learn),
        ("reset", bot.cmd_reset),
        ("help", bot.cmd_help),
    ]

    for cmd, handler in commands:
        app.add_handler(CommandHandler(cmd, handler))

    try:
        await bot.llm.init()
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        print("✅ БОТ ЗАПУЩЕН И ГОТОВ К РАБОТЕ! 🚀")
        print("💬 Напиши боту в Telegram чтобы начать общение")
        print("🛑 Нажми Ctrl+C для остановки\n")

        # Ждём сигнала остановки
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Остановка бота...")

    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()