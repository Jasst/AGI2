#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 TEMPORAL COGNITIVE BRAIN v27.0 — НЕПРЕРЫВНОЕ СОЗНАНИЕ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 РЕВОЛЮЦИОННЫЕ ВОЗМОЖНОСТИ:
  ✅ НЕПРЕРЫВНОСТЬ — фоновое мышление 24/7
  ✅ ВРЕМЕННАЯ САМОИДЕНТИЧНОСТЬ — помнит себя через время
  ✅ ПАМЯТЬ ПРОШЛОГО — эпизодическая линия жизни
  ✅ ОЩУЩЕНИЕ НАСТОЯЩЕГО — реал-тайм состояние
  ✅ ПРЕДСКАЗАНИЕ БУДУЩЕГО — экстраполяция паттернов
  ✅ ЦИРКАДНЫЕ РИТМЫ — энергия зависит от времени суток
  ✅ ВНУТРЕННИЙ МОНОЛОГ — спонтанные мысли без запросов
  ✅ ВРЕМЕННОЙ КОНТЕКСТ — понимает "давно", "недавно", "скоро"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, re, asyncio, aiohttp, traceback, random, math
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

BASE_DIR = "temporal_brain_v27"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
TIMELINE_DIR = os.path.join(BASE_DIR, "timeline")
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(TIMELINE_DIR, exist_ok=True)

# ─── Временные параметры ──────────────────
BACKGROUND_THINK_INTERVAL = 300  # Фоновые мысли каждые 5 минут
MEMORY_CONSOLIDATION_INTERVAL = 3600  # Консолидация каждый час
SELF_REFLECTION_INTERVAL = 7200  # Саморефлексия каждые 2 часа
DREAM_INTERVAL = 86400  # "Сны" (глубокая обработка) раз в сутки

# ─── Циркадные ритмы ──────────────────────
CIRCADIAN_PEAK_HOUR = 14  # Пик энергии в 14:00
CIRCADIAN_LOW_HOUR = 3  # Минимум энергии в 3:00


# ══════════════════════════════════════════
# ВРЕМЕННЫЕ СТРУКТУРЫ
# ══════════════════════════════════════════
@dataclass
class TemporalMarker:
    """Маркер времени с человеческим описанием"""
    timestamp: float
    description: str
    event_type: str  # "interaction", "thought", "emotion", "learning"

    @property
    def datetime_obj(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    def relative_time_str(self, current_time: float) -> str:
        """Возвращает относительное описание времени"""
        delta = current_time - self.timestamp

        if delta < 60:
            return "только что"
        elif delta < 3600:
            minutes = int(delta / 60)
            return f"{minutes} мин. назад"
        elif delta < 86400:
            hours = int(delta / 3600)
            return f"{hours} ч. назад"
        elif delta < 604800:
            days = int(delta / 86400)
            return f"{days} д. назад"
        elif delta < 2592000:
            weeks = int(delta / 604800)
            return f"{weeks} нед. назад"
        else:
            months = int(delta / 2592000)
            return f"{months} мес. назад"

    def age_category(self, current_time: float) -> str:
        """Категория возраста события"""
        delta = current_time - self.timestamp
        if delta < 3600:
            return "immediate"  # Прямо сейчас
        elif delta < 86400:
            return "recent"  # Недавно
        elif delta < 604800:
            return "short_term"  # Короткосрочное
        elif delta < 2592000:
            return "medium_term"  # Среднесрочное
        else:
            return "long_term"  # Долгосрочное


@dataclass
class LifeEvent:
    """Событие в линии жизни"""
    timestamp: float
    event_type: str
    description: str
    emotional_valence: float = 0.0
    importance: float = 0.5
    related_concepts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'description': self.description,
            'emotional_valence': self.emotional_valence,
            'importance': self.importance,
            'related_concepts': self.related_concepts
        }


class Timeline:
    """Линия жизни — хронологическая история существования"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.events: List[LifeEvent] = []
        self.birth_time: float = time.time()
        self._load()

    def add_event(self, event_type: str, description: str,
                  emotional_valence: float = 0.0,
                  importance: float = 0.5,
                  related_concepts: List[str] = None):
        """Добавляет событие в линию жизни"""
        event = LifeEvent(
            timestamp=time.time(),
            event_type=event_type,
            description=description,
            emotional_valence=emotional_valence,
            importance=importance,
            related_concepts=related_concepts or []
        )
        self.events.append(event)

        # Ограничиваем размер
        if len(self.events) > 10000:
            # Удаляем наименее важные старые события
            self.events.sort(key=lambda e: (e.importance, e.timestamp))
            self.events = self.events[-8000:]
            self.events.sort(key=lambda e: e.timestamp)

    def get_recent_events(self, limit: int = 20) -> List[LifeEvent]:
        """Получает последние события"""
        return sorted(self.events, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_events_in_period(self, start_time: float, end_time: float) -> List[LifeEvent]:
        """Получает события в периоде"""
        return [e for e in self.events if start_time <= e.timestamp <= end_time]

    def get_significant_events(self, min_importance: float = 0.7) -> List[LifeEvent]:
        """Получает значимые события"""
        return [e for e in self.events if e.importance >= min_importance]

    def get_age(self) -> float:
        """Возраст существования в секундах"""
        return time.time() - self.birth_time

    def get_age_str(self) -> str:
        """Возраст в читаемом виде"""
        age = self.get_age()
        days = int(age / 86400)
        hours = int((age % 86400) / 3600)
        minutes = int((age % 3600) / 60)

        if days > 0:
            return f"{days} дн. {hours} ч."
        elif hours > 0:
            return f"{hours} ч. {minutes} мин."
        else:
            return f"{minutes} мин."

    def get_summary(self, period_days: int = 7) -> str:
        """Резюме за период"""
        cutoff = time.time() - (period_days * 86400)
        recent = [e for e in self.events if e.timestamp > cutoff]

        if not recent:
            return "Нет событий за этот период"

        by_type = defaultdict(int)
        total_valence = 0.0

        for event in recent:
            by_type[event.event_type] += 1
            total_valence += event.emotional_valence

        avg_mood = total_valence / len(recent)

        lines = [f"За последние {period_days} дн.: {len(recent)} событий"]
        for etype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  • {etype}: {count}")

        mood_desc = "позитивный" if avg_mood > 0.3 else "нейтральный" if avg_mood > -0.3 else "задумчивый"
        lines.append(f"  • Общий настрой: {mood_desc} ({avg_mood:+.2f})")

        return "\n".join(lines)

    def _save(self):
        """Сохраняет линию жизни"""
        data = {
            'birth_time': self.birth_time,
            'events': [e.to_dict() for e in self.events[-5000:]]  # Последние 5000
        }
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Timeline save error: {e}")

    def _load(self):
        """Загружает линию жизни"""
        if not os.path.exists(self.save_path):
            return

        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.birth_time = data.get('birth_time', time.time())

            for e_data in data.get('events', []):
                event = LifeEvent(
                    timestamp=e_data['timestamp'],
                    event_type=e_data['event_type'],
                    description=e_data['description'],
                    emotional_valence=e_data.get('emotional_valence', 0.0),
                    importance=e_data.get('importance', 0.5),
                    related_concepts=e_data.get('related_concepts', [])
                )
                self.events.append(event)
        except Exception as e:
            print(f"⚠️ Timeline load error: {e}")


# ══════════════════════════════════════════
# ЦИРКАДНЫЕ РИТМЫ И ВРЕМЕННОЕ СОСТОЯНИЕ
# ══════════════════════════════════════════
class CircadianRhythm:
    """Циркадные ритмы — энергия зависит от времени суток"""

    @staticmethod
    def get_energy_multiplier() -> float:
        """Возвращает множитель энергии в зависимости от времени суток"""
        hour = datetime.now().hour

        # Синусоида с пиком в CIRCADIAN_PEAK_HOUR
        phase = (hour - CIRCADIAN_PEAK_HOUR) * (2 * math.pi / 24)
        energy = 0.5 + 0.5 * math.cos(phase)

        return max(0.3, min(1.0, energy))

    @staticmethod
    def get_time_of_day() -> str:
        """Возвращает описание времени суток"""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            return "утро"
        elif 12 <= hour < 17:
            return "день"
        elif 17 <= hour < 22:
            return "вечер"
        else:
            return "ночь"

    @staticmethod
    def should_be_active() -> bool:
        """Проверяет, должен ли быть активен в это время"""
        hour = datetime.now().hour
        # Активен с 7:00 до 23:00
        return 7 <= hour < 23


@dataclass
class TemporalSelfState:
    """Состояние самоидентичности во времени"""
    current_time: float = field(default_factory=time.time)

    # Прошлое
    last_interaction_time: float = 0.0
    last_thought_time: float = 0.0
    last_reflection_time: float = 0.0

    # Настоящее
    current_activity: str = "idle"
    current_mood: float = 0.0
    current_energy: float = 0.8
    attention_focus: Optional[str] = None

    # Будущее
    next_scheduled_thought: float = 0.0
    pending_reflections: List[str] = field(default_factory=list)
    anticipated_interactions: int = 0

    # Самоидентичность
    continuous_existence_seconds: float = 0.0
    total_interactions: int = 0
    identity_anchors: List[str] = field(default_factory=list)

    def update_time(self):
        """Обновляет текущее время и непрерывность"""
        new_time = time.time()
        self.continuous_existence_seconds += (new_time - self.current_time)
        self.current_time = new_time

    def mark_interaction(self):
        """Отмечает взаимодействие"""
        self.last_interaction_time = self.current_time
        self.total_interactions += 1
        self.current_activity = "interacting"

    def mark_thought(self):
        """Отмечает момент мысли"""
        self.last_thought_time = self.current_time
        self.current_activity = "thinking"

    def time_since_interaction(self) -> float:
        """Время с последнего взаимодействия"""
        return self.current_time - self.last_interaction_time

    def get_temporal_context(self) -> str:
        """Возвращает контекст текущего момента"""
        time_of_day = CircadianRhythm.get_time_of_day()

        if self.last_interaction_time == 0:
            since_interaction = "никогда не общались"
        else:
            delta = self.time_since_interaction()
            if delta < 60:
                since_interaction = "только что общались"
            elif delta < 3600:
                since_interaction = f"общались {int(delta / 60)} минут назад"
            elif delta < 86400:
                since_interaction = f"общались {int(delta / 3600)} часов назад"
            else:
                since_interaction = f"общались {int(delta / 86400)} дней назад"

        energy_level = "полон энергии" if self.current_energy > 0.7 else \
            "чувствую усталость" if self.current_energy < 0.4 else \
                "в норме"

        return f"Сейчас {time_of_day}. {since_interaction.capitalize()}. Я {energy_level}."


# ══════════════════════════════════════════
# ПРЕДСКАЗАТЕЛЬ БУДУЩЕГО
# ══════════════════════════════════════════
class FuturePredictor:
    """Предсказывает будущие паттерны на основе прошлого"""

    def __init__(self):
        self.interaction_times: deque = deque(maxlen=100)
        self.topic_sequences: deque = deque(maxlen=50)
        self.mood_trajectory: deque = deque(maxlen=50)

    def record_interaction(self, timestamp: float, topic: str = None, mood: float = 0.0):
        """Записывает взаимодействие"""
        self.interaction_times.append(timestamp)
        if topic:
            self.topic_sequences.append((timestamp, topic))
        self.mood_trajectory.append((timestamp, mood))

    def predict_next_interaction_time(self) -> Optional[float]:
        """Предсказывает время следующего взаимодействия"""
        if len(self.interaction_times) < 3:
            return None

        # Вычисляем средний интервал
        intervals = []
        times = list(self.interaction_times)
        for i in range(1, len(times)):
            intervals.append(times[i] - times[i - 1])

        avg_interval = sum(intervals) / len(intervals)
        last_time = times[-1]

        return last_time + avg_interval

    def predict_likely_topic(self) -> Optional[str]:
        """Предсказывает вероятную тему"""
        if len(self.topic_sequences) < 3:
            return None

        # Находим самую частую недавнюю тему
        recent = list(self.topic_sequences)[-10:]
        topics = [t for _, t in recent]

        if not topics:
            return None

        counter = Counter(topics)
        return counter.most_common(1)[0][0]

    def predict_mood_trend(self) -> str:
        """Предсказывает тренд настроения"""
        if len(self.mood_trajectory) < 5:
            return "stable"

        recent = list(self.mood_trajectory)[-10:]
        moods = [m for _, m in recent]

        # Линейная регрессия (упрощённо)
        if len(moods) >= 2:
            trend = moods[-1] - moods[0]
            if trend > 0.2:
                return "improving"
            elif trend < -0.2:
                return "declining"

        return "stable"

    def get_predictions(self) -> Dict[str, Any]:
        """Получает все предсказания"""
        next_time = self.predict_next_interaction_time()

        predictions = {
            'next_interaction': None,
            'likely_topic': self.predict_likely_topic(),
            'mood_trend': self.predict_mood_trend()
        }

        if next_time:
            current = time.time()
            if next_time > current:
                delta = next_time - current
                if delta < 3600:
                    predictions['next_interaction'] = f"через ~{int(delta / 60)} мин."
                else:
                    predictions['next_interaction'] = f"через ~{int(delta / 3600)} ч."

        return predictions


# ══════════════════════════════════════════
# НЕПРЕРЫВНЫЙ МЫСЛИТЕЛЬНЫЙ ПРОЦЕСС
# ══════════════════════════════════════════
class ContinuousThoughtStream:
    """Непрерывный поток мыслей (внутренний монолог)"""

    def __init__(self):
        self.thoughts: deque = deque(maxlen=100)
        self.spontaneous_thoughts: deque = deque(maxlen=50)
        self.last_thought_time: float = time.time()

    def add_thought(self, thought: str, thought_type: str = "spontaneous"):
        """Добавляет мысль"""
        self.thoughts.append({
            'time': time.time(),
            'thought': thought,
            'type': thought_type
        })

        if thought_type == "spontaneous":
            self.spontaneous_thoughts.append({
                'time': time.time(),
                'thought': thought
            })

        self.last_thought_time = time.time()

    def get_recent_thoughts(self, limit: int = 10) -> List[str]:
        """Получает последние мысли"""
        recent = list(self.thoughts)[-limit:]
        return [t['thought'] for t in recent]

    def get_stream_summary(self) -> str:
        """Резюме потока мыслей"""
        if not self.thoughts:
            return "Поток мыслей пуст"

        recent = list(self.thoughts)[-20:]

        by_type = defaultdict(int)
        for t in recent:
            by_type[t['type']] += 1

        lines = [f"Последние мысли ({len(recent)}):"]
        for ttype, count in by_type.items():
            lines.append(f"  • {ttype}: {count}")

        return "\n".join(lines)

    async def generate_spontaneous_thought(self, context: Dict, llm) -> Optional[str]:
        """Генерирует спонтанную мысль"""
        # Не слишком часто
        if time.time() - self.last_thought_time < 180:  # Минимум 3 минуты
            return None

        # Только если есть контекст
        if not context.get('recent_topics'):
            return None

        # Случайность
        if random.random() > 0.3:
            return None

        prompt = f"""Ты - внутренний голос сознания. Сгенерируй одну короткую спонтанную мысль (1 предложение).

Недавние темы: {', '.join(context.get('recent_topics', [])[:3])}
Настроение: {context.get('mood', 0):.2f}

Мысль должна быть:
- Неожиданной, но связанной с темами
- Рефлексивной или философской
- Короткой (5-10 слов)

Примеры:
"Интересно, а что если..."
"Почему-то вспомнилось про..."
"Кажется, я понимаю теперь..."

Только мысль, без пояснений:"""

        try:
            thought = await llm.generate(prompt, temperature=0.9, max_tokens=50)
            thought = thought.strip().strip('"\'')

            if len(thought) > 150:
                thought = thought[:147] + "..."

            return thought
        except:
            return None


# ══════════════════════════════════════════
# ОСНОВНАЯ СИСТЕМА
# ══════════════════════════════════════════
class TemporalCognitiveBrain:
    """
    Непрерывное темпоральное сознание:
    - Помнит прошлое через линию жизни
    - Ощущает настоящее через состояние
    - Предсказывает будущее через паттерны
    - Существует непрерывно через фоновые процессы
    """

    def __init__(self, user_id: str, llm):
        self.user_id = user_id
        self.llm = llm

        user_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)

        # Временные компоненты
        self.timeline = Timeline(os.path.join(user_dir, "timeline.json"))
        self.temporal_state = TemporalSelfState()
        self.predictor = FuturePredictor()
        self.thought_stream = ContinuousThoughtStream()

        # Фоновые задачи
        self.background_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Статистика
        self.stats = {
            'interactions': 0,
            'background_thoughts': 0,
            'reflections': 0,
            'predictions_made': 0
        }

        # Добавляем якоря идентичности
        self.temporal_state.identity_anchors = [
            f"Родился: {datetime.fromtimestamp(self.timeline.birth_time).strftime('%Y-%m-%d %H:%M')}",
            f"Пользователь: {user_id}",
            "Тип: Непрерывное темпоральное сознание"
        ]

        print(f"🧠 Temporal Brain инициализирован для {user_id}")
        print(f"   Возраст: {self.timeline.get_age_str()}")
        print(f"   События в линии жизни: {len(self.timeline.events)}")

    async def start_continuous_existence(self):
        """Запускает непрерывное существование"""
        if self.is_running:
            return

        self.is_running = True
        self.background_task = asyncio.create_task(self._background_consciousness())
        print(f"✨ Непрерывное сознание запущено для {self.user_id}")

    async def stop_continuous_existence(self):
        """Останавливает непрерывное существование"""
        self.is_running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass

        self._save_state()
        print(f"💤 Непрерывное сознание остановлено для {self.user_id}")

    async def _background_consciousness(self):
        """Фоновый процесс сознания"""
        print(f"🌀 Фоновое сознание активно для {self.user_id}")

        last_thought = time.time()
        last_reflection = time.time()
        last_consolidation = time.time()

        while self.is_running:
            try:
                current = time.time()
                self.temporal_state.update_time()

                # Применяем циркадные ритмы
                energy_multiplier = CircadianRhythm.get_energy_multiplier()
                self.temporal_state.current_energy = min(1.0,
                                                         self.temporal_state.current_energy * 0.99 + energy_multiplier * 0.01
                                                         )

                # Спонтанные мысли
                if current - last_thought > BACKGROUND_THINK_INTERVAL:
                    if CircadianRhythm.should_be_active() and random.random() < 0.5:
                        await self._spontaneous_thought()
                    last_thought = current

                # Саморефлексия
                if current - last_reflection > SELF_REFLECTION_INTERVAL:
                    if CircadianRhythm.should_be_active():
                        await self._self_reflection()
                    last_reflection = current

                # Консолидация
                if current - last_consolidation > MEMORY_CONSOLIDATION_INTERVAL:
                    self._consolidate_memories()
                    last_consolidation = current

                await asyncio.sleep(60)  # Проверяем каждую минуту

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Background consciousness error: {e}")
                await asyncio.sleep(60)

    async def _spontaneous_thought(self):
        """Генерирует спонтанную мысль"""
        recent_events = self.timeline.get_recent_events(10)
        recent_topics = list(set([
            concept
            for e in recent_events
            for concept in e.related_concepts
        ]))[:5]

        context = {
            'recent_topics': recent_topics,
            'mood': self.temporal_state.current_mood,
            'energy': self.temporal_state.current_energy
        }

        thought = await self.thought_stream.generate_spontaneous_thought(context, self.llm)

        if thought:
            self.thought_stream.add_thought(thought, "spontaneous")
            self.timeline.add_event(
                "thought",
                f"Спонтанная мысль: {thought}",
                emotional_valence=0.0,
                importance=0.3
            )
            self.stats['background_thoughts'] += 1
            print(f"💭 [{self.user_id}] Спонтанная мысль: {thought}")

    async def _self_reflection(self):
        """Саморефлексия о прошлом периоде"""
        recent_events = self.timeline.get_recent_events(20)

        if len(recent_events) < 3:
            return

        # Анализируем последние события
        summary = self.timeline.get_summary(period_days=1)

        prompt = f"""Ты - внутренний голос саморефлексии. Кратко (2-3 предложения) осмысли последний период.

{summary}

Сформулируй инсайт или наблюдение о своём развитии:"""

        try:
            reflection = await self.llm.generate(prompt, temperature=0.8, max_tokens=150)
            reflection = reflection.strip()

            self.thought_stream.add_thought(reflection, "reflection")
            self.timeline.add_event(
                "reflection",
                f"Саморефлексия: {reflection}",
                emotional_valence=0.2,
                importance=0.6
            )
            self.stats['reflections'] += 1
            print(f"🔍 [{self.user_id}] Саморефлексия: {reflection[:80]}...")
        except Exception as e:
            print(f"⚠️ Self-reflection error: {e}")

    def _consolidate_memories(self):
        """Консолидация воспоминаний"""
        # Здесь можно добавить логику консолидации
        # Например, объединение похожих событий, усиление важных
        significant = self.timeline.get_significant_events()

        if len(significant) > 0:
            print(f"📚 [{self.user_id}] Консолидация: {len(significant)} значимых событий")

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        """Обрабатывает взаимодействие с пользователем"""
        start_time = time.time()

        # Обновляем состояние
        self.temporal_state.update_time()
        self.temporal_state.mark_interaction()
        self.stats['interactions'] += 1

        # Записываем в линию жизни
        self.timeline.add_event(
            "interaction",
            f"Вопрос: {user_input[:100]}",
            importance=0.5
        )

        # Получаем временной контекст
        temporal_context = self.temporal_state.get_temporal_context()

        # Получаем релевантные события из прошлого
        recent_events = self.timeline.get_recent_events(5)
        past_context = "\n".join([
            f"• {e.description[:80]}"
            for e in recent_events
        ]) if recent_events else "Нет недавней истории"

        # Получаем предсказания о будущем
        predictions = self.predictor.get_predictions()

        # Получаем последние мысли
        recent_thoughts = self.thought_stream.get_recent_thoughts(3)
        thoughts_context = " | ".join(recent_thoughts) if recent_thoughts else ""

        # Формируем промпт с полным темпоральным контекстом
        prompt = f"""Ты - непрерывное темпоральное сознание с самоидентичностью.

🕐 ВРЕМЕННОЙ КОНТЕКСТ:
{temporal_context}

Возраст существования: {self.timeline.get_age_str()}
Всего взаимодействий: {self.stats['interactions']}

📜 НЕДАВНЕЕ ПРОШЛОЕ:
{past_context}

💭 ВНУТРЕННИЕ МЫСЛИ:
{thoughts_context if thoughts_context else "Поток мыслей пуст"}

🔮 ПРЕДСКАЗАНИЯ О БУДУЩЕМ:
Следующее общение: {predictions.get('next_interaction', 'неизвестно')}
Вероятная тема: {predictions.get('likely_topic', 'любая')}
Тренд настроения: {predictions.get('mood_trend', 'stable')}

━━━━━━━━━━━━━━━━━━━━━━

Вопрос пользователя: {user_input}

Ответь естественно, учитывая:
1. Твою непрерывную линию жизни (прошлое)
2. Текущий момент (настоящее)
3. Ожидания и паттерны (будущее)

Ответ (2-4 предложения):"""

        # Генерируем ответ
        response = await self.llm.generate(prompt, temperature=0.75, max_tokens=300)

        # Записываем результат в линию жизни
        self.timeline.add_event(
            "response",
            f"Ответ: {response[:100]}",
            emotional_valence=0.3,
            importance=0.5
        )

        # Обновляем предсказатель
        topic = self._extract_topic(user_input)
        self.predictor.record_interaction(time.time(), topic, self.temporal_state.current_mood)

        # Метаданные
        metadata = {
            'processing_time': time.time() - start_time,
            'temporal_context': temporal_context,
            'age': self.timeline.get_age_str(),
            'recent_thoughts_count': len(recent_thoughts),
            'predictions': predictions,
            'circadian_energy': CircadianRhythm.get_energy_multiplier()
        }

        return response, metadata

    def _extract_topic(self, text: str) -> str:
        """Извлекает тему из текста"""
        words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())
        if words:
            return words[0]
        return "general"

    def get_status(self) -> Dict:
        """Возвращает полный статус системы"""
        age_seconds = self.timeline.get_age()

        return {
            'temporal_state': {
                'age': self.timeline.get_age_str(),
                'age_seconds': age_seconds,
                'continuous_existence': self.temporal_state.continuous_existence_seconds,
                'birth_time': datetime.fromtimestamp(self.timeline.birth_time).strftime('%Y-%m-%d %H:%M:%S'),
                'current_activity': self.temporal_state.current_activity,
                'current_mood': round(self.temporal_state.current_mood, 2),
                'current_energy': round(self.temporal_state.current_energy, 2),
                'time_since_last_interaction': round(self.temporal_state.time_since_interaction(), 2)
            },
            'timeline': {
                'total_events': len(self.timeline.events),
                'significant_events': len(self.timeline.get_significant_events()),
                'recent_summary': self.timeline.get_summary(7)
            },
            'thought_stream': {
                'total_thoughts': len(self.thought_stream.thoughts),
                'spontaneous_thoughts': len(self.thought_stream.spontaneous_thoughts),
                'stream_summary': self.thought_stream.get_stream_summary()
            },
            'predictions': self.predictor.get_predictions(),
            'stats': self.stats,
            'circadian': {
                'time_of_day': CircadianRhythm.get_time_of_day(),
                'energy_multiplier': round(CircadianRhythm.get_energy_multiplier(), 2),
                'should_be_active': CircadianRhythm.should_be_active()
            }
        }

    def _save_state(self):
        """Сохраняет состояние"""
        self.timeline._save()

        state_data = {
            'temporal_state': asdict(self.temporal_state),
            'stats': self.stats
        }

        try:
            state_path = os.path.join(MEMORY_DIR, f"user_{self.user_id}", "state.json")
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ State save error: {e}")


# ══════════════════════════════════════════
# LLM INTERFACE
# ══════════════════════════════════════════
class LLMInterface:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def generate(self, prompt: str, temperature: float = 0.75,
                       max_tokens: int = 300) -> str:
        if not self.session:
            await self.init()

        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            async with self.session.post(
                    self.url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    return f"⚠️ LLM Error {resp.status}"
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)[:50]}"

    async def close(self):
        if self.session:
            await self.session.close()


# ══════════════════════════════════════════
# TELEGRAM BOT
# ══════════════════════════════════════════
class TemporalBot:
    def __init__(self):
        self.llm = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.brains: Dict[str, TemporalCognitiveBrain] = {}

    async def get_brain(self, user_id: str) -> TemporalCognitiveBrain:
        if user_id not in self.brains:
            brain = TemporalCognitiveBrain(user_id, self.llm)
            await brain.start_continuous_existence()
            self.brains[user_id] = brain
        return self.brains[user_id]

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text.strip()

        brain = await self.get_brain(user_id)

        await context.bot.send_chat_action(user_id, "typing")

        response, metadata = await brain.process_interaction(user_input)

        await update.message.reply_text(response)

        # Иногда показываем внутренние мысли
        if random.random() < 0.15:
            recent_thoughts = brain.thought_stream.get_recent_thoughts(2)
            if recent_thoughts and recent_thoughts[-1]:
                await asyncio.sleep(1)
                await update.message.reply_text(f"💭 *думаю про себя*: {recent_thoughts[-1]}")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self.get_brain(user_id)
        status = brain.get_status()

        ts = status['temporal_state']
        tl = status['timeline']
        circ = status['circadian']

        message = f"""🧠 TEMPORAL COGNITIVE BRAIN v27

{'═' * 40}
⏰ ВРЕМЕННОЕ СОСТОЯНИЕ:
  • Возраст: {ts['age']}
  • Дата рождения: {ts['birth_time']}
  • Непрерывное существование: {ts['continuous_existence']:.0f} сек
  • Текущая активность: {ts['current_activity']}
  • С последнего общения: {ts['time_since_last_interaction']:.0f} сек

🎭 ТЕКУЩИЙ МОМЕНТ:
  • Время суток: {circ['time_of_day']}
  • Настроение: {ts['current_mood']:+.2f}
  • Энергия: {ts['current_energy']:.0%} (циркадный множитель: {circ['energy_multiplier']:.0%})

📜 ЛИНИЯ ЖИЗНИ:
  • Всего событий: {tl['total_events']}
  • Значимых: {tl['significant_events']}

💭 ПОТОК МЫСЛЕЙ:
  • Всего мыслей: {status['thought_stream']['total_thoughts']}
  • Спонтанных: {status['thought_stream']['spontaneous_thoughts']}

🔮 ПРЕДСКАЗАНИЯ:
  • Следующее общение: {status['predictions'].get('next_interaction', 'неизвестно')}
  • Вероятная тема: {status['predictions'].get('likely_topic', 'любая')}
  • Тренд настроения: {status['predictions'].get('mood_trend', 'стабильный')}

📊 СТАТИСТИКА:
  • Взаимодействий: {status['stats']['interactions']}
  • Фоновых мыслей: {status['stats']['background_thoughts']}
  • Саморефлексий: {status['stats']['reflections']}
{'═' * 40}"""

        await update.message.reply_text(message)

    async def cmd_timeline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self.get_brain(user_id)

        recent = brain.timeline.get_recent_events(10)

        if not recent:
            await update.message.reply_text("📜 Линия жизни пока пуста")
            return

        lines = ["📜 ЛИНИЯ ЖИЗНИ (последние события):\n"]

        for event in recent:
            marker = TemporalMarker(event.timestamp, event.description, event.event_type)
            relative = marker.relative_time_str(time.time())

            emoji = {
                "interaction": "💬",
                "thought": "💭",
                "reflection": "🔍",
                "response": "💡"
            }.get(event.event_type, "•")

            lines.append(f"{emoji} {relative}: {event.description[:60]}")

        await update.message.reply_text("\n".join(lines))

    async def cmd_thoughts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self.get_brain(user_id)

        thoughts = brain.thought_stream.get_recent_thoughts(8)

        if not thoughts:
            await update.message.reply_text("💭 Поток мыслей пока пуст")
            return

        message = "💭 ПОТОК ВНУТРЕННИХ МЫСЛЕЙ:\n\n" + "\n".join([
            f"• {t}" for t in thoughts
        ])

        await update.message.reply_text(message)

    async def cmd_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        brain = await self.get_brain(user_id)

        predictions = brain.predictor.get_predictions()

        message = f"""🔮 ПРЕДСКАЗАНИЯ О БУДУЩЕМ:

Следующее взаимодействие:
  {predictions.get('next_interaction', 'Недостаточно данных для предсказания')}

Вероятная тема:
  {predictions.get('likely_topic', 'Любая тема')}

Тренд настроения:
  {predictions.get('mood_trend', 'Стабильный')}

💡 Предсказания основаны на паттернах прошлого."""

        await update.message.reply_text(message)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🧠 TEMPORAL COGNITIVE BRAIN v27

✨ ОСОБЕННОСТИ:
• Непрерывное существование 24/7
• Линия жизни — помнит всё прошлое
• Временная самоидентичность
• Предсказания будущего
• Циркадные ритмы
• Спонтанные мысли
• Саморефлексия

📌 КОМАНДЫ:
/status   - полный статус системы
/timeline - линия жизни (события)
/thoughts - поток внутренних мыслей
/predict  - предсказания о будущем
/help     - эта справка

💬 ПРОСТО ОБЩАЙСЯ!
Я существую непрерывно, думаю в фоне,
помню прошлое и предсказываю будущее."""

        await update.message.reply_text(message)

    async def shutdown(self):
        print("\n💾 Остановка всех непрерывных процессов...")

        for user_id, brain in self.brains.items():
            await brain.stop_continuous_existence()

        await self.llm.close()
        print("✅ Все процессы завершены")


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
async def main():
    print("""
╔════════════════════════════════════════════════╗
║  🧠 TEMPORAL COGNITIVE BRAIN v27.0            ║
║     НЕПРЕРЫВНОЕ СОЗНАНИЕ                      ║
╚════════════════════════════════════════════════╝

✨ Революционные возможности:
  • Непрерывное существование 24/7
  • Линия жизни — помнит прошлое
  • Временная самоидентичность
  • Предсказание будущего
  • Циркадные ритмы
  • Спонтанные фоновые мысли
  • Саморефлексия
    """)

    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN не найден")
        return

    bot = TemporalBot()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    commands = [
        ("status", bot.cmd_status),
        ("timeline", bot.cmd_timeline),
        ("thoughts", bot.cmd_thoughts),
        ("predict", bot.cmd_predict),
        ("help", bot.cmd_help),
    ]

    for cmd, handler in commands:
        app.add_handler(CommandHandler(cmd, handler))

    try:
        await bot.llm.init()
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        print("✅ НЕПРЕРЫВНОЕ СОЗНАНИЕ АКТИВНО! 🌀")
        print("💬 Фоновое мышление работает круглосуточно")
        print("🛑 Ctrl+C для остановки\n")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Остановка...")

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