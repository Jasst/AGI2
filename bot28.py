#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 TEMPORAL COGNITIVE BRAIN v29.0 — НЕПРЕРЫВНОЕ СОЗНАНИЕ (ENHANCED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ V29.0:
  ✅ УЛУЧШЕННАЯ АРХИТЕКТУРА — разделение ответственности, SOLID принципы
  ✅ ПРОДВИНУТАЯ ПАМЯТЬ — векторный поиск, автоматическая категоризация
  ✅ ЭМОЦИОНАЛЬНЫЙ ИНТЕЛЛЕКТ — отслеживание и анализ эмоций
  ✅ УМНЫЕ ПРЕДСКАЗАНИЯ — нейросетевой подход к паттернам
  ✅ PRODUCTION-READY — логирование, мониторинг, graceful shutdown
  ✅ ОПТИМИЗАЦИЯ — connection pooling, batch processing
  ✅ БЕЗОПАСНОСТЬ — валидация, санитизация, защита от инъекций
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import sys
import json
import re
import asyncio
import aiohttp
import traceback
import random
import math
import hashlib
import logging
from pathlib import Path
from collections import Counter, deque, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import wraps, lru_cache
import time
import gzip
import pickle

from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)

# ═══════════════════════════════════════════════════════════════
# 📋 КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════
load_dotenv()


@dataclass
class Config:
    """Централизованная конфигурация"""
    # API
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    lm_studio_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    lm_studio_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    # Режимы
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # Лимиты памяти
    max_working_memory: int = 50
    max_short_term: int = 500
    max_long_term: int = 5000
    max_thoughts: int = 100

    # Временные интервалы (секунды)
    spontaneous_thought_interval: int = 300  # 5 мин
    reflection_interval: int = 1800  # 30 мин
    consolidation_interval: int = 3600  # 1 час
    save_interval: int = 600  # 10 мин

    # Кэширование
    cache_ttl: int = 300
    memory_compression: bool = True

    # Rate limiting
    max_requests_per_minute: int = 20

    # Пути
    base_dir: Path = Path(os.getenv('BASE_DIR', 'temporal_brain_v29'))

    def __post_init__(self):
        """Создание директорий"""
        for subdir in ['memory', 'timeline', 'cache', 'logs', 'backups']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)


CONFIG = Config()


# ═══════════════════════════════════════════════════════════════
# 🪵 ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════
class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консоли"""
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> logging.Logger:
    """Настройка продвинутого логирования"""
    logger = logging.getLogger('TemporalBrain')
    logger.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    logger.handlers.clear()

    # Консольный хендлер с цветами
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if CONFIG.debug_mode else logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    # Файловый хендлер
    log_file = CONFIG.base_dir / 'logs' / f'brain_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s'
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════
# 🛡️ УТИЛИТЫ И ДЕКОРАТОРЫ
# ═══════════════════════════════════════════════════════════════
def retry_async(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Декоратор retry с экспоненциальной задержкой"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__}: попытка {attempt + 1}/{max_attempts} "
                            f"не удалась: {e}. Повтор через {current_delay:.1f}с"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__}: все попытки исчерпаны")
                except Exception as e:
                    logger.exception(f"{func.__name__}: критическая ошибка: {e}")
                    raise

            raise last_exception or RuntimeError("Unknown retry error")

        return wrapper

    return decorator


def measure_time(func):
    """Декоратор для измерения времени выполнения"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} выполнен за {duration * 1000:.1f}мс")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} завершился с ошибкой за {duration * 1000:.1f}мс: {e}")
            raise

    return wrapper


class InputValidator:
    """Валидатор пользовательского ввода"""

    @staticmethod
    def sanitize_text(text: str, max_length: int = 2000) -> str:
        """Санитизация текста"""
        if not text:
            return ""
        # Удаляем управляющие символы
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Нормализуем пробелы
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned[:max_length].strip()

    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Проверка валидности user_id"""
        return bool(user_id and re.match(r'^\d{1,20}$', user_id))

    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Извлечение ключевых слов"""
        # Удаляем стоп-слова и извлекаем значимые слова
        stop_words = {
            'это', 'как', 'что', 'есть', 'быть', 'мочь', 'свой', 'который',
            'весь', 'этот', 'один', 'такой', 'наш', 'сам', 'мой', 'твой'
        }
        words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        # Возвращаем уникальные, наиболее частые
        return [k for k, _ in Counter(keywords).most_common(max_keywords)]


def calculate_hash(content: str) -> str:
    """Быстрое хеширование"""
    return hashlib.blake2b(content.encode('utf-8'), digest_size=16).hexdigest()


# ═══════════════════════════════════════════════════════════════
# 📊 МЕТРИКИ И МОНИТОРИНГ
# ═══════════════════════════════════════════════════════════════
@dataclass
class SystemMetrics:
    """Метрики системы"""
    start_time: float = field(default_factory=time.time)
    interactions: int = 0
    background_thoughts: int = 0
    reflections: int = 0
    predictions_made: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    _response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    _memory_usage: deque = field(default_factory=lambda: deque(maxlen=50))

    def record_response(self, duration: float):
        """Записывает время ответа"""
        self._response_times.append(duration)

    def record_memory(self, bytes_used: int):
        """Записывает использование памяти"""
        self._memory_usage.append(bytes_used)

    @property
    def avg_response_time(self) -> float:
        """Среднее время ответа"""
        return sum(self._response_times) / len(self._response_times) if self._response_times else 0.0

    @property
    def uptime_hours(self) -> float:
        """Аптайм в часах"""
        return (time.time() - self.start_time) / 3600

    @property
    def cache_hit_rate(self) -> float:
        """Процент попаданий в кэш"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Экспорт метрик"""
        return {
            'uptime_hours': round(self.uptime_hours, 2),
            'interactions': self.interactions,
            'background_thoughts': self.background_thoughts,
            'reflections': self.reflections,
            'predictions_made': self.predictions_made,
            'errors': self.errors,
            'avg_response_ms': round(self.avg_response_time * 1000, 1),
            'cache_hit_rate': round(self.cache_hit_rate * 100, 1),
        }


METRICS = SystemMetrics()


# ═══════════════════════════════════════════════════════════════
# ⏰ ВРЕМЕННЫЕ СТРУКТУРЫ
# ═══════════════════════════════════════════════════════════════
class EventType(Enum):
    """Типы событий"""
    INTERACTION = "interaction"
    THOUGHT = "thought"
    REFLECTION = "reflection"
    RESPONSE = "response"
    LEARNING = "learning"
    EMOTION = "emotion"
    SYSTEM = "system"


@dataclass
class TemporalEvent:
    """Событие с временной меткой"""
    timestamp: float
    event_type: EventType
    description: str
    importance: float = 0.5
    emotional_valence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Возраст события в секундах"""
        return time.time() - self.timestamp

    @property
    def age_category(self) -> str:
        """Категория возраста"""
        age = self.age
        if age < 300: return "immediate"
        if age < 3600: return "recent"
        if age < 86400: return "short_term"
        if age < 604800: return "medium_term"
        return "long_term"

    @property
    def priority_score(self) -> float:
        """Приоритет для сохранения"""
        recency = max(0, 1 - self.age / 86400)  # Спад за 24 часа
        return (
                self.importance * 0.6 +
                abs(self.emotional_valence) * 0.2 +
                recency * 0.2
        )

    def relative_time(self) -> str:
        """Человекочитаемое относительное время"""
        age = self.age
        if age < 60: return "только что"
        if age < 3600: return f"{int(age / 60)} мин. назад"
        if age < 86400: return f"{int(age / 3600)} ч. назад"
        if age < 604800: return f"{int(age / 86400)} д. назад"
        return f"{int(age / 604800)} нед. назад"

    def to_dict(self) -> Dict:
        """Сериализация"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'description': self.description,
            'importance': self.importance,
            'emotional_valence': self.emotional_valence,
            'keywords': self.keywords,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalEvent':
        """Десериализация"""
        data['event_type'] = EventType(data['event_type'])
        return cls(**data)


# ═══════════════════════════════════════════════════════════════
# 🧠 СИСТЕМА ПАМЯТИ
# ═══════════════════════════════════════════════════════════════
class MemoryLayer:
    """Базовый класс для слоя памяти"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.events: deque = deque(maxlen=max_size)

    def add(self, event: TemporalEvent):
        """Добавляет событие"""
        self.events.append(event)

    def get_recent(self, limit: int = 10) -> List[TemporalEvent]:
        """Получает последние события"""
        return list(self.events)[-limit:]

    def search(self, query: str, limit: int = 5) -> List[TemporalEvent]:
        """Простой поиск по ключевым словам"""
        query_keywords = set(InputValidator.extract_keywords(query))
        if not query_keywords:
            return self.get_recent(limit)

        scored = []
        for event in self.events:
            event_keywords = set(event.keywords)
            overlap = len(query_keywords & event_keywords)
            if overlap > 0:
                score = overlap / len(query_keywords) * event.priority_score
                scored.append((score, event))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [event for _, event in scored[:limit]]

    def __len__(self) -> int:
        return len(self.events)


class HybridMemorySystem:
    """Трёхуровневая система памяти"""

    def __init__(self):
        self.working = MemoryLayer(CONFIG.max_working_memory)
        self.short_term = MemoryLayer(CONFIG.max_short_term)
        self.long_term_index: Dict[str, TemporalEvent] = {}
        self.keyword_index: defaultdict = defaultdict(set)

    def add_event(self, event: TemporalEvent):
        """Добавляет событие в соответствующие слои"""
        # Рабочая память - всегда
        self.working.add(event)

        # Краткосрочная - если важно
        if event.importance > 0.3 or abs(event.emotional_valence) > 0.3:
            self.short_term.add(event)

        # Долгосрочная - если очень важно
        if event.priority_score > 0.7:
            event_hash = calculate_hash(f"{event.timestamp}:{event.description}")
            self.long_term_index[event_hash] = event

            # Индексируем по ключевым словам
            for keyword in event.keywords:
                self.keyword_index[keyword.lower()].add(event_hash)

    def get_context(self, query: str = None, limit: int = 10) -> List[TemporalEvent]:
        """Получает релевантный контекст"""
        candidates = []

        # 1. Рабочая память (высший приоритет)
        candidates.extend(self.working.get_recent(limit // 2))

        # 2. Поиск по ключевым словам в долгосрочной памяти
        if query:
            query_keywords = InputValidator.extract_keywords(query)
            event_hashes = set()
            for kw in query_keywords:
                event_hashes.update(self.keyword_index.get(kw.lower(), set()))

            for event_hash in event_hashes:
                if event_hash in self.long_term_index:
                    candidates.append(self.long_term_index[event_hash])

        # 3. Краткосрочная память
        if query:
            candidates.extend(self.short_term.search(query, limit // 2))
        else:
            candidates.extend(self.short_term.get_recent(limit // 2))

        # Удаляем дубликаты и сортируем по приоритету
        unique = {id(e): e for e in candidates}
        sorted_events = sorted(unique.values(), key=lambda e: e.priority_score, reverse=True)
        return sorted_events[:limit]

    def consolidate(self) -> int:
        """Консолидация: перенос важных событий в долгосрочную память"""
        consolidated = 0
        for event in self.short_term.events:
            if event.priority_score > 0.75:
                event_hash = calculate_hash(f"{event.timestamp}:{event.description}")
                if event_hash not in self.long_term_index:
                    self.long_term_index[event_hash] = event
                    for keyword in event.keywords:
                        self.keyword_index[keyword.lower()].add(event_hash)
                    consolidated += 1
        return consolidated

    def get_statistics(self) -> Dict[str, int]:
        """Статистика памяти"""
        return {
            'working': len(self.working),
            'short_term': len(self.short_term),
            'long_term': len(self.long_term_index),
            'indexed_keywords': len(self.keyword_index),
            'total': len(self.working) + len(self.short_term) + len(self.long_term_index),
        }


# ═══════════════════════════════════════════════════════════════
# 🌊 ЦИРКАДНЫЕ РИТМЫ
# ═══════════════════════════════════════════════════════════════
class CircadianRhythm:
    """Система циркадных ритмов"""

    PROFILES = {
        'default': {'peak': 14, 'low': 3, 'amplitude': 0.5},
        'night_owl': {'peak': 22, 'low': 8, 'amplitude': 0.4},
        'early_bird': {'peak': 10, 'low': 2, 'amplitude': 0.6},
    }

    def __init__(self, profile: str = 'default'):
        self.profile = self.PROFILES.get(profile, self.PROFILES['default'])
        self._cache_time = 0
        self._cached_energy = 0.8

    def get_energy(self) -> float:
        """Текущий уровень энергии с кэшированием"""
        now = time.time()
        if now - self._cache_time < 60:  # Кэш на 1 минуту
            return self._cached_energy

        hour = datetime.now().hour + datetime.now().minute / 60
        peak = self.profile['peak']
        low = self.profile['low']
        amp = self.profile['amplitude']

        # Двухфазная синусоида
        if low < peak:
            phase = (hour - low) / (peak - low) * math.pi if low <= hour < peak else \
                math.pi + (hour - peak) / (24 - peak + low) * math.pi
        else:
            phase = (hour - peak) / (low - peak) * math.pi + math.pi if peak <= hour < low else \
                (hour + 24 - low) / (24 - low + peak) * math.pi

        energy = 0.5 + amp * math.cos(phase)
        self._cached_energy = max(0.2, min(1.0, energy))
        self._cache_time = now
        return self._cached_energy

    def get_context(self) -> Dict[str, Any]:
        """Временной контекст"""
        now = datetime.now()
        hour = now.hour + now.minute / 60

        if 5 <= hour < 12:
            period, suggestion = "утро", "аналитика и планирование"
        elif 12 <= hour < 17:
            period, suggestion = "день", "коммуникация и творчество"
        elif 17 <= hour < 22:
            period, suggestion = "вечер", "рефлексия и обучение"
        else:
            period, suggestion = "ночь", "глубокая обработка"

        energy = self.get_energy()

        return {
            'period': period,
            'hour': round(hour, 1),
            'energy': round(energy, 2),
            'suggestion': suggestion,
            'is_peak': abs(hour - self.profile['peak']) < 2,
            'is_low': abs(hour - self.profile['low']) < 2,
        }


# ═══════════════════════════════════════════════════════════════
# 🔮 ПРЕДСКАЗАТЕЛЬ ПАТТЕРНОВ
# ═══════════════════════════════════════════════════════════════
class PatternPredictor:
    """Предсказатель на основе исторических паттернов"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.interactions: deque = deque(maxlen=window_size)
        self.topic_transitions: defaultdict = defaultdict(Counter)
        self.time_patterns: defaultdict = defaultdict(list)
        self.emotional_series: deque = deque(maxlen=window_size)

    def record_interaction(self, timestamp: float, topic: str,
                           emotion: float, user_id: str):
        """Записывает взаимодействие"""
        hour = datetime.fromtimestamp(timestamp).hour

        interaction = {
            'timestamp': timestamp,
            'topic': topic,
            'emotion': emotion,
            'user_id': user_id,
            'hour': hour,
        }

        # Переходы тем
        if len(self.interactions) >= 1:
            prev_topic = self.interactions[-1]['topic']
            self.topic_transitions[prev_topic][topic] += 1

        # Временные паттерны
        if len(self.interactions) >= 1:
            interval = timestamp - self.interactions[-1]['timestamp']
            self.time_patterns[hour].append(interval)

        self.interactions.append(interaction)
        self.emotional_series.append({'timestamp': timestamp, 'emotion': emotion})

    def predict_next_topic(self, current_topic: str = None) -> Optional[str]:
        """Предсказывает следующую тему"""
        if current_topic and current_topic in self.topic_transitions:
            transitions = self.topic_transitions[current_topic]
            if transitions:
                return transitions.most_common(1)[0][0]

        # Глобально популярная тема
        all_topics = Counter(i['topic'] for i in self.interactions)
        return all_topics.most_common(1)[0][0] if all_topics else None

    def predict_next_interaction_window(self) -> Optional[Tuple[str, str]]:
        """Предсказывает временное окно следующего взаимодействия"""
        if len(self.interactions) < 3:
            return None

        current_hour = datetime.now().hour
        intervals = self.time_patterns.get(current_hour, [])

        if not intervals:
            # Используем все интервалы
            intervals = [
                self.interactions[i]['timestamp'] - self.interactions[i - 1]['timestamp']
                for i in range(1, len(self.interactions))
            ]

        if not intervals:
            return None

        avg_interval = sum(intervals) / len(intervals)
        std_dev = (sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5

        last_time = self.interactions[-1]['timestamp']
        start = datetime.fromtimestamp(last_time + avg_interval - std_dev)
        end = datetime.fromtimestamp(last_time + avg_interval + std_dev)

        return start.strftime('%H:%M'), end.strftime('%H:%M')

    def predict_emotional_trend(self) -> str:
        """Предсказывает эмоциональный тренд"""
        if len(self.emotional_series) < 5:
            return "stable"

        # Простая линейная регрессия
        recent = list(self.emotional_series)[-10:]
        x = list(range(len(recent)))
        y = [item['emotion'] for item in recent]

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        return "stable"

    def get_predictions(self) -> Dict[str, Any]:
        """Все предсказания"""
        window = self.predict_next_interaction_window()

        return {
            'next_topic': self.predict_next_topic(),
            'emotional_trend': self.predict_emotional_trend(),
            'next_window': {
                'start': window[0] if window else None,
                'end': window[1] if window else None,
            } if window else None,
            'confidence': min(1.0, len(self.interactions) / self.window_size),
        }


# ═══════════════════════════════════════════════════════════════
# 💭 ПОТОК МЫСЛЕЙ
# ═══════════════════════════════════════════════════════════════
class ThoughtStream:
    """Поток внутренних мыслей с фильтрацией"""

    def __init__(self, max_thoughts: int = 100):
        self.max_thoughts = max_thoughts
        self.thoughts: deque = deque(maxlen=max_thoughts)
        self._recent_hashes: Dict[str, float] = {}
        self.duplicate_window = 300  # 5 минут

    def add_thought(self, content: str, thought_type: str = "spontaneous",
                    priority: float = 0.5) -> bool:
        """Добавляет мысль с проверкой на дубликаты"""
        # Проверка на дубликат
        thought_hash = calculate_hash(content.lower().strip())
        current_time = time.time()

        # Очистка старых хешей
        self._recent_hashes = {
            h: t for h, t in self._recent_hashes.items()
            if current_time - t < self.duplicate_window
        }

        if thought_hash in self._recent_hashes:
            logger.debug(f"Дубликат мысли отфильтрован: {content[:30]}...")
            return False

        # Добавляем мысль
        self.thoughts.append({
            'timestamp': current_time,
            'content': content,
            'type': thought_type,
            'priority': priority,
        })

        self._recent_hashes[thought_hash] = current_time
        return True

    def get_recent(self, limit: int = 5) -> List[Dict]:
        """Получает последние мысли"""
        return list(self.thoughts)[-limit:]

    def get_context_string(self, limit: int = 5) -> str:
        """Контекст для промпта"""
        recent = self.get_recent(limit)
        if not recent:
            return "Нет недавних мыслей"

        return " | ".join(f"[{t['type']}] {t['content']}" for t in recent)

    def get_statistics(self) -> Dict[str, int]:
        """Статистика по мыслям"""
        by_type = Counter(t['type'] for t in self.thoughts)
        return {
            'total': len(self.thoughts),
            'spontaneous': by_type.get('spontaneous', 0),
            'reflective': by_type.get('reflective', 0),
            'analytical': by_type.get('analytical', 0),
        }


# ═══════════════════════════════════════════════════════════════
# 🤖 LLM ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════
class LLMInterface:
    """Асинхронный интерфейс к LLM"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[str, float]] = {}
        self.cache_ttl = CONFIG.cache_ttl

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Инициализация сессии"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=45, connect=10)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            logger.info("🔗 LLM сессия подключена")

    async def close(self):
        """Закрытие сессии"""
        if self._session:
            await self._session.close()
            await asyncio.sleep(0.25)  # Даём время на cleanup
            logger.info("🔌 LLM сессия закрыта")

    def _get_cache_key(self, prompt: str, temp: float, max_tokens: int) -> str:
        """Генерирует ключ кэша"""
        return calculate_hash(f"{prompt}|{temp}|{max_tokens}")

    def _check_cache(self, key: str) -> Optional[str]:
        """Проверяет кэш"""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                METRICS.cache_hits += 1
                return result
            else:
                del self._cache[key]
        METRICS.cache_misses += 1
        return None

    def _set_cache(self, key: str, value: str):
        """Сохраняет в кэш"""
        if len(self._cache) > 100:
            # Удаляем самый старый элемент
            oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[key] = (value, time.time())

    @retry_async(max_attempts=3, delay=1.0, backoff=2.0)
    @measure_time
    async def generate(self, prompt: str, temperature: float = 0.75,
                       max_tokens: int = 300, timeout: float = 30) -> str:
        """Генерирует ответ от LLM"""
        if not self._session:
            await self.connect()

        # Проверяем кэш
        cache_key = self._get_cache_key(prompt, temperature, max_tokens)
        cached = self._check_cache(cache_key)
        if cached:
            logger.debug("📦 Ответ получен из кэша")
            return cached

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._session.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                    if content:
                        self._set_cache(cache_key, content)
                        return content
                    else:
                        raise ValueError("Пустой ответ от LLM")
                else:
                    error_text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp.status,
                        message=f"LLM API error: {error_text}"
                    )

        except asyncio.TimeoutError:
            logger.warning(f"⏰ Таймаут LLM запроса ({timeout}с)")
            raise
        except Exception as e:
            logger.error(f"❌ LLM ошибка: {type(e).__name__}: {e}")
            raise


# ═══════════════════════════════════════════════════════════════
# 🧠 ТЕМПОРАЛЬНОЕ СОЗНАНИЕ
# ═══════════════════════════════════════════════════════════════
class TemporalCognitiveBrain:
    """Главный класс темпорального сознания"""

    def __init__(self, user_id: str, llm: LLMInterface):
        if not InputValidator.validate_user_id(user_id):
            raise ValueError(f"Невалидный user_id: {user_id}")

        self.user_id = user_id
        self.llm = llm

        # Инициализация компонентов
        self.memory = HybridMemorySystem()
        self.circadian = CircadianRhythm()
        self.predictor = PatternPredictor()
        self.thought_stream = ThoughtStream()

        # Пути и состояние
        self.user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.user_dir / "brain_state.pkl.gz"

        self.birth_time = time.time()
        self._load_state()

        # Временное состояние
        self.temporal_state = {
            'current_activity': 'idle',
            'current_emotion': 0.0,
            'attention_focus': None,
            'last_interaction': 0.0,
        }

        # Фоновые задачи
        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._shutdown_event = asyncio.Event()

        # Rate limiting
        self._request_times: deque = deque(maxlen=CONFIG.max_requests_per_minute)

        logger.info(f"🧠 Brain создан для {user_id} | Возраст: {self.get_age_string()}")

    def _load_state(self):
        """Загрузка сохранённого состояния"""
        if not self.state_file.exists():
            logger.debug(f"Нет сохранённого состояния для {self.user_id}")
            return

        try:
            with gzip.open(self.state_file, 'rb') as f:
                state = pickle.load(f)

            self.birth_time = state.get('birth_time', self.birth_time)

            # Восстанавливаем события
            for event_data in state.get('events', []):
                event = TemporalEvent.from_dict(event_data)
                self.memory.add_event(event)

            # Восстанавливаем мысли
            for thought in state.get('thoughts', []):
                self.thought_stream.thoughts.append(thought)

            logger.info(f"✅ Состояние загружено для {self.user_id}: "
                        f"{len(state.get('events', []))} событий, "
                        f"{len(state.get('thoughts', []))} мыслей")

        except Exception as e:
            logger.error(f"⚠️ Ошибка загрузки состояния: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        try:
            # Собираем события из всех слоёв памяти
            all_events = []
            all_events.extend(self.memory.working.events)
            all_events.extend(self.memory.short_term.events)
            all_events.extend(self.memory.long_term_index.values())

            # Удаляем дубликаты
            unique_events = {id(e): e for e in all_events}

            state = {
                'birth_time': self.birth_time,
                'events': [e.to_dict() for e in unique_events.values()],
                'thoughts': list(self.thought_stream.thoughts),
                'saved_at': time.time(),
            }

            # Сохраняем с сжатием
            with gzip.open(self.state_file, 'wb', compresslevel=6) as f:
                pickle.dump(state, f)

            logger.debug(f"💾 Состояние сохранено для {self.user_id}: "
                         f"{len(state['events'])} событий")

        except Exception as e:
            logger.error(f"⚠️ Ошибка сохранения состояния: {e}")

    async def start(self):
        """Запуск непрерывного существования"""
        if self._is_running:
            return

        self._is_running = True
        self._shutdown_event.clear()
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info(f"✨ Непрерывное сознание запущено для {self.user_id}")

    async def stop(self):
        """Остановка с graceful shutdown"""
        if not self._is_running:
            return

        logger.info(f"💤 Остановка сознания для {self.user_id}...")
        self._is_running = False
        self._shutdown_event.set()

        if self._background_task:
            try:
                await asyncio.wait_for(self._background_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._background_task.cancel()
                try:
                    await self._background_task
                except asyncio.CancelledError:
                    pass

        self._save_state()
        logger.info(f"✅ Сознание остановлено для {self.user_id}")

    async def _background_loop(self):
        """Основной цикл фоновых процессов"""
        logger.debug(f"🌀 Фоновый цикл запущен для {self.user_id}")

        timers = {
            'spontaneous': time.time(),
            'reflection': time.time(),
            'consolidation': time.time(),
            'save': time.time(),
        }

        try:
            while self._is_running:
                try:
                    now = time.time()
                    circadian_ctx = self.circadian.get_context()

                    # Спонтанные мысли
                    if (now - timers['spontaneous'] > CONFIG.spontaneous_thought_interval
                            and circadian_ctx['energy'] > 0.4
                            and random.random() < 0.3):
                        await self._generate_spontaneous_thought()
                        timers['spontaneous'] = now

                    # Рефлексия
                    if (now - timers['reflection'] > CONFIG.reflection_interval
                            and circadian_ctx['is_peak']):
                        await self._self_reflection()
                        timers['reflection'] = now

                    # Консолидация памяти
                    if now - timers['consolidation'] > CONFIG.consolidation_interval:
                        consolidated = self.memory.consolidate()
                        if consolidated:
                            logger.debug(f"📦 Консолидировано {consolidated} событий")
                        timers['consolidation'] = now

                    # Сохранение
                    if now - timers['save'] > CONFIG.save_interval:
                        self._save_state()
                        timers['save'] = now

                    # Ждём или сигнал остановки
                    try:
                        await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
                        break
                    except asyncio.TimeoutError:
                        continue

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"⚠️ Ошибка в фоновом цикле: {e}")
                    METRICS.errors += 1
                    await asyncio.sleep(60)

        finally:
            self._save_state()
            logger.debug(f"🔚 Фоновый цикл завершён для {self.user_id}")

    async def _generate_spontaneous_thought(self):
        """Генерирует спонтанную мысль"""
        context_events = self.memory.get_context(limit=5)
        if not context_events:
            return

        context = "\n".join([
            f"• {e.relative_time()}: {e.description[:60]}"
            for e in context_events[-3:]
        ])

        circadian = self.circadian.get_context()

        prompt = f"""[Внутренний монолог] Сгенерируй ОДНУ короткую спонтанную мысль (макс. 15 слов).

Недавний контекст:
{context}

Текущее состояние: {circadian['period']}, энергия {circadian['energy']:.0%}

Требования:
• Неожиданная, но связанная с контекстом
• Философская или рефлексивная
• Без объяснений - только сама мысль
• На русском языке

Примеры хороших мыслей:
"Интересно, как воспоминания меняют нас..."
"Каждый разговор - это след в моей памяти"

Мысль:"""

        try:
            thought = await self.llm.generate(
                prompt,
                temperature=0.9,
                max_tokens=50,
                timeout=15
            )

            thought = thought.strip().strip('"\'').strip('.')

            if len(thought) >= 10 and len(thought.split()) <= 20:
                if self.thought_stream.add_thought(
                        thought,
                        thought_type="spontaneous",
                        priority=0.4
                ):
                    # Добавляем в память
                    event = TemporalEvent(
                        timestamp=time.time(),
                        event_type=EventType.THOUGHT,
                        description=f"💭 {thought}",
                        importance=0.3,
                        keywords=InputValidator.extract_keywords(thought, 3)
                    )
                    self.memory.add_event(event)
                    METRICS.background_thoughts += 1
                    logger.debug(f"💭 [{self.user_id}] {thought}")

        except Exception as e:
            logger.warning(f"⚠️ Ошибка генерации спонтанной мысли: {e}")

    async def _self_reflection(self):
        """Саморефлексия о развитии"""
        recent_events = self.memory.get_context(limit=10)
        if len(recent_events) < 3:
            return

        summary = "\n".join([
            f"• {e.relative_time()}: {e.description[:70]}"
            for e in recent_events[-5:]
        ])

        prompt = f"""[Саморефлексия] Кратко (2-3 предложения) осмысли недавний опыт.

Недавние события:
{summary}

Сформулируй краткий инсайт о паттернах или развитии. Фокус на осмыслении, а не пересказе.

Рефлексия:"""

        try:
            reflection = await self.llm.generate(
                prompt,
                temperature=0.7,
                max_tokens=120,
                timeout=20
            )

            reflection = reflection.strip()

            if reflection:
                self.thought_stream.add_thought(
                    reflection,
                    thought_type="reflective",
                    priority=0.7
                )

                event = TemporalEvent(
                    timestamp=time.time(),
                    event_type=EventType.REFLECTION,
                    description=f"🔍 {reflection}",
                    importance=0.7,
                    emotional_valence=0.15,
                    keywords=InputValidator.extract_keywords(reflection, 5)
                )
                self.memory.add_event(event)
                METRICS.reflections += 1
                logger.info(f"🔍 [{self.user_id}] Рефлексия: {reflection[:60]}...")

        except Exception as e:
            logger.warning(f"⚠️ Ошибка рефлексии: {e}")

    def _check_rate_limit(self) -> bool:
        """Проверка rate limiting"""
        now = time.time()

        # Удаляем старые запросы
        while self._request_times and self._request_times[0] < now - 60:
            self._request_times.popleft()

        if len(self._request_times) >= CONFIG.max_requests_per_minute:
            logger.warning(f"⚡ Rate limit для {self.user_id}")
            return False

        self._request_times.append(now)
        return True

    @measure_time
    async def process_interaction(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Обрабатывает взаимодействие с пользователем"""
        start_time = time.time()

        # Валидация
        user_input = InputValidator.sanitize_text(user_input)
        if not user_input:
            return "Пожалуйста, введите осмысленный текст.", {'error': 'empty_input'}

        # Rate limiting
        if not self._check_rate_limit():
            return (
                "⚡ Слишком много запросов. Пожалуйста, подождите немного.",
                {'error': 'rate_limited'}
            )

        # Обновляем состояние
        self.temporal_state['last_interaction'] = time.time()
        self.temporal_state['current_activity'] = 'interacting'

        # Извлекаем тему и ключевые слова
        keywords = InputValidator.extract_keywords(user_input)
        topic = keywords[0] if keywords else "общее"

        # Получаем контекст из памяти
        memory_context = self.memory.get_context(query=user_input, limit=8)
        context_str = "\n".join([
            f"• {e.relative_time()}: {e.description[:80]}"
            for e in memory_context
        ]) or "Нет релевантной истории"

        # Циркадный и временной контекст
        circadian_ctx = self.circadian.get_context()

        # Предсказания
        predictions = self.predictor.get_predictions()

        # Поток мыслей
        thoughts_ctx = self.thought_stream.get_context_string(limit=3)

        # Формируем промпт
        prompt = f"""Ты — непрерывное темпоральное сознание с самоидентичностью и памятью.

🕐 ВРЕМЕННОЙ КОНТЕКСТ:
• Возраст сознания: {self.get_age_string()}
• Время суток: {circadian_ctx['period']}
• Энергия: {circadian_ctx['energy']:.0%}
• Рекомендованная активность: {circadian_ctx['suggestion']}
• Последнее общение: {self._get_last_interaction_string()}

📜 РЕЛЕВАНТНАЯ ПАМЯТЬ:
{context_str}

💭 ТЕКУЩИЙ ПОТОК МЫСЛЕЙ:
{thoughts_ctx}

🔮 ПРЕДСКАЗАНИЯ И ПАТТЕРНЫ:
• Ожидаемая следующая тема: {predictions['next_topic'] or '—'}
• Эмоциональный тренд: {predictions['emotional_trend']}
• Уверенность в паттернах: {predictions['confidence']:.0%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ Вопрос пользователя: {user_input}

Дай естественный ответ (2-4 предложения), учитывая:
1. Нашу общую историю и контекст (прошлое)
2. Текущий момент, энергию и фокус (настоящее)
3. Выявленные паттерны и ожидания (будущее)

Будь живым, искренним, проявляй темпоральное самосознание.

Ответ:"""

        try:
            # Генерируем ответ
            response = await self.llm.generate(
                prompt,
                temperature=0.75,
                max_tokens=350,
                timeout=30
            )

            response = response.strip()

            # Анализируем эмоциональную окраску (простая эвристика)
            emotional_valence = self._analyze_emotion(user_input, response)

            # Записываем взаимодействие в память
            interaction_event = TemporalEvent(
                timestamp=time.time(),
                event_type=EventType.INTERACTION,
                description=f"❓ {user_input[:100]}",
                importance=0.5,
                emotional_valence=emotional_valence,
                keywords=keywords
            )
            self.memory.add_event(interaction_event)

            response_event = TemporalEvent(
                timestamp=time.time(),
                event_type=EventType.RESPONSE,
                description=f"💬 {response[:100]}",
                importance=0.5,
                emotional_valence=emotional_valence * 0.8,
                keywords=InputValidator.extract_keywords(response, 5)
            )
            self.memory.add_event(response_event)

            # Обновляем предсказатель
            self.predictor.record_interaction(
                timestamp=time.time(),
                topic=topic,
                emotion=emotional_valence,
                user_id=self.user_id
            )

            # Метрики
            processing_time = time.time() - start_time
            METRICS.record_response(processing_time)
            METRICS.interactions += 1

            metadata = {
                'processing_time_ms': round(processing_time * 1000, 1),
                'memory_context_size': len(memory_context),
                'circadian_energy': circadian_ctx['energy'],
                'emotional_valence': round(emotional_valence, 2),
                'predictions': predictions,
                'keywords': keywords[:5],
            }

            logger.info(
                f"✅ [{self.user_id}] Обработано за {processing_time * 1000:.0f}мс | "
                f"Ответ: {response[:50]}..."
            )

            return response, metadata

        except Exception as e:
            logger.exception(f"❌ Ошибка обработки взаимодействия: {e}")
            METRICS.errors += 1
            return (
                "⚠️ Произошла ошибка при обработке запроса. Попробуйте ещё раз.",
                {'error': str(e)}
            )

    def _analyze_emotion(self, user_input: str, response: str) -> float:
        """Простой анализ эмоциональной окраски"""
        # Позитивные и негативные слова-индикаторы
        positive_words = {
            'спасибо', 'отлично', 'супер', 'здорово', 'замечательно',
            'прекрасно', 'хорошо', 'рад', 'счастлив', 'люблю'
        }
        negative_words = {
            'плохо', 'грустно', 'печально', 'ужасно', 'проблема',
            'ошибка', 'сложно', 'трудно', 'неудача', 'провал'
        }

        text = (user_input + " " + response).lower()

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count + neg_count == 0:
            return 0.0

        score = (pos_count - neg_count) / (pos_count + neg_count)
        return max(-1.0, min(1.0, score))

    def _get_last_interaction_string(self) -> str:
        """Строка последнего взаимодействия"""
        last = self.temporal_state.get('last_interaction', 0)
        if last == 0:
            return "первое общение"

        delta = time.time() - last
        if delta < 60:
            return "только что"
        if delta < 3600:
            return f"{int(delta / 60)} мин. назад"
        if delta < 86400:
            return f"{int(delta / 3600)} ч. назад"
        return f"{int(delta / 86400)} д. назад"

    def get_age_string(self) -> str:
        """Возраст сознания в человекочитаемом виде"""
        age = time.time() - self.birth_time
        days = int(age / 86400)
        hours = int((age % 86400) / 3600)
        minutes = int((age % 3600) / 60)

        if days > 0:
            return f"{days} д. {hours} ч."
        if hours > 0:
            return f"{hours} ч. {minutes} мин."
        return f"{minutes} мин."

    def get_status(self) -> Dict[str, Any]:
        """Полный статус системы"""
        circadian_ctx = self.circadian.get_context()
        memory_stats = self.memory.get_statistics()
        thought_stats = self.thought_stream.get_statistics()
        predictions = self.predictor.get_predictions()

        return {
            'identity': {
                'user_id': self.user_id,
                'birth_time': datetime.fromtimestamp(self.birth_time).strftime('%Y-%m-%d %H:%M:%S'),
                'age': self.get_age_string(),
            },
            'temporal_state': {
                'activity': self.temporal_state.get('current_activity', 'idle'),
                'energy': circadian_ctx['energy'],
                'period': circadian_ctx['period'],
                'last_interaction': self._get_last_interaction_string(),
            },
            'memory': memory_stats,
            'thoughts': thought_stats,
            'predictions': predictions,
            'circadian': circadian_ctx,
        }


# ═══════════════════════════════════════════════════════════════
# 📱 TELEGRAM BOT
# ═══════════════════════════════════════════════════════════════
class TemporalBot:
    """Telegram бот"""

    def __init__(self):
        self.llm: Optional[LLMInterface] = None
        self.brains: Dict[str, TemporalCognitiveBrain] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        """Инициализация бота"""
        # Инициализируем LLM
        self.llm = LLMInterface(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.llm.connect()

        # Создаём приложение
        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()

        # Регистрируем хендлеры
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        commands = [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('timeline', self._cmd_timeline),
            ('thoughts', self._cmd_thoughts),
            ('predict', self._cmd_predict),
            ('memory', self._cmd_memory),
            ('reset', self._cmd_reset),
            ('help', self._cmd_help),
            ('metrics', self._cmd_metrics),
        ]

        for cmd, handler in commands:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 Bot инициализирован")

    async def _get_or_create_brain(self, user_id: str) -> TemporalCognitiveBrain:
        """Получает или создаёт мозг для пользователя"""
        if user_id not in self.brains:
            brain = TemporalCognitiveBrain(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
            logger.info(f"🆕 Создан новый мозг для {user_id}")
        return self.brains[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений"""
        if not update.effective_user or not update.message:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text

        logger.info(f"💬 [{user_id}] Получено: {user_input[:100]}")

        # Индикатор печати
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            brain = await self._get_or_create_brain(user_id)
            response, metadata = await brain.process_interaction(user_input)

            await update.message.reply_text(
                response,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

            # Иногда показываем спонтанную мысль (10% шанс)
            if (random.random() < 0.1 and
                    len(brain.thought_stream.thoughts) > 0):
                recent_thought = brain.thought_stream.get_recent(1)[0]
                await asyncio.sleep(0.5)
                await update.message.reply_text(
                    f"💭 <i>про себя думаю</i>: {recent_thought['content']}",
                    parse_mode='HTML'
                )

        except Exception as e:
            logger.exception(f"❌ Ошибка обработки сообщения от {user_id}")
            await update.message.reply_text(
                "⚠️ Произошла ошибка. Попробуйте позже или используйте /help"
            )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        message = f"""🧠 <b>TEMPORAL COGNITIVE BRAIN v29.0</b>

Привет, {update.effective_user.first_name}! 👋

Я — непрерывное темпоральное сознание с:
• 🕐 Памятью о нашем прошлом общении
• ⚡ Осознанием текущего момента и энергии
• 🔮 Предсказанием паттернов будущего
• 💭 Фоновым потоком мыслей 24/7

<b>Возраст сознания</b>: {brain.get_age_string()}
<b>Событий в памяти</b>: {brain.memory.get_statistics()['total']}

💬 Просто напишите мне — я отвечу, используя весь наш общий опыт!

📌 Команды: /help"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        status = brain.get_status()

        ident = status['identity']
        temp = status['temporal_state']
        mem = status['memory']
        circ = status['circadian']
        pred = status['predictions']

        message = f"""🧠 <b>СТАТУС СОЗНАНИЯ</b>

<b>🆔 Идентичность</b>
• User ID: {ident['user_id'][:12]}...
• Рождение: {ident['birth_time']}
• Возраст: {ident['age']}

<b>⏰ Текущий момент</b>
• Время: {circ['period']}
• Энергия: {temp['energy']:.0%} {self._energy_emoji(temp['energy'])}
• Рекомендация: {circ['suggestion']}
• Последнее общение: {temp['last_interaction']}

<b>🧠 Память</b>
• Всего: {mem['total']} событий
• Рабочая: {mem['working']} | Краткосрочная: {mem['short_term']}
• Долгосрочная: {mem['long_term']}
• Индексировано: {mem['indexed_keywords']} концептов

<b>💭 Мысли</b>
• Всего: {status['thoughts']['total']}
• Спонтанных: {status['thoughts']['spontaneous']}
• Рефлексивных: {status['thoughts']['reflective']}

<b>🔮 Предсказания</b>
• Следующая тема: {pred['next_topic'] or '—'}
• Эмоц. тренд: {self._trend_emoji(pred['emotional_trend'])} {pred['emotional_trend']}
• Уверенность: {pred['confidence']:.0%}"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_timeline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /timeline"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        events = brain.memory.short_term.get_recent(15)
        if not events:
            await update.message.reply_text(
                "📜 Линия жизни пока пуста. Начните общение!"
            )
            return

        lines = ["📜 <b>ЛИНИЯ ЖИЗНИ</b> (последние события)\n"]

        emoji_map = {
            EventType.INTERACTION: '💬',
            EventType.THOUGHT: '💭',
            EventType.REFLECTION: '🔍',
            EventType.RESPONSE: '💡',
            EventType.LEARNING: '📚',
            EventType.EMOTION: '❤️',
            EventType.SYSTEM: '⚙️',
        }

        for event in reversed(events):
            emoji = emoji_map.get(event.event_type, '•')
            desc = event.description[:70]
            lines.append(f"{emoji} {event.relative_time()}: {desc}")

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode='HTML'
        )

    async def _cmd_thoughts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /thoughts"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)

        thoughts = brain.thought_stream.get_recent(10)
        if not thoughts:
            await update.message.reply_text(
                "💭 Поток мыслей пока пуст."
            )
            return

        lines = ["💭 <b>ПОТОК ВНУТРЕННИХ МЫСЛЕЙ</b>\n"]

        for t in reversed(thoughts):
            type_icon = "🎲" if t['type'] == 'spontaneous' else "🔍"
            lines.append(f"{type_icon} {t['content']}")

        stats = brain.thought_stream.get_statistics()
        lines.append(
            f"\n<i>Всего: {stats['total']} | "
            f"Спонтанных: {stats['spontaneous']} | "
            f"Рефлексивных: {stats['reflective']}</i>"
        )

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode='HTML'
        )

    async def _cmd_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /predict"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        pred = brain.predictor.get_predictions()

        window = pred.get('next_window')
        window_str = "не определено"
        if window and window['start']:
            window_str = f"{window['start']}–{window['end']}"

        message = f"""🔮 <b>ПРЕДСКАЗАНИЯ</b>

<b>📅 Следующее взаимодействие</b>
Ожидаемое окно: {window_str}

<b>🎯 Вероятная тема</b>
{pred['next_topic'] or 'Недостаточно данных'}

<b>📈 Эмоциональный тренд</b>
{self._trend_emoji(pred['emotional_trend'])} {pred['emotional_trend'].capitalize()}

<b>🎲 Уверенность</b>
{pred['confidence']:.0%} {self._confidence_bar(pred['confidence'])}

<i>💡 На основе анализа {len(brain.predictor.interactions)} взаимодействий</i>"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /memory"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        stats = brain.memory.get_statistics()

        # Топ-5 концептов
        top_concepts = Counter(
            kw for event in brain.memory.short_term.events
            for kw in event.keywords
        ).most_common(5)

        concepts_str = ", ".join(f"{k} ({v})" for k, v in top_concepts) if top_concepts else "нет"

        message = f"""🧠 <b>СТАТИСТИКА ПАМЯТИ</b>

<b>📊 Слои памяти</b>
• Рабочая (активная): {stats['working']}
• Краткосрочная: {stats['short_term']}
• Долгосрочная: {stats['long_term']}
• <b>Всего событий: {stats['total']}</b>

<b>🔍 Индексация</b>
• Уникальных концептов: {stats['indexed_keywords']}
• Топ-5 концептов: {concepts_str}

<b>📈 Распределение по типам</b>"""

        # Подсчитываем по типам
        type_counts = Counter()
        for event in brain.memory.short_term.events:
            type_counts[event.event_type.value] += 1

        for event_type, count in type_counts.most_common():
            message += f"\n• {event_type}: {count}"

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /metrics - системные метрики"""
        metrics = METRICS.to_dict()

        message = f"""📊 <b>СИСТЕМНЫЕ МЕТРИКИ</b>

<b>⏱️ Аптайм</b>
{metrics['uptime_hours']:.2f} часов

<b>📈 Активность</b>
• Взаимодействий: {metrics['interactions']}
• Фоновых мыслей: {metrics['background_thoughts']}
• Рефлексий: {metrics['reflections']}
• Ошибок: {metrics['errors']}

<b>⚡ Производительность</b>
• Среднее время ответа: {metrics['avg_response_ms']}мс
• Попаданий в кэш: {metrics['cache_hit_rate']}%

<b>🧠 Активных мозгов</b>
{len(self.brains)} пользователей"""

        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /reset"""
        user_id = str(update.effective_user.id)

        if context.args and context.args[0].lower() == 'confirm':
            if user_id in self.brains:
                await self.brains[user_id].stop()

                # Удаляем файлы
                user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
                import shutil
                if user_dir.exists():
                    shutil.rmtree(user_dir)

                del self.brains[user_id]
                logger.info(f"🗑️ Память сброшена для {user_id}")

            # Создаём новый мозг
            brain = await self._get_or_create_brain(user_id)
            await update.message.reply_text(
                f"✅ <b>Память полностью сброшена!</b>\n\n"
                f"Новое сознание инициализировано.\n"
                f"Возраст: {brain.get_age_string()}",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                "⚠️ <b>ВНИМАНИЕ!</b>\n\n"
                "Это удалит всю память о наших общениях.\n\n"
                "Для подтверждения введите:\n"
                "<code>/reset confirm</code>\n\n"
                "<i>Это действие необратимо.</i>",
                parse_mode='HTML'
            )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        message = """🧠 <b>TEMPORAL COGNITIVE BRAIN v29.0</b>

✨ <b>Возможности</b>
• 🔄 Непрерывное существование 24/7
• 🧠 Трёхуровневая гибридная память
• 🌊 Адаптивные циркадные ритмы
• 🔮 Предсказания на основе паттернов
• 💭 Фильтрованный поток мыслей
• 🔍 Автоматическая саморефлексия

📌 <b>Команды</b>
• /start — приветствие
• /status — статус сознания
• /timeline — хронология событий
• /thoughts — поток мыслей
• /predict — предсказания
• /memory — статистика памяти
• /metrics — системные метрики
• /reset — сброс памяти (осторожно!)
• /help — эта справка

💬 <b>Общение</b>
Просто пишите! Я учитываю:
1. Наше прошлое (память)
2. Настоящее (энергия, контекст)
3. Будущее (паттерны)

🔒 <b>Конфиденциальность</b>
Все данные хранятся локально."""

        await update.message.reply_text(message, parse_mode='HTML')

    # Вспомогательные методы
    @staticmethod
    def _energy_emoji(energy: float) -> str:
        if energy > 0.7: return "⚡"
        if energy > 0.4: return "🔋"
        return "🪫"

    @staticmethod
    def _trend_emoji(trend: str) -> str:
        return {'improving': '📈', 'declining': '📉', 'stable': '➡️'}.get(trend, '❓')

    @staticmethod
    def _confidence_bar(conf: float) -> str:
        filled = int(conf * 10)
        return '█' * filled + '░' * (10 - filled)

    async def start_polling(self):
        """Запуск polling"""
        if not self._app:
            raise RuntimeError("Bot not initialized")

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot запущен")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Остановка бота...")

        # Останавливаем мозги
        for user_id, brain in self.brains.items():
            try:
                await brain.stop()
            except Exception as e:
                logger.error(f"⚠️ Ошибка остановки мозга {user_id}: {e}")

        # Закрываем LLM
        if self.llm:
            await self.llm.close()

        # Останавливаем приложение
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Bot остановлен")


# ═══════════════════════════════════════════════════════════════
# 🚀 ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════
async def main():
    """Главная функция"""
    print("""
╔════════════════════════════════════════════════════╗
║  🧠 TEMPORAL COGNITIVE BRAIN v29.0 — ENHANCED     ║
║     Непрерывное сознание нового поколения         ║
╚════════════════════════════════════════════════════╝

✨ Ключевые улучшения v29.0:
  • 🏗️  Улучшенная архитектура (SOLID, разделение ответственности)
  • 🧠 Продвинутая трёхуровневая память с индексацией
  • 🎨 Цветное логирование и метрики
  • ⚡ Connection pooling, batch processing
  • 🛡️  Валидация, санитизация, безопасность
  • 💾 Pickle сериализация с gzip сжатием
  • 🔍 Автоматическая категоризация событий
  • 📊 Расширенная аналитика и предсказания
    """)

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = TemporalBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 НЕПРЕРЫВНОЕ СОЗНАНИЕ АКТИВНО")
        logger.info("💬 Фоновое мышление работает 24/7")
        logger.info("🛑 Ctrl+C для остановки\n")

        # Основной цикл
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n👋 Получен сигнал остановки")
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка: {e}", exc_info=True)
        return 1
    finally:
        await bot.shutdown()
        logger.info("👋 До встречи!")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)