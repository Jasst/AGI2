# coding: utf-8
"""
AGI_Cognitive_Agent_LMStudio_v4.1.py — ИСПРАВЛЕННАЯ ВЕРСИЯ С АКТУАЛЬНЫМИ ДАННЫМИ
✅ Веб-поиск через DuckDuckGo (без API ключей)
✅ Системное время для запросов "какой сейчас год/дата/время" (100% актуально!)
✅ Исправлены ошибки кнопок и нормализации запросов
✅ Метакогнитивный мониторинг качества ответов
✅ Приоритизация мыслей по интеллектуальным критериям
✅ Декомпозиция сложных целей на подзадачи
✅ Эпизодическая память для ключевых событий
✅ 100% приватность для локальных запросов
"""
import asyncio
import logging
import os
import sys
import sqlite3
import hashlib
import re
import json
import time
import requests
import aiohttp
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, Optional, List, Any, Set, Tuple
import random

# ================= ДОПОЛНИТЕЛЬНЫЕ ИМПОРТЫ ДЛЯ ВЕБ-ПОИСКА И АНАЛИЗА =================
try:
    from ddgs import AsyncDDGS

    DDGS = AsyncDDGS
except ImportError:
    from duckduckgo_search import DDGS

try:
    from bs4 import BeautifulSoup

    HAS_WEB_SEARCH = True
    print("✅ Модули веб-поиска загружены (ddgs/dduckduckgo-search, beautifulsoup4)")
except ImportError as e:
    HAS_WEB_SEARCH = False
    print(f"⚠️ Модули веб-поиска недоступны: {e}")
    print("📦 Установите: pip install ddgs beautifulsoup4 lxml")


# ================= ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =================
def load_dotenv_simple(path: Path = Path(".env")):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            print(f"✅ Загружены переменные из {path}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки .env: {e}")


load_dotenv_simple()

# ================= СОВМЕСТИМОСТЬ С PYTHON 3.13 =================
if sys.version_info >= (3, 13):
    print("✅ Python 3.13 обнаружен — применяются настройки совместимости")
    if sys.platform == 'win32':
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass

# ================= ИМПОРТЫ TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )

    print("✅ Библиотека python-telegram-bot загружена успешно (v20+)")
except ImportError as e:
    print(f"❌ Ошибка импорта telegram: {e}")
    print("📦 Установите: pip install 'python-telegram-bot>=20.7' aiohttp requests")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ ДЛЯ ЛОКАЛЬНОГО СЕРВЕРА =================
class Config:
    ROOT = Path("./cognitive_system_telegram")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"
    CACHE_PATH = ROOT / "cache.json"
    LOG_PATH = ROOT / "system.log"

    # 🔥 ЛОКАЛЬНЫЙ СЕРВЕР LM STUDIO
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODELS_URL = "http://localhost:1234/v1/models"
    TIMEOUT = 180
    MAX_TOKENS = 4096

    # Когнитивные параметры
    REFLECTION_INTERVAL = 5
    DEEP_THINKING_THRESHOLD = 0.7
    CONTEXT_WINDOW_SIZE = 12
    MEMORY_DECAY_RATE = 0.07
    MAX_MEMORY_ITEMS = 5000

    # Параметры поиска
    WEB_SEARCH_ENABLED = HAS_WEB_SEARCH
    SEARCH_TRIGGERS = ['найди', 'поищи', 'что говорят', 'последние новости', 'актуальная информация',
                       'сейчас', 'сегодня', 'новост', 'текущая ситуация']
    SEARCH_THRESHOLD = 0.65

    # Параметры бота
    MAX_MESSAGE_LENGTH = 4096
    SESSION_TIMEOUT = 7200

    # Типы мышления
    THOUGHT_TYPES = ['рефлексия', 'анализ', 'планирование', 'обучение', 'наблюдение']

    @classmethod
    def get_telegram_token(cls) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("\n" + "=" * 70)
            print("🔑 НЕОБХОДИМ ТОКЕН TELEGRAM БОТА")
            print("=" * 70)
            print("\nКак получить токен:")
            print("1. Найдите в Telegram бота @BotFather")
            print("2. Отправьте команду /newbot и следуйте инструкциям")
            print("3. Скопируйте токен вида '123456789:AAH_ABC123...'")
            token = input("\nВведите токен Telegram бота: ").strip()
            if not token:
                raise ValueError("❌ Токен не введён. Запуск невозможен.")
            if not re.match(r'^\d+:[A-Za-z0-9_-]{35,}$', token):
                raise ValueError("❌ Неверный формат токена. Пример: 123456789:AAHdQwerty12345uiop67890")
            try:
                with open(".env", "a", encoding="utf-8") as f:
                    f.write(f'\nTELEGRAM_BOT_TOKEN="{token}"\n')
                print("✅ Токен сохранён в .env")
                os.environ["TELEGRAM_BOT_TOKEN"] = token
            except Exception as e:
                print(f"⚠️ Не удалось сохранить токен: {e}")
        return token

    @classmethod
    def get_lmstudio_config(cls) -> Dict[str, Any]:
        config = {
            'url': cls.LM_STUDIO_URL,
            'api_key': os.getenv("LM_STUDIO_API_KEY", ""),
            'model': os.getenv("LM_STUDIO_MODEL", "local-model")
        }
        if not os.getenv("LM_STUDIO_CONFIGURED"):
            print("\n" + "=" * 70)
            print("⚙️  НАСТРОЙКА ЛОКАЛЬНОГО СЕРВЕРА LM STUDIO")
            print("=" * 70)
            print("\n🔍 Проверка сервера LM Studio...")
            try:
                response = requests.get(cls.LM_STUDIO_MODELS_URL, timeout=8)
                if response.status_code == 200:
                    print("✅ Сервер обнаружен!")
                else:
                    raise ConnectionError(f"Статус {response.status_code}")
            except Exception as e:
                print(f"⚠️ Сервер не отвечает: {e}")
                print("\n📌 ЗАПУСТИТЕ LM STUDIO:")
                print("   1. Скачайте: https://lmstudio.ai/")
                print("   2. Загрузите модель (рекомендуется):")
                print("      • Phi-3-mini-4k-instruct-q4.gguf (быстро, 3.8B) — для слабых ПК")
                print("      • Mistral-7B-Instruct-v0.2-Q5_K_M.gguf (качество, 7B) — баланс")
                print("      • Nous-Hermes-2-Mixtral-8x7B (46B) — максимальное качество")
                print("   3. Включите сервер: вкладка 'Server' → 'Start Server'")
                input("\nНажмите Enter после запуска сервера...")
            print("\n🔍 Поиск моделей...")
            try:
                response = requests.get(cls.LM_STUDIO_MODELS_URL, timeout=10)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    if models:
                        best = next((m for m in models if
                                     'instruct' in m.get('id', '').lower() or 'chat' in m.get('id', '').lower()),
                                    models[0])
                        config['model'] = best.get('id', config['model'])
                        print(f"✅ Обнаружена модель: {config['model']}")
                        with open(".env", "a", encoding="utf-8") as f:
                            f.write(f'\nLM_STUDIO_CONFIGURED="true"\nLM_STUDIO_MODEL="{config["model"]}"\n')
                        print("💾 Конфигурация сохранена в .env")
                    else:
                        config['model'] = input("Введите ID модели вручную: ").strip() or config['model']
                else:
                    config['model'] = input("Введите ID модели вручную: ").strip() or config['model']
            except Exception as e:
                print(f"⚠️ Ошибка при определении модели: {e}")
                config['model'] = input("Введите ID модели вручную: ").strip() or config['model']
            print("\n" + "=" * 70)
            print("💡 РЕКОМЕНДАЦИИ ПО ПРОИЗВОДИТЕЛЬНОСТИ:")
            print("   • CPU (16 ГБ ОЗУ): используйте Phi-3-mini — 2-3 сек/ответ")
            print("   • CPU (32 ГБ ОЗУ): используйте Mistral-7B — 4-6 сек/ответ")
            print("   • GPU (8+ ГБ VRAM): скорость возрастёт в 5-10 раз")
            if Config.WEB_SEARCH_ENABLED:
                print("   • 🔍 Веб-поиск: активирован (бесплатно через DuckDuckGo)")
            else:
                print("   • ⚠️ Веб-поиск: отключён (установите ddgs)")
            print("=" * 70 + "\n")
        return config


# ================= УТИЛИТЫ ОБРАБОТКИ ТЕКСТА =================
def calculate_text_similarity(text1: str, text2: str) -> float:
    """Расчёт схожести текстов с учётом n-грамм"""
    if not text1 or not text2:
        return 0.0

    def get_ngrams(text: str, n: int = 2) -> Set[str]:
        words = re.findall(r'\w+', text.lower())
        if len(words) < n:
            return set([' '.join(words)])
        return set(' '.join(words[i:i + n]) for i in range(len(words) - n + 1))

    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0

    unigram_sim = len(words1 & words2) / max(len(words1), len(words2))
    bigrams1 = get_ngrams(text1, 2)
    bigrams2 = get_ngrams(text2, 2)
    bigram_sim = len(bigrams1 & bigrams2) / max(len(bigrams1 | bigrams2), 1) if bigrams1 and bigrams2 else 0.0

    return 0.6 * unigram_sim + 0.4 * bigram_sim


def extract_semantic_features(text: str) -> Dict[str, Any]:
    """Извлечение семантических характеристик"""
    text_lower = text.lower()
    words = text.split()
    features = {
        'length': len(words),
        'complexity': len(set(text_lower.split())) / max(len(words), 1),
        'question_words': len(re.findall(r'\b(как|что|почему|зачем|когда|где|кто|сколько)\b', text_lower)),
        'numbers': len(re.findall(r'\b\d+\b', text)),
        'imperatives': len(
            re.findall(r'\b(сделай|создай|найди|покажи|расскажи|объясни|запомни|сохрани)\b', text_lower)),
        'has_question': '?' in text,
        'sentiment': analyze_sentiment(text)
    }
    return features


def analyze_sentiment(text: str) -> float:
    """Простой анализ тональности (-1 до 1)"""
    positive = ['хорошо', 'отлично', 'прекрасно', 'замечательно', 'классно', 'супер', 'рад', 'счастлив', 'восхищён',
                'люблю', 'приятно', 'удовлетворён']
    negative = ['плохо', 'ужасно', 'отвратительно', 'кошмар', 'провал', 'грустно', 'ненавижу', 'злой', 'разочарован',
                'раздражён', 'печально']
    text_lower = text.lower()
    pos_count = sum(1 for word in positive if word in text_lower)
    neg_count = sum(1 for word in negative if word in text_lower)
    total = pos_count + neg_count
    return (pos_count - neg_count) / total if total > 0 else 0.0


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Извлечение сущностей из текста"""
    entities = {
        'numbers': re.findall(r'\b\d+\b', text),
        'dates': re.findall(
            r'\b\d{1,2}[./]\d{1,2}[./]?\d{2,4}\b|\b(?:янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\b',
            text, re.IGNORECASE
        ),
        'names': re.findall(r'\b(?:[А-Я][а-я]+)\b', text),
        'emails': re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text),
        'urls': re.findall(r'https?://\S+', text)
    }
    return {k: v for k, v in entities.items() if v}


def split_message(text: str, max_length: int = 4096) -> list:
    """Разбивает длинное сообщение на части"""
    if not text:
        return [""]
    if len(text) <= max_length:
        return [text]

    parts = []
    current = ""
    paragraphs = re.split(r'(\n\s*\n)', text)

    for para in paragraphs:
        if len(current) + len(para) <= max_length:
            current += para
        else:
            if current:
                parts.append(current.rstrip())
            if len(para) > max_length:
                sentences = re.split(r'([.!?]+)', para)
                temp = ""
                for i in range(0, len(sentences), 2):
                    chunk = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                    if len(temp) + len(chunk) <= max_length:
                        temp += chunk
                    else:
                        if temp:
                            parts.append(temp.rstrip())
                        temp = chunk
                if temp:
                    current = temp
            else:
                current = para

    if current:
        parts.append(current.rstrip())

    if len(parts) > 5:
        parts = parts[:5]
        parts.append("📝 ...сообщение сокращено из-за ограничений Telegram")

    return parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    """Создание главной клавиатуры"""
    keyboard = [
        [
            InlineKeyboardButton("🧠 Глубокое мышление", callback_data="deep_think"),
            InlineKeyboardButton("🔍 Поиск", callback_data="search")
        ],
        [
            InlineKeyboardButton("📊 Статистика", callback_data="stats"),
            InlineKeyboardButton("💡 Инсайты", callback_data="insights")
        ],
        [
            InlineKeyboardButton("🎯 Цели", callback_data="goals"),
            InlineKeyboardButton("🔗 Паттерны", callback_data="patterns")
        ],
        [
            InlineKeyboardButton("🧹 Очистить", callback_data="clear")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= ДВИЖОК ВЕБ-ПОИСКА =================
class WebSearchEngine:
    """Движок веб-поиска с кэшированием результатов"""

    def __init__(self, cache_ttl: int = 3600):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.search_count = 0
        self.last_search_time = 0
        self.ddgs = DDGS() if HAS_WEB_SEARCH else None

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Поиск в интернете с кэшированием"""
        if not HAS_WEB_SEARCH:
            return []

        cache_key = hashlib.md5(query.encode()).hexdigest()

        # Проверка кэша
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logging.info(f"🔍 Кэшированный поиск для: {query[:30]}")
                return cached_results

        try:
            # Ограничение частоты запросов
            elapsed = time.time() - self.last_search_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)

            results = []
            search_results = await self.ddgs.text(query, max_results=max_results, safesearch='moderate')

            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })

            # Кэшируем результаты
            self.cache[cache_key] = (time.time(), results)
            self.search_count += 1
            self.last_search_time = time.time()

            logging.info(f"🌐 Найдено {len(results)} результатов для: {query[:30]}")
            return results

        except Exception as e:
            logging.error(f"Ошибка поиска: {e}")
            return []

    async def fetch_page_content(self, url: str, max_length: int = 5000) -> str:
        """Извлечение текста со страницы"""
        if not url or not url.startswith(('http://', 'https://')):
            return ""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Удаляем скрипты, стили и навигацию
                        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                            tag.decompose()

                        # Извлекаем основной контент
                        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(
                            'content|article'))
                        if main_content:
                            text = main_content.get_text()
                        else:
                            text = soup.get_text()

                        # Очистка текста
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)

                        return text[:max_length]
        except Exception as e:
            logging.error(f"Ошибка загрузки {url[:50]}: {e}")
            return ""


# ================= СИСТЕМА ПРИОРИТИЗАЦИИ МЫСЛЕЙ =================
class ThoughtPrioritizer:
    """Интеллектуальная приоритизация мыслей"""

    def __init__(self):
        self.priority_weights = {
            'новизна': 0.3,
            'релевантность': 0.25,
            'практичность': 0.2,
            'глубина': 0.15,
            'креативность': 0.1
        }
        self.thought_history = deque(maxlen=50)

    def calculate_priority(self, thought: str, context: str, history: Optional[List[str]] = None) -> float:
        """Расчёт приоритета мысли"""
        if history is None:
            history = list(self.thought_history)

        score = 0.0

        # Новизна: отличается ли от предыдущих мыслей
        if history:
            similarities = [calculate_text_similarity(thought, h) for h in history[-5:] if h]
            novelty = min(similarities) if similarities else 0.0
            score += (1 - novelty) * self.priority_weights['новизна']
        else:
            score += 0.8 * self.priority_weights['новизна']  # Первая мысль — высокий приоритет

        # Релевантность контексту
        relevance = calculate_text_similarity(thought, context)
        score += relevance * self.priority_weights['релевантность']

        # Практичность: есть ли конкретные действия
        action_words = ['сделать', 'попробовать', 'начать', 'создать', 'применить', 'использовать', 'разработать']
        has_action = any(word in thought.lower() for word in action_words)
        score += (0.9 if has_action else 0.3) * self.priority_weights['практичность']

        # Глубина: длина и сложность
        words = thought.split()
        depth = min(len(words) / 50, 1.0)
        score += depth * self.priority_weights['глубина']

        # Креативность: уникальные слова/концепции
        unique_words = len(set(words)) / max(len(words), 1)
        creativity = unique_words * 0.7 + (1 if len(words) > 15 else 0.5) * 0.3
        score += creativity * self.priority_weights['креативность']

        final_score = min(1.0, score)
        self.thought_history.append(thought)

        return final_score


# ================= МЕТАКОГНИТИВНЫЙ МОНИТОР =================
class MetaCognitiveMonitor:
    """Мониторинг собственного мышления и качества ответов"""

    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.error_patterns = defaultdict(int)
        self.improvement_suggestions = []

    async def evaluate_response_quality(self, user_input: str, response: str, context: str = "") -> Dict[str, float]:
        """Оценка качества ответа по нескольким метрикам"""
        metrics = {
            'coherence': self._check_coherence(response),
            'relevance': calculate_text_similarity(user_input, response),
            'completeness': self._check_completeness(response, user_input),
            'clarity': self._check_clarity(response),
            'context_utilization': self._check_context_utilization(response, context) if context else 0.7
        }

        # Общий балл
        overall = sum(metrics.values()) / len(metrics)
        metrics['overall'] = overall

        # Сохраняем в историю
        self.performance_history.append({
            'timestamp': time.time(),
            'user_input': user_input[:50],
            'metrics': metrics,
            'overall': overall
        })

        # Анализ трендов и генерация предложений
        if len(self.performance_history) % 10 == 0:
            self.improvement_suggestions = self._generate_improvement_suggestions()

        return metrics

    def _check_coherence(self, text: str) -> float:
        """Проверка связности текста"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 1.0

        coherence_score = 0.0
        for i in range(len(sentences) - 1):
            similarity = calculate_text_similarity(sentences[i], sentences[i + 1])
            coherence_score += similarity

        return min(1.0, coherence_score / (len(sentences) - 1) * 1.2)

    def _check_completeness(self, response: str, query: str) -> float:
        """Проверка полноты ответа"""
        query_features = extract_semantic_features(query)
        response_words = len(response.split())

        # Если вопрос — ожидаем ответ определённой длины
        if query_features['has_question']:
            min_words = 25 if query_features['question_words'] > 1 else 15
            return min(response_words / min_words, 1.0)

        # Для утверждений — короткий ответ допустим
        return min(response_words / 10, 1.0)

    def _check_clarity(self, text: str) -> float:
        """Проверка ясности"""
        words = text.split()
        if not words:
            return 0.0

        # Средняя длина предложения
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_sentence_length = len(words) / max(len(sentences), 1)

        # Оптимально: 8-20 слов на предложение
        if 8 <= avg_sentence_length <= 20:
            clarity = 1.0
        else:
            deviation = abs(avg_sentence_length - 14) / 14
            clarity = max(0.3, 1.0 - deviation)

        # Штраф за сложные слова
        complex_words = len([w for w in words if len(w) > 12])
        complexity_penalty = min(complex_words / max(len(words), 1) * 0.3, 0.3)

        return max(0.3, clarity - complexity_penalty)

    def _check_context_utilization(self, response: str, context: str) -> float:
        """Проверка использования контекста в ответе"""
        if not context or not response:
            return 0.5

        # Извлекаем ключевые сущности из контекста
        context_entities = extract_entities(context)
        context_keywords = set(re.findall(r'\b\w{4,}\b', context.lower()))

        # Проверяем их присутствие в ответе
        response_lower = response.lower()
        matches = sum(1 for kw in context_keywords if kw in response_lower)

        utilization = min(matches / max(len(context_keywords), 1), 1.0)
        return utilization * 0.8 + 0.2  # Минимум 20% даже без совпадений

    def _generate_improvement_suggestions(self) -> List[str]:
        """Предложения по улучшению на основе анализа истории"""
        if len(self.performance_history) < 10:
            return ["Продолжайте диалог для анализа качества ответов"]

        recent = list(self.performance_history)[-20:]
        avg_metrics = {}
        for metric in ['coherence', 'relevance', 'completeness', 'clarity', 'context_utilization']:
            avg_metrics[metric] = sum(h['metrics'].get(metric, 0) for h in recent) / len(recent)

        suggestions = []
        if avg_metrics['coherence'] < 0.65:
            suggestions.append("💡 Улучшить связность: добавлять больше логических переходов между идеями")
        if avg_metrics['relevance'] < 0.7:
            suggestions.append("🎯 Повысить релевантность: точнее фокусироваться на сути запроса")
        if avg_metrics['completeness'] < 0.6:
            suggestions.append("📝 Давать более полные ответы на вопросы (минимум 2-3 предложения)")
        if avg_metrics['clarity'] < 0.65:
            suggestions.append("✨ Упростить формулировки: избегать сложных конструкций и жаргона")
        if avg_metrics['context_utilization'] < 0.5:
            suggestions.append("🧠 Лучше использовать контекст: ссылаться на предыдущие темы диалога")

        return suggestions or ["✅ Качество ответов стабильно высокое!"]

    def get_summary(self) -> str:
        """Краткий отчёт о метакогнитивной деятельности"""
        if not self.performance_history:
            return "Метакогнитивный мониторинг: недостаточно данных"

        recent = list(self.performance_history)[-10:]
        avg_overall = sum(h['overall'] for h in recent) / len(recent)
        trend = "↑" if len(self.performance_history) > 20 and \
                       sum(h['overall'] for h in list(self.performance_history)[-10:]) > \
                       sum(h['overall'] for h in list(self.performance_history)[-20:-10]) else "→"

        suggestions_text = "\n".join([f"   • {s}" for s in self.improvement_suggestions[
                                                           :3]]) if self.improvement_suggestions else "   Нет активных рекомендаций"

        return (
            f"🧠 МЕТАКОГНИТИВНЫЙ ОТЧЁТ\n{'=' * 40}\n"
            f"📊 Среднее качество: {avg_overall:.2%} {trend}\n"
            f"📈 Проанализировано ответов: {len(self.performance_history)}\n"
            f"💡 Рекомендации:\n{suggestions_text}"
        )


# ================= СИСТЕМА ДЕКОМПОЗИЦИИ ЦЕЛЕЙ =================
class GoalDecomposer:
    """Разбиение сложных целей на подцели"""

    def __init__(self, thinker: Any):
        self.thinker = thinker

    async def decompose_goal(self, goal_description: str, context: str = "") -> List[Dict[str, Any]]:
        """Декомпозиция цели на подзадачи"""
        if not goal_description or len(goal_description) < 10:
            return []

        decomposition_prompt = f"""
Цель пользователя: {goal_description}
Контекст: {context[:200]}

Разбей эту цель на 3-5 конкретных, измеримых, достижимых подзадач.
Для каждой подзадачи укажи:
1. Краткое описание (одно предложение)
2. Примерный срок выполнения (дни/недели)
3. Зависимости от других подзадач (если есть)

ВАЖНО: Ответ должен быть в формате JSON:
{{
  "subgoals": [
    {{
      "description": "описание",
      "timeframe": "срок",
      "dependencies": "зависимости или 'нет'"
    }}
  ]
}}
"""

        try:
            response = await self.thinker.call_llm(
                "Ты — эксперт по планированию и управлению проектами. Разбивай цели на чёткие, выполнимые шаги.",
                decomposition_prompt,
                temperature=0.3
            )

            # Извлекаем JSON из ответа
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                subgoals = data.get('subgoals', [])

                # Валидация и нормализация
                validated = []
                for i, sg in enumerate(subgoals[:5], 1):
                    validated.append({
                        'id': i,
                        'description': sg.get('description', f'Подзадача {i}')[:100],
                        'timeframe': sg.get('timeframe', 'не указано')[:30],
                        'dependencies': sg.get('dependencies', 'нет')[:50],
                        'progress': 0.0,
                        'status': 'pending'
                    })
                return validated

            # Если не удалось распарсить JSON — fallback на простой парсинг
            lines = response.strip().split('\n')
            subgoals = []
            for line in lines:
                if any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.', '-', '*']):
                    desc = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                    if desc and len(desc) > 10:
                        subgoals.append({
                            'id': len(subgoals) + 1,
                            'description': desc[:100],
                            'timeframe': '1 неделя',
                            'dependencies': 'нет',
                            'progress': 0.0,
                            'status': 'pending'
                        })

            return subgoals[:5]

        except Exception as e:
            logging.error(f"Ошибка декомпозиции цели: {e}")
            return [
                {
                    'id': 1,
                    'description': 'Уточнить детали цели',
                    'timeframe': '1 день',
                    'dependencies': 'нет',
                    'progress': 0.0,
                    'status': 'pending'
                },
                {
                    'id': 2,
                    'description': 'Разработать план действий',
                    'timeframe': '2 дня',
                    'dependencies': '1',
                    'progress': 0.0,
                    'status': 'pending'
                }
            ]


# ================= ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ =================
class EpisodicMemory:
    """Эпизодическая память — запоминание ключевых событий"""

    def __init__(self, db: Any):
        self.db = db
        self.episode_threshold = 0.75  # Минимальная важность для сохранения эпизода
        self.episodes_count = 0

    def add_episode(self, user_id: int, description: str, importance: float, context: Dict[str, Any]):
        """Сохранение эпизода"""
        if importance >= self.episode_threshold and len(description) > 15:
            try:
                episode_data = {
                    'description': description[:200],
                    'context_summary': context.get('summary', '')[:100],
                    'emotional_tone': context.get('emotion', 'neutral'),
                    'timestamp': time.time(),
                    'interaction_id': context.get('interaction_id', 0)
                }

                self.db.add_fact(
                    user_id,
                    f"episode_{int(time.time())}",
                    json.dumps(episode_data, ensure_ascii=False),
                    category='эпизод',
                    importance=min(1.0, importance * 1.2),
                    confidence=0.95
                )
                self.episodes_count += 1
                logging.info(f"🧠 Сохранён эпизод: {description[:50]}")
            except Exception as e:
                logging.error(f"Ошибка сохранения эпизода: {e}")

    def recall_similar_episodes(self, user_id: int, current_situation: str, limit: int = 3) -> List[Dict]:
        """Поиск похожих эпизодов из прошлого"""
        try:
            all_episodes = self.db.search_facts(user_id, "episode_", limit=50)

            scored_episodes = []
            for episode in all_episodes:
                try:
                    episode_data = json.loads(episode['value'])
                    similarity = calculate_text_similarity(current_situation, episode_data['description'])
                    if similarity > 0.4:  # Порог схожести
                        scored_episodes.append((similarity, episode_data))
                except:
                    continue

            scored_episodes.sort(reverse=True, key=lambda x: x[0])
            return [ep[1] for ep in scored_episodes[:limit]]
        except Exception as e:
            logging.error(f"Ошибка поиска эпизодов: {e}")
            return []


# ================= КЭШ ОТВЕТОВ ДЛЯ ЛОКАЛЬНЫХ МОДЕЛЕЙ =================
class ResponseCache:
    """Простой кэш ответов для ускорения повторяющихся запросов"""

    def __init__(self, max_size: int = 150):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.load()

    def _make_key(self, system_prompt: str, user_prompt: str) -> str:
        content = f"{system_prompt[:100]}|{user_prompt[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        key = self._make_key(system_prompt, user_prompt)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, system_prompt: str, user_prompt: str, response: str):
        key = self._make_key(system_prompt, user_prompt)
        self.cache[key] = response
        self.access_times[key] = time.time()

        if len(self.cache) > self.max_size:
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:self.max_size // 4]]
            for k in keys_to_remove:
                self.cache.pop(k, None)
                self.access_times.pop(k, None)

        self.save()

    def save(self):
        try:
            data = {'cache': self.cache, 'access_times': self.access_times}
            with open(Config.CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            pass

    def load(self):
        if Config.CACHE_PATH.exists():
            try:
                with open(Config.CACHE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
            except:
                pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'usage_percent': len(self.cache) / self.max_size * 100,
            'hits': getattr(self, 'hits', 0),
            'misses': getattr(self, 'misses', 0)
        }


# ================= РАСШИРЕННАЯ БАЗА ДАННЫХ =================
class EnhancedMemoryDB:
    """Продвинутая база данных с поддержкой всех когнитивных функций"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()
        self.fact_cache = {}
        self.pattern_cache = {}

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                user_id INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                system_response TEXT NOT NULL,
                context TEXT,
                emotion TEXT DEFAULT 'neutral',
                category TEXT,
                importance REAL DEFAULT 0.5,
                complexity REAL DEFAULT 0.5,
                used_web_search INTEGER DEFAULT 0
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT,
                confidence REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                last_used REAL,
                usage_count INTEGER DEFAULT 0,
                decay_factor REAL DEFAULT 1.0,
                UNIQUE(user_id, key, value)
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                thought_type TEXT NOT NULL,
                content TEXT NOT NULL,
                trigger TEXT,
                importance REAL DEFAULT 0.5,
                depth_level INTEGER DEFAULT 1,
                priority_score REAL DEFAULT 0.5
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                description TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                progress REAL DEFAULT 0.0,
                next_action TEXT,
                subgoals TEXT DEFAULT '[]'
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                last_seen REAL NOT NULL
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                timestamp REAL NOT NULL,
                results_count INTEGER DEFAULT 0
            )''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, importance DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoughts_user ON thoughts(user_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status, priority DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_user ON search_history(user_id, timestamp DESC)')
            conn.commit()

    # === Взаимодействия ===
    def add_interaction(self, user_id: int, user_input: str, system_response: str, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO interactions
            (timestamp, user_id, user_input, system_response, context, emotion, category, importance, complexity, used_web_search)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                user_id,
                user_input,
                system_response,
                kwargs.get('context', ''),
                kwargs.get('emotion', 'neutral'),
                kwargs.get('category', 'диалог'),
                kwargs.get('importance', 0.5),
                kwargs.get('complexity', 0.5),
                kwargs.get('used_web_search', 0)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_contextual_interactions(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT * FROM interactions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (user_id, limit * 3))
            all_interactions = [dict(row) for row in cursor.fetchall()]

            if not all_interactions:
                return []

            scored = []
            for interaction in all_interactions:
                relevance = calculate_text_similarity(
                    query,
                    interaction['user_input'] + ' ' + interaction['system_response']
                )
                recency = 1.0 - (time.time() - interaction['timestamp']) / (7 * 24 * 3600)
                recency = max(0, min(1, recency))
                score = 0.6 * relevance + 0.3 * interaction['importance'] + 0.1 * recency
                scored.append((score, interaction))

            scored.sort(reverse=True, key=lambda x: x[0])
            return [item[1] for item in scored[:limit]]

    # === Факты ===
    def add_fact(self, user_id: int, key: str, value: str, **kwargs):
        cache_key = f"{user_id}_{key}_{value}"
        if cache_key in self.fact_cache:
            del self.fact_cache[cache_key]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM facts WHERE user_id = ? AND key = ? AND value = ?',
                           (user_id, key, value))
            existing = cursor.fetchone()

            if existing:
                cursor.execute('''
                UPDATE facts
                SET confidence = ?, importance = ?, last_used = ?,
                    usage_count = usage_count + 1, decay_factor = 1.0
                WHERE id = ?
                ''', (
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(),
                    existing[0]
                ))
            else:
                cursor.execute('''
                INSERT INTO facts
                (user_id, key, value, category, confidence, importance, created_at, last_used,
                 usage_count, decay_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, key, value,
                    kwargs.get('category', ''),
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(), time.time(), 1, 1.0
                ))
            conn.commit()

    def get_relevant_facts(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        cache_key = f"{user_id}_{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.fact_cache:
            cached_time, result = self.fact_cache[cache_key]
            if time.time() - cached_time < 300:
                return result

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE facts
            SET decay_factor = decay_factor * (1 - ?)
            WHERE user_id = ? AND last_used < ?
            ''', (Config.MEMORY_DECAY_RATE, user_id, time.time() - 86400))

            cursor.execute('''
            SELECT * FROM facts
            WHERE user_id = ? AND confidence > 0.3 AND decay_factor > 0.1
            ORDER BY importance DESC, usage_count DESC
            LIMIT ?
            ''', (user_id, limit * 2))
            all_facts = [dict(row) for row in cursor.fetchall()]

            if not all_facts:
                return []

            scored = []
            for fact in all_facts:
                relevance = calculate_text_similarity(query, f"{fact['key']} {fact['value']}")
                score = (
                        0.4 * relevance +
                        0.3 * fact['importance'] +
                        0.2 * fact['confidence'] +
                        0.1 * fact['decay_factor']
                )
                scored.append((score, fact))

            scored.sort(reverse=True, key=lambda x: x[0])
            result = [item[1] for item in scored[:limit]]
            self.fact_cache[cache_key] = (time.time(), result)
            return result

    def search_facts(self, user_id: int, query_text: str, limit: int = 10) -> List[Dict]:
        """Поиск фактов по тексту"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            search_term = f"%{query_text}%"
            cursor.execute('''
            SELECT * FROM facts
            WHERE user_id = ? AND (key LIKE ? OR value LIKE ?)
            ORDER BY usage_count DESC, confidence DESC
            LIMIT ?
            ''', (user_id, search_term, search_term, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === Мысли ===
    def add_thought(self, user_id: int, thought_type: str, content: str, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO thoughts
            (user_id, timestamp, thought_type, content, trigger, importance, depth_level, priority_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                time.time(),
                thought_type,
                content[:300],
                kwargs.get('trigger', ''),
                kwargs.get('importance', 0.5),
                kwargs.get('depth_level', 1),
                kwargs.get('priority_score', 0.5)
            ))
            conn.commit()

    def get_recent_thoughts(self, user_id: int, limit: int = 10,
                            thought_type: Optional[str] = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if thought_type:
                cursor.execute('''
                SELECT * FROM thoughts
                WHERE user_id = ? AND thought_type = ?
                ORDER BY priority_score DESC, timestamp DESC
                LIMIT ?
                ''', (user_id, thought_type, limit))
            else:
                cursor.execute('''
                SELECT * FROM thoughts
                WHERE user_id = ?
                ORDER BY priority_score DESC, timestamp DESC
                LIMIT ?
                ''', (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === Цели ===
    def add_goal(self, user_id: int, description: str, **kwargs) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO goals
            (user_id, created_at, description, priority, status, progress, next_action, subgoals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                time.time(),
                description,
                kwargs.get('priority', 0.5),
                kwargs.get('status', 'active'),
                kwargs.get('progress', 0.0),
                kwargs.get('next_action', ''),
                json.dumps(kwargs.get('subgoals', []), ensure_ascii=False)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_active_goals(self, user_id: int, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT * FROM goals
            WHERE user_id = ? AND status = 'active'
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            ''', (user_id, limit))
            goals = [dict(row) for row in cursor.fetchall()]

            # Десериализация подцелей
            for goal in goals:
                try:
                    goal['subgoals'] = json.loads(goal['subgoals'])
                except:
                    goal['subgoals'] = []

            return goals

    def update_goal_progress(self, user_id: int, goal_id: int, progress: float):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE goals
            SET progress = ?
            WHERE id = ? AND user_id = ?
            ''', (progress, goal_id, user_id))
            conn.commit()

    # === Паттерны ===
    def add_pattern(self, user_id: int, pattern_type: str, description: str, confidence: float = 0.5):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id FROM patterns
            WHERE user_id = ? AND pattern_type = ? AND description = ?
            ''', (user_id, pattern_type, description))
            existing = cursor.fetchone()

            if existing:
                cursor.execute('''
                UPDATE patterns
                SET occurrences = occurrences + 1, last_seen = ?, confidence = ?
                WHERE id = ?
                ''', (time.time(), min(1.0, confidence * 1.1), existing[0]))
            else:
                cursor.execute('''
                INSERT INTO patterns
                (user_id, pattern_type, description, occurrences, confidence, created_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, pattern_type, description, 1, confidence, time.time(), time.time()))
            conn.commit()

    def get_patterns(self, user_id: int, min_confidence: float = 0.6, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT * FROM patterns
            WHERE user_id = ? AND confidence >= ?
            ORDER BY occurrences DESC, confidence DESC
            LIMIT ?
            ''', (user_id, min_confidence, limit))
            return [dict(row) for row in cursor.fetchall()]

    # === История поиска ===
    def add_search_record(self, user_id: int, query: str, results_count: int):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO search_history (user_id, query, timestamp, results_count)
            VALUES (?, ?, ?, ?)
            ''', (user_id, query, time.time(), results_count))
            conn.commit()

    def get_search_stats(self, user_id: int) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*), SUM(results_count) FROM search_history WHERE user_id = ?', (user_id,))
            count, total_results = cursor.fetchone()
            return {
                'total_searches': count or 0,
                'total_results': total_results or 0,
                'avg_results_per_search': (total_results / count if count > 0 else 0)
            }

    # === Статистика ===
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (user_id,))
            interactions = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM facts WHERE user_id = ? AND decay_factor > 0.3', (user_id,))
            facts = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM thoughts WHERE user_id = ?', (user_id,))
            thoughts = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM patterns WHERE user_id = ? AND confidence > 0.5', (user_id,))
            patterns = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM goals WHERE user_id = ? AND status = \'active\'', (user_id,))
            goals = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM search_history WHERE user_id = ?', (user_id,))
            searches = cursor.fetchone()[0]

            return {
                'interactions': interactions,
                'facts': facts,
                'thoughts': thoughts,
                'patterns': patterns,
                'goals': goals,
                'searches': searches,
                'first_interaction': self._get_first_interaction_time(user_id)
            }

    def _get_first_interaction_time(self, user_id: int) -> float:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT timestamp FROM interactions
            WHERE user_id = ? ORDER BY timestamp ASC LIMIT 1
            ''', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else time.time()


# ================= СИСТЕМА МЫШЛЕНИЯ ДЛЯ ЛОКАЛЬНЫХ МОДЕЛЕЙ =================
class LocalThinkingSystem:
    """Оптимизированная система мышления для локальных LLM с многоуровневым анализом"""

    def __init__(self, config: Dict[str, Any]):
        self.api_url = config['url']
        self.api_key = config.get('api_key', "")
        self.model = config['model']
        self.rate_limit = 0.3
        self.last_request_time = 0
        self.cache = ResponseCache()
        self.reasoning_history = deque(maxlen=50)

        self.SYSTEM_PROMPT_PREFIX = (
            "Ты — когнитивный ассистент. Отвечай кратко, по делу, без лишних фраз. "
            "Фокусируйся на сути запроса. Не используй маркдаун. "
        )
        print(f"✅ Инициализирована локальная система мышления")
        print(f"   Модель: {self.model}")
        print(f"   Сервер: {self.api_url}")

    async def _wait_for_rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def generate_thought(self, thought_type: str, context: str) -> Optional[str]:
        """Генерация мысли определенного типа"""
        thought_prompts = {
            'рефлексия': (
                "Ты анализируешь последние взаимодействия. Какие выводы можно сделать? "
                "Что было эффективно, а что можно улучшить?"
            ),
            'планирование': (
                "Ты планируешь следующие действия. Что нужно сделать для достижения целей? "
                "Какие шаги будут наиболее эффективными?"
            ),
            'анализ': (
                "Ты анализируешь текущую ситуацию. Какие факторы важны? "
                "Что нужно учесть при принятии решений?"
            ),
            'обучение': (
                "Ты извлекаешь уроки из опыта. Что нового ты узнал? "
                "Как это можно применить в будущем?"
            ),
            'наблюдение': (
                "Ты замечаешь паттерны и закономерности. Что повторяется? "
                "Какие связи можно увидеть между разными событиями?"
            )
        }

        if thought_type not in thought_prompts:
            return None

        system_prompt = f"Ты — когнитивная система. {thought_prompts[thought_type]}"
        user_prompt = f"Контекст:\n{context}\n\nМои мысли:"
        response = await self.call_llm(system_prompt, user_prompt, temperature=0.7)
        return response if response and len(response) > 15 else None

    async def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Вызов локальной LLM через LM Studio API с кэшированием"""
        # Проверка кэша
        cached = self.cache.get(system_prompt, user_prompt)
        if cached:
            getattr(self.cache, 'hits', 0)
            self.cache.hits = getattr(self.cache, 'hits', 0) + 1
            return cached

        getattr(self.cache, 'misses', 0)
        self.cache.misses = getattr(self.cache, 'misses', 0) + 1

        await self._wait_for_rate_limit()

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": min(2048, Config.MAX_TOKENS),
                "top_p": 0.9,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()

                        # Очистка от лишних символов
                        content = re.sub(r'^[\s\*\-\_]+|[\s\*\-\_]+$', '', content)
                        content = re.sub(r'\n{3,}', '\n\n', content)

                        # Сохраняем в кэш
                        self.cache.set(system_prompt, user_prompt, content)
                        return content
                    else:
                        error_text = await response.text()
                        return f"⚠️ Ошибка LM Studio ({response.status}): {error_text[:150]}"
        except asyncio.TimeoutError:
            return "⚠️ Таймаут запроса. Модель может быть перегружена."
        except aiohttp.ClientConnectorError:
            return "⚠️ Не удаётся подключиться к LM Studio. Запущен ли сервер на http://localhost:1234?"
        except Exception as e:
            return f"⚠️ Ошибка локальной модели: {str(e)[:120]}"


# ================= ПОЛЬЗОВАТЕЛЬСКИЙ КОГНИТИВНЫЙ АГЕНТ =================
class UserCognitiveAgent:
    """Изолированный когнитивный агент с полным набором когнитивных способностей"""

    def __init__(self, user_id: int, db: EnhancedMemoryDB, thinker: LocalThinkingSystem,
                 search_engine: Optional[WebSearchEngine],
                 thought_prioritizer: ThoughtPrioritizer,
                 meta_monitor: MetaCognitiveMonitor,
                 goal_decomposer: GoalDecomposer,
                 episodic_memory: EpisodicMemory):
        self.user_id = user_id
        self.db = db
        self.thinker = thinker
        self.search_engine = search_engine
        self.thought_prioritizer = thought_prioritizer
        self.meta_monitor = meta_monitor
        self.goal_decomposer = goal_decomposer
        self.episodic_memory = episodic_memory

        self.interaction_count = 0
        self.thoughts_generated = 0
        self.web_searches_performed = 0
        self.start_time = time.time()
        self.context_window = deque(maxlen=Config.CONTEXT_WINDOW_SIZE)
        self._init_user_goals()

    def _init_user_goals(self):
        """Инициализация базовых целей для пользователя"""
        existing_goals = self.db.get_active_goals(self.user_id, limit=1)
        if not existing_goals:
            self.db.add_goal(
                self.user_id,
                "Помогать пользователю решать задачи",
                priority=0.9,
                next_action="Анализировать запросы и предоставлять полезные ответы"
            )
            self.db.add_goal(
                self.user_id,
                "Запоминать важную информацию",
                priority=0.8,
                next_action="Извлекать и сохранять ключевые факты из диалогов"
            )
            self.db.add_goal(
                self.user_id,
                "Учиться и адаптироваться",
                priority=0.7,
                next_action="Обнаруживать паттерны в поведении пользователя"
            )

    @staticmethod
    def normalize_search_query(query: str) -> str:
        """Исправление распространённых опечаток в поисковых запросах"""
        replacements = [
            (r'\bдолар\b', 'доллар'),
            (r'\bдолларов\b', 'долларов'),
            (r'\bодолар(е|у|ом)?\b', 'о доллар'),
            (r'\bкурс долар\b', 'курс доллара'),
            (r'\bпогод[ауы]\b', 'погода'),
            (r'\bновост[иь]\b', 'новости'),
        ]

        normalized = query
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # Удаляем двойные пробелы
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized if normalized else query

    def _is_time_query(self, text: str) -> bool:
        """Определяет запросы о текущем времени/дате/годе"""
        text_lower = text.lower()
        time_keywords = [
            r'какой сейчас год', r'текущий год', r'сегодняшний год',
            r'какой год сейчас', r'год сейчас', r'какой сегодня год',
            r'какое сегодня число', r'какой сегодня день', r'текущая дата',
            r'сегодняшняя дата', r'какой месяц сейчас', r'который час',
            r'сколько времени', r'какое время', r'сколько сейчас времени'
        ]
        return any(re.search(pattern, text_lower) for pattern in time_keywords)

    def _handle_time_query(self, text: str) -> str:
        """Отвечает на временные запросы через системное время (без поиска!)"""
        now = time.localtime()
        year = now.tm_year
        month = now.tm_mon
        day = now.tm_mday
        hour = now.tm_hour
        minute = now.tm_min

        # Названия месяцев на русском
        months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                  'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']

        if 'год' in text.lower():
            return f"📅 Текущий год: **{year}** (по данным системного времени вашего компьютера)"

        if 'число' in text.lower() or 'день' in text.lower() or 'дата' in text.lower():
            return f"📅 Сегодня: **{day} {months[month - 1]} {year} года**"

        if 'время' in text.lower() or 'час' in text.lower() or 'сколько времени' in text.lower():
            return f"⏰ Текущее время: **{hour:02d}:{minute:02d}** (по данным вашего компьютера)"

        return f"📅 Сейчас: **{day} {months[month - 1]} {year} года**, время **{hour:02d}:{minute:02d}**"

    async def process_message(self, user_input: str) -> str:
        start_time = time.time()
        self.interaction_count += 1
        self.context_window.append({'type': 'user', 'content': user_input, 'timestamp': time.time()})

        # Обработка команд
        command_response = self._handle_command(user_input)
        if command_response:
            return command_response

        # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: обработка временных запросов БЕЗ ПОИСКА
        if self._is_time_query(user_input):
            response = self._handle_time_query(user_input)
            # Сохраняем взаимодействие без поиска
            self.db.add_interaction(
                user_id=self.user_id,
                user_input=user_input,
                system_response=response,
                context=self._get_context_summary(),
                category='время',
                importance=0.9,
                complexity=0.3,
                used_web_search=0  # ← ВАЖНО: 0 = без поиска
            )
            self.context_window.append({'type': 'assistant', 'content': response, 'timestamp': time.time()})
            return response

        # Анализ запроса
        features = extract_semantic_features(user_input)
        complexity = features['complexity']
        importance = self._calculate_importance(user_input, features)

        # Проверка необходимости веб-поиска ДО генерации ответа
        needs_search = await self._needs_web_search(user_input, features)
        used_web_search = False

        if needs_search and Config.WEB_SEARCH_ENABLED and self.search_engine:
            search_response = await self._perform_web_search(user_input)
            if search_response and "⚠️" not in search_response[:10]:
                used_web_search = True
                self.web_searches_performed += 1
                self.db.add_search_record(self.user_id, user_input, 3)

                # Сохраняем как эпизод
                self.episodic_memory.add_episode(
                    self.user_id,
                    f"Выполнен веб-поиск по запросу: {user_input[:50]}",
                    importance=0.8,
                    context={'summary': user_input, 'emotion': 'neutral'}
                )

                # Сохраняем взаимодействие с пометкой поиска
                self.db.add_interaction(
                    user_id=self.user_id,
                    user_input=user_input,
                    system_response=search_response,
                    context=self._get_context_summary(),
                    category='веб_поиск',
                    importance=importance,
                    complexity=complexity,
                    used_web_search=1
                )

                self.context_window.append({'type': 'assistant', 'content': search_response, 'timestamp': time.time()})
                duration = time.time() - start_time
                logging.info(f"Пользователь {self.user_id}: поиск занял {duration:.2f}с")
                return f"🔍 Результаты поиска:\n\n{search_response}"

        # Извлечение и сохранение информации
        await self._extract_and_store_information(user_input, importance)

        # Генерация ответа с учётом контекста
        response = await self._generate_contextual_response(user_input, features, complexity, importance)

        # Метакогнитивная оценка качества ответа
        context_summary = self._get_context_summary()
        quality_metrics = await self.meta_monitor.evaluate_response_quality(user_input, response, context_summary)

        # Сохранение взаимодействия
        interaction_id = self.db.add_interaction(
            user_id=self.user_id,
            user_input=user_input,
            system_response=response,
            context=context_summary,
            category=self._categorize_input(user_input, features),
            importance=importance,
            complexity=complexity,
            used_web_search=1 if used_web_search else 0
        )

        # Сохранение как эпизода, если важность высокая или качество ответа низкое
        if importance > 0.85 or quality_metrics['overall'] < 0.6:
            self.episodic_memory.add_episode(
                self.user_id,
                f"Запрос: {user_input[:100]} | Ответ: {response[:100]}",
                importance=max(importance, 0.8),
                context={
                    'summary': user_input,
                    'emotion': features['sentiment'],
                    'interaction_id': interaction_id
                }
            )

        self.context_window.append({'type': 'assistant', 'content': response, 'timestamp': time.time()})

        # Периодическое автономное мышление
        if self.interaction_count % Config.REFLECTION_INTERVAL == 0:
            await self._autonomous_thinking()

        duration = time.time() - start_time
        if duration > 2.0:
            logging.info(f"Пользователь {self.user_id}: обработка заняла {duration:.2f}с (поиск: {used_web_search})")

        return response

    def _calculate_importance(self, text: str, features: Dict) -> float:
        importance = 0.5
        if any(word in text.lower() for word in ['важно', 'срочно', 'критично', 'обязательно', 'немедленно']):
            importance += 0.25
        importance += min(0.2, features['question_words'] * 0.1)
        importance += min(0.15, features['imperatives'] * 0.08)
        importance += features['complexity'] * 0.2
        if features['length'] > 15:
            importance += 0.08
        return min(1.0, importance)

    async def _needs_web_search(self, text: str, features: Dict) -> bool:
        """Определяет, нужен ли веб-поиск на основе нескольких критериев"""
        if not Config.WEB_SEARCH_ENABLED:
            return False

        # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: не искать для временных запросов!
        if self._is_time_query(text):
            return False

        normalized = self.normalize_search_query(text)
        text_lower = normalized.lower()

        # Явные триггеры поиска
        if any(trigger in text_lower for trigger in Config.SEARCH_TRIGGERS):
            return True

        # Вопросы о текущих событиях (кроме времени/даты)
        current_keywords = ['последние', 'актуальн', 'новост', 'текущ', 'свежие']
        if any(kw in text_lower for kw in current_keywords) and features['has_question']:
            return True

        # Вопросы, требующие актуальных данных
        time_sensitive_patterns = [
            r'какой курс', r'какая погода', r'кто победил', r'что случилось',
            r'новый закон', r'последние события', r'текущая ситуация'
        ]
        if any(re.search(pattern, text_lower) for pattern in time_sensitive_patterns):
            return True

        # Эвристика: короткие вопросы часто требуют поиска
        if features['has_question'] and features['length'] < 8:
            return random.random() < 0.4  # 40% вероятность для коротких вопросов

        return False

    async def _perform_web_search(self, query: str) -> str:
        """Выполнение поиска и формирование ответа"""
        if not self.search_engine:
            return "⚠️ Веб-поиск недоступен. Установите модули ddgs и beautifulsoup4."

        results = await self.search_engine.search(query, max_results=4)

        if not results:
            return "⚠️ Не удалось найти информацию в интернете."

        # Сохраняем результаты как факты
        for i, result in enumerate(results[:3]):
            self.db.add_fact(
                self.user_id,
                f"поиск_{int(time.time())}_{i}",
                f"{result['title']}: {result['snippet']}",
                category='веб_поиск',
                importance=0.7,
                confidence=0.8
            )

        # Формируем контекст для LLM
        search_context = "\n\n".join([
            f"Источник {i + 1} ({result['url'][:30]}):\n{result['title']}\n{result['snippet']}"
            for i, result in enumerate(results[:3])
        ])

        # Генерируем синтез результатов
        synthesis_prompt = (
            f"Запрос пользователя: {query}\n\n"
            f"Результаты поиска:\n{search_context}\n\n"
            "Синтезируй краткий, информативный ответ на основе найденных источников. "
            "Укажи ключевые факты. Не упоминай источники напрямую, но будь точен. "
            "Если информация противоречива — укажи это."
        )

        response = await self.thinker.call_llm(
            "Ты — аналитик информации. Синтезируй ответ из источников кратко и точно.",
            synthesis_prompt,
            temperature=0.3
        )

        return response

    async def _extract_and_store_information(self, text: str, importance: float):
        """Извлечение информации из текста"""
        # Извлечение сущностей
        entities = extract_entities(text)

        # Сохраняем числа как факты
        for number in entities.get('numbers', [])[:3]:
            self.db.add_fact(self.user_id, 'число', number, category='информация', importance=importance * 0.4)

        # Сохраняем имена
        for name in entities.get('names', [])[:2]:
            if len(name) > 2:
                self.db.add_fact(self.user_id, 'имя', name, category='персона', importance=importance * 0.6)

        # Извлекаем факты из утверждений
        patterns = [
            (r'(\w+)\s+(?:это|является|называется)\s+([^.,!?]+)', 'определение'),
            (r'запомни[,:]?\s*(.+)', 'важная_информация'),
            (r'я люблю (\w+)', 'предпочтение'),
            (r'(\w+)\s+=\s+([^.,]+)', 'равенство'),
            (r'мой любимый (\w+)\s+(.+)', 'предпочтение')
        ]

        for pattern, category in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
            for match in matches[:1]:
                if isinstance(match, tuple) and len(match) >= 2:
                    key, value = match[0], match[1]
                    if len(key) > 2 and len(value) > 1:
                        self.db.add_fact(
                            self.user_id,
                            key.strip().lower(),
                            value.strip(),
                            category=category,
                            importance=min(1.0, importance * 1.1)
                        )

    async def _generate_contextual_response(self, user_input: str, features: Dict, complexity: float,
                                            importance: float) -> str:
        # Получение контекста
        relevant_interactions = self.db.get_contextual_interactions(self.user_id, user_input, limit=3)
        relevant_facts = self.db.get_relevant_facts(self.user_id, user_input, limit=4)
        active_goals = self.db.get_active_goals(self.user_id, limit=2)

        # Поиск похожих эпизодов
        similar_episodes = self.episodic_memory.recall_similar_episodes(self.user_id, user_input, limit=2)

        # Формирование контекста
        context_parts = []
        if relevant_interactions:
            context_parts.append("Недавний диалог:")
            for interaction in relevant_interactions[:2]:
                context_parts.append(f"Вы: {interaction['user_input'][:60]}")
                context_parts.append(f"Я: {interaction['system_response'][:60]}")

        if relevant_facts:
            context_parts.append("\nРелевантные факты:")
            for fact in relevant_facts[:3]:
                context_parts.append(f"- {fact['key']}: {fact['value']}")

        if similar_episodes:
            context_parts.append("\nПохожие ситуации из прошлого:")
            for ep in similar_episodes:
                context_parts.append(f"- {ep['description'][:70]}")

        if active_goals:
            context_parts.append("\nТекущие цели:")
            for goal in active_goals:
                progress_bar = "█" * int(goal['progress'] * 10) + "░" * (10 - int(goal['progress'] * 10))
                context_parts.append(f"- {goal['description'][:50]} [{progress_bar}]")

        context = "\n".join(context_parts) if context_parts else "Нет контекста"

        # Определение необходимости глубокого анализа
        needs_deep_thinking = (
                complexity > Config.DEEP_THINKING_THRESHOLD or
                importance > 0.8 or
                features['question_words'] > 2 or
                any(word in user_input.lower() for word in
                    ['почему', 'как сделать', 'анализ', 'проанализируй', 'сравни'])
        )

        if needs_deep_thinking and len(self.context_window) > 2:
            # Многоуровневое мышление с приоритизацией
            deep_thoughts = []
            thought_history = [t['content'] for t in self.db.get_recent_thoughts(self.user_id, limit=5)]

            for thought_type in ['анализ', 'рефлексия', 'планирование']:
                thought = await self.thinker.generate_thought(thought_type,
                                                              f"Запрос: {user_input}\nКонтекст: {context}")
                if thought:
                    priority = self.thought_prioritizer.calculate_priority(thought, context, thought_history)
                    deep_thoughts.append((priority, thought_type, thought))
                    thought_history.append(thought)

            # Сортируем по приоритету и сохраняем топ-3
            deep_thoughts.sort(reverse=True, key=lambda x: x[0])
            top_thoughts = deep_thoughts[:3]

            for priority, thought_type, content in top_thoughts:
                self.db.add_thought(
                    self.user_id,
                    thought_type,
                    content[:200],
                    trigger='глубокое_мышление',
                    importance=0.7,
                    depth_level=2,
                    priority_score=priority
                )

            # Синтез ответа
            synthesis_prompt = (
                f"Запрос: {user_input}\n"
                f"Анализ: {' | '.join([t[1] for t in top_thoughts]) if top_thoughts else 'Стандартный'}\n"
                f"Контекст: {context}\n"
                "Дай исчерпывающий, но лаконичный ответ. Включи ключевые инсайты из анализа."
            )

            system_prompt = self.thinker.SYSTEM_PROMPT_PREFIX + (
                "Синтезируй ответ на основе многоуровневого анализа. Будь полезным, точным и креативным при необходимости."
            )
            response = await self.thinker.call_llm(system_prompt, synthesis_prompt, temperature=0.65)
            self.thoughts_generated += len(top_thoughts)
        else:
            # Стандартный ответ
            system_prompt = self.thinker.SYSTEM_PROMPT_PREFIX + (
                f"КОНТЕКСТ:\n{context}\n\n"
                "Отвечай кратко, дружелюбно и по делу. Используй факты из контекста."
            )
            response = await self.thinker.call_llm(system_prompt, user_input, temperature=0.5)

        return response.strip()

    async def _autonomous_thinking(self):
        """Автономный процесс мышления с приоритизацией"""
        recent_interactions = self.db.get_contextual_interactions(self.user_id, "запросы", limit=5)
        if len(recent_interactions) < 2:
            return

        # Готовим контекст
        context_lines = []
        for i, interaction in enumerate(recent_interactions[-3:], 1):
            context_lines.append(f"{i}. {interaction['user_input'][:50]}... → {interaction['system_response'][:50]}...")
        context = "\n".join(context_lines)

        # Случайный выбор типа мышления с приоритизацией
        thought_type = random.choice(Config.THOUGHT_TYPES)
        thought_content = await self.thinker.generate_thought(thought_type, context)

        if thought_content and len(thought_content) > 20:
            # Рассчитываем приоритет мысли
            priority = self.thought_prioritizer.calculate_priority(
                thought_content,
                context,
                [t['content'] for t in self.db.get_recent_thoughts(self.user_id, limit=5)]
            )

            if priority > 0.6:  # Сохраняем только высоко-приоритетные мысли
                self.db.add_thought(
                    self.user_id,
                    thought_type,
                    thought_content[:300],
                    trigger="автономное_мышление",
                    importance=0.6,
                    depth_level=1,
                    priority_score=priority
                )
                self.thoughts_generated += 1
                logging.info(f"💡 Сохранена мысль [{thought_type}] с приоритетом {priority:.2f}: {thought_content[:50]}")

    def _handle_command(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()

        if text_lower in ['думай', 'подумай', 'мысли', '/think']:
            asyncio.create_task(self._autonomous_thinking())
            return "🧠 Запускаю процесс мышления... (результаты в /insights)"

        elif text_lower in ['/clear', 'очистить контекст', 'забудь']:
            self.context_window.clear()
            return "🧹 Контекст очищен. Долгосрочная память сохранена."

        elif text_lower.startswith('/search ') or text_lower.startswith('найди '):
            query = text_lower.replace('/search ', '').replace('найди ', '').strip()
            if query:
                asyncio.create_task(self._perform_web_search_and_respond(query))
                return f"🔍 Ищу информацию по запросу: {query}"

        return None

    async def _perform_web_search_and_respond(self, query: str):
        """Асинхронный поиск для команды /search"""
        response = await self._perform_web_search(query)
        # Результат будет отправлен в основном цикле обработки

    def _get_context_summary(self) -> str:
        if not self.context_window:
            return ""
        summary = []
        for item in list(self.context_window)[-4:]:
            prefix = "П:" if item['type'] == 'user' else "Я:"
            summary.append(f"{prefix}{item['content'][:30]}")
        return " | ".join(summary)

    def _categorize_input(self, text: str, features: Dict) -> str:
        text_lower = text.lower()
        categories = {
            'вопрос': ['что', 'как', 'почему', 'зачем', 'когда', 'где', 'кто', 'сколько', '?'],
            'память': ['запомни', 'сохрани', 'напомни', 'запиши'],
            'анализ': ['анализ', 'разбери', 'оцени', 'сравни', 'проанализируй'],
            'творчество': ['придумай', 'создай', 'идея', 'креатив', 'напиши'],
            'план': ['план', 'расписание', 'как достичь', 'шаги', 'алгоритм'],
            'поиск': ['найди', 'поищи', 'интернет', 'веб']
        }
        scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in categories.items()}
        return max(scores, key=scores.get) if any(scores.values()) else 'диалог'

    def get_comprehensive_stats(self) -> str:
        stats = self.db.get_user_stats(self.user_id)
        search_stats = self.db.get_search_stats(self.user_id)
        uptime = time.time() - stats['first_interaction']
        days = int(uptime // 86400)
        hours = int((uptime % 86400) // 3600)
        cache_stats = self.thinker.cache.get_stats()

        return (
            f"📊 ПЕРСОНАЛЬНАЯ СТАТИСТИКА\n{'=' * 40}\n"
            f"⏱️ Время знакомства: {days}д {hours}ч\n"
            f"💬 Сообщений: {stats['interactions']}\n"
            f"🧠 Мыслей сгенерировано: {self.thoughts_generated}\n"
            f"🌐 Веб-поисков: {stats['searches']} (всего результатов: {int(search_stats['total_results'])})\n"
            f"📚 Фактов: {stats['facts']}\n"
            f"💡 Сохранённых мыслей: {stats['thoughts']}\n"
            f"🔗 Паттернов: {stats['patterns']}\n"
            f"🎯 Активных целей: {stats['goals']}\n"
            f"💾 Кэш ответов: {cache_stats['size']} / {cache_stats['max_size']} ({cache_stats['usage_percent']:.1f}%)\n"
            f"{self.meta_monitor.get_summary()}"
        )

    def get_patterns_summary(self) -> str:
        patterns = self.db.get_patterns(self.user_id, min_confidence=0.55, limit=8)
        if not patterns:
            return "🔍 Паттерны не обнаружены. Продолжайте диалог!"
        lines = ["🔍 ОБНАРУЖЕННЫЕ ПАТТЕРНЫ", "=" * 40]
        for p in patterns[:5]:
            conf = int(p['confidence'] * 100)
            lines.append(f"• {p['description'][:60]} ({conf}%)")
        return "\n".join(lines)

    def get_insights_summary(self) -> str:
        thoughts = self.db.get_recent_thoughts(self.user_id, limit=8)
        if not thoughts:
            return "💡 Инсайты отсутствуют. Используйте /think для анализа."
        lines = ["💡 ПОСЛЕДНИЕ ИНСАЙТЫ", "=" * 40]
        for thought in thoughts[:6]:
            priority = thought['priority_score']
            lines.append(f"• [{thought['thought_type']}] (приоритет: {priority:.2f}) {thought['content'][:70]}...")
        return "\n".join(lines)

    def get_goals_summary(self) -> str:
        goals = self.db.get_active_goals(self.user_id, limit=5)
        if not goals:
            return "🎯 Цели не определены."
        lines = ["🎯 ВАША ИЕРАРХИЯ ЦЕЛЕЙ", "=" * 40]
        for goal in goals[:3]:
            prog = int(goal['progress'] * 100)
            progress_bar = "█" * (prog // 10) + "░" * (10 - prog // 10)
            lines.append(f"• {goal['description'][:50]}")
            lines.append(f"  Приоритет: {goal['priority']:.1f} | Прогресс: [{progress_bar}] {prog}%")
            if goal.get('next_action'):
                lines.append(f"  След. шаг: {goal['next_action'][:40]}")
            if goal.get('subgoals'):
                subgoals_count = len(goal['subgoals'])
                if subgoals_count > 0:
                    lines.append(f"  Подзадачи: {subgoals_count} шт.")
            lines.append("")
        return "\n".join(lines)

    async def decompose_current_goal(self, goal_description: str) -> str:
        """Декомпозиция цели по запросу пользователя"""
        subgoals = await self.goal_decomposer.decompose_goal(goal_description, self._get_context_summary())

        if not subgoals:
            return "⚠️ Не удалось разбить цель на подзадачи. Попробуйте уточнить формулировку."

        lines = [f"🎯 ДЕКОМПОЗИЦИЯ ЦЕЛИ: {goal_description[:60]}", "=" * 40]
        for sg in subgoals:
            lines.append(f"\n{sg['id']}. {sg['description']}")
            lines.append(f"   Срок: {sg['timeframe']} | Зависимости: {sg['dependencies']}")

        # Сохраняем цель с подзадачами
        self.db.add_goal(
            self.user_id,
            goal_description,
            priority=0.8,
            subgoals=subgoals
        )

        return "\n".join(lines)


# ================= МЕНЕДЖЕР СЕССИЙ =================
class SessionManager:
    def __init__(self, db: EnhancedMemoryDB, thinker: LocalThinkingSystem, search_engine: Optional[WebSearchEngine]):
        self.db = db
        self.thinker = thinker
        self.search_engine = search_engine
        self.sessions: Dict[int, UserCognitiveAgent] = {}
        self.session_timeout = Config.SESSION_TIMEOUT
        self.last_cleanup = time.time()

    async def get_or_create_session(self, user_id: int) -> UserCognitiveAgent:
        if time.time() - self.last_cleanup > 300:
            await self._cleanup_inactive_sessions()
            self.last_cleanup = time.time()

        if user_id not in self.sessions:
            # Создаём вспомогательные компоненты для каждого пользователя
            thought_prioritizer = ThoughtPrioritizer()
            meta_monitor = MetaCognitiveMonitor()
            goal_decomposer = GoalDecomposer(self.thinker)
            episodic_memory = EpisodicMemory(self.db)

            self.sessions[user_id] = UserCognitiveAgent(
                user_id,
                self.db,
                self.thinker,
                self.search_engine,
                thought_prioritizer,
                meta_monitor,
                goal_decomposer,
                episodic_memory
            )
            logging.info(f"🆕 Новая сессия для пользователя {user_id}")

        return self.sessions[user_id]

    async def _cleanup_inactive_sessions(self):
        current_time = time.time()
        inactive = [uid for uid, sess in self.sessions.items()
                    if current_time - sess.start_time > self.session_timeout]
        for uid in inactive:
            del self.sessions[uid]
            logging.info(f"🧹 Очищена сессия {uid}")


# ================= ОБРАБОТЧИКИ TELEGRAM =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user.id)

    welcome = (
        f"👋 Привет, {user.first_name}!\n\n"
        "🧠 Я — когнитивный ассистент с локальной ИИ-моделью и веб-поиском.\n\n"
        "✅ Полная приватность — локальные запросы не уходят в интернет\n"
        "🌐 Актуальная информация — веб-поиск через DuckDuckGo (без API ключей)\n"
        "⚡ Быстрая работа без интернета для обычных запросов\n"
        "⏰ Точное время/дата/год — беру напрямую из системных часов вашего ПК (100% актуально!)\n\n"
        "✨ **Мои когнитивные способности:**\n"
        "• 🤯 Многоуровневый анализ с приоритизацией мыслей\n"
        "• 🧠 Контекстная и эпизодическая память\n"
        "• 🔍 Автоматический и ручной веб-поиск актуальной информации (курсы, погода, новости)\n"
        "• ⏰ Системное время для запросов о дате/годе/времени (без поиска!)\n"
        "• 💡 Метакогнитивный мониторинг качества ответов\n"
        "• 🎯 Декомпозиция сложных целей на подзадачи\n"
        "• 🔗 Автоматическое обнаружение паттернов поведения\n\n"
        "📌 **Команды:**\n"
        "• /think — глубокий анализ и рефлексия\n"
        "• /search [запрос] — ручной веб-поиск\n"
        "• /stats — персональная статистика и метакогнитивный отчёт\n"
        "• /patterns — обнаруженные паттерны поведения\n"
        "• /insights — инсайты из моих размышлений с приоритетами\n"
        "• /goals — ваши цели, прогресс и подзадачи\n"
        "• /decompose [цель] — разбить цель на шаги\n"
        "• /clear — очистить контекст диалога"
    )

    await update.effective_message.reply_text(welcome, reply_markup=create_main_keyboard())


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.effective_message.reply_chat_action("typing")
    stats_text = agent.get_comprehensive_stats()
    # Экранируем для MarkdownV2
    stats_text = stats_text.replace('.', '\\.').replace('-', '\\-').replace('(', '\\(').replace(')', '\\)')
    await update.effective_message.reply_text(f"```\n{stats_text}\n```", parse_mode='MarkdownV2',
                                              reply_markup=create_main_keyboard())


async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.effective_message.reply_text(
        "🧠 Запускаю глубокое многоуровневое мышление...\n"
        "Анализирую диалоги и генерирую инсайты с приоритизацией...")
    await agent._autonomous_thinking()
    await update.effective_message.reply_text("✅ Глубокое мышление завершено! Результаты в /insights",
                                              reply_markup=create_main_keyboard())


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.effective_message.reply_text(
            "🔍 Использование: /search [запрос]\nПример: /search последние новости ИИ",
            reply_markup=create_main_keyboard()
        )
        return

    query = " ".join(context.args)
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.effective_message.reply_chat_action("typing")
    await update.effective_message.reply_text(f"🌐 Ищу информацию по запросу: {query}...")

    result = await agent._perform_web_search(query)
    parts = split_message(f"🔍 Результаты поиска:\n\n{result}")

    for i, part in enumerate(parts):
        reply_markup = create_main_keyboard() if i == len(parts) - 1 else None
        await update.effective_message.reply_text(part, reply_markup=reply_markup, disable_web_page_preview=True)
        if i < len(parts) - 1:
            await asyncio.sleep(0.3)


async def decompose_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.effective_message.reply_text(
            "🎯 Использование: /decompose [цель]\nПример: /decompose выучить английский за 3 месяца",
            reply_markup=create_main_keyboard()
        )
        return

    goal = " ".join(context.args)
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.effective_message.reply_chat_action("typing")
    await update.effective_message.reply_text(f"🎯 Анализирую цель: {goal}...")

    result = await agent.decompose_current_goal(goal)
    await update.effective_message.reply_text(result, reply_markup=create_main_keyboard())


async def patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.effective_message.reply_chat_action("typing")
    await update.effective_message.reply_text(agent.get_patterns_summary(), reply_markup=create_main_keyboard())


async def insights_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.effective_message.reply_chat_action("typing")
    await update.effective_message.reply_text(agent.get_insights_summary(), reply_markup=create_main_keyboard())


async def goals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.effective_message.reply_chat_action("typing")
    await update.effective_message.reply_text(agent.get_goals_summary(), reply_markup=create_main_keyboard())


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    agent.context_window.clear()
    await update.effective_message.reply_text("🧹 Контекст диалога очищен", reply_markup=create_main_keyboard())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_message.text:
        return

    # 🔑 КРИТИЧЕСКИ ВАЖНАЯ ПРОВЕРКА: гарантируем, что система инициализирована
    if 'session_manager' not in context.application.bot_data:
        await update.effective_message.reply_chat_action("typing")
        await asyncio.sleep(2)
        if 'session_manager' not in context.application.bot_data:
            await update.effective_message.reply_text(
                "⚠️ Система ещё инициализируется. Подождите 5 секунд и повторите запрос.",
                reply_markup=create_main_keyboard()
            )
            return

    user_id = update.effective_user.id
    text = update.effective_message.text.strip()
    if not text:
        return

    logging.info(f"📨 {user_id}: {text[:40]}...")

    try:
        session_manager = context.application.bot_data['session_manager']
        agent = await session_manager.get_or_create_session(user_id)
        await update.effective_message.reply_chat_action("typing")

        # Короткая пауза для лучшего UX
        if len(text) > 30 or '?' in text:
            await asyncio.sleep(0.3)

        response = await agent.process_message(text)
        parts = split_message(response)

        for i, part in enumerate(parts):
            reply_markup = create_main_keyboard() if (
                    i == len(parts) - 1 and agent.interaction_count % 4 == 0) else None
            await update.effective_message.reply_text(part, reply_markup=reply_markup, disable_web_page_preview=True)
            if i < len(parts) - 1:
                await asyncio.sleep(0.3)

    except Exception as e:
        logging.error(f"❌ Ошибка: {e}", exc_info=True)
        await update.effective_message.reply_text(
            "⚠️ Ошибка обработки. Попробуйте переформулировать запрос.",
            reply_markup=create_main_keyboard()
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    handlers = {
        "deep_think": think_command,
        "search": lambda u, c: query.message.reply_text(
            "🔍 Отправьте запрос в формате:\n/search [ваш запрос]",
            reply_markup=create_main_keyboard()
        ),
        "stats": stats_command,
        "goals": goals_command,
        "insights": insights_command,
        "patterns": patterns_command,
        "clear": clear_command,
    }

    handler = handlers.get(query.data)
    if handler:
        if callable(handler):
            await handler(update, context)
        else:
            await handler
    else:
        await query.message.reply_text("❓ Неизвестная команда", reply_markup=create_main_keyboard())


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Ошибка: {context.error}", exc_info=context.error)


# ================= ГЛАВНАЯ ФУНКЦИЯ С ГАРАНТИРОВАННОЙ ИНИЦИАЛИЗАЦИЕЙ =================
def setup_logging():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Config.LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


async def main():
    setup_logging()
    logging.info("🚀 Запуск улучшенного когнитивного ассистента с веб-поиском и метакогницией")

    # 🔑 ШАГ 1: СИНХРОННАЯ НАСТРОЙКА ДО СОЗДАНИЯ ПРИЛОЖЕНИЯ
    try:
        token = Config.get_telegram_token()
        lm_config = Config.get_lmstudio_config()
    except Exception as e:
        print(f"\n❌ Критическая ошибка инициализации: {e}")
        sys.exit(1)

    # 🔑 ШАГ 2: СОЗДАНИЕ КОМПОНЕНТОВ ДО ЗАПУСКА ПРИЛОЖЕНИЯ
    print("🔧 Инициализация компонентов...")
    db = EnhancedMemoryDB(Config.DB_PATH)
    thinker = LocalThinkingSystem(lm_config)
    search_engine = WebSearchEngine() if Config.WEB_SEARCH_ENABLED else None

    print("✅ Компоненты инициализированы")
    print(f"   Веб-поиск: {'✅ Активирован' if Config.WEB_SEARCH_ENABLED else '❌ Отключён (установите ddgs)'}")

    # 🔑 ШАГ 3: СОЗДАНИЕ ПРИЛОЖЕНИЯ И ПЕРЕДАЧА ГОТОВЫХ КОМПОНЕНТОВ
    application = (
        ApplicationBuilder()
        .token(token)
        .read_timeout(25)
        .write_timeout(25)
        .connect_timeout(10)
        .pool_timeout(10)
        .build()
    )

    # 🔑 ШАГ 4: СОХРАНЕНИЕ КОМПОНЕНТОВ В bot_data ДО ЗАПУСКА
    application.bot_data['session_manager'] = SessionManager(db, thinker, search_engine)
    application.bot_data['db'] = db
    application.bot_data['thinker'] = thinker
    application.bot_data['search_engine'] = search_engine

    # 🔑 ШАГ 5: РЕГИСТРАЦИЯ ОБРАБОТЧИКОВ
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("think", think_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("decompose", decompose_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("patterns", patterns_command))
    application.add_handler(CommandHandler("insights", insights_command))
    application.add_handler(CommandHandler("goals", goals_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    # 🔑 ШАГ 6: НАСТРОЙКА КОМАНД МЕНЮ
    await application.bot.set_my_commands([
        BotCommand("start", "Начать"),
        BotCommand("think", "Глубокий анализ"),
        BotCommand("search", "Веб-поиск"),
        BotCommand("decompose", "Разбить цель на шаги"),
        BotCommand("stats", "Статистика и метакогниция"),
        BotCommand("patterns", "Паттерны"),
        BotCommand("insights", "Инсайты"),
        BotCommand("goals", "Цели"),
        BotCommand("clear", "Очистить контекст")
    ])

    # 🔑 ШАГ 7: ЗАПУСК ПРИЛОЖЕНИЯ
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    # Вывод информации о запуске
    print("\n" + "=" * 70)
    print("✅ УЛУЧШЕННЫЙ КОГНИТИВНЫЙ АССИСТЕНТ ЗАПУЩЕН!")
    print("=" * 70)
    print(f"\n🤖 Модель: {lm_config['model']}")
    print("🔗 LM Studio: http://localhost:1234")
    print(f"🌐 Веб-поиск: {'активирован' if Config.WEB_SEARCH_ENABLED else 'отключён'}")
    print("\n📱 Напишите боту в Telegram /start")
    print("\n💡 КОГНИТИВНЫЕ УЛУЧШЕНИЯ:")
    print("   • Веб-поиск актуальной информации (без API ключей)")
    print("   • ⏰ Системное время для запросов о дате/годе/времени (100% актуально!)")
    print("   • Метакогнитивный мониторинг качества ответов")
    print("   • Приоритизация мыслей по интеллектуальным критериям")
    print("   • Декомпозиция сложных целей на подзадачи (/decompose)")
    print("   • Эпизодическая память для ключевых событий")
    print("   • Автоматическое определение необходимости поиска")
    print("\n🛑 Остановка: Ctrl+C")
    print("=" * 70 + "\n")

    logging.info("🔄 Бот работает. Нажмите Ctrl+C для остановки")
    while True:
        await asyncio.sleep(3600)


def run():
    print("AGI Cognitive Assistant v4.1 — Улучшенная когнитивная система с веб-поиском и актуальным временем")
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8+")
        sys.exit(1)

    # Проверка обязательных зависимостей
    required_packages = ['aiohttp', 'requests']
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing)}")
        print("📦 Установите: pip install " + " ".join(missing))
        sys.exit(1)

    if not HAS_WEB_SEARCH:
        print("⚠️  Веб-поиск недоступен. Для активации установите:")
        print("   pip install ddgs beautifulsoup4 lxml")

    print("\n🚀 Запуск улучшенного когнитивного ассистента...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Работа завершена")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()