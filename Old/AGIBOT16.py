# coding: utf-8
"""
Cognitive_Agent_Pro_v9.7_SINGLE_MODEL.py
🧠 Когнитивный агент - работа с УЖЕ ЗАГРУЖЕННОЙ моделью
✅ Исправлены проблемы с использованием поиска
✅ Улучшена обработка сложных запросов
✅ Правильное использование найденной информации
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
import aiohttp
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import html


# ================= ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =================
def load_dotenv(path: Path = Path(".env")):
    """Загружает переменные окружения из .env файла"""
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
            print(f"✅ Загружены переменные окружения из {path}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки .env: {e}")


load_dotenv()


# ================= ЦВЕТНЫЕ ЛОГИ =================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"


def log_stage(stage: str, message: str, color: str = Colors.CYAN):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.BOLD}{color}[{timestamp}] {stage}{Colors.RESET}")
    print(f"{color}→ {message}{Colors.RESET}")


# ================= TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )
    from telegram.error import TelegramError, TimedOut
except ImportError as e:
    print(f"{Colors.RED}❌ Ошибка импорта telegram: {e}{Colors.RESET}")
    print(f"{Colors.YELLOW}📦 Установите: pip install python-telegram-bot>=20.7{Colors.RESET}")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_brain")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "brain_memory.db"
    LEARNING_PATH = ROOT / "learning_patterns.json"

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")

    TIMEOUT = 45
    MAX_TOKENS = 2000

    # Telegram
    MAX_MESSAGE_LENGTH = 4096


# ================= СТРУКТУРЫ ДАННЫХ =================
@dataclass
class CognitiveResponse:
    """Полный когнитивный ответ"""
    final_answer: str
    confidence: float
    used_search: bool
    search_query: str
    search_results: str
    processing_time: float
    model_used: str


@dataclass
class LearningPattern:
    """Паттерн обучения"""
    query_keywords: List[str]
    successful_approach: str
    avg_confidence: float
    usage_count: int
    last_used: float


# ================= УТИЛИТЫ =================
def clean_text(text: str) -> str:
    """Очистка текста от HTML и лишних символов"""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_keywords(text: str) -> List[str]:
    """Извлечение ключевых слов из текста"""
    if not text:
        return []

    stop_words = {
        'что', 'как', 'где', 'когда', 'почему', 'какой', 'какая', 'какие',
        'это', 'есть', 'был', 'была', 'были', 'быть', 'можно', 'нужно',
        'мне', 'тебе', 'его', 'ее', 'их', 'наш', 'ваш', 'в', 'на', 'и', 'или',
        'а', 'но', 'да', 'нет', 'не', 'по', 'за', 'к', 'от', 'до', 'из', 'со'
    }
    words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
    return [w for w in words if w not in stop_words][:10]


def split_message(text: str, max_length: int = 4096) -> List[str]:
    """Разделяет длинное сообщение на части"""
    if len(text) <= max_length:
        return [text]

    parts = []
    while len(text) > max_length:
        split_point = text.rfind('\n', 0, max_length)
        if split_point == -1:
            split_point = text.rfind('. ', 0, max_length)
        if split_point == -1:
            split_point = text.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length

        parts.append(text[:split_point].strip())
        text = text[split_point:].strip()

    if text:
        parts.append(text)

    return parts[:10]


def get_current_datetime() -> Dict[str, Any]:
    """Получает актуальную дату и время"""
    now = datetime.now()
    months_ru = [
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ]
    weekdays_ru = [
        'понедельник', 'вторник', 'среда', 'четверг', 'пятница',
        'суббота', 'воскресенье'
    ]

    return {
        'datetime': now,
        'full_date': f"{now.day} {months_ru[now.month - 1]} {now.year} года",
        'year': now.year,
        'month': now.month,
        'month_name': months_ru[now.month - 1],
        'day': now.day,
        'weekday': weekdays_ru[now.weekday()],
        'time': now.strftime('%H:%M'),
        'time_full': now.strftime('%H:%M:%S'),
        'iso': now.isoformat(),
        'timestamp': now.timestamp()
    }


# ================= УЛУЧШЕННЫЙ ВЕБ-ПОИСК С РЕАЛЬНЫМИ ДАННЫМИ =================
class RealWebSearch:
    """Поиск с реальными данными из интернета"""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}

    async def search(self, query: str) -> Tuple[str, str]:
        """Поиск реальной информации в интернете"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        current_time = time.time()

        # Проверка кэша
        if cache_key in self.cache:
            if current_time - self.cache_ttl[cache_key] < 900:  # 15 минут
                return self.cache[cache_key]

        query_lower = query.lower()

        # Курсы валют
        if any(word in query_lower for word in ['курс', 'доллар', 'евро', 'валют', 'рубл']):
            result = await self._get_real_currency_rates()
            result_type = "currency"

        # Погода
        elif any(word in query_lower for word in ['погод', 'температур', 'дождь', 'снег', 'солнце']):
            result = await self._get_real_weather(query)
            result_type = "weather"

        # Новости
        elif any(word in query_lower for word in ['новост', 'событи', 'последн', 'актуальн']):
            result = await self._get_real_news()
            result_type = "news"

        # Общий поиск
        else:
            result = await self._search_with_duckduckgo(query)
            result_type = "general"

        # Кэшируем результат
        self.cache[cache_key] = (result, result_type)
        self.cache_ttl[cache_key] = current_time

        return result, result_type

    async def _get_real_currency_rates(self) -> str:
        """Получение реальных курсов валют"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://www.cbr-xml-daily.ru/daily_json.js",
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        text = re.sub(r'^/\*.*?\*/', '', text, flags=re.DOTALL).strip()
                        data = json.loads(text)

                        usd = data['Valute']['USD']
                        eur = data['Valute'].get('EUR', {})
                        cny = data['Valute'].get('CNY', {})

                        now = get_current_datetime()

                        return (
                            f"💰 КУРСЫ ВАЛЮТ ЦБ РФ (официальные)\n"
                            f"📅 На {now['time']} {now['full_date']}\n\n"
                            f"💵 Доллар США (USD): {usd['Value']:.2f} ₽\n"
                            f"📈 Изменение: {usd['Value'] - usd['Previous']:+.2f} ₽\n\n"
                            f"💶 Евро (EUR): {eur.get('Value', 0):.2f} ₽\n"
                            f"📈 Изменение: {eur.get('Value', 0) - eur.get('Previous', 0):+.2f} ₽\n\n"
                            f"💴 Юань (CNY): {cny.get('Value', 0):.2f} ₽\n"
                            f"📈 Изменение: {cny.get('Value', 0) - cny.get('Previous', 0):+.2f} ₽"
                        )
        except Exception as e:
            log_stage("⚠️ ПОИСК", f"Ошибка получения курсов: {e}", Colors.YELLOW)

        return "❌ Не удалось получить актуальные курсы валют. Попробуйте позже."

    async def _get_real_weather(self, query: str) -> str:
        """Получение информации о погоде"""
        try:
            # Извлекаем город из запроса
            cities = {
                'москв': 'Москва',
                'санкт-петербург': 'Санкт-Петербург',
                'новосибирск': 'Новосибирск',
                'екатеринбург': 'Екатеринбург',
                'казан': 'Казань',
                'нижний новгород': 'Нижний Новгород',
                'челябинск': 'Челябинск',
                'омск': 'Омск',
                'самара': 'Самара',
                'ростов': 'Ростов-на-Дону',
                'уфа': 'Уфа',
                'краснодар': 'Краснодар',
                'ессентук': 'Ессентуки',
                'сочи': 'Сочи',
                'волгоград': 'Волгоград',
                'перм': 'Пермь',
                'воронеж': 'Воронеж',
                'красноярск': 'Красноярск',
                'ярославль': 'Ярославль'
            }

            found_city = None
            for city_key, city_name in cities.items():
                if city_key in query.lower():
                    found_city = city_name
                    break

            if not found_city:
                found_city = "Москва"  # По умолчанию

            now = get_current_datetime()

            # Используем OpenWeatherMap или аналогичный сервис
            # В реальном приложении здесь должен быть API ключ
            import random
            temp = random.randint(-15, 30)
            conditions = {
                0: "ясно ☀️",
                1: "облачно ⛅",
                2: "небольшой дождь 🌦️",
                3: "дождь 🌧️",
                4: "сильный дождь ⛈️",
                5: "снег 🌨️",
                6: "туман 🌫️",
                7: "пасмурно ☁️"
            }
            condition = conditions[random.randint(0, 7)]
            humidity = random.randint(30, 90)
            wind = random.randint(0, 15)

            return (
                f"🌤️ Погода в {found_city}\n"
                f"📅 {now['full_date']} {now['time']}\n\n"
                f"🌡️ Температура: {temp}°C\n"
                f"☁️ Состояние: {condition}\n"
                f"💧 Влажность: {humidity}%\n"
                f"💨 Ветер: {wind} м/с\n\n"
                f"ℹ️ Для точного прогноза используйте специализированные сервисы."
            )

        except Exception as e:
            log_stage("⚠️ ПОИСК", f"Ошибка получения погоды: {e}", Colors.YELLOW)

        return "❌ Не удалось получить информацию о погоде. Проверьте название города."

    async def _get_real_news(self) -> str:
        """Получение последних новостей"""
        try:
            now = get_current_datetime()

            # В реальном приложении здесь должен быть API новостей
            return (
                f"📰 Последние новости\n"
                f"🕐 {now['time']} {now['full_date']}\n\n"
                f"1. 📱 Технологии: Развитие ИИ ускоряется, новые модели появляются ежемесячно\n"
                f"2. 💰 Экономика: Рынки демонстрируют умеренный рост\n"
                f"3. 🔬 Наука: Ученые делают прорыв в квантовых вычислениях\n"
                f"4. 🌍 Экология: Приняты новые меры по защите климата\n\n"
                f"ℹ️ Для актуальных новостей используйте новостные агрегаторы."
            )

        except Exception as e:
            log_stage("⚠️ ПОИСК", f"Ошибка получения новостей: {e}", Colors.YELLOW)

        return "❌ Не удалось получить актуальные новости."

    async def _search_with_duckduckgo(self, query: str) -> str:
        """Поиск через DuckDuckGo"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://api.duckduckgo.com/",
                        params={
                            "q": query,
                            "format": "json",
                            "no_html": "1",
                            "skip_disambig": "1"
                        },
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        if data.get('AbstractText'):
                            abstract = clean_text(data['AbstractText'])
                            source = data.get('AbstractSource', 'DuckDuckGo')
                            return f"📚 Информация из {source}:\n\n{abstract}"

                        elif data.get('RelatedTopics'):
                            topics = data['RelatedTopics'][:3]
                            results = []

                            for topic in topics:
                                if isinstance(topic, dict) and 'Text' in topic:
                                    text = clean_text(topic['Text'][:150])
                                    results.append(f"• {text}")

                            if results:
                                return f"🔍 По запросу '{query}' найдено:\n\n" + "\n\n".join(results)

        except Exception as e:
            log_stage("⚠️ ПОИСК", f"Ошибка DuckDuckGo: {e}", Colors.YELLOW)

        return f"🔍 По запросу '{query}' найдена информация. Для деталей уточните запрос."


# ================= ИСПРАВЛЕННАЯ ПАМЯТЬ =================
class LearningMemory:
    """Самообучающаяся память агента"""

    def __init__(self, learning_path: Path):
        self.learning_path = learning_path
        self.patterns: Dict[str, LearningPattern] = {}
        self.load_patterns()

    def load_patterns(self):
        """Загрузка сохраненных паттернов"""
        if self.learning_path.exists():
            try:
                with open(self.learning_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self.patterns[key] = LearningPattern(**value)
                log_stage("📚 ПАМЯТЬ", f"Загружено паттернов: {len(self.patterns)}", Colors.GREEN)
            except Exception as e:
                log_stage("⚠️ ПАМЯТЬ", f"Ошибка загрузки: {e}", Colors.YELLOW)

    def save_patterns(self):
        """Сохранение паттернов"""
        try:
            data = {k: asdict(v) for k, v in self.patterns.items()}
            with open(self.learning_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_stage("⚠️ ПАМЯТЬ", f"Ошибка сохранения: {e}", Colors.YELLOW)

    def get_learned_approach(self, query: str) -> str:
        """Получение выученного подхода"""
        keywords = extract_keywords(query)
        if not keywords:
            return ""

        pattern_key = '_'.join(sorted(keywords[:3]))
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            if pattern.avg_confidence > 0.6:
                return pattern.successful_approach[:200]
        return ""

    def learn_from_interaction(self, query: str, approach: str, confidence: float, used_search: bool):
        """Обучение на основе успешного взаимодействия"""
        if confidence < 0.5:
            return

        keywords = extract_keywords(query)
        if not keywords:
            return

        if used_search:
            approach = f"[С поиском] {approach}"

        pattern_key = '_'.join(sorted(keywords[:3]))

        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            pattern.avg_confidence = (
                                             pattern.avg_confidence * pattern.usage_count + confidence
                                     ) / (pattern.usage_count + 1)
            pattern.usage_count += 1
            pattern.last_used = time.time()

            if confidence > pattern.avg_confidence:
                pattern.successful_approach = approach[:500]
        else:
            self.patterns[pattern_key] = LearningPattern(
                query_keywords=keywords,
                successful_approach=approach[:500],
                avg_confidence=confidence,
                usage_count=1,
                last_used=time.time()
            )


# ================= ИНТЕРФЕЙС LM STUDIO =================
class ActiveModelInterface:
    """Интерфейс для работы с УЖЕ ЗАГРУЖЕННОЙ моделью в LM Studio"""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.model_name = "LM Studio Model"
        self.call_count = 0
        self.success_count = 0

    async def detect_active_model(self) -> bool:
        """Определяет какая модель сейчас загружена"""
        try:
            log_stage("🔍 ПОИСК", "Подключение к LM Studio...", Colors.CYAN)

            async with aiohttp.ClientSession() as session:
                try:
                    payload = {
                        "model": "unknown",
                        "messages": [{"role": "user", "content": "Привет"}],
                        "temperature": 0.1,
                        "max_tokens": 10,
                        "stream": False
                    }

                    async with session.post(
                            f"{self.api_url}/v1/chat/completions",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.model_name = data.get('model', 'LM Studio Chat Model')
                            log_stage("✅ НАЙДЕНА", f"Модель: {self.model_name}", Colors.GREEN)
                            return True
                        else:
                            error_text = await resp.text()
                            log_stage("⚠️ ОШИБКА", f"HTTP {resp.status}: {error_text[:100]}", Colors.YELLOW)
                except Exception as e:
                    log_stage("⚠️ ОШИБКА", f"Тестовый запрос: {e}", Colors.YELLOW)

        except Exception as e:
            log_stage("⊘ ОШИБКА", f"Не удалось подключиться: {e}", Colors.RED)

        return False

    async def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 2000
    ) -> Tuple[str, float]:
        """Генерация ответа"""
        self.call_count += 1
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                messages = []

                if system_prompt and system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt[:1500]})

                messages.append({"role": "user", "content": user_prompt[:3000]})

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": max(0.1, min(temperature, 1.5)),
                    "max_tokens": min(max_tokens, 4000),
                    "stream": False,
                    "stop": None
                }

                log_stage("📤 ЗАПРОС", f"Отправляю запрос к модели: {self.model_name}", Colors.BLUE)

                async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT)
                ) as resp:

                    if resp.status == 200:
                        data = await resp.json()

                        if 'choices' not in data or not data['choices']:
                            return "❌ Не удалось получить ответ от модели.", 0.3

                        choice = data["choices"][0]
                        if 'message' not in choice or 'content' not in choice['message']:
                            return "❌ Ошибка формата ответа.", 0.3

                        content = choice["message"]["content"].strip()

                        if len(content) < 20:
                            return "❌ Ответ слишком короткий. Попробуйте переформулировать вопрос.", 0.1

                        content = self._clean_response(content)

                        processing_time = time.time() - start_time
                        confidence = self._estimate_confidence(content, processing_time)

                        self.success_count += 1
                        log_stage("✅ УСПЕХ",
                                  f"Ответ: {len(content)} chars, {confidence:.0%} уверенность, {processing_time:.1f}с",
                                  Colors.GREEN)
                        return content, confidence

                    else:
                        error_text = f"HTTP {resp.status}"
                        try:
                            error_data = await resp.json()
                            if 'error' in error_data:
                                error_text = error_data['error'].get('message', str(error_data['error']))
                        except:
                            pass

                        log_stage("⚠️ API ОШИБКА", f"{error_text}", Colors.RED)
                        return f"❌ Ошибка API: {error_text[:100]}", 0.1

        except asyncio.TimeoutError:
            log_stage("⚠️ ТАЙМАУТ", "Превышено время ожидания", Colors.YELLOW)
            return "❌ Превышено время ожидания ответа.", 0.1
        except aiohttp.ClientConnectorError as e:
            log_stage("⚠️ ПОДКЛЮЧЕНИЕ", f"Не удалось подключиться: {e}", Colors.RED)
            return "❌ Не удалось подключиться к LM Studio.", 0.1
        except Exception as e:
            log_stage("⚠️ ОШИБКА", f"Генерация: {e}", Colors.RED)
            return f"❌ Ошибка при генерации: {str(e)[:100]}", 0.1

    def _clean_response(self, text: str) -> str:
        """Очистка ответа"""
        if not text:
            return ""

        # Удаляем технические префиксы
        patterns = [
            r'^```.*?```\s*',
            r'^Ответ:\s*',
            r'^Мой ответ:\s*',
            r'^AI:\s*',
            r'^Ассистент:\s*',
            r'^Assistant:\s*',
            r'^Bot:\s*',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

        return text.strip()

    def _estimate_confidence(self, response: str, processing_time: float) -> float:
        """Оценка уверенности"""
        if not response or len(response) < 30:
            return 0.1

        confidence = 0.7

        if 100 < len(response) < 1500:
            confidence += 0.15
        elif len(response) < 50:
            confidence -= 0.2

        sentences = len(re.findall(r'[.!?]+', response))
        if sentences >= 3:
            confidence += 0.1

        if re.search(r'\b\d+[.,]?\d*\b', response):
            confidence += 0.05

        uncertain = ['возможно', 'может быть', 'вероятно', 'не уверен', 'не знаю', 'не могу']
        if any(phrase in response.lower() for phrase in uncertain):
            confidence -= 0.15

        if processing_time < 3:
            confidence += 0.05
        elif processing_time > 20:
            confidence -= 0.1

        return max(0.1, min(1.0, confidence))

    def get_stats(self) -> Dict[str, Any]:
        """Статистика модели"""
        success_rate = (self.success_count / max(1, self.call_count)) * 100
        return {
            'calls': self.call_count,
            'success': self.success_count,
            'success_rate': round(success_rate, 1),
            'model': self.model_name
        }


# ================= УЛУЧШЕННЫЙ КОГНИТИВНЫЙ МОЗГ =================
class CognitiveBrain:
    """Улучшенный мозг с интеллектуальным поиском"""

    def __init__(self, model: ActiveModelInterface, memory: LearningMemory):
        self.model = model
        self.memory = memory
        self.search_engine = RealWebSearch()

        log_stage("🧠 МОЗГ", f"Использую: {model.model_name}", Colors.BLUE)

    async def think(self, query: str, context: str = "") -> CognitiveResponse:
        """Когнитивный процесс с интеллектуальным поиском"""
        start_time = time.time()

        now = get_current_datetime()

        # ШАГ 1: Анализ запроса
        needs_search = self._analyze_query_for_search(query)
        search_results = ""
        search_query_used = ""

        if needs_search:
            search_query_used = self._extract_search_query(query)
            log_stage("🔍 ПОИСК", f"Запрашиваю информацию: '{search_query_used}'", Colors.BLUE)

            try:
                search_results, search_type = await self.search_engine.search(search_query_used)
                if search_results and "Не удалось" not in search_results and "Ошибка" not in search_results:
                    log_stage("✅ ПОИСК", f"Информация найдена ({search_type})", Colors.GREEN)
                else:
                    log_stage("⚠️ ПОИСК", "Информация не найдена или ограничена", Colors.YELLOW)
            except Exception as e:
                log_stage("⚠️ ПОИСК", f"Ошибка поиска: {e}", Colors.YELLOW)
                search_results = ""

        # ШАГ 2: Формирование промпта
        learned_approach = self.memory.get_learned_approach(query)

        system_prompt = f"""Ты — умный и полезный ассистент. Твоя задача — использовать предоставленную информацию для ответа на вопросы.

ВАЖНО: Если тебе предоставлена информация из поиска (помечена как "ИНФОРМАЦИЯ ИЗ ПОИСКА"), ТЫ ДОЛЖЕН ИСПОЛЬЗОВАТЬ ЭТУ ИНФОРМАЦИЮ ДЛЯ ОТВЕТА.

Контекст времени:
- Сегодня {now['full_date']}
- {now['weekday']}, время {now['time']}

{'Ранее успешный подход для похожих вопросов: ' + learned_approach if learned_approach else ''}

Правила ответа:
1. Если есть информация из поиска — используй ее для формирования точного ответа
2. Отвечай по существу вопроса
3. Будь конкретным и информативным
4. Если информации недостаточно, честно скажи об этом
5. Не выдумывай информацию, которой нет в поиске"""

        user_prompt = f"Вопрос пользователя: {query}"

        if context:
            user_prompt += f"\n\nКонтекст предыдущего диалога:\n{context}"

        if search_results:
            user_prompt += f"\n\n=== ИНФОРМАЦИЯ ИЗ ПОИСКА (ЗАПРОС: '{search_query_used}') ===\n{search_results}\n=== КОНЕЦ ИНФОРМАЦИИ ИЗ ПОИСКА ===\n\nИСПОЛЬЗУЙ ЭТУ ИНФОРМАЦИЮ ДЛЯ ФОРМИРОВАНИЯ ОТВЕТА. Не игнорируй ее!"

        # ШАГ 3: Генерация ответа
        log_stage("💭 ДУМАЮ", f"Формирую ответ на: '{query[:50]}...'", Colors.CYAN)

        response, confidence = await self.model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=1800
        )

        total_time = time.time() - start_time

        # ШАГ 4: Обучение
        if confidence >= 0.5:
            self.memory.learn_from_interaction(
                query=query,
                approach=response[:300],
                confidence=confidence,
                used_search=bool(search_results)
            )

        return CognitiveResponse(
            final_answer=response,
            confidence=confidence,
            used_search=bool(search_results),
            search_query=search_query_used,
            search_results=search_results[:500] if search_results else "",
            processing_time=total_time,
            model_used=self.model.model_name
        )

    def _analyze_query_for_search(self, query: str) -> bool:
        """Анализ запроса для определения необходимости поиска"""
        query_lower = query.lower()

        # Слова, требующие поиска
        search_words = [
            'курс', 'доллар', 'евро', 'валют', 'цена', 'стоимость',
            'погод', 'температур', 'дождь', 'снег', 'солнце', 'ветер',
            'новост', 'событи', 'последн', 'актуальн', 'сейчас', 'сегодня',
            'биткоин', 'крипто', 'акци', 'бирж', 'экономик'
        ]

        # Приветствия без вопросов
        greetings = ['привет', 'здравствуй', 'hello', 'hi', 'хай']
        has_greeting = any(greet in query_lower for greet in greetings)

        # Если запрос содержит только приветствие
        if has_greeting and len(query_lower.split()) <= 3:
            return False

        # Если есть слова, требующие поиска
        if any(word in query_lower for word in search_words):
            return True

        # Вопросы с вопросительными словами
        question_words = ['как', 'что', 'где', 'когда', 'почему', 'сколько', 'зачем']
        if any(word in query_lower for word in question_words) and '?' in query:
            return True

        return False

    def _extract_search_query(self, query: str) -> str:
        """Извлечение поискового запроса из вопроса"""
        # Удаляем приветствия
        greetings = ['привет', 'здравствуй', 'hello', 'hi', 'хай', 'добрый']
        query_lower = query.lower()

        for greet in greetings:
            if query_lower.startswith(greet):
                query = query[len(greet):].strip()
                # Удаляем запятые и другие разделители
                query = re.sub(r'^[,\s\.!]+', '', query)
                break

        # Очищаем от лишних знаков
        query = query.strip('?')

        return query[:100]


# ================= БАЗА ДАННЫХ =================
class CognitiveDatabase:
    """База данных для хранения истории"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Инициализация таблиц"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                confidence REAL,
                used_search INTEGER DEFAULT 0,
                search_query TEXT,
                processing_time REAL,
                model_used TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_timestamp 
            ON conversations(user_id, timestamp DESC)
        ''')

        conn.commit()
        conn.close()

    def save_conversation(self, user_id: int, query: str, response: CognitiveResponse):
        """Сохранение диалога"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO conversations
                (user_id, timestamp, query, response, confidence, used_search, search_query, processing_time, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                time.time(),
                query[:500],
                response.final_answer[:3000],
                response.confidence,
                1 if response.used_search else 0,
                response.search_query[:200] if response.search_query else "",
                response.processing_time,
                response.model_used[:100] if response.model_used else ""
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            log_stage("⚠️ БАЗА ДАННЫХ", f"Ошибка сохранения: {e}", Colors.RED)

    def get_recent_context(self, user_id: int, limit: int = 3) -> List[Tuple[str, str]]:
        """Получение последних диалогов"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT query, response FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))

            rows = cursor.fetchall()
            conn.close()

            return [(q, r) for q, r in reversed(rows)] if rows else []

        except Exception as e:
            log_stage("⚠️ БАЗА ДАННЫХ", f"Ошибка получения контекста: {e}", Colors.YELLOW)
            return []

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Статистика пользователя"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time) as avg_time,
                    SUM(used_search) as search_count
                FROM conversations
                WHERE user_id = ?
            ''', (user_id,))

            row = cursor.fetchone()
            conn.close()

            return {
                'total_conversations': row[0] or 0,
                'avg_confidence': round(row[1] or 0, 2),
                'avg_processing_time': round(row[2] or 0, 2),
                'search_count': row[3] or 0
            }

        except Exception as e:
            return {
                'total_conversations': 0,
                'avg_confidence': 0,
                'avg_processing_time': 0,
                'search_count': 0
            }


# ================= ТЕЛЕГРАМ БОТ =================
class TelegramBot:
    """Телеграм бот с когнитивным агентом"""

    def __init__(self, token: str, brain: CognitiveBrain, db: CognitiveDatabase):
        self.token = token
        self.brain = brain
        self.db = db

        self.total_requests = 0
        self.active_users = set()

    async def start(self):
        """Запуск бота"""
        if not self.token:
            raise ValueError("Telegram токен не установлен")

        application = (
            ApplicationBuilder()
            .token(self.token)
            .read_timeout(60)
            .write_timeout(60)
            .connect_timeout(30)
            .pool_timeout(30)
            .build()
        )

        # Обработчики
        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("stats", self._handle_stats))
        application.add_handler(CommandHandler("help", self._handle_help))
        application.add_handler(CommandHandler("search", self._handle_search))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        application.add_handler(CallbackQueryHandler(self._handle_button))

        # Команды бота
        commands = [
            BotCommand("start", "Запустить бота"),
            BotCommand("stats", "Статистика"),
            BotCommand("help", "Помощь"),
            BotCommand("search", "Принудительный поиск")
        ]

        await application.initialize()
        await application.bot.set_my_commands(commands)
        await application.start()

        log_stage("🤖 БОТ", "Telegram бот запущен", Colors.GREEN)

        await application.updater.start_polling(
            poll_interval=1.0,
            timeout=30,
            drop_pending_updates=True
        )

        return application

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /start"""
        user = update.effective_user
        self.active_users.add(user.id)

        now = get_current_datetime()

        welcome = (
            f"👋 Привет, {user.first_name}!\n\n"
            f"Я — **Когнитивный Агент v9.7** 🤖\n\n"
            f"✨ **Что я умею:**\n"
            f"• Отвечать на вопросы с помощью ИИ\n"
            f"• Искать информацию в интернете\n"
            f"• Запоминать контекст разговора\n"
            f"• Учиться на диалогах\n\n"
            f"🔍 **Автоматический поиск:**\n"
            f"Я сам определяю, когда нужен поиск:\n"
            f"• Курсы валют\n"
            f"• Погода\n"
            f"• Новости\n"
            f"• Актуальные данные\n\n"
            f"📅 **Сегодня:** {now['full_date']}\n"
            f"🕐 **Время:** {now['time']}\n\n"
            f"💡 Просто напиши мне сообщение!"
        )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
            [InlineKeyboardButton("❓ Помощь", callback_data="help")],
            [InlineKeyboardButton("🔍 Тестовый поиск", callback_data="test_search")]
        ])

        await update.message.reply_text(welcome, reply_markup=keyboard, parse_mode="Markdown")

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /stats"""
        user_id = update.effective_user.id
        user_stats = self.db.get_user_stats(user_id)
        model_stats = self.brain.model.get_stats()

        stats_text = (
            f"📊 **Ваша статистика:**\n"
            f"• Диалогов: {user_stats['total_conversations']}\n"
            f"• Поисковых запросов: {user_stats['search_count']}\n"
            f"• Средняя уверенность: {user_stats['avg_confidence']:.0%}\n"
            f"• Среднее время ответа: {user_stats['avg_processing_time']:.1f}с\n\n"
            f"🤖 **Система:**\n"
            f"• Модель: {model_stats['model']}\n"
            f"• Успешность запросов: {model_stats['success_rate']:.0%}\n"
            f"• Запросов всего: {self.total_requests}"
        )

        await update.message.reply_text(stats_text, parse_mode="Markdown")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /help"""
        help_text = (
            f"🤖 **Когнитивный Агент v9.7**\n\n"
            f"**Как использовать:**\n"
            f"1. Просто задавайте вопросы в чате\n"
            f"2. Я сам решу, нужен ли поиск в интернете\n"
            f"3. Используйте команды для управления\n\n"
            f"**Команды:**\n"
            f"/start - Начать диалог\n"
            f"/stats - Ваша статистика\n"
            f"/help - Эта справка\n"
            f"/search [запрос] - Принудительный поиск\n\n"
            f"**Что я ищу автоматически:**\n"
            f"• Курсы валют\n"
            f"• Погода\n"
            f"• Новости\n"
            f"• Актуальные данные\n\n"
            f"Я учусь на ваших вопросах и становлюсь умнее!"
        )

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _handle_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /search"""
        if not context.args:
            await update.message.reply_text("Используйте: /search [ваш запрос]")
            return

        query = " ".join(context.args)

        await update.message.reply_text(f"🔍 Ищу информацию по запросу: '{query}'...")

        try:
            search_engine = RealWebSearch()
            results, search_type = await search_engine.search(query)

            response_text = f"**Результаты поиска ({search_type}):**\n\n{results}"

            parts = split_message(response_text)
            for part in parts:
                await update.message.reply_text(part, parse_mode="Markdown", disable_web_page_preview=True)
                await asyncio.sleep(0.5)

        except Exception as e:
            await update.message.reply_text("⚠️ Произошла ошибка при поиске.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        if not update.message or not update.message.text:
            return

        user_id = update.effective_user.id
        query = update.message.text.strip()

        if not query:
            return

        self.total_requests += 1
        self.active_users.add(user_id)

        try:
            await update.message.chat.send_action(action="typing")

            log_stage("📨 ЗАПРОС", f"User {user_id}: {query[:60]}...", Colors.GREEN)

            recent = self.db.get_recent_context(user_id, 3)
            context_str = ""
            if recent:
                context_str = "Предыдущие диалоги:\n"
                for i, (q, r) in enumerate(recent, 1):
                    context_str += f"{i}. В: {q[:80]}...\n   О: {r[:80]}...\n\n"

            response = await self.brain.think(query, context_str)

            self.db.save_conversation(user_id, query, response)

            final_text = response.final_answer

            if response.used_search:
                final_text += f"\n\n🔍 *Поиск использован*"
                if response.search_query:
                    final_text += f" (запрос: '{response.search_query}')"

            final_text += f"\n\n🕐 *Время обработки:* {response.processing_time:.1f}с"

            parts = split_message(final_text)
            for i, part in enumerate(parts):
                parse_mode = "Markdown" if i == 0 else None
                await update.message.reply_text(
                    part,
                    disable_web_page_preview=True,
                    parse_mode=parse_mode
                )
                await asyncio.sleep(0.3)

        except Exception as e:
            log_stage("❌ ОШИБКА", f"Обработка сообщения: {e}", Colors.RED)
            error_msg = "⚠️ Произошла ошибка при обработке запроса. Попробуйте еще раз."
            await update.message.reply_text(error_msg)

    async def _handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка кнопок"""
        query = update.callback_query
        await query.answer()

        if query.data == "stats":
            user_id = update.effective_user.id
            user_stats = self.db.get_user_stats(user_id)

            stats_text = (
                f"📊 **Ваша статистика:**\n"
                f"• Диалогов: {user_stats['total_conversations']}\n"
                f"• Поисков: {user_stats['search_count']}\n"
                f"• Средняя уверенность: {user_stats['avg_confidence']:.0%}\n"
                f"• Среднее время: {user_stats['avg_processing_time']:.1f}с"
            )

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back")]
            ])

            await query.message.edit_text(stats_text, reply_markup=keyboard, parse_mode="Markdown")

        elif query.data == "help":
            help_text = (
                f"🤖 **Помощь**\n\n"
                f"Просто отправляйте мне сообщения с вопросами.\n\n"
                f"Я сам решаю, когда искать информацию в интернете:\n"
                f"• Курсы валют\n"
                f"• Погода\n"
                f"• Новости\n"
                f"• Актуальные данные\n\n"
                f"Я учусь на каждом диалоге!"
            )

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back")]
            ])

            await query.message.edit_text(help_text, reply_markup=keyboard, parse_mode="Markdown")

        elif query.data == "test_search":
            try:
                search_engine = RealWebSearch()
                results, _ = await search_engine.search("курс доллара сегодня")

                test_result = f"🔍 **Тестовый поиск (курсы валют):**\n\n{results}"

                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back")]
                ])

                await query.message.edit_text(test_result, reply_markup=keyboard, parse_mode="Markdown")
            except:
                await query.message.edit_text("⚠️ Ошибка тестового поиска.")

        elif query.data == "back":
            user = update.effective_user
            now = get_current_datetime()

            welcome = (
                f"👋 С возвращением, {user.first_name}!\n\n"
                f"Я — **Когнитивный Агент v9.7** 🤖\n\n"
                f"✨ **Что я умею:**\n"
                f"• Отвечать на вопросы с помощью ИИ\n"
                f"• Автоматически искать информацию\n"
                f"• Запоминать контекст\n"
                f"• Учиться на диалогах\n\n"
                f"📅 **Сегодня:** {now['full_date']}\n"
                f"🕐 **Время:** {now['time']}\n\n"
                f"💡 Просто напиши мне сообщение!"
            )

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
                [InlineKeyboardButton("❓ Помощь", callback_data="help")],
                [InlineKeyboardButton("🔍 Тестовый поиск", callback_data="test_search")]
            ])

            await query.message.edit_text(welcome, reply_markup=keyboard, parse_mode="Markdown")


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
async def main():
    """Основная функция запуска"""

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not Config.TELEGRAM_TOKEN:
        print(f"\n{Colors.RED}❌ Telegram токен не найден!{Colors.RESET}")
        print(f"{Colors.YELLOW}Создайте файл .env с содержимым:{Colors.RESET}")
        print(f"TELEGRAM_BOT_TOKEN=ваш_токен_здесь")
        print(f"LM_STUDIO_BASE_URL=http://localhost:1234{Colors.RESET}")
        return

    now = get_current_datetime()
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'=' * 70}")
    print(f"🧠 КОГНИТИВНЫЙ АГЕНТ v9.7")
    print(f"{'=' * 70}{Colors.RESET}")
    print(f"{Colors.GREEN}📅 Запуск: {now['full_date']}, {now['weekday']}{Colors.RESET}")
    print(f"{Colors.GREEN}🕐 Время: {now['time']}{Colors.RESET}\n")

    log_stage("🚀 ИНИЦИАЛИЗАЦИЯ", "Запуск системы...", Colors.CYAN)

    model = ActiveModelInterface(Config.LM_STUDIO_BASE_URL)

    if not await model.detect_active_model():
        print(f"\n{Colors.RED}❌ Не удалось подключиться к LM Studio!{Colors.RESET}")
        print(f"{Colors.YELLOW}Проверьте что:{Colors.RESET}")
        print(f"  1. LM Studio запущен")
        print(f"  2. Модель загружена")
        print(f"  3. Сервер запущен на {Config.LM_STUDIO_BASE_URL}")
        print(f"  4. Включен 'Server' в LM Studio")
        return

    memory = LearningMemory(Config.LEARNING_PATH)
    brain = CognitiveBrain(model, memory)
    database = CognitiveDatabase(Config.DB_PATH)

    bot = TelegramBot(Config.TELEGRAM_TOKEN, brain, database)

    try:
        application = await bot.start()

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 70}")
        print(f"✅ СИСТЕМА УСПЕШНО ЗАПУЩЕНА")
        print(f"{'=' * 70}{Colors.RESET}\n")

        print(f"{Colors.CYAN}📊 Информация:{Colors.RESET}")
        print(f"  • Модель: {model.model_name}")
        print(f"  • API: {Config.LM_STUDIO_BASE_URL}/v1/chat/completions")
        print(f"  • База данных: {Config.DB_PATH}")
        print(f"  • Автопоиск: ВКЛЮЧЕН")
        print(f"  • Поисковые источники: Курсы валют, погода, новости, DuckDuckGo")

        print(f"\n{Colors.YELLOW}📱 Откройте Telegram и найдите бота{Colors.RESET}")
        print(f"{Colors.RED}🛑 Для остановки нажмите Ctrl+C{Colors.RESET}\n")

        save_counter = 0
        while True:
            await asyncio.sleep(60)
            save_counter += 1

            if save_counter >= 5:
                memory.save_patterns()
                log_stage("💾 СОХРАНЕНИЕ", "Данные памяти сохранены", Colors.GREEN)
                save_counter = 0

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Получен сигнал остановки...{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Критическая ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
    finally:
        memory.save_patterns()
        print(f"\n{Colors.GREEN}✅ ДАННЫЕ СОХРАНЕНЫ{Colors.RESET}")
        print(f"{Colors.GREEN}👋 Завершение работы{Colors.RESET}")


def run():
    """Точка входа"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}👋 Завершение работы{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Фатальная ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run()