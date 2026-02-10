# coding: utf-8
"""
Cognitive_Agent_Pro_v9.4_SINGLE_MODEL.py
🧠 Когнитивный агент - работа с УЖЕ ЗАГРУЖЕННОЙ моделью
✅ Исправлены ошибки HTTP 400
✅ Правильный формат запросов к LM Studio
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
    from telegram.error import TelegramError
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

    TIMEOUT = 30
    MAX_TOKENS = 1500

    # Telegram
    MAX_MESSAGE_LENGTH = 4096


# ================= СТРУКТУРЫ ДАННЫХ =================
@dataclass
class CognitiveResponse:
    """Полный когнитивный ответ"""
    final_answer: str
    confidence: float
    used_search: bool
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


# ================= УПРОЩЁННЫЙ ВЕБ-ПОИСК =================
class SimpleWebSearch:
    """Упрощённый веб-поиск"""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}

    async def search(self, query: str) -> str:
        """Поиск с кэшированием"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        current_time = time.time()

        # Проверка кэша
        if cache_key in self.cache:
            if current_time - self.cache_ttl[cache_key] < 1800:  # 30 минут
                return self.cache[cache_key]
            else:
                del self.cache[cache_key]
                del self.cache_ttl[cache_key]

        result = await self._search_finance(query)

        if not result:
            result = f"По запросу '{query}' найдена общая информация."

        # Сохраняем в кэш
        self.cache[cache_key] = result
        self.cache_ttl[cache_key] = current_time

        return result

    async def _search_finance(self, query: str) -> str:
        """Поиск финансовых данных"""
        if any(kw in query.lower() for kw in ['курс', 'доллар', 'евро', 'валют', 'рубл', 'юан', 'биткоин']):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            "https://www.cbr-xml-daily.ru/daily_json.js",
                            timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            # Очистка комментариев если есть
                            text = re.sub(r'^/\*.*?\*/', '', text, flags=re.DOTALL).strip()
                            data = json.loads(text)

                            usd = data['Valute']['USD']
                            eur = data['Valute'].get('EUR', {})
                            cny = data['Valute'].get('CNY', {})

                            now = get_current_datetime()

                            result = (
                                f"💰 КУРСЫ ВАЛЮТ ЦБ РФ на {now['time']}\n"
                                f"📅 {now['full_date']}\n\n"
                                f"💵 Доллар США (USD): {usd['Value']:.2f} ₽\n"
                                f"📈 Изменение: {usd['Value'] - usd['Previous']:+.2f} ₽\n\n"
                                f"💶 Евро (EUR): {eur.get('Value', 0):.2f} ₽\n"
                                f"📈 Изменение: {eur.get('Value', 0) - eur.get('Previous', 0):+.2f} ₽\n\n"
                                f"💴 Юань (CNY): {cny.get('Value', 0):.2f} ₽"
                            )

                            return result
            except Exception as e:
                log_stage("⚠️ ПОИСК", f"Ошибка получения курсов: {e}", Colors.YELLOW)

        return ""


# ================= ОБУЧАЮЩАЯСЯ ПАМЯТЬ =================
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

    def learn_from_interaction(self, query: str, approach: str, confidence: float):
        """Обучение на основе успешного взаимодействия"""
        if confidence < 0.5:
            return

        keywords = extract_keywords(query)
        if not keywords:
            return

        pattern_key = '_'.join(sorted(keywords[:3]))

        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            pattern.avg_confidence = (
                                             pattern.avg_confidence * pattern.usage_count + confidence
                                     ) / (pattern.usage_count + 1)
            pattern.usage_count += 1
            pattern.last_used = time.time()
        else:
            self.patterns[pattern_key] = LearningPattern(
                query_keywords=keywords,
                successful_approach=approach[:500],
                avg_confidence=confidence,
                usage_count=1,
                last_used=time.time()
            )


# ================= ИСПРАВЛЕННЫЙ ИНТЕРФЕЙС LM STUDIO =================
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

            # Проверяем доступность сервера
            async with aiohttp.ClientSession() as session:
                # Простая проверка подключения
                try:
                    async with session.get(
                            f"{self.api_url}/v1/models",
                            timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get('data'):
                                self.model_name = data['data'][0].get('id', 'Неизвестная модель')
                            log_stage("✅ НАЙДЕНА", f"Модель: {self.model_name}", Colors.GREEN)
                            return True
                except:
                    pass

                # Альтернативная проверка через chat/completions
                log_stage("🔍 ПОИСК", "Проверяем через chat/completions...", Colors.CYAN)
                try:
                    payload = {
                        "messages": [
                            {"role": "user", "content": "Привет"}
                        ],
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
                    log_stage("⚠️ ОШИБКА", f"Подключение: {e}", Colors.YELLOW)

        except Exception as e:
            log_stage("⊘ ОШИБКА", f"Не удалось подключиться: {e}", Colors.RED)

        return False

    async def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 1500
    ) -> Tuple[str, float]:
        """Генерация ответа с правильным форматом для LM Studio"""
        self.call_count += 1
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # ПРАВИЛЬНЫЙ формат для LM Studio
                messages = []

                # Добавляем системный промпт как отдельное сообщение
                if system_prompt and system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt[:800]})

                # Добавляем пользовательский промпт
                messages.append({"role": "user", "content": user_prompt[:2000]})

                payload = {
                    "messages": messages,
                    "temperature": max(0.1, min(temperature, 2.0)),  # Валидный диапазон
                    "max_tokens": min(max_tokens, 4000),  # Ограничение
                    "stream": False,
                    "stop": None  # Явно указываем stop
                }

                log_stage("📤 ЗАПРОС", f"Отправляю {len(messages)} сообщений...", Colors.BLUE)

                async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT)
                ) as resp:

                    if resp.status == 200:
                        data = await resp.json()

                        if 'choices' not in data or not data['choices']:
                            return "Ошибка: пустой ответ от модели", 0.3

                        choice = data["choices"][0]
                        if 'message' not in choice or 'content' not in choice['message']:
                            return "Ошибка: неправильный формат ответа", 0.3

                        content = choice["message"]["content"].strip()
                        content = self._clean_response(content)

                        processing_time = time.time() - start_time
                        confidence = self._estimate_confidence(content, processing_time)

                        self.success_count += 1
                        log_stage("✅ УСПЕХ", f"Ответ получен ({len(content)} chars)", Colors.GREEN)
                        return content, confidence

                    else:
                        error_text = f"HTTP {resp.status}"
                        try:
                            error_data = await resp.json()
                            if 'error' in error_data:
                                error_text = error_data['error'].get('message', str(error_data['error']))
                            else:
                                error_text = str(error_data)[:200]
                        except:
                            try:
                                error_text = await resp.text()
                            except:
                                pass

                        log_stage("⚠️ API ОШИБКА", f"{error_text}", Colors.RED)

                        # Детальная информация об ошибке
                        if resp.status == 400:
                            log_stage("🔧 ДЕБАГ", f"Payload был: {json.dumps(payload, ensure_ascii=False)[:300]}",
                                      Colors.YELLOW)

                        return f"Ошибка API ({resp.status}): {error_text[:100]}", 0.1

        except asyncio.TimeoutError:
            log_stage("⚠️ ТАЙМАУТ", "Превышено время ожидания", Colors.YELLOW)
            return "Превышено время ожидания ответа от модели", 0.1
        except aiohttp.ClientConnectorError as e:
            log_stage("⚠️ ПОДКЛЮЧЕНИЕ", f"Не удалось подключиться: {e}", Colors.RED)
            return "Не удалось подключиться к LM Studio. Проверьте запущен ли сервер.", 0.1
        except Exception as e:
            log_stage("⚠️ ОШИБКА", f"Генерация: {e}", Colors.RED)
            return f"Ошибка при генерации: {str(e)[:100]}", 0.1

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

        # Удаляем лишние пробелы и переносы
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

        return text.strip()

    def _estimate_confidence(self, response: str, processing_time: float) -> float:
        """Оценка уверенности"""
        if not response:
            return 0.1

        confidence = 0.7

        # Длина ответа
        if 50 < len(response) < 1000:
            confidence += 0.1
        elif len(response) < 20:
            confidence -= 0.3
        elif len(response) > 2000:
            confidence -= 0.1

        # Качество ответа
        sentences = len(re.findall(r'[.!?]+', response))
        if sentences >= 2:
            confidence += 0.05

        # Признаки неуверенности
        uncertain = ['возможно', 'может быть', 'вероятно', 'не уверен', 'не знаю', 'не могу', 'извините',
                     'сложно сказать']
        if any(phrase in response.lower() for phrase in uncertain):
            confidence -= 0.15

        # Время обработки
        if processing_time < 3:
            confidence += 0.05
        elif processing_time > 15:
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


# ================= КОГНИТИВНЫЙ МОЗГ =================
class CognitiveBrain:
    """Мозг работающий с одной активной моделью"""

    def __init__(self, model: ActiveModelInterface, memory: LearningMemory):
        self.model = model
        self.memory = memory

        log_stage("🧠 МОЗГ", f"Использую: {model.model_name}", Colors.BLUE)

    async def think(self, query: str, context: str = "") -> CognitiveResponse:
        """Когнитивный процесс"""
        start_time = time.time()

        now = get_current_datetime()

        # Получаем выученный подход если есть
        learned_approach = self.memory.get_learned_approach(query)

        # Формируем промпт
        system_prompt = f"""Ты — умный и полезный ассистент. Отвечай кратко, точно и по делу.

Контекст времени:
- Сегодня {now['full_date']}
- {now['weekday']}, время {now['time']}

{'Выученный подход для похожих вопросов: ' + learned_approach if learned_approach else ''}

Правила:
1. Отвечай по существу вопроса
2. Если не знаешь точно — честно скажи об этом
3. Используй современные данные
4. Будь вежливым и полезным"""

        # Добавляем контекст если есть
        if context:
            system_prompt += f"\n\nКонтекст диалога:\n{context}"

        user_prompt = f"Вопрос пользователя: {query}"

        log_stage("💭 ДУМАЮ", f"Обрабатываю: '{query[:50]}...'", Colors.CYAN)

        # Генерация ответа
        response, confidence = await self.model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=1200
        )

        total_time = time.time() - start_time

        # Обучение на успешном ответе
        if confidence >= 0.5:
            self.memory.learn_from_interaction(query, response[:300], confidence)

        return CognitiveResponse(
            final_answer=response,
            confidence=confidence,
            used_search=False,  # Используем только как флаг
            processing_time=total_time,
            model_used=self.model.model_name
        )


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
                processing_time REAL
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversations
            (user_id, timestamp, query, response, confidence, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            time.time(),
            query[:500],
            response.final_answer[:2000],
            response.confidence,
            response.processing_time
        ))

        conn.commit()
        conn.close()

    def get_recent_context(self, user_id: int, limit: int = 3) -> List[Tuple[str, str]]:
        """Получение последних диалогов"""
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

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Статистика пользователя"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_time
            FROM conversations
            WHERE user_id = ?
        ''', (user_id,))

        row = cursor.fetchone()
        conn.close()

        return {
            'total_conversations': row[0] or 0,
            'avg_confidence': round(row[1] or 0, 2),
            'avg_processing_time': round(row[2] or 0, 2)
        }


# ================= ТЕЛЕГРАМ БОТ =================
class TelegramBot:
    """Телеграм бот с когнитивным агентом"""

    def __init__(self, token: str, brain: CognitiveBrain, search: SimpleWebSearch, db: CognitiveDatabase):
        self.token = token
        self.brain = brain
        self.search = search
        self.db = db

        # Статистика
        self.total_requests = 0
        self.active_users = set()

    async def start(self):
        """Запуск бота"""
        if not self.token:
            raise ValueError("Telegram токен не установлен")

        # Создаем приложение
        application = ApplicationBuilder().token(self.token).build()

        # Добавляем обработчики
        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("stats", self._handle_stats))
        application.add_handler(CommandHandler("help", self._handle_help))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        application.add_handler(CallbackQueryHandler(self._handle_button))

        # Команды бота
        commands = [
            BotCommand("start", "Запустить бота"),
            BotCommand("stats", "Статистика"),
            BotCommand("help", "Помощь")
        ]

        # Запускаем
        await application.initialize()
        await application.bot.set_my_commands(commands)
        await application.start()

        log_stage("🤖 БОТ", "Telegram бот запущен", Colors.GREEN)

        # Запускаем polling
        await application.updater.start_polling(
            poll_interval=0.5,
            timeout=10,
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
            f"Я — **Когнитивный Агент v9.4** 🤖\n\n"
            f"✨ **Что я умею:**\n"
            f"• Отвечать на вопросы с помощью ИИ\n"
            f"• Искать актуальные данные (курсы валют)\n"
            f"• Запоминать контекст разговора\n"
            f"• Учиться на диалогах\n\n"
            f"📅 **Сегодня:** {now['full_date']}\n"
            f"🕐 **Время:** {now['time']}\n\n"
            f"💡 Просто напиши мне сообщение!"
        )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
            [InlineKeyboardButton("❓ Помощь", callback_data="help")]
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
            f"• Средняя уверенность: {user_stats['avg_confidence']:.0%}\n"
            f"• Среднее время ответа: {user_stats['avg_processing_time']:.1f}с\n\n"
            f"🤖 **Система:**\n"
            f"• Модель: {model_stats['model']}\n"
            f"• Успешность: {model_stats['success_rate']:.0%}\n"
            f"• Запросов всего: {self.total_requests}"
        )

        await update.message.reply_text(stats_text, parse_mode="Markdown")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /help"""
        help_text = (
            f"🤖 **Когнитивный Агент v9.4**\n\n"
            f"**Как использовать:**\n"
            f"1. Просто задавайте вопросы в чате\n"
            f"2. Используйте команды для управления\n\n"
            f"**Команды:**\n"
            f"/start - Начать диалог\n"
            f"/stats - Ваша статистика\n"
            f"/help - Эта справка\n\n"
            f"**Возможности:**\n"
            f"• Ответы на любые вопросы\n"
            f"• Поиск курсов валют\n"
            f"• Контекст диалога (помнит 3 последних сообщения)\n"
            f"• Самообучение на основе ваших вопросов"
        )

        await update.message.reply_text(help_text, parse_mode="Markdown")

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
            # Показываем "печатает"
            await update.message.chat.send_action(action="typing")

            # Проверка простых запросов
            if self._is_simple_query(query):
                response = self._handle_simple_query(query)
                await update.message.reply_text(response)
                return

            # Получаем контекст
            recent = self.db.get_recent_context(user_id, 2)
            context_str = ""
            if recent:
                context_str = "Предыдущие диалоги:\n"
                for i, (q, r) in enumerate(recent, 1):
                    context_str += f"{i}. Вопрос: {q[:60]}\n   Ответ: {r[:60]}\n"

            # Веб-поиск если нужно
            search_results = ""
            if self._needs_search(query):
                search_results = await self.search.search(query)
                if search_results:
                    context_str += f"\nАктуальная информация:\n{search_results}\n"

            # Обработка мозгом
            response = await self.brain.think(query, context_str)

            # Сохраняем
            self.db.save_conversation(user_id, query, response)

            # Форматируем ответ
            final_text = response.final_answer

            # Добавляем мета-информацию если низкая уверенность
            if response.confidence < 0.4:
                final_text += "\n\n⚠️ *Уверенность в ответе низкая*"

            # Отправляем
            parts = split_message(final_text)
            for i, part in enumerate(parts):
                if i == 0:
                    await update.message.reply_text(part,
                                                    disable_web_page_preview=True,
                                                    parse_mode="Markdown" if response.confidence < 0.4 else None)
                else:
                    await update.message.reply_text(part, disable_web_page_preview=True)
                await asyncio.sleep(0.3)

        except Exception as e:
            log_stage("❌ ОШИБКА", f"Обработка сообщения: {e}", Colors.RED)
            error_msg = (
                "⚠️ Произошла ошибка при обработке запроса.\n\n"
                "**Возможные причины:**\n"
                "1. LM Studio не отвечает\n"
                "2. Модель не загружена\n"
                "3. Проблемы с подключением\n\n"
                "Проверьте что LM Studio запущен и модель загружена."
            )
            await update.message.reply_text(error_msg, parse_mode="Markdown")

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
                f"• Средняя уверенность: {user_stats['avg_confidence']:.0%}\n"
                f"• Среднее время: {user_stats['avg_processing_time']:.1f}с"
            )

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back")]
            ])

            await query.message.edit_text(stats_text, reply_markup=keyboard, parse_mode="Markdown")

        elif query.data == "help":
            help_text = (
                f"🤖 **Помощь по использованию**\n\n"
                f"Просто отправляйте мне сообщения с вопросами.\n\n"
                f"Я могу:\n"
                f"• Отвечать на общие вопросы\n"
                f"• Искать актуальные данные\n"
                f"• Помнить контекст диалога\n\n"
                f"Для работы нужен запущенный LM Studio с загруженной моделью."
            )

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back")]
            ])

            await query.message.edit_text(help_text, reply_markup=keyboard, parse_mode="Markdown")

        elif query.data == "back":
            # Возврат к стартовому сообщению
            user = update.effective_user
            now = get_current_datetime()

            welcome = (
                f"👋 Привет, {user.first_name}!\n\n"
                f"Я — **Когнитивный Агент v9.4** 🤖\n\n"
                f"✨ **Что я умею:**\n"
                f"• Отвечать на вопросы с помощью ИИ\n"
                f"• Искать актуальные данные\n"
                f"• Запоминать контекст\n"
                f"• Учиться на диалогах\n\n"
                f"📅 **Сегодня:** {now['full_date']}\n"
                f"🕐 **Время:** {now['time']}\n\n"
                f"💡 Просто напиши мне сообщение!"
            )

            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
                [InlineKeyboardButton("❓ Помощь", callback_data="help")]
            ])

            await query.message.edit_text(welcome, reply_markup=keyboard, parse_mode="Markdown")

    def _is_simple_query(self, query: str) -> bool:
        """Проверка простых запросов"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['привет', 'здравствуй', 'hello', 'hi', 'хай']):
            return True

        time_patterns = [
            r'который час',
            r'сколько времени',
            r'какая дата',
            r'какое сегодня число',
            r'какой год',
            r'какой месяц',
            r'какой день',
            r'время',
            r'дата'
        ]

        return any(re.search(pattern, query_lower) for pattern in time_patterns)

    def _handle_simple_query(self, query: str) -> str:
        """Обработка простых запросов"""
        query_lower = query.lower()
        now = get_current_datetime()

        if any(word in query_lower for word in ['привет', 'здравствуй', 'hello', 'hi', 'хай']):
            return f"👋 Привет! Сейчас {now['time']}, {now['weekday']}. Чем могу помочь?"

        if re.search(r'который час|сколько времени|время', query_lower):
            return f"🕐 Сейчас {now['time_full']}"

        if re.search(r'какая дата|какое сегодня число|дата', query_lower):
            return f"📅 Сегодня {now['full_date']}"

        if re.search(r'какой год', query_lower):
            return f"📅 Сейчас {now['year']} год"

        if re.search(r'какой месяц', query_lower):
            return f"📅 Сейчас {now['month_name']}"

        if re.search(r'какой день', query_lower):
            return f"📅 Сегодня {now['weekday']}"

        return ""

    def _needs_search(self, query: str) -> bool:
        """Определение необходимости поиска"""
        search_keywords = [
            'курс', 'доллар', 'евро', 'валют', 'рубл', 'юан',
            'биткоин', 'крипто', 'цена', 'стоимость', 'акции',
            'фондовый', 'рынок', 'экономик', 'биржа'
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in search_keywords)


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
async def main():
    """Основная функция запуска"""

    # Настройка логирования
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Уменьшаем логирование httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Проверка конфигурации
    if not Config.TELEGRAM_TOKEN:
        print(f"\n{Colors.RED}❌ Telegram токен не найден!{Colors.RESET}")
        print(f"{Colors.YELLOW}Создайте файл .env с содержимым:{Colors.RESET}")
        print(f"TELEGRAM_BOT_TOKEN=ваш_токен_здесь")
        print(f"LM_STUDIO_BASE_URL=http://localhost:1234{Colors.RESET}")
        return

    now = get_current_datetime()
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'=' * 70}")
    print(f"🧠 КОГНИТИВНЫЙ АГЕНТ v9.4")
    print(f"{'=' * 70}{Colors.RESET}")
    print(f"{Colors.GREEN}📅 Запуск: {now['full_date']}, {now['weekday']}{Colors.RESET}")
    print(f"{Colors.GREEN}🕐 Время: {now['time']}{Colors.RESET}\n")

    # Инициализация компонентов
    log_stage("🚀 ИНИЦИАЛИЗАЦИЯ", "Запуск системы...", Colors.CYAN)

    # Подключение к модели
    model = ActiveModelInterface(Config.LM_STUDIO_BASE_URL)

    if not await model.detect_active_model():
        print(f"\n{Colors.RED}❌ Не удалось подключиться к LM Studio!{Colors.RESET}")
        print(f"{Colors.YELLOW}Проверьте что:{Colors.RESET}")
        print(f"  1. LM Studio запущен")
        print(f"  2. Модель загружена (кнопка 'Load')")
        print(f"  3. Сервер запущен на {Config.LM_STUDIO_BASE_URL}")
        print(f"  4. В настройках LM Studio включен 'Server' (вкладка 'Server')")
        print(f"  5. Попробуйте в LM Studio: Menu → View → Server Settings")
        return

    # Инициализация других компонентов
    memory = LearningMemory(Config.LEARNING_PATH)
    brain = CognitiveBrain(model, memory)
    search = SimpleWebSearch()
    database = CognitiveDatabase(Config.DB_PATH)

    # Создание и запуск бота
    bot = TelegramBot(Config.TELEGRAM_TOKEN, brain, search, database)

    try:
        application = await bot.start()

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 70}")
        print(f"✅ СИСТЕМА УСПЕШНО ЗАПУЩЕНА")
        print(f"{'=' * 70}{Colors.RESET}\n")

        print(f"{Colors.CYAN}📊 Информация:{Colors.RESET}")
        print(f"  • Модель: {model.model_name}")
        print(f"  • API: {Config.LM_STUDIO_BASE_URL}/v1/chat/completions")
        print(f"  • База данных: {Config.DB_PATH}")
        print(f"  • Файл обучения: {Config.LEARNING_PATH}")

        print(f"\n{Colors.YELLOW}📱 Откройте Telegram и найдите бота{Colors.RESET}")
        print(f"{Colors.RED}🛑 Для остановки нажмите Ctrl+C{Colors.RESET}\n")

        # Бесконечный цикл с периодическим сохранением
        save_counter = 0
        while True:
            await asyncio.sleep(60)  # 1 минута
            save_counter += 1

            if save_counter >= 5:  # Каждые 5 минут
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
        # Сохранение перед выходом
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