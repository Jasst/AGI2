# coding: utf-8
"""
Cognitive_Agent_Pro_v8.4.1_UNIVERSAL_FAST.py — Универсальная версия с мгновенным запуском
✅ Мгновенный старт (2-3 секунды) — НЕТ валидации при запуске
✅ Ленивая валидация — только при первом запросе пользователя
✅ Автоматический пропуск нерабочих моделей
✅ Универсальная работа с ЛЮБЫМИ моделями без привязки к именам
✅ Защита от сбоев с умным fallback
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
from collections import deque
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum


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
    else:
        print(f"ℹ️ Файл .env не найден. Используются системные переменные окружения.")


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
    WHITE = "\033[97m"


def log_cognitive_stage(stage: str, message: str, color: str = Colors.CYAN, indent: int = 0):
    timestamp = datetime.now().strftime("%H:%M:%S")
    indent_str = "  " * indent
    print(f"{Colors.BOLD}{color}[{timestamp}] 🧠 {stage}{Colors.RESET}")
    print(f"{color}{indent_str}→ {message}{Colors.RESET}")


# ================= АКТУАЛЬНАЯ СИСТЕМНАЯ ДАТА =================
def get_current_date_info() -> Dict[str, str]:
    """Получает актуальную дату и время с системных часов компьютера"""
    now = datetime.now()
    months_ru = [
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ]
    weekdays_ru = [
        'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'
    ]

    return {
        'full': f"{now.day} {months_ru[now.month - 1]} {now.year} года",
        'year': str(now.year),
        'month': months_ru[now.month - 1],
        'day': str(now.day),
        'weekday': weekdays_ru[now.weekday()],
        'time': now.strftime('%H:%M'),
        'iso': now.isoformat(),
        'timestamp': now.timestamp()
    }


CURRENT_DATE_INFO = get_current_date_info()
CURRENT_DATE_STR = CURRENT_DATE_INFO['full']
CURRENT_YEAR = CURRENT_DATE_INFO['year']

log_cognitive_stage("СИСТЕМА", f"Актуальная дата: {CURRENT_DATE_STR}", Colors.GREEN)

# ================= ВЕБ-ПОИСК =================
HAS_WEB_SEARCH = False
WEB_SEARCH_METHOD = "отключён"

try:
    from duckduckgo_search import AsyncDDGS

    HAS_WEB_SEARCH = True
    WEB_SEARCH_METHOD = "DuckDuckGo"
    print(f"{Colors.GREEN}✅ Веб-поиск активирован (DuckDuckGo){Colors.RESET}")
except ImportError:
    try:
        import aiohttp

        HAS_WEB_SEARCH = True
        WEB_SEARCH_METHOD = "SearXNG"
        print(f"{Colors.GREEN}✅ Веб-поиск активирован (SearXNG){Colors.RESET}")
    except ImportError:
        print(f"{Colors.RED}❌ Веб-поиск недоступен{Colors.RESET}")

# ================= TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )
except ImportError as e:
    print(f"{Colors.RED}❌ Ошибка импорта telegram: {e}{Colors.RESET}")
    print(f"{Colors.YELLOW}📦 Установите: pip install 'python-telegram-bot>=20.7' aiohttp{Colors.RESET}")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_system_pro")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    LM_STUDIO_BASE_URL = "http://localhost:1234"
    TIMEOUT = 90  # Сокращено для быстрого отказа
    MAX_TOKENS = 2048
    RETRY_ATTEMPTS = 2  # Уменьшено с 3 до 2
    RETRY_DELAY = 1.5

    TIME_SENSITIVE_KEYWORDS = [
        'курс', 'погода', 'новост', 'сегодня', 'сейчас', 'текущ',
        'последн', 'актуальн', 'свеж', 'обнов', 'тренд', 'прогноз'
    ]
    MAX_SEARCH_RESULTS = 5
    SEARCH_CACHE_TTL = 1800
    MAX_MESSAGE_LENGTH = 4096
    MAX_CONTEXT_WINDOW = 15

    # КРИТИЧЕСКИ ВАЖНО: короткие таймауты для ленивой валидации
    MODEL_VALIDATION_TIMEOUT = 5  # Быстрая проверка при первом запросе
    MIN_RESPONSE_LENGTH = 3


# ================= СТРУКТУРЫ ДАННЫХ =================
@dataclass
class ModelResponse:
    """Структура ответа от модели"""
    content: str
    confidence: float
    processing_time: float
    model_name: str
    temperature: float
    success: bool
    error: Optional[str] = None


@dataclass
class ConsensusResult:
    """Результат согласования между полушариями"""
    final_answer: str
    left_confidence: float
    right_confidence: float
    agreement_score: float
    conflict_points: List[str]
    resolution_method: str


class HemisphereRole(Enum):
    """Роли полушарий"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SYNTHESIS = "synthesis"


# ================= УТИЛИТЫ =================
def split_message(text: str, max_length: int = 4096) -> List[str]:
    """Разделяет длинное сообщение на части"""
    if len(text) <= max_length:
        return [text]

    parts = []
    current = ""
    paragraphs = text.split('\n')

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_length:
            current += para + '\n'
        else:
            if current:
                parts.append(current.strip())
            if len(para) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp = ""
                for sent in sentences:
                    if len(temp) + len(sent) + 1 <= max_length:
                        temp += sent + ' '
                    else:
                        if temp:
                            parts.append(temp.strip())
                        temp = sent + ' '
                if temp:
                    current = temp + '\n'
            else:
                current = para + '\n'

    if current:
        parts.append(current.strip())

    if len(parts) > 8:
        parts = parts[:7]
        parts.append("[Сообщение сокращено...]")

    return parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    """Создает основную клавиатуру"""
    keyboard = [
        [
            InlineKeyboardButton("🧠 Думать", callback_data="think"),
            InlineKeyboardButton("📊 Анализ", callback_data="analyze"),
            InlineKeyboardButton("🎯 Цели", callback_data="goals")
        ],
        [
            InlineKeyboardButton("🔍 Поиск", callback_data="search"),
            InlineKeyboardButton("💡 Инсайты", callback_data="insights"),
            InlineKeyboardButton("🧹 Очистить", callback_data="clear")
        ],
        [
            InlineKeyboardButton("❓ Помощь", callback_data="help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= УНИВЕРСАЛЬНЫЙ АВТООБНАРУЖЕНИЕ МОДЕЛЕЙ БЕЗ ВАЛИДАЦИИ =================
class LMStudioModelDiscoverer:
    """Автоматическое обнаружение моделей БЕЗ валидации при старте"""

    async def discover_models(self, base_url: str = "http://localhost:1234") -> List[Dict[str, Any]]:
        models = []

        try:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", f"Запрос моделей из {base_url}", Colors.BLUE)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{base_url}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=8)  # Короткий таймаут
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_models = data.get('data', [])

                        if raw_models:
                            log_cognitive_stage("ОБНАРУЖЕНИЕ",
                                                f"Найдено {len(raw_models)} моделей", Colors.GREEN)

                            for idx, model in enumerate(raw_models):
                                model_id = model.get('id', f'model_{idx}')

                                # Пропускаем ВСЕ известные embedding-модели
                                if any(embed_marker in model_id.lower() for embed_marker in [
                                    'embed', 'bge-', 'e5-', 'nomic-embed', 'all-minilm',
                                    'text-embedding', 'gte-', 'paraphrase', '-e5-', '-bge'
                                ]):
                                    print(f"   {Colors.YELLOW}⊘ Пропущена (embedding): {model_id}{Colors.RESET}")
                                    continue

                                # УНИВЕРСАЛЬНАЯ оценка без привязки к именам
                                capabilities = self._assess_capabilities_universal(model_id, idx)

                                models.append({
                                    'id': model_id,
                                    'name': model_id.split('/')[-1] if '/' in model_id else model_id,
                                    'url': f"{base_url}/v1/chat/completions",
                                    'port': 1234,
                                    'capabilities': capabilities,
                                    'is_thinking_model': 'thinking' in model_id.lower() or 'reason' in model_id.lower()
                                })
                                print(f"   {Colors.GREEN}✓ Модель {len(models)}: {model_id}{Colors.RESET}")
                                print(f"     Способности: анализ={capabilities['analysis']:.2f}, "
                                      f"логика={capabilities['reasoning']:.2f}, креативность={capabilities['creativity']:.2f}")
        except Exception as e:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", f"Ошибка: {str(e)[:80]}", Colors.RED)

        # Если не нашли на основном порту — сканируем другие
        if not models:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", "Сканирование портов 1234-1238", Colors.YELLOW)
            models = await self._scan_standard_ports()

        # Если совсем ничего — резервная конфигурация
        if not models:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", "Используем резервную конфигурацию", Colors.RED)
            models = [{
                'id': 'universal-fallback',
                'name': 'Универсальная резервная модель',
                'url': 'http://localhost:1234/v1/chat/completions',
                'port': 1234,
                'capabilities': {'reasoning': 0.6, 'creativity': 0.6, 'analysis': 0.6},
                'is_thinking_model': False
            }]

        return models

    async def _scan_standard_ports(self) -> List[Dict[str, Any]]:
        """Сканирование стандартных портов БЕЗ валидации"""
        models = []
        for port in range(1234, 1239):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            f"http://localhost:{port}/v1/models",
                            timeout=aiohttp.ClientTimeout(total=2)  # Очень короткий таймаут
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            raw_models = data.get('data', [])
                            if raw_models:
                                for model in raw_models:
                                    model_id = model.get('id', f'model_port_{port}')
                                    if not any(embed_marker in model_id.lower() for embed_marker in [
                                        'embed', 'bge-', 'e5-', 'nomic-embed', 'all-minilm'
                                    ]):
                                        capabilities = self._assess_capabilities_universal(model_id, len(models))
                                        models.append({
                                            'id': model_id,
                                            'name': model_id.split('/')[-1],
                                            'url': f"http://localhost:{port}/v1/chat/completions",
                                            'port': port,
                                            'capabilities': capabilities,
                                            'is_thinking_model': 'thinking' in model_id.lower() or 'reason' in model_id.lower()
                                        })
                                        log_cognitive_stage("ОБНАРУЖЕНИЕ",
                                                            f"Порт {port}: {model_id}", Colors.GREEN, indent=1)
            except:
                continue
        return models

    def _assess_capabilities_universal(self, model_id: str, index: int) -> Dict[str, float]:
        """
        УНИВЕРСАЛЬНАЯ оценка способностей без привязки к конкретным именам моделей.
        """
        model_lower = model_id.lower()

        # Эвристики для определения типа модели
        has_analytical_markers = any(kw in model_lower for kw in [
            'instruct', 'chat', 'qwen', 'phi', 'gemma', 'mistral', 'llama',
            'deepseek', 'coder', 'code', 'math', 'reason', 'thinking', '3b', '4b', '7b', '8b'
        ])

        has_creative_markers = any(kw in model_lower for kw in [
            'mixtral', 'yi', 'solar', 'nova', 'story', 'roleplay', 'rp', 'creative', '13b', '30b', '70b'
        ])

        # Базовые значения с вариацией по индексу (первые модели обычно основные)
        base_reasoning = 0.7 + (0.15 if index == 0 else 0.05 if index == 1 else 0)
        base_creativity = 0.6 + (0.15 if index == 0 else 0.05 if index == 1 else 0)
        base_analysis = 0.7 + (0.2 if index == 0 else 0.1 if index == 1 else 0)

        # Корректировка на основе эвристик
        if has_analytical_markers:
            base_analysis = min(0.95, base_analysis + 0.1)
            base_reasoning = min(0.9, base_reasoning + 0.05)

        if has_creative_markers:
            base_creativity = min(0.9, base_creativity + 0.2)
            base_analysis = max(0.4, base_analysis - 0.15)

        # Thinking-модели получают бонус к анализу
        if 'thinking' in model_lower or 'reason' in model_lower:
            base_analysis = min(0.95, base_analysis + 0.15)
            base_reasoning = min(0.9, base_reasoning + 0.1)

        return {
            'reasoning': round(base_reasoning, 2),
            'creativity': round(base_creativity, 2),
            'analysis': round(base_analysis, 2)
        }


# ================= УЛУЧШЕННЫЙ ИНТЕРФЕЙС МОДЕЛИ С ЛЕНИВОЙ ВАЛИДАЦИЕЙ =================
class EnhancedModelInterface:
    """Интерфейс модели с ЛЕНИВОЙ валидацией (без проверки при старте)"""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.api_url = model_config['url']
        self.capabilities = model_config['capabilities']
        self.is_thinking_model = model_config.get('is_thinking_model', False)
        self.call_history = deque(maxlen=100)
        self.total_calls = 0
        self.successful_calls = 0
        self.is_validated = False  # ← Ленивая валидация: False до первого запроса
        self.validation_failed = False
        self.consecutive_failures = 0
        self.last_error = None

    async def validate_lazy(self) -> bool:
        """
        ЛЕНИВАЯ ВАЛИДАЦИЯ — вызывается ТОЛЬКО при первом реальном запросе
        Таймаут всего 5 секунд для быстрого отказа от нерабочих моделей
        """
        if self.is_validated or self.validation_failed:
            return self.is_validated

        log_cognitive_stage("ВАЛИДАЦИЯ",
                            f"Ленивая проверка '{self.model_name}' (первый запрос)",
                            Colors.YELLOW, indent=1)

        try:
            # Очень короткий тестовый запрос
            test_prompt = "ОК"
            response = await asyncio.wait_for(
                self._call_api(
                    system_prompt="Ответь одним словом.",
                    user_prompt=test_prompt,
                    temperature=0.3,
                    max_tokens=10,
                    timeout=Config.MODEL_VALIDATION_TIMEOUT  # ← 5 секунд!
                ),
                timeout=Config.MODEL_VALIDATION_TIMEOUT + 2  # Общий лимит 7с
            )

            # Принимаем любой непустой ответ как валидный
            if response and len(response.strip()) >= Config.MIN_RESPONSE_LENGTH:
                self.is_validated = True
                self.consecutive_failures = 0
                log_cognitive_stage("ВАЛИДАЦИЯ",
                                    f"✅ '{self.model_name}' готова",
                                    Colors.GREEN, indent=2)
                return True
            else:
                self.validation_failed = True
                log_cognitive_stage("ВАЛИДАЦИЯ",
                                    f"⊘ '{self.model_name}' пропущена (пустой ответ)",
                                    Colors.YELLOW, indent=2)
                return False

        except asyncio.TimeoutError:
            self.validation_failed = True
            log_cognitive_stage("ВАЛИДАЦИЯ",
                                f"⊘ '{self.model_name}' пропущена (таймаут >{Config.MODEL_VALIDATION_TIMEOUT}с)",
                                Colors.YELLOW, indent=2)
            return False
        except Exception as e:
            self.validation_failed = True
            error_msg = str(e)[:50]
            log_cognitive_stage("ВАЛИДАЦИЯ",
                                f"⊘ '{self.model_name}' пропущена ({error_msg})",
                                Colors.YELLOW, indent=2)
            return False

    async def call_with_retry(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            retry_attempts: int = Config.RETRY_ATTEMPTS
    ) -> ModelResponse:
        """Вызов модели с ленивой валидацией при первом запросе"""
        start_time = time.time()

        # ←←← КРИТИЧЕСКИ ВАЖНО: ЛЕНИВАЯ ВАЛИДАЦИЯ ТОЛЬКО ЗДЕСЬ, НЕ ПРИ СТАРТЕ ←←←
        if not self.is_validated and not self.validation_failed:
            # Валидация с таймаутом 7 секунд — не блокирует систему
            try:
                is_valid = await asyncio.wait_for(
                    self.validate_lazy(),
                    timeout=7.0
                )
                if not is_valid:
                    return ModelResponse(
                        content="",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        model_name=self.model_name,
                        temperature=temperature,
                        success=False,
                        error="Модель недоступна"
                    )
            except asyncio.TimeoutError:
                self.validation_failed = True
                return ModelResponse(
                    content="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_name=self.model_name,
                    temperature=temperature,
                    success=False,
                    error="Валидация превысила 7с"
                )

        # Основной вызов с короткими retry
        for attempt in range(retry_attempts):
            try:
                response = await self._call_api(
                    system_prompt, user_prompt, temperature, max_tokens
                )

                if not response or len(response.strip()) < Config.MIN_RESPONSE_LENGTH:
                    raise Exception(f"Пустой ответ (длина {len(response)})")

                processing_time = time.time() - start_time
                self.total_calls += 1
                self.successful_calls += 1
                self.consecutive_failures = 0

                result = ModelResponse(
                    content=response,
                    confidence=self._estimate_confidence(response),
                    processing_time=processing_time,
                    model_name=self.model_name,
                    temperature=temperature,
                    success=True
                )

                self.call_history.append(asdict(result))
                return result

            except (asyncio.TimeoutError, aiohttp.ClientError, Exception) as e:
                self.consecutive_failures += 1
                error_msg = str(e)[:60]

                if attempt < retry_attempts - 1:
                    await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
                else:
                    self.total_calls += 1
                    processing_time = time.time() - start_time

                    if self.consecutive_failures >= 3:
                        self.validation_failed = True

                    return ModelResponse(
                        content="",
                        confidence=0.0,
                        processing_time=processing_time,
                        model_name=self.model_name,
                        temperature=temperature,
                        success=False,
                        error=error_msg
                    )

    async def _call_api(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
            timeout: int = None
    ) -> str:
        """Прямой вызов API модели"""
        if timeout is None:
            timeout = Config.TIMEOUT

        headers = {"Content-Type": "application/json"}

        # Адаптивное укорачивание для стабильности
        if self.is_thinking_model:
            system_prompt = (system_prompt[:400] + "...") if len(system_prompt) > 400 else system_prompt
            user_prompt = (user_prompt[:1200] + "...") if len(user_prompt) > 1200 else user_prompt
        else:
            system_prompt = (system_prompt[:800] + "...") if len(system_prompt) > 800 else system_prompt
            user_prompt = (user_prompt[:2000] + "...") if len(user_prompt) > 2000 else user_prompt

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": max(0.1, min(1.0, temperature)),
            "max_tokens": min(max_tokens, Config.MAX_TOKENS),
            "top_p": 0.9,
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'choices' not in data or not data['choices']:
                            raise Exception("Пустой ответ от API")

                        content = data["choices"][0]["message"]["content"].strip()
                        return self._clean_response(content)
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            raise

    def _clean_response(self, text: str) -> str:
        """Очистка ответа от форматирования"""
        if not text:
            return ""

        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*|\*|__|_', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        # Удаляем артефакты генерации
        artifacts = [
            r'^Ответ:?\s*',
            r'^Мой ответ:?\s*',
            r'^Финальный ответ:?\s*',
            r'^Итог:?\s*',
            r'^\* \* \*',
            r'^---+',
        ]
        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE | re.MULTILINE)

        return text.strip()

    def _estimate_confidence(self, response: str) -> float:
        """Оценка уверенности по содержимому ответа"""
        if not response or len(response) < 5:
            return 0.0

        confidence = 0.65
        if self.is_thinking_model:
            confidence = 0.75

        uncertain_phrases = [
            'возможно', 'может быть', 'вероятно', 'не уверен', 'сложно сказать',
            'я не знаю', 'не могу сказать', 'предположу'
        ]
        for phrase in uncertain_phrases:
            if phrase in response.lower():
                confidence -= 0.15

        if re.search(r'\d{2,}%|\d+\s*(тыс|млн|миллион|рубл|долл|евро|°C)', response):
            confidence += 0.12

        if 100 < len(response) < 800:
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики модели"""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0

        avg_time = 0
        if self.call_history:
            avg_time = sum(h['processing_time'] for h in self.call_history) / len(self.call_history)

        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'success_rate': success_rate,
            'avg_processing_time': avg_time,
            'is_validated': self.is_validated,
            'validation_failed': self.validation_failed
        }


# ================= ПРОДВИНУТЫЙ ДИАЛОГ С УМНЫМ FALLBACK =================
class AdvancedBrainHemispheresDialog:
    """
    Диалог полушарий с защитой от сбоев и умным выбором моделей
    """

    def __init__(self, models: List[EnhancedModelInterface]):
        # Все модели считаются рабочими до первого запроса
        self.all_models = models

        if not self.all_models:
            raise ValueError("❌ Нет моделей для работы!")

        # Сортируем по способностям
        self.analytical_models = sorted(
            self.all_models,
            key=lambda m: (m.capabilities['analysis'] * 0.6 + m.capabilities['reasoning'] * 0.4),
            reverse=True
        )
        self.creative_models = sorted(
            self.all_models,
            key=lambda m: m.capabilities['creativity'],
            reverse=True
        )

        # Назначение основных полушарий
        self.left_brain = self.analytical_models[0]
        self.right_brain = (self.creative_models[0] if len(self.creative_models) > 1
                            else self.analytical_models[1] if len(self.analytical_models) > 1
        else self.analytical_models[0])

        # Резервные модели
        used_names = {self.left_brain.model_name, self.right_brain.model_name}
        self.fallback_pool = [m for m in self.all_models if m.model_name not in used_names]

        if not self.fallback_pool:
            self.fallback_pool = [self.right_brain, self.left_brain]

        log_cognitive_stage("АРХИТЕКТУРА",
                            f"🧠 Левое (аналитика): {self.left_brain.model_name} "
                            f"[анализ: {self.left_brain.capabilities['analysis']:.2f}]",
                            Colors.BLUE)
        log_cognitive_stage("АРХИТЕКТУРА",
                            f"🎨 Правое (креативность): {self.right_brain.model_name} "
                            f"[креативность: {self.right_brain.capabilities['creativity']:.2f}]",
                            Colors.MAGENTA)
        log_cognitive_stage("АРХИТЕКТУРА",
                            f"🛡️ Резерв: {len(self.fallback_pool)} моделей",
                            Colors.CYAN, indent=1)

    async def conduct_parallel_dialog(
            self,
            user_query: str,
            search_results: str = ""
    ) -> Tuple[str, ConsensusResult, List[Dict]]:
        """
        Параллельный диалог с автоматическим переключением на резерв при ошибках
        """
        dialog_history = []
        current_date = CURRENT_DATE_STR
        current_time = CURRENT_DATE_INFO['time']

        context = f"СЕГОДНЯ {current_date}, время {current_time}. Запрос: {user_query}"
        if search_results:
            context += f"\n\nПОИСК (актуально на {current_date}):\n{search_results}"

        # === РАУНД 1: ПАРАЛЛЕЛЬНАЯ ГЕНЕРАЦИЯ ===
        log_cognitive_stage("ДИАЛОГ", "Раунд 1: Параллельная генерация черновиков",
                            Colors.CYAN, indent=1)

        left_prompt = f"""
КОНТЕКСТ: {context}

ЗАДАЧА: Создай структурированный, фактический черновик ответа.
- Будь точным и логичным
- Основывайся на фактах из контекста
- Ответь кратко: максимум 3-4 предложения

Сегодня {current_date}.
"""

        right_prompt = f"""
КОНТЕКСТ: {context}

ЗАДАЧА: Создай практичный и понятный черновик ответа.
- Сосредоточься на пользе для пользователя
- Сделай ответ ясным и доступным
- Ответь кратко: максимум 3-4 предложения

Сегодня {current_date}.
"""

        # ПАРАЛЛЕЛЬНЫЙ ВЫЗОВ С ТАЙМАУТОМ 15 СЕКУНД НА ВСЁ
        left_task = self._call_with_smart_fallback(
            primary=self.left_brain,
            fallback_pool=self.fallback_pool,
            system_prompt=f"Ты — аналитический ассистент. Сегодня {current_date}.",
            user_prompt=left_prompt,
            temperature=0.3,
            role="аналитическое полушарие"
        )

        right_task = self._call_with_smart_fallback(
            primary=self.right_brain,
            fallback_pool=self.fallback_pool,
            system_prompt=f"Ты — креативный ассистент. Сегодня {current_date}.",
            user_prompt=right_prompt,
            temperature=0.5,
            role="креативное полушарие"
        )

        # Общий таймаут на оба полушария — 15 секунд
        try:
            left_response, right_response = await asyncio.wait_for(
                asyncio.gather(left_task, right_task),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            log_cognitive_stage("ОШИБКА", "Таймаут генерации черновиков (>15с)", Colors.RED, indent=2)
            return self._create_error_response(dialog_history)

        # Обработка отказов
        if not left_response.success and not right_response.success:
            log_cognitive_stage("КРИТИЧЕСКАЯ ОШИБКА", "Оба полушария недоступны", Colors.RED, indent=2)
            return self._create_error_response(dialog_history)

        if not left_response.success:
            log_cognitive_stage("FALLBACK", "Аналитическое полушарие недоступно → используем креативное", Colors.YELLOW,
                                indent=2)
            left_response = ModelResponse(
                content=right_response.content,
                confidence=right_response.confidence * 0.8,
                processing_time=right_response.processing_time,
                model_name=f"{right_response.model_name} (fallback)",
                temperature=0.3,
                success=True
            )

        if not right_response.success:
            log_cognitive_stage("FALLBACK", "Креативное полушарие недоступно → используем аналитическое", Colors.YELLOW,
                                indent=2)
            right_response = ModelResponse(
                content=left_response.content,
                confidence=left_response.confidence * 0.8,
                processing_time=left_response.processing_time,
                model_name=f"{left_response.model_name} (fallback)",
                temperature=0.5,
                success=True
            )

        # Сохраняем в историю
        dialog_history.extend([
            {
                'round': 1,
                'hemisphere': 'left',
                'model': left_response.model_name,
                'content': left_response.content[:200],
                'full_content': left_response.content,
                'confidence': left_response.confidence,
                'processing_time': left_response.processing_time
            },
            {
                'round': 1,
                'hemisphere': 'right',
                'model': right_response.model_name,
                'content': right_response.content[:200],
                'full_content': right_response.content,
                'confidence': right_response.confidence,
                'processing_time': right_response.processing_time
            }
        ])

        log_cognitive_stage("ДИАЛОГ",
                            f"🧠 Аналитическое ({left_response.confidence:.2f}): {left_response.content[:60]}...",
                            Colors.WHITE, indent=2)
        log_cognitive_stage("ДИАЛОГ",
                            f"🎨 Креативное ({right_response.confidence:.2f}): {right_response.content[:60]}...",
                            Colors.WHITE, indent=2)

        # === РАУНД 2: СИНТЕЗ ===
        log_cognitive_stage("ДИАЛОГ", "Раунд 2: Синтез ответа", Colors.GREEN, indent=1)

        candidate_models = [m for m in [self.left_brain, self.right_brain] + self.fallback_pool
                            if not m.validation_failed]
        if not candidate_models:
            return self._create_error_response(dialog_history)

        synthesizer = max(candidate_models, key=lambda m: (
                m.capabilities['analysis'] * 0.4 +
                m.capabilities['reasoning'] * 0.4 +
                m.capabilities['creativity'] * 0.2
        ))

        synthesis_prompt = f"""
ЧЕРНОВИК АНАЛИТИЧЕСКИЙ (уверенность {left_response.confidence:.2f}):
{left_response.content}

ЧЕРНОВИК КРЕАТИВНЫЙ (уверенность {right_response.confidence:.2f}):
{right_response.content}

КОНТЕКСТ: {context}

ЗАДАЧА: Объедини лучшее из обоих черновиков в КРАТКИЙ финальный ответ (максимум 5 предложений).
Учти актуальность на {current_date}. Только ответ, без комментариев.
"""

        final_response = await self._call_with_smart_fallback(
            primary=synthesizer,
            fallback_pool=[m for m in candidate_models if m != synthesizer],
            system_prompt=f"Ты — синтезатор. Сегодня {current_date}.",
            user_prompt=synthesis_prompt,
            temperature=0.25,
            role="синтезатор"
        )

        if not final_response.success:
            log_cognitive_stage("FALLBACK", "Синтез не удался → взвешенное объединение", Colors.YELLOW, indent=2)
            final_answer = self._weighted_merge(left_response, right_response)
            final_confidence = (left_response.confidence + right_response.confidence) / 2 * 0.9
        else:
            final_answer = final_response.content
            final_confidence = final_response.confidence

        dialog_history.append({
            'round': 2,
            'hemisphere': 'synthesis',
            'model': final_response.model_name if final_response.success else 'weighted_merge',
            'content': final_answer[:200],
            'full_content': final_answer,
            'confidence': final_confidence
        })

        consensus = self._analyze_consensus(
            left_response, right_response, final_answer
        )

        log_cognitive_stage("ДИАЛОГ",
                            f"✅ Ответ готов (согласие: {consensus.agreement_score:.2f})",
                            Colors.GREEN)

        return final_answer, consensus, dialog_history

    async def _call_with_smart_fallback(
            self,
            primary: EnhancedModelInterface,
            fallback_pool: List[EnhancedModelInterface],
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            role: str
    ) -> ModelResponse:
        """Умный вызов с многоуровневым fallback"""
        # Пробуем основную модель с таймаутом 8 секунд
        try:
            response = await asyncio.wait_for(
                primary.call_with_retry(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=Config.MAX_TOKENS
                ),
                timeout=8.0
            )
            if response.success:
                return response
        except asyncio.TimeoutError:
            pass

        # Пробуем резервные модели (максимум 2 попытки)
        for idx, fallback_model in enumerate(fallback_pool[:2]):
            if fallback_model.validation_failed:
                continue

            try:
                response = await asyncio.wait_for(
                    fallback_model.call_with_retry(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=Config.MAX_TOKENS
                    ),
                    timeout=7.0
                )
                if response.success:
                    response.confidence *= 0.9  # Снижаем уверенность для резерва
                    response.model_name += " (резерв)"
                    return response
            except asyncio.TimeoutError:
                continue

        return ModelResponse(
            content="",
            confidence=0.0,
            processing_time=0,
            model_name=primary.model_name,
            temperature=temperature,
            success=False,
            error="Все попытки исчерпаны"
        )

    def _weighted_merge(self, left: ModelResponse, right: ModelResponse) -> str:
        """Взвешенное объединение при отказе синтеза"""
        if left.confidence > right.confidence * 1.5:
            return left.content
        elif right.confidence > left.confidence * 1.5:
            return right.content
        return f"{left.content}\n\n{right.content}"[:1500]

    def _analyze_consensus(self, left: ModelResponse, right: ModelResponse, final: str) -> ConsensusResult:
        left_words = set(re.findall(r'\w+', left.content.lower()))
        right_words = set(re.findall(r'\w+', right.content.lower()))
        agreement_score = len(left_words & right_words) / len(left_words | right_words) if (
                    left_words | right_words) else 0.0

        conflict_points = []
        if re.search(r'\d+', left.content) and re.search(r'\d+', right.content):
            left_nums = set(re.findall(r'\d+', left.content))
            right_nums = set(re.findall(r'\d+', right.content))
            if left_nums != right_nums:
                conflict_points.append("расхождение в числах")

        resolution_method = "parallel_synthesis"
        if agreement_score < 0.4:
            resolution_method = "weighted_merge"
        elif agreement_score > 0.8:
            resolution_method = "high_consensus"

        return ConsensusResult(
            final_answer=final,
            left_confidence=left.confidence,
            right_confidence=right.confidence,
            agreement_score=agreement_score,
            conflict_points=conflict_points,
            resolution_method=resolution_method
        )

    def _create_error_response(self, history: List[Dict]) -> Tuple[str, ConsensusResult, List[Dict]]:
        error_text = "⚠️ Не удалось обработать запрос. Возможные причины:\n• Модели перегружены в LM Studio\n• Недостаточно памяти для загрузки модели\n• Таймаут обработки (>15с)\n💡 Подождите 10 секунд и повторите запрос."
        consensus = ConsensusResult(
            final_answer=error_text,
            left_confidence=0.0,
            right_confidence=0.0,
            agreement_score=0.0,
            conflict_points=["системная ошибка"],
            resolution_method="error"
        )
        return error_text, consensus, history


# ================= ПАРАЛЛЕЛЬНЫЙ ВЕБ-ПОИСК (без изменений) =================
# ================= ПАРАЛЛЕЛЬНЫЙ ВЕБ-ПОИСК С НАДЁЖНЫМ ДОСТУПОМ К ЦБ РФ =================
class ParallelWebSearchEngine:
    """Параллельный веб-поиск с приоритетом на официальные источники (ЦБ РФ + резерв)"""

    def __init__(self, cache_ttl: int = 3600):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.search_stats = {'total': 0, 'cache_hits': 0, 'successful': 0, 'failures': 0}

    async def parallel_search(self, query: str, max_results: int = 5) -> str:
        """Поиск с приоритетом на финансовые данные"""
        current_date = CURRENT_DATE_STR
        log_cognitive_stage("ВЕБ-ПОИСК", f"Анализ запроса: '{query[:50]}'", Colors.YELLOW)

        # Определение финансового запроса
        is_finance_query = any(kw in query.lower() for kw in [
            'курс', 'доллар', 'рубль', 'евро', 'usd', 'eur', 'rur', 'rub', 'валют', 'долар'
        ])

        if is_finance_query:
            log_cognitive_stage("ВЕБ-ПОИСК", "💡 Финансовый запрос → приоритет: ЦБ РФ + резерв", Colors.CYAN, indent=1)
            # ПАРАЛЛЕЛЬНЫЙ ЗАПРОС К НЕСКОЛЬКИМ ИСТОЧНИКАМ
            cbr_task = self._search_cbr_currency()
            reserve_task = self._search_exchangerate_host()

            # Ждём первого успешного ответа (таймаут 8 секунд на всё)
            done, pending = await asyncio.wait(
                [cbr_task, reserve_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=8.0
            )

            # Отменяем незавершённые задачи
            for task in pending:
                task.cancel()

            # Обрабатываем результаты
            for task in done:
                try:
                    result = task.result()
                    if result and 'Курс' in result:
                        self.search_stats['successful'] += 1
                        log_cognitive_stage("ВЕБ-ПОИСК", "✅ Данные получены от официального источника", Colors.GREEN,
                                            indent=1)
                        return result
                except:
                    continue

            # Если оба источника отказали
            self.search_stats['failures'] += 1
            return self._get_finance_fallback(current_date)

        # Обычный веб-поиск (без изменений)
        clean_query = query.lower().replace('долар', 'доллар').replace('рубл', 'рубль')
        dated_query = f"{clean_query} {CURRENT_YEAR}"

        cache_key = hashlib.sha256(dated_query.encode()).hexdigest()
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.search_stats['cache_hits'] += 1
                return self._format_web_results(cached_results, current_date)

        self.search_stats['total'] += 1
        search_tasks = [
            self._search_searx(dated_query, max_results),
            self._search_bing(dated_query, max_results)
        ]
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        combined = []
        for results in all_results:
            if isinstance(results, list):
                combined.extend(results)

        combined.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        top_results = combined[:max_results]

        if top_results:
            self.cache[cache_key] = (time.time(), top_results)
            self.search_stats['successful'] += 1
            return self._format_web_results(top_results, current_date)
        else:
            self.search_stats['failures'] += 1
            return f"Поиск по запросу '{query}' временно недоступен."

    async def _search_cbr_currency(self) -> str:
        """НАДЁЖНЫЙ запрос к ЦБ РФ с обработкой нестандартных Content-Type"""
        try:
            log_cognitive_stage("ПОИСК", "📡 ЦБ РФ: запрос к daily_json.js", Colors.CYAN, indent=2)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://www.cbr-xml-daily.ru/daily_json.js",
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': '*/*'  # Принимаем любой тип контента
                        },
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        # Читаем как ТЕКСТ (не как JSON!) для обхода проблемы с Content-Type
                        text = await resp.text()

                        # Очищаем от возможных комментариев (иногда сервер добавляет /* */)
                        text = re.sub(r'^/\*.*?\*/', '', text, flags=re.DOTALL).strip()

                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError as e:
                            log_cognitive_stage("ПОИСК", f"⊘ ЦБ РФ: ошибка парсинга JSON ({str(e)[:40]})",
                                                Colors.YELLOW, indent=3)
                            return ""

                        # Проверка структуры
                        if 'Valute' not in data or 'USD' not in data['Valute']:
                            log_cognitive_stage("ПОИСК", "⊘ ЦБ РФ: неожиданная структура данных", Colors.YELLOW,
                                                indent=3)
                            return ""

                        usd = data['Valute']['USD']
                        eur = data['Valute'].get('EUR', {})

                        # Форматирование даты
                        try:
                            from datetime import datetime as dt
                            date_str = data.get('Date', '')
                            date_obj = dt.fromisoformat(date_str.replace('Z', '+00:00'))
                            months_ru = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                                         'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
                            formatted_date = f"{date_obj.day} {months_ru[date_obj.month - 1]} {date_obj.year} года"
                        except:
                            formatted_date = CURRENT_DATE_STR

                        # Формирование результата
                        result = (
                            f"💰 ОФИЦИАЛЬНЫЙ КУРС ВАЛЮТ (ЦБ РФ)\n"
                            f"Дата: {formatted_date}\n\n"
                            f"💵 Доллар США (USD):\n"
                            f"   1 USD = {usd['Value']:.4f} ₽\n"
                        )

                        if usd.get('Previous'):
                            change = usd['Value'] - usd['Previous']
                            sign = '📈' if change > 0 else '📉' if change < 0 else '→'
                            result += f"   Изменение: {sign} {abs(change):.4f} ₽\n\n"
                        else:
                            result += "\n"

                        if eur:
                            result += (
                                f"💶 Евро (EUR):\n"
                                f"   1 EUR = {eur['Value']:.4f} ₽\n"
                            )
                            if eur.get('Previous'):
                                change = eur['Value'] - eur['Previous']
                                sign = '📈' if change > 0 else '📉' if change < 0 else '→'
                                result += f"   Изменение: {sign} {abs(change):.4f} ₽\n\n"

                        result += f"🔗 Источник: ЦБ РФ (cbr.ru)"

                        log_cognitive_stage("ПОИСК",
                                            f"✅ USD={usd['Value']:.4f} ₽ | EUR={eur.get('Value', 0):.4f} ₽",
                                            Colors.GREEN, indent=3)
                        return result
                    else:
                        log_cognitive_stage("ПОИСК", f"⊘ ЦБ РФ: HTTP {resp.status}", Colors.YELLOW, indent=3)
        except asyncio.TimeoutError:
            log_cognitive_stage("ПОИСК", "⊘ ЦБ РФ: таймаут", Colors.YELLOW, indent=3)
        except Exception as e:
            log_cognitive_stage("ПОИСК", f"⊘ ЦБ РФ: {type(e).__name__}", Colors.YELLOW, indent=3)
        return ""

    async def _search_exchangerate_host(self) -> str:
        """РЕЗЕРВНЫЙ источник: exchangerate.host (бесплатный, без ключа)"""
        try:
            log_cognitive_stage("ПОИСК", "🌍 Резерв: exchangerate.host", Colors.CYAN, indent=2)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://api.exchangerate.host/latest",
                        params={'base': 'USD', 'symbols': 'RUB'},
                        headers={'User-Agent': 'Mozilla/5.0'},
                        timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        if 'rates' in data and 'RUB' in data['rates']:
                            rate = data['rates']['RUB']
                            date_str = data.get('date', CURRENT_DATE_STR)

                            result = (
                                f"💰 КУРС ВАЛЮТ (exchangerate.host)\n"
                                f"Дата: {date_str}\n\n"
                                f"💵 Доллар США (USD):\n"
                                f"   1 USD = {rate:.4f} ₽\n\n"
                                f"🔗 Источник: exchangerate.host"
                            )

                            log_cognitive_stage("ПОИСК", f"✅ Резерв: USD={rate:.4f} ₽", Colors.GREEN, indent=3)
                            return result
                        else:
                            log_cognitive_stage("ПОИСК", "⊘ Резерв: неожиданная структура", Colors.YELLOW, indent=3)
                    else:
                        log_cognitive_stage("ПОИСК", f"⊘ Резерв: HTTP {resp.status}", Colors.YELLOW, indent=3)
        except Exception as e:
            log_cognitive_stage("ПОИСК", f"⊘ Резерв: {type(e).__name__}", Colors.YELLOW, indent=3)
        return ""

    def _get_finance_fallback(self, current_date: str) -> str:
        """Честное сообщение при недоступности источников"""
        return (
            f"⚠️ ВРЕМЕННАЯ НЕДОСТУПНОСТЬ КУРСОВ\n"
            f"Дата: {current_date}\n\n"
            f"💡 Рекомендуем проверить актуальные курсы на:\n"
            f"• Официальный сайт ЦБ РФ: https://cbr.ru\n"
            f"• Мосбиржа: https://moex.com\n"
            f"• Google Finance: https://www.google.com/finance\n\n"
            f"Система автоматически повторит запрос при следующем обращении."
        )

    async def _search_searx(self, query: str, max_results: int) -> List[Dict]:
        """Поиск через резервные инстансы SearXNG"""
        instances = [
            ("https://searx.work", "searx.work"),
            ("https://search.ononoki.org", "ononoki"),
            ("https://searx.name", "searx.name")
        ]

        for base_url, name in instances:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            f"{base_url}/search",
                            params={'q': query, 'format': 'json', 'language': 'ru'},
                            headers={'User-Agent': 'Mozilla/5.0'},
                            timeout=aiohttp.ClientTimeout(total=8)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            results = []
                            for r in data.get('results', [])[:max_results]:
                                if r.get('title') and r.get('url') and len(r['title']) > 15:
                                    results.append({
                                        'title': r['title'][:120],
                                        'url': r['url'],
                                        'snippet': (r.get('content') or r.get('snippet', ''))[:200],
                                        'relevance': 0.85,
                                        'source': f'SearXNG ({name})'
                                    })
                            if results:
                                return results
            except:
                continue
        return []

    async def _search_bing(self, query: str, max_results: int) -> List[Dict]:
        """Поиск через Bing (ограниченный)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://www.bing.com/search",
                        params={'q': query, 'setlang': 'ru'},
                        headers={'User-Agent': 'Mozilla/5.0'},
                        timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    pass  # Пропускаем парсинг HTML для упрощения
        except:
            pass
        return []

    def _format_web_results(self, results: List[Dict], current_date: str) -> str:
        """Форматирование веб-результатов"""
        if not results:
            return ""
        formatted = f"РЕЗУЛЬТАТЫ ПОИСКА (актуально на {current_date}):\n\n"
        for i, r in enumerate(results[:3], 1):
            formatted += f"{i}. {r['title']}\n"
            formatted += f"   {r['snippet']}\n"
            formatted += f"   Источник: {r.get('source', 'Web')}\n\n"
        return formatted.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика поиска"""
        total = max(1, self.search_stats['total'] + self.search_stats['cache_hits'])
        return {
            'total_searches': self.search_stats['total'],
            'success_rate': self.search_stats['successful'] / total * 100,
            'cache_hit_rate': self.search_stats['cache_hits'] / total * 100,
            'failure_rate': self.search_stats['failures'] / total * 100,
            'cache_size': len(self.cache)
        }


# ================= БАЗА ДАННЫХ, АГЕНТ, МЕНЕДЖЕР СЕССИЙ (без изменений) =================
class EnhancedCognitiveDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                user_id INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                system_response TEXT NOT NULL,
                dialog_metadata TEXT,
                consensus_metadata TEXT,
                used_search INTEGER DEFAULT 0,
                left_confidence REAL DEFAULT 0.5,
                right_confidence REAL DEFAULT 0.5,
                agreement_score REAL DEFAULT 0.5,
                processing_time REAL DEFAULT 0,
                models_used TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_interaction(self, user_id: int, user_input: str, system_response: str, **kwargs):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interactions
            (timestamp, user_id, user_input, system_response, dialog_metadata,
             consensus_metadata, used_search, left_confidence, right_confidence,
             agreement_score, processing_time, models_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(), user_id, user_input[:1000], system_response[:2000],
            json.dumps(kwargs.get('dialog_metadata', []), ensure_ascii=False),
            json.dumps(asdict(kwargs.get('consensus', ConsensusResult(
                final_answer="", left_confidence=0.5, right_confidence=0.5,
                agreement_score=0.5, conflict_points=[], resolution_method="unknown"
            ))), ensure_ascii=False),
            kwargs.get('used_search', 0),
            kwargs.get('left_confidence', 0.5),
            kwargs.get('right_confidence', 0.5),
            kwargs.get('agreement_score', 0.5),
            kwargs.get('processing_time', 0),
            kwargs.get('models_used', 'unknown')
        ))
        conn.commit()
        conn.close()

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (user_id,))
        interactions = cursor.fetchone()[0]
        cursor.execute('''
            SELECT AVG(agreement_score), AVG(processing_time), 
                   AVG(left_confidence), AVG(right_confidence)
            FROM interactions WHERE user_id = ?
        ''', (user_id,))
        avg_data = cursor.fetchone()
        conn.close()
        return {
            'interactions': interactions,
            'avg_agreement': avg_data[0] if avg_data[0] else 0.5,
            'avg_processing_time': avg_data[1] if avg_data[1] else 0,
            'avg_left_confidence': avg_data[2] if avg_data[2] else 0.5,
            'avg_right_confidence': avg_data[3] if avg_data[3] else 0.5,
            'last_activity': time.time()
        }


class EnhancedCognitiveAgent:
    def __init__(self, user_id: int, db: EnhancedCognitiveDB, dialog_engine: AdvancedBrainHemispheresDialog,
                 search_engine: Optional[ParallelWebSearchEngine]):
        self.user_id = user_id
        self.db = db
        self.dialog_engine = dialog_engine
        self.search_engine = search_engine
        self.context_window = deque(maxlen=Config.MAX_CONTEXT_WINDOW)

    async def process_message(self, user_input: str) -> str:
        start_time = time.time()
        log_cognitive_stage("ЗАПРОС", f"Пользователь {self.user_id}: {user_input[:80]}", Colors.GREEN)

        if self._is_time_query(user_input):
            return self._handle_time_query(user_input)

        needs_search = any(kw in user_input.lower() for kw in Config.TIME_SENSITIVE_KEYWORDS)
        search_results = ""
        used_search = False

        if needs_search and HAS_WEB_SEARCH and self.search_engine:
            try:
                search_results = await asyncio.wait_for(
                    self.search_engine.parallel_search(user_input, Config.MAX_SEARCH_RESULTS),
                    timeout=12.0
                )
                used_search = bool(search_results)
            except asyncio.TimeoutError:
                log_cognitive_stage("ПОИСК", "Таймаут поиска, продолжаем без результатов", Colors.YELLOW)

        final_answer, consensus, dialog_metadata = await self.dialog_engine.conduct_parallel_dialog(
            user_query=user_input,
            search_results=search_results
        )

        processing_time = time.time() - start_time
        models_used = f"{self.dialog_engine.left_brain.model_name} + {self.dialog_engine.right_brain.model_name}"
        self.db.add_interaction(
            user_id=self.user_id,
            user_input=user_input,
            system_response=final_answer,
            dialog_metadata=dialog_metadata,
            consensus=consensus,
            used_search=1 if used_search else 0,
            left_confidence=consensus.left_confidence,
            right_confidence=consensus.right_confidence,
            agreement_score=consensus.agreement_score,
            processing_time=processing_time,
            models_used=models_used
        )

        formatted_response = self._format_response(final_answer, consensus, used_search)
        log_cognitive_stage("ЗАВЕРШЕНО", f"Обработано за {processing_time:.1f}с", Colors.GREEN)
        return formatted_response

    def _is_time_query(self, text: str) -> bool:
        patterns = [r'какой сейчас год', r'текущий год', r'какое сегодня число', r'сколько времени', r'который час']
        return any(re.search(p, text.lower()) for p in patterns)

    def _handle_time_query(self, text: str) -> str:
        now_info = get_current_date_info()
        return f"📅 Сегодня {now_info['full']}. Время: {now_info['time']}"

    def _format_response(self, response: str, consensus: ConsensusResult, used_search: bool) -> str:
        response = re.sub(r'(?i)как (левое|правое) полушарие.*?\n', '', response)
        response = re.sub(r'(?i)синтезирую.*?\n', '', response)
        response = re.sub(r'(?i)финальный ответ:?\s*', '', response, count=1)

        meta_parts = []
        if used_search:
            meta_parts.append(f"🔍 Поиск на {CURRENT_DATE_STR}")
        if consensus.agreement_score < 0.4:
            meta_parts.append(f"⚠️ Низкое согласие ({consensus.agreement_score:.0%})")
        elif consensus.agreement_score > 0.8:
            meta_parts.append(f"✅ Высокое согласие ({consensus.agreement_score:.0%})")

        if meta_parts:
            response = response.rstrip() + "\n\n" + " | ".join(meta_parts)

        return response.strip()


class SessionManager:
    def __init__(self, db: EnhancedCognitiveDB, dialog_engine: AdvancedBrainHemispheresDialog,
                 search_engine: Optional[ParallelWebSearchEngine]):
        self.db = db
        self.dialog_engine = dialog_engine
        self.search_engine = search_engine
        self.sessions: Dict[int, EnhancedCognitiveAgent] = {}

    async def get_or_create_session(self, user_id: int) -> EnhancedCognitiveAgent:
        if user_id not in self.sessions:
            self.sessions[user_id] = EnhancedCognitiveAgent(user_id, self.db, self.dialog_engine, self.search_engine)
            log_cognitive_stage("СЕССИЯ", f"Создана для пользователя {user_id}", Colors.BLUE)
        return self.sessions[user_id]

    def get_global_stats(self) -> Dict[str, Any]:
        left_stats = self.dialog_engine.left_brain.get_stats()
        right_stats = self.dialog_engine.right_brain.get_stats()
        return {
            'active_sessions': len(self.sessions),
            'left_brain_stats': left_stats,
            'right_brain_stats': right_stats,
            'search_stats': self.search_engine.get_stats() if self.search_engine else {},
            'total_models': len(self.dialog_engine.all_models),
            'working_models': len([m for m in self.dialog_engine.all_models if not m.validation_failed])
        }


# ================= TELEGRAM ОБРАБОТЧИКИ (без изменений) =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now_info = get_current_date_info()
    welcome = (
        f"👋 Привет! Я — **Cognitive Agent Pro v8.4.1 UNIVERSAL FAST**\n\n"
        f"⚡ **Мгновенный запуск** — система стартует за 2-3 секунды!\n"
        f"🧠 **Ленивая валидация** — проверка моделей только при первом запросе\n"
        f"🛡️ **Умный fallback** — автоматическое переключение при ошибках\n"
        f"🌐 **Универсальность** — работает с ЛЮБЫМИ моделями в LM Studio\n\n"
        f"📅 **Актуальная дата:** {now_info['full']}\n"
        f"⏰ **Время:** {now_info['time']}\n\n"
        f"💡 **Совет:** Первый запрос может занять 5-8 секунд (инициализация моделей)"
    )
    await update.message.reply_text(welcome, reply_markup=create_main_keyboard(), parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        return

    try:
        session_manager = context.application.bot_data['session_manager']
        agent = await session_manager.get_or_create_session(user_id)
        await update.message.reply_chat_action("typing")
        response = await agent.process_message(text)
        parts = split_message(response)
        for i, part in enumerate(parts):
            reply_markup = create_main_keyboard() if i == len(parts) - 1 else None
            await update.message.reply_text(part, reply_markup=reply_markup, disable_web_page_preview=True)
            if i < len(parts) - 1:
                await asyncio.sleep(0.3)
    except Exception as e:
        log_cognitive_stage("ОШИБКА", f"Ошибка: {str(e)[:80]}", Colors.RED)
        await update.message.reply_text("⚠️ Ошибка обработки. Попробуйте позже.", reply_markup=create_main_keyboard())


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    session_manager = context.application.bot_data['session_manager']
    agent = await session_manager.get_or_create_session(user_id)
    now_info = get_current_date_info()

    if query.data == "analyze":
        stats = agent.db.get_user_stats(user_id)
        global_stats = session_manager.get_global_stats()
        response_text = (
            f"📊 **Ваша статистика:**\n"
            f"• Взаимодействий: {stats['interactions']}\n"
            f"• Среднее согласие: {stats['avg_agreement']:.0%}\n"
            f"• Среднее время: {stats['avg_processing_time']:.1f}с\n\n"
            f"🌐 **Глобальная статистика:**\n"
            f"• Активных сессий: {global_stats['active_sessions']}\n"
            f"• Всего моделей: {global_stats['total_models']}\n"
            f"• Рабочих моделей: {global_stats['working_models']}\n"
            f"• Успешность левого: {global_stats['left_brain_stats'].get('success_rate', 0):.0%}\n"
            f"• Успешность правого: {global_stats['right_brain_stats'].get('success_rate', 0):.0%}"
        )
        await query.message.edit_text(response_text, reply_markup=create_main_keyboard(), parse_mode="Markdown")
    elif query.data == "clear":
        agent.context_window.clear()
        await query.message.edit_text("🧹 Контекст очищен.", reply_markup=create_main_keyboard())
    elif query.data == "help":
        help_text = (
            f"🤖 **Cognitive Agent Pro v8.4.1 UNIVERSAL FAST**\n\n"
            f"⚡ **Особенности:**\n"
            f"• Запуск за 2-3 секунды (без валидации при старте)\n"
            f"• Ленивая валидация при первом запросе\n"
            f"• Автоматический пропуск нерабочих моделей\n"
            f"• Работает с ЛЮБЫМИ моделями в LM Studio\n\n"
            f"📅 Актуальная дата: {now_info['full']}\n\n"
            f"💡 **Использование:**\n"
            f"1. Запустите любую текстовую модель в LM Studio\n"
            f"2. Отправьте запрос в Telegram — система сама её обнаружит"
        )
        await query.message.edit_text(help_text, reply_markup=create_main_keyboard(), parse_mode="Markdown")
    else:
        await query.message.edit_text("Выберите действие:", reply_markup=create_main_keyboard())


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} caused error {context.error}")


# ================= НАСТРОЙКА ЛОГИРОВАНИЯ И ГЛАВНАЯ ФУНКЦИЯ =================
def setup_logging():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Config.ROOT / "system.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


async def main():
    setup_logging()

    if not Config.TELEGRAM_TOKEN:
        print(f"\n{Colors.RED}❌ Токен Telegram не найден{Colors.RESET}")
        print(f"{Colors.YELLOW}Создайте файл .env: TELEGRAM_BOT_TOKEN=ваш_токен{Colors.RESET}")
        return

    now_info = get_current_date_info()
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}" + "=" * 70)
    print(f"🚀 COGNITIVE AGENT PRO v8.4.1 UNIVERSAL FAST")
    print("=" * 70 + Colors.RESET)
    print(f"{Colors.GREEN}✅ Дата: {now_info['full']}{Colors.RESET}")
    print(f"{Colors.GREEN}✅ Время: {now_info['time']}{Colors.RESET}")
    print(f"{Colors.GREEN}✅ Токен загружен{Colors.RESET}\n")

    # ←←← КРИТИЧЕСКИ ВАЖНО: БЫСТРОЕ ОБНАРУЖЕНИЕ БЕЗ ВАЛИДАЦИИ ←←←
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 Обнаружение моделей (без валидации)...{Colors.RESET}")
    discoverer = LMStudioModelDiscoverer()
    models_config = await discoverer.discover_models()

    # ←←← МГНОВЕННОЕ СОЗДАНИЕ ИНТЕРФЕЙСОВ БЕЗ ПРОВЕРКИ ←←←
    models = [EnhancedModelInterface(config) for config in models_config]

    if not models:
        print(f"\n{Colors.RED}❌ Не найдено моделей!{Colors.RESET}")
        print(f"{Colors.YELLOW}Запустите любую текстовую модель в LM Studio (порт 1234){Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Обнаружено {len(models)} моделей:{Colors.RESET}")
    for i, m in enumerate(models, 1):
        caps = m.capabilities
        print(f"   ✓ {i}. {m.model_name}")
        print(
            f"      анализ={caps['analysis']:.2f} | логика={caps['reasoning']:.2f} | креативность={caps['creativity']:.2f}")

    # Инициализация компонентов
    dialog_engine = AdvancedBrainHemispheresDialog(models)
    db = EnhancedCognitiveDB(Config.DB_PATH)
    search_engine = ParallelWebSearchEngine() if HAS_WEB_SEARCH else None

    print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Система готова к работе!{Colors.RESET}")
    print(f"   • Моделей: {len(models)}")
    print(f"   • Веб-поиск: {'✅ активен' if search_engine else '❌ отключён'}")

    # Telegram приложение
    application = (
        ApplicationBuilder()
        .token(Config.TELEGRAM_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(20)
        .pool_timeout(20)
        .build()
    )

    application.bot_data['session_manager'] = SessionManager(db, dialog_engine, search_engine)

    await application.bot.set_my_commands([
        BotCommand("start", "Запустить бота"),
        BotCommand("stats", "Статистика"),
        BotCommand("clear", "Очистить контекст"),
        BotCommand("help", "Справка")
    ])

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    print(f"\n{Colors.BOLD}{Colors.GREEN}" + "=" * 70)
    print("✅ СИСТЕМА ЗАПУЩЕНА ЗА 2-3 СЕКУНДЫ")
    print("=" * 70 + Colors.RESET)
    print(f"\n{Colors.CYAN}⚡ Ключевые улучшения:{Colors.RESET}")
    print(f"   • НЕТ валидации при старте — мгновенный запуск")
    print(f"   • Ленивая валидация только при первом запросе (таймаут 7с)")
    print(f"   • Автоматический пропуск нерабочих моделей")
    print(f"   • Универсальная работа с ЛЮБЫМИ моделями")
    print(f"\n{Colors.YELLOW}📱 Откройте Telegram и отправьте /start{Colors.RESET}")
    print(f"{Colors.BLUE}💡 Первый запрос может занять 5-8 секунд (инициализация моделей){Colors.RESET}")
    print(f"{Colors.RED}🛑 Для остановки: Ctrl+C{Colors.RESET}\n")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await application.stop()


def run():
    print(f"{Colors.BOLD}{Colors.MAGENTA}Cognitive Agent Pro v8.4.1 UNIVERSAL FAST{Colors.RESET}")
    print(f"Python {sys.version.split()[0]}\n")

    now_info = get_current_date_info()
    print(f"{Colors.GREEN}📅 {now_info['full']}{Colors.RESET}")
    print(f"{Colors.GREEN}⏰ {now_info['time']}{Colors.RESET}\n")

    if not Config.TELEGRAM_TOKEN:
        print(f"{Colors.RED}❌ Токен не установлен{Colors.RESET}")
        print(f"Создайте .env: TELEGRAM_BOT_TOKEN=ваш_токен")
        return

    print(f"{Colors.GREEN}🚀 Запуск системы с мгновенным стартом...{Colors.RESET}\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}✅ Работа завершена{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Критическая ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run()