# coding: utf-8
"""
Cognitive_Agent_Pro_v8.4_IMPROVED.py — Продвинутый диалог полушарий с синхронной работой
✅ НАСТОЯЩАЯ параллельная работа моделей-полушарий
✅ Механизм согласования и разрешения противоречий
✅ Система взвешенной оценки уверенности
✅ Улучшенная обработка ошибок с retry
✅ Параллельный веб-поиск из нескольких источников
✅ Метрики качества и производительности
✅ Интеллектуальное кэширование с TTL
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
    TIMEOUT = 180
    MAX_TOKENS = 4096
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2

    TIME_SENSITIVE_KEYWORDS = [
        'курс', 'погода', 'новост', 'сегодня', 'сейчас', 'текущ',
        'последн', 'актуальн', 'свеж', 'обнов', 'тренд', 'прогноз'
    ]
    MAX_SEARCH_RESULTS = 5
    SEARCH_CACHE_TTL = 1800
    MAX_MESSAGE_LENGTH = 4096
    MAX_CONTEXT_WINDOW = 15


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


# ================= АВТООБНАРУЖЕНИЕ МОДЕЛЕЙ =================
class LMStudioModelDiscoverer:
    """Автоматическое обнаружение моделей в LM Studio"""

    async def discover_models(self, base_url: str = "http://localhost:1234") -> List[Dict[str, Any]]:
        models = []

        try:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", f"Запрос моделей из {base_url}", Colors.BLUE)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{base_url}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_models = data.get('data', [])

                        if raw_models:
                            log_cognitive_stage("ОБНАРУЖЕНИЕ",
                                                f"Найдено {len(raw_models)} моделей", Colors.GREEN)

                            for idx, model in enumerate(raw_models):
                                model_id = model.get('id', f'model_{idx}')
                                models.append({
                                    'id': model_id,
                                    'name': model_id.split('/')[-1] if '/' in model_id else model_id,
                                    'url': f"{base_url}/v1/chat/completions",
                                    'port': 1234,
                                    'capabilities': self._assess_capabilities(model_id)
                                })
                                print(f"   {Colors.GREEN}✓ Модель {idx + 1}: {model_id}{Colors.RESET}")
        except Exception as e:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", f"Ошибка: {str(e)[:80]}", Colors.RED)

        if not models:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", "Сканирование портов 1234-1238", Colors.YELLOW)
            models = await self._scan_standard_ports()

        if not models:
            log_cognitive_stage("ОБНАРУЖЕНИЕ", "Используем резервную конфигурацию", Colors.RED)
            models = [{
                'id': 'fallback-model',
                'name': 'Резервная модель',
                'url': 'http://localhost:1234/v1/chat/completions',
                'port': 1234,
                'capabilities': {'reasoning': 0.5, 'creativity': 0.5, 'analysis': 0.5}
            }]

        return models

    async def _scan_standard_ports(self) -> List[Dict[str, Any]]:
        """Сканирование стандартных портов"""
        models = []
        for port in range(1234, 1239):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            f"http://localhost:{port}/v1/models",
                            timeout=aiohttp.ClientTimeout(total=3)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            raw_models = data.get('data', [])
                            if raw_models:
                                for model in raw_models:
                                    model_id = model.get('id', f'model_port_{port}')
                                    models.append({
                                        'id': model_id,
                                        'name': model_id.split('/')[-1],
                                        'url': f"http://localhost:{port}/v1/chat/completions",
                                        'port': port,
                                        'capabilities': self._assess_capabilities(model_id)
                                    })
                                    log_cognitive_stage("ОБНАРУЖЕНИЕ",
                                                        f"Порт {port}: {model_id}", Colors.GREEN, indent=1)
            except:
                continue
        return models

    def _assess_capabilities(self, model_id: str) -> Dict[str, float]:
        """Оценка способностей модели по имени"""
        model_lower = model_id.lower()
        is_analytical = any(
            kw in model_lower for kw in ['phi', 'gemma', 'mistral', 'qwen', 'llama-3', 'deepseek', 'thinking'])
        is_creative = any(kw in model_lower for kw in ['mixtral', 'yi', 'solar', 'deepseek', 'coder'])

        return {
            'reasoning': 0.85 if is_analytical else 0.6,
            'creativity': 0.85 if is_creative else 0.5,
            'analysis': 0.9 if 'instruct' in model_lower or 'thinking' in model_lower else 0.7
        }


# ================= УЛУЧШЕННЫЙ ИНТЕРФЕЙС МОДЕЛИ =================
class EnhancedModelInterface:
    """Интерфейс одной модели с retry и метриками"""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.api_url = model_config['url']
        self.capabilities = model_config['capabilities']
        self.call_history = deque(maxlen=100)
        self.total_calls = 0
        self.successful_calls = 0

    async def call_with_retry(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            retry_attempts: int = 3
    ) -> ModelResponse:
        """Вызов модели с автоматическим retry"""
        start_time = time.time()

        for attempt in range(retry_attempts):
            try:
                response = await self._call_api(
                    system_prompt, user_prompt, temperature, max_tokens
                )

                processing_time = time.time() - start_time
                self.total_calls += 1
                self.successful_calls += 1

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

            except asyncio.TimeoutError:
                if attempt < retry_attempts - 1:
                    log_cognitive_stage("RETRY",
                                        f"{self.model_name}: таймаут, попытка {attempt + 2}",
                                        Colors.YELLOW, indent=2)
                    await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
                else:
                    error_msg = "Превышено время ожидания"

            except Exception as e:
                if attempt < retry_attempts - 1:
                    log_cognitive_stage("RETRY",
                                        f"{self.model_name}: ошибка, попытка {attempt + 2}",
                                        Colors.YELLOW, indent=2)
                    await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
                else:
                    error_msg = str(e)[:100]

        # Все попытки исчерпаны
        self.total_calls += 1
        processing_time = time.time() - start_time

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
            max_tokens: int
    ) -> str:
        """Прямой вызов API модели"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt[:1500]},
                {"role": "user", "content": user_prompt[:3500]}
            ],
            "temperature": max(0.1, min(1.0, temperature)),
            "max_tokens": min(max_tokens, Config.MAX_TOKENS),
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
                    return self._clean_response(content)
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text[:100]}")

    def _clean_response(self, text: str) -> str:
        """Очистка ответа от форматирования"""
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*|\*|__|_', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _estimate_confidence(self, response: str) -> float:
        """Оценка уверенности по содержимому ответа"""
        if not response:
            return 0.0

        confidence = 0.7  # Базовая уверенность

        # Снижение уверенности при неопределенных фразах
        uncertain_phrases = ['возможно', 'может быть', 'вероятно', 'не уверен', 'сложно сказать']
        for phrase in uncertain_phrases:
            if phrase in response.lower():
                confidence -= 0.1

        # Повышение при наличии конкретных фактов
        if re.search(r'\d+%|\d+\s*(рубл|долл|евро)', response):
            confidence += 0.1

        # Повышение при структурированности
        if len(re.findall(r'\n\d+\.', response)) >= 2:
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
            'avg_processing_time': avg_time
        }


# ================= ПРОДВИНУТЫЙ ДИАЛОГ ПОЛУШАРИЙ =================
class AdvancedBrainHemispheresDialog:
    """
    Улучшенная симуляция диалога полушарий с синхронной работой и согласованием
    """

    def __init__(self, models: List[EnhancedModelInterface]):
        # Разделение моделей по способностям
        self.analytical_models = sorted(
            models,
            key=lambda m: m.capabilities['analysis'],
            reverse=True
        )
        self.creative_models = sorted(
            models,
            key=lambda m: m.capabilities['creativity'],
            reverse=True
        )

        # Назначение полушарий
        self.left_brain = self.analytical_models[0]
        self.right_brain = (self.creative_models[0] if len(self.creative_models) > 1
                            else self.analytical_models[1] if len(self.analytical_models) > 1
        else self.analytical_models[0])

        log_cognitive_stage("АРХИТЕКТУРА",
                            f"Левое полушарие: {self.left_brain.model_name}", Colors.BLUE)
        log_cognitive_stage("АРХИТЕКТУРА",
                            f"Правое полушарие: {self.right_brain.model_name}", Colors.MAGENTA)

    async def conduct_parallel_dialog(
            self,
            user_query: str,
            search_results: str = ""
    ) -> Tuple[str, ConsensusResult, List[Dict]]:
        """
        Проведение параллельного диалога с согласованием
        """
        dialog_history = []
        current_date = CURRENT_DATE_STR
        current_time = CURRENT_DATE_INFO['time']

        context = f"СЕГОДНЯ {current_date}, время {current_time}. Запрос: {user_query}"
        if search_results:
            context += f"\n\nПОИСК (актуально на {current_date}):\n{search_results}"

        # === РАУНД 1: ПАРАЛЛЕЛЬНАЯ ГЕНЕРАЦИЯ ЧЕРНОВИКОВ ===
        log_cognitive_stage("ДИАЛОГ", "Раунд 1: Параллельная генерация черновиков",
                            Colors.CYAN, indent=1)

        # Создаем промпты для обоих полушарий
        left_prompt = f"""
КОНТЕКСТ: {context}

РОЛЬ: Левое полушарие (аналитическое мышление)
ЗАДАЧА: Проанализируй запрос и создай структурированный, фактический черновик ответа.
Сегодня {current_date}. Будь точным и логичным. Не выдумывай факты.
"""

        right_prompt = f"""
КОНТЕКСТ: {context}

РОЛЬ: Правое полушарие (креативное мышление)
ЗАДАЧА: Создай черновик ответа с акцентом на ясность, практическую пользу и полноту.
Сегодня {current_date}. Подумай о том, что может упустить аналитический подход.
"""

        # ПАРАЛЛЕЛЬНЫЙ ВЫЗОВ ОБОИХ ПОЛУШАРИЙ
        left_task = self.left_brain.call_with_retry(
            system_prompt=f"Ты — левое полушарие. Сегодня {current_date}.",
            user_prompt=left_prompt,
            temperature=0.3
        )

        right_task = self.right_brain.call_with_retry(
            system_prompt=f"Ты — правое полушарие. Сегодня {current_date}.",
            user_prompt=right_prompt,
            temperature=0.5
        )

        # Ожидаем оба ответа одновременно
        left_response, right_response = await asyncio.gather(left_task, right_task)

        if not left_response.success or not right_response.success:
            log_cognitive_stage("ОШИБКА", "Один из черновиков не создан", Colors.RED, indent=2)
            # Используем успешный ответ или резервный текст
            if left_response.success:
                return self._create_fallback_response(left_response.content, dialog_history)
            elif right_response.success:
                return self._create_fallback_response(right_response.content, dialog_history)
            else:
                return self._create_error_response(dialog_history)

        dialog_history.append({
            'round': 1,
            'hemisphere': 'left',
            'model': left_response.model_name,
            'content': left_response.content[:200],
            'full_content': left_response.content,
            'confidence': left_response.confidence,
            'processing_time': left_response.processing_time
        })

        dialog_history.append({
            'round': 1,
            'hemisphere': 'right',
            'model': right_response.model_name,
            'content': right_response.content[:200],
            'full_content': right_response.content,
            'confidence': right_response.confidence,
            'processing_time': right_response.processing_time
        })

        log_cognitive_stage("ДИАЛОГ",
                            f"Левое ({left_response.confidence:.2f}): {left_response.content[:60]}...",
                            Colors.WHITE, indent=2)
        log_cognitive_stage("ДИАЛОГ",
                            f"Правое ({right_response.confidence:.2f}): {right_response.content[:60]}...",
                            Colors.WHITE, indent=2)

        # === РАУНД 2: АНАЛИЗ РАСХОЖДЕНИЙ И ВЗАИМНАЯ КРИТИКА ===
        log_cognitive_stage("ДИАЛОГ", "Раунд 2: Поиск расхождений и взаимная критика",
                            Colors.CYAN, indent=1)

        # Создаем промпты для взаимной критики
        left_critique_prompt = f"""
ЧЕРНОВИК ПРАВОГО ПОЛУШАРИЯ:
{right_response.content}

ТВОЙ ЧЕРНОВИК:
{left_response.content}

КОНТЕКСТ: {context}

ЗАДАЧА: Проанализируй оба черновика и выдели:
1. Ключевые расхождения между ними
2. Что упущено в твоем аналитическом подходе
3. Предложи улучшения для финального ответа
Сегодня {current_date}.
"""

        right_critique_prompt = f"""
ЧЕРНОВИК ЛЕВОГО ПОЛУШАРИЯ:
{left_response.content}

ТВОЙ ЧЕРНОВИК:
{right_response.content}

КОНТЕКСТ: {context}

ЗАДАЧА: Проанализируй оба черновика и выдели:
1. Ключевые расхождения между ними
2. Что упущено в твоем креативном подходе
3. Предложи улучшения для финального ответа
Сегодня {current_date}.
"""

        # ПАРАЛЛЕЛЬНАЯ ВЗАИМНАЯ КРИТИКА
        left_critique_task = self.left_brain.call_with_retry(
            system_prompt=f"Ты — левое полушарие. Анализируй критически. Сегодня {current_date}.",
            user_prompt=left_critique_prompt,
            temperature=0.4
        )

        right_critique_task = self.right_brain.call_with_retry(
            system_prompt=f"Ты — правое полушарие. Анализируй творчески. Сегодня {current_date}.",
            user_prompt=right_critique_prompt,
            temperature=0.4
        )

        left_critique, right_critique = await asyncio.gather(left_critique_task, right_critique_task)

        if left_critique.success and right_critique.success:
            dialog_history.extend([
                {
                    'round': 2,
                    'hemisphere': 'left_critique',
                    'model': left_critique.model_name,
                    'content': left_critique.content[:200],
                    'full_content': left_critique.content,
                    'confidence': left_critique.confidence
                },
                {
                    'round': 2,
                    'hemisphere': 'right_critique',
                    'model': right_critique.model_name,
                    'content': right_critique.content[:200],
                    'full_content': right_critique.content,
                    'confidence': right_critique.confidence
                }
            ])

            log_cognitive_stage("ДИАЛОГ",
                                f"Взаимная критика завершена", Colors.WHITE, indent=2)

        # === РАУНД 3: СИНТЕЗ И СОГЛАСОВАНИЕ ===
        log_cognitive_stage("ДИАЛОГ", "Раунд 3: Финальный синтез с согласованием",
                            Colors.GREEN, indent=1)

        synthesis_prompt = f"""
ИСХОДНЫЕ ЧЕРНОВИКИ:

ЛЕВОЕ ПОЛУШАРИЕ (уверенность {left_response.confidence:.2f}):
{left_response.content}

ПРАВОЕ ПОЛУШАРИЕ (уверенность {right_response.confidence:.2f}):
{right_response.content}

РЕЗУЛЬТАТЫ ВЗАИМНОЙ КРИТИКИ:

ОТ ЛЕВОГО:
{left_critique.content if left_critique.success else 'Не доступно'}

ОТ ПРАВОГО:
{right_critique.content if right_critique.success else 'Не доступно'}

КОНТЕКСТ: {context}

ЗАДАЧА ФИНАЛЬНОГО СИНТЕЗА:
1. Объедини лучшие элементы обоих подходов
2. Разреши все выявленные противоречия
3. Создай КРАТКИЙ, точный и полезный финальный ответ
4. Убери повторы и избыточные детали
5. Учти актуальность на {current_date}

ВАЖНО: Только финальный ответ пользователю. Без мета-комментариев.
"""

        # Используем модель с лучшей уверенностью для синтеза
        synthesizer = (self.left_brain if left_response.confidence >= right_response.confidence
                       else self.right_brain)

        final_response = await synthesizer.call_with_retry(
            system_prompt=f"Ты — синтезатор. Объединяй лучшее из обоих полушарий. Сегодня {current_date}.",
            user_prompt=synthesis_prompt,
            temperature=0.2
        )

        if not final_response.success:
            log_cognitive_stage("ОШИБКА", "Синтез не удался, используем взвешенное объединение",
                                Colors.YELLOW, indent=2)
            final_answer = self._weighted_merge(left_response, right_response)
            final_confidence = (left_response.confidence + right_response.confidence) / 2
        else:
            final_answer = final_response.content
            final_confidence = final_response.confidence

        dialog_history.append({
            'round': 3,
            'hemisphere': 'synthesis',
            'model': final_response.model_name if final_response.success else 'weighted_merge',
            'content': final_answer[:200],
            'full_content': final_answer,
            'confidence': final_confidence
        })

        # === СОЗДАНИЕ РЕЗУЛЬТАТА КОНСЕНСУСА ===
        consensus = self._analyze_consensus(
            left_response, right_response, final_answer
        )

        log_cognitive_stage("ДИАЛОГ",
                            f"✅ Консенсус достигнут (согласие: {consensus.agreement_score:.2f})",
                            Colors.GREEN)

        return final_answer, consensus, dialog_history

    def _weighted_merge(
            self,
            left_response: ModelResponse,
            right_response: ModelResponse
    ) -> str:
        """Взвешенное объединение ответов при отказе синтеза"""
        left_weight = left_response.confidence
        right_weight = right_response.confidence
        total = left_weight + right_weight

        if total == 0:
            return "⚠️ Не удалось сформировать ответ. Попробуйте переформулировать запрос."

        # Если одна модель значительно увереннее - используем её ответ
        if left_weight / total > 0.7:
            return left_response.content
        elif right_weight / total > 0.7:
            return right_response.content

        # Иначе объединяем с маркерами
        merged = f"{left_response.content}\n\n{right_response.content}"
        return merged[:2000]  # Ограничиваем длину

    def _analyze_consensus(
            self,
            left: ModelResponse,
            right: ModelResponse,
            final: str
    ) -> ConsensusResult:
        """Анализ достигнутого консенсуса"""
        # Простой анализ схожести
        left_words = set(left.content.lower().split())
        right_words = set(right.content.lower().split())

        if not left_words or not right_words:
            agreement_score = 0.0
        else:
            common = len(left_words & right_words)
            total = len(left_words | right_words)
            agreement_score = common / total if total > 0 else 0.0

        # Поиск явных конфликтных точек
        conflict_points = []

        # Проверка на противоречивые числа
        left_numbers = re.findall(r'\d+(?:\.\d+)?', left.content)
        right_numbers = re.findall(r'\d+(?:\.\d+)?', right.content)

        if left_numbers and right_numbers:
            left_nums = set(left_numbers)
            right_nums = set(right_numbers)
            if left_nums != right_nums:
                conflict_points.append("Расхождение в числовых данных")

        resolution_method = "parallel_synthesis"
        if agreement_score < 0.3:
            resolution_method = "weighted_merge"
        elif agreement_score > 0.7:
            resolution_method = "high_consensus"

        return ConsensusResult(
            final_answer=final,
            left_confidence=left.confidence,
            right_confidence=right.confidence,
            agreement_score=agreement_score,
            conflict_points=conflict_points,
            resolution_method=resolution_method
        )

    def _create_fallback_response(
            self,
            content: str,
            history: List[Dict]
    ) -> Tuple[str, ConsensusResult, List[Dict]]:
        """Создание резервного ответа"""
        consensus = ConsensusResult(
            final_answer=content,
            left_confidence=0.5,
            right_confidence=0.5,
            agreement_score=1.0,
            conflict_points=["Частичный отказ системы"],
            resolution_method="fallback"
        )
        return content, consensus, history

    def _create_error_response(
            self,
            history: List[Dict]
    ) -> Tuple[str, ConsensusResult, List[Dict]]:
        """Создание ответа об ошибке"""
        error_text = "⚠️ Произошла ошибка при обработке запроса. Попробуйте позже."
        consensus = ConsensusResult(
            final_answer=error_text,
            left_confidence=0.0,
            right_confidence=0.0,
            agreement_score=0.0,
            conflict_points=["Системная ошибка"],
            resolution_method="error"
        )
        return error_text, consensus, history


# ================= УЛУЧШЕННЫЙ ВЕБ-ПОИСК =================
class ParallelWebSearchEngine:
    """Параллельный веб-поиск из нескольких источников"""

    def __init__(self, cache_ttl: int = 1800):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.search_stats = {'total': 0, 'cache_hits': 0, 'successful': 0}

    async def parallel_search(self, query: str, max_results: int = 5) -> str:
        """Параллельный поиск по нескольким источникам"""
        current_date = CURRENT_DATE_STR
        current_year = CURRENT_YEAR

        log_cognitive_stage("ВЕБ-ПОИСК",
                            f"Параллельный поиск (актуально на {current_date})", Colors.YELLOW)
        print(f"{Colors.YELLOW}   Запрос: '{query[:60]}'{Colors.RESET}")

        # Добавляем год для актуальности
        dated_query = f"{query} {current_year}"

        # Проверка кэша
        cache_key = hashlib.sha256(dated_query.encode()).hexdigest()
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.search_stats['cache_hits'] += 1
                log_cognitive_stage("ВЕБ-ПОИСК",
                                    f"✅ Кэш ({len(cached_results)} результатов)",
                                    Colors.GREEN, indent=1)
                return self._format_results(cached_results, current_date)

        self.search_stats['total'] += 1

        # ПАРАЛЛЕЛЬНЫЙ ЗАПУСК ВСЕХ ПОИСКОВИКОВ
        search_tasks = [
            self._search_duckduckgo(dated_query, max_results),
            self._search_searx(dated_query, max_results)
        ]

        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Объединение и дедупликация результатов
        combined = []
        seen_urls = set()

        for results in all_results:
            if isinstance(results, list):
                for result in results:
                    url = result.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        combined.append(result)

        # Сортировка по релевантности
        combined.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        top_results = combined[:max_results]

        if top_results:
            self.cache[cache_key] = (time.time(), top_results)
            self.search_stats['successful'] += 1

            log_cognitive_stage("ВЕБ-ПОИСК",
                                f"✅ Найдено {len(top_results)} результатов", Colors.GREEN)
            for i, r in enumerate(top_results, 1):
                print(f"{Colors.WHITE}   {i}. {r['title'][:70]}{Colors.RESET}")

            return self._format_results(top_results, current_date)
        else:
            log_cognitive_stage("ВЕБ-ПОИСК", "❌ Результаты не найдены", Colors.RED)
            return ""

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """Поиск через DuckDuckGo"""
        try:
            from duckduckgo_search import AsyncDDGS
            async with AsyncDDGS() as ddgs:
                results = await ddgs.text(query, max_results=max_results * 2)
                filtered = []
                for r in results[:max_results * 2]:
                    title = r.get('title', '').strip()
                    url = r.get('href', '').strip()
                    snippet = r.get('body', '').strip()
                    if title and url and len(title) > 5 and 'duckduckgo' not in url:
                        filtered.append({
                            'title': title[:120],
                            'url': url,
                            'snippet': snippet[:200],
                            'relevance': self._calculate_relevance(query, title, snippet),
                            'source': 'DuckDuckGo'
                        })
                return sorted(filtered, key=lambda x: x['relevance'], reverse=True)[:max_results]
        except Exception as e:
            log_cognitive_stage("ПОИСК", f"DuckDuckGo: {str(e)[:50]}", Colors.YELLOW, indent=2)
            return []

    async def _search_searx(self, query: str, max_results: int) -> List[Dict]:
        """Поиск через SearXNG"""
        instances = [
            "https://searx.be",
            "https://search.ononoki.org",
            "https://searxng.site"
        ]

        for base_url in instances:
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
                                if r.get('title') and r.get('url'):
                                    results.append({
                                        'title': r['title'][:120],
                                        'url': r['url'],
                                        'snippet': r.get('content', r.get('snippet', ''))[:200],
                                        'relevance': 0.8,
                                        'source': 'SearXNG'
                                    })
                            if results:
                                return results[:max_results]
            except:
                continue
        return []

    def _calculate_relevance(self, query: str, title: str, snippet: str) -> float:
        """Расчет релевантности результата"""
        query_lower = query.lower()
        text_lower = (title + " " + snippet).lower()

        query_words = set(re.findall(r'\w+', query_lower))
        text_words = set(re.findall(r'\w+', text_lower))

        if not query_words:
            return 0.0

        word_match = len(query_words & text_words) / len(query_words)
        phrase_bonus = 0.2 if query_lower in text_lower else 0.0
        title_bonus = 0.15 if any(qw in title.lower() for qw in query_words) else 0.0

        return min(1.0, word_match * 0.6 + phrase_bonus + title_bonus)

    def _format_results(self, results: List[Dict], current_date: str) -> str:
        """Форматирование результатов поиска"""
        if not results:
            return ""

        formatted = f"РЕЗУЛЬТАТЫ ПОИСКА (актуально на {current_date}):\n\n"
        for i, r in enumerate(results[:4], 1):
            formatted += f"{i}. {r['title']}\n"
            formatted += f"   {r['snippet']}\n"
            formatted += f"   Источник: {r.get('source', 'Web')}\n\n"
        return formatted.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики поиска"""
        cache_hit_rate = (self.search_stats['cache_hits'] / self.search_stats['total'] * 100
                          if self.search_stats['total'] > 0 else 0)
        success_rate = (self.search_stats['successful'] / self.search_stats['total'] * 100
                        if self.search_stats['total'] > 0 else 0)

        return {
            'total_searches': self.search_stats['total'],
            'cache_hit_rate': cache_hit_rate,
            'success_rate': success_rate,
            'cache_size': len(self.cache)
        }


# ================= БАЗА ДАННЫХ =================
class EnhancedCognitiveDB:
    """Улучшенная база данных с метриками"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """Инициализация таблиц"""
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
                processing_time REAL DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def add_interaction(
            self,
            user_id: int,
            user_input: str,
            system_response: str,
            **kwargs
    ):
        """Добавление взаимодействия"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO interactions
            (timestamp, user_id, user_input, system_response, dialog_metadata, 
             consensus_metadata, used_search, left_confidence, right_confidence, 
             agreement_score, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            user_id,
            user_input[:1000],
            system_response[:2000],
            json.dumps(kwargs.get('dialog_metadata', []), ensure_ascii=False),
            json.dumps(asdict(kwargs.get('consensus', ConsensusResult(
                final_answer="", left_confidence=0.5, right_confidence=0.5,
                agreement_score=0.5, conflict_points=[], resolution_method="unknown"
            ))), ensure_ascii=False),
            kwargs.get('used_search', 0),
            kwargs.get('left_confidence', 0.5),
            kwargs.get('right_confidence', 0.5),
            kwargs.get('agreement_score', 0.5),
            kwargs.get('processing_time', 0)
        ))

        conn.commit()
        conn.close()

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Получение статистики пользователя"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (user_id,))
        interactions = cursor.fetchone()[0]

        cursor.execute('''
            SELECT AVG(agreement_score), AVG(processing_time), AVG(left_confidence), AVG(right_confidence)
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


# ================= КОГНИТИВНЫЙ АГЕНТ =================
class EnhancedCognitiveAgent:
    """Улучшенный когнитивный агент с параллельной обработкой"""

    def __init__(
            self,
            user_id: int,
            db: EnhancedCognitiveDB,
            dialog_engine: AdvancedBrainHemispheresDialog,
            search_engine: Optional[ParallelWebSearchEngine]
    ):
        self.user_id = user_id
        self.db = db
        self.dialog_engine = dialog_engine
        self.search_engine = search_engine
        self.context_window = deque(maxlen=Config.MAX_CONTEXT_WINDOW)

    async def process_message(self, user_input: str) -> str:
        """Обработка сообщения пользователя"""
        start_time = time.time()
        log_cognitive_stage("ЗАПРОС", f"Пользователь {self.user_id}: {user_input[:80]}", Colors.GREEN)

        # Быстрые запросы о времени
        if self._is_time_query(user_input):
            return self._handle_time_query(user_input)

        # Определение необходимости поиска
        needs_search = (
                any(kw in user_input.lower() for kw in Config.TIME_SENSITIVE_KEYWORDS) or
                any(kw in user_input.lower() for kw in ['2023', '2024', '2025'])
        )

        search_results = ""
        used_search = False

        # Параллельный поиск если нужен
        if needs_search and HAS_WEB_SEARCH and self.search_engine:
            log_cognitive_stage("ПОДГОТОВКА", "Запуск параллельного веб-поиска", Colors.YELLOW)
            search_results = await self.search_engine.parallel_search(
                user_input, Config.MAX_SEARCH_RESULTS
            )
            used_search = True

        # === ПАРАЛЛЕЛЬНЫЙ ДИАЛОГ ПОЛУШАРИЙ ===
        log_cognitive_stage("ОБРАБОТКА", "Запуск параллельного диалога полушарий", Colors.CYAN)

        final_answer, consensus, dialog_metadata = await self.dialog_engine.conduct_parallel_dialog(
            user_query=user_input,
            search_results=search_results
        )

        # Сохранение в БД
        processing_time = time.time() - start_time

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
            processing_time=processing_time
        )

        # Форматирование ответа
        formatted_response = self._format_response(
            final_answer, consensus, used_search
        )

        log_cognitive_stage("ЗАВЕРШЕНО",
                            f"Обработано за {processing_time:.1f}с (согласие: {consensus.agreement_score:.2f})",
                            Colors.GREEN)

        return formatted_response

    def _is_time_query(self, text: str) -> bool:
        """Проверка запроса о времени"""
        time_patterns = [
            r'какой сейчас год', r'текущий год', r'какой год',
            r'какое сегодня число', r'какой сегодня день',
            r'сколько времени', r'который час'
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in time_patterns)

    def _handle_time_query(self, text: str) -> str:
        """Обработка запроса о времени"""
        now_info = get_current_date_info()
        return f"📅 Сегодня {now_info['full']}. Время: {now_info['time']}"

    def _format_response(
            self,
            response: str,
            consensus: ConsensusResult,
            used_search: bool
    ) -> str:
        """Форматирование финального ответа"""
        # Очистка от мета-комментариев
        response = re.sub(r'(?i)как левое полушарие.*?\n', '', response)
        response = re.sub(r'(?i)как правое полушарие.*?\n', '', response)
        response = re.sub(r'(?i)синтезирую.*?\n', '', response)
        response = re.sub(r'(?i)финальный ответ:?\s*', '', response, count=1)

        # Добавление метаинформации
        meta_parts = []

        if used_search:
            meta_parts.append(f"🔍 Поиск на {CURRENT_DATE_STR}")

        if consensus.agreement_score < 0.5:
            meta_parts.append(f"⚠️ Низкое согласие полушарий ({consensus.agreement_score:.0%})")
        elif consensus.agreement_score > 0.8:
            meta_parts.append(f"✅ Высокое согласие ({consensus.agreement_score:.0%})")

        if meta_parts:
            response = response.rstrip() + "\n\n" + " | ".join(meta_parts)

        return response.strip()


# ================= МЕНЕДЖЕР СЕССИЙ =================
class SessionManager:
    """Менеджер сессий пользователей"""

    def __init__(
            self,
            db: EnhancedCognitiveDB,
            dialog_engine: AdvancedBrainHemispheresDialog,
            search_engine: Optional[ParallelWebSearchEngine]
    ):
        self.db = db
        self.dialog_engine = dialog_engine
        self.search_engine = search_engine
        self.sessions: Dict[int, EnhancedCognitiveAgent] = {}

    async def get_or_create_session(self, user_id: int) -> EnhancedCognitiveAgent:
        """Получение или создание сессии"""
        if user_id not in self.sessions:
            self.sessions[user_id] = EnhancedCognitiveAgent(
                user_id=user_id,
                db=self.db,
                dialog_engine=self.dialog_engine,
                search_engine=self.search_engine
            )
            log_cognitive_stage("СЕССИЯ", f"Создана для пользователя {user_id}", Colors.BLUE)
        return self.sessions[user_id]

    def get_global_stats(self) -> Dict[str, Any]:
        """Глобальная статистика"""
        return {
            'active_sessions': len(self.sessions),
            'left_brain_stats': self.dialog_engine.left_brain.get_stats(),
            'right_brain_stats': self.dialog_engine.right_brain.get_stats(),
            'search_stats': self.search_engine.get_stats() if self.search_engine else {}
        }


# ================= TELEGRAM ОБРАБОТЧИКИ =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    now_info = get_current_date_info()
    welcome = (
        f"👋 Привет! Я — **Cognitive Agent Pro v8.4 IMPROVED**\n\n"
        f"🧠 **Улучшенная архитектура:**\n"
        f"• ⚡ Параллельная работа полушарий\n"
        f"• 🤝 Автоматическое согласование ответов\n"
        f"• 📊 Система оценки уверенности\n"
        f"• 🔄 Retry механизм при ошибках\n"
        f"• 🔍 Параллельный веб-поиск\n\n"
        f"📅 **Актуальная дата:** {now_info['full']}\n"
        f"⏰ **Время:** {now_info['time']}\n\n"
        f"✨ Все модели работают синхронно для лучших результатов!"
    )
    await update.message.reply_text(
        welcome,
        reply_markup=create_main_keyboard(),
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка сообщений"""
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text:
        return

    log_cognitive_stage("TELEGRAM", f"Сообщение от {user_id}: {text[:50]}", Colors.WHITE)

    try:
        session_manager = context.application.bot_data['session_manager']
        agent = await session_manager.get_or_create_session(user_id)

        await update.message.reply_chat_action("typing")
        response = await agent.process_message(text)

        parts = split_message(response)
        for i, part in enumerate(parts):
            reply_markup = create_main_keyboard() if i == len(parts) - 1 else None
            await update.message.reply_text(
                part,
                reply_markup=reply_markup,
                parse_mode="Markdown",
                disable_web_page_preview=True
            )
            if i < len(parts) - 1:
                await asyncio.sleep(0.3)

    except Exception as e:
        log_cognitive_stage("ОШИБКА", f"Ошибка: {str(e)[:80]}", Colors.RED)
        await update.message.reply_text(
            "⚠️ Произошла ошибка. Попробуйте переформулировать запрос.",
            reply_markup=create_main_keyboard()
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка кнопок"""
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
            f"• Среднее согласие полушарий: {stats['avg_agreement']:.0%}\n"
            f"• Среднее время обработки: {stats['avg_processing_time']:.1f}с\n"
            f"• Левое полушарие: {stats['avg_left_confidence']:.0%} уверенности\n"
            f"• Правое полушарие: {stats['avg_right_confidence']:.0%} уверенности\n\n"
            f"🌐 **Глобальная статистика:**\n"
            f"• Активных сессий: {global_stats['active_sessions']}\n"
            f"• Успешность левого: {global_stats['left_brain_stats'].get('success_rate', 0):.0%}\n"
            f"• Успешность правого: {global_stats['right_brain_stats'].get('success_rate', 0):.0%}\n"
        )

        if global_stats['search_stats']:
            search = global_stats['search_stats']
            response_text += (
                f"\n🔍 **Веб-поиск:**\n"
                f"• Всего поисков: {search.get('total_searches', 0)}\n"
                f"• Попаданий в кэш: {search.get('cache_hit_rate', 0):.0%}\n"
                f"• Успешность: {search.get('success_rate', 0):.0%}\n"
            )

        await query.message.edit_text(
            response_text,
            reply_markup=create_main_keyboard(),
            parse_mode="Markdown"
        )

    elif query.data == "clear":
        agent.context_window.clear()
        await query.message.edit_text(
            "🧹 Контекст очищен. Память сохранена.",
            reply_markup=create_main_keyboard()
        )

    elif query.data == "help":
        help_text = (
            f"🤖 **Cognitive Agent Pro v8.4 IMPROVED**\n\n"
            f"⚡ **Новые возможности:**\n"
            f"• Модели работают параллельно\n"
            f"• Автоматическое согласование ответов\n"
            f"• Retry при ошибках\n"
            f"• Параллельный поиск по нескольким источникам\n\n"
            f"📋 **Команды:**\n"
            f"• /start - перезапуск\n"
            f"• /stats - подробная статистика\n"
            f"• /clear - очистка контекста\n\n"
            f"📅 Актуальная дата: {now_info['full']}"
        )
        await query.message.edit_text(
            help_text,
            reply_markup=create_main_keyboard(),
            parse_mode="Markdown"
        )

    else:
        await query.message.edit_text(
            "Выберите действие:",
            reply_markup=create_main_keyboard()
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logging.error(f"Update {update} caused error {context.error}")


# ================= ОСНОВНАЯ ФУНКЦИЯ =================
def setup_logging():
    """Настройка логирования"""
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
    """Главная функция"""
    setup_logging()

    # Проверка токена
    if not Config.TELEGRAM_TOKEN:
        print(f"\n{Colors.RED}❌ Токен Telegram не найден{Colors.RESET}")
        print(f"{Colors.YELLOW}Создайте файл .env со строкой:{Colors.RESET}")
        print(f"{Colors.CYAN}TELEGRAM_BOT_TOKEN=ваш_токен{Colors.RESET}")
        return

    now_info = get_current_date_info()
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}" + "=" * 70)
    print(f"🚀 COGNITIVE AGENT PRO v8.4 IMPROVED")
    print("=" * 70 + Colors.RESET)
    print(f"{Colors.GREEN}✅ Дата: {now_info['full']}{Colors.RESET}")
    print(f"{Colors.GREEN}✅ Время: {now_info['time']}{Colors.RESET}")
    print(f"{Colors.GREEN}✅ Токен загружен{Colors.RESET}\n")

    # Автообнаружение моделей
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 Обнаружение моделей...{Colors.RESET}")
    discoverer = LMStudioModelDiscoverer()
    models_config = await discoverer.discover_models()

    # Инициализация
    models = [EnhancedModelInterface(config) for config in models_config]
    dialog_engine = AdvancedBrainHemispheresDialog(models)
    db = EnhancedCognitiveDB(Config.DB_PATH)
    search_engine = ParallelWebSearchEngine() if HAS_WEB_SEARCH else None

    print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Система готова:{Colors.RESET}")
    print(f"   • Моделей: {len(models)}")
    print(f"   • Веб-поиск: {'✅ параллельный' if search_engine else '❌ отключён'}")
    print(f"   • База данных: {Config.DB_PATH}")

    # Telegram приложение
    application = (
        ApplicationBuilder()
        .token(Config.TELEGRAM_TOKEN)
        .read_timeout(40)
        .write_timeout(40)
        .connect_timeout(30)
        .pool_timeout(30)
        .build()
    )

    application.bot_data['session_manager'] = SessionManager(
        db, dialog_engine, search_engine
    )

    # Команды
    await application.bot.set_my_commands([
        BotCommand("start", "Запустить бота"),
        BotCommand("stats", "Подробная статистика"),
        BotCommand("clear", "Очистить контекст"),
        BotCommand("help", "Справка")
    ])

    # Обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

    print(f"\n{Colors.BOLD}{Colors.GREEN}" + "=" * 70)
    print("✅ СИСТЕМА ЗАПУЩЕНА")
    print("=" * 70 + Colors.RESET)
    print(f"\n{Colors.CYAN}⚡ Улучшения v8.4:{Colors.RESET}")
    print(f"   • Параллельная работа полушарий")
    print(f"   • Автоматическое согласование")
    print(f"   • Система оценки уверенности")
    print(f"   • Retry механизм")
    print(f"   • Параллельный веб-поиск")
    print(f"\n{Colors.YELLOW}📱 Откройте бота в Telegram и отправьте /start{Colors.RESET}")
    print(f"{Colors.RED}🛑 Для остановки: Ctrl+C{Colors.RESET}\n")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await application.stop()


def run():
    """Запуск приложения"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}Cognitive Agent Pro v8.4 IMPROVED{Colors.RESET}")
    print(f"Python {sys.version.split()[0]}\n")

    now_info = get_current_date_info()
    print(f"{Colors.GREEN}📅 {now_info['full']}{Colors.RESET}")
    print(f"{Colors.GREEN}⏰ {now_info['time']}{Colors.RESET}\n")

    # Проверка токена
    if not Config.TELEGRAM_TOKEN:
        print(f"{Colors.RED}❌ Токен не установлен{Colors.RESET}")
        print(f"Создайте .env: TELEGRAM_BOT_TOKEN=ваш_токен")
        return

    print(f"{Colors.GREEN}🚀 Запуск улучшенной системы...{Colors.RESET}\n")

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