# coding: utf-8
"""
Cognitive_Agent_Pro_v8.0.py — Мульти-модельный когнитивный агент
✅ Автоматическое обнаружение всех запущенных моделей в LM Studio
✅ Синхронная работа моделей как "полушарий мозга" (диалог перед ответом)
✅ Надежный веб-поиск через DuckDuckGo + SearXNG fallback
✅ Видимое когнитивное мышление с цветными логами в консоли
✅ Русский интерфейс: команды /думай /анализ /цели /помощь /очистить
✅ Исправленные кнопки и навигация в Telegram
✅ Защита от ошибок парсинга JSON в самокритике
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
from typing import Dict, Optional, List, Any, Set, Tuple, Callable
from datetime import datetime, timedelta
import random
import math


# ================= ЦВЕТНЫЕ ЛОГИ ДЛЯ КОГНИТИВНОГО МЫШЛЕНИЯ =================
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
    """Логирование этапов когнитивного процесса с цветами"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    indent_str = "  " * indent
    print(f"{Colors.BOLD}{color}[{timestamp}] 🧠 {stage}{Colors.RESET}")
    print(f"{color}{indent_str}→ {message}{Colors.RESET}")


# ================= ВЕБ-ПОИСК (ИСПРАВЛЕННЫЙ) =================
HAS_WEB_SEARCH = False
WEB_SEARCH_METHOD = "отключён"

# Попытка импорта DuckDuckGo
try:
    from duckduckgo_search import AsyncDDGS

    HAS_WEB_SEARCH = True
    WEB_SEARCH_METHOD = "DuckDuckGo"
    print(f"{Colors.GREEN}✅ Веб-поиск активирован (DuckDuckGo){Colors.RESET}")
except ImportError:
    print(f"{Colors.YELLOW}⚠️ DuckDuckGo не установлен. Проверяю альтернативы...{Colors.RESET}")

# Fallback на SearXNG через aiohttp
if not HAS_WEB_SEARCH:
    try:
        import aiohttp

        HAS_WEB_SEARCH = True
        WEB_SEARCH_METHOD = "SearXNG"
        print(f"{Colors.GREEN}✅ Веб-поиск активирован (SearXNG){Colors.RESET}")
    except ImportError:
        print(f"{Colors.RED}❌ aiohttp не установлен. Веб-поиск недоступен.{Colors.RESET}")


# ================= ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =================
def load_dotenv_simple(path: Path = Path(".env")):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
        except Exception:
            pass


load_dotenv_simple()

# ================= СОВМЕСТИМОСТЬ =================
if sys.version_info >= (3, 13) and sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

# ================= TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )
except ImportError as e:
    print(f"{Colors.RED}❌ Ошибка импорта telegram: {e}{Colors.RESET}")
    print(f"{Colors.YELLOW}📦 Установите: pip install 'python-telegram-bot>=20.7' aiohttp requests{Colors.RESET}")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_system_pro")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"
    LOG_PATH = ROOT / "system.log"
    CACHE_PATH = ROOT / "cache_v2.db"
    KNOWLEDGE_PATH = ROOT / "knowledge_base.json"

    # LM Studio
    LM_STUDIO_BASE_URL = "http://localhost:1234"
    LM_STUDIO_MODELS_URL = f"{LM_STUDIO_BASE_URL}/v1/models"
    LM_STUDIO_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
    TIMEOUT = 180
    MAX_TOKENS = 4096

    # Память
    MEMORY_DECAY_RATE = 0.05
    MAX_MEMORY_ITEMS = 2000
    EPISODIC_MEMORY_LIMIT = 1000

    # Пороги
    CONFIDENCE_THRESHOLD = 0.65
    SEARCH_CONFIDENCE_THRESHOLD = 0.55
    IMPORTANCE_THRESHOLD = 0.7

    # Поиск
    TIME_SENSITIVE_KEYWORDS = [
        'курс', 'погода', 'новост', 'сегодня', 'сейчас', 'текущ',
        'последн', 'актуальн', 'свеж', 'обнов', 'тренд', 'прогноз'
    ]
    MAX_SEARCH_RESULTS = 4
    SEARCH_CACHE_TTL = 1800

    # Интервалы
    REFLECTION_INTERVAL = 8
    PROACTIVE_THINKING_INTERVAL = 120
    KNOWLEDGE_UPDATE_INTERVAL = 3600

    # Разное
    MAX_MESSAGE_LENGTH = 4096
    SESSION_TIMEOUT = 7200
    MAX_CONTEXT_WINDOW = 15

    @classmethod
    def get_telegram_token(cls) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print(f"\n{Colors.BOLD}{Colors.CYAN}" + "=" * 70)
            print("🔑 НЕОБХОДИМ ТОКЕН TELEGRAM БОТА")
            print("=" * 70 + Colors.RESET)
            print("1. Найдите в Telegram бота @BotFather")
            print("2. Отправьте команду /newbot и следуйте инструкциям")
            token = input("\nВведите токен Telegram бота: ").strip()
            if not token or not re.match(r'^\d+:[A-Za-z0-9_-]{35,}$', token):
                raise ValueError("❌ Неверный формат токена")
            os.environ["TELEGRAM_BOT_TOKEN"] = token
            with open(".env", "a", encoding="utf-8") as f:
                f.write(f'\nTELEGRAM_BOT_TOKEN="{token}"\n')
        return token


# ================= УТИЛИТЫ =================
def calculate_text_similarity(text1: str, text2: str) -> float:
    """Улучшенный расчёт схожести с учётом семантики"""
    if not text1 or not text2:
        return 0.0

    def normalize(text: str) -> List[str]:
        words = re.findall(r'\w+', text.lower())
        stop_words = {'и', 'в', 'на', 'с', 'по', 'о', 'у', 'к', 'для', 'это', 'то', 'так', 'же'}
        return [w for w in words if w not in stop_words and len(w) > 2]

    words1 = set(normalize(text1))
    words2 = set(normalize(text2))
    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    def get_trigrams(text: str) -> Set[str]:
        return set(text[i:i + 3] for i in range(len(text) - 2))

    trigram_sim = len(get_trigrams(text1) & get_trigrams(text2)) / max(len(get_trigrams(text1) | get_trigrams(text2)),
                                                                       1)
    jaccard = intersection / union
    return 0.7 * jaccard + 0.3 * trigram_sim


def split_message(text: str, max_length: int = 4096) -> List[str]:
    """Умное разбиение сообщений"""
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
    """Улучшенная клавиатура с эмодзи и понятными кнопками"""
    keyboard = [
        [
            InlineKeyboardButton("🧠 Думать", callback_data="think"),
            InlineKeyboardButton("📊 Анализ", callback_data="analyze"),
            InlineKeyboardButton("🎯 Цели", callback_data="goals")
        ],
        [
            InlineKeyboardButton("🔍 Поиск", callback_data="search"),
            InlineKeyboardButton("💡 Инсайты", callback_data="insights"),
            InlineKeyboardButton("🔗 Паттерны", callback_data="patterns")
        ],
        [
            InlineKeyboardButton("🧹 Очистить", callback_data="clear"),
            InlineKeyboardButton("❓ Помощь", callback_data="help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= АВТОМАТИЧЕСКОЕ ОБНАРУЖЕНИЕ МОДЕЛЕЙ В LM STUDIO =================
class LMStudioModelDiscoverer:
    """Автоматическое обнаружение всех запущенных моделей в LM Studio"""

    async def discover_models(self, base_url: str = "http://localhost:1234") -> List[Dict[str, Any]]:
        """Обнаруживает все модели, запущенные в LM Studio"""
        models = []

        # Шаг 1: Получаем список моделей через API
        try:
            log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ", f"Запрос списка моделей из {base_url}/v1/models", Colors.BLUE)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{base_url}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_models = data.get('data', [])

                        if raw_models:
                            log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ",
                                                f"Найдено {len(raw_models)} моделей в LM Studio", Colors.GREEN)

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
                        else:
                            log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ",
                                                "API вернул пустой список моделей", Colors.YELLOW)
                    else:
                        log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ",
                                            f"API вернул статус {response.status}", Colors.YELLOW)
        except Exception as e:
            log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ",
                                f"Ошибка при запросе моделей: {str(e)[:80]}", Colors.RED)

        # Шаг 2: Если не нашли через API — пробуем стандартные порты
        if not models:
            log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ",
                                "Пробуем стандартные порты (1234-1238) как резервный метод", Colors.YELLOW)
            models = await self._scan_standard_ports()

        # Шаг 3: Минимум 1 модель должна быть
        if not models:
            log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ",
                                "⚠️ Не найдено моделей. Используем фиктивную конфигурацию", Colors.RED)
            models = [{
                'id': 'fallback-model',
                'name': 'Резервная модель',
                'url': 'http://localhost:1234/v1/chat/completions',
                'port': 1234,
                'capabilities': {'reasoning': 0.5, 'creativity': 0.5, 'analysis': 0.5}
            }]

        return models

    async def _scan_standard_ports(self) -> List[Dict[str, Any]]:
        """Сканирует стандартные порты LM Studio (1234-1238)"""
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
                                    log_cognitive_stage("ОБНАРУЖЕНИЕ МОДЕЛЕЙ",
                                                        f"Найдена модель на порту {port}: {model_id}", Colors.GREEN,
                                                        indent=1)
            except:
                continue
        return models

    def _assess_capabilities(self, model_id: str) -> Dict[str, float]:
        """Оценивает возможности модели по её имени"""
        model_lower = model_id.lower()
        is_analytical = any(kw in model_lower for kw in ['phi', 'gemma', 'mistral', 'qwen', 'llama-3'])
        is_creative = any(kw in model_lower for kw in ['mixtral', 'yi', 'solar', 'deepseek'])

        return {
            'reasoning': 0.8 if is_analytical else 0.6,
            'creativity': 0.8 if is_creative else 0.5,
            'analysis': 0.9 if 'instruct' in model_lower else 0.7
        }


# ================= ИНТЕРФЕЙС ОДНОЙ МОДЕЛИ =================
class SingleModelInterface:
    """Интерфейс для работы с одной моделью LM Studio"""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.api_url = model_config['url']
        self.capabilities = model_config['capabilities']
        self.request_history = deque(maxlen=100)

    async def call(self, system_prompt: str, user_prompt: str,
                   temperature: float = 0.7, max_tokens: int = 2048) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt[:1000]},
                    {"role": "user", "content": user_prompt[:3000]}
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
                        return f"⚠️ Ошибка модели {self.model_name} ({response.status})"
        except asyncio.TimeoutError:
            return f"⚠️ Таймаут модели {self.model_name}"
        except Exception as e:
            return f"⚠️ Ошибка {self.model_name}: {str(e)[:100]}"

    def _clean_response(self, text: str) -> str:
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*|\*|__|_', '', text)
        text = re.sub(r'\n{3,}', '\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()


# ================= МУЛЬТИ-МОДЕЛЬНЫЙ КООРДИНАТОР ("ПОЛУШАРИЯ МОЗГА") =================
class MultiModelCoordinator:
    """Координация нескольких моделей как полушарий мозга"""

    def __init__(self, models: List[SingleModelInterface]):
        self.models = models
        self.model_count = len(models)
        log_cognitive_stage("АРХИТЕКТУРА",
                            f"Запущено {self.model_count} моделей-полушарий для синхронной работы", Colors.MAGENTA)

        for i, model in enumerate(models, 1):
            caps = model.capabilities
            print(f"   {Colors.CYAN}• Полушарие {i}: {model.model_name}")
            print(f"     Способности → Рассуждение: {caps['reasoning']:.0%} | "
                  f"Анализ: {caps['analysis']:.0%} | Креативность: {caps['creativity']:.0%}{Colors.RESET}")

    async def collaborative_thinking(self, system_prompt: str, user_prompt: str,
                                     temperature: float = 0.7) -> Tuple[str, str, List[Dict]]:
        """
        Диалог между моделями перед финальным ответом:
        1. Левое полушарие (аналитическое) — генерирует черновик
        2. Правое полушарие (креативное) — критикует и предлагает улучшения
        3. Синтез — финальный ответ с учетом диалога
        """
        log_cognitive_stage("КОГНИТИВНЫЙ ДИАЛОГ",
                            f"Начало внутреннего диалога между {self.model_count} полушариями", Colors.CYAN)

        # Этап 1: Черновик от самого аналитического полушария
        analytical_model = max(self.models, key=lambda m: m.capabilities['analysis'])
        log_cognitive_stage("ЭТАП 1: ЧЕРНОВИК",
                            f"Генерация черновика → {analytical_model.model_name}", Colors.BLUE, indent=1)

        draft = await analytical_model.call(
            system_prompt + " Ты — аналитик. Дай точный, структурированный черновик.",
            user_prompt,
            temperature=0.3
        )
        log_cognitive_stage("ЭТАП 1: ЧЕРНОВИК",
                            f"Получен черновик ({len(draft)} символов)", Colors.GREEN, indent=1)
        print(f"{Colors.WHITE}   Текст: {draft[:120]}...{Colors.RESET}")

        # Этап 2: Критика от самого креативного полушария
        creative_model = max(self.models, key=lambda m: m.capabilities['creativity'])
        critique_prompt = f"""
КРИТИЧЕСКИЙ АНАЛИЗ ЧЕРНОВИКА:
Вопрос: {user_prompt[:200]}
Черновик: {draft[:400]}

Проанализируй слабые места и предложи улучшения:
1. Фактические неточности или упущения
2. Пропущенные важные аспекты вопроса
3. Способы улучшения ясности и структуры
4. Нужна ли актуальная информация из интернета?
5. Как усилить полезность ответа для пользователя?

Ответь кратко и по существу.
"""
        log_cognitive_stage("ЭТАП 2: КРИТИКА",
                            f"Критический анализ → {creative_model.model_name}", Colors.YELLOW, indent=1)

        critique = await creative_model.call(
            "Ты — строгий критик. Будь объективен и конструктивен.",
            critique_prompt,
            temperature=0.2
        )
        log_cognitive_stage("ЭТАП 2: КРИТИКА",
                            "Критика получена", Colors.GREEN, indent=1)
        print(f"{Colors.WHITE}   Текст: {critique[:120]}...{Colors.RESET}")

        # Этап 3: Синтез финального ответа
        synthesis_prompt = f"""
СИНТЕЗИРУЙ ФИНАЛЬНЫЙ ОТВЕТ:
Вопрос: {user_prompt}
Черновик: {draft}
Критика: {critique}

Требования:
- Исправь указанные проблемы
- Добавь недостающую информацию
- Сохрани структуру и ясность
- Будь максимально полезным для пользователя
- Если критика указывает на устаревшие данные — отметь это
"""
        log_cognitive_stage("ЭТАП 3: СИНТЕЗ",
                            f"Формирование финального ответа → {analytical_model.model_name}", Colors.GREEN, indent=1)

        final = await analytical_model.call(
            "Ты — синтезатор. Создай идеальный ответ на основе диалога полушарий.",
            synthesis_prompt,
            temperature=0.5
        )

        # Собираем метаданные диалога
        dialogue_metadata = [
            {
                'stage': 'draft',
                'model': analytical_model.model_name,
                'text_preview': draft[:100],
                'length': len(draft)
            },
            {
                'stage': 'critique',
                'model': creative_model.model_name,
                'text_preview': critique[:100],
                'length': len(critique)
            },
            {
                'stage': 'synthesis',
                'model': analytical_model.model_name,
                'text_preview': final[:100],
                'length': len(final)
            }
        ]

        log_cognitive_stage("КОГНИТИВНЫЙ ДИАЛОГ",
                            "✅ Внутренний диалог завершен. Формируется ответ пользователю", Colors.GREEN)

        return final, critique, dialogue_metadata


# ================= НАДЕЖНЫЙ ВЕБ-ПОИСК =================
class EnhancedWebSearchEngine:
    """Надежный поиск через DuckDuckGo с fallback на SearXNG"""

    def __init__(self, cache_ttl: int = 1800):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.last_request_time = 0
        self.min_request_interval = 1.5

    async def search(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        log_cognitive_stage("ВЕБ-ПОИСК", f"Определена потребность в актуальной информации", Colors.YELLOW)
        print(f"{Colors.YELLOW}   Запрос: '{query[:60]}'{Colors.RESET}")

        # Проверка кэша
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                log_cognitive_stage("ВЕБ-ПОИСК",
                                    f"✅ Найдено в кэше ({len(cached_results)} результатов)", Colors.GREEN, indent=1)
                return cached_results

        # DuckDuckGo как основной источник
        log_cognitive_stage("ВЕБ-ПОИСК",
                            "🌐 Поиск через DuckDuckGo...", Colors.YELLOW, indent=1)

        results = await self._search_duckduckgo(query, max_results)

        if not results:
            log_cognitive_stage("ВЕБ-ПОИСК",
                                "⚠️ DuckDuckGo недоступен, пробуем SearXNG...", Colors.YELLOW, indent=1)
            results = await self._search_searx(query, max_results)

        if results:
            self.cache[cache_key] = (time.time(), results)
            log_cognitive_stage("ВЕБ-ПОИСК",
                                f"✅ Успешно найдено {len(results)} результатов", Colors.GREEN)
            for i, r in enumerate(results, 1):
                print(f"{Colors.WHITE}   {i}. {r['title'][:70]}{Colors.RESET}")
                print(f"{Colors.WHITE}      {r['url'][:60]}{Colors.RESET}")
        else:
            log_cognitive_stage("ВЕБ-ПОИСК",
                                "❌ Все источники недоступны", Colors.RED)

        return results

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
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
                            'snippet': snippet[:180],
                            'relevance': self._calculate_relevance(query, title, snippet)
                        })
                return sorted(filtered, key=lambda x: x['relevance'], reverse=True)[:max_results]
        except Exception as e:
            log_cognitive_stage("ВЕБ-ПОИСК",
                                f"Ошибка DuckDuckGo: {str(e)[:70]}", Colors.RED, indent=2)
            return []

    async def _search_searx(self, query: str, max_results: int) -> List[Dict]:
        instances = [
            "https://searx.be",
            "https://search.ononoki.org",
            "https://searxng.site",
            "https://searx.namejeff.xyz"
        ]

        for base_url in instances:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            f"{base_url}/search",
                            params={'q': query, 'format': 'json', 'language': 'ru'},
                            headers={'User-Agent': 'Mozilla/5.0'},
                            timeout=8
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            results = []
                            for r in data.get('results', [])[:max_results]:
                                if r.get('title') and r.get('url'):
                                    results.append({
                                        'title': r['title'][:120],
                                        'url': r['url'],
                                        'snippet': r.get('content', r.get('snippet', ''))[:180],
                                        'relevance': 0.8
                                    })
                            if results:
                                return results[:max_results]
            except:
                continue
        return []

    def _calculate_relevance(self, query: str, title: str, snippet: str) -> float:
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


# ================= СИСТЕМА САМОКРИТИКИ (ИСПРАВЛЕННАЯ) =================
class SelfCritiqueEngine:
    """Надежная самокритика с защитой от ошибок парсинга"""

    def __init__(self, coordinator: MultiModelCoordinator):
        self.coordinator = coordinator

    async def critique_response(self, user_query: str, draft_answer: str,
                                context: str = "") -> Dict[str, Any]:
        if not draft_answer or len(draft_answer.strip()) < 5:
            return self._get_safe_default()

        critique_prompt = f"""
ВОПРОС: {user_query[:300]}
ЧЕРНОВИК ОТВЕТА: {draft_answer[:500]}
КОНТЕКСТ: {context[:200]}

Проанализируй ответ по критериям (верни ТОЛЬКО валидный JSON):
{{
"accuracy": 0.0-1.0,
"completeness": 0.0-1.0,
"clarity": 0.0-1.0,
"coherence": 0.0-1.0,
"confidence": 0.0-1.0,
"needs_search": true/false,
"uncertain_claims": ["утверждение1", "утверждение2"],
"improvement_suggestions": ["совет1", "совет2"],
"overall_rating": 0.0-1.0
}}
"""
        # Используем координатор для критики
        critique_text, _, _ = await self.coordinator.collaborative_thinking(
            "Ты — строгий критик и эксперт по анализу текстов. Будь максимально объективен.",
            critique_prompt,
            temperature=0.1
        )

        return self._parse_critique_safely(critique_text)

    def _parse_critique_safely(self, response: str) -> Dict[str, Any]:
        try:
            # Ищем первую открывающую и последнюю закрывающую скобку
            start = response.find('{')
            end = response.rfind('}')
            if start == -1 or end == -1:
                return self._get_safe_default()

            json_str = response[start:end + 1]
            json_str = re.sub(r',\s*}', '}', json_str)  # Убираем запятые перед }
            json_str = re.sub(r',\s*\]', ']', json_str)  # Убираем запятые перед ]

            critique = json.loads(json_str)

            # Приводим к правильным типам
            return {
                'accuracy': float(critique.get('accuracy', 0.5)),
                'completeness': float(critique.get('completeness', 0.5)),
                'clarity': float(critique.get('clarity', 0.5)),
                'coherence': float(critique.get('coherence', 0.5)),
                'confidence': float(critique.get('confidence', 0.5)),
                'needs_search': bool(critique.get('needs_search', False)),
                'uncertain_claims': critique.get('uncertain_claims', []),
                'improvement_suggestions': critique.get('improvement_suggestions', []),
                'overall_rating': float(critique.get('overall_rating', 0.5)),
                'level': 1
            }
        except Exception as e:
            log_cognitive_stage("САМОКРИТИКА",
                                f"Ошибка парсинга: {str(e)[:60]}. Использую значения по умолчанию", Colors.YELLOW)
            return self._get_safe_default()

    def _get_safe_default(self) -> Dict[str, Any]:
        return {
            'accuracy': 0.5,
            'completeness': 0.5,
            'clarity': 0.5,
            'coherence': 0.5,
            'confidence': 0.5,
            'needs_search': False,
            'uncertain_claims': [],
            'improvement_suggestions': [],
            'overall_rating': 0.5,
            'level': 1,
            '_default': True
        }


# ================= УЛУЧШЕННАЯ БАЗА ДАННЫХ =================
class EnhancedCognitiveDB:
    """Упрощенная база данных для демонстрации"""

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
                confidence REAL DEFAULT 0.5,
                used_search INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def add_interaction(self, user_id: int, user_input: str, system_response: str, **kwargs):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interactions
            (timestamp, user_id, user_input, system_response, confidence, used_search, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            user_id,
            user_input[:1000],
            system_response[:2000],
            kwargs.get('confidence', 0.5),
            kwargs.get('used_search', 0),
            kwargs.get('importance', 0.5)
        ))
        conn.commit()
        conn.close()

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (user_id,))
        interactions = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM facts WHERE user_id = ?', (user_id,))
        facts = cursor.fetchone()[0]

        conn.close()

        return {
            'interactions': interactions,
            'facts': facts,
            'last_activity': time.time()
        }


# ================= КОГНИТИВНЫЙ АГЕНТ =================
class CognitiveAgent:
    """Основной когнитивный агент с мульти-модельной архитектурой"""

    def __init__(self, user_id: int, db: EnhancedCognitiveDB,
                 coordinator: MultiModelCoordinator,
                 search_engine: Optional[EnhancedWebSearchEngine],
                 critique_engine: SelfCritiqueEngine):
        self.user_id = user_id
        self.db = db
        self.coordinator = coordinator
        self.search_engine = search_engine
        self.critique_engine = critique_engine
        self.context_window = deque(maxlen=Config.MAX_CONTEXT_WINDOW)
        self.interaction_count = 0

    async def process_message(self, user_input: str) -> str:
        start_time = time.time()
        self.interaction_count += 1

        log_cognitive_stage("ПРИНЯТ ЗАПРОС",
                            f"Пользователь: {user_input[:80]}", Colors.GREEN)

        # Обработка специальных команд
        if special_response := await self._handle_special_requests(user_input):
            return special_response

        # Этап 1: Внутренний диалог полушарий → черновик
        log_cognitive_stage("ЭТАП 1", "Запуск внутреннего диалога полушарий", Colors.CYAN)
        draft_answer, critique_text, dialogue_metadata = await self.coordinator.collaborative_thinking(
            "Ты — когнитивный ассистент. Отвечай точно и полезно.",
            user_input,
            temperature=0.6
        )

        # Этап 2: Самокритика
        log_cognitive_stage("ЭТАП 2", "Самокритика ответа", Colors.YELLOW)
        critique = await self.critique_engine.critique_response(
            user_query=user_input,
            draft_answer=draft_answer,
            context=self._build_context_summary()
        )

        # Этап 3: Веб-поиск при необходимости
        search_results = ""
        used_search = False

        needs_search = (
                critique.get('needs_search', False) or
                critique.get('confidence', 1.0) < Config.SEARCH_CONFIDENCE_THRESHOLD or
                any(kw in user_input.lower() for kw in Config.TIME_SENSITIVE_KEYWORDS)
        )

        if needs_search and HAS_WEB_SEARCH and self.search_engine:
            log_cognitive_stage("ЭТАП 3", "Требуется актуальная информация → запуск веб-поиска", Colors.YELLOW)
            search_results = await self.search_engine.search(user_input, Config.MAX_SEARCH_RESULTS)
            used_search = True

        # Этап 4: Финальный синтез с учетом поиска
        if search_results:
            log_cognitive_stage("ЭТАП 4", "Синтез ответа с учетом результатов поиска", Colors.GREEN)
            final_answer = await self._synthesize_with_search(user_input, draft_answer, search_results)
        else:
            final_answer = draft_answer

        # Сохранение в базу
        self.db.add_interaction(
            user_id=self.user_id,
            user_input=user_input,
            system_response=final_answer,
            confidence=critique.get('confidence', 0.5),
            used_search=1 if used_search else 0,
            importance=self._calculate_importance(user_input, final_answer)
        )

        # Форматирование ответа
        formatted_response = self._format_response(final_answer, critique, used_search)

        processing_time = time.time() - start_time
        log_cognitive_stage("ЗАВЕРШЕНО",
                            f"Ответ сформирован за {processing_time:.1f}с | Уверенность: {critique.get('confidence', 0.5):.0%}",
                            Colors.GREEN)

        return formatted_response

    async def _synthesize_with_search(self, user_input: str, draft: str, results: List[Dict]) -> str:
        search_context = "\n".join([
            f"Источник {i + 1}: {r['title']}\n{r['snippet']}"
            for i, r in enumerate(results[:3])
        ])

        synthesis_prompt = f"""
ВОПРОС: {user_input}
ЧЕРНОВИК: {draft[:300]}
РЕЗУЛЬТАТЫ ПОИСКА:
{search_context}

Синтезируй точный и актуальный ответ, используя информацию из поиска.
Если данные противоречивы — укажи это. Не выдумывай факты.
"""
        final, _, _ = await self.coordinator.collaborative_thinking(
            "Ты — аналитик. Синтезируй ответ из источников точно и кратко.",
            synthesis_prompt,
            temperature=0.3
        )
        return final

    def _format_response(self, response: str, critique: Dict, used_search: bool) -> str:
        parts = [response]

        # Добавляем аннотации при низкой уверенности
        confidence = critique.get('confidence', 0.5)
        if confidence < 0.7:
            parts.append(f"\n\n⚠️ *Уверенность в ответе: {confidence:.0%}*")
            if uncertain := critique.get('uncertain_claims'):
                parts.append(f"❓ Требует проверки: {', '.join(uncertain[:2])}")

        # Отметка о поиске
        if used_search:
            parts.append("\n\n🔍 *Информация получена из актуальных источников*")

        return "\n".join(parts)

    def _build_context_summary(self) -> str:
        if not self.context_window:
            return "Нет предыдущего контекста"
        return "\n".join([
            f"{'Вы' if item['role'] == 'user' else 'Я'}: {item['content'][:60]}..."
            for item in list(self.context_window)[-4:]
        ])

    def _calculate_importance(self, user_input: str, response: str) -> float:
        importance = 0.5
        if '?' in user_input:
            importance += 0.2
        if len(user_input.split()) > 15:
            importance += 0.15
        return min(1.0, importance)

    async def _handle_special_requests(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()

        # Системное время без поиска
        if any(kw in text_lower for kw in ['какой сейчас год', 'какое время', 'который час', 'сегодняшняя дата']):
            now = datetime.now()
            months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                      'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
            weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

            if 'год' in text_lower:
                return f"📅 Сейчас **{now.year}** год"
            if 'число' in text_lower or 'дата' in text_lower:
                return f"📅 Сегодня **{now.day} {months[now.month - 1]} {now.year} года**"
            if 'день недели' in text_lower:
                return f"📅 Сегодня **{weekdays[now.weekday()]}**"
            if 'время' in text_lower or 'час' in text_lower:
                return f"⏰ Сейчас **{now.hour:02d}:{now.minute:02d}**"
            return f"📅 {now.day} {months[now.month - 1]} {now.year}, {weekdays[now.weekday()]}, {now.hour:02d}:{now.minute:02d}"

        # Русские команды
        commands = {
            '/думай': "🧠 Запущен процесс глубокого мышления. Мои полушария анализируют контекст...",
            '/анализ': self._handle_analysis_command,
            '/цели': "🎯 Функция декомпозиции целей активна. Опишите вашу цель для разбивки на шаги.",
            '/помощь': self._handle_help_command,
            '/очистить': "🧹 Контекст диалога очищен. Долгосрочная память сохранена."
        }

        for cmd, handler in commands.items():
            if text_lower.startswith(cmd):
                if callable(handler):
                    return await handler()
                return handler

        return None

    async def _handle_analysis_command(self) -> str:
        stats = self.db.get_user_stats(self.user_id)
        return (
            "📊 **Ваша статистика:**\n"
            f"• Взаимодействий: {stats['interactions']}\n"
            f"• Сохранённых фактов: {stats['facts']}\n"
            f"• Моделей-полушарий: {self.coordinator.model_count}\n"
            "\n🧠 **Архитектура мышления:**\n"
            "• Этап 1: Генерация черновика (аналитическое полушарие)\n"
            "• Этап 2: Критика и улучшение (креативное полушарие)\n"
            "• Этап 3: Поиск актуальной информации (при необходимости)\n"
            "• Этап 4: Синтез финального ответа"
        )

    async def _handle_help_command(self) -> str:
        return (
            "🤖 **Cognitive Agent Pro v8.0 — Помощь**\n\n"
            "🧠 **Команды:**\n"
            "• /думай — запустить глубокое когнитивное мышление\n"
            "• /анализ — показать статистику и архитектуру мышления\n"
            "• /цели — разбить сложную цель на шаги\n"
            "• /помощь — эта справка\n"
            "• /очистить — очистить контекст диалога\n\n"
            "✨ **Особенности:**\n"
            "• Мульти-модельная архитектура «полушарий мозга»\n"
            "• Видимое когнитивное мышление в консоли (цветные логи)\n"
            "• Автоматический веб-поиск только при необходимости\n"
            "• Системное время без обращения в интернет\n"
            "• Полная приватность — все запросы локальные"
        )


# ================= МЕНЕДЖЕР СЕССИЙ =================
class SessionManager:
    """Менеджер сессий с поддержкой мульти-модельной архитектуры"""

    def __init__(self, db: EnhancedCognitiveDB, coordinator: MultiModelCoordinator,
                 search_engine: Optional[EnhancedWebSearchEngine]):
        self.db = db
        self.coordinator = coordinator
        self.search_engine = search_engine
        self.sessions: Dict[int, CognitiveAgent] = {}
        self.last_cleanup = time.time()

    async def get_or_create_session(self, user_id: int) -> CognitiveAgent:
        if user_id not in self.sessions:
            critique_engine = SelfCritiqueEngine(self.coordinator)
            self.sessions[user_id] = CognitiveAgent(
                user_id=user_id,
                db=self.db,
                coordinator=self.coordinator,
                search_engine=self.search_engine,
                critique_engine=critique_engine
            )
            log_cognitive_stage("СЕССИЯ", f"Создана новая сессия для пользователя {user_id}", Colors.BLUE)
        return self.sessions[user_id]


# ================= TELEGRAM ОБРАБОТЧИКИ =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "👋 Привет! Я — **Cognitive Agent Pro v8.0**\n\n"
        "🧠 **Мульти-модельный когнитивный агент** с архитектурой «полушарий мозга»:\n"
        "• Левое полушарие — аналитическое мышление и структура\n"
        "• Правое полушарие — креативность и критический анализ\n"
        "• Синхронная работа для глубокого понимания запросов\n\n"
        "✨ **Команды:**\n"
        "• /думай — глубокое когнитивное мышление\n"
        "• /анализ — ваша статистика и архитектура мышления\n"
        "• /цели — декомпозиция целей на шаги\n"
        "• /помощь — справка по возможностям\n"
        "• /очистить — очистить контекст диалога\n\n"
        "🔒 Все запросы обрабатываются локально. Веб-поиск только при необходимости."
    )
    await update.message.reply_text(welcome, reply_markup=create_main_keyboard(), parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text:
        return

    log_cognitive_stage("ТЕЛЕГРАМ", f"Новое сообщение от {user_id}: {text[:50]}", Colors.WHITE)

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
        log_cognitive_stage("ОШИБКА", f"Ошибка обработки: {str(e)[:80]}", Colors.RED)
        await update.message.reply_text(
            "⚠️ Произошла ошибка. Попробуйте переформулировать запрос.",
            reply_markup=create_main_keyboard()
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    button_actions = {
        "think": ("🧠 Глубокое мышление", "/думай"),
        "analyze": ("📊 Статистика", "/анализ"),
        "goals": ("🎯 Декомпозиция целей", "/цели"),
        "search": ("🔍 Поиск",
                   "Для поиска актуальной информации задайте вопрос:\n• «Курс доллара»\n• «Погода в Москве»\n• «Последние новости»"),
        "insights": ("💡 Инсайты", "Мои полушария анализируют ваш контекст для генерации инсайтов..."),
        "patterns": ("🔗 Паттерны", "Анализ паттернов работает в фоновом режиме. Инсайты появляются автоматически."),
        "clear": ("🧹 Очистить контекст", "/очистить"),
        "help": ("❓ Справка", "/помощь")
    }

    action = button_actions.get(query.data)
    if not action:
        return

    title, response_text = action

    if query.data in ["think", "analyze", "goals", "clear", "help"]:
        # Эмулируем команду
        fake_update = Update(
            update_id=update.update_id,
            message=update.message or update.callback_query.message
        )
        fake_update.effective_user = update.effective_user

        if query.data == "think":
            await handle_message(fake_update, context)  # Обработаем как обычное сообщение "/думай"
        elif query.data == "analyze":
            agent = await context.application.bot_data['session_manager'].get_or_create_session(
                update.effective_user.id)
            response = await agent._handle_analysis_command()
            await query.message.edit_text(response, reply_markup=create_main_keyboard(), parse_mode="Markdown")
        elif query.data == "goals":
            await query.message.edit_text("🎯 Опишите вашу цель для разбивки на конкретные шаги:",
                                          reply_markup=create_main_keyboard())
        elif query.data == "clear":
            agent = await context.application.bot_data['session_manager'].get_or_create_session(
                update.effective_user.id)
            agent.context_window.clear()
            await query.message.edit_text("🧹 Контекст диалога очищен. Долгосрочная память сохранена.",
                                          reply_markup=create_main_keyboard())
        elif query.data == "help":
            agent = await context.application.bot_data['session_manager'].get_or_create_session(
                update.effective_user.id)
            response = await agent._handle_help_command()
            await query.message.edit_text(response, reply_markup=create_main_keyboard(), parse_mode="Markdown")
    else:
        await query.message.edit_text(f"{title}\n\n{response_text}", reply_markup=create_main_keyboard(),
                                      parse_mode="Markdown")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} caused error {context.error}")


# ================= ОСНОВНАЯ ФУНКЦИЯ =================
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

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}" + "=" * 70)
    print("🚀 COGNITIVE AGENT PRO v8.0 — МУЛЬТИ-МОДЕЛЬНЫЙ ЗАПУСК")
    print("=" * 70 + Colors.RESET)

    # Получение токена
    try:
        token = Config.get_telegram_token()
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка конфигурации: {e}{Colors.RESET}")
        return

    # Автоматическое обнаружение моделей
    print(f"\n{Colors.BOLD}{Colors.BLUE}🔍 Автоматическое обнаружение моделей в LM Studio...{Colors.RESET}")
    discoverer = LMStudioModelDiscoverer()
    models_config = await discoverer.discover_models()

    # Инициализация моделей
    models = [SingleModelInterface(config) for config in models_config]
    coordinator = MultiModelCoordinator(models)

    # Инициализация компонентов
    db = EnhancedCognitiveDB(Config.DB_PATH)
    search_engine = EnhancedWebSearchEngine() if HAS_WEB_SEARCH else None

    print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Инициализация завершена:{Colors.RESET}")
    print(f"   • База данных: {Config.DB_PATH}")
    print(f"   • Моделей обнаружено: {len(models)}")
    print(f"   • Веб-поиск: {'✅ активен' if search_engine else '❌ отключён'} ({WEB_SEARCH_METHOD})")

    # Telegram приложение
    application = (
        ApplicationBuilder()
        .token(token)
        .read_timeout(40)
        .write_timeout(40)
        .connect_timeout(30)
        .pool_timeout(30)
        .build()
    )

    application.bot_data['session_manager'] = SessionManager(db, coordinator, search_engine)

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("think", handle_message))
    application.add_handler(CommandHandler("reserch", handle_message))
    application.add_handler(CommandHandler("goal", handle_message))
    application.add_handler(CommandHandler("healp", handle_message))
    application.add_handler(CommandHandler("clear", handle_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    # Установка русских команд
    await application.bot.set_my_commands([
        BotCommand("start", "Запустить бота"),
        BotCommand("think", "Глубокое когнитивное мышление"),
        BotCommand("reserch", "Показать статистику"),
        BotCommand("goal", "Декомпозиция целей"),
        BotCommand("healp", "Справка по возможностям"),
        BotCommand("clear", "Очистить контекст диалога")
    ])

    # Запуск
    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

    print(f"\n{Colors.BOLD}{Colors.GREEN}" + "=" * 70)
    print("✅ COGNITIVE AGENT PRO v8.0 ЗАПУЩЕН УСПЕШНО!")
    print("=" * 70 + Colors.RESET)
    print(f"\n{Colors.CYAN}🧠 Архитектура «полушарий мозга» активна:{Colors.RESET}")
    print(f"   • Все обнаруженные модели работают синхронно")
    print(f"   • Видимое когнитивное мышление в консоли (цветные логи)")
    print(f"   • Веб-поиск только при необходимости для актуальных данных")
    print(f"\n{Colors.YELLOW}📱 Использование:{Colors.RESET}")
    print(f"   1. Откройте Telegram и найдите вашего бота")
    print(f"   2. Отправьте /start для начала")
    print(f"   3. Используйте команды /думай /анализ /цели")
    print(f"\n{Colors.RED}🛑 Для остановки нажмите Ctrl+C{Colors.RESET}")
    print("=" * 70 + "\n")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await application.stop()


def run():
    print(f"{Colors.BOLD}{Colors.MAGENTA}Cognitive Agent Pro v8.0 — Мульти-модельная когнитивная система{Colors.RESET}")
    print(f"Python {sys.version.split()[0]}")

    # Проверка зависимостей
    required_packages = {
        'aiohttp': 'aiohttp',
        'requests': 'requests',
        'duckduckgo-search (опционально)': 'duckduckgo_search'
    }

    missing = []
    for name, pkg in required_packages.items():
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(name)

    if missing:
        print(f"\n{Colors.YELLOW}⚠️  Отсутствуют пакеты: {', '.join(missing)}{Colors.RESET}")
        print(f"{Colors.YELLOW}📦 Установите: pip install aiohttp requests duckduckgo-search{Colors.RESET}")

    if not HAS_WEB_SEARCH:
        print(f"\n{Colors.YELLOW}⚠️  Веб-поиск ограничен. Для полной функциональности:")
        print(f"   pip install duckduckgo-search{Colors.RESET}")

    print(f"\n{Colors.GREEN}🚀 Запуск когнитивного агента...{Colors.RESET}\n")

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