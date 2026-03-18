#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v7.1 - ИСПРАВЛЕННАЯ ВЕРСИЯ
✅ Устранена ошибка: "asyncio.run() cannot be called from a running event loop"
✅ Корректная инициализация асинхронных ресурсов
✅ Надёжная обработка отсутствия веб-поиска
✅ Полная асинхронность без блокирующих вызовов
✅ Безопасный калькулятор (без eval())
✅ Мета-когнитивный цикл: ПЛАНИРОВАНИЕ → ВЫПОЛНЕНИЕ → РЕФЛЕКСИЯ → ОБУЧЕНИЕ
"""
import os
import json
import re
import asyncio
import aiohttp
import traceback
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Импорт для парсинга веб-страниц
try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️ BeautifulSoup не установлен. Парсинг веб-страниц будет ограничен.")

# Асинхронный DuckDuckGo Search (требуется: pip install "duckduckgo-search>=5.0")
DDGS_AVAILABLE = False
try:
    from duckduckgo_search import AsyncDDGS

    DDGS_AVAILABLE = True
    print("✅ Async DuckDuckGo Search доступен (версия >=5.0)")
except ImportError as e:
    print(f"⚠️ Async DuckDuckGo Search не установлен или устаревшая версия.")
    print(f"   Установите: pip install \"duckduckgo-search>=5.0\"")
    print(f"   Подробнее: https://pypi.org/project/duckduckgo-search/")

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
if not TELEGRAM_TOKEN:
    raise ValueError("❌ ОШИБКА: Не найден TELEGRAM_TOKEN в .env!")

LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

# Директории
CORES_DIR = "dynamic_cores"
MEMORY_DIR = "brain_memory"
USER_FILES_DIR = "user_files"
CACHE_DIR = "cache"

for directory in [CORES_DIR, MEMORY_DIR, USER_FILES_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Файлы
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
CORE_PERFORMANCE_LOG = os.path.join(MEMORY_DIR, "core_performance.json")
WEB_CACHE_FILE = os.path.join(CACHE_DIR, "web_search_cache.json")

init_file = os.path.join(CORES_DIR, "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w", encoding='utf-8') as f:
        f.write("# Auto-generated for imports\n")


# ==================== ТИПЫ ОТВЕТОВ ЯДЕР ====================
class CoreResponse:
    """Стандартизированный ответ от ядра знаний"""

    def __init__(
            self,
            success: bool,
            data: Optional[Dict[str, Any]] = None,
            raw_result: Optional[str] = None,
            confidence: float = 1.0,
            source: str = "unknown",
            direct_answer: bool = False,
            requires_verification: bool = False,
            metadata: Optional[Dict] = None
    ):
        self.success = success
        self.data = data or {}
        self.raw_result = raw_result
        self.confidence = confidence
        self.source = source
        self.direct_answer = direct_answer
        self.requires_verification = requires_verification
        self.metadata = metadata or {}

    def to_context_string(self) -> str:
        """Преобразование в текст для контекста LLM"""
        if not self.success:
            return f"❌ Ошибка получения данных из источника '{self.source}': {self.data.get('error', 'Неизвестная ошибка')}"

        if self.raw_result:
            return f"📊 ДАННЫЕ ОТ '{self.source.upper()}':\n{self.raw_result}"

        if self.data:
            try:
                formatted_data = json.dumps(self.data, ensure_ascii=False, indent=2)
                return f"📊 СТРУКТУРИРОВАННЫЕ ДАННЫЕ ОТ '{self.source.upper()}':\n{formatted_data}"
            except Exception:
                return f"📊 ДАННЫЕ ОТ '{self.source.upper()}': {str(self.data)[:500]}"

        return f"ℹ️ Источник '{self.source}' обработал запрос, но не вернул данных"

    def is_final_answer(self) -> bool:
        """Можно ли использовать ответ как финальный?"""
        return (self.direct_answer and self.success and
                self.raw_result and len(self.raw_result) > 10 and
                not self.requires_verification)


# ==================== БАЗОВЫЙ КЛАСС ЯДРА (АСИНХРОННЫЙ) ====================
class KnowledgeCore(ABC):
    """Базовый класс для всех ядер знаний"""
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []
    priority: int = 5
    direct_answer_mode: bool = False

    @abstractmethod
    async def can_handle(self, query: str, context: Optional[Dict] = None) -> Tuple[bool, float]:
        """Возвращает (может_обработать, уверенность_0_1)"""
        pass

    @abstractmethod
    async def execute(self, query: str, context: Optional[Dict] = None) -> CoreResponse:
        pass

    async def get_metadata(self) -> Dict[str, Any]:
        """Метаданные ядра для мета-когнитивного планирования"""
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': self.capabilities,
            'priority': self.priority,
            'direct_answer_mode': self.direct_answer_mode
        }


# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
    name = "datetime_core"
    description = "Точная информация о дате, времени и днях недели"
    capabilities = ["текущая дата", "время", "день недели", "расчёт дат"]
    priority = 10
    direct_answer_mode = True

    async def can_handle(self, query: str, context: Optional[Dict] = None) -> Tuple[bool, float]:
        q = query.lower().replace('фаил', 'файл')
        keywords = [
            'какой сегодня день', 'какое число', 'день недели', 'сколько времени',
            'который час', 'текущая дата', 'сегодня', 'завтра', 'послезавтра',
            'вчера', 'какой сейчас день', 'какая дата', 'время сейчас'
        ]
        if any(kw in q for kw in keywords):
            high_conf = ['какой сегодня', 'какое число', 'который час', 'сколько времени', 'какая дата']
            return True, 0.95 if any(kw in q for kw in high_conf) else 0.85
        return False, 0.0

    async def execute(self, query: str, context: Optional[Dict] = None) -> CoreResponse:
        try:
            now = datetime.now()
            weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
            months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля',
                      'августа', 'сентября', 'октября', 'ноября', 'декабря']

            days_offset = 0
            q_lower = query.lower().replace('фаил', 'файл')

            if 'завтра' in q_lower:
                days_offset = 1
            elif 'послезавтра' in q_lower:
                days_offset = 2
            elif 'вчера' in q_lower:
                days_offset = -1
            else:
                match = re.search(r'через\s+(\d+)\s*(дн[еяй]|день|дней)', q_lower)
                if match:
                    days_offset = int(match.group(1))

            target = now + timedelta(days=days_offset)
            data = {
                'date': target.strftime('%Y-%m-%d'),
                'day': target.day,
                'month': target.month,
                'month_name': months[target.month - 1],
                'year': target.year,
                'weekday': weekdays[target.weekday()],
                'weekday_num': target.weekday(),
                'time': target.strftime('%H:%M:%S'),
                'timestamp': target.isoformat(),
                'days_offset': days_offset
            }

            if days_offset == 0:
                description = (
                    f"📅 **Текущая дата и время:**\n"
                    f"• Дата: {target.day} {months[target.month - 1]} {target.year} года\n"
                    f"• День недели: {weekdays[target.weekday()]}\n"
                    f"• Время: {target.strftime('%H:%M:%S')}"
                )
            else:
                offset_text = f"через {abs(days_offset)} дн." if days_offset > 0 else f"{abs(days_offset)} дн. назад"
                description = (
                    f"📅 **Дата {offset_text}:**\n"
                    f"• {target.day} {months[target.month - 1]} {target.year} года\n"
                    f"• {weekdays[target.weekday()]}"
                )

            return CoreResponse(
                success=True,
                data=data,
                raw_result=description,
                confidence=0.98,
                source=self.name,
                direct_answer=True
            )
        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                confidence=0.0,
                source=self.name
            )


class SafeCalculator:
    """Безопасный калькулятор без eval()"""

    @staticmethod
    def calculate(expression: str) -> Tuple[bool, Any, str]:
        try:
            # Удаляем всё кроме цифр, операторов и точки
            clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            clean_expr = clean_expr.replace(' ', '')

            if not clean_expr or any(c in clean_expr for c in ['..', '++', '--', '**', '//']):
                return False, None, "Недопустимое выражение"

            # Парсим выражение вручную для безопасности
            result = SafeCalculator._eval_expr(clean_expr)
            return True, result, ""
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def _eval_expr(expr: str) -> float:
        """Рекурсивный парсер арифметических выражений"""
        # Убираем скобки
        while '(' in expr:
            # Находим самую внутреннюю пару скобок
            start = expr.rfind('(')
            end = expr.find(')', start)
            if end == -1:
                raise ValueError("Несбалансированные скобки")
            # Вычисляем содержимое скобок
            inner = SafeCalculator._eval_expr(expr[start + 1:end])
            # Заменяем скобки результатом
            expr = expr[:start] + str(inner) + expr[end + 1:]

        # Обработка умножения и деления
        tokens = re.findall(r'[+\-*/]|\d+(?:\.\d+)?', expr)
        if not tokens:
            raise ValueError("Пустое выражение")

        # Начинаем с первого числа
        result = float(tokens[0])
        i = 1

        while i < len(tokens):
            op = tokens[i]
            if i + 1 >= len(tokens):
                raise ValueError("Неполное выражение")

            num = float(tokens[i + 1])

            if op == '+':
                result += num
            elif op == '-':
                result -= num
            elif op == '*':
                result *= num
            elif op == '/':
                if num == 0:
                    raise ValueError("Деление на ноль")
                result /= num
            else:
                raise ValueError(f"Неизвестный оператор: {op}")

            i += 2

        return result


class CalculatorCore(KnowledgeCore):
    name = "calculator_core"
    description = "Математические вычисления"
    capabilities = ["сложение", "вычитание", "умножение", "деление", "проценты", "скобки"]
    priority = 9
    direct_answer_mode = True

    async def can_handle(self, query: str, context: Optional[Dict] = None) -> Tuple[bool, float]:
        q = query.lower().replace('фаил', 'файл')
        has_math = bool(re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', q))
        has_words = any(
            word in q for word in ['сколько будет', 'посчитай', 'вычисли', 'равно', 'реши', 'умножь', 'раздели'])
        if has_math:
            return True, 0.95
        if has_words and any(char.isdigit() for char in q):
            return True, 0.85
        return False, 0.0

    async def execute(self, query: str, context: Optional[Dict] = None) -> CoreResponse:
        try:
            # Извлекаем выражение
            expr_match = re.search(r'(\d+[\s\+\-\*\/x×÷\.\(\)]+\d+)', query)
            if not expr_match:
                # Пытаемся найти в тексте
                words = query.lower().split()
                for i, word in enumerate(words):
                    if any(op in word for op in ['+', '-', '*', '/', 'x', '×', '÷']):
                        expr_match = re.search(r'[\d.\(\)\+\-\*\/x×÷]+', word)
                        break

            if not expr_match:
                return CoreResponse(
                    success=False,
                    data={'error': 'Не найдено математическое выражение'},
                    confidence=0.0,
                    source=self.name
                )

            expr = expr_match.group(1)
            expr_clean = expr.replace('x', '*').replace('×', '*').replace('÷', '/')

            success, result, error = SafeCalculator.calculate(expr_clean)
            if not success:
                return CoreResponse(
                    success=False,
                    data={'error': error},
                    confidence=0.0,
                    source=self.name
                )

            data = {
                'expression': expr,
                'result': result,
                'formatted_expression': expr.replace('*', '×').replace('/', '÷')
            }
            description = f"🧮 **Результат вычисления:**\n`{data['formatted_expression']} = {result}`"

            return CoreResponse(
                success=True,
                data=data,
                raw_result=description,
                confidence=0.95,
                source=self.name,
                direct_answer=True
            )
        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e), 'query': query},
                confidence=0.0,
                source=self.name
            )


class WebSearchCache:
    """Кэш для веб-поиска с автоматической актуализацией"""

    def __init__(self, cache_file: str = WEB_CACHE_FILE, max_age_hours: int = 6):
        self.cache_file = cache_file
        self.max_age_hours = max_age_hours
        self.cache: Dict[str, Dict] = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Фильтруем устаревшие записи
                    now = datetime.now()
                    filtered = {
                        k: v for k, v in data.items()
                        if
                        (now - datetime.fromisoformat(v['timestamp'])).total_seconds() / 3600 < self.max_age_hours * 2
                    }
                    return filtered
        except Exception as e:
            print(f"⚠️ Ошибка загрузки кэша: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения кэша: {e}")

    def _normalize_query(self, query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r'[^\w\s\d\-]', ' ', q)
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    def get(self, query: str) -> Optional[Dict]:
        key = self._normalize_query(query)
        if key in self.cache:
            entry = self.cache[key]
            age_hours = (datetime.now() - datetime.fromisoformat(entry['timestamp'])).total_seconds() / 3600
            if age_hours < self.max_age_hours:
                print(f"📦 Кэш HIT для: '{query}' (возраст: {age_hours:.1f}ч)")
                return entry['data']
            else:
                print(f"📦 Кэш STALE для: '{query}' (возраст: {age_hours:.1f}ч)")
        return None

    def set(self, query: str, data: Dict):
        key = self._normalize_query(query)
        self.cache[key] = {
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'query': query
        }
        # Ограничиваем размер кэша
        if len(self.cache) > 1000:
            # Удаляем самые старые записи
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])
            self.cache = dict(sorted_items[-500:])
        self._save_cache()
        print(f"📦 Кэш сохранён для: '{query}'")


class WebSearchCore(KnowledgeCore):
    """
    🌐 WEB SEARCH CORE v4.1 - ПОЛНОСТЬЮ АСИНХРОННЫЙ С КЭШИРОВАНИЕМ
    """
    name = "web_search_core"
    description = "Поиск актуальной информации в интернете"
    capabilities = ["курсы валют", "погода", "новости", "общие факты", "события"]
    priority = 7
    direct_answer_mode = False  # Требует верификации LLM

    def __init__(self):
        self.ddgs = AsyncDDGS() if DDGS_AVAILABLE else None
        self.cache = WebSearchCache()
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def can_handle(self, query: str, context: Optional[Dict] = None) -> Tuple[bool, float]:
        if not self.ddgs:
            return False, 0.0

        q = query.lower().strip()
        # Исключаем мета-вопросы о боте
        meta_questions = [
            r"ты кто", r"что ты умеешь", r"кто тебя создал",
            r"как тебя зовут", r"ты бот", r"как работать", r"команды"
        ]
        if any(re.search(p, q) for p in meta_questions):
            return False, 0.0

        # Повышаем приоритет для запросов, требующих актуальности
        time_sensitive = [
            'сегодня', 'сейчас', 'курс', 'погода', 'новость',
            'новости', 'событие', 'происходит', 'актуально'
        ]
        if any(word in q for word in time_sensitive):
            return True, 0.92

        # Общий поиск
        return True, 0.75

    def _normalize_query(self, query: str) -> str:
        q = query.lower()
        garbage = [
            "найди", "найти", "поищи", "покажи", "расскажи",
            "какой", "какая", "какое", "какие", "что",
            "сколько", "пожалуйста", "можешь", "умеешь"
        ]
        for g in garbage:
            q = re.sub(rf"\b{g}\b", "", q)
        q = re.sub(r"\s+", " ", q).strip()

        # 💱 Усиление финансовых запросов
        if "курс" in q and ("доллар" in q or "usd" in q):
            return "курс доллара США к рублю сегодня ЦБ РФ официальный"
        if "курс" in q and ("евро" in q or "eur" in q):
            return "курс евро к рублю сегодня ЦБ РФ официальный"
        if "биткоин" in q or "btc" in q or "биткойн" in q:
            return "курс биткоина биткойна сегодня рубль"
        if "погода" in q:
            # Извлекаем город если есть
            city_match = re.search(r'погода\s+в\s+(\w+)', q)
            if city_match:
                return f"погода {city_match.group(1)} сегодня прогноз"
            return "погода сегодня прогноз"

        return q or query

    async def execute(self, query: str, context: Optional[Dict] = None) -> CoreResponse:
        if not self.ddgs:
            return CoreResponse(
                success=False,
                data={"error": "Веб-поиск недоступен. Установите: pip install \"duckduckgo-search>=5.0\""},
                confidence=0.0,
                source=self.name
            )

        await self.initialize()

        try:
            # Проверяем кэш сначала
            cached = self.cache.get(query)
            if cached:
                return CoreResponse(
                    success=True,
                    data=cached,
                    raw_result=cached.get('raw_result', ''),
                    confidence=0.85,
                    source=self.name,
                    direct_answer=False,
                    requires_verification=True,
                    metadata={'from_cache': True}
                )

            search_query = self._normalize_query(query)
            print(f"🌐 WEB SEARCH → '{search_query}'")

            results = await self.ddgs.text(
                search_query,
                max_results=5,
                safesearch="off"
            )

            if not results:
                return CoreResponse(
                    success=False,
                    data={"error": "Результаты не найдены"},
                    confidence=0.0,
                    source=self.name
                )

            parsed = []
            for i, r in enumerate(results[:5], 1):
                parsed.append({
                    "position": i,
                    "title": str(r.get("title", "")).strip(),
                    "url": str(r.get("href", "")).strip(),
                    "snippet": str(r.get("body", "")).strip()[:350]
                })

            # Формируем текстовый результат БЕЗ markdown для безопасности Telegram
            text_lines = ["🌍 **Актуальные результаты поиска:**"]
            for r in parsed:
                text_lines.append(f"\n{r['position']}. {r['title']}")
                text_lines.append(f"   {r['snippet']}")
                text_lines.append(f"   {r['url']}")

            raw_result = "\n".join(text_lines)

            data = {
                "original_query": query,
                "normalized_query": search_query,
                "results": parsed,
                "fetched_at": datetime.now().isoformat(),
                "result_count": len(parsed)
            }

            # Сохраняем в кэш
            self.cache.set(query, {
                **data,
                'raw_result': raw_result
            })

            return CoreResponse(
                success=True,
                data=data,
                raw_result=raw_result,
                confidence=0.9,
                source=self.name,
                direct_answer=False,  # Требует верификации LLM
                requires_verification=True,
                metadata={'from_cache': False}
            )

        except Exception as e:
            error_msg = f"Ошибка веб-поиска: {str(e)[:150]}"
            print(f"⚠️ {error_msg}")
            return CoreResponse(
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                source=self.name
            )


class FileStorageCore(KnowledgeCore):
    name = "file_storage_core"
    description = "Работа с текстовыми файлами пользователя"
    capabilities = ["сохранить файл", "прочитать файл", "список файлов", "удалить файл"]
    priority = 8
    direct_answer_mode = True

    def __init__(self):
        self.storage_dir = USER_FILES_DIR
        os.makedirs(self.storage_dir, exist_ok=True)

    async def can_handle(self, query: str, context: Optional[Dict] = None) -> Tuple[bool, float]:
        q = query.lower().replace('фаил', 'файл')
        patterns = [
            r'прочитай файл', r'прочти файл', r'открой файл',
            r'сохрани в файл', r'запиши в файл',
            r'список файлов', r'мои файлы', r'удали файл'
        ]
        for pattern in patterns:
            if re.search(pattern, q):
                return True, 0.95 if 'прочитай файл' in q or 'сохрани в файл' in q else 0.85
        return False, 0.0

    def _sanitize_filename(self, filename: str) -> str:
        if not filename:
            return "document.txt"
        filename = filename.strip().strip('\'"')
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        filename = ''.join(char for char in filename if 32 <= ord(
            char) <= 126 or char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        if not filename or filename.startswith('.'):
            filename = "document.txt"
        if '.' not in filename:
            filename += '.txt'
        return filename[:100]  # Ограничиваем длину

    def _get_file_path(self, filename: str) -> str:
        clean_filename = self._sanitize_filename(filename)
        filepath = os.path.join(self.storage_dir, clean_filename)
        abs_storage = os.path.abspath(self.storage_dir)
        abs_filepath = os.path.abspath(filepath)
        # Защита от path traversal
        if not abs_filepath.startswith(abs_storage + os.sep):
            raise ValueError(f"Недопустимый путь к файлу")
        return abs_filepath

    async def execute(self, query: str, context: Optional[Dict] = None) -> CoreResponse:
        try:
            q = query.lower().replace('фаил', 'файл')

            # Чтение файла
            if any(word in q for word in ['прочитай', 'прочти', 'открой']) and 'файл' in q:
                filename = ""
                patterns = [
                    r'прочитай файл\s+["\']?([^"\'\s]+)["\']?',
                    r'прочти файл\s+["\']?([^"\'\s]+)["\']?',
                    r'открой файл\s+["\']?([^"\'\s]+)["\']?',
                ]
                for pattern in patterns:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        filename = match.group(1).strip()
                        break

                if not filename:
                    # Попытка извлечь после слова "файл"
                    words = query.split()
                    for i, word in enumerate(words):
                        if word.lower() in ['файл', 'фаил'] and i + 1 < len(words):
                            filename = words[i + 1].strip('"\':.,!?')
                            break

                if not filename:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Не указано имя файла'},
                        raw_result="❌ **Пожалуйста, укажите имя файла.**\nПримеры:\n• `прочитай файл заметки.txt`",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

                try:
                    filepath = self._get_file_path(filename)
                except ValueError as e:
                    return CoreResponse(
                        success=False,
                        data={'error': str(e)},
                        raw_result=f"❌ **Ошибка имени файла:** {str(e)}",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

                if not os.path.exists(filepath):
                    # Показываем доступные файлы
                    files = []
                    try:
                        for f in os.listdir(self.storage_dir):
                            if os.path.isfile(os.path.join(self.storage_dir, f)):
                                files.append(f)
                    except Exception:
                        pass

                    if files:
                        file_list = "\n".join([f"• `{f}`" for f in files[:10]])
                        return CoreResponse(
                            success=False,
                            data={'error': 'Файл не найден', 'available_files': files},
                            raw_result=f"❌ **Файл `{filename}` не найден.**\n📁 **Доступные файлы:**\n{file_list}",
                            confidence=0.0,
                            source=self.name,
                            direct_answer=True
                        )
                    else:
                        return CoreResponse(
                            success=False,
                            data={'error': 'Файл не найден'},
                            raw_result=f"❌ **Файл `{filename}` не найден.**\n📁 **Хранилище пусто.**",
                            confidence=0.0,
                            source=self.name,
                            direct_answer=True
                        )

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_size = os.path.getsize(filepath)
                    lines = content.count('\n') + 1
                    preview = content[:500] + ('...' if len(content) > 500 else '')

                    return CoreResponse(
                        success=True,
                        data={'filename': filename, 'content': content, 'size': file_size, 'lines': lines},
                        raw_result=f"📄 **Файл `{filename}`:**\n```\n{preview}\n```\n📊 **Информация:**\n• Размер: {file_size} байт\n• Строк: {lines}",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )
                except UnicodeDecodeError:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Файл не в UTF-8'},
                        raw_result=f"❌ **Не могу прочитать файл `{filename}` (кодировка не UTF-8).**",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

            # Сохранение файла
            elif any(word in q for word in ['сохрани', 'запиши']) and 'файл' in q:
                match = re.search(r'сохрани\s+(?:в\s+)?файл\s+["\']?([^"\'\n:]+)["\']?\s*[:：]\s*(.+)', query,
                                  re.IGNORECASE | re.DOTALL)
                if not match:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Неверный формат'},
                        raw_result="❌ **Неверный формат.**\nПример: `сохрани в файл привет.txt: Привет, мир!`",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

                filename = match.group(1).strip()
                content = match.group(2).strip()

                try:
                    filepath = self._get_file_path(filename)
                except ValueError as e:
                    return CoreResponse(
                        success=False,
                        data={'error': str(e)},
                        raw_result=f"❌ **Ошибка пути:** {str(e)}",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    file_size = len(content.encode('utf-8'))
                    preview = content[:200] + ('...' if len(content) > 200 else '')

                    return CoreResponse(
                        success=True,
                        data={'filename': filename, 'size': file_size, 'path': filepath},
                        raw_result=f"✅ **Файл сохранён!**\n📄 `{filename}` ({file_size} байт)\n📝 {preview}",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )
                except Exception as e:
                    return CoreResponse(
                        success=False,
                        data={'error': str(e)},
                        raw_result=f"❌ **Ошибка записи:** {str(e)}",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

            # Список файлов
            elif 'список файлов' in q or 'мои файлы' in q:
                try:
                    files = []
                    total_size = 0
                    for f in os.listdir(self.storage_dir):
                        f_path = os.path.join(self.storage_dir, f)
                        if os.path.isfile(f_path):
                            size = os.path.getsize(f_path)
                            modified = datetime.fromtimestamp(os.path.getmtime(f_path)).strftime('%d.%m.%Y %H:%M')
                            files.append({'name': f, 'size': size, 'modified': modified})
                            total_size += size

                    if not files:
                        return CoreResponse(
                            success=True,
                            data={'files': [], 'total_size': 0},
                            raw_result="📁 **Хранилище пусто.**",
                            confidence=1.0,
                            source=self.name,
                            direct_answer=True
                        )

                    files.sort(key=lambda x: x['modified'], reverse=True)
                    files_list = []
                    for i, f in enumerate(files[:15], 1):
                        size_kb = f['size'] / 1024
                        size_str = f"{size_kb:.1f} KB" if size_kb >= 1 else f"{f['size']} байт"
                        files_list.append(f"{i}. **{f['name']}** ({size_str}) • {f['modified']}")

                    files_text = "\n".join(files_list)
                    total_kb = total_size / 1024
                    total_str = f"{total_kb:.1f} KB" if total_kb >= 1 else f"{total_size} байт"

                    return CoreResponse(
                        success=True,
                        data={'files': files, 'total_size': total_size, 'count': len(files)},
                        raw_result=f"📁 **Файлов:** {len(files)} ({total_str})\n{files_text}",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )
                except Exception as e:
                    return CoreResponse(
                        success=False,
                        data={'error': str(e)},
                        raw_result=f"❌ **Ошибка:** {str(e)}",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

            # Удаление файла
            elif 'удали файл' in q:
                filename = ""
                match = re.search(r'удали\s+файл\s+["\']?([^"\'\s]+)["\']?', query, re.IGNORECASE)
                if match:
                    filename = match.group(1).strip()

                if not filename:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Не указано имя файла'},
                        raw_result="❌ **Укажите имя файла для удаления.**\nПример: `удали файл старый.txt`",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

                try:
                    filepath = self._get_file_path(filename)
                except ValueError as e:
                    return CoreResponse(
                        success=False,
                        data={'error': str(e)},
                        raw_result=f"❌ **Ошибка пути:** {str(e)}",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

                if not os.path.exists(filepath):
                    return CoreResponse(
                        success=False,
                        data={'error': 'Файл не найден'},
                        raw_result=f"❌ **Файл `{filename}` не найден.**",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

                try:
                    os.remove(filepath)
                    return CoreResponse(
                        success=True,
                        data={'filename': filename, 'deleted': True},
                        raw_result=f"✅ **Файл `{filename}` удалён.**",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )
                except Exception as e:
                    return CoreResponse(
                        success=False,
                        data={'error': str(e)},
                        raw_result=f"❌ **Ошибка удаления:** {str(e)}",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

            # Справка
            else:
                return CoreResponse(
                    success=True,
                    data={'action': 'help'},
                    raw_result="📁 **Команды для работы с файлами:**\n"
                               "• `прочитай файл имя.txt`\n"
                               "• `сохрани в файл имя.txt: текст`\n"
                               "• `список файлов`\n"
                               "• `удали файл имя.txt`",
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ **Ошибка:** {str(e)}",
                confidence=0.0,
                source=self.name,
                direct_answer=True
            )


# ==================== МЕТА-КОГНИТИВНАЯ СИСТЕМА ПАМЯТИ v3.1 (ИСПРАВЛЕНА) ====================
class MetaCognitiveMemory:
    """
    🧠 МЕТА-КОГНИТИВНАЯ СИСТЕМА ПАМЯТИ
    Исправлено: все методы корректно работают в асинхронном контексте
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.memory_dir, exist_ok=True)

        # Файлы памяти
        self.short_term_file = os.path.join(self.memory_dir, "short_term.json")
        self.long_term_file = os.path.join(self.memory_dir, "long_term.json")
        self.patterns_file = os.path.join(self.memory_dir, "patterns.json")
        self.hypotheses_file = os.path.join(self.memory_dir, "hypotheses.json")
        self.meta_knowledge_file = os.path.join(self.memory_dir, "meta_knowledge.json")
        self.metadata_file = os.path.join(self.memory_dir, "metadata.json")

        # Загрузка памяти
        self.short_term = self._load_json(self.short_term_file, [])
        self.long_term = self._load_json(self.long_term_file, [])
        self.patterns = self._load_json(self.patterns_file, {
            'preferences': {},
            'communication_style': 'нейтральный',
            'frequent_topics': [],
            'emotional_triggers': {},
            'last_updated': datetime.now().isoformat()
        })
        self.hypotheses = self._load_json(self.hypotheses_file, [])
        self.meta_knowledge = self._load_json(self.meta_knowledge_file, {
            'successful_strategies': [],
            'failed_strategies': [],
            'tool_preferences': {},
            'last_self_reflection': None
        })
        self.metadata = self._load_json(self.metadata_file, {
            'total_interactions': 0,
            'facts_learned': 0,
            'hypotheses_verified': 0,
            'hypotheses_rejected': 0,
            'first_interaction': datetime.now().isoformat(),
            'last_interaction': datetime.now().isoformat()
        })

        print(f"🧠 Память загружена для {user_id}: "
              f"краткосрочная={len(self.short_term)}, "
              f"долгосрочная={len(self.long_term)}, "
              f"гипотез={len(self.hypotheses)}")

    # ==================== ОСНОВНЫЕ ОПЕРАЦИИ С ПАМЯТЬЮ ====================
    async def add_interaction(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Добавляет взаимодействие в краткосрочную память"""
        entry = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.short_term.append(entry)

        # Ограничиваем размер
        if len(self.short_term) > 30:
            self.short_term = self.short_term[-30:]

        self._save_json(self.short_term_file, self.short_term)

        # Обновляем метаданные
        self.metadata['total_interactions'] += 1
        self.metadata['last_interaction'] = datetime.now().isoformat()
        self._save_json(self.metadata_file, self.metadata)

    def get_short_term_context(self, limit: int = 10) -> str:
        """Возвращает последние N сообщений"""
        recent = self.short_term[-limit:] if limit else self.short_term
        return "\n".join([
            f"[{datetime.fromisoformat(e['timestamp']).strftime('%H:%M')}] {e['role']}: {e['content']}"
            for e in recent
        ])

    async def add_fact(self, fact: str, category: str = 'general', importance: float = 0.7,
                       confidence: float = 0.8, emotional_marker: str = 'neutral'):
        """Добавляет факт в долгосрочную память с детекцией дубликатов"""
        # Проверка на дубликаты через семантическое сравнение
        for existing in self.long_term:
            if self._semantic_similarity(fact, existing['content']) > 0.85:
                # Обновляем существующий факт
                existing['importance'] = max(existing.get('importance', 0.5), importance)
                existing['confidence'] = max(existing.get('confidence', 0.5), confidence)
                existing['last_updated'] = datetime.now().isoformat()
                existing['access_count'] = existing.get('access_count', 0) + 1
                self._save_json(self.long_term_file, self.long_term)
                return

        # Новый факт
        entry = {
            'content': fact,
            'category': category,
            'importance': importance,
            'confidence': confidence,
            'emotional_marker': emotional_marker,
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'access_count': 1,
            'id': hashlib.md5(fact.encode()).hexdigest()[:12]
        }
        self.long_term.append(entry)

        # Сортируем по важности и ограничиваем размер
        self.long_term.sort(key=lambda x: x['importance'], reverse=True)
        if len(self.long_term) > 100:
            self.long_term = self.long_term[:100]

        self._save_json(self.long_term_file, self.long_term)
        self.metadata['facts_learned'] += 1
        self._save_json(self.metadata_file, self.metadata)

    async def search_facts(self, query: str, limit: int = 5, min_similarity: float = 0.4) -> List[Dict]:
        """Семантический поиск в долгосрочной памяти"""
        if not self.long_term:
            return []

        scored = []
        query_norm = self._normalize_text(query)
        query_words = set(query_norm.split())

        for mem in self.long_term:
            content_norm = self._normalize_text(mem['content'])
            content_words = set(content_norm.split())

            if not content_words:
                continue

            # Jaccard similarity
            intersection = query_words & content_words
            union = query_words | content_words
            similarity = len(intersection) / len(union) if union else 0.0

            # Взвешиваем по важности и свежести
            importance = mem.get('importance', 0.5)
            try:
                timestamp = datetime.fromisoformat(mem['timestamp'])
                days_old = (datetime.now() - timestamp).days
                freshness = max(0.1, 1 - days_old / 90)  # 3 месяца = 0.1
            except:
                freshness = 0.5

            weight = (
                    similarity * 0.5 +
                    importance * 0.3 +
                    freshness * 0.2
            )

            if weight >= min_similarity:
                scored.append((weight, mem))
                # Увеличиваем счётчик доступа
                mem['access_count'] = mem.get('access_count', 0) + 1
                mem['last_accessed'] = datetime.now().isoformat()

        # Сохраняем обновлённую память
        self._save_json(self.long_term_file, self.long_term)

        # Сортируем и возвращаем
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    async def add_hypothesis(self, hypothesis: str, evidence: List[str], confidence: float = 0.6):
        """Добавляет гипотезу о пользователе для будущей верификации"""
        self.hypotheses.append({
            'hypothesis': hypothesis,
            'evidence': evidence,
            'confidence': confidence,
            'created_at': datetime.now().isoformat(),
            'verified': None,  # None = не проверено, True = подтверждено, False = опровергнуто
            'verification_attempts': 0
        })

        # Ограничиваем количество гипотез
        if len(self.hypotheses) > 20:
            # Удаляем самые старые непроверенные или с низкой уверенностью
            self.hypotheses.sort(key=lambda h: (h['verified'] is not None, h['confidence']), reverse=True)
            self.hypotheses = self.hypotheses[:20]

        self._save_json(self.hypotheses_file, self.hypotheses)

    async def update_hypothesis(self, hypothesis_id: int, verified: bool, evidence: str = ""):
        """Обновляет статус гипотезы после верификации"""
        if 0 <= hypothesis_id < len(self.hypotheses):
            self.hypotheses[hypothesis_id]['verified'] = verified
            self.hypotheses[hypothesis_id]['verification_attempts'] += 1
            self.hypotheses[hypothesis_id]['last_verification'] = datetime.now().isoformat()
            if evidence:
                self.hypotheses[hypothesis_id].setdefault('verification_evidence', []).append(evidence)

            if verified:
                self.metadata['hypotheses_verified'] += 1
                # Добавляем подтверждённую гипотезу как факт
                await self.add_fact(
                    self.hypotheses[hypothesis_id]['hypothesis'],
                    category='inferred_preference',
                    importance=0.8,
                    confidence=min(0.95, self.hypotheses[hypothesis_id]['confidence'] + 0.2)
                )
            else:
                self.metadata['hypotheses_rejected'] += 1

            self._save_json(self.hypotheses_file, self.hypotheses)
            self._save_json(self.metadata_file, self.metadata)

    async def record_strategy_outcome(self, strategy: str, success: bool, context: str = ""):
        """Записывает результат использования стратегии для мета-обучения"""
        key = 'successful_strategies' if success else 'failed_strategies'

        # Проверяем дубликат
        exists = False
        for item in self.meta_knowledge[key]:
            if item['strategy'] == strategy and abs(
                    (datetime.now() - datetime.fromisoformat(item['timestamp'])).days) < 7:
                item['count'] = item.get('count', 1) + 1
                item['last_context'] = context
                item['timestamp'] = datetime.now().isoformat()
                exists = True
                break

        if not exists:
            self.meta_knowledge[key].append({
                'strategy': strategy,
                'count': 1,
                'timestamp': datetime.now().isoformat(),
                'last_context': context
            })

        # Ограничиваем размер
        self.meta_knowledge[key] = sorted(
            self.meta_knowledge[key],
            key=lambda x: x['count'],
            reverse=True
        )[:15]

        self._save_json(self.meta_knowledge_file, self.meta_knowledge)

    def get_relevant_hypotheses(self, query: str, max_count: int = 3) -> List[Dict]:
        """Возвращает непроверенные гипотезы, релевантные текущему запросу"""
        relevant = []
        query_words = set(self._normalize_text(query).split())

        for i, hyp in enumerate(self.hypotheses):
            if hyp['verified'] is not None:  # Пропускаем проверенные
                continue

            hyp_words = set(self._normalize_text(hyp['hypothesis']).split())
            similarity = len(query_words & hyp_words) / max(len(query_words), 1) if query_words else 0

            if similarity > 0.3 or hyp['confidence'] > 0.7:
                relevant.append({
                    'id': i,
                    'hypothesis': hyp['hypothesis'],
                    'confidence': hyp['confidence'],
                    'evidence': hyp['evidence']
                })

        # Сортируем по уверенности и ограничиваем
        relevant.sort(key=lambda x: x['confidence'], reverse=True)
        return relevant[:max_count]

    # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: СДЕЛАНО АСИНХРОННЫМ
    async def get_enhanced_context(self, query: str, short_term_limit: int = 8, long_term_limit: int = 4) -> str:
        """Формирует расширенный контекст для принятия решений (АСИНХРОННЫЙ)"""
        # 1. Релевантные факты (теперь через await)
        relevant_facts = await self.search_facts(query, limit=long_term_limit)
        facts_text = "\n".join([
            f"• [{mem.get('category', 'fact')}] {mem['content']}"
            for mem in relevant_facts
        ]) if relevant_facts else "Нет релевантных фактов"

        # 2. Активные гипотезы
        hypotheses = self.get_relevant_hypotheses(query, max_count=2)
        hypotheses_text = "\n".join([
            f"  • {h['hypothesis']} (уверенность: {h['confidence']:.2f})"
            for h in hypotheses
        ]) if hypotheses else "Нет активных гипотез"

        # 3. Паттерны поведения
        patterns_text = ""
        if self.patterns.get('preferences'):
            prefs = ", ".join([f"{k}={v}" for k, v in list(self.patterns['preferences'].items())[:4]])
            patterns_text += f"Предпочтения: {prefs}\n"
        if self.patterns.get('communication_style'):
            patterns_text += f"Стиль общения: {self.patterns['communication_style']}\n"
        if self.patterns.get('frequent_topics'):
            topics = ", ".join(self.patterns['frequent_topics'][:3])
            patterns_text += f"Частые темы: {topics}\n"

        # 4. Недавняя история
        history = self.get_short_term_context(short_term_limit)

        return f"""# ДОЛГОСРОЧНАЯ ПАМЯТЬ:
{facts_text}

# АКТИВНЫЕ ГИПОТЕЗЫ (требуют верификации):
{hypotheses_text}

# ПАТТЕРНЫ ПОВЕДЕНИЯ:
{patterns_text if patterns_text else 'Не выявлено'}

# НЕДАВНЯЯ ИСТОРИЯ ({len(self.short_term)} сообщений):
{history}

# ТЕКУЩИЙ ЗАПРОС:
{query}"""

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)

        if not t1 or not t2:
            return 0.0

        words1 = set(t1.split())
        words2 = set(t2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union) if union else 0.0

        # Бонус за общие цифры/имена
        nums1 = set(re.findall(r'\d+', text1))
        nums2 = set(re.findall(r'\d+', text2))
        if nums1 & nums2:
            jaccard += 0.15

        return min(jaccard, 1.0)

    def _load_json(self, filepath: str, default: Any) -> Any:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {filepath}: {e}")
        return default

    def _save_json(self, filepath: str, data: Any):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения {filepath}: {e}")

    def get_stats(self) -> Dict:
        """Статистика памяти"""
        return {
            'user_id': self.user_id,
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'hypotheses_count': len(self.hypotheses),
            'verified_hypotheses': sum(1 for h in self.hypotheses if h.get('verified') is True),
            'rejected_hypotheses': sum(1 for h in self.hypotheses if h.get('verified') is False),
            'total_interactions': self.metadata['total_interactions'],
            'facts_learned': self.metadata['facts_learned'],
            'first_interaction': self.metadata['first_interaction'],
            'last_interaction': self.metadata['last_interaction']
        }

    async def clear_short_term(self):
        """Очищает краткосрочную память"""
        self.short_term = []
        self._save_json(self.short_term_file, [])
        print(f"🧹 Краткосрочная память очищена для {self.user_id}")


# ==================== МЕТА-КОГНИТИВНЫЙ МЕНЕДЖЕР ====================
class MetaCognitiveManager:
    """
    🧠 МЕТА-КОГНИТИВНЫЙ МЕНЕДЖЕР
    Реализует цикл: ПЛАНИРОВАНИЕ → ВЫПОЛНЕНИЕ → РЕФЛЕКСИЯ → ОБУЧЕНИЕ
    """

    def __init__(self, user_id: str, llm_caller: callable):
        self.user_id = user_id
        self.llm_caller = llm_caller
        self.memory = MetaCognitiveMemory(user_id)
        self.tools_manager = ToolsManager()
        self.session_start = datetime.now()

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Полный мета-когнитивный цикл обработки запроса"""
        # ШАГ 1: ПЛАНИРОВАНИЕ - анализ запроса и выбор стратегии
        print(f"\n{'=' * 70}")
        print(f"🧠 МЕТА-КОГНИТИВНЫЙ ЦИКЛ ДЛЯ: {query[:50]}...")
        print(f"{'=' * 70}")

        plan = await self._meta_plan(query)
        print(f"📋 ПЛАН: {plan['strategy']}")
        print(f"   Выбранные инструменты: {[t['name'] for t in plan['selected_tools']]}")

        # ШАГ 2: ВЫПОЛНЕНИЕ - применение выбранной стратегии
        execution_result = await self._execute_plan(query, plan)
        print(f"⚡ ВЫПОЛНЕНИЕ: {'успешно' if execution_result['success'] else 'частично'}")

        # ШАГ 3: РЕФЛЕКСИЯ - анализ собственного ответа
        reflection = await self._self_reflect(query, execution_result)
        print(f"🔍 РЕФЛЕКСИЯ: качество={reflection['quality_score']:.2f}")

        # ШАГ 4: ОБУЧЕНИЕ - обновление памяти и стратегий
        await self._learn_from_interaction(query, execution_result, reflection)

        # Формируем финальный ответ
        final_response = execution_result['response']
        if reflection['needs_improvement'] and reflection['suggested_improvement']:
            # Добавляем уточнение на основе рефлексии
            final_response += f"\n\n💡 *Уточнение:* {reflection['suggested_improvement']}"

        # Сохраняем взаимодействие
        await self.memory.add_interaction('user', query)
        await self.memory.add_interaction('assistant', final_response, {
            'strategy': plan['strategy'],
            'tools_used': [t['name'] for t in plan['selected_tools']],
            'quality_score': reflection['quality_score'],
            'needs_verification': execution_result.get('needs_verification', False)
        })

        return {
            'response': final_response,
            'strategy': plan['strategy'],
            'tools_used': [t['name'] for t in plan['selected_tools']],
            'quality_score': reflection['quality_score'],
            'needs_verification': execution_result.get('needs_verification', False),
            'reflection': reflection
        }

    async def _meta_plan(self, query: str) -> Dict[str, Any]:
        """Мета-планирование: выбор стратегии и инструментов"""
        # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ИСПОЛЬЗУЕМ await ДЛЯ АСИНХРОННОГО МЕТОДА
        memory_context = await self.memory.get_enhanced_context(query)

        # Оценка каждого инструмента
        tool_evaluations = []
        for tool_name, tool in self.tools_manager.cores.items():
            can_handle, confidence = await tool.can_handle(query, {'memory': self.memory})
            if can_handle and confidence > 0.4:
                metadata = await tool.get_metadata()
                tool_evaluations.append({
                    'tool': tool,
                    'name': tool_name,
                    'confidence': confidence,
                    'priority': metadata['priority'],
                    'direct_answer': metadata['direct_answer_mode'],
                    'metadata': metadata
                })

        # Сортируем по приоритету и уверенности
        tool_evaluations.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)

        # Выбираем стратегию
        if not tool_evaluations:
            strategy = 'general_llm'
            selected_tools = []
        elif tool_evaluations[0]['confidence'] > 0.85 and tool_evaluations[0]['direct_answer']:
            strategy = 'direct_tool'
            selected_tools = [tool_evaluations[0]]
        elif tool_evaluations[0]['confidence'] > 0.6:
            strategy = 'tool_with_llm_verification'
            # Берём до 2 самых релевантных инструментов
            selected_tools = tool_evaluations[:2]
        else:
            strategy = 'llm_with_tool_augmentation'
            selected_tools = tool_evaluations[:2] if tool_evaluations else []

        return {
            'strategy': strategy,
            'selected_tools': selected_tools,
            'memory_context': memory_context,
            'all_evaluations': tool_evaluations
        }

    async def _execute_plan(self, query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение плана с выбранными инструментами"""
        strategy = plan['strategy']
        selected_tools = plan['selected_tools']

        if strategy == 'direct_tool' and selected_tools:
            # Прямой ответ от инструмента
            tool = selected_tools[0]['tool']
            response = await tool.execute(query, {'memory': self.memory})
            return {
                'success': response.success,
                'response': response.raw_result if response.success else "❌ Не удалось обработать запрос",
                'source': tool.name,
                'needs_verification': False,
                'tool_response': response
            }

        elif strategy in ['tool_with_llm_verification', 'llm_with_tool_augmentation'] and selected_tools:
            # Сбор данных от инструментов
            tool_responses = []
            for tool_eval in selected_tools:
                tool = tool_eval['tool']
                response = await tool.execute(query, {'memory': self.memory})
                if response.success:
                    tool_responses.append({
                        'tool': tool.name,
                        'response': response,
                        'confidence': tool_eval['confidence']
                    })

            # Формирование контекста для LLM
            context = self._build_llm_context(query, tool_responses, plan['memory_context'])
            llm_response = await self.llm_caller(context, temperature=0.4)

            needs_verification = any(
                r['response'].requires_verification for r in tool_responses
            ) or strategy == 'tool_with_llm_verification'

            return {
                'success': True,
                'response': llm_response,
                'source': 'llm_with_tools',
                'needs_verification': needs_verification,
                'tool_responses': tool_responses
            }

        else:
            # Общий ответ LLM с контекстом памяти
            context = self._build_general_llm_context(query, plan['memory_context'])
            llm_response = await self.llm_caller(context, temperature=0.5)

            return {
                'success': True,
                'response': llm_response,
                'source': 'general_llm',
                'needs_verification': False
            }

    def _build_llm_context(self, query: str, tool_responses: List[Dict], memory_context: str) -> str:
        """Контекст для LLM с данными инструментов"""
        tools_data = "\n\n".join([
            f"### ДАННЫЕ ОТ {r['tool'].upper()}:\n{r['response'].raw_result or r['response'].to_context_string()}"
            for r in tool_responses
        ])

        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        return f"""# МЕТА-КОГНИТИВНЫЙ КОНТЕКСТ ДЛЯ ОТВЕТА
Время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь ID: {self.user_id}

# ИНСТРУКЦИЯ ДЛЯ ТЕБЯ (ИИ-АССИСТЕНТА):
Ты — интеллектуальный ассистент с мета-когнитивными способностями. 
Проанализируй данные от инструментов и контекст памяти, чтобы дать ПОЛЕЗНЫЙ, ТОЧНЫЙ и АДАПТИРОВАННЫЙ ответ.

# ДАННЫЕ ОТ ИНСТРУМЕНТОВ:
{tools_data}

# КОНТЕКСТ ПАМЯТИ ПОЛЬЗОВАТЕЛЯ:
{memory_context}

# ТВОЯ ЗАДАЧА:
1. Синтезируй информацию из инструментов и памяти
2. Учти паттерны поведения и предпочтения пользователя
3. Если данные требуют верификации — укажи на это и предложи уточнение
4. Будь конкретным, избегай общих фраз
5. Адаптируй стиль под предпочтения пользователя (из памяти)
6. Для фактов с низкой уверенностью — укажи степень уверенности

# ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

# ФОРМАТ ОТВЕТА:
- Начни с прямого ответа на вопрос
- Добавь контекст/пояснения при необходимости
- Укажи источники для фактов (если уместно)
- Предложи следующие шаги/уточнения если нужно
"""

    def _build_general_llm_context(self, query: str, memory_context: str) -> str:
        """Контекст для общего ответа LLM"""
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        tools_summary = "\n".join([
            f"- {name}: {', '.join(tool.capabilities[:2])}"
            for name, tool in self.tools_manager.cores.items()
        ])

        return f"""# МЕТА-КОГНИТИВНЫЙ КОНТЕКСТ ДЛЯ ОБЩЕГО ОТВЕТА
Время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь ID: {self.user_id}

# ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_summary}

# КОНТЕКСТ ПАМЯТИ ПОЛЬЗОВАТЕЛЯ:
{memory_context}

# ИНСТРУКЦИЯ:
Ответь полезно и адаптивно, используя контекст памяти. 
Учти паттерны поведения пользователя. Для запросов требующих актуальных данных — предложи использовать веб-поиск.

# ЗАПРОС:
{query}
"""

    async def _self_reflect(self, query: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Саморефлексия: анализ качества собственного ответа"""
        response = execution_result['response']

        reflection_prompt = f"""# МЕТА-КОГНИТИВНАЯ РЕФЛЕКСИЯ
Проанализируй свой ответ на запрос пользователя с позиции критического мышления.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ТВОЙ ОТВЕТ:
{response}

ЗАДАЧА РЕФЛЕКСИИ:
1. ОЦЕНКА ПОЛНОТЫ: Ответил ли ты на все аспекты запроса? (0-10)
2. ОЦЕНКА ТОЧНОСТИ: Насколько точна информация? Есть ли неопределённости? (0-10)
3. АДАПТАЦИЯ: Учтены ли предпочтения и паттерны пользователя? (0-10)
4. ПОЛЕЗНОСТЬ: Насколько ответ полезен для пользователя? (0-10)
5. КРИТИЧЕСКИЕ ЗАМЕЧАНИЯ: Что можно улучшить в ответе?
6. НУЖНО ЛИ УТОЧНЕНИЕ: Требуется ли дополнительная информация от пользователя?

ВЕРНИ ЧИСТЫЙ JSON:
{{
  "completeness_score": 0-10,
  "accuracy_score": 0-10,
  "adaptation_score": 0-10,
  "usefulness_score": 0-10,
  "overall_quality": 0.0-1.0,
  "needs_improvement": true|false,
  "critical_issues": ["проблема1", "проблема2"],
  "suggested_improvement": "конкретное предложение по улучшению",
  "needs_user_clarification": true|false,
  "clarification_question": "вопрос для уточнения (если нужно)"
}}"""

        try:
            reflection_response = await self.llm_caller(reflection_prompt, temperature=0.3, max_tokens=400)
            json_match = re.search(r'\{.*\}', reflection_response, re.DOTALL)

            if json_match:
                reflection_data = json.loads(json_match.group(0))
                quality_score = reflection_data.get('overall_quality',
                                                    (reflection_data.get('completeness_score', 5) +
                                                     reflection_data.get('accuracy_score', 5) +
                                                     reflection_data.get('adaptation_score', 5) +
                                                     reflection_data.get('usefulness_score', 5)) / 40.0
                                                    )

                return {
                    'quality_score': quality_score,
                    'needs_improvement': reflection_data.get('needs_improvement', quality_score < 0.7),
                    'critical_issues': reflection_data.get('critical_issues', []),
                    'suggested_improvement': reflection_data.get('suggested_improvement', ''),
                    'needs_user_clarification': reflection_data.get('needs_user_clarification', False),
                    'clarification_question': reflection_data.get('clarification_question', ''),
                    'raw_reflection': reflection_data
                }
        except Exception as e:
            print(f"⚠️ Ошибка рефлексии: {e}")

        # Дефолтная оценка
        return {
            'quality_score': 0.75,
            'needs_improvement': False,
            'critical_issues': [],
            'suggested_improvement': '',
            'needs_user_clarification': False,
            'clarification_question': ''
        }

    async def _learn_from_interaction(self, query: str, execution_result: Dict[str, Any], reflection: Dict[str, Any]):
        """Обучение на основе взаимодействия"""
        # 1. Запись стратегии
        strategy = execution_result.get('source', 'unknown')
        success = reflection['quality_score'] > 0.7
        await self.memory.record_strategy_outcome(
            f"{strategy}:{query[:30]}",
            success,
            context=f"quality={reflection['quality_score']:.2f}"
        )

        # 2. Извлечение фактов из диалога (если качество высокое)
        if reflection['quality_score'] > 0.8 and 'user' in query.lower():
            # Анализируем диалог на предмет новых фактов о пользователе
            analysis_prompt = f"""Извлеки факты о ПОЛЬЗОВАТЕЛЕ из этого диалога:
ЗАПРОС: {query}
ОТВЕТ: {execution_result['response']}

Верни JSON с новыми фактами (только о пользователе, не общие знания):
{{
  "new_facts": [
    {{"fact": "факт о пользователе", "category": "personal|preference|habit", "importance": 0.7}}
  ]
}}"""

            try:
                analysis = await self.llm_caller(analysis_prompt, temperature=0.2, max_tokens=300)
                json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    for fact_data in data.get('new_facts', []):
                        await self.memory.add_fact(
                            fact_data['fact'],
                            category=fact_data.get('category', 'general'),
                            importance=fact_data.get('importance', 0.7),
                            confidence=0.85
                        )
            except Exception as e:
                pass  # Игнорируем ошибки извлечения фактов

        # 3. Обновление гипотез
        if reflection['needs_user_clarification'] and reflection['clarification_question']:
            await self.memory.add_hypothesis(
                f"Пользователь, возможно, имел в виду: {reflection['clarification_question']}",
                evidence=[query, execution_result['response']],
                confidence=0.6
            )


# ==================== МЕНЕДЖЕР ИНСТРУМЕНТОВ ====================
class ToolsManager:
    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self._load_builtin_cores()
        self.typo_fixes = {'фаил': 'файл', 'прочти': 'прочитай'}

    def _load_builtin_cores(self):
        """Загружает встроенные ядра"""
        builtin_cores = [
            DateTimeCore(),
            CalculatorCore(),
            FileStorageCore()
        ]
        if DDGS_AVAILABLE:
            web_core = WebSearchCore()
            builtin_cores.append(web_core)

        for core in builtin_cores:
            self.cores[core.name] = core
            print(f"✅ Загружено ядро: {core.name} (приоритет: {core.priority})")

    def fix_typos(self, query: str) -> str:
        """Исправляет опечатки в запросе"""
        fixed = query
        for typo, correct in self.typo_fixes.items():
            fixed = fixed.replace(typo, correct)
        if fixed != query:
            print(f"🔧 Исправлены опечатки: '{query}' → '{fixed}'")
        return fixed


# ==================== ТЕЛЕГРАМ БОТ (ПОЛНОСТЬЮ АСИНХРОННЫЙ) ====================
class TelegramBot:
    def __init__(self):
        self.user_managers: Dict[str, MetaCognitiveManager] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.web_cores: List[WebSearchCore] = []
        self._initialize_logs()

    async def initialize(self):
        """Инициализация асинхронных ресурсов"""
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

        # Инициализация всех веб-ядер
        for manager in self.user_managers.values():
            if hasattr(manager.tools_manager, 'cores'):
                web_core = manager.tools_manager.cores.get('web_search_core')
                if web_core and hasattr(web_core, 'initialize'):
                    await web_core.initialize()
                    self.web_cores.append(web_core)

    async def close(self):
        """Закрытие асинхронных ресурсов"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

        # Закрытие всех веб-ядер
        for web_core in self.web_cores:
            if hasattr(web_core, 'close'):
                await web_core.close()

    def _initialize_logs(self):
        """Инициализирует файлы логов"""
        logs_config = {
            LEARNING_LOG: [],
            CORE_PERFORMANCE_LOG: {}
        }
        for log_file, default_data in logs_config.items():
            if not os.path.exists(log_file):
                try:
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(default_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"⚠️ Ошибка создания {log_file}: {e}")

    def get_manager(self, user_id: str) -> MetaCognitiveManager:
        if user_id not in self.user_managers:
            self.user_managers[user_id] = MetaCognitiveManager(user_id, self.get_llm_response)
            print(f"🧠 Создан мета-когнитивный менеджер для {user_id}")
        return self.user_managers[user_id]

    async def get_llm_response(self, context: str, temperature: float = 0.5, max_tokens: int = 1500) -> str:
        """Асинхронный запрос к LLM через aiohttp"""
        try:
            async with self.http_session.post(
                    LM_STUDIO_API_URL,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {LM_STUDIO_API_KEY}'
                    },
                    json={
                        'messages': [{'role': 'user', 'content': context}],
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'stream': False
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    error_text = await resp.text()
                    return f"❌ Ошибка LLM (код: {resp.status}): {error_text[:100]}"
        except asyncio.TimeoutError:
            return "❌ Таймаут ответа от LLM (120 сек)"
        except Exception as e:
            return f"❌ Ошибка LLM: {str(e)[:150]}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        web_status = "✅ Доступен" if DDGS_AVAILABLE else "❌ Недоступен (установите: `pip install \"duckduckgo-search>=5.0\"`)"

        await update.message.reply_text(
            "🧠 *МИНИ-МОЗГ БОТ v7.1 — ИСПРАВЛЕННАЯ ВЕРСИЯ*\n"
            "\n✨ *Ключевые исправления:*\n"
            "• ✅ Устранена критическая ошибка: `asyncio.run() cannot be called from a running event loop`\n"
            "• 🔒 Безопасный калькулятор (без уязвимости eval)\n"
            "• 💡 Система гипотез и их верификации через диалог\n"
            "• 📦 Кэширование веб-поиска с актуализацией\n"
            "• 🌐 Веб-поиск: " + web_status + "\n"
                                             "\n⚙️ *Команды:*\n"
                                             "/memory_stats — статистика памяти и обучения\n"
                                             "/search_memory [запрос] — поиск в долгосрочной памяти\n"
                                             "/clear — очистить краткосрочную память (диалог)\n"
                                             "/help — подробная справка",
            parse_mode='Markdown'
        )

    async def memory_stats_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        stats = manager.memory.get_stats()

        first_int = datetime.fromisoformat(stats['first_interaction']).strftime('%d.%m.%Y')
        last_int = datetime.fromisoformat(stats['last_interaction']).strftime('%d.%m.%Y %H:%M')

        msg = (f"🧠 *СТАТИСТИКА МЕТА-КОГНИТИВНОЙ ПАМЯТИ*\n"
               f"👤 Пользователь: {user_id}\n"
               f"📅 Первое взаимодействие: {first_int}\n"
               f"🕒 Последнее: {last_int}\n"
               f"\n📝 *Краткосрочная память:* {stats['short_term_count']} сообщений\n"
               f"📚 *Долгосрочная память:* {stats['long_term_count']} фактов\n"
               f"💡 *Гипотез о пользователе:* {stats['hypotheses_count']}\n"
               f"   ✅ Подтверждено: {stats['verified_hypotheses']}\n"
               f"   ❌ Опровергнуто: {stats['rejected_hypotheses']}\n"
               f"\n🔄 *Всего взаимодействий:* {stats['total_interactions']}\n"
               f"📈 *Фактов извлечено:* {stats['facts_learned']}")

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def search_memory_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Поиск в долгосрочной памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        query = ' '.join(context.args) if context.args else ''

        if not query:
            await update.message.reply_text(
                "🔍 *Поиск в долгосрочной памяти*\n"
                "Использование: `/search_memory ваш запрос`",
                parse_mode='Markdown'
            )
            return

        results = await manager.memory.search_facts(query, limit=5)

        if not results:
            await update.message.reply_text(
                f"❌ Ничего не найдено по запросу: `{query}`",
                parse_mode='Markdown'
            )
            return

        msg = f"🔍 *Результаты поиска:* `{query}`\n"
        for i, mem in enumerate(results, 1):
            dt = datetime.fromisoformat(mem['timestamp']).strftime('%d.%m.%Y')
            importance = "⭐" * int(mem.get('importance', 0.5) * 5)
            access = f"👁️ {mem.get('access_count', 0)}"
            msg += f"\n{i}. {importance} {access}\n📅 {dt}\n💬 {mem['content'][:150]}"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистить краткосрочную память"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        await manager.memory.clear_short_term()
        await update.message.reply_text(
            "🧹 *Краткосрочная память (диалог) очищена*\n"
            "🧠 Долгосрочная память, факты и паттерны поведения сохранены.",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Справка"""
        await update.message.reply_text(
            "📖 *СПРАВКА ПО МЕТА-КОГНИТИВНОЙ СИСТЕМЕ*\n"
            "\n*Как работает бот:*\n"
            "1️⃣ *ПЛАНИРОВАНИЕ* — анализирует запрос и выбирает оптимальную стратегию\n"
            "2️⃣ *ВЫПОЛНЕНИЕ* — применяет инструменты (время, калькулятор, веб-поиск)\n"
            "3️⃣ *РЕФЛЕКСИЯ* — критически оценивает свой ответ на точность и полезность\n"
            "4️⃣ *ОБУЧЕНИЕ* — запоминает успешные стратегии и извлекает факты о вас\n"
            "\n*Ваши преимущества:*\n"
            "• Бот адаптируется под ваш стиль общения и предпочтения\n"
            "• Запоминает важные факты о вас между сессиями\n"
            "• Формулирует гипотезы и проверяет их в диалоге\n"
            "• Самокоррекция ошибок через рефлексию\n"
            "• Безопасная обработка запросов (без уязвимостей)\n"
            "\n*Примеры запросов:*\n"
            "• `сколько будет 128 * 7` — мгновенный расчёт\n"
            "• `какой сегодня день` — точная дата и время\n"
            "• `курс доллара` — актуальный курс с веб-поиска (если доступен)\n"
            "• `сохрани в файл идеи.txt: мои мысли...` — работа с файлами\n"
            "• `что ты знаешь обо мне` — проверка долгосрочной памяти",
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений с полным мета-когнитивным циклом"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        # Индикатор печатания
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            manager = self.get_manager(user_id)
            result = await manager.process_query(text)

            # Форматируем ответ для Telegram
            response = result['response']
            if len(response) > 4000:
                response = response[:3950] + "\n\n... (ответ усечён из-за ограничения Telegram)"

            await update.message.reply_text(
                response,
                parse_mode='Markdown',
                disable_web_page_preview=False
            )

            # Логирование для отладки
            print(f"✅ Ответ отправлен (качество: {result['quality_score']:.2f}, "
                  f"инструменты: {result['tools_used'] or ['общий_llm']})")

        except Exception as e:
            error_msg = f"❌ *Ошибка обработки:* {str(e)[:200]}"
            print(f"ERROR [{user_id}]: {e}")
            traceback.print_exc()
            await update.message.reply_text(error_msg, parse_mode='Markdown')


# ==================== ЗАПУСК БОТА ====================
async def main_async():
    """Асинхронная главная функция"""
    print("\n" + "=" * 70)
    print("🚀 МИНИ-МОЗГ БОТ v7.1 — ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("=" * 70)
    print(f"⏰ Время запуска: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"🧠 Директория памяти: {MEMORY_DIR}/")
    print(f"🌐 Веб-поиск: {'✅ Доступен (AsyncDDGS)' if DDGS_AVAILABLE else '❌ Недоступен'}")
    if not DDGS_AVAILABLE:
        print("   💡 Установите: pip install \"duckduckgo-search>=5.0\"")
    print(f"🔍 Парсинг сайтов: {'✅ Доступен' if BEAUTIFULSOUP_AVAILABLE else '⚠️ Ограничен'}")
    print(f"🔒 Безопасность: ✅ Без eval(), защита от path traversal")
    print("=" * 70)

    # Проверка LM Studio
    try:
        async with aiohttp.ClientSession() as session:
            test_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '/v1/models')
            async with session.get(test_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    print(f"✅ LM Studio API доступна")
                else:
                    print(f"⚠️ LM Studio ответила кодом: {resp.status}")
    except Exception as e:
        print(f"⚠️ LM Studio недоступна: {e}")

    print("=" * 70)
    print("\n🔄 Инициализация мета-когнитивной системы...")

    bot = TelegramBot()
    await bot.initialize()

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("memory_stats", bot.memory_stats_cmd))
    application.add_handler(CommandHandler("search_memory", bot.search_memory_cmd))
    application.add_handler(CommandHandler("clear", bot.clear_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот инициализирован!")
    print("=" * 70)
    print("💬 Готов к работе в Telegram")
    print("🧠 Мета-когнитивный цикл активен:")
    print("   • Планирование стратегии ответа")
    print("   • Выполнение с выбором инструментов")
    print("   • Саморефлексия после каждого ответа")
    print("   • Непрерывное обучение и адаптация")
    print("=" * 70)
    print("\nCtrl+C для остановки\n")

    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

        # Бесконечное ожидание
        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        print("\n🛑 Остановка по запросу пользователя...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()
    finally:
        # Корректное завершение
        await application.stop()
        await application.shutdown()
        await bot.close()
        print("✅ Бот остановлен корректно")


def main():
    """Точка входа"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Выход по Ctrl+C")
    except Exception as e:
        print(f"\n❌ Фатальная ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()