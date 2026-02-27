#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v4.2 - УЛУЧШЕННАЯ ВЕРСИЯ
✅ Исправлены все критические баги
✅ Улучшена стабильность работы
✅ Оптимизирована работа с файлами
✅ Улучшена обработка ошибок
"""

import os
import json
import re
import ast
import asyncio
import requests
import traceback
import hashlib
import time
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# DuckDuckGo Search импорт
try:
    from duckduckgo_search import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    try:
        from ddgs import DDGS

        DDGS_AVAILABLE = True
    except ImportError:
        print("⚠️ DuckDuckGo search не установлен. Веб-поиск будет недоступен.")
        DDGS_AVAILABLE = False

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

# Создаем директории
for directory in [CORES_DIR, MEMORY_DIR, USER_FILES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Файлы
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
CORE_PERFORMANCE_LOG = os.path.join(MEMORY_DIR, "core_performance.json")
REJECTED_CORES_LOG = os.path.join(MEMORY_DIR, "rejected_cores.json")

# Создаем __init__.py для импортов
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
            direct_answer: bool = False
    ):
        self.success = success
        self.data = data or {}
        self.raw_result = raw_result
        self.confidence = confidence
        self.source = source
        self.direct_answer = direct_answer

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
        return self.direct_answer and self.success and self.raw_result and len(self.raw_result) > 10


# ==================== БАЗОВЫЙ КЛАСС ЯДРА ====================
class KnowledgeCore(ABC):
    """Базовый класс для всех ядер знаний"""
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []
    priority: int = 5
    direct_answer_mode: bool = False

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        pass

    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        pass

    def get_confidence(self, query: str) -> float:
        """Уверенность в обработке запроса (0.0-1.0)"""
        return 0.5 if self.can_handle(query) else 0.0

    def get_description(self) -> str:
        """Полное описание возможностей"""
        caps = ', '.join(self.capabilities[:3]) if self.capabilities else 'нет описания'
        return f"{self.description} ({caps})"


# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
    name = "datetime_core"
    description = "Точная информация о дате, времени и днях недели"
    capabilities = ["текущая дата", "время", "день недели", "расчёт дат"]
    priority = 1
    direct_answer_mode = True

    def can_handle(self, query: str) -> bool:
        q = query.lower().replace('фаил', 'файл')
        keywords = [
            'какой сегодня день', 'какое число', 'день недели', 'сколько времени',
            'который час', 'текущая дата', 'сегодня', 'завтра', 'послезавтра',
            'вчера', 'какой сейчас день', 'какая дата'
        ]
        return any(kw in q for kw in keywords)

    def get_confidence(self, query: str) -> float:
        q = query.lower().replace('фаил', 'файл')
        high_conf = ['какой сегодня', 'какое число', 'который час', 'сколько времени', 'какая дата']
        if any(kw in q for kw in high_conf):
            return 0.95
        return 0.7 if self.can_handle(query) else 0.0

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
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
                confidence=0.95,
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


class CalculatorCore(KnowledgeCore):
    name = "calculator_core"
    description = "Математические вычисления"
    capabilities = ["сложение", "вычитание", "умножение", "деление", "проценты"]
    priority = 2
    direct_answer_mode = True

    def can_handle(self, query: str) -> bool:
        q = query.lower().replace('фаил', 'файл')
        has_math = bool(re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', q))
        has_words = any(word in q for word in ['сколько будет', 'посчитай', 'вычисли', 'равно', 'реши'])
        return has_math or (has_words and any(char.isdigit() for char in q))

    def get_confidence(self, query: str) -> float:
        q = query.lower().replace('фаил', 'файл')
        if re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', q):
            return 0.9
        if 'сколько будет' in q and any(char.isdigit() for char in q):
            return 0.8
        return 0.6 if self.can_handle(query) else 0.0

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            # Извлекаем математическое выражение
            expr = re.sub(r'[^\d\+\-\*\/x×÷\.\(\)\s]', '', query.lower())
            expr = expr.replace('x', '*').replace('×', '*').replace('÷', '/').replace(' ', '')

            if not expr:
                match = re.search(r'сколько будет\s+([\d\+\-\*\/\.\(\)]+)', query.lower())
                if match:
                    expr = match.group(1).replace(' ', '')
                    expr = expr.replace('x', '*').replace('×', '*').replace('÷', '/')
                else:
                    raise ValueError("Не найдено математическое выражение")

            if not expr or any(danger in expr for danger in ['import', 'exec', 'eval', '__']):
                raise ValueError("Недопустимое выражение")

            # Безопасное вычисление
            result = eval(expr, {"__builtins__": {}}, {})

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
                confidence=0.9,
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


class WebSearchCore(KnowledgeCore):
    name = "web_search_core"
    description = "Поиск актуальной информации в интернете"
    capabilities = ["новости", "курсы валют", "погода", "события", "актуальная информация"]
    priority = 8

    def __init__(self):
        self.ddgs = DDGS() if DDGS_AVAILABLE else None
        self.cache = {}

    def can_handle(self, query: str) -> bool:
        if not DDGS_AVAILABLE:
            return False

        q = query.lower().replace('фаил', 'файл')

        exclude_keywords = [
            'какой сегодня день', 'какое число', 'который час', 'сколько времени',
            'сколько будет', 'умножить', 'разделить', 'прочитай файл',
            'сохрани файл', 'список файлов', 'фаил', 'файл'
        ]

        if any(kw in q for kw in exclude_keywords):
            return False

        search_keywords = [
            'новост', 'курс', 'погод', 'прогноз', 'событи', 'произошло',
            'последни', 'актуальн', 'сейчас', 'сегодняшн', 'текущ',
            'кто выиграл', 'результат', 'итоги', 'цена', 'стоимость',
            'информаци', 'узнай', 'найди', 'что такое', 'кто такой'
        ]
        return any(kw in q for kw in search_keywords)

    def get_confidence(self, query: str) -> float:
        if not DDGS_AVAILABLE:
            return 0.0

        q = query.lower().replace('фаил', 'файл')
        high_priority = ['курс доллара', 'новости сегодня', 'погода сейчас', 'что случилось']
        if any(kw in q for kw in high_priority):
            return 0.85
        return 0.6 if self.can_handle(query) else 0.0

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        if not DDGS_AVAILABLE or not self.ddgs:
            return CoreResponse(
                success=False,
                data={'error': 'Веб-поиск недоступен'},
                confidence=0.0,
                source=self.name
            )

        try:
            print(f"  🌐 Веб-поиск: {query[:60]}...")

            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]

            if query_hash in self.cache:
                if time.time() - self.cache[query_hash]['timestamp'] < 300:
                    results = self.cache[query_hash]['results']
                else:
                    del self.cache[query_hash]
                    results = list(self.ddgs.text(query, max_results=5))
            else:
                results = list(self.ddgs.text(query, max_results=5))
                self.cache[query_hash] = {
                    'results': results,
                    'timestamp': time.time()
                }

            if not results:
                return CoreResponse(
                    success=False,
                    data={'error': 'Результаты не найдены', 'query': query},
                    confidence=0.0,
                    source=self.name
                )

            search_results = []
            for idx, r in enumerate(results[:4], 1):
                search_results.append({
                    'position': idx,
                    'title': r.get('title', '').strip(),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '').strip()[:250]
                })

            results_text = "🌐 **Результаты поиска в интернете:**\n\n"
            for r in search_results:
                results_text += f"{r['position']}. **{r['title']}**\n"
                if r['snippet']:
                    results_text += f"   _{r['snippet']}_\n"
                results_text += f"   🔗 {r['url']}\n\n"

            return CoreResponse(
                success=True,
                data={'query': query, 'results': search_results, 'results_count': len(search_results)},
                raw_result=results_text,
                confidence=0.8,
                source=self.name
            )

        except Exception as e:
            print(f"  ⚠️ Ошибка веб-поиска: {e}")
            return CoreResponse(
                success=False,
                data={'error': str(e), 'query': query},
                confidence=0.0,
                source=self.name
            )


# ==================== ФАЙЛОВОЕ ЯДРО ====================
class FileStorageCore(KnowledgeCore):
    name = "file_storage_core"
    description = "Работа с текстовыми файлами"
    capabilities = ["сохранить файл", "прочитать файл", "список файлов", "удалить файл"]
    priority = 1
    direct_answer_mode = True

    def __init__(self):
        self.storage_dir = USER_FILES_DIR
        os.makedirs(self.storage_dir, exist_ok=True)

    def can_handle(self, query: str) -> bool:
        q = query.lower().replace('фаил', 'файл')
        return any(word in q for word in
                   ['файл', 'документ', 'сохрани', 'прочитай', 'прочти', 'открой', 'удали', 'список файлов'])

    def get_confidence(self, query: str) -> float:
        q = query.lower().replace('фаил', 'файл')
        if 'прочитай файл' in q or 'прочти файл' in q:
            return 0.95
        if 'сохрани в файл' in q:
            return 0.9
        if 'список файлов' in q:
            return 0.85
        return 0.8 if self.can_handle(query) else 0.0

    def _sanitize_filename(self, filename: str) -> str:
        """Очищает имя файла от опасных символов"""
        if not filename:
            return "unnamed.txt"

        filename = filename.strip().strip('\'"')
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        filename = ''.join(char for char in filename if char.isprintable())

        if not filename or filename.startswith('.'):
            filename = "document.txt"

        if '.' not in filename:
            filename += '.txt'

        return filename

    def _get_file_path(self, filename: str) -> str:
        """Безопасное получение пути к файлу"""
        clean_filename = self._sanitize_filename(filename)
        filepath = os.path.join(self.storage_dir, clean_filename)

        abs_storage = os.path.abspath(self.storage_dir)
        abs_filepath = os.path.abspath(filepath)

        if not abs_filepath.startswith(abs_storage):
            raise ValueError(f"Недопустимый путь к файлу: {filename}")

        return abs_filepath

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            q = query.lower().replace('фаил', 'файл')

            # ЧТЕНИЕ ФАЙЛА
            if 'прочитай' in q or 'прочти' in q or 'открой' in q:
                filename = self._extract_filename_for_read(query)

                if not filename:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Не указано имя файла'},
                        raw_result="❌ **Пожалуйста, укажите имя файла.**\n\nПримеры:\n• `прочитай файл привет.txt`\n• `прочти файл заметки.txt`",
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
                    return self._file_not_found_response(filename)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    file_size = os.path.getsize(filepath)
                    lines = content.count('\n') + 1

                    return CoreResponse(
                        success=True,
                        data={'filename': filename, 'content': content, 'size': file_size, 'lines': lines},
                        raw_result=f"📄 **Файл `{filename}`:**\n\n{content}\n\n📊 **Информация:**\n• Размер: {file_size} байт\n• Строк: {lines}",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )
                except UnicodeDecodeError:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Файл не в UTF-8'},
                        raw_result=f"❌ **Не могу прочитать файл `{filename}`.**\n\nФайл содержит бинарные данные или не в кодировке UTF-8.",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

            # СОХРАНЕНИЕ ФАЙЛА
            elif 'сохрани' in q or 'запиши' in q:
                match = re.search(r'сохрани\s+(?:в\s+)?файл\s+["\']?([^"\':]+)["\']?\s*[:：]\s*(.+)', query,
                                  re.IGNORECASE | re.DOTALL)
                if not match:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Неверный формат'},
                        raw_result="❌ **Неверный формат команды.**\n\n💡 **Правильный формат:**\n`сохрани в файл имя_файла: ваш текст`\n\n**Пример:**\n`сохрани в файл привет.txt: Привет, мир!`",
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
                        raw_result=f"❌ **Ошибка имени файла:** {str(e)}",
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
                        raw_result=f"✅ **Файл сохранен!**\n\n📄 **Имя:** `{filename}`\n📊 **Размер:** {file_size} байт\n📁 **Путь:** `{filepath}`\n\n📝 **Содержимое (первые 200 символов):**\n{preview}",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )
                except Exception as e:
                    return CoreResponse(
                        success=False,
                        data={'error': str(e)},
                        raw_result=f"❌ **Ошибка сохранения файла:** {str(e)}",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

            # СПИСОК ФАЙЛОВ
            elif 'список файлов' in q or 'мои файлы' in q or 'все файлы' in q or 'покажи файлы' in q:
                return self._get_files_list()

            # УДАЛЕНИЕ ФАЙЛА
            elif 'удали файл' in q or 'удалить файл' in q:
                return self._delete_file(query)

            # ПОМОЩЬ
            else:
                return self._get_help()

        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ **Ошибка работы с файлами:**\n\n{str(e)}",
                confidence=0.0,
                source=self.name,
                direct_answer=True
            )

    def _extract_filename_for_read(self, query: str) -> str:
        """Извлекает имя файла для чтения"""
        patterns = [
            r'прочитай файл\s+["\']?([^"\'\s]+)["\']?',
            r'прочти файл\s+["\']?([^"\'\s]+)["\']?',
            r'открой файл\s+["\']?([^"\'\s]+)["\']?',
            r'файл\s+["\']?([^"\'\s]+)["\']?\s+(?:прочитай|прочти|открой)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ['файл', 'фаил'] and i + 1 < len(words):
                return words[i + 1].strip('"\':.,!?')

        return ""

    def _file_not_found_response(self, filename: str) -> CoreResponse:
        """Ответ когда файл не найден"""
        files = []
        try:
            for f in os.listdir(self.storage_dir):
                if os.path.isfile(os.path.join(self.storage_dir, f)):
                    files.append(f)
        except Exception:
            files = []

        if files:
            file_list = "\n".join([f"• `{f}`" for f in files[:10]])
            return CoreResponse(
                success=False,
                data={'error': 'Файл не найден', 'available_files': files},
                raw_result=f"❌ **Файл `{filename}` не найден.**\n\n📁 **Доступные файлы:**\n{file_list}\n\n💡 **Чтобы создать файл:**\n`сохрани в файл {filename}: ваш текст`",
                confidence=0.0,
                source=self.name,
                direct_answer=True
            )
        else:
            return CoreResponse(
                success=False,
                data={'error': 'Файл не найден'},
                raw_result=f"❌ **Файл `{filename}` не найден.**\n\n📁 **Хранилище пусто.**\n\n💡 **Чтобы создать файл:**\n`сохрани в файл {filename}: ваш текст`",
                confidence=0.0,
                source=self.name,
                direct_answer=True
            )

    def _get_files_list(self) -> CoreResponse:
        """Получить список файлов"""
        try:
            files = []
            total_size = 0

            for f in os.listdir(self.storage_dir):
                f_path = os.path.join(self.storage_dir, f)
                if os.path.isfile(f_path):
                    size = os.path.getsize(f_path)
                    modified = datetime.fromtimestamp(os.path.getmtime(f_path)).strftime('%d.%m.%Y %H:%M')
                    files.append({
                        'name': f,
                        'size': size,
                        'modified': modified
                    })
                    total_size += size

            if not files:
                return CoreResponse(
                    success=True,
                    data={'files': [], 'total_size': 0},
                    raw_result="📁 **Файловое хранилище:**\n\n📭 **Папка пуста.**\n\n💡 **Чтобы создать файл:**\n`сохрани в файл заметка.txt: ваш текст`",
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

            files.sort(key=lambda x: x['modified'], reverse=True)
            files_list = []
            for i, f in enumerate(files[:15], 1):
                size_kb = f['size'] / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb >= 1 else f"{f['size']} байт"
                files_list.append(f"{i}. **{f['name']}** ({size_str}, изменен: {f['modified']})")

            files_text = "\n".join(files_list)
            total_kb = total_size / 1024
            total_str = f"{total_kb:.1f} KB" if total_kb >= 1 else f"{total_size} байт"

            return CoreResponse(
                success=True,
                data={'files': files, 'total_size': total_size, 'count': len(files)},
                raw_result=f"📁 **Файловое хранилище:**\n\n📊 **Всего файлов:** {len(files)}\n📦 **Общий размер:** {total_str}\n\n📋 **Список файлов:**\n{files_text}",
                confidence=1.0,
                source=self.name,
                direct_answer=True
            )

        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ **Ошибка чтения списка файлов:** {str(e)}",
                confidence=0.0,
                source=self.name,
                direct_answer=True
            )

    def _delete_file(self, query: str) -> CoreResponse:
        """Удалить файл"""
        filename = ""
        match = re.search(r'удали\s+файл\s+["\']?([^"\'\s]+)["\']?', query, re.IGNORECASE)
        if match:
            filename = match.group(1).strip()

        if not filename:
            return CoreResponse(
                success=False,
                data={'error': 'Не указано имя файла'},
                raw_result="❌ **Пожалуйста, укажите имя файла для удаления.**\n\nПример: `удали файл привет.txt`",
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
                raw_result=f"✅ **Файл `{filename}` успешно удален.**",
                confidence=1.0,
                source=self.name,
                direct_answer=True
            )
        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ **Ошибка удаления файла:** {str(e)}",
                confidence=0.0,
                source=self.name,
                direct_answer=True
            )

    def _get_help(self) -> CoreResponse:
        """Помощь по работе с файлами"""
        return CoreResponse(
            success=True,
            data={'action': 'help'},
            raw_result="📁 **Файловое хранилище - команды:**\n\n"
                       "• **Прочитать файл:**\n"
                       "  `прочитай файл имя_файла.txt`\n\n"
                       "• **Сохранить файл:**\n"
                       "  `сохрани в файл имя_файла.txt: ваш текст`\n\n"
                       "• **Список файлов:**\n"
                       "  `список файлов`\n\n"
                       "• **Удалить файл:**\n"
                       "  `удали файл имя_файла.txt`\n\n"
                       "**Примеры:**\n"
                       "1. `сохрани в файл привет.txt: Привет, мир!`\n"
                       "2. `прочитай файл привет.txt`\n"
                       "3. `список файлов`\n"
                       "4. `удали файл привет.txt`",
            confidence=1.0,
            source=self.name,
            direct_answer=True
        )


# ==================== ВАЛИДАТОР ЯДЕР ====================
class CoreValidator:
    """Валидатор и тестер ядер знаний"""

    @staticmethod
    def validate_code_structure(code: str) -> Tuple[bool, str]:
        """Проверяет структуру кода ядра"""
        try:
            if "class " not in code or "KnowledgeCore" not in code:
                return False, "Отсутствует наследование от KnowledgeCore"

            if "def can_handle" not in code:
                return False, "Отсутствует метод can_handle"

            if "def execute" not in code:
                return False, "Отсутствует метод execute"

            if "CoreResponse" not in code:
                return False, "Отсутствует возврат CoreResponse"

            required_attrs = ['name', 'description', 'capabilities']
            for attr in required_attrs:
                if f"{attr} =" not in code and f'{attr} =' not in code:
                    return False, f"Отсутствует атрибут {attr}"

            ast.parse(code)
            return True, "Код валиден"

        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {e}"
        except Exception as e:
            return False, f"Ошибка валидации: {e}"

    @staticmethod
    def extract_core_info(code: str) -> Dict[str, Any]:
        """Извлекает информацию о ядре из кода"""
        info = {
            'name': 'unknown',
            'description': '',
            'capabilities': [],
            'priority': 5,
            'has_web_search': False
        }

        try:
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', code)
            if name_match:
                info['name'] = name_match.group(1)

            desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', code)
            if desc_match:
                info['description'] = desc_match.group(1)

            caps_match = re.search(r'capabilities\s*=\s*\[(.*?)\]', code, re.DOTALL)
            if caps_match:
                caps_text = caps_match.group(1)
                caps = re.findall(r'["\']([^"\']+)["\']', caps_text)
                info['capabilities'] = caps

            prio_match = re.search(r'priority\s*=\s*(\d+)', code)
            if prio_match:
                info['priority'] = int(prio_match.group(1))

            if 'web_search' in code or 'DDGS' in code:
                info['has_web_search'] = True

        except Exception as e:
            print(f"  ⚠️ Ошибка извлечения информации: {e}")

        return info

    @staticmethod
    def test_core_instance(core_instance: KnowledgeCore, test_queries: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Тестирует экземпляр ядра"""
        results = []

        for query in test_queries:
            try:
                response = core_instance.execute(query)
                results.append({
                    'query': query,
                    'success': response.success,
                    'confidence': response.confidence,
                    'has_raw_result': bool(response.raw_result)
                })
            except Exception as e:
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })

        success_count = sum(1 for r in results if r.get('success', False))
        has_raw_results = sum(1 for r in results if r.get('has_raw_result', False))

        passed = success_count >= len(test_queries) * 0.5

        return passed, {
            'total_tests': len(test_queries),
            'success_count': success_count,
            'has_raw_results': has_raw_results,
            'details': results
        }


# ==================== СИСТЕМА ПАМЯТИ ====================
class MemoryManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.short_term: List[Dict] = []
        self.max_short_term = 20
        self.memory_file = os.path.join(MEMORY_DIR, f"user_{user_id}_facts.json")
        self.long_term = self._load_long_term()

    def _load_long_term(self) -> List[Dict]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_long_term(self):
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения памяти: {e}")

    def add_short_term(self, msg: Dict):
        self.short_term.append(msg)
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)

    def get_short_term(self, last_n: int = 10) -> List[Dict]:
        return self.short_term[-last_n:] if self.short_term else []

    def save_long_term(self, fact: str, fact_type: str = "general"):
        self.long_term.append({
            'content': fact,
            'type': fact_type,
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id
        })
        self._save_long_term()

    def search_long_term(self, query: str, limit: int = 3) -> List[Dict]:
        query_lower = query.lower().replace('фаил', 'файл')
        query_words = set(re.findall(r'\w+', query_lower))
        scored = []

        for fact in self.long_term:
            fact_words = set(re.findall(r'\w+', fact['content'].lower()))
            score = len(query_words & fact_words)
            if score > 0:
                scored.append((score, fact))

        scored.sort(key=lambda x: (x[0], x[1]['timestamp']), reverse=True)
        return [fact for _, fact in scored[:limit]]


# ==================== ЖУРНАЛ ОБУЧЕНИЯ ====================
class LearningLogger:
    """Логирует процесс самообучения бота"""

    @staticmethod
    def log_new_core(core_name: str, query: str, user_id: str, success: bool = True):
        """Записывает создание нового ядра"""
        try:
            log_data = []
            if os.path.exists(LEARNING_LOG):
                try:
                    with open(LEARNING_LOG, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    if not isinstance(log_data, list):
                        log_data = []
                except Exception:
                    log_data = []

            log_data.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'core_created',
                'core_name': core_name,
                'trigger_query': query,
                'user_id': user_id,
                'success': success
            })

            with open(LEARNING_LOG, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            status = "✅" if success else "❌"
            print(f"{status} Записал в журнал: создано ядро '{core_name}'")

        except Exception as e:
            print(f"⚠️ Ошибка записи в журнал: {e}")

    @staticmethod
    def log_core_rejection(core_name: str, reason: str, query: str):
        """Логирует отклонение ядра"""
        try:
            log_data = []
            if os.path.exists(REJECTED_CORES_LOG):
                try:
                    with open(REJECTED_CORES_LOG, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    if not isinstance(log_data, list):
                        log_data = []
                except Exception:
                    log_data = []

            log_data.append({
                'timestamp': datetime.now().isoformat(),
                'core_name': core_name,
                'reason': reason,
                'query': query[:200]
            })

            with open(REJECTED_CORES_LOG, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            print(f"❌ Отклонено ядро '{core_name}': {reason}")

        except Exception as e:
            print(f"⚠️ Ошибка записи отклонений: {e}")

    @staticmethod
    def log_core_performance(core_name: str, success: bool, query: str):
        """Логирует производительность ядра"""
        try:
            perf_data = {}
            if os.path.exists(CORE_PERFORMANCE_LOG):
                try:
                    with open(CORE_PERFORMANCE_LOG, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        perf_data = data
                except Exception:
                    perf_data = {}

            if core_name not in perf_data:
                perf_data[core_name] = {
                    'success_count': 0,
                    'total_count': 0,
                    'last_used': '',
                    'queries': []
                }

            perf_data[core_name]['total_count'] += 1
            if success:
                perf_data[core_name]['success_count'] += 1
            perf_data[core_name]['last_used'] = datetime.now().isoformat()

            perf_data[core_name]['queries'].append({
                'query': query[:100],
                'success': success,
                'timestamp': datetime.now().isoformat()
            })

            perf_data[core_name]['queries'] = perf_data[core_name]['queries'][-5:]

            cores_to_remove = []
            for name, stats in perf_data.items():
                if not isinstance(stats, dict):
                    continue

                total = stats.get('total_count', 0)
                success_count = stats.get('success_count', 0)

                if total > 10:
                    success_rate = success_count / total
                    if success_rate < 0.3:
                        cores_to_remove.append(name)

            for name in cores_to_remove:
                print(f"🗑️ Удаляю неэффективное ядро из лога: {name}")
                del perf_data[name]

            with open(CORE_PERFORMANCE_LOG, 'w', encoding='utf-8') as f:
                json.dump(perf_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️ Ошибка записи производительности: {e}")

    @staticmethod
    def get_recent_learning(limit: int = 5) -> List[Dict]:
        """Возвращает последние записи обучения"""
        try:
            if os.path.exists(LEARNING_LOG):
                with open(LEARNING_LOG, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                if isinstance(log_data, list):
                    return log_data[-limit:] if log_data else []
        except Exception:
            pass
        return []


# ==================== МЕНЕДЖЕР ЯДЕР ====================
class ToolsManager:
    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self.validator = CoreValidator()
        self._load_builtin_cores()
        self._load_dynamic_cores()
        self._cleanup_bad_cores()
        self.typo_fixes = {'фаил': 'файл', 'прочти': 'прочитай'}

    def _load_builtin_cores(self):
        """Загружает встроенные ядра"""
        builtin_cores = [
            DateTimeCore(),
            CalculatorCore(),
            FileStorageCore()
        ]

        if DDGS_AVAILABLE:
            builtin_cores.append(WebSearchCore())

        for core in builtin_cores:
            self.cores[core.name] = core

        print(f"✅ Загружено встроенных ядер: {len(builtin_cores)}")
        for core in builtin_cores:
            print(f"   - {core.name}: {core.description}")

    def _load_dynamic_cores(self):
        """Загружает динамические ядра из файлов"""
        loaded_count = 0
        rejected_count = 0

        if not os.path.exists(CORES_DIR):
            return

        for fname in os.listdir(CORES_DIR):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue

            path = os.path.join(CORES_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()

                is_valid, reason = self.validator.validate_code_structure(code)
                if not is_valid:
                    print(f"  ❌ Плохое ядро {fname}: {reason}")
                    rejected_count += 1
                    continue

                core_info = self.validator.extract_core_info(code)
                core_name = core_info['name']

                if core_name in self.cores:
                    print(f"  ⚠️ Дубликат ядра {core_name}, пропускаю")
                    continue

                namespace = {
                    'KnowledgeCore': KnowledgeCore,
                    'CoreResponse': CoreResponse,
                    '__builtins__': __builtins__,
                    're': re,
                    'json': json,
                    'datetime': datetime,
                    'timedelta': timedelta,
                    'os': os,
                    'time': time,
                    'hashlib': hashlib,
                    'shutil': shutil
                }

                if DDGS_AVAILABLE and ('DDGS' in code or 'duckduckgo' in code):
                    namespace['DDGS'] = DDGS
                if 'requests' in code:
                    namespace['requests'] = requests

                exec(code, namespace)

                core_class = None
                for key, value in namespace.items():
                    if (isinstance(value, type) and
                            issubclass(value, KnowledgeCore) and
                            value != KnowledgeCore):
                        core_class = value
                        break

                if not core_class:
                    print(f"  ❌ Не найден класс ядра в {fname}")
                    rejected_count += 1
                    continue

                core_instance = core_class()

                test_queries = [
                    f"тест для {core_info['description']}",
                    "помощь",
                    "информация"
                ]

                passed, test_results = self.validator.test_core_instance(core_instance, test_queries)

                if not passed:
                    print(f"  ❌ Ядро {core_name} не прошло тесты")
                    rejected_count += 1
                    continue

                self.cores[core_name] = core_instance
                loaded_count += 1
                print(f"✅ Загружено динамическое ядро: {core_name}")

            except Exception as e:
                print(f"  ❌ Ошибка загрузки {fname}: {e}")
                rejected_count += 1

        if loaded_count > 0:
            print(f"✅ Загружено динамических ядер: {loaded_count}")
        if rejected_count > 0:
            print(f"❌ Отклонено ядер: {rejected_count}")

    def _cleanup_bad_cores(self):
        """Удаляет нерабочие ядра"""
        bad_cores = []

        for core_name, core in list(self.cores.items()):
            try:
                if not hasattr(core, 'can_handle') or not callable(core.can_handle):
                    bad_cores.append(core_name)
                    continue

                if not hasattr(core, 'execute') or not callable(core.execute):
                    bad_cores.append(core_name)
                    continue

                test_response = core.execute("тест")
                if not isinstance(test_response, CoreResponse):
                    bad_cores.append(core_name)

            except Exception as e:
                print(f"  ❌ Ошибка проверки ядра {core_name}: {e}")
                bad_cores.append(core_name)

        for core_name in bad_cores:
            if core_name in self.cores:
                del self.cores[core_name]
                print(f"🗑️ Удалено нерабочее ядро: {core_name}")

                core_file = os.path.join(CORES_DIR, f"{core_name}.py")
                if os.path.exists(core_file):
                    try:
                        os.remove(core_file)
                    except Exception:
                        pass

    def fix_typos(self, query: str) -> str:
        """Исправляет опечатки в запросе"""
        fixed = query
        for typo, correct in self.typo_fixes.items():
            fixed = fixed.replace(typo, correct)
        if fixed != query:
            print(f"🔧 Исправлены опечатки: '{query}' → '{fixed}'")
        return fixed

    def find_best_core(self, query: str) -> Optional[Tuple[KnowledgeCore, float]]:
        """Находит лучшее ядро для запроса"""
        query_fixed = self.fix_typos(query)

        best_core = None
        best_confidence = 0.0

        for core in self.cores.values():
            if core.can_handle(query_fixed):
                confidence = core.get_confidence(query_fixed)

                if confidence < 0.3:
                    continue

                q_lower = query_fixed.lower()
                core_keywords = ' '.join(core.capabilities + [core.description]).lower()

                query_words = set(re.findall(r'\w{3,}', q_lower))
                core_words = set(re.findall(r'\w{3,}', core_keywords))
                matches = len(query_words & core_words)

                if matches == 0 and confidence < 0.6:
                    continue

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_core = core

        if best_core and best_confidence > 0.4:
            return best_core, best_confidence

        return None

    def get_cores_summary(self) -> str:
        """Краткое описание существующих ядер"""
        summary = []
        for core in self.cores.values():
            caps = ', '.join(core.capabilities[:2]) if core.capabilities else 'нет описания'
            direct = "✅" if core.direct_answer_mode else "❌"
            summary.append(f"- **{core.name}** {direct} ({caps})")
        return '\n'.join(summary)


# ==================== САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ ====================
class MiniBrain:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.tools = ToolsManager()
        self.conversation_history = []

    def _web_search_tool(self, query: str, max_results: int = 4) -> List[Dict]:
        """Инструмент веб-поиска"""
        if not DDGS_AVAILABLE:
            return []

        try:
            results = list(DDGS().text(query, max_results=max_results))
            return [
                {
                    'title': r.get('title', '')[:100],
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')[:250]
                }
                for r in results[:max_results]
            ]
        except Exception as e:
            print(f"  ⚠️ Ошибка поиска: {e}")
            return []

    async def process(self, query: str, llm_caller) -> Dict[str, Any]:
        """Обработка запроса"""
        query_fixed = query.replace('фаил', 'файл').replace('файл ', 'файл')
        if query_fixed != query:
            print(f"🔧 Исправлена опечатка: '{query}' → '{query_fixed}'")

        query = query_fixed

        print(f"\n{'=' * 60}")
        print(f"🧠 Запрос от {self.user_id}: {query}")
        print(f"{'=' * 60}")

        self.conversation_history.append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        })

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        core_result = self.tools.find_best_core(query)

        if core_result:
            core, confidence = core_result
            print(f"🎯 Найдено ядро: {core.name} (уверенность: {confidence:.2f})")

            if confidence > 0.6:
                print(f"⚡ Использую ядро {core.name}...")

                context = {
                    'user_id': self.user_id,
                    'tools': {
                        'web_search': self._web_search_tool,
                        'memory': self.memory
                    }
                }

                core_response = core.execute(query, context)

                LearningLogger.log_core_performance(
                    core.name,
                    core_response.success,
                    query
                )

                if core_response.is_final_answer():
                    print(f"✅ Ядро {core.name} дало прямой ответ")

                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': core_response.raw_result,
                        'source': core.name,
                        'timestamp': datetime.now().isoformat()
                    })

                    return {
                        'type': 'direct_core_response',
                        'response': core_response.raw_result,
                        'source': core.name,
                        'confidence': confidence,
                        'need_llm': False
                    }

                llm_context = self._build_llm_context_with_core(
                    query, core_response, core.name
                )

                return {
                    'type': 'llm_with_core_data',
                    'context': llm_context,
                    'source': core.name,
                    'confidence': confidence,
                    'need_llm': True
                }

        print("💭 Использую общую обработку...")
        llm_context = self._build_general_context(query)

        return {
            'type': 'general_llm',
            'context': llm_context,
            'source': 'general',
            'need_llm': True
        }

    def _build_llm_context_with_core(self, query: str, core_response: CoreResponse, core_name: str) -> str:
        """Строит контекст для LLM с данными ядра"""
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        context = f"""# СИСТЕМА: ИИ АССИСТЕНТ С СПЕЦИАЛИЗИРОВАННЫМИ ЯДРАМИ
Текущее время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь: {self.user_id}
Ядро обработки: {core_name}

# ДАННЫЕ ОТ ЯДРА '{core_name.upper()}':
{core_response.raw_result if core_response.raw_result else core_response.to_context_string()}

# ИСТОРИЯ ДИАЛОГА (последние сообщения):
"""

        if self.conversation_history:
            for msg in self.conversation_history[-5:]:
                role = "👤 ПОЛЬЗОВАТЕЛЬ" if msg['role'] == 'user' else "🤖 АССИСТЕНТ"
                context += f"{role}: {msg['content'][:150]}\n"

        context += f"""

# ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

# ИНСТРУКЦИИ ДЛЯ ОТВЕТА:
1. ОСНОВЫВАЙСЯ на данных от ядра {core_name}
2. Если ядро предоставило информацию - ИСПОЛЬЗУЙ её
3. Если ядро сообщило об ошибке - объясни это пользователю
4. Будь полезным и конкретным
5. НЕ выдумывай информацию, которой нет в данных ядра

ТВОЙ ОТВЕТ (используй данные ядра):"""

        return context

    def _build_general_context(self, query: str) -> str:
        """Строит общий контекст для LLM"""
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        context = f"""# СИСТЕМА: ИИ АССИСТЕНТ
Время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь: {self.user_id}

# ДОСТУПНЫЕ ЯДРА:
{self.tools.get_cores_summary()}

# ИСТОРИЯ ДИАЛОГА:
"""

        if self.conversation_history:
            for msg in self.conversation_history[-5:]:
                role = "👤 ПОЛЬЗОВАТЕЛЬ" if msg['role'] == 'user' else "🤖 АССИСТЕНТ"
                context += f"{role}: {msg['content'][:150]}\n"

        context += f"""

# ВОПРОС:
{query}

# ИНСТРУКЦИИ:
Ответь как полезный ассистент. Будь честным - если не знаешь ответа, так и скажи.
Если вопрос относится к специализированной области, предложи использовать соответствующее ядро."""

        return context


# ==================== ТЕЛЕГРАМ БОТ ====================
class TelegramBot:
    def __init__(self):
        self.user_brains: Dict[str, MiniBrain] = {}
        self._initialize_logs()

    def _initialize_logs(self):
        """Инициализирует файлы логов"""
        logs_config = {
            LEARNING_LOG: [],
            CORE_PERFORMANCE_LOG: {},
            REJECTED_CORES_LOG: []
        }

        for log_file, default_data in logs_config.items():
            if not os.path.exists(log_file):
                try:
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(default_data, f, ensure_ascii=False, indent=2)
                    print(f"✅ Создан лог-файл: {os.path.basename(log_file)}")
                except Exception as e:
                    print(f"⚠️ Ошибка создания {log_file}: {e}")
            else:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if log_file == CORE_PERFORMANCE_LOG and not isinstance(data, dict):
                        print(f"⚠️ Исправляю структуру файла производительности...")
                        with open(log_file, 'w', encoding='utf-8') as f:
                            json.dump({}, f, ensure_ascii=False, indent=2)

                except json.JSONDecodeError:
                    print(f"⚠️ Файл {log_file} поврежден, пересоздаю...")
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(default_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"⚠️ Ошибка проверки файла {log_file}: {e}")

    def get_brain(self, user_id: str) -> MiniBrain:
        if user_id not in self.user_brains:
            self.user_brains[user_id] = MiniBrain(user_id)
            print(f"🧠 Создан новый мозг для пользователя {user_id}")
        return self.user_brains[user_id]

    async def get_llm_response(self, context: str, temperature: float = 0.5) -> str:
        """Запрос к LLM"""
        try:
            resp = requests.post(
                LM_STUDIO_API_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {LM_STUDIO_API_KEY}'
                },
                json={
                    'messages': [{'role': 'user', 'content': context}],
                    'temperature': temperature,
                    'max_tokens': 1500,
                    'stream': False
                },
                timeout=120
            )

            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
            else:
                print(f"❌ Ошибка LLM (HTTP {resp.status_code}): {resp.text}")
                return f"❌ Ошибка получения ответа (код: {resp.status_code})"

        except requests.exceptions.Timeout:
            return "❌ Таймаут при запросе к модели"
        except Exception as e:
            print(f"❌ Ошибка LLM: {e}")
            return f"❌ Ошибка: {str(e)[:100]}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        web_search_status = "✅ Доступен" if DDGS_AVAILABLE else "❌ Недоступен (не установлен)"

        await update.message.reply_text(
            "🧠 *САМООБУЧАЮЩИЙСЯ ИИ-АССИСТЕНТ v4.2*\n\n"
            "✨ *Новые возможности:*\n"
            "• 🔧 Автоматическое исправление опечаток\n"
            "• 📁 Встроенное файловое ядро\n"
            "• ✅ Прямые ответы от ядер без LLM\n"
            "• 🐛 Исправлены все критические баги\n"
            f"• 🌐 Веб-поиск: {web_search_status}\n\n"
            "💡 *Как это работает:*\n"
            "1. Вы задаёте вопрос\n"
            "2. Бот исправляет опечатки\n"
            "3. Если есть подходящее ядро - использует его\n"
            "4. Файловые запросы работают сразу!\n\n"
            "⚙️ *Команды:*\n"
            "/list_cores — список всех ядер\n"
            "/learning_log — журнал обучения\n"
            "/performance — статистика ядер\n"
            "/help — справка\n"
            "/clear — очистить историю\n"
            "/files — управление файлами",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        await update.message.reply_text(
            "📖 *СПРАВКА*\n\n"
            "*Автоисправление опечаток:*\n"
            "Бот автоматически исправляет 'фаил' на 'файл'\n\n"
            "*Файловое хранилище:*\n"
            "• `сохрани в файл имя.txt: текст`\n"
            "• `прочитай файл имя.txt`\n"
            "• `список файлов`\n"
            "• `удали файл имя.txt`\n\n"
            "*Команды:*\n"
            "/list_cores — все ядра\n"
            "/learning_log — журнал обучения\n"
            "/performance — статистика\n"
            "/files — файлы",
            parse_mode='Markdown'
        )

    async def list_cores(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Список ядер"""
        brain = self.get_brain(str(update.effective_user.id))

        msg = "🔧 *ДОСТУПНЫЕ ЯДРА*\n\n"

        cores_by_priority = {}
        for core in brain.tools.cores.values():
            if core.priority not in cores_by_priority:
                cores_by_priority[core.priority] = []
            cores_by_priority[core.priority].append(core)

        for priority in sorted(cores_by_priority.keys()):
            msg += f"*Приоритет {priority}:*\n"
            for core in cores_by_priority[priority]:
                caps = ', '.join(core.capabilities[:3]) if core.capabilities else 'нет описания'
                direct = "✅" if core.direct_answer_mode else "❌"
                msg += f"• `{core.name}` {direct}\n"
                msg += f"  _{core.description}_\n"
                msg += f"  📦 {caps}\n\n"

        msg += f"_Всего: {len(brain.tools.cores)} ядер_\n"
        msg += "✅ — прямой ответ, ❌ — нужна обработка LLM"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def learning_log_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Журнал обучения"""
        log = LearningLogger.get_recent_learning(limit=10)

        if not log:
            await update.message.reply_text("📝 *Журнал обучения пуст*", parse_mode='Markdown')
            return

        msg = "📚 *ЖУРНАЛ САМООБУЧЕНИЯ*\n\n"

        for entry in reversed(log):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%d.%m %H:%M')
            core_name = entry.get('core_name', 'unknown')
            trigger = entry.get('trigger_query', '')[:60]
            success = "✅" if entry.get('success', False) else "❌"

            msg += f"⏰ *{timestamp}* {success}\n"
            msg += f"🆕 Ядро: `{core_name}`\n"
            msg += f"📝 Запрос: _{trigger}_\n\n"

        msg += f"_Последние {len(log)} записей_"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def performance_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика производительности"""
        try:
            if not os.path.exists(CORE_PERFORMANCE_LOG):
                await update.message.reply_text("📊 *Статистика пока не собрана*", parse_mode='Markdown')
                return

            with open(CORE_PERFORMANCE_LOG, 'r', encoding='utf-8') as f:
                perf_data = json.load(f)

            if not isinstance(perf_data, dict) or not perf_data:
                await update.message.reply_text("📊 *Статистика пока не собрана*", parse_mode='Markdown')
                return

            msg = "📊 *СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ*\n\n"

            sorted_cores = sorted(
                perf_data.items(),
                key=lambda x: x[1].get('total_count', 0),
                reverse=True
            )

            for core_name, stats in sorted_cores[:15]:
                if not isinstance(stats, dict):
                    continue

                total = stats.get('total_count', 0)
                success = stats.get('success_count', 0)

                if total == 0:
                    rate = 0
                else:
                    rate = success / total

                if rate >= 0.8:
                    emoji = "🟢"
                elif rate >= 0.5:
                    emoji = "🟡"
                else:
                    emoji = "🔴"

                msg += f"{emoji} *{core_name}*\n"
                msg += f"  Успешность: {rate:.0%} ({success}/{total})\n"

                if stats.get('queries'):
                    last_query = stats['queries'][-1]
                    last_time = datetime.fromisoformat(last_query['timestamp']).strftime('%H:%M')
                    status = "✅" if last_query.get('success', False) else "❌"
                    query_text = last_query.get('query', '')[:30]
                    msg += f"  Последний: {last_time} {status} ({query_text}...)\n"

                msg += "\n"

            msg += f"_Всего ядер: {len(perf_data)}_"

            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)[:100]}")

    async def clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистить историю"""
        user_id = str(update.effective_user.id)

        if user_id in self.user_brains:
            self.user_brains[user_id].conversation_history = []
            self.user_brains[user_id].memory.short_term = []

            await update.message.reply_text(
                "🧹 *История диалога очищена*\n\n"
                "Все предыдущие сообщения удалены из памяти.",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("✅ История уже пуста!")

    async def files_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Управление файлами"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)

        file_core = None
        for core in brain.tools.cores.values():
            if core.name == "file_storage_core":
                file_core = core
                break

        if not file_core:
            await update.message.reply_text(
                "❌ *Файловое ядро не найдено!*",
                parse_mode='Markdown'
            )
            return

        response = file_core.execute("список файлов")

        if response.success and response.raw_result:
            await update.message.reply_text(
                response.raw_result,
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                "📁 *Файловое хранилище:*\n\n"
                "❌ Не удалось получить список файлов.",
                parse_mode='Markdown'
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        brain = self.get_brain(user_id)

        try:
            start_time = time.time()
            brain_result = await brain.process(text, self.get_llm_response)
            processing_time = time.time() - start_time

            print(f"⏱️ Время обработки: {processing_time:.2f} сек")
            print(f"📊 Тип ответа: {brain_result['type']}")

            if brain_result.get('need_llm') == False:
                response = brain_result['response']

                brain.conversation_history.append({
                    'role': 'assistant',
                    'content': response,
                    'source': brain_result.get('source', 'core'),
                    'timestamp': datetime.now().isoformat()
                })

                await update.message.reply_text(
                    response,
                    parse_mode='Markdown',
                    disable_web_page_preview=True
                )
                return

            llm_response = await self.get_llm_response(
                brain_result['context'],
                temperature=0.5
            )

            if len(llm_response) > 4000:
                llm_response = llm_response[:4000] + "\n\n... (сообщение сокращено)"

            brain.conversation_history.append({
                'role': 'assistant',
                'content': llm_response,
                'source': brain_result.get('source', 'llm'),
                'timestamp': datetime.now().isoformat()
            })

            await update.message.reply_text(
                llm_response,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )

        except Exception as e:
            error_msg = f"❌ *Ошибка:* {str(e)[:200]}"
            print(f"ERROR [{user_id}]: {e}")
            traceback.print_exc()

            await update.message.reply_text(error_msg, parse_mode='Markdown')


# ==================== ЗАПУСК ====================
def main():
    """Главная функция запуска бота"""

    print("\n" + "=" * 70)
    print("🚀 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v4.2")
    print("=" * 70)
    print(f"⏰ Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"📁 Директория ядер: {CORES_DIR}/")
    print(f"📁 Директория файлов: {USER_FILES_DIR}/")
    print(f"🧠 Директория памяти: {MEMORY_DIR}/")
    print(f"🔗 LM Studio: {LM_STUDIO_API_URL}")
    print(f"🌐 Веб-поиск: {'✅ Доступен' if DDGS_AVAILABLE else '❌ Недоступен'}")
    print("=" * 70)

    # Проверка LM Studio
    try:
        test_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '')
        test_resp = requests.get(test_url, timeout=10)

        if test_resp.status_code == 200:
            print(f"✅ LM Studio доступна")

            models_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '/v1/models')
            try:
                models_resp = requests.get(models_url, timeout=5)
                if models_resp.status_code == 200:
                    models = models_resp.json().get('data', [])
                    if models:
                        model_names = [m.get('id', 'unknown') for m in models[:2]]
                        print(f"📋 Загруженные модели: {', '.join(model_names)}")
            except Exception:
                pass
        else:
            print(f"⚠️ LM Studio код: {test_resp.status_code}")
    except Exception as e:
        print(f"⚠️ LM Studio недоступна: {e}")
        print("🤖 Бот будет работать с ограниченной функциональностью")

    print("=" * 70)
    print("\n🔄 Инициализация...\n")

    bot = TelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("list_cores", bot.list_cores))
    application.add_handler(CommandHandler("learning_log", bot.learning_log_cmd))
    application.add_handler(CommandHandler("performance", bot.performance_cmd))
    application.add_handler(CommandHandler("clear", bot.clear_cmd))
    application.add_handler(CommandHandler("files", bot.files_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот инициализирован!")
    print("=" * 70)
    print("💬 Напишите боту в Telegram")
    print("📁 Файловые операции работают сразу!")
    print("🤖 Все встроенные ядра готовы к работе")
    print("=" * 70)
    print("\nCtrl+C для остановки\n")

    try:
        application.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\n\n🛑 Остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()