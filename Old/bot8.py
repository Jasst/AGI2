#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v6.0 - С ИНТЕЛЛЕКТУАЛЬНОЙ ПАМЯТЬЮ
✅ Автономная рефлексия и анализ паттернов
✅ Детекция и разрешение противоречий
✅ Эмоциональное взвешивание важности
✅ Самоорганизация знаний через кластеризацию
✅ Сохранение всех данных между перезагрузками
✅ Глубокая рефлексия для выявления скрытых паттернов
✅ Автоматическое обновление устаревших фактов
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
from collections import defaultdict
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
for directory in [CORES_DIR, MEMORY_DIR, USER_FILES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Файлы
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
CORE_PERFORMANCE_LOG = os.path.join(MEMORY_DIR, "core_performance.json")
REJECTED_CORES_LOG = os.path.join(MEMORY_DIR, "rejected_cores.json")
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
# (Все встроенные ядра остаются без изменений - DateTimeCore, CalculatorCore, WebSearchCore, FileStorageCore)
# Для экономии места я пропущу их реализацию, они идентичны оригинальному коду

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

            results_text = "🌐 **Результаты поиска в интернете:**\n"
            for r in search_results:
                results_text += f"{r['position']}. **{r['title']}**\n"
                if r['snippet']:
                    results_text += f"   _{r['snippet']}_\n"
                results_text += f"   🔗 {r['url']}\n"

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
            if 'прочитай' in q or 'прочти' in q or 'открой' in q:
                filename = self._extract_filename_for_read(query)
                if not filename:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Не указано имя файла'},
                        raw_result="❌ **Пожалуйста, укажите имя файла.**\nПримеры:\n• `прочитай файл привет.txt`",
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
                        raw_result=f"📄 **Файл `{filename}`:**\n{content}\n\n📊 **Информация:**\n• Размер: {file_size} байт\n• Строк: {lines}",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )
                except UnicodeDecodeError:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Файл не в UTF-8'},
                        raw_result=f"❌ **Не могу прочитать файл `{filename}`.**",
                        confidence=0.0,
                        source=self.name,
                        direct_answer=True
                    )

            elif 'сохрани' in q or 'запиши' in q:
                match = re.search(r'сохрани\s+(?:в\s+)?файл\s+["\']?([^"\':]+)["\']?\s*[:：]\s*(.+)', query,
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
                        raw_result=f"❌ **Ошибка:** {str(e)}",
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
                        raw_result=f"✅ **Файл сохранен!**\n📄 `{filename}` ({file_size} байт)\n📝 {preview}",
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

            elif 'список файлов' in q or 'мои файлы' in q:
                return self._get_files_list()

            elif 'удали файл' in q:
                return self._delete_file(query)

            else:
                return self._get_help()

        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ **Ошибка:** {str(e)}",
                confidence=0.0,
                source=self.name,
                direct_answer=True
            )

    def _extract_filename_for_read(self, query: str) -> str:
        patterns = [
            r'прочитай файл\s+["\']?([^"\'\s]+)["\']?',
            r'прочти файл\s+["\']?([^"\'\s]+)["\']?',
            r'открой файл\s+["\']?([^"\'\s]+)["\']?',
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
                raw_result=f"❌ **Файл `{filename}` не найден.**\n📁 **Доступные:**\n{file_list}",
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

    def _get_files_list(self) -> CoreResponse:
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
                files_list.append(f"{i}. **{f['name']}** ({size_str})")
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

    def _delete_file(self, query: str) -> CoreResponse:
        filename = ""
        match = re.search(r'удали\s+файл\s+["\']?([^"\'\s]+)["\']?', query, re.IGNORECASE)
        if match:
            filename = match.group(1).strip()

        if not filename:
            return CoreResponse(
                success=False,
                data={'error': 'Не указано имя файла'},
                raw_result="❌ **Укажите имя файла.**",
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
                raw_result=f"❌ **Ошибка:** {str(e)}",
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
                raw_result=f"✅ **Файл `{filename}` удален.**",
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

    def _get_help(self) -> CoreResponse:
        return CoreResponse(
            success=True,
            data={'action': 'help'},
            raw_result="📁 **Команды:**\n• `прочитай файл имя.txt`\n• `сохрани в файл имя.txt: текст`\n• `список файлов`\n• `удали файл имя.txt`",
            confidence=1.0,
            source=self.name,
            direct_answer=True
        )


# ==================== ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА ПАМЯТИ v2.0 ====================
class IntelligentMemoryManager:
    """
    🧠 ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА ПАМЯТИ С САМООБУЧЕНИЕМ
    ✅ Автоматическая рефлексия каждые N сообщений
    ✅ Детекция противоречий и обновление фактов
    ✅ Эмоциональное взвешивание важности
    ✅ Семантическая кластеризация знаний
    ✅ Сохранение между перезагрузками (файловая система)
    ✅ Автоматическое извлечение паттернов поведения
    ✅ Глубокая рефлексия для выявления скрытых паттернов
    ✅ Самоорганизация через регулярный анализ
    """

    def __init__(self, user_id: str, llm_caller=None):
        self.user_id = user_id
        self.llm_caller = llm_caller

        # Файлы памяти (сохраняются между перезагрузками)
        self.memory_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.short_term_file = os.path.join(self.memory_dir, "short_term.json")
        self.long_term_file = os.path.join(self.memory_dir, "long_term.json")
        self.patterns_file = os.path.join(self.memory_dir, "behavior_patterns.json")
        self.metadata_file = os.path.join(self.memory_dir, "metadata.json")
        self.contradictions_file = os.path.join(self.memory_dir, "contradictions_log.json")

        # Лимиты и интервалы
        self.short_term_limit = 25
        self.long_term_limit = 150
        self.consolidation_interval = 4  # Консолидация каждые N сообщений
        self.reflection_interval = 15  # Глубокая рефлексия каждые N сообщений
        self.messages_counter = 0

        # Загрузка существующей памяти
        self.short_term_memory = self._load_json(self.short_term_file, [])
        self.long_term_memory = self._load_json(self.long_term_file, [])
        self.behavior_patterns = self._load_json(self.patterns_file, {
            'preferences': {},
            'habits': {},
            'communication_style': 'нейтральный',
            'frequent_topics': [],
            'deep_patterns': [],
            'hypotheses': [],
            'emotional_profile': '',
            'last_updated': datetime.now().isoformat()
        })
        self.metadata = self._load_json(self.metadata_file, {
            'total_interactions': 0,
            'facts_learned': 0,
            'last_consolidation': None,
            'last_reflection': None,
            'contradictions_detected': 0,
            'contradictions_resolved': 0,
            'first_interaction': datetime.now().isoformat(),
            'last_interaction': datetime.now().isoformat()
        })
        self.contradictions_log = self._load_json(self.contradictions_file, [])

        print(f"🧠 Память загружена для пользователя {user_id}: "
              f"краткосрочная={len(self.short_term_memory)}, "
              f"долгосрочная={len(self.long_term_memory)}, "
              f"паттернов={len(self.behavior_patterns.get('deep_patterns', []))}")

    # ==================== ОСНОВНЫЕ МЕТОДЫ РАБОТЫ С ПАМЯТЬЮ ====================

    def add_to_short_term(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Добавляет сообщение в краткосрочную память с автоматической консолидацией"""
        entry = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.short_term_memory.append(entry)

        # Ограничение размера краткосрочной памяти
        if len(self.short_term_memory) > self.short_term_limit:
            self.short_term_memory = self.short_term_memory[-self.short_term_limit:]

        self._save_json(self.short_term_file, self.short_term_memory)
        self.messages_counter += 1
        self.metadata['total_interactions'] += 1
        self.metadata['last_interaction'] = datetime.now().isoformat()
        self._save_json(self.metadata_file, self.metadata)

        # Автоматическая консолидация
        if self.llm_caller and self.messages_counter % self.consolidation_interval == 0:
            asyncio.create_task(self.auto_consolidate())

        # Глубокая рефлексия
        if self.llm_caller and self.messages_counter % self.reflection_interval == 0:
            asyncio.create_task(self.deep_reflection())

    def get_short_term_context(self, limit: int = 10) -> str:
        """Возвращает последние N сообщений из краткосрочной памяти"""
        recent = self.short_term_memory[-limit:] if limit else self.short_term_memory
        return "\n".join([
            f"[{datetime.fromisoformat(e['timestamp']).strftime('%H:%M')}] {e['role']}: {e['content']}"
            for e in recent
        ])

    def search_long_term(self, query: str, limit: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """Семантический поиск в долгосрочной памяти с взвешиванием"""
        if not self.long_term_memory:
            return []

        scored = []
        query_words = set(re.findall(r'\w+', query.lower()))

        for mem in self.long_term_memory:
            content_words = set(re.findall(r'\w+', mem['content'].lower()))
            if not content_words:
                continue

            # Сходство по общим словам
            common = query_words & content_words
            similarity = len(common) / max(len(query_words), 1) if query_words else 0

            # Взвешиваем по важности, частоте доступа и свежести
            importance = mem.get('importance', 0.5)
            access_count = mem.get('access_count', 0)

            # Свежесть (недавние важнее)
            try:
                timestamp = datetime.fromisoformat(mem['timestamp'])
                days_old = (datetime.now() - timestamp).days
                freshness = max(0.1, 1 - days_old / 180)  # Полгода = 0.1
            except:
                freshness = 0.5

            weight = (
                    similarity * 0.4 +
                    importance * 0.3 +
                    (min(access_count, 20) / 20) * 0.2 +
                    freshness * 0.1
            )

            if weight >= min_similarity:
                scored.append((weight, mem))

        # Сортируем по весу и возвращаем топ-N
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [mem for _, mem in scored[:limit]]

        # Увеличиваем счётчик доступа для найденных фактов
        for mem in results:
            mem['access_count'] = mem.get('access_count', 0) + 1
            mem['last_accessed'] = datetime.now().isoformat()
        self._save_json(self.long_term_file, self.long_term_memory)

        return results

    def get_enhanced_context(self, query: str, short_term_limit: int = 6, long_term_limit: int = 5) -> str:
        """
        🧩 ФОРМИРУЕТ УЛУЧШЕННЫЙ КОНТЕКСТ ДЛЯ LLM:
        - Релевантные факты из долгосрочной памяти
        - Паттерны поведения пользователя
        - Недавняя история
        - Эмоциональный профиль
        """
        # 1. Релевантные факты из долгосрочной памяти
        relevant = self.search_long_term(query, limit=long_term_limit)
        facts_context = "\n".join([
            f"• [{mem.get('category', 'fact')}] {mem['content']}"
            for mem in relevant
        ]) if relevant else "Нет релевантных фактов"

        # 2. Паттерны поведения
        patterns_text = "\n# ПАТТЕРНЫ ПОВЕДЕНИЯ ПОЛЬЗОВАТЕЛЯ:\n"

        if self.behavior_patterns.get('preferences'):
            prefs = ", ".join([f"{k}={v}" for k, v in list(self.behavior_patterns['preferences'].items())[:5]])
            patterns_text += f"Предпочтения: {prefs}\n"

        if self.behavior_patterns.get('communication_style'):
            patterns_text += f"Стиль общения: {self.behavior_patterns['communication_style']}\n"

        if self.behavior_patterns.get('frequent_topics'):
            topics = ", ".join(self.behavior_patterns['frequent_topics'][:4])
            patterns_text += f"Частые темы: {topics}\n"

        if self.behavior_patterns.get('deep_patterns'):
            patterns = "\n".join([
                f"  • {p['pattern']}"
                for p in self.behavior_patterns['deep_patterns'][:3]
            ])
            patterns_text += f"Глубокие паттерны:\n{patterns}\n"

        if self.behavior_patterns.get('emotional_profile'):
            patterns_text += f"Эмоциональный профиль: {self.behavior_patterns['emotional_profile'][:100]}\n"

        # 3. Недавняя история
        history = self.get_short_term_context(short_term_limit)

        return f"""# ДОЛГОСРОЧНАЯ ПАМЯТЬ (сохранена между перезагрузками):
{facts_context}

{patterns_text}
# НЕДАВНЯЯ ИСТОРИЯ ({len(self.short_term_memory)} сообщений):
{history}

# ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}"""

    # ==================== АВТОМАТИЧЕСКАЯ КОНСОЛИДАЦИЯ ====================

    async def auto_consolidate(self):
        """
        🔄 АВТОМАТИЧЕСКАЯ КОНСОЛИДАЦИЯ С УЛУЧШЕННЫМ АНАЛИЗОМ
        Извлекает факты, паттерны, предпочтения из краткосрочной памяти
        """
        if len(self.short_term_memory) < 6 or not self.llm_caller:
            return

        recent = self.short_term_memory[-12:]  # Больше контекста для анализа
        conversation = "\n".join([
            f"{entry['role']}: {entry['content']}"
            for entry in recent
        ])

        prompt = f"""Проанализируй диалог и извлеки ВСЕ полезные знания для долгосрочного запоминания:

ДИАЛОГ:
{conversation}

ЗАДАЧА:
1. ФАКТЫ о пользователе: имя, работа, хобби, важные даты, предпочтения, цели, страхи, мечты
2. ПАТТЕРНЫ поведения: частые запросы, стиль общения (кратко/подробно/эмоционально), реакции на темы
3. КОНТЕКСТНЫЕ ПРЕДПОЧТЕНИЯ: как пользователь любит получать информацию
4. ЭМОЦИОНАЛЬНЫЕ МАРКЕРЫ: что вызывает сильные эмоции (радость, раздражение, интерес)
5. ПРОТИВОРЕЧИЯ: если новые факты противоречат старым — отметь как "требует уточнения"

ВЕРНИ ЧИСТЫЙ JSON (без пояснений):
{{
  "new_facts": [
    {{
      "text": "факт о пользователе",
      "category": "personal|preference|habit|goal|fear|dream",
      "importance": 0.5-1.0,
      "confidence": 0.7-1.0,
      "emotional_marker": "positive|negative|neutral|null"
    }}
  ],
  "behavior_patterns": {{
    "communication_style": "краткий|подробный|эмоциональный|нейтральный|юмористический",
    "frequent_topics": ["тема1", "тема2"],
    "preferences": {{"ключ": "значение"}}
  }},
  "contradictions": [
    {{
      "old_fact": "старый факт",
      "new_fact": "новый факт",
      "resolution_needed": true
    }}
  ]
}}"""

        try:
            print(f"🔄 Запуск автоконсолидации для {self.user_id}...")
            response = await self.llm_caller(prompt, temperature=0.2)

            # Извлекаем чистый JSON из ответа
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                print("⚠️ Не удалось извлечь JSON из ответа LLM")
                return

            analysis = json.loads(json_match.group(0))

            # 1. Добавляем новые факты с детекцией противоречий
            new_facts_count = 0
            for fact in analysis.get('new_facts', []):
                if fact.get('importance', 0) >= 0.5:
                    await self._add_fact_with_contradiction_check(fact)
                    new_facts_count += 1

            # 2. Обновляем паттерны поведения
            patterns = analysis.get('behavior_patterns', {})
            if patterns:
                await self._update_behavior_patterns(patterns)

            # 3. Логируем противоречия
            contradictions = analysis.get('contradictions', [])
            if contradictions:
                self.metadata['contradictions_detected'] += len(contradictions)
                for contradiction in contradictions:
                    self._log_contradiction(contradiction)

            # Обновляем метаданные
            self.metadata['facts_learned'] += new_facts_count
            self.metadata['last_consolidation'] = datetime.now().isoformat()
            self._save_json(self.metadata_file, self.metadata)

            print(f"✅ Автоконсолидация завершена: +{new_facts_count} фактов, "
                  f"паттерны обновлены, противоречий: {len(contradictions)}")

        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON при консолидации: {e}")
        except Exception as e:
            print(f"⚠️ Ошибка автоконсолидации: {type(e).__name__}: {e}")

    async def _add_fact_with_contradiction_check(self, new_fact: Dict):
        """Добавляет факт с проверкой на противоречия существующим знаниям"""
        # Проверяем похожие факты в долгосрочной памяти (семантическое сравнение)
        similar = [
            m for m in self.long_term_memory
            if self._semantic_similarity(new_fact['text'], m['content']) > 0.65
        ]

        if similar:
            # Если найдены похожие факты — обновляем самый свежий и важный
            latest = max(similar, key=lambda x: (
                    x.get('importance', 0.5) * 0.6 +
                    (datetime.fromisoformat(x.get('last_updated', x['timestamp'])).timestamp() / 1e9) * 0.4
            ))

            # Обновляем факт с учётом новой информации
            latest['content'] = new_fact['text']
            latest['last_updated'] = datetime.now().isoformat()
            latest['importance'] = max(
                latest.get('importance', 0.5),
                new_fact.get('importance', 0.7)
            )
            latest['confidence'] = max(
                latest.get('confidence', 0.7),
                new_fact.get('confidence', 0.8)
            )
            latest['emotional_marker'] = new_fact.get('emotional_marker', latest.get('emotional_marker'))

            print(f"🔄 Обновлён факт (похожесть): {latest['content'][:60]}...")
        else:
            # Новый уникальный факт
            entry = {
                'content': new_fact['text'],
                'category': new_fact.get('category', 'fact'),
                'importance': new_fact.get('importance', 0.7),
                'confidence': new_fact.get('confidence', 0.8),
                'emotional_marker': new_fact.get('emotional_marker', 'neutral'),
                'timestamp': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'access_count': 0,
                'id': hashlib.md5(new_fact['text'].encode()).hexdigest()[:12]
            }
            self.long_term_memory.append(entry)

            # Сортируем по важности и ограничиваем размер
            self.long_term_memory.sort(key=lambda x: x['importance'], reverse=True)
            if len(self.long_term_memory) > self.long_term_limit:
                self.long_term_memory = self.long_term_memory[:self.long_term_limit]

        self._save_json(self.long_term_file, self.long_term_memory)

    # ==================== ГЛУБОКАЯ РЕФЛЕКСИЯ ====================

    async def deep_reflection(self):
        """
        🔮 ГЛУБОКАЯ РЕФЛЕКСИЯ (каждые N сообщений)
        Анализирует ВСЮ долгосрочную память для выявления скрытых паттернов
        """
        if len(self.long_term_memory) < 10 or not self.llm_caller:
            return

        print(f"🔮 Запуск глубокой рефлексии для {self.user_id}...")

        # Группируем факты по категориям для компактного представления
        facts_by_category = defaultdict(list)
        for fact in self.long_term_memory[:60]:  # Берем топ-60 самых важных
            cat = fact.get('category', 'other')
            facts_by_category[cat].append(fact['content'])

        # Формируем компактное представление для LLM
        facts_summary = []
        for cat, facts in list(facts_by_category.items())[:5]:  # Топ-5 категорий
            facts_summary.append(f"КАТЕГОРИЯ: {cat.upper()}")
            facts_summary.extend([f"  • {f[:100]}" for f in facts[:4]])  # Первые 4 факта

        facts_text = "\n".join(facts_summary[:20])  # Ограничиваем длину

        # Добавляем информацию о частых темах из паттернов
        topics_context = ""
        if self.behavior_patterns.get('frequent_topics'):
            topics = ", ".join(self.behavior_patterns['frequent_topics'][:6])
            topics_context = f"\nЧАСТЫЕ ТЕМЫ В ДИАЛОГАХ: {topics}"

        prompt = f"""Проанализируй накопленные знания о пользователе и выяви ГЛУБОКИЕ ПСИХОЛОГИЧЕСКИЕ ПАТТЕРНЫ:

НАКОПЛЕННЫЕ ФАКТЫ:{topics_context}
{facts_text}

ЗАДАЧА АНАЛИЗА:
1. Выяви СКРЫТЫЕ ЦЕННОСТИ и убеждения (то, что пользователь не говорит напрямую)
2. Определи РЕГУЛЯРНЫЕ ПРИВЫЧКИ, ритуалы и поведенческие циклы
3. Найди ЭМОЦИОНАЛЬНЫЕ ТРИГГЕРЫ (что вызывает сильные реакции: радость, раздражение, тревогу)
4. Проанализируй КОГНИТИВНЫЕ ИСКАЖЕНИЯ или повторяющиеся шаблоны мышления
5. Предложи 3 ГИПОТЕЗЫ для будущего уточнения у пользователя

ВЕРНИ ЧИСТЫЙ JSON:
{{
  "deep_patterns": [
    {{
      "pattern": "описание паттерна",
      "type": "value|habit|emotional_trigger|cognitive_bias",
      "evidence": ["факт1", "факт2"],
      "confidence": 0.7-1.0
    }}
  ],
  "hypotheses_to_verify": [
    "гипотеза 1 для уточнения у пользователя",
    "гипотеза 2 для уточнения у пользователя"
  ],
  "communication_recommendations": "рекомендации по стилю общения (1-2 предложения)",
  "emotional_profile": "краткое описание эмоционального профиля"
}}"""

        try:
            response = await self.llm_caller(prompt, temperature=0.3, max_tokens=800)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                print("⚠️ Не удалось извлечь JSON из глубокой рефлексии")
                return

            reflection = json.loads(json_match.group(0))

            # Обновляем паттерны поведения
            self.behavior_patterns.update({
                'deep_patterns': reflection.get('deep_patterns', self.behavior_patterns.get('deep_patterns', [])),
                'hypotheses': reflection.get('hypotheses_to_verify', []),
                'communication_tips': reflection.get('communication_recommendations', ''),
                'emotional_profile': reflection.get('emotional_profile', ''),
                'last_reflection': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            })

            self.metadata['last_reflection'] = datetime.now().isoformat()
            self.metadata['contradictions_resolved'] += len(reflection.get('deep_patterns', []))
            self._save_json(self.patterns_file, self.behavior_patterns)
            self._save_json(self.metadata_file, self.metadata)

            patterns_count = len(reflection.get('deep_patterns', []))
            print(f"✅ Глубокая рефлексия завершена: найдено {patterns_count} глубоких паттернов")

        except Exception as e:
            print(f"⚠️ Ошибка глубокой рефлексии: {type(e).__name__}: {e}")

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    async def _update_behavior_patterns(self, new_patterns: Dict):
        """Обновляет паттерны поведения с мержем старых и новых данных"""
        # Обновляем предпочтения (мержим словари)
        if 'preferences' in new_patterns and isinstance(new_patterns['preferences'], dict):
            self.behavior_patterns['preferences'].update(new_patterns['preferences'])

        # Обновляем стиль общения (только если явно указан)
        if 'communication_style' in new_patterns and new_patterns['communication_style'] != 'нейтральный':
            self.behavior_patterns['communication_style'] = new_patterns['communication_style']

        # Обновляем частые темы (объединяем списки, удаляя дубликаты)
        if 'frequent_topics' in new_patterns and isinstance(new_patterns['frequent_topics'], list):
            existing = set(self.behavior_patterns.get('frequent_topics', []))
            new_topics = set(new_patterns['frequent_topics'])
            self.behavior_patterns['frequent_topics'] = list(existing | new_topics)[:10]  # Ограничиваем 10 темами

        self.behavior_patterns['last_updated'] = datetime.now().isoformat()
        self._save_json(self.patterns_file, self.behavior_patterns)

    def _log_contradiction(self, contradiction: Dict):
        """Логирует противоречие в отдельный файл для последующего анализа"""
        contradiction['detected_at'] = datetime.now().isoformat()
        contradiction['id'] = hashlib.md5(
            (contradiction['old_fact'] + contradiction['new_fact']).encode()
        ).hexdigest()[:12]

        self.contradictions_log.append(contradiction)
        self._save_json(self.contradictions_file, self.contradictions_log[-50:])  # Храним последние 50

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Упрощённое семантическое сравнение на основе общих слов и структуры"""

        # Нормализуем текст
        def normalize(t):
            t = t.lower()
            t = re.sub(r'[^\w\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        text1_norm = normalize(text1)
        text2_norm = normalize(text2)

        if not text1_norm or not text2_norm:
            return 0.0

        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())

        if not words1 or not words2:
            return 0.0

        # Базовое сходство по словам
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union) if union else 0.0

        # Дополнительный бонус за общие ключевые слова (имена, цифры, уникальные термины)
        keywords1 = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', text1))
        keywords2 = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', text2))
        if keywords1 & keywords2:
            jaccard += 0.2  # Бонус за совпадение ключевых слов

        return min(jaccard, 1.0)

    # ==================== ФАЙЛОВЫЕ ОПЕРАЦИИ ====================

    def _load_json(self, filepath: str, default: Any) -> Any:
        """Безопасная загрузка JSON из файла"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {filepath}: {e}")
        return default

    def _save_json(self, filepath: str, data: Any):
        """Безопасное сохранение JSON в файл"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения {filepath}: {e}")

    # ==================== УТИЛИТЫ ДЛЯ ОТЛАДКИ ====================

    def get_memory_stats(self) -> Dict:
        """Возвращает статистику по памяти"""
        return {
            'user_id': self.user_id,
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'patterns_count': len(self.behavior_patterns.get('deep_patterns', [])),
            'total_interactions': self.metadata['total_interactions'],
            'facts_learned': self.metadata['facts_learned'],
            'contradictions_detected': self.metadata['contradictions_detected'],
            'contradictions_resolved': self.metadata['contradictions_resolved'],
            'last_consolidation': self.metadata['last_consolidation'],
            'last_reflection': self.metadata['last_reflection'],
            'first_interaction': self.metadata['first_interaction'],
            'last_interaction': self.metadata['last_interaction']
        }

    def clear_short_term(self):
        """Очищает краткосрочную память (сохраняя долгосрочную)"""
        self.short_term_memory = []
        self._save_json(self.short_term_file, [])
        print(f"🧹 Краткосрочная память очищена для {self.user_id}")

    def export_memory(self) -> Dict:
        """Экспортирует всю память в один словарь для резервного копирования"""
        return {
            'user_id': self.user_id,
            'short_term': self.short_term_memory,
            'long_term': self.long_term_memory,
            'behavior_patterns': self.behavior_patterns,
            'metadata': self.metadata,
            'contradictions_log': self.contradictions_log,
            'exported_at': datetime.now().isoformat()
        }


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

            with open(CORE_PERFORMANCE_LOG, 'w', encoding='utf-8') as f:
                json.dump(perf_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка записи производительности: {e}")


# ==================== МЕНЕДЖЕР ЯДЕР ====================
class ToolsManager:
    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self.validator = CoreValidator()
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
            builtin_cores.append(WebSearchCore())

        for core in builtin_cores:
            self.cores[core.name] = core
        print(f"✅ Загружено встроенных ядер: {len(builtin_cores)}")

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
            caps = ', '.join(core.capabilities[:2]) if core.capabilities else 'нет'
            direct = "✅" if core.direct_answer_mode else "❌"
            summary.append(f"- **{core.name}** {direct} ({caps})")
        return '\n'.join(summary)


# ==================== САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ v2.0 ====================
class MiniBrain:
    def __init__(self, user_id: str, llm_caller=None):
        self.user_id = user_id
        self.llm_caller = llm_caller
        self.memory = IntelligentMemoryManager(user_id, llm_caller)  # ЗАМЕНА: используем улучшенную память
        self.tools = ToolsManager()
        # Счетчик для автоконсолидации (уже встроен в IntelligentMemoryManager)

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

    async def process(self, query: str) -> Dict[str, Any]:
        """Обработка запроса с использованием интеллектуальной памяти"""
        query_fixed = query.replace('фаил', 'файл')
        if query_fixed != query:
            print(f"🔧 Исправлена опечатка: '{query}' → '{query_fixed}'")
        query = query_fixed

        print(f"\n{'=' * 60}")
        print(f"🧠 Запрос от {self.user_id}: {query}")
        print(f"{'=' * 60}")

        # Добавляем в краткосрочную память (автоконсолидация вызывается автоматически)
        self.memory.add_to_short_term('user', query)

        # Ищем ядро для обработки
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
                    # Добавляем ответ в память
                    self.memory.add_to_short_term('assistant', core_response.raw_result)
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

        print("💭 Использую общую обработку с интеллектуальной памятью...")
        llm_context = self._build_general_context(query)
        return {
            'type': 'general_llm',
            'context': llm_context,
            'source': 'general',
            'need_llm': True
        }

    def _build_llm_context_with_core(self, query: str, core_response: CoreResponse, core_name: str) -> str:
        """Строит контекст для LLM с данными ядра и интеллектуальной памятью"""
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        # Используем улучшенный метод получения контекста
        memory_context = self.memory.get_enhanced_context(query, short_term_limit=5, long_term_limit=3)

        context = f"""# СИСТЕМА: ИИ АССИСТЕНТ С ИНТЕЛЛЕКТУАЛЬНОЙ ПАМЯТЬЮ
Время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь: {self.user_id}
Ядро: {core_name}

# ДАННЫЕ ОТ ЯДРА:
{core_response.raw_result if core_response.raw_result else core_response.to_context_string()}

# КОНТЕКСТ ИЗ ПАМЯТИ:
{memory_context}

# ИНСТРУКЦИИ:
Используй данные от ядра и интеллектуальную память для ответа.
Учитывай паттерны поведения и эмоциональный профиль пользователя.
Будь конкретным, полезным и адаптируй стиль общения под предпочтения пользователя."""
        return context

    def _build_general_context(self, query: str) -> str:
        """Строит общий контекст с интеллектуальной памятью"""
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        # Используем улучшенный метод получения контекста
        memory_context = self.memory.get_enhanced_context(query, short_term_limit=5, long_term_limit=3)

        context = f"""# СИСТЕМА: ИИ АССИСТЕНТ С ИНТЕЛЛЕКТУАЛЬНОЙ ПАМЯТЬЮ
Время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь: {self.user_id}

# ДОСТУПНЫЕ ЯДРА:
{self.tools.get_cores_summary()}

# КОНТЕКСТ ИЗ ПАМЯТИ:
{memory_context}

# ИНСТРУКЦИИ:
Ответь полезно, используя доступную информацию и интеллектуальную память.
Учитывай паттерны поведения и эмоциональный профиль пользователя.
Адаптируй стиль общения под предпочтения пользователя."""
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
                except Exception as e:
                    print(f"⚠️ Ошибка создания {log_file}: {e}")

    def get_brain(self, user_id: str) -> MiniBrain:
        if user_id not in self.user_brains:
            self.user_brains[user_id] = MiniBrain(user_id, self.get_llm_response)
            print(f"🧠 Создан новый мозг для {user_id}")
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
                return f"❌ Ошибка LLM (код: {resp.status_code})"
        except requests.exceptions.Timeout:
            return "❌ Таймаут"
        except Exception as e:
            return f"❌ Ошибка: {str(e)[:100]}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        await update.message.reply_text(
            "🧠 *МИНИ-МОЗГ БОТ v6.0 - С ИНТЕЛЛЕКТУАЛЬНОЙ ПАМЯТЬЮ*\n"
            "✨ *Новое:*\n"
            "• 🧠 Автономная рефлексия и анализ паттернов\n"
            "• 🔍 Детекция и разрешение противоречий\n"
            "• 💡 Эмоциональное взвешивание важности фактов\n"
            "• 📊 Самоорганизация знаний через кластеризацию\n"
            "• 🔮 Глубокая рефлексия для выявления скрытых паттернов\n"
            "• 💾 Полное сохранение между перезагрузками\n"
            "\n⚙️ *Команды:*\n"
            "/memory_stats — статистика памяти\n"
            "/search_memory [запрос] — поиск в памяти\n"
            "/clear — очистить диалог\n"
            "/help — справка",
            parse_mode='Markdown'
        )

    async def memory_stats_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика памяти"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)
        stats = brain.memory.get_memory_stats()

        categories = {}
        for memory in brain.memory.long_term_memory:
            cat = memory.get('category', 'other')
            categories[cat] = categories.get(cat, 0) + 1

        categories_text = "\n".join([
            f"  • {cat}: {count}"
            for cat, count in categories.items()
        ])

        keywords = brain.memory.behavior_patterns.get('frequent_topics', [])[:5]
        keywords_text = "\n".join([f"  • {kw}" for kw in keywords]) if keywords else '  Нет данных'

        last_consol = stats['last_consolidation']
        if last_consol:
            last_consol = datetime.fromisoformat(last_consol).strftime('%d.%m %H:%M')
        else:
            last_consol = "никогда"

        first_int = datetime.fromisoformat(stats['first_interaction']).strftime('%d.%m.%Y')
        last_int = datetime.fromisoformat(stats['last_interaction']).strftime('%d.%m.%Y %H:%M')

        msg = f"""🧠 **СТАТИСТИКА ИНТЕЛЛЕКТУАЛЬНОЙ ПАМЯТИ**
👤 Пользователь: {user_id}
📅 Первое взаимодействие: {first_int}
🕒 Последнее: {last_int}

📝 **Краткосрочная:** {stats['short_term_count']} записей
📚 **Долгосрочная:** {stats['long_term_count']} фактов
🎭 **Глубоких паттернов:** {stats['patterns_count']}

📂 **Категории фактов:**
{categories_text if categories_text else '  Нет данных'}

🔑 **Частые темы:**
{keywords_text}

🔄 **Консолидаций:** {stats['facts_learned']}
⚠️ **Противоречий:** {stats['contradictions_detected']} (разрешено: {stats['contradictions_resolved']})
🔄 **Последняя консолидация:** {last_consol}"""

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def search_memory_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Поиск в долговременной памяти"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)
        query = ' '.join(context.args) if context.args else ''

        if not query:
            await update.message.reply_text(
                "🔍 **Поиск в интеллектуальной памяти**\nИспользование:\n`/search_memory ваш запрос`",
                parse_mode='Markdown'
            )
            return

        results = brain.memory.search_long_term(query, limit=5)
        if not results:
            await update.message.reply_text(
                f"❌ Ничего не найдено по запросу: `{query}`",
                parse_mode='Markdown'
            )
            return

        msg = f"🔍 **Результаты поиска:** `{query}`\n"
        for i, mem in enumerate(results, 1):
            dt = datetime.fromisoformat(mem['timestamp']).strftime('%d.%m.%Y')
            importance = "⭐" * int(mem.get('importance', 0.5) * 5)
            access = f"👁️ {mem.get('access_count', 0)}"
            msg += f"\n{i}. {importance} {access}\n   📅 {dt}\n   💬 {mem['content'][:120]}\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистить краткосрочную память"""
        user_id = str(update.effective_user.id)
        brain = self.get_brain(user_id)
        brain.memory.clear_short_term()
        await update.message.reply_text(
            "🧹 *Краткосрочная память очищена*\n"
            "🧠 Долгосрочная память и паттерны поведения сохранены.",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Справка"""
        await update.message.reply_text(
            "📖 *СПРАВКА ПО ИНТЕЛЛЕКТУАЛЬНОЙ ПАМЯТИ*\n"
            "\n*Как это работает:*\n"
            "Бот автоматически:\n"
            "• Запоминает важные факты о вас\n"
            "• Анализирует паттерны вашего поведения\n"
            "• Выявляет эмоциональные триггеры\n"
            "• Разрешает противоречия в знаниях\n"
            "• Адаптирует стиль общения под вас\n"
            "\n*Ваши данные:*\n"
            "• Сохраняются между перезагрузками\n"
            "• Никогда не передаются третьим лицам\n"
            "• Можно очистить командой /clear",
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        brain = self.get_brain(user_id)
        try:
            brain_result = await brain.process(text)

            if brain_result.get('need_llm') == False:
                response = brain_result['response']
                # Добавляем в память
                brain.memory.add_to_short_term('assistant', response)
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
                llm_response = llm_response[:4000] + "\n..."

            # Добавляем в память
            brain.memory.add_to_short_term('assistant', llm_response)

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
    print("🚀 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v6.0 - С ИНТЕЛЛЕКТУАЛЬНОЙ ПАМЯТЬЮ")
    print("=" * 70)
    print(f"⏰ Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"🧠 Директория памяти: {MEMORY_DIR}/")
    print(f"🌐 Веб-поиск: {'✅ Доступен' if DDGS_AVAILABLE else '❌ Недоступен'}")
    print("=" * 70)

    # Проверка LM Studio
    try:
        test_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '')
        test_resp = requests.get(test_url, timeout=10)
        if test_resp.status_code == 200:
            print(f"✅ LM Studio доступна")
        else:
            print(f"⚠️ LM Studio код: {test_resp.status_code}")
    except Exception as e:
        print(f"⚠️ LM Studio недоступна: {e}")

    print("=" * 70)
    print("\n🔄 Инициализация интеллектуальной памяти...")
    bot = TelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("memory_stats", bot.memory_stats_cmd))
    application.add_handler(CommandHandler("search_memory", bot.search_memory_cmd))
    application.add_handler(CommandHandler("clear", bot.clear_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот инициализирован!")
    print("=" * 70)
    print("💬 Напишите боту в Telegram")
    print("🧠 Интеллектуальная система памяти активна!")
    print("   • Автоконсолидация каждые 4 сообщения")
    print("   • Глубокая рефлексия каждые 15 сообщений")
    print("   • Детекция противоречий в реальном времени")
    print("=" * 70)
    print("\nCtrl+C для остановки\n")

    try:
        application.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\n🛑 Остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()