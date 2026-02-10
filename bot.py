#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v4.0 - СТАБИЛЬНАЯ ВЕРСИЯ
✅ Модель создает только РАБОЧИЕ ядра
✅ ПРИНУЖДЕНИЕ использовать ответы ядер
✅ Автоматическая валидация и тестирование
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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# DuckDuckGo Search импорт
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

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
os.makedirs(CORES_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# Файлы
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
CORE_PERFORMANCE_LOG = os.path.join(MEMORY_DIR, "core_performance.json")
REJECTED_CORES_LOG = os.path.join(MEMORY_DIR, "rejected_cores.json")

# Создаем __init__.py для импортов
with open(os.path.join(CORES_DIR, "__init__.py"), "w") as f:
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
            direct_answer: bool = False  # True = можно использовать как финальный ответ
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
            formatted_data = json.dumps(self.data, ensure_ascii=False, indent=2)
            return f"📊 СТРУКТУРИРОВАННЫЕ ДАННЫЕ ОТ '{self.source.upper()}':\n{formatted_data}"

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
    priority: int = 5  # 1 = высший, 10 = низший
    direct_answer_mode: bool = False  # Если True - ответ можно использовать напрямую

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
        return f"{self.description} ({', '.join(self.capabilities[:3])})"


# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
    name = "datetime_core"
    description = "Точная информация о дате, времени и днях недели"
    capabilities = ["текущая дата", "время", "день недели", "расчёт дат"]
    priority = 1
    direct_answer_mode = True  # Ответы всегда точные

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        keywords = [
            'какой сегодня день', 'какое число', 'день недели', 'сколько времени',
            'который час', 'текущая дата', 'сегодня', 'завтра', 'послезавтра',
            'вчера', 'через', 'дней', 'дня', 'какой день был'
        ]
        return any(kw in q for kw in keywords)

    def get_confidence(self, query: str) -> float:
        q = query.lower()
        high_conf = ['какой сегодня', 'какое число', 'который час', 'сколько времени']
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
            q_lower = query.lower()

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
        q = query.lower()
        # Математические выражения
        has_math = bool(re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', q))
        # Слова-триггеры
        has_words = any(word in q for word in ['сколько будет', 'посчитай', 'вычисли', 'равно', 'реши'])
        return has_math or (has_words and any(char.isdigit() for char in q))

    def get_confidence(self, query: str) -> float:
        q = query.lower()
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

            # Ищем выражение в тексте
            if not expr:
                # Пробуем найти "сколько будет X + Y"
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
        self.ddgs = DDGS()
        self.cache = {}

    def can_handle(self, query: str) -> bool:
        q = query.lower()

        # Исключаем запросы для других ядер
        exclude_keywords = [
            'какой сегодня день', 'какое число', 'который час', 'сколько времени',
            'сколько будет', '+', '-', '*', '/', 'умножить', 'разделить', 'прочитай файл',
            'сохрани файл', 'список файлов'
        ]
        if any(kw in q for kw in exclude_keywords):
            return False

        # Ключевые слова для поиска
        search_keywords = [
            'новост', 'курс', 'погод', 'прогноз', 'событи', 'произошло',
            'последни', 'актуальн', 'сейчас', 'сегодняшн', 'текущ',
            'кто выиграл', 'результат', 'итоги', 'цена', 'стоимость',
            'информаци', 'узнай', 'найди', 'что такое', 'кто такой'
        ]
        return any(kw in q for kw in search_keywords)

    def get_confidence(self, query: str) -> float:
        q = query.lower()
        high_priority = ['курс доллара', 'новости сегодня', 'погода сейчас', 'что случилось']
        if any(kw in q for kw in high_priority):
            return 0.85
        return 0.6 if self.can_handle(query) else 0.0

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            print(f"  🌐 Веб-поиск: {query[:60]}...")

            # Кэширование
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            if query_hash in self.cache:
                if time.time() - self.cache[query_hash]['timestamp'] < 300:  # 5 минут
                    results = self.cache[query_hash]['results']
                else:
                    del self.cache[query_hash]
                    results = self.ddgs.text(query, max_results=5)
            else:
                results = self.ddgs.text(query, max_results=5)
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


# ==================== ВАЛИДАТОР ЯДЕР ====================
class CoreValidator:
    """Валидатор и тестер ядер знаний"""

    @staticmethod
    def validate_code_structure(code: str) -> Tuple[bool, str]:
        """Проверяет структуру кода ядра"""
        try:
            # Базовые проверки
            if "class " not in code or "KnowledgeCore" not in code:
                return False, "Отсутствует наследование от KnowledgeCore"

            if "def can_handle" not in code:
                return False, "Отсутствует метод can_handle"

            if "def execute" not in code:
                return False, "Отсутствует метод execute"

            if "CoreResponse" not in code:
                return False, "Отсутствует возврат CoreResponse"

            # Проверка атрибутов
            required_attrs = ['name', 'description', 'capabilities']
            for attr in required_attrs:
                if f"{attr} =" not in code and f'{attr} =' not in code:
                    return False, f"Отсутствует атрибут {attr}"

            # Синтаксическая проверка
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
            # Имя
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', code)
            if name_match:
                info['name'] = name_match.group(1)

            # Описание
            desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', code)
            if desc_match:
                info['description'] = desc_match.group(1)

            # Возможности
            caps_match = re.search(r'capabilities\s*=\s*\[(.*?)\]', code, re.DOTALL)
            if caps_match:
                caps_text = caps_match.group(1)
                caps = re.findall(r'["\']([^"\']+)["\']', caps_text)
                info['capabilities'] = caps

            # Приоритет
            prio_match = re.search(r'priority\s*=\s*(\d+)', code)
            if prio_match:
                info['priority'] = int(prio_match.group(1))

            # Использование web_search
            if 'web_search' in code:
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

        # Анализ результатов
        success_count = sum(1 for r in results if r.get('success', False))
        has_raw_results = sum(1 for r in results if r.get('has_raw_result', False))

        passed = success_count >= len(test_queries) * 0.5  # 50% успеха

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
            except:
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
        query_lower = query.lower()
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
                with open(LEARNING_LOG, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

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
                with open(REJECTED_CORES_LOG, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

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
                with open(CORE_PERFORMANCE_LOG, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)

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

            # Сохраняем последние 5 запросов
            perf_data[core_name]['queries'].append({
                'query': query[:100],
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
            perf_data[core_name]['queries'] = perf_data[core_name]['queries'][-5:]

            # Удаляем неэффективные ядра из лога
            for name in list(perf_data.keys()):
                if perf_data[name]['total_count'] > 10:
                    success_rate = perf_data[name]['success_count'] / perf_data[name]['total_count']
                    if success_rate < 0.3:  # Успешность меньше 30%
                        print(f"🗑️ Удаляю неэффективное ядро из лога: {name} ({success_rate:.1%})")
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
                return log_data[-limit:] if log_data else []
        except:
            pass
        return []


# ==================== УЛУЧШЕННЫЙ МЕНЕДЖЕР ЯДЕР ====================
class ToolsManager:
    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self.validator = CoreValidator()
        self._load_builtin_cores()
        self._load_dynamic_cores()
        self._cleanup_bad_cores()

    def _load_builtin_cores(self):
        builtin_cores = [
            DateTimeCore(),
            CalculatorCore(),
            WebSearchCore()
        ]

        for core in builtin_cores:
            self.cores[core.name] = core

        print(f"✅ Загружено встроенных ядер: {len(builtin_cores)}")
        for core in builtin_cores:
            print(f"   - {core.name}: {core.description}")

    def _load_dynamic_cores(self):
        loaded_count = 0
        rejected_count = 0

        for fname in os.listdir(CORES_DIR):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue

            path = os.path.join(CORES_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Валидация структуры
                is_valid, reason = self.validator.validate_code_structure(code)
                if not is_valid:
                    print(f"  ❌ Плохое ядро {fname}: {reason}")
                    rejected_count += 1
                    continue

                # Загружаем ядро
                core_info = self.validator.extract_core_info(code)
                core_name = core_info['name']

                # Проверяем дубликаты
                if core_name in self.cores:
                    print(f"  ⚠️ Дубликат ядра {core_name}, пропускаю")
                    continue

                # Создаем namespace с безопасными импортами
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
                    'hashlib': hashlib
                }

                # Добавляем DDGS и requests только если они используются
                if 'DDGS' in code or 'duckduckgo' in code:
                    namespace['DDGS'] = DDGS
                if 'requests' in code:
                    namespace['requests'] = requests

                exec(code, namespace)

                # Ищем класс ядра
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

                # Создаем экземпляр и тестируем
                core_instance = core_class()

                # Тестовые запросы
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

                # Сохраняем ядро
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
                # Проверяем базовые методы
                if not hasattr(core, 'can_handle') or not callable(core.can_handle):
                    bad_cores.append(core_name)
                    continue

                if not hasattr(core, 'execute') or not callable(core.execute):
                    bad_cores.append(core_name)
                    continue

                # Тестовый запрос
                test_response = core.execute("тест")
                if not isinstance(test_response, CoreResponse):
                    bad_cores.append(core_name)

            except Exception as e:
                print(f"  ❌ Ошибка проверки ядра {core_name}: {e}")
                bad_cores.append(core_name)

        # Удаляем плохие ядра
        for core_name in bad_cores:
            if core_name in self.cores:
                del self.cores[core_name]
                print(f"🗑️ Удалено нерабочее ядро: {core_name}")

                # Удаляем файл
                core_file = os.path.join(CORES_DIR, f"{core_name}.py")
                if os.path.exists(core_file):
                    os.remove(core_file)

    def find_best_core(self, query: str) -> Optional[Tuple[KnowledgeCore, float]]:
        """
        Находит лучшее ядро для запроса
        Возвращает (ядро, уверенность)
        """
        best_core = None
        best_confidence = 0.0

        for core in self.cores.values():
            if core.can_handle(query):
                confidence = core.get_confidence(query)

                # Усиленные фильтры
                if confidence < 0.3:
                    continue  # Слишком низкая уверенность

                # Проверка на ложные срабатывания
                q_lower = query.lower()
                core_keywords = ' '.join(core.capabilities + [core.description]).lower()

                # Считаем совпадения значимых слов
                query_words = set(re.findall(r'\w{3,}', q_lower))
                core_words = set(re.findall(r'\w{3,}', core_keywords))
                matches = len(query_words & core_words)

                if matches == 0 and confidence < 0.6:
                    continue  # Нет реальных совпадений

                # Выбираем лучшее
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
            caps = ', '.join(core.capabilities[:2])
            summary.append(f"- **{core.name}** ({caps})")
        return '\n'.join(summary)

    async def create_core_for_query(self, query: str, user_id: str, llm_caller) -> Dict[str, Any]:
        """
        УМНОЕ создание ядра для запроса
        Возвращает результат создания
        """
        print(f"🧠 Анализирую запрос для создания ядра: {query}")

        # Сначала проверяем, действительно ли нужно ядро
        should_create = await self._should_create_core(query, llm_caller)
        if not should_create:
            return {
                'should_create': False,
                'reason': 'Запрос может быть обработан существующими ядрами',
                'suggestion': 'Используйте существующие возможности'
            }

        # Генерируем код ядра
        core_code = await self._generate_core_code(query, user_id, llm_caller)
        if not core_code:
            return {
                'should_create': False,
                'reason': 'Не удалось сгенерировать код ядра',
                'error': 'LLM не вернула код'
            }

        # Валидируем код
        is_valid, reason = self.validator.validate_code_structure(core_code)
        if not is_valid:
            LearningLogger.log_core_rejection(f"auto_{user_id}", reason, query)
            return {
                'should_create': False,
                'reason': f'Невалидный код: {reason}',
                'error': reason
            }

        # Извлекаем информацию
        core_info = self.validator.extract_core_info(core_code)
        core_name = core_info['name']

        # Проверяем дубликаты
        if core_name in self.cores:
            return {
                'should_create': False,
                'reason': 'Ядро с таким именем уже существует',
                'suggestion': 'Используйте существующее ядро'
            }

        # Сохраняем файл
        filename = f"{core_name}.py"
        filepath = os.path.join(CORES_DIR, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# AUTO-GENERATED CORE - Self-learning AI\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n")
                f.write(f"# For query: {query[:100]}\n\n")
                f.write(core_code)

            print(f"💾 Сохранено ядро: {filename}")

            # Пробуем загрузить и протестировать
            self._load_dynamic_cores()  # Перезагружаем ядра

            if core_name in self.cores:
                # Тестируем на реальном запросе
                test_core = self.cores[core_name]
                response = test_core.execute(query)

                if response.success:
                    LearningLogger.log_new_core(core_name, query, user_id, True)

                    return {
                        'should_create': True,
                        'success': True,
                        'core_name': core_name,
                        'description': core_info['description'],
                        'confidence': response.confidence,
                        'test_passed': True
                    }
                else:
                    # Удаляем нерабочее ядро
                    del self.cores[core_name]
                    if os.path.exists(filepath):
                        os.remove(filepath)

                    LearningLogger.log_new_core(core_name, query, user_id, False)

                    return {
                        'should_create': False,
                        'success': False,
                        'reason': 'Ядро не прошло тест выполнения',
                        'error': response.data.get('error', 'Неизвестная ошибка')
                    }

        except Exception as e:
            # Удаляем файл при ошибке
            if os.path.exists(filepath):
                os.remove(filepath)

            return {
                'should_create': False,
                'success': False,
                'reason': 'Ошибка сохранения ядра',
                'error': str(e)
            }

    async def _should_create_core(self, query: str, llm_caller) -> bool:
        """Определяет, нужно ли создавать ядро для запроса"""
        prompt = f"""Анализируй запрос пользователя и определи, стоит ли создавать специализированное ядро.

ЗАПРОС: "{query}"

КРИТЕРИИ СОЗДАНИЯ ЯДРА:
1. Запрос требует СПЕЦИФИЧЕСКИХ знаний (не общих)
2. Это ТИПОВОЙ запрос, который будет повторяться
3. Для обработки нужны ОСОБЫЕ данные или логика
4. Существующие ядра не подходят

ПРИМЕРЫ, когда НУЖНО создавать ядро:
- "курс доллара к рублю" (специфичные данные)
- "расписание автобусов" (специфичная система)
- "мои заметки" (персональные данные)
- "конвертер валют" (специальная логика)

ПРИМЕРЫ, когда НЕ нужно создавать ядро:
- "привет" (общее общение)
- "как дела" (общее общение)
- "что такое ИИ" (общие знания)
- "расскажи анекдот" (общие знания)

Ответь ТОЛЬКО "ДА" или "НЕТ", без пояснений."""

        try:
            response = await llm_caller(prompt, temperature=0.1)
            response = response.strip().upper()
            return response == "ДА"
        except:
            # По умолчанию не создаем при ошибках
            return False

    async def _generate_core_code(self, query: str, user_id: str, llm_caller) -> Optional[str]:
        """Генерирует код ядра через LLM"""
        prompt = f"""Создай РАБОЧИЙ и ПРОСТОЙ класс ядра для обработки запроса.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"

ТРЕБОВАНИЯ К ЯДРУ:
1. Класс должен наследоваться от KnowledgeCore
2. Должны быть определены: name, description, capabilities, priority
3. Метод can_handle должен проверять конкретные ключевые слова
4. Метод execute должен возвращать CoreResponse
5. Будь КОНКРЕТНЫМ и ПРАКТИЧНЫМ
6. Избегай сложной логики

ШАБЛОН КЛАССА:

class NewCore(KnowledgeCore):
    name = "название_ядра"  # snake_case, уникальное
    description = "Краткое описание возможностей"
    capabilities = ["функция1", "функция2", "функция3"]
    priority = 5  # 1-10, где 1 - высший приоритет

    def can_handle(self, query: str) -> bool:
        # Проверяй КОНКРЕТНЫЕ ключевые слова
        q = query.lower()
        return any(word in q for word in ["ключевое_слово1", "ключевое_слово2"])

    def get_confidence(self, query: str) -> float:
        # Возвращай 0.0-1.0 в зависимости от уверенности
        if "точное_слово" in query.lower():
            return 0.9
        return 0.6 if self.can_handle(query) else 0.0

    def execute(self, query: str, context=None) -> CoreResponse:
        try:
            # РЕАЛИЗАЦИЯ
            # Если нужны данные из интернета:
            # if context and 'tools' in context:
            #     results = context['tools']['web_search'](query)

            # Возвращай ОБЪЕКТ CoreResponse
            return CoreResponse(
                success=True,
                data={{'result': 'данные'}},
                raw_result="Человекочитаемый ответ",
                confidence=0.8,
                source=self.name
            )
        except Exception as e:
            return CoreResponse(
                success=False,
                data={{'error': str(e)}},
                confidence=0.0,
                source=self.name
            )

ПРИМЕР для запроса "курс доллара":

class CurrencyCore(KnowledgeCore):
    name = "currency_core"
    description = "Курсы валют"
    capabilities = ["курс доллара", "курс евро", "конвертация"]
    priority = 5

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return any(word in q for word in ["курс", "доллар", "евро", "рубл", "конверт"])

    def execute(self, query: str, context=None) -> CoreResponse:
        try:
            if context and 'tools' in context:
                results = context['tools']['web_search'](f"{query} курс сегодня")
                if results:
                    info = "\\n".join([f"• {{r['title']}}" for r in results[:2]])
                    return CoreResponse(
                        success=True,
                        data={{'results': results}},
                        raw_result=f"Актуальные курсы валют:\\n{{info}}",
                        confidence=0.8,
                        source=self.name
                    )

            return CoreResponse(
                success=False,
                data={{'error': 'Нет данных'}},
                raw_result="Не удалось получить актуальные курсы валют",
                confidence=0.0,
                source=self.name
            )
        except Exception as e:
            return CoreResponse(
                success=False,
                data={{'error': str(e)}},
                confidence=0.0,
                source=self.name
            )

Создай ТОЛЬКО код класса, без дополнительных комментариев, начиная с "class"."""

        try:
            response = await llm_caller(prompt, temperature=0.3)

            # Очищаем ответ
            response = response.strip()

            # Удаляем markdown коды если есть
            if response.startswith('```python'):
                response = response[9:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]

            # Ищем начало класса
            class_start = response.find('class ')
            if class_start == -1:
                return None

            return response[class_start:].strip()

        except Exception as e:
            print(f"  ❌ Ошибка генерации кода: {e}")
            return None


# ==================== УЛУЧШЕННЫЙ САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ ====================
class MiniBrain:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.tools = ToolsManager()
        self.conversation_history = []

    def _web_search_tool(self, query: str, max_results: int = 4) -> List[Dict]:
        """Инструмент веб-поиска"""
        try:
            results = DDGS().text(query, max_results=max_results)
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
        """
        🧠 УЛУЧШЕННАЯ ОБРАБОТКА ЗАПРОСА
        """
        print(f"\n{'=' * 60}")
        print(f"🧠 Запрос от {self.user_id}: {query}")
        print(f"{'=' * 60}")

        # Добавляем в историю
        self.conversation_history.append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        })

        # Ограничиваем историю
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        # 1. Ищем существующее ядро
        core_result = self.tools.find_best_core(query)

        if core_result:
            core, confidence = core_result
            print(f"🎯 Найдено ядро: {core.name} (уверенность: {confidence:.2f})")

            # Используем ядро если уверенность высокая
            if confidence > 0.6:
                print(f"⚡ Использую ядро {core.name}...")

                # Выполняем запрос через ядро
                context = {
                    'user_id': self.user_id,
                    'tools': {
                        'web_search': self._web_search_tool,
                        'memory': self.memory
                    }
                }

                core_response = core.execute(query, context)

                # Логируем производительность
                LearningLogger.log_core_performance(
                    core.name,
                    core_response.success,
                    query
                )

                # Если ядро может дать прямой ответ - используем его
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

                # Иначе строим контекст для LLM
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

        # 2. Попытка создания нового ядра
        print("🤖 Проверяю, нужно ли создать новое ядро...")

        # Определяем, специфический ли запрос
        is_specific_query = self._is_specific_query(query)

        if is_specific_query:
            print("💡 Запрос специфический, пробую создать ядро...")

            creation_result = await self.tools.create_core_for_query(
                query, self.user_id, llm_caller
            )

            if creation_result.get('success'):
                print(f"🎉 Создано новое ядро: {creation_result['core_name']}")

                # Пробуем использовать новое ядро
                new_core = self.tools.cores.get(creation_result['core_name'])
                if new_core and new_core.can_handle(query):
                    context = {
                        'user_id': self.user_id,
                        'tools': {
                            'web_search': self._web_search_tool,
                            'memory': self.memory
                        }
                    }

                    core_response = new_core.execute(query, context)

                    if core_response.success:
                        llm_context = self._build_llm_context_with_core(
                            query, core_response, new_core.name
                        )

                        return {
                            'type': 'new_core_created',
                            'context': llm_context,
                            'source': new_core.name,
                            'core_name': creation_result['core_name'],
                            'description': creation_result.get('description'),
                            'need_llm': True
                        }

        # 3. Общий LLM контекст (фолбэк)
        print("💭 Использую общую обработку...")

        llm_context = self._build_general_context(query)

        return {
            'type': 'general_llm',
            'context': llm_context,
            'source': 'general',
            'need_llm': True
        }

    def _is_specific_query(self, query: str) -> bool:
        """Определяет, специфический ли запрос"""
        q = query.lower()

        # Общие запросы
        general_patterns = [
            r'привет', r'как дела', r'что ты умеешь', r'помощь',
            r'спасибо', r'пока', r'до свидания', r'кто ты',
            r'расскажи о себе', r'что ты можешь'
        ]

        for pattern in general_patterns:
            if re.search(pattern, q):
                return False

        # Запросы с числовыми данными, датами, специфичными терминами
        specific_indicators = [
            r'\d+',  # Числа
            r'[A-Z]{3}',  # Коды валют (USD, EUR)
            r'файл', r'документ', r'сохрани', r'прочитай',
            r'курс', r'погода', r'расписание', r'напомни',
            r'калькулятор', r'посчитай', r'вычисли',
            r'переведи', r'конвертируй', r'закажи', r'купи'
        ]

        specific_count = 0
        for indicator in specific_indicators:
            if re.search(indicator, q, re.IGNORECASE):
                specific_count += 1

        return specific_count >= 2 or len(q.split()) >= 5

    def _build_llm_context_with_core(self, query: str, core_response: CoreResponse, core_name: str) -> str:
        """Строит контекст для LLM с данными ядра"""
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        context = f"""# СИСТЕМА: ИИ АССИСТЕНТ С СПЕЦИАЛИЗИРОВАННЫМИ ЯДРАМИ
Текущее время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь: {self.user_id}
Ядро обработки: {core_name}

# ДАННЫЕ ОТ ЯДРА '{core_name.upper()}':
{core_response.raw_result}

# ИСТОРИЯ ДИАЛОГА (последние сообщения):
"""

        # История диалога
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

# ПРИМЕРЫ:
- Если ядро нашло файл: "Вот содержимое файла: [данные]"
- Если ядро не нашло файл: "Файл не найден. [рекомендация от ядра]"
- Если ядро дало курс валют: "Согласно данным, [курс]. Источник: [от ядра]"

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
Если вопрос относится к специализированной области (файлы, курсы, расчёты), предложи создать для него ядро."""

        return context


# ==================== ТЕЛЕГРАМ БОТ ====================
class TelegramBot:
    def __init__(self):
        self.user_brains: Dict[str, MiniBrain] = {}
        self._initialize_logs()

    def _initialize_logs(self):
        """Инициализирует файлы логов"""
        for log_file in [LEARNING_LOG, CORE_PERFORMANCE_LOG, REJECTED_CORES_LOG]:
            if not os.path.exists(log_file):
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)

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
        await update.message.reply_text(
            "🧠 *САМООБУЧАЮЩИЙСЯ ИИ-АССИСТЕНТ v4.0*\n\n"
            "✨ *Новые возможности:*\n"
            "• 🤖 Автоматическая валидация создаваемых ядер\n"
            "• ✅ Гарантированное использование ответов ядер\n"
            "• 🧪 Тестирование ядер перед сохранением\n"
            "• 📊 Мониторинг производительности ядер\n\n"
            "💡 *Как это работает:*\n"
            "1. Вы задаёте вопрос\n"
            "2. Бот ищет подходящее ядро\n"
            "3. Если ядра нет - создаёт РАБОЧЕЕ ядро\n"
            "4. Ответ ядра используется напрямую\n\n"
            "⚙️ *Команды:*\n"
            "/list_cores — список всех ядер\n"
            "/learning_log — журнал обучения\n"
            "/performance — статистика ядер\n"
            "/help — справка\n"
            "/clear — очистить историю диалога",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        await update.message.reply_text(
            "📖 *КАК ЭТО РАБОТАЕТ*\n\n"
            "*Самообучение:*\n"
            "Когда вы задаёте специфический вопрос, бот анализирует его и решает - нужно ли создавать новое ядро.\n\n"
            "*Пример самообучения:*\n"
            "Вы: _курс доллара_\n"
            "Бот: 🤖 Анализирую запрос...\n"
            "Бот: 🎯 Создаю ядро для курсов валют\n"
            "Бот: ✅ Создано ядро `currency_core`\n"
            "Бот: 💰 [ответ с курсами]\n\n"
            "*Автоматическая валидация:*\n"
            "Каждое новое ядро тестируется перед сохранением. Непроходимые тесты = ядро удаляется.\n\n"
            "*Команды:*\n"
            "/list_cores — посмотреть все ядра\n"
            "/learning_log — что бот выучил\n"
            "/performance — статистика работы ядер",
            parse_mode='Markdown'
        )

    async def list_cores(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /list_cores - список ядер"""
        brain = self.get_brain(str(update.effective_user.id))

        msg = "🔧 *ДОСТУПНЫЕ ЯДРА*\n\n"

        # Группируем по приоритету
        cores_by_priority = {}
        for core in brain.tools.cores.values():
            if core.priority not in cores_by_priority:
                cores_by_priority[core.priority] = []
            cores_by_priority[core.priority].append(core)

        # Сортируем по приоритету (от высшего к низшему)
        for priority in sorted(cores_by_priority.keys()):
            msg += f"*Приоритет {priority}:*\n"
            for core in cores_by_priority[priority]:
                caps = ', '.join(core.capabilities[:3])
                direct = "✅" if core.direct_answer_mode else "❌"
                msg += f"• `{core.name}` {direct}\n"
                msg += f"  _{core.description}_\n"
                msg += f"  📦 {caps}\n\n"

        msg += f"_Всего: {len(brain.tools.cores)} ядер_\n"
        msg += "✅ — прямой ответ, ❌ — нужна обработка LLM"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def learning_log_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /learning_log - журнал обучения"""
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
        """Команда /performance - статистика производительности"""
        try:
            if not os.path.exists(CORE_PERFORMANCE_LOG):
                await update.message.reply_text("📊 *Статистика пока не собрана*", parse_mode='Markdown')
                return

            with open(CORE_PERFORMANCE_LOG, 'r', encoding='utf-8') as f:
                perf_data = json.load(f)

            if not perf_data:
                await update.message.reply_text("📊 *Статистика пока не собрана*", parse_mode='Markdown')
                return

            msg = "📊 *СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ ЯДЕР*\n\n"

            # Сортируем по использованию
            sorted_cores = sorted(
                perf_data.items(),
                key=lambda x: x[1]['total_count'],
                reverse=True
            )

            for core_name, stats in sorted_cores[:15]:  # Топ 15
                total = stats['total_count']
                success = stats['success_count']
                rate = success / total if total > 0 else 0

                # Эмодзи для успешности
                if rate >= 0.8:
                    emoji = "🟢"
                elif rate >= 0.5:
                    emoji = "🟡"
                else:
                    emoji = "🔴"

                msg += f"{emoji} *{core_name}*\n"
                msg += f"  Успешность: {rate:.0%} ({success}/{total})\n"

                # Последний запрос
                if stats['queries']:
                    last = stats['queries'][-1]
                    last_time = datetime.fromisoformat(last['timestamp']).strftime('%H:%M')
                    status = "✅" if last['success'] else "❌"
                    msg += f"  Последний: {last_time} {status}\n"

                msg += "\n"

            msg += f"_Всего ядер в статистике: {len(perf_data)}_"

            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)[:100]}")

    async def clear_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /clear - очистить историю диалога"""
        user_id = str(update.effective_user.id)

        if user_id in self.user_brains:
            # Очищаем историю
            self.user_brains[user_id].conversation_history = []
            self.user_brains[user_id].memory.short_term = []

            await update.message.reply_text(
                "🧹 *История диалога очищена*\n\n"
                "Все предыдущие сообщения удалены из памяти. Начните новый диалог!",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("✅ История уже пуста!")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений пользователя"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        # Показываем "печатает..."
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        brain = self.get_brain(user_id)

        try:
            # Обрабатываем запрос
            start_time = time.time()
            brain_result = await brain.process(text, self.get_llm_response)
            processing_time = time.time() - start_time

            print(f"⏱️ Время обработки: {processing_time:.2f} сек")
            print(f"📊 Тип ответа: {brain_result['type']}")

            # Если есть прямой ответ от ядра
            if brain_result.get('need_llm') == False:
                response = brain_result['response']

                # Сохраняем в историю
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

            # Получаем ответ от LLM
            llm_response = await self.get_llm_response(
                brain_result['context'],
                temperature=0.5
            )

            # Проверяем, не игнорирует ли LLM данные ядра
            if brain_result.get('type') in ['llm_with_core_data', 'new_core_created']:
                source = brain_result.get('source', '')

                # Если LLM говорит "не знаю", но ядро дало данные
                if any(phrase in llm_response.lower() for phrase in ['не знаю', 'не могу', 'не понимаю']):
                    # Используем raw_result от ядра если он есть
                    if 'raw_result' in brain_result.get('context', ''):
                        # Извлекаем данные ядра из контекста
                        import re
                        core_data_match = re.search(r'ДАННЫЕ ОТ ЯДРА.*?\n(.*?)(?=\n#|$)',
                                                    brain_result['context'],
                                                    re.DOTALL)
                        if core_data_match:
                            core_data = core_data_match.group(1).strip()
                            if len(core_data) > 50:
                                llm_response = f"📊 *Данные от ядра '{source}':*\n\n{core_data}"

            # Ограничиваем длину
            if len(llm_response) > 4000:
                llm_response = llm_response[:4000] + "\n\n... (сообщение сокращено)"

            # Сохраняем в историю
            brain.conversation_history.append({
                'role': 'assistant',
                'content': llm_response,
                'source': brain_result.get('source', 'llm'),
                'timestamp': datetime.now().isoformat()
            })

            # Отправляем
            await update.message.reply_text(
                llm_response,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )

            # Если создано новое ядро - информируем
            if brain_result.get('type') == 'new_core_created':
                await update.message.reply_text(
                    f"🤖 *Создано новое ядро!*\n\n"
                    f"📦 Имя: `{brain_result['core_name']}`\n"
                    f"📝 Описание: {brain_result.get('description', 'без описания')}\n\n"
                    f"Теперь бот умеет обрабатывать подобные запросы!",
                    parse_mode='Markdown'
                )

        except Exception as e:
            error_msg = f"❌ *Ошибка:* {str(e)[:200]}"
            print(f"ERROR [{user_id}]: {e}")
            traceback.print_exc()

            # Сохраняем ошибку в историю
            brain.conversation_history.append({
                'role': 'assistant',
                'content': error_msg,
                'source': 'error',
                'timestamp': datetime.now().isoformat()
            })

            await update.message.reply_text(error_msg, parse_mode='Markdown')


# ==================== ЗАПУСК ====================
def main():
    print("\n" + "=" * 70)
    print("🚀 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v4.0")
    print("=" * 70)
    print(f"⏰ Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"📁 Директория ядер: {CORES_DIR}/")
    print(f"🧠 Директория памяти: {MEMORY_DIR}/")
    print(f"🔗 LM Studio: {LM_STUDIO_API_URL}")
    print("=" * 70)

    # Проверка LM Studio
    try:
        test_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '')
        test_resp = requests.get(test_url, timeout=10)

        if test_resp.status_code == 200:
            print(f"✅ LM Studio доступна")

            # Пробуем получить список моделей
            models_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '/v1/models')
            try:
                models_resp = requests.get(models_url, timeout=5)
                if models_resp.status_code == 200:
                    models = models_resp.json().get('data', [])
                    if models:
                        model_names = [m.get('id', 'unknown') for m in models[:2]]
                        print(f"📋 Загруженные модели: {', '.join(model_names)}")
            except:
                pass
        else:
            print(f"⚠️ LM Studio код: {test_resp.status_code}")
    except Exception as e:
        print(f"⚠️ LM Studio недоступна: {e}")
        print("🤖 Бот будет работать, но без LLM функциональности")

    print("=" * 70)
    print("\n🔄 Инициализация...\n")

    bot = TelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Команды
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("list_cores", bot.list_cores))
    application.add_handler(CommandHandler("learning_log", bot.learning_log_cmd))
    application.add_handler(CommandHandler("performance", bot.performance_cmd))
    application.add_handler(CommandHandler("clear", bot.clear_cmd))

    # Обработка сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот инициализирован!")
    print("=" * 70)
    print("💬 Напишите боту в Telegram")
    print("🤖 Бот будет САМ создавать РАБОЧИЕ ядра когда нужно")
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