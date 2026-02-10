#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v3.0
✅ Модель САМА создает себе ядра когда нужно
✅ Автоматическое расширение возможностей
✅ Обучение в процессе диалога
"""

import os
import json
import re
import ast
import asyncio
import requests

try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Literal
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
if not TELEGRAM_TOKEN:
    raise ValueError("❌ ОШИБКА: Не найден TELEGRAM_TOKEN в .env!")

LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
CORES_DIR = "dynamic_cores"
MEMORY_DIR = "brain_memory"
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
os.makedirs(CORES_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)


# ==================== ТИПЫ ОТВЕТОВ ЯДЕР ====================
class CoreResponse:
    """Стандартизированный ответ от ядра знаний"""

    def __init__(
            self,
            success: bool,
            data: Optional[Dict[str, Any]] = None,
            raw_result: Optional[str] = None,
            confidence: float = 1.0,
            source: str = "unknown"
    ):
        self.success = success
        self.data = data or {}
        self.raw_result = raw_result
        self.confidence = confidence
        self.source = source

    def to_context_string(self) -> str:
        """Преобразование в текст для контекста LLM"""
        if not self.success:
            return f"❌ Ошибка получения данных из источника '{self.source}'"

        if self.raw_result:
            return f"📊 ДАННЫЕ ОТ '{self.source.upper()}':\n{self.raw_result}"

        if self.data:
            formatted_data = json.dumps(self.data, ensure_ascii=False, indent=2)
            return f"📊 СТРУКТУРИРОВАННЫЕ ДАННЫЕ ОТ '{self.source.upper()}':\n{formatted_data}"

        return f"ℹ️ Источник '{self.source}' обработал запрос, но не вернул данных"


# ==================== БАЗОВЫЙ КЛАСС ЯДРА ====================
class KnowledgeCore(ABC):
    """Базовый класс для всех ядер знаний"""
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []
    priority: int = 5

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        pass

    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        pass

    def get_confidence(self, query: str) -> float:
        return 0.5 if self.can_handle(query) else 0.0


# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
    name = "datetime_core"
    description = "Точная информация о дате, времени и днях недели"
    capabilities = ["текущая дата", "время", "день недели", "расчёт дат"]
    priority = 1

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
                    f"Текущая дата и время:\n"
                    f"📅 Дата: {target.day} {months[target.month - 1]} {target.year} года\n"
                    f"📆 День недели: {weekdays[target.weekday()]}\n"
                    f"🕐 Время: {target.strftime('%H:%M:%S')}"
                )
            else:
                offset_text = f"через {abs(days_offset)} дн." if days_offset > 0 else f"{abs(days_offset)} дн. назад"
                description = (
                    f"Дата {offset_text}:\n"
                    f"📅 {target.day} {months[target.month - 1]} {target.year} года\n"
                    f"📆 {weekdays[target.weekday()]}"
                )

            return CoreResponse(
                success=True,
                data=data,
                raw_result=description,
                confidence=0.95,
                source=self.name
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

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        has_math = bool(re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', q))
        has_words = any(word in q for word in ['сколько будет', 'посчитай', 'вычисли', 'равно'])
        return has_math or (has_words and any(char.isdigit() for char in q))

    def get_confidence(self, query: str) -> float:
        if re.search(r'\d+\s*[\+\-\*\/x×÷]\s*\d+', query):
            return 0.9
        return 0.6 if self.can_handle(query) else 0.0

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            expr = re.sub(r'[^\d\+\-\*\/x×÷\.\(\)\s]', '', query.lower())
            expr = expr.replace('x', '*').replace('×', '*').replace('÷', '/').replace(' ', '')

            if not expr or any(danger in expr for danger in ['import', 'exec', 'eval', '__']):
                raise ValueError("Недопустимое выражение")

            result = eval(expr, {"__builtins__": {}}, {})

            data = {
                'expression': expr,
                'result': result,
                'formatted_expression': expr.replace('*', '×').replace('/', '÷')
            }

            description = f"Результат вычисления:\n🧮 {data['formatted_expression']} = {result}"

            return CoreResponse(
                success=True,
                data=data,
                raw_result=description,
                confidence=0.9,
                source=self.name
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

    def can_handle(self, query: str) -> bool:
        q = query.lower()

        exclude_keywords = [
            'какой сегодня день', 'какое число', 'который час', 'сколько времени',
            'сколько будет', '+', '-', '*', '/', 'умножить', 'разделить'
        ]
        if any(kw in q for kw in exclude_keywords):
            return False

        search_keywords = [
            'новост', 'курс', 'погод', 'прогноз', 'событи', 'произошло',
            'последни', 'актуальн', 'сейчас', 'сегодняшн', 'текущ',
            'кто выиграл', 'результат', 'итоги', 'цена', 'стоимость'
        ]
        return any(kw in q for kw in search_keywords)

    def get_confidence(self, query: str) -> float:
        q = query.lower()
        high_priority = ['курс', 'новости сегодня', 'погода', 'последние новости']
        if any(kw in q for kw in high_priority):
            return 0.85
        return 0.6 if self.can_handle(query) else 0.0

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> CoreResponse:
        try:
            print(f"  🌐 Веб-поиск: {query[:60]}...")

            results = self.ddgs.text(query, max_results=5)

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

            results_text = "РЕЗУЛЬТАТЫ ПОИСКА В ИНТЕРНЕТЕ:\n\n"
            for r in search_results:
                results_text += f"{r['position']}. {r['title']}\n"
                results_text += f"   {r['snippet']}\n"
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
    def log_new_core(core_name: str, query: str, user_id: str):
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
                'user_id': user_id
            })

            with open(LEARNING_LOG, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            print(f"📝 Записал в журнал: создано ядро '{core_name}'")

        except Exception as e:
            print(f"⚠️ Ошибка записи в журнал: {e}")

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


# ==================== САМООБУЧАЮЩИЙСЯ МЕНЕДЖЕР ЯДЕР ====================
class ToolsManager:
    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self._load_builtin_cores()
        self._load_dynamic_cores()

    def _load_builtin_cores(self):
        builtin_cores = [
            DateTimeCore(),
            CalculatorCore(),
            WebSearchCore()
        ]

        for core in builtin_cores:
            self.cores[core.name] = core

        print(f"✅ Загружено встроенных ядер: {len(builtin_cores)}")

    def _load_dynamic_cores(self):
        loaded_count = 0

        for fname in os.listdir(CORES_DIR):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue

            path = os.path.join(CORES_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()

                match = re.search(r'class\s+(\w+)\s*\(\s*KnowledgeCore\s*\)', code)
                if not match:
                    continue

                class_name = match.group(1)

                namespace = {
                    'KnowledgeCore': KnowledgeCore,
                    'CoreResponse': CoreResponse,
                    '__builtins__': __builtins__,
                    'requests': requests,
                    're': re,
                    'json': json,
                    'datetime': datetime,
                    'timedelta': timedelta,
                    'Dict': Dict,
                    'List': List,
                    'Any': Any,
                    'Optional': Optional,
                    'os': os,
                    'DDGS': DDGS
                }

                exec(code, namespace)

                if class_name in namespace:
                    core_instance = namespace[class_name]()

                    if self._validate_core(core_instance):
                        self.cores[core_instance.name] = core_instance
                        loaded_count += 1
                        print(f"✅ Загружено динамическое ядро: {core_instance.name}")

            except Exception as e:
                print(f"⚠️ Ошибка загрузки {fname}: {e}")

        if loaded_count > 0:
            print(f"✅ Загружено динамических ядер: {loaded_count}")

    def _validate_core(self, core: KnowledgeCore) -> bool:
        try:
            required_attrs = ['name', 'description', 'capabilities']
            for attr in required_attrs:
                if not hasattr(core, attr):
                    return False

            if not callable(getattr(core, 'can_handle', None)):
                return False

            if not callable(getattr(core, 'execute', None)):
                return False

            test_response = core.execute("тест", {})
            if not isinstance(test_response, CoreResponse):
                return False

            return True

        except Exception as e:
            print(f"  ❌ Ошибка валидации: {e}")
            return False

    def find_best_core(self, query: str) -> Optional[KnowledgeCore]:
        candidates = []

        for core in self.cores.values():
            if core.can_handle(query):
                confidence = core.get_confidence(query)
                candidates.append((confidence, core.priority, core))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_core = candidates[0][2]

        return best_core

    async def auto_create_core(self, query: str, user_id: str, llm_caller) -> Dict[str, Any]:
        """
        🧠 АВТОМАТИЧЕСКОЕ СОЗДАНИЕ ЯДРА
        Модель сама решает, нужно ли создать ядро для данного запроса
        """

        # Шаг 1: Спрашиваем у модели, нужно ли создавать ядро
        decision_prompt = f"""Проанализируй запрос пользователя и определи, нужно ли создать специализированное ядро для его обработки.

ТЕКУЩИЕ ЯДРА:
{self._get_cores_summary()}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

КРИТЕРИИ ДЛЯ СОЗДАНИЯ ЯДРА:
- Запрос требует специфических знаний или расчётов
- Это не разовый вопрос, а тип запросов который может повториться
- Существующие ядра не подходят для обработки
- Для ответа нужны специальные данные (API, формулы, базы данных)

ОТВЕТЬ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "should_create": true/false,
  "reasoning": "краткое обоснование",
  "core_description": "описание ядра (если should_create=true)"
}}

ПРИМЕРЫ:

Запрос: "какая погода в Москве"
{{
  "should_create": false,
  "reasoning": "веб-поиск может обработать",
  "core_description": null
}}

Запрос: "переведи 100 USD в EUR"
{{
  "should_create": true,
  "reasoning": "специфическая задача конвертации валют, будет повторяться",
  "core_description": "Ядро для конвертации валют через веб-поиск актуальных курсов"
}}

Верни ТОЛЬКО JSON, без дополнительного текста."""

        try:
            decision_resp = await llm_caller(decision_prompt, temperature=0.3)

            # Очищаем от возможного мусора
            decision_text = decision_resp.strip()
            decision_text = re.sub(r'^```json\s*|^```\s*', '', decision_text, flags=re.MULTILINE)
            decision_text = re.sub(r'```\s*$', '', decision_text)

            decision = json.loads(decision_text)

            if not decision.get('should_create', False):
                print(f"🤔 Модель решила НЕ создавать ядро: {decision.get('reasoning', 'нет обоснования')}")
                return {'should_create': False, 'reasoning': decision.get('reasoning')}

            print(f"💡 Модель решила создать ядро: {decision.get('reasoning')}")
            print(f"📋 Описание: {decision.get('core_description')}")

            # Шаг 2: Создаём ядро
            result = await self._generate_core_code(decision['core_description'], user_id, llm_caller)

            if result['success']:
                # Перезагружаем ядра
                self._load_dynamic_cores()

                # Логируем обучение
                LearningLogger.log_new_core(result['core_name'], query, user_id)

                return {
                    'should_create': True,
                    'success': True,
                    'core_name': result['core_name'],
                    'reasoning': decision.get('reasoning'),
                    'description': decision.get('core_description')
                }
            else:
                return {
                    'should_create': True,
                    'success': False,
                    'error': result.get('error')
                }

        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON решения: {e}")
            return {'should_create': False, 'error': 'Ошибка парсинга решения'}
        except Exception as e:
            print(f"⚠️ Ошибка автосоздания ядра: {e}")
            return {'should_create': False, 'error': str(e)}

    async def _generate_core_code(self, description: str, user_id: str, llm_caller) -> Dict[str, Any]:
        """Генерирует код ядра"""

        prompt = f"""Создай РАБОЧИЙ Python-класс ядра знаний для ИИ-ассистента.

ОПИСАНИЕ ЯДРА:
{description}

ТРЕБОВАНИЯ:
1. Наследуется от KnowledgeCore
2. Атрибуты:
   - name = "название_core" (уникальное, snake_case)
   - description = "краткое описание"
   - capabilities = ["функция1", "функция2"]
   - priority = 5 (1-10, где 1 - самый высокий)

3. Методы:
   def can_handle(self, query: str) -> bool:
       # True если ядро может обработать запрос

   def get_confidence(self, query: str) -> float:
       # 0.0-1.0 уверенность в обработке

   def execute(self, query: str, context=None) -> CoreResponse:
       # Обработка и возврат CoreResponse

4. Для веб-поиска используй: context['tools']['web_search'](query)

5. CoreResponse формат:
   return CoreResponse(
       success=True/False,
       data={{'ключ': 'значение'}},
       raw_result="текстовое описание",
       confidence=0.8,
       source=self.name
   )

ПРИМЕР:

class CurrencyCore(KnowledgeCore):
    name = "currency_converter_core"
    description = "Конвертация валют"
    capabilities = ["конвертация", "обмен валют", "курсы"]
    priority = 5

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in ['конверт', 'переведи', 'usd', 'eur', 'в рубл'])

    def get_confidence(self, query: str) -> float:
        if re.search(r'\\d+\\s*(usd|eur|rub)', query.lower()):
            return 0.9
        return 0.6 if self.can_handle(query) else 0.0

    def execute(self, query: str, context=None) -> CoreResponse:
        try:
            if context and 'tools' in context:
                results = context['tools']['web_search'](query + " курс")

                if results:
                    info = "\\n".join([f"{{r['title']}}\\n{{r['snippet']}}" for r in results[:2]])

                    return CoreResponse(
                        success=True,
                        data={{'results': results}},
                        raw_result=f"Информация о курсах:\\n{{info}}",
                        confidence=0.85,
                        source=self.name
                    )

            return CoreResponse(
                success=False,
                data={{'error': 'Не удалось получить данные'}},
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

ВЕРНИ ТОЛЬКО КОД КЛАССА, начиная с class.
БЕЗ аннотаций типов вида -> Dict, : Dict[str, Any].
"""

        try:
            code = await llm_caller(prompt, temperature=0.2)

            # Очищаем
            code = code.strip()
            code = re.sub(r'^```python\s*|^```\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'```\s*$', '', code)

            # Проверяем синтаксис
            try:
                ast.parse(code)
            except SyntaxError as e:
                return {'success': False, 'error': f"Синтаксическая ошибка: {e}"}

            # Извлекаем имя
            core_name_match = re.search(r'name\s*=\s*["\'](\w+)["\']', code)
            if not core_name_match:
                core_name = f"auto_core_{user_id}_{datetime.now().strftime('%H%M%S')}"
            else:
                core_name = core_name_match.group(1)

            filepath = os.path.join(CORES_DIR, f"{core_name}.py")

            # Сохраняем
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# AUTO-GENERATED CORE - Self-learning AI\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n\n")
                f.write("from __main__ import KnowledgeCore, CoreResponse, Optional, Dict, Any, List\n")
                f.write("from __main__ import requests, re, json, datetime, os, DDGS\n\n")
                f.write(code)

            return {
                'success': True,
                'filename': f"{core_name}.py",
                'core_name': core_name,
                'filepath': filepath
            }

        except Exception as e:
            return {'success': False, 'error': f"Ошибка генерации: {str(e)}"}

    def _get_cores_summary(self) -> str:
        """Краткое описание существующих ядер"""
        summary = []
        for core in self.cores.values():
            caps = ', '.join(core.capabilities[:3])
            summary.append(f"- {core.name}: {core.description} ({caps})")
        return '\n'.join(summary)


# ==================== САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ ====================
class MiniBrain:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.tools = ToolsManager()

    def _web_search_tool(self, query: str, max_results: int = 4) -> List[Dict]:
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
        🧠 ОБРАБОТКА С АВТОМАТИЧЕСКИМ ОБУЧЕНИЕМ
        """
        print(f"\n{'=' * 70}")
        print(f"🧠 Обработка: {query[:60]}...")
        print(f"{'=' * 70}")

        # 1. Ищем подходящее ядро
        best_core = self.tools.find_best_core(query)

        if best_core:
            print(f"⚡ Найдено ядро: {best_core.name}")

            core_response = best_core.execute(query, context={
                'user_id': self.user_id,
                'tools': {'web_search': self._web_search_tool}
            })

            if core_response.success:
                print(f"✅ Ядро успешно обработало запрос (confidence: {core_response.confidence:.2f})")

                context = self._build_llm_context(
                    query=query,
                    core_response=core_response,
                    core_name=best_core.name
                )

                return {
                    'type': 'llm_with_core_data',
                    'context': context,
                    'source': best_core.name,
                    'used_existing_core': True
                }

        # 2. Ядро не найдено - АВТОМАТИЧЕСКИ СОЗДАЁМ НОВОЕ
        print("🤖 Ядро не найдено. Модель анализирует, нужно ли создать новое...")

        auto_create_result = await self.tools.auto_create_core(query, self.user_id, llm_caller)

        if auto_create_result.get('should_create') and auto_create_result.get('success'):
            print(f"🎉 Создано новое ядро: {auto_create_result['core_name']}")

            # Пробуем использовать новое ядро
            new_core = self.tools.cores.get(auto_create_result['core_name'])
            if new_core and new_core.can_handle(query):
                print(f"⚡ Используем новое ядро для обработки запроса")

                core_response = new_core.execute(query, context={
                    'user_id': self.user_id,
                    'tools': {'web_search': self._web_search_tool}
                })

                if core_response.success:
                    context = self._build_llm_context(query, core_response, new_core.name)

                    return {
                        'type': 'llm_with_new_core',
                        'context': context,
                        'source': new_core.name,
                        'created_core': True,
                        'core_name': auto_create_result['core_name'],
                        'reasoning': auto_create_result.get('reasoning')
                    }

        # 3. Не создали ядро - используем общий контекст
        print("💭 Использую общий контекст")
        context = self._build_general_context(query)

        return {
            'type': 'llm_general',
            'context': context,
            'source': 'general',
            'created_core': False
        }

    def _build_llm_context(self, query: str, core_response: CoreResponse, core_name: str) -> str:
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        context = f"""СИСТЕМНАЯ ИНФОРМАЦИЯ:
Текущее время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь ID: {self.user_id}

"""

        context += core_response.to_context_string()
        context += "\n\n"

        short_term = self.memory.get_short_term(last_n=6)
        if short_term:
            context += "ИСТОРИЯ ДИАЛОГА:\n"
            for msg in short_term:
                emoji = '👤' if msg['role'] == 'user' else '🤖'
                context += f"{emoji} {msg['content'][:100]}\n"
            context += "\n"

        long_term = self.memory.search_long_term(query, limit=3)
        if long_term:
            context += "ФАКТЫ О ПОЛЬЗОВАТЕЛЕ:\n"
            for fact in long_term:
                context += f"- {fact['content']}\n"
            context += "\n"

        context += f"""ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ИНСТРУКЦИИ:
1. Используй данные от ядра '{core_name}'
2. Дай точный, естественный ответ
3. Учитывай контекст диалога
4. Будь конкретным и полезным
5. НЕ выдумывай - опирайся на данные
"""

        return context

    def _build_general_context(self, query: str) -> str:
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        context = f"""СИСТЕМНАЯ ИНФОРМАЦИЯ:
Текущее время: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})
Пользователь ID: {self.user_id}

ДОСТУПНЫЕ ЯДРА:
{self.tools._get_cores_summary()}

"""

        short_term = self.memory.get_short_term(last_n=6)
        if short_term:
            context += "ИСТОРИЯ ДИАЛОГА:\n"
            for msg in short_term:
                emoji = '👤' if msg['role'] == 'user' else '🤖'
                context += f"{emoji} {msg['content'][:100]}\n"
            context += "\n"

        context += f"""ВОПРОС:
{query}

ИНСТРУКЦИИ:
1. Дай полезный ответ на основе своих знаний
2. Используй естественный стиль
3. Будь конкретным
"""

        return context


# ==================== ТЕЛЕГРАМ БОТ ====================
class TelegramBot:
    def __init__(self):
        self.user_brains: Dict[str, MiniBrain] = {}

    def get_brain(self, user_id: str) -> MiniBrain:
        if user_id not in self.user_brains:
            self.user_brains[user_id] = MiniBrain(user_id)
        return self.user_brains[user_id]

    async def get_llm_response(self, context: str, temperature: float = 0.5) -> str:
        try:
            resp = requests.post(
                LM_STUDIO_API_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {LM_STUDIO_API_KEY}'
                },
                json={
                    'messages': [{'role': 'system', 'content': context}],
                    'temperature': temperature,
                    'max_tokens': 1500
                },
                timeout=90
            )

            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
            else:
                return f"❌ Ошибка модели (HTTP {resp.status_code})"

        except Exception as e:
            return f"❌ Ошибка LLM: {str(e)}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 *САМООБУЧАЮЩИЙСЯ ИИ-АССИСТЕНТ v3.0*\n\n"
            "✨ *Что нового:*\n"
            "• 🤖 Модель САМА создаёт себе ядра когда нужно\n"
            "• 🧩 Автоматическое расширение возможностей\n"
            "• 📚 Обучение в процессе диалога\n\n"
            "💡 *Как это работает:*\n"
            "1. Ты задаёшь вопрос\n"
            "2. Если подходящего ядра нет - AI создаёт новое\n"
            "3. Новое ядро сохраняется и используется в будущем\n\n"
            "⚙️ *Команды:*\n"
            "/list_cores — список ядер\n"
            "/learning_log — журнал обучения\n"
            "/help — справка",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📖 *КАК ЭТО РАБОТАЕТ*\n\n"
            "*Самообучение:*\n"
            "Когда ты задаёшь вопрос, который существующие ядра не могут обработать, "
            "модель сама анализирует запрос и решает — нужно ли создать новое специализированное ядро.\n\n"
            "*Пример:*\n"
            "Ты: _переведи 100 USD в EUR_\n"
            "Бот: 🤖 Создаю ядро для конвертации валют...\n"
            "Бот: ✅ Создано ядро `currency_converter_core`\n"
            "Бот: [даёт ответ]\n\n"
            "В следующий раз это ядро уже будет доступно!\n\n"
            "*Команды:*\n"
            "/list_cores — посмотреть все ядра\n"
            "/learning_log — что бот выучил",
            parse_mode='Markdown'
        )

    async def list_cores(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        brain = self.get_brain(str(update.effective_user.id))

        msg = "🔧 *ДОСТУПНЫЕ ЯДРА*\n\n"

        cores_by_priority = {}
        for core in brain.tools.cores.values():
            if core.priority not in cores_by_priority:
                cores_by_priority[core.priority] = []
            cores_by_priority[core.priority].append(core)

        for priority in sorted(cores_by_priority.keys()):
            for core in cores_by_priority[priority]:
                caps = ', '.join(core.capabilities[:4])
                msg += f"*{core.name}* (приоритет: {core.priority})\n"
                msg += f"├ {core.description}\n"
                msg += f"└ {caps}\n\n"

        msg += f"_Всего: {len(brain.tools.cores)} ядер_"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def learning_log_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает журнал самообучения"""
        log = LearningLogger.get_recent_learning(limit=10)

        if not log:
            await update.message.reply_text("📝 Журнал обучения пуст")
            return

        msg = "📚 *ЖУРНАЛ САМООБУЧЕНИЯ*\n\n"

        for entry in reversed(log):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%d.%m %H:%M')
            core_name = entry.get('core_name', 'unknown')
            trigger = entry.get('trigger_query', '')[:50]

            msg += f"⏰ {timestamp}\n"
            msg += f"🆕 Создано: `{core_name}`\n"
            msg += f"📝 Запрос: _{trigger}_\n\n"

        msg += f"_Всего записей: {len(log)}_"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        brain = self.get_brain(user_id)

        try:
            # Обрабатываем с автообучением
            brain_result = await brain.process(text, self.get_llm_response)

            # Уведомляем если создано новое ядро
            if brain_result.get('created_core'):
                await update.message.reply_text(
                    f"🤖 *Создал новое ядро!*\n\n"
                    f"📦 Ядро: `{brain_result['core_name']}`\n"
                    f"💡 Причина: {brain_result.get('reasoning', 'расширение возможностей')}\n\n"
                    f"Теперь обрабатываю запрос...",
                    parse_mode='Markdown'
                )

            # Получаем ответ
            response = await self.get_llm_response(brain_result['context'])

            # Сохраняем в память
            brain.memory.add_short_term({'role': 'user', 'content': text})
            brain.memory.add_short_term({'role': 'assistant', 'content': response})

            # Отправляем
            max_length = 4000
            if len(response) > max_length:
                parts = [response[i:i + max_length] for i in range(0, len(response), max_length)]
                for part in parts:
                    await update.message.reply_text(part, parse_mode='Markdown', disable_web_page_preview=True)
            else:
                await update.message.reply_text(response, parse_mode='Markdown', disable_web_page_preview=True)

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
            print(f"ERROR [{user_id}]: {e}")
            import traceback
            traceback.print_exc()


# ==================== ЗАПУСК ====================
def main():
    print("\n" + "=" * 70)
    print("🚀 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v3.0")
    print("=" * 70)
    print(f"⏰ Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"📁 Ядра: {CORES_DIR}/")
    print(f"🧠 Память: {MEMORY_DIR}/")
    print(f"📝 Журнал: {LEARNING_LOG}")
    print(f"🔗 LM Studio: {LM_STUDIO_API_URL}")
    print("=" * 70)

    try:
        test_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '/v1/models')
        test_resp = requests.get(test_url, timeout=5)

        if test_resp.status_code == 200:
            models = [m.get('id', 'unknown') for m in test_resp.json().get('data', [])[:3]]
            print(f"✅ LM Studio доступна")
            print(f"📋 Модели: {', '.join(models)}")
        else:
            print(f"⚠️ LM Studio код: {test_resp.status_code}")
    except Exception as e:
        print(f"⚠️ LM Studio недоступна: {e}")

    print("=" * 70)
    print("\n🔄 Инициализация...\n")

    bot = TelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("list_cores", bot.list_cores))
    application.add_handler(CommandHandler("learning_log", bot.learning_log_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот запущен!")
    print("=" * 70)
    print("💬 Напиши боту в Telegram")
    print("🤖 Бот будет сам создавать себе ядра когда понадобится")
    print("=" * 70)
    print("\nCtrl+C для остановки\n")

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Остановлен")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()