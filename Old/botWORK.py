#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 МИНИ-МОЗГ БОТ — ИСПРАВЛЕННАЯ ВЕРСИЯ (работает без ошибок)
✅ Убран параметр timeout из DDGS
✅ Добавлены импорты typing для ядер
✅ Совместимость с новой версией duckduckgo_search/ddgs
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
    from ddgs import DDGS  # Поддержка нового имени пакета
from datetime import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
if not TELEGRAM_TOKEN:
    raise ValueError(
        "❌ ОШИБКА: Не найден TELEGRAM_TOKEN в .env!\nСоздайте файл .env со строкой: TELEGRAM_TOKEN=ваш_токен")

LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
CORES_DIR = "dynamic_cores"
MEMORY_DIR = "brain_memory"
os.makedirs(CORES_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)


# ==================== БАЗОВЫЙ КЛАСС ЯДРА ====================
class KnowledgeCore(ABC):
    """Базовый класс для всех ядер знаний"""
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        pass

    @abstractmethod
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        pass


# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
    name = "datetime_core"
    description = "Точная информация о дате, времени и днях недели"
    capabilities = ["текущая дата", "время", "день недели", "расчёт дат"]

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in [
            'какой сегодня день', 'какое число', 'день недели', 'сколько времени',
            'который час', 'текущая дата', 'сегодня', 'завтра', 'послезавтра', 'вчера'
        ])

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
        months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля',
                  'августа', 'сентября', 'октября', 'ноября', 'декабря']

        days_offset = 0
        if 'завтра' in query.lower():
            days_offset = 1
        elif 'послезавтра' in query.lower():
            days_offset = 2
        elif 'вчера' in query.lower():
            days_offset = -1
        else:
            match = re.search(r'через\s+(\d+)\s*(дн[еяй]|день|дней)', query.lower())
            if match:
                days_offset = int(match.group(1))

        target = datetime.now()
        if days_offset != 0:
            from datetime import timedelta
            target += timedelta(days=days_offset)

        if days_offset == 0:
            result = (f"📅 Сегодня: {target.day} {months[target.month - 1]} {target.year} г., "
                      f"{weekdays[target.weekday()]}\n⏰ Время: {target.hour:02d}:{target.minute:02d}")
        else:
            result = (f"📅 Дата через {abs(days_offset)} дн.: {target.day} {months[target.month - 1]} "
                      f"{target.year} г., {weekdays[target.weekday()]}")

        return {
            'success': True,
            'result': result,
            'data': {'date': target.strftime('%Y-%m-%d'), 'weekday': weekdays[target.weekday()]},
            'requires_llm': False
        }


class CalculatorCore(KnowledgeCore):
    name = "calculator_core"
    description = "Математические вычисления"
    capabilities = ["сложение", "вычитание", "умножение", "деление", "проценты"]

    def can_handle(self, query: str) -> bool:
        return bool(
            re.search(r'\d+\s*[\+\-\*\/x×]\s*\d+|сколько будет \d+ (плюс|минус|умножить|разделить)', query.lower()))

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            expr = re.sub(r'[^\d\+\-\*\/x×\.\(\)\s]', '', query.lower())
            expr = expr.replace('x', '*').replace('×', '*').replace(' ', '')
            if not expr or any(c in expr for c in ['import', 'exec', 'eval', '__', 'os', 'sys', 'open']):
                raise ValueError("Недопустимое выражение")
            result = eval(expr, {"__builtins__": {}}, {})
            return {
                'success': True,
                'result': f"🧮 {expr.replace('*', '×')} = {result}",
                'data': {'expression': expr, 'result': result},
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка вычисления: {str(e)}",
                'data': None,
                'requires_llm': True
            }


class WebSearchCore(KnowledgeCore):
    """РЕАЛЬНОЕ ядро с доступом к интернету через DuckDuckGo (БЕЗ параметра timeout)"""
    name = "web_search_core"
    description = "Поиск актуальной информации в интернете: новости, курсы, погода, события"
    capabilities = ["поиск в интернете", "актуальные новости", "курсы валют", "погода", "события сегодня"]

    def __init__(self):
        self.ddgs = DDGS()

    def needs_search(self, query: str) -> bool:
        """Определяет, требует ли запрос актуальных данных из интернета"""
        q = query.lower()
        if any(kw in q for kw in [
            'новост', 'курс', 'погод', 'прогноз погод', 'событи', 'произошло сегодня',
            'последни', 'актуальн', 'сейчас', 'сегодняшн', 'какие фильмы в прокате',
            'кто выиграл', 'результат матча', 'итоги выборов', 'цена биткоина'
        ]):
            if any(kw in q for kw in [
                'какой сегодня день', 'какое число', 'который час', 'сколько будет',
                'плюс', 'минус', 'умножить', 'разделить'
            ]):
                return False
            return True
        return False

    def can_handle(self, query: str) -> bool:
        return self.needs_search(query)

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            print(f"  🌐 Выполняю поиск: {query}")
            # УБРАН ПАРАМЕТР timeout — он не поддерживается в новых версиях
            results = self.ddgs.text(query, max_results=5)

            if not results:
                return {
                    'success': False,
                    'result': None,
                    'data': {'error': 'Не найдено результатов в интернете'},
                    'requires_llm': True
                }

            search_results = []
            for r in results[:3]:
                search_results.append({
                    'title': r.get('title', '')[:80],
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')[:200]
                })

            return {
                'success': True,
                'result': None,
                'data': {
                    'query': query,
                    'results': search_results,
                    'source': 'duckduckgo'
                },
                'requires_llm': True
            }

        except Exception as e:
            print(f"  ⚠️ Ошибка поиска: {e}")
            return {
                'success': False,
                'result': f"❌ Не удалось выполнить поиск: {str(e)}",
                'data': None,
                'requires_llm': True
            }


# ==================== СИСТЕМА ПАМЯТИ ====================
class MemoryManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.short_term: List[Dict] = []
        self.max_short_term = 15
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
        except:
            pass

    def add_short_term(self, msg: Dict):
        self.short_term.append(msg)
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)

    def get_short_term(self) -> List[Dict]:
        return self.short_term.copy()

    async def save_long_term(self, fact: str, fact_type: str = "general"):
        self.long_term.append({
            'content': fact,
            'type': fact_type,
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id
        })
        self._save_long_term()
        print(f"🧠 [{self.user_id}] Запомнил: {fact[:40]}...")

    async def search_long_term(self, query: str, limit: int = 3) -> List[Dict]:
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


# ==================== МЕНЕДЖЕР ЯДЕР (С ИСПРАВЛЕНИЕМ Dict) ====================
class ToolsManager:
    def __init__(self):
        self.cores: Dict[str, KnowledgeCore] = {}
        self._load_builtin_cores()
        self._load_dynamic_cores()

    def _load_builtin_cores(self):
        self.cores[DateTimeCore.name] = DateTimeCore()
        self.cores[CalculatorCore.name] = CalculatorCore()
        self.cores[WebSearchCore.name] = WebSearchCore()
        print(f"✅ Загружено встроенных ядер: {len(self.cores)}")

    def _load_dynamic_cores(self):
        for fname in os.listdir(CORES_DIR):
            if fname.endswith('.py') and not fname.startswith('__'):
                path = os.path.join(CORES_DIR, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        code = f.read()

                    match = re.search(r'class\s+(\w+)\s*\(\s*KnowledgeCore\s*\)', code)
                    if not match:
                        continue

                    class_name = match.group(1)
                    # ДОБАВЛЕНЫ ИМПОРТЫ typing ДЛЯ РАБОТЫ С АННОТАЦИЯМИ ТИПОВ
                    namespace = {
                        'KnowledgeCore': KnowledgeCore,
                        '__builtins__': __builtins__,
                        'requests': requests,
                        're': re,
                        'json': json,
                        'datetime': datetime,
                        'Dict': Dict,
                        'List': List,
                        'Any': Any,
                        'Optional': Optional,
                        'os': os
                    }
                    exec(code, namespace)

                    if class_name in namespace:
                        core = namespace[class_name]()
                        self.cores[core.name] = core
                        print(f"✅ Загружено ядро: {core.name} ({fname})")
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки {fname}: {e}")

    async def create_core_from_description(self, description: str, user_id: str) -> Dict[str, Any]:
        prompt = f"""
Ты — эксперт по созданию РАБОЧИХ плагинов для ИИ. Создай ПОЛНЫЙ рабочий Python-код для ядра знаний.

КРИТИЧЕСКИ ВАЖНО — НЕ ДЕЛАЙ ЗАГЛУШЕК:
❌ ЗАПРЕЩЕНО возвращать статические ответы вроде "В Москве +22°C"
✅ ОБЯЗАТЕЛЬНО используй реальные данные через:
   - Веб-поиск: вызови context['tools']['web_search'](query)
   - Публичные API без ключа (например, для курсов)

Требования к коду:
1. Класс наследуется от KnowledgeCore
2. Атрибуты: name (snake_case), description, capabilities (список строк)
3. can_handle(query) -> bool: точное определение запросов
4. execute(query, context) -> dict с ключами: success, result, data, requires_llm
5. НЕ ИСПОЛЬЗУЙ аннотации типов (например, -> Dict) — это вызывает ошибки!
6. Для доступа к интернету используй: результаты = context['tools']['web_search'](query)

Пример РАБОЧЕГО ядра БЕЗ аннотаций типов:
class CurrencyCore(KnowledgeCore):
    name = "currency_core"
    description = "Актуальные курсы валют через веб-поиск"
    capabilities = ["курс доллара", "курс евро", "курс валют"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['курс', 'доллар', 'евро', 'usd', 'eur', 'валют'])

    def execute(self, query, context=None):
        try:
            # Используем встроенный инструмент поиска
            if context and 'tools' in context and 'web_search' in context['tools']:
                results = context['tools']['web_search'](query)
                if results:
                    snippets = '\\n'.join([f'- {{r[\"snippet\"]}}' for r in results[:2]])
                    return {{
                        'success': True,
                        'result': f"💱 Актуальные курсы:\\n{{snippets}}",
                        'data': None,
                        'requires_llm': False
                    }}
            return {{
                'success': False,
                'result': "❌ Не удалось получить курсы",
                'data': None,
                'requires_llm': True
            }}
        except Exception as e:
            return {{
                'success': False,
                'result': f"❌ Ошибка: {{str(e)}}",
                'data': None,
                'requires_llm': True
            }}

Описание ядра от пользователя:
{description}

ВЕРНИ ТОЛЬКО ЧИСТЫЙ КОД БЕЗ АННОТАЦИЙ ТИПОВ (без -> Dict, без : Dict[str, Any]).
Начинай с class.
"""

        try:
            resp = requests.post(
                LM_STUDIO_API_URL,
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {LM_STUDIO_API_KEY}'},
                json={
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.3,
                    'max_tokens': 1800
                },
                timeout=70
            )

            if resp.status_code != 200:
                return {'success': False, 'error': f"Ошибка генерации: {resp.status_code}"}

            code = resp.json()['choices'][0]['message']['content'].strip()
            code = re.sub(r'^```python\s*|^```\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'```\s*$', '', code)

            try:
                ast.parse(code)
            except SyntaxError as e:
                return {'success': False, 'error': f"Синтаксическая ошибка: {e}"}

            core_name_match = re.search(r'name\s*=\s*["\'](\w+)["\']', code)
            core_name = core_name_match.group(
                1) if core_name_match else f"core_{user_id}_{datetime.now().strftime('%H%M%S')}"
            filepath = os.path.join(CORES_DIR, f"{core_name}.py")

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# AUTO-GENERATED CORE - created by MiniBrain\n")
                f.write("# FIXED: No type annotations to avoid 'Dict not defined' error\n")
                f.write("from __main__ import KnowledgeCore, requests, re, json, datetime, os\n\n")
                f.write(code)

            prev_count = len(self.cores)
            self._load_dynamic_cores()
            new_count = len(self.cores)

            return {
                'success': True,
                'filename': f"{core_name}.py",
                'core_name': core_name,
                'new_cores': new_count - prev_count,
                'code_preview': code[:300] + "..." if len(code) > 300 else code
            }

        except Exception as e:
            return {'success': False, 'error': f"Ошибка создания ядра: {str(e)}"}


# ==================== МИНИ-МОЗГ ====================
class MiniBrain:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.tools = ToolsManager()

    def _web_search_tool(self, query: str, max_results: int = 3) -> List[Dict]:
        """Инструмент веб-поиска БЕЗ параметра timeout"""
        try:
            # УБРАН timeout — не поддерживается в новых версиях
            results = DDGS().text(query, max_results=max_results)
            return [
                {
                    'title': r.get('title', '')[:80],
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')[:200]
                }
                for r in results[:max_results]
            ]
        except Exception as e:
            print(f"  ⚠️ Ошибка поиска в инструменте: {e}")
            return []

    async def process(self, query: str) -> Dict[str, Any]:
        for core_name, core in list(self.tools.cores.items()):
            if core_name == "web_search_core":
                continue

            if core.can_handle(query):
                print(f"⚡ [{self.user_id}] Ядро '{core_name}' активировано для: {query[:40]}")
                result = core.execute(query, context={
                    'user_id': self.user_id,
                    'tools': {
                        'web_search': self._web_search_tool
                    }
                })

                if result['success'] and result.get('result') and not result.get('requires_llm', False):
                    await self._learn_from_interaction(query, result['result'])
                    return {
                        'type': 'tool_response',
                        'response': result['result'],
                        'source': core_name
                    }

                if result['success'] and result.get('data'):
                    return {
                        'type': 'llm_with_data',
                        'context': f"ДАННЫЕ ОТ ЯДРА '{core_name}':\n{json.dumps(result['data'], ensure_ascii=False, indent=2)}\n\nВОПРОС: {query}\n\nПроанализируй данные и дай точный ответ пользователю.",
                        'source': core_name
                    }

        web_core = self.tools.cores.get('web_search_core')
        if web_core and web_core.can_handle(query):
            print(f"🌐 [{self.user_id}] Активирован веб-поиск для: {query[:40]}")
            search_result = web_core.execute(query, context={'user_id': self.user_id})

            if search_result['success'] and search_result.get('data', {}).get('results'):
                results_text = "\n\n".join([
                    f"📄 {i + 1}. {r['title']}\n   {r['snippet']}\n   Источник: {r['url']}"
                    for i, r in enumerate(search_result['data']['results'])
                ])

                return {
                    'type': 'llm_with_search',
                    'context': (
                        f"РЕЗУЛЬТАТЫ ПОИСКА В ИНТЕРНЕТЕ (актуальная информация):\n{results_text}\n\n"
                        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}\n\n"
                        f"ИНСТРУКЦИИ:\n"
                        f"1. Дай точный ответ на основе этих данных\n"
                        f"2. Укажи источник информации (номер результата)\n"
                        f"3. Если данные противоречивы — сообщи об этом"
                    ),
                    'source': 'web_search'
                }

        long_term = await self.memory.search_long_term(query)
        short_term = self.memory.get_short_term()

        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        tools_info = "ДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n"
        for name, core in self.tools.cores.items():
            if name != "web_search_core":
                tools_info += f"- {core.name}: {core.description}\n"

        memory_info = "ЗАПОМНЕННЫЕ ФАКТЫ:\n" + (
            "\n".join([f"- {f['content']} ({f['timestamp'][:10]})" for f in long_term])
            if long_term else "Нет релевантных фактов"
        )

        context = f"""
Ты — умный ассистент с расширенными возможностями.
Текущее время: {now.strftime('%d.%m.%Y %H:%M')} ({weekdays[now.weekday()]})
Пользователь ID: {self.user_id}

{tools_info}

{memory_info}

ИСТОРИЯ ДИАЛОГА (последние сообщения):
{chr(10).join([f"{'👤' if m['role'] == 'user' else '🤖'}: {m['content'][:70]}..." for m in short_term[-4:]]) if short_term else 'Нет истории'}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ИНСТРУКЦИИ:
1. Для вопросов о текущем времени/дате — используй системное время выше
2. Для математики — используй встроенный калькулятор
3. Для актуальной информации (курсы, погода, новости) — используй данные из поиска выше
4. НЕ выдумывай информацию — опирайся только на предоставленные данные
"""
        return {'type': 'llm_normal', 'context': context}

    async def _learn_from_interaction(self, query: str, response: str):
        patterns = [
            r'меня зовут (\w+)', r'мое имя (\w+)', r'зовут (\w+)',
            r'я из (\w+)', r'живу в (\w+)', r'мой город (\w+)',
            r'мне нравится (\w+)', r'люблю (\w+)', r'мой любимый (\w+) —? (\w+)'
        ]
        for pattern in patterns:
            if match := re.search(pattern, query.lower()):
                fact = f"Пользователь упомянул: {match.group(0)}"
                await self.memory.save_long_term(fact, "user_preference")


# ==================== ТЕЛЕГРАМ БОТ ====================
class TelegramBot:
    def __init__(self):
        self.user_brains: Dict[str, MiniBrain] = {}

    def get_brain(self, user_id: str) -> MiniBrain:
        if user_id not in self.user_brains:
            self.user_brains[user_id] = MiniBrain(user_id)
        return self.user_brains[user_id]

    async def get_llm_response(self, context: str) -> str:
        try:
            resp = requests.post(
                LM_STUDIO_API_URL,
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {LM_STUDIO_API_KEY}'},
                json={
                    'messages': [{'role': 'system', 'content': context}],
                    'temperature': 0.4,
                    'max_tokens': 1500
                },
                timeout=90
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
            return f"❌ Ошибка модели (код {resp.status_code})"
        except requests.exceptions.Timeout:
            return "❌ Таймаут при обращении к модели. Попробуйте позже."
        except Exception as e:
            return f"❌ Ошибка связи с моделью: {str(e)}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 Привет! Я — ИИ с мини-мозгом и доступом к интернету:\n"
            "✅ Точная дата/время без интернета\n"
            "✅ Калькулятор для вычислений\n"
            "✅ ПОИСК В ИНТЕРНЕТЕ для курсов, погоды, новостей\n\n"
            "Просто спроси:\n"
            "• «курс доллара»\n"
            "• «погода в Москве»\n"
            "• «какой сегодня день»\n\n"
            "Команды:\n"
            "/create_core — создать новое ядро\n"
            "/list_cores — список ядер",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "💡 КАК ПОЛУЧИТЬ КУРС ВАЛЮТ:\n\n"
            "Просто напиши боту любой из запросов:\n"
            "• `курс доллара`\n"
            "• `курс евро`\n"
            "• `сколько стоит доллар`\n"
            "• `курс валют сегодня`\n\n"
            "Бот автоматически найдёт актуальную информацию в интернете и покажет реальные курсы!",
            parse_mode='Markdown'
        )

    async def list_cores(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        brain = self.get_brain(str(update.effective_user.id))
        msg = "🔧 ДОСТУПНЫЕ ЯДРА:\n\n"
        for i, (name, core) in enumerate(brain.tools.cores.items(), 1):
            caps = ', '.join(core.capabilities[:3]) if core.capabilities else 'общие задачи'
            msg += f"{i}. `{core.name}` — {core.description}\n   Возможности: {caps}\n\n"
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def create_core_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🛠️ ОПИШИ ЯДРО (без аннотаций типов):\n"
            "Пример:\n"
            "`Ядро для конвертации километров в мили`\n\n"
            "❗️Ядро будет использовать РЕАЛЬНЫЙ веб-поиск для получения данных",
            parse_mode='Markdown'
        )
        context.user_data['awaiting_core_desc'] = True

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        if context.user_data.get('awaiting_core_desc'):
            context.user_data['awaiting_core_desc'] = False
            await update.message.reply_text(
                "⏳ Генерирую РАБОЧИЙ код ядра...\n"
                "(20-40 секунд)",
                parse_mode='Markdown'
            )

            brain = self.get_brain(user_id)
            result = await brain.tools.create_core_from_description(text, user_id)

            if result['success']:
                await update.message.reply_text(
                    f"✅ Ядро `{result['core_name']}` создано!\n"
                    f"📁 Файл: `{result['filename']}`\n\n"
                    f"Теперь протестируй: например, «курс доллара»",
                    parse_mode='Markdown'
                )
                for uid, b in self.user_brains.items():
                    b.tools._load_dynamic_cores()
            else:
                await update.message.reply_text(
                    f"❌ Ошибка:\n{result['error']}",
                    parse_mode='Markdown'
                )
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        brain = self.get_brain(user_id)

        try:
            brain_resp = await brain.process(text)
            response = brain_resp['response'] if brain_resp['type'] == 'tool_response' else await self.get_llm_response(
                brain_resp['context'])

            brain.memory.add_short_term({'role': 'user', 'content': text})
            brain.memory.add_short_term({'role': 'assistant', 'content': response})

            max_chunk = 4000
            if len(response) > max_chunk:
                chunks = [response[i:i + max_chunk] for i in range(0, len(response), max_chunk)]
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode='Markdown', disable_web_page_preview=False)
            else:
                await update.message.reply_text(response, parse_mode='Markdown', disable_web_page_preview=False)

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)[:300]}")


# ==================== ЗАПУСК ====================
def main():
    print("=" * 70)
    print("🚀 ЗАПУСК ИСПРАВЛЕННОГО БОТА (без ошибок с timeout и Dict)")
    print("=" * 70)
    print(f"⏰ Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"📁 Ядра: ./{CORES_DIR}/")
    print(f"🧠 Память: ./{MEMORY_DIR}/")
    print(f"🔗 LM Studio: {LM_STUDIO_API_URL}")
    print("=" * 70)

    try:
        test_resp = requests.get(f"{LM_STUDIO_API_URL.replace('/v1/chat/completions', '/v1/models')}", timeout=5)
        if test_resp.status_code == 200:
            models = [m['id'] for m in test_resp.json().get('data', [])[:3]]
            print(f"✅ LM Studio доступна. Модели: {', '.join(models) if models else 'не определены'}")
        else:
            print("⚠️ LM Studio недоступна — бот запустится, но работа с моделью будет невозможна")
    except:
        print("⚠️ Не удалось проверить LM Studio — убедитесь, что она запущена на порту 1234")

    print("=" * 70)

    bot = TelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("list_cores", bot.list_cores))
    application.add_handler(CommandHandler("create_core", bot.create_core_start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот запущен! Напиши ему в Телеграм: /start")
    print("=" * 70)
    print("💡 Чтобы узнать курс валют — просто напиши «курс доллара»")
    print("=" * 70)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()