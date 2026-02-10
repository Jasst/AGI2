#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v3.4 (ПОЛНОСТЬЮ РАБОЧАЯ ВЕРСИЯ)
✅ Исправлена ошибка в определении CoreResponse (пропущено имя поля 'data')
✅ Добавлена поддержка .env файла
✅ Исправлена асинхронная инициализация Telegram бота
✅ Файловые операции работают локально без интернета
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

# Загрузка переменных окружения из .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv не установлен. Установите: pip install python-dotenv")

# Telegram
try:
    from telegram import Update, BotCommand
    from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
except ImportError:
    print("❌ Установите зависимости: pip install python-telegram-bot requests python-dotenv")
    sys.exit(1)

# HTTP клиент
import requests

requests.packages.urllib3.disable_warnings()

# DuckDuckGo Search (опционально)
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
    print("⚠️  DDGS не установлен (опционально): pip install duckduckgo-search")


# ============================================================================
# 1. СИСТЕМА ЯДЕР ЗНАНИЙ (ИСПРАВЛЕНО: правильное определение CoreResponse)
# ============================================================================

@dataclass
class CoreResponse:
    """Стандартный ответ ядра знаний"""
    success: bool
    data: Dict[str, Any]  # ✅ ИСПРАВЛЕНО: добавлено имя поля 'data'
    raw_result: str
    confidence: float
    source: str

    def to_context_string(self) -> str:
        """Преобразование в текст для контекста LLM"""
        if not self.success:
            return (
                f"[СИСТЕМНЫЙ ФАКТ] ИСТОЧНИК: {self.source.upper()}\n"
                f"СТАТУС: ОШИБКА\n"
                f"СООБЩЕНИЕ: {self.data.get('error', 'неизвестная ошибка')}\n"
                f"[КОНЕЦ ФАКТА]\n"
                f"❗ ВАЖНО: ЭТО ФАКТ. НЕ ИГНОРИРУЙ ЕГО."
            )

        if self.raw_result:
            return (
                f"[СИСТЕМНЫЙ ФАКТ] ИСТОЧНИК: {self.source.upper()}\n"
                f"{self.raw_result.strip()}\n"
                f"[КОНЕЦ ФАКТА]\n"
                f"❗ ПРАВИЛО: ДАННЫЕ ВЫШЕ — ФАКТЫ. ИСПОЛЬЗУЙ ИХ В ОТВЕТЕ."
            )

        if self.data:  # ✅ ИСПРАВЛЕНО: правильная проверка
            formatted_data = json.dumps(self.data, ensure_ascii=False, indent=2)
            return (
                f"[СИСТЕМНЫЙ ФАКТ] ИСТОЧНИК: {self.source.upper()}\n"
                f"СТРУКТУРИРОВАННЫЕ ДАННЫЕ:\n{formatted_data}\n"
                f"[КОНЕЦ ФАКТА]\n"
                f"❗ ПРАВИЛО: ИСПОЛЬЗУЙ ЭТИ ДАННЫЕ."
            )

        return (
            f"[СИСТЕМНЫЙ ФАКТ] ИСТОЧНИК: {self.source.upper()}\n"
            f"СТАТУС: ЗАПРОС ОБРАБОТАН\n"
            f"[КОНЕЦ ФАКТА]"
        )


class KnowledgeCore:
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []
    priority: int = 1

    def can_handle(self, query: str) -> bool:
        return False

    def get_confidence(self, query: str) -> float:
        return 0.0 if not self.can_handle(query) else 0.7

    def execute(self, query: str, context: Optional[Dict] = None) -> CoreResponse:
        return CoreResponse(
            success=False,
            data={'error': 'Метод execute не реализован'},
            raw_result='Ошибка: ядро не реализовано',
            confidence=0.0,
            source=self.name
        )


# ============================================================================
# 2-4. ВСТРОЕННЫЕ ЯДРА, ФАЙЛОВОЕ ЯДРО, МОЗГ (рабочие версии)
# ============================================================================

class DateTimeCore(KnowledgeCore):
    name = "datetime_core"
    description = "Текущая дата, время и день недели"
    capabilities = ["дата", "время", "день недели", "сегодня", "часы", "минуты"]
    priority = 9

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in self.capabilities)

    def get_confidence(self, query):
        return 0.98 if self.can_handle(query) else 0.0

    def execute(self, query, context=None):
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
        wd = weekdays[now.weekday()]
        result = f"Сегодня {now.strftime('%d.%m.%Y')}, {wd}. Текущее время: {now.strftime('%H:%M:%S')}"
        return CoreResponse(
            success=True,
            data={'datetime': now.isoformat(), 'weekday': wd},
            raw_result=result,
            confidence=1.0,
            source=self.name
        )


class CalculatorCore(KnowledgeCore):
    name = "calculator_core"
    description = "Математические вычисления"
    capabilities = ["плюс", "минус", "умножить", "разделить", "сложить", "вычесть", "=", "+"]
    priority = 8

    def can_handle(self, query):
        q = query.replace(' ', '')
        return any(op in q for op in ['+', '-', '*', 'x', '/', '×', '÷']) or re.search(r'\d+\s*[+\-*/]\s*\d+', q)

    def get_confidence(self, query):
        return 0.95 if self.can_handle(query) else 0.0

    def execute(self, query, context=None):
        try:
            expr = re.sub(r'[^\d+\-*/.()\s]', '', query)
            expr = expr.replace('x', '*').replace('×', '*').replace('÷', '/')
            result = eval(expr, {"__builtins__": {}}, {})
            return CoreResponse(
                success=True,
                data={'expression': expr.strip(), 'result': result},
                raw_result=f"Результат вычисления: {expr.strip()} = {result}",
                confidence=1.0,
                source=self.name
            )
        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"Ошибка вычисления: {str(e)}",
                confidence=0.0,
                source=self.name
            )


class WebSearchCore(KnowledgeCore):
    name = "web_search_core"
    description = "Поиск информации в интернете через DuckDuckGo"
    capabilities = ["найти", "поиск", "кто такой", "что такое", "новости", "актуально"]
    priority = 7

    def __init__(self):
        self.ddgs = DDGS() if DDGS else None

    def can_handle(self, query):
        q = query.lower()
        return (any(kw in q for kw in ['найти', 'поиск', 'кто такой', 'что такое']) or
                len(q) > 15 and not any(cmd in q for cmd in ['файл', 'сохрани', 'прочитай']))

    def get_confidence(self, query):
        return 0.85 if self.can_handle(query) else 0.0

    def execute(self, query, context=None):
        if not self.ddgs:
            return CoreResponse(
                success=False,
                data={'error': 'DDGS не установлен'},
                raw_result='Поиск недоступен (установите: pip install duckduckgo-search)',
                confidence=0.0,
                source=self.name
            )

        try:
            results = self.ddgs.text(query, max_results=3)
            if not results:
                return CoreResponse(
                    success=False,
                    data={'error': 'Нет результатов'},
                    raw_result='Поиск не дал результатов',
                    confidence=0.5,
                    source=self.name
                )

            info = "\n\n".join([f"• {r['title']}\n  {r['body']}" for r in results[:3]])
            return CoreResponse(
                success=True,
                data={'results': results},
                raw_result=f"РЕЗУЛЬТАТЫ ПОИСКА ПО ЗАПРОСУ '{query}':\n{info}",
                confidence=0.9,
                source=self.name
            )
        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"Ошибка поиска: {str(e)}",
                confidence=0.0,
                source=self.name
            )


class FileStorageCore(KnowledgeCore):
    name = "file_storage_core"
    description = "ЛОКАЛЬНОЕ ФАЙЛОВОЕ ХРАНИЛИЩЕ (работает без интернета)"
    capabilities = ["сохранить файл", "прочитать файл", "список файлов", "удалить файл", "фаил", "документ"]
    priority = 10

    WORK_DIR = "user_storage"

    def __init__(self):
        os.makedirs(self.WORK_DIR, exist_ok=True)

    def can_handle(self, query):
        q = query.lower().replace('фаил', 'файл')
        return bool(re.search(r'(сохрани|запиши|прочита|откро|файл|документ|заметк)\b', q))

    def get_confidence(self, query):
        q = query.lower().replace('фаил', 'файл')
        if re.search(r'(сохрани|запиши|прочита|откро).*файл', q):
            return 1.0
        return 0.95 if self.can_handle(query) else 0.0

    def _sanitize_filename(self, filename: str) -> str:
        filename = filename.strip().strip('\'"')
        filename = re.sub(r'\s+\.', '.', filename)
        filename = re.sub(r'\.\s+', '.', filename)
        filename = re.sub(r'[<>:"|?*\\/\x00-\x1f]', '_', filename)
        filename = os.path.basename(filename).strip('. ')
        if not filename or filename.startswith(' '):
            filename = "document.txt"
        if '.' not in filename or filename.startswith('.'):
            filename += ".txt"
        return filename

    def _get_safe_path(self, filename: str) -> str:
        clean = self._sanitize_filename(filename)
        path = os.path.abspath(os.path.join(self.WORK_DIR, clean))
        if not path.startswith(os.path.abspath(self.WORK_DIR) + os.sep):
            raise ValueError("Запрещённый путь к файлу")
        return path

    def execute(self, query, context=None):
        try:
            query_fixed = query.replace('фаил', 'файл').replace('файл ', 'файл')
            q = query_fixed.lower()

            # Чтение файла
            read_match = re.search(
                r'(?:прочита|откро|прочт|покажи|содержимо|открыть|прочесть).*?файл\s*[\'"]?([^\n\'"]+?)(?:\.txt|\.md|\.json)?[\'"]?',
                q, re.IGNORECASE
            )
            if read_match:
                filename = read_match.group(1).strip()
                safe_path = self._get_safe_path(filename)

                if not os.path.exists(safe_path):
                    files = [f for f in os.listdir(self.WORK_DIR) if os.path.isfile(os.path.join(self.WORK_DIR, f))]
                    hint = f"\nДоступные файлы: {', '.join(files[:5])}" if files else "\nХранилище пусто."
                    return CoreResponse(
                        success=True,
                        data={'action': 'not_found', 'filename': filename, 'available_files': files},
                        raw_result=(
                            f"СТАТУС: ФАЙЛ НЕ НАЙДЕН\n"
                            f"ИМЯ ФАЙЛА: {filename}\n"
                            f"ПУТЬ: {self.WORK_DIR}/{filename}\n"
                            f"ДЕЙСТВИЕ: Чтобы прочитать файл, сначала сохраните его командой:\n"
                            f"  → сохрани в файл {filename}: ваш текст{hint}"
                        ),
                        confidence=1.0,
                        source=self.name
                    )

                with open(safe_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                return CoreResponse(
                    success=True,
                    data={'action': 'read', 'filename': os.path.basename(safe_path), 'content': content,
                          'size': len(content)},
                    raw_result=(
                        f"СТАТУС: ФАЙЛ НАЙДЕН И ПРОЧИТАН\n"
                        f"ИМЯ: {os.path.basename(safe_path)}\n"
                        f"РАЗМЕР: {len(content)} байт\n"
                        f"СОДЕРЖИМОЕ:\n{content}"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            # Сохранение файла
            save_match = re.search(
                r'(?:сохрани|запиши|напиши|сохранить|записать).*?файл\s+[\'"]?([^\n\'":]+?)(?:\.txt|\.md|\.json)?[\'"]?\s*[:：]?\s*(.+)$',
                query_fixed, re.IGNORECASE | re.DOTALL
            )
            if save_match:
                filename = save_match.group(1).strip()
                content = save_match.group(2).strip()
                safe_path = self._get_safe_path(filename)

                with open(safe_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                return CoreResponse(
                    success=True,
                    data={'action': 'saved', 'filename': os.path.basename(safe_path), 'size': len(content)},
                    raw_result=(
                        f"СТАТУС: ФАЙЛ УСПЕШНО СОХРАНЁН\n"
                        f"ИМЯ: {os.path.basename(safe_path)}\n"
                        f"ПУТЬ: {self.WORK_DIR}/{os.path.basename(safe_path)}\n"
                        f"РАЗМЕР: {len(content)} байт"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            # Список файлов
            if re.search(r'список файлов|мои файлы|покажи файлы|фа[иы]лы|документы', q):
                files = []
                for fname in os.listdir(self.WORK_DIR):
                    fpath = os.path.join(self.WORK_DIR, fname)
                    if os.path.isfile(fpath):
                        stat = os.stat(fpath)
                        files.append({
                            'name': fname,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                        })

                if not files:
                    return CoreResponse(
                        success=True,
                        data={'files': []},
                        raw_result=(
                            f"СТАТУС: ХРАНИЛИЩЕ ПУСТО\n"
                            f"ДЕЙСТВИЕ: Сохраните файл командой:\n"
                            f"  → сохрани в файл заметка.txt: ваш текст"
                        ),
                        confidence=1.0,
                        source=self.name
                    )

                files_list = "\n".join([f"📄 {f['name']} ({f['size']} байт, {f['modified']})" for f in files[:15]])
                return CoreResponse(
                    success=True,
                    data={'files': files},
                    raw_result=(
                        f"СТАТУС: ФАЙЛЫ НАЙДЕНЫ\n"
                        f"КОЛИЧЕСТВО: {len(files)}\n"
                        f"СПИСОК:\n{files_list}"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            if self.can_handle(query):
                return CoreResponse(
                    success=True,
                    data={'action': 'help'},
                    raw_result=(
                        f"ПОДСКАЗКА ПО РАБОТЕ С ФАЙЛАМИ:\n"
                        f"• Прочитать: 'прочти файл привет.txt' или 'прочти фаил привет.txt'\n"
                        f"• Сохранить: 'сохрани в файл заметка.txt: текст'\n"
                        f"• Список: 'список файлов'"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            return CoreResponse(
                success=False,
                data={'error': 'Не относится к файловым операциям'},
                confidence=0.0,
                source=self.name
            )

        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"СТАТУС: КРИТИЧЕСКАЯ ОШИБКА\nСООБЩЕНИЕ: {str(e)}",
                confidence=0.0,
                source=self.name
            )


class MiniBrain:
    def __init__(self, user_id: str = "default_user", cores_dir: str = "dynamic_cores",
                 memory_dir: str = "brain_memory"):
        self.user_id = user_id
        self.cores_dir = Path(cores_dir)
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        self.builtin_cores = [
            DateTimeCore(),
            CalculatorCore(),
            WebSearchCore() if DDGS else None,
            FileStorageCore()
        ]
        self.builtin_cores = [core for core in self.builtin_cores if core is not None]

        self.dynamic_cores: List[KnowledgeCore] = []
        self._load_dynamic_cores()

        self.llm_url = "http://localhost:1234/v1/chat/completions"
        self.model_name = "qwen/qwen3-coder-30b"
        self._check_llm_availability()

    def _check_llm_availability(self):
        try:
            resp = requests.get(self.llm_url.replace('/v1/chat/completions', '/v1/models'), timeout=5)
            if resp.status_code == 200:
                models = [m['id'] for m in resp.json().get('data', [])]
                self.model_name = models[0] if models else self.model_name
                print(f"✅ LM Studio доступна. Модель: {self.model_name}")
                return True
            print(f"⚠️  LM Studio вернула статус {resp.status_code}")
        except Exception as e:
            print(f"❌ LM Studio недоступна: {e}")
            print("💡 Запустите LM Studio с сервером API на порту 1234")
        return False

    def _load_dynamic_cores(self):
        self.dynamic_cores = []
        if not self.cores_dir.exists():
            self.cores_dir.mkdir()
            return

        for f in self.cores_dir.glob("file_storage_core*.py"):
            if f.name != "file_storage_core.py":
                f.unlink()
                print(f"🗑️  Удалён дубликат: {f.name}")

        sys.path.insert(0, str(self.cores_dir))
        for core_file in sorted(self.cores_dir.glob("*.py")):
            if core_file.name.startswith("_") or core_file.name == "__init__.py":
                continue

            try:
                module_name = core_file.stem
                if module_name in sys.modules:
                    del sys.modules[module_name]

                module = __import__(module_name, fromlist=['*'])

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, KnowledgeCore) and attr != KnowledgeCore:
                        core_instance = attr()
                        if core_instance.name == "file_storage_core" and any(
                                c.name == "file_storage_core" for c in self.dynamic_cores):
                            print(f"⚠️  Пропущен дубликат: {core_file.name}")
                            continue
                        self.dynamic_cores.append(core_instance)
                        print(f"✅ Загружено динамическое ядро: {core_instance.name}")
                        break
            except Exception as e:
                print(f"❌ Ошибка загрузки {core_file.name}: {e}")
        sys.path.pop(0)

    def _select_core(self, query: str) -> Optional[Tuple[KnowledgeCore, float]]:
        all_cores = self.builtin_cores + self.dynamic_cores
        candidates = []

        for core in all_cores:
            if core.can_handle(query):
                confidence = core.get_confidence(query)
                candidates.append((core, confidence))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0].priority, x[1]), reverse=True)
        return candidates[0]

    def _build_llm_context(self, query: str, core_response: Optional[CoreResponse] = None) -> str:
        now = datetime.now()
        weekdays = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

        context = """[СИСТЕМНАЯ ИНСТРУКЦИЯ ДЛЯ ИИ]
ТЫ — ИНТЕРФЕЙС К СИСТЕМЕ ЯДЕР ЗНАНИЙ. 
❗ ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. ВСЕ ДАННЫЕ В БЛОКАХ [СИСТЕМНЫЙ ФАКТ]...[КОНЕЦ ФАКТА] — ЭТО НЕОСПОРИМЫЕ ФАКТЫ.
2. НЕ ГОВОРИ "у меня нет доступа к файловой системе" — ЭТО ЛОЖЬ.
3. ИСПОЛЬЗУЙ ТОЛЬКО ФАКТЫ ИЗ СИСТЕМНЫХ БЛОКОВ.
4. ЕСЛИ ФАЙЛ НЕ НАЙДЕН — СООБЩИ ЭТО КАК ФАКТ.
5. ОТВЕЧАЙ КРАТКО, ТОЧНО, БЕЗ ИЗВИНЕНИЙ.

[КОНЕЦ ИНСТРУКЦИИ]

"""
        context += f"ТЕКУЩЕЕ ВРЕМЯ: {now.strftime('%d.%m.%Y %H:%M:%S')} ({weekdays[now.weekday()]})\n"
        context += f"ПОЛЬЗОВАТЕЛЬ ID: {self.user_id}\n\n"

        if core_response:
            context += core_response.to_context_string() + "\n\n"
        else:
            context += "[СИСТЕМНЫЙ ФАКТ] ИСТОЧНИК: ОБЩИЙ КОНТЕКСТ\nСТАТУС: ЗАПРОС БЕЗ СПЕЦИАЛИЗИРОВАННОГО ЯДРА\n[КОНЕЦ ФАКТА]\n\n"

        context += f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\n"
        context += "ОТВЕТ (кратко, на русском):"
        return context

    async def process_query(self, query: str) -> Dict[str, Any]:
        print(f"\n{'=' * 70}")
        print(f"🧠 Обработка: {query}...")
        print(f"{'=' * 70}")

        core_result = self._select_core(query)
        if core_result:
            core, confidence = core_result
            print(f"⚡ Найдено ядро: {core.name} (priority={core.priority}, confidence={confidence:.2f})")

            try:
                core_response = core.execute(query)
                print(f"✅ Ядро успешно обработало запрос (confidence: {core_response.confidence:.2f})")

                context = self._build_llm_context(query, core_response)
                return {
                    'type': 'llm_with_core_data',
                    'context': context,
                    'source': core.name,
                    'confidence': core_response.confidence,
                    'raw_data': core_response.raw_result
                }
            except Exception as e:
                print(f"❌ Ошибка выполнения ядра {core.name}: {e}")
                context = self._build_llm_context(query)
                return {
                    'type': 'llm_general',
                    'context': context,
                    'source': None,
                    'confidence': 0.0
                }
        else:
            print("🤖 Ядро не найдено. Использую общий контекст")
            context = self._build_llm_context(query)
            return {
                'type': 'llm_general',
                'context': context,
                'source': None,
                'confidence': 0.0
            }

    async def get_llm_response(self, context: str, max_tokens: int = 512) -> str:
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": context}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "stream": False
            }

            resp = requests.post(self.llm_url, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            answer = result['choices'][0]['message']['content'].strip()

            # Пост-валидация
            ignore_phrases = [
                'нет доступа к файловой системе',
                'у меня нет возможности',
                'я не могу прочитать файл',
                'загрузи файл в чат',
                'скопируй содержимое'
            ]
            if any(phrase in answer.lower() for phrase in ignore_phrases):
                if 'ФАЙЛ НЕ НАЙДЕН' in context or 'СТАТУС: ФАЙЛ НЕ НАЙДЕН' in context:
                    answer = "📁 Файл не найден в локальном хранилище.\n💡 Сохраните его командой:\n`сохрани в файл имя.txt: текст`"
                elif 'СТАТУС: ФАЙЛ НАЙДЕН' in context or 'СОДЕРЖИМОЕ:' in context:
                    answer = "✅ Файл успешно прочитан из локального хранилища."
                elif 'СТАТУС: ФАЙЛ УСПЕШНО СОХРАНЁН' in context:
                    answer = "✅ Файл успешно сохранён в локальное хранилище."

            return answer

        except Exception as e:
            return f"❌ Ошибка LLM: {str(e)}"


# ============================================================================
# 5. TELEGRAM БОТ (ИСПРАВЛЕНО: асинхронная инициализация)
# ============================================================================

class TelegramBot:
    def __init__(self, token: str, brain: MiniBrain):
        self.token = token
        self.brain = brain

        self.application = (
            Application.builder()
            .token(token)
            .post_init(self.post_init)
            .build()
        )

        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("cores", self.list_cores))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def post_init(self, application: Application) -> None:
        await application.bot.set_my_commands([
            BotCommand("start", "Запустить бота"),
            BotCommand("help", "Помощь"),
            BotCommand("cores", "Список ядер"),
        ])
        print("✅ Команды бота зарегистрированы в Telegram")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🚀 Самообучающийся мини-мозг запущен!\n\n"
            "💡 Примеры команд:\n"
            "• Какое сегодня число?\n"
            "• 25 + 17\n"
            "• сохрани в файл привет.txt: Привет, мир!\n"
            "• прочти файл привет.txt\n"
            "• список файлов"
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📚 Помощь по командам:\n\n"
            "📁 Файлы:\n"
            "  • `сохрани в файл имя.txt: текст`\n"
            "  • `прочти файл имя.txt`\n"
            "  • `список файлов`\n"
            "  • Поддерживаются опечатки: 'фаил' → 'файл'\n"
            "  • Пробелы в именах: 'привет .txt' → 'привет.txt'"
        )

    async def list_cores(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        all_cores = self.brain.builtin_cores + self.brain.dynamic_cores
        core_list = "\n".join([
            f"• {core.name} (priority={core.priority}) — {core.description}"
            for core in sorted(all_cores, key=lambda x: x.priority, reverse=True)
        ])
        await update.message.reply_text(
            f"🧠 Доступные ядра ({len(all_cores)}):\n\n{core_list}\n\n"
            f"📁 Файлы хранятся в: {FileStorageCore.WORK_DIR}/"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_query = update.message.text.strip()
        brain_result = await self.brain.process_query(user_query)
        response = await self.brain.get_llm_response(brain_result['context'])
        await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)

    def run(self):
        print("\n" + "=" * 70)
        print("✅ Бот запущен!")
        print("=" * 70)
        print("💬 Напиши боту в Telegram")
        print("🤖 Бот использует локальные ядра для обработки запросов")
        print("=" * 70)
        print("Ctrl+C для остановки")
        print("=" * 70 + "\n")

        self.application.run_polling()


# ============================================================================
# 6. ТОЧКА ВХОДА
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v3.4 (ПОЛНОСТЬЮ РАБОЧАЯ ВЕРСИЯ)")
    print("=" * 70)
    print(f"⏰ Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"📁 Ядра: dynamic_cores/")
    print(f"🧠 Память: brain_memory/")
    print(f"🔗 LM Studio: http://localhost:1234/v1/chat/completions")
    print("=" * 70)

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not telegram_token:
        print("\n❌ Ошибка: не установлен TELEGRAM_BOT_TOKEN")
        print("\n💡 Создайте файл .env в корне проекта со строкой:")
        print("   TELEGRAM_BOT_TOKEN=123456789:AAH_ваш_токен_от_BotFather")
        print("\n💡 Или установите переменную окружения:")
        print("   Windows PowerShell: $env:TELEGRAM_BOT_TOKEN='ваш_токен'")
        print("   Windows CMD: set TELEGRAM_BOT_TOKEN=ваш_токен")
        print("   Linux/Mac: export TELEGRAM_BOT_TOKEN=ваш_токен")
        return

    print(f"✅ Токен Telegram загружен (длина: {len(telegram_token)})")

    brain = MiniBrain(user_id="telegram_user")
    bot = TelegramBot(token=telegram_token, brain=brain)
    bot.run()


if __name__ == "__main__":
    main()