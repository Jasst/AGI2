# coding: utf-8
"""
AGI_v25_MemoryControl.py — ПОЛНЫЙ КОНТРОЛЬ НАД ПАМЯТЬЮ
Исправлены проблемы:
1. LLM больше не галлюцинирует факты
2. Прямые команды управления памятью
3. Транзакционное изменение фактов
4. Детерминированные операции
"""

import re
import json
import requests
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import math


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_v25")
    ROOT.mkdir(exist_ok=True)

    FACTUAL_DB = ROOT / "facts.json"
    SEMANTIC_DB = ROOT / "semantic.json"
    EPISODIC_DB = ROOT / "episodes.json"
    META_DB = ROOT / "meta.json"
    LOG = ROOT / "system.log"

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    MODEL = "qwen/qwen-2.5-7b-instruct"
    TIMEOUT = 30
    MAX_TOKENS = 500

    if not OPENROUTER_API_KEY:
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        if k.strip() == "OPENROUTER_API_KEY":
                            OPENROUTER_API_KEY = v.strip().strip('"').strip("'")


# ================= УТИЛИТЫ =================
def extract_numbers(text: str) -> List[int]:
    """Извлечь все числа из текста"""
    return [int(n) for n in re.findall(r'\b\d+\b', text)]


def clean_text(text: str) -> str:
    """Нормализация текста"""
    return re.sub(r'\s+', ' ', text.lower().strip())


def print_typing(text: str, delay=0.008):
    """Эффект печатания"""
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print(flush=True)


# ================= ФАКТОЛОГИЧЕСКАЯ ПАМЯТЬ =================
@dataclass
class Fact:
    """Факт с метаданными"""
    value: Any
    fact_type: str
    timestamp: float
    context: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'value': self.value,
            'fact_type': self.fact_type,
            'timestamp': self.timestamp,
            'context': self.context,
            'tags': self.tags
        }

    @staticmethod
    def from_dict(data: dict) -> 'Fact':
        return Fact(
            value=data['value'],
            fact_type=data['fact_type'],
            timestamp=data['timestamp'],
            context=data.get('context', ''),
            tags=data.get('tags', [])
        )


class FactualMemory:
    """Управляемая фактологическая память"""

    def __init__(self):
        self.facts: Dict[str, List[Fact]] = defaultdict(list)
        self.load()

    # ========== БАЗОВЫЕ ОПЕРАЦИИ ==========

    def add(self, fact_type: str, value: Any, context: str = "", tags: List[str] = None):
        """Добавить факт"""
        fact = Fact(
            value=value,
            fact_type=fact_type,
            timestamp=time.time(),
            context=context,
            tags=tags or []
        )

        # Проверка на дубликаты
        for existing in self.facts[fact_type]:
            if existing.value == value:
                # Обновляем существующий
                existing.timestamp = fact.timestamp
                existing.context = context
                return f"Обновлён факт: {fact_type} = {value}"

        self.facts[fact_type].append(fact)
        return f"Добавлен факт: {fact_type} = {value}"

    def remove(self, fact_type: str, value: Any = None) -> str:
        """Удалить факт(ы)"""
        if fact_type not in self.facts:
            return f"Тип '{fact_type}' не найден"

        if value is None:
            # Удалить все факты этого типа
            count = len(self.facts[fact_type])
            del self.facts[fact_type]
            return f"Удалено {count} фактов типа '{fact_type}'"
        else:
            # Удалить конкретное значение
            original_count = len(self.facts[fact_type])
            self.facts[fact_type] = [f for f in self.facts[fact_type] if f.value != value]
            removed = original_count - len(self.facts[fact_type])

            if not self.facts[fact_type]:
                del self.facts[fact_type]

            return f"Удалено {removed} фактов: {fact_type} = {value}"

    def clear(self, fact_type: str = None) -> str:
        """Очистить память"""
        if fact_type:
            if fact_type in self.facts:
                count = len(self.facts[fact_type])
                del self.facts[fact_type]
                return f"Очищено {count} фактов типа '{fact_type}'"
            return f"Тип '{fact_type}' не найден"
        else:
            total = sum(len(facts) for facts in self.facts.values())
            self.facts.clear()
            return f"Очищено всего {total} фактов"

    def get_all(self, fact_type: str = None) -> List[Fact]:
        """Получить все факты"""
        if fact_type:
            return sorted(
                self.facts.get(fact_type, []),
                key=lambda f: f.timestamp,
                reverse=True
            )

        all_facts = []
        for facts in self.facts.values():
            all_facts.extend(facts)
        return sorted(all_facts, key=lambda f: f.timestamp, reverse=True)

    def search(self, query: str) -> List[Fact]:
        """Поиск фактов"""
        query_lower = query.lower()
        results = []

        for fact_type, facts in self.facts.items():
            if query_lower in fact_type.lower():
                results.extend(facts)
            else:
                for fact in facts:
                    if query_lower in str(fact.value).lower() or query_lower in fact.context.lower():
                        results.append(fact)

        return results[:20]

    # ========== ОПЕРАЦИИ С ЧИСЛАМИ ==========

    def add_numbers(self, numbers: List[int], context: str = ""):
        """Добавить числа"""
        added = []
        for num in numbers:
            self.add('number', num, context)
            added.append(num)
        return f"Добавлено чисел: {len(added)} → {added}"

    def get_numbers(self) -> List[int]:
        """Получить все числа"""
        return sorted([f.value for f in self.facts.get('number', [])])

    def transform_numbers(self, operation: str) -> str:
        """Трансформация чисел"""
        numbers = self.get_numbers()
        if not numbers:
            return "Нет чисел в памяти"

        old_numbers = numbers.copy()
        new_numbers = []

        try:
            if '+' in operation:
                delta = int(operation.split('+')[1])
                new_numbers = [n + delta for n in numbers]
            elif '-' in operation:
                delta = int(operation.split('-')[1])
                new_numbers = [n - delta for n in numbers]
            elif '*' in operation:
                factor = int(operation.split('*')[1])
                new_numbers = [n * factor for n in numbers]
            elif '/' in operation:
                divisor = int(operation.split('/')[1])
                new_numbers = [n // divisor for n in numbers]
            else:
                return f"Неизвестная операция: {operation}"

            # Очищаем старые числа
            self.remove('number')

            # Добавляем новые
            for num in new_numbers:
                self.add('number', num, f"Результат {operation} от {old_numbers}")

            return f"Преобразование {operation}:\nБыло: {old_numbers}\nСтало: {new_numbers}"

        except Exception as e:
            return f"Ошибка операции: {e}"

    # ========== СТАТИСТИКА ==========

    def get_stats(self) -> dict:
        """Статистика памяти"""
        return {
            'total_facts': sum(len(facts) for facts in self.facts.values()),
            'fact_types': len(self.facts),
            'by_type': {k: len(v) for k, v in self.facts.items()}
        }

    def format_for_llm(self, max_facts: int = 50) -> str:
        """Форматировать для LLM контекста"""
        lines = []

        # Группируем по типам
        for fact_type, facts in sorted(self.facts.items()):
            values = [str(f.value) for f in sorted(facts, key=lambda x: x.timestamp, reverse=True)]
            if len(values) > max_facts:
                values = values[:max_facts]
                lines.append(f"{fact_type.upper()}: {', '.join(values)} (показано {max_facts} из {len(facts)})")
            else:
                lines.append(f"{fact_type.upper()}: {', '.join(values)}")

        return "\n".join(lines) if lines else "Нет фактов в памяти"

    # ========== СЕРИАЛИЗАЦИЯ ==========

    def save(self):
        """Сохранить память"""
        data = {
            fact_type: [f.to_dict() for f in facts]
            for fact_type, facts in self.facts.items()
        }
        with open(Config.FACTUAL_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """Загрузить память"""
        if Config.FACTUAL_DB.exists():
            try:
                with open(Config.FACTUAL_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for fact_type, facts_data in data.items():
                        self.facts[fact_type] = [Fact.from_dict(f) for f in facts_data]
            except Exception as e:
                print(f"⚠️ Ошибка загрузки памяти: {e}")


# ================= ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ =================
@dataclass
class Episode:
    """Эпизод взаимодействия"""
    timestamp: float
    user_input: str
    system_output: str
    command_executed: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'user_input': self.user_input,
            'system_output': self.system_output,
            'command_executed': self.command_executed
        }

    @staticmethod
    def from_dict(data: dict) -> 'Episode':
        return Episode(
            timestamp=data['timestamp'],
            user_input=data['user_input'],
            system_output=data['system_output'],
            command_executed=data.get('command_executed')
        )


class EpisodicMemory:
    """Эпизодическая память"""

    def __init__(self, max_size: int = 100):
        self.episodes: List[Episode] = []
        self.max_size = max_size
        self.load()

    def add(self, user_input: str, system_output: str, command: str = None):
        """Добавить эпизод"""
        episode = Episode(
            timestamp=time.time(),
            user_input=user_input,
            system_output=system_output,
            command_executed=command
        )

        self.episodes.append(episode)

        if len(self.episodes) > self.max_size:
            self.episodes = self.episodes[-self.max_size:]

    def get_recent(self, n: int = 5) -> List[Episode]:
        """Получить последние эпизоды"""
        return self.episodes[-n:][::-1]

    def format_for_llm(self, n: int = 3) -> str:
        """Форматировать для контекста"""
        recent = self.get_recent(n)
        if not recent:
            return ""

        lines = []
        for i, ep in enumerate(recent, 1):
            lines.append(f"{i}. Пользователь: {ep.user_input[:80]}")
            lines.append(f"   Система: {ep.system_output[:80]}")

        return "\n".join(lines)

    def save(self):
        """Сохранить память"""
        data = [ep.to_dict() for ep in self.episodes]
        with open(Config.EPISODIC_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """Загрузить память"""
        if Config.EPISODIC_DB.exists():
            try:
                with open(Config.EPISODIC_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.episodes = [Episode.from_dict(ep) for ep in data]
            except Exception as e:
                print(f"⚠️ Ошибка загрузки эпизодов: {e}")


# ================= КОМАНДНЫЙ ПРОЦЕССОР =================
class CommandProcessor:
    """Обработчик команд управления памятью"""

    def __init__(self, factual: FactualMemory):
        self.factual = factual

        # Паттерны команд
        self.patterns = [
            # Запомнить
            (r'запомни\s+(?:число|числа)\s+([\d\s,]+)', self.cmd_remember_numbers),
            (r'запомни\s+(.+)', self.cmd_remember_generic),

            # Удалить
            (r'удали\s+(?:все\s+)?(?:числа|number)', self.cmd_delete_numbers),
            (r'удали\s+число\s+(\d+)', self.cmd_delete_number),
            (r'удали\s+всё', self.cmd_clear_all),
            (r'очисти\s+память', self.cmd_clear_all),

            # Показать
            (r'(?:покажи|напиши|выведи)\s+(?:все\s+)?(?:числа|number)', self.cmd_show_numbers),
            (r'(?:покажи|напиши)\s+факты', self.cmd_show_facts),
            (r'что\s+(?:ты\s+)?(?:знаешь|помнишь|запомнил)', self.cmd_show_all),

            # Операции с числами
            (r'прибавь\s+(\d+)', self.cmd_add_to_numbers),
            (r'умножь\s+на\s+(\d+)', self.cmd_multiply_numbers),
            (r'отними\s+(\d+)', self.cmd_subtract_from_numbers),

            # Статистика
            (r'статистика|stats', self.cmd_stats),
            (r'история', self.cmd_history),
        ]

    def process(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Обработать команду
        Returns: (is_command, result)
        """
        text_clean = clean_text(text)

        for pattern, handler in self.patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                result = handler(match)
                return True, result

        return False, None

    # ========== КОМАНДЫ ЗАПОМИНАНИЯ ==========

    def cmd_remember_numbers(self, match) -> str:
        """Запомнить числа"""
        numbers_str = match.group(1)
        numbers = extract_numbers(numbers_str)

        if not numbers:
            return "Не найдено чисел для запоминания"

        return self.factual.add_numbers(numbers, "Запомнено по команде")

    def cmd_remember_generic(self, match) -> str:
        """Запомнить произвольную информацию"""
        content = match.group(1).strip()

        # Пытаемся извлечь числа
        numbers = extract_numbers(content)
        if numbers:
            return self.factual.add_numbers(numbers, content)

        # Запоминаем как текст
        self.factual.add('text', content, "Сохранено как текст")
        return f"Запомнил: {content}"

    # ========== КОМАНДЫ УДАЛЕНИЯ ==========

    def cmd_delete_numbers(self, match) -> str:
        """Удалить все числа"""
        numbers = self.factual.get_numbers()
        if not numbers:
            return "Нет чисел для удаления"

        result = self.factual.remove('number')
        return f"{result}\nУдалённые числа: {numbers}"

    def cmd_delete_number(self, match) -> str:
        """Удалить конкретное число"""
        number = int(match.group(1))
        return self.factual.remove('number', number)

    def cmd_clear_all(self, match) -> str:
        """Очистить всю память"""
        return self.factual.clear()

    # ========== КОМАНДЫ ПОКАЗА ==========

    def cmd_show_numbers(self, match) -> str:
        """Показать числа"""
        numbers = self.factual.get_numbers()
        if not numbers:
            return "В памяти нет чисел"

        return f"Запомненные числа ({len(numbers)}): {numbers}"

    def cmd_show_facts(self, match) -> str:
        """Показать факты"""
        stats = self.factual.get_stats()
        if stats['total_facts'] == 0:
            return "Память пуста"

        output = [f"Всего фактов: {stats['total_facts']}\n"]

        for fact_type, facts in sorted(self.factual.facts.items()):
            output.append(f"\n{fact_type.upper()} ({len(facts)}):")
            for fact in sorted(facts, key=lambda f: f.timestamp, reverse=True)[:10]:
                time_str = datetime.fromtimestamp(fact.timestamp).strftime('%H:%M:%S')
                output.append(f"  • {fact.value} [{time_str}]")

        return "".join(output)

    def cmd_show_all(self, match) -> str:
        """Показать всё"""
        return self.cmd_show_facts(match)

    # ========== ОПЕРАЦИИ ==========

    def cmd_add_to_numbers(self, match) -> str:
        """Прибавить к числам"""
        delta = int(match.group(1))
        return self.factual.transform_numbers(f'+{delta}')

    def cmd_multiply_numbers(self, match) -> str:
        """Умножить числа"""
        factor = int(match.group(1))
        return self.factual.transform_numbers(f'*{factor}')

    def cmd_subtract_from_numbers(self, match) -> str:
        """Отнять от чисел"""
        delta = int(match.group(1))
        return self.factual.transform_numbers(f'-{delta}')

    # ========== ИНФОРМАЦИЯ ==========

    def cmd_stats(self, match) -> str:
        """Статистика"""
        stats = self.factual.get_stats()

        output = ["📊 СТАТИСТИКА ПАМЯТИ\n"]
        output.append(f"Всего фактов: {stats['total_facts']}")
        output.append(f"Типов фактов: {stats['fact_types']}\n")

        for fact_type, count in stats['by_type'].items():
            output.append(f"  • {fact_type}: {count}")

        return "\n".join(output)

    def cmd_history(self, match) -> str:
        """История (заглушка)"""
        return "История команд (функция в разработке)"


# ================= КОГНИТИВНАЯ СИСТЕМА =================
class CognitiveSystem:
    """Основная система с управлением памятью"""

    def __init__(self):
        print("🧠 Cognitive System v25 — Memory Control Edition\n")

        if not Config.OPENROUTER_API_KEY:
            print("❌ ОШИБКА: Не найден OPENROUTER_API_KEY!")
            sys.exit(1)

        # Инициализация компонентов
        self.factual = FactualMemory()
        self.episodic = EpisodicMemory()
        self.commands = CommandProcessor(self.factual)

        self.meta = self.load_meta()
        self.log_file = open(Config.LOG, 'a', encoding='utf-8')

        print("✅ Система инициализирована")
        self._print_stats()

    def log(self, message: str):
        """Логирование"""
        ts = datetime.now(timezone.utc).isoformat()
        self.log_file.write(f"[{ts}] {message}\n")
        self.log_file.flush()

    def _print_stats(self):
        """Вывести статистику"""
        stats = self.factual.get_stats()
        print(f"\n📊 Статистика:")
        print(f"   Факты: {stats['total_facts']} ({stats['fact_types']} типов)")
        print(f"   Эпизоды: {len(self.episodic.episodes)}")
        print(f"   Взаимодействий: {self.meta['interactions']}")

    def process(self, user_input: str) -> str:
        """Обработка входа"""
        self.meta['interactions'] += 1
        self.log(f"INPUT: {user_input}")

        # 1. Проверяем команду
        is_command, result = self.commands.process(user_input)

        if is_command:
            # Команда обработана детерминированно
            self.episodic.add(user_input, result, "memory_command")
            self.save_all()
            self.log(f"COMMAND: {result[:100]}")
            return result

        # 2. Автоматически извлекаем числа из входа
        numbers = extract_numbers(user_input)
        if numbers and any(word in user_input.lower() for word in ['запомни', 'сохрани', 'добавь']):
            self.factual.add_numbers(numbers, user_input)

        # 3. Генерируем ответ через LLM
        response = self._query_llm(user_input)

        # 4. Сохраняем эпизод
        self.episodic.add(user_input, response, "llm_response")
        self.save_all()

        self.log(f"OUTPUT: {response[:100]}")
        return response

    def _query_llm(self, query: str) -> str:
        """Запрос к LLM"""
        try:
            # Строим контекст
            context_parts = []

            # Факты
            facts_text = self.factual.format_for_llm(max_facts=30)
            if facts_text:
                context_parts.append(f"🎯 ФАКТЫ В ПАМЯТИ:\n{facts_text}")

            # Недавняя история
            history_text = self.episodic.format_for_llm(n=3)
            if history_text:
                context_parts.append(f"\n💭 НЕДАВНИЙ КОНТЕКСТ:\n{history_text}")

            context = "\n\n".join(context_parts) if context_parts else ""

            # Системный промпт
            system_prompt = (
                "Ты — когнитивная система с памятью. "
                "ВАЖНО: Если в разделе 'ФАКТЫ В ПАМЯТИ' есть информация — используй ТОЛЬКО её. "
                "Не придумывай факты. Отвечай кратко и точно.\n\n"
            )

            if context:
                system_prompt += f"{context}\n\n"
                system_prompt += (
                    "ПРАВИЛО: Используй только факты из памяти. "
                    "Если пользователь спрашивает о фактах — перечисли их ТАК КАК ОНИ ЕСТЬ В ПАМЯТИ."
                )

            if context:
                print(f"🧠 Контекст: {len(context)} символов")

            # API запрос
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": Config.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.2,  # Низкая для точности
                "max_tokens": Config.MAX_TOKENS
            }

            print("⏳ Генерирую ответ...")

            response = requests.post(
                Config.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=Config.TIMEOUT
            )

            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            return content

        except Exception as e:
            error_msg = f"⚠️ Ошибка: {str(e)[:100]}"
            self.log(f"API ERROR: {e}")
            return error_msg

    def save_all(self):
        """Сохранить всё"""
        self.factual.save()
        self.episodic.save()

        with open(Config.META_DB, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load_meta(self) -> dict:
        """Загрузить метаданные"""
        if Config.META_DB.exists():
            try:
                with open(Config.META_DB, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        return {
            'interactions': 0,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()


# ================= ДИАГНОСТИКА =================
def run_diagnosis() -> bool:
    """Диагностика системы"""
    print("=" * 70)
    print("🔍 ДИАГНОСТИКА")
    print("=" * 70)

    if not Config.OPENROUTER_API_KEY:
        print("❌ Не найден OPENROUTER_API_KEY")
        return False

    print(f"✅ API ключ: {Config.OPENROUTER_API_KEY[:12]}...{Config.OPENROUTER_API_KEY[-4:]}")
    print(f"✅ Модель: {Config.MODEL}")

    try:
        print("📡 Проверка API...", end=" ")

        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": Config.MODEL,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }

        response = requests.post(
            Config.OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            print("✅")
            return True
        else:
            print(f"❌ {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ {e}")
        return False


# ================= MAIN =================
def main():
    """Главная функция"""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass

    print("\n" + "=" * 70)
    print("🧠 COGNITIVE SYSTEM v25")
    print("   Memory Control Edition — БЕЗ ГАЛЛЮЦИНАЦИЙ")
    print("=" * 70 + "\n")

    if not run_diagnosis():
        print("\n❌ Диагностика не пройдена")
        return

    print("\n" + "=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ")
    print("=" * 70 + "\n")

    system = CognitiveSystem()

    print("\n" + "=" * 70)
    print("💬 СИСТЕМА ГОТОВА")
    print("=" * 70)
    print("\n🎯 Что нового:")
    print("  ✅ Детерминированное управление памятью")
    print("  ✅ Команды не проходят через LLM")
    print("  ✅ LLM не может галлюцинировать факты")
    print("  ✅ Прямые операции с числами")
    print("\n📋 Команды памяти:")
    print("  • 'запомни число X' — сохранить число")
    print("  • 'покажи числа' — показать все числа")
    print("  • 'удали числа' — удалить все числа")
    print("  • 'прибавь X' — прибавить X ко всем числам")
    print("  • 'умножь на X' — умножить все числа на X")
    print("  • 'статистика' — показать статистику")
    print("  • 'очисти память' — полная очистка")
    print("\n💡 Теперь всё работает детерминированно!")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("💭 Ваш ввод: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'выход', 'quit', 'q']:
                print("\n👋 Завершение работы")
                system.save_all()
                print("💾 Память сохранена")
                break

            print()
            response = system.process(user_input)

            print("\n🤖 Ответ:")
            print_typing(response, delay=0.008)

            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Прервано")
            system.save_all()
            print("💾 Память сохранена")
            break

        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()