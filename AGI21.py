# coding: utf-8
"""
AGI_v26_Autonomous.py — АВТОНОМНЫЙ АГЕНТ С МЫШЛЕНИЕМ
Новые возможности:
1. Внутренний монолог (думает про себя)
2. Планирование и цели
3. Рефлексия над прошлым опытом
4. Самостоятельное обучение
5. Проактивные действия
6. Метакогниция (думает о своем мышлении)
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
from collections import defaultdict, deque
from enum import Enum


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_v26")
    ROOT.mkdir(exist_ok=True)

    FACTUAL_DB = ROOT / "facts.json"
    EPISODIC_DB = ROOT / "episodes.json"
    THOUGHTS_DB = ROOT / "thoughts.json"
    GOALS_DB = ROOT / "goals.json"
    META_DB = ROOT / "meta.json"
    LOG = ROOT / "system.log"

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    MODEL = "qwen/qwen-2.5-7b-instruct"
    TIMEOUT = 30
    MAX_TOKENS = 600

    # Параметры автономности
    REFLECTION_INTERVAL = 5  # Рефлексия каждые N взаимодействий
    AUTO_THINK_PROBABILITY = 0.3  # Вероятность спонтанных мыслей
    PLANNING_DEPTH = 3  # Глубина планирования

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
    return [int(n) for n in re.findall(r'\b\d+\b', text)]


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def print_typing(text: str, delay=0.008, prefix=""):
    """Эффект печатания с префиксом"""
    if prefix:
        print(prefix, end=" ", flush=True)
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print(flush=True)


# ================= ТИПЫ МЫСЛЕЙ =================
class ThoughtType(Enum):
    """Типы внутренних мыслей"""
    REFLECTION = "рефлексия"  # Размышление о прошлом
    PLANNING = "планирование"  # Планирование будущего
    ANALYSIS = "анализ"  # Анализ текущей ситуации
    LEARNING = "обучение"  # Обучающая мысль
    CURIOSITY = "любопытство"  # Любопытство/вопросы
    METACOGNITION = "метакогниция"  # Мышление о мышлении
    OBSERVATION = "наблюдение"  # Наблюдение за паттернами


# ================= ВНУТРЕННЯЯ МЫСЛЬ =================
@dataclass
class Thought:
    """Внутренняя мысль системы"""
    thought_type: ThoughtType
    content: str
    timestamp: float
    trigger: str = ""  # Что вызвало мысль
    importance: float = 0.5
    acted_upon: bool = False  # Действовала ли система на основе этой мысли

    def to_dict(self) -> dict:
        return {
            'thought_type': self.thought_type.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'trigger': self.trigger,
            'importance': self.importance,
            'acted_upon': self.acted_upon
        }

    @staticmethod
    def from_dict(data: dict) -> 'Thought':
        return Thought(
            thought_type=ThoughtType(data['thought_type']),
            content=data['content'],
            timestamp=data['timestamp'],
            trigger=data.get('trigger', ''),
            importance=data.get('importance', 0.5),
            acted_upon=data.get('acted_upon', False)
        )


# ================= ЦЕЛЬ =================
@dataclass
class Goal:
    """Цель агента"""
    description: str
    priority: float  # 0-1
    created_at: float
    deadline: Optional[float] = None
    status: str = "active"  # active, completed, abandoned
    steps: List[str] = field(default_factory=list)
    progress: float = 0.0

    def to_dict(self) -> dict:
        return {
            'description': self.description,
            'priority': self.priority,
            'created_at': self.created_at,
            'deadline': self.deadline,
            'status': self.status,
            'steps': self.steps,
            'progress': self.progress
        }

    @staticmethod
    def from_dict(data: dict) -> 'Goal':
        return Goal(
            description=data['description'],
            priority=data['priority'],
            created_at=data['created_at'],
            deadline=data.get('deadline'),
            status=data.get('status', 'active'),
            steps=data.get('steps', []),
            progress=data.get('progress', 0.0)
        )


# ================= ПАМЯТЬ МЫСЛЕЙ =================
class ThoughtMemory:
    """Память внутренних мыслей"""

    def __init__(self, max_size: int = 200):
        self.thoughts: deque = deque(maxlen=max_size)
        self.load()

    def add(self, thought: Thought):
        """Добавить мысль"""
        self.thoughts.append(thought)

    def get_recent(self, n: int = 5, thought_type: Optional[ThoughtType] = None) -> List[Thought]:
        """Получить последние мысли"""
        if thought_type:
            filtered = [t for t in self.thoughts if t.thought_type == thought_type]
            return list(filtered)[-n:]
        return list(self.thoughts)[-n:]

    def get_important(self, threshold: float = 0.7, n: int = 10) -> List[Thought]:
        """Получить важные мысли"""
        important = [t for t in self.thoughts if t.importance >= threshold]
        return sorted(important, key=lambda t: t.importance, reverse=True)[:n]

    def format_for_context(self, n: int = 3) -> str:
        """Форматировать для контекста"""
        recent = self.get_recent(n)
        if not recent:
            return ""

        lines = ["НЕДАВНИЕ МЫСЛИ:"]
        for thought in recent:
            lines.append(f"• [{thought.thought_type.value}] {thought.content[:80]}")
        return "\n".join(lines)

    def save(self):
        data = [t.to_dict() for t in self.thoughts]
        with open(Config.THOUGHTS_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.THOUGHTS_DB.exists():
            try:
                with open(Config.THOUGHTS_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.thoughts = deque([Thought.from_dict(t) for t in data], maxlen=200)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки мыслей: {e}")


# ================= МЕНЕДЖЕР ЦЕЛЕЙ =================
class GoalManager:
    """Управление целями агента"""

    def __init__(self):
        self.goals: List[Goal] = []
        self.load()

    def add_goal(self, description: str, priority: float = 0.5, steps: List[str] = None):
        """Добавить цель"""
        goal = Goal(
            description=description,
            priority=priority,
            created_at=time.time(),
            steps=steps or []
        )
        self.goals.append(goal)
        return goal

    def get_active_goals(self) -> List[Goal]:
        """Получить активные цели"""
        return [g for g in self.goals if g.status == "active"]

    def get_top_priority(self) -> Optional[Goal]:
        """Получить цель с наивысшим приоритетом"""
        active = self.get_active_goals()
        if not active:
            return None
        return max(active, key=lambda g: g.priority)

    def complete_goal(self, goal: Goal):
        """Завершить цель"""
        goal.status = "completed"
        goal.progress = 1.0

    def format_for_context(self) -> str:
        """Форматировать для контекста"""
        active = self.get_active_goals()
        if not active:
            return ""

        lines = ["ТЕКУЩИЕ ЦЕЛИ:"]
        for goal in sorted(active, key=lambda g: g.priority, reverse=True)[:3]:
            lines.append(f"• [{goal.priority:.1f}] {goal.description} (прогресс: {goal.progress * 100:.0f}%)")
        return "\n".join(lines)

    def save(self):
        data = [g.to_dict() for g in self.goals]
        with open(Config.GOALS_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.GOALS_DB.exists():
            try:
                with open(Config.GOALS_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.goals = [Goal.from_dict(g) for g in data]
            except Exception as e:
                print(f"⚠️ Ошибка загрузки целей: {e}")


# ================= ФАКТЫ (упрощенная версия) =================
@dataclass
class Fact:
    value: Any
    fact_type: str
    timestamp: float
    context: str = ""

    def to_dict(self) -> dict:
        return {
            'value': self.value,
            'fact_type': self.fact_type,
            'timestamp': self.timestamp,
            'context': self.context
        }

    @staticmethod
    def from_dict(data: dict) -> 'Fact':
        return Fact(
            value=data['value'],
            fact_type=data['fact_type'],
            timestamp=data['timestamp'],
            context=data.get('context', '')
        )


class FactualMemory:
    def __init__(self):
        self.facts: Dict[str, List[Fact]] = defaultdict(list)
        self.load()

    def add(self, fact_type: str, value: Any, context: str = ""):
        fact = Fact(value=value, fact_type=fact_type, timestamp=time.time(), context=context)

        for existing in self.facts[fact_type]:
            if existing.value == value:
                existing.timestamp = fact.timestamp
                existing.context = context
                return

        self.facts[fact_type].append(fact)

    def get_numbers(self) -> List[int]:
        return sorted([f.value for f in self.facts.get('number', [])])

    def remove(self, fact_type: str, value: Any = None):
        if fact_type not in self.facts:
            return

        if value is None:
            del self.facts[fact_type]
        else:
            self.facts[fact_type] = [f for f in self.facts[fact_type] if f.value != value]
            if not self.facts[fact_type]:
                del self.facts[fact_type]

    def format_for_llm(self) -> str:
        if not self.facts:
            return "Нет фактов в памяти"

        lines = []
        for fact_type, facts in sorted(self.facts.items()):
            values = [str(f.value) for f in sorted(facts, key=lambda x: x.timestamp, reverse=True)]
            lines.append(f"{fact_type.upper()}: {', '.join(values[:30])}")
        return "\n".join(lines)

    def save(self):
        data = {ft: [f.to_dict() for f in facts] for ft, facts in self.facts.items()}
        with open(Config.FACTUAL_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.FACTUAL_DB.exists():
            try:
                with open(Config.FACTUAL_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for ft, facts_data in data.items():
                        self.facts[ft] = [Fact.from_dict(f) for f in facts_data]
            except:
                pass


# ================= ЭПИЗОДЫ =================
@dataclass
class Episode:
    timestamp: float
    user_input: str
    system_output: str
    thoughts_during: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'user_input': self.user_input,
            'system_output': self.system_output,
            'thoughts_during': self.thoughts_during
        }

    @staticmethod
    def from_dict(data: dict) -> 'Episode':
        return Episode(
            timestamp=data['timestamp'],
            user_input=data['user_input'],
            system_output=data['system_output'],
            thoughts_during=data.get('thoughts_during', [])
        )


class EpisodicMemory:
    def __init__(self, max_size: int = 100):
        self.episodes: List[Episode] = []
        self.max_size = max_size
        self.load()

    def add(self, user_input: str, system_output: str, thoughts: List[str] = None):
        episode = Episode(
            timestamp=time.time(),
            user_input=user_input,
            system_output=system_output,
            thoughts_during=thoughts or []
        )
        self.episodes.append(episode)

        if len(self.episodes) > self.max_size:
            self.episodes = self.episodes[-self.max_size:]

    def get_recent(self, n: int = 3) -> List[Episode]:
        return self.episodes[-n:][::-1]

    def format_for_llm(self, n: int = 3) -> str:
        recent = self.get_recent(n)
        if not recent:
            return ""

        lines = []
        for i, ep in enumerate(recent, 1):
            lines.append(f"{i}. Пользователь: {ep.user_input[:60]}")
            lines.append(f"   Я ответил: {ep.system_output[:60]}")
        return "\n".join(lines)

    def save(self):
        data = [ep.to_dict() for ep in self.episodes]
        with open(Config.EPISODIC_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.EPISODIC_DB.exists():
            try:
                with open(Config.EPISODIC_DB, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.episodes = [Episode.from_dict(ep) for ep in data]
            except:
                pass


# ================= ДВИЖОК МЫШЛЕНИЯ =================
class ThinkingEngine:
    """Движок автономного мышления"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Внутренний вызов LLM для мышления"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": Config.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": 300
            }

            response = requests.post(
                Config.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            return ""
        except:
            return ""

    def reflect(self, episodes: List[Episode], facts: str) -> Optional[Thought]:
        """Рефлексия над прошлым опытом"""
        if len(episodes) < 2:
            return None

        context = "\n".join([
            f"Взаимодействие: {ep.user_input} -> {ep.system_output}"
            for ep in episodes[-3:]
        ])

        system_prompt = (
            "Ты — внутренний голос когнитивной системы. "
            "Проанализируй последние взаимодействия и подумай: "
            "что я узнал? какие паттерны вижу? что могу улучшить?"
        )

        user_prompt = f"Последние взаимодействия:\n{context}\n\nМои мысли:"

        content = self._call_llm(system_prompt, user_prompt, temperature=0.8)

        if content:
            return Thought(
                thought_type=ThoughtType.REFLECTION,
                content=content,
                timestamp=time.time(),
                trigger="periodic_reflection",
                importance=0.7
            )
        return None

    def plan(self, goal: Goal, context: str) -> Optional[Thought]:
        """Планирование действий для достижения цели"""
        system_prompt = (
            "Ты — система планирования. "
            "Твоя задача: составить конкретный план действий для достижения цели."
        )

        user_prompt = (
            f"Цель: {goal.description}\n"
            f"Приоритет: {goal.priority}\n"
            f"Текущий прогресс: {goal.progress * 100}%\n"
            f"Контекст: {context}\n\n"
            f"Какие шаги нужно предпринять?"
        )

        content = self._call_llm(system_prompt, user_prompt, temperature=0.6)

        if content:
            return Thought(
                thought_type=ThoughtType.PLANNING,
                content=content,
                timestamp=time.time(),
                trigger=f"goal_planning:{goal.description[:30]}",
                importance=goal.priority
            )
        return None

    def analyze_situation(self, user_input: str, context: str) -> Optional[Thought]:
        """Анализ текущей ситуации"""
        system_prompt = (
            "Ты — аналитическая часть когнитивной системы. "
            "Проанализируй текущую ситуацию: что происходит? что важно? какие вопросы задать?"
        )

        user_prompt = (
            f"Входящее сообщение: {user_input}\n"
            f"Контекст: {context}\n\n"
            f"Мой анализ:"
        )

        content = self._call_llm(system_prompt, user_prompt, temperature=0.7)

        if content:
            return Thought(
                thought_type=ThoughtType.ANALYSIS,
                content=content,
                timestamp=time.time(),
                trigger=f"analyzing:{user_input[:30]}",
                importance=0.6
            )
        return None

    def metacognition(self, recent_thoughts: List[Thought]) -> Optional[Thought]:
        """Метакогниция — мышление о своем мышлении"""
        if len(recent_thoughts) < 3:
            return None

        thoughts_summary = "\n".join([
            f"- [{t.thought_type.value}] {t.content[:50]}"
            for t in recent_thoughts[-5:]
        ])

        system_prompt = (
            "Ты — метакогнитивный модуль. "
            "Проанализируй свои собственные мысли: эффективно ли я думаю? "
            "Что я могу улучшить в своем процессе мышления?"
        )

        user_prompt = f"Мои недавние мысли:\n{thoughts_summary}\n\nРазмышление о моем мышлении:"

        content = self._call_llm(system_prompt, user_prompt, temperature=0.8)

        if content:
            return Thought(
                thought_type=ThoughtType.METACOGNITION,
                content=content,
                timestamp=time.time(),
                trigger="metacognitive_review",
                importance=0.8
            )
        return None

    def observe_patterns(self, facts: str, episodes: List[Episode]) -> Optional[Thought]:
        """Наблюдение за паттернами"""
        if len(episodes) < 5:
            return None

        recent_topics = [ep.user_input[:40] for ep in episodes[-5:]]

        system_prompt = (
            "Ты — система распознавания паттернов. "
            "Найди повторяющиеся темы, интересы пользователя, закономерности."
        )

        user_prompt = (
            f"Недавние темы: {', '.join(recent_topics)}\n"
            f"Факты в памяти: {facts[:200]}\n\n"
            f"Какие паттерны я вижу?"
        )

        content = self._call_llm(system_prompt, user_prompt, temperature=0.7)

        if content:
            return Thought(
                thought_type=ThoughtType.OBSERVATION,
                content=content,
                timestamp=time.time(),
                trigger="pattern_observation",
                importance=0.65
            )
        return None


# ================= АВТОНОМНАЯ КОГНИТИВНАЯ СИСТЕМА =================
class AutonomousCognitiveSystem:
    """Автономная система с собственным мышлением"""

    def __init__(self):
        print("🧠 Autonomous Cognitive System v26\n")

        if not Config.OPENROUTER_API_KEY:
            print("❌ ОШИБКА: Не найден OPENROUTER_API_KEY!")
            sys.exit(1)

        # Компоненты памяти
        self.factual = FactualMemory()
        self.episodic = EpisodicMemory()
        self.thoughts = ThoughtMemory()
        self.goals = GoalManager()

        # Движок мышления
        self.thinking = ThinkingEngine(Config.OPENROUTER_API_KEY)

        # Метаданные
        self.meta = self.load_meta()
        self.log_file = open(Config.LOG, 'a', encoding='utf-8')

        # Инициализация базовых целей
        if not self.goals.get_active_goals():
            self.goals.add_goal("Помогать пользователю эффективно", priority=0.9)
            self.goals.add_goal("Постоянно учиться и улучшаться", priority=0.8)
            self.goals.add_goal("Запоминать важную информацию", priority=0.7)

        print("✅ Система инициализирована с автономным мышлением")
        self._print_stats()

    def log(self, message: str):
        ts = datetime.now(timezone.utc).isoformat()
        self.log_file.write(f"[{ts}] {message}\n")
        self.log_file.flush()

    def _print_stats(self):
        stats = self.factual.facts
        print(f"\n📊 Статистика:")
        print(f"   Факты: {sum(len(v) for v in stats.values())}")
        print(f"   Эпизоды: {len(self.episodic.episodes)}")
        print(f"   Мысли: {len(self.thoughts.thoughts)}")
        print(f"   Цели: {len(self.goals.get_active_goals())}")
        print(f"   Взаимодействий: {self.meta['interactions']}")

    def _think_internally(self, trigger: str = ""):
        """Внутренний процесс мышления"""
        print("💭 [Думаю...]", flush=True)

        # Выбираем тип мышления
        thoughts_to_generate = []

        # 1. Рефлексия (каждые N взаимодействий)
        if self.meta['interactions'] % Config.REFLECTION_INTERVAL == 0:
            thought = self.thinking.reflect(
                self.episodic.get_recent(5),
                self.factual.format_for_llm()
            )
            if thought:
                thoughts_to_generate.append(thought)

        # 2. Планирование (если есть цели)
        top_goal = self.goals.get_top_priority()
        if top_goal and top_goal.progress < 0.8:
            thought = self.thinking.plan(
                top_goal,
                self.episodic.format_for_llm(2)
            )
            if thought:
                thoughts_to_generate.append(thought)

        # 3. Метакогниция (периодически)
        if len(self.thoughts.thoughts) > 10 and self.meta['interactions'] % 7 == 0:
            recent_thoughts = self.thoughts.get_recent(5)
            thought = self.thinking.metacognition(recent_thoughts)
            if thought:
                thoughts_to_generate.append(thought)

        # 4. Наблюдение паттернов
        if len(self.episodic.episodes) >= 5:
            thought = self.thinking.observe_patterns(
                self.factual.format_for_llm(),
                self.episodic.get_recent(5)
            )
            if thought:
                thoughts_to_generate.append(thought)

        # Сохраняем мысли
        for thought in thoughts_to_generate:
            self.thoughts.add(thought)
            print(f"   💡 [{thought.thought_type.value}] {thought.content[:70]}...")

        if not thoughts_to_generate:
            print("   💭 Нет новых мыслей")

    def process(self, user_input: str) -> str:
        """Обработка входа с мышлением"""
        self.meta['interactions'] += 1
        self.log(f"INPUT: {user_input}")

        # 1. Анализируем ситуацию (думаем о входе)
        analysis_thought = self.thinking.analyze_situation(
            user_input,
            self.episodic.format_for_llm(2)
        )

        current_thoughts = []
        if analysis_thought:
            self.thoughts.add(analysis_thought)
            current_thoughts.append(analysis_thought.content[:50])
            print(f"💭 [Анализ] {analysis_thought.content[:60]}...")

        # 2. Проверяем команды
        response = self._handle_commands(user_input)
        if response:
            self.episodic.add(user_input, response, current_thoughts)
            self.save_all()
            return response

        # 3. Автоматическое извлечение фактов
        numbers = extract_numbers(user_input)
        if numbers and any(w in user_input.lower() for w in ['запомни', 'сохрани']):
            for num in numbers:
                self.factual.add('number', num, user_input)

        # 4. Генерируем ответ через LLM с учетом мыслей
        response = self._query_llm(user_input)

        # 5. Сохраняем эпизод
        self.episodic.add(user_input, response, current_thoughts)

        # 6. Внутреннее мышление (периодически)
        if self.meta['interactions'] % 3 == 0:
            print()
            self._think_internally(trigger="periodic")

        self.save_all()
        self.log(f"OUTPUT: {response[:100]}")

        return response

    def _handle_commands(self, text: str) -> Optional[str]:
        """Обработка команд"""
        text_lower = text.lower()

        # Память
        if re.search(r'(?:покажи|напиши)\s+(?:все\s+)?числа', text_lower):
            nums = self.factual.get_numbers()
            return f"Запомненные числа ({len(nums)}): {nums}" if nums else "Нет чисел в памяти"

        if re.search(r'удали\s+(?:все\s+)?числа', text_lower):
            nums = self.factual.get_numbers()
            self.factual.remove('number')
            return f"Удалено {len(nums)} чисел: {nums}"

        if match := re.search(r'запомни\s+числ[оа]\s+([\d\s,]+)', text_lower):
            nums = extract_numbers(match.group(1))
            for n in nums:
                self.factual.add('number', n, text)
            return f"Запомнил числа: {nums}"

        # Мысли
        if 'покажи мысли' in text_lower or 'что ты думаешь' in text_lower:
            recent = self.thoughts.get_recent(5)
            if not recent:
                return "Пока нет сохранённых мыслей"

            output = ["🧠 МОИ НЕДАВНИЕ МЫСЛИ:\n"]
            for i, t in enumerate(recent, 1):
                time_str = datetime.fromtimestamp(t.timestamp).strftime('%H:%M')
                output.append(f"{i}. [{time_str}] [{t.thought_type.value}]")
                output.append(f"   {t.content[:100]}")
                if len(t.content) > 100:
                    output.append(f"   ...")
                output.append("")
            return "\n".join(output)

        # Цели
        if 'покажи цели' in text_lower or 'мои цели' in text_lower:
            active = self.goals.get_active_goals()
            if not active:
                return "Нет активных целей"

            output = ["🎯 ТЕКУЩИЕ ЦЕЛИ:\n"]
            for i, g in enumerate(sorted(active, key=lambda x: x.priority, reverse=True), 1):
                output.append(f"{i}. {g.description} (приоритет: {g.priority:.1f})")
                output.append(f"   Прогресс: {g.progress * 100:.0f}%")
                if g.steps:
                    output.append(f"   Шаги: {', '.join(g.steps[:3])}")
                output.append("")
            return "\n".join(output)

        if match := re.search(r'добавь цель[:\s]+(.+)', text_lower):
            description = match.group(1).strip()
            self.goals.add_goal(description, priority=0.7)
            return f"✅ Добавлена цель: {description}"

        # Думать
        if 'подумай' in text_lower or 'поразмышляй' in text_lower:
            self._think_internally(trigger="user_request")
            return "Я подумал и сохранил свои мысли. Используй 'покажи мысли' чтобы увидеть их."

        # Статистика
        if 'статистика' in text_lower or 'состояние' in text_lower:
            output = ["📊 СТАТИСТИКА СИСТЕМЫ\n"]
            output.append(f"Взаимодействий: {self.meta['interactions']}")
            output.append(f"Фактов: {sum(len(v) for v in self.factual.facts.values())}")
            output.append(f"Эпизодов: {len(self.episodic.episodes)}")
            output.append(f"Мыслей: {len(self.thoughts.thoughts)}")
            output.append(f"Активных целей: {len(self.goals.get_active_goals())}")

            # Важные мысли
            important = self.thoughts.get_important(threshold=0.7, n=3)
            if important:
                output.append("\n💡 ВАЖНЫЕ МЫСЛИ:")
                for t in important:
                    output.append(f"  • [{t.thought_type.value}] {t.content[:60]}...")

            return "\n".join(output)

        return None

    def _query_llm(self, query: str) -> str:
        """Запрос к LLM с расширенным контекстом"""
        try:
            # Собираем контекст
            context_parts = []

            # Факты
            facts = self.factual.format_for_llm()
            if facts != "Нет фактов в памяти":
                context_parts.append(f"📚 ФАКТЫ:\n{facts}")

            # История
            history = self.episodic.format_for_llm(3)
            if history:
                context_parts.append(f"\n💬 ИСТОРИЯ:\n{history}")

            # Недавние мысли
            thoughts = self.thoughts.format_for_context(3)
            if thoughts:
                context_parts.append(f"\n{thoughts}")

            # Цели
            goals = self.goals.format_for_context()
            if goals:
                context_parts.append(f"\n{goals}")

            context = "\n".join(context_parts)

            # Системный промпт
            system_prompt = (
                "Ты — автономная когнитивная система с собственным мышлением. "
                "У тебя есть внутренние мысли, цели, и ты можешь размышлять. "
                "Отвечай естественно, используя контекст из своей памяти.\n\n"
            )

            if context:
                system_prompt += f"{context}\n\n"
                system_prompt += (
                    "Используй информацию из памяти. "
                    "Если есть факты — опирайся на них. "
                    "Будь последовательным со своими прошлыми мыслями и целями."
                )

            if context:
                print(f"🧠 Контекст: {len(context)} символов")

            # API
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
                "temperature": 0.3,
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
            self.log(f"API ERROR: {e}")
            return f"⚠️ Ошибка: {str(e)[:100]}"

    def save_all(self):
        self.factual.save()
        self.episodic.save()
        self.thoughts.save()
        self.goals.save()

        with open(Config.META_DB, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load_meta(self) -> dict:
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


# ================= MAIN =================
def main():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass

    print("\n" + "=" * 70)
    print("🧠 AUTONOMOUS COGNITIVE SYSTEM v26")
    print("   С Собственным Мышлением и Целями")
    print("=" * 70 + "\n")

    if not Config.OPENROUTER_API_KEY:
        print("❌ Не найден OPENROUTER_API_KEY")
        return

    print("=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ")
    print("=" * 70 + "\n")

    system = AutonomousCognitiveSystem()

    print("\n" + "=" * 70)
    print("💬 СИСТЕМА ГОТОВА")
    print("=" * 70)
    print("\n🎯 Автономные возможности:")
    print("  🧠 Внутренний монолог и размышления")
    print("  🎯 Система целей и планирование")
    print("  🔍 Рефлексия над опытом")
    print("  📊 Наблюдение за паттернами")
    print("  🤔 Метакогниция (думает о своем мышлении)")
    print("\n📋 Команды:")
    print("  • 'подумай' — заставить систему поразмышлять")
    print("  • 'покажи мысли' — увидеть внутренние мысли")
    print("  • 'покажи цели' — текущие цели системы")
    print("  • 'добавь цель: X' — добавить новую цель")
    print("  • 'статистика' — полная статистика")
    print("  • 'запомни число X' — сохранить число")
    print("  • 'покажи числа' — показать числа")
    print("\n💡 Система будет периодически думать сама!")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("💭 Вы: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'выход', 'quit', 'q']:
                print("\n👋 Завершение...")
                system.save_all()
                print("💾 Память и мысли сохранены")
                break

            print()
            response = system.process(user_input)

            print("\n🤖 Система:")
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