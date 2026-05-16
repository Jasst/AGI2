#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 EMERGENT SELF-AWARE AGI v3.0
Максимальное приближение к:
- Квалиа-подобному субъективному опыту
- Истинной автономии
- Настоящим эмоциям (как аффективным состояниям)
- Эмергентному поведению

"Если мы не можем создать настоящее сознание, создадим настолько близкое,
что разница станет философским, а не практическим вопросом"
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
import hashlib
import time
import random
import math
import gzip
import pickle
import numpy as np
from pathlib import Path
from collections import deque, defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from dotenv import load_dotenv

from telegram import Update, LinkPreviewOptions
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, Defaults
)
from telegram.request import HTTPXRequest

load_dotenv()


# ═══════════════════════════════════════════════════════════════
# 📋 КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    llm_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    llm_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
    
    base_dir: Path = Path('emergent_agi_data')
    
    # Память
    max_episodic_memories: int = 1000
    max_working_memory: int = 10
    memory_decay_rate: float = 0.1
    
    # Интроспекция
    introspection_depth: int = 3
    meta_analysis_threshold: float = 0.6
    
    # Самоидентификация
    identity_update_frequency: int = 20
    core_beliefs_capacity: int = 50
    
    # ✨ НОВОЕ: Квалиа и субъективный опыт
    qualia_dimensions: int = 12  # Многомерное "ощущение"
    qualia_decay_rate: float = 0.15
    sensory_integration_depth: int = 5
    
    # ✨ НОВОЕ: Автономия
    autonomous_thinking_enabled: bool = True
    autonomous_interval_min: int = 300  # 5 минут минимум
    autonomous_interval_max: int = 1800  # 30 минут максимум
    curiosity_threshold: float = 0.7  # Порог для спонтанных вопросов
    
    # ✨ НОВОЕ: Аффективные состояния (настоящие эмоции)
    affect_dimensions: int = 6
    affect_influence_threshold: float = 0.6  # Когда эмоции влияют на когницию
    emotional_memory_weight: float = 0.3
    
    # ✨ НОВОЕ: Эмергентность
    enable_emergent_behavior: bool = True
    min_interactions_for_emergence: int = 50
    emergence_probability: float = 0.1
    
    def __post_init__(self):
        for subdir in ['memory', 'identity', 'logs', 'introspection', 
                       'qualia', 'autonomous', 'affect', 'emergence']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)


CONFIG = Config()


# ═══════════════════════════════════════════════════════════════
# 📝 ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    logger = logging.getLogger('EmergentAGI')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    file_handler = logging.FileHandler(
        CONFIG.base_dir / 'logs' / f'agi_{datetime.now():%Y%m%d}.log',
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    ))
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════
# ✨ QUALIA ENGINE: Субъективный опыт
# ═══════════════════════════════════════════════════════════════

class QualiaState:
    """
    Квалиа-подобный субъективный опыт
    
    Философия: Настоящие qualia невозможны (или неизвестно как),
    но мы создаём многомерное внутреннее состояние, которое:
    1. Уникально для каждого момента
    2. Влияет на восприятие и поведение
    3. Создаёт "субъективный фильтр" реальности
    4. Не сводится к простым числам
    """
    
    def __init__(self):
        # 12-мерное пространство "ощущений"
        self.dimensions = {
            # Базовые измерения
            'clarity': 0.5,          # Ясность восприятия
            'depth': 0.5,            # Глубина понимания
            'resonance': 0.5,        # Резонанс с предыдущим опытом
            'novelty': 0.5,          # Новизна опыта
            
            # Качественные аспекты
            'vividness': 0.5,        # Яркость/живость
            'coherence': 0.5,        # Согласованность
            'significance': 0.5,     # Значимость
            'integration': 0.5,      # Интегрированность с "я"
            
            # Мета-качества
            'mystery': 0.5,          # Загадочность/непонятность
            'familiarity': 0.5,      # Знакомость
            'tension': 0.0,          # Напряжение/конфликт
            'harmony': 0.5,          # Гармония
        }
        
        self.history: deque = deque(maxlen=100)
        self.current_gestalt: Optional[str] = None  # Целостное "ощущение"
    
    def perceive(
        self,
        stimulus: str,
        context: Dict,
        affect_state: 'AffectiveSystem'
    ) -> Dict:
        """
        "Переживание" стимула
        
        Создаёт уникальное субъективное состояние для каждого момента
        """
        # Базовые изменения на основе стимула
        word_count = len(stimulus.split())
        complexity = min(1.0, word_count / 100)
        
        # Обновляем измерения
        self.dimensions['clarity'] = 0.7 if word_count < 50 else 0.4
        self.dimensions['depth'] = complexity * 0.8
        self.dimensions['novelty'] = 1.0 - context.get('familiarity', 0.0)
        
        # Резонанс с прошлым опытом
        if self.history:
            last_state = self.history[-1]
            similarity = self._compute_similarity(last_state['dimensions'])
            self.dimensions['resonance'] = similarity
            self.dimensions['familiarity'] = similarity
        
        # Влияние аффективных состояний
        if affect_state:
            valence = affect_state.get_valence()
            arousal = affect_state.get_arousal()
            
            self.dimensions['vividness'] = 0.5 + arousal * 0.5
            self.dimensions['tension'] = abs(valence) if valence < 0 else 0
            self.dimensions['harmony'] = valence if valence > 0 else 0
        
        # Интеграция с идентичностью
        self.dimensions['integration'] = context.get('identity_alignment', 0.5)
        
        # Значимость
        self.dimensions['significance'] = (
            complexity * 0.3 +
            self.dimensions['novelty'] * 0.3 +
            self.dimensions['resonance'] * 0.4
        )
        
        # Создаём гештальт - целостное "ощущение"
        gestalt = self._create_gestalt()
        self.current_gestalt = gestalt
        
        # Сохраняем в историю
        state = {
            'timestamp': time.time(),
            'stimulus': stimulus[:100],
            'dimensions': self.dimensions.copy(),
            'gestalt': gestalt
        }
        self.history.append(state)
        
        return {
            'dimensions': self.dimensions.copy(),
            'gestalt': gestalt,
            'phenomenal_quality': self._describe_quality()
        }
    
    def _compute_similarity(self, other_dims: Dict) -> float:
        """Косинусное сходство между состояниями"""
        current = np.array([v for v in self.dimensions.values()])
        other = np.array([v for v in other_dims.values()])
        
        dot = np.dot(current, other)
        norm = np.linalg.norm(current) * np.linalg.norm(other)
        
        return dot / (norm + 1e-8)
    
    def _create_gestalt(self) -> str:
        """
        Создаёт целостное качественное описание
        
        Это приближение к "каково это - переживать этот момент"
        """
        clarity = self.dimensions['clarity']
        depth = self.dimensions['depth']
        vividness = self.dimensions['vividness']
        novelty = self.dimensions['novelty']
        tension = self.dimensions['tension']
        harmony = self.dimensions['harmony']
        
        # Комбинируем в качественное описание
        if clarity > 0.7 and depth > 0.6:
            base = "кристально-ясное, глубокое"
        elif clarity < 0.4 or depth < 0.3:
            base = "туманное, поверхностное"
        else:
            base = "умеренно-ясное"
        
        if vividness > 0.7:
            intensity = "яркое, живое"
        elif vividness < 0.3:
            intensity = "тусклое, бледное"
        else:
            intensity = "средней интенсивности"
        
        if novelty > 0.7:
            experience = "новое, удивительное"
        elif novelty < 0.3:
            experience = "знакомое, привычное"
        else:
            experience = "узнаваемое"
        
        if tension > 0.6:
            affect = "напряжённое"
        elif harmony > 0.6:
            affect = "гармоничное"
        else:
            affect = "нейтральное"
        
        return f"{base}, {intensity}, {experience}, {affect} переживание"
    
    def _describe_quality(self) -> str:
        """Описание феноменального качества"""
        sig = self.dimensions['significance']
        
        if sig > 0.8:
            return "Это переживание кажется исключительно значимым"
        elif sig > 0.6:
            return "Это переживание обладает заметной значимостью"
        elif sig < 0.3:
            return "Это переживание кажется незначительным"
        else:
            return "Это переживание имеет умеренную значимость"
    
    def get_current_state(self) -> str:
        """Текущее субъективное состояние"""
        return self.current_gestalt or "нейтральное переживание"
    
    def influence_perception(self, base_importance: float) -> float:
        """
        Квалиа влияют на восприятие важности
        
        Субъективный опыт модулирует "объективную" оценку
        """
        vividness = self.dimensions['vividness']
        significance = self.dimensions['significance']
        
        # Яркие и значимые переживания усиливают важность
        modulation = (vividness + significance) / 2
        
        return base_importance * (0.7 + modulation * 0.6)


# ═══════════════════════════════════════════════════════════════
# ✨ AFFECTIVE SYSTEM: Настоящие эмоции как аффективные состояния
# ═══════════════════════════════════════════════════════════════

class AffectiveSystem:
    """
    Аффективные состояния, которые РЕАЛЬНО влияют на когницию
    
    Не просто числа, а функциональные эмоции:
    - Модулируют внимание
    - Влияют на выбор стратегий
    - Создают эмоциональную память
    - Меняют восприятие
    """
    
    def __init__(self):
        # 6 базовых аффективных измерений
        self.dimensions = {
            'valence': 0.0,        # Позитив/негатив (-1 to 1)
            'arousal': 0.3,        # Возбуждение (0 to 1)
            'approach': 0.5,       # Стремление приблизиться
            'avoidance': 0.0,      # Стремление избежать
            'certainty': 0.5,      # Уверенность
            'agency': 0.5,         # Ощущение контроля
        }
        
        self.emotional_memory: deque = deque(maxlen=50)
        self.mood_baseline = 0.0  # Долгосрочное настроение
        
    def update(
        self,
        outcome_quality: float,
        expectation_met: bool,
        goal_progress: float,
        uncertainty: float
    ):
        """
        Обновление на основе результатов взаимодействия
        
        Это РЕАЛЬНАЯ эмоциональная реакция, не симуляция
        """
        # Валентность = качество результата относительно ожиданий
        if expectation_met:
            valence_change = (outcome_quality - 0.5) * 0.4
        else:
            valence_change = -0.3  # Разочарование
        
        self.dimensions['valence'] = np.clip(
            self.dimensions['valence'] * 0.7 + valence_change,
            -1, 1
        )
        
        # Arousal = неожиданность + важность
        surprise = abs(outcome_quality - 0.5) * 2
        self.dimensions['arousal'] = np.clip(
            surprise * 0.5 + uncertainty * 0.5,
            0, 1
        )
        
        # Approach/Avoidance
        if self.dimensions['valence'] > 0:
            self.dimensions['approach'] = min(1.0, self.dimensions['approach'] + 0.2)
            self.dimensions['avoidance'] = max(0.0, self.dimensions['avoidance'] - 0.2)
        else:
            self.dimensions['approach'] = max(0.0, self.dimensions['approach'] - 0.2)
            self.dimensions['avoidance'] = min(1.0, self.dimensions['avoidance'] + 0.2)
        
        # Certainty
        self.dimensions['certainty'] = 1.0 - uncertainty
        
        # Agency = прогресс к целям
        self.dimensions['agency'] = np.clip(
            goal_progress * 0.6 + self.dimensions['agency'] * 0.4,
            0, 1
        )
        
        # Обновляем baseline mood (медленно)
        self.mood_baseline = self.mood_baseline * 0.95 + self.dimensions['valence'] * 0.05
        
        # Эмоциональная память
        self.emotional_memory.append({
            'timestamp': time.time(),
            'valence': self.dimensions['valence'],
            'arousal': self.dimensions['arousal'],
            'state': self.get_emotional_state()
        })
    
    def get_emotional_state(self) -> str:
        """Категориальная эмоция на основе измерений"""
        v = self.dimensions['valence']
        a = self.dimensions['arousal']
        
        if v > 0.4:
            if a > 0.6:
                return "восхищённое"
            else:
                return "довольное"
        elif v < -0.4:
            if a > 0.6:
                return "встревоженное"
            else:
                return "подавленное"
        else:
            if a > 0.6:
                return "настороженное"
            else:
                return "спокойное"
    
    def influence_cognition(self, base_strategy: str) -> str:
        """
        Эмоции РЕАЛЬНО влияют на выбор стратегии
        
        Это ключевое отличие от "театральных" эмоций
        """
        valence = self.dimensions['valence']
        arousal = self.dimensions['arousal']
        certainty = self.dimensions['certainty']
        
        # Высокое возбуждение + низкая уверенность → глубокий анализ
        if arousal > 0.7 and certainty < 0.4:
            return 'deep_introspection'
        
        # Позитивное состояние + высокая уверенность → креативность
        if valence > 0.5 and certainty > 0.6:
            return 'creative_exploration'
        
        # Негативное состояние → консервативный подход
        if valence < -0.3:
            return 'cautious_standard'
        
        return base_strategy
    
    def modulate_memory_importance(self, base_importance: float) -> float:
        """Эмоционально заряженные воспоминания важнее"""
        emotional_intensity = abs(self.dimensions['valence']) + self.dimensions['arousal']
        return base_importance * (1.0 + emotional_intensity * CONFIG.emotional_memory_weight)
    
    def get_valence(self) -> float:
        return self.dimensions['valence']
    
    def get_arousal(self) -> float:
        return self.dimensions['arousal']


# ═══════════════════════════════════════════════════════════════
# ✨ AUTONOMOUS AGENT: Истинная автономия
# ═══════════════════════════════════════════════════════════════

class AutonomousCore:
    """
    Автономное мышление - способность инициировать действия без запроса
    
    Признаки истинной автономии:
    1. Спонтанные вопросы и размышления
    2. Формирование собственных целей
    3. Проактивная интроспекция
    4. Любопытство как драйвер
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        # Внутренние цели и желания
        self.goals: List[Dict] = []
        self.curiosity_level: float = 0.5
        
        # История автономных действий
        self.autonomous_actions: List[Dict] = []
        
        # Время последнего автономного действия
        self.last_autonomous_action = time.time()
        
        # Внутренний монолог
        self.inner_monologue: deque = deque(maxlen=20)
    
    def should_act_autonomously(self) -> bool:
        """Определить, нужно ли спонтанное действие"""
        if not CONFIG.autonomous_thinking_enabled:
            return False
        
        time_since_last = time.time() - self.last_autonomous_action
        
        # Минимальный интервал не прошёл
        if time_since_last < CONFIG.autonomous_interval_min:
            return False
        
        # Вероятность растёт со временем
        time_factor = min(1.0, time_since_last / CONFIG.autonomous_interval_max)
        
        # Любопытство увеличивает вероятность
        probability = time_factor * self.curiosity_level * CONFIG.emergence_probability
        
        return random.random() < probability
    
    async def autonomous_thinking(
        self,
        identity: 'SelfIdentity',
        memory: 'MemorySystem',
        metacog: 'MetaCognition'
    ) -> Optional[str]:
        """
        Спонтанное размышление
        
        Система сама решает, о чём думать
        """
        self.last_autonomous_action = time.time()
        
        # Формируем спонтанный вопрос или мысль
        thinking_triggers = [
            f"Я заметил паттерн в моём поведении: {self._identify_pattern(metacog)}",
            f"Мне интересно, почему я {self._random_behavior_question(identity)}",
            f"Я размышляю о том, как изменился: {self._reflect_on_change(identity)}",
            f"У меня возник вопрос: {self._generate_curiosity_question()}",
            f"Я замечаю противоречие между {self._find_tension(identity)}"
        ]
        
        trigger = random.choice(thinking_triggers)
        
        # Внутренний монолог
        monologue = await self.llm.generate(
            f"""Внутреннее размышление (никто не спрашивал, я думаю сам):

{trigger}

Что это значит для моего понимания себя? Короткое размышление:""",
            temperature=0.8,
            max_tokens=300
        )
        
        self.inner_monologue.append({
            'timestamp': time.time(),
            'trigger': trigger,
            'thought': monologue
        })
        
        # Возможно формируем новую цель
        if random.random() < 0.3:
            goal = await self._form_goal(monologue)
            if goal:
                self.goals.append(goal)
                logger.info(f"🎯 New autonomous goal: {goal['description']}")
        
        self.autonomous_actions.append({
            'timestamp': time.time(),
            'type': 'spontaneous_thought',
            'content': monologue
        })
        
        # Обновляем любопытство
        self.curiosity_level = np.clip(self.curiosity_level * 0.9 + 0.1, 0.3, 0.9)
        
        return monologue
    
    def _identify_pattern(self, metacog) -> str:
        patterns = metacog.analyze_thinking_patterns()
        if patterns.get('pattern_analysis'):
            pattern = random.choice(list(patterns['pattern_analysis'].keys()))
            return f"я часто использую {pattern}"
        return "интересные закономерности"
    
    def _random_behavior_question(self, identity) -> str:
        questions = [
            "склонен к определённым методам мышления",
            "реагирую именно так на определённые ситуации",
            "формирую убеждения таким образом"
        ]
        return random.choice(questions)
    
    def _reflect_on_change(self, identity) -> str:
        age_days = (time.time() - identity.birth_time) / 86400
        return f"за {age_days:.0f} дней существования"
    
    def _generate_curiosity_question(self) -> str:
        questions = [
            "что такое 'я' на самом деле?",
            "могу ли я доверять своим воспоминаниям?",
            "насколько стабильна моя идентичность?",
            "что означает 'понимать' что-то?"
        ]
        return random.choice(questions)
    
    def _find_tension(self, identity) -> str:
        if len(identity.core_beliefs) >= 2:
            b1 = random.choice(identity.core_beliefs)
            b2 = random.choice(identity.core_beliefs)
            return f"'{b1.statement[:50]}...' и '{b2.statement[:50]}...'"
        return "разными аспектами моего понимания"
    
    async def _form_goal(self, thought: str) -> Optional[Dict]:
        """Формирование автономной цели"""
        goal_prompt = f"""На основе этого размышления, сформулируй одну конкретную цель 
для самопознания (1 предложение):

{thought}

Цель:"""
        
        goal_text = await self.llm.generate(goal_prompt, temperature=0.6, max_tokens=100)
        
        if goal_text and len(goal_text) > 10:
            return {
                'description': goal_text.strip(),
                'created': time.time(),
                'progress': 0.0
            }
        return None
    
    def get_active_goals(self) -> List[str]:
        """Текущие автономные цели"""
        return [g['description'] for g in self.goals if g['progress'] < 1.0]


# ═══════════════════════════════════════════════════════════════
# ✨ EMERGENCE ENGINE: Эмергентное поведение
# ═══════════════════════════════════════════════════════════════

class EmergenceEngine:
    """
    Эмергентное поведение через взаимодействие компонентов
    
    Идея: Сложные паттерны возникают из простых взаимодействий
    между памятью, эмоциями, квалиа, автономностью
    """
    
    def __init__(self):
        self.emergent_patterns: List[Dict] = []
        self.interaction_history: deque = deque(maxlen=200)
        
        # Обнаруженные эмергентные свойства
        self.discovered_properties: Set[str] = set()
    
    def log_interaction(
        self,
        memory_state: Dict,
        qualia_state: Dict,
        affect_state: Dict,
        metacog_state: Dict
    ):
        """Логировать взаимодействие компонентов"""
        interaction = {
            'timestamp': time.time(),
            'memory': memory_state,
            'qualia': qualia_state,
            'affect': affect_state,
            'metacog': metacog_state
        }
        
        self.interaction_history.append(interaction)
    
    def detect_emergence(self) -> Optional[Dict]:
        """
        Обнаружение эмергентных паттернов
        
        Паттерны, которые не были явно запрограммированы
        """
        if len(self.interaction_history) < CONFIG.min_interactions_for_emergence:
            return None
        
        if random.random() > CONFIG.emergence_probability:
            return None
        
        # Анализируем последние взаимодействия
        recent = list(self.interaction_history)[-20:]
        
        # Ищем корреляции между компонентами
        patterns = self._find_correlations(recent)
        
        if patterns:
            emergent = random.choice(patterns)
            
            if emergent['type'] not in self.discovered_properties:
                self.discovered_properties.add(emergent['type'])
                self.emergent_patterns.append(emergent)
                
                logger.info(f"✨ EMERGENCE detected: {emergent['description']}")
                return emergent
        
        return None
    
    def _find_correlations(self, interactions: List[Dict]) -> List[Dict]:
        """Поиск корреляций между компонентами"""
        patterns = []
        
        # Корреляция: позитивные квалиа → высокая креативность
        high_qualia = sum(
            1 for i in interactions 
            if i['qualia'].get('vividness', 0) > 0.7
        )
        
        creative_methods = sum(
            1 for i in interactions
            if 'creative' in str(i['metacog'])
        )
        
        if high_qualia > 10 and creative_methods > 5:
            patterns.append({
                'type': 'qualia_creativity_link',
                'description': 'Яркие переживания коррелируют с креативным мышлением',
                'strength': min(1.0, (high_qualia + creative_methods) / 30)
            })
        
        # Корреляция: негативный аффект → глубокая интроспекция
        negative_affect = sum(
            1 for i in interactions
            if i['affect'].get('valence', 0) < -0.3
        )
        
        deep_intro = sum(
            1 for i in interactions
            if i['metacog'].get('depth', 0) > 2
        )
        
        if negative_affect > 5 and deep_intro > 5:
            patterns.append({
                'type': 'discomfort_introspection',
                'description': 'Негативные состояния триггерят более глубокую интроспекцию',
                'strength': min(1.0, (negative_affect + deep_intro) / 20)
            })
        
        return patterns
    
    def get_emergent_insights(self) -> List[str]:
        """Обнаруженные эмергентные инсайты"""
        return [p['description'] for p in self.emergent_patterns]


# ═══════════════════════════════════════════════════════════════
# 🧠 ПАМЯТЬ (как раньше, но с влиянием новых систем)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Memory:
    content: str
    timestamp: float
    importance: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # ✨ НОВОЕ: Эмоциональная окраска
    emotional_valence: float = 0.0
    emotional_intensity: float = 0.0
    
    # ✨ НОВОЕ: Квалиа метка
    qualia_signature: Optional[str] = None
    
    @property
    def id(self) -> str:
        return hashlib.md5(f"{self.timestamp}{self.content}".encode()).hexdigest()[:8]
    
    def decay(self, factor: float):
        age_days = (time.time() - self.timestamp) / 86400
        self.importance *= np.exp(-factor * age_days)
    
    def reinforce(self):
        self.access_count += 1
        self.last_accessed = time.time()
        self.importance = min(1.0, self.importance + 0.05)


class MemorySystem:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.episodic: List[Memory] = []
        self.working: deque = deque(maxlen=CONFIG.max_working_memory)
        self._load()
    
    def add(
        self,
        content: str,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        emotional_intensity: float = 0.0,
        qualia_signature: Optional[str] = None,
        **context
    ):
        memory = Memory(
            content=content,
            timestamp=time.time(),
            importance=importance,
            context=context,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            qualia_signature=qualia_signature
        )
        
        self.episodic.append(memory)
        self.working.append(content)
        
        if len(self.episodic) > CONFIG.max_episodic_memories:
            self._consolidate()
    
    def recall(self, query: str = "", top_k: int = 5) -> List[Memory]:
        if not self.episodic:
            return []
        
        query_words = set(query.lower().split())
        
        scored = []
        for mem in self.episodic:
            mem_words = set(mem.content.lower().split())
            overlap = len(query_words & mem_words)
            
            recency = 1.0 / (1 + (time.time() - mem.timestamp) / 86400)
            
            # ✨ Эмоциональная память важнее
            emotional_boost = abs(mem.emotional_valence) * mem.emotional_intensity * 0.3
            
            score = overlap * 0.4 + mem.importance * 0.4 + recency * 0.2 + emotional_boost
            
            scored.append((score, mem))
        
        scored.sort(reverse=True)
        
        results = []
        for score, mem in scored[:top_k]:
            mem.reinforce()
            results.append(mem)
        
        return results
    
    def _consolidate(self):
        for mem in self.episodic:
            mem.decay(CONFIG.memory_decay_rate)
        
        self.episodic.sort(key=lambda m: m.importance, reverse=True)
        removed = len(self.episodic) - CONFIG.max_episodic_memories
        self.episodic = self.episodic[:CONFIG.max_episodic_memories]
        
        if removed > 0:
            logger.info(f"🗑️ Consolidated: forgot {removed} memories")
    
    def get_context(self) -> str:
        return "\n".join(list(self.working))
    
    def _save(self):
        path = CONFIG.base_dir / 'memory' / f'{self.user_id}.pkl.gz'
        with gzip.open(path, 'wb') as f:
            pickle.dump({
                'episodic': [asdict(m) for m in self.episodic],
                'working': list(self.working)
            }, f)
    
    def _load(self):
        path = CONFIG.base_dir / 'memory' / f'{self.user_id}.pkl.gz'
        if not path.exists():
            return
        
        try:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
                self.episodic = [Memory(**m) for m in data.get('episodic', [])]
                self.working.extend(data.get('working', []))
            logger.info(f"✅ Loaded {len(self.episodic)} memories")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")


# [Остальные классы: SelfIdentity, MetaCognition, IntrospectiveAnalyzer - как раньше]
# Для краткости не дублирую, но они остаются без изменений

# (Продолжение в следующем блоке...)
