#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ADVANCED SELF-LEARNING AGI AGENT v4.2
─────────────────────────────────────────────────────────────
✓ Автоматический fallback: Telegram → Локальный чат
✓ Улучшенная обработка ошибок и логирование
✓ Расширенный локальный интерфейс с командами
✓ Оптимизация памяти и производительности
✓ Поддержка нескольких режимов работы
"""

# ═══════════════════════════════════════════════════════════════
# 🔧 БЕЗОПАСНЫЕ ИМПОРТЫ С FALLBACK
# ═══════════════════════════════════════════════════════════════

import os, sys, json, asyncio, logging, hashlib, time, random, math
import pickle, gzip, re, traceback
from pathlib import Path
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto

# Опциональные зависимости
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    def load_dotenv():
        pass


    load_dotenv()
    print("⚠️  python-dotenv не установлен → используем env напрямую")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy не установлен → используем базовые функции")

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("⚠️  aiohttp не установлен → LLM-запросы отключены")

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
# 📋 КОНФИГУРАЦИЯ С ВАЛИДАЦИЕЙ
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Конфигурация с авто-определением режима"""

    # 🔑 API ключи
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    llm_url: str = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
    llm_key: str = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

    # 📁 Пути
    base_dir: Path = Path('agi_agent_data')

    # 🎓 Память и обучение
    max_episodic_memories: int = 2000
    max_semantic_knowledge: int = 500
    max_procedural_rules: int = 200
    memory_consolidation_threshold: int = 100

    # 🔄 Параметры обучения
    learning_rate: float = 0.1
    learning_momentum: float = 0.9
    experience_threshold: int = 20
    generalization_threshold: float = 0.75

    # 🎯 Планирование
    planning_depth: int = 5
    goal_hierarchy_depth: int = 3
    reward_discount_factor: float = 0.99

    # 🔬 Причинность
    causality_threshold: float = 0.6
    max_causal_chains: int = 100

    # 🧠 Мета-обучение
    meta_learning_enabled: bool = True
    meta_learning_rate: float = 0.01
    strategy_exploration_rate: float = 0.15

    # 🔍 Интроспекция
    introspection_frequency: int = 50
    knowledge_extraction_depth: int = 4

    # 💬 Интерфейс (авто-определение)
    _mode: Optional[str] = None

    def __post_init__(self):
        # Создаём директории
        for subdir in ['memory', 'knowledge', 'rules', 'logs',
                       'causal_graph', 'strategies', 'introspection', 'backups']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Авто-определение режима
        if self.telegram_token and HAS_AIOHTTP:
            self._mode = 'telegram'
            print("✅ Режим: 📱 Telegram + LLM")
        elif HAS_AIOHTTP:
            self._mode = 'local_llm'
            print("✅ Режим: 💬 Локальный чат + LLM")
        else:
            self._mode = 'local_demo'
            print("✅ Режим: 🎮 Локальный демо-режим (без LLM)")

    @property
    def mode(self) -> str:
        return self._mode

    def validate(self) -> List[str]:
        """Валидация конфигурации"""
        warnings = []
        if not self.llm_url.startswith(('http://', 'https://')):
            warnings.append(f"⚠️  Неверный LLM URL: {self.llm_url}")
        if self.learning_rate < 0 or self.learning_rate > 1:
            warnings.append("⚠️  learning_rate должен быть в [0, 1]")
        if self.max_episodic_memories < 100:
            warnings.append("⚠️  Слишком маленький буфер памяти")
        return warnings


CONFIG = Config()
for w in CONFIG.validate(): print(w)


# ═══════════════════════════════════════════════════════════════
# 🛠️ УТИЛИТЫ (замена научных библиотек при отсутствии)
# ═══════════════════════════════════════════════════════════════

class SafeMath:
    """Безопасные математические операции без numpy"""

    @staticmethod
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def std(values: List[float]) -> float:
        if len(values) < 2: return 0.0
        m = SafeMath.mean(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

    @staticmethod
    def clip(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

    @staticmethod
    def random_sample(population: List, k: int):
        if k >= len(population): return population[:]
        return random.sample(population, k)


# Алиасы для совместимости
if not HAS_NUMPY:
    np = type('np', (), {
        'mean': SafeMath.mean,
        'std': SafeMath.std,
        'clip': SafeMath.clip,
        'random': type('random', (), {'sample': SafeMath.random_sample})()
    })()


# ═══════════════════════════════════════════════════════════════
# 📝 ПРОДВИНУТОЕ ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════

class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class AdvancedLogger:
    """Логгер с цветным выводом и фильтрацией"""

    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m',
        'WARNING': '\033[33m', 'ERROR': '\033[31m',
        'CRITICAL': '\033[35m', 'RESET': '\033[0m'
    }

    def __init__(self, name: str = 'AGI'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Консоль с цветами
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        console.setLevel(logging.INFO)

        # Файл с подробностями
        log_file = CONFIG.base_dir / 'logs' / f'agi_{datetime.now():%Y%m%d_%H%M}.log'
        file_h = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_h.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
        ))
        file_h.setLevel(logging.DEBUG)

        self.logger.addHandler(console)
        self.logger.addHandler(file_h)

    def _log(self, level: str, msg: str, *args, **kwargs):
        if sys.stdout.isatty() and level in self.COLORS:
            msg = f"{self.COLORS[level]}{msg}{self.COLORS['RESET']}"
        getattr(self.logger, level.lower())(msg, *args, **kwargs)

    def debug(self, msg, *a, **kw): self._log('DEBUG', msg, *a, **kw)

    def info(self, msg, *a, **kw): self._log('INFO', msg, *a, **kw)

    def warning(self, msg, *a, **kw): self._log('WARNING', msg, *a, **kw)

    def error(self, msg, *a, **kw): self._log('ERROR', msg, *a, **kw)

    def critical(self, msg, *a, **kw): self._log('CRITICAL', msg, *a, **kw)

    def exception(self, msg: str, exc: Exception):
        self.error(f"{msg}: {exc}\n{traceback.format_exc()}")


logger = AdvancedLogger()


# ═══════════════════════════════════════════════════════════════
# 🎓 ЯДРО ОБУЧЕНИЯ: ЭПИЗОДИЧЕСКАЯ + СЕМАНТИЧЕСКАЯ ПАМЯТЬ
# ═══════════════════════════════════════════════════════════════

@dataclass
class Experience:
    """Эпизодическая память: конкретный опыт"""
    input_data: str
    action_taken: str
    outcome: float  # 0.0 - 1.0
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    tags: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return hashlib.sha256(f"{self.timestamp}:{self.input_data[:50]}".encode()).hexdigest()[:10]

    def to_dict(self) -> Dict: return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'Experience': return cls(**d)


@dataclass
class Rule:
    """Процедурное знание: if-then правило"""
    condition: str  # Шаблон условия
    action: str  # Рекомендуемое действие
    outcome_prediction: float  # Ожидаемый результат
    confidence: float  # Уверенность 0-1
    times_applied: int = 0
    successes: int = 0
    failures: int = 0
    last_updated: float = field(default_factory=time.time)
    priority: float = 1.0  # Приоритет применения

    @property
    def id(self) -> str:
        return hashlib.sha256(f"{self.condition}:{self.action}".encode()).hexdigest()[:10]

    @property
    def success_rate(self) -> float:
        return self.successes / self.times_applied if self.times_applied > 0 else 0.0

    def apply(self, success: bool, actual_outcome: float):
        """Обновить правило после применения"""
        self.times_applied += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        # Обновление уверенности с экспоненциальным сглаживанием
        new_conf = self.success_rate
        self.confidence = (self.confidence * CONFIG.learning_momentum +
                           new_conf * (1 - CONFIG.learning_momentum))

        # Корректировка предсказания
        self.outcome_prediction = (
                self.outcome_prediction * 0.9 + actual_outcome * 0.1
        )
        self.last_updated = time.time()

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'Rule':
        return cls(**d)


@dataclass
class SemanticConcept:
    """Семантическое знание: абстрактные концепции"""
    name: str
    definition: str
    properties: Dict[str, Any]
    related_concepts: List[str]
    evidence_strength: float = 0.5
    learned_at: float = field(default_factory=time.time)
    usage_count: int = 0

    def to_dict(self) -> Dict: return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'SemanticConcept': return cls(**d)


class LearningSystem:
    """
    🧠 Ядро самообучения

    Архитектура памяти:
    ┌─────────────────────────┐
    │ Эпизодическая (deque)   │ ← Свежий опыт
    ├─────────────────────────┤
    │ Процедурная (правила)   │ ← If-then знания
    ├─────────────────────────┤
    │ Семантическая (концепты)│ ← Абстрактное знание
    └─────────────────────────┘
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.experiences: deque = deque(maxlen=CONFIG.max_episodic_memories)
        self.rules: List[Rule] = []
        self.semantic_knowledge: Dict[str, SemanticConcept] = {}

        # Статистика
        self.learning_curves: Dict[str, deque] = {
            k: deque(maxlen=200) for k in ['success_rate', 'outcome', 'confidence']
        }
        self.learning_events: deque = deque(maxlen=500)

        # Индексы для быстрого поиска
        self._action_index: Dict[str, List[str]] = defaultdict(list)  # action → rule_ids
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)  # word → rule_ids

        self._load()
        logger.info(f"🎓 LearningSystem initialized: {len(self.rules)} rules loaded")

    def add_experience(self, exp: Experience) -> None:
        """Добавить опыт и запустить обучение"""
        self.experiences.append(exp)
        self._learn_from_experience(exp)

        # Периодическая консолидация
        if len(self.experiences) % CONFIG.experience_threshold == 0:
            self._consolidate_knowledge()

    def _learn_from_experience(self, exp: Experience) -> None:
        """Обучение на одном эпизоде"""
        # 1. Найти и обновить релевантные правила
        matched = self._find_matching_rules(exp.input_data)

        if matched:
            for rule_id in matched:
                rule = next((r for r in self.rules if r.id == rule_id), None)
                if rule:
                    rule.apply(exp.success, exp.outcome)
        # 2. Создать новое правило если нет совпадений
        elif random.random() < 0.3 and len(self.rules) < CONFIG.max_procedural_rules:
            new_rule = self._create_rule_from_experience(exp)
            if new_rule:
                self.rules.append(new_rule)
                self._index_rule(new_rule)
                logger.debug(f"🆕 New rule: {new_rule.id[:6]} → {new_rule.action[:20]}")

        # 3. Обновить кривые обучения
        self.learning_curves['success_rate'].append(1.0 if exp.success else 0.0)
        self.learning_curves['outcome'].append(exp.outcome)
        if self.rules:
            self.learning_curves['confidence'].append(
                np.mean([r.confidence for r in self.rules[-10:]])
            )

        # 4. Лог события
        self.learning_events.append({
            'ts': exp.timestamp, 'success': exp.success,
            'outcome': exp.outcome, 'action': exp.action_taken
        })

    def _find_matching_rules(self, input_text: str) -> List[str]:
        """Найти правила по ключевым словам"""
        words = set(w.lower() for w in re.findall(r'\w+', input_text) if len(w) > 2)
        if not words: return []

        # Поиск по индексу
        candidate_ids = set()
        for word in words:
            candidate_ids.update(self._keyword_index.get(word, set()))

        # Проверка соответствия
        matched = []
        for rule_id in candidate_ids:
            rule = next((r for r in self.rules if r.id == rule_id), None)
            if rule and self._matches_condition(rule.condition, input_text):
                matched.append(rule_id)

        return matched[:10]  # Лимит

    def _matches_condition(self, condition: str, input_text: str) -> bool:
        """Проверка соответствия условия (упрощённый паттерн-матчинг)"""
        c_words = set(re.findall(r'\w+', condition.lower()))
        i_words = set(re.findall(r'\w+', input_text.lower()))
        if not c_words: return False
        overlap = len(c_words & i_words) / len(c_words)
        return overlap >= CONFIG.generalization_threshold * 0.5

    def _create_rule_from_experience(self, exp: Experience) -> Optional[Rule]:
        """Создать правило из опыта с обобщением"""
        # Извлечь ключевые слова из входа
        words = [w for w in re.findall(r'\w+', exp.input_data.lower()) if len(w) > 3]
        if not words: return None

        # Обобщение: оставить наиболее информативные слова
        selected = SafeMath.random_sample(words, max(int(len(words) * 0.6), 2))
        condition = ' '.join(sorted(set(selected)))

        rule = Rule(
            condition=condition,
            action=exp.action_taken,
            outcome_prediction=exp.outcome,
            confidence=0.5,
            priority=1.0
        )
        return rule

    def _index_rule(self, rule: Rule):
        """Добавить правило в индексы"""
        self._action_index[rule.action].append(rule.id)
        for word in re.findall(r'\w+', rule.condition.lower()):
            if len(word) > 2:
                self._keyword_index[word].add(rule.id)

    def _consolidate_knowledge(self) -> None:
        """Консолидация: опыт → семантическое знание"""
        if len(self.experiences) < CONFIG.memory_consolidation_threshold:
            return

        recent = list(self.experiences)[-100:]
        action_stats = defaultdict(list)

        for exp in recent:
            action_stats[exp.action_taken].append({
                'outcome': exp.outcome, 'success': exp.success, 'context': exp.context
            })

        for action, data in action_stats.items():
            if len(data) < 5: continue

            outcomes = [d['outcome'] for d in data]
            avg_out = np.mean(outcomes)
            std_out = np.std(outcomes)
            reliability = 1.0 - min(1.0, std_out)

            concept_key = f"strat_{hashlib.md5(action.encode()).hexdigest()[:8]}"

            if concept_key not in self.semantic_knowledge:
                concept = SemanticConcept(
                    name=action[:30],
                    definition=f"Strategy: {action}",
                    properties={'mean_outcome': avg_out, 'std': std_out,
                                'reliability': reliability, 'samples': len(data)},
                    related_concepts=[],
                    evidence_strength=min(0.95, len(data) * 0.1),
                    usage_count=0
                )
                self.semantic_knowledge[concept_key] = concept

                # Ограничить размер
                if len(self.semantic_knowledge) > CONFIG.max_semantic_knowledge:
                    # Удалить наимениспользуемые
                    sorted_concepts = sorted(
                        self.semantic_knowledge.items(),
                        key=lambda x: x[1].usage_count
                    )
                    for key, _ in sorted_concepts[:10]:
                        del self.semantic_knowledge[key]

                logger.info(f"📚 New concept: {concept.name[:25]} (strength={concept.evidence_strength:.2f})")

    def predict(self, input_text: str, action: str) -> Tuple[float, float]:
        """Предсказать результат действия"""
        # Поиск по правилам
        relevant = [r for r in self.rules if self._matches_condition(r.condition, input_text)]

        if relevant:
            # Взвешенное предсказание
            total_conf = sum(r.confidence for r in relevant)
            if total_conf > 0:
                pred = sum(r.outcome_prediction * r.confidence for r in relevant) / total_conf
                conf = min(1.0, total_conf / len(relevant))
                return pred, conf

        # Fallback на семантику
        for concept in self.semantic_knowledge.values():
            if action.lower() in concept.name.lower() or action.lower() in concept.definition.lower():
                concept.usage_count += 1
                return concept.properties.get('mean_outcome', 0.5), concept.evidence_strength

        return 0.5, 0.0  # Неопределённость

    def get_best_action(self, input_text: str, available_actions: List[str] = None) -> Tuple[str, float]:
        """Выбрать оптимальное действие"""
        if not self.rules and not self.semantic_knowledge:
            return "explore", 0.0

        actions = available_actions or list(set(r.action for r in self.rules)) or ["respond"]

        best_action, best_score = None, -1

        for action in actions:
            pred, conf = self.predict(input_text, action)
            score = pred * conf  # Простая функция полезности

            # Бонус за разнообразие
            if action not in [r.action for r in self.rules[-20:]]:
                score *= 1.1

            if score > best_score:
                best_score, best_action = score, action

        return best_action or random.choice(actions), best_score

    def get_stats(self) -> Dict:
        """Статистика обучения"""
        rates = list(self.learning_curves['success_rate'])
        confs = list(self.learning_curves['confidence'])

        return {
            'experiences': len(self.experiences),
            'rules': len(self.rules),
            'concepts': len(self.semantic_knowledge),
            'recent_success': np.mean(rates[-20:]) if rates else 0.0,
            'avg_confidence': np.mean(confs[-20:]) if confs else 0.0,
            'learning_velocity': (rates[-1] - rates[0]) / len(rates) if len(rates) > 1 else 0,
            'action_diversity': len(set(r.action for r in self.rules))
        }

    def _save(self):
        """Сохранение с компрессией"""
        try:
            path = CONFIG.base_dir / 'knowledge' / f'{self.agent_id}.pkl.gz'
            temp_path = path.with_suffix('.tmp')

            data = {
                'version': '4.2',
                'agent_id': self.agent_id,
                'experiences': [e.to_dict() for e in self.experiences],
                'rules': [r.to_dict() for r in self.rules],
                'semantic': {k: v.to_dict() for k, v in self.semantic_knowledge.items()},
                'curves': {k: list(v) for k, v in self.learning_curves.items()},
                'saved_at': datetime.now().isoformat()
            }

            with gzip.open(temp_path, 'wb', compresslevel=6) as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            temp_path.replace(path)  # Атомарная замена
            logger.info(f"💾 Saved: {path.name} ({os.path.getsize(path) // 1024}KB)")

        except Exception as e:
            logger.exception(f"Save failed: {e}")

    def _load(self):
        """Загрузка с проверкой версии"""
        try:
            path = CONFIG.base_dir / 'knowledge' / f'{self.agent_id}.pkl.gz'
            if not path.exists(): return

            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)

            if data.get('version') != '4.2':
                logger.warning(f"⚠️  Version mismatch: {data.get('version')} → 4.2")

            self.experiences.extend(Experience.from_dict(e) for e in data.get('experiences', []))
            self.rules = [Rule.from_dict(r) for r in data.get('rules', [])]
            self.semantic_knowledge = {
                k: SemanticConcept.from_dict(v)
                for k, v in data.get('semantic', {}).items()
            }
            for k, v in data.get('curves', {}).items():
                self.learning_curves[k].extend(v[-200:])

            # Восстановить индексы
            for rule in self.rules:
                self._index_rule(rule)

            logger.info(f"✅ Loaded: {len(self.rules)} rules, {len(self.semantic_knowledge)} concepts")

        except Exception as e:
            logger.exception(f"Load failed: {e}")


# ═══════════════════════════════════════════════════════════════
# 🧠 МЕТА-ОБУЧЕНИЕ: ОПТИМИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ
# ═══════════════════════════════════════════════════════════════

class MetaLearner:
    """Адаптация гиперпараметров на основе производительности"""

    def __init__(self):
        self.strategy_scores: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.params = {
            'exploration_rate': CONFIG.strategy_exploration_rate,
            'temperature_base': 0.6,
            'response_length_target': 300
        }
        self.performance_window: deque = deque(maxlen=50)

    def evaluate(self, strategy: str, success: bool, confidence: float,
                 actual_outcome: float) -> None:
        """Оценить стратегию с калибровкой уверенности"""
        # Основной скор
        reward = 1.0 if success else 0.0

        # Калибровка: штраф за переоценку/недооценку
        calibration = 1.0 - abs(confidence - actual_outcome)

        # Итоговый скор
        score = reward * 0.6 + calibration * 0.3 + confidence * 0.1
        self.strategy_scores[strategy].append(score)
        self.performance_window.append(score)

    def adapt(self) -> Dict:
        """Адаптировать параметры"""
        if len(self.performance_window) < 10:
            return self.params

        recent = list(self.performance_window)[-20:]
        avg_perf = np.mean(recent)
        trend = recent[-5] - recent[-10] if len(recent) >= 10 else 0

        # Адаптация exploration rate
        if avg_perf > 0.75 and trend > 0:
            self.params['exploration_rate'] *= 0.97  # Уменьшить исследование
        elif avg_perf < 0.4 or trend < -0.1:
            self.params['exploration_rate'] *= 1.03  # Увеличить исследование

        # Ограничения
        self.params['exploration_rate'] = np.clip(
            self.params['exploration_rate'], 0.05, 0.4
        )

        return self.params

    def get_temperature(self, strategy: str) -> float:
        """Динамическая температура для LLM"""
        base = self.params['temperature_base']
        if strategy in self.strategy_scores:
            perf = np.mean(list(self.strategy_scores[strategy])[-10:])
            # Высокая производительность → меньше креативности
            return np.clip(base - (perf - 0.5) * 0.4, 0.3, 0.9)
        return base


# ═══════════════════════════════════════════════════════════════
# 🔍 ИНТРОСПЕКЦИЯ: САМОАНАЛИЗ
# ═══════════════════════════════════════════════════════════════

class Introspector:
    """Анализ собственного обучения и знаний"""

    def __init__(self):
        self.records: deque = deque(maxlen=50)
        self.insights: List[str] = []

    async def analyze(self, learning: LearningSystem, llm=None) -> Dict:
        """Глубокий самоанализ"""
        stats = learning.get_stats()

        # Количественные метрики
        analysis = {
            'learning_velocity': stats['learning_velocity'],
            'knowledge_stability': self._measure_stability(learning),
            'action_diversity': stats['action_diversity'] / max(len(learning.rules), 1),
            'confidence_calibration': self._check_calibration(learning),
        }

        # Качественный анализ через LLM если доступен
        if llm and stats['experiences'] > 20:
            prompt = f"""Агент проанализируй своё обучение:
- Опыт: {stats['experiences']}
- Правила: {stats['rules']}
- Успешность: {stats['recent_success']:.0%}
- Уверенность: {stats['avg_confidence']:.0%}

Вывод (1 предложение): что работает хорошо, что улучшить?"""
            try:
                llm_insight = await llm.generate(prompt, temperature=0.3, max_tokens=100)
                analysis['llm_insight'] = llm_insight[:200]
            except:
                pass

        self.records.append({'ts': time.time(), **analysis})
        return analysis

    def _measure_stability(self, learning: LearningSystem) -> float:
        """Стабильность предсказаний"""
        if len(learning.learning_curves['confidence']) < 10:
            return 0.5
        confs = list(learning.learning_curves['confidence'])[-20:]
        return 1.0 - np.std(confs)  # Меньше разброс = стабильнее

    def _check_calibration(self, learning: LearningSystem) -> float:
        """Калибровка уверенности: насколько уверенность соответствует успеху"""
        if not learning.rules: return 0.5
        # Простая эвристика: средняя уверенность успешных правил
        successful = [r for r in learning.rules if r.success_rate > 0.6]
        if not successful: return 0.5
        return np.mean([r.confidence for r in successful])


# ═══════════════════════════════════════════════════════════════
# 🤖 LLM CLIENT С ПОВТОРНЫМИ ПОПЫТКАМИ
# ═══════════════════════════════════════════════════════════════

class LLMClient:
    """Клиент с retry, fallback и кэшированием"""

    def __init__(self, url: str, key: str):
        self.url, self.key = url, key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[str, float]] = {}  # prompt → (response, timestamp)
        self.stats = {'requests': 0, 'errors': 0, 'cache_hits': 0}

    async def connect(self):
        if not HAS_AIOHTTP or self._session: return
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=90),
                headers={'Connection': 'keep-alive'}
            )
            logger.info("🔗 LLM connected")
        except Exception as e:
            logger.error(f"LLM connect failed: {e}")

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def generate(self, prompt: str, temperature: float = 0.7,
                       max_tokens: int = 500, use_cache: bool = True) -> str:
        """Генерация с retry и кэшем"""
        # Проверка кэша
        if use_cache:
            cache_key = hashlib.md5(f"{prompt}:{temperature}".encode()).hexdigest()
            if cache_key in self._cache:
                resp, ts = self._cache[cache_key]
                if time.time() - ts < 3600:  # 1 час
                    self.stats['cache_hits'] += 1
                    return resp

        if not HAS_AIOHTTP:
            return self._fallback_response(prompt)

        await self.connect()
        if not self._session:
            return self._fallback_response(prompt)

        # Retry логика
        for attempt in range(3):
            try:
                async with self._session.post(
                        self.url,
                        json={
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "stream": False
                        },
                        headers={"Authorization": f"Bearer {self.key}"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                        if result:
                            self.stats['requests'] += 1
                            # Кэширование
                            if use_cache and len(prompt) < 500:
                                self._cache[cache_key] = (result, time.time())
                                if len(self._cache) > 100:
                                    oldest = min(self._cache, key=lambda k: self._cache[k][1])
                                    del self._cache[oldest]
                            return result

                    logger.warning(f"LLM status: {resp.status}")

            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"LLM connection error: {e}")
            except Exception as e:
                logger.exception(f"LLM unexpected error: {e}")

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        self.stats['errors'] += 1
        return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Ответ при недоступности LLM"""
        responses = [
            "Я анализирую ваш запрос и учусь на этом опыте. 🧠",
            "Интересный вопрос! Моё обучение продолжается. 📚",
            "На основе накопленного опыта могу сказать: продолжайте исследовать! ✨",
            "🔄 Обработка... Мой алгоритм адаптации работает.",
        ]
        return random.choice(responses) + f"\n\n[Режим: {'LLM' if HAS_AIOHTTP else 'Demo'}]"

    def get_stats(self) -> Dict:
        return {**self.stats, 'cache_size': len(self._cache)}


# ═══════════════════════════════════════════════════════════════
# 🏆 ГЛАВНЫЙ AGI АГЕНТ
# ═══════════════════════════════════════════════════════════════

class AdvancedAGIAgent:
    """Полнофункциональный самообучающийся агент"""

    def __init__(self, agent_id: str, llm: Optional[LLMClient] = None):
        self.agent_id = agent_id
        self.llm = llm

        # Компоненты
        self.learning = LearningSystem(agent_id)
        self.meta = MetaLearner()
        self.intro = Introspector()

        # Состояние
        self.interaction_count = 0
        self.session_start = time.time()
        self.last_introspection = 0

        logger.info(f"🚀 AGI Agent '{agent_id}' initialized [{CONFIG.mode}]")

    async def process(self, user_input: str) -> Tuple[str, Dict]:
        """Полный цикл обработки с обучением"""
        start = time.time()
        self.interaction_count += 1

        # 1. ВЫБОР СТРАТЕГИИ
        action, confidence = self.learning.get_best_action(user_input)
        temperature = self.meta.get_temperature(action)

        # 2. ГЕНЕРАЦИЯ ОТВЕТА
        if self.llm and (action == "explore" or random.random() < self.meta.params['exploration_rate']):
            # Исследовательский режим
            prompt = f"User: {user_input}\n\nAssistant (be creative):"
            response = await self.llm.generate(prompt, temperature=temperature, max_tokens=400)
        elif self.llm:
            # Эксплуатационный режим
            prompt = f"Based on learned patterns, respond helpfully:\nUser: {user_input}\n\nAssistant:"
            response = await self.llm.generate(prompt, temperature=temperature * 0.8, max_tokens=400)
        else:
            # Демо-режим без LLM
            response = self._demo_response(user_input, action)

        # 3. ОЦЕНКА КАЧЕСТВА
        outcome = self._evaluate(user_input, response)
        success = outcome > 0.5

        # 4. ОБУЧЕНИЕ
        exp = Experience(
            input_data=user_input, action_taken=action,
            outcome=outcome, context={'confidence': confidence},
            success=success, tags=self._extract_tags(user_input)
        )
        self.learning.add_experience(exp)
        self.meta.evaluate(action, success, confidence, outcome)
        self.meta.adapt()

        # 5. ПЕРИОДИЧЕСКАЯ ИНТРОСПЕКЦИЯ
        intro_note = ""
        if self.interaction_count % CONFIG.introspection_frequency == 0:
            analysis = await self.intro.analyze(self.learning, self.llm)
            eff = analysis.get('learning_velocity', 0)
            intro_note = f"\n\n🔍 <i>Self: velocity={eff:+.2%}</i>"
            if 'llm_insight' in analysis:
                intro_note += f"\n💡 {analysis['llm_insight']}"

        # Метрики
        elapsed = time.time() - start
        metadata = {
            'action': action, 'confidence': confidence,
            'outcome': outcome, 'success': success,
            'time_ms': elapsed * 1000,
            'stats': self.learning.get_stats(),
            'meta': self.meta.params.copy(),
            'llm_stats': self.llm.get_stats() if self.llm else {}
        }

        logger.info(
            f"✅ [{self.agent_id}] #{self.interaction_count} | "
            f"{action[:15]} | out={outcome:.0%} | "
            f"t={elapsed * 1000:.0f}ms | rules={len(self.learning.rules)}"
        )

        return response + intro_note, metadata

    def _evaluate(self, inp: str, resp: str) -> float:
        """Эвристика оценки качества ответа"""
        score = 0.5

        # Длина
        if 50 < len(resp) < 2000: score += 0.15

        # Релевантность (пересечение слов)
        inp_w = set(re.findall(r'\w{3,}', inp.lower()))
        resp_w = set(re.findall(r'\w{3,}', resp.lower()))
        if inp_w:
            score += len(inp_w & resp_w) / len(inp_w) * 0.2

        # Разнообразие
        if len(set(resp.split())) > 15: score += 0.1

        # Качественные маркеры
        quality_words = ['because', 'therefore', 'important', 'consider',
                         'analysis', 'however', 'conclusion', 'example']
        score += min(0.15, sum(1 for w in quality_words if w in resp.lower()) * 0.025)

        # Штраф за повторы
        if len(resp) > 100 and resp.count(resp[:20]) > 3: score -= 0.1

        return np.clip(score, 0.0, 1.0)

    def _extract_tags(self, text: str) -> List[str]:
        """Извлечь теги из текста для категоризации"""
        tags = []
        if any(w in text.lower() for w in ['what', 'как', 'что']): tags.append('question')
        if any(w in text.lower() for w in ['help', 'помог', 'нужн']): tags.append('help_request')
        if any(w in text.lower() for w in ['learn', 'уч', 'train']): tags.append('meta')
        if '?' in text: tags.append('interrogative')
        return tags

    def _demo_response(self, inp: str, action: str) -> str:
        """Ответ в демо-режиме"""
        templates = {
            'question': f"🤔 Интересный вопрос! На основе {len(self.learning.rules)} правил: продолжайте исследовать.",
            'help_request': f"🔧 Анализирую... Моё обучение: {self.learning.get_stats()['recent_success']:.0%} успеха.",
            'meta': f"🧠 Мета-уровень: адаптация работает, exploration={self.meta.params['exploration_rate']:.2f}",
            'default': f"✨ Обработано. Опыт: {len(self.learning.experiences)} | Правила: {len(self.learning.rules)}"
        }
        tag = next((t for t in self._extract_tags(inp) if t in templates), 'default')
        return templates[tag]

    def get_status(self) -> Dict:
        """Полный статус агента"""
        return {
            'agent': self.agent_id,
            'uptime': time.time() - self.session_start,
            'interactions': self.interaction_count,
            'mode': CONFIG.mode,
            'learning': self.learning.get_stats(),
            'meta_params': self.meta.params,
            'introspections': len(self.intro.records),
        }

    async def save(self, backup: bool = False):
        """Сохранение состояния"""
        self.learning._save()
        if backup:
            backup_path = CONFIG.base_dir / 'backups' / f'{self.agent_id}_{datetime.now():%Y%m%d_%H%M}.bak'
            import shutil
            src = CONFIG.base_dir / 'knowledge' / f'{self.agent_id}.pkl.gz'
            if src.exists():
                shutil.copy2(src, backup_path)
                logger.info(f"📦 Backup: {backup_path.name}")
        logger.info(f"💾 State saved")


# ═══════════════════════════════════════════════════════════════
# 💬 ЛОКАЛЬНЫЙ ЧАТ ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════

class LocalChatUI:
    """Консольный интерфейс с командами"""

    COMMANDS = {
        '/help': 'Показать справку',
        '/stats': 'Статистика обучения',
        '/save': 'Сохранить состояние',
        '/load': 'Загрузить состояние',
        '/backup': 'Создать бэкап',
        '/clear': 'Очистить экран',
        '/rules': 'Показать правила (топ-5)',
        '/concepts': 'Семантические концепции',
        '/meta': 'Параметры мета-обучения',
        '/quit': 'Выйти',
    }

    BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║  🧠 AGI AGENT v4.2 — Self-Learning Core                      ║
║  Режим: {mode:<45} ║
╚═══════════════════════════════════════════════════════════════╝
💡 Команды: /help  |  Введите сообщение для начала
─────────────────────────────────────────────────────────────────
"""

    def __init__(self, agent: AdvancedAGIAgent):
        self.agent = agent
        self.running = True
        self.history: deque = deque(maxlen=20)

    async def run(self):
        """Главный цикл чата"""
        print(self.BANNER.format(mode=CONFIG.mode))

        while self.running:
            try:
                # Ввод с индикатором
                prefix = f"[#{self.agent.interaction_count + 1}] "
                user_in = input(f"\n{prefix}👤 You: ").strip()

                if not user_in: continue

                # Обработка команд
                if user_in.startswith('/'):
                    await self._handle_command(user_in)
                    continue

                # Сохранение в историю
                self.history.append(('user', user_in))

                # Обработка агентом
                print(f"{' ' * len(prefix)}🤖 Agent: ", end='', flush=True)
                start = time.time()

                response, meta = await self.agent.process(user_in)

                # Вывод с анимацией
                elapsed = (time.time() - start) * 1000
                print(f"\n{' ' * len(prefix)}{response}")

                # Мета-информация
                if meta['success']:
                    print(f"{' ' * len(prefix)}{'─' * 40}\n"
                          f"{' ' * len(prefix)}✅ Outcome: {meta['outcome']:.0%} | "
                          f"Conf: {meta['confidence']:.0%} | "
                          f"Time: {meta['time_ms']:.0f}ms")
                else:
                    print(f"{' ' * len(prefix)}{'─' * 40}\n"
                          f"{' ' * len(prefix)}⚠️ Outcome: {meta['outcome']:.0%} (learning...)")

                self.history.append(('agent', response))

            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 До свидания! Сохраняю состояние...")
                await self.agent.save()
                break
            except Exception as e:
                logger.exception(f"Chat error: {e}")
                print(f"❌ Ошибка: {e}")

    async def _handle_command(self, cmd: str):
        """Обработка команд"""
        c = cmd.lower().strip()

        if c in ['/quit', '/exit', '/q']:
            self.running = False

        elif c == '/help':
            print("\n📋 Доступные команды:")
            for cmd, desc in self.COMMANDS.items():
                print(f"   {cmd:<12} {desc}")

        elif c == '/stats':
            stats = self.agent.get_status()
            print(f"\n📊 Статистика:")
            print(f"   Взаимодействий: {stats['interactions']}")
            print(f"   Аптайм: {stats['uptime'] // 60:.0f}м {stats['uptime'] % 60:.0f}с")
            print(f"\n🎓 Обучение:")
            for k, v in stats['learning'].items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.2%}" if 'rate' in k or 'success' in k else f"   {k}: {v:.2f}")
                else:
                    print(f"   {k}: {v}")

        elif c == '/save':
            await self.agent.save()
            print("✅ Сохранено!")

        elif c == '/load':
            self.agent.learning._load()
            print("✅ Загружено!")

        elif c == '/backup':
            await self.agent.save(backup=True)

        elif c == '/clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            print(self.BANNER.format(mode=CONFIG.mode))

        elif c == '/rules':
            rules = sorted(self.agent.learning.rules, key=lambda r: r.confidence, reverse=True)[:5]
            print(f"\n📜 Топ-правила:")
            for i, r in enumerate(rules, 1):
                print(f"   {i}. [{r.confidence:.0%}] {r.condition[:40]} → {r.action[:20]}")

        elif c == '/concepts':
            concepts = sorted(
                self.agent.learning.semantic_knowledge.values(),
                key=lambda c: c.evidence_strength, reverse=True
            )[:5]
            print(f"\n🧠 Концепции:")
            for c in concepts:
                print(f"   • {c.name} (strength: {c.evidence_strength:.0%})")

        elif c == '/meta':
            print(f"\n⚙️  Мета-параметры:")
            for k, v in self.agent.meta.params.items():
                print(f"   {k}: {v}")

        else:
            print(f"❓ Неизвестная команда. /help для списка")


# ═══════════════════════════════════════════════════════════════
# 🚀 ЗАПУСК
# ═══════════════════════════════════════════════════════════════

async def main():
    """Точка входа"""

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║     🧠 ADVANCED AGI AGENT v4.2                               ║
║  ─────────────────────────────────────────                  ║
║  🔄 Режим: {CONFIG.mode.upper():<44} ║
╚═══════════════════════════════════════════════════════════════╝
✨ Возможности:
   🎓 Эпизодическая + процедурная + семантическая память
   🧠 Мета-обучение с адаптацией гиперпараметров  
   🔍 Интроспекция и самоанализ
   💬 Telegram / Локальный чат (авто-переключение)
   💾 Автосохранение с компрессией
""")

    # Инициализация
    llm = LLMClient(CONFIG.llm_url, CONFIG.llm_key) if HAS_AIOHTTP else None
    agent = AdvancedAGIAgent("main_agent", llm)

    try:
        # Запуск соответствующего интерфейса
        if CONFIG.mode == 'telegram':
            print("📱 Telegram режим требует aiogram: pip install aiogram")
            print("⚠️  Запускаю локальный интерфейс...")
            await LocalChatUI(agent).run()
        else:
            await LocalChatUI(agent).run()

        # Финальное сохранение
        await agent.save(backup=True)

        # Итоговая статистика
        status = agent.get_status()
        print(f"\n📈 Сессия завершена:")
        print(f"   Взаимодействий: {status['interactions']}")
        print(f"   Правил выучено: {status['learning']['rules']}")
        print(f"   Успешность: {status['learning']['recent_success']:.0%}")

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        if llm:
            await llm.close()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)