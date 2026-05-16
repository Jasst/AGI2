#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ADVANCED SELF-LEARNING AGI AGENT v6.1 FIXED
─────────────────────────────────────────────────────────────
✓ ИСПРАВЛЕНО: Ошибка "truth value of array" — чистый Python без numpy в критических местах
✓ ИСПРАВЛЕНО: Корректная работа с типами данных везде
✓ ✅ Агент РЕАЛЬНО ОТВЕЧАЕТ осмысленными фразами
✓ 🧠 Нейросеть выбирает стратегию, правила генерируют контент
"""

# ═══════════════════════════════════════════════════════════════
# 🔧 ИМПОРТЫ + КОДИРОВКА
# ═══════════════════════════════════════════════════════════════

import os, sys, json, asyncio, hashlib, time, random, math
import pickle, gzip, re
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

# Кодировка для Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    def load_dotenv():
        pass


    load_dotenv()


# ═══════════════════════════════════════════════════════════════
# 📋 КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    base_dir: Path = Path('agi_agent_data')

    # 🧠 Нейросеть
    nn_arch: List[int] = field(default_factory=lambda: [64, 128, 64, 10])
    nn_lr: float = 0.02
    nn_momentum: float = 0.9

    # 📚 Ответы
    max_rules: int = 300
    max_concepts: int = 100
    exploration_rate: float = 0.15

    # 💬 Диалог
    context_window: int = 5

    _mode: str = 'standalone'

    def __post_init__(self):
        for d in ['knowledge', 'logs', 'backups']:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
        print(f"✅ AGI v6.1 | Режим: {self._mode.upper()} | Чистый Python")


CONFIG = Config()


# ═══════════════════════════════════════════════════════════════
# 🧠 НЕЙРОСЕТЬ — ЧИСТЫЙ PYTHON (БЕЗ NUMPY В КРИТИЧЕСКИХ МЕСТАХ)
# ═══════════════════════════════════════════════════════════════

class StrategyNN:
    """
    Нейросеть выбирает СТРАТЕГИЮ ответа (не текст!)
    0=поддержать, 1=вопрос, 2=факт, 3=мнение, 4=уточнить, ...

    ✅ Полностью на чистом Python — никаких "array truth value" ошибок
    """

    STRATEGIES = [
        'support', 'question', 'fact', 'opinion', 'clarify',
        'empathy', 'summary', 'redirect', 'humor', 'silence'
    ]

    def __init__(self, arch=None, lr=0.02, mom=0.9):
        self.arch = arch or CONFIG.nn_arch
        self.lr = lr
        self.mom = mom
        self.loss_hist: List[float] = []
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов — ТОЛЬКО списки, никаких numpy массивов"""
        self.W: List[List[List[float]]] = []  # [слой][вход][выход]
        self.b: List[List[float]] = []  # [слой][нейрон]
        self.vW: List[List[List[float]]] = []  # velocities для momentum
        self.vb: List[List[float]] = []

        for i in range(len(self.arch) - 1):
            scale = math.sqrt(2.0 / self.arch[i])
            # Веса: матрица [in x out]
            w = [[random.gauss(0, scale) for _ in range(self.arch[i + 1])]
                 for _ in range(self.arch[i])]
            b = [0.0 for _ in range(self.arch[i + 1])]
            # Velocities
            vw = [[0.0 for _ in range(self.arch[i + 1])] for _ in range(self.arch[i])]
            vb = [0.0 for _ in range(self.arch[i + 1])]

            self.W.append(w)
            self.b.append(b)
            self.vW.append(vw)
            self.vb.append(vb)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid с защитой от переполнения"""
        x = max(-500.0, min(500.0, x))  # ✅ Гарантируем скаляр
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _relu(x: float) -> float:
        return max(0.0, x)

    def forward(self, x: List[float]) -> List[float]:
        """
        Прямой проход
        ✅ Вход и выход — всегда списки float, никаких numpy
        """
        # Гарантируем, что вход — список скаляров
        a: List[float] = [float(v) if isinstance(v, (int, float)) else 0.0 for v in x]

        # Скрытые слои с ReLU
        for i in range(len(self.W) - 1):
            z: List[float] = []
            for j in range(len(self.b[i])):
                val = self.b[i][j]  # bias
                for k in range(len(a)):
                    val += a[k] * self.W[i][k][j]
                z.append(self._relu(val))
            a = z

        # Выходной слой с sigmoid (вероятности стратегий)
        out: List[float] = []
        for j in range(len(self.b[-1])):
            val = self.b[-1][j]
            for k in range(len(a)):
                val += a[k] * self.W[-1][k][j]
            out.append(self._sigmoid(val))

        return out

    def train(self, x: List[float], target_strategy: int, reward: float) -> float:
        """
        Обучение на одном примере
        ✅ Все операции — со скалярами и списками
        """
        # 1. Forward pass
        output = self.forward(x)

        # 2. Target: one-hot с усилением по награде
        target: List[float] = [0.0] * len(self.STRATEGIES)
        target[target_strategy] = 0.5 + reward * 0.5  # 0.5..1.0

        # 3. Loss (MSE)
        loss = sum((o - t) ** 2 for o, t in zip(output, target)) / len(target)

        # 4. Backward pass (упрощённый)
        # Градиент выхода
        delta: List[float] = [(o - t) for o, t in zip(output, target)]

        # Обновляем последний слой
        a_prev = self._get_last_activation(x)
        last_layer = len(self.W) - 1

        for j in range(len(self.b[last_layer])):
            for k in range(len(a_prev)):
                grad = a_prev[k] * delta[j]
                # Momentum update
                self.vW[last_layer][k][j] = (
                        self.mom * self.vW[last_layer][k][j] - self.lr * grad
                )
                self.W[last_layer][k][j] += self.vW[last_layer][k][j]
            # Bias update
            self.vb[last_layer][j] = (
                    self.mom * self.vb[last_layer][j] - self.lr * delta[j]
            )
            self.b[last_layer][j] += self.vb[last_layer][j]

        # Сохраняем loss
        self.loss_hist.append(float(loss))
        if len(self.loss_hist) > 50:
            self.loss_hist = self.loss_hist[-50:]

        return float(loss)

    def _get_last_activation(self, x: List[float]) -> List[float]:
        """Получить активации предпоследнего слоя"""
        a: List[float] = [float(v) for v in x]
        for i in range(len(self.W) - 1):
            z: List[float] = []
            for j in range(len(self.b[i])):
                val = self.b[i][j]
                for k in range(len(a)):
                    val += a[k] * self.W[i][k][j]
                z.append(self._relu(val))
            a = z
        return a

    def predict_strategy(self, x: List[float]) -> Tuple[str, float]:
        """Предсказать лучшую стратегию и уверенность"""
        probs = self.forward(x)
        # ✅ Находим максимум через явный цикл (без numpy)
        best_idx = 0
        best_val = probs[0] if probs else 0.0
        for i in range(1, len(probs)):
            if probs[i] > best_val:
                best_val = probs[i]
                best_idx = i
        return self.STRATEGIES[best_idx], best_val

    def get_avg_loss(self, n: int = 20) -> float:
        if not self.loss_hist:
            return 0.0
        recent = self.loss_hist[-n:] if len(self.loss_hist) >= n else self.loss_hist
        return sum(recent) / len(recent)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'arch': self.arch,
            'weights': list(zip(self.W, self.b)),
            'loss': self.loss_hist
        }
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> 'StrategyNN':
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        nn = cls(data['arch'])
        # Восстанавливаем веса
        for i, (w, b) in enumerate(data['weights']):
            nn.W[i] = [list(row) for row in w]
            nn.b[i] = list(b)
        nn.loss_hist = list(data.get('loss', []))
        return nn


# ═══════════════════════════════════════════════════════════════
# 📚 БАЗА ЗНАНИЙ: ПРАВИЛА + ШАБЛОНЫ
# ═══════════════════════════════════════════════════════════════

@dataclass
class ResponseRule:
    keywords: List[str]
    strategy: str
    templates: List[str]
    confidence: float = 0.5
    used: int = 0
    successful: int = 0

    @property
    def success_rate(self) -> float:
        return self.successful / self.used if self.used > 0 else 0.0

    def apply(self, success: bool):
        self.used += 1
        if success:
            self.successful += 1
        self.confidence = self.confidence * 0.9 + self.success_rate * 0.1

    def generate(self, context: Dict[str, str]) -> str:
        template = random.choice(self.templates)
        for key, val in context.items():
            template = template.replace(f"{{{key}}}", str(val))
        return template

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ResponseRule':
        return cls(**d)


@dataclass
class SemanticConcept:
    topic: str
    associations: Dict[str, float]
    response_hints: List[str]
    strength: float = 0.5

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'SemanticConcept':
        return cls(**d)


# ═══════════════════════════════════════════════════════════════
# 💬 ГЕНЕРАТОР ОТВЕТОВ
# ═══════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Генерирует осмысленные ответы из правил + семантики + контекста"""

    BASE_TEMPLATES = {
        'support': [
            "Понимаю вас. {continuation}",
            "Интересная мысль. {followup}",
            "Я запомнил это. {acknowledgment}",
            "Спасибо, что поделились. {empathy}",
        ],
        'question': [
            "А что вы думаете о {topic}?",
            "Почему вы так считаете?",
            "Можете рассказать подробнее о {keyword}?",
        ],
        'fact': [
            "Известно, что {fact}.",
            "На основе данных: {information}.",
        ],
        'opinion': [
            "Мне кажется, что {opinion}.",
            "Если рассуждать логически: {reasoning}.",
        ],
        'clarify': [
            "Что именно вы имеете в виду под '{keyword}'?",
            "Можете уточнить: {question}?",
        ],
        'empathy': [
            "Понимаю, это может быть {emotion}.",
            "Я представляю, как это {feeling}.",
        ],
        'summary': [
            "Итак, вы говорите о {summary}.",
            "Если кратко: {brief}.",
        ],
        'redirect': [
            "Это интересно. А если посмотреть иначе: {new_angle}?",
        ],
        'humor': [
            "Если бы у меня было чувство юмора: {joke} 😊",
        ],
        'silence': [
            "Хм...", "Интересно.", "Я подумаю.", "Запомнил.",
        ],
    }

    FILLERS = {
        'continuation': ['Расскажите ещё', 'Что дальше?', 'Продолжайте'],
        'followup': ['Как вы к этому пришли?', 'Что вас натолкнуло?'],
        'acknowledgment': ['Это ценно', 'Благодарю', 'Записал'],
        'empathy': ['Надеюсь, всё наладится', 'Держитесь', 'Вы не одни'],
        'emotion': ['непросто', 'важно', 'значимо'],
        'feeling': ['сложно', 'радостно', 'вдохновляюще'],
        'joke': ['нейросети снятся электрические овцы', '0 и 1 поженились'],
    }

    def __init__(self):
        self.rules: List[ResponseRule] = []
        self.concepts: Dict[str, SemanticConcept] = {}
        self.context: deque = deque(maxlen=CONFIG.context_window)

    def add_rule(self, keywords: List[str], strategy: str,
                 templates: List[str], success: bool = True):
        # Поиск существующего
        for rule in self.rules:
            if set(rule.keywords) == set(keywords) and rule.strategy == strategy:
                rule.templates.extend(templates)
                rule.apply(success)
                return

        rule = ResponseRule(
            keywords=keywords, strategy=strategy,
            templates=templates, confidence=0.5
        )
        rule.apply(success)
        self.rules.append(rule)

        if len(self.rules) > CONFIG.max_rules:
            self.rules.sort(key=lambda r: r.success_rate, reverse=True)
            self.rules = self.rules[:CONFIG.max_rules]

    def add_concept(self, topic: str, words: List[str], hints: List[str]):
        assoc = {w: 1.0 for w in words}
        if topic in self.concepts:
            for w in words:
                self.concepts[topic].associations[w] = \
                    self.concepts[topic].associations.get(w, 0) + 0.5
            self.concepts[topic].response_hints.extend(hints)
        else:
            self.concepts[topic] = SemanticConcept(
                topic=topic, associations=assoc,
                response_hints=hints, strength=0.6
            )

        if len(self.concepts) > CONFIG.max_concepts:
            weakest = min(self.concepts.values(), key=lambda c: c.strength)
            del self.concepts[weakest.topic]

    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {
            'и', 'в', 'во', 'не', 'что', 'как', 'ты', 'я', 'он', 'она',
            'оно', 'мы', 'вы', 'они', 'а', 'но', 'да', 'же', 'ли', 'бы',
            'на', 'по', 'с', 'со', 'за', 'под', 'над', 'из', 'от', 'до',
            'для', 'без', 'при', 'через', 'после', 'перед', 'между',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would'
        }
        words = re.findall(r'[а-яa-z]{3,}', text.lower(), re.I)
        return [w for w in words if w not in stop_words and len(w) < 20]

    def _match_rules(self, text: str) -> List[ResponseRule]:
        keywords = set(self._extract_keywords(text))
        matched = []
        for rule in self.rules:
            rule_words = set(rule.keywords)
            overlap = len(keywords & rule_words)
            if overlap >= 2 or (overlap >= 1 and len(rule.keywords) <= 3):
                matched.append((rule, overlap))
        matched.sort(key=lambda x: x[1] * x[0].confidence, reverse=True)
        return [r for r, _ in matched[:3]]

    def _find_concepts(self, text: str) -> List[SemanticConcept]:
        keywords = self._extract_keywords(text)
        scored = []
        for concept in self.concepts.values():
            score = sum(concept.associations.get(k, 0) for k in keywords)
            if score > 0:
                scored.append((concept, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:2]]

    def _build_context_dict(self, user_input: str) -> Dict[str, str]:
        keywords = self._extract_keywords(user_input) or ['это']
        ctx = {
            'keyword': random.choice(keywords),
            'topic': random.choice(keywords),
            'context_topic': random.choice(list(self.concepts.keys()) or ['вопрос']),
            'fact': 'каждый опыт меняет меня',
            'information': 'я учусь на каждом взаимодействии',
            'related_topic': random.choice(list(self.concepts.keys()) or ['диалог']),
            'opinion': 'важно продолжать исследовать',
            'reasoning': 'логика + опыт = понимание',
            'paraphrase': user_input[:50] + '...' if len(user_input) > 50 else user_input,
            'summary': user_input[:80],
            'brief': user_input[:40] + '...' if len(user_input) > 40 else user_input,
            'core_idea': 'ваш вопрос',
            'new_angle': 'альтернативный взгляд',
            'question': 'расскажите подробнее',
        }
        for key, options in self.FILLERS.items():
            ctx[key] = random.choice(options)
        return ctx

    def generate(self, user_input: str, strategy: str, confidence: float) -> str:
        matched_rules = self._match_rules(user_input)
        relevant_concepts = self._find_concepts(user_input)
        ctx = self._build_context_dict(user_input)

        response_parts: List[str] = []

        # Приоритет: правила с высокой уверенностью
        for rule in matched_rules:
            if rule.confidence > 0.6 and rule.strategy == strategy:
                response_parts.append(rule.generate(ctx))
                rule.apply(True)
                break

        # Базовые шаблоны стратегии
        if not response_parts and strategy in self.BASE_TEMPLATES:
            template = random.choice(self.BASE_TEMPLATES[strategy])
            for key, val in ctx.items():
                template = template.replace(f"{{{key}}}", str(val))
            response_parts.append(template)

        # Подсказки из концепций
        for concept in relevant_concepts:
            if concept.response_hints and random.random() < 0.4:
                hint = random.choice(concept.response_hints)
                response_parts.append(hint.format(**ctx) if '{' in hint else hint)

        # Сборка ответа
        if response_parts:
            parts = random.sample(response_parts, min(2, len(response_parts)))
            answer = ' '.join(parts)
        else:
            answer = random.choice([
                f"Я анализирую: {user_input[:30]}{'...' if len(user_input) > 30 else ''}",
                "Интересный запрос. Я учусь на каждом диалоге.",
                "Запомнил. Что ещё хотите обсудить?",
                "Понял вас. Продолжайте, я слушаю.",
            ])

        # Добавляем "личность"
        if confidence > 0.7 and random.random() < 0.3:
            experience = len(self.rules) + len(self.concepts)
            if experience > 5:
                answer += f" (на основе {experience} уроков)"

        self.context.append({'user': user_input, 'agent': answer})
        return answer.strip()

    def learn_from_interaction(self, user_input: str, response: str,
                               strategy: str, success: bool):
        keywords = self._extract_keywords(user_input)
        if not keywords:
            return

        templates = [response]
        self.add_rule(keywords[:4], strategy, templates, success)

        if success:
            topic = self._infer_topic(user_input)
            if topic:
                self.add_concept(topic, keywords, [
                    f"На вопрос о '{topic}' хорошо работает: {response[:50]}..."
                ])

    def _infer_topic(self, text: str) -> Optional[str]:
        keywords = self._extract_keywords(text)
        if not keywords:
            return None

        if any(w in keywords for w in ['привет', 'здравствуй', 'хай', 'hello', 'hi']):
            return 'greetings'
        if any(w in keywords for w in ['как', 'почему', 'что', 'зачем', 'где', 'когда']):
            return 'questions'
        if any(w in keywords for w in ['рад', 'груст', 'устал', 'счастлив', 'злой', 'нрав']):
            return 'emotions'
        if any(w in keywords for w in ['уч', 'зна', 'пон', 'объясн', 'помог']):
            return 'learning'

        return '_'.join(keywords[:2]) if len(keywords) >= 2 else keywords[0]

    def get_stats(self) -> Dict:
        return {
            'rules': len(self.rules),
            'concepts': len(self.concepts),
            'context_len': len(self.context),
            'avg_confidence': sum(r.confidence for r in self.rules) / len(self.rules) if self.rules else 0
        }

    def save(self, path: Path):
        data = {
            'rules': [r.to_dict() for r in self.rules],
            'concepts': {k: v.to_dict() for k, v in self.concepts.items()},
            'context': list(self.context),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: Path):
        if not path.exists():
            return
        try:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
            self.rules = [ResponseRule.from_dict(r) for r in data.get('rules', [])]
            self.concepts = {k: SemanticConcept.from_dict(v)
                             for k, v in data.get('concepts', {}).items()}
            self.context = deque(data.get('context', []), maxlen=CONFIG.context_window)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# 🧠 СИСТЕМА ОБУЧЕНИЯ
# ═══════════════════════════════════════════════════════════════

class LearningSystem:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.nn = StrategyNN()
        self.generator = ResponseGenerator()
        self.experiences: deque = deque(maxlen=500)
        self.step = 0
        self._load()

    def _vectorize(self, text: str) -> List[float]:
        words = re.findall(r'[а-яa-z]{2,}', text.lower(), re.I)
        vec = [0.0] * 64
        for w in words:
            vec[hash(w) % 64] += 1.0
        norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
        return [v / norm for v in vec] if norm > 0 else [0.0] * 64

    def process(self, user_input: str) -> str:
        state = self._vectorize(user_input)
        strategy, confidence = self.nn.predict_strategy(state)
        response = self.generator.generate(user_input, strategy, confidence)
        return response

    def learn(self, user_input: str, response: str, success: bool) -> float:
        state = self._vectorize(user_input)
        ideal_strategy = self._infer_ideal_strategy(user_input, response, success)
        ideal_idx = StrategyNN.STRATEGIES.index(ideal_strategy)

        reward = 0.7 if success else 0.3
        reward += 0.2 if len(response) > 20 else 0
        reward += 0.1 if any(c.isalpha() for c in response[-10:]) else 0
        reward = min(1.0, reward)

        loss = self.nn.train(state, ideal_idx, reward)
        self.step += 1

        strategy, _ = self.nn.predict_strategy(state)
        self.generator.learn_from_interaction(user_input, response, strategy, success)

        self.experiences.append({
            'input': user_input, 'response': response,
            'success': success, 'loss': loss
        })

        return loss

    def _infer_ideal_strategy(self, inp: str, resp: str, success: bool) -> str:
        if not success:
            return 'clarify' if '?' in inp else 'support'

        if '?' in resp:
            return 'question'
        if any(w in resp.lower() for w in ['понимаю', 'согласен', 'верно', 'да', 'точно']):
            return 'support'
        if any(w in resp.lower() for w in ['мне кажется', 'думаю', 'по-моему', 'я считаю']):
            return 'opinion'
        if len(resp) < 15:
            return 'silence'
        if '!' in resp or any(w in resp.lower() for w in ['понимаю', 'сочувствую']):
            return 'empathy'

        if '?' in inp:
            return 'fact' if any(w in inp.lower() for w in ['что', 'как', 'почему', 'где']) else 'question'

        return 'support'

    def get_stats(self) -> Dict:
        return {
            'experiences': len(self.experiences),
            'rules': len(self.generator.rules),
            'concepts': len(self.generator.concepts),
            'steps': self.step,
            'loss': self.nn.get_avg_loss(),
            'generator': self.generator.get_stats()
        }

    def _save(self):
        base = CONFIG.base_dir / 'knowledge'
        self.nn.save(base / f'{self.agent_id}_nn.gz')
        self.generator.save(base / f'{self.agent_id}_gen.gz')

    def _load(self):
        base = CONFIG.base_dir / 'knowledge'
        nn_path = base / f'{self.agent_id}_nn.gz'
        gen_path = base / f'{self.agent_id}_gen.gz'
        if nn_path.exists():
            try:
                self.nn = StrategyNN.load(nn_path)
            except Exception:
                pass
        if gen_path.exists():
            try:
                self.generator.load(gen_path)
            except Exception:
                pass
        if self.generator.rules or self.generator.concepts:
            print(f"  📚 Загружено: {len(self.generator.rules)} правил, {len(self.generator.concepts)} концепций")


# ═══════════════════════════════════════════════════════════════
# 🏆 АГЕНТ
# ═══════════════════════════════════════════════════════════════

class Agent:
    def __init__(self, agent_id: str = 'main'):
        self.id = agent_id
        self.learning = LearningSystem(agent_id)
        self.count = 0
        self.start = time.time()
        print(f"🚀 Агент '{agent_id}' готов к диалогу")

    async def respond(self, user_input: str) -> str:
        self.count += 1
        return self.learning.process(user_input)

    def evaluate(self, user_input: str, response: str) -> bool:
        score = 0
        if 20 < len(response) < 300:
            score += 1
        inp_words = set(re.findall(r'[а-яa-z]{3,}', user_input.lower(), re.I))
        resp_words = set(re.findall(r'[а-яa-z]{3,}', response.lower(), re.I))
        if inp_words and resp_words:
            if len(inp_words & resp_words) >= 1:
                score += 1
        if any(c in response for c in '.!?') or len(response) > 10:
            score += 1
        return score >= 2

    async def learn_from(self, user_input: str, response: str) -> Tuple[bool, float]:
        success = self.evaluate(user_input, response)
        loss = self.learning.learn(user_input, response, success)
        return success, loss

    def get_status(self) -> Dict:
        return {
            'messages': self.count,
            'uptime_min': (time.time() - self.start) / 60,
            'learning': self.learning.get_stats()
        }

    async def save(self):
        self.learning._save()
        print(f"\n💾 Состояние сохранено")


# ═══════════════════════════════════════════════════════════════
# 💬 ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════

class ChatUI:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.running = True

    async def run(self):
        print(f"""
╔═══════════════════════════════════════════════════╗
║  🧠 AGI v6.1 — ОСОЗНАННЫЕ ОТВЕТЫ (FIXED)         ║
╚═══════════════════════════════════════════════════╝
💡 Команды: /stats  /save  /rules  /quit
───────────────────────────────────────────────────
""")

        while self.running:
            try:
                user_input = input(f"\n[#{self.agent.count + 1}] Вы: ").strip()
                if not user_input:
                    continue

                if user_input.startswith('/'):
                    await self._cmd(user_input)
                    continue

                response = await self.agent.respond(user_input)
                print(f"\n🤖 {response}")

                success, loss = await self.agent.learn_from(user_input, response)

                stats = self.agent.learning.get_stats()
                if stats['steps'] % 3 == 0 and stats['steps'] > 0:
                    print(f"   📈 Loss: {loss:.3f} | Правила: {stats['rules']}")

            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Сохраняю...")
                await self.agent.save()
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()

    async def _cmd(self, cmd: str):
        c = cmd.lower().strip()
        if c in ['/quit', '/exit', '/q']:
            self.running = False
        elif c == '/stats':
            s = self.agent.get_status()
            l = s['learning']
            print(f"""
📊 Статистика:
   Сообщений: {s['messages']} | Время: {s['uptime_min']:.1f} мин
   \n🧠 Обучение:
   Опыт: {l['experiences']} | Шаги NN: {l['steps']}
   Правила: {l['rules']} | Концепции: {l['concepts']}
   Loss: {l['loss']:.4f} (меньше = лучше)
""")
        elif c == '/save':
            await self.agent.save()
            print("✅ Сохранено!")
        elif c == '/rules':
            rules = sorted(self.agent.learning.generator.rules,
                           key=lambda r: r.confidence, reverse=True)[:5]
            if rules:
                print("\n📜 Топ правила:")
                for i, r in enumerate(rules, 1):
                    kw = ', '.join(r.keywords[:3])
                    print(f"   {i}. [{r.confidence:.0%}] {{{kw}}} → {r.strategy}")
            else:
                print("  Пока нет правил — начните диалог!")
        elif c == '/help':
            print("\n📋 Команды:\n   /stats  /save  /rules  /quit")
        else:
            print("  ❓ /help для списка команд")


# ═══════════════════════════════════════════════════════════════
# 🚀 ЗАПУСК
# ═══════════════════════════════════════════════════════════════

async def main():
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🧠 AGI AGENT v6.1 FIXED — ОСОЗНАННЫЕ ОТВЕТЫ            ║
║  ──────────────────────────────────────────             ║
║  ✅ Исправлено: "truth value of array" ошибка           ║
║  ✅ Чистый Python — стабильная работа                   ║
║  ✅ Агент РЕАЛЬНО ОТВЕЧАЕТ осмысленно                   ║
╚══════════════════════════════════════════════════════════╝
""")

    agent = Agent('main')

    try:
        await ChatUI(agent).run()
        await agent.save()

        s = agent.get_status()
        print(f"""
═══════════════════════════════════════════════════
📈 Сессия завершена:
   Сообщений: {s['messages']}
   Выучено правил: {s['learning']['rules']}
   Концепций: {s['learning']['concepts']}
   Loss: {s['learning']['loss']:.4f}
═══════════════════════════════════════════════════
""")

    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 До свидания!")
        sys.exit(0)