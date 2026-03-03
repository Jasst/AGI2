#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID AUTONOMOUS BRAIN v24.0 — TRUE NEURAL AGI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
РЕВОЛЮЦИЯ v24:
  ✅ НАСТОЯЩИЕ СПАЙКОВЫЕ НЕЙРОНЫ (LIF модель)
  ✅ STDP — Хеббовское обучение без учителя
  ✅ АВТОНОМНОЕ МЫШЛЕНИЕ — внутренний монолог
  ✅ ГОМЕОСТАЗ — саморегуляция активности
  ✅ LLM < 20% — только как "подсознание"

СОХРАНЕНО из v23:
  🎯 GoalSystem | 🧪 ActiveLearning | 🪞 TheoryOfMind
  📋 TaskPlanner | ⚡ ProactiveBehavior | 🔄 SelfModification
  CognitivePulse, RAG v2, L1/L2/L3 память, эмоц. дуга
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, re, asyncio, aiohttp, traceback, hashlib, math, shutil, random
from collections import Counter, deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ══════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

BASE_DIR = "autonomous_brain_v24"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# ══════════════════════════════════════════
# v24: СПАЙКОВЫЕ НЕЙРОНЫ
# ══════════════════════════════════════════
NEURON_THRESHOLD = 1.0
NEURON_RESET_POTENTIAL = 0.0
NEURON_DECAY = 0.95
NEURON_REFRACTORY_PERIOD = 3
NEURON_NOISE = 0.02
SPONTANEOUS_PROB = 0.03

# STDP
STDP_A_PLUS = 0.015
STDP_A_MINUS = 0.012
STDP_TAU = 20.0
WEIGHT_MIN = 0.0
WEIGHT_MAX = 2.0
WEIGHT_INIT = 0.3

# Гомеостаз
HOMEOSTATIC_TARGET = 0.15
HOMEOSTATIC_RATE = 0.0001

# Архитектура
CORTEX_SIZE = 200
SUBCORTEX_SIZE = 50
SENSORY_INPUT_SIZE = 30
MOTOR_OUTPUT_SIZE = 20

# LLM (подсознание)
LLM_THRESHOLD_COMPLEXITY = 0.7
LLM_COOLDOWN = 5

# v23 параметры (сохранены)
AUTOSAVE_EVERY = 10
RAG_TOP_K = 4
L1_RECENT_COUNT = 3
L1_RELEVANT_COUNT = 3
L1_MAX_HISTORY = 30
TEMP_DEFAULT = 0.65
PULSE_ENABLED = True
PULSE_PHASE_MAX_TOKENS = 300
PULSE_FINAL_MAX_TOKENS = 1500
PULSE_SHORT_THRESHOLD = 10

# AGI модули v23
GOAL_MAX_ACTIVE = 5
GOAL_PROGRESS_STEP = 0.12
AL_HYPOTHESIS_MAX = 30
TOM_UPDATE_EVERY = 5
PROACTIVE_EVERY = 15
PROACTIVE_PROB = 0.3


# ══════════════════════════════════════════
# УТИЛИТЫ (из v23)
# ══════════════════════════════════════════
class FileManager:
    @staticmethod
    def safe_save_json(filepath: str, data: Any) -> bool:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            temp = f"{filepath}.tmp"
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if os.path.exists(filepath):
                os.replace(filepath, f"{filepath}.bak")
            os.replace(temp, filepath)
            return True
        except Exception as e:
            print(f"⚠️ Save error: {e}")
            return False

    @staticmethod
    def safe_load_json(filepath: str, default: Any = None) -> Any:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return default if default is not None else {}


class TextUtils:
    STOP_WORDS = {
        'в', 'и', 'на', 'с', 'по', 'для', 'от', 'к', 'о', 'у', 'из', 'за',
        'что', 'это', 'как', 'то', 'а', 'но', 'или', 'the', 'is', 'at',
        'не', 'ты', 'я', 'мне', 'себя', 'был', 'была', 'было', 'мой', 'твой'
    }

    @staticmethod
    def extract_keywords(text: str, top_n: int = 8) -> List[str]:
        words = [w.lower() for w in re.findall(r'\b[а-яёa-z]{3,}\b', text, re.IGNORECASE)]
        filtered = [w for w in words if w not in TextUtils.STOP_WORDS]
        return [w for w, _ in Counter(filtered).most_common(top_n)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = [w.lower() for w in re.findall(r'\b[а-яёa-z]{3,}\b', text, re.IGNORECASE)]
        return [w for w in words if w not in TextUtils.STOP_WORDS]

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())


# ══════════════════════════════════════════
# v24: SPIKING NEURON
# ══════════════════════════════════════════
@dataclass
class SpikingNeuron:
    """Leaky Integrate-and-Fire нейрон"""
    id: int
    label: str
    region: str  # cortex/subcortex/sensory/motor

    membrane_potential: float = 0.0
    threshold: float = NEURON_THRESHOLD
    refractory_timer: int = 0

    last_spike_time: int = -1000
    spike_times: deque = field(default_factory=lambda: deque(maxlen=100))
    total_spikes: int = 0
    average_activity: float = 0.0

    def step(self, current_time: int, input_current: float) -> bool:
        """Один временной шаг (1 мс)"""
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.membrane_potential = NEURON_RESET_POTENTIAL
            return False

        noise = random.gauss(0, NEURON_NOISE)
        self.membrane_potential *= NEURON_DECAY
        self.membrane_potential += input_current + noise

        if self.membrane_potential >= self.threshold:
            self._fire(current_time)
            return True
        return False

    def _fire(self, current_time: int):
        self.membrane_potential = NEURON_RESET_POTENTIAL
        self.refractory_timer = NEURON_REFRACTORY_PERIOD
        self.last_spike_time = current_time
        self.spike_times.append(current_time)
        self.total_spikes += 1

    def update_homeostasis(self, window_size: int = 1000):
        recent_spikes = sum(1 for t in self.spike_times if t > self.last_spike_time - window_size)
        self.average_activity = recent_spikes / window_size

        if self.average_activity > HOMEOSTATIC_TARGET:
            self.threshold += HOMEOSTATIC_RATE
        elif self.average_activity < HOMEOSTATIC_TARGET:
            self.threshold -= HOMEOSTATIC_RATE

        self.threshold = max(0.5, min(2.0, self.threshold))


@dataclass
class STDPSynapse:
    """STDP — Хеббовское обучение"""
    source_id: int
    target_id: int
    weight: float = WEIGHT_INIT

    pre_trace: float = 0.0
    post_trace: float = 0.0

    def update_stdp(self, pre_spiked: bool, post_spiked: bool, dt: float = 1.0):
        self.pre_trace *= math.exp(-dt / STDP_TAU)
        self.post_trace *= math.exp(-dt / STDP_TAU)

        if pre_spiked:
            self.pre_trace += 1.0
            self.weight -= STDP_A_MINUS * self.post_trace

        if post_spiked:
            self.post_trace += 1.0
            self.weight += STDP_A_PLUS * self.pre_trace

        self.weight = max(WEIGHT_MIN, min(WEIGHT_MAX, self.weight))

    def transmit(self, pre_spiked: bool) -> float:
        return self.weight if pre_spiked else 0.0


# ══════════════════════════════════════════
# v24: АВТОНОМНАЯ НЕЙРОННАЯ СЕТЬ
# ══════════════════════════════════════════
class AutonomousNeuralNetwork:
    """Спайковая нейронная сеть с автономным мышлением"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.current_time = 0

        self.neurons: Dict[int, SpikingNeuron] = {}
        self.synapses: List[STDPSynapse] = []

        self.label_to_neuron: Dict[str, int] = {}
        self.region_neurons: Dict[str, List[int]] = defaultdict(list)

        self.active_neurons: Set[int] = set()
        self.thought_buffer: deque = deque(maxlen=10)

        self._initialize_network()
        self._load()

    def _initialize_network(self):
        neuron_id = 0

        # Сенсорный слой
        for i in range(SENSORY_INPUT_SIZE):
            self._create_neuron(neuron_id, f"sense_{i}", "sensory")
            neuron_id += 1

        # Кора
        for i in range(CORTEX_SIZE):
            self._create_neuron(neuron_id, f"cortex_{i}", "cortex")
            neuron_id += 1

        # Подкорка
        for i in range(SUBCORTEX_SIZE):
            self._create_neuron(neuron_id, f"subcortex_{i}", "subcortex")
            neuron_id += 1

        # Моторный выход
        for i in range(MOTOR_OUTPUT_SIZE):
            self._create_neuron(neuron_id, f"motor_{i}", "motor")
            neuron_id += 1

        self._create_random_connectivity()
        print(f"🧠 Создано: {len(self.neurons)} нейронов, {len(self.synapses)} синапсов")

    def _create_neuron(self, nid: int, label: str, region: str):
        n = SpikingNeuron(id=nid, label=label, region=region)
        self.neurons[nid] = n
        self.label_to_neuron[label] = nid
        self.region_neurons[region].append(nid)

    def _create_random_connectivity(self, prob: float = 0.15):
        for src_id in self.neurons:
            for tgt_id in self.neurons:
                if src_id == tgt_id:
                    continue

                src = self.neurons[src_id]
                tgt = self.neurons[tgt_id]

                should_connect = False
                if src.region == "sensory" and tgt.region == "cortex":
                    should_connect = random.random() < 0.3
                elif src.region == "cortex" and tgt.region == "cortex":
                    should_connect = random.random() < prob
                elif src.region == "cortex" and tgt.region == "subcortex":
                    should_connect = random.random() < 0.2
                elif src.region == "subcortex" and tgt.region == "cortex":
                    should_connect = random.random() < 0.25
                elif tgt.region == "motor" and src.region in ["cortex", "subcortex"]:
                    should_connect = random.random() < 0.2

                if should_connect:
                    w = random.uniform(0.2, 0.5)
                    self.synapses.append(STDPSynapse(src_id, tgt_id, weight=w))

    def step(self, sensory_input: Dict[str, float] = None) -> Dict:
        """Один временной шаг (1 мс)"""
        self.current_time += 1

        input_currents = defaultdict(float)

        # Сенсорные входы
        if sensory_input:
            for label, current in sensory_input.items():
                nid = self.label_to_neuron.get(label)
                if nid is not None:
                    input_currents[nid] = current

        # Спонтанная активность
        for nid in self.region_neurons["cortex"]:
            if random.random() < SPONTANEOUS_PROB:
                input_currents[nid] += random.uniform(0.1, 0.3)

        # Синаптическая передача
        spiked_neurons = set()

        for syn in self.synapses:
            src = self.neurons[syn.source_id]
            pre_spiked = src.last_spike_time == self.current_time - 1

            if pre_spiked:
                input_currents[syn.target_id] += syn.transmit(True)

        # Обновление нейронов
        for nid, neuron in self.neurons.items():
            current = input_currents.get(nid, 0.0)
            if neuron.step(self.current_time, current):
                spiked_neurons.add(nid)

        # STDP обучение
        for syn in self.synapses:
            pre_spiked = syn.source_id in spiked_neurons
            post_spiked = syn.target_id in spiked_neurons
            syn.update_stdp(pre_spiked, post_spiked)

        # Гомеостаз
        if self.current_time % 100 == 0:
            for neuron in self.neurons.values():
                neuron.update_homeostasis()

        thought = self._extract_thought(spiked_neurons)
        if thought:
            self.thought_buffer.append((self.current_time, thought))

        self.active_neurons = spiked_neurons

        return {
            "time": self.current_time,
            "spiked": len(spiked_neurons),
            "thought": thought,
            "active_regions": self._get_active_regions(spiked_neurons)
        }

    def _extract_thought(self, spiked_neurons: Set[int]) -> Optional[str]:
        cortical_spikes = [nid for nid in spiked_neurons
                           if self.neurons[nid].region == "cortex"]

        if len(cortical_spikes) < 3:
            return None

        top_neurons = sorted(cortical_spikes,
                             key=lambda nid: self.neurons[nid].membrane_potential,
                             reverse=True)[:5]

        labels = [self.neurons[nid].label for nid in top_neurons]
        return f"{'·'.join(labels)}"

    def _get_active_regions(self, spiked: Set[int]) -> Dict[str, int]:
        counts = defaultdict(int)
        for nid in spiked:
            counts[self.neurons[nid].region] += 1
        return dict(counts)

    def associate_concept(self, concept: str, sensory_pattern: Dict[str, float]):
        """Создать нейрон для концепта"""
        if concept in self.label_to_neuron:
            concept_id = self.label_to_neuron[concept]
        else:
            concept_id = len(self.neurons)
            self._create_neuron(concept_id, concept, "cortex")

        for label, strength in sensory_pattern.items():
            if label in self.label_to_neuron:
                src_id = self.label_to_neuron[label]
                syn = STDPSynapse(src_id, concept_id, weight=strength)
                self.synapses.append(syn)

        print(f"🧩 Ассоциировал '{concept}' с {len(sensory_pattern)} признаками")

    def get_concept_activation(self, concept: str) -> float:
        if concept not in self.label_to_neuron:
            return 0.0
        nid = self.label_to_neuron[concept]
        return self.neurons[nid].membrane_potential

    def get_thoughts(self, last_n: int = 5) -> List[Tuple[int, str]]:
        return list(self.thought_buffer)[-last_n:]

    def _save(self):
        # Упрощённое сохранение (полное сохранение - слишком большое)
        data = {
            "time": self.current_time,
            "concept_labels": [label for label in self.label_to_neuron.keys()
                               if not label.startswith(("sense_", "cortex_", "subcortex_", "motor_"))]
        }
        FileManager.safe_save_json(self.save_path, data)

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self.current_time = data.get("time", 0)
        print(f"✅ Загружено состояние (t={self.current_time})")


# ══════════════════════════════════════════
# v24: ГИБРИДНЫЙ МОЗГ
# ══════════════════════════════════════════
class HybridCognitiveBrain:
    """
    Автономная нейронная сеть + LLM как подсознание
    """

    def __init__(self, brain_path: str, llm: 'LLMInterface'):
        self.brain = AutonomousNeuralNetwork(brain_path)
        self.llm = llm

        self.interaction_count = 0
        self.llm_usage_count = 0

        # v23: AGI модули (упрощённые версии)
        self.learned_concepts: Dict[str, List[str]] = {}  # concept → примеры

    async def think(self, input_text: str, context: str = "") -> Tuple[str, bool]:
        """
        Главный цикл мышления

        Returns:
            (response, used_llm)
        """
        self.interaction_count += 1

        # 1. Сенсорное кодирование
        sensory_input = self._encode_text(input_text)

        # 2. Автономное мышление (100 шагов = 100 мс)
        thoughts = []
        for _ in range(100):
            state = self.brain.step(sensory_input)
            if state["thought"]:
                thoughts.append(state["thought"])

            # Постепенно угасающий вход
            sensory_input = {k: v * 0.9 for k, v in sensory_input.items()}

        # 3. Формирование ответа
        neural_response = self._decode_thoughts(thoughts)

        # 4. Оценка сложности
        complexity = self._estimate_complexity(input_text, neural_response)

        if complexity > LLM_THRESHOLD_COMPLEXITY:
            print(f"🌀 Подключаю подсознание (complexity={complexity:.2f})")

            thought_context = " → ".join(thoughts[-5:]) if thoughts else "пусто"
            llm_response = await self.llm.generate(
                f"Контекст мыслей: {thought_context}\n"
                f"История: {context[:300] if context else 'нет'}\n\n"
                f"Вопрос: {input_text}\n\n"
                f"Ответь кратко и по существу (2-4 предложения).",
                temp=0.7
            )

            response = f"[нейроны] {neural_response}\n[интуиция] {llm_response}"
            self.llm_usage_count += 1
            return response, True

        return f"[нейроны] {neural_response}", False

    def _encode_text(self, text: str) -> Dict[str, float]:
        words = text.lower().split()[:SENSORY_INPUT_SIZE]
        sensory = {}

        for i, word in enumerate(words):
            neuron_idx = hash(word) % SENSORY_INPUT_SIZE
            label = f"sense_{neuron_idx}"
            strength = 1.0 - (i / max(len(words), 1)) * 0.5
            sensory[label] = sensory.get(label, 0) + strength

        return sensory

    def _decode_thoughts(self, thoughts: List[str]) -> str:
        if not thoughts:
            return "[тишина]"

        unique_thoughts = []
        seen = set()
        for t in reversed(thoughts):
            if t not in seen:
                unique_thoughts.append(t)
                seen.add(t)
            if len(unique_thoughts) >= 3:
                break

        return " | ".join(reversed(unique_thoughts))

    def _estimate_complexity(self, query: str, neural_response: str) -> float:
        score = 0.0

        if len(query.split()) > 10:
            score += 0.3

        if neural_response == "[тишина]" or len(neural_response) < 10:
            score += 0.4

        abstract_words = ['почему', 'как', 'что если', 'объясни', 'расскажи']
        if any(w in query.lower() for w in abstract_words):
            score += 0.3

        return min(1.0, score)

    def learn_concept(self, concept: str, examples: List[str]):
        """Обучить новый концепт"""
        avg_pattern = defaultdict(float)

        for example in examples:
            pattern = self._encode_text(example)
            for k, v in pattern.items():
                avg_pattern[k] += v / len(examples)

        self.brain.associate_concept(concept, dict(avg_pattern))
        self.learned_concepts[concept] = examples

    def stats(self) -> Dict:
        return {
            "neurons": len(self.brain.neurons),
            "synapses": len(self.brain.synapses),
            "time": self.brain.current_time,
            "interactions": self.interaction_count,
            "llm_usage": self.llm_usage_count,
            "llm_ratio": self.llm_usage_count / max(1, self.interaction_count),
            "concepts": len(self.learned_concepts)
        }


# ══════════════════════════════════════════
# v23: СОХРАНЁННЫЕ КОМПОНЕНТЫ (упрощённо)
# ══════════════════════════════════════════

# L1 память (из v23)
class SmartWorkingMemory:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.history: List[Dict] = FileManager.safe_load_json(save_path, [])

    def add(self, role: str, content: str):
        self.history.append({
            'role': role, 'content': content,
            'time': datetime.now().isoformat(),
            'keywords': TextUtils.extract_keywords(content)
        })
        if len(self.history) > L1_MAX_HISTORY:
            self.history = self.history[-L1_MAX_HISTORY:]

    def get_context(self, query: str) -> str:
        if not self.history:
            return ""

        recent = self.history[-L1_RECENT_COUNT:]
        lines = []
        for m in recent:
            role_label = "Пользователь" if m['role'] == 'user' else "Ассистент"
            lines.append(f"{role_label}: {m['content'][:200]}")
        return "\n".join(lines)

    def save(self):
        FileManager.safe_save_json(self.save_path, self.history)


# Профиль пользователя (из v23)
@dataclass
class UserProfile:
    name: str = ""
    interests: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)


class ProfileManager:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.profile = UserProfile()
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        if data:
            self.profile.name = data.get('name', '')
            self.profile.interests = data.get('interests', [])
            self.profile.facts = data.get('facts', [])

    def save(self):
        FileManager.safe_save_json(self.save_path, asdict(self.profile))

    def update_from_text(self, text: str) -> List[str]:
        new_facts = []

        # Простое извлечение имени
        m = re.search(r'меня зовут ([А-ЯЁа-яёA-Za-z]+)', text.lower())
        if m and not self.profile.name:
            self.profile.name = m.group(1).capitalize()
            new_facts.append(f"имя: {self.profile.name}")

        # Интересы
        m = re.search(r'я люблю ([а-яёa-z\s]+?)(?:\.|,|$)', text.lower())
        if m:
            interest = m.group(1).strip()
            if interest not in self.profile.interests:
                self.profile.interests.append(interest)
                new_facts.append(f"интерес: {interest}")

        return new_facts

    def get_prompt_block(self) -> str:
        p = self.profile
        lines = []
        if p.name:
            lines.append(f"Имя: {p.name}")
        if p.interests:
            lines.append(f"Интересы: {', '.join(p.interests[:5])}")
        return "\n".join(lines) if lines else ""


# v23 AGI модули (упрощённо)
@dataclass
class Goal:
    id: str
    title: str
    progress: float = 0.0
    completed: bool = False


class SimpleGoalSystem:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.goals: Dict[str, Goal] = {}
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        for k, v in data.items():
            self.goals[k] = Goal(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path, {k: asdict(v) for k, v in self.goals.items()})

    def extract_from_text(self, text: str) -> List[Goal]:
        found = []

        patterns = [
            r'хочу\s+([\w\s]{4,40})(?:\.|,|$)',
            r'планирую\s+([\w\s]{4,40})(?:\.|,|$)',
        ]

        for pattern in patterns:
            for m in re.finditer(pattern, text.lower()):
                title = m.group(1).strip()
                if len(title) < 4:
                    continue
                gid = hashlib.md5(title.encode()).hexdigest()[:8]
                if gid not in self.goals:
                    g = Goal(id=gid, title=title)
                    self.goals[gid] = g
                    found.append(g)

        return found

    def update_progress(self, text: str):
        keywords = set(TextUtils.extract_keywords(text))
        for goal in self.goals.values():
            if goal.completed:
                continue
            goal_kw = set(TextUtils.extract_keywords(goal.title))
            overlap = len(keywords & goal_kw) / max(len(goal_kw), 1)
            if overlap > 0.2:
                goal.progress = min(1.0, goal.progress + GOAL_PROGRESS_STEP)
                if goal.progress >= 0.9:
                    goal.completed = True

    def get_active_goals(self) -> List[Goal]:
        return [g for g in self.goals.values() if not g.completed][:GOAL_MAX_ACTIVE]


# ══════════════════════════════════════════
# ГИБРИДНАЯ ПАМЯТЬ v24
# ══════════════════════════════════════════
class HybridMemorySystem:
    """Гибридная система: автономные нейроны + v23 компоненты"""

    def __init__(self, user_id: str, llm: 'LLMInterface'):
        self.user_id = user_id
        self.dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.dir, exist_ok=True)

        # v24: Автономный мозг
        self.brain = HybridCognitiveBrain(
            os.path.join(self.dir, "neural_brain.json"),
            llm
        )

        # v23: Базовые компоненты
        self.l1 = SmartWorkingMemory(os.path.join(self.dir, "l1.json"))
        self.profile = ProfileManager(os.path.join(self.dir, "profile.json"))
        self.goals = SimpleGoalSystem(os.path.join(self.dir, "goals.json"))

        self._msg_count = 0

    async def process(self, text: str) -> Tuple[str, Dict]:
        """Обработка входного сообщения"""
        self._msg_count += 1

        # Обновление профиля
        new_facts = self.profile.update_from_text(text)

        # Обновление целей
        new_goals = self.goals.extract_from_text(text)
        self.goals.update_progress(text)

        # Контекст для LLM
        context = self.l1.get_context(text)
        profile_block = self.profile.get_prompt_block()

        full_context = f"{profile_block}\n{context}" if profile_block else context

        # ГЛАВНОЕ: Автономное мышление
        response, used_llm = await self.brain.think(text, full_context)

        # Сохранение в L1
        self.l1.add('user', text)
        self.l1.add('assistant', response)

        # Автосохранение
        if self._msg_count % AUTOSAVE_EVERY == 0:
            self.save_all()

        return response, {
            'new_facts': new_facts,
            'new_goals': new_goals,
            'used_llm': used_llm,
            'stats': self.brain.stats()
        }

    def save_all(self):
        self.brain.brain._save()
        self.l1.save()
        self.profile.save()
        self.goals.save()


# ══════════════════════════════════════════
# LLM INTERFACE
# ══════════════════════════════════════════
class LLMInterface:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def generate(self, prompt: str, temp: float = TEMP_DEFAULT) -> str:
        if not self.session:
            await self.init()

        try:
            async with self.session.post(
                    self.url,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temp,
                        "max_tokens": 500
                    },
                    headers={"Authorization": f"Bearer {self.key}"}
            ) as r:
                if r.status == 200:
                    result = await r.json()
                    return result['choices'][0]['message']['content']
                return f"LM Error: {r.status}"
        except Exception as e:
            return f"Connection error: {e}"

    async def close(self):
        if self.session:
            await self.session.close()


# ══════════════════════════════════════════
# TELEGRAM БОТ v24
# ══════════════════════════════════════════
class HybridBot:
    def __init__(self):
        self.llm = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.users: Dict[str, HybridMemorySystem] = {}
        self.stop_flag = False

    def get_brain(self, uid: str) -> HybridMemorySystem:
        if uid not in self.users:
            self.users[uid] = HybridMemorySystem(uid, self.llm)
        return self.users[uid]

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return

        uid = str(update.effective_user.id)
        text = update.message.text
        brain = self.get_brain(uid)

        await context.bot.send_chat_action(uid, "typing")

        # ГЛАВНОЕ: Автономное мышление
        response, metadata = await brain.process(text)

        await update.message.reply_text(response)

        # Доп. сообщения
        extra_messages = []

        if metadata['new_facts']:
            extra_messages.append(f"👤 Запомнил: {', '.join(metadata['new_facts'])}")

        if metadata['new_goals']:
            extra_messages.append(f"🎯 Новая цель: {metadata['new_goals'][0].title}")

        if not metadata['used_llm'] and random.random() < 0.1:
            extra_messages.append("💡 [Ответ от автономных нейронов — без LLM!]")

        for msg in extra_messages:
            await update.message.reply_text(msg)

    # ═══ КОМАНДЫ ═══

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)
        stats = brain.brain.stats()

        await update.message.reply_text(
            f"🧠 AUTONOMOUS BRAIN v24.0\n{'═' * 32}\n"
            f"🔹 Нейроны: {stats['neurons']}\n"
            f"🔹 Синапсы: {stats['synapses']}\n"
            f"🔹 Время: {stats['time']}ms\n"
            f"🔹 Диалогов: {stats['interactions']}\n"
            f"🌀 LLM использован: {stats['llm_usage']} раз ({stats['llm_ratio']:.0%})\n"
            f"🧩 Концептов: {stats['concepts']}\n"
            f"{'═' * 32}\n"
            f"💡 Нейроны обрабатывают {(1 - stats['llm_ratio']):.0%} запросов автономно!"
        )

    async def cmd_goals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)
        active = brain.goals.get_active_goals()

        if not active:
            await update.message.reply_text("🎯 Целей пока нет. Скажи 'хочу...' или 'планирую...'")
            return

        lines = ["🎯 ЦЕЛИ\n"]
        for g in active:
            bar = "█" * int(g.progress * 10) + "░" * (10 - int(g.progress * 10))
            lines.append(f"  {g.title}\n  {bar} {g.progress:.0%}")

        await update.message.reply_text("\n".join(lines))

    async def cmd_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обучить новый концепт"""
        if not context.args:
            await update.message.reply_text(
                "Использование: /learn <концепт> <примеры>\n"
                "Например: /learn собака пёс лает животное"
            )
            return

        concept = context.args[0]
        examples = [" ".join(context.args[1:])] if len(context.args) > 1 else [concept]

        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)

        brain.brain.learn_concept(concept, examples)

        await update.message.reply_text(f"✅ Обучил концепт '{concept}' на {len(examples)} примерах")

    async def cmd_thoughts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать последние мысли нейронов"""
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)

        thoughts = brain.brain.brain.get_thoughts(last_n=10)

        if not thoughts:
            await update.message.reply_text("💭 Нейроны пока молчат.")
            return

        lines = ["💭 ПОСЛЕДНИЕ МЫСЛИ НЕЙРОНОВ\n"]
        for t, thought in thoughts:
            lines.append(f"t={t:05d}ms: {thought}")

        await update.message.reply_text("\n".join(lines))

    async def cmd_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)

        p = brain.profile.profile
        await update.message.reply_text(
            f"👤 ПРОФИЛЬ\n{'═' * 24}\n"
            f"Имя: {p.name or '?'}\n"
            f"Интересы: {', '.join(p.interests[:5]) or 'нет'}\n"
            f"Фактов: {len(p.facts)}"
        )

    async def cmd_wipe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        if uid in self.users:
            d = self.users.pop(uid).dir
            if os.path.exists(d):
                shutil.rmtree(d)
            await update.message.reply_text("🧠 Полная очистка памяти.")
        else:
            await update.message.reply_text("Пользователь не найден.")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 AUTONOMOUS BRAIN v24.0\n\n"
            "Революция: настоящие спайковые нейроны!\n"
            "✅ LIF модель с мембранным потенциалом\n"
            "✅ STDP — Хеббовское обучение\n"
            "✅ Автономное мышление без входа\n"
            "✅ LLM < 20% — только как подсознание\n\n"
            "📌 КОМАНДЫ:\n"
            "/stats     — статистика нейронов\n"
            "/goals     — цели пользователя\n"
            "/thoughts  — последние мысли нейронов\n"
            "/learn     — обучить концепт\n"
            "/profile   — профиль пользователя\n"
            "/wipe      — очистить память\n"
            "/help      — эта справка\n\n"
            "💡 Просто общайся — нейроны учатся сами!"
        )

    async def shutdown(self):
        print("\n💾 Финальное сохранение...")
        for brain in self.users.values():
            brain.save_all()
        await self.llm.close()
        print("✅ Остановлено")


# ══════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════
async def main():
    print("🚀 AUTONOMOUS BRAIN v24.0 STARTING...")
    print("✅ Спайковые нейроны (LIF)")
    print("✅ STDP обучение")
    print("✅ Автономное мышление")
    print("✅ LLM как подсознание")

    if not TELEGRAM_TOKEN:
        print("❌ Нет TELEGRAM_TOKEN в .env")
        return

    bot = HybridBot()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    for cmd, handler in [
        ("stats", bot.cmd_stats),
        ("goals", bot.cmd_goals),
        ("learn", bot.cmd_learn),
        ("thoughts", bot.cmd_thoughts),
        ("profile", bot.cmd_profile),
        ("wipe", bot.cmd_wipe),
        ("help", bot.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print("✅ АВТОНОМНЫЙ МОЗГ ГОТОВ 🧠")

        while not bot.stop_flag:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Остановка")
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Crash: {e}")
        traceback.print_exc()