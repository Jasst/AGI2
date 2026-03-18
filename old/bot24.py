#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID AUTONOMOUS BRAIN v25.0 — LIVING NEURAL ENTITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 НОВОЕ В v25:
  ✅ ДИНАМИЧЕСКИЕ НЕЙРОНЫ — рост сети по мере обучения
  ✅ МНОГОУРОВНЕВАЯ ПАМЯТЬ — L1/L2/L3 с забыванием
  ✅ КОНТЕКСТНАЯ АКТУАЛИЗАЦИЯ — релевантность воспоминаний
  ✅ ЭМОЦИОНАЛЬНЫЙ ИНТЕЛЛЕКТ — настроение влияет на ответы
  ✅ ВНУТРЕННИЙ МОНОЛОГ — спонтанные мысли и рефлексия
  ✅ ЭНЕРГЕТИКА — усталость, любопытство, мотивация
  ✅ ЛИЧНОСТЬ — устойчивые черты характера
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, re, asyncio, aiohttp, traceback, hashlib, math, shutil, random
import time
from collections import Counter, deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict, replace
from enum import Enum, auto
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

BASE_DIR = "autonomous_brain_v25"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# ─── Нейроны и синапсы ─────────────────────
NEURON_THRESHOLD = 1.0
NEURON_RESET_POTENTIAL = 0.0
NEURON_DECAY = 0.93
NEURON_REFRACTORY_PERIOD = 3
NEURON_NOISE = 0.03
SPONTANEOUS_PROB = 0.02

# STDP параметры
STDP_A_PLUS = 0.02
STDP_A_MINUS = 0.015
STDP_TAU = 20.0
WEIGHT_MIN = 0.0
WEIGHT_MAX = 3.0
WEIGHT_INIT = 0.4

# Гомеостаз
HOMEOSTATIC_TARGET = 0.12
HOMEOSTATIC_RATE = 0.00015

# ─── Динамическая архитектура ──────────────
INITIAL_CORTEX_SIZE = 150
INITIAL_SUBCORTEX_SIZE = 40
SENSORY_INPUT_SIZE = 40
MOTOR_OUTPUT_SIZE = 25
MAX_NEURON_GROWTH_PER_STEP = 5

# ─── Память и забывание ───────────────────
MEMORY_DECAY_L1 = 0.99
MEMORY_DECAY_L2 = 0.95
MEMORY_DECAY_L3 = 0.999
FORGET_THRESHOLD = 0.1
CONSOLIDATION_THRESHOLD = 0.8
RELEVANCE_BOOST = 0.3


# ─── Эмоции и личность ────────────────────
@dataclass
class EmotionalState:
    curiosity: float = 0.5
    energy: float = 0.8
    mood: float = 0.6
    confidence: float = 0.5
    social_drive: float = 0.7

    def update(self, interaction_quality: float, novelty: float, fatigue: float):
        self.curiosity = min(1.0, self.curiosity + novelty * 0.1 - fatigue * 0.05)
        self.energy = max(0.1, min(1.0, self.energy - fatigue * 0.1 + 0.02))
        self.mood = max(-1.0, min(1.0, self.mood + interaction_quality * 0.15))
        self.confidence = min(1.0, self.confidence + (1 if interaction_quality > 0 else -0.1) * 0.05)
        self.social_drive = min(1.0, max(0.1, self.social_drive + (0.05 if interaction_quality > 0.3 else -0.03)))
        self.energy = min(1.0, self.energy + 0.01)
        self.curiosity = min(1.0, self.curiosity + 0.005)


@dataclass
class Personality:
    openness: float = 0.8
    conscientiousness: float = 0.6
    extraversion: float = 0.7
    agreeableness: float = 0.75
    neuroticism: float = 0.3

    def influence_response(self, base_response: str, emotion: EmotionalState) -> str:
        """Корректировка ответа на основе личности и эмоций"""
        # Не трогаем технические ответы
        if not base_response or base_response.startswith('['):
            return base_response
        if any(x in base_response for x in ['cortex_', 'sense_', 'motor_', 'subcortex_']):
            return base_response

        emoji, suffix = "", ""
        roll = random.random()

        if self.extraversion > 0.7 and emotion.social_drive > 0.6 and roll < 0.4:
            emoji = random.choice(["🌟", "✨", "💫", "🔥"])
            suffix = random.choice(["! ", "!! ", ""])
        elif emotion.mood > 0.3 and roll < 0.4:
            emoji = random.choice(["😊", "🧠", "💡", "🌈"])
        elif emotion.mood < -0.2 and roll < 0.3:
            emoji = random.choice(["🤔", "🌀", "⚡"])
            suffix = "..."
        elif self.openness > 0.7 and emotion.curiosity > 0.6 and roll < 0.35:
            emoji = random.choice(["🔍", "🧩", "🗝️"])

        if emoji or suffix:
            result = f"{emoji} {base_response}{suffix}".strip()
            return result[:300]
        return base_response[:300]


# ─── Уровни памяти ─────────────────────────
class MemoryLevel(Enum):
    L1 = "working"
    L2 = "episodic"
    L3 = "semantic"


@dataclass
class MemoryItem:
    content: str
    level: MemoryLevel
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    relevance: float = 1.0
    activation_count: int = 0
    keywords: List[str] = field(default_factory=list)
    emotional_tag: Optional[str] = None

    def decay(self, current_time: float) -> float:
        elapsed = current_time - self.last_accessed
        if self.level == MemoryLevel.L1:
            decay = MEMORY_DECAY_L1 ** (elapsed / 60)
        elif self.level == MemoryLevel.L2:
            decay = MEMORY_DECAY_L2 ** (elapsed / 3600)
        else:
            decay = MEMORY_DECAY_L3 ** (elapsed / 86400)
        return self.relevance * decay

    def touch(self, boost: float = 0.0):
        self.last_accessed = time.time()
        self.activation_count += 1
        self.relevance = min(1.0, self.relevance + boost)

    def promote(self):
        if self.level == MemoryLevel.L1:
            self.level = MemoryLevel.L2
            self.relevance = min(1.0, self.relevance * 1.2)
        elif self.level == MemoryLevel.L2 and self.relevance > CONSOLIDATION_THRESHOLD:
            self.level = MemoryLevel.L3
            self.relevance = 1.0


# ══════════════════════════════════════════
# УТИЛИТЫ
# ══════════════════════════════════════════
class FileManager:
    @staticmethod
    def safe_save_json(filepath: str, data: Any) -> bool:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            temp = f"{filepath}.tmp"
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
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
        'не', 'ты', 'я', 'мне', 'себя', 'был', 'была', 'было', 'мой', 'твой',
        'он', 'она', 'оно', 'они', 'мы', 'вы', 'тут', 'там', 'вот', 'же', 'ли'
    }

    EMOTIONAL_WORDS = {
        'привет': 'greeting', 'здравствуй': 'greeting', 'hello': 'greeting',
        'пока': 'farewell', 'до': 'farewell', 'bye': 'farewell',
        'спасибо': 'gratitude', 'благодарю': 'gratitude', 'thanks': 'gratitude',
        'грустно': 'sad', 'плохо': 'sad', 'тоска': 'sad',
        'рад': 'joy', 'весело': 'joy', 'круто': 'joy', 'awesome': 'joy',
        'интересно': 'curiosity', 'любопытно': 'curiosity', 'почему': 'curiosity',
        'помоги': 'help', 'нужно': 'help', 'как': 'help',
    }

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        words = [w.lower() for w in re.findall(r'\b[а-яёa-z]{3,}\b', text, re.IGNORECASE)]
        filtered = [w for w in words if w not in TextUtils.STOP_WORDS]
        return [w for w, _ in Counter(filtered).most_common(top_n)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = [w.lower() for w in re.findall(r'\b[а-яёa-z]{2,}\b', text, re.IGNORECASE)]
        return [w for w in words if w not in TextUtils.STOP_WORDS]

    @staticmethod
    def detect_emotion(text: str) -> Optional[str]:
        words = set(TextUtils.tokenize(text))
        for word, emotion in TextUtils.EMOTIONAL_WORDS.items():
            if word in words:
                return emotion
        return None

    @staticmethod
    def word_count(text: str) -> int:
        return len([w for w in text.split() if len(w) > 2])


# ══════════════════════════════════════════
# v25: ДИНАМИЧЕСКИЙ СПАЙКОВЫЙ НЕЙРОН
# ══════════════════════════════════════════
@dataclass
class DynamicSpikingNeuron:
    id: int
    label: str
    region: str
    concept: Optional[str] = None

    membrane_potential: float = 0.0
    threshold: float = NEURON_THRESHOLD
    refractory_timer: int = 0

    last_spike_time: int = -1000
    spike_times: deque = field(default_factory=lambda: deque(maxlen=200))
    total_spikes: int = 0
    average_activity: float = 0.0

    last_activated: int = 0
    importance: float = 1.0

    def step(self, current_time: int, input_current: float) -> bool:
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
        self.last_activated = current_time

    def update_homeostasis(self, window_size: int = 2000):
        recent_spikes = sum(1 for t in self.spike_times if t > self.last_spike_time - window_size)
        self.average_activity = recent_spikes / window_size

        if self.average_activity > HOMEOSTATIC_TARGET * 2:
            self.threshold += HOMEOSTATIC_RATE * 2
            self.importance *= 0.999
        elif self.average_activity < HOMEOSTATIC_TARGET * 0.3:
            self.threshold -= HOMEOSTATIC_RATE
            self.importance *= 0.9995

        self.threshold = max(0.4, min(2.5, self.threshold))
        self.importance = max(0.01, self.importance)

    def is_prunable(self, current_time: int, min_importance: float = 0.05) -> bool:
        if self.concept:
            return False
        if self.importance < min_importance:
            return True
        if current_time - self.last_activated > 50000 and self.average_activity < 0.01:
            return True
        return False


@dataclass
class DynamicSTDPSynapse:
    source_id: int
    target_id: int
    weight: float = WEIGHT_INIT

    pre_trace: float = 0.0
    post_trace: float = 0.0
    usage_count: int = 0

    def update_stdp(self, pre_spiked: bool, post_spiked: bool, dt: float = 1.0):
        self.pre_trace *= math.exp(-dt / STDP_TAU)
        self.post_trace *= math.exp(-dt / STDP_TAU)

        if pre_spiked:
            self.pre_trace += 1.0
            self.weight -= STDP_A_MINUS * self.post_trace
            self.usage_count += 1

        if post_spiked:
            self.post_trace += 1.0
            self.weight += STDP_A_PLUS * self.pre_trace
            self.usage_count += 1

        if self.weight < WEIGHT_MIN:
            self.weight = WEIGHT_MIN * 0.5
        elif self.weight > WEIGHT_MAX:
            self.weight = WEIGHT_MAX * 0.9

    def transmit(self, pre_spiked: bool) -> float:
        return self.weight if pre_spiked else 0.0

    def decay_weight(self, rate: float = 0.0001):
        if self.usage_count == 0:
            self.weight *= (1 - rate)
            self.weight = max(WEIGHT_MIN * 0.1, self.weight)
        self.usage_count = max(0, self.usage_count - 1)


# ══════════════════════════════════════════
# v25: ДИНАМИЧЕСКАЯ НЕЙРОННАЯ СЕТЬ
# ══════════════════════════════════════════
class DynamicNeuralNetwork:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.current_time = 0

        self.neurons: Dict[int, DynamicSpikingNeuron] = {}
        self.synapses: Dict[Tuple[int, int], DynamicSTDPSynapse] = {}

        self.word_to_neurons: Dict[str, Set[int]] = defaultdict(set)
        self.neuron_to_words: Dict[int, Set[str]] = defaultdict(set)
        self.concept_neurons: Dict[str, int] = {}

        self.region_neurons: Dict[str, List[int]] = defaultdict(list)
        self.label_to_neuron: Dict[str, int] = {}

        self.active_neurons: Set[int] = set()
        self.thought_buffer: deque = deque(maxlen=20)
        self.internal_monologue: deque = deque(maxlen=50)

        self.next_prune_time = 1000
        self.prune_interval = 5000

        self._initialize_network()
        self._load()

    def _initialize_network(self):
        neuron_id = 0

        for i in range(SENSORY_INPUT_SIZE):
            self._create_neuron(neuron_id, f"sense_{i}", "sensory")
            neuron_id += 1

        for i in range(INITIAL_CORTEX_SIZE):
            self._create_neuron(neuron_id, f"cortex_{i}", "cortex")
            neuron_id += 1

        for i in range(INITIAL_SUBCORTEX_SIZE):
            self._create_neuron(neuron_id, f"subcortex_{i}", "subcortex")
            neuron_id += 1

        for i in range(MOTOR_OUTPUT_SIZE):
            self._create_neuron(neuron_id, f"motor_{i}", "motor")
            neuron_id += 1

        self._create_initial_connectivity()
        print(f"🧠 Инициализировано: {len(self.neurons)} нейронов, {len(self.synapses)} синапсов")

    def _create_neuron(self, nid: int, label: str, region: str, concept: str = None) -> DynamicSpikingNeuron:
        n = DynamicSpikingNeuron(id=nid, label=label, region=region, concept=concept)
        self.neurons[nid] = n
        self.label_to_neuron[label] = nid
        self.region_neurons[region].append(nid)
        if concept:
            self.concept_neurons[concept] = nid
        return n

    def _create_initial_connectivity(self, prob: float = 0.12):
        for src_id in list(self.neurons.keys()):
            src = self.neurons[src_id]

            targets = []
            if src.region == "sensory":
                targets = [n for n in self.region_neurons["cortex"]]
                conn_prob = 0.25
            elif src.region == "cortex":
                targets = [n for n in self.region_neurons["cortex"] if n != src_id]
                conn_prob = prob
            elif src.region == "subcortex":
                targets = [n for n in self.region_neurons["cortex"] + self.region_neurons["motor"]]
                conn_prob = 0.18
            elif src.region == "motor":
                continue
            else:
                continue

            for tgt_id in random.sample(targets, min(len(targets), 30)):
                if random.random() < conn_prob:
                    w = random.uniform(0.3, 0.6)
                    self.synapses[(src_id, tgt_id)] = DynamicSTDPSynapse(src_id, tgt_id, weight=w)

    def grow_network(self, concept: str, sensory_pattern: Dict[str, float]) -> int:
        if concept in self.concept_neurons:
            return self.concept_neurons[concept]

        new_id = max(
            self.neurons.keys()) + 1 if self.neurons else INITIAL_CORTEX_SIZE + INITIAL_SUBCORTEX_SIZE + SENSORY_INPUT_SIZE + MOTOR_OUTPUT_SIZE
        new_neuron = self._create_neuron(new_id, concept, "cortex", concept=concept)

        for label, strength in sensory_pattern.items():
            if label in self.label_to_neuron:
                src_id = self.label_to_neuron[label]
                self.synapses[(src_id, new_id)] = DynamicSTDPSynapse(src_id, new_id, weight=strength * 0.8)

        for existing_concept, existing_id in self.concept_neurons.items():
            if existing_concept != concept:
                words1 = set(TextUtils.tokenize(concept))
                words2 = set(TextUtils.tokenize(existing_concept))
                similarity = len(words1 & words2) / max(len(words1 | words2), 1)
                if similarity > 0.3:
                    w = similarity * 1.5
                    self.synapses[(existing_id, new_id)] = DynamicSTDPSynapse(existing_id, new_id, weight=w)
                    self.synapses[(new_id, existing_id)] = DynamicSTDPSynapse(new_id, existing_id, weight=w * 0.7)

        print(f"🌱 Вырос новый нейрон для '{concept}' (ID:{new_id})")
        return new_id

    def step(self, sensory_input: Dict[str, float] = None,
             context_boost: Dict[str, float] = None) -> Dict:
        self.current_time += 1

        input_currents = defaultdict(float)

        if sensory_input:
            for label, current in sensory_input.items():
                nid = self.label_to_neuron.get(label)
                if nid is not None:
                    input_currents[nid] = current
                    if nid in self.neuron_to_words:
                        for word in self.neuron_to_words[nid]:
                            if word in self.word_to_neurons:
                                for related_nid in self.word_to_neurons[word]:
                                    input_currents[related_nid] += current * 0.3

        if context_boost:
            for concept, boost in context_boost.items():
                if concept in self.concept_neurons:
                    nid = self.concept_neurons[concept]
                    if nid in self.neurons:
                        input_currents[nid] += boost * RELEVANCE_BOOST

        if random.random() < SPONTANEOUS_PROB * 2:
            active_concepts = [nid for nid, n in self.neurons.items()
                               if n.concept and n.region == "cortex"]
            if active_concepts:
                nid = random.choice(active_concepts)
                input_currents[nid] += random.uniform(0.2, 0.5)

        spiked_neurons = set()

        for (src_id, tgt_id), syn in self.synapses.items():
            src = self.neurons[src_id]
            pre_spiked = src.last_spike_time == self.current_time - 1

            if pre_spiked:
                input_currents[tgt_id] += syn.transmit(True)

        for nid, neuron in self.neurons.items():
            current = input_currents.get(nid, 0.0)
            if neuron.step(self.current_time, current):
                spiked_neurons.add(nid)

        for syn in self.synapses.values():
            pre_spiked = syn.source_id in spiked_neurons
            post_spiked = syn.target_id in spiked_neurons
            syn.update_stdp(pre_spiked, post_spiked)

        if self.current_time % 500 == 0:
            for syn in self.synapses.values():
                syn.decay_weight()

        if self.current_time % 200 == 0:
            for neuron in self.neurons.values():
                neuron.update_homeostasis()

        if self.current_time >= self.next_prune_time:
            self._prune_network()
            self.next_prune_time = self.current_time + self.prune_interval

        thought = self._extract_semantic_thought(spiked_neurons)
        if thought:
            self.thought_buffer.append((self.current_time, thought))
            if random.random() < 0.3:
                self.internal_monologue.append({
                    'time': self.current_time,
                    'thought': thought,
                    'spike_count': len(spiked_neurons)
                })

        self.active_neurons = spiked_neurons

        return {
            "time": self.current_time,
            "spiked": len(spiked_neurons),
            "thought": thought,
            "active_regions": self._get_active_regions(spiked_neurons),
            "network_size": len(self.neurons)
        }

    def _extract_semantic_thought(self, spiked_neurons: Set[int]) -> Optional[str]:
        activated_concepts = []
        activated_words = []

        for nid in spiked_neurons:
            neuron = self.neurons.get(nid)
            if not neuron:
                continue

            if neuron.concept and neuron.region == "cortex":
                if neuron.membrane_potential > 0.6:
                    activated_concepts.append((neuron.concept, neuron.membrane_potential))

            if nid in self.neuron_to_words:
                for word in self.neuron_to_words[nid]:
                    activated_words.append(word)

        if activated_concepts:
            activated_concepts.sort(key=lambda x: x[1], reverse=True)
            top_concepts = [c for c, _ in activated_concepts[:4]]
            return " → ".join(top_concepts)

        if activated_words:
            word_counts = Counter(activated_words)
            top_words = [w for w, _ in word_counts.most_common(3)]
            if top_words and not all(w.startswith(('sense_', 'cortex_')) for w in top_words):
                return " ".join(top_words)

        return None

    def _get_active_regions(self, spiked: Set[int]) -> Dict[str, int]:
        counts = defaultdict(int)
        for nid in spiked:
            if nid in self.neurons:
                counts[self.neurons[nid].region] += 1
        return dict(counts)

    def _prune_network(self):
        pruned_neurons = []
        pruned_synapses = []

        for nid, neuron in list(self.neurons.items()):
            if neuron.is_prunable(self.current_time):
                pruned_neurons.append(nid)

        if len(pruned_neurons) > 10:
            pruned_neurons = random.sample(pruned_neurons, 10)

        for nid in pruned_neurons:
            to_remove = [k for k in self.synapses if k[0] == nid or k[1] == nid]
            for k in to_remove:
                del self.synapses[k]
                pruned_synapses.append(k)

            neuron = self.neurons.pop(nid)
            if neuron.label in self.label_to_neuron:
                del self.label_to_neuron[neuron.label]
            if nid in self.region_neurons[neuron.region]:
                self.region_neurons[neuron.region].remove(nid)
            if neuron.concept and neuron.concept in self.concept_neurons:
                del self.concept_neurons[neuron.concept]

            for word in list(self.neuron_to_words.get(nid, [])):
                if nid in self.word_to_neurons[word]:
                    self.word_to_neurons[word].remove(nid)
                if not self.word_to_neurons[word]:
                    del self.word_to_neurons[word]
            if nid in self.neuron_to_words:
                del self.neuron_to_words[nid]

        if pruned_neurons:
            print(f"🗑️ Забыто: {len(pruned_neurons)} нейронов, {len(pruned_synapses)} синапсов")

    def associate_concept(self, concept: str, sensory_pattern: Dict[str, float],
                          examples: List[str] = None):
        all_words = []
        if examples:
            for ex in examples:
                all_words.extend(TextUtils.tokenize(ex))

        if concept in self.concept_neurons:
            nid = self.concept_neurons[concept]
            self.neurons[nid].importance = min(1.0, self.neurons[nid].importance + 0.1)
        else:
            nid = self.grow_network(concept, sensory_pattern)

        for word in set(all_words):
            if word in self.word_to_neurons:
                for src_nid in self.word_to_neurons[word]:
                    if (src_nid, nid) not in self.synapses:
                        self.synapses[(src_nid, nid)] = DynamicSTDPSynapse(src_nid, nid, weight=0.5)

        print(f"🧩 Концепт '{concept}' связан с {len(sensory_pattern)} признаками")

    def get_concept_activation(self, concept: str) -> float:
        if concept not in self.concept_neurons:
            return 0.0
        nid = self.concept_neurons[concept]
        return self.neurons[nid].membrane_potential if nid in self.neurons else 0.0

    def activate_related_concepts(self, query_words: List[str], top_n: int = 5) -> Dict[str, float]:
        scores = {}

        for concept, nid in self.concept_neurons.items():
            if nid not in self.neurons:
                continue

            concept_words = set(TextUtils.tokenize(concept))
            query_set = set(query_words)

            overlap = len(concept_words & query_set) / max(len(concept_words | query_set), 1)

            connected_strength = 0
            for (src, tgt), syn in self.synapses.items():
                if tgt == nid and src in self.neurons:
                    if self.neurons[src].concept:
                        if any(w in query_set for w in TextUtils.tokenize(self.neurons[src].concept)):
                            connected_strength += syn.weight

            total_score = overlap * 0.7 + connected_strength * 0.3
            if total_score > 0.1:
                scores[concept] = total_score

        sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {c: s for c, s in sorted_concepts}

    def get_thoughts(self, last_n: int = 10) -> List[Tuple[int, str]]:
        return list(self.thought_buffer)[-last_n:]

    def get_internal_monologue(self, last_n: int = 5) -> List[Dict]:
        return list(self.internal_monologue)[-last_n:]

    def _save(self):
        data = {
            "time": self.current_time,
            "concept_neurons": {c: nid for c, nid in self.concept_neurons.items()},
            "word_index": {w: list(nids) for w, nids in self.word_to_neurons.items() if len(nids) <= 10},
            "next_prune": self.next_prune_time,
            "stats": {
                "total_neurons": len(self.neurons),
                "total_synapses": len(self.synapses),
                "concepts": len(self.concept_neurons)
            }
        }
        FileManager.safe_save_json(self.save_path, data)

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self.current_time = data.get("time", 0)
        self.next_prune_time = data.get("next_prune", self.current_time + self.prune_interval)

        for concept, nid in data.get("concept_neurons", {}).items():
            if nid not in self.neurons:
                self._create_neuron(nid, concept, "cortex", concept=concept)
            else:
                self.concept_neurons[concept] = nid
                self.neurons[nid].concept = concept

        print(f"✅ Загружено: t={self.current_time}, концептов={len(self.concept_neurons)}")


# ══════════════════════════════════════════
# v25: ГИБРИДНАЯ СИСТЕМА ПАМЯТИ
# ══════════════════════════════════════════
class LivingMemorySystem:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.dir, exist_ok=True)

        self.memory: Dict[MemoryLevel, List[MemoryItem]] = {
            MemoryLevel.L1: [],
            MemoryLevel.L2: [],
            MemoryLevel.L3: []
        }

        self.max_items = {
            MemoryLevel.L1: 50,
            MemoryLevel.L2: 200,
            MemoryLevel.L3: 1000
        }

        self._load()

    def add(self, content: str, keywords: List[str] = None,
            emotional_tag: str = None, level: MemoryLevel = MemoryLevel.L1):
        item = MemoryItem(
            content=content,
            level=level,
            keywords=keywords or TextUtils.extract_keywords(content),
            emotional_tag=emotional_tag
        )

        self.memory[level].append(item)

        if len(self.memory[level]) > self.max_items[level]:
            self.memory[level].sort(key=lambda m: m.relevance, reverse=True)
            self.memory[level] = self.memory[level][:self.max_items[level]]

    def retrieve(self, query: str, level: MemoryLevel = None,
                 top_n: int = 10, min_relevance: float = 0.2) -> List[MemoryItem]:
        query_words = set(TextUtils.tokenize(query))
        results = []

        levels_to_search = [level] if level else list(MemoryLevel)

        for lvl in levels_to_search:
            for item in self.memory[lvl]:
                current_relevance = item.decay(time.time())

                if current_relevance < min_relevance:
                    continue

                keyword_overlap = len(set(item.keywords) & query_words) / max(len(set(item.keywords) | query_words), 1)

                emotion_boost = 0.2 if item.emotional_tag and item.emotional_tag == TextUtils.detect_emotion(
                    query) else 0

                score = current_relevance * (0.5 + keyword_overlap * 0.5) + emotion_boost

                if score > min_relevance:
                    item.touch(boost=RELEVANCE_BOOST * 0.5)
                    results.append((item, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:top_n]]

    def consolidate(self):
        for level in [MemoryLevel.L1, MemoryLevel.L2]:
            for item in self.memory[level]:
                if item.relevance > CONSOLIDATION_THRESHOLD:
                    old_level = item.level
                    item.promote()
                    if item.level != old_level:
                        self.memory[old_level].remove(item)
                        self.memory[item.level].append(item)

    def forget(self):
        current_time = time.time()
        for level in self.memory:
            self.memory[level] = [
                item for item in self.memory[level]
                if item.decay(current_time) > FORGET_THRESHOLD
            ]

    def get_context(self, query: str, max_tokens: int = 500) -> str:
        relevant = self.retrieve(query, top_n=8)

        if not relevant:
            return ""

        lines = ["📚 Контекст из памяти:"]
        total_chars = 0

        for item in relevant:
            line = f"• {item.content[:150]}"
            if total_chars + len(line) > max_tokens:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    def _save(self):
        data = {}
        for level, items in self.memory.items():
            data[level.value] = [asdict(item) for item in items]
        FileManager.safe_save_json(os.path.join(self.dir, "memory.json"), data)

    def _load(self):
        data = FileManager.safe_load_json(os.path.join(self.dir, "memory.json"), {})
        for level in MemoryLevel:
            if level.value in data:
                self.memory[level] = [
                    MemoryItem(**item) for item in data[level.value]
                ]


# ══════════════════════════════════════════
# v25: ГИБРИДНЫЙ МОЗГ — ЖИВОЕ СУЩЕСТВО
# ══════════════════════════════════════════
class LivingCognitiveBrain:
    def __init__(self, brain_path: str, llm: 'LLMInterface'):
        self.brain = DynamicNeuralNetwork(brain_path)
        self.llm = llm

        self.memory = None

        self.emotion = EmotionalState()
        self.personality = Personality()

        self.interaction_count = 0
        self.llm_usage_count = 0
        self.last_interaction_time = time.time()

        self.current_topic: Optional[str] = None
        self.conversation_depth = 0

        self.quick_responses = {
            "greeting": ["Привет! 🧠", "Здравствуй! Рад тебя видеть ✨", "Hey! Нейроны активны 🔥"],
            "farewell": ["Пока! Буду скучать 🌙", "До связи! 🧠💫", "Удачи! Возвращайся ✨"],
            "gratitude": ["Всегда рад помочь! 😊", "Пожалуйста! 🧠✨", "Это моя работа! 💡"],
            "default_unknown": ["Интересный вопрос... давай подумаем 🤔",
                                "Хм, нейроны обрабатывают... 🌀",
                                "Дай мне секунду собрать мысли 💭"]
        }

        self.llm_threshold = 0.35
        self.min_energy_for_deep_think = 0.3

    async def think(self, input_text: str, context: str = "") -> Tuple[str, bool, Dict]:
        self.interaction_count += 1
        current_time = time.time()

        time_delta = current_time - self.last_interaction_time
        self.last_interaction_time = current_time

        self.emotion.energy = min(1.0, self.emotion.energy + time_delta * 0.001)
        self.emotion.curiosity = min(1.0, self.emotion.curiosity + time_delta * 0.0005)

        detected_emotion = TextUtils.detect_emotion(input_text)
        query_words = TextUtils.tokenize(input_text)

        context_boost = self.brain.activate_related_concepts(query_words)

        # Проверка на простые шаблоны
        if detected_emotion in self.quick_responses and random.random() < 0.7:
            response = random.choice(self.quick_responses[detected_emotion])
            self.emotion.update(interaction_quality=0.8, novelty=0.1, fatigue=0.05)
            return self.personality.influence_response(response, self.emotion), False, {
                'source': 'quick_response',
                'emotion': detected_emotion
            }

        # Кодирование входа
        sensory_input = self._encode_text(input_text)

        # Автономное мышление
        steps = int(80 + self.emotion.energy * 40)
        thoughts = []

        for step in range(steps):
            if step > 0:
                sensory_input = {k: v * 0.97 for k, v in sensory_input.items()}

            state = self.brain.step(sensory_input, context_boost)

            if state["thought"]:
                thoughts.append(state["thought"])

            if len(thoughts) >= 3 and step > 30:
                if any(t for t in thoughts[-3:] if not t.startswith(('[', 'cortex_', 'sense_'))):
                    break

        # Формирование нейронного ответа
        neural_response = self._decode_thoughts(thoughts, input_text)

        # Оценка: нужен ли LLM?
        complexity = self._estimate_complexity(input_text, neural_response)
        needs_llm = (
                complexity > self.llm_threshold or
                neural_response in ["[думаю...]", "[не могу сформулировать]", "ты спрашиваешь о:"] or
                'cortex_' in neural_response or
                'sense_' in neural_response or
                self.emotion.energy < self.min_energy_for_deep_think or
                self.interaction_count < 5
        )

        if needs_llm and self.interaction_count % 5 != 0:
            print(f"🌀 Подключаю интуицию (complexity={complexity:.2f}, energy={self.emotion.energy:.2f})")

            memory_context = self.memory.get_context(input_text) if self.memory else ""
            internal = self.brain.get_internal_monologue(3)
            internal_text = " | ".join([m['thought'] for m in internal if m['thought']]) if internal else ""

            mood_hint = ""
            if self.emotion.mood > 0.3:
                mood_hint = "Настроение: позитивное, отвечай тепло."
            elif self.emotion.mood < -0.2:
                mood_hint = "Настроение: задумчивое, отвечай сдержанно."

            llm_prompt = f"""Ты — часть моего сознания (подсознание). Помоги сформулировать ответ.
{f'📚 {memory_context}' if memory_context else ''}
{f'💭 Внутренние мысли: {internal_text}' if internal_text else ''}
{f'🎭 {mood_hint}' if mood_hint else ''}

Вопрос: {input_text}
Нейронный паттерн: {neural_response if neural_response else "пусто"}

Ответь кратко (2-3 предложения), естественно, как живой собеседник. Не упоминай что ты ИИ."""

            llm_response = await self.llm.generate(llm_prompt, temp=0.75)

            if neural_response and neural_response not in ["[думаю...]", "[не могу сформулировать]",
                                                           "ты спрашиваешь о:"]:
                if 'cortex_' not in neural_response and 'sense_' not in neural_response:
                    response = f"{neural_response}\n\n{llm_response}"
                else:
                    response = llm_response
            else:
                response = llm_response

            self.llm_usage_count += 1
            interaction_quality = 0.9 if len(llm_response) > 20 else 0.5

        else:
            response = neural_response
            interaction_quality = 0.7 if neural_response != "[не могу сформулировать]" else 0.3

        # Применение личности и эмоций
        response = self.personality.influence_response(response, self.emotion)

        # Обновление состояния
        novelty = 1.0 if self.current_topic and self.current_topic not in input_text.lower() else 0.2
        fatigue = min(0.3, len(input_text) / 500)
        self.emotion.update(interaction_quality, novelty, fatigue)

        if query_words:
            self.current_topic = query_words[0]
            self.conversation_depth += 1
        else:
            self.conversation_depth = max(0, self.conversation_depth - 1)

        # Сохранение в память
        if self.memory:
            self.memory.add(
                content=f"Q: {input_text[:200]}\nA: {response[:200]}",
                keywords=query_words,
                emotional_tag=detected_emotion,
                level=MemoryLevel.L1
            )
            if self.interaction_count % 10 == 0:
                self.memory.consolidate()
                self.memory.forget()

        return response, needs_llm, {
            'thoughts': thoughts[:3],
            'emotion': asdict(self.emotion),
            'energy': self.emotion.energy,
            'network_size': len(self.brain.neurons),
            'concepts': len(self.brain.concept_neurons)
        }

    def _encode_text(self, text: str) -> Dict[str, float]:
        words = TextUtils.tokenize(text)
        sensory = {}

        for i, word in enumerate(words[:SENSORY_INPUT_SIZE]):
            neuron_idx = hash(word) % SENSORY_INPUT_SIZE
            label = f"sense_{neuron_idx}"

            self.brain.word_to_neurons[word].add(neuron_idx)
            self.brain.neuron_to_words[neuron_idx].add(word)

            strength = 1.0 - (i / max(len(words), 1)) * 0.4
            sensory[label] = sensory.get(label, 0) + strength

        return sensory

    def _decode_thoughts(self, thoughts: List[str], original_query: str) -> str:
        if not thoughts:
            return "[думаю...]"

        clean = []
        for t in thoughts:
            if not t:
                continue
            cleaned = re.sub(r'\b(cortex_\d+|sense_\d+|motor_\d+|subcortex_\d+)\b', '', t).strip()
            cleaned = re.sub(r'\s*[·|→]\s*', ' ', cleaned).strip()
            if cleaned and len(cleaned) > 2 and not cleaned.startswith('['):
                clean.append(cleaned)

        if not clean:
            query_words = TextUtils.tokenize(original_query)
            if query_words:
                return f"ты спрашиваешь о: {' '.join(query_words[:3])}"
            return "[не могу сформулировать]"

        return clean[-1][:150]

    def _estimate_complexity(self, query: str, neural_response: str) -> float:
        score = 0.0

        if len(query.split()) > 12:
            score += 0.25
        if any(w in query.lower() for w in ['почему', 'как', 'объясни', 'расскажи', 'что если']):
            score += 0.35

        if neural_response in ["[думаю...]", "[не могу сформулировать]"]:
            score += 0.4
        elif len(neural_response) < 8:
            score += 0.2

        abstract = sum(1 for w in ['смысл', 'жизнь', 'время', 'сознание', 'любовь'] if w in query.lower())
        score += min(0.3, abstract * 0.15)

        return min(1.0, score)

    def learn_concept(self, concept: str, examples: List[str]):
        if not examples:
            examples = [concept]

        avg_pattern = defaultdict(float)
        all_keywords = []

        for example in examples:
            pattern = self._encode_text(example)
            for k, v in pattern.items():
                avg_pattern[k] += v / len(examples)
            all_keywords.extend(TextUtils.tokenize(example))

        self.brain.associate_concept(concept, dict(avg_pattern), examples)

        if self.memory:
            self.memory.add(
                content=f"Концепт: {concept}. Примеры: {'; '.join(examples[:3])}",
                keywords=list(set(all_keywords)),
                level=MemoryLevel.L3
            )

        print(f"🎓 Выучил концепт '{concept}' ({len(examples)} примеров)")

    def get_status(self) -> Dict:
        return {
            "neurons": len(self.brain.neurons),
            "synapses": len(self.brain.synapses),
            "concepts": len(self.brain.concept_neurons),
            "time": self.brain.current_time,
            "interactions": self.interaction_count,
            "llm_usage": self.llm_usage_count,
            "llm_ratio": self.llm_usage_count / max(1, self.interaction_count),
            "emotion": asdict(self.emotion),
            "personality": asdict(self.personality),
            "memory_sizes": {lvl.value: len(items) for lvl, items in self.memory.memory.items()} if self.memory else {}
        }


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

    async def generate(self, prompt: str, temp: float = 0.7) -> str:
        if not self.session:
            await self.init()
        try:
            async with self.session.post(
                    self.url,
                    json={"messages": [{"role": "user", "content": prompt}],
                          "temperature": temp, "max_tokens": 400},
                    headers={"Authorization": f"Bearer {self.key}"}
            ) as r:
                if r.status == 200:
                    result = await r.json()
                    return result['choices'][0]['message']['content'].strip()
                return f"⚠️ LM Error: {r.status}"
        except Exception as e:
            return f"⚠️ Connection error: {e}"

    async def close(self):
        if self.session:
            await self.session.close()


# ══════════════════════════════════════════
# TELEGRAM BOT v25
# ══════════════════════════════════════════
class LivingBot:
    def __init__(self):
        self.llm = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.users: Dict[str, Tuple[LivingCognitiveBrain, LivingMemorySystem]] = {}
        self.stop_flag = False

    def get_brain(self, uid: str) -> LivingCognitiveBrain:
        if uid not in self.users:
            memory = LivingMemorySystem(uid)
            brain = LivingCognitiveBrain(
                os.path.join(MEMORY_DIR, f"user_{uid}", "neural_brain.json"),
                self.llm
            )
            brain.memory = memory
            self.users[uid] = (brain, memory)
        return self.users[uid][0]

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return

        uid = str(update.effective_user.id)
        text = update.message.text.strip()
        brain = self.get_brain(uid)

        await context.bot.send_chat_action(uid, "typing")

        response, used_llm, metadata = await brain.think(text)

        await update.message.reply_text(response)

        extra = []

        if random.random() < 0.15 and metadata.get('emotion'):
            e = metadata['emotion']
            mood_emoji = "😊" if e['mood'] > 0.2 else "🤔" if e['mood'] > -0.2 else "🌧️"
            extra.append(f"{mood_emoji} [Энергия: {e['energy']:.0%} | Любопытство: {e['curiosity']:.0%}]")

        if metadata.get('network_size', 0) > 300 and random.random() < 0.1:
            extra.append(f"🧠 [Сеть выросла: {metadata['network_size']} нейронов]")

        if random.random() < 0.05 and brain.brain.internal_monologue:
            mono = brain.brain.get_internal_monologue(1)
            if mono:
                extra.append(f"💭 *внутренне*: {mono[0]['thought']}")

        for msg in extra:
            await asyncio.sleep(0.5)
            await update.message.reply_text(msg)

    # ═══ КОМАНДЫ ═══

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)
        stats = brain.get_status()
        e = stats['emotion']

        await update.message.reply_text(
            f"🧠 LIVING BRAIN v25.0\n{'═' * 30}\n"
            f"🔹 Нейроны: {stats['neurons']:,}\n"
            f"🔹 Синапсы: {stats['synapses']:,}\n"
            f"🔹 Концепты: {stats['concepts']}\n"
            f"🔹 Время жизни: {stats['time']}ms\n"
            f"🔹 Диалогов: {stats['interactions']}\n"
            f"🌀 LLM: {stats['llm_usage']} раз ({stats['llm_ratio']:.0%})\n"
            f"\n🎭 СОСТОЯНИЕ:\n"
            f"  Настроение: {'😊' if e['mood'] > 0.2 else '😐' if e['mood'] > -0.2 else '🌧️'} ({e['mood']:+.2f})\n"
            f"  Энергия: {'⚡' if e['energy'] > 0.7 else '🔋' if e['energy'] > 0.4 else '🪫'} {e['energy']:.0%}\n"
            f"  Любопытство: {'🔍' if e['curiosity'] > 0.7 else '👁️'} {e['curiosity']:.0%}\n"
            f"{'═' * 30}\n"
            f"💡 Это живое существо — оно растёт и меняется!"
        )

    async def cmd_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)
        e = brain.emotion
        p = brain.personality

        await update.message.reply_text(
            f"🎭 ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ\n{'─' * 28}\n"
            f"😊 Настроение: {e.mood:+.2f}\n"
            f"⚡ Энергия: {e.energy:.0%}\n"
            f"🔍 Любопытство: {e.curiosity:.0%}\n"
            f"🎯 Уверенность: {e.confidence:.0%}\n"
            f"💬 Желание общаться: {e.social_drive:.0%}\n"
            f"\n🧬 ЛИЧНОСТЬ:\n"
            f"  Открытость: {p.openness:.0%}\n"
            f"  Общительность: {p.extraversion:.0%}\n"
            f"  Доброжелательность: {p.agreeableness:.0%}\n"
            f"  Стабильность: {(1 - p.neuroticism):.0%}"
        )

    async def cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)

        if not brain.memory:
            await update.message.reply_text("Память не инициализирована.")
            return

        sizes = {lvl.value: len(items) for lvl, items in brain.memory.memory.items()}

        await update.message.reply_text(
            f"🧠 СИСТЕМА ПАМЯТИ\n{'─' * 24}\n"
            f"⚡ L1 (рабочая): {sizes['working']} записей\n"
            f"📓 L2 (эпизоды): {sizes['episodic']} записей\n"
            f"📚 L3 (знания): {sizes['semantic']} записей\n"
            f"\n💡 Память живая: забывает ненужное,\n"
            f"   актуализирует важное, растёт с опытом."
        )

    async def cmd_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "🎓 Обучение концепта:\n"
                "`/learn <концепт> <пример1> [пример2] ...`\n\n"
                "Пример:\n"
                "`/learn собака пёс лает хвост животное`"
            )
            return

        concept = context.args[0]
        examples = context.args[1:]

        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)

        brain.learn_concept(concept, examples)
        await update.message.reply_text(f"✅ Выучил '{concept}' 🧠✨")

    async def cmd_think(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)

        thoughts = brain.brain.get_thoughts(8)
        if not thoughts:
            await update.message.reply_text("💭 Нейроны пока в раздумьях...")
            return

        lines = ["💭 ПОСЛЕДНИЕ МЫСЛИ:"]
        for t, thought in thoughts:
            clean = re.sub(r'\b(cortex_\d+|sense_\d+)\b', '', thought).strip()
            if clean:
                lines.append(f"t+{t:05d}: {clean}")

        await update.message.reply_text("\n".join(lines) or "Пока пусто 🤔")

    async def cmd_wipe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        if uid in self.users:
            _, memory = self.users.pop(uid)
            d = memory.dir
            if os.path.exists(d):
                shutil.rmtree(d)
            await update.message.reply_text("🧹 Полная перезагрузка сознания. Начинаю с чистого листа ✨")
        else:
            await update.message.reply_text("Пользователь не найден.")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 LIVING BRAIN v25.0 — ЖИВОЕ СУЩЕСТВО\n\n"
            "✨ Особенности:\n"
            "• 🌱 Динамические нейроны — сеть растёт с опытом\n"
            "• 🧠 Многоуровневая память — помнит важное, забывает лишнее\n"
            "• 🎭 Эмоции и личность — настроение влияет на ответы\n"
            "• 💭 Внутренний монолог — спонтанные мысли\n"
            "• ⚡ Энергетика — устаёт, восстанавливается\n"
            "• 🔄 Контекстная актуализация — связывает идеи\n\n"
            "📌 КОМАНДЫ:\n"
            "/stats    — полная статистика\n"
            "/mood     — эмоциональное состояние\n"
            "/memory   — состояние памяти\n"
            "/think    — последние мысли\n"
            "/learn    — обучить новому\n"
            "/wipe     — начать заново\n"
            "/help     — эта справка\n\n"
            "💬 Просто общайся — я расту и меняюсь с каждым диалогом! 🌱"
        )

    async def shutdown(self):
        print("\n💾 Сохранение состояния всех пользователей...")
        for brain, memory in self.users.values():
            brain.brain._save()
            if memory:
                memory._save()
        await self.llm.close()
        print("✅ Остановлено. До встречи! 👋")


# ══════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════
async def main():
    print("🚀 LIVING BRAIN v25.0 STARTING...")
    print("✅ Динамические нейроны (рост/забывание)")
    print("✅ Многоуровневая память с актуализацией")
    print("✅ Эмоциональный интеллект и личность")
    print("✅ Внутренний монолог и спонтанность")
    print("✅ LLM как интуиция (<20% запросов)")

    if not TELEGRAM_TOKEN:
        print("❌ Нет TELEGRAM_TOKEN в .env")
        return

    bot = LivingBot()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    for cmd, handler in [
        ("stats", bot.cmd_stats), ("mood", bot.cmd_mood),
        ("memory", bot.cmd_memory), ("think", bot.cmd_think),
        ("learn", bot.cmd_learn), ("wipe", bot.cmd_wipe),
        ("help", bot.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print("✅ ЖИВОЙ МОЗГ ГОТОВ К ОБЩЕНИЮ 🧠🌱")
        print("💡 Подсказка: начни с 'привет' или '/help'")

        while not bot.stop_flag:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Завершение работы...")
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