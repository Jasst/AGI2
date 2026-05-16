#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 AUTONOMOUS COGNITIVE BRAIN v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
АРХИТЕКТУРА:
  1. Spiking Neural Network — настоящие нейроны с мембранным потенциалом
  2. Hebbian Learning — "что активируется вместе, связывается вместе"
  3. Homeostatic Plasticity — поддержание оптимальной активности
  4. Autonomous Thinking — спонтанная внутренняя активность
  5. LLM как "подсознание" — только для сложных абстракций
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, random, math, asyncio, aiohttp
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# ══════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════
load_dotenv()
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

# Нейронные параметры
NEURON_THRESHOLD = 1.0  # Порог спайка
NEURON_RESET_POTENTIAL = 0.0  # После спайка
NEURON_DECAY = 0.95  # Утечка мембраны
NEURON_REFRACTORY_PERIOD = 3  # мс покоя после спайка
NEURON_NOISE = 0.02  # Спонтанный шум

# Синаптическая пластичность
STDP_A_PLUS = 0.015  # LTP (усиление)
STDP_A_MINUS = 0.012  # LTD (ослабление)
STDP_TAU = 20.0  # Временное окно (мс)
WEIGHT_MIN = 0.0
WEIGHT_MAX = 2.0
WEIGHT_INIT = 0.3

# Гомеостаз
HOMEOSTATIC_TARGET = 0.15  # Целевая активность (15%)
HOMEOSTATIC_RATE = 0.0001  # Скорость адаптации порога

# Автономное мышление
SPONTANEOUS_ACTIVITY_PROB = 0.03  # Вероятность спонтанного спайка
INNER_MONOLOGUE_EVERY = 50  # Каждые N тиков
MIN_THOUGHT_NEURONS = 3  # Минимум нейронов для "мысли"

# LLM (подсознание)
LLM_THRESHOLD_COMPLEXITY = 0.7  # Когда включать LLM
LLM_COOLDOWN = 5  # Минимум тиков между вызовами

# Архитектура
CORTEX_SIZE = 200  # Нейронов в коре
SUBCORTEX_SIZE = 50  # Подкорка (эмоции, reward)
SENSORY_INPUT_SIZE = 30  # Сенсорный слой
MOTOR_OUTPUT_SIZE = 20  # Моторный выход

SAVE_DIR = "autonomous_brain"
os.makedirs(SAVE_DIR, exist_ok=True)


# ══════════════════════════════════════════
# SPIKING NEURON — НАСТОЯЩИЙ НЕЙРОН
# ══════════════════════════════════════════
@dataclass
class SpikingNeuron:
    """Leaky Integrate-and-Fire нейрон с STDP"""
    id: int
    label: str  # Семантическая метка
    region: str  # cortex/subcortex/sensory/motor

    # Мембранные свойства
    membrane_potential: float = 0.0  # Текущий потенциал
    threshold: float = NEURON_THRESHOLD  # Динамический порог (гомеостаз)
    refractory_timer: int = 0  # Время рефрактерности

    # История активности
    last_spike_time: int = -1000  # Для STDP
    spike_times: deque = field(default_factory=lambda: deque(maxlen=100))
    total_spikes: int = 0

    # Гомеостаз
    average_activity: float = 0.0  # Скользящее среднее активности

    def step(self, current_time: int, input_current: float):
        """Один временной шаг (1 мс)"""
        # Рефрактерный период
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.membrane_potential = NEURON_RESET_POTENTIAL
            return False

        # Спонтанная активность (шум)
        noise = random.gauss(0, NEURON_NOISE)

        # Интегрирование тока
        self.membrane_potential *= NEURON_DECAY
        self.membrane_potential += input_current + noise

        # Проверка порога
        if self.membrane_potential >= self.threshold:
            self._fire(current_time)
            return True

        return False

    def _fire(self, current_time: int):
        """Генерация спайка"""
        self.membrane_potential = NEURON_RESET_POTENTIAL
        self.refractory_timer = NEURON_REFRACTORY_PERIOD
        self.last_spike_time = current_time
        self.spike_times.append(current_time)
        self.total_spikes += 1

    def update_homeostasis(self, window_size: int = 1000):
        """Гомеостатическая регуляция порога"""
        # Вычисляем среднюю частоту спайков
        recent_spikes = sum(1 for t in self.spike_times if t > self.last_spike_time - window_size)
        self.average_activity = recent_spikes / window_size

        # Адаптируем порог
        if self.average_activity > HOMEOSTATIC_TARGET:
            self.threshold += HOMEOSTATIC_RATE  # Повышаем (труднее активировать)
        elif self.average_activity < HOMEOSTATIC_TARGET:
            self.threshold -= HOMEOSTATIC_RATE  # Понижаем (легче активировать)

        self.threshold = max(0.5, min(2.0, self.threshold))


# ══════════════════════════════════════════
# STDP SYNAPSE — ПЛАСТИЧНАЯ СВЯЗЬ
# ══════════════════════════════════════════
@dataclass
class STDPSynapse:
    """Spike-Timing-Dependent Plasticity синапс"""
    source_id: int
    target_id: int
    weight: float = WEIGHT_INIT

    # Следы активности для STDP
    pre_trace: float = 0.0  # След пресинаптической активности
    post_trace: float = 0.0  # След постсинаптической активности

    def update_stdp(self, pre_spiked: bool, post_spiked: bool, dt: float = 1.0):
        """
        Хеббовское обучение: "что активируется вместе, связывается вместе"

        LTP (Long-Term Potentiation): пре → пост = усиление
        LTD (Long-Term Depression):   пост → пре = ослабление
        """
        # Обновление следов с экспоненциальным затуханием
        self.pre_trace *= math.exp(-dt / STDP_TAU)
        self.post_trace *= math.exp(-dt / STDP_TAU)

        # Пресинаптический спайк
        if pre_spiked:
            self.pre_trace += 1.0
            # LTD: если пост недавно был активен
            self.weight -= STDP_A_MINUS * self.post_trace

        # Постсинаптический спайк
        if post_spiked:
            self.post_trace += 1.0
            # LTP: если пре недавно был активен
            self.weight += STDP_A_PLUS * self.pre_trace

        # Ограничение веса
        self.weight = max(WEIGHT_MIN, min(WEIGHT_MAX, self.weight))

    def transmit(self, pre_spiked: bool) -> float:
        """Передача тока через синапс"""
        return self.weight if pre_spiked else 0.0


# ══════════════════════════════════════════
# НЕЙРОННАЯ СЕТЬ — АВТОНОМНЫЙ МОЗГ
# ══════════════════════════════════════════
class AutonomousCognitiveNetwork:
    """
    Автономная когнитивная сеть с:
    - Спайковыми нейронами
    - STDP обучением
    - Гомеостазом
    - Спонтанной активностью
    """

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.current_time = 0  # мс

        # Нейроны по регионам
        self.neurons: Dict[int, SpikingNeuron] = {}
        self.synapses: List[STDPSynapse] = []

        # Индексы
        self.label_to_neuron: Dict[str, int] = {}
        self.region_neurons: Dict[str, List[int]] = defaultdict(list)

        # Состояние
        self.active_neurons: Set[int] = set()  # Кто сейчас активен
        self.thought_buffer: deque = deque(maxlen=10)  # Последние "мысли"

        # Подсознание (LLM)
        self.llm_cooldown = 0
        self.llm_last_used = 0

        self._initialize_network()
        self._load()

    def _initialize_network(self):
        """Создание начальной топологии"""
        neuron_id = 0

        # 1. Сенсорный слой (входы)
        for i in range(SENSORY_INPUT_SIZE):
            self._create_neuron(neuron_id, f"sense_{i}", "sensory")
            neuron_id += 1

        # 2. Кора (основной мозг)
        for i in range(CORTEX_SIZE):
            self._create_neuron(neuron_id, f"cortex_{i}", "cortex")
            neuron_id += 1

        # 3. Подкорка (эмоции, reward)
        for i in range(SUBCORTEX_SIZE):
            self._create_neuron(neuron_id, f"subcortex_{i}", "subcortex")
            neuron_id += 1

        # 4. Моторный выход
        for i in range(MOTOR_OUTPUT_SIZE):
            self._create_neuron(neuron_id, f"motor_{i}", "motor")
            neuron_id += 1

        # Создаём начальные случайные связи
        self._create_random_connectivity()

        print(f"🧠 Создано: {len(self.neurons)} нейронов, {len(self.synapses)} синапсов")

    def _create_neuron(self, nid: int, label: str, region: str):
        """Создать нейрон"""
        n = SpikingNeuron(id=nid, label=label, region=region)
        self.neurons[nid] = n
        self.label_to_neuron[label] = nid
        self.region_neurons[region].append(nid)

    def _create_random_connectivity(self, connection_prob: float = 0.15):
        """Случайная начальная проводка"""
        for src_id in self.neurons:
            for tgt_id in self.neurons:
                if src_id == tgt_id:
                    continue

                src = self.neurons[src_id]
                tgt = self.neurons[tgt_id]

                # Правила связности
                should_connect = False

                # Сенсоры → Кора
                if src.region == "sensory" and tgt.region == "cortex":
                    should_connect = random.random() < 0.3

                # Кора → Кора (главный вычислительный слой)
                elif src.region == "cortex" and tgt.region == "cortex":
                    should_connect = random.random() < connection_prob

                # Кора → Подкорка
                elif src.region == "cortex" and tgt.region == "subcortex":
                    should_connect = random.random() < 0.2

                # Подкорка → Кора (обратная связь)
                elif src.region == "subcortex" and tgt.region == "cortex":
                    should_connect = random.random() < 0.25

                # Кора/Подкорка → Моторный выход
                elif tgt.region == "motor" and src.region in ["cortex", "subcortex"]:
                    should_connect = random.random() < 0.2

                if should_connect:
                    w = random.uniform(0.2, 0.5)
                    self.synapses.append(STDPSynapse(src_id, tgt_id, weight=w))

    def step(self, sensory_input: Dict[str, float] = None) -> Dict:
        """
        Один временной шаг (1 мс)

        Returns:
            Состояние сети: активные нейроны, мысли, и т.д.
        """
        self.current_time += 1

        # 1. Подача сенсорных входов
        input_currents = defaultdict(float)
        if sensory_input:
            for label, current in sensory_input.items():
                nid = self.label_to_neuron.get(label)
                if nid is not None:
                    input_currents[nid] = current

        # 2. Спонтанная активность (внутренний монолог)
        for nid in self.region_neurons["cortex"]:
            if random.random() < SPONTANEOUS_ACTIVITY_PROB:
                input_currents[nid] += random.uniform(0.1, 0.3)

        # 3. Синаптическая передача
        spiked_neurons = set()

        # Сначала вычисляем входные токи от синапсов
        for syn in self.synapses:
            src = self.neurons[syn.source_id]
            pre_spiked = src.last_spike_time == self.current_time - 1

            if pre_spiked:
                input_currents[syn.target_id] += syn.transmit(True)

        # 4. Обновление всех нейронов
        for nid, neuron in self.neurons.items():
            current = input_currents.get(nid, 0.0)
            if neuron.step(self.current_time, current):
                spiked_neurons.add(nid)

        # 5. STDP — обучение связей
        for syn in self.synapses:
            pre_spiked = syn.source_id in spiked_neurons
            post_spiked = syn.target_id in spiked_neurons
            syn.update_stdp(pre_spiked, post_spiked)

        # 6. Гомеостаз (каждые 100 мс)
        if self.current_time % 100 == 0:
            for neuron in self.neurons.values():
                neuron.update_homeostasis()

        # 7. Формирование "мысли"
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
        """Извлечь символическую 'мысль' из паттерна активности"""
        # Берём только кортикальные нейроны
        cortical_spikes = [nid for nid in spiked_neurons
                           if self.neurons[nid].region == "cortex"]

        if len(cortical_spikes) < MIN_THOUGHT_NEURONS:
            return None

        # Берём топ-5 самых активных
        top_neurons = sorted(cortical_spikes,
                             key=lambda nid: self.neurons[nid].membrane_potential,
                             reverse=True)[:5]

        labels = [self.neurons[nid].label for nid in top_neurons]
        return f"{'·'.join(labels)}"

    def _get_active_regions(self, spiked: Set[int]) -> Dict[str, int]:
        """Подсчёт активности по регионам"""
        counts = defaultdict(int)
        for nid in spiked:
            counts[self.neurons[nid].region] += 1
        return dict(counts)

    def associate_concept(self, concept: str, sensory_pattern: Dict[str, float]):
        """
        Создать нейрон для концепта и связать с сенсорным паттерном

        Это аналог 'узнавания' нового понятия
        """
        # Найдём или создадим нейрон для концепта
        if concept in self.label_to_neuron:
            concept_id = self.label_to_neuron[concept]
        else:
            concept_id = len(self.neurons)
            self._create_neuron(concept_id, concept, "cortex")

        # Свяжем с активными сенсорными нейронами
        for label, strength in sensory_pattern.items():
            if label in self.label_to_neuron:
                src_id = self.label_to_neuron[label]

                # Создаём сильную связь
                syn = STDPSynapse(src_id, concept_id, weight=strength)
                self.synapses.append(syn)

        print(f"🧩 Ассоциировал концепт '{concept}' с {len(sensory_pattern)} сенсорными признаками")

    def get_thoughts(self, last_n: int = 5) -> List[Tuple[int, str]]:
        """Последние N мыслей"""
        return list(self.thought_buffer)[-last_n:]

    def get_concept_activation(self, concept: str) -> float:
        """Текущая активация концепта"""
        if concept not in self.label_to_neuron:
            return 0.0
        nid = self.label_to_neuron[concept]
        return self.neurons[nid].membrane_potential

    def _save(self):
        """Сохранить состояние"""
        data = {
            "time": self.current_time,
            "neurons": [asdict(n) for n in self.neurons.values()],
            "synapses": [asdict(s) for s in self.synapses],
            "label_map": self.label_to_neuron,
        }
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Загрузить состояние"""
        if not os.path.exists(self.save_path):
            return

        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)

            self.current_time = data.get("time", 0)
            # TODO: десериализация нейронов и синапсов
            print(f"✅ Загружено состояние с t={self.current_time}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки: {e}")


# ══════════════════════════════════════════
# ПОДСОЗНАНИЕ (LLM) — ТОЛЬКО ДЛЯ СЛОЖНОГО
# ══════════════════════════════════════════
class SubconsciousLLM:
    """
    LLM используется только для:
    1. Очень сложных абстрактных вопросов
    2. Генерации языка (output)
    3. Разрешения неоднозначности
    """

    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def intuition(self, context: str, question: str) -> str:
        """
        "Интуитивный" ответ — LLM как подсознание
        """
        if not self.session:
            await self.init()

        prompt = (
            f"Ты — подсознание когнитивной системы. Дай краткий интуитивный ответ (1-2 предложения).\n\n"
            f"Контекст мыслей: {context}\n"
            f"Вопрос: {question}"
        )

        try:
            async with self.session.post(
                    self.url,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.8,
                        "max_tokens": 150
                    },
                    headers={"Authorization": f"Bearer {self.key}"}
            ) as r:
                if r.status == 200:
                    result = await r.json()
                    return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"⚠️ LLM error: {e}")

        return "[подсознание недоступно]"

    async def close(self):
        if self.session:
            await self.session.close()


# ══════════════════════════════════════════
# ГИБРИДНАЯ СИСТЕМА — МОЗГ + ПОДСОЗНАНИЕ
# ══════════════════════════════════════════
class HybridCognitiveBrain:
    """
    Полная система:
    - Автономная нейронная сеть (основное мышление)
    - LLM как подсознание (только при необходимости)
    """

    def __init__(self, brain_path: str):
        self.brain = AutonomousCognitiveNetwork(brain_path)
        self.subconscious = SubconsciousLLM(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)

        self.interaction_count = 0
        self.llm_usage_count = 0

    async def think(self, input_text: str) -> str:
        """
        Главный цикл мышления

        1. Конвертируем текст в сенсорные сигналы
        2. Запускаем нейронную сеть на ~100 мс
        3. Если сеть не справляется — подключаем LLM
        4. Формируем ответ
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
        response = self._decode_thoughts(thoughts)

        # 4. Если ответ слабый — подключаем подсознание
        complexity = self._estimate_complexity(input_text, response)

        if complexity > LLM_THRESHOLD_COMPLEXITY:
            print(f"🌀 Подключаю подсознание (complexity={complexity:.2f})")

            thought_context = " → ".join(thoughts[-5:]) if thoughts else "пусто"
            llm_response = await self.subconscious.intuition(thought_context, input_text)

            response = self._merge_responses(response, llm_response)
            self.llm_usage_count += 1

        return response

    def _encode_text(self, text: str) -> Dict[str, float]:
        """
        Конвертируем текст в активацию сенсорных нейронов

        В реальности здесь был бы word embedding → sparse coding
        Сейчас упрощённо: хешируем слова в нейроны
        """
        words = text.lower().split()
        sensory = {}

        for i, word in enumerate(words[:SENSORY_INPUT_SIZE]):
            # Простое хеширование слова в сенсорный нейрон
            neuron_idx = hash(word) % SENSORY_INPUT_SIZE
            label = f"sense_{neuron_idx}"

            # Сила сигнала зависит от позиции (первые слова важнее)
            strength = 1.0 - (i / len(words)) * 0.5

            sensory[label] = sensory.get(label, 0) + strength

        return sensory

    def _decode_thoughts(self, thoughts: List[str]) -> str:
        """
        Конвертируем нейронную активность обратно в текст
        """
        if not thoughts:
            return "[тишина]"

        # Берём последние уникальные мысли
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
        """
        Оценка сложности: нужен ли LLM?

        Высокая сложность если:
        - Длинный вопрос
        - Нейронный ответ слабый
        - Абстрактные слова
        """
        score = 0.0

        # Длина запроса
        if len(query.split()) > 10:
            score += 0.3

        # Слабый нейронный ответ
        if neural_response == "[тишина]" or len(neural_response) < 10:
            score += 0.4

        # Абстрактные слова (требуют LLM)
        abstract_words = ['почему', 'как', 'что если', 'объясни', 'расскажи', 'зачем']
        if any(w in query.lower() for w in abstract_words):
            score += 0.3

        return min(1.0, score)

    def _merge_responses(self, neural: str, llm: str) -> str:
        """Объединить нейронный ответ и LLM"""
        return f"[мозг] {neural}\n[интуиция] {llm}"

    async def inner_monologue(self, duration_ms: int = 1000):
        """
        Автономное мышление без внешнего входа

        Это "внутренний монолог" — мозг думает сам по себе
        """
        print(f"💭 Внутренний монолог на {duration_ms} мс...")

        thoughts = []
        for _ in range(duration_ms):
            state = self.brain.step(sensory_input=None)
            if state["thought"]:
                thoughts.append((state["time"], state["thought"]))

        print(f"   Спонтанных мыслей: {len(thoughts)}")
        return thoughts

    def learn_concept(self, concept: str, examples: List[str]):
        """
        Обучить новый концепт

        Это хеббовское обучение: связываем концепт с примерами
        """
        # Усредняем сенсорные паттерны примеров
        avg_pattern = defaultdict(float)

        for example in examples:
            pattern = self._encode_text(example)
            for k, v in pattern.items():
                avg_pattern[k] += v / len(examples)

        self.brain.associate_concept(concept, dict(avg_pattern))

    def stats(self) -> Dict:
        """Статистика"""
        return {
            "neurons": len(self.brain.neurons),
            "synapses": len(self.brain.synapses),
            "time": self.brain.current_time,
            "interactions": self.interaction_count,
            "llm_usage": self.llm_usage_count,
            "llm_ratio": self.llm_usage_count / max(1, self.interaction_count),
            "avg_weight": sum(s.weight for s in self.brain.synapses) / len(self.brain.synapses),
            "thoughts_buffer": len(self.brain.thought_buffer)
        }


# ══════════════════════════════════════════
# ДЕМО
# ══════════════════════════════════════════
async def demo():
    print("🧠 AUTONOMOUS COGNITIVE BRAIN v1.0")
    print("=" * 50)

    brain = HybridCognitiveBrain(os.path.join(SAVE_DIR, "brain_state.json"))

    # 1. Обучение концептов
    print("\n📚 ОБУЧЕНИЕ КОНЦЕПТОВ")
    brain.learn_concept("привет", [
        "привет как дела",
        "здравствуй друг",
        "хай"
    ])

    brain.learn_concept("python", [
        "python код функция",
        "программирование на питоне",
        "python script"
    ])

    # 2. Внутренний монолог (автономное мышление)
    print("\n💭 ВНУТРЕННИЙ МОНОЛОГ (3 секунды)")
    thoughts = await brain.inner_monologue(duration_ms=3000)
    for t, thought in thoughts[-10:]:
        print(f"  t={t:05d}ms: {thought}")

    # 3. Диалог
    print("\n💬 ДИАЛОГ")

    queries = [
        "привет!",
        "расскажи про python",
        "как работает нейронная сеть?",
        "что ты думаешь о жизни?"  # Сложный вопрос → включит LLM
    ]

    for q in queries:
        print(f"\n👤 {q}")
        response = await brain.think(q)
        print(f"🤖 {response}")

        # Небольшая пауза для "обдумывания"
        for _ in range(50):
            brain.brain.step()

    # 4. Статистика
    print("\n" + "=" * 50)
    print("📊 СТАТИСТИКА")
    stats = brain.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    await brain.subconscious.close()


if __name__ == "__main__":
    asyncio.run(demo())