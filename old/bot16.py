#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID NEURAL BRAIN v16.0 - CHRONOS
✨ Временное сознание + Эпизодическая память + Линейный поток времени
🔗 ChronoNeurons + TimeEncoder + Temporal Reasoning
🎯 Понимает "вчера/сегодня/завтра" как внутренний ритм
"""
import os, json, re, asyncio, aiohttp, traceback, hashlib, math, signal, sys
import shutil
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

BASE_DIR = "hybrid_brain_v16"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
NEURAL_DIR = os.path.join(BASE_DIR, "neural")
for d in [MEMORY_DIR, NEURAL_DIR]:
    os.makedirs(d, exist_ok=True)


# ==================== УТИЛИТЫ ====================
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
            print(f"⚠️ Save error {filepath}: {e}")
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
    @staticmethod
    def extract_keywords(text: str, top_n: int = 5) -> List[str]:
        stop_words = {'в', 'и', 'на', 'с', 'по', 'для', 'от', 'к', 'о', 'у', 'из', 'за',
                      'что', 'это', 'как', 'то', 'а', 'но', 'или', 'the', 'is', 'at',
                      'не', 'ты', 'я', 'мне', 'себя', 'был', 'была', 'было'}
        words = [w.lower() for w in re.findall(r'\b\w{3,}\b', text)]
        filtered = [w for w in words if w not in stop_words]
        return [word for word, _ in Counter(filtered).most_common(top_n)]


# ==================== ВРЕМЕННОЙ КОДИРОВЩИК ====================
class TimeEncoder:
    """
    🕰️ Переводит относительное время ("вчера", "через час") в абсолютные диапазоны
    """
    TIME_WORDS = {
        'сейчас': {'hours': 0, 'minutes': 0},
        'только что': {'minutes': -30},
        'недавно': {'hours': -2},
        'сегодня': {'days': 0},
        'вчера': {'days': -1},
        'позавчера': {'days': -2},
        'завтра': {'days': 1},
        'послезавтра': {'days': 2},
        'на прошлой неделе': {'days': -7},
        'на следующей неделе': {'days': 7},
        'утро': {'hour_range': (6, 12)},
        'день': {'hour_range': (12, 18)},
        'вечер': {'hour_range': (18, 24)},
        'ночь': {'hour_range': (0, 6)},
    }

    @staticmethod
    def parse_time_ref(text: str, now: datetime) -> Optional[Dict[str, Any]]:
        text_lower = text.lower()

        # Прямые совпадения
        for phrase, offset in TimeEncoder.TIME_WORDS.items():
            if phrase in text_lower:
                if 'hour_range' in offset:
                    return {'type': 'time_of_day', 'range': offset['hour_range'], 'date': now.date()}
                else:
                    delta = timedelta(**{k: v for k, v in offset.items() if k != 'hour_range'})
                    target = now + delta
                    return {'type': 'date_range', 'start': target.replace(hour=0, minute=0),
                            'end': target.replace(hour=23, minute=59)}

        # Паттерны: "2 часа назад", "3 дня назад"
        match = re.search(r'(\d+)\s*(час|часа|часов|день|дня|дней|неделю|недели|месяц|месяца)\s*назад', text_lower)
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            if 'час' in unit:
                return {'type': 'date_range', 'start': now - timedelta(hours=val), 'end': now}
            elif 'день' in unit:
                return {'type': 'date_range', 'start': now - timedelta(days=val), 'end': now}

        # Паттерны: "через 2 часа", "на следующей неделе"
        match = re.search(r'через\s+(\d+)\s*(час|часа|часов|день|дня|дней)', text_lower)
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            if 'час' in unit:
                return {'type': 'future', 'in': timedelta(hours=val)}
            elif 'день' in unit:
                return {'type': 'future', 'in': timedelta(days=val)}

        return None

    @staticmethod
    def get_relative_description(event_time: datetime, now: datetime) -> str:
        diff = now - event_time
        if diff.total_seconds() < 60:
            return "только что"
        elif diff.total_seconds() < 3600:
            mins = int(diff.total_seconds() / 60)
            return f"{mins} мин. назад"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} час. назад"
        elif diff.days == 1:
            return "вчера"
        elif diff.days < 7:
            return f"{diff.days} дней назад"
        else:
            return event_time.strftime("%d.%m.%Y")


# ==================== ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ С ВРЕМЕНЕМ ====================
@dataclass
class EpisodicMemory:
    id: str
    timestamp: str  # ISO format
    content: str
    keywords: List[str]
    temporal_tags: List[str]  # "утро", "вчера", "будущее"
    user_emotion: Optional[str]
    location_hints: List[str]  # "кино", "дом", "работа"
    sequence_id: int  # Порядковый номер в потоке


class TemporalMemory:
    """
    🗄️ Хранит события в линейном временном потоке
    Позволяет запрашивать: "что было до Х?", "где я был вчера?"
    """

    def __init__(self, user_id: str, save_dir: str):
        self.user_id = user_id
        self.file = os.path.join(save_dir, "episodic_time.json")
        self.events: List[EpisodicMemory] = []
        self.sequence_counter = 0
        self.now = datetime.now()  # Внутренние часы бота

        data = FileManager.safe_load_json(self.file)
        if data:
            self.sequence_counter = data.get('seq', 0)
            for e in data.get('events', []):
                self.events.append(EpisodicMemory(**e))
            print(f"⏳ Загружено {len(self.events)} эпизодов для {user_id}")

    def add_event(self, content: str, keywords: List[str], now: datetime = None) -> EpisodicMemory:
        if now is None:
            now = datetime.now()
            self.now = now

        # Авто-детекция временных тегов из текста
        temporal_tags = []
        location_hints = []

        time_words = ['сегодня', 'вчера', 'завтра', 'утро', 'вечер', 'ночь', 'сейчас', 'потом', 'раньше']
        for tw in time_words:
            if tw in content.lower():
                temporal_tags.append(tw)

        # Детекция мест (упрощенно - слова после предлогов)
        loc_match = re.findall(r'(?:в|на|из|к)\s+([а-яёa-z]+)', content.lower())
        location_hints = [l for l in loc_match if l not in ['дом', 'домой', 'город', 'место']]

        event = EpisodicMemory(
            id=hashlib.md5(f"{content}{now.isoformat()}".encode()).hexdigest()[:12],
            timestamp=now.isoformat(),
            content=content,
            keywords=keywords,
            temporal_tags=temporal_tags,
            user_emotion=self._detect_emotion(content),
            location_hints=location_hints,
            sequence_id=self.sequence_counter
        )

        self.events.append(event)
        self.sequence_counter += 1

        # Авто-очистка старых событий (храним последние 200)
        if len(self.events) > 200:
            self.events = self.events[-200:]

        return event

    def _detect_emotion(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ['рад', 'хорошо', 'отлично', 'супер', 'люблю']):
            return 'positive'
        if any(w in t for w in ['груст', 'плохо', 'устал', 'нет', 'не хочу']):
            return 'negative'
        return None

    def query_by_time(self, time_ref: str, now: datetime = None) -> List[EpisodicMemory]:
        """Найти события по временному запросу: 'вчера', 'на прошлой неделе'"""
        if now is None:
            now = datetime.now()

        time_range = TimeEncoder.parse_time_ref(time_ref, now)
        if not time_range:
            return []

        results = []
        for event in reversed(self.events):  # От новых к старым
            event_time = datetime.fromisoformat(event.timestamp)

            if time_range['type'] == 'date_range':
                if time_range['start'] <= event_time <= time_range['end']:
                    results.append(event)
            elif time_range['type'] == 'time_of_day':
                if event_time.date() == time_range['date']:
                    hour = event_time.hour
                    if time_range['range'][0] <= hour < time_range['range'][1]:
                        results.append(event)

        return results[:10]  # Лимит результатов

    def query_by_sequence(self, reference_event: str, direction: str = 'before', limit: int = 3) -> List[
        EpisodicMemory]:
        """Найти события ДО или ПОСЛЕ указанного"""
        # Найти референсное событие
        ref = None
        for e in self.events:
            if reference_event.lower() in e.content.lower() or any(
                    k in e.keywords for k in TextUtils.extract_keywords(reference_event)):
                ref = e
                break

        if not ref:
            return []

        results = []
        if direction == 'before':
            for e in reversed(self.events):
                if e.sequence_id < ref.sequence_id:
                    results.append(e)
                    if len(results) >= limit:
                        break
        else:  # after
            for e in self.events:
                if e.sequence_id > ref.sequence_id:
                    results.append(e)
                    if len(results) >= limit:
                        break
        return results

    def get_timeline_summary(self, hours: int = 24) -> str:
        """Краткая сводка за последние N часов"""
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)
        recent = [e for e in self.events if datetime.fromisoformat(e.timestamp) >= cutoff]

        if not recent:
            return "За это время ничего не зафиксировано."

        summary = []
        for e in recent[-5:]:  # Последние 5
            rel_time = TimeEncoder.get_relative_description(datetime.fromisoformat(e.timestamp), now)
            summary.append(f"[{rel_time}] {e.content[:80]}")
        return "\n".join(summary)

    def save(self):
        data = {
            'seq': self.sequence_counter,
            'events': [asdict(e) for e in self.events]
        }
        FileManager.safe_save_json(self.file, data)


# ==================== ВРЕМЕННЫЕ НЕЙРОНЫ (ChronoNeurons) ====================
@dataclass
class ChronoNode:
    id: str
    label: str  # "вчера", "утро", "последовательность"
    node_type: str  # "temporal_ref", "time_of_day", "sequence"
    activation: float = 0.0
    linked_events: List[str] = field(default_factory=list)


class ChronoNeuralLayer:
    """
    🧠 Специальные нейроны для работы со временем
    Активируются от временных слов и связывают события в поток
    """

    def __init__(self):
        self.nodes: Dict[str, ChronoNode] = {}
        self._init_base_nodes()

    def _init_base_nodes(self):
        # Базовые временные концепты
        for label, ntype in [
            ('вчера', 'temporal_ref'), ('сегодня', 'temporal_ref'), ('завтра', 'temporal_ref'),
            ('утро', 'time_of_day'), ('день', 'time_of_day'), ('вечер', 'time_of_day'), ('ночь', 'time_of_day'),
            ('раньше', 'sequence'), ('потом', 'sequence'), ('до', 'sequence'), ('после', 'sequence')
        ]:
            self._create_node(label, ntype)

    def _create_node(self, label: str, node_type: str) -> ChronoNode:
        node_id = hashlib.md5(f"chrono_{label}".encode()).hexdigest()[:8]
        if node_id not in self.nodes:
            self.nodes[node_id] = ChronoNode(id=node_id, label=label, node_type=node_type)
        return self.nodes[node_id]

    def process_temporal_input(self, text: str) -> List[ChronoNode]:
        """Активировать временные нейроны на основе текста"""
        activated = []
        text_lower = text.lower()

        for node in self.nodes.values():
            if node.label in text_lower:
                node.activation = min(1.0, node.activation + 0.8)
                activated.append(node)

        # Затухание неактивных
        for node in self.nodes.values():
            if node not in activated:
                node.activation *= 0.85

        return activated

    def get_active_temporal_context(self) -> List[str]:
        """Вернуть активные временные теги для промпта"""
        active = [n.label for n in self.nodes.values() if n.activation > 0.4]
        return active

    def link_event_to_time(self, event: EpisodicMemory, temporal_nodes: List[ChronoNode]):
        """Связать событие с активированными временными нейронами"""
        for node in temporal_nodes:
            if event.id not in node.linked_events:
                node.linked_events.append(event.id)
                # Ограничиваем список связей
                if len(node.linked_events) > 50:
                    node.linked_events = node.linked_events[-50:]


# ==================== ДИНАМИЧЕСКОЕ НЕЙРО-ЯДРО (с поддержкой времени) ====================
@dataclass
class NeuroNode:
    id: str
    label: str
    category: str
    activation: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    temporal_weight: float = 1.0  # Вес для временных запросов


@dataclass
class NeuroSynapse:
    source: str
    target: str
    weight: float = 0.5
    plasticity: float = 0.8
    temporal_decay: float = 0.0  # Насколько связь зависит от времени
    last_fired: str = ""


class DynamicNeuralCortex:
    def __init__(self, user_id: str, save_dir: str):
        self.user_id = user_id
        self.save_path = os.path.join(save_dir, "cortex_graph.json")
        self.nodes: Dict[str, NeuroNode] = {}
        self.synapses: Dict[str, NeuroSynapse] = {}
        self.chrono_layer = ChronoNeuralLayer()

        data = FileManager.safe_load_json(self.save_path)
        if data and 'nodes' in data:
            for n in data.get('nodes', []):
                self.nodes[n['id']] = NeuroNode(**n)
            for s in data.get('synapses', []):
                self.synapses[s['source'] + "->" + s['target']] = NeuroSynapse(**s)
            print(f"🧠 Кора загружена: {len(self.nodes)} нейронов")
        else:
            self._create_node("Приветствие", "action")
            self._create_node("Помощь", "action")
            self.save()

    def _create_node(self, label: str, category: str = "concept") -> NeuroNode:
        node_id = hashlib.md5(label.encode()).hexdigest()[:8]
        if node_id not in self.nodes:
            self.nodes[node_id] = NeuroNode(id=node_id, label=label, category=category)
        return self.nodes[node_id]

    def process_input(self, text: str, now: datetime = None) -> Tuple[List[str], List[ChronoNode]]:
        if now is None:
            now = datetime.now()

        keywords = TextUtils.extract_keywords(text, top_n=10)
        activated_labels = []

        # Активация обычных нейронов
        for i, kw1 in enumerate(keywords):
            node1 = self._create_node(kw1)
            node1.access_count += 1
            node1.activation = 1.0
            activated_labels.append(kw1)
            for kw2 in keywords[i + 1:]:
                node2 = self._create_node(kw2)
                self._strengthen_connection(node1.id, node2.id, 0.1)

        # Активация временных нейронов
        chrono_nodes = self.chrono_layer.process_temporal_input(text)

        # Распространение активации
        final_context = set(activated_labels)
        for label in activated_labels:
            node_id = hashlib.md5(label.encode()).hexdigest()[:8]
            if node_id in self.nodes:
                for key, syn in self.synapses.items():
                    if syn.source == node_id and syn.weight > 0.4:
                        target = self.nodes.get(syn.target)
                        if target:
                            final_context.add(target.label)
                            target.activation = min(1.0, target.activation + 0.5)

        # Затухание
        for node in self.nodes.values():
            node.activation *= 0.9

        return list(final_context), chrono_nodes

    def _strengthen_connection(self, src: str, tgt: str, reward: float):
        if src == tgt: return
        for k in [f"{src}->{tgt}", f"{tgt}->{src}"]:
            if k not in self.synapses:
                self.synapses[k] = NeuroSynapse(source=src, target=tgt if k.startswith(src) else src, weight=0.1)
            syn = self.synapses[k]
            syn.weight = min(1.0, syn.weight + reward * syn.plasticity)
            syn.last_fired = datetime.now().isoformat()
            syn.plasticity *= 0.99

    def get_system_prompt_modifiers(self, chrono_context: List[str], temporal_memory: TemporalMemory) -> str:
        modifiers = []

        # Контекст от обычных нейронов
        active = sorted(self.nodes.values(), key=lambda x: x.activation, reverse=True)[:5]
        for node in active:
            if node.activation > 0.3:
                if node.label in ['код', 'python', 'программа']:
                    modifiers.append("Ты разработчик. Давай код.")
                elif node.label in ['груст', 'проблема']:
                    modifiers.append("Ты эмпатичный собеседник.")

        # Контекст от временных нейронов
        if chrono_context:
            modifiers.append(
                f"Пользователь упоминает время: {', '.join(chrono_context)}. Учитывай временной контекст в ответе.")

        # Если запрос про прошлое/будущее - добавить инструкцию искать в памяти
        if any(t in chrono_context for t in ['вчера', 'раньше', 'до', 'после', 'на прошлой неделе']):
            modifiers.append("Если пользователь спрашивает о прошлом, используй эпизодическую память для ответа.")

        if not modifiers:
            modifiers.append("Ты полезный ассистент с памятью о времени.")

        return " ".join(modifiers)

    def reinforce_path(self, keywords: List[str], chrono_nodes: List[ChronoNode], success: bool):
        reward = 0.1 if success else -0.1
        for i, kw1 in enumerate(keywords):
            id1 = hashlib.md5(kw1.encode()).hexdigest()[:8]
            for kw2 in keywords[i + 1:]:
                id2 = hashlib.md5(kw2.encode()).hexdigest()[:8]
                self._strengthen_connection(id1, id2, reward)
        # Укрепление временных связей
        for cn in chrono_nodes:
            cn.activation = min(1.0, cn.activation + reward)

    def save(self):
        data = {
            'nodes': [asdict(n) for n in self.nodes.values()],
            'synapses': [asdict(s) for s in self.synapses.values()]
        }
        FileManager.safe_save_json(self.save_path, data)

    def get_stats(self) -> Dict:
        return {
            'neurons': len(self.nodes),
            'synapses': len(self.synapses),
            'chrono_nodes': len(self.chrono_layer.nodes),
            'active_chrono': len([n for n in self.chrono_layer.nodes.values() if n.activation > 0.4])
        }


# ==================== ГИБРИДНАЯ ПАМЯТЬ (ОБЪЕДИНЕННАЯ) ====================
class HybridMemorySystem:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.dir, exist_ok=True)

        self.st_file = os.path.join(self.dir, "short_term.json")
        self.short_term = FileManager.safe_load_json(self.st_file, [])

        self.cortex = DynamicNeuralCortex(user_id, self.dir)
        self.temporal = TemporalMemory(user_id, self.dir)
        self.now = datetime.now()

    async def process(self, text: str) -> Tuple[str, str, List[str], List[ChronoNode]]:
        self.now = datetime.now()

        # 1. Нейро-активация + временные нейроны
        concepts, chrono_nodes = self.cortex.process_input(text, self.now)

        # 2. Добавить событие в эпизодическую память
        keywords = TextUtils.extract_keywords(text)
        event = self.temporal.add_event(text, keywords, self.now)

        # 3. Связать событие с временными нейронами
        self.cortex.chrono_layer.link_event_to_time(event, chrono_nodes)

        # 4. Системный промпт с учетом времени
        sys_mod = self.cortex.get_system_prompt_modifiers(
            [n.label for n in chrono_nodes if n.activation > 0.3],
            self.temporal
        )

        # 5. Контекст диалога
        st_ctx = "\n".join([f"{m['role']}: {m['content']}" for m in self.short_term[-5:]])

        # 6. Сохранить в short-term
        self.short_term.append({'role': 'user', 'content': text, 'time': self.now.isoformat()})
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]
            FileManager.safe_save_json(self.st_file, self.short_term)

        return st_ctx, sys_mod, concepts, chrono_nodes

    def handle_temporal_query(self, text: str) -> Optional[str]:
        """Обработать запросы типа 'где я был вчера?', 'что я делал раньше?'"""
        # Детекция запроса о прошлом
        if not any(w in text.lower() for w in
                   ['был', 'делал', 'ходил', 'говорил', 'где', 'когда', 'вчера', 'раньше', 'до', 'после']):
            return None

        # Попытка извлечь временной референс
        time_ref = TimeEncoder.parse_time_ref(text, self.now)
        if time_ref:
            events = self.temporal.query_by_time(text, self.now)
            if events:
                # Сформировать ответ из найденных событий
                answers = []
                for e in events[:3]:
                    rel = TimeEncoder.get_relative_description(datetime.fromisoformat(e.timestamp), self.now)
                    answers.append(f"{rel}: {e.content}")
                return "🕰️ В памяти найдено:\n" + "\n".join(answers)

        # Поиск по последовательности
        for direction in ['before', 'after']:
            if direction in text.lower():
                # Извлечь референсное событие (упрощенно - последние ключевые слова)
                ref = " ".join(TextUtils.extract_keywords(text)[-2:])
                if ref:
                    events = self.temporal.query_by_sequence(ref, direction)
                    if events:
                        return f"🔗 {direction.upper()} '{ref}':\n" + "\n".join([f"- {e.content[:60]}" for e in events])

        return None

    def add_response(self, text: str):
        self.short_term.append({'role': 'assistant', 'content': text, 'time': datetime.now().isoformat()})
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]
            FileManager.safe_save_json(self.st_file, self.short_term)

    def save_all(self):
        self.cortex.save()
        self.temporal.save()
        FileManager.safe_save_json(self.st_file, self.short_term)


# ==================== LLM ИНТЕРФЕЙС ====================
class LLMInterface:
    def __init__(self, url, key):
        self.url, self.key = url, key
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def generate(self, prompt, system=None, temp=0.7):
        if not self.session:
            await self.init()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            async with self.session.post(
                    self.url,
                    json={"messages": msgs, "temperature": temp, "max_tokens": 1500},
                    headers={"Authorization": f"Bearer {self.key}"}
            ) as r:
                if r.status == 200:
                    return (await r.json())['choices'][0]['message']['content']
                return f"LM Error: {r.status}"
        except Exception as e:
            return f"Connection error: {e}"

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


# ==================== БОТ ====================
class HybridBot:
    def __init__(self):
        self.llm = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.users: Dict[str, HybridMemorySystem] = {}
        self.stop_flag = False

    def get_brain(self, uid: str) -> HybridMemorySystem:
        if uid not in self.users:
            self.users[uid] = HybridMemorySystem(uid)
        return self.users[uid]

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        uid = str(update.effective_user.id)
        text = update.message.text
        brain = self.get_brain(uid)

        await context.bot.send_chat_action(uid, "typing")

        # 1. Обработка входа
        st_ctx, sys_mod, concepts, chrono_nodes = await brain.process(text)

        # 2. Проверка: это запрос о времени?
        temporal_answer = brain.handle_temporal_query(text)
        if temporal_answer:
            brain.add_response(temporal_answer)
            await update.message.reply_text(temporal_answer)
            return

        # 3. Формирование промпта
        full_system = f"{sys_mod}\n=== ДИАЛОГ ===\n{st_ctx}\n=== ВРЕМЯ СЕЙЧАС ===\n{brain.now.strftime('%H:%M %d.%m.%Y')}"

        # 4. Генерация
        response = await self.llm.generate(text, system=full_system)

        # 5. Сохранение и обучение
        brain.add_response(response)
        brain.cortex.reinforce_path(concepts, chrono_nodes, len(response) > 10)

        await update.message.reply_text(response)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)
        s = brain.cortex.get_stats()
        msg = f"🧠 CHRONOS v16.0\n🔹 Нейроны: {s['neurons']}\n🔹 Синапсы: {s['synapses']}\n🔹 Временные нейроны: {s['chrono_nodes']} (активны: {s['active_chrono']})\n🔹 Эпизодов в памяти: {len(brain.temporal.events)}"
        await update.message.reply_text(msg)

    async def cmd_timeline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        brain = self.get_brain(uid)
        hours = 24
        if context.args and context.args[0].isdigit():
            hours = int(context.args[0])
        summary = brain.temporal.get_timeline_summary(hours)
        await update.message.reply_text(f"🕰️ СОБЫТИЯ за последние {hours}ч:\n{summary}")

    async def cmd_wipe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        if uid in self.users:
            dir_path = self.users[uid].dir
            self.users.pop(uid)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            await update.message.reply_text("🧠 Полная очистка памяти и времени.")
        else:
            await update.message.reply_text("Пользователь не найден.")

    async def shutdown(self):
        print("\n💾 Сохранение...")
        for b in self.users.values():
            b.save_all()
        await self.llm.close()
        print("✅ Остановлено")


# ==================== ЗАПУСК ====================
async def main():
    print("🚀 HYBRID NEURAL BRAIN v16.0 - CHRONOS STARTING...")
    if not TELEGRAM_TOKEN:
        print("❌ Нет TELEGRAM_TOKEN")
        return

    bot = HybridBot()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    app.add_handler(CommandHandler("stats", bot.cmd_stats))
    app.add_handler(CommandHandler("timeline", bot.cmd_timeline))
    app.add_handler(CommandHandler("wipe", bot.cmd_wipe))

    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print("✅ БОТ ГОТОВ. Теперь он понимает время! 🕰️")
        print("💡 Попробуйте: 'сегодня я иду в кино' → потом спросите 'где я был сегодня?'")

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