#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID NEURAL BRAIN v18.0 - PERSISTENT MIND
✨ Затухающая активация + Фоновый тонус + Персистентное состояние сети
🔗 CausalChain + SemanticClusters + MetaLearning + ChronoNeurons
🎯 Нейросеть «помнит» своё состояние между сессиями — как настоящий мозг

Ключевые улучшения v18:
  • Активация сохраняется с экспоненциальным затуханием по времени
  • Фоновый тонус нейрона зависит от накопленного опыта (experience)
  • ChronoLayer тоже сохраняет/восстанавливает активацию с затуханием
  • «Горячие» нейроны возвращаются быстрее после паузы
  • Новая команда /hotmap — карта горячих нейронов
  • Автосохранение каждые N сообщений (не только при выключении)
  • Улучшенный промпт: бот знает свои «горячие темы» с прошлой сессии
"""
import os, json, re, asyncio, aiohttp, traceback, hashlib, math, shutil, random
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()
TELEGRAM_TOKEN    = os.getenv('TELEGRAM_TOKEN', '')
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

BASE_DIR   = "hybrid_brain_v18"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
for d in [MEMORY_DIR]:
    os.makedirs(d, exist_ok=True)

# Параметры затухания активации
ACTIVATION_DECAY_RATE = 0.1    # λ для exp(-λ * hours): через 24ч → ~9% остатка
TONIC_SCALE           = 0.03   # experience * TONIC_SCALE = базовый тонус (макс 0.25)
TONIC_MAX             = 0.25   # потолок фонового тонуса
AUTOSAVE_EVERY        = 10     # сохранять каждые N сообщений


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
    STOP_WORDS = {
        'в', 'и', 'на', 'с', 'по', 'для', 'от', 'к', 'о', 'у', 'из', 'за',
        'что', 'это', 'как', 'то', 'а', 'но', 'или', 'the', 'is', 'at',
        'не', 'ты', 'я', 'мне', 'себя', 'был', 'была', 'было', 'мой', 'твой',
        'они', 'мы', 'вы', 'он', 'она', 'его', 'её', 'их', 'там', 'тут',
        'вот', 'уже', 'ещё', 'даже', 'тоже', 'просто', 'очень', 'когда'
    }

    @staticmethod
    def extract_keywords(text: str, top_n: int = 8) -> List[str]:
        words = [w.lower() for w in re.findall(r'\b[а-яёa-z]{3,}\b', text, re.IGNORECASE)]
        filtered = [w for w in words if w not in TextUtils.STOP_WORDS]
        return [word for word, _ in Counter(filtered).most_common(top_n)]


# ==================== ЗАТУХАЮЩАЯ АКТИВАЦИЯ ====================
class ActivationDecay:
    """
    Утилита для расчёта затухания активации по прошедшему времени.
    Модель: A(t) = A0 * exp(-λ * hours_elapsed)
    """
    @staticmethod
    def decay_factor(last_active_iso: str, rate: float = ACTIVATION_DECAY_RATE) -> float:
        """Вернуть множитель затухания [0..1] с момента last_active до сейчас"""
        try:
            last = datetime.fromisoformat(last_active_iso)
            hours = (datetime.now() - last).total_seconds() / 3600.0
            return math.exp(-rate * hours)
        except:
            return 1.0

    @staticmethod
    def apply_decay(activation: float, last_active_iso: str,
                    rate: float = ACTIVATION_DECAY_RATE) -> float:
        return activation * ActivationDecay.decay_factor(last_active_iso, rate)

    @staticmethod
    def resting_tonic(experience: float) -> float:
        """Фоновый тонус: опытный нейрон никогда не падает до нуля"""
        return min(TONIC_MAX, experience * TONIC_SCALE)


# ==================== ВРЕМЕННОЙ КОДИРОВЩИК ====================
class TimeEncoder:
    TIME_WORDS = {
        'сейчас':              {'hours': 0},
        'только что':          {'minutes': -30},
        'недавно':             {'hours': -2},
        'сегодня':             {'days': 0},
        'вчера':               {'days': -1},
        'позавчера':           {'days': -2},
        'завтра':              {'days': 1},
        'послезавтра':         {'days': 2},
        'на прошлой неделе':   {'days': -7},
        'на следующей неделе': {'days': 7},
        'утро':                {'hour_range': (6, 12)},
        'утром':               {'hour_range': (6, 12)},
        'день':                {'hour_range': (12, 18)},
        'днём':                {'hour_range': (12, 18)},
        'вечер':               {'hour_range': (18, 24)},
        'вечером':             {'hour_range': (18, 24)},
        'ночь':                {'hour_range': (0, 6)},
        'ночью':               {'hour_range': (0, 6)},
    }

    @staticmethod
    def parse_time_ref(text: str, now: datetime) -> Optional[Dict[str, Any]]:
        text_lower = text.lower()
        for phrase, offset in TimeEncoder.TIME_WORDS.items():
            if phrase in text_lower:
                if 'hour_range' in offset:
                    return {'type': 'time_of_day', 'range': offset['hour_range'], 'date': now.date()}
                delta = timedelta(**{k: v for k, v in offset.items()})
                target = now + delta
                return {'type': 'date_range',
                        'start': target.replace(hour=0, minute=0, second=0),
                        'end':   target.replace(hour=23, minute=59, second=59)}

        m = re.search(r'(\d+)\s*(час|часа|часов|день|дня|дней)\s*назад', text_lower)
        if m:
            val, unit = int(m.group(1)), m.group(2)
            delta = timedelta(hours=val) if 'час' in unit else timedelta(days=val)
            return {'type': 'date_range', 'start': now - delta, 'end': now}
        return None

    @staticmethod
    def get_relative_description(event_time: datetime, now: datetime) -> str:
        diff = now - event_time
        s = diff.total_seconds()
        if s < 60:     return "только что"
        if s < 3600:   return f"{int(s/60)} мин. назад"
        if s < 86400:  return f"{int(s/3600)} ч. назад"
        if diff.days == 1: return "вчера"
        if diff.days < 7:  return f"{diff.days} дн. назад"
        return event_time.strftime("%d.%m.%Y")


# ==================== ЛОГИЧЕСКИЕ ЦЕПОЧКИ ====================
@dataclass
class CausalLink:
    cause:      str
    effect:     str
    link_type:  str   = 'positive'   # positive / negative
    strength:   float = 0.5
    evidence:   int   = 1
    created_at: str   = field(default_factory=lambda: datetime.now().isoformat())
    last_seen:  str   = field(default_factory=lambda: datetime.now().isoformat())

    def reinforce(self, delta: float = 0.12):
        self.strength  = min(1.0, self.strength + delta)
        self.evidence += 1
        self.last_seen = datetime.now().isoformat()

    def decay(self, factor: float = 0.995):
        self.strength = max(0.0, self.strength * factor)


class LogicalChainEngine:
    """Строит и хранит причинно-следственные цепочки."""
    CAUSAL_PATTERNS = [
        (r'(\w{3,})\s+потому\s+что\s+(\w{3,})',    'effect', 'cause',     'positive'),
        (r'(\w{3,})\s+так\s+как\s+(\w{3,})',        'effect', 'cause',     'positive'),
        (r'из-за\s+(\w{3,})\s+(\w{3,})',            'cause',  'effect',    'positive'),
        (r'(\w{3,})\s+приводит\s+к\s+(\w{3,})',     'cause',  'effect',    'positive'),
        (r'(\w{3,})\s+вызывает\s+(\w{3,})',         'cause',  'effect',    'positive'),
        (r'(\w{3,})\s+помогает\s+(\w{3,})',         'cause',  'effect',    'positive'),
        (r'(\w{3,})\s+мешает\s+(\w{3,})',           'cause',  'effect',    'negative'),
        (r'если\s+(\w{3,}).{0,15}то\s+(\w{3,})',   'cause',  'effect',    'positive'),
        (r'(\w{3,})\s+даёт\s+(\w{3,})',             'cause',  'effect',    'positive'),
        (r'(\w{3,})\s+влияет\s+на\s+(\w{3,})',     'cause',  'effect',    'positive'),
        (r'без\s+(\w{3,})\s+нет\s+(\w{3,})',        'cause',  'effect',    'negative'),
    ]

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.links: Dict[str, CausalLink] = {}
        self._load()

    def _key(self, cause: str, effect: str) -> str:
        return f"{cause}→{effect}"

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        for k, v in data.items():
            v.setdefault('link_type', 'positive')
            self.links[k] = CausalLink(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path,
                                   {k: asdict(v) for k, v in self.links.items()})

    def extract_from_text(self, text: str) -> List[CausalLink]:
        found = []
        tl = text.lower()
        for pattern, role1, role2, ltype in self.CAUSAL_PATTERNS:
            for m in re.finditer(pattern, tl):
                w1, w2 = m.group(1), m.group(2)
                if w1 in TextUtils.STOP_WORDS or w2 in TextUtils.STOP_WORDS:
                    continue
                cause  = w1 if role1 == 'cause' else w2
                effect = w2 if role2 == 'effect' else w1
                key = self._key(cause, effect)
                if key in self.links:
                    self.links[key].reinforce()
                else:
                    self.links[key] = CausalLink(cause=cause, effect=effect, link_type=ltype)
                found.append(self.links[key])
        return found

    def build_chain(self, start: str, max_depth: int = 6) -> List[str]:
        chain, current, visited = [start], start, {start}
        for _ in range(max_depth):
            best, best_score = None, 0.0
            for link in self.links.values():
                if link.cause == current and link.effect not in visited:
                    score = link.strength * link.evidence
                    if score > best_score:
                        best, best_score = link.effect, score
            if not best or best_score < 0.1:
                break
            chain.append(best)
            visited.add(best)
            current = best
        return chain

    def find_causes(self, effect: str, min_strength: float = 0.15) -> List[Tuple[str, float]]:
        return sorted(
            [(l.cause, l.strength) for l in self.links.values()
             if l.effect == effect and l.strength >= min_strength],
            key=lambda x: -x[1]
        )

    def get_strong_chains(self, top_n: int = 3) -> List[str]:
        seen_starts: set = set()
        chains = []
        for link in sorted(self.links.values(),
                           key=lambda x: x.strength * x.evidence, reverse=True):
            if link.cause in seen_starts:
                continue
            chain = self.build_chain(link.cause)
            if len(chain) > 1:
                chains.append(" → ".join(chain))
                seen_starts.add(link.cause)
            if len(chains) >= top_n:
                break
        return chains

    def decay_all(self):
        for link in self.links.values():
            link.decay()


# ==================== СЕМАНТИЧЕСКИЕ КЛАСТЕРЫ ====================
@dataclass
class SemanticCluster:
    id:                str
    name:              str
    members:           List[str] = field(default_factory=list)
    centroid_keywords: List[str] = field(default_factory=list)
    strength:          float = 0.5
    access_count:      int   = 0


class SemanticClusterEngine:
    SEED_CLUSTERS = {
        'кино':        ['фильм', 'кино', 'актёр', 'режиссёр', 'сцена', 'сериал', 'смотреть'],
        'технологии':  ['код', 'программа', 'компьютер', 'python', 'алгоритм', 'сервер', 'данные'],
        'еда':         ['еда', 'готовить', 'ресторан', 'вкусно', 'рецепт', 'блюдо', 'кофе'],
        'здоровье':    ['болеть', 'врач', 'здоровье', 'лекарство', 'самочувствие', 'больница'],
        'работа':      ['работа', 'офис', 'задача', 'проект', 'коллега', 'дедлайн', 'задание'],
        'отдых':       ['отдых', 'отпуск', 'путешествие', 'прогулка', 'развлечение', 'игра'],
        'эмоции':      ['рад', 'грусть', 'злость', 'страх', 'счастье', 'настроение', 'чувство'],
        'общение':     ['друг', 'разговор', 'встреча', 'звонок', 'письмо', 'общение', 'семья'],
        'учёба':       ['учить', 'знание', 'школа', 'курс', 'книга', 'читать', 'изучать'],
        'спорт':       ['спорт', 'тренировка', 'бегать', 'gym', 'фитнес', 'игра', 'команда'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.clusters: Dict[str, SemanticCluster] = {}
        self._load()
        if not self.clusters:
            self._init_seeds()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        for k, v in data.items():
            self.clusters[k] = SemanticCluster(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path,
                                   {k: asdict(v) for k, v in self.clusters.items()})

    def _init_seeds(self):
        for name, words in self.SEED_CLUSTERS.items():
            cid = hashlib.md5(name.encode()).hexdigest()[:8]
            self.clusters[cid] = SemanticCluster(
                id=cid, name=name, members=words,
                centroid_keywords=words[:3], strength=0.7
            )

    def classify(self, keywords: List[str]) -> List[Tuple[str, float]]:
        scores = []
        for cid, cluster in self.clusters.items():
            hits = sum(1 for kw in keywords if kw in cluster.members)
            if hits > 0:
                score = hits / max(len(keywords), 1)
                scores.append((cluster.name, score))
                cluster.access_count += 1
        return sorted(scores, key=lambda x: -x[1])

    def learn(self, keywords: List[str], hint_name: str = "") -> Optional[str]:
        """Обучить кластеры на новых словах. Вернуть имя лучшего кластера."""
        classified = self.classify(keywords)
        if classified and classified[0][1] >= 0.25:
            best_name = classified[0][0]
            for cid, cluster in self.clusters.items():
                if cluster.name == best_name:
                    for kw in keywords:
                        if kw not in cluster.members:
                            cluster.members.append(kw)
                            if len(cluster.members) > 120:
                                cluster.members = cluster.members[-120:]
                    cluster.strength = min(1.0, cluster.strength + 0.015)
                    break
            return best_name
        elif len(keywords) >= 2:
            name = hint_name or keywords[0]
            cid  = hashlib.md5(name.encode()).hexdigest()[:8]
            if cid not in self.clusters:
                self.clusters[cid] = SemanticCluster(
                    id=cid, name=name, members=list(keywords),
                    centroid_keywords=keywords[:2], strength=0.3
                )
                print(f"🌱 Новый кластер: [{name}] {keywords}")
            else:
                for kw in keywords:
                    if kw not in self.clusters[cid].members:
                        self.clusters[cid].members.append(kw)
        return None


# ==================== МЕТАОБУЧЕНИЕ ====================
@dataclass
class NeuralGrowthRecord:
    timestamp:    str
    event_type:   str
    description:  str
    impact_score: float = 0.0


class MetaLearner:
    LEVEL_THRESHOLDS = [0, 50, 150, 350, 750, 1500, 3000, 6000]
    LEVEL_NAMES = {1:"Новичок", 2:"Ученик", 3:"Знающий", 4:"Опытный",
                   5:"Мастер",  6:"Эксперт", 7:"Гений",  8:"Оракул"}

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.xp                 = 0
        self.level              = 1
        self.total_interactions = 0
        self.growth_log:  List[NeuralGrowthRecord] = []
        self.insights:    List[str] = []
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self.xp                 = data.get('xp', 0)
        self.level              = data.get('level', 1)
        self.total_interactions = data.get('total_interactions', 0)
        self.insights           = data.get('insights', [])
        for r in data.get('growth_log', []):
            self.growth_log.append(NeuralGrowthRecord(**r))

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'xp': self.xp, 'level': self.level,
            'total_interactions': self.total_interactions,
            'insights': self.insights,
            'growth_log': [asdict(r) for r in self.growth_log[-100:]]
        })

    def gain_xp(self, amount: int, event: str, description: str = "") -> Optional[str]:
        self.xp += amount
        self.total_interactions += 1
        self.growth_log.append(NeuralGrowthRecord(
            timestamp=datetime.now().isoformat(),
            event_type=event, description=description,
            impact_score=amount / 10.0
        ))
        return self._check_level_up()

    def _check_level_up(self) -> Optional[str]:
        for lvl, threshold in enumerate(self.LEVEL_THRESHOLDS[1:], start=2):
            if self.xp >= threshold and self.level < lvl:
                self.level = lvl
                return (f"🆙 УРОВЕНЬ {lvl} — {self.LEVEL_NAMES.get(lvl, '?')}!\n"
                        f"📊 Опыт: {self.xp} | Диалогов: {self.total_interactions}")
        return None

    def add_insight(self, insight: str):
        if insight and insight not in self.insights:
            self.insights.append(insight)
            if len(self.insights) > 60:
                self.insights = self.insights[-60:]

    def get_insights_for_prompt(self, n: int = 2) -> str:
        if not self.insights:
            return ""
        return "📌 Из прошлых сессий:\n" + "\n".join(f"  • {i}" for i in self.insights[-n:])

    def get_status(self) -> str:
        name = self.LEVEL_NAMES.get(self.level, '?')
        xp_next = next((t for t in self.LEVEL_THRESHOLDS if t > self.xp), None)
        progress = f" (до след. уровня: {xp_next - self.xp} XP)" if xp_next else " (MAX)"
        return (f"Уровень {self.level} — {name}{progress}\n"
                f"Опыт: {self.xp} | Диалогов: {self.total_interactions}\n"
                f"Инсайтов: {len(self.insights)}")


# ==================== ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ ====================
@dataclass
class EpisodicMemory:
    id:             str
    timestamp:      str
    content:        str
    keywords:       List[str]
    temporal_tags:  List[str]
    user_emotion:   Optional[str]
    location_hints: List[str]
    sequence_id:    int
    cluster_name:   Optional[str] = None
    importance:     float = 0.5


class TemporalMemory:
    def __init__(self, user_id: str, save_dir: str):
        self.user_id = user_id
        self.file    = os.path.join(save_dir, "episodic_time.json")
        self.events: List[EpisodicMemory] = []
        self.sequence_counter = 0
        self.now = datetime.now()
        data = FileManager.safe_load_json(self.file)
        if data:
            self.sequence_counter = data.get('seq', 0)
            for e in data.get('events', []):
                e.setdefault('cluster_name', None)
                e.setdefault('importance', 0.5)
                self.events.append(EpisodicMemory(**e))
        print(f"⏳ Эпизодов загружено: {len(self.events)}")

    def add_event(self, content: str, keywords: List[str],
                  cluster_name: str = None, now: datetime = None) -> EpisodicMemory:
        if now is None:
            now = datetime.now()
            self.now = now

        temporal_tags = [tw for tw in
                         ['сегодня','вчера','завтра','утром','вечером','ночью','сейчас']
                         if tw in content.lower()]
        loc_match     = re.findall(r'(?:в|на|из|к)\s+([а-яёa-z]{3,})', content.lower())
        location_hints= [l for l in loc_match if l not in {'дом','домой','город','место'}]
        importance    = min(1.0, 0.3 + len(content)/500 + (0.2 if temporal_tags else 0))

        event = EpisodicMemory(
            id=hashlib.md5(f"{content}{now.isoformat()}".encode()).hexdigest()[:12],
            timestamp=now.isoformat(), content=content, keywords=keywords,
            temporal_tags=temporal_tags, user_emotion=self._detect_emotion(content),
            location_hints=location_hints, sequence_id=self.sequence_counter,
            cluster_name=cluster_name, importance=importance
        )
        self.events.append(event)
        self.sequence_counter += 1
        if len(self.events) > 300:
            self.events.sort(key=lambda e: e.importance * 0.5 +
                             datetime.fromisoformat(e.timestamp).timestamp() * 0.5e-10)
            self.events = self.events[-300:]
            self.events.sort(key=lambda e: e.sequence_id)
        return event

    def _detect_emotion(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ['рад','хорошо','отлично','супер','люблю','нравится','кайф']):
            return 'positive'
        if any(w in t for w in ['грустно','плохо','устал','злой','проблема','страшно','боюсь']):
            return 'negative'
        return None

    def query_by_time(self, time_ref: str, now: datetime = None) -> List[EpisodicMemory]:
        if now is None:
            now = datetime.now()
        tr = TimeEncoder.parse_time_ref(time_ref, now)
        if not tr:
            return []
        results = []
        for event in reversed(self.events):
            et = datetime.fromisoformat(event.timestamp)
            if tr['type'] == 'date_range':
                if tr['start'] <= et <= tr['end']:
                    results.append(event)
            elif tr['type'] == 'time_of_day':
                if et.date() == tr['date'] and tr['range'][0] <= et.hour < tr['range'][1]:
                    results.append(event)
        return results[:10]

    def get_timeline_summary(self, hours: int = 24) -> str:
        now    = datetime.now()
        cutoff = now - timedelta(hours=hours)
        recent = [e for e in self.events if datetime.fromisoformat(e.timestamp) >= cutoff]
        if not recent:
            return "За это время ничего не зафиксировано."
        lines = []
        for e in recent[-6:]:
            rel     = TimeEncoder.get_relative_description(datetime.fromisoformat(e.timestamp), now)
            cluster = f"[{e.cluster_name}] " if e.cluster_name else ""
            lines.append(f"[{rel}] {cluster}{e.content[:90]}")
        return "\n".join(lines)

    def save(self):
        FileManager.safe_save_json(self.file, {
            'seq': self.sequence_counter,
            'events': [asdict(e) for e in self.events]
        })


# ==================== ВРЕМЕННЫЕ НЕЙРОНЫ (ChronoLayer) ====================
@dataclass
class ChronoNode:
    id:             str
    label:          str
    node_type:      str
    activation:     float = 0.0
    last_active:    str   = field(default_factory=lambda: datetime.now().isoformat())
    use_count:      int   = 0          # сколько раз активировался
    linked_events:  List[str] = field(default_factory=list)


class ChronoNeuralLayer:
    """
    Временные нейроны с сохранением затухающей активации.
    Часто используемые временные концепты (напр. «вечером») имеют
    ненулевой тонус при следующей загрузке.
    """
    def __init__(self):
        self.nodes: Dict[str, ChronoNode] = {}
        self._init_base_nodes()

    def _init_base_nodes(self):
        for label, ntype in [
            ('вчера','temporal_ref'), ('сегодня','temporal_ref'), ('завтра','temporal_ref'),
            ('утром','time_of_day'),  ('днём','time_of_day'),     ('вечером','time_of_day'),
            ('ночью','time_of_day'),  ('утро','time_of_day'),     ('вечер','time_of_day'),
            ('раньше','sequence'),    ('потом','sequence'),        ('до','sequence'),
            ('после','sequence'),     ('всегда','sequence'),       ('никогда','sequence'),
        ]:
            nid = hashlib.md5(f"chrono_{label}".encode()).hexdigest()[:8]
            if nid not in self.nodes:
                self.nodes[nid] = ChronoNode(id=nid, label=label, node_type=ntype)

    def restore_from_saved(self, saved_nodes: List[Dict]):
        """Восстановить состояние с затуханием по прошедшему времени"""
        for nd in saved_nodes:
            nid = nd['id']
            if nid in self.nodes:
                raw_act  = nd.get('activation', 0.0)
                last_act = nd.get('last_active', datetime.now().isoformat())
                use_cnt  = nd.get('use_count', 0)
                # Экспоненциальное затухание + тонус от частоты использования
                decayed = ActivationDecay.apply_decay(raw_act, last_act, rate=0.08)
                tonic   = min(0.15, use_cnt * 0.005)
                self.nodes[nid].activation  = max(decayed, tonic)
                self.nodes[nid].last_active = last_act
                self.nodes[nid].use_count   = use_cnt
                self.nodes[nid].linked_events = nd.get('linked_events', [])

    def to_save_list(self) -> List[Dict]:
        return [asdict(n) for n in self.nodes.values()]

    def process_temporal_input(self, text: str) -> List[ChronoNode]:
        activated, tl = [], text.lower()
        for node in self.nodes.values():
            if node.label in tl:
                node.activation  = min(1.0, node.activation + 0.8)
                node.last_active = datetime.now().isoformat()
                node.use_count  += 1
                activated.append(node)
        for node in self.nodes.values():
            if node not in activated:
                node.activation *= 0.88
        return activated

    def get_active_labels(self) -> List[str]:
        return [n.label for n in self.nodes.values() if n.activation > 0.3]

    def link_event(self, event: EpisodicMemory, nodes: List[ChronoNode]):
        for node in nodes:
            if event.id not in node.linked_events:
                node.linked_events.append(event.id)
                if len(node.linked_events) > 60:
                    node.linked_events = node.linked_events[-60:]


# ==================== НЕЙРО-КОРА (с персистентной активацией) ====================
@dataclass
class NeuroNode:
    id:              str
    label:           str
    category:        str
    activation:      float = 0.0
    last_active:     str   = field(default_factory=lambda: datetime.now().isoformat())
    created_at:      str   = field(default_factory=lambda: datetime.now().isoformat())
    access_count:    int   = 0
    experience:      float = 0.0   # Накопленный опыт — определяет тонус
    temporal_weight: float = 1.0


@dataclass
class NeuroSynapse:
    source:         str
    target:         str
    weight:         float = 0.5
    plasticity:     float = 0.8
    temporal_decay: float = 0.0
    last_fired:     str   = ""
    fire_count:     int   = 0


class DynamicNeuralCortex:
    """
    Нейро-кора с персистентной активацией.

    При сохранении: записываем текущую активацию + время last_active.
    При загрузке: применяем exp(-λ*Δt) к сохранённой активации,
                  затем добавляем фоновый тонус от experience.
    Итог: опытные нейроны никогда не «забываются» полностью.
    """

    def __init__(self, user_id: str, save_dir: str):
        self.user_id   = user_id
        self.save_path = os.path.join(save_dir, "cortex_graph.json")
        self.nodes:    Dict[str, NeuroNode]    = {}
        self.synapses: Dict[str, NeuroSynapse] = {}
        self.chrono_layer = ChronoNeuralLayer()

        data = FileManager.safe_load_json(self.save_path)
        if data and 'nodes' in data:
            self._load_with_decay(data)
        else:
            self._create_node("Приветствие", "action")
            self._create_node("Помощь", "action")
            self.save()

    def _load_with_decay(self, data: Dict):
        """
        Загрузка с применением затухания активации.
        Активация = max(saved * exp(-λ*Δt),  tonic(experience))
        """
        now_iso = datetime.now().isoformat()
        for n in data.get('nodes', []):
            n.setdefault('experience', 0.0)
            n.setdefault('temporal_weight', 1.0)
            n.setdefault('last_active', now_iso)

            raw_act  = n.get('activation', 0.0)
            last_act = n['last_active']
            exp_val  = n['experience']

            # Затухание по времени
            decayed = ActivationDecay.apply_decay(raw_act, last_act)
            # Фоновый тонус от опыта
            tonic   = ActivationDecay.resting_tonic(exp_val)
            # Итоговая активация — не ниже тонуса
            n['activation'] = max(decayed, tonic)

            self.nodes[n['id']] = NeuroNode(**n)

        for s in data.get('synapses', []):
            s.setdefault('fire_count', 0)
            k = f"{s['source']}->{s['target']}"
            self.synapses[k] = NeuroSynapse(**s)

        # Восстановить ChronoLayer
        if 'chrono_nodes' in data:
            self.chrono_layer.restore_from_saved(data['chrono_nodes'])

        # Статистика при загрузке
        warm = sum(1 for n in self.nodes.values() if n.activation > 0.1)
        print(f"🧠 Кора: {len(self.nodes)} нейронов ({warm} тёплых), "
              f"{len(self.synapses)} синапсов")

    def _create_node(self, label: str, category: str = "concept") -> NeuroNode:
        node_id = hashlib.md5(label.encode()).hexdigest()[:8]
        if node_id not in self.nodes:
            self.nodes[node_id] = NeuroNode(id=node_id, label=label, category=category)
        return self.nodes[node_id]

    def process_input(self, text: str, now: datetime = None) -> Tuple[List[str], List[ChronoNode]]:
        if now is None:
            now = datetime.now()

        keywords         = TextUtils.extract_keywords(text, top_n=10)
        activated_labels = []
        now_iso          = now.isoformat()

        for i, kw1 in enumerate(keywords):
            node1 = self._create_node(kw1)
            node1.access_count += 1
            node1.activation    = 1.0
            node1.last_active   = now_iso
            node1.experience   += 0.12   # Накапливаем опыт
            activated_labels.append(kw1)
            for kw2 in keywords[i + 1:]:
                node2 = self._create_node(kw2)
                self._strengthen(node1.id, node2.id, 0.1)

        # Временные нейроны
        chrono_nodes = self.chrono_layer.process_temporal_input(text)

        # Распространение активации по сильным синапсам
        final_context = set(activated_labels)
        for label in activated_labels:
            nid = hashlib.md5(label.encode()).hexdigest()[:8]
            for key, syn in list(self.synapses.items()):
                if syn.source == nid and syn.weight > 0.4:
                    target = self.nodes.get(syn.target)
                    if target:
                        final_context.add(target.label)
                        target.activation  = min(1.0, target.activation + 0.45)
                        target.last_active = now_iso
                        syn.fire_count += 1

        # Затухание текущего шага (не трогает тонус)
        for node in self.nodes.values():
            tonic = ActivationDecay.resting_tonic(node.experience)
            node.activation = max(node.activation * 0.88, tonic)

        return list(final_context), chrono_nodes

    def _strengthen(self, src: str, tgt: str, reward: float):
        if src == tgt:
            return
        for k, (s, t) in [(f"{src}->{tgt}", (src, tgt)), (f"{tgt}->{src}", (tgt, src))]:
            if k not in self.synapses:
                self.synapses[k] = NeuroSynapse(source=s, target=t, weight=0.1)
            syn = self.synapses[k]
            syn.weight      = min(1.0, syn.weight + reward * syn.plasticity)
            syn.last_fired  = datetime.now().isoformat()
            syn.plasticity *= 0.995
            syn.fire_count += 1

    def associate(self, concept: str, top_n: int = 6) -> List[Tuple[str, float]]:
        nid    = hashlib.md5(concept.encode()).hexdigest()[:8]
        assocs = []
        for key, syn in self.synapses.items():
            if syn.source == nid and syn.weight > 0.15:
                target = self.nodes.get(syn.target)
                if target:
                    assocs.append((target.label, syn.weight))
        return sorted(assocs, key=lambda x: -x[1])[:top_n]

    def get_hot_nodes(self, top_n: int = 8) -> List[NeuroNode]:
        """Горячие нейроны — по activation (включая тонус от experience)"""
        return sorted(
            [n for n in self.nodes.values() if n.activation > 0.05],
            key=lambda n: n.activation, reverse=True
        )[:top_n]

    def get_experienced_nodes(self, top_n: int = 5) -> List[NeuroNode]:
        return sorted(self.nodes.values(), key=lambda n: n.experience, reverse=True)[:top_n]

    def get_system_prompt_modifiers(self, chrono_context: List[str],
                                    chains: List[str],
                                    clusters: List[Tuple[str, float]],
                                    insights: str) -> str:
        parts = []

        # Режим по горячим нейронам
        hot = self.get_hot_nodes(3)
        for node in hot:
            if node.label in {'код', 'python', 'программа', 'алгоритм', 'сервер'}:
                parts.append("Контекст: программирование. Давай точные решения с кодом.")
                break
            if node.label in {'грустно', 'проблема', 'устал', 'страшно', 'боюсь'}:
                parts.append("Контекст: эмоциональная поддержка. Будь тёплым и внимательным.")
                break

        # Горячие темы из прошлых сессий
        if hot:
            hot_labels = [n.label for n in hot[:5]]
            parts.append(f"Горячие темы: {', '.join(hot_labels)}.")

        # Временной контекст
        if chrono_context:
            parts.append(f"Временной контекст: {', '.join(chrono_context)}.")

        # Цепочки
        if chains:
            parts.append(f"Известные закономерности: {'; '.join(chains)}.")

        # Кластер темы
        if clusters:
            parts.append(f"Тема: {clusters[0][0]}.")

        # Инсайты из прошлых сессий
        if insights:
            parts.append(insights)

        if not parts:
            parts.append("Ты умный ассистент с долговременной памятью.")

        return " ".join(parts)

    def reinforce_path(self, keywords: List[str], chrono_nodes: List[ChronoNode], success: bool):
        reward = 0.15 if success else -0.04
        for i, kw1 in enumerate(keywords):
            id1 = hashlib.md5(kw1.encode()).hexdigest()[:8]
            for kw2 in keywords[i + 1:]:
                id2 = hashlib.md5(kw2.encode()).hexdigest()[:8]
                self._strengthen(id1, id2, reward)
        for cn in chrono_nodes:
            cn.activation = min(1.0, cn.activation + reward)

    def prune_weak_synapses(self, threshold: float = 0.04) -> int:
        before = len(self.synapses)
        self.synapses = {k: v for k, v in self.synapses.items() if v.weight > threshold}
        return before - len(self.synapses)

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'nodes':       [asdict(n) for n in self.nodes.values()],
            'synapses':    [asdict(s) for s in self.synapses.values()],
            'chrono_nodes': self.chrono_layer.to_save_list(),   # ← новое: сохраняем ChronoLayer
            'saved_at':    datetime.now().isoformat()
        })

    def get_stats(self) -> Dict:
        strong = sum(1 for s in self.synapses.values() if s.weight > 0.6)
        warm   = sum(1 for n in self.nodes.values() if n.activation > 0.1)
        return {
            'neurons': len(self.nodes), 'warm_neurons': warm,
            'synapses': len(self.synapses), 'strong_synapses': strong,
            'chrono_nodes': len(self.chrono_layer.nodes),
            'active_chrono': len([n for n in self.chrono_layer.nodes.values()
                                  if n.activation > 0.3]),
        }


# ==================== РЕФЛЕКСИЯ ====================
class ReflectionEngine:
    INTERVAL = 15   # каждые 15 сообщений

    def __init__(self, llm):
        self.llm      = llm
        self._counter = 0

    def tick(self) -> bool:
        self._counter += 1
        return self._counter % self.INTERVAL == 0

    async def reflect(self, memory: 'HybridMemorySystem') -> Optional[str]:
        chains   = memory.causal.get_strong_chains(4)
        hot      = memory.cortex.get_hot_nodes(5)
        exp_top  = memory.cortex.get_experienced_nodes(3)

        if not chains and not hot:
            return None

        prompt = (
            "Ты — аналитик знаний. Кратко (1-2 предложения, ≤35 слов) опиши "
            "интересы/паттерны пользователя на основе:\n"
            f"Горячие темы: {[n.label for n in hot]}\n"
            f"Опытные концепты: {[n.label for n in exp_top]}\n"
            f"Логические цепочки: {chains}\n"
            "Только факты, без домыслов."
        )
        result = await self.llm.generate(prompt, temp=0.4)
        if result and len(result) > 8:
            memory.meta.add_insight(result)
            return result
        return None


# ==================== ГИБРИДНАЯ ПАМЯТЬ ====================
class HybridMemorySystem:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir     = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.dir, exist_ok=True)

        self.st_file    = os.path.join(self.dir, "short_term.json")
        self.short_term: List[Dict] = FileManager.safe_load_json(self.st_file, [])

        self.cortex   = DynamicNeuralCortex(user_id, self.dir)
        self.temporal = TemporalMemory(user_id, self.dir)
        self.causal   = LogicalChainEngine(os.path.join(self.dir, "causal.json"))
        self.clusters = SemanticClusterEngine(os.path.join(self.dir, "clusters.json"))
        self.meta     = MetaLearner(os.path.join(self.dir, "meta.json"))
        self.now      = datetime.now()
        self._msg_count = 0

    async def process(self, text: str):
        self.now = datetime.now()

        # 1. Нейро-активация
        concepts, chrono_nodes = self.cortex.process_input(text, self.now)

        # 2. Ключевые слова + кластеры
        keywords     = TextUtils.extract_keywords(text)
        cluster_name = self.clusters.learn(keywords)
        clusters     = self.clusters.classify(keywords)

        # 3. Логические цепочки
        causal_links = self.causal.extract_from_text(text)
        xp_gain  = 3 + len(causal_links) * 5
        xp_event = "chain_found" if causal_links else "interaction"
        xp_msg   = self.meta.gain_xp(xp_gain, xp_event,
                                      str([f"{l.cause}→{l.effect}" for l in causal_links]))

        # 4. Эпизодическая память
        event = self.temporal.add_event(text, keywords, cluster_name, self.now)
        self.chrono_layer_link(event, chrono_nodes)

        # 5. Системный промпт
        chains   = self.causal.get_strong_chains(3)
        insights = self.meta.get_insights_for_prompt(2)
        sys_mod  = self.cortex.get_system_prompt_modifiers(
            self.cortex.chrono_layer.get_active_labels(),
            chains, clusters[:2], insights
        )

        # 6. Контекст диалога
        st_ctx = "\n".join([f"{m['role']}: {m['content']}" for m in self.short_term[-6:]])

        # 7. Короткая память
        self.short_term.append({'role': 'user', 'content': text,
                                 'time': self.now.isoformat()})
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]

        # 8. Автосохранение
        self._msg_count += 1
        if self._msg_count % AUTOSAVE_EVERY == 0:
            self.save_all()
            print(f"💾 Автосохранение (сообщение #{self._msg_count})")

        return st_ctx, sys_mod, concepts, chrono_nodes, xp_msg

    def chrono_layer_link(self, event: EpisodicMemory, chrono_nodes: List[ChronoNode]):
        self.cortex.chrono_layer.link_event(event, chrono_nodes)

    def handle_temporal_query(self, text: str) -> Optional[str]:
        if not any(w in text.lower() for w in
                   ['был','делал','ходил','говорил','где','что делал',
                    'вчера','раньше','сегодня','утром','вечером']):
            return None
        events = self.temporal.query_by_time(text, self.now)
        if events:
            lines = []
            for e in events[:4]:
                rel     = TimeEncoder.get_relative_description(
                    datetime.fromisoformat(e.timestamp), self.now)
                cluster = f"[{e.cluster_name}] " if e.cluster_name else ""
                lines.append(f"{rel}: {cluster}{e.content}")
            return "🕰️ Из памяти:\n" + "\n".join(lines)
        return None

    def handle_logic_query(self, text: str) -> Optional[str]:
        tl = text.lower()
        if not any(w in tl for w in ['почему','зачем','причина','что будет','что вызывает']):
            return None
        for kw in TextUtils.extract_keywords(text):
            chain = self.causal.build_chain(kw)
            if len(chain) > 1:
                return f"🔗 Цепочка: {' → '.join(chain)}"
            causes = self.causal.find_causes(kw)
            if causes:
                return "🔍 Причины «{}»:\n{}".format(
                    kw, "\n".join(f"  • {c} ({s:.2f})" for c, s in causes[:3]))
        return None

    def add_response(self, text: str):
        self.short_term.append({'role': 'assistant', 'content': text,
                                 'time': datetime.now().isoformat()})
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]
        FileManager.safe_save_json(self.st_file, self.short_term)

    def save_all(self):
        self.cortex.save()
        self.temporal.save()
        self.causal.save()
        self.clusters.save()
        self.meta.save()
        FileManager.safe_save_json(self.st_file, self.short_term)

    def maintenance(self):
        if self.meta.total_interactions % 60 == 0:
            pruned = self.cortex.prune_weak_synapses(0.04)
            self.causal.decay_all()
            if pruned:
                print(f"✂️ Удалено {pruned} слабых синапсов")


# ==================== LLM ====================
class LLMInterface:
    def __init__(self, url: str, key: str):
        self.url, self.key = url, key
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def generate(self, prompt: str, system: str = None, temp: float = 0.7) -> str:
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
        self.llm       = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.users:    Dict[str, HybridMemorySystem] = {}
        self.reflector = None
        self.stop_flag = False

    def get_brain(self, uid: str) -> HybridMemorySystem:
        if uid not in self.users:
            self.users[uid] = HybridMemorySystem(uid)
        return self.users[uid]

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        uid   = str(update.effective_user.id)
        text  = update.message.text
        brain = self.get_brain(uid)
        await context.bot.send_chat_action(uid, "typing")

        # Обработка
        st_ctx, sys_mod, concepts, chrono_nodes, xp_msg = await brain.process(text)

        # Временной запрос?
        ta = brain.handle_temporal_query(text)
        if ta:
            brain.add_response(ta)
            await update.message.reply_text(ta)
            if xp_msg:
                await update.message.reply_text(xp_msg)
            return

        # Логический запрос?
        la = brain.handle_logic_query(text)
        if la:
            brain.add_response(la)
            await update.message.reply_text(la)
            return

        # Промпт для LLM
        full_system = (
            f"{sys_mod}\n\n"
            f"=== ИСТОРИЯ ===\n{st_ctx}\n\n"
            f"=== ВРЕМЯ ===\n{brain.now.strftime('%H:%M %d.%m.%Y')}"
        )
        response = await self.llm.generate(text, system=full_system)

        brain.cortex.reinforce_path(concepts, chrono_nodes, len(response) > 10)

        if self.reflector and self.reflector.tick():
            asyncio.create_task(self._reflect(brain))

        brain.maintenance()
        brain.add_response(response)

        await update.message.reply_text(response)
        if xp_msg:
            await update.message.reply_text(xp_msg)

    async def _reflect(self, brain: HybridMemorySystem):
        try:
            ins = await self.reflector.reflect(brain)
            if ins:
                print(f"💡 Инсайт: {ins[:80]}")
        except Exception as e:
            print(f"⚠️ Reflection: {e}")

    # ==================== КОМАНДЫ ====================

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        s     = brain.cortex.get_stats()
        chains = brain.causal.get_strong_chains(3)
        msg = (
            f"🧠 PERSISTENT MIND v18.0\n{'='*32}\n"
            f"🔹 Нейроны: {s['neurons']} (тёплых: {s['warm_neurons']})\n"
            f"🔹 Синапсы: {s['synapses']} (сильных: {s['strong_synapses']})\n"
            f"🔹 Chrono-нейроны: {s['chrono_nodes']} (активных: {s['active_chrono']})\n"
            f"🔹 Эпизодов: {len(brain.temporal.events)}\n"
            f"🔹 Причинных связей: {len(brain.causal.links)}\n"
            f"🔹 Кластеров: {len(brain.clusters.clusters)}\n"
            f"{'='*32}\n"
            f"📈 {brain.meta.get_status()}\n"
            f"{'='*32}\n"
            f"🔗 Цепочки:\n  " + ("\n  ".join(chains) if chains else "Пока нет")
        )
        await update.message.reply_text(msg)

    async def cmd_hotmap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Карта горячих нейронов — показывает «что помнит» сеть между сессиями"""
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        hot   = brain.cortex.get_hot_nodes(12)
        exp   = brain.cortex.get_experienced_nodes(5)

        if not hot:
            await update.message.reply_text("Горячих нейронов пока нет.")
            return

        lines = ["🔥 ГОРЯЧАЯ КАРТА НЕЙРОСЕТИ\n"]
        lines.append("Активные прямо сейчас:")
        for n in hot[:6]:
            bar = "█" * int(n.activation * 10)
            lines.append(f"  {n.label:<15} {bar} {n.activation:.2f}  (exp:{n.experience:.1f})")

        lines.append("\n🏆 Самые опытные нейроны:")
        for n in exp:
            tonic = ActivationDecay.resting_tonic(n.experience)
            lines.append(f"  {n.label:<15} exp:{n.experience:.1f}  тонус:{tonic:.3f}")

        # ChronoLayer
        chrono_hot = sorted(
            [n for n in brain.cortex.chrono_layer.nodes.values() if n.activation > 0.05],
            key=lambda x: -x.activation
        )[:5]
        if chrono_hot:
            lines.append("\n⏱ Временные нейроны:")
            for n in chrono_hot:
                lines.append(f"  {n.label:<12} act:{n.activation:.2f}  uses:{n.use_count}")

        await update.message.reply_text("\n".join(lines))

    async def cmd_chain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        if not context.args:
            await update.message.reply_text("Использование: /chain <слово>")
            return
        word  = context.args[0].lower()
        chain = brain.causal.build_chain(word)
        if len(chain) > 1:
            await update.message.reply_text(f"🔗 Цепочка '{word}':\n{' → '.join(chain)}")
        else:
            causes = brain.causal.find_causes(word)
            if causes:
                await update.message.reply_text(
                    f"🔍 Причины '{word}':\n" +
                    "\n".join(f"  {c} ({s:.2f})" for c, s in causes))
            else:
                await update.message.reply_text(f"Цепочек для '{word}' нет пока.")

    async def cmd_assoc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        if not context.args:
            await update.message.reply_text("Использование: /assoc <слово>")
            return
        word   = context.args[0].lower()
        assocs = brain.cortex.associate(word)
        if assocs:
            await update.message.reply_text(
                f"🧩 Ассоциации '{word}':\n" +
                "\n".join(f"  {a} ({w:.2f})" for a, w in assocs))
        else:
            await update.message.reply_text(f"Ассоциаций для '{word}' пока нет.")

    async def cmd_timeline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        hours = int(context.args[0]) if context.args and context.args[0].isdigit() else 24
        await update.message.reply_text(
            f"🕰️ События за {hours}ч:\n{brain.temporal.get_timeline_summary(hours)}")

    async def cmd_clusters(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        top   = sorted(brain.clusters.clusters.values(),
                       key=lambda c: c.access_count, reverse=True)[:8]
        if not top:
            await update.message.reply_text("Кластеры пока не сформированы.")
            return
        lines = ["🗂 КЛАСТЕРЫ ЗНАНИЙ\n"]
        for c in top:
            lines.append(f"📌 [{c.name}] ({c.access_count} обращений)")
            lines.append(f"   {', '.join(c.members[:7])}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        if not brain.meta.insights:
            await update.message.reply_text("Инсайтов пока нет. Продолжайте общаться!")
            return
        lines = ["💡 НАКОПЛЕННЫЕ ИНСАЙТЫ\n"]
        for i, ins in enumerate(brain.meta.insights[-10:], 1):
            lines.append(f"{i}. {ins}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_wipe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        if uid in self.users:
            d = self.users.pop(uid).dir
            if os.path.exists(d):
                shutil.rmtree(d)
            await update.message.reply_text("🧠 Полная очистка памяти выполнена.")
        else:
            await update.message.reply_text("Пользователь не найден.")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 PERSISTENT MIND v18.0\n\n"
            "Нейросеть помнит своё состояние между перезапусками!\n"
            "Активация затухает со временем, но опытные нейроны\n"
            "всегда сохраняют фоновый тонус.\n\n"
            "📌 КОМАНДЫ:\n"
            "/stats    — статистика и уровень развития\n"
            "/hotmap   — карта горячих нейронов 🔥\n"
            "/chain <слово> — логическая цепочка\n"
            "/assoc <слово> — ассоциации\n"
            "/clusters — смысловые кластеры\n"
            "/insights — накопленные наблюдения\n"
            "/timeline [ч] — хронология событий\n"
            "/wipe     — полная очистка\n\n"
            "💡 Подсказки:\n"
            "• «X вызывает Y», «если A то B» — учит цепочкам\n"
            "• «где я был вчера?» — ищет в памяти\n"
            "• «почему X?» — строит причинную цепочку\n"
            "• Чем больше общаешься — тем умнее сеть!"
        )

    async def shutdown(self):
        print("\n💾 Финальное сохранение...")
        for b in self.users.values():
            b.save_all()
        await self.llm.close()
        print("✅ Остановлено")


# ==================== ЗАПУСК ====================
async def main():
    print("🚀 PERSISTENT MIND v18.0 STARTING...")
    if not TELEGRAM_TOKEN:
        print("❌ Нет TELEGRAM_TOKEN в .env")
        return

    bot           = HybridBot()
    bot.reflector = ReflectionEngine(bot.llm)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    for cmd, handler in [
        ("stats",    bot.cmd_stats),
        ("hotmap",   bot.cmd_hotmap),
        ("timeline", bot.cmd_timeline),
        ("chain",    bot.cmd_chain),
        ("assoc",    bot.cmd_assoc),
        ("clusters", bot.cmd_clusters),
        ("insights", bot.cmd_insights),
        ("wipe",     bot.cmd_wipe),
        ("help",     bot.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print("✅ БОТ ГОТОВ. Нейросеть помнит себя между сессиями! 🧠")
        print("💡 /hotmap — посмотреть тёплые нейроны после перезапуска")
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