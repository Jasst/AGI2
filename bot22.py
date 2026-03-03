#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID NEURAL BRAIN v23.0 — AGI APPROXIMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Новое в v23 (6 AGI-модулей):

  🎯 Система целей (GoalSystem):
       • Долгосрочные, среднесрочные, краткосрочные цели
       • Автоматическое определение цели из диалога
       • Декомпозиция цели на подцели
       • Отслеживание прогресса через сессии
       • Команда /goals — текущие цели и прогресс

  🧪 Активное обучение (ActiveLearning):
       • Замечает пробелы: "я не знаю X — нужно выяснить"
       • Формулирует гипотезы и проверяет их в диалоге
       • Обнаруживает противоречия между старыми и новыми фактами
       • Журнал гипотез с подтверждением/опровержением
       • Команда /hypotheses

  🪞 Теория разума (TheoryOfMind):
       • Модель знаний пользователя: что он знает/не знает
       • Модель убеждений: во что он верит
       • Модель мотивации: зачем он на самом деле спрашивает
       • Определяет заблуждения и мягко корректирует
       • Команда /mindmodel

  📋 Планировщик задач (TaskPlanner):
       • Декомпозиция сложных запросов на подзадачи
       • Отслеживание выполнения плана
       • Параллельные и последовательные шаги
       • Команда /plan

  ⚡ Инициативность (ProactiveBehavior):
       • Замечает паттерны без запроса
       • Предлагает темы на основе истории
       • Предупреждает о противоречиях
       • Срабатывает каждые N сообщений

  🔄 Самомодификация (SelfModification):
       • Обновляет стратегии поведения на основе ошибок
       • Ведёт журнал "что сработало / не сработало"
       • Обновляет приоритеты фаз пульса
       • Команда /selfmodel

Сохранено из v22: CognitivePulse (5 фаз ритма мышления),
  умная L1, эмоц. RAG, активное забывание, двойная рефлексия,
  фидбек температуры, Марков 2-го порядка, глубокие цепочки,
  PulseJournal, ChronoLayer, MetaLearner, профиль, эмоц. дуга.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, re, asyncio, aiohttp, traceback, hashlib, math, shutil, random
from collections import Counter, deque
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
TELEGRAM_TOKEN    = os.getenv('TELEGRAM_TOKEN', '')
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

BASE_DIR   = "hybrid_brain_v23"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# Нейросеть
ACTIVATION_DECAY_RATE  = 0.1
TONIC_SCALE            = 0.03
TONIC_MAX              = 0.25
AUTOSAVE_EVERY         = 10
ARCHIVE_AFTER_DAYS     = 30
ARCHIVE_ACTIVATION_CAP = 0.05

# RAG v2
RAG_TOP_K         = 4
RAG_MIN_SCORE     = 0.12
RAG_BM25_K1       = 1.5
RAG_BM25_B        = 0.75
RAG_EMOTION_BOOST = 0.25

# L1 Smart Memory
L1_RECENT_COUNT   = 3
L1_RELEVANT_COUNT = 3
L1_MAX_HISTORY    = 30

# Стохастический дрейф
DRIFT_EXPLORE_PROB = 0.08
DRIFT_NOISE_SCALE  = 0.05
DRIFT_ASSOC_EVERY  = 7

# Температура
TEMP_TECHNICAL     = 0.30
TEMP_EMOTIONAL     = 0.85
TEMP_CREATIVE      = 0.90
TEMP_DEFAULT       = 0.65
TEMP_FEEDBACK_STEP = 0.05

# Рефлексия
REFLECTION_FAST_EVERY = 10
REFLECTION_DEEP_EVERY = 50

# Причинные цепочки
CAUSAL_MAX_DEPTH    = 8
CAUSAL_MIN_STRENGTH = 0.15

# Когнитивный ритм (v22)
PULSE_ENABLED          = True
PULSE_PHASE_MAX_TOKENS = 300
PULSE_FINAL_MAX_TOKENS = 1500
PULSE_SHORT_THRESHOLD  = 10
PULSE_LOG_MAX          = 20

# ══════════════════════════════════════════
# v23: AGI-МОДУЛИ — КОНФИГ
# ══════════════════════════════════════════
# Цели
GOAL_MAX_ACTIVE        = 5        # Макс активных целей
GOAL_PROGRESS_STEP     = 0.12     # Прирост прогресса за релевантное сообщение
GOAL_COMPLETE_THRESH   = 0.90     # Порог завершения цели
GOAL_DECAY_RATE        = 0.01     # Затухание неактивных целей

# Активное обучение
AL_HYPOTHESIS_MAX      = 30       # Макс гипотез в журнале
AL_CONFLICT_WINDOW     = 20       # Сколько последних фактов проверять на противоречия
AL_QUESTION_PROB       = 0.15     # Вероятность задать уточняющий вопрос

# Теория разума
TOM_MAX_BELIEFS        = 50       # Макс убеждений в модели
TOM_DECAY_SESSIONS     = 10       # Через сколько сессий убеждение ослабевает
TOM_UPDATE_EVERY       = 5        # Обновлять модель каждые N сообщений

# Планировщик
PLANNER_MAX_TASKS      = 10       # Макс задач в плане
PLANNER_DEPTH          = 3        # Глубина декомпозиции

# Инициативность
PROACTIVE_EVERY        = 15       # Инициативное сообщение каждые N диалогов
PROACTIVE_PROB         = 0.3      # Вероятность сработать (не всегда)

# Самомодификация
SELF_MOD_EVAL_EVERY    = 20       # Оценивать стратегии каждые N сообщений
SELF_MOD_MAX_RULES     = 20       # Макс правил поведения


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
        'в','и','на','с','по','для','от','к','о','у','из','за',
        'что','это','как','то','а','но','или','the','is','at',
        'не','ты','я','мне','себя','был','была','было','мой','твой',
        'они','мы','вы','он','она','его','её','их','там','тут',
        'вот','уже','ещё','даже','тоже','просто','очень','когда','хочу','могу'
    }

    @staticmethod
    def extract_keywords(text: str, top_n: int = 8) -> List[str]:
        words    = [w.lower() for w in re.findall(r'\b[а-яёa-z]{3,}\b', text, re.IGNORECASE)]
        filtered = [w for w in words if w not in TextUtils.STOP_WORDS]
        return [w for w, _ in Counter(filtered).most_common(top_n)]

    @staticmethod
    def keyword_overlap(kw1: List[str], kw2: List[str]) -> float:
        if not kw1 or not kw2: return 0.0
        s1, s2 = set(kw1), set(kw2)
        return len(s1 & s2) / max(len(s1 | s2), 1)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = [w.lower() for w in re.findall(r'\b[а-яёa-z]{3,}\b', text, re.IGNORECASE)]
        return [w for w in words if w not in TextUtils.STOP_WORDS]

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]


# ══════════════════════════════════════════
# ЗАТУХАНИЕ АКТИВАЦИИ
# ══════════════════════════════════════════
class ActivationDecay:
    @staticmethod
    def decay_factor(last_iso: str, rate: float = ACTIVATION_DECAY_RATE) -> float:
        try:
            hours = (datetime.now() - datetime.fromisoformat(last_iso)).total_seconds() / 3600.0
            return math.exp(-rate * hours)
        except:
            return 1.0

    @staticmethod
    def apply(activation: float, last_iso: str, rate: float = ACTIVATION_DECAY_RATE) -> float:
        return activation * ActivationDecay.decay_factor(last_iso, rate)

    @staticmethod
    def tonic(experience: float) -> float:
        return min(TONIC_MAX, experience * TONIC_SCALE)

    @staticmethod
    def is_archive_candidate(last_iso: str) -> bool:
        try:
            days = (datetime.now() - datetime.fromisoformat(last_iso)).days
            return days >= ARCHIVE_AFTER_DAYS
        except:
            return False


# ══════════════════════════════════════════
# СТОХАСТИЧЕСКИЙ ДРЕЙФ
# ══════════════════════════════════════════
class StochasticDrift:
    def __init__(self):
        self._step_counter = 0

    def maybe_explore(self, nodes: Dict[str, Any]) -> Optional[str]:
        if not nodes or random.random() > DRIFT_EXPLORE_PROB:
            return None
        candidates = [
            n for n in nodes.values()
            if 0.01 < n.activation < 0.15 and n.experience > 0.3
            and not getattr(n, 'archived', False)
        ]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        chosen.activation  = min(0.4, chosen.activation + 0.25 + random.uniform(0, 0.1))
        chosen.last_active = datetime.now().isoformat()
        return chosen.label

    def add_noise(self, activation: float) -> float:
        return max(0.0, min(1.0, activation + random.gauss(0, DRIFT_NOISE_SCALE)))

    def get_wild_association(self, nodes: Dict[str, Any],
                              synapses: Dict[str, Any]) -> Optional[str]:
        self._step_counter += 1
        if self._step_counter % DRIFT_ASSOC_EVERY != 0:
            return None
        mid_synapses = [s for s in synapses.values() if 0.2 < s.weight < 0.5 and s.fire_count > 1]
        if not mid_synapses:
            return None
        chosen   = random.choice(mid_synapses)
        src_node = nodes.get(chosen.source)
        tgt_node = nodes.get(chosen.target)
        if src_node and tgt_node:
            return f"{src_node.label} ↔ {tgt_node.label}"
        return None

    def describe_drift(self, explored: Optional[str], wild: Optional[str]) -> str:
        parts = []
        if explored: parts.append(f"[Случайное воспоминание: {explored}]")
        if wild:     parts.append(f"[Периферийная связь: {wild}]")
        return " ".join(parts)


# ══════════════════════════════════════════
# АДАПТИВНАЯ ТЕМПЕРАТУРА
# ══════════════════════════════════════════
class TemperatureAdapter:
    CLUSTER_TEMPS = {
        'технологии': TEMP_TECHNICAL, 'наука': TEMP_TECHNICAL,
        'работа':     TEMP_TECHNICAL,  'учёба': 0.45,
        'деньги':     0.40,  'кино':    0.70,
        'отдых':      0.75,  'общение': 0.75,
        'эмоции':     TEMP_EMOTIONAL,  'здоровье': 0.55,
        'спорт':      0.65,  'еда':     0.70,
    }
    CONFUSION_SIGNALS    = ['не понял','не понимаю','что ты имеешь','объясни',
                             'поясни','не то','неправильно','не так','ещё раз',
                             'что значит','можешь подробнее','я не']
    SATISFACTION_SIGNALS = ['спасибо','понял','отлично','классно','круто',
                             'понятно','всё ясно','именно','точно','супер','ок']

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.cluster_adjustments: Dict[str, float] = {}
        self._load()

    def _load(self):
        self.cluster_adjustments = FileManager.safe_load_json(self.save_path, {}).get('adjustments', {})

    def save(self):
        FileManager.safe_save_json(self.save_path, {'adjustments': self.cluster_adjustments})

    def feedback(self, text: str, cluster: Optional[str]):
        if not cluster: return
        tl        = text.lower()
        confused  = sum(1 for s in self.CONFUSION_SIGNALS if s in tl)
        satisfied = sum(1 for s in self.SATISFACTION_SIGNALS if s in tl)
        adj = self.cluster_adjustments.get(cluster, 0.0)
        if confused > 0:  adj = max(-0.25, adj - TEMP_FEEDBACK_STEP * confused)
        if satisfied > 0: adj = min(0.20,  adj + TEMP_FEEDBACK_STEP * 0.5)
        self.cluster_adjustments[cluster] = round(adj, 3)

    def get(self, cluster_name: Optional[str], emotion: str, text: str) -> float:
        base = self.CLUSTER_TEMPS.get(cluster_name, TEMP_DEFAULT)
        if cluster_name:
            base += self.cluster_adjustments.get(cluster_name, 0.0)
        if emotion == 'negative': base = min(TEMP_EMOTIONAL, base + 0.15)
        elif emotion == 'positive': base = min(TEMP_CREATIVE, base + 0.05)
        tl = text.lower()
        if any(w in tl for w in ['напиши код','функция','class','def ','import ']):
            base = min(base, TEMP_TECHNICAL)
        elif any(w in tl for w in ['придумай','представь','фантазия','идея','вариант']):
            base = max(base, TEMP_CREATIVE)
        elif text.strip().endswith('?'):
            base = min(base, base * 0.9)
        return round(max(0.1, min(1.0, base)), 2)


# ══════════════════════════════════════════
# УМНАЯ L1-ПАМЯТЬ
# ══════════════════════════════════════════
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

    def get_context(self, query: str) -> List[Dict]:
        if not self.history: return []
        recent     = self.history[-L1_RECENT_COUNT:]
        recent_set = set(id(m) for m in recent)
        query_kw   = set(TextUtils.extract_keywords(query))
        candidates = []
        for msg in self.history[:-L1_RECENT_COUNT]:
            msg_kw = set(msg.get('keywords', []))
            if not query_kw or not msg_kw: continue
            overlap = len(query_kw & msg_kw) / max(len(query_kw | msg_kw), 1)
            if overlap > 0.1: candidates.append((overlap, msg))
        candidates.sort(key=lambda x: -x[0])
        relevant   = [m for _, m in candidates[:L1_RELEVANT_COUNT]]
        result_set: Set[int] = set()
        result: List[Dict] = []
        for msg in self.history:
            mid = id(msg)
            if mid in recent_set or msg in relevant:
                if mid not in result_set:
                    result.append(msg); result_set.add(mid)
        return result

    def format_for_prompt(self, query: str) -> str:
        ctx = self.get_context(query)
        if not ctx: return ""
        lines = []
        for m in ctx:
            role_label = "Пользователь" if m['role'] == 'user' else "Ассистент"
            lines.append(f"{role_label}: {m['content'][:200]}")
        return "\n".join(lines)

    def save(self):
        FileManager.safe_save_json(self.save_path, self.history)


# ══════════════════════════════════════════
# RAG v2
# ══════════════════════════════════════════
@dataclass
class RAGResult:
    content: str; score: float; timestamp: str; rel_time: str
    cluster_name: Optional[str] = None; emotion: Optional[str] = None


class RAGEngineV2:
    @staticmethod
    def _compute_idf(query_terms: List[str], all_events: List[Any]) -> Dict[str, float]:
        N = max(len(all_events), 1); df: Dict[str, int] = {}
        for event in all_events:
            doc_terms = set(event.keywords)
            for term in query_terms:
                if term in doc_terms: df[term] = df.get(term, 0) + 1
        return {t: math.log((N - df.get(t,0) + 0.5) / (df.get(t,0) + 0.5) + 1)
                for t in query_terms}

    @staticmethod
    def _bm25_score(query_terms: List[str], doc_terms: List[str],
                    idf: Dict[str, float], avg_len: float) -> float:
        score = 0.0; dl = len(doc_terms); tf_map = Counter(doc_terms)
        for term in query_terms:
            if term not in tf_map: continue
            tf = tf_map[term]
            tf_norm = (tf * (RAG_BM25_K1 + 1)) / \
                      (tf + RAG_BM25_K1 * (1 - RAG_BM25_B + RAG_BM25_B * dl / max(avg_len, 1)))
            score += idf.get(term, 0.1) * tf_norm
        return score

    @staticmethod
    def search(query: str, events: List[Any], top_k: int = RAG_TOP_K,
               min_score: float = RAG_MIN_SCORE, now: datetime = None,
               current_emotion: str = 'neutral') -> List[RAGResult]:
        if not events or not query: return []
        if now is None: now = datetime.now()
        query_terms = TextUtils.tokenize(query)
        if not query_terms: return []
        idf     = RAGEngineV2._compute_idf(query_terms, events)
        avg_len = sum(len(e.keywords) for e in events) / len(events)
        scored  = []
        for event in events:
            doc_terms = TextUtils.tokenize(event.content)
            bm25      = RAGEngineV2._bm25_score(query_terms, doc_terms, idf, avg_len)
            if bm25 < 0.01: continue
            age_h   = max((now - datetime.fromisoformat(event.timestamp)).total_seconds() / 3600, 0.1)
            recency = 1.0 / (1.0 + math.log1p(age_h / 24))
            emot_b  = RAG_EMOTION_BOOST if (getattr(event,'user_emotion',None) == current_emotion
                                             and current_emotion != 'neutral') else 0.0
            score   = bm25*0.60 + recency*0.2 + event.importance*0.12 + emot_b*0.08
            if score >= min_score: scored.append((score, event))
        scored.sort(key=lambda x: -x[0])
        results: List[RAGResult] = []; cluster_counts: Dict[str,int] = {}
        for score, event in scored:
            cn = event.cluster_name or "_none"
            if cluster_counts.get(cn, 0) >= 2: continue
            cluster_counts[cn] = cluster_counts.get(cn, 0) + 1
            rel = TimeEncoder.describe(datetime.fromisoformat(event.timestamp), now)
            results.append(RAGResult(content=event.content, score=round(score,3),
                timestamp=event.timestamp, rel_time=rel,
                cluster_name=event.cluster_name,
                emotion=getattr(event,'user_emotion',None)))
            if len(results) >= top_k: break
        return results

    @staticmethod
    def format_for_prompt(results: List[RAGResult]) -> str:
        if not results: return ""
        lines = ["=== РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ (L2) ==="]
        for r in results:
            cluster = f"[{r.cluster_name}] " if r.cluster_name else ""
            etag    = f" [{r.emotion}]" if r.emotion and r.emotion != 'neutral' else ""
            lines.append(f"• [{r.rel_time}]{etag} {cluster}{r.content[:130]}")
        lines.append("=== КОНЕЦ ВОСПОМИНАНИЙ ===")
        return "\n".join(lines)


# ══════════════════════════════════════════
# ТРЁХУРОВНЕВАЯ ПАМЯТЬ L3
# ══════════════════════════════════════════
class SemanticMemoryL3:
    SEMANTIC_UPDATE_EVERY = 25

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.facts: List[str] = []
        self.topic_freq: Dict[str, int] = {}
        self.time_patterns: Dict[str, int] = {}
        self._counter = 0; self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self.facts         = data.get('facts', [])
        self.topic_freq    = data.get('topic_freq', {})
        self.time_patterns = data.get('time_patterns', {})
        self._counter      = data.get('counter', 0)

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'facts': self.facts, 'topic_freq': self.topic_freq,
            'time_patterns': self.time_patterns, 'counter': self._counter
        })

    def record(self, cluster_name: Optional[str], now: datetime):
        self._counter += 1
        if cluster_name:
            self.topic_freq[cluster_name] = self.topic_freq.get(cluster_name, 0) + 1
        hour   = now.hour
        period = ('утро' if 6<=hour<12 else 'день' if 12<=hour<18
                  else 'вечер' if 18<=hour<23 else 'ночь')
        self.time_patterns[period] = self.time_patterns.get(period, 0) + 1
        if self._counter % self.SEMANTIC_UPDATE_EVERY == 0: self._rebuild_facts()

    def force_rebuild(self): self._rebuild_facts()

    def _rebuild_facts(self):
        new_facts = []
        if self.topic_freq:
            top = sorted(self.topic_freq.items(), key=lambda x: -x[1])[:3]
            new_facts.append(f"Часто обсуждает: {', '.join(t for t,_ in top)}")
        if self.time_patterns:
            tp = max(self.time_patterns.items(), key=lambda x: x[1])
            new_facts.append(f"Чаще всего активен: {tp[0]}")
        if self.topic_freq:
            dom = max(self.topic_freq.items(), key=lambda x: x[1])
            if dom[1] >= 5: new_facts.append(f"Основной интерес: {dom[0]}")
        self.facts = new_facts

    def get_prompt_block(self) -> str:
        if not self.facts: return ""
        return "=== ДОЛГОСРОЧНЫЕ ПАТТЕРНЫ (L3) ===\n" + "\n".join(f"  • {f}" for f in self.facts)


# ══════════════════════════════════════════
# ВРЕМЕННОЙ КОДИРОВЩИК
# ══════════════════════════════════════════
class TimeEncoder:
    TIME_WORDS = {
        'сейчас': {'hours':0}, 'только что': {'minutes':-30},
        'недавно': {'hours':-2}, 'сегодня': {'days':0},
        'вчера': {'days':-1}, 'позавчера': {'days':-2},
        'завтра': {'days':1}, 'послезавтра': {'days':2},
        'на прошлой неделе': {'days':-7}, 'на следующей неделе': {'days':7},
        'утро':{'hour_range':(6,12)}, 'утром':{'hour_range':(6,12)},
        'день':{'hour_range':(12,18)}, 'днём':{'hour_range':(12,18)},
        'вечер':{'hour_range':(18,24)}, 'вечером':{'hour_range':(18,24)},
        'ночь':{'hour_range':(0,6)}, 'ночью':{'hour_range':(0,6)},
    }

    @staticmethod
    def parse_time_ref(text: str, now: datetime) -> Optional[Dict]:
        tl = text.lower()
        for phrase, offset in TimeEncoder.TIME_WORDS.items():
            if phrase in tl:
                if 'hour_range' in offset:
                    return {'type':'time_of_day','range':offset['hour_range'],'date':now.date()}
                delta = timedelta(**{k:v for k,v in offset.items()})
                t = now + delta
                return {'type':'date_range',
                        'start':t.replace(hour=0,minute=0,second=0),
                        'end':t.replace(hour=23,minute=59,second=59)}
        m = re.search(r'(\d+)\s*(час|часа|часов|день|дня|дней)\s*назад', tl)
        if m:
            val, unit = int(m.group(1)), m.group(2)
            d = timedelta(hours=val) if 'час' in unit else timedelta(days=val)
            return {'type':'date_range','start':now-d,'end':now}
        return None

    @staticmethod
    def describe(event_time: datetime, now: datetime) -> str:
        d = now - event_time; s = d.total_seconds()
        if s < 60:    return "только что"
        if s < 3600:  return f"{int(s/60)} мин. назад"
        if s < 86400: return f"{int(s/3600)} ч. назад"
        if d.days == 1: return "вчера"
        if d.days < 7:  return f"{d.days} дн. назад"
        return event_time.strftime("%d.%m.%Y")


# ══════════════════════════════════════════
# ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ
# ══════════════════════════════════════════
@dataclass
class UserProfile:
    name: str = ""; profession: str = ""
    interests: List[str] = field(default_factory=list)
    communication_style: str = "neutral"
    disliked_topics: List[str] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class ProfileManager:
    FACT_PATTERNS = [
        (r'меня зовут ([А-ЯЁа-яёA-Za-z]+)', 'name'),
        (r'я ([А-ЯЁа-яё]+(?:ист|ер|ор|ёр|ник|щик|лог|граф))', 'profession'),
        (r'работаю ([А-ЯЁа-яё\s]+?)(?:\.|,|$)', 'profession'),
        (r'я люблю ([А-ЯЁа-яё\s]+?)(?:\.|,|$)', 'interest'),
        (r'мне нравится ([А-ЯЁа-яё\s]+?)(?:\.|,|$)', 'interest'),
        (r'я увлекаюсь ([А-ЯЁа-яё\s]+?)(?:\.|,|$)', 'interest'),
        (r'не люблю ([А-ЯЁа-яё\s]+?)(?:\.|,|$)', 'dislike'),
        (r'живу в ([А-ЯЁа-яёA-Za-z\s]+?)(?:\.|,|$)', 'fact'),
        (r'мне (\d{1,2}) (?:лет|год)', 'fact'),
    ]
    STYLE_CLUES = {
        'technical': ['код','функция','алгоритм','python','class','debug','api'],
        'formal':    ['уважаемый','пожалуйста','благодарю','прошу','позвольте'],
        'casual':    ['норм','окей','ок','лол','кстати','слушай','короче'],
        'emotional': ['грустно','радостно','обидно','восторг','боюсь','тревожно'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path; self.profile = UserProfile(); self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        if data:
            for k, v in data.items():
                if hasattr(self.profile, k): setattr(self.profile, k, v)

    def save(self):
        FileManager.safe_save_json(self.save_path, asdict(self.profile))

    def update_from_text(self, text: str) -> List[str]:
        new_facts = []; tl = text.lower()
        for pattern, fact_type in self.FACT_PATTERNS:
            m = re.search(pattern, tl)
            if not m: continue
            value = m.group(1).strip()
            if not 2 <= len(value) <= 40: continue
            if fact_type == 'name' and not self.profile.name:
                self.profile.name = value.capitalize(); new_facts.append(f"имя: {self.profile.name}")
            elif fact_type == 'profession' and not self.profile.profession:
                self.profile.profession = value; new_facts.append(f"профессия: {value}")
            elif fact_type == 'interest' and value not in self.profile.interests:
                self.profile.interests.append(value)
                if len(self.profile.interests) > 30: self.profile.interests = self.profile.interests[-30:]
                new_facts.append(f"интерес: {value}")
            elif fact_type == 'dislike' and value not in self.profile.disliked_topics:
                self.profile.disliked_topics.append(value); new_facts.append(f"не нравится: {value}")
            elif fact_type == 'fact':
                if value not in self.profile.known_facts:
                    self.profile.known_facts.append(value)
                    if len(self.profile.known_facts) > 20: self.profile.known_facts = self.profile.known_facts[-20:]
                    new_facts.append(f"факт: {value}")
        for style, clues in self.STYLE_CLUES.items():
            if sum(1 for c in clues if c in tl) >= 2: self.profile.communication_style = style; break
        if new_facts: self.profile.last_updated = datetime.now().isoformat()
        return new_facts

    def get_prompt_block(self) -> str:
        p = self.profile; lines = []
        if p.name:            lines.append(f"Имя: {p.name}")
        if p.profession:      lines.append(f"Профессия: {p.profession}")
        if p.interests:       lines.append(f"Интересы: {', '.join(p.interests[:5])}")
        if p.disliked_topics: lines.append(f"Не интересует: {', '.join(p.disliked_topics[:3])}")
        if p.known_facts:     lines.append(f"Факты: {'; '.join(p.known_facts[:4])}")
        style_hint = {'technical':"Говори технически точно.",'formal':"Общайся формально.",
                      'casual':"Общайся непринуждённо.",'emotional':"Будь эмпатичным."
                      }.get(p.communication_style, "")
        if style_hint: lines.append(style_hint)
        return "\n".join(lines) if lines else ""

    def get_status(self) -> str:
        p = self.profile
        return (f"Имя: {p.name or '?'}\nПрофессия: {p.profession or '?'}\n"
                f"Интересы: {', '.join(p.interests[:5]) or 'нет'}\n"
                f"Стиль: {p.communication_style}\nФактов: {len(p.known_facts)}")


# ══════════════════════════════════════════
# ЭМОЦИОНАЛЬНАЯ ДУГА
# ══════════════════════════════════════════
@dataclass
class EmotionPoint:
    timestamp: str; emotion: str; score: float; context: str


class EmotionalArc:
    LEXICON = {
        'positive': ['рад','хорошо','отлично','супер','люблю','нравится','кайф',
                     'счастье','весело','здорово','классно','круто','успех'],
        'negative': ['грустно','плохо','устал','злой','проблема','страшно','боюсь',
                     'тревога','обидно','надоело','сложно','неудача'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path; self.history: List[EmotionPoint] = []
        self._session: List[float] = []; self._load()

    def _load(self):
        for item in FileManager.safe_load_json(self.save_path, []):
            self.history.append(EmotionPoint(**item))

    def save(self):
        FileManager.safe_save_json(self.save_path, [asdict(p) for p in self.history[-200:]])

    def detect(self, text: str) -> Tuple[str, float]:
        tl = text.lower()
        pos = sum(1 for w in self.LEXICON['positive'] if w in tl)
        neg = sum(1 for w in self.LEXICON['negative'] if w in tl)
        total = pos + neg
        if total == 0: return 'neutral', 0.0
        score   = (pos - neg) / total
        emotion = 'positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral')
        return emotion, score

    def record(self, text: str) -> Tuple[str, float]:
        emotion, score = self.detect(text)
        self.history.append(EmotionPoint(
            timestamp=datetime.now().isoformat(), emotion=emotion,
            score=score, context=text[:60]))
        self._session.append(score)
        if len(self.history) > 500: self.history = self.history[-500:]
        return emotion, score

    def get_session_mood(self) -> Tuple[str, str]:
        if not self._session: return 'neutral', ""
        avg = sum(self._session) / len(self._session)
        if avg > 0.3:  return 'positive', "Пользователь в хорошем настроении. Поддерживай позитивный тон."
        if avg < -0.3: return 'negative', "Пользователь расстроен. Будь мягким и поддерживающим."
        return 'neutral', ""

    def get_trend(self, last_n: int = 10) -> Dict:
        recent = self.history[-last_n:]
        if not recent: return {'trend':'unknown','avg':0.0,'direction':'stable','count':0}
        scores = [p.score for p in recent]; avg = sum(scores)/len(scores)
        mid = len(scores)//2
        fh = sum(scores[:mid])/max(mid,1); sh = sum(scores[mid:])/max(len(scores)-mid,1)
        direction = 'rising' if sh>fh+0.1 else ('falling' if sh<fh-0.1 else 'stable')
        trend = 'positive' if avg>0.15 else ('negative' if avg<-0.15 else 'neutral')
        return {'trend':trend,'avg':round(avg,2),'direction':direction,'count':len(recent)}

    def get_history_summary(self, days: int = 7) -> str:
        cutoff = datetime.now() - timedelta(days=days)
        recent = [p for p in self.history if datetime.fromisoformat(p.timestamp) >= cutoff]
        if not recent: return "Нет данных."
        pos = sum(1 for p in recent if p.emotion=='positive')
        neg = sum(1 for p in recent if p.emotion=='negative')
        t   = self.get_trend()
        return (f"За {days} дн.: 😊{pos} / 😐{len(recent)-pos-neg} / 😟{neg}\n"
                f"Score: {t['avg']:+.2f} | Тренд: {t['direction']}")


# ══════════════════════════════════════════
# ПРЕДСКАЗАНИЕ ТЕМ — МАРКОВ 2-го ПОРЯДКА
# ══════════════════════════════════════════
class TopicPredictor:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.transitions: Dict[str,Dict[str,int]]  = {}
        self.transitions2: Dict[str,Dict[str,int]] = {}
        self._prev: Optional[str] = None; self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self.transitions  = data.get('t1', {})
        self.transitions2 = data.get('t2', {})

    def save(self):
        FileManager.safe_save_json(self.save_path, {'t1':self.transitions,'t2':self.transitions2})

    def record_transition(self, frm: str, to: str):
        if not frm or not to or frm == to: return
        if frm not in self.transitions: self.transitions[frm] = {}
        self.transitions[frm][to] = self.transitions[frm].get(to, 0) + 1
        if self._prev and self._prev != frm:
            key2 = f"{self._prev}|{frm}"
            if key2 not in self.transitions2: self.transitions2[key2] = {}
            self.transitions2[key2][to] = self.transitions2[key2].get(to, 0) + 1
        self._prev = frm

    def predict_next(self, current: str, prev: str = None, top_n: int = 2) -> List[Tuple[str,float]]:
        if prev:
            key2 = f"{prev}|{current}"
            if key2 in self.transitions2:
                counts = self.transitions2[key2]; total = sum(counts.values())
                return sorted([(t,c/total) for t,c in counts.items()], key=lambda x:-x[1])[:top_n]
        if current not in self.transitions: return []
        counts = self.transitions[current]; total = sum(counts.values())
        return sorted([(t,c/total) for t,c in counts.items()], key=lambda x:-x[1])[:top_n]

    def get_hint(self, cluster: str, prev_cluster: str = None) -> str:
        preds = self.predict_next(cluster, prev_cluster)
        if not preds: return ""
        best, prob = preds[0]
        if prob > 0.4: return f"Вероятно, следующий вопрос о «{best}» ({prob:.0%})."
        return ""


# ══════════════════════════════════════════
# ЛОГИЧЕСКИЕ ЦЕПОЧКИ
# ══════════════════════════════════════════
@dataclass
class CausalLink:
    cause: str; effect: str; link_type: str = 'positive'
    strength: float = 0.5; evidence: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen:  str = field(default_factory=lambda: datetime.now().isoformat())

    def reinforce(self, d: float = 0.12):
        self.strength = min(1.0, self.strength+d); self.evidence += 1
        self.last_seen = datetime.now().isoformat()

    def decay(self, f: float = 0.995): self.strength = max(0.0, self.strength*f)


class LogicalChainEngine:
    PATTERNS = [
        (r'(\w{3,})\s+потому\s+что\s+(\w{3,})','effect','cause','positive'),
        (r'(\w{3,})\s+так\s+как\s+(\w{3,})','effect','cause','positive'),
        (r'из-за\s+(\w{3,})\s+(\w{3,})','cause','effect','positive'),
        (r'(\w{3,})\s+приводит\s+к\s+(\w{3,})','cause','effect','positive'),
        (r'(\w{3,})\s+вызывает\s+(\w{3,})','cause','effect','positive'),
        (r'(\w{3,})\s+помогает\s+(\w{3,})','cause','effect','positive'),
        (r'(\w{3,})\s+мешает\s+(\w{3,})','cause','effect','negative'),
        (r'если\s+(\w{3,}).{0,15}то\s+(\w{3,})','cause','effect','positive'),
        (r'(\w{3,})\s+даёт\s+(\w{3,})','cause','effect','positive'),
        (r'(\w{3,})\s+влияет\s+на\s+(\w{3,})','cause','effect','positive'),
        (r'без\s+(\w{3,})\s+нет\s+(\w{3,})','cause','effect','negative'),
        (r'(\w{3,})\s+означает\s+(\w{3,})','cause','effect','positive'),
    ]

    def __init__(self, save_path: str):
        self.save_path = save_path; self.links: Dict[str,CausalLink] = {}; self._load()

    def _key(self, c, e): return f"{c}→{e}"

    def _load(self):
        for k, v in FileManager.safe_load_json(self.save_path, {}).items():
            v.setdefault('link_type','positive'); self.links[k] = CausalLink(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path, {k:asdict(v) for k,v in self.links.items()})

    def extract_from_text(self, text: str) -> List[CausalLink]:
        found, tl = [], text.lower()
        for pat, r1, r2, lt in self.PATTERNS:
            for m in re.finditer(pat, tl):
                w1, w2 = m.group(1), m.group(2)
                if w1 in TextUtils.STOP_WORDS or w2 in TextUtils.STOP_WORDS: continue
                c = w1 if r1=='cause' else w2; e = w2 if r2=='effect' else w1
                k = self._key(c, e)
                if k in self.links: self.links[k].reinforce()
                else: self.links[k] = CausalLink(cause=c, effect=e, link_type=lt)
                found.append(self.links[k])
        return found

    def build_chain(self, start: str, max_depth: int = CAUSAL_MAX_DEPTH) -> List[str]:
        chain, cur, vis = [start], start, {start}
        for _ in range(max_depth):
            best, bs = None, 0.0
            for l in self.links.values():
                if l.cause==cur and l.effect not in vis:
                    s = l.strength*l.evidence
                    if s > bs: best, bs = l.effect, s
            if not best or bs < 0.1: break
            chain.append(best); vis.add(best); cur = best
        return chain

    def backward_chain(self, effect: str, depth: int = 3) -> Dict[str,Any]:
        def _r(node, visited, d):
            causes = []
            if d > 0:
                for l in sorted(self.links.values(), key=lambda x: -x.strength):
                    if l.effect==node and l.cause not in visited and l.strength>=CAUSAL_MIN_STRENGTH:
                        vnew = visited|{l.cause}
                        causes.append({'node':l.cause,'strength':round(l.strength,2),
                                       'type':l.link_type,'causes':_r(l.cause,vnew,d-1)['causes']})
            return {'node':node,'causes':causes[:3]}
        return _r(effect, {effect}, depth)

    def forward_chain(self, cause: str, depth: int = 3) -> Dict[str,Any]:
        def _r(node, visited, d):
            effects = []
            if d > 0:
                for l in sorted(self.links.values(), key=lambda x: -x.strength):
                    if l.cause==node and l.effect not in visited and l.strength>=CAUSAL_MIN_STRENGTH:
                        vnew = visited|{l.effect}
                        effects.append({'node':l.effect,'strength':round(l.strength,2),
                                        'type':l.link_type,'effects':_r(l.effect,vnew,d-1)['effects']})
            return {'node':node,'effects':effects[:3]}
        return _r(cause, {cause}, depth)

    def counterfactual(self, concept: str) -> List[str]:
        return [l.effect for l in self.links.values()
                if l.cause==concept and l.link_type=='positive' and l.strength>=CAUSAL_MIN_STRENGTH][:5]

    def format_causal_for_prompt(self, keyword: str, query: str) -> str:
        tl = query.lower(); lines = []
        if any(w in tl for w in ['почему','причина','из-за чего','отчего']):
            tree = self.backward_chain(keyword, depth=2)
            if tree['causes']:
                lines.append(f"[Причины «{keyword}»]")
                for c in tree['causes']:
                    sub = " ← ".join(cc['node'] for cc in c.get('causes',[]))
                    lines.append(f"  ← {c['node']} ({c['strength']:.2f})"+(f" ← {sub}" if sub else ""))
        elif any(w in tl for w in ['что будет','что если','последствия','приведёт']):
            tree = self.forward_chain(keyword, depth=2)
            if tree['effects']:
                lines.append(f"[Следствия «{keyword}»]")
                for e in tree['effects']:
                    sub = " → ".join(ee['node'] for ee in e.get('effects',[]))
                    lines.append(f"  → {e['node']} ({e['strength']:.2f})"+(f" → {sub}" if sub else ""))
        elif any(w in tl for w in ['без','если не','что если нет']):
            deps = self.counterfactual(keyword)
            if deps:
                lines.append(f"[Без «{keyword}» пострадает]"); lines.extend(f"  • {d}" for d in deps)
        else:
            chain = self.build_chain(keyword)
            if len(chain) > 1: lines.append(f"[Цепочка] {' → '.join(chain)}")
        return "\n".join(lines)

    def get_strong_chains(self, top_n: int = 3) -> List[str]:
        seen, chains = [], []
        for l in sorted(self.links.values(), key=lambda x: x.strength*x.evidence, reverse=True):
            if l.cause in seen: continue
            ch = self.build_chain(l.cause)
            if len(ch) > 1: chains.append(" → ".join(ch)); seen.append(l.cause)
            if len(chains) >= top_n: break
        return chains

    def decay_all(self):
        for l in self.links.values(): l.decay()


# ══════════════════════════════════════════
# СЕМАНТИЧЕСКИЕ КЛАСТЕРЫ
# ══════════════════════════════════════════
@dataclass
class SemanticCluster:
    id: str; name: str; members: List[str] = field(default_factory=list)
    centroid_keywords: List[str] = field(default_factory=list)
    strength: float = 0.5; access_count: int = 0


class SemanticClusterEngine:
    SEEDS = {
        'кино':       ['фильм','кино','актёр','режиссёр','сцена','сериал','смотреть'],
        'технологии': ['код','программа','компьютер','python','алгоритм','сервер','данные'],
        'еда':        ['еда','готовить','ресторан','вкусно','рецепт','блюдо','кофе'],
        'здоровье':   ['болеть','врач','здоровье','лекарство','самочувствие','больница'],
        'работа':     ['работа','офис','задача','проект','коллега','дедлайн','задание'],
        'отдых':      ['отдых','отпуск','путешествие','прогулка','развлечение','игра'],
        'эмоции':     ['рад','грусть','злость','страх','счастье','настроение','чувство'],
        'общение':    ['друг','разговор','встреча','звонок','письмо','общение','семья'],
        'учёба':      ['учить','знание','школа','курс','книга','читать','изучать'],
        'спорт':      ['спорт','тренировка','бегать','gym','фитнес','игра','команда'],
        'деньги':     ['деньги','зарплата','бюджет','расход','доход','инвестиции'],
        'наука':      ['наука','исследование','эксперимент','теория','физика','химия'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path; self.clusters: Dict[str,SemanticCluster] = {}
        self._load()
        if not self.clusters: self._init()

    def _load(self):
        for k, v in FileManager.safe_load_json(self.save_path, {}).items():
            self.clusters[k] = SemanticCluster(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path, {k:asdict(v) for k,v in self.clusters.items()})

    def _init(self):
        for name, words in self.SEEDS.items():
            cid = hashlib.md5(name.encode()).hexdigest()[:8]
            self.clusters[cid] = SemanticCluster(id=cid,name=name,members=words,
                                                  centroid_keywords=words[:3],strength=0.7)

    def classify(self, keywords: List[str]) -> List[Tuple[str,float]]:
        scores = []
        for cid, c in self.clusters.items():
            hits = sum(1 for kw in keywords if kw in c.members)
            if hits > 0: scores.append((c.name, hits/max(len(keywords),1))); c.access_count += 1
        return sorted(scores, key=lambda x: -x[1])

    def learn(self, keywords: List[str], hint: str = "") -> Optional[str]:
        classified = self.classify(keywords)
        if classified and classified[0][1] >= 0.25:
            best = classified[0][0]
            for cid, c in self.clusters.items():
                if c.name == best:
                    for kw in keywords:
                        if kw not in c.members: c.members.append(kw)
                        if len(c.members) > 120: c.members = c.members[-120:]
                    c.strength = min(1.0, c.strength+0.015); break
            return best
        if len(keywords) >= 2:
            name = hint or keywords[0]; cid = hashlib.md5(name.encode()).hexdigest()[:8]
            if cid not in self.clusters:
                self.clusters[cid] = SemanticCluster(id=cid,name=name,members=list(keywords),
                                                      centroid_keywords=keywords[:2],strength=0.3)
            else:
                for kw in keywords:
                    if kw not in self.clusters[cid].members: self.clusters[cid].members.append(kw)
        return None


# ══════════════════════════════════════════
# МЕТАОБУЧЕНИЕ
# ══════════════════════════════════════════
@dataclass
class NeuralGrowthRecord:
    timestamp: str; event_type: str; description: str; impact_score: float = 0.0


class MetaLearner:
    THRESHOLDS = [0,50,150,350,750,1500,3000,6000]
    NAMES = {1:"Новичок",2:"Ученик",3:"Знающий",4:"Опытный",
             5:"Мастер",6:"Эксперт",7:"Гений",8:"Оракул"}

    def __init__(self, save_path: str):
        self.save_path = save_path; self.xp = 0; self.level = 1
        self.total_interactions = 0; self.growth_log = []; self.insights = []; self._load()

    def _load(self):
        d = FileManager.safe_load_json(self.save_path, {})
        self.xp = d.get('xp',0); self.level = d.get('level',1)
        self.total_interactions = d.get('total_interactions',0); self.insights = d.get('insights',[])
        for r in d.get('growth_log',[]): self.growth_log.append(NeuralGrowthRecord(**r))

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'xp':self.xp,'level':self.level,'total_interactions':self.total_interactions,
            'insights':self.insights,'growth_log':[asdict(r) for r in self.growth_log[-100:]]
        })

    def gain_xp(self, amount: int, event: str, desc: str = "") -> Optional[str]:
        self.xp += amount; self.total_interactions += 1
        self.growth_log.append(NeuralGrowthRecord(
            timestamp=datetime.now().isoformat(),event_type=event,description=desc,impact_score=amount/10.0))
        return self._check_level_up()

    def _check_level_up(self) -> Optional[str]:
        for lvl, thr in enumerate(self.THRESHOLDS[1:], start=2):
            if self.xp >= thr and self.level < lvl:
                self.level = lvl; return f"🆙 УРОВЕНЬ {lvl} — {self.NAMES.get(lvl,'?')}!\n📊 Опыт: {self.xp}"
        return None

    def add_insight(self, s: str):
        if s and s not in self.insights:
            self.insights.append(s)
            if len(self.insights) > 60: self.insights = self.insights[-60:]

    def get_insights_prompt(self, n: int = 2) -> str:
        if not self.insights: return ""
        return "📌 Из прошлых сессий:\n"+"".join(f"  • {i}\n" for i in self.insights[-n:])

    def get_status(self) -> str:
        name = self.NAMES.get(self.level,'?')
        nxt  = next((t for t in self.THRESHOLDS if t > self.xp), None)
        sfx  = f" (до след.: {nxt-self.xp} XP)" if nxt else " (MAX)"
        return (f"Уровень {self.level} — {name}{sfx}\nОпыт: {self.xp} | Диалогов: {self.total_interactions}\nИнсайтов: {len(self.insights)}")


# ══════════════════════════════════════════
# ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ L2
# ══════════════════════════════════════════
@dataclass
class EpisodicMemory:
    id: str; timestamp: str; content: str; keywords: List[str]
    temporal_tags: List[str]; user_emotion: Optional[str]
    location_hints: List[str]; sequence_id: int
    cluster_name: Optional[str] = None; importance: float = 0.5


class TemporalMemory:
    def __init__(self, user_id: str, save_dir: str):
        self.file = os.path.join(save_dir,"episodic_time.json")
        self.events: List[EpisodicMemory] = []; self.sequence_counter = 0; self.now = datetime.now()
        data = FileManager.safe_load_json(self.file)
        if data:
            self.sequence_counter = data.get('seq',0)
            for e in data.get('events',[]):
                e.setdefault('cluster_name',None); e.setdefault('importance',0.5)
                self.events.append(EpisodicMemory(**e))
        print(f"⏳ Эпизодов: {len(self.events)}")

    def add_event(self, content: str, keywords: List[str], cluster_name: str=None, now: datetime=None) -> EpisodicMemory:
        if now is None: now = datetime.now(); self.now = now
        ttags = [tw for tw in ['сегодня','вчера','завтра','утром','вечером','ночью','сейчас'] if tw in content.lower()]
        locs  = [l for l in re.findall(r'(?:в|на|из|к)\s+([а-яёa-z]{3,})',content.lower()) if l not in {'дом','домой','город','место'}]
        imp   = min(1.0, 0.3+len(content)/500+(0.2 if ttags else 0))
        e = EpisodicMemory(
            id=hashlib.md5(f"{content}{now.isoformat()}".encode()).hexdigest()[:12],
            timestamp=now.isoformat(),content=content,keywords=keywords,temporal_tags=ttags,
            user_emotion=self._detect_emotion(content),location_hints=locs,
            sequence_id=self.sequence_counter,cluster_name=cluster_name,importance=imp)
        self.events.append(e); self.sequence_counter += 1
        if len(self.events) > 400:
            self.events.sort(key=lambda e: e.importance*0.5+datetime.fromisoformat(e.timestamp).timestamp()*1e-10)
            self.events = self.events[-400:]; self.events.sort(key=lambda e: e.sequence_id)
        return e

    def _detect_emotion(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ['рад','хорошо','отлично','супер','люблю','нравится']): return 'positive'
        if any(w in t for w in ['грустно','плохо','устал','злой','проблема','боюсь']): return 'negative'
        return None

    def query_by_time(self, time_ref: str, now: datetime=None) -> List[EpisodicMemory]:
        if now is None: now = datetime.now()
        tr = TimeEncoder.parse_time_ref(time_ref, now)
        if not tr: return []
        res = []
        for ev in reversed(self.events):
            et = datetime.fromisoformat(ev.timestamp)
            if tr['type']=='date_range':
                if tr['start']<=et<=tr['end']: res.append(ev)
            elif tr['type']=='time_of_day':
                if et.date()==tr['date'] and tr['range'][0]<=et.hour<tr['range'][1]: res.append(ev)
        return res[:10]

    def get_timeline_summary(self, hours: int=24) -> str:
        now = datetime.now(); cutoff = now-timedelta(hours=hours)
        recent = [e for e in self.events if datetime.fromisoformat(e.timestamp)>=cutoff]
        if not recent: return "За это время ничего не зафиксировано."
        return "\n".join(
            f"[{TimeEncoder.describe(datetime.fromisoformat(e.timestamp),now)}] "
            f"{'['+e.cluster_name+'] ' if e.cluster_name else ''}{e.content[:90]}" for e in recent[-6:])

    def save(self):
        FileManager.safe_save_json(self.file,{'seq':self.sequence_counter,'events':[asdict(e) for e in self.events]})


# ══════════════════════════════════════════
# CHRONO LAYER
# ══════════════════════════════════════════
@dataclass
class ChronoNode:
    id: str; label: str; node_type: str; activation: float = 0.0
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    use_count: int = 0; linked_events: List[str] = field(default_factory=list)


class ChronoNeuralLayer:
    def __init__(self):
        self.nodes: Dict[str,ChronoNode] = {}; self._init()

    def _init(self):
        for label, ntype in [
            ('вчера','temporal_ref'),('сегодня','temporal_ref'),('завтра','temporal_ref'),
            ('утром','time_of_day'),('днём','time_of_day'),('вечером','time_of_day'),
            ('ночью','time_of_day'),('утро','time_of_day'),('вечер','time_of_day'),
            ('раньше','sequence'),('потом','sequence'),('до','sequence'),
            ('после','sequence'),('всегда','sequence'),('никогда','sequence'),
        ]:
            nid = hashlib.md5(f"chrono_{label}".encode()).hexdigest()[:8]
            if nid not in self.nodes: self.nodes[nid] = ChronoNode(id=nid,label=label,node_type=ntype)

    def restore_from_saved(self, saved: List[Dict]):
        for nd in saved:
            nid = nd['id']
            if nid in self.nodes:
                raw=nd.get('activation',0.0); la=nd.get('last_active',datetime.now().isoformat()); uc=nd.get('use_count',0)
                self.nodes[nid].activation = max(ActivationDecay.apply(raw,la,rate=0.08),min(0.15,uc*0.005))
                self.nodes[nid].last_active=la; self.nodes[nid].use_count=uc
                self.nodes[nid].linked_events=nd.get('linked_events',[])

    def to_save_list(self) -> List[Dict]: return [asdict(n) for n in self.nodes.values()]

    def process_temporal_input(self, text: str) -> List[ChronoNode]:
        activated, tl = [], text.lower()
        for n in self.nodes.values():
            if n.label in tl:
                n.activation=min(1.0,n.activation+0.8); n.last_active=datetime.now().isoformat()
                n.use_count+=1; activated.append(n)
        for n in self.nodes.values():
            if n not in activated: n.activation *= 0.88
        return activated

    def get_active_labels(self) -> List[str]:
        return [n.label for n in self.nodes.values() if n.activation>0.3]

    def link_event(self, event: EpisodicMemory, nodes: List[ChronoNode]):
        for n in nodes:
            if event.id not in n.linked_events:
                n.linked_events.append(event.id)
                if len(n.linked_events)>60: n.linked_events=n.linked_events[-60:]


# ══════════════════════════════════════════
# НЕЙРО-КОРА
# ══════════════════════════════════════════
@dataclass
class NeuroNode:
    id: str; label: str; category: str; activation: float = 0.0
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0; experience: float = 0.0; temporal_weight: float = 1.0
    archived: bool = False


@dataclass
class NeuroSynapse:
    source: str; target: str; weight: float = 0.5; plasticity: float = 0.8
    temporal_decay: float = 0.0; last_fired: str = ""; fire_count: int = 0


class DynamicNeuralCortex:
    def __init__(self, user_id: str, save_dir: str):
        self.user_id   = user_id
        self.save_path = os.path.join(save_dir,"cortex_graph.json")
        self.nodes:    Dict[str,NeuroNode]    = {}
        self.synapses: Dict[str,NeuroSynapse] = {}
        self.chrono_layer = ChronoNeuralLayer()
        self.drift        = StochasticDrift()
        data = FileManager.safe_load_json(self.save_path)
        if data and 'nodes' in data: self._load_with_decay(data)
        else: self._create_node("Приветствие","action"); self._create_node("Помощь","action"); self.save()

    def _load_with_decay(self, data: Dict):
        now_iso = datetime.now().isoformat()
        for n in data.get('nodes',[]):
            n.setdefault('experience',0.0); n.setdefault('temporal_weight',1.0)
            n.setdefault('last_active',now_iso); n.setdefault('archived',False)
            decayed = ActivationDecay.apply(n.get('activation',0.0),n['last_active'])
            tonic   = ActivationDecay.tonic(n['experience'])
            if n.get('archived',False): n['activation'] = min(ARCHIVE_ACTIVATION_CAP,max(decayed,tonic))
            else:
                n['activation'] = max(decayed,tonic)
                if ActivationDecay.is_archive_candidate(n['last_active']) and n['experience']<0.5: n['archived']=True
            self.nodes[n['id']] = NeuroNode(**n)
        for s in data.get('synapses',[]):
            s.setdefault('fire_count',0); k=f"{s['source']}->{s['target']}"
            self.synapses[k] = NeuroSynapse(**s)
        if 'chrono_nodes' in data: self.chrono_layer.restore_from_saved(data['chrono_nodes'])
        warm=sum(1 for n in self.nodes.values() if n.activation>0.1 and not n.archived)
        arc =sum(1 for n in self.nodes.values() if n.archived)
        print(f"🧠 Кора: {len(self.nodes)} нейронов ({warm} тёплых, {arc} архивных), {len(self.synapses)} синапсов")

    def _create_node(self, label: str, category: str="concept") -> NeuroNode:
        nid = hashlib.md5(label.encode()).hexdigest()[:8]
        if nid not in self.nodes: self.nodes[nid] = NeuroNode(id=nid,label=label,category=category)
        return self.nodes[nid]

    def _reactivate_if_archived(self, node: NeuroNode):
        if node.archived: node.archived=False; node.activation=min(0.5,node.activation+0.3)

    def process_input(self, text: str, now: datetime=None) -> Tuple[List[str],List[ChronoNode],Optional[str],Optional[str]]:
        if now is None: now = datetime.now()
        keywords=TextUtils.extract_keywords(text,top_n=10); activated_labels=[]; now_iso=now.isoformat()
        for i, kw1 in enumerate(keywords):
            n1=self._create_node(kw1); self._reactivate_if_archived(n1)
            n1.access_count+=1; n1.activation=1.0; n1.last_active=now_iso; n1.experience+=0.12
            activated_labels.append(kw1)
            for kw2 in keywords[i+1:]: self._strengthen(n1.id,self._create_node(kw2).id,0.1)
        chrono_nodes=self.chrono_layer.process_temporal_input(text); final_context=set(activated_labels)
        for label in activated_labels:
            nid=hashlib.md5(label.encode()).hexdigest()[:8]
            for key,syn in list(self.synapses.items()):
                if syn.source==nid and syn.weight>0.4:
                    target=self.nodes.get(syn.target)
                    if target and not target.archived:
                        target.activation=min(1.0,target.activation+self.drift.add_noise(0.45))
                        target.last_active=now_iso; final_context.add(target.label); syn.fire_count+=1
        for node in self.nodes.values():
            tonic=ActivationDecay.tonic(node.experience)
            if node.archived: node.activation=min(ARCHIVE_ACTIVATION_CAP,node.activation*0.88)
            else:
                node.activation=max(node.activation*0.88,tonic)
                if ActivationDecay.is_archive_candidate(node.last_active) and node.experience<0.5 and node.access_count<5:
                    node.archived=True
        explored=self.drift.maybe_explore(self.nodes); wild=self.drift.get_wild_association(self.nodes,self.synapses)
        return list(final_context), chrono_nodes, explored, wild

    def _strengthen(self, src: str, tgt: str, reward: float):
        if src==tgt: return
        for k,(s,t) in [(f"{src}->{tgt}",(src,tgt)),(f"{tgt}->{src}",(tgt,src))]:
            if k not in self.synapses: self.synapses[k]=NeuroSynapse(source=s,target=t,weight=0.1)
            syn=self.synapses[k]; syn.weight=min(1.0,syn.weight+reward*syn.plasticity)
            syn.last_fired=datetime.now().isoformat(); syn.plasticity*=0.995; syn.fire_count+=1

    def associate(self, concept: str, top_n: int=6) -> List[Tuple[str,float]]:
        nid=hashlib.md5(concept.encode()).hexdigest()[:8]
        return sorted([(self.nodes[s.target].label,s.weight) for k,s in self.synapses.items()
                       if s.source==nid and s.weight>0.15 and s.target in self.nodes
                       and not self.nodes[s.target].archived], key=lambda x:-x[1])[:top_n]

    def get_hot_nodes(self, top_n: int=8) -> List[NeuroNode]:
        return sorted([n for n in self.nodes.values() if n.activation>0.05 and not n.archived],
                      key=lambda n:n.activation,reverse=True)[:top_n]

    def get_experienced_nodes(self, top_n: int=5) -> List[NeuroNode]:
        return sorted([n for n in self.nodes.values() if not n.archived],key=lambda n:n.experience,reverse=True)[:top_n]

    def get_archived_nodes(self, top_n: int=10) -> List[NeuroNode]:
        return sorted([n for n in self.nodes.values() if n.archived],key=lambda n:n.experience,reverse=True)[:top_n]

    def reinforce_path(self, keywords: List[str], chrono_nodes: List[ChronoNode], success: bool):
        reward=0.15 if success else -0.04
        for i,kw1 in enumerate(keywords):
            id1=hashlib.md5(kw1.encode()).hexdigest()[:8]
            for kw2 in keywords[i+1:]: self._strengthen(id1,hashlib.md5(kw2.encode()).hexdigest()[:8],reward)
        for cn in chrono_nodes: cn.activation=min(1.0,cn.activation+reward)

    def prune_weak_synapses(self, threshold: float=0.04) -> int:
        before=len(self.synapses)
        self.synapses={k:v for k,v in self.synapses.items() if v.weight>threshold}
        return before-len(self.synapses)

    def save(self):
        FileManager.safe_save_json(self.save_path,{
            'nodes':[asdict(n) for n in self.nodes.values()],
            'synapses':[asdict(s) for s in self.synapses.values()],
            'chrono_nodes':self.chrono_layer.to_save_list(),
            'saved_at':datetime.now().isoformat()
        })

    def get_stats(self) -> Dict:
        archived=sum(1 for n in self.nodes.values() if n.archived)
        return {'neurons':len(self.nodes),
                'warm':sum(1 for n in self.nodes.values() if n.activation>0.1 and not n.archived),
                'archived':archived,'synapses':len(self.synapses),
                'strong_syn':sum(1 for s in self.synapses.values() if s.weight>0.6),
                'chrono':len(self.chrono_layer.nodes),
                'active_chrono':sum(1 for n in self.chrono_layer.nodes.values() if n.activation>0.3)}


# ══════════════════════════════════════════
# v23 AGI MODULE 1: СИСТЕМА ЦЕЛЕЙ
# ══════════════════════════════════════════
@dataclass
class Goal:
    id:          str
    title:       str
    description: str
    goal_type:   str        # 'long', 'medium', 'short'
    progress:    float = 0.0
    subgoals:    List[str] = field(default_factory=list)
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    completed:   bool = False
    cluster_hint: Optional[str] = None


class GoalSystem:
    """
    🎯 Система целей v23.

    Определяет цели пользователя из диалога, отслеживает прогресс,
    декомпозирует сложные цели на подцели.
    Цели влияют на фокус ответов — AI помнит, к чему стремится пользователь.
    """
    GOAL_PATTERNS = [
        (r'хочу\s+([\w\s]{4,40})(?:\.|,|$)', 'short'),
        (r'планирую\s+([\w\s]{4,40})(?:\.|,|$)', 'medium'),
        (r'моя\s+цель\s+[-—]?\s*([\w\s]{4,50})(?:\.|,|$)', 'long'),
        (r'мечтаю\s+([\w\s]{4,40})(?:\.|,|$)', 'long'),
        (r'собираюсь\s+([\w\s]{4,40})(?:\.|,|$)', 'medium'),
        (r'нужно\s+([\w\s]{4,40})(?:\.|,|$)', 'short'),
        (r'стараюсь\s+([\w\s]{4,40})(?:\.|,|$)', 'medium'),
        (r'учусь\s+([\w\s]{4,40})(?:\.|,|$)', 'medium'),
    ]

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.goals: Dict[str, Goal] = {}
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        for k, v in data.items():
            v.setdefault('subgoals', [])
            v.setdefault('completed', False)
            v.setdefault('cluster_hint', None)
            self.goals[k] = Goal(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path, {k: asdict(v) for k, v in self.goals.items()})

    def extract_from_text(self, text: str, cluster: Optional[str] = None) -> List[Goal]:
        found = []
        tl    = text.lower()
        for pattern, gtype in self.GOAL_PATTERNS:
            for m in re.finditer(pattern, tl):
                title = m.group(1).strip()
                if len(title) < 4 or len(title) > 50:
                    continue
                gid = hashlib.md5(title.encode()).hexdigest()[:8]
                if gid not in self.goals:
                    g = Goal(id=gid, title=title, description=text[:100],
                             goal_type=gtype, cluster_hint=cluster)
                    self.goals[gid] = g
                    found.append(g)
                    print(f"🎯 Новая цель [{gtype}]: {title}")
        return found

    def update_progress(self, text: str, keywords: List[str]):
        """Обновляет прогресс целей на основе релевантности нового сообщения"""
        kw_set = set(keywords)
        for goal in self.goals.values():
            if goal.completed:
                continue
            goal_kw = set(TextUtils.extract_keywords(goal.title + " " + goal.description))
            overlap = len(kw_set & goal_kw) / max(len(goal_kw), 1)
            if overlap > 0.2:
                goal.progress = min(1.0, goal.progress + GOAL_PROGRESS_STEP * overlap)
                goal.updated_at = datetime.now().isoformat()
                if goal.progress >= GOAL_COMPLETE_THRESH:
                    goal.completed = True
                    print(f"✅ Цель выполнена: {goal.title}")
            else:
                goal.progress = max(0.0, goal.progress - GOAL_DECAY_RATE)

    def add_subgoal(self, goal_id: str, subgoal: str):
        if goal_id in self.goals:
            self.goals[goal_id].subgoals.append(subgoal)

    def get_active_goals(self) -> List[Goal]:
        return [g for g in self.goals.values() if not g.completed][:GOAL_MAX_ACTIVE]

    def get_prompt_block(self) -> str:
        active = self.get_active_goals()
        if not active:
            return ""
        lines = ["=== ЦЕЛИ ПОЛЬЗОВАТЕЛЯ ==="]
        for g in active[:3]:
            bar   = "█" * int(g.progress * 10) + "░" * (10 - int(g.progress * 10))
            gtype = {'long':'долгосрочная','medium':'среднесрочная','short':'краткосрочная'}.get(g.goal_type,'')
            lines.append(f"  [{gtype}] {g.title} {bar} {g.progress:.0%}")
            if g.subgoals:
                lines.append(f"    Подцели: {', '.join(g.subgoals[:3])}")
        lines.append("=========================")
        return "\n".join(lines)

    def get_status(self) -> str:
        active    = self.get_active_goals()
        completed = [g for g in self.goals.values() if g.completed]
        lines     = [f"🎯 СИСТЕМА ЦЕЛЕЙ\n{'═'*24}"]
        if not active and not completed:
            lines.append("Целей пока нет. Расскажите, чего хотите достичь!")
            return "\n".join(lines)
        if active:
            lines.append("Активные:")
            for g in active:
                bar = "█" * int(g.progress*10) + "░"*(10-int(g.progress*10))
                lines.append(f"  [{g.goal_type[0].upper()}] {g.title}\n    {bar} {g.progress:.0%}")
        if completed:
            lines.append(f"\nВыполнено: {len(completed)} целей ✅")
        return "\n".join(lines)


# ══════════════════════════════════════════
# v23 AGI MODULE 2: АКТИВНОЕ ОБУЧЕНИЕ
# ══════════════════════════════════════════
@dataclass
class Hypothesis:
    id:          str
    statement:   str
    confidence:  float = 0.5     # 0..1
    status:      str   = 'open'  # open / confirmed / refuted
    evidence_for:  List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at:  str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KnowledgeGap:
    topic:      str
    question:   str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved:   bool = False


class ActiveLearning:
    """
    🧪 Активное обучение v23.

    AI замечает пробелы в своих знаниях о пользователе,
    формулирует гипотезы и уточняет их через диалог.
    Ищет противоречия между старыми и новыми фактами.
    """

    UNCERTAINTY_PHRASES = [
        'наверное', 'может быть', 'не знаю', 'не уверен',
        'возможно', 'кажется', 'не помню', 'забыл', 'точно не знаю'
    ]

    def __init__(self, save_path: str):
        self.save_path  = save_path
        self.hypotheses: Dict[str, Hypothesis]  = {}
        self.gaps:       List[KnowledgeGap]     = []
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        for k, v in data.get('hypotheses', {}).items():
            v.setdefault('evidence_for', [])
            v.setdefault('evidence_against', [])
            self.hypotheses[k] = Hypothesis(**v)
        for g in data.get('gaps', []):
            self.gaps.append(KnowledgeGap(**g))

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'hypotheses': {k: asdict(v) for k, v in self.hypotheses.items()},
            'gaps': [asdict(g) for g in self.gaps[-20:]]
        })

    def detect_uncertainty(self, text: str) -> bool:
        tl = text.lower()
        return any(p in tl for p in self.UNCERTAINTY_PHRASES)

    def add_hypothesis(self, statement: str, confidence: float = 0.5) -> Hypothesis:
        hid = hashlib.md5(statement.encode()).hexdigest()[:8]
        if hid not in self.hypotheses:
            h = Hypothesis(id=hid, statement=statement, confidence=confidence)
            self.hypotheses[hid] = h
            if len(self.hypotheses) > AL_HYPOTHESIS_MAX:
                oldest = sorted(self.hypotheses.values(), key=lambda x: x.created_at)
                del self.hypotheses[oldest[0].id]
        return self.hypotheses[hid]

    def update_hypothesis(self, text: str):
        """Подтверждает или опровергает гипотезы по новому сообщению"""
        kw = set(TextUtils.extract_keywords(text))
        for h in self.hypotheses.values():
            if h.status != 'open':
                continue
            h_kw = set(TextUtils.extract_keywords(h.statement))
            if not h_kw:
                continue
            overlap = len(kw & h_kw) / max(len(h_kw), 1)
            if overlap > 0.4:
                tl = text.lower()
                if any(w in tl for w in ['нет','не','никогда','неправда','неверно','ошибка']):
                    h.evidence_against.append(text[:80])
                    h.confidence = max(0.0, h.confidence - 0.2)
                    if h.confidence < 0.2:
                        h.status = 'refuted'
                else:
                    h.evidence_for.append(text[:80])
                    h.confidence = min(1.0, h.confidence + 0.15)
                    if h.confidence > 0.8:
                        h.status = 'confirmed'
                h.updated_at = datetime.now().isoformat()

    def detect_contradiction(self, new_text: str, history: List[Dict]) -> Optional[str]:
        """Ищет противоречие между новым сообщением и историей"""
        new_kw = set(TextUtils.extract_keywords(new_text))
        recent = history[-AL_CONFLICT_WINDOW:]
        for old_msg in recent:
            if old_msg.get('role') != 'user':
                continue
            old_kw = set(old_msg.get('keywords', []))
            overlap = len(new_kw & old_kw) / max(len(new_kw | old_kw), 1)
            if overlap > 0.5:
                # Ищем явные противоречия по маркерам
                new_tl = new_text.lower()
                old_tl = old_msg['content'].lower()
                markers_new = [w for w in ['всегда','никогда','люблю','не люблю','умею','не умею'] if w in new_tl]
                markers_old = [w for w in ['всегда','никогда','люблю','не люблю','умею','не умею'] if w in old_tl]
                if markers_new and markers_old and markers_new != markers_old:
                    return f"Ранее: «{old_msg['content'][:60]}», сейчас: «{new_text[:60]}»"
        return None

    def add_knowledge_gap(self, topic: str, question: str):
        self.gaps.append(KnowledgeGap(topic=topic, question=question))
        if len(self.gaps) > 30:
            self.gaps = self.gaps[-30:]

    def should_ask_clarification(self) -> Optional[str]:
        """Возвращает вопрос для уточнения, если накопилось достаточно пробелов"""
        unresolved = [g for g in self.gaps if not g.resolved]
        if not unresolved or random.random() > AL_QUESTION_PROB:
            return None
        gap = random.choice(unresolved[-5:])
        gap.resolved = True
        return gap.question

    def get_open_hypotheses(self, top_n: int = 3) -> List[Hypothesis]:
        return sorted([h for h in self.hypotheses.values() if h.status == 'open'],
                      key=lambda x: -x.confidence)[:top_n]

    def get_status(self) -> str:
        total     = len(self.hypotheses)
        confirmed = sum(1 for h in self.hypotheses.values() if h.status == 'confirmed')
        refuted   = sum(1 for h in self.hypotheses.values() if h.status == 'refuted')
        opens     = total - confirmed - refuted
        gaps_open = sum(1 for g in self.gaps if not g.resolved)
        lines     = [f"🧪 АКТИВНОЕ ОБУЧЕНИЕ\n{'═'*24}",
                     f"Гипотез: {total} (открытых: {opens}, подтверждено: {confirmed}, опровергнуто: {refuted})",
                     f"Пробелов знаний: {gaps_open}"]
        for h in self.get_open_hypotheses():
            lines.append(f"  • [{h.confidence:.0%}] {h.statement}")
        return "\n".join(lines)


# ══════════════════════════════════════════
# v23 AGI MODULE 3: ТЕОРИЯ РАЗУМА
# ══════════════════════════════════════════
@dataclass
class MindBelief:
    topic:      str
    belief:     str         # что пользователь думает/знает
    confidence: float = 0.5
    is_correct: Optional[bool] = None  # None = неизвестно
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MindModel:
    knowledge_level: Dict[str, str]  = field(default_factory=dict)  # тема → 'expert'/'intermediate'/'beginner'
    beliefs:         List[MindBelief] = field(default_factory=list)
    motivations:     List[str]        = field(default_factory=list)  # зачем спрашивает
    misconceptions:  List[str]        = field(default_factory=list)  # заблуждения
    communication_pref: str = 'neutral'   # как предпочитает общение
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class TheoryOfMind:
    """
    🪞 Теория разума v23.

    Строит модель того, что пользователь знает, во что верит,
    чего хочет, и где может заблуждаться.
    Используется для адаптации объяснений и выявления misconceptions.
    """

    EXPERTISE_SIGNALS = {
        'expert':       ['очевидно','как известно','сложно','нюанс','архитектура','оптимизация'],
        'intermediate': ['понимаю','знаком','разбираюсь','слышал','пробовал'],
        'beginner':     ['не знаю','объясни','что такое','как работает','почему','с чего начать'],
    }
    MOTIVATION_PATTERNS = [
        (r'чтобы\s+([\w\s]{4,50})', 'цель'),
        (r'потому что\s+([\w\s]{4,50})', 'причина'),
        (r'для\s+([\w\s]{4,40})', 'применение'),
        (r'хочу\s+(?:знать|понять|научиться)\s+([\w\s]{4,40})', 'обучение'),
    ]
    MISCONCEPTION_MARKERS = [
        'всегда', 'никогда', 'все', 'никто', 'обязательно', 'точно знаю',
        '100%', 'абсолютно', 'невозможно', 'невозможно'
    ]

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.model     = MindModel()
        self._msg_count = 0
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        if data:
            self.model.knowledge_level   = data.get('knowledge_level', {})
            self.model.motivations       = data.get('motivations', [])
            self.model.misconceptions    = data.get('misconceptions', [])
            self.model.communication_pref = data.get('communication_pref', 'neutral')
            for b in data.get('beliefs', []):
                b.setdefault('is_correct', None)
                self.model.beliefs.append(MindBelief(**b))

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'knowledge_level':    self.model.knowledge_level,
            'beliefs':            [asdict(b) for b in self.model.beliefs[-TOM_MAX_BELIEFS:]],
            'motivations':        self.model.motivations[-20:],
            'misconceptions':     self.model.misconceptions[-20:],
            'communication_pref': self.model.communication_pref,
            'last_updated':       datetime.now().isoformat()
        })

    def update(self, text: str, cluster: Optional[str]):
        self._msg_count += 1
        if self._msg_count % TOM_UPDATE_EVERY != 0 and self._msg_count > 1:
            # Быстрое частичное обновление
            self._quick_update(text, cluster)
            return
        tl = text.lower()

        # Уровень экспертизы по теме
        if cluster:
            for level, signals in self.EXPERTISE_SIGNALS.items():
                if sum(1 for s in signals if s in tl) >= 2:
                    self.model.knowledge_level[cluster] = level
                    break

        # Мотивации
        for pattern, mtype in self.MOTIVATION_PATTERNS:
            m = re.search(pattern, tl)
            if m:
                motivation = f"[{mtype}] {m.group(1).strip()}"
                if motivation not in self.model.motivations:
                    self.model.motivations.append(motivation)

        # Misconceptions (абсолютные утверждения)
        if sum(1 for mk in self.MISCONCEPTION_MARKERS if mk in tl) >= 2:
            snippet = text[:80]
            if snippet not in self.model.misconceptions:
                self.model.misconceptions.append(snippet)

        self.model.last_updated = datetime.now().isoformat()

    def _quick_update(self, text: str, cluster: Optional[str]):
        if not cluster:
            return
        tl = text.lower()
        for level, signals in self.EXPERTISE_SIGNALS.items():
            if sum(1 for s in signals if s in tl) >= 1:
                if cluster not in self.model.knowledge_level:
                    self.model.knowledge_level[cluster] = level
                break

    def get_expertise_hint(self, cluster: Optional[str]) -> str:
        if not cluster:
            return ""
        level = self.model.knowledge_level.get(cluster, '')
        if level == 'expert':      return "Пользователь — эксперт в этой теме. Говори на равных."
        if level == 'beginner':    return "Пользователь только знакомится с темой. Объясняй просто."
        if level == 'intermediate': return "Пользователь знаком с базой. Можно углубляться."
        return ""

    def get_misconception_warning(self) -> str:
        if not self.model.misconceptions:
            return ""
        last = self.model.misconceptions[-1]
        return f"⚠️ Возможное заблуждение: {last[:60]}"

    def get_prompt_block(self, cluster: Optional[str] = None) -> str:
        lines = []
        hint = self.get_expertise_hint(cluster)
        if hint: lines.append(hint)
        if self.model.motivations:
            lines.append(f"Мотивация: {self.model.motivations[-1]}")
        warn = self.get_misconception_warning()
        if warn: lines.append(warn)
        return "\n".join(lines)

    def get_status(self) -> str:
        lines = [f"🪞 МОДЕЛЬ РАЗУМА\n{'═'*24}"]
        if self.model.knowledge_level:
            lines.append("Уровень знаний по темам:")
            for topic, level in list(self.model.knowledge_level.items())[:6]:
                emoji = {'expert':'🟢','intermediate':'🟡','beginner':'🔴'}.get(level,'⚪')
                lines.append(f"  {emoji} {topic}: {level}")
        if self.model.motivations:
            lines.append(f"Последняя мотивация: {self.model.motivations[-1]}")
        if self.model.misconceptions:
            lines.append(f"Заблуждений замечено: {len(self.model.misconceptions)}")
        return "\n".join(lines)


# ══════════════════════════════════════════
# v23 AGI MODULE 4: ПЛАНИРОВЩИК ЗАДАЧ
# ══════════════════════════════════════════
@dataclass
class Task:
    id:          str
    title:       str
    steps:       List[str] = field(default_factory=list)
    current_step: int = 0
    completed:   bool = False
    parent_id:   Optional[str] = None
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    query:       str = ""


class TaskPlanner:
    """
    📋 Планировщик задач v23.

    Распознаёт сложные многошаговые запросы,
    декомпозирует их на последовательные шаги,
    отслеживает выполнение.
    """

    COMPLEX_PATTERNS = [
        r'как\s+(?:мне|можно|сделать|создать|написать|разработать|построить)',
        r'помоги\s+(?:мне|с|создать|написать|разработать)',
        r'нужно\s+(?:создать|написать|разработать|сделать|построить)',
        r'хочу\s+(?:создать|написать|разработать|сделать|построить)',
        r'объясни\s+(?:как|почему|зачем|что)',
        r'составь\s+(?:план|список|инструкцию|алгоритм)',
    ]

    def __init__(self, save_path: str):
        self.save_path  = save_path
        self.tasks: Dict[str, Task] = {}
        self._active_task_id: Optional[str] = None
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        for k, v in data.items():
            v.setdefault('parent_id', None); v.setdefault('query', '')
            self.tasks[k] = Task(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path, {k: asdict(v) for k, v in self.tasks.items()})

    def is_complex_query(self, text: str) -> bool:
        tl = text.lower()
        return (any(re.search(p, tl) for p in self.COMPLEX_PATTERNS)
                and TextUtils.word_count(text) > 8)

    def create_task(self, query: str, steps: List[str]) -> Task:
        tid  = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        task = Task(id=tid, title=query[:60], steps=steps, query=query)
        self.tasks[tid] = task
        self._active_task_id = tid
        if len(self.tasks) > PLANNER_MAX_TASKS:
            oldest = sorted(self.tasks.values(), key=lambda x: x.created_at)
            del self.tasks[oldest[0].id]
        return task

    def advance_step(self) -> Optional[str]:
        if not self._active_task_id or self._active_task_id not in self.tasks:
            return None
        task = self.tasks[self._active_task_id]
        if task.completed:
            return None
        task.current_step = min(task.current_step + 1, len(task.steps) - 1)
        if task.current_step >= len(task.steps) - 1:
            task.completed = True
        return task.steps[task.current_step] if task.steps else None

    def get_active_task(self) -> Optional[Task]:
        if not self._active_task_id:
            return None
        return self.tasks.get(self._active_task_id)

    def get_prompt_block(self) -> str:
        task = self.get_active_task()
        if not task or task.completed:
            return ""
        lines = ["=== АКТИВНЫЙ ПЛАН ==="]
        for i, step in enumerate(task.steps):
            marker = "▶" if i == task.current_step else ("✅" if i < task.current_step else "○")
            lines.append(f"  {marker} Шаг {i+1}: {step}")
        lines.append("=====================")
        return "\n".join(lines)

    def get_status(self) -> str:
        active    = [t for t in self.tasks.values() if not t.completed]
        completed = [t for t in self.tasks.values() if t.completed]
        lines     = [f"📋 ПЛАНИРОВЩИК\n{'═'*24}"]
        if active:
            lines.append("Активные задачи:")
            for t in active[:3]:
                progress = f"{t.current_step}/{len(t.steps)}" if t.steps else "0/0"
                lines.append(f"  • {t.title} [{progress}]")
        lines.append(f"Выполнено: {len(completed)}")
        return "\n".join(lines)


# ══════════════════════════════════════════
# v23 AGI MODULE 5: ИНИЦИАТИВНОСТЬ
# ══════════════════════════════════════════
class ProactiveBehavior:
    """
    ⚡ Инициативное поведение v23.

    AI сам замечает паттерны и делится наблюдениями,
    предлагает темы, предупреждает о противоречиях.
    Срабатывает с вероятностью PROACTIVE_PROB каждые PROACTIVE_EVERY сообщений.
    """

    def __init__(self, save_path: str):
        self.save_path     = save_path
        self._counter      = 0
        self._observations: List[Dict] = []
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self._observations = data.get('observations', [])
        self._counter      = data.get('counter', 0)

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'observations': self._observations[-30:],
            'counter':      self._counter
        })

    def tick(self) -> bool:
        """Возвращает True если пора проявить инициативу"""
        self._counter += 1
        return (self._counter % PROACTIVE_EVERY == 0 and random.random() < PROACTIVE_PROB)

    def add_observation(self, observation: str, obs_type: str = 'pattern'):
        self._observations.append({
            'text':       observation,
            'type':       obs_type,
            'timestamp':  datetime.now().isoformat()
        })
        if len(self._observations) > 50:
            self._observations = self._observations[-50:]

    def generate_proactive_message(self,
                                    hot_nodes: List[Any],
                                    goals: List[Any],
                                    emotion_trend: Dict,
                                    topic_hint: str) -> Optional[str]:
        """Генерирует инициативное наблюдение без LLM (rule-based)"""
        messages = []

        # Наблюдение о теме
        if hot_nodes:
            top = hot_nodes[0].label
            messages.append(f"💡 Замечаю, что тема «{top}» часто всплывает в нашем разговоре.")

        # Наблюдение о целях
        stalled = [g for g in goals if 0.1 < g.progress < 0.3 and not g.completed]
        if stalled:
            messages.append(f"🎯 Хочу напомнить: у тебя есть незакрытая цель — «{stalled[0].title}». Хочешь продолжить?")

        # Наблюдение об эмоциях
        if emotion_trend.get('direction') == 'falling' and emotion_trend.get('avg', 0) < -0.2:
            messages.append("💚 Вижу, что настроение немного снижается. Всё в порядке?")

        # Предсказание темы
        if topic_hint:
            messages.append(f"🔮 {topic_hint}")

        if not messages:
            return None

        chosen = random.choice(messages)
        self.add_observation(chosen, 'proactive')
        return chosen

    def get_status(self) -> str:
        recent = self._observations[-5:]
        lines  = [f"⚡ ИНИЦИАТИВНОСТЬ\n{'═'*24}",
                  f"Счётчик: {self._counter} | Интервал: каждые {PROACTIVE_EVERY}",
                  f"Наблюдений: {len(self._observations)}"]
        if recent:
            lines.append("Последние:")
            for obs in recent:
                lines.append(f"  [{obs['type']}] {obs['text'][:60]}")
        return "\n".join(lines)


# ══════════════════════════════════════════
# v23 AGI MODULE 6: САМОМОДИФИКАЦИЯ
# ══════════════════════════════════════════
@dataclass
class BehaviorRule:
    id:          str
    description: str
    condition:   str    # когда применять
    action:      str    # что делать
    success_count: int = 0
    failure_count: int = 0
    active:      bool = True
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    last_used:   str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def score(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0: return 0.5
        return self.success_count / total


class SelfModification:
    """
    🔄 Самомодификация v23.

    AI ведёт журнал своих стратегий поведения.
    Отслеживает что работает, что нет.
    Периодически пересматривает правила.
    """

    DEFAULT_RULES = [
        ("Технический запрос",   "когда пользователь пишет код или просит алгоритм",
         "используй точный технический язык и примеры кода"),
        ("Эмоциональная поддержка", "когда пользователь расстроен или устал",
         "проявляй эмпатию, не торопись с решениями"),
        ("Короткий ответ",       "когда вопрос простой и конкретный",
         "отвечай кратко и по делу"),
        ("Подтверждение понимания", "когда объяснение сложное",
         "спроси в конце — всё ли понятно"),
        ("Использование примеров", "когда тема абстрактная",
         "приводи конкретные жизненные примеры"),
    ]

    def __init__(self, save_path: str):
        self.save_path   = save_path
        self.rules: Dict[str, BehaviorRule] = {}
        self._eval_counter = 0
        self._session_feedback: List[Tuple[str, bool]] = []  # (rule_id, success)
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        for k, v in data.items():
            self.rules[k] = BehaviorRule(**v)
        if not self.rules:
            self._init_defaults()

    def save(self):
        FileManager.safe_save_json(self.save_path, {k: asdict(v) for k, v in self.rules.items()})

    def _init_defaults(self):
        for desc, cond, action in self.DEFAULT_RULES:
            rid  = hashlib.md5(desc.encode()).hexdigest()[:8]
            rule = BehaviorRule(id=rid, description=desc, condition=cond, action=action)
            self.rules[rid] = rule

    def add_rule(self, description: str, condition: str, action: str) -> BehaviorRule:
        rid  = hashlib.md5(f"{description}{condition}".encode()).hexdigest()[:8]
        rule = BehaviorRule(id=rid, description=description, condition=condition, action=action)
        self.rules[rid] = rule
        if len(self.rules) > SELF_MOD_MAX_RULES:
            # Удаляем худшее правило
            worst = min([r for r in self.rules.values() if r.active],
                        key=lambda x: x.score, default=None)
            if worst: del self.rules[worst.id]
        return rule

    def get_relevant_rules(self, text: str, cluster: Optional[str], emotion: str) -> List[BehaviorRule]:
        """Возвращает активные правила, релевантные текущему контексту"""
        relevant = []
        tl = text.lower()
        for rule in self.rules.values():
            if not rule.active:
                continue
            cond = rule.condition.lower()
            # Проверяем совпадение условия с контекстом
            cond_kw = set(TextUtils.tokenize(cond))
            text_kw = set(TextUtils.tokenize(tl))
            if cluster and cluster.lower() in cond:
                relevant.append(rule)
            elif emotion != 'neutral' and emotion in cond:
                relevant.append(rule)
            elif len(cond_kw & text_kw) >= 2:
                relevant.append(rule)
        return sorted(relevant, key=lambda x: -x.score)[:3]

    def record_feedback(self, rule_id: str, success: bool):
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            if success: rule.success_count += 1
            else:       rule.failure_count += 1
            rule.last_used = datetime.now().isoformat()
        self._session_feedback.append((rule_id, success))

    def evaluate(self) -> Optional[str]:
        """Периодически пересматривает правила"""
        self._eval_counter += 1
        if self._eval_counter % SELF_MOD_EVAL_EVERY != 0:
            return None
        # Деактивируем правила с плохим score
        deactivated = []
        for rule in self.rules.values():
            total = rule.success_count + rule.failure_count
            if total >= 5 and rule.score < 0.25:
                rule.active = False
                deactivated.append(rule.description)
        if deactivated:
            return f"⚙️ Деактивировал неэффективные стратегии: {', '.join(deactivated[:2])}"
        return None

    def get_prompt_block(self, text: str, cluster: Optional[str], emotion: str) -> str:
        rules = self.get_relevant_rules(text, cluster, emotion)
        if not rules:
            return ""
        lines = ["=== СТРАТЕГИИ ПОВЕДЕНИЯ ==="]
        for rule in rules[:2]:
            lines.append(f"  • {rule.action}  [эффект: {rule.score:.0%}]")
        return "\n".join(lines)

    def get_status(self) -> str:
        active   = [r for r in self.rules.values() if r.active]
        inactive = [r for r in self.rules.values() if not r.active]
        lines    = [f"🔄 САМОМОДИФИКАЦИЯ\n{'═'*24}",
                    f"Правил: {len(active)} активных, {len(inactive)} деактивировано"]
        for rule in sorted(active, key=lambda x: -x.score)[:5]:
            bar = "█"*int(rule.score*10) + "░"*(10-int(rule.score*10))
            lines.append(f"  {bar} {rule.description}")
        return "\n".join(lines)


# ══════════════════════════════════════════
# v22: КОГНИТИВНЫЙ РИТМ (PULSE) — СОХРАНЕНО
# ══════════════════════════════════════════
@dataclass
class PulsePhaseRecord:
    phase: str; input_len: int; output: str; duration_ms: float
    skipped: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PulseLog:
    query: str; phases: List[PulsePhaseRecord] = field(default_factory=list)
    total_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, phase: PulsePhaseRecord): self.phases.append(phase)

    def summary(self) -> str:
        lines = [f"🥁 Пульс [{self.timestamp[:16]}]", f"   Запрос: {self.query[:60]}"]
        for p in self.phases:
            icon = "⏭️" if p.skipped else "✅"
            out  = p.output[:50].replace('\n',' ') if p.output else "—"
            lines.append(f"   {icon} {p.phase:<12} {p.duration_ms:>6.0f}мс → {out}")
        lines.append(f"   Итого: {self.total_ms:.0f}мс")
        return "\n".join(lines)


class PulseJournal:
    def __init__(self, save_path: str):
        self.save_path = save_path; self.logs: List[PulseLog] = []; self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, [])
        for entry in data:
            phases = [PulsePhaseRecord(**p) for p in entry.get('phases',[])]
            self.logs.append(PulseLog(query=entry['query'],phases=phases,
                total_ms=entry.get('total_ms',0.0),timestamp=entry.get('timestamp',datetime.now().isoformat())))

    def save(self):
        data = []
        for log in self.logs[-PULSE_LOG_MAX:]:
            data.append({'query':log.query,'phases':[asdict(p) for p in log.phases],
                         'total_ms':log.total_ms,'timestamp':log.timestamp})
        FileManager.safe_save_json(self.save_path, data)

    def add(self, log: PulseLog):
        self.logs.append(log)
        if len(self.logs) > PULSE_LOG_MAX: self.logs = self.logs[-PULSE_LOG_MAX:]

    def last(self, n: int=3) -> List[PulseLog]: return self.logs[-n:]
    def get_status(self) -> str: return self.logs[-1].summary() if self.logs else "Журнал пуст."


class CognitivePulse:
    PHASE_PROMPTS = {
        'PERCEIVE': (
            "Когнитивный анализ запроса. Ответь КРАТКО (2-3 предложения):\n"
            "1. Явный запрос: что конкретно спрашивает пользователь?\n"
            "2. Скрытый запрос: что он на самом деле хочет?\n"
            "3. Тип: [фактический/эмоциональный/творческий/технический]\n"
            "---\nЗапрос: {query}\nКонтекст: {context_hint}"
        ),
        'UNDERSTAND': (
            "Сбор знаний. Ответь КРАТКО (3-4 предложения):\n"
            "Что релевантно из памяти и контекста? Какие ключевые факты?\n"
            "---\nАнализ: {perceive_output}\nПамять: {memory_block}\nL3: {l3_block}"
        ),
        'REFLECT': (
            "Критический анализ. Ответь КРАТКО (2-3 предложения):\n"
            "Есть ли противоречия? Что важно не упустить?\n"
            "---\nАнализ: {perceive_output}\nЗнания: {understand_output}\n"
            "Профиль: {profile_hint}\nЭмоция: {emotion_hint}"
        ),
        'SYNTHESIZE': (
            "Построй ПЛАН ответа в 3-5 пунктах. Учти стиль и эмоцию.\n"
            "---\nВосприятие: {perceive_output}\nЗнания: {understand_output}\n"
            "Рефлексия: {reflect_output}\nТема: {cluster_hint} | Темп: {temp_hint}"
        ),
        'RESPOND': (
            "Ты — умный ассистент с долговременной памятью.\n{system_block}\n\n"
            "=== ПЛАН ===\n{synthesize_output}\n=== КОНЕЦ ПЛАНА ===\n\n"
            "Напиши финальный ответ пользователю, следуя плану. Будь естественным.\n"
            "---\nЗапрос: {query}"
        )
    }

    def __init__(self, llm: 'LLMInterface', journal: PulseJournal):
        self.llm = llm; self.journal = journal

    def _is_short_query(self, text: str) -> bool:
        return TextUtils.word_count(text) < PULSE_SHORT_THRESHOLD

    async def _run_phase(self, phase_name: str, prompt: str, temp: float,
                         max_tokens: int = PULSE_PHASE_MAX_TOKENS) -> Tuple[str, float]:
        t0     = asyncio.get_event_loop().time()
        result = await self.llm.generate_raw(prompt, temp=temp, max_tokens=max_tokens)
        dt     = (asyncio.get_event_loop().time() - t0) * 1000
        return result or "", dt

    async def think(self, query: str, system_block: str, memory_block: str,
                    l3_block: str, profile_hint: str, emotion_hint: str,
                    cluster_hint: str, context_hint: str, temp: float) -> Tuple[str, PulseLog]:
        pulse_log = PulseLog(query=query)
        t_total   = asyncio.get_event_loop().time()

        # PERCEIVE
        po, pms = await self._run_phase('PERCEIVE', self.PHASE_PROMPTS['PERCEIVE'].format(
            query=query, context_hint=context_hint[:200] if context_hint else "нет"), temp=0.3)
        pulse_log.add(PulsePhaseRecord(phase='PERCEIVE',input_len=len(query),output=po,duration_ms=pms))

        # UNDERSTAND
        uo, ums = await self._run_phase('UNDERSTAND', self.PHASE_PROMPTS['UNDERSTAND'].format(
            perceive_output=po[:300],
            memory_block=memory_block[:400] if memory_block else "нет",
            l3_block=l3_block[:200] if l3_block else "нет"), temp=0.35)
        pulse_log.add(PulsePhaseRecord(phase='UNDERSTAND',input_len=len(uo),output=uo,duration_ms=ums))

        # REFLECT
        ro, rms = await self._run_phase('REFLECT', self.PHASE_PROMPTS['REFLECT'].format(
            perceive_output=po[:250], understand_output=uo[:250],
            profile_hint=profile_hint[:150] if profile_hint else "нет",
            emotion_hint=emotion_hint[:100] if emotion_hint else "нейтральный"), temp=0.4)
        pulse_log.add(PulsePhaseRecord(phase='REFLECT',input_len=len(ro),output=ro,duration_ms=rms))

        # SYNTHESIZE
        so, sms = await self._run_phase('SYNTHESIZE', self.PHASE_PROMPTS['SYNTHESIZE'].format(
            perceive_output=po[:200], understand_output=uo[:200], reflect_output=ro[:200],
            cluster_hint=cluster_hint or "общий", temp_hint=f"{temp:.2f}"), temp=0.4)
        pulse_log.add(PulsePhaseRecord(phase='SYNTHESIZE',input_len=len(so),output=so,duration_ms=sms))

        # RESPOND
        resp, rems = await self._run_phase('RESPOND', self.PHASE_PROMPTS['RESPOND'].format(
            system_block=system_block[:800], synthesize_output=so[:500], query=query),
            temp=temp, max_tokens=PULSE_FINAL_MAX_TOKENS)
        pulse_log.add(PulsePhaseRecord(phase='RESPOND',input_len=len(system_block),output=resp,duration_ms=rems))

        pulse_log.total_ms = (asyncio.get_event_loop().time() - t_total) * 1000
        self.journal.add(pulse_log)
        print(f"🥁 Пульс: {pulse_log.total_ms:.0f}мс")
        return resp, pulse_log

    async def think_fast(self, query: str, system_block: str, temp: float) -> Tuple[str, PulseLog]:
        pulse_log = PulseLog(query=query)
        t0 = asyncio.get_event_loop().time()
        resp, rems = await self._run_phase('RESPOND',
            f"Ты — умный ассистент.\n{system_block[:600]}\n\nЗапрос: {query}",
            temp=temp, max_tokens=PULSE_FINAL_MAX_TOKENS)
        for phase in ['PERCEIVE','UNDERSTAND','REFLECT','SYNTHESIZE']:
            pulse_log.add(PulsePhaseRecord(phase=phase,input_len=0,
                output="[пропущено]",duration_ms=0,skipped=True))
        pulse_log.add(PulsePhaseRecord(phase='RESPOND',input_len=len(system_block),output=resp,duration_ms=rems))
        pulse_log.total_ms = (asyncio.get_event_loop().time() - t0) * 1000
        self.journal.add(pulse_log)
        return resp, pulse_log


# ══════════════════════════════════════════
# РЕФЛЕКСИЯ
# ══════════════════════════════════════════
class ReflectionEngine:
    def __init__(self, llm: 'LLMInterface'):
        self.llm = llm; self._counter = 0

    def tick(self) -> Tuple[bool, bool]:
        self._counter += 1
        return (self._counter % REFLECTION_FAST_EVERY == 0,
                self._counter % REFLECTION_DEEP_EVERY == 0)

    async def reflect_fast(self, memory: 'HybridMemorySystem') -> Optional[str]:
        hot    = memory.cortex.get_hot_nodes(5)
        recent = memory.l1.history[-10:]
        if not hot or not recent: return None
        session_topics = list(dict.fromkeys(
            memory.clusters.classify(TextUtils.extract_keywords(m['content']))[0][0]
            for m in recent
            if memory.clusters.classify(TextUtils.extract_keywords(m['content']))
        ))[:5]
        prompt = (
            "Сделай 1 краткое наблюдение (до 30 слов) о сессии:\n"
            f"Горячие темы: {[n.label for n in hot]}\nЧередование: {session_topics}\n"
            f"Эмоция: {memory.emotion.get_trend()['trend']}\n"
            "Если есть противоречие — отметь.")
        result = await self.llm.generate(prompt, temp=0.35)
        if result and len(result) > 8: memory.meta.add_insight(f"[быстрая] {result}"); return result
        return None

    async def reflect_deep(self, memory: 'HybridMemorySystem') -> Optional[str]:
        memory.semantic_l3.force_rebuild()
        chains = memory.causal.get_strong_chains(4)
        hot    = memory.cortex.get_hot_nodes(5)
        arc    = memory.cortex.get_archived_nodes(3)
        prompt = (
            "Опиши 2 ключевых наблюдения об интересах пользователя (≤50 слов):\n"
            f"Профиль: {memory.profile.get_prompt_block() or 'нет'}\n"
            f"L3: {memory.semantic_l3.facts}\nГорячие: {[n.label for n in hot]}\n"
            f"Архив: {[n.label for n in arc]}\nЦепочки: {chains}\nТолько факты.")
        result = await self.llm.generate(prompt, temp=0.40)
        if result and len(result) > 8: memory.meta.add_insight(f"[глубокая] {result}"); return result
        return None


# ══════════════════════════════════════════
# ГИБРИДНАЯ ПАМЯТЬ v23
# ══════════════════════════════════════════
class HybridMemorySystem:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir     = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.dir, exist_ok=True)

        # v21/v22 компоненты
        self.l1          = SmartWorkingMemory(os.path.join(self.dir, "short_term.json"))
        self.cortex      = DynamicNeuralCortex(user_id, self.dir)
        self.temporal    = TemporalMemory(user_id, self.dir)
        self.causal      = LogicalChainEngine(os.path.join(self.dir, "causal.json"))
        self.clusters    = SemanticClusterEngine(os.path.join(self.dir, "clusters.json"))
        self.meta        = MetaLearner(os.path.join(self.dir, "meta.json"))
        self.profile     = ProfileManager(os.path.join(self.dir, "profile.json"))
        self.emotion     = EmotionalArc(os.path.join(self.dir, "emotions.json"))
        self.predictor   = TopicPredictor(os.path.join(self.dir, "topic_transitions.json"))
        self.semantic_l3 = SemanticMemoryL3(os.path.join(self.dir, "semantic_l3.json"))
        self.temp_adapter= TemperatureAdapter(os.path.join(self.dir, "temp_feedback.json"))
        self.pulse_journal= PulseJournal(os.path.join(self.dir, "pulse_journal.json"))

        # v23: AGI-модули
        self.goals        = GoalSystem(os.path.join(self.dir, "goals.json"))
        self.active_learn = ActiveLearning(os.path.join(self.dir, "active_learning.json"))
        self.mind         = TheoryOfMind(os.path.join(self.dir, "theory_of_mind.json"))
        self.planner      = TaskPlanner(os.path.join(self.dir, "task_planner.json"))
        self.proactive    = ProactiveBehavior(os.path.join(self.dir, "proactive.json"))
        self.self_mod     = SelfModification(os.path.join(self.dir, "self_modification.json"))

        self.now = datetime.now()
        self._msg_count      = 0
        self._prev_cluster:  Optional[str] = None
        self._pprev_cluster: Optional[str] = None

    async def process(self, text: str):
        self.now = datetime.now()

        # ── Базовая обработка (v21) ────────────────────────────
        concepts, chrono_nodes, explored, wild = self.cortex.process_input(text, self.now)
        keywords     = TextUtils.extract_keywords(text)
        cluster_name = self.clusters.learn(keywords)
        clusters     = self.clusters.classify(keywords)

        if self._prev_cluster:
            self.temp_adapter.feedback(text, self._prev_cluster)
        if self._prev_cluster and cluster_name:
            self.predictor.record_transition(self._prev_cluster, cluster_name)
        if cluster_name:
            self._pprev_cluster = self._prev_cluster
            self._prev_cluster  = cluster_name

        causal_links = self.causal.extract_from_text(text)
        xp_gain      = 3 + len(causal_links) * 5
        xp_msg       = self.meta.gain_xp(xp_gain, "chain_found" if causal_links else "interaction",
                                          str([f"{l.cause}→{l.effect}" for l in causal_links]))
        new_facts          = self.profile.update_from_text(text)
        emotion_str, _     = self.emotion.record(text)
        session_emotion, mood_hint = self.emotion.get_session_mood()
        event              = self.temporal.add_event(text, keywords, cluster_name, self.now)
        self.cortex.chrono_layer.link_event(event, chrono_nodes)
        self.semantic_l3.record(cluster_name, self.now)

        rag_results = RAGEngineV2.search(text, self.temporal.events, top_k=RAG_TOP_K,
                                          now=self.now, current_emotion=session_emotion)
        rag_block   = RAGEngineV2.format_for_prompt(rag_results)

        causal_ctx = ""
        for kw in keywords[:3]:
            ctx = self.causal.format_causal_for_prompt(kw, text)
            if ctx: causal_ctx = ctx; break

        drift_hint   = self.cortex.drift.describe_drift(explored, wild)
        temp         = self.temp_adapter.get(cluster_name, session_emotion, text)
        prediction   = self.predictor.get_hint(cluster_name or "", self._pprev_cluster)
        l1_ctx       = self.l1.format_for_prompt(text)
        profile_block = self.profile.get_prompt_block()
        l3_block      = self.semantic_l3.get_prompt_block()
        chains        = self.causal.get_strong_chains(3)
        insights      = self.meta.get_insights_prompt(2)
        hot_labels    = [n.label for n in self.cortex.get_hot_nodes(5)]
        chrono_active = self.cortex.chrono_layer.get_active_labels()

        # ── v23: AGI-обновления ────────────────────────────────
        new_goals     = self.goals.extract_from_text(text, cluster_name)
        self.goals.update_progress(text, keywords)
        self.active_learn.update_hypothesis(text)
        contradiction = self.active_learn.detect_contradiction(text, self.l1.history)
        self.mind.update(text, cluster_name)
        self_mod_eval = self.self_mod.evaluate()
        proactive_msg: Optional[str] = None
        if self.proactive.tick():
            proactive_msg = self.proactive.generate_proactive_message(
                self.cortex.get_hot_nodes(3),
                self.goals.get_active_goals(),
                self.emotion.get_trend(),
                prediction
            )

        # Уточняющий вопрос от активного обучения
        clarification = self.active_learn.should_ask_clarification()

        # ── Сборка системного промпта ──────────────────────────
        sys_parts = []
        if profile_block:  sys_parts.append(f"=== ПРОФИЛЬ ===\n{profile_block}")
        if l3_block:       sys_parts.append(l3_block)
        if hot_labels:     sys_parts.append(f"Горячие темы: {', '.join(hot_labels)}.")
        if chrono_active:  sys_parts.append(f"Временной контекст: {', '.join(chrono_active)}.")
        if chains:         sys_parts.append(f"Закономерности: {'; '.join(chains)}.")
        if clusters:       sys_parts.append(f"Тема: {clusters[0][0]}.")
        if mood_hint:      sys_parts.append(mood_hint)
        if prediction:     sys_parts.append(prediction)
        if drift_hint:     sys_parts.append(drift_hint)
        if insights:       sys_parts.append(insights)
        if causal_ctx:     sys_parts.append(f"=== ПРИЧИННЫЙ КОНТЕКСТ ===\n{causal_ctx}")

        # v23 блоки
        goals_block   = self.goals.get_prompt_block()
        mind_block    = self.mind.get_prompt_block(cluster_name)
        plan_block    = self.planner.get_prompt_block()
        selfmod_block = self.self_mod.get_prompt_block(text, cluster_name, session_emotion)

        if goals_block:   sys_parts.append(goals_block)
        if mind_block:    sys_parts.append(f"=== МОДЕЛЬ РАЗУМА ===\n{mind_block}")
        if plan_block:    sys_parts.append(plan_block)
        if selfmod_block: sys_parts.append(selfmod_block)
        if contradiction: sys_parts.append(f"⚠️ ПРОТИВОРЕЧИЕ: {contradiction}")

        sys_mod = "\n".join(sys_parts) if sys_parts else "Ты умный ассистент с долговременной памятью."

        full_system = (
            f"{sys_mod}\n\n"
            + (f"{rag_block}\n\n" if rag_block else "")
            + f"=== ИСТОРИЯ (L1 — умная выборка) ===\n{l1_ctx}\n\n"
            + f"=== ВРЕМЯ ===\n{self.now.strftime('%H:%M %d.%m.%Y')}"
        )

        self.l1.add('user', text)
        self._msg_count += 1
        if self._msg_count % AUTOSAVE_EVERY == 0:
            self.save_all(); print(f"💾 Автосохранение #{self._msg_count}")

        return (
            full_system, temp, concepts, chrono_nodes, xp_msg, new_facts,
            rag_block, l3_block, profile_block, mood_hint,
            clusters[0][0] if clusters else None, l1_ctx,
            new_goals, proactive_msg, clarification, contradiction, self_mod_eval
        )

    def handle_temporal_query(self, text: str) -> Optional[str]:
        if not any(w in text.lower() for w in ['был','делал','ходил','где','вчера','раньше','утром','вечером']):
            return None
        events = self.temporal.query_by_time(text, self.now)
        if events:
            lines = []
            for e in events[:4]:
                rel = TimeEncoder.describe(datetime.fromisoformat(e.timestamp), self.now)
                lines.append(f"{rel}: {'['+e.cluster_name+'] ' if e.cluster_name else ''}{e.content}")
            return "🕰️ Из памяти:\n" + "\n".join(lines)
        return None

    def handle_logic_query(self, text: str) -> Optional[str]:
        if not any(w in text.lower() for w in ['почему','зачем','причина','что будет','если не','без ']):
            return None
        for kw in TextUtils.extract_keywords(text):
            ctx = self.causal.format_causal_for_prompt(kw, text)
            if ctx: return f"🔗 {ctx}"
        return None

    def add_response(self, text: str): self.l1.add('assistant', text)

    def save_all(self):
        self.cortex.save(); self.temporal.save(); self.causal.save()
        self.clusters.save(); self.meta.save(); self.profile.save()
        self.emotion.save(); self.predictor.save(); self.semantic_l3.save()
        self.l1.save(); self.temp_adapter.save(); self.pulse_journal.save()
        # v23
        self.goals.save(); self.active_learn.save(); self.mind.save()
        self.planner.save(); self.proactive.save(); self.self_mod.save()

    def maintenance(self):
        if self.meta.total_interactions % 60 == 0:
            pruned = self.cortex.prune_weak_synapses(0.04)
            self.causal.decay_all()
            if pruned: print(f"✂️ Удалено {pruned} слабых синапсов")


# ══════════════════════════════════════════
# LLM
# ══════════════════════════════════════════
class LLMInterface:
    def __init__(self, url: str, key: str):
        self.url = url; self.key = key; self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not self.session: self.session = aiohttp.ClientSession()

    async def generate_raw(self, prompt: str, system: str=None,
                            temp: float=TEMP_DEFAULT, max_tokens: int=1500) -> str:
        if not self.session: await self.init()
        msgs = []
        if system: msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        try:
            async with self.session.post(self.url,
                json={"messages":msgs,"temperature":temp,"max_tokens":max_tokens},
                headers={"Authorization":f"Bearer {self.key}"}) as r:
                if r.status == 200: return (await r.json())['choices'][0]['message']['content']
                return f"LM Error: {r.status}"
        except Exception as e: return f"Connection error: {e}"

    async def generate(self, prompt: str, system: str=None, temp: float=TEMP_DEFAULT) -> str:
        return await self.generate_raw(prompt, system=system, temp=temp, max_tokens=1500)

    async def close(self):
        if self.session: await self.session.close(); self.session = None


# ══════════════════════════════════════════
# БОТ v23
# ══════════════════════════════════════════
class HybridBot:
    def __init__(self):
        self.llm       = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.users:    Dict[str, HybridMemorySystem] = {}
        self.reflector = None
        self.stop_flag = False

    def get_brain(self, uid: str) -> HybridMemorySystem:
        if uid not in self.users: self.users[uid] = HybridMemorySystem(uid)
        return self.users[uid]

    def get_pulse(self, brain: HybridMemorySystem) -> CognitivePulse:
        return CognitivePulse(self.llm, brain.pulse_journal)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text: return
        uid   = str(update.effective_user.id)
        text  = update.message.text
        brain = self.get_brain(uid)
        await context.bot.send_chat_action(uid, "typing")

        result = await brain.process(text)
        (full_system, temp, concepts, chrono_nodes, xp_msg, new_facts,
         rag_block, l3_block, profile_block, mood_hint, cluster_hint, l1_ctx,
         new_goals, proactive_msg, clarification, contradiction, self_mod_eval) = result

        # Временной запрос
        ta = brain.handle_temporal_query(text)
        if ta:
            brain.add_response(ta); await update.message.reply_text(ta)
            if xp_msg: await update.message.reply_text(xp_msg)
            return

        # Логический запрос
        la = brain.handle_logic_query(text)
        if la:
            brain.add_response(la); await update.message.reply_text(la); return

        # v23: Планировщик для сложных запросов
        if brain.planner.is_complex_query(text):
            # Создаём базовый план (будет уточнён LLM)
            default_steps = ["Проанализировать запрос", "Собрать информацию", "Подготовить ответ", "Проверить результат"]
            brain.planner.create_task(text, default_steps)

        # Когнитивный ритм (v22)
        pulse = self.get_pulse(brain)
        if PULSE_ENABLED and not pulse._is_short_query(text):
            response, pulse_log = await pulse.think(
                query=text, system_block=full_system, memory_block=rag_block,
                l3_block=l3_block, profile_hint=profile_block, emotion_hint=mood_hint,
                cluster_hint=cluster_hint or "", context_hint=l1_ctx, temp=temp)
        else:
            response, pulse_log = await pulse.think_fast(
                query=text, system_block=full_system, temp=temp)

        brain.cortex.reinforce_path(concepts, chrono_nodes, len(response) > 10)

        # Рефлексия
        if self.reflector:
            fast_trigger, deep_trigger = self.reflector.tick()
            if deep_trigger:   asyncio.create_task(self._reflect_deep(brain))
            elif fast_trigger: asyncio.create_task(self._reflect_fast(brain))

        brain.maintenance()
        brain.add_response(response)

        # Добавляем уточняющий вопрос в конец ответа (активное обучение)
        if clarification and len(response) > 50:
            response += f"\n\n💭 {clarification}"

        await update.message.reply_text(response)

        # Доп. сообщения
        extra_messages = []
        if xp_msg:         extra_messages.append(xp_msg)
        if new_facts:      extra_messages.append(f"👤 Запомнил: {', '.join(new_facts)}")
        if new_goals:      extra_messages.append(f"🎯 Зафиксировал цель: {new_goals[0].title}")
        if proactive_msg:  extra_messages.append(proactive_msg)
        if self_mod_eval:  extra_messages.append(self_mod_eval)

        for msg in extra_messages:
            await update.message.reply_text(msg)

    async def _reflect_fast(self, brain: HybridMemorySystem):
        try:
            ins = await self.reflector.reflect_fast(brain)
            if ins: print(f"💡 [Быстрая] {ins[:80]}")
        except Exception as e: print(f"⚠️ Fast reflection: {e}")

    async def _reflect_deep(self, brain: HybridMemorySystem):
        try:
            ins = await self.reflector.reflect_deep(brain)
            if ins: print(f"🔍 [Глубокая] {ins[:100]}")
        except Exception as e: print(f"⚠️ Deep reflection: {e}")

    # ═══ КОМАНДЫ ═══

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        s   = brain.cortex.get_stats(); chains = brain.causal.get_strong_chains(3)
        adj_str = ", ".join(f"{k}:{v:+.2f}" for k,v in brain.temp_adapter.cluster_adjustments.items()) or "нет"
        await update.message.reply_text(
            f"🧠 AGI APPROXIMATION v23.0\n{'═'*32}\n"
            f"🔹 Нейроны: {s['neurons']} (тёплых: {s['warm']}, архив: {s['archived']})\n"
            f"🔹 Синапсы: {s['synapses']} (сильных: {s['strong_syn']})\n"
            f"🔹 Эпизодов L2: {len(brain.temporal.events)}\n"
            f"🔹 Фактов L3: {len(brain.semantic_l3.facts)}\n"
            f"🔹 Причинных связей: {len(brain.causal.links)}\n"
            f"🥁 Пульсов: {len(brain.pulse_journal.logs)}\n"
            f"🎯 Целей: {len(brain.goals.get_active_goals())} активных\n"
            f"🧪 Гипотез: {len(brain.active_learn.hypotheses)}\n"
            f"🪞 Уровней знания: {len(brain.mind.model.knowledge_level)}\n"
            f"📋 Задач: {len(brain.planner.tasks)}\n"
            f"🔄 Правил поведения: {len([r for r in brain.self_mod.rules.values() if r.active])}\n"
            f"🌡️ Темп. коррекции: {adj_str}\n"
            f"{'═'*32}\n{brain.meta.get_status()}\n{'═'*32}\n"
            f"🔗 Цепочки:\n  " + ("\n  ".join(chains) if chains else "Пока нет"))

    async def cmd_goals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        await update.message.reply_text(brain.goals.get_status())

    async def cmd_hypotheses(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        await update.message.reply_text(brain.active_learn.get_status())

    async def cmd_mindmodel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        await update.message.reply_text(brain.mind.get_status())

    async def cmd_plan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        task = brain.planner.get_active_task()
        if not task:
            await update.message.reply_text("Активных планов нет.\n\nЗадайте сложный многошаговый вопрос — я составлю план.")
            return
        lines = [f"📋 АКТИВНЫЙ ПЛАН: {task.title}\n{'═'*24}"]
        for i, step in enumerate(task.steps):
            marker = "▶" if i==task.current_step else ("✅" if i<task.current_step else "○")
            lines.append(f"{marker} {i+1}. {step}")
        lines.append(f"\nПрогресс: {task.current_step}/{len(task.steps)}")
        if task.completed: lines.append("✅ Выполнено!")
        await update.message.reply_text("\n".join(lines))

    async def cmd_selfmodel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        await update.message.reply_text(brain.self_mod.get_status())

    async def cmd_proactive(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        await update.message.reply_text(brain.proactive.get_status())

    async def cmd_pulse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        n     = int(context.args[0]) if context.args and context.args[0].isdigit() else 1
        logs  = brain.pulse_journal.last(n)
        if not logs:
            await update.message.reply_text("Журнал пульса пуст. Напишите что-нибудь."); return
        lines = [f"🥁 КОГНИТИВНЫЙ РИТМ\n{'═'*30}"]
        for log in logs: lines.append(log.summary()); lines.append("─"*30)
        await update.message.reply_text("\n".join(lines))

    async def cmd_phase(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid    = str(update.effective_user.id); brain = self.get_brain(uid)
        pname  = context.args[0].upper() if context.args else "SYNTHESIZE"
        logs   = brain.pulse_journal.last(1)
        if not logs: await update.message.reply_text("Нет данных."); return
        target = next((p for p in logs[0].phases if p.phase==pname), None)
        if not target:
            await update.message.reply_text(f"Фаза «{pname}» не найдена."); return
        await update.message.reply_text(
            f"🔬 ФАЗА {target.phase}\n{'═'*28}\n"
            f"Время: {target.duration_ms:.0f}мс\n\nВывод:\n{target.output[:600] or '—'}")

    async def cmd_hotmap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        hot = brain.cortex.get_hot_nodes(10); exp = brain.cortex.get_experienced_nodes(5)
        arc = brain.cortex.get_archived_nodes(5)
        if not hot: await update.message.reply_text("Горячих нейронов нет."); return
        lines = ["🔥 ГОРЯЧАЯ КАРТА\n", "Активные:"]
        for n in hot[:6]:
            bar = "█"*int(n.activation*10)
            lines.append(f"  {n.label:<15} {bar} {n.activation:.2f} (exp:{n.experience:.1f})")
        lines.append("\n🏆 Опытные:")
        for n in exp: lines.append(f"  {n.label:<15} exp:{n.experience:.1f}")
        if arc:
            lines.append("\n💤 Архивные:")
            for n in arc[:3]: lines.append(f"  {n.label:<15} exp:{n.experience:.1f}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        l3  = brain.semantic_l3.get_prompt_block()
        await update.message.reply_text(
            f"👤 ПРОФИЛЬ\n{'═'*24}\n{brain.profile.get_status()}\n\n{l3 or 'L3 пока пуст'}")

    async def cmd_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid  = str(update.effective_user.id); brain = self.get_brain(uid)
        days = int(context.args[0]) if context.args and context.args[0].isdigit() else 7
        t    = brain.emotion.get_trend()
        await update.message.reply_text(
            f"💚 ЭМОЦИОНАЛЬНАЯ ДУГА\n{'═'*24}\n"
            f"{brain.emotion.get_history_summary(days)}\nТренд: {t['trend']} | {t['direction']}")

    async def cmd_chain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /chain <слово>"); return
        word = context.args[0].lower(); chain = brain.causal.build_chain(word)
        lines = []
        if len(chain)>1: lines.append(f"🔗 Прямая: {' → '.join(chain)}")
        bt = brain.causal.backward_chain(word, depth=2)
        if bt['causes']:
            lines.append(f"\n⬆️ Причины «{word}»:")
            for c in bt['causes']:
                sub = " ← ".join(cc['node'] for cc in c.get('causes',[]))
                lines.append(f"  ← {c['node']} ({c['strength']:.2f})"+(f" ← {sub}" if sub else ""))
        ft = brain.causal.forward_chain(word, depth=2)
        if ft['effects']:
            lines.append(f"\n⬇️ Следствия «{word}»:")
            for e in ft['effects']:
                sub = " → ".join(ee['node'] for ee in e.get('effects',[]))
                lines.append(f"  → {e['node']} ({e['strength']:.2f})"+(f" → {sub}" if sub else ""))
        cf = brain.causal.counterfactual(word)
        if cf: lines.append(f"\n🔴 Без «{word}»: {', '.join(cf)}")
        if not lines: await update.message.reply_text(f"Цепочек для '{word}' нет.")
        else: await update.message.reply_text("\n".join(lines))

    async def cmd_assoc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /assoc <слово>"); return
        word = context.args[0].lower(); assocs = brain.cortex.associate(word)
        if assocs: await update.message.reply_text(
            f"🧩 Ассоциации '{word}':\n" + "\n".join(f"  {a} ({w:.2f})" for a,w in assocs))
        else: await update.message.reply_text(f"Ассоциаций для '{word}' нет.")

    async def cmd_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid  = str(update.effective_user.id); brain = self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /predict <тема>"); return
        topic = " ".join(context.args).lower()
        p1 = brain.predictor.predict_next(topic, top_n=3)
        p2 = brain.predictor.predict_next(topic, prev=brain._pprev_cluster, top_n=2)
        lines = [f"🔮 После «{topic}»:"]
        if p1:
            lines.append("  1-й порядок:")
            for t,prob in p1: lines.append(f"    {t:<15} {'█'*int(prob*10)} {prob:.0%}")
        if p2 and brain._pprev_cluster:
            lines.append(f"  2-й порядок:")
            for t,prob in p2: lines.append(f"    {t:<15} {'█'*int(prob*10)} {prob:.0%}")
        if len(lines)==1: await update.message.reply_text("Данных пока нет.")
        else: await update.message.reply_text("\n".join(lines))

    async def cmd_drift(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        hot = brain.cortex.get_hot_nodes(3)
        cands = [n for n in brain.cortex.nodes.values() if 0.01<n.activation<0.15 and n.experience>0.3 and not n.archived]
        mid_syn = [s for s in brain.cortex.synapses.values() if 0.2<s.weight<0.5 and s.fire_count>1]
        archived = sum(1 for n in brain.cortex.nodes.values() if n.archived)
        await update.message.reply_text(
            f"🎲 СТОХАСТИЧЕСКИЙ ДРЕЙФ\n{'═'*24}\n"
            f"Exploration prob: {DRIFT_EXPLORE_PROB:.0%}\nНовых кандидатов: {len(cands)}\n"
            f"Периферийных синапсов: {len(mid_syn)}\nАрхивных: {archived}\n"
            f"Горячие: {', '.join(n.label for n in hot)}")

    async def cmd_timeline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        hours = int(context.args[0]) if context.args and context.args[0].isdigit() else 24
        await update.message.reply_text(f"🕰️ За {hours}ч:\n{brain.temporal.get_timeline_summary(hours)}")

    async def cmd_clusters(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        top = sorted(brain.clusters.clusters.values(), key=lambda c:c.access_count, reverse=True)[:8]
        if not top: await update.message.reply_text("Кластеры не сформированы."); return
        lines = ["🗂 КЛАСТЕРЫ\n"]
        for c in top:
            lines.append(f"📌 [{c.name}] ({c.access_count})")
            lines.append(f"   {', '.join(c.members[:7])}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        if not brain.meta.insights: await update.message.reply_text("Инсайтов нет."); return
        lines = ["💡 ИНСАЙТЫ\n"]
        for i, ins in enumerate(brain.meta.insights[-12:], 1): lines.append(f"{i}. {ins}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_l1(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        query = " ".join(context.args) if context.args else "общий контекст"
        ctx   = brain.l1.get_context(query)
        if not ctx: await update.message.reply_text("L1 пуста."); return
        lines = [f"🎯 L1 для «{query}»:"]
        for m in ctx:
            role = "👤" if m['role']=='user' else "🤖"
            lines.append(f"{role} {m['content'][:100]}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_temp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        lines = ["🌡️ ТЕМПЕРАТУРНЫЕ ПРОФИЛИ\n"]
        for cluster, base in TemperatureAdapter.CLUSTER_TEMPS.items():
            adj  = brain.temp_adapter.cluster_adjustments.get(cluster, 0.0)
            real = round(base+adj, 2)
            lines.append(f"  {cluster:<12} base:{base} → real:{real}" + (f" ({adj:+.3f})" if adj else ""))
        await update.message.reply_text("\n".join(lines))

    async def cmd_wipe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        if uid in self.users:
            d = self.users.pop(uid).dir
            if os.path.exists(d): shutil.rmtree(d)
            await update.message.reply_text("🧠 Полная очистка.")
        else: await update.message.reply_text("Пользователь не найден.")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 AGI APPROXIMATION v23.0\n\n"
            "Когнитивный ритм (5 фаз) + 6 AGI-модулей:\n"
            "Цели · Активное обучение · Теория разума\n"
            "Планировщик · Инициативность · Самомодификация\n\n"
            "📌 ВСЕ КОМАНДЫ:\n"
            "/stats           — полная статистика\n"
            "/hotmap          — карта нейронов 🔥\n"
            "/profile         — профиль 👤\n"
            "/mood [дней]     — эмоции 💚\n"
            "/predict <тема>  — предсказание 🔮\n"
            "/drift           — дрейф 🎲\n"
            "/chain <слово>   — цепочки причин\n"
            "/assoc <слово>   — ассоциации\n"
            "/clusters        — кластеры\n"
            "/insights        — инсайты\n"
            "/timeline [ч]    — хронология\n"
            "/l1 [запрос]     — умная L1 🎯\n"
            "/temp            — температуры 🌡️\n"
            "/pulse [n]       — ритм мышления 🥁\n"
            "/phase <ФАЗА>    — детали фазы\n"
            "\n🆕 v23 AGI-КОМАНДЫ:\n"
            "/goals           — цели и прогресс 🎯\n"
            "/hypotheses      — гипотезы и пробелы 🧪\n"
            "/mindmodel       — модель разума пользователя 🪞\n"
            "/plan            — активный план задачи 📋\n"
            "/selfmodel       — стратегии поведения 🔄\n"
            "/proactive       — инициативность ⚡\n"
            "/wipe            — очистка памяти\n\n"
            "💡 v23 работает автоматически:\n"
            "• Определяет ваши цели из диалога\n"
            "• Замечает заблуждения и противоречия\n"
            "• Адаптирует объяснения под ваш уровень\n"
            "• Иногда сам инициирует наблюдения\n"
            "• Учится на ошибках и улучшает стратегии"
        )

    async def shutdown(self):
        print("\n💾 Финальное сохранение...")
        for b in self.users.values(): b.save_all()
        await self.llm.close(); print("✅ Остановлено")


# ══════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════
async def main():
    print("🚀 AGI APPROXIMATION v23.0 STARTING...")
    print("🎯 GoalSystem | 🧪 ActiveLearning | 🪞 TheoryOfMind")
    print("📋 TaskPlanner | ⚡ ProactiveBehavior | 🔄 SelfModification")
    if not TELEGRAM_TOKEN:
        print("❌ Нет TELEGRAM_TOKEN в .env"); return

    bot = HybridBot()
    bot.reflector = ReflectionEngine(bot.llm)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    for cmd, handler in [
        ("stats",       bot.cmd_stats),
        ("hotmap",      bot.cmd_hotmap),
        ("profile",     bot.cmd_profile),
        ("mood",        bot.cmd_mood),
        ("predict",     bot.cmd_predict),
        ("drift",       bot.cmd_drift),
        ("timeline",    bot.cmd_timeline),
        ("chain",       bot.cmd_chain),
        ("assoc",       bot.cmd_assoc),
        ("clusters",    bot.cmd_clusters),
        ("insights",    bot.cmd_insights),
        ("l1",          bot.cmd_l1),
        ("temp",        bot.cmd_temp),
        ("wipe",        bot.cmd_wipe),
        ("help",        bot.cmd_help),
        ("pulse",       bot.cmd_pulse),
        ("phase",       bot.cmd_phase),
        # v23 AGI
        ("goals",       bot.cmd_goals),
        ("hypotheses",  bot.cmd_hypotheses),
        ("mindmodel",   bot.cmd_mindmodel),
        ("plan",        bot.cmd_plan),
        ("selfmodel",   bot.cmd_selfmodel),
        ("proactive",   bot.cmd_proactive),
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    try:
        await app.initialize(); await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print("✅ AGI v23 ГОТОВ 🧠🎯🧪🪞📋⚡🔄")
        while not bot.stop_flag: await asyncio.sleep(1)
    except KeyboardInterrupt: print("\n🛑 Остановка")
    finally:
        await app.updater.stop(); await app.stop()
        await app.shutdown(); await bot.shutdown()


if __name__ == "__main__":
    try: asyncio.run(main())
    except Exception as e: print(f"❌ Crash: {e}"); traceback.print_exc()