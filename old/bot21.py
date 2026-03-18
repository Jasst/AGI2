#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID NEURAL BRAIN v22.0 — COGNITIVE RHYTHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Новое в v22 (когнитивный ритм / CognitivePulse):

  🥁 Ритм мышления (CognitivePulse):
       • 5 обязательных фаз на каждое сообщение:
         PERCEIVE → UNDERSTAND → REFLECT → SYNTHESIZE → RESPOND
       • Каждая фаза — отдельный mini-LLM pass с ограниченным контекстом
       • Результат каждой фазы передаётся в следующую (цепочка мысли)
       • Финальный ответ строится из синтезированной цепочки
       • Фазы можно "пропустить" если LLM вернул пустоту (graceful degrade)

  🧩 Фазы подробно:
       PERCEIVE   — "Что именно спрашивает пользователь?"
                    Извлекает явный и скрытый смысл запроса
       UNDERSTAND — "Что я об этом знаю?" + RAG + L1 + L3
                    Собирает релевантный контекст из памяти
       REFLECT    — "Есть ли противоречие? Что я думаю об этом?"
                    Критическая пауза перед ответом
       SYNTHESIZE — "Как лучше всего ответить с учётом всего выше?"
                    Строит план ответа из фаз 1-3
       RESPOND    — Финальный ответ пользователю из синтеза

  ⏱️ Ритмический метроном (RhythmMetronome):
       • Каждая фаза получает "бюджет токенов"
       • Слишком быстрый ответ (без фаз) → штраф к уверенности
       • Поддерживает async: фазы выполняются последовательно
       • Логирует "пульс" каждого диалога

  📊 Журнал пульса (PulseJournal):
       • Сохраняет историю фаз по диалогу
       • Команда /pulse — просмотр последнего ритма мышления
       • Отображает какая фаза "застряла" или дала инсайт

Сохранено из v21: стохастический дрейф, BM25 RAG,
  трёхуровневая память L1/L2/L3, адаптивная температура,
  ChronoLayer, кластеры, MetaLearner, профиль, эмоциональная дуга,
  умная L1, эмоциональный RAG, активное забывание,
  двухуровневая рефлексия, фидбек температуры, Марков 2-го порядка,
  глубокие причинные цепочки.
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

BASE_DIR   = "hybrid_brain_v22"
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
DRIFT_EXPLORE_PROB  = 0.08
DRIFT_NOISE_SCALE   = 0.05
DRIFT_ASSOC_EVERY   = 7

# Адаптивная температура
TEMP_TECHNICAL       = 0.30
TEMP_EMOTIONAL       = 0.85
TEMP_CREATIVE        = 0.90
TEMP_DEFAULT         = 0.65
TEMP_FEEDBACK_STEP   = 0.05

# Рефлексия
REFLECTION_FAST_EVERY = 10
REFLECTION_DEEP_EVERY = 50

# Причинные цепочки
CAUSAL_MAX_DEPTH    = 8
CAUSAL_MIN_STRENGTH = 0.15

# ══════════════════════════════════════════
# v22: КОГНИТИВНЫЙ РИТМ — КОНФИГ
# ══════════════════════════════════════════
PULSE_ENABLED         = True      # Включить ритм мышления
PULSE_PHASE_MAX_TOKENS = 300      # Макс токенов на промежуточную фазу
PULSE_FINAL_MAX_TOKENS = 1500     # Макс токенов на финальный ответ
PULSE_SKIP_ON_SHORT   = True      # Пропускать фазы для коротких запросов (<10 слов)
PULSE_SHORT_THRESHOLD  = 10       # Порог "короткого" запроса (слов)
PULSE_LOG_MAX          = 20       # Максимум записей в журнале пульса


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
        chosen.activation = min(0.4, chosen.activation + 0.25 + random.uniform(0, 0.1))
        chosen.last_active = datetime.now().isoformat()
        return chosen.label

    def add_noise(self, activation: float) -> float:
        noise = random.gauss(0, DRIFT_NOISE_SCALE)
        return max(0.0, min(1.0, activation + noise))

    def get_wild_association(self, nodes: Dict[str, Any],
                              synapses: Dict[str, Any]) -> Optional[str]:
        self._step_counter += 1
        if self._step_counter % DRIFT_ASSOC_EVERY != 0:
            return None
        mid_synapses = [
            s for s in synapses.values()
            if 0.2 < s.weight < 0.5 and s.fire_count > 1
        ]
        if not mid_synapses:
            return None
        chosen = random.choice(mid_synapses)
        src_node = nodes.get(chosen.source)
        tgt_node = nodes.get(chosen.target)
        if src_node and tgt_node:
            return f"{src_node.label} ↔ {tgt_node.label}"
        return None

    def describe_drift(self, explored: Optional[str], wild: Optional[str]) -> str:
        parts = []
        if explored:
            parts.append(f"[Случайное воспоминание: {explored}]")
        if wild:
            parts.append(f"[Периферийная связь: {wild}]")
        return " ".join(parts)


# ══════════════════════════════════════════
# АДАПТИВНАЯ ТЕМПЕРАТУРА + ФИДБЕК-ПЕТЛЯ
# ══════════════════════════════════════════
class TemperatureAdapter:
    CLUSTER_TEMPS = {
        'технологии': TEMP_TECHNICAL,
        'наука':      TEMP_TECHNICAL,
        'работа':     TEMP_TECHNICAL,
        'учёба':      0.45,
        'деньги':     0.40,
        'кино':       0.70,
        'отдых':      0.75,
        'общение':    0.75,
        'эмоции':     TEMP_EMOTIONAL,
        'здоровье':   0.55,
        'спорт':      0.65,
        'еда':        0.70,
    }
    CONFUSION_SIGNALS = [
        'не понял', 'не понимаю', 'что ты имеешь', 'объясни',
        'поясни', 'не то', 'неправильно', 'не так', 'ещё раз',
        'что значит', 'можешь подробнее', 'я не'
    ]
    SATISFACTION_SIGNALS = [
        'спасибо', 'понял', 'отлично', 'классно', 'круто',
        'понятно', 'всё ясно', 'именно', 'точно', 'супер', 'ок'
    ]

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.cluster_adjustments: Dict[str, float] = {}
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self.cluster_adjustments = data.get('adjustments', {})

    def save(self):
        FileManager.safe_save_json(self.save_path, {'adjustments': self.cluster_adjustments})

    def feedback(self, text: str, cluster: Optional[str]):
        if not cluster:
            return
        tl = text.lower()
        confused  = sum(1 for s in self.CONFUSION_SIGNALS if s in tl)
        satisfied = sum(1 for s in self.SATISFACTION_SIGNALS if s in tl)
        adj = self.cluster_adjustments.get(cluster, 0.0)
        if confused > 0:
            adj = max(-0.25, adj - TEMP_FEEDBACK_STEP * confused)
        if satisfied > 0:
            adj = min(0.20, adj + TEMP_FEEDBACK_STEP * 0.5)
        self.cluster_adjustments[cluster] = round(adj, 3)

    def get(self, cluster_name: Optional[str], emotion: str, text: str) -> float:
        base = self.CLUSTER_TEMPS.get(cluster_name, TEMP_DEFAULT)
        if cluster_name:
            base += self.cluster_adjustments.get(cluster_name, 0.0)
        if emotion == 'negative':
            base = min(TEMP_EMOTIONAL, base + 0.15)
        elif emotion == 'positive':
            base = min(TEMP_CREATIVE, base + 0.05)
        tl = text.lower()
        if any(w in tl for w in ['напиши код', 'функция', 'class', 'def ', 'import ']):
            base = min(base, TEMP_TECHNICAL)
        elif any(w in tl for w in ['придумай', 'представь', 'фантазия', 'идея', 'вариант']):
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
            'role': role,
            'content': content,
            'time': datetime.now().isoformat(),
            'keywords': TextUtils.extract_keywords(content)
        })
        if len(self.history) > L1_MAX_HISTORY:
            self.history = self.history[-L1_MAX_HISTORY:]

    def get_context(self, query: str) -> List[Dict]:
        if not self.history:
            return []
        recent    = self.history[-L1_RECENT_COUNT:]
        recent_set = set(id(m) for m in recent)
        query_kw   = set(TextUtils.extract_keywords(query))
        candidates = []
        for msg in self.history[:-L1_RECENT_COUNT]:
            msg_kw = set(msg.get('keywords', []))
            if not query_kw or not msg_kw:
                continue
            overlap = len(query_kw & msg_kw) / max(len(query_kw | msg_kw), 1)
            if overlap > 0.1:
                candidates.append((overlap, msg))
        candidates.sort(key=lambda x: -x[0])
        relevant   = [m for _, m in candidates[:L1_RELEVANT_COUNT]]
        result_set: Set[int] = set()
        result: List[Dict] = []
        for msg in self.history:
            mid = id(msg)
            if mid in recent_set or msg in relevant:
                if mid not in result_set:
                    result.append(msg)
                    result_set.add(mid)
        return result

    def format_for_prompt(self, query: str) -> str:
        ctx = self.get_context(query)
        if not ctx:
            return ""
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
    content:      str
    score:        float
    timestamp:    str
    rel_time:     str
    cluster_name: Optional[str] = None
    emotion:      Optional[str] = None


class RAGEngineV2:
    @staticmethod
    def _compute_idf(query_terms: List[str], all_events: List[Any]) -> Dict[str, float]:
        N = max(len(all_events), 1)
        df: Dict[str, int] = {}
        for event in all_events:
            doc_terms = set(event.keywords)
            for term in query_terms:
                if term in doc_terms:
                    df[term] = df.get(term, 0) + 1
        idf = {}
        for term in query_terms:
            df_t = df.get(term, 0)
            idf[term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        return idf

    @staticmethod
    def _bm25_score(query_terms: List[str], doc_terms: List[str],
                    idf: Dict[str, float], avg_len: float) -> float:
        score = 0.0
        dl    = len(doc_terms)
        term_freq = Counter(doc_terms)
        for term in query_terms:
            if term not in term_freq:
                continue
            tf     = term_freq[term]
            idf_t  = idf.get(term, 0.1)
            tf_norm = (tf * (RAG_BM25_K1 + 1)) / \
                      (tf + RAG_BM25_K1 * (1 - RAG_BM25_B + RAG_BM25_B * dl / max(avg_len, 1)))
            score += idf_t * tf_norm
        return score

    @staticmethod
    def search(query: str, events: List[Any], top_k: int = RAG_TOP_K,
               min_score: float = RAG_MIN_SCORE, now: datetime = None,
               current_emotion: str = 'neutral') -> List[RAGResult]:
        if not events or not query:
            return []
        if now is None:
            now = datetime.now()
        query_terms = TextUtils.tokenize(query)
        if not query_terms:
            return []
        idf     = RAGEngineV2._compute_idf(query_terms, events)
        avg_len = sum(len(e.keywords) for e in events) / len(events)
        scored  = []
        for event in events:
            doc_terms = TextUtils.tokenize(event.content)
            bm25      = RAGEngineV2._bm25_score(query_terms, doc_terms, idf, avg_len)
            if bm25 < 0.01:
                continue
            age_h   = max((now - datetime.fromisoformat(event.timestamp)).total_seconds() / 3600, 0.1)
            recency = 1.0 / (1.0 + math.log1p(age_h / 24))
            emotion_boost = 0.0
            ev_emotion    = getattr(event, 'user_emotion', None)
            if ev_emotion and ev_emotion == current_emotion and current_emotion != 'neutral':
                emotion_boost = RAG_EMOTION_BOOST
            score = bm25 * 0.60 + recency * 0.2 + event.importance * 0.12 + emotion_boost * 0.08
            if score >= min_score:
                scored.append((score, event))
        scored.sort(key=lambda x: -x[0])
        results: List[RAGResult] = []
        cluster_counts: Dict[str, int] = {}
        for score, event in scored:
            cn = event.cluster_name or "_none"
            if cluster_counts.get(cn, 0) >= 2:
                continue
            cluster_counts[cn] = cluster_counts.get(cn, 0) + 1
            rel = TimeEncoder.describe(datetime.fromisoformat(event.timestamp), now)
            results.append(RAGResult(
                content      = event.content,
                score        = round(score, 3),
                timestamp    = event.timestamp,
                rel_time     = rel,
                cluster_name = event.cluster_name,
                emotion      = getattr(event, 'user_emotion', None)
            ))
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def format_for_prompt(results: List[RAGResult]) -> str:
        if not results:
            return ""
        lines = ["=== РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ (L2) ==="]
        for r in results:
            cluster    = f"[{r.cluster_name}] " if r.cluster_name else ""
            emotion_tag = f" [{r.emotion}]" if r.emotion and r.emotion != 'neutral' else ""
            lines.append(f"• [{r.rel_time}]{emotion_tag} {cluster}{r.content[:130]}")
        lines.append("=== КОНЕЦ ВОСПОМИНАНИЙ ===")
        return "\n".join(lines)


# ══════════════════════════════════════════
# ТРЁХУРОВНЕВАЯ ПАМЯТЬ — L3
# ══════════════════════════════════════════
class SemanticMemoryL3:
    SEMANTIC_UPDATE_EVERY = 25

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.facts: List[str] = []
        self.topic_freq: Dict[str, int] = {}
        self.time_patterns: Dict[str, int] = {}
        self._counter = 0
        self._load()

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
        period = ('утро' if 6 <= hour < 12 else
                  'день' if 12 <= hour < 18 else
                  'вечер' if 18 <= hour < 23 else 'ночь')
        self.time_patterns[period] = self.time_patterns.get(period, 0) + 1
        if self._counter % self.SEMANTIC_UPDATE_EVERY == 0:
            self._rebuild_facts()

    def force_rebuild(self):
        self._rebuild_facts()

    def _rebuild_facts(self):
        new_facts = []
        if self.topic_freq:
            top_topics  = sorted(self.topic_freq.items(), key=lambda x: -x[1])[:3]
            topics_str  = ', '.join(f"{t}" for t, _ in top_topics)
            new_facts.append(f"Часто обсуждает: {topics_str}")
        if self.time_patterns:
            top_period = max(self.time_patterns.items(), key=lambda x: x[1])
            new_facts.append(f"Чаще всего активен: {top_period[0]}")
        if self.topic_freq:
            dominant = max(self.topic_freq.items(), key=lambda x: x[1])
            if dominant[1] >= 5:
                new_facts.append(f"Основной интерес: {dominant[0]}")
        self.facts = new_facts

    def get_prompt_block(self) -> str:
        if not self.facts:
            return ""
        return "=== ДОЛГОСРОЧНЫЕ ПАТТЕРНЫ (L3) ===\n" + "\n".join(f"  • {f}" for f in self.facts)


# ══════════════════════════════════════════
# ВРЕМЕННОЙ КОДИРОВЩИК
# ══════════════════════════════════════════
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
        'утро':   {'hour_range': (6, 12)},  'утром':   {'hour_range': (6, 12)},
        'день':   {'hour_range': (12, 18)}, 'днём':    {'hour_range': (12, 18)},
        'вечер':  {'hour_range': (18, 24)}, 'вечером': {'hour_range': (18, 24)},
        'ночь':   {'hour_range': (0, 6)},   'ночью':   {'hour_range': (0, 6)},
    }

    @staticmethod
    def parse_time_ref(text: str, now: datetime) -> Optional[Dict]:
        tl = text.lower()
        for phrase, offset in TimeEncoder.TIME_WORDS.items():
            if phrase in tl:
                if 'hour_range' in offset:
                    return {'type': 'time_of_day', 'range': offset['hour_range'], 'date': now.date()}
                delta  = timedelta(**{k: v for k, v in offset.items()})
                target = now + delta
                return {'type': 'date_range',
                        'start': target.replace(hour=0, minute=0, second=0),
                        'end':   target.replace(hour=23, minute=59, second=59)}
        m = re.search(r'(\d+)\s*(час|часа|часов|день|дня|дней)\s*назад', tl)
        if m:
            val, unit = int(m.group(1)), m.group(2)
            d = timedelta(hours=val) if 'час' in unit else timedelta(days=val)
            return {'type': 'date_range', 'start': now - d, 'end': now}
        return None

    @staticmethod
    def describe(event_time: datetime, now: datetime) -> str:
        d = now - event_time; s = d.total_seconds()
        if s < 60:    return "только что"
        if s < 3600:  return f"{int(s / 60)} мин. назад"
        if s < 86400: return f"{int(s / 3600)} ч. назад"
        if d.days == 1: return "вчера"
        if d.days < 7:  return f"{d.days} дн. назад"
        return event_time.strftime("%d.%m.%Y")


# ══════════════════════════════════════════
# ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ
# ══════════════════════════════════════════
@dataclass
class UserProfile:
    name:                str       = ""
    profession:          str       = ""
    interests:           List[str] = field(default_factory=list)
    communication_style: str       = "neutral"
    disliked_topics:     List[str] = field(default_factory=list)
    known_facts:         List[str] = field(default_factory=list)
    last_updated:        str       = field(default_factory=lambda: datetime.now().isoformat())


class ProfileManager:
    FACT_PATTERNS = [
        (r'меня зовут ([А-ЯЁа-яёA-Za-z]+)',          'name'),
        (r'я ([А-ЯЁа-яё]+(?:ист|ер|ор|ёр|ник|щик|лог|граф))', 'profession'),
        (r'работаю ([А-ЯЁа-яё\s]+?)(?:\.|,|$)',       'profession'),
        (r'я люблю ([А-ЯЁа-яё\s]+?)(?:\.|,|$)',       'interest'),
        (r'мне нравится ([А-ЯЁа-яё\s]+?)(?:\.|,|$)',  'interest'),
        (r'я увлекаюсь ([А-ЯЁа-яё\s]+?)(?:\.|,|$)',   'interest'),
        (r'не люблю ([А-ЯЁа-яё\s]+?)(?:\.|,|$)',      'dislike'),
        (r'живу в ([А-ЯЁа-яёA-Za-z\s]+?)(?:\.|,|$)',  'fact'),
        (r'мне (\d{1,2}) (?:лет|год)',                 'fact'),
    ]
    STYLE_CLUES = {
        'technical': ['код', 'функция', 'алгоритм', 'python', 'class', 'debug', 'api'],
        'formal':    ['уважаемый', 'пожалуйста', 'благодарю', 'прошу', 'позвольте'],
        'casual':    ['норм', 'окей', 'ок', 'лол', 'кстати', 'слушай', 'короче'],
        'emotional': ['грустно', 'радостно', 'обидно', 'восторг', 'боюсь', 'тревожно'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.profile   = UserProfile()
        self._load()

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
                self.profile.name = value.capitalize()
                new_facts.append(f"имя: {self.profile.name}")
            elif fact_type == 'profession' and not self.profile.profession:
                self.profile.profession = value
                new_facts.append(f"профессия: {value}")
            elif fact_type == 'interest' and value not in self.profile.interests:
                self.profile.interests.append(value)
                if len(self.profile.interests) > 30:
                    self.profile.interests = self.profile.interests[-30:]
                new_facts.append(f"интерес: {value}")
            elif fact_type == 'dislike' and value not in self.profile.disliked_topics:
                self.profile.disliked_topics.append(value)
                new_facts.append(f"не нравится: {value}")
            elif fact_type == 'fact':
                fs = f"{value}"
                if fs not in self.profile.known_facts:
                    self.profile.known_facts.append(fs)
                    if len(self.profile.known_facts) > 20:
                        self.profile.known_facts = self.profile.known_facts[-20:]
                    new_facts.append(f"факт: {value}")
        for style, clues in self.STYLE_CLUES.items():
            if sum(1 for c in clues if c in tl) >= 2:
                self.profile.communication_style = style
                break
        if new_facts:
            self.profile.last_updated = datetime.now().isoformat()
        return new_facts

    def get_prompt_block(self) -> str:
        p = self.profile; lines = []
        if p.name:            lines.append(f"Имя: {p.name}")
        if p.profession:      lines.append(f"Профессия: {p.profession}")
        if p.interests:       lines.append(f"Интересы: {', '.join(p.interests[:5])}")
        if p.disliked_topics: lines.append(f"Не интересует: {', '.join(p.disliked_topics[:3])}")
        if p.known_facts:     lines.append(f"Факты: {'; '.join(p.known_facts[:4])}")
        style_hint = {
            'technical': "Говори технически точно.",
            'formal':    "Общайся формально и уважительно.",
            'casual':    "Общайся непринуждённо.",
            'emotional': "Будь эмпатичным."
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
        'positive': ['рад', 'хорошо', 'отлично', 'супер', 'люблю', 'нравится', 'кайф',
                     'счастье', 'весело', 'здорово', 'классно', 'круто', 'успех'],
        'negative': ['грустно', 'плохо', 'устал', 'злой', 'проблема', 'страшно', 'боюсь',
                     'тревога', 'обидно', 'надоело', 'сложно', 'неудача'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.history: List[EmotionPoint] = []
        self._session: List[float] = []
        self._load()

    def _load(self):
        for item in FileManager.safe_load_json(self.save_path, []):
            self.history.append(EmotionPoint(**item))

    def save(self):
        FileManager.safe_save_json(self.save_path, [asdict(p) for p in self.history[-200:]])

    def detect(self, text: str) -> Tuple[str, float]:
        tl  = text.lower()
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
        if len(self.history) > 500:
            self.history = self.history[-500:]
        return emotion, score

    def get_session_mood(self) -> Tuple[str, str]:
        if not self._session: return 'neutral', ""
        avg = sum(self._session) / len(self._session)
        if avg > 0.3:  return 'positive', "Пользователь в хорошем настроении. Поддерживай позитивный тон."
        if avg < -0.3: return 'negative', "Пользователь расстроен или устал. Будь особенно мягким и поддерживающим."
        return 'neutral', ""

    def get_trend(self, last_n: int = 10) -> Dict:
        recent = self.history[-last_n:]
        if not recent:
            return {'trend': 'unknown', 'avg': 0.0, 'direction': 'stable', 'count': 0}
        scores = [p.score for p in recent]
        avg    = sum(scores) / len(scores)
        mid    = len(scores) // 2
        fh = sum(scores[:mid]) / max(mid, 1)
        sh = sum(scores[mid:]) / max(len(scores) - mid, 1)
        direction = 'rising' if sh > fh + 0.1 else ('falling' if sh < fh - 0.1 else 'stable')
        trend = 'positive' if avg > 0.15 else ('negative' if avg < -0.15 else 'neutral')
        return {'trend': trend, 'avg': round(avg, 2), 'direction': direction, 'count': len(recent)}

    def get_history_summary(self, days: int = 7) -> str:
        cutoff = datetime.now() - timedelta(days=days)
        recent = [p for p in self.history if datetime.fromisoformat(p.timestamp) >= cutoff]
        if not recent: return "Нет данных."
        pos = sum(1 for p in recent if p.emotion == 'positive')
        neg = sum(1 for p in recent if p.emotion == 'negative')
        t   = self.get_trend()
        return (f"За {days} дн.: 😊{pos} / 😐{len(recent) - pos - neg} / 😟{neg}\n"
                f"Score: {t['avg']:+.2f} | Тренд: {t['direction']}")


# ══════════════════════════════════════════
# ПРЕДСКАЗАНИЕ ТЕМАТИКИ — МАРКОВ 2-го ПОРЯДКА
# ══════════════════════════════════════════
class TopicPredictor:
    def __init__(self, save_path: str):
        self.save_path    = save_path
        self.transitions:  Dict[str, Dict[str, int]] = {}
        self.transitions2: Dict[str, Dict[str, int]] = {}
        self._prev: Optional[str] = None
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, {})
        self.transitions  = data.get('t1', {})
        self.transitions2 = data.get('t2', {})

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            't1': self.transitions, 't2': self.transitions2
        })

    def record_transition(self, frm: str, to: str):
        if not frm or not to or frm == to: return
        if frm not in self.transitions: self.transitions[frm] = {}
        self.transitions[frm][to] = self.transitions[frm].get(to, 0) + 1
        if self._prev and self._prev != frm:
            key2 = f"{self._prev}|{frm}"
            if key2 not in self.transitions2: self.transitions2[key2] = {}
            self.transitions2[key2][to] = self.transitions2[key2].get(to, 0) + 1
        self._prev = frm

    def predict_next(self, current: str, prev: str = None, top_n: int = 2) -> List[Tuple[str, float]]:
        if prev:
            key2 = f"{prev}|{current}"
            if key2 in self.transitions2:
                counts = self.transitions2[key2]
                total  = sum(counts.values())
                return sorted([(t, c / total) for t, c in counts.items()],
                               key=lambda x: -x[1])[:top_n]
        if current not in self.transitions: return []
        counts = self.transitions[current]; total = sum(counts.values())
        return sorted([(t, c / total) for t, c in counts.items()],
                       key=lambda x: -x[1])[:top_n]

    def get_hint(self, cluster: str, prev_cluster: str = None) -> str:
        preds = self.predict_next(cluster, prev_cluster)
        if not preds: return ""
        best, prob = preds[0]
        order = "2-го порядка" if prev_cluster else ""
        if prob > 0.4:
            return f"Вероятно, следующий вопрос о «{best}» ({prob:.0%}){' '+order if order else ''}."
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
        self.strength = min(1.0, self.strength + d)
        self.evidence += 1
        self.last_seen = datetime.now().isoformat()

    def decay(self, f: float = 0.995):
        self.strength = max(0.0, self.strength * f)


class LogicalChainEngine:
    PATTERNS = [
        (r'(\w{3,})\s+потому\s+что\s+(\w{3,})',   'effect', 'cause',  'positive'),
        (r'(\w{3,})\s+так\s+как\s+(\w{3,})',       'effect', 'cause',  'positive'),
        (r'из-за\s+(\w{3,})\s+(\w{3,})',           'cause',  'effect', 'positive'),
        (r'(\w{3,})\s+приводит\s+к\s+(\w{3,})',    'cause',  'effect', 'positive'),
        (r'(\w{3,})\s+вызывает\s+(\w{3,})',        'cause',  'effect', 'positive'),
        (r'(\w{3,})\s+помогает\s+(\w{3,})',        'cause',  'effect', 'positive'),
        (r'(\w{3,})\s+мешает\s+(\w{3,})',          'cause',  'effect', 'negative'),
        (r'если\s+(\w{3,}).{0,15}то\s+(\w{3,})',  'cause',  'effect', 'positive'),
        (r'(\w{3,})\s+даёт\s+(\w{3,})',            'cause',  'effect', 'positive'),
        (r'(\w{3,})\s+влияет\s+на\s+(\w{3,})',    'cause',  'effect', 'positive'),
        (r'без\s+(\w{3,})\s+нет\s+(\w{3,})',       'cause',  'effect', 'negative'),
        (r'(\w{3,})\s+означает\s+(\w{3,})',        'cause',  'effect', 'positive'),
    ]

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.links: Dict[str, CausalLink] = {}
        self._load()

    def _key(self, c, e): return f"{c}→{e}"

    def _load(self):
        for k, v in FileManager.safe_load_json(self.save_path, {}).items():
            v.setdefault('link_type', 'positive')
            self.links[k] = CausalLink(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path, {k: asdict(v) for k, v in self.links.items()})

    def extract_from_text(self, text: str) -> List[CausalLink]:
        found, tl = [], text.lower()
        for pat, r1, r2, lt in self.PATTERNS:
            for m in re.finditer(pat, tl):
                w1, w2 = m.group(1), m.group(2)
                if w1 in TextUtils.STOP_WORDS or w2 in TextUtils.STOP_WORDS: continue
                c = w1 if r1 == 'cause' else w2
                e = w2 if r2 == 'effect' else w1
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
                if l.cause == cur and l.effect not in vis:
                    s = l.strength * l.evidence
                    if s > bs: best, bs = l.effect, s
            if not best or bs < 0.1: break
            chain.append(best); vis.add(best); cur = best
        return chain

    def backward_chain(self, effect: str, depth: int = 3) -> Dict[str, Any]:
        def _recurse(node: str, visited: Set[str], d: int) -> Dict:
            causes = []
            if d > 0:
                for l in sorted(self.links.values(), key=lambda x: -x.strength):
                    if l.effect == node and l.cause not in visited and l.strength >= CAUSAL_MIN_STRENGTH:
                        visited_new = visited | {l.cause}
                        causes.append({
                            'node': l.cause, 'strength': round(l.strength, 2),
                            'type': l.link_type,
                            'causes': _recurse(l.cause, visited_new, d - 1)['causes']
                        })
            return {'node': node, 'causes': causes[:3]}
        return _recurse(effect, {effect}, depth)

    def forward_chain(self, cause: str, depth: int = 3) -> Dict[str, Any]:
        def _recurse(node: str, visited: Set[str], d: int) -> Dict:
            effects = []
            if d > 0:
                for l in sorted(self.links.values(), key=lambda x: -x.strength):
                    if l.cause == node and l.effect not in visited and l.strength >= CAUSAL_MIN_STRENGTH:
                        visited_new = visited | {l.effect}
                        effects.append({
                            'node': l.effect, 'strength': round(l.strength, 2),
                            'type': l.link_type,
                            'effects': _recurse(l.effect, visited_new, d - 1)['effects']
                        })
            return {'node': node, 'effects': effects[:3]}
        return _recurse(cause, {cause}, depth)

    def counterfactual(self, concept: str) -> List[str]:
        return [
            l.effect for l in self.links.values()
            if l.cause == concept and l.link_type == 'positive' and l.strength >= CAUSAL_MIN_STRENGTH
        ][:5]

    def format_causal_for_prompt(self, keyword: str, query: str) -> str:
        tl = query.lower(); lines = []
        if any(w in tl for w in ['почему', 'причина', 'из-за чего', 'отчего']):
            tree = self.backward_chain(keyword, depth=2)
            if tree['causes']:
                lines.append(f"[Причины «{keyword}»]")
                for c in tree['causes']:
                    sub = " ← ".join([cc['node'] for cc in c.get('causes', [])])
                    lines.append(f"  ← {c['node']} ({c['strength']:.2f})" +
                                 (f" ← {sub}" if sub else ""))
        elif any(w in tl for w in ['что будет', 'что если', 'последствия', 'приведёт']):
            tree = self.forward_chain(keyword, depth=2)
            if tree['effects']:
                lines.append(f"[Следствия «{keyword}»]")
                for e in tree['effects']:
                    sub = " → ".join([ee['node'] for ee in e.get('effects', [])])
                    lines.append(f"  → {e['node']} ({e['strength']:.2f})" +
                                 (f" → {sub}" if sub else ""))
        elif any(w in tl for w in ['без', 'если не', 'что если нет']):
            deps = self.counterfactual(keyword)
            if deps:
                lines.append(f"[Без «{keyword}» пострадает]")
                lines.extend(f"  • {d}" for d in deps)
        else:
            chain = self.build_chain(keyword)
            if len(chain) > 1:
                lines.append(f"[Цепочка] {' → '.join(chain)}")
        return "\n".join(lines)

    def find_causes(self, effect: str) -> List[Tuple[str, float]]:
        return sorted([(l.cause, l.strength) for l in self.links.values()
                       if l.effect == effect and l.strength >= CAUSAL_MIN_STRENGTH],
                      key=lambda x: -x[1])

    def get_strong_chains(self, top_n: int = 3) -> List[str]:
        seen, chains = [], []
        for l in sorted(self.links.values(), key=lambda x: x.strength * x.evidence, reverse=True):
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
        'деньги':      ['деньги', 'зарплата', 'бюджет', 'расход', 'доход', 'инвестиции'],
        'наука':       ['наука', 'исследование', 'эксперимент', 'теория', 'физика', 'химия'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.clusters: Dict[str, SemanticCluster] = {}
        self._load()
        if not self.clusters: self._init()

    def _load(self):
        for k, v in FileManager.safe_load_json(self.save_path, {}).items():
            self.clusters[k] = SemanticCluster(**v)

    def save(self):
        FileManager.safe_save_json(self.save_path, {k: asdict(v) for k, v in self.clusters.items()})

    def _init(self):
        for name, words in self.SEEDS.items():
            cid = hashlib.md5(name.encode()).hexdigest()[:8]
            self.clusters[cid] = SemanticCluster(id=cid, name=name, members=words,
                                                  centroid_keywords=words[:3], strength=0.7)

    def classify(self, keywords: List[str]) -> List[Tuple[str, float]]:
        scores = []
        for cid, c in self.clusters.items():
            hits = sum(1 for kw in keywords if kw in c.members)
            if hits > 0:
                scores.append((c.name, hits / max(len(keywords), 1)))
                c.access_count += 1
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
                    c.strength = min(1.0, c.strength + 0.015); break
            return best
        if len(keywords) >= 2:
            name = hint or keywords[0]
            cid  = hashlib.md5(name.encode()).hexdigest()[:8]
            if cid not in self.clusters:
                self.clusters[cid] = SemanticCluster(id=cid, name=name, members=list(keywords),
                                                      centroid_keywords=keywords[:2], strength=0.3)
            else:
                for kw in keywords:
                    if kw not in self.clusters[cid].members:
                        self.clusters[cid].members.append(kw)
        return None


# ══════════════════════════════════════════
# МЕТАОБУЧЕНИЕ
# ══════════════════════════════════════════
@dataclass
class NeuralGrowthRecord:
    timestamp: str; event_type: str; description: str; impact_score: float = 0.0


class MetaLearner:
    THRESHOLDS = [0, 50, 150, 350, 750, 1500, 3000, 6000]
    NAMES = {1: "Новичок", 2: "Ученик", 3: "Знающий", 4: "Опытный",
             5: "Мастер", 6: "Эксперт", 7: "Гений", 8: "Оракул"}

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.xp = 0; self.level = 1
        self.total_interactions = 0
        self.growth_log = []; self.insights = []
        self._load()

    def _load(self):
        d = FileManager.safe_load_json(self.save_path, {})
        self.xp                 = d.get('xp', 0)
        self.level              = d.get('level', 1)
        self.total_interactions = d.get('total_interactions', 0)
        self.insights           = d.get('insights', [])
        for r in d.get('growth_log', []):
            self.growth_log.append(NeuralGrowthRecord(**r))

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'xp': self.xp, 'level': self.level,
            'total_interactions': self.total_interactions,
            'insights': self.insights,
            'growth_log': [asdict(r) for r in self.growth_log[-100:]]
        })

    def gain_xp(self, amount: int, event: str, desc: str = "") -> Optional[str]:
        self.xp += amount; self.total_interactions += 1
        self.growth_log.append(NeuralGrowthRecord(
            timestamp=datetime.now().isoformat(), event_type=event,
            description=desc, impact_score=amount / 10.0))
        return self._check_level_up()

    def _check_level_up(self) -> Optional[str]:
        for lvl, thr in enumerate(self.THRESHOLDS[1:], start=2):
            if self.xp >= thr and self.level < lvl:
                self.level = lvl
                return f"🆙 УРОВЕНЬ {lvl} — {self.NAMES.get(lvl, '?')}!\n📊 Опыт: {self.xp}"
        return None

    def add_insight(self, s: str):
        if s and s not in self.insights:
            self.insights.append(s)
            if len(self.insights) > 60:
                self.insights = self.insights[-60:]

    def get_insights_prompt(self, n: int = 2) -> str:
        if not self.insights: return ""
        return "📌 Из прошлых сессий:\n" + "".join(f"  • {i}\n" for i in self.insights[-n:])

    def get_status(self) -> str:
        name = self.NAMES.get(self.level, '?')
        nxt  = next((t for t in self.THRESHOLDS if t > self.xp), None)
        sfx  = f" (до след.: {nxt - self.xp} XP)" if nxt else " (MAX)"
        return (f"Уровень {self.level} — {name}{sfx}\n"
                f"Опыт: {self.xp} | Диалогов: {self.total_interactions}\n"
                f"Инсайтов: {len(self.insights)}")


# ══════════════════════════════════════════
# ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ (L2)
# ══════════════════════════════════════════
@dataclass
class EpisodicMemory:
    id: str; timestamp: str; content: str; keywords: List[str]
    temporal_tags: List[str]; user_emotion: Optional[str]
    location_hints: List[str]; sequence_id: int
    cluster_name: Optional[str] = None; importance: float = 0.5


class TemporalMemory:
    def __init__(self, user_id: str, save_dir: str):
        self.file    = os.path.join(save_dir, "episodic_time.json")
        self.events: List[EpisodicMemory] = []
        self.sequence_counter = 0
        self.now = datetime.now()
        data = FileManager.safe_load_json(self.file)
        if data:
            self.sequence_counter = data.get('seq', 0)
            for e in data.get('events', []):
                e.setdefault('cluster_name', None); e.setdefault('importance', 0.5)
                self.events.append(EpisodicMemory(**e))
        print(f"⏳ Эпизодов: {len(self.events)}")

    def add_event(self, content: str, keywords: List[str],
                  cluster_name: str = None, now: datetime = None) -> EpisodicMemory:
        if now is None: now = datetime.now(); self.now = now
        ttags = [tw for tw in ['сегодня','вчера','завтра','утром','вечером','ночью','сейчас']
                 if tw in content.lower()]
        locs  = [l for l in re.findall(r'(?:в|на|из|к)\s+([а-яёa-z]{3,})', content.lower())
                 if l not in {'дом', 'домой', 'город', 'место'}]
        imp = min(1.0, 0.3 + len(content) / 500 + (0.2 if ttags else 0))
        e = EpisodicMemory(
            id=hashlib.md5(f"{content}{now.isoformat()}".encode()).hexdigest()[:12],
            timestamp=now.isoformat(), content=content, keywords=keywords,
            temporal_tags=ttags, user_emotion=self._detect_emotion(content),
            location_hints=locs, sequence_id=self.sequence_counter,
            cluster_name=cluster_name, importance=imp)
        self.events.append(e); self.sequence_counter += 1
        if len(self.events) > 400:
            self.events.sort(key=lambda e: e.importance * 0.5 +
                             datetime.fromisoformat(e.timestamp).timestamp() * 1e-10)
            self.events = self.events[-400:]
            self.events.sort(key=lambda e: e.sequence_id)
        return e

    def _detect_emotion(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ['рад','хорошо','отлично','супер','люблю','нравится']): return 'positive'
        if any(w in t for w in ['грустно','плохо','устал','злой','проблема','боюсь']): return 'negative'
        return None

    def query_by_time(self, time_ref: str, now: datetime = None) -> List[EpisodicMemory]:
        if now is None: now = datetime.now()
        tr = TimeEncoder.parse_time_ref(time_ref, now)
        if not tr: return []
        res = []
        for ev in reversed(self.events):
            et = datetime.fromisoformat(ev.timestamp)
            if tr['type'] == 'date_range':
                if tr['start'] <= et <= tr['end']: res.append(ev)
            elif tr['type'] == 'time_of_day':
                if et.date() == tr['date'] and tr['range'][0] <= et.hour < tr['range'][1]:
                    res.append(ev)
        return res[:10]

    def get_timeline_summary(self, hours: int = 24) -> str:
        now = datetime.now(); cutoff = now - timedelta(hours=hours)
        recent = [e for e in self.events if datetime.fromisoformat(e.timestamp) >= cutoff]
        if not recent: return "За это время ничего не зафиксировано."
        return "\n".join(
            f"[{TimeEncoder.describe(datetime.fromisoformat(e.timestamp), now)}] "
            f"{'[' + e.cluster_name + '] ' if e.cluster_name else ''}{e.content[:90]}"
            for e in recent[-6:])

    def save(self):
        FileManager.safe_save_json(self.file, {
            'seq': self.sequence_counter,
            'events': [asdict(e) for e in self.events]
        })


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
        self.nodes: Dict[str, ChronoNode] = {}
        self._init()

    def _init(self):
        for label, ntype in [
            ('вчера','temporal_ref'), ('сегодня','temporal_ref'), ('завтра','temporal_ref'),
            ('утром','time_of_day'), ('днём','time_of_day'), ('вечером','time_of_day'),
            ('ночью','time_of_day'), ('утро','time_of_day'), ('вечер','time_of_day'),
            ('раньше','sequence'), ('потом','sequence'), ('до','sequence'),
            ('после','sequence'), ('всегда','sequence'), ('никогда','sequence'),
        ]:
            nid = hashlib.md5(f"chrono_{label}".encode()).hexdigest()[:8]
            if nid not in self.nodes:
                self.nodes[nid] = ChronoNode(id=nid, label=label, node_type=ntype)

    def restore_from_saved(self, saved: List[Dict]):
        for nd in saved:
            nid = nd['id']
            if nid in self.nodes:
                raw = nd.get('activation', 0.0)
                la  = nd.get('last_active', datetime.now().isoformat())
                uc  = nd.get('use_count', 0)
                self.nodes[nid].activation = max(ActivationDecay.apply(raw, la, rate=0.08),
                                                  min(0.15, uc * 0.005))
                self.nodes[nid].last_active    = la
                self.nodes[nid].use_count      = uc
                self.nodes[nid].linked_events  = nd.get('linked_events', [])

    def to_save_list(self) -> List[Dict]:
        return [asdict(n) for n in self.nodes.values()]

    def process_temporal_input(self, text: str) -> List[ChronoNode]:
        activated, tl = [], text.lower()
        for n in self.nodes.values():
            if n.label in tl:
                n.activation  = min(1.0, n.activation + 0.8)
                n.last_active = datetime.now().isoformat()
                n.use_count  += 1
                activated.append(n)
        for n in self.nodes.values():
            if n not in activated: n.activation *= 0.88
        return activated

    def get_active_labels(self) -> List[str]:
        return [n.label for n in self.nodes.values() if n.activation > 0.3]

    def link_event(self, event: EpisodicMemory, nodes: List[ChronoNode]):
        for n in nodes:
            if event.id not in n.linked_events:
                n.linked_events.append(event.id)
                if len(n.linked_events) > 60:
                    n.linked_events = n.linked_events[-60:]


# ══════════════════════════════════════════
# НЕЙРО-КОРА — с активным забыванием
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
        self.save_path = os.path.join(save_dir, "cortex_graph.json")
        self.nodes:    Dict[str, NeuroNode]    = {}
        self.synapses: Dict[str, NeuroSynapse] = {}
        self.chrono_layer = ChronoNeuralLayer()
        self.drift        = StochasticDrift()
        data = FileManager.safe_load_json(self.save_path)
        if data and 'nodes' in data: self._load_with_decay(data)
        else:
            self._create_node("Приветствие", "action")
            self._create_node("Помощь", "action")
            self.save()

    def _load_with_decay(self, data: Dict):
        now_iso = datetime.now().isoformat()
        for n in data.get('nodes', []):
            n.setdefault('experience', 0.0)
            n.setdefault('temporal_weight', 1.0)
            n.setdefault('last_active', now_iso)
            n.setdefault('archived', False)
            decayed = ActivationDecay.apply(n.get('activation', 0.0), n['last_active'])
            tonic   = ActivationDecay.tonic(n['experience'])
            if n.get('archived', False):
                n['activation'] = min(ARCHIVE_ACTIVATION_CAP, max(decayed, tonic))
            else:
                n['activation'] = max(decayed, tonic)
                if ActivationDecay.is_archive_candidate(n['last_active']) and n['experience'] < 0.5:
                    n['archived'] = True
            self.nodes[n['id']] = NeuroNode(**n)
        for s in data.get('synapses', []):
            s.setdefault('fire_count', 0)
            k = f"{s['source']}->{s['target']}"
            self.synapses[k] = NeuroSynapse(**s)
        if 'chrono_nodes' in data:
            self.chrono_layer.restore_from_saved(data['chrono_nodes'])
        warm     = sum(1 for n in self.nodes.values() if n.activation > 0.1 and not n.archived)
        archived = sum(1 for n in self.nodes.values() if n.archived)
        print(f"🧠 Кора: {len(self.nodes)} нейронов ({warm} тёплых, {archived} архивных), {len(self.synapses)} синапсов")

    def _create_node(self, label: str, category: str = "concept") -> NeuroNode:
        nid = hashlib.md5(label.encode()).hexdigest()[:8]
        if nid not in self.nodes:
            self.nodes[nid] = NeuroNode(id=nid, label=label, category=category)
        return self.nodes[nid]

    def _reactivate_if_archived(self, node: NeuroNode):
        if node.archived:
            node.archived = False
            node.activation = min(0.5, node.activation + 0.3)

    def process_input(self, text: str, now: datetime = None) -> Tuple[List[str], List[ChronoNode], Optional[str], Optional[str]]:
        if now is None: now = datetime.now()
        keywords = TextUtils.extract_keywords(text, top_n=10)
        activated_labels = []; now_iso = now.isoformat()
        for i, kw1 in enumerate(keywords):
            n1 = self._create_node(kw1)
            self._reactivate_if_archived(n1)
            n1.access_count += 1; n1.activation = 1.0
            n1.last_active = now_iso; n1.experience += 0.12
            activated_labels.append(kw1)
            for kw2 in keywords[i + 1:]:
                self._strengthen(n1.id, self._create_node(kw2).id, 0.1)
        chrono_nodes  = self.chrono_layer.process_temporal_input(text)
        final_context = set(activated_labels)
        for label in activated_labels:
            nid = hashlib.md5(label.encode()).hexdigest()[:8]
            for key, syn in list(self.synapses.items()):
                if syn.source == nid and syn.weight > 0.4:
                    target = self.nodes.get(syn.target)
                    if target and not target.archived:
                        noisy_act = self.drift.add_noise(0.45)
                        target.activation = min(1.0, target.activation + noisy_act)
                        target.last_active = now_iso
                        final_context.add(target.label)
                        syn.fire_count += 1
        for node in self.nodes.values():
            tonic = ActivationDecay.tonic(node.experience)
            if node.archived:
                node.activation = min(ARCHIVE_ACTIVATION_CAP, node.activation * 0.88)
            else:
                node.activation = max(node.activation * 0.88, tonic)
                if (ActivationDecay.is_archive_candidate(node.last_active)
                        and node.experience < 0.5 and node.access_count < 5):
                    node.archived = True
        explored = self.drift.maybe_explore(self.nodes)
        wild     = self.drift.get_wild_association(self.nodes, self.synapses)
        return list(final_context), chrono_nodes, explored, wild

    def _strengthen(self, src: str, tgt: str, reward: float):
        if src == tgt: return
        for k, (s, t) in [(f"{src}->{tgt}", (src, tgt)), (f"{tgt}->{src}", (tgt, src))]:
            if k not in self.synapses:
                self.synapses[k] = NeuroSynapse(source=s, target=t, weight=0.1)
            syn = self.synapses[k]
            syn.weight = min(1.0, syn.weight + reward * syn.plasticity)
            syn.last_fired = datetime.now().isoformat()
            syn.plasticity *= 0.995; syn.fire_count += 1

    def associate(self, concept: str, top_n: int = 6) -> List[Tuple[str, float]]:
        nid = hashlib.md5(concept.encode()).hexdigest()[:8]
        return sorted([(self.nodes[s.target].label, s.weight)
                       for k, s in self.synapses.items()
                       if s.source == nid and s.weight > 0.15 and s.target in self.nodes
                       and not self.nodes[s.target].archived],
                      key=lambda x: -x[1])[:top_n]

    def get_hot_nodes(self, top_n: int = 8) -> List[NeuroNode]:
        return sorted([n for n in self.nodes.values() if n.activation > 0.05 and not n.archived],
                      key=lambda n: n.activation, reverse=True)[:top_n]

    def get_experienced_nodes(self, top_n: int = 5) -> List[NeuroNode]:
        return sorted([n for n in self.nodes.values() if not n.archived],
                      key=lambda n: n.experience, reverse=True)[:top_n]

    def get_archived_nodes(self, top_n: int = 10) -> List[NeuroNode]:
        return sorted([n for n in self.nodes.values() if n.archived],
                      key=lambda n: n.experience, reverse=True)[:top_n]

    def reinforce_path(self, keywords: List[str], chrono_nodes: List[ChronoNode], success: bool):
        reward = 0.15 if success else -0.04
        for i, kw1 in enumerate(keywords):
            id1 = hashlib.md5(kw1.encode()).hexdigest()[:8]
            for kw2 in keywords[i + 1:]:
                self._strengthen(id1, hashlib.md5(kw2.encode()).hexdigest()[:8], reward)
        for cn in chrono_nodes:
            cn.activation = min(1.0, cn.activation + reward)

    def prune_weak_synapses(self, threshold: float = 0.04) -> int:
        before = len(self.synapses)
        self.synapses = {k: v for k, v in self.synapses.items() if v.weight > threshold}
        return before - len(self.synapses)

    def save(self):
        FileManager.safe_save_json(self.save_path, {
            'nodes':        [asdict(n) for n in self.nodes.values()],
            'synapses':     [asdict(s) for s in self.synapses.values()],
            'chrono_nodes': self.chrono_layer.to_save_list(),
            'saved_at':     datetime.now().isoformat()
        })

    def get_stats(self) -> Dict:
        archived = sum(1 for n in self.nodes.values() if n.archived)
        return {
            'neurons':       len(self.nodes),
            'warm':          sum(1 for n in self.nodes.values() if n.activation > 0.1 and not n.archived),
            'archived':      archived,
            'synapses':      len(self.synapses),
            'strong_syn':    sum(1 for s in self.synapses.values() if s.weight > 0.6),
            'chrono':        len(self.chrono_layer.nodes),
            'active_chrono': sum(1 for n in self.chrono_layer.nodes.values() if n.activation > 0.3)
        }


# ══════════════════════════════════════════
# ДВУХУРОВНЕВАЯ РЕФЛЕКСИЯ
# ══════════════════════════════════════════
class ReflectionEngine:
    def __init__(self, llm: 'LLMInterface'):
        self.llm      = llm
        self._counter = 0

    def tick(self) -> Tuple[bool, bool]:
        self._counter += 1
        fast = self._counter % REFLECTION_FAST_EVERY == 0
        deep = self._counter % REFLECTION_DEEP_EVERY == 0
        return fast, deep

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
            "Ты — наблюдатель. Сделай 1 краткое наблюдение (до 30 слов) о текущей сессии:\n"
            f"Горячие темы: {[n.label for n in hot]}\n"
            f"Чередование тем: {session_topics}\n"
            f"Эмоция: {memory.emotion.get_trend()['trend']}\n"
            "Если есть противоречие или резкая смена — отметь это."
        )
        result = await self.llm.generate(prompt, temp=0.35)
        if result and len(result) > 8:
            memory.meta.add_insight(f"[быстрая] {result}")
            return result
        return None

    async def reflect_deep(self, memory: 'HybridMemorySystem') -> Optional[str]:
        memory.semantic_l3.force_rebuild()
        chains = memory.causal.get_strong_chains(4)
        hot    = memory.cortex.get_hot_nodes(5)
        arc    = memory.cortex.get_archived_nodes(3)
        prompt = (
            "Ты — аналитик долгосрочных паттернов. "
            "Опиши 2 ключевых наблюдения об интересах и паттернах пользователя (≤50 слов):\n"
            f"Профиль: {memory.profile.get_prompt_block() or 'нет данных'}\n"
            f"L3 паттерны: {memory.semantic_l3.facts}\n"
            f"Горячие темы сейчас: {[n.label for n in hot]}\n"
            f"Забытые темы (архив): {[n.label for n in arc]}\n"
            f"Причинные цепочки: {chains}\n"
            "Только факты, без предположений."
        )
        result = await self.llm.generate(prompt, temp=0.40)
        if result and len(result) > 8:
            memory.meta.add_insight(f"[глубокая] {result}")
            return result
        return None


# ══════════════════════════════════════════
# v22: КОГНИТИВНЫЙ РИТМ — ЯДРО
# ══════════════════════════════════════════

@dataclass
class PulsePhaseRecord:
    """Запись одной фазы когнитивного пульса"""
    phase:      str       # PERCEIVE / UNDERSTAND / REFLECT / SYNTHESIZE / RESPOND
    input_len:  int       # Длина входного контекста (символов)
    output:     str       # Результат фазы
    duration_ms: float    # Время выполнения (мс)
    skipped:    bool = False  # Была ли фаза пропущена
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PulseLog:
    """Полный лог одного цикла когнитивного ритма"""
    query:   str
    phases:  List[PulsePhaseRecord] = field(default_factory=list)
    total_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, phase: PulsePhaseRecord):
        self.phases.append(phase)

    def summary(self) -> str:
        lines = [f"🥁 Пульс [{self.timestamp[:16]}]",
                 f"   Запрос: {self.query[:60]}"]
        for p in self.phases:
            icon = "⏭️" if p.skipped else "✅"
            out_preview = p.output[:50].replace('\n', ' ') if p.output else "—"
            lines.append(f"   {icon} {p.phase:<12} {p.duration_ms:>6.0f}мс → {out_preview}")
        lines.append(f"   Итого: {self.total_ms:.0f}мс")
        return "\n".join(lines)


class PulseJournal:
    """Журнал когнитивного пульса — сохраняет историю фаз"""
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.logs: List[PulseLog] = []
        self._load()

    def _load(self):
        data = FileManager.safe_load_json(self.save_path, [])
        for entry in data:
            phases = [PulsePhaseRecord(**p) for p in entry.get('phases', [])]
            self.logs.append(PulseLog(
                query=entry['query'], phases=phases,
                total_ms=entry.get('total_ms', 0.0),
                timestamp=entry.get('timestamp', datetime.now().isoformat())
            ))

    def save(self):
        data = []
        for log in self.logs[-PULSE_LOG_MAX:]:
            data.append({
                'query':     log.query,
                'phases':    [asdict(p) for p in log.phases],
                'total_ms':  log.total_ms,
                'timestamp': log.timestamp
            })
        FileManager.safe_save_json(self.save_path, data)

    def add(self, log: PulseLog):
        self.logs.append(log)
        if len(self.logs) > PULSE_LOG_MAX:
            self.logs = self.logs[-PULSE_LOG_MAX:]

    def last(self, n: int = 3) -> List[PulseLog]:
        return self.logs[-n:]

    def get_status(self) -> str:
        if not self.logs:
            return "Журнал пуст."
        last = self.logs[-1]
        return last.summary()


class CognitivePulse:
    """
    🥁 Ритм мышления v22.

    Пять фаз обработки каждого запроса:

    1. PERCEIVE   — Что именно спрашивает пользователь?
                    Явный и скрытый смысл, эмоция, тип запроса.

    2. UNDERSTAND — Что я знаю об этой теме?
                    Собирает релевантный контекст из памяти, RAG, L3.

    3. REFLECT    — Есть ли противоречие? Что важно учесть?
                    Критическая пауза — пересматривает предположения.

    4. SYNTHESIZE — Как лучше всего ответить?
                    Строит план / структуру ответа из фаз 1-3.

    5. RESPOND    — Финальный ответ пользователю.
                    Использует синтез как внутренний черновик.

    Каждая фаза — отдельный LLM-вызов с ограниченным бюджетом токенов.
    Результат каждой фазы передаётся в следующую.
    """

    # Промпты для каждой фазы
    PHASE_PROMPTS = {
        'PERCEIVE': (
            "Ты — когнитивный анализатор. Задача: проанализируй запрос пользователя.\n"
            "Ответь КРАТКО (2-3 предложения):\n"
            "1. Что явно спрашивает пользователь?\n"
            "2. Что он на самом деле хочет (скрытый запрос)?\n"
            "3. Тип запроса: [фактический / эмоциональный / творческий / технический / смешанный]\n"
            "---\nЗапрос пользователя: {query}\nКонтекст: {context_hint}"
        ),
        'UNDERSTAND': (
            "Ты — система поиска знаний. Задача: собери всё релевантное для ответа.\n"
            "Ответь КРАТКО (3-4 предложения):\n"
            "Что ты знаешь об этой теме из памяти и контекста?\n"
            "Есть ли важные факты, которые стоит упомянуть?\n"
            "---\nАнализ запроса: {perceive_output}\n"
            "Память и RAG: {memory_block}\n"
            "L3 паттерны: {l3_block}"
        ),
        'REFLECT': (
            "Ты — критический наблюдатель. Задача: найди слабые места в предстоящем ответе.\n"
            "Ответь КРАТКО (2-3 предложения):\n"
            "Есть ли противоречия или неточности в том, что мы собираемся сказать?\n"
            "Что важно НЕ упустить? Что стоит подчеркнуть?\n"
            "---\nАнализ: {perceive_output}\nЗнания: {understand_output}\n"
            "Профиль пользователя: {profile_hint}\nЭмоция: {emotion_hint}"
        ),
        'SYNTHESIZE': (
            "Ты — архитектор ответа. Задача: составь ПЛАН лучшего ответа.\n"
            "Напиши структуру в 3-5 пунктах — что и в каком порядке сказать.\n"
            "Учти стиль общения пользователя и его эмоцию.\n"
            "---\nВосприятие: {perceive_output}\n"
            "Знания: {understand_output}\nРефлексия: {reflect_output}\n"
            "Тема: {cluster_hint} | Температура: {temp_hint}"
        ),
        'RESPOND': (
            "Ты — умный ассистент с долговременной памятью.\n"
            "{system_block}\n\n"
            "=== ВНУТРЕННИЙ ПЛАН ОТВЕТА ===\n{synthesize_output}\n"
            "=== КОНЕЦ ПЛАНА ===\n\n"
            "Теперь напиши финальный ответ пользователю, следуя плану выше.\n"
            "Будь естественным, не пересказывай план буквально.\n"
            "---\nЗапрос: {query}"
        )
    }

    def __init__(self, llm: 'LLMInterface', journal: PulseJournal):
        self.llm     = llm
        self.journal = journal

    def _is_short_query(self, text: str) -> bool:
        return TextUtils.word_count(text) < PULSE_SHORT_THRESHOLD

    async def _run_phase(self, phase_name: str, prompt: str, temp: float,
                         max_tokens: int = PULSE_PHASE_MAX_TOKENS) -> Tuple[str, float]:
        """Запускает одну фазу и возвращает (результат, время_мс)"""
        t0 = asyncio.get_event_loop().time()
        result = await self.llm.generate_raw(prompt, temp=temp, max_tokens=max_tokens)
        dt = (asyncio.get_event_loop().time() - t0) * 1000
        return result or "", dt

    async def think(
        self,
        query:         str,
        system_block:  str,
        memory_block:  str,
        l3_block:      str,
        profile_hint:  str,
        emotion_hint:  str,
        cluster_hint:  str,
        context_hint:  str,
        temp:          float,
    ) -> Tuple[str, PulseLog]:
        """
        Главный метод: прогоняет запрос через все 5 фаз.
        Возвращает (финальный_ответ, лог_пульса).
        """
        pulse_log = PulseLog(query=query)
        t_total   = asyncio.get_event_loop().time()

        # ── PERCEIVE ──────────────────────────────────────────
        perceive_prompt = self.PHASE_PROMPTS['PERCEIVE'].format(
            query=query,
            context_hint=context_hint[:200] if context_hint else "нет"
        )
        perceive_out, perceive_ms = await self._run_phase('PERCEIVE', perceive_prompt,
                                                           temp=0.3)
        pulse_log.add(PulsePhaseRecord(
            phase='PERCEIVE', input_len=len(perceive_prompt),
            output=perceive_out, duration_ms=perceive_ms
        ))

        # ── UNDERSTAND ────────────────────────────────────────
        understand_prompt = self.PHASE_PROMPTS['UNDERSTAND'].format(
            perceive_output=perceive_out[:300],
            memory_block=memory_block[:400] if memory_block else "нет данных",
            l3_block=l3_block[:200] if l3_block else "нет данных"
        )
        understand_out, understand_ms = await self._run_phase('UNDERSTAND', understand_prompt,
                                                               temp=0.35)
        pulse_log.add(PulsePhaseRecord(
            phase='UNDERSTAND', input_len=len(understand_prompt),
            output=understand_out, duration_ms=understand_ms
        ))

        # ── REFLECT ───────────────────────────────────────────
        reflect_prompt = self.PHASE_PROMPTS['REFLECT'].format(
            perceive_output=perceive_out[:250],
            understand_output=understand_out[:250],
            profile_hint=profile_hint[:150] if profile_hint else "нет",
            emotion_hint=emotion_hint[:100] if emotion_hint else "нейтральный"
        )
        reflect_out, reflect_ms = await self._run_phase('REFLECT', reflect_prompt,
                                                         temp=0.4)
        pulse_log.add(PulsePhaseRecord(
            phase='REFLECT', input_len=len(reflect_prompt),
            output=reflect_out, duration_ms=reflect_ms
        ))

        # ── SYNTHESIZE ────────────────────────────────────────
        synthesize_prompt = self.PHASE_PROMPTS['SYNTHESIZE'].format(
            perceive_output=perceive_out[:200],
            understand_output=understand_out[:200],
            reflect_output=reflect_out[:200],
            cluster_hint=cluster_hint or "общий",
            temp_hint=f"{temp:.2f}"
        )
        synthesize_out, synthesize_ms = await self._run_phase('SYNTHESIZE', synthesize_prompt,
                                                               temp=0.4)
        pulse_log.add(PulsePhaseRecord(
            phase='SYNTHESIZE', input_len=len(synthesize_prompt),
            output=synthesize_out, duration_ms=synthesize_ms
        ))

        # ── RESPOND ───────────────────────────────────────────
        respond_prompt = self.PHASE_PROMPTS['RESPOND'].format(
            system_block=system_block[:800],
            synthesize_output=synthesize_out[:500],
            query=query
        )
        respond_out, respond_ms = await self._run_phase(
            'RESPOND', respond_prompt, temp=temp,
            max_tokens=PULSE_FINAL_MAX_TOKENS
        )
        pulse_log.add(PulsePhaseRecord(
            phase='RESPOND', input_len=len(respond_prompt),
            output=respond_out, duration_ms=respond_ms
        ))

        pulse_log.total_ms = (asyncio.get_event_loop().time() - t_total) * 1000
        self.journal.add(pulse_log)

        print(f"🥁 Пульс завершён: {pulse_log.total_ms:.0f}мс | "
              f"фаз: {len(pulse_log.phases)}")

        return respond_out, pulse_log

    async def think_fast(
        self,
        query:        str,
        system_block: str,
        temp:         float,
    ) -> Tuple[str, PulseLog]:
        """
        Быстрый режим для коротких запросов — пропускаем промежуточные фазы,
        но всё равно логируем пульс.
        """
        pulse_log = PulseLog(query=query)
        t0 = asyncio.get_event_loop().time()

        # Одна фаза RESPOND напрямую
        respond_prompt = (
            f"Ты — умный ассистент.\n{system_block[:600]}\n\nЗапрос: {query}"
        )
        respond_out, respond_ms = await self._run_phase(
            'RESPOND', respond_prompt, temp=temp,
            max_tokens=PULSE_FINAL_MAX_TOKENS
        )
        for phase in ['PERCEIVE', 'UNDERSTAND', 'REFLECT', 'SYNTHESIZE']:
            pulse_log.add(PulsePhaseRecord(
                phase=phase, input_len=0,
                output="[пропущено — короткий запрос]",
                duration_ms=0, skipped=True
            ))
        pulse_log.add(PulsePhaseRecord(
            phase='RESPOND', input_len=len(respond_prompt),
            output=respond_out, duration_ms=respond_ms
        ))
        pulse_log.total_ms = (asyncio.get_event_loop().time() - t0) * 1000
        self.journal.add(pulse_log)
        return respond_out, pulse_log


# ══════════════════════════════════════════
# ГИБРИДНАЯ ПАМЯТЬ
# ══════════════════════════════════════════
class HybridMemorySystem:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir     = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.dir, exist_ok=True)

        self.l1         = SmartWorkingMemory(os.path.join(self.dir, "short_term.json"))
        self.cortex     = DynamicNeuralCortex(user_id, self.dir)
        self.temporal   = TemporalMemory(user_id, self.dir)
        self.causal     = LogicalChainEngine(os.path.join(self.dir, "causal.json"))
        self.clusters   = SemanticClusterEngine(os.path.join(self.dir, "clusters.json"))
        self.meta       = MetaLearner(os.path.join(self.dir, "meta.json"))
        self.profile    = ProfileManager(os.path.join(self.dir, "profile.json"))
        self.emotion    = EmotionalArc(os.path.join(self.dir, "emotions.json"))
        self.predictor  = TopicPredictor(os.path.join(self.dir, "topic_transitions.json"))
        self.semantic_l3 = SemanticMemoryL3(os.path.join(self.dir, "semantic_l3.json"))
        self.temp_adapter = TemperatureAdapter(os.path.join(self.dir, "temp_feedback.json"))

        # v22: журнал пульса
        self.pulse_journal = PulseJournal(os.path.join(self.dir, "pulse_journal.json"))

        self.now = datetime.now()
        self._msg_count     = 0
        self._prev_cluster: Optional[str] = None
        self._pprev_cluster: Optional[str] = None

    async def process(self, text: str):
        self.now = datetime.now()

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

        new_facts = self.profile.update_from_text(text)

        emotion_str, emotion_score = self.emotion.record(text)
        session_emotion, mood_hint = self.emotion.get_session_mood()

        event = self.temporal.add_event(text, keywords, cluster_name, self.now)
        self.cortex.chrono_layer.link_event(event, chrono_nodes)

        self.semantic_l3.record(cluster_name, self.now)

        rag_results = RAGEngineV2.search(text, self.temporal.events, top_k=RAG_TOP_K,
                                          now=self.now, current_emotion=session_emotion)
        rag_block   = RAGEngineV2.format_for_prompt(rag_results)

        causal_ctx = ""
        for kw in keywords[:3]:
            ctx = self.causal.format_causal_for_prompt(kw, text)
            if ctx:
                causal_ctx = ctx
                break

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
            self.save_all()
            print(f"💾 Автосохранение #{self._msg_count}")

        # v22: возвращаем также блоки для CognitivePulse
        return (
            full_system, temp, concepts, chrono_nodes, xp_msg, new_facts,
            rag_block, l3_block, profile_block, mood_hint,
            clusters[0][0] if clusters else None,
            l1_ctx
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
        if not any(w in text.lower() for w in ['почему','зачем','причина','что будет','что вызывает',
                                                 'если не','без ','последствия']):
            return None
        for kw in TextUtils.extract_keywords(text):
            ctx = self.causal.format_causal_for_prompt(kw, text)
            if ctx:
                return f"🔗 {ctx}"
        return None

    def add_response(self, text: str):
        self.l1.add('assistant', text)

    def save_all(self):
        self.cortex.save(); self.temporal.save(); self.causal.save()
        self.clusters.save(); self.meta.save(); self.profile.save()
        self.emotion.save(); self.predictor.save(); self.semantic_l3.save()
        self.l1.save(); self.temp_adapter.save(); self.pulse_journal.save()

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
        self.url = url; self.key = key
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def generate_raw(self, prompt: str, system: str = None,
                            temp: float = TEMP_DEFAULT, max_tokens: int = 1500) -> str:
        """Низкоуровневый вызов LLM с явным max_tokens"""
        if not self.session: await self.init()
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            async with self.session.post(
                self.url,
                json={"messages": msgs, "temperature": temp, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {self.key}"}
            ) as r:
                if r.status == 200:
                    return (await r.json())['choices'][0]['message']['content']
                return f"LM Error: {r.status}"
        except Exception as e:
            return f"Connection error: {e}"

    async def generate(self, prompt: str, system: str = None,
                       temp: float = TEMP_DEFAULT) -> str:
        """Совместимый метод с v21"""
        return await self.generate_raw(prompt, system=system, temp=temp, max_tokens=1500)

    async def close(self):
        if self.session:
            await self.session.close(); self.session = None


# ══════════════════════════════════════════
# БОТ
# ══════════════════════════════════════════
class HybridBot:
    def __init__(self):
        self.llm       = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.users:    Dict[str, HybridMemorySystem] = {}
        self.reflector = None
        self.stop_flag = False
        # v22: CognitivePulse создаётся после LLMInterface
        self._pulse: Optional[CognitivePulse] = None

    def get_brain(self, uid: str) -> HybridMemorySystem:
        if uid not in self.users:
            self.users[uid] = HybridMemorySystem(uid)
        return self.users[uid]

    def get_pulse(self, brain: HybridMemorySystem) -> CognitivePulse:
        """Возвращает CognitivePulse, привязанный к журналу пользователя"""
        return CognitivePulse(self.llm, brain.pulse_journal)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text: return
        uid   = str(update.effective_user.id)
        text  = update.message.text
        brain = self.get_brain(uid)
        await context.bot.send_chat_action(uid, "typing")

        (full_system, temp, concepts, chrono_nodes,
         xp_msg, new_facts, rag_block, l3_block,
         profile_block, mood_hint, cluster_hint, l1_ctx) = await brain.process(text)

        # Временной запрос
        ta = brain.handle_temporal_query(text)
        if ta:
            brain.add_response(ta)
            await update.message.reply_text(ta)
            if xp_msg: await update.message.reply_text(xp_msg)
            return

        # Логический запрос
        la = brain.handle_logic_query(text)
        if la:
            brain.add_response(la)
            await update.message.reply_text(la)
            return

        # ── v22: КОГНИТИВНЫЙ РИТМ ─────────────────────────────
        pulse = self.get_pulse(brain)

        if PULSE_ENABLED and not pulse._is_short_query(text):
            # Полный 5-фазный ритм мышления
            response, pulse_log = await pulse.think(
                query        = text,
                system_block = full_system,
                memory_block = rag_block,
                l3_block     = l3_block,
                profile_hint = profile_block,
                emotion_hint = mood_hint,
                cluster_hint = cluster_hint or "",
                context_hint = l1_ctx,
                temp         = temp,
            )
        else:
            # Быстрый режим для коротких запросов
            response, pulse_log = await pulse.think_fast(
                query        = text,
                system_block = full_system,
                temp         = temp,
            )
        # ─────────────────────────────────────────────────────

        brain.cortex.reinforce_path(concepts, chrono_nodes, len(response) > 10)

        # Двухуровневая рефлексия
        if self.reflector:
            fast_trigger, deep_trigger = self.reflector.tick()
            if deep_trigger:
                asyncio.create_task(self._reflect_deep(brain))
            elif fast_trigger:
                asyncio.create_task(self._reflect_fast(brain))

        brain.maintenance()
        brain.add_response(response)
        await update.message.reply_text(response)

        for msg in filter(None, [
            xp_msg,
            f"👤 Запомнил: {', '.join(new_facts)}" if new_facts else None
        ]):
            await update.message.reply_text(msg)

    async def _reflect_fast(self, brain: HybridMemorySystem):
        try:
            ins = await self.reflector.reflect_fast(brain)
            if ins: print(f"💡 [Быстрая] Инсайт: {ins[:80]}")
        except Exception as e:
            print(f"⚠️ Fast reflection: {e}")

    async def _reflect_deep(self, brain: HybridMemorySystem):
        try:
            ins = await self.reflector.reflect_deep(brain)
            if ins: print(f"🔍 [Глубокая] Инсайт: {ins[:100]}")
        except Exception as e:
            print(f"⚠️ Deep reflection: {e}")

    # ═══ КОМАНДЫ ═══

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        s     = brain.cortex.get_stats(); chains = brain.causal.get_strong_chains(3)
        temp_adj = brain.temp_adapter.cluster_adjustments
        adj_str  = ", ".join(f"{k}:{v:+.2f}" for k, v in temp_adj.items()) if temp_adj else "нет"
        pulse_count = len(brain.pulse_journal.logs)
        await update.message.reply_text(
            f"🧠 COGNITIVE RHYTHM v22.0\n{'═' * 32}\n"
            f"🔹 Нейроны: {s['neurons']} (тёплых: {s['warm']}, архив: {s['archived']})\n"
            f"🔹 Синапсы: {s['synapses']} (сильных: {s['strong_syn']})\n"
            f"🔹 Chrono: {s['chrono']} (активных: {s['active_chrono']})\n"
            f"🔹 Эпизодов L2: {len(brain.temporal.events)}\n"
            f"🔹 Фактов L3: {len(brain.semantic_l3.facts)}\n"
            f"🔹 L1 история: {len(brain.l1.history)} сообщений\n"
            f"🔹 Причинных связей: {len(brain.causal.links)}\n"
            f"🔹 Кластеров: {len(brain.clusters.clusters)}\n"
            f"🥁 Пульсов записано: {pulse_count}\n"
            f"🌡️ Темп. коррекции: {adj_str}\n"
            f"{'═' * 32}\n"
            f"📈 {brain.meta.get_status()}\n"
            f"{'═' * 32}\n"
            f"🔗 Цепочки:\n  " + ("\n  ".join(chains) if chains else "Пока нет"))

    async def cmd_pulse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """v22: Показать последний когнитивный ритм мышления"""
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        n     = int(context.args[0]) if context.args and context.args[0].isdigit() else 1
        logs  = brain.pulse_journal.last(n)
        if not logs:
            await update.message.reply_text("Журнал пульса пуст. Напишите что-нибудь сначала.")
            return
        lines = [f"🥁 КОГНИТИВНЫЙ РИТМ (последние {len(logs)} цикла)\n{'═' * 30}"]
        for log in logs:
            lines.append(log.summary())
            lines.append("─" * 30)
        await update.message.reply_text("\n".join(lines))

    async def cmd_phase(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """v22: Показать детальный вывод одной фазы из последнего пульса"""
        uid   = str(update.effective_user.id)
        brain = self.get_brain(uid)
        phase_name = context.args[0].upper() if context.args else "SYNTHESIZE"
        logs = brain.pulse_journal.last(1)
        if not logs:
            await update.message.reply_text("Нет данных. Сначала задайте вопрос.")
            return
        last_log = logs[0]
        target   = next((p for p in last_log.phases if p.phase == phase_name), None)
        if not target:
            await update.message.reply_text(
                f"Фаза «{phase_name}» не найдена.\n"
                f"Доступные: {', '.join(p.phase for p in last_log.phases)}")
            return
        status = "⏭️ ПРОПУЩЕНО" if target.skipped else "✅ ВЫПОЛНЕНО"
        await update.message.reply_text(
            f"🔬 ФАЗА {target.phase} [{status}]\n"
            f"{'═' * 28}\n"
            f"Время: {target.duration_ms:.0f}мс\n"
            f"Запрос: {last_log.query[:80]}\n\n"
            f"Вывод:\n{target.output[:600] or '—'}")

    async def cmd_hotmap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        hot   = brain.cortex.get_hot_nodes(10); exp = brain.cortex.get_experienced_nodes(5)
        arc   = brain.cortex.get_archived_nodes(5)
        if not hot: await update.message.reply_text("Горячих нейронов пока нет."); return
        lines = ["🔥 ГОРЯЧАЯ КАРТА\n", "Активные сейчас:"]
        for n in hot[:6]:
            bar = "█" * int(n.activation * 10)
            lines.append(f"  {n.label:<15} {bar} {n.activation:.2f} (exp:{n.experience:.1f})")
        lines.append("\n🏆 Самые опытные:")
        for n in exp:
            lines.append(f"  {n.label:<15} exp:{n.experience:.1f}  тонус:{ActivationDecay.tonic(n.experience):.3f}")
        if arc:
            lines.append("\n💤 Архивные (забытые):")
            for n in arc[:3]: lines.append(f"  {n.label:<15} exp:{n.experience:.1f}")
        chrono_hot = sorted([n for n in brain.cortex.chrono_layer.nodes.values() if n.activation > 0.05],
                             key=lambda x: -x.activation)[:5]
        if chrono_hot:
            lines.append("\n⏱ Временные нейроны:")
            for n in chrono_hot:
                lines.append(f"  {n.label:<12} act:{n.activation:.2f} uses:{n.use_count}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        l3  = brain.semantic_l3.get_prompt_block()
        await update.message.reply_text(
            f"👤 ПРОФИЛЬ\n{'═' * 24}\n{brain.profile.get_status()}\n\n{l3 or 'L3 пока пуст'}")

    async def cmd_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid  = str(update.effective_user.id); brain = self.get_brain(uid)
        days = int(context.args[0]) if context.args and context.args[0].isdigit() else 7
        t    = brain.emotion.get_trend()
        await update.message.reply_text(
            f"💚 ЭМОЦИОНАЛЬНАЯ ДУГА\n{'═' * 24}\n"
            f"{brain.emotion.get_history_summary(days)}\n"
            f"Тренд: {t['trend']} | Направление: {t['direction']}")

    async def cmd_chain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /chain <слово>"); return
        word  = context.args[0].lower()
        chain = brain.causal.build_chain(word)
        lines = []
        if len(chain) > 1: lines.append(f"🔗 Прямая цепочка:\n  {' → '.join(chain)}")
        btree = brain.causal.backward_chain(word, depth=2)
        if btree['causes']:
            lines.append(f"\n⬆️ Причины «{word}»:")
            for c in btree['causes']:
                sub = " ← ".join(cc['node'] for cc in c.get('causes', []))
                lines.append(f"  ← {c['node']} ({c['strength']:.2f})" + (f" ← {sub}" if sub else ""))
        ftree = brain.causal.forward_chain(word, depth=2)
        if ftree['effects']:
            lines.append(f"\n⬇️ Следствия «{word}»:")
            for e in ftree['effects']:
                sub = " → ".join(ee['node'] for ee in e.get('effects', []))
                lines.append(f"  → {e['node']} ({e['strength']:.2f})" + (f" → {sub}" if sub else ""))
        cf = brain.causal.counterfactual(word)
        if cf: lines.append(f"\n🔴 Без «{word}» пострадает: {', '.join(cf)}")
        if not lines: await update.message.reply_text(f"Цепочек для '{word}' пока нет.")
        else: await update.message.reply_text("\n".join(lines))

    async def cmd_assoc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid  = str(update.effective_user.id); brain = self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /assoc <слово>"); return
        word  = context.args[0].lower(); assocs = brain.cortex.associate(word)
        if assocs: await update.message.reply_text(
            f"🧩 Ассоциации '{word}':\n" + "\n".join(f"  {a} ({w:.2f})" for a, w in assocs))
        else: await update.message.reply_text(f"Ассоциаций для '{word}' нет.")

    async def cmd_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid  = str(update.effective_user.id); brain = self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /predict <тема>"); return
        topic  = " ".join(context.args).lower()
        preds1 = brain.predictor.predict_next(topic, top_n=3)
        preds2 = brain.predictor.predict_next(topic, prev=brain._pprev_cluster, top_n=2)
        lines  = [f"🔮 После «{topic}»:"]
        if preds1:
            lines.append("  1-й порядок:")
            for t, prob in preds1: lines.append(f"    {t:<15} {'█' * int(prob * 10)} {prob:.0%}")
        if preds2 and brain._pprev_cluster:
            lines.append(f"  2-й порядок (после [{brain._pprev_cluster}→{topic}]):")
            for t, prob in preds2: lines.append(f"    {t:<15} {'█' * int(prob * 10)} {prob:.0%}")
        if len(lines) == 1: await update.message.reply_text(f"Данных о переходах от «{topic}» ещё нет.")
        else: await update.message.reply_text("\n".join(lines))

    async def cmd_drift(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid  = str(update.effective_user.id); brain = self.get_brain(uid)
        hot  = brain.cortex.get_hot_nodes(3)
        candidates = [n for n in brain.cortex.nodes.values() if 0.01 < n.activation < 0.15
                      and n.experience > 0.3 and not n.archived]
        mid_syn    = [s for s in brain.cortex.synapses.values() if 0.2 < s.weight < 0.5 and s.fire_count > 1]
        archived   = sum(1 for n in brain.cortex.nodes.values() if n.archived)
        await update.message.reply_text(
            f"🎲 СТОХАСТИЧЕСКИЙ ДРЕЙФ\n{'═' * 24}\n"
            f"Вероятность exploration: {DRIFT_EXPLORE_PROB:.0%}\n"
            f"Масштаб шума: {DRIFT_NOISE_SCALE}\n"
            f"Wild-ассоциация каждые: {DRIFT_ASSOC_EVERY} шагов\n\n"
            f"Кандидаты для exploration: {len(candidates)} нейронов\n"
            f"Периферийных синапсов: {len(mid_syn)}\n"
            f"Архивных нейронов: {archived}\n"
            f"Текущие горячие: {', '.join(n.label for n in hot)}")

    async def cmd_timeline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        hours = int(context.args[0]) if context.args and context.args[0].isdigit() else 24
        await update.message.reply_text(f"🕰️ События за {hours}ч:\n{brain.temporal.get_timeline_summary(hours)}")

    async def cmd_clusters(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        top = sorted(brain.clusters.clusters.values(), key=lambda c: c.access_count, reverse=True)[:8]
        if not top: await update.message.reply_text("Кластеры пока не сформированы."); return
        lines = ["🗂 КЛАСТЕРЫ ЗНАНИЙ\n"]
        for c in top:
            lines.append(f"📌 [{c.name}] ({c.access_count} обращений)")
            lines.append(f"   {', '.join(c.members[:7])}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id); brain = self.get_brain(uid)
        if not brain.meta.insights: await update.message.reply_text("Инсайтов пока нет."); return
        lines = ["💡 ИНСАЙТЫ (быстрые + глубокие)\n"]
        for i, ins in enumerate(brain.meta.insights[-12:], 1): lines.append(f"{i}. {ins}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_l1(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        query = " ".join(context.args) if context.args else "общий контекст"
        ctx   = brain.l1.get_context(query)
        if not ctx: await update.message.reply_text("L1 пуста."); return
        lines = [f"🎯 L1 (умная выборка для «{query}»):"]
        for m in ctx:
            role = "👤" if m['role'] == 'user' else "🤖"
            lines.append(f"{role} {m['content'][:100]}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_temp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid   = str(update.effective_user.id); brain = self.get_brain(uid)
        lines = ["🌡️ ТЕМПЕРАТУРНЫЕ ПРОФИЛИ\n"]
        for cluster, base_temp in TemperatureAdapter.CLUSTER_TEMPS.items():
            adj  = brain.temp_adapter.cluster_adjustments.get(cluster, 0.0)
            real = round(base_temp + adj, 2)
            adj_str = f" ({adj:+.3f})" if adj != 0 else ""
            lines.append(f"  {cluster:<12} base:{base_temp} → real:{real}{adj_str}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_wipe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = str(update.effective_user.id)
        if uid in self.users:
            d = self.users.pop(uid).dir
            if os.path.exists(d): shutil.rmtree(d)
            await update.message.reply_text("🧠 Полная очистка выполнена.")
        else: await update.message.reply_text("Пользователь не найден.")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 COGNITIVE RHYTHM v22.0\n\n"
            "Умная L1 + эмоц. RAG + глубокие цепочки +\n"
            "активное забывание + двойная рефлексия +\n"
            "фидбек температуры + марков 2-го порядка +\n"
            "🥁 РИТМ МЫШЛЕНИЯ (5 фаз: Восприятие→Понимание→Рефлексия→Синтез→Ответ)\n\n"
            "📌 КОМАНДЫ:\n"
            "/stats           — статистика нейросети\n"
            "/hotmap          — карта горячих нейронов 🔥\n"
            "/profile         — профиль и L3 память 👤\n"
            "/mood [дней]     — эмоциональная дуга 💚\n"
            "/predict <тема>  — предсказание (2-й порядок) 🔮\n"
            "/drift           — параметры дрейфа 🎲\n"
            "/chain <слово>   — цепочки + дерево причин/следствий\n"
            "/assoc <слово>   — ассоциации\n"
            "/clusters        — смысловые кластеры\n"
            "/insights        — накопленные инсайты\n"
            "/timeline [ч]    — хронология\n"
            "/l1 [запрос]     — умная выборка L1 🎯\n"
            "/temp            — температурные профили 🌡️\n"
            "/wipe            — очистка памяти\n\n"
            "🥁 Новое в v22 (Ритм мышления):\n"
            "/pulse [n]       — показать последние n пульсов мышления\n"
            "/phase <ФАЗА>    — детальный вывод фазы\n"
            "  Фазы: PERCEIVE | UNDERSTAND | REFLECT | SYNTHESIZE | RESPOND\n\n"
            "💡 Как работает ритм:\n"
            "  1. PERCEIVE   — Что спрашивает пользователь?\n"
            "  2. UNDERSTAND — Что я знаю об этом?\n"
            "  3. REFLECT    — Есть ли противоречие?\n"
            "  4. SYNTHESIZE — Как лучше ответить?\n"
            "  5. RESPOND    — Финальный ответ\n"
            "  Короткие запросы (<10 слов) — быстрый режим (1 фаза)."
        )

    async def shutdown(self):
        print("\n💾 Финальное сохранение...")
        for b in self.users.values(): b.save_all()
        await self.llm.close()
        print("✅ Остановлено")


# ══════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════
async def main():
    print("🚀 COGNITIVE RHYTHM v22.0 STARTING...")
    print("🥁 Ритм мышления: PERCEIVE → UNDERSTAND → REFLECT → SYNTHESIZE → RESPOND")
    if not TELEGRAM_TOKEN:
        print("❌ Нет TELEGRAM_TOKEN в .env"); return

    bot = HybridBot()
    bot.reflector = ReflectionEngine(bot.llm)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    for cmd, handler in [
        ("stats",    bot.cmd_stats),
        ("hotmap",   bot.cmd_hotmap),
        ("profile",  bot.cmd_profile),
        ("mood",     bot.cmd_mood),
        ("predict",  bot.cmd_predict),
        ("drift",    bot.cmd_drift),
        ("timeline", bot.cmd_timeline),
        ("chain",    bot.cmd_chain),
        ("assoc",    bot.cmd_assoc),
        ("clusters", bot.cmd_clusters),
        ("insights", bot.cmd_insights),
        ("l1",       bot.cmd_l1),
        ("temp",     bot.cmd_temp),
        ("wipe",     bot.cmd_wipe),
        ("help",     bot.cmd_help),
        # v22
        ("pulse",    bot.cmd_pulse),
        ("phase",    bot.cmd_phase),
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print("✅ COGNITIVE RHYTHM v22 ГОТОВ 🧠🥁")
        print("💡 Новые команды: /pulse, /phase")
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