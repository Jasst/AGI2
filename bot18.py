#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 HYBRID NEURAL BRAIN v20.0 - STOCHASTIC MIND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Новое в v20:
  🎲 Стохастический дрейф — полезная случайность в нейросети:
       • случайное «подбуживание» редких нейронов (exploration)
       • шум при распространении активации (не застрять в одних связях)
       • случайный выбор «свежей» ассоциации раз в N шагов

  🔍 Улучшенный RAG v2 — BM25-подобный скоринг:
       • учитывает частоту слова в документе (TF)
       • и редкость слова во всей памяти (IDF)
       • результат: находит реально релевантные, а не просто похожие эпизоды
       • дедупликация по cluster_name — не вставляет 3 одинаковых эпизода

  🗂️ Трёхуровневая память:
       L1 — рабочая (last 6 сообщений, RAM)
       L2 — эпизодическая (до 400 событий, диск)
       L3 — семантическая (сжатые факты о пользователе, диск)
       Промпт строится с явными секциями L1/L2/L3

  🌡️ Адаптивная температура LLM:
       • техническая тема      → temp 0.3  (точность важнее)
       • эмоциональный контекст → temp 0.85 (теплее, живее)
       • нейтральный диалог    → temp 0.65
       • предсказание / идеи   → temp 0.9  (больше вариативности)

Сохранено из v19: RAG, профиль, предсказание тем,
  эмоциональная дуга, затухающая активация, фоновый тонус,
  ChronoLayer, CausalChain, кластеры, MetaLearner, рефлексия.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, re, asyncio, aiohttp, traceback, hashlib, math, shutil, random
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
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

BASE_DIR   = "hybrid_brain_v20"
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# Нейросеть
ACTIVATION_DECAY_RATE = 0.1
TONIC_SCALE           = 0.03
TONIC_MAX             = 0.25
AUTOSAVE_EVERY        = 10

# RAG v2
RAG_TOP_K      = 4
RAG_MIN_SCORE  = 0.12
RAG_BM25_K1    = 1.5    # насыщение TF
RAG_BM25_B     = 0.75   # нормализация по длине документа

# Стохастический дрейф
DRIFT_EXPLORE_PROB  = 0.08   # вероятность случайного подбуживания нейрона
DRIFT_NOISE_SCALE   = 0.05   # масштаб шума при распространении активации
DRIFT_ASSOC_EVERY   = 7      # каждые N шагов — случайная ассоциация в промпт

# Адаптивная температура
TEMP_TECHNICAL  = 0.30
TEMP_EMOTIONAL  = 0.85
TEMP_CREATIVE   = 0.90
TEMP_DEFAULT    = 0.65


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


# ══════════════════════════════════════════
# СТОХАСТИЧЕСКИЙ ДРЕЙФ  (новое в v20)
# ══════════════════════════════════════════
class StochasticDrift:
    """
    🎲 Управляет полезной случайностью в нейросети.

    Зачем нужна случайность?
    • Без неё сеть «застрянет» в одних и тех же связях —
      если ты всегда говоришь о Python, бот никогда не вспомнит
      про другие твои интересы даже если они есть в памяти.
    • С дрейфом сеть периодически «разогревает» случайные нейроны
      — это называется exploration (исследование) в противовес
      exploitation (использование известного).

    Три механизма:
    1. Exploration burst — случайный нейрон получает импульс активации
    2. Activation noise   — небольшой шум при распространении (не детерминизм)
    3. Wild association   — раз в N шагов в промпт добавляется неожиданная связь
    """

    def __init__(self):
        self._step_counter = 0

    def maybe_explore(self, nodes: Dict[str, Any]) -> Optional[str]:
        """
        С вероятностью DRIFT_EXPLORE_PROB случайно активировать
        редко используемый нейрон. Возвращает его label или None.
        """
        if not nodes or random.random() > DRIFT_EXPLORE_PROB:
            return None

        # Выбираем нейроны с низкой активацией но ненулевым опытом
        # — это «забытые, но знакомые» концепты
        candidates = [
            n for n in nodes.values()
            if 0.01 < n.activation < 0.15 and n.experience > 0.3
        ]
        if not candidates:
            return None

        chosen = random.choice(candidates)
        # Небольшой импульс — не полная активация, чтобы не доминировал
        chosen.activation = min(0.4, chosen.activation + 0.25 + random.uniform(0, 0.1))
        chosen.last_active = datetime.now().isoformat()
        return chosen.label

    def add_noise(self, activation: float) -> float:
        """Добавить небольшой гауссов шум к активации при распространении"""
        noise = random.gauss(0, DRIFT_NOISE_SCALE)
        return max(0.0, min(1.0, activation + noise))

    def get_wild_association(self, nodes: Dict[str, Any],
                              synapses: Dict[str, Any]) -> Optional[str]:
        """
        Раз в DRIFT_ASSOC_EVERY шагов — вернуть неожиданную но связанную ассоциацию.
        Находит синапс со средним весом (не самый сильный, не случайный) —
        это «периферийная» связь, которую бот обычно игнорирует.
        """
        self._step_counter += 1
        if self._step_counter % DRIFT_ASSOC_EVERY != 0:
            return None

        # Синапсы среднего веса — «периферийные»
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

    def describe_drift(self, explored: Optional[str],
                       wild: Optional[str]) -> str:
        """Блок для системного промпта"""
        parts = []
        if explored:
            parts.append(f"[Случайное воспоминание: {explored}]")
        if wild:
            parts.append(f"[Периферийная связь: {wild}]")
        return " ".join(parts)


# ══════════════════════════════════════════
# АДАПТИВНАЯ ТЕМПЕРАТУРА  (новое в v20)
# ══════════════════════════════════════════
class TemperatureAdapter:
    """
    🌡️ Подбирает оптимальную температуру для LLM на основе контекста.

    temperature = степень случайности/творчества ответа LLM.
    Низкая (0.2-0.4) → точные, детерминированные ответы (код, факты)
    Высокая (0.8-1.0) → живые, вариативные, творческие ответы

    Адаптируем по трём сигналам:
    • cluster_name — тип темы
    • emotion_trend — настроение пользователя
    • query_type    — тип запроса (вопрос, команда, рассказ)
    """
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

    @staticmethod
    def get(cluster_name: Optional[str],
            emotion: str,
            text: str) -> float:
        base = TemperatureAdapter.CLUSTER_TEMPS.get(cluster_name, TEMP_DEFAULT)

        # Модификатор по эмоции
        if emotion == 'negative':
            base = min(TEMP_EMOTIONAL, base + 0.15)   # теплее при негативе
        elif emotion == 'positive':
            base = min(TEMP_CREATIVE, base + 0.05)

        # Модификатор по типу запроса
        tl = text.lower()
        if any(w in tl for w in ['напиши код','функция','class','def ','import ']):
            base = min(base, TEMP_TECHNICAL)           # код — всегда точно
        elif any(w in tl for w in ['придумай','представь','фантазия','идея','вариант']):
            base = max(base, TEMP_CREATIVE)            # творческое — выше
        elif text.strip().endswith('?'):
            base = min(base, base * 0.9)               # вопрос — чуть точнее

        return round(max(0.1, min(1.0, base)), 2)


# ══════════════════════════════════════════
# RAG v2 — BM25-подобный скоринг  (улучшено в v20)
# ══════════════════════════════════════════
@dataclass
class RAGResult:
    content:      str
    score:        float
    timestamp:    str
    rel_time:     str
    cluster_name: Optional[str] = None


class RAGEngineV2:
    """
    🔍 RAG с BM25-подобным скорингом.

    BM25 — стандартный алгоритм поиска в информационных системах.
    Учитывает:
    - TF (term frequency) — насколько часто слово встречается в документе
    - IDF (inverse document frequency) — насколько редкое слово во всей коллекции
    - Длину документа — короткие документы не получают несправедливое преимущество

    Дополнительно:
    - recency_weight — свежие эпизоды получают бонус
    - importance     — важные эпизоды поднимаются выше
    - дедупликация   — не вставляем 3 эпизода из одного кластера
    """

    @staticmethod
    def _compute_idf(query_terms: List[str], all_events: List[Any]) -> Dict[str, float]:
        """IDF: log((N - df + 0.5) / (df + 0.5)) где df = число документов с термом"""
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
    def _bm25_score(query_terms: List[str],
                    doc_terms: List[str],
                    idf: Dict[str, float],
                    avg_len: float) -> float:
        """BM25 score для одного документа"""
        score = 0.0
        dl    = len(doc_terms)
        term_freq = Counter(doc_terms)
        for term in query_terms:
            if term not in term_freq:
                continue
            tf    = term_freq[term]
            idf_t = idf.get(term, 0.1)
            # BM25 формула
            tf_norm = (tf * (RAG_BM25_K1 + 1)) / \
                      (tf + RAG_BM25_K1 * (1 - RAG_BM25_B + RAG_BM25_B * dl / max(avg_len, 1)))
            score += idf_t * tf_norm
        return score

    @staticmethod
    def search(query: str,
               events: List[Any],
               top_k: int = RAG_TOP_K,
               min_score: float = RAG_MIN_SCORE,
               now: datetime = None) -> List[RAGResult]:
        if not events or not query:
            return []
        if now is None:
            now = datetime.now()

        query_terms = TextUtils.tokenize(query)
        if not query_terms:
            return []

        # Предвычисляем IDF по всей коллекции
        idf      = RAGEngineV2._compute_idf(query_terms, events)
        avg_len  = sum(len(e.keywords) for e in events) / len(events)

        scored   = []
        for event in events:
            doc_terms = TextUtils.tokenize(event.content)
            bm25      = RAGEngineV2._bm25_score(query_terms, doc_terms, idf, avg_len)
            if bm25 < 0.01:
                continue

            # Recency weight: логарифмическое затухание по времени
            age_h   = max((now - datetime.fromisoformat(event.timestamp)).total_seconds()/3600, 0.1)
            recency = 1.0 / (1.0 + math.log1p(age_h / 24))

            # Итоговый score
            score = bm25 * 0.65 + recency * 0.2 + event.importance * 0.15
            if score >= min_score:
                scored.append((score, event))

        scored.sort(key=lambda x: -x[0])

        # Дедупликация по кластеру — не более 2 эпизодов из одной темы
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
                cluster_name = event.cluster_name
            ))
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def format_for_prompt(results: List[RAGResult]) -> str:
        if not results:
            return ""
        lines = ["=== РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ (из долгосрочной памяти) ==="]
        for r in results:
            cluster = f"[{r.cluster_name}] " if r.cluster_name else ""
            lines.append(f"• [{r.rel_time}] {cluster}{r.content[:130]}")
        lines.append("=== КОНЕЦ ВОСПОМИНАНИЙ ===")
        return "\n".join(lines)


# ══════════════════════════════════════════
# ТРЁХУРОВНЕВАЯ ПАМЯТЬ — L3 семантическая  (новое в v20)
# ══════════════════════════════════════════
class SemanticMemoryL3:
    """
    🗂️ L3 — семантическая память: сжатые долгосрочные факты.

    В отличие от L2 (эпизодическая — «что было вчера»),
    L3 хранит абстракции: «пользователь любит Python»,
    «обычно общается вечером», «интересуется алгоритмами».

    Эти факты вставляются в промпт как короткий блок контекста —
    LLM сразу «знает» ключевое о пользователе, не тратя токены
    на пролистывание истории.

    Обновляется автоматически каждые SEMANTIC_UPDATE_EVERY сообщений.
    """
    SEMANTIC_UPDATE_EVERY = 25

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.facts: List[str] = []          # Сжатые факты
        self.topic_freq: Dict[str, int] = {}  # Частота тем
        self.time_patterns: Dict[str, int] = {}  # Паттерны времени
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
            'facts':         self.facts,
            'topic_freq':    self.topic_freq,
            'time_patterns': self.time_patterns,
            'counter':       self._counter
        })

    def record(self, cluster_name: Optional[str], now: datetime):
        """Обновить статистику при каждом сообщении"""
        self._counter += 1
        if cluster_name:
            self.topic_freq[cluster_name] = self.topic_freq.get(cluster_name, 0) + 1

        # Паттерн времени суток
        hour = now.hour
        period = ('утро' if 6 <= hour < 12 else
                  'день' if 12 <= hour < 18 else
                  'вечер' if 18 <= hour < 23 else 'ночь')
        self.time_patterns[period] = self.time_patterns.get(period, 0) + 1

        # Раз в N сообщений — пересчитать факты
        if self._counter % self.SEMANTIC_UPDATE_EVERY == 0:
            self._rebuild_facts()

    def _rebuild_facts(self):
        """Пересобрать список фактов из статистики"""
        new_facts = []

        # Топ темы
        if self.topic_freq:
            top_topics = sorted(self.topic_freq.items(), key=lambda x: -x[1])[:3]
            topics_str = ', '.join(f"{t}" for t, _ in top_topics)
            new_facts.append(f"Часто обсуждает: {topics_str}")

        # Активное время
        if self.time_patterns:
            top_period = max(self.time_patterns.items(), key=lambda x: x[1])
            new_facts.append(f"Чаще всего активен: {top_period[0]}")

        # Доминирующая тема
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
        'утро':   {'hour_range':(6,12)},  'утром':   {'hour_range':(6,12)},
        'день':   {'hour_range':(12,18)}, 'днём':    {'hour_range':(12,18)},
        'вечер':  {'hour_range':(18,24)}, 'вечером': {'hour_range':(18,24)},
        'ночь':   {'hour_range':(0,6)},   'ночью':   {'hour_range':(0,6)},
    }

    @staticmethod
    def parse_time_ref(text: str, now: datetime) -> Optional[Dict]:
        tl = text.lower()
        for phrase, offset in TimeEncoder.TIME_WORDS.items():
            if phrase in tl:
                if 'hour_range' in offset:
                    return {'type':'time_of_day','range':offset['hour_range'],'date':now.date()}
                delta  = timedelta(**{k:v for k,v in offset.items()})
                target = now + delta
                return {'type':'date_range',
                        'start':target.replace(hour=0,minute=0,second=0),
                        'end':  target.replace(hour=23,minute=59,second=59)}
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
        if d.days==1: return "вчера"
        if d.days<7:  return f"{d.days} дн. назад"
        return event_time.strftime("%d.%m.%Y")


# ══════════════════════════════════════════
# ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ
# ══════════════════════════════════════════
@dataclass
class UserProfile:
    name:           str       = ""
    profession:     str       = ""
    interests:      List[str] = field(default_factory=list)
    communication_style: str  = "neutral"
    disliked_topics: List[str]= field(default_factory=list)
    known_facts:    List[str] = field(default_factory=list)
    last_updated:   str       = field(default_factory=lambda: datetime.now().isoformat())


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
        'technical': ['код','функция','алгоритм','python','class','debug','api'],
        'formal':    ['уважаемый','пожалуйста','благодарю','прошу','позвольте'],
        'casual':    ['норм','окей','ок','лол','кстати','слушай','короче'],
        'emotional': ['грустно','радостно','обидно','восторг','боюсь','тревожно'],
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
                fs = f"{value}"
                if fs not in self.profile.known_facts:
                    self.profile.known_facts.append(fs)
                    if len(self.profile.known_facts) > 20: self.profile.known_facts = self.profile.known_facts[-20:]
                    new_facts.append(f"факт: {value}")
        for style, clues in self.STYLE_CLUES.items():
            if sum(1 for c in clues if c in tl) >= 2:
                self.profile.communication_style = style; break
        if new_facts: self.profile.last_updated = datetime.now().isoformat()
        return new_facts

    def get_prompt_block(self) -> str:
        p = self.profile; lines = []
        if p.name:        lines.append(f"Имя: {p.name}")
        if p.profession:  lines.append(f"Профессия: {p.profession}")
        if p.interests:   lines.append(f"Интересы: {', '.join(p.interests[:5])}")
        if p.disliked_topics: lines.append(f"Не интересует: {', '.join(p.disliked_topics[:3])}")
        if p.known_facts: lines.append(f"Факты: {'; '.join(p.known_facts[:4])}")
        style_hint = {'technical':"Говори технически точно.",
                      'formal':   "Общайся формально и уважительно.",
                      'casual':   "Общайся непринуждённо.",
                      'emotional':"Будь эмпатичным."}.get(p.communication_style, "")
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
        """Вернуть (emotion_str, prompt_hint)"""
        if not self._session: return 'neutral', ""
        avg = sum(self._session) / len(self._session)
        if avg > 0.3:   return 'positive', "Пользователь в хорошем настроении. Поддерживай позитивный тон."
        if avg < -0.3:  return 'negative', "Пользователь расстроен или устал. Будь особенно мягким."
        return 'neutral', ""

    def get_trend(self, last_n: int = 10) -> Dict:
        recent = self.history[-last_n:]
        if not recent: return {'trend':'unknown','avg':0.0,'direction':'stable','count':0}
        scores = [p.score for p in recent]
        avg    = sum(scores)/len(scores)
        mid    = len(scores)//2
        fh = sum(scores[:mid])/max(mid,1); sh = sum(scores[mid:])/max(len(scores)-mid,1)
        direction = 'rising' if sh>fh+0.1 else ('falling' if sh<fh-0.1 else 'stable')
        trend = 'positive' if avg>0.15 else ('negative' if avg<-0.15 else 'neutral')
        return {'trend':trend,'avg':round(avg,2),'direction':direction,'count':len(recent)}

    def get_history_summary(self, days: int=7) -> str:
        cutoff = datetime.now()-timedelta(days=days)
        recent = [p for p in self.history if datetime.fromisoformat(p.timestamp)>=cutoff]
        if not recent: return "Нет данных."
        pos = sum(1 for p in recent if p.emotion=='positive')
        neg = sum(1 for p in recent if p.emotion=='negative')
        t   = self.get_trend()
        return (f"За {days} дн.: 😊{pos} / 😐{len(recent)-pos-neg} / 😟{neg}\n"
                f"Score: {t['avg']:+.2f} | Тренд: {t['direction']}")


# ══════════════════════════════════════════
# ПРЕДСКАЗАНИЕ ТЕМАТИКИ
# ══════════════════════════════════════════
class TopicPredictor:
    def __init__(self, save_path: str):
        self.save_path   = save_path
        self.transitions: Dict[str,Dict[str,int]] = {}
        self._load()

    def _load(self): self.transitions = FileManager.safe_load_json(self.save_path, {})
    def save(self):  FileManager.safe_save_json(self.save_path, self.transitions)

    def record_transition(self, frm: str, to: str):
        if not frm or not to or frm==to: return
        if frm not in self.transitions: self.transitions[frm] = {}
        self.transitions[frm][to] = self.transitions[frm].get(to, 0) + 1

    def predict_next(self, current: str, top_n: int=2) -> List[Tuple[str,float]]:
        if current not in self.transitions: return []
        counts = self.transitions[current]; total = sum(counts.values())
        return sorted([(t,c/total) for t,c in counts.items()], key=lambda x:-x[1])[:top_n]

    def get_hint(self, cluster: str) -> str:
        preds = self.predict_next(cluster)
        if not preds: return ""
        best, prob = preds[0]
        if prob > 0.4: return f"Вероятно, следующий вопрос о теме «{best}» ({prob:.0%})."
        return ""


# ══════════════════════════════════════════
# ЛОГИЧЕСКИЕ ЦЕПОЧКИ
# ══════════════════════════════════════════
@dataclass
class CausalLink:
    cause: str; effect: str; link_type: str='positive'
    strength: float=0.5; evidence: int=1
    created_at: str=field(default_factory=lambda:datetime.now().isoformat())
    last_seen:  str=field(default_factory=lambda:datetime.now().isoformat())

    def reinforce(self, d: float=0.12):
        self.strength=min(1.0,self.strength+d); self.evidence+=1
        self.last_seen=datetime.now().isoformat()
    def decay(self, f: float=0.995): self.strength=max(0.0,self.strength*f)


class LogicalChainEngine:
    PATTERNS = [
        (r'(\w{3,})\s+потому\s+что\s+(\w{3,})',    'effect','cause',  'positive'),
        (r'(\w{3,})\s+так\s+как\s+(\w{3,})',        'effect','cause',  'positive'),
        (r'из-за\s+(\w{3,})\s+(\w{3,})',            'cause', 'effect', 'positive'),
        (r'(\w{3,})\s+приводит\s+к\s+(\w{3,})',     'cause', 'effect', 'positive'),
        (r'(\w{3,})\s+вызывает\s+(\w{3,})',         'cause', 'effect', 'positive'),
        (r'(\w{3,})\s+помогает\s+(\w{3,})',         'cause', 'effect', 'positive'),
        (r'(\w{3,})\s+мешает\s+(\w{3,})',           'cause', 'effect', 'negative'),
        (r'если\s+(\w{3,}).{0,15}то\s+(\w{3,})',   'cause', 'effect', 'positive'),
        (r'(\w{3,})\s+даёт\s+(\w{3,})',             'cause', 'effect', 'positive'),
        (r'(\w{3,})\s+влияет\s+на\s+(\w{3,})',     'cause', 'effect', 'positive'),
        (r'без\s+(\w{3,})\s+нет\s+(\w{3,})',        'cause', 'effect', 'negative'),
        (r'(\w{3,})\s+означает\s+(\w{3,})',         'cause', 'effect', 'positive'),
    ]

    def __init__(self, save_path: str):
        self.save_path = save_path; self.links: Dict[str,CausalLink] = {}; self._load()

    def _key(self, c, e): return f"{c}→{e}"
    def _load(self):
        for k,v in FileManager.safe_load_json(self.save_path,{}).items():
            v.setdefault('link_type','positive'); self.links[k]=CausalLink(**v)
    def save(self): FileManager.safe_save_json(self.save_path,{k:asdict(v) for k,v in self.links.items()})

    def extract_from_text(self, text: str) -> List[CausalLink]:
        found, tl = [], text.lower()
        for pat, r1, r2, lt in self.PATTERNS:
            for m in re.finditer(pat, tl):
                w1,w2 = m.group(1),m.group(2)
                if w1 in TextUtils.STOP_WORDS or w2 in TextUtils.STOP_WORDS: continue
                c = w1 if r1=='cause' else w2; e = w2 if r2=='effect' else w1
                k = self._key(c,e)
                if k in self.links: self.links[k].reinforce()
                else: self.links[k]=CausalLink(cause=c,effect=e,link_type=lt)
                found.append(self.links[k])
        return found

    def build_chain(self, start: str, max_depth: int=6) -> List[str]:
        chain,cur,vis=[start],start,{start}
        for _ in range(max_depth):
            best,bs=None,0.0
            for l in self.links.values():
                if l.cause==cur and l.effect not in vis:
                    s=l.strength*l.evidence
                    if s>bs: best,bs=l.effect,s
            if not best or bs<0.1: break
            chain.append(best);vis.add(best);cur=best
        return chain

    def find_causes(self, effect: str) -> List[Tuple[str,float]]:
        return sorted([(l.cause,l.strength) for l in self.links.values()
                       if l.effect==effect and l.strength>=0.15], key=lambda x:-x[1])

    def get_strong_chains(self, top_n: int=3) -> List[str]:
        seen,chains=[],[]
        for l in sorted(self.links.values(), key=lambda x:x.strength*x.evidence, reverse=True):
            if l.cause in seen: continue
            ch=self.build_chain(l.cause)
            if len(ch)>1: chains.append(" → ".join(ch)); seen.append(l.cause)
            if len(chains)>=top_n: break
        return chains

    def decay_all(self):
        for l in self.links.values(): l.decay()


# ══════════════════════════════════════════
# СЕМАНТИЧЕСКИЕ КЛАСТЕРЫ
# ══════════════════════════════════════════
@dataclass
class SemanticCluster:
    id: str; name: str; members: List[str]=field(default_factory=list)
    centroid_keywords: List[str]=field(default_factory=list)
    strength: float=0.5; access_count: int=0


class SemanticClusterEngine:
    SEEDS = {
        'кино':        ['фильм','кино','актёр','режиссёр','сцена','сериал','смотреть'],
        'технологии':  ['код','программа','компьютер','python','алгоритм','сервер','данные'],
        'еда':         ['еда','готовить','ресторан','вкусно','рецепт','блюдо','кофе'],
        'здоровье':    ['болеть','врач','здоровье','лекарство','самочувствие','больница'],
        'работа':      ['работа','офис','задача','проект','коллега','дедлайн','задание'],
        'отдых':       ['отдых','отпуск','путешествие','прогулка','развлечение','игра'],
        'эмоции':      ['рад','грусть','злость','страх','счастье','настроение','чувство'],
        'общение':     ['друг','разговор','встреча','звонок','письмо','общение','семья'],
        'учёба':       ['учить','знание','школа','курс','книга','читать','изучать'],
        'спорт':       ['спорт','тренировка','бегать','gym','фитнес','игра','команда'],
        'деньги':      ['деньги','зарплата','бюджет','расход','доход','инвестиции'],
        'наука':       ['наука','исследование','эксперимент','теория','физика','химия'],
    }

    def __init__(self, save_path: str):
        self.save_path = save_path; self.clusters: Dict[str,SemanticCluster]={}
        self._load()
        if not self.clusters: self._init()

    def _load(self):
        for k,v in FileManager.safe_load_json(self.save_path,{}).items():
            self.clusters[k]=SemanticCluster(**v)

    def save(self): FileManager.safe_save_json(self.save_path,{k:asdict(v) for k,v in self.clusters.items()})

    def _init(self):
        for name,words in self.SEEDS.items():
            cid=hashlib.md5(name.encode()).hexdigest()[:8]
            self.clusters[cid]=SemanticCluster(id=cid,name=name,members=words,
                                               centroid_keywords=words[:3],strength=0.7)

    def classify(self, keywords: List[str]) -> List[Tuple[str,float]]:
        scores=[]
        for cid,c in self.clusters.items():
            hits=sum(1 for kw in keywords if kw in c.members)
            if hits>0:
                scores.append((c.name, hits/max(len(keywords),1)))
                c.access_count+=1
        return sorted(scores,key=lambda x:-x[1])

    def learn(self, keywords: List[str], hint: str="") -> Optional[str]:
        classified=self.classify(keywords)
        if classified and classified[0][1]>=0.25:
            best=classified[0][0]
            for cid,c in self.clusters.items():
                if c.name==best:
                    for kw in keywords:
                        if kw not in c.members: c.members.append(kw)
                        if len(c.members)>120: c.members=c.members[-120:]
                    c.strength=min(1.0,c.strength+0.015); break
            return best
        if len(keywords)>=2:
            name=hint or keywords[0]
            cid=hashlib.md5(name.encode()).hexdigest()[:8]
            if cid not in self.clusters:
                self.clusters[cid]=SemanticCluster(id=cid,name=name,members=list(keywords),
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
    timestamp:str; event_type:str; description:str; impact_score:float=0.0


class MetaLearner:
    THRESHOLDS=[0,50,150,350,750,1500,3000,6000]
    NAMES={1:"Новичок",2:"Ученик",3:"Знающий",4:"Опытный",5:"Мастер",6:"Эксперт",7:"Гений",8:"Оракул"}

    def __init__(self, save_path: str):
        self.save_path=save_path; self.xp=0; self.level=1
        self.total_interactions=0; self.growth_log=[]; self.insights=[]; self._load()

    def _load(self):
        d=FileManager.safe_load_json(self.save_path,{})
        self.xp=d.get('xp',0); self.level=d.get('level',1)
        self.total_interactions=d.get('total_interactions',0)
        self.insights=d.get('insights',[])
        for r in d.get('growth_log',[]): self.growth_log.append(NeuralGrowthRecord(**r))

    def save(self):
        FileManager.safe_save_json(self.save_path,{
            'xp':self.xp,'level':self.level,
            'total_interactions':self.total_interactions,
            'insights':self.insights,
            'growth_log':[asdict(r) for r in self.growth_log[-100:]]})

    def gain_xp(self, amount: int, event: str, desc: str="") -> Optional[str]:
        self.xp+=amount; self.total_interactions+=1
        self.growth_log.append(NeuralGrowthRecord(
            timestamp=datetime.now().isoformat(),event_type=event,
            description=desc,impact_score=amount/10.0))
        return self._check_level_up()

    def _check_level_up(self) -> Optional[str]:
        for lvl,thr in enumerate(self.THRESHOLDS[1:],start=2):
            if self.xp>=thr and self.level<lvl:
                self.level=lvl
                return f"🆙 УРОВЕНЬ {lvl} — {self.NAMES.get(lvl,'?')}!\n📊 Опыт: {self.xp}"
        return None

    def add_insight(self, s: str):
        if s and s not in self.insights:
            self.insights.append(s)
            if len(self.insights)>60: self.insights=self.insights[-60:]

    def get_insights_prompt(self, n: int=2) -> str:
        if not self.insights: return ""
        return "📌 Из прошлых сессий:\n"+"".join(f"  • {i}\n" for i in self.insights[-n:])

    def get_status(self) -> str:
        name=self.NAMES.get(self.level,'?')
        nxt=next((t for t in self.THRESHOLDS if t>self.xp),None)
        sfx=f" (до след.: {nxt-self.xp} XP)" if nxt else " (MAX)"
        return (f"Уровень {self.level} — {name}{sfx}\n"
                f"Опыт: {self.xp} | Диалогов: {self.total_interactions}\n"
                f"Инсайтов: {len(self.insights)}")


# ══════════════════════════════════════════
# ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ (L2)
# ══════════════════════════════════════════
@dataclass
class EpisodicMemory:
    id:str; timestamp:str; content:str; keywords:List[str]
    temporal_tags:List[str]; user_emotion:Optional[str]
    location_hints:List[str]; sequence_id:int
    cluster_name:Optional[str]=None; importance:float=0.5


class TemporalMemory:
    def __init__(self, user_id:str, save_dir:str):
        self.file=os.path.join(save_dir,"episodic_time.json")
        self.events:List[EpisodicMemory]=[]; self.sequence_counter=0; self.now=datetime.now()
        data=FileManager.safe_load_json(self.file)
        if data:
            self.sequence_counter=data.get('seq',0)
            for e in data.get('events',[]):
                e.setdefault('cluster_name',None); e.setdefault('importance',0.5)
                self.events.append(EpisodicMemory(**e))
        print(f"⏳ Эпизодов: {len(self.events)}")

    def add_event(self, content:str, keywords:List[str],
                  cluster_name:str=None, now:datetime=None) -> EpisodicMemory:
        if now is None: now=datetime.now(); self.now=now
        ttags=[tw for tw in ['сегодня','вчера','завтра','утром','вечером','ночью','сейчас']
               if tw in content.lower()]
        locs=[l for l in re.findall(r'(?:в|на|из|к)\s+([а-яёa-z]{3,})',content.lower())
              if l not in {'дом','домой','город','место'}]
        imp=min(1.0,0.3+len(content)/500+(0.2 if ttags else 0))
        e=EpisodicMemory(
            id=hashlib.md5(f"{content}{now.isoformat()}".encode()).hexdigest()[:12],
            timestamp=now.isoformat(),content=content,keywords=keywords,
            temporal_tags=ttags,user_emotion=self._detect_emotion(content),
            location_hints=locs,sequence_id=self.sequence_counter,
            cluster_name=cluster_name,importance=imp)
        self.events.append(e); self.sequence_counter+=1
        if len(self.events)>400:
            self.events.sort(key=lambda e:e.importance*0.5+datetime.fromisoformat(e.timestamp).timestamp()*1e-10)
            self.events=self.events[-400:]; self.events.sort(key=lambda e:e.sequence_id)
        return e

    def _detect_emotion(self, text:str)->Optional[str]:
        t=text.lower()
        if any(w in t for w in ['рад','хорошо','отлично','супер','люблю','нравится']): return 'positive'
        if any(w in t for w in ['грустно','плохо','устал','злой','проблема','боюсь']): return 'negative'
        return None

    def query_by_time(self, time_ref:str, now:datetime=None) -> List[EpisodicMemory]:
        if now is None: now=datetime.now()
        tr=TimeEncoder.parse_time_ref(time_ref,now)
        if not tr: return []
        res=[]
        for ev in reversed(self.events):
            et=datetime.fromisoformat(ev.timestamp)
            if tr['type']=='date_range':
                if tr['start']<=et<=tr['end']: res.append(ev)
            elif tr['type']=='time_of_day':
                if et.date()==tr['date'] and tr['range'][0]<=et.hour<tr['range'][1]: res.append(ev)
        return res[:10]

    def get_timeline_summary(self, hours:int=24) -> str:
        now=datetime.now(); cutoff=now-timedelta(hours=hours)
        recent=[e for e in self.events if datetime.fromisoformat(e.timestamp)>=cutoff]
        if not recent: return "За это время ничего не зафиксировано."
        return "\n".join(
            f"[{TimeEncoder.describe(datetime.fromisoformat(e.timestamp),now)}] "
            f"{'['+e.cluster_name+'] ' if e.cluster_name else ''}{e.content[:90]}"
            for e in recent[-6:])

    def save(self):
        FileManager.safe_save_json(self.file,{'seq':self.sequence_counter,
                                              'events':[asdict(e) for e in self.events]})


# ══════════════════════════════════════════
# CHRONO LAYER
# ══════════════════════════════════════════
@dataclass
class ChronoNode:
    id:str; label:str; node_type:str; activation:float=0.0
    last_active:str=field(default_factory=lambda:datetime.now().isoformat())
    use_count:int=0; linked_events:List[str]=field(default_factory=list)


class ChronoNeuralLayer:
    def __init__(self):
        self.nodes:Dict[str,ChronoNode]={}; self._init()

    def _init(self):
        for label,ntype in [
            ('вчера','temporal_ref'),('сегодня','temporal_ref'),('завтра','temporal_ref'),
            ('утром','time_of_day'),('днём','time_of_day'),('вечером','time_of_day'),
            ('ночью','time_of_day'),('утро','time_of_day'),('вечер','time_of_day'),
            ('раньше','sequence'),('потом','sequence'),('до','sequence'),
            ('после','sequence'),('всегда','sequence'),('никогда','sequence'),
        ]:
            nid=hashlib.md5(f"chrono_{label}".encode()).hexdigest()[:8]
            if nid not in self.nodes: self.nodes[nid]=ChronoNode(id=nid,label=label,node_type=ntype)

    def restore_from_saved(self, saved:List[Dict]):
        for nd in saved:
            nid=nd['id']
            if nid in self.nodes:
                raw=nd.get('activation',0.0); la=nd.get('last_active',datetime.now().isoformat())
                uc=nd.get('use_count',0)
                self.nodes[nid].activation=max(ActivationDecay.apply(raw,la,rate=0.08),min(0.15,uc*0.005))
                self.nodes[nid].last_active=la; self.nodes[nid].use_count=uc
                self.nodes[nid].linked_events=nd.get('linked_events',[])

    def to_save_list(self)->List[Dict]: return [asdict(n) for n in self.nodes.values()]

    def process_temporal_input(self, text:str) -> List[ChronoNode]:
        activated,tl=[],text.lower()
        for n in self.nodes.values():
            if n.label in tl:
                n.activation=min(1.0,n.activation+0.8); n.last_active=datetime.now().isoformat()
                n.use_count+=1; activated.append(n)
        for n in self.nodes.values():
            if n not in activated: n.activation*=0.88
        return activated

    def get_active_labels(self)->List[str]:
        return [n.label for n in self.nodes.values() if n.activation>0.3]

    def link_event(self, event:EpisodicMemory, nodes:List[ChronoNode]):
        for n in nodes:
            if event.id not in n.linked_events:
                n.linked_events.append(event.id)
                if len(n.linked_events)>60: n.linked_events=n.linked_events[-60:]


# ══════════════════════════════════════════
# НЕЙРО-КОРА (со стохастическим дрейфом)
# ══════════════════════════════════════════
@dataclass
class NeuroNode:
    id:str; label:str; category:str; activation:float=0.0
    last_active:str=field(default_factory=lambda:datetime.now().isoformat())
    created_at:str=field(default_factory=lambda:datetime.now().isoformat())
    access_count:int=0; experience:float=0.0; temporal_weight:float=1.0


@dataclass
class NeuroSynapse:
    source:str; target:str; weight:float=0.5; plasticity:float=0.8
    temporal_decay:float=0.0; last_fired:str=""; fire_count:int=0


class DynamicNeuralCortex:
    def __init__(self, user_id:str, save_dir:str):
        self.user_id=user_id; self.save_path=os.path.join(save_dir,"cortex_graph.json")
        self.nodes:Dict[str,NeuroNode]={}; self.synapses:Dict[str,NeuroSynapse]={}
        self.chrono_layer=ChronoNeuralLayer()
        self.drift=StochasticDrift()   # ← новое в v20
        data=FileManager.safe_load_json(self.save_path)
        if data and 'nodes' in data: self._load_with_decay(data)
        else:
            self._create_node("Приветствие","action"); self._create_node("Помощь","action"); self.save()

    def _load_with_decay(self, data:Dict):
        now_iso=datetime.now().isoformat()
        for n in data.get('nodes',[]):
            n.setdefault('experience',0.0); n.setdefault('temporal_weight',1.0); n.setdefault('last_active',now_iso)
            decayed=ActivationDecay.apply(n.get('activation',0.0),n['last_active'])
            tonic=ActivationDecay.tonic(n['experience'])
            n['activation']=max(decayed,tonic)
            self.nodes[n['id']]=NeuroNode(**n)
        for s in data.get('synapses',[]):
            s.setdefault('fire_count',0); k=f"{s['source']}->{s['target']}"
            self.synapses[k]=NeuroSynapse(**s)
        if 'chrono_nodes' in data: self.chrono_layer.restore_from_saved(data['chrono_nodes'])
        warm=sum(1 for n in self.nodes.values() if n.activation>0.1)
        print(f"🧠 Кора: {len(self.nodes)} нейронов ({warm} тёплых), {len(self.synapses)} синапсов")

    def _create_node(self, label:str, category:str="concept") -> NeuroNode:
        nid=hashlib.md5(label.encode()).hexdigest()[:8]
        if nid not in self.nodes: self.nodes[nid]=NeuroNode(id=nid,label=label,category=category)
        return self.nodes[nid]

    def process_input(self, text:str, now:datetime=None) -> Tuple[List[str],List[ChronoNode]]:
        if now is None: now=datetime.now()
        keywords=TextUtils.extract_keywords(text,top_n=10)
        activated_labels=[]; now_iso=now.isoformat()
        for i,kw1 in enumerate(keywords):
            n1=self._create_node(kw1); n1.access_count+=1; n1.activation=1.0
            n1.last_active=now_iso; n1.experience+=0.12; activated_labels.append(kw1)
            for kw2 in keywords[i+1:]: self._strengthen(n1.id,self._create_node(kw2).id,0.1)
        chrono_nodes=self.chrono_layer.process_temporal_input(text)
        final_context=set(activated_labels)
        for label in activated_labels:
            nid=hashlib.md5(label.encode()).hexdigest()[:8]
            for key,syn in list(self.synapses.items()):
                if syn.source==nid and syn.weight>0.4:
                    target=self.nodes.get(syn.target)
                    if target:
                        # Добавляем шум дрейфа при распространении
                        noisy_act=self.drift.add_noise(0.45)
                        target.activation=min(1.0,target.activation+noisy_act)
                        target.last_active=now_iso; final_context.add(target.label); syn.fire_count+=1
        for node in self.nodes.values():
            tonic=ActivationDecay.tonic(node.experience)
            node.activation=max(node.activation*0.88,tonic)
        # Случайное исследование (exploration)
        explored=self.drift.maybe_explore(self.nodes)
        # Случайная ассоциация
        wild=self.drift.get_wild_association(self.nodes,self.synapses)
        return list(final_context), chrono_nodes, explored, wild

    def _strengthen(self, src:str, tgt:str, reward:float):
        if src==tgt: return
        for k,(s,t) in [(f"{src}->{tgt}",(src,tgt)),(f"{tgt}->{src}",(tgt,src))]:
            if k not in self.synapses: self.synapses[k]=NeuroSynapse(source=s,target=t,weight=0.1)
            syn=self.synapses[k]; syn.weight=min(1.0,syn.weight+reward*syn.plasticity)
            syn.last_fired=datetime.now().isoformat(); syn.plasticity*=0.995; syn.fire_count+=1

    def associate(self, concept:str, top_n:int=6) -> List[Tuple[str,float]]:
        nid=hashlib.md5(concept.encode()).hexdigest()[:8]
        return sorted([(self.nodes[s.target].label,s.weight)
                       for k,s in self.synapses.items()
                       if s.source==nid and s.weight>0.15 and s.target in self.nodes],
                      key=lambda x:-x[1])[:top_n]

    def get_hot_nodes(self, top_n:int=8) -> List[NeuroNode]:
        return sorted([n for n in self.nodes.values() if n.activation>0.05],
                      key=lambda n:n.activation,reverse=True)[:top_n]

    def get_experienced_nodes(self, top_n:int=5) -> List[NeuroNode]:
        return sorted(self.nodes.values(),key=lambda n:n.experience,reverse=True)[:top_n]

    def reinforce_path(self, keywords:List[str], chrono_nodes:List[ChronoNode], success:bool):
        reward=0.15 if success else -0.04
        for i,kw1 in enumerate(keywords):
            id1=hashlib.md5(kw1.encode()).hexdigest()[:8]
            for kw2 in keywords[i+1:]: self._strengthen(id1,hashlib.md5(kw2.encode()).hexdigest()[:8],reward)
        for cn in chrono_nodes: cn.activation=min(1.0,cn.activation+reward)

    def prune_weak_synapses(self, threshold:float=0.04) -> int:
        before=len(self.synapses)
        self.synapses={k:v for k,v in self.synapses.items() if v.weight>threshold}
        return before-len(self.synapses)

    def save(self):
        FileManager.safe_save_json(self.save_path,{
            'nodes':[asdict(n) for n in self.nodes.values()],
            'synapses':[asdict(s) for s in self.synapses.values()],
            'chrono_nodes':self.chrono_layer.to_save_list(),
            'saved_at':datetime.now().isoformat()})

    def get_stats(self) -> Dict:
        return {'neurons':len(self.nodes),
                'warm':sum(1 for n in self.nodes.values() if n.activation>0.1),
                'synapses':len(self.synapses),
                'strong_syn':sum(1 for s in self.synapses.values() if s.weight>0.6),
                'chrono':len(self.chrono_layer.nodes),
                'active_chrono':sum(1 for n in self.chrono_layer.nodes.values() if n.activation>0.3)}


# ══════════════════════════════════════════
# РЕФЛЕКСИЯ
# ══════════════════════════════════════════
class ReflectionEngine:
    INTERVAL=15
    def __init__(self,llm): self.llm=llm; self._counter=0
    def tick(self)->bool: self._counter+=1; return self._counter%self.INTERVAL==0

    async def reflect(self, memory:'HybridMemorySystem') -> Optional[str]:
        chains=memory.causal.get_strong_chains(4); hot=memory.cortex.get_hot_nodes(5)
        if not chains and not hot: return None
        prompt=(
            "Ты — аналитик. Опиши 1-2 наблюдения об интересах/паттернах пользователя (≤40 слов):\n"
            f"Профиль: {memory.profile.get_prompt_block() or 'нет данных'}\n"
            f"Горячие темы: {[n.label for n in hot]}\n"
            f"Цепочки: {chains}\n"
            f"L3 паттерны: {memory.semantic_l3.facts}\n"
            "Только факты."
        )
        result=await self.llm.generate(prompt,temp=0.4)
        if result and len(result)>8: memory.meta.add_insight(result); return result
        return None


# ══════════════════════════════════════════
# ГИБРИДНАЯ ПАМЯТЬ
# ══════════════════════════════════════════
class HybridMemorySystem:
    def __init__(self, user_id:str):
        self.user_id=user_id; self.dir=os.path.join(MEMORY_DIR,f"user_{user_id}")
        os.makedirs(self.dir,exist_ok=True)
        self.st_file=os.path.join(self.dir,"short_term.json")
        self.short_term:List[Dict]=FileManager.safe_load_json(self.st_file,[])
        self.cortex    =DynamicNeuralCortex(user_id,self.dir)
        self.temporal  =TemporalMemory(user_id,self.dir)
        self.causal    =LogicalChainEngine(os.path.join(self.dir,"causal.json"))
        self.clusters  =SemanticClusterEngine(os.path.join(self.dir,"clusters.json"))
        self.meta      =MetaLearner(os.path.join(self.dir,"meta.json"))
        self.profile   =ProfileManager(os.path.join(self.dir,"profile.json"))
        self.emotion   =EmotionalArc(os.path.join(self.dir,"emotions.json"))
        self.predictor =TopicPredictor(os.path.join(self.dir,"topic_transitions.json"))
        self.semantic_l3=SemanticMemoryL3(os.path.join(self.dir,"semantic_l3.json"))
        self.now=datetime.now(); self._msg_count=0; self._prev_cluster:Optional[str]=None

    async def process(self, text:str):
        self.now=datetime.now()
        # 1. Нейро-активация (теперь возвращает explored и wild)
        concepts, chrono_nodes, explored, wild = self.cortex.process_input(text,self.now)
        # 2. Кластеры
        keywords=TextUtils.extract_keywords(text)
        cluster_name=self.clusters.learn(keywords); clusters=self.clusters.classify(keywords)
        # 3. Переход тем
        if self._prev_cluster and cluster_name: self.predictor.record_transition(self._prev_cluster,cluster_name)
        if cluster_name: self._prev_cluster=cluster_name
        # 4. Логические цепочки
        causal_links=self.causal.extract_from_text(text)
        xp_gain=3+len(causal_links)*5
        xp_msg=self.meta.gain_xp(xp_gain,"chain_found" if causal_links else "interaction",
                                  str([f"{l.cause}→{l.effect}" for l in causal_links]))
        # 5. Профиль
        new_facts=self.profile.update_from_text(text)
        # 6. Эмоция
        emotion_str, emotion_score=self.emotion.record(text)
        session_emotion, mood_hint=self.emotion.get_session_mood()
        # 7. L2 эпизоды
        event=self.temporal.add_event(text,keywords,cluster_name,self.now)
        self.cortex.chrono_layer.link_event(event,chrono_nodes)
        # 8. L3 семантика
        self.semantic_l3.record(cluster_name,self.now)
        # 9. RAG v2 — BM25
        rag_results=RAGEngineV2.search(text,self.temporal.events,top_k=RAG_TOP_K,now=self.now)
        rag_block  =RAGEngineV2.format_for_prompt(rag_results)
        # 10. Дрейф — блок для промпта
        drift_hint=self.cortex.drift.describe_drift(explored,wild)
        # 11. Адаптивная температура
        temp=TemperatureAdapter.get(cluster_name,session_emotion,text)
        # 12. Собираем промпт (L1+L2+L3)
        profile_block =self.profile.get_prompt_block()
        l3_block      =self.semantic_l3.get_prompt_block()
        chains        =self.causal.get_strong_chains(3)
        insights      =self.meta.get_insights_prompt(2)
        prediction    =self.predictor.get_hint(cluster_name or "")
        hot_labels    =[n.label for n in self.cortex.get_hot_nodes(5)]
        chrono_active =self.cortex.chrono_layer.get_active_labels()
        # L1 — рабочая память (история)
        l1_ctx="\n".join([f"{m['role']}: {m['content']}" for m in self.short_term[-6:]])
        # Системная часть
        sys_parts=[]
        if profile_block: sys_parts.append(f"=== ПРОФИЛЬ ===\n{profile_block}")
        if l3_block:      sys_parts.append(l3_block)
        if hot_labels:    sys_parts.append(f"Горячие темы: {', '.join(hot_labels)}.")
        if chrono_active: sys_parts.append(f"Временной контекст: {', '.join(chrono_active)}.")
        if chains:        sys_parts.append(f"Закономерности: {'; '.join(chains)}.")
        if clusters:      sys_parts.append(f"Тема: {clusters[0][0]}.")
        if mood_hint:     sys_parts.append(mood_hint)
        if prediction:    sys_parts.append(prediction)
        if drift_hint:    sys_parts.append(drift_hint)
        if insights:      sys_parts.append(insights)
        sys_mod="\n".join(sys_parts) if sys_parts else "Ты умный ассистент с долговременной памятью."
        # Полный промпт: sys + L2 (RAG) + L1 (история)
        full_system=(
            f"{sys_mod}\n\n"
            f"{rag_block}\n\n" if rag_block else f"{sys_mod}\n\n"
        )+(f"=== ИСТОРИЯ (L1) ===\n{l1_ctx}\n\n"
           f"=== ВРЕМЯ ===\n{self.now.strftime('%H:%M %d.%m.%Y')}")
        # Обновить L1
        self.short_term.append({'role':'user','content':text,'time':self.now.isoformat()})
        if len(self.short_term)>20: self.short_term=self.short_term[-20:]
        # Автосохранение
        self._msg_count+=1
        if self._msg_count%AUTOSAVE_EVERY==0: self.save_all(); print(f"💾 Автосохранение #{self._msg_count}")
        return full_system, temp, concepts, chrono_nodes, xp_msg, new_facts

    def handle_temporal_query(self, text:str) -> Optional[str]:
        if not any(w in text.lower() for w in ['был','делал','ходил','где','вчера','раньше','утром','вечером']): return None
        events=self.temporal.query_by_time(text,self.now)
        if events:
            lines=[]
            for e in events[:4]:
                rel=TimeEncoder.describe(datetime.fromisoformat(e.timestamp),self.now)
                lines.append(f"{rel}: {'['+e.cluster_name+'] ' if e.cluster_name else ''}{e.content}")
            return "🕰️ Из памяти:\n"+"\n".join(lines)
        return None

    def handle_logic_query(self, text:str) -> Optional[str]:
        if not any(w in text.lower() for w in ['почему','зачем','причина','что будет','что вызывает']): return None
        for kw in TextUtils.extract_keywords(text):
            ch=self.causal.build_chain(kw)
            if len(ch)>1: return f"🔗 Цепочка: {' → '.join(ch)}"
            causes=self.causal.find_causes(kw)
            if causes: return "🔍 Причины «{}»:\n{}".format(kw,"\n".join(f"  • {c} ({s:.2f})" for c,s in causes[:3]))
        return None

    def add_response(self, text:str):
        self.short_term.append({'role':'assistant','content':text,'time':datetime.now().isoformat()})
        if len(self.short_term)>20: self.short_term=self.short_term[-20:]
        FileManager.safe_save_json(self.st_file,self.short_term)

    def save_all(self):
        self.cortex.save(); self.temporal.save(); self.causal.save(); self.clusters.save()
        self.meta.save(); self.profile.save(); self.emotion.save(); self.predictor.save()
        self.semantic_l3.save(); FileManager.safe_save_json(self.st_file,self.short_term)

    def maintenance(self):
        if self.meta.total_interactions%60==0:
            pruned=self.cortex.prune_weak_synapses(0.04); self.causal.decay_all()
            if pruned: print(f"✂️ Удалено {pruned} слабых синапсов")


# ══════════════════════════════════════════
# LLM
# ══════════════════════════════════════════
class LLMInterface:
    def __init__(self, url:str, key:str):
        self.url=url; self.key=key; self.session:Optional[aiohttp.ClientSession]=None

    async def init(self):
        if not self.session: self.session=aiohttp.ClientSession()

    async def generate(self, prompt:str, system:str=None, temp:float=TEMP_DEFAULT) -> str:
        if not self.session: await self.init()
        msgs=[];
        if system: msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        try:
            async with self.session.post(
                self.url,
                json={"messages":msgs,"temperature":temp,"max_tokens":1500},
                headers={"Authorization":f"Bearer {self.key}"}
            ) as r:
                if r.status==200: return (await r.json())['choices'][0]['message']['content']
                return f"LM Error: {r.status}"
        except Exception as e: return f"Connection error: {e}"

    async def close(self):
        if self.session: await self.session.close(); self.session=None


# ══════════════════════════════════════════
# БОТ
# ══════════════════════════════════════════
class HybridBot:
    def __init__(self):
        self.llm=LLMInterface(LM_STUDIO_API_URL,LM_STUDIO_API_KEY)
        self.users:Dict[str,HybridMemorySystem]={}; self.reflector=None; self.stop_flag=False

    def get_brain(self, uid:str) -> HybridMemorySystem:
        if uid not in self.users: self.users[uid]=HybridMemorySystem(uid)
        return self.users[uid]

    async def handle_message(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text: return
        uid=str(update.effective_user.id); text=update.message.text
        brain=self.get_brain(uid)
        await context.bot.send_chat_action(uid,"typing")
        full_system,temp,concepts,chrono_nodes,xp_msg,new_facts=await brain.process(text)
        ta=brain.handle_temporal_query(text)
        if ta:
            brain.add_response(ta); await update.message.reply_text(ta)
            if xp_msg: await update.message.reply_text(xp_msg)
            return
        la=brain.handle_logic_query(text)
        if la:
            brain.add_response(la); await update.message.reply_text(la); return
        # Генерация с адаптивной температурой
        response=await self.llm.generate(text,system=full_system,temp=temp)
        brain.cortex.reinforce_path(concepts,chrono_nodes,len(response)>10)
        if self.reflector and self.reflector.tick():
            asyncio.create_task(self._reflect(brain))
        brain.maintenance(); brain.add_response(response)
        await update.message.reply_text(response)
        for msg in filter(None,[xp_msg, f"👤 Запомнил: {', '.join(new_facts)}" if new_facts else None]):
            await update.message.reply_text(msg)

    async def _reflect(self, brain:HybridMemorySystem):
        try:
            ins=await self.reflector.reflect(brain)
            if ins: print(f"💡 Инсайт: {ins[:80]}")
        except Exception as e: print(f"⚠️ Reflection: {e}")

    # ═══ КОМАНДЫ ═══

    async def cmd_stats(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        s=brain.cortex.get_stats(); chains=brain.causal.get_strong_chains(3)
        await update.message.reply_text(
            f"🧠 STOCHASTIC MIND v20.0\n{'═'*32}\n"
            f"🔹 Нейроны: {s['neurons']} (тёплых: {s['warm']})\n"
            f"🔹 Синапсы: {s['synapses']} (сильных: {s['strong_syn']})\n"
            f"🔹 Chrono: {s['chrono']} (активных: {s['active_chrono']})\n"
            f"🔹 Эпизодов L2: {len(brain.temporal.events)}\n"
            f"🔹 Фактов L3: {len(brain.semantic_l3.facts)}\n"
            f"🔹 Причинных связей: {len(brain.causal.links)}\n"
            f"🔹 Кластеров: {len(brain.clusters.clusters)}\n"
            f"{'═'*32}\n"
            f"📈 {brain.meta.get_status()}\n"
            f"{'═'*32}\n"
            f"🔗 Цепочки:\n  "+("\n  ".join(chains) if chains else "Пока нет"))

    async def cmd_hotmap(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        hot=brain.cortex.get_hot_nodes(10); exp=brain.cortex.get_experienced_nodes(5)
        if not hot: await update.message.reply_text("Горячих нейронов пока нет."); return
        lines=["🔥 ГОРЯЧАЯ КАРТА\n","Активные сейчас:"]
        for n in hot[:6]:
            bar="█"*int(n.activation*10)
            lines.append(f"  {n.label:<15} {bar} {n.activation:.2f} (exp:{n.experience:.1f})")
        lines.append("\n🏆 Самые опытные:")
        for n in exp:
            lines.append(f"  {n.label:<15} exp:{n.experience:.1f}  тонус:{ActivationDecay.tonic(n.experience):.3f}")
        chrono_hot=sorted([n for n in brain.cortex.chrono_layer.nodes.values() if n.activation>0.05],
                          key=lambda x:-x.activation)[:5]
        if chrono_hot:
            lines.append("\n⏱ Временные нейроны:")
            for n in chrono_hot: lines.append(f"  {n.label:<12} act:{n.activation:.2f} uses:{n.use_count}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_profile(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        l3=brain.semantic_l3.get_prompt_block()
        await update.message.reply_text(
            f"👤 ПРОФИЛЬ\n{'═'*24}\n{brain.profile.get_status()}\n\n{l3 or 'L3 пока пуст'}")

    async def cmd_mood(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        days=int(context.args[0]) if context.args and context.args[0].isdigit() else 7
        t=brain.emotion.get_trend()
        await update.message.reply_text(
            f"💚 ЭМОЦИОНАЛЬНАЯ ДУГА\n{'═'*24}\n"
            f"{brain.emotion.get_history_summary(days)}\n"
            f"Тренд: {t['trend']} | Направление: {t['direction']}")

    async def cmd_chain(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /chain <слово>"); return
        word=context.args[0].lower(); chain=brain.causal.build_chain(word)
        if len(chain)>1: await update.message.reply_text(f"🔗 Цепочка '{word}':\n{' → '.join(chain)}")
        else:
            causes=brain.causal.find_causes(word)
            if causes: await update.message.reply_text(f"🔍 Причины '{word}':\n"+"\n".join(f"  {c} ({s:.2f})" for c,s in causes))
            else: await update.message.reply_text(f"Цепочек для '{word}' пока нет.")

    async def cmd_assoc(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /assoc <слово>"); return
        word=context.args[0].lower(); assocs=brain.cortex.associate(word)
        if assocs: await update.message.reply_text(f"🧩 Ассоциации '{word}':\n"+"\n".join(f"  {a} ({w:.2f})" for a,w in assocs))
        else: await update.message.reply_text(f"Ассоциаций для '{word}' нет.")

    async def cmd_predict(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        if not context.args: await update.message.reply_text("Использование: /predict <тема>"); return
        topic=" ".join(context.args).lower(); preds=brain.predictor.predict_next(topic,top_n=3)
        if preds:
            lines=[f"🔮 После темы «{topic}» вероятно:"]
            for t,prob in preds: lines.append(f"  {t:<15} {'█'*int(prob*10)} {prob:.0%}")
            await update.message.reply_text("\n".join(lines))
        else: await update.message.reply_text(f"Данных о переходах от «{topic}» ещё нет.")

    async def cmd_drift(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        """Показать параметры стохастического дрейфа"""
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        hot=brain.cortex.get_hot_nodes(3)
        candidates=[n for n in brain.cortex.nodes.values() if 0.01<n.activation<0.15 and n.experience>0.3]
        mid_syn=[s for s in brain.cortex.synapses.values() if 0.2<s.weight<0.5 and s.fire_count>1]
        await update.message.reply_text(
            f"🎲 СТОХАСТИЧЕСКИЙ ДРЕЙФ\n{'═'*24}\n"
            f"Вероятность exploration: {DRIFT_EXPLORE_PROB:.0%}\n"
            f"Масштаб шума: {DRIFT_NOISE_SCALE}\n"
            f"Wild-ассоциация каждые: {DRIFT_ASSOC_EVERY} шагов\n\n"
            f"Кандидаты для exploration: {len(candidates)} нейронов\n"
            f"Периферийных синапсов: {len(mid_syn)}\n"
            f"Текущие горячие: {', '.join(n.label for n in hot)}")

    async def cmd_timeline(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        hours=int(context.args[0]) if context.args and context.args[0].isdigit() else 24
        await update.message.reply_text(f"🕰️ События за {hours}ч:\n{brain.temporal.get_timeline_summary(hours)}")

    async def cmd_clusters(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        top=sorted(brain.clusters.clusters.values(),key=lambda c:c.access_count,reverse=True)[:8]
        if not top: await update.message.reply_text("Кластеры пока не сформированы."); return
        lines=["🗂 КЛАСТЕРЫ ЗНАНИЙ\n"]
        for c in top:
            lines.append(f"📌 [{c.name}] ({c.access_count} обращений)")
            lines.append(f"   {', '.join(c.members[:7])}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_insights(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id); brain=self.get_brain(uid)
        if not brain.meta.insights: await update.message.reply_text("Инсайтов пока нет."); return
        lines=["💡 ИНСАЙТЫ\n"]
        for i,ins in enumerate(brain.meta.insights[-10:],1): lines.append(f"{i}. {ins}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_wipe(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        uid=str(update.effective_user.id)
        if uid in self.users:
            d=self.users.pop(uid).dir
            if os.path.exists(d): shutil.rmtree(d)
            await update.message.reply_text("🧠 Полная очистка выполнена.")
        else: await update.message.reply_text("Пользователь не найден.")

    async def cmd_help(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 STOCHASTIC MIND v20.0\n\n"
            "Трёхуровневая память + BM25 RAG +\n"
            "стохастический дрейф + адаптивная температура.\n\n"
            "📌 КОМАНДЫ:\n"
            "/stats          — статистика нейросети\n"
            "/hotmap         — карта горячих нейронов 🔥\n"
            "/profile        — профиль и L3 память 👤\n"
            "/mood [дней]    — эмоциональная дуга 💚\n"
            "/predict <тема> — предсказание темы 🔮\n"
            "/drift          — параметры дрейфа 🎲\n"
            "/chain <слово>  — логическая цепочка\n"
            "/assoc <слово>  — ассоциации\n"
            "/clusters       — смысловые кластеры\n"
            "/insights       — накопленные инсайты\n"
            "/timeline [ч]   — хронология\n"
            "/wipe           — очистка памяти\n\n"
            "💡 Подсказки:\n"
            "• «Меня зовут Иван, я программист» → /profile\n"
            "• «X вызывает Y» → /chain X\n"
            "• «Почему X?» → автоматически строит цепочку\n"
            "• Случайные воспоминания в промпте — это дрейф!\n"
            "• Температура LLM меняется автоматически по теме"
        )

    async def shutdown(self):
        print("\n💾 Финальное сохранение...")
        for b in self.users.values(): b.save_all()
        await self.llm.close(); print("✅ Остановлено")


# ══════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════
async def main():
    print("🚀 STOCHASTIC MIND v20.0 STARTING...")
    if not TELEGRAM_TOKEN: print("❌ Нет TELEGRAM_TOKEN в .env"); return
    bot=HybridBot(); bot.reflector=ReflectionEngine(bot.llm)
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    for cmd,handler in [
        ("stats",bot.cmd_stats),("hotmap",bot.cmd_hotmap),("profile",bot.cmd_profile),
        ("mood",bot.cmd_mood),("predict",bot.cmd_predict),("drift",bot.cmd_drift),
        ("timeline",bot.cmd_timeline),("chain",bot.cmd_chain),("assoc",bot.cmd_assoc),
        ("clusters",bot.cmd_clusters),("insights",bot.cmd_insights),
        ("wipe",bot.cmd_wipe),("help",bot.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd,handler))
    try:
        await app.initialize(); await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print("✅ STOCHASTIC MIND ГОТОВ 🧠🎲")
        print("💡 /drift — посмотреть параметры случайности")
        while not bot.stop_flag: await asyncio.sleep(1)
    except KeyboardInterrupt: print("\n🛑 Остановка")
    finally:
        await app.updater.stop(); await app.stop(); await app.shutdown(); await bot.shutdown()


if __name__=="__main__":
    try: asyncio.run(main())
    except Exception as e: print(f"❌ Crash: {e}"); traceback.print_exc()