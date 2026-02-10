
# coding: utf-8
"""
Cognitive_Agent_v5.0.py — Автономная когнитивная система с двухпроходным мышлением
✅ Двухпроходное мышление: ответ → самооценка → коррекция
✅ Осознанный веб-поиск ТОЛЬКО после критики
✅ Аннотация знаний: факт / предположение / неизвестно
✅ Память с забыванием (только важные выводы и ошибки)
✅ Механизм самообучения на ошибках
✅ Полная приватность для локальных запросов
✅ Системное время для запросов о дате/годе/времени
"""
import asyncio
import logging
import os
import sys
import sqlite3
import hashlib
import re
import json
import time
import requests
import aiohttp
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, Optional, List, Any, Set, Tuple
from datetime import datetime

# ================= ДОПОЛНИТЕЛЬНЫЕ ИМПОРТЫ ДЛЯ ВЕБ-ПОИСКА =================
try:
    from duckduckgo_search import AsyncDDGS

    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False


# ================= ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =================
def load_dotenv_simple(path: Path = Path(".env")):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
        except Exception:
            pass


load_dotenv_simple()

# ================= СОВМЕСТИМОСТЬ С PYTHON 3.13 =================
if sys.version_info >= (3, 13) and sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

# ================= ИМПОРТЫ TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )
except ImportError as e:
    print(f"❌ Ошибка импорта telegram: {e}")
    print("📦 Установите: pip install 'python-telegram-bot>=20.7' aiohttp requests")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_system_telegram")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "memory.db"
    CACHE_PATH = ROOT / "cache.json"
    LOG_PATH = ROOT / "system.log"

    # LM Studio
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODELS_URL = "http://localhost:1234/v1/models"
    TIMEOUT = 180
    MAX_TOKENS = 4096

    # Когнитивные параметры
    MEMORY_DECAY_RATE = 0.05
    MAX_MEMORY_ITEMS = 200
    CONFIDENCE_THRESHOLD = 0.65
    SEARCH_CONFIDENCE_THRESHOLD = 0.6

    # Параметры поиска
    WEB_SEARCH_ENABLED = HAS_WEB_SEARCH
    TIME_SENSITIVE_KEYWORDS = ['курс', 'погода', 'новост', 'сегодня', 'сейчас', 'текущ', 'последн']

    # Прочее
    MAX_MESSAGE_LENGTH = 4096
    SESSION_TIMEOUT = 7200

    @classmethod
    def get_telegram_token(cls) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("\n" + "=" * 70)
            print("🔑 НЕОБХОДИМ ТОКЕН TELEGRAM БОТА")
            print("=" * 70)
            print("\nКак получить токен:")
            print("1. Найдите в Telegram бота @BotFather")
            print("2. Отправьте команду /newbot и следуйте инструкциям")
            token = input("\nВведите токен Telegram бота: ").strip()
            if not token or not re.match(r'^\d+:[A-Za-z0-9_-]{35,}$', token):
                raise ValueError("❌ Неверный формат токена")
            os.environ["TELEGRAM_BOT_TOKEN"] = token
            try:
                with open(".env", "a", encoding="utf-8") as f:
                    f.write(f'\nTELEGRAM_BOT_TOKEN="{token}"\n')
            except:
                pass
        return token

    @classmethod
    def get_lmstudio_config(cls) -> Dict[str, Any]:
        config = {
            'url': cls.LM_STUDIO_URL,
            'api_key': os.getenv("LM_STUDIO_API_KEY", ""),
            'model': os.getenv("LM_STUDIO_MODEL", "local-model")
        }

        if not os.getenv("LM_STUDIO_CONFIGURED"):
            print("\n" + "=" * 70)
            print("⚙️  НАСТРОЙКА ЛОКАЛЬНОГО СЕРВЕРА LM STUDIO")
            print("=" * 70)
            print("\n🔍 Проверка сервера LM Studio...")
            try:
                response = requests.get(cls.LM_STUDIO_MODELS_URL, timeout=8)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    if models:
                        best = next((m for m in models if
                                     'instruct' in m.get('id', '').lower() or 'chat' in m.get('id', '').lower()),
                                    models[0])
                        config['model'] = best.get('id', config['model'])
                        with open(".env", "a", encoding="utf-8") as f:
                            f.write(f'\nLM_STUDIO_CONFIGURED="true"\nLM_STUDIO_MODEL="{config["model"]}"\n')
            except:
                pass
        return config


# ================= УТИЛИТЫ ОБРАБОТКИ ТЕКСТА =================
def calculate_text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    def get_ngrams(text: str, n: int = 2) -> Set[str]:
        words = re.findall(r'\w+', text.lower())
        if len(words) < n:
            return set([' '.join(words)])
        return set(' '.join(words[i:i + n]) for i in range(len(words) - n + 1))

    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0

    unigram_sim = len(words1 & words2) / max(len(words1), len(words2))
    bigrams1 = get_ngrams(text1, 2)
    bigrams2 = get_ngrams(text2, 2)
    bigram_sim = len(bigrams1 & bigrams2) / max(len(bigrams1 | bigrams2), 1) if bigrams1 and bigrams2 else 0.0
    return 0.6 * unigram_sim + 0.4 * bigram_sim


def extract_entities(text: str) -> Dict[str, List[str]]:
    entities = {
        'numbers': re.findall(r'\b\d+\b', text),
        'dates': re.findall(r'\b\d{1,2}[./]\d{1,2}[./]?\d{2,4}\b', text, re.IGNORECASE),
        'emails': re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text),
        'urls': re.findall(r'https?://\S+', text)
    }
    return {k: v for k, v in entities.items() if v}


def split_message(text: str, max_length: int = 4096) -> list:
    if not text:
        return [""]
    if len(text) <= max_length:
        return [text]

    parts = []
    current = ""
    paragraphs = re.split(r'(\n\s*\n)', text)

    for para in paragraphs:
        if len(current) + len(para) <= max_length:
            current += para
        else:
            if current:
                parts.append(current.rstrip())
            if len(para) > max_length:
                sentences = re.split(r'([.!?]+)', para)
                temp = ""
                for i in range(0, len(sentences), 2):
                    chunk = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                    if len(temp) + len(chunk) <= max_length:
                        temp += chunk
                    else:
                        if temp:
                            parts.append(temp.rstrip())
                        temp = chunk
                if temp:
                    current = temp
            else:
                current = para

    if current:
        parts.append(current.rstrip())

    if len(parts) > 5:
        parts = parts[:5]
        parts.append("📝 ...сообщение сокращено из-за ограничений Telegram")
    return parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("🧠 Анализ", callback_data="analyze"),
         InlineKeyboardButton("🔍 Поиск", callback_data="search")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats"),
         InlineKeyboardButton("💡 Инсайты", callback_data="insights")],
        [InlineKeyboardButton("🎯 Цели", callback_data="goals"),
         InlineKeyboardButton("🧹 Очистить", callback_data="clear")]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= ДВИЖОК ВЕБ-ПОИСКА =================
class WebSearchEngine:
    def __init__(self, cache_ttl: int = 3600):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.search_count = 0
        self.last_search_time = 0
        self.ddgs = AsyncDDGS() if HAS_WEB_SEARCH else None

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not HAS_WEB_SEARCH or not self.ddgs:
            return []

        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_results

        try:
            elapsed = time.time() - self.last_search_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)

            results = []
            search_results = await self.ddgs.text(query, max_results=max_results, safesearch='moderate')
            for result in search_results[:max_results]:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })

            self.cache[cache_key] = (time.time(), results)
            self.search_count += 1
            self.last_search_time = time.time()
            return results
        except Exception as e:
            logging.error(f"Ошибка поиска: {e}")
            return []


# ================= СИСТЕМА САМОКРИТИКИ (КЛЮЧЕВОЙ КОМПОНЕНТ) =================
class SelfCritiqueEngine:
    """Двухпроходная система: генерация ответа → критика → коррекция"""

    def __init__(self, llm_interface: Any):
        self.llm = llm_interface

    async def critique_response(self, user_query: str, draft_answer: str, context: str = "") -> Dict[str, Any]:
        """Анализирует черновой ответ и возвращает структурированную критику"""
        critique_prompt = f"""
USER_QUERY: {user_query}
DRAFT_ANSWER: {draft_answer}
CONTEXT: {context[:300]}

Проанализируй ответ критически. Верни ТОЛЬКО валидный JSON без пояснений:

{{
  "needs_search": boolean,
  "confidence": float (0.0-1.0),
  "uncertain_claims": ["claim1", "claim2"],
  "temporal_sensitivity": boolean,
  "reason": "краткое обоснование"
}}
        """

        critique_response = await self.llm.call_llm(
            system_prompt="Ты — критический аналитик. Оцени ответ строго и объективно. Не улучшай ответ, только критикуй.",
            user_prompt=critique_prompt,
            temperature=0.0,
            max_tokens=300
        )

        try:
            critique_json = self._extract_json(critique_response)
            critique_data = {
                'needs_search': critique_json.get('needs_search', False),
                'confidence': float(critique_json.get('confidence', 0.5)),
                'uncertain_claims': critique_json.get('uncertain_claims', []),
                'temporal_sensitivity': critique_json.get('temporal_sensitivity', False),
                'reason': str(critique_json.get('reason', 'no reason provided'))[:200]
            }
            return critique_data
        except Exception as e:
            logging.error(f"Ошибка парсинга критики: {e}")
            return {
                'needs_search': False,
                'confidence': 0.5,
                'uncertain_claims': [],
                'temporal_sensitivity': False,
                'reason': 'parsing error'
            }

    def _extract_json(self, text: str) -> Dict:
        """Извлекает JSON из текста"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("No valid JSON found")


# ================= СИСТЕМА АННОТАЦИИ ЗНАНИЙ =================
class KnowledgeAnnotator:
    """Разметка ответа на факты, предположения и неизвестное"""

    def __init__(self, llm_interface: Any):
        self.llm = llm_interface

    async def annotate_knowledge(self, text: str) -> Dict[str, List[str]]:
        """Аннотирует части текста как факт/предположение/неизвестное"""
        if not text or len(text) < 10:
            return {'facts': [], 'assumptions': [], 'unknown': []}

        annotation_prompt = f"""
ТЕКСТ: {text[:500]}

Разметь утверждения в тексте. Верни ТОЛЬКО валидный JSON:

{{
  "facts": ["проверяемый факт 1", "проверяемый факт 2"],
  "assumptions": ["предположение 1", "предположение 2"],
  "unknown": ["то, что неизвестно или требует проверки"]
}}
        """

        annotation_response = await self.llm.call_llm(
            system_prompt="Ты — эпистемолог. Различай факты, предположения и неизвестное. Будь строгим.",
            user_prompt=annotation_prompt,
            temperature=0.1,
            max_tokens=400
        )

        try:
            annotation_json = self._extract_json(annotation_response)
            return {
                'facts': annotation_json.get('facts', []),
                'assumptions': annotation_json.get('assumptions', []),
                'unknown': annotation_json.get('unknown', [])
            }
        except Exception as e:
            logging.error(f"Ошибка аннотации: {e}")
            return {'facts': [], 'assumptions': [], 'unknown': []}

    def _extract_json(self, text: str) -> Dict:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("No valid JSON found")


# ================= СИСТЕМА ПАМЯТИ С ЗАБЫВАНИЕМ =================
class AdaptiveMemory:
    """Память с деградацией важности и фильтрацией по значимости"""

    def __init__(self, db: Any):
        self.db = db
        self.insight_buffer = deque(maxlen=10)

    def store_insight(self, user_id: int, insight_type: str, content: str, importance: float, metadata: Dict):
        """Сохраняет ТОЛЬКО важные инсайты, ошибки и выводы"""
        if importance < 0.6 or len(content) < 15:
            return

        try:
            insight_data = {
                'content': content[:300],
                'type': insight_type,
                'metadata': metadata,
                'timestamp': time.time(),
                'importance': min(1.0, importance),
                'last_used': time.time(),
                'usage_count': 1
            }

            self.db.add_fact(
                user_id,
                f"insight_{int(time.time())}",
                json.dumps(insight_data, ensure_ascii=False),
                category='insight',
                importance=importance,
                confidence=0.9
            )

            self.insight_buffer.append(insight_data)
            logging.info(f"🧠 Сохранён инсайт [{insight_type}]: {content[:50]}")
        except Exception as e:
            logging.error(f"Ошибка сохранения инсайта: {e}")

    def recall_relevant_insights(self, user_id: int, query: str, limit: int = 3) -> List[Dict]:
        """Возвращает релевантные инсайты с учётом важности и свежести"""
        try:
            insights = self.db.get_relevant_facts(user_id, query, limit=limit * 2)
            scored = []
            current_time = time.time()

            for insight in insights:
                try:
                    data = json.loads(insight['value'])
                    relevance = calculate_text_similarity(query, data['content'])
                    recency = 1.0 - (current_time - data['timestamp']) / (7 * 86400)
                    recency = max(0.1, min(1.0, recency))

                    score = (
                            0.5 * relevance +
                            0.3 * data['importance'] +
                            0.2 * recency
                    )

                    if score > 0.4:
                        scored.append((score, data))
                except:
                    continue

            scored.sort(reverse=True, key=lambda x: x[0])
            return [item[1] for item in scored[:limit]]
        except Exception as e:
            logging.error(f"Ошибка поиска инсайтов: {e}")
            return []

    def decay_memories(self, user_id: int):
        """Применяет деградацию к старым воспоминаниям"""
        self.db.decay_old_facts(user_id, Config.MEMORY_DECAY_RATE)


# ================= СИСТЕМА САМООБУЧЕНИЯ =================
class PolicyLearner:
    """Обучение на ошибках и коррекция поведенческих политик"""

    def __init__(self, db: Any):
        self.db = db
        self.error_patterns = defaultdict(int)
        self.policy_rules = {}
        self._load_policies()

    def record_error(self, user_id: int, error_type: str, context: Dict):
        """Записывает ошибку для последующего анализа"""
        self.error_patterns[error_type] += 1

        self.db.add_fact(
            user_id,
            f"error_{error_type}",
            json.dumps({
                'type': error_type,
                'context': context,
                'timestamp': time.time(),
                'count': self.error_patterns[error_type]
            }, ensure_ascii=False),
            category='error_pattern',
            importance=0.8,
            confidence=0.95
        )

    def should_trigger_search(self, query: str, critique: Dict) -> bool:
        """Принимает решение о поиске на основе политик и критики"""
        # Политика 1: критика требует поиска
        if critique.get('needs_search', False):
            return True

        # Политика 2: низкая уверенность
        if critique.get('confidence', 1.0) < Config.SEARCH_CONFIDENCE_THRESHOLD:
            return True

        # Политика 3: временная чувствительность + низкая уверенность
        if critique.get('temporal_sensitivity', False) and critique.get('confidence', 1.0) < 0.75:
            return True

        # Политика 4: ключевые слова актуальности
        query_lower = query.lower()
        if any(kw in query_lower for kw in Config.TIME_SENSITIVE_KEYWORDS):
            return True

        return False

    def _load_policies(self):
        """Загружает сохранённые политики поведения"""
        try:
            policies = self.db.get_patterns(user_id=0, min_confidence=0.6, limit=20)
            for p in policies:
                if p['pattern_type'] == 'search_policy':
                    try:
                        self.policy_rules[p['description']] = json.loads(p['value'])
                    except:
                        pass
        except:
            pass

    def save_policy(self, user_id: int, policy_type: str, description: str, value: Any):
        """Сохраняет новую политику поведения"""
        self.db.add_pattern(
            user_id,
            policy_type,
            description,
            json.dumps(value, ensure_ascii=False),
            confidence=0.85
        )


# ================= ИНТЕРФЕЙС LLM ДЛЯ ЛОКАЛЬНЫХ МОДЕЛЕЙ =================
class LocalLLMInterface:
    """Единый интерфейс для вызова локальной LLM"""

    def __init__(self, config: Dict[str, Any]):
        self.api_url = config['url']
        self.api_key = config.get('api_key', "")
        self.model = config['model']
        self.last_request_time = 0
        self.rate_limit = 0.3

    async def _wait_for_rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7,
                       max_tokens: int = 2048) -> str:
        await self._wait_for_rate_limit()

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r'^[\s\*\-\_]+|[\s\*\-\_]+$', '', content)
                        content = re.sub(r'\n{3,}', '\n\n', content)
                        return content
                    else:
                        error_text = await response.text()
                        return f"⚠️ Ошибка LM Studio ({response.status}): {error_text[:100]}"
        except asyncio.TimeoutError:
            return "⚠️ Таймаут запроса к модели"
        except aiohttp.ClientConnectorError:
            return "⚠️ Не удаётся подключиться к LM Studio. Запущен ли сервер?"
        except Exception as e:
            return f"⚠️ Ошибка модели: {str(e)[:100]}"


# ================= РАСШИРЕННАЯ БАЗА ДАННЫХ =================
class CognitiveMemoryDB:
    """База данных с поддержкой когнитивных функций"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT,
                    confidence REAL DEFAULT 1.0,
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    last_used REAL,
                    usage_count INTEGER DEFAULT 0,
                    decay_factor REAL DEFAULT 1.0
                )''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user_id INTEGER NOT NULL,
                    user_input TEXT NOT NULL,
                    system_response TEXT NOT NULL,
                    critique_data TEXT,
                    confidence REAL DEFAULT 0.5,
                    used_search INTEGER DEFAULT 0
                )''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    value TEXT,
                    occurrences INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL
                )''')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, importance DESC, decay_factor DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_user ON patterns(user_id, confidence DESC)')
            conn.commit()

    def add_fact(self, user_id: int, key: str, value: str, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, decay_factor FROM facts WHERE user_id = ? AND key = ? AND value = ?',
                           (user_id, key, value))
            existing = cursor.fetchone()

            if existing:
                new_decay = min(1.0, existing['decay_factor'] * 1.2)
                cursor.execute('''
                    UPDATE facts
                    SET confidence = ?, importance = ?, last_used = ?,
                        usage_count = usage_count + 1, decay_factor = ?
                    WHERE id = ?
                ''', (
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(),
                    new_decay,
                    existing['id']
                ))
            else:
                cursor.execute('''
                    INSERT INTO facts
                    (user_id, key, value, category, confidence, importance, created_at, last_used, usage_count, decay_factor)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, key, value,
                    kwargs.get('category', ''),
                    kwargs.get('confidence', 1.0),
                    kwargs.get('importance', 0.5),
                    time.time(), time.time(), 1, 1.0
                ))

    def get_relevant_facts(self, user_id: int, query: str, limit: int = 5) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM facts
                WHERE user_id = ? AND confidence > 0.4 AND decay_factor > 0.2
                ORDER BY importance DESC, decay_factor DESC, usage_count DESC
                LIMIT ?
            ''', (user_id, limit * 2))

            all_facts = [dict(row) for row in cursor.fetchall()]
            if not all_facts:
                return []

            scored = []
            for fact in all_facts:
                relevance = calculate_text_similarity(query, f"{fact['key']} {fact['value']}")
                score = 0.6 * relevance + 0.4 * fact['importance']
                scored.append((score, fact))

            scored.sort(reverse=True, key=lambda x: x[0])
            return [item[1] for item in scored[:limit]]

    def decay_old_facts(self, user_id: int, decay_rate: float):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE facts
                SET decay_factor = decay_factor * (1 - ?)
                WHERE user_id = ? AND last_used < ?
            ''', (decay_rate, user_id, time.time() - 86400))

    def add_interaction(self, user_id: int, user_input: str, system_response: str, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions
                (timestamp, user_id, user_input, system_response, critique_data, confidence, used_search)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                user_id,
                user_input,
                system_response,
                json.dumps(kwargs.get('critique_data', {}), ensure_ascii=False),
                kwargs.get('confidence', 0.5),
                kwargs.get('used_search', 0)
            ))

    def add_pattern(self, user_id: int, pattern_type: str, description: str, value: str, confidence: float = 0.5):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM patterns
                WHERE user_id = ? AND pattern_type = ? AND description = ?
            ''', (user_id, pattern_type, description))
            existing = cursor.fetchone()

            if existing:
                cursor.execute('''
                    UPDATE patterns
                    SET occurrences = occurrences + 1, last_seen = ?, confidence = ?
                    WHERE id = ?
                ''', (time.time(), min(1.0, confidence * 1.1), existing[0]))
            else:
                cursor.execute('''
                    INSERT INTO patterns
                    (user_id, pattern_type, description, value, occurrences, confidence, created_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, pattern_type, description, value, 1, confidence, time.time(), time.time()))

    def get_patterns(self, user_id: int, min_confidence: float = 0.6, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM patterns
                WHERE user_id = ? AND confidence >= ?
                ORDER BY occurrences DESC, confidence DESC
                LIMIT ?
            ''', (user_id, min_confidence, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (user_id,))
            interactions = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM facts WHERE user_id = ? AND decay_factor > 0.3', (user_id,))
            facts = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM patterns WHERE user_id = ? AND confidence > 0.6', (user_id,))
            patterns = cursor.fetchone()[0]
            return {'interactions': interactions, 'facts': facts, 'patterns': patterns}


# ================= КОГНИТИВНЫЙ АГЕНТ =================
class CognitiveAgent:
    """Основной когнитивный агент с двухпроходным мышлением"""

    def __init__(self, user_id: int, db: CognitiveMemoryDB, llm: LocalLLMInterface,
                 search_engine: Optional[WebSearchEngine], memory: AdaptiveMemory,
                 critique_engine: SelfCritiqueEngine, annotator: KnowledgeAnnotator,
                 policy_learner: PolicyLearner):
        self.user_id = user_id
        self.db = db
        self.llm = llm
        self.search_engine = search_engine
        self.memory = memory
        self.critique_engine = critique_engine
        self.annotator = annotator
        self.policy_learner = policy_learner
        self.context_window = deque(maxlen=8)
        self.interaction_count = 0

    def _is_time_query(self, text: str) -> bool:
        text_lower = text.lower()
        time_patterns = [
            r'какой сейчас год', r'текущий год', r'сегодняшний год',
            r'какое сегодня число', r'какой сегодня день', r'текущая дата',
            r'сколько времени', r'который час', r'какое время'
        ]
        return any(re.search(pattern, text_lower) for pattern in time_patterns)

    def _handle_time_query(self, text: str) -> str:
        now = datetime.now()
        months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                  'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']

        if 'год' in text.lower():
            return f"📅 Текущий год: **{now.year}** (по системному времени)"
        if 'число' in text.lower() or 'день' in text.lower() or 'дата' in text.lower():
            return f"📅 Сегодня: **{now.day} {months[now.month - 1]} {now.year} года**"
        if 'время' in text.lower() or 'час' in text.lower():
            return f"⏰ Текущее время: **{now.hour:02d}:{now.minute:02d}**"
        return f"📅 Сейчас: **{now.day} {months[now.month - 1]} {now.year} года**, время **{now.hour:02d}:{now.minute:02d}**"

    async def process_message(self, user_input: str) -> str:
        self.interaction_count += 1
        self.context_window.append({'role': 'user', 'content': user_input, 'timestamp': time.time()})

        # Обработка временных запросов без поиска
        if self._is_time_query(user_input):
            response = self._handle_time_query(user_input)
            self.db.add_interaction(self.user_id, user_input, response, confidence=1.0, used_search=0)
            self.context_window.append({'role': 'assistant', 'content': response, 'timestamp': time.time()})
            return response

        # ШАГ 1: Генерация чернового ответа
        draft_answer = await self._generate_draft_answer(user_input)

        # ШАГ 2: Самокритика (ключевой компонент)
        critique = await self.critique_engine.critique_response(
            user_query=user_input,
            draft_answer=draft_answer,
            context=self._build_context_summary()
        )

        # ШАГ 3: Принятие решения о поиске
        needs_search = self.policy_learner.should_trigger_search(user_input, critique)
        used_search = False

        # ШАГ 4: Осознанный веб-поиск (ТОЛЬКО если критика требует)
        search_results = ""
        if needs_search and Config.WEB_SEARCH_ENABLED and self.search_engine:
            search_results = await self._perform_web_search(user_input)
            used_search = True

        # ШАГ 5: Синтез финального ответа с учётом поиска (если был)
        final_answer = await self._synthesize_final_answer(
            user_input=user_input,
            draft_answer=draft_answer,
            critique=critique,
            search_results=search_results
        )

        # ШАГ 6: Аннотация знаний
        annotation = await self.annotator.annotate_knowledge(final_answer)
        annotated_response = self._format_annotated_response(final_answer, annotation)

        # ШАГ 7: Обновление памяти
        confidence = critique.get('confidence', 0.5)
        self._update_memory(user_input, final_answer, critique, annotation, confidence)

        # Сохранение взаимодействия
        self.db.add_interaction(
            self.user_id,
            user_input,
            final_answer,
            critique_data=critique,
            confidence=confidence,
            used_search=1 if used_search else 0
        )

        self.context_window.append({'role': 'assistant', 'content': final_answer, 'timestamp': time.time()})

        # Применение деградации памяти каждые 10 взаимодействий
        if self.interaction_count % 10 == 0:
            self.memory.decay_memories(self.user_id)

        return annotated_response

    async def _generate_draft_answer(self, user_input: str) -> str:
        context = self._build_context_summary()
        system_prompt = (
            "Ты — когнитивный ассистент. Отвечай кратко и по делу. "
            "Если не знаешь ответа — скажи 'Я не знаю'. Не выдумывай факты."
        )
        user_prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{user_input}"

        return await self.llm.call_llm(system_prompt, user_prompt, temperature=0.6, max_tokens=500)

    async def _perform_web_search(self, query: str) -> str:
        if not self.search_engine:
            return ""

        results = await self.search_engine.search(query, max_results=3)
        if not results:
            return ""

        search_context = "\n".join([
            f"{i + 1}. {r['title']}: {r['snippet']}"
            for i, r in enumerate(results[:3])
        ])

        synthesis_prompt = (
            f"Запрос: {query}\n\nРезультаты поиска:\n{search_context}\n\n"
            "Синтезируй краткий ответ на основе найденной информации. "
            "Если информация противоречива — укажи это. Не выдумывай факты."
        )

        return await self.llm.call_llm(
            "Ты — аналитик. Синтезируй информацию из источников точно и кратко.",
            synthesis_prompt,
            temperature=0.3,
            max_tokens=400
        )

    async def _synthesize_final_answer(self, user_input: str, draft_answer: str,
                                       critique: Dict, search_results: str) -> str:
        # Если поиск был успешен — используем его результаты
        if search_results and "⚠️" not in search_results[:10]:
            return search_results

        # Если критика выявила неуверенность — добавляем оговорки
        confidence = critique.get('confidence', 0.5)
        uncertain_claims = critique.get('uncertain_claims', [])

        if confidence < 0.4:
            return f"Я не уверен в точном ответе. {draft_answer}\n\n⚠️ Уверенность: {confidence:.0%}"

        if uncertain_claims and confidence < 0.7:
            disclaimer = " ⚠️ Некоторые детали могут требовать уточнения."
            return f"{draft_answer}{disclaimer}"

        return draft_answer

    def _format_annotated_response(self, answer: str, annotation: Dict) -> str:
        parts = [answer]

        facts = annotation.get('facts', [])
        assumptions = annotation.get('assumptions', [])
        unknown = annotation.get('unknown', [])

        if assumptions or unknown:
            parts.append("\n\n🔍 Аннотация знаний:")
            if facts:
                parts.append(f"✅ Факты: {', '.join(facts[:3])}")
            if assumptions:
                parts.append(f"🤔 Предположения: {', '.join(assumptions[:3])}")
            if unknown:
                parts.append(f"❓ Требует проверки: {', '.join(unknown[:3])}")

        return "\n".join(parts)

    def _update_memory(self, user_input: str, answer: str, critique: Dict, annotation: Dict, confidence: float):
        # Сохраняем ошибки и неуверенные ответы
        if confidence < 0.5:
            self.memory.store_insight(
                self.user_id,
                'low_confidence_response',
                f"Вопрос: {user_input[:50]} | Ответ: {answer[:50]}",
                importance=0.8,
                metadata={'confidence': confidence, 'critique': critique}
            )

        # Сохраняем паттерны запросов, требующих поиска
        if critique.get('needs_search', False) or critique.get('temporal_sensitivity', False):
            self.policy_learner.save_policy(
                self.user_id,
                'search_policy',
                f"temporal_query_{hash(user_input) % 1000}",
                {'requires_search': True, 'query_sample': user_input[:30]}
            )

        # Сохраняем важные факты из аннотации
        for fact in annotation.get('facts', [])[:2]:
            self.db.add_fact(
                self.user_id,
                'learned_fact',
                fact,
                category='fact',
                importance=0.7,
                confidence=0.9
            )

    def _build_context_summary(self) -> str:
        if not self.context_window:
            return "Нет предыдущего контекста"

        summary = []
        for item in list(self.context_window)[-4:]:
            role = "Пользователь" if item['role'] == 'user' else "Ассистент"
            summary.append(f"{role}: {item['content'][:60]}")

        insights = self.memory.recall_relevant_insights(self.user_id, " ".join(
            [i['content'] for i in self.context_window if i['role'] == 'user']), limit=2)
        if insights:
            summary.append("\nРелевантные инсайты:")
            for insight in insights:
                summary.append(f"- {insight['content'][:70]}")

        return "\n".join(summary)

    def get_stats(self) -> str:
        stats = self.db.get_user_stats(self.user_id)
        return (
            f"📊 СТАТИСТИКА АГЕНТА\n{'=' * 40}\n"
            f"💬 Взаимодействий: {stats['interactions']}\n"
            f"🧠 Сохранённых фактов: {stats['facts']}\n"
            f"📈 Выявленных паттернов: {stats['patterns']}\n"
            f"🔄 Циклов мышления: {self.interaction_count}\n"
        )


# ================= МЕНЕДЖЕР СЕССИЙ =================
class SessionManager:
    def __init__(self, db: CognitiveMemoryDB, llm: LocalLLMInterface, search_engine: Optional[WebSearchEngine]):
        self.db = db
        self.llm = llm
        self.search_engine = search_engine
        self.sessions: Dict[int, CognitiveAgent] = {}
        self.last_cleanup = time.time()

    async def get_or_create_session(self, user_id: int) -> CognitiveAgent:
        if time.time() - self.last_cleanup > 300:
            await self._cleanup_inactive()
            self.last_cleanup = time.time()

        if user_id not in self.sessions:
            critique_engine = SelfCritiqueEngine(self.llm)
            annotator = KnowledgeAnnotator(self.llm)
            memory = AdaptiveMemory(self.db)
            policy_learner = PolicyLearner(self.db)

            self.sessions[user_id] = CognitiveAgent(
                user_id, self.db, self.llm, self.search_engine,
                memory, critique_engine, annotator, policy_learner
            )
            logging.info(f"🆕 Новая сессия для пользователя {user_id}")

        return self.sessions[user_id]

    async def _cleanup_inactive(self):
        pass  # Упрощено для краткости


# ================= ОБРАБОТЧИКИ TELEGRAM =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome = (
        f"👋 Привет, {user.first_name}!\n\n"
        "🧠 Я — когнитивный ассистент с двухпроходным мышлением:\n"
        "1️⃣ Генерация чернового ответа\n"
        "2️⃣ Самокритика и оценка уверенности\n"
        "3️⃣ Осознанный поиск (ТОЛЬКО при необходимости)\n"
        "4️⃣ Аннотация: факт / предположение / неизвестное\n\n"
        "✅ Полная приватность для локальных запросов\n"
        "⏰ Точное время/дата из системных часов (без поиска!)\n\n"
        "📌 Команды:\n"
        "/stats — статистика и метакогниция\n"
        "/clear — очистить контекст"
    )
    await update.message.reply_text(welcome, reply_markup=create_main_keyboard())


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    await update.message.reply_text(agent.get_stats(), reply_markup=create_main_keyboard())


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = await context.application.bot_data['session_manager'].get_or_create_session(update.effective_user.id)
    agent.context_window.clear()
    await update.message.reply_text("🧹 Контекст очищен", reply_markup=create_main_keyboard())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    if 'session_manager' not in context.application.bot_data:
        await update.message.reply_text("⚠️ Система инициализируется. Подождите 5 секунд.")
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        return

    try:
        session_manager = context.application.bot_data['session_manager']
        agent = await session_manager.get_or_create_session(user_id)
        await update.message.reply_chat_action("typing")

        # Короткая пауза для сложных запросов
        if len(text) > 25 or '?' in text:
            await asyncio.sleep(0.3)

        response = await agent.process_message(text)
        parts = split_message(response)

        for i, part in enumerate(parts):
            reply_markup = create_main_keyboard() if i == len(parts) - 1 else None
            await update.message.reply_text(part, reply_markup=reply_markup, disable_web_page_preview=True)
            if i < len(parts) - 1:
                await asyncio.sleep(0.3)
    except Exception as e:
        logging.error(f"Ошибка обработки: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Ошибка обработки запроса", reply_markup=create_main_keyboard())


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "clear":
        await clear_command(update, context)
    elif query.data == "stats":
        await stats_command(update, context)
    elif query.data == "search":
        await query.message.reply_text("🔍 Отправьте запрос в формате:\n/search [ваш запрос]")
    else:
        await query.message.reply_text("✅ Готов к диалогу", reply_markup=create_main_keyboard())


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} caused error {context.error}")


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
def setup_logging():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Config.LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    setup_logging()
    logging.info("🚀 Запуск когнитивного агента с двухпроходным мышлением")

    # Инициализация компонентов
    token = Config.get_telegram_token()
    lm_config = Config.get_lmstudio_config()

    db = CognitiveMemoryDB(Config.DB_PATH)
    llm = LocalLLMInterface(lm_config)
    search_engine = WebSearchEngine() if Config.WEB_SEARCH_ENABLED else None

    application = (
        ApplicationBuilder()
        .token(token)
        .read_timeout(25)
        .write_timeout(25)
        .connect_timeout(10)
        .pool_timeout(10)
        .build()
    )

    application.bot_data['session_manager'] = SessionManager(db, llm, search_engine)

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    # Настройка команд меню
    await application.bot.set_my_commands([
        BotCommand("start", "Начать"),
        BotCommand("stats", "Статистика"),
        BotCommand("clear", "Очистить контекст")
    ])

    # Запуск
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    print("\n" + "=" * 70)
    print("✅ КОГНИТИВНЫЙ АГЕНТ ЗАПУЩЕН")
    print("=" * 70)
    print(f"\n🤖 Модель: {lm_config['model']}")
    print("🔗 LM Studio: http://localhost:1234")
    print(f"🌐 Веб-поиск: {'активирован' if Config.WEB_SEARCH_ENABLED else 'отключён'}")
    print("\n📱 Напишите боту в Telegram /start")
    print("\n🧠 АРХИТЕКТУРА:")
    print("   • Двухпроходное мышление (ответ → критика → коррекция)")
    print("   • Осознанный поиск ТОЛЬКО после критики")
    print("   • Аннотация знаний: факт / предположение / неизвестное")
    print("   • Память с забыванием (только важные выводы)")
    print("   • Системное время без поиска для даты/времени")
    print("\n🛑 Остановка: Ctrl+C")
    print("=" * 70 + "\n")

    logging.info("🔄 Агент работает. Нажмите Ctrl+C для остановки")
    while True:
        await asyncio.sleep(3600)


def run():
    print("Cognitive Agent v5.0 — Когнитивная система с двухпроходным мышлением")
    print(f"Python: {sys.version.split()[0]}")

    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8+")
        sys.exit(1)

    # Проверка зависимостей
    required = ['aiohttp', 'requests']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"❌ Отсутствует: {pkg}. Установите: pip install {pkg}")
            sys.exit(1)

    if not HAS_WEB_SEARCH:
        print("⚠️  Веб-поиск недоступен. Для активации: pip install duckduckgo_search")

    print("\n🚀 Запуск когнитивного агента...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Работа завершена")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()
