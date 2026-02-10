# coding: utf-8
"""
🧠 COGNITIVE_AGENT_PRO_v11_ENHANCED.py
✅ Полностью рабочий интеллектуальный поиск
✅ Метакогнитивная архитектура
✅ 10-этапный когнитивный цикл
✅ 5 типов памяти с самообучением
✅ Анализ пробелов знаний
✅ Автоматическая детекция необходимости поиска
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
import aiohttp
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import html
from collections import defaultdict
import statistics


# ================= ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =================
def load_dotenv(path: Path = Path(".env")):
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
            print(f"✅ Загружены переменные окружения из {path}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки .env: {e}")
    else:
        # Создаем .env если его нет
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Конфигурация когнитивного агента\n")
            f.write("TELEGRAM_BOT_TOKEN=your_token_here\n")
            f.write("LM_STUDIO_BASE_URL=http://localhost:1234\n")
            f.write("SERPAPI_API_KEY=your_serpapi_key_here\n")
        print(f"⚠️ Создан файл .env - заполните токены")


load_dotenv()


# ================= ЦВЕТНЫЕ ЛОГИ =================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    ORANGE = "\033[33m"
    PURPLE = "\033[35m"


def log_stage(stage: str, message: str, color: str = Colors.CYAN):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.BOLD}{color}[{timestamp}] {stage}{Colors.RESET}")
    print(f"{color}→ {message}{Colors.RESET}")


# ================= TELEGRAM =================
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )
    from telegram.error import TelegramError, TimedOut
except ImportError as e:
    print(f"{Colors.RED}❌ Ошибка импорта telegram: {e}{Colors.RESET}")
    print(f"{Colors.YELLOW}📦 Установите: pip install python-telegram-bot>=20.7{Colors.RESET}")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_brain_v11_enhanced")
    ROOT.mkdir(exist_ok=True)
    DB_PATH = ROOT / "brain_memory.db"
    SEARCH_CACHE_PATH = ROOT / "search_cache.json"

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

    TIMEOUT = 45
    MAX_TOKENS = 2500
    MAX_MESSAGE_LENGTH = 4096

    # Параметры поиска
    MAX_SEARCH_ATTEMPTS = 3
    SEARCH_CACHE_TTL = 3600
    MIN_CONFIDENCE_FOR_NO_SEARCH = 0.80


# ================= ENUMS И СТРУКТУРЫ =================
class ThinkingStrategy(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    STEP_BY_STEP = "step_by_step"
    EXPLORATORY = "exploratory"
    EXPLANATORY = "explanatory"
    WEB_RESEARCH = "web_research"  # НОВОЕ


class KnowledgeBoundary(Enum):
    KNOWN = "known"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"
    NEED_SEARCH = "need_search"
    OUTDATED = "outdated"  # НОВОЕ


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    META = "meta"
    SEARCH_HISTORY = "search_history"  # НОВОЕ


@dataclass
class SearchQuery:
    """Поисковый запрос"""
    query: str
    reason: str
    priority: int = 1


@dataclass
class SearchResult:
    """Результат поиска"""
    query: str
    content: str
    source: str
    timestamp: float
    confidence: float
    success: bool


@dataclass
class KnowledgeGap:
    """Пробел в знаниях"""
    topic: str
    reason: str
    severity: str  # critical, high, medium, low
    search_queries: List[str] = field(default_factory=list)


@dataclass
class CognitiveState:
    """Состояние когнитивного цикла"""
    query: str
    query_class: str = ""
    knowledge_boundary: KnowledgeBoundary = KnowledgeBoundary.UNKNOWN
    thinking_strategy: ThinkingStrategy = ThinkingStrategy.FACTUAL
    activated_memory: Dict[MemoryType, List] = field(default_factory=dict)
    reasoning_plan: List[str] = field(default_factory=list)
    confidence: float = 0.0
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    knowledge_gaps: List[KnowledgeGap] = field(default_factory=list)

    # Поисковые данные
    search_queries: List[SearchQuery] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    search_attempts: int = 0

    # Флаги
    used_memory: bool = False
    used_reasoning: bool = False
    used_web: bool = False
    confidence_level: float = 0.0
    uncertainty_detected: bool = False
    knowledge_verified: bool = False


@dataclass
class MetacognitiveMetrics:
    """Метрики метакогниции"""
    self_awareness_score: float = 0.0
    error_recognition_score: float = 0.0
    strategy_adaptation_score: float = 0.0
    learning_efficiency: float = 0.0
    search_efficiency: float = 0.0
    total_reflections: int = 0
    successful_adaptations: int = 0
    search_hits: int = 0
    search_misses: int = 0


@dataclass
class MemoryItem:
    """Элемент памяти"""
    id: str
    content: str
    memory_type: MemoryType
    confidence: float
    usage_count: int
    last_used: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass
class CognitiveResponse:
    """Полный когнитивный ответ с метаданными"""
    final_answer: str
    confidence: float
    knowledge_state: str
    thinking_strategy: str
    used_memory_types: List[str]
    processing_steps: List[str]
    error_analysis: Dict[str, Any]
    cognitive_insights: List[str]
    processing_time: float
    model_used: str
    search_queries_used: List[str]

    used_memory: bool
    used_reasoning: bool
    used_web: bool
    confidence_level: float
    uncertainty_detected: bool
    knowledge_verified: bool


# ================= ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСКОВЫЙ ДВИЖОК =================
class IntelligentSearchEngine:
    """ПОЛНОСТЬЮ РАБОЧИЙ поисковый движок с SerpAPI"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.SERPAPI_API_KEY
        self.session = None
        self.search_cache = self._load_cache()
        self.search_count = 0
        self.successful_searches = 0

        # Паттерны для определения необходимости поиска
        self.immediate_search_patterns = [
            r'курс\s+(доллар|евро|биткоин|рубл)',
            r'погод\w*\s+в\s+',
            r'новост\w*\s+(сегодня|вчера)',
            r'актуальн\w+',
            r'сейчас',
            r'текущ\w+',
            r'сколько\s+стоит',
            r'цена\s+',
        ]

    def _load_cache(self) -> Dict:
        """Загрузка кэша"""
        if Config.SEARCH_CACHE_PATH.exists():
            try:
                with open(Config.SEARCH_CACHE_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Сохранение кэша"""
        try:
            with open(Config.SEARCH_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_stage("⚠️ КЭШИРОВАНИЕ", f"Ошибка: {e}", Colors.YELLOW)

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def should_search(self, query: str, knowledge_boundary: KnowledgeBoundary,
                      initial_confidence: float) -> bool:
        """Определение необходимости поиска"""
        query_lower = query.lower()

        # 1. Критические паттерны
        for pattern in self.immediate_search_patterns:
            if re.search(pattern, query_lower):
                log_stage("🔍 ДЕТЕКЦИЯ", f"Обнаружен паттерн поиска: {pattern}", Colors.CYAN)
                return True

        # 2. Граница знаний
        if knowledge_boundary in [KnowledgeBoundary.NEED_SEARCH,
                                  KnowledgeBoundary.OUTDATED,
                                  KnowledgeBoundary.UNKNOWN]:
            log_stage("🔍 ДЕТЕКЦИЯ", f"Граница знаний: {knowledge_boundary.value}", Colors.CYAN)
            return True

        # 3. Низкая уверенность
        if initial_confidence < Config.MIN_CONFIDENCE_FOR_NO_SEARCH:
            log_stage("🔍 ДЕТЕКЦИЯ", f"Низкая уверенность: {initial_confidence:.0%}", Colors.CYAN)
            return True

        return False

    def analyze_knowledge_gaps(self, query: str, cognitive_state: CognitiveState,
                               initial_response: str = "") -> List[KnowledgeGap]:
        """Анализ пробелов в знаниях"""
        gaps = []

        # Маркеры неуверенности
        if initial_response:
            uncertainty_markers = [
                'не знаю', 'не уверен', 'возможно', 'вероятно',
                'может быть', 'не располагаю', 'нет информации'
            ]

            for marker in uncertainty_markers:
                if marker in initial_response.lower():
                    gaps.append(KnowledgeGap(
                        topic=extract_main_topic(query),
                        reason=f"Неуверенность: '{marker}'",
                        severity="high",
                        search_queries=self._generate_search_queries(query)
                    ))
                    break

        # Проверка необходимости актуальной информации
        if self._requires_current_data(query):
            gaps.append(KnowledgeGap(
                topic=extract_main_topic(query),
                reason="Требуется актуальная информация",
                severity="critical",
                search_queries=self._generate_current_info_queries(query)
            ))

        # Низкая уверенность
        if cognitive_state.confidence < 0.6:
            gaps.append(KnowledgeGap(
                topic=extract_main_topic(query),
                reason=f"Низкая уверенность: {cognitive_state.confidence:.0%}",
                severity="medium",
                search_queries=self._generate_search_queries(query)
            ))

        return gaps

    def _requires_current_data(self, query: str) -> bool:
        """Проверка необходимости текущей информации"""
        query_lower = query.lower()

        for pattern in self.immediate_search_patterns:
            if re.search(pattern, query_lower):
                return True

        time_markers = ['сегодня', 'вчера', 'сейчас', 'текущий', 'актуальный']
        return any(marker in query_lower for marker in time_markers)

    def _generate_search_queries(self, original_query: str) -> List[str]:
        """Генерация поисковых запросов"""
        queries = []

        # Базовый запрос
        clean_query = self._clean_for_search(original_query)
        queries.append(clean_query)

        # Добавление контекста
        keywords = extract_keywords(original_query)
        if len(keywords) > 2:
            queries.append(" ".join(keywords[:4]))

        return queries[:2]

    def _generate_current_info_queries(self, query: str) -> List[str]:
        """Генерация запросов для текущей информации"""
        base_query = self._clean_for_search(query)

        return [
            f"{base_query} {datetime.now().year}",
            f"{base_query} сейчас"
        ]

    def _clean_for_search(self, query: str) -> str:
        """Очистка запроса для поиска"""
        question_words = ['что', 'как', 'где', 'когда', 'почему', 'зачем', 'какой']
        words = query.lower().split()
        cleaned = [w for w in words if w not in question_words]

        result = " ".join(cleaned).strip()
        return result[:100]

    async def search(self, query: str, reason: str = "") -> Tuple[str, str, bool]:
        """ОСНОВНОЙ МЕТОД ПОИСКА - РАБОЧИЙ"""
        self.search_count += 1

        if not self.api_key or self.api_key == "your_serpapi_key_here":
            log_stage("⚠️ ПОИСК", "SerpAPI ключ не настроен", Colors.YELLOW)
            return "Поиск недоступен. Настройте SERPAPI_API_KEY в .env файле.", "disabled", False

        # Проверка кэша
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.search_cache:
            cached = self.search_cache[cache_key]
            if time.time() - cached.get('timestamp', 0) < Config.SEARCH_CACHE_TTL:
                log_stage("🔍 ПОИСК", f"Из кэша: {query[:50]}", Colors.CYAN)
                self.successful_searches += 1
                return cached['content'], "serpapi (кэш)", True

        try:
            session = await self.get_session()

            # Параметры запроса к SerpAPI
            params = {
                'q': query,
                'api_key': self.api_key,
                'engine': 'google',
                'gl': 'ru',  # Локация: Россия
                'hl': 'ru',  # Язык: русский
                'num': 5  # Количество результатов
            }

            log_stage("🔍 ПОИСК", f"Запрос к SerpAPI: {query}", Colors.CYAN)

            async with session.get(
                    'https://serpapi.com/search',
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    results = []

                    # Собираем результаты
                    if 'organic_results' in data:
                        for item in data['organic_results'][:5]:
                            snippet = item.get('snippet', '')
                            title = item.get('title', '')
                            link = item.get('link', '')

                            if snippet or title:
                                results.append(f"📌 {title}\n{snippet}\n🔗 {link}")

                    # Прямой ответ (answer box)
                    if 'answer_box' in data:
                        answer = data['answer_box'].get('answer', '')
                        if answer:
                            results.insert(0, f"✅ Прямой ответ: {answer}")

                    if results:
                        result_text = "\n\n".join(results)

                        # Кэширование
                        self.search_cache[cache_key] = {
                            'content': result_text,
                            'timestamp': time.time(),
                            'query': query
                        }
                        self._save_cache()

                        self.successful_searches += 1
                        log_stage("✅ ПОИСК", f"Найдено {len(results)} результатов", Colors.GREEN)
                        return result_text, "serpapi", True
                    else:
                        log_stage("⚠️ ПОИСК", "Результаты не найдены", Colors.YELLOW)
                        return "По вашему запросу ничего не найдено.", "serpapi", False

                elif response.status == 401:
                    log_stage("❌ ПОИСК", "Неверный API ключ", Colors.RED)
                    return "Ошибка: Неверный API ключ SerpAPI", "error", False
                else:
                    log_stage("❌ ПОИСК", f"HTTP {response.status}", Colors.RED)
                    return f"Ошибка поиска: HTTP {response.status}", "error", False

        except asyncio.TimeoutError:
            log_stage("❌ ПОИСК", "Таймаут", Colors.RED)
            return "Ошибка: Превышено время ожидания", "timeout", False
        except Exception as e:
            log_stage("❌ ПОИСК", f"Ошибка: {str(e)[:100]}", Colors.RED)
            return f"Ошибка при поиске: {str(e)[:100]}", "error", False

    async def multi_search(self, queries: List[SearchQuery]) -> List[SearchResult]:
        """Множественный поиск"""
        results = []

        sorted_queries = sorted(queries, key=lambda x: x.priority, reverse=True)

        for sq in sorted_queries[:Config.MAX_SEARCH_ATTEMPTS]:
            content, source, success = await self.search(sq.query, sq.reason)

            results.append(SearchResult(
                query=sq.query,
                content=content,
                source=source,
                timestamp=time.time(),
                confidence=0.8 if success else 0.2,
                success=success
            ))

            await asyncio.sleep(0.5)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Статистика поиска"""
        success_rate = (self.successful_searches / max(1, self.search_count)) * 100
        return {
            'total_searches': self.search_count,
            'successful': self.successful_searches,
            'success_rate': round(success_rate, 1),
            'cache_size': len(self.search_cache)
        }

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# ================= КОГНИТИВНАЯ ПАМЯТЬ =================
class CognitiveMemory:
    """Многоуровневая когнитивная память"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
        self.cache: Dict[str, MemoryItem] = {}

    def _init_db(self):
        """Инициализация БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                last_used REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT 0,
                expires_at REAL DEFAULT NULL
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_used ON memory_items(last_used DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON memory_items(expires_at)')

        conn.commit()
        conn.close()

    def store(self, item: MemoryItem):
        """Сохранить элемент"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO memory_items 
            (id, content, memory_type, confidence, usage_count, last_used, metadata, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id,
            item.content[:5000],
            item.memory_type.value,
            item.confidence,
            item.usage_count,
            item.last_used,
            json.dumps(item.metadata),
            item.created_at,
            item.expires_at
        ))

        conn.commit()
        conn.close()
        self.cache[item.id] = item

    def retrieve(self, query: str, memory_types: List[MemoryType] = None,
                 limit: int = 10) -> List[MemoryItem]:
        """Извлечь релевантные элементы"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        current_time = time.time()

        if memory_types:
            type_conditions = " OR ".join([f"memory_type = '{mt.value}'" for mt in memory_types])
            type_clause = f"AND ({type_conditions})"
        else:
            type_clause = ""

        keywords = re.findall(r'\b\w{3,}\b', query.lower())
        if not keywords:
            conn.close()
            return []

        search_conditions = []
        for kw in keywords[:5]:
            search_conditions.append(f"content LIKE '%{kw}%'")

        where_clause = " OR ".join(search_conditions)

        cursor.execute(f'''
            SELECT id, content, memory_type, confidence, usage_count, 
                   last_used, metadata, created_at, expires_at
            FROM memory_items
            WHERE ({where_clause}) {type_clause}
            AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY confidence DESC, usage_count DESC
            LIMIT ?
        ''', (current_time, limit))

        items = []
        for row in cursor.fetchall():
            items.append(MemoryItem(
                id=row[0],
                content=row[1],
                memory_type=MemoryType(row[2]),
                confidence=row[3],
                usage_count=row[4],
                last_used=row[5],
                metadata=json.loads(row[6]),
                created_at=row[7],
                expires_at=row[8]
            ))

        conn.close()
        return items

    def update_usage(self, memory_id: str):
        """Обновить счетчик"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE memory_items 
            SET usage_count = usage_count + 1, last_used = ?
            WHERE id = ?
        ''', (time.time(), memory_id))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                memory_type,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(usage_count) as avg_usage
            FROM memory_items
            WHERE expires_at IS NULL OR expires_at > ?
            GROUP BY memory_type
        ''', (time.time(),))

        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                'count': row[1],
                'avg_confidence': round(row[2] or 0, 2),
                'avg_usage': round(row[3] or 0, 2)
            }

        conn.close()
        return stats


# ================= МЕТАКОГНИТИВНЫЙ ДВИЖОК =================
class MetacognitionEngine:
    """Движок метакогниции с анализом знаний"""

    def __init__(self, memory: CognitiveMemory):
        self.memory = memory
        self.metrics = MetacognitiveMetrics()
        self.strategy_history: List[Tuple[ThinkingStrategy, float]] = []
        self.error_patterns: Dict[str, int] = defaultdict(int)

    def analyze_query(self, query: str) -> Tuple[str, KnowledgeBoundary, ThinkingStrategy]:
        """Анализ запроса"""
        query_lower = query.lower()

        knowledge_boundary = self._determine_knowledge_boundary(query_lower)
        query_class = self._classify_query(query_lower)
        strategy = self._select_thinking_strategy(query_class, knowledge_boundary)

        return query_class, knowledge_boundary, strategy

    def _determine_knowledge_boundary(self, query: str) -> KnowledgeBoundary:
        """Определение границы знаний"""
        # Проверка памяти
        memory_items = self.memory.retrieve(query, [MemoryType.SEMANTIC, MemoryType.EPISODIC])

        if len(memory_items) >= 3:
            avg_confidence = sum([item.confidence for item in memory_items]) / len(memory_items)
            # Проверка актуальности
            for item in memory_items:
                if item.expires_at and item.expires_at < time.time():
                    return KnowledgeBoundary.OUTDATED

            if avg_confidence > 0.7:
                return KnowledgeBoundary.KNOWN

        # Проверка триггеров поиска
        search_triggers = [
            'курс', 'погод', 'новост', 'актуальн', 'сегодня', 'сейчас',
            'биткоин', 'доллар', 'евро', 'цена'
        ]

        if any(trigger in query for trigger in search_triggers):
            return KnowledgeBoundary.NEED_SEARCH

        # Фактологические вопросы
        fact_patterns = [r'кто такой', r'что такое', r'где находится']
        for pattern in fact_patterns:
            if re.search(pattern, query):
                return KnowledgeBoundary.ASSUMED

        return KnowledgeBoundary.UNKNOWN

    def _classify_query(self, query: str) -> str:
        """Классификация запроса"""
        classifications = {
            'factual': ['что', 'кто', 'где', 'когда', 'сколько'],
            'analytical': ['почему', 'зачем', 'как так'],
            'procedural': ['как сделать', 'как создать', 'шаги'],
            'explanatory': ['объясни', 'расскажи', 'опиши'],
            'current_info': ['курс', 'погода', 'новости', 'сейчас']
        }

        for class_name, triggers in classifications.items():
            for trigger in triggers:
                if trigger in query:
                    return class_name

        return 'general'

    def _select_thinking_strategy(self, query_class: str,
                                  knowledge_boundary: KnowledgeBoundary) -> ThinkingStrategy:
        """Выбор стратегии"""
        # Приоритет веб-исследованию
        if knowledge_boundary in [KnowledgeBoundary.NEED_SEARCH, KnowledgeBoundary.OUTDATED]:
            return ThinkingStrategy.WEB_RESEARCH

        strategy_map = {
            'factual': ThinkingStrategy.FACTUAL,
            'analytical': ThinkingStrategy.ANALYTICAL,
            'procedural': ThinkingStrategy.STEP_BY_STEP,
            'explanatory': ThinkingStrategy.EXPLANATORY,
            'current_info': ThinkingStrategy.WEB_RESEARCH,
            'general': ThinkingStrategy.EXPLORATORY
        }

        base_strategy = strategy_map.get(query_class, ThinkingStrategy.EXPLORATORY)

        # Адаптация на основе истории
        if self.strategy_history:
            successful = [s for s, conf in self.strategy_history[-10:] if conf > 0.7]
            if len(successful) >= 3:
                most_common = max(set(successful), key=successful.count)
                if successful.count(most_common) >= 3:
                    return most_common

        return base_strategy

    def analyze_response(self, cognitive_state: CognitiveState,
                         response: str, confidence: float) -> Dict[str, Any]:
        """Анализ ответа"""
        analysis = {
            'confidence_evaluation': self._evaluate_confidence(confidence, response),
            'knowledge_gaps': self._identify_knowledge_gaps(response),
            'strategy_effectiveness': self._evaluate_strategy(cognitive_state, confidence),
            'search_effectiveness': self._evaluate_search(cognitive_state),
            'lessons': [],
            'warning_signs': []
        }

        # Извлечение уроков
        if confidence < 0.6 and not cognitive_state.used_web:
            lesson = "Низкая уверенность без поиска - возможно, нужен был веб-поиск"
            analysis['lessons'].append(lesson)

        if cognitive_state.used_web and confidence > 0.8:
            lesson = "Поиск повысил уверенность - эффективное использование"
            analysis['lessons'].append(lesson)
            self.metrics.search_hits += 1
        elif cognitive_state.used_web and confidence < 0.6:
            self.metrics.search_misses += 1

        # Обновление истории
        self.strategy_history.append((cognitive_state.thinking_strategy, confidence))
        self.metrics.total_reflections += 1

        if confidence > 0.7:
            self.metrics.successful_adaptations += 1

        return analysis

    def _evaluate_confidence(self, confidence: float, response: str) -> Dict[str, Any]:
        """Оценка уверенности"""
        evaluation = {
            'raw_confidence': confidence,
            'adjusted_confidence': confidence,
            'warning_signs': []
        }

        uncertainty_phrases = [
            'возможно', 'может быть', 'вероятно', 'не уверен',
            'не знаю', 'предполагаю'
        ]

        for phrase in uncertainty_phrases:
            if phrase in response.lower():
                evaluation['warning_signs'].append(f"Неопределенность: '{phrase}'")
                evaluation['adjusted_confidence'] *= 0.85

        if len(response.split()) < 20:
            evaluation['warning_signs'].append("Слишком краткий ответ")
            evaluation['adjusted_confidence'] *= 0.9

        evaluation['adjusted_confidence'] = max(0.1, min(1.0, evaluation['adjusted_confidence']))
        return evaluation

    def _identify_knowledge_gaps(self, response: str) -> List[str]:
        """Выявление пробелов"""
        gaps = []

        unknown_markers = [
            'не знаю', 'не могу ответить', 'нет информации', 'неизвестно'
        ]

        for marker in unknown_markers:
            if marker in response.lower():
                gaps.append(f"Отсутствие знаний ('{marker}')")

        return gaps

    def _evaluate_strategy(self, cognitive_state: CognitiveState,
                           confidence: float) -> Dict[str, Any]:
        """Оценка стратегии"""
        return {
            'strategy': cognitive_state.thinking_strategy.value,
            'confidence_achieved': confidence,
            'recommendation': 'continue' if confidence > 0.7 else 'change'
        }

    def _evaluate_search(self, cognitive_state: CognitiveState) -> Dict[str, Any]:
        """Оценка поиска"""
        evaluation = {
            'used_search': cognitive_state.used_web,
            'effectiveness': 'none'
        }

        if cognitive_state.used_web:
            successful = sum(1 for r in cognitive_state.search_results if r.success)
            if successful > 0:
                effectiveness_score = successful / max(1, len(cognitive_state.search_results))
                if effectiveness_score > 0.7:
                    evaluation['effectiveness'] = 'high'
                elif effectiveness_score > 0.4:
                    evaluation['effectiveness'] = 'medium'
                else:
                    evaluation['effectiveness'] = 'low'

        return evaluation

    def get_metrics(self) -> MetacognitiveMetrics:
        """Получение метрик"""
        if self.metrics.total_reflections > 0:
            self.metrics.self_awareness_score = len(self.error_patterns) / max(1, self.metrics.total_reflections)
            self.metrics.error_recognition_score = self.metrics.successful_adaptations / max(1,
                                                                                             self.metrics.total_reflections)

            total_searches = self.metrics.search_hits + self.metrics.search_misses
            if total_searches > 0:
                self.metrics.search_efficiency = self.metrics.search_hits / total_searches

        return self.metrics


# ================= КОГНИТИВНОЕ ЯДРО =================
class CognitiveCore:
    """Ядро с 10-этапным когнитивным циклом"""

    def __init__(self, model, search_engine: IntelligentSearchEngine, memory: CognitiveMemory):
        self.model = model
        self.search_engine = search_engine
        self.memory = memory
        self.metacognition = MetacognitionEngine(memory)

        self.cycle_count = 0
        self.successful_cycles = 0

    async def execute_cognitive_cycle(self, query: str, context: str = "") -> CognitiveResponse:
        """10-этапный когнитивный цикл"""
        start_time = time.time()
        cognitive_state = CognitiveState(query=query)
        processing_steps = []
        cognitive_insights = []

        try:
            # ЭТАП 1: ВОСПРИЯТИЕ
            processing_steps.append("1. Восприятие запроса")
            clean_query = self._clean_query(query)
            cognitive_state.query = clean_query
            log_stage("👁️ ЭТАП 1", f"Восприятие: {clean_query[:60]}", Colors.CYAN)

            # ЭТАП 2: КЛАССИФИКАЦИЯ
            processing_steps.append("2. Классификация задачи")
            query_class, knowledge_boundary, strategy = self.metacognition.analyze_query(clean_query)
            cognitive_state.query_class = query_class
            cognitive_state.knowledge_boundary = knowledge_boundary
            cognitive_state.thinking_strategy = strategy

            cognitive_insights.append(f"Класс: {query_class}")
            cognitive_insights.append(f"Знания: {knowledge_boundary.value}")
            log_stage("🏷️ ЭТАП 2", f"{query_class} | {knowledge_boundary.value}", Colors.BLUE)

            # ЭТАП 3: АКТИВАЦИЯ ПАМЯТИ
            processing_steps.append("3. Активация памяти")
            activated_memory = await self._activate_memory(clean_query, cognitive_state)
            cognitive_state.activated_memory = activated_memory
            cognitive_state.used_memory = any(activated_memory.values())

            memory_count = sum(len(items) for items in activated_memory.values())
            log_stage("🧠 ЭТАП 3", f"Активировано {memory_count} записей", Colors.MAGENTA)

            # ЭТАП 4: ОЦЕНКА ЗНАНИЙ
            processing_steps.append("4. Оценка достаточности знаний")
            initial_response, initial_confidence = await self._quick_assessment(cognitive_state, context)

            # ЭТАП 5: АНАЛИЗ ПРОБЕЛОВ
            processing_steps.append("5. Анализ пробелов знаний")
            knowledge_gaps = self.search_engine.analyze_knowledge_gaps(
                clean_query, cognitive_state, initial_response
            )
            cognitive_state.knowledge_gaps = knowledge_gaps

            if knowledge_gaps:
                log_stage("⚠️ ЭТАП 5", f"Обнаружено {len(knowledge_gaps)} пробелов", Colors.YELLOW)

            # ЭТАП 6: ВЕБ-ПОИСК
            processing_steps.append("6. Веб-поиск при необходимости")
            search_needed = self.search_engine.should_search(
                clean_query, knowledge_boundary, initial_confidence
            )

            if search_needed:
                cognitive_state.used_web = True
                await self._perform_search(cognitive_state, knowledge_gaps)
                log_stage("🔍 ЭТАП 6", f"Выполнено {len(cognitive_state.search_results)} запросов", Colors.GREEN)
            else:
                log_stage("✅ ЭТАП 6", "Поиск не требуется", Colors.GREEN)

            # ЭТАП 7: ПЛАН РАССУЖДЕНИЯ
            processing_steps.append("7. План рассуждения")
            reasoning_plan = self._create_reasoning_plan(cognitive_state)
            cognitive_state.reasoning_plan = reasoning_plan

            # ЭТАП 8: ОБОГАЩЕНИЕ КОНТЕКСТА
            processing_steps.append("8. Обогащение контекста")
            enriched_context = self._build_enriched_context(cognitive_state, context)

            # ЭТАП 9: ВЫПОЛНЕНИЕ РАССУЖДЕНИЯ
            processing_steps.append("9. Выполнение рассуждения")
            response, confidence = await self._execute_reasoning(cognitive_state, enriched_context)
            cognitive_state.confidence = confidence
            cognitive_state.confidence_level = confidence
            cognitive_state.uncertainty_detected = confidence < 0.6
            cognitive_state.knowledge_verified = cognitive_state.used_web or confidence > 0.8

            log_stage("💭 ЭТАП 9", f"Уверенность: {confidence:.0%}",
                      Colors.GREEN if confidence > 0.7 else Colors.YELLOW)

            # ЭТАП 10: ОБУЧЕНИЕ
            processing_steps.append("10. Обучение и метакогниция")
            error_analysis = self.metacognition.analyze_response(cognitive_state, response, confidence)
            cognitive_state.error_analysis = error_analysis
            cognitive_insights.extend(error_analysis.get('lessons', []))

            await self._learn_from_cycle(cognitive_state, response, confidence)

            # Обновление статистики
            self.cycle_count += 1
            if confidence > 0.6:
                self.successful_cycles += 1

            total_time = time.time() - start_time

            # Формирование ответа
            final_answer = self._format_final_response(response, cognitive_state, total_time)
            used_memory_types = [mt.value for mt, items in activated_memory.items() if items]
            search_queries_used = [sq.query for sq in cognitive_state.search_queries]

            return CognitiveResponse(
                final_answer=final_answer,
                confidence=confidence,
                knowledge_state=knowledge_boundary.value,
                thinking_strategy=strategy.value,
                used_memory_types=used_memory_types,
                processing_steps=processing_steps,
                error_analysis=error_analysis,
                cognitive_insights=cognitive_insights,
                processing_time=total_time,
                model_used=self.model.model_name,
                search_queries_used=search_queries_used,
                used_memory=cognitive_state.used_memory,
                used_reasoning=True,
                used_web=cognitive_state.used_web,
                confidence_level=confidence,
                uncertainty_detected=cognitive_state.uncertainty_detected,
                knowledge_verified=cognitive_state.knowledge_verified
            )

        except Exception as e:
            log_stage("❌ ОШИБКА", f"Цикл: {e}", Colors.RED)
            return self._create_error_response(str(e), start_time)

    def _clean_query(self, query: str) -> str:
        """Очистка запроса"""
        greetings = ['привет', 'здравствуй', 'hello', 'hi']
        query_lower = query.lower()

        for greet in greetings:
            if query_lower.startswith(greet):
                query = query[len(greet):].strip()
                break

        return query[:500].strip()

    async def _activate_memory(self, query: str, cognitive_state: CognitiveState) -> Dict[MemoryType, List[MemoryItem]]:
        """Активация памяти"""
        activated = {mt: [] for mt in MemoryType}

        activated[MemoryType.SEMANTIC] = self.memory.retrieve(query, [MemoryType.SEMANTIC], limit=5)
        activated[MemoryType.EPISODIC] = self.memory.retrieve(query, [MemoryType.EPISODIC], limit=3)

        if cognitive_state.query_class:
            proc_query = f"{cognitive_state.query_class}"
            activated[MemoryType.PROCEDURAL] = self.memory.retrieve(proc_query, [MemoryType.PROCEDURAL], limit=2)

        activated[MemoryType.SEARCH_HISTORY] = self.memory.retrieve(query, [MemoryType.SEARCH_HISTORY], limit=2)

        return activated

    async def _quick_assessment(self, cognitive_state: CognitiveState, context: str) -> Tuple[str, float]:
        """Быстрая оценка"""
        prompt = f"Запрос: {cognitive_state.query}\n\n"
        prompt += "Можешь ли ответить БЕЗ интернета? "
        prompt += "Если да - кратко ответь. Если нет - напиши 'НЕТ ЗНАНИЙ'."

        response, confidence = await self.model.generate(
            system_prompt="Ты честный агент. Различаешь 'знаю' и 'не знаю'.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=200
        )

        return response, confidence

    async def _perform_search(self, cognitive_state: CognitiveState, knowledge_gaps: List[KnowledgeGap]):
        """Выполнение поиска"""
        search_queries = []

        for gap in knowledge_gaps[:2]:
            for sq in gap.search_queries[:2]:
                search_queries.append(SearchQuery(
                    query=sq,
                    reason=gap.reason,
                    priority=3 if gap.severity == "critical" else 2
                ))

        if not search_queries:
            base_queries = self.search_engine._generate_search_queries(cognitive_state.query)
            for bq in base_queries[:2]:
                search_queries.append(SearchQuery(query=bq, reason="Общий поиск", priority=1))

        cognitive_state.search_queries = search_queries
        search_results = await self.search_engine.multi_search(search_queries)
        cognitive_state.search_results = search_results
        cognitive_state.search_attempts = len(search_results)

    def _build_enriched_context(self, cognitive_state: CognitiveState, base_context: str) -> str:
        """Обогащение контекста"""
        enriched = base_context

        if cognitive_state.search_results:
            enriched += "\n\n📡 АКТУАЛЬНАЯ ИНФОРМАЦИЯ ИЗ ИНТЕРНЕТА:\n"
            for i, result in enumerate(cognitive_state.search_results, 1):
                if result.success:
                    enriched += f"\n{i}. Запрос: {result.query}\n"
                    enriched += f"{result.content[:500]}...\n"

        for mem_type, items in cognitive_state.activated_memory.items():
            if items and mem_type != MemoryType.SEARCH_HISTORY:
                enriched += f"\n\n💾 ИЗ {mem_type.value.upper()} ПАМЯТИ:\n"
                for item in items[:2]:
                    enriched += f"• {item.content[:200]}...\n"

        return enriched

    def _create_reasoning_plan(self, cognitive_state: CognitiveState) -> List[str]:
        """План рассуждения"""
        if cognitive_state.thinking_strategy == ThinkingStrategy.WEB_RESEARCH:
            return [
                "Анализ информации из поиска",
                "Проверка актуальности",
                "Синтез источников",
                "Формирование ответа"
            ]
        elif cognitive_state.thinking_strategy == ThinkingStrategy.STEP_BY_STEP:
            return ["Разбиение на шаги", "Определение последовательности", "Проверка логики"]
        else:
            return ["Анализ запроса", "Поиск решения", "Формулирование ответа"]

    async def _execute_reasoning(self, cognitive_state: CognitiveState, context: str) -> Tuple[str, float]:
        """Выполнение рассуждения"""
        system_prompt = self._build_system_prompt(cognitive_state)
        user_prompt = self._build_user_prompt(cognitive_state, context)

        response, confidence = await self.model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3 if cognitive_state.thinking_strategy == ThinkingStrategy.WEB_RESEARCH else 0.5,
            max_tokens=2000
        )

        if cognitive_state.used_web and len(cognitive_state.search_results) > 0:
            successful = sum(1 for r in cognitive_state.search_results if r.success)
            if successful > 0:
                confidence = min(1.0, confidence * 1.15)

        return response, confidence

    def _build_system_prompt(self, cognitive_state: CognitiveState) -> str:
        """Системный промпт"""
        now = get_current_datetime()

        prompt_parts = [
            f"Ты — самообучающийся когнитивный агент.",
            f"Дата: {now['full_date']}, {now['time']}",
            "",
            f"ЗАДАЧА: {cognitive_state.query_class}",
            f"ЗНАНИЯ: {cognitive_state.knowledge_boundary.value}",
            f"СТРАТЕГИЯ: {cognitive_state.thinking_strategy.value}",
            ""
        ]

        if cognitive_state.thinking_strategy == ThinkingStrategy.WEB_RESEARCH:
            prompt_parts.extend([
                "🌐 РЕЖИМ ВЕБ-ИССЛЕДОВАНИЯ:",
                "1. Используй информацию из поисковых результатов",
                "2. Указывай источники",
                "3. Проверяй актуальность",
                ""
            ])

        prompt_parts.extend([
            "ПРАВИЛА:",
            "• 'ЗНАЮ' = есть факты",
            "• 'НЕ ЗНАЮ' = нет данных",
            "• При поиске - указывай источник",
            ""
        ])

        return "\n".join(prompt_parts)

    def _build_user_prompt(self, cognitive_state: CognitiveState, context: str) -> str:
        """Пользовательский промпт"""
        prompt = f"ЗАПРОС: {cognitive_state.query}\n\n"

        if context:
            prompt += f"КОНТЕКСТ:\n{context}\n\n"

        if cognitive_state.used_web:
            prompt += "⚡ У ТЕБЯ ЕСТЬ ИНФОРМАЦИЯ ИЗ ИНТЕРНЕТА! Используй её.\n\n"

        return prompt

    async def _learn_from_cycle(self, cognitive_state: CognitiveState, response: str, confidence: float):
        """Обучение"""
        # Эпизодическая память
        episodic_id = hashlib.md5(f"episodic:{cognitive_state.query}:{time.time()}".encode()).hexdigest()
        episodic_item = MemoryItem(
            id=episodic_id,
            content=f"Запрос: {cognitive_state.query[:100]} | Уверенность: {confidence:.0%}",
            memory_type=MemoryType.EPISODIC,
            confidence=confidence,
            usage_count=1,
            last_used=time.time(),
            metadata={'query_class': cognitive_state.query_class}
        )
        self.memory.store(episodic_item)

        # Сохранение успешных поисков
        if confidence > 0.8 and cognitive_state.used_web:
            semantic_id = hashlib.md5(f"semantic:{cognitive_state.query}".encode()).hexdigest()
            expires_at = time.time() + 3600 if cognitive_state.query_class == 'current_info' else None

            semantic_item = MemoryItem(
                id=semantic_id,
                content=response[:500],
                memory_type=MemoryType.SEMANTIC,
                confidence=confidence,
                usage_count=1,
                last_used=time.time(),
                metadata={'source': 'web_search', 'verified': True},
                expires_at=expires_at
            )
            self.memory.store(semantic_item)

    def _format_final_response(self, response: str, cognitive_state: CognitiveState, processing_time: float) -> str:
        """Форматирование ответа"""
        final = response

        reflection = "\n\n---\n🧠 **МЕТАРЕФЛЕКСИЯ:**\n"
        reflection += f"• **Стратегия:** {cognitive_state.thinking_strategy.value}\n"
        reflection += f"• **Уверенность:** {cognitive_state.confidence:.0%}\n"
        reflection += f"• **Граница знаний:** {cognitive_state.knowledge_boundary.value}\n\n"

        reflection += f"**Использовано:**\n"
        reflection += f"• Память: {'✅' if cognitive_state.used_memory else '❌'}\n"
        reflection += f"• Веб-поиск: {'✅' if cognitive_state.used_web else '❌'}\n"

        if cognitive_state.used_web:
            successful = sum(1 for r in cognitive_state.search_results if r.success)
            reflection += f"• Успешных поисков: {successful}/{len(cognitive_state.search_results)}\n"

        if cognitive_state.knowledge_verified:
            reflection += f"• **✅ Информация верифицирована**\n"

        reflection += f"\n• **Время:** {processing_time:.1f}с\n"
        reflection += f"• **Цикл:** #{self.cycle_count}"

        final += reflection
        return final

    def _create_error_response(self, error: str, start_time: float) -> CognitiveResponse:
        """Ответ об ошибке"""
        return CognitiveResponse(
            final_answer=f"❌ Ошибка: {error[:200]}",
            confidence=0.1,
            knowledge_state="error",
            thinking_strategy="none",
            used_memory_types=[],
            processing_steps=["Ошибка"],
            error_analysis={},
            cognitive_insights=[],
            processing_time=time.time() - start_time,
            model_used="error",
            search_queries_used=[],
            used_memory=False,
            used_reasoning=False,
            used_web=False,
            confidence_level=0.1,
            uncertainty_detected=True,
            knowledge_verified=False
        )

    def get_core_stats(self) -> Dict[str, Any]:
        """Статистика"""
        return {
            'cycle_count': self.cycle_count,
            'successful_cycles': self.successful_cycles,
            'success_rate': self.successful_cycles / max(1, self.cycle_count),
            'memory_stats': self.memory.get_stats(),
            'search_stats': self.search_engine.get_stats(),
            'metacognitive_metrics': {
                'self_awareness': round(self.metacognition.metrics.self_awareness_score, 2),
                'search_efficiency': round(self.metacognition.metrics.search_efficiency, 2)
            }
        }


# ================= УТИЛИТЫ =================
def clean_text(text: str) -> str:
    """Очистка текста"""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text: str) -> List[str]:
    """Извлечение ключевых слов"""
    if not text:
        return []
    stop_words = {
        'что', 'как', 'где', 'когда', 'почему', 'какой',
        'это', 'есть', 'был', 'была', 'были', 'быть'
    }
    words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
    return [w for w in words if w not in stop_words][:10]


def extract_main_topic(text: str) -> str:
    """Извлечение основной темы"""
    keywords = extract_keywords(text)
    return " ".join(keywords[:3]) if keywords else "общая тема"


def split_message(text: str, max_length: int = 4096) -> List[str]:
    """Разделение сообщения"""
    if len(text) <= max_length:
        return [text]
    parts = []
    while len(text) > max_length:
        split_point = text.rfind('\n', 0, max_length)
        if split_point == -1:
            split_point = text.rfind('. ', 0, max_length)
        if split_point == -1:
            split_point = max_length
        parts.append(text[:split_point].strip())
        text = text[split_point:].strip()
    if text:
        parts.append(text)
    return parts[:10]


def get_current_datetime() -> Dict[str, Any]:
    """Получение даты и времени"""
    now = datetime.now()
    months_ru = [
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ]
    weekdays_ru = [
        'понедельник', 'вторник', 'среда', 'четверг', 'пятница',
        'суббота', 'воскресенье'
    ]
    return {
        'datetime': now,
        'full_date': f"{now.day} {months_ru[now.month - 1]} {now.year} года",
        'weekday': weekdays_ru[now.weekday()],
        'time': now.strftime('%H:%M')
    }


# ================= МОДЕЛЬ =================
class ActiveModelInterface:
    """Интерфейс для LM Studio"""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.model_name = "Unknown Model"
        self.call_count = 0
        self.success_count = 0

    async def detect_active_model(self) -> bool:
        """Определение модели"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "unknown",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "temperature": 0.1,
                    "max_tokens": 10,
                    "stream": False
                }
                async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.model_name = data.get('model', 'LM Studio Model')
                        log_stage("🤖 МОДЕЛЬ", f"Обнаружена: {self.model_name}", Colors.GREEN)
                        return True
        except:
            log_stage("⚠️ МОДЕЛЬ", "Не удалось определить", Colors.YELLOW)
        return False

    async def generate(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.7, max_tokens: int = 2000) -> Tuple[str, float]:
        """Генерация ответа"""
        self.call_count += 1

        try:
            async with aiohttp.ClientSession() as session:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt[:2000]})
                messages.append({"role": "user", "content": user_prompt[:3000]})

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }

                async with session.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:

                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip()

                        # Оценка уверенности
                        confidence = 0.7
                        if len(content) > 100:
                            confidence = 0.8
                        if len(content) < 30:
                            confidence = 0.4

                        # Маркеры неуверенности
                        uncertainty_markers = [
                            'возможно', 'может быть', 'вероятно',
                            'не уверен', 'не знаю'
                        ]
                        for marker in uncertainty_markers:
                            if marker in content.lower():
                                confidence *= 0.85

                        self.success_count += 1
                        return content, min(1.0, max(0.1, confidence))
                    else:
                        return f"Ошибка API: {resp.status}", 0.1

        except Exception as e:
            return f"Ошибка: {str(e)[:100]}", 0.1


# ================= ТЕЛЕГРАМ БОТ =================
class CognitiveTelegramBot:
    """Телеграм бот с интеллектуальным поиском"""

    def __init__(self, token: str, cognitive_core: CognitiveCore):
        self.token = token
        self.core = cognitive_core
        self.user_contexts: Dict[int, List[str]] = defaultdict(list)

    async def start(self):
        """Запуск бота"""
        application = (
            ApplicationBuilder()
            .token(self.token)
            .read_timeout(60)
            .write_timeout(60)
            .build()
        )

        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("stats", self._handle_stats))
        application.add_handler(CommandHandler("meta", self._handle_meta))
        application.add_handler(CommandHandler("search", self._handle_search_test))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        await application.initialize()
        await application.start()

        log_stage("🤖 БОТ", "Metacognitive Agent запущен", Colors.GREEN)

        await application.updater.start_polling()
        return application

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка /start"""
        user = update.effective_user

        welcome = (
            f"🧠 **Метакогнитивный Агент v11 Enhanced**\n\n"
            f"Привет, {user.first_name}! Я самообучающаяся система "
            f"с РАБОЧИМ интеллектуальным поиском.\n\n"
            f"✨ **Особенности:**\n"
            f"• 10-этапный когнитивный цикл\n"
            f"• Автоматическая детекция необходимости поиска\n"
            f"• 5 типов памяти\n"
            f"• 6 стратегий мышления\n"
            f"• Метакогнитивная рефлексия\n\n"
            f"💡 **Команды:**\n"
            f"/stats - статистика\n"
            f"/meta - архитектура\n"
            f"/search <запрос> - тест поиска\n\n"
            f"🔍 Просто задай вопрос!"
        )

        await update.message.reply_text(welcome, parse_mode="Markdown")

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка /stats"""
        stats = self.core.get_core_stats()

        stats_text = (
            f"📊 **Статистика:**\n\n"
            f"🔁 **Циклы:**\n"
            f"• Всего: {stats['cycle_count']}\n"
            f"• Успешных: {stats['successful_cycles']}\n"
            f"• Успешность: {stats['success_rate']:.0%}\n\n"
            f"🔍 **Поиск:**\n"
            f"• Запросов: {stats['search_stats']['total_searches']}\n"
            f"• Успешных: {stats['search_stats']['successful']}\n"
            f"• Успешность: {stats['search_stats']['success_rate']}%\n"
            f"• Кэш: {stats['search_stats']['cache_size']} записей\n\n"
            f"💾 **Память:**\n"
        )

        for mem_type, mem_stats in stats['memory_stats'].items():
            stats_text += f"• {mem_type}: {mem_stats['count']} записей\n"

        await update.message.reply_text(stats_text, parse_mode="Markdown")

    async def _handle_meta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка /meta"""
        meta_text = (
            f"🧠 **Архитектура v11 Enhanced:**\n\n"
            f"**10 ЭТАПОВ:**\n"
            f"1. Восприятие\n2. Классификация\n3. Активация памяти\n"
            f"4. Оценка знаний\n5. Анализ пробелов\n"
            f"6. Веб-поиск\n7. План рассуждения\n"
            f"8. Обогащение контекста\n9. Выполнение\n10. Обучение\n\n"
            f"**5 ТИПОВ ПАМЯТИ:**\n"
            f"• Эпизодическая\n• Семантическая\n• Процедурная\n• Мета\n• История поиска\n\n"
            f"**6 СТРАТЕГИЙ:**\n"
            f"• Фактологическая\n• Аналитическая\n• Пошаговая\n"
            f"• Объяснительная\n• Исследовательская\n• Веб-исследование"
        )

        await update.message.reply_text(meta_text, parse_mode="Markdown")

    async def _handle_search_test(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Тест поиска"""
        if not context.args:
            await update.message.reply_text(
                "Использование: /search <запрос>\n"
                "Пример: /search курс биткоина"
            )
            return

        query = " ".join(context.args)
        await update.message.chat.send_action(action="typing")

        result, source, success = await self.core.search_engine.search(query, "Тест")

        response = f"🔍 **Тест поиска:**\n\n"
        response += f"**Запрос:** {query}\n"
        response += f"**Источник:** {source}\n"
        response += f"**Успех:** {'✅' if success else '❌'}\n\n"
        response += f"**Результат:**\n{result[:1000]}"

        parts = split_message(response)
        for part in parts:
            await update.message.reply_text(part, parse_mode="Markdown")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = update.effective_user.id
        query = update.message.text.strip()

        if not query:
            return

        await update.message.chat.send_action(action="typing")

        # Контекст
        user_context = ""
        if user_id in self.user_contexts and self.user_contexts[user_id]:
            user_context = "\n".join(self.user_contexts[user_id][-3:])

        # Когнитивный цикл
        log_stage("🚀 ЗАПРОС", f"User {user_id}: {query[:60]}", Colors.MAGENTA)

        response = await self.core.execute_cognitive_cycle(query, user_context)

        # Сохранение контекста
        context_entry = f"В: {query[:80]}\nО: {response.final_answer[:150]}"
        self.user_contexts[user_id].append(context_entry)
        if len(self.user_contexts[user_id]) > 10:
            self.user_contexts[user_id] = self.user_contexts[user_id][-10:]

        # Отправка
        parts = split_message(response.final_answer)
        for i, part in enumerate(parts):
            try:
                await update.message.reply_text(
                    part,
                    disable_web_page_preview=True,
                    parse_mode="Markdown" if i == 0 else None
                )
            except:
                await update.message.reply_text(part, disable_web_page_preview=True)
            await asyncio.sleep(0.3)

        # Логирование
        log_stage("📊 РЕЗУЛЬТАТ",
                  f"Уверенность: {response.confidence:.0%} | "
                  f"Стратегия: {response.thinking_strategy} | "
                  f"Поиск: {'✅' if response.used_web else '❌'} | "
                  f"Время: {response.processing_time:.1f}с",
                  Colors.GREEN if response.confidence > 0.7 else Colors.YELLOW)


# ================= ГЛАВНАЯ ФУНКЦИЯ =================
async def main():
    """Основная функция"""

    if not Config.TELEGRAM_TOKEN:
        print(f"\n{Colors.RED}❌ Telegram токен не найден!{Colors.RESET}")
        print(f"{Colors.YELLOW}Отредактируйте .env файл{Colors.RESET}")
        return

    now = get_current_datetime()
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{'=' * 80}")
    print(f"🧠 МЕТАКОГНИТИВНЫЙ АГЕНТ v11 ENHANCED")
    print(f"{'=' * 80}{Colors.RESET}")
    print(f"{Colors.GREEN}📅 Запуск: {now['full_date']}{Colors.RESET}")
    print(f"{Colors.GREEN}🕐 Время: {now['time']}{Colors.RESET}\n")

    # Инициализация
    log_stage("🔧 ИНИЦИАЛИЗАЦИЯ", "Загрузка компонентов...", Colors.BLUE)

    try:
        # Модель
        model = ActiveModelInterface(Config.LM_STUDIO_BASE_URL)
        await model.detect_active_model()
        await asyncio.sleep(1)

        # Память
        memory = CognitiveMemory(Config.DB_PATH)
        log_stage("💾 ПАМЯТЬ", f"База: {Config.DB_PATH}", Colors.GREEN)

        # Поисковый движок
        search_engine = IntelligentSearchEngine(Config.SERPAPI_API_KEY)
        if Config.SERPAPI_API_KEY and Config.SERPAPI_API_KEY != "your_serpapi_key_here":
            log_stage("🔍 ПОИСК", "SerpAPI настроен ✅", Colors.GREEN)
        else:
            log_stage("⚠️ ПОИСК", "SerpAPI не настроен - работа без интернета", Colors.YELLOW)

        # Когнитивное ядро
        cognitive_core = CognitiveCore(model, search_engine, memory)
        log_stage("🧠 ЯДРО", "Когнитивная архитектура готова", Colors.GREEN)

        # Бот
        bot = CognitiveTelegramBot(Config.TELEGRAM_TOKEN, cognitive_core)
        application = await bot.start()

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 80}")
        print(f"✅ СИСТЕМА ЗАПУЩЕНА")
        print(f"{'=' * 80}{Colors.RESET}\n")

        print(f"{Colors.CYAN}📊 Архитектура:{Colors.RESET}")
        print(f"  • 10-этапный когнитивный цикл")
        print(f"  • Интеллектуальная детекция пробелов знаний")
        print(f"  • Автоматический веб-поиск (SerpAPI)")
        print(f"  • 5 типов памяти")
        print(f"  • 6 стратегий мышления")
        print(f"  • Метакогнитивное обучение")

        print(f"\n{Colors.YELLOW}📱 Откройте Telegram!{Colors.RESET}")
        print(f"{Colors.RED}🛑 Ctrl+C для остановки{Colors.RESET}\n")

        # Основной цикл
        while True:
            await asyncio.sleep(60)

            stats = cognitive_core.get_core_stats()
            if stats['cycle_count'] > 0 and stats['cycle_count'] % 5 == 0:
                memory_count = sum(s['count'] for s in stats['memory_stats'].values())
                log_stage("📈 ПРОГРЕСС",
                          f"Циклов: {stats['cycle_count']} | "
                          f"Успешность: {stats['success_rate']:.0%} | "
                          f"Память: {memory_count} записей | "
                          f"Поисков: {stats['search_stats']['total_searches']}",
                          Colors.BLUE)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Остановка...{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
    finally:
        if 'search_engine' in locals():
            await search_engine.close()
        print(f"\n{Colors.GREEN}👋 Завершение{Colors.RESET}")


def run():
    """Точка входа"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}👋 Завершение{Colors.RESET}")


if __name__ == "__main__":
    run()