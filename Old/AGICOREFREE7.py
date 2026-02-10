# coding: utf-8
"""
🧠 КОГНИТИВНЫЙ AI БОТ v2.0
Функции:
- Векторная память (семантический поиск)
- Автономный агент с внутренним монологом
- Полноценный интернет-поиск (SerpAPI/Google)
- Chain of Thought reasoning
- Self-reflection и планирование
"""

import os
import json
import aiohttp
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from dataclasses import dataclass, asdict

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from telegram.error import TimedOut, NetworkError

# ================== КОНФИГУРАЦИЯ ==================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")  # Для реального поиска
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")  # Альтернатива

MODEL_NAME = "local-model"
MAX_TOKENS = 2000
TIMEOUT = 60
DB_NAME = "cognitive_memory.db"

if not TELEGRAM_TOKEN or ":" not in TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не найден")


# ================== СТРУКТУРЫ ДАННЫХ ==================
@dataclass
class Memory:
    content: str
    role: str
    timestamp: str
    importance: float  # 0.0 - 1.0
    category: str
    embedding: Optional[List[float]] = None


@dataclass
class ThoughtProcess:
    question: str
    internal_monologue: List[str]
    reasoning_chain: List[str]
    actions_taken: List[str]
    reflection: str
    final_answer: str


@dataclass
class AgentDecision:
    should_search: bool
    search_queries: List[str]
    needs_clarification: bool
    clarification_questions: List[str]
    confidence: float


# ================== БАЗА ДАННЫХ ==================
def init_db():
    """Расширенная схема БД с векторной памятью"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Основная память
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            importance REAL,
            category TEXT,
            embedding TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT
        )
    """)

    # Краткосрочная рабочая память
    c.execute("""
        CREATE TABLE IF NOT EXISTS working_memory (
            user_id INTEGER PRIMARY KEY,
            current_context TEXT,
            active_goals TEXT,
            updated TEXT
        )
    """)

    # Долгосрочные знания и выводы
    c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            fact TEXT,
            confidence REAL,
            source TEXT,
            timestamp TEXT,
            embedding TEXT
        )
    """)

    # История мыслительных процессов
    c.execute("""
        CREATE TABLE IF NOT EXISTS thought_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            thought_process TEXT,
            timestamp TEXT
        )
    """)

    # Интернет-кэш
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY,
            results TEXT,
            timestamp TEXT,
            expires TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ================== ВЕКТОРНАЯ ПАМЯТЬ (простая версия) ==================
class SimpleEmbedding:
    """Упрощенные эмбеддинги на основе TF-IDF для автономной работы"""

    @staticmethod
    def encode(text: str) -> List[float]:
        """Простой хеш-вектор для демонстрации"""
        # В продакшене использовать sentence-transformers
        words = text.lower().split()
        vector = [0.0] * 128
        for word in words:
            idx = hash(word) % 128
            vector[idx] += 1.0
        # Нормализация
        norm = sum(v ** 2 for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    @staticmethod
    def similarity(v1: List[float], v2: List[float]) -> float:
        """Косинусное сходство"""
        return sum(a * b for a, b in zip(v1, v2))


embedding_model = SimpleEmbedding()


# ================== ПАМЯТЬ ==================
class MemoryManager:
    """Управление многоуровневой памятью"""

    @staticmethod
    def save_memory(user_id: int, role: str, content: str,
                    importance: float = 0.5, category: str = "general"):
        """Сохранение с эмбеддингом и метаданными"""
        embedding = embedding_model.encode(content)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO memories 
            (user_id, role, content, timestamp, importance, category, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, role, content,
            datetime.utcnow().isoformat(),
            importance, category,
            json.dumps(embedding)
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def semantic_search(user_id: int, query: str, limit: int = 5) -> List[Dict]:
        """Семантический поиск в памяти"""
        query_embedding = embedding_model.encode(query)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            SELECT content, role, importance, category, embedding, timestamp
            FROM memories WHERE user_id = ?
        """, (user_id,))

        results = []
        for row in c.fetchall():
            stored_embedding = json.loads(row[4])
            similarity = embedding_model.similarity(query_embedding, stored_embedding)
            results.append({
                'content': row[0],
                'role': row[1],
                'importance': row[2],
                'category': row[3],
                'timestamp': row[5],
                'relevance': similarity
            })

        conn.close()

        # Сортировка по релевантности * важности
        results.sort(key=lambda x: x['relevance'] * x['importance'], reverse=True)
        return results[:limit]

    @staticmethod
    def get_recent_context(user_id: int, limit: int = 6) -> str:
        """Получить недавний контекст"""
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            SELECT role, content FROM memories
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        rows = c.fetchall()
        conn.close()

        rows.reverse()
        return "\n".join(f"{r[0]}: {r[1]}" for r in rows)

    @staticmethod
    def save_knowledge(user_id: int, fact: str, confidence: float, source: str):
        """Сохранение долгосрочного знания"""
        embedding = embedding_model.encode(fact)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO knowledge_base
            (user_id, fact, confidence, source, timestamp, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, fact, confidence, source,
            datetime.utcnow().isoformat(),
            json.dumps(embedding)
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def retrieve_knowledge(user_id: int, query: str) -> List[Dict]:
        """Извлечение релевантных знаний"""
        query_embedding = embedding_model.encode(query)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            SELECT fact, confidence, source, embedding
            FROM knowledge_base WHERE user_id = ?
        """, (user_id,))

        results = []
        for row in c.fetchall():
            stored_embedding = json.loads(row[3])
            similarity = embedding_model.similarity(query_embedding, stored_embedding)
            results.append({
                'fact': row[0],
                'confidence': row[1],
                'source': row[2],
                'relevance': similarity
            })

        conn.close()
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:3]


# ================== LM STUDIO API ==================
async def ask_model(prompt: str, temperature: float = 0.7) -> str:
    """Запрос к локальной модели"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system",
             "content": "Ты автономный когнитивный AI агент с возможностью рассуждать, планировать и действовать."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{LM_STUDIO_URL}/v1/chat/completions",
                    json=payload,
                    timeout=TIMEOUT
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Ошибка модели: {e}"


# ================== ПОЛНОЦЕННЫЙ ИНТЕРНЕТ-ПОИСК ==================
class WebSearch:
    """Настоящий поиск через API"""

    @staticmethod
    async def serpapi_search(query: str) -> Dict:
        """Поиск через SerpAPI (Google)"""
        if not SERPAPI_KEY:
            return {"error": "SerpAPI ключ не настроен"}

        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 5
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://serpapi.com/search",
                        params=params,
                        timeout=20
                ) as resp:
                    return await resp.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    async def brave_search(query: str) -> Dict:
        """Поиск через Brave Search API"""
        if not BRAVE_API_KEY:
            return {"error": "Brave API ключ не настроен"}

        headers = {
            "X-Subscription-Token": BRAVE_API_KEY,
            "Accept": "application/json"
        }

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                        f"https://api.search.brave.com/res/v1/web/search?q={query}",
                        timeout=20
                ) as resp:
                    return await resp.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    async def duckduckgo_search(query: str) -> str:
        """Fallback: DuckDuckGo HTML парсинг"""
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=20) as resp:
                    html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            snippets = soup.select(".result__snippet")
            results = [s.get_text(strip=True) for s in snippets[:5]]
            return "\n---\n".join(results) if results else "Результатов не найдено"
        except Exception as e:
            return f"Ошибка поиска: {e}"

    @staticmethod
    async def comprehensive_search(query: str) -> Dict:
        """Комплексный поиск с несколькими источниками"""
        results = {
            "query": query,
            "sources": []
        }

        # Пробуем API в порядке приоритета
        if SERPAPI_KEY:
            serp_data = await WebSearch.serpapi_search(query)
            if "organic_results" in serp_data:
                results["sources"].append({
                    "name": "Google (SerpAPI)",
                    "results": [
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("snippet", ""),
                            "link": r.get("link", "")
                        }
                        for r in serp_data["organic_results"][:3]
                    ]
                })

        if BRAVE_API_KEY:
            brave_data = await WebSearch.brave_search(query)
            if "web" in brave_data and "results" in brave_data["web"]:
                results["sources"].append({
                    "name": "Brave Search",
                    "results": [
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("description", ""),
                            "link": r.get("url", "")
                        }
                        for r in brave_data["web"]["results"][:3]
                    ]
                })

        # Fallback на DuckDuckGo
        if not results["sources"]:
            ddg_results = await WebSearch.duckduckgo_search(query)
            results["sources"].append({
                "name": "DuckDuckGo",
                "text": ddg_results
            })

        return results


# ================== АВТОНОМНЫЙ АГЕНТ ==================
class CognitiveAgent:
    """Автономный агент с внутренним монологом"""

    @staticmethod
    async def internal_monologue(question: str, context: str) -> List[str]:
        """Внутренний монолог агента"""
        prompt = f"""
Ты автономный AI агент. Проанализируй вопрос и подумай вслух.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

Проведи внутренний анализ (3-5 мыслей):
1. Что именно спрашивают?
2. Что я знаю об этом?
3. Достаточно ли у меня информации?
4. Какие действия мне нужно предпринять?
5. Нужны ли уточнения?

Формат: каждая мысль с новой строки, начиная с "💭"
"""
        response = await ask_model(prompt, temperature=0.8)
        thoughts = [line.strip() for line in response.split('\n') if line.strip().startswith('💭')]
        return thoughts or ["💭 Анализирую запрос..."]

    @staticmethod
    async def decide_actions(question: str, thoughts: List[str]) -> AgentDecision:
        """Принятие решения о действиях"""
        prompt = f"""
ВОПРОС: {question}

МОИ МЫСЛИ:
{chr(10).join(thoughts)}

Прими решение в JSON формате:
{{
    "should_search": true/false,
    "search_queries": ["запрос 1", "запрос 2"],
    "needs_clarification": true/false,
    "clarification_questions": ["вопрос?"],
    "confidence": 0.0-1.0
}}

Ответь ТОЛЬКО JSON, без пояснений.
"""
        response = await ask_model(prompt, temperature=0.3)

        try:
            # Извлекаем JSON из ответа
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                decision_data = json.loads(response[json_start:json_end])
                return AgentDecision(**decision_data)
        except:
            pass

        # Fallback решение
        return AgentDecision(
            should_search="?" in question or any(
                word in question.lower() for word in ["сейчас", "сегодня", "курс", "цена"]),
            search_queries=[question],
            needs_clarification=False,
            clarification_questions=[],
            confidence=0.5
        )

    @staticmethod
    async def chain_of_thought(question: str, context: str, web_data: Optional[Dict] = None) -> List[str]:
        """Chain of Thought рассуждение"""
        web_context = ""
        if web_data and "sources" in web_data:
            web_context = "\n\nДАННЫЕ ИЗ ИНТЕРНЕТА:\n"
            for source in web_data["sources"]:
                if "results" in source:
                    for r in source["results"]:
                        web_context += f"- {r['title']}: {r['snippet']}\n"
                elif "text" in source:
                    web_context += source["text"] + "\n"

        prompt = f"""
КОНТЕКСТ:
{context}
{web_context}

ВОПРОС: {question}

Прояви chain-of-thought рассуждение (шаг за шагом):
Шаг 1: [анализ вопроса]
Шаг 2: [поиск релевантной информации]
Шаг 3: [синтез ответа]
Шаг 4: [проверка логики]
"""
        response = await ask_model(prompt, temperature=0.6)
        steps = [line.strip() for line in response.split('\n') if line.strip().startswith('Шаг')]
        return steps or ["Шаг 1: Анализирую вопрос..."]

    @staticmethod
    async def self_reflect(answer: str, question: str) -> str:
        """Саморефлексия над ответом"""
        prompt = f"""
ВОПРОС: {question}
МОЙ ОТВЕТ: {answer}

Проведи саморефлексию:
- Ответил ли я точно на вопрос?
- Достаточно ли полон ответ?
- Есть ли противоречия?
- Что можно улучшить?

Дай краткую рефлексию (2-3 предложения).
"""
        return await ask_model(prompt, temperature=0.5)

    @staticmethod
    async def generate_questions(context: str) -> List[str]:
        """Генерация уточняющих вопросов"""
        prompt = f"""
КОНТЕКСТ ДИАЛОГА:
{context}

Как автономный агент, какие вопросы я мог бы задать для лучшего понимания?
Сгенерируй 2-3 умных вопроса, начиная с "❓"
"""
        response = await ask_model(prompt, temperature=0.9)
        questions = [line.strip() for line in response.split('\n') if line.strip().startswith('❓')]
        return questions[:3]


# ================== ГЛАВНЫЙ ОБРАБОТЧИК ==================
async def process_message(user_id: int, text: str) -> ThoughtProcess:
    """Полный когнитивный процесс обработки"""

    # 1. Загрузка контекста из памяти
    recent_context = MemoryManager.get_recent_context(user_id, limit=4)
    relevant_memories = MemoryManager.semantic_search(user_id, text, limit=3)
    knowledge = MemoryManager.retrieve_knowledge(user_id, text)

    context = f"НЕДАВНИЙ ДИАЛОГ:\n{recent_context}\n\n"

    if relevant_memories:
        context += "РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ:\n"
        for mem in relevant_memories:
            context += f"- {mem['content']} (релевантность: {mem['relevance']:.2f})\n"
        context += "\n"

    if knowledge:
        context += "ДОЛГОСРОЧНЫЕ ЗНАНИЯ:\n"
        for k in knowledge:
            context += f"- {k['fact']} (уверенность: {k['confidence']:.2f})\n"
        context += "\n"

    # 2. Внутренний монолог
    thoughts = await CognitiveAgent.internal_monologue(text, context)

    # 3. Принятие решения о действиях
    decision = await CognitiveAgent.decide_actions(text, thoughts)

    actions_log = []
    web_data = None

    # 4. Выполнение поиска если необходимо
    if decision.should_search and decision.search_queries:
        actions_log.append(f"🔍 Провожу поиск: {decision.search_queries}")
        # Берем первый запрос для поиска
        web_data = await WebSearch.comprehensive_search(decision.search_queries[0])

    # 5. Chain of Thought рассуждение
    reasoning_steps = await CognitiveAgent.chain_of_thought(text, context, web_data)

    # 6. Генерация финального ответа
    final_prompt = f"""
КОНТЕКСТ:
{context}

МОИ МЫСЛИ:
{chr(10).join(thoughts)}

МОИ РАССУЖДЕНИЯ:
{chr(10).join(reasoning_steps)}
"""

    if web_data:
        final_prompt += f"\n\nДАННЫЕ ИЗ ИНТЕРНЕТА:\n{json.dumps(web_data, ensure_ascii=False, indent=2)}\n"

    final_prompt += f"\nВОПРОС: {text}\n\nДай точный, полный и хорошо обоснованный ответ."

    final_answer = await ask_model(final_prompt, temperature=0.7)

    # 7. Саморефлексия
    reflection = await CognitiveAgent.self_reflect(final_answer, text)

    # 8. Сохранение в память с оценкой важности
    importance = min(1.0, decision.confidence + 0.3)  # Базовая эвристика
    MemoryManager.save_memory(user_id, "user", text, importance, "question")
    MemoryManager.save_memory(user_id, "assistant", final_answer, importance, "answer")

    # 9. Извлечение и сохранение новых знаний
    if web_data:
        knowledge_prompt = f"Извлеки ключевой факт из этого ответа (одно предложение):\n{final_answer}"
        new_fact = await ask_model(knowledge_prompt, temperature=0.3)
        MemoryManager.save_knowledge(user_id, new_fact, decision.confidence, "web_search")

    return ThoughtProcess(
        question=text,
        internal_monologue=thoughts,
        reasoning_chain=reasoning_steps,
        actions_taken=actions_log,
        reflection=reflection,
        final_answer=final_answer
    )


# ================== TELEGRAM HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 **Когнитивный AI Агент v2.0**\n\n"
        "Возможности:\n"
        "• Векторная память\n"
        "• Автономное мышление\n"
        "• Полноценный интернет-поиск\n"
        "• Chain-of-Thought рассуждения\n"
        "• Саморефлексия\n\n"
        "Команды:\n"
        "/think - показать процесс мышления\n"
        "/memory - статистика памяти\n"
        "/knowledge - база знаний\n\n"
        "Задай любой вопрос! 🚀"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    # Индикатор обработки
    await update.message.reply_text("🤔 Думаю...")

    try:
        # Полный когнитивный процесс
        thought_process = await process_message(user_id, text)

        # Отправка ответа
        await update.message.reply_text(
            thought_process.final_answer,
            disable_web_page_preview=True
        )

        # Сохранение лога мышления
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO thought_logs (user_id, question, thought_process, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            user_id, text,
            json.dumps(asdict(thought_process), ensure_ascii=False),
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()

    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {e}")


async def show_thinking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать последний процесс мышления"""
    user_id = update.message.from_user.id

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT question, thought_process FROM thought_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (user_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        await update.message.reply_text("Пока нет записей мышления")
        return

    question, process_json = row
    process = json.loads(process_json)

    response = f"🧠 **ПРОЦЕСС МЫШЛЕНИЯ**\n\n"
    response += f"❓ Вопрос: {question}\n\n"
    response += "💭 Внутренний монолог:\n"
    for thought in process['internal_monologue']:
        response += f"{thought}\n"
    response += f"\n🔗 Рассуждения:\n"
    for step in process['reasoning_chain']:
        response += f"{step}\n"
    if process['actions_taken']:
        response += f"\n⚡ Действия:\n"
        for action in process['actions_taken']:
            response += f"{action}\n"
    response += f"\n🪞 Рефлексия:\n{process['reflection']}"

    # Telegram ограничение 4096 символов
    if len(response) > 4000:
        response = response[:4000] + "\n\n... (обрезано)"

    await update.message.reply_text(response)


async def show_memory_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика памяти"""
    user_id = update.message.from_user.id

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
    total_memories = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM knowledge_base WHERE user_id = ?", (user_id,))
    total_knowledge = c.fetchone()[0]

    c.execute("""
        SELECT category, COUNT(*) FROM memories 
        WHERE user_id = ? GROUP BY category
    """, (user_id,))
    categories = c.fetchall()

    conn.close()

    response = f"📊 **СТАТИСТИКА ПАМЯТИ**\n\n"
    response += f"💾 Всего воспоминаний: {total_memories}\n"
    response += f"🧠 База знаний: {total_knowledge}\n\n"
    response += "📁 По категориям:\n"
    for cat, count in categories:
        response += f"  • {cat}: {count}\n"

    await update.message.reply_text(response)


async def show_knowledge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать базу знаний"""
    user_id = update.message.from_user.id

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT fact, confidence, source FROM knowledge_base
        WHERE user_id = ?
        ORDER BY confidence DESC
        LIMIT 10
    """, (user_id,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        await update.message.reply_text("База знаний пуста")
        return

    response = "🎓 **БАЗА ЗНАНИЙ**\n\n"
    for fact, conf, source in rows:
        response += f"• {fact}\n  (уверенность: {conf:.0%}, источник: {source})\n\n"

    if len(response) > 4000:
        response = response[:4000] + "..."

    await update.message.reply_text(response)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    if isinstance(context.error, (TimedOut, NetworkError)):
        return
    print(f"❌ Ошибка: {context.error}")


# ================== MAIN ==================
def main():
    print("=" * 50)
    print("🧠 КОГНИТИВНЫЙ AI БОТ v2.0")
    print("=" * 50)
    print(f"✅ Telegram: OK")
    print(f"🤖 LM Studio: {LM_STUDIO_URL}")
    print(f"🔍 SerpAPI: {'✓' if SERPAPI_KEY else '✗'}")
    print(f"🔍 Brave: {'✓' if BRAVE_API_KEY else '✗'}")
    print("=" * 50)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("think", show_thinking))
    app.add_handler(CommandHandler("memory", show_memory_stats))
    app.add_handler(CommandHandler("knowledge", show_knowledge))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🚀 Бот запущен!\n")
    app.run_polling()


if __name__ == "__main__":
    main()