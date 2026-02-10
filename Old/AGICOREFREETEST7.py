# coding: utf-8
"""
🧠 КОГНИТИВНЫЙ AI БОТ v2.1 - БЕЗ API
Полноценный интернет-поиск БЕЗ необходимости в API ключах!
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
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

# Импортируем наш модуль поиска БЕЗ API
from web_search_no_api import MultiSearch, SpecializedSearch

# ================== КОНФИГУРАЦИЯ ==================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")

MODEL_NAME = "local-model"
MAX_TOKENS = 2000
TIMEOUT = 60
DB_NAME = "cognitive_memory.db"

if not TELEGRAM_TOKEN or ":" not in TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не найден")

# ================== ИНИЦИАЛИЗАЦИЯ ПОИСКА ==================
# ✅ Поиск работает БЕЗ API ключей!
multi_search = MultiSearch()
specialized_search = SpecializedSearch()

print("✅ Поиск инициализирован БЕЗ API ключей!")
print("🔍 Доступные источники:")
print("   • DuckDuckGo")
print("   • Searx (метапоиск)")
print("   • Wikipedia")
print("   • Yandex")


# ================== СТРУКТУРЫ ДАННЫХ ==================
@dataclass
class Memory:
    content: str
    role: str
    timestamp: str
    importance: float
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
    """Инициализация БД"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

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

    c.execute("""
        CREATE TABLE IF NOT EXISTS thought_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            thought_process TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ================== ВЕКТОРНЫЕ ЭМБЕДДИНГИ ==================
class SimpleEmbedding:
    """Простые эмбеддинги"""

    @staticmethod
    def encode(text: str) -> List[float]:
        words = text.lower().split()
        vector = [0.0] * 128
        for word in words:
            idx = hash(word) % 128
            vector[idx] += 1.0
        norm = sum(v ** 2 for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    @staticmethod
    def similarity(v1: List[float], v2: List[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))


embedding_model = SimpleEmbedding()


# ================== УПРАВЛЕНИЕ ПАМЯТЬЮ ==================
class MemoryManager:
    """Управление памятью"""

    @staticmethod
    def save_memory(user_id: int, role: str, content: str,
                    importance: float = 0.5, category: str = "general"):
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
        results.sort(key=lambda x: x['relevance'] * x['importance'], reverse=True)
        return results[:limit]

    @staticmethod
    def get_recent_context(user_id: int, limit: int = 6) -> str:
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


# ================== LM STUDIO ==================
import aiohttp


async def ask_model(prompt: str, temperature: float = 0.7) -> str:
    """Запрос к локальной модели"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Ты автономный когнитивный AI агент."},
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


# ================== КОГНИТИВНЫЙ АГЕНТ ==================
class CognitiveAgent:
    """Автономный когнитивный агент"""

    @staticmethod
    async def internal_monologue(question: str, context: str) -> List[str]:
        """Внутренний монолог"""
        prompt = f"""
Проанализируй вопрос и подумай вслух.

КОНТЕКСТ: {context}
ВОПРОС: {question}

Проведи внутренний анализ (3-4 мысли):
1. Что именно спрашивают?
2. Что я знаю об этом?
3. Достаточно ли информации?
4. Какие действия предпринять?

Формат: каждая мысль с новой строки, начиная с "💭"
"""
        response = await ask_model(prompt, temperature=0.8)
        thoughts = [line.strip() for line in response.split('\n') if line.strip().startswith('💭')]
        return thoughts or ["💭 Анализирую запрос..."]

    @staticmethod
    async def decide_actions(question: str, thoughts: List[str]) -> AgentDecision:
        """Принятие решения"""
        # Простая эвристика для определения необходимости поиска
        search_keywords = [
            'сейчас', 'сегодня', 'актуально', 'курс', 'цена',
            'новости', 'погода', 'последние', 'текущий', 'биткоин',
            'доллар', 'евро', 'что происходит', 'температура'
        ]

        question_lower = question.lower()
        should_search = any(keyword in question_lower for keyword in search_keywords)

        # Определяем уверенность
        confidence = 0.8 if should_search else 0.6

        return AgentDecision(
            should_search=should_search,
            search_queries=[question] if should_search else [],
            needs_clarification=False,
            clarification_questions=[],
            confidence=confidence
        )

    @staticmethod
    async def chain_of_thought(question: str, context: str, web_data: Optional[Dict] = None) -> List[str]:
        """Chain-of-Thought рассуждение"""
        web_context = ""
        if web_data and web_data.get('results'):
            web_context = "\n\nДАННЫЕ ИЗ ИНТЕРНЕТА:\n"
            for r in web_data['results'][:3]:
                web_context += f"- {r['title']}: {r['snippet'][:200]}\n"

        prompt = f"""
КОНТЕКСТ: {context}
{web_context}

ВОПРОС: {question}

Проведи chain-of-thought рассуждение:
Шаг 1: [анализ вопроса]
Шаг 2: [поиск информации]
Шаг 3: [синтез ответа]
Шаг 4: [проверка]
"""
        response = await ask_model(prompt, temperature=0.6)
        steps = [line.strip() for line in response.split('\n') if line.strip().startswith('Шаг')]
        return steps or ["Шаг 1: Анализирую..."]

    @staticmethod
    async def self_reflect(answer: str, question: str) -> str:
        """Саморефлексия"""
        prompt = f"""
ВОПРОС: {question}
МОЙ ОТВЕТ: {answer}

Краткая саморефлексия (2-3 предложения):
- Ответил ли точно?
- Достаточно ли полно?
- Что можно улучшить?
"""
        return await ask_model(prompt, temperature=0.5)


# ================== ОБРАБОТКА СООБЩЕНИЙ ==================
async def process_message(user_id: int, text: str) -> ThoughtProcess:
    """Полный когнитивный процесс"""

    # 1. Загрузка контекста
    recent_context = MemoryManager.get_recent_context(user_id, limit=4)
    relevant_memories = MemoryManager.semantic_search(user_id, text, limit=3)

    context = f"НЕДАВНИЙ ДИАЛОГ:\n{recent_context}\n\n"

    if relevant_memories:
        context += "РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ:\n"
        for mem in relevant_memories:
            context += f"- {mem['content']}\n"
        context += "\n"

    # 2. Внутренний монолог
    thoughts = await CognitiveAgent.internal_monologue(text, context)

    # 3. Принятие решения
    decision = await CognitiveAgent.decide_actions(text, thoughts)

    actions_log = []
    web_data = None

    # 4. Специализированные запросы (БЕЗ API!)
    text_lower = text.lower()

    # Биткоин
    if any(word in text_lower for word in ['биткоин', 'bitcoin', 'btc']):
        actions_log.append("🔍 Получаю курс Bitcoin...")
        btc_data = await specialized_search.get_bitcoin_price()
        if 'price_usd' in btc_data:
            price = btc_data['price_usd']
            final_answer = f"₿ Bitcoin сейчас:\n\n**${price:,.2f}** USD\n\nДанные от {btc_data['source']}"

            MemoryManager.save_memory(user_id, "user", text, 0.8, "question")
            MemoryManager.save_memory(user_id, "assistant", final_answer, 0.8, "answer")

            return ThoughtProcess(
                question=text,
                internal_monologue=thoughts,
                reasoning_chain=["Определил запрос цены Bitcoin", "Получил данные напрямую"],
                actions_taken=actions_log,
                reflection="Предоставил актуальную цену Bitcoin",
                final_answer=final_answer
            )

    # Погода
    if 'погода' in text_lower:
        # Пытаемся извлечь город
        city = "Moscow"  # По умолчанию
        for word in text.split():
            if word.istitle() and len(word) > 3:
                city = word
                break

        actions_log.append(f"🔍 Получаю погоду для {city}...")
        weather_data = await specialized_search.get_weather(city)

        if 'temperature_c' in weather_data:
            final_answer = f"🌡️ Погода в {city}:\n\n"
            final_answer += f"Температура: {weather_data['temperature_c']}°C\n"
            final_answer += f"Ощущается как: {weather_data['feels_like_c']}°C\n"
            final_answer += f"Описание: {weather_data['description']}\n"
            final_answer += f"Влажность: {weather_data['humidity']}%\n"
            final_answer += f"Ветер: {weather_data['wind_kph']} км/ч"

            MemoryManager.save_memory(user_id, "user", text, 0.7, "question")
            MemoryManager.save_memory(user_id, "assistant", final_answer, 0.7, "answer")

            return ThoughtProcess(
                question=text,
                internal_monologue=thoughts,
                reasoning_chain=["Определил запрос погоды", "Получил данные"],
                actions_taken=actions_log,
                reflection="Предоставил актуальную погоду",
                final_answer=final_answer
            )

    # 5. Общий веб-поиск (БЕЗ API!)
    if decision.should_search and decision.search_queries:
        actions_log.append(f"🔍 Провожу поиск: {decision.search_queries[0]}")

        # Глубокий поиск для сложных вопросов
        deep = len(text.split()) > 5
        web_data = await multi_search.search(decision.search_queries[0], deep=deep)

        if web_data and web_data['total_found'] > 0:
            actions_log.append(f"✅ Найдено: {web_data['total_found']} результатов")

    # 6. Chain-of-Thought
    reasoning_steps = await CognitiveAgent.chain_of_thought(text, context, web_data)

    # 7. Генерация ответа
    final_prompt = f"""
КОНТЕКСТ: {context}

МОИ МЫСЛИ:
{chr(10).join(thoughts)}

РАССУЖДЕНИЯ:
{chr(10).join(reasoning_steps)}
"""

    if web_data and web_data.get('results'):
        final_prompt += f"\n\nДАННЫЕ ИЗ ИНТЕРНЕТА:\n"
        for r in web_data['results'][:3]:
            final_prompt += f"- {r['title']}: {r['snippet'][:300]}\n"
            if 'full_content' in r:
                final_prompt += f"  Контент: {r['full_content'][:500]}\n"

    final_prompt += f"\nВОПРОС: {text}\n\nДай точный, полный ответ."

    final_answer = await ask_model(final_prompt, temperature=0.7)

    # 8. Саморефлексия
    reflection = await CognitiveAgent.self_reflect(final_answer, text)

    # 9. Сохранение в память
    importance = min(1.0, decision.confidence + 0.3)
    MemoryManager.save_memory(user_id, "user", text, importance, "question")
    MemoryManager.save_memory(user_id, "assistant", final_answer, importance, "answer")

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
        "🧠 **Когнитивный AI Агент v2.1**\n\n"
        "✅ **БЕЗ API КЛЮЧЕЙ!**\n\n"
        "Возможности:\n"
        "• Векторная память\n"
        "• Автономное мышление\n"
        "• Интернет-поиск (DuckDuckGo, Searx, Wikipedia, Yandex)\n"
        "• Курс Bitcoin в реальном времени\n"
        "• Погода в любом городе\n"
        "• Chain-of-Thought рассуждения\n\n"
        "Команды:\n"
        "/think - процесс мышления\n"
        "/memory - статистика памяти\n\n"
        "Задай любой вопрос! 🚀"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    await update.message.reply_text("🤔 Думаю...")

    try:
        thought_process = await process_message(user_id, text)

        await update.message.reply_text(
            thought_process.final_answer,
            disable_web_page_preview=True
        )

        # Сохранение лога
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
    """Показать процесс мышления"""
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

    if process['actions_taken']:
        response += f"\n⚡ Действия:\n"
        for action in process['actions_taken']:
            response += f"{action}\n"

    response += f"\n🪞 Рефлексия:\n{process['reflection']}"

    if len(response) > 4000:
        response = response[:4000] + "..."

    await update.message.reply_text(response)


async def show_memory_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика памяти"""
    user_id = update.message.from_user.id

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
    total = c.fetchone()[0]

    c.execute("""
        SELECT category, COUNT(*) FROM memories 
        WHERE user_id = ? GROUP BY category
    """, (user_id,))
    categories = c.fetchall()

    conn.close()

    response = f"📊 **СТАТИСТИКА ПАМЯТИ**\n\n"
    response += f"💾 Всего воспоминаний: {total}\n\n"
    response += "📁 По категориям:\n"
    for cat, count in categories:
        response += f"  • {cat}: {count}\n"

    await update.message.reply_text(response)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    if isinstance(context.error, (TimedOut, NetworkError)):
        return


# ================== MAIN ==================
def main():
    print("=" * 50)
    print("🧠 КОГНИТИВНЫЙ AI БОТ v2.1")
    print("✅ БЕЗ API КЛЮЧЕЙ!")
    print("=" * 50)
    print(f"✅ Telegram: OK")
    print(f"🤖 LM Studio: {LM_STUDIO_URL}")
    print(f"🔍 Поиск: Мультисорсовый (БЕЗ API)")
    print("=" * 50)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("think", show_thinking))
    app.add_handler(CommandHandler("memory", show_memory_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🚀 Бот запущен!\n")
    app.run_polling()


if __name__ == "__main__":
    main()