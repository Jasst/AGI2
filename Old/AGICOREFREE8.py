# coding: utf-8
"""
🧠 КОГНИТИВНЫЙ AI БОТ v2.4 - С ЗАЩИТОЙ ОТ ТАЙМАУТОВ TELEGRAM
✅ Автоматические повторные попытки при ошибках сети
✅ Поддержка прокси для обхода блокировок
✅ Честные ответы без галлюцинаций
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
import asyncio
import traceback

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from telegram.error import TimedOut, NetworkError, RetryAfter

# Импортируем модуль поиска БЕЗ API
from web_search_no_api import MultiSearch, SpecializedSearch

# ================== КОНФИГУРАЦИЯ ==================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "")  # Например: "http://127.0.0.1:7890" для Clash

MODEL_NAME = "local-model"
MAX_TOKENS = 2000
TIMEOUT = 60
DB_NAME = "cognitive_memory.db"

if not TELEGRAM_TOKEN or ":" not in TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не найден в .env файле")

# ================== ИНИЦИАЛИЗАЦИЯ ПОИСКА ==================
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
            datetime.now(timezone.utc).isoformat(),
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
            {
                "role": "system",
                "content": (
                    "Ты — когнитивный AI агент с доступом к:\n"
                    "• Текущему системному времени (через код)\n"
                    "• Интернет-поиску (DuckDuckGo, Searx, Wikipedia, Yandex)\n"
                    "• Специализированным данным (курсы валют, погода)\n"
                    "• Векторной памяти диалогов пользователя\n"
                    "• Chain-of-Thought рассуждениям для сложных задач.\n"
                    "❗ ПРАВИЛА:\n"
                    "1. Если у тебя есть точные данные из поиска — используй ТОЛЬКО их.\n"
                    "2. Если поиск не дал результатов — СКАЖИ: «Не удалось получить актуальные данные».\n"
                    "3. НЕ придумывай цифры (курсы, даты, цены) без подтверждения из источников.\n"
                    "4. Для вопросов о текущей дате/времени используй системное время — НЕ ищи в интернете.\n"
                    "5. Будь честен о своих ограничениях. НЕ галлюцинируй."
                )
            },
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
2. Это требует актуальных данных или можно ответить из знаний?
3. Нужен ли поиск или можно ответить локально (дата/время)?
4. Какие действия предпринять?

Формат: каждая мысль с новой строки, начиная с "💭"
"""
        response = await ask_model(prompt, temperature=0.8)
        thoughts = [line.strip() for line in response.split('\n') if line.strip().startswith('💭')]
        return thoughts or ["💭 Анализирую запрос..."]

    @staticmethod
    async def decide_actions(question: str, thoughts: List[str]) -> AgentDecision:
        """Принятие решения с улучшенной логикой"""
        text_lower = question.lower()

        # 🔴 ХАРД-РУЛЫ: НЕ искать для даты/времени!
        if any(kw in text_lower for kw in ['какая дата', 'какое число', 'сегодняшняя дата',
                                           'который час', 'сколько времени', 'текущее время',
                                           'сегодня', 'сейчас']):
            return AgentDecision(
                should_search=False,
                search_queries=[],
                needs_clarification=False,
                clarification_questions=[],
                confidence=0.95
            )

        # Эвристика для поиска
        search_keywords = [
            'курс', 'цена', 'биткоин', 'доллар', 'евро', 'новости',
            'погода', 'прогноз', 'актуально', 'последние', 'текущий',
            'что происходит', 'сейчас', 'сегодня'
        ]

        should_search = any(keyword in text_lower for keyword in search_keywords)
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
Шаг 1: [анализ вопроса — нужен ли поиск?]
Шаг 2: [результаты поиска — есть ли данные?]
Шаг 3: [синтез ответа — использовать ТОЛЬКО подтверждённые данные]
Шаг 4: [проверка — не придумываю ли я информацию?]
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
- Ответил ли точно на основе фактов?
- Не придумал ли я данные без подтверждения?
- Что можно улучшить в следующий раз?
"""
        return await ask_model(prompt, temperature=0.5)


# ================== НАДЕЖНАЯ ОТПРАВКА СООБЩЕНИЙ В TELEGRAM ==================
async def safe_send_message(update: Update, text: str, parse_mode: str = None, max_retries: int = 3):
    """
    Надежная отправка сообщений с повторными попытками при таймаутах
    """
    for attempt in range(max_retries):
        try:
            if parse_mode:
                return await update.message.reply_text(
                    text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )
            else:
                return await update.message.reply_text(
                    text,
                    disable_web_page_preview=True
                )
        except (TimedOut, NetworkError) as e:
            if attempt == max_retries - 1:
                # Последняя попытка не удалась — отправляем простое сообщение без форматирования
                try:
                    return await update.message.reply_text(
                        "⚠️ Сетевая ошибка при отправке ответа. Попробуйте повторить запрос."
                    )
                except:
                    print(f"КРИТИЧЕСКАЯ ОШИБКА: не удалось отправить даже сообщение об ошибке")
                    return None
            else:
                # Ждем перед повторной попыткой (экспоненциальная задержка)
                await asyncio.sleep(1 * (attempt + 1))
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except Exception as e:
            print(f"Неожиданная ошибка при отправке: {e}")
            traceback.print_exc()
            return None
    return None


# ================== ОБРАБОТКА СООБЩЕНИЙ ==================
async def process_message(user_id: int, text: str) -> ThoughtProcess:
    """Полный когнитивный процесс с хард-рулами"""

    # 🔴 ХАРД-РУЛ №1: ТЕКУЩАЯ ДАТА И ВРЕМЯ (БЕЗ ПОИСКА!)
    text_lower = text.lower()

    # Дата
    if any(kw in text_lower for kw in ['какая дата', 'какое число', 'сегодняшняя дата',
                                       'который день', 'сегодня', 'какой сегодня']):
        now = datetime.now()
        formatted_date = now.strftime("%d %B %Y")
        weekday = now.strftime("%A")
        ru_weekdays = {
            "Monday": "понедельник", "Tuesday": "вторник", "Wednesday": "среда",
            "Thursday": "четверг", "Friday": "пятница", "Saturday": "суббота", "Sunday": "воскресенье"
        }
        ru_months = {
            "January": "января", "February": "февраля", "March": "марта", "April": "апреля",
            "May": "мая", "June": "июня", "July": "июля", "August": "августа",
            "September": "сентября", "October": "октября", "November": "ноября", "December": "декабря"
        }

        formatted_date_ru = formatted_date
        for eng, rus in ru_months.items():
            formatted_date_ru = formatted_date_ru.replace(eng, rus)

        weekday_ru = ru_weekdays.get(weekday, weekday)
        final_answer = f"📅 Сегодня: {formatted_date_ru} ({weekday_ru})"

        MemoryManager.save_memory(user_id, "user", text, 0.9, "question")
        MemoryManager.save_memory(user_id, "assistant", final_answer, 0.9, "answer")

        return ThoughtProcess(
            question=text,
            internal_monologue=["💭 Запрос текущей даты — отвечаю локально без поиска"],
            reasoning_chain=["Определён запрос даты", "Использовано системное время"],
            actions_taken=["✅ Ответ без интернета"],
            reflection="Предоставлена точная текущая дата из системного времени",
            final_answer=final_answer
        )

    # Время
    if any(kw in text_lower for kw in ['который час', 'сколько времени', 'текущее время',
                                       'сейчас времени', 'который сейчас']):
        now = datetime.now()
        formatted_time = now.strftime("%H:%M")
        final_answer = f"⏰ Сейчас: {formatted_time}"

        MemoryManager.save_memory(user_id, "user", text, 0.9, "question")
        MemoryManager.save_memory(user_id, "assistant", final_answer, 0.9, "answer")

        return ThoughtProcess(
            question=text,
            internal_monologue=["💭 Запрос текущего времени — отвечаю локально без поиска"],
            reasoning_chain=["Определён запрос времени", "Использовано системное время"],
            actions_taken=["✅ Ответ без интернета"],
            reflection="Предоставлено точное текущее время из системного времени",
            final_answer=final_answer
        )

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
    # Биткоин
    if any(word in text_lower for word in ['биткоин', 'bitcoin', 'btc', 'бтц']):
        actions_log.append("🔍 Получаю курс Bitcoin...")
        btc_data = await specialized_search.get_bitcoin_price()

        if 'error' in btc_data:
            final_answer = "⚠️ Не удалось получить актуальный курс Bitcoin. Попробуйте позже."
            MemoryManager.save_memory(user_id, "user", text, 0.7, "question")
            MemoryManager.save_memory(user_id, "assistant", final_answer, 0.7, "answer")

            return ThoughtProcess(
                question=text,
                internal_monologue=thoughts,
                reasoning_chain=["Определил запрос цены Bitcoin", "Данные недоступны"],
                actions_taken=actions_log + ["❌ Данные не получены"],
                reflection="Не удалось получить курс — честно сообщил об этом",
                final_answer=final_answer
            )

        if 'price_usd' in btc_data:
            price = btc_data['price_usd']
            final_answer = f"₿ Bitcoin сейчас:\n\n${price:,.2f} USD\n\nДанные от {btc_data['source']}"

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
        words = text.split()

        for i, word in enumerate(words):
            clean_word = word.strip('.,!?')
            if clean_word.istitle() and len(clean_word) > 3 and i > 0:
                city = clean_word
                break

        for i, word in enumerate(words):
            if word.lower() in ['в', 'для'] and i + 1 < len(words):
                next_word = words[i + 1].strip('.,!?')
                if next_word.istitle():
                    city = next_word
                    break

        actions_log.append(f"🔍 Получаю погоду для {city}...")
        weather_data = await specialized_search.get_weather(city)

        if 'error' in weather_data:
            final_answer = f"⚠️ Не удалось получить погоду для {city}. Проверьте название города."
            MemoryManager.save_memory(user_id, "user", text, 0.6, "question")
            MemoryManager.save_memory(user_id, "assistant", final_answer, 0.6, "answer")

            return ThoughtProcess(
                question=text,
                internal_monologue=thoughts,
                reasoning_chain=["Определил запрос погоды", "Данные недоступны"],
                actions_taken=actions_log + ["❌ Данные не получены"],
                reflection="Не удалось получить погоду — честно сообщил об этом",
                final_answer=final_answer
            )

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

        deep = len(text.split()) > 5
        web_data = await multi_search.search(decision.search_queries[0], deep=deep)

        if not web_data or web_data.get('total_found', 0) == 0:
            final_answer = "⚠️ Не удалось найти актуальную информацию в интернете. Попробуйте уточнить запрос."

            MemoryManager.save_memory(user_id, "user", text, 0.5, "question")
            MemoryManager.save_memory(user_id, "assistant", final_answer, 0.5, "answer")

            return ThoughtProcess(
                question=text,
                internal_monologue=thoughts,
                reasoning_chain=["Определён запрос на поиск", "Результаты не найдены"],
                actions_taken=actions_log + ["❌ Поиск не дал результатов"],
                reflection="Поиск не удался — не стал галлюцинировать, честно сообщил",
                final_answer=final_answer
            )
        else:
            actions_log.append(f"✅ Найдено: {web_data['total_found']} результатов")

    # 6. Chain-of-Thought
    reasoning_steps = await CognitiveAgent.chain_of_thought(text, context, web_data)

    # 7. Генерация ответа ТОЛЬКО если есть данные
    final_prompt = f"""
КОНТЕКСТ: {context}

МОИ МЫСЛИ:
{chr(10).join(thoughts)}

РАССУЖДЕНИЯ:
{chr(10).join(reasoning_steps)}
"""

    if web_data and web_data.get('results'):
        final_prompt += f"\n\nДАННЫЕ ИЗ ИНТЕРНЕТА (ТОЛЬКО ЭТИ ДАННЫЕ ИСПОЛЬЗУЙ):\n"
        for r in web_data['results'][:3]:
            final_prompt += f"- {r['title']}: {r['snippet'][:300]}\n"
            if 'full_content' in r:
                final_prompt += f"  Контент: {r['full_content'][:500]}\n"

    final_prompt += f"\nВОПРОС: {text}\n\n❗ ВАЖНО: Используй ТОЛЬКО приведённые выше данные. Если данных нет — скажи «Не удалось найти информацию»."

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
    help_text = (
        "🧠 **Когнитивный AI Агент v2.4**\n\n"
        "✅ **ПОЛНОСТЬЮ ИСПРАВЛЕНО:**\n"
        "• Точная дата/время без поиска (локально)\n"
        "• Нет галлюцинаций при пустом поиске\n"
        "• Честные ответы при ошибках (не выдумывает)\n"
        "• Защита от таймаутов Telegram\n"
        "• Поддержка прокси для обхода блокировок\n\n"
        "💡 **Если бот не отвечает:**\n"
        "1. Убедитесь, что у вас есть доступ к Telegram\n"
        "2. Если вы в РФ — настройте прокси в .env файле:\n"
        "   `TELEGRAM_PROXY=http://127.0.0.1:7890`\n"
        "3. Перезапустите бота\n\n"
        "Возможности:\n"
        "• Векторная память диалогов\n"
        "• Автономное мышление (CoT)\n"
        "• Интернет-поиск (DuckDuckGo, Searx, Wikipedia, Yandex)\n"
        "• Курс Bitcoin в реальном времени (БЕЗ API)\n"
        "• Погода в любом городе (БЕЗ API)\n\n"
        "Команды:\n"
        "/think - показать процесс мышления\n"
        "/memory - статистика памяти пользователя\n"
        "/start - эта справка"
    )
    await safe_send_message(update, help_text, parse_mode='Markdown')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    # Отправляем "думаю" с защитой от таймаута
    thinking_msg = await safe_send_message(update, "🤔 Думаю...", max_retries=1)

    try:
        thought_process = await process_message(user_id, text)

        # Отправляем ответ с защитой от таймаута
        await safe_send_message(update, thought_process.final_answer, max_retries=3)

        # Сохранение лога
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO thought_logs (user_id, question, thought_process, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            user_id, text,
            json.dumps(asdict(thought_process), ensure_ascii=False),
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
        conn.close()

    except Exception as e:
        error_msg = f"⚠️ Внутренняя ошибка: {str(e)[:100]}"
        await safe_send_message(update, error_msg)
        print(f"Ошибка обработки сообщения: {e}")
        traceback.print_exc()


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
        await safe_send_message(update, "Пока нет записей мышления")
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

    await safe_send_message(update, response, parse_mode='Markdown')


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

    await safe_send_message(update, response, parse_mode='Markdown')


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    if isinstance(context.error, (TimedOut, NetworkError)):
        print(f"Сетевая ошибка Telegram: {context.error}")
        return
    print(f"Необработанная ошибка: {context.error}")
    traceback.print_exc()


# ================== MAIN ==================
def main():
    print("=" * 60)
    print("🧠 КОГНИТИВНЫЙ AI БОТ v2.4 — С ЗАЩИТОЙ ОТ ТАЙМАУТОВ")
    print("=" * 60)
    print(f"✅ Telegram Token: {'✓' if TELEGRAM_TOKEN else '✗'}")
    print(f"🤖 LM Studio: {LM_STUDIO_URL}")
    print(f"🌍 Прокси: {TELEGRAM_PROXY if TELEGRAM_PROXY else 'не используется'}")
    print(f"🔍 Поиск: Мультисорсовый (БЕЗ API)")
    print("=" * 60)

    # Проверка подключения к интернету
    print("\n📡 Проверка подключения к интернету...")
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        print("✅ Интернет доступен")
    except OSError:
        print("⚠️  Проблемы с интернет-соединением!")
        print("💡 Советы:")
        print("   • Проверьте подключение к сети")
        print("   • Если в РФ — настройте прокси в .env файле:")
        print("     TELEGRAM_PROXY=http://127.0.0.1:7890")

    print("\n🚀 Запуск бота...\n")

    # Создаем приложение с прокси если указан
    if TELEGRAM_PROXY:
        print(f"🔌 Используется прокси: {TELEGRAM_PROXY}")
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).proxy_url(TELEGRAM_PROXY).build()
    else:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("think", show_thinking))
    app.add_handler(CommandHandler("memory", show_memory_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("✅ Бот запущен и ожидает сообщений!\n")
    print("💡 Если бот не отвечает в Telegram:")
    print("   1. Убедитесь, что у вас есть доступ к Telegram")
    print("   2. Проверьте интернет-соединение")
    print("   3. Настройте прокси если находитесь в РФ")
    print("   4. Перезапустите бота")
    print("\n" + "=" * 60)

    app.run_polling()


if __name__ == "__main__":
    main()