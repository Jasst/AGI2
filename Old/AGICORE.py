import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from duckduckgo_search import DDGS
import re

# Загрузка переменных окружения
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')


class SmartWebSearchBot:
    def __init__(self):
        self.ddgs = DDGS()
        self.search_keywords = [
            'найди', 'поищи', 'последние', 'сегодняшние', 'актуальные', 'новости',
            'погода', 'курс', 'цена', 'события', 'происходит сейчас'
        ]
        # Запросы, НЕ требующие поиска (системная информация)
        self.no_search_keywords = [
            'какой сегодня день', 'какое число', 'какой день недели',
            'который час', 'сколько времени', 'текущая дата', 'сегодня'
        ]

    def get_current_datetime_info(self) -> str:
        """Получение актуальной даты и времени"""
        now = datetime.now()
        weekdays_ru = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
        months_ru = [
            'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
            'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
        ]

        return (
            f"Текущая дата и время (системное время сервера):\n"
            f"Дата: {now.day} {months_ru[now.month - 1]} {now.year} года\n"
            f"День недели: {weekdays_ru[now.weekday()]}\n"
            f"Время: {now.hour:02d}:{now.minute:02d}\n\n"
            f"ВАЖНО: Эта информация получена напрямую от системы, а не из интернета."
        )

    def needs_web_search(self, query: str) -> bool:
        """Умное определение необходимости поиска"""
        query_lower = query.lower().strip()

        # Исключаем запросы о текущей дате/времени
        for kw in self.no_search_keywords:
            if kw in query_lower:
                return False

        # Проверяем на наличие "поисковых" ключевых слов
        for kw in self.search_keywords:
            if kw in query_lower:
                return True

        # Эвристика: короткие вопросы о фактах часто требуют поиска
        if len(query_lower.split()) <= 5 and (
                query_lower.startswith('кто') or
                query_lower.startswith('что') or
                query_lower.startswith('где') or
                query_lower.startswith('когда') or
                'последние' in query_lower or
                'новый' in query_lower or
                '2024' in query_lower or
                '2025' in query_lower or
                '2026' in query_lower
        ):
            return True

        return False

    async def search_web(self, query: str, max_results: int = 5) -> list:
        """Поиск с фильтрацией нерелевантных результатов"""
        try:
            # Улучшаем поисковый запрос
            refined_query = self.refine_search_query(query)
            results = self.ddgs.text(refined_query, max_results=max_results * 2)  # Берём больше для фильтрации

            # Фильтруем нерелевантные результаты
            filtered = []
            query_keywords = set(re.findall(r'\w+', query.lower()))

            for result in results:
                title_keywords = set(re.findall(r'\w+', result.get('title', '').lower()))
                body_keywords = set(re.findall(r'\w+', result.get('body', '').lower()))

                # Проверяем пересечение ключевых слов
                if len(query_keywords & (title_keywords | body_keywords)) >= 2:
                    filtered.append(result)
                    if len(filtered) >= max_results:
                        break

            return filtered[:max_results]
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

    def refine_search_query(self, query: str) -> str:
        """Улучшение поискового запроса"""
        query = query.strip()
        # Убираем лишние слова-паразиты
        for prefix in ['скажи', 'расскажи', 'объясни', 'пожалуйста', 'можешь']:
            if query.lower().startswith(prefix):
                query = re.sub(f'^{prefix}\\s+', '', query, flags=re.IGNORECASE)
        return query + ' 2026' if '2026' not in query and any(
            kw in query.lower() for kw in ['новости', 'события', 'погода']) else query

    async def get_llm_response(self, messages: list) -> str:
        """Получение ответа от локальной модели"""
        try:
            response = requests.post(
                LM_STUDIO_API_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {LM_STUDIO_API_KEY}'
                },
                json={
                    'messages': messages,
                    'temperature': 0.3,  # Снижаем для более точных ответов
                    'max_tokens': 1500,
                    'stream': False
                },
                timeout=90
            )

            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                return f"❌ Ошибка обращения к модели (код {response.status_code})"

        except requests.exceptions.Timeout:
            return "❌ Таймаут при обращении к модели. Попробуйте ещё раз."
        except Exception as e:
            return f"❌ Ошибка модели: {str(e)}"

    async def process_message(self, user_message: str) -> str:
        """Умная обработка сообщения"""
        current_time_info = self.get_current_datetime_info()

        # Проверяем, спрашивает ли пользователь о текущей дате/времени
        if any(kw in user_message.lower() for kw in self.no_search_keywords):
            # Отвечаем напрямую системной информацией
            return (
                f"📅 **Текущая дата:**\n"
                f"{current_time_info.split(chr(10))[1]}\n"
                f"{current_time_info.split(chr(10))[2]}"
            )

        # Определяем, нужен ли поиск
        needs_search = self.needs_web_search(user_message)

        if needs_search:
            # Выполняем поиск
            search_results = await self.search_web(user_message)

            if search_results:
                # Формируем контекст для модели
                context = "РЕЗУЛЬТАТЫ ПОИСКА В ИНТЕРНЕТЕ:\n\n"
                for i, result in enumerate(search_results, 1):
                    context += f"Источник {i}:\n"
                    context += f"Заголовок: {result.get('title', 'Без названия')}\n"
                    context += f"URL: {result.get('href', 'Нет ссылки')}\n"
                    context += f"Описание: {result.get('body', 'Нет описания')}\n\n"

                system_prompt = (
                    f"Ты — умный ассистент. Текущее системное время: {datetime.now().strftime('%A, %d %B %Y, %H:%M')}\n\n"
                    f"ИНСТРУКЦИИ:\n"
                    f"1. Используй ТОЛЬКО предоставленную информацию из интернета для ответа\n"
                    f"2. Если информация не релевантна запросу — СКАЖИ ЭТО ЧЕТКО и не выдумывай ответ\n"
                    f"3. Всегда указывай, что информация получена из интернета и может быть устаревшей\n"
                    f"4. Для вопросов о текущей дате/времени используй системное время выше, а НЕ данные из поиска\n"
                    f"5. Будь кратким и точным"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_message}\n\n{context}\n\nДай точный ответ на основе этих данных:"}
                ]

                response = await self.get_llm_response(messages)

                # Добавляем источники
                sources = "\n\n🔍 **Источники:**\n"
                for i, result in enumerate(search_results[:3], 1):
                    title = result.get('title', 'Источник')[:60] + '...' if len(
                        result.get('title', '')) > 60 else result.get('title', 'Источник')
                    sources += f"{i}. [{title}]({result.get('href', '')})\n"

                return response + sources
            else:
                # Нет релевантных результатов — отвечаем без поиска
                messages = [
                    {"role": "system",
                     "content": f"Ты — полезный ассистент. Текущее время: {datetime.now().strftime('%d.%m.%Y %H:%M')}"},
                    {"role": "user", "content": user_message}
                ]
                return await self.get_llm_response(
                    messages) + "\n\n⚠️ Не удалось найти актуальную информацию в интернете."
        else:
            # Обычный запрос без поиска
            messages = [
                {"role": "system",
                 "content": f"Ты — полезный ассистент. Текущая дата: {datetime.now().strftime('%A, %d %B %Y')}. Отвечай точно и кратко."},
                {"role": "user", "content": user_message}
            ]
            return await self.get_llm_response(messages)


# Инициализация бота
bot = SmartWebSearchBot()


# Обработчики
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Привет! Я умный ИИ-ассистент с доступом к интернету.\n\n"
        "✅ Могу отвечать на вопросы о текущей дате/времени без поиска\n"
        "✅ Ищу актуальную информацию в интернете при необходимости\n"
        "✅ Фильтрую нерелевантные результаты поиска\n\n"
        "Просто задайте вопрос!",
        parse_mode='Markdown'
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    # Отправляем "печатает..." статус
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    try:
        response = await bot.process_message(user_message)

        # Отправляем ответ частями, если он слишком длинный (>4096 символов)
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                await update.message.reply_text(
                    response[i:i + 4096],
                    parse_mode='Markdown',
                    disable_web_page_preview=False
                )
        else:
            await update.message.reply_text(
                response,
                parse_mode='Markdown',
                disable_web_page_preview=False
            )

    except Exception as e:
        await update.message.reply_text(
            f"❌ Произошла ошибка: {str(e)}\n\nПопробуйте задать вопрос иначе.",
            parse_mode='Markdown'
        )


def main():
    print("🚀 Запуск умного бота с интернет-поиском...")
    print(f"⏰ Текущее системное время: {datetime.now().strftime('%A, %d %B %Y %H:%M:%S')}")

    if not TELEGRAM_TOKEN:
        raise ValueError("❌ Не найден TELEGRAM_TOKEN в .env файле!")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Бот запущен и готов к работе!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()