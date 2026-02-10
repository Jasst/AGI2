# coding: utf-8
"""
telegram_bot.py — TELEGRAM БОТ ДЛЯ КОГНИТИВНОЙ СИСТЕМЫ

Интеграция продвинутого когнитивного агента с Telegram.
Поддерживает все функции основной системы через удобный интерфейс.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
import os
from pathlib import Path

# Импортируем когнитивную систему из документа пользователя
import sys

sys.path.append(str(Path(__file__).parent))

try:
    # ✅ СТАЛО (правильно):
    from AGI24 import (
        EnhancedAutonomousAgent,
        extract_semantic_features
    )

    # Config не используется в боте — уберите его из импорта

    print("✅ Когнитивная система успешно импортирована")
except ImportError as e:
    print(f"⚠️ Ошибка импорта AGI_v29_Enhanced.py: {e}")
    print("Убедитесь что файл AGI_v29_Enhanced.py находится в той же папке")
    sys.exit(1)

# Telegram Bot API
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters
    )
except ImportError:
    print("⚠️ Установите python-telegram-bot: pip install python-telegram-bot --break-system-packages")
    sys.exit(1)


# ================= КОНФИГУРАЦИЯ БОТА =================

class BotConfig:
    """Конфигурация Telegram бота"""

    @staticmethod
    def get_telegram_token() -> str:
        """Получение токена из .env"""
        # Проверяем переменную окружения
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            return token

        # Проверяем файл .env
        env_path = Path(".env")
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TELEGRAM_BOT_TOKEN="):
                            return line.split("=", 1)[1].strip('"\' ')
            except Exception as e:
                print(f"⚠️ Ошибка чтения .env: {e}")

        raise ValueError(
            "TELEGRAM_BOT_TOKEN не найден в .env файле.\n"
            "Добавьте строку: TELEGRAM_BOT_TOKEN=ваш_токен"
        )

    # Ограничения
    MAX_MESSAGE_LENGTH = 4096  # Лимит Telegram
    MAX_RESPONSE_CHUNKS = 5
    TYPING_DELAY = 1.5  # Секунды до ответа (реалистичность)


# ================= ХРАНИЛИЩЕ СЕССИЙ =================

class UserSessionManager:
    """Управление пользовательскими сессиями"""

    def __init__(self):
        self.sessions: Dict[int, Dict] = {}
        self.global_agent: Optional[EnhancedAutonomousAgent] = None

    async def get_or_create_session(self, user_id: int) -> Dict:
        """Получить или создать сессию пользователя"""
        if user_id not in self.sessions:
            print(f"🆕 Создание новой сессии для пользователя {user_id}")

            # Инициализируем глобального агента если нужно
            if self.global_agent is None:
                self.global_agent = EnhancedAutonomousAgent()

            self.sessions[user_id] = {
                'agent': self.global_agent,  # Можно сделать отдельного агента для каждого
                'created_at': datetime.now(),
                'message_count': 0,
                'last_activity': datetime.now()
            }

        # Обновляем активность
        self.sessions[user_id]['last_activity'] = datetime.now()
        return self.sessions[user_id]

    def get_stats(self) -> Dict:
        """Статистика сессий"""
        total_messages = sum(s['message_count'] for s in self.sessions.values())
        return {
            'active_users': len(self.sessions),
            'total_messages': total_messages
        }


# Глобальный менеджер сессий
session_manager = UserSessionManager()


# ================= УТИЛИТЫ =================

def split_message(text: str, max_length: int = BotConfig.MAX_MESSAGE_LENGTH) -> list:
    """Разбивает длинное сообщение на части"""
    if len(text) <= max_length:
        return [text]

    parts = []
    current = ""

    # Разбиваем по абзацам
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_length:
            current += para + '\n\n'
        else:
            if current:
                parts.append(current.strip())
            current = para + '\n\n'

    if current:
        parts.append(current.strip())

    return parts


def create_main_keyboard() -> InlineKeyboardMarkup:
    """Создание главной клавиатуры"""
    keyboard = [
        [
            InlineKeyboardButton("🧠 Глубокое мышление", callback_data="deep_think"),
            InlineKeyboardButton("🔍 Анализ", callback_data="analysis")
        ],
        [
            InlineKeyboardButton("📊 Статистика", callback_data="stats"),
            InlineKeyboardButton("🎯 Цели", callback_data="goals")
        ],
        [
            InlineKeyboardButton("💡 Инсайты", callback_data="insights"),
            InlineKeyboardButton("🔗 Паттерны", callback_data="patterns")
        ],
        [
            InlineKeyboardButton("📚 Факты", callback_data="facts"),
            InlineKeyboardButton("❓ Помощь", callback_data="help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ================= ОБРАБОТЧИКИ КОМАНД =================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    user_id = user.id

    # Создаём сессию
    await session_manager.get_or_create_session(user_id)

    welcome_text = f"""👋 Привет, {user.first_name}!

🧠 Я — продвинутый когнитивный агент с расширенными возможностями:

✨ **Мои способности:**
• Многоуровневое мышление
• Контекстная память
• Обнаружение паттернов
• Предсказательный анализ
• Творческое решение задач

💬 Просто напиши мне что-нибудь, и я помогу!

📌 Используй кнопки ниже для доступа к функциям или команды:
/help — список команд
/stats — статистика
/think — глубокое мышление
/clear — очистить контекст"""

    await update.message.reply_text(
        welcome_text,
        reply_markup=create_main_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """📖 **ДОСТУПНЫЕ КОМАНДЫ:**

**Основные:**
/start — начать работу
/help — это сообщение
/stats — полная статистика системы
/clear — очистить контекст разговора

**Когнитивные функции:**
/think — активировать глубокое мышление
/analyze — анализ текущего состояния
/goals — показать иерархию целей
/patterns — обнаруженные паттерны
/insights — инсайты из мыслей
/facts — показать сохранённые факты

**Поиск:**
/search <запрос> — поиск в памяти

**Примеры использования:**
• Просто пиши любые вопросы или команды
• "Запомни что Python — мой любимый язык"
• "Сколько будет 25 * 34?"
• "Придумай креативное решение для..."
• "Проанализируй что я часто спрашиваю"

💡 Я запоминаю контекст и учусь на наших диалогах!"""

    await update.message.reply_text(help_text, parse_mode='Markdown')


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stats"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    # Показываем индикатор печати
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Получаем статистику
    stats = agent._get_comprehensive_stats()

    # Добавляем статистику бота
    bot_stats = session_manager.get_stats()
    stats += f"\n\n🤖 **Статистика бота:**"
    stats += f"\nАктивных пользователей: {bot_stats['active_users']}"
    stats += f"\nВсего сообщений: {bot_stats['total_messages']}"
    stats += f"\nСообщений в вашей сессии: {session['message_count']}"

    # Разбиваем на части если нужно
    parts = split_message(stats)
    for part in parts:
        await update.message.reply_text(part)


async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /think"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    await update.message.reply_text("🧠 Активирую глубокое многоуровневое мышление...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Запускаем глубокое мышление
    await agent._deep_autonomous_thinking()

    await update.message.reply_text("✅ Глубокое мышление завершено! Проверьте /insights для результатов.")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /analyze"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    analysis = agent._get_comprehensive_analysis()
    parts = split_message(analysis)

    for part in parts:
        await update.message.reply_text(part)


async def goals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /goals"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    goals = agent._format_goal_hierarchy()
    parts = split_message(goals)

    for part in parts:
        await update.message.reply_text(part)


async def patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /patterns"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    patterns = agent._format_patterns()
    parts = split_message(patterns)

    for part in parts:
        await update.message.reply_text(part)


async def insights_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /insights"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    insights = agent._format_insights()
    parts = split_message(insights)

    for part in parts:
        await update.message.reply_text(part)


async def facts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /facts"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    # Получаем факты из базы
    facts = agent.db.get_relevant_facts("все факты", limit=20)

    if not facts:
        await update.message.reply_text("📚 Фактов пока не сохранено.")
        return

    # Группируем по категориям
    from collections import defaultdict
    categories = defaultdict(list)
    for fact in facts:
        categories[fact.get('category', 'разное')].append(fact)

    lines = ["📚 **СОХРАНЁННЫЕ ФАКТЫ:**\n"]

    for category, category_facts in categories.items():
        lines.append(f"\n📌 **{category.upper()}:**")
        for fact in category_facts[:5]:
            confidence_stars = "★" * int(fact['confidence'] * 5)
            lines.append(f"• {fact['key']}: {fact['value']} [{confidence_stars}]")

        if len(category_facts) > 5:
            lines.append(f"... и ещё {len(category_facts) - 5}")

    text = "\n".join(lines)
    parts = split_message(text)

    for part in parts:
        await update.message.reply_text(part)




async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clear"""
    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    # Очищаем контекстное окно
    agent.context_window.clear()

    await update.message.reply_text("🧹 Контекст разговора очищен. Начинаем с чистого листа!")


# ================= ОБРАБОТЧИК КНОПОК =================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']

    callback_data = query.data

    # Показываем индикатор
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = ""

    if callback_data == "deep_think":
        await query.message.reply_text("🧠 Активирую глубокое мышление...")
        await agent._deep_autonomous_thinking()
        response = "✅ Глубокое мышление завершено!"

    elif callback_data == "analysis":
        response = agent._get_comprehensive_analysis()

    elif callback_data == "stats":
        response = agent._get_comprehensive_stats()
        bot_stats = session_manager.get_stats()
        response += f"\n\n🤖 Активных пользователей: {bot_stats['active_users']}"

    elif callback_data == "goals":
        response = agent._format_goal_hierarchy()

    elif callback_data == "insights":
        response = agent._format_insights()

    elif callback_data == "patterns":
        response = agent._format_patterns()

    elif callback_data == "facts":
        facts = agent.db.get_relevant_facts("все", limit=15)
        if facts:
            response = "📚 **ФАКТЫ:**\n\n"
            for fact in facts[:10]:
                response += f"• {fact['key']}: {fact['value']}\n"
        else:
            response = "Фактов пока нет."

    elif callback_data == "help":
        response = """📖 **ПОМОЩЬ:**

Просто пиши мне сообщения, и я буду помогать!

Используй команды:
/help — подробная справка
/stats — статистика
/think — глубокое мышление
/search <запрос> — поиск

Или используй кнопки для быстрого доступа к функциям."""

    # Отправляем ответ
    parts = split_message(response)
    for part in parts:
        await query.message.reply_text(part)


# ================= ОБРАБОТЧИК СООБЩЕНИЙ =================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Основной обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    user_message = update.message.text

    # Получаем сессию
    session = await session_manager.get_or_create_session(user_id)
    agent = session['agent']
    session['message_count'] += 1

    # Показываем индикатор печати
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Имитируем задержку для реалистичности
    await asyncio.sleep(BotConfig.TYPING_DELAY)

    try:
        # Обрабатываем сообщение через агента
        response = await agent.process_input(user_message)

        # Если это команда выхода
        if response == "SYSTEM_EXIT":
            await update.message.reply_text("👋 До встречи!")
            return

        # Разбиваем длинный ответ
        parts = split_message(response, BotConfig.MAX_MESSAGE_LENGTH)

        # Ограничиваем количество частей
        if len(parts) > BotConfig.MAX_RESPONSE_CHUNKS:
            parts = parts[:BotConfig.MAX_RESPONSE_CHUNKS]
            parts.append("... *(ответ слишком длинный, показана часть)*")

        # Отправляем ответ
        for i, part in enumerate(parts):
            await update.message.reply_text(part)

            # Задержка между частями
            if i < len(parts) - 1:
                await asyncio.sleep(0.5)

        # После каждого ответа показываем клавиатуру
        if session['message_count'] % 5 == 0:
            await update.message.reply_text(
                "💡 Что ещё могу сделать?",
                reply_markup=create_main_keyboard()
            )

    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {e}")
        await update.message.reply_text(
            f"⚠️ Произошла ошибка при обработке. Попробуйте ещё раз.\n\nОшибка: {str(e)[:100]}"
        )


# ================= ОБРАБОТКА ОШИБОК =================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logging.error(f"Update {update} caused error {context.error}")

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ Произошла ошибка. Попробуйте позже или используйте /start для перезапуска."
        )


# ================= ГЛАВНАЯ ФУНКЦИЯ =================

async def main():
    """Запуск бота"""
    print("=" * 70)
    print("🤖 ЗАПУСК TELEGRAM БОТА С КОГНИТИВНОЙ СИСТЕМОЙ")
    print("=" * 70)

    # Настройка логирования
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

        level=logging.DEBUG
    )

    try:
        # Получаем токен
        token = BotConfig.get_telegram_token()
        print(f"✅ Токен получен: {token[:10]}...")

        # Создаём приложение
        app = Application.builder().token(token).build()

        # Регистрируем обработчики команд
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("stats", stats_command))
        app.add_handler(CommandHandler("think", think_command))
        app.add_handler(CommandHandler("analyze", analyze_command))
        app.add_handler(CommandHandler("goals", goals_command))
        app.add_handler(CommandHandler("patterns", patterns_command))
        app.add_handler(CommandHandler("insights", insights_command))
        app.add_handler(CommandHandler("facts", facts_command))

        app.add_handler(CommandHandler("clear", clear_command))

        # Обработчик кнопок
        app.add_handler(CallbackQueryHandler(button_callback))

        # Обработчик обычных сообщений
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Обработчик ошибок
        app.add_error_handler(error_handler)

        print("\n✅ Бот запущен и готов к работе!")
        print("📱 Найдите бота в Telegram и напишите /start")
        print("\n🛑 Для остановки нажмите Ctrl+C\n")
        print("=" * 70 + "\n")

        # Запускаем polling
        await app.run_polling(allowed_updates=Update.ALL_TYPES)

    except ValueError as e:
        print(f"\n❌ Ошибка конфигурации: {e}")
        print("\n💡 Добавьте токен в .env файл:")
        print("TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather")
    except Exception as e:
        print(f"\n🚨 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


def run():
    """Точка входа"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n🚨 Ошибка запуска: {e}")


if __name__ == "__main__":
    run()