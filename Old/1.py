# ================= ОБРАБОТЧИКИ TELEGRAM =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome = (
        f"👋 Привет, {user.first_name}!\n\n"
        "🧠 Я — **Cognitive Agent Pro v7.0** — улучшенный когнитивный ассистент.\n\n"

        "**✨ Основные возможности:**\n"
        "• 🤯 Трёхпроходное мышление (черновик → критика → коррекция)\n"
        "• 🔍 Осознанный веб-поиск (только при необходимости)\n"
        "• 📊 Метакогнитивный мониторинг с автокоррекцией\n"
        "• 🎯 Многоуровневая декомпозиция целей\n"
        "• 🧠 Эпизодическая память с временной привязкой\n"
        "• 🔮 Проактивное мышление и предсказания\n"
        "• 💾 Семантический кэш ответов\n"
        "• ⏰ Точное системное время (без поиска!)\n\n"

        "**📌 Команды:**\n"
        "• /think — запустить проактивное мышление\n"
        "• /stats — показать статистику\n"
        "• /help — справка по возможностям\n"
        "• /clear — очистить контекст\n\n"

        "**🔒 Приватность:**\n"
        "• Локальные запросы не покидают ваш компьютер\n"
        "• Веб-поиск только при явной необходимости\n"
        "• Все данные хранятся локально\n\n"

        "Готов помочь с любыми вопросами! 🚀"
    )

    await update.message.reply_text(welcome, reply_markup=create_main_keyboard())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 **Cognitive Agent Pro v7.0**\n\n"
        "**Основные команды:**\n"
        "• Просто напишите сообщение — обычный диалог\n"
        "• `/think` — глубокий анализ и проактивные мысли\n"
        "• `/stats` — ваша персональная статистика\n"
        "• `/clear` — очистить контекст диалога\n\n"

        "**Как это работает:**\n"
        "1. 🏗️ Генерация чернового ответа\n"
        "2. 🔎 Самокритика и оценка уверенности\n"
        "3. 🌐 Веб-поиск (если нужно)\n"
        "4. ✨ Синтез финального ответа\n"
        "5. 📊 Оценка качества и автокоррекция\n"
        "6. 💾 Сохранение в память\n\n"

        "**Особенности:**\n"
        "• Автоматически определяет, нужен ли поиск\n"
        "• Аннотирует ответы (факты/предположения)\n"
        "• Запоминает важные эпизоды\n"
        "• Генерирует проактивные мысли\n"
        "• Мониторит и улучшает качество ответов\n\n"

        "Примеры запросов:\n"
        "• «Какая погода сегодня?»\n"
        "• «Объясни квантовую физику»\n"
        "• «Помоги спланировать проект»\n"
        "• «Какие тренды в ИИ?»\n"
    )

    await update.message.reply_text(help_text, reply_markup=create_main_keyboard())

# ================= ОСНОВНАЯ ФУНКЦИЯ =================
def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Config.LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

async def main():
    """Основная функция"""
    setup_logging()

    print("\n" + "=" * 70)
    print("🚀 COGNITIVE AGENT PRO v7.0 — ЗАПУСК")
    print("=" * 70)

    # Загрузка конфигурации
    try:
        token = Config.get_telegram_token()
        lm_config = Config.get_lmstudio_config()
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return

    # Инициализация компонентов
    print("\n🔧 Инициализация компонентов...")

    db = EnhancedCognitiveDB(Config.DB_PATH)
    llm = EnhancedLLMInterface(lm_config)
    search_engine = EnhancedWebSearchEngine() if HAS_WEB_SEARCH else None

    print(f"   ✅ База данных: {Config.DB_PATH}")
    print(f"   ✅ LM Studio: {lm_config['model']}")
    print(f"   ✅ Веб-поиск: {'активен' if search_engine else 'отключён'}")

    # Создание приложения Telegram
    application = (
        ApplicationBuilder()
        .token(token)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(15)
        .pool_timeout(15)
        .build()
    )

    # Сохранение менеджера сессий
    application.bot_data['session_manager'] = SessionManager(db, llm, search_engine)

    # Регистрация обработчиков (используем только английские имена команд)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("think", think_command))
    application.add_handler(CommandHandler("stats", analyze_command))
    application.add_handler(CommandHandler("clear", clear_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    # Настройка команд меню (используем только английские имена)
    await application.bot.set_my_commands([
        BotCommand("start", "Запустить бота"),
        BotCommand("think", "Глубокое мышление"),
        BotCommand("stats", "Статистика"),
        BotCommand("help", "Справка"),
        BotCommand("clear", "Очистить контекст")
    ])

    # Запуск
    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

    # Информация о запуске
    print("\n" + "=" * 70)
    print("✅ COGNITIVE AGENT PRO ЗАПУЩЕН УСПЕШНО!")
    print("=" * 70)
    print(f"\n🤖 Модель: {lm_config['model']}")
    print(f"🔗 Сервер: {lm_config['url']}")
    print(f"🌐 Веб-поиск: {'✅ активирован' if search_engine else '❌ отключён'}")
    print(f"💾 Кэш: семантический (до 1000 записей)")
    print(f"🧠 Память: эпизодическая с временной привязкой")
    print(f"⚡ Мышление: трёхпроходное с автокоррекцией")

    print("\n📱 **Инструкция:**")
    print("   1. Откройте Telegram")
    print("   2. Найдите вашего бота")
    print("   3. Отправьте /start")
    print("   4. Начните диалог!")

    print("\n🔧 **Архитектура:**")
    print("   • Трёхпроходное мышление")
    print("   • Самокритика через отдельный движок")
    print("   • Осознанный поиск только при необходимости")
    print("   • Метакогнитивный мониторинг с коррекцией")
    print("   • Проактивное мышление и предсказания")
    print("   • Семантический кэш ответов")

    print("\n🛑 Для остановки нажмите Ctrl+C")
    print("=" * 70 + "\n")

    # Основной цикл
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await application.stop()

def run():
    """Точка входа"""
    print("Cognitive Agent Pro v7.0 — Продвинутая когнитивная система")
    print(f"Python {sys.version.split()[0]}")

    # Проверка зависимостей
    required_packages = ['aiohttp', 'requests']
    missing = []

    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing)}")
        print(f"📦 Установите: pip install {' '.join(missing)}")
        return

    if not HAS_WEB_SEARCH:
        print("⚠️  Веб-поиск недоступен. Для активации:")
        print("    pip install duckduckgo-search aiofiles")

    print("\n🚀 Запуск когнитивного агента...\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Работа завершена")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()