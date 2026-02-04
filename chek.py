#!/usr/bin/env python3
# coding: utf-8
"""
check_integration.py - Проверка готовности системы к запуску

Запустите этот скрипт перед запуском бота чтобы убедиться что всё настроено правильно.
"""

import sys
from pathlib import Path
import os


def print_status(message: str, status: bool):
    """Красивый вывод статуса"""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {message}")
    return status


def check_files():
    """Проверка наличия необходимых файлов"""
    print("\n📁 Проверка файлов:")

    all_ok = True

    # AGI система
    agi_exists = Path("AGI_v29_Enhanced.py").exists()
    all_ok &= print_status("AGI_v29_Enhanced.py найден", agi_exists)

    # Telegram бот
    bot_exists = Path("telegram_bot.py").exists()
    all_ok &= print_status("telegram_bot.py найден", bot_exists)

    # .env файл
    env_exists = Path(".env").exists()
    all_ok &= print_status(".env файл найден", env_exists)

    if not env_exists:
        print("\n⚠️  Создайте .env файл с содержимым:")
        print("OPENROUTER_API_KEY=ваш_ключ")
        print("TELEGRAM_BOT_TOKEN=8288420211:AAHFhDpqRxZwLSEs5MOAS2_DBlUlhU1MzX8")

    return all_ok


def check_env_variables():
    """Проверка переменных окружения"""
    print("\n🔑 Проверка API ключей:")

    all_ok = True

    # Читаем .env если существует
    env_vars = {}
    if Path(".env").exists():
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip().strip('"\'')

    # Проверяем OpenRouter
    openrouter = env_vars.get("OPENROUTER_API_KEY", "")
    has_openrouter = len(openrouter) > 10 and openrouter != "ваш_ключ"
    all_ok &= print_status(f"OpenRouter API ключ {'найден' if has_openrouter else 'НЕ НАЙДЕН'}", has_openrouter)

    if not has_openrouter:
        print("   💡 Получите ключ на: https://openrouter.ai/keys")

    # Проверяем Telegram
    telegram = env_vars.get("TELEGRAM_BOT_TOKEN", "")
    has_telegram = len(telegram) > 30
    all_ok &= print_status(f"Telegram токен {'найден' if has_telegram else 'НЕ НАЙДЕН'}", has_telegram)

    if has_telegram:
        print(f"   📱 Токен: {telegram[:10]}...{telegram[-10:]}")

    return all_ok


def check_dependencies():
    """Проверка установленных библиотек"""
    print("\n📦 Проверка зависимостей:")

    all_ok = True

    # python-telegram-bot
    try:
        import telegram
        version = telegram.__version__
        all_ok &= print_status(f"python-telegram-bot установлен (v{version})", True)
    except ImportError:
        all_ok &= print_status("python-telegram-bot НЕ установлен", False)
        print("   💡 Установите: pip install python-telegram-bot --break-system-packages")

    # aiohttp
    try:
        import aiohttp
        version = aiohttp.__version__
        all_ok &= print_status(f"aiohttp установлен (v{version})", True)
    except ImportError:
        all_ok &= print_status("aiohttp НЕ установлен", False)
        print("   💡 Установите: pip install aiohttp --break-system-packages")

    # sqlite3 (встроенный)
    try:
        import sqlite3
        all_ok &= print_status("sqlite3 доступен", True)
    except ImportError:
        all_ok &= print_status("sqlite3 НЕ доступен", False)

    return all_ok


def check_agi_import():
    """Проверка возможности импорта AGI системы"""
    print("\n🧠 Проверка когнитивной системы:")

    try:
        from AGI_v29_Enhanced import EnhancedAutonomousAgent, Config
        print_status("AGI_v29_Enhanced импортируется успешно", True)
        return True
    except Exception as e:
        print_status(f"Ошибка импорта AGI: {e}", False)
        return False


def check_telegram_connection():
    """Проверка подключения к Telegram (опционально)"""
    print("\n📡 Проверка связи с Telegram:")

    # Читаем токен
    token = None
    if Path(".env").exists():
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if "TELEGRAM_BOT_TOKEN=" in line:
                    token = line.split("=", 1)[1].strip().strip('"\'')
                    break

    if not token or len(token) < 30:
        print_status("Невозможно проверить - токен не найден", False)
        return False

    try:
        import asyncio
        from telegram import Bot

        async def test_bot():
            bot = Bot(token=token)
            me = await bot.get_me()
            return me

        me = asyncio.run(test_bot())
        print_status(f"Бот @{me.username} доступен", True)
        print(f"   🤖 Имя: {me.first_name}")
        return True

    except Exception as e:
        print_status(f"Ошибка подключения: {e}", False)
        print("   ⚠️  Проверьте токен или интернет соединение")
        return False


def print_summary(results: dict):
    """Итоговая сводка"""
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ ПРОВЕРКА")
    print("=" * 60)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✅ ПРОЙДЕНО" if passed else "❌ ПРОВАЛЕНО"
        print(f"{status}: {check}")

    print("=" * 60)

    if all_passed:
        print("\n🎉 ВСЁ ГОТОВО! Можете запускать бота:")
        print("   python telegram_bot.py")
    else:
        print("\n⚠️  ЕСТЬ ПРОБЛЕМЫ. Исправьте ошибки выше перед запуском.")
        print("\n💡 Частые решения:")
        print("   • Нет .env? → Создайте файл .env с ключами")
        print("   • Нет библиотек? → pip install python-telegram-bot aiohttp --break-system-packages")
        print("   • Не импортируется AGI? → Проверьте что файлы в одной папке")

    print()


def main():
    """Основная функция проверки"""
    print("=" * 60)
    print("🔍 ПРОВЕРКА ГОТОВНОСТИ TELEGRAM БОТА")
    print("=" * 60)

    results = {}

    # Проверки
    results["Файлы"] = check_files()
    results["API ключи"] = check_env_variables()
    results["Зависимости"] = check_dependencies()
    results["AGI система"] = check_agi_import()

    # Опциональная проверка (может быть медленной)
    print("\n❓ Проверить подключение к Telegram? (медленно, но надёжно)")
    choice = input("y/n [n]: ").strip().lower()
    if choice == 'y':
        results["Telegram подключение"] = check_telegram_connection()

    # Итоги
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Проверка прервана")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()