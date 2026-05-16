# test_connection.py
import asyncio
import httpx
import os
from dotenv import load_dotenv

# Загружаем переменные из .env (как в вашем основном проекте)
load_dotenv()


async def test():
    token = os.getenv('TELEGRAM_TOKEN')

    if not token:
        print("❌ TELEGRAM_TOKEN не найден в .env файле!")
        print("📁 Проверьте, что файл .env находится в той же директории")
        print("📝 Формат: TELEGRAM_TOKEN=123456:AAH...")
        return

    # Маскируем токен для безопасного вывода
    masked = f"{token[:10]}...{token[-5:]}" if len(token) > 15 else "***"
    print(f"🔑 Токен загружен: {masked}")

    url = f"https://api.telegram.org/bot{token}/getMe"

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            print("🔄 Подключение к Telegram API...")
            resp = await client.get(url)

            if resp.status_code == 200:
                data = resp.json()
                if data.get('ok'):
                    bot = data['result']
                    print(f"✅ Успех! Бот: @{bot.get('username')} ({bot.get('first_name')})")
                    print(f"🆔 ID: {bot.get('id')}")
                else:
                    print(f"⚠️ API вернул ошибку: {data}")
            else:
                print(f"❌ HTTP {resp.status_code}: {resp.text}")

        except httpx.ConnectTimeout:
            print("❌ Таймаут подключения — проверьте интернет/прокси/блокировки")
        except httpx.ConnectError as e:
            print(f"❌ Ошибка подключения: {e}")
        except httpx.ReadTimeout:
            print("❌ Таймаут чтения — сервер ответил, но медленно")
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("🧪 Тест подключения к Telegram API")
    print("=" * 50)
    asyncio.run(test())