# coding: utf-8
"""
AGI_Hybrid_v22_OPENROUTER_FIXED.py — ИСПРАВЛЕНО (февраль 2026)
"""

import os, sys, time, requests
from pathlib import Path


# ================= ЗАГРУЗКА КЛЮЧА (до определения класса) =================
def load_api_key():
    """Загружает ключ из переменной окружения или файла .env"""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "OPENROUTER_API_KEY":
                                key = v.strip().strip('"').strip("'")
    return key


# ================= КОНФИГУРАЦИЯ =================
class Config:
    ROOT = Path("./cognitive_v22")
    ROOT.mkdir(exist_ok=True)

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = load_api_key()  # ✅ Ключ загружается ДО использования

    # ✅ ВЫБЕРИТЕ МОДЕЛЬ (без :free!)
    MODEL = "meta-llama/llama-3.2-3b-instruct"  # Рекомендуется — бесплатная и стабильная

    # Альтернативы (раскомментируйте нужную):
    # MODEL = "google/gemma-2-2b-it"
    # MODEL = "qwen/qwen-2.5-7b-instruct"  # Использует $1 кредит


# ================= ДИАГНОСТИКА =================
def diagnose():
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА OpenRouter")
    print("=" * 60)

    if not Config.OPENROUTER_API_KEY:
        print("❌ ОШИБКА: Не найден OPENROUTER_API_KEY!")
        print("\nКак исправить:")
        print("1. Создайте файл .env в папке проекта со строкой:")
        print("   OPENROUTER_API_KEY=sk-or-v1-ваш_ключ_здесь")
        print("2. Или установите переменную окружения:")
        print("   Windows PowerShell: $env:OPENROUTER_API_KEY='sk-or-v1-...'")
        print("   Linux/Mac Terminal: export OPENROUTER_API_KEY='sk-or-v1-...'")
        print("\nПолучить ключ: https://openrouter.ai/settings/keys")
        return False

    print(f"✅ Ключ загружен: {Config.OPENROUTER_API_KEY[:8]}...{Config.OPENROUTER_API_KEY[-4:]}")
    print(f"✅ Модель: {Config.MODEL}")
    print(f"✅ URL: {Config.OPENROUTER_URL}")

    # Тестовый запрос
    print("\n📡 Отправляю тестовый запрос...")
    try:
        r = requests.post(
            Config.OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",  # ОБЯЗАТЕЛЬНО!
                "X-Title": "CognitiveSystem"
            },
            json={
                "model": Config.MODEL,
                "messages": [{"role": "user", "content": "Привет! Ты работаешь?"}],
                "max_tokens": 30
            },
            timeout=15
        )

        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            print(f"✅ УСПЕХ! Ответ модели:\n   {content.strip()}")
            return True
        else:
            error_msg = r.json().get("error", {}).get("message", r.text[:300])
            print(f"❌ Ошибка {r.status_code}: {error_msg}")

            if "invalid model" in error_msg.lower() or "not found" in error_msg.lower():
                print("\n💡 ИСПРАВЛЕНИЕ: Используйте проверенные модели:")
                print("   • meta-llama/llama-3.2-3b-instruct  ← РЕКОМЕНДУЕТСЯ")
                print("   • google/gemma-2-2b-it")
                print("   • qwen/qwen-2.5-7b-instruct")
            return False

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False


# ================= ДИАЛОГ =================
def chat_loop():
    print("\n" + "=" * 60)
    print(f"💬 Диалог с моделью: {Config.MODEL}")
    print("=" * 60)
    print("Введите 'exit' для выхода\n")

    while True:
        try:
            q = input("Вопрос: ").strip()
            if q.lower() in ("exit", "выход", "quit"):
                print("\n👋 До свидания!")
                break
            if not q:
                continue

            print("\n🧠 Думаю...\n")

            try:
                r = requests.post(
                    Config.OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "CognitiveSystemV22"
                    },
                    json={
                        "model": Config.MODEL,
                        "messages": [
                            {"role": "system", "content": "Отвечай кратко и по делу на русском языке."},
                            {"role": "user", "content": q}
                        ],
                        "temperature": 0.4,
                        "max_tokens": 400
                    },
                    timeout=25
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]

                # Эффект печатания
                for c in content:
                    print(c, end="", flush=True)
                    time.sleep(0.01)
                print("\n")

            except requests.exceptions.HTTPError as e:
                try:
                    error_msg = r.json().get("error", {}).get("message", str(e))
                except:
                    error_msg = str(e)
                print(f"❌ Ошибка API: {error_msg}")
            except Exception as e:
                print(f"❌ Ошибка: {e}")

        except KeyboardInterrupt:
            print("\n\n👋 Диалог прерван.")
            break


# ================= ЗАПУСК =================
if __name__ == "__main__":
    # Поддержка кириллицы в Windows
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass

    print("\n" + "=" * 60)
    print("🧠 COGNITIVE SYSTEM v22 — OpenRouter Edition")
    print("=" * 60)

    if diagnose():
        chat_loop()
    else:
        print("\n" + "=" * 60)
        print("🛠️  ИНСТРУКЦИЯ ПО ИСПРАВЛЕНИЮ")
        print("=" * 60)
        print("1. Создайте файл .env в папке проекта (рядом с этим скриптом)")
        print("2. Вставьте строку (замените на ваш ключ):")
        print("   OPENROUTER_API_KEY=sk-or-v1-ваш_ключ_здесь")
        print("\n3. Убедитесь, что модель указана БЕЗ ':free':")
        print("   ✅ Правильно: MODEL = 'meta-llama/llama-3.2-3b-instruct'")
        print("   ❌ Неправильно: MODEL = 'model-name:free'")
        print("\n4. Сохраните файл и запустите скрипт снова")