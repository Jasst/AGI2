#!/usr/bin/env python3
"""
Скрипт для объединения всех частей Enhanced AGI Brain v31.0 в один файл
"""

import os
from pathlib import Path


def merge_files():
    """Объединение всех частей в один файл"""

    # Файлы для объединения (в правильном порядке)
    files = [
        'temporal_brain_v31_enhanced.py',
        'temporal_brain_v31_part2.py',
        'temporal_brain_v31_part3.py',
        'temporal_brain_v31_final.py',
    ]

    output_file = 'temporal_brain_v31_complete.py'

    print("🔧 Объединение файлов Enhanced AGI Brain v31.0...")
    print()

    # Проверяем наличие всех файлов
    missing = []
    for file in files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("❌ Отсутствуют файлы:")
        for file in missing:
            print(f"  - {file}")
        return False

    # Объединяем файлы
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, filename in enumerate(files):
            print(f"📄 Обработка {filename}...")

            with open(filename, 'r', encoding='utf-8') as infile:
                content = infile.read()

                # Для первого файла пишем как есть
                if i == 0:
                    outfile.write(content)
                else:
                    # Для остальных пропускаем imports и shebang
                    lines = content.split('\n')

                    # Находим первую строку после imports
                    start_idx = 0
                    for idx, line in enumerate(lines):
                        # Пропускаем комментарии в начале
                        if line.strip().startswith('#'):
                            continue
                        # Пропускаем пустые строки
                        if not line.strip():
                            continue
                        # Пропускаем imports
                        if line.startswith('import ') or line.startswith('from '):
                            continue

                        # Нашли начало основного кода
                        start_idx = idx
                        break

                    # Пишем основной код
                    outfile.write('\n\n')
                    outfile.write('\n'.join(lines[start_idx:]))

            print(f"  ✅ {filename} добавлен")

    print()
    print(f"✅ Все файлы объединены в {output_file}")

    # Проверяем размер
    size = os.path.getsize(output_file)
    size_kb = size / 1024
    print(f"📊 Размер итогового файла: {size_kb:.1f} KB ({size:,} байт)")

    # Считаем строки
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = len(f.readlines())

    print(f"📏 Количество строк: {lines:,}")

    print()
    print("🚀 Готово! Теперь можно запустить:")
    print(f"   python {output_file}")

    return True


def create_env_template():
    """Создание шаблона .env файла"""
    env_content = """# Enhanced AGI Brain v31.0 Configuration

# Telegram Bot Token (получите у @BotFather)
TELEGRAM_TOKEN=your_telegram_bot_token_here

# LM Studio API настройки
LM_STUDIO_API_URL=http://localhost:1234/v1/chat/completions
LM_STUDIO_API_KEY=lm-studio

# Debug режим (true/false)
DEBUG_MODE=false

# Базовая директория для хранения данных
BASE_DIR=temporal_brain_v31
"""

    env_file = '.env.template'

    if not Path(env_file).exists():
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"📝 Создан шаблон {env_file}")
        print("   Скопируйте его в .env и заполните значения")

    return True


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🛠️  Enhanced AGI Brain v31.0 - Merge Tool                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Объединение файлов
    success = merge_files()

    if success:
        print()
        # Создание шаблона .env
        create_env_template()

        print()
        print("📚 Дополнительные шаги:")
        print("   1. Скопируйте .env.template в .env")
        print("   2. Заполните TELEGRAM_TOKEN в .env")
        print("   3. Убедитесь, что LM Studio запущен")
        print("   4. python temporal_brain_v31_complete.py")
        print()
        print("📖 Подробная документация: README_v31.md")
    else:
        print()
        print("❌ Не удалось объединить файлы")
        print("   Убедитесь, что все части находятся в текущей директории")