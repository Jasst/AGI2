# AUTO-GENERATED CORE - Self-learning AI
# Created: 2026-02-11T02:35:00.000000

import os
import re
from __main__ import KnowledgeCore, CoreResponse
from datetime import datetime


class FileStorageCore(KnowledgeCore):
    name = "file_storage_core"
    description = "Работа с текстовыми файлами"
    capabilities = ["сохранить файл", "прочитать файл", "список файлов"]
    priority = 1  # Высокий приоритет
    direct_answer_mode = True

    def __init__(self):
        self.storage_dir = "user_files"
        os.makedirs(self.storage_dir, exist_ok=True)
        print(f"📁 Инициализировано файловое хранилище: {self.storage_dir}")

    def can_handle(self, query: str) -> bool:
        q = query.lower().replace('фаил', 'файл')
        return any(word in q for word in ['файл', 'документ', 'сохрани', 'прочитай', 'прочти', 'открой'])

    def get_confidence(self, query: str) -> float:
        q = query.lower().replace('фаил', 'файл')
        if 'прочитай файл' in q or 'прочти файл' in q:
            return 0.95
        if 'сохрани в файл' in q:
            return 0.9
        return 0.8 if self.can_handle(query) else 0.0

    def _get_file_path(self, filename: str) -> str:
        """Безопасное получение пути к файлу"""
        # Очищаем имя файла
        filename = re.sub(r'[<>:"|?*]', '', filename)
        filename = filename.strip().strip('.')
        if not filename.endswith('.txt'):
            filename += '.txt'
        return os.path.join(self.storage_dir, filename)

    def execute(self, query: str, context=None) -> CoreResponse:
        try:
            q = query.lower().replace('фаил', 'файл')

            # 1. ЧТЕНИЕ ФАЙЛА
            if 'прочитай' in q or 'прочти' in q or 'открой' in q:
                # Извлекаем имя файла
                filename = ""
                patterns = [
                    r'прочитай файл\s+([^\s,.!?]+)',
                    r'прочти файл\s+([^\s,.!?]+)',
                    r'открой файл\s+([^\s,.!?]+)',
                    r'файл\s+([^\s,.!?]+)\s+прочитай'
                ]

                for pattern in patterns:
                    match = re.search(pattern, q, re.IGNORECASE)
                    if match:
                        filename = match.group(1).strip()
                        break

                if not filename:
                    # Пробуем найти любое слово после "файл"
                    match = re.search(r'файл\s+(\S+)', query, re.IGNORECASE)
                    if match:
                        filename = match.group(1).strip('"\'.,!?')

                if not filename:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Не указано имя файла'},
                        raw_result="❌ Пожалуйста, укажите имя файла.\nПример: 'прочитай файл привет.txt'",
                        confidence=0.0,
                        source=self.name
                    )

                filepath = self._get_file_path(filename)

                if not os.path.exists(filepath):
                    # Показываем список доступных файлов
                    files = os.listdir(self.storage_dir)
                    if files:
                        file_list = "\n".join([f"• {f}" for f in files[:5]])
                        return CoreResponse(
                            success=False,
                            data={'error': 'Файл не найден', 'available_files': files},
                            raw_result=f"❌ Файл '{filename}' не найден.\n\n📁 Доступные файлы:\n{file_list}\n\nЧтобы создать файл: 'сохрани в файл {filename}: ваш текст'",
                            confidence=0.0,
                            source=self.name
                        )
                    else:
                        return CoreResponse(
                            success=False,
                            data={'error': 'Файл не найден'},
                            raw_result=f"❌ Файл '{filename}' не найден.\n\n📁 Хранилище пусто.\nСоздайте файл: 'сохрани в файл {filename}: ваш текст'",
                            confidence=0.0,
                            source=self.name
                        )

                # Читаем файл
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                return CoreResponse(
                    success=True,
                    data={'filename': filename, 'content': content, 'size': len(content)},
                    raw_result=f"📄 **Файл '{filename}':**\n\n{content}\n\n📊 Размер: {len(content)} символов",
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

            # 2. СОХРАНЕНИЕ ФАЙЛА
            elif 'сохрани' in q or 'запиши' in q:
                # Ищем шаблон: "сохрани в файл [имя]: [текст]"
                match = re.search(r'сохрани\s+(?:в\s+)?файл\s+(\S+?)\s*[:]\s*(.+)', query, re.IGNORECASE | re.DOTALL)
                if not match:
                    return CoreResponse(
                        success=False,
                        data={'error': 'Неверный формат'},
                        raw_result="❌ Неверный формат.\nИспользуйте: 'сохрани в файл имя.txt: ваш текст'",
                        confidence=0.0,
                        source=self.name
                    )

                filename = match.group(1).strip('"\'')
                content = match.group(2).strip()

                filepath = self._get_file_path(filename)

                # Сохраняем файл
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                return CoreResponse(
                    success=True,
                    data={'filename': filename, 'size': len(content)},
                    raw_result=f"✅ **Файл сохранен:**\n\n📄 Имя: {filename}\n📊 Размер: {len(content)} символов\n📁 Папка: {self.storage_dir}/\n\nСодержимое:\n{content[:200]}{'...' if len(content) > 200 else ''}",
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

            # 3. СПИСОК ФАЙЛОВ
            elif 'список файлов' in q or 'мои файлы' in q or 'все файлы' in q:
                files = os.listdir(self.storage_dir)

                if not files:
                    return CoreResponse(
                        success=True,
                        data={'files': []},
                        raw_result="📁 **Файловое хранилище:**\n\nПапка пуста.\n\nСоздайте файл командой:\n'сохрани в файл заметка.txt: ваш текст'",
                        confidence=1.0,
                        source=self.name,
                        direct_answer=True
                    )

                file_info = []
                for fname in files:
                    fpath = os.path.join(self.storage_dir, fname)
                    if os.path.isfile(fpath):
                        size = os.path.getsize(fpath)
                        file_info.append(f"• **{fname}** ({size} байт)")

                return CoreResponse(
                    success=True,
                    data={'files': files, 'count': len(files)},
                    raw_result=f"📁 **Файловое хранилище:**\n\nФайлов: {len(files)}\n\n" + "\n".join(file_info),
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

            # 4. ПОМОЩЬ
            else:
                return CoreResponse(
                    success=True,
                    data={'action': 'help'},
                    raw_result="📁 **Файловое хранилище - команды:**\n\n"
                               "• Прочитать файл:\n  'прочитай файл имя.txt'\n\n"
                               "• Сохранить файл:\n  'сохрани в файл имя.txt: ваш текст'\n\n"
                               "• Список файлов:\n  'список файлов'\n\n"
                               "• Пример:\n  'сохрани в файл привет.txt: Привет, мир!'",
                    confidence=1.0,
                    source=self.name,
                    direct_answer=True
                )

        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=f"❌ **Ошибка работы с файлами:**\n\n{str(e)}",
                confidence=0.0,
                source=self.name
            )