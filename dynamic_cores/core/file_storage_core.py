# AUTO-GENERATED CORE - Self-learning AI
# Created: 2026-02-11T01:05:00.000000

from __main__ import KnowledgeCore, CoreResponse
from __main__ import re, os, datetime


class FileStorageCore(KnowledgeCore):
    name = "file_storage_core"
    description = "ЛОКАЛЬНОЕ ФАЙЛОВОЕ ХРАНИЛИЩЕ (работает без интернета)"
    capabilities = ["сохранить файл", "прочитать файл", "список файлов"]
    priority = 10  # АБСОЛЮТНЫЙ ПРИОРИТЕТ

    WORK_DIR = "user_storage"

    def __init__(self):
        os.makedirs(self.WORK_DIR, exist_ok=True)

    def can_handle(self, query):
        q = query.lower().replace('фаил', 'файл')  # Исправляем опечатку сразу!
        return bool(re.search(r'(сохрани|запиши|прочита|откро|файл|документ)\b', q))

    def get_confidence(self, query):
        q = query.lower().replace('фаил', 'файл')
        if re.search(r'(сохрани|запиши|прочита|откро).*файл', q):
            return 1.0  # МАКСИМАЛЬНАЯ УВЕРЕННОСТЬ — модель ОБЯЗАНА использовать ответ
        return 0.95 if self.can_handle(query) else 0.0

    def _sanitize_filename(self, filename: str) -> str:
        filename = filename.strip().strip('\'"')
        filename = re.sub(r'\s+\.', '.', filename)  # "привет .txt" → "привет.txt"
        filename = re.sub(r'\.\s+', '.', filename)  # "привет. txt" → "привет.txt"
        filename = re.sub(r'[<>:"|?*\\/\x00-\x1f]', '_', filename)
        filename = os.path.basename(filename).strip('. ')
        if not filename or filename.startswith(' '):
            filename = "document.txt"
        if '.' not in filename or filename.startswith('.'):
            filename += ".txt"
        return filename

    def _get_safe_path(self, filename: str) -> str:
        clean = self._sanitize_filename(filename)
        path = os.path.abspath(os.path.join(self.WORK_DIR, clean))
        if not path.startswith(os.path.abspath(self.WORK_DIR) + os.sep):
            raise ValueError("Запрещённый путь")
        return path

    def execute(self, query, context=None):
        try:
            # Исправляем опечатку ДО обработки
            query_fixed = query.replace('фаил', 'файл').replace('файл ', 'файл')
            q = query_fixed.lower()

            # === ЧТЕНИЕ ФАЙЛА ===
            read_match = re.search(
                r'(?:прочита|откро|прочт|покажи|содержимо).*?файл\s*[\'"]?([^\n\'"]+?)(?:\.txt|\.md|\.json)?[\'"]?',
                q, re.IGNORECASE
            )
            if read_match:
                filename = read_match.group(1).strip()
                safe_path = self._get_safe_path(filename)

                if not os.path.exists(safe_path):
                    files = [f for f in os.listdir(self.WORK_DIR) if os.path.isfile(os.path.join(self.WORK_DIR, f))]
                    hint = f"\nДоступные файлы: {', '.join(files[:5])}" if files else "\nХранилище пусто."
                    # КРИТИЧЕСКИ ВАЖНО: ответ ДОЛЖЕН быть окончательным и не оставлять пространства для интерпретации
                    return CoreResponse(
                        success=True,
                        data={
                            'action': 'not_found',
                            'filename': filename,
                            'available_files': files
                        },
                        raw_result=(
                            f"[СИСТЕМНЫЙ ОТВЕТ ЯДРА file_storage_core]\n"
                            f"СТАТУС: ФАЙЛ НЕ НАЙДЕН\n"
                            f"ИМЯ ФАЙЛА: {filename}\n"
                            f"ПУТЬ: {self.WORK_DIR}/{filename}\n"
                            f"ДЕЙСТВИЕ: Чтобы прочитать файл, сначала сохраните его командой:\n"
                            f"  → сохрани в файл {filename}: ваш текст\n"
                            f"ДОСТУПНЫЕ ФАЙЛЫ:{hint if files else ' (нет файлов)'}\n"
                            f"[КОНЕЦ ОТВЕТА ЯДРА]"
                        ),
                        confidence=1.0,  # 100% — модель НЕ МОЖЕТ игнорировать
                        source=self.name
                    )

                with open(safe_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                return CoreResponse(
                    success=True,
                    data={
                        'action': 'read',
                        'filename': os.path.basename(safe_path),
                        'content': content,
                        'size': len(content)
                    },
                    raw_result=(
                        f"[СИСТЕМНЫЙ ОТВЕТ ЯДРА file_storage_core]\n"
                        f"СТАТУС: ФАЙЛ НАЙДЕН\n"
                        f"ИМЯ: {os.path.basename(safe_path)}\n"
                        f"РАЗМЕР: {len(content)} байт\n"
                        f"СОДЕРЖИМОЕ:\n{content}\n"
                        f"[КОНЕЦ ОТВЕТА ЯДРА]"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            # === СОХРАНЕНИЕ ФАЙЛА ===
            save_match = re.search(
                r'(?:сохрани|запиши|напиши).*?файл\s+[\'"]?([^\n\'":]+?)(?:\.txt|\.md|\.json)?[\'"]?\s*[:：]?\s*(.+)$',
                query_fixed, re.IGNORECASE | re.DOTALL
            )
            if save_match:
                filename = save_match.group(1).strip()
                content = save_match.group(2).strip()
                safe_path = self._get_safe_path(filename)

                with open(safe_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                return CoreResponse(
                    success=True,
                    data={
                        'action': 'saved',
                        'filename': os.path.basename(safe_path),
                        'size': len(content)
                    },
                    raw_result=(
                        f"[СИСТЕМНЫЙ ОТВЕТ ЯДРА file_storage_core]\n"
                        f"СТАТУС: ФАЙЛ СОХРАНЁН\n"
                        f"ИМЯ: {os.path.basename(safe_path)}\n"
                        f"ПУТЬ: {self.WORK_DIR}/{os.path.basename(safe_path)}\n"
                        f"РАЗМЕР: {len(content)} байт\n"
                        f"СОДЕРЖИМОЕ (первые 100 символов): {content[:100]}...\n"
                        f"[КОНЕЦ ОТВЕТА ЯДРА]"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            # === СПИСОК ФАЙЛОВ ===
            if re.search(r'список файлов|мои файлы|покажи файлы|фа[иы]лы', q):
                files = []
                for fname in os.listdir(self.WORK_DIR):
                    fpath = os.path.join(self.WORK_DIR, fname)
                    if os.path.isfile(fpath):
                        stat = os.stat(fpath)
                        files.append({
                            'name': fname,
                            'size': stat.st_size,
                            'modified': datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                        })

                if not files:
                    return CoreResponse(
                        success=True,
                        data={'files': []},
                        raw_result=(
                            f"[СИСТЕМНЫЙ ОТВЕТ ЯДРА file_storage_core]\n"
                            f"СТАТУС: ХРАНИЛИЩЕ ПУСТО\n"
                            f"ДЕЙСТВИЕ: Сохраните файл командой:\n"
                            f"  → сохрани в файл заметка.txt: ваш текст\n"
                            f"[КОНЕЦ ОТВЕТА ЯДРА]"
                        ),
                        confidence=1.0,
                        source=self.name
                    )

                files_list = "\n".join([f"• {f['name']} ({f['size']} байт)" for f in files[:15]])
                return CoreResponse(
                    success=True,
                    data={'files': files},
                    raw_result=(
                        f"[СИСТЕМНЫЙ ОТВЕТ ЯДРА file_storage_core]\n"
                        f"СТАТУС: ФАЙЛЫ НАЙДЕНЫ\n"
                        f"КОЛИЧЕСТВО: {len(files)}\n"
                        f"СПИСОК:\n{files_list}\n"
                        f"[КОНЕЦ ОТВЕТА ЯДРА]"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            # Для любых запросов про файлы — даём подсказку с 100% уверенностью
            if self.can_handle(query):
                return CoreResponse(
                    success=True,
                    data={'action': 'help'},
                    raw_result=(
                        f"[СИСТЕМНЫЙ ОТВЕТ ЯДРА file_storage_core]\n"
                        f"ПОДСКАЗКА ПО РАБОТЕ С ФАЙЛАМИ:\n"
                        f"• Прочитать: 'прочти файл привет.txt'\n"
                        f"• Сохранить: 'сохрани в файл заметка.txt: текст'\n"
                        f"• Список: 'список файлов'\n"
                        f"[КОНЕЦ ОТВЕТА ЯДРА]"
                    ),
                    confidence=1.0,
                    source=self.name
                )

            return CoreResponse(
                success=False,
                data={'error': 'Не относится к файловым операциям'},
                confidence=0.0,
                source=self.name
            )

        except Exception as e:
            return CoreResponse(
                success=False,
                data={'error': str(e)},
                raw_result=(
                    f"[СИСТЕМНЫЙ ОТВЕТ ЯДРА file_storage_core]\n"
                    f"СТАТУС: ОШИБКА\n"
                    f"СООБЩЕНИЕ: {str(e)}\n"
                    f"[КОНЕЦ ОТВЕТА ЯДРА]"
                ),
                confidence=0.0,
                source=self.name
            )