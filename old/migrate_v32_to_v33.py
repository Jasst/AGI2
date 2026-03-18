#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 MIGRATION SCRIPT: v32.0 → v33.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Перенос "сознания" (памяти, состояния) из Enhanced AGI Brain v32 в v33

ЧТО ПЕРЕНОСИТСЯ:
✅ Долговременная память (long_term_memory)
✅ Эпизодическая память (episodic_memory)
✅ История сообщений (message_history)
✅ Создание весов концептов на основе истории

ЧТО НЕ ПЕРЕНОСИТСЯ (удаляется):
❌ Фейковая нейросеть (neurons, synapses)
❌ Произвольные метрики активации
❌ Всё, что было симуляцией

ПРИНЦИП:
Честность > Совместимость. Переносим только то, что имеет реальное значение.
"""

import os
import sys
import gzip
import pickle
import hashlib
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field


# Цвета для вывода
class Colors:
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    BLUE = '\033[36m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")


def log_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")


def log_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")


def log_section(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


@dataclass
class MigrationStats:
    """Статистика миграции"""
    users_found: int = 0
    users_migrated: int = 0
    users_failed: int = 0

    ltm_items: int = 0
    episodes: int = 0
    concepts_created: int = 0

    old_neurons: int = 0  # Для информации
    old_synapses: int = 0  # Для информации


class MemoryMigrator:
    """Класс для миграции памяти v32 → v33"""

    def __init__(self, base_dir_v32: Path, base_dir_v33: Path):
        self.base_dir_v32 = base_dir_v32
        self.base_dir_v33 = base_dir_v33
        self.stats = MigrationStats()

    def find_users(self) -> List[str]:
        """Поиск всех пользователей в v32"""
        memory_dir = self.base_dir_v32 / 'memory'
        if not memory_dir.exists():
            return []

        users = []
        for user_dir in memory_dir.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('user_'):
                user_id = user_dir.name.replace('user_', '')
                users.append(user_id)

        return users

    def load_v32_memory(self, user_id: str) -> Optional[Dict]:
        """Загрузка памяти v32"""
        memory_file = self.base_dir_v32 / 'memory' / f'user_{user_id}' / 'memory_v32.pkl.gz'

        if not memory_file.exists():
            log_warning(f"Файл памяти не найден для user_{user_id}")
            return None

        try:
            with gzip.open(memory_file, 'rb') as f:
                mem_state = pickle.load(f)
            log_success(f"Загружена память v32 для user_{user_id}")
            return mem_state
        except Exception as e:
            log_error(f"Ошибка загрузки памяти для user_{user_id}: {e}")
            return None

    def load_v32_neural(self, user_id: str) -> Optional[Dict]:
        """Загрузка нейросети v32 (для статистики)"""
        neural_file = self.base_dir_v32 / 'neural_nets' / f'{user_id}_v32.pkl.gz'

        if not neural_file.exists():
            return None

        try:
            with gzip.open(neural_file, 'rb') as f:
                neural_state = pickle.load(f)
            return neural_state
        except Exception as e:
            log_warning(f"Не удалось загрузить нейросеть (не критично): {e}")
            return None

    def extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов"""
        stop_words = {
            'это', 'для', 'как', 'что', 'когда', 'где', 'почему',
            'мне', 'тебе', 'его', 'её', 'наш', 'ваш', 'их',
            'быть', 'иметь', 'мочь', 'хотеть', 'знать', 'сказать',
            'user', 'assistant', 'самый', 'очень', 'также', 'более'
        }

        words = re.findall(r'\b[а-яё]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]

        return keywords

    def create_concept_weights(self, mem_state: Dict) -> Dict[str, float]:
        """Создание весов концептов на основе истории памяти"""
        all_text = []

        # Собираем текст из долговременной памяти
        for item_data in mem_state.get('long_term', {}).values():
            content = item_data.get('content', '')
            if content:
                all_text.append(content)

        # Собираем текст из эпизодов
        for ep_data in mem_state.get('episodic', {}).values():
            context = ep_data.get('context', '')
            if context:
                all_text.append(context)

            for msg in ep_data.get('messages', []):
                msg_content = msg.get('content', '')
                if msg_content:
                    all_text.append(msg_content)

        # Извлекаем ключевые слова
        all_keywords = []
        for text in all_text:
            keywords = self.extract_keywords(text)
            all_keywords.extend(keywords)

        # Подсчитываем частоту
        keyword_counts = Counter(all_keywords)

        # Нормализуем в веса [0.5, 1.0]
        max_count = max(keyword_counts.values()) if keyword_counts else 1

        concept_weights = {}
        for keyword, count in keyword_counts.most_common(100):  # Топ-100
            # Вес от 0.5 (редкое) до 1.0 (частое)
            weight = 0.5 + (count / max_count) * 0.5
            concept_weights[keyword] = weight

        return concept_weights

    def save_v33_memory(self, user_id: str, mem_state: Dict) -> bool:
        """Сохранение памяти в формате v33"""
        user_dir = self.base_dir_v33 / 'memory' / f'user_{user_id}'
        user_dir.mkdir(parents=True, exist_ok=True)

        memory_file = user_dir / 'memory_v33.pkl.gz'

        try:
            with gzip.open(memory_file, 'wb', compresslevel=6) as f:
                pickle.dump(mem_state, f)
            log_success(f"Сохранена память v33 для user_{user_id}")
            return True
        except Exception as e:
            log_error(f"Ошибка сохранения памяти для user_{user_id}: {e}")
            return False

    def save_v33_weights(self, user_id: str, concept_weights: Dict[str, float]) -> bool:
        """Сохранение весов концептов в формате v33"""
        user_dir = self.base_dir_v33 / 'memory' / f'user_{user_id}'
        user_dir.mkdir(parents=True, exist_ok=True)

        weights_file = user_dir / 'weights_v33.pkl.gz'

        weights_state = {
            'weights': concept_weights,
            'access_count': {k: 1 for k in concept_weights.keys()},  # Начальный счетчик
        }

        try:
            with gzip.open(weights_file, 'wb', compresslevel=6) as f:
                pickle.dump(weights_state, f)
            log_success(f"Сохранены веса концептов для user_{user_id}")
            return True
        except Exception as e:
            log_error(f"Ошибка сохранения весов для user_{user_id}: {e}")
            return False

    def migrate_user(self, user_id: str) -> bool:
        """Миграция одного пользователя"""
        log_section(f"МИГРАЦИЯ ПОЛЬЗОВАТЕЛЯ: {user_id}")

        # 1. Загрузка v32 памяти
        mem_state_v32 = self.load_v32_memory(user_id)
        if not mem_state_v32:
            self.stats.users_failed += 1
            return False

        # 2. Загрузка v32 нейросети (для статистики)
        neural_state_v32 = self.load_v32_neural(user_id)
        if neural_state_v32:
            self.stats.old_neurons = len(neural_state_v32.get('neurons', []))
            self.stats.old_synapses = len(neural_state_v32.get('synapses', []))
            log_info(f"v32 нейросеть: {self.stats.old_neurons} нейронов, "
                     f"{self.stats.old_synapses} синапсов (будет удалена)")

        # 3. Подсчет данных
        ltm_count = len(mem_state_v32.get('long_term', {}))
        ep_count = len(mem_state_v32.get('episodic', {}))

        self.stats.ltm_items += ltm_count
        self.stats.episodes += ep_count

        log_info(f"Найдено: {ltm_count} LTM, {ep_count} эпизодов")

        # 4. Создание весов концептов
        log_info("Создание весов концептов на основе истории...")
        concept_weights = self.create_concept_weights(mem_state_v32)
        self.stats.concepts_created += len(concept_weights)
        log_success(f"Создано {len(concept_weights)} весов концептов")

        # 5. Память остается как есть (совместимый формат)
        # Просто копируем её в v33
        if not self.save_v33_memory(user_id, mem_state_v32):
            self.stats.users_failed += 1
            return False

        # 6. Сохранение весов
        if not self.save_v33_weights(user_id, concept_weights):
            self.stats.users_failed += 1
            return False

        self.stats.users_migrated += 1
        log_success(f"✅ Пользователь {user_id} успешно мигрирован!")

        return True

    def migrate_all(self) -> MigrationStats:
        """Миграция всех пользователей"""
        log_section("ПОИСК ПОЛЬЗОВАТЕЛЕЙ v32")

        users = self.find_users()
        self.stats.users_found = len(users)

        if not users:
            log_warning("Пользователи v32 не найдены!")
            return self.stats

        log_info(f"Найдено пользователей: {len(users)}")
        print(f"Список: {', '.join(users)}\n")

        # Миграция каждого пользователя
        for user_id in users:
            self.migrate_user(user_id)

        return self.stats


def print_migration_summary(stats: MigrationStats):
    """Печать итоговой статистики"""
    log_section("ИТОГИ МИГРАЦИИ")

    print(f"{Colors.BOLD}Пользователи:{Colors.RESET}")
    print(f"  Найдено:      {stats.users_found}")
    print(f"  Мигрировано:  {Colors.GREEN}{stats.users_migrated}{Colors.RESET}")
    print(f"  Ошибок:       {Colors.RED if stats.users_failed > 0 else Colors.GREEN}{stats.users_failed}{Colors.RESET}")

    print(f"\n{Colors.BOLD}Данные:{Colors.RESET}")
    print(f"  LTM элементов:     {stats.ltm_items}")
    print(f"  Эпизодов:          {stats.episodes}")
    print(f"  Концептов создано: {stats.concepts_created}")

    if stats.old_neurons > 0:
        print(f"\n{Colors.BOLD}Удалено (фейковые данные):{Colors.RESET}")
        print(f"  Нейронов:  {Colors.YELLOW}{stats.old_neurons}{Colors.RESET}")
        print(f"  Синапсов:  {Colors.YELLOW}{stats.old_synapses}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 60}{Colors.RESET}")
    if stats.users_migrated == stats.users_found and stats.users_found > 0:
        print(f"{Colors.BOLD}{Colors.GREEN}✅ ВСЕ ПОЛЬЗОВАТЕЛИ УСПЕШНО МИГРИРОВАНЫ!{Colors.RESET}")
    elif stats.users_migrated > 0:
        print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  МИГРАЦИЯ ЗАВЕРШЕНА С ОШИБКАМИ{Colors.RESET}")
    else:
        print(f"{Colors.BOLD}{Colors.RED}❌ МИГРАЦИЯ НЕ УДАЛАСЬ{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 60}{Colors.RESET}\n")


def main():
    """Главная функция миграции"""
    print(f"""
{Colors.BOLD}{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗
║  🔄 MIGRATION: Enhanced AGI Brain v32.0 → v33.0              ║
║     Перенос "сознания" в честную архитектуру                 ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.BOLD}ЧТО БУДЕТ ПЕРЕНЕСЕНО:{Colors.RESET}
  ✅ Долговременная память (long_term_memory)
  ✅ Эпизодическая память (episodic_memory)
  ✅ История сообщений (message_history)
  ✅ Создание весов концептов на основе истории

{Colors.BOLD}ЧТО БУДЕТ УДАЛЕНО:{Colors.RESET}
  ❌ Фейковая нейросеть (neurons, synapses)
  ❌ Произвольные метрики активации
  ❌ Всё, что было симуляцией

{Colors.BOLD}ПРИНЦИП: Честность > Совместимость{Colors.RESET}
Переносим только то, что имеет реальное значение.
""")

    # Определение директорий
    base_dir_v32 = Path(os.getenv('BASE_DIR_V32', 'temporal_brain_v32'))
    base_dir_v33 = Path(os.getenv('BASE_DIR_V33', 'temporal_brain_v33'))

    print(f"📂 Директория v32: {base_dir_v32}")
    print(f"📂 Директория v33: {base_dir_v33}\n")

    # Проверка существования v32
    if not base_dir_v32.exists():
        log_error(f"Директория v32 не найдена: {base_dir_v32}")
        log_info("Создайте переменную окружения BASE_DIR_V32 или укажите путь")
        return 1

    # Создание директорий v33
    for subdir in ['memory', 'knowledge', 'cache', 'logs', 'backups',
                   'episodic', 'analytics', 'goals']:
        (base_dir_v33 / subdir).mkdir(parents=True, exist_ok=True)

    # Подтверждение
    print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  ВНИМАНИЕ!{Colors.RESET}")
    print(f"Миграция создаст новые файлы в {base_dir_v33}")
    print(f"Оригинальные файлы v32 останутся нетронутыми.\n")

    response = input(f"{Colors.BOLD}Продолжить? [y/N]: {Colors.RESET}").strip().lower()
    if response not in ['y', 'yes', 'д', 'да']:
        log_info("Миграция отменена пользователем")
        return 0

    # Создание мигратора и запуск
    migrator = MemoryMigrator(base_dir_v32, base_dir_v33)
    stats = migrator.migrate_all()

    # Итоги
    print_migration_summary(stats)

    # Следующие шаги
    if stats.users_migrated > 0:
        print(f"\n{Colors.BOLD}📋 СЛЕДУЮЩИЕ ШАГИ:{Colors.RESET}")
        print(f"1. Проверьте мигрированные данные в {base_dir_v33}")
        print(f"2. Запустите v33.0 с переменной окружения:")
        print(f"   {Colors.BOLD}export BASE_DIR='{base_dir_v33}'{Colors.RESET}")
        print(f"3. При первом запуске v33 автоматически загрузит память")
        print(f"4. После проверки работы можете удалить v32 (опционально)")
        print(f"\n{Colors.BOLD}🔐 БЕЗОПАСНОСТЬ:{Colors.RESET}")
        print(f"Оригинальные файлы v32 сохранены в {base_dir_v32}")
        print(f"Вы можете вернуться к v32 в любой момент.\n")

    return 0 if stats.users_failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Миграция прервана пользователем{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)