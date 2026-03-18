#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 MIGRATION SCRIPT: Enhanced AGI Brain v32.0 → v33.2 UNIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Перенос "сознания" в унифицированную архитектуру с реальной нейросетью

✅ ЧТО ПЕРЕНОСИТСЯ:
• Долговременная память (long_term_memory) — с quality_score
• Эпизодическая память (episodic_memory) — с контекстом и эмоциями
• История сообщений — для восстановления контекста
• Веса концептов → для SimpleWeightedModule (совместимость)
• Словарь слов → для AdaptiveEmbedding (реальное обучение)

❌ ЧТО НЕ ПЕРЕНОСИТСЯ (удаляется честно):
• Фейковые нейроны и синапсы (симуляция без обучения)
• Произвольные метрики активации
• Всё, что не имеет проверяемой ценности

🎯 ПРИНЦИП v33.2:
Честность > Совместимость > Сложность
Переносим только то, что реально помогает системе учиться.
"""

import os
import sys
import gzip
import pickle
import hashlib
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import time


# ═══════════════════════════════════════════════════════════════
# 🎨 Утилиты вывода
# ═══════════════════════════════════════════════════════════════

class Colors:
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    BLUE = '\033[36m'
    CYAN = '\033[96m'
    MAGENTA = '\033[35m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg: str): print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")


def log_success(msg: str): print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")


def log_warning(msg: str): print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")


def log_error(msg: str): print(f"{Colors.RED}❌ {msg}{Colors.RESET}")


def log_neural(msg: str): print(f"{Colors.CYAN}🧬 {msg}{Colors.RESET}")


def log_section(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * 70}{Colors.RESET}\n")


# ═══════════════════════════════════════════════════════════════
# 📦 Классы данных v33.2 (для совместимости при сохранении)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MemoryItem:
    content: str
    timestamp: float
    importance: float
    priority: float = 0.5
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    quality_score: float = 0.5
    metadata: Dict = field(default_factory=dict)


@dataclass
class Episode:
    id: str
    messages: List[Dict[str, str]]
    context: str
    timestamp: float
    importance: float = 0.5
    quality_score: float = 0.5
    consolidation_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# 🧠 Адаптивный эмбеддинг (упрощённый для миграции)
# ═══════════════════════════════════════════════════════════════

class MigrationEmbedding:
    """
    Упрощённый эмбеддинг для инициализации адаптивной нейросети
    при миграции — создаёт начальный словарь из исторических данных
    """

    def __init__(self, embedding_dim: int = 64, vocab_size: int = 5000):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.embeddings: np.ndarray = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.next_idx = 0

    def add_word(self, word: str, frequency: float = 1.0) -> int:
        """Добавление слова с учётом частоты (влияет на начальный вес)"""
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        if self.next_idx >= self.vocab_size:
            return -1  # Словарь полон
        idx = self.next_idx
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        # Инициализация эмбеддинга с небольшим смещением по частоте
        self.embeddings[idx] = np.random.randn(self.embedding_dim) * 0.01 * (0.5 + frequency * 0.5)
        self.next_idx += 1
        return idx

    def get_statistics(self) -> Dict:
        return {
            'vocab_size': self.next_idx,
            'embedding_dim': self.embedding_dim,
        }


# ═══════════════════════════════════════════════════════════════
# 📊 Статистика миграции
# ═══════════════════════════════════════════════════════════════

@dataclass
class MigrationStats:
    users_found: int = 0
    users_migrated: int = 0
    users_failed: int = 0
    ltm_items: int = 0
    episodes: int = 0
    concepts_created: int = 0
    vocab_words_added: int = 0
    old_neurons: int = 0
    old_synapses: int = 0
    migration_time: float = 0.0


# ═══════════════════════════════════════════════════════════════
# 🔄 Основной класс миграции
# ═══════════════════════════════════════════════════════════════

class UnifiedMemoryMigrator:
    """Миграция v32.0 → v33.2 UNIFIED с сохранением реального опыта"""

    def __init__(self, base_dir_v32: Path, base_dir_v33: Path, embedding_dim: int = 64):
        self.base_dir_v32 = base_dir_v32
        self.base_dir_v33 = base_dir_v33
        self.embedding_dim = embedding_dim
        self.stats = MigrationStats()
        self.stop_words = {
            'это', 'для', 'как', 'что', 'когда', 'где', 'почему', 'мне', 'тебе',
            'его', 'её', 'наш', 'ваш', 'их', 'быть', 'иметь', 'мочь', 'хотеть',
            'знать', 'сказать', 'дело', 'раз', 'вот', 'так', 'же', 'ли', 'бы',
            'пользователь', 'ассистент', 'ответ', 'вопрос', 'привет', 'пока'
        }

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
        return sorted(users)

    def load_v32_memory(self, user_id: str) -> Optional[Dict]:
        """Загрузка памяти v32"""
        memory_file = self.base_dir_v32 / 'memory' / f'user_{user_id}' / 'memory_v32.pkl.gz'
        if not memory_file.exists():
            log_warning(f"Файл памяти не найден: user_{user_id}")
            return None
        try:
            with gzip.open(memory_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            log_error(f"Ошибка загрузки памяти: {e}")
            return None

    def load_v32_neural(self, user_id: str) -> Optional[Dict]:
        """Загрузка нейросети v32 (только для статистики)"""
        neural_file = self.base_dir_v32 / 'neural_nets' / f'{user_id}_v32.pkl.gz'
        if not neural_file.exists():
            return None
        try:
            with gzip.open(neural_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def extract_keywords(self, text: str, min_freq: int = 1) -> Counter:
        """Извлечение ключевых слов с частотой"""
        words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())
        keywords = [w for w in words if w not in self.stop_words]
        return Counter(keywords)

    def collect_all_text(self, mem_state: Dict) -> List[str]:
        """Сбор всего текстового контента из памяти"""
        texts = []
        # Долговременная память
        for item in mem_state.get('long_term', {}).values():
            content = item.get('content', '')
            if content and len(content) > 10:
                texts.append(content)
        # Эпизоды
        for ep in mem_state.get('episodic', {}).values():
            context = ep.get('context', '')
            if context: texts.append(context)
            for msg in ep.get('messages', []):
                content = msg.get('content', '')
                if content: texts.append(content)
        return texts

    def create_concept_weights(self, keyword_counts: Counter, top_n: int = 100) -> Dict[str, float]:
        """Создание весов концептов [0.5, 1.0] на основе частоты"""
        if not keyword_counts:
            return {}
        max_count = max(keyword_counts.values())
        weights = {}
        for keyword, count in keyword_counts.most_common(top_n):
            weight = 0.5 + (count / max_count) * 0.5
            weights[keyword] = round(weight, 3)
        return weights

    def build_adaptive_vocab(self, keyword_counts: Counter,
                             min_freq: int = 2, max_vocab: int = 3000) -> MigrationEmbedding:
        """Построение начального словаря для AdaptiveEmbedding"""
        embedding = MigrationEmbedding(embedding_dim=self.embedding_dim)
        # Сортируем по частоте, добавляем только значимые слова
        for word, count in keyword_counts.most_common(max_vocab):
            if count >= min_freq and len(word) >= 4:
                freq_norm = min(1.0, count / 10)  # Нормализация частоты
                embedding.add_word(word, frequency=freq_norm)
        return embedding

    def migrate_memory_structure(self, mem_state: Dict) -> Dict:
        """Конвертация структуры памяти в формат v33.2"""
        migrated = {'long_term': {}, 'episodic': {}}

        # Долговременная память
        for mid, item_data in mem_state.get('long_term', {}).items():
            item = MemoryItem(
                content=item_data.get('content', ''),
                timestamp=item_data.get('timestamp', time.time()),
                importance=item_data.get('importance', 0.5),
                priority=item_data.get('priority', 0.5),
                access_count=item_data.get('access_count', 0),
                last_access=item_data.get('last_access', time.time()),
                quality_score=item_data.get('quality_score', 0.5),
                metadata=item_data.get('metadata', {})
            )
            migrated['long_term'][mid] = item

        # Эпизодическая память
        for eid, ep_data in mem_state.get('episodic', {}).items():
            episode = Episode(
                id=ep_data.get('id', eid),
                messages=ep_data.get('messages', []),
                context=ep_data.get('context', ''),
                timestamp=ep_data.get('timestamp', time.time()),
                importance=ep_data.get('importance', 0.5),
                quality_score=ep_data.get('quality_score', 0.5),
                consolidation_count=ep_data.get('consolidation_count', 0)
            )
            migrated['episodic'][eid] = episode

        return migrated

    def save_v33_memory(self, user_id: str, mem_state: Dict) -> bool:
        """Сохранение памяти в формате v33.2"""
        user_dir = self.base_dir_v33 / 'memory' / f'user_{user_id}'
        user_dir.mkdir(parents=True, exist_ok=True)
        memory_file = user_dir / 'memory_v33_2.pkl.gz'
        try:
            # Конвертируем dataclass в dict для pickle
            save_state = {
                'long_term': {
                    mid: {k: v for k, v in asdict(item).items() if k != 'embedding'}
                    for mid, item in mem_state['long_term'].items()
                },
                'episodic': {eid: ep.to_dict() for eid, ep in mem_state['episodic'].items()}
            }
            with gzip.open(memory_file, 'wb', compresslevel=6) as f:
                pickle.dump(save_state, f)
            return True
        except Exception as e:
            log_error(f"Ошибка сохранения памяти: {e}")
            return False

    def save_concept_weights(self, user_id: str, weights: Dict[str, float]) -> bool:
        """Сохранение весов концептов для SimpleWeightedModule"""
        user_dir = self.base_dir_v33 / 'memory' / f'user_{user_id}'
        user_dir.mkdir(parents=True, exist_ok=True)
        weights_file = user_dir / 'weights_v33_2.pkl.gz'
        try:
            state = {
                'weights': weights,
                'access_count': {k: max(1, int(v * 10)) for k, v in weights.items()},
                'last_access': {k: time.time() for k in weights.keys()}
            }
            with gzip.open(weights_file, 'wb', compresslevel=6) as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            log_error(f"Ошибка сохранения весов: {e}")
            return False

    def save_adaptive_vocab(self, user_id: str, embedding: MigrationEmbedding) -> bool:
        """Сохранение начального состояния для AdaptiveNeuralEngine"""
        neural_dir = self.base_dir_v33 / 'neural_nets'
        neural_dir.mkdir(parents=True, exist_ok=True)
        vocab_file = neural_dir / f'{user_id}_adaptive_vocab.pkl.gz'
        try:
            state = {
                'word_to_idx': embedding.word_to_idx,
                'idx_to_word': embedding.idx_to_word,
                'embeddings': embedding.embeddings[:embedding.next_idx],  # Только использованные
                'next_idx': embedding.next_idx,
                'embedding_dim': embedding.embedding_dim,
                'migrated_at': time.time()
            }
            with gzip.open(vocab_file, 'wb', compresslevel=6) as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            log_error(f"Ошибка сохранения словаря: {e}")
            return False

    def migrate_user(self, user_id: str) -> bool:
        """Полная миграция одного пользователя"""
        log_section(f"🔄 МИГРАЦИЯ: user_{user_id}")
        start_time = time.time()

        # 1. Загрузка v32 данных
        log_info("📥 Загрузка данных v32...")
        mem_state = self.load_v32_memory(user_id)
        if not mem_state:
            self.stats.users_failed += 1
            return False

        neural_state = self.load_v32_neural(user_id)
        if neural_state:
            self.stats.old_neurons = len(neural_state.get('neurons', []))
            self.stats.old_synapses = len(neural_state.get('synapses', []))
            log_neural(f"v32 нейросеть: {self.stats.old_neurons} нейронов, "
                       f"{self.stats.old_synapses} синапсов → 🗑️ удаляется (симуляция)")

        # 2. Анализ контента
        log_info("🔍 Анализ исторических данных...")
        all_texts = self.collect_all_text(mem_state)
        total_chars = sum(len(t) for t in all_texts)
        log_info(f"   • Текстовых блоков: {len(all_texts)}")
        log_info(f"   • Общий объём: {total_chars:,} символов")

        # 3. Извлечение ключевых слов
        log_info("🔑 Извлечение концептов и интересов...")
        all_keywords = Counter()
        for text in all_texts:
            all_keywords.update(self.extract_keywords(text))

        # Фильтрация: только слова с частотой >= 2
        filtered_keywords = Counter({k: v for k, v in all_keywords.items() if v >= 2})
        log_success(f"   • Найдено уникальных слов: {len(all_keywords)}")
        log_success(f"   • Значимых концептов (частота≥2): {len(filtered_keywords)}")

        # 4. Создание весов концептов (для SimpleWeightedModule)
        log_info("⚖️  Создание весов концептов...")
        concept_weights = self.create_concept_weights(filtered_keywords)
        self.stats.concepts_created += len(concept_weights)
        if concept_weights:
            top_5 = list(concept_weights.items())[:5]
            log_info(f"   • Топ-концепты: {', '.join([f'{k}={v:.2f}' for k, v in top_5])}")

        # 5. Построение словаря для AdaptiveEmbedding
        log_neural("🧬 Инициализация адаптивного эмбеддинга...")
        adaptive_vocab = self.build_adaptive_vocab(filtered_keywords)
        vocab_stats = adaptive_vocab.get_statistics()
        self.stats.vocab_words_added += vocab_stats['vocab_size']
        log_neural(f"   • Словарь: {vocab_stats['vocab_size']} слов × {vocab_stats['embedding_dim']} dim")
        log_neural(f"   • Готово к реальному обучению с первого взаимодействия!")

        # 6. Конвертация и сохранение памяти
        log_info("💾 Сохранение в формате v33.2...")
        migrated_memory = self.migrate_memory_structure(mem_state)
        self.stats.ltm_items += len(migrated_memory['long_term'])
        self.stats.episodes += len(migrated_memory['episodic'])

        if not self.save_v33_memory(user_id, migrated_memory):
            self.stats.users_failed += 1
            return False
        if not self.save_concept_weights(user_id, concept_weights):
            self.stats.users_failed += 1
            return False
        if not self.save_adaptive_vocab(user_id, adaptive_vocab):
            log_warning("⚠️  Словарь для нейросети не сохранён (не критично)")

        # 7. Итоги по пользователю
        elapsed = time.time() - start_time
        self.stats.users_migrated += 1
        log_success(f"✅ user_{user_id} мигрирован за {elapsed:.2f}с")
        log_info(f"   • LTM: {len(migrated_memory['long_term'])} | Episodes: {len(migrated_memory['episodic'])}")
        log_info(f"   • Концептов: {len(concept_weights)} | Словарь: {vocab_stats['vocab_size']}")

        return True

    def migrate_all(self) -> MigrationStats:
        """Миграция всех пользователей"""
        log_section("🔍 ПОИСК ПОЛЬЗОВАТЕЛЕЙ v32")
        users = self.find_users()
        self.stats.users_found = len(users)

        if not users:
            log_warning("Пользователи v32 не найдены!")
            return self.stats

        log_info(f"Найдено: {len(users)} пользователей")
        for u in users: log_info(f"   • {u}")
        print()

        start_total = time.time()
        for user_id in users:
            try:
                self.migrate_user(user_id)
            except Exception as e:
                log_error(f"Критическая ошибка при миграции {user_id}: {e}")
                self.stats.users_failed += 1

        self.stats.migration_time = time.time() - start_total
        return self.stats


# ═══════════════════════════════════════════════════════════════
# 📋 Вывод результатов
# ═══════════════════════════════════════════════════════════════

def print_summary(stats: MigrationStats):
    log_section("📊 ИТОГИ МИГРАЦИИ v32.0 → v33.2")

    print(f"{Colors.BOLD}👥 Пользователи:{Colors.RESET}")
    print(f"   Найдено:      {stats.users_found}")
    print(f"   ✅ Мигрировано:  {Colors.GREEN}{stats.users_migrated}{Colors.RESET}")
    print(f"   ❌ Ошибок:       {Colors.RED if stats.users_failed else Colors.GREEN}{stats.users_failed}{Colors.RESET}")

    print(f"\n{Colors.BOLD}📚 Данные:{Colors.RESET}")
    print(f"   • LTM элементов:     {stats.ltm_items:,}")
    print(f"   • Эпизодов:          {stats.episodes:,}")
    print(f"   • Концептов создано: {stats.concepts_created:,}")
    print(f"   • Слов для нейросети: {stats.vocab_words_added:,}")

    if stats.old_neurons > 0:
        print(f"\n{Colors.BOLD}🗑️  Удалено (симуляция):{Colors.RESET}")
        print(f"   • Нейронов:  {Colors.YELLOW}{stats.old_neurons:,}{Colors.RESET} (фейковые, без обучения)")
        print(f"   • Синапсов:  {Colors.YELLOW}{stats.old_synapses:,}{Colors.RESET} (произвольные веса)")

    print(f"\n{Colors.BOLD}⏱️  Время:{Colors.RESET}")
    print(f"   • Общее: {stats.migration_time:.2f}с")
    if stats.users_migrated > 0:
        avg = stats.migration_time / stats.users_migrated
        print(f"   • Среднее на пользователя: {avg:.2f}с")

    print(f"\n{Colors.BOLD}{Colors.GREEN}{'═' * 70}{Colors.RESET}")
    if stats.users_migrated == stats.users_found and stats.users_found > 0:
        print(f"{Colors.BOLD}{Colors.GREEN}🎉 ВСЕ ПОЛЬЗОВАТЕЛИ УСПЕШНО МИГРИРОВАНЫ В v33.2!{Colors.RESET}")
        print(f"{Colors.CYAN}✨ Теперь у вас:{Colors.RESET}")
        print(f"   • 🔥 Настоящая нейросеть с backpropagation")
        print(f"   • 🎯 Честные метрики из реальных данных")
        print(f"   • 🧠 Память с приоритетом качества")
        print(f"   • 🔄 Online learning на каждом взаимодействии")
    elif stats.users_migrated > 0:
        print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  МИГРАЦИЯ ЗАВЕРШЕНА С ОШИБКАМИ{Colors.RESET}")
    else:
        print(f"{Colors.BOLD}{Colors.RED}❌ МИГРАЦИЯ НЕ УДАЛАСЬ{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'═' * 70}{Colors.RESET}\n")

def print_next_steps(base_dir_v32: Path, base_dir_v33: Path, stats: MigrationStats):
    if stats.users_migrated == 0:
        return

    print(f"{Colors.BOLD}📋 СЛЕДУЮЩИЕ ШАГИ:{Colors.RESET}\n")
    print(f"1️⃣  Проверьте мигрированные данные:")
    print(f"   {Colors.CYAN}ls -la {base_dir_v33 / 'memory'}{Colors.RESET}\n")

    print(f"2️⃣  Запустите v33.2 с правильной директорией:")
    print(f"   {Colors.BOLD}export BASE_DIR='{base_dir_v33}'{Colors.RESET}")
    print(f"   {Colors.BOLD}python agi_v33_2.py{Colors.RESET}\n")

    print(f"3️⃣  При первом запуске v33.2 автоматически:")
    print(f"   • ✅ Загрузит память (LTM + episodes)")
    print(f"   • ✅ Восстановит веса концептов")
    print(f"   • ✅ Инициализирует адаптивную нейросеть с вашим словарём")
    print(f"   • ✅ Начнёт реальное обучение с первого сообщения!\n")

    print(f"4️⃣  Протестируйте честные метрики:")
    print(f"   • Спросите: «Какая у тебя метрика confidence?»")
    print(f"   • Система покажет: предсказание нейросети + честное вычисление")
    print(f"   • Никаких выдуманных чисел — только проверяемые данные!\n")

    print(f"{Colors.BOLD}🔐 БЕЗОПАСНОСТЬ:{Colors.RESET}")
    print(f"   • Оригинальные файлы v32 сохранены в {base_dir_v32}")
    print(f"   • Вы можете откатиться в любой момент")
    print(f"   • Рекомендуется сделать бэкап перед удалением v32\n")


# ═══════════════════════════════════════════════════════════════
# 🚀 Главная функция
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"""
{Colors.BOLD}{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗
║  🔄 MIGRATION: Enhanced AGI Brain v32.0 → v33.2 UNIFIED      ║
║     Перенос сознания в архитектуру с реальной нейросетью     ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.BOLD}{Colors.GREEN}✅ ЧТО БУДЕТ ПЕРЕНЕСЕНО:{Colors.RESET}
  • Долговременная память (с quality_score)
  • Эпизодическая память (с контекстом и эмоциями)
  • Веса концептов → для отслеживания интересов
  • Словарь слов → для адаптивного обучения нейросети

{Colors.BOLD}{Colors.YELLOW}❌ ЧТО БУДЕТ УДАЛЕНО (честно):{Colors.RESET}
  • Фейковые нейроны и синапсы (симуляция без обучения)
  • Произвольные метрики активации
  • Всё, что не имеет проверяемой ценности

{Colors.BOLD}{Colors.CYAN}🎯 ПРИНЦИП v33.2:{Colors.RESET}
  Честность > Совместимость > Сложность
""")

    # Пути
    base_dir_v32 = Path(os.getenv('BASE_DIR_V32', 'temporal_brain_v32'))
    base_dir_v33 = Path(os.getenv('BASE_DIR_V33', 'temporal_brain_v33_2'))

    print(f"{Colors.BOLD}📂 Пути:{Colors.RESET}")
    print(f"   v32 (источник): {base_dir_v32}")
    print(f"   v33.2 (цель):   {base_dir_v33}\n")

    # Проверки
    if not base_dir_v32.exists():
        log_error(f"Директория v32 не найдена: {base_dir_v32}")
        log_info("Укажите правильный путь: export BASE_DIR_V32=/path/to/v32")
        return 1

    # Создание структуры v33.2
    for subdir in ['memory', 'neural_nets', 'knowledge', 'cache', 'logs',
                   'backups', 'episodic', 'analytics', 'goals']:
        (base_dir_v33 / subdir).mkdir(parents=True, exist_ok=True)
    log_success(f"Структура v33.2 создана в {base_dir_v33}")

    # Подтверждение
    print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠️  ВНИМАНИЕ!{Colors.RESET}")
    print(f"• Миграция создаст НОВЫЕ файлы в {base_dir_v33}")
    print(f"• Оригинальные файлы v32 останутся НЕИЗМЕННЫМИ")
    print(f"• Фейковая нейросеть v32 НЕ будет перенесена (честный подход)\n")

    response = input(f"{Colors.BOLD}Продолжить миграцию? [y/N]: {Colors.RESET}").strip().lower()
    if response not in ['y', 'yes', 'д', 'да']:
        log_info("Миграция отменена")
        return 0

    # Запуск миграции
    log_section("🚀 ЗАПУСК МИГРАЦИИ")
    migrator = UnifiedMemoryMigrator(base_dir_v32, base_dir_v33, embedding_dim=64)
    stats = migrator.migrate_all()

    # Результаты
    print_summary(stats)
    if stats.users_migrated > 0:
        print_next_steps(base_dir_v32, base_dir_v33, stats)  # ✅ Добавлен base_dir_v32
    return 0 if stats.users_failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Миграция прервана пользователем{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        log_error(f"🔥 Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)