#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 УНИВЕРСАЛЬНАЯ МИГРАЦИЯ → v35
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Поддерживает миграцию из:
- temporal_brain_v32_2
- temporal_brain_v33_2
- dynamic_brain_v34
- любой другой директории
"""

import sys
import gzip
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
import numpy as np
import time

# Импорты
sys.path.insert(0, str(Path(__file__).parent))
from dynamic_agi_v35_enhanced import (
    CONFIG,
    EpisodicMemory, SemanticMemory,
    CognitiveMemorySystem, DynamicVocabulary,
    DynamicNeuralNetwork, WordMetadata
)


class UniversalMigrator:
    """Универсальный мигратор"""

    def __init__(self):
        self.old_base_dir = self._find_old_directory()
        self.new_base_dir = Path('cognitive_brain_v35')

        self.stats = {
            'users_migrated': 0,
            'words_migrated': 0,
            'episodes_created': 0,
            'concepts_created': 0,
            'failures': []
        }

    def _find_old_directory(self) -> Path:
        """Автопоиск старой директории"""
        search_dirs = [
            'temporal_brain_v32_2',
            'temporal_brain_v33_2',
            'dynamic_brain_v34',
            'temporal_brain_v33',
            'temporal_brain_v32',
        ]

        for dirname in search_dirs:
            for location in [Path('.'), Path('..')]:
                path = location / dirname
                if path.exists():
                    print(f"✅ Найдена: {path}")
                    return path

        print("❓ Автопоиск не нашёл директорию")
        custom_path = input("Введите путь к директории: ").strip()
        path = Path(custom_path)

        if not path.exists():
            raise ValueError(f"Не найдена: {path}")

        return path

    def migrate_all_users(self) -> bool:
        """Миграция всех пользователей"""
        print(f"\n{'=' * 70}")
        print(f"🔄 МИГРАЦИЯ: {self.old_base_dir.name} → cognitive_brain_v35")
        print(f"{'=' * 70}\n")

        user_files = self._find_user_files()

        if not user_files:
            print("❌ Файлы не найдены")
            return False

        print(f"📦 Найдено пользователей: {len(user_files)}\n")

        for user_id, user_file in user_files.items():
            try:
                print(f"{'─' * 70}")
                self.migrate_user(user_id, user_file)
                self.stats['users_migrated'] += 1
            except Exception as e:
                print(f"❌ Ошибка {user_id}: {e}")
                self.stats['failures'].append(str(e))

        print(f"\n{'=' * 70}")
        print(f"✅ Мигрировано: {self.stats['users_migrated']}")
        print(f"{'=' * 70}\n")

        return True

    def _find_user_files(self) -> Dict[str, Path]:
        """Поиск файлов пользователей"""
        user_files = {}

        search_patterns = [
            self.old_base_dir / 'neural_nets' / '*.pkl.gz',
            self.old_base_dir / '*.pkl.gz',
            self.old_base_dir / 'memory' / '*.pkl.gz',
        ]

        for pattern in search_patterns:
            if not pattern.parent.exists():
                continue

            for file_path in pattern.parent.glob(pattern.name):
                filename = file_path.stem.replace('.pkl', '')

                # Очистка от суффиксов
                for suffix in ['_v32', '_v33', '_v34', '_adaptive', '_vocab', '_neural', '_memory']:
                    filename = filename.replace(suffix, '')

                if filename and filename not in user_files:
                    user_files[filename] = file_path

        return user_files

    def migrate_user(self, user_id: str, state_file: Path):
        """Миграция пользователя"""
        print(f"👤 {user_id}")
        print(f"   Файл: {state_file.name}")

        # Загрузка
        old_state = self._load_state(state_file)
        if not old_state:
            print(f"   ⚠️ Не удалось загрузить")
            return

        print(f"   📂 Загружено ({len(old_state)} ключей)")

        # Создание компонентов
        vocabulary = DynamicVocabulary(
            initial_size=CONFIG.initial_vocab_size,
            embedding_dim=CONFIG.embedding_dim,
            enable_async=False
        )

        neural = DynamicNeuralNetwork(
            input_dim=CONFIG.embedding_dim,
            initial_hidden_dim=CONFIG.initial_hidden_dim,
            output_dim=CONFIG.output_metrics_dim
        )

        memory = CognitiveMemorySystem(
            embedding_dim=CONFIG.embedding_dim,
            embed_func=vocabulary.encode_text
        )

        # Миграция
        self._migrate_vocab(old_state, vocabulary)
        self._migrate_neural(old_state, neural)
        self._migrate_memory(old_state, memory, vocabulary)

        # Сохранение
        self._save(user_id, vocabulary, neural, memory, old_state)

        print(f"   ✅ Успешно")

    def _load_state(self, path: Path) -> Dict:
        """Загрузка состояния"""
        try:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return {}

    def _migrate_vocab(self, old_state: Dict, vocab: DynamicVocabulary):
        """Миграция словаря"""
        print(f"   📚 Словарь...", end=' ')

        # Поиск эмбеддингов
        emb_key = None
        for key in ['vocab_embeddings', 'embedding_matrix', 'embeddings']:
            if key in old_state:
                emb_key = key
                break

        if not emb_key:
            print("⚠️ не найден")
            return

        old_emb = old_state[emb_key]
        word_to_idx = old_state.get('vocab_word_to_idx', old_state.get('word_to_idx', {}))

        vocab_size, old_dim = old_emb.shape
        new_dim = vocab.embedding_dim

        # Адаптация
        if old_dim == new_dim:
            if vocab_size > vocab.current_vocab_size:
                vocab._expand_vocabulary(vocab_size)
            vocab.embeddings[:vocab_size] = old_emb
            vocab.word_to_idx = word_to_idx
            vocab.idx_to_word = {v: k for k, v in word_to_idx.items()}
            vocab.next_idx = len(word_to_idx)
        else:
            for word, old_idx in word_to_idx.items():
                if old_idx >= len(old_emb):
                    continue

                old_vec = old_emb[old_idx]

                if new_dim > old_dim:
                    new_vec = np.zeros(new_dim)
                    new_vec[:old_dim] = old_vec
                    new_vec[old_dim:] = np.random.randn(new_dim - old_dim) * 0.01
                else:
                    new_vec = old_vec[:new_dim]

                new_idx = vocab.add_word(word, 0.7)
                vocab.embeddings[new_idx] = new_vec

        # Метаданные
        for word, meta in old_state.get('vocab_metadata', {}).items():
            if word in vocab.word_to_idx:
                try:
                    vocab.word_metadata[word] = WordMetadata(**meta)
                except:
                    vocab.word_metadata[word] = WordMetadata(word=word)

        self.stats['words_migrated'] += vocab.next_idx
        print(f"✅ {vocab.next_idx} слов")

    def _migrate_neural(self, old_state: Dict, neural: DynamicNeuralNetwork):
        """Миграция нейросети"""
        print(f"   🧬 Нейросеть...", end=' ')

        # Поиск весов
        w1_key = 'neural_W1' if 'neural_W1' in old_state else 'W1'

        if w1_key not in old_state:
            print("⚠️ не найдена")
            return

        old_W1 = old_state[w1_key]
        old_b1 = old_state.get('neural_b1', old_state.get('b1'))
        old_W2 = old_state.get('neural_W2', old_state.get('W2'))
        old_b2 = old_state.get('neural_b2', old_state.get('b2'))

        old_in, old_hid = old_W1.shape
        old_out = old_W2.shape[1]

        # Адаптация входа
        if old_in != neural.input_dim:
            min_in = min(old_in, neural.input_dim)
            min_hid = min(old_hid, neural.hidden_dim)

            while neural.hidden_dim < min_hid:
                neural._expand_network()

            neural.W1[:min_in, :min_hid] = old_W1[:min_in, :min_hid]
            neural.b1[:min_hid] = old_b1[:min_hid]

            if neural.input_dim > old_in:
                neural.W1[old_in:, :min_hid] = np.random.randn(
                    neural.input_dim - old_in, min_hid
                ) * 0.01
        else:
            while neural.hidden_dim < old_hid:
                neural._expand_network()

            neural.W1[:, :old_hid] = old_W1
            neural.b1[:old_hid] = old_b1

        # Адаптация выхода
        if old_out != neural.output_dim:
            min_hid = min(old_hid, neural.hidden_dim)
            min_out = min(old_out, neural.output_dim)

            neural.W2[:min_hid, :min_out] = old_W2[:min_hid, :min_out]
            neural.b2[:min_out] = old_b2[:min_out]

            if neural.output_dim > old_out:
                neural.W2[:min_hid, old_out:] = np.random.randn(
                    min_hid, neural.output_dim - old_out
                ) * 0.01
                neural.b2[old_out:] = np.random.randn(
                    neural.output_dim - old_out
                ) * 0.01
        else:
            min_hid = min(old_hid, neural.hidden_dim)
            neural.W2[:min_hid, :] = old_W2[:min_hid, :]
            neural.b2 = old_b2

        # Статистика
        neural.total_updates = old_state.get('neural_total_updates',
                                             old_state.get('total_updates', 0))

        print(f"✅ {old_in}→{old_hid}→{old_out}")

    def _migrate_memory(self, old_state: Dict, memory: CognitiveMemorySystem, vocab: DynamicVocabulary):
        """Создание памяти из истории"""
        print(f"   🧠 Память...", end=' ')

        history = old_state.get('interaction_history', [])

        if not history:
            print("⚠️ нет истории")
            return

        episodes = 0
        concepts = 0
        concepts_set = set()

        for i, item in enumerate(history):
            user_input = item.get('user_input', item.get('input', ''))
            response = item.get('response', item.get('output', ''))

            if not user_input or not response:
                continue

            content = f"User: {user_input}\nAssistant: {response}"

            importance = item.get('actual_metrics', {}).get('relevance', 0.5)
            quality = item.get('quality_score', importance)

            embedding = vocab.encode_text(content)
            timestamp = item.get('timestamp', time.time() - (len(history) - i) * 3600)

            episode = EpisodicMemory(
                content=content,
                timestamp=timestamp,
                embedding=embedding,
                importance=importance,
                emotional_valence=(quality - 0.5) * 2,
                arousal=min(1.0, len(user_input.split()) / 15),
                context={'quality': quality}
            )

            memory.episodic.add(episode)
            episodes += 1

            # Концепты
            if quality >= 0.6:
                for word in user_input.lower().split():
                    if len(word) > 4 and word not in concepts_set:
                        concept_emb = vocab.encode_text(f"{word}: {user_input[:100]}")

                        semantic = SemanticMemory(
                            concept=word,
                            definition=f"Из: {user_input[:60]}...",
                            embedding=concept_emb,
                            confidence=quality
                        )

                        memory.semantic.add(semantic)
                        concepts_set.add(word)
                        concepts += 1

        self.stats['episodes_created'] += episodes
        self.stats['concepts_created'] += concepts

        print(f"✅ {episodes} эпизодов, {concepts} концептов")

    def _save(self, user_id: str, vocab: DynamicVocabulary,
              neural: DynamicNeuralNetwork, memory: CognitiveMemorySystem, old_state: Dict):
        """Сохранение"""
        print(f"   💾 Сохранение...", end=' ')

        # Нейросеть
        neural_path = self.new_base_dir / 'neural_nets' / f'{user_id}_v35.pkl.gz'
        neural_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'version': '35.0',
            'timestamp': time.time(),
            'migrated_from': self.old_base_dir.name,

            'vocab_embeddings': vocab.embeddings,
            'vocab_word_to_idx': vocab.word_to_idx,
            'vocab_idx_to_word': vocab.idx_to_word,
            'vocab_metadata': {w: asdict(m) for w, m in vocab.word_metadata.items()},
            'vocab_next_idx': vocab.next_idx,
            'vocab_current_size': vocab.current_vocab_size,

            'neural_W1': neural.W1,
            'neural_b1': neural.b1,
            'neural_W2': neural.W2,
            'neural_b2': neural.b2,
            'neural_hidden_dim': neural.hidden_dim,
            'neural_total_updates': neural.total_updates,

            'total_interactions': old_state.get('total_interactions', 0),
        }

        with gzip.open(neural_path, 'wb') as f:
            pickle.dump(state, f)

        # Память
        memory_path = self.new_base_dir / 'memory' / f'{user_id}_memory.pkl.gz'
        memory.save(memory_path)

        print("✅")

    def save_report(self):
        """Отчёт"""
        report_path = self.new_base_dir / 'migrations' / f'report_{int(time.time())}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"\n📄 Отчёт: {report_path}")


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║  🔄 УНИВЕРСАЛЬНАЯ МИГРАЦИЯ → v35                               ║
╚════════════════════════════════════════════════════════════════╝

Поддерживает:
✅ temporal_brain_v32_2
✅ temporal_brain_v33_2
✅ dynamic_brain_v34
✅ любую другую директорию
""")

    try:
        migrator = UniversalMigrator()

        input("\nНажмите Enter...")

        migrator.migrate_all_users()
        migrator.save_report()

        print(f"""
╔════════════════════════════════════════════════════════════════╗
║  ✅ МИГРАЦИЯ ЗАВЕРШЕНА                                         ║
╚════════════════════════════════════════════════════════════════╝

📊 СТАТИСТИКА:
  • Пользователей: {migrator.stats['users_migrated']}
  • Слов: {migrator.stats['words_migrated']}
  • Эпизодов: {migrator.stats['episodes_created']}
  • Концептов: {migrator.stats['concepts_created']}
  • Ошибок: {len(migrator.stats['failures'])}

🚀 Запустите: python dynamic_agi_v35_enhanced.py
""")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()