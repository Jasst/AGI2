#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 МИГРАЦИЯ ПАМЯТИ v31.1 → v32.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Переносит все данные из старой версии в новую с сохранением:
✅ Долговременная память
✅ Эпизодическая память
✅ Нейронная сеть (нейроны + синапсы)
✅ Мета-данные
"""
import os
import sys
import gzip
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Пути к данным
OLD_BASE_DIR = Path('temporal_brain_v31')  # Старая версия
NEW_BASE_DIR = Path('temporal_brain_v32')  # Новая версия


def migrate_memory_item_v31_to_v32(old_item: Dict) -> Dict:
    """Конвертация элемента памяти v31.1 → v32.0"""
    new_item = {
        'content': old_item.get('content', ''),
        'timestamp': old_item.get('timestamp', 0.0),
        'importance': old_item.get('importance', 0.5),
        # 🔧 НОВЫЕ ПОЛЯ v32.0:
        'priority': 0.5,  # Default для приоритизации
        'access_count': old_item.get('access_count', 0),
        'last_access': old_item.get('last_access', 0.0),
        'metadata': old_item.get('metadata', {}),
        'quality_score': 0.5,  # Default для качества
    }
    return new_item


def migrate_episode_v31_to_v32(old_ep: Dict) -> Dict:
    """Конвертация эпизода v31.1 → v32.0"""
    new_ep = {
        'id': old_ep.get('id', ''),
        'messages': old_ep.get('messages', []),
        'context': old_ep.get('context', ''),
        'timestamp': old_ep.get('timestamp', 0.0),
        'importance': old_ep.get('importance', 0.5),
        # 🔧 НОВЫЕ ПОЛЯ v32.0:
        'quality_score': 0.5,  # Default для качества
        'consolidation_count': 0,  # Счётчик консолидаций
    }
    return new_ep


def migrate_neuron_v31_to_v32(old_neuron: Dict) -> Dict:
    """Конвертация нейрона v31.1 → v32.0"""
    new_neuron = {
        'id': old_neuron.get('id', ''),
        'layer': old_neuron.get('layer', 0),
        'module': old_neuron.get('module', 'general'),
        'bias': old_neuron.get('bias', 0.0),
        'neuron_type': old_neuron.get('neuron_type', 'general'),
        'created_at': old_neuron.get('created_at', 0.0),
        'activation_count': old_neuron.get('activation_count', 0),
        'specialization': old_neuron.get('specialization', None),
        'importance_score': old_neuron.get('importance_score', 0.5),
        # 🔧 НОВОЕ ПОЛЕ v32.0:
        'synergy_bonus': 0.0,  # Для кросс-модульной синергии
    }
    return new_neuron


def migrate_synapse_v31_to_v32(old_synapse: Dict) -> Dict:
    """Конвертация синапса v31.1 → v32.0"""
    new_synapse = {
        'source_id': old_synapse.get('source_id', ''),
        'target_id': old_synapse.get('target_id', ''),
        'weight': old_synapse.get('weight', 0.0),
        'strength': old_synapse.get('strength', 1.0),
        'plasticity': old_synapse.get('plasticity', 1.0),
        'activation_count': old_synapse.get('activation_count', 0),
        'created_at': old_synapse.get('created_at', 0.0),
        'attention_weight': old_synapse.get('attention_weight', 1.0),
        # 🔧 НОВОЕ ПОЛЕ v32.0:
        'cross_module': False,  # Для межмодульных связей
    }
    return new_synapse


def migrate_user_memory(user_id: str) -> bool:
    """Миграция памяти конкретного пользователя"""
    print(f"\n🔄 Миграция пользователя {user_id}...")

    # Пути к файлам
    old_memory_path = OLD_BASE_DIR / 'memory' / f'user_{user_id}' / 'memory_v31.pkl.gz'
    new_memory_path = NEW_BASE_DIR / 'memory' / f'user_{user_id}' / 'memory_v32.pkl.gz'

    if not old_memory_path.exists():
        print(f"⚠️  Файл памяти не найден: {old_memory_path}")
        return False

    try:
        # Загрузка старой памяти
        with gzip.open(old_memory_path, 'rb') as f:
            old_mem_state = pickle.load(f)

        # Конвертация долговременной памяти
        new_ltm = {}
        for mid, item_data in old_mem_state.get('long_term', {}).items():
            new_ltm[mid] = migrate_memory_item_v31_to_v32(item_data)

        # Конвертация эпизодов
        new_episodic = {}
        for eid, ep_data in old_mem_state.get('episodic', {}).items():
            new_episodic[eid] = migrate_episode_v31_to_v32(ep_data)

        # Сохранение новой памяти
        new_mem_state = {
            'long_term': new_ltm,
            'episodic': new_episodic,
        }

        new_memory_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(new_memory_path, 'wb', compresslevel=6) as f:
            pickle.dump(new_mem_state, f)

        print(f"✅ Память мигрирована: {len(new_ltm)} LTM, {len(new_episodic)} episodes")
        return True

    except Exception as e:
        print(f"❌ Ошибка миграции памяти: {e}")
        import traceback
        traceback.print_exc()
        return False


def migrate_user_neural(user_id: str) -> bool:
    """Миграция нейросети конкретного пользователя"""
    print(f"\n🧬 Миграция нейросети пользователя {user_id}...")

    # Пути к файлам
    old_neural_path = OLD_BASE_DIR / 'neural_nets' / f'{user_id}_v31.pkl.gz'
    new_neural_path = NEW_BASE_DIR / 'neural_nets' / f'{user_id}_v32.pkl.gz'

    if not old_neural_path.exists():
        print(f"⚠️  Файл нейросети не найден: {old_neural_path}")
        return False

    try:
        # Загрузка старой нейросети
        with gzip.open(old_neural_path, 'rb') as f:
            old_neural_state = pickle.load(f)

        # Конвертация нейронов
        new_neurons = []
        for neuron_data in old_neural_state.get('neurons', []):
            new_neurons.append(migrate_neuron_v31_to_v32(neuron_data))

        # Конвертация синапсов
        new_synapses = []
        for synapse_data in old_neural_state.get('synapses', []):
            new_synapses.append(migrate_synapse_v31_to_v32(synapse_data))

        # Сохранение новой нейросети
        new_neural_state = {
            'neurons': new_neurons,
            'synapses': new_synapses,
            'layers': old_neural_state.get('layers', {}),
            'modules': old_neural_state.get('modules', {}),
            'meta': {
                'total_activations': old_neural_state.get('meta', {}).get('total_activations', 0),
                'neurogenesis_events': old_neural_state.get('meta', {}).get('neurogenesis_events', 0),
                'pruning_events': old_neural_state.get('meta', {}).get('pruning_events', 0),
                'meta_learning_score': old_neural_state.get('meta', {}).get('meta_learning_score', 0.5),
                # 🔧 НОВЫЕ ПОЛЯ v32.0:
                'plateau_cycles': 0,
                'current_learning_rate': 0.12,
            }
        }

        new_neural_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(new_neural_path, 'wb', compresslevel=6) as f:
            pickle.dump(new_neural_state, f)

        print(f"✅ Нейросеть мигрирована: {len(new_neurons)} нейронов, {len(new_synapses)} синапсов")
        return True

    except Exception as e:
        print(f"❌ Ошибка миграции нейросети: {e}")
        import traceback
        traceback.print_exc()
        return False


def migrate_all_users():
    """Миграция всех пользователей"""
    print("=" * 60)
    print("🔄 МИГРАЦИЯ v31.1 → v32.0")
    print("=" * 60)

    # Поиск всех пользователей в старой версии
    old_memory_dir = OLD_BASE_DIR / 'memory'
    if not old_memory_dir.exists():
        print(f"❌ Директория старой версии не найдена: {old_memory_dir}")
        return

    user_dirs = [d for d in old_memory_dir.iterdir() if d.is_dir() and d.name.startswith('user_')]

    if not user_dirs:
        print("⚠️  Пользователи не найдены")
        return

    print(f"📊 Найдено пользователей: {len(user_dirs)}")

    success_count = 0
    for user_dir in user_dirs:
        user_id = user_dir.name.replace('user_', '')
        print(f"\n{'=' * 60}")

        memory_ok = migrate_user_memory(user_id)
        neural_ok = migrate_user_neural(user_id)

        if memory_ok and neural_ok:
            success_count += 1
            print(f"✅ {user_id}: УСПЕШНО")
        else:
            print(f"⚠️  {user_id}: ЧАСТИЧНО или ОШИБКА")

    print(f"\n{'=' * 60}")
    print(f"📊 ИТОГИ: {success_count}/{len(user_dirs)} пользователей мигрировано")
    print("=" * 60)


if __name__ == "__main__":
    # Вариант A: Миграция всех пользователей
    migrate_all_users()

    # Вариант B: Миграция конкретного пользователя (раскомментируйте)
    # migrate_user_memory('1198981287')
    # migrate_user_neural('1198981287')