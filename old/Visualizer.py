#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 NEURAL ACTIVITY VISUALIZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Визуализация работы когнитивного мозга в реальном времени
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import asyncio
import random
from autonomous_cognitive_brain import HybridCognitiveBrain
from collections import defaultdict


class BrainVisualizer:
    """ASCII визуализация активности мозга"""

    def __init__(self, brain: HybridCognitiveBrain):
        self.brain = brain
        self.history = defaultdict(lambda: [0] * 50)  # История активности

    def visualize_step(self, state: dict):
        """Отобразить один временной шаг"""
        regions = state.get("active_regions", {})
        thought = state.get("thought", "")

        print(f"\r⏱️  t={state['time']:05d}ms | ", end="")

        # Активность по регионам
        for region in ["sensory", "cortex", "subcortex", "motor"]:
            count = regions.get(region, 0)
            bar_len = min(20, count)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"{region[:4].upper()}:{bar[:10]} | ", end="")

        # Текущая мысль
        if thought:
            print(f"💭 {thought[:30]}", end="")

        print("", flush=True)

    def plot_raster(self, duration_ms: int = 1000, display_neurons: int = 50):
        """
        Растровая диаграмма (raster plot) спайков

        Показывает какие нейроны активны в каждый момент времени
        """
        print(f"\n📊 RASTER PLOT ({duration_ms}ms, {display_neurons} neurons)")
        print("=" * 80)

        # Собираем данные
        spike_data = defaultdict(list)

        for t in range(duration_ms):
            state = self.brain.brain.step(sensory_input=None)

            # Записываем спайки
            for nid in list(state.get("active_regions", {}).keys())[:display_neurons]:
                spike_data[nid].append(t)

        # Отображаем
        neuron_ids = sorted(spike_data.keys())[:display_neurons]

        for nid in neuron_ids:
            neuron = self.brain.brain.neurons[nid]
            line = [" "] * (duration_ms // 10)  # Каждые 10ms = 1 символ

            for spike_time in spike_data[nid]:
                idx = spike_time // 10
                if idx < len(line):
                    line[idx] = "█"

            region_icon = {"sensory": "👁️", "cortex": "🧠", "subcortex": "💚", "motor": "🦾"}.get(neuron.region, "⚪")
            print(f"{region_icon} {neuron.label[:15]:<15} | {''.join(line)}")

        print("=" * 80)

    def show_connectivity(self, neuron_label: str):
        """Показать связи конкретного нейрона"""
        if neuron_label not in self.brain.brain.label_to_neuron:
            print(f"❌ Нейрон '{neuron_label}' не найден")
            return

        nid = self.brain.brain.label_to_neuron[neuron_label]
        neuron = self.brain.brain.neurons[nid]

        print(f"\n🔗 СВЯЗИ НЕЙРОНА '{neuron_label}'")
        print("=" * 60)
        print(f"Регион: {neuron.region}")
        print(f"Мембранный потенциал: {neuron.membrane_potential:.3f}")
        print(f"Порог: {neuron.threshold:.3f}")
        print(f"Всего спайков: {neuron.total_spikes}")

        # Входящие связи
        incoming = [s for s in self.brain.brain.synapses if s.target_id == nid]
        print(f"\n📥 Входящие ({len(incoming)}):")
        for syn in sorted(incoming, key=lambda s: -s.weight)[:10]:
            src = self.brain.brain.neurons[syn.source_id]
            bar = "█" * int(syn.weight * 10)
            print(f"  {src.label[:15]:<15} ──[{syn.weight:.2f}]─→ {bar}")

        # Исходящие связи
        outgoing = [s for s in self.brain.brain.synapses if s.source_id == nid]
        print(f"\n📤 Исходящие ({len(outgoing)}):")
        for syn in sorted(outgoing, key=lambda s: -s.weight)[:10]:
            tgt = self.brain.brain.neurons[syn.target_id]
            bar = "█" * int(syn.weight * 10)
            print(f"  {bar} ──[{syn.weight:.2f}]─→ {tgt.label[:15]:<15}")

        print("=" * 60)

    def weight_distribution(self):
        """Распределение весов синапсов"""
        weights = [s.weight for s in self.brain.brain.synapses]

        print("\n📊 РАСПРЕДЕЛЕНИЕ ВЕСОВ СИНАПСОВ")
        print("=" * 60)

        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        hist = [0] * (len(bins) - 1)

        for w in weights:
            for i in range(len(bins) - 1):
                if bins[i] <= w < bins[i + 1]:
                    hist[i] += 1
                    break

        max_count = max(hist)
        for i, count in enumerate(hist):
            bar_len = int((count / max_count) * 40) if max_count > 0 else 0
            bar = "█" * bar_len
            print(f"  [{bins[i]:.1f}-{bins[i + 1]:.1f}] {bar} {count}")

        print(f"\nСредний вес: {sum(weights) / len(weights):.3f}")
        print(f"Сильных связей (>1.0): {sum(1 for w in weights if w > 1.0)}")
        print("=" * 60)


async def interactive_demo():
    """Интерактивная демонстрация"""
    print("🧠 NEURAL ACTIVITY VISUALIZER")
    print("=" * 80)

    brain = HybridCognitiveBrain("brain_viz.json")
    viz = BrainVisualizer(brain)

    while True:
        print("\n" + "=" * 80)
        print("МЕНЮ:")
        print("  1. Внутренний монолог (3 сек)")
        print("  2. Raster plot")
        print("  3. Показать связи нейрона")
        print("  4. Распределение весов")
        print("  5. Отправить запрос")
        print("  6. Обучить концепт")
        print("  7. Статистика")
        print("  0. Выход")
        print("=" * 80)

        choice = input("\n>>> Выбор: ").strip()

        if choice == "1":
            print("\n💭 ВНУТРЕННИЙ МОНОЛОГ (3000 мс)")
            print("-" * 80)

            for _ in range(300):  # 3 секунды
                state = brain.brain.step(sensory_input=None)

                if _ % 10 == 0:  # Обновляем каждые 10 мс
                    viz.visualize_step(state)

                await asyncio.sleep(0.001)  # Имитация реального времени

            print("\n✅ Завершено")

        elif choice == "2":
            duration = input("Длительность (мс) [1000]: ").strip()
            duration = int(duration) if duration.isdigit() else 1000

            viz.plot_raster(duration_ms=duration)

        elif choice == "3":
            neuron = input("Имя нейрона (например, cortex_0): ").strip()
            viz.show_connectivity(neuron)

        elif choice == "4":
            viz.weight_distribution()

        elif choice == "5":
            query = input("Ваш запрос: ").strip()

            if query:
                print("\n🤔 Обрабатываю...")
                response = await brain.think(query)
                print(f"\n🤖 {response}")

        elif choice == "6":
            concept = input("Концепт: ").strip()
            examples = []

            print("Примеры (пустая строка = конец):")
            while True:
                ex = input("  > ").strip()
                if not ex:
                    break
                examples.append(ex)

            if concept and examples:
                brain.learn_concept(concept, examples)
                print(f"✅ Концепт '{concept}' обучен на {len(examples)} примерах")

        elif choice == "7":
            stats = brain.stats()
            print("\n📊 СТАТИСТИКА")
            print("=" * 60)
            for k, v in stats.items():
                print(f"  {k:20}: {v}")
            print("=" * 60)

        elif choice == "0":
            print("\n👋 До встречи!")
            break

        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    try:
        asyncio.run(interactive_demo())
    except KeyboardInterrupt:
        print("\n\n👋 Остановлено")