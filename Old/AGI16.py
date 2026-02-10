# coding: utf-8
"""
AGI_Hybrid_v22_OPENROUTER_FINAL.py — С ВИЗУАЛЬНОЙ ОБРАТНОЙ СВЯЗЬЮ
Полностью рабочая когнитивная система с отображением всех ответов
"""

import re, json, requests, time, os, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone


# ================= ЗАГРУЗКА КЛЮЧА =================
def load_api_key():
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
    OBJECTS = ROOT / "objects.json"
    CAUSAL = ROOT / "causal.json"
    META = ROOT / "meta.json"
    EPISODE = ROOT / "episodes.json"
    LOG = ROOT / "log.txt"

    TIMEOUT = 25
    MAX_CHAIN = 8
    MIN_CONF = 0.15
    FORGET_RATE = 0.01
    MEMORY_LIMIT = 150
    WORKING_SIZE = 30

    # 🔑 ИСПРАВЛЕНО: убраны пробелы в конце URL!
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = load_api_key()
    MODEL = "qwen/qwen-2.5-7b-instruct"  # Официальный Qwen


# ================= УТИЛИТЫ =================
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def extract(text: str) -> List[str]:
    stop = {"что", "как", "почему", "если", "то", "это", "я", "ты", "мы", "они", "он", "она", "оно"}
    return [w for w in clean(text).split() if len(w) > 3 and w not in stop]


def print_typing(text: str, delay=0.012):
    """Эффект печатания с гарантированным отображением"""
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print(flush=True)


# ================= КОНЦЕПТЫ =================
@dataclass
class Concept:
    name: str
    confidence: float = 0.2
    effects: Dict[str, float] = field(default_factory=dict)
    abstract: bool = False
    freq: int = 0

    def reinforce(self, k=0.1):
        self.confidence = min(1.0, self.confidence + k)
        self.freq += 1

    def decay(self):
        self.confidence *= (1 - Config.FORGET_RATE)
        self.freq = max(0, self.freq - 1)


# ================= ЭПИЗОДЫ =================
@dataclass
class Episode:
    time: str
    input: str
    focus: Optional[str]
    result: str


# ================= РАБОЧАЯ ПАМЯТЬ =================
@dataclass
class WorkingMemoryItem:
    content: str
    timestamp: float
    importance: float


class WorkingMemory:
    def __init__(self, size=Config.WORKING_SIZE):
        self.items: List[WorkingMemoryItem] = []
        self.size = size

    def add(self, content, importance=0.5):
        self.items.append(WorkingMemoryItem(content, time.time(), importance))
        self.items.sort(key=lambda x: -x.importance)
        self.items = self.items[:self.size]

    def recall(self):
        return [i.content for i in self.items]


# ================= СЕМАНТИЧЕСКАЯ ПАМЯТЬ =================
class SemanticMemory:
    def __init__(self):
        self.data: Dict[str, Concept] = {}
        self.load()

    def get(self, name: str) -> Concept:
        if name not in self.data:
            self.data[name] = Concept(name=name)
        return self.data[name]

    def link(self, a: str, b: str):
        c = self.get(a)
        c.effects[b] = min(1.0, c.effects.get(b, 0.1) + 0.3)
        c.reinforce()

    def decay_all(self):
        for c in self.data.values():
            c.decay()

    def generate_abstracts(self):
        names = list(self.data.keys())
        for i, c1 in enumerate(names):
            for c2 in names[i + 1:]:
                common = set(self.data[c1].effects) & set(self.data[c2].effects)
                if len(common) / max(len(self.data[c1].effects), 1) > 0.5:
                    abs_name = f"{c1}_{c2}_meta"
                    self.get(abs_name).abstract = True
                    self.link(c1, abs_name)
                    self.link(c2, abs_name)

    def save(self):
        with open(Config.OBJECTS, "w", encoding="utf-8") as f:
            json.dump({k: v.__dict__ for k, v in self.data.items()}, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.OBJECTS.exists():
            with open(Config.OBJECTS, "r", encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    self.data[k] = Concept(**v)


# ================= ПРИЧИННО-СЛЕДСТВЕННАЯ ПАМЯТЬ =================
class CausalMemory:
    def __init__(self):
        self.graph: Dict[str, Dict[str, float]] = {}
        self.load()

    def add(self, a: str, b: str):
        self.graph.setdefault(a, {})
        self.graph[a][b] = min(1.0, self.graph[a].get(b, 0.1) + 0.25)

    def chain(self, start: str) -> List[str]:
        chain = [start];
        cur = start
        for _ in range(Config.MAX_CHAIN):
            if cur not in self.graph or not self.graph[cur]: break
            nxt = max(self.graph[cur], key=self.graph[cur].get)
            if nxt in chain: break
            chain.append(nxt)
            cur = nxt
        return chain

    def predict(self, start: str, steps=3) -> List[str]:
        result = []
        cur = start
        for _ in range(steps):
            if cur not in self.graph or not self.graph[cur]: break
            nxt = max(self.graph[cur], key=self.graph[cur].get)
            result.append(nxt)
            cur = nxt
        return result

    def prune(self):
        for a in list(self.graph.keys()):
            for b in list(self.graph[a].keys()):
                self.graph[a][b] *= (1 - Config.FORGET_RATE)
                if self.graph[a][b] < Config.MIN_CONF:
                    del self.graph[a][b]
            if not self.graph[a]:
                del self.graph[a]

    def save(self):
        with open(Config.CAUSAL, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.CAUSAL.exists():
            with open(Config.CAUSAL, "r", encoding="utf-8") as f:
                self.graph = json.load(f)


# ================= САМОМОДЕЛЬ =================
class SelfModel:
    def describe(self, stats: Dict) -> str:
        return (f"Я когнитивная система v22.\n"
                f"Взаимодействий: {stats['interactions']}\n"
                f"Изученных связей: {stats['links']}\n"
                f"Я обучаюсь, прогнозирую последствия и могу уточнять детали.")

    def reflect(self, semantic: SemanticMemory, query: str) -> str:
        concepts = extract(query)
        if not concepts: return ""
        confidences = [semantic.get(c).confidence for c in concepts]
        known_words = {"привет", "здравствуй", "работа", "ты", "я", "система", "да", "нет"}
        for c in concepts:
            if c in known_words:
                semantic.get(c).confidence = max(semantic.get(c).confidence, 0.5)
        if confidences and min(confidences) < 0.2:
            return "Я не уверен в этом, могу уточнить у вас?"
        return ""


# ================= КОГНИТИВНАЯ СИСТЕМА =================
class CognitiveSystemV22:
    def __init__(self):
        print("🧠 Cognitive System v22 — OpenRouter Integration\n")

        if not Config.OPENROUTER_API_KEY:
            print("❌ КРИТИЧЕСКАЯ ОШИБКА: Не найден OPENROUTER_API_KEY!")
            print("   Создайте файл .env со строкой:")
            print("   OPENROUTER_API_KEY=sk-or-v1-ваш_ключ")
            sys.exit(1)

        self.semantic = SemanticMemory()
        self.causal = CausalMemory()
        self.working = WorkingMemory()
        self.self_model = SelfModel()
        self.episodes: List[Episode] = []
        self.meta = self.load_meta()
        self.log_fd = open(Config.LOG, "a", encoding="utf-8")
        self.internal_log("Инициализация завершена")

    def internal_log(self, msg: str):
        ts = datetime.now(timezone.utc).isoformat()
        self.log_fd.write(f"[{ts}] {msg}\n")
        self.log_fd.flush()

    # ------------------------------------------------------------------
    def process(self, text: str) -> str:
        """ВСЕГДА возвращает строку ответа (никаких скрытых возвратов без вывода!)"""
        self.meta["interactions"] += 1
        self.working.add(text)
        words = extract(text)
        focus = words[0] if words else None
        answer = ""

        # 1️⃣ Причинное обучение (если-то)
        if "если" in text and "то" in text:
            parts = clean(text).split("то", 1)
            c = extract(parts[0]);
            e = extract(parts[1])
            if c and e:
                self.causal.add(c[-1], e[0])
                self.semantic.link(c[-1], e[0])
                self.meta["links"] += 1
                self.save()
                answer = f"🧠 Усвоена причинность: {c[-1]} → {e[0]}"
                self.internal_log(f"Причинность: {c[-1]} → {e[0]}")
                return answer

        # 2️⃣ Причинная цепь
        if focus:
            chain = self.causal.chain(focus)
            if len(chain) > 1:
                answer = "🔗 Причинная цепь: " + " → ".join(chain)
                self.internal_log(f"Цепь для '{focus}': {chain}")
                return answer

        # 3️⃣ Метакогнитивный отклик
        reflect = self.self_model.reflect(self.semantic, text)
        if reflect and len(words) > 1:
            answer = "🤔 " + reflect
            return answer

        # 4️⃣ Запрос к внешней модели
        answer = self.learn_from_openrouter(text)

        # 5️⃣ Прогноз последствий
        if focus:
            prediction = self.causal.predict(focus)
            if prediction:
                answer += f"\n\n🔮 Прогноз: {' → '.join(prediction)}"

        # 6️⃣ Обновление памяти
        self.semantic.decay_all()
        self.causal.prune()
        self.semantic.generate_abstracts()

        # 7️⃣ Логирование
        now = datetime.now(timezone.utc).isoformat()
        self.episodes.append(Episode(now, text, focus, answer))
        self.episodes = self.episodes[-Config.MEMORY_LIMIT:]
        self.save()

        return answer

    # ------------------------------------------------------------------
    def learn_from_openrouter(self, q: str) -> str:
        try:
            print("⏳ Запрашиваю ответ у нейросети...", flush=True)
            time.sleep(0.3)  # Визуальная пауза для обратной связи

            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "CognitiveSystemV22"
            }
            payload = {
                "model": Config.MODEL,
                "messages": [
                    {"role": "system", "content": "Ты — когнитивная система. Отвечай кратко на русском языке."},
                    {"role": "user", "content": q}
                ],
                "temperature": 0.3,
                "max_tokens": 250
            }

            r = requests.post(
                Config.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=Config.TIMEOUT
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            # Интеграция в память
            txt = clean(content)
            w = extract(txt)
            for i in range(len(w) - 1):
                self.semantic.link(w[i], w[i + 1])
                self.causal.add(w[i], w[i + 1])
                self.meta["links"] += 1

            return content

        except Exception as e:
            error_msg = f"⚠️ Ошибка: {str(e)[:80]}"
            self.internal_log(f"OpenRouter ошибка: {e}")
            return error_msg

    # ------------------------------------------------------------------
    def save(self):
        self.semantic.save()
        self.causal.save()
        with open(Config.EPISODE, "w", encoding="utf-8") as f:
            json.dump([e.__dict__ for e in self.episodes], f, ensure_ascii=False, indent=2)
        with open(Config.META, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load_meta(self) -> Dict:
        if Config.META.exists():
            with open(Config.META, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"interactions": 0, "links": 0}


# ================= ДИАГНОСТИКА =================
def run_diagnosis() -> bool:
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА СИСТЕМЫ")
    print("=" * 60)

    if not Config.OPENROUTER_API_KEY:
        print("❌ Не найден OPENROUTER_API_KEY")
        return False

    print(f"✅ Ключ: {Config.OPENROUTER_API_KEY[:8]}...{Config.OPENROUTER_API_KEY[-4:]}")
    print(f"✅ Модель: {Config.MODEL}")

    try:
        print("📡 Проверка подключения...", end=" ", flush=True)
        r = requests.post(
            Config.OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Test"
            },
            json={"model": Config.MODEL, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 5},
            timeout=10
        )
        if r.status_code == 200:
            print("✅ УСПЕХ")
            return True
        else:
            print(f"❌ ОШИБКА {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ СЕТЬ: {e}")
        return False


# ================= ОСНОВНОЙ ЦИКЛ =================
def main():
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
    print("=" * 60 + "\n")

    if not run_diagnosis():
        print("\n💡 Решение проблем:")
        print("   1. Создайте файл .env с ключом OPENROUTER_API_KEY")
        print("   2. Убедитесь, что модель указана БЕЗ ':free'")
        print("   3. URL API не должен содержать пробелов в конце")
        return

    system = CognitiveSystemV22()

    print("\n" + "=" * 60)
    print("💬 ГОТОВ К ДИАЛОГУ")
    print("=" * 60)
    print("Команды:")
    print("  • 'exit' — выход")
    print("  • 'если А то Б' — обучение причинности")
    print("  • 'запомни Х' — система запомнит концепт")
    print("=" * 60 + "\n")

    while True:
        try:
            q = input("Ваш вопрос: ").strip()
            if q.lower() in ("exit", "выход", "quit", "q"):
                print("\n👋 Система деактивирована. До встречи!")
                break
            if not q:
                continue

            print()
            answer = system.process(q)

            # 🔥 ГАРАНТИРОВАННЫЙ ВЫВОД ОТВЕТА (решение проблемы "невидимых ответов")
            if answer.strip():
                print("\n💬 Ответ:")
                print_typing(answer, delay=0.015)
            else:
                print_typing("🤔 Я получил ваш вопрос, но не сформировал ответ. Попробуйте уточнить.", delay=0.015)

            print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем.")
            break
        except Exception as e:
            print(f"\n❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()