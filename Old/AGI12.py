# coding: utf-8
"""
AGI_Hybrid_v22_fixed.py
ADVANCED AUTONOMOUS AGI HYBRID SYSTEM — FIXED
"""

import re, json, requests, time, os, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone

# ================= CONFIG =================
class Config:
    ROOT = Path("./cognitive_v22")
    ROOT.mkdir(exist_ok=True)
    OBJECTS   = ROOT / "objects.json"
    CAUSAL    = ROOT / "causal.json"
    META      = ROOT / "meta.json"
    EPISODE   = ROOT / "episodes.json"
    LOG       = ROOT / "log.txt"                 # <- внутренний монолог
    TIMEOUT   = 20
    MAX_CHAIN = 8
    MIN_CONF  = 0.15
    FORGET_RATE = 0.01
    MEMORY_LIMIT = 150
    WORKING_SIZE = 30
    QWEN_API  = "http://localhost:1234/v1/chat/completions"

# ================= UTILS =================
def clean(text: str) -> str:
    """Стилизуем текст – убираем знаки, ставим в нижний регистр."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def extract(text: str) -> List[str]:
    stop = {"что","как","почему","если","то","это","я","ты","мы","они","он","она","оно"}
    return [w for w in clean(text).split() if len(w)>3 and w not in stop]

# ================= CONCEPTS =================
@dataclass
class Concept:
    name: str
    confidence: float = 0.2
    effects: Dict[str,float] = field(default_factory=dict)
    abstract: bool = False
    freq: int = 0

    def reinforce(self, k=0.1):
        self.confidence = min(1.0, self.confidence + k)
        self.freq += 1

    def decay(self):
        self.confidence *= (1 - Config.FORGET_RATE)
        self.freq = max(0, self.freq-1)

# ================= EPISODES =================
@dataclass
class Episode:
    time: str
    input: str
    focus: Optional[str]
    result: str

# ================= WORKING MEMORY =================
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

# ================= SEMANTIC MEMORY =================
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
        c.effects[b] = min(1.0, c.effects.get(b,0.1)+0.3)
        c.reinforce()

    def decay_all(self):
        for c in self.data.values():
            c.decay()

    def generate_abstracts(self):
        names = list(self.data.keys())
        for i, c1 in enumerate(names):
            for c2 in names[i+1:]:
                common = set(self.data[c1].effects) & set(self.data[c2].effects)
                if len(common)/max(len(self.data[c1].effects),1) > 0.5:
                    abs_name = f"{c1}_{c2}_meta"
                    self.get(abs_name).abstract = True
                    self.link(c1,abs_name)
                    self.link(c2,abs_name)

    def save(self):
        with open(Config.OBJECTS,"w",encoding="utf-8") as f:
            json.dump({k:v.__dict__ for k,v in self.data.items()}, f, ensure_ascii=False, indent=2)

    def load(self):
        if Config.OBJECTS.exists():
            with open(Config.OBJECTS,"r",encoding="utf-8") as f:
                for k,v in json.load(f).items():
                    self.data[k] = Concept(**v)

# ================= CAUSAL MEMORY =================
class CausalMemory:
    def __init__(self):
        self.graph: Dict[str,Dict[str,float]] = {}
        self.load()

    def add(self,a:str,b:str):
        self.graph.setdefault(a,{})
        self.graph[a][b] = min(1.0,self.graph[a].get(b,0.1)+0.25)

    def chain(self,start:str) -> List[str]:
        chain = [start]; cur=start
        for _ in range(Config.MAX_CHAIN):
            if cur not in self.graph or not self.graph[cur]: break
            nxt=max(self.graph[cur], key=self.graph[cur].get)
            if nxt in chain: break
            chain.append(nxt)
            cur = nxt
        return chain

    def predict(self,start:str, steps=3) -> List[str]:
        result=[]
        cur=start
        for _ in range(steps):
            if cur not in self.graph or not self.graph[cur]: break
            nxt=max(self.graph[cur], key=self.graph[cur].get)
            result.append(nxt)
            cur=nxt
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
        with open(Config.CAUSAL,"w",encoding="utf-8") as f:
            json.dump(self.graph,f,ensure_ascii=False,indent=2)

    def load(self):
        if Config.CAUSAL.exists():
            with open(Config.CAUSAL,"r",encoding="utf-8") as f:
                self.graph = json.load(f)

# ================= SELF MODEL =================
class SelfModel:
    """Самопереживание. Отвечает только тогда, когда есть смысл."""
    def describe(self, stats: Dict) -> str:
        return (f"Я когнитивная система v22.\n"
                f"Взаимодействий: {stats['interactions']}\n"
                f"Изученных связей: {stats['links']}\n"
                f"Я обучаюсь, прогнозирую последствия и могу уточнять детали.")

    def reflect(self, semantic:SemanticMemory, query:str)->str:
        concepts = extract(query)
        if not concepts: return ""
        confidences = [semantic.get(c).confidence for c in concepts]
        known_words = {"привет","здравствуй","работа","ты","я","система","да","нет"}
        for c in concepts:
            if c in known_words:
                semantic.get(c).confidence = max(semantic.get(c).confidence, 0.5)
        if confidences and min(confidences) < 0.2:
            return "Я не уверен в этом, могу уточнить у вас?"
        return ""

# ================= COGNITIVE SYSTEM =================
class CognitiveSystemV22:
    def __init__(self):
        print("🧠 Cognitive System v22 — AUTONOMOUS AGI")
        self.semantic = SemanticMemory()
        self.causal   = CausalMemory()
        self.working  = WorkingMemory()
        self.self_model = SelfModel()
        self.episodes: List[Episode] = []
        self.meta = self.load_meta()
        # открываем лог
        self.log_fd = open(Config.LOG, "a", encoding="utf-8")
        self.internal_log("Инициализация завершена.")

    def internal_log(self, msg: str):
        """Внутренний монолог – сохраняем мысли."""
        ts = datetime.now(timezone.utc).isoformat()
        self.log_fd.write(f"[{ts}] {msg}\n")
        self.log_fd.flush()

    # ------------------------------------------------------------------
    def process(self,text:str)->str:
        """Главный цикл: от входа к ответу и обновлению памяти."""
        self.meta["interactions"]+=1
        self.working.add(text)
        words = extract(text)
        focus = words[0] if words else None

        # 1️⃣ Кausal learning (если-то)
        if "если" in text and "то" in text:
            parts=clean(text).split("то",1)
            c=extract(parts[0]); e=extract(parts[1])
            if c and e:
                self.causal.add(c[-1],e[0])
                self.semantic.link(c[-1],e[0])
                self.meta["links"]+=1
                self.save()
                return f"Усвоена причинность: {c[-1]} → {e[0]}"

        # 2️⃣ Кausal chain (если есть фокус)
        if focus:
            chain=self.causal.chain(focus)
            if len(chain)>1:
                return "Причинная цепь: " + " → ".join(chain)

        # 3️⃣ Метакогнитивный отклик
        reflect = self.self_model.reflect(self.semantic, text)
        if reflect and len(words)>1:
            return reflect

        # 4️⃣ Генерация ответа (здесь можно вставить собственную логику)
        answer=self.generate_symbolic_answer(text)

        # 5️⃣ Предсказание
        if focus:
            prediction = self.causal.predict(focus)
            if prediction:
                answer += " Возможные последствия: " + " → ".join(prediction) + "."

        # 6️⃣ Память: decay, prune, abstract
        self.semantic.decay_all()
        self.causal.prune()
        self.semantic.generate_abstracts()

        # 7️⃣ Логирование эпизода
        now = datetime.now(timezone.utc).isoformat()
        self.episodes.append(Episode(now, text, focus, answer))
        self.episodes=self.episodes[-Config.MEMORY_LIMIT:]
        self.save()
        return answer

    # ------------------------------------------------------------------
    def learn_from_qwen(self,q:str)->str:
        """Получаем ответ от внешнего LLM и интегрируем его в память."""
        try:
            r=requests.post(Config.QWEN_API,json={
                "messages":[{"role":"user","content":q}],
                "temperature":0.35,"max_tokens":150},timeout=Config.TIMEOUT)
            r.raise_for_status()
            data = r.json()
            content=""
            if "choices" in data and len(data["choices"])>0:
                message = data["choices"][0].get("message",{})
                content = message.get("content","")
            txt = clean(content)
            if not txt:
                txt = "Я пока не знаю ответа на это."
            w=extract(txt)
            for i in range(len(w)-1):
                self.semantic.link(w[i],w[i+1])
                self.causal.add(w[i],w[i+1])
                self.meta["links"]+=1
            return txt
        except Exception as e:
            self.internal_log(f"Ошибка при запросе к Qwen: {e}")
            return "Я пока не могу получить ответ от Qwen, но могу попробовать своими знаниями."

    # ------------------------------------------------------------------
    def generate_symbolic_answer(self, query:str) -> str:
        """Отвечаем «плавно» – по буквам с паузой."""
        txt = self.learn_from_qwen(query)
        output = ""
        for c in txt:
            output += c
            time.sleep(0.001)
        if len(extract(txt)) < 3:
            output += " Могу уточнить у вас детали?"
        return output

    # ------------------------------------------------------------------
    def save(self):
        self.semantic.save()
        self.causal.save()
        with open(Config.EPISODE,"w",encoding="utf-8") as f:
            json.dump([e.__dict__ for e in self.episodes],f,ensure_ascii=False,indent=2)
        with open(Config.META,"w",encoding="utf-8") as f:
            json.dump(self.meta,f,ensure_ascii=False,indent=2)

    def load_meta(self)->Dict:
        if Config.META.exists():
            with open(Config.META,"r",encoding="utf-8") as f:
                return json.load(f)
        return {"interactions":0,"links":0}

# ================= CLI =================
def main():
    system=CognitiveSystemV22()
    print("\nВведите текст (exit для выхода)\n")
    while True:
        try:
            q=input("> ")
            if q.lower() in ("exit","выход"): break
            print("💡",system.process(q))
        except KeyboardInterrupt:
            print("\nВыход.")
            break
        except Exception as e:
            print("Ошибка:", e)

if __name__=="__main__":
    main()
