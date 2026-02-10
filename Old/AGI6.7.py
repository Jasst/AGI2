# coding: utf-8
"""
AGI6.7_enhanced_thought_engine_FIXED.py
ИСПРАВЛЕННАЯ АРХИТЕКТУРА МЫШЛЕНИЯ С РЕФЛЕКСИЕЙ
"""
import os
import re
import random
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST_MODEL = True
except Exception:
    _HAS_ST_MODEL = False


# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

def clean_qwen_response(text: str) -> str:
    if not isinstance(text, str):
        return "Хорошо."
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    text = re.sub(r'>\s*', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'^[\*\.\!\?\:\-\–—\s]+', '', text)
    text = re.sub(r'[\*\.\!\?\:\-\–—\s]+$', '', text)
    words = text.split()
    if len(words) > 60:
        text = ' '.join(words[:60])
        if not text.endswith(('.', '!', '?')):
            text += '.'
    return text or "Хорошо."


def safe_cell_name(base: str) -> str:
    name = re.sub(r'[^a-zA-Zа-яА-Я0-9_]', '_', base)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else "unknown"


def clean_for_similarity(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


# ======================
# ТИПЫ МЫШЛЕНИЯ И ТЕГИ
# ======================

TAGS = {"[SOC]", "[FCT]", "[CAU]", "[PRC]", "[OPN]", "[MET]", "[CRT]"}
TAG_PATTERN = re.compile(r'\[(SOC|FCT|CAU|PRC|OPN|MET|CRT)\]')


def classify_and_tag_response(text: str) -> str:
    if not text.strip():
        return "[SOC] Хорошо."
    text = clean_qwen_response(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    tagged = []
    for sent in sentences:
        if sent.strip():
            tag = _detect_sentence_type(sent)
            tagged.append(f"[{tag}] {sent}")
    return " ".join(tagged)


def _detect_sentence_type(sentence: str) -> str:
    s = sentence.lower().strip()
    if re.search(r'\b(что такое|почему|как работает|объясни|значит ли|противоречие|размышля|думаешь|что значит)\b', s):
        return "MET"
    if re.search(r'\b(чтобы|нужно|следует|шаг|сначала|потом|затем|инструкция|алгоритм|как приготовить|как сделать)\b',
                 s):
        return "PRC"
    if re.search(r'\b(потому что|так как|из-за|следствие|причина|происходит из-за|ведёт к|обусловлено)\b', s):
        return "CAU"
    if re.search(r'.+ — это .+|.+ называется .+|.+ состоит из .+|столица .+ — .+|формула .+ — .+', s):
        return "FCT"
    if re.search(r'\b(я думаю|мне кажется|по моему мнению|я считаю|мне нравится|скучный|отличный|лучше|хуже)\b', s):
        return "OPN"
    if re.search(r'\b(представь|вообрази|как будто|словно|подобно|жизнь —|мир как|если бы|фантазия)\b', s):
        return "CRT"
    social_keywords = ["привет", "здравствуй", "добрый", "спасибо", "пожалуйста", "извини", "хорошо", "ладно", "ок",
                       "пока"]
    if any(kw in s for kw in social_keywords) or len(s.split()) <= 3:
        return "SOC"
    return "FCT"


def detect_input_type(user_input: str) -> str:
    s = user_input.lower().strip()
    if re.search(r'\b(привет|здравствуй|добрый день|как дела|пока)\b', s):
        return "SOC"
    if re.search(r'\b(что такое|кто такой|где находится|какая столица|формула|определение)\b', s):
        return "FCT"
    if re.search(r'\b(почему|зачем|отчего|причина)\b', s):
        return "CAU"
    if re.search(r'\b(как сделать|как приготовить|инструкция|шаг|алгоритм)\b', s):
        return "PRC"
    if re.search(r'\b(как ты думаешь|твоё мнение|лучше ли|нравится ли)\b', s):
        return "OPN"
    if re.search(r'\b(представь|вообрази|сочини|опиши как|метафора)\b', s):
        return "CRT"
    if re.search(r'\b(почему ты|как ты понял|что ты имел в виду|объясни свой ответ)\b', s):
        return "MET"
    return "FCT"


INPUT_TYPE_TO_STAGES = {
    "SOC": ["social"],
    "FCT": ["fact", "meta"],
    "CAU": ["cause", "fact", "meta"],
    "PRC": ["procedure", "fact"],
    "OPN": ["opinion", "meta"],
    "CRT": ["creative", "metaphor", "meta"],
    "MET": ["meta", "fact"]
}


def get_allowed_stages(input_type: str) -> List[str]:
    return INPUT_TYPE_TO_STAGES.get(input_type, ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"])


# ======================
# УЛУЧШЕННОЕ ГЛОБАЛЬНОЕ СОСТОЯНИЕ МЫШЛЕНИЯ
# ======================

class EnhancedGlobalThoughtState:
    """Улучшенное состояние с рабочей памятью и осознанностью."""

    def __init__(self, batch_size: int, hidden_size: int, device: torch.device):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.context_vector = torch.zeros(batch_size, hidden_size, device=device)
        self.active_stages: List[str] = []
        self.stage_history: List[str] = []
        self.confidence = 1.0

        # Рабочие памяти для разных типов мышления
        self.working_memory = {
            "facts": [],
            "goals": [],
            "hypotheses": [],
            "constraints": [],
            "insights": []
        }
        self.attention_weights = torch.ones(batch_size, hidden_size, device=device)
        self.consciousness_level = 0.7
        self.thinking_depth = 0

    def update_working_memory(self, category: str, content: Any, importance: float = 1.0):
        """Обновляем рабочую память с учетом важности"""
        self.working_memory[category].append((content, importance, self.thinking_depth))
        # Ограничиваем размер рабочей памяти
        if len(self.working_memory[category]) > 10:
            self.working_memory[category].pop(0)

    def get_relevant_context(self, current_focus: str) -> torch.Tensor:
        """Получаем релевантный контекст для текущего фокуса"""
        relevant_items = []
        for category, items in self.working_memory.items():
            for content, importance, depth in items:
                if self.thinking_depth - depth <= 3:  # Только недавние мысли
                    relevant_items.append((content, importance))

        if relevant_items:
            # Усиливаем контекст на основе релевантности
            avg_importance = sum(imp for _, imp in relevant_items) / len(relevant_items)
            return self.context_vector * (1 + avg_importance * 0.3)
        return self.context_vector

    def update(self, new_info: torch.Tensor, stage: str):
        self.context_vector = 0.8 * self.context_vector + 0.2 * new_info
        if stage not in self.active_stages:
            self.active_stages.append(stage)
        self.stage_history.append(stage)
        self.thinking_depth += 1

    def reset(self):
        self.context_vector.zero_()
        self.active_stages.clear()
        self.stage_history.clear()
        self.confidence = 1.0
        self.thinking_depth = 0
        for key in self.working_memory:
            self.working_memory[key].clear()


# ======================
# РЕФЛЕКСИВНЫЕ КОГНИТИВНЫЕ КЛЕТКИ (ИСПРАВЛЕННЫЕ)
# ======================

class ReflectiveBrainCell(nn.Module):
    def __init__(self, cell_id: int, input_size: int, hidden_size: int, cell_type: str = "generic"):
        super().__init__()
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_adapter = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

        # Механизм внутреннего диалога - ИСПРАВЛЕННЫЙ РАЗМЕР
        self.internal_dialogue = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),  # Исправленный размер
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # Механизм самоконтроля
        self.self_monitoring = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 3)  # [confidence, novelty, importance]
        )

        self.perception = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.memory = nn.LSTMCell(hidden_size, hidden_size)
        self.association = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.activation_level = 0.0

        # Кэш предыдущих размышлений - ИСПРАВЛЕННЫЙ
        self.reflection_cache: List[torch.Tensor] = []
        self.cache_size = 5

        # Механизм предложения следующего этапа
        self.stage_transition = {
            "social": ["social", "fact"],
            "fact": ["cause", "meta", "procedure"],
            "cause": ["fact", "meta", "procedure"],
            "procedure": ["fact", "cause", "meta"],
            "opinion": ["meta", "fact", "cause"],
            "meta": ["fact", "opinion", "cause", "creative"],
            "creative": ["metaphor", "meta", "opinion"],
            "metaphor": ["creative", "meta"],
            "generic": ["fact", "meta"]
        }

    def propose_next_stage(self, current_stage: str) -> str:
        candidates = self.stage_transition.get(current_stage, ["fact", "meta"])
        return random.choice(candidates)

    def reset_cache(self):
        """Сброс кэша размышлений"""
        self.reflection_cache.clear()

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None, cx: Optional[torch.Tensor] = None,
                global_state: Optional[EnhancedGlobalThoughtState] = None):
        batch_size = x.size(0)
        device = x.device

        # Базовая обработка
        x_adapted = self.input_adapter(x)

        # Внутренний диалог с предыдущими состояниями - ИСПРАВЛЕННЫЙ
        if self.reflection_cache:
            # Безопасное усреднение предыдущих состояний
            try:
                # Берем последние состояния и убеждаемся в совместимости размеров
                recent_states = self.reflection_cache[-min(3, len(self.reflection_cache)):]
                stacked_states = torch.stack(recent_states)
                prev_context = torch.mean(stacked_states, dim=0)

                # Проверка совместимости размеров
                if x_adapted.shape == prev_context.shape:
                    internal_input = torch.cat([x_adapted, prev_context], dim=-1)
                    internal_reflection = self.internal_dialogue(internal_input)
                    x_enhanced = x_adapted + 0.3 * internal_reflection
                else:
                    # Если размеры не совпадают, используем только текущий вход
                    x_enhanced = x_adapted
            except Exception as e:
                # В случае ошибки используем базовый вход
                print(f"⚠️ Ошибка внутреннего диалога: {e}")
                x_enhanced = x_adapted
        else:
            x_enhanced = x_adapted

        # Самоконтроль
        self_assessment = self.self_monitoring(x_enhanced)
        confidence, novelty, importance = torch.sigmoid(self_assessment).unbind(dim=-1)

        # Обновляем кэш размышлений - ИСПРАВЛЕННЫЙ
        try:
            if len(self.reflection_cache) >= self.cache_size:
                self.reflection_cache.pop(0)
            # Сохраняем клон с отключенным градиентом
            self.reflection_cache.append(x_enhanced.detach().clone())
        except Exception as e:
            print(f"⚠️ Ошибка обновления кэша: {e}")

        # Интеграция с глобальным состоянием
        if global_state is not None:
            try:
                contextual_input = global_state.get_relevant_context(self.cell_type)
                # Проверка совместимости размеров
                if x_enhanced.shape == contextual_input.shape:
                    x_enhanced = x_enhanced + 0.2 * contextual_input
            except Exception as e:
                print(f"⚠️ Ошибка интеграции с глобальным состоянием: {e}")

        # Основная обработка с улучшенным входом
        perceived = self.perception(x_enhanced)

        # Сохраняем инсайты в рабочую память
        if global_state is not None and torch.mean(confidence) > 0.3:
            try:
                insight_content = f"{self.cell_type}:{self.cell_id}:{torch.mean(confidence).item():.3f}"
                global_state.update_working_memory("insights", insight_content, torch.mean(confidence).item())
            except Exception:
                pass

        self.activation_level = float(torch.mean(confidence * novelty * importance).detach().cpu().item())

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=device)
        if cx is None:
            cx = torch.zeros(batch_size, self.hidden_size, device=device)

        hx, cx = self.memory(perceived, (hx, cx))
        associated = self.association(hx)

        # Динамический гейтинг на основе уверенности
        dynamic_gate = torch.sigmoid(self.gate_logit + confidence.mean())
        output = associated * dynamic_gate

        return output, hx, cx, self.activation_level


# ======================
# РАСШИРЕННЫЙ ПЛАНИРОВЩИК МЫШЛЕНИЯ
# ======================

class EnhancedThoughtPlanner:
    """Динамически строит стратегические последовательности этапов мышления."""

    def __init__(self, allowed_stages: List[str], hidden_size: int):
        self.allowed_stages = allowed_stages[:]
        self.planned_stages = allowed_stages[:]
        self.iteration = 0
        self.hidden_size = hidden_size
        self.stage_relationships = {
            "fact": ["cause", "meta", "procedure"],
            "cause": ["fact", "meta", "procedure"],
            "procedure": ["fact", "cause", "meta"],
            "opinion": ["meta", "fact", "cause"],
            "meta": ["fact", "opinion", "cause", "creative"],
            "creative": ["metaphor", "meta", "opinion"],
            "social": ["fact", "opinion"],
            "metaphor": ["creative", "meta"]
        }
        self.conversation_history = []

    def plan_thought_strategy(self, user_input: str, context: torch.Tensor) -> List[str]:
        """Планирует стратегию мышления на основе контекста"""
        input_type = detect_input_type(user_input)
        base_stages = get_allowed_stages(input_type)

        # Анализ сложности запроса
        complexity = self._assess_complexity(user_input, context)

        # Планирование на основе сложности
        if complexity < 0.3:
            # Простые запросы - короткая цепочка
            return base_stages[:2]
        elif complexity < 0.6:
            # Средняя сложность
            stages = base_stages.copy()
            if "meta" not in stages:
                stages.append("meta")
            return stages
        else:
            # Сложные запросы - полный цикл мышления
            stages = base_stages.copy()
            # Добавляем рефлексивные этапы для сложных вопросов
            if "meta" not in stages:
                stages.insert(1, "meta")
            if "cause" not in stages and input_type != "SOC":
                stages.append("cause")
            return stages

    def plan_next_cycle(self, active_cells: Dict[str, float], current_stages: List[str],
                        user_input: str, context: torch.Tensor) -> List[str]:
        self.iteration += 1
        if self.iteration == 1:
            return self.plan_thought_strategy(user_input, context)

        # Стратегическое планирование на основе активации клеток
        stage_activation = defaultdict(float)
        for cell_id, act in active_cells.items():
            if act < 0.15:
                continue
            cell_type = cell_id.split('_')[0]
            if cell_type in self.stage_relationships:
                stage_activation[cell_type] += act

        # Усиливаем наиболее активные стадии
        highly_active = [stage for stage, act in stage_activation.items() if act > 0.3]
        new_plan = self.planned_stages.copy()

        for stage in highly_active:
            related = self.stage_relationships.get(stage, [])
            for rel in related:
                if rel not in new_plan and rel in self.allowed_stages:
                    new_plan.append(rel)

        self.planned_stages = list(dict.fromkeys(new_plan))
        return self.planned_stages[:6]  # Ограничиваем длину цепочки

    def _assess_complexity(self, text: str, context: torch.Tensor) -> float:
        """Оценивает сложность запроса"""
        words = len(text.split())
        question_marks = text.count('?')
        depth_indicators = ['почему', 'как', 'объясни', 'сравни', 'проанализируй']

        lexical_complexity = min(words / 50, 1.0)
        structural_complexity = min(question_marks * 0.3, 1.0)
        semantic_complexity = sum(1 for word in depth_indicators if word in text.lower()) * 0.2

        context_variance = torch.var(context).item() if context.numel() > 1 else 0
        context_complexity = min(context_variance * 10, 1.0)

        return (lexical_complexity + structural_complexity +
                semantic_complexity + context_complexity) / 4


# ======================
# МЕХАНИЗМ РЕФЛЕКСИВНОГО ОБУЧЕНИЯ
# ======================

class ReflectiveLearner:
    def __init__(self, network: 'EnhancedCognitiveNetwork'):
        self.network = network
        self.learning_episodes = []
        self.insights = []

    def record_learning_episode(self, user_input: str, response: str,
                                success_metrics: Dict[str, float]):
        episode = {
            'input': user_input,
            'response': response,
            'metrics': success_metrics,
            'timestamp': datetime.now(),
            'activated_cells': self.network.find_activated_cells_by_type()
        }
        self.learning_episodes.append(episode)

    def extract_insights(self):
        """Извлекает инсайты из записанных эпизодов"""
        if len(self.learning_episodes) < 3:
            return

        recent_episodes = self.learning_episodes[-10:]

        # Анализ успешных паттернов мышления
        successful_patterns = []
        for episode in recent_episodes:
            if episode['metrics'].get('confidence', 0) > 0.7:
                successful_patterns.append(episode['activated_cells'])

        if successful_patterns:
            # Находим наиболее частые паттерны
            cell_usage = Counter()
            for pattern in successful_patterns:
                for cell_type, cells in pattern.items():
                    cell_usage[cell_type] += len(cells)

            # Создаем новые клетки на основе успешных паттернов
            for cell_type, count in cell_usage.most_common(2):
                if count >= 3:  # Паттерн встречается достаточно часто
                    self._create_specialized_cell(cell_type)

    def _create_specialized_cell(self, cell_type: str):
        """Создает специализированную клетку на основе успешного паттерна"""
        new_cell_id = self.network.add_cell(
            f"reflective_{cell_type}",
            self.network.hidden_size,
            self.network.hidden_size,
            cell_type
        )
        print(f"🧠 Создана рефлексивная клетка: {new_cell_id}")


# ======================
# PRIORITIZED REPLAY
# ======================

class PrioritizedReplay:
    def __init__(self, capacity: int = 5000, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer: List[Dict[str, Any]] = []
        self.priorities: List[float] = []
        self.position = 0

    def add(self, inp: List[int], target: List[int], meta: Optional[Dict] = None, priority: Optional[float] = None):
        data = {"input": inp, "target": target, "meta": meta or {}, "len": len(inp)}
        p = max(self.priorities) if self.priorities else 1.0 if priority is None else priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(p)
        else:
            self.buffer[self.position] = data
            self.priorities[self.position] = p
            self.position = (self.position + 1) % self.capacity

    def _get_probabilities(self):
        scaled = np.array(self.priorities) ** self.alpha
        if scaled.sum() == 0:
            scaled = np.ones_like(scaled)
        probs = scaled / scaled.sum()
        return probs

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.buffer)
        batch_size = min(batch_size, n)
        if n == 0:
            return [], [], []
        probs = self._get_probabilities()
        idxs = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in idxs]
        weights = (n * probs[idxs]) ** (-beta)
        weights = weights / weights.max()
        return samples, idxs.tolist(), weights.tolist()

    def update_priorities(self, idxs: List[int], priorities: List[float]):
        for i, p in zip(idxs, priorities):
            self.priorities[i] = max(p + self.eps, self.eps)

    def __len__(self):
        return len(self.buffer)

    def serialize(self):
        return {"buffer": self.buffer, "priorities": self.priorities, "position": self.position}

    def load(self, data: Dict[str, Any]):
        try:
            self.buffer = data.get("buffer", [])
            self.priorities = data.get("priorities", [1.0] * len(self.buffer))
            self.position = data.get("position", 0)
        except Exception:
            pass


# ======================
# META-COGNITION
# ======================

class MetaCognition:
    def __init__(self, vocab: Dict[str, int], network: Optional['EnhancedCognitiveNetwork'] = None,
                 external_model_name: str = "all-MiniLM-L6-v2"):
        self.vocab = vocab
        self.unknown_concepts: Set[str] = set()
        self.reflection_log: List[Dict[str, Any]] = []
        self.network = network
        self.external_model = None
        if _HAS_ST_MODEL:
            try:
                self.external_model = SentenceTransformer(external_model_name)
            except Exception:
                self.external_model = None

    def detect_unknown_words(self, text: str) -> Set[str]:
        words = set(clean_for_similarity(text).split())
        known = set(k for k in self.vocab.keys() if clean_for_similarity(k))
        return words - known

    def log_reflection(self, user_input: str, qwen_resp: str, brain_resp: str, question: Optional[str], reason: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "qwen_response": qwen_resp,
            "brain_response": brain_resp,
            "generated_question": question,
            "reason": reason
        }
        self.reflection_log.append(entry)

    def _embed_text_external(self, text: str) -> np.ndarray:
        if self.external_model is not None:
            try:
                return self.external_model.encode([text], normalize_embeddings=True)[0]
            except Exception:
                pass
        if self.network is not None:
            tokens = self.network.text_to_tokens(text, self.vocab, include_tags=True)
            if tokens:
                emb = self.network.embedding(torch.tensor(tokens, dtype=torch.long, device=self.network.device))
                emb = emb.detach().cpu().numpy()
                vec = emb.mean(axis=0)
                norm = np.linalg.norm(vec)
                return vec / (norm + 1e-8)
        toks = clean_for_similarity(text).split()
        vec = np.zeros(128, dtype=float)
        for i, w in enumerate(toks):
            vec[i % 128] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _estimate_similarity_semantic(self, resp1: str, resp2: str) -> float:
        v1 = self._embed_text_external(resp1)
        v2 = self._embed_text_external(resp2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def should_reflect(self, user_input: str, qwen_resp: str, brain_resp: str, brain_confidence: float = 1.0) -> Tuple[
        bool, str]:
        expected_type = detect_input_type(user_input)
        actual_type = "FCT"
        marker_match = TAG_PATTERN.search(qwen_resp)
        if marker_match:
            actual_type = marker_match.group(1)
        type_mismatch = False
        reason = ""
        if expected_type == "FCT" and actual_type in ["SOC", "OPN"]:
            type_mismatch = True
            reason = f"ожидался факт [FCT], но Qwen дал {actual_type}"
        elif expected_type == "PRC" and actual_type not in ["PRC", "FCT"]:
            type_mismatch = True
            reason = f"ожидалась инструкция [PRC], но Qwen дал {actual_type}"
        elif expected_type == "CRT" and actual_type in ["SOC", "FCT"]:
            type_mismatch = True
            reason = f"ождалось творчество [CRT], но Qwen дал {actual_type}"
        similarity = self._estimate_similarity_semantic(brain_resp, qwen_resp)
        if type_mismatch or similarity < 0.35 or brain_confidence < 0.6:
            return True, reason if type_mismatch else f"низкое сходство ({similarity:.2f}) или низкая уверенность ({brain_confidence:.2f})"
        return False, ""


# ======================
# УЛУЧШЕННАЯ КОГНИТИВНАЯ СЕТЬ (ИСПРАВЛЕННАЯ)
# ======================

class EnhancedCognitiveNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_size: int = 512, eos_token_id: int = 1,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id

        # Основные компоненты сети
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_proj = nn.Linear(embedding_dim, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=max(1, min(8, hidden_size // 64)),
                                               batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, vocab_size)

        # Улучшенные компоненты мышления
        self.cells = nn.ModuleDict()
        self.cell_states: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        self.cell_activations: Dict[str, float] = {}
        self.cell_counter = 0

        # Механизмы рефлексии и обучения
        self.meta_cog: Optional[MetaCognition] = None
        self.reflective_learner: Optional[ReflectiveLearner] = None
        self.replay = PrioritizedReplay(capacity=5000, alpha=0.6)

        self.to(self.device)
        self.global_thought_state: Optional[EnhancedGlobalThoughtState] = None

        # Инициализация улучшенных клеток
        self._initialize_enhanced_cells()

    def _initialize_enhanced_cells(self):
        """Инициализирует улучшенные рефлексивные клетки"""
        base_cells = [
            ("social_greeting", "social"),
            ("social_thanks", "social"),
            ("fact_definition", "fact"),
            ("cause_explanation", "cause"),
            ("procedure_step", "procedure"),
            ("opinion_expression", "opinion"),
            ("meta_question", "meta"),
            ("creative_metaphor", "creative"),
            ("metaphor_imagery", "metaphor"),
            ("logic_reasoning", "logic"),
            ("context_analysis", "context")
        ]

        for name, ctype in base_cells:
            cell_id = f"{name}_{self.cell_counter}"
            self.cells[cell_id] = ReflectiveBrainCell(
                self.cell_counter, self.hidden_size, self.hidden_size, ctype
            )
            self.cells[cell_id].to(self.device)
            self.cell_states[cell_id] = (None, None)
            self.cell_activations[cell_id] = 0.0
            self.cell_counter += 1

        # Инициализируем рефлексивного ученика
        self.reflective_learner = ReflectiveLearner(self)

    def add_cell(self, cell_type: str, input_size: int, hidden_size: int, cell_subtype: str = "association") -> str:
        safe_type = safe_cell_name(cell_type)
        cell_id = f"{safe_type}_{self.cell_counter}"
        self.cells[cell_id] = ReflectiveBrainCell(self.cell_counter, input_size, hidden_size, cell_subtype)
        self.cells[cell_id].to(self.device)
        self.cell_states[cell_id] = (None, None)
        self.cell_activations[cell_id] = 0.0
        self.cell_counter += 1
        return cell_id

    def reset_cell_states(self, batch_size: int, device: Optional[torch.device] = None):
        device = device or self.device
        for cell_id in self.cells:
            self.cell_states[cell_id] = (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device)
            )

    def reset_cell_caches(self):
        """Сброс кэшей всех клеток"""
        for cell in self.cells.values():
            if hasattr(cell, 'reset_cache'):
                cell.reset_cache()

    def compute_weighted_loss(self, logits: torch.Tensor, targets: torch.Tensor, vocab: Dict[str, int],
                              ivocab: Dict[int, str]) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        losses = criterion(logits.view(-1, self.vocab_size), targets.view(-1))
        losses = losses.view(targets.size(0), targets.size(1))
        weights = torch.ones_like(targets, dtype=torch.float32, device=self.device)
        tag_ids = {vocab.get(tag, -1) for tag in TAGS if tag in vocab}
        for b in range(targets.size(0)):
            for t in range(targets.size(1)):
                tid = targets[b, t].item()
                if tid in tag_ids:
                    weights[b, t] = 3.0
                elif ivocab.get(tid, "").startswith(("почему", "как", "что", "зачем", "объясни")):
                    weights[b, t] = 1.8
        mask = (targets != 0).float()
        weighted_loss = losses * weights * mask
        return weighted_loss.sum() / (mask.sum() + 1e-8)

    def _enhanced_thought_cycles(self, token_emb: torch.Tensor, allowed_stages: List[str],
                                 user_input: str, max_cycles: int = 7):
        batch_size = token_emb.size(0)
        x = token_emb

        # Инициализация улучшенного глобального состояния
        self.global_thought_state = EnhancedGlobalThoughtState(batch_size, self.hidden_size, self.device)

        # Планирование стратегии мышления
        planner = EnhancedThoughtPlanner(allowed_stages, self.hidden_size)
        thought_plan = planner.plan_thought_strategy(user_input, x)

        print(f"🧠 Стратегия мышления: {thought_plan}")

        for cycle in range(max_cycles):
            self.global_thought_state.thinking_depth = cycle

            # Динамическое обновление плана на основе активации клеток
            if cycle % 2 == 0 and cycle > 0:
                thought_plan = planner.plan_next_cycle(
                    self.cell_activations,
                    self.global_thought_state.active_stages,
                    user_input,
                    x
                )
                print(f"🧠 Обновленный план (цикл {cycle}): {thought_plan}")

            stage_outputs = []
            stage_insights = []

            for stage in thought_plan:
                stage_cells = [cid for cid in self.cells if stage in self.cells[cid].cell_type]

                for cid in stage_cells:
                    cell = self.cells[cid]
                    hx, cx = self.cell_states.get(cid, (None, None))

                    # Получаем релевантный контекст для этой стадии
                    contextual_input = self.global_thought_state.get_relevant_context(stage)
                    enhanced_input = x + 0.2 * contextual_input

                    cell_output, new_hx, new_cx, act = cell(
                        enhanced_input, hx, cx, global_state=self.global_thought_state
                    )

                    self.cell_states[cid] = (new_hx, new_cx)
                    self.cell_activations[cid] = act

                    stage_outputs.append(cell_output)
                    stage_insights.append((stage, act))

            if stage_outputs:
                # Взвешенное объединение выходов на основе активации
                weights = torch.tensor([act for _, act in stage_insights],
                                       device=self.device).unsqueeze(1)
                weights_normalized = F.softmax(weights, dim=0)

                stacked_outputs = torch.stack(stage_outputs)
                weighted_output = (stacked_outputs * weights_normalized.unsqueeze(-1)).sum(dim=0)

                # Обновляем глобальное состояние
                dominant_stage = max(stage_insights, key=lambda x: x[1])[0] if stage_insights else "fact"
                self.global_thought_state.update(weighted_output, dominant_stage)

                x = weighted_output

        self.global_thought_state = None
        return self.final_proj(x)

    def process_sequence(self, input_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                         allowed_stages: Optional[List[str]] = None, user_input: str = "") -> torch.Tensor:
        if allowed_stages is None:
            allowed_stages = ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"]
        batch_size, seq_len = input_tokens.shape

        # Сброс состояний и кэшей
        self.reset_cell_states(batch_size, self.device)
        self.reset_cell_caches()

        embedded = self.embedding(input_tokens)
        embedded = self.embed_proj(embedded)
        if attention_mask is None:
            attn_out, _ = self.attention(embedded, embedded, embedded)
        else:
            key_padding_mask = attention_mask == 0
            attn_out, _ = self.attention(embedded, embedded, embedded, key_padding_mask=key_padding_mask)
        combined = self.norm(embedded + attn_out)
        outputs = []
        for t in range(seq_len):
            token_emb = combined[:, t, :]
            logits = self._enhanced_thought_cycles(token_emb, allowed_stages, user_input, max_cycles=5)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    def generate_sequence(self, input_tokens: torch.Tensor, allowed_stages: List[str],
                          max_length: int = 30, temperature: float = 0.8, user_input: str = "") -> List[int]:
        batch_size = input_tokens.size(0)

        # Сброс состояний и кэшей
        self.reset_cell_states(batch_size, self.device)
        self.reset_cell_caches()

        with torch.no_grad():
            embedded = self.embedding(input_tokens)
            embedded = self.embed_proj(embedded)
            attn_out, _ = self.attention(embedded, embedded, embedded, is_causal=True)
            combined = self.norm(embedded + attn_out)
            for t in range(combined.size(1)):
                token_emb = combined[:, t, :]
                _ = self._enhanced_thought_cycles(token_emb, allowed_stages, user_input, max_cycles=4)
            generated = []
            current = input_tokens[:, -1:].clone()
            for _ in range(max_length):
                emb = self.embedding(current).squeeze(1)
                emb = self.embed_proj(emb)
                logits = self._enhanced_thought_cycles(emb, allowed_stages, user_input, max_cycles=4)
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                next_id_val = next_id.item()
                if next_id_val == self.eos_token_id:
                    break
                generated.append(next_id_val)
                current = next_id
            return generated

    def expand_vocab(self, new_vocab_size: int):
        if new_vocab_size <= self.vocab_size:
            return
        old_emb = self.embedding.weight.data.cpu().clone()
        old_vocab = self.vocab_size
        new_embedding = nn.Embedding(new_vocab_size, self.embed_dim, padding_idx=0)
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
        new_embedding.weight.data[:old_vocab, :self.embed_dim] = old_emb
        unk_idx = 2
        if unk_idx < old_vocab:
            unk_vec = old_emb[unk_idx]
            for i in range(old_vocab, new_vocab_size):
                new_embedding.weight.data[i] = unk_vec.clone()
        self.embedding = new_embedding.to(self.device)
        new_final = nn.Linear(self.hidden_size, new_vocab_size)
        nn.init.xavier_uniform_(new_final.weight)
        new_final.bias.data.zero_()
        self.final_proj = new_final.to(self.device)
        self.vocab_size = new_vocab_size

    def _check_cell_creation(self, user_input: str, qwen_resp: str, brain_resp: str):
        if self.meta_cog is None:
            return False
        unknown = self.meta_cog.detect_unknown_words(qwen_resp)
        if unknown:
            for word in list(unknown)[:2]:
                clean_word = safe_cell_name(word)
                self.meta_cog.unknown_concepts.add(word)
                self.add_cell(f"concept_{clean_word}", self.hidden_size, self.hidden_size, "association")
                print(f"🧬 Создана клетка для нового понятия: {word} → concept_{clean_word}")
        return False

    def reflect_and_learn(self, user_input: str, qwen_response: str, vocab: Dict[str, int], ivocab: Dict[int, str]) -> \
    Optional[str]:
        input_tokens = self.text_to_tokens(user_input, vocab, include_tags=False)
        if not input_tokens:
            return None
        allowed_stages = get_allowed_stages(detect_input_type(user_input))
        with torch.no_grad():
            brain_tokens = self.generate_sequence(
                torch.tensor([input_tokens], dtype=torch.long, device=self.device),
                allowed_stages=allowed_stages,
                user_input=user_input
            )
            brain_response = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)
        tokens = brain_response.split()
        if not tokens:
            return None
        unk_ratio = brain_response.count("<UNK>") / len(tokens)
        if unk_ratio > 0.7:
            return "не понял: слишком много неизвестных слов"
        assert self.meta_cog is not None
        should_reflect, reason = self.meta_cog.should_reflect(user_input, qwen_response, brain_response, 1.0)
        if should_reflect:
            question = self._formulate_deep_question(user_input, qwen_response, brain_response, reason, vocab)
            self.meta_cog.log_reflection(user_input, qwen_response, brain_response, question, reason)
            return question
        return None

    def _formulate_deep_question(self, user_input: str, qwen_resp: str, brain_resp: str, reason: str,
                                 vocab: Dict[str, int]) -> str:
        active_cells = [cid for cid, act in self.cell_activations.items() if act > 0.2]
        if active_cells:
            dominant_type = Counter([self.cells[cid].cell_type for cid in active_cells]).most_common(1)[0][0]
            templates = {
                "fact": f"Что именно означает '{user_input}'? Дай точное определение.",
                "cause": f"Почему происходит то, о чём спрашивают в '{user_input}'?",
                "procedure": f"Как пошагово достичь цели из '{user_input}'?",
                "opinion": f"Какие аргументы за и против утверждения в '{user_input}'?",
                "meta": f"Как лучше всего подойти к пониманию '{user_input}'?",
                "creative": f"Опиши '{user_input}' через метафору или образ.",
                "social": f"Как корректно ответить на '{user_input}' в социальном контексте?",
            }
            return templates.get(dominant_type, f"Объясни глубже: {user_input}")
        return f"объясни подробнее: {user_input}"

    def text_to_tokens(self, text: str, vocab: Dict[str, int], include_tags: bool = True) -> List[int]:
        if not include_tags:
            text = TAG_PATTERN.sub('', text)
        text_proc = text.lower()
        text_proc = text_proc.replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('!', ' ! ')
        words = text_proc.split()
        tokens = []
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
            tokens.append(vocab[word])
        if self.vocab_size < len(vocab):
            self.expand_vocab(len(vocab) + 128)
        return tokens

    def find_activated_cells_by_type(self, threshold: float = 0.15) -> Dict[str, List[str]]:
        grouped = {k: [] for k in ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"]}
        for cid, act in self.cell_activations.items():
            if act <= threshold:
                continue
            cell_type = self.cells[cid].cell_type
            for key in grouped:
                if key in cell_type:
                    grouped[key].append(cid)
                    break
        return {k: v for k, v in grouped.items() if v}

    def print_thought_flow(self):
        if not self.cell_activations:
            return
        print("\n🧠 Детальный поток мышления:")
        for stage in ["social", "fact", "cause", "procedure", "opinion", "meta", "creative"]:
            cells = [cid for cid in self.cells if stage in self.cells[cid].cell_type]
            if cells:
                activations = [self.cell_activations.get(cid, 0) for cid in cells]
                avg_act = np.mean(activations)
                max_act = max(activations) if activations else 0
                if avg_act > 0.05:
                    active_count = sum(1 for act in activations if act > 0.1)
                    print(
                        f"  → {stage.upper()}: {avg_act:.3f} (пик: {max_act:.3f}, активных: {active_count}/{len(cells)})")

    def save_knowledge(self, filepath: str, vocab: Dict[str, int]):
        cell_configs = []
        for cell_id, cell in self.cells.items():
            config = {
                'cell_id': cell_id,
                'cell_counter': cell.cell_id,
                'input_size': cell.input_size,
                'hidden_size': cell.hidden_size,
                'cell_type': cell.cell_type,
                'gate_logit': float(cell.gate_logit.detach().cpu().item())
            }
            cell_configs.append(config)
        knowledge = {
            'model_state': self.state_dict(),
            'vocab': vocab,
            'cell_configs': cell_configs,
            'cell_counter': self.cell_counter,
            'meta_cog_log': self.meta_cog.reflection_log if self.meta_cog else [],
            'unknown_concepts': list(self.meta_cog.unknown_concepts) if self.meta_cog else [],
            'replay': self.replay.serialize()
        }
        torch.save(knowledge, filepath)

    def load_knowledge(self, filepath: str):
        if not os.path.exists(filepath):
            return None
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            saved_vocab = checkpoint.get('vocab', {})
            cell_configs = checkpoint.get('cell_configs', [])
            saved_state = checkpoint.get('model_state', {})
            self.cells = nn.ModuleDict()
            self.cell_states = {}
            self.cell_activations = {}
            for config in cell_configs:
                cid = config['cell_id']
                input_size = int(config['input_size'])
                hidden_size = int(config['hidden_size'])
                cell_type = config.get('cell_type', 'generic')
                self.cells[cid] = ReflectiveBrainCell(config.get('cell_counter', 0), input_size, hidden_size, cell_type)
                self.cells[cid].to(self.device)
                gate_val = float(config.get('gate_logit', 0.0))
                with torch.no_grad():
                    self.cells[cid].gate_logit.copy_(torch.tensor(gate_val, device=self.device))
                self.cell_states[cid] = (None, None)
                self.cell_activations[cid] = 0.0
            self.cell_counter = checkpoint.get('cell_counter', len(self.cells))
            self.load_state_dict(saved_state, strict=False)
            self.meta_cog = MetaCognition(saved_vocab, network=self)
            self.meta_cog.reflection_log = checkpoint.get('meta_cog_log', [])
            self.meta_cog.unknown_concepts = set(checkpoint.get('unknown_concepts', []))
            replay_data = checkpoint.get('replay', {})
            try:
                self.replay.load(replay_data)
            except Exception:
                pass
            return saved_vocab
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            traceback.print_exc()
            return None


# ======================
# УЧИТЕЛЬ (ОБНОВЛЁННЫЙ)
# ======================

class BrainTeacher:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions",
                 device: Optional[torch.device] = None):
        self.api_url = api_url
        self.conversation_history: List[Dict[str, Any]] = []
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def query_qwen(self, prompt: str, timeout: int = 20) -> str:
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 120,
                    "stream": False
                },
                timeout=timeout
            )
            if response.status_code == 200:
                raw = response.json()['choices'][0]['message']['content'].strip()
                return clean_qwen_response(raw)
            else:
                return "Хорошо."
        except Exception as e:
            print(f"⚠️ Ошибка LM Studio: {e}")
            return "Хорошо."

    @staticmethod
    def _pad_batch(sequences: List[List[int]], pad_id: int = 0):
        lengths = [len(s) for s in sequences]
        max_len = max(lengths) if lengths else 0
        padded = [s + [pad_id] * (max_len - len(s)) for s in sequences]
        mask = [[1] * l + [0] * (max_len - l) for l in lengths]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long), lengths

    def teach_brain(self, brain: EnhancedCognitiveNetwork, user_input: str, vocab: Dict[str, int],
                    ivocab: Dict[int, str],
                    epochs: int = 3, batch_size: int = 12):
        raw_qwen = self.query_qwen(user_input)
        tagged_qwen = classify_and_tag_response(raw_qwen)
        print(f"👤: {user_input}")
        print(f"🤖: {tagged_qwen}")
        input_type = detect_input_type(user_input)
        allowed_stages = get_allowed_stages(input_type)
        input_tokens = brain.text_to_tokens(user_input, vocab, include_tags=False)
        response_tokens = brain.text_to_tokens(tagged_qwen, vocab, include_tags=True)
        if not response_tokens:
            response_tokens = [vocab.get("хорошо", 2), vocab.get(".", 38)]
        eos_id = vocab.get('<EOS>', 1)
        full_seq = input_tokens + response_tokens + [eos_id]
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]
        priority = 2.0 if "[MET]" in tagged_qwen or "рефлексия" in user_input.lower() else 1.0
        brain.replay.add(input_seq, target_seq, meta={"qwen": tagged_qwen}, priority=priority)
        samples, idxs, is_weights = brain.replay.sample(batch_size - 1, beta=0.4) if len(brain.replay) > 1 else ([], [],
                                                                                                                 [])
        batch_inputs = [input_seq] + [s['input'] for s in samples]
        batch_targets = [target_seq] + [s['target'] for s in samples]
        sorted_pairs = sorted(zip(batch_inputs, batch_targets, [1.0] + is_weights, [None] + idxs),
                              key=lambda x: len(x[0]), reverse=True)
        batch_inputs_sorted = [p[0] for p in sorted_pairs]
        batch_targets_sorted = [p[1] for p in sorted_pairs]
        importance_weights = [p[2] for p in sorted_pairs]
        sample_idxs = [p[3] for p in sorted_pairs]
        input_tensor, attn_mask, _ = self._pad_batch(batch_inputs_sorted, pad_id=0)
        target_tensor, _, _ = self._pad_batch(batch_targets_sorted, pad_id=0)
        input_tensor = input_tensor.to(brain.device)
        target_tensor = target_tensor.to(brain.device)
        attn_mask = attn_mask.to(brain.device)
        optimizer = torch.optim.AdamW(brain.parameters(), lr=3e-4, weight_decay=1e-6)
        brain.train()
        total_loss = 0.0
        for ep in range(epochs):
            optimizer.zero_grad()
            logits = brain.process_sequence(input_tensor, attention_mask=attn_mask,
                                            allowed_stages=allowed_stages, user_input=user_input)
            loss = brain.compute_weighted_loss(logits, target_tensor, vocab, ivocab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.8)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / epochs
        print(f"📚 Потеря: {avg_loss:.4f}")
        brain.eval()
        with torch.no_grad():
            brain_tokens = brain.generate_sequence(
                torch.tensor([input_seq], dtype=torch.long, device=brain.device),
                allowed_stages=allowed_stages,
                user_input=user_input
            )
            brain_response_raw = " ".join(ivocab.get(tid, "<UNK>") for tid in brain_tokens)
        brain_response_clean = TAG_PATTERN.sub('', brain_response_raw).strip()
        self.conversation_history.append({
            'input': user_input,
            'qwen_tagged': tagged_qwen,
            'brain_raw': brain_response_raw,
            'brain_clean': brain_response_clean,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        })
        print(f"🧠 Сырой: {brain_response_raw}")
        print(f"💬 Чистый: {brain_response_clean}")
        brain.print_thought_flow()

        # Записываем эпизод для рефлексивного обучения
        if brain.reflective_learner:
            brain.reflective_learner.record_learning_episode(
                user_input, brain_response_clean,
                {'confidence': 1.0 - avg_loss, 'complexity': len(user_input.split()) / 50}
            )
            brain.reflective_learner.extract_insights()

        return tagged_qwen, brain_response_clean


# ======================
# ИНИЦИАЛИЗАЦИЯ
# ======================

def create_initial_vocabulary() -> Dict[str, int]:
    return {
        '<PAD>': 0, '<EOS>': 1, '<UNK>': 2,
        '[SOC]': 3, '[FCT]': 4, '[CAU]': 5,
        '[PRC]': 6, '[OPN]': 7, '[MET]': 8, '[CRT]': 9,
        'привет': 10, 'здравствуй': 11, 'добрый': 12, 'день': 13,
        'как': 14, 'ты': 15, 'дела': 16, 'что': 17, 'такое': 18,
        'это': 19, 'потому': 20, 'чтобы': 21,
        'я': 22, 'думаю': 23, 'мне': 24, 'кажется': 25,
        'представь': 26, 'вообрази': 27, 'жизнь': 28,
        'спасибо': 29, 'пожалуйста': 30, 'извини': 31,
        'хорошо': 32, 'ладно': 33, 'ок': 34,
        'нейрон': 35, 'гравитация': 36, 'вода': 37,
        '.': 38, ',': 39, '?': 40, '!': 41
    }


# ======================
# MAIN
# ======================

def main():
    print("🧠 ЗАПУСК УЛУЧШЕННОЙ АРХИТЕКТУРЫ МЫШЛЕНИЯ С РЕФЛЕКСИЕЙ")
    print("   • Рабочая память и осознанность")
    print("   • Стратегическое планирование мышления")
    print("   • Внутренний диалог клеток с самоконтролем")
    print("   • Рефлексивное обучение и извлечение инсайтов")

    vocab = create_initial_vocabulary()
    vocab_size = len(vocab) + 4096
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    brain = EnhancedCognitiveNetwork(vocab_size=vocab_size, embedding_dim=192, hidden_size=384,
                                     eos_token_id=vocab['<EOS>'], device=device)
    teacher = BrainTeacher(device=device)

    loaded_vocab = brain.load_knowledge("enhanced_brain_knowledge.pth")
    if loaded_vocab is not None:
        vocab = loaded_vocab
        print("✅ Знания загружены")
    else:
        brain.meta_cog = MetaCognition(vocab, network=brain)
        print("📝 Создана новая улучшенная сеть")

    ivocab = {v: k for k, v in vocab.items()}
    print(f"🔢 Клеток: {len(brain.cells)} | 📚 Словарь: {len(vocab)} слов")
    print("💬 Готов к диалогу! (введите 'выход')")

    conversation_count = 0
    reflection_buffer: List[str] = []
    MAX_REFLECTION_CHAIN = 3
    reflection_chain = 0

    try:
        while True:
            if reflection_buffer:
                user_input = reflection_buffer.pop(0)
                print(f"\n💭 Мозг задаёт вопрос: {user_input}")
            else:
                user_input = input("\n👤 Вы: ").strip()
                if user_input.lower() in ['выход', 'exit', 'quit']:
                    break
                if not user_input:
                    continue
                reflection_chain = 0

            qwen_resp, brain_resp = teacher.teach_brain(brain, user_input, vocab, ivocab)
            ivocab = {v: k for k, v in vocab.items()}

            new_question = brain.reflect_and_learn(user_input, qwen_resp, vocab, ivocab)
            if new_question:
                recent_inputs = {entry['input'] for entry in teacher.conversation_history[-5:]}
                if new_question not in recent_inputs and new_question not in reflection_buffer:
                    if reflection_chain < MAX_REFLECTION_CHAIN:
                        reflection_buffer.append(new_question)
                        reflection_chain += 1
                        print(f"💡 Запланирован рефлексивный вопрос: {new_question}")

            brain._check_cell_creation(user_input, qwen_resp, brain_resp)
            conversation_count += 1

            if conversation_count % 2 == 0:
                brain.save_knowledge("enhanced_brain_knowledge.pth", vocab)
                print("💾 Сохранено!")

    except KeyboardInterrupt:
        print("\n🛑 Прервано")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
    finally:
        brain.save_knowledge("enhanced_brain_knowledge.pth", vocab)
        print(f"\n🧠 Итог: клеток = {len(brain.cells)}, словарь = {len(vocab)} слов")
        print("🎯 Активность клеток по типам:")
        brain.print_thought_flow()


if __name__ == "__main__":
    main()