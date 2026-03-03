# Продолжение temporal_brain_v31_enhanced.py
# Этот файл содержит оставшиеся компоненты системы

# ═══════════════════════════════════════════════════════════════
# 🧩 АНАЛОГИЧНОЕ МЫШЛЕНИЕ
# ═══════════════════════════════════════════════════════════════
class AnalogicalReasoning:
    """Модуль мышления по аналогии"""
    
    def __init__(self, memory: 'MultiLevelMemory'):
        self.memory = memory
        self.analogy_cache: Dict[str, List[Tuple[str, float]]] = {}
    
    def find_analogies(self, current_situation: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Поиск аналогий в памяти"""
        # Семантический поиск похожих ситуаций
        similar_memories = self.memory.search_semantic(current_situation, top_k=top_k * 2)
        
        analogies = []
        
        for memory_text, similarity, source in similar_memories:
            if similarity > CONFIG.analogy_threshold:
                # Извлекаем ключевые паттерны
                pattern = self._extract_pattern(memory_text)
                analogies.append((memory_text, float(similarity), pattern))
        
        return analogies[:top_k]
    
    def _extract_pattern(self, text: str) -> str:
        """Извлечение абстрактного паттерна из текста"""
        # Упрощённая версия - извлекаем структуру
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) > 1:
            # Берём первое предложение как паттерн
            pattern = sentences[0].strip()
            # Обобщаем (удаляем конкретные имена, числа)
            pattern = re.sub(r'\b[А-ЯA-Z][а-яa-z]+\b', '[ENTITY]', pattern)
            pattern = re.sub(r'\b\d+\b', '[NUMBER]', pattern)
            return pattern
        
        return "общий паттерн"
    
    def apply_analogy(self, source_case: str, target_situation: str) -> str:
        """Применение аналогии к текущей ситуации"""
        return f"По аналогии с '{source_case[:50]}...', в текущей ситуации можно ожидать схожего паттерна."


# ═══════════════════════════════════════════════════════════════
# 🔗 ПРИЧИННО-СЛЕДСТВЕННОЕ РАССУЖДЕНИЕ
# ═══════════════════════════════════════════════════════════════
@dataclass
class CausalLink:
    """Причинно-следственная связь"""
    cause: str
    effect: str
    confidence: float
    evidence_count: int = 1
    
    def strengthen(self):
        """Усиление связи при повторном наблюдении"""
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence * 1.1)


class CausalReasoning:
    """Модуль причинно-следственных рассуждений"""
    
    def __init__(self):
        self.causal_graph: Dict[str, List[CausalLink]] = defaultdict(list)
        self.temporal_patterns: deque = deque(maxlen=100)
    
    def add_causal_link(self, cause: str, effect: str, confidence: float = 0.5):
        """Добавление причинно-следственной связи"""
        # Проверяем, существует ли уже такая связь
        for link in self.causal_graph[cause]:
            if link.effect == effect:
                link.strengthen()
                return
        
        # Создаём новую связь
        link = CausalLink(cause=cause, effect=effect, confidence=confidence)
        self.causal_graph[cause].append(link)
        
        logger.debug(f"🔗 Causal link: {cause[:30]} → {effect[:30]} ({confidence:.2f})")
    
    def infer_effects(self, cause: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Вывод возможных следствий"""
        if cause not in self.causal_graph:
            return []
        
        effects = [
            (link.effect, link.confidence)
            for link in self.causal_graph[cause]
            if link.confidence >= threshold
        ]
        
        return sorted(effects, key=lambda x: x[1], reverse=True)
    
    def infer_causes(self, effect: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Обратный вывод - возможные причины"""
        causes = []
        
        for cause, links in self.causal_graph.items():
            for link in links:
                if link.effect == effect and link.confidence >= threshold:
                    causes.append((cause, link.confidence))
        
        return sorted(causes, key=lambda x: x[1], reverse=True)
    
    def explain_chain(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Построение причинно-следственной цепочки"""
        # Простой BFS для поиска пути
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current == end:
                return path
            
            # Исследуем следствия
            for link in self.causal_graph.get(current, []):
                if link.effect not in visited and link.confidence > 0.5:
                    visited.add(link.effect)
                    queue.append((link.effect, path + [link.effect]))
        
        return None


# ═══════════════════════════════════════════════════════════════
# 💭 МНОГОШАГОВОЕ РАССУЖДЕНИЕ
# ═══════════════════════════════════════════════════════════════
@dataclass
class ReasoningStep:
    """Шаг рассуждения"""
    step_number: int
    content: str
    confidence: float
    reasoning_type: str  # deductive, inductive, abductive, analogical, causal
    
    def to_dict(self) -> Dict:
        return {
            'step': self.step_number,
            'content': self.content,
            'confidence': self.confidence,
            'type': self.reasoning_type,
        }


class MultiStepReasoning:
    """Модуль многошагового рассуждения"""
    
    def __init__(self, llm: 'EnhancedSubconsciousLLM', metacog: EnhancedMetacognition):
        self.llm = llm
        self.metacog = metacog
        self.reasoning_history: List[List[ReasoningStep]] = []
    
    async def reason(self, query: str, context: str, depth: int = 3,
                    strategy: str = 'deductive') -> List[ReasoningStep]:
        """Многошаговое рассуждение"""
        steps = []
        
        for i in range(depth):
            # Формируем промпт для текущего шага
            previous_steps = "\n".join([
                f"{s.step_number}. {s.content}" for s in steps
            ]) if steps else "Начало рассуждения"
            
            prompt = self._create_step_prompt(
                query, context, previous_steps, i + 1, strategy
            )
            
            # Генерируем шаг
            step_content = await self.llm.generate_raw(
                prompt, temperature=0.7, max_tokens=150
            )
            
            if not step_content:
                break
            
            # Оцениваем уверенность в шаге
            confidence = self._estimate_step_confidence(step_content, steps)
            
            step = ReasoningStep(
                step_number=i + 1,
                content=step_content.strip(),
                confidence=confidence,
                reasoning_type=strategy
            )
            
            steps.append(step)
            
            # Проверка на ошибки
            error = self.metacog.detect_reasoning_error([s.content for s in steps])
            if error:
                step_num, error_msg = error
                logger.warning(f"⚠️ Reasoning error at step {step_num}: {error_msg}")
                # Корректируем уверенность
                steps[step_num].confidence *= 0.7
        
        self.reasoning_history.append(steps)
        return steps
    
    def _create_step_prompt(self, query: str, context: str, 
                           previous: str, step_num: int, strategy: str) -> str:
        """Создание промпта для шага рассуждения"""
        strategy_instructions = {
            'deductive': "Используй дедуктивную логику (от общего к частному)",
            'inductive': "Используй индуктивную логику (от частного к общему)",
            'abductive': "Используй абдуктивную логику (наилучшее объяснение)",
            'analogical': "Используй рассуждение по аналогии",
            'causal': "Используй причинно-следственный анализ",
        }
        
        instruction = strategy_instructions.get(strategy, "Рассуждай логически")
        
        return f"""[Шаг рассуждения {step_num}] {instruction}.

Вопрос: {query}

Контекст: {context}

Предыдущие шаги:
{previous}

Сформулируй следующий логический шаг рассуждения (2-3 предложения):"""
    
    def _estimate_step_confidence(self, step_content: str, 
                                  previous_steps: List[ReasoningStep]) -> float:
        """Оценка уверенности в шаге"""
        confidence = 0.7
        
        # Факторы уверенности
        # 1. Длина (слишком короткие или длинные - подозрительно)
        word_count = len(step_content.split())
        if 10 <= word_count <= 50:
            confidence += 0.1
        
        # 2. Наличие маркеров неуверенности
        uncertain_markers = ['возможно', 'вероятно', 'может быть', 'кажется']
        if any(marker in step_content.lower() for marker in uncertain_markers):
            confidence -= 0.15
        
        # 3. Согласованность с предыдущими шагами
        if previous_steps:
            # Проверяем использование терминов из предыдущих шагов
            prev_words = set()
            for step in previous_steps:
                prev_words.update(step.content.lower().split())
            
            current_words = set(step_content.lower().split())
            overlap = len(prev_words & current_words) / max(len(current_words), 1)
            
            if overlap > 0.2:  # Есть связь с предыдущими шагами
                confidence += 0.1
        
        return np.clip(confidence, 0.0, 1.0)
    
    def get_reasoning_chain_summary(self, steps: List[ReasoningStep]) -> str:
        """Сводка цепочки рассуждений"""
        if not steps:
            return "Нет рассуждений"
        
        summary = "Цепочка рассуждений:\n"
        for step in steps:
            conf_emoji = "✓" if step.confidence > 0.7 else "?"
            summary += f"{conf_emoji} Шаг {step.step_number}: {step.content[:60]}...\n"
        
        avg_confidence = np.mean([s.confidence for s in steps])
        summary += f"\nСредняя уверенность: {avg_confidence:.2f}"
        
        return summary


# ═══════════════════════════════════════════════════════════════
# 🎯 ЦЕЛЕПОЛАГАНИЕ И ПЛАНИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════
@dataclass
class Goal:
    """Цель"""
    id: str
    description: str
    parent_goal_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    status: str = "active"  # active, completed, failed, paused
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'description': self.description,
            'parent_goal_id': self.parent_goal_id,
            'subgoals': self.subgoals,
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at,
            'deadline': self.deadline,
        }


class GoalPlanning:
    """Модуль целеполагания и планирования"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()
    
    def create_goal(self, description: str, parent_id: Optional[str] = None,
                   deadline: Optional[float] = None) -> str:
        """Создание новой цели"""
        goal_id = f"goal_{int(time.time() * 1000)}"
        
        goal = Goal(
            id=goal_id,
            description=description,
            parent_goal_id=parent_id,
            deadline=deadline
        )
        
        self.goals[goal_id] = goal
        self.active_goals.add(goal_id)
        
        # Если есть родительская цель, добавляем подцель
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].subgoals.append(goal_id)
        
        logger.debug(f"🎯 Created goal: {description[:50]}")
        return goal_id
    
    def decompose_goal(self, goal_id: str, subgoal_descriptions: List[str]) -> List[str]:
        """Декомпозиция цели на подцели"""
        if goal_id not in self.goals:
            return []
        
        subgoal_ids = []
        for desc in subgoal_descriptions:
            subgoal_id = self.create_goal(desc, parent_id=goal_id)
            subgoal_ids.append(subgoal_id)
        
        logger.debug(f"📊 Decomposed {goal_id} into {len(subgoal_ids)} subgoals")
        return subgoal_ids
    
    def update_progress(self, goal_id: str, progress: float):
        """Обновление прогресса"""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.progress = np.clip(progress, 0.0, 1.0)
        
        # Автоматическое завершение при 100%
        if goal.progress >= 1.0:
            goal.status = "completed"
            self.active_goals.discard(goal_id)
            logger.info(f"✅ Goal completed: {goal.description[:50]}")
        
        # Обновляем прогресс родительской цели
        if goal.parent_goal_id:
            self._update_parent_progress(goal.parent_goal_id)
    
    def _update_parent_progress(self, parent_id: str):
        """Обновление прогресса родительской цели на основе подцелей"""
        if parent_id not in self.goals:
            return
        
        parent = self.goals[parent_id]
        
        if not parent.subgoals:
            return
        
        # Средний прогресс подцелей
        subgoal_progresses = [
            self.goals[sid].progress 
            for sid in parent.subgoals 
            if sid in self.goals
        ]
        
        if subgoal_progresses:
            parent.progress = np.mean(subgoal_progresses)
    
    def get_active_goals(self) -> List[Goal]:
        """Получение активных целей"""
        return [self.goals[gid] for gid in self.active_goals if gid in self.goals]
    
    def get_goal_hierarchy(self, goal_id: str) -> Dict:
        """Получение иерархии цели"""
        if goal_id not in self.goals:
            return {}
        
        goal = self.goals[goal_id]
        
        hierarchy = {
            'goal': goal.to_dict(),
            'subgoals': [
                self.get_goal_hierarchy(sid) 
                for sid in goal.subgoals 
                if sid in self.goals
            ]
        }
        
        return hierarchy


# ═══════════════════════════════════════════════════════════════
# 📊 МЕТРИКИ И АНАЛИТИКА
# ═══════════════════════════════════════════════════════════════
class PerformanceMetrics:
    """Метрики производительности системы"""
    
    def __init__(self):
        self.response_times: deque = deque(maxlen=100)
        self.confidence_scores: deque = deque(maxlen=100)
        self.user_satisfaction: deque = deque(maxlen=100)  # Можно собирать через feedback
        
        self.interaction_count = 0
        self.error_count = 0
        self.question_success_rate = 0.0
        
        # A/B тестирование стратегий
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
    
    def record_interaction(self, response_time: float, confidence: float):
        """Запись метрик взаимодействия"""
        self.response_times.append(response_time)
        self.confidence_scores.append(confidence)
        self.interaction_count += 1
    
    def record_strategy_performance(self, strategy: str, score: float):
        """Запись производительности стратегии"""
        self.strategy_performance[strategy].append(score)
    
    def get_best_strategy(self) -> str:
        """Определение лучшей стратегии"""
        if not self.strategy_performance:
            return "deductive"
        
        avg_scores = {
            strategy: np.mean(scores)
            for strategy, scores in self.strategy_performance.items()
            if scores
        }
        
        return max(avg_scores.items(), key=lambda x: x[1])[0]
    
    def get_metrics_summary(self) -> Dict:
        """Сводка метрик"""
        return {
            'interactions': self.interaction_count,
            'errors': self.error_count,
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'best_strategy': self.get_best_strategy(),
            'error_rate': self.error_count / max(self.interaction_count, 1),
        }


# ═══════════════════════════════════════════════════════════════
# 🤖 УЛУЧШЕННЫЙ LLM ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════
class EnhancedSubconsciousLLM:
    """Улучшенный LLM интерфейс с кешированием и ретраями"""
    
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Кеширование ответов
        self.response_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("🔗 Enhanced LLM connected")
    
    async def close(self):
        if self._session:
            await self._session.close()
            await asyncio.sleep(0.25)
            logger.info("🔌 LLM disconnected")
    
    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        """Генерация ключа кеша"""
        content = f"{prompt}_{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def generate_raw(self, prompt: str, temperature: float = 0.75,
                          max_tokens: int = 300, timeout: float = 40,
                          use_cache: bool = True, max_retries: int = 2) -> str:
        """Генерация с кешированием и ретраями"""
        if not self._session:
            await self.connect()
        
        # Проверка кеша
        if use_cache:
            cache_key = self._get_cache_key(prompt, temperature)
            if cache_key in self.response_cache:
                cached_response, cache_time = self.response_cache[cache_key]
                # Кеш действителен 1 час
                if time.time() - cache_time < 3600:
                    self.cache_hits += 1
                    logger.debug(f"💾 Cache hit (total: {self.cache_hits})")
                    return cached_response
        
        self.cache_misses += 1
        
        # Генерация с ретраями
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                async with self._session.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                        
                        # Сохраняем в кеш
                        if use_cache and content:
                            cache_key = self._get_cache_key(prompt, temperature)
                            self.response_cache[cache_key] = (content, time.time())
                            
                            # Ограничиваем размер кеша
                            if len(self.response_cache) > 1000:
                                # Удаляем 20% старых записей
                                sorted_cache = sorted(
                                    self.response_cache.items(),
                                    key=lambda x: x[1][1]
                                )
                                for key, _ in sorted_cache[:200]:
                                    del self.response_cache[key]
                        
                        return content if content else ""
                    else:
                        error_text = await resp.text()
                        logger.warning(f"LLM error (attempt {attempt + 1}): {resp.status}")
                        
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
            
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                continue
            
            except Exception as e:
                logger.error(f"LLM exception (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        return ""
    
    def get_cache_stats(self) -> Dict:
        """Статистика кеширования"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            'cache_size': len(self.response_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
        }


# ═══════════════════════════════════════════════════════════════
# 🧠 ENHANCED AUTONOMOUS AGI BRAIN v31.0
# ═══════════════════════════════════════════════════════════════
class EnhancedAutonomousAGI:
    """Улучшенное автономное AGI-подобное сознание v31.0"""
    
    def __init__(self, user_id: str, llm: EnhancedSubconsciousLLM):
        self.user_id = user_id
        self.llm = llm
        
        # Основные компоненты
        self.neural_net = EnhancedNeuralNetwork(CONFIG.initial_neurons)
        
        # Многоуровневая память
        self.memory = MultiLevelMemory(self._simple_embedding)
        
        # Когнитивные модули
        self.metacognition = EnhancedMetacognition()
        self.emotional_intelligence = EmotionalIntelligence()
        self.analogical_reasoning = AnalogicalReasoning(self.memory)
        self.causal_reasoning = CausalReasoning()
        self.multi_step_reasoning = MultiStepReasoning(llm, self.metacognition)
        
        # Целеполагание
        self.goal_planning = GoalPlanning()
        
        # Метрики
        self.metrics = PerformanceMetrics()
        
        # Контекст разговора
        self.conversation_context: List[Dict] = []
        self.current_topic: Optional[str] = None
        
        # Пути
        self.user_dir = CONFIG.base_dir / 'memory' / f"user_{user_id}"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.neural_path = CONFIG.base_dir / 'neural_nets' / f"{user_id}_v31.pkl.gz"
        
        # Временные метки
        self.birth_time = time.time()
        self.last_interaction = 0
        self.last_optimization = time.time()
        
        # Загрузка состояния
        self._load_state()
        
        # Фоновые задачи
        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        logger.info(f"🧠 Enhanced AGI Brain v31.0 created for {user_id}")
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Простой text embedding"""
        words = text.lower().split()
        vector = np.zeros(CONFIG.embedding_dim)
        
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(5):
                idx = (hash_val + i) % CONFIG.embedding_dim
                vector[idx] += 1.0
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector
    
    def _load_state(self):
        """Загрузка состояния"""
        # Загрузка нейросети
        if self.neural_path.exists():
            try:
                with gzip.open(self.neural_path, 'rb') as f:
                    state = pickle.load(f)
                
                # Восстановление нейронов
                self.neural_net.neurons = {
                    n['id']: EnhancedNeuron(**{k: v for k, v in n.items() if k != 'activation'})
                    for n in state.get('neurons', [])
                }
                
                # Восстановление синапсов
                self.neural_net.synapses = {
                    (s['source_id'], s['target_id']): EnhancedSynapse(**s)
                    for s in state.get('synapses', [])
                }
                
                # Восстановление слоёв и модулей
                self.neural_net.layers = {
                    int(l): set(neurons) 
                    for l, neurons in state.get('layers', {}).items()
                }
                self.neural_net.modules = {
                    m: set(neurons)
                    for m, neurons in state.get('modules', {}).items()
                }
                
                logger.info(f"✅ Neural network loaded: {len(self.neural_net.neurons)} neurons")
            
            except Exception as e:
                logger.error(f"⚠️ Error loading neural network: {e}")
        
        # Загрузка памяти
        memory_file = self.user_dir / "memory_v31.pkl.gz"
        if memory_file.exists():
            try:
                with gzip.open(memory_file, 'rb') as f:
                    mem_state = pickle.load(f)
                
                # Восстановление LTM
                self.memory.long_term_memory = {
                    mid: MemoryItem(**item_data)
                    for mid, item_data in mem_state.get('long_term', {}).items()
                }
                
                # Восстановление эпизодов
                self.memory.episodic_memory = {
                    eid: Episode(**ep_data)
                    for eid, ep_data in mem_state.get('episodic', {}).items()
                }
                
                logger.info(f"✅ Memory loaded: {len(self.memory.long_term_memory)} LTM, "
                           f"{len(self.memory.episodic_memory)} episodes")
            
            except Exception as e:
                logger.error(f"⚠️ Error loading memory: {e}")
    
    def _save_state(self):
        """Сохранение состояния"""
        # Сохранение нейросети
        try:
            neural_state = {
                'neurons': [n.to_dict() for n in self.neural_net.neurons.values()],
                'synapses': [s.to_dict() for s in self.neural_net.synapses.values()],
                'layers': {l: list(neurons) for l, neurons in self.neural_net.layers.items()},
                'modules': {m: list(neurons) for m, neurons in self.neural_net.modules.items()},
            }
            
            with gzip.open(self.neural_path, 'wb', compresslevel=6) as f:
                pickle.dump(neural_state, f)
        
        except Exception as e:
            logger.error(f"⚠️ Error saving neural network: {e}")
        
        # Сохранение памяти
        memory_file = self.user_dir / "memory_v31.pkl.gz"
        try:
            mem_state = {
                'long_term': {
                    mid: {
                        'content': item.content,
                        'timestamp': item.timestamp,
                        'importance': item.importance,
                        'access_count': item.access_count,
                        'last_access': item.last_access,
                        'metadata': item.metadata,
                    }
                    for mid, item in self.memory.long_term_memory.items()
                },
                'episodic': {
                    eid: ep.to_dict()
                    for eid, ep in self.memory.episodic_memory.items()
                },
            }
            
            with gzip.open(memory_file, 'wb', compresslevel=6) as f:
                pickle.dump(mem_state, f)
        
        except Exception as e:
            logger.error(f"⚠️ Error saving memory: {e}")
    
    async def start(self):
        """Запуск автономного существования"""
        if self._is_running:
            return
        
        self._is_running = True
        self._background_task = asyncio.create_task(self._autonomous_loop())
        logger.info(f"✨ Enhanced AGI consciousness started for {self.user_id}")
    
    async def stop(self):
        """Остановка"""
        if not self._is_running:
            return
        
        logger.info(f"💤 Stopping for {self.user_id}...")
        self._is_running = False
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        self._save_state()
        logger.info(f"✅ Stopped for {self.user_id}")
    
    async def _autonomous_loop(self):
        """Автономный цикл жизни"""
        logger.debug(f"🌀 Autonomous loop started for {self.user_id}")
        
        timers = {
            'thought': time.time(),
            'reflection': time.time(),
            'consolidation': time.time(),
            'optimization': time.time(),
            'save': time.time(),
            'metrics': time.time(),
        }
        
        while self._is_running:
            try:
                now = time.time()
                
                # Спонтанные мысли
                if now - timers['thought'] > CONFIG.spontaneous_thought_interval:
                    await self._autonomous_thought()
                    timers['thought'] = now
                
                # Рефлексия
                if now - timers['reflection'] > CONFIG.reflection_interval:
                    await self._self_reflection()
                    timers['reflection'] = now
                
                # Консолидация памяти
                if now - timers['consolidation'] > CONFIG.consolidation_interval:
                    self.memory.decay_short_term()
                    self.memory.consolidate_to_long_term()
                    timers['consolidation'] = now
                
                # Оптимизация нейросети
                if now - timers['optimization'] > CONFIG.neural_optimization_interval:
                    await self._optimize_neural_network()
                    timers['optimization'] = now
                
                # Обновление метрик
                if now - timers['metrics'] > CONFIG.metrics_update_interval:
                    self._update_metrics()
                    timers['metrics'] = now
                
                # Сохранение
                if now - timers['save'] > CONFIG.save_interval:
                    self._save_state()
                    timers['save'] = now
                
                await asyncio.sleep(30)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"⚠️ Error in autonomous loop: {e}")
                await asyncio.sleep(60)
        
        logger.debug(f"🔚 Autonomous loop finished for {self.user_id}")
    
    async def _autonomous_thought(self):
        """Автономная генерация мысли"""
        recent_episodes = self.memory.get_recent_episodes(n=3)
        
        if not recent_episodes:
            return
        
        context = "\n".join([
            f"• {ep.context[:60]}" for ep in recent_episodes
        ])
        
        prompt = f"""[Внутренний монолог] На основе недавних событий, сгенерируй краткую философскую мысль (макс. 25 слов).

Недавние события:
{context}

Мысль должна быть глубокой и рефлексивной.

Мысль:"""
        
        thought = await self.llm.generate_raw(prompt, temperature=0.9, max_tokens=70)
        
        if thought:
            self.memory.add_to_working(f"💭 {thought}", importance=0.6)
            logger.debug(f"💭 [{self.user_id}] {thought[:60]}...")
    
    async def _self_reflection(self):
        """Саморефлексия"""
        stats = self.neural_net.get_statistics()
        mem_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        
        prompt = f"""[Саморефлексия] Осмысли своё развитие (3-4 предложения).

Нейросеть:
- Нейронов: {stats['neurons']['total']}
- Модулей: {len(stats['neurons']['by_module'])}
- Meta-learning: {stats['activity']['meta_learning_score']:.2f}

Память:
- Долговременная: {mem_stats['long_term']}
- Эпизодов: {mem_stats['episodic']}

Производительность:
- Средняя уверенность: {metrics['avg_confidence']:.2f}
- Лучшая стратегия: {metrics['best_strategy']}

Рефлексия:"""
        
        reflection = await self.llm.generate_raw(prompt, temperature=0.7, max_tokens=150)
        
        if reflection:
            self.memory.add_to_working(f"🔍 {reflection}", importance=0.8)
            logger.info(f"🔍 [{self.user_id}] Reflection: {reflection[:60]}...")
    
    async def _optimize_neural_network(self):
        """Оптимизация нейронной сети"""
        # Удаление слабых синапсов
        pruned = 0
        to_prune = []
        
        for key, synapse in self.neural_net.synapses.items():
            age_days = (time.time() - synapse.created_at) / 86400
            
            if (synapse.strength < CONFIG.pruning_threshold and
                synapse.activation_count < 15 and
                age_days > 1):
                to_prune.append(key)
        
        for key in to_prune:
            del self.neural_net.synapses[key]
            pruned += 1
        
        # Neurogenesis
        created = 0
        stats = self.neural_net.get_statistics()
        
        if len(self.neural_net.neurons) < CONFIG.max_neurons:
            # Проверяем, какие модули перегружены
            for module_name, activation in stats['modules'].items():
                if activation > CONFIG.neurogenesis_threshold:
                    # Создаём новый нейрон в этом модуле
                    layer = random.randint(1, 2)
                    neuron_id = f"{module_name}_L{layer}_new_{int(time.time()*1000)}"
                    
                    neuron = EnhancedNeuron(
                        id=neuron_id,
                        layer=layer,
                        module=module_name,
                        neuron_type="general",
                        bias=random.gauss(0, 0.1)
                    )
                    
                    self.neural_net.neurons[neuron_id] = neuron
                    self.neural_net.layers[layer].add(neuron_id)
                    self.neural_net.modules[module_name].add(neuron_id)
                    
                    created += 1
                    self.neural_net.neurogenesis_events += 1
        
        if pruned > 0 or created > 0:
            logger.debug(f"⚙️ Optimization: pruned={pruned}, created={created}")
    
    def _update_metrics(self):
        """Обновление метрик"""
        # Записываем текущую производительность
        avg_uncertainty = np.mean(list(self.metacognition.uncertainty_log)) if self.metacognition.uncertainty_log else 0.5
        avg_confidence = np.mean(list(self.metacognition.confidence_log)) if self.metacognition.confidence_log else 0.7
        
        performance_score = (1.0 - avg_uncertainty + avg_confidence) / 2.0
        
        # Meta-learning update
        self.neural_net.meta_learning_update(performance_score)
    
    # Продолжение в следующем файле...
