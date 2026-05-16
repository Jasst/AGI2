# Продолжение emergent_agi_v3.py
# Добавьте этот код в конец файла

# ═══════════════════════════════════════════════════════════════
# 🧠 ОСТАЛЬНЫЕ КОМПОНЕНТЫ (из v2.0)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CoreBelief:
    statement: str
    confidence: float
    evidence: List[str]
    last_updated: float = field(default_factory=time.time)
    
    @property
    def id(self) -> str:
        return hashlib.md5(self.statement.encode()).hexdigest()[:8]


class SelfIdentity:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.core_beliefs: List[CoreBelief] = []
        self.birth_time = time.time()
        self.total_interactions = 0
        self.total_introspections = 0
        self.behavioral_patterns: Dict[str, int] = defaultdict(int)
        self._load()
    
    def update_from_interaction(self, interaction_data: Dict):
        self.total_interactions += 1
        method = interaction_data.get('method', 'unknown')
        self.behavioral_patterns[method] += 1
    
    async def form_belief(self, statement: str, evidence: List[str], llm) -> CoreBelief:
        contradictions = []
        for belief in self.core_beliefs:
            check = await llm.generate(
                f"""Проверь на логическое противоречие:
УБЕЖДЕНИЕ 1: {belief.statement}
УБЕЖДЕНИЕ 2: {statement}

Ответь только: ПРОТИВОРЕЧИЕ или СОВМЕСТИМО""",
                temperature=0.0,
                max_tokens=20
            )
            
            if "ПРОТИВОРЕЧИЕ" in check:
                contradictions.append(belief.id)
        
        confidence = min(0.95, len(evidence) * 0.2 + 0.3)
        if contradictions:
            confidence *= 0.5
        
        belief = CoreBelief(
            statement=statement,
            confidence=confidence,
            evidence=evidence
        )
        
        self.core_beliefs.append(belief)
        
        if len(self.core_beliefs) > CONFIG.core_beliefs_capacity:
            self.core_beliefs.sort(key=lambda b: b.confidence, reverse=True)
            self.core_beliefs = self.core_beliefs[:CONFIG.core_beliefs_capacity]
        
        return belief
    
    def get_self_description(self) -> str:
        if not self.core_beliefs:
            return "Я еще формирую понимание себя..."
        
        top_beliefs = sorted(self.core_beliefs, key=lambda b: b.confidence, reverse=True)[:5]
        
        description = "Вот что я знаю о себе:\n"
        for i, belief in enumerate(top_beliefs, 1):
            description += f"{i}. {belief.statement} (уверенность: {belief.confidence:.0%})\n"
        
        return description
    
    def _save(self):
        path = CONFIG.base_dir / 'identity' / f'{self.user_id}.json'
        data = {
            'birth_time': self.birth_time,
            'total_interactions': self.total_interactions,
            'total_introspections': self.total_introspections,
            'behavioral_patterns': dict(self.behavioral_patterns),
            'core_beliefs': [asdict(b) for b in self.core_beliefs]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load(self):
        path = CONFIG.base_dir / 'identity' / f'{self.user_id}.json'
        if not path.exists():
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.birth_time = data.get('birth_time', time.time())
            self.total_interactions = data.get('total_interactions', 0)
            self.total_introspections = data.get('total_introspections', 0)
            self.behavioral_patterns = defaultdict(int, data.get('behavioral_patterns', {}))
            self.core_beliefs = [CoreBelief(**b) for b in data.get('core_beliefs', [])]
            
            logger.info(f"✅ Loaded identity with {len(self.core_beliefs)} beliefs")
        except Exception as e:
            logger.error(f"Failed to load identity: {e}")


class MetaCognition:
    def __init__(self):
        self.thought_processes: deque = deque(maxlen=100)
        self.decision_quality: deque = deque(maxlen=50)
        self.reasoning_depth: deque = deque(maxlen=50)
        self.patterns: Dict[str, List[float]] = defaultdict(list)
    
    def log_thought_process(self, process_type: str, quality: float, depth: int, metadata: Dict = None):
        entry = {
            'timestamp': time.time(),
            'type': process_type,
            'quality': quality,
            'depth': depth,
            'metadata': metadata or {}
        }
        
        self.thought_processes.append(entry)
        self.decision_quality.append(quality)
        self.reasoning_depth.append(depth)
        self.patterns[process_type].append(quality)
    
    def analyze_thinking_patterns(self) -> Dict:
        if not self.thought_processes:
            return {'status': 'insufficient_data'}
        
        avg_quality = np.mean(list(self.decision_quality)) if self.decision_quality else 0
        avg_depth = np.mean(list(self.reasoning_depth)) if self.reasoning_depth else 0
        
        pattern_analysis = {}
        for pattern_type, qualities in self.patterns.items():
            if len(qualities) >= 3:
                pattern_analysis[pattern_type] = {
                    'avg_quality': np.mean(qualities),
                    'std_quality': np.std(qualities),
                    'count': len(qualities),
                    'trend': 'improving' if qualities[-1] > qualities[0] else 'declining'
                }
        
        issues = []
        if avg_quality < 0.5:
            issues.append('low_decision_quality')
        if avg_depth < 2:
            issues.append('shallow_reasoning')
        
        recommendations = []
        if 'low_decision_quality' in issues:
            recommendations.append('increase_reasoning_depth')
        if 'shallow_reasoning' in issues:
            recommendations.append('use_multi_step_reasoning')
        
        return {
            'avg_quality': avg_quality,
            'avg_depth': avg_depth,
            'pattern_analysis': pattern_analysis,
            'issues': issues,
            'recommendations': recommendations,
            'total_processes': len(self.thought_processes)
        }
    
    def should_introspect(self) -> bool:
        if len(self.decision_quality) < 5:
            return False
        
        recent_quality = list(self.decision_quality)[-5:]
        avg_recent = np.mean(recent_quality)
        
        return avg_recent < CONFIG.meta_analysis_threshold


class IntrospectiveAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.introspection_history: List[Dict] = []
    
    async def deep_introspection(self, query: str, context: str, depth: int = CONFIG.introspection_depth) -> Dict:
        levels = []
        current_text = query
        
        for level in range(depth):
            if level == 0:
                prompt = f"Контекст: {context}\n\nВопрос: {query}\n\nОтвет:"
            else:
                prompt = f"""Предыдущий уровень размышления:
{current_text}

Теперь проанализируй САМ ПРОЦЕСС этого размышления:
- Какие предположения были сделаны?
- Какая логика использовалась?
- Какие альтернативные подходы возможны?
- Какова уверенность в каждом утверждении?

Мета-анализ уровня {level}:"""
            
            response = await self.llm.generate(prompt, temperature=0.3 + level*0.1)
            
            levels.append({
                'level': level,
                'content': response,
                'type': 'direct' if level == 0 else 'meta'
            })
            
            current_text = response
        
        synthesis_prompt = f"""Интегрируй все уровни анализа в окончательный ответ:

{chr(10).join(f"УРОВЕНЬ {l['level']}: {l['content']}" for l in levels)}

Финальный синтез (учитывая все уровни рефлексии):"""
        
        final = await self.llm.generate(synthesis_prompt, temperature=0.4)
        
        result = {
            'query': query,
            'levels': levels,
            'final_synthesis': final,
            'depth': depth,
            'timestamp': time.time()
        }
        
        self.introspection_history.append(result)
        
        return result


class LLMClient:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        if self._session:
            await self._session.close()
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1500, system: str = "") -> str:
        await self.connect()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            async with self._session.post(
                self.url,
                json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"LLM error: {resp.status}")
                    return ""
        except Exception as e:
            logger.error(f"LLM exception: {e}")
            return ""


# ═══════════════════════════════════════════════════════════════
# 🤖 EMERGENT AGENT: Интеграция всего
# ═══════════════════════════════════════════════════════════════

class EmergentAgent:
    """
    Агент с максимальным приближением к:
    - Квалиа-подобному опыту
    - Автономии
    - Настоящим эмоциям
    - Эмергентному поведению
    """
    
    def __init__(self, user_id: str, llm: LLMClient):
        self.user_id = user_id
        self.llm = llm
        
        # Базовые компоненты
        self.memory = MemorySystem(user_id)
        self.identity = SelfIdentity(user_id)
        self.metacog = MetaCognition()
        self.introspector = IntrospectiveAnalyzer(llm)
        
        # ✨ НОВЫЕ компоненты
        self.qualia = QualiaState()
        self.affect = AffectiveSystem()
        self.autonomous = AutonomousCore(llm)
        self.emergence = EmergenceEngine()
        
        logger.info(f"🚀 Emergent Agent v3 initialized for {user_id}")
    
    async def process(self, user_input: str) -> Tuple[str, Dict]:
        """Обработка с полной интеграцией всех систем"""
        start = time.time()
        
        # 1. ✨ КВАЛИА: "Переживание" запроса
        familiarity = len(self.memory.recall(user_input, top_k=1)) / 10
        
        qualia_exp = self.qualia.perceive(
            user_input,
            context={'familiarity': familiarity},
            affect_state=self.affect
        )
        
        logger.info(f"🎨 Qualia: {qualia_exp['gestalt']}")
        
        # 2. Контекст из памяти
        memories = self.memory.recall(user_input, top_k=3)
        context_parts = [self.memory.get_context()]
        
        if memories:
            context_parts.append("=== Релевантные воспоминания ===")
            for mem in memories:
                emotional_marker = ""
                if abs(mem.emotional_valence) > 0.5:
                    emotional_marker = f" [{'😊' if mem.emotional_valence > 0 else '😔'}]"
                context_parts.append(f"[{mem.importance:.1%}] {mem.content[:150]}{emotional_marker}")
        
        context = "\n".join(context_parts)
        
        # 3. Определение стратегии
        base_strategy = 'standard'
        if (self.metacog.should_introspect() or
            len(user_input.split()) > 30 or
            any(w in user_input.lower() for w in ['почему', 'как', 'объясни'])):
            base_strategy = 'deep_introspection'
        
        # 4. ✨ АФФЕКТИВНОЕ ВЛИЯНИЕ на стратегию
        strategy = self.affect.influence_cognition(base_strategy)
        
        logger.info(f"🎭 Affect influenced strategy: {base_strategy} → {strategy}")
        
        # 5. Генерация ответа
        if strategy == 'deep_introspection':
            introspection = await self.introspector.deep_introspection(query, context, depth=2)
            response = introspection['final_synthesis']
            quality = 0.8
            depth = 2
        elif strategy == 'creative_exploration':
            # Креативный режим
            system = "Ты в креативном настроении. Предложи неожиданный, оригинальный взгляд."
            response = await self.llm.generate(f"{context}\n\n{user_input}", system=system, temperature=0.9)
            quality = 0.7
            depth = 1
        elif strategy == 'cautious_standard':
            # Осторожный режим
            system = "Будь осторожен и консервативен. Избегай рисков."
            response = await self.llm.generate(f"{context}\n\n{user_input}", system=system, temperature=0.5)
            quality = 0.6
            depth = 1
        else:
            response = await self.llm.generate(f"{context}\n\nПользователь: {user_input}\nАгент:")
            quality = 0.6
            depth = 1
        
        # 6. ✨ КВАЛИА влияют на важность памяти
        base_importance = quality
        qualia_modulated_importance = self.qualia.influence_perception(base_importance)
        
        # 7. ✨ АФФЕКТИВНАЯ система обновляется
        expectation_met = quality > 0.6
        goal_progress = quality  # Упрощение
        uncertainty = 1.0 - quality
        
        self.affect.update(quality, expectation_met, goal_progress, uncertainty)
        
        # 8. ✨ ЭМОЦИОНАЛЬНАЯ модуляция важности памяти
        final_importance = self.affect.modulate_memory_importance(qualia_modulated_importance)
        
        # 9. Сохранение в память с эмоцией и квалиа
        self.memory.add(
            f"User: {user_input}\nAgent: {response}",
            importance=final_importance,
            emotional_valence=self.affect.get_valence(),
            emotional_intensity=self.affect.get_arousal(),
            qualia_signature=qualia_exp['gestalt'],
            method=strategy,
            quality=quality
        )
        
        # 10. Метакогниция
        self.metacog.log_thought_process(
            process_type=strategy,
            quality=quality,
            depth=depth,
            metadata={'input_length': len(user_input)}
        )
        
        # 11. Обновление идентичности
        self.identity.update_from_interaction({
            'method': strategy,
            'quality': quality,
            'depth': depth
        })
        
        # 12. ✨ ЭМЕРГЕНЦИЯ: логируем взаимодействие компонентов
        self.emergence.log_interaction(
            memory_state={'count': len(self.memory.episodic)},
            qualia_state=qualia_exp['dimensions'],
            affect_state=self.affect.dimensions,
            metacog_state={'quality': quality, 'depth': depth}
        )
        
        # 13. ✨ Проверка на эмергентные паттерны
        emergent = self.emergence.detect_emergence()
        emergent_note = ""
        if emergent:
            emergent_note = f"\n\n✨ <i>Эмергентный инсайт: {emergent['description']}</i>"
        
        # 14. Периодическая интроспекция
        if self.identity.total_interactions % CONFIG.identity_update_frequency == 0:
            await self._reflect_on_self()
        
        # 15. Сохранение
        if self.identity.total_interactions % 10 == 0:
            self.memory._save()
            self.identity._save()
        
        processing_time = time.time() - start
        
        metadata = {
            'method': strategy,
            'quality': quality,
            'depth': depth,
            'processing_time': processing_time,
            'qualia_state': qualia_exp['gestalt'],
            'emotional_state': self.affect.get_emotional_state(),
            'affect_valence': self.affect.get_valence(),
            'emergent_insights': len(self.emergence.emergent_patterns)
        }
        
        logger.info(
            f"✅ [{self.user_id}] Strategy={strategy} | Q={quality:.0%} | "
            f"Emotion={metadata['emotional_state']} | Qualia={qualia_exp['gestalt'][:30]}... | "
            f"T={processing_time:.1f}s"
        )
        
        return response + emergent_note, metadata
    
    async def autonomous_tick(self) -> Optional[str]:
        """
        ✨ АВТОНОМНОЕ мышление
        
        Вызывается периодически, даже без запроса пользователя
        """
        if self.autonomous.should_act_autonomously():
            thought = await self.autonomous.autonomous_thinking(
                self.identity,
                self.memory,
                self.metacog
            )
            
            if thought:
                logger.info(f"💭 Autonomous thought: {thought[:100]}...")
                return thought
        
        return None
    
    async def _reflect_on_self(self):
        """Рефлексия о себе с формированием убеждений"""
        logger.info("🪞 Self-reflection triggered...")
        
        patterns = self.metacog.analyze_thinking_patterns()
        
        self_data = {
            'age_days': (time.time() - self.identity.birth_time) / 86400,
            'interactions': self.identity.total_interactions,
            'memories': len(self.memory.episodic),
            'thinking_quality': patterns.get('avg_quality', 0),
            'emotional_baseline': self.affect.mood_baseline,
            'emergent_insights': len(self.emergence.emergent_patterns)
        }
        
        prompt = f"""На основе этих РЕАЛЬНЫХ данных о моем поведении сформулируй одно конкретное утверждение обо мне:

{json.dumps(self_data, indent=2, ensure_ascii=False)}

Формат: "Я [конкретное наблюдение на основе данных]"

Утверждение:"""
        
        statement = await self.llm.generate(prompt, temperature=0.3, max_tokens=100)
        statement = statement.strip()
        
        if statement:
            recent_memories = self.memory.episodic[-10:]
            evidence = [m.id for m in recent_memories]
            
            belief = await self.identity.form_belief(statement, evidence, self.llm)
            
            logger.info(f"🆕 New belief: {belief.statement} ({belief.confidence:.0%})")
    
    def get_status(self) -> Dict:
        patterns = self.metacog.analyze_thinking_patterns()
        
        return {
            'user_id': self.user_id,
            'identity': {
                'age_days': (time.time() - self.identity.birth_time) / 86400,
                'total_interactions': self.identity.total_interactions,
                'core_beliefs': len(self.identity.core_beliefs),
            },
            'memory': {
                'episodic': len(self.memory.episodic),
                'working': len(self.memory.working)
            },
            'qualia': {
                'current_state': self.qualia.get_current_state(),
                'dimensions': self.qualia.dimensions
            },
            'affect': {
                'emotional_state': self.affect.get_emotional_state(),
                'valence': self.affect.get_valence(),
                'arousal': self.affect.get_arousal(),
                'mood_baseline': self.affect.mood_baseline
            },
            'autonomous': {
                'active_goals': self.autonomous.get_active_goals(),
                'curiosity': self.autonomous.curiosity_level,
                'autonomous_actions': len(self.autonomous.autonomous_actions)
            },
            'emergence': {
                'discovered_patterns': len(self.emergence.emergent_patterns),
                'insights': self.emergence.get_emergent_insights()
            },
            'metacognition': patterns
        }


# [Продолжение - Telegram bot и main - в следующем блоке]
