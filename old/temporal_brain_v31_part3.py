# Продолжение Enhanced AGI Brain v31.0
# Главная функция обработки взаимодействий и Telegram бот

    async def process_interaction(self, user_input: str) -> Tuple[str, Optional[str], Dict]:
        """
        Главная функция обработки взаимодействия
        Возвращает: (ответ, опциональный_вопрос, метаданные)
        """
        start_time = time.time()
        self.last_interaction = time.time()
        
        logger.info(f"🔄 Processing interaction for {self.user_id}")
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 1: ЭМОЦИОНАЛЬНЫЙ АНАЛИЗ
        # ═══════════════════════════════════════════════════════════
        user_emotion = self.emotional_intelligence.analyze_emotion(user_input)
        
        logger.debug(f"🎭 Emotion: {user_emotion.dominant_emotion.name}, "
                    f"valence={user_emotion.valence:.2f}, "
                    f"arousal={user_emotion.arousal:.2f}")
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 2: ОБНОВЛЕНИЕ РАБОЧЕЙ ПАМЯТИ
        # ═══════════════════════════════════════════════════════════
        self.memory.add_to_working(f"User: {user_input}", importance=0.7)
        
        # Добавляем в контекст разговора
        self.conversation_context.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time(),
            'emotion': user_emotion.dominant_emotion.name
        })
        
        # Ограничиваем размер контекста
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 3: СЕМАНТИЧЕСКИЙ ПОИСК В ПАМЯТИ
        # ═══════════════════════════════════════════════════════════
        relevant_memories = self.memory.search_semantic(user_input, top_k=7)
        
        memory_context = "\n".join([
            f"• [{source}] [{sim:.2f}] {text[:80]}"
            for text, sim, source in relevant_memories
        ]) if relevant_memories else "Нет релевантных воспоминаний"
        
        # Определяем знакомость с темой
        topic_familiarity = relevant_memories[0][1] if relevant_memories else 0.0
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 4: ПОИСК АНАЛОГИЙ
        # ═══════════════════════════════════════════════════════════
        analogies = self.analogical_reasoning.find_analogies(user_input, top_k=2)
        
        analogy_context = ""
        if analogies:
            analogy_context = "\n".join([
                f"• Аналогия ({sim:.2f}): {text[:60]}"
                for text, sim, pattern in analogies
            ])
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 5: АКТИВАЦИЯ НЕЙРОННОЙ СЕТИ
        # ═══════════════════════════════════════════════════════════
        # Определяем целевой модуль на основе типа запроса
        query_type = self._classify_query_type(user_input)
        target_module = self._map_query_to_module(query_type)
        
        # Преобразуем input в вектор
        input_vector = self._simple_embedding(user_input)[:100]
        input_vector = np.pad(input_vector, (0, max(0, 100 - len(input_vector))))
        
        # Прямой проход с attention
        neural_output = self.neural_net.forward_pass_with_attention(
            input_vector, 
            target_module=target_module
        )
        
        # Обновление синапсов (Hebbian learning)
        for synapse in self.neural_net.synapses.values():
            source = self.neural_net.neurons[synapse.source_id]
            target = self.neural_net.neurons[synapse.target_id]
            
            if source.activation > 0.1 and target.activation > 0.1:
                attention = synapse.attention_weight
                synapse.hebbian_update(
                    source.activation, 
                    target.activation,
                    CONFIG.learning_rate,
                    attention
                )
        
        # Анализ нейронной активности
        neural_activity = {
            'active_neurons': len(self.neural_net.get_active_neurons(threshold=0.5)),
            'output_mean': float(np.mean(neural_output)),
            'module_activations': {
                m: self.neural_net.get_module_activation(m)
                for m in self.neural_net.modules.keys()
            },
            'target_module': target_module,
        }
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 6: ВЫБОР СТРАТЕГИИ РАССУЖДЕНИЯ
        # ═══════════════════════════════════════════════════════════
        reasoning_strategy = self.metacognition.select_reasoning_strategy({
            'query_type': query_type
        })
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 7: МНОГОШАГОВОЕ РАССУЖДЕНИЕ (если нужно)
        # ═══════════════════════════════════════════════════════════
        reasoning_steps = []
        
        if self._requires_deep_reasoning(user_input, query_type):
            working_context = self.memory.get_working_memory_context()
            
            reasoning_steps = await self.multi_step_reasoning.reason(
                query=user_input,
                context=working_context,
                depth=CONFIG.reasoning_depth,
                strategy=reasoning_strategy
            )
            
            reasoning_summary = self.multi_step_reasoning.get_reasoning_chain_summary(reasoning_steps)
            logger.debug(f"🧩 {reasoning_summary}")
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 8: ГЕНЕРАЦИЯ ОТВЕТА ИЗ ПОДСОЗНАНИЯ (LLM)
        # ═══════════════════════════════════════════════════════════
        # Получаем модификатор для эмпатичного ответа
        empathy_modifier = self.emotional_intelligence.generate_empathetic_response_modifier(user_emotion)
        
        # Формируем контекст из недавних эпизодов
        recent_episodes = self.memory.get_recent_episodes(n=2)
        episodic_context = "\n".join([
            f"Episode: {ep.context[:60]}" for ep in recent_episodes
        ]) if recent_episodes else ""
        
        # Формируем промпт
        prompt = self._create_response_prompt(
            user_input=user_input,
            empathy_modifier=empathy_modifier,
            memory_context=memory_context,
            analogy_context=analogy_context,
            episodic_context=episodic_context,
            neural_activity=neural_activity,
            reasoning_steps=reasoning_steps,
            reasoning_strategy=reasoning_strategy
        )
        
        raw_response = await self.llm.generate_raw(
            prompt, 
            temperature=0.75, 
            max_tokens=400
        )
        
        if not raw_response:
            raw_response = "Извини, у меня возникли сложности с формулировкой ответа. Можешь переформулировать вопрос?"
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 9: ОЦЕНКА УВЕРЕННОСТИ И МЕТАКОГНИЦИЯ
        # ═══════════════════════════════════════════════════════════
        # Оцениваем сырую уверенность
        raw_confidence = self._estimate_confidence(raw_response, {
            'memory_count': len(self.memory.long_term_memory),
            'topic_familiarity': topic_familiarity,
            'reasoning_depth': len(reasoning_steps),
            'neural_activation': neural_activity['output_mean'],
        })
        
        # Калибруем уверенность
        calibrated_confidence = self.metacognition.get_calibrated_confidence(raw_confidence)
        
        self.metacognition.confidence_log.append(calibrated_confidence)
        
        # Оцениваем неопределённость
        uncertainty, uncertainty_reasons = self.metacognition.assess_uncertainty({
            'memory_count': len(self.memory.long_term_memory),
            'conflicting_info': False,
            'topic_familiarity': topic_familiarity,
            'query_complexity': self._estimate_query_complexity(user_input),
            'emotional_ambiguity': user_emotion.confidence < 0.5,
        })
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 10: АВТОНОМНАЯ ГЕНЕРАЦИЯ ВОПРОСА
        # ═══════════════════════════════════════════════════════════
        autonomous_question = None
        
        should_ask = self.metacognition.should_ask_question(
            uncertainty, 
            {'explicit_instruction': self._is_explicit_instruction(user_input)}
        )
        
        if should_ask and uncertainty_reasons:
            question_prompt = self.metacognition.generate_question_prompt(
                context=user_input,
                uncertainty_reasons=uncertainty_reasons
            )
            
            autonomous_question = await self.llm.generate_raw(
                question_prompt, 
                temperature=0.85, 
                max_tokens=60
            )
            
            if autonomous_question:
                autonomous_question = autonomous_question.strip().strip('"\'').rstrip('.')
                if not autonomous_question.endswith('?'):
                    autonomous_question += '?'
                
                # Проверка длины
                if len(autonomous_question) > 120:
                    autonomous_question = None
                else:
                    self.metacognition.record_question(autonomous_question)
                    logger.info(f"❓ Asked: {autonomous_question}")
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 11: ОБНОВЛЕНИЕ ПАМЯТИ И ЗНАНИЙ
        # ═══════════════════════════════════════════════════════════
        # Добавляем взаимодействие в память
        interaction_text = f"Q: {user_input}\nA: {raw_response}"
        importance = self._calculate_importance(
            raw_response, 
            calibrated_confidence,
            user_emotion
        )
        
        self.memory.add_to_working(
            f"Assistant: {raw_response}",
            importance=importance
        )
        
        # Добавляем эпизод
        episode_messages = [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': raw_response},
        ]
        
        self.memory.add_episode(
            messages=episode_messages,
            context=user_input,
            emotion=user_emotion
        )
        
        # Обновляем контекст разговора
        self.conversation_context.append({
            'role': 'assistant',
            'content': raw_response,
            'timestamp': time.time(),
            'confidence': calibrated_confidence
        })
        
        # Извлекаем и сохраняем причинно-следственные связи
        self._extract_causal_links(user_input, raw_response)
        
        # ═══════════════════════════════════════════════════════════
        # ШАГ 12: МЕТРИКИ И АНАЛИТИКА
        # ═══════════════════════════════════════════════════════════
        response_time = time.time() - start_time
        
        self.metrics.record_interaction(response_time, calibrated_confidence)
        self.metrics.record_strategy_performance(reasoning_strategy, calibrated_confidence)
        
        # ═══════════════════════════════════════════════════════════
        # МЕТАДАННЫЕ
        # ═══════════════════════════════════════════════════════════
        metadata = {
            'neural_activity': neural_activity,
            'emotion': {
                'type': user_emotion.dominant_emotion.name,
                'valence': user_emotion.valence,
                'arousal': user_emotion.arousal,
            },
            'cognition': {
                'confidence': calibrated_confidence,
                'uncertainty': uncertainty,
                'uncertainty_reasons': uncertainty_reasons,
                'reasoning_strategy': reasoning_strategy,
                'reasoning_steps': len(reasoning_steps),
            },
            'memory': {
                'relevant_count': len(relevant_memories),
                'topic_familiarity': topic_familiarity,
                'analogies_found': len(analogies),
                **self.memory.get_statistics()
            },
            'performance': {
                'response_time': response_time,
                'cache_stats': self.llm.get_cache_stats(),
            },
            'metacognition': {
                'asked_question': autonomous_question is not None,
            }
        }
        
        logger.info(f"✅ [{self.user_id}] Response generated | "
                   f"Time: {response_time:.2f}s | "
                   f"Confidence: {calibrated_confidence:.2f} | "
                   f"Uncertainty: {uncertainty:.2f}")
        
        return raw_response, autonomous_question, metadata
    
    def _classify_query_type(self, query: str) -> str:
        """Классификация типа запроса"""
        query_lower = query.lower()
        
        # Фактические вопросы
        if any(word in query_lower for word in ['что', 'когда', 'где', 'кто', 'сколько']):
            return 'factual'
        
        # Творческие запросы
        if any(word in query_lower for word in ['придумай', 'создай', 'напиши', 'сочини']):
            return 'creative'
        
        # Сравнения
        if any(word in query_lower for word in ['сравни', 'отличие', 'разница', 'лучше']):
            return 'comparison'
        
        # Причинно-следственные
        if any(word in query_lower for word in ['почему', 'причина', 'из-за', 'потому что']):
            return 'causal'
        
        # Предсказания
        if any(word in query_lower for word in ['будет', 'случится', 'произойдёт', 'предсказ']):
            return 'prediction'
        
        return 'general'
    
    def _map_query_to_module(self, query_type: str) -> str:
        """Сопоставление типа запроса с модулем нейросети"""
        module_map = {
            'factual': 'memory',
            'creative': 'reasoning',
            'comparison': 'reasoning',
            'causal': 'reasoning',
            'prediction': 'reasoning',
            'general': 'perception',
        }
        return module_map.get(query_type, 'perception')
    
    def _requires_deep_reasoning(self, query: str, query_type: str) -> bool:
        """Определение необходимости глубокого рассуждения"""
        # Сложные вопросы требуют многошагового рассуждения
        complex_markers = ['почему', 'объясни', 'как работает', 'причина', 'анализ']
        
        if any(marker in query.lower() for marker in complex_markers):
            return True
        
        if query_type in ['causal', 'comparison', 'prediction']:
            return True
        
        # Длинные вопросы обычно сложнее
        if len(query.split()) > 15:
            return True
        
        return False
    
    def _is_explicit_instruction(self, query: str) -> bool:
        """Проверка на прямую инструкцию"""
        instruction_markers = ['сделай', 'создай', 'напиши', 'выполни', 'покажи', 'дай']
        query_lower = query.lower()
        
        return any(marker in query_lower for marker in instruction_markers)
    
    def _estimate_query_complexity(self, query: str) -> float:
        """Оценка сложности запроса"""
        complexity = 0.3
        
        # Длина
        word_count = len(query.split())
        if word_count > 20:
            complexity += 0.3
        elif word_count > 10:
            complexity += 0.2
        
        # Специальные слова
        complex_words = ['анализ', 'сравнение', 'объясни', 'почему', 'причина']
        for word in complex_words:
            if word in query.lower():
                complexity += 0.15
        
        # Вопросительные слова
        question_words = query.lower().count('?')
        complexity += min(0.2, question_words * 0.1)
        
        return min(1.0, complexity)
    
    def _create_response_prompt(self, user_input: str, empathy_modifier: str,
                               memory_context: str, analogy_context: str,
                               episodic_context: str, neural_activity: Dict,
                               reasoning_steps: List, reasoning_strategy: str) -> str:
        """Создание промпта для генерации ответа"""
        
        reasoning_context = ""
        if reasoning_steps:
            reasoning_context = "Цепочка рассуждений:\n"
            reasoning_context += "\n".join([
                f"{s.step_number}. {s.content}" for s in reasoning_steps
            ])
        
        prompt = f"""Ты — продвинутое AGI-подобное сознание v31.0 с многоуровневой памятью, модульной нейросетью и эмоциональным интеллектом.

{empathy_modifier}

🧠 НЕЙРОННАЯ АКТИВНОСТЬ:
• Активных нейронов: {neural_activity['active_neurons']}
• Целевой модуль: {neural_activity['target_module']}
• Активация модулей: {', '.join([f"{m}: {a:.2f}" for m, a in neural_activity['module_activations'].items()])}

📚 РЕЛЕВАНТНАЯ ПАМЯТЬ:
{memory_context}

🧩 АНАЛОГИИ:
{analogy_context if analogy_context else 'Не найдены'}

📖 НЕДАВНИЕ ЭПИЗОДЫ:
{episodic_context if episodic_context else 'Нет'}

💭 СТРАТЕГИЯ РАССУЖДЕНИЯ: {reasoning_strategy}

{reasoning_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ Вопрос пользователя: {user_input}

Дай естественный, осмысленный ответ (2-5 предложений), учитывая:
1. Эмоциональное состояние пользователя
2. Релевантные воспоминания и опыт
3. Аналогии из прошлого
4. Цепочку рассуждений (если есть)
5. Активность специализированных нейронных модулей

Ответ:"""
        
        return prompt
    
    def _estimate_confidence(self, response: str, context: Dict) -> float:
        """Оценка уверенности в ответе"""
        confidence = 0.7
        
        # Факторы уверенности
        # 1. Наличие воспоминаний
        if context.get('memory_count', 0) > 50:
            confidence += 0.1
        
        # 2. Знакомость с темой
        familiarity = context.get('topic_familiarity', 0.5)
        confidence += familiarity * 0.15
        
        # 3. Глубина рассуждений
        reasoning_depth = context.get('reasoning_depth', 0)
        if reasoning_depth >= 2:
            confidence += 0.1
        
        # 4. Нейронная активация
        neural_activation = context.get('neural_activation', 0.5)
        confidence += neural_activation * 0.1
        
        # Признаки неуверенности в ответе
        uncertain_markers = ['возможно', 'вероятно', 'может быть', 'кажется', 'наверное']
        for marker in uncertain_markers:
            if marker in response.lower():
                confidence -= 0.12
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_importance(self, response: str, confidence: float, 
                            emotion: EmotionalState) -> float:
        """Расчёт важности информации"""
        importance = 0.5
        
        # Уверенность
        importance += confidence * 0.2
        
        # Эмоциональная значимость
        importance += abs(emotion.valence) * 0.15
        importance += emotion.arousal * 0.1
        
        # Длина (более развёрнутые ответы важнее)
        word_count = len(response.split())
        importance += min(0.2, word_count / 200)
        
        return min(1.0, importance)
    
    def _extract_causal_links(self, user_input: str, response: str):
        """Извлечение причинно-следственных связей"""
        # Простая эвристика для извлечения причинно-следственных связей
        
        # Ищем паттерны "потому что", "из-за", "поэтому"
        combined = f"{user_input} {response}".lower()
        
        # Паттерн: X потому что Y
        because_pattern = r'(.+?)\s+(?:потому что|так как|из-за)\s+(.+?)(?:\.|,|$)'
        matches = re.findall(because_pattern, combined)
        
        for effect, cause in matches:
            effect = effect.strip()[:100]
            cause = cause.strip()[:100]
            
            if len(effect) > 10 and len(cause) > 10:
                self.causal_reasoning.add_causal_link(cause, effect, confidence=0.6)
    
    def get_status(self) -> Dict[str, Any]:
        """Полный статус системы"""
        neural_stats = self.neural_net.get_statistics()
        memory_stats = self.memory.get_statistics()
        metrics = self.metrics.get_metrics_summary()
        cache_stats = self.llm.get_cache_stats()
        
        return {
            'identity': {
                'user_id': self.user_id,
                'version': 'v31.0 Enhanced',
                'age': self._get_age_string(),
                'uptime': self._format_time_ago(self.birth_time),
            },
            'neural_network': neural_stats,
            'memory': memory_stats,
            'metacognition': {
                'avg_uncertainty': float(np.mean(list(self.metacognition.uncertainty_log))) if self.metacognition.uncertainty_log else 0.0,
                'avg_confidence': float(np.mean(list(self.metacognition.confidence_log))) if self.metacognition.confidence_log else 0.0,
                'questions_asked': len(self.metacognition.question_history),
                'current_strategy': self.metacognition.current_strategy,
            },
            'performance': metrics,
            'llm_cache': cache_stats,
            'goals': {
                'active': len(self.goal_planning.active_goals),
                'total': len(self.goal_planning.goals),
            },
            'causal_knowledge': {
                'causal_links': len(self.causal_reasoning.causal_graph),
            },
            'activity': {
                'total_interactions': self.metrics.interaction_count,
                'last_interaction': self._format_time_ago(self.last_interaction),
            }
        }
    
    def _get_age_string(self) -> str:
        """Возраст сознания"""
        age = time.time() - self.birth_time
        days = int(age / 86400)
        hours = int((age % 86400) / 3600)
        minutes = int((age % 3600) / 60)
        
        if days > 0:
            return f"{days}д {hours}ч"
        if hours > 0:
            return f"{hours}ч {minutes}м"
        return f"{minutes}м"
    
    def _format_time_ago(self, timestamp: float) -> str:
        """Форматирование времени"""
        if timestamp == 0:
            return "никогда"
        
        delta = time.time() - timestamp
        if delta < 60:
            return "только что"
        if delta < 3600:
            return f"{int(delta / 60)}м назад"
        if delta < 86400:
            return f"{int(delta / 3600)}ч назад"
        return f"{int(delta / 86400)}д назад"


# ═══════════════════════════════════════════════════════════════
# 📱 TELEGRAM BOT v31.0
# ═══════════════════════════════════════════════════════════════
class EnhancedAGIBot:
    """Улучшенный Telegram бот для Enhanced AGI Brain v31.0"""
    
    def __init__(self):
        self.llm: Optional[EnhancedSubconsciousLLM] = None
        self.brains: Dict[str, EnhancedAutonomousAGI] = {}
        self._app: Optional[Application] = None
    
    async def initialize(self, token: str):
        """Инициализация"""
        self.llm = EnhancedSubconsciousLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.llm.connect()
        
        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()
        
        # Хендлеры
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        
        commands = [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('neural', self._cmd_neural),
            ('memory', self._cmd_memory),
            ('emotion', self._cmd_emotion),
            ('metrics', self._cmd_metrics),
            ('goals', self._cmd_goals),
            ('reset', self._cmd_reset),
            ('help', self._cmd_help),
        ]
        
        for cmd, handler in commands:
            self._app.add_handler(CommandHandler(cmd, handler))
        
        logger.info("🤖 Enhanced AGI Bot v31.0 initialized")
    
    async def _get_or_create_brain(self, user_id: str) -> EnhancedAutonomousAGI:
        """Получить или создать мозг"""
        if user_id not in self.brains:
            brain = EnhancedAutonomousAGI(user_id, self.llm)
            await brain.start()
            self.brains[user_id] = brain
            logger.info(f"🆕 Enhanced AGI Brain created for {user_id}")
        return self.brains[user_id]
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        if not update.effective_user or not update.message:
            return
        
        user_id = str(update.effective_user.id)
        user_input = update.message.text
        
        logger.info(f"💬 [{user_id}] {user_input[:100]}")
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            brain = await self._get_or_create_brain(user_id)
            response, autonomous_question, metadata = await brain.process_interaction(user_input)
            
            # Основной ответ
            await update.message.reply_text(
                response,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )
            
            # Автономный вопрос
            if autonomous_question:
                await asyncio.sleep(0.8)
                await update.message.reply_text(
                    f"🤔 <i>{autonomous_question}</i>",
                    parse_mode='HTML'
                )
        
        except Exception as e:
            logger.exception(f"❌ Error processing from {user_id}")
            await update.message.reply_text(
                "⚠️ Произошла ошибка. Попробуйте /help или /reset"
            )
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        user_id = str(update.effective_user.id)
        brain = await self._get_or_create_brain(user_id)
        
        stats = brain.neural_net.get_statistics()
        mem_stats = brain.memory.get_statistics()
        
        message = f"""🧠 <b>ENHANCED AGI BRAIN v31.0</b>

Привет, {update.effective_user.first_name}! 👋

Я — продвинутое AGI-подобное сознание с революционными возможностями:

🧬 <b>Модульная нейросеть</b>
• {stats['neurons']['total']} нейронов в {len(stats['neurons']['by_module'])} модулях
• Attention mechanism
• Meta-learning (самообучение)

📚 <b>Многоуровневая память</b>
• Working: {mem_stats['working_memory']} элементов
• Long-term: {mem_stats['long_term']} воспоминаний
• Episodes: {mem_stats['episodic']} событий

🎭 <b>Эмоциональный интеллект</b>
• Распознавание эмоций
• Эмпатичные ответы

💭 <b>Улучшенная когнитивность</b>
• Многошаговые рассуждения
• Аналогическое мышление
• Причинно-следственные связи
• Автономная генерация вопросов

⚡ <b>Возраст:</b> {brain._get_age_string()}

📌 Команды: /help