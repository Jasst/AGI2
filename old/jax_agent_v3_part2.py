# Продолжение jax_agent_v3.py - часть 2

# ══════════════════════════════════════════════════════════════
# 🎓 JAX TRAINER
# ══════════════════════════════════════════════════════════════

class JAXTrainer:
    """Тренер на JAX"""

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer

        # Создание модели
        self.model = TransformerModel(
            vocab_size=CONFIG.vocab_size,
            d_model=CONFIG.d_model,
            num_heads=CONFIG.n_heads,
            num_layers=CONFIG.n_layers,
            d_ff=CONFIG.d_ff,
            max_seq_length=CONFIG.max_seq_length,
            dropout_rate=CONFIG.dropout_rate,
            use_temporal=CONFIG.use_temporal,
            time_dim=CONFIG.time_dim
        )

        # Инициализация
        rng = random.PRNGKey(0)
        self.state = create_train_state(rng, self.model, CONFIG.learning_rate)

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=CONFIG.replay_buffer_size)

        # Stats
        self.training_steps = 0
        self.total_loss = 0.0
        self.losses_history: deque = deque(maxlen=1000)

        # Подсчёт параметров
        params_count = sum(x.size for x in jax.tree_util.tree_leaves(self.state.params))
        logger.info(f"🧠 JAX Model: {params_count / 1e6:.1f}M parameters")

    def add_to_replay_buffer(self, prompt: str, teacher_response: str):
        """Добавление в replay buffer"""
        self.replay_buffer.append({
            'prompt': prompt,
            'response': teacher_response,
            'timestamp': time.time()
        })

    async def train_on_interaction(self, prompt: str, teacher_response: str) -> float:
        """Обучение на взаимодействии"""
        self.add_to_replay_buffer(prompt, teacher_response)

        # Подготовка данных
        text = f"{prompt} {teacher_response}"
        input_ids = self.tokenizer.encode(text, max_length=CONFIG.max_seq_length)

        # Входы и метки
        input_tensor = jnp.array([input_ids[:-1]])  # [batch=1, seq_len-1]
        labels = jnp.array([input_ids[1:]])  # [batch=1, seq_len-1]

        # Training step
        self.state, loss = train_step(self.state, input_tensor, labels)

        # Stats
        self.training_steps += 1
        loss_val = float(loss)
        self.total_loss += loss_val
        self.losses_history.append(loss_val)

        return loss_val

    async def train_on_replay_buffer(self, num_batches: int = 5) -> float:
        """Обучение на replay buffer"""
        if len(self.replay_buffer) < CONFIG.batch_size:
            return 0.0

        total_loss = 0.0

        for _ in range(num_batches):
            batch_indices = np.random.choice(
                len(self.replay_buffer),
                size=min(CONFIG.batch_size, len(self.replay_buffer)),
                replace=False
            )

            batch = [self.replay_buffer[i] for i in batch_indices]

            for item in batch:
                loss = await self.train_on_interaction(
                    item['prompt'],
                    item['response']
                )
                total_loss += loss

        return total_loss / (num_batches * CONFIG.batch_size)

    def generate(
            self,
            prompt_ids: np.ndarray,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            eos_token_id: int = 3
    ) -> Tuple[np.ndarray, float]:
        """Генерация текста"""
        generated = list(prompt_ids)
        confidences = []

        for _ in range(max_length):
            # Подготовка входа
            input_tensor = jnp.array([generated[-CONFIG.max_seq_length:]])

            # Forward pass (JIT-compiled!)
            probs = generate_step(self.state, input_tensor)
            probs = np.array(probs[0])  # Конвертируем в NumPy

            # Temperature
            if temperature != 1.0:
                logits = np.log(probs + 1e-10)
                logits = logits / temperature
                probs = np.exp(logits) / np.sum(np.exp(logits))

            # Top-k filtering
            if top_k > 0:
                top_k_indices = np.argsort(probs)[-top_k:]
                probs_filtered = np.zeros_like(probs)
                probs_filtered[top_k_indices] = probs[top_k_indices]
                probs = probs_filtered / probs_filtered.sum()

            # Sample
            next_token = np.random.choice(len(probs), p=probs)

            # Confidence
            confidence = float(probs[next_token])
            confidences.append(confidence)

            # Append
            generated.append(int(next_token))

            # Check EOS
            if next_token == eos_token_id:
                break

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return np.array(generated), avg_confidence

    def get_statistics(self) -> Dict:
        """Статистика обучения"""
        return {
            'training_steps': self.training_steps,
            'avg_loss': np.mean(list(self.losses_history)) if self.losses_history else 0.0,
            'recent_loss': self.losses_history[-1] if self.losses_history else 0.0,
            'replay_buffer_size': len(self.replay_buffer),
        }


# ══════════════════════════════════════════════════════════════
# 🤖 JAX AUTONOMOUS AGENT
# ══════════════════════════════════════════════════════════════

class JAXAutonomousAgent:
    """Автономный агент на JAX"""

    def __init__(self, user_id: str, teacher: TeacherLLM):
        self.user_id = user_id
        self.teacher = teacher

        # Tokenizer
        self.tokenizer = BPETokenizer(vocab_size=CONFIG.vocab_size)

        # Trainer (содержит модель)
        self.trainer = JAXTrainer(self.tokenizer)

        # Autonomy
        self.teacher_usage_probability = CONFIG.initial_teacher_usage
        self.autonomy_level = 0.0

        # Stats
        self.total_interactions = 0
        self.teacher_calls = 0
        self.autonomous_responses = 0
        self.successful_autonomous = 0

        # Paths
        self.user_dir = CONFIG.base_dir / 'models' / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self._load_state()

        logger.info(f"🚀 JAX Agent created for {user_id}")

    def _count_parameters(self) -> int:
        """Подсчёт параметров"""
        return sum(x.size for x in jax.tree_util.tree_leaves(self.trainer.state.params))

    def _load_state(self):
        """Загрузка состояния"""
        # Tokenizer
        tokenizer_path = self.user_dir / 'tokenizer'
        if (tokenizer_path / 'tokenizer.json').exists() and TOKENIZERS_AVAILABLE:
            from tokenizers import Tokenizer as TokenizerLoader
            self.tokenizer.tokenizer = TokenizerLoader.from_file(str(tokenizer_path / 'tokenizer.json'))
            logger.info("✅ Tokenizer loaded")

        # Model state
        model_path = self.user_dir / 'model_state.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    saved_state = pickle.load(f)

                # Restore JAX state
                self.trainer.state = train_state.TrainState.create(
                    apply_fn=self.trainer.model.apply,
                    params=saved_state['params'],
                    tx=self.trainer.state.tx
                )

                self.total_interactions = saved_state.get('total_interactions', 0)
                self.teacher_calls = saved_state.get('teacher_calls', 0)
                self.autonomous_responses = saved_state.get('autonomous_responses', 0)
                self.autonomy_level = saved_state.get('autonomy_level', 0.0)
                self.teacher_usage_probability = saved_state.get('teacher_usage_probability',
                                                                 CONFIG.initial_teacher_usage)

                logger.info(
                    f"✅ Model loaded: {self.total_interactions} interactions, autonomy={self.autonomy_level:.2%}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        # Tokenizer
        tokenizer_path = self.user_dir / 'tokenizer'
        tokenizer_path.mkdir(exist_ok=True)

        if TOKENIZERS_AVAILABLE and hasattr(self.tokenizer, 'tokenizer'):
            self.tokenizer.tokenizer.save(str(tokenizer_path / 'tokenizer.json'))

        # Model state
        model_path = self.user_dir / 'model_state.pkl'
        saved_state = {
            'params': self.trainer.state.params,
            'total_interactions': self.total_interactions,
            'teacher_calls': self.teacher_calls,
            'autonomous_responses': self.autonomous_responses,
            'autonomy_level': self.autonomy_level,
            'teacher_usage_probability': self.teacher_usage_probability,
        }

        with open(model_path, 'wb') as f:
            pickle.dump(saved_state, f)

        logger.debug(f"💾 State saved: autonomy={self.autonomy_level:.2%}")

    def _should_use_teacher(self) -> bool:
        return np.random.random() < self.teacher_usage_probability

    def _update_autonomy(self, was_successful: bool):
        if was_successful:
            self.autonomy_level = min(1.0, self.autonomy_level + CONFIG.autonomy_growth_rate)
            self.successful_autonomous += 1
        else:
            self.autonomy_level = max(0.0, self.autonomy_level - CONFIG.autonomy_growth_rate * 0.5)

        self.teacher_usage_probability = max(
            CONFIG.min_teacher_usage,
            1.0 - self.autonomy_level
        )

    async def generate_autonomous(self, prompt: str) -> Tuple[str, float]:
        """Автономная генерация"""
        # Encode
        prompt_ids = self.tokenizer.encode(prompt, max_length=CONFIG.max_seq_length // 2)

        # Generate
        generated_ids, confidence = self.trainer.generate(
            prompt_ids,
            max_length=CONFIG.max_seq_length // 2,
            temperature=0.8,
            eos_token_id=self.tokenizer.special_tokens.get('<EOS>', 3)
        )

        # Decode
        response = self.tokenizer.decode(generated_ids, skip_special=True)

        return response, confidence

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        """Главная обработка"""
        start_time = time.time()
        self.total_interactions += 1

        response = ""
        confidence = 0.0
        used_teacher = False
        autonomous_attempt = False

        # Decision: use teacher?
        use_teacher = self._should_use_teacher()

        if not use_teacher:
            autonomous_attempt = True
            self.autonomous_responses += 1

            response, confidence = await self.generate_autonomous(user_input)

            if confidence < CONFIG.confidence_threshold:
                logger.info(f"🤔 Low confidence ({confidence:.2f}) - asking teacher")
                use_teacher = True

        if use_teacher:
            self.teacher_calls += 1
            used_teacher = True

            teacher_response, _ = await self.teacher.generate(user_input)

            if teacher_response:
                response = teacher_response
                confidence = 1.0

                # Train
                if self.total_interactions % CONFIG.training_frequency == 0:
                    loss = await self.trainer.train_on_interaction(user_input, teacher_response)
                    logger.debug(f"📚 Trained, loss={loss:.4f}")
                else:
                    self.trainer.add_to_replay_buffer(user_input, teacher_response)
            else:
                response = "Извините, возникла проблема с генерацией ответа."
                confidence = 0.0

        # Update tokenizer
        if self.total_interactions % 100 == 0:
            self.tokenizer.train([user_input, response])

        # Replay training
        if self.total_interactions % (CONFIG.training_frequency * 5) == 0:
            replay_loss = await self.trainer.train_on_replay_buffer(num_batches=3)
            logger.debug(f"🔄 Replay training, loss={replay_loss:.4f}")

        # Update autonomy
        was_successful = confidence >= CONFIG.confidence_threshold
        if autonomous_attempt:
            self._update_autonomy(was_successful)

        # Save
        if self.total_interactions % CONFIG.save_frequency == 0:
            self._save_state()

        # Metadata
        response_time = time.time() - start_time

        metadata = {
            'used_teacher': used_teacher,
            'autonomous_attempt': autonomous_attempt,
            'confidence': confidence,
            'autonomy_level': self.autonomy_level,
            'teacher_usage_prob': self.teacher_usage_probability,
            'response_time': response_time,
            'training_stats': self.trainer.get_statistics(),
            'model_size': f"{self._count_parameters() / 1e6:.1f}M",
            'backend': jax.default_backend(),
        }

        logger.info(
            f"✅ [{self.user_id}] "
            f"Teacher={'Yes' if used_teacher else 'No'} | "
            f"Conf={confidence:.2f} | "
            f"Autonomy={self.autonomy_level:.1%} | "
            f"T={response_time:.1f}s | "
            f"Backend={jax.default_backend()}"
        )

        return response, metadata

    def get_status(self) -> Dict:
        """Статус агента"""
        return {
            'user_id': self.user_id,
            'model_parameters': self._count_parameters(),
            'jax_backend': jax.default_backend(),
            'jax_devices': str(jax.devices()),

            'autonomy': {
                'level': self.autonomy_level,
                'teacher_usage_probability': self.teacher_usage_probability,
                'success_rate': self.successful_autonomous / max(1, self.autonomous_responses),
            },

            'interactions': {
                'total': self.total_interactions,
                'teacher_calls': self.teacher_calls,
                'autonomous_responses': self.autonomous_responses,
                'successful_autonomous': self.successful_autonomous,
            },

            'training': self.trainer.get_statistics(),

            'tokenizer': {
                'type': 'BPE' if TOKENIZERS_AVAILABLE else 'Fallback',
                'vocab_size': CONFIG.vocab_size,
            }
        }


# ══════════════════════════════════════════════════════════════
# 🤖 TELEGRAM BOT
# ══════════════════════════════════════════════════════════════

class JAXBot:
    """Telegram бот с JAX агентом"""

    def __init__(self):
        self.teacher: Optional[TeacherLLM] = None
        self.agents: Dict[str, JAXAutonomousAgent] = {}
        self._app: Optional[Application] = None

    async def initialize(self, token: str):
        self.teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
        await self.teacher.connect()

        defaults = Defaults(parse_mode='HTML')
        self._app = Application.builder().token(token).defaults(defaults).build()

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        for cmd, handler in [
            ('start', self._cmd_start),
            ('status', self._cmd_status),
            ('help', self._cmd_help),
        ]:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("🤖 JAX Bot initialized")

    async def _get_or_create_agent(self, user_id: str) -> JAXAutonomousAgent:
        if user_id not in self.agents:
            self.agents[user_id] = JAXAutonomousAgent(user_id, self.teacher)
        return self.agents[user_id]

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or not update.message:
            return

        user_id = str(update.effective_user.id)
        user_input = update.message.text

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            agent = await self._get_or_create_agent(user_id)
            response, metadata = await agent.process_interaction(user_input)

            footer = f"\n\n<i>"
            if metadata['used_teacher']:
                footer += "👨‍🏫 Teacher"
            else:
                footer += f"🤖 Auto ({metadata['confidence']:.0%})"

            footer += f" | 🎯 {metadata['autonomy_level']:.0%}"
            footer += f" | ⚡ {metadata['model_size']}"
            footer += f" | 🔥 JAX-{metadata['backend']}"
            footer += "</i>"

            await update.message.reply_text(
                response + footer,
                parse_mode='HTML',
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

        except Exception as e:
            logger.exception(f"Error from {user_id}")
            await update.message.reply_text("⚠️ Произошла ошибка")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🔥 <b>JAX AUTONOMOUS AGENT v3.0</b>

Привет! Я на <b>JAX</b> - современной альтернативе PyTorch от Google!

<b>✨ ПРЕИМУЩЕСТВА JAX:</b>

⚡ <b>Скорость</b>
• JIT компиляция
• XLA оптимизация
• Быстрее PyTorch на многих задачах

🎮 <b>GPU/TPU</b>
• Отличная поддержка
• Автоматическая оптимизация
• Работает на {jax.default_backend()}

🧠 <b>Модель</b>
• {CONFIG.n_layers} слоёв
• {CONFIG.d_model} размерность
• Temporal embeddings

<b>Команды:</b> /help | /status"""

        await update.message.reply_text(message)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        agent = await self._get_or_create_agent(user_id)
        status = agent.get_status()

        message = f"""🔥 <b>JAX AGENT STATUS</b>

<b>🧠 Модель</b>
• Параметров: {status['model_parameters']:,}
• Backend: {status['jax_backend']}
• Устройства: {status['jax_devices'][:50]}...

<b>🎯 Автономность</b>
• Уровень: {status['autonomy']['level']:.1%}
• Учитель: {status['autonomy']['teacher_usage_probability']:.1%}
• Успех: {status['autonomy']['success_rate']:.1%}

<b>📊 Взаимодействия</b>
• Всего: {status['interactions']['total']}
• К учителю: {status['interactions']['teacher_calls']}
• Автономных: {status['interactions']['autonomous_responses']}

<b>📚 Обучение</b>
• Шагов: {status['training']['training_steps']}
• Потеря: {status['training']['avg_loss']:.4f}

<b>🔤 Tokenizer</b>
• Тип: {status['tokenizer']['type']}
• Словарь: {status['tokenizer']['vocab_size']:,}"""

        await update.message.reply_text(message)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """🔥 <b>JAX AGENT v3 — СПРАВКА</b>

<b>ЧТО ТАКОЕ JAX?</b>

JAX - это библиотека от Google для:
• Быстрых вычислений
• Автоматической дифференциации
• GPU/TPU оптимизации

<b>ПОЧЕМУ JAX vs PyTorch?</b>

✅ Работает на Python 3.8-3.11
✅ JIT компиляция → быстрее
✅ Проще для понимания
✅ Отличная поддержка GPU

<b>ВОЗМОЖНОСТИ:</b>

🧠 Transformer модель
⏰ Внутреннее время
🎓 Дистилляция знаний
📈 Рост автономности

<b>КОМАНДЫ:</b>
• /start — приветствие
• /status — статус
• /help — эта справка

<b>Backend:</b> {jax.default_backend()}"""

        await update.message.reply_text(message)

    async def start_polling(self):
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ JAX Bot started")

    async def shutdown(self):
        logger.info("🛑 Shutting down...")

        for user_id, agent in self.agents.items():
            try:
                agent._save_state()
                logger.info(f"💾 Saved: {user_id}")
            except Exception as e:
                logger.error(f"Error saving {user_id}: {e}")

        if self.teacher:
            await self.teacher.close()

        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        logger.info("✅ Shutdown complete")


# ══════════════════════════════════════════════════════════════
# 🚀 MAIN
# ══════════════════════════════════════════════════════════════

async def main():
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  🔥 JAX AUTONOMOUS AGENT v3.0                                 ║
║     Powered by Google JAX - альтернатива PyTorch              ║
╚═══════════════════════════════════════════════════════════════╝

🔥 JAX VERSION: {jax.__version__}
🎮 BACKEND: {jax.default_backend()}
🖥️  DEVICES: {jax.devices()}

✅ ПРЕИМУЩЕСТВА JAX:
   • JIT компиляция → Быстрее
   • XLA оптимизация → Эффективнее
   • Работает на Python 3.8-3.11
   • Отлично с GPU/TPU

🧠 МОДЕЛЬ:
   • {CONFIG.n_layers} слоёв
   • {CONFIG.d_model} размерность
   • {CONFIG.vocab_size:,} токенов
   • Temporal embeddings

🎯 УСТРОЙСТВО: {jax.default_backend().upper()}
""")

    if not CONFIG.telegram_token:
        logger.critical("❌ TELEGRAM_TOKEN не установлен!")
        return 1

    bot = JAXBot()

    try:
        await bot.initialize(CONFIG.telegram_token)
        await bot.start_polling()

        logger.info("🌀 JAX AGENT АКТИВЕН")
        logger.info(f"🔥 Backend: {jax.default_backend()}")
        logger.info(f"🎮 Devices: {jax.devices()}")
        logger.info("🛑 Ctrl+C для остановки")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n👋 Остановка...")
    except Exception as e:
        logger.critical(f"❌ Ошибка: {e}", exc_info=True)
        return 1
    finally:
        await bot.shutdown()
        logger.info("👋 До встречи!")
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)