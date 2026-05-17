"""
🤖 agent.py — Advanced Autonomous Agent (per-user)
"""

import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple

from config import CONFIG
from models import (
    AdvancedBPETokenizer,
    AdvancedStudentTransformer,
    TeacherLLM,
    AdvancedDistillationTrainer,
)

logger = logging.getLogger("AdvancedAgent_v4")


class AdvancedAutonomousAgent:
    """
    Per-user agent that progressively reduces reliance on the teacher
    as its student model improves through distillation.
    """

    def __init__(self, user_id: str, teacher: TeacherLLM) -> None:
        self.user_id = user_id
        self.teacher = teacher

        # --- Build model components ---
        self.tokenizer = AdvancedBPETokenizer(vocab_size=CONFIG.vocab_size)
        self.student_model = AdvancedStudentTransformer(
            vocab_size=CONFIG.vocab_size,
            d_model=CONFIG.d_model,
            n_heads=CONFIG.n_heads,
            n_layers=CONFIG.n_layers,
            d_ff=CONFIG.d_ff,
            max_seq_length=CONFIG.max_seq_length,
            dropout=CONFIG.dropout,
            use_temporal=CONFIG.temporal_embeddings,
            time_dim=CONFIG.time_embedding_dim,
        )
        self.trainer = AdvancedDistillationTrainer(
            self.student_model, self.tokenizer, device=CONFIG.device
        )

        # --- Autonomy state ---
        self.teacher_usage_prob: float = CONFIG.initial_teacher_usage
        self.autonomy_level: float = 0.0
        self.total_interactions: int = 0
        self.teacher_calls: int = 0
        self.autonomous_responses: int = 0
        self.successful_autonomous: int = 0

        # --- Persistence directory ---
        self.user_dir: Path = CONFIG.base_dir / "models" / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self._load_state()
        logger.info(
            f"🚀 Agent v4 for '{user_id}' | "
            f"{self._param_count() / 1e6:.1f}M params | "
            f"device={CONFIG.device}"
        )

    # ════════════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════════════

    async def process_interaction(self, user_input: str) -> Tuple[str, Dict]:
        start = time.time()
        self.total_interactions += 1

        response = ""
        confidence = 0.0
        used_teacher = False
        autonomous_attempt = False

        # 1. Try autonomous generation first (if teacher not required)
        if not self._should_use_teacher():
            autonomous_attempt = True
            self.autonomous_responses += 1
            response, confidence = await self._generate_autonomous(user_input)

            # Fall back to teacher if not confident enough
            if confidence < CONFIG.confidence_threshold:
                used_teacher = True

        if not autonomous_attempt or used_teacher:
            used_teacher = True
            self.teacher_calls += 1
            teacher_response, _ = await self.teacher.generate(user_input)

            if teacher_response:
                response = teacher_response
                confidence = 1.0

                # Train or buffer
                if self.total_interactions % CONFIG.training_frequency == 0:
                    await self.trainer.train_on_interaction(user_input, teacher_response)
                else:
                    self.trainer.add_to_replay_buffer(user_input, teacher_response)
            else:
                response = "Извините, возникла проблема с генерацией ответа."
                confidence = 0.0

        # 2. Update autonomy score
        if autonomous_attempt:
            self._update_autonomy(confidence >= CONFIG.confidence_threshold)

        # 3. Periodic checkpoint
        if self.total_interactions % CONFIG.save_frequency == 0:
            self._save_state()

        metadata = {
            "used_teacher": used_teacher,
            "autonomous_attempt": autonomous_attempt,
            "confidence": confidence,
            "autonomy_level": self.autonomy_level,
            "teacher_usage_prob": self.teacher_usage_prob,
            "response_time": round(time.time() - start, 3),
            "total_interactions": self.total_interactions,
            "autonomous_responses": self.autonomous_responses,
            "model_size": f"{self._param_count() / 1e6:.1f}M",
            "training_stats": {
                "training_steps": self.trainer.training_steps,
                "avg_loss": round(self.trainer.avg_loss, 4),
            },
            "autonomy": {
                "level": self.autonomy_level,
                "teacher_usage_probability": self.teacher_usage_prob,
                "success_rate": self.successful_autonomous / max(1, self.autonomous_responses),
            },
        }

        logger.info(
            f"[{self.user_id}] teacher={'Y' if used_teacher else 'N'} "
            f"conf={confidence:.2f} autonomy={self.autonomy_level:.1%}"
        )
        return response, metadata

    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        return {
            "user_id": self.user_id,
            "model_parameters": self._param_count(),
            "model_size_mb": round(self._param_count() * 4 / 1e6, 2),
            "autonomy": {
                "level": self.autonomy_level,
                "teacher_usage_probability": self.teacher_usage_prob,
                "success_rate": self.successful_autonomous / max(1, self.autonomous_responses),
            },
            "interactions": {
                "total": self.total_interactions,
                "teacher_calls": self.teacher_calls,
                "autonomous_responses": self.autonomous_responses,
                "successful_autonomous": self.successful_autonomous,
            },
            "training": {
                "training_steps": self.trainer.training_steps,
                "avg_loss": round(self.trainer.avg_loss, 4),
                "replay_buffer_size": len(self.trainer.replay_buffer),
            },
            "model_size": f"{self._param_count() / 1e6:.1f}M",
            "training_stats": {
                "training_steps": self.trainer.training_steps,
                "avg_loss": round(self.trainer.avg_loss, 4),
            },
            "features": {
                "rag_enabled": CONFIG.rag_enabled,
                "meta_learning": CONFIG.meta_learning_enabled,
                "temporal_embeddings": CONFIG.temporal_embeddings,
                "mixed_precision": CONFIG.mixed_precision,
            },
        }

    # ════════════════════════════════════════════════════════════════
    # Internal helpers
    # ════════════════════════════════════════════════════════════════

    async def _generate_autonomous(self, prompt: str) -> Tuple[str, float]:
        prompt_ids = self.tokenizer.encode(prompt, max_length=CONFIG.max_seq_length // 2)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=CONFIG.device)
        generated, confidence = self.student_model.generate(
            prompt_tensor,
            max_new_tokens=CONFIG.max_seq_length // 2,
            temperature=0.8,
            eos_token_id=self.tokenizer.special_tokens.get("<EOS>", 3),
        )
        text = self.tokenizer.decode(generated[0].cpu().tolist(), skip_special=True)
        return text, confidence

    def _should_use_teacher(self) -> bool:
        return np.random.random() < self.teacher_usage_prob

    def _update_autonomy(self, was_successful: bool) -> None:
        if was_successful:
            self.autonomy_level = min(1.0, self.autonomy_level + CONFIG.autonomy_growth_rate)
            self.successful_autonomous += 1
        else:
            self.autonomy_level = max(0.0, self.autonomy_level - CONFIG.autonomy_growth_rate * 0.5)

        self.teacher_usage_prob = max(CONFIG.min_teacher_usage, 1.0 - self.autonomy_level)

    def _param_count(self) -> int:
        return sum(p.numel() for p in self.student_model.parameters())

    # ════════════════════════════════════════════════════════════════
    # Persistence
    # ════════════════════════════════════════════════════════════════

    def _save_state(self) -> None:
        torch.save(
            {
                "model_state_dict": self.student_model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "total_interactions": self.total_interactions,
                "teacher_calls": self.teacher_calls,
                "autonomous_responses": self.autonomous_responses,
                "successful_autonomous": self.successful_autonomous,
                "autonomy_level": self.autonomy_level,
                "teacher_usage_prob": self.teacher_usage_prob,
            },
            self.user_dir / "student_model.pt",
        )
        self.tokenizer.save(self.user_dir / "tokenizer")
        logger.info(f"💾 State saved for '{self.user_id}'")

    def _load_state(self) -> None:
        ckpt_path = self.user_dir / "student_model.pt"
        tok_path = self.user_dir / "tokenizer"

        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=CONFIG.device)
                self.student_model.load_state_dict(ckpt["model_state_dict"])
                self.trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self.total_interactions = ckpt.get("total_interactions", 0)
                self.teacher_calls = ckpt.get("teacher_calls", 0)
                self.autonomous_responses = ckpt.get("autonomous_responses", 0)
                self.successful_autonomous = ckpt.get("successful_autonomous", 0)
                self.autonomy_level = ckpt.get("autonomy_level", 0.0)
                self.teacher_usage_prob = ckpt.get("teacher_usage_prob", CONFIG.initial_teacher_usage)
                logger.info(
                    f"✅ Loaded checkpoint: {self.total_interactions} interactions, "
                    f"autonomy={self.autonomy_level:.1%}"
                )
            except Exception as exc:
                logger.error(f"Failed to load checkpoint: {exc}")

        if tok_path.exists():
            self.tokenizer.load(tok_path)
