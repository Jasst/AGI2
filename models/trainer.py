"""
🎓 models/trainer.py — Knowledge-Distillation Trainer with Experience Replay
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.transformer import AdvancedStudentTransformer
    from models.tokenizer import AdvancedBPETokenizer

from config import CONFIG

logger = logging.getLogger("AdvancedAgent_v4")


class AdvancedDistillationTrainer:
    def __init__(
        self,
        student_model: "AdvancedStudentTransformer",
        tokenizer: "AdvancedBPETokenizer",
        device: str = "cpu",
    ) -> None:
        self.student = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=CONFIG.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )

        # Mixed precision
        if CONFIG.mixed_precision:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.replay_buffer: deque = deque(maxlen=CONFIG.replay_buffer_size)
        self.training_steps: int = 0
        self.losses_history: deque = deque(maxlen=1_000)

    # ------------------------------------------------------------------
    def add_to_replay_buffer(self, prompt: str, teacher_response: str) -> None:
        self.replay_buffer.append(
            {"prompt": prompt, "response": teacher_response, "timestamp": time.time()}
        )

    # ------------------------------------------------------------------
    async def train_on_interaction(self, prompt: str, teacher_response: str) -> float:
        """One training step on a single (prompt, response) pair. Returns loss."""
        self.student.train()

        text = f"{prompt} {teacher_response}"
        input_ids = self.tokenizer.encode(text, max_length=CONFIG.max_seq_length)
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        src = tensor[:, :-1]          # model input
        tgt = tensor[:, 1:].clone()  # target labels

        self.optimizer.zero_grad()

        if self.scaler:
            with torch.amp.autocast("cuda"):
                logits = self.student(src, return_logits=True)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=0
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.student(src, return_logits=True)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()
        self.training_steps += 1
        loss_val = loss.item()
        self.losses_history.append(loss_val)

        logger.debug(f"Training step {self.training_steps} | loss={loss_val:.4f}")
        return loss_val

    # ------------------------------------------------------------------
    @property
    def avg_loss(self) -> float:
        return float(np.mean(list(self.losses_history))) if self.losses_history else 0.0
