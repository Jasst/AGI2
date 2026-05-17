"""
🧠 models/transformer.py — Transformer Architecture + Temporal Embeddings
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════
# ⏰ TEMPORAL EMBEDDINGS
# ══════════════════════════════════════════════════════════════

class TemporalEmbeddings(nn.Module):
    """Encodes current wall-clock time as an embedding vector."""

    def __init__(self, time_dim: int = 64) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_scale = 10_000.0

        self.circadian_emb = nn.Embedding(24, time_dim)   # hour of day
        self.weekday_emb = nn.Embedding(7, time_dim)       # day of week
        self.month_emb = nn.Embedding(12, time_dim)        # month

        self.register_buffer("birth_timestamp", torch.tensor(time.time()))

    # ------------------------------------------------------------------
    def _time_features(self) -> dict:
        now = datetime.now()
        return {
            "hour": now.hour,
            "weekday": now.weekday(),
            "month": now.month - 1,
            "seconds_since_birth": int(time.time() - self.birth_timestamp.item()),
        }

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """Returns (batch_size, time_dim)."""
        feat = self._time_features()
        dev = next(self.parameters()).device

        hour_e = self.circadian_emb(torch.tensor([feat["hour"]], device=dev)).expand(batch_size, -1)
        wd_e = self.weekday_emb(torch.tensor([feat["weekday"]], device=dev)).expand(batch_size, -1)
        mo_e = self.month_emb(torch.tensor([feat["month"]], device=dev)).expand(batch_size, -1)

        # Sinusoidal continuous encoding
        secs = feat["seconds_since_birth"]
        pos = torch.arange(self.time_dim, device=dev).float()
        div = torch.exp(pos * -(np.log(self.time_scale) / self.time_dim))
        cont = torch.zeros(batch_size, self.time_dim, device=dev)
        cont[:, 0::2] = torch.sin(secs * div[0::2])
        cont[:, 1::2] = torch.cos(secs * div[1::2])

        return hour_e + wd_e + mo_e + cont


# ══════════════════════════════════════════════════════════════
# 🧱 TRANSFORMER BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k**0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, _ = x.shape

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = _split(self.W_q(x)), _split(self.W_k(x)), _split(self.W_v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(ctx)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════
# 🎓 STUDENT TRANSFORMER
# ══════════════════════════════════════════════════════════════

class AdvancedStudentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        d_ff: int = 4096,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
        use_temporal: bool = True,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_temporal = use_temporal

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)

        if use_temporal:
            self.temporal_emb = TemporalEmbeddings(time_dim)
            self.temporal_proj = nn.Linear(time_dim, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        # Weight tying
        self.out_proj.weight = self.token_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        dev = input_ids.device

        pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        if self.use_temporal:
            time_emb = self.temporal_proj(self.temporal_emb(B))  # (B, d_model)
            x = x + time_emb.unsqueeze(1)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.out_proj(self.norm(x))
        return logits if return_logits else F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 3,
    ) -> Tuple[torch.Tensor, float]:
        """Auto-regressive generation. Returns (generated_ids, mean_confidence)."""
        self.eval()
        generated = prompt_ids.clone()
        confidences: list[float] = []

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                kth_val = torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits = next_logits.masked_fill(next_logits < kth_val, -float("Inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                next_logits[remove.scatter(1, sorted_idx, remove)] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            confidences.append(probs.max().item())
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return generated, mean_conf
