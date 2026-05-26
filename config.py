"""
⚙️ config.py — Advanced Agent v4 Configuration
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AdvancedConfig:
    """Продвинутая конфигурация v4"""

    # === Teacher LLM ===
    lm_studio_url: str = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/v1/chat/completions")
    lm_studio_key: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

    # === Server ===
    host: str = "0.0.0.0"
    port: int = 5000

    # === Transformer Architecture ===
    vocab_size: int = 50_000
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    d_ff: int = 4096
    max_seq_length: int = 1024
    dropout: float = 0.1

    # === Auto-scaling ===
    auto_scale_model: bool = True
    max_gpu_memory_gb: float = 8.0

    # === Training ===
    learning_rate: float = 5e-5
    meta_learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.7

    # === Meta-Learning ===
    meta_learning_enabled: bool = True
    few_shot_examples: int = 5
    meta_batch_size: int = 4
    inner_loop_steps: int = 3

    # === RAG ===
    rag_enabled: bool = False  # overridden at runtime after chromadb check
    rag_top_k: int = 5
    rag_chunk_size: int = 512
    rag_embedding_dim: int = 768

    # === Temporal Embeddings ===
    temporal_embeddings: bool = True
    time_embedding_dim: int = 64
    circadian_cycle_hours: int = 24
    memory_decay_rate: float = 0.01

    # === Autonomy ===
    initial_teacher_usage: float = 1.0
    min_teacher_usage: float = 0.05
    autonomy_growth_rate: float = 0.002
    confidence_threshold: float = 0.75

    # === Memory / Checkpoints ===
    replay_buffer_size: int = 50_000
    training_frequency: int = 5
    save_frequency: int = 50

    # === Paths ===
    base_dir: Path = Path("advanced_agent_v4_data")

    # === Device ===
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = torch.cuda.is_available()

    def __post_init__(self):
        # Resolve rag_enabled at runtime
        try:
            import chromadb  # noqa: F401
            self.rag_enabled = True
        except ImportError:
            self.rag_enabled = False

        # Create directory tree
        for subdir in ["models", "memory", "logs", "checkpoints", "rag", "tokenizer"]:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Auto-scale to available GPU memory
        if self.auto_scale_model and self.device == "cuda":
            self._auto_scale_to_gpu()

    # ------------------------------------------------------------------
    def _auto_scale_to_gpu(self) -> None:
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU Memory: {gpu_mem_gb:.2f} GB")

            estimated_params = (
                self.vocab_size * self.d_model * 2
                + self.n_layers * (4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff)
            )
            estimated_mem_gb = estimated_params * 4 / 1e9 * 1.5

            if estimated_mem_gb > gpu_mem_gb * 0.7:
                scale = (gpu_mem_gb * 0.7) / estimated_mem_gb
                self.d_model = int(self.d_model * scale**0.5)
                self.d_ff = int(self.d_ff * scale**0.5)
                self.n_layers = max(6, int(self.n_layers * scale**0.25))
                # Keep d_model divisible by n_heads
                self.d_model = (self.d_model // self.n_heads) * self.n_heads
                print(
                    f"⚙️  Auto-scaled: d_model={self.d_model}, "
                    f"n_layers={self.n_layers}, d_ff={self.d_ff}"
                )
        except Exception as exc:
            print(f"⚠️  Auto-scaling failed: {exc}")


# Singleton config — import this everywhere
CONFIG = AdvancedConfig()
