"""
🔤 models/tokenizer.py — Advanced BPE Tokenizer
"""

import gzip
import pickle
from pathlib import Path
from typing import List, Optional

# Optional fast tokenizer
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("⚠️  tokenizers library not available — falling back to word tokenizer")


class AdvancedBPETokenizer:
    """BPE tokenizer with a word-level fallback."""

    SPECIAL_TOKENS: dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self, vocab_size: int = 50_000) -> None:
        self.vocab_size = vocab_size
        self.tokenizer: Optional["Tokenizer"] = None
        self.special_tokens = dict(self.SPECIAL_TOKENS)

        if TOKENIZERS_AVAILABLE:
            self._init_bpe()
        else:
            self._init_fallback()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_bpe(self) -> None:
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.add_special_tokens(list(self.special_tokens.keys()))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", self.special_tokens["<BOS>"]),
                ("<EOS>", self.special_tokens["<EOS>"]),
            ],
        )

    def _init_fallback(self) -> None:
        self.word_to_id: dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_word: dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.next_id: int = len(self.SPECIAL_TOKENS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text).ids
            except Exception:
                tokens = self._fallback_encode(text)
        else:
            tokens = self._fallback_encode(text)

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.special_tokens["<PAD>"]] * (max_length - len(tokens))
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            if skip_special:
                tokens = [t for t in tokens if t >= len(self.SPECIAL_TOKENS)]
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special)

        words = []
        for t in tokens:
            w = self.id_to_word.get(t, "<UNK>")
            if not skip_special or w not in self.SPECIAL_TOKENS:
                words.append(w)
        return " ".join(words)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            self.tokenizer.save(str(path / "tokenizer.json"))
        else:
            with gzip.open(path / "tokenizer_fallback.pkl.gz", "wb") as fh:
                pickle.dump(
                    {
                        "word_to_id": self.word_to_id,
                        "id_to_word": self.id_to_word,
                        "next_id": self.next_id,
                    },
                    fh,
                )

    def load(self, path: Path) -> bool:
        if (path / "tokenizer.json").exists() and TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
            return True
        if (path / "tokenizer_fallback.pkl.gz").exists():
            with gzip.open(path / "tokenizer_fallback.pkl.gz", "rb") as fh:
                state = pickle.load(fh)
            self.word_to_id = state["word_to_id"]
            self.id_to_word = state["id_to_word"]
            self.next_id = state["next_id"]
            return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fallback_encode(self, text: str) -> List[int]:
        words = text.lower().split()
        unk = self.special_tokens["<UNK>"]
        return (
            [self.special_tokens["<BOS>"]]
            + [self.word_to_id.get(w, unk) for w in words]
            + [self.special_tokens["<EOS>"]]
        )
