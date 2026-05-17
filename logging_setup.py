"""
📊 logging_setup.py — Centralized logging configuration
"""

import sys
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(base_dir: Path) -> logging.Logger:
    logger = logging.getLogger("AdvancedAgent_v4")
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_path = base_dir / "logs" / f"agent_v4_{datetime.now():%Y%m%d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
