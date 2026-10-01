"""gen_utils.py - tiny metric helpers shared by the baseline and the SmolLM2 generator."""

from __future__ import annotations

import re
from collections import Counter


def words(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+", text.upper())


def token_f1(pred: str, gold: str) -> float:
    """Word-overlap F1 between a generated fix and the real fix (0 = nothing shared, 1 = identical words)."""
    p, g = Counter(words(pred)), Counter(words(gold))
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def first_word(text: str) -> str:
    w = words(text)
    return w[0] if w else ""