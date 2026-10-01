"""Wrappers around the two fine-tuned SmolLM2 models.

IMPORTANT: the prompt format and token limits below must match train_smollm_generator.py,
otherwise the model sees a different format than it was trained on and quality drops.

torch/transformers are imported inside the classes, so the API (and its tests) can run
without them when the language models are switched off.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

MAX_PROBLEM_TOKENS = 120   # same as training
MAX_FIX_TOKENS = 64        # same as training
CLASSIFIER_MAX_LEN = 160   # same as training

# The model learned to end fixes with phrases like "OPS CHECKED GOOD". Those are claims that a
# test was performed and passed - the model cannot know that, so we never show them.
VERIFICATION_CLAIM = re.compile(
    r"\b(OPS?\s*CHECK(?:ED)?|OPERATIONAL(?:LY)?\s*CHECK(?:ED)?|LEAK\s*(?:CHECK(?:ED)?|CK)|"
    r"FUNCTIONS?\s+(?:AS\s+REQUIRED|NORMAL(?:LY)?|GOOD)|RETURN(?:ED)?\s+TO\s+SERVICE|"
    r"NO\s+(?:FURTHER\s+)?DEFECTS?\s+NOTED|CHECKED?\s+GOOD|CHK\s+GOOD|SERVICEABLE)\b",
    flags=re.I,
)


def strip_verification_claims(text: str) -> tuple[str | None, bool]:
    """Remove sentences that claim a check was done. Returns (cleaned text or None, was_anything_removed)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    kept = [s for s in sentences if not VERIFICATION_CLAIM.search(s)]
    removed = len(kept) != len(sentences)
    return (" ".join(kept) if kept else None), removed


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class TextClassifier:
    """Report text -> most likely system (ATA chapter)."""

    def __init__(self, path: Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = _device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).float().to(self.device).eval()
        self.id2label = {int(k): v for k, v in json.loads((path / "labels.json").read_text()).items()}
        self._lock = threading.Lock()

    def predict(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        import torch
        enc = self.tok(text, truncation=True, max_length=CLASSIFIER_MAX_LEN, return_tensors="pt").to(self.device)
        with self._lock, torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits[0], dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.numel()))
        return [(self.id2label[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


class FixGenerator:
    """Problem description -> drafted fix (greedy decoding, so the same input gives the same output)."""

    def __init__(self, path: Path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = _device()
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path).float().to(self.device).eval()
        self.model.config.use_cache = True
        self._lock = threading.Lock()

    def _prompt_ids(self, problem: str) -> list[int]:
        p = self.tok("PROBLEM: " + problem, add_special_tokens=False)["input_ids"][:MAX_PROBLEM_TOKENS]
        return p + self.tok("\nFIX:", add_special_tokens=False)["input_ids"]

    def generate(self, problem: str) -> str:
        import torch
        ids = torch.tensor([self._prompt_ids(problem)], device=self.device)
        with self._lock, torch.no_grad():
            out = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_FIX_TOKENS,
                do_sample=False, pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()