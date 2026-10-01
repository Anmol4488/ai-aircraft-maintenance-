"""
train_smollm_generator.py - Step 7: teach SmolLM2-135M to WRITE the fix for a problem.

Training text looks like:   PROBLEM: <problem text>\nFIX: <fix text><end>
The model learns to continue after "FIX:". The loss is computed ONLY on the fix part
(we mask the problem tokens), so it learns to write fixes, not to copy problems.

Smoke test (checks the pipeline, a few minutes):
    python train_smollm_generator.py --max-train 300 --max-eval 100 --n-gen 20 --epochs 1 --batch-size 8

Real run on a 4 GB GPU:
    python train_smollm_generator.py --max-train 40000 --epochs 1 --batch-size 16 --grad-ckpt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_utils import first_word, token_f1

MAX_PROBLEM_TOKENS = 120
MAX_FIX_TOKENS = 64


def prompt_ids(tok, problem: str) -> list[int]:
    """The exact same prompt is used for training and for generating."""
    p = tok("PROBLEM: " + problem, add_special_tokens=False)["input_ids"][:MAX_PROBLEM_TOKENS]
    return p + tok("\nFIX:", add_special_tokens=False)["input_ids"]


def encode_example(tok, problem: str, fix: str) -> tuple[list[int], list[int]]:
    p = prompt_ids(tok, problem)
    t = tok(" " + fix, add_special_tokens=False)["input_ids"][:MAX_FIX_TOKENS] + [tok.eos_token_id]
    return p + t, [-100] * len(p) + t            # -100 = "ignore these positions in the loss"


def make_loader(tok, problems, fixes, batch_size: int, shuffle: bool) -> DataLoader:
    data = [encode_example(tok, p, f) for p, f in zip(problems, fixes)]

    def collate(batch):
        longest = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), longest), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            labels[i, :len(y)] = torch.tensor(y)
            mask[i, :len(x)] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(model(**batch).loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def generate_fixes(model, tok, problems: list[str], device: str, batch_size: int = 16) -> list[str]:
    """Greedy decoding (always pick the most likely next word) - repeatable and safe for a first test."""
    model.eval()
    model.config.use_cache = True
    outs = []
    for start in range(0, len(problems), batch_size):
        chunk = [prompt_ids(tok, p) for p in problems[start:start + batch_size]]
        longest = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), longest), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), longest), dtype=torch.long)
        for i, c in enumerate(chunk):                      # LEFT-pad so generation starts right after the prompt
            ids[i, longest - len(c):] = torch.tensor(c)
            mask[i, longest - len(c):] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                             max_new_tokens=MAX_FIX_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row in gen[:, longest:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("processed/gen_dataset.parquet"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out-dir", type=Path, default=Path("models/smollm2_fix_generator"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--max-eval", type=int, default=2000, help="validation examples for the loss")
    ap.add_argument("--n-gen", type=int, default=500, help="test problems to actually generate fixes for")
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(42)

    df = pd.read_parquet(args.data)
    train = df[df["split"] == "train"].sample(min(args.max_train, (df["split"] == "train").sum()), random_state=1)
    val = df[df["split"] == "val"].head(args.max_eval)
    test = df[df["split"] == "test"].sample(args.n_gen, random_state=0)
    print(f"train={len(train)} val={len(val)} test-for-generation={len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.float()                                   # fp32 master weights (checkpoint loads as bfloat16)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    train_loader = make_loader(tok, train["problem"].tolist(), train["action"].tolist(), args.batch_size, True)
    val_loader = make_loader(tok, val["problem"].tolist(), val["action"].tolist(), args.batch_size, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(1, int(0.05 * total)), max(0.0, (total - s) / total)))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"validation loss before training: {eval_loss(model, val_loader, device):.3f}")
    step, start = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {loss.item():.3f} ({(time.time() - start) / 60:.1f} min)")
        print(f"== epoch {epoch} validation loss {eval_loss(model, val_loader, device):.3f}")

    # ---- evaluation by actually writing fixes ----
    if args.grad_ckpt:
        model.gradient_checkpointing_disable()
    preds = generate_fixes(model, tok, test["problem"].tolist(), device)
    golds = test["action"].tolist()
    f1 = np.mean([token_f1(p, g) for p, g in zip(preds, golds)])
    verb = np.mean([first_word(p) == first_word(g) for p, g in zip(preds, golds)])
    print(f"\nTEST (2025+ reports, {len(test)} problems): word-F1={f1:.3f}  verb-match={verb:.3f}")
    print("Bars to beat (3000-report test sample): constant 0.030 / 0.105, nearest-similar-problem 0.271 / 0.377\n")
    for p, pr, g in list(zip(test["problem"], preds, golds))[:8]:
        print(f"PROBLEM: {p}\nMODEL:   {pr}\nREAL:    {g}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()