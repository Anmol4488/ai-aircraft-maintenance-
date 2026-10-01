"""
train_smollm_classifier.py - Step 5: fine-tune SmolLM2-135M to read a defect report
and predict its system (ATA chapter).

How it works (plain words):
  * SmolLM2 normally predicts the next word. We put a small "classification head" on top
    (one layer that turns the model's summary of the text into 32 scores, one per system).
  * We keep the pretrained knowledge of English/technical text and only nudge it with our data.
  * We use a plain PyTorch loop (no Trainer) so every step is visible and version-proof.

Smoke test on a CPU (checks the pipeline works, takes a few minutes):
    python train_smollm_classifier.py --max-train 200 --max-eval 100 --epochs 1 --batch-size 4

Real run on an NVIDIA GPU (4 GB: keep --batch-size small and use --grad-ckpt):
    python train_smollm_classifier.py --epochs 2 --batch-size 8 --grad-ckpt

If you run out of GPU memory, add --lora (trains ~1% of the weights, needs less memory).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def make_loader(texts: list[str], labels: list[int], tok, batch_size: int, max_len: int, shuffle: bool):
    """Batches of (tokenised text, label). Padding is done per batch, so short reports stay cheap."""
    items = list(zip(texts, labels))

    def collate(batch):
        batch_texts, batch_labels = zip(*batch)
        enc = tok(list(batch_texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(batch_labels)
        return enc

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    preds, golds = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds += logits.argmax(-1).cpu().tolist()
        golds += labels.tolist()
    return accuracy_score(golds, preds), f1_score(golds, preds, average="macro")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("processed/text_dataset.parquet"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out-dir", type=Path, default=Path("models/smollm2_ata_classifier"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=160, help="tokens per report (longer ones are cut)")
    ap.add_argument("--lr", type=float, default=None, help="default: 2e-5 full fine-tune, 2e-4 with --lora")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--grad-ckpt", action="store_true", help="save GPU memory at the cost of speed")
    ap.add_argument("--lora", action="store_true", help="train small adapter layers only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(42)

    df = pd.read_parquet(args.data)
    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    def part(name: str, limit: int | None) -> tuple[list[str], list[int]]:
        p = df[df["split"] == name]
        if limit:
            p = p.head(limit)
        return p["text"].tolist(), [label2id[x] for x in p["label"]]

    x_train, y_train = part("train", args.max_train)
    x_val, y_val = part("val", args.max_eval)
    x_test, y_test = part("test", args.max_eval)
    print(f"train={len(x_train)} val={len(x_val)} test={len(x_test)} classes={len(classes)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:                      # the model needs a padding token to batch texts
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes), id2label=id2label, label2id=label2id
    )
    model.config.pad_token_id = tok.pad_token_id   # tells the model which position is the last real token
    model.float()  # the checkpoint loads as bfloat16; keep master weights in fp32 (autocast handles fp16 speed)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    model.to(device)

    lr = args.lr or (2e-4 if args.lora else 2e-5)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    train_loader = make_loader(x_train, y_train, tok, args.batch_size, args.max_len, shuffle=True)
    val_loader = make_loader(x_val, y_val, tok, args.batch_size * 2, args.max_len, shuffle=False)
    test_loader = make_loader(x_test, y_test, tok, args.batch_size * 2, args.max_len, shuffle=False)

    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(  # short warm-up, then linear decay to 0
        opt, lambda s: min((s + 1) / max(1, int(0.05 * total_steps)), max(0.0, (total_steps - s) / total_steps))
    )
    use_amp = device == "cuda"                     # fp16 speeds up + saves memory on NVIDIA GPUs
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stops rare huge updates
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"({(time.time() - start) / 60:.1f} min)")
        acc, f1 = evaluate(model, val_loader, device)
        print(f"== epoch {epoch} VALIDATION acc={acc:.3f} macro-F1={f1:.3f}")

    acc, f1 = evaluate(model, test_loader, device)
    print(f"\nTEST (2025+ reports): acc={acc:.3f} macro-F1={f1:.3f}")
    print("Compare with TF-IDF baseline: acc=0.904 macro-F1=0.699 (only meaningful for a full-size run)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    (args.out_dir / "labels.json").write_text(json.dumps(id2label), encoding="utf-8")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()