"""
Fine-tune LayoutLMv3 on exported annotations.

Single machine:
    python train.py --data dataset/train.json --output models/layoutlmv3-finetuned

Multi-machine (run on EVERY machine — see README for full instructions):
    accelerate launch --config_file accelerate_config.yaml train.py \
        --data dataset/train.json --output models/layoutlmv3-finetuned

TensorBoard:
    tensorboard --logdir models/layoutlmv3-finetuned/runs
    then open http://localhost:6006
"""

import argparse
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    get_linear_schedule_with_warmup,
)

BASE_MODEL = "microsoft/layoutlmv3-base"
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DocumentDataset(Dataset):
    def __init__(self, examples: list[dict], processor: LayoutLMv3Processor):
        self.examples  = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = Image.open(ex["image_path"]).convert("RGB")
        encoding = self.processor(
            image,
            ex["words"],
            boxes=ex["bboxes"],
            word_labels=ex["ner_tags"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, accelerator, id2label: dict) -> dict:
    """Run eval loop, return dict of metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            preds  = outputs.logits.argmax(dim=-1).cpu().numpy()   # (B, seq)
            labels = batch["labels"].cpu().numpy()                  # (B, seq)

            for pred_row, label_row in zip(preds, labels):
                mask = label_row != IGNORE_INDEX
                all_preds.extend(pred_row[mask].tolist())
                all_labels.extend(label_row[mask].tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    label_names = [id2label[i] for i in sorted(id2label)]
    report = classification_report(
        all_labels, all_preds,
        labels=list(sorted(id2label.keys())),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    macro_f1    = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    per_class = {
        name: {
            "f1":        report[name]["f1-score"],
            "precision": report[name]["precision"],
            "recall":    report[name]["recall"],
            "support":   report[name]["support"],
        }
        for name in label_names
        if name in report
    }

    return {
        "loss":         avg_loss,
        "accuracy":     float(accuracy),
        "macro_f1":     macro_f1,
        "weighted_f1":  weighted_f1,
        "per_class":    per_class,
        "all_preds":    all_preds,
        "all_labels":   all_labels,
    }


def confusion_matrix_image(all_labels, all_preds, id2label: dict) -> "torch.Tensor":
    """Render a confusion matrix and return as a CHW float tensor for TensorBoard."""
    label_names = [id2label[i] for i in sorted(id2label)]
    cm = confusion_matrix(all_labels, all_preds, labels=list(sorted(id2label.keys())))

    # Normalise rows to [0,1]
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm.astype(float) / row_sums

    n = len(label_names)
    fig_size = max(12, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1)   # CHW


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    accelerator = Accelerator()

    with open(args.data) as f:
        examples = json.load(f)
    with open(Path(args.data).parent / "label2id.json") as f:
        label2id = json.load(f)
    id2label   = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    # Train / val split
    n_val   = max(1, int(len(examples) * args.val_split))
    n_train = len(examples) - n_val
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(examples))
    train_examples = [examples[i] for i in idx[:n_train]]
    val_examples   = [examples[i] for i in idx[n_train:]]

    steps_per_epoch = max(1, len(train_examples) // args.batch_size)
    total_steps     = steps_per_epoch * args.epochs

    if accelerator.is_main_process:
        print(f"\n{'='*56}")
        print(f"  Total examples    : {len(examples)}")
        print(f"  Train / val       : {n_train} / {n_val}")
        print(f"  Labels            : {num_labels}")
        print(f"  Device            : {accelerator.device}")
        print(f"  Num processes     : {accelerator.num_processes}")
        print(f"  Epochs            : {args.epochs}")
        print(f"  Batch size        : {args.batch_size}")
        print(f"  Steps/epoch       : {steps_per_epoch}")
        print(f"  Total steps       : {total_steps}")
        print(f"  Learning rate     : {args.lr}")
        print(f"{'='*56}\n")

        writer = SummaryWriter(log_dir=str(Path(args.output) / "runs"))

    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels
    )

    train_dataset = DocumentDataset(train_examples, processor)
    val_dataset   = DocumentDataset(val_examples,   processor)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            step_start = time.time()

            outputs = model(**batch)
            loss    = outputs.loss
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_loss    = loss.item()
            epoch_loss  += step_loss
            global_step += 1

            if accelerator.is_main_process:
                step_time = time.time() - step_start
                lr_now    = scheduler.get_last_lr()[0]

                print(
                    f"Epoch {epoch+1:>2}/{args.epochs} "
                    f"| Step {step+1:>3}/{steps_per_epoch} "
                    f"| Loss {step_loss:.4f} "
                    f"| LR {lr_now:.2e} "
                    f"| GradNorm {float(grad_norm):.3f} "
                    f"| {step_time:.1f}s/step",
                    flush=True,
                )

                writer.add_scalar("Train/loss_step",     step_loss,          global_step)
                writer.add_scalar("Train/lr",            lr_now,             global_step)
                writer.add_scalar("Train/grad_norm",     float(grad_norm),   global_step)

        if accelerator.is_main_process:
            avg_train_loss = epoch_loss / len(train_loader)
            epoch_time     = time.time() - epoch_start

            # Validation
            metrics = evaluate(model, val_loader, accelerator, id2label)

            remaining = epoch_time * (args.epochs - epoch - 1)
            print(f"\n{'─'*72}")
            print(
                f"  Epoch {epoch+1}/{args.epochs}"
                f"  |  train loss: {avg_train_loss:.4f}"
                f"  |  val loss: {metrics['loss']:.4f}"
                f"  |  val acc: {metrics['accuracy']*100:.1f}%"
                f"  |  macro F1: {metrics['macro_f1']:.4f}"
                f"  |  weighted F1: {metrics['weighted_f1']:.4f}"
                f"  |  {epoch_time/60:.1f} min"
                f"  |  ETA: {remaining/60:.0f} min"
            )

            # Top-5 and bottom-5 classes by F1
            ranked = sorted(
                [(n, v["f1"]) for n, v in metrics["per_class"].items() if v["support"] > 0],
                key=lambda x: x[1], reverse=True,
            )
            if ranked:
                print(f"  Best  F1: " + "  ".join(f"{n}={f:.2f}" for n, f in ranked[:5]))
                print(f"  Worst F1: " + "  ".join(f"{n}={f:.2f}" for n, f in ranked[-5:]))
            print(f"{'─'*72}\n")

            # TensorBoard — epoch-level
            ep = epoch + 1
            writer.add_scalars("Loss/epoch",    {"train": avg_train_loss, "val": metrics["loss"]},    ep)
            writer.add_scalar("Val/accuracy",   metrics["accuracy"],   ep)
            writer.add_scalar("Val/macro_f1",   metrics["macro_f1"],   ep)
            writer.add_scalar("Val/weighted_f1",metrics["weighted_f1"],ep)

            # Per-class F1, precision, recall
            for name, vals in metrics["per_class"].items():
                safe = name.replace("/", "_")
                writer.add_scalar(f"PerClass_F1/{safe}",        vals["f1"],        ep)
                writer.add_scalar(f"PerClass_Precision/{safe}", vals["precision"], ep)
                writer.add_scalar(f"PerClass_Recall/{safe}",    vals["recall"],    ep)

            # Confusion matrix image every 2 epochs and at the end
            if ep % 2 == 0 or ep == args.epochs:
                cm_img = confusion_matrix_image(
                    metrics["all_labels"], metrics["all_preds"], id2label
                )
                writer.add_image("ConfusionMatrix/val", cm_img, ep)

            writer.flush()

        model.train()

    if accelerator.is_main_process:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(out))
        processor.save_pretrained(str(out))
        writer.close()
        print(f"\nModel saved → {out}")
        print(f"TensorBoard logs → {out}/runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True, help="Path to dataset/train.json")
    parser.add_argument("--output",     default="models/layoutlmv3-finetuned")
    parser.add_argument("--epochs",     default=10,  type=int)
    parser.add_argument("--batch-size", default=2,   type=int)
    parser.add_argument("--lr",         default=5e-5, type=float)
    parser.add_argument("--val-split",  default=0.1, type=float,
                        help="Fraction of data held out for validation (default: 0.1)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
