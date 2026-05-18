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
import json
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    get_linear_schedule_with_warmup,
)

BASE_MODEL = "microsoft/layoutlmv3-base"


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
# Training
# ---------------------------------------------------------------------------

def train(args):
    accelerator = Accelerator()

    with open(args.data) as f:
        examples = json.load(f)
    with open(Path(args.data).parent / "label2id.json") as f:
        label2id = json.load(f)
    num_labels = len(label2id)

    steps_per_epoch = max(1, len(examples) // args.batch_size)
    total_steps     = steps_per_epoch * args.epochs

    if accelerator.is_main_process:
        print(f"\n{'='*50}")
        print(f"  Training examples : {len(examples)}")
        print(f"  Labels            : {num_labels}")
        print(f"  Device            : {accelerator.device}")
        print(f"  Num processes     : {accelerator.num_processes}")
        print(f"  Epochs            : {args.epochs}")
        print(f"  Batch size        : {args.batch_size}")
        print(f"  Steps/epoch       : {steps_per_epoch}")
        print(f"  Total steps       : {total_steps}")
        print(f"  Learning rate     : {args.lr}")
        print(f"{'='*50}\n")

        writer = SummaryWriter(log_dir=str(Path(args.output) / "runs"))

    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels
    )

    dataset    = DocumentDataset(examples, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            step_start = time.time()

            outputs = model(**batch)
            loss    = outputs.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_loss    = loss.item()
            epoch_loss  += step_loss
            global_step += 1

            if accelerator.is_main_process:
                step_time = time.time() - step_start
                lr_now    = scheduler.get_last_lr()[0]

                # Terminal: print every step
                print(
                    f"Epoch {epoch+1:>2}/{args.epochs} "
                    f"| Step {step+1:>3}/{steps_per_epoch} "
                    f"| Loss {step_loss:.4f} "
                    f"| LR {lr_now:.2e} "
                    f"| {step_time:.1f}s/step",
                    flush=True,
                )

                # TensorBoard: step-level metrics
                writer.add_scalar("Loss/step",    step_loss, global_step)
                writer.add_scalar("LR/step",      lr_now,    global_step)

        if accelerator.is_main_process:
            avg_loss   = epoch_loss / len(dataloader)
            epoch_time = time.time() - epoch_start
            remaining  = epoch_time * (args.epochs - epoch - 1)

            print(f"\n{'─'*60}")
            print(
                f"  Epoch {epoch+1}/{args.epochs} complete"
                f"  |  avg loss: {avg_loss:.4f}"
                f"  |  time: {epoch_time/60:.1f} min"
                f"  |  ETA: {remaining/60:.0f} min remaining"
            )
            print(f"{'─'*60}\n")

            # TensorBoard: epoch-level metrics
            writer.add_scalar("Loss/epoch", avg_loss, epoch + 1)
            writer.flush()

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
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
