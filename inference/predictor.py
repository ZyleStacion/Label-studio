"""
Shared LayoutLMv3 inference logic used by both prelabel.py and inference_pipeline.py.

Given a page image, runs OCR then LayoutLMv3 to produce block-level predictions:
    [{"label": "H1_Heading", "bbox": [x0, y0, x1, y1], "score": 0.95, "words": [...]}]
Bboxes are in pixel coordinates relative to the image.
"""

from __future__ import annotations
import dataclasses
from pathlib import Path

import pytesseract
import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor



@dataclasses.dataclass
class Block:
    label: str
    bbox: list[int]   # [x0, y0, x1, y1] pixel coords
    score: float
    words: list[str]


def ocr_page(image: Image.Image) -> tuple[list[str], list[list[int]]]:
    """Return (words, bboxes) from Tesseract. Bboxes are pixel coords [x0,y0,x1,y1]."""
    w, h = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, bboxes = [], []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        x, y = data["left"][i], data["top"][i]
        bw, bh = data["width"][i], data["height"][i]
        # LayoutLMv3 expects 0-1000 normalized bboxes
        bbox = [
            int(x / w * 1000),
            int(y / h * 1000),
            int((x + bw) / w * 1000),
            int((y + bh) / h * 1000),
        ]
        words.append(word)
        bboxes.append(bbox)
    return words, bboxes


def _merge_tokens_to_blocks(
    words: list[str],
    pixel_bboxes: list[list[int]],
    labels: list[str],
    scores: list[float],
) -> list[Block]:
    """Merge consecutive word tokens with the same base label into blocks."""
    blocks: list[Block] = []
    current: Block | None = None

    for word, bbox, label, score in zip(words, pixel_bboxes, labels, scores):
        base = label[2:] if label.startswith(("B-", "I-")) else label
        is_begin = label.startswith("B-") or label == "O"

        if base == "O":
            current = None
            continue

        if is_begin or current is None or current.label != base:
            current = Block(label=base, bbox=list(bbox), score=score, words=[word])
            blocks.append(current)
        else:
            # Extend current block's bbox to include this token
            current.bbox[0] = min(current.bbox[0], bbox[0])
            current.bbox[1] = min(current.bbox[1], bbox[1])
            current.bbox[2] = max(current.bbox[2], bbox[2])
            current.bbox[3] = max(current.bbox[3], bbox[3])
            current.score = (current.score + score) / 2
            current.words.append(word)

    return blocks


class LayoutLMv3Predictor:
    def __init__(self, model_path: str):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.processor = LayoutLMv3Processor.from_pretrained(model_path)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    def predict_page(self, image: Image.Image) -> list[Block]:
        """Run inference on one page image. Returns block-level predictions."""
        w, h = image.size
        words, norm_bboxes = ocr_page(image)
        if not words:
            return []

        # Sliding window: process page in overlapping chunks of CHUNK words.
        # LayoutLMv3 is limited to 512 tokens; a dense legal page can have
        # 500-1000+ words, so a single pass silently drops most of the page.
        CHUNK  = 200   # words per chunk (conservative — each word can be 2-3 sub-tokens)
        OVERLAP = 30   # overlap so boundary words get clean context

        word_labels = ["O"] * len(words)
        word_scores = [0.0]  * len(words)

        start = 0
        while start < len(words):
            end          = min(start + CHUNK, len(words))
            chunk_words  = words[start:end]
            chunk_bboxes = norm_bboxes[start:end]

            encoding = self.processor(
                image,
                chunk_words,
                boxes=chunk_bboxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            word_ids    = encoding.encodings[0].word_ids
            model_inputs = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            logits      = outputs.logits[0]
            probs       = torch.softmax(logits, dim=-1)
            pred_ids    = logits.argmax(dim=-1).cpu().tolist()
            pred_scores = probs.max(dim=-1).values.cpu().tolist()

            seen = set()
            for idx, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                global_wid = start + wid
                if global_wid < len(words):
                    word_labels[global_wid] = self.id2label.get(pred_ids[idx], "O")
                    word_scores[global_wid] = pred_scores[idx]

            if end >= len(words):
                break
            start = end - OVERLAP   # step back by overlap for context continuity

        # Convert norm bboxes back to pixel coords
        pixel_bboxes = [
            [
                int(b[0] / 1000 * w), int(b[1] / 1000 * h),
                int(b[2] / 1000 * w), int(b[3] / 1000 * h),
            ]
            for b in norm_bboxes
        ]

        return _merge_tokens_to_blocks(words, pixel_bboxes, word_labels, word_scores)
