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

LABEL2ID = {
    "O": 0,
    "B-H0_Part_Page": 1,           "I-H0_Part_Page": 2,
    "B-H1_Heading": 3,             "I-H1_Heading": 4,
    "B-H2_Subheading": 5,          "I-H2_Subheading": 6,
    "B-H3_Stylistic": 7,           "I-H3_Stylistic": 8,
    "B-H4": 9,                     "I-H4": 10,
    "B-H5": 11,                    "I-H5": 12,
    "B-List_Item": 13,             "I-List_Item": 14,
    "B-Paragraph_Text": 15,        "I-Paragraph_Text": 16,
    "B-Cover_Page": 17,            "I-Cover_Page": 18,
    "B-Table_of_Contents": 19,     "I-Table_of_Contents": 20,
    "B-Executive_Summary": 21,     "I-Executive_Summary": 22,
    "B-Specific_Front_Matter": 23, "I-Specific_Front_Matter": 24,
    "B-Footnote_Text": 25,         "I-Footnote_Text": 26,
    "B-Page_Number": 27,           "I-Page_Number": 28,
    "B-Running_Header_Footer": 29, "I-Running_Header_Footer": 30,
    "B-Table": 31,                 "I-Table": 32,
    "B-Figure_Image": 33,          "I-Figure_Image": 34,
    "B-Caption": 35,               "I-Caption": 36,
    "B-Image_with_Embedded_Text": 37, "I-Image_with_Embedded_Text": 38,
    "B-Title_Block": 39,           "I-Title_Block": 40,
    "B-Author": 41,                "I-Author": 42,
    "B-Date": 43,                  "I-Date": 44,
    "B-Jurisdiction": 45,          "I-Jurisdiction": 46,
    "B-Unclear_Needs_Review": 47,  "I-Unclear_Needs_Review": 48,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = LayoutLMv3Processor.from_pretrained(model_path)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()

    def predict_page(self, image: Image.Image) -> list[Block]:
        """Run inference on one page image. Returns block-level predictions."""
        w, h = image.size
        words, norm_bboxes = ocr_page(image)
        if not words:
            return []

        encoding = self.processor(
            image,
            words,
            boxes=norm_bboxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        pred_scores = probs.max(dim=-1).values.cpu().tolist()

        # Map subword tokens back to words (first token per word)
        word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else \
                   encoding.encodings[0].word_ids
        seen, word_labels, word_scores = set(), [], []
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            word_labels.append(ID2LABEL.get(pred_ids[idx], "O"))
            word_scores.append(pred_scores[idx])

        # Convert norm bboxes back to pixel coords
        pixel_bboxes = [
            [
                int(b[0] / 1000 * w), int(b[1] / 1000 * h),
                int(b[2] / 1000 * w), int(b[3] / 1000 * h),
            ]
            for b in norm_bboxes
        ]

        return _merge_tokens_to_blocks(words, pixel_bboxes, word_labels, word_scores)
