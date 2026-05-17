"""
Exports Label Studio annotations → LayoutLMv3 fine-tuning dataset.

Each annotated page becomes one training example with:
  words, bboxes (0-1000), ner_tags (BIO), image_url

Fetches annotations directly via Label Studio API (no manual export step).

Usage:
    python export_to_layoutlmv3.py --api-key TOKEN --project-id 4 --out dataset/
"""

import argparse
import json
import sys
from pathlib import Path

import pytesseract
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from inference.predictor import LABEL2ID

LS_URL  = "http://localhost:8080"
IMG_URL = "http://localhost:9090"


def fetch_annotated_tasks(api_key: str, project_id: int) -> list[dict]:
    headers = {"Authorization": f"Token {api_key}"}
    tasks, page = [], 1
    while True:
        r = requests.get(
            f"{LS_URL}/api/tasks/?project={project_id}&page={page}&page_size=100",
            headers=headers,
        )
        batch = r.json().get("tasks", [])
        if not batch:
            break
        tasks.extend(t for t in batch if t.get("total_annotations", 0) > 0)
        if len(batch) < 100:
            break
        page += 1
    return tasks


def load_image(url: str) -> Image.Image:
    from io import BytesIO
    full = url if url.startswith("http") else IMG_URL + url
    resp = requests.get(full)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def get_label_for_word(norm_bbox: list[int], annotations: list[dict], img_w: int, img_h: int) -> str:
    """Find which annotated box this word's centre falls into."""
    cx = (norm_bbox[0] + norm_bbox[2]) / 2
    cy = (norm_bbox[1] + norm_bbox[3]) / 2
    for ann in annotations:
        v = ann.get("value", {})
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        ax0 = v["x"] / 100 * 1000
        ay0 = v["y"] / 100 * 1000
        ax1 = ax0 + v["width"] / 100 * 1000
        ay1 = ay0 + v["height"] / 100 * 1000
        if ax0 <= cx <= ax1 and ay0 <= cy <= ay1:
            return labels[0]
    return "O"


def task_to_examples(task: dict) -> list[dict]:
    pages = task["data"].get("pages", [])
    annotation = next(
        (a for a in task.get("annotations", []) if not a.get("was_cancelled")), None
    )
    if not annotation:
        return []

    # Group results by page index
    page_results: dict[int, list[dict]] = {}
    for r in annotation.get("result", []):
        page_idx = r.get("value", {}).get("page", 0)
        page_results.setdefault(page_idx, []).append(r)

    examples = []
    for page_idx, page_url in enumerate(pages):
        anns = page_results.get(page_idx, [])
        if not anns:
            continue

        image = load_image(page_url)
        img_w, img_h = image.size
        ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        words, bboxes, ner_tags = [], [], []
        prev_label = None
        for i, word in enumerate(ocr["text"]):
            if not word.strip():
                continue
            x, y, bw, bh = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
            norm_bbox = [
                int(x / img_w * 1000), int(y / img_h * 1000),
                int((x + bw) / img_w * 1000), int((y + bh) / img_h * 1000),
            ]
            label = get_label_for_word(norm_bbox, anns, img_w, img_h)
            if label == "O":
                bio = "O"
                prev_label = None
            elif label == prev_label:
                bio = f"I-{label}"
            else:
                bio = f"B-{label}"
                prev_label = label

            words.append(word)
            bboxes.append(norm_bbox)
            ner_tags.append(LABEL2ID.get(bio, 0))

        if words:
            examples.append({
                "id":        f"{task['id']}_page{page_idx}",
                "image_url": page_url,
                "words":     words,
                "bboxes":    bboxes,
                "ner_tags":  ner_tags,
            })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--out",        default="dataset")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching annotated tasks...")
    tasks = fetch_annotated_tasks(args.api_key, args.project_id)
    print(f"  {len(tasks)} annotated tasks")

    all_examples = []
    for task in tasks:
        pdf_name = task["data"].get("pdf_name", f"task-{task['id']}")
        print(f"  Processing {pdf_name}...", end=" ", flush=True)
        examples = task_to_examples(task)
        all_examples.extend(examples)
        print(f"{len(examples)} page examples")

    with open(out_dir / "train.json", "w") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    with open(out_dir / "label2id.json", "w") as f:
        json.dump(LABEL2ID, f, indent=2)

    print(f"\nSaved {len(all_examples)} training examples → {out_dir}/train.json")


if __name__ == "__main__":
    main()
