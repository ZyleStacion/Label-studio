"""
Human-in-the-Loop pre-labeling: runs LayoutLMv3 on unannotated Label Studio tasks
and pushes predictions as pre-annotations so reviewers fix instead of draw from scratch.

Usage:
    python prelabel.py --model ./models/layoutlmv3-finetuned --api-key TOKEN --project-id 4

The script only processes tasks with zero annotations (skips already-reviewed ones).
After running, open Label Studio → each task shows model predictions as a draft to accept/edit.
"""

import argparse
import sys
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from inference.predictor import LayoutLMv3Predictor

LS_URL    = "http://localhost:8080"
IMG_URL   = "http://localhost:9090"


def bbox_to_ls(bbox: list[int], img_w: int, img_h: int) -> dict:
    """Convert pixel bbox to Label Studio % format."""
    return {
        "x":      bbox[0] / img_w * 100,
        "y":      bbox[1] / img_h * 100,
        "width":  (bbox[2] - bbox[0]) / img_w * 100,
        "height": (bbox[3] - bbox[1]) / img_h * 100,
        "rotation": 0,
    }


def predict_task(predictor: LayoutLMv3Predictor, task: dict) -> list[dict]:
    """Run model on every page of a task, return Label Studio result list."""
    results = []
    pages = task["data"].get("pages", [])

    for page_idx, page_url in enumerate(pages):
        # Fetch the image from the image server
        resp = requests.get(page_url if page_url.startswith("http") else IMG_URL + page_url)
        resp.raise_for_status()
        from io import BytesIO
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        img_w, img_h = image.size

        blocks = predictor.predict_page(image)
        for block in blocks:
            ls_bbox = bbox_to_ls(block.bbox, img_w, img_h)
            results.append({
                "from_name": "layout_label",
                "to_name":   "pdf",
                "type":      "rectanglelabels",
                "score":     round(block.score, 4),
                "value": {
                    **ls_bbox,
                    "rectanglelabels": [block.label],
                    "page": page_idx,          # tells LS which page this bbox belongs to
                },
            })

    return results


def push_prediction(api_key: str, task_id: int, results: list[dict], model_version: str):
    resp = requests.post(
        f"{LS_URL}/api/predictions/",
        headers={"Authorization": f"Token {api_key}"},
        json={
            "task":          task_id,
            "model_version": model_version,
            "score":         sum(r["score"] for r in results) / max(len(results), 1),
            "result":        results,
        },
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, help="Path to fine-tuned LayoutLMv3")
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--skip-annotated", action="store_true", default=True,
                        help="Skip tasks that already have human annotations")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    predictor = LayoutLMv3Predictor(args.model)

    headers = {"Authorization": f"Token {args.api_key}"}
    tasks, page = [], 1
    while True:
        r = requests.get(
            f"{LS_URL}/api/tasks/?project={args.project_id}&page={page}&page_size=100",
            headers=headers,
        )
        batch = r.json().get("tasks", [])
        if not batch:
            break
        tasks.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    print(f"Found {len(tasks)} tasks")

    for task in tasks:
        if args.skip_annotated and task.get("total_annotations", 0) > 0:
            print(f"  Task {task['id']}: skipping (already annotated)")
            continue

        pdf_name = task["data"].get("pdf_name", f"task-{task['id']}")
        pages = task["data"].get("pages", [])
        print(f"  Task {task['id']} ({pdf_name}): {len(pages)} pages...", end=" ", flush=True)

        results = predict_task(predictor, task)
        push_prediction(args.api_key, task["id"], results, args.model_version)
        print(f"{len(results)} predictions pushed")

    print("\nDone. Open Label Studio to review predictions.")


if __name__ == "__main__":
    main()
