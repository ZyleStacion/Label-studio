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

# Map old model label names → new Label Studio label names.
# Any label not in this dict (or mapped to None) is dropped.
LABEL_MAP: dict[str, str | None] = {
    "H0_Part_Page":             "Chapter_title",
    "H1_Heading":               "H1",
    "H2_Subheading":            "H2",
    "H3_Stylistic":             "H3",
    "H4":                       "H4",
    "H5":                       "H5",
    "List_Item":                "Numbered_list",
    "Paragraph_Text":           "Paragraph_text",
    "Cover_Page":               "Title_block",
    "Table_of_Contents":        "Table_of_content",
    "Executive_Summary":        "Box_section",
    "Specific_Front_Matter":    "Title_block",
    "Footnote_Text":            "Footnotes",
    "Page_Number":              "Page_number",
    "Running_Header_Footer":    "Running_header_footer",
    "Table":                    "Table",
    "Figure_Image":             "Image",
    "Caption":                  "Caption",
    "Image_with_Embedded_Text": "Image_embedded_text",
    "Title_Block":              "Title_block",
    "Author":                   "Title_block",
    "Date":                     "Title_block",
    "Jurisdiction":             "Paragraph_text",
    "Unclear_Needs_Review":     None,   # drop — reviewer decides
    # New labels pass through unchanged
    "H1":                       "H1",
    "H2":                       "H2",
    "H3":                       "H3",
    "Chapter_title":            "Chapter_title",
    "Chapter_number":           "Chapter_number",
    "Chapter_TOC":              "Chapter_TOC",
    "Paragraph_text":           "Paragraph_text",
    "Numbered_list":            "Numbered_list",
    "Unordered_list":           "Unordered_list",
    "Footnotes":                "Footnotes",
    "Borderless_table":         "Borderless_table",
    "Table_of_content":         "Table_of_content",
    "Table_caption":            "Table_caption",
    "Image_embedded_text":      "Image_embedded_text",
    "Figure_caption":           "Figure_caption",
    "Running_header_footer":    "Running_header_footer",
    "Title_block":              "Title_block",
    "Box_section":              "Box_section",
    "Form":                     "Form",
}


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
            mapped = LABEL_MAP.get(block.label, block.label)
            if mapped is None:
                continue
            ls_bbox = bbox_to_ls(block.bbox, img_w, img_h)
            results.append({
                "from_name":  "layout_label",
                "to_name":    "pdf",
                "type":       "rectanglelabels",
                "score":      round(block.score, 4),
                "item_index": page_idx,
                "value": {
                    **ls_bbox,
                    "rectanglelabels": [mapped],
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
    parser.add_argument("--task-id", type=int, default=None,
                        help="Only run on this specific task ID (ignores --skip-annotated)")
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
        if args.task_id and task["id"] != args.task_id:
            continue
        if not args.task_id and args.skip_annotated and task.get("total_annotations", 0) > 0:
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
