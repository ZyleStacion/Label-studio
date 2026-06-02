"""
Hybrid pre-labeling: LayoutLMv3 predictions with DeepSeek fallback for low-confidence regions.

For each unannotated task:
  1. Run LayoutLMv3 on every page image
  2. Extract text for each block via pdfplumber (same approach as inference_pipeline.py)
  3. For blocks below --threshold whose label is not visual-only,
     call DeepSeek with the block text + surrounding context
  4. Push the merged predictions back to Label Studio as pre-annotations

Usage:
    python hybrid_prelabel.py \
        --model ./models/layoutlmv3-finetuned \
        --api-key TOKEN \
        --project-id 4 \
        --deepseek-key YOUR_DEEPSEEK_KEY \
        --threshold 0.70
"""

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import pdfplumber
import requests
from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from inference.predictor import LayoutLMv3Predictor
from prelabel import LABEL_MAP

LS_URL   = "http://localhost:8080"
IMG_URL  = "http://localhost:9090"
PDF_DIR  = Path("data/pdfs")

ALL_LABELS = [
    "H1", "H2", "H3", "H4", "H5",
    "Chapter_title", "Chapter_number", "Chapter_TOC",
    "Paragraph_text", "Numbered_list", "Unordered_list", "Footnotes",
    "Table", "Borderless_table", "Table_of_content", "Table_caption",
    "Image", "Image_embedded_text", "Figure_caption", "Caption",
    "Page_number", "Running_header_footer",
    "Title_block", "Box_section", "Form",
]

# These depend on visual/positional cues — LLM can't improve them
VISUAL_ONLY = {"Table", "Image", "Image_embedded_text", "Borderless_table"}


def bbox_to_ls(bbox: list[int], img_w: int, img_h: int) -> dict:
    return {
        "x":        bbox[0] / img_w * 100,
        "y":        bbox[1] / img_h * 100,
        "width":    (bbox[2] - bbox[0]) / img_w * 100,
        "height":   (bbox[3] - bbox[1]) / img_h * 100,
        "rotation": 0,
    }


def image_bbox_to_pdf(bbox: list[int], img_w: int, img_h: int, pdf_page) -> tuple:
    """Convert image pixel bbox to pdfplumber coordinate space."""
    pw, ph = float(pdf_page.width), float(pdf_page.height)
    return (
        bbox[0] / img_w * pw,
        bbox[1] / img_h * ph,
        bbox[2] / img_w * pw,
        bbox[3] / img_h * ph,
    )


def extract_text(pdf_page, bbox_pdf: tuple) -> str:
    x0, y0, x1, y1 = bbox_pdf
    try:
        return pdf_page.within_bbox((x0, y0, x1, y1)).extract_text(
            x_tolerance=3, y_tolerance=3
        ) or ""
    except Exception:
        return ""


def build_context(blocks: list[dict], target_idx: int, window: int = 3) -> str:
    lines = []
    start = max(0, target_idx - window)
    end   = min(len(blocks), target_idx + window + 1)
    for i in range(start, end):
        b = blocks[i]
        if i == target_idx:
            lines.append(f">>> TARGET: \"{b['text']}\"")
        elif b["text"].strip():
            lines.append(f"[{b['label']} | conf={b['score']:.2f}] \"{b['text'][:120]}\"")
    return "\n".join(lines)


def deepseek_label(
    client: OpenAI, text: str, context: str, model: str
) -> tuple[str | None, int, int]:
    """Returns (label_or_None, prompt_tokens, completion_tokens)."""
    if not text.strip():
        return None, 0, 0

    prompt = f"""You are classifying regions of a legal/government PDF document for layout analysis.

Available labels (return EXACTLY one):
{", ".join(ALL_LABELS)}

Surrounding context (neighbouring blocks already classified):
{context}

Classify the TARGET block above. Reply with ONLY the label name — no explanation."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        raw            = resp.choices[0].message.content.strip().strip('"').strip("'")
        prompt_tok     = resp.usage.prompt_tokens
        completion_tok = resp.usage.completion_tokens

        # exact match first
        if raw in ALL_LABELS:
            return raw, prompt_tok, completion_tok

        # case-insensitive fallback
        raw_lower      = raw.lower()
        label_map      = {l.lower(): l for l in ALL_LABELS}
        matched        = label_map.get(raw_lower)
        if matched:
            return matched, prompt_tok, completion_tok

        # partial match: model may return e.g. "H1_Heading" for "H1"
        for canonical in ALL_LABELS:
            if canonical.lower() in raw_lower or raw_lower in canonical.lower():
                return canonical, prompt_tok, completion_tok

        print(f"    [warn] DeepSeek returned unrecognised label: {raw!r}")
        return None, prompt_tok, completion_tok
    except Exception as e:
        print(f"    DeepSeek error: {e}")
        return None, 0, 0


def find_pdf(pdf_name: str) -> Path | None:
    """Locate the source PDF by name (with or without extension)."""
    stem = Path(pdf_name).stem
    for suffix in (".pdf", ".PDF"):
        candidate = PDF_DIR / (stem + suffix)
        if candidate.exists():
            return candidate
    # fuzzy: find any PDF whose stem matches
    for p in PDF_DIR.glob("*.pdf"):
        if p.stem == stem:
            return p
    return None


def predict_task(
    predictor: LayoutLMv3Predictor,
    task: dict,
    deepseek_client: OpenAI,
    threshold: float,
    deepseek_delay: float,
    deepseek_model: str,
) -> tuple[list[dict], int, int, int]:
    results        = []
    deepseek_hits  = 0
    prompt_tokens  = 0
    compl_tokens   = 0
    pages          = task["data"].get("pages", [])
    pdf_name      = task["data"].get("pdf_name", "")

    pdf_path = find_pdf(pdf_name)
    if pdf_path:
        pdf_handle = pdfplumber.open(str(pdf_path))
        pdf_pages  = pdf_handle.pages
    else:
        print(f"\n    [warn] PDF not found for '{pdf_name}' — text extraction disabled", end="")
        pdf_handle = None
        pdf_pages  = []

    try:
        for page_idx, page_url in enumerate(pages):
            url  = page_url if page_url.startswith("http") else IMG_URL + page_url
            resp = requests.get(url)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            img_w, img_h = image.size

            raw_blocks = predictor.predict_page(image)

            # Extract text for every block upfront so context is populated
            pdf_page = pdf_pages[page_idx] if pdf_pages and page_idx < len(pdf_pages) else None
            block_dicts = []
            for b in raw_blocks:
                text = ""
                if pdf_page is not None:
                    bbox_pdf = image_bbox_to_pdf(b.bbox, img_w, img_h, pdf_page)
                    text = extract_text(pdf_page, bbox_pdf).strip()
                block_dicts.append({"label": b.label, "score": b.score, "text": text, "bbox": b.bbox})

            for i, block in enumerate(raw_blocks):
                label = LABEL_MAP.get(block.label, block.label)
                if label is None:
                    continue
                score = block.score

                if score < threshold and label not in VISUAL_ONLY:
                    context                  = build_context(block_dicts, i)
                    llm_label, ptok, ctok    = deepseek_label(
                        deepseek_client, block_dicts[i]["text"], context, deepseek_model
                    )
                    prompt_tokens += ptok
                    compl_tokens  += ctok
                    if llm_label:
                        label = llm_label
                        deepseek_hits += 1
                        time.sleep(deepseek_delay)

                ls_bbox = bbox_to_ls(block.bbox, img_w, img_h)
                results.append({
                    "from_name":  "layout_label",
                    "to_name":    "pdf",
                    "type":       "rectanglelabels",
                    "score":      round(score, 4),
                    "item_index": page_idx,
                    "value": {
                        **ls_bbox,
                        "rectanglelabels": [label],
                    },
                })
    finally:
        if pdf_handle:
            pdf_handle.close()

    return results, deepseek_hits, prompt_tokens, compl_tokens


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
    parser.add_argument("--model",          required=True, help="Path to fine-tuned LayoutLMv3")
    parser.add_argument("--api-key",        required=True, help="Label Studio API token")
    parser.add_argument("--project-id",     required=True, type=int)
    parser.add_argument("--deepseek-key",   required=True, help="DeepSeek API key")
    parser.add_argument("--threshold",      type=float, default=0.70,
                        help="Confidence below this triggers DeepSeek fallback (default: 0.70)")
    parser.add_argument("--model-version",  default="hybrid-v1")
    parser.add_argument("--deepseek-model",  default="deepseek-chat",
                        help="DeepSeek model to use (default: deepseek-chat = DeepSeek-V3)")
    parser.add_argument("--deepseek-delay", type=float, default=0.3,
                        help="Seconds between DeepSeek calls (default: 0.3)")
    parser.add_argument("--task-id",        type=int, default=None,
                        help="Process a single task ID only")
    args = parser.parse_args()

    print(f"Loading LayoutLMv3 from {args.model}...")
    predictor = LayoutLMv3Predictor(args.model)

    deepseek = OpenAI(
        api_key=args.deepseek_key,
        base_url="https://api.deepseek.com",
    )

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

    print(f"Found {len(tasks)} tasks  |  threshold: {args.threshold}  |  model: {args.deepseek_model}\n")

    total_preds    = 0
    total_deepseek = 0
    total_prompt   = 0
    total_compl    = 0

    for task in tasks:
        if args.task_id and task["id"] != args.task_id:
            continue
        if not args.task_id and task.get("total_annotations", 0) > 0:
            print(f"  Task {task['id']}: skipping (already annotated)")
            continue

        pdf_name = task["data"].get("pdf_name", f"task-{task['id']}")
        n_pages  = len(task["data"].get("pages", []))
        print(f"  Task {task['id']} ({pdf_name}): {n_pages} pages...", end=" ", flush=True)

        results, ds_hits, ptok, ctok = predict_task(
            predictor, task, deepseek, args.threshold, args.deepseek_delay, args.deepseek_model
        )
        push_prediction(args.api_key, task["id"], results, args.model_version)

        total_preds    += len(results)
        total_deepseek += ds_hits
        total_prompt   += ptok
        total_compl    += ctok
        print(f"{len(results)} predictions  ({ds_hits} via DeepSeek  |  +{ptok}p/{ctok}c tokens)")

    total_tok = total_prompt + total_compl
    # DeepSeek-V3 pricing: $0.27/1M input, $1.10/1M output (cache-miss rates)
    est_cost  = (total_prompt / 1_000_000 * 0.27) + (total_compl / 1_000_000 * 1.10)

    print(f"""
Done.
  Predictions   : {total_preds}
  DeepSeek fixes: {total_deepseek}
  Model used    : {args.deepseek_model}
  Tokens used   : {total_prompt:,} prompt + {total_compl:,} completion = {total_tok:,} total
  Est. cost     : ${est_cost:.4f} USD
Open Label Studio to review.""")


if __name__ == "__main__":
    main()
