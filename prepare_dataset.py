"""
Converts Label Studio export JSON files → LayoutLMv3 training dataset.

Handles the multi-page export format (item_index for page tracking).
Renders missing page images from PDFs automatically.

Usage:
    python prepare_dataset.py --export-dir data/export/ --out dataset/
"""

import argparse
import json
import re
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

BASE_DIR   = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "data" / "images"
PDF_DIR    = BASE_DIR / "data" / "pdfs"
DPI        = 150

_LABELS = [
    "H1", "H2", "H3", "H4", "H5",
    "Chapter_title", "Chapter_number", "Chapter_TOC",
    "Paragraph_text", "Numbered_list", "Unordered_list", "Footnotes",
    "Table", "Borderless_table", "Table_of_content", "Table_caption",
    "Image", "Image_embedded_text", "Figure_caption", "Caption",
    "Page_number", "Running_header_footer",
    "Title_block", "Box_section", "Form",
]
LABEL2ID = {"O": 0}
_idx = 1
for _lbl in _LABELS:
    LABEL2ID[f"B-{_lbl}"] = _idx;  _idx += 1
    LABEL2ID[f"I-{_lbl}"] = _idx;  _idx += 1


def load_export(path: Path) -> dict:
    """Load a single-task export dict, handling UTF-16 and UTF-8."""
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            data = json.loads(path.read_text(encoding=enc))
            # Normalise: single task dict or list → always a single dict
            if isinstance(data, list):
                # Pick the task with the most annotations
                data = max(data, key=lambda t: sum(
                    len(a.get("result", [])) for a in t.get("annotations", [])
                ))
            return data
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode {path}")


def ensure_images(pdf_stem: str, num_pages: int) -> Path:
    """Render PDF to images if not already done. Returns image directory."""
    out_dir = IMAGES_DIR / pdf_stem
    existing = list(out_dir.glob("page_*.png")) if out_dir.exists() else []
    if len(existing) >= num_pages:
        return out_dir

    pdf_path = PDF_DIR / f"{pdf_stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Missing PDF: {pdf_path}\n"
            f"Copy it to data/pdfs/ and re-run."
        )

    print(f"  Rendering {pdf_stem} ({num_pages} pages) at {DPI} DPI...")
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=DPI)
    for i, page in enumerate(pages):
        img_path = out_dir / f"page_{i+1:04d}.png"
        if not img_path.exists():
            page.save(str(img_path), "PNG")
    return out_dir


def word_in_box(norm_cx: float, norm_cy: float, ann: dict, img_w: int, img_h: int) -> str:
    """Return label if word centre falls inside annotation bbox, else empty string."""
    v = ann["value"]
    labels = v.get("rectanglelabels", [])
    if not labels:
        return ""
    # annotation coords are % of original_width/original_height from the result entry
    ow = ann.get("original_width",  img_w)
    oh = ann.get("original_height", img_h)
    ax0 = v["x"] / 100 * ow / img_w * 1000
    ay0 = v["y"] / 100 * oh / img_h * 1000
    ax1 = ax0 + v["width"]  / 100 * ow / img_w * 1000
    ay1 = ay0 + v["height"] / 100 * oh / img_h * 1000
    if ax0 <= norm_cx <= ax1 and ay0 <= norm_cy <= ay1:
        return labels[0]
    return ""


def page_to_example(image: Image.Image, page_anns: list[dict], page_id: str) -> dict | None:
    img_w, img_h = image.size
    ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words, bboxes, ner_tags = [], [], []
    prev_label = None

    for i, word in enumerate(ocr["text"]):
        if not word.strip():
            continue
        x, y   = ocr["left"][i], ocr["top"][i]
        bw, bh = ocr["width"][i], ocr["height"][i]
        norm_bbox = [
            int(x / img_w * 1000), int(y / img_h * 1000),
            int((x + bw) / img_w * 1000), int((y + bh) / img_h * 1000),
        ]
        norm_cx = (norm_bbox[0] + norm_bbox[2]) / 2
        norm_cy = (norm_bbox[1] + norm_bbox[3]) / 2

        label = ""
        for ann in page_anns:
            label = word_in_box(norm_cx, norm_cy, ann, img_w, img_h)
            if label:
                break

        if not label:
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

    if not words:
        return None
    return {"id": page_id, "words": words, "bboxes": bboxes, "ner_tags": ner_tags}


def resolve_pdf_stem(task: dict, export_path: Path) -> str:
    """Return the pdf_stem to use, falling back gracefully when URLs point to a wrong PDF."""
    pages = task["data"].get("pages", [])
    candidates = []
    if task["data"].get("pdf_name"):
        candidates.append(task["data"]["pdf_name"])
    if pages:
        candidates.append(Path(pages[0]).parent.name)
    candidates.append(export_path.stem)

    for stem in candidates:
        if (IMAGES_DIR / stem).exists() or (PDF_DIR / f"{stem}.pdf").exists():
            return stem
    return candidates[0] if candidates else export_path.stem


def process_export(path: Path) -> list[dict]:
    print(f"\nProcessing {path.name}...")
    task = load_export(path)

    pages     = task["data"].get("pages", [])
    pdf_stem  = resolve_pdf_stem(task, path)
    anns_raw  = task.get("annotations", [])
    valid     = [a for a in anns_raw if not a.get("was_cancelled")]
    ann       = max(valid, key=lambda a: len(a.get("result", [])), default=None)
    if not ann:
        print("  No annotations — skipping")
        return []

    results = ann.get("result", [])
    # Group results by item_index (page)
    by_page: dict[int, list[dict]] = {}
    for r in results:
        if r.get("type") == "rectanglelabels":
            by_page.setdefault(r["item_index"], []).append(r)

    annotated_pages = sorted(by_page.keys())
    print(f"  PDF: {pdf_stem}  |  {len(pages)} pages  |  {len(annotated_pages)} pages annotated  |  {len(results)} bboxes")

    img_dir = ensure_images(pdf_stem, len(pages))
    examples = []

    for page_idx in annotated_pages:
        img_path = img_dir / f"page_{page_idx+1:04d}.png"
        if not img_path.exists():
            print(f"  Warning: missing image {img_path.name} — skipping")
            continue
        image = Image.open(img_path).convert("RGB")
        ex = page_to_example(image, by_page[page_idx], f"{pdf_stem}_p{page_idx}")
        if ex:
            ex["image_path"] = str(img_path)
            examples.append(ex)

    print(f"  → {len(examples)} training examples")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="data/export")
    parser.add_argument("--out",        default="dataset")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []
    missing_pdfs = []

    for export_file in sorted(export_dir.glob("*.json")):
        try:
            examples = process_export(export_file)
            all_examples.extend(examples)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            missing_pdfs.append(str(e).split("\n")[0])

    if missing_pdfs:
        print("\nMissing PDFs (add to data/pdfs/ and re-run):")
        for m in missing_pdfs:
            print(" ", m)

    if not all_examples:
        print("\nNo examples generated. Add the missing PDFs and re-run.")
        return

    with open(out_dir / "train.json", "w") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    with open(out_dir / "label2id.json", "w") as f:
        json.dump(LABEL2ID, f, indent=2)

    print(f"\nTotal: {len(all_examples)} training examples → {out_dir}/train.json")


if __name__ == "__main__":
    main()
