"""
Production inference pipeline: PDF → LayoutLMv3 block detection → pdfplumber text → JSON

For each page:
  1. Render page to image
  2. Run LayoutLMv3 to detect layout blocks (bbox + label)
  3. Map image bboxes back to PDF coordinate space
  4. Use pdfplumber to extract clean text from each block region
  5. Output structured JSON

Usage:
    python inference_pipeline.py --model ./models/layoutlmv3-finetuned --pdf path/to/doc.pdf
    python inference_pipeline.py --model ./models/layoutlmv3-finetuned --pdf-dir data/pdfs/ --out-dir outputs/
"""

import argparse
import json
import sys
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from inference.predictor import LayoutLMv3Predictor, Block

DPI = 150


def image_bbox_to_pdf(bbox: list[int], img_w: int, img_h: int, page) -> tuple[float, float, float, float]:
    """
    Convert image pixel bbox to pdfplumber bbox (PDF points, origin bottom-left).
    pdfplumber uses (x0, top, x1, bottom) with origin at top-left.
    """
    pdf_w = float(page.width)
    pdf_h = float(page.height)
    x0 = bbox[0] / img_w * pdf_w
    y0 = bbox[1] / img_h * pdf_h
    x1 = bbox[2] / img_w * pdf_w
    y1 = bbox[3] / img_h * pdf_h
    return (x0, y0, x1, y1)


def extract_text_in_bbox(pdf_page, bbox_pdf: tuple) -> str:
    """Extract text from a region of a pdfplumber page."""
    x0, y0, x1, y1 = bbox_pdf
    cropped = pdf_page.within_bbox((x0, y0, x1, y1), relative=False)
    return cropped.extract_text(x_tolerance=3, y_tolerance=3) or ""


def process_pdf(pdf_path: Path, predictor: LayoutLMv3Predictor) -> dict:
    images = convert_from_path(str(pdf_path), dpi=DPI)
    output = {"pdf": pdf_path.name, "pages": []}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, (image, pdf_page) in enumerate(zip(images, pdf.pages)):
            img_w, img_h = image.size
            blocks = predictor.predict_page(image)

            page_blocks = []
            for block in blocks:
                bbox_pdf = image_bbox_to_pdf(block.bbox, img_w, img_h, pdf_page)
                text = extract_text_in_bbox(pdf_page, bbox_pdf)

                page_blocks.append({
                    "label":       block.label,
                    "score":       round(block.score, 4),
                    "bbox_image":  block.bbox,       # [x0,y0,x1,y1] pixels
                    "bbox_pdf":    [round(v, 2) for v in bbox_pdf],   # PDF points
                    "text":        text.strip(),
                })

            output["pages"].append({
                "page_number": page_idx + 1,
                "blocks": page_blocks,
            })

            print(f"  Page {page_idx + 1}/{len(images)}: {len(blocks)} blocks detected")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Path to fine-tuned LayoutLMv3")
    parser.add_argument("--pdf",     help="Single PDF to process")
    parser.add_argument("--pdf-dir", help="Directory of PDFs to process")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for JSON files")
    parser.add_argument("--dpi",     default=DPI, type=int)
    args = parser.parse_args()

    if not args.pdf and not args.pdf_dir:
        parser.error("Provide --pdf or --pdf-dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}...")
    predictor = LayoutLMv3Predictor(args.model)

    pdfs = [Path(args.pdf)] if args.pdf else sorted(Path(args.pdf_dir).glob("*.pdf"))
    print(f"Processing {len(pdfs)} PDF(s)...\n")

    for pdf_path in pdfs:
        print(f"--- {pdf_path.name} ---")
        result = process_pdf(pdf_path, predictor)

        out_path = out_dir / f"{pdf_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
