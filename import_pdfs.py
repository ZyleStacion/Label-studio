"""
Converts PDFs in data/pdfs/ to images and imports them into Label Studio.
Each PDF becomes one task with all its pages listed (for multi-page annotation).
Images are served from the image server on port 9090.

Usage:
    python import_pdfs.py --api-key TOKEN --project-id 4
"""

import argparse
import requests
from pathlib import Path
from pdf2image import convert_from_path

BASE_DIR   = Path(__file__).parent
PDF_DIR    = BASE_DIR / "data" / "pdfs"
IMAGES_DIR = BASE_DIR / "data" / "images"
LS_URL     = "http://localhost:8080"
IMG_URL    = "http://localhost:9090"


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[Path]:
    out_dir = IMAGES_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    saved = []
    for i, page in enumerate(pages):
        img_path = out_dir / f"page_{i+1:04d}.png"
        if not img_path.exists():
            page.save(str(img_path), "PNG")
        saved.append(img_path)
    return saved


def import_pdf_task(api_key: str, project_id: int, pdf_path: Path, image_paths: list[Path]):
    page_urls = [f"{IMG_URL}/{pdf_path.stem}/{p.name}" for p in image_paths]
    task = {"data": {"pages": page_urls, "pdf_name": pdf_path.stem}}
    resp = requests.post(
        f"{LS_URL}/api/projects/{project_id}/import",
        headers={"Authorization": f"Token {api_key}"},
        json=[task],
    )
    resp.raise_for_status()
    return resp.json()


def get_existing_pdf_names(api_key: str, project_id: int) -> set[str]:
    names, page = set(), 1
    while True:
        r = requests.get(
            f"{LS_URL}/api/tasks/?project={project_id}&page={page}&page_size=100",
            headers={"Authorization": f"Token {api_key}"},
        )
        batch = r.json().get("tasks", [])
        if not batch:
            break
        for t in batch:
            names.add(t["data"].get("pdf_name", ""))
        if len(batch) < 100:
            break
        page += 1
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--dpi",        default=150,   type=int)
    args = parser.parse_args()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        return

    existing = get_existing_pdf_names(args.api_key, args.project_id)
    print(f"Already imported: {existing or 'none'}\n")

    imported = 0
    for pdf in pdfs:
        if pdf.stem in existing:
            print(f"Skipping {pdf.name} (already in project)")
            continue
        print(f"Converting {pdf.name}...")
        images = pdf_to_images(pdf, dpi=args.dpi)
        print(f"  {len(images)} pages → creating task...")
        import_pdf_task(args.api_key, args.project_id, pdf, images)
        print(f"  Done")
        imported += 1

    print(f"\nImported {imported} new tasks. Make sure image_server.py is running on port 9090.")


if __name__ == "__main__":
    main()
