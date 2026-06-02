# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Label Studio annotation pipeline for PDF documents, producing training data for LayoutLMv3 (document layout analysis). The pipeline is: PDFs → page images → Label Studio annotation → LayoutLMv3 JSON dataset.

## Environment

All scripts must run inside the `labelstudio` conda environment (Python 3.11):

```bash
conda activate labelstudio
```

Label Studio is installed at `/opt/anaconda3/envs/labelstudio/`.

## Starting Label Studio

```bash
conda activate labelstudio
label-studio start --data-dir /Users/yethuaung/Label-studio/data
```

The `data/.env` file sets `SENTRY_DSN=` (empty) — this is required. Without it, Label Studio crashes on startup due to a sentry-sdk 2.x API incompatibility.

Label Studio runs at `http://localhost:8080`. The active project is **project ID 4**.

## API token

The real API token (not the JWT shown in the UI) is stored in the SQLite database:

```bash
sqlite3 data/label_studio.sqlite3 "SELECT key FROM authtoken_token LIMIT 1;"
```

## Importing PDFs

1. Place PDFs in `data/pdfs/`
2. Run:

```bash
python import_pdfs.py --api-key TOKEN --project-id 4
```

This converts each PDF page to a PNG at 150 DPI (saved to `data/images/<pdf-stem>/`) and uploads in batches of 50. Already-converted pages are skipped.

## Exporting for LayoutLMv3 training

1. In Label Studio UI: Export → JSON → save to `exports/`
2. Run:

```bash
python export_to_layoutlmv3.py --export exports/annotations.json --out dataset/
```

Output: `dataset/layoutlmv3_data.json` (words, bboxes in 0–1000 scale, ner_tags in BIO format) and `dataset/label2id.json`.

## Labeling config

`configs/layoutlmv3_labeling_config.xml` (mirrored from `label.txt`) defines the annotation interface. **`label.txt` is the authoritative source** — always sync the XML from it.

The config uses multi-page image display (`valueList="$pages"`), bounding boxes (`RectangleLabels`), per-region text transcription, confidence rating, and review flags.

Label set (27 labels):

| Group | Labels |
|---|---|
| Headings | H1, H2, H3, H4, H5, Chapter_title, Chapter_number, Chapter_TOC |
| Body | Paragraph_text, Numbered_list, Unordered_list, Footnotes |
| Tables | Table, Borderless_table, Table_of_content, Table_caption |
| Visual elements | Image, Image_embedded_text, Figure_caption, Caption |
| Page furniture | Page_number, Running_header_footer |
| Structure | Title_block, Box_section, Form |

## Directory layout

```
data/pdfs/       ← drop source PDFs here
data/images/     ← auto-generated page PNGs (one subdir per PDF)
data/exports/    ← Label Studio JSON exports
exports/         ← alternative export location
configs/         ← Label Studio labeling XML config
dataset/         ← final LayoutLMv3 training data (generated)
```
