---
tags: [scripts, reference, commands]
created: 2026-05-20
---

# Scripts Reference

> [!important] All scripts must run inside the `labelstudio` conda environment
> ```bash
> conda activate labelstudio
> ```

## `import_pdfs.py` — Import PDFs into Label Studio

Converts PDFs to page images and creates one task per PDF in Label Studio. Skips PDFs already imported.

```bash
python import_pdfs.py \
  --api-key TOKEN \
  --project-id 4
```

- Renders pages at 150 DPI → `data/images/<pdf-stem>/page_XXXX.png`
- Skips already-converted pages
- Skips already-imported PDFs (deduplication added 2026-05-20)

---

## `prelabel.py` — Run Model Predictions on Tasks

Runs LayoutLMv3 on task pages and pushes predictions to Label Studio as draft annotations.

```bash
# All unannotated tasks
python prelabel.py \
  --model ./models/layoutlmv3-general \
  --api-key TOKEN \
  --project-id 4 \
  --model-version layoutlmv3-general-v2

# Specific task (bypasses skip-annotated check)
python prelabel.py \
  --model ./models/layoutlmv3-general \
  --api-key TOKEN \
  --project-id 4 \
  --task-id 3 \
  --model-version layoutlmv3-general-v2
```

---

## `prepare_dataset.py` — Build Training Dataset from Exports

Converts Label Studio export JSONs into LayoutLMv3 training format. OCRs each page with Tesseract and assigns BIO tags.

```bash
python prepare_dataset.py \
  --export-dir data/export/ \
  --out dataset-combined/
```

- Picks the annotation with the most regions per task (fixed 2026-05-20)
- Auto-renders missing page images from PDFs
- Outputs `train.json` and `label2id.json`

---

## `train.py` — Fine-tune LayoutLMv3

```bash
python train.py \
  --data dataset-combined/train.json \
  --output models/layoutlmv3-general \
  --epochs 10
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to train.json |
| `--output` | required | Where to save the model |
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 5e-5 | Learning rate |
| `--val-split` | 0.1 | Validation fraction |

- Uses MPS (Apple Silicon), CUDA, or CPU automatically
- Logs to TensorBoard: `{output}/runs/`
- Saves best model checkpoint

---

## `push_annotations.py` — Push Exports Back into Label Studio

Reads export JSONs and pushes their annotations to the matching Label Studio tasks.

```bash
python push_annotations.py
```

> Edit the `EXPORT_TO_TASK` dict in the script to map export files → task IDs.

---

## `image_server.py` — Serve Page Images

Starts a CORS-enabled HTTP server so Label Studio can display page images.

```bash
python image_server.py
# Runs on port 9090
```

---

## `inference_pipeline.py` — Batch PDF → JSON

Production pipeline: PDF → LayoutLMv3 layout detection → pdfplumber text extraction → structured JSON.

```bash
# Single PDF
python inference_pipeline.py \
  --model ./models/layoutlmv3-general \
  --pdf data/pdfs/MyDocument.pdf

# All PDFs in a directory
python inference_pipeline.py \
  --model ./models/layoutlmv3-general \
  --pdf-dir data/pdfs/ \
  --out-dir outputs/
```

Output JSON structure per page:
```json
{
  "page_number": 1,
  "blocks": [
    {
      "label": "H1_Heading",
      "score": 0.94,
      "bbox_image": [x0, y0, x1, y1],
      "bbox_pdf": [x0, y0, x1, y1],
      "text": "Chapter 1: Introduction"
    }
  ]
}
```

---

## API Token

```
c726dc3f9c729652f3316343963f4feade22bcc0
```

Retrieve fresh from DB:
```bash
sqlite3 data/label_studio.sqlite3 "SELECT key FROM authtoken_token LIMIT 1;"
```
