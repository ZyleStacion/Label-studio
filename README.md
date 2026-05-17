# LayoutLMv3 PDF Annotation Pipeline

End-to-end pipeline for annotating PDF documents in Label Studio and training LayoutLMv3 for document layout analysis.

## Prerequisites

> Python 3.11 is required. Label Studio is not compatible with Python 3.12+.

---

### macOS

**System dependencies**
```bash
brew install poppler tesseract
```

**Option A — conda (recommended)**
```bash
conda create -n labelstudio python=3.11 -y
conda activate labelstudio
pip install label-studio==1.23.0 pdf2image pytesseract pdfplumber transformers torch torchvision Pillow requests
```

**Option B — pip + virtualenv**
```bash
# If python3.11 is not available: brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install label-studio==1.23.0 pdf2image pytesseract pdfplumber transformers torch torchvision Pillow requests
```

When using the venv, replace `conda activate labelstudio` with `source .venv/bin/activate` throughout this guide, and replace `./start.sh` with:
```bash
source .venv/bin/activate
LOCAL_FILES_SERVING_ENABLED=true \
LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/data/images \
label-studio start --data-dir data/
```

---

### Windows

**System dependencies**

1. **Poppler** — download the latest Windows build from [github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases), extract it, and add the `bin\` folder to your `PATH`.

2. **Tesseract** — download the installer from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) and install it. Note the install path (default: `C:\Program Files\Tesseract-OCR`).

3. **Tell pytesseract where Tesseract is** — add this to the top of any script that uses OCR, or set it once as a system environment variable:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

**Option A — conda (recommended)**

Open Anaconda Prompt:
```bat
conda create -n labelstudio python=3.11 -y
conda activate labelstudio
pip install label-studio==1.23.0 pdf2image pytesseract pdfplumber transformers torch torchvision Pillow requests
```

**Option B — pip + virtualenv**

Open PowerShell (Python 3.11 must be installed from [python.org](https://www.python.org/downloads/)):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install label-studio==1.23.0 pdf2image pytesseract pdfplumber transformers torch torchvision Pillow requests
```

> If you see "running scripts is disabled", run PowerShell as Administrator and execute:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Starting servers on Windows**

Use `start.ps1` instead of `start.sh`:
```powershell
# Terminal 1 — Image server
conda activate labelstudio   # or: .venv\Scripts\Activate.ps1
python image_server.py

# Terminal 2 — Label Studio
.\start.ps1
```

**Getting your API token on Windows** (no `sqlite3` by default):
```powershell
# Option 1: install sqlite3 via winget
winget install SQLite.SQLite
sqlite3 data\label_studio.sqlite3 "SELECT key FROM authtoken_token LIMIT 1;"

# Option 2: use Python
python -c "import sqlite3; c=sqlite3.connect('data/label_studio.sqlite3'); print(c.execute('SELECT key FROM authtoken_token').fetchone()[0])"
```

---

## 1. Start the servers

Every session requires two terminals:

**Terminal 1 — Image server (port 9090)**
```bash
conda activate labelstudio
python image_server.py
```

**Terminal 2 — Label Studio (port 8080)**
```bash
./start.sh
```

Open [http://localhost:8080](http://localhost:8080). Your API token:
```bash
sqlite3 data/label_studio.sqlite3 "SELECT key FROM authtoken_token LIMIT 1;"
```

---

## 2. Import PDFs

Drop PDF files into `data/pdfs/`, then run:

```bash
python import_pdfs.py --api-key TOKEN --project-id 4
```

This converts each PDF to page images at 150 DPI (saved to `data/images/<pdf-stem>/`) and creates one Label Studio task per PDF with all pages listed. Already-converted pages are skipped on re-runs.

---

## 3. Annotate in Label Studio

Open [http://localhost:8080/projects/4/](http://localhost:8080/projects/4/) and click a task.

**Controls:**
- Draw bounding boxes around each document element
- Select the label from the right panel
- Use the page arrows (◀ ▶) to move between pages within a document
- Per-region fields: transcription text, confidence (High/Medium/Low), review flags

**Label reference:**

| Group | Labels |
|---|---|
| Headings | H0_Part_Page, H1_Heading, H2_Subheading, H3_Stylistic, H4, H5 |
| Body | Paragraph_Text, List_Item |
| Front matter | Cover_Page, Table_of_Contents, Executive_Summary, Specific_Front_Matter |
| Page furniture | Footnote_Text, Page_Number, Running_Header_Footer |
| Visual | Table, Figure_Image, Caption, Image_with_Embedded_Text |
| Metadata | Title_Block, Author, Date, Jurisdiction |
| Other | Unclear_Needs_Review |

**Seed set strategy:** annotate 5–7 PDFs with diverse layouts (title-heavy reports, tables-heavy, image-heavy) before training the first model.

---

## 4. Export annotations for training

After annotating, pull the data directly from Label Studio:

```bash
python export_to_layoutlmv3.py --api-key TOKEN --project-id 4 --out dataset/
```

Output:
- `dataset/train.json` — list of page-level examples with `words`, `bboxes` (0–1000 scale), `ner_tags` (BIO format)
- `dataset/label2id.json` — label mapping for the model

---

## 5. Fine-tune LayoutLMv3

Train on the exported dataset using HuggingFace Trainer. Point `--output-dir` to wherever you want to save the checkpoint:

```python
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
)
# Load dataset/train.json, create a datasets.Dataset, then:
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=49,   # matches label2id.json
)
# ... standard HuggingFace training loop
```

Save the checkpoint to `models/layoutlmv3-finetuned/`.

---

## 6. Human-in-the-Loop: pre-label remaining PDFs

After training, push model predictions into Label Studio so reviewers fix boxes instead of drawing from scratch:

```bash
python prelabel.py \
  --model models/layoutlmv3-finetuned \
  --api-key TOKEN \
  --project-id 4
```

Label Studio will show the model's bounding boxes as draft annotations on unannotated tasks. Reviewers accept correct ones and fix wrong ones — much faster than annotating from blank.

**HITL cycle:**
1. Annotate 5–7 seed PDFs manually → export → fine-tune
2. Pre-label next batch of 3–5 PDFs → review & fix → export → retrain
3. Repeat — model improves with each batch

---

## 7. Production inference

Run the full pipeline on new PDFs to get structured JSON output:

```bash
# Single PDF
python inference_pipeline.py \
  --model models/layoutlmv3-finetuned \
  --pdf path/to/document.pdf \
  --out-dir outputs/

# Whole directory
python inference_pipeline.py \
  --model models/layoutlmv3-finetuned \
  --pdf-dir data/pdfs/ \
  --out-dir outputs/
```

**Pipeline:** PDF → render pages → LayoutLMv3 detects blocks → pdfplumber extracts text inside each box → structured JSON

**Output format** (`outputs/document.json`):
```json
{
  "pdf": "document.pdf",
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {
          "label": "H1_Heading",
          "score": 0.97,
          "bbox_image": [42, 80, 650, 110],
          "bbox_pdf":   [30.1, 57.6, 468.0, 79.2],
          "text": "1. Introduction"
        }
      ]
    }
  ]
}
```

`bbox_image` is pixel coordinates on the rendered image. `bbox_pdf` is PDF point coordinates usable directly with pdfplumber.

---

## Directory layout

```
data/pdfs/          ← drop source PDFs here
data/images/        ← rendered page PNGs (one subdir per PDF, auto-generated)
data/               ← Label Studio database and media
models/             ← fine-tuned model checkpoints
dataset/            ← exported training data
outputs/            ← inference JSON results
inference/          ← shared predictor code
configs/            ← Label Studio labeling XML config
```

---

## Troubleshooting

**Images not loading in Label Studio**
Make sure `image_server.py` is running on port 9090. Label Studio tasks use `http://localhost:9090/...` URLs.

**Label Studio crashes on startup**
The `data/.env` file must exist with `SENTRY_DSN=` (empty). It's already committed — don't delete it.

**Wrong Python used by scripts**
Always activate the environment first: `conda activate labelstudio` (macOS/Windows) or `source .venv/bin/activate` (macOS venv) or `.venv\Scripts\Activate.ps1` (Windows venv).

**API token expired / not working**
Retrieve it fresh from the database:
```bash
# macOS / Linux
sqlite3 data/label_studio.sqlite3 "SELECT key FROM authtoken_token LIMIT 1;"

# Windows (PowerShell)
python -c "import sqlite3; c=sqlite3.connect('data/label_studio.sqlite3'); print(c.execute('SELECT key FROM authtoken_token').fetchone()[0])"
```

**Windows: `pdf2image` fails with "Unable to get page count"**
Poppler is not on your PATH. Either add `C:\path\to\poppler\bin` to your system PATH, or pass the path directly in `import_pdfs.py`:
```python
pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=r"C:\path\to\poppler\bin")
```

**Windows: pytesseract raises `TesseractNotFound`**
Set the path at the top of the script:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```
