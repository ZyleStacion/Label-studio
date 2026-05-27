# LayoutLMv3 Pipeline — Session Summary

## What We've Done

### Pipeline Fixes & New Scripts
- **`prepare_dataset.py`** — fixed annotation selection to pick the largest annotation (most regions) instead of the first one, ensuring corrected annotations are always used for training
- **`import_pdfs.py`** — added deduplication so re-running won't create duplicate tasks in Label Studio
- **`push_annotations.py`** — new script to push existing export JSONs back into Label Studio as annotations

### Label Studio
- Imported **5 new PDFs** as tasks — project went from 2 → 7 tasks total
- Pushed existing human annotations into Tasks 4–7 from export files
- All 7 tasks now have `layoutlmv3-general-v2` predictions ready to review

### Training Runs

| Run | Training Data | Val Accuracy | Weighted F1 | Macro F1 |
|-----|--------------|-------------|-------------|----------|
| `layoutlmv3-bail-act` | 219 examples (Bail Act only) | — | — | — |
| `layoutlmv3-general` v1 | 925 examples (5 documents) | 78.1% | 0.751 | 0.282 |
| `layoutlmv3-general` v2 | 925 examples (corrected Jury Empanelment) | 76.9% | 0.732 | 0.314 |

---

## Current Model Status

### `models/layoutlmv3-bail-act`
- Trained on Bail Act Report only (219 pages)
- ~50% confidence on unseen documents
- Use only for Bail Act documents

### `models/layoutlmv3-general` ← active model
- Trained on 5 documents, 925 examples
- Val accuracy: **76.9%** | Weighted F1: **0.73** | Macro F1: **0.31**
- Strong labels: Running_Header_Footer (0.98), Paragraph_Text (0.93), Caption (0.86)
- Weak labels: Author, Title_Block, Jurisdiction — rare in training data, need more annotated examples

---

## Current Task Status in Label Studio

| Task | PDF | Pages | Human Annotations | Predictions |
|------|-----|-------|------------------|-------------|
| 1 | Review-of-Family-Violence-Laws-Report | 505 | None | layoutlmv3-general-v2 |
| 2 | VLRC_Jury_Empanelment_Report | 151 | ✅ Corrected | layoutlmv3-general-v2 |
| 3 | Funeral_and_Burial_Instructions_Report | 164 | None | layoutlmv3-general-v2 |
| 4 | ResidentialTenancyDatabases_FinalReport | 66 | ✅ Imported | layoutlmv3-general-v2 |
| 5 | Review_of_the_Bail_Act_Report_Web | 228 | ✅ Imported | layoutlmv3-general-v2 |
| 6 | VLRC_Medicinal_Cannabis_Report_web | 288 | ✅ Imported | layoutlmv3-general-v2 |
| 7 | VLRC_Recklessness_Report_fnl_Parl | 244 | ✅ Imported | layoutlmv3-general-v2 |

---

## Annotation → Retrain Cycle

```
1. Open Label Studio (http://localhost:8080)
2. Open a task → find layoutlmv3-general-v2 prediction (robot icon)
3. Click "..." → "Copy to annotation"
4. Correct the annotations (fix labels, resize boxes, delete/add regions)
5. Export → JSON → save to data/export/
6. Rebuild dataset:
   python prepare_dataset.py --export-dir data/export/ --out dataset-combined/
7. Retrain:
   python train.py --data dataset-combined/train.json --output models/layoutlmv3-general --epochs 10
8. Re-run predictions:
   python prelabel.py --model ./models/layoutlmv3-general --api-key TOKEN --project-id 4 --task-id N --model-version layoutlmv3-general-v3
```

## Services

| Service | Command | URL |
|---------|---------|-----|
| Label Studio | `label-studio start --data-dir data/` | http://localhost:8080 |
| Image server | `python image_server.py` | http://localhost:9090 |
| TensorBoard | `tensorboard --logdir models/layoutlmv3-general/runs` | http://localhost:6006 |

## API Token
```
c726dc3f9c729652f3316343963f4feade22bcc0
```
Retrieve from DB: `sqlite3 data/label_studio.sqlite3 "SELECT key FROM authtoken_token LIMIT 1;"`

## Next Steps
1. Correct Task 3 (Funeral & Burial) — duplicate prediction → fix → export
2. Retrain for v3 model
3. Continue annotation cycle for Tasks 1, 3 (no human annotations yet)
4. Focus corrections on weak labels: **Author, Title_Block, Jurisdiction**
