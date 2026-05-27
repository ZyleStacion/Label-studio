---
tags: [training, dataset, history]
created: 2026-05-20
---

# Training History

## Dataset Builds

### Combined Dataset v2 (current)
- **File:** `dataset-combined/train.json`
- **Total examples:** 925
- **Train / Val split:** 833 / 92

| Document                                | Pages Annotated | Training Examples |
| --------------------------------------- | --------------- | ----------------- |
| Review_of_the_Bail_Act_Report_Web       | 219             | 219               |
| VLRC_Medicinal_Cannabis_Report_web      | 278             | 274               |
| VLRC_Recklessness_Report_fnl_Parl       | 224             | 220               |
| VLRC_Jury_Empanelment_Report            | 147             | 146               |
| ResidentialTenancyDatabases_FinalReport | 66              | 66                |
| **Total**                               | **1,134**       | **925**           |

> [!note] Jury Empanelment annotations were corrected manually — the model's predictions were duplicated and fixed by the annotator before being exported.

---

## Training Runs

### Run 1 — `layoutlmv3-bail-act`
- **Date:** Pre-session
- **Data:** 219 examples (Bail Act only)
- **Result:** Specialist model, ~50% on unseen documents

### Run 2 — `layoutlmv3-general` v1
- **Date:** 2026-05-19
- **Data:** 925 examples (5 documents, first combined build)
- **Epochs:** 10 | **Batch size:** 2 | **LR:** 5e-5

| Epoch | Train Loss | Val Loss | Val Acc | Weighted F1 |
|-------|-----------|----------|---------|-------------|
| 1 | 1.891 | 1.544 | 62.0% | 0.540 |
| 3 | 1.024 | 1.231 | 72.6% | 0.666 |
| 5 | 0.800 | 1.176 | 74.0% | 0.704 |
| 8 | 0.538 | 1.158 | 77.5% | 0.742 |
| 10 | 0.433 | 1.204 | 78.1% | 0.751 |

> [!info] Best val loss at epoch 8 (1.158) — slight overfitting after that.

### Run 3 — `layoutlmv3-general` v2 (current)
- **Date:** 2026-05-20
- **Data:** 925 examples (corrected Jury Empanelment annotations)
- **Epochs:** 10 | **Batch size:** 2 | **LR:** 5e-5

| Epoch | Train Loss | Val Loss | Val Acc | Weighted F1 | Macro F1  |
| ----- | ---------- | -------- | ------- | ----------- | --------- |
| 1     | 1.937      | 1.707    | 66.1%   | 0.576       | 0.052     |
| 3     | 1.055      | 1.395    | 72.5%   | 0.653       | 0.128     |
| 5     | 0.734      | 1.334    | 72.7%   | 0.674       | 0.234     |
| 8     | 0.556      | 1.223    | 76.8%   | 0.724       | 0.304     |
| 10    | 0.450      | 1.229    | 76.9%   | 0.732       | **0.314** |

> [!success] Macro F1 improved from 0.282 → 0.314 — model is getting better at less common labels.

---

## How to Rebuild & Retrain

```bash
conda activate labelstudio

# Step 1 — rebuild dataset from all exports
python prepare_dataset.py --export-dir data/export/ --out dataset-combined/

# Step 2 — retrain
python train.py \
  --data dataset-combined/train.json \
  --output models/layoutlmv3-general \
  --epochs 10

# Step 3 — watch progress
tensorboard --logdir models/layoutlmv3-general/runs --port 6006
```

## Possible Causes for low f1
- rare labels
- OCR noise
- inconsistent annotations
- layout complexity 