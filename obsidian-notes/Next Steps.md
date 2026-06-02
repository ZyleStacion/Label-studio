---
tags: [planning, next-steps]
created: 2026-05-20
---

# Next Steps

## Immediate — Annotation

> [!todo] Priority order for annotation

- [ ] **Task 3 — Funeral & Burial Instructions (164 pages)**
  - Duplicate `layoutlmv3-general-v2` prediction → correct → export
  - This is the only unannotated task with a manageable page count

- [ ] **Task 1 — Family Violence Laws Report (505 pages)**
  - Largest document — consider splitting annotation work across team
  - Prediction is loaded and ready in Label Studio

## After Each Annotation Round — Retrain

```bash
# 1. Rebuild dataset
python prepare_dataset.py --export-dir data/export/ --out dataset-combined/

# 2. Retrain
python train.py \
  --data dataset-combined/train.json \
  --output models/layoutlmv3-general \
  --epochs 10

# 3. Re-run predictions on remaining tasks
python prelabel.py \
  --model ./models/layoutlmv3-general \
  --api-key c726dc3f9c729652f3316343963f4feade22bcc0 \
  --project-id 4 \
  --task-id N \
  --model-version layoutlmv3-general-v3
```

## Model Improvement Targets

| Issue | Action |
|-------|--------|
| Author F1 = 0.00 | Annotate more Author regions across all tasks |
| Title_Block F1 = 0.00 | Same — ensure all title blocks are labelled |
| Jurisdiction F1 = 0.00 | Same — rare but important for legal docs |
| Task 3 score only 66% | Annotate Task 3 → include in next training run |
| Task 1 completely unannotated | Annotate at least 50–100 pages → include in training |

## Week 6 Pre-work — Already Done ✅

- [x] `prepare_dataset.py` working and tested
- [x] `train.py` working end-to-end
- [x] Combined dataset built (925 examples)
- [x] Model trained and producing predictions
- [x] All 7 tasks loaded in Label Studio with predictions

> [!success] Week 6 training is unblocked. The pipeline is proven end-to-end.

## Architecture Diagram (v1.1)

- [ ] Meet with team to agree on absorb / extend / park decisions
- [ ] Update diagram based on team agreement
- [ ] See [[Pipeline Architecture]] for current architecture reference

## Team Read-out Talking Points

1. **What's built:** Full annotation → training → inference pipeline
2. **Model performance:** 76.9% accuracy, 0.73 weighted F1 on validation set
3. **Data scale:** 925 training examples across 5 VLRC documents, 1,646 total pages loaded
4. **Human-in-the-loop:** Model predictions cut annotation time significantly — annotators correct rather than draw from scratch
5. **Iteration speed:** Full retrain takes ~2 hours on Apple Silicon (MPS)
6. **Remaining work:** Tasks 1 & 3 need human annotation; weak labels (Author, Title_Block, Jurisdiction) need more examples
