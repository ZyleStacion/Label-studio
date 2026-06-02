---
tags: [model, status, performance]
created: 2026-05-20
---

# Model Status

## Active Model — `layoutlmv3-general` (v2)

> [!success] Currently in use for all predictions

| Metric | Value |
|--------|-------|
| Base model | microsoft/layoutlmv3-base |
| Training examples | 925 (5 documents) |
| Val accuracy | 76.9% |
| Weighted F1 | 0.732 |
| Macro F1 | 0.314 |
| Epochs trained | 10 |
| Device | Apple MPS |
| Saved to | `models/layoutlmv3-general` |

### Per-Class Performance

**Strong labels (F1 ≥ 0.85)**

| Label | F1 |
|-------|----|
| Running_Header_Footer | 0.98 |
| Paragraph_Text | 0.93 |
| Caption | 0.86 |
| Footnote_Text | 0.86 |

**Weak labels (F1 = 0.00) — needs more annotated examples**

| Label | Issue |
|-------|-------|
| Author | Rarely appears in training data |
| Title_Block | Rarely appears in training data |
| Jurisdiction | Rarely appears in training data |
| Image_with_Embedded_Text | Rarely appears in training data |

> [!tip] How to improve weak labels
> When correcting annotations, pay special attention to Author, Title_Block, and Jurisdiction regions. Even 10–20 more annotated examples of each will significantly improve their F1.

---

## Previous Model — `layoutlmv3-bail-act`

| Metric | Value |
|--------|-------|
| Training examples | 219 (Bail Act only) |
| Confidence on unseen docs | ~50% |
| Saved to | `models/layoutlmv3-bail-act` |

> [!warning] Limited use
> This model was trained on a single document type. It performs poorly on other VLRC reports. Retained for reference only.

---

## Prediction Confidence by Task

| Task | Model | Score |
|------|-------|-------|
| Task 2 — VLRC_Jury_Empanelment | layoutlmv3-bail-act | 50.3% |
| Task 2 — VLRC_Jury_Empanelment | layoutlmv3-general v1 | 75.4% |
| Task 3 — Funeral & Burial | layoutlmv3-general v1 | 66.7% |
| Task 3 — Funeral & Burial | layoutlmv3-general v2 | 66.3% |

> [!note]
> Task 3 scores ~66% because the Funeral & Burial report is unseen training data. Once annotated and included in training, confidence will rise significantly.
