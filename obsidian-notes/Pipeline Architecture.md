---
tags: [architecture, pipeline]
created: 2026-05-20
---

# Pipeline Architecture

## End-to-End Flow

```
PDF Documents
     │
     ▼
┌─────────────────────────────┐
│  import_pdfs.py             │  Converts each PDF page → PNG at 150 DPI
│  Uploads to Label Studio    │  Each PDF = 1 task, pages served via image server
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Label Studio (port 8080)   │  Annotators draw bounding boxes + assign labels
│  Project ID: 4              │  24 label types, multi-page view
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  prelabel.py                │  Model runs on unannotated pages
│  (Human-in-the-Loop)        │  Pushes predictions as draft annotations
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  prepare_dataset.py         │  Export JSON → LayoutLMv3 training format
│                             │  OCR via Tesseract, BIO tag assignment
│                             │  Output: words, bboxes (0–1000), ner_tags
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  train.py                   │  Fine-tunes microsoft/layoutlmv3-base
│                             │  AdamW + linear warmup, MPS/CUDA/CPU
│                             │  TensorBoard logging, per-class F1 metrics
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Fine-tuned Model           │  Saved to models/layoutlmv3-general
│  models/layoutlmv3-general  │  Used for next round of pre-labelling
└─────────────────────────────┘
```

## Human-in-the-Loop Cycle

```
Model predicts → Annotator corrects → Export → Retrain → Better predictions
```
Each cycle improves model confidence and reduces annotator effort.

## Directory Layout

```
Label-studio/
├── data/
│   ├── pdfs/           ← source PDFs (drop new ones here)
│   ├── images/         ← auto-generated page PNGs (150 DPI)
│   ├── export/         ← Label Studio JSON exports (one file per PDF)
│   └── label_studio.sqlite3
├── models/
│   ├── layoutlmv3-bail-act/     ← v1 specialist model (Bail Act only)
│   └── layoutlmv3-general/      ← v2 general model (active)
├── dataset-combined/
│   ├── train.json      ← 925 training examples
│   └── label2id.json
├── inference/
│   └── predictor.py    ← shared inference logic (OCR + LayoutLMv3)
├── import_pdfs.py
├── prelabel.py
├── prepare_dataset.py
├── train.py
├── push_annotations.py
└── image_server.py
```

## Label Schema (24 classes)

| Group | Labels |
|-------|--------|
| Headings | H0_Part_Page, H1_Heading, H2_Subheading, H3_Stylistic, H4, H5 |
| Body | Paragraph_Text, List_Item |
| Front matter | Cover_Page, Table_of_Contents, Executive_Summary, Specific_Front_Matter |
| Page furniture | Footnote_Text, Page_Number, Running_Header_Footer |
| Visual elements | Table, Figure_Image, Caption, Image_with_Embedded_Text |
| Metadata | Title_Block, Author, Date, Jurisdiction |
| Other | Unclear_Needs_Review |
