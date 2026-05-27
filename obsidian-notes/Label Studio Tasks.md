---
tags: [annotation, tasks, labelstudio]
created: 2026-05-20
---

# Label Studio Tasks

**Project:** ID 4 | **URL:** http://localhost:8080
**API Token:** `c726dc3f9c729652f3316343963f4feade22bcc0`

## Task Overview

| Task | Document | Pages | Human Annotations | Latest Prediction | Prediction Score |
|------|----------|-------|------------------|-------------------|-----------------|
| 1 | Review-of-Family-Violence-Laws-Report | 505 | ❌ None | layoutlmv3-general-v2 | — |
| 2 | VLRC_Jury_Empanelment_Report | 151 | ✅ Corrected (1,099 regions) | layoutlmv3-general-v2 | 75.4% |
| 3 | Funeral_and_Burial_Instructions_Report | 164 | ❌ None | layoutlmv3-general-v2 | 66.3% |
| 4 | ResidentialTenancyDatabases_FinalReport | 66 | ✅ Imported (453 regions) | layoutlmv3-general-v2 | — |
| 5 | Review_of_the_Bail_Act_Report_Web | 228 | ✅ Imported (1,674 regions) | layoutlmv3-general-v2 | — |
| 6 | VLRC_Medicinal_Cannabis_Report_web | 288 | ✅ Imported (1,395 regions) | layoutlmv3-general-v2 | — |
| 7 | VLRC_Recklessness_Report_fnl_Parl | 244 | ✅ Imported (1,383 regions) | layoutlmv3-general-v2 | — |

**Total pages:** 1,646 | **Annotated pages:** ~1,134 | **Unannotated:** ~512

---

## Task Detail

### Task 1 — Review of Family Violence Laws Report
- **Pages:** 505 (largest document)
- **Status:** Predictions only — no human annotations yet
- **Priority:** Low (largest effort, annotate last)

### Task 2 — VLRC Jury Empanelment Report ✅
- **Pages:** 151
- **Status:** Human-corrected annotations exported and used in training
- **Annotation history:**
  - Model predicted 2,983 regions (layoutlmv3-bail-act, 50% confidence)
  - Model predicted 1,276 regions (layoutlmv3-general v1, 75% confidence)
  - Annotator duplicated prediction → manually corrected → 1,099 regions
  - Corrected export used in v2 training run

### Task 3 — Funeral and Burial Instructions Report
- **Pages:** 164
- **Status:** Predictions only — needs human correction
- **Priority:** High (next to annotate)
- **Note:** No prior human annotations existed for this document

### Task 4 — Residential Tenancy Databases ✅
- **Pages:** 66
- **Status:** Original human annotations imported (453 regions, 66 pages annotated)
- **Source:** `data/export/ResidentialTenancyDatabases_FinalReport.json`

### Task 5 — Review of the Bail Act Report ✅
- **Pages:** 228
- **Status:** Original human annotations imported (1,674 regions, 219 pages annotated)
- **Source:** `data/export/Review_of_the_Bail_Act_Report_Web.json`

### Task 6 — VLRC Medicinal Cannabis Report ✅
- **Pages:** 288
- **Status:** Original human annotations imported (1,395 regions, 278 pages annotated)
- **Source:** `data/export/VLRC_Medicinal_Cannabis_Report_web.json`

### Task 7 — VLRC Recklessness Report ✅
- **Pages:** 244
- **Status:** Original human annotations imported (1,383 regions, 224 pages annotated)
- **Source:** `data/export/VLRC_Recklessness_Report_fnl_Parl.json`

---

## Annotation Workflow (per task)

```
1. Open Label Studio → http://localhost:8080
2. Select task
3. In right panel, find prediction with robot icon (layoutlmv3-general-v2)
4. Click "..." → "Copy to annotation"
5. Review page by page:
   - Fix wrong labels (click box → change label)
   - Resize boxes that miss content
   - Delete false positives
   - Draw new boxes for missed regions
6. Focus on weak labels: Author, Title_Block, Jurisdiction
7. When done → Export → JSON → save to data/export/
```

---

## Export Files in `data/export/`

| File | Document | Regions |
|------|----------|---------|
| VLRC_Jury_Empanelment_Report.json | Jury Empanelment | 1,099 |
| Review_of_the_Bail_Act_Report_Web.json | Bail Act | 1,674 |
| VLRC_Medicinal_Cannabis_Report_web.json | Medicinal Cannabis | 1,395 |
| VLRC_Recklessness_Report_fnl_Parl.json | Recklessness | 1,383 |
| ResidentialTenancyDatabases_FinalReport.json | Residential Tenancy | 453 |
