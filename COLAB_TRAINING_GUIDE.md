# LayoutLMv3 Training on Google Colab

## Before You Start

You need `colab_training.zip` uploaded to your Google Drive.  
Ask Ye Thu for the zip file if you don't have it.

---

## Step 0 — Set Runtime to GPU (CRITICAL)

> [!IMPORTANT]  
> If you skip this, training will take 40+ hours instead of ~1 hour.

1. In Colab menu: **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**

You should see a green GPU indicator in the top right of Colab.

---

## Step 1 — Mount Google Drive & Unzip

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os

with zipfile.ZipFile('/content/drive/MyDrive/colab_training.zip', 'r') as z:
    z.extractall('/content/layoutlm/')

os.chdir('/content/layoutlm')
print("Done — files extracted")
```

> If your zip is in a subfolder on Drive, update the path:  
> e.g. `/content/drive/MyDrive/Team/colab_training.zip`

---

## Step 2 — Install Dependencies

```python
!pip install transformers torch torchvision pillow \
             pytesseract scikit-learn tensorboard accelerate -q
!apt-get install -y tesseract-ocr -q
print("Dependencies installed")
```

---

## Step 3 — Fix Image Paths

The training data was created on a Mac — image paths need to be updated for Colab.

```python
import json, re

with open('dataset-combined/train.json') as f:
    examples = json.load(f)

for ex in examples:
    ex['image_path'] = re.sub(
        r'.*/data/images/',
        '/content/layoutlm/data/images/',
        ex['image_path']
    )

with open('dataset-combined/train.json', 'w') as f:
    json.dump(examples, f)

print(f"Fixed {len(examples)} examples")
print("Sample path:", examples[0]['image_path'])
```

---

## Step 4 — Verify GPU is Active

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU — go back to Step 0")
```

Expected output:
```
CUDA available: True
Device: Tesla T4
```

---

## Step 5 — Run Training

```python
!python train.py \
  --data dataset-combined/train.json \
  --output models/layoutlmv3-general \
  --epochs 10 \
  --batch-size 4
```

### What you should see:
```
Epoch 1/10 | Step 1/208 | Loss 3.80 | LR 2.40e-07 | GradNorm 8.7 | 0.8s/step
```

> **If `s/step` is above 5s** — GPU is not active. Stop and redo Step 0.

### Expected training time:
| GPU | Time |
|-----|------|
| T4 (free Colab) | ~40–60 min |
| A100 (Colab Pro) | ~10–15 min |

---

## Step 6 — Save Model to Google Drive

Run this after training completes:

```python
import shutil
shutil.copytree(
    'models/layoutlmv3-general',
    '/content/drive/MyDrive/layoutlmv3-general',
    dirs_exist_ok=True
)
print("Model saved to Google Drive")
```

---

## Step 7 — Download Model Locally

Once saved to Drive, download the `layoutlmv3-general` folder and place it at:

```
Label-studio/models/layoutlmv3-general/
```

Then re-run predictions in Label Studio:

```bash
conda activate labelstudio

python prelabel.py \
  --model ./models/layoutlmv3-general \
  --api-key YOUR_API_TOKEN \
  --project-id 4 \
  --task-id N \
  --model-version layoutlmv3-general-v3
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `70s/step` or higher | GPU not active — redo Step 0 |
| `CUDA available: False` | Runtime not set to GPU — redo Step 0 |
| `FileNotFoundError: image_path` | Step 3 (fix paths) was skipped or failed |
| Colab disconnects mid-training | Re-run from Step 1 — training starts fresh |
| `zip file not found` | Check the path in Step 1 matches where you uploaded on Drive |
| MISSING/UNEXPECTED warnings | Normal — safe to ignore |

---

## Notes

- Colab free tier may disconnect after ~90 min of inactivity — keep the tab open
- Do not close the browser while training
- The model is saved to Drive in Step 6 — even if Colab resets, your model is safe
