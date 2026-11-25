#  Fruit Ripeness Grader

> **DCIT 407 — Image Processing Semester Project**  
> Transfer-learning CNN that classifies fruit images as **unripe**, **ripe**, or **overripe**.

---

## Overview

This project fine-tunes a pretrained **EfficientNet-B2** backbone on the *Fruit Ripeness Level Dataset* from Kaggle. It covers the full ML pipeline — data loading, stratified splitting, per-dataset normalisation, training with early stopping, evaluation, and model export — entirely within a single Jupyter notebook designed to run on **Kaggle Kernels** (free GPU).

---

## Repository Structure

```
fruit-ripeness-grader07/
├── fruit-grader.ipynb   # Main notebook (all code)
└── README.md
```

---

## Notebook Walkthrough

| # | Section | What it does |
|---|---|---|
| 1 | **Imports & Device** | Libraries, seeds, CUDA/CPU detection |
| 2 | **Dataset Loading** | Scans `Fruit Ripeness Level Dataset/`, builds a `DataFrame` of paths + labels |
| 3 | **Stratified Split** | 70 % train / 15 % val / 15 % test (stratified by class) |
| 4 | **Normalisation Stats** | Computes per-channel mean & std from the **train split only** |
| 5 | **Augmentation** | Train: random crop, flip, rotation ±15°, colour jitter · Eval: resize only |
| 6 | **Model Definition** | EfficientNet-B2 (ImageNet weights) → custom head: `Dropout(0.3) → Linear(NUM_CLASSES)` |
| 7 | **Loss / Optimiser / Scheduler** | CrossEntropyLoss · AdamW (lr 3e-4) · CosineAnnealingLR |
| 8 | **Training Loop** | Up to 50 epochs, early stopping (patience = 7), best-model checkpoint |
| 9 | **Loss & Accuracy Curves** | Side-by-side plots saved as `training_curves.png` |
| 10 | **Test Evaluation** | Classification report + confusion matrix heatmap (`confusion_matrix.png`) |
| 11 | **Model Export** | Full checkpoint + weights-only file; optional push to Kaggle Models |

---

## Model

| Property | Value |
|---|---|
| Backbone | EfficientNet-B2 (pretrained, ImageNet) |
| Input size | 224 × 224 × 3 |
| Head | Dropout(0.3) → Linear → *N* classes |
| Loss | Cross-Entropy |
| Optimiser | AdamW — lr `3e-4`, weight decay `1e-4` |
| Scheduler | Cosine Annealing (`T_max = 50`) |
| Batch size | 64 |
| Max epochs | 50 (early stopping, patience = 7) |

---

## Outputs

After a completed run the following files appear in `/kaggle/working/` (Output tab):

| File | Description |
|---|---|
| `best_model.pt` | Best-epoch weights (saved during training) |
| `fruit_ripeness_full.pt` | Full checkpoint — weights, optimizer state, class map, normalisation stats, history |
| `fruit_ripeness_weights.pt` | Weights-only — lighter file for inference / deployment |
| `training_curves.png` | Loss & accuracy curves with best-epoch marker |
| `confusion_matrix.png` | Per-class confusion matrix on the test set |

---

## Running on Kaggle

1. **Upload** this notebook to [kaggle.com/kernels](https://www.kaggle.com/kernels).
2. **Add the dataset** — click *+ Add Data* and search for *Fruit Ripeness Level Dataset*. Update `ROOT` in cell 5 with the correct dataset slug.
3. **Enable GPU** — *Session options → Accelerator → GPU T4 × 1*.
4. **Run All** — *Run → Run All*.
5. Download outputs from the **Output** tab when the kernel finishes.

> **Internet** must be **ON** only if you want to push to Kaggle Models (last cell). Training itself does not need internet once the dataset is attached.

---

## Running Locally

```bash
# 1. Clone
git clone git@github.com:DCIT-407-IMAGE-PROCESSING/fruit-ripeness-grader07.git
cd fruit-ripeness-grader07

# 2. Create & activate virtual environment
python3 -m venv .imgvenv
source .imgvenv/bin/activate

# 3. Install dependencies
pip install torch torchvision torchsummary scikit-learn \
            pandas matplotlib seaborn tqdm Pillow opencv-python

# 4. Edit ROOT in cell 5 to point to your local dataset path
# 5. Open & run
jupyter notebook fruit-grader.ipynb
```

---

## Dependencies

```
torch · torchvision · torchsummary
scikit-learn · pandas · numpy
matplotlib · seaborn
tqdm · Pillow · opencv-python
```

---

## Course

**DCIT 407 — Image Processing**  
University of Ghana, Department of Computer Science  
Semester Project
