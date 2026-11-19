# Generalization Gap & Data Augmentation Ablation in Colorectal Cancer Histopathology Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Datasets-NCT--CRC--HE--100K%20%7C%20CRC--VAL--HE--7K-blue)](https://zenodo.org/record/1214456)
[![Demo](https://img.shields.io/badge/Prototype-Gradio%20%2B%20Grad--CAM-orange)](https://gradio.app/)

> **Research Study:** *Multi-Class Machine Learning Model for Automated Detection and Classification of Cancer Using Image Classification and Deep Learning*  
> **Authors:** Koushiki Ghosh, Kunal Maheshwari, and Dr. Sagithya T*  
> **Affiliation:** Department of Mathematics, School of Advanced Sciences, Vellore Institute of Technology (VIT), Chennai, India

---

## Abstract

Convolutional neural networks (CNNs) trained to classify hematoxylin and eosin (H&E) colorectal cancer tissue patches routinely report ~99% accuracy on held-out internal splits, yet suffer substantial accuracy collapse when evaluated on unseen patients. This study conducts a controlled empirical ablation using a **50-layer Residual Network (ResNet50) trained strictly from scratch** (`weights=None`) on **NCT-CRC-HE-100K** (100,000 patches, 86 slides) under four isolated augmentation regimes—**Baseline**, **Geometric-only (Spatial-only)**, **Colour-only (Stain-only)**, and **Combined**—and evaluates generalization on **CRC-VAL-HE-7K** (7,180 patches from 50 independent patients).

Without augmentation, accuracy drops by **11.12 percentage points** (98.99% internal → 87.87% external), driven primarily by failures in **adipose (61.3% recall)** and **stroma (55.1% recall)**. Every augmented regime significantly narrows this gap by **+5.15 to +5.74 pp** (exact McNemar $p < 0.001$), reaching **93.02%–93.61%** external accuracy with **no statistically significant difference among the three augmented regimes** ($p = 0.057\text{–}0.621$, Bonferroni-corrected $\alpha = 0.0083$). Although colour jitter confers strong invariance under a 28-setting synthetic colour stress test (~92% vs. ~46% mean accuracy), geometric augmentation alone achieves statistically indistinguishable external accuracy ($p = 0.250$). On this Macenko-normalized benchmark, **residual colour variation does not appear to be the primary cause of the external drop**; implicit regularization against shortcut learning plays the larger role, and **Colour-only (Stain-only)** emerges as the best-supported strategy (ranking first on 6 of 10 evaluation criteria while breaking the fewest baseline-correct predictions).

---

## Research Gaps Addressed

1. **Optimistic Evaluation via Patch-Level Leakage:** Random patch-level splits place tiles from the same slide into both training and test partitions, inflating internal metrics and obscuring true patient-level generalization.
2. **Leaderboard Chasing vs. Failure Analysis:** Existing studies vary simultaneously in architecture, ImageNet pretraining, ensembles, and outlier filtering, making direct comparison impossible and leaving *why* accuracy falls—and *which* tissue classes break—unexplained.
3. **Unisolated Augmentation Factors:** Prior histopathology benchmarks primarily compare stain normalization against stain augmentation rather than cleanly separating **geometric** vs. **colour** vs. **combined** augmentations on a fixed from-scratch backbone using paired statistical tests and per-class fixed/broken auditing.
4. **Role of Residual Colour Shift After Macenko Normalization:** Even on colour-normalized slides (`NCT-CRC-HE-100K`), it remained untested whether remaining colour differences explain external domain shift or whether non-colour inter-patient/inter-slide cues dominate.

---

## Research Objectives

1. **Quantify the Generalization Gap:** Measure the exact internal-to-external performance drop of an unaugmented ResNet50 trained from scratch across all nine colorectal tissue classes.
2. **Isolate Augmentation Mechanisms:** Train `Baseline`, `Geometric-only`, `Colour-only`, and `Combined` models under identical data splits, hyperparameters, and training schedules to isolate each regime's contribution.
3. **Perform Rigorous Paired Statistical & Per-Class Auditing:** Apply Bonferroni-corrected exact McNemar tests across all six pairwise model comparisons, quantify images **fixed** vs. **broken** per class, and explicitly track **tumour recall and precision** as clinical safety checks.
4. **Test the Colour-Shift Hypothesis:** Quantify RGB and optical-density (OD) shifts between datasets and subject all four models to a 28-setting synthetic colour stress test to determine whether colour variation drives the accuracy drop.

---

## Datasets & Class Distribution

Both datasets consist of $224 \times 224$ px H&E histopathology patches at $0.5\,\mu\text{m/pixel}$ across **9 tissue classes** (originally curated and Macenko-normalized by Kather et al., 2019):

| Code | Tissue Class | Internal Test Set (`NCT-CRC-HE-100K`, 15% split) | External Cohort (`CRC-VAL-HE-7K`, 50 patients) |
| :--- | :--- | :---: | :---: |
| `ADI` | Adipose | 1,613 | 1,338 |
| `BACK` | Background | 1,575 | 847 |
| `DEB` | Debris | 1,765 | 339 |
| `LYM` | Lymphocytes | 1,684 | 634 |
| `MUC` | Mucus | 1,297 | 1,035 |
| `MUS` | Smooth Muscle | 2,056 | 592 |
| `NORM` | Normal Mucosa | 1,303 | 741 |
| `STR` | Cancer-Associated Stroma | 1,579 | 421 |
| `TUM` | Colorectal Adenocarcinoma Epithelium (Tumour) | 2,128 | 1,233 |
| **Total** | **9 Classes** | **15,000** | **7,180** |

* **Internal Split (`NCT-CRC-HE-100K`, $n = 100,000$ from 86 FFPE slides):** Reproducible random patch-level split (`seed=42`) into **70,000 train / 15,000 validation / 15,000 internal test**. Because tiles from the same slide span partitions, internal accuracy is explicitly treated as an **optimistic upper bound**.
* **External Cohort (`CRC-VAL-HE-7K`, $n = 7,180$ from 50 patients):** Strictly held out for zero-shot patient-level testing; zero patient overlap with training data.

---

## Experimental Methodology

### Fixed Architecture & Training Protocol
To avoid confounding augmentation effects with ImageNet pretraining or tuning heuristics, all four models share an identical pipeline:
* **Backbone:** `torchvision.models.resnet50(weights=None)` (~23.5M parameters), final fully connected layer replaced with `Linear(2048, 9)`.
* **Optimization:** Cross-Entropy Loss, Adam optimizer (`lr=1e-3`, default $\beta$), batch size `32`, `30` epochs, ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
* **Selection Discipline:** No learning-rate scheduler, weight decay, mixed precision, or early stopping; **final-epoch (epoch 30) weights** are evaluated directly. Validation set used strictly for monitoring.
* **Compute Environment:** Python `3.12.13`, PyTorch on NVIDIA Tesla T4 GPU (~6.4 hours per regime).

### Augmentation Regimes (Training Split Only)

| Paper Name | Notebook Label | Training Transforms Applied On-the-Fly |
| :--- | :--- | :--- |
| **Baseline** | `Baseline` | None (`Resize(224, 224)` + ImageNet normalization only) |
| **Geometric-only** | `Spatial-only` | `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)`, `RandomRotation(degrees=90)` |
| **Colour-only** | `Stain-only` | `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` |
| **Combined** | `Combined` | Geometric transforms followed by `ColorJitter` |

---

## Results

### 1. Overall Generalization & 10-Criterion Model Selection (`CRC-VAL-HE-7K`, $n = 7,180$)

| Criterion | Baseline | Geometric-only | Colour-only | Combined |
| :--- | :---: | :---: | :---: | :---: |
| **Internal Test Accuracy (%)** | **98.99** | 98.89 | 98.25 | 97.25 |
| **External Accuracy (%)** | 87.87 | 93.20 | **93.61** | 93.02 |
| **95% Wilson CI (%)** | 87.09–88.60 | 92.60–93.76 | **93.02–94.15** | 92.41–93.59 |
| **Generalization Gap (Internal − External, pp)** | −11.12 | −5.69 | −4.64 | −4.23 |
| **Total External Errors (/ 7,180)** | 871 (12.1%) | 488 (6.8%) | **459 (6.4%)** | 501 (7.0%) |
| **Macro-F1** | 0.872 | 0.909 | **0.918** | 0.905 |
| **Balanced Accuracy / Macro Recall (%)** | 88.1 | 91.3 | **91.9** | 90.8 |
| **Macro AUC (One-vs-Rest)** | 0.991 | 0.982 | **0.996** | 0.995 |
| **Log-Loss** *(lower is better)* | 0.564 | 0.535 | 0.294 | **0.284** |
| **ECE, 15 bins** *(lower is better)* | 0.084 | **0.038** | 0.042 | 0.039 |
| **Fixed / Broken vs. Baseline (Net Gain)** | — | 561 / 178 (+383) | **564 / 152 (+412)** | 620 / 250 (+370) |
| **Synthetic Colour Stress Accuracy (%)** | 45.5 | 46.3 | 91.9 | **92.2** |
| **Tumour Recall (%) (Missed / 1,233)** | 97.57 (30) | **98.13 (23)** | 96.92 (38) | 96.19 (47) |
| **Tumour Precision (%)** | 91.97 | 96.34 | **98.84** | 96.42 |

### 2. Paired Exact McNemar Tests (Bonferroni-Corrected $\alpha = 0.05 / 6 = 0.0083$)

| Comparison | Fixed | Broken | Net Change (pp) | 95% CI (pp) | McNemar $p$-value | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline → Geometric-only | 561 | 178 | +5.33 | [+4.60, +6.07] | `< 0.001` | **Yes** |
| Baseline → Colour-only | 564 | 152 | +5.74 | [+5.02, +6.46] | `< 0.001` | **Yes** |
| Baseline → Combined | 620 | 250 | +5.15 | [+4.36, +5.95] | `< 0.001` | **Yes** |
| Geometric-only → Colour-only | 311 | 282 | +0.40 | [−0.26, +1.07] | `0.250` | No |
| Geometric-only → Combined | 288 | 301 | −0.18 | [−0.84, +0.48] | `0.621` | No |
| Colour-only → Combined | 211 | 253 | −0.58 | [−1.17, +0.00] | `0.057` | No |

### 3. Per-Class Recall on `CRC-VAL-HE-7K` (%) & Error Redistribution

| Tissue Class | $n$ | Baseline | Geometric-only | Colour-only | Combined | RGB Colour Shift (0–255) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ADI` (Adipose) | 1,338 | **61.3** | 92.2 | 88.8 | 95.1 | 15 |
| `BACK` (Background) | 847 | 100.0 | 100.0 | 100.0 | 99.5 | **54** |
| `DEB` (Debris) | 339 | 97.0 | 92.3 | 89.7 | 92.6 | **44** |
| `LYM` (Lymphocytes) | 634 | 100.0 | 100.0 | 99.5 | 93.2 | 12 |
| `MUC` (Mucus) | 1,035 | 96.9 | 95.5 | 99.8 | 97.6 | 8 |
| `MUS` (Smooth Muscle) | 592 | 87.8 | **75.2** | **84.1** | **73.1** | 22 |
| `NORM` (Normal Mucosa) | 741 | 97.3 | 97.2 | 98.5 | 98.2 | 4 |
| `STR` (Stroma) | 421 | **55.1** | 71.7 | 70.1 | 71.7 | 11 |
| `TUM` (Tumour) | 1,233 | 97.6 | 98.1 | 96.9 | 96.2 | 10 |

### 4. Key Scientific Takeaways

* **Augmentation Halves External Errors—Mostly via Adipose & Stroma:** Total errors drop from 871 (Baseline) to 459 (Colour-only), driven primarily by massive adipose recovery (+368 to +452 net patches) and stroma recovery (+63 to +70 net patches).
* **Error Redistribution in Smooth Muscle:** Every augmented model degrades on `MUS` (−22 to −87 net patches), and `Combined` additionally loses 43 `LYM` patches and breaks the most images overall (250). Stroma/smooth-muscle confusion shifts direction rather than disappearing.
* **Colour Variation Is Not the Main Driver on This Normalized Benchmark:**
  * Under 28 synthetic colour perturbations (brightness `0.5–1.5`, contrast `0.5–1.5`, saturation `0.3–1.8`, hue `−0.15 to +0.15` on 2,000 external patches, `seed=1`), colour-augmented models retain **91.9%–92.2%** mean accuracy while Baseline and Geometric-only collapse to **45.5%–46.3%**.
  * Yet **Geometric-only**—despite being highly colour-sensitive—matches **Colour-only** on unperturbed `CRC-VAL-HE-7K` ($p = 0.250$).
  * Per-class RGB shift does not align with Baseline failures: `BACK` (shift = 54) and `DEB` (shift = 44) achieve 100.0% and 97.0% Baseline recall, whereas `ADI` (shift = 15) and `STR` (shift = 11) suffer the worst drops.
* **Recommendation:** **Colour-only (`ColorJitter`) alone** is the best-supported strategy (not proven statistically superior due to single training runs). Combining flips/rotations adds no measurable gain on external accuracy while increasing broken predictions.

---

## Interactive Prototype (Gradio + Grad-CAM)

A local CPU-compatible **Gradio + PyTorch** web demo loads all four trained ResNet50 checkpoints for qualitative inspection:
* Upload any $224 \times 224$ H&E patch to view **Top-5 class probabilities** and **Grad-CAM saliency maps**.
* **Interactive Stain-Shift Simulator:** Adjust brightness, contrast, saturation, and hue sliders to compare predictions and Grad-CAM focus across all four regimes side by side.
* **Safety Warnings:** Automatically flags low-confidence predictions (`< 60%`) or images resized from non-standard dimensions.

---

## Repository Structure

```text
├── notebooks/
│   ├── 01_dataset_eda_and_stain_shift.ipynb     # Class counts, sample grids, RGB/saturation histograms, OD stain-space shift (d1–d5)
│   ├── 02_train_and_eval_resnet50_regimes.ipynb # ResNet50 from-scratch pipeline, 4 augmentation regimes, internal & external reports + CMs
│   └── 03_method_comparison_figures.ipynb       # Wilson CI bar charts, per-class metric atlas, internal→external recall dumbbell plots
├── paper/
│   ├── sop_final_paper.pdf                      # Full IEEE-format research manuscript (34 verified references)
│   └── sop_final_review.pptx                    # Final M.Sc. project review presentation
├── app/
│   └── gradio_demo.py                           # Local Gradio + Grad-CAM side-by-side inference & stain-shift demo
└── README.md
```

---

## Reproducing the Experiments

### Requirements
```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib seaborn gradio
```

### Execution Pipeline
1. Download `NCT-CRC-HE-100K` and `CRC-VAL-HE-7K` (Kather et al., 2019; available on Zenodo/Kaggle) and point `BASE` / `base_path` to the root directory containing both folders.
2. Run `01_dataset_eda_and_stain_shift.ipynb` to verify the exact `seed=42` 70/15/15 split and generate dataset & optical-density colour shift diagnostics.
3. Run `02_train_and_eval_resnet50_regimes.ipynb` on a CUDA GPU (tested on NVIDIA Tesla T4) to train or evaluate the four `ResNet50(weights=None)` checkpoints on both the 15,000 internal test set and the 7,180 external `CRC-VAL-HE-7K` cohort.
4. Run `03_method_comparison_figures.ipynb` (CPU-only) to reproduce publication figures (`fig1_accuracy_internal_vs_external`, `fig2_metric_atlas_external`, `fig3_recall_shift_dumbbell`).

---

## Limitations & Future Work

* **Single Training Runs:** Only the data split was seeded (`seed=42`); weight initialization, batch order, and augmentations varied per run. Because epoch-to-epoch validation accuracy fluctuates by 1–2 pp, repeating experiments across multiple random seeds is necessary to confirm regime rankings.
* **Selection Bias on a Single External Set:** `CRC-VAL-HE-7K` was used both to compare models and to select the best-supported regime; evaluation on a third independent patient cohort is required.
* **Same-Biobank Normalization:** `CRC-VAL-HE-7K` tests generalization to **new patients** within the NCT biobank rather than cross-scanner or cross-laboratory stain shifts. Future work will benchmark the unnormalized `NCT-CRC-HE-100K-NONORM` dataset, enforce strict patient-level training/validation splits, and target `STR`/`MUS` morphological confusion.
* **Not Clinically Validated:** Models are proof-of-concept patch classifiers (missing 1.9%–3.8% of external tumour patches) and are not suitable for clinical deployment.

---

## Citation

If you use this code, analysis, or findings in your research, please cite:

```bibtex
@misc{ghosh2026colorectal_aug_ablation,
  title  = {Multi-Class Machine Learning Model for Automated Detection and Classification of Cancer Using Image Classification and Deep Learning},
  author = {Koushiki Ghosh, Kunal Maheshwari, Sagithya T},
  year   = {2026},
  school = {Department of Mathematics, School of Advanced Sciences, Vellore Institute of Technology, Chennai}
}
```

### Primary Dataset & Benchmark References
* J. N. Kather *et al.*, "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study," *PLOS Medicine*, vol. 16, no. 1, e1002730, 2019.
* M. Macenko *et al.*, "A method for normalizing histology slides for quantitative analysis," *Proc. IEEE ISBI*, pp. 1107–1110, 2009.
* D. Tellez *et al.*, "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology," *Medical Image Analysis*, vol. 58, 101544, 2019.
* N. Bussola *et al.*, "AI slipping on tiles: Data leakage in digital pathology," *ICPR Workshops (LNCS 12661)*, pp. 167–182, 2021.
* R. Geirhos *et al.*, "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, pp. 665–673, 2020.
