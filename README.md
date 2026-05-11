# 🍎 Fruits-360 Image Classification

> Benchmarks three CNN architectures on 131-class fruit recognition. Built as a machine learning course final project — runs end-to-end on Kaggle with a single **Run All**.

---



> Fill in the `—` values from `working_outputs/reports/model_comparison_results.csv` after running the notebook.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Requirements](#-requirements)

---

## 🔍 Overview

This project benchmarks **three CNN architectures** on a multi-class fruit image classification task:

| # | Model | Description |
|---|---|---|
| 1 | **Baseline CNN** | Lightweight sequential network — establishes a performance floor |
| 2 | **Custom CNN + Augmentation** | Deeper network with BatchNorm, Dropout, and on-the-fly augmentation |
| 3 | **MobileNetV2 (Transfer Learning)** | ImageNet-pretrained backbone with fine-tuning of the last 20 layers |

Each model is trained, evaluated, and compared using **accuracy, precision, recall, and weighted F1-score** on a held-out test set. All outputs (models, plots, reports) are saved automatically to `/kaggle/working`.

---

## 🍇 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — moltean/fruits](https://www.kaggle.com/datasets/moltean/fruits) |
| Image size | `100 × 100 px` (RGB) |
| Total classes | 131 fruits and vegetables |
| Selected classes | Up to 20 (configurable via `MAX_CLASSES`) |
| Train / val split | 80% / 20% from `Training/` folder |
| Test set | Separate `Test/` folder |

The notebook **auto-detects the dataset root** across multiple possible Kaggle directory layouts — no manual path configuration needed.

---

## 🧠 Models

### 1. Baseline CNN

A simple sequential model to establish a performance floor.

```
Input → Rescaling → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
      → GlobalAvgPool → Dense(128) → Dropout(0.3) → Softmax
```

### 2. Custom CNN + Augmentation

Deeper architecture with double-conv blocks, BatchNorm, and built-in data augmentation
(horizontal flip, rotation, zoom, contrast).

```
Input → Augmentation → Rescaling
      → [Conv2D → BN → Conv2D → BN → MaxPool → Dropout] × 3   (filters: 32 → 64 → 128)
      → GlobalAvgPool → Dense(256) → Dropout(0.4) → Softmax
```

### 3. MobileNetV2 (Transfer Learning)

Uses ImageNet pretrained weights. The base is initially **frozen**, then the last 20 layers are **unfrozen** for fine-tuning at a low learning rate (`1e-5`).

```
Input → Augmentation → MobileNetV2 Preprocessing → MobileNetV2 (frozen → fine-tuned)
      → GlobalAvgPool → Dropout(0.3) → Softmax
```

**All models share:**

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| EarlyStopping | patience = 3 |
| ReduceLROnPlateau | patience = 2 |


## 🗂️ Project Structure

```
working_outputs/
├── models/
│   ├── baseline_cnn.keras
│   ├── custom_cnn.keras
│   └── mobilenetv2.keras
├── plots/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── accuracy_baseline_cnn.png
│   ├── accuracy_custom_cnn.png
│   ├── accuracy_mobilenetv2.png
│   ├── loss_baseline_cnn.png
│   ├── loss_custom_cnn.png
│   ├── loss_mobilenetv2.png
│   ├── confusion_matrix_baseline_cnn.png
│   ├── confusion_matrix_custom_cnn.png
│   ├── confusion_matrix_mobilenetv2.png
│   ├── model_comparison_accuracy.png
│   └── model_comparison_f1.png
└── reports/
    ├── eda_class_distribution.csv
    ├── history_baseline_cnn.csv
    ├── history_custom_cnn.csv
    ├── history_mobilenetv2.csv
    ├── classification_report_baseline_cnn.txt
    ├── classification_report_custom_cnn.txt
    ├── classification_report_mobilenetv2.txt
    ├── confusion_matrix_baseline_cnn.csv
    ├── confusion_matrix_custom_cnn.csv
    ├── confusion_matrix_mobilenetv2.csv
    ├── model_comparison_results.csv
    └── experiment_summary.json
```

---

## 🚀 Getting Started

### ▶️ Run on Kaggle (recommended)

1. Open a new Kaggle notebook and upload `fruits-360-classification-computer-vesion-project.ipynb`
2. **Add Data** → search for `moltean/fruits` and attach it
3. **Settings → Accelerator → GPU** *(optional but speeds up training)*
4. Click **Run All**

All outputs are saved automatically to `/kaggle/working`.

**To add your result images to this README after training:**

1. Go to Kaggle's **Output** panel → download the `plots/` folder
2. In your GitHub repo, create the folder `working_outputs/plots/`
3. Upload all `.png` files — the images in this README will appear automatically

---

### 💻 Run Locally

```bash
# Install dependencies
pip install tensorflow scikit-learn pandas matplotlib numpy

# Download the dataset from Kaggle and arrange as:
#   data/
#     Training/   <class folders>
#     Test/       <class folders>

# Launch Jupyter
jupyter notebook fruits-360-classification-computer-vesion-project.ipynb
```

> **Note:** The notebook auto-detects the dataset root. As long as a `Training/` and `Test/` directory pair is discoverable, no manual path setup is needed.

> **Low-resource tip:** If you hit RAM or GPU errors on Kaggle, reduce `MAX_CLASSES` to `10` and `BATCH_SIZE` to `16`.

---

## ⚙️ Configuration

All key hyperparameters are defined at the top of the notebook (Section 1):

| Parameter | Default | Description |
|---|:---:|---|
| `IMG_SIZE` | `(100, 100)` | Input image dimensions |
| `BATCH_SIZE` | `32` | Training batch size |
| `MAX_CLASSES` | `20` | Number of fruit classes to use (10–20) |
| `EPOCHS_BASELINE` | `6` | Max epochs for Baseline CNN |
| `EPOCHS_CUSTOM_CNN` | `8` | Max epochs for Custom CNN |
| `EPOCHS_MOBILENET` | `8` | Max epochs for MobileNetV2 |
| `VALIDATION_SPLIT` | `0.20` | Fraction of training data used for validation |
| `SEED` | `42` | Global random seed for reproducibility |

---

## 📦 Requirements

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | ≥ 2.x | Model building & training |
| `scikit-learn` | latest | Evaluation metrics & confusion matrix |
| `pandas` | latest | Data tables & CSV export |
| `matplotlib` | latest | Plots & visualizations |
| `numpy` | latest | Numerical operations |

```bash
pip install tensorflow scikit-learn pandas matplotlib numpy
```

---

## 📄 License

This project was created for **educational purposes** as part of a machine learning course.
The Fruits-360 dataset is provided by [Mihai Oltean](https://www.kaggle.com/datasets/moltean/fruits) under its own license — please refer to the dataset page for usage terms.
