# MLOps Lab Assignment 4 & 5: Advanced MLflow & Data Versioning with DVC

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6.svg)](https://dvc.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end reproducible Machine Learning Operations project demonstrating experiment tracking, model lifecycle management with **MLflow**, and robust dataset version control with **Data Version Control (DVC)**.

---

## 📌 Project Overview

This project answers the critical MLOps question:
> **"Can another engineer identify the model that was deployed and recover the exact code and dataset version used to create it?"**

By establishing a unified integration between **Git**, **DVC**, and **MLflow**, this repository provides full lineage, auditability, and reproducibility across:
- **Code & Metadata** (tracked via Git)
- **Large Dataset Versions** (tracked via DVC with remote storage)
- **Experiments, Signatures, Artifacts & Model Registry** (tracked via MLflow)

---

## 📂 Repository Structure

```text
mlops-lab-4-5/
├── .dvc/                       # DVC internal configuration and cache settings
│   ├── .gitignore
│   └── config                  # DVC remote storage configuration (localremote)
├── configs/
│   └── config.yaml             # Pipeline and model hyperparameter configurations
├── data/
│   ├── .gitignore              # Ignores large raw dataset from Git
│   ├── dataset.csv             # Raw dataset (tracked by DVC, ignored by Git)
│   └── dataset.csv.dvc         # DVC metadata pointer file committed to Git
├── models/
│   └── .gitkeep                # Exported model checkpoints directory
├── reports/
│   └── figures/
│       └── .gitkeep            # Experiment evaluation plots and diagnostic figures
├── src/
│   └── train.py                # Model training script with MLflow autologging & registry
├── .dvcignore                  # Files excluded from DVC tracking
├── .gitignore                  # Git exclusions (venv, sqlite, mlflow artifacts, etc.)
├── requirements.txt            # Python environment dependencies
└── README.md                   # Comprehensive project documentation
```

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ppratik765/mlops-lab-4-5.git
cd mlops-lab-4-5
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Part A: Advanced MLflow (Lab 4)

### 1. MLflow Architecture & Core Concepts
- **Autologging (`mlflow.sklearn.autolog()`)**: Automatically captures estimator parameters (e.g., `criterion`, `n_estimators`, `max_depth`, `min_samples_split`), training metrics (`training_rmse`, `training_r2`, `training_mae`), and fitted estimators without boilerplate code.
- **Model Signature**: Defines the schema contract for inputs and outputs (e.g. `rooms: int64`, `sqft: int64` $\rightarrow$ `price: float64`). It guarantees type safety and catches malformed inference payloads early.
- **Input Example**: Captures sample non-sensitive rows (`X_train.head(3)`) to illustrate valid requests and automatically validate request formatting.
- **Model Registry (`HousePriceModel`)**: A centralized model store managing version revisions, aliases, tags, and lifecycle transitions (Candidate/Staging $\rightarrow$ Production $\rightarrow$ Archived).

### 2. Experiment Runs Comparison
Three candidate models were trained with varying hyperparameter configurations under the experiment `House_Price_Advanced`:

| Version | Run ID | Algorithm | Hyperparameters | Test RMSE | Test R² | Training RMSE | Registry Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | `a49552a4bbbc4f19926eb57532e290f1` | Random Forest | `n_estimators=50`, `max_depth=5` | **109,831.18** | **-0.0696** | 88,822.51 | **Production Candidate** |
| **v2** | `cb6029068bb34b8b93d4bd08f108c8fe` | Random Forest | `n_estimators=100`, `max_depth=10` | 111,351.32 | -0.0994 | 57,063.92 | Staging / Superseded |
| **v3** | `32e69676b4a84e0aa262f422b72241c2` | Random Forest | `n_estimators=200`, `max_depth=15` | 111,500.05 | -0.1024 | 51,825.63 | Archived / Overfitted |

#### Evaluation & Promotion Decision
- **Observations**: As tree depth increased from 5 to 15, the model severely overfitted the training split (Training RMSE dropped from $88,822$ to $51,825$), leading to worse generalization on the unseen test split.
- **Promotion Decision**: **Model Version 1** achieved the lowest test RMSE ($109,831.18$) and the best generalization profile among the candidates, making it the selected candidate for the **Production** stage. Versions 2 and 3 are retained for auditability and marked as superseded/archived.

### 3. Launching MLflow Tracking UI
To inspect the experiments, artifact artifacts, model signatures, and registered models:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```
Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 🗃️ Part B: Data Versioning with DVC (Lab 5)

### 1. DVC Role & Mechanics
Git is optimized for source code and textual diffs, but inefficient for large or evolving binary/tabular datasets. DVC decouples data storage from metadata tracking:
- Large dataset files are managed outside Git and cached locally/remotely.
- Lightweight human-readable pointer files (`data/dataset.csv.dvc`) store unique content hashes (MD5) and file sizes, and are version-controlled in Git.

### 2. Dataset Version History

| Dataset Version | Git Commit | Rows | Columns | MD5 Hash | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | `47aaa78` | 100 | 3 (`rooms`, `sqft`, `price`) | `ad32526ac18b0c1d44c08bbe52e712b0` | Baseline synthetic housing dataset (100 rows) |
| **v2** | `df646dc` | 150 | 3 (`rooms`, `sqft`, `price`) | `aea8245c208c5238d07667a3f807086b` | Augmented dataset with 50 additional records (150 rows) |

### 3. Remote Storage Configuration
A local DVC remote named `localremote` is configured pointing to `../../dvc-storage`:
```bash
# Verify remote configuration
dvc remote list

# Push cached data to remote
dvc push

# Pull data from remote
dvc pull
```

### 4. Recovering an Earlier Dataset Version
To roll back and reproduce experiments on Dataset v1:
```bash
# 1. Checkout the dataset pointer file from the v1 commit
git checkout 47aaa78 -- data/dataset.csv.dvc

# 2. Restore the actual dataset corresponding to that commit's hash
dvc checkout

# 3. Verify row count (100 rows + 1 header = 101 lines)
# Windows PowerShell:
(Get-Content data/dataset.csv).Length

# Linux/macOS:
wc -l data/dataset.csv

# 4. Return to latest v2 dataset
git checkout HEAD -- data/dataset.csv.dvc
dvc checkout
```

---

## 🔄 Part C: Git, DVC & MLflow Integration

| Tool | What It Tracks | Storage Location | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Git** | Source code, pipeline scripts, config, `.dvc` pointer files | GitHub repository | Code versioning, branching, workflow history |
| **DVC** | Raw datasets, features, large assets | DVC Cache & Remote storage (`dvc-storage`) | Large data versioning without repository bloat |
| **MLflow** | Hyperparameters, metrics, signatures, artifacts, model weights | SQLite DB (`mlflow.db`) & artifact store (`mlruns/`) | Experiment lineage, comparison & Model Registry |

### End-to-End Auditability Workflow
When investigating a prediction from the registered `HousePriceModel`:
1. **Model Registry**: Query MLflow for the active Production model version (e.g. `v1`).
2. **Experiment Run**: Inspect the associated run ID (`a49552a4bbbc4f19926eb57532e290f1`) to inspect exact hyperparameter values, test metrics, and logged model signature.
3. **Data Provenance**: Look up the dataset hash recorded during training and checkout the exact DVC metadata matching that hash (`dvc checkout`).
4. **Code Provenance**: Checkout the exact Git commit SHA that executed the run (`git checkout <commit-sha>`).
5. **Exact Replication**: Rerun `python src/train.py` to achieve identical results.

---

## 🚀 Running the Training Pipeline

To run training with custom hyperparameters:
```bash
# Usage: python src/train.py <n_estimators> <max_depth>
python src/train.py 50 5
python src/train.py 100 10
python src/train.py 200 15
```
Each execution logs metrics, model signature, input examples, and registers new iterations to `HousePriceModel` under the `House_Price_Advanced` experiment.
