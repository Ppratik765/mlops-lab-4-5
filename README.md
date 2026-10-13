# MLOps Lab 4 & 5: Advanced MLflow Tracking and Data Versioning with DVC

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6.svg)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end reproducible Machine Learning Operations (MLOps) project demonstrating systematic experiment tracking, model signature enforcement, Model Registry lifecycle management using **MLflow**, and robust dataset version control with **Data Version Control (DVC)**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository and Set Up Environment](#1-clone-repository-and-set-up-environment)
  - [2. Install Required Dependencies](#2-install-required-dependencies)
  - [3. Ensure Data Availability with DVC](#3-ensure-data-availability-with-dvc)
  - [4. Execute Model Training Runs](#4-execute-model-training-runs)
  - [5. Launch MLflow Dashboard](#5-launch-mlflow-dashboard)
  - [6. Verify Dataset Recovery Workflow](#6-verify-dataset-recovery-workflow)
- [Part A: Advanced MLflow (Lab 4)](#part-a-advanced-mlflow-lab-4)
  - [MLflow Architecture and Components](#mlflow-architecture-and-components)
  - [Model Signatures and Input Examples](#model-signatures-and-input-examples)
  - [Experiment Comparison and Evaluation](#experiment-comparison-and-evaluation)
  - [Model Registry and Promotion Decision](#model-registry-and-promotion-decision)
- [Part B: Data Versioning with DVC (Lab 5)](#part-b-data-versioning-with-dvc-lab-5)
  - [Motivation and Design](#motivation-and-design)
  - [Dataset Version History](#dataset-version-history)
  - [Remote Storage Configuration](#remote-storage-configuration)
  - [Version Restoration and Verification](#version-restoration-and-verification)
- [Part C: Git, DVC, and MLflow Integration](#part-c-git-dvc-and-mlflow-integration)
  - [Role Division](#role-division)
  - [Full Lineage and Reproducibility Audit](#full-lineage-and-reproducibility-audit)

---

## Project Overview

In production machine learning systems, reproducibility requires answering a fundamental governance question:
> **Can an engineer identify the deployed model version and systematically reconstruct the exact codebase, hyperparameter configuration, and dataset snapshot used to train it?**

This repository addresses this challenge by integrating three complementary tools:
- **Git**: Tracks application code, pipeline configurations, and lightweight metadata files.
- **DVC**: Manages dataset versions and synchronization with external/local storage without storing heavy binaries in Git.
- **MLflow**: Tracks experiment parameters, evaluation metrics, model signatures, generated artifacts, and centralized model lifecycle states.

---

## Repository Structure

```text
mlops-lab-4-5/
├── .dvc/                       # DVC configuration and storage definitions
│   ├── .gitignore
│   └── config                  # Local remote endpoint definition
├── configs/
│   └── config.yaml             # Pipeline and model configuration parameters
├── data/
│   ├── .gitignore              # Prevents dataset files from being tracked by Git
│   ├── dataset.csv             # Raw dataset snapshot (managed by DVC)
│   └── dataset.csv.dvc         # DVC metadata tracking pointer file (committed to Git)
├── models/
│   └── .gitkeep                # Exported model checkpoints directory
├── reports/
│   └── figures/
│       └── .gitkeep            # Diagnostic plots and evaluation figures
├── src/
│   └── train.py                # Model training script with MLflow autologging and registry
├── .dvcignore                  # Exclusions for DVC processing
├── .gitignore                  # Git exclusions (venv, sqlite db, mlflow artifacts)
├── requirements.txt            # Frozen environment dependencies
└── README.md                   # System documentation
```

---

## How to Run

Follow the steps below to set up the environment, run experiments, inspect results, and verify reproducibility.

### Prerequisites
- Python 3.10 or higher
- Git 2.30 or higher

### 1. Clone Repository and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Ppratik765/mlops-lab-4-5.git
cd mlops-lab-4-5

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Required Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ensure Data Availability with DVC

The dataset is versioned using DVC. To ensure the working file matches the current pointer:

```bash
# Verify the dataset is present or pull from remote cache
dvc checkout
```

To verify the current dataset row count:
```bash
# Windows PowerShell:
(Get-Content data/dataset.csv).Length

# Linux / macOS:
wc -l data/dataset.csv
```
*(Expected output for v2: 151 lines, representing 1 header row + 150 data rows).*

### 4. Execute Model Training Runs

The training script located at `src/train.py` accepts two command-line arguments:
- `n_estimators`: Number of decision trees in the Random Forest ensemble (default: `50`)
- `max_depth`: Maximum depth of each decision tree (default: `5`)

Execute the three candidate configurations to generate and log experiments:

```bash
# Run 1: Baseline Candidate (Version 1)
python src/train.py 50 5

# Run 2: Intermediate Depth Model (Version 2)
python src/train.py 100 10

# Run 3: Deep Forest Model (Version 3)
python src/train.py 200 15
```

Each run automatically:
1. Loads `data/dataset.csv`.
2. Splits data into train and test partitions (`80/20`).
3. Fits a `RandomForestRegressor`.
4. Evaluates test metrics (`RMSE`, `R²`).
5. Logs parameters, metrics, model artifacts, and signatures to MLflow.
6. Registers the resulting model version under `HousePriceModel` in the MLflow Model Registry.

### 5. Launch MLflow Dashboard

Start the MLflow tracking UI connected to the local SQLite database backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

Inside the UI:
- **Experiments**: Select `House_Price_Advanced` to inspect parameter vs metric comparisons, artifact outputs, and training plots.
- **Models**: Select `HousePriceModel` to inspect registered model versions (`v1`, `v2`, `v3`), their input schemas, signatures, and associated run provenance.

### 6. Verify Dataset Recovery Workflow

To verify DVC's ability to recover an earlier dataset version:

```bash
# Step 1: Checkout the DVC metadata pointer file from dataset v1 commit
git checkout 47aaa78 -- data/dataset.csv.dvc

# Step 2: Restore the corresponding data file
dvc checkout

# Step 3: Verify restored row count (101 lines: 1 header + 100 rows)
# Windows PowerShell:
(Get-Content data/dataset.csv).Length
# Linux / macOS:
wc -l data/dataset.csv

# Step 4: Return to the latest dataset version (v2)
git checkout HEAD -- data/dataset.csv.dvc
dvc checkout
```

---

## Part A: Advanced MLflow (Lab 4)

### MLflow Architecture and Components

- **Autologging (`mlflow.sklearn.autolog()`)**: Automatically tracks estimator parameters (`n_estimators`, `max_depth`, `criterion`, `min_samples_split`), fitted model coefficients, and standard training metrics without requiring manual metric statements for each parameter.
- **Custom Metrics**: Explicit evaluation on unseen holdout test data is logged via `mlflow.log_metric()` for `test_rmse` and `test_r2`.
- **Model Signatures**: Captures expected input schema and output specification to prevent silent prediction errors and format mismatches.
- **Model Registry**: A centralized registry facilitating model discovery, lineage tracking, and staging workflows.

### Model Signatures and Input Examples

The model signature is inferred directly from training features and initial predictions:
```python
signature = infer_signature(X_train, preds)
input_example = X_train.head(3)
```

- **Input Schema**:
  - `rooms`: Integer (64-bit)
  - `sqft`: Integer (64-bit)
- **Output Schema**:
  - `price`: Floating point (64-bit)
- **Input Example**: First 3 records of `X_train` saved alongside model metadata to validate schema compliance on future inference requests.

### Experiment Comparison and Evaluation

Three distinct configurations were evaluated under the `House_Price_Advanced` experiment:

| Version | Run ID | Parameters | Test RMSE | Test R² | Training RMSE | Registry Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | `a49552a4bbbc4f19926eb57532e290f1` | `n_estimators=50`, `max_depth=5` | **109,831.18** | **-0.0696** | 88,822.51 | **Production Candidate** |
| **v2** | `cb6029068bb34b8b93d4bd08f108c8fe` | `n_estimators=100`, `max_depth=10` | 111,351.32 | -0.0994 | 57,063.92 | Staging / Superseded |
| **v3** | `32e69676b4a84e0aa262f422b72241c2` | `n_estimators=200`, `max_depth=15` | 111,500.05 | -0.1024 | 51,825.63 | Archived / Overfitted |

### Model Registry and Promotion Decision

- **Analysis**: As model capacity was expanded from `max_depth=5` to `max_depth=15`, training error dropped considerably from $88,822$ to $51,825$. However, generalization error on the test partition degraded (Test RMSE increased from $109,831$ to $111,500$). This indicates substantial overfitting on the deeper ensembles.
- **Promotion Decision**:
  - **Version 1** is selected and designated as the **Production** model based on superior generalization performance (lowest Test RMSE).
  - **Version 2** is marked as **Staging** / superseded candidate.
  - **Version 3** is marked as **Archived** due to overfitting.

---

## Part B: Data Versioning with DVC (Lab 5)

### Motivation and Design

Standard version control systems like Git are not architected to store large, mutable datasets or binary files. Committing multi-megabyte or gigabyte files to Git causes repository bloat and limits scalability. DVC solves this by:
1. Placing the raw data file (`data/dataset.csv`) outside Git tracking via `data/.gitignore`.
2. Tracking a lightweight metadata file (`data/dataset.csv.dvc`) containing unique content hashes in Git.
3. Managing actual data storage through local or external storage backends.

### Dataset Version History

| Dataset Version | Git Commit | Data Rows | Total Columns | Content Hash (MD5) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset v1** | `47aaa78` | 100 | 3 | `ad32526ac18b0c1d44c08bbe52e712b0` | Initial baseline housing dataset |
| **Dataset v2** | `df646dc` | 150 | 3 | `aea8245c208c5238d07667a3f807086b` | Expanded dataset with 50 supplementary rows |

### Remote Storage Configuration

The local DVC storage remote is configured under `.dvc/config`:
```ini
[core]
    remote = localremote
['remote "localremote"']
    url = ../../dvc-storage
```

To sync data between workspace and storage:
```bash
# Push current tracked datasets to remote storage
dvc push

# Pull datasets from remote storage
dvc pull
```

### Version Restoration and Verification

Restoration of historical data relies on Git's commit graph and DVC's content-addressable storage:
1. Reverting the pointer (`dataset.csv.dvc`) tells DVC which MD5 checksum is required.
2. Running `dvc checkout` restores the exact file payload matching that checksum into `data/dataset.csv`.
3. Validation is confirmed through row count verification and checksum matching.

---

## Part C: Git, DVC, and MLflow Integration

### Role Division

| Tool | Managed Artifacts | Storage Medium | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Git** | Source code, pipeline scripts, YAML configs, `.dvc` files | GitHub Repository | Code lineage, branching, peer review, versioning |
| **DVC** | Tabular datasets, binary files, data dependencies | DVC Local Remote / Shared Storage | Data provenance, storage decoupling |
| **MLflow** | Hyperparameters, metrics, signatures, model weights, logs | SQLite DB (`mlflow.db`) / `mlruns/` | Experiment tracking, model comparison, registry |

### Full Lineage and Reproducibility Audit

To audit or reproduce any deployed model:
1. **Model Identification**: In the MLflow Model Registry, locate the production model (`HousePriceModel`, Version 1) and note its Source Run ID (`a49552a4bbbc4f19926eb57532e290f1`).
2. **Experiment Parameters**: View recorded hyperparameters (`n_estimators=50`, `max_depth=5`) and model signatures.
3. **Data Snapshot**: Retrieve the corresponding dataset version hash from DVC history and restore it (`dvc checkout`).
4. **Codebase State**: Checkout the Git commit associated with that experiment run (`git checkout <commit-sha>`).
5. **Exact Replication**: Rerun `python src/train.py 50 5` to reproduce the model weights, metrics, and behavior deterministically.
