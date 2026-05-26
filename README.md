# Lightweight Privacy-Preserving Synthetic ICU Data Generation Using Tabular GANs on MIMIC-IV

**Authors:** Mohammadreza Momenzadeh¹², Atiyeh Oshaghi³
**Affiliation:** Isfahan University of Medical Sciences, Isfahan, Iran


---

## Overview

This repository contains all code for the manuscript:

> *Lightweight Privacy-Preserving Synthetic ICU Data Generation Using Tabular GANs on MIMIC-IV*

We benchmark three tabular generative models — **CTGAN**, **TVAE**, and **MedGAN** — on 15,742 ICU admissions from MIMIC-IV (v2.2), evaluating statistical fidelity, clinical utility (TSTR), empirical privacy, and computational efficiency.

**Key results (CTGAN):**
| Metric | Value |
|--------|-------|
| Wasserstein distance | 0.047 |
| TSTR Mortality AUROC | 0.847 (vs 0.862 real) |
| ML Efficacy Score | 0.983 (98.3% retention) |
| MIA Accuracy | 0.53 (near random) |
| k-anonymity (min) | 47 |
| Training time | 2.3 h (RTX 3080) |

---

## Repository Structure

```
synthetic-icu-tabular-gans/
├── data/
│   ├── sql_queries/
│   │   └── 01_cohort_extraction.sql   # Full MIMIC-IV SQL extraction
│   └── preprocessing.py               # Outlier handling, imputation, encoding, split
├── models/
│   ├── ctgan_model.py                 # CTGAN with PacGAN + GMM normalizer
│   ├── tvae_model.py                  # Tabular VAE with ELBO + KL annealing
│   └── medgan_model.py                # MedGAN with autoencoder pretraining
├── training/
│   └── train.py                       # HPO (Optuna) + final training (5 seeds)
├── evaluation/
│   ├── fidelity.py                    # KS, Wasserstein, Frobenius, PCA, NMI, JSD
│   └── utility_privacy.py             # TSTR, DCR, NNDR, MIA, k-anonymity, stat tests
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/momenzadeh-mr/synthetic-icu-tabular-gans.git
cd synthetic-icu-tabular-gans

# 2. Create and activate conda environment
conda create -n synthetic-icu python=3.10
conda activate synthetic-icu

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt key packages:**
```
torch==2.0.1
xgboost==1.7.6
scikit-learn==1.3.0
optuna==3.3.0
codecarbon==2.3.1
pandas==2.0.3
numpy==1.24.4
scipy==1.11.2
```

---

## Data Access

MIMIC-IV requires credentialed access via [PhysioNet](https://physionet.org/content/mimiciv):
1. Complete CITI "Data or Specimens Only Research" training
2. Sign the PhysioNet data use agreement
3. Download MIMIC-IV v2.2 and load into PostgreSQL 14

---

## Workflow

### Step 1 — Extract data from MIMIC-IV

```bash
# Run SQL in PostgreSQL with MIMIC-IV schema loaded
psql -d mimic -f data/sql_queries/01_cohort_extraction.sql -o data/raw/mimic_icu_raw.csv
```

### Step 2 — Preprocess

```bash
python data/preprocessing.py \
    --input  data/raw/mimic_icu_raw.csv \
    --output data/processed/ \
    --seed   42
```

Outputs: `train.csv`, `val.csv`, `test.csv`, `scaler.pkl`, `encoder_info.json`

### Step 3 — Hyperparameter optimization (100 trials per model, ~18h for CTGAN)

```bash
python training/train.py \
    --model    ctgan \
    --mode     hpo \
    --data_dir data/processed/ \
    --output   results/ \
    --n_trials 100
```

### Step 4 — Final training (5 seeds)

```bash
python training/train.py \
    --model    ctgan \
    --mode     final \
    --data_dir data/processed/ \
    --output   results/ \
    --config   results/ctgan_best_config.json
```

Repeat for `--model tvae` and `--model medgan`.

### Step 5 — Evaluate fidelity, utility, and privacy

```python
import pandas as pd
from evaluation.fidelity       import full_fidelity_report
from evaluation.utility_privacy import run_utility_evaluation, full_privacy_report

train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")
synth_df = pd.read_csv("results/ctgan_seed42/synthetic_data.csv")

feature_cols     = [c for c in train_df.columns if c not in ["mortality"]]
continuous_cols  = [c for c in feature_cols if train_df[c].dtype == float]
categorical_cols = [c for c in feature_cols if train_df[c].dtype == object]

# Fidelity
fidelity = full_fidelity_report(train_df, synth_df, continuous_cols, categorical_cols)

# Utility (TSTR)
utility  = run_utility_evaluation(train_df, synth_df, test_df, feature_cols, "mortality")

# Privacy
privacy  = full_privacy_report(
    train_df.values, test_df.values, synth_df[feature_cols].values,
    synth_df=synth_df,
    quasi_identifiers=["age", "gender", "ethnicity", "first_careunit"],
)
```

---

## Reproducibility

All experiments were repeated **5 times** with random seeds: `[42, 123, 456, 789, 1024]`.

Seeds are fixed for:
```python
import random, numpy as np, torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Results are reported as mean ± SD with 95% bootstrap confidence intervals (BCa, 1,000 iterations).

---

## Computational Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3080 (10 GB VRAM) |
| CPU | Intel Core i9-10900K |
| RAM | 64 GB DDR4 |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.12 |
| PyTorch | 2.0.1 + CUDA 11.8 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{momenzadeh2024synthetic,
  title   = {Lightweight Privacy-Preserving Synthetic ICU Data Generation Using Tabular GANs on MIMIC-IV},
  author  = {Momenzadeh, Mohammadreza and Oshaghi, Atiyeh},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License.
Data from MIMIC-IV is subject to PhysioNet's data use agreement.
