# Explainable Anomaly Scoring for Ethereum Multisignature Transactions

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the full replication package for the paper:

> **Explainable Anomaly Scoring for Ethereum Multisignature Transactions Using Temporal Validation and LightGBM**  
> Submitted to *Computers, Materials & Continua (CMC)*, 2025.

## Overview

Multisignature (multisig) wallets are fundamental to institutional-grade asset security on the Ethereum blockchain, yet Security Operations Centers (SOCs) currently rely on manual threshold rules to flag anomalous executions. This work addresses three methodological deficiencies in existing approaches: (i) reliance on random train-test splits that leak future information, (ii) inclusion of post-hoc execution features unavailable at prediction time, and (iii) absence of cross-architectural benchmarking to justify algorithmic choices.

## Key Results

| Metric | LightGBM (Ours) | 95% CI |
|--------|-----------------|--------|
| **ROC-AUC** | **0.9971** | 0.9961–0.9980 |
| **PR-AUC** | **0.9865** | 0.9833–0.9895 |
| **F1-Score** | **0.9739** | 0.9691–0.9784 |
| **Accuracy** | **0.9949** | 0.9941–0.9957 |
| **Precision** | **0.9927** | 0.9891–0.9960 |
| **Recall** | **0.9513** | 0.9428–0.9596 |
| **MCC** | **0.9690** | 0.9638–0.9740 |

**Cross-Architectural Benchmarking (8 models, 4 families):**

| Model | ROC-AUC | PR-AUC | F1 | Inference (ms/tx) |
|-------|---------|--------|----|-------------------|
| XGBoost | 0.9976 | 0.9876 | 0.9707 | 0.13 |
| **LightGBM (Ours)** | **0.9971** | **0.9865** | **0.9739** | **0.21** |
| Random Forest | 0.9959 | 0.9853 | 0.9739 | 0.21 |
| GraphSAGE | 0.9824 | 0.9663 | 0.9045 | 2.40 |
| MLP | 0.9657 | 0.9174 | 0.8687 | 0.008 |
| GAT | 0.9415 | 0.8665 | 0.6717 | 9.28 |
| Logistic Regression | 0.9290 | 0.8530 | 0.6652 | 0.003 |
| Isolation Forest | 0.9020 | 0.6256 | 0.5876 | 0.06 |

All differences statistically significant under Bonferroni correction (α_adj = 0.007, m = 7).

**Speedup:** LightGBM achieves **11.5× faster inference** than GraphSAGE (0.21 ms vs. 2.40 ms per transaction).

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Hyperparameters and experiment settings
├── src/
│   ├── 01_data_preprocessing.py       # BigQuery extraction and labeling
│   ├── 02_temporal_split.py           # Block-height temporal validation
│   ├── 03_feature_engineering.py      # Leakage removal and z-score normalization
│   ├── 04_train_lightgbm.py           # LightGBM training with SHAP support
│   ├── 05_train_baselines.py         # All 7 baseline models
│   ├── 06_evaluate.py                # Metrics, bootstrap CIs, significance tests
│   ├── 07_shap_analysis.py           # Explainability and feature importance
│   ├── 08_stress_test.py            # Gaussian perturbation robustness
│   ├── 09_forensic_validation.py     # External exploit corpus validation
│   └── utils.py                      # Shared utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset and labeling analysis
│   ├── 02_model_training.ipynb        # Training and cross-validation
│   └── 03_evaluation.ipynb            # Benchmarking and SHAP analysis
├── data/                              # Data directory (not committed)
│   ├── raw/                          # BigQuery extraction output
│   ├── processed/                    # Cleaned and labeled datasets
│   └── forensic/                     # 12 verified exploit transactions
├── models/                           # Saved model artifacts
└── results/                          # Figures, tables, and outputs
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Data from BigQuery

```bash
python src/01_data_preprocessing.py
```

This downloads 100,000 Gnosis Safe multisig transactions from `bigquery-public-data.crypto_ethereum.transactions` (Jan–Dec 2023) and applies heuristic labeling.

### 3. Run Full Pipeline

```bash
# Temporal split → Feature engineering → Training → Evaluation
python src/02_temporal_split.py
python src/03_feature_engineering.py
python src/04_train_lightgbm.py
python src/05_train_baselines.py
python src/06_evaluate.py
python src/07_shap_analysis.py
python src/08_stress_test.py
python src/09_forensic_validation.py
```

Or run the full pipeline end-to-end:

```bash
python -m src.run_all
```

### 4. Reproduce Figures and Tables

```bash
jupyter notebook notebooks/03_evaluation.ipynb
```

## Methodology Summary

### Dataset
- **Source:** Google BigQuery `bigquery-public-data.crypto_ethereum.transactions`
- **Target:** Gnosis Safe multisig contract interactions
- **Period:** January – December 2023
- **Size:** 100,000 transactions
  - Train: 60,000 (10.69% anomaly) — blocks [23586544, 24469834] — Jan–Aug 2023
  - Validation: 15,000 (7.67% anomaly) — blocks [24469842, 24663889] — Aug–Oct 2023
  - Test: 25,000 (9.12% anomaly) — blocks [24663902, 24881458] — Oct–Dec 2023

### Labeling (Heuristic + Isolation Forest Hybrid)
Anomaly labels are operational heuristics calibrated to known execution-complexity signatures:
1. **Gas limit spike:** > 3 MAD above training-set median
2. **Internal call complexity:** > 98th percentile of training distribution
3. **Calldata anomaly:** < 4 bytes or > 10,000 bytes
4. **Replay attempt:** duplicate nonce within same wallet epoch
5. **Transaction failure:** revert or out-of-gas exhaustion

Isolation Forest provides auxiliary signal enhancement for latent patterns not covered by deterministic rules. The forensic corpus of 12 independently verified 2023 exploits ($286M total losses) validates that execution-complexity thresholds align with real attack signatures.

### Leakage Prevention (5 Categories, 14 Features Removed)
| Category | Count | Rationale |
|----------|-------|-----------|
| Identifiers | 4 | Prevent memorization of specific transactions |
| Post-Hoc | 5 | Unavailable at prediction time |
| Correlated | 2 | Encode target information |
| Temporal | 2 | Leak future context |
| Hash-Derived | 1 | Pattern memorization |

### Features Retained (16)
- **Raw On-Chain Metrics (6):** `gas_price`, `gas_limit`, `calldata_length`, `eth_value`, `internal_calls`, `log_count`
- **Temporal Indicators (5):** `hour_of_day`, `day_of_week`, `is_weekend`, `is_night`
- **Normalized Deviation (5):** `gas_price_zscore_clean`, `gas_limit_zscore_clean`, `calldata_length_zscore_clean`, `internal_calls_zscore_clean`, `log_count_zscore_clean` (median/MAD computed exclusively from training data)

### Temporal Validation
Strict block-height-based partitioning enforces chronological separation:
- Training data strictly precedes validation and test data
- Temporal validation gap vs. random split: **only 0.17% AUC inflation**
- Distribution drift confirms temporal isolation: ETH value 92.8%, gas_price 58.2%

### Hybrid Anomaly Scoring
```
S(x) = λ · f(x) + (1 − λ) · σ(w₁·z_gas + w₂·z_calls + w₃·z_logs)
```
where `f(x)` is LightGBM probability, `z` are robust z-scores, and `σ` is sigmoid.

### Statistical Rigor
- **Bootstrap inference:** B = 1,000 resamples, percentile method for 95% CIs
- **Significance testing:** Bonferroni-corrected paired t-tests (α_adj = 0.007, m = 7)
- **Non-parametric check:** Wilcoxon signed-rank tests

## Ablation Study

| Configuration | ROC-AUC | F1-Score (τ=0.5) | Degradation |
|--------------|---------|-------------------|-------------|
| Full Model (16 features) | 0.9971 | 0.9739 | — |
| Without gas_limit + zscore | 0.9678 | 0.7298 | −2.9% |
| Without all execution features | 0.7723 | 0.2970 | −22.8% |

Removing the dominant univariate signal (`gas_limit`) still yields **96.8% ROC-AUC retention**, confirming multivariate pattern learning beyond simple thresholding.

## Stress Test Results

| Noise (σ) | ROC-AUC | F1 Score | Brier Score | AUC Retention |
|-----------|---------|----------|-------------|---------------|
| 0.00 | 0.9971 | 0.9739 | 0.0045 | 100.0% |
| 0.02 | 0.9800 | 0.9248 | 0.0117 | 98.3% |
| 0.05 | 0.9664 | 0.7780 | 0.0445 | 96.9% |
| 0.10 | 0.9420 | 0.5886 | 0.1078 | 94.5% |
| 0.20 | 0.9092 | 0.4449 | 0.1802 | 91.2% |

## SHAP Global Feature Importance (Top 10)

| Rank | Feature | Mean |SHAP| |
|------|---------|----------|
| 1 | `gas_limit` | 1.391 |
| 2 | `internal_calls` | 1.160 |
| 3 | `log_count` | 0.925 |
| 4 | `gas_limit_zscore_clean` | 0.607 |
| 5 | `gas_price` | 0.471 |
| 6 | `calldata_length` | 0.416 |
| 7 | `log_count_zscore_clean` | 0.192 |
| 8 | `internal_calls_zscore_clean` | 0.182 |
| 9 | `day_of_week` | 0.179 |
| 10 | `gas_price_zscore_clean` | 0.165 |

## Forensic Construct-Validity Corpus

12 independently verified Ethereum exploit transactions from 2023:

| Transaction Hash | Incident | Attack Vector | Loss (USD) | Source |
|------------------|----------|---------------|------------|--------|
| 0xc310...111d | Euler Finance | Flash-Loan Reentrancy | $197,000,000 | Rekt News |
| 0xea34...45e8 | SushiSwap RouteProcessor2 | Bad Callback | $3,300,000 | Rekt News |
| 0xd55e...a95d | Yearn Finance yUSDT | Misconfiguration | $11,540,000 | QuillAudits |
| 0x8db0...3138 | Yearn Finance yUSDT | Misconfiguration | $11,540,000 | QuillAudits |
| 0xeb87...9eb7 | Sturdy Finance | Read-Only Reentrancy | $775,000 | ImmuneBytes |
| 0x485e...f0f3 | KyberSwap Elastic | Tick Manipulation | $48,000,000 | SlowMist |
| 0x09a3...75e8 | KyberSwap Elastic | Tick Manipulation | $48,000,000 | SlowMist |
| 0x396a...5475 | KyberSwap Elastic | Tick Manipulation | $48,000,000 | SlowMist |
| 0xfeed...ace7 | Raft Protocol | Precision Loss | $3,300,000 | Halborn |
| 0xa137...794e | Raft Protocol | Precision Loss | $3,300,000 | ImmuneBytes |
| 0xf63d...5ad5 | Curve Finance DNS Hijack | Malicious Approval | $573,000 | Numen Cyber |
| 0x525f...a5e4 | Curve Finance DNS Hijack | Asset Transfer | $573,000 | Numen Cyber |

**Total documented losses: $286,141,000.**

All 12 transactions scored above the 96th percentile of the test-set anomaly distribution, demonstrating alignment between heuristic thresholds and real-world exploit signatures.

## Reproducibility

All experiments use fixed random seeds:
- Training seed: `43`
- Evaluation seed: `123`
- Bootstrap seed: `456`

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{multisig_anomaly_2025,
  title={Explainable Anomaly Scoring for Ethereum Multisignature Transactions Using Temporal Validation and LightGBM},
  journal={Computers, Materials \& Continua},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported by the Deanship of Scientific Research, Vice Presidency for Graduate Studies and Scientific Research, King Faisal University, Saudi Arabia (Grant No. KFU260311).
