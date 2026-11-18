# MSIPA: Multi-Scale Interval Pattern-Aware Network for ICU Transfer Prediction

<p align="center">
  <img src="MSIPA.png" alt="MSIPA Architecture" width="888">
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3458284">
    <img src="https://img.shields.io/badge/ACM_TKDD-2022-blue" alt="ACM TKDD">
  </a>
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange" alt="TensorFlow / Keras">
</p>

> **[MSIPA: Multi-Scale Interval Pattern-Aware Network for ICU Transfer Prediction](https://doi.org/10.1145/3458284)**  
> Wu Lee, Yuliang Shi, Hongfeng Sun, Lin Cheng, Kun Zhang, Xinjun Wang, Zhiyong Chen  
> *ACM Transactions on Knowledge Discovery from Data (TKDD), Vol. 16, No. 1, Article 17, 2022.*

## Overview

This repository provides the implementation of **MSIPA (Multi-Scale Interval Pattern-Aware Network)**, proposed in our paper published in ACM Transactions on Knowledge Discovery from Data (TKDD).

MSIPA is designed for ICU transfer prediction using longitudinal electronic health records (EHRs). The model captures short-, medium-, and long-term interval patterns through multi-scale temporal modeling and employs scaled dot-product attention and Transformer-based representation learning to integrate contextual information for accurate ICU transfer prediction.

The experiments are conducted on the **MIMIC-III** database and include two prediction tasks: Ward Assignment (WA) and Care Unit Prediction (CU).

## Requirements

The implementation is based on TensorFlow and Keras.

The main dependencies include:

```text
tensorflow
keras
numpy
pandas
scikit-learn
fancyimpute
```

We recommend using a compatible TensorFlow/Keras environment to reproduce the original experiments, as the implementation relies on legacy Keras APIs.

## Dataset

### MIMIC-III Database

MSIPA uses the **MIMIC-III** critical care database, which contains de-identified clinical records collected from intensive care units.

The original database is available from:

**[MIMIC-III — PhysioNet](https://physionet.org/content/mimiciii/1.4/)**

Access to the original MIMIC-III data is subject to PhysioNet credentialing and data-use requirements.

### Project Data

The project-related CSV files are available through Google Drive:

**[MSIPA Data — Google Drive](https://drive.google.com/drive/folders/1EzaOGHMl9FOHm3ntb4jQK8jGGzLO_-E9)**

The preprocessing pipeline requires two files:

| File | Description |
|---|---|
| `lw_icu.csv` | Clinical records and ICU transfer-related features |
| `lw_icu_hd.csv` | Hospital admission identifiers for constructing longitudinal sequences |

## Data Preprocessing

The preprocessing pipeline transforms the original clinical records into model-ready sequences for ICU transfer prediction.

### Step 1. Prepare the Dataset

Download the project-related CSV files from Google Drive and place them in the directory specified by the preprocessing script.

```text
icu_data/
├── lw_icu.csv
└── lw_icu_hd.csv
```

### Step 2. Process Clinical Features

Run the provided ICU data preprocessing script.

The script performs the following operations:

1. Groups clinical records according to hospital admission identifiers (`hadm_id`).
2. Applies forward and backward filling to missing care-unit and ward information.
3. Performs one-hot encoding of categorical features.
4. Uses KNN imputation with `k=10` to handle remaining missing values.
5. Constructs task-specific input sequences and prediction targets.

The intermediate files include:

```text
lw_icu_onehot.csv
lw_icu_knnimpute.csv
lw_icu_knnimpute.npy
```

### Step 3. Generate Model-Ready Sequences

The preprocessing script generates four NumPy files:

| File | Description |
|---|---|
| `x_ward.npy` | Input sequences for Ward Assignment |
| `y_ward.npy` | Ward Assignment prediction targets |
| `x_careunit.npy` | Input sequences for Care Unit Prediction |
| `y_careunit.npy` | Care Unit Prediction targets |

These files are used directly for model training and evaluation.

**Note:** Update the dataset paths in the preprocessing and training scripts according to your local environment.

## MSIPA Training

After preprocessing the dataset, run the corresponding MSIPA training script.

The implementation supports two ICU transfer prediction tasks:

### Ward Assignment (WA)

Use the following files:

```text
x_ward.npy
y_ward.npy
```

The model predicts the destination ward from longitudinal clinical records.

### Care Unit Prediction (CU)

Use the following files:

```text
x_careunit.npy
y_careunit.npy
```

The model predicts the destination ICU care unit.

The original implementation uses five-fold cross-validation and evaluates prediction performance using Micro-F1, Macro-F1, AUROC, and AUPRC.

Please refer to the source code for the original model configuration, training settings, and evaluation procedures.

## Notes

- The implementation uses custom attention and Transformer components defined in `Transformer.py`.
- The preprocessing and training scripts contain local file paths that should be updated before execution.
- The Ward Assignment and Care Unit Prediction tasks use different output dimensions.
- Researchers using MIMIC-III must comply with the applicable PhysioNet data-use agreement.

## Citation

If you find this repository or our work useful in your research, please cite:

```bibtex
@article{10.1145/3458284,
  author = {Lee, Wu and Shi, Yuliang and Sun, Hongfeng and Cheng, Lin and Zhang, Kun and Wang, Xinjun and Chen, Zhiyong},
  title = {MSIPA: Multi-Scale Interval Pattern-Aware Network for ICU Transfer Prediction},
  year = {2021},
  issue_date = {February 2022},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {16},
  number = {1},
  issn = {1556-4681},
  url = {https://doi.org/10.1145/3458284},
  doi = {10.1145/3458284},
  journal = {ACM Trans. Knowl. Discov. Data},
  month = jul,
  articleno = {17},
  numpages = {17},
  keywords = {attention, EHR, interval pattern-aware, ICU transfer prediction}
}
```
