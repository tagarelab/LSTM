# LSTM-Powered Progression Index for Parkinson's Disease Monitoring 

**Author**: Felice Dong \
**Advisor**: Dr. Hemant Tagare \
**Department**: Statistics and Data Science, Yale University \
**Completion Date**: April 30, 2025 

## Abstract 
This repository contains the implementation of a novel progression index for Parkinson's disease (PD) monitoring using wearable sensor data and 
Long Short-Term Memory (LSTM) neural networks. The model transforms daily physical activity patterns from ambulatory data into a
meaningful disease progression metric, demonstrating consistent visual separation between PD patients and healthy controls across cross-validation splits.

## Contents 
1. [Project Overview](#Project-Overview)
2. [Repo Structure](##Repo-Structure)
3. [Quick Start](##Quick-Start)

## Project Overview 
Parkinson's disease affects over 10 million people worldwide as the most common age-related motor disorder. Current clinical assessments using the 
Movement Disorder Society-Unified Parkinson's Disease Rating Scale (MDS-UPDRS) have limitations including subjectivity and accessibility constraints. 
This project addresses these challenges by:

- Developing an objective, continuous monitoring approach using wearable sensor data
- Leveraging LSTM networks to capture temporal dependencies in activity patterns
- Creating a unified progression index that transforms ambulatory activity into meaningful clinical metrics
- Comparing feature representations (conventional weekdays vs. activity-sorted features)

#### Key Findings

- Consistent visual separation between PD and healthy control subjects across 20 independent train-test splits
- Superior performance with activity features sorted by intensity compared to conventional weekday labels
- Peak activity capability identified as the most salient discriminator for disease status
- Weekend activities showed higher discriminative power than structured weekday routines

## Repo Structure 
```
├── data/                          # Data files
│   ├── data_preimp.csv            # Main preprocessed dataset
│   ├── enhanced_fake_data.csv     # Enhanced synthetic dataset
│   ├── fake_data.csv              # Basic synthetic dataset
│   └── fake_data_generation.ipynb # Synthetic data generation notebook
│
├── docs/                          # Documentation and presentations
│   ├── fd_poster.pdf              # Research poster presentation
│   └── fd_thesis.pdf              # Complete thesis report document
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── felice_lstm_fake_data.ipynb # LSTM analysis on synthetic data
│   └── thesis_model_cleaned.ipynb  # Clean version of thesis model
│
├── src/                           # Source code
│   ├── data_cleaning.qmd          # Data preprocessing pipeline (R/Quarto)
│   ├── lstm_pd_progression.py     # Main LSTM implementation script (Python)
│   └── requirements.txt           # Python dependencies
│
└── README.md                      # This file
```
## Quick Start 
