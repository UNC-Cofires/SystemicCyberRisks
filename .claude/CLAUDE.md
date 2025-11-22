# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository evaluates systemic cyber risks using vulnerability data from MITRE CVE and NIST NVD databases. The goal is to predict which vulnerabilities are likely to be exploited in the wild using machine learning models trained on CVSS scores and other technical characteristics.

**Key Dataset**: CISA Known Exploited Vulnerabilities (KEV) catalog serves as ground truth labels for supervised learning.

## Common Commands

### Embedding-Based Prediction Pipeline (NEW)

Predict exploitation using text embeddings of vulnerability descriptions:

```bash
# Full pipeline with Sentence Transformers (local, no API needed)
./embedding_pipeline/run_full_pipeline.sh sentencetransformer

# OR: Full pipeline with OpenAI embeddings (requires API key in .env)
./embedding_pipeline/run_full_pipeline.sh openai

# OR: Run steps individually
python embedding_pipeline/1_extract_descriptions.py
python embedding_pipeline/2_preprocess_text.py
python embedding_pipeline/3b_generate_embeddings_sentencetransformer.py
python embedding_pipeline/4_train_models.py --embedding-type sentencetransformer
```

**Note**: Embedding generation takes 2-3 hours for 330K descriptions on Apple Silicon. Results include both Logistic Regression and Random Forest models with full performance comparison.

**Train/Test Split**: All models use temporal split - training on 2015-2023 data, testing on 2024+ data.

### Data Generation Pipeline

Generate the raw vulnerability dataset from CVE JSON files and NIST NVD API:

```bash
cd generate_data
python create_vulnerabilities_dataset.py
```

This orchestrates three sequential steps:
1. `read_nvd_api.py` - Queries NIST NVD API (takes several hours due to rate limits)
2. `merge_files.py` - Combines annual vulnerability files
3. `pull_description.py` - Extracts vulnerability descriptions

**Note**: Requires CVE data in `cves/YYYY/XXXX/CVE-YYYY-XXXX.json` format. Download from https://www.cve.org/Downloads or UNC Longleaf server.

### Baseline Modeling (Quick Start)

Run the complete data cleaning and baseline modeling pipeline:

```bash
python modeling/baseline_abel_koshy_07_25.py
```

Produces:
- `data/data.csv` - Cleaned dataset
- `roc_curve.png` - ROC curve visualization
- `confusion_matrix_baseline.png` - Confusion matrix visualization
- Console output with model metrics

### Interactive Analysis with Jupyter Notebooks

For detailed exploratory analysis and advanced modeling:

```bash
# Start Jupyter and run notebooks in sequence:
jupyter notebook notebooks/Baseline_Model/0_Data_Loading_and_EDA.ipynb
jupyter notebook notebooks/Baseline_Model/1_Data_Preprocessing.ipynb
jupyter notebook notebooks/Baseline_Model/2_Baseline_Modeling.ipynb
jupyter notebook notebooks/Baseline_Model/3_Advanced_Modeling.ipynb
jupyter notebook notebooks/Baseline_Model/4_Cross_Validation.ipynb
```

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install requests pandas numpy matplotlib scikit-learn seaborn jupyter
```

## Architecture and Data Flow

### Two Complementary Approaches

**1. CVSS Feature-Based (Original)**
Uses structured vulnerability metrics (severity scores, attack vectors, etc.)

**2. Embedding-Based (NEW)**
Uses NLP embeddings of vulnerability descriptions for prediction

Both approaches predict the same target: CISA Known Exploited Vulnerabilities.

### Three-Stage CVSS Pipeline Architecture

```
┌─────────────────┐
│ 1. Data Gen     │  CVE JSON files → NVD API → vulnerabilities.parquet
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Cleaning     │  vulnerabilities.parquet + KEV catalog → data.csv
└────────┬────────┘         (filters post-2015, creates target labels)
         │
         ▼
┌─────────────────┐
│ 3. Modeling     │  data.csv → feature engineering → trained models
│                 │  (Temporal split: train 2015-2023, test 2024+)
└─────────────────┘
```

### Critical Design Decisions

**Year Filtering**: Dataset is filtered to post-2015 vulnerabilities because:
- CVSS v3.0 was released in 2015; earlier data uses inconsistent v2.0 scoring
- Pre-2016 data has ~26% missing values in key categorical fields
- Modern threat landscape is more relevant for current predictions

**Target Variable Creation**: Binary classification where `target=1` means the CVE appears in CISA's KEV catalog (known exploited). This creates severe class imbalance (~0.6% positive class) that accurately reflects real-world exploitation rates.

**Feature Engineering Strategy**:
- Binary encoding: Scores ≥7 encoded as high-risk (1), else low-risk (0)
- Categorical encoding: One-hot encoding with `drop='first'` to avoid dummy variable trap
- Scaling: MinMaxScaler applied to `numScores` and `agreement` columns
- Dropped features: Version 4.0 CVEs, environmental/temporal CVSS metrics (too sparse), and raw text descriptions

### Embedding Pipeline Architecture (NEW)

```
embedding_pipeline/
├── 1. Extract Descriptions    → Raw CVE descriptions + KEV labels
├── 2. Preprocess Text          → NLP cleaning (stop words, URLs, etc.)
├── 3a. OpenAI Embeddings       → text-embedding-3-small (1536-dim)
├── 3b. Sentence Transformers   → all-MiniLM-L6-v2 (384-dim, local)
└── 4. Train Models             → Logistic Reg + Random Forest
                                  Train: 2015-2023, Test: 2024+
```

**Key Differences from CVSS Approach:**
- Input: Unstructured text vs structured features
- Features: 384/1536 embedding dims vs ~40 CVSS features
- Interpretability: Black box vs feature importances
- Computation: Slower (embedding gen) vs fast
- Expected ROC-AUC: ~0.65-0.80 vs ~0.87

See `embedding_pipeline/README.md` for complete documentation.

### Key Scripts

**embedding_pipeline/** (NEW)
- `1_extract_descriptions.py`: Extract descriptions + KEV labels, filter 2015-2025
- `2_preprocess_text.py`: Comprehensive NLP cleaning (stop words, URLs, versions)
- `3a_generate_embeddings_openai.py`: OpenAI API embeddings (requires .env with API key)
- `3b_generate_embeddings_sentencetransformer.py`: Local embeddings (no API needed)
- `4_train_models.py`: Train both LR and RF, temporal split, generate visualizations
- `run_full_pipeline.sh`: Master script to run complete pipeline

**generate_data/create_vulnerabilities_dataset.py**
- Orchestrates the three-step data collection pipeline
- Sequential execution with error handling between steps
- Progress tracking and timing information

**modeling/baseline_abel_koshy_07_25.py**
- Combined data cleaning + logistic regression baseline (single script execution)
- Reads from `.parquet` format (fallback to `.csv`)
- Uses temporal split: train 2015-2023, test 2024+
- Uses `class_weight='balanced'` to handle class imbalance
- Produces ROC curve, confusion matrix, and classification report

**notebooks/Baseline_Model/** (5 notebooks in sequence)
- `0_Data_Loading_and_EDA.ipynb`: Load raw data, filter to 2016+, create target variable, temporal analysis
- `1_Data_Preprocessing.ipynb`: Feature engineering, encoding, scaling
- `2_Baseline_Modeling.ipynb`: Logistic regression baseline with class imbalance handling
- `3_Advanced_Modeling.ipynb`: Random Forest, Gradient Boosting, ensemble methods
- `4_Cross_Validation.ipynb`: Stratified k-fold validation, hyperparameter tuning

**keyword_generator.py** (standalone utility)
- RAG-based keyword extraction using OpenAI embeddings + FAISS
- Requires `MITRE Tactics.docx` and `descriptions_only.csv` as inputs
- Uses MITRE ATT&CK framework to ground keyword extraction
- Outputs `extracted_keywords.csv`

## Data Files

### Input Data (Required)

- `cves/YYYY/XXXX/CVE-YYYY-XXXX.json` - Raw CVE records from MITRE
- `data/known_exploited_vulnerabilities.csv` - CISA KEV catalog (target labels)
- `.env` - Environment variables (OPENAI_API_KEY) - copy from .env.template

### Generated Data

- `data/vulnerabilities.parquet` - Complete vulnerability dataset from generate_data pipeline (Parquet format for efficient storage)
- `data/data.csv` - Cleaned dataset ready for modeling (from baseline_model script)
- `data/processed_vulnerabilities.csv` - Intermediate output from notebook 0 (filtered to 2016+)
- `data/vulnerabilities_YYYY.csv` - Annual vulnerability files (intermediate from pipeline)

### Embedding Pipeline Data (NEW)

- `embedding_pipeline/data/descriptions_with_labels.csv` - Extracted descriptions + targets
- `embedding_pipeline/data/descriptions_preprocessed.csv` - Cleaned text for embedding
- `embedding_pipeline/data/embeddings_*.npz` - Compressed embedding arrays
- `embedding_pipeline/data/metadata_*.csv` - CVE IDs, years, targets

### Output Files

- `roc_curve.png` - ROC curve visualization (from baseline_model script)
- `confusion_matrix_baseline.png` - Confusion matrix visualization (from baseline_model script)
- `extracted_keywords.csv` - RAG-extracted keywords (from keyword_generator.py)
- `embedding_pipeline/results/results_*.json` - Detailed model metrics (LR + RF)
- `embedding_pipeline/results/predictions_*.csv` - Test set predictions with probabilities
- `embedding_pipeline/visualizations/*.png` - ROC curves, PR curves, confusion matrices, model comparisons
- `embedding_pipeline/visualizations/hybrid_confusion_matrices.png` - Confusion matrices for CVSS, Embedding, and Hybrid models

## Class Imbalance Handling

The extreme class imbalance (~205:1 ratio) requires specialized techniques:

1. **Class weighting**: Use `class_weight='balanced'` in scikit-learn models
2. **Temporal splitting**: Train on 2015-2023, test on 2024+ (prevents data leakage and simulates real-world prediction)
3. **Evaluation metrics**: Focus on ROC-AUC, precision-recall curves, not accuracy
4. **Threshold tuning**: Optimize decision threshold for recall vs precision trade-off

The baseline model achieves ~0.87 ROC-AUC, demonstrating that technical vulnerability characteristics have predictive power for exploitation likelihood.

## Important Notes

- The NVD API step (`read_nvd_api.py`) respects rate limits (50 requests per 30 seconds) and may take several hours
- API key is included in `read_nvd_api.py` for higher rate limits
- All scripts in `generate_data/` are designed to run from that subdirectory (use relative paths like `../data/`)
- CVSS version 4.0 records are filtered out as they are inconsistent or erroneous
- Duplicate CVE IDs are dropped (keeping first occurrence) to prevent training bias
- The `year` column is extracted from CVE ID format: `CVE-YYYY-NNNNN`
- Missing values in categorical features (attackVector, attackComplexity, etc.) are typically from pre-2016 CVSS v2.0 records

## RAG Ingestion Material

The `RAG Ingestion Material/` directory contains MITRE ATT&CK tactics in YAML format (TA001-TA040). These are used by `keyword_generator.py` for grounded keyword extraction from vulnerability descriptions.
