# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predicts which CVEs (Common Vulnerabilities and Exposures) will be exploited in the wild using ML models. Uses CISA Known Exploited Vulnerabilities (KEV) catalog as ground truth labels for binary classification (`target=1` = exploited). The dataset has severe class imbalance (~0.6% positive, 205:1 ratio).

## Common Commands

```bash
# Activate environment (always do this first)
source venv/bin/activate

# Baseline CVSS model (quick, produces ROC curve + confusion matrix)
python modeling/baseline_abel_koshy_07_25.py

# Embedding pipeline (local, no API needed)
./embedding_pipeline/run_full_pipeline.sh sentencetransformer

# Embedding pipeline (OpenAI, requires .env with OPENAI_API_KEY)
./embedding_pipeline/run_full_pipeline.sh openai

# Individual embedding pipeline steps
python embedding_pipeline/1_extract_descriptions.py
python embedding_pipeline/2_preprocess_text.py
python embedding_pipeline/3b_generate_embeddings_sentencetransformer.py
python embedding_pipeline/4_train_models.py --embedding-type sentencetransformer

# Final hybrid model (best performance, requires OpenAI embeddings)
python embedding_pipeline/final_hybrid.py

# Generate individual CVSS-only and embedding-only visualizations (outputs to repo root)
python embedding_pipeline/generate_individual_visualizations.py

# Data generation from scratch (several hours due to NVD API rate limits)
cd generate_data && python create_vulnerabilities_dataset.py

# Jupyter notebooks (run 0-5 in sequence)
jupyter notebook notebooks/Baseline_Model/0_Data_Loading_and_EDA.ipynb
```

**Dependencies**: `pip install requests pandas numpy matplotlib scikit-learn seaborn jupyter sentence-transformers openai python-dotenv nltk torch`

**No automated tests exist.** Validation is done through model metrics, notebooks, and manual execution.

## Architecture

Three complementary modeling approaches predict the same target (CISA KEV):

### 1. CVSS Feature-Based (Baseline)
- **Scripts**: `modeling/baseline_abel_koshy_07_25.py`, `notebooks/Baseline_Model/0-4`
- **Input**: Structured CVSS metrics → ~40 features after one-hot encoding
- **ROC-AUC**: ~0.87

### 2. Embedding-Based (NLP)
- **Scripts**: `embedding_pipeline/1-4` + `run_full_pipeline.sh`
- **Input**: CVE description text → embeddings (384-dim local or 1536-dim OpenAI)
- **Models**: Logistic Regression + Random Forest
- **ROC-AUC**: ~0.87

### 3. Hybrid (Best Performance)
- **Scripts**: `embedding_pipeline/final_hybrid.py`, `embedding_pipeline/run_hybrid_model.py`
- **Input**: OpenAI embeddings (1536-dim) + CVSS features (38-dim) → 1,574 features
- **Model**: Logistic Regression with balanced class weights
- **ROC-AUC**: 0.92 | Recall: 59.3% | Specificity: 94.7%
- **Results**: `embedding_pipeline/final_hybrid_results/`

### Data Flow

```
CVE JSON files → generate_data/ → data/vulnerabilities.parquet
                                          ↓
                              + data/known_exploited_vulnerabilities.csv (KEV labels)
                                          ↓
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
            CVSS Modeling         Embedding Pipeline      Hybrid Model
         (modeling/ + notebooks)  (embedding_pipeline/)   (final_hybrid.py)
```

## Critical Design Decisions

**Temporal train/test split**: Train on 2015-2023, test on 2024+. Never use random splits — this simulates real-world deployment and prevents data leakage.

**Post-2015 filtering**: CVSS v3.0 was released in 2015; earlier data has inconsistent v2.0 scoring and ~26% missing values.

**Class imbalance handling**: Always use `class_weight='balanced'` in scikit-learn models. Evaluate with ROC-AUC and precision-recall curves, never accuracy.

**CVSS v4.0 exclusion**: Filtered out as inconsistent/erroneous.

**Feature engineering**: Binary encoding (scores ≥7 = high-risk), one-hot encoding with `drop='first'`, MinMaxScaler on numeric columns. Duplicate CVE IDs dropped (keep first).

**NLP preprocessing**: 210 custom stop words (CVE-specific terms like 'vulnerability', 'allows'), URL/version removal, token filtering (3-20 chars). Results in ~41% word count reduction.

**Reproducibility**: `random_state=42` throughout.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `modeling/baseline_abel_koshy_07_25.py` | Data cleaning + logistic regression baseline (reads parquet, outputs data.csv + visualizations) |
| `embedding_pipeline/final_hybrid.py` | Production hybrid model (OpenAI embeddings + CVSS features) |
| `embedding_pipeline/run_hybrid_model.py` | Trains and compares all 3 approaches side-by-side |
| `embedding_pipeline/4_train_models.py` | Train LR + RF on embeddings, accepts `--embedding-type` arg |
| `embedding_pipeline/3a_generate_embeddings_openai.py` | OpenAI API embeddings (requires `.env`) |
| `embedding_pipeline/3b_generate_embeddings_sentencetransformer.py` | Local embeddings (all-MiniLM-L6-v2, uses MPS/CUDA/CPU) |
| `generate_data/create_vulnerabilities_dataset.py` | Orchestrates NVD API collection → merge → description extraction |
| `embedding_pipeline/generate_individual_visualizations.py` | Trains CVSS-only and embedding-only LR models, outputs 4 PNGs to repo root |
| `keyword_generator.py` | RAG-based keyword extraction using MITRE ATT&CK + FAISS |

**Notebooks** (`notebooks/Baseline_Model/`): 6 sequential notebooks (0-5) covering EDA, preprocessing, baseline, advanced modeling, cross-validation, and hybrid model exploration.

## Data Files

**Committed to repo**:
- `data/vulnerabilities.parquet` — Complete vulnerability dataset (~330K CVEs, 2015-2025)
- `data/known_exploited_vulnerabilities.csv` — CISA KEV catalog (ground truth labels)

**Generated** (gitignored): `data/data.csv`, `embedding_pipeline/data/` (descriptions, embeddings as `.npz`, metadata), `embedding_pipeline/results/`, `embedding_pipeline/final_hybrid_results/`

**Root-level visualizations**: `final_hybrid_roc_curve.png`, `final_hybrid_confusion_matrix.png`, `cvss_roc_curve.png`, `cvss_confusion_matrix.png`, `openai_embedding_roc_curve.png`, `openai_embedding_confusion_matrix.png` — all use the same aligned dataset (45K test CVEs) for direct comparison across the three approaches.

**Environment**: `.env` with `OPENAI_API_KEY` (copy from `.env.template`; gitignored)

## Conventions

- Use Parquet for large datasets, CSV for small/intermediate files
- `generate_data/` scripts use relative paths — run from that subdirectory
- Embedding generation is a one-time process (~2-3 hours on Apple Silicon); results are saved and reused
- Visualization backend: `matplotlib.use('Agg')` with `plt.style.use('seaborn-v0_8-darkgrid')`, save at `dpi=150+` with `bbox_inches='tight'`
- Save models with pickle, results with JSON, predictions with CSV
- All pipeline scripts print progress information (batch counts, percentages, timing)
- `RAG Ingestion Material/` contains MITRE ATT&CK tactics (YAML, TA001-TA040) used by `keyword_generator.py`
