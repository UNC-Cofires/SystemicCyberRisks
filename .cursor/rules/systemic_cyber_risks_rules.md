# Systemic Cyber Risks Project Rules

## Project Overview

This repository predicts vulnerability exploitation likelihood using machine learning. The goal is to identify which CVEs (Common Vulnerabilities and Exposures) from the MITRE/NIST databases are likely to be exploited in the wild, based on CISA's Known Exploited Vulnerabilities (KEV) catalog as ground truth.

### Core Objective
Predict binary classification: Will a CVE be exploited? (0 = No, 1 = Yes)

### Key Challenge
Severe class imbalance: ~0.6% of CVEs are exploited (205:1 ratio)

## Three Modeling Approaches

### 1. CVSS Feature-Based (Baseline)
- **Location**: `modeling/baseline_abel_koshy_07_25.py`, `notebooks/Baseline_Model/`
- **Input**: Structured CVSS metrics (severity scores, attack vectors, complexity, etc.)
- **Features**: ~40 features after one-hot encoding
- **Performance**: ROC-AUC ~0.87
- **Advantages**: Fast, interpretable, explicit domain knowledge
- **Train/Test Split**: 2015-2023 train, 2024+ test (temporal)

### 2. Embedding-Based (NLP Approach)
- **Location**: `embedding_pipeline/`
- **Input**: Unstructured text descriptions of vulnerabilities
- **Features**: 384-dim (Sentence Transformers) or 1536-dim (OpenAI)
- **Performance**: ROC-AUC ~0.87
- **Advantages**: Captures semantic patterns, no need for structured metrics
- **Train/Test Split**: 2015-2024 train, 2025 test (temporal)

### 3. Hybrid Model (Combined)
- **Location**: `embedding_pipeline/run_hybrid_model.py`
- **Input**: CVSS features + embedding vectors concatenated
- **Expected Performance**: May exceed individual approaches
- **Advantages**: Complementary signals from structure and semantics

## Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA GENERATION (generate_data/)                            │
│    CVE JSON → NVD API → vulnerabilities.parquet                │
│    Time: Several hours due to API rate limits                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. DATA CLEANING (modeling/baseline_*.py or notebooks/)        │
│    vulnerabilities.parquet + KEV → data.csv                    │
│    Filters: Post-2015, CVSS v3.0+, remove v4.0                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3A. CVSS MODELING (modeling/ or notebooks/)                    │
│     Feature engineering → train models → evaluate              │
└─────────────────────────────────────────────────────────────────┘
                            OR
┌─────────────────────────────────────────────────────────────────┐
│ 3B. EMBEDDING MODELING (embedding_pipeline/)                   │
│     Extract descriptions → preprocess → embed → train          │
└─────────────────────────────────────────────────────────────────┘
                            OR
┌─────────────────────────────────────────────────────────────────┐
│ 3C. HYBRID MODELING (embedding_pipeline/run_hybrid_model.py)  │
│     Load CVSS data + embeddings → concatenate → train          │
└─────────────────────────────────────────────────────────────────┘
```

## Embedding Pipeline Details

### Pipeline Steps

1. **Extract Descriptions** (`1_extract_descriptions.py`)
   - Input: `data/vulnerabilities.parquet`, `data/known_exploited_vulnerabilities.csv`
   - Output: `embedding_pipeline/data/descriptions_with_labels.csv`
   - Filters to 2015-2025, adds KEV labels
   - ~330K CVEs, ~2K exploited

2. **Preprocess Text** (`2_preprocess_text.py`)
   - Input: `descriptions_with_labels.csv`
   - Output: `descriptions_preprocessed.csv`
   - Removes: URLs, emails, versions, stop words, code snippets
   - Result: ~41% word count reduction

3. **Generate Embeddings** (choice of 3a or 3b)
   - **3a. OpenAI** (`3a_generate_embeddings_openai.py`)
     - Model: `text-embedding-3-small` (1536-dim)
     - Requires: `.env` with `OPENAI_API_KEY`
     - Cost: ~$5-10 for full dataset
     - Speed: Moderate (rate limited)
   - **3b. Sentence Transformers** (`3b_generate_embeddings_sentencetransformer.py`)
     - Model: `all-MiniLM-L6-v2` (384-dim)
     - Requires: No API, runs locally
     - Cost: FREE
     - Speed: 2-3 hours on Apple Silicon
     - Device: Auto-detects MPS/CUDA/CPU

4. **Train Models** (`4_train_models.py`)
   - Models: Logistic Regression + Random Forest
   - Class weight: 'balanced' (handles imbalance)
   - Outputs: ROC curves, PR curves, confusion matrices, comparison charts
   - Saves: JSON results, prediction CSVs

5. **Hybrid Model** (`run_hybrid_model.py`)
   - Trains 3 models: CVSS-only, Embedding-only, Hybrid
   - Generates comparison visualizations
   - Shows which approach works best

### Running the Embedding Pipeline

```bash
# Sentence Transformers (recommended default)
cd /Users/abelkoshy/Documents/GitHub/SystemicCyberRisks
source venv/bin/activate
./embedding_pipeline/run_full_pipeline.sh sentencetransformer

# OpenAI (if you have API key)
./embedding_pipeline/run_full_pipeline.sh openai

# Individual steps (if pipeline interrupted)
python embedding_pipeline/1_extract_descriptions.py
python embedding_pipeline/2_preprocess_text.py
python embedding_pipeline/3b_generate_embeddings_sentencetransformer.py
python embedding_pipeline/4_train_models.py --embedding-type sentencetransformer

# Hybrid model (after sentence transformer completes)
python embedding_pipeline/run_hybrid_model.py
```

## Critical Design Decisions

### Year Filtering (Post-2015)
- **Why**: CVSS v3.0 was released in 2015
- **Before**: v2.0 had ~26% missing values, inconsistent scoring
- **Impact**: More reliable, modern threat landscape

### Temporal Train/Test Split
- **CVSS Models**: Train 2015-2023, Test 2024+
- **Embedding Models**: Train 2015-2024, Test 2025
- **Why**: Simulates real-world deployment (predict future exploits)
- **Prevents**: Data leakage, overly optimistic results

### Class Imbalance Handling
- **Technique 1**: `class_weight='balanced'` in all models
- **Technique 2**: Focus on ROC-AUC, not accuracy
- **Technique 3**: Precision-Recall curves for evaluation
- **Why**: Accuracy is misleading with 99.4% negative class

### Feature Engineering (CVSS)
- Binary encoding: Scores ≥7 = high-risk (1), else (0)
- One-hot encoding: Drop first to avoid multicollinearity
- Scaling: MinMaxScaler on numeric features
- Dropped: CVSS v4.0, temporal metrics (too sparse)

### NLP Preprocessing (Embeddings)
- Custom stop words: 'cve', 'vulnerability', 'allows', 'could', 'via', etc.
- Token filtering: 3-20 character length
- Removes: URLs, emails, version numbers, code snippets
- Result: Cleaner semantic signal

## File Structure

### Input Data (Required)
- `cves/YYYY/XXXX/CVE-YYYY-XXXX.json` - Raw CVE records (large, gitignored)
- `data/known_exploited_vulnerabilities.csv` - CISA KEV catalog (target labels)
- `data/vulnerabilities.parquet` - Generated dataset (committed)
- `.env` - OpenAI API key (gitignored, use `.env.template`)

### Generated Data (Gitignored except noted)
- `data/data.csv` - Cleaned CVSS dataset
- `data/processed_vulnerabilities.csv` - Intermediate from notebooks
- `embedding_pipeline/data/descriptions_with_labels.csv` - Raw descriptions
- `embedding_pipeline/data/descriptions_preprocessed.csv` - Cleaned text
- `embedding_pipeline/data/embeddings_*.npz` - Compressed embeddings
- `embedding_pipeline/data/metadata_*.csv` - CVE IDs, years, targets

### Output Files
- `roc_curve.png`, `confusion_matrix_baseline.png` - Baseline model outputs
- `embedding_pipeline/results/results_*.json` - Model metrics
- `embedding_pipeline/results/predictions_*.csv` - Test set predictions
- `embedding_pipeline/visualizations/*.png` - All visualization plots
- `embedding_pipeline/hybrid_results/` - Hybrid model outputs

## Code Style & Best Practices

### Python
- Use type hints where practical
- Include comprehensive docstrings (especially in pipeline scripts)
- Print progress information (percentage, time estimates, record counts)
- Use `argparse` for command-line arguments
- Handle errors gracefully with informative messages

### Machine Learning
- Always use temporal splits (never random splits)
- Always use `class_weight='balanced'` for imbalanced data
- Save models with pickle, results with JSON, predictions with CSV
- Generate multiple visualizations: ROC, PR, confusion matrix, comparison bars
- Report multiple metrics: ROC-AUC, Average Precision, F1, confusion matrix

### Data Processing
- Use Parquet for large datasets (faster than CSV)
- Validate data after each pipeline step
- Print summary statistics (counts, distributions, missing values)
- Use chunking/batching for large datasets (embeddings)
- Save intermediate results (enables resuming if interrupted)

### Visualization
- Use `matplotlib.use('Agg')` for non-interactive backend
- Set style: `plt.style.use('seaborn-v0_8-darkgrid')`
- Save high resolution: `dpi=150` or `dpi=300`
- Always use `bbox_inches='tight'`
- Include legends, titles, axis labels
- Add metric values in plot titles or legends

## Common Commands

### Environment Setup
```bash
cd /Users/abelkoshy/Documents/GitHub/SystemicCyberRisks
source venv/bin/activate  # Always activate venv first
```

### Data Generation (Slow - Only Run When Needed)
```bash
cd generate_data
python create_vulnerabilities_dataset.py  # Takes several hours
```

### Quick Baseline Model
```bash
python modeling/baseline_abel_koshy_07_25.py
```

### Embedding Pipeline
```bash
# See "Running the Embedding Pipeline" section above
./embedding_pipeline/run_full_pipeline.sh sentencetransformer
```

### Jupyter Notebooks (Interactive Analysis)
```bash
jupyter notebook notebooks/Baseline_Model/0_Data_Loading_and_EDA.ipynb
# Run notebooks 0-5 in sequence
```

## Troubleshooting

### Issue: Embeddings generation too slow
- **Solution**: Reduce batch size in script (default 256 → 128 or 64)
- **Check**: GPU acceleration enabled (MPS for Mac, CUDA for Linux)

### Issue: Out of memory during embedding generation
- **Solution**: Reduce batch size, close other applications
- **Alternative**: Use OpenAI API (offloads computation)

### Issue: OpenAI API rate limit errors
- **Solution**: Increase `RATE_LIMIT_DELAY` in `3a_generate_embeddings_openai.py`
- **Check**: API key has sufficient quota

### Issue: Model performance unexpectedly low
- **Check 1**: Temporal split used (not random)?
- **Check 2**: Class weights balanced?
- **Check 3**: Evaluation on correct test set (not training data)?

### Issue: Visualization not generating
- **Check 1**: `embedding_pipeline/visualizations/` directory exists
- **Check 2**: Script completed successfully (check for errors)
- **Solution**: Run `4_train_models.py` or `run_hybrid_model.py` again

### Issue: Missing .env file for OpenAI
- **Solution**: Create from template: `cp .env.template .env`
- **Edit**: Add your API key: `OPENAI_API_KEY=sk-your-key-here`

## Performance Benchmarks

| Approach | ROC-AUC | Avg Precision | F1 Score | Speed |
|----------|---------|---------------|----------|-------|
| CVSS Features (Baseline) | 0.87 | ~0.02 | ~0.04 | Fast |
| Sentence Transformer Embeddings | 0.87 | ~0.02 | ~0.04 | Moderate |
| OpenAI Embeddings | TBD | TBD | TBD | Moderate |
| Hybrid (CVSS + ST Embeddings) | TBD | TBD | TBD | Moderate |

**Note**: All models use temporal splits and balanced class weights.

## Research Context

### Domain
- **Field**: Cybersecurity, Vulnerability Management, Risk Assessment
- **Problem**: Triage vulnerabilities by exploitation likelihood
- **Impact**: Prioritize patching, allocate security resources efficiently

### Dataset
- **Source**: MITRE CVE + NIST NVD + CISA KEV
- **Size**: ~330K CVEs (2015-2025)
- **Labels**: 2,045 known exploited (~0.6%)
- **Features**: CVSS metrics + text descriptions

### Evaluation
- **Primary Metric**: ROC-AUC (measures discrimination)
- **Secondary Metric**: Average Precision (PR-AUC, handles imbalance)
- **Tertiary Metrics**: F1, confusion matrix, precision, recall
- **Baseline**: Random classifier (ROC-AUC = 0.50), prevalence (AP = 0.006)

## Future Improvements

1. **Temporal Cross-Validation**: Multiple year-based folds
2. **Fine-tuning**: Adapt sentence transformer on CVE domain
3. **Ensemble Methods**: Stack multiple model predictions
4. **Active Learning**: Focus on uncertain predictions
5. **Explainability**: SHAP values, attention visualization
6. **Multi-modal**: Incorporate patches, code snippets, metadata
7. **Online Learning**: Update models as new exploits discovered
8. **Threshold Optimization**: Tune decision threshold for operational goals

## Dependencies

### Core
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning
- matplotlib, seaborn - Visualization

### Embedding-Specific
- sentence-transformers - Local embeddings
- torch - PyTorch backend (MPS/CUDA support)
- openai - OpenAI API client
- python-dotenv - Environment variable management
- nltk - NLP preprocessing

### Optional
- jupyter - Interactive notebooks
- requests - API calls for data generation

**Installation**: All dependencies are in `requirements.txt` (if exists) or install individually as needed.

## Important Notes

- **Do not commit** large data files (handled by .gitignore)
- **Do commit** small config files, scripts, documentation
- **Parquet format** preferred for large datasets (vulnerabilities.parquet)
- **API key security**: Never commit `.env` file
- **Model versioning**: Save models with date/version in filename
- **Reproducibility**: Set random seeds where applicable (`random_state=42`)

## Contact & Collaboration

- Open GitHub issues for bugs or feature requests
- Consult `README.md` and `CLAUDE.md` for additional documentation
- Check `embedding_pipeline/README.md` for embedding-specific details
- Review `embedding_pipeline/IMPLEMENTATION_SUMMARY.md` for current status

---

**Last Updated**: 2025-11-22  
**Maintainer**: Abel Koshy  
**Project**: Systemic Cyber Risks - Vulnerability Exploitation Prediction

