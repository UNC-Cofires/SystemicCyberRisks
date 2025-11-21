# Embedding-Based Vulnerability Exploitation Prediction

This directory contains a complete pipeline for predicting vulnerability exploitation using text embeddings of CVE descriptions.

## Overview

Instead of using structured CVSS features, this approach uses **natural language embeddings** of vulnerability descriptions to predict which CVEs are likely to be exploited in the wild (based on CISA's Known Exploited Vulnerabilities catalog).

### Key Features

- **NLP Preprocessing**: Comprehensive text cleaning with stop word removal, URL filtering, and token normalization
- **Dual Embedding Approaches**: Both OpenAI API and local Sentence Transformers
- **Temporal Evaluation**: Train on 2015-2024, test on 2025 data
- **Model Comparison**: Logistic Regression vs Random Forest with detailed performance metrics
- **Class Imbalance Handling**: Balanced class weights for the ~0.6% exploitation rate

## Pipeline Architecture

```
1. Extract Descriptions    → descriptions_with_labels.csv
   ├─ Load vulnerabilities.csv.gz
   ├─ Filter to 2015-2025
   └─ Add KEV target labels

2. Preprocess Text         → descriptions_preprocessed.csv
   ├─ Lowercase & clean URLs/emails/versions
   ├─ Remove stop words
   └─ Filter short tokens

3a. Generate Embeddings    → embeddings_openai.npz
    (OpenAI API)
   ├─ text-embedding-3-small (1536-dim)
   └─ Batch processing with rate limiting

3b. Generate Embeddings    → embeddings_sentencetransformer.npz
    (Local)
   ├─ all-MiniLM-L6-v2 (384-dim)
   └─ GPU-accelerated on MPS/CUDA

4. Train Models            → results/ + visualizations/
   ├─ Temporal split: 2015-2024 train, 2025 test
   ├─ Logistic Regression (class_weight='balanced')
   ├─ Random Forest (100 estimators, balanced)
   └─ Performance comparison + visualizations
```

## Quick Start

### Option 1: Sentence Transformers (Local, No API)

```bash
# Run the complete pipeline
python embedding_pipeline/1_extract_descriptions.py
python embedding_pipeline/2_preprocess_text.py
python embedding_pipeline/3b_generate_embeddings_sentencetransformer.py
python embedding_pipeline/4_train_models.py --embedding-type sentencetransformer
```

### Option 2: OpenAI Embeddings (Requires API Key)

```bash
# Setup API key
cp .env.template .env
# Edit .env and add your OpenAI API key

# Run pipeline with OpenAI embeddings
python embedding_pipeline/1_extract_descriptions.py
python embedding_pipeline/2_preprocess_text.py
python embedding_pipeline/3a_generate_embeddings_openai.py
python embedding_pipeline/4_train_models.py --embedding-type openai
```

## Directory Structure

```
embedding_pipeline/
├── README.md                                    # This file
├── 1_extract_descriptions.py                   # Extract CVE descriptions + labels
├── 2_preprocess_text.py                        # NLP preprocessing
├── 3a_generate_embeddings_openai.py           # OpenAI embeddings
├── 3b_generate_embeddings_sentencetransformer.py  # Local embeddings
├── 4_train_models.py                           # Train & evaluate models
├── data/                                       # Generated datasets
│   ├── descriptions_with_labels.csv           # Raw descriptions
│   ├── descriptions_preprocessed.csv          # Cleaned text
│   ├── embeddings_openai.npz                  # OpenAI embeddings
│   ├── embeddings_sentencetransformer.npz     # ST embeddings
│   ├── metadata_openai.csv                    # IDs/years/targets
│   └── metadata_sentencetransformer.csv       # IDs/years/targets
├── results/                                    # Model outputs
│   ├── results_openai.json                    # Detailed metrics
│   ├── results_sentencetransformer.json       # Detailed metrics
│   ├── predictions_openai.csv                 # Test set predictions
│   └── predictions_sentencetransformer.csv    # Test set predictions
└── visualizations/                             # Performance plots
    ├── roc_curves_*.png                       # ROC curve comparison
    ├── precision_recall_*.png                 # PR curve comparison
    ├── confusion_matrices_*.png               # Confusion matrices
    └── model_comparison_*.png                 # Bar chart comparison
```

## Dependencies

All dependencies are listed in the main project requirements. Key libraries:

```
pandas
numpy
scikit-learn
sentence-transformers  # For local embeddings
openai                 # For API embeddings
python-dotenv          # For .env file
nltk                   # For NLP preprocessing
matplotlib             # For visualizations
seaborn                # For visualizations
```

## Preprocessing Details

The NLP preprocessing pipeline (`2_preprocess_text.py`) performs:

1. **Lowercasing**: Normalize all text
2. **URL/Email Removal**: Clean web addresses and emails
3. **Version Number Removal**: Remove software versions (e.g., "v1.2.3")
4. **Code Snippet Removal**: Remove backtick-enclosed code
5. **CVE ID Removal**: Remove CVE references
6. **Special Character Removal**: Keep only alphanumeric + spaces
7. **Stop Word Removal**: Remove common English words + custom CVE terms
8. **Token Filtering**: Remove very short (<3 chars) and very long (>20 chars) tokens

**Custom stop words added**: `cve, vulnerability, allows, could, may, via, using, remote, attackers, attacker, user, users`

**Result**: ~41% reduction in average word count while preserving semantic content.

## Embedding Models

### Sentence Transformers (Local)

- **Model**: `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Advantages**: Free, runs locally, GPU-accelerated
- **Speed**: ~2-3 hours for 330K descriptions on Apple M-series
- **Quality**: Good performance for semantic similarity

### OpenAI API

- **Model**: `text-embedding-3-small`
- **Dimension**: 1536
- **Advantages**: State-of-the-art quality
- **Cost**: ~$0.02 per 1M tokens (est. $5-10 for full dataset)
- **Speed**: Faster if you have good internet, limited by rate limits

## Model Training Strategy

### Temporal Train/Test Split

- **Training**: 2015-2024 data (~317K CVEs)
- **Testing**: 2025 data (~13K CVEs)
- **Rationale**: Simulates real-world deployment where we predict future exploits

### Class Imbalance Handling

With only ~0.6% of CVEs being exploited:

- Use `class_weight='balanced'` in both models
- Stratified splitting (though temporal split overrides this)
- Evaluate with ROC-AUC and Average Precision (not accuracy)
- Focus on Precision-Recall curves for imbalanced evaluation

### Models Trained

**Logistic Regression**
- Simple, interpretable, fast
- Good baseline for high-dimensional embeddings
- `max_iter=1000`, `class_weight='balanced'`

**Random Forest**
- Handles non-linear patterns
- Ensemble of 100 trees
- `max_depth=20`, `min_samples_split=10`
- `class_weight='balanced'`

## Evaluation Metrics

The pipeline reports:

1. **ROC-AUC**: Area under ROC curve (discrimination ability)
2. **Average Precision**: Area under PR curve (precision-recall trade-off)
3. **F1 Score**: Harmonic mean of precision and recall
4. **Confusion Matrix**: Detailed breakdown of predictions
5. **Classification Report**: Precision, recall, F1 per class

## Visualizations

Each run generates four visualizations:

1. **ROC Curves**: Shows true positive vs false positive rate
2. **Precision-Recall Curves**: Critical for imbalanced data
3. **Confusion Matrices**: Actual vs predicted breakdown
4. **Model Comparison**: Bar chart of all metrics side-by-side

## Expected Performance

Based on the architecture and data characteristics:

- **ROC-AUC**: Expected 0.65-0.80 (embeddings capture semantic risk indicators)
- **Average Precision**: Expected 0.05-0.15 (baseline is ~0.004 given 0.4% prevalence)
- **F1 Score**: Variable depending on threshold tuning

Performance depends heavily on:
- Quality of vulnerability descriptions
- Embedding model's domain knowledge
- Temporal distribution shift (2025 may have different patterns)

## Troubleshooting

### Out of Memory (Embeddings)

Reduce batch size in the embedding scripts:
```python
BATCH_SIZE = 128  # or 64
```

### Slow Embedding Generation

- **Sentence Transformers**: Ensure MPS/CUDA is available, reduce batch size
- **OpenAI API**: Increase `RATE_LIMIT_DELAY` if hitting rate limits

### Missing .env File

For OpenAI version, ensure `.env` exists with:
```
OPENAI_API_KEY=sk-...your_key_here
```

### GPU Not Detected

The script will automatically fall back to CPU, but it will be slower. Check:
```python
import torch
print(torch.backends.mps.is_available())  # For Apple Silicon
print(torch.cuda.is_available())          # For NVIDIA GPUs
```

## Extending the Pipeline

### Add New Embedding Models

Create a new script `3c_generate_embeddings_<model>.py`:

```python
# Use any HuggingFace model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('your-model-name')
embeddings = model.encode(descriptions)
```

### Add New Classifiers

In `4_train_models.py`, add your model:

```python
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True)
svm_model.fit(X_train, y_train)
```

### Different Train/Test Splits

Modify the split logic in `4_train_models.py`:

```python
# Example: 80/20 random split instead of temporal
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, metadata['target'], test_size=0.2,
    stratify=metadata['target'], random_state=42
)
```

## Comparison with CVSS-Based Approach

| Aspect | CVSS Features (baseline_model.py) | Embeddings (this pipeline) |
|--------|-----------------------------------|----------------------------|
| **Input** | Structured scores (severity, impact, etc.) | Unstructured text descriptions |
| **Feature Dim** | ~40 features after one-hot encoding | 384 or 1536 dimensions |
| **Interpretability** | High (feature importances) | Low (black box embeddings) |
| **Domain Knowledge** | Explicit (CVSS metrics) | Implicit (learned from text) |
| **Expected ROC-AUC** | ~0.87 | ~0.65-0.80 |
| **Data Requirements** | CVSS scores from NVD | Vulnerability descriptions |
| **Computation** | Fast (simple features) | Slower (embedding generation) |

Both approaches are complementary and could be combined in an ensemble model.

## Future Improvements

1. **Hybrid Model**: Combine CVSS features + embeddings
2. **Fine-tuning**: Fine-tune sentence transformer on CVE-exploit pairs
3. **Attention Mechanisms**: Use transformer models directly (BERT, RoBERTa)
4. **Multi-modal**: Include code snippets, patches, and metadata
5. **Active Learning**: Prioritize uncertain predictions for manual review
6. **Temporal Cross-Validation**: Multiple year-based folds

## Citation

If you use this pipeline in research, please cite:

```
@misc{systemic_cyber_risks,
  title={Embedding-Based Vulnerability Exploitation Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/SystemicCyberRisks}
}
```

## License

Same as the parent repository.

## Contact

For questions or issues with this pipeline, please open a GitHub issue.
