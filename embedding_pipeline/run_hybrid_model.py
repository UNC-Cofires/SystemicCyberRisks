#!/usr/bin/env python3
"""
Hybrid Model: CVSS Features + Description Embeddings

This script combines two complementary approaches for predicting vulnerability exploitation:
1. CVSS Features: Structured vulnerability metrics (severity, impact, attack vectors)
2. Description Embeddings: Semantic text features from CVE descriptions
"""

# Standard libraries
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Embeddings
from sentence_transformers import SentenceTransformer
import torch

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All imports successful!")
print(f"\n🖥️  Device: {'MPS (Apple Silicon)' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU'}")

# =============================================================================
# 2. Load and Cache Sentence Transformer Model
# =============================================================================

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_CACHE_DIR = 'embedding_pipeline/models/sentence_transformer_cache'
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def load_or_download_model(model_name=MODEL_NAME, cache_dir=MODEL_CACHE_DIR, device=DEVICE):
    """
    Load sentence transformer model from local cache or download if not present.

    Returns:
        SentenceTransformer: Loaded model ready for encoding
    """
    cache_path = Path(cache_dir)

    if cache_path.exists() and len(list(cache_path.iterdir())) > 0:
        print(f"📂 Loading model from local cache: {cache_dir}")
        model = SentenceTransformer(str(cache_path), device=device)
        print(f"   ✅ Model loaded from cache")
    else:
        print(f"📥 Downloading model: {model_name}")
        print(f"   This may take a few minutes on first run...")
        model = SentenceTransformer(model_name, device=device)

        # Save to local cache
        print(f"💾 Saving model to local cache: {cache_dir}")
        cache_path.mkdir(parents=True, exist_ok=True)
        model.save(str(cache_path))
        print(f"   ✅ Model cached for future use")

    return model

# Load model
embedding_model = load_or_download_model()
embedding_dim = embedding_model.get_sentence_embedding_dimension()

print(f"\n📊 Model info:")
print(f"   Name: {MODEL_NAME}")
print(f"   Embedding dimension: {embedding_dim}")
print(f"   Max sequence length: {embedding_model.max_seq_length}")
print(f"   Device: {DEVICE}")

# =============================================================================
# 3. Load Data
# =============================================================================

print("="*70)
print("LOADING DATA")
print("="*70)

# 1. Load CVSS features (cleaned dataset from baseline model)
print("\n📂 Loading CVSS features...")
if os.path.exists('data/data.csv'):
    cvss_df = pd.read_csv('data/data.csv')
    print(f"   Loaded from: data/data.csv")
else:
    # Fall back to processed vulnerabilities
    cvss_df = pd.read_csv('data/processed_vulnerabilities.csv')
    print(f"   Loaded from: data/processed_vulnerabilities.csv")

print(f"   Shape: {cvss_df.shape}")
print(f"   Columns: {list(cvss_df.columns[:10])}...")

# 2. Load pre-generated embeddings
print("\n📂 Loading pre-generated embeddings...")
embedding_data = np.load('embedding_pipeline/data/embeddings_sentencetransformer.npz')
embeddings = embedding_data['embeddings']
embedding_metadata = pd.read_csv('embedding_pipeline/data/metadata_sentencetransformer.csv')

print(f"   Embeddings shape: {embeddings.shape}")
print(f"   Metadata shape: {embedding_metadata.shape}")

# 3. Load target labels
print("\n📂 Loading KEV catalog for targets...")
kev_df = pd.read_csv('data/known_exploited_vulnerabilities.csv')
kev_set = set(kev_df['cveID'].values)
print(f"   KEV catalog size: {len(kev_set):,} known exploited CVEs")

print("\n✅ All data loaded successfully!")

# =============================================================================
# 4. Align and Merge Datasets
# =============================================================================

print("="*70)
print("ALIGNING DATASETS BY CVE ID")
print("="*70)

# Add embeddings to metadata
print("\n🔗 Merging embeddings with metadata...")
embedding_metadata['embedding_idx'] = range(len(embedding_metadata))

# Ensure CVSS data has target column (if not, create it)
if 'target' not in cvss_df.columns:
    print("\n🎯 Creating target variable for CVSS data...")
    cvss_df['target'] = cvss_df['id'].apply(lambda x: 1 if x in kev_set else 0)
    print(f"   Exploited CVEs in CVSS data: {cvss_df['target'].sum():,}")

# Find common CVE IDs between CVSS and embedding datasets
print("\n🔍 Finding common CVEs...")
cvss_ids = set(cvss_df['id'].values)
embedding_ids = set(embedding_metadata['id'].values)
common_ids = cvss_ids.intersection(embedding_ids)

print(f"   CVSS dataset: {len(cvss_ids):,} CVEs")
print(f"   Embedding dataset: {len(embedding_ids):,} CVEs")
print(f"   Common CVEs: {len(common_ids):,} ({len(common_ids)/min(len(cvss_ids), len(embedding_ids))*100:.1f}% overlap)")

# Filter to common IDs
cvss_df_filtered = cvss_df[cvss_df['id'].isin(common_ids)].copy()
embedding_metadata_filtered = embedding_metadata[embedding_metadata['id'].isin(common_ids)].copy()

# Check for and remove duplicates in embedding metadata
if embedding_metadata_filtered['id'].duplicated().any():
    print(f"\n⚠️  Found {embedding_metadata_filtered['id'].duplicated().sum()} duplicate IDs in embedding metadata")
    print(f"   Keeping first occurrence of each ID...")
    embedding_metadata_filtered = embedding_metadata_filtered.drop_duplicates(subset='id', keep='first')

# Sort both by ID to ensure alignment
cvss_df_filtered = cvss_df_filtered.sort_values('id').reset_index(drop=True)
embedding_metadata_filtered = embedding_metadata_filtered.sort_values('id').reset_index(drop=True)

# Verify alignment
if len(cvss_df_filtered) != len(embedding_metadata_filtered):
    print(f"\n❌ ERROR: Size mismatch after filtering!")
    print(f"   CVSS filtered: {len(cvss_df_filtered)}")
    print(f"   Embedding filtered: {len(embedding_metadata_filtered)}")
    raise ValueError("Dataset sizes do not match after filtering")

assert (cvss_df_filtered['id'].values == embedding_metadata_filtered['id'].values).all(), "IDs not aligned!"
print("\n✅ Datasets aligned successfully!")

# Extract aligned embeddings
aligned_embedding_indices = embedding_metadata_filtered['embedding_idx'].values
aligned_embeddings = embeddings[aligned_embedding_indices]

print(f"\n📊 Final aligned dataset:")
print(f"   Total CVEs: {len(cvss_df_filtered):,}")
print(f"   CVSS features shape: {cvss_df_filtered.shape}")
print(f"   Embeddings shape: {aligned_embeddings.shape}")
print(f"   Exploited CVEs: {cvss_df_filtered['target'].sum():,} ({cvss_df_filtered['target'].mean()*100:.2f}%)")

# =============================================================================
# 5. Prepare Feature Sets
# =============================================================================

print("="*70)
print("PREPARING FEATURE SETS")
print("="*70)

# Extract year for temporal split (2015-2024 train, 2025 test)
if 'year' not in cvss_df_filtered.columns:
    cvss_df_filtered['year'] = cvss_df_filtered['id'].str.extract(r'CVE-(\d{4})-')[0].astype(int)

# 1. CVSS Features
print("\n🔧 Preparing CVSS features...")

# Define CVSS feature columns (exclude id, target, year, description)
exclude_cols = ['id', 'target', 'year', 'description', 'embedding_idx']
cvss_feature_cols = [col for col in cvss_df_filtered.columns if col not in exclude_cols]

# Handle categorical columns with one-hot encoding if needed
categorical_cols = cvss_df_filtered[cvss_feature_cols].select_dtypes(include=['object']).columns.tolist()

if len(categorical_cols) > 0:
    print(f"   Encoding {len(categorical_cols)} categorical columns...")
    cvss_df_encoded = cvss_df_filtered[cvss_feature_cols].copy()
    cvss_df_encoded[categorical_cols] = cvss_df_encoded[categorical_cols].fillna('MISSING').astype(str)

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(cvss_df_encoded[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    # Combine numeric and encoded categorical
    numeric_cols = [col for col in cvss_feature_cols if col not in categorical_cols]
    cvss_numeric = cvss_df_filtered[numeric_cols].fillna(0).values
    cvss_features = np.hstack([cvss_numeric, encoded_array])

    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Encoded categorical features: {len(encoded_cols)}")
else:
    cvss_features = cvss_df_filtered[cvss_feature_cols].fillna(0).values

print(f"   CVSS features shape: {cvss_features.shape}")

# 2. Embedding Features (already prepared)
print(f"\n🔧 Embedding features shape: {aligned_embeddings.shape}")

# 3. Hybrid Features (concatenate)
print("\n🔧 Creating hybrid features...")
hybrid_features = np.hstack([cvss_features, aligned_embeddings])
print(f"   Hybrid features shape: {hybrid_features.shape}")
print(f"   = {cvss_features.shape[1]} CVSS features + {aligned_embeddings.shape[1]} embedding features")

# Target and metadata
y = cvss_df_filtered['target'].values
years = cvss_df_filtered['year'].values
cve_ids = cvss_df_filtered['id'].values

print(f"\n📊 Feature summary:")
print(f"   Total samples: {len(y):,}")
print(f"   CVSS-only dimensions: {cvss_features.shape[1]}")
print(f"   Embedding-only dimensions: {aligned_embeddings.shape[1]}")
print(f"   Hybrid dimensions: {hybrid_features.shape[1]}")
print(f"   Target distribution: {y.sum():,} exploited ({y.mean()*100:.2f}%)")

print("\n✅ All feature sets prepared!")

# =============================================================================
# 6. Train/Test Split (Temporal)
# =============================================================================

print("="*70)
print("TEMPORAL TRAIN/TEST SPLIT")
print("="*70)

# Create temporal split
train_mask = years < 2025
test_mask = years == 2025

print(f"\n📊 Split configuration:")
print(f"   Training: 2015-2024")
print(f"   Testing: 2025")

# Split all feature sets
X_cvss_train, X_cvss_test = cvss_features[train_mask], cvss_features[test_mask]
X_emb_train, X_emb_test = aligned_embeddings[train_mask], aligned_embeddings[test_mask]
X_hybrid_train, X_hybrid_test = hybrid_features[train_mask], hybrid_features[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\n📊 Dataset sizes:")
print(f"   Training samples: {len(y_train):,}")
print(f"   Test samples: {len(y_test):,}")
print(f"   Training exploited: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   Test exploited: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# Year distribution
print(f"\n📊 Year distribution:")
year_df = pd.DataFrame({'year': years, 'target': y})
for year in sorted(year_df['year'].unique()):
    year_data = year_df[year_df['year'] == year]
    split_type = 'TRAIN' if year < 2025 else 'TEST '
    print(f"   [{split_type}] {year}: {len(year_data):,} CVEs ({year_data['target'].sum()} exploited)")

print("\n✅ Train/test split complete!")

# =============================================================================
# 7. Train Models
# =============================================================================

print("="*70)
print("TRAINING MODELS")
print("="*70)

# Shared configuration
model_config = {
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}

# Dictionary to store models and results
models = {}
results = {}

# 1. CVSS-only Model
print("\n" + "="*70)
print("MODEL 1: CVSS-ONLY")
print("="*70)
print("\n🏋️  Training...")
model_cvss = LogisticRegression(**model_config)
model_cvss.fit(X_cvss_train, y_train)
models['CVSS-only'] = model_cvss
print("   ✅ Training complete")

# Predictions
y_pred_cvss = model_cvss.predict(X_cvss_test)
y_proba_cvss = model_cvss.predict_proba(X_cvss_test)[:, 1]

# Metrics
results['CVSS-only'] = {
    'roc_auc': roc_auc_score(y_test, y_proba_cvss),
    'avg_precision': average_precision_score(y_test, y_proba_cvss),
    'f1': f1_score(y_test, y_pred_cvss),
    'y_pred': y_pred_cvss,
    'y_proba': y_proba_cvss
}

print(f"\n📊 Results:")
print(f"   ROC-AUC: {results['CVSS-only']['roc_auc']:.4f}")
print(f"   Average Precision: {results['CVSS-only']['avg_precision']:.4f}")
print(f"   F1 Score: {results['CVSS-only']['f1']:.4f}")

# 2. Embedding-only Model
print("\n" + "="*70)
print("MODEL 2: EMBEDDING-ONLY")
print("="*70)
print("\n🏋️  Training...")
model_emb = LogisticRegression(**model_config)
model_emb.fit(X_emb_train, y_train)
models['Embedding-only'] = model_emb
print("   ✅ Training complete")

# Predictions
y_pred_emb = model_emb.predict(X_emb_test)
y_proba_emb = model_emb.predict_proba(X_emb_test)[:, 1]

# Metrics
results['Embedding-only'] = {
    'roc_auc': roc_auc_score(y_test, y_proba_emb),
    'avg_precision': average_precision_score(y_test, y_proba_emb),
    'f1': f1_score(y_test, y_pred_emb),
    'y_pred': y_pred_emb,
    'y_proba': y_proba_emb
}

print(f"\n📊 Results:")
print(f"   ROC-AUC: {results['Embedding-only']['roc_auc']:.4f}")
print(f"   Average Precision: {results['Embedding-only']['avg_precision']:.4f}")
print(f"   F1 Score: {results['Embedding-only']['f1']:.4f}")

# 3. Hybrid Model
print("\n" + "="*70)
print("MODEL 3: HYBRID (CVSS + EMBEDDINGS)")
print("="*70)
print("\n🏋️  Training...")
model_hybrid = LogisticRegression(**model_config)
model_hybrid.fit(X_hybrid_train, y_train)
models['Hybrid'] = model_hybrid
print("   ✅ Training complete")

# Predictions
y_pred_hybrid = model_hybrid.predict(X_hybrid_test)
y_proba_hybrid = model_hybrid.predict_proba(X_hybrid_test)[:, 1]

# Metrics
results['Hybrid'] = {
    'roc_auc': roc_auc_score(y_test, y_proba_hybrid),
    'avg_precision': average_precision_score(y_test, y_proba_hybrid),
    'f1': f1_score(y_test, y_pred_hybrid),
    'y_pred': y_pred_hybrid,
    'y_proba': y_proba_hybrid
}

print(f"\n📊 Results:")
print(f"   ROC-AUC: {results['Hybrid']['roc_auc']:.4f}")
print(f"   Average Precision: {results['Hybrid']['avg_precision']:.4f}")
print(f"   F1 Score: {results['Hybrid']['f1']:.4f}")

print("\n✅ All models trained!")

# =============================================================================
# 8. Model Comparison
# =============================================================================

print("="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()],
    'Avg Precision': [results[m]['avg_precision'] for m in results.keys()],
    'F1 Score': [results[m]['f1'] for m in results.keys()]
})

# Sort by ROC-AUC
comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)

print("\n📊 Performance Metrics:")
print(comparison_df.to_string(index=False))

# Calculate improvements
print("\n📈 Hybrid Model Improvements:")
hybrid_roc = results['Hybrid']['roc_auc']
cvss_roc = results['CVSS-only']['roc_auc']
emb_roc = results['Embedding-only']['roc_auc']

print(f"   vs CVSS-only: {(hybrid_roc - cvss_roc)*100:+.2f} percentage points")
print(f"   vs Embedding-only: {(hybrid_roc - emb_roc)*100:+.2f} percentage points")

if hybrid_roc > max(cvss_roc, emb_roc):
    print("\n🏆 Hybrid model OUTPERFORMS both individual approaches!")
elif hybrid_roc > cvss_roc:
    print("\n✅ Hybrid model improves over CVSS-only")
elif hybrid_roc > emb_roc:
    print("\n✅ Hybrid model improves over Embedding-only")
else:
    print("\n⚠️  Hybrid model does not improve over individual approaches")

# Determine best model
best_model = comparison_df.iloc[0]['Model']
best_roc = comparison_df.iloc[0]['ROC-AUC']
print(f"\n🥇 BEST MODEL: {best_model} (ROC-AUC: {best_roc:.4f})")

# =============================================================================
# 9. Visualizations
# =============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Create visualizations directory
vis_dir = Path('embedding_pipeline/visualizations')
vis_dir.mkdir(parents=True, exist_ok=True)

# ROC Curves
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors = {'CVSS-only': 'blue', 'Embedding-only': 'green', 'Hybrid': 'red'}

for model_name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_proba'])
    roc_auc = results[model_name]['roc_auc']
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
            linewidth=2.5, color=colors[model_name])

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves: Model Comparison', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = vis_dir / 'hybrid_roc_curves.png'
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {roc_path}")
plt.close()

# Precision-Recall Curves
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for model_name in results.keys():
    precision, recall, _ = precision_recall_curve(y_test, results[model_name]['y_proba'])
    avg_prec = results[model_name]['avg_precision']
    ax.plot(recall, precision, label=f'{model_name} (AP = {avg_prec:.3f})',
            linewidth=2.5, color=colors[model_name])

ax.axhline(y=y_test.mean(), color='k', linestyle='--',
           label=f'Baseline (prevalence = {y_test.mean():.3f})', linewidth=1)
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curves: Model Comparison', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = vis_dir / 'hybrid_precision_recall_curves.png'
plt.savefig(pr_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {pr_path}")
plt.close()

# Performance Bar Chart
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

x = np.arange(len(results))
width = 0.25

roc_scores = [results[m]['roc_auc'] for m in results.keys()]
ap_scores = [results[m]['avg_precision'] for m in results.keys()]
f1_scores = [results[m]['f1'] for m in results.keys()]

ax.bar(x - width, roc_scores, width, label='ROC-AUC', color='steelblue')
ax.bar(x, ap_scores, width, label='Avg Precision', color='coral')
ax.bar(x + width, f1_scores, width, label='F1 Score', color='mediumseagreen')

ax.set_xlabel('Models', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Performance Comparison Across Metrics', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results.keys())
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 1)

# Add value labels
for i, (roc, ap, f1) in enumerate(zip(roc_scores, ap_scores, f1_scores)):
    ax.text(i - width, roc + 0.02, f'{roc:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(i, ap + 0.02, f'{ap:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
bar_path = vis_dir / 'hybrid_performance_comparison.png'
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {bar_path}")
plt.close()

print("\n✅ All visualizations saved!")

# =============================================================================
# 10. Save Results and Models
# =============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create output directory
output_dir = Path('embedding_pipeline/hybrid_results')
output_dir.mkdir(parents=True, exist_ok=True)

# Save results JSON
results_dict = {
    'models': {
        model_name: {
            'roc_auc': float(metrics['roc_auc']),
            'average_precision': float(metrics['avg_precision']),
            'f1_score': float(metrics['f1'])
        }
        for model_name, metrics in results.items()
    },
    'best_model': best_model,
    'best_roc_auc': float(best_roc),
    'dataset': {
        'train_size': int(len(y_train)),
        'test_size': int(len(y_test)),
        'train_exploited': int(y_train.sum()),
        'test_exploited': int(y_test.sum())
    }
}

results_path = output_dir / 'hybrid_model_results.json'
with open(results_path, 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"\n💾 Results saved: {results_path}")

# Save comparison table
comparison_path = output_dir / 'model_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f"💾 Comparison table saved: {comparison_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'id': cve_ids[test_mask],
    'year': years[test_mask],
    'true_label': y_test,
    'cvss_prediction': y_pred_cvss,
    'cvss_probability': y_proba_cvss,
    'embedding_prediction': y_pred_emb,
    'embedding_probability': y_proba_emb,
    'hybrid_prediction': y_pred_hybrid,
    'hybrid_probability': y_proba_hybrid
})

predictions_path = output_dir / 'predictions_all_models.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"💾 Predictions saved: {predictions_path}")

# Save models
models_dir = output_dir / 'models'
models_dir.mkdir(exist_ok=True)

for model_name, model in models.items():
    model_path = models_dir / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Model saved: {model_path}")

print("\n✅ All results and models saved!")
print(f"\n📁 Output directory: {output_dir}")

# =============================================================================
# 11. Final Summary
# =============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\n🎯 Objective: Combine CVSS features + Description embeddings")
print(f"   to maximize vulnerability exploitation prediction performance")

print(f"\n📊 Dataset:")
print(f"   Total CVEs: {len(y):,}")
print(f"   Training (2015-2024): {len(y_train):,} CVEs")
print(f"   Testing (2025): {len(y_test):,} CVEs")
print(f"   Class imbalance: {(1 - y_test.mean())/y_test.mean():.1f}:1 ratio")

print(f"\n🏆 RESULTS:")
print(comparison_df.to_string(index=False))

print(f"\n💡 KEY FINDINGS:")

if hybrid_roc > max(cvss_roc, emb_roc):
    improvement = (hybrid_roc - max(cvss_roc, emb_roc)) * 100
    print(f"   ✅ Hybrid model achieves {hybrid_roc:.4f} ROC-AUC")
    print(f"   ✅ Improves over best individual model by {improvement:.2f} percentage points")
    print(f"   ✅ Demonstrates that CVSS + embeddings provide complementary signals")
else:
    print(f"   ⚠️  Hybrid model ({hybrid_roc:.4f}) does not exceed individual models")
    print(f"   ⚠️  Best single approach: {best_model} ({best_roc:.4f})")
    print(f"   💡 Consider: feature selection, different combination strategies")

if hybrid_roc >= 0.90:
    print(f"\n🎉 MILESTONE ACHIEVED: ROC-AUC ≥ 0.90!")
elif hybrid_roc >= 0.88:
    print(f"\n✨ EXCELLENT PERFORMANCE: ROC-AUC {hybrid_roc:.4f}")
else:
    print(f"\n📈 GOOD PERFORMANCE: ROC-AUC {hybrid_roc:.4f}")

print(f"\n📁 All results saved to: {output_dir}")
print(f"\n🚀 Next steps:")
print(f"   - Analyze predictions on high-risk vulnerabilities")
print(f"   - Deploy best model for operational use")
print(f"   - Retrain periodically with new data")

print("\n" + "="*70)
print("✨ HYBRID MODEL ANALYSIS COMPLETE ✨")
print("="*70)
