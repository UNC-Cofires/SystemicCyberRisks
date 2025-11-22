#!/usr/bin/env python3
"""
Final Hybrid Model: OpenAI Embeddings + CVSS Features
======================================================
Combines OpenAI text embeddings (1536-dim) with CVSS features (38-dim)
using Logistic Regression with temporal train/test split.

This script provides a production-ready hybrid model that leverages both:
1. Structured CVSS vulnerability metrics
2. Semantic information from OpenAI embeddings

Usage:
    python embedding_pipeline/final_hybrid.py

Output:
    - Trained model saved as .pkl
    - ROC curve visualization
    - Confusion matrix visualization
    - JSON results with detailed metrics
    - CSV with predictions and probabilities
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

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("FINAL HYBRID MODEL: OpenAI Embeddings + CVSS Features")
print("=" * 80)
print("\n✅ All imports successful!")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG = {
    'cvss_data_path': 'data/data.csv',
    'openai_embeddings_path': 'embedding_pipeline/data/embeddings_openai.npz',
    'openai_metadata_path': 'embedding_pipeline/data/metadata_openai.csv',
    'kev_catalog_path': 'data/known_exploited_vulnerabilities.csv',
    'output_dir': 'embedding_pipeline/final_hybrid_results',
    'visualizations_dir': 'embedding_pipeline/final_hybrid_results/visualizations',
    'train_years': (2015, 2023),  # Inclusive
    'test_years': (2024, 2025),   # Inclusive
    'random_state': 42
}

# Create output directories
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['visualizations_dir'], exist_ok=True)

print(f"\n📁 Configuration:")
print(f"   Output directory: {CONFIG['output_dir']}")
print(f"   Training years: {CONFIG['train_years'][0]}-{CONFIG['train_years'][1]}")
print(f"   Testing years: {CONFIG['test_years'][0]}-{CONFIG['test_years'][1]}")

# =============================================================================
# 2. LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load CVSS features
print(f"\n📂 Loading CVSS features from: {CONFIG['cvss_data_path']}")
cvss_df = pd.read_csv(CONFIG['cvss_data_path'])
print(f"   Loaded {len(cvss_df):,} CVEs with {cvss_df.shape[1]} columns")

# Load OpenAI embeddings
print(f"\n📂 Loading OpenAI embeddings from: {CONFIG['openai_embeddings_path']}")
embeddings_data = np.load(CONFIG['openai_embeddings_path'])
embeddings = embeddings_data['embeddings']
print(f"   Embeddings shape: {embeddings.shape}")

# Load embedding metadata
print(f"\n📂 Loading embedding metadata from: {CONFIG['openai_metadata_path']}")
embedding_metadata = pd.read_csv(CONFIG['openai_metadata_path'])
print(f"   Metadata shape: {embedding_metadata.shape}")

# Load KEV catalog for target labels
print(f"\n📂 Loading KEV catalog from: {CONFIG['kev_catalog_path']}")
kev_df = pd.read_csv(CONFIG['kev_catalog_path'])
kev_cves = set(kev_df['cveID'].str.upper().str.strip())
print(f"   KEV catalog size: {len(kev_cves):,} known exploited CVEs")

print("\n✅ All data loaded successfully!")

# =============================================================================
# 3. ALIGN DATASETS BY CVE ID
# =============================================================================

print("\n" + "=" * 80)
print("ALIGNING DATASETS")
print("=" * 80)

# Merge embedding metadata with embeddings
print("\n🔗 Merging embeddings with metadata...")
embedding_df = embedding_metadata.copy()
embedding_df['original_idx'] = range(len(embedding_df))

# Handle duplicate IDs in embedding metadata (keep first occurrence)
if embedding_df['id'].duplicated().any():
    n_dupes = embedding_df['id'].duplicated().sum()
    print(f"   ⚠️  Found {n_dupes:,} duplicate IDs in embedding metadata")
    print(f"   Keeping first occurrence of each ID...")
    # Get the original indices we want to keep
    keep_indices = embedding_df['original_idx'][~embedding_df['id'].duplicated()].values
    # Filter both metadata and embeddings
    embedding_df = embedding_df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    embeddings = embeddings[keep_indices]
    
# Add new sequential index after deduplication
embedding_df['embedding_idx'] = range(len(embedding_df))

# Standardize CVE IDs
cvss_df['id'] = cvss_df['id'].str.upper().str.strip()
embedding_df['id'] = embedding_df['id'].str.upper().str.strip()

# Find common CVEs
print("\n🔍 Finding common CVEs...")
cvss_cves = set(cvss_df['id'])
embedding_cves = set(embedding_df['id'])
common_cves = cvss_cves.intersection(embedding_cves)

print(f"   CVSS dataset: {len(cvss_cves):,} CVEs")
print(f"   Embedding dataset: {len(embedding_cves):,} CVEs")
print(f"   Common CVEs: {len(common_cves):,} ({len(common_cves)/len(cvss_cves)*100:.1f}% overlap)")

# Filter to common CVEs
cvss_df = cvss_df[cvss_df['id'].isin(common_cves)].copy()
embedding_df = embedding_df[embedding_df['id'].isin(common_cves)].copy()

# Sort both by ID to ensure alignment
cvss_df = cvss_df.sort_values('id').reset_index(drop=True)
embedding_df = embedding_df.sort_values('id').reset_index(drop=True)

# Verify alignment
assert (cvss_df['id'].values == embedding_df['id'].values).all(), "CVE IDs not aligned!"

# Get embeddings for aligned CVEs
embedding_indices = embedding_df['embedding_idx'].values
embeddings_aligned = embeddings[embedding_indices]

# Create target variable from KEV catalog
cvss_df['target'] = cvss_df['id'].apply(lambda x: 1 if x in kev_cves else 0)

print("\n✅ Datasets aligned successfully!")
print(f"\n📊 Final aligned dataset:")
print(f"   Total CVEs: {len(cvss_df):,}")
print(f"   Exploited CVEs: {cvss_df['target'].sum():,} ({cvss_df['target'].mean()*100:.2f}%)")

# =============================================================================
# 4. PREPARE FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("PREPARING FEATURES")
print("=" * 80)

# Prepare CVSS features
print("\n🔧 Preparing CVSS features...")

# Identify numeric and categorical columns
# Exclude 'id', 'year', and 'target' from features
feature_cols = [col for col in cvss_df.columns if col not in ['id', 'year', 'target']]

# Separate numeric and categorical features
numeric_cols = cvss_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = cvss_df[feature_cols].select_dtypes(include=['object']).columns.tolist()

print(f"   Numeric features: {len(numeric_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")

# Create CVSS feature matrix
X_cvss = cvss_df[feature_cols].copy()

# Encode categorical features
if categorical_cols:
    print(f"   Encoding {len(categorical_cols)} categorical columns...")
    # One-hot encode categorical features (drop first to avoid multicollinearity)
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_categorical = encoder.fit_transform(X_cvss[categorical_cols])
    
    # Get feature names
    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        categories = encoder.categories_[i][1:]  # Skip first (dropped)
        cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    # Combine with numeric features
    X_cvss_numeric = X_cvss[numeric_cols].values
    X_cvss_features = np.hstack([X_cvss_numeric, X_categorical])
    
    cvss_feature_names = numeric_cols + cat_feature_names
else:
    X_cvss_features = X_cvss[numeric_cols].values
    cvss_feature_names = numeric_cols

print(f"   CVSS features shape: {X_cvss_features.shape}")

# Prepare OpenAI embedding features
print("\n🔧 Preparing OpenAI embedding features...")
X_embeddings = embeddings_aligned
print(f"   Embedding features shape: {X_embeddings.shape}")

# Create hybrid features (concatenate CVSS + embeddings)
print("\n🔧 Creating hybrid features...")
X_hybrid = np.hstack([X_cvss_features, X_embeddings])
print(f"   Hybrid features shape: {X_hybrid.shape}")
print(f"   = {X_cvss_features.shape[1]} CVSS features + {X_embeddings.shape[1]} embedding features")

# Target variable
y = cvss_df['target'].values

# Extract year and CVE ID for splitting
years = cvss_df['year'].values if 'year' in cvss_df.columns else embedding_df['year'].values
cve_ids = cvss_df['id'].values

print("\n📊 Feature summary:")
print(f"   Total samples: {len(X_hybrid):,}")
print(f"   CVSS dimensions: {X_cvss_features.shape[1]}")
print(f"   Embedding dimensions: {X_embeddings.shape[1]}")
print(f"   Hybrid dimensions: {X_hybrid.shape[1]}")
print(f"   Target distribution: {y.sum():,} exploited ({y.mean()*100:.2f}%)")

print("\n✅ All features prepared!")

# =============================================================================
# 5. TEMPORAL TRAIN/TEST SPLIT
# =============================================================================

print("\n" + "=" * 80)
print("TEMPORAL TRAIN/TEST SPLIT")
print("=" * 80)

# Create train/test masks based on year
train_mask = (years >= CONFIG['train_years'][0]) & (years <= CONFIG['train_years'][1])
test_mask = (years >= CONFIG['test_years'][0]) & (years <= CONFIG['test_years'][1])

# Split data
X_train = X_hybrid[train_mask]
y_train = y[train_mask]
X_test = X_hybrid[test_mask]
y_test = y[test_mask]
cve_ids_test = cve_ids[test_mask]
years_test = years[test_mask]

print(f"\n📊 Split configuration:")
print(f"   Training: {CONFIG['train_years'][0]}-{CONFIG['train_years'][1]}")
print(f"   Testing: {CONFIG['test_years'][0]}-{CONFIG['test_years'][1]}")

print(f"\n📊 Dataset sizes:")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Training exploited: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   Test exploited: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# Show year distribution
print(f"\n📊 Year distribution:")
unique_years = np.unique(years)
for year in unique_years:
    year_mask = years == year
    n_samples = year_mask.sum()
    n_exploited = y[year_mask].sum()
    split_type = "TRAIN" if (year >= CONFIG['train_years'][0] and year <= CONFIG['train_years'][1]) else "TEST"
    print(f"   [{split_type:5}] {year}: {n_samples:,} CVEs ({n_exploited:,} exploited)")

print("\n✅ Train/test split complete!")

# =============================================================================
# 6. TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 80)

print("\n🏋️  Training Logistic Regression...")
print("   Configuration:")
print("   - class_weight='balanced' (handles class imbalance)")
print("   - max_iter=1000")
print(f"   - random_state={CONFIG['random_state']}")

# Train model
# Note: Using n_jobs=1 to avoid Python 3.13 multiprocessing warnings
# These warnings are harmless but can be distracting
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=CONFIG['random_state'],
    n_jobs=1,  # Set to 1 to avoid multiprocessing warnings in Python 3.13
    verbose=0
)

model.fit(X_train, y_train)
print("\n   ✅ Training complete!")

# =============================================================================
# 7. GENERATE PREDICTIONS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

print("\n🔮 Generating predictions on test set...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"   Predictions generated for {len(y_pred):,} test samples")

# =============================================================================
# 8. EVALUATE MODEL
# =============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print("\n📊 Performance Metrics:")
print(f"   ROC-AUC: {roc_auc:.4f}")
print(f"   Average Precision: {avg_precision:.4f}")
print(f"   F1 Score: {f1:.4f}")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Exploited', 'Exploited'], digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Confusion Matrix Breakdown:")
print(f"   True Negatives (TN):  {tn:,} - Correctly predicted NOT exploited")
print(f"   False Positives (FP): {fp:,} - Incorrectly predicted AS exploited")
print(f"   False Negatives (FN): {fn:,} - Incorrectly predicted AS NOT exploited")
print(f"   True Positives (TP):  {tp:,} - Correctly predicted AS exploited")

# Additional metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n📈 Additional Metrics:")
print(f"   Sensitivity (Recall): {sensitivity:.2%}")
print(f"   Specificity: {specificity:.2%}")
print(f"   Precision: {precision:.2%}")

# =============================================================================
# 9. GENERATE VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. ROC Curve
print("\n📊 Creating ROC curve...")
fig, ax = plt.subplots(figsize=(10, 8))

fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, label=f'Hybrid Model (AUC = {roc_auc:.3f})', 
        linewidth=3, color='darkblue')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)

ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - Final Hybrid Model\n(OpenAI Embeddings + CVSS Features)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
roc_path = Path(CONFIG['visualizations_dir']) / 'final_hybrid_roc_curve.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {roc_path}")
plt.close()

# 2. Confusion Matrix
print("\n📊 Creating confusion matrix...")
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
            xticklabels=['Not Exploited', 'Exploited'],
            yticklabels=['Not Exploited', 'Exploited'],
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - Final Hybrid Model\n'
             f'OpenAI Embeddings + CVSS | ROC-AUC: {roc_auc:.4f}',
             fontsize=16, fontweight='bold', pad=20)

# Add performance metrics as text annotation
metrics_text = f"""
Performance Metrics:
• ROC-AUC: {roc_auc:.4f}
• Avg Precision: {avg_precision:.4f}
• F1 Score: {f1:.4f}
• Recall: {sensitivity:.2%}
• Precision: {precision:.2%}
• Specificity: {specificity:.2%}

Detected: {tp}/{tp + fn} exploited CVEs
False alarms: {fp}/{fp + tn} safe CVEs
"""

plt.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
         fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))

plt.tight_layout()
cm_path = Path(CONFIG['visualizations_dir']) / 'final_hybrid_confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {cm_path}")
plt.close()

# 3. Precision-Recall Curve
print("\n📊 Creating precision-recall curve...")
fig, ax = plt.subplots(figsize=(10, 8))

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
ax.plot(recall_curve, precision_curve, label=f'Hybrid Model (AP = {avg_precision:.3f})',
        linewidth=3, color='darkgreen')
ax.axhline(y=y_test.mean(), color='k', linestyle='--',
           label=f'Baseline (prevalence = {y_test.mean():.4f})', linewidth=1.5)

ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
ax.set_title('Precision-Recall Curve - Final Hybrid Model\n(OpenAI Embeddings + CVSS Features)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
pr_path = Path(CONFIG['visualizations_dir']) / 'final_hybrid_precision_recall.png'
plt.savefig(pr_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {pr_path}")
plt.close()

print("\n✅ All visualizations saved!")

# =============================================================================
# 10. SAVE RESULTS AND MODEL
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save detailed results as JSON
results = {
    'model_type': 'Logistic Regression',
    'features': 'OpenAI Embeddings (1536-dim) + CVSS Features (38-dim)',
    'total_features': int(X_hybrid.shape[1]),
    'cvss_features': int(X_cvss_features.shape[1]),
    'embedding_features': int(X_embeddings.shape[1]),
    'train_years': f"{CONFIG['train_years'][0]}-{CONFIG['train_years'][1]}",
    'test_years': f"{CONFIG['test_years'][0]}-{CONFIG['test_years'][1]}",
    'dataset': {
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'train_exploited': int(y_train.sum()),
        'test_exploited': int(y_test.sum()),
        'train_prevalence': float(y_train.mean()),
        'test_prevalence': float(y_test.mean())
    },
    'performance': {
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'f1_score': float(f1),
        'sensitivity_recall': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision)
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'config': CONFIG
}

results_path = Path(CONFIG['output_dir']) / 'final_hybrid_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n💾 Results saved: {results_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'cve_id': cve_ids_test,
    'year': years_test,
    'true_label': y_test,
    'predicted_label': y_pred,
    'probability_exploited': y_proba
})

predictions_path = Path(CONFIG['output_dir']) / 'final_hybrid_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"💾 Predictions saved: {predictions_path}")

# Save model
model_path = Path(CONFIG['output_dir']) / 'final_hybrid_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'cvss_feature_names': cvss_feature_names,
        'config': CONFIG,
        'performance': results['performance']
    }, f)
print(f"💾 Model saved: {model_path}")

print("\n✅ All results and model saved!")

# =============================================================================
# 11. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\n🎯 Model: Logistic Regression (Hybrid)")
print(f"   Features: OpenAI Embeddings (1536-dim) + CVSS Features ({X_cvss_features.shape[1]}-dim)")
print(f"   Total dimensions: {X_hybrid.shape[1]}")

print(f"\n📊 Dataset:")
print(f"   Total CVEs: {len(X_hybrid):,}")
print(f"   Training ({CONFIG['train_years'][0]}-{CONFIG['train_years'][1]}): {len(X_train):,} CVEs")
print(f"   Testing ({CONFIG['test_years'][0]}-{CONFIG['test_years'][1]}): {len(X_test):,} CVEs")
print(f"   Class imbalance: {(1 - y_test.mean())/y_test.mean():.1f}:1 ratio")

print(f"\n🏆 PERFORMANCE:")
print(f"   ROC-AUC: {roc_auc:.4f}")
print(f"   Average Precision: {avg_precision:.4f}")
print(f"   F1 Score: {f1:.4f}")
print(f"   Recall: {sensitivity:.2%}")
print(f"   Precision: {precision:.2%}")

print(f"\n💡 KEY RESULTS:")
print(f"   ✅ Detected {tp} out of {tp + fn} exploited vulnerabilities ({sensitivity:.1%} recall)")
print(f"   ✅ Generated {fp} false alarms out of {fp + tn} safe CVEs")
print(f"   ✅ Hybrid approach combines structured and semantic features")

print(f"\n📁 Output files:")
print(f"   Results: {results_path}")
print(f"   Predictions: {predictions_path}")
print(f"   Model: {model_path}")
print(f"   Visualizations: {CONFIG['visualizations_dir']}/")

print(f"\n🚀 Next steps:")
print(f"   - Review visualizations: open {CONFIG['visualizations_dir']}/")
print(f"   - Analyze predictions: head {predictions_path}")
print(f"   - Deploy model for production use")
print(f"   - Retrain periodically with new KEV data")

print("\n" + "=" * 80)
print("✨ FINAL HYBRID MODEL COMPLETE ✨")
print("=" * 80)

