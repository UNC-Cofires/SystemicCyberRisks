#!/usr/bin/env python3
"""
Generate Individual Visualizations: CVSS-Only and OpenAI Embedding-Only Models
===============================================================================
Trains two separate Logistic Regression models on the aligned dataset
and generates 4 individual visualizations:
  1. cvss_roc_curve.png
  2. cvss_confusion_matrix.png
  3. openai_embedding_roc_curve.png
  4. openai_embedding_confusion_matrix.png

Uses the same data loading, alignment, and temporal split as final_hybrid.py
to ensure results are directly comparable to the hybrid model.

Usage:
    python embedding_pipeline/generate_individual_visualizations.py
"""

# Standard libraries
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import OneHotEncoder

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("INDIVIDUAL MODEL VISUALIZATIONS: CVSS-Only & OpenAI Embedding-Only")
print("=" * 80)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG = {
    'cvss_data_path': 'data/data.csv',
    'openai_embeddings_path': 'embedding_pipeline/data/embeddings_openai.npz',
    'openai_metadata_path': 'embedding_pipeline/data/metadata_openai.csv',
    'kev_catalog_path': 'data/known_exploited_vulnerabilities.csv',
    'output_dir': '.',  # Repository root
    'train_years': (2015, 2023),
    'test_years': (2024, 2025),
    'random_state': 42
}

print(f"\n   Training years: {CONFIG['train_years'][0]}-{CONFIG['train_years'][1]}")
print(f"   Testing years: {CONFIG['test_years'][0]}-{CONFIG['test_years'][1]}")
print(f"   Output directory: {CONFIG['output_dir']}")

# =============================================================================
# 2. LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

print(f"\n   Loading CVSS features from: {CONFIG['cvss_data_path']}")
cvss_df = pd.read_csv(CONFIG['cvss_data_path'])
print(f"   Loaded {len(cvss_df):,} CVEs with {cvss_df.shape[1]} columns")

print(f"\n   Loading OpenAI embeddings from: {CONFIG['openai_embeddings_path']}")
embeddings_data = np.load(CONFIG['openai_embeddings_path'])
embeddings = embeddings_data['embeddings']
print(f"   Embeddings shape: {embeddings.shape}")

print(f"\n   Loading embedding metadata from: {CONFIG['openai_metadata_path']}")
embedding_metadata = pd.read_csv(CONFIG['openai_metadata_path'])
print(f"   Metadata shape: {embedding_metadata.shape}")

print(f"\n   Loading KEV catalog from: {CONFIG['kev_catalog_path']}")
kev_df = pd.read_csv(CONFIG['kev_catalog_path'])
kev_cves = set(kev_df['cveID'].str.upper().str.strip())
print(f"   KEV catalog size: {len(kev_cves):,} known exploited CVEs")

# =============================================================================
# 3. ALIGN DATASETS BY CVE ID
# =============================================================================

print("\n" + "=" * 80)
print("ALIGNING DATASETS")
print("=" * 80)

embedding_df = embedding_metadata.copy()
embedding_df['original_idx'] = range(len(embedding_df))

# Handle duplicate IDs in embedding metadata
if embedding_df['id'].duplicated().any():
    n_dupes = embedding_df['id'].duplicated().sum()
    print(f"   Found {n_dupes:,} duplicate IDs in embedding metadata, keeping first")
    keep_indices = embedding_df['original_idx'][~embedding_df['id'].duplicated()].values
    embedding_df = embedding_df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    embeddings = embeddings[keep_indices]

embedding_df['embedding_idx'] = range(len(embedding_df))

# Standardize CVE IDs
cvss_df['id'] = cvss_df['id'].str.upper().str.strip()
embedding_df['id'] = embedding_df['id'].str.upper().str.strip()

# Find common CVEs
cvss_cves = set(cvss_df['id'])
embedding_cves = set(embedding_df['id'])
common_cves = cvss_cves.intersection(embedding_cves)

print(f"   CVSS dataset: {len(cvss_cves):,} CVEs")
print(f"   Embedding dataset: {len(embedding_cves):,} CVEs")
print(f"   Common CVEs: {len(common_cves):,} ({len(common_cves)/len(cvss_cves)*100:.1f}% overlap)")

# Filter and sort
cvss_df = cvss_df[cvss_df['id'].isin(common_cves)].copy()
embedding_df = embedding_df[embedding_df['id'].isin(common_cves)].copy()
cvss_df = cvss_df.sort_values('id').reset_index(drop=True)
embedding_df = embedding_df.sort_values('id').reset_index(drop=True)

assert (cvss_df['id'].values == embedding_df['id'].values).all(), "CVE IDs not aligned!"

# Get aligned embeddings
embedding_indices = embedding_df['embedding_idx'].values
embeddings_aligned = embeddings[embedding_indices]

# Create target variable from KEV catalog
cvss_df['target'] = cvss_df['id'].apply(lambda x: 1 if x in kev_cves else 0)

print(f"   Aligned dataset: {len(cvss_df):,} CVEs, {cvss_df['target'].sum():,} exploited ({cvss_df['target'].mean()*100:.2f}%)")

# =============================================================================
# 4. PREPARE FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("PREPARING FEATURES")
print("=" * 80)

# CVSS features
feature_cols = [col for col in cvss_df.columns if col not in ['id', 'year', 'target']]
numeric_cols = cvss_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = cvss_df[feature_cols].select_dtypes(include=['object']).columns.tolist()

print(f"   Numeric features: {len(numeric_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")

X_cvss = cvss_df[feature_cols].copy()

if categorical_cols:
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_categorical = encoder.fit_transform(X_cvss[categorical_cols])
    X_cvss_numeric = X_cvss[numeric_cols].values
    X_cvss_features = np.hstack([X_cvss_numeric, X_categorical])
else:
    X_cvss_features = X_cvss[numeric_cols].values

print(f"   CVSS features shape: {X_cvss_features.shape}")

# Embedding features
X_embeddings = embeddings_aligned
print(f"   Embedding features shape: {X_embeddings.shape}")

# Target and metadata
y = cvss_df['target'].values
years = cvss_df['year'].values if 'year' in cvss_df.columns else embedding_df['year'].values
cve_ids = cvss_df['id'].values

# =============================================================================
# 5. TEMPORAL TRAIN/TEST SPLIT
# =============================================================================

print("\n" + "=" * 80)
print("TEMPORAL TRAIN/TEST SPLIT")
print("=" * 80)

train_mask = (years >= CONFIG['train_years'][0]) & (years <= CONFIG['train_years'][1])
test_mask = (years >= CONFIG['test_years'][0]) & (years <= CONFIG['test_years'][1])

X_cvss_train, X_cvss_test = X_cvss_features[train_mask], X_cvss_features[test_mask]
X_emb_train, X_emb_test = X_embeddings[train_mask], X_embeddings[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"   Training: {X_cvss_train.shape[0]:,} samples ({y_train.sum():,} exploited)")
print(f"   Testing:  {X_cvss_test.shape[0]:,} samples ({y_test.sum():,} exploited)")

# =============================================================================
# 6. TRAIN TWO MODELS
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

model_config = {
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': CONFIG['random_state'],
    'n_jobs': 1,
    'verbose': 0
}

# Model 1: CVSS-only
print("\n   Training CVSS-only Logistic Regression...")
model_cvss = LogisticRegression(**model_config)
model_cvss.fit(X_cvss_train, y_train)
y_pred_cvss = model_cvss.predict(X_cvss_test)
y_proba_cvss = model_cvss.predict_proba(X_cvss_test)[:, 1]
print("   Done.")

# Model 2: OpenAI Embedding-only
print("   Training OpenAI Embedding-only Logistic Regression...")
model_emb = LogisticRegression(**model_config)
model_emb.fit(X_emb_train, y_train)
y_pred_emb = model_emb.predict(X_emb_test)
y_proba_emb = model_emb.predict_proba(X_emb_test)[:, 1]
print("   Done.")

# =============================================================================
# 7. COMPUTE METRICS
# =============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# CVSS metrics
cvss_roc_auc = roc_auc_score(y_test, y_proba_cvss)
cvss_avg_precision = average_precision_score(y_test, y_proba_cvss)
cvss_f1 = f1_score(y_test, y_pred_cvss)
cvss_cm = confusion_matrix(y_test, y_pred_cvss)
cvss_tn, cvss_fp, cvss_fn, cvss_tp = cvss_cm.ravel()
cvss_sensitivity = cvss_tp / (cvss_tp + cvss_fn) if (cvss_tp + cvss_fn) > 0 else 0
cvss_specificity = cvss_tn / (cvss_tn + cvss_fp) if (cvss_tn + cvss_fp) > 0 else 0
cvss_precision_val = cvss_tp / (cvss_tp + cvss_fp) if (cvss_tp + cvss_fp) > 0 else 0

print(f"\n   CVSS-Only Performance:")
print(f"   ROC-AUC: {cvss_roc_auc:.4f}")
print(f"   Avg Precision: {cvss_avg_precision:.4f}")
print(f"   F1 Score: {cvss_f1:.4f}")
print(f"   Recall: {cvss_sensitivity:.2%} | Specificity: {cvss_specificity:.2%}")
print(classification_report(y_test, y_pred_cvss, target_names=['Not Exploited', 'Exploited'], digits=4))

# Embedding metrics
emb_roc_auc = roc_auc_score(y_test, y_proba_emb)
emb_avg_precision = average_precision_score(y_test, y_proba_emb)
emb_f1 = f1_score(y_test, y_pred_emb)
emb_cm = confusion_matrix(y_test, y_pred_emb)
emb_tn, emb_fp, emb_fn, emb_tp = emb_cm.ravel()
emb_sensitivity = emb_tp / (emb_tp + emb_fn) if (emb_tp + emb_fn) > 0 else 0
emb_specificity = emb_tn / (emb_tn + emb_fp) if (emb_tn + emb_fp) > 0 else 0
emb_precision_val = emb_tp / (emb_tp + emb_fp) if (emb_tp + emb_fp) > 0 else 0

print(f"\n   OpenAI Embedding-Only Performance:")
print(f"   ROC-AUC: {emb_roc_auc:.4f}")
print(f"   Avg Precision: {emb_avg_precision:.4f}")
print(f"   F1 Score: {emb_f1:.4f}")
print(f"   Recall: {emb_sensitivity:.2%} | Specificity: {emb_specificity:.2%}")
print(classification_report(y_test, y_pred_emb, target_names=['Not Exploited', 'Exploited'], digits=4))

# =============================================================================
# 8. GENERATE VISUALIZATIONS
# =============================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# --- Visualization 1: CVSS-Only ROC Curve ---
print("\n   Creating CVSS-only ROC curve...")
fig, ax = plt.subplots(figsize=(10, 8))

fpr, tpr, _ = roc_curve(y_test, y_proba_cvss)
ax.plot(fpr, tpr, label=f'CVSS-Only Model (AUC = {cvss_roc_auc:.3f})',
        linewidth=3, color='darkred')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)

ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - CVSS-Only Logistic Regression\n(Structured Vulnerability Features)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_dir'], 'cvss_roc_curve.png'),
            dpi=300, bbox_inches='tight')
print(f"   Saved: cvss_roc_curve.png")
plt.close()

# --- Visualization 2: CVSS-Only Confusion Matrix ---
print("   Creating CVSS-only confusion matrix...")
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(cvss_cm, annot=True, fmt='d', cmap='Oranges', ax=ax, cbar=True,
            xticklabels=['Not Exploited', 'Exploited'],
            yticklabels=['Not Exploited', 'Exploited'],
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - CVSS-Only Logistic Regression\n'
             f'Structured Features | ROC-AUC: {cvss_roc_auc:.4f}',
             fontsize=16, fontweight='bold', pad=20)

metrics_text = (
    f"\nPerformance Metrics:\n"
    f"\u2022 ROC-AUC: {cvss_roc_auc:.4f}\n"
    f"\u2022 Avg Precision: {cvss_avg_precision:.4f}\n"
    f"\u2022 F1 Score: {cvss_f1:.4f}\n"
    f"\u2022 Recall: {cvss_sensitivity:.2%}\n"
    f"\u2022 Precision: {cvss_precision_val:.2%}\n"
    f"\u2022 Specificity: {cvss_specificity:.2%}\n"
    f"\nDetected: {cvss_tp}/{cvss_tp + cvss_fn} exploited CVEs\n"
    f"False alarms: {cvss_fp}/{cvss_fp + cvss_tn} safe CVEs\n"
)

plt.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
         fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_dir'], 'cvss_confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
print(f"   Saved: cvss_confusion_matrix.png")
plt.close()

# --- Visualization 3: OpenAI Embedding-Only ROC Curve ---
print("   Creating OpenAI embedding-only ROC curve...")
fig, ax = plt.subplots(figsize=(10, 8))

fpr, tpr, _ = roc_curve(y_test, y_proba_emb)
ax.plot(fpr, tpr, label=f'OpenAI Embedding Model (AUC = {emb_roc_auc:.3f})',
        linewidth=3, color='darkgreen')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)

ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve - OpenAI Embedding-Only Logistic Regression\n(text-embedding-3-small, 1536-dim)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_dir'], 'openai_embedding_roc_curve.png'),
            dpi=300, bbox_inches='tight')
print(f"   Saved: openai_embedding_roc_curve.png")
plt.close()

# --- Visualization 4: OpenAI Embedding-Only Confusion Matrix ---
print("   Creating OpenAI embedding-only confusion matrix...")
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(emb_cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=True,
            xticklabels=['Not Exploited', 'Exploited'],
            yticklabels=['Not Exploited', 'Exploited'],
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - OpenAI Embedding-Only Logistic Regression\n'
             f'text-embedding-3-small (1536-dim) | ROC-AUC: {emb_roc_auc:.4f}',
             fontsize=16, fontweight='bold', pad=20)

metrics_text = (
    f"\nPerformance Metrics:\n"
    f"\u2022 ROC-AUC: {emb_roc_auc:.4f}\n"
    f"\u2022 Avg Precision: {emb_avg_precision:.4f}\n"
    f"\u2022 F1 Score: {emb_f1:.4f}\n"
    f"\u2022 Recall: {emb_sensitivity:.2%}\n"
    f"\u2022 Precision: {emb_precision_val:.2%}\n"
    f"\u2022 Specificity: {emb_specificity:.2%}\n"
    f"\nDetected: {emb_tp}/{emb_tp + emb_fn} exploited CVEs\n"
    f"False alarms: {emb_fp}/{emb_fp + emb_tn} safe CVEs\n"
)

plt.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
         fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.4))

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_dir'], 'openai_embedding_confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
print(f"   Saved: openai_embedding_confusion_matrix.png")
plt.close()

# =============================================================================
# 9. SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n   Model Comparison (same aligned dataset, {len(X_cvss_test):,} test CVEs):")
print(f"   {'Model':<25} {'ROC-AUC':>10} {'Avg Prec':>10} {'F1':>10} {'Recall':>10}")
print(f"   {'-'*65}")
print(f"   {'CVSS-Only':<25} {cvss_roc_auc:>10.4f} {cvss_avg_precision:>10.4f} {cvss_f1:>10.4f} {cvss_sensitivity:>10.2%}")
print(f"   {'OpenAI Embedding-Only':<25} {emb_roc_auc:>10.4f} {emb_avg_precision:>10.4f} {emb_f1:>10.4f} {emb_sensitivity:>10.2%}")

print(f"\n   Output files (repository root):")
print(f"   1. cvss_roc_curve.png")
print(f"   2. cvss_confusion_matrix.png")
print(f"   3. openai_embedding_roc_curve.png")
print(f"   4. openai_embedding_confusion_matrix.png")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
