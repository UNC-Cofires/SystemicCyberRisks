#!/usr/bin/env python3
"""
Train Models on Embeddings
===========================
Trains both Logistic Regression and Random Forest classifiers on embeddings
to predict vulnerability exploitation using temporal train/test split.

Train: 2015-2023 data
Test:  2024+ data

Supports both OpenAI and Sentence Transformer embeddings.

Usage:
    python embedding_pipeline/4_train_models.py --embedding-type sentencetransformer
    python embedding_pipeline/4_train_models.py --embedding-type openai
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("STEP 4: TRAIN MODELS ON EMBEDDINGS")
print("=" * 70)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train models on embeddings')
parser.add_argument(
    '--embedding-type',
    type=str,
    choices=['sentencetransformer', 'openai'],
    default='sentencetransformer',
    help='Type of embeddings to use (default: sentencetransformer)'
)
args = parser.parse_args()

embedding_type = args.embedding_type
print(f"\n📊 Using {embedding_type} embeddings")

# Create output directories
os.makedirs('embedding_pipeline/results', exist_ok=True)
os.makedirs('embedding_pipeline/models', exist_ok=True)
os.makedirs('embedding_pipeline/visualizations', exist_ok=True)

# Load embeddings
print(f"\n📂 Loading embeddings...")
embeddings_path = f'embedding_pipeline/data/embeddings_{embedding_type}.npz'
metadata_path = f'embedding_pipeline/data/metadata_{embedding_type}.csv'

if not os.path.exists(embeddings_path):
    print(f"❌ ERROR: Embeddings file not found: {embeddings_path}")
    print(f"   Please run the embedding generation script first:")
    print(f"   python embedding_pipeline/3b_generate_embeddings_sentencetransformer.py")
    exit(1)

embeddings_data = np.load(embeddings_path)
embeddings = embeddings_data['embeddings']
metadata = pd.read_csv(metadata_path)

print(f"   Loaded embeddings: {embeddings.shape}")
print(f"   Loaded metadata: {len(metadata):,} records")

# Temporal train/test split: 2015-2023 for training, 2024+ for testing
print(f"\n🔄 Creating temporal train/test split...")
print(f"   Training set: 2015-2023")
print(f"   Test set: 2024+")

train_mask = metadata['year'] < 2024
test_mask = metadata['year'] >= 2024

X_train = embeddings[train_mask]
y_train = metadata.loc[train_mask, 'target'].values
X_test = embeddings[test_mask]
y_test = metadata.loc[test_mask, 'target'].values

print(f"\n📊 Dataset split:")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Training exploited: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   Test exploited: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

print("\n🏋️  Training Logistic Regression...")
print("   Using class_weight='balanced' to handle class imbalance")

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=1,
    verbose=0
)

lr_model.fit(X_train, y_train)
print("   ✅ Training complete")

# Predictions
print("\n🔮 Generating predictions...")
lr_y_pred = lr_model.predict(X_test)
lr_y_proba = lr_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n📊 Logistic Regression Results:")
print("\n" + classification_report(y_test, lr_y_pred, target_names=['Not Exploited', 'Exploited']))

lr_roc_auc = roc_auc_score(y_test, lr_y_proba)
lr_avg_precision = average_precision_score(y_test, lr_y_proba)
lr_f1 = f1_score(y_test, lr_y_pred)

print(f"ROC-AUC Score: {lr_roc_auc:.4f}")
print(f"Average Precision: {lr_avg_precision:.4f}")
print(f"F1 Score: {lr_f1:.4f}")

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 2: RANDOM FOREST")
print("=" * 70)

print("\n🏋️  Training Random Forest...")
print("   Using class_weight='balanced' and 100 estimators")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=1,
    verbose=0
)

rf_model.fit(X_train, y_train)
print("   ✅ Training complete")

# Predictions
print("\n🔮 Generating predictions...")
rf_y_pred = rf_model.predict(X_test)
rf_y_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n📊 Random Forest Results:")
print("\n" + classification_report(y_test, rf_y_pred, target_names=['Not Exploited', 'Exploited']))

rf_roc_auc = roc_auc_score(y_test, rf_y_proba)
rf_avg_precision = average_precision_score(y_test, rf_y_proba)
rf_f1 = f1_score(y_test, rf_y_pred)

print(f"ROC-AUC Score: {rf_roc_auc:.4f}")
print(f"Average Precision: {rf_avg_precision:.4f}")
print(f"F1 Score: {rf_f1:.4f}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = {
    'Logistic Regression': {
        'ROC-AUC': lr_roc_auc,
        'Average Precision': lr_avg_precision,
        'F1 Score': lr_f1
    },
    'Random Forest': {
        'ROC-AUC': rf_roc_auc,
        'Average Precision': rf_avg_precision,
        'F1 Score': rf_f1
    }
}

print("\n📊 Performance Comparison:")
print(f"\n{'Metric':<20} {'Logistic Reg':>15} {'Random Forest':>15} {'Winner':>15}")
print("-" * 70)
for metric in ['ROC-AUC', 'Average Precision', 'F1 Score']:
    lr_val = comparison['Logistic Regression'][metric]
    rf_val = comparison['Random Forest'][metric]
    winner = 'Logistic Reg' if lr_val > rf_val else ('Random Forest' if rf_val > lr_val else 'Tie')
    print(f"{metric:<20} {lr_val:>15.4f} {rf_val:>15.4f} {winner:>15}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n📊 Generating visualizations...")

# 1. ROC Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Logistic Regression ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_proba)
ax1.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_roc_auc:.3f})', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_proba)
ax2.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_roc_auc:.3f})', linewidth=2, color='green')
ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
roc_path = f'embedding_pipeline/visualizations/roc_curves_{embedding_type}.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {roc_path}")
plt.close()

# 2. Precision-Recall Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Logistic Regression PR
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_y_proba)
ax1.plot(lr_recall, lr_precision, label=f'Logistic Regression (AP = {lr_avg_precision:.3f})', linewidth=2)
ax1.axhline(y=y_test.mean(), color='k', linestyle='--', label='Baseline (prevalence)', linewidth=1)
ax1.set_xlabel('Recall', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Precision-Recall Curve - Logistic Regression', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Random Forest PR
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_y_proba)
ax2.plot(rf_recall, rf_precision, label=f'Random Forest (AP = {rf_avg_precision:.3f})', linewidth=2, color='green')
ax2.axhline(y=y_test.mean(), color='k', linestyle='--', label='Baseline (prevalence)', linewidth=1)
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve - Random Forest', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
pr_path = f'embedding_pipeline/visualizations/precision_recall_{embedding_type}.png'
plt.savefig(pr_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {pr_path}")
plt.close()

# 3. Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression CM
lr_cm = confusion_matrix(y_test, lr_y_pred)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=True)
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
ax1.set_title('Confusion Matrix - Logistic Regression')
ax1.set_xticklabels(['Not Exploited', 'Exploited'])
ax1.set_yticklabels(['Not Exploited', 'Exploited'])

# Random Forest CM
rf_cm = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=True)
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
ax2.set_title('Confusion Matrix - Random Forest')
ax2.set_xticklabels(['Not Exploited', 'Exploited'])
ax2.set_yticklabels(['Not Exploited', 'Exploited'])

plt.tight_layout()
cm_path = f'embedding_pipeline/visualizations/confusion_matrices_{embedding_type}.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {cm_path}")
plt.close()

# 4. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['ROC-AUC', 'Average Precision', 'F1 Score']
lr_scores = [comparison['Logistic Regression'][m] for m in metrics]
rf_scores = [comparison['Random Forest'][m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='steelblue')
ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='seagreen')

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 1)

# Add value labels on bars
for i, (lr_val, rf_val) in enumerate(zip(lr_scores, rf_scores)):
    ax.text(i - width/2, lr_val + 0.02, f'{lr_val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, rf_val + 0.02, f'{rf_val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
comp_path = f'embedding_pipeline/visualizations/model_comparison_{embedding_type}.png'
plt.savefig(comp_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {comp_path}")
plt.close()

# 5. COMBINED ROC Curve (Both models on one plot)
print("\n📊 Generating combined visualizations...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot both models
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_proba)

ax.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_roc_auc:.3f})', 
        linewidth=2.5, color='steelblue')
ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_roc_auc:.3f})', 
        linewidth=2.5, color='seagreen')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title(f'ROC Curve Comparison - {embedding_type.upper()} Embeddings', 
             fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
roc_combined_path = f'embedding_pipeline/visualizations/roc_curve_combined_{embedding_type}.png'
plt.savefig(roc_combined_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {roc_combined_path}")
plt.close()

# 6. Best Model Confusion Matrix (Single, larger plot)
# Determine best model
best_model_name = 'Logistic Regression' if lr_roc_auc > rf_roc_auc else 'Random Forest'
best_cm = lr_cm if lr_roc_auc > rf_roc_auc else rf_cm
best_roc = max(lr_roc_auc, rf_roc_auc)
best_ap = lr_avg_precision if lr_roc_auc > rf_roc_auc else rf_avg_precision
best_f1 = lr_f1 if lr_roc_auc > rf_roc_auc else rf_f1

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
            xticklabels=['Not Exploited', 'Exploited'],
            yticklabels=['Not Exploited', 'Exploited'],
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - Best Model ({best_model_name})\n'
             f'{embedding_type.upper()} Embeddings | ROC-AUC: {best_roc:.4f}',
             fontsize=15, fontweight='bold', pad=15)

# Add performance metrics as text
tn, fp, fn, tp = best_cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

metrics_text = f"""
Performance Metrics:
• ROC-AUC: {best_roc:.4f}
• Avg Precision: {best_ap:.4f}
• F1 Score: {best_f1:.4f}
• Recall: {sensitivity:.2%}
• Precision: {precision:.2%}
• Specificity: {specificity:.2%}

Exploited CVEs detected: {tp}/{tp + fn}
False alarms: {fp}/{fp + tn}
"""

plt.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
         fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
cm_best_path = f'embedding_pipeline/visualizations/confusion_matrix_best_{embedding_type}.png'
plt.savefig(cm_best_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {cm_best_path}")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n💾 Saving results...")

# Save detailed results as JSON
results = {
    'embedding_type': embedding_type,
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'train_exploited': int(y_train.sum()),
        'test_exploited': int(y_test.sum()),
        'train_years': '2015-2023',
        'test_years': '2024+'
    },
    'logistic_regression': {
        'roc_auc': float(lr_roc_auc),
        'average_precision': float(lr_avg_precision),
        'f1_score': float(lr_f1),
        'confusion_matrix': lr_cm.tolist()
    },
    'random_forest': {
        'roc_auc': float(rf_roc_auc),
        'average_precision': float(rf_avg_precision),
        'f1_score': float(rf_f1),
        'confusion_matrix': rf_cm.tolist()
    }
}

results_path = f'embedding_pipeline/results/results_{embedding_type}.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"   Saved: {results_path}")

# Save predictions for further analysis
predictions_df = pd.DataFrame({
    'id': metadata.loc[test_mask, 'id'].values,
    'year': metadata.loc[test_mask, 'year'].values,
    'true_label': y_test,
    'lr_prediction': lr_y_pred,
    'lr_probability': lr_y_proba,
    'rf_prediction': rf_y_pred,
    'rf_probability': rf_y_proba
})

pred_path = f'embedding_pipeline/results/predictions_{embedding_type}.csv'
predictions_df.to_csv(pred_path, index=False)
print(f"   Saved: {pred_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE - All results saved")
print("=" * 70)

# Final summary
print(f"\n🎉 FINAL SUMMARY:")
print(f"\n   Embedding Type: {embedding_type.upper()}")
print(f"   Train/Test Split: 2015-2023 / 2024+")
print(f"\n   🏆 BEST MODEL (by ROC-AUC): {winner}")
print(f"\n   Logistic Regression ROC-AUC: {lr_roc_auc:.4f}")
print(f"   Random Forest ROC-AUC: {rf_roc_auc:.4f}")
print(f"\n   Results saved in: embedding_pipeline/results/")
print(f"   Visualizations saved in: embedding_pipeline/visualizations/")
