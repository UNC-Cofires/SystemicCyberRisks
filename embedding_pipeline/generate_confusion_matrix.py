#!/usr/bin/env python3
"""
Generate Confusion Matrix for Hybrid Model Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("HYBRID MODEL CONFUSION MATRIX ANALYSIS")
print("="*70)

# Load predictions
print("\n📂 Loading predictions...")
predictions_df = pd.read_csv('embedding_pipeline/hybrid_results/predictions_all_models.csv')

print(f"   Total predictions: {len(predictions_df):,}")
print(f"   True exploited: {predictions_df['true_label'].sum()}")
print(f"   True not exploited: {(predictions_df['true_label'] == 0).sum()}")

# Extract hybrid model predictions
y_true = predictions_df['true_label'].values
y_pred_hybrid = predictions_df['hybrid_prediction'].values
y_proba_hybrid = predictions_df['hybrid_probability'].values

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_hybrid)

print("\n📊 Confusion Matrix:")
print(cm)

# Calculate metrics
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Breakdown:")
print(f"   True Negatives (TN):  {tn:,} - Correctly predicted NOT exploited")
print(f"   False Positives (FP): {fp:,} - Incorrectly predicted AS exploited")
print(f"   False Negatives (FN): {fn:,} - Incorrectly predicted AS NOT exploited")
print(f"   True Positives (TP):  {tp:,} - Correctly predicted AS exploited")

# Calculate rates
print(f"\n📈 Performance Metrics:")
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"   Sensitivity (Recall):    {sensitivity:.2%} - {tp}/{tp + fn} exploited CVEs detected")
print(f"   Specificity:             {specificity:.2%} - {tn}/{tn + fp} non-exploited CVEs correctly identified")
print(f"   Precision (PPV):         {precision:.2%} - {tp}/{tp + fp} positive predictions were correct")
print(f"   Negative Predictive Value: {npv:.2%}")
print(f"   Accuracy:                {accuracy:.2%}")
print(f"   F1 Score:                {f1:.4f}")

# Calculate false alarm rate and miss rate
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n⚠️  Error Analysis:")
print(f"   False Positive Rate:     {fpr:.2%} - {fp} false alarms out of {fp + tn} non-exploited CVEs")
print(f"   False Negative Rate:     {fnr:.2%} - {fn} missed exploits out of {fn + tp} exploited CVEs")

# Create visualizations
print("\n🎨 Creating visualizations...")

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Confusion Matrix Heatmap (Counts)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
            xticklabels=['Not Exploited', 'Exploited'],
            yticklabels=['Not Exploited', 'Exploited'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax1.set_title('Confusion Matrix - Hybrid Model\n(Absolute Counts)',
              fontsize=15, fontweight='bold', pad=15)

# Add text annotations with percentages
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm.sum() * 100
        ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                ha='center', va='center', fontsize=10, color='gray')

# 2. Normalized Confusion Matrix (Percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', cbar=True, ax=ax2,
            xticklabels=['Not Exploited', 'Exploited'],
            yticklabels=['Not Exploited', 'Exploited'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax2.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax2.set_title('Confusion Matrix - Hybrid Model\n(Row-Normalized Percentages)',
              fontsize=15, fontweight='bold', pad=15)

plt.tight_layout()

# Save figure
output_path = Path('embedding_pipeline/visualizations/hybrid_confusion_matrix_detailed.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {output_path}")

# Create a single larger confusion matrix visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Custom annotations with both counts and percentages
annot_labels = np.array([[f'{cm[i,j]:,}\n({cm[i,j]/cm.sum()*100:.1f}%)'
                          for j in range(2)] for i in range(2)])

sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', cbar=True, ax=ax,
            xticklabels=['Predicted: Not Exploited', 'Predicted: Exploited'],
            yticklabels=['Actual: Not Exploited', 'Actual: Exploited'],
            cbar_kws={'label': 'Number of CVEs'},
            linewidths=2, linecolor='white')

ax.set_title('Hybrid Model Confusion Matrix\n2025 Test Set (9,651 CVEs)',
             fontsize=16, fontweight='bold', pad=20)

# Add performance metrics as text
metrics_text = f"""
Performance Metrics:
• Recall (Sensitivity): {sensitivity:.1%}
• Precision: {precision:.1%}
• Specificity: {specificity:.2%}
• F1 Score: {f1:.4f}
• Accuracy: {accuracy:.2%}

{tp} out of {tp + fn} exploited CVEs detected
{fp} false alarms out of {fp + tn} safe CVEs
"""

plt.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
         fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save single confusion matrix
output_path_single = Path('embedding_pipeline/visualizations/hybrid_confusion_matrix.png')
plt.savefig(output_path_single, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {output_path_single}")

plt.close('all')

# Print classification report
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred_hybrid,
                          target_names=['Not Exploited', 'Exploited'],
                          digits=4))

# Analyze predictions by probability threshold
print("\n" + "="*70)
print("THRESHOLD ANALYSIS")
print("="*70)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n{'Threshold':<12} {'TP':<6} {'FP':<8} {'FN':<6} {'Precision':<12} {'Recall':<10} {'F1':<10}")
print("-" * 70)

for thresh in thresholds:
    y_pred_thresh = (y_proba_hybrid >= thresh).astype(int)
    cm_thresh = confusion_matrix(y_true, y_pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()

    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0

    print(f"{thresh:<12.1f} {tp_t:<6} {fp_t:<8} {fn_t:<6} {prec_t:<12.2%} {rec_t:<10.2%} {f1_t:<10.4f}")

print("\n" + "="*70)
print("✅ CONFUSION MATRIX ANALYSIS COMPLETE")
print("="*70)

# Summary
print(f"\n🎯 Key Takeaway:")
print(f"   The hybrid model correctly identified {tp} out of {tp + fn} exploited")
print(f"   vulnerabilities in 2025 ({sensitivity:.1%} recall) with {precision:.1%} precision,")
print(f"   generating {fp} false alarms out of {fp + tn} non-exploited CVEs.")
