#!/usr/bin/env python3
"""
Baseline Model - Abel Koshy 07/25

This script combines data cleaning and baseline modeling into a single workflow:
1. Data Cleaning: Preprocesses raw vulnerability data
2. Baseline Modeling: Creates and evaluates a logistic regression model

Input: data/vulnerabilities.csv (from generate_data pipeline)
Output: data/data.csv (cleaned data) + model evaluation + ROC curve
"""

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

print("🧹 Starting Data Cleaning Phase...")
print("="*60)

# =============================================================================
# DATA CLEANING PHASE
# =============================================================================

# Read the raw CSV file and drop unwanted columns
print("📂 Loading raw vulnerability data...")
df = pd.read_csv('data/vulnerabilities.csv', low_memory=False)
print(f"   Initial data shape: {df.shape}")

print("🗑️  Dropping unwanted columns...")
# Drop unwanted columns (only if they exist)
columns_to_drop = ['Unnamed: 0', 'userInteractions', 'accessVector', 'accessComplexity',
                   'authentication', 'attackRequirements', 'vulnConfidentialityImpact',
                   'vulnIntegrityImpact', 'vulnAvailabilityImpact','subConfidentialityImpact', 
                   'subIntegrityImpact','subAvailabilityImpact', 'exploitMaturity',
                   'confidentialityRequirement', 'integrityRequirement', 'availabilityRequirement', 
                   'modifiedAttackVector', 'modifiedAttackComplexity', 'modifiedAttackRequirements',
                   'modifiedPrivilegesRequired', 'modifiedUserInteraction', 'modifiedVulnConfidentialityImpact', 
                   'modifiedVulnIntegrityImpact', 'modifiedVulnAvailabilityImpact', 'modifiedSubConfidentialityImpact', 
                   'modifiedSubIntegrityImpact', 'modifiedSubAvailabilityImpact', 'Safety','Automatable', 'Recovery', 
                   'valueDensity', 'vulnerabilityResponseEffort', 'providerUrgency', 'vectorString', 'description']

# Only drop columns that actually exist in the dataframe
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_columns_to_drop:
    df = df.drop(columns=existing_columns_to_drop)
    print(f"   Dropped {len(existing_columns_to_drop)} unwanted columns")

# Drop version 4, and keep only 1 of each CveID
print("📋 Filtering data...")
df = df[df['version'] != 4.0].copy()
df = df.drop_duplicates(subset=['id'], keep='first')
print(f"   Data shape after filtering: {df.shape}")

# Extract the year from the 'id' column
df['year'] = df['id'].str.extract(r'CVE-(\d{4})-')[0].astype(int)

# Drop everything from 2015 and before
print("📅 Filtering to post-2015 vulnerabilities...")
df = df[df['year'] > 2015]
print(f"   Data shape after year filtering: {df.shape}")

# Read targets (known exploited vulnerabilities)
print("🎯 Loading target data (known exploited vulnerabilities)...")
dt = pd.read_csv('data/known_exploited_vulnerabilities.csv')
print(f"   Loaded {len(dt)} known exploited vulnerabilities")

# Extract the year from the 'id' column in targets
df['target'] = df['id'].apply(lambda x: 1 if x in dt['cveID'].values else 0)
target_count = df['target'].sum()
print(f"   Found {target_count} matches ({target_count/len(df)*100:.2f}% of dataset)")

# Fill null values 
print("🔧 Filling null values...")
df.fillna(0, inplace=True)

# Drop remaining unwanted columns
df = df.drop(columns=['version', 'year'])

# Save the cleaned data to a new CSV file
print("💾 Saving cleaned data...")
df.to_csv('data/data.csv', index=False)
print(f"   Cleaned dataset saved to: data/data.csv")
print(f"   Final data shape: {df.shape}")

print("\n🤖 Starting Baseline Modeling Phase...")
print("="*60)

# =============================================================================
# BASELINE MODELING PHASE
# =============================================================================

# Read the cleaned data from the CSV file
print("📂 Loading cleaned data for modeling...")
data = pd.read_csv('data/data.csv')
print(f"   Loaded data shape: {data.shape}")

# Binary encode each of the 'Scores' as 1 if 7+ and 0 otherwise
print("🔢 Binary encoding vulnerability scores...")
columns_to_encode = ['baseScoreAv', 'exploitScoreAv', 'impactScoreAv', 'baseScoreMax', 'exploitScoreMax', 'impactScoreMax']

for col in columns_to_encode:
    if col in data.columns:
        data[col] = data[col].apply(lambda x: 1 if x >= 7 else 0)
        print(f"   Encoded {col}: {data[col].sum()} high-risk scores (≥7)")

# Scale the numeric columns
print("📐 Scaling numeric features...")
cols_to_scale = ['numScores', 'agreement']
available_cols_to_scale = [col for col in cols_to_scale if col in data.columns]

if available_cols_to_scale:
    scaler = MinMaxScaler()
    data[available_cols_to_scale] = scaler.fit_transform(data[available_cols_to_scale])
    print(f"   Scaled columns: {available_cols_to_scale}")

# Define the list of categorical columns to encode
print("🏷️  One-hot encoding categorical features...")
categorical_cols = ['baseSeverity', 'attackVector', 'attackComplexity',
                    'privilegesRequired', 'scope', 'confidentialityImpact',
                    'integrityImpact', 'availabilityImpact', 'userInteraction']

# Filter to only include columns that exist in the dataset
available_categorical_cols = [col for col in categorical_cols if col in data.columns]
print(f"   Available categorical columns: {available_categorical_cols}")

if available_categorical_cols:
    # Convert all categorical columns to string
    data[available_categorical_cols] = data[available_categorical_cols].astype(str)
    
    # Create encoder instance
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' to avoid dummy variable trap
    
    # Fit and transform the data
    encoded_array = encoder.fit_transform(data[available_categorical_cols])
    encoded_cols = encoder.get_feature_names_out(available_categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)
    print(f"   Created {len(encoded_cols)} one-hot encoded features")
    
    # Combine with original DataFrame (excluding original categorical columns)
    data_final = pd.concat([data.drop(columns=available_categorical_cols).reset_index(drop=True),
                          encoded_df.reset_index(drop=True)], axis=1)
else:
    data_final = data.copy()

print("\n🎯 Preparing Features and Target...")
print("="*40)

# Define features (X) and target (y)
feature_cols = [col for col in data_final.columns if col not in ['id', 'target', 'description']]
X = data_final[feature_cols]
y = data_final['target']

print(f"   Features: {len(feature_cols)} columns")
print(f"   Samples: {len(X)} total")
print(f"   Target distribution: {y.value_counts().to_dict()}")
print(f"   Target balance: {y.mean()*100:.2f}% positive class")

# Perform train-test split
print("\n🔄 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

print("\n🏋️  Training Baseline Model...")
print("="*40)

# Initialize and train a simple classification model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("✅ Model training completed!")

print("\n📊 Model Evaluation Results:")
print("="*40)

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)

# Generate an ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n📈 ROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
print("📊 Generating ROC curve visualization...")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)

plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
print("💾 ROC curve saved to: roc_curve.png")

print("\n🎉 BASELINE MODEL PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print("📊 Generated Files:")
print("   - data/data.csv (cleaned dataset)")
print("   - roc_curve.png (model performance visualization)")
print(f"📈 Final Model Performance: AUC = {roc_auc:.4f}") 