# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score


# read the data from the CSV file
data = pd.read_csv('data.csv')

# bianary encode each of the 'Scores' as 1 if 7+ and 0 otherwise
columns_to_encode = ['baseScoreAv', 'exploitScoreAv', 'impactScoreAv', 'baseScoreMax', 'exploitScoreMax', 'impactScoreMax']

for col in columns_to_encode:
  data[col] = data[col].apply(lambda x: 1 if x >= 7 else 0)

# scale the neumeric columns
cols_to_scale = ['numScores', 'agreement']

scaler = MinMaxScaler()
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

# Define the list of categorical columns to encode
categorical_cols = ['baseSeverity',	'attackVector',	'attackComplexity',
                    'privilegesRequired', 'scope', 'confidentialityImpact',
                    'integrityImpact',	'availabilityImpact', 'userInteraction']

# Convert all categorical columns to string
data[categorical_cols] = data[categorical_cols].astype(str)

# Create encoder instance
encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' if you want to avoid dummy variable trap

# Fit and transform the data
encoded_array = encoder.fit_transform(data[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

# Combine with original DataFrame (excluding original categorical columns)
data_final = pd.concat([data.drop(columns=categorical_cols).reset_index(drop=True),
                      encoded_df.reset_index(drop=True)], axis=1)

#### Model ####
# Define features (X) and target (y)
X = data_final.drop(['id','target'], axis=1)
y = data_final['target']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Initialize and train a simple classification model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)

# Generate an ROC curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)

plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
print("ROC curve saved to roc_curve.png")