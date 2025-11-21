#!/usr/bin/env python3
"""
NLP Text Preprocessing
======================
Performs comprehensive text preprocessing on vulnerability descriptions:
- Lowercasing
- URL/email/code removal
- Stop word filtering
- Punctuation handling
- Token cleaning
- Length filtering

Input:  embedding_pipeline/data/descriptions_with_labels.csv
Output: embedding_pipeline/data/descriptions_preprocessed.csv
"""

import pandas as pd
import re
import string
from collections import Counter
import nltk

print("=" * 70)
print("STEP 2: NLP TEXT PREPROCESSING")
print("=" * 70)

# Download required NLTK data
print("\n📥 Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords

# Initialize stopwords
STOP_WORDS = set(stopwords.words('english'))

# Add custom stop words common in CVE descriptions that don't add semantic value
CUSTOM_STOP_WORDS = {
    'cve', 'vulnerability', 'allows', 'could', 'may', 'via', 'using',
    'remote', 'attackers', 'attacker', 'user', 'users', 'allows'
}
STOP_WORDS.update(CUSTOM_STOP_WORDS)

print(f"   Using {len(STOP_WORDS)} stop words")

def clean_text(text):
    """
    Comprehensive text cleaning pipeline.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove version numbers (e.g., "1.2.3", "v2.0")
    5. Remove code snippets in backticks
    6. Remove special characters but keep spaces
    7. Remove extra whitespace
    8. Remove stop words
    9. Filter very short tokens
    """
    if pd.isna(text) or text == '':
        return ''

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http://, https://, ftp://, www.)
    text = re.sub(r'http\S+|www\.\S+|ftp\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove version numbers (e.g., "v1.2.3", "1.2.3", "version 2.0")
    text = re.sub(r'\bv?\d+\.\d+(\.\d+)*\b', ' ', text)
    text = re.sub(r'\bversion\s+\d+(\.\d+)*\b', ' ', text)

    # Remove code snippets in backticks
    text = re.sub(r'`[^`]*`', ' ', text)

    # Remove CVE IDs (e.g., CVE-2021-1234)
    text = re.sub(r'\bcve-\d{4}-\d+\b', ' ', text)

    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # Filter very short tokens (less than 3 characters) and very long (> 20, likely garbage)
    tokens = [word for word in tokens if 3 <= len(word) <= 20]

    # Rejoin tokens
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Load descriptions dataset
print("\n📂 Loading descriptions dataset...")
df = pd.read_csv('embedding_pipeline/data/descriptions_with_labels.csv')
print(f"   Loaded {len(df):,} records")

# Show sample before preprocessing
print("\n📝 Sample BEFORE preprocessing:")
for idx in range(min(3, len(df))):
    print(f"   [{df.iloc[idx]['id']}]: {df.iloc[idx]['description'][:120]}...")

# Apply preprocessing
print("\n🔄 Applying text preprocessing pipeline...")
print("   Steps: lowercase → remove URLs/emails → remove versions → remove code →")
print("          remove special chars → remove stop words → filter short tokens")

df['description_original'] = df['description']
df['description_cleaned'] = df['description'].apply(clean_text)

# Check for empty descriptions after cleaning
empty_after_cleaning = (df['description_cleaned'].str.strip() == '').sum()
print(f"\n   Empty descriptions after cleaning: {empty_after_cleaning}")

if empty_after_cleaning > 0:
    print(f"   Removing {empty_after_cleaning} records with no content after cleaning...")
    df = df[df['description_cleaned'].str.strip() != ''].copy()

# Calculate cleaning statistics
print(f"\n📊 Preprocessing statistics:")
avg_original_length = df['description_original'].str.split().str.len().mean()
avg_cleaned_length = df['description_cleaned'].str.split().str.len().mean()
reduction_pct = ((avg_original_length - avg_cleaned_length) / avg_original_length) * 100

print(f"   Original avg length: {avg_original_length:.1f} words")
print(f"   Cleaned avg length: {avg_cleaned_length:.1f} words")
print(f"   Average reduction: {reduction_pct:.1f}%")

# Show sample after preprocessing
print("\n📝 Sample AFTER preprocessing:")
for idx in range(min(3, len(df))):
    print(f"   [{df.iloc[idx]['id']}]: {df.iloc[idx]['description_cleaned'][:120]}...")

# Analyze vocabulary
print("\n📚 Vocabulary analysis:")
all_tokens = ' '.join(df['description_cleaned'].values).split()
vocab = set(all_tokens)
word_freq = Counter(all_tokens)
most_common = word_freq.most_common(20)

print(f"   Total tokens: {len(all_tokens):,}")
print(f"   Unique tokens (vocabulary size): {len(vocab):,}")
print(f"   Top 20 most common words:")
for word, count in most_common:
    print(f"      {word}: {count:,} occurrences")

# Save preprocessed dataset
output_path = 'embedding_pipeline/data/descriptions_preprocessed.csv'
print(f"\n💾 Saving preprocessed dataset...")

# Keep necessary columns
output_df = df[['id', 'year', 'description_cleaned', 'target']].copy()
output_df.rename(columns={'description_cleaned': 'description'}, inplace=True)

output_df.to_csv(output_path, index=False)
import os
file_size = os.path.getsize(output_path) / 1024**2

print(f"\n✅ Preprocessing complete!")
print(f"   Output: {output_path}")
print(f"   Records: {len(output_df):,}")
print(f"   File size: {file_size:.2f} MB")
print(f"   Target distribution: {output_df['target'].sum():,} exploited ({output_df['target'].mean()*100:.2f}%)")

# Show year distribution after preprocessing
print(f"\n📊 Year distribution after preprocessing:")
year_dist = output_df['year'].value_counts().sort_index()
for year, count in year_dist.items():
    exploited = output_df[output_df['year'] == year]['target'].sum()
    print(f"   {year}: {count:,} CVEs ({exploited} exploited)")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE - Ready for embedding")
print("=" * 70)
