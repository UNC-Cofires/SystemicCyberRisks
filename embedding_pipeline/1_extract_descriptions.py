#!/usr/bin/env python3
"""
Extract Descriptions Dataset
=============================
Extracts CVE descriptions along with their IDs and target labels (KEV status)
from the raw vulnerability dataset for embedding-based modeling.

Input:  data/vulnerabilities.csv.gz (raw vulnerability dataset)
        data/known_exploited_vulnerabilities.csv (KEV catalog)
Output: embedding_pipeline/data/descriptions_with_labels.csv
"""

import pandas as pd
import os
import gzip

print("=" * 70)
print("STEP 1: EXTRACT DESCRIPTIONS DATASET")
print("=" * 70)

# Create output directory
os.makedirs('embedding_pipeline/data', exist_ok=True)

# Load the raw vulnerability dataset
print("\n📂 Loading vulnerability dataset...")
if os.path.exists('data/vulnerabilities.parquet'):
    df = pd.read_parquet('data/vulnerabilities.parquet')
elif os.path.exists('data/vulnerabilities.csv'):
    df = pd.read_csv('data/vulnerabilities.csv', low_memory=False)
else:
    raise FileNotFoundError("Vulnerability dataset not found! Run generate_data pipeline first.")

print(f"   Loaded {len(df):,} vulnerability records")

# Extract year from CVE ID
print("\n📅 Extracting year from CVE IDs...")
df['year'] = df['id'].str.extract(r'CVE-(\d{4})-')[0].astype(int)
print(f"   Year range: {df['year'].min()}-{df['year'].max()}")

# Filter to 2015-2025 as specified
print("\n🔄 Filtering to 2015-2025...")
df = df[(df['year'] >= 2015) & (df['year'] <= 2025)].copy()
print(f"   Filtered to {len(df):,} records")

# Load Known Exploited Vulnerabilities catalog
print("\n🎯 Loading KEV catalog for target labels...")
kev_df = pd.read_csv('data/known_exploited_vulnerabilities.csv')
kev_set = set(kev_df['cveID'].values)
print(f"   Loaded {len(kev_set):,} known exploited vulnerabilities")

# Create target variable
df['target'] = df['id'].apply(lambda x: 1 if x in kev_set else 0)
exploit_count = df['target'].sum()
print(f"   Found {exploit_count:,} exploited CVEs ({exploit_count/len(df)*100:.2f}%)")

# Check for missing descriptions
print("\n🔍 Checking description completeness...")
missing_desc = df['description'].isna().sum()
print(f"   Missing descriptions: {missing_desc:,} ({missing_desc/len(df)*100:.2f}%)")

# Remove records without descriptions (can't embed empty text)
df_with_desc = df[df['description'].notna() & (df['description'].str.strip() != '')].copy()
removed = len(df) - len(df_with_desc)
print(f"   Removed {removed:,} records without descriptions")
print(f"   Final dataset: {len(df_with_desc):,} records")

# Check target distribution after filtering
final_exploited = df_with_desc['target'].sum()
print(f"\n📊 Final target distribution:")
print(f"   Exploited (target=1): {final_exploited:,} ({final_exploited/len(df_with_desc)*100:.2f}%)")
print(f"   Not exploited (target=0): {len(df_with_desc)-final_exploited:,}")

# Create clean dataset with only needed columns
print("\n💾 Creating descriptions dataset...")
descriptions_df = df_with_desc[['id', 'year', 'description', 'target']].copy()

# Save to CSV
output_path = 'embedding_pipeline/data/descriptions_with_labels.csv'
descriptions_df.to_csv(output_path, index=False)
file_size = os.path.getsize(output_path) / 1024**2

print(f"\n✅ Descriptions dataset created successfully!")
print(f"   Output: {output_path}")
print(f"   Records: {len(descriptions_df):,}")
print(f"   File size: {file_size:.2f} MB")
print(f"   Columns: {list(descriptions_df.columns)}")

# Show year distribution
print(f"\n📊 Year distribution:")
year_dist = descriptions_df['year'].value_counts().sort_index()
for year, count in year_dist.items():
    exploited_in_year = descriptions_df[descriptions_df['year'] == year]['target'].sum()
    print(f"   {year}: {count:,} CVEs ({exploited_in_year} exploited)")

# Show sample descriptions
print(f"\n📋 Sample descriptions:")
for idx, row in descriptions_df.head(3).iterrows():
    desc_preview = row['description'][:100] + "..." if len(row['description']) > 100 else row['description']
    print(f"   {row['id']} (target={row['target']}): {desc_preview}")

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE - Ready for preprocessing")
print("=" * 70)
