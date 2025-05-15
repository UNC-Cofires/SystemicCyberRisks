import os
import json
import pandas as pd

# Load CSV and normalize CVE IDs
csv_path = 'vulnerabilities_2025.csv'
df = pd.read_csv(csv_path)
df['id'] = df['id'].astype(str).str.strip().str.upper()

# Directory containing all CVE folders and JSON files
root_dir = 'cves/2025'

# Collect descriptions from JSON files
cve_descriptions = {}
target_cves = set(df['id'])

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if not file.endswith('.json'):
            continue

        cve_id = file.replace('.json', '').strip().upper()
        if cve_id not in target_cves:
            continue

        try:
            with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                descriptions = data.get("containers", {}).get("cna", {}).get("descriptions", [])
                value = next((desc.get("value", "") for desc in descriptions if desc.get("value")), "")
                cve_descriptions[cve_id] = value
        except:
            pass  # Skip files that can't be read

# Add description to DataFrame and save
df['description'] = df['id'].map(cve_descriptions)
df.to_csv('vulnerabilities_2025_with_descriptions.csv', index=False)