import os
import json
import pandas as pd

# Load the merged vulnerabilities file
if not os.path.exists('vulnerabilities.csv'):
    print("Error: vulnerabilities.csv not found. Please run merge_files.py first.")
    exit(1)

df = pd.read_csv('vulnerabilities.csv', low_memory=False)
target_cves = set(df['id'])
cve_descriptions = {}

print(f"Processing {len(target_cves)} CVEs for descriptions...")

# Loop through each year directory to find JSON files
for year in range(1999, 2026):
    root_dir = f'cves/{year}'
    
    # Skip if JSON directory doesn't exist
    if not os.path.exists(root_dir):
        continue

    print(f"Searching year {year} directory...")

    # Walk through JSON files
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
                pass

# Add description column to the dataframe
df['description'] = df['id'].map(cve_descriptions)

# Save the updated dataset with descriptions
df.to_csv('vulnerabilities.csv', index=False)
print(f"\nUpdated dataset with descriptions saved to: vulnerabilities.csv")
print(f"Found descriptions for {len(cve_descriptions)} out of {len(target_cves)} CVEs")