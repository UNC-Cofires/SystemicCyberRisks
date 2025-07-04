import os
import json
import pandas as pd

all_data = []

# Loop through each year
for year in range(1999, 2026):
    csv_path = f'vulnerabilities_{year}.csv'
    root_dir = f'cves/{year}'

    # Skip if CSV or JSON directory doesn't exist
    if not os.path.exists(csv_path):
        print(f"Skipping {year} — CSV not found.")
        continue
    if not os.path.exists(root_dir):
        print(f"Skipping {year} — JSON directory not found.")
        continue

    # Load CSV 
    df = pd.read_csv(csv_path, low_memory=False)
    target_cves = set(df['id'])
    cve_descriptions = {}

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

    # Add description
    df['description'] = df['id'].map(cve_descriptions)

    # Append to full list
    all_data.append(df)
    print(f"Processed year: {year} with {len(df)} entries.")

# Combine all years into a single DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Save the full dataset
combined_df.to_csv('vulnerabilities.csv', index=False)
print(f"\nCombined dataset saved to: vulnerabilities.csv")