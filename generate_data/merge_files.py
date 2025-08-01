import os
import pandas as pd

all_data = []

# Loop through each year
for year in range(1999, 2026):
    csv_path = f'../data/vulnerabilities_{year}.csv'

    # Skip if CSV doesn't exist
    if not os.path.exists(csv_path):
        print(f"Skipping {year} — CSV not found.")
        continue

    # Load CSV 
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Append to full list
    all_data.append(df)
    print(f"Processed year: {year} with {len(df)} entries.")

# Combine all years into a single DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Save the merged dataset - adjusted path for subdirectory
combined_df.to_csv('../data/vulnerabilities.csv', index=False)
print(f"\nMerged dataset saved to: ../data/vulnerabilities.csv") 