# Data Generation Pipeline

This folder contains the complete pipeline for generating the vulnerability dataset from raw CVE data and the NIST NVD API.

## Overview

The data generation process consists of three sequential steps:

1. **`read_nvd_api.py`** - Collects CVE data from NIST NVD API
2. **`merge_files.py`** - Merges annual vulnerability files into a single dataset  
3. **`pull_description.py`** - Extracts and adds vulnerability descriptions

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
cd generate_data
python create_vulnerabilities.py
```

This will run all three scripts in the correct sequence and generate the final `../data/vulnerabilities.csv` file.

**⚠️ Warning**: The NVD API collection step may take several hours to complete due to API rate limits.

### Option 2: Run Individual Scripts

If you need to run scripts individually or restart from a specific step:

```bash
cd generate_data

# Step 1: Collect CVE data from NVD API (takes hours)
python read_nvd_api.py

# Step 2: Merge annual files
python merge_files.py

# Step 3: Add descriptions
python pull_description.py
```

## Prerequisites

1. **CVE Data**: Ensure you have downloaded the CVE data and placed it in the `../cves/` directory
   - Download from: https://www.cve.org/Downloads
   - Or from UNC Longleaf: longleaf.unc.edu/proj/zefflab/cves
   - Expected structure: `../cves/YYYY/XXXX/CVE-YYYY-XXXX.json`

2. **Known Exploited Vulnerabilities**: Ensure `../data/known_exploited_vulnerabilities.csv` exists
   - This file should contain the CISA KEV catalog

3. **Python Dependencies**: 
   ```bash
   pip install requests pandas numpy
   ```

## Output Files

The pipeline generates:

- `../data/vulnerabilities_YYYY.csv` - Annual vulnerability files (from step 1)
- `../data/vulnerabilities.csv` - Final merged dataset with descriptions

## Script Details

### `read_nvd_api.py`
- Reads CVE JSON files from `../cves/` directory
- Queries NIST NVD API for vulnerability scores and metadata
- Handles API rate limiting (50 requests per 30 seconds)
- Outputs annual CSV files to `../data/vulnerabilities_YYYY.csv`

### `merge_files.py`
- Combines all annual vulnerability files into a single dataset
- Outputs merged data to `../data/vulnerabilities.csv`

### `pull_description.py`
- Reads the merged vulnerability dataset
- Extracts vulnerability descriptions from original CVE JSON files
- Updates the merged dataset with description column

### `create_vulnerabilities.py`
- Orchestrates the complete pipeline
- Handles error checking between steps
- Provides progress updates and timing information

## Notes

- All scripts are designed to work from the `generate_data/` subdirectory
- File paths are adjusted to reference parent directories (`../data/`, `../cves/`)
- The pipeline can be interrupted and restarted at any step
- API key is included in `read_nvd_api.py` for higher rate limits

## Troubleshooting

**Rate Limiting**: If you encounter 403 errors, the script will automatically wait 30 seconds and retry.

**Missing Files**: Ensure the CVE data is properly downloaded and the directory structure matches expectations.

**Memory Issues**: For large datasets, consider running scripts with increased memory limits or processing smaller year ranges.

## Next Steps

After generating `../data/vulnerabilities.csv`, you can proceed with:

1. **Data Cleaning & Modeling**: Run `../baseline_model_abel_koshy_07_25.py`
2. **Notebook Analysis**: Explore the Jupyter notebooks in `../notebooks/Baseline_Model/` 