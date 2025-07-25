# SystemicCyberRisks
Contains code to evaluate systemic cyber risks using data from MITRE CVE and NIST NVD databases     

The MITRE CVE is a list of critical cybersecurity vulnerabilities that is updated daily      

The NIST NVD takes those vulnerabilities and assigns them a 'severity' score based expert judgement

## Repository Structure

This repository contains the following components:

### Data Collection Scripts
- `read_nvd_api.py` - Main script to collect CVE data from NIST NVD API

### Modeling Pipeline
- `modeling/merge_files.py` - Merges multiple data sources
- `modeling/data_cleaning.py` - Cleans and preprocesses data
- `modeling/baseline_model.py` - Creates baseline machine learning model
- `modeling/pull_description.py` - Extracts vulnerability descriptions

### Analysis and Exploration
- `notebooks/EDA.ipynb` - Exploratory Data Analysis notebook for understanding the vulnerability dataset
- `notebooks/Baseline_Model/` - Collection of modeling notebooks for comprehensive analysis:
  - `0_Data_Loading_and_EDA.ipynb` - Data loading and exploratory data analysis
  - `1_Data_Preprocessing.ipynb` - Data preprocessing and feature engineering
  - `2_Baseline_Modeling.ipynb` - Baseline model development
  - `3_Advanced_Modeling.ipynb` - Advanced modeling techniques
  - `4_Cross_Validation.ipynb` - Model validation and performance evaluation


## Getting Started
 This script uses data from the CVE list (cveV5)   
 * (https://www.cve.org/Downloads)    
 
 and uses the cve_id associated with each item to make an API query the NIST NVD    
 
 * https://nvd.nist.gov/developers/start-here    
 
 This will link the CVE data with vulnerability scores which reflect expert judgement on the severity of the software vulnerability

### Dependencies

Python Libraries:

* requests
* json
* pandas
* numpy
* sys
* os
* time

### Executing program

First, download data from the cve vulnerabilties list, either from the CVE website   

* (https://www.cve.org/Downloads)     

or from the UNC Longleaf Computing Cluster    

* longleaf.unc.edu/proj/zefflab/cves    

This dataset is ~2GB but there are ~200,000 small files so download speeds will be slow    

When the data is in this repository, the pathways should be set updated such that the path to the first CVE in 1999 is:   

* SystemicCyberRisks/cves/1999/0xxx/CVE-1999-0001   

Once the data is in the repository, call the NVD API by running:   

```
python -W ignore read_nvd_api.py
```
This will read through all the CVE IDs, and write them to annual files (i.e., one output file per year)    

To run a single year (example - 2024) run:   

```
python -W ignore read_nvd_api.py 2024
```

## Creating the Baseline Model

**Important:** Before running any modeling scripts, ensure that you have completed the data collection process using `read_nvd_api.py` and that all individual year files have been created.

To generate the baseline model for systemic cyber risk evaluation, follow these steps in order:

1. **First, merge the data files (required for all subsequent steps):**
   ```
   python modeling/merge_files.py
   ```

2. **Clean and preprocess the data:**
   ```
   python modeling/data_cleaning.py
   ```

3. **Generate the baseline model:**
   ```
   python modeling/baseline_model.py
   ```

These scripts should be run in sequence as each step depends on the output of the previous step.

## Using the Modeling Notebooks

After completing the baseline model creation (specifically after running `modeling/merge_files.py`), you can explore the comprehensive modeling notebooks located in `notebooks/Baseline_Model/`. These notebooks provide:

- Detailed exploratory data analysis
- Step-by-step data preprocessing workflows
- Multiple modeling approaches and comparisons
- Cross-validation and performance evaluation

Run the notebooks in numerical order (0 through 4) for the complete modeling workflow. These notebooks offer a more interactive and detailed approach to the modeling process compared to the standalone Python scripts.

