# SystemicCyberRisks

Contains code to evaluate systemic cyber risks using data from MITRE CVE and NIST NVD databases.

The MITRE CVE is a list of critical cybersecurity vulnerabilities that is updated daily.  
The NIST NVD takes those vulnerabilities and assigns them a 'severity' score based on expert judgement.

## Repository Structure

This repository contains the following components:

### Data Generation Pipeline
- `generate_data/` - Complete pipeline for generating vulnerability datasets:
  - `create_vulnerabilities_dataset.py` - Main orchestration script to run the entire pipeline
  - `read_nvd_api.py` - Collects CVE data from NIST NVD API
  - `merge_files.py` - Merges annual vulnerability files into a single dataset
  - `pull_description.py` - Extracts and adds vulnerability descriptions
  - `README.md` - Detailed documentation for the data generation process

### Data Storage
- `data/` - Directory containing all data files:
  - `vulnerabilities.csv.gz` - Compressed version of vulnerability dataset (from generate_data pipeline)
  - `known_exploited_vulnerabilities.csv` - CISA KEV catalog (target labels)

### Modeling and Analysis
- `modeling/baseline_model_abel_koshy_07_25.py` - Combined data cleaning and baseline modeling script
- `notebooks/` - Interactive analysis and exploration:
  - `EDA.ipynb` - Exploratory Data Analysis notebook
  - `Baseline_Model/` - Complete modeling workflow:
    - `0_Data_Loading_and_EDA.ipynb` - Data loading and initial analysis
    - `1_Data_Preprocessing.ipynb` - Feature engineering and preprocessing
    - `2_Baseline_Modeling.ipynb` - Baseline model development
    - `3_Advanced_Modeling.ipynb` - Advanced modeling techniques
    - `4_Cross_Validation.ipynb` - Model validation and performance evaluation

## Getting Started

### Prerequisites

1. **CVE Data**: Download the CVE data and place it in the `cves/` directory
   - Download from: https://www.cve.org/Downloads
   - Or from UNC Longleaf: longleaf.unc.edu/proj/zefflab/cves
   - Expected structure: `cves/YYYY/XXXX/CVE-YYYY-XXXX.json`

2. **Known Exploited Vulnerabilities**: Ensure `data/known_exploited_vulnerabilities.csv` exists
   - This file should contain the CISA KEV catalog for target labels

3. **Python Dependencies**:
   ```bash
   pip install requests pandas numpy matplotlib scikit-learn
   ```

### Workflow

#### 1. Generate Raw Vulnerability Dataset

```bash
cd generate_data
python create_vulnerabilities_dataset.py
```

This runs the complete data generation pipeline and creates `data/vulnerabilities.csv`.

**⚠️ Note**: The NVD API collection step may take several hours to complete due to API rate limits.

#### 2. Run Baseline Modeling (Quick Start)

```bash
python modeling/baseline_model_abel_koshy_07_25.py
```

This combines data cleaning and baseline modeling in a single script, producing:
- `data/data.csv` (cleaned dataset)
- `data/roc_curve.png` (model performance visualization)
- Console output with model evaluation metrics

#### 3. Interactive Analysis (Detailed Exploration)

For comprehensive analysis, use the Jupyter notebooks in order:

```bash
# Start with exploratory analysis
jupyter notebook notebooks/EDA.ipynb

# Follow the complete modeling pipeline
jupyter notebook notebooks/Baseline_Model/0_Data_Loading_and_EDA.ipynb
jupyter notebook notebooks/Baseline_Model/1_Data_Preprocessing.ipynb
jupyter notebook notebooks/Baseline_Model/2_Baseline_Modeling.ipynb
jupyter notebook notebooks/Baseline_Model/3_Advanced_Modeling.ipynb
jupyter notebook notebooks/Baseline_Model/4_Cross_Validation.ipynb
```

## Key Features

- **Automated Data Pipeline**: Complete automation from raw CVE data to processed datasets
- **NIST NVD Integration**: Real-time vulnerability scoring via API
- **Known Exploitation Labels**: Integration with CISA's Known Exploited Vulnerabilities catalog
- **Multiple Modeling Approaches**: From simple logistic regression to advanced ensemble methods
- **Class Imbalance Handling**: Specialized techniques for rare event prediction (~0.6% exploitation rate)
- **Comprehensive Evaluation**: ROC curves, precision-recall analysis, and cross-validation
- **Interactive Notebooks**: Step-by-step analysis with detailed explanations

## Data Flow

```
CVE JSON Files → generate_data pipeline → data/vulnerabilities.csv
                                      ↓
Known Exploited Vulnerabilities → modeling/baseline_model_abel_koshy_07_25.py → data/data.csv + models
                                      ↓
                              notebooks pipeline → processed datasets + advanced models
```

## Model Performance

The baseline logistic regression model achieves:
- **ROC-AUC**: ~0.87 (87% discriminative accuracy)
- **Precision-Recall**: Optimized for high recall on exploited vulnerabilities
- **Interpretability**: Clear feature importance rankings for vulnerability risk factors

## Output Files

All generated files are organized in the `data/` directory:
- Raw datasets: `vulnerabilities.csv`, `vulnerabilities_YYYY.csv`
- Processed datasets: `data.csv`, `processed_vulnerabilities_features.csv`
- Visualizations: `roc_curve.png`
- Model artifacts: Saved in `notebooks/Baseline_Model/models/`
- Results: Saved in `notebooks/Baseline_Model/results/`

## Repository Organization

This repository follows a clean, organized structure:
- **Data generation** is centralized in `generate_data/`
- **Modeling scripts** provide quick execution in `modeling/`
- **Interactive analysis** is available through `notebooks/`
- **All outputs** are consistently saved to `data/`

For detailed documentation on specific components, see the README files in respective directories.

