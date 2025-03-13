# SystemicCyberRisks
Contains code to evaluate systemic cyber risks using data from MITRE CVE and NIST NVD databases     

The MITRE CVE is a list of critical cybersecurity vulnerabilities that is updated daily      

The NIST NVD takes those vulnerabilities and assigns them a 'severity' score based expert judgement

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

* First, download data from the cve vulnerabilties list, either from the CVE website 
** (https://www.cve.org/Downloads) 
* or from the UNC Longleaf Computing Cluster
** longleaf.unc.edu/proj/zefflab/cves
* This dataset is ~2GB but there are ~200,000 small files so download speeds will be slow
* When the data is in this repository, the pathways should be set updated such that the path to the first CVE in 1999 is:
** SystemicCyberRisks/cves/1999/0xxx/CVE-1999-0001
* Once the data is in the repository, call the NVD API by running:
```
python -W ignore read_nvd_api.py
```
* This will read through all the CVE IDs, and write them to annual files (i.e., one output file per year)
* To run a single year (example - 2024) run:
```
python -W ignore read_nvd_api.py 2024
```

