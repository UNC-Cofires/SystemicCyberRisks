import pandas as pd 

# Read the  CSV file and drop unwanted columns
df = pd.read_csv('vulnerabilities.csv', low_memory=False)
df = df.drop(columns=['Unnamed: 0', 'userInteractions', 'accessVector', 'accessComplexity',
                      'authentication', 'attackRequirements', 'vulnConfidentialityImpact',
                      'vulnIntegrityImpact', 'vulnAvailabilityImpact','subConfidentialityImpact', 
                      'subIntegrityImpact','subAvailabilityImpact', 'exploitMaturity',
                      'confidentialityRequirement', 'integrityRequirement', 'availabilityRequirement', 
                      'modifiedAttackVector', 'modifiedAttackComplexity', 'modifiedAttackRequirements',
                      'modifiedPrivilegesRequired', 'modifiedUserInteraction', 'modifiedVulnConfidentialityImpact', 
                      'modifiedVulnIntegrityImpact', 'modifiedVulnAvailabilityImpact', 'modifiedSubConfidentialityImpact', 
                      'modifiedSubIntegrityImpact', 'modifiedSubAvailabilityImpact', 'Safety','Automatable', 'Recovery', 
                      'valueDensity', 'vulnerabilityResponseEffort', 'providerUrgency', 'vectorString'])

# Drop version 4, and keep only 1 of each CveID
df = df[df['version'] != 4.0].copy()
df = df.drop_duplicates(subset=['id'], keep='last')

# Extract the year from the 'id' column
df['year'] = df['id'].str.extract(r'CVE-(\d{4})-')[0].astype(int)

# drop everyhting from 2015 and before
df = df[df['year'] > 2015]

# read targets
dt = pd.read_csv('known_exploited_vulnerabilities.csv')

# Extract the year from the 'id' column in targets
df['target'] = df['id'].apply(lambda x: 1 if x in dt['cveID'].values else 0)

# fill null values 
df.fillna(0, inplace=True)

# drop remaining unwanted columns
df = df.drop(columns=['version', 'year'])

# save the cleaned data to a new CSV file
df.to_csv('data.csv', index=False)