import requests
import json
import pandas as pd
import os
import time
import sys
import numpy as np

## This code takes json data from the CVE list
## cveV5: (https://www.cve.org/Downloads)
## and uses the cve_id associated with each
## item to make an API query the NIST NVD 
## Vulnerability list: https://nvd.nist.gov/developers/start-here
## to link the CVE data with vulnerability scores
## which reflect expert judgement on the severity
## of the vulnerability

# cve data is formatted in json files
# and organized by year - command line
# inputs specify which year to use
# if no year is specified, use full range
try:
  years_to_query = [sys.argv[1], ]
except:
  years_to_query = np.arange(1999, 2026)  
cve_dir = '../cves' # directory with cveV5 data - adjusted for subdirectory

# API key (allows 50 queries every 30 seconds instead of 5)
API_KEY = 'c144f37f-5d22-4998-8dfb-af9338ecfa15'
# url to make the api request
nvd_url = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
# parameter add-on to the API to query a specific CVE ID
cveIDparam = '?cveId='
headers = {'content-type': 'application/json', 'apiKey': API_KEY}
# data columns to pull from NVD database
col_list = ['id', 'baseScoreAv', 'exploitScoreAv', 'impactScoreAv', 
            'baseScoreMax', 'exploitScoreMax', 'impactScoreMax', 
            'version', 'vectorString', 'baseSeverity', 'attackVector', 
            'attackComplexity', 'privilegesRequired', 'userInteractions', 'scope',
            'confidentialityImpact', 'integrityImpact', 'availabilityImpact',
            'numScores', 'agreement']

# loop through each selected year
for year in years_to_query:
  # generate path to this year's CVE data
  year_path = os.path.join(cve_dir, str(year))
  # create new dataframe to store NVD data
  this_year_file = pd.DataFrame(columns = col_list)
  cve_idx = 0 # cve counter
  # each year directory contains a subdirectory with ~100 json files
  # we do not need to make a distinction between these subdirectories
  # so they can all go into the same DataFrame - loop through all
  for dir_name in os.listdir(year_path):
    dir_path = os.path.join(year_path, dir_name) #subdirectory path
    if os.path.isdir(dir_path):
      all_cves = os.listdir(dir_path) # list all json CVE entries
      for cve_name in all_cves:
        cve_pathway = os.path.join(dir_path, cve_name) # path to CVE data
        # open CVE json data file
        with open(cve_pathway, 'r', errors = 'replace') as cve_file:
          cve_data = json.load(cve_file) # load json as dictionary
          # only record data from CVEs that have been 'published'
          # some records are rejected - need to understand what this means
          if cve_data['cveMetadata']['state'] == 'PUBLISHED':
            # get CVE ID from json file to link with NVD API
            cve_id = cve_data['cveMetadata']['cveId']
            print(cve_id, end = " ")
            print(cve_idx)
            # response query to NVD API for a specific CVE ID
            response = requests.get(nvd_url + cveIDparam + cve_id, headers=headers)
            # more than 50 requests in 30 seconds will cause API request to errors
            # i.e. - there is a rate limit to the NVD API
            if response.status_code == 403:
              # if you get rate limited, wait 30 seconds then 
              # make a new request
              print(response.status_code, end = " ")
              print(len(this_year_file.index))
              time.sleep(30) # wait 30 seconds
              response = requests.get(nvd_url + cveIDparam + cve_id, headers = headers)
            # load the json data that came back from the NVD API request              
            json_data = json.loads(response.text)
            # record NVD data into DataFrame
            # each entry can have multiple 'scores' from different versions
            # of the CVSS system - record each system scoring as a new entry
            # loops through each of the 'score types'
            for st in json_data['vulnerabilities'][0]['cve']['metrics']:
              # for each CVSS version type, multiple groups can generate a score
              # we take the max and average of the different group scores
              # (note: these are multiple groups making a score with the same
              # CVSS version - a new version gets a completely new entry)
              list_of_scores = json_data['vulnerabilities'][0]['cve']['metrics'][st]
              # initialize a average/max score for each category
              # categories include 'base score' (bsA/bsM)
              # 'exploitability score' (esA/esM)
              # and 'impact score' (isA/isM)
              bsA = 0.0
              bsM = 0.0
              esA = 0.0
              esM = 0.0
              isA = 0.0
              isM = 0.0
              totB = 0.0
              totE = 0.0
              totI = 0.0
              # we also want a column to keep track of the 
              # agreement between different group scores on
              # the non-numeric assessment categories
              # instead of keeping track of all of them, we
              # record the value from the first listed assessment
              # and then add one to the 'agreement' value each
              # time there is a disagreement between categories
              # need to initialize this value to zero before
              # we start to keep track
              this_year_file.loc[cve_idx, 'agreement'] = 0
              # save the CVE ID 
              this_year_file.loc[cve_idx, 'id'] = cve_id
              # loop through multiple scores from different
              # groups
              for x in range(0, len(list_of_scores)):
                # base score - max & average
                try:
                  bsA += list_of_scores[x]['cvssData']['baseScore']
                  bsM = max(bsM, list_of_scores[x]['cvssData']['baseScore'])
                  totB += 1.0
                except:
                  pass
                # exploitability score - max & average
                try:
                  esA += list_of_scores[x]['exploitabilityScore']         
                  esM = max(esM, list_of_scores[x]['exploitabilityScore'])
                  totE += 1.0
                except:
                  pass
                # impact score - max & average
                try:
                  isA += list_of_scores[x]['impactScore']            
                  isM = max(isM, list_of_scores[x]['impactScore'])                
                  totI += 1.0
                except:
                  pass
                # loop through all the non-numeric
                # vulnerability assessment categories
                for col in list_of_scores[x]['cvssData']:
                  if col != 'baseScore':
                    # record only the first group's non-numeric scores
                    if x == 0:
                      this_year_file.loc[cve_idx, col] = list_of_scores[x]['cvssData'][col]
                    else:
                      # for the rest of the groups, just keep track of how many times
                      # their assessment differed from the first group
                      if list_of_scores[x]['cvssData'][col] != this_year_file.loc[cve_idx, col]:
                        this_year_file.loc[cve_idx, 'agreement'] += 1
              # record max/average scores
              if totB>0:
                this_year_file.loc[cve_idx, 'baseScoreAv'] = bsA / totB
                this_year_file.loc[cve_idx, 'baseScoreMax'] = bsM * 1.0
              if totE>0:
                this_year_file.loc[cve_idx, 'exploitScoreAv'] = esA / totE           
                this_year_file.loc[cve_idx, 'exploitScoreMax'] = esM * 1.0
              if totI>0:                
                this_year_file.loc[cve_idx, 'impactScoreAv'] = isA / totI            
                this_year_file.loc[cve_idx, 'impactScoreMax'] = isM * 1.0
              this_year_file.loc[cve_idx, 'numScores'] = len(list_of_scores)
              # increase cve index value by one
              # this creates a new entry in the 
              # vulnerability score dataframe
              cve_idx += 1
          else:
            # print out non-published CVEs (should be 'REJECTED')
            print(cve_data['cveMetadata']['state'])
  # save dataframe for each year - adjusted path for subdirectory
  this_year_file.to_csv('../data/vulnerabilities_' + str(year) + '.csv') 