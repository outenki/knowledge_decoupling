'''
This script extracts the results from the LM evaluation JSON file and prints them in a human-readable format.
LM_EVAL Result example:
{
  "results": {
    "blimp_adjunct_island": {
      "name": "blimp_adjunct_island",
      "alias": "blimp_adjunct_island",
      "sample_len": 1000,
      "acc,none": 0.628,
      "acc_stderr,none": 0.015292149942040577
    },
    "blimp_anaphor_gender_agreement": {
      "name": "blimp_anaphor_gender_agreement",
      "alias": "blimp_anaphor_gender_agreement",
      "sample_len": 1000,
      "acc,none": 0.39,
      "acc_stderr,none": 0.015431725053866604
    },
    "blimp_anaphor_number_agreement": {
      "name": "blimp_anaphor_number_agreement",
      "alias": "blimp_anaphor_number_agreement",
      "sample_len": 1000,
      "acc,none": 0.554,
      "acc_stderr,none": 0.015726771166750357
    },
  }
}

extract result:
blimp_adjunct_island: 0.628
blimp_anaphor_gender_agreement: 0.39
blimp_anaphor_number_agreement: 0.554
...
'''

import json
import sys


input_json = sys.argv[1]
with open(input_json, 'r') as f:
    data = json.load(f)
results = data['results']
for key, value in results.items():
    acc = value['acc,none']
    print(f'{key}: {acc}')