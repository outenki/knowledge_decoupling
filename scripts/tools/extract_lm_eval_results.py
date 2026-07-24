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

metrics = {
    "squad_completion": "contains,none",
    "boolq": "acc,none",
    "race": "acc,none",
    "arc_easy": "acc_norm,none",
    "squadv2_core": "f1,none",
    "boolq_local": "acc_norm,none",
    "google_boolq_core": "acc_norm,none",
    "arc_challenge": "acc,none",
    "commonsense_qa_norm": "acc,none",
    "triviaqa_rc_context": "exact_match,remove_whitespace",
    "triviaqa_rc_context_core": "exact_match,remove_whitespace",
    "triviaqa": "exact_match,remove_whitespace",
    "drop": "f1,none"
}

input_json = sys.argv[1]
metric = sys.argv[2]
avg = False
if len(sys.argv) >= 4 and sys.argv[3] == "avg":
    avg=True
with open(input_json, 'r') as f:
    data = json.load(f)
results = data['results']
acc_list = []
for key, value in results.items():
    _metric = metric
    if metric == "none":
        _metric = metrics[key]
    
    acc = value.get(_metric, "N/A")
    if acc != "N/A":
        if acc > 1:
            acc = acc / 100
        acc_list.append(acc)
    print(f'{key}: {acc}')

if avg:
    print(f"ewok: {sum(acc_list)/len(acc_list)}")
