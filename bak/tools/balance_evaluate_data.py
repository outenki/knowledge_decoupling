# make the number of samples whose result is true or false the smae
import json
import sys
import random

f_input_json = sys.argv[1]
f_output_json = sys.argv[2]

# load data
with open(f_input_json, 'r') as f:
    input_json = json.load(f)


samples_yes = [d for d in input_json if d['answer'] == 'Yes.']
samples_no = [d for d in input_json if d['answer'] == 'No.']

num = min(len(samples_no), len(samples_yes))

samples_yes = samples_yes[:num]
samples_no = samples_no[:num]

output_json = random.sample(samples_no + samples_yes, 2 * num)

with open(f_output_json, 'w') as f:
    json.dump(output_json, f, indent=2)

print("input samples:", len(input_json))
print("output samples:", len(output_json))