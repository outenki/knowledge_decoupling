'''
{
    "doc_id": 0,
    "doc": {
        "question": "does ethanol take more energy make that produces",
        "answer": false,
        "passage": "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.",
        "target": "no"
    },
    "target": "no",
    "arguments": {
        "gen_args_0": {
            "arg_0": "Background: All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.\n\nQuestion: does ethanol take more energy make that produces?\n\nAnswer:",
            "arg_1": " no"
        },
        "gen_args_1": {
            "arg_0": "Background: All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.\n\nQuestion: does ethanol take more energy make that produces?\n\nAnswer:",
            "arg_1": " yes"
        }
    },
    "resps": [
        [
            [
                "-3.78125",
                "False"
            ]
        ],
        [
            [
                "-3.78125",
                "False"
            ]
        ]
    ],
    "filtered_resps": [
        [
            "-3.78125",
            "False"
        ],
        [
            "-3.78125",
            "False"
        ]
    ],
    "filter": "none",
    "metrics": [
        "acc",
        "f1",
        "acc_norm"
    ],
    "doc_hash": "f51e909050ef1699d4dbae89a986f509f516cfd6b901c151ad19f3ea4ff57955",
    "prompt_hash": "5c04b34f3b064be1baf4b0f6756dcaeaf756deb5e2460a0a42e38ed4449b212b",
    "target_hash": "9390298f3fb0c5b160498935d79cb139aef28e1c47358b4bbba61862b9c26e59",
    "acc": 1.0,
    "f1": [
        0,
        0
    ],
    "acc_norm": 0.0
}
'''

import json
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm


in_json = sys.argv[1]

output_path = Path(in_json).parent
output_stem = Path(in_json).stem
in_name = Path(in_json).name

samples = []
with open(in_json, "r") as f:
    for line in tqdm(f, desc=f"Loading data from {in_name}", total=sum(1 for _ in open(in_json, 'r'))):
      samples.append(json.loads(line))

extracted = []
for spl in tqdm(samples, total=len(samples), desc="Extracting samples"):
    prob_0 = spl["resps"][0][0][0]
    prob_1 = spl["resps"][1][0][0]
    if prob_1 > prob_0:
        response = spl["arguments"]["gen_args_1"]["arg_1"]
    else:
        response = spl["arguments"]["gen_args_0"]["arg_1"]
    extracted.append({
        "doc_id": spl["doc_id"],
        "prompt": spl["arguments"]["gen_args_0"]["arg_0"],
        "target": spl["target"],
        "response": response,
        "filtered_resps": spl["filtered_resps"][0]
    })


# save json samples
output_fn = output_path/f"simpled_{output_stem}.json"
print(f"Saving json samples to {output_fn} ...")
with open(output_fn, "w") as f:
  json.dump(extracted, f, indent=4)


# save csv samples
output_fn = output_path/f"simpled_{output_stem}.csv"
print(f"Saving csv samples to {output_fn} ...")
df = pd.DataFrame(extracted)
df.to_csv(output_fn, index=False)