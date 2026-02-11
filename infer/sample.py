import json
from collections import defaultdict
import random
import re

file_pre = 'imagenet_left_grpo_qwen2_5vl_3b_cold_img'
output_file = "./data/imagenet_middle_sft_3b.json"
json_files = [f"./infer/rollout_output/{file_pre}_{i}.json" for i in range(1,9)]

random.seed(42)
index_data = defaultdict(list)

all_data = []
for file_name in json_files:
    with open(file_name, 'r') as f:
        data = json.load(f)
        all_data.append(data['results'])

middle_indices = []
hard_indices = []
easy_indices = []
all_indices = set(range(len(all_data[0])))

# Ensure all result lists have the same length before proceeding.
num_items = len(all_data[0])
if not all(len(data) == num_items for data in all_data):
    print("Error: JSON files do not contain the same number of results.")
    exit()

for i in range(len(all_data[0])):
    correct_values = [data[i]['correct'] for data in all_data]
    if len(set(correct_values)) > 1:
        middle_indices.append(i)
    else:
        if correct_values[0]==0:
            hard_indices.append(i)
        else:
            easy_indices.append(i)

output_indices = middle_indices

output_data = []

if 'ref' in file_pre:
    for i in output_indices:
        bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
        sol_match = re.search(bbox_pattern, all_data[0][i]['ground_truth'])
        bbox_sol = [float(sol_match.group(1)), float(sol_match.group(2)), float(sol_match.group(3)), float(sol_match.group(4))]
        rounded_solution = [int(x) for x in bbox_sol]
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>{all_data[0][i]['question']}"
                },
                {
                    "role": "assistant",
                    "content": f"[{rounded_solution[0]}, {rounded_solution[1]}, {rounded_solution[2]}, {rounded_solution[3]}]"
                }
            ],
            "images": [all_data[0][i]['image']]
        }
        output_data.append(item)
else:
    for i in output_indices:
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>{all_data[0][i]['question']}"
                },
                {
                    "role": "assistant",
                    "content": all_data[0][i]['ground_truth']
                }
            ],
            "images": [all_data[0][i]['image']]
        }
        output_data.append(item)

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Found {len(output_indices)} items. Saved to {output_file}.")