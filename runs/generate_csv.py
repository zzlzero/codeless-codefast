import json
import csv
import os

runs = [
    "CodeLlama-7b-hf-gamma-0p1",
    "CodeLlama-7b-hf-gamma-0p5",
    "CodeLlama-7b-hf-gamma-1p0",
    "CodeLlama-7b-hf-gamma-2p0"
]

base_dir = "/data/zzl/codeless-codefast/runs"
all_data = {}

for run in runs:
    state_file = os.path.join(base_dir, run, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            data = json.load(f)
        for log in data.get('log_history', []):
            step = log.get('step')
            if step is None: continue
            if step not in all_data:
                all_data[step] = {}
            for k, v in log.items():
                if isinstance(v, (int, float)):
                    all_data[step][f"{run}_{k}"] = v

with open(os.path.join(base_dir, "combined_training_curves.csv"), "w", newline='') as f:
    if not all_data:
        writer = csv.writer(f)
        writer.writerow(["No data found"])
    else:
        steps = sorted(list(all_data.keys()))
        keys = set()
        for step in steps:
            keys.update(all_data[step].keys())
        keys = ["step"] + sorted(list(keys))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for step in steps:
            row = {"step": step}
            row.update(all_data[step])
            writer.writerow(row)

print("Generated " + os.path.join(base_dir, "combined_training_curves.csv"))
