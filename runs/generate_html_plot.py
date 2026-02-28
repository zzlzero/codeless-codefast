import json
import os

runs = [
    "CodeLlama-7b-hf-gamma-0p1",
    "CodeLlama-7b-hf-gamma-0p5",
    "CodeLlama-7b-hf-gamma-1p0",
    "CodeLlama-7b-hf-gamma-2p0"
]

base_dir = "/data/zzl/codeless-codefast/runs"
metrics_to_plot = ["completion_length", "rewards/correct_code_reward_func", "rewards/len_reward_func"]

results = {metric: {} for metric in metrics_to_plot}
run_labels = [f"γ={r.split('-gamma-')[-1].replace('p', '.')}" for r in runs]
colors = ['red', 'blue', 'green', 'purple']

for run in runs:
    state_file = os.path.join(base_dir, run, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            data = json.load(f)
            
        for log in data.get('log_history', []):
            step = log.get('step')
            if step is None: continue
            for metric in metrics_to_plot:
                if metric in log:
                    if run not in results[metric]:
                        results[metric][run] = []
                    results[metric][run].append({"x": step, "y": log[metric]})

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Curves</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .chart-container { width: 45%; display: inline-block; margin: 10px; }
    </style>
</head>
<body>
    <h2>Training Curves for Different Gamma Values</h2>
"""

for i, metric in enumerate(metrics_to_plot):
    html_content += f'<div class="chart-container"><canvas id="chart{i}"></canvas></div>\n'

html_content += "<script>\n"

for i, metric in enumerate(metrics_to_plot):
    datasets_js = []
    for j, run in enumerate(runs):
        if run in results[metric]:
            data_points = results[metric][run]
            data_js = json.dumps(data_points)
            datasets_js.append(f"""
                {{
                    label: '{run_labels[j]}',
                    data: {data_js},
                    borderColor: '{colors[j % len(colors)]}',
                    fill: false,
                    tension: 0.1
                }}
            """)
    
    html_content += f"""
    new Chart(document.getElementById('chart{i}'), {{
        type: 'line',
        data: {{
            datasets: [{','.join(datasets_js)}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{
                    display: true,
                    text: '{metric}'
                }}
            }},
            scales: {{
                x: {{
                    type: 'linear',
                    title: {{ display: true, text: 'Step' }}
                }},
                y: {{
                    title: {{ display: true, text: 'Value' }}
                }}
            }}
        }}
    }});
    """

html_content += "</script>\n</body>\n</html>"

with open(os.path.join(base_dir, "combined_training_curves.html"), "w") as f:
    f.write(html_content)

print(os.path.join(base_dir, "combined_training_curves.html") + " generated!")
