import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request

runs = [
    "CodeLlama-7b-hf-gamma-0p1",
    "CodeLlama-7b-hf-gamma-0p5",
    "CodeLlama-7b-hf-gamma-1p0",
    "CodeLlama-7b-hf-gamma-2p0"
]

base_dir = "/data/zzl/codeless-codefast/runs"
font_path = os.path.join(base_dir, "SimHei.ttf")

# Download SimHei font if it doesn't exist
if not os.path.exists(font_path):
    print("Downloading SimHei font...")
    urllib.request.urlretrieve("https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf", font_path)

# Set the downloaded font
my_font = fm.FontProperties(fname=font_path)

plt.rcParams['axes.unicode_minus'] = False # For displaying minus signs correctly

metrics_to_plot = ["completion_length", "rewards/correct_code_reward_func", "rewards/len_reward_func"]

filename_map = {
    "completion_length": "completion_length.png",
    "rewards/correct_code_reward_func": "correct_code_reward.png",
    "rewards/len_reward_func": "length_reward.png"
}

# Assign a specific color to each gamma value to maintain consistency
color_map = {
    "CodeLlama-7b-hf-gamma-0p1": "tab:blue",
    "CodeLlama-7b-hf-gamma-0p5": "tab:orange",
    "CodeLlama-7b-hf-gamma-1p0": "tab:green",
    "CodeLlama-7b-hf-gamma-2p0": "tab:red"
}

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    for run in runs:
        state_file = os.path.join(base_dir, run, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            steps = []
            values = []
            for log in data.get('log_history', []):
                if metric in log and 'step' in log:
                    steps.append(log['step'])
                    values.append(log[metric])
            
            if steps:
                gamma_val = run.split("-gamma-")[-1].replace("p", ".")
                plt.plot(steps, values, label=f"γ = {gamma_val}", color=color_map.get(run))
    
    plt.xlabel("步数", fontproperties=my_font, fontsize=14)
    if metric == "rewards/correct_code_reward_func":
        ylabel = "正确性奖励"
    elif metric == "rewards/len_reward_func":
        ylabel = "长度奖励"
    elif "length" in metric:
        ylabel = "生成长度"
    else:
        ylabel = "数值"
    plt.ylabel(ylabel, fontproperties=my_font, fontsize=14)
    plt.legend(prop=my_font, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(base_dir, filename_map[metric])
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

    # For length reward, generate a separate chart specifically for gamma=0.1
    if metric == "rewards/len_reward_func":
        plt.figure(figsize=(10, 6))
        
        run_0p1 = "CodeLlama-7b-hf-gamma-0p1"
        state_file = os.path.join(base_dir, run_0p1, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            steps = []
            values = []
            for log in data.get('log_history', []):
                if metric in log and 'step' in log:
                    steps.append(log['step'])
                    values.append(log[metric])
            
            if steps:
                plt.plot(steps, values, label="γ = 0.1", color=color_map[run_0p1])
                
        plt.xlabel("步数", fontproperties=my_font, fontsize=14)
        plt.ylabel("长度奖励", fontproperties=my_font, fontsize=14)
        plt.legend(prop=my_font, fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        output_path_0p1 = os.path.join(base_dir, "length_reward_gamma_0.1.png")
        plt.savefig(output_path_0p1)
        plt.close()
        print(f"Plot saved to {output_path_0p1}")
