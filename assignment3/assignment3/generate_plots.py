import re
import matplotlib.pyplot as plt

# File paths (Replace with your actual file paths)
files = {
    'model_1': '../new_results/first_model_rewards.txt',
    'model_2': '../new_results/second_model_rewards.txt',
    'model_3': '../new_results/third_model_rewards.txt'
}

data = {}

# Regex patterns to parse your log files
pattern = r"Update (\d+)/(\d+) \| True Score: ([\d.-]+) \| Optim Score: ([\d.-]+) \| Std: ([\d.-]+)"

# --- PARSING MODEL 1 ---
try:
    with open(files['model_1'], 'r') as f:
        content = f.read()
        matches = re.findall(pattern, content)
        data['model_1'] = {
            'epochs': [int(m[0]) for m in matches],
            'rewards': [float(m[2]) for m in matches]
        }
except FileNotFoundError:
    print("File 1 not found.")

# --- PARSING MODEL 2 ---
try:
    with open(files['model_2'], 'r') as f:
        content = f.read()
        matches = re.findall(pattern, content)
        data['model_2'] = {
            'epochs': [int(m[0]) for m in matches],
            'rewards': [float(m[2]) for m in matches],
            'optim': [float(m[3]) for m in matches]
        }
except FileNotFoundError:
    print("File 2 not found.")

# --- PARSING MODEL 3 ---
try:
    with open(files['model_3'], 'r') as f:
        content = f.read()
        matches = re.findall(pattern, content)
        data['model_3'] = {
            'epochs': [int(m[0]) for m in matches],
            'rewards': [float(m[2]) for m in matches],
            'optim': [float(m[3]) for m in matches]
        }
except FileNotFoundError:
    print("File 3 not found.")

# --- PLOTTING ---

# Plot 1: Model 1 (Baseline)
if 'model_1' in data:
    plt.figure(figsize=(10, 6))
    plt.plot(data['model_1']['epochs'], data['model_1']['rewards'], label='Mean Reward', color='blue', linewidth=2)
    plt.title('Model 1: Training Reward Curve (Baseline)')
    plt.xlabel('Epochs (Updates)')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_1_curve.png')
    plt.show()

# Plot 2: Model 2 (Over-Constrained)
if 'model_2' in data:
    plt.figure(figsize=(10, 6))
    plt.plot(data['model_2']['epochs'], data['model_2']['rewards'], label='True Reward (Environment)', color='red', linewidth=2)
    plt.plot(data['model_2']['epochs'], data['model_2']['optim'], label='Optimized Reward (Shaped)', color='orange', linestyle='--', alpha=0.7)
    plt.title('Model 2: Training Reward Curve (Refined Shaping)')
    plt.xlabel('Epochs (Updates)')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_2_curve.png')
    plt.show()

# Plot 3: Model 3 (Refined Shaping)
if 'model_3' in data:
    plt.figure(figsize=(10, 6))
    plt.plot(data['model_3']['epochs'], data['model_3']['rewards'], label='True Reward (Environment)', color='green', linewidth=2)
    plt.plot(data['model_3']['epochs'], data['model_3']['optim'], label='Optimized Reward (Shaped)', color='lightgreen', linestyle='--', alpha=0.7)
    plt.title('Model 3: Training Reward Curve (Over-Constrained)')
    plt.xlabel('Epochs (Updates)')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_3_curve.png')
    plt.show()