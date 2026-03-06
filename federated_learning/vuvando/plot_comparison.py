import matplotlib.pyplot as plt
import re
import os

def parse_log(filename):
    """Reads a log file and extracts Training and Evaluation Loss."""
    train_losses = []
    eval_losses = []
    
    if not os.path.exists(filename):
        print(f"WARNING: {filename} not found!")
        return [], []

    with open(filename, 'r') as f:
        log_content = f.read()

    # Regex to extract numbers from the log format
    train_matches = re.findall(r"'train_loss':\s*([0-9\.]+)", log_content)
    eval_matches = re.findall(r"'eval_loss':\s*([0-9\.]+)", log_content)

    return [float(x) for x in train_matches], [float(x) for x in eval_matches]

def plot_comparison():
    # 1. Parse both log files
    # Make sure these filenames match exactly what you saved
    base_train, base_eval = parse_log('log_baseline.txt')
    moreau_train, moreau_eval = parse_log('log_moreau.txt')
    
    # Create x-axis (Rounds)
    rounds = range(1, len(base_train) + 1)
    
    # 2. Setup the Plot
    plt.figure(figsize=(12, 6))
    
    # --- Plot Training Loss (The Constraint) ---
    plt.subplot(1, 2, 1)
    plt.plot(rounds, base_train, 'b--o', label='Baseline (FedAvg)', alpha=0.7)
    plt.plot(rounds, moreau_train, 'g-^', label='Moreau Envelopes (Ours)', linewidth=2)
    plt.title('Training Loss (The "Leash" Effect)')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss (MSE + Proximal)')
    plt.legend()
    plt.grid(True)
    
    # --- Plot Evaluation Loss (The Performance) ---
    plt.subplot(1, 2, 2)
    plt.plot(rounds, base_eval, 'b--o', label='Baseline (FedAvg)', alpha=0.7)
    plt.plot(rounds, moreau_eval, 'g-^', label='Moreau Envelopes (Ours)', linewidth=2)
    plt.title('Evaluation Loss (Anomaly Detection)')
    plt.xlabel('Communication Round')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True)
    
    # 3. Save
    plt.tight_layout()
    plt.savefig('thesis_comparison_graph.png', dpi=300)
    print("Success! Graph saved as 'thesis_comparison_graph.png'")

if __name__ == "__main__":
    plot_comparison()
