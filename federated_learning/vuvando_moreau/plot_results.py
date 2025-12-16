import matplotlib.pyplot as plt
import re

def parse_log(filename):
    train_losses = []
    eval_losses = []
    rounds = []

    try:
        with open(filename, 'r') as f:
            log_content = f.read()

        # Regex to find the Aggregated Metrics from the Server
        # Looking for patterns like: {'train_loss': 0.0059...}
        train_matches = re.findall(r"'train_loss':\s*([0-9\.]+)", log_content)
        eval_matches = re.findall(r"'eval_loss':\s*([0-9\.]+)", log_content)

        # Convert strings to floats
        train_losses = [float(x) for x in train_matches]
        eval_losses = [float(x) for x in eval_matches]
        
        # Create round numbers (1, 2, 3...)
        rounds = list(range(1, len(train_losses) + 1))
        
        return rounds, train_losses, eval_losses

    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Did you run the simulation with 'tee'?")
        return [], [], []

def plot_results(rounds, train_losses, eval_losses):
    plt.figure(figsize=(10, 6))
    
    # Plot Training Loss
    plt.plot(rounds, train_losses, 'b-o', label='Training Loss (Normal Beats)', linewidth=2)
    
    # Plot Evaluation Loss
    plt.plot(rounds, eval_losses, 'r--s', label='Evaluation Loss (Anomalies)', linewidth=2)

    plt.title('Federated Anomaly Detection: Moreau Envelope (pFedMe)', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Force integer ticks for rounds (Round 1, 2, 3...)
    plt.xticks(rounds)
    
    # Save the plot
    plt.savefig('moreau_results.png')
    print("Graph saved as 'Moreau_Envelope_results.png'")
    # plt.show() # Uncomment if you have a GUI

if __name__ == "__main__":
    rounds, t_loss, e_loss = parse_log('thesis_moreau_log.txt')
    
    if rounds:
        print(f"Found {len(rounds)} rounds.")
        print(f"Final Train Loss: {t_loss[-1]}")
        print(f"Final Eval Loss:  {e_loss[-1]}")
        plot_results(rounds, t_loss, e_loss)
    else:
        print("No data found to plot.")
