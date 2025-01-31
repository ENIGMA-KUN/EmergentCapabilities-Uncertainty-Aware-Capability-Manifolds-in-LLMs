import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style properties directly instead of using style sheets
def set_plot_style():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14

def plot_accuracy_vs_ucs():
    # Data from the paper's Table 1
    models = ['DistilGPT2', 'GPT2', 'GPT2-Med', 'GPT2-Large', 'GPT2-XL', 'GPT-J-6B', 'Llama-2-7B', 'Qwen-7B']
    params = [0.082, 0.124, 0.345, 0.774, 1.5, 6, 7, 7]  # In billions
    accuracy = [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.34, 0.37]
    ucs = [0.09, 0.14, 0.18, 0.22, 0.25, 0.26, 0.29, 0.32]

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot both metrics
    plt.plot(params, accuracy, marker='o', color='#2ecc71', label='Accuracy (C(M))', linewidth=2)
    plt.plot(params, ucs, marker='s', color='#e74c3c', label='UCS(M)', linewidth=2)
    
    # Customize plot
    plt.xscale('log')
    plt.xlabel('Model Parameters (Billions)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Accuracy vs UCS Across Model Scales', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add model names as annotations
    for i, model in enumerate(models):
        plt.annotate(model, (params[i], accuracy[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_ucs.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_emergent_fraction():
    # Generate threshold values
    tau = np.linspace(0, 1, 100)
    
    # Simulate emergence patterns for different model sizes
    # Using sigmoid functions to approximate the phase transitions
    def emergence_curve(tau, scale, shift):
        return 1 / (1 + np.exp((tau - shift) * scale))
    
    # Different curves for different model sizes
    small_model = emergence_curve(tau, scale=15, shift=0.3)
    medium_model = emergence_curve(tau, scale=12, shift=0.5)
    large_model = emergence_curve(tau, scale=10, shift=0.7)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot curves with specific colors
    plt.plot(tau, small_model, color='#3498db', label='Small Model (~100M)', linewidth=2)
    plt.plot(tau, medium_model, color='#2ecc71', label='Medium Model (~1B)', linewidth=2)
    plt.plot(tau, large_model, color='#e74c3c', label='Large Model (~7B)', linewidth=2)
    
    # Add vertical lines at key thresholds
    plt.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    plt.xlabel('Threshold τ', fontsize=12)
    plt.ylabel('Fraction of Items Exceeding Threshold', fontsize=12)
    plt.title('Emergent Fraction vs Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add annotations for phase transitions
    plt.annotate('Small models\nsaturate here', xy=(0.3, 0.5), xytext=(0.15, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    plt.annotate('Large models\nmaintain coverage', xy=(0.6, 0.5), xytext=(0.75, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('emergent_fraction.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Set the style
    set_plot_style()
    
    try:
        # Generate both plots
        print("Generating Accuracy vs UCS plot...")
        plot_accuracy_vs_ucs()
        print("Accuracy vs UCS plot saved as 'accuracy_vs_ucs.png'")
        
        print("\nGenerating Emergent Fraction plot...")
        plot_emergent_fraction()
        print("Emergent Fraction plot saved as 'emergent_fraction.png'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")