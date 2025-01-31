import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def create_uncertainty_patterns():
    # Set up the figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Model configurations
    models = [
        ('DistilGPT2', 0.082, '#1f77b4'),
        ('GPT2-XL', 1.5, '#ff7f0e'),
        ('Llama-2-7B', 7, '#2ca02c'),
        ('Qwen-7B', 7, '#d62728')
    ]
    
    # Plot 1: Entropy Distributions
    x = np.linspace(0, 1, 1000)
    for model_name, size, color in models:
        # Generate entropy distribution based on model size
        if size < 1:  # Small models
            mu, sigma = 0.7, 0.15
        elif size < 5:  # Medium models
            mu, sigma = 0.5, 0.12
        else:  # Large models
            mu, sigma = 0.3, 0.1
            
        distribution = stats.beta.pdf(x, 
                                    mu * 10, 
                                    (1 - mu) * 10)
        ax1.plot(x, distribution, label=model_name, color=color, linewidth=2)
        
    ax1.set_title('Entropy Distributions Across Model Scales', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Entropy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Conformal Set Size Distributions
    set_sizes = {
        'DistilGPT2': [1, 2, 2, 3, 3, 3, 2, 2, 3, 4] * 100,
        'GPT2-XL': [1, 1, 2, 2, 2, 1, 1, 2, 2, 3] * 100,
        'Llama-2-7B': [1, 1, 1, 2, 1, 1, 1, 2, 1, 2] * 100,
        'Qwen-7B': [1, 1, 1, 1, 2, 1, 1, 1, 1, 2] * 100
    }
    
    positions = np.arange(len(models))
    violin_parts = ax2.violinplot(
        [set_sizes[model[0]] for model in models],
        positions,
        widths=0.8,
        showmeans=True,
        showextrema=True
    )
    
    # Color the violin plots
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(models[i][2])
        pc.set_alpha(0.6)
    
    ax2.set_title('Conformal Set Size Distributions', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([m[0] for m in models])
    ax2.set_ylabel('Set Size', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Task-specific Uncertainty Patterns
    tasks = ['QA', 'RC', 'CI', 'DR', 'Sum']
    uncertainty_data = {
        'DistilGPT2': [0.30, 0.31, 0.25, 0.25, 0.20],
        'GPT2-XL': [0.42, 0.43, 0.42, 0.42, 0.38],
        'Llama-2-7B': [0.48, 0.49, 0.47, 0.48, 0.44],
        'Qwen-7B': [0.47, 0.47, 0.46, 0.48, 0.45]
    }
    
    # Create heatmap data
    heatmap_data = np.array([uncertainty_data[model[0]] for model in models])
    
    # Plot heatmap with annotations
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Uncertainty Score', fontsize=12, fontweight='bold')
    
    # Add annotations
    for i in range(len(models)):
        for j in range(len(tasks)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black",
                          fontweight='bold')
    
    # Customize heatmap
    ax3.set_title('Task-specific Uncertainty Patterns', 
                 fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(np.arange(len(tasks)))
    ax3.set_yticks(np.arange(len(models)))
    ax3.set_xticklabels(tasks)
    ax3.set_yticklabels([m[0] for m in models])
    
    # Rotate x-axis labels
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    return fig

def create_conformal_prediction_example():
    """Additional visualization showing example conformal prediction sets"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Example data
    x = np.linspace(0, 10, 100)
    predictions = 0.5 * np.sin(x) + 0.5
    conformal_upper = predictions + 0.2 * np.random.rand(len(x))
    conformal_lower = predictions - 0.2 * np.random.rand(len(x))
    
    # Plot
    ax.plot(x, predictions, 'b-', label='Predicted Value')
    ax.fill_between(x, conformal_lower, conformal_upper, 
                   color='blue', alpha=0.2, label='Conformal Set')
    ax.set_title('Example Conformal Prediction Sets', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == "__main__":
    # Generate main uncertainty patterns plot
    fig_uncertainty = create_uncertainty_patterns()
    plt.savefig('uncertainty_patterns.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Generate conformal prediction example
    fig_conformal = create_conformal_prediction_example()
    plt.savefig('conformal_example.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    plt.close()