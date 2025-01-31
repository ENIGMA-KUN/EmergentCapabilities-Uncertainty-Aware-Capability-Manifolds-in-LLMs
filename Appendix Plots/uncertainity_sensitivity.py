import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def create_threshold_sensitivity_plot():
    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Model configurations
    models = [
        ('DistilGPT2', 0.082, '#1f77b4'),
        ('GPT2', 0.124, '#ff7f0e'),
        ('GPT2-XL', 1.5, '#2ca02c'),
        ('Llama-2-7B', 7, '#d62728'),
        ('Qwen-7B', 7, '#9467bd')
    ]
    
    # Plot 1: Emergence fraction vs. threshold
    thresholds = np.linspace(0.1, 0.9, 50)
    
    for model_name, size, color in models:
        # Generate emergence fractions based on model size
        if size < 0.5:  # Small models
            fraction = 1 / (1 + np.exp(10 * (thresholds - 0.3)))
        elif size < 2:  # Medium models
            fraction = 1 / (1 + np.exp(8 * (thresholds - 0.5)))
        else:  # Large models
            fraction = 1 / (1 + np.exp(6 * (thresholds - 0.7)))
            
        ax1.plot(thresholds, fraction, label=model_name, 
                color=color, linewidth=2)
    
    ax1.set_title('Emergence Fraction vs. Threshold', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Threshold', fontsize=10)
    ax1.set_ylabel('Emergence Fraction', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model ranking stability
    thresholds_sparse = np.linspace(0.2, 0.8, 4)
    rankings = np.zeros((len(models), len(thresholds_sparse)))
    
    # Generate rankings for different thresholds
    for i, thresh in enumerate(thresholds_sparse):
        # Simulate performance scores at each threshold
        scores = []
        for _, size, _ in models:
            base_score = size / 8  # Normalize to 0-1 range
            noise = np.random.normal(0, 0.05)
            score = base_score + noise
            scores.append(score)
        
        # Convert to rankings
        rankings[:, i] = len(models) - stats.rankdata(scores)
    
    # Plot ranking lines
    for i, (model_name, _, color) in enumerate(models):
        ax2.plot(thresholds_sparse, rankings[i], 'o-',
                label=model_name, color=color, linewidth=2)
    
    ax2.set_title('Model Ranking Stability', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Threshold', fontsize=10)
    ax2.set_ylabel('Rank', fontsize=10)
    ax2.set_yticks(range(len(models)))
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Plot 3: Task-specific threshold effects
    tasks = ['QA', 'RC', 'CI', 'DR', 'Sum']
    threshold_effects = np.zeros((len(tasks), len(models)))
    
    # Generate task-specific effects
    for i, task in enumerate(tasks):
        for j, (_, size, _) in enumerate(models):
            base_effect = size / 8
            task_modifier = np.random.normal(0, 0.1)
            threshold_effects[i, j] = base_effect + task_modifier
    
    # Create heatmap
    im = ax3.imshow(threshold_effects, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Effect Strength', fontsize=10)
    
    # Add annotations
    for i in range(len(tasks)):
        for j in range(len(models)):
            text = ax3.text(j, i, f'{threshold_effects[i, j]:.2f}',
                          ha="center", va="center", color="black",
                          fontsize=8, fontweight='bold')
    
    ax3.set_title('Task-specific Threshold Effects', 
                 fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_yticks(range(len(tasks)))
    ax3.set_xticklabels([m[0] for m in models], rotation=45, ha='right')
    ax3.set_yticklabels(tasks)
    
    # Additional visualization enhancement
    def add_significance_arrows(ax, x, y, text):
        ax.annotate(text, xy=(x, y), xytext=(x, y+0.1),
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   ha='center', va='bottom')
    
    # Add significant threshold points in Plot 1
    add_significance_arrows(ax1, 0.3, 0.2, 'Small\nModel\nThreshold')
    add_significance_arrows(ax1, 0.5, 0.4, 'Medium\nModel\nThreshold')
    add_significance_arrows(ax1, 0.7, 0.6, 'Large\nModel\nThreshold')
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    return fig

# Additional function for detailed threshold analysis
def create_threshold_detail_plot():
    """Creates additional detailed threshold analysis"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate detailed threshold data
    thresholds = np.linspace(0.1, 0.9, 100)
    performance = 1 / (1 + np.exp(-10 * (thresholds - 0.5)))
    uncertainty = 1 - performance
    
    # Plot performance and uncertainty trade-off
    ax.plot(thresholds, performance, 'b-', label='Performance', linewidth=2)
    ax.plot(thresholds, uncertainty, 'r--', label='Uncertainty', linewidth=2)
    ax.fill_between(thresholds, performance, uncertainty, 
                    alpha=0.2, color='gray')
    
    ax.set_title('Threshold Performance-Uncertainty Trade-off',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == "__main__":
    # Generate main threshold sensitivity plot
    fig_threshold = create_threshold_sensitivity_plot()
    plt.savefig('threshold_sensitivity.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Generate detailed threshold analysis
    fig_detail = create_threshold_detail_plot()
    plt.savefig('threshold_detail.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()