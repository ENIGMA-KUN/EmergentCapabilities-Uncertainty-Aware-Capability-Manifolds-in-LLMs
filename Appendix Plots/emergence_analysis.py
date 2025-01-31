import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def create_emergence_analysis():
    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Model scales (in billions of parameters)
    model_sizes = np.array([0.082, 0.124, 0.345, 0.774, 1.5, 6, 7, 7, 7])
    
    # Plot 1: Reading Comprehension - Sharp Transition
    # ------------------------------------
    # Generate data showing sharp transition at 1B parameters
    rc_performance = []
    rc_uncertainty = []
    
    for size in model_sizes:
        if size < 1:
            perf = 0.15 + 0.05 * np.log(size + 0.1)
            unc = 0.4 - 0.05 * np.log(size + 0.1)
        else:
            perf = 0.3 + 0.15 * np.log(size + 0.1)
            unc = 0.2 - 0.02 * np.log(size + 0.1)
        rc_performance.append(perf)
        rc_uncertainty.append(unc)
    
    # Plot main performance curve
    ax1.plot(model_sizes, rc_performance, 'b-', linewidth=2.5, label='Performance')
    ax1.fill_between(model_sizes, 
                     np.array(rc_performance) - 0.05,
                     np.array(rc_performance) + 0.05,
                     alpha=0.2, color='blue')
    
    # Plot uncertainty
    ax1_twin = ax1.twinx()
    ax1_twin.plot(model_sizes, rc_uncertainty, 'r--', linewidth=2, label='Uncertainty')
    
    # Add transition marker
    ax1.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax1.text(1.1, 0.1, 'Sharp\nTransition', rotation=90, alpha=0.7)
    
    ax1.set_xscale('log')
    ax1.set_title('Reading Comprehension\nEmergence', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model Size (B params)', fontsize=10)
    ax1.set_ylabel('Performance', fontsize=10)
    ax1_twin.set_ylabel('Uncertainty', fontsize=10, color='red')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: QA - Gradual Emergence
    # ------------------------------------
    # Generate data showing gradual emergence
    qa_performance = 0.1 + 0.3 * np.log(model_sizes + 0.1) / np.log(8)
    qa_uncertainty = 0.5 - 0.2 * np.log(model_sizes + 0.1) / np.log(8)
    
    # Plot with confidence bands
    ax2.plot(model_sizes, qa_performance, 'g-', linewidth=2.5, label='Performance')
    ax2.fill_between(model_sizes, 
                     qa_performance - 0.05,
                     qa_performance + 0.05,
                     alpha=0.2, color='green')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(model_sizes, qa_uncertainty, 'r--', linewidth=2, label='Uncertainty')
    
    ax2.set_xscale('log')
    ax2.set_title('QA Capability\nGradual Emergence', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model Size (B params)', fontsize=10)
    ax2.set_ylabel('Performance', fontsize=10)
    ax2_twin.set_ylabel('Uncertainty', fontsize=10, color='red')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 3: Dialogue - Stepwise Improvements
    # ------------------------------------
    # Generate stepwise improvement data
    dialogue_performance = []
    dialogue_uncertainty = []
    
    for size in model_sizes:
        # Create stepwise pattern
        if size < 0.2:
            perf = 0.15
            unc = 0.45
        elif size < 1:
            perf = 0.25
            unc = 0.35
        elif size < 5:
            perf = 0.35
            unc = 0.25
        else:
            perf = 0.45
            unc = 0.20
            
        # Add small noise
        perf += np.random.normal(0, 0.01)
        unc += np.random.normal(0, 0.01)
        
        dialogue_performance.append(perf)
        dialogue_uncertainty.append(unc)
    
    # Plot stepwise performance
    ax3.plot(model_sizes, dialogue_performance, 'purple', linewidth=2.5, 
            label='Performance')
    ax3.fill_between(model_sizes, 
                     np.array(dialogue_performance) - 0.05,
                     np.array(dialogue_performance) + 0.05,
                     alpha=0.2, color='purple')
    
    # Plot uncertainty
    ax3_twin = ax3.twinx()
    ax3_twin.plot(model_sizes, dialogue_uncertainty, 'r--', linewidth=2, 
                 label='Uncertainty')
    
    # Add step markers
    for step in [0.2, 1, 5]:
        ax3.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
        ax3.text(step*1.1, 0.1, f'Step\n{step}B', rotation=90, alpha=0.7)
    
    ax3.set_xscale('log')
    ax3.set_title('Dialogue Capability\nStepwise Emergence', fontsize=12, 
                 fontweight='bold')
    ax3.set_xlabel('Model Size (B params)', fontsize=10)
    ax3.set_ylabel('Performance', fontsize=10)
    ax3_twin.set_ylabel('Uncertainty', fontsize=10, color='red')
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    return fig

def create_detailed_emergence_view():
    """Creates additional visualization showing emergence patterns in detail"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate detailed model size range
    model_sizes = np.logspace(-2, 1, 1000)
    
    # Generate different emergence patterns
    sharp = expit(3 * (np.log10(model_sizes) + 0.5))
    gradual = 0.1 + 0.4 * np.log10(model_sizes + 0.1) / np.log10(11)
    stepwise = np.minimum(0.5, np.maximum(0.1, 
                         np.floor(2 * np.log10(model_sizes) + 1) * 0.1))
    
    # Plot patterns
    ax.plot(model_sizes, sharp, 'b-', label='Sharp Transition', linewidth=2)
    ax.plot(model_sizes, gradual, 'g-', label='Gradual Emergence', linewidth=2)
    ax.plot(model_sizes, stepwise, 'purple', label='Stepwise', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_title('Comparison of Emergence Patterns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Size (B params)', fontsize=10)
    ax.set_ylabel('Capability Level', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == "__main__":
    # Generate main emergence analysis plot
    fig_emergence = create_emergence_analysis()
    plt.savefig('emergence_analysis.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Generate detailed emergence view
    fig_detail = create_detailed_emergence_view()
    plt.savefig('emergence_patterns_detail.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()