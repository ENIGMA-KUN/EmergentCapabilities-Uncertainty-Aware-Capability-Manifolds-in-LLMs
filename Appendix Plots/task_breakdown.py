import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set basic matplotlib parameters
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'

def create_task_breakdown_plot():
    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Model names and sizes for x-axis
    model_names = ['DistilGPT2', 'GPT2', 'GPT2-Med', 'GPT2-L', 'GPT2-XL', 
                  'GPT-J-6B', 'Llama-2-7B', 'Mistral-7B', 'Qwen-7B']
    model_sizes = [0.082, 0.124, 0.345, 0.774, 1.5, 6, 7, 7, 7]
    
    # Task data from our experiments
    qa_data = {
        'accuracy': [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.34, 0.35, 0.37],
        'uncertainty': [0.30, 0.32, 0.35, 0.40, 0.42, 0.45, 0.47, 0.46, 0.47]
    }
    
    rc_data = {
        'accuracy': [0.09, 0.10, 0.15, 0.20, 0.22, 0.28, 0.32, 0.33, 0.35],
        'uncertainty': [0.31, 0.29, 0.34, 0.40, 0.43, 0.46, 0.49, 0.48, 0.47]
    }
    
    ci_data = {
        'accuracy': [0.05, 0.06, 0.10, 0.14, 0.18, 0.25, 0.28, 0.30, 0.32],
        'uncertainty': [0.25, 0.28, 0.33, 0.38, 0.42, 0.45, 0.47, 0.45, 0.46]
    }
    
    dialogue_data = {
        'accuracy': [0.12, 0.18, 0.24, 0.30, 0.34, 0.38, 0.42, 0.44, 0.46],
        'uncertainty': [0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.47, 0.48]
    }
    
    summary_data = {
        'accuracy': [0.05, 0.08, 0.14, 0.20, 0.25, 0.30, 0.34, 0.36, 0.38],
        'uncertainty': [0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.44, 0.42, 0.45]
    }
    
    # Create subplots for each task
    tasks = [
        ('QA Performance', qa_data, gs[0, 0]),
        ('Reading Comprehension', rc_data, gs[0, 1]),
        ('Commonsense Inference', ci_data, gs[0, 2]),
        ('Dialogue Response', dialogue_data, gs[1, 0]),
        ('Summarization', summary_data, gs[1, 1])
    ]
    
    for title, data, position in tasks:
        ax = fig.add_subplot(position)
        
        # Plot accuracy and uncertainty
        ln1 = ax.plot(model_sizes, data['accuracy'], 'o-', 
                     color='#1f77b4', label='Accuracy', linewidth=2)
        ax.fill_between(model_sizes, 
                       np.array(data['accuracy']) - 0.05,
                       np.array(data['accuracy']) + 0.05,
                       color='#1f77b4', alpha=0.2)
        
        ax2 = ax.twinx()
        ln2 = ax2.plot(model_sizes, data['uncertainty'], 's--',
                      color='#d62728', label='Uncertainty', linewidth=2)
        
        # Combine legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', fontsize=9)
        
        # Style axes
        ax.set_xscale('log')
        ax.set_xlabel('Model Size (B params)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy', color='#1f77b4', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Uncertainty', color='#d62728', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        # Set tick parameters
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        # Set y-axis limits
        ax.set_ylim(0, 0.5)
        ax2.set_ylim(0, 0.6)
        
    # Add emergence visualization in the last subplot
    ax_emerge = fig.add_subplot(gs[1, 2])
    
    # Create emergence visualization
    task_thresholds = [0.25, 0.22, 0.20, 0.35, 0.28]  # Emergence thresholds
    task_names = ['QA', 'RC', 'CI', 'DR', 'Sum']
    emergence_points = []
    
    for thresh, task in zip(task_thresholds, task_names):
        # Find emergence point
        model_idx = next(i for i, size in enumerate(model_sizes) 
                        if size >= thresh)
        emergence_points.append((model_sizes[model_idx], task))
    
    # Plot emergence points
    for i, (point, task) in enumerate(emergence_points):
        ax_emerge.scatter(point, i, s=100, marker='o')
        ax_emerge.annotate(task, (point, i), 
                          xytext=(5, 0), textcoords='offset points')
    
    ax_emerge.set_xscale('log')
    ax_emerge.set_title('Capability Emergence Points', 
                       fontsize=12, fontweight='bold', pad=10)
    ax_emerge.set_xlabel('Model Size (B params)', fontsize=10, fontweight='bold')
    ax_emerge.set_yticks(range(len(task_names)))
    ax_emerge.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    return fig

if __name__ == "__main__":
    # Generate and save the plot
    fig = create_task_breakdown_plot()
    plt.savefig('task_breakdown.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    plt.close()