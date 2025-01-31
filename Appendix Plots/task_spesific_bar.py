import numpy as np
import matplotlib.pyplot as plt

def create_task_specific_comparison():
    # Set up the figure with 5 subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 6))
    
    # Model names and their configurations
    models = [
        'DistilGPT2',
        'GPT2',
        'GPT2-Medium',
        'GPT2-Large',
        'GPT2-XL',
        'GPT-J-6B',
        'Llama-2-7B',
        'Mistral-7B',
        'Qwen-7B'
    ]

    # Task data from our experiments
    tasks_data = {
        'MMLU': {
            'accuracy': [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.34, 0.35, 0.37],
            'ucs': [0.09, 0.14, 0.18, 0.22, 0.25, 0.26, 0.29, 0.30, 0.32],
            'title': '(a) MMLU 10k\nGeneral Knowledge'
        },
        'CosmosQA': {
            'accuracy': [0.09, 0.10, 0.15, 0.20, 0.22, 0.28, 0.32, 0.33, 0.35],
            'ucs': [0.08, 0.091, 0.14, 0.18, 0.19, 0.24, 0.27, 0.28, 0.30],
            'title': '(b) CosmosQA 10k\nReading Comprehension'
        },
        'HellaSwag': {
            'accuracy': [0.05, 0.06, 0.10, 0.14, 0.18, 0.25, 0.28, 0.30, 0.32],
            'ucs': [0.046, 0.06, 0.09, 0.12, 0.16, 0.22, 0.24, 0.26, 0.28],
            'title': '(c) HellaSwag 10k\nCommonsense Inference'
        },
        'Dialogue': {
            'accuracy': [0.12, 0.18, 0.24, 0.30, 0.34, 0.38, 0.42, 0.44, 0.46],
            'ucs': [0.111, 0.16, 0.21, 0.26, 0.30, 0.33, 0.36, 0.38, 0.39],
            'title': '(d) HaluEval\nDialogue'
        },
        'Summary': {
            'accuracy': [0.05, 0.08, 0.14, 0.20, 0.25, 0.30, 0.34, 0.36, 0.38],
            'ucs': [0.047, 0.07, 0.13, 0.18, 0.22, 0.26, 0.30, 0.32, 0.33],
            'title': '(e) HaluEval\nSummarization'
        }
    }

    # Create bar plots for each task
    for idx, (task_name, task_data) in enumerate(tasks_data.items()):
        ax = axs[idx]
        
        x = np.arange(len(models))
        width = 0.35
        
        # Create bars
        accuracy_bars = ax.bar(x - width/2, task_data['accuracy'], width, 
                             label='Accuracy', color='#2196F3', alpha=0.8)
        ucs_bars = ax.bar(x + width/2, task_data['ucs'], width, 
                         label='UCS', color='#F44336', alpha=0.8)
        
        # Customize plot
        ax.set_title(task_data['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Score' if idx == 0 else '')
        
        # Set y-axis limits
        ax.set_ylim(0, max(max(task_data['accuracy']), max(task_data['ucs'])) * 1.2)
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Only add legend to first subplot
        if idx == 0:
            ax.legend()

        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8, rotation=90)

        add_labels(accuracy_bars)
        add_labels(ucs_bars)
        
        # Add improvement trend line for accuracy
        trend_x = np.arange(len(models))
        z = np.polyfit(trend_x, task_data['accuracy'], 2)
        p = np.poly1d(z)
        ax.plot(trend_x, p(trend_x), '--', color='gray', alpha=0.5)

    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Add super title
    fig.suptitle('Task-specific Performance Comparison: Accuracy vs UCS', 
                fontsize=14, fontweight='bold', y=1.05)
    
    return fig

def create_trend_analysis():
    """Creates additional visualization showing improvement trends"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Model sizes in billions
    model_sizes = np.array([0.082, 0.124, 0.345, 0.774, 1.5, 6, 7, 7, 7])
    
    # Plot improvement trends for each task
    tasks_data = {
        'MMLU': {
            'accuracy': [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.34, 0.35, 0.37],
        },
        'CosmosQA': {
            'accuracy': [0.09, 0.10, 0.15, 0.20, 0.22, 0.28, 0.32, 0.33, 0.35],
        },
        'HellaSwag': {
            'accuracy': [0.05, 0.06, 0.10, 0.14, 0.18, 0.25, 0.28, 0.30, 0.32],
        },
        'Dialogue': {
            'accuracy': [0.12, 0.18, 0.24, 0.30, 0.34, 0.38, 0.42, 0.44, 0.46],
        },
        'Summary': {
            'accuracy': [0.05, 0.08, 0.14, 0.20, 0.25, 0.30, 0.34, 0.36, 0.38],
        }
    }
    
    tasks = ['MMLU', 'CosmosQA', 'HellaSwag', 'Dialogue', 'Summary']
    colors = ['#2196F3', '#4CAF50', '#FFC107', '#9C27B0', '#FF5722']
    
    for task, color in zip(tasks, colors):
        accuracies = np.array(tasks_data[task]['accuracy'])
        ax.plot(np.log10(model_sizes), accuracies, 'o-', 
                label=task, color=color, alpha=0.7)
    
    ax.set_xlabel('Log Model Size (B params)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Scaling Trends Across Tasks', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == "__main__":
    # Generate main task-specific comparison plot
    fig_comparison = create_task_specific_comparison()
    plt.savefig('task_specific_bars.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Generate trend analysis plot
    fig_trends = create_trend_analysis()
    plt.savefig('task_trends.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()