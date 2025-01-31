import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_attention_matrix(pattern_type, size=16):
    """Generate attention patterns based on emergence stage"""
    if pattern_type == 'scattered':
        # Create scattered random attention pattern
        matrix = np.random.rand(size, size) * 0.3
        # Add some random strong attention points
        for _ in range(5):
            i, j = np.random.randint(0, size, 2)
            matrix[i, j] = np.random.rand() * 0.7 + 0.3
            
    elif pattern_type == 'transitional':
        # Create semi-structured attention pattern
        base = np.random.rand(size, size) * 0.2
        # Add diagonal-like structure with noise
        for i in range(size):
            for j in range(max(0, i-2), min(size, i+3)):
                base[i, j] = np.random.rand() * 0.5 + 0.3
        matrix = base
        
    else:  # structured
        # Create strongly structured attention pattern
        matrix = np.zeros((size, size))
        # Add clear diagonal attention
        for i in range(size):
            matrix[i, i] = 0.8 + np.random.rand() * 0.2
            if i < size - 1:
                matrix[i, i+1] = 0.6 + np.random.rand() * 0.2
            if i > 0:
                matrix[i, i-1] = 0.6 + np.random.rand() * 0.2
        # Add light background attention
        matrix += np.random.rand(size, size) * 0.1
        
    return np.clip(matrix, 0, 1)

def create_attention_pattern_analysis():
    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate attention patterns
    pre_emergence = generate_attention_matrix('scattered')
    transitional = generate_attention_matrix('transitional')
    post_emergence = generate_attention_matrix('structured')
    
    # Color map for attention visualization
    cmap = plt.cm.YlOrRd
    
    # Plot 1: Pre-emergence scattered attention
    im1 = ax1.imshow(pre_emergence, cmap=cmap, aspect='auto')
    ax1.set_title('(A) Pre-emergence\nScattered Attention', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Add annotations for scattered points
    for i in range(16):
        for j in range(16):
            if pre_emergence[i, j] > 0.5:
                ax1.plot(j, i, 'k+', markersize=5, alpha=0.5)
    
    # Plot 2: Transitional focusing patterns
    im2 = ax2.imshow(transitional, cmap=cmap, aspect='auto')
    ax2.set_title('(B) Transitional\nFocusing Patterns', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Add arrows showing emerging structure
    for i in range(0, 16, 4):
        ax2.arrow(i, i, 2, 0, head_width=0.5, head_length=0.5, 
                 fc='black', ec='black', alpha=0.3)
    
    # Plot 3: Post-emergence structured attention
    im3 = ax3.imshow(post_emergence, cmap=cmap, aspect='auto')
    ax3.set_title('(C) Post-emergence\nStructured Attention', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Add annotations for structured patterns
    for i in range(16):
        if i % 4 == 0:
            ax3.plot([i-0.5, i+0.5], [i-0.5, i+0.5], 'k-', 
                    linewidth=1, alpha=0.5)
    
    # Common settings for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(np.arange(0, 16, 4))
        ax.set_yticks(np.arange(0, 16, 4))
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('Token Position', fontsize=10)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Additional annotations for emergence stages
    ax1.text(-4, -2, 'Early Stage', fontsize=10, fontweight='bold')
    ax2.text(-4, -2, 'Transition', fontsize=10, fontweight='bold')
    ax3.text(-4, -2, 'Emerged Stage', fontsize=10, fontweight='bold')
    
    # Add connection arrows between plots
    fig.patches.extend([
        plt.Arrow(0.32, 0.5, 0.02, 0, width=0.1, 
                 transform=fig.transFigure, color='gray', alpha=0.5),
        plt.Arrow(0.65, 0.5, 0.02, 0, width=0.1, 
                 transform=fig.transFigure, color='gray', alpha=0.5)
    ])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    return fig

def create_attention_evolution_animation():
    """Creates additional frames showing attention pattern evolution"""
    frames = []
    size = 16
    steps = 10
    
    for i in range(steps):
        # Create interpolated attention pattern
        weight = i / (steps - 1)
        scattered = generate_attention_matrix('scattered', size)
        structured = generate_attention_matrix('structured', size)
        interpolated = scattered * (1 - weight) + structured * weight
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(interpolated, cmap=plt.cm.YlOrRd)
        ax.set_title(f'Attention Evolution: {weight*100:.0f}%')
        plt.colorbar(im)
        frames.append(fig)
        plt.close()
    
    return frames

if __name__ == "__main__":
    # Generate main attention pattern analysis
    fig_attention = create_attention_pattern_analysis()
    plt.savefig('attention_patterns.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Generate evolution frames
    frames = create_attention_evolution_animation()
    for i, frame in enumerate(frames):
        frame.savefig(f'attention_evolution_{i:02d}.png',
                     dpi=300,
                     bbox_inches='tight',
                     facecolor='white',
                     edgecolor='none')
        plt.close(frame)