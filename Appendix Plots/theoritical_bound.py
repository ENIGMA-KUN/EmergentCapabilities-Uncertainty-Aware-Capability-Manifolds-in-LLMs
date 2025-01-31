import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set basic matplotlib parameters
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'

def create_theoretical_bounds_plot():
    # Set up the figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Information-theoretic lower bounds
    ax1 = fig.add_subplot(131)
    
    # Model sizes (in billions of parameters)
    model_sizes = np.array([0.082, 0.124, 0.345, 0.774, 1.5, 6, 7, 7, 7])
    
    # Theoretical minimum capacity required
    min_capacity = 0.2 * np.log2(model_sizes * 1e9) + 0.3
    actual_capacity = 0.35 * np.log2(model_sizes * 1e9) + 0.4
    
    # Plot with enhanced styling
    ax1.plot(model_sizes, min_capacity, color='#1f77b4', linestyle='-', 
             label='Theoretical Minimum', linewidth=2.5)
    ax1.plot(model_sizes, actual_capacity, color='#d62728', linestyle='--', 
             label='Observed Capacity', linewidth=2.5)
    ax1.fill_between(model_sizes, min_capacity, actual_capacity, 
                     color='#1f77b4', alpha=0.2)
    
    ax1.set_xlabel('Model Size (B params)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Capacity', fontsize=11, fontweight='bold')
    ax1.set_title('Information-Theoretic Bounds', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=9)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    
    # Plot 2: Phase transition boundaries
    ax2 = fig.add_subplot(132)
    
    # Generate phase transition data
    params = np.linspace(0, 7, 100)
    performance = 1 / (1 + np.exp(-2 * (params - 2)))
    
    # Add controlled noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=len(params))
    performance_noisy = performance + noise
    
    ax2.plot(params, performance, color='#2ca02c', linestyle='-', 
             label='Mean Performance', linewidth=2.5)
    ax2.fill_between(params, 
                     performance_noisy - 0.1, 
                     performance_noisy + 0.1, 
                     color='#2ca02c', alpha=0.3,
                     label='Uncertainty Range')
    ax2.axvline(x=2, color='#d62728', linestyle='--', 
                label='Phase Transition', linewidth=2)
    
    ax2.set_xlabel('Model Scale', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Performance', fontsize=11, fontweight='bold')
    ax2.set_title('Phase Transition Boundary', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    
    # Plot 3: Uncertainty scaling
    ax3 = fig.add_subplot(133)
    
    # Real model data
    model_sizes = [0.082, 0.124, 0.345, 0.774, 1.5, 6, 7, 7, 7]
    uncertainties = [0.30, 0.32, 0.35, 0.40, 0.42, 0.45, 0.47, 0.46, 0.47]
    accuracies = [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.34, 0.35, 0.37]
    
    # Create scatter plot with improved styling
    scatter = ax3.scatter(model_sizes, uncertainties, 
                         s=np.array(accuracies)*500,
                         c=accuracies,
                         cmap='viridis',
                         alpha=0.7)
    
    # Add and style colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Accuracy', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    ax3.set_xlabel('Model Size (B params)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Uncertainty', fontsize=11, fontweight='bold')
    ax3.set_title('Uncertainty vs. Scale', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    
    # Add tight layout with adequate spacing
    plt.tight_layout(pad=2.5)
    
    return fig

def create_phase_transition_surface():
    """Creates 3D visualization of phase transition surface"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create detailed meshgrid
    x = np.linspace(0, 7, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    
    # Generate surface data
    Z = 1 / (1 + np.exp(-3 * (X - 2*Y - 1)))
    
    # Plot surface with enhanced styling
    surf = ax.plot_surface(X, Y, Z, 
                          cmap='viridis',
                          alpha=0.8,
                          antialiased=True)
    
    # Style the axes
    ax.set_xlabel('Model Size (B params)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Task Difficulty', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Performance', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Phase Transition Surface', fontsize=13, fontweight='bold', pad=20)
    
    # Style colorbar
    cbar = plt.colorbar(surf)
    cbar.set_label('Performance Level', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=45)
    
    return fig

if __name__ == "__main__":
    # Generate and save main plot
    fig = create_theoretical_bounds_plot()
    plt.savefig('theoretical_bounds.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Generate and save 3D surface plot
    fig_3d = create_phase_transition_surface()
    plt.savefig('phase_transition_surface.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    plt.close()