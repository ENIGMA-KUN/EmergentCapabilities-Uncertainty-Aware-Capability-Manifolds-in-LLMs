import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D

# Configuration
np.random.seed(42)
GRID_RES = 50  # Grid resolution
PHASE_TRANSITION_SCALE = 3.0  # Controls "sharpness" of capability emergence
NOISE_LEVEL = 0.4  # Uncertainty in capability predictions
TAU = 0.7  # Performance threshold
EPSILON = 0.1  # Confidence level (1-ε)

# Generate parameter space grid
x = np.linspace(-5, 5, GRID_RES)
y = np.linspace(-5, 5, GRID_RES)
z = np.linspace(-5, 5, GRID_RES)
X, Y, Z = np.meshgrid(x, y, z)

def capability_score(params, n_samples=10):
    """
    Simulates capability emergence with phase transitions
    
    Args:
        params: Array of shape (..., 3) containing parameter coordinates
        n_samples: Number of Monte Carlo samples for uncertainty estimation
    
    Returns:
        Array of capability scores
    """
    norm_params = np.linalg.norm(params, axis=-1)
    
    # Generate multiple samples to estimate uncertainty
    samples = []
    for _ in range(n_samples):
        # Base capability (sigmoid transition)
        base_capability = 1 / (1 + np.exp(-PHASE_TRANSITION_SCALE*(norm_params - 3)))
        
        # Add uncertainty (heteroscedastic noise)
        noise = NOISE_LEVEL * np.random.randn(*norm_params.shape)
        samples.append(base_capability + noise)
    
    return np.stack(samples)

# Calculate probability of exceeding threshold with multiple samples
capabilities = capability_score(np.stack([X, Y, Z], axis=-1))
prob_above_tau = np.mean(capabilities >= TAU, axis=0)

# Create figure with improved style
plt.style.use('default')
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set background color to white for better visibility
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Plot ε-manifold surface with improved resolution
try:
    # Try the newer version first
    verts, faces, _, _ = measure.marching_cubes_lewiner(
        prob_above_tau,
        level=1-EPSILON,
        spacing=(0.2, 0.2, 0.2)
    )
except AttributeError:
    # Fall back to the newest version if available
    verts, faces, _, _ = measure.marching_cubes(
        prob_above_tau,
        level=1-EPSILON,
        spacing=(0.2, 0.2, 0.2)
    )

# Transform mesh to original coordinates
surf = ax.plot_trisurf(
    verts[:, 0]*0.2 - 5,
    verts[:, 1]*0.2 - 5,
    faces,
    verts[:, 2]*0.2 - 5,
    cmap='viridis',
    alpha=0.8,
    edgecolor='none'
)

# Add phase transition markers with improved detection
gradients = np.gradient(prob_above_tau)
gradient_magnitude = np.sqrt(sum(grad**2 for grad in gradients))
critical_points = np.where(gradient_magnitude > np.percentile(gradient_magnitude, 95))

scatter = ax.scatter(
    x[critical_points[0]],
    y[critical_points[1]],
    z[critical_points[2]],
    c='red',
    s=20,
    alpha=0.6,
    label='Phase Transitions'
)

# Improved formatting
ax.set_xlabel('Parameter Dimension 1', fontsize=12, labelpad=10)
ax.set_ylabel('Parameter Dimension 2', fontsize=12, labelpad=10)
ax.set_zlabel('Parameter Dimension 3', fontsize=12, labelpad=10)
ax.set_title(f'ε-Capability Manifold (τ={TAU}, ε={EPSILON})', fontsize=14, pad=20)
ax.legend(loc='upper right', fontsize=10)

# Add colorbar with improved styling
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, label='P(Capability ≥ τ)')
cbar.ax.set_ylabel('P(Capability ≥ τ)', fontsize=10, rotation=270, labelpad=15)

# Adjust view angle for better visualization
ax.view_init(elev=20, azim=45)

# Improve overall layout
plt.tight_layout()
plt.show()