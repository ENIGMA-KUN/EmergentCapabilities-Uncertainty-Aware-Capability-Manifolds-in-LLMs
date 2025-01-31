import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# EXAMPLE DATA (Replace with real data)
# -----------------------------
np.random.seed(42)

# Suppose we have 10 models
num_models = 10
param_counts = np.logspace(6, 9, num_models)      # from ~1e6 to ~1e9
performance = np.random.uniform(0.2, 0.9, num_models)
accuracy = performance  # or separate metric
uncertainty = 1 - performance  # a trivial inverse for demonstration

# For 3D, re-use the same arrays, or you can define separate ones
param_3d = param_counts
acc_3d   = accuracy
unc_3d   = uncertainty

# -----------------------------
# PLOTTING
# -----------------------------
fig = plt.figure(figsize=(15, 5))

# --- LEFT SUBPLOT: param_count vs performance
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(param_counts, performance, marker='o', color='blue')
ax1.set_xscale('log')  # if param counts span large magnitudes
ax1.set_xlabel('Parameter Count')
ax1.set_ylabel('Performance')
ax1.set_title('Parameter Count vs. Performance')

# --- MIDDLE SUBPLOT: scatter of accuracy vs uncertainty
ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(accuracy, uncertainty, c='green')
ax2.set_xlabel('Accuracy')
ax2.set_ylabel('Uncertainty')
ax2.set_title('Accuracy vs. Uncertainty')

# --- RIGHT SUBPLOT: 3D (param, accuracy, uncertainty)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
p = ax3.scatter(param_3d, acc_3d, unc_3d, c=unc_3d, cmap='viridis')
ax3.set_xlabel('Param Count')
ax3.set_ylabel('Accuracy')
ax3.set_zlabel('Uncertainty')
ax3.set_title('3D Interaction: Params, Acc, Unc')

# Optional: add a colorbar for the 3D scatter
cb = fig.colorbar(p, ax=ax3, shrink=0.6)
cb.set_label('Uncertainty')

plt.tight_layout()
plt.show()
