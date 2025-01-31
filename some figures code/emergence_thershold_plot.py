#!/usr/bin/env python3
# coding: utf-8

"""
threshold_emergence.py

Plots fraction of test items with UCS >= tau for different models, 
showing an 'emergence' perspective.
"""

import numpy as np
import matplotlib.pyplot as plt

# Suppose we have per-item UCS for each model. 
# For brevity, let's pretend we only have aggregated "Mean UCS" or an estimate of fraction>0.5
# But the real approach requires you to store each item’s UCS. We'll do a conceptual approach:

models = ["DistilGPT2","GPT2","GPT2-Medium","GPT2-Large","Qwen/Qwen-7B"]
# Example fraction of items that pass UCS >= 0.5 might be:
# We'll vary threshold from 0.0 to 1.0 in small increments.
thresholds = np.linspace(0.0, 1.0, 11)
# Toy "UCS distributions"
# For each model, an imaginary distribution function => fraction above each threshold
def fraction_above(model, tau):
    # A dummy mapping, bigger model => more fraction
    base = {
        "DistilGPT2": 0.2,
        "GPT2":       0.3,
        "GPT2-Medium":0.45,
        "GPT2-Large": 0.55,
        "Qwen/Qwen-7B":0.65
    }[model]
    # We'll do a linear drop with threshold for demonstration
    return max(0.0, base - 0.8*tau)

plt.figure(figsize=(6,4))
for mdl in models:
    fracs = [ fraction_above(mdl, t) for t in thresholds ]
    plt.plot(thresholds, fracs, label=mdl)

plt.xlabel("Threshold (tau)")
plt.ylabel("Fraction Emergent (UCS >= tau)")
plt.title("Emergence Fraction vs. Tau")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("emergence_fraction_vs_tau.png", dpi=300)
plt.close()
print("Saved emergence_fraction_vs_tau.png.")
