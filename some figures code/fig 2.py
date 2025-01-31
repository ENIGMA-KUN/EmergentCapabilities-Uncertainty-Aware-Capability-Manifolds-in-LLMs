#!/usr/bin/env python3
# coding: utf-8

"""
emergence_plot.py

Generates a synthetic "meaningful" plot of emergent behavior
across five tasks, computing Uncertainty-Aware Capability Score (UCS)
for each model scale and visualizing them as lines.

This simulates emergent-like improvements and also includes an
uncertainty measure that slightly penalizes final scores.
"""

import numpy as np
import matplotlib.pyplot as plt

def logistic_emergence(x, x0, k, floor=0.05, ceil=0.95):
    """
    Logistic function to simulate emergent transitions.
    x: param scale
    x0: center of transition
    k: steepness
    floor, ceil: min/max possible range
    """
    # logistic base formula: 1 / (1 + e^{-k(x - x0)})
    # then scale to [floor, ceil]
    return floor + (ceil - floor) / (1.0 + np.exp(-k * (x - x0)))

def generate_synthetic_data_scales():
    """
    Returns an array of param scales (in millions),
    e.g. from 82M to 7000M (7B).
    We'll do ~7 points on a log-ish scale.
    """
    return np.array([82, 124, 345, 774, 1500, 3500, 7000], dtype=float)

def generate_accuracy_and_uncertainty(task_name, param_scales):
    """
    Generate synthetic accuracy and uncertainty for each param scale.
    We'll define emergent transitions using logistic, with different
    centers or slopes for each task for variety.
    """
    # Different tasks might have different emergent centers & slopes
    # for illustration. We'll define some constants:
    # these could be tuned per task
    if task_name == "QA":
        x0, k = 500.0, 0.003
        base_uncert = 0.4
    elif task_name == "RC":
        x0, k = 1000.0, 0.002
        base_uncert = 0.35
    elif task_name == "CI":
        x0, k = 800.0, 0.0025
        base_uncert = 0.38
    elif task_name == "DRS":
        x0, k = 1200.0, 0.0018
        base_uncert = 0.30
    elif task_name == "DS":
        x0, k = 2000.0, 0.0015
        base_uncert = 0.45
    else:
        x0, k = 1000.0, 0.002
        base_uncert = 0.40
    
    # We'll produce an array of accuracies, plus a bit of random noise
    accuracies = []
    uncertainties = []
    
    for p in param_scales:
        # logistic for accuracy:
        acc = logistic_emergence(p, x0, k, floor=0.05, ceil=0.90)
        # small random jitter:
        acc += np.random.normal(0.0, 0.01)
        acc = np.clip(acc, 0.0, 1.0)
        
        # some "base" uncertainty that might *decrease* with scale
        # e.g. base_uncert - logistic
        # or simpler approach: We define a mild downward slope:
        # or we can do: unc = 0.4 - 0.1 * logistic
        # let's keep it simpler:
        unc = base_uncert - 0.1 * logistic_emergence(p, x0, k, floor=0.0, ceil=1.0)
        unc += np.random.normal(0.0, 0.01)
        # clamp to [0.0, 1.0]
        unc = np.clip(unc, 0.0, 1.0)
        
        accuracies.append(acc)
        uncertainties.append(unc)
    
    return np.array(accuracies), np.array(uncertainties)

def main():
    # The tasks we'll simulate
    tasks = ["QA", "RC", "CI", "DRS", "DS"]
    alpha = 0.3   # penalty factor for uncertainty
    
    param_scales = generate_synthetic_data_scales()  # array of shape (7,)
    
    # We'll store results in a dict
    # e.g. results[task] = { "acc": ..., "unc": ..., "ucs": ...}
    results = {}
    
    for t in tasks:
        acc, unc = generate_accuracy_and_uncertainty(t, param_scales)
        # compute UCS
        ucs = acc * (1.0 - alpha * unc)
        results[t] = {
            "acc": acc,
            "unc": unc,
            "ucs": ucs
        }
    
    # Let's plot a multi-line chart of param vs. UCS for each task
    plt.figure(figsize=(7,5))
    for t in tasks:
        yvals = results[t]["ucs"]
        plt.plot(param_scales, yvals, marker='o', label=f"{t} (UCS)")
    
    plt.title("Synthetic Emergence of UCS vs. Model Scale")
    plt.xlabel("Parameter Count (millions)")
    plt.ylabel("Uncertainty-Aware Capability Score (UCS)")
    plt.xscale("log")  # maybe log scale for param
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("emergence_plot.png", dpi=300)
    plt.show()
    
    # Optionally, you can also plot the raw accuracy or uncertainty, etc.
    # For demonstration, let's do an optional second figure comparing Acc vs. param:
    fig2, ax2 = plt.subplots(figsize=(7,5))
    for t in tasks:
        ax2.plot(param_scales, results[t]["acc"], marker='^', label=f"{t} (Accuracy)")
    ax2.set_title("Synthetic Accuracy vs. Model Scale")
    ax2.set_xlabel("Parameter Count (millions)")
    ax2.set_ylabel("Accuracy")
    ax2.set_xscale("log")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig("accuracy_plot.png", dpi=300)
    plt.show()
    
    # And a third figure for Uncertainty, if desired
    fig3, ax3 = plt.subplots(figsize=(7,5))
    for t in tasks:
        ax3.plot(param_scales, results[t]["unc"], marker='v', label=f"{t} (Uncertainty)")
    ax3.set_title("Synthetic Uncertainty vs. Model Scale")
    ax3.set_xlabel("Parameter Count (millions)")
    ax3.set_ylabel("Uncertainty")
    ax3.set_xscale("log")
    ax3.set_ylim(0.0, 1.0)
    ax3.legend()
    ax3.grid(True)
    fig3.tight_layout()
    fig3.savefig("uncertainty_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
