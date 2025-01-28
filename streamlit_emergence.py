# streamlit_emergence.py
import streamlit as st
import json
import numpy as np

# Suppose you have a single dataset's results in a JSON
# where each item has "capability" (0 or 1), "entropy", etc.
# We'll demonstrate how to see fraction emergent vs. tau.

def compute_ucs(capabilities, entropies, alpha):
    return capabilities * (1 - alpha * entropies)

def fraction_above_threshold(values, tau):
    return np.mean(values >= tau)

# For demonstration, we'll assume we've loaded a small data array:
# You can adapt to your real data
data = [
  {"capability": 1.0, "entropy": 0.314},
  {"capability": 0.0, "entropy": 0.200},
  {"capability": 1.0, "entropy": 0.500},
  {"capability": 1.0, "entropy": 1.200},
  {"capability": 0.0, "entropy": 0.050},
  # ...
]

capabilities = np.array([d["capability"] for d in data])
entropies    = np.array([d["entropy"]    for d in data])

st.title("Emergent Capability Dashboard")

alpha = st.slider("Uncertainty Penalty (alpha)", 0.0, 2.0, 0.3, 0.1)
tau   = st.slider("Threshold (tau)", 0.0, 2.0, 0.5, 0.1)

ucs_vals = compute_ucs(capabilities, entropies, alpha)
frac_emergent = fraction_above_threshold(ucs_vals, tau)

st.write(f"Fraction Emergent (UCS >= {tau:.2f}): {frac_emergent*100:.2f}%")

st.bar_chart(ucs_vals)
