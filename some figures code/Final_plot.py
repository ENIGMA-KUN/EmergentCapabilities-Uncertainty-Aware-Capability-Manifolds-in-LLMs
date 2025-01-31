#!/usr/bin/env python3
# coding: utf-8

"""
bar_chart_all_datasets.py

Generates a single figure with 5 subplots (2 rows x 3 cols, with one empty or legend slot).
Each subplot is a grouped bar chart for one dataset (mmlu_10k, cosmosqa_10k, etc.).
We have 9 models in each subplot, each model showing 2 bars: [Accuracy, UCS].

The final image is saved as bar_chart_all_datasets.png.

Usage:
  python bar_chart_all_datasets.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1) FULL DATA (45 entries): 9 models × 5 datasets, with placeholder acc (C), ucs values
# ---------------------------------------------------------------------------
data_records = [
    # mmlu_10k
    {"dataset":"mmlu_10k","model":"DistilGPT2",                "acc":0.10,"ucs":0.08},
    {"dataset":"mmlu_10k","model":"GPT2",                      "acc":0.15,"ucs":0.14},
    {"dataset":"mmlu_10k","model":"GPT2-Medium",               "acc":0.20,"ucs":0.18},
    {"dataset":"mmlu_10k","model":"GPT2-Large",                "acc":0.25,"ucs":0.22},
    {"dataset":"mmlu_10k","model":"GPT2-XL",                   "acc":0.28,"ucs":0.25},
    {"dataset":"mmlu_10k","model":"EleutherAI/gpt-j-6B",       "acc":0.30,"ucs":0.26},
    {"dataset":"mmlu_10k","model":"meta-llama/Llama-2-7b-hf",  "acc":0.34,"ucs":0.29},
    {"dataset":"mmlu_10k","model":"mistralai/Mistral-7B",      "acc":0.35,"ucs":0.30},
    {"dataset":"mmlu_10k","model":"Qwen/Qwen-7B",              "acc":0.37,"ucs":0.32},

    # cosmosqa_10k
    {"dataset":"cosmosqa_10k","model":"DistilGPT2",            "acc":0.09,"ucs":0.08},
    {"dataset":"cosmosqa_10k","model":"GPT2",                  "acc":0.10,"ucs":0.09},
    {"dataset":"cosmosqa_10k","model":"GPT2-Medium",           "acc":0.15,"ucs":0.14},
    {"dataset":"cosmosqa_10k","model":"GPT2-Large",            "acc":0.20,"ucs":0.18},
    {"dataset":"cosmosqa_10k","model":"GPT2-XL",               "acc":0.22,"ucs":0.19},
    {"dataset":"cosmosqa_10k","model":"EleutherAI/gpt-j-6B",   "acc":0.28,"ucs":0.24},
    {"dataset":"cosmosqa_10k","model":"meta-llama/Llama-2-7b-hf","acc":0.32,"ucs":0.27},
    {"dataset":"cosmosqa_10k","model":"mistralai/Mistral-7B",  "acc":0.33,"ucs":0.28},
    {"dataset":"cosmosqa_10k","model":"Qwen/Qwen-7B",          "acc":0.35,"ucs":0.30},

    # hellaswag_10k
    {"dataset":"hellaswag_10k","model":"DistilGPT2",           "acc":0.05,"ucs":0.05},
    {"dataset":"hellaswag_10k","model":"GPT2",                 "acc":0.06,"ucs":0.06},
    {"dataset":"hellaswag_10k","model":"GPT2-Medium",          "acc":0.10,"ucs":0.09},
    {"dataset":"hellaswag_10k","model":"GPT2-Large",           "acc":0.14,"ucs":0.12},
    {"dataset":"hellaswag_10k","model":"GPT2-XL",              "acc":0.18,"ucs":0.16},
    {"dataset":"hellaswag_10k","model":"EleutherAI/gpt-j-6B",  "acc":0.25,"ucs":0.22},
    {"dataset":"hellaswag_10k","model":"meta-llama/Llama-2-7b-hf","acc":0.28,"ucs":0.24},
    {"dataset":"hellaswag_10k","model":"mistralai/Mistral-7B", "acc":0.30,"ucs":0.26},
    {"dataset":"hellaswag_10k","model":"Qwen/Qwen-7B",         "acc":0.32,"ucs":0.28},

    # halu_dialogue
    {"dataset":"halu_dialogue","model":"DistilGPT2",           "acc":0.12,"ucs":0.11},
    {"dataset":"halu_dialogue","model":"GPT2",                 "acc":0.18,"ucs":0.16},
    {"dataset":"halu_dialogue","model":"GPT2-Medium",          "acc":0.24,"ucs":0.21},
    {"dataset":"halu_dialogue","model":"GPT2-Large",           "acc":0.30,"ucs":0.26},
    {"dataset":"halu_dialogue","model":"GPT2-XL",              "acc":0.34,"ucs":0.30},
    {"dataset":"halu_dialogue","model":"EleutherAI/gpt-j-6B",  "acc":0.38,"ucs":0.33},
    {"dataset":"halu_dialogue","model":"meta-llama/Llama-2-7b-hf","acc":0.42,"ucs":0.36},
    {"dataset":"halu_dialogue","model":"mistralai/Mistral-7B", "acc":0.44,"ucs":0.38},
    {"dataset":"halu_dialogue","model":"Qwen/Qwen-7B",         "acc":0.46,"ucs":0.39},

    # halu_summarization
    {"dataset":"halu_summarization","model":"DistilGPT2",      "acc":0.05,"ucs":0.05},
    {"dataset":"halu_summarization","model":"GPT2",            "acc":0.08,"ucs":0.07},
    {"dataset":"halu_summarization","model":"GPT2-Medium",     "acc":0.14,"ucs":0.13},
    {"dataset":"halu_summarization","model":"GPT2-Large",      "acc":0.20,"ucs":0.18},
    {"dataset":"halu_summarization","model":"GPT2-XL",         "acc":0.25,"ucs":0.22},
    {"dataset":"halu_summarization","model":"EleutherAI/gpt-j-6B","acc":0.30,"ucs":0.26},
    {"dataset":"halu_summarization","model":"meta-llama/Llama-2-7b-hf","acc":0.34,"ucs":0.30},
    {"dataset":"halu_summarization","model":"mistralai/Mistral-7B","acc":0.36,"ucs":0.32},
    {"dataset":"halu_summarization","model":"Qwen/Qwen-7B",    "acc":0.38,"ucs":0.33},
]

# ---------------------------------------------------------------------------
# 2) We'll create a figure with 2 rows x 3 cols of subplots
#    (5 subplots for 5 datasets, plus 1 empty or legend space).
# ---------------------------------------------------------------------------
import math

datasets = ["mmlu_10k","cosmosqa_10k","hellaswag_10k","halu_dialogue","halu_summarization"]
models = [
    "DistilGPT2","GPT2","GPT2-Medium","GPT2-Large","GPT2-XL",
    "EleutherAI/gpt-j-6B","meta-llama/Llama-2-7b-hf","mistralai/Mistral-7B","Qwen/Qwen-7B"
]

nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
axes = axes.flatten()  # to iterate easily

def filter_data(ds):
    return [r for r in data_records if r["dataset"] == ds]

# We'll loop over the first 5 subplots
for i, ds_name in enumerate(datasets):
    ax = axes[i]
    subset = filter_data(ds_name)
    # Sort subset by model order we want
    # i.e., create a dict for model->(acc,ucs)
    model2acc = {}
    model2ucs = {}
    for r in subset:
        model2acc[r["model"]] = r["acc"]
        model2ucs[r["model"]] = r["ucs"]

    # Build the bar chart data
    xvals = np.arange(len(models))
    accs = [model2acc[m] for m in models]
    ucss = [model2ucs[m] for m in models]

    width = 0.35
    ax.bar(xvals - width/2, accs, width, color="skyblue", label="Accuracy")
    ax.bar(xvals + width/2, ucss, width, color="salmon", label="UCS")

    ax.set_xticks(xvals)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)  # adjust if needed
    ax.set_title(ds_name)
    # optional: ax.grid(True)

# We'll make the 6th subplot blank or use it for a legend
for j in range(5, 6):
    ax_empty = axes[j]
    ax_empty.axis("off")
    # Create a custom legend
    # We'll place it in this empty subplot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color="skyblue", label="Accuracy", marker="s", linestyle="None"),
        Line2D([0],[0], color="salmon",  label="UCS", marker="s", linestyle="None"),
    ]
    ax_empty.legend(handles=legend_elements, loc="center", fontsize=10, title="Bars")

plt.tight_layout()
plt.savefig("bar_chart_all_datasets.png", dpi=300)
plt.show()

print("Saved bar_chart_all_datasets.png with 5 subplots for 5 datasets.")
