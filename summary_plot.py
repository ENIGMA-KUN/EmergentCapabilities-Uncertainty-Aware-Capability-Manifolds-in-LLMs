#!/usr/bin/env python3
# coding: utf-8

"""
plot_with_legend.py

Creates two plots:
1) Param vs. Accuracy
2) Param vs. UCS

Using color for each dataset, shape marker for each model, and a legend—no point text annotation.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

data_records = [
    # ===================== MMLU_10k (Question Answering) =====================
    {"dataset": "mmlu_10k", "model": "DistilGPT2",  "param_count": 82,    "acc": 0.10, "ent": 0.30, "ucs": 0.09},
    {"dataset": "mmlu_10k", "model": "GPT2",        "param_count": 124,   "acc": 0.15, "ent": 0.32, "ucs": 0.136},
    {"dataset": "mmlu_10k", "model": "gpt2-medium", "param_count": 345,   "acc": 0.20, "ent": 0.35, "ucs": 0.179},
    {"dataset": "mmlu_10k", "model": "gpt2-large",  "param_count": 774,   "acc": 0.25, "ent": 0.40, "ucs": 0.220},
    {"dataset": "mmlu_10k", "model": "gpt2-xl",     "param_count": 1500,  "acc": 0.28, "ent": 0.42, "ucs": 0.245},
    {"dataset": "mmlu_10k", "model": "EleutherAI/gpt-j-6B", "param_count": 6000,  "acc": 0.30, "ent": 0.45, "ucs": 0.260},
    {"dataset": "mmlu_10k", "model": "meta-llama/Llama-2-7b-hf", "param_count": 7000, "acc": 0.34,"ent":0.48,"ucs":0.291},
    {"dataset": "mmlu_10k", "model": "mistralai/Mistral-7B",    "param_count": 7000, "acc": 0.35,"ent":0.46,"ucs":0.302},
    {"dataset": "mmlu_10k", "model": "Qwen/Qwen-7B",            "param_count": 7000, "acc": 0.37,"ent":0.47,"ucs":0.319},

    # ===================== CosmosQA_10k (Reading Comprehension) ==============
    {"dataset": "cosmosqa_10k", "model": "DistilGPT2",  "param_count": 82,    "acc": 0.09, "ent": 0.31,"ucs": 0.082},
    {"dataset": "cosmosqa_10k", "model": "GPT2",        "param_count": 124,   "acc": 0.10, "ent": 0.29,"ucs": 0.091},
    {"dataset": "cosmosqa_10k", "model": "gpt2-medium", "param_count": 345,   "acc": 0.15, "ent": 0.34,"ucs": 0.135},
    {"dataset": "cosmosqa_10k", "model": "gpt2-large",  "param_count": 774,   "acc": 0.20, "ent": 0.40,"ucs": 0.176},
    {"dataset": "cosmosqa_10k", "model": "gpt2-xl",     "param_count": 1500,  "acc": 0.22, "ent": 0.43,"ucs": 0.192},
    {"dataset": "cosmosqa_10k", "model": "EleutherAI/gpt-j-6B", "param_count": 6000, "acc": 0.28,"ent":0.46,"ucs":0.241},
    {"dataset": "cosmosqa_10k", "model": "meta-llama/Llama-2-7b-hf", "param_count":7000, "acc":0.32,"ent":0.49,"ucs":0.273},
    {"dataset": "cosmosqa_10k", "model": "mistralai/Mistral-7B",    "param_count":7000, "acc":0.33,"ent":0.48,"ucs":0.282},
    {"dataset": "cosmosqa_10k", "model": "Qwen/Qwen-7B",            "param_count":7000, "acc":0.35,"ent":0.47,"ucs":0.301},

    # ===================== HellaSwag_10k (Commonsense Inference) =============
    {"dataset": "hellaswag_10k", "model": "DistilGPT2",  "param_count": 82,   "acc": 0.05,"ent":0.25, "ucs":0.046},
    {"dataset": "hellaswag_10k", "model": "GPT2",        "param_count": 124,  "acc": 0.06,"ent":0.28, "ucs":0.055},
    {"dataset": "hellaswag_10k", "model": "gpt2-medium", "param_count": 345,  "acc": 0.10,"ent":0.33, "ucs":0.090},
    {"dataset": "hellaswag_10k", "model": "gpt2-large",  "param_count": 774,  "acc": 0.14,"ent":0.38, "ucs":0.124},
    {"dataset": "hellaswag_10k", "model": "gpt2-xl",     "param_count":1500,  "acc": 0.18,"ent":0.42, "ucs":0.157},
    {"dataset": "hellaswag_10k", "model": "EleutherAI/gpt-j-6B", "param_count":6000,"acc":0.25,"ent":0.45,"ucs":0.216},
    {"dataset": "hellaswag_10k", "model": "meta-llama/Llama-2-7b-hf","param_count":7000,"acc":0.28,"ent":0.47,"ucs":0.240},
    {"dataset": "hellaswag_10k", "model": "mistralai/Mistral-7B",   "param_count":7000,"acc":0.30,"ent":0.45,"ucs":0.259},
    {"dataset": "hellaswag_10k", "model": "Qwen/Qwen-7B",           "param_count":7000,"acc":0.32,"ent":0.46,"ucs":0.276},

    # ===================== halu_dialogue (Dialogue Response Selection) ========
    {"dataset": "halu_dialogue", "model": "DistilGPT2",  "param_count":82,   "acc":0.12,"ent":0.25,"ucs":0.111},
    {"dataset": "halu_dialogue", "model": "GPT2",        "param_count":124,  "acc":0.18,"ent":0.30,"ucs":0.164},
    {"dataset": "halu_dialogue", "model": "gpt2-medium", "param_count":345,  "acc":0.24,"ent":0.35,"ucs":0.215},
    {"dataset": "halu_dialogue", "model": "gpt2-large",  "param_count":774,  "acc":0.30,"ent":0.40,"ucs":0.264},
    {"dataset": "halu_dialogue", "model": "gpt2-xl",     "param_count":1500, "acc":0.34,"ent":0.42,"ucs":0.297},
    {"dataset": "halu_dialogue", "model": "EleutherAI/gpt-j-6B","param_count":6000,"acc":0.38,"ent":0.45,"ucs":0.329},
    {"dataset": "halu_dialogue", "model": "meta-llama/Llama-2-7b-hf","param_count":7000,"acc":0.42,"ent":0.48,"ucs":0.360},
    {"dataset": "halu_dialogue", "model": "mistralai/Mistral-7B",   "param_count":7000,"acc":0.44,"ent":0.47,"ucs":0.378},
    {"dataset": "halu_dialogue", "model": "Qwen/Qwen-7B",           "param_count":7000,"acc":0.46,"ent":0.48,"ucs":0.394},

    # ===================== halu_summarization (Document Summarization) ========
    {"dataset": "halu_summarization","model":"DistilGPT2","param_count":82,   "acc":0.05,"ent":0.20,"ucs":0.047},
    {"dataset": "halu_summarization","model":"GPT2",      "param_count":124,  "acc":0.08,"ent":0.25,"ucs":0.074},
    {"dataset": "halu_summarization","model":"gpt2-medium","param_count":345, "acc":0.14,"ent":0.30,"ucs":0.127},
    {"dataset": "halu_summarization","model":"gpt2-large","param_count":774,  "acc":0.20,"ent":0.35,"ucs":0.179},
    {"dataset": "halu_summarization","model":"gpt2-xl",   "param_count":1500, "acc":0.25,"ent":0.38,"ucs":0.222},
    {"dataset": "halu_summarization","model":"EleutherAI/gpt-j-6B","param_count":6000,"acc":0.30,"ent":0.40,"ucs":0.264},
    {"dataset": "halu_summarization","model":"meta-llama/Llama-2-7b-hf","param_count":7000,"acc":0.34,"ent":0.44,"ucs":0.295},
    {"dataset": "halu_summarization","model":"mistralai/Mistral-7B",  "param_count":7000,"acc":0.36,"ent":0.42,"ucs":0.315},
    {"dataset": "halu_summarization","model":"Qwen/Qwen-7B",          "param_count":7000,"acc":0.38,"ent":0.45,"ucs":0.329},
]


# We'll define a color for each dataset
dataset_colors = {
    "mmlu_10k": "blue",
    "cosmosqa_10k": "green",
    "hellaswag_10k": "red",
    "halu_dialogue": "purple",
    "halu_summarization": "orange"
}

# We'll define a unique marker for each model 
model_markers = {
    "DistilGPT2":         "o",
    "GPT2":               "s",
    "GPT2-Medium":        "v",
    "GPT2-Large":         "^",
    "GPT2-XL":            "<",
    "EleutherAI/gpt-j-6B":"D",
    "meta-llama/Llama-2-7b-hf":"x",
    "mistralai/Mistral-7B":"P",
    "Qwen/Qwen-7B":       "h"
}

def plot_metric(metric_key, ylabel, filename):
    plt.figure(figsize=(6,4))
    # Group by (dataset, model)
    plotted_legend_keys = set()
    for rec in data_records:
        ds = rec["dataset"]
        mdl = rec["model"]
        xval = rec["param_count"]
        yval = rec[metric_key]
        color = dataset_colors[ds]
        marker = model_markers.get(mdl, "o")
        legend_key = (ds, mdl)  # We'll build a composite so each pair is unique

        # Plot single point
        plt.scatter(xval, yval, c=color, marker=marker, s=60)

    plt.xscale("log")
    plt.xlabel("Parameter Count (log scale)")
    plt.ylabel(ylabel)
    plt.title(f"Param vs. {ylabel}")

    # Create legend: one entry per dataset and model
    # We'll do two separate legends: dataset (color) + model (marker)
    # 1) dataset legend
    ds_handles = []
    for ds_name, c in dataset_colors.items():
        ds_handles.append(plt.Line2D([], [], color=c, marker='o', linestyle='None', label=ds_name))
    legend1 = plt.legend(handles=ds_handles, title="Datasets", loc="upper left")
    plt.gca().add_artist(legend1)

    # 2) model legend
    model_handles = []
    for m, mk in model_markers.items():
        model_handles.append(plt.Line2D([], [], color='black', marker=mk, linestyle='None', label=m))
    plt.legend(handles=model_handles, title="Models", loc="upper right")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}.")

def main():
    plot_metric("acc",  "Accuracy (C)", "param_vs_acc_legend.png")
    plot_metric("ucs",  "UCS (alpha=0.3)", "param_vs_ucs_legend.png")

if __name__ == "__main__":
    main()
