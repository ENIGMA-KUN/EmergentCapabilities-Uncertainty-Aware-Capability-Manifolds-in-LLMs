#!/usr/bin/env python3
# coding: utf-8

"""
plots_for_all_datasets.py

Generates:
1) Param count vs. Accuracy plots (one per dataset)
2) Param count vs. UCS   plots (one per dataset)

Saves them to results/figures/datasetname_acc.png and datasetname_ucs.png.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Same data structure as the tables script
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

def filter_by_dataset(records, ds_name):
    return [r for r in records if r["dataset"] == ds_name]

def main():
    os.makedirs("results/figures", exist_ok=True)

    datasets = ["mmlu_10k","cosmosqa_10k","hellaswag_10k","halu_dialogue","halu_summarization"]
    for ds in datasets:
        subset = filter_by_dataset(data_records, ds)
        if not subset:
            continue
        # Sort by param_count
        subset_sorted = sorted(subset, key=lambda x: x["param_count"])

        params = [r["param_count"] for r in subset_sorted]
        acc    = [r["acc"]         for r in subset_sorted]
        ucs    = [r["ucs"]         for r in subset_sorted]
        models = [r["model"]       for r in subset_sorted]

        # 1) Plot: Param vs. Accuracy
        plt.figure()
        plt.plot(params, acc, marker='o')
        for i, txt in enumerate(models):
            plt.annotate(txt, (params[i], acc[i]), xytext=(5, -5),
                         textcoords='offset points', fontsize=8)
        plt.xscale("log")
        plt.xlabel("Parameter Count (log scale)")
        plt.ylabel("Accuracy (C)")
        plt.title(f"{ds}: Accuracy vs. Model Size")
        plt.grid(True)
        out_acc = f"results/figures/{ds}_acc.png"
        plt.savefig(out_acc, dpi=300)
        plt.close()

        # 2) Plot: Param vs. UCS
        plt.figure()
        plt.plot(params, ucs, marker='s', color='red')
        for i, txt in enumerate(models):
            plt.annotate(txt, (params[i], ucs[i]), xytext=(5, -5),
                         textcoords='offset points', fontsize=8)
        plt.xscale("log")
        plt.xlabel("Parameter Count (log scale)")
        plt.ylabel("UCS (alpha=0.3)")
        plt.title(f"{ds}: UCS vs. Model Size")
        plt.grid(True)
        out_ucs = f"results/figures/{ds}_ucs.png"
        plt.savefig(out_ucs, dpi=300)
        plt.close()

        print(f"Saved plots for dataset={ds} -> {out_acc}, {out_ucs}")

if __name__ == "__main__":
    main()
