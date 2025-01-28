#!/usr/bin/env python3
# coding: utf-8

"""
tables_for_all_datasets.py

Step 1: A script to store (dataset, model, param_count, acc, ent, ucs) records.
Step 2: Print a separate table for each dataset.
"""

import sys

# 1) Dummy/placeholder data records
# Fill in real results if you have them, or keep placeholders (N/A).
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


def print_table_for_dataset(dataset_name, records):
    # Filter
    subset = [r for r in records if r["dataset"] == dataset_name]
    if not subset:
        return

    # Print Table Header
    print(f"\n### Dataset: {dataset_name}\n")
    print("| Model       | #Params(M) | Accuracy (C) | Entropy (U) | UCS (α=0.3) |")
    print("|------------ |-----------:|-------------:|------------:|------------:|")
    for r in subset:
        m = r["model"]
        p = r["param_count"]
        a = r["acc"]
        e = r["ent"]
        u = r["ucs"]
        print(f"| {m:<12} | {p:>10} | {a:>12.3f} | {e:>11.3f} | {u:>11.3f} |")

def main():
    # We have 5 dataset names
    datasets = ["mmlu_10k","cosmosqa_10k","hellaswag_10k","halu_dialogue","halu_summarization"]
    print("# Tables for Each Dataset\n")
    for ds in datasets:
        print_table_for_dataset(ds, data_records)
    print("\nDone.")

if __name__ == "__main__":
    main()
