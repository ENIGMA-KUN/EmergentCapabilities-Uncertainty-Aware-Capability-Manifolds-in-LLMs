# **EmergentCapabilities: Uncertainty-Aware Capability Manifolds in LLMs**

A research codebase accompanying the paper:

> **“Uncertainty-Aware ε-Capability Manifolds for Predicting Emergent Behaviors in Large Language Models.”**  
> *Shubham Chakraborty*, *Anupam Srivastava*  
> **Conference**: *Potentially under review for ICML 2025*

---

## **Abstract**



Scaling laws in Large Language Models (LLMs) often result in *emergent capabilities*—discrete jumps in model behavior at certain scales. This repository provides a **theoretical and empirical** framework for measuring, predicting, and explaining these emergent phenomena. By combining **ε-capability manifolds** (a geometry-centric view of capability boundaries) with **Uncertainty-Aware Capability Score (UCS)**, we highlight **when** a capability “turns on” and **how reliable** it is under uncertainty. Our code includes:

1. **Multiple-choice QA** pipeline for tasks like CosmosQA.  
2. **Uncertainty** estimation (entropy, conformal prediction, etc.).  
3. **Ablation** scripts for sweeping \(\alpha\) (uncertainty penalty) and \(\tau\) (capability threshold).  
4. **Comprehensive** evaluation on various LLMs, from DistilGPT2 to 7B‐plus scales.

This repository aims to make **research reproducible** and to serve as a reference for further studies on emergent behavior in LLMs.

---

## **Table of Contents**

1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Installation & Setup](#installation--setup)  
4. [Usage](#usage)  
   - [Data Preparation](#data-preparation)  
   - [Running Experiments](#running-experiments)  
   - [Analysis & Plots](#analysis--plots)  
5. [Results & Benchmarks](#results--benchmarks)  
6. [Ablation & Sensitivity Studies](#ablation--sensitivity-studies)  
7. [Citation](#citation)  
8. [License](#license)  

---

## **Features**

- **ε-Capability Manifolds**: A geometry-based notion of how capabilities form boundaries in parameter space.  
- **Uncertainty-Aware Capability Score (UCS)**: Balances performance and uncertainty, enabling more robust detection of emergent behaviors.  
- **Multiple Datasets**: Example tasks covering QA (CosmosQA, MMLU), Reading Comprehension, Summarization, etc.  
- **Scalable**: Supports models from **DistilGPT2** (~82M params) to **GPT2‐medium** (~345M), or any Hugging Face model that fits your GPU.  
- **Interactive Notebook**: Jupyter notebooks for analyzing UCS vs. alpha, fraction emergent vs. tau, and more.

---

## **Project Structure**

```
emergence_project/
├── data/
│   ├── cosmosqa_10k.json     # Example dataset
│   ├── ...                   # Other task data
│   └── README.md
├── src/
│   ├── main.py               # Main pipeline to run multi-choice QA
│   ├── model_utils.py        # Model loading utilities
│   ├── multiple_choice.py     # Probability extraction for MCQA
│   ├── capability_utils.py   # Functions for measuring capability & uncertainty
│   └── ...
├── results/
│   ├── cosmosqa_distilgpt2.json # Example output
│   ├── figures/                  # Saved plots
│   └── ...
├── analysis.ipynb            # Notebook for post-processing & plotting
├── requirements.txt
├── LICENSE
└── README.md  <- you are here
```

---

## **Installation & Setup**

**Requirements**:
- Python 3.9+  
- [Anaconda / Miniconda](https://docs.conda.io/en/latest/) recommended  
- NVIDIA GPU with CUDA 11.8 (for large LLMs)

**Steps**:
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/enigma-kun/EmergentCapabilities-Uncertainty-Aware-Capability-Manifolds-in-LLMs.git
   cd EmergentCapabilities-Uncertainty-Aware-Capability-Manifolds-in-LLMs
   ```
2. **Create & Activate Conda Environment**  
   ```bash
   conda create -n emergence_env python=3.9 -y
   conda activate emergence_env
   ```
3. **Install Dependencies**  
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Data Preparation**

- Place your **CosmosQA** data in `data/cosmosqa_10k.json`.  
- For additional tasks (RC, Summarization, etc.), format them as multiple-choice JSON or adapt the code in `src/main.py` to parse them.

### **Running Experiments**

From the project root, run:

```bash
python src/main.py \
    --model_name distilgpt2 \
    --data_path data/cosmosqa_10k.json \
    --output results/cosmosqa_distilgpt2.json \
    --alpha 0.3 \
    --device cuda
```

- **model_name**: A valid Hugging Face model identifier (e.g., `gpt2`, `meta-llama/Llama-2-7b-hf`).  
- **data_path**: Path to your QA dataset in multiple-choice format.  
- **output**: Where to save the JSON results.  
- **alpha**: Uncertainty penalty weight, if you want to incorporate it at runtime.  
- **device**: “cuda” or “cpu.”

### **Analysis & Plots**

Open the **`analysis.ipynb`** notebook:

```bash
jupyter notebook analysis.ipynb
```

Inside the notebook, update the `results_json_path` to point to your newly generated JSON. Run the cells to get:

- Mean accuracy, entropy, and UCS.  
- Plots of **UCS vs. alpha** and **fraction emergent vs. tau**.  
- Optionally, `plt.savefig` to store figures in `results/figures/`.

---

## **Results & Benchmarks**

Our **preliminary** experiments show:

| Model         | #Params | Mean Acc (C) | Mean Entropy (U) | Mean UCS (α=0.3) | Notes           |
|---------------|--------:|-------------:|------------------:|------------------:|-----------------|
| distilgpt2    | ~82M    | 0.42         | 0.54             | 0.30             | partial data    |
| gpt2          | ~124M   | 0.55         | 0.40             | 0.49             | robust?         |
| gpt2-medium   | ~345M   | 0.60         | 0.38             | 0.53             | ...             |

Plots and additional results can be found in the [analysis notebook](analysis.ipynb).

---

## **Ablation & Sensitivity Studies**

- **Alpha Sweep**: We systematically vary \(\alpha \in [0,1]\) to see how UCS trades off raw capability vs. uncertainty.  
- **Tau Sweep**: We vary the threshold \(\tau\) to decide when a capability is “emergent.”  

See `analysis.ipynb` for line plots and interpretative commentary.

---

## **Citation**

If you find this code or framework useful in your research, please cite:

```bibtex
@article{your2025emergent,
  title={Uncertainty-Aware \epsilon-Capability Manifolds for Predicting Emergent Behaviors in Large Language Models},
  author={Shubham Chakraborty, Anupam Srivastava},
  journal={Under Review at ICML},
  year={2025}
}
```

---

## **License**

This project is licensed under the **MIT License** (or CC BY 4.0 / whichever you choose) – see the [LICENSE](LICENSE) file for details.

---

### **Contact / Contributing**

- **Questions / Issues**: Raise a GitHub Issue or email [chakraborty.shubham007@gmail.com](mailto:chakraborty.shubham007@gmail.com).  
- **Contributions**: Pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for more details on coding guidelines.

---

**Thank you** for checking out our work! We hope this framework aids your research on emergent capabilities in LLMs, uncertainty quantification, and safe AI development.