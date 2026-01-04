# ClusterFewshot: Experimental Guide

This guide provides reproducible steps to run BetterTogether experiments with a focus on evaluating and showcasing the capabilities of **ClusterFewshot**, a newly proposed diversity and feedback-driven prompt optimizer built on top of the [DSPy](https://github.com/stanfordnlp/dspy) framework. ClusterFewshot is designed to improve demonstration selection through semantic clustering and scoring-driven selection mechanisms, and integrates seamlessly within hybrid optimization pipelines introduced in [BetterTogether (2024)](https://arxiv.org/abs/2407.10930). The guide also supports comparison against other bootstrap-based optimizers and serves as the official documentation for reproducing results in the accompanying paper.

> ✅ Validated on:
> * Ubuntu 20.04 / 22.04
> * NVIDIA-compatible GPUs (driver version: **nvidia-driver-560** recommended):
>   * 1 × L4 minimum (Prompt Optimization Only)
>   * 1 × A100 80GB minimum (Prompt Optimization + LoRA Fine-tuning)
---

## Setup Instructions

[//]: # (### 1. Clone the ClusterFewshot repository)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone https://github.com/omrirh/clusterfewshot.git)

[//]: # (cd ClusterFewshot)

[//]: # (```)

### 1. Set up environment variables

```bash
cp remote_setup/vm_vars.env.template vm_vars.env
vi vm_vars.env  # Add your HuggingFace token
```

### 2. (Optional) Install NVIDIA GPU drivers

```bash
bash remote_setup/install_nvidia_drivers.sh
nvidia-smi  # validate NVIDIA GPU driver is installed
```

### 3. Prepare virtual environment with all dependencies

```bash
bash remote_setup/prepare_virtualenv.sh
```

### 4. Launch an SGLang-compatible local model in a separate shell (example: Qwen2.5)

```bash
bash remote_setup/run_sglang_model.sh --model-name Qwen/Qwen2.5-7B-Instruct
# Wait for the model to fully load before proceeding to Step 5
```

### 5. Run the BetterTogether experiment

```bash
bash better_together_experiment_driver.sh \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset iris \
  --strategy "p -> w" \
  --prompt-optimizer clusterfs
```

## Supported Configuration Options

### Prompt Optimizers (`--prompt-optimizer`)

* `bfrs` - BootstrapFewshotRS (baseline, random search implementation on top of BootstrapFewshot optimizer)
* `miprov2` - MIPROv2 (baseline, jointly optimizes instructions and bootstrapped few-shot examples using Bayesian Optimization)
* * `clusterfs` - ClusterFewshot (Semantic-aware few-shot optimizer that combines bootstrapping with task-adaptive sampling strategies)

### Experiment Strategies (`--strategy`)

* `p` — Prompt only
* `w` — Weight tuning only
* `p -> w`
* `w -> p`
* `p -> w -> p`
* `p -> p`
* `p -> p -> p`

### Base Models (`--model`)

* `meta-llama/Llama-2-7b-chat-hf`
* `meta-llama/Meta-Llama-3-8B-Instruct`
* `mistralai/Mistral-7B-Instruct-v0.2`
* `Qwen/Qwen2.5-7B-Instruct`
* `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
* `Qwen/Qwen3-8B`
* `google/gemma-3-4b-it`
* `Qwen/Qwen2-7B-Instruct`
* `meta-llama/Llama-3.1-8B-Instruct`
* `meta-llama/Llama-3.2-3B-Instruct`

**Note: Ensure that your Hugging Face token has access to the selected model above.**


### Datasets (`--dataset`)
* `gsm8k`
* `hotpotqa`
* `iris`

---

## Output and Logs

Experiment logs and outputs are stored under the local repository.
Each log file includes:

* Prompt optimization metrics and trace outputs
* LoRA fine-tuning summaries (if applicable)
* Final accuracy and configuration snapshot

For ClusterFewshot prompt optimizer, visualizations of Training/Validation PCA clusters as well as One-shot scores and distribution are stored in the local repository path.

