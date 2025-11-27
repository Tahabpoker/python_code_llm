# Pythia-70M Python Fine-Tuner

**Training an LLM on an RTX 3050 (4GB VRAM): From 20 days (CPU) to 7 hours.**

## About The Project
This repository documents my journey engineering an efficient end-to-end fine-tuning pipeline for small language models on highly constrained consumer hardware.

The goal was simple: **How far can we push a model on a 4GB GPU?**

Using **EleutherAI’s Pythia-70M**, I built a custom training workflow that leverages 4-bit quantization and LoRA adapters to fit a code-completion model entirely into the memory of a laptop GPU—without sacrificing training stability.

## The Stack
- **Model:** [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m)  
- **Dataset:** CodeSearchNet Python (Cleaned & formatted)  
- **Techniques:** QLoRA (4-bit quantization + LoRA), PEFT  
- **Libraries:** `PyTorch`, `Transformers`, `Accelerate`, `bitsandbytes`  
- **Environment:** WSL2 + CUDA on NVIDIA RTX 3050 (4GB)

## Key Highlights
- **Data Engineering:** Converted raw Python code into strict **Signature → Implementation** prompt-completion pairs for structured code learning.  
- **Memory Optimization:** Used `bitsandbytes` for efficient 4-bit QLoRA training to minimize VRAM footprint.  
- **Performance:** Reduced estimated training time from **~20 days (CPU)** to **~7 hours (GPU)** using batching, gradient accumulation, and quantization.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Tahabpoker/python_code_llm.git
cd python_code_llm
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

### 3. Prepare the Dataset
Run the processing scripts to clean and format the CodeSearchNet Python data:

```bash
# Step 1: Download and clean the dataset
python build_clean_code_ds.py

# Step 2: Format into prompt-completion pairs
python make_prompt_completion.py
```

### 4. Train the Model
This script automatically sets up LoRA adapters and 4-bit quantization:

```bash
python main.py
```

Training logs and checkpoints will be saved locally.

### 5. Run Inference
Test the model’s ability to generate Python code:

```bash
python model_test.py
```

---


