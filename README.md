# "qwen2.5_coder_7b_lora" Python Fine-Tuner

**Training an LLM on an RTX 3050 (4GB VRAM): From 20 days (CPU) to 7 hours.**

## About The Project
This repository contains an efficient end-to-end fine-tuning pipeline for the Qwen2.5-Coder-7B language model on consumer hardware.

The goal is to fine-tune a state-of-the-art coding model using 4-bit quantization and LoRA adapters to efficiently leverage available GPU VRAM.

## The Stack
- **Model:** [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)  
- **Dataset:** [Python Code Instructions (18k, Alpaca format)](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)  
- **Techniques:** QLoRA (4-bit quantization + LoRA), PEFT  
- **Libraries:** `PyTorch`, `Transformers`, `Accelerate`, `bitsandbytes`, `datasets`  
- **Environment:** WSL2 + CUDA

## Key Highlights
- **Modular Architecture:** Separated concerns with dedicated modules: `config.py` for settings, `model_utils.py` for model loading, `dataset_utils.py` for data handling.  
- **Memory Optimization:** Uses `bitsandbytes` for efficient 4-bit QLoRA training to minimize VRAM footprint.  
- **Easy Training:** Streamlined training pipeline orchestrated by `train.py` with built-in logging and checkpointing.

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


