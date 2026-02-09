# "qwen2.5_coder_7b_lora" Python Fine-Tuner

**Efficient fine-tuning of Qwen2.5-Coder-7B using QLoRA on consumer hardware.**

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

### 3. Train the Model
This script automatically loads the dataset, sets up LoRA adapters and 4-bit quantization, and begins training:

```bash
python train.py
```

Training logs and checkpoints will be saved to the output directory specified in `config.py`.

### 4. Run Inference
Test the fine-tuned model's ability to generate Python code:

```bash
python model_test.py
```

### Project Files Reference
- **`config.py`** - Central configuration (model, dataset, hyperparameters)
- **`train.py`** - Main training pipeline  
- **`model_utils.py`** - Model loading and configuration utilities  
- **`dataset_utils.py`** - Dataset loading and preprocessing utilities  
- **`model_test.py`** - Inference script for testing the trained model  
- **`requirement.txt`** - Python dependencies

---


