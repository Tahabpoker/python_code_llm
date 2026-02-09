"""
Configuration Module
====================

This module centralizes all configuration parameters for the Python Code LLM fine-tuning pipeline.
It includes settings for:
- Model selection and tokenizer
- Dataset paths and splitting
- Training hyperparameters (batch size, learning rate, epochs)
- LoRA (Low-Rank Adaptation) configuration
- Hardware and quantization settings

Rationale:
    Centralizing configuration allows for easier experiments and reproducibility. 
    Changes can be made in one place without modifying the core logic in training scripts.
"""

import torch

# =============================================================================
# Model & Tokenizer Configuration
# =============================================================================

# The base model to be fine-tuned. 
# We use Qwen2.5-Coder-7B-Instruct, a state-of-the-art coding model 
# that outperforms many larger models in code generation tasks.
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# =============================================================================
# Dataset Configuration
# =============================================================================

# The dataset used for instruction tuning.
# 'iamtarun/python_code_instructions_18k_alpaca' provides ~18k Python-specific
# instruction-response pairs, improving the model's ability to act as a coding assistant.
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"

# The specific split of the dataset to use for training.
DATASET_SPLIT = "train"

# =============================================================================
# Training Configuration
# =============================================================================

# Directory where the fine-tuned model adapter and logs will be saved.
OUTPUT_DIR = "qwen2.5_coder_7b_lora"

# Maximum sequence length for the model.
# Qwen2.5 supports up to 32k context, but we limit to 2048 to fit in consumer GPU memory
# while still allowing for substantial code contexts.
MAX_LENGTH = 2048

# Batch size per GPU.
# Set to 1 to minimize VRAM usage for the 7B parameter model.
BATCH_SIZE = 1

# Gradient accumulation steps.
# simulates a larger effective batch size (BATCH_SIZE * GRAD_ACCUM).
# 1 * 16 = 16 effective batch size, helping with training stability.
GRAD_ACCUM = 16

# Number of training epochs.
# 1 epoch is often sufficient for instruction tuning on a specific domain to avoid overfitting.
EPOCHS = 1

# Learning rate for the optimizer.
# 2e-4 is a standard starting point for QLoRA fine-tuning.
LEARNING_RATE = 2e-4

# =============================================================================
# LoRA (Low-Rank Adaptation) Configuration
# =============================================================================

# Rank of the low-rank matrices. Higher 'r' means more trainable parameters.
# 16 is a good balance between performance and parameter efficiency.
LORA_R = 16

# Scaling factor for LoRA updates. Typically set to 2x rank (r).
LORA_ALPHA = 32

# Dropout probability for LoRA layers to prevent overfitting.
LORA_DROPOUT = 0.05

# Modules to apply LoRA to.
# We target all linear projection layers in the attention mechanism and MLP 
# to maximize the model's adaptability to the new domain.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]

# Task type for the PEFT configuration.
TASK_TYPE = "CAUSAL_LM"

# =============================================================================
# Hardware & Quantization Configuration
# =============================================================================

# Use FP16 (half-precision) for training if BF16 is not available.
USE_FP16 = True

# Use BF16 (Brain Floating Point) if the hardware supports it (e.g., Ampere GPUs).
# BF16 offers better numerical stability than FP16.
USE_BF16 = torch.cuda.is_bf16_supported()

# Device mapping strategy. "auto" allows Accelerate to handle device placement.
DEVICE_MAP = "auto"

# Enable loading the model in 4-bit precision (QLoRA).
# This drastically reduces memory usage, allowing 7B models to run on consumer GPUs.
LOAD_IN_4BIT = True

# Compute data type for 4-bit quantization.
# Should match the training precision (BF16 or FP16).
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# Quantization type. "nf4" (Normalized Float 4) is recommended for QLoRA.
BNB_4BIT_QUANT_TYPE = "nf4"

# Use nested quantization (double quantization) to further reduce memory usage 
# with minimal performance degradation.
BNB_4BIT_USE_DOUBLE_QUANT = True
