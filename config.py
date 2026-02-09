import torch

# --- Model & Tokenizer Configuration ---
# Using Qwen2.5-Coder-7B-Instruct as the base model
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# --- Dataset Configuration ---
# Using a high-quality Python instruction dataset
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
dataset_split = "train" # The dataset usually comes with a 'train' split

# --- Training Configuration ---
OUTPUT_DIR = "qwen2.5_coder_7b_lora"
MAX_LENGTH = 2048  # Increased context length for better code understanding
BATCH_SIZE = 1     # Batch size per device (keep small for 7B model on consumer GPU)
GRAD_ACCUM = 16    # Increase gradient accumulation to simulate effective batch size of 16
EPOCHS = 1
LEARNING_RATE = 2e-4

# --- LoRA Configuration ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers for better performance
TASK_TYPE = "CAUSAL_LM"

# --- Hardware Configuration ---
USE_FP16 = True
USE_BF16 = torch.cuda.is_bf16_supported()
DEVICE_MAP = "auto"

# --- Quantization Configuration ---
LOAD_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
