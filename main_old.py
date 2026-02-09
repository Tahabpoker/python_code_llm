import os
import torch
import logging
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Configuration Constants ---
# Model checkpoint to fine-tune
BASE_MODEL = "EleutherAI/pythia-70m"   
# Path to the dataset directory
DATASET_PATH = "pc_code_dataset"
# Output directory for the fine-tuned model adapter
OUTPUT_DIR = "pythia_lora_out"

# Hyperparameters
MAX_LENGTH = 256
BATCH = 1          # Batch size per device
EPOCHS = 1         # Number of training epochs
GRAD_ACCUM = 8     # Gradient accumulation steps to simulate larger effective batch size
LR = 3e-4          # Learning rate

log.info("Torch CUDA available: %s, device count: %s",
         torch.cuda.is_available(), torch.cuda.device_count())

# --- Model Loading with Quantization ---
# Configure BitsAndBytes for 4-bit quantization (reduces memory usage)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # Normalized Float 4
    bnb_4bit_use_double_quant=True,  # usage of double quantization
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Ensure padding token is set (required for some models like Pythia/GPT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

log.info("Loaded model with 4-bit quantization")

# --- LoRA (Low-Rank Adaptation) Configuration ---
lora_config = LoraConfig(
    r=8,                  # Rank of the low-rank matrices
    lora_alpha=16,        # Scaling factor for LoRA
    target_modules=["q_proj", "v_proj"], # Projection layers to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA adapter to the model
model = get_peft_model(model, lora_config)
log.info("LoRA applied. Trainable params: %s",
         sum(p.numel() for p in model.parameters() if p.requires_grad))

# --- Dataset Preparation ---
# Load dataset from disk
ds = load_from_disk(DATASET_PATH)

def preprocess(example):
    """
    Tokenizes the input prompt and completion.
    Sets labels equal to input_ids for Causal Language Modeling.
    """
    text = example["prompt"] + example["completion"]
    tok = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    tok["labels"] = tok["input_ids"].copy()
    return tok

log.info("Tokenizing (this will take some time)...")
tok_ds = ds.map(preprocess, batched=False)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# --- Training Configuration ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    save_steps=200,
    fp16=False,
    bf16=torch.cuda.is_available(), # Use bfloat16 if GPU supports it
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=collator
)

# --- Train and Save ---
trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
log.info("Training complete. Adapter saved.")
