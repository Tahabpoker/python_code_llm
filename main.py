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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_MODEL = "EleutherAI/pythia-70m"   
DATASET_PATH = "pc_code_dataset"
OUTPUT_DIR = "pythia_lora_out"

MAX_LENGTH = 256
BATCH = 1
EPOCHS = 1
GRAD_ACCUM = 8
LR = 3e-4

log.info("Torch CUDA available: %s, device count: %s",
         torch.cuda.is_available(), torch.cuda.device_count())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

log.info("Loaded model with 4-bit quantization")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
log.info("LoRA applied. Trainable params: %s",
         sum(p.numel() for p in model.parameters() if p.requires_grad))

# Load dataset
ds = load_from_disk(DATASET_PATH)

def preprocess(example):
    text = example["prompt"] + example["completion"]
    tok = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    tok["labels"] = tok["input_ids"].copy()
    return tok

log.info("Tokenizing (this will take some time)...")
tok_ds = ds.map(preprocess, batched=False)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    save_steps=200,
    fp16=False,
    bf16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=collator
)

trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
log.info("Training complete. Adapter saved.")
