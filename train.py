import os
import logging
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import config
import dataset_utils
import model_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train():
    log.info("Starting training pipeline...")
    
    # 1. Load Tokenizer
    tokenizer = model_utils.load_tokenizer()
    
    # 2. Load Dataset
    log.info("Loading and processing dataset...")
    # Passing the generic tokenizer to preprocess, handled in dataset_utils
    # Note: Qwen tokenizer might require specific chat template handling, 
    # which is done in dataset_utils.load_and_preprocess_dataset
    dataset = dataset_utils.load_and_preprocess_dataset(tokenizer)
    
    # 3. Load Model
    log.info("Loading model...")
    model = model_utils.load_model()
    
    # 4. Apply LoRA
    log.info("Applying LoRA...")
    model = model_utils.create_peft_config(model)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        fp16=config.USE_FP16,
        bf16=config.USE_BF16,
        optim="paged_adamw_32bit", # Paged optimizer for better memory management
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none", # Change to "wandb" if you want logging
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 7. Train
    log.info("Starting training...")
    trainer.train()
    
    # 8. Save Model
    log.info("Saving model...")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    log.info(f"Model saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    train()
