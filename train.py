"""
Training Script
===============

This is the main entry point for the fine-tuning pipeline.
It orchestrates the entire process:
1.  Loading configuration.
2.  Initializing tokenizer and dataset using `dataset_utils`.
3.  Loading and configuring the model using `model_utils`.
4.  Setting up training arguments.
5.  Executing the training loop via HuggingFace Trainer.
6.  Saving the final adapter and tokenizer.

Usage:
    Run directly from the command line:
    $ python train.py
"""

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

# Configure logging to standard output
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def train():
    """
    Main training function. Orchestrates the loading, training, and saving steps.
    """
    logger.info("=== Starting Python Code LLM Fine-tuning Pipeline ===")
    
    # -------------------------------------------------------------------------
    # 1. Load Tokenizer
    # -------------------------------------------------------------------------
    logger.info("Step 1/6: Loading Tokenizer")
    tokenizer = model_utils.load_tokenizer()
    
    # -------------------------------------------------------------------------
    # 2. Load Dataset
    # -------------------------------------------------------------------------
    logger.info("Step 2/6: Loading and Processing Dataset")
    # Load dataset using utility which handles chat formatting and tokenization
    dataset = dataset_utils.load_and_preprocess_dataset(tokenizer)
    
    # -------------------------------------------------------------------------
    # 3. Load Model
    # -------------------------------------------------------------------------
    logger.info("Step 3/6: Loading Base Model")
    model = model_utils.load_model()
    
    # -------------------------------------------------------------------------
    # 4. Apply LoRA
    # -------------------------------------------------------------------------
    logger.info("Step 4/6: Applying LoRA Adapters")
    model = model_utils.create_peft_config(model)
    
    # -------------------------------------------------------------------------
    # 5. Training Setup
    # -------------------------------------------------------------------------
    logger.info("Step 5/6: Configuring Trainer")
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        logging_steps=10,             # Log training loss every 10 steps
        save_strategy="epoch",        # Save checkpoint at the end of each epoch
        fp16=config.USE_FP16,         # Use mixed precision (FP16)
        bf16=config.USE_BF16,         # Use BF16 if supported
        optim="paged_adamw_32bit",    # Paged optimizer efficiently manages memory
        warmup_ratio=0.03,            # Warmup learning rate for first 3% of steps
        group_by_length=True,         # Group sequences of similar length for efficiency
        lr_scheduler_type="constant", # Constant LR is often good for QLoRA
        max_grad_norm=0.3,            # Gradient clipping to prevent exploding gradients
        report_to="none",             # Disable external reporting (wandb, etc.) unless configured
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        # Standard causal language modeling data collator. 
        # Handles masking automatically for standard CLM.
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # -------------------------------------------------------------------------
    # 6. Train and Save
    # -------------------------------------------------------------------------
    logger.info("Step 6/6: Starting Training Loop")
    trainer.train()
    
    logger.info("Training complete. Saving final model adapter...")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    logger.info(f"Success! Model saved to: {os.path.abspath(config.OUTPUT_DIR)}")

if __name__ == "__main__":
    train()
