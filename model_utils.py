"""
Model Utilities Module
======================

This module provides helper functions for loading, configuring, and preparing large language models
for fine-tuning. It handles:
- Tokenizer loading with proper padding configuration.
- Base model loading with 4-bit quantization (QLoRA).
- Preparation of models for k-bit training.
- Application of LoRA (Low-Rank Adaptation) adapters.

Functions:
    load_tokenizer: Loads the tokenizer for the base model.
    load_model: Loads the quantized base model.
    create_peft_config: Configures and applies LoRA to the base model.
"""

import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import config

# Configure logging
logger = logging.getLogger(__name__)

def load_tokenizer() -> PreTrainedTokenizer:
    """
    Loads the tokenizer associated with the BASE_MODEL defined in config.py.
    
    Ensures that a padding token is set, defaulting to the EOS token if none exists.
    This is critical for batched training where sequences must be of equal length.

    Returns:
        PreTrainedTokenizer: The loaded and configured tokenizer.
    """
    logger.info(f"Loading tokenizer for model: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL, 
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad_token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

def load_model() -> PreTrainedModel:
    """
    Loads the base model with 4-bit quantization configuration using BitsAndBytes.
    
    This function:
    1. Configures the BitsAndBytesConfig for 4-bit loading (QLoRA).
    2. Loads the AutoModelForCausalLM.
    3. Prepares the model for k-bit training (stabilizes layers for quantization).
    4. Enables gradient checkpointing to save memory during training.

    Returns:
        PreTrainedModel: The loaded 4-bit quantized model ready for LoRA application.
    """
    logger.info("Configuring BitsAndBytes for 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE
    )

    logger.info(f"Loading base model: {config.BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map=config.DEVICE_MAP,
        trust_remote_code=True,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE
    )
    
    # Prepare model for k-bit training
    # This wraps layers to ensure stability during training with quantized weights
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing
    # This trades compute for memory by re-calculating activations during the backward pass
    model.gradient_checkpointing_enable()

    return model

def create_peft_config(model: PreTrainedModel) -> PeftModel:
    """
    Configures and applies Low-Rank Adaptation (LoRA) to the model.
    
    This function:
    1. Defines the LoraConfig based on parameters in config.py.
    2. Wraps the base model with the PeftModel class.
    3. Logs the number of trainable parameters vs total parameters.

    Args:
        model (PreTrainedModel): The base model to apply LoRA to.

    Returns:
        PeftModel: The model with LoRA adapters attached.
    """
    logger.info("Creating LoRA configuration...")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=config.TASK_TYPE
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters for verification
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA Applied:\n"
        f"  - Trainable params: {trainable_params:,d}\n"
        f"  - All params: {all_param:,d}\n"
        f"  - Trainable %: {100 * trainable_params / all_param:.4f}%"
    )
    
    return model
