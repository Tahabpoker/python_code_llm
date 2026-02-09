import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import config
import logging

log = logging.getLogger(__name__)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map=config.DEVICE_MAP,
        trust_remote_code=True,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    return model

def create_peft_config(model):
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=config.TASK_TYPE
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params, all_param = model.get_nb_trainable_parameters()
    log.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    
    return model
