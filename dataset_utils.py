"""
Dataset Utilities Module
========================

This module handles the loading, preprocessing, and formatting of datasets for instruction tuning.
It ensures that raw data is converted into the specific chat template format required by the 
Qwen2.5-Coder model/tokenizer.

Functions:
    load_and_preprocess_dataset: Main entry point for dataset preparation.
"""

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
import config
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)

def load_and_preprocess_dataset(tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Loads the instruction dataset and preprocesses it into the chat format
    expected by Qwen2.5-Coder-Instruct.

    This function performs the following steps:
    1. Loads the raw dataset from HuggingFace.
    2. Formats each example into a chat structure (System, User, Assistant).
    3. Applies the tokenizer's chat template.
    4. Tokenizes the text and prepares input_ids and attention_masks.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing text. 
                                         Must support chat templates.

    Returns:
        Dataset: The processed and tokenized dataset ready for training.
    """
    logger.info(f"Loading dataset: {config.DATASET_NAME} (split: {config.DATASET_SPLIT})")
    
    # Load dataset from HuggingFace Hub
    try:
        ds = load_dataset(config.DATASET_NAME, split=config.dataset_split)
    except Exception as e:
        logger.error(f"Failed to load dataset {config.DATASET_NAME}: {e}")
        raise e

    def format_chat(example: dict) -> dict:
        """
        Transforms a raw dataset example into a chat-template formatted string.
        
        Args:
            example (dict): A single record from the raw dataset containing 
                            'instruction', 'input', and 'output'.
                            
        Returns:
            dict: A dictionary containing the formatted 'text'.
        """
        # Construct the user content. Append context/input if available.
        user_content = example.get('instruction', '')
        if example.get('input'):
            user_content += f"\n\nContext:\n{example['input']}"
            
        # Define the conversation messages
        messages = [
            # System prompt establishes the persona
            {"role": "system", "content": "You are a helpful and expert Python programming assistant."},
            # User instruction
            {"role": "user", "content": user_content},
            # Model response (ground truth)
            {"role": "assistant", "content": example.get('output', '')}
        ]
        
        # Apply the tokenizer's chat template to convert list of messages into a single string.
        # tokenize=False ensures we get the string back for inspection/debugging if needed before tokenization.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text}

    logger.info("Formatting dataset with chat template...")
    # Apply formatting. num_proc=4 parallelizes the operation for speed.
    ds = ds.map(format_chat, num_proc=4)
    
    def tokenize_function(example: dict) -> dict:
        """
        Tokenizes the formatted text.
        
        Args:
            example (dict): An example containing the 'text' field.
            
        Returns:
            dict: Tokenized output including 'input_ids' and 'attention_mask'.
        """
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length", # Pad to max_length for consistent batch shapes
        )
        
    logger.info("Tokenizing dataset...")
    tokenized_ds = ds.map(tokenize_function, batched=True, num_proc=4)
    
    # Note: For Causal Language Modeling (CLM), the 'labels' are typically automatically 
    # handled by the DataCollatorForLanguageModeling by shifting input_ids.
    # If using a specialized collator like DataCollatorForCompletionOnlyLM (from trl),
    # we would need to be mindful of masking, but for standard Trainer + CLM collator,
    # this setup is sufficient for a baseline.
    
    return tokenized_ds
