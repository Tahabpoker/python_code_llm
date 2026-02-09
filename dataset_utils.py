from datasets import load_dataset
import config

def load_and_preprocess_dataset(tokenizer):
    """
    Loads the instruction dataset and preprocesses it into the chat format
    expected by Qwen2.5-Coder-Instruct.
    """
    print(f"Loading dataset: {config.DATASET_NAME}")
    ds = load_dataset(config.DATASET_NAME, split=config.dataset_split)

    def format_chat(example):
        # The dataset has 'instruction', 'input', 'output', 'prompt' columns.
        # We will use 'instruction', 'input' (optional context), and 'output'.
        
        user_content = example['instruction']
        if example.get('input'):
            user_content += f"\n\nContext:\n{example['input']}"
            
        messages = [
            {"role": "system", "content": "You are a helpful and expert Python programming assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text}

    print("Formatting dataset with chat template...")
    # Use config for num_proc if needed, defaulting to 4 for speed
    ds = ds.map(format_chat, num_proc=4)
    
    # Tokenize
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length", # Pad to max_length for consistent batching
        )
        
    print("Tokenizing dataset...")
    tokenized_ds = ds.map(tokenize_function, batched=True, num_proc=4)
    
    # We need 'input_ids', 'attention_mask', and 'labels'
    # For Causal LM, labels are usually input_ids (shifted inside the model)
    # However, for instruction tuning, we might want to mask the user prompt in the loss
    # For simplicity in this script, we'll let DataCollatorForLanguageModeling handle it 
    # (or DataCollatorForCompletionOnlyLM if we were using TRL, but we stick to Trainer for now)
    
    return tokenized_ds

