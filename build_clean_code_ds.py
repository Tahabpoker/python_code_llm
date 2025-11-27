from datasets import load_from_disk

SRC = "dataset"        
OUT = "clean_code_dataset"

ds = load_from_disk(SRC)

if isinstance(ds, dict) and "train" in ds:
    split = ds["train"]
else:
    split = ds

def keep_code(example):
    code = example.get("code") or example.get("original_string") or example.get("summary") or ""
    return {"text": code}

clean = split.map(keep_code, remove_columns=split.column_names, num_proc=4)
clean = clean.filter(lambda x: len(x["text"].strip()) > 0, num_proc=4)
clean.save_to_disk(OUT)

print("Saved cleaned dataset to:", OUT)
