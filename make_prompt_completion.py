from datasets import load_from_disk
import re

SRC = "clean_code_dataset"
OUT = "pc_code_dataset"

ds = load_from_disk(SRC)

def split_signature(example):
    code = example.get("text", "")
    lines = code.splitlines()

    # find first def/class signature inside first 6 lines
    for i, ln in enumerate(lines[:6]):
        if re.match(r'^\s*(def|class)\s+\w+\s*\(.*\)\s*:', ln):
            prompt = "\n".join(lines[:i+1]) + "\n"
            completion = "\n".join(lines[i+1:]).lstrip("\n")
            return {"prompt": prompt, "completion": completion}

    # fallback when no signature found → skip
    return {"prompt": "", "completion": ""}

pc = ds.map(split_signature, remove_columns=ds.column_names, num_proc=4)
pc = pc.filter(lambda x: x["prompt"].strip() != "" and len(x["completion"].strip()) > 10, num_proc=4)
pc.save_to_disk(OUT)

print("Saved prompt-completion dataset to:", OUT)
