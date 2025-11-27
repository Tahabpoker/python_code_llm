import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "EleutherAI/pythia-70m"
ADAPTER = "pythia_lora_out/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER)

prompt = "def binary_search(arr, target):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    top_p=0.95,
    temperature=0.7
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
