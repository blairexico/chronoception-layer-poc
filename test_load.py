from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading Mistral 7B (will download ~14GB first time)...")

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded!")
print(f"Model has {len(model.model.layers)} transformer layers")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Device: {next(model.parameters()).device}")  # Check device

# Quick test - FIX: Move inputs to same device as model
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)  # ← Added .to()

outputs = model.generate(**inputs, max_new_tokens=20)

print("\nTest generation:")
print(tokenizer.decode(outputs[0]))