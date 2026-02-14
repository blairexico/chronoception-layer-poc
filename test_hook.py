from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

# HOOK: Intercept hidden states in last layer
def my_hook(module, input, output):
    hidden_states = output[0]
    print(f"🔍 Hook fired! Shape: {hidden_states.shape}, Device: {hidden_states.device}")
    return output

# Register hook
print(f"Registering hook on layer {len(model.model.layers)-1}...")
model.model.layers[-1].register_forward_hook(my_hook)

# Generate - FIX device placement
print("\nGenerating...\n")
inputs = tokenizer("I got sober on", return_tensors="pt").to(model.device)  # ← Fix
outputs = model.generate(**inputs, max_new_tokens=10)

print("\n" + "="*50)
print("Final:", tokenizer.decode(outputs[0], skip_special_tokens=True))