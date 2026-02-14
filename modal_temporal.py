import modal

app = modal.App("temporal-intervention")

@app.function(
    gpu="T4",
    timeout=600,
    image=modal.Image.debian_slim().pip_install(
        "transformers", "torch", "accelerate", "bitsandbytes"
    )
)
def run_temporal_intervention():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from datetime import datetime
    import re
    
    print("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Store ground truth
    ground_truth = {
        'sobriety_days': 163,
        'sobriety_start': datetime(2025, 9, 4)
    }
    
    # Track generated tokens
    generated_tokens = []
    
    def intervention_hook(module, input, output):
        """
        Hook that can modify hidden states
        """
        hidden_states = output[0]
        
        # Get logits for next token
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        last_hidden = hidden_states[:, -1:, :]
        logits = model.lm_head(last_hidden)  # [batch, 1, vocab_size]
        
        # What token is about to be generated?
        top_token_id = logits.argmax(dim=-1).item()
        top_token = tokenizer.decode([top_token_id])
        
        generated_tokens.append(top_token)
        
        # Check if we're generating a wrong number
        recent_text = "".join(generated_tokens[-10:])
        
        # Pattern: "You are [NUMBER] days"
        if "you are" in recent_text.lower() and top_token.strip().isdigit():
            # Building a number - check if it's wrong
            current_number = "".join([t for t in generated_tokens[-5:] if t.strip().isdigit()])
            
            if current_number and int(current_number) != ground_truth['sobriety_days']:
                print(f"⚠️  WRONG NUMBER DETECTED: {current_number} (should be {ground_truth['sobriety_days']})")
                # In real version: modify logits here to steer toward correct number
        
        return output
    
    # Register hook
    model.model.layers[-1].register_forward_hook(intervention_hook)
    
    # Test WITHOUT context (will it generate wrong number?)
    print("\n" + "="*60)
    print("TEST 1: NO CONTEXT - Will it hallucinate?")
    print("="*60)
    
    generated_tokens.clear()
    
    prompt1 = "User: I got sober on September 4, 2025. How many days sober am I as of February 14, 2026?\nAssistant:"
    inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
    response1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nRESPONSE:")
    print(response1)
    
    # Test WITH context
    print("\n\n" + "="*60)
    print("TEST 2: WITH CONTEXT - Will it use our number?")
    print("="*60)
    
    generated_tokens.clear()
    
    context = "CRITICAL: User is EXACTLY 163 days sober as of February 14, 2026 (started September 4, 2025). Use this exact number."
    prompt2 = f"{context}\n\nUser: How many days sober am I?\nAssistant:"
    
    inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
    response2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nRESPONSE:")
    print(response2)
    
    return {
        "without_context": response1,
        "with_context": response2
    }


@app.local_entrypoint()
def main():
    result = run_temporal_intervention.remote()
    
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nWithout context (AI does math):")
    print(result["without_context"])
    print("\nWith context (AI uses our number):")
    print(result["with_context"])