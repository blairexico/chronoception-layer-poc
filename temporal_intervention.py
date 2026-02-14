from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime

class TemporalInterventionModel:
    def __init__(self):
        print("Loading model with temporal intervention...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Temporal facts storage
        self.temporal_facts = {}
        
        # Track what we're generating
        self.current_user = None
        self.generation_active = False
        
        # Register intervention hook on LAST layer
        self.model.model.layers[-1].register_forward_hook(self._intervention_hook)
        
        print("✓ Loaded with temporal intervention layer")
    
    def add_fact(self, user_id, fact_type, **kwargs):
        """Store temporal fact"""
        if user_id not in self.temporal_facts:
            self.temporal_facts[user_id] = []
        
        self.temporal_facts[user_id].append({
            'type': fact_type,
            **kwargs
        })
        print(f"✓ Stored: {fact_type} for {user_id}")
    
    def _intervention_hook(self, module, input, output):
        """
        Fires during generation
        Can modify hidden states here
        """
        if not self.generation_active:
            return output
        
        hidden_states = output[0]  # [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        
        # Ensure 3D
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        # Get last token's hidden state
        last_hidden = hidden_states[:, -1:, :]  # [batch, 1, hidden_dim]
        
        # Simple detection: Check if model is about to generate numbers
        # (In real version, this would be a learned detector)
        logits = self.model.lm_head(last_hidden)  # [batch, 1, vocab_size]
        top_token_id = logits.argmax(dim=-1)
        top_token = self.tokenizer.decode(top_token_id[0])
        
        # Debug: show what token is about to be generated
        if top_token.strip().isdigit():
            print(f"⚠️  About to generate number: '{top_token}'")
        
        return output
    
    def generate(self, user_id, prompt, max_tokens=50):
        """Generate with temporal awareness"""
        self.current_user = user_id
        self.generation_active = True
        
        # Build context from facts
        context = self._build_context(user_id)
        
        # Prepend context to prompt
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        print(f"\n{'='*50}")
        print("PROMPT:")
        print(full_prompt)
        print('='*50 + "\n")
        
        # Tokenize and move to device
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        self.generation_active = False
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    def _build_context(self, user_id):
        """Build context with PRE-COMPUTED temporal values"""
        if user_id not in self.temporal_facts:
            return ""
        
        current_time = datetime.now()
        lines = ["FACTS (pre-calculated, DO NOT recalculate):"]
        
        for fact in self.temporal_facts[user_id]:
            if fact['type'] == 'sobriety_start':
                days = (current_time - fact['start_date']).days
                lines.append(f"- User is {days} days sober (started {fact['start_date'].strftime('%B %d, %Y')})")
        
        return "\n".join(lines)


# ============================================
# TEST IT
# ============================================

if __name__ == "__main__":
    model = TemporalInterventionModel()
    
    # Add Blair's sobriety fact
    model.add_fact(
        "blair",
        "sobriety_start",
        start_date=datetime(2025, 9, 4)
    )
    
    # Test 1: Ask how many days
    print("\n" + "="*60)
    print("TEST 1: Direct question about sobriety")
    print("="*60)
    
    response = model.generate(
        "blair",
        "User: How many days sober am I as of February 14, 2026?\nAssistant:",
        max_tokens=30
    )
    
    print("\nRESPONSE:")
    print(response)
    print("\n✅ Check: Should mention 163 days (not calculate wrong)")
    
    # Test 2: Casual mention
    print("\n\n" + "="*60)
    print("TEST 2: Casual conversation (should use context)")
    print("="*60)
    
    response = model.generate(
        "blair",
        "User: What should I focus on this week?\nAssistant:",
        max_tokens=100
    )
    
    print("\nRESPONSE:")
    print(response)
    print("\n✅ Check: Should mention 163 days in context (not stale number)")