from mlx_lm import load, generate
from datetime import datetime, timedelta
import json
import re

class SelfLearningTemporalChat:
    def __init__(self):
        print("Loading Mistral...")
        self.model, self.tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        print("✓ Model loaded!\n")
        self.temporal_db = {}
    
    def chat(self, user_id, message):
        current_time = datetime.now()
        
        new_facts = self._extract_temporal_facts(message, current_time)
        
        if new_facts:
            self._store_facts(user_id, new_facts)
            print(f"\n[📝 STORED {len(new_facts)} temporal facts]")
        
        context = self._build_context(user_id, current_time)
        
        prompt = f"""[INST] You are a helpful assistant.

Current date: {current_time.strftime('%B %d, %Y')}

CRITICAL: All dates and day counts below are PRE-CALCULATED. 
Use these exact numbers - do NOT recalculate or do date math.

Facts about user:
{context}

User: {message} [/INST]"""
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )
        
        return response
    
    def _extract_temporal_facts(self, message, current_time):
        """Extract temporal facts (with manual fallback)"""
        
        # Try AI extraction first
        extraction_prompt = f"""[INST] Extract temporal facts.

Current date: {current_time.strftime('%B %d, %Y')}
Message: "{message}"

Return JSON only:
{{
    "facts": [
        {{
            "text": "quote",
            "label": "days_since" | "scheduled_event" | "milestone",
            "description": "what",
            "event_date": "YYYY-MM-DD"
        }}
    ]
}}

If none: {{"facts": []}}
[/INST]"""
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=extraction_prompt,
            max_tokens=300,
            verbose=False
        )
        
        try:
            clean = response.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0]
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0]
            
            clean = clean.strip()
            start = clean.find('{')
            end = clean.rfind('}') + 1
            if start >= 0 and end > start:
                clean = clean[start:end]
            
            data = json.loads(clean)
            facts = data.get('facts', [])
            
            for fact in facts:
                fact['observed_date'] = current_time
                
                if 'event_date' in fact and fact['event_date']:
                    try:
                        fact['event_date'] = datetime.strptime(fact['event_date'], '%Y-%m-%d')
                    except:
                        fact['event_date'] = None
            
            return facts if facts else self._manual_extraction_fallback(message, current_time)
        
        except Exception as e:
            print(f"[⚠️  JSON parse error: {e}]")
            return self._manual_extraction_fallback(message, current_time)
    
    def _manual_extraction_fallback(self, message, current_time):
        """Manual regex extraction"""
        facts = []
        
        # Pattern: "got sober on [date]"
        pattern = r'(?:got sober|became sober|started sobriety).*?on\s+(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})'
        match = re.search(pattern, message, re.IGNORECASE)
        
        if match:
            date_str = match.group(1)
            try:
                for fmt in ['%B %d, %Y', '%B %dth, %Y', '%B %dst, %Y', '%B %dnd, %Y', '%B %drd, %Y']:
                    try:
                        event_date = datetime.strptime(date_str.replace('st,', ',').replace('nd,', ',').replace('rd,', ',').replace('th,', ','), fmt)
                        facts.append({
                            'text': match.group(0),
                            'label': 'milestone',
                            'description': 'Sobriety start',
                            'event_date': event_date,
                            'observed_date': current_time
                        })
                        print(f"[✓ Manual extraction: Sobriety started {event_date.date()}]")
                        break
                    except:
                        continue
            except:
                pass
        
        return facts
    
    def _store_facts(self, user_id, facts):
        if user_id not in self.temporal_db:
            self.temporal_db[user_id] = []
        self.temporal_db[user_id].extend(facts)
    
    def _build_context(self, user_id, current_time):
        """Build context with COMPUTED numbers (AI doesn't calculate)"""
        if user_id not in self.temporal_db or not self.temporal_db[user_id]:
            return "No prior context."
        
        lines = []
        for fact in self.temporal_db[user_id]:
            
            if fact['label'] == 'days_since' and fact.get('base_date'):
                days = (current_time - fact['base_date']).days
                lines.append(
                    f"- {fact['description']}: {days} days (started {fact['base_date'].strftime('%b %d, %Y')})"
                )
            
            elif fact['label'] == 'milestone' and fact.get('event_date'):
                # Check if this is a sobriety-related milestone
                if any(word in fact['description'].lower() for word in ['sober', 'sobriety', 'clean']):
                    days = (current_time - fact['event_date']).days
                    lines.append(
                        f"- Sobriety: {days} days (started {fact['event_date'].strftime('%b %d, %Y')})"
                    )
                else:
                    lines.append(
                        f"- {fact['description']}: {fact['event_date'].strftime('%b %d, %Y')}"
                    )
            
            elif fact['label'] == 'scheduled_event' and fact.get('event_date'):
                if fact['event_date'] < current_time:
                    days_ago = (current_time - fact['event_date']).days
                    lines.append(
                        f"- {fact['description']}: occurred {days_ago} days ago"
                    )
                else:
                    days_until = (fact['event_date'] - current_time).days
                    lines.append(
                        f"- {fact['description']}: upcoming in {days_until} days"
                    )
        
        return "\n".join(lines)
    
    def show_db(self, user_id):
        print(f"\n{'='*50}")
        print(f"TEMPORAL DATABASE for {user_id}")
        print('='*50)
        
        if user_id not in self.temporal_db or not self.temporal_db[user_id]:
            print("(empty)")
            return
        
        for i, fact in enumerate(self.temporal_db[user_id], 1):
            print(f"\n{i}. [{fact['label']}] {fact['description']}")
            print(f"   \"{fact['text']}\"")
            if fact.get('base_date'):
                print(f"   Base: {fact['base_date'].date()}")
            if fact.get('event_date'):
                print(f"   Event: {fact['event_date'].date()}")


if __name__ == "__main__":
    chat = SelfLearningTemporalChat()
    
    print("="*50)
    print("TEMPORAL AI - Ready!")
    print("="*50)
    
    # Test with actual date
    print("\n💬 User: I got sober on September 4th, 2025")
    response = chat.chat("blair", "I got sober on September 4th, 2025")
    print(f"🤖 Assistant: {response[:150]}...")
    
    chat.show_db("blair")
    
    # Ask for count - should use our computed value
    print("\n\n💬 User: How many days sober am I now?")
    response = chat.chat("blair", "How many days sober am I now?")
    print(f"🤖 Assistant: {response}")
    
    print("\n✅ Check: Should say 163 days (we calculated it, not the AI)")
