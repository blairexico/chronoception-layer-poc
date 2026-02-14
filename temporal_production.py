import modal
from modal import Volume
import sqlite3
from datetime import datetime
import uuid
import torch
from typing import List, Dict, Optional

app = modal.App("temporal-production")

volume = Volume.from_name("temporal-facts-db", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch", 
    "accelerate",
    "bitsandbytes"
)

# ============================================
# DATABASE LAYER
# ============================================

class TemporalDatabase:
    """Manages persistent temporal facts"""
    
    def __init__(self, db_path: str = "/data/temporal_facts.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.epoch = datetime(1970, 1, 1)
        self._initialize_schema()
    
    def _initialize_schema(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS temporal_facts (
                fact_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                base_date TEXT NOT NULL,
                position_absolute INTEGER NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS fact_aliases (
                alias_id TEXT PRIMARY KEY,
                fact_id TEXT NOT NULL,
                alias_text TEXT NOT NULL,
                usage_count INTEGER DEFAULT 1,
                FOREIGN KEY (fact_id) REFERENCES temporal_facts(fact_id)
            )
        ''')
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_position 
            ON temporal_facts(user_id, position_absolute)
        ''')
        
        self.conn.commit()
    
    def add_fact(
        self,
        user_id: str,
        canonical_name: str,
        fact_type: str,
        base_date: datetime,
        aliases: Optional[List[str]] = None
    ) -> str:
        """Store a new temporal fact"""
        fact_id = str(uuid.uuid4())
        
        if isinstance(base_date, str):
            base_date = datetime.fromisoformat(base_date)
        
        position_abs = (base_date - self.epoch).days
        now = datetime.now().isoformat()
        
        self.conn.execute('''
            INSERT INTO temporal_facts 
            (fact_id, user_id, canonical_name, fact_type, base_date, 
             position_absolute, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (fact_id, user_id, canonical_name, fact_type, 
              base_date.isoformat(), position_abs, now, now))
        
        if aliases:
            for alias in aliases:
                self.conn.execute('''
                    INSERT INTO fact_aliases (alias_id, fact_id, alias_text)
                    VALUES (?, ?, ?)
                ''', (str(uuid.uuid4()), fact_id, alias.lower()))
        
        self.conn.commit()
        return fact_id
    
    def get_facts_for_user(self, user_id: str) -> List[Dict]:
        """Get all facts for a user"""
        rows = self.conn.execute(
            "SELECT * FROM temporal_facts WHERE user_id = ?",
            [user_id]
        ).fetchall()
        return [dict(row) for row in rows]
    
    def find_fact_by_alias(self, user_id: str, alias_text: str) -> Optional[Dict]:
        """Resolve alias to fact"""
        row = self.conn.execute('''
            SELECT f.* FROM temporal_facts f
            JOIN fact_aliases a ON f.fact_id = a.fact_id
            WHERE f.user_id = ? AND LOWER(a.alias_text) = LOWER(?)
        ''', (user_id, alias_text)).fetchone()
        
        return dict(row) if row else None


# ============================================
# TEMPORAL DISTANCE LAYER
# ============================================

class TemporalDistanceLayer:
    """Manages NOW and distance calculations"""
    
    def __init__(self):
        self.now_position: Optional[int] = None
        self.epoch = datetime(1970, 1, 1)
    
    def update_now(self, current_date: Optional[datetime] = None):
        """Update NOW position"""
        if current_date is None:
            current_date = datetime.now()
        
        self.now_position = (current_date - self.epoch).days
        print(f"📅 NOW: {current_date.strftime('%Y-%m-%d')}")
    
    def compute_duration(self, base_position_absolute: int) -> int:
        """Days since base_date"""
        if self.now_position is None:
            raise ValueError("NOW not initialized")
        return self.now_position - base_position_absolute


# ============================================
# AUTOMATIC FACT EXTRACTOR
# ============================================

class TemporalFactExtractor:
    """Extracts temporal facts from natural language"""
    
    def __init__(self, model, tokenizer, db: TemporalDatabase, temporal_layer: TemporalDistanceLayer):
        self.model = model
        self.tokenizer = tokenizer
        self.db = db
        self.temporal_layer = temporal_layer
    
    def extract_and_store(self, user_id: str, message: str) -> List[str]:
        """
        Extract temporal facts from user message and store them.
        
        Returns:
            List of fact IDs that were created
        """
        
        # Ask AI to extract temporal facts
        extraction_prompt = f"""[INST] Extract temporal facts from this message. Find any dates, events, or milestones mentioned.

Current date: {datetime.now().strftime('%B %d, %Y')}

Message: "{message}"

For each temporal fact found, provide:
- event_name: brief name (e.g., "blog_start", "moved_cities")
- event_date: YYYY-MM-DD format
- aliases: comma-separated alternative names

Format as simple text, one fact per line:
event_name | YYYY-MM-DD | alias1, alias2

If no temporal facts found, respond with "none"
[/INST]"""
        
        inputs = self.tokenizer(extraction_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("[/INST]")[-1].strip()
        
        print(f"\n🔍 Extraction result:\n{answer}\n")
        
        if "none" in answer.lower():
            return []
        
        # Parse extracted facts
        fact_ids = []
        lines = answer.strip().split('\n')
        
        for line in lines:
            try:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    event_name = parts[0]
                    event_date_str = parts[1]
                    aliases = [a.strip() for a in parts[2].split(',')] if len(parts) > 2 else []
                    
                    # Parse date
                    event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                    
                    # Normalize event name
                    canonical_name = event_name.lower().replace(' ', '_')
                    
                    # Check if fact already exists
                    existing = self.db.find_fact_by_alias(user_id, canonical_name)
                    
                    if existing:
                        print(f"   ⚠️  Fact already exists: {canonical_name}")
                        continue
                    
                    # Store new fact
                    fact_id = self.db.add_fact(
                        user_id=user_id,
                        canonical_name=canonical_name,
                        fact_type="point_in_time",
                        base_date=event_date,
                        aliases=[canonical_name.replace('_', ' ')] + aliases
                    )
                    
                    fact_ids.append(fact_id)
                    
                    # Calculate how long ago
                    days_ago = self.temporal_layer.compute_duration(
                        (event_date - self.temporal_layer.epoch).days
                    )
                    
                    print(f"   ✅ Stored: {canonical_name} ({days_ago} days ago)")
            
            except Exception as e:
                print(f"   ⚠️  Failed to parse line: {line} - {e}")
                continue
        
        return fact_ids
    

# ============================================
# CONVERSATIONAL CHAT WITH AUTO-LEARNING
# ============================================

@app.function(
    gpu="T4",
    volumes={"/data": volume},
    image=image,
    timeout=900
)
def conversational_chat(user_id: str, messages: List[Dict[str, str]]):
    """
    Multi-turn conversation with automatic temporal fact learning.
    
    Args:
        user_id: Unique user identifier
        messages: List of {"role": "user"/"assistant", "content": "..."}
        
    Returns:
        Assistant's response + learned facts
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "="*60)
    print(f"CONVERSATIONAL CHAT - User: {user_id}")
    print("="*60)
    
    # Initialize components
    db = TemporalDatabase()
    temporal = TemporalDistanceLayer()
    temporal.update_now()
    
    # Load model
    print("\n🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
        dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create extractor
    extractor = TemporalFactExtractor(model, tokenizer, db, temporal)
    
    # Get last user message
    last_user_message = [m for m in messages if m['role'] == 'user'][-1]['content']
    
    # Extract temporal facts from user message
    print("\n📝 Checking for temporal facts...")
    new_fact_ids = extractor.extract_and_store(user_id, last_user_message)
    
    if new_fact_ids:
        print(f"✅ Learned {len(new_fact_ids)} new facts")
    
    # Load all known facts for context
    all_facts = db.get_facts_for_user(user_id)
    
    # Build temporal context
    context_lines = []
    if all_facts:
        context_lines.append("What I know about you:")
        for fact in all_facts:
            days = temporal.compute_duration(fact['position_absolute'])
            context_lines.append(
                f"- {fact['canonical_name'].replace('_', ' ')}: {days} days ago"
            )
    
    temporal_context = "\n".join(context_lines) if context_lines else ""
    
    # Build conversation prompt
    conversation = ""
    for msg in messages:
        if msg['role'] == 'user':
            conversation += f"User: {msg['content']}\n"
        else:
            conversation += f"Assistant: {msg['content']}\n"
    
    # Add current turn
    prompt = f"""[INST] {temporal_context}

{conversation}Assistant: [/INST]"""
    
    print("\n💬 Generating response...")
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("[/INST]")[-1].strip()
    
    print(f"\n🤖 Response: {answer}\n")
    
    return {
        "response": answer,
        "learned_facts": len(new_fact_ids),
        "total_facts": len(all_facts),
        "temporal_context": temporal_context
    }

@app.local_entrypoint()
def demo_conversation():
    """Demo multi-turn conversation with learning"""
    
    user_id = "demo_user"
    
    print("\n" + "="*60)
    print("TEMPORAL AI - CONVERSATIONAL DEMO")
    print("="*60)
    
    # Turn 1: User mentions starting a blog
    print("\n\n" + "="*60)
    print("TURN 1")
    print("="*60)
    
    result1 = conversational_chat.remote(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"}
        ]
    )
    
    print(f"Assistant: {result1['response']}")
    print(f"Learned: {result1['learned_facts']} facts")
    
    # Turn 2: User mentions another event
    print("\n\n" + "="*60)
    print("TURN 2")
    print("="*60)
    
    result2 = conversational_chat.remote(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": result1['response']},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"}
        ]
    )
    
    print(f"Assistant: {result2['response']}")
    print(f"Learned: {result2['learned_facts']} new facts")
    print(f"Total known: {result2['total_facts']} facts")
    
    # Turn 3: Ask about time since blog
    print("\n\n" + "="*60)
    print("TURN 3 - Query temporal fact")
    print("="*60)
    
    result3 = conversational_chat.remote(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I started my blog on January 15, 2024"},
            {"role": "assistant", "content": result1['response']},
            {"role": "user", "content": "I also moved to a new city on June 1st, 2024"},
            {"role": "assistant", "content": result2['response']},
            {"role": "user", "content": "How long has it been since I started my blog?"}
        ]
    )
    
    print(f"Assistant: {result3['response']}")
    
    print("\n" + "="*60)
    print("TEMPORAL CONTEXT:")
    print(result3['temporal_context'])
    print("="*60)
    
    print("\n✅ Demo complete - facts learned and persisted!")