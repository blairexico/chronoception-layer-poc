"""
Chronoception - Temporal awareness for LLMs via generation-time intervention.

Core system: persistent fact storage, temporal distance calculations,
automatic fact extraction, and conversational chat with context injection.

For logits steering (Method B), see intervention.py.
"""

import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

EPOCH = datetime(1970, 1, 1)


# ============================================
# TYPES
# ============================================

@dataclass
class TemporalFact:
    """A single temporal fact about a user."""
    fact_id: str
    user_id: str
    canonical_name: str
    fact_type: str
    base_date: datetime
    position_absolute: int  # days since epoch
    confidence: float = 1.0
    aliases: List[str] = field(default_factory=list)


@dataclass
class ExtractedFact:
    """A fact parsed from user text, before storage."""
    event_name: str
    event_date: datetime
    aliases: List[str] = field(default_factory=list)
    fact_type: str = "point_in_time"


@dataclass
class ChatResult:
    """Result from a conversational chat turn."""
    response: str
    learned_facts: int
    total_facts: int
    temporal_context: str
    intervention_active: bool = False


class TemporalError(Exception):
    pass

class DatabaseError(TemporalError):
    pass


# ============================================
# DATABASE - Persistent fact storage (SQLite)
# ============================================

class TemporalDatabase:
    """Stores temporal facts with absolute positions for O(1) distance queries."""

    def __init__(self, db_path: str = "/data/temporal_facts.db"):
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self._init_schema()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to {db_path}") from e

    def _init_schema(self) -> None:
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
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Store a new temporal fact. Returns fact_id."""
        fact_id = str(uuid.uuid4())
        if isinstance(base_date, str):
            base_date = datetime.fromisoformat(base_date)

        position_abs = (base_date - EPOCH).days
        now = datetime.now().isoformat()

        try:
            self.conn.execute('''
                INSERT INTO temporal_facts
                (fact_id, user_id, canonical_name, fact_type, base_date,
                 position_absolute, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (fact_id, user_id, canonical_name, fact_type,
                  base_date.isoformat(), position_abs, now, now))

            for alias in (aliases or []):
                self.conn.execute(
                    'INSERT INTO fact_aliases (alias_id, fact_id, alias_text) VALUES (?, ?, ?)',
                    (str(uuid.uuid4()), fact_id, alias.lower())
                )
            self.conn.commit()
            logger.info("Stored: %s for user %s", canonical_name, user_id)
            return fact_id

        except sqlite3.Error as e:
            self.conn.rollback()
            raise DatabaseError(f"Failed to store fact '{canonical_name}'") from e

    def get_facts_for_user(self, user_id: str) -> List[TemporalFact]:
        """Get all facts for a user."""
        rows = self.conn.execute(
            "SELECT * FROM temporal_facts WHERE user_id = ?", [user_id]
        ).fetchall()

        facts = []
        for row in rows:
            r = dict(row)
            alias_rows = self.conn.execute(
                "SELECT alias_text FROM fact_aliases WHERE fact_id = ?",
                [r['fact_id']]
            ).fetchall()

            facts.append(TemporalFact(
                fact_id=r['fact_id'],
                user_id=r['user_id'],
                canonical_name=r['canonical_name'],
                fact_type=r['fact_type'],
                base_date=datetime.fromisoformat(r['base_date']),
                position_absolute=r['position_absolute'],
                confidence=r['confidence'],
                aliases=[a['alias_text'] for a in alias_rows],
            ))
        return facts

    def find_fact_by_alias(self, user_id: str, alias_text: str) -> Optional[TemporalFact]:
        """Resolve an alias to a fact."""
        row = self.conn.execute('''
            SELECT f.* FROM temporal_facts f
            JOIN fact_aliases a ON f.fact_id = a.fact_id
            WHERE f.user_id = ? AND LOWER(a.alias_text) = LOWER(?)
        ''', (user_id, alias_text)).fetchone()

        if row is None:
            return None
        r = dict(row)
        return TemporalFact(
            fact_id=r['fact_id'], user_id=r['user_id'],
            canonical_name=r['canonical_name'], fact_type=r['fact_type'],
            base_date=datetime.fromisoformat(r['base_date']),
            position_absolute=r['position_absolute'], confidence=r['confidence'],
        )

    def close(self) -> None:
        self.conn.close()


# ============================================
# TEMPORAL DISTANCE LAYER - O(1) duration math
# ============================================

class TemporalDistanceLayer:
    """Maintains NOW position. All durations are just subtraction."""

    def __init__(self):
        self.now_position: Optional[int] = None

    def update_now(self, current_date: Optional[datetime] = None) -> None:
        """Call at the start of each conversation turn."""
        dt = current_date or datetime.now()
        self.now_position = (dt - EPOCH).days
        logger.info("NOW: %s (position %d)", dt.strftime('%Y-%m-%d'), self.now_position)

    def days_since(self, fact: TemporalFact) -> int:
        """How many days ago did this fact occur?"""
        if self.now_position is None:
            raise ValueError("Call update_now() first")
        return self.now_position - fact.position_absolute

    def date_to_position(self, dt: datetime) -> int:
        return (dt - EPOCH).days


# ============================================
# FACT EXTRACTION - AI + regex fallback
# ============================================

_DATE_FORMATS = ['%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y']


def _clean_ordinal(s: str) -> str:
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s)


def _parse_date(s: str) -> datetime:
    cleaned = _clean_ordinal(s.strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(cleaned)  # last resort, may raise


class TemporalFactExtractor:
    """Extracts temporal facts from user messages.

    Tries AI extraction first (pipe-separated format), falls back to regex.
    """

    def __init__(self, model, tokenizer, db: TemporalDatabase, temporal: TemporalDistanceLayer):
        self.model = model
        self.tokenizer = tokenizer
        self.db = db
        self.temporal = temporal

    def extract_and_store(self, user_id: str, message: str) -> List[str]:
        """Extract facts from message, store new ones. Returns new fact_ids."""
        # Try AI, fall back to regex
        extracted = self._ai_extract(message)
        if not extracted:
            extracted = self._regex_extract(message)
        if not extracted:
            return []

        fact_ids: List[str] = []
        for ef in extracted:
            if self.db.find_fact_by_alias(user_id, ef.event_name):
                logger.info("Already known: %s", ef.event_name)
                continue

            fact_id = self.db.add_fact(
                user_id=user_id,
                canonical_name=ef.event_name,
                fact_type=ef.fact_type,
                base_date=ef.event_date,
                aliases=[ef.event_name.replace('_', ' ')] + ef.aliases,
            )
            fact_ids.append(fact_id)

            days = self.temporal.now_position - self.temporal.date_to_position(ef.event_date)
            logger.info("Learned: %s (%d days ago)", ef.event_name, days)

        return fact_ids

    def _ai_extract(self, message: str) -> List[ExtractedFact]:
        """Use the LLM to extract temporal facts."""
        prompt = (
            f"[INST] Extract temporal facts from this message. "
            f"Find any dates, events, or milestones mentioned.\n\n"
            f"Current date: {datetime.now().strftime('%B %d, %Y')}\n\n"
            f'Message: "{message}"\n\n'
            f"Format: event_name | YYYY-MM-DD | alias1, alias2\n"
            f'If none found, say "none"\n[/INST]'
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=150,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("[/INST]")[-1].strip()
        logger.debug("Extraction: %s", answer)

        if "none" in answer.lower():
            return []

        results: List[ExtractedFact] = []
        for line in answer.strip().split('\n'):
            if '|' not in line:
                continue
            try:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 2:
                    continue
                name = parts[0].lower().replace(' ', '_')
                date = datetime.strptime(parts[1], '%Y-%m-%d')
                aliases = [a.strip() for a in parts[2].split(',')] if len(parts) > 2 else []
                results.append(ExtractedFact(event_name=name, event_date=date, aliases=aliases))
            except (ValueError, IndexError) as e:
                logger.warning("Parse failed: %s - %s", line, e)
        return results

    def _regex_extract(self, message: str) -> List[ExtractedFact]:
        """Fallback: extract using regex patterns."""
        results: List[ExtractedFact] = []

        # "got sober on [date]"
        m = re.search(
            r'(?:got sober|became sober|started sobriety|been sober).*?(?:on|since)\s+'
            r'(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})', message, re.IGNORECASE
        )
        if m:
            try:
                results.append(ExtractedFact(
                    event_name="sobriety_start", event_date=_parse_date(m.group(1)),
                    aliases=["sober", "sobriety", "clean"], fact_type="milestone",
                ))
            except ValueError:
                pass

        # "started [activity] on [date]"
        m = re.search(
            r'started\s+(?:my\s+)?(\w+(?:\s+\w+)?)\s+(?:on|in)\s+'
            r'(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})', message, re.IGNORECASE
        )
        if m:
            try:
                activity = m.group(1).lower().replace(' ', '_')
                results.append(ExtractedFact(
                    event_name=f"{activity}_start", event_date=_parse_date(m.group(2)),
                    aliases=[activity, activity.replace('_', ' ')],
                ))
            except ValueError:
                pass

        # "moved to [place] on [date]"
        m = re.search(
            r'moved\s+to\s+(.+?)\s+(?:on|in)\s+'
            r'(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})', message, re.IGNORECASE
        )
        if m:
            try:
                place = m.group(1).strip().lower().replace(' ', '_')
                results.append(ExtractedFact(
                    event_name=f"moved_{place}", event_date=_parse_date(m.group(2)),
                    aliases=["moved", "move", place],
                ))
            except ValueError:
                pass

        return results


# ============================================
# CHAT - Conversational interface (Method A)
# ============================================

class ChronoceptionChat:
    """Multi-turn chat with temporal awareness via context injection.

    Method A: Pre-compute all temporal values, inject into prompt.
    Method B: Pass use_intervention=True to also steer logits (see intervention.py).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        db_path: str = "/data/temporal_facts.db",
    ):
        self.model_name = model_name
        self.db_path = db_path
        self.model = None
        self.tokenizer = None
        self.db: Optional[TemporalDatabase] = None
        self.temporal: Optional[TemporalDistanceLayer] = None
        self._extractor: Optional[TemporalFactExtractor] = None

    def initialize(self) -> None:
        """Load model and set up components. Call once before chat()."""
        logger.info("Loading model: %s", self.model_name)

        self.db = TemporalDatabase(self.db_path)
        self.temporal = TemporalDistanceLayer()
        self.temporal.update_now()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._extractor = TemporalFactExtractor(
            self.model, self.tokenizer, self.db, self.temporal,
        )
        logger.info("Ready.")

    def chat(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        use_intervention: bool = False,
    ) -> ChatResult:
        """Process one conversation turn.

        Args:
            user_id: Who is talking.
            messages: [{"role": "user"/"assistant", "content": "..."}]
            use_intervention: Also apply logits steering (Method B).
        """
        self.temporal.update_now()

        # Learn facts from latest user message
        last_msg = [m for m in messages if m['role'] == 'user'][-1]['content']
        new_ids = self._extractor.extract_and_store(user_id, last_msg)
        logger.info("Learned %d new facts", len(new_ids))

        # Load all facts, build context
        all_facts = self.db.get_facts_for_user(user_id)
        context = self._build_context(all_facts)

        # Build prompt
        convo = ""
        for msg in messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            convo += f"{role}: {msg['content']}\n"

        if context:
            prompt = f"[INST] {context}\n\n{convo}Assistant: [/INST]"
        else:
            prompt = f"[INST] {convo}Assistant: [/INST]"

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=200, do_sample=True, temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Method B: logits steering
        if use_intervention and all_facts:
            from intervention import TemporalLogitsProcessor
            processor = TemporalLogitsProcessor(
                tokenizer=self.tokenizer,
                facts=all_facts,
                temporal_layer=self.temporal,
            )
            gen_kwargs["logits_processor"] = [processor]
            logger.info("Logits steering ON (%d facts)", len(all_facts))

        outputs = self.model.generate(**inputs, **gen_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("[/INST]")[-1].strip()

        return ChatResult(
            response=answer,
            learned_facts=len(new_ids),
            total_facts=len(all_facts),
            temporal_context=context,
            intervention_active=use_intervention and bool(all_facts),
        )

    def _build_context(self, facts: List[TemporalFact]) -> str:
        """Pre-compute all durations. The AI uses these numbers, not its own math."""
        if not facts:
            return ""
        lines = ["TEMPORAL FACTS (pre-calculated, use these exact numbers):"]
        for fact in facts:
            days = self.temporal.days_since(fact)
            label = fact.canonical_name.replace('_', ' ')
            lines.append(f"- {label}: {days} days ago (since {fact.base_date.strftime('%B %d, %Y')})")
        return "\n".join(lines)

    def close(self) -> None:
        if self.db:
            self.db.close()
