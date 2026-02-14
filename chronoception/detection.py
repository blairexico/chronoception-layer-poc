"""Smart temporal context detection during generation.

Determines when the model is generating text that involves temporal claims,
so that the intervention layer knows when to activate.
"""

import logging
import re
from typing import List, Optional

from .types import TemporalFact

logger = logging.getLogger(__name__)

# Patterns that suggest the model is making a temporal claim
_TEMPORAL_CLAIM_PATTERNS = [
    r'\b\d+\s+days?\b',
    r'\b\d+\s+weeks?\b',
    r'\b\d+\s+months?\b',
    r'\b\d+\s+years?\b',
    r'\byou(?:\'ve| have) been\b',
    r'\bit(?:\'s| has) been\b',
    r'\bfor\s+\d+\b',
    r'\bsince\b',
    r'\bago\b',
]

# Patterns that indicate a number is being generated in temporal context
_NUMBER_CONTEXT_PATTERNS = [
    r'(?:been|for|about|approximately|exactly|nearly|over|around)\s*$',
    r'(?:is|are|was|were)\s*$',
    r'(?:you are|you\'re|it\'s been|it has been)\s*$',
]


class SmartTemporalDetector:
    """Detects when generated text involves temporal claims about known facts.

    Used by the intervention layer to know when to activate logits steering.
    """

    def __init__(self, facts: List[TemporalFact], window_size: int = 10):
        self.facts = facts
        self.window_size = window_size
        # Build keyword sets from fact names and aliases
        self._keywords = set()
        for fact in facts:
            self._keywords.add(fact.canonical_name.lower())
            name_words = fact.canonical_name.replace('_', ' ').lower()
            self._keywords.add(name_words)
            for alias in fact.aliases:
                self._keywords.add(alias.lower())

    def is_temporal_context(self, recent_tokens: List[str]) -> bool:
        """Check if recent tokens suggest we're in a temporal claim context."""
        recent_text = "".join(recent_tokens[-self.window_size:]).lower()

        # Check if any temporal claim pattern matches
        for pattern in _TEMPORAL_CLAIM_PATTERNS:
            if re.search(pattern, recent_text):
                return True

        return False

    def is_number_position(self, recent_tokens: List[str]) -> bool:
        """Check if the next token position expects a number in temporal context."""
        recent_text = "".join(recent_tokens[-self.window_size:]).lower()

        for pattern in _NUMBER_CONTEXT_PATTERNS:
            if re.search(pattern, recent_text):
                return True

        return False

    def find_relevant_fact(self, recent_tokens: List[str]) -> Optional[TemporalFact]:
        """Find which temporal fact is being referenced in recent context."""
        recent_text = "".join(recent_tokens[-self.window_size * 2:]).lower()

        best_match: Optional[TemporalFact] = None
        best_score = 0

        for fact in self.facts:
            score = 0
            # Check canonical name
            name = fact.canonical_name.replace('_', ' ').lower()
            if name in recent_text:
                score += 2

            # Check aliases
            for alias in fact.aliases:
                if alias.lower() in recent_text:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = fact

        return best_match if best_score > 0 else None
