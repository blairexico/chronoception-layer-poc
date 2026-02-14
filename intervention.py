"""
Logits steering for temporal accuracy (Method B).

Hooks into model.generate() to detect temporal claims mid-generation
and steer logits toward the correct numbers.

Usage:
    from intervention import TemporalLogitsProcessor

    processor = TemporalLogitsProcessor(
        tokenizer=tokenizer,
        facts=all_facts,
        temporal_layer=temporal,
    )
    outputs = model.generate(**inputs, logits_processor=[processor])

Or just pass use_intervention=True to ChronoceptionChat.chat().
"""

import logging
import re
from typing import Dict, List, Optional

import torch

from chronoception import TemporalDistanceLayer, TemporalFact

logger = logging.getLogger(__name__)


# ============================================
# DETECTION - When to intervene
# ============================================

# Patterns suggesting a temporal claim is being generated
_TEMPORAL_PATTERNS = [
    r'\b\d+\s+days?\b', r'\b\d+\s+weeks?\b',
    r'\b\d+\s+months?\b', r'\b\d+\s+years?\b',
    r'\byou(?:\'ve| have) been\b', r'\bit(?:\'s| has) been\b',
    r'\bfor\s+\d+\b', r'\bsince\b', r'\bago\b',
]

# Patterns indicating the next token should be a number
_NUMBER_POSITION_PATTERNS = [
    r'(?:been|for|about|approximately|exactly|nearly|over|around)\s*$',
    r'(?:is|are|was|were)\s*$',
    r'(?:you are|you\'re|it\'s been|it has been)\s*$',
]


class SmartTemporalDetector:
    """Detects when generated text involves temporal claims about known facts."""

    def __init__(self, facts: List[TemporalFact], window_size: int = 10):
        self.facts = facts
        self.window_size = window_size
        self._keywords = set()
        for fact in facts:
            self._keywords.add(fact.canonical_name.lower())
            self._keywords.add(fact.canonical_name.replace('_', ' ').lower())
            for alias in fact.aliases:
                self._keywords.add(alias.lower())

    def is_temporal_context(self, recent_tokens: List[str]) -> bool:
        text = "".join(recent_tokens[-self.window_size:]).lower()
        return any(re.search(p, text) for p in _TEMPORAL_PATTERNS)

    def is_number_position(self, recent_tokens: List[str]) -> bool:
        text = "".join(recent_tokens[-self.window_size:]).lower()
        return any(re.search(p, text) for p in _NUMBER_POSITION_PATTERNS)

    def find_relevant_fact(self, recent_tokens: List[str]) -> Optional[TemporalFact]:
        """Which fact is being talked about?"""
        text = "".join(recent_tokens[-self.window_size * 2:]).lower()
        best, best_score = None, 0

        for fact in self.facts:
            score = 0
            if fact.canonical_name.replace('_', ' ').lower() in text:
                score += 2
            for alias in fact.aliases:
                if alias.lower() in text:
                    score += 1
            if score > best_score:
                best, best_score = fact, score

        return best if best_score > 0 else None


# ============================================
# LOGITS PROCESSOR - The actual steering
# ============================================

class TemporalLogitsProcessor:
    """Steers generation toward correct temporal values.

    When the model is about to output a number in temporal context
    (e.g., "you've been blogging for ___  days"), this processor:
    - Boosts logits for the correct digit tokens
    - Suppresses logits for wrong digit tokens

    Implements the HuggingFace LogitsProcessor protocol.
    """

    def __init__(
        self,
        tokenizer,
        facts: List[TemporalFact],
        temporal_layer: TemporalDistanceLayer,
        boost: float = 5.0,
        suppress: float = -10.0,
        detection_window: int = 10,
    ):
        self.tokenizer = tokenizer
        self.temporal = temporal_layer
        self.detector = SmartTemporalDetector(facts, window_size=detection_window)
        self.boost = boost
        self.suppress = suppress

        self._generated_tokens: List[str] = []
        self._target_number: Optional[str] = None
        self._digit_pos: int = 0
        self._interventions: int = 0

        # Map digit -> token ID
        self._digit_ids: Dict[int, int] = {}
        for d in range(10):
            ids = tokenizer.encode(str(d), add_special_tokens=False)
            if ids:
                self._digit_ids[d] = ids[-1]

    @property
    def intervention_count(self) -> int:
        return self._interventions

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Called by model.generate() for each token."""
        # Track what's been generated
        if input_ids.shape[1] > 0:
            last_id = input_ids[0, -1].item()
            self._generated_tokens.append(self.tokenizer.decode([last_id]))

        # Currently steering a number? Keep going.
        if self._target_number is not None:
            return self._steer_digit(scores)

        # Check if we should START steering
        top_token = self.tokenizer.decode([scores[0].argmax().item()])

        if (top_token.strip().isdigit()
                and self.detector.is_temporal_context(self._generated_tokens)
                and self.detector.is_number_position(self._generated_tokens)):

            fact = self.detector.find_relevant_fact(self._generated_tokens)
            if fact:
                correct_days = self.temporal.days_since(fact)
                self._target_number = str(correct_days)
                self._digit_pos = 0
                logger.info("Steering toward %s for '%s'", self._target_number, fact.canonical_name)
                return self._steer_digit(scores)

        return scores

    def _steer_digit(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Boost correct digit, suppress wrong digits."""
        if self._target_number is None or self._digit_pos >= len(self._target_number):
            self._target_number = None
            return scores

        target = int(self._target_number[self._digit_pos])
        self._digit_pos += 1

        # Boost correct, suppress wrong
        if target in self._digit_ids:
            scores[0, self._digit_ids[target]] += self.boost
        for d, tid in self._digit_ids.items():
            if d != target:
                scores[0, tid] += self.suppress

        self._interventions += 1

        # After last digit, suppress all digits to prevent "761" → "7619"
        if self._digit_pos >= len(self._target_number):
            for tid in self._digit_ids.values():
                scores[0, tid] += self.suppress
            self._target_number = None

        return scores
