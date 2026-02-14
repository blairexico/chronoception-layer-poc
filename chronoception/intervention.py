"""Logits steering intervention for temporal accuracy during generation.

This module implements Method B from the architecture: hooking into the
generation process to detect temporal claims and steer logits toward
correct values. Uses HuggingFace's LogitsProcessor protocol.
"""

import logging
import re
from typing import List, Optional

import torch

from .config import InterventionConfig
from .detection import SmartTemporalDetector
from .distance import TemporalDistanceLayer
from .types import InterventionError, TemporalFact

logger = logging.getLogger(__name__)


class TemporalLogitsProcessor:
    """Steers generation toward correct temporal values.

    Integrates with model.generate() via the logits_processor parameter.
    When the model is about to generate a number in temporal context,
    this processor boosts the correct digits and suppresses wrong ones.

    Usage:
        processor = TemporalLogitsProcessor(
            tokenizer=tokenizer,
            facts=facts,
            temporal_layer=temporal,
            config=intervention_config,
        )
        outputs = model.generate(
            **inputs,
            logits_processor=[processor],
        )
    """

    def __init__(
        self,
        tokenizer,
        facts: List[TemporalFact],
        temporal_layer: TemporalDistanceLayer,
        config: Optional[InterventionConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.facts = facts
        self.temporal_layer = temporal_layer
        self.config = config or InterventionConfig(enabled=True)
        self.detector = SmartTemporalDetector(facts, window_size=self.config.detection_window)

        # Track tokens generated so far (reset per generate call)
        self._generated_tokens: List[str] = []
        self._active_number: str = ""
        self._target_number: Optional[str] = None
        self._digit_position: int = 0
        self._interventions: int = 0

        # Pre-compute digit token IDs
        self._digit_token_ids = {}
        for d in range(10):
            token_str = str(d)
            ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if ids:
                self._digit_token_ids[d] = ids[-1]

    @property
    def intervention_count(self) -> int:
        return self._interventions

    def reset(self) -> None:
        """Reset state for a new generation call."""
        self._generated_tokens = []
        self._active_number = ""
        self._target_number = None
        self._digit_position = 0
        self._interventions = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Called by model.generate() for each token position.

        Args:
            input_ids: Current sequence of token IDs [batch_size, seq_len]
            scores: Logits for the next token [batch_size, vocab_size]

        Returns:
            Modified scores tensor.
        """
        if not self.config.enabled:
            return scores

        # Decode the most recent token to track context
        if input_ids.shape[1] > 0:
            last_token_id = input_ids[0, -1].item()
            last_token = self.tokenizer.decode([last_token_id])
            self._generated_tokens.append(last_token)

        # Check if we're currently building a number
        top_token_id = scores[0].argmax().item()
        top_token = self.tokenizer.decode([top_token_id])

        if self._target_number is not None:
            # We're in the middle of steering a number
            return self._steer_digit(scores)

        # Check if we should start steering
        if (top_token.strip().isdigit()
                and self.detector.is_temporal_context(self._generated_tokens)
                and self.detector.is_number_position(self._generated_tokens)):

            fact = self.detector.find_relevant_fact(self._generated_tokens)
            if fact is not None:
                correct_days = self.temporal_layer.compute_fact_age(fact)
                self._target_number = str(correct_days)
                self._digit_position = 0
                self._active_number = ""
                logger.info(
                    "Intervention activated: steering toward %s for fact '%s'",
                    self._target_number, fact.canonical_name
                )
                return self._steer_digit(scores)

        return scores

    def _steer_digit(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Steer logits toward the correct digit at the current position."""
        if self._target_number is None or self._digit_position >= len(self._target_number):
            # Done with this number
            self._target_number = None
            self._digit_position = 0
            return scores

        target_char = self._target_number[self._digit_position]

        if not target_char.isdigit():
            self._target_number = None
            return scores

        target_digit = int(target_char)
        self._digit_position += 1

        # Boost the correct digit token
        if target_digit in self._digit_token_ids:
            correct_id = self._digit_token_ids[target_digit]
            scores[0, correct_id] += self.config.boost_correct

        # Suppress wrong digit tokens
        for d, token_id in self._digit_token_ids.items():
            if d != target_digit:
                scores[0, token_id] += self.config.suppress_wrong

        self._interventions += 1
        logger.debug("Steered digit %d/%d: %s", self._digit_position, len(self._target_number), target_char)

        # If we've output all digits, check if next token should NOT be a digit
        if self._digit_position >= len(self._target_number):
            # Suppress all digits to prevent extra digits being appended
            for d, token_id in self._digit_token_ids.items():
                scores[0, token_id] += self.config.suppress_wrong
            self._target_number = None

        return scores
