"""Conversational chat with temporal awareness.

Brings together all components: database, distance layer, fact extraction,
and optional logits steering intervention.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ChronoceptionConfig
from .database import TemporalDatabase
from .distance import TemporalDistanceLayer
from .extraction import TemporalFactExtractor
from .intervention import TemporalLogitsProcessor
from .types import ChatResult, TemporalContext, TemporalFact

logger = logging.getLogger(__name__)


class ChronoceptionChat:
    """Multi-turn conversational chat with temporal awareness.

    Supports two intervention methods:
        Method A (Context Injection): Pre-compute temporal values and inject
            them into the prompt. The model reads and uses them naturally.
        Method B (Logits Steering): Hook into generation and steer logits
            toward correct temporal values when the model makes claims.

    Both methods can be used simultaneously for maximum accuracy.
    """

    def __init__(self, config: Optional[ChronoceptionConfig] = None):
        self.config = config or ChronoceptionConfig()
        self._model = None
        self._tokenizer = None
        self._db: Optional[TemporalDatabase] = None
        self._temporal: Optional[TemporalDistanceLayer] = None
        self._extractor: Optional[TemporalFactExtractor] = None

    def initialize(self) -> None:
        """Load model and initialize all components.

        Call this once before using chat(). Separated from __init__
        so callers can control when the heavy model load happens.
        """
        logger.info("Initializing ChronoceptionChat...")

        # Database
        self._db = TemporalDatabase(self.config.database)

        # Temporal distance layer
        self._temporal = TemporalDistanceLayer()
        self._temporal.update_now()

        # Model
        logger.info("Loading model: %s", self.config.model.model_name)
        dtype = getattr(torch, self.config.model.torch_dtype, torch.float16)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            device_map=self.config.model.device_map,
            torch_dtype=dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Fact extractor
        self._extractor = TemporalFactExtractor(
            self._model, self._tokenizer, self._db, self._temporal,
        )

        logger.info("ChronoceptionChat initialized")

    @property
    def db(self) -> TemporalDatabase:
        assert self._db is not None, "Call initialize() first"
        return self._db

    @property
    def temporal(self) -> TemporalDistanceLayer:
        assert self._temporal is not None, "Call initialize() first"
        return self._temporal

    def chat(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        use_intervention: Optional[bool] = None,
    ) -> ChatResult:
        """Process a conversational turn with temporal awareness.

        Args:
            user_id: Unique user identifier.
            messages: Conversation history as [{"role": "user"/"assistant", "content": "..."}].
            use_intervention: Override config to enable/disable logits steering.
                Defaults to config.intervention.enabled.

        Returns:
            ChatResult with response, learned facts count, total facts count, and context.
        """
        assert self._model is not None, "Call initialize() first"
        assert self._extractor is not None

        if use_intervention is None:
            use_intervention = self.config.intervention.enabled

        # Update NOW each turn
        self._temporal.update_now()

        # Extract facts from the latest user message
        last_user_msg = [m for m in messages if m['role'] == 'user'][-1]['content']
        new_fact_ids = self._extractor.extract_and_store(user_id, last_user_msg)
        logger.info("Learned %d new facts from user message", len(new_fact_ids))

        # Load all known facts for this user
        all_facts = self._db.get_facts_for_user(user_id)

        # Build temporal context (Method A: context injection)
        context = self._build_context(all_facts)

        # Build the full prompt
        prompt = self._build_prompt(messages, context)

        # Generate response
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        generate_kwargs = {
            "max_new_tokens": self.config.model.max_response_tokens,
            "do_sample": self.config.model.do_sample,
            "temperature": self.config.model.temperature,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        # Method B: logits steering
        if use_intervention and all_facts:
            processor = TemporalLogitsProcessor(
                tokenizer=self._tokenizer,
                facts=all_facts,
                temporal_layer=self._temporal,
                config=self.config.intervention,
            )
            generate_kwargs["logits_processor"] = [processor]
            logger.info("Logits steering enabled with %d facts", len(all_facts))

        outputs = self._model.generate(**inputs, **generate_kwargs)

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("[/INST]")[-1].strip()

        intervention_active = use_intervention and bool(all_facts)

        return ChatResult(
            response=answer,
            learned_facts=len(new_fact_ids),
            total_facts=len(all_facts),
            temporal_context=context.render(),
            intervention_active=intervention_active,
        )

    def _build_context(self, facts: List[TemporalFact]) -> TemporalContext:
        """Build pre-computed temporal context from stored facts."""
        lines: List[str] = []
        now_pos = self._temporal.now_position

        if facts:
            lines.append("TEMPORAL FACTS (pre-calculated, use these exact numbers):")
            for fact in facts:
                days = self._temporal.compute_fact_age(fact)
                label = fact.canonical_name.replace('_', ' ')
                lines.append(f"- {label}: {days} days ago (since {fact.base_date.strftime('%B %d, %Y')})")

        return TemporalContext(facts=facts, now_position=now_pos, lines=lines)

    def _build_prompt(
        self,
        messages: List[Dict[str, str]],
        context: TemporalContext,
    ) -> str:
        """Build the full prompt with temporal context injected."""
        conversation = ""
        for msg in messages:
            if msg['role'] == 'user':
                conversation += f"User: {msg['content']}\n"
            else:
                conversation += f"Assistant: {msg['content']}\n"

        ctx = context.render()
        if ctx:
            return f"[INST] {ctx}\n\n{conversation}Assistant: [/INST]"
        return f"[INST] {conversation}Assistant: [/INST]"

    def close(self) -> None:
        """Clean up resources."""
        if self._db:
            self._db.close()
