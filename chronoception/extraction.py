"""Extract temporal facts from natural language user messages."""

import logging
import re
from datetime import datetime
from typing import List

import torch

from .database import TemporalDatabase
from .distance import TemporalDistanceLayer
from .types import ExtractionError, ExtractedFact

logger = logging.getLogger(__name__)

# Regex patterns for manual fallback extraction
_DATE_PATTERNS = [
    # "on January 15, 2024" / "on January 15th, 2024"
    (r'on\s+(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})', None),
    # "since September 4, 2025"
    (r'since\s+(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})', None),
    # ISO-style "2024-01-15"
    (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
]

_DATE_FORMATS = ['%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y']


def _clean_ordinal(date_str: str) -> str:
    """Remove ordinal suffixes from date strings."""
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)


def _parse_date(date_str: str) -> datetime:
    """Try multiple formats to parse a date string."""
    cleaned = _clean_ordinal(date_str.strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    # Try ISO format
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass
    raise ValueError(f"Could not parse date: {date_str}")


class TemporalFactExtractor:
    """Extracts temporal facts from user messages using AI with regex fallback."""

    def __init__(
        self,
        model,
        tokenizer,
        db: TemporalDatabase,
        temporal_layer: TemporalDistanceLayer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db = db
        self.temporal_layer = temporal_layer

    def extract_and_store(self, user_id: str, message: str) -> List[str]:
        """
        Extract temporal facts from a user message and store them.

        Returns list of newly created fact_ids.
        """
        # Try AI extraction first, fall back to regex
        try:
            extracted = self._ai_extract(message)
        except ExtractionError:
            logger.warning("AI extraction failed, trying regex fallback")
            extracted = []

        if not extracted:
            extracted = self._regex_extract(message)

        if not extracted:
            return []

        # Store each extracted fact
        fact_ids: List[str] = []
        for ef in extracted:
            # Skip if fact already exists
            existing = self.db.find_fact_by_alias(user_id, ef.event_name)
            if existing:
                logger.info("Fact already exists: %s", ef.event_name)
                continue

            fact_id = self.db.add_fact(
                user_id=user_id,
                canonical_name=ef.event_name,
                fact_type=ef.fact_type,
                base_date=ef.event_date,
                aliases=[ef.event_name.replace('_', ' ')] + ef.aliases,
                confidence=ef.confidence,
            )
            fact_ids.append(fact_id)

            days_ago = self.temporal_layer.compute_duration(
                self.temporal_layer.date_to_position(ef.event_date)
            )
            logger.info("Stored: %s (%d days ago)", ef.event_name, days_ago)

        return fact_ids

    def _ai_extract(self, message: str) -> List[ExtractedFact]:
        """Use the LLM to extract temporal facts from a message."""
        extraction_prompt = (
            f"[INST] Extract temporal facts from this message. "
            f"Find any dates, events, or milestones mentioned.\n\n"
            f"Current date: {datetime.now().strftime('%B %d, %Y')}\n\n"
            f'Message: "{message}"\n\n'
            f"For each temporal fact found, provide:\n"
            f"- event_name: brief name (e.g., \"blog_start\", \"moved_cities\")\n"
            f"- event_date: YYYY-MM-DD format\n"
            f"- aliases: comma-separated alternative names\n\n"
            f"Format as simple text, one fact per line:\n"
            f"event_name | YYYY-MM-DD | alias1, alias2\n\n"
            f'If no temporal facts found, respond with "none"\n'
            f"[/INST]"
        )

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

        logger.debug("Extraction result: %s", answer)

        if "none" in answer.lower():
            return []

        return self._parse_pipe_format(answer)

    def _parse_pipe_format(self, text: str) -> List[ExtractedFact]:
        """Parse the pipe-separated extraction format."""
        results: List[ExtractedFact] = []

        for line in text.strip().split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue
            try:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 2:
                    continue

                event_name = parts[0].lower().replace(' ', '_')
                event_date = datetime.strptime(parts[1], '%Y-%m-%d')
                aliases = [a.strip() for a in parts[2].split(',')] if len(parts) > 2 else []

                results.append(ExtractedFact(
                    event_name=event_name,
                    event_date=event_date,
                    aliases=aliases,
                ))
            except (ValueError, IndexError) as e:
                logger.warning("Failed to parse extraction line: %s - %s", line, e)
                continue

        return results

    def _regex_extract(self, message: str) -> List[ExtractedFact]:
        """Fallback: extract temporal facts using regex patterns."""
        results: List[ExtractedFact] = []

        # Pattern: "got sober on [date]" / "became sober on [date]"
        sober_pattern = r'(?:got sober|became sober|started sobriety|been sober).*?(?:on|since)\s+(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})'
        match = re.search(sober_pattern, message, re.IGNORECASE)
        if match:
            try:
                event_date = _parse_date(match.group(1))
                results.append(ExtractedFact(
                    event_name="sobriety_start",
                    event_date=event_date,
                    aliases=["sober", "sobriety", "clean"],
                    fact_type="milestone",
                ))
                logger.info("Regex extraction: sobriety start %s", event_date.date())
            except ValueError:
                pass

        # Pattern: "started [activity] on [date]"
        started_pattern = r'started\s+(?:my\s+)?(\w+(?:\s+\w+)?)\s+(?:on|in)\s+(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})'
        match = re.search(started_pattern, message, re.IGNORECASE)
        if match:
            try:
                activity = match.group(1).lower().replace(' ', '_')
                event_date = _parse_date(match.group(2))
                results.append(ExtractedFact(
                    event_name=f"{activity}_start",
                    event_date=event_date,
                    aliases=[activity, activity.replace('_', ' ')],
                ))
                logger.info("Regex extraction: %s start %s", activity, event_date.date())
            except ValueError:
                pass

        # Pattern: "moved to [place] on [date]"
        moved_pattern = r'moved\s+to\s+(.+?)\s+(?:on|in)\s+(\w+\s+\d+(?:st|nd|rd|th)?,?\s+\d{4})'
        match = re.search(moved_pattern, message, re.IGNORECASE)
        if match:
            try:
                place = match.group(1).strip().lower().replace(' ', '_')
                event_date = _parse_date(match.group(2))
                results.append(ExtractedFact(
                    event_name=f"moved_{place}",
                    event_date=event_date,
                    aliases=["moved", "move", place],
                ))
                logger.info("Regex extraction: moved to %s on %s", place, event_date.date())
            except ValueError:
                pass

        return results
