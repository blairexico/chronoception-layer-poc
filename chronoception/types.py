"""Core types and exceptions for the chronoception system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ============================================
# Exceptions
# ============================================

class TemporalError(Exception):
    """Base exception for temporal operations."""
    pass


class DatabaseError(TemporalError):
    """Raised when a database operation fails."""
    pass


class ExtractionError(TemporalError):
    """Raised when fact extraction fails."""
    pass


class InterventionError(TemporalError):
    """Raised when logits intervention fails."""
    pass


# ============================================
# Data classes
# ============================================

EPOCH = datetime(1970, 1, 1)


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
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    aliases: List[str] = field(default_factory=list)

    @property
    def days_since_epoch(self) -> int:
        return (self.base_date - EPOCH).days

    def duration_from(self, now_position: int) -> int:
        """Days elapsed from this fact's date to given position."""
        return now_position - self.position_absolute


@dataclass
class ExtractedFact:
    """A fact extracted from user text, before storage."""
    event_name: str
    event_date: datetime
    aliases: List[str] = field(default_factory=list)
    fact_type: str = "point_in_time"
    confidence: float = 1.0


@dataclass
class TemporalContext:
    """Pre-computed temporal context for injection into prompts."""
    facts: List[TemporalFact]
    now_position: int
    lines: List[str] = field(default_factory=list)

    def render(self) -> str:
        """Render context as a string for prompt injection."""
        if not self.lines:
            return ""
        return "\n".join(self.lines)


@dataclass
class ChatResult:
    """Result from a conversational chat turn."""
    response: str
    learned_facts: int
    total_facts: int
    temporal_context: str
    intervention_active: bool = False
