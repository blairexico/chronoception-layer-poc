"""Temporal distance layer - manages NOW position and duration calculations."""

import logging
from datetime import datetime
from typing import Optional

from .types import EPOCH, TemporalFact

logger = logging.getLogger(__name__)


class TemporalDistanceLayer:
    """Maintains a sliding NOW position and computes O(1) temporal distances."""

    def __init__(self):
        self.now_position: Optional[int] = None
        self._current_date: Optional[datetime] = None

    @property
    def current_date(self) -> Optional[datetime]:
        return self._current_date

    def update_now(self, current_date: Optional[datetime] = None) -> None:
        """Update the NOW position. Call this at the start of each conversation turn."""
        if current_date is None:
            current_date = datetime.now()

        self._current_date = current_date
        self.now_position = (current_date - EPOCH).days
        logger.info("NOW updated: %s (position %d)", current_date.strftime('%Y-%m-%d'), self.now_position)

    def compute_duration(self, base_position_absolute: int) -> int:
        """Days elapsed since a fact's base date. O(1) subtraction."""
        if self.now_position is None:
            raise ValueError("NOW not initialized - call update_now() first")
        return self.now_position - base_position_absolute

    def compute_fact_age(self, fact: TemporalFact) -> int:
        """Days elapsed since a fact occurred."""
        return self.compute_duration(fact.position_absolute)

    def date_to_position(self, date: datetime) -> int:
        """Convert a datetime to an absolute position (days since epoch)."""
        return (date - EPOCH).days
