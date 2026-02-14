"""Chronoception - Temporal awareness for LLMs via generation-time intervention.

Two intervention methods:
    Method A (Context Injection): Pre-compute temporal values, inject into prompt.
    Method B (Logits Steering): Hook into generation, steer toward correct values.
"""

from .chat import ChronoceptionChat
from .config import ChronoceptionConfig, DatabaseConfig, InterventionConfig, ModelConfig
from .database import TemporalDatabase
from .detection import SmartTemporalDetector
from .distance import TemporalDistanceLayer
from .extraction import TemporalFactExtractor
from .intervention import TemporalLogitsProcessor
from .types import (
    ChatResult,
    DatabaseError,
    ExtractionError,
    ExtractedFact,
    InterventionError,
    TemporalContext,
    TemporalError,
    TemporalFact,
)

__all__ = [
    # Main entry point
    "ChronoceptionChat",
    # Config
    "ChronoceptionConfig",
    "ModelConfig",
    "DatabaseConfig",
    "InterventionConfig",
    # Core components
    "TemporalDatabase",
    "TemporalDistanceLayer",
    "TemporalFactExtractor",
    "SmartTemporalDetector",
    "TemporalLogitsProcessor",
    # Types
    "TemporalFact",
    "ExtractedFact",
    "TemporalContext",
    "ChatResult",
    # Exceptions
    "TemporalError",
    "DatabaseError",
    "ExtractionError",
    "InterventionError",
]
