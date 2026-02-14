"""Configuration constants for the chronoception system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    device_map: str = "auto"
    torch_dtype: str = "float16"
    max_extraction_tokens: int = 150
    max_response_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for the temporal database."""
    db_path: str = "/data/temporal_facts.db"


@dataclass
class InterventionConfig:
    """Configuration for logits steering intervention."""
    enabled: bool = False
    boost_correct: float = 5.0
    suppress_wrong: float = -10.0
    hook_layer: int = -1  # last layer by default
    detection_window: int = 10  # tokens of context for detection


@dataclass
class ChronoceptionConfig:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    default_user_id: str = "default_user"
