"""Configuration for Java Test Agent."""

from dataclasses import dataclass


@dataclass
class JavaTestAgentConfig:
    """Configuration keys for Java Test Agent."""

    MODEL_NAME = "model_name"


# Default configuration values
DEFAULT_CONFIG = {
    JavaTestAgentConfig.MODEL_NAME: "gpt-4o-mini",
}
