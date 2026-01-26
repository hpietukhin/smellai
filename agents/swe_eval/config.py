"""Configuration for SWE evaluation agent."""

from enum import Enum


class SWEEvalAgentConfig(str, Enum):
    """Configuration keys for SWE evaluation agent."""

    MODEL_NAME = "model_name"
    WORKSPACE_DIR = "workspace_dir"
    MAX_RETRIES = "max_retries"
    COMPILE_TIMEOUT = "compile_timeout"
    TEST_TIMEOUT = "test_timeout"


DEFAULT_CONFIG = {
    SWEEvalAgentConfig.MODEL_NAME: "claude-sonnet-4-5-20250929",
    SWEEvalAgentConfig.WORKSPACE_DIR: "/tmp/swe-eval-workspace",
    SWEEvalAgentConfig.MAX_RETRIES: 3,
    SWEEvalAgentConfig.COMPILE_TIMEOUT: 600,
    SWEEvalAgentConfig.TEST_TIMEOUT: 600,
}
