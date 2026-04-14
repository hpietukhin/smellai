"""Default configuration for mini-swe-agent ablation wrapper."""

DEFAULT_MINI_CONFIG: dict = {
    "agent": {
        "step_limit": 80,
        "cost_limit": 2.0,
    },
    "environment": {
        "timeout": 180,
        "env": {"PAGER": "cat", "TQDM_DISABLE": "1"},
    },
    "model": {
        "model_kwargs": {"temperature": 0.0, "drop_params": True},
    },
}
