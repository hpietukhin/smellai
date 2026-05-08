"""Shared pytest configuration and fixtures.

- Default ignores for tests requiring external databases (td_V2.db)
- Concrete dataset samples for replacing flaky Hypothesis strategies
"""

# ── Default ignores ──────────────────────────────────────────────────────
# These are applied automatically so you can run `uv run pytest tests/`
# without manually adding --ignore flags.

collect_ignore_glob = [
    "test_td_v2.py",
    "test_td_v2_invariants.py",
    "test_dataset_loaders.py",
]
