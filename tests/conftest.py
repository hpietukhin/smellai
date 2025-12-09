"""Configure test environment imports and logging."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up consistent logging for all tests.

    This fixture runs automatically before all tests and configures
    logging using the centralized logging_config module.
    """
    from smellai.logging_config import setup_logging

    # Set up logging for tests - pytest.ini will control output
    setup_logging()
    yield
