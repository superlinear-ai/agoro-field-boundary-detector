"""Pytest configuration."""

import os
from typing import Any, Dict

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> Dict[str, Any]:
    """VCR configuration."""
    # Usage: add `@pytest.mark.vcr()` to record and replay all HTTP requests.
    vcr_config = {
        # Record mode:
        # - Use "none" in CI: only replay cassettes or throw an error if they are missing.
        # - Use "once" locally: record if there is no cassette, and if there is one replay it.
        "record_mode": "none" if os.environ.get("CI") else "once",
        # Remove sensitive information [1]. To make these secrets available when running tests, run
        # $ env SECRET=swordfish invoke test
        # and access them in Python with `os.environ.get("SECRET")`.
        # [1] https://vcrpy.readthedocs.io/en/latest/advanced.html#filter-sensitive-data-from-the-request
        "filter_headers": ["authorization"],
        "filter_query_parameters": ["api_key", "password"],
    }
    return vcr_config
