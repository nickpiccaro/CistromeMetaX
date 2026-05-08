"""Shared pytest configuration for CistromeMetaX tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run tests that call a live LLM (requires API key)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: mark test as requiring a live LLM call")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-llm"):
        skip_llm = pytest.mark.skip(reason="need --run-llm option to run")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_llm)
