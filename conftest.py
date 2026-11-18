"""Implementation switch (repository root, so pytest always loads it first).

    pytest                         -> tests run against workspace/ (your code)
    pytest --impl solutions        -> tests run against solutions/ (reference)
    COURSE_IMPL=solutions pytest   -> same as above

Tests import learner code only through tests.helpers.load().
"""

import os

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--impl",
        action="store",
        default=os.environ.get("COURSE_IMPL", "workspace"),
        choices=["workspace", "solutions"],
        help="which implementation to test (default: workspace)",
    )


def pytest_configure(config):
    os.environ["COURSE_IMPL"] = config.getoption("--impl")


def pytest_report_header(config):
    return f"implementation: {config.getoption('--impl')}"


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
