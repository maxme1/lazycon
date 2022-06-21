import os
from pathlib import Path

import pytest


@pytest.fixture
def tests_path():
    return Path(__file__).parent


@pytest.fixture
def inside_tests(tests_path):
    wd = os.getcwd()
    try:
        os.chdir(tests_path)
        yield
    finally:
        os.chdir(wd)
