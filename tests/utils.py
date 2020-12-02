from pathlib import Path

import pytest


@pytest.fixture
def tests_path():
    return Path(__file__).parent
