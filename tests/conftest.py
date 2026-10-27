import pytest


@pytest.fixture(autouse=True)
def enable_test_narration(monkeypatch):
    monkeypatch.setenv('TDPY_VERBOSITY', '1')
