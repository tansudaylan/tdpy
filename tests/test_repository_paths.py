import pytest

from tdpy.paths import get_data_path, get_repository_path, get_visuals_path


def test_repository_runtime_paths(monkeypatch, tmp_path):
    """TDPY_PATH owns repository-local data and visualization directories."""

    monkeypatch.setenv("TDPY_PATH", str(tmp_path))

    assert get_repository_path() == tmp_path
    assert get_data_path() == tmp_path / "data"
    assert get_visuals_path() == tmp_path / "visuals"


def test_repository_path_is_required(monkeypatch):
    """Missing repository configuration fails with an actionable message."""

    monkeypatch.delenv("TDPY_PATH", raising=False)

    with pytest.raises(EnvironmentError, match="TDPY_PATH"):
        get_repository_path()