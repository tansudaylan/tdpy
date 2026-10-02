import pytest

from tdpy.paths import (
    RepositoryPaths,
    get_data_path,
    get_repository_path,
    get_visuals_path,
    make_directory,
    make_symlink,
)


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


def test_repository_paths_supports_package_specific_configuration(monkeypatch, tmp_path):
    """One shared resolver preserves a consumer package's environment contract."""

    monkeypatch.setenv("EXAMPLE_PATH", str(tmp_path))
    paths = RepositoryPaths("EXAMPLE_PATH")

    assert paths.get_repository_path() == tmp_path
    assert paths.get_data_path() == tmp_path / "data"
    assert paths.get_visuals_path() == tmp_path / "visuals"


def test_narrated_filesystem_helpers_replace_symlink(tmp_path, capsys):
    """Shared filesystem helpers create directories and replace stale links."""

    directory = tmp_path / "output directory"
    first_target = directory / "first.txt"
    second_target = directory / "second.txt"
    link_path = directory / "current.txt"

    make_directory(directory)
    first_target.write_text("first", encoding="utf-8")
    second_target.write_text("second", encoding="utf-8")
    make_symlink(first_target, link_path)
    make_symlink(second_target, link_path)

    assert link_path.read_text(encoding="utf-8") == "second"
    assert capsys.readouterr().out.splitlines() == [
        f"Writing to {directory}...",
        f"Writing to {link_path}...",
        f"Writing to {link_path}...",
    ]


def test_make_symlink_preserves_regular_file_unless_overwrite_is_explicit(tmp_path):
    target = tmp_path / "target.txt"
    destination = tmp_path / "destination.txt"
    target.write_text("target", encoding="utf-8")
    destination.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        make_symlink(target, destination)
    assert destination.read_text(encoding="utf-8") == "keep"

    make_symlink(target, destination, overwrite=True)
    assert destination.is_symlink()
    assert destination.read_text(encoding="utf-8") == "target"