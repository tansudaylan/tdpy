import pathlib
import tomllib


def test_unconditional_imports_are_declared_as_dependencies():
    path = pathlib.Path(__file__).parents[1] / 'pyproject.toml'
    dependencies = set(tomllib.loads(path.read_text(encoding='utf-8'))['project']['dependencies'])

    assert {'tesswcs', 'tqdm'} <= dependencies