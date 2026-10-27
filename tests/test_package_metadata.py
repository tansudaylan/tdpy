import pathlib
import tomllib
from email.parser import Parser
import os
import shutil
import subprocess
import sys
import zipfile


def test_unconditional_imports_are_declared_as_dependencies():
    path = pathlib.Path(__file__).parents[1] / 'pyproject.toml'
    dependencies = set(tomllib.loads(path.read_text(encoding='utf-8'))['project']['dependencies'])

    assert {'Pillow', 'tesswcs', 'tqdm'} <= dependencies


def test_built_wheel_preserves_declared_python_floor_and_dependencies(tmp_path):
    repository = pathlib.Path(__file__).parents[1]
    stage = tmp_path / 'source'
    stage.mkdir()
    for name in ('setup.py', 'pyproject.toml', 'README.md'):
        shutil.copyfile(repository / name, stage / name)
    for source in repository.glob('LICENSE*'):
        shutil.copyfile(source, stage / source.name)
    shutil.copytree(repository / 'tdpy', stage / 'tdpy', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    project = tomllib.loads((stage / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)

    completed = subprocess.run(
        [sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
         '--wheel-dir', str(tmp_path / 'wheel'), str(stage)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheel, = (tmp_path / 'wheel').glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        metadata_path, = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        metadata = Parser().parsestr(archive.read(metadata_path).decode('utf-8'))
        assert 'tdpy/numerics.py' in archive.namelist()
        assert 'tdpy/plotting.py' in archive.namelist()
    assert metadata['Requires-Python'] == project['requires-python']
    assert metadata['Version'] == project['version']
    assert metadata['Summary'] == project['description']
    assert set(project['dependencies']).issubset(metadata.get_all('Requires-Dist'))
    assert project['optional-dependencies']['healpix'] == ['healpy']
    assert 'healpy' not in project['dependencies']
    assert 'healpix' in metadata.get_all('Provides-Extra')
    assert 'healpy; extra == "healpix"' in metadata.get_all('Requires-Dist')
    assert 'Repository, https://github.com/tansudaylan/tdpy' in metadata.get_all('Project-URL')
    installed = tmp_path / 'installed'
    completed = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(installed), str(wheel)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    environment['PYTHONPATH'] = str(installed)
    code = (
        "from pathlib import Path\nimport importlib.metadata as metadata\nimport numpy as np\n"
        "original = getattr(np, 'trapz', None)\nimport tdpy\n"
        f"assert Path(tdpy.__file__).resolve().is_relative_to(Path({str(installed)!r}).resolve())\n"
        f"assert metadata.metadata('tdpy')['Requires-Python'] == {project['requires-python']!r}\n"
        "assert getattr(np, 'trapz', None) is original\n"
        "np.testing.assert_allclose(tdpy.grid_quantiles(np.array([0., 1.]), np.ones(2), [.5]), [.5])\n"
        "from tdpy.plotting import preserve_text_color\nassert callable(preserve_text_color)\n"
    )
    completed = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                               env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert not list(repository.glob('*.egg-info'))