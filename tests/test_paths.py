import os

import tdpy


def test_retr_pathbase_normalizes_and_requires_env(monkeypatch, tmp_path):
    monkeypatch.delenv('TDPY_DATA_PATH', raising=False)

    try:
        tdpy.retr_pathbase('tdpy')
    except EnvironmentError as exc:
        assert 'TDPY_DATA_PATH' in str(exc)
    else:
        raise AssertionError('retr_pathbase() should fail when the environment variable is unset.')

    raw_path = tmp_path / 'nested' / '..' / 'science'
    monkeypatch.setenv('TDPY_DATA_PATH', str(raw_path))

    pathbase = tdpy.retr_pathbase('tdpy')

    assert pathbase.endswith(os.sep)
    assert os.path.isabs(pathbase)
    assert os.path.normpath(pathbase) == os.path.normpath(str(tmp_path / 'science'))


def test_retr_pathenv_normalizes_generic_env(monkeypatch, tmp_path):
    monkeypatch.setenv('LION_PATH', str(tmp_path / 'code' / '..' / 'lion-src'))

    pathbase = tdpy.retr_pathenv('LION_PATH')

    assert pathbase.endswith(os.sep)
    assert os.path.isabs(pathbase)
    assert os.path.normpath(pathbase) == os.path.normpath(str(tmp_path / 'lion-src'))


def test_retr_path_creates_visual_and_data_directories(monkeypatch, tmp_path):
    monkeypatch.setenv('TDPY_DATA_PATH', str(tmp_path))

    pathvisu, pathdata = tdpy.retr_path('tdpy', pathextndata='demo', pathextnimag='demo', rtag='case')

    assert pathdata.endswith(os.sep)
    assert pathvisu.endswith(os.sep)
    assert os.path.isdir(pathdata)
    assert os.path.isdir(pathvisu)
    assert os.path.normpath(pathdata).endswith(os.path.normpath('demo/data'))
    assert os.path.normpath(pathvisu).endswith(os.path.normpath('demo/visuals/case'))