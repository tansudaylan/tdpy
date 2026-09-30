import io

import tdpy.verbosity as verbosity


def test_print_is_silent_by_default_and_restored_by_the_environment(capsys, monkeypatch):
    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    verbosity.print('hidden')
    assert capsys.readouterr().out == ''
    monkeypatch.setenv('TDPY_VERBOSITY', '1')
    verbosity.print('shown')
    assert capsys.readouterr().out == 'shown\n'


def test_print_to_an_explicit_file_is_never_suppressed(monkeypatch):
    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    buffer = io.StringIO()
    verbosity.print('kept', file=buffer)
    assert buffer.getvalue() == 'kept\n'
    assert verbosity.tqdm(range(3)).disable
