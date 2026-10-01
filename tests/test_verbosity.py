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


def test_tqdm_cannot_enable_console_output_when_verbosity_is_off(monkeypatch):
    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    assert verbosity.tqdm(range(3), disable=False).disable


def test_verbosity_requires_a_positive_integer(monkeypatch):
    for value, expected in (
        ('1', True),
        (' 2 ', True),
        ('0', False),
        ('-1', False),
        ('false', False),
        ('', False),
    ):
        monkeypatch.setenv('TDPY_VERBOSITY', value)
        assert verbosity.retr_boolverb() is expected
