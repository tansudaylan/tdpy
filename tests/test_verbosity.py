import io
import logging
import sys
from types import SimpleNamespace

import tdpy
import tdpy.verbosity as verbosity


def test_resident_memory_reporting_uses_current_process_and_respects_verbosity(capsys, monkeypatch):
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=2**20),
        memory_percent=lambda: 12.5,
    )
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(Process=lambda: process))

    assert tdpy.retr_memoresi() == (2**20, 12.5)

    monkeypatch.setenv('TDPY_VERBOSITY', '1')
    assert tdpy.show_memo_simp() is None
    assert capsys.readouterr().out == 'Resident memory: 1 MB, 12.5%\n'

    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    assert tdpy.show_memo_simp() is None
    assert capsys.readouterr().out == ''


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


def test_console_logging_and_context_managed_progress_follow_verbosity(capsys, monkeypatch):
    handler = verbosity.VerbosityStreamHandler()
    record = logging.makeLogRecord({'msg': 'pipeline message', 'levelno': logging.INFO})

    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    handler.emit(record)
    with verbosity.tqdm(total=1) as progress:
        progress.update(1)
    assert capsys.readouterr().err == ''

    monkeypatch.setenv('TDPY_VERBOSITY', '1')
    handler.emit(record)
    assert capsys.readouterr().err == 'pipeline message\n'


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
