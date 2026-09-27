import pytest
import tdpy

from tdpy.cli import parse_plot_arguments


def test_parse_plot_arguments_defaults_to_png():
    assert parse_plot_arguments(arguments=[]).typefileplot == "png"


def test_parse_plot_arguments_accepts_pdf():
    assert parse_plot_arguments(arguments=["--typefileplot", "pdf"]).typefileplot == "pdf"


def test_parse_plot_arguments_rejects_other_formats():
    with pytest.raises(SystemExit):
        parse_plot_arguments(arguments=["--typefileplot", "svg"])


def test_dispatch_cli_forwards_arguments():
    calls = []

    def run(*arguments):
        calls.append(arguments)
        return 'complete'

    result = tdpy.dispatch_cli({'run': run}, ['run', 'first', 'second'])

    assert result == 'complete'
    assert calls == [('first', 'second')]


def test_dispatch_cli_can_ignore_trailing_arguments():
    calls = []

    def run():
        calls.append(True)

    tdpy.dispatch_cli({'run': run}, ['run', 'ignored'], forward_arguments=False)

    assert calls == [True]


@pytest.mark.parametrize('arguments', [[], ['missing'], ['_private']])
def test_dispatch_cli_rejects_invalid_commands(arguments):
    with pytest.raises(SystemExit):
        tdpy.dispatch_cli({'_private': lambda: None}, arguments)