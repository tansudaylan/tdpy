import pytest
import tdpy

from tdpy.cli import add_plot_arguments, dispatch_cli, parse_plot_arguments, run_plot_cli


def test_parse_plot_arguments_defaults_to_png():
    assert parse_plot_arguments(arguments=[]).typefileplot == "png"


def test_parse_plot_arguments_accepts_pdf():
    assert parse_plot_arguments(arguments=["--typefileplot", "pdf"]).typefileplot == "pdf"


def test_parse_plot_arguments_rejects_other_formats():
    with pytest.raises(SystemExit):
        parse_plot_arguments(arguments=["--typefileplot", "svg"])


def test_add_plot_arguments_composes_with_existing_options():
    import argparse

    parser = add_plot_arguments(argparse.ArgumentParser())
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args(["--typefileplot", "pdf", "--quick"])

    assert arguments.typefileplot == "pdf"
    assert arguments.quick is True


def test_run_plot_cli_forwards_output_format():
    calls = []

    def run(typefileplot):
        calls.append(typefileplot)

    assert run_plot_cli(run, arguments=["--typefileplot", "pdf"]) == 0
    assert calls == ["pdf"]


def test_run_plot_cli_applies_requested_theme_and_restores_settings(tmp_path):
    import matplotlib.pyplot as plt
    from PIL import Image

    original = dict(plt.rcParams)

    def run(typefileplot):
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1], color='red')
        assert axis.xaxis.label.get_color() == 'white'
        tdpy.save_figure(figure, tmp_path / 'dark', typefileplot, close_figure=True)

    run_plot_cli(run, arguments=['--typeplotback', 'dark'])

    with Image.open(tmp_path / 'dark.png') as image:
        assert image.convert('RGB').getpixel((0, 0)) == (0, 0, 0)
    assert dict(plt.rcParams) == original


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


def test_dispatch_cli_aliases_share_one_implementation():
    from tdpy import util

    assert tdpy.dispatch_cli is dispatch_cli
    assert util.dispatch_cli is dispatch_cli
    assert dispatch_cli.__module__ == 'tdpy.cli'


def test_dispatch_cli_reads_sys_argv(monkeypatch):
    import sys

    monkeypatch.setattr(sys, 'argv', ['analysis.py', 'run', 'argument'])

    assert dispatch_cli({'run': lambda value: value}) == 'argument'


def test_dispatch_cli_does_not_load_legacy_utilities():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, '-c', (
            "import sys; import tdpy; "
            "assert tdpy.dispatch_cli({'run': lambda value: value}, ['run', 'ok']) == 'ok'; "
            "assert 'tdpy.util' not in sys.modules"
        )],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('arguments', [[], ['missing'], ['_private']])
def test_dispatch_cli_rejects_invalid_commands(arguments):
    with pytest.raises(SystemExit):
        tdpy.dispatch_cli({'_private': lambda: None}, arguments)