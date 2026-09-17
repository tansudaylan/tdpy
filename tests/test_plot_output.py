import matplotlib.pyplot as plt
import numpy as np
import pytest

import tdpy


def test_plot_file_path_replaces_extension(tmp_path):
    path = tdpy.plot_file_path(tmp_path / 'figure.old', 'pdf')

    assert path == str(tmp_path / 'figure.pdf')


@pytest.mark.parametrize(('typefileplot', 'typeplotback', 'color'), [
    ('png', 'norm', (1.0, 1.0, 1.0, 1.0)),
    ('pdf', 'dark', (0.0, 0.0, 0.0, 1.0)),
])
def test_save_figure_writes_styled_plot(tmp_path, capsys, typefileplot, typeplotback, color):
    figure, axis = plt.subplots()
    axis.grid(True)
    path = tdpy.save_figure(figure, tmp_path / 'figure.old', typefileplot, typeplotback)

    assert path == str(tmp_path / ('figure.' + typefileplot))
    assert (tmp_path / ('figure.' + typefileplot)).is_file()
    assert figure.get_facecolor() == color
    assert axis.get_facecolor() == color
    assert capsys.readouterr().out == 'Writing to %s...\n' % path
    plt.close(figure)


def test_save_current_figure_uses_active_figure(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 0.0])

    path = tdpy.save_current_figure(tmp_path / 'current', 'pdf')

    assert path == str(tmp_path / 'current.pdf')
    assert (tmp_path / 'current.pdf').is_file()
    plt.close(figure)


def test_save_current_figure_forwards_savefig_options(tmp_path, monkeypatch):
    figure = plt.figure()
    arguments = {}
    monkeypatch.setattr(figure, 'savefig', lambda path, **kwargs: arguments.update(kwargs))

    tdpy.save_current_figure(tmp_path / 'current', dpi=150, transparent=True)

    assert arguments['dpi'] == 150
    assert arguments['transparent'] is True
    plt.close(figure)


def test_load_text_data_logs_and_loads_csv(tmp_path, capsys):
    path = tmp_path / 'data.csv'
    np.savetxt(path, [[1.0, 2.0], [3.0, 4.0]], delimiter=',')

    data = tdpy.load_text_data(path, delimiter=',')

    np.testing.assert_array_equal(data, [[1.0, 2.0], [3.0, 4.0]])
    assert capsys.readouterr().out == 'Reading from %s...\n' % path


@pytest.mark.parametrize(('function', 'value'), [
    (tdpy.plot_file_path, 'svg'),
    (lambda path, value: tdpy.save_figure(plt.figure(), path, typeplotback=value), 'blue'),
])
def test_plot_output_rejects_unsupported_options(tmp_path, function, value):
    with pytest.raises(ValueError):
        function(tmp_path / 'figure', value)