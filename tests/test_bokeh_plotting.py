import pytest

bokeh = pytest.importorskip("bokeh")

from bokeh.plotting import figure

from tdpy.plotting import save_bokeh_html


def test_save_bokeh_html_writes_standalone_interactive_document(tmp_path, capsys):
    plot = figure(title="Transit model", width=500, height=320)
    plot.line([0.0, 1.0, 2.0], [1.0, 0.99, 1.0])
    output_path = tmp_path / "nested" / "transit.html"

    result = save_bokeh_html(plot, output_path)

    assert result == output_path
    assert result.is_file()
    html = result.read_text(encoding="utf-8")
    assert "Bokeh" in html
    assert "Transit model" in html
    assert f"Writing to {output_path}..." in capsys.readouterr().out


def test_save_bokeh_html_requires_html_extension(tmp_path):
    plot = figure()

    with pytest.raises(ValueError, match=".html extension"):
        save_bokeh_html(plot, tmp_path / "transit.png")


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_save_bokeh_html_styles_layout_axes_and_legends(tmp_path, typeplotback):
    from bokeh.layouts import row
    from tdpy.plotting import plot_background_colors

    plots = [figure(title='Title'), figure(title='Second panel')]
    for plot in plots:
        plot.line([0, 1], [0, 1], color='red', legend_label='data')
        plot.xaxis.axis_label = 'Time'
    background, foreground = plot_background_colors(typeplotback)

    path = save_bokeh_html(row(*plots), tmp_path / 'layout.html', typeplotback=typeplotback)

    assert 'Second panel' in path.read_text()
    for plot in plots:
        assert plot.background_fill_color == background
        assert plot.border_fill_color == background
        assert plot.title.text_color == foreground
        for axis in plot.axis:
            assert axis.axis_line_color == foreground
            assert axis.major_tick_line_color == foreground
            assert axis.major_label_text_color == foreground
            assert axis.axis_label_text_color == foreground
        assert all(not grid.visible for grid in plot.grid)
        assert plot.legend[0].label_text_color == foreground
        assert plot.legend[0].background_fill_color == background
        assert plot.legend[0].background_fill_alpha == 1.0
        assert plot.renderers[0].glyph.line_color == 'red'