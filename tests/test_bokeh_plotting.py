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