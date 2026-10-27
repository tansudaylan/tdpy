import matplotlib.pyplot as plt
import numpy as np
import pytest

import tdpy


def test_plot_file_path_replaces_extension(tmp_path):
    path = tdpy.plot_file_path(tmp_path / 'figure.old', 'pdf')

    assert path == str(tmp_path / 'figure.pdf')


def test_sign_code_renders_plain_text_without_latex():
    figure, axis = plt.subplots()
    tdpy.sign_code(axis, 'miletos')

    assert axis.texts[-1].get_text() == 'github.com/tansudaylan/miletos'
    assert axis.texts[-1].get_fontweight() == 'bold'
    plt.close(figure)


@pytest.mark.parametrize('typeplotback', ['norm', 'white', 'dark', 'black'])
def test_sign_code_accepts_shared_theme_aliases(typeplotback):
    from matplotlib.colors import to_rgba

    figure, axis = plt.subplots()
    tdpy.sign_code(axis, 'miletos', typeplotback=typeplotback)
    background, foreground = tdpy.plot_background_colors(typeplotback)
    frame = axis.texts[-1].get_bbox_patch()

    assert frame.get_facecolor() == to_rgba(background)
    assert frame.get_edgecolor() == to_rgba(foreground)
    plt.close(figure)


@pytest.mark.parametrize(('typeplotback', 'colors'), [
    ('norm', ('white', 'black')),
    ('white', ('white', 'black')),
    ('dark', ('black', 'white')),
    ('black', ('black', 'white')),
])
def test_plot_background_colors_supports_aliases(typeplotback, colors):
    assert tdpy.plot_background_colors(typeplotback) == colors


@pytest.mark.parametrize(('typefileplot', 'typeplotback', 'color'), [
    ('png', 'norm', (1.0, 1.0, 1.0, 1.0)),
    ('png', 'white', (1.0, 1.0, 1.0, 1.0)),
    ('pdf', 'dark', (0.0, 0.0, 0.0, 1.0)),
    ('pdf', 'black', (0.0, 0.0, 0.0, 1.0)),
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
    if typefileplot == 'png':
        from PIL import Image

        with Image.open(path) as saved:
            np.testing.assert_allclose(saved.info['dpi'], [300., 300.], atol=0.01)
    plt.close(figure)


def test_write_animation_pads_frames_and_shares_one_palette(tmp_path):
    from PIL import Image

    frames = []
    for index, size in enumerate(((3.0, 2.0), (3.5, 2.0))):
        figure, axis = plt.subplots(figsize=size, dpi=40)
        axis.imshow(np.arange(16.0).reshape(4, 4) * (index + 1), cmap='magma')
        frames.append(tdpy.figure_to_frame(figure))
    path = tdpy.write_animation(frames, tmp_path / 'movie.png', duration_ms=50)

    assert path == tmp_path / 'movie.gif'
    with Image.open(path) as animation:
        assert animation.n_frames == 2
        assert animation.size == (max(f.width for f in frames), max(f.height for f in frames))
    with pytest.raises(ValueError):
        tdpy.write_animation([], tmp_path / 'empty.gif')


@pytest.mark.parametrize('typefileplot', ['png', 'pdf'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('projection', [None, '3d'])
def test_save_figure_matches_document_point_size_without_changing_data(tmp_path, typefileplot, typeplotback, projection):
    from pathlib import Path
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text

    figure = plt.figure(figsize=(8.0, 6.0), layout='constrained')
    axis = figure.add_subplot(projection=projection)
    line, = axis.plot([0., 1., 2.], [1., 3., 2.], label='Measurements')
    axis.set(xlabel='Position', ylabel='Value', title='Panel title')
    axis.set_yscale('log')
    axis.annotate('A point', (1., 3.), fontsize=7)
    if projection == '3d':
        axis.set_zlabel('Third coordinate')
    axis.legend(title='Series', fontsize=8, title_fontsize=15)
    figure.suptitle('Figure title', fontsize=19)
    figure.legend(title='Whole figure', fontsize=9)
    get_coordinates = line.get_data_3d if projection == '3d' else line.get_data
    coordinates = [np.asarray(values).copy() for values in get_coordinates()]
    labels = [text.get_text() for text in figure.findobj(match=Text) if text.get_text()]
    settings = dict(plt.rcParams)

    path = tdpy.save_figure(figure, tmp_path / 'document_font', typefileplot,
                            typeplotback, font_size=11.0)

    FigureCanvasAgg(figure).draw()
    assert all(text.get_fontsize() == 11.0 for text in figure.findobj(match=Text))
    for label in labels:
        assert label in [text.get_text() for text in figure.findobj(match=Text)]
    for actual, expected in zip(get_coordinates(), coordinates):
        np.testing.assert_array_equal(actual, expected)
    assert dict(plt.rcParams) == settings
    assert Path(path).is_file()
    plt.close(figure)


@pytest.mark.parametrize('font_size', [0.0, -1.0, np.nan, np.inf, True, 11+0j, [10., 11.], '11', np.ma.masked])
def test_save_figure_rejects_invalid_document_font_before_writing(tmp_path, font_size):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])
    with pytest.raises(ValueError, match='font_size'):
        tdpy.save_figure(figure, tmp_path / 'not_created/invalid.png', font_size=font_size)
    assert not (tmp_path / 'not_created').exists()
    plt.close(figure)


def test_save_figure_omitted_document_font_retains_caller_sizes(tmp_path):
    figure, axis = plt.subplots()
    axis.set_title('Caller title', fontsize=17)
    note = axis.annotate('Caller annotation', (0.5, 0.5), fontsize=8)
    tdpy.save_figure(figure, tmp_path / 'caller_font.png')
    assert axis.title.get_fontsize() == 17
    assert note.get_fontsize() == 8
    plt.close(figure)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_plot_style_is_scoped_and_renders_theme(typeplotback, monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba
    from matplotlib.patches import BoxStyle

    monkeypatch.setitem(plt.rcParams, 'legend.frameon', False)
    monkeypatch.setitem(plt.rcParams, 'legend.fancybox', False)
    original = dict(plt.rcParams)
    background, foreground = tdpy.plot_background_colors(typeplotback)
    with tdpy.plot_style(typeplotback):
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1], color='red', label='data')
        axis.set(xlabel='x', ylabel='y', title='Title')
        legend = axis.legend()
        assert axis.get_facecolor() == to_rgba(background)
        assert axis.xaxis.label.get_color() == foreground
        assert to_rgba(legend.get_texts()[0].get_color()) == to_rgba(foreground)
        assert legend.get_frame_on()
        assert isinstance(legend.get_frame().get_boxstyle(), BoxStyle.Round)
        FigureCanvasAgg(figure).draw()
        np.testing.assert_array_equal(
            np.asarray(figure.canvas.buffer_rgba())[0, 0],
            np.array(to_rgba(background)) * 255,
        )
        plt.close(figure)
    assert dict(plt.rcParams) == original


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('projection', [None, '3d'])
@pytest.mark.parametrize('frame_on', [False, True])
def test_apply_plot_theme_styles_existing_axes_and_legends(typeplotback, projection, frame_on):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba
    from matplotlib.patches import BoxStyle

    figure = plt.figure()
    axis = figure.add_subplot(projection=projection)
    line, = axis.plot([0, 1], [0, 1], color='red', marker='o', label='data')
    axis.set(xlabel='x', ylabel='y', title='Title')
    axis.set_title('Left', loc='left')
    neutral_text = axis.annotate('Neutral', (0.3, 0.3), color='black')
    colored_text = axis.annotate('Colored', (0.6, 0.6), color='red')
    axis.grid(True)
    legend = axis.legend(title='Legend', frameon=frame_on, fancybox=False)
    figure_legend = figure.legend(frameon=frame_on, fancybox=False)
    figure.suptitle('Figure title')
    status_text = figure.text(0.5, 0.02, 'Status', color='red')
    original = dict(plt.rcParams)
    background, foreground = tdpy.plot_background_colors(typeplotback)

    assert tdpy.apply_plot_theme(figure, typeplotback) is figure

    FigureCanvasAgg(figure).draw()
    assert figure.get_facecolor() == to_rgba(background)
    assert axis.get_facecolor() == to_rgba(background)
    for coordinate in (axis.xaxis, axis.yaxis):
        assert coordinate.label.get_color() == foreground
        assert coordinate.get_offset_text().get_color() == foreground
        assert all(tick.get_color() == foreground for tick in coordinate.get_ticklabels())
    assert all(spine.get_edgecolor() == to_rgba(foreground) for spine in axis.spines.values())
    assert axis._left_title.get_color() == foreground
    assert figure._suptitle.get_color() == foreground
    assert status_text.get_color() == 'red'
    for current in (legend, figure_legend):
        assert current.get_frame_on()
        assert isinstance(current.get_frame().get_boxstyle(), BoxStyle.Round)
        assert current.get_frame().get_facecolor() == to_rgba(background)
        assert current.get_frame().get_alpha() == 1.0
        assert all(text.get_color() == foreground for text in current.get_texts())
        assert current.get_lines()[0].get_marker() == line.get_marker()
        assert current.get_lines()[0].get_color() == line.get_color()
    assert line.get_color() == 'red'
    assert neutral_text.get_color() == foreground
    assert colored_text.get_color() == 'red'
    assert dict(plt.rcParams) == original
    plt.close(figure)


def test_save_current_figure_uses_active_figure(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 0.0])

    path = tdpy.save_current_figure(tmp_path / 'current', 'pdf')

    assert path == str(tmp_path / 'current.pdf')
    assert (tmp_path / 'current.pdf').is_file()
    plt.close(figure)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_explicit_annotation_contrast_can_be_preserved_across_themes(typeplotback):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba
    from tdpy.plotting import preserve_text_color

    figure, axis = plt.subplots()
    values = np.array([[0.0, 1.0]])
    image = axis.imshow(values, vmin=0.0, vmax=1.0)
    dark_cell = axis.text(0, 0, '0%', color='white')
    bright_cell = axis.text(1, 0, '100%', color='black')
    default_text = axis.text(0.5, 0.9, 'Default', transform=axis.transAxes, color='black')
    figure_text = figure.text(0.5, 0.01, 'White inset', color='white')
    for text in (dark_cell, bright_cell, figure_text):
        assert preserve_text_color(text) is text

    for selected in (typeplotback, 'dark', 'white'):
        tdpy.apply_plot_theme(figure, selected)
        FigureCanvasAgg(figure).draw()
        assert to_rgba(dark_cell.get_color()) == to_rgba('white')
        assert to_rgba(bright_cell.get_color()) == to_rgba('black')
        assert to_rgba(figure_text.get_color()) == to_rgba('white')
        assert default_text.get_color() == tdpy.plot_background_colors(selected)[1]
        np.testing.assert_array_equal(image.get_array(), values)
    plt.close(figure)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_rounded_legend_preserves_line_markers_and_filled_area(typeplotback):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import BoxStyle

    figure, axis = plt.subplots()
    line, = axis.plot([0., 1.], [1., 2.], marker='s', color='red', label='Measured')
    region = axis.fill_between([0., 1.], [0.8, 1.8], [1.2, 2.2],
                               color='blue', alpha=0.3, label='Interval')
    legend = axis.legend(fancybox=False, frameon=False)
    coordinates = line.get_xydata().copy()
    vertices = region.get_paths()[0].vertices.copy()
    tdpy.apply_plot_theme(figure, typeplotback)
    FigureCanvasAgg(figure).draw()
    assert legend.get_frame_on() and legend.get_frame().get_alpha() == 1.
    assert isinstance(legend.get_frame().get_boxstyle(), BoxStyle.Round)
    assert [text.get_text() for text in legend.get_texts()] == ['Measured', 'Interval']
    assert legend.get_lines()[0].get_marker() == line.get_marker()
    assert legend.get_lines()[0].get_color() == line.get_color()
    np.testing.assert_allclose(legend.get_patches()[0].get_facecolor(), region.get_facecolor()[0])
    np.testing.assert_array_equal(line.get_xydata(), coordinates)
    np.testing.assert_array_equal(region.get_paths()[0].vertices, vertices)
    plt.close(figure)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('result_kind', ['figure', 'axis', 'tuple', 'dictionary', 'none'])
def test_themed_plot_preserves_return_values_and_scopes_settings(typeplotback, result_kind):
    import inspect
    from matplotlib.colors import to_rgba

    captured = []

    @tdpy.themed_plot
    def plot_data(values, *, label='Data'):
        figure, axis = plt.subplots(facecolor='gray')
        axis.plot(values, color='red', label=label)
        captured.append(figure)
        return {'figure': figure, 'axis': axis, 'tuple': (figure, values),
                'dictionary': {'figure': figure, 'values': values}, 'none': None}[result_kind]

    assert inspect.signature(plot_data).parameters['typeplotback'].kind == inspect.Parameter.KEYWORD_ONLY
    values = np.array([1.0, 3.0, 2.0])
    with tdpy.plot_style('dark' if typeplotback == 'white' else 'white'):
        original = dict(plt.rcParams)
        result = plot_data(values, typeplotback=typeplotback)
        assert dict(plt.rcParams) == original
        figure = captured[0]
        try:
            background, foreground = tdpy.plot_background_colors(typeplotback)
            assert figure.get_facecolor() == to_rgba(background)
            assert figure.axes[0].xaxis.label.get_color() == foreground
            assert figure.axes[0].lines[0].get_color() == 'red'
            np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), values)
            if result_kind == 'figure':
                assert result is figure
            elif result_kind == 'axis':
                assert result is figure.axes[0]
            elif result_kind == 'tuple':
                assert result[1] is values
            elif result_kind == 'dictionary':
                assert result['values'] is values
            else:
                assert result is None
        finally:
            plt.close(figure)


def test_themed_plot_restores_settings_after_original_failure():
    @tdpy.themed_plot
    def fail():
        raise RuntimeError('Original failure')

    with tdpy.plot_style('white'):
        original = dict(plt.rcParams)
        with pytest.raises(RuntimeError, match='Original failure'):
            fail(typeplotback='dark')
        assert dict(plt.rcParams) == original
        assert tdpy.plot_background_colors() == ('white', 'black')


def test_themed_plot_styles_input_axis_and_preserves_current_figure():
    first, axis = plt.subplots()
    second = None

    @tdpy.themed_plot
    def plot_on_axis(values, ax):
        ax.plot(values, color='red')
        plt.figure()
        plt.figure(first.number)

    try:
        plot_on_axis([1, 2], ax=axis, typeplotback='dark')
        second = next(number for number in plt.get_fignums() if number != first.number)
        assert plt.gcf() is first
        assert first.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
        assert axis.xaxis.label.get_color() == 'white'
    finally:
        plt.close(first)
        if second is not None:
            plt.close(second)


def test_save_plot_preserves_format_options_and_return_value(monkeypatch):
    figure = plt.figure()
    captured = {}

    def save(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return 'original return'

    monkeypatch.setattr(figure, 'savefig', save)
    try:
        assert tdpy.save_plot(figure, 'plot.svg', dpi=150, transparent=True, typeplotback='dark') == 'original return'
        assert captured['args'] == ('plot.svg',)
        assert captured['kwargs'] == {'dpi': 150, 'transparent': True, 'facecolor': (0.0, 0.0, 0.0, 1.0)}
    finally:
        plt.close(figure)


def test_themed_plot_supports_existing_arrays_of_axes():
    figure, axes = plt.subplots(2, 1, squeeze=False)

    @tdpy.themed_plot
    def draw(values, axes):
        for axis in axes.flat:
            axis.plot(values, color='red')
        return axes

    try:
        assert draw([1, 2, 3], axes=axes, typeplotback='dark') is axes
        assert figure.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
        assert all(axis.xaxis.label.get_color() == 'white' for axis in axes.flat)
    finally:
        plt.close(figure)


def test_themed_plot_supports_existing_figure_owners():
    from types import SimpleNamespace

    figure, axis = plt.subplots()
    owner = SimpleNamespace(fig=figure)

    @tdpy.themed_plot
    def draw(owner):
        axis.plot([1, 2], color='red')
        return owner

    try:
        assert draw(owner, typeplotback='dark') is owner
        assert figure.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
        assert axis.xaxis.label.get_color() == 'white'
    finally:
        plt.close(figure)


@pytest.mark.parametrize('mode', ['default', 'positional', 'keyword'])
def test_themed_plot_preserves_existing_background_parameter(mode):
    import inspect

    calls = []

    def original(values, typeplotback='white'):
        calls.append((values, typeplotback, tdpy.plot_background_colors()))
        return values

    draw = tdpy.themed_plot(original)
    assert inspect.signature(draw) == inspect.signature(original)
    values = np.array([1, 2, 3])
    with tdpy.plot_style('dark'):
        if mode == 'default':
            assert draw(values) is values
            assert calls[-1][1:] == ('white', ('white', 'black'))
        elif mode == 'positional':
            assert draw(values, 'dark') is values
            assert calls[-1][1:] == ('dark', ('black', 'white'))
        else:
            assert draw(values, typeplotback='dark') is values
            assert calls[-1][1:] == ('dark', ('black', 'white'))
        assert tdpy.plot_background_colors() == ('black', 'white')


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_show_figures_themes_all_open_figures_without_changing_current(monkeypatch, typeplotback):
    from matplotlib.colors import to_rgba

    plt.close('all')
    figures = [plt.subplots()[0], plt.subplots()[0]]
    plt.figure(figures[0].number)
    background, foreground = tdpy.plot_background_colors(typeplotback)
    displayed = []

    def inspect_display(**kwargs):
        displayed.append(kwargs)
        assert plt.gcf() is figures[0]
        for figure in figures:
            assert figure.get_facecolor() == to_rgba(background)
            assert figure.axes[0].xaxis.label.get_color() == foreground

    monkeypatch.setattr(plt, 'show', inspect_display)
    original_settings = dict(plt.rcParams)
    try:
        assert tdpy.show_figures(typeplotback=typeplotback, block=False) is None
        assert displayed == [{'block': False}]
        assert dict(plt.rcParams) == original_settings
    finally:
        for figure in figures:
            plt.close(figure)


def test_scoped_dark_theme_is_inherited_by_saves_and_animation_frames(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    with tdpy.plot_style('dark'):
        with tdpy.plot_style():
            figure, _ = plt.subplots()
            tdpy.apply_plot_theme(figure)
            assert figure.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
            FigureCanvasAgg(figure)
            image_path = tdpy.save_figure(figure, tmp_path / 'inherited')
            frame = tdpy.figure_to_frame(figure)
            assert frame.getpixel((0, 0)) == (0, 0, 0)
            animation_path = tdpy.write_animation(
                [frame, Image.new('RGB', (frame.width - 1, frame.height), 'red')],
                tmp_path / 'inherited.gif',
            )
        with Image.open(image_path) as image:
            assert image.convert('RGB').getpixel((0, 0)) == (0, 0, 0)
        with Image.open(animation_path) as animation:
            animation.seek(1)
            assert animation.convert('RGB').getpixel((frame.width - 1, 0)) == (0, 0, 0)
    figure, _ = plt.subplots()
    FigureCanvasAgg(figure)
    assert tdpy.figure_to_frame(figure).getpixel((0, 0)) == (255, 255, 255)


def test_save_current_figure_forwards_savefig_options(tmp_path, monkeypatch):
    figure = plt.figure()
    arguments = {}
    monkeypatch.setattr(figure, 'savefig', lambda path, **kwargs: arguments.update(kwargs))

    tdpy.save_current_figure(tmp_path / 'current', dpi=150, transparent=True)

    assert arguments['dpi'] == 150
    assert arguments['transparent'] is True
    plt.close(figure)


@pytest.mark.parametrize('kind', ['generic', 'line', 'histogram', 'map', 'timeline'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_legacy_plot_entry_points_save_requested_theme(tmp_path, monkeypatch, kind, typeplotback, inherit):
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy import util

    x_values = np.linspace(0.1, 1.0, 10)
    y_values = x_values ** 2
    background, foreground = tdpy.plot_background_colors(typeplotback)
    save = plt.Figure.savefig
    captured = []

    def capture_and_save(figure, path, **kwargs):
        captured.append(figure)
        assert figure.get_facecolor() == to_rgba(background)
        for axis in figure.axes:
            assert axis.get_facecolor() == to_rgba(background)
            assert axis.xaxis.label.get_color() == foreground
            assert axis.yaxis.label.get_color() == foreground
        if kind in {'generic', 'line'}:
            np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), x_values)
            np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), y_values)
        elif kind == 'histogram':
            assert sum(patch.get_height() for patch in figure.axes[0].patches) == len(x_values)
        elif kind == 'map':
            np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), np.arange(16.0).reshape(4, 4).T)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    path = tmp_path / 'theme.png'
    with tdpy.plot_style(outer_theme):
        original = dict(plt.rcParams)
        if kind == 'generic':
            util.plot_gene(path, x_values, y_values, listlegd=['Data'], **options)
        elif kind == 'line':
            util.plot_plot(path, x_values, y_values, 'x', 'y', 'self', **options)
        elif kind == 'histogram':
            util.plot_hist(str(tmp_path / 'theme'), x_values, 'Samples', **options)
            path = tmp_path / 'theme_hist.png'
        elif kind == 'map':
            util.plot_maps(path, np.arange(16.0), pixltype='cart', **options)
        else:
            rows = {'Project': {'listdictelem': [
                {'limtdate': ['2020-01-01', '2022-01-01'], 'colr': 'tab:blue', 'strg': 'Project'},
            ]}}
            util.plot_timeline(rows, pathbase=str(tmp_path) + '/', strgplot='theme', typetickdate='yearly', **options)
        assert dict(plt.rcParams) == original
    assert len(captured) == 1
    with Image.open(path) as image:
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert image.convert('RGB').getpixel((0, 0)) == expected


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_legacy_footprint_wrapper_forwards_theme_to_owner(monkeypatch, tmp_path, typeplotback):
    import tdpy.tess as tess
    from tdpy import util

    captured = {}

    def plot_map(sectors, path, **kwargs):
        captured.update(kwargs)
        return {'Field': tuple(sectors)}, path

    monkeypatch.setattr(tess, 'plot_tess_sector_map', plot_map)
    result, path = util.plot_lsst_ddf_tess_footprint(
        1, tmp_path, fields=[('Field', 10.0, 20.0)], typeplotback=typeplotback,
    )
    assert tdpy.plot_background_colors(captured['typeplotback']) == tdpy.plot_background_colors(typeplotback)
    assert result == {'Field': [1]}
    assert path == str(tmp_path / 'Sector1_DDF.png')


def test_save_figure_creates_parent_and_optionally_closes(tmp_path):
    figure = plt.figure()
    figure_number = figure.number

    path = tdpy.save_figure(
        figure,
        tmp_path / 'nested' / 'figure',
        close_figure=True,
    )

    assert path == str(tmp_path / 'nested' / 'figure.png')
    assert (tmp_path / 'nested' / 'figure.png').is_file()
    assert not plt.fignum_exists(figure_number)


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