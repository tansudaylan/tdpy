from types import SimpleNamespace

import numpy as np
import pytest

from tdpy import (
    minimum_distance_to_polyline_pixels,
    padded_text_bounds,
    rectangles_overlap,
)


class TextStub:
    def get_window_extent(self, renderer):
        assert renderer == "renderer"
        return BoundsStub()


class BoundsStub:
    def expanded(self, width_scale, height_scale):
        assert width_scale == 1.06
        assert height_scale == 1.12
        return SimpleNamespace(x0=1.0, x1=2.0, y0=3.0, y1=4.0)


def test_rectangles_overlap_including_shared_edges():
    assert rectangles_overlap((0, 2, 0, 2), (2, 3, 1, 3))
    assert not rectangles_overlap((0, 1, 0, 1), (2, 3, 2, 3))


def test_padded_text_bounds():
    assert padded_text_bounds(TextStub(), "renderer") == (1.0, 2.0, 3.0, 4.0)


def test_minimum_distance_to_polyline_pixels():
    line = np.array([[0.0, 0.0], [3.0, 4.0], [8.0, 4.0]])
    assert minimum_distance_to_polyline_pixels((0.0, 4.0), line) == 3.0


def test_minimum_distance_subsamples_long_lines():
    line = np.column_stack((np.arange(10.0), np.zeros(10)))
    assert minimum_distance_to_polyline_pixels((5.0, 3.0), line, 2) == 3.0


@pytest.mark.parametrize('point_container', ['tuple', 'list', 'array'])
@pytest.mark.parametrize('dtype,coordinates,point', [
    (np.uint8, [[2, 3]], (3, 3)),
    (np.int8, [[-128, 0]], (127, 0)),
    (np.uint64, [[2**63 + 1, 0]], (2**63, 0)),
    (np.int64, [[-2**63, 0]], (2**63 - 1, 0)),
])
def test_minimum_distance_avoids_integer_subtraction_wrap(dtype, coordinates, point, point_container):
    from math import hypot

    line = np.array(coordinates, dtype=dtype)
    original = line.copy()
    expected = min(hypot(int(row[0]) - point[0], int(row[1]) - point[1]) for row in line)
    supplied_point = {'tuple': point, 'list': list(point), 'array': np.array(point, dtype=dtype)}[point_container]
    original_point = np.array(supplied_point, dtype=dtype)

    with np.errstate(all='raise'):
        result = minimum_distance_to_polyline_pixels(supplied_point, line)

    assert result == expected
    np.testing.assert_array_equal(line, original)
    np.testing.assert_array_equal(np.array(supplied_point, dtype=dtype), original_point)