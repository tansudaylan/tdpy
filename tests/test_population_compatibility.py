import numpy as np
import pytest

from tdpy import retr_subp


def test_population_adapter_retains_unit_labels_counts_and_indices():
    populations = {'all': {'mass': [np.array([1.0, 2.0, 3.0]), 'kg']}}
    counts = {}
    indices = {'all': {}}

    retr_subp(populations, counts, indices, 'all', 'chosen', np.array([0, 2]))

    np.testing.assert_array_equal(populations['chosen']['mass'][0], [1.0, 3.0])
    assert populations['chosen']['mass'][1] == 'kg'
    assert counts['chosen'] == 2
    np.testing.assert_array_equal(indices['all']['chosen'], [0, 2])
    assert indices['chosen'] == {}


def test_population_adapter_retains_list_rejection():
    with pytest.raises(ValueError):
        retr_subp({'all': {}}, {}, {'all': {}}, 'all', 'chosen', [0])