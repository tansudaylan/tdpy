import numpy as np
import pytest

from tdpy.population import subset_population


def test_subset_population_copies_values_units_and_metadata():
    populations = {
        "all": {
            "mass": [np.array([1.0, 2.0, 3.0]), "kg"],
            "selected": [np.array([False, True, True]), ""],
        }
    }
    sample_counts = {}
    sample_indices = {}

    indices = subset_population(
        populations,
        "all",
        "chosen",
        [0, 2],
        sample_counts,
        sample_indices,
    )

    np.testing.assert_array_equal(indices, np.array([0, 2]))
    np.testing.assert_array_equal(populations["chosen"]["mass"][0], np.array([1.0, 3.0]))
    assert populations["chosen"]["mass"][1] == "kg"
    assert sample_counts["chosen"] == 2
    np.testing.assert_array_equal(sample_indices["all"]["chosen"], indices)
    assert sample_indices["chosen"] == {}


def test_subset_population_supports_empty_indices_without_metadata():
    populations = {"all": {"mass": [np.array([1.0]), "kg"]}}

    indices = subset_population(populations, "all", "chosen", None)

    assert indices.dtype == int
    assert indices.size == 0
    assert populations["chosen"]["mass"][0].size == 0
    assert populations["chosen"]["mass"][1] == "kg"


def test_subset_population_rejects_malformed_feature():
    populations = {"all": {"mass": np.array([1.0])}}

    with pytest.raises(ValueError, match=r"must be a \[values, unit\] list"):
        subset_population(populations, "all", "chosen", [0])