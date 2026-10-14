import numpy as np
import pytest

from tdpy.population import partition_classified_population, population_feature_values, subset_population


def test_population_feature_values_preserves_nested_arrays_and_metadata():
    values = np.array([1.0, 2.0])  # [kg]
    labeled = {
        "mass": [values, "kg"],
        "nested": {"mask": [np.array([True, False]), []], "raw": values},
        "label": "Toy population",
        "metadata": ["first", "second", "third"],
        "tuple": ("first", "second"),
        "count": 2,
    }
    flattened = population_feature_values(labeled)
    assert flattened is not labeled
    assert flattened["mass"] is values
    assert flattened["nested"] is not labeled["nested"]
    assert flattened["nested"]["mask"] is labeled["nested"]["mask"][0]
    assert flattened["nested"]["raw"] is values
    assert flattened["metadata"] is labeled["metadata"]
    assert flattened["tuple"] is labeled["tuple"]
    assert flattened["label"] == "Toy population"
    assert flattened["count"] == 2
    assert labeled["mass"][1] == "kg"
    assert len(labeled["nested"]["mask"]) == 2


def test_population_feature_values_retains_empty_population_and_features():
    assert population_feature_values({}) == {}
    assert population_feature_values({"empty": []}) == {"empty": []}


def test_partition_classified_population():
    relevant = np.array([0, 1, 2])
    irrelevant = np.array([3, 4, 5])
    positive = np.array([1, 2, 3])
    negative = np.array([0, 4, 5])

    parts = partition_classified_population(relevant, irrelevant, positive, negative)

    np.testing.assert_array_equal(parts["trpo"], [1, 2])
    np.testing.assert_array_equal(parts["trne"], [4, 5])
    np.testing.assert_array_equal(parts["flpo"], [3])
    np.testing.assert_array_equal(parts["flne"], [0])


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