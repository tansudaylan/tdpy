import numpy as np

from tdpy.serialization import read_first_line, summarize_json_value
from tdpy.serialization import read_json, write_json


def test_read_first_line_narrates_path(tmp_path, capsys):
    path = tmp_path / "version.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    assert read_first_line(path) == "one"
    assert capsys.readouterr().out == f"Reading from {path}...\n"


def test_json_helpers_round_trip_and_log(tmp_path, capsys):
    path = tmp_path / "value.json"
    write_json(path, {"value": [1, 2]})
    assert read_json(path) == {"value": [1, 2]}
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert capsys.readouterr().out == f"Writing to {path}...\nReading from {path}...\n"


def test_summarize_json_value_compacts_arrays_and_scalars():
    value = {
        "array": np.array([2.0, np.nan, 5.0]),
        "scalar": np.int64(4),
        "sequence": (np.float64(1.5),),
    }
    assert summarize_json_value(value) == {
        "array": {"shape": [3], "minimum": 2.0, "maximum": 5.0},
        "scalar": 4,
        "sequence": [1.5],
    }