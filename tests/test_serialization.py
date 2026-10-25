import numpy as np
import pytest

from tdpy.serialization import read_first_line, summarize_json_value
from tdpy.serialization import read_json, write_json, write_fits, write_bytes


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


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
def test_fits_writer_preserves_data_header_narration_and_overwrite(tmp_path, capsys, shape):
    from astropy.io import fits
    import tdpy

    assert tdpy.write_fits is write_fits
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    path = tmp_path / "array.fits"
    header = fits.Header({"BUNIT": "count", "OBJECT": "synthetic fixture"})
    write_fits(path, data, header=header)
    assert capsys.readouterr().out == f"Writing to {path}...\n"
    print(f"Reading from {path}...")
    restored, restored_header = fits.getdata(path, header=True)
    np.testing.assert_array_equal(restored, data)
    assert restored_header["BUNIT"] == "count"
    assert restored_header["OBJECT"] == "synthetic fixture"
    with pytest.raises(OSError):
        write_fits(path, data + 1.0)
    print(f"Reading from {path}...")
    np.testing.assert_array_equal(fits.getdata(path), data)
    write_fits(path, data + 1.0, header=header, overwrite=True)
    print(f"Reading from {path}...")
    np.testing.assert_array_equal(fits.getdata(path), data + 1.0)


def test_planck_fits_read_helpers_narrate_and_preserve_results(
    monkeypatch, capsys
):
    import ast
    from types import SimpleNamespace
    from pathlib import Path
    from tdpy.verbosity import print as verbosity_print

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    util_path = Path(__file__).resolve().parents[1] / "tdpy" / "util.py"
    verbosity_print(f"Reading from {util_path}...")
    source = util_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(util_path))
    helpers = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_read_fits_data", "_open_fits"}
    ]
    conversion = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "retr_mapsplnkfreq"
    )
    direct_reads = [
        node for node in ast.walk(conversion)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pf"
        and node.func.attr in {"getdata", "open"}
    ]
    narrated_reads = [
        node for node in ast.walk(conversion)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_read_fits_data", "_open_fits"}
    ]
    assert not direct_reads
    assert len(narrated_reads) == 7

    data = object()
    hdus = object()
    read_calls = []
    open_calls = []
    namespace = {
        "pf": SimpleNamespace(
            getdata=lambda path, *args, **kwargs: read_calls.append((path, args, kwargs)) or data,
            open=lambda path, *args, **kwargs: open_calls.append((path, args, kwargs)) or hdus,
        ),
        "print": verbosity_print,
    }
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(util_path), "exec"),
        namespace,
    )
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    assert namespace["_read_fits_data"]("map.fits", 1) is data
    assert namespace["_open_fits"]("bandpass.fits") is hdus
    assert capsys.readouterr().out == ""

    read_calls.clear()
    open_calls.clear()
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    assert namespace["_read_fits_data"]("map.fits", 1) is data
    assert namespace["_open_fits"]("bandpass.fits") is hdus
    assert read_calls == [("map.fits", (1,), {})]
    assert open_calls == [("bandpass.fits", (), {})]
    assert capsys.readouterr().out == (
        "Reading from map.fits...\nReading from bandpass.fits...\n"
    )


def test_diffuse_map_writer_uses_shared_fits_output(tmp_path, monkeypatch):
    from astropy.io import fits
    from unittest.mock import Mock
    from tdpy import util

    monkeypatch.setenv("PCAT_PATH", str(tmp_path))
    diffuse_map = np.arange(60, dtype=float).reshape(5, 12)
    monkeypatch.setattr(util, "retr_sbrtfdfm", lambda bins, side: diffuse_map)
    monkeypatch.setattr(util, "smth_ferm", lambda values, energies, reconstruction: np.repeat(values, 2, axis=2))
    writer = Mock(wraps=write_fits)
    monkeypatch.setattr(util, "write_fits", writer)
    path = tmp_path / "data" / "sbrtfdfm0001rec7fulligal.fits"
    util.writ_sbrtfdfm(numbside=1)
    writer.assert_called_once()
    assert writer.call_args.args[0] == str(path)
    assert writer.call_args.kwargs == {"overwrite": True}
    print(f"Reading from {path}...")
    np.testing.assert_array_equal(fits.getdata(path), np.repeat(diffuse_map[:, :, None], 2, axis=2))
    util.writ_sbrtfdfm(numbside=1)


def test_text_writer_uses_exact_narration_and_respects_verbosity(
    tmp_path, monkeypatch, capsys
):
    from tdpy import util
    from tdpy.verbosity import print as verbosity_print

    path = tmp_path / "nested" / "note.txt"
    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    util.write_text(path, "quiet")
    assert capsys.readouterr().out == ""
    verbosity_print(f"Reading from {path}...")
    assert path.read_text(encoding="utf-8") == "quiet"
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    util.write_text(path, "verbose")
    verbosity_print(f"Reading from {path}...")
    assert path.read_text(encoding="utf-8") == "verbose"
    assert capsys.readouterr().out == (
        f"Writing to {path}...\nReading from {path}...\n"
    )


def test_fermi_diffuse_model_reader_narrates_fits_path(
    tmp_path, monkeypatch, capsys
):
    import ast
    import os
    from pathlib import Path
    from scipy import interpolate
    from tdpy.verbosity import print as verbosity_print

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    util_path = Path(__file__).resolve().parents[1] / "tdpy" / "util.py"
    print(f"Reading from {util_path}...")
    tree = ast.parse(util_path.read_text(encoding="utf-8"), filename=str(util_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "retr_sbrtfdfm"
    )
    read_calls = []

    def getdata(path, extension):
        read_calls.append((path, extension))
        if extension == 0:
            return np.arange(12, dtype=float).reshape(3, 2, 2) + 1.0
        return np.array([1000.0, 2000.0, 4000.0])

    namespace = {
        "np": np,
        "os": os,
        "get_data_path": lambda: str(tmp_path),
        "pf": type("Fits", (), {"getdata": staticmethod(getdata)}),
        "cart_heal": lambda image, numbside: np.full(12, image.sum()),
        "interpolate": interpolate,
        "trapz": np.trapezoid,
        "print": verbosity_print,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(util_path), "exec"),
        namespace,
    )
    capsys.readouterr()
    path = os.path.join(str(tmp_path), "gll_iem_v06.fits")
    bins = np.array([1.0, 2.0, 4.0])

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    quiet_result = namespace["retr_sbrtfdfm"](bins, numbside=1)
    assert read_calls == [(path, 0), (path, 1)]
    assert quiet_result.shape == (2, 12)
    assert np.isfinite(quiet_result).all()
    assert capsys.readouterr().out == ""

    read_calls.clear()
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    verbose_result = namespace["retr_sbrtfdfm"](bins, numbside=1)
    np.testing.assert_array_equal(verbose_result, quiet_result)
    assert read_calls == [(path, 0), (path, 1)]
    assert capsys.readouterr().out == f"Reading from {path}...\n"


@pytest.mark.parametrize('as_string', [False, True])
def test_binary_writer_preserves_bytes_overwrites_and_narrates(tmp_path, capsys, as_string):
    import tdpy

    assert tdpy.write_bytes is write_bytes
    path = tmp_path / 'nested' / 'artifact.bin'
    argument = str(path) if as_string else path
    for content in (b'\x00\xff\r\n\x80', b'replacement', b''):
        assert write_bytes(argument, content) is None
        assert capsys.readouterr().out == f'Writing to {path}...\n'
        print(f'Reading from {path}...')
        assert path.read_bytes() == content
        capsys.readouterr()


def test_binary_writer_respects_verbosity(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    path = tmp_path / 'quiet.bin'

    write_bytes(path, b'quiet')

    assert capsys.readouterr().out == ''
    print(f'Reading from {path}...')
    assert path.read_bytes() == b'quiet'


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