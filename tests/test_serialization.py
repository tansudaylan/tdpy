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