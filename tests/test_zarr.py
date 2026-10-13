"""NWB files stored as hdmf-zarr stores read exactly like their HDF5 originals."""

import functools
import http.server
import math
import threading

import numpy as np
import pytest

from ezmsg.nwb import NWBAxisArrayIterator, NWBIteratorSettings, ReferenceClockType
from ezmsg.nwb.slicer import is_zarr_store

NWBZarrIO = pytest.importorskip("hdmf_zarr.nwb").NWBZarrIO


@pytest.fixture(scope="session")
def test_zarr_path(test_nwb_path, tmp_path_factory):
    from pynwb import NWBHDF5IO

    path = tmp_path_factory.mktemp("zarr") / "test_synthetic.nwb.zarr"
    with NWBHDF5IO(str(test_nwb_path), "r") as src, NWBZarrIO(str(path), mode="w") as dst:
        dst.export(src_io=src, write_args=dict(link_data=False))
    # The fixture writes a non-schema ``rate`` attribute onto Broadband's timestamps with h5py
    # (create_test_nwb.py), which an NWB export does not carry; copy it so both files say the same.
    import h5py
    import zarr

    with h5py.File(test_nwb_path, "r") as f:
        rate = float(f["acquisition/Broadband/timestamps"].attrs["rate"])
    zarr.open_group(str(path), mode="r+")["acquisition/Broadband/timestamps"].attrs["rate"] = rate
    zarr.consolidate_metadata(str(path))
    return path


def _messages(filepath, **kw):
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(filepath=filepath, chunk_dur=1.0, reference_clock=ReferenceClockType.UNKNOWN, **kw)
    )
    return [m for m in it if math.prod(m.data.shape) > 0]


def _same(a, b):
    assert a.key == b.key and a.dims == b.dims
    if a.data.dtype.kind in "OSU":
        assert [str(x) for x in a.data.ravel()] == [str(x) for x in b.data.ravel()]
    else:
        np.testing.assert_array_equal(a.data, b.data)
    for dim in a.dims:
        ax_a, ax_b = a.axes.get(dim), b.axes.get(dim)
        if hasattr(ax_a, "gain"):
            assert (ax_a.gain, ax_a.offset) == pytest.approx((ax_b.gain, ax_b.offset))
        elif hasattr(ax_a, "data"):
            np.testing.assert_array_equal(np.asarray(ax_a.data), np.asarray(ax_b.data))
    scaling = {k: v for k, v in a.attrs.items() if k.startswith("nwb_scaling")}
    assert scaling == {k: v for k, v in b.attrs.items() if k.startswith("nwb_scaling")}


def test_is_zarr_store(test_nwb_path, test_zarr_path):
    assert is_zarr_store(test_zarr_path) and is_zarr_store("abfs://c/x/a.nwb.zarr/")
    assert is_zarr_store("https://h/c/a.nwb.zarr?sig=1")
    assert not is_zarr_store(test_nwb_path)


def test_every_stream_reads_the_same_from_zarr(test_nwb_path, test_zarr_path):
    h5, zr = _messages(test_nwb_path), _messages(test_zarr_path)

    assert [m.key for m in zr] == [m.key for m in h5]
    for a, b in zip(h5, zr):
        _same(a, b)


def test_a_remote_zarr_store_reads_the_same(test_nwb_path, test_zarr_path):
    """Over HTTP (fsspec), as a store in object storage is read."""
    pytest.importorskip("aiohttp")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(test_zarr_path.parent))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/{test_zarr_path.name}"
        keys = ["Broadband", "trials"]
        h5 = _messages(test_nwb_path, stream_keys=keys)
        zr = _messages(url, stream_keys=keys)
        assert [m.key for m in zr] == [m.key for m in h5]
        for a, b in zip(h5, zr):
            _same(a, b)
    finally:
        server.shutdown()
