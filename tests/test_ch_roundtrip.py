"""``NWBSink`` writes a structured ``ch`` axis field for field (issue #46)."""

from __future__ import annotations

import numpy as np
import pynwb
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.nwb import (
    CHANNEL_DTYPE,
    NWBSinkConsumer,
    NWBSinkSettings,
    NWBSlicer,
    ReferenceClockType,
    array_identity,
)


@pytest.fixture(autouse=True)
def _reset_nwbsink_shared():
    yield
    NWBSinkConsumer.shared_start_datetime = None
    NWBSinkConsumer.shared_t0 = None
    NWBSinkConsumer.shared_clock_type = None


def _msg(key, ch, n=20, offset=0.0):
    return AxisArray(
        data=np.random.randn(n, len(ch)).astype(np.float32),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=100.0, offset=offset), "ch": AxisArray.CoordinateAxis(ch, dims=["ch"])},
        key=key,
    )


def _write(path, *msgs, split=False, **settings):
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(filepath=path, overwrite_old=True, inc_clock=ReferenceClockType.UNKNOWN, **settings)
    )
    for msg in msgs:
        sink._process(msg)
    if split:
        sink._split()
        for msg in msgs:
            sink._process(msg)
    sink.close(write=True)


def _read_ch(path, key, structured=True):
    slicer = NWBSlicer(path, dejitter=False, structured_ch_axis=structured, stream_keys=[key])
    try:
        return slicer.get_stream_info(key).template.axes["ch"].data
    finally:
        slicer.close()


def _electrode_ch(labels, extra=True):
    """Live-source records plus an extra ``unit`` field, labels deliberately out of order."""
    dtype = CHANNEL_DTYPE.descr + ([("unit", "U2")] if extra else [])
    ch = np.zeros(len(labels), dtype=dtype)
    ch["label"] = labels
    ch["x"] = np.arange(len(labels)) * 400
    ch["y"] = 800
    ch["size"] = 40
    ch["bank"] = "B"
    ch["elec"] = np.arange(len(labels)) + 1
    ch["headstage"] = 2
    ch["array"] = [array_identity(lbl, 2, "B") for lbl in labels]
    if extra:
        ch["unit"] = "uV"
    return ch


def _decoder_ch():
    ch = np.zeros(2, dtype=[("label", "U2"), ("unit", "U3"), ("gain", "f8"), ("index", "i4")])
    ch["label"] = ["vx", "vy"]
    ch["unit"] = "m/s"
    ch["gain"] = [1.5, 2.0]
    ch["index"] = [3, 7]
    return ch


def _assert_same(actual, expected):
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_electrode_stream_roundtrips_with_extra_field(tmp_path):
    path = tmp_path / "el.nwb"
    ch = _electrode_ch(["elec1-c", "elec1-a", "elec2-b"])
    # A second stream over a subset in another order: the region must follow each stream's order.
    sub = ch[[2, 0]]
    _write(path, _msg("Neural", ch), _msg("Sub", sub))

    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        assert isinstance(nwb.acquisition["Neural"], pynwb.ecephys.ElectricalSeries)
        assert {"rel_x", "rel_y", "term", "bank", "headstage", "size", "unit"} <= set(nwb.electrodes.colnames)
        assert "array" not in nwb.electrodes.colnames
        assert len(nwb.electrodes) == 3
        assert list(nwb.electrodes["unit"][:]) == ["uV"] * 3
        np.testing.assert_array_equal(nwb.electrodes["rel_x"][:], ch["x"])
        np.testing.assert_array_equal(nwb.electrodes["term"][:], ch["elec"])
    np.testing.assert_array_equal(_read_ch(path, "Neural", structured=False), ch["label"])
    np.testing.assert_array_equal(_read_ch(path, "Sub", structured=False), sub["label"])


def test_live_shaped_axis_roundtrips_to_the_live_dtype(tmp_path):
    path = tmp_path / "live.nwb"
    ch = _electrode_ch(["elec1-1", "elec1-2"], extra=False)
    _write(path, _msg("Neural", ch))
    _assert_same(_read_ch(path, "Neural"), ch)


def test_non_electrode_stream_writes_a_channel_table(tmp_path):
    path = tmp_path / "dec.nwb"
    ch = _decoder_ch()
    _write(path, _msg("Decoder", ch))

    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        series = nwb.acquisition["Decoder"]
        assert type(series) is pynwb.TimeSeries
        assert series.unit == "m/s"
        assert series.conversion == 1.0
        assert nwb.electrodes is None
        table = nwb.acquisition["Decoder_channels"]
        assert table.colnames == ("label", "unit", "gain", "index")
        for name in ch.dtype.names:
            np.testing.assert_array_equal(table[name][:], ch[name])
        assert table["index"].data.dtype == np.int32


def test_mixed_units_are_marked(tmp_path):
    path = tmp_path / "mixed.nwb"
    ch = _decoder_ch()
    ch["unit"] = ["m/s", "rad"]
    _write(path, _msg("Decoder", ch))
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        assert io.read().acquisition["Decoder"].unit == "mixed"


def test_labelled_stream_can_opt_out_of_the_electrodes_table(tmp_path):
    path = tmp_path / "optout.nwb"
    _write(path, _msg("Kin", np.array(["px", "py"])), _msg("Neural", np.array(["e1"])), non_electrode_streams=("Kin",))
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        assert type(nwb.acquisition["Kin"]) is pynwb.TimeSeries
        assert list(nwb.electrodes["label"][:]) == ["e1"]
        assert list(nwb.acquisition["Kin_channels"]["label"][:]) == ["px", "py"]


def test_split_keeps_electrodes_and_channel_table(tmp_path):
    """The second file of a split gets the same electrodes and channel table (#45)."""
    path = tmp_path / "split.nwb"
    el, dec = _electrode_ch(["elec1-a", "elec1-b"], extra=False), _decoder_ch()
    _write(path, _msg("Neural", el), _msg("Decoder", dec), split=True, split_bytes=10**9)
    second = tmp_path / "split_01.nwb"
    with pynwb.NWBHDF5IO(str(second), "r") as io:
        nwb = io.read()
        assert isinstance(nwb.acquisition["Neural"], pynwb.ecephys.ElectricalSeries)
        assert list(nwb.acquisition["Decoder_channels"]["label"][:]) == ["vx", "vy"]
    _assert_same(_read_ch(second, "Neural"), el)
