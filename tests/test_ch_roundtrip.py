"""A ``ch`` axis written by ``NWBSink`` reads back field for field (issues #46, #47)."""

from __future__ import annotations

import datetime

import numpy as np
import pynwb
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from hdmf.common import DynamicTable, VectorData

from ezmsg.nwb import (
    CHANNEL_DTYPE,
    NWBAxisArrayIterator,
    NWBIteratorSettings,
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
    _assert_same(_read_ch(path, "Neural"), ch)
    _assert_same(_read_ch(path, "Sub"), sub)


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
        assert nwb.acquisition["Decoder_channels"].colnames == ("label", "unit", "gain", "index")
    _assert_same(_read_ch(path, "Decoder"), ch)
    np.testing.assert_array_equal(_read_ch(path, "Decoder", structured=False), ["vx", "vy"])


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
    np.testing.assert_array_equal(_read_ch(path, "Kin", structured=False), ["px", "py"])


def test_split_keeps_electrodes_and_channel_table(tmp_path):
    """The second file of a split gets the same electrodes and channel table (#45)."""
    path = tmp_path / "split.nwb"
    el, dec = _electrode_ch(["elec1-a", "elec1-b"]), _decoder_ch()
    _write(path, _msg("Neural", el), _msg("Decoder", dec), split=True, split_bytes=10**9)
    second = tmp_path / "split_01.nwb"
    with pynwb.NWBHDF5IO(str(second), "r") as io:
        assert isinstance(io.read().acquisition["Neural"], pynwb.ecephys.ElectricalSeries)
    _assert_same(_read_ch(second, "Neural"), el)
    _assert_same(_read_ch(second, "Decoder"), dec)


def test_iterator_carries_the_channel_table_axis(tmp_path):
    path = tmp_path / "it.nwb"
    ch = _decoder_ch()
    _write(path, _msg("Decoder", ch))
    it = NWBAxisArrayIterator(
        settings=NWBIteratorSettings(
            filepath=str(path),
            chunk_dur=0.1,
            reference_clock=ReferenceClockType.UNKNOWN,
            dejitter=False,
            structured_ch_axis=True,
        )
    )
    try:
        msg = next(m for m in it if m.data.shape[0])
    finally:
        it.close()
    _assert_same(msg.axes["ch"].data, ch)


def _hand_built(path, table_columns=None, description="fixture"):
    nwb = pynwb.NWBFile(
        session_description="hand built",
        identifier="hand",
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )
    nwb.add_acquisition(
        pynwb.TimeSeries(
            name="Kin", data=np.zeros((10, 2)), unit="n/a", rate=10.0, starting_time=0.0, description=description
        )
    )
    if table_columns:
        nwb.add_acquisition(
            DynamicTable(
                name="Kin_channels",
                description="text only",
                columns=[VectorData(name=k, description=k, data=v) for k, v in table_columns.items()],
            )
        )
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def test_text_channel_table_is_cast(tmp_path):
    path = tmp_path / "text.nwb"
    _hand_built(path, {"label": ["1", "2"], "index": ["3", "4"], "gain": ["1.5", "2"], "kind": ["a", "1"]})
    ch = _read_ch(path, "Kin")
    assert ch.dtype.names == ("label", "index", "gain", "kind")
    assert ch.dtype["label"].kind == "U" and ch.dtype["kind"].kind == "U"
    assert ch.dtype["index"].kind == "i" and ch.dtype["gain"].kind == "f"
    np.testing.assert_array_equal(ch["label"], ["1", "2"])
    np.testing.assert_array_equal(ch["gain"], [1.5, 2.0])


def test_description_labels_are_a_fallback(tmp_path):
    path = tmp_path / "desc.nwb"
    _hand_built(path, description="kinematics channel_labels=px, py")
    np.testing.assert_array_equal(_read_ch(path, "Kin", structured=False), ["px", "py"])
    np.testing.assert_array_equal(_read_ch(path, "Kin")["label"], ["px", "py"])


def test_undescribed_timeseries_has_no_ch_axis(tmp_path):
    path = tmp_path / "bare.nwb"
    _hand_built(path)
    slicer = NWBSlicer(path, dejitter=False, structured_ch_axis=True)
    try:
        assert "ch" not in slicer.get_stream_info("Kin").template.axes
    finally:
        slicer.close()
