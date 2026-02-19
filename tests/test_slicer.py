"""Tests for NWBSlicer."""

import numpy as np
import pytest

from ezmsg.nwb.slicer import NWBSlicer
from ezmsg.nwb.util import ReferenceClockType


@pytest.fixture
def slicer(test_nwb_path):
    s = NWBSlicer(
        filepath=test_nwb_path,
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    yield s
    s.close()


# --- Stream discovery ---


def test_stream_discovery(slicer):
    """Slicer discovers all 6 streams in the test file."""
    names = set(slicer.stream_names)
    assert "Broadband" in names
    assert "RawAnalog" in names
    assert "BinnedSpikes" in names
    assert "Force" in names
    assert "trials" in names
    assert "phonemes" in names


def test_stream_discovery_filter(test_nwb_path):
    """stream_keys filter limits discovered streams."""
    s = NWBSlicer(
        filepath=test_nwb_path,
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["Broadband", "trials"],
    )
    assert set(s.stream_names) == {"Broadband", "trials"}
    s.close()


def test_stream_info_continuous(slicer):
    """Continuous timestamped stream metadata is correct."""
    info = slicer.get_stream_info("Broadband")
    assert info.fs == 1000.0
    assert info.n_samples == 3000
    assert info.has_timestamps is True
    assert info.is_event is False
    assert info.template.data.ndim == 2
    assert info.template.data.shape[1] == 8


def test_stream_info_rate_only(slicer):
    """Rate-only stream (no explicit timestamps) metadata is correct."""
    info = slicer.get_stream_info("BinnedSpikes")
    assert info.fs == 50.0
    assert info.n_samples == 150
    assert info.has_timestamps is False
    assert info.is_event is False


def test_stream_info_rate_only_2ch(slicer):
    """Rate-only 2-channel stream metadata is correct."""
    info = slicer.get_stream_info("RawAnalog")
    assert info.fs == 500.0
    assert info.n_samples == 1500
    assert info.has_timestamps is False
    assert info.is_event is False
    assert info.template.data.shape[1] == 2


def test_stream_info_1d(slicer):
    """1D timeseries metadata is correct."""
    info = slicer.get_stream_info("Force")
    assert info.fs == 100.0
    assert info.n_samples == 300
    assert info.has_timestamps is False
    assert info.is_event is False
    assert info.template.dims == ["time"]


def test_stream_info_event(slicer):
    """Event/interval stream metadata is correct."""
    info = slicer.get_stream_info("trials")
    assert info.is_event is True
    assert info.has_timestamps is True
    assert info.n_samples == 3
    assert info.fs == 0.0

    info_ph = slicer.get_stream_info("phonemes")
    assert info_ph.is_event is True
    assert info_ph.n_samples == 10


def test_stream_info_electrodes(slicer):
    """Broadband stream has electrode labels."""
    info = slicer.get_stream_info("Broadband")
    ch_axis = info.template.axes["ch"]
    labels = list(ch_axis.data)
    assert labels == [f"elec{i}" for i in range(8)]


def test_ts_off_unknown(slicer):
    """ts_off is 0 with UNKNOWN reference clock."""
    assert slicer.ts_off == 0.0


def test_start_stop_time(slicer):
    """Global start/stop time are computed."""
    assert slicer.start_time < slicer.stop_time
    assert slicer.start_time == 0.0


# --- Continuous slicing ---


def test_read_by_index_basic(slicer):
    """read_by_index returns correct data shape and key."""
    msg = slicer.read_by_index("BinnedSpikes", 0, 100)
    assert msg.data.shape[0] == 100
    assert msg.key == "BinnedSpikes"


def test_read_by_index_offset(slicer):
    """read_by_index respects start index."""
    msg1 = slicer.read_by_index("BinnedSpikes", 0, 50)
    msg2 = slicer.read_by_index("BinnedSpikes", 50, 100)
    msg_full = slicer.read_by_index("BinnedSpikes", 0, 100)
    np.testing.assert_array_equal(
        np.concatenate([msg1.data, msg2.data], axis=0),
        msg_full.data,
    )


def test_read_by_index_has_linear_axis(slicer):
    """read_by_index on rate-only stream produces LinearAxis time axis."""
    msg = slicer.read_by_index("BinnedSpikes", 0, 10)
    assert hasattr(msg.axes["time"], "gain")  # LinearAxis


# --- Timestamped continuous slicing ---


def test_read_by_time_continuous(slicer):
    """read_by_time on timestamped continuous stream returns data."""
    info = slicer.get_stream_info("Broadband")
    t_start = info.t0
    t_end = t_start + 0.01
    msg = slicer.read_by_time("Broadband", t_start, t_end)
    assert msg.data.shape[0] > 0
    assert msg.data.shape[1] == 8
    assert msg.key == "Broadband"


# --- Event slicing ---


def test_read_by_time_events(slicer):
    """read_by_time on event stream returns events in the window."""
    msg = slicer.read_by_time("trials", 0.0, 60.0)
    assert msg.data.ndim == 2
    assert msg.key == "trials"
    assert msg.data.shape[0] == 3


def test_read_by_time_events_empty_window(slicer):
    """read_by_time on event stream with no events returns zero-length template."""
    msg = slicer.read_by_time("trials", 99999.0, 100000.0)
    assert msg.data.shape[0] == 0


# --- Lifecycle ---


def test_close_idempotent(slicer):
    """Calling close multiple times is safe."""
    slicer.close()
    slicer.close()  # Should not raise
