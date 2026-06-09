"""Tests for NWBSlicer."""

import numpy as np
import pytest
from conftest import gappy_timestamps

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


@pytest.fixture
def gappy_slicer(gappy_nwb_path):
    s = NWBSlicer(
        filepath=gappy_nwb_path,
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["Gappy"],
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


def _build_manufacturer_prefixed_nwb(path):
    """Build a tiny NWB with an ElectricalSeries named ``CereLink_NPLAY``
    backed by a Device whose ``manufacturer`` attribute is ``"CereLink"``.

    Mirrors the layout Orion writes: storage container path prefixed by the
    manufacturer, with the manufacturer also stamped on the Device so
    downstream readers can reconstruct the bare device name.
    """
    import datetime

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject

    rng = np.random.default_rng(0)
    nwbfile = NWBFile(
        session_description="prefix-test",
        identifier="manufacturer-prefix-test",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
        subject=Subject(subject_id="sub1", age="P30Y", sex="U"),
    )
    device = nwbfile.create_device(name="CereLink_NPLAY", manufacturer="CereLink")
    group = nwbfile.create_electrode_group(
        name="CereLink_NPLAY", description="prefix-test group", location="cortex", device=device
    )
    nwbfile.add_electrode_column(name="label", description="Electrode label")
    for i in range(4):
        nwbfile.add_electrode(x=float(i), y=0.0, z=0.0, location="cortex", group=group, label=f"e{i}")
    region = nwbfile.create_electrode_table_region(region=list(range(4)), description="all")

    bb_n = 100
    es = ElectricalSeries(
        name="CereLink_NPLAY",
        data=rng.standard_normal((bb_n, 4)).astype(np.float32),
        starting_time=0.0,
        rate=1000.0,
        electrodes=region,
    )
    nwbfile.add_acquisition(es)

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)


def test_stream_keys_match_via_manufacturer_prefix(tmp_path):
    """stream_keys=['NPLAY'] matches a container 'CereLink_NPLAY' whose
    Device.manufacturer == 'CereLink'. The stream is exposed under the
    user-requested bare key, and messages carry key='NPLAY' so downstream
    fitters keyed by the request find their data."""
    nwb_path = tmp_path / "prefix.nwb"
    _build_manufacturer_prefixed_nwb(nwb_path)

    s = NWBSlicer(
        filepath=str(nwb_path),
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["NPLAY"],
    )
    try:
        assert s.stream_names == ["NPLAY"], f"expected the matched key to be the bare request, got {s.stream_names}"
        info = s.get_stream_info("NPLAY")
        assert info.template.key == "NPLAY"
        msg = s.read_by_index("NPLAY", 0, 10)
        assert msg.key == "NPLAY"
        assert msg.data.shape == (10, 4)
    finally:
        s.close()


def test_stream_keys_exact_match_wins_over_manufacturer(tmp_path):
    """Exact-match takes precedence: stream_keys=['CereLink_NPLAY'] yields
    the literal container name."""
    nwb_path = tmp_path / "prefix.nwb"
    _build_manufacturer_prefixed_nwb(nwb_path)

    s = NWBSlicer(
        filepath=str(nwb_path),
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["CereLink_NPLAY"],
    )
    try:
        assert s.stream_names == ["CereLink_NPLAY"]
        msg = s.read_by_index("CereLink_NPLAY", 0, 5)
        assert msg.key == "CereLink_NPLAY"
    finally:
        s.close()


def test_stream_keys_manufacturer_unknown_does_not_match(tmp_path):
    """A bare request like 'NPLAY' must NOT match 'CereLink_NPLAY' when the
    Device has no real manufacturer (unset or 'unknown'). Otherwise legacy
    files without manufacturer metadata could accidentally match unrelated
    streams that happen to share a suffix."""
    import datetime

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject

    rng = np.random.default_rng(0)
    path = tmp_path / "unknown_mfg.nwb"
    nwbfile = NWBFile(
        session_description="x",
        identifier="x",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
        subject=Subject(subject_id="s", age="P30Y", sex="U"),
    )
    # No manufacturer (or "unknown") on the device → suffix match must be
    # rejected.
    device = nwbfile.create_device(name="CereLink_NPLAY", manufacturer="unknown")
    group = nwbfile.create_electrode_group(name="CereLink_NPLAY", description="x", location="cortex", device=device)
    nwbfile.add_electrode_column(name="label", description="Electrode label")
    for i in range(2):
        nwbfile.add_electrode(x=0.0, y=0.0, z=0.0, location="cortex", group=group, label=f"e{i}")
    region = nwbfile.create_electrode_table_region(region=list(range(2)), description="all")
    es = ElectricalSeries(
        name="CereLink_NPLAY",
        data=rng.standard_normal((50, 2)).astype(np.float32),
        starting_time=0.0,
        rate=500.0,
        electrodes=region,
    )
    nwbfile.add_acquisition(es)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    s = NWBSlicer(
        filepath=str(path),
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["NPLAY"],
    )
    try:
        assert s.stream_names == [], f"unknown manufacturer should not enable suffix match, got {s.stream_names}"
    finally:
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


def test_read_by_index_offset_includes_starting_time(tmp_path):
    """For a rate-only stream with non-zero ``starting_time``, the emitted time
    offset must include that start time (sample indices are 0-based from the
    stream start). Regression: previously the offset was ``ts_off + gain*idx``,
    omitting ``t0``, which mis-timed late-starting streams.
    """
    import datetime

    from pynwb import NWBHDF5IO, NWBFile, TimeSeries

    path = tmp_path / "late_start.nwb"
    nwb = NWBFile(
        session_description="m",
        identifier="m",
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )
    rate = 100.0
    nwb.add_acquisition(
        TimeSeries(
            name="Late",
            data=np.arange(int(3 * rate), dtype=np.float32)[:, None],
            unit="V",
            rate=rate,
            starting_time=2.0,
        )
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)

    s = NWBSlicer(filepath=path, reference_clock=ReferenceClockType.UNKNOWN)
    try:
        gain = 1.0 / rate
        # Sample 0 is at the stream's start time (2.0 s); sample 50 is 0.5 s later.
        msg0 = s.read_by_index("Late", 0, 10)
        msg50 = s.read_by_index("Late", 50, 60)
        assert msg0.axes["time"].offset == pytest.approx(2.0)
        assert msg50.axes["time"].offset == pytest.approx(2.0 + 50 * gain)
    finally:
        s.close()


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


# --- Timestamp gaps (Solution B: CoordinateAxis on gappy windows) ---------


def test_read_by_time_clean_window_linear_axis(gappy_slicer):
    """A gap-free window keeps the cheap regular LinearAxis representation."""
    msg = gappy_slicer.read_by_time("Gappy", 0.0, 1.0)
    assert msg.data.shape[0] == 100  # samples 0..99 @ 100 Hz
    assert hasattr(msg.axes["time"], "gain")  # LinearAxis
    assert not hasattr(msg.axes["time"], "data")
    assert msg.axes["time"].offset == pytest.approx(0.0, abs=1e-6)


def test_read_by_time_gap_spanning_window_coordinate_axis(gappy_slicer):
    """A window that straddles the gap must emit a CoordinateAxis carrying the
    true per-sample timestamps rather than a uniform LinearAxis that would
    misplace every post-gap sample.
    """
    msg = gappy_slicer.read_by_time("Gappy", 1.0, 3.0)
    # 50 pre-gap (t 1.00..1.49) + 50 post-gap (t 2.50..2.99).
    assert msg.data.shape[0] == 100
    assert hasattr(msg.axes["time"], "data")  # CoordinateAxis
    assert not hasattr(msg.axes["time"], "gain")
    ts = np.asarray(msg.axes["time"].data)
    assert ts.shape[0] == 100
    # The jump across the gap is preserved, not flattened.
    assert ts[49] == pytest.approx(1.49, abs=1e-6)
    assert ts[50] == pytest.approx(2.50, abs=1e-6)


def test_read_by_time_coordinate_timestamps_match_true(gappy_slicer):
    """CoordinateAxis timestamps equal the file's true timestamps per sample."""
    msg = gappy_slicer.read_by_time("Gappy", 1.0, 3.0)
    true_ts = gappy_timestamps()
    idx = msg.data[:, 0].astype(int)  # data value == global sample index
    np.testing.assert_allclose(np.asarray(msg.axes["time"].data), true_ts[idx], atol=1e-9)


def test_read_by_time_window_inside_gap_empty(gappy_slicer):
    """A window entirely inside the gap returns a zero-length result."""
    msg = gappy_slicer.read_by_time("Gappy", 1.7, 2.3)
    assert msg.data.shape[0] == 0


def test_read_by_time_gap_tol_disables_coordinate_axis(gappy_slicer):
    """A large gap_tol widens the threshold so even a gap-spanning window stays
    a single LinearAxis chunk — the knob works (and loses no samples)."""
    msg = gappy_slicer.read_by_time("Gappy", 1.0, 3.0, gap_tol=1e6)
    assert hasattr(msg.axes["time"], "gain")  # LinearAxis
    assert msg.data.shape[0] == 100


# --- Lifecycle ---


def test_close_idempotent(slicer):
    """Calling close multiple times is safe."""
    slicer.close()
    slicer.close()  # Should not raise
