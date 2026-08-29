"""Shared fixtures for ezmsg-nwb tests."""

import datetime
from pathlib import Path

import numpy as np
import pytest
from create_test_nwb import create_test_nwb
from filelock import FileLock
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.ecephys import ElectricalSeries


@pytest.fixture(scope="session")
def test_nwb_path():
    """Generate-and-cache synthetic NWB test file.

    Under pytest-xdist, ``scope="session"`` is per-worker, so each worker
    would independently race to create the same file and collide on the
    HDF5 write lock. The filelock serializes the check-and-create across
    workers; the first builds it, the rest skip.
    """
    path = Path(__file__).parent / "data" / "test_synthetic.nwb"
    path.parent.mkdir(exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            create_test_nwb(path)
    return path


# --- Gappy-stream fixture ---------------------------------------------------
#
# A timestamped continuous stream whose explicit timestamps contain a single
# large gap, while still advertising a nominal ``rate`` (so the slicer picks a
# regular ``LinearAxis`` for it). This is the dangerous case for both Solution
# A (iterator) and Solution B (slicer / clock-driven): 100 Hz, 150 samples
# before a 1.0 s gap and 150 after. The per-sample data value equals its global
# sample index so callers can recover where each emitted sample came from.

GAPPY_RATE = 100.0
GAPPY_GAIN = 1.0 / GAPPY_RATE
GAPPY_N_PRE = 150
GAPPY_N_POST = 150
GAPPY_GAP = 1.0


def gappy_timestamps() -> np.ndarray:
    """True per-sample timestamps of the gappy stream (file-relative)."""
    pre = np.arange(GAPPY_N_PRE) * GAPPY_GAIN  # 0.00 .. 1.49
    post = pre[-1] + GAPPY_GAIN + GAPPY_GAP + np.arange(GAPPY_N_POST) * GAPPY_GAIN  # 2.50 .. 3.99
    return np.concatenate([pre, post])


def _index_data(n: int, n_ch: int = 3) -> np.ndarray:
    """Data whose row ``i`` carries the constant value ``i`` across channels, so
    a caller can recover each emitted sample's global index from its value."""
    return np.arange(n, dtype=np.float32)[:, None] + np.zeros((1, n_ch), dtype=np.float32)


def _write_nwb(path, series, rate_attrs):
    """Write *series* (list of TimeSeries) to *path*; stamp ``rate`` attrs after
    so the slicer treats those streams as regular (LinearAxis)."""
    import h5py
    from pynwb import NWBHDF5IO, NWBFile

    nwbfile = NWBFile(
        session_description="synthetic",
        identifier="synthetic001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    for s in series:
        nwbfile.add_acquisition(s)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with h5py.File(str(path), "a") as f:
        for name, rate in rate_attrs.items():
            f[f"acquisition/{name}/timestamps"].attrs["rate"] = rate
    return path


@pytest.fixture(scope="session")
def gappy_nwb_path(tmp_path_factory):
    """Build a minimal NWB file with one gappy timestamped stream."""
    from pynwb import TimeSeries

    path = tmp_path_factory.mktemp("gappy") / "gappy.nwb"
    n = GAPPY_N_PRE + GAPPY_N_POST
    series = TimeSeries(
        name="Gappy",
        data=_index_data(n),
        unit="V",
        timestamps=gappy_timestamps(),
        description="gappy timestamped stream",
    )
    # The rate attr makes the slicer assign a regular LinearAxis, mirroring a
    # recording with a known fs but dropped samples.
    return _write_nwb(path, [series], {"Gappy": GAPPY_RATE})


@pytest.fixture(scope="session")
def gappy_and_clean_nwb_path(tmp_path_factory):
    """Two regular (rate-stamped) containers: ``Gappy`` (has the 1.0 s gap) and
    ``Clean`` (400 contiguous samples, no gap), both 100 Hz spanning 0..3.99 s.

    Mirrors the CA001 layout — multiple acquisition containers in one file — so
    a test can drive one clock-driven producer per stream and confirm a gap in
    one container does not desync the other.
    """
    from pynwb import TimeSeries

    path = tmp_path_factory.mktemp("gappy_clean") / "gappy_clean.nwb"
    gappy = TimeSeries(
        name="Gappy",
        data=_index_data(GAPPY_N_PRE + GAPPY_N_POST),
        unit="V",
        timestamps=gappy_timestamps(),
        description="gappy stream",
    )
    n_clean = 400
    clean = TimeSeries(
        name="Clean",
        data=_index_data(n_clean),
        unit="V",
        timestamps=np.arange(n_clean) * GAPPY_GAIN,
        description="contiguous stream",
    )
    return _write_nwb(path, [gappy, clean], {"Gappy": GAPPY_RATE, "Clean": GAPPY_RATE})


# --- Irregular-stream fixture -----------------------------------------------
#
# A timestamped continuous stream with NO rate attr and high inter-sample
# variance, so the slicer's rate heuristic gives up (rate=0) and assigns a
# CoordinateAxis template (no ``gain``). Exercises the ``not has_gain`` branch
# of ``read_by_time``, which must still emit true per-sample timestamps.

IRREGULAR_N = 200


def irregular_timestamps() -> np.ndarray:
    """Monotonic, highly-irregular per-sample timestamps (fixed seed)."""
    rng = np.random.default_rng(7)
    intervals = rng.uniform(0.01, 0.5, size=IRREGULAR_N)
    return np.cumsum(intervals)


# --- ASCII-string fixture ---------------------------------------------------
#
# A file whose strings were declared ``H5T_CSET_ASCII`` rather than UTF-8, the
# way aqnwb writes them and so the way every recording Orion produces reads
# back: hdmf keys its decoding decision off the character set, so these arrive
# as ``bytes`` where a pynwb-written file gives ``str``. Everything else in
# this suite is written by pynwb, so without this fixture no test can tell the
# two apart -- which is exactly how the bug reached hardware.

ASCII_MARKERS = [
    '{"cause": {"event": "trial-start"}}',
    '{"cause": {"event": "go-cue"}}',
    '{"cause": {"event": "trial-stop"}}',
]
ASCII_MARKER_TIMES = [0.5, 1.0, 1.5]
ASCII_MANUFACTURER = "CereLink"
ASCII_N_CHANNELS = 4
ASCII_RATE = 100.0
ASCII_N_SAMPLES = 200


def _as_fixed_length_ascii(dataset_parent, name: str) -> None:
    """Restate one text dataset as fixed-length ASCII, attributes intact."""
    attributes = dict(dataset_parent[name].attrs)
    values = [v.encode("utf-8") if isinstance(v, str) else v for v in dataset_parent[name][:]]
    del dataset_parent[name]
    column = dataset_parent.create_dataset(name, data=np.array(values, dtype="S64"))
    column.attrs.update(attributes)


@pytest.fixture(scope="session")
def ascii_nwb_path(tmp_path_factory):
    """An NWB file whose every string reads back as ``bytes``.

    Carries one of each kind of text a reader has to survive: spec-defined and
    custom electrodes columns, a custom interval-table column, a marker series,
    and the Device ``manufacturer`` attribute that stream-key matching parses.
    """
    import h5py
    from create_test_nwb import _downgrade_electrodes_table
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.misc import AnnotationSeries

    path = tmp_path_factory.mktemp("ascii") / "ascii_strings.nwb"
    nwbfile = NWBFile(
        session_description="ASCII-declared strings",
        identifier="ascii001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    device = nwbfile.create_device(name="TestArray", manufacturer=ASCII_MANUFACTURER)
    group = nwbfile.create_electrode_group(
        name="TestGroup", description="test electrodes", location="M1", device=device
    )
    nwbfile.add_electrode_column(name="label", description="CMP electrode label")
    for i in range(ASCII_N_CHANNELS):
        nwbfile.add_electrode(location="M1", group=group, label=f"elec1-m1-{i}")
    region = nwbfile.create_electrode_table_region(region=list(range(ASCII_N_CHANNELS)), description="all")

    # Named "<manufacturer>_<key>" so a stream_keys=["NPLAY"] filter has to read
    # the Device's manufacturer to match it.
    nwbfile.add_acquisition(
        ElectricalSeries(
            name=f"{ASCII_MANUFACTURER}_NPLAY",
            data=np.zeros((ASCII_N_SAMPLES, ASCII_N_CHANNELS), dtype=np.float32),
            electrodes=region,
            rate=ASCII_RATE,
            starting_time=0.0,
            description="synthetic broadband",
        )
    )
    nwbfile.add_acquisition(
        AnnotationSeries(name="Markers", data=list(ASCII_MARKERS), timestamps=list(ASCII_MARKER_TIMES))
    )
    nwbfile.add_trial_column(name="condition", description="trial condition label")
    for i in range(3):
        nwbfile.add_trial(start_time=float(i), stop_time=float(i) + 0.8, condition=f"cond_{i % 2}")

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    with h5py.File(str(path), "a") as f:
        electrodes = f["general/extracellular_ephys/electrodes"]
        _as_fixed_length_ascii(electrodes, "location")
        _as_fixed_length_ascii(electrodes, "label")
        _as_fixed_length_ascii(f["intervals/trials"], "condition")
        _as_fixed_length_ascii(f["acquisition/Markers"], "data")
        # An attribute, not a dataset: h5py hands ASCII attrs back as bytes too.
        f["general/devices/TestArray"].attrs["manufacturer"] = np.bytes_(ASCII_MANUFACTURER.encode())

    _downgrade_electrodes_table(path)
    return path


# --- Dejitter fixture -------------------------------------------------------
#
# A jittery acquisition stream paired with a ``*_device_ts`` sibling, mirroring
# the real Orion/CereLink layout: the same data is stored twice, once in
# ``/acquisition`` with converted (session-relative) timestamps and once in
# ``/processing/ecephys`` with device-clock (absolute-epoch) timestamps, the two
# ``data`` datasets hard-linked to one object. The converted timestamps carry
# per-sample jitter large enough to fragment the stream under the gap-splitter,
# so the slicer's dejitter pass has something to fix. The ``rate`` attr is stored
# float32 and the device timestamps are epoch-scale on purpose -- together they
# reproduce the float32/epoch bound blow-up the recompute guards against.

DEJITTER_RATE = 1000.0
DEJITTER_N = 3000
DEJITTER_EPOCH = 1704110400.0  # == session_start 2024-01-01 12:00:00 UTC


def dejitter_timestamps():
    """``(truth, converted, device)`` for the dejitter fixture.

    ``truth`` is the clean session-relative time (gentle non-linear drift);
    ``converted`` is ``truth`` plus per-sample jitter (~0.6 sample periods);
    ``device`` is the shared epoch clock carrying the same jitter.
    """
    idx = np.arange(DEJITTER_N)
    truth = idx / DEJITTER_RATE + 0.003 * np.sin(2 * np.pi * idx / DEJITTER_N)
    jitter = np.random.default_rng(11).normal(scale=0.6 / DEJITTER_RATE, size=DEJITTER_N)
    converted = truth + jitter
    device = DEJITTER_EPOCH + idx / DEJITTER_RATE + jitter
    return truth, converted, device


DEJITTER_GAP_AT = 1500  # left-edge index of the injected real gap
DEJITTER_GAP_S = 0.3  # real gap duration (dropped ~300 samples of data)


def dejitter_gapped_timestamps():
    """Like :func:`dejitter_timestamps` but with a genuine data gap: every sample
    from ``DEJITTER_GAP_AT`` on is shifted later by ``DEJITTER_GAP_S`` on both the
    converted and (shared) device clocks."""
    truth, converted, device = dejitter_timestamps()
    truth = truth.copy()
    converted = converted.copy()
    device = device.copy()
    for arr in (truth, converted, device):
        arr[DEJITTER_GAP_AT:] += DEJITTER_GAP_S
    return truth, converted, device


def _write_dejitter_nwb(path, converted, device, data):
    import h5py
    from create_test_nwb import _downgrade_electrodes_table
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.ecephys import ElectricalSeries

    nwbfile = NWBFile(
        session_description="dejitter",
        identifier="dejitter001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    dev = nwbfile.create_device(name="TestArray")
    group = nwbfile.create_electrode_group(name="G", description="g", location="M1", device=dev)
    nwbfile.add_electrode_column(name="label", description="label")
    for i in range(4):
        nwbfile.add_electrode(location="M1", group=group, label=f"e{i}")
    region = nwbfile.create_electrode_table_region(region=list(range(4)), description="all")

    nwbfile.add_acquisition(
        ElectricalSeries(name="HUB", data=data, timestamps=converted, electrodes=region, description="jittery")
    )
    ecephys = nwbfile.create_processing_module(name="ecephys", description="ecephys")
    ecephys.add(TimeSeries(name="HUB_device_ts", data=data.copy(), unit="V", timestamps=device, description="device"))

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    with h5py.File(str(path), "a") as f:
        f["acquisition/HUB/timestamps"].attrs["rate"] = np.float32(DEJITTER_RATE)
        del f["processing/ecephys/HUB_device_ts/data"]
        f["processing/ecephys/HUB_device_ts/data"] = f["acquisition/HUB/data"]

    _downgrade_electrodes_table(path)
    return path


@pytest.fixture(scope="session")
def dejitter_gapped_nwb_path(tmp_path_factory):
    """Dejitter fixture carrying one genuine data gap (see dejitter_gapped_timestamps)."""
    path = tmp_path_factory.mktemp("dejitter_gap") / "dejitter_gap.nwb"
    _truth, converted, device = dejitter_gapped_timestamps()
    return _write_dejitter_nwb(path, converted, device, _index_data(DEJITTER_N, 4))


@pytest.fixture(scope="session")
def dejitter_nwb_path(tmp_path_factory):
    """NWB file with one jittery acquisition stream + hard-linked device_ts partner."""
    import h5py
    from create_test_nwb import _downgrade_electrodes_table
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.ecephys import ElectricalSeries

    path = tmp_path_factory.mktemp("dejitter") / "dejitter.nwb"
    _truth, converted, device = dejitter_timestamps()
    data = _index_data(DEJITTER_N, 4)

    nwbfile = NWBFile(
        session_description="dejitter",
        identifier="dejitter001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    dev = nwbfile.create_device(name="TestArray")
    group = nwbfile.create_electrode_group(name="G", description="g", location="M1", device=dev)
    nwbfile.add_electrode_column(name="label", description="label")
    for i in range(4):
        nwbfile.add_electrode(location="M1", group=group, label=f"e{i}")
    region = nwbfile.create_electrode_table_region(region=list(range(4)), description="all")

    nwbfile.add_acquisition(
        ElectricalSeries(name="HUB", data=data, timestamps=converted, electrodes=region, description="jittery")
    )
    ecephys = nwbfile.create_processing_module(name="ecephys", description="ecephys")
    ecephys.add(TimeSeries(name="HUB_device_ts", data=data.copy(), unit="V", timestamps=device, description="device"))

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    with h5py.File(str(path), "a") as f:
        # float32 rate attr, like the real recordings (drives the float32 path).
        f["acquisition/HUB/timestamps"].attrs["rate"] = np.float32(DEJITTER_RATE)
        # Hard-link the device_ts data to the acquisition data so the slicer's
        # object-identity pairing fires (the structural mark of a re-timestamped
        # acquisition), not merely the name convention.
        del f["processing/ecephys/HUB_device_ts/data"]
        f["processing/ecephys/HUB_device_ts/data"] = f["acquisition/HUB/data"]

    _downgrade_electrodes_table(path)
    return path


@pytest.fixture(scope="session")
def irregular_nwb_path(tmp_path_factory):
    """Build an NWB file with one irregular (rate-0 → CoordinateAxis) stream."""
    from pynwb import TimeSeries

    path = tmp_path_factory.mktemp("irregular") / "irregular.nwb"
    series = TimeSeries(
        name="Irregular",
        data=_index_data(IRREGULAR_N),
        unit="V",
        timestamps=irregular_timestamps(),
        description="irregular timestamped stream (no rate)",
    )
    # No rate attr -> slicer detects rate 0 -> CoordinateAxis template.
    return _write_nwb(path, [series], {})


# --- Quantised-timestamp fixture --------------------------------------------
#
# A regular stream whose per-sample stamps are rounded to a grid coarser than
# the true sample period is fine-grained, which is how real acquisition writes
# them: the intervals then take only two values, straddling the truth. Any
# median-based rate estimate snaps to one of those two and stays there for every
# sample count, so this fixture is the one that catches the bias. No ``rate``
# attr, so the slicer has to estimate.

QUANTISED_N = 20_000
QUANTISED_RATE = 30000.104
"""Delivered rate. Deliberately not a round number and not on the stamp grid."""
QUANTISED_STEP = 4e-8
"""Timestamp quantum. Matches recordings from CereLink-class hardware, where the
median interval reads 30012.005 Hz against this 30000.104 Hz stream."""


def quantised_timestamps() -> np.ndarray:
    """Regular timestamps at ``QUANTISED_RATE``, rounded onto a ``QUANTISED_STEP`` grid."""
    return np.round(np.arange(QUANTISED_N) / QUANTISED_RATE / QUANTISED_STEP) * QUANTISED_STEP


@pytest.fixture(scope="session")
def quantised_nwb_path(tmp_path_factory):
    """NWB file with one regular stream carrying coarsely-quantised timestamps."""
    from pynwb import TimeSeries

    path = tmp_path_factory.mktemp("quantised") / "quantised.nwb"
    series = TimeSeries(
        name="Quantised",
        data=_index_data(QUANTISED_N),
        unit="V",
        timestamps=quantised_timestamps(),
        description="regular stream, quantised timestamps, no rate attr",
    )
    return _write_nwb(path, [series], {})


# --- Scaling: a file whose conversion factors are all non-trivial ------------
#
# All three factors are deliberately non-trivial -- a non-unit ``conversion``, a
# non-zero ``offset``, and a ``channel_conversion`` whose entries differ --
# because each is separately easy to drop, and a file where any of them is
# 1.0/0.0 cannot tell you whether it was applied.

N_SAMPLES = 500
N_CH = 4
RATE = 500.0

CONVERSION = 0.25
OFFSET = 3.5
CHANNEL_CONVERSION = np.array([1.0, 2.0, 0.5, 4.0], dtype=np.float32)
DECLARED_UNIT = "microvolts"

# Marker stream: conversion 1.0 / offset 0.0, i.e. the values are already in the
# unit declared. Present to pin that such a stream is left alone rather than
# copied into float -- see ``is_identity_scaling``.
MARKER_N = 50


@pytest.fixture(scope="module")
def scaled_nwb_path(tmp_path_factory):
    """An NWB whose broadband declares all three scaling factors non-trivially."""
    path = tmp_path_factory.mktemp("scaling") / "scaled.nwb"
    rng = np.random.default_rng(7)

    nwbfile = NWBFile(
        session_description="scaling fixture",
        identifier="scaling001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    device = nwbfile.create_device(name="Array", description="fixture")
    group = nwbfile.create_electrode_group(name="Group", description="fixture", location="cortex", device=device)
    nwbfile.add_electrode_column(name="label", description="Electrode label")
    for i in range(N_CH):
        nwbfile.add_electrode(x=float(i), y=0.0, z=0.0, location="cortex", group=group, label=f"elec{i}")
    region = nwbfile.create_electrode_table_region(region=list(range(N_CH)), description="all")

    # int16 spanning a good part of the range, so a dropped conversion shows up
    # as a large error rather than a rounding one.
    counts = rng.integers(-30000, 30000, size=(N_SAMPLES, N_CH), dtype=np.int16)
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="Broadband",
            data=counts,
            rate=RATE,
            starting_time=0.0,
            electrodes=region,
            conversion=CONVERSION,
            offset=OFFSET,
            channel_conversion=CHANNEL_CONVERSION,
            description="fixture broadband",
        )
    )
    nwbfile.add_acquisition(
        TimeSeries(
            name="Marker",
            data=np.arange(MARKER_N, dtype=np.int16),
            unit="n/a",
            rate=RATE,
            starting_time=0.0,
            description="already in its declared unit",
        )
    )
    # Voltage that is *not* an ElectricalSeries. Writers do this routinely -- a
    # re-timestamped companion of an acquisition stream, pointing at the same
    # samples, comes out as a plain TimeSeries -- and its declared unit is then
    # the only evidence that it is voltage at all.
    nwbfile.add_acquisition(
        TimeSeries(
            name="AuxVoltage",
            data=counts,
            unit="volts",
            conversion=CONVERSION,
            offset=OFFSET,
            rate=RATE,
            starting_time=0.0,
            description="voltage written as a plain TimeSeries",
        )
    )
    # Non-voltage with a real (non-identity) scaling, so "left alone" is a
    # decision about its unit and not a side effect of having nothing to apply.
    nwbfile.add_acquisition(
        TimeSeries(
            name="Cursor",
            data=np.arange(MARKER_N, dtype=np.int16),
            unit="pixels",
            conversion=CONVERSION,
            rate=RATE,
            starting_time=0.0,
            description="not a voltage at all",
        )
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    # pynwb pins ElectricalSeries.unit to the schema's "volts" and will not write
    # another value, so stamp the declared unit the way a real acquisition writer
    # does: straight onto the data dataset's attribute.
    import h5py

    with h5py.File(str(path), "a") as f:
        f["acquisition/Broadband/data"].attrs["unit"] = DECLARED_UNIT
    return path


@pytest.fixture(scope="module")
def counts(scaled_nwb_path) -> np.ndarray:
    import h5py

    with h5py.File(str(scaled_nwb_path), "r") as f:
        return np.asarray(f["acquisition/Broadband/data"][:])


def expected(counts: np.ndarray) -> np.ndarray:
    """``data * conversion * channel_conversion + offset``, the NWB definition."""
    return counts.astype(np.float32) * np.float32(CONVERSION) * CHANNEL_CONVERSION.astype(np.float32) + np.float32(
        OFFSET
    )


# --- An electrodes table shaped like an acquisition stack's ------------------
#
# Two connectors on one headstage plus one on a second, so the array-identity
# derivation has something to separate and the bank letters span more than one
# value. Geometry is a 400 um grid, matching real Utah-array pitch, and the
# columns are named the way the CereLink writer names them (rel_x/rel_y/term).

ELEC_N_CH = 8
ELEC_PITCH = 400


@pytest.fixture(scope="session")
def mapped_nwb_path(tmp_path_factory):
    """An NWB whose electrodes table carries channel-map columns."""
    path = tmp_path_factory.mktemp("electrodes") / "mapped.nwb"
    nwbfile = NWBFile(
        session_description="channel map fixture",
        identifier="map001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    device = nwbfile.create_device(name="Hub", description="fixture")
    for col, desc in (
        ("label", "electrode label"),
        ("rel_x", "x within array, um"),
        ("rel_y", "y within array, um"),
        ("size", "electrode size, um"),
        ("bank", "1-based connector bank"),
        ("term", "1-based terminal within bank"),
        ("headstage", "1-based headstage id"),
    ):
        nwbfile.add_electrode_column(name=col, description=desc)

    # (connector, region, headstage) per channel: two arrays on hs1, one on hs2.
    layout = [("elec1", "m1", 1)] * 4 + [("elec2", "dlpfc", 1)] * 2 + [("elec1", "s1l", 2)] * 2
    groups = {}
    for i, (connector, region, hs) in enumerate(layout):
        gname = f"Hub-hs{hs}-{connector}-{region}"
        if gname not in groups:
            groups[gname] = nwbfile.create_electrode_group(
                name=gname, description="fixture", location=region, device=device
            )
        nwbfile.add_electrode(
            location=region,
            group=groups[gname],
            label=f"{connector}-{region}-{i + 1}",
            rel_x=float((i % 2) * ELEC_PITCH),
            rel_y=float((i // 2) * ELEC_PITCH),
            size=float(ELEC_PITCH),
            bank=i // 4 + 1,  # 1,1,1,1,2,2,3,3 -> letters A,A,A,A,B,B,C,C
            term=i % 4 + 1,
            headstage=hs,
        )
    region = nwbfile.create_electrode_table_region(region=list(range(ELEC_N_CH)), description="all")
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="Mapped",
            data=np.zeros((20, ELEC_N_CH), dtype=np.int16),
            rate=100.0,
            starting_time=0.0,
            electrodes=region,
            description="fixture",
        )
    )
    # A second series over a strict subset, in a different order, so the region
    # handling is exercised rather than assumed.
    subset = nwbfile.create_electrode_table_region(region=[5, 1, 0], description="subset")
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="Subset",
            data=np.zeros((20, 3), dtype=np.int16),
            rate=100.0,
            starting_time=0.0,
            electrodes=subset,
            description="fixture subset",
        )
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return path
