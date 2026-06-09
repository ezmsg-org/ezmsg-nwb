"""Shared fixtures for ezmsg-nwb tests."""

import datetime
from pathlib import Path

import numpy as np
import pytest
from create_test_nwb import create_test_nwb
from filelock import FileLock


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
        name="Gappy", data=_index_data(GAPPY_N_PRE + GAPPY_N_POST), unit="V",
        timestamps=gappy_timestamps(), description="gappy stream",
    )
    n_clean = 400
    clean = TimeSeries(
        name="Clean", data=_index_data(n_clean), unit="V",
        timestamps=np.arange(n_clean) * GAPPY_GAIN, description="contiguous stream",
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
