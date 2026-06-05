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


@pytest.fixture(scope="session")
def gappy_nwb_path(tmp_path_factory):
    """Build a minimal NWB file with one gappy timestamped stream."""
    import h5py
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries

    path = tmp_path_factory.mktemp("gappy") / "gappy.nwb"
    ts = gappy_timestamps()
    n = GAPPY_N_PRE + GAPPY_N_POST
    # Row i carries the constant value i across 3 channels.
    data = np.arange(n, dtype=np.float32)[:, None] + np.zeros((1, 3), dtype=np.float32)

    nwbfile = NWBFile(
        session_description="gappy",
        identifier="gappy001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    nwbfile.add_acquisition(
        TimeSeries(
            name="Gappy",
            data=data,
            unit="V",
            timestamps=ts,
            description="gappy timestamped stream",
        )
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    # Advertise a nominal rate so the slicer assigns a regular LinearAxis,
    # mirroring a recording with a known fs but dropped samples.
    with h5py.File(str(path), "a") as f:
        f["acquisition/Gappy/timestamps"].attrs["rate"] = GAPPY_RATE
    return path
