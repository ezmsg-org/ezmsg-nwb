"""Shared fixtures for ezmsg-nwb tests."""

from pathlib import Path

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
