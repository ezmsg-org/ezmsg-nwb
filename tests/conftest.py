"""Shared fixtures for ezmsg-nwb tests."""

from pathlib import Path

import pytest
from create_test_nwb import create_test_nwb


@pytest.fixture(scope="session")
def test_nwb_path():
    """Generate-and-cache synthetic NWB test file."""
    path = Path(__file__).parent / "data" / "test_synthetic.nwb"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        create_test_nwb(path)
    return path
