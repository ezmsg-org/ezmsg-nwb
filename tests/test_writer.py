"""Tests for NWB writer module."""

from ezmsg.nwb import NWBSinkSettings, ReferenceClockType


def test_sink_settings_defaults():
    """Test NWBSinkSettings default values."""
    settings = NWBSinkSettings(filepath="/tmp/test.nwb")
    assert settings.overwrite_old is False
    assert settings.axis == "time"
    assert settings.recording is True
    assert settings.inc_clock == ReferenceClockType.SYSTEM
    assert settings.meta_yaml is None
    assert settings.split_bytes == 0
    assert settings.expected_series is None


def test_sink_settings_custom():
    """Test NWBSinkSettings with custom values."""
    settings = NWBSinkSettings(
        filepath="/tmp/test.nwb",
        overwrite_old=True,
        axis="win",
        recording=False,
        inc_clock=ReferenceClockType.MONOTONIC,
        split_bytes=1024,
    )
    assert settings.overwrite_old is True
    assert settings.axis == "win"
    assert settings.recording is False
    assert settings.inc_clock == ReferenceClockType.MONOTONIC
    assert settings.split_bytes == 1024
