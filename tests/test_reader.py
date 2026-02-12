"""Tests for NWB reader module."""

from ezmsg.nwb import NWBIteratorSettings, ReferenceClockType


def test_iterator_settings_defaults():
    """Test NWBIteratorSettings default values."""
    settings = NWBIteratorSettings(filepath="/tmp/test.nwb")
    assert settings.chunk_dur == 1.0
    assert settings.reference_clock == ReferenceClockType.SYSTEM
    assert settings.reref_now is False
    assert settings.self_terminating is True


def test_iterator_settings_custom():
    """Test NWBIteratorSettings with custom values."""
    settings = NWBIteratorSettings(
        filepath="/tmp/test.nwb",
        chunk_dur=0.5,
        reference_clock=ReferenceClockType.MONOTONIC,
        reref_now=True,
        self_terminating=False,
    )
    assert settings.chunk_dur == 0.5
    assert settings.reference_clock == ReferenceClockType.MONOTONIC
    assert settings.reref_now is True
    assert settings.self_terminating is False
