"""Tests for NWBClockDrivenProducer."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.nwb.clockdriven import NWBClockDrivenProducer, NWBClockDrivenSettings
from ezmsg.nwb.util import ReferenceClockType


# --- Rate-only continuous stream ---


def test_rate_only_fixed_n_time(test_nwb_path):
    """Rate-only stream with fixed n_time produces correct chunk sizes."""
    settings = NWBClockDrivenSettings(
        fs=50.0,
        n_time=100,
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    clock_tick = AxisArray.LinearAxis(gain=2.0, offset=0.0)
    result = producer(clock_tick)

    assert result is not None
    assert result.data.shape[0] == 100
    assert result.key == "BinnedSpikes"


def test_rate_only_variable_chunk(test_nwb_path):
    """Rate-only stream derives chunk size from clock gain when n_time is None."""
    settings = NWBClockDrivenSettings(
        fs=50.0,
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    # 1 second at 50 Hz = 50 samples
    clock_tick = AxisArray.LinearAxis(gain=1.0, offset=0.0)
    result = producer(clock_tick)

    assert result is not None
    assert result.data.shape[0] == 50


def test_rate_only_auto_detect_fs(test_nwb_path):
    """Auto-detected fs (fs=0) uses the stream's nominal rate."""
    settings = NWBClockDrivenSettings(
        fs=0.0,  # auto-detect
        n_time=100,
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    clock_tick = AxisArray.LinearAxis(gain=1.0, offset=0.0)
    result = producer(clock_tick)

    assert result is not None
    assert result.data.shape[0] == 100
    assert producer._state.detected_fs == 50.0


def test_rate_only_exhaustion(test_nwb_path):
    """Rate-only stream reports exhaustion when all data consumed."""
    settings = NWBClockDrivenSettings(
        fs=50.0,
        n_time=150,  # Read everything at once
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)
    assert not producer.exhausted

    clock_tick = AxisArray.LinearAxis(gain=1.0, offset=0.0)
    result = producer(clock_tick)
    assert result is not None

    # After consuming all samples, should be exhausted
    assert producer.exhausted

    # Next call should return None
    result2 = producer(clock_tick)
    assert result2 is None


def test_rate_only_multiple_ticks(test_nwb_path):
    """Multiple clock ticks advance through the data correctly."""
    settings = NWBClockDrivenSettings(
        fs=50.0,
        n_time=25,
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    results = []
    for i in range(5):
        clock_tick = AxisArray.LinearAxis(gain=1.0, offset=float(i))
        result = producer(clock_tick)
        assert result is not None
        results.append(result)

    # Total samples should be 5 * 25 = 125
    total = sum(r.data.shape[0] for r in results)
    assert total == 125


def test_rate_only_afap_mode(test_nwb_path):
    """AFAP mode (gain=0) with fixed n_time works."""
    settings = NWBClockDrivenSettings(
        fs=50.0,
        n_time=100,
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    clock_tick = AxisArray.LinearAxis(gain=0.0, offset=0.0)
    result = producer(clock_tick)
    assert result is not None
    assert result.data.shape[0] == 100


# --- Timestamped continuous stream ---


def test_timestamped_continuous(test_nwb_path):
    """Timestamped continuous stream uses time-window extraction."""
    settings = NWBClockDrivenSettings(
        fs=0.0,  # auto-detect
        n_time=1000,  # ~1 second at 1000 Hz
        filepath=test_nwb_path,
        stream_key="Broadband",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    clock_tick = AxisArray.LinearAxis(gain=1.0, offset=0.0)
    result = producer(clock_tick)

    assert result is not None
    assert result.data.ndim == 2
    assert result.data.shape[1] == 8
    assert result.key == "Broadband"
    # Should get ~1000 samples for 1 second at 1000 Hz
    assert result.data.shape[0] >= 900


def test_timestamped_continuous_sequential_windows(test_nwb_path):
    """Sequential time windows don't overlap or skip data."""
    settings = NWBClockDrivenSettings(
        fs=0.0,
        filepath=test_nwb_path,
        stream_key="Broadband",
        reference_clock=ReferenceClockType.UNKNOWN,
        n_time=100,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    # Two adjacent 0.5-second windows
    r1 = producer(AxisArray.LinearAxis(gain=0.5, offset=0.0))
    r2 = producer(AxisArray.LinearAxis(gain=0.5, offset=0.5))

    assert r1 is not None
    assert r2 is not None

    # Combined should equal one 1-second window
    total = r1.data.shape[0] + r2.data.shape[0]
    r_full = NWBClockDrivenProducer(settings=settings)(AxisArray.LinearAxis(gain=1.0, offset=0.0))
    assert total == r_full.data.shape[0]


# --- Event/interval stream ---


def test_event_stream(test_nwb_path):
    """Event stream returns events within the time window."""
    settings = NWBClockDrivenSettings(
        fs=0.0,
        n_time=100,
        filepath=test_nwb_path,
        stream_key="trials",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    # First 5 seconds should have all 3 trials
    clock_tick = AxisArray.LinearAxis(gain=5.0, offset=0.0)
    result = producer(clock_tick)

    assert result is not None
    assert result.key == "trials"
    assert result.data.shape[0] == 3


def test_event_stream_empty_window(test_nwb_path):
    """Event stream returns empty result for window with no events."""
    settings = NWBClockDrivenSettings(
        fs=0.0,
        n_time=100,
        filepath=test_nwb_path,
        stream_key="trials",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    clock_tick = AxisArray.LinearAxis(gain=1.0, offset=99999.0)
    result = producer(clock_tick)

    assert result is not None
    assert result.data.shape[0] == 0


def test_event_stream_not_exhausted(test_nwb_path):
    """Event streams never report exhausted (time-window based)."""
    settings = NWBClockDrivenSettings(
        fs=0.0,
        n_time=100,
        filepath=test_nwb_path,
        stream_key="trials",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    clock_tick = AxisArray.LinearAxis(gain=1.0, offset=0.0)
    producer(clock_tick)
    assert not producer.exhausted


# --- Unit class ---


def test_unit_class_creation():
    """NWBClockDrivenUnit can be instantiated."""
    from ezmsg.nwb.clockdriven import NWBClockDrivenUnit

    unit = NWBClockDrivenUnit()
    assert hasattr(unit, "INPUT_CLOCK")
    assert hasattr(unit, "OUTPUT_SIGNAL")
