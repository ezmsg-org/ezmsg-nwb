"""Tests for NWBClockDrivenProducer."""

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
    """Event stream returns empty result for window with no events.

    Starts playback past the end of the test file via ``start_offset``;
    with the new design the clock-tick offset is not a seek knob.
    """
    settings = NWBClockDrivenSettings(
        fs=0.0,
        n_time=100,
        filepath=test_nwb_path,
        stream_key="trials",
        reference_clock=ReferenceClockType.UNKNOWN,
        start_offset=99999.0,
    )
    producer = NWBClockDrivenProducer(settings=settings)

    clock_tick = AxisArray.LinearAxis(gain=1.0, offset=0.0)
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


# --- start_offset / playback_rate ---


def test_start_offset_rate_only(test_nwb_path):
    """start_offset shifts the initial sample index for rate-only streams."""
    settings = NWBClockDrivenSettings(
        fs=50.0,
        n_time=25,
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
        start_offset=1.0,  # 1s * 50 Hz = 50 samples in
    )
    producer = NWBClockDrivenProducer(settings=settings)

    full = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=50.0,
            n_time=150,
            filepath=test_nwb_path,
            stream_key="BinnedSpikes",
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )(AxisArray.LinearAxis(gain=1.0, offset=0.0))

    result = producer(AxisArray.LinearAxis(gain=1.0, offset=0.0))
    assert result is not None
    # First 25 samples after the offset should equal full[50:75]
    import numpy as np

    np.testing.assert_array_equal(np.asarray(result.data), np.asarray(full.data[50:75]))


def test_start_offset_time_window(test_nwb_path):
    """start_offset shifts the file_t cursor for time-window streams."""
    settings_base = dict(
        fs=0.0,
        filepath=test_nwb_path,
        stream_key="Broadband",
        reference_clock=ReferenceClockType.UNKNOWN,
        n_time=100,
    )
    offset = NWBClockDrivenProducer(settings=NWBClockDrivenSettings(start_offset=0.5, **settings_base))
    baseline = NWBClockDrivenProducer(settings=NWBClockDrivenSettings(**settings_base))

    # At playback_rate=1, a 0.5s window from start_offset=0.5 should equal the
    # second half of a 1s window from start_offset=0.
    r_offset = offset(AxisArray.LinearAxis(gain=0.5, offset=0.0))
    r_full = baseline(AxisArray.LinearAxis(gain=1.0, offset=0.0))
    assert r_offset is not None
    assert r_full is not None
    # Compare the tails (second half of the full window).
    import numpy as np

    tail = np.asarray(r_full.data[-r_offset.data.shape[0] :])
    np.testing.assert_array_equal(np.asarray(r_offset.data), tail)


def test_playback_rate_doubles_chunk(test_nwb_path):
    """playback_rate scales variable-chunk size (rate-only path)."""
    base = NWBClockDrivenSettings(
        fs=50.0,
        filepath=test_nwb_path,
        stream_key="BinnedSpikes",
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    normal = NWBClockDrivenProducer(settings=base)
    fast = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=50.0,
            filepath=test_nwb_path,
            stream_key="BinnedSpikes",
            reference_clock=ReferenceClockType.UNKNOWN,
            playback_rate=2.0,
        )
    )

    # 1s clock tick at fs=50 → 50 samples at 1x, 100 at 2x.
    r_normal = normal(AxisArray.LinearAxis(gain=1.0, offset=0.0))
    r_fast = fast(AxisArray.LinearAxis(gain=1.0, offset=0.0))

    assert r_normal.data.shape[0] == 50
    assert r_fast.data.shape[0] == 100


def test_playback_rate_time_window(test_nwb_path):
    """playback_rate widens the time-window advance per tick."""
    fast = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=0.0,
            filepath=test_nwb_path,
            stream_key="Broadband",
            reference_clock=ReferenceClockType.UNKNOWN,
            playback_rate=2.0,
            n_time=100,
        )
    )
    # 0.5s clock tick at 2x should pull ~1s of data.
    r = fast(AxisArray.LinearAxis(gain=0.5, offset=0.0))
    assert r is not None
    # ~1 second at 1000 Hz, allow slack for searchsorted boundaries.
    assert r.data.shape[0] >= 900


def test_playback_rate_zero_pauses(test_nwb_path):
    """playback_rate=0 halts emission without resetting state."""
    producer = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=50.0,
            filepath=test_nwb_path,
            stream_key="BinnedSpikes",
            reference_clock=ReferenceClockType.UNKNOWN,
            playback_rate=0.0,
        )
    )
    # Every tick should yield nothing, and the sample cursor must stay at 0.
    for _ in range(3):
        assert producer(AxisArray.LinearAxis(gain=1.0, offset=0.0)) is None
    assert producer._state.sample_idx == 0


# --- Unit class ---


def test_unit_class_creation():
    """NWBClockDrivenUnit can be instantiated."""
    from ezmsg.nwb.clockdriven import NWBClockDrivenUnit

    unit = NWBClockDrivenUnit()
    assert hasattr(unit, "INPUT_CLOCK")
    assert hasattr(unit, "OUTPUT_SIGNAL")


def test_idle_startup_with_empty_filepath_does_not_raise():
    """Producer constructed with an empty filepath stays idle without raising.

    Matches the GUI startup path where the user launches with no NWB file
    selected yet; ticks should return None until a settings push lands a
    filepath.
    """
    producer = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=50.0,
            n_time=25,
            filepath="",  # idle — no file yet
            stream_key="",
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )
    for _ in range(3):
        assert producer(AxisArray.LinearAxis(gain=1.0, offset=0.0)) is None
    assert producer._state.slicer is None
    assert not producer.exhausted


def test_idle_startup_then_populate_filepath_recovers(test_nwb_path):
    """Going from empty filepath → valid filepath via a fresh producer works.

    Simulates NWBClockDrivenUnit.on_settings routing through _RESET_FIELDS:
    the Unit rebuilds the producer when ``filepath`` changes, and the new
    producer's first tick loads the file.
    """
    idle = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=50.0,
            n_time=25,
            filepath="",
            stream_key="",
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )
    assert idle(AxisArray.LinearAxis(gain=1.0, offset=0.0)) is None

    # Unit would recreate the producer with the new filepath; simulate.
    live = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=50.0,
            n_time=25,
            filepath=str(test_nwb_path),
            stream_key="BinnedSpikes",
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )
    result = live(AxisArray.LinearAxis(gain=1.0, offset=0.0))
    assert result is not None
    assert result.data.shape[0] == 25


def test_seek_preserves_slicer(test_nwb_path):
    """seek() moves the cursor without touching the open file handle."""
    producer = NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            fs=50.0,
            n_time=25,
            filepath=test_nwb_path,
            stream_key="BinnedSpikes",
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )
    producer(AxisArray.LinearAxis(gain=1.0, offset=0.0))  # force _reset_state
    slicer_before = producer._state.slicer

    producer.seek(1.0)
    assert producer._state.slicer is slicer_before  # same file handle
    assert producer._state.sample_idx == 50
    assert producer._state.file_t == producer._state.t0 + 1.0

    # Next read continues from the new cursor.
    r = producer(AxisArray.LinearAxis(gain=1.0, offset=0.0))
    assert r is not None
    assert r.data.shape[0] == 25
    assert producer._state.sample_idx == 75
