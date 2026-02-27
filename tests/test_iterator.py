"""Tests for NWBAxisArrayIterator."""

import math
from collections import Counter

import numpy as np
import pytest

from ezmsg.nwb import NWBAxisArrayIterator, NWBIteratorSettings, ReferenceClockType

# --- Stream discovery ---


def test_all_streams_discovered(test_nwb_path):
    """Iterator discovers all streams including /processing and custom intervals."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )

    stream_names = set(it._state.streams.keys())
    assert stream_names == {"Broadband", "RawAnalog", "BinnedSpikes", "Force", "trials", "phonemes"}

    counts = Counter()
    for msg in it:
        if math.prod(msg.data.shape) > 0:
            counts[msg.key] += 1

    assert counts["Broadband"] > 0
    assert counts["BinnedSpikes"] > 0
    assert counts["trials"] == 3
    assert counts["phonemes"] == 10


def test_stream_keys_filter(test_nwb_path):
    """stream_keys setting filters which streams are discovered and yielded."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband", "trials"],
        )
    )

    assert set(it._state.streams.keys()) == {"Broadband", "trials"}

    keys_seen = set()
    for msg in it:
        keys_seen.add(msg.key)

    assert keys_seen == {"Broadband", "trials"}


# --- Message shape and structure ---


def test_continuous_data_shape(test_nwb_path):
    """Continuous data chunks have correct shape, dims, and LinearAxis time axis."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
        )
    )
    msg = next(it)

    assert msg.key == "Broadband"
    assert msg.data.ndim == 2
    assert msg.data.shape[1] == 8
    assert msg.dims == ["time", "ch"]
    assert type(msg.axes["time"]).__name__ == "LinearAxis"


def test_1d_timeseries_dims(test_nwb_path):
    """1D timeseries data gets dims=['time'], not ['time', 'ch']."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Force"],
        )
    )
    msg = next(it)
    assert msg.data.ndim == 1
    assert msg.dims == ["time"]


def test_interval_table_structure(test_nwb_path):
    """Interval tables produce correct AxisArray messages (sample-by-sample, CoordinateAxis)."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["phonemes"],
        )
    )
    msg = next(it)

    assert msg.key == "phonemes"
    assert msg.data.ndim == 2
    assert msg.data.shape[0] == 1  # sample-by-sample
    assert "time" in msg.axes
    assert hasattr(msg.axes["time"], "data")  # CoordinateAxis for events


# --- Exhaustion and __next__ protocol ---


def test_exhausted_false_initially(test_nwb_path):
    """Iterator is not exhausted right after construction."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
        )
    )
    assert not it.exhausted


def test_exhausted_after_full_consumption(test_nwb_path):
    """Iterator reports exhausted after all messages are consumed."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=10.0,  # single chunk covers all data
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
        )
    )
    list(it)  # consume all
    assert it.exhausted


def test_stop_iteration(test_nwb_path):
    """__next__ raises StopIteration when data is exhausted."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=10.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
        )
    )
    list(it)
    with pytest.raises(StopIteration):
        next(it)


# --- Total sample accounting ---


def test_total_samples_rate_only(test_nwb_path):
    """All samples from a rate-only stream are emitted exactly once."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
        )
    )
    total = sum(m.data.shape[0] for m in it)
    assert total == 150


def test_total_samples_timestamped(test_nwb_path):
    """All samples from a timestamped stream are emitted exactly once."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
        )
    )
    total = sum(m.data.shape[0] for m in it)
    assert total == 3000


def test_total_samples_1d(test_nwb_path):
    """All samples from a 1D stream are emitted exactly once."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Force"],
        )
    )
    total = sum(m.data.shape[0] for m in it)
    assert total == 300


def test_total_events(test_nwb_path):
    """All events from an interval table are emitted exactly once."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["phonemes"],
        )
    )
    counts = Counter()
    for m in it:
        if math.prod(m.data.shape) > 0:
            counts[m.key] += m.data.shape[0]
    assert counts["phonemes"] == 10


# --- chunk_dur behaviour ---


def test_chunk_dur_determines_chunk_count(test_nwb_path):
    """Smaller chunk_dur produces more chunks (messages) for a continuous stream."""

    def count_messages(chunk_dur):
        it = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=test_nwb_path,
                chunk_dur=chunk_dur,
                reference_clock=ReferenceClockType.UNKNOWN,
                stream_keys=["RawAnalog"],
            )
        )
        return sum(1 for m in it if m.data.shape[0] > 0)

    n_big = count_messages(10.0)
    n_small = count_messages(0.5)
    assert n_small > n_big


def test_chunk_dur_preserves_total_samples(test_nwb_path):
    """Different chunk_dur values still emit the same total sample count."""

    def total_samples(chunk_dur):
        it = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=test_nwb_path,
                chunk_dur=chunk_dur,
                reference_clock=ReferenceClockType.UNKNOWN,
                stream_keys=["RawAnalog"],
            )
        )
        return sum(m.data.shape[0] for m in it)

    assert total_samples(0.5) == total_samples(2.0) == 1500


# --- Time axis correctness ---


def test_time_axis_offset_advances(test_nwb_path):
    """Successive chunks have increasing time axis offsets for rate-only streams."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
        )
    )
    offsets = [m.axes["time"].offset for m in it if m.data.shape[0] > 0]
    assert len(offsets) > 1
    assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))


def test_timestamped_time_axis_offset_advances(test_nwb_path):
    """Successive chunks have increasing time axis offsets for timestamped streams."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
        )
    )
    offsets = [m.axes["time"].offset for m in it if m.data.shape[0] > 0]
    assert len(offsets) > 1
    assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))


def test_event_time_axis_has_coordinate_data(test_nwb_path):
    """Event messages have CoordinateAxis with actual timestamp data."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=10.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["trials"],
        )
    )
    event_msgs = [m for m in it if m.data.shape[0] > 0]
    assert len(event_msgs) == 3  # one per event
    for m in event_msgs:
        assert hasattr(m.axes["time"], "data")
        assert len(m.axes["time"].data) == 1


# --- Data integrity ---


def test_data_not_corrupted(test_nwb_path):
    """Data emitted by the iterator matches direct slicer reads."""
    from ezmsg.nwb.slicer import NWBSlicer

    # Read all BinnedSpikes via iterator
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
        )
    )
    iter_data = np.concatenate([m.data for m in it], axis=0)

    # Read all BinnedSpikes via slicer
    slicer = NWBSlicer(
        filepath=test_nwb_path,
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["BinnedSpikes"],
    )
    slicer_msg = slicer.read_by_index("BinnedSpikes", 0, 150)
    slicer.close()

    np.testing.assert_array_equal(iter_data, slicer_msg.data)


# --- Multi-stream interleaving ---


def test_multi_stream_interleaving(test_nwb_path):
    """When iterating multiple streams, messages from all streams are interleaved per chunk."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes", "RawAnalog"],
        )
    )
    keys = [m.key for m in it]
    # Both streams should appear
    assert "BinnedSpikes" in keys
    assert "RawAnalog" in keys
    # They should be interleaved (not all of one then all of another)
    first_binned = keys.index("BinnedSpikes")
    first_raw = keys.index("RawAnalog")
    # Both appear in the first chunk's worth of messages
    assert abs(first_binned - first_raw) <= 1


# --- Channel axis preserved ---


def test_electrode_labels_preserved(test_nwb_path):
    """Electrode labels from the file are present in the ch axis of emitted messages."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
        )
    )
    msg = next(it)
    ch_labels = list(msg.axes["ch"].data)
    assert ch_labels == [f"elec{i}" for i in range(8)]
