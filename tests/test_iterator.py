"""Tests for NWBAxisArrayIterator."""

import math
import threading
import time
from collections import Counter

import numpy as np
import pytest

from ezmsg.nwb import NWBAxisArrayIterator, NWBIteratorSettings, ReferenceClockType


async def test_areset_state_runs_reset_in_worker_thread(test_nwb_path):
    """``_areset_state`` must offload sync ``_reset_state`` to a worker
    thread so the unit's event loop stays responsive during the NWB open."""
    main_tid = threading.get_ident()
    seen_tids: list[int] = []

    class Spy(NWBAxisArrayIterator):
        def _reset_state(self):
            seen_tids.append(threading.get_ident())
            super()._reset_state()

    producer = Spy(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )
    # Discard the eager sync invocation from __init__.
    seen_tids.clear()

    await producer._areset_state()

    assert len(seen_tids) == 1
    assert seen_tids[0] != main_tid, "_reset_state ran on the main event-loop thread"


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


# --- Prefetch ---


@pytest.mark.parametrize("prefetch_chunks", [1, 2, 4])
def test_prefetch_data_integrity(test_nwb_path, prefetch_chunks):
    """Prefetched output is byte-identical to the synchronous path."""

    def collect(prefetch):
        it = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=test_nwb_path,
                chunk_dur=1.0,
                reference_clock=ReferenceClockType.UNKNOWN,
                stream_keys=["Broadband", "BinnedSpikes", "phonemes"],
                prefetch_chunks=prefetch,
            )
        )
        return [(m.key, np.array(m.data, copy=True)) for m in it]

    sync_msgs = collect(0)
    pref_msgs = collect(prefetch_chunks)

    assert len(sync_msgs) == len(pref_msgs)
    for (k1, d1), (k2, d2) in zip(sync_msgs, pref_msgs):
        assert k1 == k2
        np.testing.assert_array_equal(d1, d2)


def test_prefetch_runs_in_worker_thread(test_nwb_path):
    """Build calls happen on a non-main thread when prefetch is enabled."""
    main_tid = threading.get_ident()
    seen_tids: list[int] = []

    class Spy(NWBAxisArrayIterator):
        def _build_chunk_messages(self, chunk_ix):
            seen_tids.append(threading.get_ident())
            return super()._build_chunk_messages(chunk_ix)

    it = Spy(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
            prefetch_chunks=2,
        )
    )
    # Drain — forces every chunk through the worker.
    list(it)

    assert seen_tids, "prefetch worker never produced"
    assert all(tid != main_tid for tid in seen_tids), "_build_chunk_messages ran on the main thread"


def test_prefetch_stop_iteration(test_nwb_path):
    """End-of-stream raises StopIteration (no deadlock on the queue)."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=10.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
            prefetch_chunks=2,
        )
    )
    list(it)
    with pytest.raises(StopIteration):
        next(it)
    assert it.exhausted


def test_prefetch_partial_consumption_clean_close(test_nwb_path):
    """Closing/destroying after partial consumption joins the prefetch worker
    cleanly without leaking the thread.
    """
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=0.1,  # many small chunks so the worker stays busy
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
            prefetch_chunks=2,
        )
    )
    # Consume just one message, leave the rest pending.
    next(it)
    worker = it._state.prefetch_thread
    assert worker is not None and worker.is_alive()

    # __del__ must stop and join the worker.
    it.__del__()

    # Give the OS a beat for the thread to finish, then verify.
    deadline = time.time() + 2.0
    while worker.is_alive() and time.time() < deadline:
        time.sleep(0.01)
    assert not worker.is_alive(), "prefetch thread did not stop within 2s"


# --- Sync fast path ---


def test_next_bypasses_run_coroutine_sync(test_nwb_path):
    """``__next__`` reads the queue directly via ``_produce_sync`` and never
    routes through the async ``_produce`` (which would force an event loop).
    """
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
        )
    )

    sync_calls = 0
    async_calls = 0

    orig_sync = it._produce_sync
    orig_async = it._produce

    def spy_sync():
        nonlocal sync_calls
        sync_calls += 1
        return orig_sync()

    async def spy_async():
        nonlocal async_calls
        async_calls += 1
        return await orig_async()

    it._produce_sync = spy_sync
    it._produce = spy_async

    next(it)
    assert sync_calls == 1
    assert async_calls == 0


# --- HDF5 chunk cache plumbing ---


def test_rdcc_settings_forwarded_to_h5py(test_nwb_path, monkeypatch):
    """Custom rdcc_nbytes / rdcc_nslots are passed to ``h5py.File`` on open.

    We can't read them back off the resulting file: HDF5 caches chunk-cache
    settings from the *first* open of a given file in the process, so the
    fapl on a second open reflects the first open's values regardless of
    the kwargs we pass. Verify the plumbing at the call site instead.
    """
    import h5py

    import ezmsg.nwb.slicer as slicer_mod

    seen_kwargs: list[dict] = []
    real_file = h5py.File

    def spy_file(*args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return real_file(*args, **kwargs)

    monkeypatch.setattr(slicer_mod.h5py, "File", spy_file)

    custom_nbytes = 8 * 1024 * 1024
    custom_nslots = 521  # prime, distinct from default

    NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes"],
            rdcc_nbytes=custom_nbytes,
            rdcc_nslots=custom_nslots,
        )
    )

    assert seen_kwargs, "h5py.File was never called"
    open_kwargs = seen_kwargs[0]
    assert open_kwargs["rdcc_nbytes"] == custom_nbytes
    assert open_kwargs["rdcc_nslots"] == custom_nslots
