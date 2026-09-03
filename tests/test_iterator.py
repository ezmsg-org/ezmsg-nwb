"""Tests for NWBAxisArrayIterator."""

import math
import threading
import time
from collections import Counter

import numpy as np
import pytest
from conftest import GAPPY_N_POST, GAPPY_N_PRE

from ezmsg.nwb import NWBAxisArrayIterator, NWBIteratorSettings, ReferenceClockType

# The gappy-stream fixture (``gappy_nwb_path``) and its GAPPY_* parameters live
# in conftest.py so the slicer and clock-driven tests can share them.


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


def test_stream_keys_no_match_raises(test_nwb_path):
    """When stream_keys matches nothing, fail loudly rather than overflow on int(-inf)."""
    with pytest.raises(ValueError, match="No streams discovered"):
        NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=test_nwb_path,
                chunk_dur=1.0,
                reference_clock=ReferenceClockType.UNKNOWN,
                stream_keys=["does_not_exist"],
            )
        )


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


def test_ragged_stream_lengths_do_not_crash(test_nwb_path):
    """Defensive guard: a stream whose chunk_offsets table is shorter than the
    file-wide ``n_chunks`` must not raise an IndexError; that stream simply
    stops contributing once its table runs out. Offset tables are normally
    built one-entry-per-chunk (see ``test_late_starting_stream_stays_aligned``),
    so this exercises the bounds guard against a short table directly.
    """
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["BinnedSpikes", "RawAnalog"],
        )
    )
    # Simulate a ragged stream: truncate RawAnalog's offset table so it has
    # fewer chunks than the file-wide n_chunks (3). Without the guard in
    # _build_chunk_messages_static this raises IndexError at the last chunk.
    short = it._state.streams["RawAnalog"]
    short["chunk_offsets"] = short["chunk_offsets"][:-1]

    keys = [m.key for m in it]  # full iteration must not raise

    # The full-length stream still produces messages for every chunk...
    assert keys.count("BinnedSpikes") == it._state.n_chunks
    # ...while the truncated stream contributes one fewer.
    assert keys.count("RawAnalog") == it._state.n_chunks - 1


def test_late_starting_stream_stays_aligned(tmp_path):
    """A stream that starts partway into the recording must be emitted in the
    chunks that match its real wall-clock time — not shifted to chunk 0 — and
    its message time offsets must reflect its true start time. Regression for
    streams with different ``starting_time`` (e.g. CereLink Hub2 vs NPLAY).
    """
    import datetime

    from pynwb import NWBHDF5IO, NWBFile, TimeSeries

    path = tmp_path / "late_start.nwb"
    nwb = NWBFile(
        session_description="m",
        identifier="m",
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )
    rate = 100.0
    # Stream A: t=0..5s. Stream B: t=2..5s (starts 2 chunks / 2 s later).
    nwb.add_acquisition(
        TimeSeries(
            name="A",
            data=np.arange(int(5 * rate), dtype=np.float32)[:, None],
            unit="V",
            rate=rate,
            starting_time=0.0,
        )
    )
    nwb.add_acquisition(
        TimeSeries(
            name="B",
            data=(1000 + np.arange(int(3 * rate), dtype=np.float32))[:, None],
            unit="V",
            rate=rate,
            starting_time=2.0,
        )
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)

    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )
    # Offset tables are built one entry per global chunk for every stream.
    n_chunks = it._state.n_chunks
    assert len(it._state.streams["A"]["chunk_offsets"]) == n_chunks
    assert len(it._state.streams["B"]["chunk_offsets"]) == n_chunks

    # First non-empty message for each stream: when does each first appear and
    # at what time offset?
    first_offset = {}
    for m in it:
        if m.data.size and m.key not in first_offset:
            first_offset[m.key] = (m.axes["time"].offset, float(m.data.flat[0]))

    # A begins at t=0; B begins at t=2.0 with its own first sample (1000) —
    # NOT shifted to t=0.
    assert first_offset["A"][0] == pytest.approx(0.0)
    assert first_offset["B"][0] == pytest.approx(2.0)
    assert first_offset["B"][1] == 1000.0


def test_chunk_offsets_match_searchsorted_for_noninteger_period(tmp_path):
    """When chunk_dur is not an integer multiple of the sample period, chunk
    offsets must be the first sample at/after each boundary (searchsorted
    side='left' / ceil), not the nearest sample (round) — otherwise a
    pre-boundary sample leaks into the next chunk and disagrees with the
    event/timestamped paths.
    """
    import datetime

    from pynwb import NWBHDF5IO, NWBFile, TimeSeries

    path = tmp_path / "noninteger.nwb"
    nwb = NWBFile(
        session_description="m",
        identifier="m",
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )
    rate = 256.0  # 256 samples/s -> 25.6 samples per 0.1 s chunk (non-integer)
    nwb.add_acquisition(
        TimeSeries(
            name="S",
            data=np.arange(int(3 * rate), dtype=np.float32)[:, None],
            unit="V",
            rate=rate,
            starting_time=0.0,
        )
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)

    chunk_dur = 0.1
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=path,
            chunk_dur=chunk_dur,
            reference_clock=ReferenceClockType.UNKNOWN,
        )
    )
    offsets = np.asarray(it._state.streams["S"]["chunk_offsets"])
    n_chunks = it._state.n_chunks
    n_samples = int(3 * rate)

    boundaries = np.arange(n_chunks) * chunk_dur  # start_time=0, ts_off=0, t0=0
    sample_times = np.arange(n_samples) / rate
    expected = np.clip(np.searchsorted(sample_times, boundaries, side="left"), 0, n_samples)
    np.testing.assert_array_equal(offsets, expected)


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


def test_prefetch_runs_in_worker_thread(test_nwb_path, monkeypatch):
    """Build calls happen on a non-main thread when prefetch is enabled."""
    import ezmsg.nwb.iterator as iterator_mod

    main_tid = threading.get_ident()
    seen_tids: list[int] = []
    real = iterator_mod._build_chunk_messages_static

    def spy(slicer, streams, chunk_ix, gap_tol=0.5):
        seen_tids.append(threading.get_ident())
        return real(slicer, streams, chunk_ix, gap_tol)

    monkeypatch.setattr(iterator_mod, "_build_chunk_messages_static", spy)

    it = NWBAxisArrayIterator(
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
    assert all(tid != main_tid for tid in seen_tids), "_build_chunk_messages_static ran on the main thread"


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


def test_prefetch_worker_does_not_keep_iterator_alive(test_nwb_path):
    """``del it`` must drop the last reference and trigger ``__del__``,
    even while the prefetch worker is running.

    Regression: the worker used to capture ``self`` via a bound method,
    keeping the iterator's refcount above zero. ``del it`` then never ran
    ``__del__`` / ``_stop_prefetch``, leaving the worker alive past the
    iterator's intended lifetime — which deadlocks at process exit when
    h5py's atexit close path contends with the worker's phil lock.
    """
    import gc
    import weakref

    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=0.1,  # many chunks so the worker is actively running
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
            prefetch_chunks=2,
        )
    )
    next(it)  # ensure worker has started and is producing
    assert it._state.prefetch_thread is not None
    assert it._state.prefetch_thread.is_alive()

    ref = weakref.ref(it)
    del it
    # Without forcing GC: refcount alone should be enough to drop the
    # iterator if the worker doesn't capture self.
    assert ref() is None, (
        "iterator survived `del` — something (most likely the prefetch worker) is holding a strong reference to self"
    )
    gc.collect()  # belt-and-suspenders for any cyclic refs


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


def test_prefetch_end_sentinel_survives_slow_consumer():
    """Regression: the end-of-stream sentinel must be delivered even when the
    consumer keeps the bounded queue full past the worker's per-put timeout.

    The sentinel (and worker exceptions) were previously put with a fixed 1s
    timeout and silently dropped on ``queue.Full``, so a consumer that stayed
    full longer than that never received the sentinel and its final ``get()``
    blocked forever.
    """
    import queue as _queue

    from ezmsg.nwb.iterator import _PREFETCH_END, _prefetch_worker

    q: _queue.Queue = _queue.Queue(maxsize=1)
    q.put(object())  # occupy the single slot so the worker's sentinel put blocks
    stop = threading.Event()

    # n_chunks=0 -> the worker goes straight to the finally sentinel put; slicer
    # and streams are unused on that path.
    worker = threading.Thread(target=_prefetch_worker, args=(None, {}, 0, q, stop), daemon=True)
    worker.start()

    # Free the slot only after the old fixed 1s put timeout would have elapsed,
    # so a non-retrying worker has already dropped the sentinel by now.
    time.sleep(1.3)
    q.get_nowait()  # remove the occupying item, making room for the sentinel

    got = q.get(timeout=2.0)  # a retrying worker delivers it once room appears
    assert got is _PREFETCH_END
    worker.join(timeout=2.0)
    assert not worker.is_alive()


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


# --- Timestamp gaps (Solution A: split chunks at gaps) ----------------------


def test_gappy_whole_stream_not_one_gap_spanning_chunk(gappy_nwb_path):
    """A chunk big enough to cover the whole gappy stream must be split into
    two gap-free messages, not emitted as one chunk that silently spans the
    gap with a uniform LinearAxis.
    """
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=gappy_nwb_path,
            chunk_dur=100.0,  # whole stream in a single chunk
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Gappy"],
        )
    )
    msgs = [m for m in it if m.data.shape[0] > 0]

    assert len(msgs) == 2, "gappy stream was not split at the gap"
    assert msgs[0].data.shape[0] == GAPPY_N_PRE
    assert msgs[1].data.shape[0] == GAPPY_N_POST
    # Offsets reflect the real first-timestamp of each gap-free run.
    assert msgs[0].axes["time"].offset == pytest.approx(0.0, abs=1e-6)
    assert msgs[1].axes["time"].offset == pytest.approx(2.50, abs=1e-6)
    # Sample ordering and identity preserved across the split.
    assert msgs[0].data[0, 0] == 0
    assert msgs[0].data[-1, 0] == GAPPY_N_PRE - 1
    assert msgs[1].data[0, 0] == GAPPY_N_PRE
    assert msgs[1].data[-1, 0] == GAPPY_N_PRE + GAPPY_N_POST - 1


def test_gappy_midchunk_split(gappy_nwb_path):
    """When the gap falls in the middle of an index-based chunk, that chunk is
    split into two messages while gap-free chunks pass through unchanged.

    chunk_dur=1.0 @ 100 Hz -> 100 samples/chunk. The gap sits between sample
    149 and 150, i.e. inside the second chunk (samples 100..199).
    """
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=gappy_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Gappy"],
        )
    )
    msgs = [m for m in it if m.data.shape[0] > 0]

    # chunk0: samples 0..99 (gap-free) -> 1 msg
    # chunk1: samples 100..199 spans gap -> split 100..149 / 150..199
    # chunk2: samples 200..299 (gap-free) -> 1 msg
    assert len(msgs) == 4
    sizes = [m.data.shape[0] for m in msgs]
    assert sizes == [100, 50, 50, 100]
    # The two halves of the split chunk sit on either side of the gap.
    assert msgs[1].axes["time"].offset == pytest.approx(1.00, abs=1e-6)
    assert msgs[2].axes["time"].offset == pytest.approx(2.50, abs=1e-6)


def test_gappy_segments_match_true_timestamps(gappy_nwb_path):
    """Every emitted message's reconstructed time axis (offset + i*gain) must
    match the file's true per-sample timestamps within a fraction of a sample.
    """
    from ezmsg.nwb.slicer import NWBSlicer

    slicer = NWBSlicer(
        filepath=gappy_nwb_path,
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["Gappy"],
    )
    true_ts = np.asarray(slicer.get_stream_info("Gappy").timestamps[:])
    slicer.close()

    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=gappy_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Gappy"],
        )
    )
    for m in it:
        if m.data.shape[0] == 0:
            continue
        gain = m.axes["time"].gain
        offset = m.axes["time"].offset
        idx = m.data[:, 0].astype(int)  # data value == global sample index
        reconstructed = offset + np.arange(m.data.shape[0]) * gain
        np.testing.assert_allclose(reconstructed, true_ts[idx], atol=gain * 0.5)


def test_gappy_total_samples_and_order_preserved(gappy_nwb_path):
    """Splitting at gaps must not drop, duplicate, or reorder samples."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=gappy_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Gappy"],
        )
    )
    data = np.concatenate([m.data for m in it if m.data.shape[0] > 0], axis=0)
    n = GAPPY_N_PRE + GAPPY_N_POST
    assert data.shape[0] == n
    np.testing.assert_array_equal(data[:, 0], np.arange(n, dtype=np.float32))


def test_gap_tol_disables_split(gappy_nwb_path):
    """A large ``gap_tol`` widens the gap threshold enough that the stream is
    emitted as a single (gap-spanning) chunk again — the knob works.
    """
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=gappy_nwb_path,
            chunk_dur=100.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Gappy"],
            gap_tol=1e6,
        )
    )
    msgs = [m for m in it if m.data.shape[0] > 0]
    assert len(msgs) == 1
    assert msgs[0].data.shape[0] == GAPPY_N_PRE + GAPPY_N_POST


def test_jittered_stream_not_oversplit(test_nwb_path):
    """The lightly-jittered Broadband stream (no real gaps) must not be split:
    sub-microsecond jitter stays well under the gap threshold.
    """
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
        )
    )
    msgs = [m for m in it if m.data.shape[0] > 0]
    # 3 s of 1 kHz data in 1 s chunks -> exactly 3 non-empty messages.
    assert len(msgs) == 3


class TestMessagesArriveReadyForConsumers:
    """Two things only the source can supply, both set once per file.

    ``chunk_dim`` names the dimension messages accumulate along -- the one whose
    length is just however much of the file this chunk covered, and which a
    consumer must leave out of the state it caches against the stream's
    configuration. ``fingerprint`` is the channel axis's content digest, cached
    on the axis and pickled with it; priming it here spares the first consumer
    in every process from recomputing it on every message.
    """

    @staticmethod
    def _messages(path):
        it = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=path,
                chunk_dur=1.0,
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )
        return [msg for msg in it if math.prod(msg.data.shape) > 0]

    def test_every_message_declares_its_chunk_dim(self, test_nwb_path):
        msgs = self._messages(test_nwb_path)
        assert msgs, "no messages produced"
        undeclared = sorted({m.key for m in msgs if m.chunk_dim != "time"})
        assert not undeclared, f"streams not declaring chunk_dim='time': {undeclared}"

    def test_every_channel_axis_is_primed(self, test_nwb_path):
        cold = sorted(
            {
                m.key
                for m in self._messages(test_nwb_path)
                if "ch" in m.axes and "_fingerprint" not in m.axes["ch"].__dict__
            }
        )
        assert not cold, f"streams handing over a cold ch axis: {cold}"

    def test_the_chunk_axis_is_left_cold(self, test_nwb_path):
        """Digesting per-message timestamps would be pure cost: no consumer reads
        the chunk axis's fingerprint."""
        irregular = [m for m in self._messages(test_nwb_path) if hasattr(m.axes.get("time"), "data")]
        assert irregular, "expected at least one stream with coordinate timestamps"
        assert all("_fingerprint" not in m.axes["time"].__dict__ for m in irregular)

    def test_it_all_survives_the_transport(self, test_nwb_path):
        import pickle

        msg = next(m for m in self._messages(test_nwb_path) if "ch" in m.axes)
        landed = pickle.loads(pickle.dumps(msg))
        assert landed.chunk_dim == "time"
        assert "_fingerprint" in landed.axes["ch"].__dict__
