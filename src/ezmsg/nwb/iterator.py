from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import typing
from collections import deque

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc.protocols import processor_state
from ezmsg.baseproc.stateful import BaseStatefulProducer
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .slicer import DEFAULT_GAP_TOL, NWBSlicer, find_gaps
from .util import ReferenceClockType

# Sentinel pushed to the prefetch queue to indicate end-of-stream. Identity-compared.
_PREFETCH_END = object()


class NWBIteratorSettings(ez.Settings):
    filepath: typing.Union[os.PathLike, str]
    chunk_dur: float = 1.0
    # start_time: typing.Optional[float] = None
    # stop_time: typing.Optional[float] = None
    reference_clock: ReferenceClockType = ReferenceClockType.SYSTEM
    reref_now: bool = False
    self_terminating: bool = True
    stream_keys: typing.Optional[list[str]] = None
    prefetch_chunks: int = 0
    """Number of chunks to prefetch in a background thread.

    ``0`` (default) keeps the original synchronous behaviour: each ``next()``
    call blocks on h5py I/O for the next chunk. ``> 0`` spawns a daemon
    thread that fills a bounded queue with up to ``prefetch_chunks`` chunks
    of pre-read data, so the consumer rarely waits on disk. ``2``–``4`` is
    typically enough; the win is overlapping one chunk's read with one
    chunk's downstream compute.
    """
    rdcc_nbytes: int = NWBSlicer.DEFAULT_RDCC_NBYTES
    """HDF5 raw data chunk cache size in bytes (forwarded to NWBSlicer)."""
    rdcc_nslots: int = NWBSlicer.DEFAULT_RDCC_NSLOTS
    """HDF5 raw data chunk cache slot count (forwarded to NWBSlicer)."""
    gap_tol: float = DEFAULT_GAP_TOL
    """Gap threshold for timestamped continuous streams, as a fraction of the
    nominal sample period: a gap is declared when an interval exceeds
    ``(1 + gap_tol) / fs``. Chunks spanning a gap are split into gap-free
    messages so the regular ``LinearAxis`` never misplaces post-gap samples.

    The default ``0.5`` (1.5x period) sits between neural-data jitter (<~1.05x)
    and the smallest real gap (one dropped sample = ~2x), so it catches every
    gap without splitting on jitter. Set very large to disable splitting. No
    effect on rate-only streams or event tables.
    """
    dejitter: bool = True
    """Reconstruct smoothed monotone timestamps for streams with a paired
    ``*_device_ts`` sibling before chunking (forwarded to ``NWBSlicer``). Removes
    the per-sample clock jitter that would otherwise gap-split such a stream into
    thousands of tiny messages. Set False to chunk raw timestamps unchanged."""
    clock_groups: typing.Optional[list[list[str]]] = None
    """Override auto clock-group detection (forwarded to ``NWBSlicer``). ``None``
    auto-detects PTP-shared groups; a list of key-lists forces them."""
    dejitter_cache: bool = True
    """Cache reconstructed timestamps on disk across opens (forwarded to
    ``NWBSlicer``)."""
    real_gap_threshold: typing.Optional[float] = None
    """Real-gap guard threshold in seconds (forwarded to ``NWBSlicer``). ``None``
    auto-derives per stream; genuine gaps are preserved so chunks still split at
    them. ``float("inf")`` disables the guard."""
    structured_ch_axis: bool = False
    """Build the ``ch`` axis as a structured record array from the electrodes
    table (forwarded to ``NWBSlicer``), matching what a live acquisition source
    emits: position, label, bank, headstage, array identity. Default False --
    these columns are an acquisition-stack convention, not part of the NWB
    schema. See :mod:`~ezmsg.nwb.electrodes`."""


@processor_state
class NWBIteratorState:
    n_chunks: int = 0
    chunk_ix: int = 0
    slicer: NWBSlicer | None = None
    streams: dict | None = None
    deque: deque | None = None
    prefetch_queue: typing.Any = None
    prefetch_thread: typing.Any = None
    prefetch_stop: typing.Any = None


def _build_chunk_messages_static(
    slicer: NWBSlicer,
    streams: dict,
    chunk_ix: int,
    gap_tol: float = DEFAULT_GAP_TOL,
) -> list[AxisArray]:
    """Build the messages for ``chunk_ix`` from explicit slicer/streams refs.

    Module-level (not a method) on purpose: the prefetch worker captures only
    the values it needs, never ``self``. If it captured ``self``, the
    iterator's refcount would never drop to 0 while the worker is alive, so
    ``del it`` would not trigger ``__del__`` / ``_stop_prefetch`` — the worker
    would then outlive its intended scope and can deadlock at process exit
    against h5py's atexit close path (both contend for the file's phil lock).
    """
    ts_off = slicer.ts_off
    out: list[AxisArray] = []
    for strm_name, strm_dict in streams.items():
        info = strm_dict["info"]
        chunk_offsets = strm_dict["chunk_offsets"]
        # Defensive: offset tables are built one entry per global chunk, so this
        # never trips in normal operation. It guards against a caller handing us
        # a stream whose table is shorter than ``n_chunks`` — index out of bounds
        # would otherwise crash the whole chunk instead of just dropping that
        # stream for this index.
        if chunk_ix >= len(chunk_offsets):
            continue
        start_idx = chunk_offsets[chunk_ix]
        if chunk_ix + 1 < len(chunk_offsets):
            stop_idx = chunk_offsets[chunk_ix + 1]
        else:
            stop_idx = info.dset.shape[0]
        template = info.template

        if info.is_event:
            if start_idx < stop_idx:
                table = info.table_ref
                for idx in range(start_idx, stop_idx):
                    out.append(
                        replace(
                            template,
                            data=info.dset[idx : idx + 1],
                            axes={
                                **template.axes,
                                "time": replace(
                                    template.axes["time"],
                                    data=ts_off + table.start_time[idx : idx + 1],
                                ),
                            },
                            key=strm_name,
                        )
                    )
            else:
                out.append(template)
        else:
            out_data = info.dset[start_idx:stop_idx]
            time_axis = template.axes["time"]

            # Timestamped continuous stream on a regular LinearAxis: a single
            # chunk that spans a gap in the explicit timestamps would emit a
            # uniform time axis that misplaces every post-gap sample. Split it
            # into gap-free runs, each anchored on its own first timestamp.
            # Rate-only streams (no per-sample timestamps) and CoordinateAxis
            # streams (no ``gain`` to compare against) keep the old single-chunk
            # path.
            if info.has_timestamps and info.timestamps is not None and hasattr(time_axis, "gain") and len(out_data):
                ts_chunk = np.asarray(info.timestamps[start_idx:stop_idx])
                gap_after = find_gaps(ts_chunk, time_axis.gain, gap_tol)
                # Run boundaries within the chunk: [0, gap1+1, gap2+1, ..., len].
                bounds = [0, *(gap_after + 1).tolist(), ts_chunk.shape[0]]
                for b0, b1 in zip(bounds[:-1], bounds[1:]):
                    out.append(
                        replace(
                            template,
                            data=out_data[b0:b1],
                            axes={
                                **template.axes,
                                "time": replace(time_axis, offset=ts_off + ts_chunk[b0]),
                            },
                            key=strm_name,
                        )
                    )
            else:
                if info.timestamps is not None and start_idx < len(info.timestamps):
                    # Explicit timestamps are already absolute (file-relative) times.
                    chunk_t0 = info.timestamps[start_idx]
                else:
                    # Rate-only: the absolute time of ``start_idx`` is the stream's
                    # own start (``info.t0``) plus the within-stream offset. Omitting
                    # ``info.t0`` would label a late-starting stream as if it began
                    # at the file origin, mis-timing it against other streams.
                    chunk_t0 = float(info.t0) + time_axis.gain * start_idx
                out.append(
                    replace(
                        template,
                        data=out_data,
                        axes={
                            **template.axes,
                            "time": replace(
                                time_axis,
                                offset=ts_off + chunk_t0,
                            ),
                        },
                        key=strm_name,
                    )
                )
    return out


def _prefetch_worker(
    slicer: NWBSlicer,
    streams: dict,
    n_chunks: int,
    q: queue.Queue,
    stop: threading.Event,
    gap_tol: float = DEFAULT_GAP_TOL,
) -> None:
    """Prefetch worker target. Top-level function (no closure over the
    iterator) so the iterator can be garbage-collected as soon as the user
    drops their reference; ``__del__`` then runs and cleanly shuts the
    worker down.
    """
    try:
        for chunk_ix in range(n_chunks):
            if stop.is_set():
                return
            msgs = _build_chunk_messages_static(slicer, streams, chunk_ix, gap_tol)
            # Block on a full queue, but wake periodically to honour stop.
            while not stop.is_set():
                try:
                    q.put(msgs, timeout=0.1)
                    break
                except queue.Full:
                    continue
            else:
                return
    except Exception as exc:  # pragma: no cover — surfaces in get()
        ez.logger.exception("NWBAxisArrayIterator prefetch worker failed: %s", exc)
        while not stop.is_set():
            try:
                q.put(exc, timeout=0.1)
                break
            except queue.Full:
                continue
        return
    finally:
        # Always signal end-of-stream so the consumer wakes up.
        while not stop.is_set():
            try:
                q.put(_PREFETCH_END, timeout=0.1)
                break
            except queue.Full:
                continue


class NWBAxisArrayIterator(BaseStatefulProducer[NWBIteratorSettings, AxisArray, NWBIteratorState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Eagerly initialize state (load NWB file metadata) so that
        # _state.streams is available immediately after construction.
        self._reset_state()
        self._hash = 0

    @property
    def exhausted(self) -> bool:
        if self._state.deque:
            return False
        if self._state.prefetch_thread is not None:
            # End-of-stream is signalled by _PREFETCH_END landing in the queue.
            # The flag below flips to True once the consumer has popped it.
            return self._prefetch_drained
        return self._state.chunk_ix >= self._state.n_chunks

    async def _areset_state(self) -> None:
        """Offload the slow ``NWBSlicer`` open and chunk-table build onto
        a worker thread so the unit's event loop can keep servicing other
        async tasks during the multi-second first-open. See the matching
        override on ``NWBClockDrivenProducer`` for context.
        """
        await asyncio.to_thread(self._reset_state)

    def _reset_state(self) -> None:
        # Tear down any existing prefetch worker before mutating state.
        self._stop_prefetch()

        self._state.n_chunks = 0
        self._state.chunk_ix = 0
        self._state.streams = {}
        self._state.deque = deque()
        self._prefetch_drained = False

        if self._state.slicer is not None:
            self._state.slicer.close()
            self._state.slicer = None

        self._preload()

        if self.settings.prefetch_chunks > 0 and self._state.n_chunks > 0:
            self._start_prefetch()

    def _preload(self):
        slicer = NWBSlicer(
            filepath=self.settings.filepath,
            reference_clock=self.settings.reference_clock,
            reref_now=self.settings.reref_now,
            stream_keys=self.settings.stream_keys,
            rdcc_nbytes=self.settings.rdcc_nbytes,
            rdcc_nslots=self.settings.rdcc_nslots,
            dejitter=self.settings.dejitter,
            clock_groups=self.settings.clock_groups,
            dejitter_cache=self.settings.dejitter_cache,
            real_gap_threshold=self.settings.real_gap_threshold,
            structured_ch_axis=self.settings.structured_ch_axis,
        )
        self._state.slicer = slicer

        # Fail loudly when no streams were discovered.
        if not slicer.stream_names:
            raise ValueError(
                f"No streams discovered in {self.settings.filepath!s}"
                + (
                    f" matching stream_keys={self.settings.stream_keys!r}"
                    if self.settings.stream_keys is not None
                    else ""
                )
                + "."
            )

        # Build per-stream chunk offset tables from slicer metadata
        start_time = slicer.start_time
        stop_time = slicer.stop_time
        t_range = stop_time - start_time
        n_chunks = int(np.ceil(t_range / self.settings.chunk_dur))

        self._state.streams = {}
        for name in slicer.stream_names:
            info = slicer.get_stream_info(name)
            template = info.template

            if info.is_event:
                # Irregular interval stream — find first sample index in each chunk.
                timestamps = info.timestamps
                chunk_boundaries = start_time + np.arange(n_chunks) * self.settings.chunk_dur - slicer.ts_off
                chunk_ix_offsets = np.searchsorted(timestamps, chunk_boundaries, side="left").astype(int)
            else:
                # Sample index at each GLOBAL chunk boundary, computed from the
                # stream's own start time (``info.t0``) and nominal gain. Building
                # offsets on the shared global grid — rather than from each
                # stream's own first sample — keeps streams that start at
                # different times mutually aligned: chunk ``j`` covers the same
                # wall-clock window for every stream. Boundaries before this
                # stream begins go negative and boundaries past its end overshoot
                # ``n_samples``; clamping turns both into empty slices, so a
                # late-starting / early-ending stream simply contributes nothing
                # to the chunks outside its span instead of being shifted.
                gain = template.axes["time"].gain
                chunk_boundaries = start_time + np.arange(n_chunks) * self.settings.chunk_dur - slicer.ts_off
                # First sample at/after each boundary — ``searchsorted(side="left")``
                # semantics on a regular grid, matching the event branch above.
                # Use ceil, not round: round assigns a boundary to the nearest
                # sample, which can pull a pre-boundary sample into the chunk when
                # chunk_dur isn't an integer multiple of the sample period and
                # disagree with the timestamped/event paths. The epsilon absorbs
                # floating-point drift so an exact boundary isn't bumped up a sample.
                rel = (chunk_boundaries - float(info.t0)) / gain
                chunk_ix_offsets = np.ceil(rel - 1e-6).astype(int)
                chunk_ix_offsets = np.clip(chunk_ix_offsets, 0, info.dset.shape[0])

            self._state.streams[name] = {
                "info": info,
                "chunk_offsets": chunk_ix_offsets,
            }

        self._state.n_chunks = n_chunks

    def _build_chunk_messages(self, chunk_ix: int) -> list[AxisArray]:
        """Sync-side wrapper around :func:`_build_chunk_messages_static`."""
        return _build_chunk_messages_static(self._state.slicer, self._state.streams, chunk_ix, self.settings.gap_tol)

    def _chunk_step(self):
        """Sync path: build the next chunk and append to the deque."""
        msgs = self._build_chunk_messages(self._state.chunk_ix)
        self._state.deque.extend(msgs)
        self._state.chunk_ix += 1

    # --- Prefetch worker ---

    def _start_prefetch(self) -> None:
        """Spawn the prefetch worker. The worker is the sole reader of
        ``info.dset`` once started; the main thread must not slice the
        h5py datasets until the worker has been joined.
        """
        self._state.prefetch_queue = queue.Queue(maxsize=self.settings.prefetch_chunks)
        self._state.prefetch_stop = threading.Event()

        # Pass refs as args, not via closure-over-self. Keeping ``self`` out
        # of the worker's closure is what lets ``del it`` actually free the
        # iterator and trigger ``__del__``; see ``_prefetch_worker`` for the
        # full rationale.
        t = threading.Thread(
            target=_prefetch_worker,
            args=(
                self._state.slicer,
                self._state.streams,
                self._state.n_chunks,
                self._state.prefetch_queue,
                self._state.prefetch_stop,
                self.settings.gap_tol,
            ),
            name="NWBIterator-prefetch",
            daemon=True,
        )
        self._state.prefetch_thread = t
        t.start()

    def _stop_prefetch(self) -> None:
        if self._state.prefetch_stop is not None:
            self._state.prefetch_stop.set()
        if self._state.prefetch_queue is not None:
            # Drain so a worker blocked on put() can finish and observe stop.
            try:
                while True:
                    self._state.prefetch_queue.get_nowait()
            except queue.Empty:
                pass
        if self._state.prefetch_thread is not None:
            # Generous timeout: a single h5py read on a slow remote/USB-C
            # device can take seconds for a multi-MB chunk. Anything shorter
            # risks orphaning a thread that's still holding HDF5's per-file
            # lock, which then deadlocks the next ``slicer.close()``.
            self._state.prefetch_thread.join(timeout=30.0)
        self._state.prefetch_thread = None
        self._state.prefetch_queue = None
        self._state.prefetch_stop = None

    def _ingest_prefetch_item(self, item: typing.Any) -> bool:
        """Handle a value pulled from the prefetch queue.

        Returns ``True`` if the stream is now drained (no more messages will
        ever arrive), ``False`` otherwise. Raises if the worker reported an
        exception. On success, extends ``self._state.deque`` with the chunk's
        messages.
        """
        if item is _PREFETCH_END:
            self._prefetch_drained = True
            return True
        if isinstance(item, BaseException):
            self._prefetch_drained = True
            raise item
        self._state.deque.extend(item)
        return False

    # --- Production paths ---

    def _produce_sync(self) -> AxisArray | None:
        """Synchronous next-message production. Used by ``__next__`` to
        bypass the ``run_coroutine_sync`` overhead of ``BaseProducer.__call__``.
        """
        while not self._state.deque:
            if self._state.prefetch_thread is not None:
                if self._prefetch_drained:
                    self._cleanup_after_drain()
                    return None
                item = self._state.prefetch_queue.get()
                if self._ingest_prefetch_item(item):
                    self._cleanup_after_drain()
                    return None
            else:
                if self._state.chunk_ix >= self._state.n_chunks:
                    self._cleanup_after_drain()
                    return None
                self._chunk_step()
        return self._state.deque.popleft()

    async def _produce(self) -> AxisArray | None:
        while not self._state.deque:
            if self._state.prefetch_thread is not None:
                if self._prefetch_drained:
                    self._cleanup_after_drain()
                    return None
                # Don't block the event loop on the queue.
                item = await asyncio.to_thread(self._state.prefetch_queue.get)
                if self._ingest_prefetch_item(item):
                    self._cleanup_after_drain()
                    return None
            else:
                if self._state.chunk_ix >= self._state.n_chunks:
                    self._cleanup_after_drain()
                    return None
                self._chunk_step()
        return self._state.deque.popleft()

    def _cleanup_after_drain(self) -> None:
        """Close the slicer once we've emitted the last message. Mirrors
        the close-on-exhaustion semantics of the original implementation.
        """
        self._stop_prefetch()
        # Pin chunk_ix at the end so a re-entry to ``_produce_sync`` /
        # ``_produce`` short-circuits via the sync end-of-stream check
        # (otherwise the prefetch path's exhaustion is invisible to it).
        self._state.chunk_ix = self._state.n_chunks
        self._prefetch_drained = True
        if self._state.slicer is not None:
            self._state.slicer.close()
            self._state.slicer = None

    def close(self) -> None:
        """Shut down the prefetch worker (if any) and close the slicer.

        Safe to call repeatedly. Prefer this over relying on ``__del__`` —
        in the rare cases where the iterator survives a ``del`` because some
        other reference is still alive, ``close()`` lets the caller drop
        resources deterministically rather than waiting for GC.
        """
        try:
            self._stop_prefetch()
        except Exception:
            pass
        if self._state.slicer is not None:
            self._state.slicer.close()
            self._state.slicer = None

    def __next__(self) -> AxisArray:
        # Fast path: skip BaseProducer.__call__'s run_coroutine_sync round-trip
        # on every call. _reset_state already ran in __init__ and set _hash=0.
        if self._hash == -1:
            self._reset_state()
            self._hash = 0
        result = self._produce_sync()
        if result is None:
            raise StopIteration
        return result

    def __del__(self):
        if not hasattr(self, "_state"):
            return
        # During interpreter finalization, daemon threads can be hung in
        # ``PyThread_hang_thread`` (CPython sets each thread's
        # ``_status.finalizing`` and ``take_gil`` then suspends them). If the
        # prefetch worker is mid-``H5Dread`` when this happens it will never
        # release HDF5's per-file lock, so calling ``h5py.File.close()`` here
        # deadlocks forever. The OS reclaims file handles on process exit
        # anyway, so during finalization just leave them. Since the worker
        # no longer captures ``self``, normal ``del it`` paths reach this
        # method *before* finalization — which is the path that actually
        # cleans up.
        if sys.is_finalizing():
            return
        self.close()
