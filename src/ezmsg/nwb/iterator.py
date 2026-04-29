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

from .slicer import NWBSlicer
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
        start_idx = strm_dict["chunk_offsets"][chunk_ix]
        if chunk_ix + 1 < len(strm_dict["chunk_offsets"]):
            stop_idx = strm_dict["chunk_offsets"][chunk_ix + 1]
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
            if info.timestamps is not None and start_idx < len(info.timestamps):
                chunk_t0 = info.timestamps[start_idx]
            else:
                chunk_t0 = template.axes["time"].gain * start_idx
            out.append(
                replace(
                    template,
                    data=out_data,
                    axes={
                        **template.axes,
                        "time": replace(
                            template.axes["time"],
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
            msgs = _build_chunk_messages_static(slicer, streams, chunk_ix)
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
        try:
            q.put(exc, timeout=1.0)
        except queue.Full:
            pass
        return
    finally:
        # Always signal end-of-stream so the consumer wakes up.
        try:
            q.put(_PREFETCH_END, timeout=1.0)
        except queue.Full:
            pass


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
        )
        self._state.slicer = slicer

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
                samps_per_chunk = self.settings.chunk_dur / template.axes["time"].gain
                t0_abs = float(info.t0) + float(slicer.ts_off)
                first_chunk = max(0, int((t0_abs - float(start_time)) // self.settings.chunk_dur))
                chunk_ix_offsets = np.arange(n_chunks - first_chunk) * samps_per_chunk
                chunk_ix_offsets = chunk_ix_offsets.astype(int)

            self._state.streams[name] = {
                "info": info,
                "chunk_offsets": chunk_ix_offsets,
            }

        self._state.n_chunks = n_chunks

    def _build_chunk_messages(self, chunk_ix: int) -> list[AxisArray]:
        """Sync-side wrapper around :func:`_build_chunk_messages_static`."""
        return _build_chunk_messages_static(self._state.slicer, self._state.streams, chunk_ix)

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
