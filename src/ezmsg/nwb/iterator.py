from __future__ import annotations

import os
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


class NWBIteratorSettings(ez.Settings):
    filepath: typing.Union[os.PathLike, str]
    chunk_dur: float = 1.0
    # start_time: typing.Optional[float] = None
    # stop_time: typing.Optional[float] = None
    reference_clock: ReferenceClockType = ReferenceClockType.SYSTEM
    reref_now: bool = False
    self_terminating: bool = True
    stream_keys: typing.Optional[list[str]] = None


@processor_state
class NWBIteratorState:
    n_chunks: int = 0
    chunk_ix: int = 0
    slicer: NWBSlicer | None = None
    streams: dict | None = None
    deque: deque | None = None


class NWBAxisArrayIterator(BaseStatefulProducer[NWBIteratorSettings, AxisArray, NWBIteratorState]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Eagerly initialize state (load NWB file metadata) so that
        # _state.streams is available immediately after construction.
        self._reset_state()
        self._hash = 0

    @property
    def exhausted(self) -> bool:
        return self._state.chunk_ix >= self._state.n_chunks and not self._state.deque

    def _reset_state(self) -> None:
        self._state.n_chunks = 0
        self._state.chunk_ix = 0
        self._state.streams = {}
        self._state.deque = deque()

        if self._state.slicer is not None:
            self._state.slicer.close()
            self._state.slicer = None

        self._preload()

    def _preload(self):
        slicer = NWBSlicer(
            filepath=self.settings.filepath,
            reference_clock=self.settings.reference_clock,
            reref_now=self.settings.reref_now,
            stream_keys=self.settings.stream_keys,
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

    def _chunk_step(self):
        slicer = self._state.slicer
        for strm_name, strm_dict in self._state.streams.items():
            info = strm_dict["info"]
            start_idx = strm_dict["chunk_offsets"][self._state.chunk_ix]
            if self._state.chunk_ix + 1 < len(strm_dict["chunk_offsets"]):
                stop_idx = strm_dict["chunk_offsets"][self._state.chunk_ix + 1]
            else:
                stop_idx = info.dset.shape[0]
            template = info.template

            if info.is_event:
                # Irregular time intervals — one message per event,
                # or one zero-length template if no events in this chunk.
                if start_idx < stop_idx:
                    table = info.table_ref
                    for idx in range(start_idx, stop_idx):
                        self._state.deque.append(
                            replace(
                                template,
                                data=info.dset[idx : idx + 1],
                                axes={
                                    **template.axes,
                                    "time": replace(
                                        template.axes["time"],
                                        data=slicer.ts_off + table.start_time[idx : idx + 1],
                                    ),
                                },
                                key=strm_name,
                            )
                        )
                else:
                    self._state.deque.append(template)
            else:
                # Continuous — one message per chunk (possibly zero-length).
                out_data = info.dset[start_idx:stop_idx]
                if info.timestamps is not None and start_idx < len(info.timestamps):
                    chunk_t0 = info.timestamps[start_idx]
                else:
                    chunk_t0 = template.axes["time"].gain * start_idx
                self._state.deque.append(
                    replace(
                        template,
                        data=out_data,
                        axes={
                            **template.axes,
                            "time": replace(
                                template.axes["time"],
                                offset=slicer.ts_off + chunk_t0,
                            ),
                        },
                        key=strm_name,
                    ),
                )
        self._state.chunk_ix += 1

    async def _produce(self) -> AxisArray | None:
        if not self._state.deque:
            if self._state.chunk_ix >= self._state.n_chunks:
                # No more chunks.
                if self._state.slicer is not None:
                    self._state.slicer.close()
                    self._state.slicer = None
                return None
            # Load the next chunk.
            self._chunk_step()
        return self._state.deque.popleft()

    def __next__(self) -> AxisArray:
        result = self()
        if result is None:
            raise StopIteration
        return result

    def __del__(self):
        if hasattr(self, "_state") and self._state.slicer is not None:
            self._state.slicer.close()
