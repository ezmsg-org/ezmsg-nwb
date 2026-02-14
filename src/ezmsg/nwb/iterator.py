from __future__ import annotations

import datetime
import math
import os
import time
import typing
from collections import deque
from pathlib import Path

import ezmsg.core as ez
import h5py
import numpy as np
import pynwb
import remfile
from ezmsg.baseproc.protocols import processor_state
from ezmsg.baseproc.stateful import BaseStatefulProducer
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

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
    io: pynwb.NWBHDF5IO | None = None
    ts_off: float = 0.0
    streams: dict | None = None
    deque: deque | None = None


def _extract_timeseries_from_container(
    container, address: str | None = None
) -> list[tuple[str, pynwb.TimeSeries]]:
    """Recursively extract all TimeSeries objects from a container."""
    if address is None:
        address = container.name
    timeseries = []
    for obj in container.children:
        joint_address = f"{address}/{obj.name}"
        if isinstance(obj, pynwb.TimeSeries):
            timeseries.append((joint_address, obj))
        elif hasattr(obj, "children") and len(obj.children) > 0:
            timeseries.extend(_extract_timeseries_from_container(obj, joint_address))
    return timeseries


class NWBAxisArrayIterator(BaseStatefulProducer[NWBIteratorSettings, AxisArray, NWBIteratorState]):
    disk_cache = remfile.DiskCache(str(Path("~").expanduser() / ".ezmsg" / "nwb-cache"))

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
        self._state.ts_off = 0.0
        self._state.streams = {}
        self._state.deque = deque()

        if self._state.io is not None:
            self._state.io.close()
            self._state.io = None

        self._preload()

    def _preload(self):
        self._state.streams = {}
        if str(self.settings.filepath).startswith("http"):
            # If filepath is URL then use remfile
            f = remfile.File(self.settings.filepath, disk_cache=self.disk_cache)
            file = h5py.File(f)
            self._state.io = pynwb.NWBHDF5IO(file=file)
        else:
            # Provide the path, not the file. This is different to the remfile usage.
            self._state.io = pynwb.NWBHDF5IO(self.settings.filepath, "r")
        nwbfile = self._state.io.read()

        # Determine the offset to apply to timestamps. The NWB file session_start_time is a datetime that represents
        #  the first timestamp. We convert this time to one of a few reference clocks then add that value to all
        #  timestamps.
        if self.settings.reref_now:
            if self.settings.reference_clock == ReferenceClockType.SYSTEM:
                self._state.ts_off = time.time()
            elif self.settings.reference_clock == ReferenceClockType.MONOTONIC:
                self._state.ts_off = time.monotonic()
            else:
                raise ValueError("Cannot re-reference to unknown clock.")
        else:
            if self.settings.reference_clock in [
                ReferenceClockType.SYSTEM,
                ReferenceClockType.MONOTONIC,
            ]:
                local_dt = nwbfile.session_start_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
                self._state.ts_off = local_dt.timestamp()
                if self.settings.reference_clock == ReferenceClockType.MONOTONIC:
                    self._state.ts_off += time.monotonic() - time.time()
            else:  # UNKNOWN. Do not adjust.
                self._state.ts_off = 0.0

        # For each (supported) data source, update the common min-max timestamp range,
        #  grab a reference to its dataset, and create a template AxisArray
        #  message for the output.
        start_time = np.inf
        stop_time = -np.inf

        # First for time interval tables (trials, epochs, and any custom intervals).
        time_intervals = getattr(nwbfile, "intervals", None)
        if time_intervals is not None:
            for attr, table in time_intervals.items():
                if table is not None and (self.settings.stream_keys is None or attr in self.settings.stream_keys):
                    start_time = min(start_time, self._state.ts_off + table.start_time[0])
                    stop_time = max(stop_time, self._state.ts_off + table.stop_time[-1])  # Was start_time. stop_time OK?
                    # For tables, dset cannot be lazily loaded.
                    ch_labels = list(set(table.colnames) - {"start_time", "stop_time"})
                    dset = np.array([list(map(str, table[_].data)) for _ in ch_labels]).T
                    self._state.streams[attr] = {
                        "dset": dset,
                        "template": AxisArray(
                            data=np.zeros((0, len(ch_labels)), dtype=dset.dtype),
                            dims=["time", "ch"],
                            axes={
                                "time": AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s"),
                                "ch": AxisArray.CoordinateAxis(data=np.array(ch_labels), dims=["ch"]),
                            },
                            key=attr,
                        ),
                        "table_ref": table,
                    }

        # Next for timeseries — recursively extract from root and processing modules.
        all_timeseries = _extract_timeseries_from_container(nwbfile)
        # Also explicitly scan processing modules in case the NWB structure doesn't
        # expose them as root children.
        for module_name, module in nwbfile.processing.items():
            all_timeseries.extend(_extract_timeseries_from_container(module, address=f"root/{module_name}"))
        all_timeseries = set(all_timeseries)  # Remove duplicates
        for address, child in all_timeseries:
            if type(child) is pynwb.misc.Units:
                ez.logger.warning("Units found in NWB file. Not yet supported.")
            elif isinstance(child, pynwb.TimeSeries) and (
                self.settings.stream_keys is None or child.name in self.settings.stream_keys
            ):
                if child.data.size == 0:
                    ez.logger.warning(f"Skipping empty TimeSeries: {child.name} {type(child)}")
                    continue
                if hasattr(child, "timestamps") and child.timestamps is not None:
                    # The timeseries has timestamps for each sample. This might be because the
                    #  timeseries is irregularly sampled, or because we purposely stored timestamps
                    #  for more accuracy.
                    # Despite having all timestamps, we still want to know the nominal rate.
                    #  The rate is usually not included when timestamps are provided, but some of our custom tools
                    #  hide the rate when creating the NWB file -- retrieve that.
                    #  Otherwise, calculate the rate from the timestamps.
                    if hasattr(child, "rate") and child.rate is not None:
                        rate = child.rate
                    elif "rate" in child.timestamps.attrs:
                        rate = child.timestamps.attrs["rate"]
                    else:
                        dts = np.diff(child.timestamps[:])  # Have to materialize all timestamps.
                        if np.var(dts) < 1e-3 or np.var(dts) < 0.05 * np.median(dts):
                            rate = 1 / np.median(dts)
                        else:
                            rate = 0.0
                    start_time = min(start_time, self._state.ts_off + child.timestamps[0])
                    # Chunking will be based on span of start-stop time and the nominal rate.
                    gain = 1 / rate if rate != 0 else 1.0
                    stop_time = max(stop_time, self._state.ts_off + child.timestamps[-1] + gain)
                    # However, the timestamps might be misleading so we need to expand our stop_time to make
                    #  sure we capture all samples when we assume the nominal rate.
                    stop_time = max(
                        stop_time,
                        self._state.ts_off + child.timestamps[0] + (child.data.shape[0] + 1) * gain,
                    )
                else:
                    # Timestamps are trivially generated from the rate and starting_time.
                    rate = child.rate
                    gain = 1 / rate if rate != 0 else 1.0
                    start_time = min(start_time, self._state.ts_off + child.starting_time)
                    stop_time = max(
                        stop_time,
                        self._state.ts_off + child.starting_time + (child.data.shape[0] + 1) * gain,
                    )

                # Build up axes metadata
                axes = {}
                if math.isclose(rate, 0.0):
                    axes["time"] = AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s")
                else:
                    axes["time"] = AxisArray.LinearAxis.create_time_axis(fs=rate, offset=self._state.ts_off)
                if hasattr(child, "electrodes"):
                    el_inds = child.electrodes.table.id[:]
                    el_df = child.electrodes.table.to_dataframe().loc[el_inds]
                    if "label" in el_df.columns:
                        ch_labels = el_df["label"].values.tolist()
                    else:
                        ch_labels = [f"ch_{_}" for _ in el_inds]
                    axes["ch"] = AxisArray.CoordinateAxis(data=np.array(ch_labels), dims=["ch"])
                if hasattr(child, "timestamps") and child.timestamps is not None:
                    tvec = child.timestamps
                else:
                    tvec = child.starting_time + np.arange(child.data.shape[0]) / rate
                self._state.streams[child.name] = {
                    "t0": (
                        child.starting_time
                        if (hasattr(child, "starting_time") and child.starting_time is not None)
                        else child.timestamps[0]
                    ),
                    "dset": child.data,
                    "timestamps": tvec,
                    "template": AxisArray(
                        data=np.zeros((0,) + child.data.shape[1:], dtype=child.data.dtype),
                        dims=(["time", "ch"] + [f"dim_{_}" for _ in range(2, child.data.ndim)])
                        if child.data.ndim > 1
                        else ["time"],
                        axes=axes,
                        key=child.name,
                    ),
                }

        # TODO: Use self.settings.start_time and .stop_time to constrain the ranges of chunks.
        # start_time = max(start_time, self.settings.start_time)
        # stop_time = min(stop_time, self.settings.stop_time)
        t_range = stop_time - start_time
        n_chunks = int(np.ceil(t_range / self.settings.chunk_dur))

        # For each stream, for each chunk, identify the first sample index for that chunk.
        for n, strm in self._state.streams.items():
            if hasattr(strm["template"].axes["time"], "data"):
                # Irregular interval stream — find first sample index in each chunk.
                timestamps = time_intervals[n].start_time[:]
                # Build the boundary times for each chunk and searchsorted to find the
                # first sample that falls at or after each boundary.
                chunk_boundaries = start_time + np.arange(n_chunks) * self.settings.chunk_dur - self._state.ts_off
                chunk_ix_offsets = np.searchsorted(timestamps, chunk_boundaries, side="left").astype(int)
            else:
                samps_per_chunk = self.settings.chunk_dur / strm["template"].axes["time"].gain
                t0_abs = float(strm["t0"]) + float(self._state.ts_off)
                first_chunk = max(0, int((t0_abs - float(start_time)) // self.settings.chunk_dur))
                chunk_ix_offsets = np.arange(n_chunks - first_chunk) * samps_per_chunk
                chunk_ix_offsets = chunk_ix_offsets.astype(int)

            strm["chunk_offsets"] = chunk_ix_offsets

        self._state.n_chunks = n_chunks

    def _chunk_step(self):
        for strm_name, strm_dict in self._state.streams.items():
            start_idx = strm_dict["chunk_offsets"][self._state.chunk_ix]
            if self._state.chunk_ix + 1 < len(strm_dict["chunk_offsets"]):
                stop_idx = strm_dict["chunk_offsets"][self._state.chunk_ix + 1]
            else:
                stop_idx = strm_dict["dset"].shape[0]
            template = strm_dict["template"]

            if hasattr(template.axes["time"], "data"):
                # Irregular time intervals — one message per event,
                # or one zero-length template if no events in this chunk.
                if start_idx < stop_idx:
                    table = strm_dict["table_ref"]
                    for idx in range(start_idx, stop_idx):
                        self._state.deque.append(
                            replace(
                                template,
                                data=strm_dict["dset"][idx : idx + 1],
                                axes={
                                    **template.axes,
                                    "time": replace(
                                        template.axes["time"],
                                        data=self._state.ts_off + table.start_time[idx : idx + 1],
                                    ),
                                },
                                key=strm_name,
                            )
                        )
                else:
                    self._state.deque.append(template)
            else:
                # Continuous — one message per chunk (possibly zero-length).
                out_data = strm_dict["dset"][start_idx:stop_idx]
                if start_idx < len(strm_dict["timestamps"]):
                    chunk_t0 = strm_dict["timestamps"][start_idx]
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
                                offset=self._state.ts_off + chunk_t0,
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
                if self._state.io is not None:
                    self._state.io.close()
                    self._state.io = None
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
        if hasattr(self, "_state") and self._state.io is not None:
            self._state.io.close()
