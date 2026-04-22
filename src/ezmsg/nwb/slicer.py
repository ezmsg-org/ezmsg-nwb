"""NWBSlicer: Shared NWB file handling for iterators and clock-driven producers."""

from __future__ import annotations

import datetime
import math
import os
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import ezmsg.core as ez
import h5py
import numpy as np
import pynwb
import remfile
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .util import ReferenceClockType


@dataclass
class StreamInfo:
    """Per-stream metadata discovered from an NWB file."""

    dset: typing.Any
    """h5py dataset or numpy array containing the data."""

    template: AxisArray
    """AxisArray with zero-length time dim, used as a template for output messages."""

    fs: float = 0.0
    """Nominal sample rate (0 for events/irregular)."""

    t0: float = 0.0
    """Starting time (relative to file, before ts_off adjustment)."""

    n_samples: int = 0
    """Total sample count in the stream."""

    timestamps: typing.Any = None
    """h5py dataset or numpy array of explicit timestamps, or None for rate-only."""

    has_timestamps: bool = False
    """Whether the file has explicit per-sample timestamps for this stream."""

    is_event: bool = False
    """Whether this stream is an interval table (events) vs timeseries."""

    table_ref: typing.Any = None
    """Reference to the interval table (events only)."""


def _extract_timeseries_from_container(container, address: str | None = None) -> list[tuple[str, pynwb.TimeSeries]]:
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


class NWBSlicer:
    """Shared NWB file handling: open, discover streams, and slice data.

    Args:
        filepath: Path to NWB file (local or remote URL).
        reference_clock: Which reference clock to use for timestamps.
        reref_now: If True, re-reference timestamps to the current time.
        stream_keys: Optional filter for which streams to discover.
    """

    disk_cache = remfile.DiskCache(str(Path("~").expanduser() / ".ezmsg" / "nwb-cache"))

    def __init__(
        self,
        filepath: typing.Union[os.PathLike, str],
        reference_clock: ReferenceClockType = ReferenceClockType.SYSTEM,
        reref_now: bool = False,
        stream_keys: typing.Optional[list[str]] = None,
    ):
        self._filepath = filepath
        self._reference_clock = reference_clock
        self._reref_now = reref_now
        self._stream_keys = stream_keys

        self._io: pynwb.NWBHDF5IO | None = None
        self._ts_off: float = 0.0
        self._streams: dict[str, StreamInfo] = {}
        self._start_time: float = np.inf
        self._stop_time: float = -np.inf

        self._load()

    def _load(self) -> None:
        """Open the NWB file and discover all streams."""
        if str(self._filepath).startswith("http"):
            f = remfile.File(self._filepath, disk_cache=self.disk_cache)
            file = h5py.File(f)
            self._io = pynwb.NWBHDF5IO(file=file)
        else:
            self._io = pynwb.NWBHDF5IO(self._filepath, "r")
        nwbfile = self._io.read()

        # Compute timestamp offset
        if self._reref_now:
            if self._reference_clock == ReferenceClockType.SYSTEM:
                self._ts_off = time.time()
            elif self._reference_clock == ReferenceClockType.MONOTONIC:
                self._ts_off = time.monotonic()
            else:
                raise ValueError("Cannot re-reference to unknown clock.")
        else:
            if self._reference_clock in [ReferenceClockType.SYSTEM, ReferenceClockType.MONOTONIC]:
                local_dt = nwbfile.session_start_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
                self._ts_off = local_dt.timestamp()
                if self._reference_clock == ReferenceClockType.MONOTONIC:
                    self._ts_off += time.monotonic() - time.time()
            else:  # UNKNOWN
                self._ts_off = 0.0

        start_time = np.inf
        stop_time = -np.inf

        # Discover interval tables (trials, epochs, custom intervals)
        time_intervals = getattr(nwbfile, "intervals", None)
        if time_intervals is not None:
            for attr, table in time_intervals.items():
                if table is not None and (self._stream_keys is None or attr in self._stream_keys):
                    start_time = min(start_time, self._ts_off + table.start_time[0])
                    stop_time = max(stop_time, self._ts_off + table.stop_time[-1])
                    ch_labels = list(set(table.colnames) - {"start_time", "stop_time"})
                    dset = np.array([list(map(str, table[_].data)) for _ in ch_labels]).T
                    self._streams[attr] = StreamInfo(
                        dset=dset,
                        template=AxisArray(
                            data=np.zeros((0, len(ch_labels)), dtype=dset.dtype),
                            dims=["time", "ch"],
                            axes={
                                "time": AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s"),
                                "ch": AxisArray.CoordinateAxis(data=np.array(ch_labels), dims=["ch"]),
                            },
                            key=attr,
                        ),
                        fs=0.0,
                        t0=float(table.start_time[0]),
                        n_samples=len(table.start_time[:]),
                        timestamps=table.start_time[:],
                        has_timestamps=True,
                        is_event=True,
                        table_ref=table,
                    )

        # Discover timeseries
        all_timeseries = _extract_timeseries_from_container(nwbfile)
        for module_name, module in nwbfile.processing.items():
            all_timeseries.extend(_extract_timeseries_from_container(module, address=f"root/{module_name}"))
        all_timeseries = set(all_timeseries)  # Remove duplicates

        for address, child in all_timeseries:
            if type(child) is pynwb.misc.Units:
                ez.logger.warning("Units found in NWB file. Not yet supported.")
            elif isinstance(child, pynwb.TimeSeries) and (self._stream_keys is None or child.name in self._stream_keys):
                if child.data.size == 0:
                    ez.logger.warning(f"Skipping empty TimeSeries: {child.name} {type(child)}")
                    continue

                has_timestamps = hasattr(child, "timestamps") and child.timestamps is not None

                if has_timestamps:
                    # Determine nominal rate
                    if hasattr(child, "rate") and child.rate is not None:
                        rate = child.rate
                    elif "rate" in child.timestamps.attrs:
                        rate = child.timestamps.attrs["rate"]
                    else:
                        dts = np.diff(child.timestamps[:])
                        if np.var(dts) < 1e-3 or np.var(dts) < 0.05 * np.median(dts):
                            rate = 1 / np.median(dts)
                        else:
                            rate = 0.0

                    t0_val = child.timestamps[0]
                    start_time = min(start_time, self._ts_off + t0_val)
                    gain = 1 / rate if rate != 0 else 1.0
                    stop_time = max(stop_time, self._ts_off + child.timestamps[-1] + gain)
                    stop_time = max(
                        stop_time,
                        self._ts_off + t0_val + (child.data.shape[0] + 1) * gain,
                    )
                    tvec = child.timestamps
                else:
                    rate = child.rate
                    t0_val = child.starting_time
                    gain = 1 / rate if rate != 0 else 1.0
                    start_time = min(start_time, self._ts_off + t0_val)
                    stop_time = max(
                        stop_time,
                        self._ts_off + t0_val + (child.data.shape[0] + 1) * gain,
                    )
                    tvec = child.starting_time + np.arange(child.data.shape[0]) / rate

                # Build axes metadata
                axes: dict[str, typing.Any] = {}
                if math.isclose(rate, 0.0):
                    axes["time"] = AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s")
                else:
                    axes["time"] = AxisArray.LinearAxis.create_time_axis(fs=rate, offset=self._ts_off)
                if hasattr(child, "electrodes") and child.electrodes is not None:
                    # ``child.electrodes`` is a DynamicTableRegion whose
                    # ``.data`` holds the positional indices into the full
                    # electrodes table. Subset with iloc so the returned
                    # channel labels line up 1:1 with the data columns —
                    # otherwise an ElectricalSeries that references a
                    # strict subset of the electrodes table produces a
                    # ch-axis whose length does not match data.shape[1].
                    region_idx = np.asarray(child.electrodes.data)
                    full_df = child.electrodes.table.to_dataframe()
                    el_df = full_df.iloc[region_idx]
                    if "label" in el_df.columns:
                        ch_labels = el_df["label"].values.tolist()
                    else:
                        ch_labels = [f"ch_{idx}" for idx in el_df.index.tolist()]
                    axes["ch"] = AxisArray.CoordinateAxis(data=np.array(ch_labels), dims=["ch"])

                self._streams[child.name] = StreamInfo(
                    dset=child.data,
                    template=AxisArray(
                        data=np.zeros((0,) + child.data.shape[1:], dtype=child.data.dtype),
                        dims=(["time", "ch"] + [f"dim_{_}" for _ in range(2, child.data.ndim)])
                        if child.data.ndim > 1
                        else ["time"],
                        axes=axes,
                        key=child.name,
                    ),
                    fs=rate,
                    t0=(
                        child.starting_time
                        if (hasattr(child, "starting_time") and child.starting_time is not None)
                        else child.timestamps[0]
                    ),
                    n_samples=child.data.shape[0],
                    timestamps=tvec if has_timestamps else None,
                    has_timestamps=has_timestamps,
                    is_event=False,
                    table_ref=None,
                )

        self._start_time = start_time
        self._stop_time = stop_time

    # --- Public properties ---

    @property
    def ts_off(self) -> float:
        """Timestamp offset applied to all timestamps."""
        return self._ts_off

    @property
    def stream_names(self) -> list[str]:
        """Names of all discovered streams."""
        return list(self._streams.keys())

    @property
    def start_time(self) -> float:
        """Global start time across all streams (with ts_off applied)."""
        return self._start_time

    @property
    def stop_time(self) -> float:
        """Global stop time across all streams (with ts_off applied)."""
        return self._stop_time

    def get_stream_info(self, key: str) -> StreamInfo:
        """Return the StreamInfo for a given stream key."""
        return self._streams[key]

    # --- Slicing methods ---

    def read_by_index(self, stream_key: str, start_idx: int, stop_idx: int) -> AxisArray:
        """Read continuous data by sample index range.

        For rate-only continuous streams. Returns an AxisArray with data[start_idx:stop_idx].
        """
        info = self._streams[stream_key]
        out_data = info.dset[start_idx:stop_idx]
        template = info.template

        if info.timestamps is not None and start_idx < len(info.timestamps):
            chunk_t0 = info.timestamps[start_idx]
        else:
            chunk_t0 = template.axes["time"].gain * start_idx

        return replace(
            template,
            data=out_data,
            axes={
                **template.axes,
                "time": replace(
                    template.axes["time"],
                    offset=self._ts_off + chunk_t0,
                ),
            },
            key=stream_key,
        )

    def read_by_time(self, stream_key: str, t_start: float, t_end: float) -> AxisArray:
        """Read data by time window [t_start, t_end).

        For timestamped continuous streams and event/interval tables.
        t_start and t_end are in the same reference frame as the stored timestamps
        (i.e., file-relative, before ts_off).
        """
        info = self._streams[stream_key]
        template = info.template

        if info.is_event:
            # Event/interval table: timestamps are start_time values
            timestamps = info.timestamps
            start_idx = int(np.searchsorted(timestamps, t_start, side="left"))
            stop_idx = int(np.searchsorted(timestamps, t_end, side="left"))

            if start_idx >= stop_idx:
                return template  # Zero-length template

            # Return all events in the window as a single AxisArray
            out_data = info.dset[start_idx:stop_idx]
            table = info.table_ref
            event_times = self._ts_off + table.start_time[start_idx:stop_idx]

            return replace(
                template,
                data=out_data,
                axes={
                    **template.axes,
                    "time": replace(
                        template.axes["time"],
                        data=np.asarray(event_times),
                    ),
                },
                key=stream_key,
            )
        else:
            # Timestamped continuous stream
            if info.timestamps is None:
                raise ValueError(f"Stream '{stream_key}' has no explicit timestamps. Use read_by_index instead.")
            timestamps = info.timestamps
            # Materialize if h5py dataset for searchsorted
            if hasattr(timestamps, "shape") and not isinstance(timestamps, np.ndarray):
                ts_arr = timestamps[:]
            else:
                ts_arr = timestamps
            start_idx = int(np.searchsorted(ts_arr, t_start, side="left"))
            stop_idx = int(np.searchsorted(ts_arr, t_end, side="left"))

            out_data = info.dset[start_idx:stop_idx]

            if start_idx < len(ts_arr):
                chunk_t0 = ts_arr[start_idx]
            else:
                chunk_t0 = template.axes["time"].gain * start_idx if hasattr(template.axes["time"], "gain") else 0.0

            return replace(
                template,
                data=out_data,
                axes={
                    **template.axes,
                    "time": replace(
                        template.axes["time"],
                        offset=self._ts_off + chunk_t0,
                    ),
                },
                key=stream_key,
            )

    # --- Lifecycle ---

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        if self._io is not None:
            self._io.close()
            self._io = None

    def __del__(self) -> None:
        self.close()
