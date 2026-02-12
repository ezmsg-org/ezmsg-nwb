import datetime
import os
import time
import typing
from collections import deque
from pathlib import Path

import ezmsg.core as ez
import fsspec
import h5py
import numpy as np
import pynwb
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from fsspec.implementations.cached import CachingFileSystem

from .util import ReferenceClockType


class NWBIteratorSettings(ez.Settings):
    filepath: typing.Union[os.PathLike, str]
    chunk_dur: float = 1.0
    # start_time: typing.Optional[float] = None
    # stop_time: typing.Optional[float] = None
    reference_clock: ReferenceClockType = ReferenceClockType.SYSTEM
    reref_now: bool = False
    self_terminating: bool = True


class NWBAxisArrayIterator:
    _fs: typing.ClassVar[typing.Optional[CachingFileSystem]] = None

    @classmethod
    def _get_fs(cls) -> CachingFileSystem:
        if cls._fs is None:
            cls._fs = CachingFileSystem(
                fs=fsspec.filesystem("http"),
                cache_storage=str(Path("~").expanduser() / ".ezmsg" / "nwb-cache"),
            )
        return cls._fs

    def __init__(self, settings: NWBIteratorSettings):
        self._settings = settings
        self._n_chunks = 0
        self._chunk_ix = 0
        self._last_time = 0.0
        self._io: typing.Optional[pynwb.NWBHDF5IO] = None
        self._ts_off = 0.0
        self._streams = {}
        self._force_single_sample = set()
        self._deque = deque()
        self._preload()

    def _preload(self):
        self._streams = {}
        if str(self._settings.filepath).startswith("http"):
            # If filepath is URL then use fsspec
            f = h5py.File(self._get_fs().open(self._settings.filepath, "rb"))
            self._io = pynwb.NWBHDF5IO(file=f)
        else:
            self._io = pynwb.NWBHDF5IO(self._settings.filepath, "r")
        nwbfile = self._io.read()

        # Determine the offset to apply to timestamps. The NWB file session_start_time is a datetime that represents
        #  the first timestamp. We convert this time to one of a few reference clocks then add that value to all
        #  timestamps.
        if self._settings.reref_now:
            if self._settings.reference_clock == ReferenceClockType.SYSTEM:
                self._ts_off = time.time()
            elif self._settings.reference_clock == ReferenceClockType.MONOTONIC:
                self._ts_off = time.monotonic()
            else:
                raise ValueError("Cannot re-reference to unknown clock.")
        else:
            if self._settings.reference_clock in [
                ReferenceClockType.SYSTEM,
                ReferenceClockType.MONOTONIC,
            ]:
                local_dt = nwbfile.session_start_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
                self._ts_off = local_dt.timestamp()
                if self._settings.reference_clock == ReferenceClockType.MONOTONIC:
                    self._ts_off += time.monotonic() - time.time()
            else:  # UNKNOWN. Do not adjust.
                self._ts_off = 0.0

        # For each (supported) data source, update the common min-max timestamp range,
        #  grab a reference to its dataset, and create a template AxisArray
        #  message for the output.
        start_time = np.inf
        stop_time = -np.inf

        # First for trials and epochs tables.
        for attr in ["trials", "epochs"]:
            if hasattr(nwbfile, attr) and getattr(nwbfile, attr) is not None:
                table = getattr(nwbfile, attr)
                start_time = min(start_time, self._ts_off + table.start_time[0])
                stop_time = max(stop_time, self._ts_off + table.start_time[-1])
                # For tables, dset cannot be lazily loaded.
                ch_labels = list(set(table.colnames) - {"start_time", "stop_time"})
                dset = np.array([list(map(str, table[_].data)) for _ in ch_labels]).T
                self._streams[attr] = {
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

        # Next for all other supported datasets
        for child in nwbfile.children:
            if type(child) is pynwb.misc.Units:
                ez.logger.warning("Units found in NWB file. Not yet supported.")
            elif isinstance(child, pynwb.TimeSeries):
                if child.data.size == 0:
                    ez.logger.warning(f"Skipping empty TimeSeries: {child.name} {type(child)}")
                    continue
                if hasattr(child, "timestamps") and child.timestamps is not None:
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
                    start_time = min(start_time, self._ts_off + child.timestamps[0])
                    # Chunking will be based on span of start-stop time and the nominal rate.
                    gain = 1 / rate if rate != 0 else 1.0
                    stop_time = max(stop_time, self._ts_off + child.timestamps[-1] + gain)
                    # However, the timestamps might be misleading so we need to expand our stop_time to make
                    #  sure we capture all samples when we assume the nominal rate.
                    stop_time = max(
                        stop_time,
                        self._ts_off + child.timestamps[0] + (child.data.shape[0] + 1) * gain,
                    )
                else:
                    # Timestamps are trivially generated from the rate and starting_time.
                    rate = child.rate
                    gain = 1 / rate if rate != 0 else 1.0
                    start_time = min(start_time, self._ts_off + child.starting_time)
                    stop_time = max(
                        stop_time,
                        self._ts_off + child.starting_time + (child.data.shape[0] + 1) * gain,
                    )

                # Build up axes metadata
                axes = {}
                if rate == 0.0:
                    axes["time"] = AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s")
                else:
                    axes["time"] = AxisArray.Axis.TimeAxis(fs=rate, offset=self._ts_off)
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
                self._streams[child.name] = {
                    "t0": (
                        child.starting_time
                        if (hasattr(child, "starting_time") and child.starting_time is not None)
                        else child.timestamps[0]
                    ),
                    "dset": child.data,
                    "timestamps": tvec,
                    "template": AxisArray(
                        data=np.zeros((0,) + child.data.shape[1:], dtype=child.data.dtype),
                        dims=["time", "ch"] + [f"dim_{_}" for _ in range(2, child.data.ndim)],
                        axes=axes,
                        key=child.name,
                    ),
                }

        # TODO: Use self._settings.start_time and .stop_time to constrain the ranges of chunks.
        # start_time = max(start_time, self._settings.start_time)
        # stop_time = min(stop_time, self._settings.stop_time)
        t_range = stop_time - start_time
        n_chunks = int(np.ceil(t_range / self._settings.chunk_dur))

        # For each stream, for each chunk, identify the first sample index for that chunk.
        for n, strm in self._streams.items():
            if hasattr(strm["template"].axes["time"], "data"):
                # Irregular interval stream
                timestamps = getattr(nwbfile, n).start_time[:]
                # For each sample, find the first chunk in which it appears, index +1,
                #  then set chunk_ix[x:] to that sample index.
                chunk_ix_offsets = np.zeros(n_chunks, dtype=int)
                for samp_ix, ts in enumerate(timestamps):
                    chunk_ix = int((ts + self._ts_off - start_time) // self._settings.chunk_dur)
                    chunk_ix_offsets[chunk_ix + 1 :] = samp_ix + 1
            else:
                samps_per_chunk = self._settings.chunk_dur / strm["template"].axes["time"].gain
                first_chunk = max(0, int(strm["t0"] // self._settings.chunk_dur))
                chunk_ix_offsets = np.arange(n_chunks - first_chunk) * samps_per_chunk
                chunk_ix_offsets = np.hstack((np.zeros(first_chunk), chunk_ix_offsets)).astype(int)

            strm["chunk_offsets"] = chunk_ix_offsets

        self._n_chunks = n_chunks

    def __iter__(self):
        self._chunk_ix = 0
        self._preload()
        return self

    def _chunk_step(self):
        for strm_name, strm_dict in self._streams.items():
            start_idx = strm_dict["chunk_offsets"][self._chunk_ix]
            if self._chunk_ix + 1 < len(strm_dict["chunk_offsets"]):
                stop_idx = strm_dict["chunk_offsets"][self._chunk_ix + 1]
            else:
                stop_idx = strm_dict["dset"].shape[0]
            if stop_idx != start_idx:
                if hasattr(strm_dict["template"].axes["time"], "data"):
                    # Irregular time intervals.
                    table = strm_dict["table_ref"]
                    # Sample-by-sample
                    for idx in range(start_idx, stop_idx):
                        self._deque.append(
                            replace(
                                strm_dict["template"],
                                data=strm_dict["dset"][idx : idx + 1],
                                axes={
                                    **strm_dict["template"].axes,
                                    "time": replace(
                                        strm_dict["template"].axes["time"],
                                        data=self._ts_off + table.start_time[idx : idx + 1],
                                    ),
                                },
                                key=strm_name,
                            )
                        )
                else:
                    out_data = strm_dict["dset"][start_idx:stop_idx]
                    if out_data.size:
                        if len(strm_dict["timestamps"]) > start_idx:
                            chunk_t0 = strm_dict["timestamps"][start_idx]
                        else:
                            chunk_t0 = strm_dict["template"].axes["time"].gain * start_idx
                        self._deque.append(
                            replace(
                                strm_dict["template"],
                                data=out_data,
                                axes={
                                    **strm_dict["template"].axes,
                                    "time": replace(
                                        strm_dict["template"].axes["time"],
                                        offset=self._ts_off + chunk_t0,
                                    ),
                                },
                                key=strm_name,
                            ),
                        )
        self._chunk_ix += 1

    def __next__(self) -> AxisArray:
        if not self._deque:
            # Deque is empty.
            if self._chunk_ix >= self._n_chunks:
                # No more chunks.
                if self._io is not None:
                    self._io.close()
                raise StopIteration
            # Load the next chunk.
            self._chunk_step()

        if not self._deque:
            # Still no data.
            raise StopIteration

        return self._deque.popleft()

    def __del__(self):
        if self._io is not None:
            self._io.close()
