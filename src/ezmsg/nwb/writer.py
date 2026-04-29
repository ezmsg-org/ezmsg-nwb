"""
This module provides a sink node that writes incoming data to an NWB file.

Iteratively writing data to an NWB file has a few quirks.

The main approach to iterative writing in the docs -- wrapping the data source in a DataChunkIterator -- does not work
 well for our scenario. First, we would need to use some sort of queues to communicate between the data-receiving
 function and the thread / task that is yielding data to the iterator; this seems doable. Second, it's not clear if
 each DataChunkIterator must exhaust itself before moving onto the next one or if they can run in parallel; this is a
 deal-breaker for us. Third, it seems impossible to add new TimeSeries after the io.write(nwbfile) process has begun;
 this is also a deal-breaker for us.

Instead, we use the "user-defined dataset write" approach, described
`here <https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/plot_iterative_write.html#alternative-approach-user-defined-dataset-write>`_.
With this approach, we use pynwb to construct the NWB structure, but we specify the data containers (both data and
timestamps) as H5DataIO objects. After the NWB structure is written to disk, the containers expose h5py Dataset objects
that we can resize and append to as data arrive.

This approach has some quirks that we have worked around and landed on the following strategies:

A - When all incoming streams are known *a priori*

* Load metadata (shape, rate, channel labels) from a yaml or json file.
* Create all H5DataIO objects for epochs and trials: id, start_time, stop_time, and custom "label" column.
  * Append dummy event to table. Remove this step after https://github.com/hdmf-dev/hdmf/issues/1000
* Create device, electrode group, and add all known electrodes.
* Create all H5DataIO objects for continuous data (.data and .timestamps)
* Attach .data and .timestamps to new TimeSeries, or new ElectricalSeries if electrodes are known.
* Flush the NWBFile to disk -- Slow!! but necessary to create h5py Datasets
* Reopen the NWBFile in append mode -- Fast; necessary to make tables appendable, but freezes the electrodes table.
* Add any .rate attributes to the timestamps h5py datasets.
* Get references to the .data and .timestamps datasets for each continuous data appending.

B - When streams are not known *a priori*

* For the first message of a stream...
  * If it is continuous data, even size-zero
    * Create the H5DataIO objects for .data and .timestamps
    * If it is the FIRST continuous data stream with channel labels then create the device, elec group and electrodes
        * "FIRST" limitation should go away: https://github.com/hdmf-dev/hdmf/issues/1181
    * Attach .data and .timestamps to TimeSeries, or ElectricalSeries if electrodes are known (i.e. FIRST).
    * Flush the NWBFile to disk -- Slow!! but necessary to create h5py Datasets
    * Add any .rate attributes to the timestamps h5py datasets

  * If it is an event...
    * Create all H5DataIO objects for epochs and trials: id, start_time, stop_time, and custom "label" column.
    * Append dummy event to table. Remove this step after https://github.com/hdmf-dev/hdmf/issues/1000
    * Flush the NWBFile to disk -- Slow!! but necessary to create h5py Datasets
    * Reopen the NWBFile in append mode -- Fast; necessary to make tables appendable, but freezes the electrodes table.

In this case, we must call io.write(nwbfile) once for each stream, which can significantly slow down performance
for 1.5-4 seconds * number of unique streams.
Thus, we strongly recommend running NWBSink in a separate process where possible.

C - When closing a file on quit or when splitting

* When done appending to datasets and events tables, Flush the NWB file once more -- Slow!!
* If the file has no data other than the dummy event and recording is disabled, delete the file.

D - When opening the *next* file when splitting

* Copy as much information from the old NWBFile to the new NWBFile as possible.
* Close as in C.
* Follow pattern A where the metadata comes from the old datasets instead of loaded from a file.

"""

import asyncio
import dataclasses
import datetime
import os
import re
import threading
import time
import typing
import warnings
from pathlib import Path
from uuid import uuid4

import ezmsg.core as ez
import h5py
import numpy as np
import pynwb
from ezmsg.baseproc import BaseConsumerUnit, BaseStatefulConsumer, processor_state
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from hdmf.backends.hdf5.h5_utils import H5DataIO
from neuroconv.utils import DeepDict, dict_deep_update, load_dict_from_file

from .util import (
    ReferenceClockType,
    build_nwb_fname,
    flatten_component_settings,
)

try:
    from ezmsg.baseproc import SampleTriggerMessage

    _HAS_SAMPLE_TRIGGER = True
except ImportError:
    _HAS_SAMPLE_TRIGGER = False


class NWBSinkSettings(ez.Settings):
    filepath: typing.Union[str, os.PathLike]
    overwrite_old: bool = False
    axis: str = "time"
    recording: bool = True
    inc_clock: ReferenceClockType = ReferenceClockType.SYSTEM
    meta_yaml: typing.Optional[typing.Union[str, os.PathLike]] = None
    split_bytes: int = 0
    expected_series: typing.Optional[typing.Union[str, os.PathLike]] = None


@dataclasses.dataclass
class SeriesState:
    """Per-stream bookkeeping for one NWB series.

    ``shape`` is the trailing (non-time) shape used by
    :meth:`NWBSinkConsumer._check_msg_consistency`. ``data`` and ``ts`` hold
    the live ``h5py.Dataset`` (or pynwb container) references used by
    :meth:`NWBSinkConsumer._process` to append rows. ``bytes_written``
    tracks cumulative append size for the file-split threshold.
    """

    shape: typing.Tuple[int, ...]
    data: typing.Any = None
    ts: typing.Any = None
    bytes_written: int = 0


@processor_state
class NWBSinkState:
    hash: int = -1
    filepath: typing.Optional[Path] = None
    io: typing.Optional[pynwb.NWBHDF5IO] = None
    nwbfile: typing.Optional[pynwb.NWBFile] = None
    series: typing.Optional[typing.Dict[str, SeriesState]] = None
    start_timestamp: float = 0.0
    split_count: int = 0
    # Pipeline-settings tracker (per-file). Cleared on every _reset_state
    # so a file swap starts with an empty table; the unit re-seeds via
    # _pending_settings_seed (option B). Watcher updates are serialized
    # against _process via NWBSinkConsumer._lock.
    settings_columns: typing.Optional[typing.List[str]] = None
    settings_state: typing.Optional[typing.Dict[str, typing.Any]] = None
    settings_active_since: typing.Optional[float] = None
    settings_prev_component: str = "__init__"


class NWBSinkConsumer(BaseStatefulConsumer[NWBSinkSettings, AxisArray, NWBSinkState]):
    # Session start datetime. It should have a valid timezone and that should be UTC.
    shared_start_datetime: typing.Optional[datetime.datetime] = None
    # Session start time.time. It does not have a timezone but unqualified conversions assume local time.
    shared_t0: typing.Optional[float] = None
    shared_clock_type: typing.Optional[ReferenceClockType] = None

    # Fields that can change without reconstructing the NWB file. Both are
    # read live from ``self.settings`` inside ``_process``. Everything else
    # (filepath, axis, inc_clock, meta_yaml, expected_series, overwrite_old)
    # is consumed in ``_reset_state`` side effects; ``update_settings``
    # requests a reset so the next message closes the old file and reopens.
    NONRESET_SETTINGS_FIELDS = frozenset({"recording", "split_bytes"})

    # Name of the per-file pipeline-settings TimeIntervals table.
    SETTINGS_TABLE_NAME: typing.ClassVar[str] = "pipeline_settings"

    def __init__(self, *args, settings: typing.Optional[NWBSinkSettings] = None, **kwargs):
        super().__init__(*args, settings=settings, **kwargs)
        # Serializes _process appends and update_settings_table calls from
        # the unit's GraphContext watcher task; HDF5 writes from two threads
        # to the same file are unsafe.
        self._lock = threading.RLock()
        # When the unit pre-fetches a graph settings snapshot in
        # ``on_settings``, it stashes it here so the next ``_reset_state``
        # (which opens the new file) can seed the table on the new file.
        # See option (B) re-seed-on-swap design.
        self._pending_settings_seed: typing.Optional[typing.Tuple[typing.Dict[str, typing.Any], float]] = None
        # ``_current_msg`` is scratch space shared between ``_process`` and
        # helpers like ``_prep_continuous_io`` / ``_prep_from_meta``. Deferred
        # migration into state — see _prep_from_meta cleanup follow-up.
        self._current_msg: typing.Optional[AxisArray] = None
        # Eagerly open the file so construction-time errors (file exists,
        # permission denied, bad metadata yaml) surface immediately rather
        # than on the first message. Match _hash_message() so the first
        # inbound message does not re-trigger reset.
        self._reset_state(None)
        self._hash = 0

    def _hash_message(self, message: typing.Optional[AxisArray]) -> int:
        # Reset is driven exclusively by settings changes via
        # ``_request_reset``; message identity does not force a rebuild.
        return 0

    def _reset_state(self, message: typing.Optional[AxisArray]) -> None:
        """(Re)open the NWB file using the current settings.

        Called at construction (with ``None``) and again after
        ``update_settings`` flags a reset, which happens whenever any non-
        ``NONRESET_SETTINGS_FIELDS`` field changes. In the reset case the
        prior file is flushed and closed before the new one opens.
        """
        # Settings-triggered reset: flush the prior file before rebuilding.
        if self._state.io is not None:
            self.close(write=True)

        self._state.filepath = Path(self.settings.filepath)
        self._state.series = {}
        self._state.start_timestamp = 0.0
        self._state.split_count = 0
        # Per-file pipeline-settings tracker resets on every reset; the
        # unit's option-B re-seed populates it again below if a snapshot
        # was stashed before this reset was triggered.
        self._state.settings_columns = []
        self._state.settings_state = {}
        self._state.settings_active_since = None
        self._state.settings_prev_component = "__init__"

        self._check_filepath()
        self._nwb_create_or_fail()

        if self.settings.expected_series is not None and Path(self.settings.expected_series).expanduser().exists():
            expected_series = Path(self.settings.expected_series).expanduser()
            meta = load_dict_from_file(expected_series)
            _ = self.get_session_datetime(None)
            self._state.start_timestamp = self.get_session_timestamp(None)
            self._prep_from_meta(meta)

        # Option (B): if the unit pre-fetched a graph settings snapshot
        # before triggering this reset, seed the new file's settings table.
        if self._pending_settings_seed is not None:
            flat, ts = self._pending_settings_seed
            self._pending_settings_seed = None
            if flat:
                self.initialize_settings_table(flat, ts)

    def __del__(self):
        if not hasattr(self, "_state"):
            return
        self.close(write=False, log=False)

    async def _aprocess(self, message: AxisArray) -> None:
        """Run _process in a thread since NWB I/O can be slow."""
        await asyncio.to_thread(self._process, message)

    def _process(self, message: AxisArray) -> None:
        with self._lock:
            self._current_msg = message

            # Adjust incoming data
            if _HAS_SAMPLE_TRIGGER and isinstance(self._current_msg, SampleTriggerMessage):
                # SampleTriggerMessage. Rewrite as AxisArray.
                timestamp = self._current_msg.timestamp
                period = self._current_msg.period
                if period is not None and len(period) > 0:
                    timestamp = timestamp + period[0]
                # Wrap value in a 2D shape (1, 1) so ``_append_events`` iterates
                # rows of label-iterables (matching the convention used by the
                # plain ``key="epochs"`` AxisArray path). A 1D shape would make
                # ``",".join(ev_str)`` join the characters of a single string.
                self._current_msg = AxisArray(
                    data=np.array([[self._current_msg.value]]),
                    dims=["time", "ch"],
                    axes={"time": AxisArray.LinearAxis(gain=1.0, offset=timestamp)},
                    key="epochs",
                )
            elif not hasattr(self._current_msg, "data"):
                return
            else:
                axis = self.settings.axis
                targ_ax_ix = self._current_msg.get_axis_idx(axis)
                if targ_ax_ix != 0:
                    self._current_msg = replace(
                        self._current_msg,
                        data=np.moveaxis(self._current_msg.data, targ_ax_ix, 0),
                        dims=[axis] + self._current_msg.dims[:targ_ax_ix] + self._current_msg.dims[targ_ax_ix + 1 :],
                    )

            # Is this a new series?
            b_new = self._state.io is None
            b_new = b_new or self._current_msg.key not in self._state.series

            # If inc message key is in state.series but properties do not match previous dataset properties
            #  then close io and raise error
            if not b_new and not self._check_msg_consistency():
                nwbfile = self._state.nwbfile
                b_final_write = hasattr(nwbfile, "epochs") and nwbfile.epochs is not None
                b_final_write = b_final_write or (hasattr(nwbfile, "trials") and nwbfile.trials is not None)
                self.close(write=b_final_write)
                raise ValueError("Data provided to NWBSink has changed shape. Closing NWB file.")

            if b_new:
                # Use first incoming timestamp to set the session start time.
                key = self._current_msg.key
                axis = self.settings.axis
                t0 = None
                if axis in ["time", "win"] or "time" in self._current_msg.axes:
                    targ_dim = axis if axis in ["time", "win"] else "time"
                    if hasattr(self._current_msg.axes[targ_dim], "data"):
                        t0 = self._current_msg.axes[targ_dim].data[0]
                    else:
                        t0 = self._current_msg.axes[targ_dim].offset
                _ = self.get_session_datetime(t0)
                self._state.start_timestamp = self.get_session_timestamp(t0)
                if self.settings.inc_clock == ReferenceClockType.MONOTONIC:
                    self._state.start_timestamp += time.monotonic() - time.time()

                # Create the container(s) for the new stream.
                if key in ["epochs", "trials"]:
                    self._prep_event_io()
                    self._flush_io(reopen=True)
                elif self._current_msg.data.dtype.type is np.str_:
                    raise ValueError(
                        f"Cannot stream varlen str data to series {key}. Use 'epochs' or 'trials' instead."
                    )
                else:
                    self._prep_continuous_io()
                    self._flush_io(reopen=True)
                    self._update_rate_for_current()

            if self.settings.recording and self._current_msg.data.size:
                axis = self.settings.axis
                timestamps = None
                if axis in ["time", "win"] or "time" in self._current_msg.axes:
                    targ_dim = axis if axis in ["time", "win"] else "time"
                    time_ax = self._current_msg.axes[targ_dim]
                    if hasattr(time_ax, "data"):
                        timestamps = time_ax.data - self._state.start_timestamp
                    else:
                        timestamps = (np.arange(len(self._current_msg.data)) * time_ax.gain) + (
                            time_ax.offset - self._state.start_timestamp
                        )

                key = self._current_msg.key
                if key in ["epochs", "trials"]:
                    self._append_events(key, timestamps, self._current_msg.data)
                else:
                    series_state = self._state.series[key]
                    # Write data
                    dataset = series_state.data
                    dataset.resize(len(dataset) + len(self._current_msg.data), axis=0)
                    dataset[-len(self._current_msg.data) :] = self._current_msg.data
                    series_state.bytes_written += self._current_msg.data.nbytes

                    # Write timestamps
                    if timestamps is not None:
                        ts = series_state.ts
                        ts.resize(len(ts) + len(timestamps), axis=0)
                        ts[-len(timestamps) :] = timestamps
                        series_state.bytes_written += timestamps.nbytes

                total_bytes = sum(s.bytes_written for s in self._state.series.values())
                if 0 < self.settings.split_bytes <= total_bytes and "%d" not in str(self._state.filepath):
                    # Rotate into a fresh file segment, carrying the current
                    # pipeline-settings state forward so the new file's table
                    # opens already populated.
                    self._rotate_file(
                        timestamp=time.time(),
                        next_settings_state=dict(self._state.settings_state or {}),
                        next_settings_prev_component="__split__",
                    )

    @property
    def path_on_disk(self) -> Path:
        fp = Path(self._state.filepath)
        if "%d" in str(fp):
            return Path(re.sub("%d", "0", str(fp)))
        # Suffix when ``split_bytes`` is configured (uniform "<stem>_NN" naming
        # for the whole split-set, starting at _00) OR when a settings-driven
        # rotation has incremented split_count past 0 (the original file kept
        # its bare name, subsequent segments get "_01", "_02", ...).
        if self.settings.split_bytes > 0 or self._state.split_count > 0:
            return fp.parent / (fp.stem + f"_{self._state.split_count:02}" + fp.suffix)
        return fp

    def get_session_datetime(self, try_t0: typing.Optional[float] = None) -> datetime.datetime:
        """
        Retrieve session datetime. If it does not already exist, set it with try_t0 if provided
         and clock is understood, else with datetime.now.

        Args:
            try_t0: (Optional) The first incoming timestamp. If provided, it is used to set the session start time.
              The value is timezone-naive and assumed to be in the same clock as time.time().

        Returns:
            Common session starttime among all instances of this class
        """
        inc_clock = self.settings.inc_clock
        if self.__class__.shared_clock_type is not None and self.__class__.shared_clock_type != inc_clock:
            raise ValueError(
                f"All instances must share the same clock type. {inc_clock} != {self.__class__.shared_clock_type}"
            )
        if self.__class__.shared_start_datetime is None:
            if try_t0 is not None and inc_clock in [
                ReferenceClockType.SYSTEM,
                ReferenceClockType.MONOTONIC,
            ]:
                if inc_clock == ReferenceClockType.MONOTONIC:
                    try_t0 = try_t0 - time.monotonic() + time.time()
                self.__class__.shared_start_datetime = datetime.datetime.fromtimestamp(try_t0, datetime.timezone.utc)
            else:
                self.__class__.shared_start_datetime = datetime.datetime.now(datetime.timezone.utc)
            # Latch the clock type so subsequent instances are forced to
            # match — otherwise the mismatch check above never fires.
            self.__class__.shared_clock_type = inc_clock
        return self.__class__.shared_start_datetime

    def get_session_timestamp(self, try_t0: typing.Optional[float] = None) -> float:
        """
        Retrieve session timestamp. If it does not already exist, set it with try_t0 if provided.

        Args:
            try_t0: (Optional) The first incoming timestamp.

        Returns:
            Common session timestamp among all instances of this class.
        """
        inc_clock = self.settings.inc_clock
        if self.__class__.shared_clock_type is not None and self.__class__.shared_clock_type != inc_clock:
            raise ValueError(
                f"All instances must share the same clock type. {inc_clock} != {self.__class__.shared_clock_type}"
            )
        if self.__class__.shared_t0 is None:
            if try_t0 is None or inc_clock in [
                ReferenceClockType.SYSTEM,
                ReferenceClockType.MONOTONIC,
            ]:
                self.__class__.shared_t0 = self.get_session_datetime(try_t0).timestamp()
            else:
                self.__class__.shared_t0 = try_t0 if try_t0 is not None else 0.0
                ez.logger.warning(
                    "Clock type is UNKNOWN. Timestamps are relative to the first incoming timestamp "
                    "but this value is NOT recoverable as it is not stored in the NWB file."
                )
            # Latch the clock type if get_session_datetime didn't already
            # (the UNKNOWN branch above bypasses it).
            if self.__class__.shared_clock_type is None:
                self.__class__.shared_clock_type = inc_clock
        return self.__class__.shared_t0

    def _check_filepath(self) -> None:
        """
        Normalize ``self._state.filepath`` (reading the raw path from
        ``self.settings.filepath``). If the resolved path already exists,
        delete it when ``overwrite_old`` is enabled or raise otherwise.
        """
        _suffix = ".nwb"

        filepath = Path(self._state.filepath)
        if filepath.name.startswith("."):
            raise FileNotFoundError(
                f"filepath {filepath} name begins with `.` -- cannot discriminate name from extension."
            )

        filepath = filepath.expanduser()

        # If provided path is merely a directory then create a new filename.
        is_dir = (isinstance(filepath, Path) and filepath.is_dir()) or (
            isinstance(filepath, str) and filepath[-1] == "/"
        )
        if is_dir:
            meta = self._read_meta_dict()
            if "Subject" not in meta:
                meta["Subject"] = {"subject_id": "P001"}
            if "session_start_time" not in meta["NWBFile"]:
                meta["NWBFile"]["session_start_time"] = self.get_session_datetime()
            filepath = filepath / build_nwb_fname(meta)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not filepath.suffix:
            filepath = filepath.parent / (filepath.name + _suffix)

        self._state.filepath = filepath

        if self.path_on_disk.exists():
            age = (time.time() - os.path.getctime(self.path_on_disk)) / 60
            ez.logger.info(f"File at {self.path_on_disk} is {age:.2f} minutes old.")
            if self.settings.overwrite_old:
                ez.logger.info("File will be overwritten.")
                self.path_on_disk.unlink(missing_ok=False)
            else:
                msg = "File exists but overwriting is disabled. Set overwrite_old=True to overwrite."
                ez.logger.error(msg)
                raise ValueError(msg)

    def _read_meta_dict(self) -> typing.Union[typing.Mapping, dict]:
        """
        Load the metadata from ``self.settings.meta_yaml`` if that path
        exists, else load it from the default location.

        Returns:
            A dict containing the metadata for this NWB file.
        """
        meta_yaml = self.settings.meta_yaml
        if meta_yaml is None or not Path(meta_yaml).expanduser().exists():
            default_path = Path(__file__).parent
            meta_dict = DeepDict()
            for yaml_name in ["nwb_metadata", "nwb_session"]:
                yaml_path = default_path / f"{yaml_name}.yaml"
                meta_dict = dict_deep_update(meta_dict, load_dict_from_file(yaml_path))
        else:
            yaml_path = Path(meta_yaml).expanduser()
            meta_dict = load_dict_from_file(yaml_path)
        return meta_dict

    def _prep_from_meta(self, meta: dict):
        """
        Prepare the NWBFile and NWBHDF5IO objects from metadata.

        Args:
            meta: The metadata dict to use to create the NWBFile and NWBHDF5IO objects.
        """

        def _sanitize_shape(
            shape: typing.Union[typing.List[int], typing.Tuple[int, ...]],
        ) -> typing.List[int]:
            shape = [0 if _ in [None, -1] else _ for _ in shape]
            n_zero = len([_ for _ in shape if _ == 0])
            if n_zero > 1:
                raise ValueError(f"Cannot have more than one 0 in shape: {shape}")
            else:
                shape = [0] + [_ for _ in shape if _ != 0]
            return shape

        # Create the trials and epochs tables
        for key in ["epochs", "trials"]:
            if key in meta:
                ss = meta[key]
                shape = _sanitize_shape(ss["shape"])
                self._current_msg = AxisArray(
                    data=np.zeros(shape, dtype="U"),
                    dims=["time", "ch"] + [f"dim{_}" for _ in range(len(shape) - 2)],
                    axes={"time": AxisArray.CoordinateAxis(np.array([]), dims=["time"], unit="s")},
                    key=key,
                )
                self._prep_event_io()

        # Pre-allocate the pipeline-settings table from a metadata snapshot
        # if one was carried over (e.g., after a file split via _rotate_file).
        if self.SETTINGS_TABLE_NAME in meta:
            settings_meta = meta[self.SETTINGS_TABLE_NAME]
            sample_values = dict(settings_meta.get("settings_state") or {})
            if not sample_values:
                sample_values = {col: "" for col in settings_meta.get("settings_columns", [])}
            if sample_values:
                self._prep_settings_table(sample_values)

        for key, ss in meta.items():
            if key in ("epochs", "trials", self.SETTINGS_TABLE_NAME):
                continue
            shape = _sanitize_shape(ss["shape"])
            self._current_msg = AxisArray(
                data=np.zeros(shape, dtype=self._current_msg.data.dtype),
                dims=["time", "ch"] + [f"dim{_}" for _ in range(len(shape) - 2)],
                axes={
                    "time": AxisArray.TimeAxis(fs=ss["fs"]),
                },
                key=key,
            )
            self._prep_continuous_io()

        # Flush and reopen to make h5py datasets
        self._flush_io(reopen=True)

        # Add the rate attribute to the timestamps series. Can only do this after flushing.
        for key, ss in meta.items():
            if key in ("epochs", "trials", self.SETTINGS_TABLE_NAME):
                continue
            series = self._state.nwbfile.acquisition[key]
            series.timestamps.attrs["rate"] = ss["fs"]

    def close(self, write=False, log=True) -> None:
        """
        Close the file. This will also delete the file if it is empty.

        Args:
            write: Set True to write the file to disk before closing.
            log: Set True to log the closing and deletion of the file.
              This must be kept False when calling from __del__.
        """
        state = getattr(self, "_state", None)
        if state is None or state.io is None:
            return
        # If __init__ raised before the lock was set up, fall through without
        # locking; nothing else can be racing on this half-built instance.
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._close_locked(state, write=write, log=log)
            return
        with lock:
            self._close_locked(state, write=write, log=log)

    def _close_locked(self, state: NWBSinkState, write: bool, log: bool) -> None:
        if state.io is None:
            return
        # Flush any open pipeline-settings interval so it lands in the file.
        if state.settings_state and state.settings_active_since is not None:
            self._flush_settings_interval(time.time(), state.settings_prev_component)
            state.settings_active_since = None
        nwbfile = state.nwbfile
        io = state.io
        if write:
            io.write(nwbfile)
        src_str = f"{io.source}"
        b_delete = sum(s.bytes_written for s in state.series.values()) == 0
        for key in ["epochs", "trials"]:
            if hasattr(nwbfile, key) and getattr(nwbfile, key) is not None:
                b_delete = b_delete and len(getattr(nwbfile, key)) == 1  # EZNWB-START
        # A populated pipeline_settings table is also "content" — keep the
        # file even if no acquisition data was written.
        settings_table = self._get_settings_table()
        if settings_table is not None and len(settings_table.id) > 0:
            b_delete = False
        io.close()
        state.nwbfile = None
        state.io = None
        state.series = {}
        state.settings_columns = []
        state.settings_state = {}
        state.settings_prev_component = "__init__"
        if log:
            ez.logger.info(f"Closed file at {src_str}")
        if b_delete:
            self.path_on_disk.unlink(missing_ok=True)
            if log:
                ez.logger.info(f"Deleted empty file at {src_str}.")

    def toggle_recording(self, recording: typing.Optional[bool] = None):
        """Deprecated. Send a ``NWBSinkSettings`` update with the desired
        ``recording`` value instead — the update is routed through
        :meth:`update_settings` and takes effect on the next message.
        """
        warnings.warn(
            "NWBSinkConsumer.toggle_recording is deprecated; publish a "
            "NWBSinkSettings update with the desired `recording` value "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        new_value = recording if recording is not None else not self.settings.recording
        self.update_settings(dataclasses.replace(self.settings, recording=new_value))

    def _check_msg_consistency(self) -> bool:
        axis = self.settings.axis
        key = self._current_msg.key
        series_state = self._state.series[key]
        in_ax = self._current_msg.axes[axis]
        b_rate_change = (
            axis in self._current_msg.axes
            and series_state.ts is not None
            and not hasattr(in_ax, "data")
            and series_state.ts.attrs["rate"] != 1 / in_ax.gain
        )
        b_shape_change = series_state.shape != self._current_msg.data.shape[1:]
        return not (b_rate_change or b_shape_change)

    def _update_rate_for_current(self):
        axis = self.settings.axis
        if axis in ["time", "win"]:
            time_ax = self._current_msg.axes[axis]
            if hasattr(time_ax, "data"):
                rate = 0.0
            else:
                rate = 1 / time_ax.gain if time_ax.gain != 0 else 0
            self._state.series[self._current_msg.key].ts.attrs["rate"] = rate

    def _copy_nwb(
        self,
        settings_columns: typing.Optional[typing.List[str]] = None,
        settings_state: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.Tuple[pynwb.NWBFile, dict]:
        copy_keys = [
            "session_description",
            "session_start_time",
            "experimenter",
            "experiment_description",
            "session_id",
            "institution",
            "notes",
            "pharmacology",
            "protocol",
            "related_publications",
            "slices",
            "source_script",
            "source_script_file_name",
            "data_collection",
            "surgery",
            "virus",
            "stimulus_notes",
            "lab",
        ]
        old_nwbfile = self._state.nwbfile
        new_nwb_kwargs = {k: getattr(old_nwbfile, k) for k in copy_keys if hasattr(old_nwbfile, k)}
        new_nwb_kwargs["keywords"] = (
            old_nwbfile.keywords if isinstance(old_nwbfile.keywords, list) else old_nwbfile.keywords[:].tolist()
        )
        nwbfile = pynwb.NWBFile(identifier=str(uuid4()), **new_nwb_kwargs)
        nwbfile.subject = pynwb.file.Subject(**old_nwbfile.subject.fields)
        meta = {}
        for key in ["epochs", "trials"]:
            if hasattr(old_nwbfile, key) and getattr(old_nwbfile, key) is not None:
                meta[key] = {"fs": 0.0, "shape": (0, 1)}
        for key, ss in self._state.series.items():
            if key not in ["epochs", "trials"]:
                meta[key] = {"fs": ss.ts.attrs["rate"], "shape": (0,) + ss.shape}
        # Carry pipeline-settings schema/state forward so the next file can
        # reopen its table populated. Caller may override via args.
        cols = list(self._state.settings_columns or []) if settings_columns is None else list(settings_columns)
        state_dict = dict(self._state.settings_state or {}) if settings_state is None else dict(settings_state)
        if cols or state_dict:
            meta[self.SETTINGS_TABLE_NAME] = {
                "settings_columns": cols,
                "settings_state": state_dict,
            }

        return nwbfile, meta

    def _nwb_create_or_fail(self, nwbfile: typing.Optional[pynwb.NWBFile] = None) -> None:
        """
        Create the NWBFile and NWBHDF5IO for writing.
        Note: This does not yet write the NWBFile object to disk.
        """
        if self.path_on_disk.exists():
            raise ValueError(f"File {self.path_on_disk} already exists. Set overwrite_old=True to overwrite.")

        if nwbfile is None:
            meta_dict = self._read_meta_dict()
            nwbfile = pynwb.NWBFile(
                identifier=str(uuid4()),
                session_start_time=self.get_session_datetime(),
                **meta_dict["NWBFile"],
            )
            if "Subject" in meta_dict:
                nwbfile.subject = pynwb.file.Subject(**meta_dict["Subject"])

        if "%d" in str(self._state.filepath):
            io_file = h5py.File(
                name=self._state.filepath,
                mode="w",
                driver="family",
                memb_size=self.settings.split_bytes,
            )
            io = pynwb.NWBHDF5IO(file=io_file, mode="w")
        else:
            io = pynwb.NWBHDF5IO(self.path_on_disk, "w")

        self._state.io = io
        self._state.nwbfile = nwbfile
        # Fresh series map; prior entries (if any) belong to the closed file.
        self._state.series = {}

    def _flush_io(self, reopen: bool = True):
        """
        Write the header to the NWBFile.

        This step is also necessary to:
        * enable appending to our epochs/trials table (but only after it has an entry).
        * create the appendable datasets for our continuous data.
        """
        self._state.io.write(self._state.nwbfile)
        if reopen:
            if self._state.io:
                self._state.io.close()
            if "%d" in str(self._state.filepath):
                io_file = h5py.File(
                    name=self._state.filepath,
                    mode="a",
                    driver="family",
                    memb_size=self.settings.split_bytes,
                )
                io = pynwb.NWBHDF5IO(file=io_file, mode="a")
            else:
                io = pynwb.NWBHDF5IO(self.path_on_disk, "a")
            self._state.io = io
            self._state.nwbfile = self._state.io.read()

        # Get references to our continuous datasets
        for k, ss in self._state.series.items():
            if k in self._state.nwbfile.acquisition:
                series = self._state.nwbfile.acquisition[k]
                if isinstance(series.data, H5DataIO):
                    ss.data = series.data.dataset
                    ss.ts = series.timestamps.dataset
                else:
                    ss.data = series.data
                    ss.ts = series.timestamps

    def _append_events(
        self,
        key: str,
        timestamps: typing.Iterable[float],
        data: typing.Iterable[typing.Iterable[str]],
    ):
        nwbfile = self._state.nwbfile
        fun = {"epochs": nwbfile.add_epoch, "trials": nwbfile.add_trial}[key]
        for ev_t, ev_str in zip(timestamps, data):
            fun(start_time=ev_t, stop_time=ev_t + 0, **{"label": ",".join(ev_str)})

    # ------------------------------------------------------------------
    # Pipeline-settings tracker
    #
    # The unit's GraphContext watcher feeds prepared settings events into
    # ``update_settings_table``; we record them as a per-file ``TimeIntervals``
    # table named ``SETTINGS_TABLE_NAME``. A schema-incompatible update (new
    # column, scalar↔array shape change) rotates into a fresh file segment
    # so the table layout stays consistent within any one file.
    # ------------------------------------------------------------------

    def _configure_appendable_table(
        self,
        table: typing.Any,
        sample_values: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> None:
        """Configure a dynamic table so its columns remain appendable after flush/reopen."""
        table.id.set_data_io(H5DataIO, {"maxshape": (None,), "chunks": True})
        for col in table.colnames:
            sample_value = None if sample_values is None else sample_values.get(col)
            if sample_value is None:
                col_shape = getattr(table[col].data, "shape", ())
                maxshape = (None,) + tuple(col_shape[1:]) if len(col_shape) > 1 else (None,)
            elif np.isscalar(sample_value) or isinstance(sample_value, (str, bytes)):
                maxshape = (None,)
            else:
                col_shape = np.asarray(sample_value).shape
                maxshape = (None,) + tuple(col_shape)
            table[col].set_data_io(H5DataIO, {"maxshape": maxshape, "chunks": True})

    def _get_settings_table(self) -> typing.Any:
        """Return the pipeline-settings intervals table if it exists."""
        nwbfile = self._state.nwbfile
        if nwbfile is None or nwbfile.intervals is None:
            return None
        try:
            return nwbfile.intervals[self.SETTINGS_TABLE_NAME]
        except Exception:
            return None

    def _prep_settings_table(self, flat_settings: typing.Dict[str, typing.Any]) -> None:
        """Create the pipeline-settings intervals table and its columns."""
        if self._get_settings_table() is not None:
            return

        intervals = pynwb.epoch.TimeIntervals(
            name=self.SETTINGS_TABLE_NAME,
            description="Flattened ezmsg settings snapshots active over each logged interval",
        )
        intervals.add_column(
            name="updated_component",
            description="component that triggered the snapshot transition",
        )
        for column_name in flat_settings:
            intervals.add_column(name=column_name, description="flattened ezmsg setting")
        self._state.nwbfile.add_time_intervals(intervals)

        table = self._get_settings_table()
        if table is None:
            raise RuntimeError("Failed to create settings table")

        self._state.settings_columns = list(flat_settings.keys())
        self._configure_appendable_table(table, flat_settings)

    def _settings_relative_time(self, timestamp: float) -> float:
        """Convert an absolute timestamp into file-relative session time."""
        return timestamp - self.get_session_timestamp(None)

    def _settings_value_shape(self, value: typing.Any) -> typing.Tuple[int, ...]:
        """Return the per-cell shape occupied by a settings value."""
        if value is None or np.isscalar(value) or isinstance(value, (str, bytes)):
            return ()
        try:
            return tuple(np.asarray(value).shape)
        except Exception:
            return ()

    def _column_value_shape(self, column_name: str) -> typing.Tuple[int, ...]:
        """Return the current per-cell shape stored for a settings column."""
        table = self._get_settings_table()
        if table is not None and column_name in table.colnames:
            data = table[column_name].data
            if hasattr(data, "shape"):
                shape = tuple(data.shape)
                return shape[1:] if len(shape) > 1 else ()

        state = self._state.settings_state or {}
        if column_name in state:
            return self._settings_value_shape(state[column_name])
        return ()

    def _validate_settings_columns(self, flat_settings: typing.Dict[str, typing.Any]) -> None:
        """Ensure incoming settings keys match the established settings-table schema."""
        cols = self._state.settings_columns or []
        missing = [name for name in flat_settings if name not in cols]
        if missing:
            raise ValueError(
                f"Received settings fields not present in settings table schema: {', '.join(sorted(missing))}"
            )

    def _settings_update_requires_rotation(self, flat_settings: typing.Dict[str, typing.Any]) -> bool:
        """Return whether an incoming settings update is incompatible with the current table schema."""
        cols = self._state.settings_columns or []
        for column_name, value in flat_settings.items():
            if column_name not in cols:
                return True
            current_shape = self._column_value_shape(column_name)
            new_shape = self._settings_value_shape(value)
            current_is_scalar = current_shape == ()
            new_is_scalar = new_shape == ()
            if current_is_scalar != new_is_scalar:
                return True
            if not current_is_scalar and current_shape != new_shape:
                return True
        return False

    def _merged_settings_state(self, flat_settings: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """Return the full settings snapshot after applying a partial settings update."""
        cols = self._state.settings_columns or []
        state = self._state.settings_state or {}
        merged = {column_name: state.get(column_name, "") for column_name in cols}
        merged.update(flat_settings)
        return merged

    def _rotate_file(
        self,
        timestamp: float,
        next_settings_state: typing.Optional[typing.Dict[str, typing.Any]],
        next_settings_prev_component: str,
    ) -> None:
        """Close the current NWB file and open a new segment seeded with the given settings state."""
        state = self._state
        if state.settings_state and state.settings_active_since is not None:
            self._flush_settings_interval(timestamp, state.settings_prev_component)

        next_settings_state = dict(next_settings_state or {})
        next_settings_columns = list(next_settings_state.keys())

        # Prevent close() from re-appending the interval we just flushed.
        state.settings_active_since = None

        # Advance to the next file segment before creating the replacement file.
        state.split_count += 1
        self.path_on_disk.unlink(missing_ok=True)

        new_nwbfile, new_meta = self._copy_nwb(
            settings_columns=[],
            settings_state={},
        )
        self.close()
        self._nwb_create_or_fail(nwbfile=new_nwbfile)
        self._prep_from_meta(new_meta)

        if next_settings_state:
            self._prep_settings_table(next_settings_state)
            self._state.settings_state = {
                col: next_settings_state.get(col, "") for col in (self._state.settings_columns or [])
            }
            self._state.settings_active_since = timestamp
            self._state.settings_prev_component = next_settings_prev_component
            self._apply_settings_update(next_settings_prev_component, next_settings_state, timestamp)
            self._flush_io(reopen=True)
        else:
            self._state.settings_columns = next_settings_columns
            self._state.settings_state = next_settings_state
            self._state.settings_active_since = None
            self._state.settings_prev_component = next_settings_prev_component

    def _apply_settings_update(
        self,
        component_address: str,
        flat_settings: typing.Dict[str, typing.Any],
        timestamp: float,
    ) -> None:
        """Commit a prepared settings snapshot into the in-memory active state."""
        self._flush_settings_interval(timestamp, self._state.settings_prev_component)
        if not self._state.settings_state:
            self._state.settings_state = {col: "" for col in (self._state.settings_columns or [])}
        self._state.settings_state.update(flat_settings)
        self._state.settings_active_since = timestamp
        self._state.settings_prev_component = component_address

    def initialize_settings_table(self, flat_settings: typing.Dict[str, typing.Any], timestamp: float) -> None:
        """Seed the pipeline-settings table from a prepared flat settings snapshot.

        Renamed from ``initialize_settings`` to disambiguate from the
        baseproc ``update_settings``/settings-message machinery that handles
        ``NWBSinkSettings`` updates on the unit.
        """
        with self._lock:
            if not flat_settings:
                return
            if not self._state.settings_columns:
                self._prep_settings_table(flat_settings)
            self._validate_settings_columns(flat_settings)
            self._state.settings_state = {
                col: flat_settings.get(col, "") for col in (self._state.settings_columns or [])
            }
            self._state.settings_active_since = timestamp
            self._state.settings_prev_component = "__init__"
            self._apply_settings_update("__init__", flat_settings, self.get_session_timestamp())
            self._flush_io(reopen=True)

    def update_settings_table(
        self,
        component_address: str,
        flat_settings: typing.Dict[str, typing.Any],
        timestamp: float,
    ) -> None:
        """Record a prepared settings update into the pipeline-settings table.

        Renamed from ``update_settings`` to avoid shadowing the inherited
        baseproc method of the same name (which applies ``NWBSinkSettings``
        updates from ``INPUT_SETTINGS``). This one is the GraphContext
        watcher's callback for ANY component's settings changes.
        """
        with self._lock:
            if not flat_settings:
                return
            if not self._state.settings_columns:
                self._prep_settings_table(flat_settings)
                self._state.settings_state = {col: "" for col in (self._state.settings_columns or [])}
                self._state.settings_active_since = timestamp

            if self._settings_update_requires_rotation(flat_settings):
                self._rotate_file(
                    timestamp=timestamp,
                    next_settings_state=self._merged_settings_state(flat_settings),
                    next_settings_prev_component=component_address,
                )
                return

            self._validate_settings_columns(flat_settings)
            self._apply_settings_update(component_address, flat_settings, timestamp)

    def _flush_settings_interval(self, end_timestamp: float, updated_component: str) -> None:
        """Append the currently active settings snapshot as a closed NWB interval."""
        state = self._state
        if not state.settings_state or state.settings_active_since is None:
            return

        table = self._get_settings_table()
        if table is None:
            return

        start_time = self._settings_relative_time(state.settings_active_since)
        stop_time = self._settings_relative_time(end_timestamp)
        if stop_time < start_time:
            stop_time = start_time

        table.add_interval(
            start_time=start_time,
            stop_time=stop_time,
            updated_component=updated_component,
            **state.settings_state,
        )

    def _prep_event_io(self):
        """
        Prepare the NWB file to receive event data, either "epochs" or "trials".
        """
        colname = "label"
        key = self._current_msg.key
        nwbfile = self._state.nwbfile
        fun = {
            "epochs": nwbfile.add_epoch_column,
            "trials": nwbfile.add_trial_column,
        }[key]
        fun(name=colname, description=f"{colname} {key}")

        self._state.series[key] = SeriesState(shape=self._current_msg.data.shape[1:])

        table = {"epochs": nwbfile.epochs, "trials": nwbfile.trials}[key]
        table.id.set_data_io(H5DataIO, {"maxshape": (None,)})
        table.start_time.set_data_io(H5DataIO, {"maxshape": (None,)})
        table.stop_time.set_data_io(H5DataIO, {"maxshape": (None,)})
        getattr(table, colname).set_data_io(H5DataIO, {"maxshape": (None,)})

        # Note: We cannot io.write(...) yet because a bug in hdmf requires that a custom column
        #  must have data before it may be written:
        #  https://github.com/hdmf-dev/hdmf/issues/1000
        # So we append a dummy event
        self._append_events(key, [0.0], [["EZNWB-START"]])

    def _prep_continuous_io(self):
        """
        Prepare NWB file to receive continuous data into a pynwb.TimeSeries (or pynwb.ecephys.ElectricalSeries if
        channel info is provided).
        """
        key = self._current_msg.key
        nwbfile = self._state.nwbfile
        targ_ax_ix = self._current_msg.get_axis_idx(self.settings.axis)
        shape = self._current_msg.data.shape[:targ_ax_ix] + self._current_msg.data.shape[targ_ax_ix + 1 :]
        self._state.series[key] = SeriesState(shape=shape)
        dataio = H5DataIO(
            shape=(0,) + shape,
            dtype=self._current_msg.data.dtype,
            maxshape=(None,) + shape,
            fillvalue=np.nan,
        )
        tsio = H5DataIO(
            shape=(0,),
            dtype=np.float64,
            maxshape=(None,),
            fillvalue=np.nan,
        )
        series_description = "created by ezmsg nwbsink"

        # If channel label info is provided then we create an electrode table and our timeseries is
        #  a pynwb.ecephys.ElectricalSeries.
        if (
            "ch" in self._current_msg.axes
            and hasattr(self._current_msg.axes["ch"], "data")
            and len(self._current_msg.axes["ch"].data)
        ):
            b_first = nwbfile.electrodes is None or "label" not in nwbfile.electrodes.colnames
            if b_first:
                nwbfile.add_electrode_column(name="label", description="electrode label")

            dev_name = "unified device"
            if dev_name in nwbfile.devices:
                device = nwbfile.devices[dev_name]
            else:
                device = nwbfile.create_device(name=dev_name, description="created by ezmsg nwbsink")

            el_grp_name = "unified electrode group"
            if el_grp_name in nwbfile.electrode_groups:
                el_grp = nwbfile.electrode_groups[el_grp_name]
            else:
                el_grp = nwbfile.create_electrode_group(
                    name=el_grp_name,
                    description="electrode group created by ezmsg nwbsink",
                    device=device,
                    location="unknown",
                )
                if not b_first:
                    self._flush_io(reopen=True)
                    # _flush_io swaps nwbfile out from under us; refresh.
                    nwbfile = self._state.nwbfile

            el_df = nwbfile.electrodes.to_dataframe()
            el_df = el_df[el_df["group"] == el_grp]
            for ll in self._current_msg.axes["ch"].data:
                if ll not in el_df["label"].values:
                    nwbfile.add_electrode(label=ll, location="unknown", group=el_grp)

            if type(nwbfile.electrodes.id.data) is list:
                for fn in ["id", "location", "group_name", "group", "label"]:
                    getattr(nwbfile.electrodes, fn).set_data_io(H5DataIO, {"maxshape": (None,), "chunks": True})

            el_df = nwbfile.electrodes.to_dataframe()
            el_df = el_df[el_df["group"] == el_grp]
            b_in = el_df["label"].isin(self._current_msg.axes["ch"].data)
            el_tbl_region = nwbfile.create_electrode_table_region(
                region=el_df[b_in].index.tolist(),
                description=f"electrodes for {key}",
            )
            series = pynwb.ecephys.ElectricalSeries(
                name=key,
                data=dataio,
                timestamps=tsio,
                conversion=1e-6,
                electrodes=el_tbl_region,
                description=series_description,
            )

        else:
            series = pynwb.TimeSeries(
                name=key,
                data=dataio,
                timestamps=tsio,
                unit="volts",
                conversion=1e-6,
                description=series_description,
            )
        nwbfile.add_acquisition(series)


class NWBSink(BaseConsumerUnit[NWBSinkSettings, AxisArray, NWBSinkConsumer]):
    SETTINGS = NWBSinkSettings

    async def initialize(self) -> None:
        """Discover the components in this session and start the GraphServer
        settings watcher. The watcher mirrors EVERY component's settings
        changes (including this sink's own) into the consumer's
        ``pipeline_settings`` TimeIntervals table.
        """
        await super().initialize()
        self._settings_ctx: typing.Optional[ez.GraphContext] = None
        self._settings_watch_task: typing.Optional[asyncio.Task[None]] = None
        self._settings_component_addresses: typing.Set[str] = set()
        self._settings_last_timestamp: float = time.time()

        try:
            ctx = ez.GraphContext(auto_start=False)
            await ctx.__aenter__()
            self._settings_ctx = ctx

            snapshot = await ctx.snapshot()
            session_components = self._session_component_addresses(snapshot)
            if not session_components:
                return

            self._settings_component_addresses = session_components
            settings_snapshot = await ctx.settings_snapshot()
            settings_events = await ctx.settings_events(after_seq=0)

            flat_snapshot: typing.Dict[str, typing.Any] = {}
            for component_address in sorted(session_components):
                if component_address in settings_snapshot:
                    flat_snapshot.update(
                        flatten_component_settings(component_address, settings_snapshot[component_address])
                    )

            latest_timestamp = max(
                (event.timestamp for event in settings_events if event.component_address in session_components),
                default=time.time(),
            )
            last_seq = max((event.seq for event in settings_events), default=0)
            self._settings_last_timestamp = latest_timestamp

            if flat_snapshot:
                await asyncio.to_thread(
                    self.processor.initialize_settings_table,
                    flat_snapshot,
                    latest_timestamp,
                )

            self._settings_watch_task = asyncio.create_task(
                self._watch_graph_settings(after_seq=last_seq),
                name=f"{self.address}-settings-watch",
            )

        except Exception as exc:
            ez.logger.warning(f"{self.address} could not initialize GraphServer-backed settings logging: {exc}")
            if self._settings_ctx is not None:
                await self._settings_ctx.__aexit__(None, None, None)
                self._settings_ctx = None

    def _session_component_addresses(self, graph_snapshot: typing.Any) -> typing.Set[str]:
        """Return the component addresses that belong to this unit's active session."""
        for session in graph_snapshot.sessions.values():
            metadata = session.metadata
            if metadata is None:
                continue
            if self.address in metadata.components:
                return set(metadata.components.keys())
        return set()

    async def _watch_graph_settings(self, after_seq: int) -> None:
        """Forward graph settings events to the consumer's settings-table writer."""
        if self._settings_ctx is None:
            return
        async for event in self._settings_ctx.subscribe_settings_events(after_seq=after_seq):
            if event.component_address not in self._settings_component_addresses:
                continue
            flat_settings = flatten_component_settings(event.component_address, event.value)
            try:
                await asyncio.to_thread(
                    self.processor.update_settings_table,
                    event.component_address,
                    flat_settings,
                    event.timestamp,
                )
                self._settings_last_timestamp = event.timestamp
            except Exception as exc:
                ez.logger.warning(
                    f"{self.address} failed to record settings update for {event.component_address}: {exc}"
                )

    async def _take_pipeline_snapshot(self) -> typing.Dict[str, typing.Any]:
        """Build a flat settings snapshot covering this session's components."""
        if self._settings_ctx is None or not self._settings_component_addresses:
            return {}
        settings_snapshot = await self._settings_ctx.settings_snapshot()
        flat: typing.Dict[str, typing.Any] = {}
        for component_address in sorted(self._settings_component_addresses):
            if component_address in settings_snapshot:
                flat.update(flatten_component_settings(component_address, settings_snapshot[component_address]))
        return flat

    @ez.subscriber(BaseConsumerUnit.INPUT_SETTINGS)
    async def on_settings(self, msg: NWBSinkSettings) -> None:
        """Apply ``NWBSinkSettings`` updates and, when the change requires a
        file swap, pre-fetch a pipeline-settings snapshot so the new file's
        table opens already populated (option B re-seed-on-swap).
        """
        will_reset = bool(
            {f.name for f in dataclasses.fields(msg) if getattr(self.SETTINGS, f.name) != getattr(msg, f.name)}
            - NWBSinkConsumer.NONRESET_SETTINGS_FIELDS
        )
        if will_reset and getattr(self, "_settings_ctx", None) is not None:
            snapshot = await self._take_pipeline_snapshot()
            if snapshot:
                self.processor._pending_settings_seed = (snapshot, time.time())
        # Replicate the inherited base behaviour: rebind unit settings then
        # delegate to the processor so it can diff + maybe request reset.
        self.apply_settings(msg)
        self.processor.update_settings(self.SETTINGS)

    async def shutdown(self) -> None:
        if getattr(self, "_settings_watch_task", None) is not None:
            self._settings_watch_task.cancel()
            try:
                await self._settings_watch_task
            except asyncio.CancelledError:
                pass
        if getattr(self, "_settings_ctx", None) is not None:
            await self._settings_ctx.__aexit__(None, None, None)
        await super().shutdown()
        self.processor.close()
