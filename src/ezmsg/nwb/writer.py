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
import datetime
import os
import re
import threading
import time
import typing
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import ezmsg.core as ez
import h5py
import numpy as np
import pynwb
from ezmsg.baseproc import BaseConsumer, BaseConsumerUnit
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


class NWBSinkConsumer(BaseConsumer[NWBSinkSettings, AxisArray]):
    # Session start datetime. It should have a valid timezone and that should be UTC.
    shared_start_datetime: typing.Optional[datetime.datetime] = None
    # Session start time.time. It does not have a timezone but unqualified conversions assume local time.
    shared_t0: typing.Optional[float] = None
    shared_clock_type: typing.Optional[ReferenceClockType] = None

    def __init__(self, *args, settings: typing.Optional[NWBSinkSettings] = None, **kwargs):
        super().__init__(*args, settings=settings, **kwargs)

        self._lock = threading.RLock()
        self._filepath = Path(self.settings.filepath)
        self._overwrite_old = self.settings.overwrite_old
        self._axis = self.settings.axis
        self._recording = self.settings.recording
        self._inc_clock = self.settings.inc_clock
        self._meta_yaml = self.settings.meta_yaml
        self._split_bytes = self.settings.split_bytes

        self._start_timestamp: float = 0.0
        self._split_count: int = 0
        self._stream_bytes = defaultdict(lambda: 0)
        self._current_msg: typing.Optional[AxisArray] = None
        self._io: typing.Optional[pynwb.NWBHDF5IO] = None
        self._nwbfile: typing.Optional[pynwb.NWBFile] = None
        self._datasets: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        self._settings_intervals_name = "pipeline_settings"
        self._settings_columns: list[str] = []
        self._settings_state: dict[str, typing.Any] = {}
        self._settings_active_since: typing.Optional[float] = None
        self._settings_prev_component: str = "__init__"

        # Normalize filepath and delete existing file if enabled.
        self._check_filepath()

        # Create the self._nwbfile and self._io objects. Note: Nothing written to disk yet!
        self._nwb_create_or_fail()

        if self.settings.expected_series is not None and Path(self.settings.expected_series).expanduser().exists():
            expected_series = Path(self.settings.expected_series).expanduser()
            meta = load_dict_from_file(expected_series)
            _ = self.get_session_datetime(None)
            self._start_timestamp = self.get_session_timestamp(None)
            self._prep_from_meta(meta)

    def __del__(self):
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
                if hasattr(self._current_msg, "period") and len(self._current_msg.period) > 0:
                    timestamp = timestamp + self._current_msg.period[0]
                self._current_msg = AxisArray(
                    data=np.array([self._current_msg.value]),
                    dims=["time"],
                    axes={"time": AxisArray.Axis(gain=1.0, offset=timestamp)},
                    key="epochs",
                )
            elif not hasattr(self._current_msg, "data"):
                return
            else:
                targ_ax_ix = self._current_msg.get_axis_idx(self._axis)
                if targ_ax_ix != 0:
                    self._current_msg = replace(
                        self._current_msg,
                        data=np.moveaxis(self._current_msg.data, targ_ax_ix, 0),
                        dims=[self._axis]
                        + self._current_msg.dims[:targ_ax_ix]
                        + self._current_msg.dims[targ_ax_ix + 1 :],
                    )

            # Is this a new series?
            b_new = self._io is None
            b_new = b_new or self._current_msg.key not in self._datasets

            # If inc message key is in datasets but properties do not match previous dataset properties
            #  then close io and raise error
            if not b_new and not self._check_msg_consistency():
                b_final_write = hasattr(self._nwbfile, "epochs") and self._nwbfile.epochs is not None
                b_final_write = b_final_write or (hasattr(self._nwbfile, "trials") and self._nwbfile.trials is not None)
                self.close(write=b_final_write)
                raise ValueError("Data provided to NWBSink has changed shape. Closing NWB file.")

            if b_new:
                # Use first incoming timestamp to set the session start time.
                key = self._current_msg.key
                t0 = None
                if self._axis in ["time", "win"] or "time" in self._current_msg.axes:
                    targ_dim = self._axis if self._axis in ["time", "win"] else "time"
                    if hasattr(self._current_msg.axes[targ_dim], "data"):
                        t0 = self._current_msg.axes[targ_dim].data[0]
                    else:
                        t0 = self._current_msg.axes[targ_dim].offset
                _ = self.get_session_datetime(t0)
                self._start_timestamp = self.get_session_timestamp(t0)
                if self._inc_clock == ReferenceClockType.MONOTONIC:
                    self._start_timestamp += time.monotonic() - time.time()

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

            if self._recording and self._current_msg.data.size:
                timestamps = None
                if self._axis in ["time", "win"] or "time" in self._current_msg.axes:
                    targ_dim = self._axis if self._axis in ["time", "win"] else "time"
                    time_ax = self._current_msg.axes[targ_dim]
                    if hasattr(time_ax, "data"):
                        timestamps = time_ax.data - self._start_timestamp
                    else:
                        timestamps = (np.arange(len(self._current_msg.data)) * time_ax.gain) + (
                            time_ax.offset - self._start_timestamp
                        )

                if self._current_msg.key in ["epochs", "trials"]:
                    self._append_events(self._current_msg.key, timestamps, self._current_msg.data)
                else:
                    # Write data
                    dataset = self._datasets[self._current_msg.key]["data"]
                    dataset.resize(len(dataset) + len(self._current_msg.data), axis=0)
                    dataset[-len(self._current_msg.data) :] = self._current_msg.data
                    self._stream_bytes[self._current_msg.key] += self._current_msg.data.nbytes

                    # Write timestamps
                    if timestamps is not None:
                        ts = self._datasets[self._current_msg.key]["ts"]
                        ts.resize(len(ts) + len(timestamps), axis=0)
                        ts[-len(timestamps) :] = timestamps
                        self._stream_bytes[self._current_msg.key] += timestamps.nbytes

                if 0 < self._split_bytes <= sum(self._stream_bytes.values()) and "%d" not in str(self._filepath):
                    split_timestamp = time.time()
                    self._flush_settings_interval(split_timestamp, self._settings_prev_component)
                    self._settings_prev_component = "__split__"
                    reopen_settings_since = split_timestamp if self._settings_state else None
                    self._settings_active_since = None
                    self._split_count += 1
                    self.path_on_disk.unlink(missing_ok=True)
                    new_nwbfile, new_meta = self._copy_nwb()
                    self.close()
                    self._nwb_create_or_fail(nwbfile=new_nwbfile)
                    self._prep_from_meta(new_meta)
                    self._settings_active_since = reopen_settings_since

    @property
    def path_on_disk(self) -> Path:
        fp = Path(self._filepath)
        if self._split_bytes > 0:
            if "%d" in str(fp):
                return Path(re.sub("%d", "0", str(fp)))
            else:
                return fp.parent / (fp.stem + f"_{self._split_count:02}" + fp.suffix)
        else:
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
        if self.__class__.shared_clock_type is not None and self.__class__.shared_clock_type != self._inc_clock:
            raise ValueError(
                f"All instances must share the same clock type. {self._inc_clock} != {self.__class__.shared_clock_type}"
            )
        if self.__class__.shared_start_datetime is None:
            if try_t0 is not None and self._inc_clock in [
                ReferenceClockType.SYSTEM,
                ReferenceClockType.MONOTONIC,
            ]:
                if self._inc_clock == ReferenceClockType.MONOTONIC:
                    try_t0 = try_t0 - time.monotonic() + time.time()
                self.__class__.shared_start_datetime = datetime.datetime.fromtimestamp(try_t0, datetime.timezone.utc)
            else:
                self.__class__.shared_start_datetime = datetime.datetime.now(datetime.timezone.utc)
        return self.__class__.shared_start_datetime

    def get_session_timestamp(self, try_t0: typing.Optional[float] = None) -> float:
        """
        Retrieve session timestamp. If it does not already exist, set it with try_t0 if provided.

        Args:
            try_t0: (Optional) The first incoming timestamp.

        Returns:
            Common session timestamp among all instances of this class.
        """
        if self.__class__.shared_clock_type is not None and self.__class__.shared_clock_type != self._inc_clock:
            raise ValueError(
                f"All instances must share the same clock type. {self._inc_clock} != {self.__class__.shared_clock_type}"
            )
        if self.__class__.shared_t0 is None:
            if try_t0 is None or self._inc_clock in [
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
        return self.__class__.shared_t0

    def _check_filepath(self) -> None:
        """
        Check self._filepath. Update path if necessary. Check if the path already exists and potentially raise
        an error if overwriting is disabled.
        """
        _suffix = ".nwb"

        if self._filepath.name.startswith("."):
            raise FileNotFoundError(
                f"filepath {self._filepath} name begins with `.` -- cannot discriminate name from extension."
            )

        filepath = Path(self._filepath).expanduser()

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

        self._filepath = filepath

        if self.path_on_disk.exists():
            age = (time.time() - os.path.getctime(self.path_on_disk)) / 60
            ez.logger.info(f"File at {self.path_on_disk} is {age:.2f} minutes old.")
            if self._overwrite_old:
                ez.logger.info("File will be overwritten.")
                self.path_on_disk.unlink(missing_ok=False)
            else:
                msg = "File exists but overwriting is disabled. Set overwrite_old=True to overwrite."
                ez.logger.error(msg)
                raise ValueError(msg)
        self._filepath = filepath

    def _read_meta_dict(self) -> typing.Union[typing.Mapping, dict]:
        """
        Load the metadata from self._meta_yaml if that path exists, else load it from the default location.

        Returns:
            A dict containing the metadata for this NWB file.
        """
        if self._meta_yaml is None or not Path(self._meta_yaml).expanduser().exists():
            default_path = Path(__file__).parent
            meta_dict = DeepDict()
            for yaml_name in ["nwb_metadata", "nwb_session"]:
                yaml_path = default_path / f"{yaml_name}.yaml"
                meta_dict = dict_deep_update(meta_dict, load_dict_from_file(yaml_path))
        else:
            yaml_path = Path(self._meta_yaml).expanduser()
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

        if self._settings_intervals_name in meta:
            settings_meta = meta[self._settings_intervals_name]
            self._prep_settings_intervals(settings_meta.get("columns", []))

        for key, ss in meta.items():
            if key not in ["epochs", "trials", self._settings_intervals_name]:
                shape = _sanitize_shape(ss["shape"])
                self._current_msg = AxisArray(
                    data=np.zeros(shape, dtype=self._current_msg.data.dtype),
                    dims=["time", "ch"] + [f"dim{_}" for _ in range(len(shape) - 2)],
                    axes={
                        "time": AxisArray.Axis.TimeAxis(fs=ss["fs"]),
                    },
                    key=key,
                )
                self._prep_continuous_io()

        # Flush and reopen to make h5py datasets
        self._flush_io(reopen=True)

        # Add the rate attribute to the timestamps series. Can only do this after flushing.
        for key, ss in meta.items():
            if key not in ["epochs", "trials"]:
                series = self._nwbfile.acquisition[key]
                series.timestamps.attrs["rate"] = ss["fs"]

    def close(self, write=False, log=True) -> None:
        """
        Close the file. This will also delete the file if it is empty.

        Args:
            write: Set True to write the file to disk before closing.
            log: Set True to log the closing and deletion of the file.
              This must be kept False when calling from __del__.
        """
        with self._lock:
            if self._io is not None:
                self._flush_settings_interval(time.time(), self._settings_prev_component)
                if write:
                    self._io.write(self._nwbfile)
                src_str = f"{self._io.source}"
                b_delete = sum(self._stream_bytes.values()) == 0
                for key in ["epochs", "trials"]:
                    if hasattr(self._nwbfile, key) and getattr(self._nwbfile, key) is not None:
                        b_delete = b_delete and len(getattr(self._nwbfile, key)) == 1  # EZNWB-START
                settings_intervals = self._get_settings_intervals()
                if settings_intervals is not None:
                    b_delete = b_delete and len(settings_intervals) == 1  # EZNWB-SETTINGS-START
                self._io.close()
                del self._nwbfile
                del self._io
                self._nwbfile = None
                self._io = None
                if log:
                    ez.logger.info(f"Closed file at {src_str}")
                if b_delete:
                    self.path_on_disk.unlink(missing_ok=True)
                    if log:
                        ez.logger.info(f"Deleted empty file at {src_str}.")

    def toggle_recording(self, recording: typing.Optional[bool] = None):
        with self._lock:
            self._recording = recording if recording is not None else not self._recording

    def _check_msg_consistency(self) -> bool:
        key = self._current_msg.key
        in_ax = self._current_msg.axes[self._axis]
        b_rate_change = (
            self._axis in self._current_msg.axes
            and "ts" in self._datasets[key]
            and not hasattr(in_ax, "data")
            and self._datasets[key]["ts"].attrs["rate"] != 1 / in_ax.gain
        )
        b_shape_change = self._datasets[key]["shape"] != self._current_msg.data.shape[1:]
        return not (b_rate_change or b_shape_change)

    def _update_rate_for_current(self):
        if self._axis in ["time", "win"]:
            time_ax = self._current_msg.axes[self._axis]
            if hasattr(time_ax, "data"):
                rate = 0.0
            else:
                rate = 1 / time_ax.gain if time_ax.gain != 0 else 0
            self._datasets[self._current_msg.key]["ts"].attrs["rate"] = rate

    def _copy_nwb(self) -> typing.Tuple[pynwb.NWBFile, dict]:
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
        new_nwb_kwargs = {k: getattr(self._nwbfile, k) for k in copy_keys if hasattr(self._nwbfile, k)}
        new_nwb_kwargs["keywords"] = (
            self._nwbfile.keywords if isinstance(self._nwbfile.keywords, list) else self._nwbfile.keywords[:].tolist()
        )
        nwbfile = pynwb.NWBFile(identifier=str(uuid4()), **new_nwb_kwargs)
        nwbfile.subject = pynwb.file.Subject(**self._nwbfile.subject.fields)
        meta = {}
        for key in ["epochs", "trials"]:
            if hasattr(self._nwbfile, key) and getattr(self._nwbfile, key) is not None:
                meta[key] = {"fs": 0.0, "shape": (0, 1)}
        if self._settings_columns:
            meta[self._settings_intervals_name] = {"columns": list(self._settings_columns)}
        for key, ds in self._datasets.items():
            if key not in ["epochs", "trials"]:
                meta[key] = {"fs": ds["ts"].attrs["rate"], "shape": (0,) + ds["shape"]}

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

        if "%d" in str(self._filepath):
            io_file = h5py.File(
                name=self._filepath,
                mode="w",
                driver="family",
                memb_size=self._split_bytes,
            )
            io = pynwb.NWBHDF5IO(file=io_file, mode="w")
        else:
            io = pynwb.NWBHDF5IO(self.path_on_disk, "w")

        self._io = io
        self._nwbfile = nwbfile
        self._stream_bytes = defaultdict(lambda: 0)

    def _flush_io(self, reopen: bool = True):
        """
        Write the header to the NWBFile.

        This step is also necessary to:
        * enable appending to our epochs/trials table (but only after it has an entry).
        * create the appendable datasets for our continuous data.
        """
        self._io.write(self._nwbfile)
        if reopen:
            if self._io:
                self._io.close()
            if "%d" in str(self._filepath):
                io_file = h5py.File(
                    name=self._filepath,
                    mode="a",
                    driver="family",
                    memb_size=self._split_bytes,
                )
                io = pynwb.NWBHDF5IO(file=io_file, mode="a")
            else:
                io = pynwb.NWBHDF5IO(self.path_on_disk, "a")
            self._io = io
            self._nwbfile = self._io.read()

        # Get references to our continuous datasets
        for k, v in self._datasets.items():
            if k in self._nwbfile.acquisition:
                series = self._nwbfile.acquisition[k]
                if isinstance(series.data, H5DataIO):
                    v["data"] = series.data.dataset
                    v["ts"] = series.timestamps.dataset
                else:
                    v["data"] = series.data
                    v["ts"] = series.timestamps

    def _configure_appendable_table(self, table: typing.Any) -> None:
        table.id.set_data_io(H5DataIO, {"maxshape": (None,), "chunks": True})
        for col in table.colnames:
            table[col].set_data_io(H5DataIO, {"maxshape": (None,), "chunks": True})

    def _append_events(
        self,
        key: str,
        timestamps: typing.Iterable[float],
        data: typing.Iterable[typing.Iterable[str]],
    ):
        fun = {"epochs": self._nwbfile.add_epoch, "trials": self._nwbfile.add_trial}[key]
        for ev_t, ev_str in zip(timestamps, data):
            fun(start_time=ev_t, stop_time=ev_t + 0, **{"label": ",".join(ev_str)})

    def _get_settings_intervals(self) -> typing.Any:
        if self._nwbfile is None or self._nwbfile.intervals is None:
            return None
        try:
            return self._nwbfile.intervals[self._settings_intervals_name]
        except Exception:
            return None

    def _prep_settings_intervals(self, settings_columns: typing.Iterable[str]) -> None:
        existing = self._get_settings_intervals()
        if existing is not None:
            return

        intervals = pynwb.epoch.TimeIntervals(
            name=self._settings_intervals_name,
            description="Flattened ezmsg settings snapshots active over each logged interval",
        )
        intervals.add_column(name="updated_component", description="component that triggered the snapshot transition")
        for column_name in settings_columns:
            intervals.add_column(name=column_name, description="flattened ezmsg setting")
        self._nwbfile.add_time_intervals(intervals)

        table = self._get_settings_intervals()
        if table is None:
            raise RuntimeError("Failed to create settings_intervals table")

        self._settings_columns = list(settings_columns)
        self._configure_appendable_table(table)

    def _settings_relative_time(self, timestamp: float) -> float:
        return timestamp - self.get_session_timestamp(None)

    def initialize_settings_state(self, flat_settings: dict[str, typing.Any], timestamp: float) -> None:
        with self._lock:
            if not flat_settings:
                return
            if not self._settings_columns:
                self._prep_settings_intervals(flat_settings.keys())
            missing_columns = [name for name in flat_settings if name not in self._settings_columns]
            if missing_columns:
                raise ValueError(
                    "Received settings fields not present in settings_intervals schema: "
                    f"{', '.join(sorted(missing_columns))}"
                )
            self._settings_state = {
                column_name: flat_settings.get(column_name, "") for column_name in self._settings_columns
            }
            self._settings_active_since = timestamp

            self.update_settings_state("__init__", flat_settings, self.get_session_timestamp())

            self._flush_io(reopen=True)

    def update_settings_state(
        self,
        component_address: str,
        flat_settings: dict[str, typing.Any],
        timestamp: float,
    ) -> None:
        with self._lock:
            if not flat_settings:
                return
            if not self._settings_columns:
                self._prep_settings_intervals(flat_settings.keys())
                self._settings_state = {column_name: "" for column_name in self._settings_columns}
                self._settings_active_since = timestamp

            missing_columns = [name for name in flat_settings if name not in self._settings_columns]
            if missing_columns:
                raise ValueError(
                    "Received settings fields not present in settings_intervals schema: "
                    f"{', '.join(sorted(missing_columns))}"
                )

            self._flush_settings_interval(timestamp, self._settings_prev_component)
            if not self._settings_state:
                self._settings_state = {column_name: "" for column_name in self._settings_columns}
            self._settings_state.update(flat_settings)
            self._settings_active_since = timestamp
            self._settings_prev_component = component_address

    def _flush_settings_interval(self, end_timestamp: float, updated_component: str) -> None:
        if not self._settings_state or self._settings_active_since is None:
            return

        table = self._get_settings_intervals()
        if table is None:
            return

        start_time = self._settings_relative_time(self._settings_active_since)
        stop_time = self._settings_relative_time(end_timestamp)
        if stop_time < start_time:
            stop_time = start_time

        table.add_interval(
            start_time=start_time,
            stop_time=stop_time,
            updated_component=updated_component,
            **self._settings_state,
        )

    def _prep_event_io(self):
        """
        Prepare the NWB file to receive event data, either "epochs" or "trials".
        """
        colname = "label"
        key = self._current_msg.key
        fun = {
            "epochs": self._nwbfile.add_epoch_column,
            "trials": self._nwbfile.add_trial_column,
        }[key]
        fun(name=colname, description=f"{colname} {key}")

        self._datasets[key] = {"shape": self._current_msg.data.shape[1:]}

        table = {"epochs": self._nwbfile.epochs, "trials": self._nwbfile.trials}[key]
        self._configure_appendable_table(table)

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
        targ_ax_ix = self._current_msg.get_axis_idx(self._axis)
        shape = self._current_msg.data.shape[:targ_ax_ix] + self._current_msg.data.shape[targ_ax_ix + 1 :]
        self._datasets[key] = {"shape": shape}
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
            b_first = self._nwbfile.electrodes is None or "label" not in self._nwbfile.electrodes.colnames
            if b_first:
                self._nwbfile.add_electrode_column(name="label", description="electrode label")

            dev_name = "unified device"
            if dev_name in self._nwbfile.devices:
                device = self._nwbfile.devices[dev_name]
            else:
                device = self._nwbfile.create_device(name=dev_name, description="created by ezmsg nwbsink")

            el_grp_name = "unified electrode group"
            if el_grp_name in self._nwbfile.electrode_groups:
                el_grp = self._nwbfile.electrode_groups[el_grp_name]
            else:
                el_grp = self._nwbfile.create_electrode_group(
                    name=el_grp_name,
                    description="electrode group created by ezmsg nwbsink",
                    device=device,
                    location="unknown",
                )
                if not b_first:
                    self._flush_io(reopen=True)

            el_df = self._nwbfile.electrodes.to_dataframe()
            el_df = el_df[el_df["group"] == el_grp]
            for ll in self._current_msg.axes["ch"].data:
                if ll not in el_df["label"].values:
                    self._nwbfile.add_electrode(label=ll, location="unknown", group=el_grp)

            if type(self._nwbfile.electrodes.id.data) is list:
                for fn in ["id", "location", "group_name", "group", "label"]:
                    getattr(self._nwbfile.electrodes, fn).set_data_io(H5DataIO, {"maxshape": (None,), "chunks": True})

            el_df = self._nwbfile.electrodes.to_dataframe()
            el_df = el_df[el_df["group"] == el_grp]
            b_in = el_df["label"].isin(self._current_msg.axes["ch"].data)
            el_tbl_region = self._nwbfile.create_electrode_table_region(
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
        self._nwbfile.add_acquisition(series)


class NWBSink(BaseConsumerUnit[NWBSinkSettings, AxisArray, NWBSinkConsumer]):
    SETTINGS = NWBSinkSettings

    INPUT_SETTINGS = ez.InputStream(NWBSinkSettings)

    async def initialize(self) -> None:
        await super().initialize()
        self._settings_ctx: typing.Optional[ez.GraphContext] = None
        self._settings_watch_task: typing.Optional[asyncio.Task[None]] = None
        self._settings_component_addresses: set[str] = set()
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

            flat_snapshot: dict[str, typing.Any] = {}
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
                await asyncio.to_thread(self.processor.initialize_settings_state, flat_snapshot, latest_timestamp)

            self._settings_watch_task = asyncio.create_task(
                self._watch_graph_settings(after_seq=last_seq),
                name=f"{self.address}-settings-watch",
            )

        except Exception as exc:
            ez.logger.warning(f"{self.address} could not initialize GraphServer-backed settings logging: {exc}")
            if self._settings_ctx is not None:
                await self._settings_ctx.__aexit__(None, None, None)
                self._settings_ctx = None

    def _session_component_addresses(self, graph_snapshot: typing.Any) -> set[str]:
        for session in graph_snapshot.sessions.values():
            metadata = session.metadata
            if metadata is None:
                continue
            if self.address in metadata.components:
                return set(metadata.components.keys())
        return set()

    async def _watch_graph_settings(self, after_seq: int) -> None:
        if self._settings_ctx is None:
            return
        async for event in self._settings_ctx.subscribe_settings_events(after_seq=after_seq):
            if event.component_address not in self._settings_component_addresses:
                continue
            flat_settings = flatten_component_settings(event.component_address, event.value)
            try:
                await asyncio.to_thread(
                    self.processor.update_settings_state,
                    event.component_address,
                    flat_settings,
                    event.timestamp,
                )
                self._settings_last_timestamp = event.timestamp
            except Exception as exc:
                ez.logger.warning(
                    f"{self.address} failed to record settings update for {event.component_address}: {exc}"
                )

    async def _bootstrap_processor_settings(self) -> None:
        if self._settings_ctx is None or not self._settings_component_addresses:
            return

        settings_snapshot = await self._settings_ctx.settings_snapshot()
        flat_snapshot: dict[str, typing.Any] = {}
        for component_address in sorted(self._settings_component_addresses):
            if component_address in settings_snapshot:
                flat_snapshot.update(
                    flatten_component_settings(component_address, settings_snapshot[component_address])
                )
        if flat_snapshot:
            await asyncio.to_thread(
                self.processor.initialize_settings_state,
                flat_snapshot,
                self._settings_last_timestamp,
            )

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: NWBSinkSettings) -> None:
        # Reset if settings _other than `recording`_ have changed.
        b_reset = msg.filepath != self.SETTINGS.filepath
        b_reset = b_reset or msg.overwrite_old != self.SETTINGS.overwrite_old
        b_reset = b_reset or msg.axis != self.SETTINGS.axis
        b_reset = b_reset or msg.inc_clock != self.SETTINGS.inc_clock
        b_reset = b_reset or msg.meta_yaml != self.SETTINGS.meta_yaml
        if b_reset:
            self.apply_settings(msg)
            self.create_processor()
            await self._bootstrap_processor_settings()
        elif msg.recording != self.SETTINGS.recording:
            self.processor.toggle_recording(msg.recording)

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
