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

from .clockmodel import (
    DEFAULT_DEJITTER_KNOTS,
    DEFAULT_DEJITTER_METHOD,
    cache_lookup,
    cache_store,
    group_clocks,
    reconstruct_group,
)
from .util import ReferenceClockType, as_text, as_text_array

# Default gap threshold as a fraction of the nominal sample period (1.5x period).
# Sits between neural-data jitter (<~1.05x) and the smallest real gap (one dropped
# sample = ~2x), so it catches every gap without splitting on jitter. Shared as the
# default for the slicer, iterator, and clock-driven producer.
DEFAULT_GAP_TOL = 0.5


def find_gaps(timestamps: np.ndarray, gain: float, gap_tol: float) -> np.ndarray:
    """Indices ``i`` where a gap separates sample ``i`` from sample ``i + 1``.

    A gap is an inter-sample interval exceeding ``(1 + gap_tol) * gain`` — i.e.
    the timestamps jump by more than ``(1 + gap_tol)`` nominal sample periods.
    Returns the left-edge indices as an ``int`` array (empty when there are
    fewer than two samples, no usable ``gain``, or no gaps). Shared by the
    iterator (which splits chunks at gaps) and the slicer's ``read_by_time``
    (which switches to a CoordinateAxis when a window spans one).
    """
    if gain <= 0.0 or timestamps.shape[0] < 2:
        return np.empty(0, dtype=int)
    return np.flatnonzero(np.diff(timestamps) > gain * (1.0 + gap_tol))


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


def _is_bytes_text(dset: typing.Any) -> bool:
    """True when this dataset's values arrive as ``bytes`` rather than ``str``.

    Covers both shapes the character-set problem takes: a fixed-length column
    (dtype ``S``) and a variable-length one declared ASCII, which h5py hands
    back as an object array of ``bytes``. See :func:`~.util.as_text`.
    """
    dtype = getattr(dset, "dtype", None)
    if dtype is None:
        return False
    if dtype.kind == "S":
        return True
    if dtype.kind != "O" or 0 in dset.shape:
        return False
    return isinstance(dset[(0,) * dset.ndim], bytes)


def _electrode_group_manufacturer(child: pynwb.TimeSeries) -> str:
    """Return the manufacturer of the Device backing this TimeSeries.

    Only meaningful for ElectricalSeries / FeatureExtraction-style containers
    whose ``electrodes`` DynamicTableRegion links back to an ElectrodeGroup →
    Device. Returns ``""`` when the link is absent, unreadable, or the
    Device's ``manufacturer`` attribute is unset/"unknown".
    """
    try:
        electrodes = getattr(child, "electrodes", None)
        if electrodes is None:
            return ""
        # ``electrodes`` is a DynamicTableRegion; .table → ElectrodesTable.
        table = getattr(electrodes, "table", None)
        if table is None:
            return ""
        # Grab any electrode row's group (they all share the device for an
        # ElectricalSeries with a single ElectrodeGroup).
        group_col = table["group"]
        if len(group_col) == 0:
            return ""
        group = group_col[0]
        device = getattr(group, "device", None)
        if device is None:
            return ""
        # ``as_text``, not ``str``: this value becomes a prefix to match on, and
        # a bytes repr ("b'CereLink'") matches nothing, so every stream the
        # caller named by bare device would be skipped in silence. hdmf happens
        # to decode ASCII *attributes* on the way out even though it leaves
        # datasets alone -- so this is a guard against that asymmetry changing,
        # not a live bug.
        manufacturer = as_text(getattr(device, "manufacturer", "") or "")
        if manufacturer.lower() == "unknown":
            return ""
        return manufacturer
    except (AttributeError, KeyError, IndexError, TypeError):
        return ""


def _match_stream_key(child: pynwb.TimeSeries, stream_keys: list[str] | None) -> str | None:
    """Resolve the user-facing key for a TimeSeries given the *stream_keys* filter.

    Returns:
    * ``child.name`` if *stream_keys* is None (no filter — keep the literal
      NWB container name).
    * ``child.name`` if it appears in *stream_keys* (exact match).
    * The bare ``<key>`` portion if ``child.name == f"{manufacturer}_{key}"``
      for some ``<key>`` in *stream_keys* and the linked Device's
      ``manufacturer`` attribute equals ``<manufacturer>``. This keeps
      configs that name the bare device (e.g. ``["NPLAY"]``) working when
      the NWB writer prefixes containers with the manufacturer (e.g.
      ``"CereLink_NPLAY"``). The returned ``<key>`` is then used as the
      stream's storage key and template ``.key``, so downstream consumers
      see the name they asked for.
    * ``None`` when nothing matches and the stream should be skipped.
    """
    if stream_keys is None:
        return child.name
    if child.name in stream_keys:
        return child.name
    manufacturer = _electrode_group_manufacturer(child)
    if not manufacturer:
        return None
    prefix = f"{manufacturer}_"
    if not child.name.startswith(prefix):
        return None
    stripped = child.name[len(prefix) :]
    if stripped in stream_keys:
        return stripped
    return None


class NWBSlicer:
    """Shared NWB file handling: open, discover streams, and slice data.

    Args:
        filepath: Path to NWB file (local or remote URL).
        reference_clock: Which reference clock to use for timestamps.
        reref_now: If True, re-reference timestamps to the current time.
        stream_keys: Optional filter for which streams to discover.
        rdcc_nbytes: HDF5 raw data chunk cache size in bytes. Larger values reduce
            re-decoding when consecutive reads cross HDF5 chunk boundaries.
        rdcc_nslots: HDF5 raw data chunk cache slot count. Should be a prime number
            roughly 10x the number of chunks expected to be hot at once.
        dejitter: When True (default), reconstruct smoothed, strictly-monotone
            timestamps for continuous streams that carry a paired ``*_device_ts``
            sibling (the structural mark of a re-timestamped fixed-rate
            acquisition). This collapses the per-sample clock jitter that would
            otherwise shatter such a stream into thousands of gap-split chunks.
            Streams that share a device clock (PTP-synchronised hubs) are
            detected automatically and the cleanest one supplies the clock model
            for the rest; see :mod:`~ezmsg.nwb.clockmodel`. Set False to read raw
            timestamps unchanged.
        clock_groups: Override auto clock-group detection. ``None`` auto-detects;
            a list of key-lists forces exactly those groupings (unlisted
            correctable streams become singleton self-fits). Ignored when
            ``dejitter`` is False.
        dejitter_cache: Cache reconstructed timestamp vectors on disk (default
            True) so repeated opens of the same file -- e.g. multi-pass training
            -- reconstruct once and reload thereafter.
        real_gap_threshold: Guard against smoothing over *genuine* data gaps
            (dropped packets). ``None`` (default) auto-derives a per-stream jump
            threshold from that stream's jitter envelope, so only jumps
            unambiguously above jitter are treated as real gaps; a float sets an
            explicit threshold in seconds. Real gaps are preserved (each inter-gap
            segment is reconstructed independently) so the iterator still splits a
            chunk there. Pass ``float("inf")`` to disable the guard and smooth
            everything. Ignored when ``dejitter`` is False.
    """

    disk_cache = remfile.DiskCache(str(Path("~").expanduser() / ".ezmsg" / "nwb-cache"))

    DEFAULT_RDCC_NBYTES: typing.ClassVar[int] = 64 * 1024 * 1024
    DEFAULT_RDCC_NSLOTS: typing.ClassVar[int] = 10007

    def __init__(
        self,
        filepath: typing.Union[os.PathLike, str],
        reference_clock: ReferenceClockType = ReferenceClockType.SYSTEM,
        reref_now: bool = False,
        stream_keys: typing.Optional[list[str]] = None,
        rdcc_nbytes: int = DEFAULT_RDCC_NBYTES,
        rdcc_nslots: int = DEFAULT_RDCC_NSLOTS,
        dejitter: bool = True,
        clock_groups: typing.Optional[list[list[str]]] = None,
        dejitter_cache: bool = True,
        real_gap_threshold: typing.Optional[float] = None,
    ):
        self._filepath = filepath
        self._reference_clock = reference_clock
        self._reref_now = reref_now
        self._stream_keys = stream_keys
        self._rdcc_nbytes = rdcc_nbytes
        self._rdcc_nslots = rdcc_nslots
        self._dejitter = dejitter
        self._clock_groups = clock_groups
        self._dejitter_cache = dejitter_cache
        self._real_gap_threshold = real_gap_threshold

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
            file = h5py.File(f, rdcc_nbytes=self._rdcc_nbytes, rdcc_nslots=self._rdcc_nslots)
            self._io = pynwb.NWBHDF5IO(file=file)
        else:
            file = h5py.File(self._filepath, "r", rdcc_nbytes=self._rdcc_nbytes, rdcc_nslots=self._rdcc_nslots)
            self._io = pynwb.NWBHDF5IO(file=file)
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
                    # ``as_text_array``, not ``map(str, ...)``: an ASCII-declared
                    # column reads back as ``bytes``, and stringifying those gives
                    # the repr -- an event payload of "b'{\"cause\": ...}'" that no
                    # consumer can match or parse, with nothing raised to say so.
                    dset = np.array([as_text_array(table[_].data) for _ in ch_labels]).T
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
        # Dedupe by NWB hierarchy address while preserving discovery order.
        # ``set(all_timeseries)`` works but its iteration order depends on
        # object hashes (which differ between file opens), so it produced
        # a different stream order on every NWBSlicer instance — surprising
        # for any code that compares messages from two iterators.
        seen_addresses: set[str] = set()
        deduped: list[tuple[str, pynwb.TimeSeries]] = []
        for addr, ts in all_timeseries:
            if addr in seen_addresses:
                continue
            seen_addresses.add(addr)
            deduped.append((addr, ts))
        all_timeseries = deduped

        # When a stream_keys filter is set and dejitter is on, also admit each
        # requested stream's ``*_device_ts`` partner into discovery even though
        # the user didn't list it: the dejitter pass needs the device timebase.
        # These partners are pruned from the public stream set after
        # reconstruction (see ``_apply_dejitter``), so they never surface as
        # iterable streams -- the filter behaves as requested while dejitter
        # still fires. Without a filter (None) every stream is already present.
        match_keys = self._stream_keys
        if self._dejitter and self._stream_keys is not None:
            match_keys = list(self._stream_keys) + [f"{k}_device_ts" for k in self._stream_keys]

        for address, child in all_timeseries:
            if type(child) is pynwb.misc.Units:
                ez.logger.warning("Units found in NWB file. Not yet supported.")
                continue
            if not isinstance(child, pynwb.TimeSeries):
                continue
            matched_key = _match_stream_key(child, match_keys)
            if matched_key is None:
                continue
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
                    # Decoded here, at the read boundary: these labels become the
                    # ch-axis coordinates every downstream name-based channel
                    # selection matches against.
                    ch_labels = as_text_array(el_df["label"].values)
                else:
                    ch_labels = np.array([f"ch_{idx}" for idx in el_df.index.tolist()])
                axes["ch"] = AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"])

            # ``matched_key`` is the user-facing key — equal to ``child.name``
            # when there's no stream_keys filter or an exact match, or the
            # bare ``<key>`` portion when the user requested an unprefixed
            # device name and the container is ``"<manufacturer>_<key>"``.
            # A text series (markers, annotations) whose writer declared its
            # strings ASCII reads back as ``bytes``. Decode it once here rather
            # than at each slice: every caller then sees ``str`` regardless of
            # writer, and the three places that index ``dset`` -- both slicer
            # read paths and the iterator's -- can't each forget to. Safe to
            # materialize because text series are markers: short and few, on the
            # order of the interval tables already read whole just above.
            dset = as_text_array(child.data[:]) if _is_bytes_text(child.data) else child.data

            self._streams[matched_key] = StreamInfo(
                dset=dset,
                template=AxisArray(
                    data=np.zeros((0,) + dset.shape[1:], dtype=dset.dtype),
                    dims=(["time", "ch"] + [f"dim_{_}" for _ in range(2, child.data.ndim)])
                    if child.data.ndim > 1
                    else ["time"],
                    axes=axes,
                    key=matched_key,
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

        self._apply_dejitter()

    # --- Dejitter / clock reconstruction ---

    @staticmethod
    def _data_addr(dset: typing.Any) -> int | None:
        """Object-header address of an h5py dataset, or None if unavailable.

        Two streams whose ``data`` share an address are the same underlying
        object (a hard link) -- here, the ``/acquisition`` copy and its
        ``*_device_ts`` sibling in ``/processing`` are literally one dataset
        re-timestamped, which is the reliable structural signal that a stream is
        a re-timestamped fixed-rate acquisition. Non-datasets (materialized text
        arrays) and any error yield None, so the caller falls back to the name
        convention rather than crashing.
        """
        try:
            if isinstance(dset, h5py.Dataset):
                return int(h5py.h5o.get_info(dset.id).addr)
        except Exception:
            return None
        return None

    def _file_signature(self) -> str:
        """Cache key for the source file: path + size + mtime (path only for
        remote URLs, which have no cheap stat)."""
        p = str(self._filepath)
        if p.startswith("http"):
            return p
        try:
            st = os.stat(p)
            return f"{p}:{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            return p

    def _find_device_pairs(self) -> dict[str, str]:
        """Map each correctable converted stream to its ``*_device_ts`` partner.

        A stream is correctable when a sibling named ``<key>_device_ts`` exists
        and (when both expose an object address) the two share the same ``data``
        object. When addresses are unavailable the name convention alone is
        trusted; when addresses are available and differ, the pair is rejected.
        """
        pairs: dict[str, str] = {}
        suffix = "_device_ts"
        for k, info in self._streams.items():
            if k.endswith(suffix):
                continue
            if info.is_event or not info.has_timestamps or info.timestamps is None:
                continue
            dk = f"{k}{suffix}"
            dinfo = self._streams.get(dk)
            if dinfo is None or dinfo.timestamps is None:
                continue
            a = self._data_addr(info.dset)
            b = self._data_addr(dinfo.dset)
            if a is not None and b is not None and a != b:
                continue
            pairs[k] = dk
        return pairs

    def _resolve_groups(self, pairs: dict[str, str]) -> list[list[str]]:
        """Decide which correctable streams share a clock.

        Honours an explicit ``clock_groups`` override (keeping only correctable
        keys, with any leftover correctable stream as its own singleton);
        otherwise auto-detects from the device-time endpoints, which are two
        cheap scalar reads per stream rather than a full timestamp materialization.
        """
        if self._clock_groups is not None:
            groups = [[k for k in g if k in pairs] for g in self._clock_groups]
            groups = [g for g in groups if g]
            covered = {k for g in groups for k in g}
            groups.extend([k] for k in pairs if k not in covered)
            return groups

        endpoints: list[dict] = []
        for ck, dk in pairs.items():
            dts = self._streams[dk].timestamps
            endpoints.append({"key": ck, "t0": float(dts[0]), "tend": float(dts[-1]), "n": len(dts)})
        return group_clocks(endpoints)

    def _replace_timestamps(self, conv_key: str, arr: np.ndarray) -> None:
        """Swap in reconstructed timestamps for a stream.

        ``info.t0`` is intentionally left as ``_load`` set it (the raw first
        timestamp): it feeds the iterator's epoch-scale chunk-grid subtraction,
        whose rounding is knife-edge sensitive, and keeping it identical to the
        non-dejitter path avoids dropping a boundary sample. The reconstructed
        first-sample time differs from the raw one only by sub-sample jitter, so
        the grid is unaffected while the per-sample ``timestamps`` -- what the
        gap-splitter and chunk anchoring actually read -- are the clean ones.
        """
        self._streams[conv_key].timestamps = arr

    def _apply_dejitter(self) -> None:
        """Reconstruct smoothed monotone timestamps for correctable streams.

        No-op unless ``dejitter`` is set and at least one stream carries a
        ``*_device_ts`` partner. Groups sharing a clock are reconstructed
        together (the cleanest member supplies the model); results are cached on
        disk keyed by file signature and fit params so repeat opens skip the work.
        """
        if not self._dejitter:
            return
        pairs = self._find_device_pairs()
        if not pairs:
            return

        params = (DEFAULT_DEJITTER_METHOD, DEFAULT_DEJITTER_KNOTS, self._real_gap_threshold)
        signature = self._file_signature()

        for group in self._resolve_groups(pairs):
            cached = {ck: (cache_lookup(signature, ck, params) if self._dejitter_cache else None) for ck in group}
            if all(v is not None for v in cached.values()):
                for ck, arr in cached.items():
                    self._replace_timestamps(ck, arr)
                continue

            # Cache miss somewhere in the group: materialize the whole group's
            # timestamps (a shared model needs every member) and reconstruct.
            members = []
            for ck in group:
                dk = pairs[ck]
                members.append(
                    {
                        "key": ck,
                        "dataset": np.asarray(self._streams[ck].timestamps[:], dtype=float),
                        "device": np.asarray(self._streams[dk].timestamps[:], dtype=float),
                    }
                )
            recon = reconstruct_group(
                members, DEFAULT_DEJITTER_KNOTS, DEFAULT_DEJITTER_METHOD, self._real_gap_threshold
            )
            for ck, arr in recon.items():
                self._replace_timestamps(ck, arr)
                if self._dejitter_cache:
                    cache_store(signature, ck, params, arr)

        self._prune_hidden_device_streams()
        self._recompute_time_bounds()

    def _recompute_time_bounds(self) -> None:
        """Recompute the global stop time over the real playback streams.

        A ``*_device_ts`` partner carries absolute PTP-epoch timestamps
        (~1.78e9 s), not session-relative ones. Their epoch values never lower
        the ``min`` that sets ``start_time`` (so :meth:`_load`'s value stands),
        but they do dominate the ``max`` that sets ``stop_time``, inflating it by
        a whole epoch and blowing up the iterator's chunk count. Only
        ``stop_time`` is recomputed here, excluding those partners by name;
        ``start_time`` is deliberately left untouched so it stays bit-for-bit
        equal to the non-dejitter path -- the iterator derives its chunk grid
        from ``start_time`` and each stream's ``t0`` via an epoch-scale
        subtraction whose rounding is knife-edge sensitive, so shifting
        ``start_time`` by even a reconstructed microsecond can drop a boundary
        sample. Mirrors the per-stream ``stop_time`` contribution in :meth:`_load`.
        """
        ts_off = float(self._ts_off)
        stop_time = -np.inf
        for key, info in self._streams.items():
            if key.endswith("_device_ts"):
                continue
            # ``float(...)``: ``info.fs`` is often a float32 rate attribute, and a
            # float32 ``gain`` would drag this epoch-scale (~1.78e9 s) sum down to
            # float32 under NumPy's weak-scalar promotion -- whose ULP up there is
            # ~128 s, enough to shift the bound by minutes. Keep every term a
            # Python float so the arithmetic stays float64.
            gain = 1.0 / float(info.fs) if info.fs else 1.0
            if info.is_event and info.table_ref is not None:
                stop_time = max(stop_time, ts_off + float(info.table_ref.stop_time[-1]))
            elif info.has_timestamps and info.timestamps is not None:
                ts = info.timestamps
                stop_time = max(stop_time, ts_off + float(ts[-1]) + gain)
                stop_time = max(stop_time, ts_off + float(ts[0]) + (info.n_samples + 1) * gain)
            else:
                stop_time = max(stop_time, ts_off + float(info.t0) + (info.n_samples + 1) * gain)
        if np.isfinite(stop_time):
            self._stop_time = stop_time

    def _prune_hidden_device_streams(self) -> None:
        """Drop ``*_device_ts`` partners that were auto-added under a filter.

        Only removes partners the user did not explicitly request, so a filtered
        read exposes exactly the streams asked for while dejitter still had the
        device timebase it needed. No-op without a ``stream_keys`` filter (every
        stream is intentionally present then).
        """
        if not self._dejitter or self._stream_keys is None:
            return
        wanted = set(self._stream_keys)
        for k in list(self._streams):
            if k.endswith("_device_ts") and k not in wanted:
                del self._streams[k]

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
            # Explicit timestamps are already absolute (file-relative) times.
            chunk_t0 = info.timestamps[start_idx]
        else:
            # Rate-only: ``start_idx`` is 0-based from the stream's own start, so
            # its absolute time is ``info.t0`` plus the within-stream offset.
            # Omitting ``info.t0`` would mis-time a stream whose ``starting_time``
            # is non-zero and disagree with the clock-driven producer's
            # ``file_t = t0 + idx/fs`` bookkeeping.
            chunk_t0 = float(info.t0) + template.axes["time"].gain * start_idx

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

    def read_by_time(
        self, stream_key: str, t_start: float, t_end: float, gap_tol: float = DEFAULT_GAP_TOL
    ) -> AxisArray:
        """Read data by time window [t_start, t_end).

        For timestamped continuous streams and event/interval tables.
        t_start and t_end are in the same reference frame as the stored timestamps
        (i.e., file-relative, before ts_off).

        ``gap_tol`` is the gap threshold as a fraction of the nominal sample
        period (see :func:`find_gaps`). When a continuous window's samples are
        regularly spaced the result carries a cheap ``LinearAxis``; when the
        window straddles a gap, the time axis becomes a ``CoordinateAxis``
        carrying the true per-sample timestamps so the gap is represented
        faithfully instead of silently flattening post-gap samples onto a
        uniform axis. Ignored for event streams.
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
            time_axis = template.axes["time"]
            ts_window = np.asarray(ts_arr[start_idx:stop_idx])
            has_gain = hasattr(time_axis, "gain")

            # A window straddling a gap (or any stream with no usable rate)
            # cannot be described by a uniform LinearAxis without misplacing
            # samples. Emit a CoordinateAxis carrying the true per-sample
            # timestamps instead — the same representation events use. Gap-free
            # windows keep the cheaper LinearAxis.
            emit_coords = (not has_gain) or find_gaps(ts_window, time_axis.gain, gap_tol).size > 0
            if emit_coords:
                return replace(
                    template,
                    data=out_data,
                    axes={
                        **template.axes,
                        "time": AxisArray.CoordinateAxis(
                            data=self._ts_off + ts_window,
                            dims=["time"],
                            unit=getattr(time_axis, "unit", "s"),
                        ),
                    },
                    key=stream_key,
                )

            if start_idx < len(ts_arr):
                chunk_t0 = ts_arr[start_idx]
            else:
                chunk_t0 = time_axis.gain * start_idx

            return replace(
                template,
                data=out_data,
                axes={
                    **template.axes,
                    "time": replace(
                        time_axis,
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
