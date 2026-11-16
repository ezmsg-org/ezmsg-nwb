"""Clock-driven NWB producer for synchronizing NWB data to a shared clock."""

from __future__ import annotations

import asyncio
import os
import typing
import warnings

import ezmsg.core as ez
from ezmsg.baseproc import end_of_stream
from ezmsg.baseproc.clockdriven import BaseClockDrivenProducer, ClockDrivenSettings, ClockDrivenState
from ezmsg.baseproc.protocols import processor_state
from ezmsg.baseproc.units import BaseClockDrivenUnit
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis
from ezmsg.util.messages.util import replace

from .clockmodel import GAP_MIN_S
from .slicer import NWBSlicer
from .util import ReferenceClockType


class NWBClockDrivenSettings(ClockDrivenSettings):
    filepath: typing.Union[os.PathLike, str] = ""
    stream_key: str = ""
    reference_clock: ReferenceClockType = ReferenceClockType.SYSTEM
    reref_now: bool = False
    fs: float = 0.0  # 0 = auto-detect from file
    start_offset: float = 0.0
    """Playback position (seconds from the stream's t0) at which to begin reading."""
    playback_rate: float = 1.0
    """Multiplier applied to the clock-tick duration when advancing through the file.

    ``1.0`` is real-time, ``2.0`` consumes two seconds of file per wall-clock second,
    ``0.0`` pauses (the producer emits nothing). Values other than 0.0/1.0 are only
    meaningful when ``n_time`` is ``None`` (variable chunk size); with a fixed
    ``n_time`` the rate acts as a pause/unpause gate since chunk size is fixed.
    """
    gap_tol: typing.Optional[float] = None
    """.. deprecated:: Ignored. Gaps are now found with ``gap_tolerance``."""
    gap_tolerance: float = GAP_MIN_S
    """The smallest margin, in seconds, above one sample period that an interval
    between samples must exceed to count as a gap. The margin is never less than
    one period, and a lone late timestamp is not a gap; see
    :meth:`~ezmsg.nwb.slicer.NWBSlicer.segment_starts`. Gaps are looked for in the
    stream's ``*_device_ts`` partner when it has one.

    The clock cadence is never altered: ``file_t`` always advances by ``dt`` per
    tick so sibling producers on the same clock stay aligned, and windows that
    fall inside a gap emit zero-length chunks. A window that straddles a gap emits
    only the samples before it; those after it go out on the next tick, as the
    start of a new :attr:`~ezmsg.util.messages.axisarray.AxisArray.segment`, so
    that tick's chunk is longer than usual. A seek starts a new segment too.
    No effect on rate-only or event streams, which have no gaps.

    Files with heavy timestamp jitter (several ms) should keep ``dejitter`` on:
    only that pass widens the threshold to clear the stream's own jitter.
    """
    dejitter: bool = True
    """Reconstruct smoothed monotone timestamps for streams with a paired
    ``*_device_ts`` sibling (forwarded to ``NWBSlicer``). Set False to read raw
    timestamps unchanged."""
    clock_groups: typing.Optional[list[list[str]]] = None
    """Override auto clock-group detection (forwarded to ``NWBSlicer``)."""
    dejitter_cache: bool = True
    """Cache reconstructed timestamps on disk across opens (forwarded to
    ``NWBSlicer``)."""
    real_gap_threshold: typing.Optional[float] = None
    """Explicit gap threshold in seconds, as an interval between samples (forwarded
    to ``NWBSlicer``). ``None`` derives it from ``gap_tolerance``;
    ``float("inf")`` turns gap handling off."""
    structured_ch_axis: bool = False
    """Build the ``ch`` axis as a structured record array from the electrodes
    table (forwarded to ``NWBSlicer``), matching what a live acquisition source
    emits: position, label, bank, headstage, array identity. Default False --
    these columns are an acquisition-stack convention, not part of the NWB
    schema. See :mod:`~ezmsg.nwb.electrodes`."""
    storage_options: typing.Optional[dict] = None
    """fsspec options for a remote Zarr store (e.g. ``account_name`` for ``abfs://``)."""

    def __post_init__(self) -> None:
        if self.gap_tol is not None:
            warnings.warn("gap_tol is ignored; use gap_tolerance (seconds).", DeprecationWarning, stacklevel=3)


@processor_state
class NWBClockDrivenState(ClockDrivenState):
    slicer: NWBSlicer | None = None
    detected_fs: float = 0.0
    has_timestamps: bool = False
    is_event: bool = False
    template: AxisArray | None = None
    sample_idx: int = 0
    n_total_samples: int = 0
    t0: float = 0.0  # Stream start time (file-relative, before ts_off)
    file_t: float = 0.0  # Current playback position (file-relative, from stream t0)
    ts_idx: int | None = None
    """Timestamped streams: index of the next sample to emit. None after a seek,
    when it has to be looked up from ``file_t``."""
    segment: int = -1
    """Counter for the output ``segment``; the seek on opening the file makes it 0. Never rewound,
    so a seek or a reopened file cannot repeat the previous number."""
    last: AxisArray | None = None
    """The last message with samples, until the end-of-stream marker that follows it is sent."""


class NWBClockDrivenProducer(BaseClockDrivenProducer[NWBClockDrivenSettings, NWBClockDrivenState]):
    """Produces NWB data synchronized to clock ticks.

    Three extraction strategies based on stream type:
      1. Rate-only continuous: sample-index-based using fs and counter.
      2. Timestamped continuous: time-window-based using searchsorted.
      3. Events/intervals: time-window-based, returning all events in window.

    Playback can start at an arbitrary offset and run at an arbitrary rate
    relative to the driving clock; see ``start_offset`` and ``playback_rate``
    on :class:`NWBClockDrivenSettings`.
    """

    # Fields that can change without forcing a slicer teardown + reopen. The
    # producer rebinds ``self.settings`` live and, for ``start_offset``,
    # follows up with :meth:`seek`. Anything else — filepath, stream_key,
    # reference_clock, reref_now, fs — routes through ``_request_reset`` and
    # the next clock tick triggers a fresh :meth:`_reset_state`.
    NONRESET_SETTINGS_FIELDS = frozenset({"start_offset", "playback_rate", "n_time"})

    def update_settings(self, new_settings: NWBClockDrivenSettings) -> None:
        old_offset = self.settings.start_offset
        super().update_settings(new_settings)
        # If super() queued a reset, _reset_state will re-seek from the new
        # start_offset; only seek on the hot path.
        if self._hash != -1 and new_settings.start_offset != old_offset:
            self.seek(new_settings.start_offset)

    @property
    def exhausted(self) -> bool:
        if self._state.slicer is None:
            return False  # Not yet initialized / idle — not exhausted
        if self._state.is_event or self._state.has_timestamps:
            return False  # Time-window streams don't track exhaustion via index
        return self._state.sample_idx >= self._state.n_total_samples and self._state.last is None

    async def _areset_state(self, time_axis: LinearAxis) -> None:
        """Offload the slow ``NWBSlicer`` open onto a worker thread so that
        sibling subscribers (notably ``INPUT_SETTINGS`` for live scrubbing)
        can keep running while the file is opened. First-message NWB I/O
        can take 1.5–4 seconds per the writer-module docstring; without
        this, the unit's event loop is held for the whole duration.
        """
        await asyncio.to_thread(self._reset_state, time_axis)

    def _reset_state(self, time_axis: LinearAxis) -> None:
        if self._state.slicer is not None:
            self._state.slicer.close()
            self._state.slicer = None

        # Idle mode: no file / stream configured yet. Leave the slicer as
        # None so ``_process`` short-circuits. A later settings push that
        # sets ``filepath`` / ``stream_key`` falls outside
        # ``NONRESET_SETTINGS_FIELDS``, so ``update_settings`` queues a reset
        # and the next clock tick reruns this method and loads the file.
        if not self.settings.filepath or not self.settings.stream_key:
            self._state.template = None
            self._state.has_timestamps = False
            self._state.is_event = False
            self._state.n_total_samples = 0
            self._state.t0 = 0.0
            self._state.sample_idx = 0
            self._state.file_t = 0.0
            self._state.detected_fs = 0.0
            self._state.fractional_samples = 0.0
            return

        # Only ``stream_key`` is requested; when dejitter is on the slicer
        # transparently pulls in this stream's ``*_device_ts`` partner for the
        # reconstruction and hides it again, so a single-stream read is still
        # dejittered (self-fit, since only this stream's group is loaded -- the
        # iterator path gets the full cross-hub shared model).
        slicer = NWBSlicer(
            filepath=self.settings.filepath,
            reference_clock=self.settings.reference_clock,
            reref_now=self.settings.reref_now,
            stream_keys=[self.settings.stream_key],
            dejitter=self.settings.dejitter,
            clock_groups=self.settings.clock_groups,
            dejitter_cache=self.settings.dejitter_cache,
            real_gap_threshold=self.settings.real_gap_threshold,
            structured_ch_axis=self.settings.structured_ch_axis,
            storage_options=self.settings.storage_options,
            gap_tolerance=self.settings.gap_tolerance,
        )
        self._state.slicer = slicer

        info = slicer.get_stream_info(self.settings.stream_key)
        self._state.has_timestamps = info.has_timestamps and not info.is_event
        self._state.is_event = info.is_event
        self._state.template = info.template
        self._state.n_total_samples = info.n_samples
        self._state.t0 = info.t0

        # Auto-detect fs if not specified
        if self.settings.fs == 0.0:
            if info.fs == 0.0 and not info.is_event:
                ez.logger.warning(
                    f"Stream '{self.settings.stream_key}' has fs=0 and is not an event stream. "
                    "Clock-driven chunk sizing may not work correctly."
                )
            self._state.detected_fs = info.fs
        else:
            self._state.detected_fs = self.settings.fs

        # Apply start_offset. sample_idx is used by the rate-only path;
        # file_t is used by the time-window path. Both are kept in sync so
        # that hot-updates that switch between read strategies stay coherent.
        self._state.fractional_samples = 0.0
        self.seek(self.settings.start_offset)

    def seek(self, start_offset: float) -> None:
        """Move the playback position to ``start_offset`` seconds past the stream's t0.

        Safe to call after ``_reset_state`` has run. The file is not reopened
        and no slicer state is torn down, so this is cheap enough to call
        every time a scrub-widget emits.
        """
        fs = self._state.detected_fs
        if fs > 0:
            self._state.sample_idx = max(0, int(start_offset * fs))
        else:
            self._state.sample_idx = 0
        self._state.file_t = self._state.t0 + start_offset
        # What comes next does not continue what came before.
        self._state.ts_idx = None
        self._state.segment += 1
        self._state.last = None

    def _get_fs(self) -> float:
        """Return the effective sample rate (user-specified or auto-detected)."""
        return self._state.detected_fs

    def _process(self, clock_tick: LinearAxis) -> AxisArray | None:
        # Idle: no file configured yet — wait for a settings push to
        # populate ``filepath`` / ``stream_key``.
        if self._state.slicer is None:
            return None
        # Pause gate: ``playback_rate=0`` halts emission without tearing
        # anything down, so a resume simply continues from file_t / sample_idx.
        if self.settings.playback_rate == 0.0:
            return None
        if self._state.is_event or self._state.has_timestamps:
            output = self._process_time_window(clock_tick)
        else:
            output = self._process_rate_only(clock_tick)
        if output is not None and output.data.size:
            self._state.last = replace(output, segment=self._state.segment)
            return self._state.last
        if self._state.last is not None and self._at_end():
            # Once, on the first empty tick after the stream's last sample. A gap emits no marker: the
            # data after it is the next segment. Empty outputs after it carry its segment.
            output, self._state.last = end_of_stream(self._state.last), None
            self._state.segment = output.segment
            return output
        return None if output is None else replace(output, segment=self._state.segment)

    def _at_end(self) -> bool:
        """Whether every sample of the stream has been read."""
        state = self._state
        if not (state.is_event or state.has_timestamps):
            return state.sample_idx >= state.n_total_samples
        if state.is_event or self._get_fs() == 0.0:
            return state.file_t > state.slicer.get_stream_info(self.settings.stream_key).timestamps[-1]
        return state.ts_idx is not None and state.ts_idx >= state.n_total_samples

    def _process_rate_only(self, clock_tick: LinearAxis) -> AxisArray | None:
        """Rate-only continuous: replicate base class counter logic with auto-detected fs."""
        fs = self._get_fs()
        if fs == 0.0:
            return None

        # Compute n_samples. Fixed n_time ignores playback_rate (the chunk
        # size is the user's contract); variable chunk scales by playback_rate
        # so that wall-clock → file-time conversion honours the requested speed.
        if self.settings.n_time is not None:
            n_samples = self.settings.n_time
        else:
            if clock_tick.gain == 0.0:
                raise ValueError("Cannot use clock with gain=0 (AFAP) without specifying n_time")
            samples_float = fs * clock_tick.gain * self.settings.playback_rate + self._state.fractional_samples
            n_samples = int(samples_float + 1e-9)
            self._state.fractional_samples = samples_float - n_samples
            if n_samples == 0:
                return None

        # Check bounds
        start_idx = self._state.sample_idx
        if start_idx >= self._state.n_total_samples:
            return None
        stop_idx = min(start_idx + n_samples, self._state.n_total_samples)

        output = self._state.slicer.read_by_index(self.settings.stream_key, start_idx, stop_idx)

        self._state.sample_idx = stop_idx
        # Keep file_t in sync so a mid-stream switch to a time-window
        # strategy (via settings hot-update) picks up where we left off.
        if fs > 0:
            self._state.file_t = self._state.t0 + stop_idx / fs
        self._state.counter += stop_idx - start_idx

        return output

    def _process_time_window(self, clock_tick: LinearAxis) -> AxisArray | None:
        """Timestamped continuous or event streams: time-window extraction.

        The window is anchored on the internal ``file_t`` cursor (initialised
        from ``start_offset`` in :meth:`_reset_state`) rather than
        ``clock_tick.offset``, so playback position is decoupled from wall-
        clock alignment and respects ``playback_rate``.
        """
        slicer = self._state.slicer
        if slicer is None:
            return None

        if clock_tick.gain == 0.0:
            # AFAP mode: use n_time / fs as the window width.
            fs = self._get_fs()
            if fs == 0.0:
                return None
            if self.settings.n_time is not None:
                dt = self.settings.n_time / fs
            else:
                raise ValueError("Cannot use clock with gain=0 (AFAP) without specifying n_time")
            self._state.counter += self.settings.n_time
        else:
            dt = clock_tick.gain * self.settings.playback_rate

        key = self.settings.stream_key
        t_start = self._state.file_t
        t_end = t_start + dt
        self._state.file_t = t_end
        if self._state.is_event or self._get_fs() == 0.0:
            return slicer.read_by_time(key, t_start, t_end, gap_tol=float("inf"))

        # Continue from where the last tick stopped, which is short of its window's
        # end when it stopped at a gap.
        start = self._state.ts_idx
        if start is None:
            start = slicer.index_at(key, t_start)
        stop = slicer.stop_index(key, start, t_end)
        starts = slicer.segment_starts(key, start, stop)
        if starts.size and starts[0] == start:
            self._state.segment += 1
            starts = starts[1:]
        if starts.size:
            stop = int(starts[0])
        self._state.ts_idx = stop
        return slicer.read_by_index(key, start, stop)

    def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
        # Required by ABC but we override _process directly
        raise NotImplementedError("NWBClockDrivenProducer overrides _process directly")

    def __del__(self):
        if hasattr(self, "_state") and self._state.slicer is not None:
            self._state.slicer.close()


class NWBClockDrivenUnit(BaseClockDrivenUnit[NWBClockDrivenSettings, NWBClockDrivenProducer]):
    """Clock-driven NWB unit that reads one stream synchronized to a shared clock."""

    SETTINGS = NWBClockDrivenSettings
