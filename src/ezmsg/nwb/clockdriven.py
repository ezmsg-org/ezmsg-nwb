"""Clock-driven NWB producer for synchronizing NWB data to a shared clock."""

from __future__ import annotations

import os
import typing

import ezmsg.core as ez
from ezmsg.baseproc.clockdriven import BaseClockDrivenProducer, ClockDrivenSettings, ClockDrivenState
from ezmsg.baseproc.protocols import processor_state
from ezmsg.baseproc.units import BaseClockDrivenUnit
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis

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
        return self._state.sample_idx >= self._state.n_total_samples

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

        slicer = NWBSlicer(
            filepath=self.settings.filepath,
            reference_clock=self.settings.reference_clock,
            reref_now=self.settings.reref_now,
            stream_keys=[self.settings.stream_key],
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
            return self._process_time_window(clock_tick)
        else:
            return self._process_rate_only(clock_tick)

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

        t_start = self._state.file_t
        t_end = t_start + dt
        output = slicer.read_by_time(self.settings.stream_key, t_start, t_end)
        self._state.file_t = t_end
        return output

    def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
        # Required by ABC but we override _process directly
        raise NotImplementedError("NWBClockDrivenProducer overrides _process directly")

    def __del__(self):
        if hasattr(self, "_state") and self._state.slicer is not None:
            self._state.slicer.close()


class NWBClockDrivenUnit(BaseClockDrivenUnit[NWBClockDrivenSettings, NWBClockDrivenProducer]):
    """Clock-driven NWB unit that reads one stream synchronized to a shared clock."""

    SETTINGS = NWBClockDrivenSettings
