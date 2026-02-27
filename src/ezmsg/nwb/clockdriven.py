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


class NWBClockDrivenProducer(BaseClockDrivenProducer[NWBClockDrivenSettings, NWBClockDrivenState]):
    """Produces NWB data synchronized to clock ticks.

    Three extraction strategies based on stream type:
      1. Rate-only continuous: sample-index-based using fs and counter.
      2. Timestamped continuous: time-window-based using searchsorted.
      3. Events/intervals: time-window-based, returning all events in window.
    """

    @property
    def exhausted(self) -> bool:
        if self._state.slicer is None:
            return False  # Not yet initialized — not exhausted
        if self._state.is_event or self._state.has_timestamps:
            return False  # Time-window streams don't track exhaustion via index
        return self._state.sample_idx >= self._state.n_total_samples

    def _reset_state(self, time_axis: LinearAxis) -> None:
        if self._state.slicer is not None:
            self._state.slicer.close()

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
        self._state.sample_idx = 0
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

    def _get_fs(self) -> float:
        """Return the effective sample rate (user-specified or auto-detected)."""
        return self._state.detected_fs

    def _process(self, clock_tick: LinearAxis) -> AxisArray | None:
        if self._state.is_event or self._state.has_timestamps:
            return self._process_time_window(clock_tick)
        else:
            return self._process_rate_only(clock_tick)

    def _process_rate_only(self, clock_tick: LinearAxis) -> AxisArray | None:
        """Rate-only continuous: replicate base class counter logic with auto-detected fs."""
        fs = self._get_fs()
        if fs == 0.0:
            return None

        # Compute n_samples (same logic as base class but using detected fs)
        if self.settings.n_time is not None:
            n_samples = self.settings.n_time
        else:
            if clock_tick.gain == 0.0:
                raise ValueError("Cannot use clock with gain=0 (AFAP) without specifying n_time")
            samples_float = fs * clock_tick.gain + self._state.fractional_samples
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
        self._state.counter += stop_idx - start_idx

        return output

    def _process_time_window(self, clock_tick: LinearAxis) -> AxisArray | None:
        """Timestamped continuous or event streams: time-window extraction."""
        slicer = self._state.slicer
        if slicer is None:
            return None

        if clock_tick.gain == 0.0:
            # AFAP mode: use counter-based synthetic time
            fs = self._get_fs()
            if fs == 0.0:
                return None
            if self.settings.n_time is not None:
                dt = self.settings.n_time / fs
            else:
                raise ValueError("Cannot use clock with gain=0 (AFAP) without specifying n_time")
            t_start = self._state.t0 + self._state.counter * (1.0 / fs)
            t_end = t_start + dt
            self._state.counter += self.settings.n_time
        else:
            # Time-window from clock tick (file-relative, subtract ts_off)
            t_start = clock_tick.offset - slicer.ts_off
            t_end = t_start + clock_tick.gain

        output = slicer.read_by_time(self.settings.stream_key, t_start, t_end)
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
