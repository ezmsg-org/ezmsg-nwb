"""Apply an NWB stream's scaling downstream, instead of at the reader.

:class:`~ezmsg.nwb.slicer.NWBSlicer` never applies a file's conversion -- it
hands back the stored integers and describes them, carrying the factors under
``nwb_scaling_*`` attrs (see :mod:`~ezmsg.nwb.scaling`). This is what picks them
up later in the graph, so a stage far from the file can do exactly what the
reader would have.

Why defer at all? Reading raw is the only way to keep a broadband stream int16
through the parts of a graph that do not care about units. Scaling at the reader
doubles every message to float32 immediately -- ~80 GB per recording-hour going
to ~160 -- and a graph that decimates, selects a channel subset, or windows
before it needs volts pays that on samples it is about to discard. Putting this
transformer after those stages converts what survives.

The other use is a graph whose source is not always a file: the reader emits
what the file declares, and one of these pinned to ``target_unit`` at the head
of the processing chain makes every branch downstream agree on a scale
regardless.

**Everything per-stream is resolved once.** A stream's scaling does not change
from message to message, so resolving overrides, parsing units, and building the
output ``attrs`` on every message is pure waste -- and at simulated-online sizes
(30 samples) that waste was 60% of the call. :class:`NWBScalingTransformer`
caches a :class:`_Plan` per stream key and does nothing per message but one
multiply.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .scaling import (
    DEFAULT_CONVERSION_DTYPE,
    SCALING_ATTRS,
    StreamScaling,
    VoltageUnit,
    coerce_voltage_unit,
    convert_to_target_unit,
    is_identity_scaling,
    is_voltage_stream,
    override_for,
    scaling_fingerprint,
    unit_ratio,
)


class NWBScalingSettings(ez.Settings):
    """Settings for :class:`NWBScalingTransformer`.

    Deliberately the same names and forms as the corresponding
    ``NWBIteratorSettings`` fields, so moving the work down the graph is a
    matter of moving the settings, not rewriting them.
    """

    conversion_dtype: str = DEFAULT_CONVERSION_DTYPE
    """Output dtype for a message that gets scaled (default ``float32``)."""
    scale_override: typing.Union[float, dict[str, float], None] = None
    """Replace the gain the message reports, for a file whose recorded factors
    are wrong. Bare value or ``{stream_key: value}``, matched against
    ``msg.key``."""
    unit_override: typing.Union[str, dict[str, str], None] = None
    """Replace the reported unit without changing the gain. Same form."""
    target_unit: typing.Union[str, VoltageUnit, dict[str, typing.Union[str, VoltageUnit]], None] = None
    """Deliver voltage streams in this unit. Same form. Unlike the other two this
    also applies to a message that was already scaled and carries only its
    ``unit``, since a prefix change on top of a known unit is well defined."""


def _scaling_from_unit(attrs: typing.Mapping[str, typing.Any]) -> typing.Optional[StreamScaling]:
    """Treat a message that declares only a ``unit`` as already-scaled.

    Once this transformer has run it strips the ``nwb_scaling_*`` keys and leaves
    ``unit`` behind, so a second one downstream has nothing else to go on. That is
    enough for the only operation still meaningful at that point: ``target_unit``
    is a prefix change on top of a known unit, which needs the unit and nothing
    else. The gain and offset it would have read are the *file's*, describing
    stored counts that no longer exist -- which is exactly why they are dropped.

    Returns None for a unit that is not a voltage, so a ``pixels`` stream is left
    alone rather than reinterpreted.
    """
    unit = attrs.get("unit")
    if not unit or not is_voltage_stream(str(unit), is_electrical=False):
        return None
    # gain/offset are the identity: the values are already in ``unit``, and a
    # retarget scales them from there. Nothing here claims to describe counts.
    return StreamScaling(gain=1.0, offset=0.0, unit=str(unit), applied=True, voltage=True)


@dataclass
class _Plan:
    """Everything a stream needs, resolved once and reused per message.

    ``source`` is the :func:`~ezmsg.nwb.scaling.scaling_fingerprint` of the attrs
    this was built from -- a by-value summary, so a stage that rebuilds attrs
    without changing the scaling still hits, and a stream whose scaling genuinely
    changes mid-run rebuilds. None when the message reported no scaling.

    ``attrs_ref`` is the attrs dict itself, kept only to pin it alive. The cheap
    per-message check in :meth:`NWBScalingTransformer._hash_message` is keyed on
    its ``id()``, and an object we hold cannot be freed and have its address
    reused by a different dict -- which is the trap in caching on ``id()``.
    """

    source: typing.Any
    attrs_ref: typing.Any
    out_attrs: dict[str, typing.Any]
    """The complete attrs for output messages, prebuilt and shared. Shared is
    safe for the same reason the slicer shares its template's: nothing mutates
    an attrs dict in flight."""
    passthrough: bool = False
    """Nothing to do -- either no scaling reported, or already in the unit
    asked for. The message is returned as-is, not copied."""
    identity: bool = False
    """The scaling is a no-op arithmetically but the message's labels still need
    updating: say so in attrs without copying int16 into float to do it."""
    gain: typing.Any = None
    """Pre-cast to the output dtype and pre-reshaped for broadcasting."""
    offset: typing.Any = None
    """None when zero, so the common case skips a whole pass over the data."""
    dtype: typing.Any = None
    n_ch: int = -1
    """Channel count a vector gain was built for; -1 for a scalar gain."""
    ndim: int = -1


@processor_state
class NWBScalingState:
    # ``processor_state`` builds the dataclass with ``init=False``, so a
    # ``default_factory`` would never run; the repo's convention is a plain
    # ``None`` default assigned on first reset.
    plans: dict[str, _Plan] | None = None
    settings_ref: typing.Any = None


class NWBScalingTransformer(BaseStatefulTransformer[NWBScalingSettings, AxisArray, AxisArray, NWBScalingState]):
    """Apply the scaling a message reports, and update what it reports.

    Idempotent, and structurally so: applying a scaling *consumes* it. The
    output carries no ``nwb_scaling_*`` keys, so a second of these in a chain
    finds nothing pending and has nothing to double-apply. Never inferred from
    the dtype -- "does this look like integers?" is exactly the kind of guess
    that makes wrong units silent.

    What it emits is ``unit`` and nothing else from this module. ``gain`` and
    ``offset`` described stored counts, and the multiply consumed them; a filter
    or a decimation downstream makes them less true still, so carrying them
    would hand a later stage a factor that looks authoritative and reconstructs
    nothing. Re-digitizing is a genuine need, but it happens after
    transformations those numbers never saw and so takes its range and
    resolution from the caller.

    A message with no ``nwb_scaling_*`` attrs and no ``unit`` passes through
    untouched: it did not come from an NWB reader, and inventing a scaling for
    it would be worse than doing nothing. One that kept its ``unit`` is still
    retargetable -- see :func:`_scaling_from_unit`.

    State is a plan per stream key, not a single current plan, because one
    reader publishes every stream on one output: with a single plan, two
    interleaved streams would evict each other on every message and the cache
    would cost more than it saved.
    """

    def _hash_message(self, message: AxisArray) -> int:
        """Cheap and allowed to be wrong -- this runs on *every* message.

        Correctness lives in :meth:`_reset_state`, which re-validates by value
        against the fingerprint it holds; this only decides whether that check is
        worth running. So it keys on the identity of the attrs dict, which the
        reader builds once per stream and every message of that stream shares.

        Reading the five ``nwb_scaling_*`` values here instead would be right but
        wasteful: it costs ~160 ns against a ~2 us call, and :meth:`_reset_state`
        pays it again on the way through. ``id()`` is safe because the plan holds
        the dict (see :class:`_Plan`), so no other object can take its address.
        A stage that rebuilds attrs per message merely lands in ``_reset_state``,
        whose by-value check then hits without rebuilding the plan.
        """
        return hash((message.key, id(message.attrs)))

    def _reset_state(self, message: AxisArray) -> None:
        plans = self._state.plans
        if plans is None or self._state.settings_ref is not self.settings:
            # First message, or settings replaced at runtime -- every existing
            # plan was resolved against the old ones.
            plans = self._state.plans = {}
            self._state.settings_ref = self.settings
        fingerprint = scaling_fingerprint(message.attrs)
        plan = plans.get(message.key)
        if plan is not None and plan.source == fingerprint:
            # Same scaling, different attrs dict -- an upstream stage rebuilt it.
            # Re-pin to the dict ``_hash_message`` is now keyed on, or that id()
            # would name an object nothing holds, free to be released and have
            # its address taken by attrs carrying a *different* scaling.
            plan.attrs_ref = message.attrs
            return
        plans[message.key] = self._build_plan(message, fingerprint)

    def _build_plan(self, message: AxisArray, fingerprint: typing.Optional[tuple]) -> _Plan:
        scaling = StreamScaling.from_attrs(message.attrs) or _scaling_from_unit(message.attrs)
        if scaling is None:
            return _Plan(source=None, attrs_ref=message.attrs, out_attrs=message.attrs, passthrough=True)
        requested = override_for(self.settings.target_unit, message.key)
        target = coerce_voltage_unit(requested) if requested is not None and scaling.voltage else None

        if scaling.applied:
            resolved = self._plan_retarget(message, scaling, target)
        else:
            resolved = self._plan_apply(message, scaling, target)
        if resolved is None:
            return _Plan(source=fingerprint, attrs_ref=message.attrs, out_attrs=message.attrs, passthrough=True)

        gain, offset, final = resolved
        # The scaling is spent: drop it and keep only what the data now is.
        #
        # ``gain`` and ``offset`` describe stored counts -> values. Once the
        # multiply has happened those counts are gone, and every later stage --
        # a filter, a decimation, a re-reference -- makes them less true still.
        # Carrying them would offer a downstream consumer a factor that looks
        # authoritative and reconstructs nothing. Re-digitizing is a real need,
        # but it happens after transformations these numbers never saw, so it
        # takes its range and resolution from the caller instead.
        #
        # ``unit`` is the exception, and the whole point: it describes the data
        # in hand rather than its history, and it is what a downstream
        # ``target_unit`` needs (see :func:`_scaling_from_unit`).
        out_attrs = {k: v for k, v in message.attrs.items() if k not in SCALING_ATTRS}
        out_attrs["unit"] = final.unit
        if is_identity_scaling(gain, offset):
            # Arithmetically a no-op: record the unit, keep the stored dtype.
            return _Plan(source=fingerprint, attrs_ref=message.attrs, out_attrs=out_attrs, identity=True)

        out_dtype = np.dtype(self.settings.conversion_dtype)
        gain_arr = np.asarray(gain, dtype=out_dtype)
        n_ch = -1
        if gain_arr.ndim == 1:
            n_ch = gain_arr.size
            _check_channels(gain_arr.size, message.data.shape, message.key)
            gain_arr = gain_arr.reshape((1, gain_arr.size) + (1,) * (message.data.ndim - 2))
        return _Plan(
            source=fingerprint,
            attrs_ref=message.attrs,
            out_attrs=out_attrs,
            gain=gain_arr,
            offset=out_dtype.type(offset) if offset else None,
            dtype=out_dtype,
            n_ch=n_ch,
            ndim=message.data.ndim,
        )

    def _plan_retarget(
        self, message: AxisArray, scaling: StreamScaling, target: typing.Optional[VoltageUnit]
    ) -> typing.Optional[tuple[typing.Any, float, StreamScaling]]:
        """Data already carries its gain and offset; only a unit change is left.

        And that is one multiply: for ``value = stored * gain + offset``, scaling
        the values by the unit ratio carries the offset along with them, so
        there is nothing to unwind.
        """
        for name, value in (
            ("scale_override", self.settings.scale_override),
            ("unit_override", self.settings.unit_override),
        ):
            if override_for(value, message.key) is not None:
                # Correcting a gain that was already applied would mean dividing
                # it back out -- an extra pass over the data and float error
                # accumulated to fix something the reader was better placed to
                # fix. Say so rather than half-doing it.
                raise ValueError(
                    f"{name} cannot be applied to {message.key!r}: its scaling was already applied upstream, "
                    f"and the factors it was applied with are gone. Move this transformer (or set {name} on "
                    f"it) ahead of whatever already converted the stream."
                )
        if target is None:
            return None
        ratio = unit_ratio(scaling.unit, target, stream_key=message.key)
        if ratio == 1.0:
            return None
        return (
            ratio,
            0.0,
            StreamScaling(scaling.gain * ratio, scaling.offset * ratio, target.value, True, scaling.voltage),
        )

    def _plan_apply(
        self, message: AxisArray, scaling: StreamScaling, target: typing.Optional[VoltageUnit]
    ) -> tuple[typing.Any, float, StreamScaling]:
        """Apply a pending scaling to stored samples, as the reader would have."""
        gain, offset, unit = scaling.gain, scaling.offset, scaling.unit
        override = override_for(self.settings.scale_override, message.key)
        if override is not None:
            gain = override
        forced = override_for(self.settings.unit_override, message.key)
        if forced is not None:
            unit = str(forced)
        if target is not None:
            gain, offset, unit = convert_to_target_unit(gain, offset, unit, target, stream_key=message.key)
        return gain, offset, StreamScaling(gain, offset, unit, True, scaling.voltage)

    def _process(self, message: AxisArray) -> AxisArray:
        plan = self._state.plans[message.key]
        if plan.passthrough:
            return message
        if plan.identity:
            return replace(message, attrs=plan.out_attrs)
        data = message.data
        if plan.n_ch >= 0 and (data.ndim != plan.ndim or data.shape[1] != plan.n_ch):
            # Re-checked per message, not just per plan: the plan is keyed on the
            # reported scaling, which an upstream channel selection does not
            # change even though it invalidates a positional gain.
            _check_channels(plan.n_ch, data.shape, message.key)
        out = np.multiply(data, plan.gain, dtype=plan.dtype)
        if plan.offset is not None:
            out += plan.offset
        return replace(message, data=out, attrs=plan.out_attrs)


def _check_channels(n_gain: int, shape: tuple[int, ...], stream_key: str) -> None:
    """A per-channel gain is positional, so a stage that dropped or reordered
    channels upstream has silently invalidated it. Length is the only check
    available, and it catches the common case (a channel subset)."""
    if len(shape) < 2 or n_gain != shape[1]:
        raise ValueError(
            f"per-channel gain of length {n_gain} does not fit {stream_key!r} with shape {shape}. "
            f"A stage upstream changed the channel axis after the scaling was recorded; apply the conversion "
            f"before that stage, or at the reader."
        )


class NWBScalingUnit(BaseTransformerUnit[NWBScalingSettings, AxisArray, AxisArray, NWBScalingTransformer]):
    SETTINGS = NWBScalingSettings
