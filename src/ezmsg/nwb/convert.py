"""Apply an NWB stream's scaling downstream, instead of at the reader.

:class:`~ezmsg.nwb.slicer.NWBSlicer` applies a file's conversion on read by
default. Passing ``apply_conversion=False`` defers it, and this is what picks it
up later in the graph: every message carries the factors under
``attrs[SCALING_ATTR]`` (see :mod:`~ezmsg.nwb.scaling`), so a stage far from the
file can still do exactly what the reader would have.

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
"""

from __future__ import annotations

import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .scaling import (
    DEFAULT_CONVERSION_DTYPE,
    SCALING_ATTR,
    StreamScaling,
    VoltageUnit,
    coerce_voltage_unit,
    convert_to_target_unit,
    is_identity_scaling,
    override_for,
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
    """Deliver voltage streams in this unit. Same form. Unlike the other two
    this also applies to messages the reader already scaled, since a prefix
    change on top of a known unit is well defined."""


class NWBScalingTransformer(BaseTransformer[NWBScalingSettings, AxisArray, AxisArray]):
    """Apply the scaling a message reports, and update what it reports.

    Stateless and idempotent. Idempotent because it acts on the message's
    ``applied`` flag rather than on its dtype: two of these in a chain, or one
    behind a reader that already converted, cannot double-scale. That property
    is the reason the flag exists -- without it, "has this been scaled?" would
    have to be guessed from whether the data looks like integers, which is
    exactly the kind of inference that makes wrong units silent.

    A message with no ``attrs[SCALING_ATTR]`` passes through untouched: it did
    not come from an NWB reader, and inventing a scaling for it would be worse
    than doing nothing.
    """

    def _process(self, message: AxisArray) -> AxisArray:
        payload = message.attrs.get(SCALING_ATTR)
        if payload is None:
            return message

        scaling = StreamScaling.from_attr(payload)
        requested = override_for(self.settings.target_unit, message.key)
        target = coerce_voltage_unit(requested) if requested is not None and scaling.voltage else None

        if scaling.applied:
            return self._retarget(message, scaling, target)
        return self._apply(message, scaling, target)

    def _retarget(self, message: AxisArray, scaling: StreamScaling, target: typing.Optional[VoltageUnit]) -> AxisArray:
        """Convert data that already carries its gain and offset.

        Only a unit change is available here, and it is one multiply: for
        ``value = stored * gain + offset``, scaling the values by the unit ratio
        carries the offset with them, so there is nothing to unwind.
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
                    f"{name} cannot be applied to {message.key!r}: its scaling was already applied upstream. "
                    f"Set {name} on the reader (NWBSlicer / NWBIteratorSettings), or read with "
                    f"apply_conversion=False and correct it here."
                )
        if target is None:
            return message
        ratio = unit_ratio(scaling.unit, target, stream_key=message.key)
        if ratio == 1.0:
            return message
        return self._emit(
            message,
            _scale(message.data, ratio, 0.0, self.settings.conversion_dtype, message.key),
            StreamScaling(scaling.gain * ratio, scaling.offset * ratio, target.value, True, scaling.voltage),
        )

    def _apply(self, message: AxisArray, scaling: StreamScaling, target: typing.Optional[VoltageUnit]) -> AxisArray:
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

        resolved = StreamScaling(gain, offset, unit, True, scaling.voltage)
        if is_identity_scaling(gain, offset):
            # The stored samples are already in `unit` -- an integer marker
            # channel, say. Record that, but do not copy it into float to do so.
            return self._emit(message, message.data, resolved)
        return self._emit(
            message, _scale(message.data, gain, offset, self.settings.conversion_dtype, message.key), resolved
        )

    @staticmethod
    def _emit(message: AxisArray, data: np.ndarray, scaling: StreamScaling) -> AxisArray:
        return replace(
            message,
            data=data,
            attrs={
                **message.attrs,
                "unit": scaling.unit,
                # Cumulative, stored -> now, so the message keeps describing its
                # relationship to the file rather than to its previous stage.
                SCALING_ATTR: scaling.as_attr(),
            },
        )


def _scale(
    data: np.ndarray,
    gain: typing.Union[float, np.ndarray],
    offset: float,
    dtype: typing.Any,
    stream_key: str,
) -> np.ndarray:
    """``data * gain + offset`` in *dtype*, with the channel axis respected."""
    out_dtype = np.dtype(dtype)
    gain_arr = np.asarray(gain, dtype=out_dtype)
    if gain_arr.ndim == 1:
        # A per-channel gain is positional, so any upstream stage that dropped
        # or reordered channels has silently invalidated it. Length is the only
        # check available, and it catches the common case (a channel subset).
        if data.ndim < 2 or gain_arr.size != data.shape[1]:
            raise ValueError(
                f"per-channel gain of length {gain_arr.size} does not fit {stream_key!r} with shape {data.shape}. "
                f"A stage upstream changed the channel axis after the scaling was recorded; apply the conversion "
                f"before that stage, or at the reader."
            )
        gain_arr = gain_arr.reshape((1, gain_arr.size) + (1,) * (data.ndim - 2))
    out = np.multiply(data, gain_arr, dtype=out_dtype)
    if offset:
        out += out_dtype.type(offset)
    return out


class NWBScalingUnit(BaseTransformerUnit[NWBScalingSettings, AxisArray, AxisArray, NWBScalingTransformer]):
    SETTINGS = NWBScalingSettings
