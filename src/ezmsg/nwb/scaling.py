"""Apply the scaling an NWB file stores alongside its samples.

An NWB ``TimeSeries`` almost never stores the values it means. It stores integer
samples plus the factors that turn them into the unit it declares::

    value = data * conversion * channel_conversion[i] + offset

``conversion`` is a scalar, ``channel_conversion`` an optional per-channel
vector (``ElectricalSeries`` only), ``offset`` an additive constant. Reading
``data`` and ignoring the rest hands the caller raw ADC counts while every
label on the file says otherwise -- and nothing raises, because counts and
volts are both just numbers. Any downstream stage with an absolute threshold
(a rail guard, a power clip) is then comparing against the wrong scale.

So this module resolves those three factors into a single gain and offset
(:func:`resolve_scaling`) and applies them lazily on read
(:class:`ScaledDataset`), leaving the file's chunked I/O and the slicer's
laziness intact.

**The declared unit cannot be trusted, and neither can the conversion.** All
four of these combinations have been seen on files describing the same
hardware at the same true scale of 0.25 uV/count:

==============  ==============  ================================  ==========
``conversion``  ``unit``        ``channel_conversion``            True scale
==============  ==============  ================================  ==========
0.25            ``microvolts``  absent                            0.25 uV
1.0             ``volts``       5e-08                             0.25 uV
0.25            ``volts``       absent                            0.25 uV
1.0             ``uV``          absent                            0.25 uV
==============  ==============  ================================  ==========

Rows 1 and 3 agree about the number and disagree about the unit, so trusting
the label is a factor of 1e6 either way. Row 2 is the dangerous one: the whole
scale sits in a ``channel_conversion`` that is a writing library's default for
some *other* amplifier, 5x off for the hardware that actually recorded it --
and no amount of care in a *reader* can tell a wrong gain from a real one.

That is why the scaling resolved here is overridable per stream
(``scale_override`` / ``unit_override``): honouring the file is the right
default, but a caller who knows a particular writer lies needs to say so
without patching the file.
"""

from __future__ import annotations

import typing

import h5py
import numpy as np

from .util import as_text

MICROVOLT_UNITS = frozenset({"microvolt", "microvolts", "uv", "µv", "μv"})
"""Spellings of "microvolts" seen in the wild, lowercased.

Both micro signs are here on purpose: U+00B5 MICRO SIGN and U+03BC GREEK SMALL
LETTER MU render identically and writers use both.
"""

DEFAULT_CONVERSION_DTYPE = "float32"
"""Output dtype once a gain has been applied.

float32, not float64: broadband is ~80 GB per recording-hour, and int16 ->
float64 quadruples that for no benefit. float32 carries 24 bits of mantissa
against int16's 16 bits of input, so applying a gain is exact for any gain that
is a power of two (0.25 is) and otherwise correctly rounded -- there is no
precision to recover by going wider.
"""


class ScaledDataset:
    """Lazy ``data * gain + offset`` view over an h5py dataset.

    Quacks like the dataset it wraps (``shape``, ``dtype``, ``ndim``, ``len``,
    ``__getitem__``) so every read path -- the iterator's chunk builder and both
    of :class:`~ezmsg.nwb.slicer.NWBSlicer`'s slice methods -- picks up the
    scaling without knowing it exists. Nothing is materialized: the wrapped
    dataset is still sliced lazily and only the requested block is converted.

    ``base`` stays reachable because the dejitter pass identifies a stream's
    ``*_device_ts`` partner by comparing HDF5 object addresses, which only exist
    on the real dataset (see :meth:`~ezmsg.nwb.slicer.NWBSlicer._data_addr`).
    """

    __slots__ = ("base", "shape", "dtype", "_gain", "_offset", "_gain_shape")

    def __init__(
        self,
        base: typing.Any,
        gain: typing.Union[float, np.ndarray],
        offset: float = 0.0,
        dtype: typing.Any = DEFAULT_CONVERSION_DTYPE,
    ) -> None:
        self.base = base
        self.dtype = np.dtype(dtype)
        self.shape = tuple(base.shape)
        gain_arr = np.asarray(gain, dtype=self.dtype)
        if gain_arr.ndim > 1:
            raise ValueError(f"gain must be scalar or 1-D, got shape {gain_arr.shape}.")
        if gain_arr.ndim == 1:
            # Per-channel gains live on axis 1 (the ``ch`` axis; axis 0 is time).
            if len(self.shape) < 2 or gain_arr.size != self.shape[1]:
                raise ValueError(
                    f"per-channel gain of length {gain_arr.size} does not match dataset shape {self.shape}."
                )
            self._gain_shape = (1, gain_arr.size) + (1,) * (len(self.shape) - 2)
        else:
            self._gain_shape = ()
        self._gain = gain_arr
        self._offset = self.dtype.type(offset)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: typing.Any) -> np.ndarray:
        raw = self.base[key]
        gain = self._gain
        if self._gain_shape:
            # An integer index on the time axis drops a leading dimension from
            # the result; trim the same number off the front of the broadcast
            # shape so the channel gains still land on the channel axis.
            dropped = self.ndim - np.ndim(raw)
            gain = gain.reshape(self._gain_shape[dropped:])
        # One allocation: multiply reads the int16 straight into a float32 out,
        # rather than casting the whole block first and scaling that copy.
        out = np.multiply(raw, gain, dtype=self.dtype)
        if self._offset:
            out += self._offset
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScaledDataset({self.base!r}, gain={self._gain!r}, offset={self._offset!r})"


def unwrap_dataset(dset: typing.Any) -> typing.Any:
    """The underlying dataset, whether or not it is wrapped for scaling."""
    return dset.base if isinstance(dset, ScaledDataset) else dset


def _override_for(override: typing.Any, key: str) -> typing.Any:
    """Resolve a bare-value-or-per-stream-mapping override for one stream.

    Both spellings are accepted because both are natural: a single value when
    every stream in the file shares the problem (one writer wrote them all), a
    ``{stream_key: value}`` mapping when only one does.
    """
    if override is None:
        return None
    if isinstance(override, dict):
        return override.get(key)
    return override


def read_stored_scaling(dset: typing.Any) -> tuple[float, float, str]:
    """``(conversion, offset, unit)`` as literally stored on a data dataset.

    Read off the h5py attributes rather than through ``pynwb``, because pynwb
    does not report what the file says: ``unit`` is a *fixed* field on
    ``ElectricalSeries`` in the NWB schema, so pynwb returns the schema's
    ``"volts"`` for every electrical series regardless of the string on disk.
    On a recording whose ``data`` attribute reads ``microvolts``, that is the
    difference between a scale factor of 0.25 and one of 2.5e-07.

    Missing attributes fall back to the NWB defaults (1.0, 0.0, "").
    """
    attrs = getattr(dset, "attrs", None)
    if attrs is None:
        return 1.0, 0.0, ""
    return (
        float(attrs.get("conversion", 1.0)),
        float(attrs.get("offset", 0.0)),
        as_text(attrs.get("unit", "")),
    )


def resolve_scaling(
    dset: typing.Any,
    channel_conversion: typing.Any = None,
    *,
    scale_override: typing.Any = None,
    unit_override: typing.Any = None,
    stream_key: str = "",
) -> tuple[typing.Union[float, np.ndarray], float, str]:
    """Collapse a stream's stored scaling into one ``(gain, offset, unit)``.

    ``gain`` is ``conversion * channel_conversion`` -- a scalar when the
    per-channel factors are absent or all equal (the common case, and the one
    that keeps the multiply a scalar broadcast), a vector only when the channels
    genuinely disagree.

    ``scale_override`` replaces the resolved gain outright, for a file whose
    recorded factors are known to be wrong; ``unit_override`` replaces only the
    declared unit, for one whose number is right and whose label is not. Either
    may be a bare value (applies to every stream) or a ``{stream_key: value}``
    mapping.
    """
    conversion, offset, unit = read_stored_scaling(dset)

    gain: typing.Union[float, np.ndarray] = conversion
    if channel_conversion is not None:
        per_channel = np.asarray(channel_conversion[:], dtype=np.float64)
        # A vector gain costs a broadcast on every read and forces the gain to
        # track the channel axis through any reshape. Only pay that when the
        # channels actually differ; an all-equal vector is a scalar.
        if per_channel.size:
            if np.all(per_channel == per_channel[0]):
                gain = conversion * float(per_channel[0])
            else:
                gain = conversion * per_channel

    scale = _override_for(scale_override, stream_key)
    if scale is not None:
        gain = scale
    forced_unit = _override_for(unit_override, stream_key)
    if forced_unit is not None:
        unit = str(forced_unit)

    return gain, offset, unit


def is_identity_scaling(gain: typing.Union[float, np.ndarray], offset: float) -> bool:
    """Whether applying this scaling would leave every sample unchanged.

    Worth checking: a stream that declares no scaling is already in the unit it
    claims, so wrapping it would only trade its stored dtype for a float32 copy
    of the same numbers -- a needless doubling on an int16 broadband stream, and
    a surprise for a caller reading an integer marker channel.
    """
    return bool(np.all(np.asarray(gain) == 1.0)) and offset == 0.0


def maybe_scale(
    dset: typing.Any,
    channel_conversion: typing.Any = None,
    *,
    dtype: typing.Any = DEFAULT_CONVERSION_DTYPE,
    scale_override: typing.Any = None,
    unit_override: typing.Any = None,
    stream_key: str = "",
) -> tuple[typing.Any, str]:
    """``(dataset, declared_unit)`` with the stored scaling applied on read.

    Returns *dset* unwrapped and unchanged when the resolved scaling is the
    identity, or when *dset* is not an h5py dataset (a materialized text column
    has no gain to apply). Otherwise returns a :class:`ScaledDataset`.
    """
    if not isinstance(dset, h5py.Dataset) or dset.dtype.kind not in "iuf":
        return dset, ""
    gain, offset, unit = resolve_scaling(
        dset,
        channel_conversion,
        scale_override=scale_override,
        unit_override=unit_override,
        stream_key=stream_key,
    )
    if is_identity_scaling(gain, offset):
        return dset, unit
    return ScaledDataset(dset, gain, offset, dtype), unit
