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

Separately from *what the file means*, a pipeline has an opinion about *what it
wants on the wire*: ``target_unit`` converts a voltage stream to a requested
:class:`VoltageUnit` on top of whatever the file (as corrected) turned out to
be, so a graph can be written against microvolts and fed by files that disagree
about their own scale. It applies only to electrical streams -- see
:func:`convert_to_target_unit`.
"""

from __future__ import annotations

import enum
import typing

import h5py
import numpy as np

from .util import as_text


class VoltageUnit(str, enum.Enum):
    """A unit a voltage stream can be delivered in.

    Voltage only, and deliberately so: an ``ElectricalSeries`` is voltage by
    definition (the NWB schema fixes its ``unit`` to ``volts``), so a prefix
    change is the only conversion that is pure arithmetic on the samples.
    Anything crossing a dimension -- volts to amperes, say -- needs a model of
    the circuit, which belongs in a processing stage that can be told about it,
    not in a file reader guessing from a label.

    Subclasses ``str`` so settings can be given as plain strings and survive a
    YAML round trip: ``VoltageUnit("microvolts") == "microvolts"``.
    """

    VOLTS = "volts"
    MILLIVOLTS = "millivolts"
    MICROVOLTS = "microvolts"
    NANOVOLTS = "nanovolts"


VOLTS_PER_UNIT: dict[VoltageUnit, float] = {
    VoltageUnit.VOLTS: 1.0,
    VoltageUnit.MILLIVOLTS: 1e-3,
    VoltageUnit.MICROVOLTS: 1e-6,
    VoltageUnit.NANOVOLTS: 1e-9,
}
"""How many volts one of each unit is. Ratios of these are the conversions."""

VOLTAGE_UNIT_SPELLINGS: dict[str, VoltageUnit] = {
    "v": VoltageUnit.VOLTS,
    "volt": VoltageUnit.VOLTS,
    "volts": VoltageUnit.VOLTS,
    "mv": VoltageUnit.MILLIVOLTS,
    "millivolt": VoltageUnit.MILLIVOLTS,
    "millivolts": VoltageUnit.MILLIVOLTS,
    "uv": VoltageUnit.MICROVOLTS,
    "µv": VoltageUnit.MICROVOLTS,
    "μv": VoltageUnit.MICROVOLTS,
    "microvolt": VoltageUnit.MICROVOLTS,
    "microvolts": VoltageUnit.MICROVOLTS,
    "nv": VoltageUnit.NANOVOLTS,
    "nanovolt": VoltageUnit.NANOVOLTS,
    "nanovolts": VoltageUnit.NANOVOLTS,
}
"""Unit strings seen in the wild, lowercased, mapped to what they mean.

Both micro signs are here on purpose: U+00B5 MICRO SIGN and U+03BC GREEK SMALL
LETTER MU render identically and writers use both.
"""

MICROVOLT_UNITS = frozenset(
    spelling for spelling, unit in VOLTAGE_UNIT_SPELLINGS.items() if unit is VoltageUnit.MICROVOLTS
)
"""Spellings of "microvolts", lowercased."""


def parse_voltage_unit(unit: str) -> typing.Optional[VoltageUnit]:
    """The :class:`VoltageUnit` a declared unit string names, or None.

    None for a string this module does not recognize *and* for the empty string,
    which are different problems for the caller: see
    :func:`convert_to_target_unit`.
    """
    return VOLTAGE_UNIT_SPELLINGS.get(unit.strip().lower())


def coerce_voltage_unit(value: typing.Any) -> VoltageUnit:
    """A :class:`VoltageUnit` from an enum member or any spelling of one.

    So ``target_unit="uV"`` works as well as ``VoltageUnit.MICROVOLTS`` -- the
    caller writing the setting should not have to know which spelling this
    module canonicalized on.
    """
    if isinstance(value, VoltageUnit):
        return value
    parsed = parse_voltage_unit(str(value))
    if parsed is None:
        raise ValueError(
            f"{value!r} does not name a voltage unit. Expected one of "
            f"{[u.value for u in VoltageUnit]} (or an abbreviation: V, mV, uV, nV)."
        )
    return parsed


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


def override_for(override: typing.Any, key: str) -> typing.Any:
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


def is_voltage_stream(unit: str, is_electrical: bool) -> bool:
    """Whether ``target_unit`` should convert a stream declaring *unit*.

    Two independent pieces of evidence, either sufficient. *is_electrical* is a
    structural fact -- an ``ElectricalSeries`` is voltage by schema, whatever
    string it carries, including none. A unit that :func:`parse_voltage_unit`
    places is a *declared* fact, and it is the only evidence available for a
    plain ``TimeSeries`` holding voltage: writers store electrical data outside
    ``ElectricalSeries`` routinely -- a re-timestamped companion of an
    acquisition stream, pointing at the very same HDF5 dataset, is written as a
    bare ``TimeSeries``. Converting one and not the other would hand two scales
    for the same bytes to the same graph, silently, which is the failure
    ``target_unit`` exists to remove.

    Trusting the label here does not contradict distrusting it elsewhere. The
    unit string is unreliable about *magnitude* -- that is what this module's
    table of contradictions is about, and what ``scale_override`` /
    ``unit_override`` exist for -- but about *dimension* it is all there is, and
    a wrong dimension is not a failure mode writers exhibit: no one labels a
    cursor position in volts. Streams whose unit names something else
    (``pixels``, ``n/a``) do not parse, so they are left alone.
    """
    return is_electrical or parse_voltage_unit(unit) is not None


def convert_to_target_unit(
    gain: typing.Union[float, np.ndarray],
    offset: float,
    unit: str,
    target: VoltageUnit,
    *,
    stream_key: str = "",
) -> tuple[typing.Union[float, np.ndarray], float, str]:
    """Fold a unit change into an already-resolved ``(gain, offset, unit)``.

    The samples are ``data * gain + offset`` in *unit*; scaling both by the
    ratio between the units leaves them in *target*. Offset included -- it is an
    additive constant in the same unit as the values, so a conversion that
    scaled only the gain would leave a stream with a non-zero offset wrong by a
    power of ten, which is worse than not converting at all.

    Applied after the overrides, on purpose: ``scale_override`` and
    ``unit_override`` establish what the file *actually* holds, and this
    converts from that to what the caller asked for. So a file that lies about
    both can still be corrected once and then requested in any unit.

    Raises ``ValueError`` when *unit* is a string this module cannot place --
    the conversion needs a starting point, and passing the data through
    unconverted while relabelling it *target* would manufacture exactly the
    silently-wrong units the rest of this module exists to prevent. An empty
    *unit* is the one exception: the NWB schema fixes ``ElectricalSeries.unit``
    to ``volts``, so a writer that stamped nothing has told us volts by
    omission.
    """
    ratio = unit_ratio(unit, target, stream_key=stream_key)
    if ratio == 1.0:
        return gain, offset, target.value
    return gain * ratio, offset * ratio, target.value


def unit_ratio(unit: str, target: VoltageUnit, *, stream_key: str = "") -> float:
    """The factor taking a value in *unit* to the same value in *target*.

    Split out from :func:`convert_to_target_unit` because a consumer holding
    values that were *already* scaled needs the factor alone: for data that is
    ``stored * gain + offset``, multiplying the values by this ratio is the whole
    conversion, since it carries the offset along with them. Recomputing a gain
    and an offset for that case would mean unwinding what was applied.
    """
    declared = VoltageUnit.VOLTS if not unit.strip() else parse_voltage_unit(unit)
    if declared is None:
        where = f" on stream {stream_key!r}" if stream_key else ""
        raise ValueError(
            f"Cannot convert to {target.value}{where}: the unit is {unit!r}, "
            f"which is not a recognized voltage unit. Pass unit_override to say what it really is, "
            f"or leave target_unit unset to take the file's own scaling as-is."
        )
    return VOLTS_PER_UNIT[declared] / VOLTS_PER_UNIT[target]


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
    target_unit: typing.Any = None,
    is_electrical: bool = False,
    stream_key: str = "",
) -> tuple[typing.Union[float, np.ndarray], float, str]:
    """Collapse a stream's stored scaling into one ``(gain, offset, unit)``.

    ``gain`` is ``conversion * channel_conversion`` -- a scalar when the
    per-channel factors are absent or all equal (the common case, and the one
    that keeps the multiply a scalar broadcast), a vector only when the channels
    genuinely disagree.

    ``scale_override`` replaces the resolved gain outright, for a file whose
    recorded factors are known to be wrong; ``unit_override`` replaces only the
    declared unit, for one whose number is right and whose label is not.
    ``target_unit`` then converts the result into the unit the caller wants, for
    the voltage streams only -- see :func:`is_voltage_stream`, which *is_electrical*
    feeds. Any of the three may be a bare value (applies to every stream) or a
    ``{stream_key: value}`` mapping.
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

    scale = override_for(scale_override, stream_key)
    if scale is not None:
        gain = scale
    forced_unit = override_for(unit_override, stream_key)
    if forced_unit is not None:
        unit = str(forced_unit)

    target = override_for(target_unit, stream_key)
    if target is not None and is_voltage_stream(unit, is_electrical):
        gain, offset, unit = convert_to_target_unit(
            gain, offset, unit, coerce_voltage_unit(target), stream_key=stream_key
        )

    return gain, offset, unit


def is_identity_scaling(gain: typing.Union[float, np.ndarray], offset: float) -> bool:
    """Whether applying this scaling would leave every sample unchanged.

    Worth checking: a stream that declares no scaling is already in the unit it
    claims, so wrapping it would only trade its stored dtype for a float32 copy
    of the same numbers -- a needless doubling on an int16 broadband stream, and
    a surprise for a caller reading an integer marker channel.
    """
    return bool(np.all(np.asarray(gain) == 1.0)) and offset == 0.0


SCALING_ATTR = "nwb_scaling"
"""Key under which a message reports the scaling that relates it to the file.

Namespaced rather than flat ``attrs["conversion"]``/``attrs["offset"]``: attrs
is a shared dict that any stage downstream may add to, and three unqualified
generic words are a collision waiting to happen.
"""


class StreamScaling(typing.NamedTuple):
    """The total transformation from a stream's stored samples to its values.

    ``value = stored * gain + offset``, expressed in ``unit``. Total, not the
    file's raw attributes: any ``scale_override`` and ``target_unit`` are
    already folded in, so this describes the numbers a caller actually holds
    (or, when ``applied`` is False, the ones they would hold after applying it
    themselves).
    """

    gain: typing.Union[float, np.ndarray]
    """Scalar, or a per-channel vector aligned with the ``ch`` axis."""
    offset: float
    unit: str
    """Unit the values are in once *gain* and *offset* are applied."""
    applied: bool
    """False when the caller asked for the stored samples (``apply_conversion``
    off), in which case the data is raw and this is what to do about it."""
    voltage: bool = False
    """Whether ``target_unit`` may convert this stream -- :func:`is_voltage_stream`
    decided, at read time.

    Carried rather than re-derived downstream because one of its two inputs does
    not survive the trip: an ``ElectricalSeries`` that stamped no unit is voltage
    by schema, but off the file all a consumer sees is an empty string. Deciding
    once, where both the type and the label are in hand, is what lets a later
    stage convert exactly the streams the reader would have.
    """

    def as_attr(self) -> dict[str, typing.Any]:
        """The ``attrs[SCALING_ATTR]`` payload -- a plain dict, so it survives
        the message codec and anything else that walks attrs generically."""
        return {
            "gain": self.gain,
            "offset": self.offset,
            "unit": self.unit,
            "applied": self.applied,
            "voltage": self.voltage,
        }

    @classmethod
    def from_attr(cls, payload: typing.Mapping[str, typing.Any]) -> StreamScaling:
        """Rebuild from an ``attrs[SCALING_ATTR]`` payload."""
        return cls(
            gain=payload["gain"],
            offset=float(payload["offset"]),
            unit=str(payload["unit"]),
            applied=bool(payload["applied"]),
            voltage=bool(payload.get("voltage", False)),
        )


class ScalingResult(typing.NamedTuple):
    """What :func:`resolve_stream_scaling` hands the slicer."""

    dset: typing.Any
    """The dataset to read through -- wrapped, or the original."""
    unit: str
    """Unit of what ``dset`` yields; ``""`` when that is raw stored samples,
    because counts are not volts and labelling them so is the whole bug."""
    scaling: typing.Optional[StreamScaling]
    """None only for a stream that has no scaling to speak of (text)."""


def resolve_stream_scaling(
    dset: typing.Any,
    channel_conversion: typing.Any = None,
    *,
    apply: bool = True,
    dtype: typing.Any = DEFAULT_CONVERSION_DTYPE,
    scale_override: typing.Any = None,
    unit_override: typing.Any = None,
    target_unit: typing.Any = None,
    is_electrical: bool = False,
    stream_key: str = "",
) -> ScalingResult:
    """Resolve a stream's scaling, and apply it unless *apply* is False.

    Resolved either way, on purpose. ``apply=False`` asks for the stored
    samples, not for the factors to be forgotten: a caller who wants counts
    usually wants to scale them later (in a processing stage, on a GPU, after
    decimation), and dropping the factors here would make them reopen the file
    to recover what this function already had in hand. So the numbers are
    reported on the message either way, and ``applied`` says which it is.

    Returns *dset* unwrapped when the resolved scaling is the identity -- there
    is nothing to compute, so a stored int16 stays int16 rather than becoming a
    float32 copy of the same values.
    """
    if not isinstance(dset, h5py.Dataset) or dset.dtype.kind not in "iuf":
        # A materialized text column: no gain, and nothing to say about one.
        return ScalingResult(dset, "", None)

    gain, offset, unit = resolve_scaling(
        dset,
        channel_conversion,
        scale_override=scale_override,
        unit_override=unit_override,
        target_unit=target_unit,
        is_electrical=is_electrical,
        stream_key=stream_key,
    )
    gain = gain if isinstance(gain, np.ndarray) else float(gain)
    voltage = is_voltage_stream(unit, is_electrical)

    if not apply:
        return ScalingResult(dset, "", StreamScaling(gain, offset, unit, applied=False, voltage=voltage))
    scaling = StreamScaling(gain, offset, unit, applied=True, voltage=voltage)
    if is_identity_scaling(gain, offset):
        return ScalingResult(dset, unit, scaling)
    return ScalingResult(ScaledDataset(dset, gain, offset, dtype), unit, scaling)
