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
(:func:`resolve_scaling`) and describes them for the reader to attach to every
message (:func:`describe_stream_scaling`, :class:`StreamScaling`).

**Nothing here applies anything.** The reader hands back the integers the file
stores, plus this description of what they mean;
:class:`~ezmsg.nwb.convert.NWBScalingUnit` does the arithmetic, wherever in the
graph it belongs. Splitting it that way is what keeps the two halves honest: the
reader has one job and states a fact about the file, the transformer has one job
and states what it did. It is also no slower -- measured on a real recording at
1 s chunks, reading int16 and converting downstream beats converting during the
read, because the reader then moves half as many bytes and only the messages
that survive get widened.

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

That is why the scaling is overridable per stream on the transformer
(``scale_override`` / ``unit_override``): reporting the file verbatim is the
right thing for a reader to do, but a caller who knows a particular writer lies
needs to say so without patching the file.

Separately from *what the file means*, a pipeline has an opinion about *what it
wants on the wire*: ``target_unit`` converts a voltage stream to a requested
:class:`VoltageUnit` on top of whatever the file (as corrected) turned out to
be, so a graph can be written against microvolts and fed by files that disagree
about their own scale. It applies only to voltage streams -- see
:func:`is_voltage_stream` and :func:`convert_to_target_unit`.
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
) -> tuple[typing.Union[float, np.ndarray], float, str]:
    """Collapse a stream's stored scaling into one ``(gain, offset, unit)``.

    Verbatim: what the file says, with nothing corrected and nothing converted.
    Overrides and target units belong to :mod:`~ezmsg.nwb.convert`, which is
    where the applying happens; a reader that pre-corrected would leave the
    reported scaling describing neither the file nor the data.

    ``gain`` is ``conversion * channel_conversion`` -- a scalar when the
    per-channel factors are absent or all equal (the common case, and the one
    that keeps the multiply a scalar broadcast), a vector only when the channels
    genuinely disagree.
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

    return gain, offset, unit


def is_identity_scaling(gain: typing.Union[float, np.ndarray], offset: float) -> bool:
    """Whether applying this scaling would leave every sample unchanged.

    Worth checking: a stream that declares no scaling is already in the unit it
    claims, so wrapping it would only trade its stored dtype for a float32 copy
    of the same numbers -- a needless doubling on an int16 broadband stream, and
    a surprise for a caller reading an integer marker channel.

    The scalar case avoids numpy entirely. ``np.all(np.asarray(x) == 1.0)`` on a
    Python float costs ~1.5 us -- two array allocations and a reduction to
    answer a question ``==`` answers -- which is a third of the budget for a
    30-sample message and was the single largest per-message cost in the
    downstream transformer.
    """
    if offset:
        return False
    if isinstance(gain, float):
        return gain == 1.0
    return bool(np.all(np.asarray(gain) == 1.0))


SCALING_ATTR = "nwb_scaling"
"""Prefix under which a message reports the scaling that relates it to the file.

Namespaced rather than flat ``attrs["conversion"]``/``attrs["offset"]``: attrs
is a shared dict that any stage downstream may add to, and three unqualified
generic words are a collision waiting to happen.

Five prefixed keys rather than one nested dict under this name, because a dict
is opaque to every stage that walks attrs generically. ``ezmsg-sigproc``'s
``concat`` is the case that forced it: it merges two messages' attrs and rejects
any value that is not a scalar, so a nested dict raised ``TypeError`` even when
both sides carried an identical one. Flat keys also let it do the *right* thing
when two streams disagree -- an unequal ``nwb_scaling_gain`` gets promoted onto
the ``ch`` axis, which is where a per-channel gain belongs and is something a
dict could never express.
"""

GAIN_ATTR = f"{SCALING_ATTR}_gain"
OFFSET_ATTR = f"{SCALING_ATTR}_offset"
UNIT_ATTR = f"{SCALING_ATTR}_unit"
APPLIED_ATTR = f"{SCALING_ATTR}_applied"
VOLTAGE_ATTR = f"{SCALING_ATTR}_voltage"

SCALING_ATTRS = (GAIN_ATTR, OFFSET_ATTR, UNIT_ATTR, APPLIED_ATTR, VOLTAGE_ATTR)
"""Every key :meth:`StreamScaling.as_attrs` writes, for stripping or checking."""


def scaling_fingerprint(attrs: typing.Mapping[str, typing.Any]) -> typing.Optional[tuple]:
    """A hashable summary of a message's reported scaling, or None if it has none.

    Compares *by value*. The nested-dict form could be cached on the identity of
    the one payload object, which is cheaper, but that only worked because the
    reader built each stream's attrs once and every message shared it: an
    upstream stage that rebuilds attrs per message would have thrashed such a
    cache anyway, and with five separate keys there is no single object left to
    anchor on.

    A vector gain is summarised by its bytes rather than its ``id``, so that an
    array rebuilt with the same contents still hits. That costs a pass over the
    channel axis, but it is a few hundred floats against a message of samples,
    and it avoids having to pin the array alive to keep its address unique.
    """
    if GAIN_ATTR not in attrs:
        return None
    gain = attrs[GAIN_ATTR]
    return (
        gain.tobytes() if isinstance(gain, np.ndarray) else gain,
        attrs[OFFSET_ATTR],
        attrs[UNIT_ATTR],
        attrs[APPLIED_ATTR],
        attrs.get(VOLTAGE_ATTR, False),
    )


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

    def as_attrs(self) -> dict[str, typing.Any]:
        """The ``nwb_scaling_*`` entries to merge into a message's attrs.

        Flat scalars, so every stage that walks attrs generically can read,
        compare, and merge them -- see :data:`SCALING_ATTR`. ``gain`` is the one
        that can still be a non-scalar, when the channels genuinely disagree.
        """
        return {
            GAIN_ATTR: self.gain,
            OFFSET_ATTR: self.offset,
            UNIT_ATTR: self.unit,
            APPLIED_ATTR: self.applied,
            VOLTAGE_ATTR: self.voltage,
        }

    @classmethod
    def from_attrs(cls, attrs: typing.Mapping[str, typing.Any]) -> typing.Optional[StreamScaling]:
        """Rebuild from a message's attrs, or None if it reports no scaling.

        Keyed on ``gain``: a message either carries the whole set or none of it,
        and treating a partial set as absent is the safe reading -- inventing a
        gain for a message that never declared one is the failure this module
        exists to prevent.
        """
        if GAIN_ATTR not in attrs:
            return None
        return cls(
            gain=attrs[GAIN_ATTR],
            offset=float(attrs[OFFSET_ATTR]),
            unit=str(attrs[UNIT_ATTR]),
            applied=bool(attrs[APPLIED_ATTR]),
            voltage=bool(attrs.get(VOLTAGE_ATTR, False)),
        )


def describe_stream_scaling(
    dset: typing.Any,
    channel_conversion: typing.Any = None,
    *,
    is_electrical: bool = False,
) -> typing.Optional[StreamScaling]:
    """What a stream's stored samples mean, for the reader to report.

    Describes; never applies. The reader hands back the integers the file
    stores and this alongside them, so :mod:`~ezmsg.nwb.convert` can do the
    arithmetic wherever in the graph it belongs. ``applied`` is therefore always
    False here -- it turns True only once something has actually multiplied.

    None for a stream with no scaling to speak of: a materialized text column
    has no gain, and reporting an identity one would invite a consumer to
    multiply strings.
    """
    if not isinstance(dset, h5py.Dataset) or dset.dtype.kind not in "iuf":
        return None
    gain, offset, unit = resolve_scaling(dset, channel_conversion)
    return StreamScaling(
        gain=gain if isinstance(gain, np.ndarray) else float(gain),
        offset=offset,
        unit=unit,
        applied=False,
        voltage=is_voltage_stream(unit, is_electrical),
    )
