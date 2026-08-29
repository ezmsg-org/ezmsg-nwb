"""Build a structured ``ch`` axis from an NWB electrodes table.

A live acquisition source hands downstream stages a structured ``ch``
:class:`~ezmsg.util.messages.axisarray.CoordinateAxis` -- per-channel geometry,
labels, and array identity -- rather than a bare list of names. Rereferencing
per array, plotting by position, and selecting by headstage all read those
fields. Replaying the same recording through the same graph gave a plain string
array instead, so every one of those stages had to special-case its source.

This builds the live layout from what an NWB writer put in the electrodes table,
so an offline graph sees the same axis as an online one. :data:`CHANNEL_DTYPE`
deliberately matches ``ezmsg.blackrock.channel_map.CHANNEL_DTYPE`` field for
field; the point is that weights fitted offline and applied live cluster
channels identically.

Off by default. The columns this reads beyond ``label`` are conventions of one
acquisition stack, not part of the NWB schema -- most files have none of them,
and inventing geometry for a file that never recorded any would be worse than
saying so.

**Coordinates are taken verbatim.** ``rel_x``/``rel_y`` are stored here in
micrometers, matching the device's own channel-map units and the ``int32``
micrometers the live source emits. NWB's schema nominally describes electrode
positions in meters, so a file that followed that reading would land three
orders of magnitude away. Converting on a guess is how a scale factor becomes
silently wrong (see :mod:`~ezmsg.nwb.scaling` for the same lesson learned on
sample values), so nothing here rescales; a file in meters needs its writer
fixed or its axis rebuilt downstream.
"""

from __future__ import annotations

import typing

import ezmsg.core as ez
import numpy as np

from .util import as_text_array

CHANNEL_DTYPE = np.dtype(
    [
        ("x", "i4"),  # electrode x, µm
        ("y", "i4"),  # electrode y, µm
        ("size", "i4"),  # electrode size, µm (0 = unspecified)
        ("label", "U16"),
        ("bank", "U1"),
        ("elec", "i4"),
        ("headstage", "i4"),  # 1-based headstage id (0 = none/auto)
        ("array", "U32"),  # electrode-array identity; see array_identity
    ]
)
"""Per-channel record, field-for-field identical to the live source's.

Kept in lockstep deliberately. A downstream stage that reads ``ch['array']`` or
``ch['x']`` must not need to know whether its data came from a device or a file.
"""

COLUMN_SOURCES: dict[str, tuple[str, ...]] = {
    "x": ("rel_x",),
    "y": ("rel_y",),
    "size": ("size",),
    "label": ("label",),
    "bank": ("bank",),
    "elec": ("term", "elec"),
    "headstage": ("headstage",),
}
"""Which electrodes-table columns feed each field, in order of preference.

``rel_x``/``rel_y`` only, never NWB's ``x``/``y``. They are different
quantities: ``rel_*`` is the position *within the electrode group*, which is
what a channel map describes and what the live source puts in these fields,
while ``x``/``y``/``z`` are the location in the brain, nominally in metres.
Accepting the latter as a fallback would fill a micrometre field with
millimetre-scale brain coordinates -- and, because almost every NWB file with
electrodes carries ``x``/``y``, it would also make :func:`has_channel_metadata`
answer True for files that have no channel map at all.
"""


def array_identity(label: str, headstage: int, bank: str = "") -> str:
    """Electrode-array identity for a channel.

    A transcription of ``ezmsg.blackrock.channel_map._array_identity`` -- two
    channels belong to the same array iff they share a connector (the label
    prefix before the first ``-``) on the same headstage::

        ("elec1-m1-63", 1) -> "hs1-elec1"
        ("elec1-m1-63", 2) -> "hs2-elec1"

    Labels with no connector structure fall back to the bank, so grouping
    degrades to bank-level rather than collapsing every such channel into one
    cluster. Returns ``""`` when neither is available.

    Copied rather than imported: ezmsg-blackrock is not a dependency of this
    package and should not become one just to read a file. The duplication is
    the price of that, and :func:`array_identity` is small and pinned by a test
    that spells out the mapping it must produce.
    """
    hs = int(headstage or 0)
    prefix = f"hs{hs}-" if hs > 0 else ""
    connector, sep, _rest = str(label or "").partition("-")
    if sep and connector:
        return f"{prefix}{connector}"
    bank = str(bank or "")
    return f"{prefix}bank{bank}" if bank else ""


def _bank_letter(value: typing.Any) -> str:
    """A 1-based numeric bank as the letter the live source uses (1 -> ``A``).

    Already-lettered banks pass through, so a writer that stored ``"A"`` and one
    that stored ``1`` land in the same place.
    """
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        text = value.decode() if isinstance(value, bytes) else str(value)
        return text[:1]
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    return chr(ord("A") + number - 1) if 1 <= number <= 26 else ""


def _column(table: typing.Any, names: tuple[str, ...], region_idx: np.ndarray) -> typing.Optional[np.ndarray]:
    """One electrodes column, subset to a series' region, or None if absent.

    Reads the column directly rather than through ``to_dataframe()``: the whole
    table would be materialized into pandas to fetch a few of its columns, which
    costs ~12 ms per series at open.
    """
    available = set(getattr(table, "colnames", ()) or ())
    for name in names:
        if name in available:
            return np.asarray(table[name].data[:])[region_idx]
    return None


def has_channel_metadata(table: typing.Any) -> bool:
    """Whether this table carries anything beyond labels to build an axis from."""
    available = set(getattr(table, "colnames", ()) or ())
    return any(
        candidate in available for field, names in COLUMN_SOURCES.items() if field != "label" for candidate in names
    )


def build_channel_axis(
    table: typing.Any,
    region_idx: np.ndarray,
    *,
    stream_key: str = "",
) -> np.ndarray:
    """A :data:`CHANNEL_DTYPE` record per channel of one series.

    *region_idx* is the series' ``electrodes`` region -- the positional indices
    into the full table -- so a series referencing a subset of the electrodes
    gets that subset's rows, in its own order.

    Columns absent from the table leave their field at its zero value rather
    than failing: a writer that recorded positions but no headstage still
    produces a usable geometry. A table with no recognized column at all raises,
    because the caller asked for this explicitly and a record of all zeros would
    answer that request with fabricated data.
    """
    if not has_channel_metadata(table):
        available = sorted(getattr(table, "colnames", ()) or ())
        wanted = sorted({c for f, names in COLUMN_SOURCES.items() if f != "label" for c in names})
        raise ValueError(
            f"structured_ch_axis was requested but the electrodes table for {stream_key!r} has none of the "
            f"columns it is built from. Looked for {wanted}; the table has {available}. "
            f"These columns are an acquisition-stack convention, not part of the NWB schema -- leave "
            f"structured_ch_axis off for files that do not carry them."
        )

    n = len(region_idx)
    out = np.zeros(n, dtype=CHANNEL_DTYPE)

    for field in ("x", "y", "size", "elec", "headstage"):
        values = _column(table, COLUMN_SOURCES[field], region_idx)
        if values is not None:
            # Positions arrive as float µm; the live source's fields are int32.
            out[field] = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int32)

    labels = _column(table, COLUMN_SOURCES["label"], region_idx)
    if labels is not None:
        out["label"] = as_text_array(labels)

    banks = _column(table, COLUMN_SOURCES["bank"], region_idx)
    if banks is not None:
        out["bank"] = [_bank_letter(b) for b in banks]

    out["array"] = [array_identity(lbl, hs, bank) for lbl, hs, bank in zip(out["label"], out["headstage"], out["bank"])]
    _warn_on_merged_arrays(table, region_idx, out, stream_key)
    return out


def _warn_on_merged_arrays(
    table: typing.Any,
    region_idx: np.ndarray,
    out: np.ndarray,
    stream_key: str,
) -> None:
    """Warn when two electrode groups collapse into one ``array`` value.

    ``array`` is derived from the connector and headstage, which repeat across
    devices: two hubs each with an ``elec1`` on headstage 1 both produce
    ``hs1-elec1``. Within one series that is usually harmless -- a series is one
    device -- but a file that put two devices' electrodes in one series would
    silently merge two physically distinct arrays, and per-array rereferencing
    would then mix hemispheres. ``group_name`` distinguishes them, so it can say
    when this has happened even though it is not what the identity is built from
    (the live source has no such column, and matching it is the point).
    """
    groups = _column(table, ("group_name",), region_idx)
    if groups is None:
        return
    merged: dict[str, set[str]] = {}
    for ident, group in zip(out["array"], as_text_array(groups)):
        merged.setdefault(str(ident), set()).add(str(group))
    collisions = {k: v for k, v in merged.items() if len(v) > 1}
    if collisions:
        detail = "; ".join(f"{k!r} <- {sorted(v)}" for k, v in sorted(collisions.items()))
        ez.logger.warning(
            f"{stream_key or 'stream'}: distinct electrode groups share one array identity ({detail}). "
            f"Channels from physically separate arrays will group together; anything keyed on ch['array'] "
            f"-- per-array rereferencing especially -- will mix them."
        )
