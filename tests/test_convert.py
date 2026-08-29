"""Applying an NWB stream's scaling, which only this module does.

The reader reports; this converts. So the expectations here are written against
the NWB definition directly -- ``data * conversion * channel_conversion +
offset`` -- rather than against another code path, since there is no longer a
second implementation to agree with.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import CHANNEL_CONVERSION, CONVERSION, DECLARED_UNIT, MARKER_N, OFFSET, expected
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.nwb import (
    APPLIED_ATTR,
    GAIN_ATTR,
    OFFSET_ATTR,
    UNIT_ATTR,
    VOLTAGE_ATTR,
    NWBScalingSettings,
    NWBScalingTransformer,
    NWBSlicer,
    StreamScaling,
)


def message(data: np.ndarray, scaling: StreamScaling | None, key: str = "Broadband") -> AxisArray:
    attrs = {} if scaling is None else scaling.as_attrs()
    if scaling is not None and scaling.applied:
        attrs["unit"] = scaling.unit
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.LinearAxis.create_time_axis(fs=500.0)},
        attrs=attrs,
        key=key,
    )


def assert_same_scaling(got: AxisArray, want: AxisArray) -> None:
    """Compare two messages' reported scaling; ``gain`` may be an array, so
    ``==`` on the values is ambiguous rather than False."""
    np.testing.assert_allclose(got.attrs[GAIN_ATTR], want.attrs[GAIN_ATTR], rtol=1e-9)
    assert got.attrs[OFFSET_ATTR] == pytest.approx(want.attrs[OFFSET_ATTR])
    assert [got.attrs[k] for k in (UNIT_ATTR, APPLIED_ATTR, VOLTAGE_ATTR)] == [
        want.attrs[k] for k in (UNIT_ATTR, APPLIED_ATTR, VOLTAGE_ATTR)
    ]


def read(path, key: str = "Broadband", stop: int = 20) -> AxisArray:
    """A message as the reader emits it: stored samples plus their description."""
    slicer = NWBSlicer(path, dejitter=False)
    try:
        return slicer.read_by_index(key, 0, stop)
    finally:
        slicer.close()


def scaled(path, **settings) -> AxisArray:
    """...and the same message with the conversion applied."""
    return NWBScalingTransformer(settings=NWBScalingSettings(**settings))(read(path))


# --- The point of the thing --------------------------------------------------


def test_it_applies_what_the_message_reports(scaled_nwb_path):
    """The reader hands over int16 and a description; this turns one into the
    other. Checked against the NWB definition directly -- data * conversion *
    channel_conversion + offset -- not against another code path."""
    raw = read(scaled_nwb_path)
    out = NWBScalingTransformer(settings=NWBScalingSettings())(raw)

    assert raw.data.dtype == np.int16 and "unit" not in raw.attrs
    np.testing.assert_array_equal(out.data, expected(raw.data))
    assert out.attrs["unit"] == DECLARED_UNIT
    assert out.attrs[APPLIED_ATTR] is True


def test_target_unit_converts_on_top_of_the_files_own(scaled_nwb_path):
    """The fixture declares microvolts; asking for volts scales gain and offset
    alike, so the samples come out 1e6 times smaller."""
    raw = read(scaled_nwb_path)
    out = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="volts"))(raw)
    np.testing.assert_allclose(out.data, expected(raw.data) * 1e-6, rtol=1e-6)
    assert out.attrs["unit"] == out.attrs[UNIT_ATTR] == "volts"


def test_overrides_correct_the_file_then_target_converts(scaled_nwb_path):
    """Order matters: the overrides say what the file really holds, target_unit
    says what to deliver. Relabelling as volts and asking for microvolts is a net
    1e6, where believing the file's "microvolts" would have been a no-op."""
    raw = read(scaled_nwb_path)
    out = NWBScalingTransformer(
        settings=NWBScalingSettings(scale_override=0.1, unit_override="volts", target_unit="microvolts")
    )(raw)
    np.testing.assert_allclose(out.data, (raw.data * 0.1 + OFFSET) * 1e6, rtol=1e-5)
    assert out.attrs["unit"] == "microvolts"


# --- Idempotence -------------------------------------------------------------


def test_an_already_scaled_message_passes_through_untouched(scaled_nwb_path):
    """Off the ``applied`` flag, not off the dtype -- so a transformer behind a
    reader that already converted cannot double-scale."""
    at_reader = scaled(scaled_nwb_path)
    out = NWBScalingTransformer(settings=NWBScalingSettings())(at_reader)
    assert out is at_reader


def test_two_in_a_chain_scale_once(scaled_nwb_path):
    raw = read(scaled_nwb_path)
    tx = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="volts"))
    once = tx(raw)
    twice = tx(once)
    np.testing.assert_array_equal(twice.data, once.data)
    assert_same_scaling(twice, once)


def test_a_message_with_no_scaling_passes_through(scaled_nwb_path):
    """Not from an NWB reader. Inventing a scaling would be worse than nothing."""
    msg = message(np.ones((4, 2), dtype=np.int16), None)
    out = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="volts"))(msg)
    assert out is msg


# --- Retargeting something already applied -----------------------------------


def test_target_unit_retargets_an_already_scaled_message(scaled_nwb_path):
    """A prefix change on top of a known unit is well defined, so this is
    allowed where the gain overrides are not. One multiply: scaling the values
    carries their offset with them."""
    at_reader = scaled(scaled_nwb_path)  # microvolts
    out = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="volts"))(at_reader)
    np.testing.assert_allclose(out.data, at_reader.data * 1e-6, rtol=1e-6)
    assert out.attrs["unit"] == "volts"
    # The reported scaling stays cumulative: stored -> now, not previous -> now.
    np.testing.assert_allclose(out.attrs[GAIN_ATTR], np.asarray(CONVERSION * CHANNEL_CONVERSION) * 1e-6, rtol=1e-6)
    assert out.attrs[OFFSET_ATTR] == pytest.approx(OFFSET * 1e-6)


def test_retargeting_matches_asking_the_reader_directly(scaled_nwb_path):
    at_reader = scaled(scaled_nwb_path)
    direct = scaled(scaled_nwb_path, target_unit="nanovolts")
    out = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="nanovolts"))(at_reader)
    np.testing.assert_allclose(out.data, direct.data, rtol=1e-5)


@pytest.mark.parametrize("setting", ["scale_override", "unit_override"])
def test_gain_overrides_refuse_an_already_scaled_message(scaled_nwb_path, setting):
    """Correcting a gain already applied would mean dividing it back out. Say
    so rather than half-doing it or silently ignoring the setting."""
    at_reader = scaled(scaled_nwb_path)
    value = 0.1 if setting == "scale_override" else "volts"
    tx = NWBScalingTransformer(settings=NWBScalingSettings(**{setting: value}))
    with pytest.raises(ValueError, match="already applied upstream"):
        tx(at_reader)


# --- Non-voltage, identity, and the per-channel hazard ------------------------


def test_non_voltage_streams_are_scaled_but_not_retargeted(scaled_nwb_path):
    """ "pixels" gets its gain applied -- it has one -- and keeps its unit."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        raw = slicer.read_by_index("Cursor", 0, 10)
    finally:
        slicer.close()
    out = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="volts"))(raw)
    assert out.attrs["unit"] == "pixels"
    np.testing.assert_allclose(out.data, raw.data * np.float32(CONVERSION), rtol=1e-6)


def test_identity_scaling_keeps_the_stored_dtype(scaled_nwb_path):
    """The marker stream is already in its declared unit. Record that without
    copying int16 into float to do it."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        raw = slicer.read_by_index("Marker", 0, MARKER_N)
    finally:
        slicer.close()
    out = NWBScalingTransformer(settings=NWBScalingSettings())(raw)
    assert out.data.dtype == np.int16
    assert out.attrs["unit"] == "n/a"
    assert out.attrs[APPLIED_ATTR] is True


def test_a_stale_per_channel_gain_raises_rather_than_broadcasting(scaled_nwb_path):
    """A per-channel gain is positional, so a stage that dropped channels
    upstream has invalidated it. Length is the only check available, and a
    misaligned multiply that happened to broadcast would be silently wrong."""
    raw = read(scaled_nwb_path)
    subset = AxisArray(data=raw.data[:, :2], dims=raw.dims, axes=raw.axes, attrs=dict(raw.attrs), key=raw.key)
    with pytest.raises(ValueError, match="does not fit"):
        NWBScalingTransformer(settings=NWBScalingSettings())(subset)


def test_conversion_dtype_is_honoured(scaled_nwb_path, counts):
    raw = read(scaled_nwb_path)
    out = NWBScalingTransformer(settings=NWBScalingSettings(conversion_dtype="float64"))(raw)
    assert out.data.dtype == np.float64
    np.testing.assert_allclose(
        out.data, counts[:20].astype(np.float64) * CONVERSION * CHANNEL_CONVERSION + OFFSET, rtol=1e-12
    )


# --- The per-stream plan cache -----------------------------------------------


def test_interleaved_streams_do_not_evict_each_other(scaled_nwb_path):
    """One reader publishes every stream on one output, so a single cached plan
    would be rebuilt on every message and two streams would swap scalings. Pin
    that alternating keys stay correct."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        broadband = slicer.read_by_index("Broadband", 0, 10)
        aux = slicer.read_by_index("AuxVoltage", 0, 10)
    finally:
        slicer.close()
    tx = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="microvolts"))

    # Interleaved, several rounds, so a one-entry cache would thrash and a
    # cross-contaminated one would show up as the wrong gain.
    outs = [tx(m) for _ in range(3) for m in (broadband, aux)]
    for i in range(0, len(outs), 2):
        # Broadband declares microvolts already: its per-channel gain, no unit change.
        np.testing.assert_allclose(outs[i].data, expected(broadband.data), rtol=1e-5)
        # AuxVoltage declares volts: scalar gain, and a 1e6 unit change on top.
        np.testing.assert_allclose(outs[i + 1].data, (aux.data * CONVERSION + OFFSET) * 1e6, rtol=1e-5)
    # Both land in the requested unit despite starting from different ones.
    assert {o.attrs["unit"] for o in outs} == {"microvolts"}


def test_a_changed_scaling_rebuilds_the_plan(scaled_nwb_path):
    """The cache is keyed on the reported scaling, held by reference. A message
    carrying a different one must not reuse the old plan."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        msg = slicer.read_by_index("Broadband", 0, 10)
    finally:
        slicer.close()
    tx = NWBScalingTransformer(settings=NWBScalingSettings())
    first = tx(msg)

    doubled = StreamScaling(np.asarray(msg.attrs[GAIN_ATTR]) * 2.0, OFFSET, DECLARED_UNIT, False, True)
    changed = AxisArray(
        data=msg.data,
        dims=msg.dims,
        axes=msg.axes,
        attrs=doubled.as_attrs(),
        key=msg.key,
    )
    second = tx(changed)
    np.testing.assert_allclose(second.data, (msg.data * CONVERSION * CHANNEL_CONVERSION * 2.0) + OFFSET, rtol=1e-5)
    assert not np.allclose(second.data, first.data)


def test_changed_settings_invalidate_cached_plans(scaled_nwb_path):
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        msg = slicer.read_by_index("Broadband", 0, 10)
    finally:
        slicer.close()
    tx = NWBScalingTransformer(settings=NWBScalingSettings(target_unit="microvolts"))
    before = tx(msg)
    # ``update_settings`` is the framework's path: it diffs the fields and asks
    # for a reset, which is what gives the cache a chance to notice.
    tx.update_settings(NWBScalingSettings(target_unit="volts"))
    after = tx(msg)
    assert before.attrs["unit"] == "microvolts"
    assert after.attrs["unit"] == "volts"
    np.testing.assert_allclose(after.data, before.data * 1e-6, rtol=1e-5)


def test_rebuilt_attrs_with_the_same_scaling_reuse_the_plan():
    """The per-message check keys on the attrs dict's identity, so a stage that
    rebuilds attrs upstream lands in ``_reset_state`` -- whose by-value check
    must then recognise the scaling and keep the plan rather than re-resolving
    overrides and units on every message."""
    scaling = StreamScaling(gain=0.25, offset=0.0, unit="volts", applied=False, voltage=True)
    tx = NWBScalingTransformer(settings=NWBScalingSettings())

    first = message(np.ones((4, 2), dtype=np.int16), scaling)
    tx(first)
    plan = tx.state.plans["Broadband"]

    # Same values, different dict object -- as an upstream stage would produce.
    rebuilt = message(np.ones((4, 2), dtype=np.int16), scaling)
    assert rebuilt.attrs is not first.attrs
    tx(rebuilt)

    assert tx.state.plans["Broadband"] is plan
    # And the plan must now pin the dict the identity check is keyed on.
    assert tx.state.plans["Broadband"].attrs_ref is rebuilt.attrs


def test_a_genuinely_changed_scaling_still_rebuilds_the_plan():
    tx = NWBScalingTransformer(settings=NWBScalingSettings())
    tx(message(np.ones((4, 2), dtype=np.int16), StreamScaling(0.25, 0.0, "volts", False, True)))
    plan = tx.state.plans["Broadband"]

    tx(message(np.ones((4, 2), dtype=np.int16), StreamScaling(0.50, 0.0, "volts", False, True)))

    assert tx.state.plans["Broadband"] is not plan
