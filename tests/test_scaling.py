"""What an NWB file says its samples mean, and how the reader reports it.

The reader never applies a scaling -- it hands back the stored integers and
describes them. Everything about *applying* lives in ``test_convert.py``. The
``scaled_nwb_path`` fixture is shared via ``conftest.py`` because the two
modules are two halves of one contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import (
    CHANNEL_CONVERSION,
    CONVERSION,
    DECLARED_UNIT,
    MARKER_N,
    N_CH,
    N_SAMPLES,
    OFFSET,
)

from ezmsg.nwb import (
    APPLIED_ATTR,
    GAIN_ATTR,
    MICROVOLT_UNITS,
    OFFSET_ATTR,
    SCALING_ATTRS,
    UNIT_ATTR,
    VOLTAGE_ATTR,
    NWBAxisArrayIterator,
    NWBIteratorSettings,
    NWBSlicer,
    ReferenceClockType,
    StreamScaling,
    VoltageUnit,
    describe_stream_scaling,
    is_voltage_stream,
    parse_voltage_unit,
    read_stored_scaling,
    resolve_scaling,
)

# --- The stored factors, read back ------------------------------------------


def test_read_stored_scaling_reports_the_declared_unit_not_the_schema_one(scaled_nwb_path):
    """pynwb answers "volts" for every ElectricalSeries because the schema fixes
    that field. The file says microvolts; a reader that believes pynwb is off by
    1e6."""
    import h5py
    import pynwb

    with pynwb.NWBHDF5IO(str(scaled_nwb_path), "r") as io:
        assert io.read().acquisition["Broadband"].unit == "volts"

    with h5py.File(str(scaled_nwb_path), "r") as f:
        conversion, offset, unit = read_stored_scaling(f["acquisition/Broadband/data"])
    assert conversion == pytest.approx(CONVERSION)
    assert offset == pytest.approx(OFFSET)
    assert unit == DECLARED_UNIT
    assert unit.lower() in MICROVOLT_UNITS


def test_resolve_scaling_folds_channel_conversion_into_the_gain(scaled_nwb_path):
    import h5py

    with h5py.File(str(scaled_nwb_path), "r") as f:
        gain, offset, unit = resolve_scaling(
            f["acquisition/Broadband/data"], f["acquisition/Broadband/channel_conversion"]
        )
    np.testing.assert_allclose(gain, CONVERSION * CHANNEL_CONVERSION, rtol=1e-6)
    assert offset == pytest.approx(OFFSET)
    assert unit == DECLARED_UNIT


def test_resolve_scaling_keeps_uniform_channel_conversion_scalar(scaled_nwb_path):
    """All-equal per-channel gains collapse to a scalar, so the multiply stays a
    scalar broadcast instead of a vector one."""
    import h5py

    with h5py.File(str(scaled_nwb_path), "r") as f:
        gain, _, _ = resolve_scaling(f["acquisition/Broadband/data"], np.full(N_CH, 2.0))
    assert np.ndim(gain) == 0
    assert gain == pytest.approx(CONVERSION * 2.0)


def test_text_streams_are_described_as_having_no_scaling():
    """A text series is materialized into a str array before this point. It has
    no gain, so it reports no scaling at all rather than a misleading identity
    one -- which would invite a consumer to multiply strings."""
    assert describe_stream_scaling(np.array(["left", "right"])) is None


# --- Through the slicer: stored samples, plus a description ------------------


def test_the_slicer_returns_the_stored_samples(scaled_nwb_path, counts):
    """No conversion on read, ever. Applying it is ``NWBScalingUnit``'s job, and
    keeping the reader to one job is why the split exists."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        info = slicer.get_stream_info("Broadband")
        np.testing.assert_array_equal(info.dset[:], counts)
        assert info.dset.dtype == np.int16
    finally:
        slicer.close()


def test_the_slicer_reports_what_the_samples_mean(scaled_nwb_path):
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        info = slicer.get_stream_info("Broadband")
        sc = StreamScaling.from_attrs(info.template.attrs)
    finally:
        slicer.close()
    np.testing.assert_allclose(sc.gain, CONVERSION * CHANNEL_CONVERSION, rtol=1e-6)
    assert sc.offset == pytest.approx(OFFSET)
    assert sc.unit == DECLARED_UNIT
    assert sc.applied is False
    assert sc.voltage is True
    # No ``unit`` key: the data is counts, and calling counts microvolts is the
    # bug this whole module exists to prevent. The unit lives inside the
    # scaling, where it reads as pending rather than as a claim about the data.
    assert "unit" not in info.template.attrs


def test_the_reported_scaling_is_the_files_own_numbers(scaled_nwb_path):
    """Verbatim, uncorrected. A reader that pre-corrected would leave the report
    describing neither the file nor the data."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        assert slicer.get_stream_info("Broadband").scaling.unit == DECLARED_UNIT
        assert slicer.get_stream_info("Cursor").scaling.unit == "pixels"
        assert slicer.get_stream_info("Marker").scaling.unit == "n/a"
    finally:
        slicer.close()


def test_read_by_index_carries_the_scaling(scaled_nwb_path, counts):
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        msg = slicer.read_by_index("Broadband", 10, 20)
    finally:
        slicer.close()
    np.testing.assert_array_equal(msg.data, counts[10:20])
    assert msg.attrs[UNIT_ATTR] == DECLARED_UNIT


def test_reported_gain_is_scalar_when_the_channels_agree(scaled_nwb_path):
    """Uniform per-channel factors collapse, so the common case reports one
    number rather than an array a downstream stage has to keep aligned with the
    ch axis through every channel selection."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        broadband = slicer.read_by_index("Broadband", 0, 5).attrs[GAIN_ATTR]
        aux = slicer.read_by_index("AuxVoltage", 0, 5).attrs[GAIN_ATTR]
    finally:
        slicer.close()
    # Broadband's channel_conversion entries differ, so its gain stays a vector.
    assert np.ndim(broadband) == 1
    # AuxVoltage has none at all.
    assert isinstance(aux, float)


def test_reported_scaling_survives_the_message_codec(scaled_nwb_path):
    """A vector gain puts a numpy array inside attrs, which is a shape nothing
    else in these messages has. Pin that it logs and reloads."""
    from ezmsg.util.messagecodec import MessageDecoder, MessageEncoder

    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        msg = slicer.read_by_index("Broadband", 0, 5)
    finally:
        slicer.close()
    reloaded = json.loads(json.dumps(msg, cls=MessageEncoder), cls=MessageDecoder)
    np.testing.assert_allclose(reloaded.attrs[GAIN_ATTR], msg.attrs[GAIN_ATTR], rtol=1e-9)
    assert reloaded.attrs[UNIT_ATTR] == DECLARED_UNIT and reloaded.attrs[APPLIED_ATTR] is False


def test_iterator_messages_are_stored_samples_carrying_the_scaling(scaled_nwb_path, counts):
    it = NWBAxisArrayIterator(
        settings=NWBIteratorSettings(
            filepath=str(scaled_nwb_path),
            chunk_dur=0.2,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
            dejitter=False,
        )
    )
    try:
        chunks = [msg for msg in it if msg.data.shape[0]]
    finally:
        it.close()
    got = np.concatenate([m.data for m in chunks], axis=0)
    assert got.dtype == np.int16
    np.testing.assert_array_equal(got, counts[: got.shape[0]])
    np.testing.assert_allclose(chunks[0].attrs[GAIN_ATTR], CONVERSION * CHANNEL_CONVERSION, rtol=1e-6)
    assert chunks[0].attrs[APPLIED_ATTR] is False


def test_marker_streams_keep_their_dtype_and_declare_identity(scaled_nwb_path):
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        info = slicer.get_stream_info("Marker")
        assert info.dset.dtype == np.int16
        np.testing.assert_array_equal(info.dset[:], np.arange(MARKER_N, dtype=np.int16))
        assert info.scaling.gain == 1.0 and info.scaling.offset == 0.0
    finally:
        slicer.close()


def test_streams_are_all_loaded(scaled_nwb_path):
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        assert slicer.get_stream_info("Broadband").n_samples == N_SAMPLES
    finally:
        slicer.close()


# --- Units ------------------------------------------------------------------


def test_is_voltage_stream_takes_either_kind_of_evidence():
    assert is_voltage_stream("volts", is_electrical=False)
    assert is_voltage_stream("uV", is_electrical=False)
    # An ElectricalSeries is voltage by schema whatever string it carries --
    # including none, which is what pynwb writes when nobody stamped one.
    assert is_voltage_stream("", is_electrical=True)
    assert is_voltage_stream("garbage", is_electrical=True)
    # A bare TimeSeries gets no such benefit of the doubt: an unstamped unit
    # says nothing, and assuming volts would invent a dimension.
    assert not is_voltage_stream("", is_electrical=False)
    assert not is_voltage_stream("pixels", is_electrical=False)


def test_voltage_written_as_a_plain_timeseries_is_recognized(scaled_nwb_path):
    """Regression: gating on ``isinstance(..., ElectricalSeries)`` alone left a
    ``*_device_ts`` companion -- a plain TimeSeries over the *same* samples as
    its acquisition partner -- unconvertible while the partner converted."""
    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        assert slicer.get_stream_info("AuxVoltage").scaling.voltage is True
        assert slicer.get_stream_info("Broadband").scaling.voltage is True
        assert slicer.get_stream_info("Cursor").scaling.voltage is False
    finally:
        slicer.close()


def test_unit_spellings_are_all_accepted():
    """A caller writing a setting should not have to know which spelling this
    module canonicalized on."""
    assert parse_voltage_unit("uV") is parse_voltage_unit("µV") is VoltageUnit.MICROVOLTS
    assert parse_voltage_unit("V") is VoltageUnit.VOLTS
    assert parse_voltage_unit("nanovolts") is VoltageUnit.NANOVOLTS
    assert parse_voltage_unit("pixels") is None


# --- The shape the scaling takes in attrs ------------------------------------


class TestFlatAttrShape:
    """Five prefixed scalars, not one nested dict.

    The shape is part of the contract, not an implementation detail: a nested
    dict is opaque to any stage that walks attrs generically, and ezmsg-sigproc's
    ``concat`` rejects one outright -- even when both messages carry an identical
    one. See :data:`~ezmsg.nwb.scaling.SCALING_ATTR`.
    """

    def test_no_value_is_a_container(self, scaled_nwb_path):
        slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
        try:
            attrs = slicer.read_by_index("AuxVoltage", 0, 5).attrs
        finally:
            slicer.close()
        assert set(attrs) == set(SCALING_ATTRS)
        for key, value in attrs.items():
            assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value).__name__}"

    def test_a_vector_gain_is_the_one_documented_exception(self, scaled_nwb_path):
        """Broadband's channel_conversion entries genuinely differ, so its gain
        stays an array. Every other key is still a scalar."""
        slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
        try:
            attrs = slicer.read_by_index("Broadband", 0, 5).attrs
        finally:
            slicer.close()
        assert isinstance(attrs[GAIN_ATTR], np.ndarray)
        for key in (OFFSET_ATTR, UNIT_ATTR, APPLIED_ATTR, VOLTAGE_ATTR):
            assert isinstance(attrs[key], (str, int, float, bool))

    def test_attrs_round_trip_through_stream_scaling(self):
        original = StreamScaling(gain=0.25, offset=1.5, unit="volts", applied=True, voltage=True)

        assert StreamScaling.from_attrs(original.as_attrs()) == original

    def test_a_message_with_no_scaling_reads_back_as_none(self):
        assert StreamScaling.from_attrs({}) is None
        assert StreamScaling.from_attrs({"unit": "volts"}) is None
