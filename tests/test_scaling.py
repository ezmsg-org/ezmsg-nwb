"""Applying the scaling an NWB file stores alongside its samples.

The fixture deliberately uses a *non-trivial* value for all three factors --
a non-unit ``conversion``, a non-zero ``offset``, and a ``channel_conversion``
whose entries differ -- because each of them is separately easy to drop, and a
file where any of them is 1.0/0.0 cannot tell you whether it was applied.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.ecephys import ElectricalSeries

from ezmsg.nwb import (
    MICROVOLT_UNITS,
    NWBAxisArrayIterator,
    NWBIteratorSettings,
    NWBSlicer,
    ReferenceClockType,
    ScaledDataset,
    read_stored_scaling,
    resolve_scaling,
)

N_SAMPLES = 500
N_CH = 4
RATE = 500.0

CONVERSION = 0.25
OFFSET = 3.5
CHANNEL_CONVERSION = np.array([1.0, 2.0, 0.5, 4.0], dtype=np.float32)
DECLARED_UNIT = "microvolts"

# Marker stream: conversion 1.0 / offset 0.0, i.e. the values are already in the
# unit declared. Present to pin that such a stream is left alone rather than
# copied into float -- see ``is_identity_scaling``.
MARKER_N = 50


@pytest.fixture(scope="module")
def scaled_nwb_path(tmp_path_factory):
    """An NWB whose broadband declares all three scaling factors non-trivially."""
    path = tmp_path_factory.mktemp("scaling") / "scaled.nwb"
    rng = np.random.default_rng(7)

    nwbfile = NWBFile(
        session_description="scaling fixture",
        identifier="scaling001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    device = nwbfile.create_device(name="Array", description="fixture")
    group = nwbfile.create_electrode_group(name="Group", description="fixture", location="cortex", device=device)
    nwbfile.add_electrode_column(name="label", description="Electrode label")
    for i in range(N_CH):
        nwbfile.add_electrode(x=float(i), y=0.0, z=0.0, location="cortex", group=group, label=f"elec{i}")
    region = nwbfile.create_electrode_table_region(region=list(range(N_CH)), description="all")

    # int16 spanning a good part of the range, so a dropped conversion shows up
    # as a large error rather than a rounding one.
    counts = rng.integers(-30000, 30000, size=(N_SAMPLES, N_CH), dtype=np.int16)
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="Broadband",
            data=counts,
            rate=RATE,
            starting_time=0.0,
            electrodes=region,
            conversion=CONVERSION,
            offset=OFFSET,
            channel_conversion=CHANNEL_CONVERSION,
            description="fixture broadband",
        )
    )
    nwbfile.add_acquisition(
        TimeSeries(
            name="Marker",
            data=np.arange(MARKER_N, dtype=np.int16),
            unit="n/a",
            rate=RATE,
            starting_time=0.0,
            description="already in its declared unit",
        )
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    # pynwb pins ElectricalSeries.unit to the schema's "volts" and will not write
    # another value, so stamp the declared unit the way a real acquisition writer
    # does: straight onto the data dataset's attribute.
    import h5py

    with h5py.File(str(path), "a") as f:
        f["acquisition/Broadband/data"].attrs["unit"] = DECLARED_UNIT
    return path


@pytest.fixture(scope="module")
def counts(scaled_nwb_path) -> np.ndarray:
    import h5py

    with h5py.File(str(scaled_nwb_path), "r") as f:
        return np.asarray(f["acquisition/Broadband/data"][:])


def expected(counts: np.ndarray) -> np.ndarray:
    """``data * conversion * channel_conversion + offset``, the NWB definition."""
    return counts.astype(np.float32) * np.float32(CONVERSION) * CHANNEL_CONVERSION.astype(np.float32) + np.float32(
        OFFSET
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
    """All-equal per-channel gains collapse to a scalar, so the hot path stays a
    scalar broadcast instead of a vector one."""
    import h5py

    with h5py.File(str(scaled_nwb_path), "r") as f:
        gain, _, _ = resolve_scaling(f["acquisition/Broadband/data"], np.full(N_CH, 2.0))
    assert np.ndim(gain) == 0
    assert gain == pytest.approx(CONVERSION * 2.0)


# --- Through the slicer -----------------------------------------------------


def test_slicer_applies_all_three_factors(scaled_nwb_path, counts):
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, dejitter=False)
    try:
        info = slicer.get_stream_info("Broadband")
        np.testing.assert_array_equal(info.dset[:], expected(counts))
        assert info.dset.dtype == np.float32
        assert info.unit == DECLARED_UNIT
        assert info.template.attrs["unit"] == DECLARED_UNIT
    finally:
        slicer.close()


def test_slicer_off_returns_the_stored_counts(scaled_nwb_path, counts):
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=False, dejitter=False)
    try:
        info = slicer.get_stream_info("Broadband")
        np.testing.assert_array_equal(info.dset[:], counts)
        assert info.dset.dtype == np.int16
        assert info.unit == ""
    finally:
        slicer.close()


def test_identity_scaling_leaves_the_stored_dtype_alone(scaled_nwb_path):
    """A stream declaring conversion=1.0/offset=0.0 is already in its declared
    unit. Wrapping it would buy nothing and cost a float32 copy of every sample."""
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, dejitter=False)
    try:
        info = slicer.get_stream_info("Marker")
        assert info.dset.dtype == np.int16
        assert not isinstance(info.dset, ScaledDataset)
        np.testing.assert_array_equal(info.dset[:], np.arange(MARKER_N, dtype=np.int16))
    finally:
        slicer.close()


def test_read_by_index_is_scaled(scaled_nwb_path, counts):
    """The slice paths go through ``info.dset``, so they pick the scaling up too
    -- this is what the clock-driven replay producer reads through. The fixture
    stream is rate-only, so ``read_by_time`` (which needs explicit timestamps)
    is covered by the gappy-stream tests instead."""
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, dejitter=False)
    try:
        by_index = slicer.read_by_index("Broadband", 10, 20)
        np.testing.assert_array_equal(by_index.data, expected(counts)[10:20])
        assert by_index.attrs["unit"] == DECLARED_UNIT
    finally:
        slicer.close()


def test_conversion_dtype_is_honoured(scaled_nwb_path, counts):
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, conversion_dtype="float64", dejitter=False)
    try:
        info = slicer.get_stream_info("Broadband")
        assert info.dset.dtype == np.float64
        np.testing.assert_allclose(
            info.dset[:5],
            counts[:5].astype(np.float64) * CONVERSION * CHANNEL_CONVERSION.astype(np.float64) + OFFSET,
            rtol=1e-12,
        )
    finally:
        slicer.close()


# --- Overrides, for files whose own metadata is wrong ------------------------


def test_scale_override_replaces_the_recorded_gain(scaled_nwb_path, counts):
    """Recorded conversion factors are sometimes a library default rather than
    the hardware's -- a reader cannot detect that, so a caller who knows must be
    able to say so."""
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, scale_override=0.1, dejitter=False)
    try:
        got = slicer.get_stream_info("Broadband").dset[:]
        np.testing.assert_allclose(got, counts.astype(np.float32) * np.float32(0.1) + np.float32(OFFSET), rtol=1e-6)
    finally:
        slicer.close()


def test_scale_override_accepts_a_per_stream_mapping(scaled_nwb_path, counts):
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, scale_override={"Broadband": 0.1}, dejitter=False)
    try:
        np.testing.assert_allclose(
            slicer.get_stream_info("Broadband").dset[:],
            counts.astype(np.float32) * np.float32(0.1) + np.float32(OFFSET),
            rtol=1e-6,
        )
        # Unnamed streams keep the file's own scaling.
        assert slicer.get_stream_info("Marker").dset.dtype == np.int16
    finally:
        slicer.close()


def test_unit_override_relabels_without_rescaling(scaled_nwb_path, counts):
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, unit_override="volts", dejitter=False)
    try:
        info = slicer.get_stream_info("Broadband")
        assert info.unit == "volts"
        np.testing.assert_array_equal(info.dset[:], expected(counts))
    finally:
        slicer.close()


# --- Through the iterator ---------------------------------------------------


def test_iterator_emits_scaled_messages(scaled_nwb_path, counts):
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
        chunks = [msg.data for msg in it if msg.data.shape[0]]
    finally:
        it.close()
    got = np.concatenate(chunks, axis=0)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected(counts)[: got.shape[0]])


def test_iterator_default_is_on(scaled_nwb_path):
    """The default is deliberately "apply", not "preserve the old behaviour":
    the failure mode of not applying is silently-wrong units, which no exception
    announces and which downstream absolute thresholds turn into wrong results."""
    assert NWBIteratorSettings(filepath=str(scaled_nwb_path)).apply_conversion is True


# --- ScaledDataset itself ---------------------------------------------------


def test_scaled_dataset_indexing_forms(scaled_nwb_path, counts):
    """Integer indexing drops the time axis; the per-channel gain still has to
    land on the channel axis."""
    slicer = NWBSlicer(scaled_nwb_path, apply_conversion=True, dejitter=False)
    try:
        dset = slicer.get_stream_info("Broadband").dset
        ref = expected(counts)
        np.testing.assert_array_equal(dset[3], ref[3])
        np.testing.assert_array_equal(dset[3:9], ref[3:9])
        np.testing.assert_array_equal(dset[-2:], ref[-2:])
        assert len(dset) == N_SAMPLES
        assert dset.shape == (N_SAMPLES, N_CH)
        assert dset.ndim == 2
    finally:
        slicer.close()


def test_scaled_dataset_rejects_a_mismatched_channel_gain():
    base = np.zeros((10, 4), dtype=np.int16)
    with pytest.raises(ValueError, match="does not match dataset shape"):
        ScaledDataset(base, np.ones(3), 0.0, "float32")
