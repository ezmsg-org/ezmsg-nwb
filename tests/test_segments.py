"""Gaps in a stream start a new ``AxisArray.segment``, and are looked for on device time."""

import numpy as np
import pytest
from conftest import (
    DEJITTER_EPOCH,
    DEJITTER_N,
    DEJITTER_RATE,
    GAPPY_GAIN,
    GAPPY_RATE,
    _index_data,
    _write_dejitter_nwb,
    _write_nwb,
)

from ezmsg.nwb import NWBAxisArrayIterator, NWBIteratorSettings, ReferenceClockType
from ezmsg.nwb.clockmodel import detect_group_gaps, gap_threshold
from ezmsg.nwb.slicer import NWBSlicer

STEP_AT = 1000  # the corrected clock steps here; no data was lost
GAP_AT = 2000  # the device stopped here for GAP_S
GAP_S = 0.3


def _stepped_timestamps():
    """A clean device clock with one real gap, and a corrected clock that also steps elsewhere."""
    truth = np.arange(DEJITTER_N) / DEJITTER_RATE
    truth[GAP_AT:] += GAP_S
    converted = truth.copy()
    converted[STEP_AT:] += 0.05
    return converted, DEJITTER_EPOCH + truth


@pytest.fixture(scope="module")
def stepped_nwb_path(tmp_path_factory):
    converted, device = _stepped_timestamps()
    path = tmp_path_factory.mktemp("stepped") / "stepped.nwb"
    return _write_dejitter_nwb(path, converted, device, _index_data(DEJITTER_N, 4))


@pytest.fixture(scope="module")
def late_stamp_nwb_path(tmp_path_factory):
    """An unpaired 100 Hz stream with one timestamp 50 ms late and nothing missing."""
    from pynwb import TimeSeries

    ts = np.arange(300) * GAPPY_GAIN
    ts[100] += 0.05
    series = TimeSeries(name="Late", data=_index_data(300), unit="V", timestamps=ts, description="late stamp")
    return _write_nwb(tmp_path_factory.mktemp("late") / "late.nwb", [series], {"Late": GAPPY_RATE})


def _messages(path, key, **settings):
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(filepath=path, reference_clock=ReferenceClockType.UNKNOWN, stream_keys=[key], **settings)
    )
    msgs = [m for m in it if m.data.shape[0] > 0]
    return it, msgs


def test_gap_threshold_is_never_under_two_periods():
    assert gap_threshold(1 / 30000, 0.010) == pytest.approx(1 / 30000 + 0.010)
    assert gap_threshold(0.0125, 0.010) == pytest.approx(0.025)  # 80 Hz: one period beats 10 ms


def test_group_gaps_come_from_device_time():
    converted, device = _stepped_timestamps()
    member = {"key": "HUB", "dataset": converted, "device": device, "period": 1 / DEJITTER_RATE}
    assert detect_group_gaps([member])["HUB"].tolist() == [GAP_AT - 1]
    # Without device time the corrected clock's own step is all there is to go on.
    assert detect_group_gaps([{**member, "device": None}])["HUB"].tolist() == [STEP_AT - 1, GAP_AT - 1]


def test_iterator_segment_changes_at_a_gap(gappy_nwb_path):
    _, msgs = _messages(gappy_nwb_path, "Gappy", chunk_dur=1.0)
    assert [m.data.shape[0] for m in msgs] == [100, 50, 50, 100]
    assert [m.segment for m in msgs] == [0, 0, 1, 1]


def test_iterator_gap_on_a_chunk_boundary_is_a_segment_without_a_split(gappy_nwb_path):
    _, msgs = _messages(gappy_nwb_path, "Gappy", chunk_dur=1.5)
    assert [m.data.shape[0] for m in msgs] == [150, 150]
    assert [m.segment for m in msgs] == [0, 1]


def test_iterator_reopen_does_not_repeat_the_segment(gappy_nwb_path):
    it, msgs = _messages(gappy_nwb_path, "Gappy", chunk_dur=1.0)
    it._reset_state()
    assert next(it).segment == msgs[-1].segment + 1
    it.close()


def test_iterator_gap_tol_is_deprecated(gappy_nwb_path):
    with pytest.warns(DeprecationWarning, match="gap_tolerance"):
        NWBIteratorSettings(filepath=gappy_nwb_path, gap_tol=0.5)


@pytest.mark.parametrize("dejitter", [True, False])
def test_paired_stream_splits_where_the_device_clock_has_a_gap(stepped_nwb_path, dejitter):
    """Not where the corrected clock steps, whether or not it is dejittered."""
    _, msgs = _messages(stepped_nwb_path, "HUB", chunk_dur=10.0, dejitter=dejitter, dejitter_cache=False)
    assert [m.data.shape[0] for m in msgs] == [GAP_AT, DEJITTER_N - GAP_AT]
    assert [m.segment for m in msgs] == [0, 1]


def test_a_lone_late_timestamp_is_not_a_gap(late_stamp_nwb_path):
    slicer = NWBSlicer(late_stamp_nwb_path, reference_clock=ReferenceClockType.UNKNOWN)
    try:
        assert slicer.segment_starts("Late", 0, 300).size == 0
    finally:
        slicer.close()


def test_segment_starts_sees_a_gap_from_any_chunk(gappy_nwb_path):
    """The chunk-by-chunk search reads a little either side, so the answer does not depend on the chunking."""
    slicer = NWBSlicer(gappy_nwb_path, reference_clock=ReferenceClockType.UNKNOWN)
    try:
        assert slicer.segment_starts("Gappy", 0, 300).tolist() == [150]
        assert slicer.segment_starts("Gappy", 150, 151).tolist() == [150]
        assert slicer.segment_starts("Gappy", 149, 150).size == 0
        assert slicer.segment_starts("Gappy", 151, 300).size == 0
    finally:
        slicer.close()
