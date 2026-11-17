"""Gaps in a stream start a new ``AxisArray.segment``, and are looked for on device time."""

import asyncio

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
from ezmsg.util.messages.axisarray import AxisArray, chain_config_id, compute_config_id

from ezmsg.nwb import (
    NWBAxisArrayIterator,
    NWBClockDrivenProducer,
    NWBClockDrivenSettings,
    NWBIteratorSettings,
    NWBIteratorUnit,
    ReferenceClockType,
)
from ezmsg.nwb.clockmodel import detect_group_gaps, gap_threshold
from ezmsg.nwb.scaling import scaling_fingerprint
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
    assert [m.segment for m in msgs] == [0, 0, 1, 1]  # no marker between them


def test_iterator_gap_on_a_chunk_boundary_is_a_segment_without_a_split(gappy_nwb_path):
    _, msgs = _messages(gappy_nwb_path, "Gappy", chunk_dur=1.5)
    assert [m.data.shape[0] for m in msgs] == [150, 150]
    assert [m.segment for m in msgs] == [0, 1]


def assert_segment_rule(msgs):
    """Per stream: segments never decrease, a marker is its data's segment + 1, and no number is
    shared by a marker and data or by two runs of data. Empty messages that do not change the segment
    (idle chunks) are not markers."""
    for key in {m.key for m in msgs}:
        stream = [(m.segment, m.data.shape[0]) for m in msgs if m.key == key]
        segs = [seg for seg, _ in stream]
        assert segs == sorted(segs)
        data_segs = {seg for seg, n in stream if n}
        runs = [seg for i, (seg, n) in enumerate(stream) if n and (i == 0 or stream[i - 1][0] != seg)]
        assert len(runs) == len(set(runs))
        for (prev, n_prev), (seg, n) in zip(stream, stream[1:]):
            if not n and seg != prev:
                assert n_prev and seg == prev + 1 and seg not in data_segs


@pytest.mark.parametrize("chunk_dur", [1.0, 1.5])  # the gap mid-chunk, and on a chunk boundary
def test_iterator_follows_the_segment_rule_and_reopen_continues_upward(gappy_nwb_path, chunk_dur):
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(filepath=gappy_nwb_path, chunk_dur=chunk_dur, reference_clock=ReferenceClockType.UNKNOWN)
    )
    first = list(it)
    it._reset_state()
    second = list(it)
    it.close()
    assert next(m.segment for m in first if m.data.shape[0]) == 0
    assert_segment_rule(first)
    assert_segment_rule(first + second)
    assert [m.segment for m in second] == [m.segment + 3 for m in first]


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


# --- config_id and end of stream ------------------------------------------------


def _expected_config_id(m):
    """The layout's id, chained with the scaling the stream reports, if any."""
    scaling = scaling_fingerprint(m.attrs)
    return compute_config_id(m) if scaling is None else chain_config_id(compute_config_id(m), scaling)


def _published(msgs):
    """What NWBIteratorUnit publishes: an empty message only where it starts a segment of its stream."""
    last, out = {}, []
    for m in msgs:
        if m.data.size or last.get(m.key, m.segment) != m.segment:
            out.append(m)
        last[m.key] = m.segment
    return out


def test_iterator_declares_one_config_id_per_stream(test_nwb_path, monkeypatch):
    import ezmsg.nwb.slicer as slicer_mod

    calls = []
    monkeypatch.setattr(slicer_mod, "compute_config_id", lambda m: calls.append(m) or compute_config_id(m))
    it = NWBAxisArrayIterator(NWBIteratorSettings(filepath=test_nwb_path, reference_clock=ReferenceClockType.UNKNOWN))
    msgs = list(it)
    assert len(calls) == len(it._state.streams) > 1  # once per stream, not per message
    ids = {}
    for m in msgs:
        assert m.config_id != 0 and m.config_id == _expected_config_id(m)
        ids.setdefault(m.key, set()).add(m.config_id)
    assert all(len(v) == 1 for v in ids.values())
    assert len(set.union(*ids.values())) == len(ids)


def _stream_config_id(path, conversion, unit="V", offset=0.0):
    from pynwb import TimeSeries

    series = TimeSeries(
        name="S",
        data=_index_data(100),
        unit=unit,
        conversion=conversion,
        offset=offset,
        starting_time=0.0,
        rate=100.0,
        description="scaled",
    )
    slicer = NWBSlicer(_write_nwb(path, [series], {}), reference_clock=ReferenceClockType.UNKNOWN)
    try:
        return slicer.read_by_index("S", 0, 10).config_id
    finally:
        slicer.close()


def test_config_id_follows_the_scaling_not_just_the_layout(tmp_path):
    base = _stream_config_id(tmp_path / "a.nwb", 0.25)
    assert _stream_config_id(tmp_path / "b.nwb", 0.25) == base
    assert _stream_config_id(tmp_path / "c.nwb", 0.5) != base
    assert _stream_config_id(tmp_path / "d.nwb", 0.25, unit="mV") != base
    assert _stream_config_id(tmp_path / "e.nwb", 0.25, offset=1.0) != base


def test_two_iterators_number_a_file_the_same(gappy_nwb_path):
    runs = [[m.segment for m in _messages(gappy_nwb_path, "Gappy")[1]] for _ in range(2)]
    assert runs[0] == runs[1]


@pytest.mark.parametrize(
    "chunk_dur, sizes",
    [
        (1.0, [100, 50, 50, 100, 0]),  # the gap mid-chunk: split there, no marker between the parts
        (1.5, [150, 150, 0]),  # the gap on a chunk boundary
    ],
)
def test_iterator_gap_is_one_segment_on_and_only_the_end_gets_a_marker(gappy_nwb_path, chunk_dur, sizes):
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(
            filepath=gappy_nwb_path,
            chunk_dur=chunk_dur,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Gappy"],
        )
    )
    msgs = _published(it)
    assert [m.data.shape[0] for m in msgs] == sizes
    eos = [i for i, n in enumerate(sizes) if n == 0]
    for i in eos:
        last, marker = msgs[i - 1], msgs[i]
        assert marker.config_id == last.config_id and marker.segment == last.segment + 1
        gain = last.axes["time"].gain
        assert marker.axes["time"].offset == pytest.approx(last.axes["time"].offset + last.data.shape[0] * gain)
    # Before, after, one marker.
    assert [m.segment for m in msgs if m.data.shape[0] == 0] == [2]
    assert [m.segment for m in msgs if m.data.shape[0]] == [0] * (len(msgs) // 2) + [1] * (len(msgs) // 2)
    assert_segment_rule(msgs)


def test_iterator_unit_publishes_the_markers_and_binning_releases_its_tail(gappy_nwb_path):
    binned = pytest.importorskip("ezmsg.sigproc.binned_aggregate", reason="requires the 'sigproc' extra")
    BinnedAggregate, BinnedAggregateSettings = binned.BinnedAggregate, binned.BinnedAggregateSettings

    unit = NWBIteratorUnit(
        NWBIteratorSettings(
            filepath=gappy_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Gappy"],
            self_terminating=False,
        )
    )
    unit.create_producer()
    binner = BinnedAggregate(BinnedAggregateSettings(bin_duration=0.4))  # 40 samples; each run is 150
    binner.create_processor()

    async def run():
        published, bins = [], []
        async for stream, m in unit.produce():
            if stream is unit.OUTPUT_SIGNAL:
                published.append(m)
                bins += [b async for _, b in binner.on_signal(m) if b.data.size]
        return published, bins

    published, bins = asyncio.run(run())
    assert [m.data.shape[0] for m in published] == [100, 50, 50, 100, 0]
    means = np.concatenate([b.data[:, 0] for b in bins]).tolist()
    # 3 full bins and the 30-sample partial one on each side of the gap; the last only because of the marker.
    assert len(means) == 8
    assert means[-1] == pytest.approx(np.mean(np.arange(270, 300)))


def _clockdriven(path, key, **kwargs):
    return NWBClockDrivenProducer(
        settings=NWBClockDrivenSettings(
            filepath=path, stream_key=key, reference_clock=ReferenceClockType.UNKNOWN, **kwargs
        )
    )


def test_clockdriven_declares_its_config_id_once_per_open(gappy_nwb_path, monkeypatch):
    import ezmsg.nwb.slicer as slicer_mod

    calls = []
    monkeypatch.setattr(slicer_mod, "compute_config_id", lambda m: calls.append(m) or compute_config_id(m))
    producer = _clockdriven(gappy_nwb_path, "Gappy")
    outs = [producer(AxisArray.LinearAxis(gain=0.5, offset=i * 0.5)) for i in range(12)]
    assert len(calls) == 1
    assert {o.config_id for o in outs} == {_expected_config_id(outs[0])}


def test_clockdriven_segments_change_at_the_gap_and_at_the_end(gappy_nwb_path):
    producer = _clockdriven(gappy_nwb_path, "Gappy")
    outs = [producer(AxisArray.LinearAxis(gain=0.5, offset=i * 0.5)) for i in range(12)]
    sizes = [o.data.shape[0] for o in outs]
    # Windows of 0.5 s; the 1 s gap leaves two empty ones before the second run, neither a marker.
    assert sizes == [50, 50, 50, 0, 0, 50, 50, 50, 0, 0, 0, 0]
    segs = [o.segment for o in outs]
    # Empty windows carry the segment before them; only the end gets a marker.
    assert segs == [0] * 5 + [1] * 3 + [2] * 4
    assert_segment_rule(outs)
    for i in (8,):
        assert outs[i].config_id == outs[i - 1].config_id
        assert outs[i].axes["time"].offset == pytest.approx(outs[i - 1].axes["time"].offset + 50 * GAPPY_GAIN)
    # Seeking after the end, or reopening, continues upward.
    producer.seek(0.0)
    sought = [producer(AxisArray.LinearAxis(gain=0.5, offset=i * 0.5)) for i in range(12)]
    assert [o.segment for o in sought] == [s + 3 for s in segs]
    producer._reset_state(AxisArray.LinearAxis(gain=0.5, offset=0.0))
    reopened = [producer(AxisArray.LinearAxis(gain=0.5, offset=i * 0.5)) for i in range(12)]
    assert [o.segment for o in reopened] == [s + 6 for s in segs]
    assert_segment_rule(outs + sought + reopened)
    # A second producer over the same file numbers it the same.
    fresh = _clockdriven(gappy_nwb_path, "Gappy")
    assert [fresh(AxisArray.LinearAxis(gain=0.5, offset=i * 0.5)).segment for i in range(12)] == segs


def test_clockdriven_event_stream_ends_with_one_marker(test_nwb_path):
    producer = _clockdriven(test_nwb_path, "trials")
    outs = [producer(AxisArray.LinearAxis(gain=0.5, offset=i * 0.5)) for i in range(12)]
    events = [o for o in outs if o.data.size]
    assert len(events) == 3
    markers = [i for i, o in enumerate(outs[1:], 1) if o.segment != outs[i - 1].segment]
    assert len(markers) == 1 and outs[markers[0]].data.size == 0
    assert outs[markers[0]].config_id == events[-1].config_id
