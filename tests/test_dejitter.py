"""Integration tests for read-time dejitter in NWBSlicer / NWBAxisArrayIterator."""

import numpy as np
from conftest import (
    DEJITTER_GAP_AT,
    DEJITTER_GAP_S,
    DEJITTER_RATE,
    dejitter_timestamps,
)

from ezmsg.nwb.iterator import NWBAxisArrayIterator, NWBIteratorSettings
from ezmsg.nwb.slicer import NWBSlicer

GAIN = 1.0 / DEJITTER_RATE


def _n_gaps(ts, gain=GAIN, gap_tol=0.5):
    return int(np.sum(np.diff(ts) > gain * (1.0 + gap_tol)))


def test_raw_stream_actually_fragments(dejitter_nwb_path):
    """Sanity: without dejitter the converted timestamps do fragment."""
    _truth, converted, _device = dejitter_timestamps()
    assert _n_gaps(converted) > 50


def test_dejitter_eliminates_gaps_and_tracks_truth(dejitter_nwb_path):
    truth, _converted, _device = dejitter_timestamps()
    s = NWBSlicer(dejitter_nwb_path, stream_keys=["HUB"], dejitter=True)
    rec = np.asarray(s.get_stream_info("HUB").timestamps[:], dtype=float)
    s.close()
    assert np.all(np.diff(rec) >= 0), "must not run backwards"
    assert _n_gaps(rec) == 0, "must eliminate gaps"
    assert np.abs(rec - truth).max() < 0.01, "must track the clean truth"


def test_dejitter_off_leaves_raw_timestamps(dejitter_nwb_path):
    _truth, converted, _device = dejitter_timestamps()
    s = NWBSlicer(dejitter_nwb_path, stream_keys=["HUB"], dejitter=False)
    rec = np.asarray(s.get_stream_info("HUB").timestamps[:], dtype=float)
    s.close()
    assert np.allclose(rec, converted)


def test_device_partner_hidden_under_filter(dejitter_nwb_path):
    """A filtered read exposes only what was asked for; the auto-loaded
    ``*_device_ts`` partner is pruned after serving the reconstruction."""
    s = NWBSlicer(dejitter_nwb_path, stream_keys=["HUB"], dejitter=True)
    names = s.stream_names
    s.close()
    assert names == ["HUB"]


def test_device_partner_visible_without_filter(dejitter_nwb_path):
    s = NWBSlicer(dejitter_nwb_path, dejitter=True)
    names = set(s.stream_names)
    s.close()
    assert {"HUB", "HUB_device_ts"} <= names


def test_bounds_not_polluted_by_epoch_partner(dejitter_nwb_path):
    """Regression: the partner's absolute-epoch timestamps (plus an epoch-scale
    ts_off and a float32 rate) once inflated stop_time by a whole epoch, blowing
    up the iterator's chunk count. Bounds must stay at the session scale."""
    for keys in (["HUB"], None):
        s = NWBSlicer(dejitter_nwb_path, stream_keys=keys, dejitter=True)
        span = s.stop_time - s.start_time
        s.close()
        assert 2.0 < span < 10.0, f"span {span} not session-scale (keys={keys})"


def test_iterator_collapses_fragments(dejitter_nwb_path):
    def drain(dejitter):
        it = NWBAxisArrayIterator(
            NWBIteratorSettings(filepath=dejitter_nwb_path, chunk_dur=1.0, stream_keys=["HUB"], dejitter=dejitter)
        )
        msgs = list(it)
        it.close()
        return msgs

    off = drain(False)
    on = drain(True)
    # Dejitter removes the intra-chunk gap splits: far fewer messages.
    assert len(on) < len(off)
    # No samples lost either way.
    assert sum(m.data.shape[0] for m in on) == sum(m.data.shape[0] for m in off)


def test_real_gap_preserved_by_default_guard(dejitter_gapped_nwb_path):
    """A genuine data gap survives reconstruction (auto guard on by default):
    the jump remains and the rest is still dejittered."""
    s = NWBSlicer(dejitter_gapped_nwb_path, stream_keys=["HUB"], dejitter=True)
    rec = np.asarray(s.get_stream_info("HUB").timestamps[:], dtype=float)
    s.close()
    dts = np.diff(rec)
    big = np.flatnonzero(dts > 0.1)
    assert big.size == 1 and big[0] == DEJITTER_GAP_AT - 1
    assert abs(dts[big[0]] - DEJITTER_GAP_S) < 0.02
    # Both sides are otherwise gap-free and monotone.
    assert _n_gaps(rec[:DEJITTER_GAP_AT]) == 0
    assert _n_gaps(rec[DEJITTER_GAP_AT:]) == 0
    assert np.all(dts >= 0)


def test_real_gap_smoothed_when_guard_disabled(dejitter_gapped_nwb_path):
    s = NWBSlicer(dejitter_gapped_nwb_path, stream_keys=["HUB"], dejitter=True, real_gap_threshold=float("inf"))
    rec = np.asarray(s.get_stream_info("HUB").timestamps[:], dtype=float)
    s.close()
    # Guard off -> the real gap is smoothed away; no large jump remains.
    assert np.max(np.diff(rec)) < 0.05


def test_real_gap_splits_iterator_chunk(dejitter_gapped_nwb_path):
    """With the gap preserved the iterator still emits a boundary there."""
    it = NWBAxisArrayIterator(
        NWBIteratorSettings(filepath=dejitter_gapped_nwb_path, chunk_dur=10.0, stream_keys=["HUB"], dejitter=True)
    )
    msgs = list(it)
    it.close()
    # chunk_dur exceeds the record, so absent a gap this would be one message;
    # the preserved gap forces a split into two contiguous runs.
    assert len(msgs) == 2
    assert sum(m.data.shape[0] for m in msgs) == 3000


def test_dejitter_cache_reused_across_opens(dejitter_nwb_path, tmp_path, monkeypatch):
    """Second open of the same file loads reconstructed timestamps from cache."""
    import ezmsg.nwb.clockmodel as cm

    # The slicer's cache_lookup/cache_store are cm's functions, which read
    # cm.CACHE_DIR at call time -- so redirecting the dir is enough to isolate.
    monkeypatch.setattr(cm, "CACHE_DIR", tmp_path / "dejitter")

    s1 = NWBSlicer(dejitter_nwb_path, stream_keys=["HUB"], dejitter=True)
    rec1 = np.asarray(s1.get_stream_info("HUB").timestamps[:], dtype=float)
    s1.close()
    cache_files = list((tmp_path / "dejitter").glob("*.npy"))
    assert cache_files, "reconstruction should have been cached"

    s2 = NWBSlicer(dejitter_nwb_path, stream_keys=["HUB"], dejitter=True)
    rec2 = np.asarray(s2.get_stream_info("HUB").timestamps[:], dtype=float)
    s2.close()
    assert np.array_equal(rec1, rec2)
