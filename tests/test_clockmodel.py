"""Tests for the clock-jitter reconstruction math (pure, file-agnostic)."""

import numpy as np
import pytest

from ezmsg.nwb.clockmodel import (
    ClockModel,
    cache_lookup,
    cache_store,
    find_real_gaps,
    fit_map,
    group_clocks,
    reconstruct_group,
    reconstruct_self,
    reconstruct_shared,
)

RATE = 1000.0
N = 5000
GAIN = 1.0 / RATE


def _smooth_truth(n=N, rate=RATE):
    """Clean, strictly-increasing timestamps with a gentle non-linear drift."""
    idx = np.arange(n)
    drift = 0.003 * np.sin(2 * np.pi * idx / n)  # +/-3 ms, non-linear
    return idx / rate + drift


def _jitter(truth, scale_periods=0.6, seed=0):
    """Add zero-mean per-sample jitter of ``scale_periods`` sample periods."""
    rng = np.random.default_rng(seed)
    return truth + rng.normal(scale=scale_periods * GAIN, size=truth.shape[0])


def _n_gaps(ts, gain=GAIN, gap_tol=0.5):
    return int(np.sum(np.diff(ts) > gain * (1.0 + gap_tol)))


# --- fit_map / ClockModel ---


def test_fit_map_recovers_line():
    x = np.arange(N, dtype=float)
    y = 2.0 * x + 5.0
    model = fit_map(x, y, n_knots=20)
    assert isinstance(model, ClockModel)
    out = model(x)
    assert np.allclose(out, y, atol=1e-6)


def test_fit_map_unknown_method_raises():
    with pytest.raises(ValueError, match="piecewise_linear"):
        fit_map(np.arange(10.0), np.arange(10.0), method="pchip")


def test_fit_map_knots_strictly_increasing():
    model = fit_map(np.arange(N, dtype=float), _jitter(_smooth_truth()), n_knots=40)
    assert np.all(np.diff(model.knots_x) > 0)
    assert np.all(np.diff(model.knots_y) >= 0)


# --- reconstruct_self ---


def test_reconstruct_self_is_monotone_and_gap_free():
    truth = _smooth_truth()
    jittered = _jitter(truth)
    assert _n_gaps(jittered) > 100, "fixture should actually fragment"
    recon = reconstruct_self(jittered)
    assert np.all(np.diff(recon) >= 0), "reconstruction must not run backwards"
    assert _n_gaps(recon) == 0, "reconstruction must eliminate gaps"


def test_reconstruct_self_tracks_truth_including_endpoints():
    """Regression: binned-median knots sit at bin centres; without endpoint
    extrapolation np.interp clamps the outer half-bins flat, mis-timing them by
    up to half a bin (seconds on a long stream)."""
    truth = _smooth_truth()
    recon = reconstruct_self(_jitter(truth))
    err = np.abs(recon - truth)
    # Whole-array error stays at the jitter scale, not the bin scale.
    assert err.max() < 0.005, f"max error {err.max()} too large -- endpoint clamp?"
    # Endpoints specifically must not be flattened.
    assert abs(recon[0] - truth[0]) < 0.002
    assert abs(recon[-1] - truth[-1]) < 0.002


def test_reconstruct_self_length_preserved():
    truth = _smooth_truth()
    assert reconstruct_self(_jitter(truth)).shape == truth.shape


# --- reconstruct_shared ---


def test_reconstruct_shared_rescues_target_from_clean_sibling():
    """A jittery target on a shared device clock is reconstructed via a clean
    sibling's device->dataset conversion, then re-anchored to its own offset."""
    truth = _smooth_truth()
    epoch = 1.7e9
    # Shared device clock; the two hubs differ only by a fixed per-hub latency.
    clean_device = epoch + np.arange(N) / RATE
    clean_dataset = truth  # clean member: dataset == truth
    target_offset = 0.010  # 10 ms fixed latency on the target
    target_device = epoch + np.arange(N) / RATE
    target_dataset = _jitter(truth + target_offset, seed=1)

    recon = reconstruct_shared(target_device, clean_device, clean_dataset, target_dataset=target_dataset)
    assert np.all(np.diff(recon) >= 0)
    assert _n_gaps(recon) == 0
    # Recovers the target's own timeline (truth + its fixed offset) within jitter.
    expected = truth + target_offset
    assert np.abs(recon - expected).max() < 0.005


def test_reconstruct_shared_without_anchor_lands_on_clean_timeline():
    truth = _smooth_truth()
    epoch = 1.7e9
    device = epoch + np.arange(N) / RATE
    recon = reconstruct_shared(device, device, truth)  # no target_dataset anchor
    assert np.abs(recon - truth).max() < 0.005


# --- reconstruct_group ---


def test_reconstruct_group_single_member_is_self_fit():
    truth = _smooth_truth()
    jittered = _jitter(truth)
    out = reconstruct_group([{"key": "A", "device": None, "dataset": jittered}])
    assert set(out) == {"A"}
    assert np.abs(out["A"] - truth).max() < 0.005


def test_reconstruct_group_picks_cleanest_as_model_source():
    truth = _smooth_truth()
    epoch = 1.7e9
    device = epoch + np.arange(N) / RATE
    members = [
        {"key": "dirty", "device": device.copy(), "dataset": _jitter(truth, scale_periods=0.8, seed=2)},
        {"key": "clean", "device": device.copy(), "dataset": _jitter(truth, scale_periods=0.02, seed=3)},
    ]
    out = reconstruct_group(members)
    assert set(out) == {"dirty", "clean"}
    for key in out:
        assert np.all(np.diff(out[key]) >= 0)
        assert _n_gaps(out[key]) == 0
        assert np.abs(out[key] - truth).max() < 0.01


# --- real-gap guard ---


def test_find_real_gaps_detects_true_gap_rejects_jitter():
    truth = _smooth_truth()
    jittered = _jitter(truth)  # jitter but no real gap
    period = float(np.median(np.diff(truth)))
    # A generous threshold well above jitter finds nothing on a jitter-only stream.
    assert find_real_gaps(jittered, period, threshold=0.05).size == 0
    # Insert a genuine 0.5 s gap: every sample after the break shifts later.
    gapped = jittered.copy()
    gapped[2500:] += 0.5
    gaps = find_real_gaps(gapped, period, threshold=0.05)
    assert gaps.size == 1 and gaps[0] == 2499


def test_reconstruct_self_preserves_real_gap():
    truth = _smooth_truth()
    gapped_truth = truth.copy()
    gapped_truth[2500:] += 0.5  # real 0.5 s gap
    recon = reconstruct_self(_jitter(gapped_truth), gap_threshold_s=0.05)
    # The gap survives as a single ~0.5 s jump...
    dts = np.diff(recon)
    big = np.flatnonzero(dts > 0.1)
    assert big.size == 1 and big[0] == 2499
    assert abs(dts[big[0]] - 0.5) < 0.02
    # ...and each side is otherwise smooth (no spurious gaps).
    assert _n_gaps(recon[:2500]) == 0
    assert _n_gaps(recon[2500:]) == 0


def test_reconstruct_self_disable_guard_smooths_over_gap():
    truth = _smooth_truth()
    gapped_truth = truth.copy()
    gapped_truth[2500:] += 0.5
    recon = reconstruct_self(_jitter(gapped_truth), gap_threshold_s=float("inf"))
    # With the guard disabled the jump is smoothed away -- no big step remains.
    assert np.max(np.diff(recon)) < 0.01


def test_reconstruct_shared_preserves_gap_on_device_clock():
    truth = _smooth_truth()
    epoch = 1.7e9
    clean_device = epoch + np.arange(N) / RATE
    # Target shares the clock but drops ~0.3 s of packets at index 3000.
    target_device = epoch + np.arange(N) / RATE
    target_device[3000:] += 0.3
    target_dataset = _jitter(truth, seed=5)
    target_dataset[3000:] += 0.3
    recon = reconstruct_shared(target_device, clean_device, truth, target_dataset=target_dataset, gap_threshold_s=0.05)
    big = np.flatnonzero(np.diff(recon) > 0.1)
    assert big.size == 1 and big[0] == 2999
    assert abs(np.diff(recon)[big[0]] - 0.3) < 0.02


# --- group_clocks ---


def test_group_clocks_groups_shared_and_splits_unrelated():
    epoch = 1.7e9
    a = {"key": "A", "t0": epoch, "tend": epoch + 273.0, "n": 8_000_000}
    b = {"key": "B", "t0": epoch + 0.001, "tend": epoch + 273.0, "n": 8_000_000}  # PTP sibling
    c = {"key": "C", "t0": 0.0, "tend": 273.0, "n": 8_000_000}  # file-relative, different clock
    groups = group_clocks([a, b, c])
    as_sets = sorted((sorted(g) for g in groups), key=len)
    assert ["C"] in as_sets
    assert ["A", "B"] in as_sets


def test_group_clocks_rate_mismatch_splits():
    epoch = 1.7e9
    a = {"key": "A", "t0": epoch, "tend": epoch + 100.0, "n": 100_000}  # 1 kHz
    b = {"key": "B", "t0": epoch, "tend": epoch + 100.0, "n": 3_000_000}  # 30 kHz
    groups = group_clocks([a, b])
    assert sorted(sorted(g) for g in groups) == [["A"], ["B"]]


# --- cache ---


def test_cache_roundtrip(tmp_path, monkeypatch):
    import ezmsg.nwb.clockmodel as cm

    monkeypatch.setattr(cm, "CACHE_DIR", tmp_path / "dejitter")
    sig, key, params = "sig123", "HUB1", ("piecewise_linear", 40)
    arr = np.linspace(0, 1, 100)
    assert cache_lookup(sig, key, params) is None
    cache_store(sig, key, params, arr)
    got = cache_lookup(sig, key, params)
    assert got is not None and np.array_equal(got, arr)
    # A different fit param is a cache miss.
    assert cache_lookup(sig, key, ("piecewise_linear", 20)) is None
