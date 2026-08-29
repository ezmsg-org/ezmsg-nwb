"""Shortcuts taken at open, and the properties that make them safe.

Both are pure speedups: they must not move a single value. What is pinned here
is equivalence with the straightforward computation, not the speed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ezmsg.nwb.slicer import MEDIAN_SUBSAMPLE_MAX, _fast_median, infer_nominal_rate


def quantised_intervals(n: int, period: float = 1 / 30000, grid: float = 4e-8, seed: int = 3) -> np.ndarray:
    """Intervals as a real acquisition delivers them: jittered, then snapped to
    the device's timestamp grid. The snapping is what makes a subsampled median
    land on exactly the same value as a full one."""
    rng = np.random.default_rng(seed)
    return np.round((period + rng.normal(0, period * 0.02, n)) / grid) * grid


@pytest.mark.parametrize("n", [10, MEDIAN_SUBSAMPLE_MAX, MEDIAN_SUBSAMPLE_MAX + 1, 3_000_000])
def test_fast_median_matches_the_real_one(n):
    dts = quantised_intervals(n)
    assert _fast_median(dts) == float(np.median(dts))


def test_small_arrays_are_not_subsampled():
    """Below the threshold there is nothing to gain, so take the exact answer."""
    dts = np.linspace(1.0, 2.0, 1001)  # even-length medians interpolate; no grid
    assert _fast_median(dts) == float(np.median(dts))


def test_inferred_rate_is_unchanged_by_subsampling():
    """The median only centres the trim window; the returned estimate is a mean
    over what survives it. So the median may be approximate, and here it is not
    even that."""
    dts = quantised_intervals(3_000_000)
    small = dts[:1000]
    assert infer_nominal_rate(small) == pytest.approx(1 / np.mean(small), rel=1e-9)
    assert infer_nominal_rate(dts) == pytest.approx(30000.0, rel=1e-3)


def test_gaps_still_do_not_drag_the_estimate():
    """The trim is the whole reason the mean is usable. Subsampling the median
    must not weaken it: a stream with real gaps still reports its sample rate,
    not an average that includes the holes."""
    dts = quantised_intervals(2_000_000)
    dts[::5000] = 0.01  # 400 dropped-packet holes, 300x the sample period
    assert infer_nominal_rate(dts) == pytest.approx(30000.0, rel=1e-3)


def test_ch_labels_come_from_the_electrodes_region(scaled_nwb_path):
    """Reading the label column directly must still honour the region: a series
    referencing a subset of the table gets that subset's labels, in order."""
    from ezmsg.nwb import NWBSlicer

    slicer = NWBSlicer(scaled_nwb_path, dejitter=False)
    try:
        labels = slicer.get_stream_info("Broadband").template.axes["ch"].data
    finally:
        slicer.close()
    assert list(labels) == ["elec0", "elec1", "elec2", "elec3"]
    assert labels.dtype.kind == "U"
