"""Clock-jitter reconstruction for NWB timestamps.

Some acquisition streams carry per-sample timestamps that are correct in the
*aggregate* (averaged over enough samples they give the true sample rate -- note
the median does not, when stamps are quantised; see
:func:`~ezmsg.nwb.slicer.infer_nominal_rate`) but jittery sample-to-sample: the
recording pipeline stamps each sample at arrival time
rather than from the ADC clock, so neighbouring intervals swing by roughly a
whole sample period and occasionally hop backwards. The iterator's gap-splitter
then shatters such a stream into tens of thousands of tiny gap-free runs -- one
recorded file here fragments an 8.2 M-sample stream into ~285 k messages.

The fix is to replace those timestamps with a *smoothed, strictly monotone*
version that preserves the true (slowly drifting, non-linear) clock while
discarding the jitter. Two regimes:

* **Self-fit** -- fit a monotone piecewise-linear curve of ``dataset_time`` vs
  ``sample_index`` and evaluate it at every index. Removes jitter and gaps on a
  single stream with no help from siblings.

* **Shared-clock (PTP group)** -- when several streams share one device clock
  (e.g. two PTP-synchronised acquisition hubs), the *device -> dataset* clock
  conversion is one smooth function common to all of them. Estimate that
  conversion ``C`` from the cleanest member, clean the target's device time from
  its own sample index, then map through ``C``. A jittery stream is thereby
  rescued by a clean sibling; only the target's fixed per-stream latency is
  taken from the target itself (via a robust median anchor).

Everything here is pure numpy and file-agnostic so it can be driven both at read
time (see :class:`~ezmsg.nwb.slicer.NWBSlicer`) and by an offline tool that
rewrites corrected timestamps into an NWB file. The fit family is pluggable via
``method``; ``"piecewise_linear"`` is the only one implemented today and needs
no dependency beyond numpy. A monotone cubic (PCHIP) variant can drop in later
behind an optional scipy extra without changing callers.
"""

from __future__ import annotations

import hashlib
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .util import interval_median

DEFAULT_DEJITTER_KNOTS = 40
"""Number of knots for the piecewise-linear fit. ~40 over a multi-million-sample
stream puts a knot every few seconds -- fine enough to track sub-millisecond
clock drift, coarse enough that each knot's binned median is rock-steady."""

DEFAULT_DEJITTER_METHOD = "piecewise_linear"

# --- Real-gap guard --------------------------------------------------------
# Dejittering smooths per-sample noise, which would also erase a *genuine* data
# gap (dropped packets -> real elapsed time with no samples). The guard detects
# such gaps and reconstructs each inter-gap segment independently so the true
# jump survives -- the iterator then still splits a chunk there. Detection is on
# the reliable timebase (device time when available), thresholded above the
# stream's own jitter envelope so ordinary jitter never trips it.
GAP_ENVELOPE_PCT = 99.9
"""Percentile of the interval deviation used as the jitter-envelope estimate.
High enough to sit at the top of the jitter distribution yet, for any stream
with <0.1% real gaps, still below the gaps themselves -- so the envelope tracks
jitter, not gaps, regardless of sample count."""
GAP_ENVELOPE_MULT = 50.0
"""Auto threshold = ``period + this * envelope``. Deliberately large: sample
timestamping jitter is heavy-tailed (on real hardware the worst interval can be
tens of times the 99.9th percentile), and a small persistent clock-sync *step*
is not a data gap -- it should be smoothed, not preserved. Only a jump well
beyond that tail is treated as genuine data loss. Tune via ``real_gap_threshold``
when you know a file's real gap sizes."""
GAP_MIN_S = 0.005
"""Floor for the auto threshold's margin above one sample period (s), so a
near-perfect stream doesn't get a threshold so tight that sub-sample noise reads
as a gap."""
GAP_VERIFY_WIN = 16
"""Half-window (samples) for the net-displacement check that separates a real
gap (a sustained level shift) from a lone jitter spike (self-cancelling)."""

SHARED_MODEL_JITTER_RATIO = 5.0
"""A group member is reconstructed via the shared clock model only when its own
jitter exceeds the cleanest member's by at least this factor. Self-fitting an
already-clean stream tracks its own timestamps to sub-sample accuracy, whereas
the shared model imposes a sibling's device->dataset conversion whose slightly
different effective rate compounds over the recording (measured ~6.6 ms of drift
across 51 s on a clean stream that self-fit to 0.08 ms). So the sibling's clock
is borrowed only when the target is genuinely too corrupted to trust its own."""

SHARED_MODEL_JITTER_FLOOR_PERIODS = 1.0
"""Absolute jitter floor for the shared model, in sample periods: a target must
also be jittier than this before a sibling's clock is borrowed.

Serves two purposes. It keeps the ratio test meaningful when the cleanest member
is *perfect* -- a stream whose stamps were regenerated as ``i / fs`` self-fits to
~1e-14 s (or exactly zero), and a pure ratio test against that either rescues
everything or, with a ``> 0`` guard, rescues nothing precisely when the reference
is most trustworthy. And it stops a large *relative* difference between two
already-clean streams from triggering a rescue that cannot help: sub-sample
jitter is not corruption, whatever its ratio to a cleaner sibling.

One period separates the observed populations with room to spare: clean
recordings self-fit to 0.3-0.5 periods, while a stream whose stamps were written
at the wrong intra-chunk rate lands at ~37."""

CACHE_DIR = Path("~").expanduser() / ".ezmsg" / "nwb-cache" / "dejitter"
"""Where reconstructed timestamp vectors are cached across file opens, alongside
remfile's ``nwb-cache``. Multi-pass workloads (e.g. training over many epochs of
the same files) reconstruct once and reload thereafter."""


# --- Monotone map ---------------------------------------------------------


def _binned_knots(x: np.ndarray, y: np.ndarray, n_knots: int) -> tuple[np.ndarray, np.ndarray]:
    """Knot ``(x, y)`` pairs: the per-bin medians over ``n_knots`` index bins.

    ``x`` and ``y`` are paired samples ordered by acquisition index; ``x`` is a
    monotone-trend quantity (sample index, or a device/dataset clock). Binning
    by index position and taking the median in each bin gives a knot pair that
    is robust to per-sample jitter and to the rare backward hop. The returned
    ``knots_x`` is forced strictly increasing (ties dropped) so it is a valid
    ``np.interp`` abscissa, and ``knots_y`` is clamped non-decreasing so the
    resulting map cannot run backwards -- a guard that only bites on a knot
    inversion, which a stable median over hundreds of thousands of samples per
    bin does not produce in practice.

    A binned median sits at the *centre* of its bin, so the first and last knots
    leave the outer half-bin uncovered; ``np.interp`` would clamp everything
    past them flat, mis-timing those samples by up to half a bin. To avoid that,
    the end segments are linearly extrapolated out to the true domain edges
    (``min``/``max`` of ``x``) and added as boundary knots -- accurate because
    the underlying clock drift is smooth and near-linear over a single bin.
    """
    n = x.shape[0]
    if n == 0:
        return np.empty(0), np.empty(0)
    edges = np.linspace(0, n, n_knots + 1).astype(int)
    kx_list: list[float] = []
    ky_list: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        kx_list.append(float(np.median(x[a:b])))
        ky_list.append(float(np.median(y[a:b])))
    kx = np.asarray(kx_list, dtype=float)
    ky = np.asarray(ky_list, dtype=float)
    if kx.shape[0] == 0:
        return kx, ky
    # Drop non-increasing x so np.interp has a valid abscissa.
    keep = np.concatenate(([True], np.diff(kx) > 0))
    kx, ky = kx[keep], ky[keep]

    # Extend the end segments to the true domain edges so the outer half-bins
    # aren't clamped flat by np.interp.
    if kx.shape[0] >= 2:
        x_lo = float(np.min(x))
        x_hi = float(np.max(x))
        if x_lo < kx[0]:
            slope = (ky[1] - ky[0]) / (kx[1] - kx[0])
            y_lo = ky[0] + slope * (x_lo - kx[0])
            kx = np.concatenate(([x_lo], kx))
            ky = np.concatenate(([y_lo], ky))
        if x_hi > kx[-1]:
            slope = (ky[-1] - ky[-2]) / (kx[-1] - kx[-2])
            y_hi = ky[-1] + slope * (x_hi - kx[-1])
            kx = np.concatenate((kx, [x_hi]))
            ky = np.concatenate((ky, [y_hi]))

    # Guarantee a non-decreasing map even if binned medians ever invert.
    ky = np.maximum.accumulate(ky)
    return kx, ky


@dataclass
class ClockModel:
    """A monotone map from one time base to another, evaluated by interpolation.

    ``knots_x`` must be strictly increasing (see :func:`_binned_knots`). Inputs
    outside the knot range clamp to the endpoint values -- appropriate here,
    where the domain is always the fitted stream's own span.
    """

    knots_x: np.ndarray
    knots_y: np.ndarray
    method: str = DEFAULT_DEJITTER_METHOD

    def __call__(self, x: typing.Any) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), self.knots_x, self.knots_y)


def fit_map(
    x: typing.Any,
    y: typing.Any,
    n_knots: int = DEFAULT_DEJITTER_KNOTS,
    method: str = DEFAULT_DEJITTER_METHOD,
) -> ClockModel:
    """Fit a monotone :class:`ClockModel` mapping ``x -> y``.

    ``x``/``y`` are index-ordered paired samples. Only ``"piecewise_linear"`` is
    implemented; other names raise so a typo fails loudly rather than silently
    using the default.
    """
    if method != "piecewise_linear":
        raise ValueError(f"Unknown dejitter method {method!r}; only 'piecewise_linear' is implemented.")
    kx, ky = _binned_knots(np.asarray(x, dtype=float), np.asarray(y, dtype=float), n_knots)
    return ClockModel(knots_x=kx, knots_y=ky, method=method)


# --- Real-gap detection ---------------------------------------------------


def _nominal_period(times: np.ndarray) -> float:
    """Robust sample period: the median inter-sample interval (gaps are rare, so
    the median is unmoved by them)."""
    if times.shape[0] < 2:
        return 0.0
    return interval_median(np.diff(times))


def _auto_gap_threshold(times: np.ndarray) -> float:
    """Per-stream real-gap threshold (an absolute inter-sample interval, s).

    Built from the interval statistics, not a smooth-fit residual: a real gap is
    a rare outlier, so a high percentile of the interval deviation tracks the
    jitter envelope while excluding the gaps themselves, whereas a fit residual
    smears each gap across a bin and would push the threshold above the very gap
    it should catch. Threshold =
    ``median_interval + max(GAP_MIN_S, GAP_ENVELOPE_MULT * envelope)``.
    """
    dt = np.diff(np.asarray(times, dtype=float))
    if dt.size == 0:
        return np.inf
    period = interval_median(dt)
    # The percentile stays on the full array. It is the one statistic here that a
    # subsample genuinely moves: the 99.9th percentile of 200k values rests on
    # their top 200 rather than on 7000, which shifted this threshold by 1.4% on
    # a real stream. The threshold decides which jumps count as real gaps, so a
    # shift there re-segments the stream and re-times its samples -- a behaviour
    # change, not a speedup.
    envelope = float(np.percentile(np.abs(dt - period), GAP_ENVELOPE_PCT))
    return period + max(GAP_MIN_S, GAP_ENVELOPE_MULT * envelope)


def find_real_gaps(
    times: typing.Any,
    period: float,
    threshold: float,
    win: int = GAP_VERIFY_WIN,
) -> np.ndarray:
    """Left-edge indices of genuine data gaps in ``times``.

    A candidate is any interval exceeding ``threshold`` seconds. Each is then
    verified as a *sustained* level shift, not a lone jitter spike: the mean
    detrended level (``time - period * index``) over a window just after the jump
    must sit at least half a threshold above the window just before. A jitter
    spike self-cancels (its neighbours compensate), so its before/after levels
    match and it is rejected; a real gap shifts every later sample and survives.
    """
    t = np.asarray(times, dtype=float)
    n = t.shape[0]
    if n < 2 or threshold <= 0.0 or not np.isfinite(threshold):
        return np.empty(0, dtype=int)
    candidates = np.flatnonzero(np.diff(t) > threshold)
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    real: list[int] = []
    for i in candidates:
        lo = max(0, i - win)
        hi = min(n, i + 1 + win)
        pre_idx = np.arange(lo, i + 1)
        post_idx = np.arange(i + 1, hi)
        level_pre = float(np.mean(t[lo : i + 1] - period * pre_idx))
        level_post = float(np.mean(t[i + 1 : hi] - period * post_idx))
        if level_post - level_pre > 0.5 * threshold:
            real.append(int(i))
    return np.asarray(real, dtype=int)


def _segments(n: int, gaps: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` segments split at each gap's left edge."""
    bounds = [0, *(gaps + 1).tolist(), n]
    return list(zip(bounds[:-1], bounds[1:]))


def _smooth_eval(x: np.ndarray, y: np.ndarray, n_knots: int, method: str) -> np.ndarray:
    """Fit ``x -> y`` and evaluate at ``x`` -- one smooth pass over a segment."""
    return fit_map(x, y, n_knots, method)(x)


def _enforce_join_monotonicity(
    out: np.ndarray, segments: list[tuple[int, int]], period: float, raw: np.ndarray
) -> np.ndarray:
    """Shift whole segments forward so no segment boundary runs backwards.

    Each segment is fit independently so a real gap survives, which leaves the
    *joins* unconstrained: a segment's smoothed first sample is an estimate, and
    in the shared-model path each segment also carries its own median anchor, so
    segment ``k+1`` can begin before segment ``k`` ended. Observed on real data
    at 33 sample periods (1.1 ms) of backward step -- far past rounding, and a
    contract violation for callers (the gap-splitter and chunk anchoring both
    read these timestamps as monotone).

    A broken join is re-cut to the jump the *raw* timestamps show across it
    (floored at one period), and the whole following segment is shifted to match,
    cumulatively. Using the raw jump is what keeps a real gap intact: shifting by
    just enough to restore order would collapse the gap to a single period, and
    the raw stamps are the only remaining evidence of how long the recording
    actually stopped. Healthy joins are left alone.

    Shifting rather than clamping is what preserves the fit: a segment keeps its
    internal shape and its duration. ``np.maximum.accumulate`` would instead
    flatten the overlap into a plateau of duplicate stamps, discarding the
    segment's own timing.

    Shifts are cumulative, so a late sample can move by the sum of all earlier
    corrections. That is the honest cost, and it is paid only where the
    reconstruction had already gone wrong.
    """
    if len(segments) < 2:
        return out
    shift = 0.0
    for a, b in segments[1:]:
        if shift:
            out[a:b] += shift
        jump = out[a] - out[a - 1]
        if jump < period:
            want = max(float(raw[a] - raw[a - 1]), period)
            extra = want - jump
            out[a:b] += extra
            shift += extra
    return out


# --- Reconstruction -------------------------------------------------------


def reconstruct_self(
    dataset_times: typing.Any,
    n_knots: int = DEFAULT_DEJITTER_KNOTS,
    method: str = DEFAULT_DEJITTER_METHOD,
    gap_threshold_s: float | None = None,
) -> np.ndarray:
    """Dejitter a stream from its own timestamps: fit ``index -> dataset_time``.

    Returns a monotone, jitter-free timestamp vector the same length as
    ``dataset_times`` that tracks the stream's slow clock drift. Genuine data
    gaps are preserved: the series is split at gaps detected on its own
    timestamps (see :func:`find_real_gaps`; ``gap_threshold_s`` is the jump
    threshold in seconds, or ``None`` to auto-derive per stream), and each
    segment is fit independently so the true jump between segments survives.
    Used when a stream has no clean sibling to borrow a clock model from.
    """
    ds = np.asarray(dataset_times, dtype=float)
    n = ds.shape[0]
    idx = np.arange(n, dtype=float)
    base = _smooth_eval(idx, ds, n_knots, method)
    gaps = _detect_gaps(ds, gap_threshold_s)
    if gaps.size == 0:
        return base
    out = base.copy()
    segs = _segments(n, gaps)
    for a, b in segs:
        if b - a >= 2:
            out[a:b] = _smooth_eval(np.arange(b - a, dtype=float), ds[a:b], n_knots, method)
        else:
            out[a:b] = ds[a:b]
    return _enforce_join_monotonicity(out, segs, _nominal_period(ds), ds)


def reconstruct_shared(
    target_device: typing.Any,
    clean_device: typing.Any,
    clean_dataset: typing.Any,
    target_dataset: typing.Any = None,
    n_knots: int = DEFAULT_DEJITTER_KNOTS,
    method: str = DEFAULT_DEJITTER_METHOD,
    gap_threshold_s: float | None = None,
) -> np.ndarray:
    """Reconstruct a target stream's dataset timestamps via a shared clock.

    ``clean_device -> clean_dataset`` defines the smooth device->dataset clock
    conversion ``C`` (estimated from the cleanest group member). The target's
    own device time is cleaned from its sample index (``index -> device``) and
    pushed through ``C``. When ``target_dataset`` is supplied, a robust median
    offset re-anchors the result onto the target's own timeline, preserving the
    target's fixed per-stream latency while still borrowing the drift *shape*
    from the clean sibling.

    Genuine gaps in the target's device clock are preserved: the stream is split
    at gaps detected on ``target_device`` and each segment is cleaned and
    re-anchored on its own, so the true jump survives while jitter is smoothed.
    ``C`` stays global -- it is continuous over device time regardless of gaps.

    Returns a monotone vector the length of ``target_device``.
    """
    c_map = fit_map(clean_device, clean_dataset, n_knots, method)
    dev = np.asarray(target_device, dtype=float)
    tgt = None if target_dataset is None else np.asarray(target_dataset, dtype=float)
    n = dev.shape[0]
    idx = np.arange(n, dtype=float)
    dev_base = _smooth_eval(idx, dev, n_knots, method)

    def through_c(dev_clean: np.ndarray, tgt_seg: np.ndarray | None) -> np.ndarray:
        r = c_map(dev_clean)
        if tgt_seg is not None:
            r = r + float(np.median(tgt_seg - r))
        return r

    gaps = _detect_gaps(dev, gap_threshold_s)
    if gaps.size == 0:
        return through_c(dev_base, tgt)
    out = np.empty(n, dtype=float)
    segs = _segments(n, gaps)
    for a, b in segs:
        dev_clean = _smooth_eval(np.arange(b - a, dtype=float), dev[a:b], n_knots, method) if b - a >= 2 else dev[a:b]
        out[a:b] = through_c(dev_clean, None if tgt is None else tgt[a:b])
    # Each segment carries its own median anchor, so joins are doubly
    # unconstrained here -- this is the path where the 33-period backward step
    # was observed.
    return _enforce_join_monotonicity(out, segs, _nominal_period(dev), dev)


def _detect_gaps(times: np.ndarray, gap_threshold_s: float | None) -> np.ndarray:
    """Real-gap left edges in ``times``, thresholded against its own jitter."""
    period = _nominal_period(times)
    threshold = gap_threshold_s if gap_threshold_s is not None else _auto_gap_threshold(times)
    return find_real_gaps(times, period, threshold)


def _self_fit_resid_std(dataset_times: np.ndarray, n_knots: int, method: str) -> float:
    """Std of a stream's timestamps about its own self-fit -- a jitter score."""
    recon = reconstruct_self(dataset_times, n_knots, method)
    return float(np.std(np.asarray(dataset_times, dtype=float) - recon))


def reconstruct_group(
    members: list[dict],
    n_knots: int = DEFAULT_DEJITTER_KNOTS,
    method: str = DEFAULT_DEJITTER_METHOD,
    gap_threshold_s: float | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct every member of a shared-clock group.

    ``members`` is a list of ``{"key", "device", "dataset"}`` dicts (``device``
    may be ``None`` for a lone stream). The cleanest member -- least jitter about
    its own self-fit -- is dejittered directly and can supply the clock model
    ``C`` for the rest. A member is reconstructed via that shared model only when
    it carries device timestamps and is jitterier than both
    :data:`SHARED_MODEL_JITTER_RATIO` times the clean source and
    :data:`SHARED_MODEL_JITTER_FLOOR_PERIODS` sample periods; otherwise it is
    self-fit, which never imposes a sibling's clock. A single-member group falls
    back to a self-fit. Genuine data gaps are preserved per member
    (``gap_threshold_s`` forwarded; ``None`` = auto), and segment joins are kept
    monotone (:func:`_enforce_join_monotonicity`). Returns
    ``{key: reconstructed_dataset_times}``.

    This is the file-agnostic entry point shared by the read-time slicer and any
    offline NWB-correcting tool.
    """
    if not members:
        return {}
    if len(members) == 1:
        m = members[0]
        return {m["key"]: reconstruct_self(m["dataset"], n_knots, method, gap_threshold_s)}

    resids = {m["key"]: _self_fit_resid_std(np.asarray(m["dataset"], dtype=float), n_knots, method) for m in members}
    clean_key = min(resids, key=resids.get)
    clean = next(m for m in members if m["key"] == clean_key)
    clean_recon = reconstruct_self(clean["dataset"], n_knots, method, gap_threshold_s)
    clean_resid = resids[clean_key]

    out: dict[str, np.ndarray] = {clean_key: clean_recon}
    for m in members:
        if m["key"] == clean_key:
            continue
        # Borrow the clean sibling's clock only for a genuinely corrupted target;
        # an already-clean stream self-fits to sub-sample accuracy and the shared
        # model would only inject the sibling's slightly-different rate. The
        # target must clear both bars -- jitterier than the source by the ratio,
        # AND jittery in absolute terms -- so a perfect source (ratio threshold
        # 0) still rescues a corrupted sibling, while two clean streams never
        # trigger each other however far apart their ratio happens to fall.
        threshold = max(
            SHARED_MODEL_JITTER_RATIO * clean_resid,
            SHARED_MODEL_JITTER_FLOOR_PERIODS * _nominal_period(np.asarray(m["dataset"], dtype=float)),
        )
        rescue = m["device"] is not None and resids[m["key"]] >= threshold
        if rescue:
            out[m["key"]] = reconstruct_shared(
                target_device=m["device"],
                clean_device=clean["device"],
                clean_dataset=clean_recon,
                target_dataset=m["dataset"],
                n_knots=n_knots,
                method=method,
                gap_threshold_s=gap_threshold_s,
            )
        else:
            out[m["key"]] = reconstruct_self(m["dataset"], n_knots, method, gap_threshold_s)
    return out


# --- Clock-group detection ------------------------------------------------


def group_clocks(members: list[dict], center_tol: float = 1.0, rate_tol: float = 0.01) -> list[list[str]]:
    """Partition streams into groups that share a device clock.

    Each member is ``{"key", "t0", "tend", "n"}`` -- only the device-time
    endpoints and sample count, so the caller can probe them cheaply (two scalar
    reads per stream) without materialising millions of timestamps.

    Two streams are judged to share a clock when their device-time *spans
    overlap*, their nominal rates agree to within ``rate_tol`` (fractional), and
    their span *centres* agree to within ``center_tol`` seconds. PTP-synchronised
    devices land within a millisecond; an unrelated clock (a different epoch, or
    a file-relative axis near zero) is off by seconds or more, so this separates
    cleanly while staying conservative -- a missed grouping merely falls back to
    per-stream self-fit, which is always safe.
    """
    keys = [m["key"] for m in members]
    parent = {k: k for k in keys}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    def agree(u: dict, v: dict) -> bool:
        lo = max(u["t0"], v["t0"])
        hi = min(u["tend"], v["tend"])
        if hi <= lo:
            return False
        span_u = u["tend"] - u["t0"]
        span_v = v["tend"] - v["t0"]
        if span_u <= 0 or span_v <= 0:
            return False
        rate_u = (u["n"] - 1) / span_u
        rate_v = (v["n"] - 1) / span_v
        if abs(rate_u - rate_v) > rate_tol * max(rate_u, rate_v):
            return False
        center_u = 0.5 * (u["t0"] + u["tend"])
        center_v = 0.5 * (v["t0"] + v["tend"])
        return abs(center_u - center_v) <= center_tol

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if agree(members[i], members[j]):
                union(members[i]["key"], members[j]["key"])

    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


# --- Disk cache -----------------------------------------------------------


def cache_lookup(signature: str, key: str, params: tuple) -> np.ndarray | None:
    """Return a cached reconstruction, or ``None`` on miss / unreadable cache.

    ``signature`` identifies the source file (path + size + mtime); ``key`` the
    stream; ``params`` the fit parameters. Any read error degrades to a miss so
    a corrupt cache entry never breaks a load.
    """
    path = _cache_path(signature, key, params)
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def cache_store(signature: str, key: str, params: tuple, arr: np.ndarray) -> None:
    """Persist a reconstruction. Failures are swallowed -- caching is best-effort."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(signature, key, params)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(arr))
        tmp.replace(path)
    except Exception:
        pass


def _cache_path(signature: str, key: str, params: tuple) -> Path:
    digest = hashlib.sha1(f"{signature}|{key}|{params!r}".encode()).hexdigest()
    return CACHE_DIR / f"{digest}.npy"
