"""Benchmark NWBAxisArrayIterator: prefetch depth + simulated downstream work.

Usage:
    python benchmarks/bench_iterator.py PATH [--limit N] [--sim-ms F] [--prefetch 0 2 4 8]

The "simulated work" knob (sim-ms) approximates a real online pipeline that
takes some time per chunk. Without it, a fast consumer drains chunks faster
than the worker can read, masking the prefetch win. With it, the worker
reads the next chunk while the consumer is busy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ezmsg.nwb import NWBAxisArrayIterator, NWBIteratorSettings, ReferenceClockType


def _print(*args, **kwargs):
    """Print and flush so partial progress shows up under tee/pipe."""
    print(*args, **kwargs)
    sys.stdout.flush()


def run_one(
    filepath: Path,
    prefetch_chunks: int,
    chunk_dur: float,
    limit: int,
    sim_ms: float,
    stream_keys: list[str] | None,
) -> dict:
    settings = NWBIteratorSettings(
        filepath=str(filepath),
        chunk_dur=chunk_dur,
        reference_clock=ReferenceClockType.UNKNOWN,
        prefetch_chunks=prefetch_chunks,
        stream_keys=stream_keys,
    )

    t0 = time.perf_counter()
    it = NWBAxisArrayIterator(settings)
    t_open = time.perf_counter() - t0

    n_msgs = 0
    n_samples = 0
    n_bytes = 0
    sim_dur = sim_ms / 1000.0

    t0 = time.perf_counter()
    for msg in it:
        n_msgs += 1
        if msg.data.size > 0:
            n_samples += msg.data.shape[0]
            n_bytes += msg.data.nbytes
        if sim_dur > 0:
            time.sleep(sim_dur)
        if limit and n_msgs >= limit:
            break
    t_iter = time.perf_counter() - t0

    # Explicit teardown so the prefetch worker is joined before the next run.
    # gc.collect() is intentionally not called — it can interact poorly with
    # daemon threads mid-h5py-read during interpreter shutdown.
    del it

    return {
        "prefetch": prefetch_chunks,
        "t_open": t_open,
        "t_iter": t_iter,
        "n_msgs": n_msgs,
        "n_samples": n_samples,
        "n_bytes": n_bytes,
    }


def fmt_row(r: dict) -> str:
    iter_t = r["t_iter"]
    samp_per_s = r["n_samples"] / iter_t if iter_t > 0 else 0
    mb_per_s = (r["n_bytes"] / 1e6) / iter_t if iter_t > 0 else 0
    return (
        f"{r['prefetch']:>9d}  {r['t_open']:>7.2f}  {iter_t:>8.2f}  "
        f"{r['n_msgs']:>7d}  {r['n_samples']:>12d}  {samp_per_s:>14,.0f}  {mb_per_s:>8.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=Path)
    parser.add_argument("--chunk-dur", type=float, default=1.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max messages per run (0 = whole file). Default 200.",
    )
    parser.add_argument(
        "--sim-ms",
        type=float,
        default=0.0,
        help="Simulated downstream work per message, in ms. Default 0.",
    )
    parser.add_argument("--prefetch", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--streams", type=str, nargs="+", default=None)
    parser.add_argument("--inspect", action="store_true", help="Print discovered streams and exit.")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a small warmup pass first to populate the OS page cache. "
        "Off by default — the open phase already touches a lot of metadata, "
        "and a warmup adds a full second open (which can be slow for big files).",
    )
    args = parser.parse_args()

    if args.inspect:
        it = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=str(args.filepath),
                chunk_dur=args.chunk_dur,
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )
        slicer = it._state.slicer
        _print(f"duration: {slicer.stop_time - slicer.start_time:.1f}s, n_chunks={it._state.n_chunks}")
        _print()
        _print(f"{'stream':40s} {'shape':25s} {'dtype':12s} {'MB':>10}")
        for name, sd in it._state.streams.items():
            d = sd["info"].dset
            shape = getattr(d, "shape", None)
            dtype = getattr(d, "dtype", None)
            if shape is not None and dtype is not None:
                nbytes = 1
                for s in shape:
                    nbytes *= s
                nbytes *= dtype.itemsize
                _print(f"{name:40s} {str(shape):25s} {str(dtype):12s} {nbytes / 1e6:>10.1f}")
            else:
                _print(f"{name:40s} {type(d).__name__:25s}")
        return

    _print(f"file:        {args.filepath.name}")
    _print(f"chunk_dur:   {args.chunk_dur}s")
    _print(f"limit:       {args.limit if args.limit else 'whole file'} messages")
    _print(f"sim work:    {args.sim_ms} ms / message")
    if args.streams:
        _print(f"streams:     {args.streams}")
    _print()

    if args.warmup:
        _print("warming page cache...")
        _ = run_one(args.filepath, 0, args.chunk_dur, min(args.limit or 50, 50), 0.0, args.streams)

    header = (
        f"{'prefetch':>9}  {'open(s)':>7}  {'iter(s)':>8}  {'msgs':>7}  {'samples':>12}  {'samples/s':>14}  {'MB/s':>8}"
    )
    _print(header)
    _print("-" * len(header))

    results = []
    for pf in args.prefetch:
        r = run_one(args.filepath, pf, args.chunk_dur, args.limit, args.sim_ms, args.streams)
        results.append(r)
        _print(fmt_row(r))

    if len(results) > 1:
        _print()
        _print("speedup vs prefetch=0:")
        baseline_t = results[0]["t_iter"]
        for r in results[1:]:
            speedup = baseline_t / r["t_iter"] if r["t_iter"] > 0 else float("nan")
            _print(f"  prefetch={r['prefetch']:>3d}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
