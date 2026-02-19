#!/usr/bin/env python
"""Create a small synthetic NWB file for testing ezmsg-nwb.

Runnable standalone:
    python create_test_nwb.py

Also callable:
    from create_test_nwb import create_test_nwb
    create_test_nwb(Path("data/test_synthetic.nwb"))
"""

from __future__ import annotations

import datetime
from pathlib import Path

import h5py
import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.file import Subject, TimeIntervals


DURATION = 3.0  # seconds


def _downgrade_electrodes_table(filepath: str | Path) -> None:
    """Downgrade ElectrodesTable neurodata_type to DynamicTable for pynwb 2.x/3.0 compat.

    pynwb >= 3.1 writes ElectrodesTable (NWB schema 2.9.0+) which older pynwb can't read.
    ElectrodesTable extends DynamicTable, so downgrading the type is safe.
    """
    import json

    with h5py.File(str(filepath), "a") as f:
        # Downgrade the electrodes group neurodata_type
        grp = f.get("general/extracellular_ephys/electrodes")
        if grp is not None and grp.attrs.get("neurodata_type") == "ElectrodesTable":
            grp.attrs["neurodata_type"] = "DynamicTable"
            grp.attrs["namespace"] = "hdmf-common"

        # Patch cached specs so older readers don't choke on the unknown type
        specs_grp = f.get("specifications")
        if specs_grp is None:
            return
        for ns in specs_grp:
            for ver in specs_grp[ns]:
                for spec_name in specs_grp[ns][ver]:
                    dset = specs_grp[ns][ver][spec_name]
                    raw = dset[()]
                    text = raw.decode() if isinstance(raw, bytes) else raw
                    if "ElectrodesTable" not in text:
                        continue
                    spec_obj = json.loads(text)
                    _replace_type_refs(spec_obj, "ElectrodesTable", "DynamicTable")
                    dset[()] = json.dumps(spec_obj)


def _replace_type_refs(obj, old_type: str, new_type: str) -> None:
    """Recursively replace data_type_def/inc values in a spec dict."""
    if isinstance(obj, dict):
        for key in ("data_type_def", "data_type_inc"):
            if obj.get(key) == old_type:
                obj[key] = new_type
        for v in obj.values():
            _replace_type_refs(v, old_type, new_type)
    elif isinstance(obj, list):
        for item in obj:
            _replace_type_refs(item, old_type, new_type)


def create_test_nwb(output_path: Path | str) -> Path:
    """Create a synthetic NWB file exercising all slicer/iterator/clockdriven code paths.

    Streams created:
        Broadband      - ElectricalSeries, 1000 Hz, 8ch, explicit timestamps + rate attr (acquisition)
        RawAnalog      - TimeSeries, 500 Hz, 2ch, rate-only (acquisition)
        BinnedSpikes   - TimeSeries, 50 Hz, 4ch, rate-only (processing/ecephys)
        Force          - TimeSeries, 100 Hz, 1D, rate-only (processing/behavior)
        trials         - TimeIntervals, 3 events
        phonemes       - TimeIntervals, 10 events
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    nwbfile = NWBFile(
        session_description="Synthetic test data for ezmsg-nwb",
        identifier="test_synthetic_001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
        experimenter=["User, Test"],
        experiment_description="Synthetic data for unit testing ezmsg-nwb",
        institution="Test Institution",
        keywords=["test", "synthetic"],
        subject=Subject(
            subject_id="synth01",
            description="Synthetic test subject",
            species="Homo sapiens",
            age="P30Y",
            sex="U",
        ),
    )

    # --- Electrode table (8 electrodes with labels) ---
    device = nwbfile.create_device(name="TestArray", description="Synthetic electrode array")
    group = nwbfile.create_electrode_group(
        name="TestGroup", description="Test electrodes", location="cortex", device=device
    )
    nwbfile.add_electrode_column(name="label", description="Electrode label")
    for i in range(8):
        nwbfile.add_electrode(
            x=float(i), y=0.0, z=0.0,
            location="cortex",
            group=group,
            label=f"elec{i}",
        )
    electrode_region = nwbfile.create_electrode_table_region(
        region=list(range(8)), description="all electrodes"
    )

    # --- Broadband: ElectricalSeries, 1000 Hz, 8ch, explicit timestamps + rate attr ---
    #   Exercises: timestamped continuous + electrodes + labels + rate-from-attr
    bb_rate = 1000.0
    bb_n = int(DURATION * bb_rate)
    # Add small jitter so timestamps are not perfectly regular (avoids nwbinspector
    # check_regular_timestamps violation while still exercising the timestamped path).
    bb_timestamps = np.arange(bb_n, dtype=np.float64) / bb_rate
    bb_timestamps += rng.normal(scale=1e-6, size=bb_n)
    bb_timestamps.sort()  # ensure monotonic
    bb_data = rng.standard_normal((bb_n, 8)).astype(np.float32)

    broadband = ElectricalSeries(
        name="Broadband",
        data=bb_data,
        timestamps=bb_timestamps,
        electrodes=electrode_region,
        description="Synthetic broadband, timestamped with rate attr",
    )
    nwbfile.add_acquisition(broadband)

    # --- RawAnalog: TimeSeries, 500 Hz, 2ch, rate-only (no timestamps) ---
    #   Exercises: rate-only continuous, 2D without electrodes
    ra_rate = 500.0
    ra_n = int(DURATION * ra_rate)
    ra_data = rng.standard_normal((ra_n, 2)).astype(np.float32)

    from pynwb import TimeSeries

    raw_analog = TimeSeries(
        name="RawAnalog",
        data=ra_data,
        unit="V",
        rate=ra_rate,
        starting_time=0.0,
        description="Synthetic analog, rate-only, 2ch",
    )
    nwbfile.add_acquisition(raw_analog)

    # --- BinnedSpikes: TimeSeries, 50 Hz, 4ch, rate-only, in processing/ecephys ---
    #   Exercises: processing module discovery, rate-only 2D
    bs_rate = 50.0
    bs_n = int(DURATION * bs_rate)
    bs_data = rng.standard_normal((bs_n, 4)).astype(np.float32)

    binned_spikes = TimeSeries(
        name="BinnedSpikes",
        data=bs_data,
        unit="spikes/s",
        rate=bs_rate,
        starting_time=0.0,
        description="Synthetic binned spikes, rate-only, 4ch",
    )
    ecephys_module = nwbfile.create_processing_module(name="ecephys", description="ecephys processing")
    ecephys_module.add(binned_spikes)

    # --- Force: TimeSeries, 100 Hz, 1D, rate-only, in processing/behavior ---
    #   Exercises: 1D data + second processing module
    f_rate = 100.0
    f_n = int(DURATION * f_rate)
    f_data = rng.standard_normal(f_n).astype(np.float32)

    force = TimeSeries(
        name="Force",
        data=f_data,
        unit="N",
        rate=f_rate,
        starting_time=0.0,
        description="Synthetic force, 1D, rate-only",
    )
    behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior processing")
    behavior_module.add(force)

    # --- Trials: 3 events with custom columns ---
    nwbfile.add_trial_column(name="condition", description="Trial condition label")
    nwbfile.add_trial_column(name="correct", description="Whether response was correct")
    for i in range(3):
        nwbfile.add_trial(
            start_time=float(i),
            stop_time=float(i) + 0.8,
            condition=f"cond_{i % 2}",
            correct=str(i % 2 == 0),
        )

    # --- Phonemes: 10 events via custom interval table ---
    phonemes = TimeIntervals(name="phonemes", description="Phoneme events")
    phonemes.add_column(name="phoneme", description="Phoneme label")
    phoneme_labels = ["aa", "eh", "ih", "oh", "uh", "ss", "tt", "nn", "mm", "ll"]
    for i in range(10):
        t = 0.1 + i * 0.25
        phonemes.add_interval(
            start_time=t,
            stop_time=t + 0.15,
            phoneme=phoneme_labels[i],
        )
    nwbfile.add_time_intervals(phonemes)

    # --- Write ---
    with NWBHDF5IO(str(output_path), "w") as io:
        io.write(nwbfile)

    # --- Post-process with h5py ---
    with h5py.File(str(output_path), "a") as f:
        # Add rate attr to Broadband timestamps (exercises rate-from-attr path in slicer)
        f["acquisition/Broadband/timestamps"].attrs["rate"] = bb_rate

    # Downgrade ElectrodesTable for cross-version pynwb compatibility
    _downgrade_electrodes_table(output_path)

    # --- Print summary ---
    print(f"Created: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Broadband:    {bb_n} samples x 8 ch @ {bb_rate} Hz (timestamped + rate attr)")
    print(f"  RawAnalog:    {ra_n} samples x 2 ch @ {ra_rate} Hz (rate-only)")
    print(f"  BinnedSpikes: {bs_n} samples x 4 ch @ {bs_rate} Hz (rate-only, processing/ecephys)")
    print(f"  Force:        {f_n} samples x 1    @ {f_rate} Hz (1D, processing/behavior)")
    print(f"  trials:       3 events")
    print(f"  phonemes:     10 events")

    return output_path


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "test_synthetic.nwb"
    create_test_nwb(out)
