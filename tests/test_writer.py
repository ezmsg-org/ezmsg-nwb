"""Tests for NWB writer module."""

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import pynwb
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from pynwb import NWBHDF5IO

from ezmsg.nwb import NWBSinkConsumer, NWBSinkSettings, ReferenceClockType


@pytest.fixture(autouse=True)
def _reset_nwbsink_shared():
    """The ``shared_*`` class attrs are deliberately sticky across instances
    within a process; reset them between tests so session-start semantics
    don't leak."""
    NWBSinkConsumer.shared_start_datetime = None
    NWBSinkConsumer.shared_t0 = None
    NWBSinkConsumer.shared_clock_type = None
    yield
    NWBSinkConsumer.shared_start_datetime = None
    NWBSinkConsumer.shared_t0 = None
    NWBSinkConsumer.shared_clock_type = None


def _make_continuous(
    n: int = 50,
    ch: int = 4,
    fs: float = 1000.0,
    offset: float = 0.0,
    key: str = "TestStream",
    ch_labels: list[str] | None = None,
) -> AxisArray:
    axes = {"time": AxisArray.TimeAxis(fs=fs, offset=offset)}
    if ch_labels is not None:
        axes["ch"] = AxisArray.CoordinateAxis(np.asarray(ch_labels), dims=["ch"], unit="")
    return AxisArray(
        data=np.random.randn(n, ch).astype(np.float32),
        dims=["time", "ch"],
        axes=axes,
        key=key,
    )


def _fresh_path(stem: str) -> Path:
    p = Path(tempfile.gettempdir()) / f"ezmsg_nwb_{stem}.nwb"
    p.unlink(missing_ok=True)
    return p


def _sink(filepath: Path, **overrides) -> NWBSinkConsumer:
    base = dict(
        filepath=filepath,
        overwrite_old=True,
        inc_clock=ReferenceClockType.UNKNOWN,
    )
    base.update(overrides)
    return NWBSinkConsumer(settings=NWBSinkSettings(**base))


def test_sink_settings_defaults():
    """Test NWBSinkSettings default values."""
    settings = NWBSinkSettings(filepath="/tmp/test.nwb")
    assert settings.overwrite_old is False
    assert settings.axis == "time"
    assert settings.recording is True
    assert settings.inc_clock == ReferenceClockType.SYSTEM
    assert settings.meta_yaml is None
    assert settings.split_bytes == 0
    assert settings.expected_series is None


def test_sink_settings_custom():
    """Test NWBSinkSettings with custom values."""
    settings = NWBSinkSettings(
        filepath="/tmp/test.nwb",
        overwrite_old=True,
        axis="win",
        recording=False,
        inc_clock=ReferenceClockType.MONOTONIC,
        split_bytes=1024,
    )
    assert settings.overwrite_old is True
    assert settings.axis == "win"
    assert settings.recording is False
    assert settings.inc_clock == ReferenceClockType.MONOTONIC
    assert settings.split_bytes == 1024


# -- update_settings / reset machinery --


def test_update_settings_recording_does_not_reset():
    path = _fresh_path("update_recording")
    sink = _sink(path, recording=True)
    io_before = sink._state.io
    hash_before = sink._hash

    sink.update_settings(dataclasses.replace(sink.settings, recording=False))

    assert sink._state.io is io_before, "recording-only change must not swap io"
    assert sink._hash == hash_before, "recording-only change must not request reset"
    assert sink.settings.recording is False

    sink.close(write=False)
    path.unlink(missing_ok=True)


def test_update_settings_split_bytes_does_not_reset():
    path = _fresh_path("update_splitbytes")
    sink = _sink(path, split_bytes=0)
    io_before = sink._state.io
    hash_before = sink._hash

    sink.update_settings(dataclasses.replace(sink.settings, split_bytes=1024))

    assert sink._state.io is io_before
    assert sink._hash == hash_before
    assert sink.settings.split_bytes == 1024

    sink.close(write=False)
    path.unlink(missing_ok=True)


def test_update_settings_filepath_swaps_file():
    path_a = _fresh_path("update_filepath_a")
    path_b = _fresh_path("update_filepath_b")
    sink = _sink(path_a)
    io_a = sink._state.io

    sink.update_settings(dataclasses.replace(sink.settings, filepath=path_b, overwrite_old=True))
    assert sink._hash == -1, "reset-field change must request reset"

    # Simulate what the stateful machinery does on the next message:
    # hash mismatch triggers _reset_state, which closes A and opens B.
    sink._reset_state(None)
    sink._hash = sink._hash_message(None)

    assert sink._state.io is not io_a
    assert sink._state.filepath.name == path_b.name

    sink.close(write=False)
    for p in (path_a, path_b):
        p.unlink(missing_ok=True)


def test_update_settings_axis_requires_reset():
    path = _fresh_path("update_axis")
    sink = _sink(path)
    hash_before = sink._hash

    sink.update_settings(dataclasses.replace(sink.settings, axis="win"))

    assert sink._hash == -1, "axis change must request reset"
    assert hash_before == 0

    sink.close(write=False)
    path.unlink(missing_ok=True)


# -- recording gate --


def test_recording_false_skips_data_write():
    path = _fresh_path("recording_gate")
    sink = _sink(path, recording=False)

    msg = _make_continuous(n=100, ch=2, key="Gated")
    sink._process(msg)

    # Series was created (prep ran), but no data should have been appended.
    assert "Gated" in sink._state.series
    ss = sink._state.series["Gated"]
    assert ss.bytes_written == 0
    assert len(ss.data) == 0

    sink.close(write=False)
    path.unlink(missing_ok=True)


def test_recording_toggled_live_resumes_writes():
    path = _fresh_path("recording_live")
    sink = _sink(path, recording=False)

    msg1 = _make_continuous(n=50, key="Live")
    sink._process(msg1)
    assert sink._state.series["Live"].bytes_written == 0

    # Flip the recording flag with a NONRESET update.
    sink.update_settings(dataclasses.replace(sink.settings, recording=True))

    msg2 = _make_continuous(n=50, offset=0.05, key="Live")
    sink._process(msg2)
    assert sink._state.series["Live"].bytes_written > 0

    sink.close(write=False)
    path.unlink(missing_ok=True)


# -- error paths --


def test_shape_mismatch_raises_and_closes():
    path = _fresh_path("shape_mismatch")
    sink = _sink(path)

    sink._process(_make_continuous(n=50, ch=4, key="S"))
    bad = _make_continuous(n=50, ch=8, key="S")  # different trailing shape

    with pytest.raises(ValueError, match="changed shape"):
        sink._process(bad)
    assert sink._state.io is None, "close() should run on shape mismatch"

    path.unlink(missing_ok=True)


def test_str_data_without_event_key_raises():
    path = _fresh_path("str_no_event")
    sink = _sink(path)

    msg = AxisArray(
        data=np.array([["a"], ["b"]], dtype="<U10"),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=10.0, offset=0.0)},
        key="NotAnEventKey",
    )
    with pytest.raises(ValueError, match="varlen str"):
        sink._process(msg)

    sink.close(write=False)
    path.unlink(missing_ok=True)


def test_overwrite_old_false_with_existing_file_raises():
    path = _fresh_path("no_overwrite")
    path.write_bytes(b"")  # Make the file exist.
    try:
        with pytest.raises(ValueError, match="overwriting is disabled"):
            NWBSinkConsumer(
                settings=NWBSinkSettings(
                    filepath=path,
                    overwrite_old=False,
                    inc_clock=ReferenceClockType.UNKNOWN,
                )
            )
    finally:
        path.unlink(missing_ok=True)


# -- write-path variations --


def test_channel_labels_create_electrodes_table():
    path = _fresh_path("electrodes")
    sink = _sink(path)
    labels = ["ch0", "ch1", "ch2", "ch3"]

    msg = _make_continuous(n=20, ch=4, key="Labelled", ch_labels=labels)
    sink._process(msg)
    sink.close(write=True)

    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        assert "Labelled" in nwbfile.acquisition
        series = nwbfile.acquisition["Labelled"]
        assert isinstance(series, pynwb.ecephys.ElectricalSeries)
        assert nwbfile.electrodes is not None
        assert list(nwbfile.electrodes["label"][:]) == labels

    path.unlink(missing_ok=True)


def test_events_roundtrip():
    path = _fresh_path("events")
    sink = _sink(path)

    msg = AxisArray(
        data=np.array([["event_a"], ["event_b"], ["event_c"]], dtype="<U16"),
        dims=["time", "ch"],
        axes={"time": AxisArray.CoordinateAxis(np.array([0.1, 0.5, 1.2]), dims=["time"], unit="s")},
        key="epochs",
    )
    sink._process(msg)
    sink.close(write=True)

    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        epochs_df = nwbfile.epochs.to_dataframe()
        # Writer prepends an "EZNWB-START" dummy row; real entries follow.
        labels = list(epochs_df["label"])
        assert "EZNWB-START" in labels
        for expected in ("event_a", "event_b", "event_c"):
            assert expected in labels

    path.unlink(missing_ok=True)


def test_multiple_streams_coexist():
    """Two continuous streams to the same sink → both appear in acquisition."""
    path = _fresh_path("multi_stream")
    sink = _sink(path)

    sink._process(_make_continuous(n=30, ch=2, key="Alpha"))
    sink._process(_make_continuous(n=30, ch=2, offset=0.03, key="Beta"))
    sink.close(write=True)

    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        assert "Alpha" in nwbfile.acquisition
        assert "Beta" in nwbfile.acquisition
        assert len(nwbfile.acquisition["Alpha"].data) == 30
        assert len(nwbfile.acquisition["Beta"].data) == 30

    path.unlink(missing_ok=True)


def test_split_bytes_creates_new_file():
    """When cumulative writes exceed split_bytes, a new file is opened."""
    nominal = _fresh_path("split")
    # With split_bytes>0 and no "%d" in path, files land at <stem>_00.nwb,
    # <stem>_01.nwb, ...; the nominal path itself is never created.
    file_00 = nominal.parent / (nominal.stem + "_00" + nominal.suffix)
    file_01 = nominal.parent / (nominal.stem + "_01" + nominal.suffix)
    for p in (file_00, file_01):
        p.unlink(missing_ok=True)

    threshold = 1_000_000  # ~1 MB
    sink = _sink(nominal, split_bytes=threshold)

    # Each chunk: 1000 × 100 × float32 (= 400 KB data) + 1000 × float64
    # (= 8 KB timestamps) ≈ 408 KB. Three chunks → ~1.22 MB, crossing the
    # 1 MB threshold and triggering a split right after the third chunk.
    n_samples, n_ch = 1000, 100
    chunks = [_make_continuous(n=n_samples, ch=n_ch, fs=1000.0, offset=i * 1.0, key="Big") for i in range(3)]
    for c in chunks:
        sink._process(c)

    assert sink._state.split_count == 1, "expected exactly one split after ~1.2 MB"
    assert file_00.exists(), "first split file should be on disk"
    # _state.io now points at file_01 — confirm it's open and empty so far.
    assert sink._state.io is not None
    assert sink._state.series["Big"].bytes_written == 0

    # Drop one more chunk into the new file, then close.
    sink._process(_make_continuous(n=n_samples, ch=n_ch, fs=1000.0, offset=4.0, key="Big"))
    sink.close(write=True)

    assert file_01.exists(), "second split file should be on disk"

    # Verify each file holds its expected slice.
    with pynwb.NWBHDF5IO(str(file_00), "r") as io:
        assert len(io.read().acquisition["Big"].data) == 3 * n_samples
    with pynwb.NWBHDF5IO(str(file_01), "r") as io:
        assert len(io.read().acquisition["Big"].data) == n_samples

    for p in (file_00, file_01):
        p.unlink(missing_ok=True)


def test_multi_sink_share_session_start():
    """Two sinks with the same ``inc_clock`` should anchor session start once
    and share it — that's the entire point of the ``shared_*`` class attrs."""
    path_a = _fresh_path("multi_session_a")
    path_b = _fresh_path("multi_session_b")

    # Anchor the session via sink A's first message.
    sink_a = _sink(path_a, inc_clock=ReferenceClockType.SYSTEM)
    sink_a._process(_make_continuous(n=10, fs=100.0, offset=1_700_000_000.0, key="A"))
    anchor_dt = NWBSinkConsumer.shared_start_datetime
    anchor_t0 = NWBSinkConsumer.shared_t0
    assert anchor_dt is not None
    assert anchor_t0 is not None

    # Sink B should reuse the same anchor without overwriting it.
    sink_b = _sink(path_b, inc_clock=ReferenceClockType.SYSTEM)
    sink_b._process(_make_continuous(n=10, fs=100.0, offset=1_700_000_500.0, key="B"))

    assert NWBSinkConsumer.shared_start_datetime == anchor_dt
    assert NWBSinkConsumer.shared_t0 == anchor_t0

    sink_a.close(write=True)
    sink_b.close(write=True)

    # session_start_time on disk should match across files.
    with pynwb.NWBHDF5IO(str(path_a), "r") as io_a, pynwb.NWBHDF5IO(str(path_b), "r") as io_b:
        assert io_a.read().session_start_time == io_b.read().session_start_time

    for p in (path_a, path_b):
        p.unlink(missing_ok=True)


def test_clock_type_mismatch_raises():
    """A second sink with a different ``inc_clock`` than the established
    shared one must refuse — preventing inconsistent timestamps in the file."""
    path_a = _fresh_path("clock_a")
    path_b = _fresh_path("clock_b")

    sink_a = _sink(path_a, inc_clock=ReferenceClockType.SYSTEM)
    sink_a._process(_make_continuous(n=5, key="A"))
    assert NWBSinkConsumer.shared_clock_type == ReferenceClockType.SYSTEM

    # Construction triggers _nwb_create_or_fail → get_session_datetime which
    # checks the shared_clock_type. Mismatched clock must raise.
    with pytest.raises(ValueError, match="share the same clock type"):
        _sink(path_b, inc_clock=ReferenceClockType.MONOTONIC)

    sink_a.close(write=False)
    for p in (path_a, path_b):
        p.unlink(missing_ok=True)


def test_meta_yaml_roundtrip(tmp_path):
    """Custom ``meta_yaml`` should populate NWBFile / Subject fields readable
    after the file closes."""
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "NWBFile:\n"
        "  session_description: ezmsg-nwb meta_yaml test\n"
        "  experimenter:\n"
        "    - Tester, Test\n"
        "  institution: Test Institute\n"
        "Subject:\n"
        "  subject_id: TestSubject001\n"
        "  species: Mus musculus\n"
        "  sex: U\n"
        "  age: P30D\n"
    )
    nwb_path = tmp_path / "meta.nwb"
    sink = _sink(nwb_path, meta_yaml=meta_path)
    sink._process(_make_continuous(n=10, key="K"))
    sink.close(write=True)

    with pynwb.NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        assert nwbfile.session_description == "ezmsg-nwb meta_yaml test"
        assert "Tester, Test" in list(nwbfile.experimenter)
        assert nwbfile.institution == "Test Institute"
        assert nwbfile.subject is not None
        assert nwbfile.subject.subject_id == "TestSubject001"
        assert nwbfile.subject.species == "Mus musculus"


def test_rate_change_raises_and_closes():
    """Same key, different sample rate → consistency check fires alongside
    the shape check and tears the file down."""
    path = _fresh_path("rate_change")
    sink = _sink(path)

    sink._process(_make_continuous(n=50, ch=4, fs=1000.0, key="K"))
    bad = _make_continuous(n=50, ch=4, fs=2000.0, offset=0.05, key="K")

    with pytest.raises(ValueError, match="changed shape"):
        sink._process(bad)
    assert sink._state.io is None, "rate change must close the file"

    path.unlink(missing_ok=True)


def test_expected_series_smoke(tmp_path):
    """``expected_series`` should pre-allocate stream containers from a
    metadata yaml so the file is fully shaped before any message arrives."""
    expected_yaml = tmp_path / "expected.yaml"
    expected_yaml.write_text("PreAlloc:\n" "  fs: 100.0\n" "  shape: [-1, 4]\n")
    nwb_path = tmp_path / "expected.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=nwb_path,
            overwrite_old=True,
            expected_series=expected_yaml,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    assert "PreAlloc" in sink._state.series
    sink.close(write=True)

    with pynwb.NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        assert "PreAlloc" in nwbfile.acquisition


def test_sample_trigger_routes_to_epochs():
    """SampleTriggerMessage with a string label should land in nwbfile.epochs."""
    pytest.importorskip("ezmsg.baseproc")
    from ezmsg.baseproc import SampleTriggerMessage

    path = _fresh_path("sample_trigger")
    sink = _sink(path)

    # Prime the session start so the trigger's timestamp resolves to
    # a positive offset rather than a near-zero "now" datetime.
    NWBSinkConsumer.shared_t0 = 0.0
    NWBSinkConsumer.shared_clock_type = ReferenceClockType.UNKNOWN

    trigger = SampleTriggerMessage(
        timestamp=1.5,
        period=(0.0, 0.0),
        value="trigger_a",
    )
    sink._process(trigger)
    sink.close(write=True)

    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        assert nwbfile.epochs is not None
        labels = list(nwbfile.epochs.to_dataframe()["label"])
        # Writer always prepends the EZNWB-START sentinel; assert the
        # trigger label is present alongside it.
        assert "EZNWB-START" in labels
        assert "trigger_a" in labels

    path.unlink(missing_ok=True)


# -- annotation series writer --


def test_write_annotation_creates_series_and_appends(tmp_path):
    """First annotation write should create the AnnotationSeries; subsequent
    writes should append rows."""
    outpath = tmp_path / "ezmsg_nwb_annotation_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.write_annotation("settings_annotations", timestamp=1.0, data='{"k": "a"}')
    sink.write_annotation("settings_annotations", timestamp=2.0, data='{"k": "b"}')
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        series = nwbfile.acquisition["settings_annotations"]
        assert list(series.data[:]) == ['{"k": "a"}', '{"k": "b"}']
        np.testing.assert_array_equal(series.timestamps[:], np.array([1.0, 2.0]))


def test_write_annotation_keeps_file_with_no_acquisition_data(tmp_path):
    """A file with only annotations (no acquisition) should not be deleted on close."""
    outpath = tmp_path / "ezmsg_nwb_annotation_only_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink.write_annotation("settings_annotations", timestamp=1.0, data="row")
    sink.close(write=False)
    assert outpath.exists()


def test_write_annotation_after_data_uses_data_baseline(tmp_path):
    """Annotation timestamps stored in-file should be relative to the data anchor.

    With UNKNOWN clock, ``start_timestamp`` is latched off the first data
    message's axis offset (here, 0). An annotation at wall-clock 1.0 should
    therefore land at relative time 1.0 in the file.
    """
    outpath = tmp_path / "ezmsg_nwb_annotation_after_data_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink._process(
        AxisArray(
            data=np.arange(6, dtype=float).reshape(3, 2),
            dims=["time", "ch"],
            axes={"time": AxisArray.TimeAxis(fs=100.0)},
            key="sig",
        )
    )
    sink.write_annotation("settings_annotations", timestamp=1.0, data="row")
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        ts = nwbfile.acquisition["settings_annotations"].timestamps[:]
    np.testing.assert_array_equal(ts, np.array([1.0]))


def test_write_annotation_multiple_tables_coexist(tmp_path):
    """Two distinct table_names should produce two distinct AnnotationSeries."""
    outpath = tmp_path / "ezmsg_nwb_annotation_multi_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.write_annotation("settings_annotations", timestamp=1.0, data="A")
    sink.write_annotation("user_notes", timestamp=2.0, data="B")
    sink.write_annotation("settings_annotations", timestamp=3.0, data="C")
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        assert list(nwbfile.acquisition["settings_annotations"].data[:]) == ["A", "C"]
        assert list(nwbfile.acquisition["user_notes"].data[:]) == ["B"]


def test_write_annotation_serializes_json_payload(tmp_path):
    """Sanity check: a self-describing JSON payload round-trips through the file."""
    outpath = tmp_path / "ezmsg_nwb_annotation_json_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    payload = json.dumps({"component": "X.Y", "event_type": "INITIAL", "seq": 0, "settings": {"foo": 1}})
    sink.write_annotation("settings_annotations", timestamp=0.5, data=payload)
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        round_tripped = json.loads(nwbfile.acquisition["settings_annotations"].data[0])
    assert round_tripped["component"] == "X.Y"
    assert round_tripped["settings"] == {"foo": 1}


def test_close_is_idempotent_after_annotation(tmp_path):
    """Closing twice after writing annotations should not error."""
    outpath = tmp_path / "ezmsg_nwb_annotation_double_close_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink.write_annotation("settings_annotations", timestamp=1.0, data="row")
    sink.close(write=False)
    sink.close(write=False)
    assert outpath.exists()
