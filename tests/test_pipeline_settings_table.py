"""Tests for the typed-column pipeline-settings sink (Phase 2)."""

import asyncio
import socket
import typing
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.baseproc import (
    INIT_FINAL_COMPONENT_ADDRESS,
    PipelineSettingsEvent,
    PipelineSettingsEventType,
    PipelineSettingsProducerSettings,
)
from pynwb import NWBHDF5IO

from ezmsg.nwb import (
    NWBPipelineSettingsSinkConsumer,
    NWBPipelineSettingsSinkSettings,
    PipelineSettingsTableCollection,
    PipelineSettingsTableCollectionSettings,
    ReferenceClockType,
)


@pytest.fixture(autouse=True)
def _reset_nwbsink_shared():
    NWBPipelineSettingsSinkConsumer.shared_start_datetime = None
    NWBPipelineSettingsSinkConsumer.shared_t0 = None
    NWBPipelineSettingsSinkConsumer.shared_clock_type = None
    yield
    NWBPipelineSettingsSinkConsumer.shared_start_datetime = None
    NWBPipelineSettingsSinkConsumer.shared_t0 = None
    NWBPipelineSettingsSinkConsumer.shared_clock_type = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_event(
    *,
    component: str = "X.Y",
    seq: int = 1,
    timestamp: float = 1.0,
    event_type: PipelineSettingsEventType = PipelineSettingsEventType.UPDATED,
    structured: typing.Optional[dict] = None,
) -> PipelineSettingsEvent:
    sv = structured if structured is not None else {"foo": 1, "bar": 2.5}
    return PipelineSettingsEvent(
        seq=seq,
        timestamp=timestamp,
        component_address=component,
        event_type=event_type,
        repr_value=sv,
        structured_value=sv,
    )


def _sink_consumer(filepath: Path, **overrides) -> NWBPipelineSettingsSinkConsumer:
    base = dict(
        filepath=filepath,
        overwrite_old=True,
        inc_clock=ReferenceClockType.UNKNOWN,
    )
    base.update(overrides)
    return NWBPipelineSettingsSinkConsumer(settings=NWBPipelineSettingsSinkSettings(**base))


# ---------------------------------------------------------------------------
# Single-event semantics
# ---------------------------------------------------------------------------


def test_single_event_creates_table_with_anchor_and_close_rows(tmp_path):
    """First event eagerly writes an anchor row ``[t, t]`` so the initial
    settings hit disk immediately. Close adds the open-interval row
    ``[t, close_time]``."""
    outpath = tmp_path / "single_event.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(_make_event(structured={"alpha": 1, "beta": "hello"}))
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        table = nwbfile.intervals["pipeline_settings"]
        df = table.to_dataframe()

    assert "X.Y.alpha" in df.columns
    assert "X.Y.beta" in df.columns
    # 2 rows: the anchor [1.0, 1.0] and the close-flush [1.0, close].
    assert len(df) == 2
    assert df["start_time"].iloc[0] == df["stop_time"].iloc[0] == 1.0
    assert df["start_time"].iloc[1] == 1.0
    assert df["stop_time"].iloc[1] >= 1.0
    assert (df["X.Y.alpha"] == 1).all()
    assert (df["X.Y.beta"] == "hello").all()
    assert (df["updated_component"] == "X.Y").all()


def test_two_events_close_first_interval(tmp_path):
    """Two events plus close = anchor + 2 closed intervals = 3 rows."""
    outpath = tmp_path / "two_events.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(_make_event(component="A", seq=1, timestamp=10.0, structured={"x": 1}))
    sink.write_settings_event(_make_event(component="A", seq=2, timestamp=20.0, structured={"x": 2}))
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    # Three rows in order: anchor [10, 10] @ values=1, closed [10, 20] @
    # values=1, open-on-close [20, close-time] @ values=2.
    assert len(df) == 3
    assert df["A.x"].tolist() == [1, 1, 2]
    assert df["start_time"].tolist()[:2] == [10.0, 10.0]
    assert df["stop_time"].iloc[0] == 10.0
    assert df["stop_time"].iloc[1] == 20.0
    assert df["start_time"].iloc[2] == 20.0
    assert df["stop_time"].iloc[2] >= 20.0


def test_native_dtypes_preserved(tmp_path):
    """Floats stay floats, ints stay ints, strings stay strings, fixed-shape
    lists become arrays."""
    outpath = tmp_path / "dtypes.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(
        _make_event(
            structured={
                "a_int": 7,
                "a_float": 3.14,
                "a_str": "x",
                "a_list": [1.0, 2.0, 3.0],
            }
        )
    )
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    assert df["X.Y.a_int"].iloc[-1] == 7
    assert df["X.Y.a_float"].iloc[-1] == pytest.approx(3.14)
    assert df["X.Y.a_str"].iloc[-1] == "x"
    np.testing.assert_array_equal(df["X.Y.a_list"].iloc[-1], np.array([1.0, 2.0, 3.0]))


def test_close_with_no_events_deletes_empty_file(tmp_path):
    """A sink that received no settings events should still get the
    parent's empty-file cleanup behavior."""
    outpath = tmp_path / "empty.nwb"
    sink = _sink_consumer(outpath)
    sink.close(write=False)
    assert not outpath.exists()


def test_close_with_only_settings_keeps_file(tmp_path):
    """A populated settings table is content — file should not be deleted
    even if no acquisition data was written."""
    outpath = tmp_path / "settings_only.nwb"
    sink = _sink_consumer(outpath)
    sink.write_settings_event(_make_event())
    sink.close(write=False)
    assert outpath.exists()


# ---------------------------------------------------------------------------
# Schema-compatible append
# ---------------------------------------------------------------------------


def test_compatible_update_appends_without_rotation(tmp_path):
    """Same keys, same shapes → no rotation; rows accumulate in one table.

    Three events + close = 1 anchor + 3 closed-by-next-event/close = 4 rows.
    """
    outpath = tmp_path / "compat.nwb"
    rotated = tmp_path / "compat_01.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(_make_event(seq=1, timestamp=1.0, structured={"x": 1, "y": "a"}))
    sink.write_settings_event(_make_event(seq=2, timestamp=2.0, structured={"x": 2, "y": "b"}))
    sink.write_settings_event(_make_event(seq=3, timestamp=3.0, structured={"x": 3, "y": "c"}))
    sink.close(write=False)

    assert outpath.exists()
    assert not rotated.exists(), "no schema change → no rotation"

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    # Anchor row holds event 1's values; rows 1-3 are the proper closed
    # intervals as each successive event arrives (and close).
    assert df["X.Y.x"].tolist() == [1, 1, 2, 3]
    assert df["X.Y.y"].tolist() == ["a", "a", "b", "c"]


# ---------------------------------------------------------------------------
# Schema rotation
# ---------------------------------------------------------------------------


def test_new_column_triggers_rotation(tmp_path):
    """A new column key in a later event should rotate to a fresh file
    segment whose table opens with both old + new columns."""
    outpath = tmp_path / "rotate_new_col.nwb"
    rotated = tmp_path / "rotate_new_col_01.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(_make_event(seq=1, timestamp=1.0, structured={"x": 1}))
    sink.write_settings_event(_make_event(seq=2, timestamp=2.0, structured={"x": 2, "y": 99}))
    sink.close(write=False)

    assert outpath.exists()
    assert rotated.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df0 = nwbfile.intervals["pipeline_settings"].to_dataframe()
        assert "X.Y.x" in df0.columns
        assert "X.Y.y" not in df0.columns

    with NWBHDF5IO(rotated, "r") as io:
        nwbfile = io.read()
        df1 = nwbfile.intervals["pipeline_settings"].to_dataframe()
        assert "X.Y.x" in df1.columns
        assert "X.Y.y" in df1.columns
        assert df1["X.Y.x"].iloc[-1] == 2
        assert df1["X.Y.y"].iloc[-1] == 99


def test_scalar_to_array_triggers_rotation(tmp_path):
    """Scalar→array transition for an existing column rotates files."""
    outpath = tmp_path / "rotate_scalar_array.nwb"
    rotated = tmp_path / "rotate_scalar_array_01.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(_make_event(seq=1, timestamp=1.0, structured={"v": "foo"}))
    sink.write_settings_event(_make_event(seq=2, timestamp=2.0, structured={"v": [1, 2, 3]}))
    sink.close(write=False)

    assert outpath.exists()
    assert rotated.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df0 = nwbfile.intervals["pipeline_settings"].to_dataframe()
        assert df0["X.Y.v"].iloc[-1] == "foo"

    with NWBHDF5IO(rotated, "r") as io:
        nwbfile = io.read()
        df1 = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df1["X.Y.v"].iloc[-1], np.array([1, 2, 3]))


def test_rank_change_triggers_rotation(tmp_path):
    """1-D → 2-D shape change rotates files."""
    outpath = tmp_path / "rotate_rank.nwb"
    rotated = tmp_path / "rotate_rank_01.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(_make_event(seq=1, timestamp=1.0, structured={"v": [1, 2, 3]}))
    sink.write_settings_event(_make_event(seq=2, timestamp=2.0, structured={"v": [[1, 2, 3]]}))
    sink.close(write=False)

    assert outpath.exists()
    assert rotated.exists()

    with NWBHDF5IO(rotated, "r") as io:
        nwbfile = io.read()
        df1 = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df1["X.Y.v"].iloc[-1], np.array([[1, 2, 3]]))


def test_inner_dim_shape_change_triggers_rotation(tmp_path):
    """Same rank, different inner-dim shape (e.g. (2,2) → (2,3)) rotates."""
    outpath = tmp_path / "rotate_inner.nwb"
    rotated = tmp_path / "rotate_inner_01.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(_make_event(seq=1, timestamp=1.0, structured={"v": [[1, 2], [3, 4]]}))
    sink.write_settings_event(_make_event(seq=2, timestamp=2.0, structured={"v": [[1, 2, 3], [4, 5, 6]]}))
    sink.close(write=False)

    assert outpath.exists()
    assert rotated.exists()

    with NWBHDF5IO(rotated, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df["X.Y.v"].iloc[-1], np.array([[1, 2, 3], [4, 5, 6]]))


# ---------------------------------------------------------------------------
# Multi-component INITIAL snapshot — aggregation via INIT_FINAL sentinel
# ---------------------------------------------------------------------------


def _init_final_event(timestamp: float = 0.0) -> PipelineSettingsEvent:
    """Build a sentinel PipelineSettingsEvent matching what the producer emits."""
    return PipelineSettingsEvent(
        seq=999,
        timestamp=timestamp,
        component_address=INIT_FINAL_COMPONENT_ADDRESS,
        event_type=PipelineSettingsEventType.INITIAL,
        repr_value="",
        structured_value=None,
    )


def test_initial_events_buffer_until_sentinel_then_merge(tmp_path):
    """Per-component INITIAL events with disjoint columns should buffer
    in memory; the sentinel flushes one merged anchor row that contains
    every component's settings — no rotation required."""
    outpath = tmp_path / "initial_buffered.nwb"
    rotated = tmp_path / "initial_buffered_01.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(
        _make_event(
            component="A",
            seq=1,
            timestamp=1.0,
            event_type=PipelineSettingsEventType.INITIAL,
            structured={"foo": 1},
        )
    )
    sink.write_settings_event(
        _make_event(
            component="B",
            seq=2,
            timestamp=2.0,
            event_type=PipelineSettingsEventType.INITIAL,
            structured={"bar": "hi"},
        )
    )
    sink.write_settings_event(_init_final_event(timestamp=3.0))
    sink.close(write=False)

    assert outpath.exists()
    assert not rotated.exists(), "merged anchor → no rotation"

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    # Anchor row + close-flush row, both with the merged columns.
    assert "A.foo" in df.columns
    assert "B.bar" in df.columns
    assert (df["A.foo"] == 1).all()
    assert (df["B.bar"] == "hi").all()
    # Anchor is at the FIRST INITIAL's timestamp (1.0), not the sentinel's.
    assert df["start_time"].iloc[0] == 1.0
    assert df["stop_time"].iloc[0] == 1.0
    # close-flush extends through close.
    assert df["start_time"].iloc[-1] == 1.0
    assert df["stop_time"].iloc[-1] >= 1.0
    assert df["updated_component"].iloc[0] == INIT_FINAL_COMPONENT_ADDRESS


def test_updated_event_after_buffered_initials_flushes_buffer(tmp_path):
    """If an UPDATED event arrives without an INIT_FINAL sentinel
    (producer dropped it), the buffer should flush as a merged anchor
    before the UPDATED event closes the open interval."""
    outpath = tmp_path / "initial_no_sentinel.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(
        _make_event(
            component="A",
            seq=1,
            timestamp=1.0,
            event_type=PipelineSettingsEventType.INITIAL,
            structured={"x": 10},
        )
    )
    sink.write_settings_event(
        _make_event(
            component="B",
            seq=2,
            timestamp=2.0,
            event_type=PipelineSettingsEventType.INITIAL,
            structured={"y": 20},
        )
    )
    # No sentinel — go straight to an UPDATED.
    sink.write_settings_event(
        _make_event(
            component="A",
            seq=3,
            timestamp=5.0,
            event_type=PipelineSettingsEventType.UPDATED,
            structured={"x": 11},
        )
    )
    sink.close(write=False)

    rotated = tmp_path / "initial_no_sentinel_01.nwb"
    assert not rotated.exists(), "buffer-flush should not require a rotation"

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    # 3 rows: anchor, closed-by-UPDATED, close-flush.
    assert len(df) == 3
    assert "A.x" in df.columns
    assert "B.y" in df.columns
    # Anchor + first interval carry initial values; close-flush has the
    # UPDATED value.
    assert df["A.x"].tolist() == [10, 10, 11]
    assert df["B.y"].tolist() == [20, 20, 20]


def test_initial_buffered_then_close_without_sentinel(tmp_path):
    """Buffered INITIALs but neither sentinel nor UPDATED ever arrived;
    close should still flush them as a merged anchor."""
    outpath = tmp_path / "initial_only_close.nwb"
    sink = _sink_consumer(outpath)

    sink.write_settings_event(
        _make_event(
            component="A",
            seq=1,
            timestamp=1.0,
            event_type=PipelineSettingsEventType.INITIAL,
            structured={"foo": 7},
        )
    )
    sink.close(write=False)

    assert outpath.exists()
    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
    assert "A.foo" in df.columns
    assert (df["A.foo"] == 7).all()


def test_late_initial_after_anchor_takes_normal_path(tmp_path):
    """An INITIAL event arriving AFTER the table is registered (e.g. a
    runtime new component) should go through the normal update/rotation
    path rather than the buffer."""
    outpath = tmp_path / "late_initial.nwb"
    rotated = tmp_path / "late_initial_01.nwb"
    sink = _sink_consumer(outpath)

    # Establish the table first via a buffered initial + sentinel.
    sink.write_settings_event(
        _make_event(
            component="A",
            seq=1,
            timestamp=1.0,
            event_type=PipelineSettingsEventType.INITIAL,
            structured={"x": 1},
        )
    )
    sink.write_settings_event(_init_final_event(timestamp=2.0))

    # A new component appears mid-run and announces itself as INITIAL —
    # introduces a brand-new column, so this should rotate.
    sink.write_settings_event(
        _make_event(
            component="LATE",
            seq=2,
            timestamp=5.0,
            event_type=PipelineSettingsEventType.INITIAL,
            structured={"y": "hi"},
        )
    )
    sink.close(write=False)

    assert outpath.exists()
    assert rotated.exists()


def test_double_close_idempotent(tmp_path):
    """Closing twice after writing settings should not error."""
    outpath = tmp_path / "double_close.nwb"
    sink = _sink_consumer(outpath)
    sink.write_settings_event(_make_event())
    sink.close(write=False)
    sink.close(write=False)
    assert outpath.exists()


# ---------------------------------------------------------------------------
# End-to-end through the graph: PipelineSettingsTableCollection
# ---------------------------------------------------------------------------


class _SinkSettingsPokerSettings(ez.Settings):
    sink_filepath: Path
    target_recording: bool = False
    publish_after_s: float = 0.5
    terminate_after_s: float = 2.0


class _SinkSettingsPoker(ez.Unit):
    """Mid-run settings update on the sink, modeled on the Phase 1
    integration test. Triggers an UPDATED event from the producer."""

    SETTINGS = _SinkSettingsPokerSettings
    OUTPUT_SETTINGS = ez.OutputStream(NWBPipelineSettingsSinkSettings)

    @ez.publisher(OUTPUT_SETTINGS)
    async def poke(self) -> typing.AsyncGenerator:
        s = self.SETTINGS
        await asyncio.sleep(s.publish_after_s)
        yield (
            self.OUTPUT_SETTINGS,
            NWBPipelineSettingsSinkSettings(
                filepath=s.sink_filepath,
                overwrite_old=True,
                recording=s.target_recording,
                inc_clock=ReferenceClockType.UNKNOWN,
            ),
        )
        await asyncio.sleep(s.terminate_after_s)
        raise ez.NormalTermination


def test_collection_via_graph_records_initial_and_updated(tmp_path):
    """End-to-end: PipelineSettingsTableCollection drops into a graph,
    records INITIAL + an UPDATED row in the typed-column table."""
    from ezmsg.core.graphserver import GraphServer

    outpath = tmp_path / "collection_e2e.nwb"
    outpath.unlink(missing_ok=True)

    graph_address = ("127.0.0.1", _find_free_port())

    class _Settings(ez.Settings):
        coll: PipelineSettingsTableCollectionSettings
        poker: _SinkSettingsPokerSettings

    class _Pipeline(ez.Collection):
        SETTINGS = _Settings

        COLL = PipelineSettingsTableCollection()
        POKER = _SinkSettingsPoker()

        def configure(self) -> None:
            self.COLL.apply_settings(self.SETTINGS.coll)
            self.POKER.apply_settings(self.SETTINGS.poker)

        def network(self) -> ez.NetworkDefinition:
            return ((self.POKER.OUTPUT_SETTINGS, self.COLL.INPUT_SETTINGS),)

    system = _Pipeline(
        _Settings(
            coll=PipelineSettingsTableCollectionSettings(
                producer=PipelineSettingsProducerSettings(graph_address=graph_address),
                sink=NWBPipelineSettingsSinkSettings(
                    filepath=outpath,
                    overwrite_old=True,
                    recording=True,
                    inc_clock=ReferenceClockType.UNKNOWN,
                ),
            ),
            poker=_SinkSettingsPokerSettings(
                sink_filepath=outpath,
                target_recording=False,
                publish_after_s=0.5,
                terminate_after_s=1.5,
            ),
        )
    )

    server = GraphServer()
    server.start(graph_address)
    try:
        ez.run(SYSTEM=system, graph_address=graph_address)
    finally:
        server.stop()

    # With INIT_FINAL aggregation in place, the per-component INITIAL
    # events buffer in memory and flush as ONE merged anchor row when
    # the sentinel arrives — so we expect a single output file (no
    # rotation from the snapshot).
    base_stem = outpath.stem
    files = sorted(tmp_path.glob(f"{base_stem}*.nwb"))
    assert len(files) == 1, f"expected single output file; got {[f.name for f in files]}"

    with NWBHDF5IO(str(files[0]), "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    # Anchor row should cover every component in the session.
    # Column paths use dots (sanitize_settings_column_name converts the
    # graph's "/" separators), so e.g. "SYSTEM.COLL.PUB.target_table".
    cols = [c for c in df.columns if c not in ("start_time", "stop_time", "updated_component")]
    assert any(".PUB." in c for c in cols), cols
    assert any(".SINK." in c for c in cols), cols
    assert any(".POKER." in c for c in cols), cols

    # The SINK's recording flag should land True (initial), then flip to
    # False after the poker fires. The last row across all writes must
    # show False on the SINK's recording column.
    sink_recording_cols = [
        c for c in df.columns if c.endswith(".SINK.recording") and "POKER" not in c and "PUB" not in c
    ]
    assert sink_recording_cols, f"expected a SINK '.recording' column; got {list(df.columns)}"
    for col in sink_recording_cols:
        assert df[col].iloc[-1] in (False, "False", 0), f"expected {col}=False after poke; got {df[col].iloc[-1]!r}"
