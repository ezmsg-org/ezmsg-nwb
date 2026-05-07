"""Integration tests for ezmsg-nwb: ezmsg system tests and writer round-trip."""

import asyncio
import json
import tempfile
import typing
from dataclasses import field
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pynwb
import pytest
from ezmsg.baseproc.clock import Clock, ClockSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import (
    TerminateOnTimeout,
    TerminateOnTimeoutSettings,
    TerminateOnTotal,
    TerminateOnTotalSettings,
)
from pynwb import NWBHDF5IO

from ezmsg.nwb import (
    NWBAxisArrayIterator,
    NWBIteratorSettings,
    NWBSink,
    NWBSinkConsumer,
    NWBSinkSettings,
    ReferenceClockType,
)
from ezmsg.nwb.clockdriven import NWBClockDrivenSettings, NWBClockDrivenUnit
from ezmsg.nwb.reader import NWBIteratorUnit

# -- NWBIteratorUnit system test --


class NWBIteratorTestSettings(ez.Settings):
    nwbiterator_settings: NWBIteratorSettings
    log_settings: MessageLoggerSettings


class NWBIteratorIntegrationTest(ez.Collection):
    SETTINGS = NWBIteratorTestSettings

    SOURCE = NWBIteratorUnit()
    SINK = MessageLogger()

    def configure(self) -> None:
        self.SOURCE.apply_settings(self.SETTINGS.nwbiterator_settings)
        self.SINK.apply_settings(self.SETTINGS.log_settings)

    def network(self) -> ez.NetworkDefinition:
        return ((self.SOURCE.OUTPUT_SIGNAL, self.SINK.INPUT_MESSAGE),)


def test_nwbiterator_unit_system(test_nwb_path):
    """Test NWBIteratorUnit running in a full ezmsg system with MessageLogger sink."""
    log_file = Path(tempfile.gettempdir()) / "test_nwbiterator_unit.txt"
    log_file.write_text("")

    settings = NWBIteratorTestSettings(
        nwbiterator_settings=NWBIteratorSettings(
            filepath=test_nwb_path,
            chunk_dur=1.0,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Broadband"],
        ),
        log_settings=MessageLoggerSettings(output=log_file),
    )

    system = NWBIteratorIntegrationTest(settings)
    ez.run(SYSTEM=system)

    messages = list(message_log(log_file))
    log_file.unlink(missing_ok=True)

    assert len(messages) > 0
    assert all(isinstance(m, AxisArray) for m in messages)
    assert all(m.key == "Broadband" for m in messages)
    assert all(m.data.shape[1] == 8 for m in messages)

    total_samples = sum(m.data.shape[0] for m in messages)
    assert total_samples == 3000


# -- NWBClockDrivenUnit system test --


class NWBClockDrivenTestSettings(ez.Settings):
    clock_settings: ClockSettings
    nwb_settings: NWBClockDrivenSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class NWBClockDrivenIntegrationTest(ez.Collection):
    SETTINGS = NWBClockDrivenTestSettings

    CLOCK = Clock()
    NWB = NWBClockDrivenUnit()
    SINK = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.CLOCK.apply_settings(self.SETTINGS.clock_settings)
        self.NWB.apply_settings(self.SETTINGS.nwb_settings)
        self.SINK.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.NWB.INPUT_CLOCK),
            (self.NWB.OUTPUT_SIGNAL, self.SINK.INPUT_MESSAGE),
            (self.SINK.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


def test_nwb_clockdriven_unit_system(test_nwb_path):
    """Test NWBClockDrivenUnit in a Clock -> NWB -> Logger ezmsg system."""
    log_file = Path(tempfile.gettempdir()) / "test_nwb_clockdriven.txt"
    log_file.write_text("")

    # BinnedSpikes: 150 samples at 50 Hz, n_time=50 => 3 messages to exhaust
    n_time = 50
    target_messages = 3

    settings = NWBClockDrivenTestSettings(
        clock_settings=ClockSettings(dispatch_rate=float("inf")),
        nwb_settings=NWBClockDrivenSettings(
            filepath=test_nwb_path,
            stream_key="BinnedSpikes",
            fs=50.0,
            n_time=n_time,
            reference_clock=ReferenceClockType.UNKNOWN,
        ),
        log_settings=MessageLoggerSettings(output=log_file),
        term_settings=TerminateOnTotalSettings(total=target_messages),
    )

    system = NWBClockDrivenIntegrationTest(settings)
    ez.run(SYSTEM=system)

    messages = list(message_log(log_file))
    log_file.unlink(missing_ok=True)

    assert len(messages) >= target_messages
    messages = messages[:target_messages]

    assert all(isinstance(m, AxisArray) for m in messages)
    assert all(m.key == "BinnedSpikes" for m in messages)
    assert all(m.data.shape == (n_time, 4) for m in messages)

    total_samples = sum(m.data.shape[0] for m in messages)
    assert total_samples == 150


# -- Writer round-trip tests --


def test_writer_roundtrip_continuous(test_nwb_path):
    """Test writing continuous data and reading it back."""
    settings = NWBIteratorSettings(
        filepath=test_nwb_path,
        chunk_dur=0.1,
        reference_clock=ReferenceClockType.UNKNOWN,
        stream_keys=["Broadband"],
    )
    it = NWBAxisArrayIterator(settings)
    src_msgs = [next(it) for _ in range(3)]
    del it

    outpath = Path(tempfile.gettempdir()) / "ezmsg_nwb_roundtrip_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    for m in src_msgs:
        sink._process(m)
    sink.close(write=True)

    assert outpath.exists()

    # Read back
    read_settings = NWBIteratorSettings(
        filepath=outpath,
        chunk_dur=10.0,  # Large chunk to get all data in one message
        reference_clock=ReferenceClockType.UNKNOWN,
    )
    it2 = NWBAxisArrayIterator(read_settings)
    read_msgs = list(it2)

    outpath.unlink(missing_ok=True)

    total_written = sum(m.data.shape[0] for m in src_msgs)
    total_read = sum(m.data.shape[0] for m in read_msgs)
    assert total_read == total_written
    assert read_msgs[0].data.shape[1] == 8

    # Verify data content matches
    src_data = np.concatenate([m.data for m in src_msgs], axis=0)
    read_data = np.concatenate([m.data for m in read_msgs], axis=0)
    np.testing.assert_array_almost_equal(src_data, read_data)


def test_writer_empty_file_deleted():
    """Test that an NWB file with no data written is deleted on close."""
    outpath = Path(tempfile.gettempdir()) / "ezmsg_nwb_empty_test.nwb"
    outpath.unlink(missing_ok=True)

    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink.close(write=False)

    assert not outpath.exists(), "Empty NWB file should be deleted on close"


def test_writer_recording_toggle():
    """Test that toggling recording on/off works."""
    outpath = Path(tempfile.gettempdir()) / "ezmsg_nwb_toggle_test.nwb"
    outpath.unlink(missing_ok=True)

    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            recording=False,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    assert sink.settings.recording is False
    with pytest.warns(DeprecationWarning):
        sink.toggle_recording(True)
    assert sink.settings.recording is True
    with pytest.warns(DeprecationWarning):
        sink.toggle_recording()
    assert sink.settings.recording is False
    sink.close(write=False)
    outpath.unlink(missing_ok=True)


# -- Settings-push through running graph --


class EnableRecordingPubSettings(ez.Settings):
    sink_filepath: Path
    n_messages: int
    chunk_samples: int
    chunk_channels: int


class EnableRecordingPub(ez.Unit):
    """Pushes a settings update that flips ``recording`` from ``False`` to
    ``True``, waits for it to land at the sink, then publishes data. The
    inverse direction (``True → False``) races: NWB's first-message I/O
    takes seconds, so a fast settings flip beats every queued data message
    to the consumer and we observe nothing on disk. Going ``False → True``
    pushes the slow side ahead of the fast side, so a settled sleep is
    enough to guarantee ordering."""

    SETTINGS = EnableRecordingPubSettings

    OUTPUT_SETTINGS = ez.OutputStream(NWBSinkSettings)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.publisher(OUTPUT_SIGNAL)
    @ez.publisher(OUTPUT_SETTINGS)
    async def go(self) -> typing.AsyncGenerator:
        s = self.SETTINGS

        # Flip recording on. Same filepath → only ``recording`` differs →
        # NONRESET path (no file rebuild), settings just rebinds.
        yield (
            self.OUTPUT_SETTINGS,
            NWBSinkSettings(
                filepath=s.sink_filepath,
                overwrite_old=True,
                recording=True,
                inc_clock=ReferenceClockType.UNKNOWN,
            ),
        )
        # Give the settings subscriber time to process before any data hits
        # the data subscriber. Settings handling is fast (no I/O) so 1s is
        # plenty.
        await asyncio.sleep(1.0)

        for i in range(s.n_messages):
            yield (
                self.OUTPUT_SIGNAL,
                AxisArray(
                    data=np.ones((s.chunk_samples, s.chunk_channels), dtype=np.float32) * float(i),
                    dims=["time", "ch"],
                    axes={"time": AxisArray.TimeAxis(fs=1000.0, offset=i * 0.05)},
                    key="K",
                ),
            )
            await asyncio.sleep(0.05)

        # Wait long enough for the slow first-message NWB write to finish
        # before the runtime tears the graph down.
        await asyncio.sleep(3.0)
        raise ez.NormalTermination


class SettingsPushTestSettings(ez.Settings):
    pub: EnableRecordingPubSettings
    sink: NWBSinkSettings


class SettingsPushCollection(ez.Collection):
    SETTINGS = SettingsPushTestSettings

    PUB = EnableRecordingPub()
    SINK = NWBSink()

    def configure(self) -> None:
        self.PUB.apply_settings(self.SETTINGS.pub)
        self.SINK.apply_settings(self.SETTINGS.sink)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.PUB.OUTPUT_SIGNAL, self.SINK.INPUT_SIGNAL),
            (self.PUB.OUTPUT_SETTINGS, self.SINK.INPUT_SETTINGS),
        )


def test_settings_push_through_graph_enables_recording():
    """A NWBSinkSettings message published into ``NWBSink.INPUT_SETTINGS``
    while the graph is running should reach the consumer's
    ``update_settings`` and propagate to subsequent writes."""
    outpath = Path(tempfile.gettempdir()) / "ezmsg_nwb_settings_push.nwb"
    outpath.unlink(missing_ok=True)

    n_messages, chunk = 3, 10

    system = SettingsPushCollection(
        SettingsPushTestSettings(
            pub=EnableRecordingPubSettings(
                sink_filepath=outpath,
                n_messages=n_messages,
                chunk_samples=chunk,
                chunk_channels=2,
            ),
            # Sink starts with recording=False — without the in-graph
            # settings push, no data would land on disk.
            sink=NWBSinkSettings(
                filepath=outpath,
                overwrite_old=True,
                recording=False,
                inc_clock=ReferenceClockType.UNKNOWN,
            ),
        )
    )
    ez.run(SYSTEM=system)

    assert outpath.exists(), (
        "If the in-graph settings update never reached the consumer, "
        "recording stays False and the empty-file cleanup would have "
        "deleted this path on shutdown."
    )
    with pynwb.NWBHDF5IO(str(outpath), "r") as io:
        nwbfile = io.read()
        assert "K" in nwbfile.acquisition
        assert len(nwbfile.acquisition["K"].data) == n_messages * chunk

    outpath.unlink(missing_ok=True)


# -- Pipeline-settings table integration --


def _make_writer_continuous_msg() -> AxisArray:
    return AxisArray(
        data=np.arange(6, dtype=float).reshape(3, 2),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=100.0)},
        key="sig",
    )


def _make_writer_epochs_msg() -> AxisArray:
    return AxisArray(
        data=np.array([["a"], ["b"]], dtype="U"),
        dims=["time", "ch"],
        axes={"time": AxisArray.CoordinateAxis(np.array([0.0, 1.0]), dims=["time"], unit="s")},
        key="epochs",
    )


def test_writer_annotation_then_data_lands_in_acquisition(tmp_path):
    """Annotations written before data should still materialize on close."""
    outpath = tmp_path / "ezmsg_nwb_annotation_then_data_test.nwb"

    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink.write_annotation("settings_annotations", timestamp=0.5, data='{"step": "init"}')
    sink._process(_make_writer_continuous_msg())
    sink.write_annotation("settings_annotations", timestamp=1.5, data='{"step": "running"}')
    sink.close(write=False)

    assert outpath.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        series = nwbfile.acquisition["settings_annotations"]
        assert list(series.data[:]) == ['{"step": "init"}', '{"step": "running"}']


def _find_free_port() -> int:
    """Pick a free TCP port for an isolated GraphServer."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_writer_pipeline_settings_event_via_graph(tmp_path):
    """End-to-end: ``PipelineSettingsUnit`` publishes its session's settings
    snapshot through the graph and ``NWBSink`` lands the events in a
    ``settings_annotations`` AnnotationSeries.

    Uses the real producer (which opens a ``GraphContext`` against the
    running ``GraphServer``) so this exercises Phase 1 end-to-end rather
    than a mock publisher. We pre-start a ``GraphServer`` at a known free
    port and pin both ``ez.run`` and the producer to that address so the
    producer's ``GraphContext`` connects to the same server the runner is
    using. (When ``ez.run`` is given an explicit ``graph_address``,
    ``GraphService.ensure`` will *not* auto-start a server — it expects
    one already listening at that address. That's why we start one here.)
    """
    from ezmsg.baseproc import PipelineSettingsProducerSettings, PipelineSettingsUnit
    from ezmsg.core.graphserver import GraphServer

    outpath = tmp_path / "ezmsg_nwb_pipeline_settings_via_graph.nwb"
    outpath.unlink(missing_ok=True)

    graph_address = ("127.0.0.1", _find_free_port())

    class _Settings(ez.Settings):
        producer: PipelineSettingsProducerSettings
        sink: NWBSinkSettings
        term: TerminateOnTimeoutSettings

    class _Pipeline(ez.Collection):
        SETTINGS = _Settings

        PUB = PipelineSettingsUnit()
        SINK = NWBSink()
        TERM = TerminateOnTimeout()

        def configure(self) -> None:
            self.PUB.apply_settings(self.SETTINGS.producer)
            self.SINK.apply_settings(self.SETTINGS.sink)
            self.TERM.apply_settings(self.SETTINGS.term)

        def network(self) -> ez.NetworkDefinition:
            # Fan PUB's events out to the sink AND the terminator: the
            # terminator resets its idle clock on every event, then fires
            # ``NormalTermination`` after the producer goes quiet (no more
            # settings changes after the initial snapshot).
            return (
                (self.PUB.OUTPUT_SIGNAL, self.SINK.INPUT_ANNOTATION),
                (self.PUB.OUTPUT_SIGNAL, self.TERM.INPUT),
            )

    system = _Pipeline(
        _Settings(
            producer=PipelineSettingsProducerSettings(graph_address=graph_address),
            sink=NWBSinkSettings(
                filepath=outpath,
                overwrite_old=True,
                inc_clock=ReferenceClockType.UNKNOWN,
            ),
            term=TerminateOnTimeoutSettings(time=1.5),
        )
    )

    server = GraphServer()
    server.start(graph_address)
    try:
        ez.run(SYSTEM=system, graph_address=graph_address)
    finally:
        server.stop()

    assert outpath.exists()
    with NWBHDF5IO(str(outpath), "r") as io:
        nwbfile = io.read()
        series = nwbfile.acquisition["settings_annotations"]
        rows = [json.loads(s) for s in series.data[:]]

    # PipelineSettingsUnit emits one INITIAL row per in-scope component;
    # the session contains the Collection (SYSTEM) plus PUB + SINK + TERM.
    assert all(r["event_type"] == "INITIAL" for r in rows)
    components = {r["component"] for r in rows}
    # Addresses use ``/`` as the separator (e.g. "SYSTEM/PUB"). Don't pin
    # the root ("SYSTEM" only because we passed it that way to ez.run);
    # confirm the snapshot covers each unit.
    assert any(c.endswith("/PUB") for c in components)
    assert any(c.endswith("/SINK") for c in components)
    assert any(c.endswith("/TERM") for c in components)

    outpath.unlink(missing_ok=True)


def test_writer_event_append_after_reopen(tmp_path):
    """Epoch rows should remain appendable after the initial write/reopen cycle."""
    outpath = tmp_path / "ezmsg_nwb_event_append_test.nwb"

    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink._process(_make_writer_epochs_msg())
    sink.close(write=False)

    assert outpath.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.epochs.to_dataframe()

    assert df["label"].tolist() == ["EZNWB-START", "a", "b"]


class _SinkSettingsPokerSettings(ez.Settings):
    sink_filepath: Path
    target_recording: bool = False
    publish_after_s: float = 0.5
    terminate_after_s: float = 2.0


class _SinkSettingsPoker(ez.Unit):
    """Publish a ``NWBSinkSettings`` update to ``NWBSink.INPUT_SETTINGS``
    mid-run, then raise ``NormalTermination`` after a settle delay so the
    producer has time to emit the resulting UPDATED event."""

    SETTINGS = _SinkSettingsPokerSettings
    OUTPUT_SETTINGS = ez.OutputStream(NWBSinkSettings)

    @ez.publisher(OUTPUT_SETTINGS)
    async def poke(self) -> typing.AsyncGenerator:
        s = self.SETTINGS
        await asyncio.sleep(s.publish_after_s)
        # Flip the ``recording`` flag — same filepath, NONRESET field, so
        # the sink's file isn't disturbed; the only observable side-effect
        # is the graph server recording a SettingsChangedEvent for SINK.
        yield (
            self.OUTPUT_SETTINGS,
            NWBSinkSettings(
                filepath=s.sink_filepath,
                overwrite_old=True,
                recording=s.target_recording,
                inc_clock=ReferenceClockType.UNKNOWN,
            ),
        )
        await asyncio.sleep(s.terminate_after_s)
        raise ez.NormalTermination


def test_writer_pipeline_settings_updated_event_via_graph(tmp_path):
    """End-to-end: ``PipelineSettingsUnit`` emits an UPDATED event when a
    settings message is published into another unit's ``INPUT_SETTINGS``
    while the graph is running, and ``NWBSink`` lands it in the
    ``settings_annotations`` series alongside the INITIAL snapshot.

    The mid-run change is driven by ``_SinkSettingsPoker`` — when its
    ``NWBSinkSettings`` message lands at ``NWBSink.INPUT_SETTINGS``, the
    backend reports the new value to the graph server, which broadcasts a
    ``SettingsChangedEvent`` that the running ``PipelineSettingsProducer``
    receives via its subscription and forwards as an ``UPDATED`` event.
    """
    from ezmsg.baseproc import PipelineSettingsProducerSettings, PipelineSettingsUnit
    from ezmsg.core.graphserver import GraphServer

    outpath = tmp_path / "ezmsg_nwb_pipeline_settings_updated.nwb"
    outpath.unlink(missing_ok=True)

    graph_address = ("127.0.0.1", _find_free_port())

    class _Settings(ez.Settings):
        producer: PipelineSettingsProducerSettings
        sink: NWBSinkSettings
        poker: _SinkSettingsPokerSettings

    class _Pipeline(ez.Collection):
        SETTINGS = _Settings

        PUB = PipelineSettingsUnit()
        SINK = NWBSink()
        POKER = _SinkSettingsPoker()

        def configure(self) -> None:
            self.PUB.apply_settings(self.SETTINGS.producer)
            self.SINK.apply_settings(self.SETTINGS.sink)
            self.POKER.apply_settings(self.SETTINGS.poker)

        def network(self) -> ez.NetworkDefinition:
            return (
                (self.PUB.OUTPUT_SIGNAL, self.SINK.INPUT_ANNOTATION),
                (self.POKER.OUTPUT_SETTINGS, self.SINK.INPUT_SETTINGS),
            )

    system = _Pipeline(
        _Settings(
            producer=PipelineSettingsProducerSettings(graph_address=graph_address),
            sink=NWBSinkSettings(
                filepath=outpath,
                overwrite_old=True,
                recording=True,
                inc_clock=ReferenceClockType.UNKNOWN,
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

    assert outpath.exists()
    with NWBHDF5IO(str(outpath), "r") as io:
        nwbfile = io.read()
        rows = [json.loads(s) for s in nwbfile.acquisition["settings_annotations"].data[:]]

    initial = [r for r in rows if r["event_type"] == "INITIAL"]
    updated = [r for r in rows if r["event_type"] == "UPDATED"]

    # Initial snapshot covers the whole session.
    assert len(initial) >= 3
    initial_components = {r["component"] for r in initial}
    assert any(c.endswith("/PUB") for c in initial_components)
    assert any(c.endswith("/SINK") for c in initial_components)
    assert any(c.endswith("/POKER") for c in initial_components)

    # POKER's settings push lands as one (or more) UPDATED row whose
    # component is SINK and whose ``recording`` field is the new value.
    assert len(updated) >= 1
    sink_updates = [r for r in updated if r["component"].endswith("/SINK")]
    assert sink_updates, f"expected an UPDATED row for SINK; got components {sorted({r['component'] for r in updated})}"
    last = sink_updates[-1]
    assert last["settings"]["recording"] is False

    outpath.unlink(missing_ok=True)
