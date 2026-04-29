"""Integration tests for ezmsg-nwb: ezmsg system tests and writer round-trip."""

import asyncio
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
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings
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


def _make_writer_continuous_msg() -> AxisArray:
    return AxisArray(
        data=np.arange(6, dtype=float).reshape(3, 2),
        dims=["time", "ch"],
        axes={"time": AxisArray.Axis.TimeAxis(fs=100.0)},
        key="sig",
    )


def _make_writer_epochs_msg() -> AxisArray:
    return AxisArray(
        data=np.array([["a"], ["b"]], dtype="U"),
        dims=["time", "ch"],
        axes={"time": AxisArray.CoordinateAxis(np.array([0.0, 1.0]), dims=["time"], unit="s")},
        key="epochs",
    )


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


def test_writer_pipeline_settings_shutdown_persist(tmp_path):
    """Pipeline settings created before first data should persist and append on close."""
    outpath = tmp_path / "ezmsg_nwb_settings_shutdown_test.nwb"

    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink.initialize_settings_table({"component_foo": "bar"}, timestamp=1.0)
    sink._process(_make_writer_continuous_msg())
    sink.close(write=False)

    assert outpath.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        table = nwbfile.intervals["pipeline_settings"]
        df = table.to_dataframe()

    assert df["updated_component"].tolist() == ["__init__", "__init__"]
    assert df["component_foo"].tolist() == ["bar", "bar"]


def test_writer_pipeline_settings_created_after_first_flush(tmp_path):
    """Pipeline settings created after the first stream flush should still be materialized."""
    outpath = tmp_path / "ezmsg_nwb_settings_late_create_test.nwb"

    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink._process(_make_writer_continuous_msg())
    sink.initialize_settings_table({"component_foo": "bar"}, timestamp=1.0)
    sink.close(write=False)

    assert outpath.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        assert nwbfile.intervals is not None
        table = nwbfile.intervals["pipeline_settings"]
        df = table.to_dataframe()

    assert df["updated_component"].tolist() == ["__init__", "__init__"]
    assert df["component_foo"].tolist() == ["bar", "bar"]


def test_writer_pipeline_settings_shape_change_rotates_file(tmp_path):
    """An incompatible settings update should rotate to a new file instead of erroring on close."""
    outpath = tmp_path / "ezmsg_nwb_settings_rotate_integration_test.nwb"
    rotated_path = tmp_path / "ezmsg_nwb_settings_rotate_integration_test_01.nwb"

    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )
    sink.initialize_settings_table({"component_list": [[1, 2], [3, 4]]}, timestamp=1.0)
    sink._process(_make_writer_continuous_msg())
    sink.update_settings_table("component", {"component_list": [[1, 2], [3, 4], [5, 6]]}, timestamp=2.0)
    sink.close(write=False)

    assert outpath.exists()
    assert rotated_path.exists()

    with NWBHDF5IO(rotated_path, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    np.testing.assert_array_equal(df["component_list"].iloc[-1], np.array([[1, 2], [3, 4], [5, 6]]))


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
