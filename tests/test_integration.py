"""Integration tests for ezmsg-nwb: ezmsg system tests and writer round-trip."""

import tempfile
from dataclasses import field
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.baseproc.clock import Clock, ClockSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.nwb import (
    NWBAxisArrayIterator,
    NWBIteratorSettings,
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
