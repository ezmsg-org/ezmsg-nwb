"""An ``AnnotationSeries`` is an event stream, not a 1 Hz recording.

``pynwb.misc.AnnotationSeries`` subclasses ``TimeSeries``, so discovery used to
route it down the continuous path, where the rate heuristic landed it on
``rate=0.0`` by accident. Two things then went wrong, both on real session
files: the ``stop_time`` extrapolation read the 1.0 s fallback ``gain`` as a
sample period and stretched N markers into an N-second recording, and the
iterator read ``.gain`` off the resulting ``CoordinateAxis`` and raised --
taking the whole file down, not just the marker stream.

The ``ascii_nwb_path`` fixture carries a ``Markers`` AnnotationSeries at
t = 0.5/1.0/1.5 s alongside a 2 s ElectricalSeries and a trials table ending at
2.8 s, so an inflated span is distinguishable from an honest one.
"""

import datetime
import json

import numpy as np
import pytest
from conftest import ASCII_MANUFACTURER, ASCII_MARKER_TIMES, ASCII_MARKERS

from ezmsg.nwb import NWBAxisArrayIterator, NWBIteratorSettings, ReferenceClockType
from ezmsg.nwb.slicer import NWBSlicer

# Deliberately uneven, and that is the whole point. Evenly-spaced markers (the
# ``ascii_nwb_path`` fixture's 0.5/1.0/1.5) sail through the rate heuristic as a
# 2 Hz stream and never reach the rate-0 path where the bugs lived. Real session
# markers arrive when the task says so, so their spacing has high variance and
# the heuristic gives up with ``rate=0.0``.
IRREGULAR_TIMES = [0.2, 0.9, 1.05, 2.4, 2.95]
IRREGULAR_LABELS = [f"marker-{i}" for i in range(len(IRREGULAR_TIMES))]
IRREGULAR_RATE = 100.0
IRREGULAR_N_SAMPLES = 300  # 3.0 s of continuous data


@pytest.fixture(scope="module")
def irregular_markers_nwb_path(tmp_path_factory):
    """A continuous stream plus irregularly-timed annotations, as sessions have."""
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.misc import AnnotationSeries

    path = tmp_path_factory.mktemp("irregular") / "irregular_markers.nwb"
    nwbfile = NWBFile(
        session_description="irregularly-timed annotations",
        identifier="irregular001",
        session_start_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    nwbfile.add_acquisition(
        TimeSeries(
            name="Continuous",
            data=np.arange(IRREGULAR_N_SAMPLES, dtype=np.float32),
            unit="V",
            rate=IRREGULAR_RATE,
            starting_time=0.0,
        )
    )
    nwbfile.add_acquisition(
        AnnotationSeries(name="Markers", data=list(IRREGULAR_LABELS), timestamps=list(IRREGULAR_TIMES))
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return path


@pytest.fixture
def irregular_slicer(irregular_markers_nwb_path):
    s = NWBSlicer(filepath=irregular_markers_nwb_path, reference_clock=ReferenceClockType.UNKNOWN)
    yield s
    s.close()


@pytest.fixture
def slicer(ascii_nwb_path):
    s = NWBSlicer(filepath=ascii_nwb_path, reference_clock=ReferenceClockType.UNKNOWN)
    yield s
    s.close()


async def _drain(producer, key):
    """Every message ``producer`` emits for ``key``, concatenated in order."""
    payloads, times = [], []
    while not producer.exhausted:
        msg = await producer.__anext__()
        if msg.key == key and msg.data.size:
            payloads.append(msg.data)
            times.append(msg.axes["time"].data)
    if not payloads:
        return np.array([]), np.array([])
    return np.concatenate(payloads), np.concatenate(times)


class TestClassification:
    def test_it_is_discovered_as_an_event_stream(self, slicer):
        info = slicer.get_stream_info("Markers")

        assert info.is_event
        assert info.fs == 0.0

    def test_it_carries_its_own_times_rather_than_a_backing_table(self, slicer):
        """The event read paths must not reach into ``table_ref``: an annotation
        series is an event stream with no table behind it, unlike TimeIntervals.
        """
        info = slicer.get_stream_info("Markers")

        assert info.table_ref is None
        assert np.array_equal(np.asarray(info.timestamps), ASCII_MARKER_TIMES)


class TestFileSpan:
    def test_sparse_markers_do_not_stretch_the_file(self, irregular_slicer):
        """Five markers are five markers, not a five-second recording.

        The old ``t0 + (n + 1) * gain`` extrapolation read the 1.0 s fallback
        ``gain`` as a sample period and put stop_time at 0.2 + 6 = 6.2 s. The
        real end of this file is the continuous stream just past 3.0 s. On a
        2187-marker session the same arithmetic reported 2191 s for a 309 s
        recording.
        """
        assert irregular_slicer.stop_time == pytest.approx(3.01)

    def test_the_chunk_count_follows_the_real_span(self, irregular_markers_nwb_path):
        """7 chunks for a 3-second file meant ~half of them were empty; at
        session scale it was 2191 chunks for 310 seconds of data."""
        producer = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=irregular_markers_nwb_path,
                chunk_dur=1.0,
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )

        assert producer._state.n_chunks == 4


class TestSlicerReads:
    def test_read_by_time_returns_the_markers_in_the_window(self, slicer):
        markers = slicer.read_by_time("Markers", 0.0, 1.2)

        assert [json.loads(str(v))["cause"]["event"] for v in markers.data] == ["trial-start", "go-cue"]
        assert np.array_equal(markers.axes["time"].data, [0.5, 1.0])

    def test_read_by_index_does_not_raise_on_a_rateless_stream(self, slicer):
        """``replace(CoordinateAxis, offset=...)`` used to raise TypeError here."""
        markers = slicer.read_by_index("Markers", 0, 2)

        assert markers.data.shape[0] == 2
        assert np.array_equal(markers.axes["time"].data, [0.5, 1.0])


class TestIterator:
    def test_a_file_containing_one_can_be_opened(self, irregular_markers_nwb_path):
        """The regression: ``.gain`` on the marker stream's CoordinateAxis raised
        during ``_preload``, so *no* stream in the file was readable -- opening a
        session file with annotations failed outright, ephys included.
        """
        producer = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=irregular_markers_nwb_path,
                chunk_dur=1.0,
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )

        assert "Markers" in producer._state.streams
        assert "Continuous" in producer._state.streams

    async def test_irregular_markers_all_arrive_with_their_true_times(self, irregular_markers_nwb_path):
        producer = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=irregular_markers_nwb_path,
                chunk_dur=1.0,
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )

        payloads, times = await _drain(producer, "Markers")

        assert [str(v) for v in payloads] == IRREGULAR_LABELS
        assert np.array_equal(times, IRREGULAR_TIMES)

    async def test_every_marker_arrives_in_order(self, ascii_nwb_path):
        producer = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=ascii_nwb_path,
                chunk_dur=1.0,
                stream_keys=["Markers"],
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )

        payloads, times = await _drain(producer, "Markers")

        assert [json.loads(str(v))["cause"]["event"] for v in payloads] == [
            json.loads(m)["cause"]["event"] for m in ASCII_MARKERS
        ]
        assert np.array_equal(times, ASCII_MARKER_TIMES)

    async def test_it_agrees_with_the_slicer(self, ascii_nwb_path, slicer):
        """Same rows, same times, whichever way you read them."""
        reference = slicer.read_by_time("Markers", 0.0, 10.0)
        producer = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=ascii_nwb_path,
                chunk_dur=1.0,
                stream_keys=["Markers"],
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )

        payloads, times = await _drain(producer, "Markers")

        assert np.array_equal(payloads, reference.data)
        assert np.array_equal(times, reference.axes["time"].data)

    async def test_markers_interleave_with_the_continuous_stream(self, ascii_nwb_path):
        """Chunk 0 covers [0, 1): one marker at 0.5 s, plus that second of ephys."""
        producer = NWBAxisArrayIterator(
            NWBIteratorSettings(
                filepath=ascii_nwb_path,
                chunk_dur=1.0,
                reference_clock=ReferenceClockType.UNKNOWN,
            )
        )

        first_chunk = [await producer.__anext__()]
        while producer._state.deque:
            first_chunk.append(await producer.__anext__())

        marker_times = [m.axes["time"].data[0] for m in first_chunk if m.key == "Markers" and m.data.size]
        assert marker_times == [0.5]
        assert any(m.key == f"{ASCII_MANUFACTURER}_NPLAY" and m.data.size for m in first_chunk)
