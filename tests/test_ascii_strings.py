"""Reading a file whose strings were declared ASCII rather than UTF-8.

hdmf keys its string-decoding decision directly off the character set, so a
writer that declares ``H5T_CSET_ASCII`` -- aqnwb, and therefore every recording
Orion produces -- hands every string back as ``bytes``. Nothing raises: the
values simply stop matching, stop parsing, and stop being the text they say
they are, at whatever distance downstream first depends on it.

See https://github.com/NeurodataWithoutBorders/aqnwb/issues/316. Fixing that
upstream will not retire these tests -- files already on disk keep their
character set, and a reader has to survive both.
"""

import json

import numpy as np
import pytest
from conftest import (
    ASCII_MANUFACTURER,
    ASCII_MARKER_TIMES,
    ASCII_MARKERS,
    ASCII_N_CHANNELS,
)
from pynwb import NWBHDF5IO

from ezmsg.nwb.slicer import NWBSlicer
from ezmsg.nwb.util import ReferenceClockType, as_text, as_text_array

# --- The helpers themselves --------------------------------------------------


class TestAsText:
    def test_bytes_become_the_text_they_hold(self):
        assert as_text(b"elec1-m1-3") == "elec1-m1-3"

    def test_str_passes_through(self):
        assert as_text("elec1-m1-3") == "elec1-m1-3"

    def test_the_repr_is_what_str_would_have_given(self):
        # Stating the trap the helper exists to avoid: str() on bytes yields a
        # value that looks like text, compares equal to nothing, and parses as
        # no JSON.
        assert str(b"elec1-m1-3") == "b'elec1-m1-3'"

    def test_a_fixed_length_column_becomes_unicode(self):
        decoded = as_text_array(np.array([b"aa", b"bb"], dtype="S8"))
        assert decoded.dtype.kind == "U"
        assert decoded.tolist() == ["aa", "bb"]

    def test_an_object_array_of_bytes_becomes_unicode(self):
        decoded = as_text_array(np.array([b"aa", b"bb"], dtype=object))
        assert decoded.dtype.kind == "U"
        assert decoded.tolist() == ["aa", "bb"]

    def test_a_unicode_array_is_returned_as_is(self):
        original = np.array(["aa", "bb"])
        assert as_text_array(original) is original

    def test_shape_survives(self):
        assert as_text_array(np.array([[b"a", b"b"], [b"c", b"d"]], dtype="S4")).shape == (2, 2)

    def test_an_empty_column_is_still_unicode(self):
        assert as_text_array(np.array([], dtype="S8")).dtype.kind == "U"


# --- The fixture ------------------------------------------------------------


def test_the_fixture_really_reads_back_as_bytes(ascii_nwb_path):
    """Guards the guard.

    Were these to read back as ``str``, every test below would pass against
    the very bug it exists to catch -- which is what the rest of this suite,
    written by pynwb throughout, does today.
    """
    with NWBHDF5IO(str(ascii_nwb_path), "r") as io:
        nwbfile = io.read()
        assert isinstance(nwbfile.electrodes["label"][0], bytes)
        assert isinstance(nwbfile.trials["condition"][0], bytes)
        assert isinstance(nwbfile.acquisition["Markers"].data[0], bytes)


def test_attributes_come_back_decoded_even_though_datasets_do_not(ascii_nwb_path):
    """Pins the asymmetry the rest of this file is about.

    The fixture writes ``manufacturer`` as an ASCII attribute exactly as it
    writes the columns above, yet hdmf decodes attributes on the way out and
    leaves datasets alone. Worth stating: it is the only reason stream-key
    matching survived a bug that hit every column beside it.
    """
    with NWBHDF5IO(str(ascii_nwb_path), "r") as io:
        assert isinstance(io.read().devices["TestArray"].manufacturer, str)


# --- What the slicer hands downstream ---------------------------------------


@pytest.fixture
def ascii_slicer(ascii_nwb_path):
    s = NWBSlicer(filepath=ascii_nwb_path, reference_clock=ReferenceClockType.UNKNOWN)
    yield s
    s.close()


class TestIntervalTableColumns:
    def test_event_payloads_are_text_not_reprs(self, ascii_slicer):
        events = ascii_slicer.read_by_time("trials", 0.0, 10.0)
        column = list(events.axes["ch"].data).index("condition")

        assert [str(v) for v in events.data[:, column]] == ["cond_0", "cond_1", "cond_0"]

    def test_no_value_smuggles_a_bytes_repr_through(self, ascii_slicer):
        events = ascii_slicer.read_by_time("trials", 0.0, 10.0)

        assert not [v for v in events.data.ravel() if str(v).startswith("b'")]


class TestChannelLabels:
    def test_labels_are_the_names_a_selection_would_match(self, ascii_slicer):
        stream = ascii_slicer.read_by_index(f"{ASCII_MANUFACTURER}_NPLAY", 0, 10)

        assert stream.axes["ch"].data.dtype.kind == "U"
        assert list(stream.axes["ch"].data) == [f"elec1-m1-{i}" for i in range(ASCII_N_CHANNELS)]


class TestMarkerSeries:
    def test_marker_payloads_parse_as_the_json_they_are(self, ascii_slicer):
        markers = ascii_slicer.read_by_time("Markers", 0.0, 10.0)

        assert [json.loads(str(v))["cause"]["event"] for v in markers.data] == [
            json.loads(m)["cause"]["event"] for m in ASCII_MARKERS
        ]

    def test_every_marker_in_the_window_arrives(self, ascii_slicer):
        markers = ascii_slicer.read_by_time("Markers", 0.0, 10.0)

        assert markers.data.shape[0] == len(ASCII_MARKER_TIMES)


class TestManufacturerPrefixedStreamKeys:
    """A bare device name in ``stream_keys`` has to match ``"<manufacturer>_<key>"``.

    The match parses ``Device.manufacturer`` as a prefix; were that value ever
    to arrive as bytes, the stream would be discarded in silence and the caller
    would get an empty file rather than an error.
    """

    def test_a_bare_device_name_still_matches(self, ascii_nwb_path):
        slicer = NWBSlicer(
            filepath=ascii_nwb_path,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["NPLAY"],
        )
        try:
            assert slicer.stream_names == ["NPLAY"]
        finally:
            slicer.close()
