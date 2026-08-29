"""Building a live-shaped ``ch`` axis from an NWB electrodes table.

The point of the feature is that a graph cannot tell whether its channel
metadata came from a device or a file, so the tests are mostly about matching:
the dtype, the field values, and the array-identity rule.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest
from conftest import ELEC_N_CH, ELEC_PITCH

from ezmsg.nwb import (
    CHANNEL_DTYPE,
    NWBAxisArrayIterator,
    NWBIteratorSettings,
    NWBSlicer,
    ReferenceClockType,
    array_identity,
    has_channel_metadata,
)

LIVE_SOURCE = "/Users/chad/Work/Tools/Neurophys/EZMSG/ezmsg-blackrock/src/ezmsg/blackrock/channel_map.py"


def ch_axis(path, key="Mapped", **kw):
    slicer = NWBSlicer(path, dejitter=False, structured_ch_axis=True, **kw)
    try:
        return slicer.get_stream_info(key).template.axes["ch"]
    finally:
        slicer.close()


# --- Matching the live source ------------------------------------------------


def test_dtype_matches_the_live_source_exactly():
    """The whole feature rests on this. Parsed out of the blackrock source
    rather than imported, because that package needs ``pycbsdk`` and is not a
    dependency here -- the check should not quietly skip when it is absent."""
    try:
        source = open(LIVE_SOURCE).read()
    except OSError:
        pytest.skip("ezmsg-blackrock checkout not available")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CHANNEL_DTYPE":
            live = np.dtype(ast.literal_eval(node.value.args[0]))
            assert CHANNEL_DTYPE == live
            assert CHANNEL_DTYPE.names == live.names
            return
    pytest.fail("CHANNEL_DTYPE not found in the live source")


@pytest.mark.parametrize(
    ("label", "headstage", "bank", "expected"),
    [
        ("elec1-m1-63", 1, "A", "hs1-elec1"),
        ("elec1-m1-63", 2, "A", "hs2-elec1"),
        ("elec1-m1-63", 0, "A", "elec1"),
        # No connector structure: fall back to the bank, so grouping degrades to
        # bank-level rather than collapsing every such channel into one cluster.
        ("chan1", 1, "A", "hs1-bankA"),
        ("", 0, "B", "bankB"),
        ("", 0, "", ""),
    ],
)
def test_array_identity_rule(label, headstage, bank, expected):
    assert array_identity(label, headstage, bank) == expected


# --- Building from a table ---------------------------------------------------


def test_fields_come_from_the_expected_columns(mapped_nwb_path):
    axis = ch_axis(mapped_nwb_path)
    ch = axis.data
    assert ch.dtype == CHANNEL_DTYPE
    assert axis.unit == "struct"
    assert ch.shape == (ELEC_N_CH,)

    # Geometry: rel_x/rel_y taken verbatim as micrometres, rounded to int32.
    np.testing.assert_array_equal(ch["x"], [(i % 2) * ELEC_PITCH for i in range(ELEC_N_CH)])
    np.testing.assert_array_equal(ch["y"], [(i // 2) * ELEC_PITCH for i in range(ELEC_N_CH)])
    assert set(ch["size"].tolist()) == {ELEC_PITCH}

    # bank is stored 1-based and numeric; the live source uses a letter.
    # The fixture puts four channels in bank 1 and four in bank 2.
    np.testing.assert_array_equal(ch["bank"], list("AAAABBBB"))
    np.testing.assert_array_equal(ch["elec"], [i % 4 + 1 for i in range(ELEC_N_CH)])
    np.testing.assert_array_equal(ch["headstage"], [1, 1, 1, 1, 1, 1, 2, 2])
    assert ch["label"][0] == "elec1-m1-1"


def test_array_identity_separates_connectors_and_headstages(mapped_nwb_path):
    """Four channels on hs1/elec1, two on hs1/elec2, two on hs2/elec1 -- and the
    last group must not merge with the first despite sharing a connector name."""
    ch = ch_axis(mapped_nwb_path).data
    np.testing.assert_array_equal(
        ch["array"],
        ["hs1-elec1"] * 4 + ["hs1-elec2"] * 2 + ["hs2-elec1"] * 2,
    )


def test_the_region_is_honoured(mapped_nwb_path):
    """A series over a subset, in its own order, gets that subset in that order
    -- not the first N rows of the table."""
    ch = ch_axis(mapped_nwb_path, key="Subset").data
    assert ch.shape == (3,)
    # Region is [5, 1, 0]: rows 5 and 1 and 0 of the layout above.
    np.testing.assert_array_equal(ch["label"], ["elec2-dlpfc-6", "elec1-m1-2", "elec1-m1-1"])
    np.testing.assert_array_equal(ch["array"], ["hs1-elec2", "hs1-elec1", "hs1-elec1"])
    np.testing.assert_array_equal(ch["headstage"], [1, 1, 1])


def test_off_by_default(mapped_nwb_path):
    """The columns read here are an acquisition-stack convention, so a file that
    happens to have them still gets plain labels unless asked."""
    slicer = NWBSlicer(mapped_nwb_path, dejitter=False)
    try:
        ch = slicer.get_stream_info("Mapped").template.axes["ch"].data
    finally:
        slicer.close()
    assert ch.dtype.names is None
    assert ch.dtype.kind == "U"
    assert ch[0] == "elec1-m1-1"


def test_a_file_without_the_columns_raises_rather_than_inventing(test_nwb_path):
    """The request was explicit, so answering it with a record of zeros would be
    fabricating geometry. The message names what was looked for."""
    with pytest.raises(ValueError, match="none of the columns it is built from"):
        NWBSlicer(test_nwb_path, dejitter=False, structured_ch_axis=True)


def test_has_channel_metadata_ignores_a_label_only_table(test_nwb_path, mapped_nwb_path):
    """``label`` alone is what the plain path already provides, so it does not
    count as something to build a structured axis from."""
    import pynwb

    for path, expected in ((test_nwb_path, False), (mapped_nwb_path, True)):
        with pynwb.NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            series = next(iter(nwb.acquisition.values()))
            table = series.electrodes.table
            assert has_channel_metadata(table) is expected


# --- Through the iterator ----------------------------------------------------


def test_iterator_messages_carry_the_structured_axis(mapped_nwb_path):
    it = NWBAxisArrayIterator(
        settings=NWBIteratorSettings(
            filepath=str(mapped_nwb_path),
            chunk_dur=0.1,
            reference_clock=ReferenceClockType.UNKNOWN,
            stream_keys=["Mapped"],
            dejitter=False,
            structured_ch_axis=True,
        )
    )
    try:
        msg = next(m for m in it if m.data.shape[0])
    finally:
        it.close()
    ch = msg.axes["ch"].data
    assert ch.dtype == CHANNEL_DTYPE
    assert msg.data.shape[1] == ch.shape[0] == ELEC_N_CH
    np.testing.assert_array_equal(ch["array"][:4], ["hs1-elec1"] * 4)
