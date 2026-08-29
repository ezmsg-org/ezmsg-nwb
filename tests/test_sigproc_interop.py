"""The reported scaling has to survive stages that walk attrs generically.

``ezmsg-sigproc``'s ``concat`` merges two messages' attrs and rejects any value
that is not a scalar. A nested ``nwb_scaling`` dict raised ``TypeError`` there
even when both sides carried an identical one, so any concat touching an
NWB-sourced stream failed outright. Flat ``nwb_scaling_*`` keys fix that, and
give concat something it can act on: an unequal gain is promoted onto the ``ch``
axis, which is where a per-channel gain belongs.

Skipped unless the ``sigproc`` extra is installed.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ezmsg.nwb import GAIN_ATTR, UNIT_ATTR, StreamScaling

concat = pytest.importorskip("ezmsg.sigproc.concat", reason="requires the 'sigproc' extra")


def message(gain, key: str) -> "np.ndarray":
    from ezmsg.util.messages.axisarray import AxisArray

    return AxisArray(
        data=np.zeros((10, 2), dtype=np.int16),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=30000.0, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"{key}{i}" for i in range(2)]), dims=["ch"]),
        },
        attrs=StreamScaling(gain=gain, offset=0.0, unit="volts", applied=False, voltage=True).as_attrs(),
        key=key,
    )


def concatenate(a, b):
    processor = concat.ConcatProcessor(concat.ConcatSettings(axis="ch"))
    processor.push_a(a)
    processor.push_b(b)
    return asyncio.run(processor.__acall__())


def test_two_streams_with_the_same_scaling_concatenate():
    """The regression: this raised TypeError on the nested-dict form."""
    out = concatenate(message(0.25, "a"), message(0.25, "b"))

    assert out.data.shape == (10, 4)
    assert out.attrs[GAIN_ATTR] == 0.25
    assert out.attrs[UNIT_ATTR] == "volts"


def test_a_differing_gain_is_promoted_onto_the_ch_axis():
    """Two streams recorded at different gains describe their channels
    differently, so the gain stops being a property of the message and becomes
    one of each channel. A nested dict could never express that."""
    out = concatenate(message(0.25, "a"), message(0.50, "b"))

    assert GAIN_ATTR not in out.attrs
    assert GAIN_ATTR in (out.axes["ch"].data.dtype.names or ())
    np.testing.assert_allclose(out.axes["ch"].data[GAIN_ATTR], [0.25, 0.25, 0.50, 0.50])
    # The keys they agree on stay put.
    assert out.attrs[UNIT_ATTR] == "volts"


def test_a_vector_gain_is_still_rejected():
    """Known hole, left deliberately: ``_ALLOWED_ATTR_SCALARS`` has no ndarray,
    so a genuinely per-channel gain cannot ride in attrs at all. Only reachable
    when a file's channel_conversion entries differ -- uniform ones collapse to
    a scalar in ``resolve_scaling``. Tracked upstream in ezmsg-sigproc."""
    with pytest.raises(TypeError, match="ndarray"):
        concatenate(message(np.array([0.25, 0.50]), "a"), message(np.array([0.25, 0.50]), "b"))
