"""Tests for NWB writer module."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from pynwb import NWBHDF5IO

from ezmsg.nwb import NWBSinkConsumer, NWBSinkSettings, ReferenceClockType
from ezmsg.nwb.util import flatten_component_settings


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


def test_sink_close_is_idempotent_after_settings_activation(tmp_path):
    """Closing twice after settings logging is active should be a no-op on the second call."""
    outpath = tmp_path / "ezmsg_nwb_double_close_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings_state({"component_foo": "bar"}, timestamp=1.0)
    sink._process(
        AxisArray(
            data=np.arange(6, dtype=float).reshape(3, 2),
            dims=["time", "ch"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=100.0)},
            key="sig",
        )
    )

    sink.close(write=False)
    sink.close(write=False)

    assert outpath.exists()


def test_sink_serializes_none_settings_values(tmp_path):
    """None-valued settings should be serialized so pipeline settings shutdown does not fail."""
    outpath = tmp_path / "ezmsg_nwb_none_settings_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings_state(flatten_component_settings({"component_optional": None}), timestamp=1.0)
    sink._process(
        AxisArray(
            data=np.arange(6, dtype=float).reshape(3, 2),
            dims=["time", "ch"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=100.0)},
            key="sig",
        )
    )
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    assert df["component_optional"].tolist() == ["None", "None"]


def test_sink_serializes_list_settings_values(tmp_path):
    """List-valued settings should be serialized into scalar table cells."""
    outpath = tmp_path / "ezmsg_nwb_list_settings_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings_state(flatten_component_settings({"component_list": [1, 2]}), timestamp=1.0)
    sink._process(
        AxisArray(
            data=np.arange(6, dtype=float).reshape(3, 2),
            dims=["time", "ch"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=100.0)},
            key="sig",
        )
    )
    sink.close(write=False)

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()

    assert df["component_list"].tolist() == ["[1, 2]", "[1, 2]"]
