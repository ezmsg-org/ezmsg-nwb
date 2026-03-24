"""Tests for NWB writer module."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import ezmsg.core as ez
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

    sink.initialize_settings({"component_foo": "bar"}, timestamp=1.0)
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

    sink.initialize_settings(flatten_component_settings("component", {"component_optional": None}), timestamp=1.0)
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

    assert df["component.component_optional"].tolist() == ["None", "None"]


def test_sink_serializes_list_settings_values(tmp_path):
    """List-valued settings should be preserved as arrays in table cells."""
    outpath = tmp_path / "ezmsg_nwb_list_settings_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings({"component_list": [1, 2]}, timestamp=1.0)
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

    np.testing.assert_array_equal(df["component_list"].iloc[0], np.array([1, 2]))
    np.testing.assert_array_equal(df["component_list"].iloc[1], np.array([1, 2]))


@dataclass
class _Endpoint:
    host: str = "127.0.0.1"
    port: int = 5000


class _NestedSettings(ez.Settings):
    endpoint: _Endpoint = field(default_factory=_Endpoint)
    route_map: dict[str, str] = field(default_factory=lambda: {"open": "A", "closed": "B"})
    bands: list[tuple[float, float]] = field(default_factory=lambda: [(70.0, 200.0)])


def test_flatten_component_settings_flattens_dataclasses_and_mappings():
    """Structured setting values should become native NWB-compatible columns where possible."""
    flat = flatten_component_settings("PIPELINE.UNIT", _NestedSettings())

    assert flat["PIPELINE.UNIT._NestedSettings.endpoint.host"] == "127.0.0.1"
    assert flat["PIPELINE.UNIT._NestedSettings.endpoint.port"] == 5000
    assert flat["PIPELINE.UNIT._NestedSettings.route_map.open"] == "A"
    assert flat["PIPELINE.UNIT._NestedSettings.route_map.closed"] == "B"
    np.testing.assert_array_equal(flat["PIPELINE.UNIT._NestedSettings.bands"], np.array([[70.0, 200.0]]))


class _Mode(Enum):
    TRAIN = "train"


class _EdgeCaseSettings(ez.Settings):
    mode: _Mode = _Mode.TRAIN
    config_path: Path = Path("/tmp/model.pt")
    sample_count: np.int64 = np.int64(7)
    labels: set[str] = field(default_factory=lambda: {"b", "a"})


def test_flatten_component_settings_sanitizes_common_edge_types():
    """Enums, paths, NumPy scalars, and sets should normalize predictably."""
    flat = flatten_component_settings("PIPELINE.EDGE", _EdgeCaseSettings())

    assert flat["PIPELINE.EDGE._EdgeCaseSettings.mode"] == "train"
    assert flat["PIPELINE.EDGE._EdgeCaseSettings.config_path"] == "/tmp/model.pt"
    assert flat["PIPELINE.EDGE._EdgeCaseSettings.sample_count"] == 7
    np.testing.assert_array_equal(flat["PIPELINE.EDGE._EdgeCaseSettings.labels"], np.array(["a", "b"]))


def test_sink_updates_list_settings_with_same_shape(tmp_path):
    """Fixed-shape list settings should remain appendable across updates."""
    outpath = tmp_path / "ezmsg_nwb_list_settings_update_test.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings({"component_list": [[1, 2], [3, 4]]}, timestamp=1.0)
    sink.update_settings("component", {"component_list": [[5, 6], [7, 8]]}, timestamp=2.0)
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

    np.testing.assert_array_equal(df["component_list"].iloc[0], np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(df["component_list"].iloc[-1], np.array([[5, 6], [7, 8]]))


def test_sink_rotates_file_on_incompatible_list_shape_update(tmp_path):
    """A settings shape change should close the current file and continue in a new segment."""
    outpath = tmp_path / "ezmsg_nwb_settings_rotate_test.nwb"
    rotated_path = tmp_path / "ezmsg_nwb_settings_rotate_test_01.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings({"component_list": [[1, 2], [3, 4]]}, timestamp=1.0)
    sink._process(
        AxisArray(
            data=np.arange(4, dtype=float).reshape(2, 2),
            dims=["time", "ch"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=100.0)},
            key="sig",
        )
    )

    sink.update_settings("component", {"component_list": [[1, 2], [3, 4], [5, 6]]}, timestamp=2.0)
    sink._process(
        AxisArray(
            data=np.arange(4, 8, dtype=float).reshape(2, 2),
            dims=["time", "ch"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=100.0, offset=2.0)},
            key="sig",
        )
    )
    sink.close(write=False)

    assert outpath.exists()
    assert rotated_path.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df["component_list"].iloc[-1], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(
            np.asarray(nwbfile.acquisition["sig"].data[:]), np.arange(4, dtype=float).reshape(2, 2)
        )

    with NWBHDF5IO(rotated_path, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df["component_list"].iloc[-1], np.array([[1, 2], [3, 4], [5, 6]]))
        np.testing.assert_array_equal(
            np.asarray(nwbfile.acquisition["sig"].data[:]),
            np.arange(4, 8, dtype=float).reshape(2, 2),
        )


def test_sink_rotates_file_on_scalar_to_array_update(tmp_path):
    """A scalar-to-array settings transition should rotate to a new file segment."""
    outpath = tmp_path / "ezmsg_nwb_settings_scalar_rotate_test.nwb"
    rotated_path = tmp_path / "ezmsg_nwb_settings_scalar_rotate_test_01.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings({"component_value": "foo"}, timestamp=1.0)
    sink.update_settings("component", {"component_value": [1, 2, 3]}, timestamp=2.0)
    sink.close(write=False)

    assert outpath.exists()
    assert rotated_path.exists()

    with NWBHDF5IO(outpath, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
        assert df["component_value"].iloc[-1] == "foo"

    with NWBHDF5IO(rotated_path, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df["component_value"].iloc[-1], np.array([1, 2, 3]))


def test_sink_rotates_file_on_rank_change_update(tmp_path):
    """A settings rank change should rotate to a new file segment."""
    outpath = tmp_path / "ezmsg_nwb_settings_rank_rotate_test.nwb"
    rotated_path = tmp_path / "ezmsg_nwb_settings_rank_rotate_test_01.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings({"component_value": [1, 2, 3]}, timestamp=1.0)
    sink.update_settings("component", {"component_value": [[1, 2, 3]]}, timestamp=2.0)
    sink.close(write=False)

    assert outpath.exists()
    assert rotated_path.exists()

    with NWBHDF5IO(rotated_path, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df["component_value"].iloc[-1], np.array([[1, 2, 3]]))


def test_sink_rotation_before_new_data_preserves_new_settings_file(tmp_path):
    """A settings-triggered rotation should keep the new file even before more samples arrive."""
    outpath = tmp_path / "ezmsg_nwb_settings_no_data_rotate_test.nwb"
    rotated_path = tmp_path / "ezmsg_nwb_settings_no_data_rotate_test_01.nwb"
    sink = NWBSinkConsumer(
        settings=NWBSinkSettings(
            filepath=outpath,
            overwrite_old=True,
            inc_clock=ReferenceClockType.UNKNOWN,
        )
    )

    sink.initialize_settings({"component_list": [1, 2]}, timestamp=1.0)
    sink.update_settings("component", {"component_list": [[1, 2], [3, 4]]}, timestamp=2.0)
    sink.close(write=False)

    assert outpath.exists()
    assert rotated_path.exists()

    with NWBHDF5IO(rotated_path, "r") as io:
        nwbfile = io.read()
        df = nwbfile.intervals["pipeline_settings"].to_dataframe()
        np.testing.assert_array_equal(df["component_list"].iloc[-1], np.array([[1, 2], [3, 4]]))
