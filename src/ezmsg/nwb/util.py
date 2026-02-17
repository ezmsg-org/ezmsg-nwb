from enum import Enum
from typing import Any

import ezmsg.core as ez
from neuroconv.utils import DeepDict


class ReferenceClockType(Enum):
    SYSTEM = "system"  # Streamed timestamps use time.time()
    MONOTONIC = "monotonic"  # Streamed timestamps use time.monotonic()
    UNKNOWN = "unknown"  # Streamed timestamps are not modified.


def build_nwb_fname(metadata: DeepDict) -> str:
    fname_str = f"sub-{metadata['Subject']['subject_id']}"
    ses = metadata["NWBFile"].get("session_id", metadata["NWBFile"]["session_start_time"].strftime("%Y%m%dT%H%M%S"))
    fname_str += f"_ses-{ses}"
    return f"{fname_str}_ephys.nwb"


def flatten_settings(settings: ez.Settings, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}

    settings_prefix = settings.__class__.__name__ if not prefix else prefix

    for name, value in settings.__dict__.items():
        if isinstance(value, ez.Settings):
            sub_result = flatten_settings(value, f"{settings_prefix}.{name}")
            result.update(sub_result)
        else:
            result[f"{settings_prefix}.{name}"] = value

    return result
