import os
import re
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


def sanitize_settings_column_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", ".", name).strip("_")
    if not sanitized:
        sanitized = "setting"
    if sanitized[0].isdigit():
        sanitized = f"setting_{sanitized}"
    return sanitized


def sanitize_settings_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix if prefix else "value": value}

    result: dict[str, Any] = {}
    for key, sub_value in value.items():
        key_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(sub_value, dict):
            result.update(flatten_mapping(sub_value, key_prefix))
        else:
            result[key_prefix] = sub_value
    return result


def flatten_component_settings(component_address: str, value: Any) -> dict[str, Any]:
    if isinstance(value, ez.Settings):
        flat = flatten_settings(value)
    elif hasattr(value, "structured_value") and getattr(value, "structured_value") is not None:
        flat = flatten_mapping(getattr(value, "structured_value"))
    elif hasattr(value, "repr_value") and isinstance(getattr(value, "repr_value"), dict):
        flat = flatten_mapping(getattr(value, "repr_value"))
    else:
        flat = flatten_mapping(value)

    return {
        sanitize_settings_column_name(f"{component_address}.{field_name}"): sanitize_settings_value(field_value)
        for field_name, field_value in flat.items()
    }
