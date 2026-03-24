import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import ezmsg.core as ez
import numpy as np
from neuroconv.utils import DeepDict


class ReferenceClockType(Enum):
    SYSTEM = "system"  # Streamed timestamps use time.time()
    MONOTONIC = "monotonic"  # Streamed timestamps use time.monotonic()
    UNKNOWN = "unknown"  # Streamed timestamps are not modified.


def build_nwb_fname(metadata: DeepDict) -> str:
    """Build a default NWB filename from the session metadata."""
    fname_str = f"sub-{metadata['Subject']['subject_id']}"
    ses = metadata["NWBFile"].get("session_id", metadata["NWBFile"]["session_start_time"].strftime("%Y%m%dT%H%M%S"))
    fname_str += f"_ses-{ses}"
    return f"{fname_str}_ephys.nwb"


def _flatten_value(value: Any, prefix: str) -> dict[str, Any]:
    """Flatten a settings value into dotted key/value pairs."""
    if isinstance(value, ez.Settings):
        return flatten_ez_settings(value, prefix)

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, sub_value in value.items():
            key_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_value(sub_value, key_prefix))
        return result

    return {prefix if prefix else "value": value}


def flatten_ez_settings(settings: ez.Settings, prefix: str = "") -> dict[str, Any]:
    """Flatten an ezmsg settings object into dotted key/value pairs."""
    settings_prefix = settings.__class__.__name__ if not prefix else prefix
    result: dict[str, Any] = {}
    for name, value in settings.__dict__.items():
        result.update(_flatten_value(value, f"{settings_prefix}.{name}"))
    return result


def sanitize_settings_column_name(name: str) -> str:
    """Convert a settings field path into an NWB-safe column name."""
    sanitized = re.sub(r"[^0-9A-Za-z_]+", ".", name).strip("_")
    if not sanitized:
        sanitized = "setting"
    if sanitized[0].isdigit():
        sanitized = f"setting_{sanitized}"
    return sanitized


def _sanitize_sequence_value(value: Sequence[Any]) -> Any:
    """Sanitize a sequence while preserving array-like values when possible."""
    sanitized = [sanitize_settings_value(item) for item in value]

    try:
        array_value = np.asarray(sanitized)
    except Exception:
        return json.dumps(sanitized, default=str)

    if array_value.dtype != object:
        return array_value

    if all(isinstance(item, str) for item in sanitized):
        return np.asarray(sanitized, dtype=str)

    if all(isinstance(item, bytes) for item in sanitized):
        return np.asarray(sanitized, dtype="S")

    try:
        stacked = np.stack(sanitized)
    except Exception:
        return json.dumps(sanitized, default=str)

    if stacked.dtype != object:
        return stacked

    return json.dumps(sanitized, default=str)


def sanitize_settings_value(value: Any) -> Any:
    """Convert a settings value into an NWB-compatible scalar or array value."""
    if value is None:
        return "None"
    if isinstance(value, Enum):
        return sanitize_settings_value(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value if value.dtype != object else _sanitize_sequence_value(value.tolist())
    if isinstance(value, (bool, int, float, str, bytes)):
        # Keep primitive types
        return value
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return json.dumps({str(k): sanitize_settings_value(v) for k, v in value.items()}, default=str)
    if isinstance(value, (set, frozenset)):
        return _sanitize_sequence_value(sorted(value, key=repr))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _sanitize_sequence_value(value)
    try:
        return json.dumps(value)
    except TypeError:
        # Force everything else to str
        return str(value)


def flatten_component_settings(component_address: str, value: Any) -> dict[str, Any]:
    """Flatten and sanitize a component settings payload for NWB storage."""
    if isinstance(value, ez.Settings):
        flat = flatten_ez_settings(value)
    elif hasattr(value, "structured_value") and getattr(value, "structured_value") is not None:
        flat = _flatten_value(getattr(value, "structured_value"))
    elif hasattr(value, "repr_value") and isinstance(getattr(value, "repr_value"), dict):
        flat = _flatten_value(getattr(value, "repr_value"))
    else:
        flat = _flatten_value(value, "")

    named_flat = {
        sanitize_settings_column_name(f"{component_address}.{field_name}"): sanitize_settings_value(field_value)
        for field_name, field_value in flat.items()
    }
    return named_flat
