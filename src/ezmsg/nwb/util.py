from enum import Enum

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
