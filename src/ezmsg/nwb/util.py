import typing
from enum import Enum

import numpy as np
from neuroconv.utils import DeepDict


def as_text(value: typing.Any) -> str:
    """One HDF5 value as text, whichever string flavour the writer used.

    Which one you get depends on who wrote the file. pynwb stores text as
    variable-length UTF-8 and h5py hands it back as ``str``; writers that
    declare their strings ``H5T_CSET_ASCII`` (aqnwb, and so every recording
    Orion writes) read back as ``bytes``, because hdmf keys its decoding
    decision directly off the character set.

    ``str(value)`` is *not* a safe substitute: on ``bytes`` it yields the repr
    ``"b'cond_0'"``, which compares equal to nothing and parses as no JSON --
    a corruption that no exception announces.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def as_text_array(values: typing.Any) -> np.ndarray:
    """A whole string column as a unicode array. See :func:`as_text`."""
    array = np.asarray(values)
    if array.dtype.kind == "U":
        return array
    return np.array([as_text(value) for value in array.ravel()], dtype=str).reshape(array.shape)


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
