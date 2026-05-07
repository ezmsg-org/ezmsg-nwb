"""Typed-column pipeline-settings sink for ezmsg-nwb.

Phase 2 of the pipeline-settings story. Where the generic :class:`NWBSink`
lands :class:`PipelineSettingsEvent` messages in a JSON-encoded
``AnnotationSeries`` (one string column, fixed schema), this subclass
projects each event's ``structured_value`` into a
``pynwb.epoch.TimeIntervals`` table named ``pipeline_settings`` with one
column per dotted-key path. Columns get their values' native dtypes, so
analysis tools can read settings as a normal pandas DataFrame.

Interval semantics
------------------

Each event opens a new interval. The PREVIOUS interval is closed (and
written to disk) at the moment the next event arrives — its ``stop_time``
is the new event's ``timestamp``. The currently-active (open) interval is
held in memory and flushed at file close.

This means: a non-graceful crash will lose the most recent open interval.
That's accepted as the cost of interval semantics — your collaborator
chose this trade-off.

File rotation on schema change
------------------------------

If an incoming event introduces a new column, or changes the per-cell
shape of an existing column (scalar↔array, rank change, fixed-shape
mismatch), the sink rotates into a fresh file segment whose table opens
already populated with the new schema. Each segment's table is
internally consistent. Reading back means iterating ``<stem>_NN.nwb``
files in order.

Use the bundled :class:`PipelineSettingsTableCollection` to wire a
:class:`PipelineSettingsUnit` (from ezmsg-baseproc) to this sink in a
single component.
"""

from __future__ import annotations

import asyncio
import time
import typing

import ezmsg.core as ez
import numpy as np
import pynwb
from ezmsg.baseproc import (
    INIT_FINAL_COMPONENT_ADDRESS,
    PipelineSettingsEvent,
    PipelineSettingsEventType,
    PipelineSettingsProducerSettings,
    PipelineSettingsUnit,
    flatten_component_settings,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from hdmf.backends.hdf5.h5_utils import H5DataIO

from .writer import (
    NWBSink,
    NWBSinkConsumer,
    NWBSinkSettings,
    NWBSinkState,
)

# ---------------------------------------------------------------------------
# Settings + State
# ---------------------------------------------------------------------------


class NWBPipelineSettingsSinkSettings(NWBSinkSettings):
    """Settings for :class:`NWBPipelineSettingsSink`.

    Inherits all file-level fields from :class:`NWBSinkSettings`
    (``filepath``, ``overwrite_old``, ``inc_clock``, ``recording``,
    ``split_bytes``, ``meta_yaml``, ``expected_series``, ``axis``).
    """

    pipeline_settings_table_name: str = "pipeline_settings"
    """Name of the per-file ``TimeIntervals`` table to write into. Lives
    under ``nwbfile.intervals[<name>]``."""


@processor_state
class NWBPipelineSettingsSinkState(NWBSinkState):
    """State for :class:`NWBPipelineSettingsSinkConsumer`.

    Adds in-memory bookkeeping for the open interval. ``settings_columns``
    locks the table schema for the current file segment;
    ``settings_state`` mirrors the most-recent flat settings dict so a
    closed interval can be written without re-flattening.
    """

    settings_columns: typing.Optional[typing.List[str]] = None
    settings_state: typing.Optional[typing.Dict[str, typing.Any]] = None
    settings_active_since: typing.Optional[float] = None
    settings_prev_component: str = "__init__"
    pending_initial_state: typing.Optional[typing.Dict[str, typing.Any]] = None
    """Per-component INITIAL events accumulate here while the startup
    snapshot is in flight. ``None`` means we're not currently buffering;
    the dict is populated by per-component INITIALs and flushed to one
    merged anchor row on the ``INIT_FINAL_COMPONENT_ADDRESS`` sentinel
    (or on the first non-INITIAL event, if the producer dropped the
    sentinel)."""
    pending_initial_first_seen: typing.Optional[float] = None
    """Timestamp of the first INITIAL event in the current buffer — used
    as the merged anchor row's ``start_time`` when we eventually flush."""


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


class NWBPipelineSettingsSinkConsumer(NWBSinkConsumer):
    """Adds typed-column ``TimeIntervals`` writing on top of :class:`NWBSinkConsumer`.

    Public method: :meth:`write_settings_event`. The unit subclass calls
    it on each :class:`PipelineSettingsEvent`. Internally the consumer
    detects schema-incompatible changes and rotates the file via
    :meth:`_rotate_file`.
    """

    @classmethod
    def get_state_type(cls) -> type:
        return NWBPipelineSettingsSinkState

    def _reset_state(self, message: typing.Optional[AxisArray]) -> None:
        super()._reset_state(message)
        # Per-file pipeline-settings tracker; cleared on every reset so a
        # fresh file starts empty. The next event re-seeds the table.
        self._state.settings_columns = []
        self._state.settings_state = {}
        self._state.settings_active_since = None
        self._state.settings_prev_component = "__init__"
        self._state.pending_initial_state = None
        self._state.pending_initial_first_seen = None

    @property
    def _settings_table_name(self) -> str:
        return self.settings.pipeline_settings_table_name

    # ------------------------------------------------------------------
    # Public API: write one settings event
    # ------------------------------------------------------------------

    def write_settings_event(self, event: PipelineSettingsEvent) -> None:
        """Project a :class:`PipelineSettingsEvent` into native columns and append/rotate.

        Aggregation rules:

        - **Per-component INITIAL events** at startup are buffered in
          memory (``pending_initial_state``). The table is *not*
          registered with the file yet — registering it per component
          would force a schema-driven rotation per component.
        - **The ``INIT_FINAL_COMPONENT_ADDRESS`` sentinel** flushes the
          buffer as ONE merged anchor row, registers the table, and
          opens an interval tracking the merged state.
        - **Any non-INITIAL event arriving while the buffer is
          populated** (producer dropped the sentinel) flushes the buffer
          first as if the sentinel had arrived, then processes the
          event normally.
        - **Subsequent schema-compatible event** closes the prior open
          interval (writes a row ``[prev.timestamp, this.timestamp]``)
          and opens a new one.
        - **Schema-incompatible event** rotates into a new file segment
          whose table opens with the merged schema and a fresh anchor.
        - **Close** flushes any pending initial buffer (merged anchor)
          and the still-open interval as ``[active_since, close_time]``.
        """
        with self._lock:
            if self._state.io is None:
                # Reset cleared state and a new file isn't open yet; drop.
                return

            is_init_final = (
                event.event_type == PipelineSettingsEventType.INITIAL
                and event.component_address == INIT_FINAL_COMPONENT_ADDRESS
            )

            if is_init_final:
                self._flush_pending_initial(event.timestamp)
                return

            value = event.structured_value if event.structured_value is not None else event.repr_value
            flat = flatten_component_settings(event.component_address, value)
            if not flat:
                return

            is_initial = event.event_type == PipelineSettingsEventType.INITIAL

            # Buffer per-component INITIALs only while the table hasn't
            # been registered yet. After the first anchor row exists,
            # late-arriving INITIALs (e.g. a runtime new component) are
            # processed as ordinary updates so they go through the
            # rotation path if their schema diverges.
            if is_initial and not self._state.settings_columns:
                if self._state.pending_initial_state is None:
                    self._state.pending_initial_state = {}
                    self._state.pending_initial_first_seen = event.timestamp
                self._state.pending_initial_state.update(flat)
                self._state.settings_prev_component = event.component_address
                return

            # Non-INITIAL event arrived but we never saw the sentinel —
            # flush the buffer as a merged anchor before processing.
            if self._state.pending_initial_state is not None:
                self._flush_pending_initial(event.timestamp)

            if not self._state.settings_columns:
                # No INITIAL events were ever buffered — first event is
                # a one-component snapshot. Register the table with an
                # anchor row so the state hits disk immediately.
                self._state.settings_columns = list(flat.keys())
                self._state.settings_state = {col: "" for col in self._state.settings_columns}
                self._state.settings_state.update(flat)
                self._state.settings_active_since = event.timestamp
                self._state.settings_prev_component = event.component_address
                rel_t = self._settings_relative_time(event.timestamp)
                self._register_settings_table_with_first_row(
                    start_time=rel_t,
                    stop_time=rel_t,
                    updated_component=event.component_address,
                )
                return

            if self._settings_update_requires_rotation(flat):
                self._rotate_file(
                    timestamp=event.timestamp,
                    next_settings_state=self._merged_settings_state(flat),
                    next_settings_prev_component=event.component_address,
                )
                return

            self._validate_settings_columns(flat)
            self._apply_settings_update(event.component_address, flat, event.timestamp)

    def _flush_pending_initial(self, sentinel_timestamp: float) -> None:
        """Materialize the buffered per-component INITIAL state as one anchor row.

        Called either when the producer's ``INIT_FINAL_COMPONENT_ADDRESS``
        sentinel arrives, on a non-INITIAL event arriving with a
        non-empty buffer (producer dropped the sentinel), or from
        ``_close_locked`` if neither happened before close.
        """
        pending = self._state.pending_initial_state
        first_seen = self._state.pending_initial_first_seen
        self._state.pending_initial_state = None
        self._state.pending_initial_first_seen = None
        if not pending:
            return
        # Anchor at the FIRST INITIAL's timestamp (when the snapshot
        # started accumulating); the open interval tracks from there
        # until the next non-INITIAL event closes it.
        anchor_t = first_seen if first_seen is not None else sentinel_timestamp
        self._state.settings_columns = list(pending.keys())
        self._state.settings_state = dict(pending)
        self._state.settings_active_since = anchor_t
        self._state.settings_prev_component = INIT_FINAL_COMPONENT_ADDRESS
        rel_t = self._settings_relative_time(anchor_t)
        self._register_settings_table_with_first_row(
            start_time=rel_t,
            stop_time=rel_t,
            updated_component=INIT_FINAL_COMPONENT_ADDRESS,
        )

    # ------------------------------------------------------------------
    # Table prep
    # ------------------------------------------------------------------

    def _configure_appendable_table(
        self,
        table: typing.Any,
        sample_values: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> None:
        """Configure a dynamic table so its columns remain appendable after flush/reopen."""
        table.id.set_data_io(H5DataIO, {"maxshape": (None,), "chunks": True})
        for col in table.colnames:
            sample_value = None if sample_values is None else sample_values.get(col)
            if sample_value is None:
                col_shape = getattr(table[col].data, "shape", ())
                maxshape = (None,) + tuple(col_shape[1:]) if len(col_shape) > 1 else (None,)
            elif np.isscalar(sample_value) or isinstance(sample_value, (str, bytes)):
                maxshape = (None,)
            else:
                col_shape = np.asarray(sample_value).shape
                maxshape = (None,) + tuple(col_shape)
            table[col].set_data_io(H5DataIO, {"maxshape": maxshape, "chunks": True})

    def _get_settings_table(self) -> typing.Any:
        nwbfile = self._state.nwbfile
        if nwbfile is None or nwbfile.intervals is None:
            return None
        try:
            return nwbfile.intervals[self._settings_table_name]
        except Exception:
            return None

    def _register_settings_table_with_first_row(
        self,
        start_time: float,
        stop_time: float,
        updated_component: str,
    ) -> None:
        """Build the table with one row already populated, then register it.

        pynwb's ``VectorData`` columns can't be serialized while empty
        (dtype inference fails). The cheapest workaround is to register
        the table only when we have a closed interval to write — the
        first row supplies dtypes for every column. Caller must have
        ``self._state.settings_state`` filled with the row's values.
        """
        intervals = pynwb.epoch.TimeIntervals(
            name=self._settings_table_name,
            description="Flattened ezmsg settings snapshots active over each logged interval",
        )
        intervals.add_column(
            name="updated_component",
            description="component that triggered the snapshot transition",
        )
        for column_name in self._state.settings_columns or []:
            intervals.add_column(name=column_name, description="flattened ezmsg setting")
        intervals.add_interval(
            start_time=start_time,
            stop_time=stop_time,
            updated_component=updated_component,
            **(self._state.settings_state or {}),
        )
        self._state.nwbfile.add_time_intervals(intervals)
        self._configure_appendable_table(intervals, self._state.settings_state)
        # Flush so the file gains a valid NWB structure with our table
        # populated and h5py datasets become reachable for subsequent
        # in-place appends.
        self._flush_io(reopen=True)

    # ------------------------------------------------------------------
    # Schema validation / shape tracking
    # ------------------------------------------------------------------

    def _settings_relative_time(self, timestamp: float) -> float:
        """Convert a wall-clock timestamp into file-relative session time.

        Mirrors :meth:`NWBSinkConsumer.write_annotation`'s behavior: when
        ``start_timestamp`` is unset (no data has anchored the file yet),
        the wall-clock value is passed through unmodified. Once data has
        set ``start_timestamp``, every subsequent row uses that baseline.
        """
        if self._state.start_timestamp != 0.0:
            return float(timestamp) - self._state.start_timestamp
        return float(timestamp)

    def _settings_value_shape(self, value: typing.Any) -> typing.Tuple[int, ...]:
        if value is None or np.isscalar(value) or isinstance(value, (str, bytes)):
            return ()
        try:
            return tuple(np.asarray(value).shape)
        except Exception:
            return ()

    def _column_value_shape(self, column_name: str) -> typing.Tuple[int, ...]:
        table = self._get_settings_table()
        if table is not None and column_name in table.colnames:
            data = table[column_name].data
            if hasattr(data, "shape"):
                shape = tuple(data.shape)
                return shape[1:] if len(shape) > 1 else ()

        state = self._state.settings_state or {}
        if column_name in state:
            return self._settings_value_shape(state[column_name])
        return ()

    def _validate_settings_columns(self, flat_settings: typing.Dict[str, typing.Any]) -> None:
        cols = self._state.settings_columns or []
        missing = [name for name in flat_settings if name not in cols]
        if missing:
            raise ValueError(
                f"Received settings fields not present in settings table schema: {', '.join(sorted(missing))}"
            )

    def _settings_update_requires_rotation(self, flat_settings: typing.Dict[str, typing.Any]) -> bool:
        cols = self._state.settings_columns or []
        for column_name, value in flat_settings.items():
            if column_name not in cols:
                return True
            current_shape = self._column_value_shape(column_name)
            new_shape = self._settings_value_shape(value)
            current_is_scalar = current_shape == ()
            new_is_scalar = new_shape == ()
            if current_is_scalar != new_is_scalar:
                return True
            if not current_is_scalar and current_shape != new_shape:
                return True
        return False

    def _merged_settings_state(self, flat_settings: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """Merge an incoming partial update over the current state for the next file."""
        cols = self._state.settings_columns or []
        state = self._state.settings_state or {}
        merged: typing.Dict[str, typing.Any] = {column_name: state.get(column_name, "") for column_name in cols}
        merged.update(flat_settings)
        return merged

    # ------------------------------------------------------------------
    # Apply update / flush interval
    # ------------------------------------------------------------------

    def _apply_settings_update(
        self,
        component_address: str,
        flat_settings: typing.Dict[str, typing.Any],
        timestamp: float,
    ) -> None:
        """Close the current interval (writing a row) and open a new one."""
        self._flush_settings_interval(timestamp, self._state.settings_prev_component)
        if not self._state.settings_state:
            self._state.settings_state = {col: "" for col in (self._state.settings_columns or [])}
        self._state.settings_state.update(flat_settings)
        self._state.settings_active_since = timestamp
        self._state.settings_prev_component = component_address

    def _flush_settings_interval(self, end_timestamp: float, updated_component: str) -> None:
        """Append the currently-active settings snapshot as a closed interval row.

        On the first row in a file, registers the table with the NWB
        structure (see :meth:`_register_settings_table_with_first_row`).
        On subsequent rows, appends to the existing table.
        """
        state = self._state
        if not state.settings_state or state.settings_active_since is None:
            return

        start_time = self._settings_relative_time(state.settings_active_since)
        stop_time = self._settings_relative_time(end_timestamp)
        if stop_time < start_time:
            stop_time = start_time

        table = self._get_settings_table()
        if table is None:
            self._register_settings_table_with_first_row(
                start_time=start_time,
                stop_time=stop_time,
                updated_component=updated_component,
            )
            return

        table.add_interval(
            start_time=start_time,
            stop_time=stop_time,
            updated_component=updated_component,
            **state.settings_state,
        )

    # ------------------------------------------------------------------
    # File rotation on schema change
    # ------------------------------------------------------------------

    def _rotate_file(
        self,
        timestamp: float,
        next_settings_state: typing.Optional[typing.Dict[str, typing.Any]],
        next_settings_prev_component: str,
    ) -> None:
        """Close the current segment and open a new one tracking ``next_settings_state``.

        The current open interval is flushed to the OLD file as a closed
        interval ``[active_since, timestamp]``. The NEW file opens with
        no settings table registered yet — the next event (or close)
        will lazily register it via :meth:`_register_settings_table_with_first_row`,
        with ``next_settings_state`` as the open interval starting at
        ``timestamp``.
        """
        state = self._state
        if state.settings_state and state.settings_active_since is not None:
            self._flush_settings_interval(timestamp, state.settings_prev_component)

        next_settings_state = dict(next_settings_state or {})

        # Prevent close() from re-appending the interval we just flushed.
        state.settings_active_since = None

        # Advance to the next file segment before creating the replacement file.
        state.split_count += 1
        self.path_on_disk.unlink(missing_ok=True)

        new_nwbfile, new_meta = self._copy_nwb()
        self.close()
        self._nwb_create_or_fail(nwbfile=new_nwbfile)
        self._prep_from_meta(new_meta)

        if next_settings_state:
            # Track the migrated state as a new open interval AND write
            # an anchor row in the new segment immediately, mirroring the
            # first-event behaviour on a fresh file. A crash between
            # rotations should not lose the rotation snapshot.
            self._state.settings_columns = list(next_settings_state.keys())
            self._state.settings_state = dict(next_settings_state)
            self._state.settings_active_since = timestamp
            self._state.settings_prev_component = next_settings_prev_component
            rel_t = self._settings_relative_time(timestamp)
            self._register_settings_table_with_first_row(
                start_time=rel_t,
                stop_time=rel_t,
                updated_component=next_settings_prev_component,
            )
        else:
            self._state.settings_columns = []
            self._state.settings_state = {}
            self._state.settings_active_since = None
            self._state.settings_prev_component = next_settings_prev_component

    # ------------------------------------------------------------------
    # Close: flush the open interval; keep the file if the table has rows
    # ------------------------------------------------------------------

    def _close_locked(self, state: NWBPipelineSettingsSinkState, write: bool, log: bool) -> None:
        if state.io is None:
            return
        # If we received per-component INITIAL events but never saw the
        # sentinel before close, materialize the buffer as a merged anchor
        # row now so the snapshot still hits disk.
        if state.pending_initial_state is not None:
            self._flush_pending_initial(time.time())
        # Flush any open pipeline-settings interval so it lands in the file.
        if state.settings_state and state.settings_active_since is not None:
            self._flush_settings_interval(time.time(), state.settings_prev_component)
            state.settings_active_since = None

        # Compute b_delete here (parent's logic + settings-table check) so
        # we can override the parent's potential delete-on-empty.
        nwbfile = state.nwbfile
        io = state.io
        if write:
            io.write(nwbfile)
        src_str = f"{io.source}"
        b_delete = sum(s.bytes_written for s in state.series.values()) == 0
        for key in ["epochs", "trials"]:
            if hasattr(nwbfile, key) and getattr(nwbfile, key) is not None:
                b_delete = b_delete and len(getattr(nwbfile, key)) == 1  # EZNWB-START
        if state.annotation_ts and any(len(ts) > 0 for ts in state.annotation_ts.values()):
            b_delete = False
        # A populated pipeline-settings table is also "content".
        settings_table = self._get_settings_table()
        if settings_table is not None and len(settings_table.id) > 0:
            b_delete = False
        io.close()
        state.nwbfile = None
        state.io = None
        state.series = {}
        state.annotation_data = {}
        state.annotation_ts = {}
        state.settings_columns = []
        state.settings_state = {}
        state.settings_prev_component = "__init__"
        state.pending_initial_state = None
        state.pending_initial_first_seen = None
        if log:
            ez.logger.info(f"Closed file at {src_str}")
        if b_delete:
            self.path_on_disk.unlink(missing_ok=True)
            if log:
                ez.logger.info(f"Deleted empty file at {src_str}.")


# ---------------------------------------------------------------------------
# Sink unit
# ---------------------------------------------------------------------------


class NWBPipelineSettingsSink(NWBSink):
    """Specialized :class:`NWBSink` that writes :class:`PipelineSettingsEvent`
    messages into a typed-column ``TimeIntervals`` table.

    Drop-in replacement for :class:`NWBSink` when used inside
    :class:`PipelineSettingsTableCollection`. Inherits all of
    ``NWBSink``'s data-recording behaviour; replaces ``on_annotation`` so
    only :class:`PipelineSettingsEvent` traffic on ``INPUT_ANNOTATION``
    is consumed (other annotation-shaped messages are silently dropped —
    use :class:`NWBSink` directly if you need the JSON-AnnotationSeries
    fallback).
    """

    SETTINGS = NWBPipelineSettingsSinkSettings

    def create_processor(self) -> None:
        """Construct the typed-column consumer rather than the generic one."""
        from ezmsg.baseproc.units import _close_previous

        _close_previous(getattr(self, "processor", None))
        self.processor = NWBPipelineSettingsSinkConsumer(settings=self.SETTINGS)

    @ez.subscriber(NWBSink.INPUT_ANNOTATION)
    async def on_annotation(self, msg: typing.Any) -> None:
        """Dispatch :class:`PipelineSettingsEvent` to the typed-column writer.

        Non-:class:`PipelineSettingsEvent` messages are silently ignored
        — keep the typed semantics clean. Wrap with another sink (or run
        a plain :class:`NWBSink` in parallel) if you need JSON
        annotations alongside.
        """
        if not isinstance(msg, PipelineSettingsEvent):
            return
        try:
            await asyncio.to_thread(self.processor.write_settings_event, msg)
        except Exception as exc:
            ez.logger.warning(
                f"{self.address} failed to write pipeline-settings event for {msg.component_address}: {exc}"
            )


# ---------------------------------------------------------------------------
# Bundled Collection
# ---------------------------------------------------------------------------


class PipelineSettingsTableCollectionSettings(ez.Settings):
    producer: PipelineSettingsProducerSettings
    sink: NWBPipelineSettingsSinkSettings


class PipelineSettingsTableCollection(ez.Collection):
    """Bundle a :class:`PipelineSettingsUnit` with an :class:`NWBPipelineSettingsSink`.

    Accepts an external ``INPUT_SIGNAL`` (relayed to the sink) so the
    Collection drops into a pipeline as a normal NWB sink, with settings
    logging happening transparently alongside acquisition recording.
    Also relays ``INPUT_SETTINGS`` so users can push :class:`NWBSinkSettings`
    updates from outside the Collection.
    """

    SETTINGS = PipelineSettingsTableCollectionSettings

    PUB = PipelineSettingsUnit()
    SINK = NWBPipelineSettingsSink()

    INPUT_SIGNAL = ez.InputTopic(AxisArray)
    INPUT_SETTINGS = ez.InputTopic(NWBPipelineSettingsSinkSettings)

    def configure(self) -> None:
        self.PUB.apply_settings(self.SETTINGS.producer)
        self.SINK.apply_settings(self.SETTINGS.sink)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.PUB.OUTPUT_SIGNAL, self.SINK.INPUT_ANNOTATION),
            (self.INPUT_SIGNAL, self.SINK.INPUT_SIGNAL),
            (self.INPUT_SETTINGS, self.SINK.INPUT_SETTINGS),
        )
