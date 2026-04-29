# ezmsg-nwb

NWB (Neurodata Without Borders) file reading and writing for the [ezmsg](https://www.ezmsg.org) framework.

## Overview

`ezmsg-nwb` provides streaming NWB file I/O as ezmsg Units.

Key features:

* **NWB Reader** - Stream data from NWB files (local or remote) as AxisArray messages
* **NWB Writer** - Write incoming AxisArray streams to NWB files with automatic container management
* **Flexible clock handling** - Support for system, monotonic, and unknown reference clocks
* **Pipeline settings logging** - Automatically record every component's settings into a `pipeline_settings` intervals table inside the NWB file

## Installation

Install from PyPI:

```bash
pip install ezmsg-nwb
```

Or install the latest development version:

```bash
pip install git+https://github.com/ezmsg-org/ezmsg-nwb@main
```

## Dependencies

- `ezmsg`
- `ezmsg-baseproc`
- `numpy`
- `pynwb`
- `h5py`
- `neuroconv`
- `remfile`
- `pyyaml`

## Usage

See the `examples` folder for usage examples.

```python
import ezmsg.core as ez
from ezmsg.nwb import NWBIteratorUnit, NWBSink
```

For general ezmsg tutorials and guides, visit [ezmsg.org](https://www.ezmsg.org).

### Pipeline settings table

When `NWBSink` is used inside an `ez.run` pipeline (with `ezmsg>=3.9.0`), it
opens a `GraphContext` against the running graph server, snapshots the
settings of every component in its session, and subscribes to subsequent
settings change events. Each snapshot is flattened into dotted column names
(e.g. `MY.UNIT.MyUnitSettings.endpoint.host`) and appended as a row to a
`pipeline_settings` `TimeIntervals` table inside the NWB file, alongside an
`updated_component` column identifying which component triggered the
transition. Reading back is straightforward:

```python
from pynwb import NWBHDF5IO

with NWBHDF5IO(path, "r") as io:
    nwbfile = io.read()
    df = nwbfile.intervals["pipeline_settings"].to_dataframe()
```

Notes:

* Settings logging is best-effort. If the writer cannot connect to the graph
  server (e.g. when running the consumer outside of `ez.run`), it logs a
  warning and continues writing data without the table.
* Settings values are sanitized for NWB storage: primitives, NumPy scalars,
  enums, paths, and fixed-shape sequences/arrays are stored natively;
  mappings and irregular structures are JSON-encoded; `None` becomes the
  string `"None"`.
* If a settings update changes a column's shape (scalar↔array or rank
  change), the writer rotates into a new file segment (`<name>_01.nwb`,
  `<name>_02.nwb`, …) so each segment's table stays internally consistent.

## Development

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork this repository and clone your fork locally.
3. Open a terminal and `cd` to the cloned folder.
4. Run `uv sync` to create a `.venv` and install dependencies.
5. (Optional) Install pre-commit hooks: `uv run pre-commit install`
6. After making changes, run the test suite: `uv run pytest tests`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

This project is supported by [the Wyss Center for Bio and Neuroengineering](https://wysscenter.ch)
and by [Blackrock Neurotech](https://www.blackrockneurotech.com).
