# ezmsg-nwb

NWB (Neurodata Without Borders) file reading and writing for the [ezmsg](https://www.ezmsg.org) framework.

## Overview

`ezmsg-nwb` provides streaming NWB file I/O as ezmsg Units.

Key features:

* **NWB Reader** - Stream data from NWB files (local or remote) as AxisArray messages
* **NWB Writer** - Write incoming AxisArray streams to NWB files with automatic container management
* **Flexible clock handling** - Support for system, monotonic, and unknown reference clocks

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
