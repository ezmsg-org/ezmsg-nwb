# ezmsg-example

Short description of your ezmsg package.

## Overview

`ezmsg-example` provides ... for the `ezmsg <https://www.ezmsg.org>`_ framework.

Key features:

* **Feature 1** - Description
* **Feature 2** - Description
* **Feature 3** - Description

## Installation

Install from PyPI:

```bash
pip install ezmsg-example
```

Or install the latest development version:

```bash
pip install git+https://github.com/ezmsg-org/ezmsg-example@main
```

## Dependencies

- `ezmsg`
- `numpy`

## Usage

See the `examples` folder for usage examples.

```python
import ezmsg.core as ez
from ezmsg.example import MyUnit

# Your usage example here
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
