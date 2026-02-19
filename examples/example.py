"""Example usage of ezmsg-nwb package."""

import asyncio
import importlib


async def main():
    """Run the example."""
    print("ezmsg-nwb loaded successfully!")
    print(f"Version: {importlib.import_module('ezmsg.nwb').__version__}")

    # Example: Read from an NWB file
    # reader = NWBIteratorUnit()
    # reader.apply_settings(NWBIteratorSettings(filepath="path/to/file.nwb"))

    # Example: Write to an NWB file
    # writer = NWBSink()
    # writer.apply_settings(NWBSinkSettings(filepath="/tmp/output.nwb", overwrite_old=True))


if __name__ == "__main__":
    asyncio.run(main())
