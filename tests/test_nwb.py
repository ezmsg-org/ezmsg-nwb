"""Basic tests for ezmsg-nwb package."""


def test_import():
    """Test that the package can be imported."""
    import ezmsg.nwb

    assert hasattr(ezmsg.nwb, "__version__")


def test_version():
    """Test that version is a string."""
    from ezmsg.nwb import __version__

    assert isinstance(__version__, str)


def test_exports():
    """Test that key classes are exported."""
    import ezmsg.nwb

    expected = [
        "NWBAxisArrayIterator",
        "NWBIteratorSettings",
        "NWBIteratorUnit",
        "NWBIteratorState",
        "NWBSink",
        "NWBSinkConsumer",
        "NWBSinkSettings",
        "ReferenceClockType",
        "build_nwb_fname",
    ]
    for name in expected:
        assert hasattr(ezmsg.nwb, name), f"Missing export: {name}"

    assert ezmsg.nwb.ReferenceClockType.SYSTEM.value == "system"
    assert ezmsg.nwb.ReferenceClockType.MONOTONIC.value == "monotonic"
    assert ezmsg.nwb.ReferenceClockType.UNKNOWN.value == "unknown"
