"""Basic package tests."""

import solar_challenge


def test_version_exists():
    """Package has a version string."""
    assert hasattr(solar_challenge, "__version__")
    assert isinstance(solar_challenge.__version__, str)


def test_version_format():
    """Version follows semantic versioning format."""
    version = solar_challenge.__version__
    parts = version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
