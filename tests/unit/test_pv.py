"""Tests for PV configuration."""

import pytest
from solar_challenge.pv import PVConfig


class TestPVConfigBasics:
    """Test basic PVConfig functionality."""

    def test_create_with_all_params(self):
        """PVConfig can be created with all parameters."""
        config = PVConfig(
            capacity_kw=5.0,
            azimuth=170.0,
            tilt=30.0,
            name="Test system"
        )
        assert config.capacity_kw == 5.0
        assert config.azimuth == 170.0
        assert config.tilt == 30.0
        assert config.name == "Test system"

    def test_default_values(self):
        """PVConfig uses correct defaults."""
        config = PVConfig(capacity_kw=4.0)
        assert config.azimuth == 180.0  # South-facing
        assert config.tilt == 35.0  # UK optimal
        assert config.name == ""


class TestPVConfigDefaults:
    """Test default system configurations."""

    def test_default_4kw(self):
        """Default 4 kW system has correct values."""
        config = PVConfig.default_4kw()
        assert config.capacity_kw == 4.0
        assert config.azimuth == 180.0
        assert config.tilt == 35.0
        assert config.name  # Has a name


class TestPVConfigValidation:
    """Test parameter validation."""

    def test_capacity_must_be_positive(self):
        """Capacity <= 0 raises error."""
        with pytest.raises(ValueError, match="Capacity"):
            PVConfig(capacity_kw=0)
        with pytest.raises(ValueError, match="Capacity"):
            PVConfig(capacity_kw=-1.0)

    def test_azimuth_range(self):
        """Azimuth must be 0-360."""
        # Valid boundary values
        PVConfig(capacity_kw=1.0, azimuth=0.0)
        PVConfig(capacity_kw=1.0, azimuth=360.0)

        # Invalid values
        with pytest.raises(ValueError, match="Azimuth"):
            PVConfig(capacity_kw=1.0, azimuth=-1.0)
        with pytest.raises(ValueError, match="Azimuth"):
            PVConfig(capacity_kw=1.0, azimuth=361.0)

    def test_tilt_range(self):
        """Tilt must be 0-90."""
        # Valid boundary values
        PVConfig(capacity_kw=1.0, tilt=0.0)
        PVConfig(capacity_kw=1.0, tilt=90.0)

        # Invalid values
        with pytest.raises(ValueError, match="Tilt"):
            PVConfig(capacity_kw=1.0, tilt=-1.0)
        with pytest.raises(ValueError, match="Tilt"):
            PVConfig(capacity_kw=1.0, tilt=91.0)
