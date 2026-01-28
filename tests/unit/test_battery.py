"""Tests for Battery configuration."""

import pytest
from solar_challenge.battery import BatteryConfig


class TestBatteryConfigBasics:
    """Test basic BatteryConfig functionality."""

    def test_create_with_all_params(self):
        """BatteryConfig can be created with all parameters."""
        config = BatteryConfig(
            capacity_kwh=10.0,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            name="Test battery"
        )
        assert config.capacity_kwh == 10.0
        assert config.max_charge_kw == 5.0
        assert config.max_discharge_kw == 5.0
        assert config.name == "Test battery"

    def test_default_values(self):
        """BatteryConfig uses correct defaults."""
        config = BatteryConfig(capacity_kwh=5.0)
        assert config.max_charge_kw == 2.5
        assert config.max_discharge_kw == 2.5
        assert config.name == ""


class TestBatteryConfigDefaults:
    """Test default battery configurations."""

    def test_default_5kwh(self):
        """Default 5 kWh battery has correct values."""
        config = BatteryConfig.default_5kwh()
        assert config.capacity_kwh == 5.0
        assert config.max_charge_kw == 2.5
        assert config.max_discharge_kw == 2.5
        assert config.name  # Has a name


class TestBatteryConfigValidation:
    """Test parameter validation."""

    def test_capacity_must_be_positive(self):
        """Capacity <= 0 raises error."""
        with pytest.raises(ValueError, match="Capacity"):
            BatteryConfig(capacity_kwh=0)
        with pytest.raises(ValueError, match="Capacity"):
            BatteryConfig(capacity_kwh=-1.0)

    def test_max_charge_must_be_positive(self):
        """Max charge <= 0 raises error."""
        with pytest.raises(ValueError, match="charge"):
            BatteryConfig(capacity_kwh=5.0, max_charge_kw=0)
        with pytest.raises(ValueError, match="charge"):
            BatteryConfig(capacity_kwh=5.0, max_charge_kw=-1.0)

    def test_max_discharge_must_be_positive(self):
        """Max discharge <= 0 raises error."""
        with pytest.raises(ValueError, match="discharge"):
            BatteryConfig(capacity_kwh=5.0, max_discharge_kw=0)
        with pytest.raises(ValueError, match="discharge"):
            BatteryConfig(capacity_kwh=5.0, max_discharge_kw=-1.0)
