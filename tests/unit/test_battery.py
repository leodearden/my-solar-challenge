"""Tests for Battery configuration and state."""

import pytest
from solar_challenge.battery import BatteryConfig, Battery


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


@pytest.fixture
def default_config():
    """Create a default 5 kWh battery config."""
    return BatteryConfig.default_5kwh()


@pytest.fixture
def default_battery(default_config):
    """Create a default battery with standard settings."""
    return Battery(default_config)


class TestBatterySOCTracking:
    """Test state of charge tracking (BAT-002)."""

    def test_initial_soc_default(self, default_config):
        """Default initial SOC is midpoint of usable range."""
        battery = Battery(default_config)
        # For 5 kWh with 10-90% limits: min=0.5, max=4.5, mid=2.5
        assert battery.soc_kwh == 2.5

    def test_initial_soc_custom(self, default_config):
        """Can set custom initial SOC."""
        battery = Battery(default_config, initial_soc_kwh=3.0)
        assert battery.soc_kwh == 3.0

    def test_initial_soc_out_of_range_raises(self, default_config):
        """Initial SOC outside limits raises error."""
        with pytest.raises(ValueError, match="outside allowed range"):
            Battery(default_config, initial_soc_kwh=0.0)  # Below min
        with pytest.raises(ValueError, match="outside allowed range"):
            Battery(default_config, initial_soc_kwh=5.0)  # Above max

    def test_soc_updated_after_charge(self, default_battery):
        """SOC increases after charging."""
        initial_soc = default_battery.soc_kwh
        default_battery.charge(power_kw=1.0, duration_minutes=60)
        assert default_battery.soc_kwh > initial_soc

    def test_soc_updated_after_discharge(self, default_battery):
        """SOC decreases after discharging."""
        initial_soc = default_battery.soc_kwh
        default_battery.discharge(power_kw=1.0, duration_minutes=60)
        assert default_battery.soc_kwh < initial_soc

    def test_soc_queryable(self, default_battery):
        """SOC is queryable at any time."""
        assert isinstance(default_battery.soc_kwh, float)
        assert isinstance(default_battery.soc_fraction, float)


class TestBatterySOCLimits:
    """Test SOC limit enforcement (BAT-005)."""

    def test_default_limits(self, default_config):
        """Default limits are 10% min, 90% max."""
        battery = Battery(default_config)
        assert battery.min_soc_fraction == 0.1
        assert battery.max_soc_fraction == 0.9
        assert battery.min_soc_kwh == 0.5  # 10% of 5 kWh
        assert battery.max_soc_kwh == 4.5  # 90% of 5 kWh

    def test_usable_capacity(self, default_battery):
        """Usable capacity is max - min."""
        assert default_battery.usable_capacity_kwh == 4.0  # 4.5 - 0.5

    def test_charge_stops_at_max_soc(self, default_config):
        """Charging stops when max SOC reached."""
        # Start at max SOC
        battery = Battery(default_config, initial_soc_kwh=4.5)
        energy_charged = battery.charge(power_kw=2.5, duration_minutes=60)
        assert energy_charged == 0.0
        assert battery.soc_kwh == 4.5

    def test_discharge_stops_at_min_soc(self, default_config):
        """Discharging stops when min SOC reached."""
        # Start at min SOC
        battery = Battery(default_config, initial_soc_kwh=0.5)
        energy_discharged = battery.discharge(power_kw=2.5, duration_minutes=60)
        assert energy_discharged == 0.0
        assert battery.soc_kwh == 0.5


class TestBatteryChargeEfficiency:
    """Test charge efficiency (BAT-003)."""

    def test_default_charge_efficiency(self, default_battery):
        """Default charge efficiency is 97.5%."""
        assert default_battery.charge_efficiency == 0.975

    def test_energy_stored_with_efficiency(self, default_config):
        """Energy stored = input * efficiency."""
        battery = Battery(default_config, initial_soc_kwh=1.0, charge_efficiency=0.95)
        initial_soc = battery.soc_kwh

        # Charge 1 kWh input
        battery.charge(power_kw=1.0, duration_minutes=60)

        # Should store 0.95 kWh
        assert battery.soc_kwh == pytest.approx(initial_soc + 0.95, rel=1e-6)


class TestBatteryDischargeEfficiency:
    """Test discharge efficiency (BAT-004)."""

    def test_default_discharge_efficiency(self, default_battery):
        """Default discharge efficiency is 97.5%."""
        assert default_battery.discharge_efficiency == 0.975

    def test_energy_output_with_efficiency(self, default_config):
        """Energy output = withdrawn * efficiency."""
        battery = Battery(default_config, initial_soc_kwh=3.0, discharge_efficiency=0.95)
        initial_soc = battery.soc_kwh

        # Request 1 kWh output
        energy_out = battery.discharge(power_kw=1.0, duration_minutes=60)

        # Should get ~0.95 kWh output (limited by efficiency)
        # Actually, we request 1 kWh power for 1 hour, withdraw 1/0.95 kWh, output 1 kWh
        # Wait - the logic is: we request power, we limit by rate, we calculate needed from battery
        # With 1 kW for 1 hour, we need to withdraw 1/0.95 = 1.053 kWh to output 1 kWh
        # So output is actually 1 kWh if we have capacity
        assert energy_out == pytest.approx(1.0, rel=0.01)
        # SOC drops by 1/0.95 = 1.053 kWh
        assert battery.soc_kwh == pytest.approx(initial_soc - 1.0 / 0.95, rel=0.01)


class TestBatteryChargeFromExcess:
    """Test charging from excess PV (BAT-006)."""

    def test_charge_method_basic(self, default_battery):
        """Charge method accepts power and duration."""
        initial_soc = default_battery.soc_kwh
        energy = default_battery.charge(power_kw=1.0, duration_minutes=30)
        assert energy > 0
        assert default_battery.soc_kwh > initial_soc

    def test_charge_respects_max_rate(self, default_config):
        """Charge rate limited to max_charge_kw."""
        battery = Battery(default_config, initial_soc_kwh=1.0)

        # Try to charge at 10 kW (max is 2.5 kW)
        energy = battery.charge(power_kw=10.0, duration_minutes=60)

        # Should only charge at 2.5 kW rate
        # 2.5 kW * 1 hour * 0.975 efficiency = 2.4375 kWh
        assert energy == pytest.approx(2.4375, rel=0.01)

    def test_charge_returns_actual_energy(self, default_battery):
        """Charge returns actual energy stored."""
        energy = default_battery.charge(power_kw=1.0, duration_minutes=60)
        assert isinstance(energy, float)
        assert energy >= 0


class TestBatteryDischargeToMeetDemand:
    """Test discharging to meet demand (BAT-007)."""

    def test_discharge_method_basic(self, default_battery):
        """Discharge method accepts power and duration."""
        initial_soc = default_battery.soc_kwh
        energy = default_battery.discharge(power_kw=1.0, duration_minutes=30)
        assert energy > 0
        assert default_battery.soc_kwh < initial_soc

    def test_discharge_respects_max_rate(self, default_config):
        """Discharge rate limited to max_discharge_kw."""
        battery = Battery(default_config, initial_soc_kwh=4.0)

        # Try to discharge at 10 kW (max is 2.5 kW)
        energy = battery.discharge(power_kw=10.0, duration_minutes=60)

        # Should only discharge at 2.5 kW rate
        # Limited by rate: 2.5 kWh output (approximately, with efficiency)
        assert energy <= 2.5 * 1.0  # max_rate * duration

    def test_discharge_returns_actual_energy(self, default_battery):
        """Discharge returns actual energy output."""
        energy = default_battery.discharge(power_kw=1.0, duration_minutes=60)
        assert isinstance(energy, float)
        assert energy >= 0
