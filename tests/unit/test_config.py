"""Tests for configuration file support."""

import json
import tempfile
from pathlib import Path

import pytest

from solar_challenge.battery import BatteryConfig
from solar_challenge.config import (
    ConfigurationError,
    OutputConfig,
    ParameterSweepConfig,
    ScenarioConfig,
    SimulationPeriod,
    create_bristol_phase1_scenario,
    load_config,
    load_config_json,
    load_config_yaml,
    load_fleet_config,
    load_home_config,
    load_scenarios,
)
from solar_challenge.home import HomeConfig
from solar_challenge.load import LoadConfig
from solar_challenge.location import Location
from solar_challenge.pv import PVConfig


class TestSimulationPeriod:
    """Tests for SimulationPeriod class."""

    def test_string_dates(self) -> None:
        """Test period with string dates."""
        period = SimulationPeriod(
            start_date="2024-01-01",
            end_date="2024-01-07",
        )
        assert period.start_date == "2024-01-01"
        assert period.end_date == "2024-01-07"

    def test_get_timestamps(self) -> None:
        """Test getting timestamps from string dates."""
        period = SimulationPeriod(
            start_date="2024-01-01",
            end_date="2024-01-07",
        )
        start = period.get_start_timestamp("Europe/London")
        end = period.get_end_timestamp("Europe/London")
        assert start.year == 2024
        assert start.month == 1
        assert start.day == 1
        assert end.day == 7


class TestOutputConfig:
    """Tests for OutputConfig class."""

    def test_defaults(self) -> None:
        """Test default output configuration."""
        config = OutputConfig()
        assert config.csv_path is None
        assert config.include_minute_data is True
        assert config.include_summary is True
        assert config.aggregation == "minute"

    def test_custom_values(self) -> None:
        """Test custom output configuration."""
        config = OutputConfig(
            csv_path="/output/results.csv",
            include_minute_data=False,
            aggregation="daily",
        )
        assert config.csv_path == "/output/results.csv"
        assert config.include_minute_data is False
        assert config.aggregation == "daily"


class TestScenarioConfig:
    """Tests for ScenarioConfig class."""

    def test_single_home_scenario(self) -> None:
        """Test scenario with single home."""
        home = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(annual_consumption_kwh=3400),
        )
        scenario = ScenarioConfig(
            name="Test",
            period=SimulationPeriod("2024-01-01", "2024-01-07"),
            home=home,
        )
        assert not scenario.is_fleet
        assert scenario.home == home
        assert len(scenario.homes) == 0

    def test_fleet_scenario(self) -> None:
        """Test scenario with multiple homes."""
        homes = [
            HomeConfig(
                pv_config=PVConfig(capacity_kw=i),
                load_config=LoadConfig(annual_consumption_kwh=3000),
            )
            for i in [3.0, 4.0, 5.0]
        ]
        scenario = ScenarioConfig(
            name="Test Fleet",
            period=SimulationPeriod("2024-01-01", "2024-01-07"),
            homes=homes,
        )
        assert scenario.is_fleet
        assert len(scenario.homes) == 3

    def test_requires_home_or_homes(self) -> None:
        """Test that scenario requires home or homes."""
        with pytest.raises(ConfigurationError, match="must define either"):
            ScenarioConfig(
                name="Empty",
                period=SimulationPeriod("2024-01-01", "2024-01-07"),
            )

    def test_cannot_have_both(self) -> None:
        """Test that scenario cannot have both home and homes."""
        home = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(),
        )
        with pytest.raises(ConfigurationError, match="cannot define both"):
            ScenarioConfig(
                name="Both",
                period=SimulationPeriod("2024-01-01", "2024-01-07"),
                home=home,
                homes=[home],
            )

    def test_get_location_default(self) -> None:
        """Test default location is Bristol."""
        home = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(),
        )
        scenario = ScenarioConfig(
            name="Test",
            period=SimulationPeriod("2024-01-01", "2024-01-07"),
            home=home,
        )
        loc = scenario.get_location()
        assert loc.latitude == pytest.approx(51.45, abs=0.01)


class TestParameterSweepConfig:
    """Tests for ParameterSweepConfig class."""

    def test_explicit_values(self) -> None:
        """Test sweep with explicit values."""
        sweep = ParameterSweepConfig(
            parameter_name="battery_capacity_kwh",
            values=[0, 5, 10, 15],
        )
        assert sweep.get_values() == [0, 5, 10, 15]

    def test_range_with_step(self) -> None:
        """Test sweep with range and step."""
        sweep = ParameterSweepConfig(
            parameter_name="battery_capacity_kwh",
            min_value=0,
            max_value=10,
            step=2,
        )
        values = sweep.get_values()
        assert values == [0, 2, 4, 6, 8, 10]

    def test_range_with_n_steps(self) -> None:
        """Test sweep with range and n_steps."""
        sweep = ParameterSweepConfig(
            parameter_name="pv_capacity_kw",
            min_value=2,
            max_value=6,
            n_steps=4,
        )
        values = sweep.get_values()
        assert len(values) == 5
        assert values[0] == 2
        assert values[-1] == 6

    def test_empty_values_raises(self) -> None:
        """Test that empty values list raises error."""
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            ParameterSweepConfig(
                parameter_name="test",
                values=[],
            )

    def test_invalid_range_raises(self) -> None:
        """Test that invalid range raises error."""
        with pytest.raises(ConfigurationError, match="must be less than"):
            ParameterSweepConfig(
                parameter_name="test",
                min_value=10,
                max_value=5,
                step=1,
            )

    def test_missing_step_raises(self) -> None:
        """Test that missing step raises error."""
        with pytest.raises(ConfigurationError, match="requires either"):
            ParameterSweepConfig(
                parameter_name="test",
                min_value=0,
                max_value=10,
            )


class TestLoadConfigYaml:
    """Tests for YAML configuration loading."""

    def test_load_yaml_file(self) -> None:
        """Test loading a YAML configuration file."""
        yaml_content = """
name: Test Scenario
period:
  start_date: "2024-01-01"
  end_date: "2024-01-07"
home:
  pv:
    capacity_kw: 4.0
    azimuth: 180
    tilt: 35
  load:
    annual_consumption_kwh: 3400
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config_yaml(path)
            assert config["name"] == "Test Scenario"
            assert config["home"]["pv"]["capacity_kw"] == 4.0
        finally:
            path.unlink()

    def test_load_nonexistent_yaml_raises(self) -> None:
        """Test loading nonexistent YAML file raises error."""
        with pytest.raises(ConfigurationError, match="not found"):
            load_config_yaml("/nonexistent/path.yaml")


class TestLoadConfigJson:
    """Tests for JSON configuration loading."""

    def test_load_json_file(self) -> None:
        """Test loading a JSON configuration file."""
        json_content = {
            "name": "Test Scenario",
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-07",
            },
            "home": {
                "pv": {"capacity_kw": 4.0},
                "load": {"annual_consumption_kwh": 3400},
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config_json(path)
            assert config["name"] == "Test Scenario"
            assert config["home"]["pv"]["capacity_kw"] == 4.0
        finally:
            path.unlink()

    def test_load_invalid_json_raises(self) -> None:
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{ invalid json }")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError, match="Invalid JSON"):
                load_config_json(path)
        finally:
            path.unlink()


class TestLoadConfig:
    """Tests for auto-detecting configuration format."""

    def test_auto_detect_yaml(self) -> None:
        """Test auto-detecting YAML format."""
        yaml_content = "name: Test\nvalue: 123"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)
            assert config["name"] == "Test"
        finally:
            path.unlink()

    def test_auto_detect_yml(self) -> None:
        """Test auto-detecting .yml format."""
        yaml_content = "name: Test\nvalue: 123"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)
            assert config["name"] == "Test"
        finally:
            path.unlink()

    def test_auto_detect_json(self) -> None:
        """Test auto-detecting JSON format."""
        json_content = {"name": "Test", "value": 123}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)
            assert config["name"] == "Test"
        finally:
            path.unlink()

    def test_unknown_format_raises(self) -> None:
        """Test unknown format raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("some content")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError, match="Unknown.*format"):
                load_config(path)
        finally:
            path.unlink()


class TestLoadScenarios:
    """Tests for loading scenarios from configuration files."""

    def test_load_single_scenario(self) -> None:
        """Test loading a single scenario."""
        json_content = {
            "name": "Single Home Test",
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-07",
            },
            "home": {
                "pv": {"capacity_kw": 4.0},
                "load": {"annual_consumption_kwh": 3400},
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            scenarios = load_scenarios(path)
            assert len(scenarios) == 1
            assert scenarios[0].name == "Single Home Test"
            assert scenarios[0].home is not None
        finally:
            path.unlink()

    def test_load_multiple_scenarios(self) -> None:
        """Test loading multiple scenarios."""
        json_content = {
            "scenarios": [
                {
                    "name": "Scenario 1",
                    "period": {"start_date": "2024-01-01", "end_date": "2024-01-07"},
                    "home": {"pv": {"capacity_kw": 3.0}, "load": {}},
                },
                {
                    "name": "Scenario 2",
                    "period": {"start_date": "2024-01-01", "end_date": "2024-01-07"},
                    "home": {"pv": {"capacity_kw": 5.0}, "load": {}},
                },
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            scenarios = load_scenarios(path)
            assert len(scenarios) == 2
            assert scenarios[0].name == "Scenario 1"
            assert scenarios[1].name == "Scenario 2"
        finally:
            path.unlink()


class TestLoadHomeConfig:
    """Tests for loading home configuration."""

    def test_load_home_config(self) -> None:
        """Test loading home configuration from file."""
        json_content = {
            "home": {
                "pv": {"capacity_kw": 5.0, "tilt": 30},
                "battery": {"capacity_kwh": 10.0},
                "load": {"annual_consumption_kwh": 4000},
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            home = load_home_config(path)
            assert home.pv_config.capacity_kw == 5.0
            assert home.pv_config.tilt == 30
            assert home.battery_config is not None
            assert home.battery_config.capacity_kwh == 10.0
            assert home.load_config.annual_consumption_kwh == 4000
        finally:
            path.unlink()

    def test_load_home_without_battery(self) -> None:
        """Test loading home without battery."""
        json_content = {
            "pv": {"capacity_kw": 4.0},
            "load": {"annual_consumption_kwh": 3400},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            home = load_home_config(path)
            assert home.pv_config.capacity_kw == 4.0
            assert home.battery_config is None
        finally:
            path.unlink()


class TestLoadFleetConfig:
    """Tests for loading fleet configuration."""

    def test_load_fleet_config(self) -> None:
        """Test loading fleet configuration from file."""
        json_content = {
            "name": "Test Fleet",
            "homes": [
                {"pv": {"capacity_kw": 3.0}, "load": {}},
                {"pv": {"capacity_kw": 4.0}, "load": {}},
                {"pv": {"capacity_kw": 5.0}, "load": {}},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            fleet = load_fleet_config(path)
            assert fleet.name == "Test Fleet"
            assert len(fleet.homes) == 3
        finally:
            path.unlink()

    def test_load_fleet_requires_homes(self) -> None:
        """Test that fleet config requires homes list."""
        json_content = {"name": "Empty Fleet"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(json_content, f)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError, match="requires 'homes'"):
                load_fleet_config(path)
        finally:
            path.unlink()


class TestBristolPhase1Scenario:
    """Tests for Bristol Phase 1 default scenario."""

    def test_create_bristol_phase1(self) -> None:
        """Test creating Bristol Phase 1 scenario."""
        scenario = create_bristol_phase1_scenario()

        assert scenario.name == "Bristol Phase 1"
        assert len(scenario.homes) == 100
        assert scenario.location is not None
        assert scenario.location.latitude == pytest.approx(51.45, abs=0.01)

    def test_pv_distribution(self) -> None:
        """Test PV capacity distribution."""
        scenario = create_bristol_phase1_scenario()

        pv_sizes = [h.pv_config.capacity_kw for h in scenario.homes]

        # Check all sizes are in expected range
        assert all(3.0 <= s <= 6.0 for s in pv_sizes)

        # Check rough distribution
        count_3kw = sum(1 for s in pv_sizes if s == 3.0)
        count_4kw = sum(1 for s in pv_sizes if s == 4.0)
        count_5kw = sum(1 for s in pv_sizes if s == 5.0)
        count_6kw = sum(1 for s in pv_sizes if s == 6.0)

        assert count_3kw == 20
        assert count_4kw == 40
        assert count_5kw == 30
        assert count_6kw == 10

    def test_battery_distribution(self) -> None:
        """Test battery distribution."""
        scenario = create_bristol_phase1_scenario()

        no_battery = sum(1 for h in scenario.homes if h.battery_config is None)
        battery_5kwh = sum(
            1 for h in scenario.homes
            if h.battery_config is not None and h.battery_config.capacity_kwh == 5.0
        )
        battery_10kwh = sum(
            1 for h in scenario.homes
            if h.battery_config is not None and h.battery_config.capacity_kwh == 10.0
        )

        assert no_battery == 40
        assert battery_5kwh == 40
        assert battery_10kwh == 20

    def test_consumption_distribution(self) -> None:
        """Test consumption distribution."""
        scenario = create_bristol_phase1_scenario()

        consumptions = [
            h.load_config.annual_consumption_kwh
            for h in scenario.homes
            if h.load_config.annual_consumption_kwh is not None
        ]

        # All should be in valid range
        assert all(2000 <= c <= 6000 for c in consumptions)

        # Mean should be around 3400
        mean_consumption = sum(consumptions) / len(consumptions)
        assert 3000 <= mean_consumption <= 3800

    def test_reproducible(self) -> None:
        """Test that scenario is reproducible (seeded random)."""
        scenario1 = create_bristol_phase1_scenario()
        scenario2 = create_bristol_phase1_scenario()

        for h1, h2 in zip(scenario1.homes, scenario2.homes, strict=True):
            assert h1.pv_config.capacity_kw == h2.pv_config.capacity_kw
            assert h1.battery_config == h2.battery_config
