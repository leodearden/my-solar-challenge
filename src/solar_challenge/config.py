"""Configuration file support for simulation scenarios.

Supports loading configurations from YAML and JSON files,
scenario definitions, and parameter sweep functionality.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import pandas as pd

# Import yaml with fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

from solar_challenge.battery import BatteryConfig
from solar_challenge.fleet import FleetConfig, FleetResults, simulate_fleet
from solar_challenge.home import HomeConfig, SimulationResults, simulate_home
from solar_challenge.load import LoadConfig
from solar_challenge.location import Location
from solar_challenge.pv import PVConfig


class ConfigurationError(Exception):
    """Raised when configuration file is invalid."""

    pass


@dataclass
class SimulationPeriod:
    """Defines the time period for a simulation.

    Attributes:
        start_date: Start date as string (YYYY-MM-DD) or Timestamp
        end_date: End date as string (YYYY-MM-DD) or Timestamp
    """

    start_date: Union[str, pd.Timestamp]
    end_date: Union[str, pd.Timestamp]

    def get_start_timestamp(self, timezone: str = "Europe/London") -> pd.Timestamp:
        """Get start date as Timestamp."""
        if isinstance(self.start_date, pd.Timestamp):
            return self.start_date
        ts = pd.Timestamp(self.start_date)
        if ts.tz is None:
            ts = ts.tz_localize(timezone)
        return ts

    def get_end_timestamp(self, timezone: str = "Europe/London") -> pd.Timestamp:
        """Get end date as Timestamp."""
        if isinstance(self.end_date, pd.Timestamp):
            return self.end_date
        ts = pd.Timestamp(self.end_date)
        if ts.tz is None:
            ts = ts.tz_localize(timezone)
        return ts


@dataclass
class OutputConfig:
    """Configuration for simulation output.

    Attributes:
        csv_path: Path to save CSV results (optional)
        include_minute_data: Include 1-minute resolution data in output
        include_summary: Include summary statistics
        aggregation: Aggregation level for output (minute, daily, monthly, annual)
    """

    csv_path: Optional[str] = None
    include_minute_data: bool = True
    include_summary: bool = True
    aggregation: str = "minute"


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario.

    Attributes:
        name: Scenario identifier
        description: Human-readable description
        location: Geographic location (defaults to Bristol)
        period: Simulation period
        homes: List of home configurations (for fleet simulation)
        home: Single home configuration (for single-home simulation)
        output: Output preferences
    """

    name: str
    period: SimulationPeriod
    description: str = ""
    location: Optional[Location] = None
    homes: list[HomeConfig] = field(default_factory=list)
    home: Optional[HomeConfig] = None
    output: Optional[OutputConfig] = None

    def __post_init__(self) -> None:
        """Validate scenario configuration."""
        if not self.homes and self.home is None:
            raise ConfigurationError(
                f"Scenario '{self.name}' must define either 'home' or 'homes'"
            )
        if self.homes and self.home is not None:
            raise ConfigurationError(
                f"Scenario '{self.name}' cannot define both 'home' and 'homes'"
            )

    @property
    def is_fleet(self) -> bool:
        """Whether this is a fleet simulation."""
        return len(self.homes) > 1 or (len(self.homes) == 1 and self.home is None)

    def get_location(self) -> Location:
        """Get location, defaulting to Bristol."""
        return self.location if self.location is not None else Location.bristol()


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweep analysis.

    Attributes:
        parameter_name: Name of parameter to sweep (e.g., "battery_capacity_kwh")
        values: Explicit list of values to test
        min_value: Minimum value for range generation
        max_value: Maximum value for range generation
        step: Step size for range generation
        n_steps: Number of steps (alternative to step)
    """

    parameter_name: str
    values: Optional[list[float]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    n_steps: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate sweep configuration."""
        if self.values is not None:
            if len(self.values) == 0:
                raise ConfigurationError("Parameter sweep values list cannot be empty")
            return

        if self.min_value is None or self.max_value is None:
            raise ConfigurationError(
                "Parameter sweep requires either 'values' list or 'min_value' and 'max_value'"
            )
        if self.min_value >= self.max_value:
            raise ConfigurationError(
                f"min_value ({self.min_value}) must be less than max_value ({self.max_value})"
            )
        if self.step is None and self.n_steps is None:
            raise ConfigurationError(
                "Parameter sweep requires either 'step' or 'n_steps'"
            )

    def get_values(self) -> list[float]:
        """Get list of parameter values to sweep."""
        if self.values is not None:
            return self.values

        if self.min_value is None or self.max_value is None:
            raise ConfigurationError("Range parameters not configured")

        if self.step is not None:
            values: list[float] = []
            current = self.min_value
            while current <= self.max_value:
                values.append(current)
                current += self.step
            return values
        elif self.n_steps is not None:
            step = (self.max_value - self.min_value) / self.n_steps
            return [self.min_value + i * step for i in range(self.n_steps + 1)]
        else:
            return [self.min_value, self.max_value]


@dataclass
class SweepResult:
    """Result from a single parameter sweep iteration.

    Attributes:
        parameter_value: The parameter value used
        results: Simulation results (SimulationResults or FleetResults)
    """

    parameter_value: float
    results: Union[SimulationResults, FleetResults]


def _parse_location(data: dict[str, Any]) -> Location:
    """Parse location from config data."""
    return Location(
        latitude=data.get("latitude", 51.45),
        longitude=data.get("longitude", -2.58),
        timezone=data.get("timezone", "Europe/London"),
        altitude=data.get("altitude", 11.0),
        name=data.get("name", ""),
    )


def _parse_pv_config(data: dict[str, Any]) -> PVConfig:
    """Parse PV configuration from config data."""
    return PVConfig(
        capacity_kw=data.get("capacity_kw", 4.0),
        azimuth=data.get("azimuth", 180.0),
        tilt=data.get("tilt", 35.0),
        name=data.get("name", ""),
        module_efficiency=data.get("module_efficiency", 0.20),
        temperature_coefficient=data.get("temperature_coefficient", -0.004),
        inverter_efficiency=data.get("inverter_efficiency", 0.96),
        inverter_capacity_kw=data.get("inverter_capacity_kw"),
    )


def _parse_battery_config(data: Optional[dict[str, Any]]) -> Optional[BatteryConfig]:
    """Parse battery configuration from config data."""
    if data is None:
        return None
    return BatteryConfig(
        capacity_kwh=data.get("capacity_kwh", 5.0),
        max_charge_kw=data.get("max_charge_kw", 2.5),
        max_discharge_kw=data.get("max_discharge_kw", 2.5),
        name=data.get("name", ""),
    )


def _parse_load_config(data: dict[str, Any]) -> LoadConfig:
    """Parse load configuration from config data."""
    return LoadConfig(
        annual_consumption_kwh=data.get("annual_consumption_kwh"),
        household_occupants=data.get("household_occupants", 3),
        name=data.get("name", ""),
        use_stochastic=data.get("use_stochastic", True),
    )


def _parse_home_config(data: dict[str, Any], location: Location) -> HomeConfig:
    """Parse home configuration from config data."""
    pv_data = data.get("pv", {})
    battery_data = data.get("battery")
    load_data = data.get("load", {})

    return HomeConfig(
        pv_config=_parse_pv_config(pv_data),
        battery_config=_parse_battery_config(battery_data),
        load_config=_parse_load_config(load_data),
        location=location,
        name=data.get("name", ""),
    )


def _parse_period(data: dict[str, Any]) -> SimulationPeriod:
    """Parse simulation period from config data."""
    if "start_date" not in data or "end_date" not in data:
        raise ConfigurationError("Simulation period requires 'start_date' and 'end_date'")
    return SimulationPeriod(
        start_date=data["start_date"],
        end_date=data["end_date"],
    )


def _parse_output_config(data: Optional[dict[str, Any]]) -> Optional[OutputConfig]:
    """Parse output configuration from config data."""
    if data is None:
        return None
    return OutputConfig(
        csv_path=data.get("csv_path"),
        include_minute_data=data.get("include_minute_data", True),
        include_summary=data.get("include_summary", True),
        aggregation=data.get("aggregation", "minute"),
    )


def _parse_scenario(data: dict[str, Any]) -> ScenarioConfig:
    """Parse a scenario from config data."""
    if "name" not in data:
        raise ConfigurationError("Scenario must have a 'name' field")
    if "period" not in data:
        raise ConfigurationError(f"Scenario '{data['name']}' must have a 'period' field")

    location_data = data.get("location")
    location = _parse_location(location_data) if location_data else Location.bristol()

    homes: list[HomeConfig] = []
    home: Optional[HomeConfig] = None

    if "homes" in data:
        for home_data in data["homes"]:
            homes.append(_parse_home_config(home_data, location))
    elif "home" in data:
        home = _parse_home_config(data["home"], location)
    else:
        raise ConfigurationError(
            f"Scenario '{data['name']}' must define either 'home' or 'homes'"
        )

    return ScenarioConfig(
        name=data["name"],
        description=data.get("description", ""),
        location=location,
        period=_parse_period(data["period"]),
        homes=homes,
        home=home,
        output=_parse_output_config(data.get("output")),
    )


def load_config_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        ConfigurationError: If file cannot be read or parsed
    """
    if not YAML_AVAILABLE:
        raise ConfigurationError(
            "YAML support requires pyyaml: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f)
        if config is None:
            raise ConfigurationError(f"Empty configuration file: {path}")
        return config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e


def load_config_json(path: Union[str, Path]) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        ConfigurationError: If file cannot be read or parsed
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}") from e


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Auto-detects format by file extension.

    Args:
        path: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        Parsed configuration dictionary

    Raises:
        ConfigurationError: If file cannot be read or format unknown
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return load_config_yaml(path)
    elif suffix == ".json":
        return load_config_json(path)
    else:
        raise ConfigurationError(
            f"Unknown configuration file format: {suffix}. "
            "Supported formats: .yaml, .yml, .json"
        )


def load_scenarios(path: Union[str, Path]) -> list[ScenarioConfig]:
    """Load scenarios from a configuration file.

    Args:
        path: Path to configuration file

    Returns:
        List of ScenarioConfig objects

    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = load_config(path)

    if "scenarios" in config:
        return [_parse_scenario(s) for s in config["scenarios"]]
    elif "scenario" in config:
        return [_parse_scenario(config["scenario"])]
    else:
        # Try to parse the entire config as a single scenario
        return [_parse_scenario(config)]


def load_home_config(path: Union[str, Path]) -> HomeConfig:
    """Load a single home configuration from file.

    Args:
        path: Path to configuration file

    Returns:
        HomeConfig object

    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = load_config(path)

    # Check for home section or parse entire config as home
    home_data = config.get("home", config)
    location_data = config.get("location")
    location = _parse_location(location_data) if location_data else Location.bristol()

    return _parse_home_config(home_data, location)


def load_fleet_config(path: Union[str, Path]) -> FleetConfig:
    """Load a fleet configuration from file.

    Args:
        path: Path to configuration file

    Returns:
        FleetConfig object

    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = load_config(path)

    location_data = config.get("location")
    location = _parse_location(location_data) if location_data else Location.bristol()

    homes_data = config.get("homes", [])
    if not homes_data:
        raise ConfigurationError("Fleet configuration requires 'homes' list")

    homes = [_parse_home_config(h, location) for h in homes_data]
    return FleetConfig(homes=homes, name=config.get("name", ""))


def create_bristol_phase1_scenario() -> ScenarioConfig:
    """Create the default Bristol Phase 1 scenario.

    This scenario represents 100 homes in Bristol with a realistic
    mix of PV sizes (3-6 kW), battery sizes (0, 5, 10 kWh),
    and consumption levels.

    Returns:
        ScenarioConfig for Bristol Phase 1
    """
    import random

    random.seed(42)  # Reproducible distribution

    location = Location.bristol()
    homes: list[HomeConfig] = []

    # Distribution:
    # PV: 20% 3kW, 40% 4kW, 30% 5kW, 10% 6kW
    pv_distribution = [3.0] * 20 + [4.0] * 40 + [5.0] * 30 + [6.0] * 10
    # Battery: 40% none, 40% 5kWh, 20% 10kWh
    battery_distribution: list[Optional[float]] = (
        [None] * 40 + [5.0] * 40 + [10.0] * 20
    )
    # Consumption: Normal distribution around 3400 kWh, std 800 kWh
    consumption_distribution = [
        max(2000.0, min(6000.0, random.gauss(3400, 800)))
        for _ in range(100)
    ]

    random.shuffle(pv_distribution)
    random.shuffle(battery_distribution)

    for i in range(100):
        pv_kw = pv_distribution[i]
        bat_kwh = battery_distribution[i]
        consumption = consumption_distribution[i]

        battery_config = (
            BatteryConfig(capacity_kwh=bat_kwh) if bat_kwh is not None else None
        )

        homes.append(
            HomeConfig(
                pv_config=PVConfig(capacity_kw=pv_kw),
                load_config=LoadConfig(annual_consumption_kwh=consumption),
                battery_config=battery_config,
                location=location,
                name=f"Home {i + 1}",
            )
        )

    return ScenarioConfig(
        name="Bristol Phase 1",
        description=(
            "100 homes in Bristol with realistic mix of PV (3-6 kW), "
            "batteries (0/5/10 kWh), and consumption levels"
        ),
        location=location,
        period=SimulationPeriod(
            start_date="2024-01-01",
            end_date="2024-12-31",
        ),
        homes=homes,
    )


def run_parameter_sweep(
    base_scenario: ScenarioConfig,
    sweep_config: ParameterSweepConfig,
) -> Iterator[SweepResult]:
    """Run a parameter sweep over a scenario.

    Yields results for each parameter value in the sweep.

    Args:
        base_scenario: Base scenario configuration
        sweep_config: Parameter sweep configuration

    Yields:
        SweepResult for each parameter value

    Raises:
        ConfigurationError: If parameter cannot be swept
    """
    values = sweep_config.get_values()
    param_name = sweep_config.parameter_name
    location = base_scenario.get_location()

    for value in values:
        # Create modified scenario for this parameter value
        if base_scenario.home is not None:
            # Single home simulation
            home = _apply_parameter_to_home(base_scenario.home, param_name, value, location)
            start = base_scenario.period.get_start_timestamp(location.timezone)
            end = base_scenario.period.get_end_timestamp(location.timezone)
            home_results = simulate_home(home, start, end)
            yield SweepResult(parameter_value=value, results=home_results)
        else:
            # Fleet simulation
            homes = [
                _apply_parameter_to_home(h, param_name, value, location)
                for h in base_scenario.homes
            ]
            fleet = FleetConfig(homes=homes, name=base_scenario.name)
            start = base_scenario.period.get_start_timestamp(location.timezone)
            end = base_scenario.period.get_end_timestamp(location.timezone)
            fleet_results = simulate_fleet(fleet, start, end)
            yield SweepResult(parameter_value=value, results=fleet_results)


def _apply_parameter_to_home(
    home: HomeConfig,
    param_name: str,
    value: float,
    location: Location,
) -> HomeConfig:
    """Apply a parameter value to a home configuration.

    Args:
        home: Base home configuration
        param_name: Parameter name to modify
        value: New parameter value
        location: Location for the home

    Returns:
        Modified HomeConfig
    """
    # Map parameter names to config modifications
    pv_params = {"pv_capacity_kw", "pv_tilt", "pv_azimuth"}
    battery_params = {"battery_capacity_kwh", "battery_charge_kw", "battery_discharge_kw"}
    load_params = {"annual_consumption_kwh", "household_occupants"}

    if param_name in pv_params:
        pv_config = _modify_pv_config(home.pv_config, param_name, value)
        return HomeConfig(
            pv_config=pv_config,
            load_config=home.load_config,
            battery_config=home.battery_config,
            location=location,
            name=home.name,
        )
    elif param_name in battery_params:
        battery_config = _modify_battery_config(home.battery_config, param_name, value)
        return HomeConfig(
            pv_config=home.pv_config,
            load_config=home.load_config,
            battery_config=battery_config,
            location=location,
            name=home.name,
        )
    elif param_name in load_params:
        load_config = _modify_load_config(home.load_config, param_name, value)
        return HomeConfig(
            pv_config=home.pv_config,
            load_config=load_config,
            battery_config=home.battery_config,
            location=location,
            name=home.name,
        )
    else:
        raise ConfigurationError(f"Unknown parameter for sweep: {param_name}")


def _modify_pv_config(config: PVConfig, param_name: str, value: float) -> PVConfig:
    """Modify PV config with new parameter value."""
    if param_name == "pv_capacity_kw":
        return PVConfig(
            capacity_kw=value,
            azimuth=config.azimuth,
            tilt=config.tilt,
            name=config.name,
            module_efficiency=config.module_efficiency,
            temperature_coefficient=config.temperature_coefficient,
            inverter_efficiency=config.inverter_efficiency,
            inverter_capacity_kw=config.inverter_capacity_kw,
        )
    elif param_name == "pv_tilt":
        return PVConfig(
            capacity_kw=config.capacity_kw,
            azimuth=config.azimuth,
            tilt=value,
            name=config.name,
            module_efficiency=config.module_efficiency,
            temperature_coefficient=config.temperature_coefficient,
            inverter_efficiency=config.inverter_efficiency,
            inverter_capacity_kw=config.inverter_capacity_kw,
        )
    elif param_name == "pv_azimuth":
        return PVConfig(
            capacity_kw=config.capacity_kw,
            azimuth=value,
            tilt=config.tilt,
            name=config.name,
            module_efficiency=config.module_efficiency,
            temperature_coefficient=config.temperature_coefficient,
            inverter_efficiency=config.inverter_efficiency,
            inverter_capacity_kw=config.inverter_capacity_kw,
        )
    return config


def _modify_battery_config(
    config: Optional[BatteryConfig],
    param_name: str,
    value: float,
) -> Optional[BatteryConfig]:
    """Modify battery config with new parameter value."""
    if config is None:
        if param_name == "battery_capacity_kwh" and value > 0:
            return BatteryConfig(capacity_kwh=value)
        return None

    if param_name == "battery_capacity_kwh":
        if value <= 0:
            return None  # Remove battery
        return BatteryConfig(
            capacity_kwh=value,
            max_charge_kw=config.max_charge_kw,
            max_discharge_kw=config.max_discharge_kw,
            name=config.name,
        )
    elif param_name == "battery_charge_kw":
        return BatteryConfig(
            capacity_kwh=config.capacity_kwh,
            max_charge_kw=value,
            max_discharge_kw=config.max_discharge_kw,
            name=config.name,
        )
    elif param_name == "battery_discharge_kw":
        return BatteryConfig(
            capacity_kwh=config.capacity_kwh,
            max_charge_kw=config.max_charge_kw,
            max_discharge_kw=value,
            name=config.name,
        )
    return config


def _modify_load_config(config: LoadConfig, param_name: str, value: float) -> LoadConfig:
    """Modify load config with new parameter value."""
    if param_name == "annual_consumption_kwh":
        return LoadConfig(
            annual_consumption_kwh=value,
            household_occupants=config.household_occupants,
            name=config.name,
            use_stochastic=config.use_stochastic,
        )
    elif param_name == "household_occupants":
        return LoadConfig(
            annual_consumption_kwh=config.annual_consumption_kwh,
            household_occupants=int(value),
            name=config.name,
            use_stochastic=config.use_stochastic,
        )
    return config
