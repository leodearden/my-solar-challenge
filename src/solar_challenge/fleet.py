"""Fleet simulation for multiple homes."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from solar_challenge.battery import BatteryConfig
from solar_challenge.home import (
    HomeConfig,
    SimulationResults,
    SummaryStatistics,
    calculate_summary,
    simulate_home,
)
from solar_challenge.load import LoadConfig
from solar_challenge.location import Location
from solar_challenge.pv import PVConfig


@dataclass
class FleetConfig:
    """Configuration for a fleet of homes.

    Attributes:
        homes: List of HomeConfig objects for each home
        name: Optional identifier for the fleet
    """

    homes: list[HomeConfig] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        """Validate fleet configuration."""
        if not self.homes:
            raise ValueError("Fleet must have at least one home")

        # Validate all homes have compatible timezones
        timezones = {h.location.timezone for h in self.homes}
        if len(timezones) > 1:
            raise ValueError(
                f"All homes must have same timezone. Found: {timezones}"
            )

    @classmethod
    def create_uniform(
        cls,
        n_homes: int,
        pv_config: PVConfig,
        load_config: LoadConfig,
        battery_config: Optional[BatteryConfig] = None,
        location: Location = Location.bristol(),
        name: str = "",
    ) -> "FleetConfig":
        """Create a fleet with uniform home configurations.

        Args:
            n_homes: Number of homes in the fleet
            pv_config: PV configuration for all homes
            load_config: Load configuration for all homes
            battery_config: Battery configuration (or None) for all homes
            location: Location for all homes
            name: Fleet name

        Returns:
            FleetConfig with identical home configurations
        """
        homes = [
            HomeConfig(
                pv_config=pv_config,
                load_config=load_config,
                battery_config=battery_config,
                location=location,
                name=f"Home {i+1}",
            )
            for i in range(n_homes)
        ]
        return cls(homes=homes, name=name)

    @classmethod
    def create_heterogeneous(
        cls,
        pv_capacities_kw: list[float],
        battery_capacities_kwh: list[Optional[float]],
        annual_consumptions_kwh: list[float],
        location: Location = Location.bristol(),
        name: str = "",
    ) -> "FleetConfig":
        """Create a fleet with heterogeneous home configurations.

        Args:
            pv_capacities_kw: PV capacity for each home
            battery_capacities_kwh: Battery capacity (or None) for each home
            annual_consumptions_kwh: Annual consumption for each home
            location: Location for all homes
            name: Fleet name

        Returns:
            FleetConfig with varied home configurations
        """
        if not (
            len(pv_capacities_kw)
            == len(battery_capacities_kwh)
            == len(annual_consumptions_kwh)
        ):
            raise ValueError("All configuration lists must have the same length")

        homes = []
        for i, (pv_kw, bat_kwh, load_kwh) in enumerate(
            zip(pv_capacities_kw, battery_capacities_kwh, annual_consumptions_kwh, strict=True)
        ):
            battery_config = (
                BatteryConfig(capacity_kwh=bat_kwh) if bat_kwh is not None else None
            )
            homes.append(
                HomeConfig(
                    pv_config=PVConfig(capacity_kw=pv_kw),
                    load_config=LoadConfig(annual_consumption_kwh=load_kwh),
                    battery_config=battery_config,
                    location=location,
                    name=f"Home {i+1}",
                )
            )

        return cls(homes=homes, name=name)


@dataclass
class FleetResults:
    """Results from a fleet simulation.

    Attributes:
        per_home_results: List of SimulationResults for each home
        home_configs: List of HomeConfig for each home (for reference)
    """

    per_home_results: list[SimulationResults]
    home_configs: list[HomeConfig]

    def __len__(self) -> int:
        """Return number of homes in fleet."""
        return len(self.per_home_results)

    def __getitem__(self, index: int) -> SimulationResults:
        """Get results for a specific home by index."""
        return self.per_home_results[index]

    def get_aggregate_series(self, series_name: str) -> pd.Series:
        """Get aggregate (sum) of a series across all homes.

        Args:
            series_name: Name of the series (e.g., 'generation', 'demand')

        Returns:
            Sum of the series across all homes
        """
        series_list = [getattr(r, series_name) for r in self.per_home_results]
        return sum(series_list[1:], series_list[0])

    @property
    def total_generation(self) -> pd.Series:
        """Total fleet generation (sum across homes)."""
        return self.get_aggregate_series("generation")

    @property
    def total_demand(self) -> pd.Series:
        """Total fleet demand (sum across homes)."""
        return self.get_aggregate_series("demand")

    @property
    def total_grid_import(self) -> pd.Series:
        """Total fleet grid import (sum across homes)."""
        return self.get_aggregate_series("grid_import")

    @property
    def total_grid_export(self) -> pd.Series:
        """Total fleet grid export (sum across homes)."""
        return self.get_aggregate_series("grid_export")

    @property
    def total_self_consumption(self) -> pd.Series:
        """Total fleet self-consumption (sum across homes)."""
        return self.get_aggregate_series("self_consumption")

    def to_aggregate_dataframe(self) -> pd.DataFrame:
        """Get aggregate results as DataFrame."""
        return pd.DataFrame({
            "generation_kw": self.total_generation,
            "demand_kw": self.total_demand,
            "self_consumption_kw": self.total_self_consumption,
            "grid_import_kw": self.total_grid_import,
            "grid_export_kw": self.total_grid_export,
        })


@dataclass
class FleetSummary:
    """Summary statistics for fleet simulation.

    All energy values in kWh.
    """

    n_homes: int
    total_generation_kwh: float
    total_demand_kwh: float
    total_self_consumption_kwh: float
    total_grid_import_kwh: float
    total_grid_export_kwh: float
    fleet_self_consumption_ratio: float
    fleet_grid_dependency_ratio: float

    # Distribution stats across homes
    per_home_generation_min_kwh: float
    per_home_generation_max_kwh: float
    per_home_generation_mean_kwh: float
    per_home_generation_median_kwh: float

    per_home_self_consumption_ratio_min: float
    per_home_self_consumption_ratio_max: float
    per_home_self_consumption_ratio_mean: float

    simulation_days: int


def simulate_fleet(
    config: FleetConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    validate_balance: bool = True,
) -> FleetResults:
    """Simulate all homes in a fleet for a date range.

    Weather data is retrieved once and shared across all homes
    (assumes same location).

    Args:
        config: Fleet configuration
        start_date: Start of simulation period
        end_date: End of simulation period (inclusive)
        validate_balance: Whether to validate energy balance

    Returns:
        FleetResults with per-home results
    """
    results: list[SimulationResults] = []

    for home_config in config.homes:
        home_results = simulate_home(
            home_config,
            start_date,
            end_date,
            validate_balance=validate_balance,
        )
        results.append(home_results)

    return FleetResults(
        per_home_results=results,
        home_configs=config.homes,
    )


def calculate_fleet_summary(results: FleetResults) -> FleetSummary:
    """Calculate summary statistics for fleet simulation.

    Args:
        results: Fleet simulation results

    Returns:
        FleetSummary with totals and distribution statistics
    """
    # Calculate per-home summaries
    home_summaries: list[SummaryStatistics] = [
        calculate_summary(r) for r in results.per_home_results
    ]

    # Fleet totals
    total_gen = sum(s.total_generation_kwh for s in home_summaries)
    total_demand = sum(s.total_demand_kwh for s in home_summaries)
    total_self = sum(s.total_self_consumption_kwh for s in home_summaries)
    total_import = sum(s.total_grid_import_kwh for s in home_summaries)
    total_export = sum(s.total_grid_export_kwh for s in home_summaries)

    # Fleet ratios
    fleet_self_ratio = total_self / total_gen if total_gen > 0 else 0.0
    fleet_grid_dep = total_import / total_demand if total_demand > 0 else 0.0

    # Per-home generation distribution
    gen_values = [s.total_generation_kwh for s in home_summaries]
    gen_series = pd.Series(gen_values)

    # Per-home self-consumption ratio distribution
    sc_ratios = [s.self_consumption_ratio for s in home_summaries]
    sc_series = pd.Series(sc_ratios)

    return FleetSummary(
        n_homes=len(results),
        total_generation_kwh=total_gen,
        total_demand_kwh=total_demand,
        total_self_consumption_kwh=total_self,
        total_grid_import_kwh=total_import,
        total_grid_export_kwh=total_export,
        fleet_self_consumption_ratio=fleet_self_ratio,
        fleet_grid_dependency_ratio=fleet_grid_dep,
        per_home_generation_min_kwh=float(gen_series.min()),
        per_home_generation_max_kwh=float(gen_series.max()),
        per_home_generation_mean_kwh=float(gen_series.mean()),
        per_home_generation_median_kwh=float(gen_series.median()),
        per_home_self_consumption_ratio_min=float(sc_series.min()),
        per_home_self_consumption_ratio_max=float(sc_series.max()),
        per_home_self_consumption_ratio_mean=float(sc_series.mean()),
        simulation_days=home_summaries[0].simulation_days if home_summaries else 0,
    )
