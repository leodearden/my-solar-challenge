"""Tests for fleet simulation."""

import pandas as pd
import pytest
from solar_challenge.battery import BatteryConfig
from solar_challenge.fleet import (
    FleetConfig,
    FleetResults,
    FleetSummary,
    calculate_fleet_summary,
)
from solar_challenge.home import SimulationResults
from solar_challenge.load import LoadConfig
from solar_challenge.location import Location
from solar_challenge.pv import PVConfig


class TestFleetConfigBasics:
    """Test FLEET-001: FleetConfig functionality."""

    def test_create_with_homes_list(self):
        """FleetConfig can be created with list of homes."""
        homes = [
            HomeConfig(
                pv_config=PVConfig(capacity_kw=4.0),
                load_config=LoadConfig(),
            )
            for _ in range(3)
        ]
        config = FleetConfig(homes=homes, name="Test Fleet")

        assert len(config.homes) == 3
        assert config.name == "Test Fleet"

    def test_requires_at_least_one_home(self):
        """Fleet must have at least one home."""
        with pytest.raises(ValueError, match="at least one home"):
            FleetConfig(homes=[])

    def test_validates_consistent_timezones(self):
        """All homes must have same timezone."""
        from solar_challenge.home import HomeConfig

        home1 = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(),
            location=Location(latitude=51.5, longitude=-0.1, timezone="Europe/London"),
        )
        home2 = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(),
            location=Location(latitude=40.7, longitude=-74.0, timezone="America/New_York"),
        )

        with pytest.raises(ValueError, match="timezone"):
            FleetConfig(homes=[home1, home2])


class TestFleetConfigCreation:
    """Test fleet creation convenience methods."""

    def test_create_uniform_fleet(self):
        """FLEET-001: Create uniform fleet with same config."""
        config = FleetConfig.create_uniform(
            n_homes=5,
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(annual_consumption_kwh=3400.0),
            battery_config=BatteryConfig(capacity_kwh=5.0),
            name="Uniform Fleet",
        )

        assert len(config.homes) == 5
        assert all(h.pv_config.capacity_kw == 4.0 for h in config.homes)
        assert all(h.battery_config is not None for h in config.homes)

    def test_create_heterogeneous_fleet(self):
        """FLEET-002: Create fleet with varied configs."""
        config = FleetConfig.create_heterogeneous(
            pv_capacities_kw=[3.0, 4.0, 5.0, 6.0],
            battery_capacities_kwh=[None, 5.0, 5.0, 10.0],
            annual_consumptions_kwh=[2800, 3400, 3400, 4200],
        )

        assert len(config.homes) == 4
        assert config.homes[0].pv_config.capacity_kw == 3.0
        assert config.homes[0].battery_config is None
        assert config.homes[1].battery_config is not None
        assert config.homes[3].pv_config.capacity_kw == 6.0

    def test_heterogeneous_requires_matching_lengths(self):
        """Heterogeneous config lists must have same length."""
        with pytest.raises(ValueError, match="same length"):
            FleetConfig.create_heterogeneous(
                pv_capacities_kw=[3.0, 4.0],
                battery_capacities_kwh=[None, 5.0, 5.0],  # Wrong length
                annual_consumptions_kwh=[3400, 3400],
            )


# Import HomeConfig for tests
from solar_challenge.home import HomeConfig


class TestFleetResults:
    """Test FLEET-004/005: Fleet results functionality."""

    @pytest.fixture
    def sample_results(self) -> FleetResults:
        """Create sample fleet results for 2 homes."""
        index = pd.date_range("2024-06-21 00:00", periods=1440, freq="1min")

        home1_results = SimulationResults(
            generation=pd.Series([3.0] * 1440, index=index),
            demand=pd.Series([2.0] * 1440, index=index),
            self_consumption=pd.Series([2.0] * 1440, index=index),
            battery_charge=pd.Series([0.5] * 1440, index=index),
            battery_discharge=pd.Series([0.0] * 1440, index=index),
            battery_soc=pd.Series([2.5] * 1440, index=index),
            grid_import=pd.Series([0.0] * 1440, index=index),
            grid_export=pd.Series([0.5] * 1440, index=index),
        )

        home2_results = SimulationResults(
            generation=pd.Series([4.0] * 1440, index=index),
            demand=pd.Series([3.0] * 1440, index=index),
            self_consumption=pd.Series([3.0] * 1440, index=index),
            battery_charge=pd.Series([0.5] * 1440, index=index),
            battery_discharge=pd.Series([0.0] * 1440, index=index),
            battery_soc=pd.Series([2.5] * 1440, index=index),
            grid_import=pd.Series([0.0] * 1440, index=index),
            grid_export=pd.Series([0.5] * 1440, index=index),
        )

        configs = [
            HomeConfig(pv_config=PVConfig(capacity_kw=3.0), load_config=LoadConfig()),
            HomeConfig(pv_config=PVConfig(capacity_kw=4.0), load_config=LoadConfig()),
        ]

        return FleetResults(
            per_home_results=[home1_results, home2_results],
            home_configs=configs,
        )

    def test_len_returns_num_homes(self, sample_results):
        """len() returns number of homes."""
        assert len(sample_results) == 2

    def test_indexing_returns_home_results(self, sample_results):
        """Indexing returns per-home results."""
        assert isinstance(sample_results[0], SimulationResults)
        assert sample_results[0].generation.iloc[0] == 3.0
        assert sample_results[1].generation.iloc[0] == 4.0

    def test_total_generation_sums_homes(self, sample_results):
        """total_generation sums across all homes."""
        total = sample_results.total_generation
        # 3.0 + 4.0 = 7.0 kW constant
        assert (total == 7.0).all()

    def test_total_demand_sums_homes(self, sample_results):
        """total_demand sums across all homes."""
        total = sample_results.total_demand
        # 2.0 + 3.0 = 5.0 kW constant
        assert (total == 5.0).all()

    def test_to_aggregate_dataframe(self, sample_results):
        """Converts aggregate results to DataFrame."""
        df = sample_results.to_aggregate_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "generation_kw" in df.columns
        assert "demand_kw" in df.columns
        assert len(df) == 1440


class TestFleetSummary:
    """Test FLEET-006: Fleet summary statistics."""

    @pytest.fixture
    def sample_results(self) -> FleetResults:
        """Create sample fleet results."""
        index = pd.date_range("2024-06-21 00:00", periods=1440, freq="1min")

        results = []
        configs = []
        for i, gen_kw in enumerate([3.0, 4.0, 5.0]):
            results.append(
                SimulationResults(
                    generation=pd.Series([gen_kw] * 1440, index=index),
                    demand=pd.Series([2.0] * 1440, index=index),
                    self_consumption=pd.Series([2.0] * 1440, index=index),
                    battery_charge=pd.Series([0.0] * 1440, index=index),
                    battery_discharge=pd.Series([0.0] * 1440, index=index),
                    battery_soc=pd.Series([0.0] * 1440, index=index),
                    grid_import=pd.Series([0.0] * 1440, index=index),
                    grid_export=pd.Series([gen_kw - 2.0] * 1440, index=index),
                )
            )
            configs.append(
                HomeConfig(pv_config=PVConfig(capacity_kw=gen_kw), load_config=LoadConfig())
            )

        return FleetResults(per_home_results=results, home_configs=configs)

    def test_calculates_fleet_totals(self, sample_results):
        """Calculates fleet-wide totals."""
        summary = calculate_fleet_summary(sample_results)

        assert summary.n_homes == 3

        # Total generation: (3+4+5) kW * 24 hours = 288 kWh
        assert summary.total_generation_kwh == pytest.approx(288.0, rel=0.01)

        # Total demand: 3 homes * 2 kW * 24 hours = 144 kWh
        assert summary.total_demand_kwh == pytest.approx(144.0, rel=0.01)

    def test_calculates_distribution_stats(self, sample_results):
        """Calculates distribution stats across homes."""
        summary = calculate_fleet_summary(sample_results)

        # Per-home generation: 72, 96, 120 kWh
        assert summary.per_home_generation_min_kwh == pytest.approx(72.0, rel=0.01)
        assert summary.per_home_generation_max_kwh == pytest.approx(120.0, rel=0.01)
        assert summary.per_home_generation_mean_kwh == pytest.approx(96.0, rel=0.01)
        assert summary.per_home_generation_median_kwh == pytest.approx(96.0, rel=0.01)

    def test_calculates_fleet_ratios(self, sample_results):
        """Calculates fleet-level efficiency ratios."""
        summary = calculate_fleet_summary(sample_results)

        # Self-consumption = 144 kWh (all demand met by PV)
        # Generation = 288 kWh
        # Fleet SC ratio = 144/288 = 0.5
        assert summary.fleet_self_consumption_ratio == pytest.approx(0.5, rel=0.01)

        # Grid dependency = 0 (no imports)
        assert summary.fleet_grid_dependency_ratio == 0.0
