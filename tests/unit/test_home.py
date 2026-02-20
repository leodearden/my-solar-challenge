"""Tests for home simulation."""

import pandas as pd
import pytest
from solar_challenge.battery import BatteryConfig
from solar_challenge.home import (
    HomeConfig,
    SimulationResults,
    SummaryStatistics,
    _align_tmy_to_demand,
    calculate_summary,
)
from solar_challenge.load import LoadConfig
from solar_challenge.location import Location
from solar_challenge.pv import PVConfig


class TestHomeConfigBasics:
    """Test HOME-001: HomeConfig dataclass."""

    def test_create_with_all_params(self):
        """HomeConfig can be created with all parameters."""
        config = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(annual_consumption_kwh=3400.0),
            battery_config=BatteryConfig(capacity_kwh=5.0),
            location=Location.bristol(),
            name="Test home",
        )
        assert config.pv_config.capacity_kw == 4.0
        assert config.load_config.annual_consumption_kwh == 3400.0
        assert config.battery_config is not None
        assert config.battery_config.capacity_kwh == 5.0
        assert config.name == "Test home"

    def test_battery_optional(self):
        """Battery config is optional (PV-only home)."""
        config = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(),
        )
        assert config.battery_config is None

    def test_default_location_is_bristol(self):
        """Default location is Bristol."""
        config = HomeConfig(
            pv_config=PVConfig(capacity_kw=4.0),
            load_config=LoadConfig(),
        )
        assert config.location.latitude == pytest.approx(51.45, rel=0.01)


class TestSimulationResults:
    """Test SimulationResults functionality."""

    @pytest.fixture
    def sample_results(self) -> SimulationResults:
        """Create sample simulation results."""
        index = pd.date_range("2024-06-21 10:00", periods=60, freq="1min")
        return SimulationResults(
            generation=pd.Series([2.0] * 60, index=index),
            demand=pd.Series([1.0] * 60, index=index),
            self_consumption=pd.Series([1.0] * 60, index=index),
            battery_charge=pd.Series([0.5] * 60, index=index),
            battery_discharge=pd.Series([0.0] * 60, index=index),
            battery_soc=pd.Series([2.5] * 60, index=index),
            grid_import=pd.Series([0.0] * 60, index=index),
            grid_export=pd.Series([0.5] * 60, index=index),
        )

    def test_to_dataframe(self, sample_results):
        """Results can be converted to DataFrame."""
        df = sample_results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 60
        assert "generation_kw" in df.columns
        assert "demand_kw" in df.columns
        assert "battery_soc_kwh" in df.columns


class TestAlignTMYToDemand:
    """Test TMY data alignment."""

    def test_aligns_by_time_of_year(self):
        """TMY data aligned by month-day-hour-minute."""
        # TMY data for June 21
        tmy_index = pd.date_range("2024-06-21 10:00", periods=60, freq="1min")
        tmy_gen = pd.Series(range(60), index=tmy_index, dtype=float)

        # Demand for same time in a different year
        demand_index = pd.date_range("2025-06-21 10:00", periods=60, freq="1min")
        demand = pd.Series([1.0] * 60, index=demand_index)

        aligned = _align_tmy_to_demand(tmy_gen, demand)

        assert len(aligned) == 60
        # Values should be preserved from TMY
        assert aligned.iloc[0] == 0.0
        assert aligned.iloc[59] == 59.0

    def test_missing_tmy_data_returns_zero(self):
        """Missing TMY timestamps return zero."""
        # TMY data only for noon
        tmy_index = pd.date_range("2024-06-21 12:00", periods=1, freq="1min")
        tmy_gen = pd.Series([5.0], index=tmy_index)

        # Demand for earlier time
        demand_index = pd.date_range("2025-06-21 10:00", periods=60, freq="1min")
        demand = pd.Series([1.0] * 60, index=demand_index)

        aligned = _align_tmy_to_demand(tmy_gen, demand)

        # Most values should be zero since TMY data doesn't cover this time
        assert aligned.iloc[0] == 0.0


class TestCalculateSummary:
    """Test HOME-005: Summary statistics calculation."""

    @pytest.fixture
    def sample_results(self) -> SimulationResults:
        """Create sample results for 1 day (1440 minutes)."""
        index = pd.date_range("2024-06-21 00:00", periods=1440, freq="1min")
        return SimulationResults(
            generation=pd.Series([3.0] * 1440, index=index),  # 3 kW constant
            demand=pd.Series([2.0] * 1440, index=index),  # 2 kW constant
            self_consumption=pd.Series([2.0] * 1440, index=index),
            battery_charge=pd.Series([0.5] * 1440, index=index),
            battery_discharge=pd.Series([0.0] * 1440, index=index),
            battery_soc=pd.Series([2.5] * 1440, index=index),
            grid_import=pd.Series([0.0] * 1440, index=index),
            grid_export=pd.Series([0.5] * 1440, index=index),
        )

    def test_calculates_totals(self, sample_results):
        """Calculates total energy values."""
        summary = calculate_summary(sample_results)

        # 3 kW for 1440 minutes = 3 * 24 = 72 kWh generation
        assert summary.total_generation_kwh == pytest.approx(72.0, rel=0.01)

        # 2 kW for 1440 minutes = 48 kWh demand
        assert summary.total_demand_kwh == pytest.approx(48.0, rel=0.01)

    def test_calculates_peaks(self, sample_results):
        """Calculates peak values."""
        summary = calculate_summary(sample_results)
        assert summary.peak_generation_kw == 3.0
        assert summary.peak_demand_kw == 2.0

    def test_calculates_ratios(self, sample_results):
        """Calculates efficiency ratios."""
        summary = calculate_summary(sample_results)

        # self_consumption_ratio = 48/72 = 0.667
        assert summary.self_consumption_ratio == pytest.approx(0.667, rel=0.01)

        # grid_dependency = 0/48 = 0
        assert summary.grid_dependency_ratio == 0.0

        # export_ratio = 12/72 = 0.167
        assert summary.export_ratio == pytest.approx(0.167, rel=0.01)

    def test_handles_zero_generation(self):
        """Handles zero generation gracefully."""
        index = pd.date_range("2024-06-21 00:00", periods=60, freq="1min")
        results = SimulationResults(
            generation=pd.Series([0.0] * 60, index=index),
            demand=pd.Series([1.0] * 60, index=index),
            self_consumption=pd.Series([0.0] * 60, index=index),
            battery_charge=pd.Series([0.0] * 60, index=index),
            battery_discharge=pd.Series([0.0] * 60, index=index),
            battery_soc=pd.Series([0.0] * 60, index=index),
            grid_import=pd.Series([1.0] * 60, index=index),
            grid_export=pd.Series([0.0] * 60, index=index),
        )

        summary = calculate_summary(results)
        assert summary.self_consumption_ratio == 0.0
        assert summary.export_ratio == 0.0

    def test_SEG_revenue_with_tariff(self, sample_results):
        """SEG revenue is computed when tariff is provided."""
        # total_grid_export_kwh = 0.5 kW * 24 h = 12 kWh
        # seg_revenue_gbp = 12 * 15 / 100 = 1.80 GBP
        summary = calculate_summary(sample_results, seg_tariff_pence_per_kwh=15.0)

        assert summary.seg_revenue_gbp is not None
        assert summary.seg_revenue_gbp == pytest.approx(1.80, rel=0.01)

    def test_SEG_revenue_without_tariff(self, sample_results):
        """seg_revenue_gbp is None when no tariff is provided."""
        summary = calculate_summary(sample_results)

        assert summary.seg_revenue_gbp is None


class TestSummaryStatistics:
    """Test SummaryStatistics dataclass."""

    def test_all_fields_present(self):
        """SummaryStatistics has all required fields."""
        stats = SummaryStatistics(
            total_generation_kwh=100.0,
            total_demand_kwh=80.0,
            total_self_consumption_kwh=60.0,
            total_grid_import_kwh=20.0,
            total_grid_export_kwh=40.0,
            total_battery_charge_kwh=15.0,
            total_battery_discharge_kwh=15.0,
            peak_generation_kw=4.0,
            peak_demand_kw=3.0,
            self_consumption_ratio=0.6,
            grid_dependency_ratio=0.25,
            export_ratio=0.4,
            simulation_days=7,
        )
        assert stats.total_generation_kwh == 100.0
        assert stats.simulation_days == 7
