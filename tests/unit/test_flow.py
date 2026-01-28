"""Tests for energy flow calculations."""

import pytest
import pandas as pd
import numpy as np
from solar_challenge.flow import (
    calculate_self_consumption,
    calculate_excess_pv,
    calculate_shortfall,
)


@pytest.fixture
def sample_index():
    """Create a sample datetime index."""
    return pd.date_range("2024-01-01", periods=5, freq="h")


@pytest.fixture
def sample_generation(sample_index):
    """Sample generation series: [0, 1, 3, 2, 0] kW."""
    return pd.Series([0.0, 1.0, 3.0, 2.0, 0.0], index=sample_index, name="gen")


@pytest.fixture
def sample_demand(sample_index):
    """Sample demand series: [0.5, 0.5, 1.0, 2.5, 1.0] kW."""
    return pd.Series([0.5, 0.5, 1.0, 2.5, 1.0], index=sample_index, name="demand")


class TestSelfConsumption:
    """Test self-consumption calculation."""

    def test_self_consumption_is_min(self, sample_generation, sample_demand):
        """Self-consumption is min(generation, demand)."""
        result = calculate_self_consumption(sample_generation, sample_demand)
        expected = pd.Series([0.0, 0.5, 1.0, 2.0, 0.0], index=sample_generation.index)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_self_consumption_same_length(self, sample_generation, sample_demand):
        """Result has same length as inputs."""
        result = calculate_self_consumption(sample_generation, sample_demand)
        assert len(result) == len(sample_generation)

    def test_self_consumption_non_negative(self, sample_generation, sample_demand):
        """All values are non-negative."""
        result = calculate_self_consumption(sample_generation, sample_demand)
        assert (result >= 0).all()

    def test_mismatched_lengths_raises(self, sample_generation):
        """Different length series raises error."""
        short_demand = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            calculate_self_consumption(sample_generation, short_demand)

    def test_negative_generation_raises(self, sample_index, sample_demand):
        """Negative generation values raise error."""
        bad_gen = pd.Series([-1.0, 1.0, 1.0, 1.0, 1.0], index=sample_index)
        with pytest.raises(ValueError, match="negative"):
            calculate_self_consumption(bad_gen, sample_demand)

    def test_negative_demand_raises(self, sample_index, sample_generation):
        """Negative demand values raise error."""
        bad_demand = pd.Series([1.0, -1.0, 1.0, 1.0, 1.0], index=sample_index)
        with pytest.raises(ValueError, match="negative"):
            calculate_self_consumption(sample_generation, bad_demand)


class TestExcessPV:
    """Test excess PV calculation."""

    def test_excess_when_generation_higher(self, sample_generation, sample_demand):
        """Excess = generation - demand when positive."""
        result = calculate_excess_pv(sample_generation, sample_demand)
        # [0-0.5, 1-0.5, 3-1, 2-2.5, 0-1] = [-0.5, 0.5, 2, -0.5, -1] -> [0, 0.5, 2, 0, 0]
        expected = pd.Series([0.0, 0.5, 2.0, 0.0, 0.0], index=sample_generation.index)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_excess_same_length(self, sample_generation, sample_demand):
        """Result has same length as inputs."""
        result = calculate_excess_pv(sample_generation, sample_demand)
        assert len(result) == len(sample_generation)

    def test_excess_non_negative(self, sample_generation, sample_demand):
        """All values are non-negative."""
        result = calculate_excess_pv(sample_generation, sample_demand)
        assert (result >= 0).all()


class TestShortfall:
    """Test shortfall calculation."""

    def test_shortfall_when_demand_higher(self, sample_generation, sample_demand):
        """Shortfall = demand - generation when positive."""
        result = calculate_shortfall(sample_generation, sample_demand)
        # [0.5-0, 0.5-1, 1-3, 2.5-2, 1-0] = [0.5, -0.5, -2, 0.5, 1] -> [0.5, 0, 0, 0.5, 1]
        expected = pd.Series([0.5, 0.0, 0.0, 0.5, 1.0], index=sample_generation.index)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_shortfall_same_length(self, sample_generation, sample_demand):
        """Result has same length as inputs."""
        result = calculate_shortfall(sample_generation, sample_demand)
        assert len(result) == len(sample_generation)

    def test_shortfall_non_negative(self, sample_generation, sample_demand):
        """All values are non-negative."""
        result = calculate_shortfall(sample_generation, sample_demand)
        assert (result >= 0).all()


class TestEnergyBalance:
    """Test that flow calculations maintain energy balance."""

    def test_self_consumption_plus_excess_equals_generation(
        self, sample_generation, sample_demand
    ):
        """Self-consumption + excess = generation."""
        self_consumption = calculate_self_consumption(sample_generation, sample_demand)
        excess = calculate_excess_pv(sample_generation, sample_demand)
        total = self_consumption + excess
        pd.testing.assert_series_equal(
            total, sample_generation, check_names=False, atol=1e-10
        )

    def test_self_consumption_plus_shortfall_equals_demand(
        self, sample_generation, sample_demand
    ):
        """Self-consumption + shortfall = demand."""
        self_consumption = calculate_self_consumption(sample_generation, sample_demand)
        shortfall = calculate_shortfall(sample_generation, sample_demand)
        total = self_consumption + shortfall
        pd.testing.assert_series_equal(
            total, sample_demand, check_names=False, atol=1e-10
        )
