"""Tests for weather data handling."""

import pytest
import pandas as pd
import numpy as np

from solar_challenge.weather import (
    validate_irradiance_data,
    extract_temperature_data,
)


@pytest.fixture
def sample_index():
    """Create sample datetime index."""
    return pd.date_range("2024-01-01", periods=24, freq="h")


@pytest.fixture
def valid_weather_data(sample_index):
    """Create valid weather data DataFrame."""
    return pd.DataFrame({
        "ghi": np.linspace(0, 800, 24),
        "dni": np.linspace(0, 600, 24),
        "dhi": np.linspace(0, 300, 24),
        "temp_air": np.linspace(5, 15, 24),
        "wind_speed": np.full(24, 3.0),
    }, index=sample_index)


class TestValidateIrradianceData:
    """Test irradiance data validation."""

    def test_valid_data_passes(self, valid_weather_data):
        """Valid data passes validation without error."""
        validate_irradiance_data(valid_weather_data)  # Should not raise

    def test_missing_ghi_raises(self, sample_index):
        """Missing GHI column raises error."""
        data = pd.DataFrame({
            "dni": [100, 200],
            "dhi": [50, 100],
        }, index=sample_index[:2])
        with pytest.raises(ValueError, match="ghi"):
            validate_irradiance_data(data)

    def test_missing_dni_raises(self, sample_index):
        """Missing DNI column raises error."""
        data = pd.DataFrame({
            "ghi": [100, 200],
            "dhi": [50, 100],
        }, index=sample_index[:2])
        with pytest.raises(ValueError, match="dni"):
            validate_irradiance_data(data)

    def test_missing_dhi_raises(self, sample_index):
        """Missing DHI column raises error."""
        data = pd.DataFrame({
            "ghi": [100, 200],
            "dni": [50, 100],
        }, index=sample_index[:2])
        with pytest.raises(ValueError, match="dhi"):
            validate_irradiance_data(data)

    def test_negative_ghi_raises(self, sample_index):
        """Negative GHI values raise error."""
        data = pd.DataFrame({
            "ghi": [-10, 100],
            "dni": [50, 100],
            "dhi": [50, 50],
        }, index=sample_index[:2])
        with pytest.raises(ValueError, match="negative"):
            validate_irradiance_data(data)

    def test_ghi_exceeds_sum_raises(self, sample_index):
        """GHI > DNI + DHI raises error (physically impossible)."""
        data = pd.DataFrame({
            "ghi": [200, 100],
            "dni": [50, 50],
            "dhi": [50, 50],  # GHI=200 > DNI+DHI=100
        }, index=sample_index[:2])
        with pytest.raises(ValueError, match="exceeds"):
            validate_irradiance_data(data)


class TestExtractTemperatureData:
    """Test temperature data extraction."""

    def test_extracts_temp_and_wind(self, valid_weather_data):
        """Extracts temp_air and wind_speed columns."""
        result = extract_temperature_data(valid_weather_data)
        assert "temp_air" in result.columns
        assert "wind_speed" in result.columns
        assert len(result) == len(valid_weather_data)

    def test_uses_default_wind_speed(self, sample_index):
        """Uses default wind speed when not present."""
        data = pd.DataFrame({
            "temp_air": np.linspace(5, 15, 24),
        }, index=sample_index)
        result = extract_temperature_data(data, default_wind_speed=2.5)
        assert (result["wind_speed"] == 2.5).all()

    def test_missing_temp_raises(self, sample_index):
        """Missing temp_air raises error."""
        data = pd.DataFrame({
            "wind_speed": [3.0, 3.0],
        }, index=sample_index[:2])
        with pytest.raises(ValueError, match="temp_air"):
            extract_temperature_data(data)

    def test_preserves_index(self, valid_weather_data):
        """Output index matches input index."""
        result = extract_temperature_data(valid_weather_data)
        pd.testing.assert_index_equal(result.index, valid_weather_data.index)
