"""Weather data retrieval and handling."""

from typing import Any

import pandas as pd
from pvlib.iotools import get_pvgis_tmy

from solar_challenge.location import Location


def get_tmy_data(location: Location) -> pd.DataFrame:
    """Retrieve Typical Meteorological Year (TMY) data from PVGIS.

    Uses pvlib.iotools.get_pvgis_tmy() to fetch TMY data for the given location.

    Args:
        location: Location object with latitude, longitude, and altitude

    Returns:
        DataFrame with columns including:
        - temp_air: Ambient temperature (°C)
        - ghi: Global horizontal irradiance (W/m²)
        - dni: Direct normal irradiance (W/m²)
        - dhi: Diffuse horizontal irradiance (W/m²)
        - wind_speed: Wind speed at 10m (m/s)
        Index is DatetimeIndex in UTC.

    Raises:
        RuntimeError: If PVGIS API request fails
    """
    try:
        # PVGIS returns a tuple: (data, months_selected, inputs, metadata)
        data: tuple[pd.DataFrame, Any, Any, Any] = get_pvgis_tmy(
            latitude=location.latitude,
            longitude=location.longitude,
            outputformat="json",
            usehorizon=True,
            startyear=2005,
            endyear=2020,
            map_variables=True,  # Map to standard pvlib column names
        )
        tmy_data = data[0]

        # Ensure we have the expected columns
        required_columns = {"temp_air", "ghi", "dni", "dhi"}
        if not required_columns.issubset(tmy_data.columns):
            missing = required_columns - set(tmy_data.columns)
            raise RuntimeError(f"TMY data missing required columns: {missing}")

        return tmy_data

    except Exception as e:
        raise RuntimeError(f"Failed to retrieve TMY data from PVGIS: {e}") from e


def validate_irradiance_data(data: pd.DataFrame) -> None:
    """Validate irradiance data quality.

    Args:
        data: DataFrame with ghi, dni, dhi columns

    Raises:
        ValueError: If validation fails with specific issue identified
    """
    required = ["ghi", "dni", "dhi"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in required:
        if (data[col] < 0).any():
            neg_count = (data[col] < 0).sum()
            raise ValueError(
                f"Column '{col}' contains {neg_count} negative values"
            )

    # GHI should approximately equal DNI * cos(zenith) + DHI
    # For simplicity, check GHI <= DNI + DHI (conservative upper bound)
    if (data["ghi"] > data["dni"] + data["dhi"] + 1).any():  # 1 W/m² tolerance
        violations = (data["ghi"] > data["dni"] + data["dhi"] + 1).sum()
        raise ValueError(
            f"GHI exceeds DNI + DHI in {violations} rows (physical impossibility)"
        )


def extract_temperature_data(
    weather_data: pd.DataFrame,
    default_wind_speed: float = 1.0
) -> pd.DataFrame:
    """Extract temperature and wind data for cell temperature modelling.

    Args:
        weather_data: DataFrame containing weather data
        default_wind_speed: Default wind speed if not available (m/s)

    Returns:
        DataFrame with columns:
        - temp_air: Ambient temperature (°C)
        - wind_speed: Wind speed (m/s)
        Aligned with input DataFrame index.
    """
    result = pd.DataFrame(index=weather_data.index)

    if "temp_air" in weather_data.columns:
        result["temp_air"] = weather_data["temp_air"]
    else:
        raise ValueError("Weather data must contain 'temp_air' column")

    if "wind_speed" in weather_data.columns:
        result["wind_speed"] = weather_data["wind_speed"]
    else:
        # Use default wind speed if not available
        result["wind_speed"] = default_wind_speed

    return result
