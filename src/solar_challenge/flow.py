"""Energy flow calculations for PV and battery systems."""

import pandas as pd


def calculate_self_consumption(
    generation: pd.Series,
    demand: pd.Series
) -> pd.Series:
    """Calculate instantaneous self-consumption.

    Self-consumption is the portion of PV generation that is consumed
    on-site at the moment of generation.

    Args:
        generation: PV generation time series in kW
        demand: Demand/consumption time series in kW

    Returns:
        Self-consumption time series in kW (min of generation and demand)

    Raises:
        ValueError: If series have different lengths or contain negative values
    """
    if len(generation) != len(demand):
        raise ValueError(
            f"Series must have same length: generation={len(generation)}, "
            f"demand={len(demand)}"
        )

    if (generation < 0).any():
        raise ValueError("Generation series contains negative values")
    if (demand < 0).any():
        raise ValueError("Demand series contains negative values")

    result = pd.concat([generation, demand], axis=1).min(axis=1)
    result.name = "self_consumption"
    return result


def calculate_excess_pv(
    generation: pd.Series,
    demand: pd.Series
) -> pd.Series:
    """Calculate excess PV generation available for export or battery charging.

    Excess = generation - demand when positive, else 0.

    Args:
        generation: PV generation time series in kW
        demand: Demand/consumption time series in kW

    Returns:
        Excess PV time series in kW (non-negative)

    Raises:
        ValueError: If series have different lengths or contain negative values
    """
    if len(generation) != len(demand):
        raise ValueError(
            f"Series must have same length: generation={len(generation)}, "
            f"demand={len(demand)}"
        )

    if (generation < 0).any():
        raise ValueError("Generation series contains negative values")
    if (demand < 0).any():
        raise ValueError("Demand series contains negative values")

    result = (generation - demand).clip(lower=0)
    result.name = "excess_pv"
    return result


def calculate_shortfall(
    generation: pd.Series,
    demand: pd.Series
) -> pd.Series:
    """Calculate demand shortfall requiring battery or grid import.

    Shortfall = demand - generation when positive, else 0.

    Args:
        generation: PV generation time series in kW
        demand: Demand/consumption time series in kW

    Returns:
        Shortfall time series in kW (non-negative)

    Raises:
        ValueError: If series have different lengths or contain negative values
    """
    if len(generation) != len(demand):
        raise ValueError(
            f"Series must have same length: generation={len(generation)}, "
            f"demand={len(demand)}"
        )

    if (generation < 0).any():
        raise ValueError("Generation series contains negative values")
    if (demand < 0).any():
        raise ValueError("Demand series contains negative values")

    result = (demand - generation).clip(lower=0)
    result.name = "shortfall"
    return result
