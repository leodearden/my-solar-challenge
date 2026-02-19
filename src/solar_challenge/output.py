"""Output and reporting functions for simulation results."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from solar_challenge.home import SimulationResults, SummaryStatistics, calculate_summary


def export_to_csv(
    results: SimulationResults,
    filepath: Union[str, Path],
    include_index: bool = True,
) -> Path:
    """Export simulation results to CSV file.

    Args:
        results: Simulation results to export
        filepath: Output file path
        include_index: Whether to include datetime index in output

    Returns:
        Path to the created file
    """
    filepath = Path(filepath)
    df = results.to_dataframe()
    df.to_csv(filepath, index=include_index)
    return filepath


def generate_summary_report(
    results: SimulationResults,
    home_name: Optional[str] = None,
    seg_tariff_pence_per_kwh: Optional[float] = None,
) -> str:
    """Generate a text summary report of simulation results.

    Args:
        results: Simulation results
        home_name: Optional name for the home
        seg_tariff_pence_per_kwh: Smart Export Guarantee tariff in pence per kWh.
            If provided, a SEG Revenue section is included in the report.

    Returns:
        Formatted markdown text report
    """
    summary = calculate_summary(results, seg_tariff_pence_per_kwh=seg_tariff_pence_per_kwh)

    title = f"# Simulation Report: {home_name}" if home_name else "# Simulation Report"

    report = f"""{title}

## Simulation Period
- Duration: {summary.simulation_days} days
- Start: {results.generation.index[0]}
- End: {results.generation.index[-1]}

## Energy Totals (kWh)
| Metric | Value |
|--------|-------|
| Generation | {summary.total_generation_kwh:.1f} |
| Demand | {summary.total_demand_kwh:.1f} |
| Self-Consumption | {summary.total_self_consumption_kwh:.1f} |
| Grid Import | {summary.total_grid_import_kwh:.1f} |
| Grid Export | {summary.total_grid_export_kwh:.1f} |

## Battery (kWh)
| Metric | Value |
|--------|-------|
| Total Charged | {summary.total_battery_charge_kwh:.1f} |
| Total Discharged | {summary.total_battery_discharge_kwh:.1f} |

## Peak Values (kW)
| Metric | Value |
|--------|-------|
| Peak Generation | {summary.peak_generation_kw:.2f} |
| Peak Demand | {summary.peak_demand_kw:.2f} |

## Efficiency Ratios
| Metric | Value |
|--------|-------|
| Self-Consumption Ratio | {summary.self_consumption_ratio:.1%} |
| Grid Dependency | {summary.grid_dependency_ratio:.1%} |
| Export Ratio | {summary.export_ratio:.1%} |

## Daily Averages (kWh/day)
| Metric | Value |
|--------|-------|
| Average Generation | {summary.total_generation_kwh / summary.simulation_days:.1f} |
| Average Demand | {summary.total_demand_kwh / summary.simulation_days:.1f} |
| Average Self-Consumption | {summary.total_self_consumption_kwh / summary.simulation_days:.1f} |
"""

    if summary.seg_revenue_gbp is not None:
        report += f"""
## SEG Revenue
| Metric | Value |
|--------|-------|
| Tariff Rate | {seg_tariff_pence_per_kwh:.1f} p/kWh |
| Total Export | {summary.total_grid_export_kwh:.1f} kWh |
| SEG Revenue | £{summary.seg_revenue_gbp:.2f} |
"""

    return report


def calculate_self_consumption_ratio(results: SimulationResults) -> float:
    """Calculate self-consumption ratio.

    Self-consumption ratio = self_consumed / total_generation

    Args:
        results: Simulation results

    Returns:
        Ratio between 0 and 1 (or 0 if no generation)
    """
    summary = calculate_summary(results)
    return summary.self_consumption_ratio


def calculate_grid_dependency_ratio(results: SimulationResults) -> float:
    """Calculate grid dependency ratio.

    Grid dependency = grid_import / total_consumption

    Lower values indicate more self-sufficiency.

    Args:
        results: Simulation results

    Returns:
        Ratio between 0 and 1 (or 0 if no consumption)
    """
    summary = calculate_summary(results)
    return summary.grid_dependency_ratio


def calculate_export_ratio(results: SimulationResults) -> float:
    """Calculate export ratio.

    Export ratio = grid_export / total_generation

    Higher values indicate more excess PV.

    Args:
        results: Simulation results

    Returns:
        Ratio between 0 and 1 (or 0 if no generation)
    """
    summary = calculate_summary(results)
    return summary.export_ratio


def aggregate_daily(results: SimulationResults) -> pd.DataFrame:
    """Aggregate 1-minute results to daily totals.

    Args:
        results: Simulation results with 1-minute resolution

    Returns:
        DataFrame with daily DatetimeIndex and energy totals in kWh
    """
    df = results.to_dataframe()

    # Convert power (kW) to energy (kWh) - sum of 1-minute kW values / 60
    # Resample to daily and sum, then divide by 60 to get kWh
    daily = df.resample("D").sum() / 60

    # Rename columns to indicate energy
    daily.columns = [col.replace("_kw", "_kwh") for col in daily.columns]

    # Also add daily peak values
    peaks = df.resample("D").max()
    daily["peak_generation_kw"] = peaks["generation_kw"]
    daily["peak_demand_kw"] = peaks["demand_kw"]

    return daily


def aggregate_monthly(results: SimulationResults) -> pd.DataFrame:
    """Aggregate results to monthly totals.

    Args:
        results: Simulation results

    Returns:
        DataFrame with monthly period index and energy totals in kWh
    """
    daily = aggregate_daily(results)

    # Resample to monthly - sum energy columns
    energy_cols = [col for col in daily.columns if "_kwh" in col]
    peak_cols = [col for col in daily.columns if "peak_" in col]

    monthly_energy = daily[energy_cols].resample("ME").sum()
    monthly_peaks = daily[peak_cols].resample("ME").max()

    return pd.concat([monthly_energy, monthly_peaks], axis=1)


def aggregate_annual(
    results: SimulationResults,
    seg_tariff_pence_per_kwh: Optional[float] = None,
) -> dict[str, float]:
    """Aggregate results to annual totals.

    Args:
        results: Simulation results
        seg_tariff_pence_per_kwh: Smart Export Guarantee tariff in pence per kWh.
            If provided, seg_revenue_gbp is included in the returned dictionary.

    Returns:
        Dictionary with annual energy totals in kWh, and optionally SEG revenue in GBP
    """
    summary = calculate_summary(results, seg_tariff_pence_per_kwh=seg_tariff_pence_per_kwh)

    annual: dict[str, float] = {
        "generation_kwh": summary.total_generation_kwh,
        "demand_kwh": summary.total_demand_kwh,
        "self_consumption_kwh": summary.total_self_consumption_kwh,
        "grid_import_kwh": summary.total_grid_import_kwh,
        "grid_export_kwh": summary.total_grid_export_kwh,
        "battery_charge_kwh": summary.total_battery_charge_kwh,
        "battery_discharge_kwh": summary.total_battery_discharge_kwh,
        "peak_generation_kw": summary.peak_generation_kw,
        "peak_demand_kw": summary.peak_demand_kw,
        "self_consumption_ratio": summary.self_consumption_ratio,
        "grid_dependency_ratio": summary.grid_dependency_ratio,
        "export_ratio": summary.export_ratio,
        "simulation_days": float(summary.simulation_days),
    }

    if summary.seg_revenue_gbp is not None:
        annual["seg_revenue_gbp"] = summary.seg_revenue_gbp

    return annual
