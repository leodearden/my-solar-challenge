"""Fleet simulation commands."""

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

from solar_challenge.cli.utils import (
    console,
    create_progress,
    create_summary_table,
    handle_errors,
    print_info,
    print_success,
)
from solar_challenge.config import (
    create_bristol_phase1_scenario,
    load_fleet_config,
)
from solar_challenge.fleet import FleetResults, calculate_fleet_summary, simulate_fleet

app = typer.Typer(help="Fleet simulation commands")


def _export_fleet_results(results: FleetResults, output: Path) -> None:
    """Export fleet results to CSV."""
    df = results.to_aggregate_dataframe()
    df.to_csv(output)


@app.command()
@handle_errors
def run(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to fleet config YAML/JSON file",
            exists=True,
            dir_okay=False,
        ),
    ],
    start: Annotated[
        str,
        typer.Option(
            "--start",
            help="Start date (YYYY-MM-DD)",
        ),
    ] = "2024-01-01",
    end: Annotated[
        str,
        typer.Option(
            "--end",
            help="End date (YYYY-MM-DD)",
        ),
    ] = "2024-12-31",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output CSV file path for aggregate results",
        ),
    ] = None,
) -> None:
    """Run a fleet simulation from config file.

    The config file should define a list of homes with their PV, battery,
    and load configurations.
    """
    fleet_config = load_fleet_config(config)
    n_homes = len(fleet_config.homes)

    # Get location from first home for timezone
    loc = fleet_config.homes[0].location

    # Parse dates
    start_date = pd.Timestamp(start, tz=loc.timezone)
    end_date = pd.Timestamp(end, tz=loc.timezone)

    days = (end_date - start_date).days + 1
    print_info(f"Simulating fleet of {n_homes} homes for {days} days")

    with create_progress() as progress:
        task = progress.add_task(f"Simulating {n_homes} homes...", total=None)
        results = simulate_fleet(fleet_config, start_date, end_date)
        progress.update(task, completed=True)

    summary = calculate_fleet_summary(results)

    # Create and display summary table
    table = create_summary_table(summary, title=f"Fleet Results: {fleet_config.name or 'Fleet'}")

    # Add fleet-specific stats
    table.add_row("", "")  # Separator
    table.add_row(
        "Per-Home Generation (min/max)",
        f"{summary.per_home_generation_min_kwh:.1f} / {summary.per_home_generation_max_kwh:.1f} kWh",
    )
    table.add_row(
        "Per-Home Generation (mean)",
        f"{summary.per_home_generation_mean_kwh:.1f} kWh",
    )
    table.add_row(
        "Self-Consumption Ratio (min/max)",
        f"{summary.per_home_self_consumption_ratio_min:.1%} / {summary.per_home_self_consumption_ratio_max:.1%}",
    )

    console.print(table)

    if output is not None:
        _export_fleet_results(results, output)
        print_success(f"Aggregate results saved to {output}")


@app.command(name="bristol-phase1")
@handle_errors
def bristol_phase1(
    start: Annotated[
        str,
        typer.Option(
            "--start",
            help="Start date (YYYY-MM-DD)",
        ),
    ] = "2024-01-01",
    end: Annotated[
        str,
        typer.Option(
            "--end",
            help="End date (YYYY-MM-DD)",
        ),
    ] = "2024-12-31",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output CSV file path for aggregate results",
        ),
    ] = None,
    days: Annotated[
        Optional[int],
        typer.Option(
            "--days", "-d",
            help="Number of days to simulate (overrides --end)",
        ),
    ] = None,
) -> None:
    """Run the built-in Bristol Phase 1 scenario.

    This scenario simulates 100 homes in Bristol with a realistic mix of:
    - PV sizes: 3-6 kW (20% 3kW, 40% 4kW, 30% 5kW, 10% 6kW)
    - Batteries: 40% none, 40% 5kWh, 20% 10kWh
    - Consumption: Normal distribution around 3400 kWh/year
    """
    scenario = create_bristol_phase1_scenario()
    loc = scenario.get_location()

    # Parse dates
    start_date = pd.Timestamp(start, tz=loc.timezone)
    if days is not None:
        end_date = start_date + pd.Timedelta(days=days - 1)
    else:
        end_date = pd.Timestamp(end, tz=loc.timezone)

    sim_days = (end_date - start_date).days + 1
    n_homes = len(scenario.homes)

    print_info(f"Bristol Phase 1: {n_homes} homes, {sim_days} days")
    print_info("Distribution: 20% 3kW, 40% 4kW, 30% 5kW, 10% 6kW PV")
    print_info("Batteries: 40% none, 40% 5kWh, 20% 10kWh")

    from solar_challenge.fleet import FleetConfig

    fleet_config = FleetConfig(homes=scenario.homes, name="Bristol Phase 1")

    with create_progress() as progress:
        task = progress.add_task(f"Simulating {n_homes} homes...", total=None)
        results = simulate_fleet(fleet_config, start_date, end_date)
        progress.update(task, completed=True)

    summary = calculate_fleet_summary(results)

    table = create_summary_table(summary, title="Bristol Phase 1 Results")
    table.add_row("", "")
    table.add_row(
        "Per-Home Generation (min/max)",
        f"{summary.per_home_generation_min_kwh:.1f} / {summary.per_home_generation_max_kwh:.1f} kWh",
    )
    table.add_row(
        "Per-Home Generation (mean)",
        f"{summary.per_home_generation_mean_kwh:.1f} kWh",
    )

    console.print(table)

    if output is not None:
        _export_fleet_results(results, output)
        print_success(f"Aggregate results saved to {output}")
