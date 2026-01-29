"""Fleet simulation commands."""

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

from solar_challenge.cli.utils import (
    console,
    create_fleet_progress,
    create_progress,
    create_summary_table,
    handle_errors,
    print_info,
    print_success,
)
from solar_challenge.config import load_fleet_config
from solar_challenge.fleet import (
    FleetConfig,
    FleetResults,
    calculate_fleet_summary,
    simulate_fleet_iter,
)
from solar_challenge.home import SimulationResults

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
    workers: Annotated[
        Optional[int],
        typer.Option(
            "--workers", "-w",
            help="Number of parallel workers",
        ),
    ] = None,
    sequential: Annotated[
        bool,
        typer.Option(
            "--sequential",
            help="Disable parallelization",
        ),
    ] = False,
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

    results_list: list[SimulationResults | None] = [None] * n_homes

    with create_fleet_progress() as progress:
        task = progress.add_task(f"Simulating {n_homes} homes...", total=n_homes)
        for idx, result in simulate_fleet_iter(
            fleet_config, start_date, end_date,
            parallel=not sequential, max_workers=workers
        ):
            results_list[idx] = result
            progress.update(task, advance=1)

    # Build FleetResults from results_list
    final_results: list[SimulationResults] = []
    for r in results_list:
        assert r is not None
        final_results.append(r)

    results = FleetResults(
        per_home_results=final_results,
        home_configs=fleet_config.homes,
    )

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
