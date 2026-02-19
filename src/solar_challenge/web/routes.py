"""Flask Blueprint routes for the Solar Challenge web dashboard."""

import io
import uuid
from typing import Any

import pandas as pd
from flask import (
    Blueprint,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from solar_challenge.battery import BatteryConfig
from solar_challenge.home import HomeConfig, calculate_summary, simulate_home
from solar_challenge.load import LoadConfig
from solar_challenge.location import Location
from solar_challenge.output import generate_summary_report
from solar_challenge.pv import PVConfig

bp = Blueprint("main", __name__)

# In-memory result cache keyed by session result_key.
# Suitable for single-process deployment (dev/demo use).
_result_cache: dict[str, Any] = {}


@bp.route("/", methods=["GET"])
def index() -> str:
    """Render the simulation configuration form."""
    return render_template("index.html")


@bp.route("/simulate", methods=["POST"])
def simulate():
    """Run simulation with posted form data and store results in cache."""
    try:
        # Parse form data with sensible defaults
        pv_kw = float(request.form.get("pv_kw", 4.0))
        battery_kwh = float(request.form.get("battery_kwh", 0.0))
        consumption_kwh_raw = request.form.get("consumption_kwh", "")
        occupants = int(request.form.get("occupants", 3))
        start = request.form.get("start", "2024-01-01")
        end = request.form.get("end", "2024-12-31")
        location_preset = request.form.get("location", "bristol")

        # Resolve location
        if location_preset == "bristol":
            loc = Location.bristol()
        else:
            try:
                lat, lon = map(float, location_preset.split(","))
                loc = Location(latitude=lat, longitude=lon)
            except ValueError:
                loc = Location.bristol()

        # Build PV config
        pv_config = PVConfig(capacity_kw=pv_kw)

        # Build battery config (omit if zero capacity)
        battery_config = None
        if battery_kwh > 0:
            battery_config = BatteryConfig(capacity_kwh=battery_kwh)

        # Build load config (annual consumption optional; derive from occupants if blank)
        annual_consumption: float | None = None
        if consumption_kwh_raw.strip():
            annual_consumption = float(consumption_kwh_raw)

        load_config = LoadConfig(
            annual_consumption_kwh=annual_consumption,
            household_occupants=occupants,
        )

        # Assemble home config
        home_config = HomeConfig(
            pv_config=pv_config,
            load_config=load_config,
            battery_config=battery_config,
            location=loc,
            name="Web Simulation",
        )

        # Parse simulation dates
        start_date = pd.Timestamp(start, tz=loc.timezone)
        end_date = pd.Timestamp(end, tz=loc.timezone)

        # Run simulation
        results = simulate_home(home_config, start_date, end_date)
        summary = calculate_summary(results)

        # Cache results with a unique key
        result_key = str(uuid.uuid4())
        _result_cache[result_key] = {
            "results": results,
            "summary": summary,
            "home_name": home_config.name,
        }

        # Store key and serialisable summary stats in session
        session["result_key"] = result_key
        session["summary"] = {
            "total_generation_kwh": round(summary.total_generation_kwh, 2),
            "total_demand_kwh": round(summary.total_demand_kwh, 2),
            "total_self_consumption_kwh": round(summary.total_self_consumption_kwh, 2),
            "total_grid_import_kwh": round(summary.total_grid_import_kwh, 2),
            "total_grid_export_kwh": round(summary.total_grid_export_kwh, 2),
            "total_battery_charge_kwh": round(summary.total_battery_charge_kwh, 2),
            "total_battery_discharge_kwh": round(summary.total_battery_discharge_kwh, 2),
            "peak_generation_kw": round(summary.peak_generation_kw, 2),
            "peak_demand_kw": round(summary.peak_demand_kw, 2),
            "self_consumption_ratio": round(summary.self_consumption_ratio, 4),
            "grid_dependency_ratio": round(summary.grid_dependency_ratio, 4),
            "export_ratio": round(summary.export_ratio, 4),
            "simulation_days": summary.simulation_days,
        }

        return redirect(url_for("main.results"))

    except Exception as exc:  # noqa: BLE001
        flash(f"Simulation failed: {exc}", "error")
        return redirect(url_for("main.index"))


@bp.route("/results", methods=["GET"])
def results():
    """Display simulation results page."""
    if "result_key" not in session:
        flash("No simulation results found. Please run a simulation first.", "info")
        return redirect(url_for("main.index"))

    summary = session.get("summary", {})
    return render_template("results.html", summary=summary)


@bp.route("/download/csv", methods=["GET"])
def download_csv() -> Response:
    """Stream simulation results as a CSV file download."""
    result_key = session.get("result_key")
    if not result_key or result_key not in _result_cache:
        flash("No simulation results available. Please run a simulation first.", "error")
        return redirect(url_for("main.index"))  # type: ignore[return-value]

    cached = _result_cache[result_key]
    sim_results = cached["results"]

    df: pd.DataFrame = sim_results.to_dataframe()
    buffer = io.StringIO()
    df.to_csv(buffer, index=True)
    csv_data = buffer.getvalue()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=simulation_results.csv"},
    )


@bp.route("/download/report", methods=["GET"])
def download_report() -> Response:
    """Stream simulation markdown report as a plain-text file download."""
    result_key = session.get("result_key")
    if not result_key or result_key not in _result_cache:
        flash("No simulation results available. Please run a simulation first.", "error")
        return redirect(url_for("main.index"))  # type: ignore[return-value]

    cached = _result_cache[result_key]
    sim_results = cached["results"]
    home_name = cached.get("home_name", "Home")

    report_text = generate_summary_report(sim_results, home_name)

    return Response(
        report_text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=simulation_report.txt"},
    )
