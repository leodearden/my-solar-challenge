"""Shared helpers for the Solar Challenge web application."""

from __future__ import annotations

from typing import Any

from flask import current_app

from solar_challenge.location import Location
from solar_challenge.web.storage import RunStorage


def get_storage() -> RunStorage:
    """Return a RunStorage instance from current Flask app config."""
    db_path: str = current_app.config["DATABASE"]
    data_dir: str = current_app.config["DATA_DIR"]
    return RunStorage(db_path=db_path, data_dir=data_dir)


LOCATION_PRESETS: dict[str, Location] = {
    "bristol": Location(latitude=51.45, longitude=-2.58, altitude=11.0, name="Bristol, UK"),
    "london": Location(latitude=51.51, longitude=-0.13, altitude=11.0, name="London, UK"),
    "edinburgh": Location(latitude=55.95, longitude=-3.19, altitude=47.0, name="Edinburgh, UK"),
    "manchester": Location(latitude=53.48, longitude=-2.24, altitude=38.0, name="Manchester, UK"),
}


def location_presets_as_dicts() -> dict[str, dict[str, Any]]:
    """Return location presets as plain dicts (for scenarios.py compatibility)."""
    return {
        key: {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "altitude": loc.altitude,
            "name": loc.name,
        }
        for key, loc in LOCATION_PRESETS.items()
    }
