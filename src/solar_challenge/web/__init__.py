"""Solar Challenge Web Dashboard.

A web-based dashboard for visualising solar PV and battery simulation results,
providing interactive charts and metrics for individual home and fleet analysis.
"""

__all__ = [
    "create_app",
]


def create_app():
    """Create and configure the web dashboard application.

    Returns:
        The configured web application instance.
    """
    from solar_challenge.web.app import create_app as _create_app
    return _create_app()
