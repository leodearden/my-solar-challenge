"""Shared fixtures for Playwright e2e tests.

Provides a live Flask server running in a background thread and
configures Playwright's base_url so tests can use relative paths.
"""

import socket
import threading

import pytest
from werkzeug.serving import make_server

from solar_challenge.web.app import create_app


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server(tmp_path_factory):
    """Start the Flask app on a random port in a daemon thread.

    Yields the base URL (e.g. ``http://127.0.0.1:54321``).
    """
    tmp = tmp_path_factory.mktemp("e2e")
    db_path = tmp / "test.db"

    app = create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "e2e-test-secret",
            "DATABASE": str(db_path),
            "DATA_DIR": str(tmp),
        }
    )

    port = _find_free_port()
    server = make_server("127.0.0.1", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()


@pytest.fixture(scope="session")
def base_url(live_server):
    """Override pytest-playwright's base_url with our live server."""
    return live_server
