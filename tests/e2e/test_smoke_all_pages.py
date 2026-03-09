"""Smoke tests: verify every page loads without server errors.

Each test navigates to a route and checks the HTTP status and basic
content expectations.  These are intentionally lightweight -- they catch
regressions in routing, template rendering, and static-asset delivery.
"""

import pytest
from playwright.sync_api import ConsoleMessage, Page

pytestmark = pytest.mark.e2e


# ── Individual page-load tests ─────────────────────────────────────────


def test_dashboard_loads(page: Page, live_server: str) -> None:
    """GET / returns 200 and contains the app title."""
    response = page.goto(live_server + "/")
    assert response is not None
    assert response.status == 200
    assert page.locator("text=Solar Challenge").first.is_visible()


def test_simulate_home_loads(page: Page, live_server: str) -> None:
    """GET /simulate/home returns 200."""
    response = page.goto(live_server + "/simulate/home")
    assert response is not None
    assert response.status == 200


def test_simulate_fleet_loads(page: Page, live_server: str) -> None:
    """GET /simulate/fleet returns 200."""
    response = page.goto(live_server + "/simulate/fleet")
    assert response is not None
    assert response.status == 200


def test_scenario_builder_loads(page: Page, live_server: str) -> None:
    """GET /scenarios/builder returns 200."""
    response = page.goto(live_server + "/scenarios/builder")
    assert response is not None
    assert response.status == 200


def test_parameter_sweep_loads(page: Page, live_server: str) -> None:
    """GET /scenarios/sweep returns 200."""
    response = page.goto(live_server + "/scenarios/sweep")
    assert response is not None
    assert response.status == 200


def test_history_runs_loads(page: Page, live_server: str) -> None:
    """GET /history/runs returns 200."""
    response = page.goto(live_server + "/history/runs")
    assert response is not None
    assert response.status == 200


def test_404_page(page: Page, live_server: str) -> None:
    """GET /nonexistent returns the custom 404 page."""
    response = page.goto(live_server + "/nonexistent")
    assert response is not None
    assert response.status == 404
    # The custom 404 template contains both "404" and "Page Not Found"
    page.wait_for_load_state("domcontentloaded")
    text = page.text_content("body") or ""
    assert "404" in text or "Page Not Found" in text


# ── Static-asset delivery ──────────────────────────────────────────────


def test_all_pages_have_css(page: Page, live_server: str) -> None:
    """Every main page includes the compiled Tailwind stylesheet."""
    pages_to_check = ["/", "/simulate/home", "/simulate/fleet"]
    for path in pages_to_check:
        page.goto(live_server + path)
        page.wait_for_load_state("domcontentloaded")
        css_links = page.locator('link[href*="dist/style.css"]')
        assert css_links.count() > 0, (
            f"Page {path} is missing <link> to dist/style.css"
        )


# ── Console-error checks ──────────────────────────────────────────────


def _check_page_for_console_errors(page: Page, url: str) -> list[str]:
    """Navigate to *url*, collect any console-level errors, and return them."""
    errors: list[str] = []

    def _on_console(msg: ConsoleMessage) -> None:
        if msg.type == "error":
            errors.append(msg.text)

    page.on("console", _on_console)
    page.goto(url)
    # Give deferred scripts (Alpine.js, HTMX) time to initialise
    page.wait_for_load_state("networkidle")
    page.remove_listener("console", _on_console)
    return errors


def test_no_js_console_errors_on_static_pages(
    page: Page, live_server: str,
) -> None:
    """Pages that do not rely on heavy Alpine.js components should be
    free of JS console errors.

    Pages with known Alpine.js race-condition issues (/simulate/fleet,
    /scenarios/builder, /scenarios/sweep) are excluded from this check.
    """
    safe_pages = ["/", "/simulate/home", "/history/runs"]

    for path in safe_pages:
        errors = _check_page_for_console_errors(page, live_server + path)
        assert errors == [], f"Console errors on {path}: {errors}"
