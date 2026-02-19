"""Flask application factory for the Solar Challenge web dashboard."""

import os
from pathlib import Path

from flask import Flask


def create_app(test_config: dict | None = None) -> Flask:
    """Create and configure the Flask web dashboard application.

    Uses the application factory pattern to allow multiple instances
    and easy testing with different configurations.

    Args:
        test_config: Optional configuration dict to override defaults.
            Useful for testing.

    Returns:
        Flask: The configured Flask application instance.
    """
    # Resolve template and static folder paths relative to this file
    web_dir = Path(__file__).parent
    template_folder = str(web_dir / "templates")
    static_folder = str(web_dir / "static")

    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
    )

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    if test_config is not None:
        # Override with test-specific configuration when provided
        app.config.from_mapping(test_config)

    # Ensure template and static directories exist
    for folder in (template_folder, static_folder):
        os.makedirs(folder, exist_ok=True)

    # Register blueprints (deferred to allow routes.py to exist independently)
    _register_blueprints(app)

    return app


def _register_blueprints(app: Flask) -> None:
    """Register application blueprints.

    Args:
        app: The Flask application instance.
    """
    try:
        from solar_challenge.web.routes import bp
        app.register_blueprint(bp)
    except ImportError:
        # Routes not yet implemented - skip registration during setup phase
        pass


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True)
