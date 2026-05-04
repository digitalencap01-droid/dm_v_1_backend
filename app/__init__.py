from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Make `app.*` imports work both from the backend directory and the repo root.
__path__ = extend_path(__path__, __name__)

_backend_app_dir = (
    Path(__file__).resolve().parent.parent
    / "digital-marketing-ai"
    / "backend"
    / "app"
)

if _backend_app_dir.is_dir():
    __path__.append(str(_backend_app_dir))
