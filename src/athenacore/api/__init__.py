"""REST + SSE API. Requires the ``api`` extra: ``pip install 'athenacore[api]'``."""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):
    # Imported lazily so that `import athenacore` never requires FastAPI.
    if name == "create_app":
        from athenacore.api.server import create_app

        return create_app
    raise AttributeError(name)
