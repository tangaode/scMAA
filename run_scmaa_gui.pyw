"""Desktop launcher for scMAA."""

from __future__ import annotations

from pathlib import Path

from scrt_agent.gui_app import run_desktop_app


if __name__ == "__main__":
    run_desktop_app(Path(__file__).resolve().parent)

