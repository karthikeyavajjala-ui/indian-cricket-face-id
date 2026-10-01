#!/usr/bin/env python3
"""
Convenience launcher.

    python run.py                 # http://localhost:8000
    python run.py --port 9000 --reload
    python run.py --host 0.0.0.0  # expose on the local network (read the docs first)
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the INDIAN CRICKET FACE ID app")
    parser.add_argument("--host", default=os.getenv("IC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("IC_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # Importing the app here ensures dirs/DB exist before uvicorn binds.
    from backend.config import settings
    settings.ensure_dirs()

    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        try:
            webbrowser.open(f"http://{args.host}:{args.port}/")
        except Exception:
            pass

    import uvicorn
    uvicorn.run("backend.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level=settings.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
