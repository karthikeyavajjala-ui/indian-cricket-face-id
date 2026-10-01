#!/usr/bin/env python3
"""
Export the deployment's own data for analysis or reporting.

    python scripts/export_report.py --out exports
    python scripts/export_report.py --out exports --csv-only

Writes (using pandas when available, plain csv otherwise):
    exports/predictions.csv      every identification attempt (matches AND rejections)
    exports/model_metrics.csv    per-player held-out metrics of the active model
    exports/dataset_inventory.csv  image counts per player folder
    exports/summary.json         dashboard aggregates + model metadata
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import session_scope            # noqa: E402
from backend.services import dashboard_service        # noqa: E402
from ml import config                                 # noqa: E402
from ml.model import load_metrics_json                # noqa: E402
from ml.predictor import FaceIdentifier               # noqa: E402

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:                                   # pragma: no cover
    HAVE_PANDAS = False


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"  {path.name}: no rows")
        return
    if HAVE_PANDAS:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  {path.name}: {len(rows)} row(s)")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python scripts/export_report.py")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "exports")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    print(f"\nExporting to {args.out} (pandas {'on' if HAVE_PANDAS else 'off'})")

    with session_scope() as session:
        summary = dashboard_service.summary(session)
        predictions = dashboard_service.recent_predictions(session, args.limit)

    _write([{
        "id": p["id"], "timestamp": p["created_at"], "status": p["status"],
        "reason": p["reason"], "player": p["predicted_name"], "confidence": p["confidence"],
        "latency_ms": p["latency_ms"], "source": p["source"],
        "model_version": p["model_version"], "image": p["image_reference"],
    } for p in predictions], args.out / "predictions.csv")

    identifier = FaceIdentifier(model_path=args.model or None)
    metrics = (identifier.bundle.metrics if identifier.ready else None) or load_metrics_json(
        args.model or config.DEFAULT_MODEL_PATH) or {}
    per_class = (metrics.get("evaluation", {}) or {}).get("per_class", [])
    _write(per_class, args.out / "model_metrics.csv")

    dataset = config.dataset_summary_safe(config.DATASET_DIR)
    demo = config.dataset_summary_safe(config.DEMO_DATASET_DIR)
    rows = [{"player_folder": name, "images": count, "source": "dataset"}
            for name, count in (dataset.get("images") or {}).items()]
    rows += [{"player_folder": name, "images": count, "source": "dataset_demo"}
             for name, count in (demo.get("images") or {}).items()]
    _write(rows, args.out / "dataset_inventory.csv")

    import datetime as _dt
    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "dashboard": {k: v for k, v in summary.items()
                      if k not in ("recent", "performance", "daily_trend")},
        "performance": summary.get("performance"),
        "model_metadata": (identifier.bundle.metadata if identifier.ready else None),
        "thresholds": {
            "confidence_threshold": (identifier.bundle.cfg.get("confidence_threshold")
                                     if identifier.ready else None),
            "min_margin": (identifier.bundle.cfg.get("min_margin") if identifier.ready else None),
        },
        "responsible_use": ("Demonstration only. Not for surveillance, law enforcement, access "
                            "control or high-impact decisions. Recognises only people present "
                            "in the training dataset."),
    }
    (args.out / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, default=str),
                                           encoding="utf-8")
    print("  summary.json written")
    print("\nRemember: personal data. Store these files securely and delete them when done.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
