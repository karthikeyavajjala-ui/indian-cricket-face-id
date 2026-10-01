#!/usr/bin/env python3
"""
End-to-end smoke test - run this after training (or after starting the API).

    python scripts/smoke_test.py                       # local ML pipeline
    python scripts/smoke_test.py --api http://localhost:8000
    python scripts/smoke_test.py --model models/demo_model.pkl --image dataset_demo/Demo_Persona_A/05_photo.jpg

Exit code 0 = everything that was checked passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml import config                      # noqa: E402
from ml.predictor import FaceIdentifier    # noqa: E402

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{c(PASS, GREEN) if ok else c(FAIL, RED)}] {name}"
          + (f"  {c(detail, DIM)}" if detail else ""))
    return ok


def api_get(base: str, path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base + path, timeout=20) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except Exception as exc:                        # pragma: no cover
        return 0, {"error": str(exc)}


def test_api(base: str, sample: Path | None) -> int:
    print(c(f"\nAPI checks · {base}", BOLD))
    status, payload = api_get(base, "/api/health")
    check("GET /api/health", status == 200, f"status={status}")
    status, payload = api_get(base, "/api/model/status")
    ready = status == 200 and payload.get("ready")
    check("GET /api/model/status", status == 200, f"ready={ready}")
    status, players = api_get(base, "/api/players")
    check("GET /api/players", status == 200 and players.get("count", 0) >= 0,
          f"count={players.get('count')}")
    status, metrics = api_get(base, "/api/model/metrics")
    check("GET /api/model/metrics", status == 200, f"ready={metrics.get('ready')}")
    status, dash = api_get(base, "/api/dashboard/summary")
    check("GET /api/dashboard/summary", status == 200,
          f"predictions={dash.get('predictions_total')}")

    if sample and sample.is_file():
        import base64
        body = json.dumps({
            "image_base64": base64.b64encode(sample.read_bytes()).decode(),
            "source": "api", "save": True,
        }).encode()
        request = urllib.request.Request(base + "/api/predict/image", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode())
            check("POST /api/predict/image", response.status == 200,
                  f"status={result.get('status')} player={(result.get('player') or {}).get('name')} "
                  f"confidence={result.get('confidence')}")
        except Exception as exc:
            check("POST /api/predict/image", False, str(exc))
    else:
        print(c("  (skipped prediction check - no --image given)", DIM))
    return 0


def test_local(model: Path, images: list[Path]) -> int:
    print(c(f"\nLocal pipeline checks · model={model}", BOLD))
    if not check("model file exists", model.is_file(), str(model)):
        print(c("  Train first: python -m ml.train --dataset dataset", YELLOW))
        return 2
    identifier = FaceIdentifier(model_path=model)
    if not check("model loads", identifier.ready, identifier.error or ""):
        return 2
    engine = identifier.engine_info()
    check("engine reports classes", engine["n_classes"] >= 2,
          f"{engine['n_classes']} classes · {engine['feature_extractor']} + {engine['detector']}")
    if not images:
        images = sorted(Path(config.DEMO_DATASET_DIR).glob("Demo_Persona_*/*.jpg"))[:3]
    if not images:
        print(c("  (no images available for a prediction check)", DIM))
        return 0
    for image in images:
        result = identifier.identify_image_bytes(image.read_bytes())
        ok = result.get("status") in ("identified", "unknown")
        check(f"predict {image.name}", ok,
              f"{result.get('status')} · {(result.get('player') or {}).get('name')} · "
              f"conf={result.get('confidence')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="INDIAN CRICKET FACE ID smoke test")
    parser.add_argument("--api", default=None, help="base URL of a running API")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--image", type=Path, action="append", default=[])
    args = parser.parse_args()

    print(c("\nINDIAN CRICKET FACE ID · smoke test", BOLD))
    if args.model is None:
        for candidate in (config.DEFAULT_MODEL_PATH, config.DEMO_MODEL_PATH):
            if candidate.is_file():
                args.model = candidate
                break
    if not args.api:
        test_local(args.model or config.DEFAULT_MODEL_PATH, args.image)
    test_api(args.api or "http://127.0.0.1:8000",
             args.image[0] if args.image else
             (sorted(Path(config.DEMO_DATASET_DIR).glob("Demo_Persona_*/*.jpg")) or [None])[0])

    failed = [r for r in results if r[0] == FAIL]
    print()
    if failed:
        print(c(f"{len(failed)} check(s) failed.", RED))
        return 1
    print(c(f"All {len(results)} checks passed.", GREEN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
