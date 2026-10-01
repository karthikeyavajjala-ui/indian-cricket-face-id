# Troubleshooting

Start here:

```bash
python scripts/smoke_test.py --api http://localhost:8000     # subsystem check
python -m pytest -q                                          # ML core check
curl -s localhost:8000/api/health | python -m json.tool       # engine + dataset snapshot
```

## Installation / startup

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: fastapi` (or cv2, sklearn) | activate the venv and `pip install -r requirements.txt` |
| `error: Microsoft Visual C++ 14.0 required` on Windows | use Python 3.10–3.12 (64-bit) wheels, or `pip install --only-binary :all: -r requirements.txt` |
| `Address already in use` | another process owns the port: `python run.py --port 8010`, or stop the old server |
| `ModuleNotFoundError: mediapipe` | optional and unused by default — the app uses Haar/YuNet; see `docs/ARCHITECTURE.md` |
| YAML/`uvicorn[standard]` build failure on very new Python | `pip install uvicorn` (non-standard) — pure-Python asyncio loop is enough |

## Model

| symptom | cause / fix |
|---|---|
| Header pill says **no model**, result is `model_missing` | nothing trained yet. With `IC_AUTO_BOOTSTRAP=1` the demo trains on first launch (watch the API log); otherwise `python -m ml.train --dataset dataset` |
| `No trained model at models/model.pkl` | train, or point `IC_MODEL_PATH` at your model file |
| Trained via CLI but the API still serves the demo | the API resolves pointer → `models/model.pkl` → `models/demo_model.pkl`. `POST /api/model/reload`, or `POST /api/model/activate?mode=trained` |
| Model file loads with a scikit-learn warning after upgrading sklearn | retrain (`python -m ml.train --dataset dataset`); pickle compatibility is version-bound |
| `models/demo_model.pkl` fails to unpickle | retrain it: `python scripts/make_demo_dataset.py --train` |

## Training

| symptom | cause / fix |
|---|---|
| `Dataset at ... needs at least 2 player folders with images` | create ≥ 2 folders with images: `python scripts/prepare_dataset.py --init` |
| `No usable faces were extracted` | the detector found no face in any image — run `python scripts/prepare_dataset.py --audit`, add closer/front-facing photos, or enable YuNet (`python scripts/fetch_models.py --yes`) |
| Many images skipped with “no face detected” | Haar is weak on profiles/low light. Enable YuNet; the audit lists exactly which files failed |
| Accuracy is `null` | not enough samples for a held-out split (needs ≥ 2 per class); add images |
| Accuracy is suspiciously 1.0 | few images + augmentation leakage across splits. Add fresh photos and read `docs/DATASET.md` §4 |
| `database is locked` | SQLite write contention; the app sets a 30 s busy timeout — retry, and avoid running the CLI trainer and an API training job simultaneously |
| Training is slow | feature extraction dominates. Lower `--augment`, use `--pca 200`, use `sface` (128-D) instead of `hog_lbp` (4912-D), or smaller images (`--optimize`) |
| `401` from `POST /api/train` | send `X-API-Key` matching `IC_TRAIN_API_KEY`, or set `IC_ALLOW_UNSIGNED_TRAINING=1` for local use only |
| `409 Another training job is running` | one job at a time; check `GET /api/train/jobs` |

## Prediction quality

| symptom | cause / fix |
|---|---|
| Always `unknown` | threshold too high for your data (`IC_CONFIDENCE_THRESHOLD`, or the Model page field), too few images per player, or the person simply is not in the dataset |
| Always the same wrong player | that player dominates the training set (class imbalance) — add images for the others, keep `class_weight="balanced"`, check the confusion matrix |
| Works on dataset photos, fails on new ones | classic small-data overfitting: add varied, real photos; prefer `sface` embeddings; keep augmentation moderate |
| Two similar-looking players get confused | expected with few images. Add profile-diverse photos for both, and check per-class recall |
| Confidence jumps around between shots | normal for tiny datasets; consider averaging several frames, or raise `min_margin` to force rejection when unsure |
| “No face detected” on a good photo | face smaller than ~40 px, heavy rotation, sunglasses/mask covering the eyes, or blown-out highlights. Crop closer, or enable YuNet |
| Group photo: wrong person reported | the **largest** face is treated as the subject; crop to the person you care about |

## Camera / browser

| symptom | cause / fix |
|---|---|
| Camera button does nothing / permission error | browsers expose `getUserMedia` only on `https://` or `localhost`. Serve over TLS or use upload mode |
| Preview is mirrored but the analysis is not (or vice versa) | the capture mirrors the preview by design; disable by editing `Camera.capture(..., mirror=false)` in `frontend/assets/js/camera.js` |
| Camera works but the frame is black | camera still warming up — wait for the preview, then capture |
| UI shows “Backend unreachable” | API not running, or a proxy blocking `/api/*`; check `curl localhost:8000/api/health` |
| Charts empty | no predictions yet, or no model trained (metrics page). Run an identification, then reload |

## Files / data

| symptom | cause / fix |
|---|---|
| `Unsupported file type '.heic'` | HEIC is not supported; convert to JPG/PNG first |
| `413` on upload | file exceeds `IC_MAX_UPLOAD_MB` (default 12 MB) |
| Profile images 404 | the folder was moved/renamed after the last scan; restart the API or re-run the dataset sync (`GET /api/players`) |
| Disk filling up | uploads and face crops accumulate in `data/uploads/`. Set `IC_STORE_UPLOADS=0`, or `DELETE /api/predictions` + remove the folders |
| Want a totally clean slate | `make clean` (keeps `dataset/`, `dataset_demo/`, `models/demo_model.pkl`) |

Still stuck? Run with more detail:

```bash
IC_LOG_LEVEL=DEBUG python run.py
python -m ml.train --dataset dataset --scan-only        # dataset problems only
python -m ml.predict --image photo.jpg --json | python -m json.tool
```
