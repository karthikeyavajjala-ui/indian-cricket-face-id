# Architecture

## 1. Request flow

```
┌──────────────┐        ┌──────────────────────────────────────────────────────┐
│  frontend/   │  REST  │  backend/ (FastAPI)                                  │
│  index.html  │ ─────▶ │  routers ── services ── SQLAlchemy ── data/app.db     │
│  app.js      │ ◀───── │     │            │                                │
└──────────────┘  JSON  │     │            ▼                                │
                        │     │      ml/predictor.FaceIdentifier             │
                        │     │            │                                │
                        │     │   ml/face ── ml/features ── ml/model         │
                        └─────┴─────────────────────────────────────────────┘
                                          │
                        models/model.pkl  ── trained offline by  ml/train.py
```

The ML core (`ml/`) has **no web dependency**: it can be scripted, tested and retrained without
starting the API. The backend only orchestrates it and stores results.

## 2. Module map

| module | responsibility | extension point |
|---|---|---|
| `ml/config.py` | paths, `PipelineConfig` (persisted inside the model bundle) | add fields to tune the pipeline |
| `ml/face.py` | `HaarFaceDetector`, `YuNetFaceDetector`, NMS, landmark alignment, crops | implement the `FaceDetector` protocol for MediaPipe/RetinaFace/SCRFD |
| `ml/features.py` | extractor registry: `hog_lbp`, `sface`, `pixel` | subclass `FeatureExtractor`, decorate with `@register` |
| `ml/pipeline.py` | scan → validate → detect → align → augment → feature matrix | swap augmentation strategy in `augment_face` |
| `ml/model.py` | classifier factory, train/evaluate, `ModelBundle` save/load | add estimators to `CLASSIFIERS` |
| `ml/predictor.py` | inference + decision policy (threshold, margin, similarity) | tighten/loosen the policy in `_decide` |
| `backend/services/*` | model lifecycle, player sync, prediction audit, training jobs, dashboard aggregates | — |
| `backend/routers/*` | HTTP surface (see `docs/API.md`) | — |

## 3. Why these components

**Detection.** Haar (bundled with OpenCV) is always available and needs no download; it is good
enough for frontal, well-lit photographs. YuNet (OpenCV Zoo, ~230 KB ONNX) handles pose/scale
variation much better and also provides 5 landmarks, enabling a proper similarity-transform
alignment. `IC_DETECTOR=auto` picks YuNet when present, Haar otherwise — the rest of the pipeline
is detector-agnostic.

**Alignment.** Training and inference must feed the extractor *the same geometry*. Eye positions
plus mouth centre are fitted to the canonical SFace reference layout with a least-squares
similarity transform (rotation + uniform scale + translation, no shear). This removes roll/scale
variance and makes the classifier reproducible. If landmarks are missing, a padded bounding-box
crop with an eyes-in-the-upper-third heuristic is used instead.

**Features.**
* `hog_lbp` — HOG on a CLAHE-normalised 128×128 patch + uniform-LBP spatial histograms (6×6 grid,
  59 bins) + a 32×32 intensity map = 4912-D. Classic, fast (~1 ms/face on CPU), no downloads.
* `sface` — OpenCV-Zoo SFace (MobileFaceNet, 128-D L2-normalised embedding, Apache-2.0). This is
  true face-embedding recognition and is the better choice when you have 20+ images per player.
  The extractor records its own dimensionality; nothing else in the codebase assumes 4912.
* `pixel` — 24×24 normalised intensities, kept as a floor for comparison.

Because the trained bundle stores *which* extractor produced its features, you can retrain with a
stronger encoder (ArcFace, FaceNet, InsightFace, a fine-tuned CNN) and the API immediately serves
the new one — the 1-pair decision (`ml/face.py` + `ml/features.py`) is the only place to touch.

**Classifier.** `StandardScaler` is mandatory here: HOG/LBP dimensions differ by orders of
magnitude. SVM-RBF with `probability=True` and balanced class weights is the default because it
behaves best on small, high-dimensional, many-class problems. Optional PCA is available
(`IC_PCA_COMPONENTS`) for speed or when samples ≪ dimensions.

**Decision policy.** Three independent guards:

| guard | setting | behaviour |
|---|---|---|
| probability floor | `confidence_threshold` (0.55) | below → `status="unknown"`, `reason="low_confidence"` |
| top-2 margin | `min_margin` (0.12) | closer than this → `status="unknown"`, `reason="ambiguous"` |
| prototype similarity | `min_similarity` (0.28) | available to embedding mode as an additional gate |

Rejections are returned **with** the top-3 list so a human can see what the model was considering.

## 4. Database schema

```
players(id, slug, folder, display_name, role, batting_style, bowling_style, jersey_number,
        born, bio, career_highlights, fictional, supported, dataset_images, created_at, updated_at)
datasets(id, player_id → players.id, path, source[local|demo], images_count, valid_images,
        invalid_images, licence_note, last_scanned_at, created_at)
predictions(id, player_id → players.id NULL, predicted_folder, predicted_name, status, reason,
        confidence, margin, similarity, top_predictions(JSON), faces_detected, source,
        image_reference, face_image, model_version, latency_ms, created_at)
model_versions(id, version, model_path, dataset_path, dataset_fingerprint, classifier,
        feature_extractor, detector, n_classes, training_samples, validation_samples,
        accuracy, precision_macro, recall_macro, f1_macro, cross_val_accuracy, trained_at,
        metrics_json, is_active, created_at)
training_runs(id, job_id, status, dataset_path, options_json, log, progress, stage, error,
        model_version_id → model_versions.id, started_at, finished_at)
```

SQLite in WAL mode with a 30 s busy timeout (the training worker writes progress while the API
serves requests). PostgreSQL works by changing `IC_DATABASE_URL`; the schema is portable
(SQLAlchemy Core types only, no SQLite-specific SQL).

## 5. Concurrency model

* **API** — FastAPI/uvicorn, one process. The `FaceIdentifier` is a process-wide singleton guarded
  by path comparison; `POST /api/model/reload` swaps it atomically.
* **Training** — one job at a time, run in a daemon thread with a single long-lived DB session
  (`JobWriter`) that commits progress at most every 1.5 s. Job state lives in `training_runs`, so
  the UI can reload and resume polling.
* **Model switching** — the active model is resolved as: pointer file (`data/active_model.json`)
  → `models/model.pkl` → `models/demo_model.pkl`. A pointer to the demo model is ignored as soon
  as a real model exists, so `python -m ml.train` always takes over.

## 6. Performance (single CPU core, measured in this project)

| stage | hog_lbp engine | sface engine |
|---|---|---|
| detection (Haar, 900×1100 photo) | 20–60 ms | — |
| detection (YuNet) | — | 15–40 ms |
| alignment | < 2 ms | < 2 ms |
| feature extraction | ~1.2 ms | ~6 ms |
| classification (SVM, ≤ 20 classes) | < 5 ms | < 5 ms |
| **end-to-end** | **~70–150 ms** | **~60–120 ms** |

Training: feature extraction dominates; ~30 images × (1 + `augment`) variants takes a few seconds,
and the SVM fit is milliseconds-to-seconds depending on class count.

## 7. Deployment notes

* Single machine / demo: `python run.py` (SQLite, filesystem uploads).
* Server: run behind a reverse proxy with TLS (cameras need a secure context), set
  `IC_ALLOW_UNSIGNED_TRAINING=0` + a strong `IC_TRAIN_API_KEY`, and consider
  `IC_STORE_UPLOADS=0` so personal images are not retained.
* Vertical scaling (multiple uvicorn workers) works, but each worker loads its own copy of the
  model and **only the worker that accepted `POST /api/train` knows about that job**; use one
  worker (the default) or move training to a queue/Celery with a shared DB before scaling out.
* Before real production use, the pieces missing are: consent + audit workflow, encryption at rest,
  retention policy, rate limiting, authentication for read endpoints, and a documented DPIA.
  See `docs/RESPONSIBLE_AI.md`.
