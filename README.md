# 🏏 INDIAN CRICKET FACE ID

**AI-powered Indian cricket player identification** — a complete, runnable computer-vision
application: face detection → alignment → feature extraction → trained classifier → player
profile, with a FastAPI backend, a SQLite/PostgreSQL database, and a premium sports-analytics
dark UI.

> **Responsible use (read this first).**
> This system is a demonstration of facial identification technology and should not be used for
> surveillance, law enforcement, access control, or high-impact decisions without appropriate
> safeguards. It recognises **only the people present in its training dataset** — it cannot
> identify every Indian cricketer, and it never guesses: below the configured confidence
> threshold it answers **“Unknown / Low Confidence”**.

---

## 1. What is inside

```
indian-cricket-face-id/
├── frontend/                 # premium dark UI (vanilla JS · no CDN, no build step)
│   ├── index.html            #   landing · identify · players · dashboard · model · dataset · responsible AI
│   └── assets/
│       ├── css/app.css       #   glassmorphism, stadium light beams, particles, responsive layout
│       ├── js/app.js         #   hash router + view renderers
│       ├── js/api.js         #   typed-ish fetch wrapper for the REST API
│       ├── js/charts.js      #   dependency-free SVG charts (bars, trend, confusion heatmap, rings)
│       ├── js/camera.js      #   getUserMedia capture with permission handling
│       ├── js/particles.js   #   animated stadium particle field
│       └── img/favicon.svg
├── backend/                  # FastAPI application
│   ├── main.py               #   app factory, static mounts, demo bootstrap, lifespan
│   ├── config.py             #   env-driven settings
│   ├── database.py           #   SQLAlchemy engine/session (SQLite default)
│   ├── models_db.py          #   players · datasets · predictions · model_versions · training_runs
│   ├── schemas.py            #   Pydantic request/response models
│   ├── catalog.py            #   player facts (roles, careers) + fictional demo personas
│   ├── routers/              #   health · players · predict · model · train · dashboard
│   └── services/             #   model lifecycle · players · predictions · training jobs · dashboard
├── ml/                       # the machine-learning pipeline (framework-free core)
│   ├── config.py             #   paths + PipelineConfig (persisted inside model.pkl)
│   ├── face.py               #   Haar & YuNet detectors, landmark alignment, NMS
│   ├── features.py           #   pluggable extractors: hog_lbp · sface (deep) · pixel
│   ├── pipeline.py           #   scan → validate → detect → align → augment → feature matrix
│   ├── model.py              #   classifier factory, train, metrics, bundle save/load
│   ├── predictor.py          #   inference engine + responsible-AI decision policy
│   ├── train.py              #   CLI:  python -m ml.train
│   └── predict.py            #   CLI:  python -m ml.predict --image photo.jpg
├── scripts/
│   ├── prepare_dataset.py    # init · audit · optimize (resize/re-encode) your image folders
│   ├── fetch_models.py       # optional: YuNet detector + SFace embedding ONNX (OpenCV Zoo)
│   ├── make_demo_dataset.py  # builds the fictional AI-generated demo dataset (+ trains it)
│   ├── smoke_test.py         # end-to-end self check (pipeline + live API)
│   └── export_report.py      # CSV/JSON export of predictions, metrics and dataset inventory
├── dataset/                  # ← YOUR images go here (one folder per player)
├── dataset_demo/             # AI-generated fictional personas (safe demo data)
├── models/                   # model.pkl · demo_model.pkl · metrics.json
├── data/                     # SQLite DB, uploads, detected face crops
├── docs/                     # ARCHITECTURE · DATASET · API · RESPONSIBLE_AI · TROUBLESHOOTING
├── tests/                    # pytest suite for the ML core
├── run.py                    # one-command launcher
├── requirements.txt
└── .env.example
```

---

## 2. Quick start (2 minutes)

```bash
# 1 · environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2 · run it
python run.py                        # → http://localhost:8000  (docs at /docs)
```

On first launch the app finds no trained model and **automatically builds + trains the bundled
demo** (three clearly-labelled, AI-generated fictional personas, 30 images) so every screen —
upload, camera, results, dashboard, metrics — is fully functional immediately. Nothing is faked:
the demo model is trained by the real pipeline on real files in `dataset_demo/`.

Prefer a model of your own right away? Skip to §3, then:

```bash
python -m ml.train --dataset dataset      # writes models/model.pkl  (takes over automatically)
```

### Optional stronger engines

```bash
python scripts/fetch_models.py --yes
python -m ml.train --dataset dataset --extractor auto --detector auto
```

This downloads two Apache-2.0 OpenCV-Zoo ONNX models (no player images — just network weights):

| engine | picks up | effect |
|---|---|---|
| **YuNet** (`cv2.FaceDetectorYN`) | better detection on angles, lighting, group photos | replaces the Haar cascade |
| **SFace** (`cv2.FaceRecognizerSF`, 128-D MobileFaceNet embedding) | true face-embedding recognition | replaces HOG+LBP descriptors |

Without them everything still works — the app auto-selects Haar + HOG/LBP.

---

## 3. Adding your own players (the real dataset)

The pipeline **never downloads player photographs for you** — you supply images you are legally
allowed to use. Folder name = class label:

```bash
python scripts/prepare_dataset.py --init          # creates dataset/<Player_Name>/ folders
python scripts/prepare_dataset.py --roster        # shows the folder names for the 12-player catalog
```

```
dataset/
├── Virat_Kohli/        01.jpg 02.jpg …
├── Rohit_Sharma/       01.jpg …
├── Jasprit_Bumrah/     …
└── …                   (any number of players, ≥ 2)
```

Then:

```bash
python scripts/prepare_dataset.py --audit --min-images 8   # validate: size, blur, faces, duplicates
python -m ml.train --dataset dataset                       # train + evaluate + save
python scripts/smoke_test.py                               # verify end to end
```

Guidance: **10–30 clear, front-facing photos per player** beat 3 photos by a wide margin. Vary
lighting, background, expression, head angle and distance. Keep folders small (≥ 8 images) or the
per-player metrics will be meaningless — the trainer warns you when they are.

---

## 4. How the pipeline works

```
dataset/ ──▶ image validation ──▶ face detection ──▶ alignment ──▶ augmentation
                                                                        │
                     model.pkl ◀── metrics ◀── training ◀── X / y ◀─────┘
                         │                                    ▲
                         ▼                              feature extraction
   upload / camera ──▶ detection ──▶ alignment ──▶ same extractor ──▶ classifier
                         │
                         ▼
        confidence + top-3 ──▶ threshold & margin policy ──▶ player profile  or  "Unknown"
```

* **Detection** — Haar cascade (bundled, zero downloads) or YuNet CNN when the ONNX file exists.
* **Alignment** — landmark similarity transform (both eyes + mouth) to a canonical 112×112 pose;
  training and inference share the exact same geometry, which is what makes the classifier repeatable.
* **Features** — `hog_lbp` (HOG + uniform-LBP histograms + intensity map, 4912-D, no downloads),
  `sface` (128-D deep embedding, optional), `pixel` (baseline). Add your own by subclassing
  `ml.features.FeatureExtractor` and registering it — the trained model records which extractor it
  used, so retraining with a stronger encoder needs no other code changes.
* **Classifier** — SVM (default, best on small high-dimensional sets), logistic regression, random
  forest or k-NN, always on standardised features, with optional PCA.
* **Decision policy** — refuses to name anyone when the top probability is below
  `IC_CONFIDENCE_THRESHOLD` (default 0.55) or when the top-2 gap is below `IC_MIN_MARGIN` (0.12);
  embedding mode can additionally require a minimum prototype similarity.

---

## 5. Running the API

```bash
python -m uvicorn backend.main:app --reload --port 8000     # or: python run.py --reload
```

| method | endpoint | purpose |
|---|---|---|
| GET | `/api/health` | liveness, engines, dataset + DB snapshot |
| GET | `/api/config` | public frontend configuration and disclaimers |
| GET | `/api/players` | catalog with dataset counts + model support |
| GET | `/api/players/{id}` | single profile (facts + model recognition info) |
| POST | `/api/predict` | multipart image upload → identification |
| POST | `/api/predict/image` | base64 frame (browser camera) → identification |
| GET | `/api/model/status` | active model, engine, thresholds, dataset status |
| GET | `/api/model/metrics` | accuracy / precision / recall / F1 / confusion matrix / CV |
| GET | `/api/model/versions` | every trained model recorded in the DB |
| POST | `/api/model/reload` \| `/api/model/activate` | hot-reload, switch demo ↔ trained |
| POST | `/api/train` | **protected** training job (`X-API-Key`) |
| GET | `/api/train/status/{job_id}` | live progress + log tail |
| GET | `/api/dashboard/summary` | stat cards, charts, recent predictions |
| GET | `/api/predictions` | audit trail (including rejections) |
| GET | `/docs` | interactive OpenAPI documentation |

Example:

```bash
curl -F "file=@photo.jpg" http://localhost:8000/api/predict | jq
```

```json
{
  "status": "identified",
  "player": { "id": "virat-kohli", "name": "Virat Kohli", "role": "Right-handed top-order batter" },
  "confidence": 0.947,
  "threshold": 0.55,
  "margin": 0.916,
  "top_predictions": [
    { "display_name": "Virat Kohli", "confidence": 0.947 },
    { "display_name": "Rohit Sharma", "confidence": 0.031 },
    { "display_name": "K. L. Rahul", "confidence": 0.012 }
  ],
  "detected_face": { "image": "data:image/jpeg;base64,…", "box": [212, 96, 288, 288] },
  "faces_detected": 1,
  "timings_ms": { "detect": 41.2, "encode": 68.9, "classify": 4.1, "total": 121.4 }
}
```

### Protecting the training endpoint

```bash
# .env
IC_TRAIN_API_KEY=choose-a-long-random-string
IC_ALLOW_UNSIGNED_TRAINING=0        # key now mandatory
```

```bash
curl -X POST http://localhost:8000/api/train \
     -H "X-API-Key: choose-a-long-random-string" \
     -H "Content-Type: application/json" \
     -d '{"dataset":"dataset","classifier":"svc","extractor":"auto","augment":6,"activate":true}'
```

Only one training job runs at a time; progress and the log tail come from
`GET /api/train/status/{job_id}` and are shown live on the Model page.

---

## 6. Database

SQLite by default (`data/app.db`), PostgreSQL by changing one env var
(`IC_DATABASE_URL=postgresql+psycopg://user:pass@host/db`). Tables:

| table | contents |
|---|---|
| `players` | catalog profiles: role, batting/bowling style, jersey, bio, support flags, dataset counts |
| `datasets` | per-folder inventory: path, source (local/demo), valid/invalid counts, licence note, last scan |
| `predictions` | **every** attempt: player, status, reason, confidence, margin, similarity, top-3 JSON, faces detected, source, image + face-crop paths, model version, latency, timestamp |
| `model_versions` | each trained model: version, algorithm, extractor, detector, sample counts, accuracy/precision/recall/F1/CV, metrics JSON, active flag |
| `training_runs` | job log for `POST /api/train` (status, stage, progress, log, error, produced model version) |

Rejections are stored as well as matches — a system that hides its failures cannot be audited.

---

## 7. Evaluation: what the numbers mean

The Model page and `GET /api/model/metrics` report exactly what was measured:

* held-out split accuracy, macro precision/recall/F1, weighted F1, top-3 accuracy, ROC-AUC;
* a full confusion matrix and per-player precision/recall/F1/support;
* stratified k-fold cross-validation (mean ± std, per-player recall);
* training-set composition (images, augmented variants, training vectors per player).

**Caveat, stated in the UI as well:** with a handful of photographs per player, augmented variants
of the *same* photo can land in both the training and the validation split, so the reported accuracy
is optimistic compared with unseen photographs. Nothing is invented — if a split isn't possible,
accuracy is reported as `null` and the UI says so.

---

## 8. Demo mode

`dataset_demo/` ships three **fictional, AI-generated personas** (Aadhya Rao, Arjun Nair, Vikram
Iyer) — no real person, no likeness rights involved — plus one hold-out stranger
(`dataset_demo/_holdout/`) used to demonstrate that the model answers **Unknown** for people it was
never trained on. Rebuild or retrain any time:

```bash
python scripts/make_demo_dataset.py --train
```

---

## 9. Testing

```bash
pip install pytest
pytest -q                                   # ML core: features, training, decision policy, persistence
python scripts/smoke_test.py                # pipeline + live API (start the server first)
python scripts/smoke_test.py --api http://localhost:8000
```

---

## 10. Configuration reference

Every knob lives in `.env` (see `.env.example`). The most useful ones:

| variable | default | meaning |
|---|---|---|
| `IC_CONFIDENCE_THRESHOLD` | `0.55` | below this → “Unknown / Low Confidence” |
| `IC_MIN_MARGIN` | `0.12` | required gap between #1 and #2 |
| `IC_CLASSIFIER` | `svc` | `svc` · `logistic` · `rf` · `knn` |
| `IC_EXTRACTOR` | `auto` | `auto` · `hog_lbp` · `sface` · `pixel` |
| `IC_DETECTOR` | `auto` | `auto` · `yunet` · `haar` |
| `IC_AUGMENT_PER_IMAGE` | `6` | augmented variants per image |
| `IC_DATABASE_URL` | `sqlite:///./data/app.db` | any SQLAlchemy URL |
| `IC_DEMO_ENABLED` / `IC_AUTO_BOOTSTRAP` | `1` / `1` | demo personas + first-run demo training |
| `IC_STORE_UPLOADS` | `1` | persist uploads & face crops (set `0` to keep nothing) |
| `IC_TRAIN_API_KEY` / `IC_ALLOW_UNSIGNED_TRAINING` | — / `1` | protection for `POST /api/train` |

---

## 11. Troubleshooting

| symptom | fix |
|---|---|
| “No trained model” | `python -m ml.train --dataset dataset` (or let the demo bootstrap finish) |
| “No face detected” on good photos | run `python scripts/fetch_models.py --yes` for YuNet, or crop closer |
| Training accuracy looks too good | normal with few images per player — read §7, add more photos |
| Camera button does nothing | browsers allow `getUserMedia` only on `https://` or `localhost`; use upload mode |
| `Unsupported file type` | JPG, JPEG, PNG, WEBP or BMP only |
| `401` from `/api/train` | send `X-API-Key` matching `IC_TRAIN_API_KEY` |
| Slow first prediction | the model and detectors warm up on the first call (watch `timings_ms`) |

More detail: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/API.md`](docs/API.md),
[`docs/DATASET.md`](docs/DATASET.md), [`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md).

---

## 12. Legal & privacy notes

* **Images.** The project ships no photographs of real people. The bundled demo faces were
  generated by an AI image model and depict nobody. Images you add to `dataset/` are your
  responsibility — licence them properly and respect personality/publicity rights.
* **Faces are sensitive personal data.** Uploaded images and detected face crops are stored under
  `data/uploads/`. Set `IC_STORE_UPLOADS=0` to keep nothing, or delete the folder, the
  `predictions` table (`DELETE /api/predictions`) and the database when you are finished.
  If you process personal data, comply with the law that applies to you (in India, the DPDP Act).
* **Model artefacts.** OpenCV-Zoo ONNX weights fetched by `scripts/fetch_models.py` are
  Apache-2.0; OpenCV itself is Apache-2.0; scikit-learn is BSD-3. This project's code is MIT
  (see `LICENSE`).

MIT licensed · built as a demonstration, not a production identity-verification product.
