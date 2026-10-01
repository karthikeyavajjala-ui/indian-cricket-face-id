# API reference

Base URL: `http://localhost:8000` · interactive docs: `/docs` · OpenAPI schema: `/openapi.json`

All responses are JSON. Error responses use `{"detail": "…"}` with a standard status code
(400 invalid input, 401 missing/invalid API key, 404 unknown resource, 409 conflicting job,
413 payload too large, 415 unsupported media type).

| method | path | auth | purpose |
|---|---|---|---|
| GET | `/api/health` | — | liveness, engines, dataset & DB snapshot |
| GET | `/api/config` | — | public frontend configuration, thresholds, disclaimers |
| GET | `/api/players` | — | catalog + dataset counts + model support |
| GET | `/api/players/{id}` | — | one profile |
| GET | `/api/players/{id}/avatar.svg` | — | deterministic monogram avatar |
| POST | `/api/predict` | — | multipart upload → identification |
| POST | `/api/predict/image` | — | base64 image → identification |
| GET | `/api/model/status` | — | active model, engine, thresholds, dataset state |
| GET | `/api/model/metrics` | — | honest evaluation report of the active model |
| GET | `/api/model/versions` | — | every recorded model version |
| GET | `/api/model/available` | — | model files present on disk |
| GET | `/api/model/dataset` | — | dataset inventory (real + demo) |
| POST | `/api/model/reload` | — | hot-reload the model from disk |
| POST | `/api/model/activate` | **key** | switch demo ↔ trained model |
| POST | `/api/train` | **key** | start a training job |
| GET | `/api/train/status/{job_id}` | — | job progress + log tail |
| GET | `/api/train/jobs` | — | recent jobs |
| GET | `/api/train/capabilities` | — | classifiers / extractors / detector availability |
| GET | `/api/dashboard/summary` | — | cards, charts, recent predictions |
| GET | `/api/predictions` | — | prediction audit trail |
| GET | `/api/predictions/{id}/detail` | — | stored top-k for one attempt |
| DELETE | `/api/predictions` | — | clear the audit trail (admin convenience) |
| GET | `/uploads/...`, `/media/{source}/...` | — | stored uploads / read-only dataset images |

**Auth:** training and model switching take `X-API-Key: <IC_TRAIN_API_KEY>`. With the default local
configuration (`IC_ALLOW_UNSIGNED_TRAINING=1` and the placeholder key) they are open for
convenience; set `IC_ALLOW_UNSIGNED_TRAINING=0` and a real key before exposing the API.

---

## POST /api/predict — upload a file

```bash
curl -F "file=@kohli.jpg" "http://localhost:8000/api/predict?source=upload&save=true"
```

Query parameters: `source` (`upload` | `camera` | `api`), `save` (persist the attempt).
Accepted files: JPG, JPEG, PNG, WEBP, BMP (≤ `IC_MAX_UPLOAD_MB`).

```json
{
  "status": "identified",
  "reason": null,
  "message": null,
  "player": {
    "id": "virat-kohli", "name": "Virat Kohli", "confidence": 0.947,
    "role": "Right-handed top-order batter · former captain (all formats)",
    "batting_style": "Right-handed", "bowling_style": "Right-arm medium pace (rarely used)",
    "jersey_number": 18
  },
  "confidence": 0.947, "threshold": 0.55, "margin": 0.916, "similarity": 0.812,
  "top_predictions": [
    {"class_name": "Virat_Kohli", "player_id": "virat-kohli", "display_name": "Virat Kohli",
     "confidence": 0.947, "similarity": 0.812, "role": "Right-handed top-order batter …"},
    {"class_name": "Rohit_Sharma", "display_name": "Rohit Sharma", "confidence": 0.031},
    {"class_name": "KL_Rahul", "display_name": "K. L. Rahul", "confidence": 0.012}
  ],
  "detected_face": {"image": "data:image/jpeg;base64,…", "box": [212, 96, 288, 288],
                    "detection_score": 1.0, "image_size": [900, 1200]},
  "faces_detected": 1,
  "other_faces": [],
  "engine": {"ready": true, "detector": "yunet", "feature_extractor": "sface",
             "classifier": "svc", "n_classes": 12, "confidence_threshold": 0.55,
             "model_version": "v2-1a2b3c4d"},
  "dataset_status": {"configured": true, "num_classes": 12, "total_images": 214, "images": {…}},
  "timings_ms": {"detect": 41.2, "encode": 68.9, "classify": 4.1, "total": 121.4},
  "prediction_id": 87,
  "disclaimer": "This system is a demonstration of facial identification technology …"
}
```

### `status` values

| status | meaning | HTTP |
|---|---|---|
| `identified` | a name was returned; confidence ≥ threshold **and** margin ≥ `min_margin` | 200 |
| `unknown` | refusal; `reason` is `low_confidence` or `ambiguous` — top-3 is still returned | 200 |
| `no_face` | detection found nothing usable | 200 |
| `invalid_image` | the bytes could not be decoded | 200 |
| `model_missing` | no trained model → `message` explains how to train one | 200 |

The API never turns a rejection into a guess, and never fabricates a match when no model exists.

## POST /api/predict/image — base64 (browser camera)

```bash
curl -X POST http://localhost:8000/api/predict/image \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"data:image/jpeg;base64,/9j/4AAQ…","source":"camera","top_k":3,"save":true}'
```

Same response shape; `top_k` truncates the ranking, `save:false` skips persistence.

## GET /api/players

```json
{
  "supported": true, "count": 15,
  "players": [{
    "id": "virat-kohli", "folder": "Virat_Kohli", "name": "Virat Kohli",
    "role": "Right-handed top-order batter …", "batting_style": "Right-handed",
    "bowling_style": "Right-arm medium pace (rarely used)", "jersey_number": 18,
    "born": "5 November 1988 · Delhi, India", "bio": "…", "career_highlights": ["…"],
    "fictional": false, "supported": true, "dataset_images": 24, "dataset_source": "local",
    "avatar_url": "/media/local/Virat_Kohli/01.jpg", "sample_images": ["…"],
    "model": {"trained": true, "dataset_images": 24, "augmented_variants": 144,
              "training_vectors": 168, "validation_support": 6,
              "precision": 0.98, "recall": 0.97, "f1": 0.975, "note": "Recognised by the active model"},
    "source_note": "Factual fields are drawn from public career records …"
  }],
  "dataset": {"path": "...", "configured": true, "num_classes": 12, "total_images": 214, "images": {…}},
  "disclaimer": "Only the players marked 'supported' exist in the trained model's dataset …"
}
```

`include_catalog=false` returns only supported players / players with dataset images.

## GET /api/model/metrics

```json
{
  "ready": true, "model_version": "v2-1a2b3c4d", "trained_at": "2026-10-01T07:54:39+00:00",
  "training": {"training_samples": 112, "validation_samples": 38, "feature_dim": 4912,
               "n_classes": 3, "classifier": "svc", "feature_extractor": "hog_lbp",
               "detector": "haar", "training_seconds": 0.21, "dataset": {…},
               "class_samples": {"Virat_Kohli": {"images": 24, "augmented": 144, "training_vectors": 168}}},
  "evaluation": {"accuracy": 0.947, "precision_macro": 0.95, "recall_macro": 0.94,
                 "f1_macro": 0.945, "f1_weighted": 0.947, "top3_accuracy": 1.0,
                 "roc_auc_macro": 0.99,
                 "confusion_matrix": [[…]], "confusion_matrix_labels": ["Virat_Kohli", "…"],
                 "per_class": [{"class": "Virat_Kohli", "precision": 0.98, "recall": 0.97,
                                "f1": 0.975, "support": 6}], "split": {…}},
  "cross_validation": {"available": true, "folds": 5, "accuracy_mean": 0.94,
                       "accuracy_std": 0.03, "per_class_recall": {…}},
  "confusion": {"labels": ["…"], "matrix": [[…]]},
  "notes": ["Held-out metrics come from the same image sources as training …"],
  "transparency": ["Metrics come from a held-out split of your own dataset …"]
}
```

When no split is possible, `evaluation.accuracy` is `null` and `notes` explains why. Nothing is
estimated or back-filled.

## POST /api/train

```bash
curl -X POST http://localhost:8000/api/train \
  -H "X-API-Key: $IC_TRAIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"dataset":"dataset","classifier":"svc","extractor":"auto","detector":"auto",
       "augment":6,"pca_components":0,"test_size":0.25,"confidence_threshold":0.55,
       "min_margin":0.12,"seed":2026,"activate":true}'
```

`202 Accepted`

```json
{"job_id":"a6e1810a2a70","status":"queued","dataset":"/…/dataset",
 "message":"Training started. Poll /api/train/status/{job_id} for progress. …"}
```

Poll `GET /api/train/status/{job_id}`:

```json
{"job_id":"a6e1810a2a70","status":"running","stage":"extracting features",
 "progress":38.5,"error":null,"log":"dataset: …\noptions: {…}\n[ 12.5%] Processing 03.jpg\n…",
 "started_at":"2026-10-01T08:12:01","finished_at":null,
 "result":{"model_version":"v2-1a2b3c4d","accuracy":0.947,"f1_macro":0.945,
           "n_classes":12,"training_samples":1210,"validation_samples":402}}
```

`status ∈ queued | running | succeeded | failed`. Only one job runs at a time (409 otherwise).
`dataset` accepts `dataset`, `demo`/`dataset_demo`, or a path inside the project directory.

## GET /api/dashboard/summary

Stat cards, chart series and the recent-prediction feed:

```json
{"total_players":15,"supported_players":12,"total_images":214,"dataset_configured":true,
 "model_accuracy":0.947,"model_accuracy_note":"Held-out split of your own images; …",
 "predictions_total":428,"predictions_today":31,"identified_today":27,"unknown_today":4,
 "avg_confidence":0.86,"avg_latency_ms":96.4,
 "recent":[{"id":428,"predicted_name":"Virat Kohli","status":"identified","confidence":0.947,
            "face_image":"/uploads/faces/9f2c….jpg","latency_ms":88.1,
            "created_at":"2026-10-01T08:31:44"}],
 "confidence_distribution":[{"label":"<40%","value":3}, …],
 "prediction_distribution":[{"label":"Virat Kohli","value":120}, …],
 "status_distribution":[…], "daily_trend":[{"label":"01 Oct","total":31,"identified":27,"unknown":4}],
 "performance":{"accuracy":0.947,"per_class":[…],"confusion":{…}},
 "model":{"ready":true,"mode":"trained","version":"v2-1a2b3c4d","feature_extractor":"hog_lbp", …},
 "dataset":{…},"disclaimer":"Statistics describe this deployment only. …"}
```

## Python client example

```python
import base64, requests

BASE = "http://localhost:8000"
r = requests.post(f"{BASE}/api/predict", files={"file": open("photo.jpg", "rb")}, timeout=60)
data = r.json()
print(data["status"], (data.get("player") or {}).get("name"), data.get("confidence"))

# rejection is data, not an error:
if data["status"] == "unknown":
    print("refused:", data["reason"], data["top_predictions"])
    # add more training images for this person, or lower IC_CONFIDENCE_THRESHOLD knowingly
```

## Storage side effects

With `IC_STORE_UPLOADS=1` (default) each saved call writes:

* `data/uploads/<uuid>.jpg` — the uploaded/captured frame (`image_reference`)
* `data/uploads/faces/<uuid>.jpg` — the detected face crop (`face_image`)
* one `predictions` row (including rejections)

Set `IC_STORE_UPLOADS=0` to keep nothing on disk, or send `save=false` per request.
