# models/ — trained artefacts

| file | produced by | notes |
|---|---|---|
| `model.pkl` | `python -m ml.train --dataset dataset` (or `POST /api/train`) | your model: classifier + scaler/PCA + class list + pipeline config + metrics |
| `metrics.json` | the same run | human-readable metrics sidecar (same content as the Model page) |
| `demo_model.pkl` | `python scripts/make_demo_dataset.py --train` | bundled demo model trained on AI-generated **fictional** personas |
| `face_detector/*.onnx` | `python scripts/fetch_models.py` | optional YuNet detector (OpenCV Zoo, Apache-2.0) |
| `face_models/*.onnx` | `python scripts/fetch_models.py` | optional SFace 128-D face embedding (OpenCV Zoo, Apache-2.0) |

`model.pkl` is a `joblib` pickle of `ml.model.ModelBundle`. It contains everything needed at
inference time: the estimator, the feature scaler (and optional PCA), the class order, the class
centroids used for the prototype-similarity score, the pipeline configuration that produced the
features, and the full training/evaluation report.

Pickle files are code — only load ones you produced yourself.

Delete `model.pkl` / `metrics.json` to start over, or `make clean` for a full reset.
