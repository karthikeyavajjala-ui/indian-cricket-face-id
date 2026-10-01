# data/ — runtime state (git-ignored)

| path | contents |
|---|---|
| `app.db` | SQLite database: players, datasets, predictions, model_versions, training_runs |
| `uploads/` | uploaded/captured frames (`<uuid>.jpg`) |
| `uploads/faces/` | the detected face crop stored with every prediction (dashboard thumbnails) |
| `active_model.json` | which model file the API is currently serving |
| `cache/` | scratch space for the pipeline |

Nothing here is needed to run the app — it is created automatically on first start.

**Personal data.** Uploaded images and face crops are biometric data. Set
`IC_STORE_UPLOADS=0` to keep nothing, `DELETE /api/predictions` (or the dashboard button) to clear
the audit trail, and delete this folder when you are finished. `make clean` resets it.
