# INDIAN CRICKET FACE ID · developer shortcuts
.PHONY: help install run dev dataset-audit dataset-init models train train-demo predict smoke test clean demo zip

help:
	@echo "make install        install requirements (+ pytest)"
	@echo "make run            start the app on http://localhost:8000"
	@echo "make dev            start with auto-reload"
	@echo "make models         fetch optional YuNet + SFace ONNX models"
	@echo "make dataset-init   create dataset/<Player_Name>/ folders"
	@echo "make dataset-audit  validate images (size, blur, faces, duplicates)"
	@echo "make demo           rebuild + retrain the fictional demo dataset"
	@echo "make train          train on dataset/ -> models/model.pkl"
	@echo "make predict IMG=photo.jpg"
	@echo "make smoke          end-to-end self check"
	@echo "make test           pytest"
	@echo "make clean          remove caches, DB, uploads and trained models"

install:
	python -m pip install -r requirements.txt pytest

run:
	python run.py

dev:
	python run.py --reload

models:
	python scripts/fetch_models.py --yes

dataset-init:
	python scripts/prepare_dataset.py --init

dataset-audit:
	python scripts/prepare_dataset.py --audit --min-images 8

demo:
	python scripts/make_demo_dataset.py --train

train:
	python -m ml.train --dataset dataset

predict:
	python -m ml.predict --image $(IMG)

smoke:
	python scripts/smoke_test.py

test:
	python -m pytest -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache data/*.db data/*.db-wal data/*.db-shm data/uploads data/active_model.json models/model.pkl models/metrics.json
	@echo "cleaned (dataset/, dataset_demo/ and models/demo_model.pkl kept)"
