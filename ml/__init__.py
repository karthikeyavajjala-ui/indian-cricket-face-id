"""INDIAN CRICKET FACE ID - machine-learning package.

Public entry points
-------------------
``ml.train``      training CLI           (python -m ml.train --dataset dataset)
``ml.predict``    prediction CLI         (python -m ml.predict --image photo.jpg)
``ml.pipeline``   dataset scan + feature-matrix builder
``ml.face``       face detection + alignment (Haar or YuNet)
``ml.features``   feature extractors (hog_lbp / sface / pixel, pluggable)
``ml.model``      classifier training, evaluation, persistence
``ml.predictor``  inference engine + decision policy
"""
from . import config, face, features, model, pipeline, predictor   # noqa: F401

__all__ = ["config", "face", "features", "model", "pipeline", "predictor"]
__version__ = "1.0.0"
