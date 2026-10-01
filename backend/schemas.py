"""Pydantic request/response models (API documentation + validation)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetStatus(BaseModel):
    path: str
    configured: bool
    num_classes: int = 0
    total_images: int = 0
    images: dict[str, int] = Field(default_factory=dict)
    hint: Optional[str] = None


class PlayerModelInfo(BaseModel):
    trained: bool = False
    dataset_images: int = 0
    augmented_variants: int = 0
    training_vectors: int = 0
    validation_support: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    note: Optional[str] = None


class PlayerProfile(BaseModel):
    id: str
    folder: str
    name: str
    role: Optional[str] = None
    batting_style: Optional[str] = None
    bowling_style: Optional[str] = None
    jersey_number: Optional[int] = None
    born: Optional[str] = None
    bio: Optional[str] = None
    career_highlights: list[str] = Field(default_factory=list)
    fictional: bool = False
    supported: bool = False
    dataset_images: int = 0
    dataset_source: str = "none"          # local | demo | none
    avatar_url: str
    sample_images: list[str] = Field(default_factory=list)
    model: PlayerModelInfo = Field(default_factory=PlayerModelInfo)
    source_note: Optional[str] = None


class PlayersResponse(BaseModel):
    supported: bool
    count: int
    players: list[PlayerProfile]
    dataset: DatasetStatus
    disclaimer: str


class PredictionDetail(BaseModel):
    class_name: str
    player_id: str
    display_name: str
    confidence: float
    similarity: Optional[float] = None
    role: Optional[str] = None


class DetectedFace(BaseModel):
    image: Optional[str] = None
    box: Optional[list[int]] = None
    detection_score: Optional[float] = None
    image_size: Optional[list[int]] = None


class PredictionResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    message: Optional[str] = None
    player: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    margin: Optional[float] = None
    similarity: Optional[float] = None
    top_predictions: list[PredictionDetail] = Field(default_factory=list)
    detected_face: Optional[DetectedFace] = None
    faces_detected: int = 0
    other_faces: list[dict[str, Any]] = Field(default_factory=list)
    engine: dict[str, Any] = Field(default_factory=dict)
    dataset_status: Optional[DatasetStatus] = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    disclaimer: Optional[str] = None
    prediction_id: Optional[int] = None


class PredictBase64Request(BaseModel):
    image_base64: str = Field(..., description="data:image/...;base64,... or raw base64")
    source: str = Field("camera", description="camera | upload | api")
    top_k: int = 3
    save: bool = True


class ModelStatusResponse(BaseModel):
    ready: bool
    active_model: Optional[str] = None
    mode: Optional[str] = None            # trained | demo | none
    error: Optional[str] = None
    engine: dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetStatus
    demo_dataset: DatasetStatus
    training_in_progress: bool = False
    disclaimer: str


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class ModelMetricsResponse(BaseModel):
    ready: bool
    message: Optional[str] = None
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    training: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    cross_validation: dict[str, Any] = Field(default_factory=dict)
    per_class: list[dict[str, Any]] = Field(default_factory=list)
    confusion: Optional[ConfusionMatrix] = None
    notes: list[str] = Field(default_factory=list)
    transparency: list[str] = Field(default_factory=list)


class ReloadResponse(BaseModel):
    ok: bool
    message: str
    engine: dict[str, Any] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    dataset: str = Field("dataset", description="dataset | dataset_demo | absolute path")
    classifier: Optional[str] = Field(None, description="svc | logistic | rf | knn")
    extractor: Optional[str] = Field(None, description="auto | hog_lbp | sface | pixel")
    detector: Optional[str] = Field(None, description="auto | yunet | haar")
    augment: Optional[int] = Field(None, ge=0, le=20)
    pca_components: Optional[int] = Field(None, ge=0, le=2048)
    test_size: Optional[float] = Field(None, gt=0.0, lt=0.9)
    confidence_threshold: Optional[float] = Field(None, gt=0.0, lt=1.0)
    min_margin: Optional[float] = Field(None, ge=0.0, lt=0.5)
    seed: Optional[int] = None
    activate: bool = Field(True, description="make this model the active one")


class TrainAcceptedResponse(BaseModel):
    job_id: str
    status: str
    message: str
    dataset: str


class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    progress: float = 0.0
    dataset: Optional[str] = None
    error: Optional[str] = None
    log: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class PredictionRow(BaseModel):
    id: int
    player_id: Optional[str] = None
    predicted_name: Optional[str] = None
    status: str
    reason: Optional[str] = None
    confidence: Optional[float] = None
    source: str
    image_reference: Optional[str] = None
    face_image: Optional[str] = None
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    created_at: Optional[str] = None


class DashboardSummary(BaseModel):
    total_players: int
    supported_players: int
    total_images: int
    dataset_configured: bool
    model_accuracy: Optional[float]
    model_accuracy_note: str
    predictions_total: int
    predictions_today: int
    identified_today: int
    unknown_today: int
    avg_confidence: Optional[float]
    avg_latency_ms: Optional[float]
    recent: list[PredictionRow]
    confidence_distribution: list[dict[str, Any]]
    prediction_distribution: list[dict[str, Any]]
    status_distribution: list[dict[str, Any]]
    daily_trend: list[dict[str, Any]]
    performance: dict[str, Any]
    model: dict[str, Any]
    dataset: DatasetStatus
    disclaimer: str
