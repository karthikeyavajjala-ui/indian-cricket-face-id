"""
SQLAlchemy schema.

    players        - every person the app knows about (real roster + demo personas)
    datasets       - image inventory per player (counts, validity, source folder)
    predictions    - one row per identification attempt (audit trail)
    model_versions - one row per trained model, with the honest metrics snapshot
    training_runs  - job log for POST /api/train

The prediction table deliberately stores the *decision* (status/reason) and the
full top-k list, so a human can audit rejections as well as matches.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String,
                        Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    folder: Mapped[str] = mapped_column(String(120), unique=True)
    display_name: Mapped[str] = mapped_column(String(160))
    role: Mapped[str | None] = mapped_column(String(240), nullable=True)
    batting_style: Mapped[str | None] = mapped_column(String(120), nullable=True)
    bowling_style: Mapped[str | None] = mapped_column(String(120), nullable=True)
    jersey_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    born: Mapped[str | None] = mapped_column(String(160), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    career_highlights: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    fictional: Mapped[bool] = mapped_column(Boolean, default=False)
    supported: Mapped[bool] = mapped_column(Boolean, default=False)  # present in active model
    dataset_images: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    datasets: Mapped[list["DatasetRecord"]] = relationship(back_populates="player",
                                                           cascade="all, delete-orphan")


class DatasetRecord(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("player_id", "path", name="uq_dataset_player_path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    path: Mapped[str] = mapped_column(String(400))
    source: Mapped[str] = mapped_column(String(40), default="local")   # local | demo
    images_count: Mapped[int] = mapped_column(Integer, default=0)
    valid_images: Mapped[int] = mapped_column(Integer, default=0)
    invalid_images: Mapped[int] = mapped_column(Integer, default=0)
    licence_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_scanned_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    player: Mapped["Player"] = relationship(back_populates="datasets")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), index=True)
    model_path: Mapped[str] = mapped_column(String(400))
    dataset_path: Mapped[str | None] = mapped_column(String(400), nullable=True)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(80), nullable=True)
    classifier: Mapped[str | None] = mapped_column(String(40), nullable=True)
    feature_extractor: Mapped[str | None] = mapped_column(String(40), nullable=True)
    detector: Mapped[str | None] = mapped_column(String(40), nullable=True)
    n_classes: Mapped[int] = mapped_column(Integer, default=0)
    training_samples: Mapped[int] = mapped_column(Integer, default=0)
    validation_samples: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    cross_val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    trained_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    predicted_folder: Mapped[str | None] = mapped_column(String(120), nullable=True)
    predicted_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)     # identified | unknown | ...
    reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_predictions: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON
    faces_detected: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload | camera | api
    image_reference: Mapped[str | None] = mapped_column(String(400), nullable=True)
    face_image: Mapped[str | None] = mapped_column(String(400), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)

    player: Mapped["Player | None"] = relationship()


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    dataset_path: Mapped[str | None] = mapped_column(String(400), nullable=True)
    options_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    log: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"),
                                                         nullable=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
