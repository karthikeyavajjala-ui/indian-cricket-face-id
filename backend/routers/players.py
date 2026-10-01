"""Player catalog endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from ml import config as ml_config

from ..config import settings
from ..database import get_session
from ..schemas import PlayerProfile, PlayersResponse
from ..services import model_service, player_service

router = APIRouter(prefix="/api/players", tags=["players"])


def _model_block() -> dict:
    bundle = model_service.current_bundle()
    metrics = (bundle.metrics if bundle else {}) or {}
    return {
        "per_class": {row["class"]: row for row in
                      (metrics.get("evaluation", {}) or {}).get("per_class", [])},
        "class_samples": metrics.get("class_samples", {}),
    }


@router.get("", response_model=PlayersResponse, summary="List supported players")
def list_players(include_catalog: bool = Query(True, description="include players with no dataset images"),
                 session: Session = Depends(get_session)) -> PlayersResponse:
    player_service.sync_catalog(session)
    player_service.sync_dataset(session)
    player_service.sync_supported_flags(session, model_service.current_bundle().classes
                                        if model_service.current_bundle() else [])
    session.commit()

    dataset = ml_config.dataset_summary_safe(settings.dataset_dir)
    demo = ml_config.dataset_summary_safe(settings.demo_dataset_dir)
    active = dataset if dataset["configured"] else demo

    players = player_service.list_players(session, _model_block(),
                                          dataset_configured=active["configured"],
                                          include_unsupported=include_catalog)
    return PlayersResponse(
        supported=bool([p for p in players if p["supported"]]),
        count=len(players),
        players=players,
        dataset=active,
        disclaimer=("Only the players marked 'supported' exist in the trained model's "
                    "dataset. This app does not claim to identify every Indian cricketer."),
    )


@router.get("/{player_id}", response_model=PlayerProfile, summary="Single player profile")
def get_player(player_id: str, session: Session = Depends(get_session)) -> PlayerProfile:
    dataset = ml_config.dataset_summary_safe(settings.dataset_dir)
    demo = ml_config.dataset_summary_safe(settings.demo_dataset_dir)
    active = dataset if dataset["configured"] else demo
    profile = player_service.get_player(session, player_id, _model_block(),
                                        dataset_configured=active["configured"])
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown player '{player_id}'")
    session.commit()
    return PlayerProfile(**profile)


@router.get("/{player_id}/avatar.svg", include_in_schema=False)
def avatar(player_id: str, session: Session = Depends(get_session)) -> Response:
    profile = player_service.get_player(session, player_id)
    name = profile["name"] if profile else player_id
    return Response(content=player_service.avatar_svg(name, player_id),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})
