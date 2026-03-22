from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable

import pygame

from .game import SnakeGame
from .panel_ui import SidePanelsRenderer
from .protocols import AgentLike, TrainingLike
from .settings import DropoutConfig, ObsConfig, PpoConfig, RewardConfig, Settings


@dataclass(frozen=True)
class AppRuntime:
    game: SnakeGame
    state_file: Path
    obs_config: ObsConfig
    reward_config: RewardConfig
    agent: AgentLike
    training: TrainingLike
    panel_renderer: SidePanelsRenderer
    experiment_name: str = "baseline"


def build_runtime(
    settings: Settings,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    on_score: Callable[[int], None],
    on_episode_info: Callable[[dict], None] | None = None,
    agent_cls=None,
    training_cls=None,
    panel_renderer_cls=None,
    ppo_config: PpoConfig | None = None,
    reward_config: RewardConfig | None = None,
    obs_config: ObsConfig | None = None,
    dropout_config: DropoutConfig | None = None,
    state_dir: Path | None = None,
    experiment_name: str = "baseline",
) -> AppRuntime:
    if agent_cls is None:
        from .ppo_agent import PpoSnakeAgent as agent_cls
    if training_cls is None:
        from .training import PpoTrainingController as training_cls
    if panel_renderer_cls is None:
        panel_renderer_cls = SidePanelsRenderer

    ppo_config = ppo_config or PpoConfig()
    reward_config = reward_config or RewardConfig()
    game = SnakeGame(settings, starvation_factor=int(reward_config.board_starvation_factor))
    state_dir = state_dir or _resolve_state_dir()
    state_file = state_dir / "ui_state.json"
    artifact_dir = state_dir / "ppo" / experiment_name
    legacy_model_file = state_dir / "ppo_snake_model.zip"
    obs_config = obs_config or ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True, use_free_space_features=True, use_tail_trend_features=True)
    agent = agent_cls(
        settings=settings,
        artifact_dir=artifact_dir,
        config=ppo_config,
        reward_config=reward_config,
        obs_config=obs_config,
        dropout_config=dropout_config,
        autoload=False,
        legacy_model_path=legacy_model_file,
    )
    training = training_cls(agent=agent, on_score=on_score, on_episode_info=on_episode_info)
    panel_renderer = panel_renderer_cls(
        settings=settings,
        font=font,
        small_font=small_font,
    )
    return AppRuntime(
        game=game,
        state_file=state_file,
        obs_config=obs_config,
        reward_config=reward_config,
        agent=agent,
        training=training,
        panel_renderer=panel_renderer,
        experiment_name=experiment_name,
    )


def _resolve_state_dir() -> Path:
    if bool(getattr(sys, "frozen", False)):
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / "SnakeFrame" / "state"
        return Path.home() / ".snakeframe" / "state"
    return Path(__file__).resolve().parents[1] / "state"
