from __future__ import annotations

from dataclasses import dataclass, field


def empty_death_counts() -> dict[str, int]:
    return {
        "wall": 0,
        "body": 0,
        "starvation": 0,
        "fill": 0,
        "none": 0,
        "other": 0,
    }


@dataclass
class AppState:
    game_running: bool = False
    options_open: bool = False
    snake_style: str = "topdown_3d"
    fog_density: str = "off"
    debug_overlay: bool = False
    debug_reachable_overlay: bool = False
    space_strategy_enabled: bool = True
    tail_trend_enabled: bool = True
    dropout_enabled: bool = False
    right_panel_tab: str = "train"
    training_episode_scores: list[int] = field(default_factory=list)
    training_episode_steps: list[int] = field(default_factory=list)
    training_death_counts: dict[str, int] = field(default_factory=empty_death_counts)
    ui_state_version: int = 2
    status_severity: str = "info"
    status_text: str = "Ready"
    status_until: float = 0.0
    last_train_message: str = "No training run yet"
    last_error_code: str = ""
    last_error_message: str = ""
    last_error_at: float = 0.0
    last_action_result: str = "idle"
    model_dirty: bool = False
    model_save_state: str = "unknown"
    last_model_save_ok_at: float = 0.0
