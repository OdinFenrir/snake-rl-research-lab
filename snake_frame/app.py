from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Callable

import pygame

from .app_actions import AppActions
from . import app_events, app_orchestrator, app_rendering
from .app_factory import build_runtime
from .app_state import AppState, empty_death_counts
from .controls_builder import build_controls
from .gameplay_controller import GameplayController, GameplayTelemetrySnapshot
from .input_controller import KeyboardInputController
from .layout_engine import LayoutEngine, LayoutSnapshot
from .holdout_eval import HoldoutEvalController
from .panel_ui import PanelControls, SidePanelsRenderer
from .settings import Settings
from .state_io import UiStateErrorCode, load_ui_state_result, save_ui_state
from .theme import available_themes, get_design_tokens, get_theme, normalize_theme_name
from .training_metrics import avg_last, overfit_signal
from .ui_state_model import (
    ControlAuthorityPolicy,
    ModelState,
    TrainingState,
    UIStateModel,
    derive_control_authority_policy,
)
from .ui import Button, NumericInput

logger = logging.getLogger(__name__)


class SnakeFrameApp:
    DETACHED_EXPERIMENT: str = "_detached_session"
    _LIVE_TPM_MIN: int = 1
    _LIVE_TPM_MAX: int = 12
    _HOLDOUT_MAX_STEPS: int = 5000

    def __init__(self, *, startup_route: str | None = None) -> None:
        self.settings = Settings()
        self.settings.theme_name = normalize_theme_name(getattr(self.settings, "theme_name", ""))
        self.theme = get_theme(getattr(self.settings, "theme_name", ""))
        self.design_tokens = get_design_tokens(getattr(self.settings, "theme_name", ""))
        self.layout_engine = LayoutEngine(self.settings)
        self.layout: LayoutSnapshot = self.layout_engine.update(
            self.settings.window_width_px,
            int(self.settings.window_height_px or self.settings.window_px),
        )
        pygame.init()
        pygame.display.set_caption("Snake Frame (PPO Only)")
        self._is_fullscreen = False
        self._windowed_size = (self.layout.window.width, self.layout.window.height)
        window_flags = self._window_flags()
        self.surface = pygame.display.set_mode(
            (self.layout.window.width, self.layout.window.height),
            window_flags,
        )
        self.clock = pygame.time.Clock()
        self._return_to_workspace_menu = False
        self.font = pygame.font.SysFont(
            self.design_tokens.typography.title_family,
            int(self.design_tokens.typography.title_size),
            bold=bool(self.design_tokens.typography.title_bold),
        )
        self.small_font = pygame.font.SysFont(
            self.design_tokens.typography.body_family,
            int(self.design_tokens.typography.body_size),
            bold=bool(self.design_tokens.typography.body_bold),
        )
        self._frame_ms_samples: deque[float] = deque(maxlen=240)

        runtime = build_runtime(
            settings=self.settings,
            font=self.font,
            small_font=self.small_font,
            on_score=self._append_episode_score,
            on_episode_info=self._append_training_episode_info,
            experiment_name=self.DETACHED_EXPERIMENT,
        )
        self.game = runtime.game
        self.app_state = AppState()
        self.state_file = runtime.state_file
        self.ui_prefs_file = self.state_file.with_name("ui_prefs.json")
        self.obs_config = runtime.obs_config
        self.agent = runtime.agent
        self.experiment_name = runtime.experiment_name
        self._detached_mode = bool(self.experiment_name == self.DETACHED_EXPERIMENT)
        self.training = runtime.training
        self.holdout_eval = HoldoutEvalController(
            agent=self.agent,
            settings=self.settings,
            obs_config=runtime.obs_config,
            reward_config=runtime.reward_config,
            out_dir=Path(__file__).resolve().parents[1] / "artifacts" / "live_eval",
        )
        self._holdout_eval_mode = HoldoutEvalController.MODE_PPO_ONLY
        self.panel_renderer: SidePanelsRenderer = runtime.panel_renderer
        self.input_controller = KeyboardInputController(self.game)
        self.gameplay = GameplayController(
            game=self.game,
            agent=self.agent,
            settings=self.settings,
            obs_config=self.obs_config,
            space_strategy_enabled=self.app_state.space_strategy_enabled,
            artifact_dir=runtime.state_file.parent / "ppo" / runtime.experiment_name,
        )
        self.gameplay.set_tail_trend_enabled(bool(getattr(self.app_state, "tail_trend_enabled", True)))
        self._build_controls()
        self.actions = AppActions(
            app_state=self.app_state,
            game=self.game,
            agent=self.agent,
            training=self.training,
            generations_input=self.generations_input,
            state_file=self.state_file,
            ui_state_provider=self._derive_ui_state,
            get_theme_name=lambda: self.theme.name,
            set_theme_name=lambda name: self._apply_theme(name, announce=False),
            get_experiment_name=lambda: self.experiment_name,
            switch_experiment=self._switch_experiment,
        )
        self._bind_button_actions()
        self._run_session_log_legacy_path = Path(__file__).resolve().parents[1] / "artifacts" / "live_eval" / "run_session_log.jsonl"
        self._run_session_log_path = self._experiment_run_log_path(self.experiment_name)
        self._eval_suite_dir = Path(__file__).resolve().parents[1] / "artifacts" / "live_eval" / "suites"
        self._eval_suite_active = False
        self._eval_suite_phase = "idle"
        self._eval_suite_ppo_summary: dict | None = None
        self._eval_suite_controller_summary: dict | None = None
        self._eval_suite_started_at_unix_s: float = 0.0
        self._eval_suite_max_steps = int(self._HOLDOUT_MAX_STEPS)
        self._last_logged_run_episodes = int(len(getattr(self.game, "episode_scores", [])))
        self._runlog_prev_decisions = 0
        self._runlog_prev_interventions = 0
        self._runlog_prev_stuck_episodes = 0
        self._train_rate_last_time_s = float(time.perf_counter())
        self._train_rate_last_done_steps = 0
        self._train_rate_ema_sps = 0.0
        self._runtime_health_next_refresh_s = 0.0
        self._runtime_health_cached_lines: list[str] = []
        restored = self._load_ui_preferences()
        self._apply_startup_route(startup_route)
        startup_warnings = self._run_startup_self_checks()
        if startup_warnings:
            self.actions.set_status(
                "Startup degraded mode: " + "; ".join(startup_warnings[:2]),
                severity="warn",
            )
            self.app_state.last_error_code = "startup_degraded"
            self.app_state.last_error_message = "; ".join(startup_warnings[:2])
        elif restored:
            self.actions.set_status("UI preferences restored")
        else:
            self.actions.set_status("New session (blank). Use Load to restore state/model.")

    def _apply_startup_route(self, startup_route: str | None) -> None:
        route = str(startup_route or "").strip().lower()
        if route == "settings":
            self.app_state.options_open = True
            return
        if route == "analysis_tools":
            self.app_state.options_open = False
            self.app_state.right_panel_tab = "run"
            return
        if route == "live_training":
            self.app_state.options_open = False

    def _build_controls(self) -> None:
        controls = build_controls(
            self.settings,
            min_graph_height=self.layout.graph.min_graph_height,
            max_graph_height=self.layout.graph.max_graph_height,
            graph_margin=self.layout.graph.graph_margin,
            graph_top=self.layout.graph.graph_top,
            control_row_height=self.layout.graph.control_row_height,
            control_gap=self.layout.graph.control_gap,
            status_line_height=self.layout.graph.status_line_height,
            status_line_count=self.layout.graph.status_line_count,
        )
        self.graph_rect = pygame.Rect(controls.graph_rect)
        self.training_graph_rect = pygame.Rect(controls.training_graph_rect)
        self.run_graph_rect = pygame.Rect(controls.run_graph_rect)
        self.training_header_y = controls.training_header_y
        self.training_badges_y = controls.training_badges_y
        self.run_header_y = controls.run_header_y
        self.run_badges_y = controls.run_badges_y
        self.panel_controls: PanelControls = controls.panel_controls
        self.generations_input: NumericInput = controls.generations_input
        self.btn_train_start: Button = controls.btn_train_start
        self.btn_train_stop: Button = controls.btn_train_stop
        self.btn_save: Button = controls.btn_save
        self.btn_load: Button = controls.btn_load
        self.btn_delete: Button = controls.btn_delete
        self.btn_game_start: Button = controls.btn_game_start
        self.btn_game_stop: Button = controls.btn_game_stop
        self.btn_restart: Button = controls.btn_restart
        self.btn_options: Button = controls.btn_options
        self.btn_options_close: Button = controls.btn_options_close
        self.btn_adaptive_toggle: Button = controls.btn_adaptive_toggle
        self.btn_space_strategy_toggle: Button = controls.btn_space_strategy_toggle
        self.btn_tail_trend_toggle: Button = controls.btn_tail_trend_toggle
        self.btn_dropout_toggle: Button = controls.btn_dropout_toggle
        self.btn_theme_cycle: Button = controls.btn_theme_cycle
        self.btn_board_bg_cycle: Button = controls.btn_board_bg_cycle
        self.btn_snake_style_cycle: Button = controls.btn_snake_style_cycle
        self.btn_fog_cycle: Button = controls.btn_fog_cycle
        self.btn_speed_down: Button = controls.btn_speed_down
        self.btn_speed_up: Button = controls.btn_speed_up
        self.btn_eval_suite: Button = controls.btn_eval_suite
        self.btn_eval_mode_ppo: Button = controls.btn_eval_mode_ppo
        self.btn_eval_mode_controller: Button = controls.btn_eval_mode_controller
        self.btn_eval_holdout: Button = controls.btn_eval_holdout
        self.btn_debug_toggle: Button = controls.btn_debug_toggle
        self.btn_reachable_toggle: Button = controls.btn_reachable_toggle
        self.btn_diagnostics: Button = controls.btn_diagnostics
        self.btn_tab_train: Button = controls.btn_tab_train
        self.btn_tab_run: Button = controls.btn_tab_run
        self.btn_tab_debug: Button = controls.btn_tab_debug
        if hasattr(self, "actions"):
            self.actions.set_generations_input(self.generations_input)

    def _bind_button_actions(self) -> None:
        self._button_actions_main: list[tuple[Button, Callable[[], None]]] = [
            (self.btn_tab_train, self._on_tab_train_clicked),
            (self.btn_tab_run, self._on_tab_run_clicked),
            (self.btn_tab_debug, self._on_tab_debug_clicked),
            (self.btn_train_start, self.actions.on_train_start_clicked),
            (self.btn_train_stop, self.actions.on_train_stop_clicked),
            (self.btn_save, self.actions.handle_save_clicked),
            (self.btn_load, self.actions.handle_load_clicked),
            (self.btn_delete, self.actions.handle_delete_clicked),
            (self.btn_game_start, self.actions.on_game_start_clicked),
            (self.btn_game_stop, self.actions.on_game_stop_clicked),
            (self.btn_restart, self._on_restart_clicked),
            (self.btn_options, self._on_options_open_clicked),
        ]
        self._button_actions_options: list[tuple[Button, Callable[[], None]]] = [
            (self.btn_tab_train, self._on_tab_train_clicked),
            (self.btn_tab_run, self._on_tab_run_clicked),
            (self.btn_tab_debug, self._on_tab_debug_clicked),
            (self.btn_adaptive_toggle, self.actions.on_adaptive_toggle_clicked),
            (self.btn_space_strategy_toggle, self._on_space_strategy_toggle_clicked),
            (self.btn_tail_trend_toggle, self._on_tail_trend_toggle_clicked),
            (self.btn_dropout_toggle, self.actions.on_dropout_toggle_clicked),
            (self.btn_theme_cycle, self._on_theme_cycle_clicked),
            (self.btn_board_bg_cycle, self._on_board_background_cycle_clicked),
            (self.btn_snake_style_cycle, self._on_snake_style_cycle_clicked),
            (self.btn_fog_cycle, self._on_fog_cycle_clicked),
            (self.btn_speed_down, self._on_live_speed_down_clicked),
            (self.btn_speed_up, self._on_live_speed_up_clicked),
            (self.btn_eval_suite, self._on_eval_suite_clicked),
            (self.btn_eval_holdout, self._on_eval_holdout_clicked),
            (self.btn_debug_toggle, self.actions.on_debug_toggle_clicked),
            (self.btn_reachable_toggle, self.actions.on_reachable_toggle_clicked),
            (self.btn_diagnostics, self.actions.handle_diagnostics_clicked),
            (self.btn_options_close, self._on_options_close_clicked),
        ]

    def _on_tab_train_clicked(self) -> None:
        self.app_state.right_panel_tab = "train"
        self.actions.set_status("Panel tab: Train")

    def _on_tab_run_clicked(self) -> None:
        self.app_state.right_panel_tab = "run"
        self.actions.set_status("Panel tab: Run")

    def _on_tab_debug_clicked(self) -> None:
        self.app_state.right_panel_tab = "debug"
        self.actions.set_status("Panel tab: Debug")

    def _on_space_strategy_toggle_clicked(self) -> None:
        enabled = not bool(self.app_state.space_strategy_enabled)
        self.app_state.space_strategy_enabled = bool(enabled)
        self.gameplay.set_space_strategy_enabled(enabled)
        state = "ON" if enabled else "OFF"
        self.actions.set_status(f"Space strategy {state}")

    def _on_tail_trend_toggle_clicked(self) -> None:
        current = getattr(self.app_state, 'tail_trend_enabled', True)
        new_state = not current
        self.app_state.tail_trend_enabled = bool(new_state)
        self.gameplay.set_tail_trend_enabled(bool(new_state))
        state = "ON" if new_state else "OFF"
        self.actions.set_status(f"Tail trend features {state}")

    def _on_theme_cycle_clicked(self) -> None:
        themes = available_themes()
        current = normalize_theme_name(getattr(self.settings, "theme_name", ""))
        if not themes:
            return
        try:
            idx = themes.index(current)
        except ValueError:
            idx = -1
        next_theme = themes[(idx + 1) % len(themes)]
        self._apply_theme(next_theme, announce=True)

    def _on_board_background_cycle_clicked(self) -> None:
        mode = self.game.cycle_board_background_mode()
        self.actions.set_status(f"Board background: {self.game.board_background_label()} ({mode})")

    def _on_snake_style_cycle_clicked(self) -> None:
        style = self.game.cycle_snake_style()
        self.app_state.snake_style = str(style)
        self.actions.set_status(f"Snake style: {self.game.snake_style_label()} ({style})")

    def _on_fog_cycle_clicked(self) -> None:
        density = self.game.cycle_fog_density()
        self.app_state.fog_density = str(density)
        self.actions.set_status(f"Fog density: {self.game.fog_density_label()} ({density})")

    def _set_live_ticks_per_move(self, value: int) -> None:
        clamped = max(int(self._LIVE_TPM_MIN), min(int(self._LIVE_TPM_MAX), int(value)))
        self.settings.ticks_per_move = int(clamped)
        speed_pct = int(round(100.0 * float(5) / float(max(1, clamped))))
        self.actions.set_status(f"Live speed set: tpm={clamped} ({speed_pct}% of base)")

    def _on_live_speed_down_clicked(self) -> None:
        self._set_live_ticks_per_move(int(self.settings.ticks_per_move) + 1)

    def _on_live_speed_up_clicked(self) -> None:
        self._set_live_ticks_per_move(int(self.settings.ticks_per_move) - 1)

    def _on_eval_mode_ppo_clicked(self) -> None:
        snap = self.holdout_eval.snapshot()
        if bool(snap.active):
            self.actions.set_status("Holdout eval already running", severity="warn")
            return
        self._holdout_eval_mode = HoldoutEvalController.MODE_PPO_ONLY
        self.actions.set_status("Eval mode set: PPO-only")

    def _on_eval_mode_controller_clicked(self) -> None:
        snap = self.holdout_eval.snapshot()
        if bool(snap.active):
            self.actions.set_status("Holdout eval already running", severity="warn")
            return
        self._holdout_eval_mode = HoldoutEvalController.MODE_CONTROLLER_ON
        self.actions.set_status("Eval mode set: Controller ON")

    def _on_eval_holdout_clicked(self) -> None:
        train_snap = self.training.snapshot()
        if bool(train_snap.active):
            self.actions.set_status("Cannot run holdout eval while training is active", severity="warn")
            return
        if bool(self._eval_suite_active):
            self.actions.set_status("Eval suite is running", severity="warn")
            return
        selector = "best"
        get_selector = getattr(self.agent, "get_model_selector", None)
        if callable(get_selector):
            try:
                selector = str(get_selector() or "best").strip().lower()
            except Exception:
                selector = "best"
        started = self.holdout_eval.start(
            mode=str(getattr(self, "_holdout_eval_mode", HoldoutEvalController.MODE_PPO_ONLY)),
            model_selector=selector,
            max_steps=int(self._HOLDOUT_MAX_STEPS),
        )
        if not bool(started):
            self.actions.set_status("Holdout eval already running", severity="warn")
            return
        mode_label = "Controller ON" if str(getattr(self, "_holdout_eval_mode", "ppo_only")) == HoldoutEvalController.MODE_CONTROLLER_ON else "PPO-only"
        self.actions.set_status(f"Holdout eval started ({mode_label}, selector={selector})")

    def _on_eval_suite_clicked(self) -> None:
        train_snap = self.training.snapshot()
        if bool(train_snap.active):
            self.actions.set_status("Cannot run eval suite while training is active", severity="warn")
            return
        if bool(self._eval_suite_active):
            self.actions.set_status("Eval suite already running", severity="warn")
            return
        snap = self.holdout_eval.snapshot()
        if bool(snap.active):
            self.actions.set_status("Holdout eval already running", severity="warn")
            return
        self._eval_suite_active = True
        self._eval_suite_phase = "ppo"
        self._eval_suite_ppo_summary = None
        self._eval_suite_controller_summary = None
        self._eval_suite_started_at_unix_s = float(time.time())
        self._eval_suite_max_steps = int(self._HOLDOUT_MAX_STEPS)
        selector = "best"
        get_selector = getattr(self.agent, "get_model_selector", None)
        if callable(get_selector):
            try:
                selector = str(get_selector() or "best").strip().lower()
            except Exception:
                selector = "best"
        started = self.holdout_eval.start(
            mode=HoldoutEvalController.MODE_PPO_ONLY,
            model_selector=selector,
            max_steps=int(self._eval_suite_max_steps),
        )
        if not bool(started):
            self._eval_suite_active = False
            self._eval_suite_phase = "idle"
            self.actions.set_status("Could not start eval suite", severity="warn")
            return
        self.actions.set_status(f"Eval suite started (phase 1/2: PPO-only, selector={selector})")

    def _load_latest_holdout_summary(self) -> dict | None:
        snap = self.holdout_eval.snapshot()
        path_str = str(getattr(snap, "latest_summary_path", "") or "").strip()
        path = Path(path_str) if path_str else (Path(__file__).resolve().parents[1] / "artifacts" / "live_eval" / "latest_summary.json")
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed reading holdout summary: %s", path)
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _suite_delta(ppo_scores: dict, controller_scores: dict) -> dict:
        p_mean = float(ppo_scores.get("mean", 0.0))
        c_mean = float(controller_scores.get("mean", 0.0))
        p_median = float(ppo_scores.get("median", 0.0))
        c_median = float(controller_scores.get("median", 0.0))
        p_p90 = float(ppo_scores.get("p90", 0.0))
        c_p90 = float(controller_scores.get("p90", 0.0))
        return {
            "mean_delta_controller_minus_ppo": float(c_mean - p_mean),
            "median_delta_controller_minus_ppo": float(c_median - p_median),
            "p90_delta_controller_minus_ppo": float(c_p90 - p_p90),
            "mean_ratio_controller_over_ppo": float(c_mean / p_mean) if abs(p_mean) > 1e-9 else 0.0,
        }

    @staticmethod
    def _paired_seed_delta_stats(ppo_rows: list[dict] | None, controller_rows: list[dict] | None) -> dict:
        ppo_by_seed: dict[int, int] = {}
        for row in list(ppo_rows or []):
            if not isinstance(row, dict):
                continue
            try:
                seed = int(row.get("seed"))
                score = int(row.get("score"))
            except Exception:
                continue
            ppo_by_seed[seed] = score

        ctrl_by_seed: dict[int, int] = {}
        for row in list(controller_rows or []):
            if not isinstance(row, dict):
                continue
            try:
                seed = int(row.get("seed"))
                score = int(row.get("score"))
            except Exception:
                continue
            ctrl_by_seed[seed] = score

        paired_seeds = sorted(set(ppo_by_seed.keys()) & set(ctrl_by_seed.keys()))
        deltas = [int(ctrl_by_seed[s] - ppo_by_seed[s]) for s in paired_seeds]
        if not deltas:
            return {
                "paired_seed_count": 0,
                "paired_worse_count": 0,
                "paired_improved_count": 0,
                "paired_equal_count": 0,
                "paired_mean_delta_controller_minus_ppo": 0.0,
                "paired_median_delta_controller_minus_ppo": 0.0,
            }

        ordered = sorted(deltas)
        n = len(ordered)
        if n % 2 == 1:
            median = float(ordered[n // 2])
        else:
            median = float((ordered[(n // 2) - 1] + ordered[n // 2]) / 2.0)
        return {
            "paired_seed_count": int(n),
            "paired_worse_count": int(sum(1 for d in deltas if d < 0)),
            "paired_improved_count": int(sum(1 for d in deltas if d > 0)),
            "paired_equal_count": int(sum(1 for d in deltas if d == 0)),
            "paired_mean_delta_controller_minus_ppo": float(sum(deltas) / float(n)),
            "paired_median_delta_controller_minus_ppo": float(median),
        }

    @staticmethod
    def _extract_mean_interventions_pct(controller_summary: dict) -> float | None:
        if not isinstance(controller_summary, dict):
            return None
        raw = controller_summary.get("mean_interventions_pct")
        try:
            if raw is not None:
                return float(raw)
        except Exception:
            pass
        rows = list(controller_summary.get("controller_telemetry_rows") or [])
        vals: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                vals.append(float(row.get("interventions_pct", 0.0)))
            except Exception:
                continue
        if not vals:
            return None
        return float(sum(vals) / float(len(vals)))

    def _persist_eval_suite_bundle(self) -> tuple[Path | None, str]:
        if self._eval_suite_ppo_summary is None or self._eval_suite_controller_summary is None:
            return None, "Suite incomplete"
        live_eval_dir = Path(__file__).resolve().parents[1] / "artifacts" / "live_eval"
        self._eval_suite_dir.mkdir(parents=True, exist_ok=True)
        generated = datetime.now(timezone.utc)
        selector = "best"
        get_selector = getattr(self.agent, "get_model_selector", None)
        if callable(get_selector):
            try:
                selector = str(get_selector() or "best").strip().lower()
            except Exception:
                selector = "best"
        comparison = self._suite_delta(
            dict(self._eval_suite_ppo_summary.get("scores", {})),
            dict(self._eval_suite_controller_summary.get("scores", {})),
        )
        comparison.update(
            self._paired_seed_delta_stats(
                list(self._eval_suite_ppo_summary.get("rows", [])),
                list(self._eval_suite_controller_summary.get("rows", [])),
            )
        )
        mean_interventions_pct = self._extract_mean_interventions_pct(dict(self._eval_suite_controller_summary))
        if mean_interventions_pct is not None:
            comparison["mean_interventions_pct"] = float(mean_interventions_pct)
        suite = {
            "generated_at_utc": generated.isoformat(),
            "suite_started_at_unix_s": float(self._eval_suite_started_at_unix_s),
            "max_steps": int(self._eval_suite_max_steps),
            "model_selector": str(selector),
            "ppo_only": dict(self._eval_suite_ppo_summary),
            "controller_on": dict(self._eval_suite_controller_summary),
            "comparison": comparison,
            "source_files": {
                "live_eval_dir": str(live_eval_dir),
            },
        }
        stamped = self._eval_suite_dir / f"suite_{generated.strftime('%Y%m%d_%H%M%S')}.json"
        latest = self._eval_suite_dir / "latest_suite.json"
        payload = json.dumps(suite, indent=2, allow_nan=False)
        stamped.write_text(payload, encoding="utf-8")
        latest.write_text(payload, encoding="utf-8")

        suites = sorted(
            [p for p in self._eval_suite_dir.glob("suite_*.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in suites[2:]:
            try:
                old.unlink()
            except Exception:
                logger.exception("Failed pruning old suite file: %s", old)
        cmp = suite["comparison"]
        message = (
            f"Suite done: PPO {suite['ppo_only']['scores']['mean']:.1f} vs Ctrl {suite['controller_on']['scores']['mean']:.1f} "
            f"(delta {cmp['mean_delta_controller_minus_ppo']:+.1f})"
        )
        return stamped, message

    def _poll_holdout_eval(self) -> None:
        message = self.holdout_eval.poll_completion()
        if not message:
            return
        if bool(self._eval_suite_active):
            lower = str(message).lower()
            if "failed" in lower:
                self._eval_suite_active = False
                self._eval_suite_phase = "idle"
                self.actions.set_status(str(message), severity="warn", duration=6.0)
                return
            latest = self._load_latest_holdout_summary()
            if self._eval_suite_phase == "ppo":
                self._eval_suite_ppo_summary = latest
                started = self.holdout_eval.start(
                    mode=HoldoutEvalController.MODE_CONTROLLER_ON,
                    model_selector=str(getattr(self.agent, "get_model_selector", lambda: "best")()),
                    max_steps=int(self._eval_suite_max_steps),
                )
                if not bool(started):
                    self._eval_suite_active = False
                    self._eval_suite_phase = "idle"
                    self.actions.set_status("Eval suite phase switch failed", severity="warn", duration=6.0)
                    return
                self._eval_suite_phase = "controller"
                self.actions.set_status("Eval suite phase 2/2: Controller ON", severity="info", duration=4.0)
                return
            if self._eval_suite_phase == "controller":
                self._eval_suite_controller_summary = latest
                self._eval_suite_active = False
                self._eval_suite_phase = "idle"
                _path, suite_message = self._persist_eval_suite_bundle()
                self.actions.set_status(suite_message, severity="info", duration=7.0)
                return
        severity = "warn" if "failed" in str(message).lower() else "info"
        self.actions.set_status(str(message), severity=severity, duration=6.0)

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")

    def _experiment_run_log_path(self, experiment_name: str | None = None) -> Path:
        exp = str(experiment_name or getattr(self, "experiment_name", "") or "").strip()
        if not exp:
            exp = "baseline"
        return Path(self.state_file).parent / "ppo" / exp / "run_logs" / "run_session_log.jsonl"

    def _resolve_latest_run_id_for_logging(self) -> str:
        agent = getattr(self, "agent", None)
        live_run_id = str(getattr(agent, "latest_run_id", "") or "").strip()
        if live_run_id:
            return live_run_id
        try:
            metadata_path = Path(self.state_file).parent / "ppo" / str(self.experiment_name) / "metadata.json"
            if not metadata_path.exists():
                return ""
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return ""
            run_id = str(payload.get("latest_run_id", "") or "").strip()
            return run_id
        except Exception:
            return ""

    def _append_run_session_log_if_needed(self) -> None:
        scores = [int(v) for v in getattr(self.game, "episode_scores", [])]
        if len(scores) <= int(self._last_logged_run_episodes):
            return
        telemetry: GameplayTelemetrySnapshot = self.gameplay.telemetry_snapshot()
        run_id = self._resolve_latest_run_id_for_logging()
        experiment = str(getattr(self, "experiment_name", "") or "").strip()
        for idx in range(int(self._last_logged_run_episodes), len(scores)):
            decisions_total = int(telemetry.decisions_total)
            interventions_total = int(telemetry.interventions_total)
            stuck_total = int(telemetry.stuck_episodes_total)
            delta_decisions = max(0, int(decisions_total - int(self._runlog_prev_decisions)))
            delta_interventions = max(0, int(interventions_total - int(self._runlog_prev_interventions)))
            intervention_pct = 100.0 * float(delta_interventions) / float(max(1, delta_decisions))
            payload = {
                "generated_at_unix_s": time.time(),
                "run_id": str(run_id),
                "experiment": str(experiment),
                "episode_index": int(idx + 1),
                "score": int(scores[idx]),
                "death_reason": str(telemetry.last_death_reason),
                "mode": str(telemetry.current_mode),
                "train_total_steps": int(self.training.snapshot().current_steps),
                "interventions_pct": float(intervention_pct),
                "interventions_delta": int(delta_interventions),
                "decisions_delta": int(delta_decisions),
                "risk_total": int(telemetry.pocket_risk_total),
                "stuck_episode_delta": int(max(0, int(stuck_total - int(self._runlog_prev_stuck_episodes)))),
                "loop_escape_activations_total": int(telemetry.loop_escape_activations_total),
            }
            try:
                targets: list[Path] = []
                primary = getattr(self, "_run_session_log_path", None)
                legacy = getattr(self, "_run_session_log_legacy_path", None)
                if isinstance(primary, Path):
                    targets.append(primary)
                if isinstance(legacy, Path):
                    targets.append(legacy)
                unique_targets: list[Path] = []
                seen: set[str] = set()
                for target in targets:
                    key = str(target.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_targets.append(target)
                for target in unique_targets:
                    self._append_jsonl(target, payload)
            except Exception:
                logger.exception("Failed to append run session log")
            self._runlog_prev_decisions = int(decisions_total)
            self._runlog_prev_interventions = int(interventions_total)
            self._runlog_prev_stuck_episodes = int(stuck_total)
        self._last_logged_run_episodes = int(len(scores))

    def _on_options_open_clicked(self) -> None:
        self.app_state.options_open = True
        self.actions.set_status("Options opened")

    def _on_options_close_clicked(self) -> None:
        self.app_state.options_open = False
        self.actions.set_status("Options closed")

    def _on_restart_clicked(self) -> None:
        self.actions.on_restart_clicked()
        self.gameplay.reset_episode_tracking()

    def _switch_experiment(self, experiment_name: str) -> bool:
        name = str(experiment_name or "").strip()
        if not name:
            return False
        ppo_root = Path(self.state_file).parent / "ppo"
        target_dir = ppo_root / name
        try:
            ppo_root.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Failed preparing experiment directory: %s", target_dir)
            return False
        try:
            self.training.reset_tracking_from_agent()
            self.agent.switch_artifact_dir(target_dir)
            self.gameplay.set_artifact_dir(target_dir)
            self.experiment_name = name
            self._detached_mode = bool(self.experiment_name == self.DETACHED_EXPERIMENT)
            self._run_session_log_path = self._experiment_run_log_path(self.experiment_name)
            self.app_state.model_dirty = False
            self.app_state.model_save_state = "saved" if bool(getattr(self.agent, "is_ready", False)) else "no_model"
            self.app_state.last_model_save_ok_at = 0.0
            self._last_logged_run_episodes = int(len(getattr(self.game, "episode_scores", [])))
            self._runlog_prev_decisions = 0
            self._runlog_prev_interventions = 0
            self._runlog_prev_stuck_episodes = 0
            return True
        except Exception:
            logger.exception("Failed switching experiment runtime to %s", name)
            return False

    def run(self) -> bool:
        return bool(app_orchestrator.run_loop(self))

    def _handle_global_event(self, event: pygame.event.Event) -> bool:
        return app_events.handle_global_event(self, event)

    def _handle_buttons(self, event: pygame.event.Event) -> None:
        app_events.handle_buttons(self, event)

    def _display_is_fullscreen(self) -> bool:
        return app_events.display_is_fullscreen(self)

    def _window_flags(self) -> int:
        return app_events.window_flags(self)

    def _recreate_window(self, target_size: tuple[int, int] | None = None) -> None:
        app_events.recreate_window(self, target_size)

    def _resize(self, width: int, height: int) -> None:
        app_events.resize(self, width, height)

    def _draw(self) -> None:
        app_rendering.draw(self)

    def _draw_board_frame(self) -> None:
        app_rendering.draw_board_frame(self)

    def _draw_window_chrome(self) -> None:
        app_rendering.draw_window_chrome(self)

    def _draw_runtime_banners(self, control_policy: ControlAuthorityPolicy) -> None:
        app_rendering.draw_runtime_banners(self, control_policy)

    def _draw_perf_overlay(self) -> None:
        app_rendering.draw_perf_overlay(self)

    def _safe_render_text(self, text: str, color: tuple[int, int, int], *, small: bool) -> pygame.Surface:
        return app_rendering.safe_render_text(self, text, color, small=small)

    def _draw_options_window(self) -> None:
        app_rendering.draw_options_window(self)

    @staticmethod
    def _safe_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return max(int(minimum), min(int(maximum), int(parsed)))

    def _load_ui_preferences(self) -> bool:
        try:
            result = load_ui_state_result(self.ui_prefs_file)
        except OSError:
            logger.exception("Failed to load UI preferences: %s", self.ui_prefs_file)
            return False
        payload = result.payload or {}
        if result.invalid or not payload:
            return False

        # Detached startup mode: never switch experiment from preferences.

        if "themeName" in payload:
            self._apply_theme(normalize_theme_name(str(payload.get("themeName"))), announce=False)

        self.settings.window_borderless = bool(payload.get("windowBorderless", self.settings.window_borderless))
        self.app_state.debug_overlay = bool(payload.get("debugOverlay", self.app_state.debug_overlay))
        self.app_state.debug_reachable_overlay = bool(payload.get("debugReachableOverlay", self.app_state.debug_reachable_overlay))
        self.app_state.space_strategy_enabled = bool(payload.get("spaceStrategyEnabled", self.app_state.space_strategy_enabled))
        tab_name = str(payload.get("rightPanelTab", getattr(self.app_state, "right_panel_tab", "train"))).strip().lower()
        if tab_name in ("train", "run", "debug"):
            self.app_state.right_panel_tab = tab_name
        self.game.set_board_background_mode(str(payload.get("boardBackgroundMode", self.game.board_background_mode)))
        self.app_state.snake_style = str(payload.get("snakeStyle", self.app_state.snake_style))
        self.game.set_snake_style(self.app_state.snake_style)
        self.app_state.fog_density = str(payload.get("fogDensity", self.app_state.fog_density))
        self.game.set_fog_density(self.app_state.fog_density)
        holdout_mode = str(payload.get("holdoutEvalMode", getattr(self, "_holdout_eval_mode", HoldoutEvalController.MODE_PPO_ONLY))).strip().lower()
        if holdout_mode in (HoldoutEvalController.MODE_PPO_ONLY, HoldoutEvalController.MODE_CONTROLLER_ON):
            self._holdout_eval_mode = holdout_mode
        self.settings.ticks_per_move = self._safe_int(
            payload.get("liveTicksPerMove"),
            default=self.settings.ticks_per_move,
            minimum=self._LIVE_TPM_MIN,
            maximum=self._LIVE_TPM_MAX,
        )

        width = self._safe_int(payload.get("windowWidth"), default=self.layout.window.width, minimum=1280, maximum=10000)
        height = self._safe_int(payload.get("windowHeight"), default=self.layout.window.height, minimum=720, maximum=10000)
        self._windowed_size = (width, height)
        self._is_fullscreen = False
        # Apply saved frame mode immediately, even if layout size is unchanged.
        self._recreate_window(target_size=self._windowed_size)
        self._resize(width, height)
        return True

    def _run_startup_self_checks(self) -> list[str]:
        warnings: list[str] = []
        for candidate in (self.state_file, self.ui_prefs_file):
            try:
                result = load_ui_state_result(candidate)
            except OSError:
                logger.exception("Startup self-check failed for %s", candidate)
                warnings.append(f"{candidate.name}: filesystem check failed")
                continue
            if result.error_code == UiStateErrorCode.FILESYSTEM:
                warnings.append(f"{candidate.name}: filesystem read failed")
            elif result.invalid:
                warnings.append(f"{candidate.name}: invalid/corrupted")
            elif result.error_code == UiStateErrorCode.PARTIAL_WRITE:
                warnings.append(f"{candidate.name}: recovered after interrupted write")
            if result.cleanup_warnings:
                warnings.append(f"{candidate.name}: cleanup warnings")
        model_path = getattr(self.agent, "model_path", None)
        if model_path is not None:
            try:
                _ = model_path.exists()
            except OSError:
                warnings.append("model path: filesystem probe failed")
        return warnings

    def _save_ui_preferences(self) -> None:
        if self._is_fullscreen:
            width = int(self._windowed_size[0])
            height = int(self._windowed_size[1])
        else:
            width = int(self.layout.window.width)
            height = int(self.layout.window.height)
        active_experiment = str(getattr(self, "experiment_name", "baseline") or "baseline")
        if bool(getattr(self, "_detached_mode", False)):
            active_experiment = "baseline"
        payload = {
            "uiPrefsVersion": 1,
            "activeExperiment": active_experiment,
            "themeName": self.theme.name,
            "windowBorderless": bool(self.settings.window_borderless),
            "windowWidth": width,
            "windowHeight": height,
            "debugOverlay": bool(self.app_state.debug_overlay),
            "debugReachableOverlay": bool(self.app_state.debug_reachable_overlay),
            "spaceStrategyEnabled": bool(self.app_state.space_strategy_enabled),
            "rightPanelTab": str(getattr(self.app_state, "right_panel_tab", "train")),
            "boardBackgroundMode": str(self.game.board_background_mode),
            "snakeStyle": str(self.game.snake_style),
            "fogDensity": str(self.game.fog_density),
            "liveTicksPerMove": int(self.settings.ticks_per_move),
            "holdoutEvalMode": str(getattr(self, "_holdout_eval_mode", HoldoutEvalController.MODE_PPO_ONLY)),
        }
        try:
            save_ui_state(self.ui_prefs_file, payload)
        except OSError:
            logger.exception("Failed to save UI preferences: %s", self.ui_prefs_file)

    @staticmethod
    def _set_toggle_button_visual(
        button: Button,
        *,
        label: str,
        enabled: bool,
        on_color: tuple[tuple[int, int, int], tuple[int, int, int]],
        off_color: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> None:
        button.label = f"{label}: {'ON' if enabled else 'OFF'}"
        active_palette = on_color if enabled else off_color
        button.bg = active_palette[0]
        button.bg_hover = active_palette[1]

    def _derive_ui_state(self) -> UIStateModel:
        snap = self.training.snapshot()
        if snap.active and snap.stop_requested:
            train_state = TrainingState.STOPPING
        elif snap.active:
            train_state = TrainingState.RUNNING
        elif bool(snap.last_error):
            train_state = TrainingState.ERROR
        elif self.app_state.last_train_message == "Training complete":
            train_state = TrainingState.COMPLETED
        else:
            train_state = TrainingState.IDLE

        is_inference_available = bool(getattr(self.agent, "is_inference_available", False))
        is_ready = bool(getattr(self.agent, "is_ready", False))
        is_sync_pending = bool(getattr(self.agent, "is_sync_pending", False))
        if is_inference_available:
            model_state = ModelState.READY
        elif is_ready and is_sync_pending:
            model_state = ModelState.SYNCING
        elif is_ready:
            model_state = ModelState.UNAVAILABLE
        else:
            model_state = ModelState.NONE
        return UIStateModel(
            model_state=model_state,
            training_state=train_state,
            game_running=bool(self.app_state.game_running),
        )

    def _derive_control_policy(self) -> ControlAuthorityPolicy:
        return derive_control_authority_policy(
            is_ready=bool(getattr(self.agent, "is_ready", False)),
            is_inference_available=bool(getattr(self.agent, "is_inference_available", False)),
            is_sync_pending=bool(getattr(self.agent, "is_sync_pending", False)),
            game_running=bool(self.app_state.game_running),
        )

    def _apply_ui_state_model(self) -> None:
        ui_state = self._derive_ui_state()
        button_map = {
            "train_start": self.btn_train_start,
            "train_stop": self.btn_train_stop,
            "save": self.btn_save,
            "load": self.btn_load,
            "delete": self.btn_delete,
        }
        for action, button in button_map.items():
            button.enabled = bool(ui_state.is_action_enabled(action))
        self.btn_game_start.enabled = not bool(self.app_state.game_running)
        self.btn_game_stop.enabled = bool(self.app_state.game_running)
        self.btn_restart.enabled = True
        holdout = getattr(self, "holdout_eval", None)
        holdout_active = bool(holdout.snapshot().active) if holdout is not None else False
        suite_active = bool(getattr(self, "_eval_suite_active", False))
        train_snap = self.training.snapshot()
        eval_enabled = not bool(holdout_active or suite_active or train_snap.active)
        if hasattr(self, "btn_eval_suite"):
            self.btn_eval_suite.enabled = bool(eval_enabled)
        if hasattr(self, "btn_eval_holdout"):
            self.btn_eval_holdout.enabled = bool(eval_enabled)
        control_policy = self._derive_control_policy()
        if control_policy.run_paused_waiting_snapshot:
            self.btn_game_start.label = "Start Wait (waiting)"
        elif control_policy.manual_can_steer:
            self.btn_game_start.label = "Start Manual"
        else:
            self.btn_game_start.label = "Start Agent"

    def _append_episode_score(self, score: int) -> None:
        self.app_state.training_episode_scores.append(int(score))
        limit = int(getattr(self.game, "EPISODE_HISTORY_LIMIT", 240))
        if len(self.app_state.training_episode_scores) > limit:
            self.app_state.training_episode_scores = self.app_state.training_episode_scores[-limit:]

    def _append_training_episode_info(self, info: dict) -> None:
        # `on_episode_info` is the most reliable end-of-episode signal; keep the
        # training graph alive even if a separate score callback is delayed/missed.
        score = info.get("score")
        if score is not None:
            death_total_before = sum(max(0, int(v)) for v in self.app_state.training_death_counts.values())
            score_window_len = len(self.app_state.training_episode_scores)
            window_limit = int(getattr(self.game, "EPISODE_HISTORY_LIMIT", 240))
            if score_window_len < window_limit and score_window_len <= int(death_total_before):
                self._append_episode_score(int(score))
        steps = info.get("steps")
        if steps is not None:
            self.app_state.training_episode_steps.append(int(steps))
            limit = int(getattr(self.game, "EPISODE_HISTORY_LIMIT", 240))
            if len(self.app_state.training_episode_steps) > limit:
                self.app_state.training_episode_steps = self.app_state.training_episode_steps[-limit:]
        reason = self._normalize_death_reason(str(info.get("death_reason", "other")))
        counts = self.app_state.training_death_counts
        if reason not in counts:
            counts[reason] = 0
        counts[reason] = int(counts.get(reason, 0) + 1)

    @staticmethod
    def _normalize_death_reason(reason: str) -> str:
        raw = str(reason or "").strip().lower()
        if raw in ("wall", "body", "starvation", "fill", "none"):
            return raw
        return "other"

    @staticmethod
    def _format_death_counts(counts: dict[str, int]) -> str:
        return (
            f"W{int(counts.get('wall', 0))} "
            f"B{int(counts.get('body', 0))} "
            f"S{int(counts.get('starvation', 0))} "
            f"F{int(counts.get('fill', 0))} "
            f"N{int(counts.get('none', 0))} "
            f"O{int(counts.get('other', 0))}"
        )

    @staticmethod
    def _training_episode_total(scores: list[int], death_counts: dict[str, int]) -> int:
        score_window = int(len(scores))
        death_total = sum(max(0, int(v)) for v in dict(death_counts).values())
        return max(score_window, int(death_total))

    @staticmethod
    def _compact_int(value: int) -> str:
        v = int(value)
        sign = "-" if v < 0 else ""
        n = abs(v)
        if n >= 1_000_000_000:
            return f"{sign}{(n / 1_000_000_000):.1f}b"
        if n >= 1_000_000:
            return f"{sign}{(n / 1_000_000):.1f}m"
        if n >= 1_000:
            return f"{sign}{(n / 1_000):.1f}k"
        return f"{sign}{n}"

    @staticmethod
    def _format_age_short(age_seconds: float | None) -> str:
        if age_seconds is None:
            return "n/a"
        age = max(0, int(age_seconds))
        if age < 60:
            return "<1m"
        mins = age // 60
        if mins < 60:
            return f"{mins}m"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h"
        days = hours // 24
        return f"{days}d"

    @staticmethod
    def _path_age_seconds(path: Path) -> float | None:
        try:
            if not path.exists():
                return None
            return float(max(0.0, time.time() - float(path.stat().st_mtime)))
        except OSError:
            return None

    def _checkpoint_age_seconds(self) -> float | None:
        state_file = getattr(self, "state_file", None)
        if state_file is None:
            return None
        checkpoint_dir = Path(state_file).parent / "ppo" / str(self.experiment_name) / "checkpoints"
        try:
            latest = max(
                (p for p in checkpoint_dir.glob("step_*_steps.zip") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except ValueError:
            return None
        except OSError:
            return None
        return self._path_age_seconds(latest)

    def _train_steps_per_sec(self, snap) -> float:
        now_s = float(time.perf_counter())
        current_done = max(0, int(snap.current_steps) - int(snap.start_steps))
        prev_time = float(getattr(self, "_train_rate_last_time_s", now_s))
        prev_done = int(getattr(self, "_train_rate_last_done_steps", current_done))
        ema = float(getattr(self, "_train_rate_ema_sps", 0.0))
        dt = float(max(0.0, now_s - prev_time))
        if dt >= 0.4:
            dsteps = max(0, int(current_done - prev_done))
            instant = (float(dsteps) / dt) if dt > 1e-6 else 0.0
            if instant > 0.0:
                ema = float(instant if ema <= 0.0 else (0.75 * ema + 0.25 * instant))
            self._train_rate_last_time_s = now_s
            self._train_rate_last_done_steps = int(current_done)
            self._train_rate_ema_sps = float(ema)
        return float(ema)

    def _runtime_health_lines(self, snap, telemetry: GameplayTelemetrySnapshot) -> list[str]:
        now_s = float(time.perf_counter())
        if now_s < float(getattr(self, "_runtime_health_next_refresh_s", 0.0)) and self._runtime_health_cached_lines:
            return list(self._runtime_health_cached_lines)

        sps = self._train_steps_per_sec(snap)
        lines: list[str] = []
        lines.append(f"Train SPS: {sps:.0f}" if sps > 0.0 else "Train SPS: n/a")
        if bool(snap.active) and sps > 0.0:
            remaining = max(0, int(snap.target_steps) - int(snap.done_steps))
            eta_seconds = float(remaining) / float(sps)
            lines.append(f"Train ETA: {self._format_age_short(eta_seconds)}")
        else:
            lines.append("Train ETA: n/a")

        selector = "best"
        get_selector = getattr(self.agent, "get_model_selector", None)
        if callable(get_selector):
            try:
                selector = str(get_selector() or "best").strip().lower()
            except Exception:
                selector = "best"
        inference_ready = bool(getattr(self.agent, "is_inference_available", False))
        model_ready = bool(getattr(self.agent, "is_ready", False))
        sync_pending = bool(getattr(self.agent, "is_sync_pending", False))
        model_state = "ready" if inference_ready else ("syncing" if (model_ready and sync_pending) else ("loaded/no-inf" if model_ready else "none"))
        lines.append(f"Model src: {selector} ({model_state})")
        lines.append(
            f"IntvN: {int(telemetry.interventions_total)}/{int(telemetry.decisions_total)}"
        )
        state_file = getattr(self, "state_file", None)
        state_path = Path(state_file) if state_file is not None else None
        metadata_age = (
            self._path_age_seconds(state_path.parent / "ppo" / str(self.experiment_name) / "metadata.json")
            if state_path is not None
            else None
        )
        checkpoint_age = self._checkpoint_age_seconds()
        state_age = self._path_age_seconds(state_path) if state_path is not None else None
        lines.append(
            f"Freshness eval/chk/ui: {self._format_age_short(metadata_age)}/{self._format_age_short(checkpoint_age)}/{self._format_age_short(state_age)}"
        )
        self._runtime_health_cached_lines = list(lines)
        self._runtime_health_next_refresh_s = float(now_s + 1.0)
        return lines

    def _build_training_graph_badges(self) -> list[str]:
        scores = [int(v) for v in self.app_state.training_episode_scores]
        snap = self.training.snapshot()
        avg20 = avg_last(scores, 20)
        avg100 = avg_last(scores, 100)
        best = int(max(scores)) if scores else 0
        last = int(scores[-1]) if scores else 0
        ofit = overfit_signal(scores)
        target = max(1, int(snap.target_steps))
        train_deaths = self.app_state.training_death_counts or empty_death_counts()
        episode_total = self._training_episode_total(scores, train_deaths)
        best_eval = snap.best_eval_score
        last_eval = snap.last_eval_score
        sps = self._train_steps_per_sec(snap)
        remaining = max(0, int(target) - int(snap.done_steps))
        return [
            f"Run {self._compact_int(snap.done_steps)}/{self._compact_int(target)}",
            f"Total {self._compact_int(snap.current_steps)}",
            f"SPS {sps:.0f}" if sps > 0.0 else "SPS n/a",
            f"ETA {self._format_age_short((float(remaining) / float(sps)) if sps > 0.0 else None)}" if bool(snap.active) else "ETA n/a",
            f"Avg20 {avg20:.1f}",
            f"Avg100 {avg100:.1f}",
            f"Best {best}",
            f"Last {last}",
            f"Eval {last_eval:.2f}" if last_eval is not None else "Eval n/a",
            f"BestEval {best_eval:.2f}" if best_eval is not None else "BestEval n/a",
            f"Eps {self._compact_int(episode_total)}",
            ofit.label,
        ]

    def _build_run_graph_badges(self) -> list[str]:
        run_scores = self._run_graph_scores()
        avg20 = avg_last(run_scores, 20)
        avg100 = avg_last(run_scores, 100)
        best = int(max(run_scores)) if run_scores else 0
        last = int(run_scores[-1]) if run_scores else 0
        live_score = int(getattr(self.game, "score", 0))
        telemetry: GameplayTelemetrySnapshot = self.gameplay.telemetry_snapshot()
        interventions_pct = (
            (100.0 * float(telemetry.interventions_total) / float(telemetry.decisions_total))
            if telemetry.decisions_total > 0
            else 0.0
        )
        return [
            f"RunEps {len(run_scores)}",
            f"Avg20 {avg20:.1f}",
            f"Avg100 {avg100:.1f}",
            f"Best {best}",
            f"Last {last}",
            f"Live {live_score}",
            f"Intv {interventions_pct:.1f}%",
            f"D {self._format_death_counts({'wall': telemetry.deaths_wall, 'body': telemetry.deaths_body, 'starvation': telemetry.deaths_starvation, 'fill': telemetry.deaths_fill, 'other': telemetry.deaths_other})}",
        ]

    def _run_graph_scores(self) -> list[int]:
        # Keep run chart/stats aligned to completed episodes only.
        return [int(v) for v in self.game.episode_scores]

    def _build_settings_lines(self) -> list[str]:
        cfg = getattr(self.agent, "config", None)
        reward_cfg = getattr(self.agent, "reward_config", None)
        exp_display = "New (not loaded)" if bool(getattr(self, "_detached_mode", False)) else str(self.experiment_name)
        lines = [
            f"Experiment: {exp_display}",
            f"Board: {self.settings.board_cells}x{self.settings.board_cells} cell={self.settings.cell_px} tpm={self.settings.ticks_per_move} fps={self.settings.fps}",
            f"Safety override: {'on' if self.settings.agent_safety_override else 'off'}",
            f"Space strategy: {'on' if self.gameplay.is_space_strategy_enabled() else 'off'}",
            f"Theme: {self.theme.name}",
            f"Board BG: {self.game.board_background_label()}",
            f"Snake style: {self.game.snake_style_label()}",
            f"Food: {self.game.food_label()}",
        ]
        if cfg is not None:
            lines.append(
                f"PPO: n_steps={int(getattr(cfg, 'n_steps', 0))} batch={int(getattr(cfg, 'batch_size', 0))} gamma={float(getattr(cfg, 'gamma', 0.0)):.3f}"
            )
        if reward_cfg is not None:
            adaptive_now = bool(
                getattr(self.agent, "is_adaptive_reward_enabled", lambda: getattr(reward_cfg, "use_reachable_space_penalty", False))()
            )
            lines.append(
                f"Trap penalty: {'on' if adaptive_now else 'off'} thr={float(getattr(reward_cfg, 'trap_penalty_threshold', 0.0)):.2f}"
            )
        return lines

    def _build_dynamic_status_lines(self) -> list[str]:
        snap = self.training.snapshot()
        telemetry: GameplayTelemetrySnapshot = self.gameplay.telemetry_snapshot()
        train_deaths = self.app_state.training_death_counts or empty_death_counts()
        run_deaths = {
            "wall": telemetry.deaths_wall,
            "body": telemetry.deaths_body,
            "starvation": telemetry.deaths_starvation,
            "fill": telemetry.deaths_fill,
            "other": telemetry.deaths_other,
        }
        starvation_progress = "n/a"
        if telemetry.starvation_limit > 0:
            pct = (100.0 * float(telemetry.starvation_steps) / float(max(1, telemetry.starvation_limit)))
            starvation_progress = f"{telemetry.starvation_steps}/{telemetry.starvation_limit} ({pct:.0f}%)"
        return [
            f"Mode: {telemetry.current_mode}",
            f"Last death: {telemetry.last_death_reason}",
            f"Ctrl switch: {telemetry.last_switch_reason}",
            f"No food: {telemetry.no_progress_steps} (move-steps)",
            f"TrainD: {self._format_death_counts(train_deaths)}",
            f"RunD: {self._format_death_counts(run_deaths)}",
            f"Cycle rpt: {telemetry.cycle_repeats_total} breaks:{telemetry.cycle_breaks_total}",
            f"LoopEsc: actv {telemetry.loop_escape_activations_total} left {telemetry.loop_escape_steps_left}",
            f"Loop stuck eps: {telemetry.stuck_episodes_total}",
            f"Starve: {starvation_progress}",
        ] + self._runtime_health_lines(snap, telemetry)

    def _apply_theme(self, theme_name: str, *, announce: bool) -> None:
        normalized = normalize_theme_name(theme_name)
        self.settings.theme_name = normalized
        self.theme = get_theme(normalized)
        compact = int(self.layout.window.height) < int(self.design_tokens.spacing.graph_margin_compact_threshold)
        self.design_tokens = get_design_tokens(normalized, compact=compact)
        self.game.theme = self.theme
        self.game._grid_cache = None
        self.game._food_sprite_cache = None
        self.game._board_background_cache = None
        self.panel_renderer.theme = self.theme
        self.panel_renderer.tokens = get_design_tokens(normalized, compact=compact)
        self.panel_renderer.graph.theme = self.theme
        self.panel_renderer.graph.tokens = get_design_tokens(normalized, compact=compact)
        self.panel_renderer.clear_caches()
        self._build_controls()
        self._bind_button_actions()
        if announce:
            self.actions.set_status(f"Theme set to {self.theme.name}")


def run(*, startup_route: str | None = None) -> bool:
    return bool(SnakeFrameApp(startup_route=startup_route).run())
