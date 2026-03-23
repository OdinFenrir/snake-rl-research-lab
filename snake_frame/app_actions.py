from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable
import re
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
except Exception:  # pragma: no cover - platform-dependent optional UI dependency
    tk = None
    filedialog = None
    simpledialog = None

from .protocols import AgentLike, GameLike, NumericInputLike, TrainingLike
from .app_state import AppState, empty_death_counts
from .diagnostics import DIAGNOSTICS_CLEANUP_FAILED, DiagnosticsBundleResult, create_diagnostics_bundle
from .state_io import (
    UiStateErrorCode,
    delete_ui_state_result,
    load_ui_state_result,
    migrate_ui_payload,
    save_ui_state_result,
)
from .theme import normalize_theme_name
from .ui_state_model import UIStateModel, derive_control_authority_policy

logger = logging.getLogger(__name__)


class AppActions:
    TRAIN_STEPS_MIN = 1_000
    TRAIN_STEPS_MAX = 100_000_000
    EPISODE_HISTORY_LIMIT = 240
    _STATUS_DURATIONS = {
        "info": 2.8,
        "warn": 4.0,
        "error": 8.0,
    }
    ERR_IO_INVALID_JSON = "io_invalid_json"
    ERR_IO_SCHEMA_UNSUPPORTED = "io_schema_unsupported"
    ERR_IO_PARTIAL_WRITE_RECOVERED = "io_partial_write_recovered"

    def __init__(
        self,
        *,
        app_state: AppState,
        game: GameLike,
        agent: AgentLike,
        training: TrainingLike,
        generations_input: NumericInputLike,
        state_file: Path,
        ui_state_provider: Callable[[], UIStateModel] | None = None,
        get_theme_name: Callable[[], str] | None = None,
        set_theme_name: Callable[[str], None] | None = None,
        get_experiment_name: Callable[[], str] | None = None,
        switch_experiment: Callable[[str], bool] | None = None,
    ) -> None:
        self.app_state = app_state
        self.game = game
        self.agent = agent
        self.training = training
        self.generations_input = generations_input
        self.state_file = state_file
        self.ui_state_provider = ui_state_provider
        self.get_theme_name = get_theme_name
        self.set_theme_name = set_theme_name
        self.get_experiment_name = get_experiment_name
        self.switch_experiment = switch_experiment
        has_model = bool(getattr(self.agent, "is_ready", False))
        self.app_state.model_dirty = False
        self.app_state.model_save_state = "saved" if has_model else "no_model"
        if not has_model:
            self.app_state.last_model_save_ok_at = 0.0

    def set_generations_input(self, generations_input: NumericInputLike) -> None:
        self.generations_input = generations_input

    def set_status(self, text: str, duration: float | None = None, severity: str = "info") -> None:
        sev = str(severity).lower()
        if sev not in self._STATUS_DURATIONS:
            sev = "info"
        effective_duration = (
            float(self._STATUS_DURATIONS[sev]) if duration is None else float(duration)
        )
        self.app_state.status_severity = sev
        self.app_state.status_text = str(text)
        self.app_state.status_until = time.perf_counter() + max(0.5, float(effective_duration))
        self.app_state.last_action_result = f"{sev}:{text}"

    def set_error(self, code: str, message: str, *, duration: float | None = None) -> None:
        code_text = str(code or "").strip().lower()
        msg = str(message)
        self.app_state.last_error_code = code_text
        self.app_state.last_error_message = msg
        self.app_state.last_error_at = time.time()
        self.set_status(msg, duration=duration, severity="error")

    @staticmethod
    def _stable_io_error_code(error_code: UiStateErrorCode) -> str:
        if error_code == UiStateErrorCode.UNSUPPORTED_SCHEMA:
            return AppActions.ERR_IO_SCHEMA_UNSUPPORTED
        if error_code == UiStateErrorCode.PARTIAL_WRITE:
            return AppActions.ERR_IO_PARTIAL_WRITE_RECOVERED
        if error_code == UiStateErrorCode.INVALID:
            return AppActions.ERR_IO_INVALID_JSON
        if error_code == UiStateErrorCode.FILESYSTEM:
            return "filesystem"
        return "unknown"

    def can_mutate_storage(self, action: str) -> bool:
        if self.ui_state_provider is not None:
            ui_state = self.ui_state_provider()
            if not bool(ui_state.is_action_enabled(action)):
                self.set_status(f"Cannot {action} while training is active")
                return False
        if self.training.snapshot().active:
            self.set_status(f"Cannot {action} while training is active")
            return False
        return True

    def on_train_start_clicked(self) -> None:
        target_steps = self.generations_input.as_int(self.TRAIN_STEPS_MIN, self.TRAIN_STEPS_MAX)
        started = self.training.start(target_steps=target_steps)
        if not started:
            self.set_status("Training already running")
            return
        self.app_state.model_dirty = True
        self.app_state.model_save_state = "pending"
        snap = self.training.snapshot()
        self.set_status(
            f"PPO training started (+{int(target_steps)} steps from total {int(snap.start_steps)})"
        )

    def on_train_stop_clicked(self) -> None:
        self.training.stop()
        self.set_status("Training stop requested")

    def on_game_start_clicked(self) -> None:
        self.app_state.game_running = True
        self.set_status("Game started")

    def on_game_stop_clicked(self) -> None:
        self.app_state.game_running = False
        self.set_status("Game stopped")

    def on_restart_clicked(self) -> None:
        self.agent.request_inference_sync()
        self.game.reset()
        self.app_state.game_running = True
        self.set_status("Run restarted")

    def on_adaptive_toggle_clicked(self) -> None:
        getter = getattr(self.agent, "is_adaptive_reward_enabled", None)
        setter = getattr(self.agent, "set_adaptive_reward_enabled", None)
        if not callable(getter) or not callable(setter):
            self.set_status("Adaptive reward toggle unavailable")
            return
        enabled = bool(getter())
        setter(not enabled)
        state = "ON" if not enabled else "OFF"
        self.set_status(f"Adaptive reward {state} (applies to next training steps)")

    def on_dropout_toggle_clicked(self) -> None:
        snap = self.training.snapshot()
        if snap.active:
            self.set_status("Stop training first to change mask setting")
            return
        self.app_state.dropout_enabled = not bool(self.app_state.dropout_enabled)
        agent = getattr(self, "agent", None)
        if agent is not None:
            from .settings import DropoutConfig
            agent.dropout_config = DropoutConfig(enabled=self.app_state.dropout_enabled)
        state = "ON" if self.app_state.dropout_enabled else "OFF"
        self.set_status(f"Full Mask {state} (applies to next training run)")

    def on_debug_toggle_clicked(self) -> None:
        self.app_state.debug_overlay = not bool(self.app_state.debug_overlay)
        state = "ON" if self.app_state.debug_overlay else "OFF"
        self.set_status(f"Debug overlay {state}")

    def on_reachable_toggle_clicked(self) -> None:
        self.app_state.debug_reachable_overlay = not bool(self.app_state.debug_reachable_overlay)
        state = "ON" if self.app_state.debug_reachable_overlay else "OFF"
        self.set_status(f"Reachable overlay {state}")

    def handle_diagnostics_clicked(self) -> None:
        output_dir = self.state_file.parent / "diagnostics"
        try:
            bundle_result = create_diagnostics_bundle(
                output_dir=output_dir,
                settings=getattr(self.agent, "settings", None),
                state_paths=[
                    self.state_file,
                    self.state_file.with_name("ui_prefs.json"),
                ],
                extra={
                    "model_path": str(getattr(self.agent, "model_path", "")),
                    "device": str(getattr(self.agent, "device", "unknown")),
                    "app_status": str(self.app_state.status_text),
                    "runtimeHealth": self.build_runtime_health_snapshot(),
                    "lastError": {
                        "code": str(self.app_state.last_error_code),
                        "message": str(self.app_state.last_error_message),
                        "at_epoch_s": float(self.app_state.last_error_at),
                    },
                },
            )
        except Exception:
            logger.exception("Failed to create diagnostics bundle")
            self.set_error("diagnostics_failed", "Failed to build diagnostics bundle")
            return
        bundle_path = bundle_result.bundle_path if isinstance(bundle_result, DiagnosticsBundleResult) else bundle_result
        warning_count = 0
        if isinstance(bundle_result, DiagnosticsBundleResult):
            warning_count = len(bundle_result.cleanup_warnings)
            if warning_count > 0:
                self.app_state.last_error_code = DIAGNOSTICS_CLEANUP_FAILED
                self.app_state.last_error_message = "; ".join(bundle_result.cleanup_warnings[:2])
                self.app_state.last_error_at = time.time()
        suffix = f" ({warning_count} cleanup warnings)" if warning_count > 0 else ""
        self.set_status(f"Diagnostics bundle created: {bundle_path.name}{suffix}", severity="warn" if warning_count > 0 else "info")

    def handle_save_clicked(self) -> None:
        if not self.can_mutate_storage("save"):
            return
        requested_experiment: str | None = None
        current_experiment: str | None = None
        if callable(self.switch_experiment):
            selected = self._choose_experiment_for_save()
            if selected is None:
                self.set_status("Save canceled", severity="warn")
                return
            requested_experiment = str(selected).strip()
            if callable(self.get_experiment_name):
                try:
                    current_experiment = str(self.get_experiment_name() or "").strip()
                except Exception:
                    current_experiment = None
        if self.ui_state_provider is not None:
            ui_state = self.ui_state_provider()
            model_state = str(getattr(ui_state.model_state, "value", ui_state.model_state))
            if model_state in ("none", "loading"):
                self.app_state.model_save_state = "no_model"
                self.app_state.model_dirty = False
                self.set_status("No trained/loaded model to save", severity="warn")
                return
        snap = self.training.snapshot()
        scores = (
            [int(v) for v in self.app_state.training_episode_scores]
            if self.app_state.training_episode_scores
            else [int(v) for v in self.game.episode_scores]
        )
        payload = {
            "episodeScores": scores,
            "trainingEpisodeScores": scores,
            "trainingEpisodeSteps": [int(v) for v in self.app_state.training_episode_steps],
            "runEpisodeScores": [int(v) for v in self.game.episode_scores],
            "trainingDeathCounts": self._sanitize_death_counts(self.app_state.training_death_counts),
            "lastTrainMessage": str(getattr(self.app_state, "last_train_message", "No training run yet")),
            "trainingTarget": int(snap.target_steps),
            "trainingCurrent": int(snap.current_steps),
            "gameRunning": bool(self.app_state.game_running),
            "spaceStrategyEnabled": bool(self.app_state.space_strategy_enabled),
            "tailTrendEnabled": bool(getattr(self.app_state, "tail_trend_enabled", True)),
            "dropoutEnabled": bool(self.app_state.dropout_enabled),
            "debugOverlay": bool(self.app_state.debug_overlay),
            "debugReachableOverlay": bool(self.app_state.debug_reachable_overlay),
            "rightPanelTab": str(getattr(self.app_state, "right_panel_tab", "train")),
            "snakeStyle": str(getattr(self.game, "snake_style", getattr(self.app_state, "snake_style", "topdown_3d"))),
            "fogDensity": str(getattr(self.game, "fog_density", getattr(self.app_state, "fog_density", "off"))),
            "savedAt": time.time(),
            "uiStateVersion": 2,
        }
        if callable(self.get_theme_name):
            payload["themeName"] = normalize_theme_name(self.get_theme_name())
        adaptive_getter = getattr(self.agent, "is_adaptive_reward_enabled", None)
        if callable(adaptive_getter):
            payload["adaptiveRewardEnabled"] = bool(adaptive_getter())
        try:
            save_result = save_ui_state_result(self.state_file, payload)
        except OSError:
            logger.exception("Failed to save UI state file: %s", self.state_file)
            self.set_error("filesystem", "Failed to save UI state (filesystem error)")
            return
        except Exception:
            logger.exception("Failed to save UI state file: %s", self.state_file)
            self.set_error("unknown", "Failed to save UI state (unknown error)")
            return
        if not save_result.ok:
            if save_result.error_code == UiStateErrorCode.PARTIAL_WRITE:
                msg = "Failed to save UI state (partial write recovered)"
            elif save_result.error_code == UiStateErrorCode.FILESYSTEM:
                msg = "Failed to save UI state (filesystem error)"
            else:
                msg = "Failed to save UI state"
            self.set_error(self._stable_io_error_code(save_result.error_code), msg)
            return
        model_detail = getattr(self.agent, "save_detailed", None)
        if callable(model_detail):
            model_result = model_detail()
            model_saved = bool(model_result.ok)
            model_code = str(getattr(model_result, "code", ""))
        else:
            model_saved = bool(getattr(self.agent, "save", lambda: False)())
            model_code = "unknown"
        status = f"Saved UI ({self.state_file.name})"
        if model_saved:
            status += " + PPO model"
            self.app_state.model_dirty = False
            self.app_state.model_save_state = "saved"
            self.app_state.last_model_save_ok_at = time.time()
        else:
            self.app_state.model_dirty = True
            self.app_state.model_save_state = "pending"
            if "filesystem" in model_code:
                status += " (model save filesystem error)"
            else:
                status += " (model save failed)"
        if save_result.cleanup_warnings:
            self.app_state.last_error_code = "cleanup_warning"
            self.app_state.last_error_message = "; ".join(save_result.cleanup_warnings[:2])
            self.app_state.last_error_at = time.time()
        if (
            model_saved
            and requested_experiment
            and current_experiment
            and requested_experiment != current_experiment
            and callable(self.switch_experiment)
        ):
            cloned = self._clone_experiment_artifacts(current_experiment, requested_experiment)
            if not cloned:
                self.set_status(
                    f"{status}; failed to copy artifacts to {requested_experiment}",
                    severity="warn",
                )
                return
            switched = self._switch_experiment_if_needed(requested_experiment)
            if not switched:
                self.set_status(
                    f"{status}; copied to {requested_experiment} but failed to switch",
                    severity="warn",
                )
                return
            runtime_loaded = False
            runtime_code = "missing"
            runtime_detail = ""
            runtime_loader = getattr(self.agent, "load_if_exists_detailed", None)
            if callable(runtime_loader):
                runtime_result = runtime_loader()
                runtime_loaded = bool(getattr(runtime_result, "ok", False))
                runtime_code = str(getattr(runtime_result, "code", runtime_code))
                runtime_detail = str(getattr(runtime_result, "detail", "") or "").strip()
            else:
                fallback_loader = getattr(self.agent, "load_if_exists", None)
                runtime_loaded = bool(fallback_loader()) if callable(fallback_loader) else False
                runtime_code = "ok" if runtime_loaded else "missing"
            if runtime_loaded:
                self.app_state.model_dirty = False
                self.app_state.model_save_state = "saved"
                self.app_state.last_model_save_ok_at = time.time()
                self.set_status(
                    f"{status}; copied to {requested_experiment} and loaded",
                    severity="info",
                )
            else:
                self.app_state.model_dirty = False
                self.app_state.model_save_state = "no_model"
                detail_suffix = f" ({runtime_detail})" if runtime_detail else ""
                self.set_status(
                    f"{status}; copied to {requested_experiment} but runtime load failed: {runtime_code}{detail_suffix}",
                    severity="warn",
                )
            return
        self.set_status(status, severity="info" if model_saved else "warn")

    def handle_load_clicked(self) -> None:
        if not self.can_mutate_storage("load"):
            return
        if callable(self.switch_experiment):
            selected = self._choose_existing_experiment(title="Select experiment folder to load")
            if selected is None:
                self.set_status("Load canceled", severity="warn")
                return
            if not self._switch_experiment_if_needed(selected):
                return
        try:
            state_result = load_ui_state_result(self.state_file)
        except OSError:
            logger.exception("Failed to load UI state file: %s", self.state_file)
            self.set_error("filesystem", "Failed to load UI state (filesystem error)")
            return
        except Exception:
            logger.exception("Failed to load UI state file: %s", self.state_file)
            self.set_error("unknown", "Failed to load UI state (unknown error)")
            return
        if state_result.error_code == UiStateErrorCode.FILESYSTEM:
            self.set_error("filesystem", "Failed to load UI state (filesystem error)")
            return
        migrated_result = migrate_ui_payload(state_result.payload or {})
        payload = migrated_result.payload or {}
        model_loaded = False
        model_code = "missing"
        model_source = "model"
        model_loader = getattr(self.agent, "load_if_exists_detailed", None)
        if callable(model_loader):
            model_result = model_loader()
            model_loaded = bool(model_result.ok)
            model_code = str(getattr(model_result, "code", ""))
            model_source = "model"
        else:
            model_loaded = bool(self.agent.load_if_exists())
            model_code = "ok" if model_loaded else "missing"
            model_source = "model"
        if not model_loaded:
            checkpoint_loader = getattr(self.agent, "load_latest_checkpoint_detailed", None)
            if callable(checkpoint_loader):
                checkpoint_result = checkpoint_loader()
                if bool(getattr(checkpoint_result, "ok", False)):
                    model_loaded = True
                    model_code = str(getattr(checkpoint_result, "code", "ok"))
                    model_source = "checkpoint"
                else:
                    model_code = str(getattr(checkpoint_result, "code", model_code))
        self.training.reset_tracking_from_agent()
        if migrated_result.error_code == UiStateErrorCode.UNSUPPORTED_SCHEMA:
            self.set_error(self.ERR_IO_SCHEMA_UNSUPPORTED, "Saved UI state schema is unsupported; please upgrade/downgrade safely")
            return
        if state_result.invalid:
            if model_loaded:
                self.set_status("Loaded PPO model (saved UI is invalid/corrupted and was ignored)")
                self.app_state.last_error_code = self._stable_io_error_code(state_result.error_code)
                self.app_state.last_error_message = "Saved UI state is invalid/corrupted and was ignored"
                self.app_state.last_error_at = time.time()
            else:
                self.set_status("Saved UI state is invalid/corrupted; nothing loaded", severity="warn")
                self.app_state.last_error_code = self._stable_io_error_code(state_result.error_code)
                self.app_state.last_error_message = "Saved UI state is invalid/corrupted"
                self.app_state.last_error_at = time.time()
            return
        if state_result.error_code == UiStateErrorCode.PARTIAL_WRITE:
            self.app_state.last_error_code = self.ERR_IO_PARTIAL_WRITE_RECOVERED
            self.app_state.last_error_message = "Recovered UI state after interrupted write"
            self.app_state.last_error_at = time.time()
        loaded_training_scores = self._sanitize_episode_scores(
            payload.get("trainingEpisodeScores", payload.get("episodeScores"))
        )
        loaded_run_scores = self._sanitize_episode_scores(
            payload.get("runEpisodeScores", payload.get("episodeScores"))
        )
        loaded_training_steps = self._sanitize_episode_scores(payload.get("trainingEpisodeSteps"))
        self.app_state.training_episode_scores = list(loaded_training_scores)
        self.app_state.training_episode_steps = list(loaded_training_steps)
        self.app_state.training_death_counts = self._sanitize_death_counts(payload.get("trainingDeathCounts"))
        loaded_last_train_message = str(payload.get("lastTrainMessage", "") or "").strip()
        if loaded_last_train_message:
            self.app_state.last_train_message = loaded_last_train_message
        else:
            self.app_state.last_train_message = "No training run yet"
        self.game.episode_scores = list(loaded_run_scores)
        self._hydrate_training_debug_from_trace_if_missing(payload)
        self.app_state.ui_state_version = self._sanitize_int(
            payload.get("uiStateVersion"),
            default=2,
            minimum=1,
            maximum=100,
        )
        self.app_state.game_running = bool(payload.get("gameRunning", False))
        self.app_state.space_strategy_enabled = bool(
            payload.get("spaceStrategyEnabled", self.app_state.space_strategy_enabled)
        )
        self.app_state.tail_trend_enabled = bool(
            payload.get("tailTrendEnabled", getattr(self.app_state, "tail_trend_enabled", True))
        )
        self.app_state.dropout_enabled = bool(
            payload.get("dropoutEnabled", self.app_state.dropout_enabled)
        )
        self.app_state.debug_overlay = bool(payload.get("debugOverlay", self.app_state.debug_overlay))
        self.app_state.debug_reachable_overlay = bool(
            payload.get("debugReachableOverlay", self.app_state.debug_reachable_overlay)
        )
        tab_name = str(payload.get("rightPanelTab", getattr(self.app_state, "right_panel_tab", "train"))).strip().lower()
        if tab_name in ("train", "run", "debug"):
            self.app_state.right_panel_tab = tab_name
        self.app_state.snake_style = str(payload.get("snakeStyle", getattr(self.app_state, "snake_style", "topdown_3d")))
        style_setter = getattr(self.game, "set_snake_style", None)
        if callable(style_setter):
            style_setter(self.app_state.snake_style)
        self.app_state.fog_density = str(payload.get("fogDensity", getattr(self.app_state, "fog_density", "off")))
        fog_setter = getattr(self.game, "set_fog_density", None)
        if callable(fog_setter):
            fog_setter(self.app_state.fog_density)
        if callable(self.set_theme_name) and "themeName" in payload:
            self.set_theme_name(normalize_theme_name(str(payload.get("themeName"))))
        adaptive_setter = getattr(self.agent, "set_adaptive_reward_enabled", None)
        if callable(adaptive_setter) and "adaptiveRewardEnabled" in payload:
            adaptive_setter(bool(payload.get("adaptiveRewardEnabled")))
        default_target = self.generations_input.as_int(self.TRAIN_STEPS_MIN, self.TRAIN_STEPS_MAX)
        target_steps = self._sanitize_int(
            payload.get("trainingTarget"),
            default=default_target,
            minimum=1,
            maximum=self.TRAIN_STEPS_MAX,
        )
        self.generations_input.value = str(target_steps)
        if model_loaded or payload:
            parts: list[str] = []
            if payload:
                parts.append("UI")
            if model_loaded:
                parts.append("PPO checkpoint" if model_source == "checkpoint" else "PPO model")
            elif "legacy_format_unsupported" in model_code or "incompatible" in model_code:
                parts.append("PPO model failed (legacy unsupported)")
            elif "corrupt" in model_code:
                parts.append("PPO model failed (corrupt)")
            elif "filesystem" in model_code:
                parts.append("PPO model failed (filesystem)")
            self.set_status("Loaded " + " + ".join(parts))
            if model_loaded:
                self.app_state.model_dirty = False
                self.app_state.model_save_state = "saved"
            return
        if "legacy_format_unsupported" in model_code or "incompatible" in model_code:
            self.set_status("Saved model uses deprecated legacy format; retrain with baseline artifacts", severity="warn")
            return
        self.app_state.model_save_state = "no_model"
        self.app_state.model_dirty = False
        self.set_status("No saved UI/model to load", severity="warn")

    def _current_experiment_name(self) -> str:
        if not callable(self.get_experiment_name):
            return ""
        try:
            return str(self.get_experiment_name() or "").strip()
        except Exception:
            return ""

    def _latest_training_trace_summary(self) -> dict[str, Any] | None:
        exp = self._current_experiment_name()
        if not exp:
            return None
        trace_path = self.state_file.parent / "ppo" / exp / "training_trace.jsonl"
        if not trace_path.exists():
            return None
        last_obj: dict[str, Any] | None = None
        try:
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                text = str(line).strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    last_obj = obj
        except OSError:
            return None
        return last_obj

    def _hydrate_training_debug_from_trace_if_missing(self, payload: dict[str, Any]) -> None:
        have_last_train = bool(str(payload.get("lastTrainMessage", "") or "").strip())
        deaths = dict(getattr(self.app_state, "training_death_counts", {}) or {})
        have_deaths = any(int(v) > 0 for v in deaths.values())
        if have_last_train and have_deaths:
            return
        summary = self._latest_training_trace_summary()
        if not isinstance(summary, dict):
            return
        if not have_last_train:
            episodes_total = int(summary.get("episodes_total", 0) or 0)
            if episodes_total > 0:
                self.app_state.last_train_message = f"Training complete ({episodes_total} eps)"
        if not have_deaths:
            death_obj = summary.get("deaths")
            if isinstance(death_obj, dict):
                self.app_state.training_death_counts = self._sanitize_death_counts(death_obj)

    def handle_load_latest_checkpoint_clicked(self) -> None:
        if not self.can_mutate_storage("load"):
            return
        model_loader = getattr(self.agent, "load_latest_checkpoint_detailed", None)
        if callable(model_loader):
            model_result = model_loader()
            model_loaded = bool(model_result.ok)
            model_code = str(getattr(model_result, "code", ""))
            model_detail = str(getattr(model_result, "detail", "")).strip()
        else:
            model_loaded = bool(getattr(self.agent, "load_if_exists", lambda: False)())
            model_code = "ok" if model_loaded else "missing"
            model_detail = ""
        if not model_loaded:
            if "missing" in model_code:
                self.set_status("No checkpoint artifacts available to load", severity="warn")
            elif "filesystem" in model_code:
                self.set_status("Checkpoint load failed (filesystem)", severity="error")
            elif "corrupt" in model_code:
                self.set_status("Checkpoint load failed (corrupt artifacts)", severity="error")
            else:
                self.set_status("Checkpoint load failed", severity="warn")
            return
        self.training.reset_tracking_from_agent()
        self.app_state.model_dirty = False
        self.app_state.model_save_state = "saved"
        if model_detail:
            self.set_status(f"Loaded latest checkpoint ({model_detail})")
            return
        self.set_status("Loaded latest checkpoint")

    def handle_delete_clicked(self) -> None:
        if not self.can_mutate_storage("delete"):
            return
        if callable(self.switch_experiment):
            selected = self._choose_existing_experiment(title="Select experiment folder to delete")
            if selected is None:
                self.set_status("Delete canceled", severity="warn")
                return
            if not self._switch_experiment_if_needed(selected):
                return
        delete_result = None
        removed_ui = False
        ui_error = False
        try:
            delete_result = delete_ui_state_result(self.state_file)
        except OSError:
            logger.exception("Failed to delete UI state file: %s", self.state_file)
            ui_error = True
        except Exception:
            logger.exception("Failed to delete UI state file: %s", self.state_file)
            ui_error = True
        if delete_result is not None:
            removed_ui = bool(delete_result.removed)
            ui_error = delete_result.error_code == UiStateErrorCode.FILESYSTEM
        model_delete = getattr(self.agent, "delete_detailed", None)
        if callable(model_delete):
            model_result = model_delete()
            removed_model = bool(model_result.ok)
            model_error = "filesystem" in str(getattr(model_result, "code", ""))
        else:
            removed_model = bool(self.agent.delete())
            model_error = False
        if ui_error or model_error:
            self.set_error("filesystem", "Failed to delete some saved files (filesystem error)")
            return
        if removed_ui or removed_model:
            if removed_model:
                self.app_state.model_dirty = False
                self.app_state.model_save_state = "no_model"
                self.app_state.last_model_save_ok_at = 0.0
            self.set_status("Deleted saved state/model")
            return
        self.set_status("No state/model to delete", severity="warn")

    def poll_training_state(self) -> None:
        message = self.training.poll_completion()
        if not message:
            return
        self.app_state.last_train_message = message
        if message.startswith("Training error:"):
            self.set_error("training_error", message, duration=20.0)
            return
        self.set_status(message)

    def build_status_lines(self) -> list[str]:
        snap = self.training.snapshot()
        experiment_name = "New (not loaded)"
        if callable(self.get_experiment_name):
            try:
                current_name = str(self.get_experiment_name() or "").strip()
            except Exception:
                current_name = ""
            if current_name and not current_name.startswith("_"):
                experiment_name = current_name
        if snap.active:
            last_train = "running"
        else:
            raw_last_train = str(self.app_state.last_train_message or "").strip().lower()
            if raw_last_train in ("", "no training run yet"):
                last_train = "none yet"
            elif raw_last_train.startswith("training complete"):
                last_train = "complete"
            elif raw_last_train.startswith("training error"):
                last_train = "error"
            elif len(raw_last_train) > 18:
                last_train = raw_last_train[:15].rstrip() + "..."
            else:
                last_train = raw_last_train or "idle"
        inference_available = bool(getattr(self.agent, "is_inference_available", self.agent.is_ready))
        control_policy = derive_control_authority_policy(
            is_ready=bool(getattr(self.agent, "is_ready", False)),
            is_inference_available=bool(inference_available),
            is_sync_pending=bool(getattr(self.agent, "is_sync_pending", False)),
            game_running=bool(self.app_state.game_running),
        )
        control_label = "agent"
        if control_policy.run_paused_waiting_snapshot:
            control_label = "paused (loading snapshot)"
        elif control_policy.manual_can_steer:
            control_label = "manual"
        lines = [
            f"Experiment: {experiment_name}",
            f"Algo: PPO (device={self.agent.device})",
            f"Model: {'ready' if inference_available else ('loaded/no-inference' if self.agent.is_ready else 'none')}",
            f"Saved: {self._build_model_save_report()}",
            f"Last train: {last_train}",
            f"Game: {'running' if self.app_state.game_running else 'stopped'}",
            f"Control: {control_label}",
        ]
        health = self.build_runtime_health_snapshot()
        last_token = str(health["last_action_result"])
        if len(last_token) > 44:
            last_token = last_token[:41].rstrip() + "..."
        lines.append(
            "Health: "
            f"model={health['model_state']} "
            f"train={health['training_state']} "
            f"game={health['game_state']} "
            f"last={last_token}"
        )
        if self.app_state.last_error_code:
            lines.append(f"Last error: {self.app_state.last_error_code} ({self.app_state.last_error_message})")
        if time.perf_counter() < self.app_state.status_until:
            lines.append(f"[{self.app_state.status_severity}] {self.app_state.status_text}")
        return lines

    def _build_model_save_report(self) -> str:
        save_state = str(getattr(self.app_state, "model_save_state", "unknown")).strip().lower()
        has_model = bool(getattr(self.agent, "is_ready", False))
        if not has_model or save_state == "no_model":
            return "no model on disk"
        if bool(getattr(self.app_state, "model_dirty", False)) or save_state == "pending":
            return "unsaved changes"
        saved_at = float(getattr(self.app_state, "last_model_save_ok_at", 0.0))
        if saved_at > 0.0:
            age_s = max(0, int(time.time() - saved_at))
            if age_s < 60:
                return "saved (<1m ago)"
            mins = age_s // 60
            if mins < 60:
                return f"saved ({mins}m ago)"
            hours = mins // 60
            return f"saved ({hours}h ago)"
        return "saved"

    def build_runtime_health_snapshot(self) -> dict[str, str]:
        ui_state = self.ui_state_provider() if callable(self.ui_state_provider) else None
        if ui_state is not None:
            model_state = str(getattr(ui_state.model_state, "value", ui_state.model_state))
            training_state = str(getattr(ui_state.training_state, "value", ui_state.training_state))
            game_state = "running" if bool(getattr(ui_state, "game_running", False)) else "stopped"
        else:
            model_state = "ready" if bool(getattr(self.agent, "is_inference_available", False)) else (
                "unavailable" if bool(getattr(self.agent, "is_ready", False)) else "none"
            )
            training_state = "running" if bool(self.training.snapshot().active) else "idle"
            game_state = "running" if bool(self.app_state.game_running) else "stopped"
        starvation_steps = int(getattr(self.game, "steps_without_food", 0))
        starvation_limit = 0
        starvation_limit_fn = getattr(self.game, "starvation_limit", None)
        if callable(starvation_limit_fn):
            starvation_limit = int(starvation_limit_fn())
        return {
            "model_state": model_state,
            "training_state": training_state,
            "game_state": game_state,
            "last_action_result": str(self.app_state.last_action_result),
            "last_error_code": str(self.app_state.last_error_code),
            "steps_without_food": str(starvation_steps),
            "starvation_limit": str(starvation_limit),
        }

    @staticmethod
    def _sanitize_episode_scores(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        scores: list[int] = []
        for item in value:
            try:
                scores.append(int(item))
            except (TypeError, ValueError):
                continue
        return scores[-int(AppActions.EPISODE_HISTORY_LIMIT) :]

    @staticmethod
    def _sanitize_death_counts(value: Any) -> dict[str, int]:
        base = empty_death_counts()
        if not isinstance(value, dict):
            return base
        for key in base:
            try:
                parsed = int(value.get(key, 0))
            except (TypeError, ValueError):
                parsed = 0
            base[key] = max(0, int(parsed))
        return base

    @staticmethod
    def _sanitize_int(value: Any, *, default: int, minimum: int = 1, maximum: int | None = None) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        lower_bounded = max(int(minimum), int(parsed))
        if maximum is None:
            return lower_bounded
        return min(int(maximum), lower_bounded)

    def _state_root(self) -> Path:
        return Path(self.state_file).parent / "ppo"

    @staticmethod
    def _sanitize_experiment_name(name: str) -> str:
        raw = str(name or "").strip()
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
        cleaned = cleaned.strip("._- ")
        return cleaned

    def _choose_experiment_for_save(self) -> str | None:
        current = "baseline"
        if callable(self.get_experiment_name):
            try:
                current = str(self.get_experiment_name() or "baseline")
            except Exception:
                current = "baseline"
        if tk is None or simpledialog is None:
            return current
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            value = simpledialog.askstring(
                title="Save Model",
                prompt="Experiment name to save into:",
                initialvalue=current,
                parent=root,
            )
            root.destroy()
        except Exception:
            logger.exception("Experiment save dialog failed")
            return current
        if value is None:
            return None
        selected = self._sanitize_experiment_name(value)
        if not selected:
            self.set_status("Invalid experiment name", severity="warn")
            return None
        if selected.startswith("_"):
            self.set_status("Experiment names starting with '_' are reserved", severity="warn")
            return None
        return selected

    def _choose_existing_experiment(self, *, title: str) -> str | None:
        if tk is None or filedialog is None:
            self.set_status("Folder picker unavailable on this platform", severity="warn")
            return None
        try:
            base = self._state_root()
            base.mkdir(parents=True, exist_ok=True)
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected_dir = filedialog.askdirectory(
                title=title,
                initialdir=str(base),
                mustexist=True,
                parent=root,
            )
            root.destroy()
        except Exception:
            logger.exception("Experiment folder dialog failed")
            return None
        if not selected_dir:
            return None
        selected_path = Path(selected_dir)
        try:
            is_under_root = selected_path.parent.resolve() == base.resolve()
        except OSError:
            is_under_root = False
        if not is_under_root:
            self.set_status("Select a folder directly inside state/ppo", severity="warn")
            return None
        name = self._sanitize_experiment_name(selected_path.name)
        if not name:
            self.set_status("Invalid experiment folder", severity="warn")
            return None
        if name.startswith("_"):
            self.set_status("Internal folders cannot be selected", severity="warn")
            return None
        return name

    def _switch_experiment_if_needed(self, target: str) -> bool:
        desired = self._sanitize_experiment_name(target)
        if not desired:
            self.set_status("Invalid experiment target", severity="warn")
            return False
        current = None
        if callable(self.get_experiment_name):
            try:
                current = str(self.get_experiment_name() or "").strip()
            except Exception:
                current = None
        if current == desired:
            return True
        if not callable(self.switch_experiment):
            self.set_status("Experiment switching is unavailable", severity="warn")
            return False
        ok = False
        try:
            ok = bool(self.switch_experiment(desired))
        except Exception:
            logger.exception("Experiment switch failed")
            ok = False
        if not ok:
            self.set_status(f"Failed to switch experiment: {desired}", severity="warn")
            return False
        self.set_status(f"Experiment: {desired}")
        return True

    def _clone_experiment_artifacts(self, source_experiment: str, target_experiment: str) -> bool:
        source = self._state_root() / str(source_experiment)
        target = self._state_root() / str(target_experiment)
        if source.resolve() == target.resolve():
            return True
        if not source.exists():
            self.set_status(f"Source experiment not found: {source_experiment}", severity="warn")
            return False
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            metadata_path = target / "metadata.json"
            if metadata_path.exists():
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload["artifact_dir"] = str(target)
                    payload["experiment_name"] = str(target_experiment)
                    metadata_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            return True
        except OSError:
            logger.exception(
                "Failed cloning experiment artifacts from %s to %s",
                source,
                target,
            )
            return False
        except Exception:
            logger.exception(
                "Failed rewriting cloned metadata from %s to %s",
                source,
                target,
            )
            return False
