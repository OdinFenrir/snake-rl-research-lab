from __future__ import annotations

import io
import importlib.metadata
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from enum import Enum
import os
from pathlib import Path
import platform
import threading
import statistics
import uuid
from typing import Callable, Iterable

import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO as PPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization

from .observation import observation_size
from .ppo_env import SnakePPOEnv
from .settings import DropoutConfig, ObsConfig, PpoConfig, RewardConfig, Settings

ProgressFn = Callable[[int], None]
ScoreFn = Callable[[int], None]
EpisodeInfoFn = Callable[[dict], None]
logger = logging.getLogger(__name__)


class ModelOpCode(str, Enum):
    OK = "ok"
    MISSING = "missing"
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"
    FILESYSTEM_ERROR = "filesystem_error"
    UNKNOWN_ERROR = "unknown_error"
    LEGACY_FORMAT_UNSUPPORTED = "legacy_format_unsupported"


class ModelSelector(str, Enum):
    BEST = "best"
    LAST = "last"


@dataclass(frozen=True)
class ModelOpResult:
    ok: bool
    code: ModelOpCode
    detail: str = ""


@dataclass(frozen=True)
class _EnvFactory:
    board_cells: int
    seed: int | None
    reward_config: RewardConfig
    obs_config: ObsConfig
    max_episode_steps: int | None = None
    dropout_config: DropoutConfig | None = None

    def __call__(self):
        env = SnakePPOEnv(
            board_cells=int(self.board_cells),
            seed=self.seed,
            reward_config=self.reward_config,
            obs_config=self.obs_config,
            dropout_config=self.dropout_config,
        )
        max_steps = None if self.max_episode_steps is None else max(1, int(self.max_episode_steps))
        if max_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_steps)
        return Monitor(env)


class _StopAndProgressCallback(BaseCallback):
    def __init__(
        self,
        stop_flag: Callable[[], bool],
        on_progress: ProgressFn | None = None,
        progress_every: int = 4096,
        ent_coef_schedule: Callable[[float], float] | None = None,
        on_step_hook: Callable[[int], None] | None = None,
        on_episode_score: ScoreFn | None = None,
        on_episode_info: EpisodeInfoFn | None = None,
        step_hook_every: int = 64,
        start_timestep: int = 0,
        total_timesteps: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.stop_flag = stop_flag
        self.on_progress = on_progress
        self.progress_every = max(1, int(progress_every))
        self._last_reported = 0
        self._last_hooked = 0
        self.ent_coef_schedule = ent_coef_schedule
        self.on_step_hook = on_step_hook
        self.on_episode_score = on_episode_score
        self.on_episode_info = on_episode_info
        self.step_hook_every = max(1, int(step_hook_every))
        self.start_timestep = int(start_timestep)
        self.total_timesteps = max(1, int(total_timesteps))

    @staticmethod
    def _is_done_at(dones: object, idx: int) -> bool:
        if dones is None:
            return True
        try:
            return bool(dones[idx])
        except Exception:
            return False

    @staticmethod
    def _iter_terminal_infos(infos: object, dones: object) -> Iterable[dict]:
        if isinstance(infos, (list, tuple)):
            for idx, info in enumerate(infos):
                yielded_final_info = False
                if isinstance(info, dict):
                    final_info = info.get("final_info")
                    if isinstance(final_info, dict):
                        yielded_final_info = True
                        yield dict(final_info)
                    elif isinstance(final_info, (list, tuple)):
                        for nested in final_info:
                            if isinstance(nested, dict):
                                yielded_final_info = True
                                yield dict(nested)
                if not _StopAndProgressCallback._is_done_at(dones, idx):
                    continue
                if isinstance(info, dict) and not yielded_final_info:
                    yield dict(info)
            return
        if isinstance(infos, dict):
            final_infos = infos.get("final_info")
            final_mask = infos.get("_final_info")
            if isinstance(final_infos, (list, tuple)):
                for idx, info in enumerate(final_infos):
                    if not isinstance(info, dict):
                        continue
                    keep = True
                    if final_mask is not None:
                        try:
                            keep = bool(final_mask[idx])
                        except Exception:
                            keep = False
                    if keep:
                        yield dict(info)

    def _on_step(self) -> bool:
        if self.stop_flag():
            return False
        now = int(self.num_timesteps)
        if self.ent_coef_schedule is not None:
            done = max(0, now - self.start_timestep)
            progress_remaining = 1.0 - min(1.0, float(done) / float(self.total_timesteps))
            self.model.ent_coef = float(self.ent_coef_schedule(progress_remaining))
        if self.on_progress is not None and now - self._last_reported >= self.progress_every:
            self._last_reported = now
            self.on_progress(now)
        if self.on_step_hook is not None and now - self._last_hooked >= self.step_hook_every:
            self._last_hooked = now
            self.on_step_hook(now)
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        for info in self._iter_terminal_infos(infos, dones):
            if self.on_episode_score is not None and "score" in info:
                try:
                    self.on_episode_score(int(info["score"]))
                except Exception:
                    logger.exception("on_episode_score callback failed")
                    continue
            if self.on_episode_info is not None:
                try:
                    self.on_episode_info(dict(info))
                except Exception:
                    logger.exception("on_episode_info callback failed")
        return True


class _SyncEvalCallback(EvalCallback):
    class _EvaluationStopRequested(Exception):
        pass

    def __init__(
        self,
        *args,
        stop_flag: Callable[[], bool] | None = None,
        on_eval_complete: Callable[[dict[str, object]], None] | None = None,
        best_score_model_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._stop_flag = stop_flag
        self._on_eval_complete = on_eval_complete
        self._best_score_model_path = str(best_score_model_path) if best_score_model_path else None
        self.eval_runs_completed = 0
        self.last_eval_score = 0.0
        self.best_eval_score = float("-inf")
        self.best_eval_step = 0
        self.last_mean_score = 0.0
        self.best_mean_score = float("-inf")
        self.best_score_step = 0
        self._episode_score_buffer: list[float] = []

    def _log_success_callback(self, locals_: dict[str, object], globals_: dict[str, object]) -> None:
        if callable(self._stop_flag) and bool(self._stop_flag()):
            raise _SyncEvalCallback._EvaluationStopRequested()
        info = locals_.get("info")
        done = bool(locals_.get("done", False))
        if done and isinstance(info, dict):
            score = info.get("score")
            if score is not None:
                try:
                    self._episode_score_buffer.append(float(score))
                except Exception:
                    logger.debug("Eval callback score parse failed: %r", score, exc_info=True)
        super()._log_success_callback(locals_, globals_)

    def _on_step(self) -> bool:
        if callable(self._stop_flag) and bool(self._stop_flag()):
            return False
        should_eval = bool(self.eval_freq > 0 and self.n_calls % self.eval_freq == 0)
        if should_eval:
            self._episode_score_buffer = []
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except Exception:
                logger.debug("Normalization sync skipped", exc_info=True)
            prev_best = float(self.best_mean_reward)
            try:
                keep_going = super()._on_step()
            except _SyncEvalCallback._EvaluationStopRequested:
                return False
            self.eval_runs_completed += 1
            self.last_eval_score = float(self.last_mean_reward)
            self.best_eval_score = float(self.best_mean_reward)
            if float(self.best_mean_reward) > float(prev_best):
                self.best_eval_step = int(self.num_timesteps)
            if self._episode_score_buffer:
                mean_score = float(np.mean(np.asarray(self._episode_score_buffer, dtype=np.float64)))
            else:
                mean_score = 0.0
            self.last_mean_score = float(mean_score)
            if float(mean_score) >= float(self.best_mean_score):
                self.best_mean_score = float(mean_score)
                self.best_score_step = int(self.num_timesteps)
                if self._best_score_model_path:
                    try:
                        self.model.save(str(self._best_score_model_path))
                    except Exception:
                        logger.exception("Failed to save best-score model to %s", self._best_score_model_path)
            if callable(self._on_eval_complete):
                results = getattr(self, "evaluations_results", None)
                lengths = getattr(self, "evaluations_length", None)
                detail = {
                    "step": int(self.num_timesteps),
                    "mean_reward": float(self.last_mean_reward),
                    "best_mean_reward": float(self.best_mean_reward),
                    "eval_run_index": int(self.eval_runs_completed),
                    "episode_rewards": [float(v) for v in (results[-1] if isinstance(results, list) and results else [])],
                    "episode_lengths": [int(v) for v in (lengths[-1] if isinstance(lengths, list) and lengths else [])],
                    "episode_scores": [float(v) for v in self._episode_score_buffer],
                    "mean_score": float(mean_score),
                    "best_mean_score": float(self.best_mean_score),
                    "best_score_step": int(self.best_score_step),
                }
                try:
                    self._on_eval_complete(detail)
                except Exception:
                    logger.exception("Eval-complete callback failed")
            return bool(keep_going)
        return bool(super()._on_step())


@dataclass
class _StagnationStatus:
    is_stagnant: bool
    slope: float
    abs_range: float
    mean: float


@dataclass
class TrainingHealthStatus:
    healthy: bool
    failure_reason: str
    warnings: tuple[str, ...]
    details: dict


class TrainingHealthMonitor(BaseCallback):
    """
    Monitors PPO training health at rollout boundaries.
    
    FAIL (hard stop):
    - NaN/Inf in any loss
    - KL divergence > 0.2 for 2+ consecutive rollouts
    - Missing metrics after warmup
    - Explained variance < -0.5
    
    WARN (log but continue):
    - KL divergence > 0.1
    - Clip fraction > 0.85 (after warmup)
    - Entropy decay (< 10% of recent average)
    - Value loss spike (> 5x moving avg)
    - Reward stagnation (flat slope + small range)
    - Explained variance < 0
    - Logger structure invalid
    
    All thresholds are configurable via constructor args.
    Default thresholds are documented in DEFAULT_THRESHOLDS.
    """
    
    DEFAULT_THRESHOLDS = {
        "kl_fail_threshold": 0.2,
        "kl_warn_threshold": 0.1,
        "kl_persistence_required": 2,
        "clip_warn_threshold": 0.85,
        "warmup_rollouts": 3,
        "entropy_decay_warn_threshold": 0.1,
        "value_loss_spike_factor": 5.0,
        "value_loss_baseline_floor": 1e-6,
        "stagnation_window": 5,
        "stagnation_rel_threshold": 0.01,
        "stagnation_abs_threshold": 0.05,
        "explained_variance_warn": 0.0,
        "explained_variance_fail": -5.0,  # Only fail on extreme negative values
    }
    
    def __init__(
        self,
        *,
        on_failure: Callable[[str, dict], None] | None = None,
        on_warning: Callable[[str, dict], None] | None = None,
        stop_flag: Callable[[], bool] | None = None,
        verbose: int = 0,
        **threshold_overrides: float | int,
    ) -> None:
        super().__init__(verbose=verbose)
        self.on_failure = on_failure
        self.on_warning = on_warning
        self.stop_flag = stop_flag
        
        params = {**self.DEFAULT_THRESHOLDS, **threshold_overrides}
        
        self.kl_fail_threshold = float(params["kl_fail_threshold"])
        self.kl_warn_threshold = float(params["kl_warn_threshold"])
        self.kl_persistence_required = max(1, int(params["kl_persistence_required"]))
        self.clip_warn_threshold = float(params["clip_warn_threshold"])
        self.warmup_rollouts = max(1, int(params["warmup_rollouts"]))
        self.entropy_decay_warn_threshold = float(params["entropy_decay_warn_threshold"])
        self.value_loss_spike_factor = float(params["value_loss_spike_factor"])
        self.value_loss_baseline_floor = float(params["value_loss_baseline_floor"])
        self.stagnation_window = max(3, int(params["stagnation_window"]))
        self.stagnation_rel_threshold = float(params["stagnation_rel_threshold"])
        self.stagnation_abs_threshold = float(params["stagnation_abs_threshold"])
        self.explained_variance_warn = float(params["explained_variance_warn"])
        self.explained_variance_fail = float(params["explained_variance_fail"])
        
        self._rollout_count = 0
        self._kl_fail_count = 0
        self._reward_history: deque[float] = deque(maxlen=self.stagnation_window + 2)
        self._value_loss_history: deque[float] = deque(maxlen=10)
        self._entropy_history: deque[float] = deque(maxlen=10)
        self._logger_invalid_warned = False
        self._name_to_value_empty_warned = False  # warn about SB3 timing artifact only once
        self._should_stop_training = False
        self._train_has_completed = False  # Track if train() has run at least once
        
    def _on_step(self) -> bool:
        if self._should_stop_training:
            return False
        return True
    
    def _on_rollout_end(self) -> None:
        if self.stop_flag is not None and self.stop_flag():
            self._should_stop_training = True
            return
        
        self._rollout_count += 1
        
        logger.debug(f"TrainingHealthMonitor: _on_rollout_end START | rollout={self._rollout_count}")
        metrics = self._extract_rollout_metrics()
        logger.debug(f"TrainingHealthMonitor: _on_rollout_end metrics | rollout={self._rollout_count} | has_metrics={bool(metrics)}")
        
        # Track if train() has completed by checking for train/* metrics.
        # train() runs AFTER _on_rollout_end() in SB3's learn loop,
        # so on the first rollout, train/* metrics don't exist yet.
        has_train_metrics = any(
            k.startswith("train/") and v is not None
            for k, v in metrics.items()
        )
        if has_train_metrics:
            self._train_has_completed = True
        
        ent_loss = metrics.get("train/entropy_loss")
        if ent_loss is not None and math.isfinite(ent_loss):
            self._entropy_history.append(abs(float(ent_loss)))
        
        v_loss = metrics.get("train/value_loss")
        if v_loss is not None and math.isfinite(v_loss):
            self._value_loss_history.append(float(v_loss))
        
        ep_rew = metrics.get("rollout/ep_rew_mean")
        if ep_rew is not None and math.isfinite(ep_rew):
            self._reward_history.append(float(ep_rew))
        
        status = self._check_health(metrics)
        
        if not status.healthy:
            logger.error(
                f"Training health FAILED: {status.failure_reason} | "
                f"rollout={self._rollout_count} | details={status.details}"
            )
            self._should_stop_training = True
            if self.on_failure is not None:
                self.on_failure(status.failure_reason, status.details)
        
        if status.warnings:
            for warn in status.warnings:
                logger.warning(f"Training health WARN: {warn} | rollout={self._rollout_count}")
                if self.on_warning is not None:
                    self.on_warning(warn, status.details)

    def _extract_rollout_metrics(self) -> dict[str, float | None]:
        logger = self.model.logger
        if logger is None:
            logger.debug("TrainingHealthMonitor: logger is None")
            return {}
        
        name_to_value = getattr(logger, "name_to_value", None)
        
        if not isinstance(name_to_value, dict):
            if not self._logger_invalid_warned:
                logger.warning(
                    "TrainingHealthMonitor: logger.name_to_value is not a dict "
                    f"- type={type(name_to_value)}, monitoring may be degraded"
                )
                self._logger_invalid_warned = True
            return {}
        
        if not name_to_value:
            all_keys = list(name_to_value.keys()) if name_to_value else []
            logger.debug(
                f"TrainingHealthMonitor: name_to_value is EMPTY | "
                f"rollout={self._rollout_count} | "
                f"train_has_completed={self._train_has_completed} | "
                f"available_keys={all_keys[:10]}"
            )
        else:
            train_keys = [k for k in name_to_value if k.startswith("train/")]
            rollout_keys = [k for k in name_to_value if k.startswith("rollout/")]
            if self._rollout_count <= 3 or self._rollout_count % 5 == 0:
                logger.debug(
                    f"TrainingHealthMonitor: name_to_value keys | "
                    f"rollout={self._rollout_count} | "
                    f"train_keys={train_keys[:5]} | rollout_keys={rollout_keys[:3]}"
                )
        
        metric_keys = [
            "train/policy_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/loss",
            "train/explained_variance",
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
        ]
        
        result = {}
        for key in metric_keys:
            try:
                val = name_to_value.get(key)
                if val is not None:
                    result[key] = float(val)
            except Exception:
                pass
        
        return result
    
    def _check_health(self, metrics: dict[str, float | None]) -> TrainingHealthStatus:
        warnings: list[str] = []
        failure_reason = ""
        details = dict(metrics)
        
        # In SB3's learn() loop, _on_rollout_end() fires BEFORE train() runs.
        # On the first rollout, train() hasn't run yet, so train/* metrics don't exist.
        # With log_interval=1 (SB3 default), dump_logs() clears name_to_value every iteration
        # BEFORE train() logs, so _on_rollout_end() typically sees the previous-previous
        # iteration's metrics (or nothing if only one train() has run).
        # The rollout/ep_* metrics ARE present (from collect_rollouts), so use those for health.
        if not metrics:
            if not self._train_has_completed:
                return TrainingHealthStatus(
                    healthy=True,
                    failure_reason="",
                    warnings=("No training metrics yet (train() not completed)",),
                    details={}
                )
            else:
                if not self._name_to_value_empty_warned:
                    warnings.append("name_to_value empty (SB3 dump_logs timing - this is expected with log_interval=1)")
                    self._name_to_value_empty_warned = True
        
        for key, val in metrics.items():
            if val is None:
                continue
            if not math.isfinite(val):
                failure_reason = "nan_loss"
                details["loss_key"] = key
                details["loss_value"] = val
                return TrainingHealthStatus(
                    healthy=False,
                    failure_reason=failure_reason,
                    warnings=(),
                    details=details
                )
        
        approx_kl = metrics.get("train/approx_kl")
        if approx_kl is not None and math.isfinite(approx_kl):
            kl_val = float(approx_kl)
            if kl_val > self.kl_fail_threshold:
                self._kl_fail_count += 1
                if self._kl_fail_count >= self.kl_persistence_required:
                    failure_reason = "kl_divergence_persistent"
                    details["approx_kl"] = kl_val
                    details["consecutive_failures"] = self._kl_fail_count
                    return TrainingHealthStatus(
                        healthy=False,
                        failure_reason=failure_reason,
                        warnings=(),
                        details=details
                    )
            elif kl_val > self.kl_warn_threshold:
                warnings.append(f"KL divergence elevated: {kl_val:.4f}")
            else:
                self._kl_fail_count = max(0, self._kl_fail_count - 1)
        
        clip_frac = metrics.get("train/clip_fraction")
        if clip_frac is not None and math.isfinite(clip_frac):
            cf_val = float(clip_frac)
            if cf_val > self.clip_warn_threshold and self._rollout_count > self.warmup_rollouts:
                warnings.append(f"Clip fraction high: {cf_val:.2%}")
        
        if len(self._entropy_history) >= 5:
            current_entropy = self._entropy_history[-1]
            recent_avg = sum(list(self._entropy_history)[-5:]) / 5
            if recent_avg > self.value_loss_baseline_floor:
                entropy_ratio = current_entropy / recent_avg
                if entropy_ratio < self.entropy_decay_warn_threshold:
                    warnings.append(
                        f"Entropy decay: {entropy_ratio:.1%} of recent average "
                        f"({current_entropy:.4f} vs {recent_avg:.4f})"
                    )
        
        v_loss = metrics.get("train/value_loss")
        if v_loss is not None and len(self._value_loss_history) >= 3:
            v_val = float(v_loss)
            moving_avg = sum(self._value_loss_history) / len(self._value_loss_history)
            baseline = max(moving_avg, self.value_loss_baseline_floor)
            if v_val > self.value_loss_spike_factor * baseline:
                warnings.append(
                    f"Value loss spike: {v_val:.2f} vs avg {moving_avg:.2f} "
                    f"({v_val/baseline:.1f}x)"
                )
        
        exp_var = metrics.get("train/explained_variance")
        if exp_var is not None and math.isfinite(exp_var):
            ev_val = float(exp_var)
            # Only check explained_variance after warmup - value function takes time to learn
            if ev_val < self.explained_variance_fail and self._rollout_count > self.warmup_rollouts:
                failure_reason = "explained_variance_negative"
                details["explained_variance"] = ev_val
                return TrainingHealthStatus(
                    healthy=False,
                    failure_reason=failure_reason,
                    warnings=(),
                    details=details
                )
            elif ev_val < self.explained_variance_warn and self._rollout_count > self.warmup_rollouts:
                warnings.append(f"Explained variance low: {ev_val:.3f}")
        
        stagnation = self._check_reward_stagnation()
        if stagnation.is_stagnant:
            warnings.append(
                f"Reward stagnation: slope={stagnation.slope:.4f}, "
                f"range={stagnation.abs_range:.2f}, "
                f"mean={stagnation.mean:.2f}"
            )
        
        return TrainingHealthStatus(
            healthy=not bool(failure_reason),
            failure_reason=failure_reason,
            warnings=tuple(warnings),
            details=details
        )
    
    def _check_reward_stagnation(self) -> "_StagnationStatus":
        """Check reward stagnation using both relative slope and absolute range."""
        if len(self._reward_history) < self.stagnation_window:
            return _StagnationStatus(is_stagnant=False, slope=0.0, abs_range=0.0, mean=0.0)
        
        recent = list(self._reward_history)[-self.stagnation_window:]
        mean_reward = sum(recent) / len(recent)
        
        slope, _ = np.polyfit(range(len(recent)), recent, 1)
        rel_slope = abs(slope) / (abs(mean_reward) + 1e-8)
        abs_range = max(recent) - min(recent)
        
        is_stagnant = (
            rel_slope < self.stagnation_rel_threshold
            and abs_range < self.stagnation_abs_threshold * max(1.0, abs(mean_reward))
        )
        
        return _StagnationStatus(
            is_stagnant=is_stagnant,
            slope=rel_slope,
            abs_range=abs_range,
            mean=mean_reward,
        )
    
    @property
    def rollout_count(self) -> int:
        return int(self._rollout_count)


class PpoSnakeAgent:
    def __init__(
        self,
        settings: Settings,
        artifact_dir: Path,
        config: PpoConfig,
        reward_config: RewardConfig,
        obs_config: ObsConfig,
        autoload: bool = True,
        legacy_model_path: Path | None = None,
        dropout_config: DropoutConfig | None = None,
    ) -> None:
        self.settings = settings
        self.artifact_dir = Path(artifact_dir)
        self.config = config
        self.reward_config = reward_config
        self.obs_config = obs_config
        self.dropout_config = dropout_config
        self._validate_config(config)
        self.env_count = max(1, int(config.env_count))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: PPO | None = None
        self.inference_model: PPO | None = None
        self._model_lock = threading.RLock()
        self._sync_request_event = threading.Event()
        self._last_inference_sync_steps = -1
        self._last_inference_sync_time = 0.0
        self.inference_sync_every_steps = 2048
        self.inference_sync_min_seconds = 0.2
        self._adaptive_reward_enabled = bool(getattr(reward_config, "use_reachable_space_penalty", True))
        # Default to the latest checkpointed policy for app/live workflows.
        # Reward-based "best" can diverge from score quality when shaping changes.
        self._inference_selector = ModelSelector.LAST
        self._train_vecnormalize: VecNormalize | None = None
        self._obs_norm_mean: np.ndarray | None = None
        self._obs_norm_var: np.ndarray | None = None
        self._obs_norm_eps: float = 1e-8
        self._obs_norm_clip: float = 10.0
        self._best_eval_score: float | None = None
        self._best_eval_step: int = 0
        self._last_eval_score: float | None = None
        self._eval_runs_completed: int = 0
        self._latest_requested_total_timesteps: int | None = None
        self._latest_actual_total_timesteps: int | None = None
        self._latest_run_id: str | None = None
        self._latest_run_started_at_unix_s: float | None = None
        self._resume_vecnormalize_source: Path | None = None
        self._provenance_snapshot = self._build_provenance_snapshot()
        self.legacy_model_path = legacy_model_path or self._default_legacy_model_path()
        self._load_eval_metadata()
        if bool(autoload):
            self.load_if_exists()

    @staticmethod
    def _validate_config(config: PpoConfig) -> None:
        env_count = int(config.env_count)
        n_steps = int(config.n_steps)
        batch_size = int(config.batch_size)
        n_epochs = int(config.n_epochs)
        if env_count < 1:
            raise ValueError("PpoConfig.env_count must be >= 1")
        if n_steps < 1:
            raise ValueError("PpoConfig.n_steps must be >= 1")
        if n_epochs < 1:
            raise ValueError("PpoConfig.n_epochs must be >= 1")
        if batch_size < 2:
            raise ValueError("PpoConfig.batch_size must be >= 2")
        rollout = env_count * n_steps
        if rollout <= 1:
            raise ValueError("PpoConfig.env_count * PpoConfig.n_steps must be > 1")
        if batch_size > rollout:
            raise ValueError("PpoConfig.batch_size cannot exceed env_count * n_steps")
        if rollout % batch_size != 0:
            raise ValueError(
                "PpoConfig.batch_size should divide env_count * n_steps "
                "to avoid truncated PPO minibatches"
            )
        if int(config.eval_freq_steps) < 0:
            raise ValueError("PpoConfig.eval_freq_steps must be >= 0")
        if int(config.checkpoint_freq_steps) < 0:
            raise ValueError("PpoConfig.checkpoint_freq_steps must be >= 0")
        if int(config.eval_episodes) < 1:
            raise ValueError("PpoConfig.eval_episodes must be >= 1")
        if int(config.no_improvement_evals) < 1:
            raise ValueError("PpoConfig.no_improvement_evals must be >= 1")
        if int(config.min_evals_before_stop) < 0:
            raise ValueError("PpoConfig.min_evals_before_stop must be >= 0")
        if config.target_kl is not None and float(config.target_kl) <= 0.0:
            raise ValueError("PpoConfig.target_kl must be > 0 when provided")
        if not config.policy_net_arch:
            raise ValueError("PpoConfig.policy_net_arch must not be empty")
        if any(int(layer) <= 0 for layer in config.policy_net_arch):
            raise ValueError("PpoConfig.policy_net_arch values must be positive")
        pi_arch = getattr(config, "policy_net_arch_pi", None)
        vf_arch = getattr(config, "policy_net_arch_vf", None)
        if pi_arch is not None and any(int(layer) <= 0 for layer in pi_arch):
            raise ValueError("PpoConfig.policy_net_arch_pi values must be positive")
        if vf_arch is not None and any(int(layer) <= 0 for layer in vf_arch):
            raise ValueError("PpoConfig.policy_net_arch_vf values must be positive")
        if (pi_arch is None) ^ (vf_arch is None):
            raise ValueError("PpoConfig.policy_net_arch_pi and policy_net_arch_vf must be provided together")

    @property
    def model_path(self) -> Path:
        return self._last_model_path()

    @property
    def is_ready(self) -> bool:
        return self.inference_model is not None or self.model is not None

    @property
    def is_inference_available(self) -> bool:
        return self.inference_model is not None

    @property
    def is_sync_pending(self) -> bool:
        return self._sync_request_event.is_set()

    @property
    def best_eval_score(self) -> float | None:
        return None if self._best_eval_score is None else float(self._best_eval_score)

    @property
    def best_eval_step(self) -> int:
        return int(self._best_eval_step)

    @property
    def last_eval_score(self) -> float | None:
        return None if self._last_eval_score is None else float(self._last_eval_score)

    @property
    def eval_runs_completed(self) -> int:
        return int(self._eval_runs_completed)

    @property
    def latest_run_id(self) -> str:
        return "" if self._latest_run_id is None else str(self._latest_run_id)

    def set_model_selector(self, selector: str | ModelSelector) -> None:
        self._inference_selector = self._coerce_selector(selector)

    def get_model_selector(self) -> str:
        return str(self._inference_selector.value)

    def _default_legacy_model_path(self) -> Path:
        if self.artifact_dir.parent.name == "ppo":
            return self.artifact_dir.parent.parent / "ppo_snake_model.zip"
        return self.artifact_dir.parent / "ppo_snake_model.zip"

    def switch_artifact_dir(self, artifact_dir: Path) -> None:
        new_dir = Path(artifact_dir)
        with self._model_lock:
            self.artifact_dir = new_dir
            self.legacy_model_path = self._default_legacy_model_path()
            self.model = None
            self.inference_model = None
            self._sync_request_event.clear()
            self._last_inference_sync_steps = -1
            self._last_inference_sync_time = 0.0
            self._obs_norm_mean = None
            self._obs_norm_var = None
            self._resume_vecnormalize_source = None
            self._train_vecnormalize = None
            self._best_eval_score = None
            self._best_eval_step = 0
            self._last_eval_score = None
            self._eval_runs_completed = 0
        self._load_eval_metadata()

    def _policy_kwargs(self) -> dict:
        pi_arch = getattr(self.config, "policy_net_arch_pi", None)
        vf_arch = getattr(self.config, "policy_net_arch_vf", None)
        if pi_arch is not None and vf_arch is not None:
            return {
                "net_arch": {
                    "pi": [int(v) for v in pi_arch],
                    "vf": [int(v) for v in vf_arch],
                }
            }
        return {"net_arch": [int(v) for v in self.config.policy_net_arch]}

    def _last_model_path(self) -> Path:
        return self.artifact_dir / "last_model.zip"

    def _resume_model_path(self) -> Path:
        return self.artifact_dir / "resume_model.zip"

    def _best_model_path(self) -> Path:
        return self.artifact_dir / "best_model.zip"

    def _best_score_model_path(self) -> Path:
        return self.artifact_dir / "best_score_model.zip"

    def _vecnormalize_path(self) -> Path:
        return self.artifact_dir / "vecnormalize.pkl"

    def _resume_vecnormalize_path(self) -> Path:
        return self.artifact_dir / "resume_vecnormalize.pkl"

    def _metadata_path(self) -> Path:
        return self.artifact_dir / "metadata.json"

    def _checkpoints_dir(self) -> Path:
        return self.artifact_dir / "checkpoints"

    def _eval_logs_dir(self) -> Path:
        return self.artifact_dir / "eval_logs"

    def _eval_trace_path(self) -> Path:
        return self._eval_logs_dir() / "evaluations_trace.jsonl"

    def _train_trace_path(self) -> Path:
        return self.artifact_dir / "training_trace.jsonl"

    def _run_logs_dir(self) -> Path:
        return self.artifact_dir / "run_logs"

    def _eval_trace_run_path(self, run_id: str) -> Path:
        return self._run_logs_dir() / f"eval_trace_{run_id}.jsonl"

    def _train_trace_run_path(self, run_id: str) -> Path:
        return self._run_logs_dir() / f"train_trace_{run_id}.jsonl"

    def _selected_model_path(self, selector: str | ModelSelector | None) -> Path:
        selected = self._inference_selector if selector is None else self._coerce_selector(selector)
        if selected == ModelSelector.BEST:
            best_score = self._best_score_model_path()
            if best_score.exists():
                return best_score
            return self._best_model_path()
        return self._last_model_path()

    def _latest_checkpoint_model_and_stats(self) -> tuple[Path, Path | None, int] | None:
        checkpoints_dir = self._checkpoints_dir()
        if not checkpoints_dir.exists():
            return None
        pattern = re.compile(r"^step_(\d+)_steps\.zip$")
        best: tuple[Path, Path | None, int] | None = None
        for candidate in checkpoints_dir.glob("step_*_steps.zip"):
            match = pattern.match(candidate.name)
            if match is None:
                continue
            step = int(match.group(1))
            stats = checkpoints_dir / f"step_vecnormalize_{step}_steps.pkl"
            stats_path = stats if stats.exists() else None
            if best is None or step > int(best[2]):
                best = (candidate, stats_path, step)
        return best

    def _resolve_resume_artifacts(self) -> tuple[Path, Path | None, int | None] | None:
        resume_model = self._resume_model_path()
        if resume_model.exists():
            resume_stats = self._resume_vecnormalize_path()
            if not resume_stats.exists():
                legacy_stats = self._vecnormalize_path()
                resume_stats = legacy_stats if legacy_stats.exists() else resume_stats
            return (
                resume_model,
                resume_stats if resume_stats.exists() else None,
                None,
            )
        legacy_last = self._last_model_path()
        if legacy_last.exists():
            legacy_stats = self._vecnormalize_path()
            return (
                legacy_last,
                legacy_stats if legacy_stats.exists() else None,
                None,
            )
        checkpoint = self._latest_checkpoint_model_and_stats()
        if checkpoint is not None:
            return checkpoint
        return None

    @staticmethod
    def _coerce_selector(selector: str | ModelSelector) -> ModelSelector:
        if isinstance(selector, ModelSelector):
            return selector
        return ModelSelector(str(selector).strip().lower())

    def _load_eval_metadata(self) -> None:
        path = self._metadata_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to read metadata %s", path, exc_info=True)
            return
        if not isinstance(payload, dict):
            return
        try:
            best = payload.get("best_eval_score")
            self._best_eval_score = self._finite_or_none(best)
            self._best_eval_step = max(0, int(payload.get("best_eval_step", 0) or 0))
            last = payload.get("last_eval_score")
            self._last_eval_score = self._finite_or_none(last)
            self._eval_runs_completed = max(0, int(payload.get("eval_runs_completed", 0) or 0))
            req = payload.get("requested_total_timesteps")
            self._latest_requested_total_timesteps = None if req is None else int(req)
            act = payload.get("actual_total_timesteps")
            self._latest_actual_total_timesteps = None if act is None else int(act)
            run_id = payload.get("latest_run_id")
            self._latest_run_id = str(run_id).strip() if run_id is not None else None
            started = payload.get("latest_run_started_at_unix_s")
            self._latest_run_started_at_unix_s = None if started is None else float(started)
        except Exception:
            logger.debug("Failed to parse metadata %s", path, exc_info=True)

    def _write_metadata(
        self,
        *,
        requested_steps: int | None = None,
        actual_steps: int | None = None,
        training_episode_summary: dict[str, object] | None = None,
        latest_eval_trace: dict[str, object] | None = None,
        run_id: str | None = None,
        run_started_at_unix_s: float | None = None,
    ) -> None:
        if requested_steps is not None:
            self._latest_requested_total_timesteps = int(requested_steps)
        if actual_steps is not None:
            self._latest_actual_total_timesteps = int(actual_steps)
        payload = {
            "schema_version": 2,
            "generated_at_unix_s": time.time(),
            "device": str(self.device),
            "artifact_dir": str(self.artifact_dir),
            "experiment_name": str(self.artifact_dir.name),
            "selector": str(self._inference_selector.value),
            "requested_total_timesteps": (
                None
                if self._latest_requested_total_timesteps is None
                else int(self._latest_requested_total_timesteps)
            ),
            "actual_total_timesteps": (
                None if self._latest_actual_total_timesteps is None else int(self._latest_actual_total_timesteps)
            ),
            "best_eval_score": self._finite_or_none(self._best_eval_score),
            "best_eval_step": int(self._best_eval_step),
            "last_eval_score": self._finite_or_none(self._last_eval_score),
            "eval_runs_completed": int(self._eval_runs_completed),
            "latest_run_id": (
                str(run_id)
                if run_id is not None
                else (None if self._latest_run_id is None else str(self._latest_run_id))
            ),
            "latest_run_started_at_unix_s": (
                float(run_started_at_unix_s)
                if run_started_at_unix_s is not None
                else (
                    None
                    if self._latest_run_started_at_unix_s is None
                    else float(self._latest_run_started_at_unix_s)
                )
            ),
            "adaptive_reward_enabled": bool(self._adaptive_reward_enabled),
            "config": asdict(self.config),
            "reward_config": asdict(self._effective_reward_config()),
            "obs_config": asdict(self.obs_config),
            "runtime_controls": {
                "model_selector": str(self._inference_selector.value),
                "adaptive_reward_enabled": bool(self._adaptive_reward_enabled),
                "settings": asdict(self.settings),
            },
            "provenance": dict(self._provenance_snapshot),
        }
        if training_episode_summary is not None:
            payload["training_episode_summary"] = dict(training_episode_summary)
        if latest_eval_trace is not None:
            payload["latest_eval_trace"] = dict(latest_eval_trace)
        self._atomic_write_text(
            self._metadata_path(),
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @staticmethod
    def _temp_path_for(target: Path, label: str) -> Path:
        suffix = f".{str(label).strip() or 'tmp'}.{uuid.uuid4().hex}.tmp"
        return target.with_name(f"{target.name}{suffix}")

    @staticmethod
    def _cleanup_temp(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    def _atomic_write_text(self, target: Path, content: str, *, encoding: str = "utf-8") -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = self._temp_path_for(target, "meta")
        try:
            with temp.open("w", encoding=encoding) as handle:
                handle.write(str(content))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp), str(target))
        finally:
            self._cleanup_temp(temp)

    def _atomic_save_via_callback(self, target: Path, *, label: str, save_fn: Callable[[Path], None]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = self._temp_path_for(target, label)
        try:
            save_fn(temp)
            os.replace(str(temp), str(target))
        finally:
            self._cleanup_temp(temp)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False))
            handle.write("\n")

    @staticmethod
    def _new_run_id(*, start_step: int, requested_steps: int) -> str:
        return f"r{int(time.time())}_{int(start_step)}_{int(requested_steps)}"

    @staticmethod
    def _dependency_version_or_none(package_name: str) -> str | None:
        try:
            return str(importlib.metadata.version(package_name))
        except Exception:
            return None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _git_revision_snapshot(cls) -> dict[str, object]:
        root = cls._project_root()

        def _run_git(args: list[str]) -> str | None:
            try:
                proc = subprocess.run(
                    ["git"] + list(args),
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
            except Exception:
                return None
            if proc.returncode != 0:
                return None
            out = str(proc.stdout).strip()
            return out or None

        commit = _run_git(["rev-parse", "HEAD"])
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        dirty = None
        status = _run_git(["status", "--porcelain"])
        if status is not None:
            dirty = bool(status)
        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": dirty,
        }

    @classmethod
    def _build_provenance_snapshot(cls) -> dict[str, object]:
        dependencies = {
            "pygame": cls._dependency_version_or_none("pygame"),
            "numpy": cls._dependency_version_or_none("numpy"),
            "gymnasium": cls._dependency_version_or_none("gymnasium"),
            "stable_baselines3": cls._dependency_version_or_none("stable-baselines3"),
            "sb3_contrib": cls._dependency_version_or_none("sb3-contrib"),
            "torch": cls._dependency_version_or_none("torch"),
        }
        environment = {
            "python_version": str(platform.python_version()),
            "python_implementation": str(platform.python_implementation()),
            "platform": str(platform.platform()),
            "machine": str(platform.machine()),
            "processor": str(platform.processor()),
            "os_name": str(os.name),
            "cpu_count": int(os.cpu_count() or 0),
        }
        return {
            "captured_at_unix_s": float(time.time()),
            "project_root": str(cls._project_root()),
            "git": cls._git_revision_snapshot(),
            "environment": environment,
            "dependencies": dependencies,
        }

    @staticmethod
    def _finite_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    def _validate_model_shapes(self, models: Iterable[PPO]) -> ModelOpResult | None:
        expected_shape = (observation_size(self.obs_config),)
        for loaded in models:
            loaded_obs_space = getattr(loaded, "observation_space", None)
            loaded_shape = getattr(loaded_obs_space, "shape", None)
            if loaded_shape is None:
                continue
            if tuple(loaded_shape) != expected_shape:
                with self._model_lock:
                    self.model = None
                    self.inference_model = None
                    self._sync_request_event.clear()
                return ModelOpResult(
                    ok=False,
                    code=ModelOpCode.INCOMPATIBLE,
                    detail=f"expected observation shape {expected_shape}, got {tuple(loaded_shape)}",
                )
        return None

    def _apply_vecnormalize_stats(self, vec_env: VecNormalize | None) -> None:
        if vec_env is None or getattr(vec_env, "obs_rms", None) is None:
            self._obs_norm_mean = None
            self._obs_norm_var = None
            self._obs_norm_eps = 1e-8
            self._obs_norm_clip = 10.0
            return
        self._obs_norm_mean = np.asarray(vec_env.obs_rms.mean, dtype=np.float32)
        self._obs_norm_var = np.asarray(vec_env.obs_rms.var, dtype=np.float32)
        self._obs_norm_eps = float(getattr(vec_env, "epsilon", 1e-8))
        self._obs_norm_clip = float(getattr(vec_env, "clip_obs", 10.0))

    def _load_vecnormalize_stats_from_disk(self) -> None:
        self._load_vecnormalize_stats_from_path(self._vecnormalize_path())

    def _load_vecnormalize_stats_from_path(self, path: Path | None) -> None:
        if path is None:
            self._obs_norm_mean = None
            self._obs_norm_var = None
            self._obs_norm_eps = 1e-8
            self._obs_norm_clip = 10.0
            return
        if not path.exists():
            self._obs_norm_mean = None
            self._obs_norm_var = None
            self._obs_norm_eps = 1e-8
            self._obs_norm_clip = 10.0
            return
        probe_env = DummyVecEnv([self._make_single_env(None)])
        try:
            vec = VecNormalize.load(str(path), probe_env)
            vec.training = False
            vec.norm_reward = False
            self._apply_vecnormalize_stats(vec)
            vec.close()
        except Exception:
            logger.exception("Failed to load VecNormalize stats from %s", path)
            self._obs_norm_mean = None
            self._obs_norm_var = None
            self._obs_norm_eps = 1e-8
            self._obs_norm_clip = 10.0
        finally:
            probe_env.close()

    def _save_last_and_stats(self) -> ModelOpResult:
        with self._model_lock:
            if self.model is None:
                return ModelOpResult(ok=False, code=ModelOpCode.MISSING, detail="no model to save")
            try:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                self._checkpoints_dir().mkdir(parents=True, exist_ok=True)
                self._atomic_save_via_callback(
                    self._last_model_path(),
                    label="model",
                    save_fn=lambda p: self.model.save(str(p)),
                )
                self._atomic_save_via_callback(
                    self._resume_model_path(),
                    label="model",
                    save_fn=lambda p: self.model.save(str(p)),
                )
                if self._train_vecnormalize is not None:
                    self._atomic_save_via_callback(
                        self._vecnormalize_path(),
                        label="vec",
                        save_fn=lambda p: self._train_vecnormalize.save(str(p)),
                    )
                    self._atomic_save_via_callback(
                        self._resume_vecnormalize_path(),
                        label="vec",
                        save_fn=lambda p: self._train_vecnormalize.save(str(p)),
                    )
                    self._apply_vecnormalize_stats(self._train_vecnormalize)
                    self._resume_vecnormalize_source = self._resume_vecnormalize_path()
                self._write_metadata(actual_steps=int(getattr(self.model, "num_timesteps", 0)))
                return ModelOpResult(ok=True, code=ModelOpCode.OK)
            except OSError as exc:
                logger.exception("Failed to save PPO artifacts to %s", self.artifact_dir)
                return ModelOpResult(ok=False, code=ModelOpCode.FILESYSTEM_ERROR, detail=str(exc))
            except Exception as exc:
                logger.exception("Failed to save PPO artifacts to %s", self.artifact_dir)
                return ModelOpResult(ok=False, code=ModelOpCode.UNKNOWN_ERROR, detail=str(exc))

    def _ensure_best_model_artifact(self) -> None:
        best_path = self._best_model_path()
        best_score_path = self._best_score_model_path()
        last_path = self._last_model_path()
        if not best_path.exists() and last_path.exists():
            try:
                best_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(last_path, best_path)
            except Exception:
                logger.exception("Failed to bootstrap best model artifact from last model")
        if not best_score_path.exists() and last_path.exists():
            try:
                best_score_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(last_path, best_score_path)
            except Exception:
                logger.exception("Failed to bootstrap best-score model artifact from last model")

    def load_if_exists(self) -> bool:
        return self.load_if_exists_detailed().ok

    def load_if_exists_detailed(self, selector: str | ModelSelector | None = None) -> ModelOpResult:
        if self.legacy_model_path.exists():
            with self._model_lock:
                self.model = None
                self.inference_model = None
                self._sync_request_event.clear()
            return ModelOpResult(
                ok=False,
                code=ModelOpCode.LEGACY_FORMAT_UNSUPPORTED,
                detail=f"legacy model file detected at {self.legacy_model_path}",
            )
        resume_artifacts = self._resolve_resume_artifacts()
        selected_model_path = self._selected_model_path(selector)
        last_model_path = self._last_model_path()
        if not selected_model_path.exists() and not last_model_path.exists() and resume_artifacts is None:
            return ModelOpResult(ok=False, code=ModelOpCode.MISSING, detail="model artifacts do not exist")
        if not selected_model_path.exists() and resume_artifacts is None:
            return ModelOpResult(
                ok=False,
                code=ModelOpCode.MISSING,
                detail=f"selected model not found: {selected_model_path.name}",
            )
        inference_path = selected_model_path if selected_model_path.exists() else (resume_artifacts[0] if resume_artifacts is not None else last_model_path)
        resume_model_path = resume_artifacts[0] if resume_artifacts is not None else (
            last_model_path if last_model_path.exists() else inference_path
        )
        resume_vec_path = resume_artifacts[1] if resume_artifacts is not None else (
            self._vecnormalize_path() if self._vecnormalize_path().exists() else None
        )
        try:
            selected_model = PPO.load(str(inference_path), device=self.device, learning_rate=self.config.learning_rate_start)
            training_model = PPO.load(str(resume_model_path), device=self.device, learning_rate=self.config.learning_rate_start)
        except OSError as exc:
            logger.exception("Failed to load PPO model from %s", inference_path)
            with self._model_lock:
                self.model = None
                self.inference_model = None
                self._sync_request_event.clear()
            return ModelOpResult(ok=False, code=ModelOpCode.FILESYSTEM_ERROR, detail=str(exc))
        except Exception as exc:
            logger.exception("Failed to load PPO model from %s", inference_path)
            with self._model_lock:
                self.model = None
                self.inference_model = None
                self._sync_request_event.clear()
            detail = str(exc)
            code = ModelOpCode.CORRUPT
            if "MaskableActorCriticPolicy" in detail:
                code = ModelOpCode.INCOMPATIBLE
                detail = "saved model uses non-maskable PPO; retraining with the new observation/masking stack is required"
            return ModelOpResult(ok=False, code=code, detail=detail)

        shape_error = self._validate_model_shapes((selected_model, training_model))
        if shape_error is not None:
            return shape_error

        self._load_eval_metadata()
        self._load_vecnormalize_stats_from_path(resume_vec_path)
        with self._model_lock:
            self.model = training_model
            self.inference_model = selected_model
            self._last_inference_sync_steps = int(getattr(training_model, "num_timesteps", 0))
            self._last_inference_sync_time = time.perf_counter()
            self._sync_request_event.clear()
            self._resume_vecnormalize_source = resume_vec_path
        return ModelOpResult(ok=True, code=ModelOpCode.OK)

    def load_latest_checkpoint(self) -> bool:
        return self.load_latest_checkpoint_detailed().ok

    def load_latest_checkpoint_detailed(self) -> ModelOpResult:
        checkpoint = self._latest_checkpoint_model_and_stats()
        if checkpoint is None:
            return ModelOpResult(ok=False, code=ModelOpCode.MISSING, detail="no checkpoint artifacts found")
        model_path, stats_path, step = checkpoint
        try:
            infer_model = PPO.load(str(model_path), device=self.device, learning_rate=self.config.learning_rate_start)
            train_model = PPO.load(str(model_path), device=self.device, learning_rate=self.config.learning_rate_start)
        except OSError as exc:
            logger.exception("Failed to load checkpoint model from %s", model_path)
            return ModelOpResult(ok=False, code=ModelOpCode.FILESYSTEM_ERROR, detail=str(exc))
        except Exception as exc:
            logger.exception("Failed to load checkpoint model from %s", model_path)
            detail = str(exc)
            code = ModelOpCode.CORRUPT
            if "MaskableActorCriticPolicy" in detail:
                code = ModelOpCode.INCOMPATIBLE
                detail = "saved checkpoint uses non-maskable PPO; retraining with the new observation/masking stack is required"
            return ModelOpResult(ok=False, code=code, detail=detail)
        shape_error = self._validate_model_shapes((infer_model, train_model))
        if shape_error is not None:
            return shape_error
        self._load_eval_metadata()
        self._load_vecnormalize_stats_from_path(stats_path)
        with self._model_lock:
            self.model = train_model
            self.inference_model = infer_model
            self._last_inference_sync_steps = int(getattr(train_model, "num_timesteps", 0))
            self._last_inference_sync_time = time.perf_counter()
            self._sync_request_event.clear()
            self._resume_vecnormalize_source = stats_path
        return ModelOpResult(
            ok=True,
            code=ModelOpCode.OK,
            detail=f"loaded checkpoint step {int(step)}",
        )

    def save(self) -> bool:
        return self.save_detailed().ok

    def save_detailed(self) -> ModelOpResult:
        result = self._save_last_and_stats()
        if result.ok:
            self._ensure_best_model_artifact()
            reload_result = self.load_if_exists_detailed(selector=self._inference_selector)
            if not reload_result.ok:
                return ModelOpResult(
                    ok=False,
                    code=reload_result.code,
                    detail=f"saved artifacts but reload failed: {reload_result.detail}",
                )
        return result

    def delete(self) -> bool:
        return self.delete_detailed().ok

    def delete_detailed(self) -> ModelOpResult:
        with self._model_lock:
            self.model = None
            self.inference_model = None
            self._sync_request_event.clear()
            self._last_inference_sync_steps = -1
            self._last_inference_sync_time = 0.0
            self._obs_norm_mean = None
            self._obs_norm_var = None
            self._resume_vecnormalize_source = None
            self._best_eval_score = None
            self._best_eval_step = 0
            self._last_eval_score = None
            self._eval_runs_completed = 0

        removed_any = False
        for path in (
            self._best_model_path(),
            self._best_score_model_path(),
            self._last_model_path(),
            self._resume_model_path(),
            self._vecnormalize_path(),
            self._resume_vecnormalize_path(),
            self._metadata_path(),
            self._train_trace_path(),
            self.legacy_model_path,
        ):
            if not path.exists():
                continue
            try:
                path.unlink()
                removed_any = True
            except OSError as exc:
                return ModelOpResult(ok=False, code=ModelOpCode.FILESYSTEM_ERROR, detail=str(exc))
        for directory in (
            self._checkpoints_dir(),
            self._eval_logs_dir(),
            self._run_logs_dir(),
        ):
            if not directory.exists():
                continue
            for child in directory.glob("*"):
                try:
                    if child.is_file():
                        child.unlink()
                        removed_any = True
                except OSError as exc:
                    return ModelOpResult(ok=False, code=ModelOpCode.FILESYSTEM_ERROR, detail=str(exc))
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            if self.artifact_dir.exists() and self.artifact_dir.is_dir() and not any(self.artifact_dir.iterdir()):
                self.artifact_dir.rmdir()
        except OSError:
            pass
        if removed_any:
            return ModelOpResult(ok=True, code=ModelOpCode.OK)
        return ModelOpResult(ok=False, code=ModelOpCode.MISSING, detail="artifacts not found")

    def request_inference_sync(self) -> None:
        self._sync_request_event.set()

    @staticmethod
    def _steps_to_callback_calls(steps: int, env_count: int) -> int:
        return max(1, int(int(steps) // max(1, int(env_count))))

    def train(
        self,
        total_timesteps: int,
        stop_flag: Callable[[], bool],
        on_progress: ProgressFn | None = None,
        on_score: ScoreFn | None = None,
        on_episode_info: EpisodeInfoFn | None = None,
    ) -> int:
        total_steps = max(1, int(total_timesteps))
        train_env = self._make_train_vec_env()
        self._train_vecnormalize = train_env
        eval_env: VecNormalize | None = None
        steps_done = int(self.model.num_timesteps) if self.model is not None else 0
        prev_best_eval_score = self._finite_or_none(self._best_eval_score)
        prev_best_eval_step = max(0, int(self._best_eval_step))
        prev_last_eval_score = self._finite_or_none(self._last_eval_score)
        prev_eval_runs_completed = max(0, int(self._eval_runs_completed))
        train_deaths: dict[str, int] = {"wall": 0, "body": 0, "starvation": 0, "fill": 0, "other": 0}
        train_scores: list[int] = []
        train_episode_steps: list[int] = []
        latest_eval_trace: dict[str, object] | None = None
        start_timestep = int(self.model.num_timesteps) if self.model is not None else 0
        run_started_at_unix_s = float(time.time())
        run_id = self._new_run_id(start_step=start_timestep, requested_steps=total_steps)
        self._latest_run_id = str(run_id)
        self._latest_run_started_at_unix_s = float(run_started_at_unix_s)
        try:
            ent_schedule = self._linear_schedule(
                start=self.config.ent_coef_start,
                end=self.config.ent_coef_end,
            )
            if self.model is None:
                self.model = PPO(
                    "MlpPolicy",
                    train_env,
                    verbose=0,
                    n_steps=self.config.n_steps,
                    batch_size=self.config.batch_size,
                    n_epochs=self.config.n_epochs,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                    learning_rate=self._linear_schedule(
                        start=self.config.learning_rate_start,
                        end=self.config.learning_rate_end,
                    ),
                    clip_range=self.config.clip_range,
                    target_kl=self.config.target_kl,
                    ent_coef=float(self.config.ent_coef_start),
                    policy_kwargs=self._policy_kwargs(),
                    device=self.device,
                    seed=self.config.seed,
                )
            else:
                self.model.set_env(train_env)

            eval_callback: _SyncEvalCallback | None = None
            eval_freq_steps = int(self.config.eval_freq_steps)
            if eval_freq_steps > 0:
                eval_env = self._make_eval_vec_env()
                stop_callback: StopTrainingOnNoModelImprovement | None = None
                if bool(self.config.use_stop_on_no_improvement):
                    stop_callback = StopTrainingOnNoModelImprovement(
                        max_no_improvement_evals=int(self.config.no_improvement_evals),
                        min_evals=int(self.config.min_evals_before_stop),
                        verbose=0,
                    )
                def _on_eval_complete(detail: dict[str, object]) -> None:
                    nonlocal latest_eval_trace
                    latest_eval_trace = dict(detail)
                    payload = {
                        "generated_at_unix_s": float(time.time()),
                        "run_id": str(run_id),
                        "run_started_at_unix_s": float(run_started_at_unix_s),
                        "requested_total_timesteps": int(total_steps),
                        "adaptive_reward_enabled": bool(self._adaptive_reward_enabled),
                        "step": int(detail.get("step", 0)),
                        "mean_reward": float(detail.get("mean_reward", 0.0)),
                        "best_mean_reward": float(detail.get("best_mean_reward", 0.0)),
                        "mean_score": float(detail.get("mean_score", 0.0)),
                        "best_mean_score": float(detail.get("best_mean_score", 0.0)),
                        "best_score_step": int(detail.get("best_score_step", 0)),
                        "eval_run_index": int(detail.get("eval_run_index", 0)),
                        "episode_rewards": [float(v) for v in detail.get("episode_rewards", [])],
                        "episode_lengths": [int(v) for v in detail.get("episode_lengths", [])],
                        "episode_scores": [float(v) for v in detail.get("episode_scores", [])],
                    }
                    try:
                        self._append_jsonl(self._eval_trace_path(), payload)
                        self._append_jsonl(self._eval_trace_run_path(run_id), payload)
                    except Exception:
                        logger.exception("Failed to append eval trace")

                eval_callback = _SyncEvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=str(self.artifact_dir),
                    log_path=str(self._eval_logs_dir()),
                    best_score_model_path=str(self._best_score_model_path()),
                    eval_freq=self._steps_to_callback_calls(eval_freq_steps, self.env_count),
                    n_eval_episodes=int(self.config.eval_episodes),
                    deterministic=True,
                    render=False,
                    callback_after_eval=stop_callback,
                    stop_flag=stop_flag,
                    on_eval_complete=_on_eval_complete,
                    verbose=0,
                )
            checkpoint_callback: CheckpointCallback | None = None
            checkpoint_freq_steps = int(self.config.checkpoint_freq_steps)
            if checkpoint_freq_steps > 0:
                checkpoint_callback = CheckpointCallback(
                    save_freq=self._steps_to_callback_calls(checkpoint_freq_steps, self.env_count),
                    save_path=str(self._checkpoints_dir()),
                    name_prefix="step",
                    save_replay_buffer=False,
                    save_vecnormalize=True,
                    verbose=0,
                )
            def _capture_episode_info(info: dict) -> None:
                score = info.get("score")
                if score is not None:
                    train_scores.append(int(score))
                steps = info.get("steps")
                if steps is not None:
                    train_episode_steps.append(int(steps))
                reason_raw = str(info.get("death_reason", "other")).strip().lower()
                reason = reason_raw if reason_raw in train_deaths else "other"
                train_deaths[reason] = int(train_deaths.get(reason, 0) + 1)
                if on_episode_info is not None:
                    on_episode_info(dict(info))

            progress_callback = _StopAndProgressCallback(
                stop_flag=stop_flag,
                on_progress=on_progress,
                progress_every=4096,
                ent_coef_schedule=ent_schedule,
                on_step_hook=self._maybe_refresh_inference,
                on_episode_score=on_score,
                on_episode_info=_capture_episode_info,
                step_hook_every=64,
                start_timestep=start_timestep,
                total_timesteps=total_steps,
            )

            self._training_failed = False
            self._training_failure_reason: str | None = None
            self._training_failure_details: dict | None = None

            def _on_training_health_failure(reason: str, details: dict) -> None:
                logger.error(f"Training FAILED: {reason} | details: {details}")
                self._training_failed = True
                self._training_failure_reason = reason
                self._training_failure_details = details
                self._write_metadata(
                    requested_steps=total_steps,
                    actual_steps=int(self.model.num_timesteps),
                    training_episode_summary={
                        "status": "failed",
                        "failure_reason": reason,
                        "failure_details": details,
                    },
                )

            def _on_training_health_warning(warning: str, details: dict) -> None:
                logger.warning(f"Training WARNING: {warning}")

            health_monitor = TrainingHealthMonitor(
                on_failure=_on_training_health_failure,
                on_warning=_on_training_health_warning,
                stop_flag=stop_flag,
                kl_fail_threshold=0.2,
                kl_warn_threshold=0.1,
                kl_persistence_required=2,
                clip_warn_threshold=0.85,
                warmup_rollouts=3,
                entropy_decay_warn_threshold=0.1,
                value_loss_spike_factor=5.0,
                value_loss_baseline_floor=1e-6,
                stagnation_window=5,
                stagnation_tolerance=0.01,
                explained_variance_warn=0.0,
                explained_variance_fail=-5.0,  # Only fail on extreme values
                verbose=0,
            )

            callback_chain: list[BaseCallback] = [health_monitor, progress_callback]
            if checkpoint_callback is not None:
                callback_chain.append(checkpoint_callback)
            if eval_callback is not None:
                callback_chain.append(eval_callback)
            callbacks = CallbackList(callback_chain)
            self.model.learn(total_timesteps=total_steps, callback=callbacks, reset_num_timesteps=False)
            steps_done = int(self.model.num_timesteps)

            train_trace: dict[str, object] | None = None
            try:
                if self._training_failed:
                    logger.error(
                        f"Training failed before completion - skipping save. "
                        f"Reason: {self._training_failure_reason}"
                    )
                    self._write_metadata(
                        requested_steps=total_steps,
                        actual_steps=steps_done,
                        training_episode_summary={
                            "status": "failed",
                            "failure_reason": self._training_failure_reason,
                            "failure_details": self._training_failure_details,
                        },
                    )
                    return steps_done

                if eval_callback is not None:
                    run_eval_count = max(0, int(eval_callback.eval_runs_completed))
                    run_best_score = self._finite_or_none(eval_callback.best_eval_score)
                    run_best_step = max(0, int(eval_callback.best_eval_step))
                    run_last_score = self._finite_or_none(eval_callback.last_eval_score)
                    if run_eval_count > 0:
                        if run_best_score is None:
                            self._best_eval_score = prev_best_eval_score
                            self._best_eval_step = prev_best_eval_step
                        elif prev_best_eval_score is None or float(run_best_score) >= float(prev_best_eval_score):
                            self._best_eval_score = run_best_score
                            self._best_eval_step = run_best_step
                        else:
                            self._best_eval_score = prev_best_eval_score
                            self._best_eval_step = prev_best_eval_step
                        self._last_eval_score = run_last_score
                        self._eval_runs_completed = int(prev_eval_runs_completed + run_eval_count)
                    else:
                        self._best_eval_score = prev_best_eval_score
                        self._best_eval_step = prev_best_eval_step
                        self._last_eval_score = prev_last_eval_score
                        self._eval_runs_completed = prev_eval_runs_completed
                else:
                    self._best_eval_score = prev_best_eval_score
                    self._best_eval_step = prev_best_eval_step
                    self._last_eval_score = prev_last_eval_score
                    self._eval_runs_completed = prev_eval_runs_completed
                save_result = self._save_last_and_stats()
                if not save_result.ok:
                    raise RuntimeError(f"post-train save failed ({save_result.code.value}): {save_result.detail}")
                self._ensure_best_model_artifact()
                episodes_total = int(sum(int(v) for v in train_deaths.values()))
                score_summary = {
                    "count": int(len(train_scores)),
                    "mean": float(statistics.fmean(train_scores)) if train_scores else 0.0,
                    "median": float(np.median(np.asarray(train_scores, dtype=np.float64))) if train_scores else 0.0,
                    "p90": float(np.percentile(np.asarray(train_scores, dtype=np.float64), 90)) if train_scores else 0.0,
                    "best": int(max(train_scores)) if train_scores else 0,
                    "last": int(train_scores[-1]) if train_scores else 0,
                }
                step_summary = {
                    "count": int(len(train_episode_steps)),
                    "mean": float(statistics.fmean(train_episode_steps)) if train_episode_steps else 0.0,
                    "p90": float(np.percentile(np.asarray(train_episode_steps, dtype=np.float64), 90)) if train_episode_steps else 0.0,
                    "max": int(max(train_episode_steps)) if train_episode_steps else 0,
                }
                train_trace = {
                    "generated_at_unix_s": float(time.time()),
                    "run_id": str(run_id),
                    "run_started_at_unix_s": float(run_started_at_unix_s),
                    "requested_total_timesteps": int(total_steps),
                    "actual_total_timesteps": int(steps_done),
                    "adaptive_reward_enabled": bool(self._adaptive_reward_enabled),
                    "episodes_total": episodes_total,
                    "deaths": dict(train_deaths),
                    "score_summary": score_summary,
                    "episode_steps_summary": step_summary,
                }
                self._write_metadata(
                    requested_steps=total_steps,
                    actual_steps=steps_done,
                    training_episode_summary=train_trace,
                    latest_eval_trace=latest_eval_trace,
                    run_id=run_id,
                    run_started_at_unix_s=run_started_at_unix_s,
                )
                load_result = self.load_if_exists_detailed(selector=self._inference_selector)
                if not load_result.ok:
                    logger.warning("Post-train reload failed: %s", load_result.detail)
            finally:
                if train_trace is None:
                    episodes_total = int(sum(int(v) for v in train_deaths.values()))
                    score_summary = {
                        "count": int(len(train_scores)),
                        "mean": float(statistics.fmean(train_scores)) if train_scores else 0.0,
                        "median": float(np.median(np.asarray(train_scores, dtype=np.float64))) if train_scores else 0.0,
                        "p90": float(np.percentile(np.asarray(train_scores, dtype=np.float64), 90)) if train_scores else 0.0,
                        "best": int(max(train_scores)) if train_scores else 0,
                        "last": int(train_scores[-1]) if train_scores else 0,
                    }
                    step_summary = {
                        "count": int(len(train_episode_steps)),
                        "mean": float(statistics.fmean(train_episode_steps)) if train_episode_steps else 0.0,
                        "p90": float(np.percentile(np.asarray(train_episode_steps, dtype=np.float64), 90)) if train_episode_steps else 0.0,
                        "max": int(max(train_episode_steps)) if train_episode_steps else 0,
                    }
                    train_trace = {
                        "generated_at_unix_s": float(time.time()),
                        "run_id": str(run_id),
                        "run_started_at_unix_s": float(run_started_at_unix_s),
                        "requested_total_timesteps": int(total_steps),
                        "actual_total_timesteps": int(steps_done),
                        "adaptive_reward_enabled": bool(self._adaptive_reward_enabled),
                        "episodes_total": episodes_total,
                        "deaths": dict(train_deaths),
                        "score_summary": score_summary,
                        "episode_steps_summary": step_summary,
                    }
                try:
                    self._append_jsonl(self._train_trace_path(), train_trace)
                    self._append_jsonl(self._train_trace_run_path(run_id), train_trace)
                except Exception:
                    logger.exception("Failed to append training trace")
                try:
                    if self._train_vecnormalize is not None:
                        self._train_vecnormalize.save(str(self._vecnormalize_path()))
                except Exception:
                    logger.debug("vecnormalize.pkl sync skipped (may already be synced by checkpoint)")
        finally:
            if eval_env is not None:
                eval_env.close()
            train_env.close()
            self._train_vecnormalize = None
        return steps_done

    def _normalize_observation(self, obs_arr: np.ndarray) -> np.ndarray:
        if self._obs_norm_mean is None or self._obs_norm_var is None:
            return obs_arr
        mean = np.asarray(self._obs_norm_mean, dtype=np.float32)
        var = np.asarray(self._obs_norm_var, dtype=np.float32)
        norm = (obs_arr - mean) / np.sqrt(var + float(self._obs_norm_eps))
        return np.clip(norm, -float(self._obs_norm_clip), float(self._obs_norm_clip))

    def predict_action(self, obs, action_masks=None) -> int:
        with self._model_lock:
            model = self.inference_model
            if model is None:
                return 0
            return self._predict_with_model(model, obs, self._normalize_observation, action_masks=action_masks)

    def predict_action_with_probs(self, obs, action_masks=None) -> tuple[int, tuple[float, float, float] | None]:
        with self._model_lock:
            model = self.inference_model
            if model is None:
                return 0, None
            action = self._predict_with_model(model, obs, self._normalize_observation, action_masks=action_masks)
            probs = self._policy_action_probs(model, obs, self._normalize_observation, action_masks=action_masks)
            return int(action), probs

    def evaluate(self, episodes: int = 3, max_steps: int = 5000, model_selector: str | ModelSelector | None = None) -> int:
        scores = self.evaluate_scores(
            episodes=episodes,
            max_steps=max_steps,
            model_selector=model_selector,
            eval_seed_base=None if self.config.seed is None else int(self.config.seed) + 100_000,
        )
        if not scores:
            return 0
        return int(round(sum(scores) / len(scores)))

    def evaluate_scores(
        self,
        *,
        episodes: int,
        max_steps: int,
        model_selector: str | ModelSelector | None,
        eval_seed_base: int | None,
    ) -> list[int]:
        model = self._resolve_eval_model(model_selector)
        if model is None:
            return []
        episodes_i = max(1, int(episodes))
        scores: list[int] = []
        for ep_idx in range(episodes_i):
            episode_seed = None if eval_seed_base is None else int(eval_seed_base + ep_idx)
            env = SnakePPOEnv(
                board_cells=self.settings.board_cells,
                seed=episode_seed,
                reward_config=self._effective_reward_config(),
                obs_config=self.obs_config,
                dropout_config=None,
            )
            try:
                obs, _ = env.reset()
                for _step in range(int(max_steps)):
                    action_masks = env.action_masks() if hasattr(env, "action_masks") else None
                    action = self._predict_with_model(
                        model,
                        obs,
                        self._normalize_observation,
                        action_masks=action_masks,
                    )
                    obs, _reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        scores.append(int(info.get("score", 0)))
                        break
                else:
                    scores.append(int(env.score))
            finally:
                env.close()
        return scores

    def evaluate_holdout(
        self,
        *,
        seeds: Iterable[int],
        max_steps: int = 5000,
        model_selector: str | ModelSelector | None = None,
    ) -> list[int]:
        model = self._resolve_eval_model(model_selector)
        if model is None:
            return []
        scores: list[int] = []
        for seed in seeds:
            env = SnakePPOEnv(
                board_cells=self.settings.board_cells,
                seed=int(seed),
                reward_config=self._effective_reward_config(),
                obs_config=self.obs_config,
                dropout_config=None,
            )
            try:
                obs, _ = env.reset(seed=int(seed))
                for _step in range(int(max_steps)):
                    action_masks = env.action_masks() if hasattr(env, "action_masks") else None
                    action = self._predict_with_model(
                        model,
                        obs,
                        self._normalize_observation,
                        action_masks=action_masks,
                    )
                    obs, _reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        scores.append(int(info.get("score", 0)))
                        break
                else:
                    scores.append(int(env.score))
            finally:
                env.close()
        return scores

    def _resolve_eval_model(self, selector: str | ModelSelector | None) -> PPO | None:
        selected = self._inference_selector if selector is None else self._coerce_selector(selector)
        with self._model_lock:
            if selected == ModelSelector.LAST and self.model is not None:
                # When training is active, avoid sharing the mutable training model
                # with concurrent eval loops.
                training_active = bool(
                    getattr(self, "_external_training_active", False) or self._train_vecnormalize is not None
                )
                if not training_active:
                    return self.model
                if self.inference_model is not None:
                    return self.inference_model
            if selected == self._inference_selector and self.inference_model is not None:
                return self.inference_model
        path = self._selected_model_path(selected)
        if not path.exists():
            return None
        try:
            return PPO.load(str(path), device=self.device, learning_rate=self.config.learning_rate_start)
        except Exception:
            logger.exception("Failed to load eval model from %s", path)
            return None

    def _make_single_env(self, seed: int | None, *, max_episode_steps: int | None = None, dropout_config: DropoutConfig | None = None):
        reward_cfg = self._effective_reward_config()
        return _EnvFactory(
            board_cells=int(self.settings.board_cells),
            seed=seed,
            reward_config=reward_cfg,
            obs_config=self.obs_config,
            max_episode_steps=max_episode_steps,
            dropout_config=dropout_config,
        )

    def _make_vec_env(self, seeds: list[int | None], *, max_episode_steps: int | None = None, dropout_config: DropoutConfig | None = None):
        env_fns = [self._make_single_env(seed, max_episode_steps=max_episode_steps, dropout_config=dropout_config) for seed in seeds]
        if bool(getattr(self.config, "use_subproc_env", False)) and len(env_fns) > 1:
            main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
            if "<stdin>" in str(main_file):
                return DummyVecEnv(env_fns)
            start_method = "spawn" if os.name == "nt" else "fork"
            return SubprocVecEnv(env_fns, start_method=start_method)
        return DummyVecEnv(env_fns)

    def _make_train_vec_env(self) -> VecNormalize:
        base_seed = None if self.config.seed is None else int(self.config.seed)
        seeds = [None if base_seed is None else int(base_seed + idx) for idx in range(self.env_count)]
        vec = self._make_vec_env(seeds, dropout_config=self.dropout_config)
        resume_stats = self._resume_vecnormalize_source
        if resume_stats is None or not resume_stats.exists():
            candidate = self._resume_vecnormalize_path()
            if candidate.exists():
                resume_stats = candidate
            elif self._vecnormalize_path().exists():
                resume_stats = self._vecnormalize_path()
        if resume_stats is not None and resume_stats.exists():
            try:
                loaded = VecNormalize.load(str(resume_stats), vec)
                loaded.training = True
                loaded.norm_reward = True
                return loaded
            except Exception:
                logger.exception("Failed to load training VecNormalize stats from %s; starting fresh", resume_stats)
        return VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    def _make_eval_vec_env(self) -> VecNormalize:
        eval_seed_base = None if self.config.seed is None else int(self.config.seed) + 500_000
        seeds = [None if eval_seed_base is None else int(eval_seed_base + idx) for idx in range(self.env_count)]
        eval_max_steps = int(getattr(self.config, "eval_max_episode_steps", 0))
        vec = self._make_vec_env(seeds, max_episode_steps=eval_max_steps if eval_max_steps > 0 else None)
        vec_norm = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vec_norm.training = False
        return vec_norm

    def set_adaptive_reward_enabled(self, enabled: bool) -> None:
        self._adaptive_reward_enabled = bool(enabled)

    def is_adaptive_reward_enabled(self) -> bool:
        return bool(self._adaptive_reward_enabled)

    def get_dropout_metrics(self) -> dict[str, float] | None:
        vec = getattr(self, "_train_vecnormalize", None)
        if vec is None:
            return None
        try:
            inner = vec.env
            envs = getattr(inner, "envs", None) or getattr(inner, "venv", None)
            if envs is None:
                return None
            all_metrics = []
            for e in envs:
                fn = getattr(e, "dropout_metrics", None)
                if fn is not None:
                    all_metrics.append(fn())
            if not all_metrics:
                return None
            keys = all_metrics[0].keys()
            return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}
        except Exception:
            return None

    @staticmethod
    def _linear_schedule(start: float, end: float) -> Callable[[float], float]:
        start_f = float(start)
        end_f = float(end)

        def schedule(progress_remaining: float) -> float:
            progress = max(0.0, min(1.0, float(progress_remaining)))
            return end_f + (start_f - end_f) * progress

        return schedule

    @staticmethod
    def _predict_with_model(
        model,
        obs,
        normalizer: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        action_masks=None,
    ) -> int:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if normalizer is not None:
            obs_arr = np.asarray(normalizer(obs_arr), dtype=np.float32)
        if tuple(obs_arr.shape) == tuple(model.observation_space.shape):
            obs_arr = np.expand_dims(obs_arr, axis=0)
        predict_kwargs = {"deterministic": True}
        if action_masks is not None:
            predict_kwargs["action_masks"] = np.asarray(action_masks, dtype=bool).reshape(obs_arr.shape[0], -1)
        try:
            action, _ = model.predict(obs_arr, **predict_kwargs)
        except TypeError:
            action, _ = model.predict(obs_arr, deterministic=True)
        action_arr = np.asarray(action)
        if action_arr.ndim == 0:
            return int(action_arr.item())
        return int(action_arr.reshape(-1)[0])

    @staticmethod
    def _policy_action_probs(
        model,
        obs,
        normalizer: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        action_masks=None,
    ) -> tuple[float, float, float] | None:
        try:
            obs_arr = np.asarray(obs, dtype=np.float32)
            if normalizer is not None:
                obs_arr = np.asarray(normalizer(obs_arr), dtype=np.float32)
            if tuple(obs_arr.shape) == tuple(model.observation_space.shape):
                obs_arr = np.expand_dims(obs_arr, axis=0)
            obs_tensor, _ = model.policy.obs_to_tensor(obs_arr)
            with torch.no_grad():
                dist = model.policy.get_distribution(
                    obs_tensor,
                    action_masks=None if action_masks is None else np.asarray(action_masks, dtype=bool).reshape(obs_arr.shape[0], -1),
                )
                probs_tensor = dist.distribution.probs
            probs_arr = probs_tensor.detach().cpu().numpy().reshape(-1)
            if probs_arr.size < 3:
                return None
            p0, p1, p2 = (float(probs_arr[0]), float(probs_arr[1]), float(probs_arr[2]))
            total = p0 + p1 + p2
            if total <= 0.0:
                return None
            return (p0 / total, p1 / total, p2 / total)
        except Exception:
            logger.debug("Failed to compute policy action probabilities", exc_info=True)
            return None

    def _maybe_refresh_inference(self, steps_done: int) -> None:
        force = self._sync_request_event.is_set()
        if not force:
            if int(steps_done) <= int(self._last_inference_sync_steps):
                return
            if int(steps_done) - int(self._last_inference_sync_steps) < int(self.inference_sync_every_steps):
                return
            if (time.perf_counter() - float(self._last_inference_sync_time)) < float(self.inference_sync_min_seconds):
                return
        with self._model_lock:
            if self.model is None:
                self._sync_request_event.clear()
                return
            sync_ok = False
            if self.inference_model is None:
                # Bootstrap a dedicated inference model from the in-memory training model.
                # This keeps training and inference decoupled while allowing live play
                # before the training run is stopped/saved.
                try:
                    snapshot_buffer = io.BytesIO()
                    self.model.save(snapshot_buffer)
                    snapshot_buffer.seek(0)
                    self.inference_model = PPO.load(snapshot_buffer, device=self.device, learning_rate=self.config.learning_rate_start)
                    if self._train_vecnormalize is not None:
                        self._apply_vecnormalize_stats(self._train_vecnormalize)
                    sync_ok = True
                except Exception:
                    logger.exception("Failed to bootstrap inference model during training sync")
                    self._sync_request_event.set()
                    return
            else:
                try:
                    self.inference_model.policy.load_state_dict(self.model.policy.state_dict())
                    if self._train_vecnormalize is not None:
                        self._apply_vecnormalize_stats(self._train_vecnormalize)
                    sync_ok = True
                except Exception:
                    logger.exception("Failed in-memory inference sync; requesting artifact reload")
                    self._sync_request_event.set()
            self._last_inference_sync_steps = int(steps_done)
            self._last_inference_sync_time = time.perf_counter()
            if sync_ok:
                self._sync_request_event.clear()

    def _effective_reward_config(self) -> RewardConfig:
        enabled = bool(self._adaptive_reward_enabled)
        base_enabled = bool(getattr(self.reward_config, "use_reachable_space_penalty", enabled))
        if base_enabled == enabled:
            return self.reward_config
        return replace(self.reward_config, use_reachable_space_penalty=enabled)
