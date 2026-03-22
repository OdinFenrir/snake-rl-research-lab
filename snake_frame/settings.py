from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PpoConfig:
    env_count: int = 8
    use_subproc_env: bool = True
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 8
    gamma: float = 0.995
    gae_lambda: float = 0.95
    learning_rate_start: float = 2.0e-4
    learning_rate_end: float = 1.0e-5
    clip_range: float = 0.15
    target_kl: float | None = 0.015
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.0008
    policy_net_arch: tuple[int, ...] = (512, 512)  # Increased for NewTest3.
    policy_net_arch_pi: tuple[int, ...] | None = None
    policy_net_arch_vf: tuple[int, ...] | None = None
    seed: int | None = None
    eval_freq_steps: int = 200_000
    eval_episodes: int = 5
    eval_max_episode_steps: int = 5_000
    checkpoint_freq_steps: int = 100_000
    use_stop_on_no_improvement: bool = False
    no_improvement_evals: int = 6
    min_evals_before_stop: int = 5


@dataclass(frozen=True)
class RewardConfig:
    eat_reward: float = 30.0
    death_penalty: float = 30.0
    living_penalty: float = 0.02
    approach_food_reward: float = 0.05
    retreat_food_penalty: float = 0.05
    starvation_penalty: float = 10.0
    low_safe_options_penalty: float = 0.05
    high_safe_options_bonus: float = 0.03
    use_reachable_space_penalty: bool = True
    trap_penalty_threshold: float = 0.15
    trap_penalty: float = 0.3
    endgame_length_ratio_start: float = 0.2
    endgame_trap_penalty_scale: float = 1.5
    board_starvation_factor: int = 2
    fill_board_bonus: float = 100.0


@dataclass(frozen=True)
class ObsConfig:
    use_extended_features: bool = False
    use_path_features: bool = False
    use_tail_path_features: bool = False
    use_free_space_features: bool = False
    use_tail_trend_features: bool = False


@dataclass(frozen=True)
class DropoutConfig:
    enabled: bool = False
    p_start: float = 0.0
    p_max: float = 0.30
    warmup_steps: int = 1_000_000
    drop_body: bool = True
    drop_wall: bool = True


@dataclass(frozen=True)
class DynamicControlConfig:
    enable_dynamic_control: bool = True
    cycle_window_steps: int = 36
    cycle_repeat_threshold: int = 4
    no_progress_steps_escape: int = 64
    no_progress_steps_space_fill: int = 128
    risk_recovery_window: int = 20
    mode_switch_cooldown_steps: int = 14
    space_fill_tail_reachable_bonus: float = 6000.0
    space_fill_tail_unreachable_penalty: float = 5500.0
    space_fill_reachable_margin_weight: float = 95.0
    space_fill_capacity_shortfall_penalty: float = 300.0
    space_fill_wall_distance_weight: float = 3.0
    space_fill_food_distance_weight: float = 0.02
    space_fill_zigzag_penalty: float = 8.0
    space_fill_low_liberty_penalty: float = 200.0
    loop_escape_base_steps: int = 10
    loop_escape_max_steps: int = 36
    loop_escape_cooldown_steps: int = 36
    loop_escape_starvation_trigger_ratio: float = 0.55
    loop_escape_stall_window: int = 32
    ppo_confidence_trust_threshold: float = 0.90
    ppo_confidence_trust_food_pressure_max: float = 0.72
    ppo_confidence_trust_min_free_ratio: float = 0.24
    ppo_confidence_trust_min_safe_options: int = 2
    ppo_high_conf_override_guard_threshold: float = 0.99
    ppo_high_conf_override_guard_food_pressure_max: float = 0.6
    ppo_high_conf_override_guard_min_safe_options: int = 2
    ppo_high_conf_override_guard_min_shortfall_gain: int = 1
    enable_risk_switch_guard: bool = False
    risk_switch_guard_confidence_min: float = 0.90
    risk_switch_guard_min_safe_options: int = 1
    risk_switch_guard_min_shortfall_gain: int = 2
    risk_switch_guard_no_progress_margin: int = 20
    risk_switch_guard_allow_narrow_corridor: bool = False
    risk_switch_guard_narrow_confidence_min: float = 0.97
    risk_switch_guard_narrow_min_no_progress_steps: int = 16
    risk_switch_guard_narrow_no_progress_margin: int = 0
    enable_pocket_exit_guard: bool = False
    pocket_exit_guard_max_safe_options: int = 2
    pocket_exit_guard_min_no_progress_steps: int = 32
    pocket_exit_guard_min_food_pressure: float = 0.25
    pocket_exit_guard_min_shortfall_gain: int = 2
    enable_pre_no_exit_guard: bool = False
    pre_no_exit_guard_max_safe_options: int = 2
    pre_no_exit_guard_min_no_progress_steps: int = 24
    pre_no_exit_guard_no_exit_safe_options: int = 1
    pre_no_exit_guard_min_shortfall_gain: int = 1
    pre_no_exit_guard_require_collapsing_safe_options: bool = True
    pre_no_exit_guard_require_no_exit_signal: bool = True
    ppo_open_field_trust_food_pressure_max: float = 0.35
    narrow_corridor_trigger_steps: int = 6
    dynamic_warmup_steps: int = 120
    enable_learned_arbiter: bool = True
    arbiter_threshold: float = 0.56
    arbiter_learning_rate: float = 0.04
    arbiter_l2: float = 1.0e-4
    enable_tactic_memory: bool = True
    tactic_memory_max_clusters: int = 96
    tactic_memory_merge_radius: float = 0.18
    tactic_memory_weight: float = 120.0
    tactic_memory_adaptive_merge: bool = False
    tactic_memory_merge_radius_crowded: float = 0.22
    tactic_memory_merge_radius_open: float = 0.14
    tactic_memory_merge_ratio_low: float = 0.35
    tactic_memory_merge_ratio_high: float = 0.65
    lookahead_depth: int = 3
    lookahead_weight: float = 220.0


@dataclass
class Settings:
    board_cells: int = 20
    cell_px: int = 54
    fps: int = 60
    ticks_per_move: int = 5
    left_panel_px: int = 520
    right_panel_px: int = 760
    agent_safety_override: bool = True
    window_height_px: int | None = None
    window_borderless: bool = True
    layout_preset: str = "standard"
    theme_name: str = "retro_forest_noir"
    ui_scale: float = 0.95
    min_cell_px: int = 24
    max_cell_px: int = 72
    min_left_panel_px: int = 330
    min_right_panel_px: int = 520
    left_panel_ratio: float = 0.37
    dynamic_control: DynamicControlConfig = field(default_factory=DynamicControlConfig)

    def __post_init__(self) -> None:
        if self.window_height_px is None:
            self.window_height_px = int(self.window_px)

    def apply_window_size(self, width: int, height: int) -> None:
        target_w = max(1024, int(width))
        target_h = max(600, int(height))
        board_max_by_height = target_h
        board_max_by_width = target_w - int(self.min_left_panel_px) - int(self.min_right_panel_px)
        board_px = max(
            int(self.min_cell_px) * int(self.board_cells),
            min(int(board_max_by_height), int(board_max_by_width)),
        )
        cell_px = max(
            int(self.min_cell_px),
            min(int(self.max_cell_px), int(board_px // max(1, int(self.board_cells)))),
        )
        self.cell_px = int(cell_px)
        board_px = int(self.window_px)
        remaining_w = max(1, int(target_w - board_px))
        left = int(round(float(remaining_w) * float(self.left_panel_ratio)))
        right = int(remaining_w - left)
        left = max(int(self.min_left_panel_px), left)
        right = max(int(self.min_right_panel_px), right)
        overflow = (left + right + board_px) - target_w
        if overflow > 0:
            shrink_right = min(overflow, max(0, right - int(self.min_right_panel_px)))
            right -= shrink_right
            overflow -= shrink_right
            if overflow > 0:
                shrink_left = min(overflow, max(0, left - int(self.min_left_panel_px)))
                left -= shrink_left
        self.left_panel_px = int(left)
        self.right_panel_px = int(right)
        self.window_height_px = int(target_h)

    @property
    def window_px(self) -> int:
        return self.board_cells * self.cell_px

    @property
    def window_width_px(self) -> int:
        return self.left_panel_px + self.window_px + self.right_panel_px

    @property
    def board_offset_x(self) -> int:
        return int(self.left_panel_px)

    @property
    def board_offset_y(self) -> int:
        return max(0, int((int(self.window_height_px or self.window_px) - self.window_px) // 2))

    @property
    def right_panel_offset_x(self) -> int:
        return int(self.left_panel_px + self.window_px)


def ppo_profile_config(profile: str, *, seed: int | None = None) -> PpoConfig:
    key = str(profile or "").strip().lower()
    if key in ("", "app", "default"):
        return PpoConfig(seed=seed, use_subproc_env=False)
    if key == "research_long":
        return PpoConfig(
            env_count=16,
            n_steps=2048,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate_start=3e-4,
            learning_rate_end=1e-5,
            clip_range=0.2,
            ent_coef_start=0.02,
            ent_coef_end=5e-4,
            policy_net_arch=(256, 256),
            seed=seed,
            eval_freq_steps=50_000,
            eval_episodes=20,
            checkpoint_freq_steps=50_000,
            use_stop_on_no_improvement=True,
            no_improvement_evals=6,
            min_evals_before_stop=5,
        )
    if key == "fast":
        return PpoConfig(
            env_count=8,
            use_subproc_env=False,
            n_steps=256,
            batch_size=256,
            n_epochs=2,
            seed=seed,
            eval_freq_steps=10_000,
            eval_episodes=10,
            checkpoint_freq_steps=10_000,
            use_stop_on_no_improvement=False,
            no_improvement_evals=20,
            min_evals_before_stop=10,
        )
    return PpoConfig(seed=seed)
