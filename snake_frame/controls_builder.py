from __future__ import annotations

from dataclasses import dataclass

import pygame

from .panel_layout import build_panel_layout, build_right_panel_layout
from .panel_ui import PanelControls
from .settings import Settings
from .theme import get_design_tokens, get_theme
from .ui import Button, NumericInput


@dataclass(frozen=True)
class ControlsBuildResult:
    graph_rect: pygame.Rect
    training_graph_rect: pygame.Rect
    run_graph_rect: pygame.Rect
    training_header_y: int
    training_badges_y: int
    run_header_y: int
    run_badges_y: int
    panel_controls: PanelControls
    generations_input: NumericInput
    btn_train_start: Button
    btn_train_stop: Button
    btn_save: Button
    btn_load: Button
    btn_delete: Button
    btn_game_start: Button
    btn_game_stop: Button
    btn_restart: Button
    btn_options: Button
    btn_options_close: Button
    btn_adaptive_toggle: Button
    btn_space_strategy_toggle: Button
    btn_tail_trend_toggle: Button
    btn_dropout_toggle: Button
    btn_theme_cycle: Button
    btn_board_bg_cycle: Button
    btn_snake_style_cycle: Button
    btn_fog_cycle: Button
    btn_speed_down: Button
    btn_speed_up: Button
    btn_eval_suite: Button
    btn_eval_mode_ppo: Button
    btn_eval_mode_controller: Button
    btn_eval_holdout: Button
    btn_debug_toggle: Button
    btn_reachable_toggle: Button
    btn_diagnostics: Button
    btn_tab_train: Button
    btn_tab_run: Button
    btn_tab_debug: Button


def build_controls(
    settings: Settings,
    *,
    min_graph_height: int,
    max_graph_height: int,
    graph_margin: int,
    graph_top: int,
    control_row_height: int,
    control_gap: int,
    status_line_height: int,
    status_line_count: int = 10,
) -> ControlsBuildResult:
    compact = int(settings.window_height_px or settings.window_px) < int(get_design_tokens(settings.theme_name).spacing.graph_margin_compact_threshold)
    tokens = get_design_tokens(getattr(settings, "theme_name", ""), compact=compact)
    theme = get_theme(getattr(settings, "theme_name", ""))
    input_top_offset = int(tokens.spacing.input_top_offset)
    input_height = int(tokens.components.input_height)
    input_to_buttons_gap = int(tokens.spacing.input_to_buttons_gap)
    button_row_height = max(int(control_row_height), int(tokens.components.button_row_height))
    button_gap = max(int(control_gap), int(tokens.spacing.section_gap // 2))
    controls_stack_height = (
        input_top_offset  # label->input top
        + input_height  # input height
        + input_to_buttons_gap  # input->first row spacing
        + (11 * int(button_row_height))  # train/save/delete/game/restart/options/adaptive/space/theme/board-bg/snake rows
        + (15 * int(button_gap))  # gaps between rows (increased for more spacing)
        + int(tokens.spacing.status_top_gap)  # game buttons->status spacing
    )
    reserve_for_controls_and_status = int(controls_stack_height + (int(status_line_count) * int(status_line_height)))
    controls_layout = build_panel_layout(
        settings,
        min_graph_height=min_graph_height,
        max_graph_height=max_graph_height,
        graph_margin=graph_margin,
        graph_top=graph_top,
        control_row_height=button_row_height,
        control_gap=button_gap,
        reserve_for_controls_and_status=reserve_for_controls_and_status,
        panel_x=0,
        panel_width=int(settings.left_panel_px),
    )
    controls_top = int(max(int(tokens.spacing.left_controls_top_padding), graph_top - int(tokens.spacing.left_controls_raise_px)))
    # Use fixed right panel layout for stable KPI dashboard
    right_layout = build_right_panel_layout(settings)
    right_inner_x = right_layout.inner_x
    right_inner_w = right_layout.inner_width
    training_graph_rect = right_layout.training_graph_rect
    run_graph_rect = right_layout.run_graph_rect
    graph_rect = pygame.Rect(run_graph_rect)
    generations_input = NumericInput(
        pygame.Rect(controls_layout.x, controls_top + input_top_offset, controls_layout.width, input_height),
        "500000",
    )

    y = int(controls_top + input_top_offset + input_height + input_to_buttons_gap)
    use_stacked_pairs = bool(controls_layout.half_width < 140)
    if use_stacked_pairs:
        btn_train_start = Button(
            "Start Train",
            pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
            bg=theme.train_start_bg,
            bg_hover=theme.train_start_hover,
        )
        y += int(controls_layout.row_height + controls_layout.gap)
        btn_train_stop = Button(
            "Stop Train",
            pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        )
    else:
        btn_train_start = Button(
            "Start Train",
            pygame.Rect(controls_layout.x, y, controls_layout.half_width, controls_layout.row_height),
            bg=theme.train_start_bg,
            bg_hover=theme.train_start_hover,
        )
        btn_train_stop = Button(
            "Stop Train",
            pygame.Rect(
                controls_layout.x + controls_layout.half_width + controls_layout.gap,
                y,
                controls_layout.half_width,
                controls_layout.row_height,
            ),
        )
    y += int(controls_layout.row_height + controls_layout.gap)
    if use_stacked_pairs:
        btn_save = Button(
            "Save",
            pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
            bg=theme.save_bg,
            bg_hover=theme.save_hover,
        )
        y += int(controls_layout.row_height + controls_layout.gap)
        btn_load = Button(
            "Load",
            pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
            bg=theme.load_bg,
            bg_hover=theme.load_hover,
        )
    else:
        btn_save = Button(
            "Save",
            pygame.Rect(controls_layout.x, y, controls_layout.half_width, controls_layout.row_height),
            bg=theme.save_bg,
            bg_hover=theme.save_hover,
        )
        btn_load = Button(
            "Load",
            pygame.Rect(
                controls_layout.x + controls_layout.half_width + controls_layout.gap,
                y,
                controls_layout.half_width,
                controls_layout.row_height,
            ),
            bg=theme.load_bg,
            bg_hover=theme.load_hover,
        )

    y += int(controls_layout.row_height + controls_layout.gap)
    btn_delete = Button(
        "Delete",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.delete_bg,
        bg_hover=theme.delete_hover,
    )

    y += int(controls_layout.row_height + controls_layout.gap)
    if use_stacked_pairs:
        btn_game_start = Button(
            "Start Game",
            pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
            bg=theme.game_start_bg,
            bg_hover=theme.game_start_hover,
        )
        y += int(controls_layout.row_height + controls_layout.gap)
        btn_game_stop = Button(
            "Stop Game",
            pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
            bg=theme.game_stop_bg,
            bg_hover=theme.game_stop_hover,
        )
    else:
        btn_game_start = Button(
            "Start Game",
            pygame.Rect(controls_layout.x, y, controls_layout.half_width, controls_layout.row_height),
            bg=theme.game_start_bg,
            bg_hover=theme.game_start_hover,
        )
        btn_game_stop = Button(
            "Stop Game",
            pygame.Rect(
                controls_layout.x + controls_layout.half_width + controls_layout.gap,
                y,
                controls_layout.half_width,
                controls_layout.row_height,
            ),
            bg=theme.game_stop_bg,
            bg_hover=theme.game_stop_hover,
        )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_restart = Button(
        "Restart",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.restart_bg,
        bg_hover=theme.restart_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_options = Button(
        "Options",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_adaptive_toggle = Button(
        "Adaptive Reward: ON",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_positive_bg,
        bg_hover=theme.toggle_positive_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_space_strategy_toggle = Button(
        "Space Strategy: ON",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_positive_bg,
        bg_hover=theme.toggle_positive_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_tail_trend_toggle = Button(
        "Tail Trend: ON",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_positive_bg,
        bg_hover=theme.toggle_positive_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_dropout_toggle = Button(
        "Full Mask: OFF",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_negative_bg,
        bg_hover=theme.toggle_negative_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_theme_cycle = Button(
        f"Theme: {theme.name}",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_board_bg_cycle = Button(
        "Board BG: Background",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_snake_style_cycle = Button(
        "Snake: Topdown 3D",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_warm_bg,
        bg_hover=theme.toggle_warm_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_fog_cycle = Button(
        "Fog: Off",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )
    btn_speed_down = Button(
        "Live Speed -",
        pygame.Rect(controls_layout.x, y, controls_layout.half_width, controls_layout.row_height),
        bg=theme.toggle_warm_bg,
        bg_hover=theme.toggle_warm_hover,
    )
    btn_speed_up = Button(
        "Live Speed +",
        pygame.Rect(
            controls_layout.x + controls_layout.half_width + controls_layout.gap,
            y,
            controls_layout.half_width,
            controls_layout.row_height,
        ),
        bg=theme.toggle_positive_bg,
        bg_hover=theme.toggle_positive_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_eval_suite = Button(
        "Run Eval Suite (PPO + Controller)",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_positive_bg,
        bg_hover=theme.toggle_positive_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_eval_mode_ppo = Button(
        "Set Eval: PPO Only",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_eval_mode_controller = Button(
        "Set Eval: Controller ON",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_warm_bg,
        bg_hover=theme.toggle_warm_hover,
    )
    y += int(controls_layout.row_height + controls_layout.gap)
    btn_eval_holdout = Button(
        "Eval Holdout",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_positive_bg,
        bg_hover=theme.toggle_positive_hover,
    )
    btn_options_close = Button(
        "Close",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.delete_bg,
        bg_hover=theme.delete_hover,
    )
    btn_diagnostics = Button(
        "Diagnostics Bundle",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )
    # Debug toggles - positioned for Options modal only (not in right panel)
    btn_debug_toggle = Button(
        "Debug: OFF",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.debug_off_bg,
        bg_hover=theme.debug_off_hover,
    )
    btn_reachable_toggle = Button(
        "Reach: OFF",
        pygame.Rect(controls_layout.x, y, controls_layout.width, controls_layout.row_height),
        bg=theme.reach_off_bg,
        bg_hover=theme.reach_off_hover,
    )
    tab_gap = 8
    tab_y = int(right_layout.utility_row_y)
    tab_h = int(right_layout.utility_row_height)
    tab_w = max(90, int((right_inner_w - (2 * tab_gap)) // 3))
    btn_tab_train = Button(
        "Train",
        pygame.Rect(right_inner_x, tab_y, tab_w, tab_h),
        bg=theme.toggle_positive_bg,
        bg_hover=theme.toggle_positive_hover,
    )
    btn_tab_run = Button(
        "Run",
        pygame.Rect(right_inner_x + tab_w + tab_gap, tab_y, tab_w, tab_h),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )
    btn_tab_debug = Button(
        "Debug",
        pygame.Rect(right_inner_x + ((tab_w + tab_gap) * 2), tab_y, tab_w, tab_h),
        bg=theme.toggle_info_bg,
        bg_hover=theme.toggle_info_hover,
    )

    panel_controls = PanelControls(
        generations_input=generations_input,
        btn_train_start=btn_train_start,
        btn_train_stop=btn_train_stop,
        btn_save=btn_save,
        btn_load=btn_load,
        btn_delete=btn_delete,
        btn_game_start=btn_game_start,
        btn_game_stop=btn_game_stop,
        btn_restart=btn_restart,
        btn_options=btn_options,
        btn_options_close=btn_options_close,
        btn_adaptive_toggle=btn_adaptive_toggle,
        btn_space_strategy_toggle=btn_space_strategy_toggle,
        btn_tail_trend_toggle=btn_tail_trend_toggle,
        btn_dropout_toggle=btn_dropout_toggle,
        btn_theme_cycle=btn_theme_cycle,
        btn_board_bg_cycle=btn_board_bg_cycle,
        btn_snake_style_cycle=btn_snake_style_cycle,
        btn_fog_cycle=btn_fog_cycle,
        btn_speed_down=btn_speed_down,
        btn_speed_up=btn_speed_up,
        btn_eval_suite=btn_eval_suite,
        btn_eval_mode_ppo=btn_eval_mode_ppo,
        btn_eval_mode_controller=btn_eval_mode_controller,
        btn_eval_holdout=btn_eval_holdout,
        btn_debug_toggle=btn_debug_toggle,
        btn_reachable_toggle=btn_reachable_toggle,
        btn_diagnostics=btn_diagnostics,
        btn_tab_train=btn_tab_train,
        btn_tab_run=btn_tab_run,
        btn_tab_debug=btn_tab_debug,
    )
    return ControlsBuildResult(
        graph_rect=graph_rect,
        training_graph_rect=training_graph_rect,
        run_graph_rect=run_graph_rect,
        training_header_y=right_layout.training_header_y,
        training_badges_y=right_layout.training_badges_y,
        run_header_y=right_layout.run_header_y,
        run_badges_y=right_layout.run_badges_y,
        panel_controls=panel_controls,
        btn_debug_toggle=btn_debug_toggle,
        btn_reachable_toggle=btn_reachable_toggle,
        generations_input=generations_input,
        btn_train_start=btn_train_start,
        btn_train_stop=btn_train_stop,
        btn_save=btn_save,
        btn_load=btn_load,
        btn_delete=btn_delete,
        btn_game_start=btn_game_start,
        btn_game_stop=btn_game_stop,
        btn_restart=btn_restart,
        btn_options=btn_options,
        btn_options_close=btn_options_close,
        btn_adaptive_toggle=btn_adaptive_toggle,
        btn_space_strategy_toggle=btn_space_strategy_toggle,
        btn_tail_trend_toggle=btn_tail_trend_toggle,
        btn_dropout_toggle=btn_dropout_toggle,
        btn_theme_cycle=btn_theme_cycle,
        btn_board_bg_cycle=btn_board_bg_cycle,
        btn_snake_style_cycle=btn_snake_style_cycle,
        btn_fog_cycle=btn_fog_cycle,
        btn_speed_down=btn_speed_down,
        btn_speed_up=btn_speed_up,
        btn_eval_suite=btn_eval_suite,
        btn_eval_mode_ppo=btn_eval_mode_ppo,
        btn_eval_mode_controller=btn_eval_mode_controller,
        btn_eval_holdout=btn_eval_holdout,
        btn_diagnostics=btn_diagnostics,
        btn_tab_train=btn_tab_train,
        btn_tab_run=btn_tab_run,
        btn_tab_debug=btn_tab_debug,
    )
