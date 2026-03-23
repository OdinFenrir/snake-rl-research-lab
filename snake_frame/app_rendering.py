from __future__ import annotations

import logging
import pygame

from .panel_ui import PanelRenderData

logger = logging.getLogger(__name__)


def draw(app) -> None:
    app.surface.fill(app.theme.surface_bg)
    draw_board_column_background(app)
    app._apply_ui_state_model()
    app.gameplay.set_space_strategy_enabled(app.app_state.space_strategy_enabled)
    control_policy = app._derive_control_policy()
    adaptive_enabled = bool(
        getattr(app.agent, "is_adaptive_reward_enabled", lambda: True)()
    )
    app._set_toggle_button_visual(
        app.btn_adaptive_toggle,
        label="Reward Shaping",
        enabled=adaptive_enabled,
        on_color=(app.theme.toggle_positive_bg, app.theme.toggle_positive_hover),
        off_color=(app.theme.toggle_negative_bg, app.theme.toggle_negative_hover),
    )
    space_strategy_enabled = bool(app.gameplay.is_space_strategy_enabled())
    app._set_toggle_button_visual(
        app.btn_space_strategy_toggle,
        label="Safe Space Bias",
        enabled=space_strategy_enabled,
        on_color=(app.theme.toggle_positive_bg, app.theme.toggle_positive_hover),
        off_color=(app.theme.toggle_negative_bg, app.theme.toggle_negative_hover),
    )
    tail_trend_enabled = bool(getattr(app.app_state, 'tail_trend_enabled', True))
    app._set_toggle_button_visual(
        app.btn_tail_trend_toggle,
        label="Tail Trend Assist",
        enabled=tail_trend_enabled,
        on_color=(app.theme.toggle_positive_bg, app.theme.toggle_positive_hover),
        off_color=(app.theme.toggle_negative_bg, app.theme.toggle_negative_hover),
    )
    dropout_enabled = bool(app.app_state.dropout_enabled)
    app._set_toggle_button_visual(
        app.btn_dropout_toggle,
        label="Full Mask",
        enabled=dropout_enabled,
        on_color=(app.theme.toggle_positive_bg, app.theme.toggle_positive_hover),
        off_color=(app.theme.toggle_negative_bg, app.theme.toggle_negative_hover),
    )
    debug_enabled = bool(app.app_state.debug_overlay)
    app._set_toggle_button_visual(
        app.btn_debug_toggle,
        label="Debug Overlay",
        enabled=debug_enabled,
        on_color=(app.theme.toggle_info_bg, app.theme.toggle_info_hover),
        off_color=(app.theme.debug_off_bg, app.theme.debug_off_hover),
    )
    reachable_enabled = bool(app.app_state.debug_reachable_overlay)
    app._set_toggle_button_visual(
        app.btn_reachable_toggle,
        label="Reachability Overlay",
        enabled=reachable_enabled,
        on_color=(app.theme.toggle_warm_bg, app.theme.toggle_warm_hover),
        off_color=(app.theme.reach_off_bg, app.theme.reach_off_hover),
    )
    app.btn_theme_cycle.label = f"Theme: {app.theme.name}"
    app.btn_board_bg_cycle.label = f"Board Style: {app.game.board_background_label()}"
    app.btn_snake_style_cycle.label = f"Snake Look: {app.game.snake_style_label()}"
    app.btn_fog_cycle.label = f"Fog Level: {app.game.fog_density_label()}"
    app.btn_speed_down.label = f"Slower (TPM {int(app.settings.ticks_per_move)})"
    app.btn_speed_up.label = "Faster"
    ppo_mode = str(getattr(app, "_holdout_eval_mode", "ppo_only")) == "ppo_only"
    app.btn_eval_mode_ppo.label = "Eval Uses PPO [ON]" if ppo_mode else "Eval Uses PPO"
    app.btn_eval_mode_controller.label = "Eval Uses Controller [ON]" if not ppo_mode else "Eval Uses Controller"
    holdout_eval = getattr(app, "holdout_eval", None)
    eval_snap = holdout_eval.snapshot() if holdout_eval is not None else None
    suite_active = bool(getattr(app, "_eval_suite_active", False))
    holdout_mode = str(getattr(app, "_holdout_eval_mode", "ppo_only")).strip().lower()
    holdout_mode_label = "CTRL" if holdout_mode == "controller_on" else "PPO"
    selector = "best"
    get_selector = getattr(app.agent, "get_model_selector", None)
    if callable(get_selector):
        try:
            selector = str(get_selector() or "best").strip().lower()
        except Exception:
            selector = "best"
    if suite_active:
        app.btn_eval_holdout.label = "Run Holdout Check (disabled during full eval)"
    elif eval_snap is not None and bool(eval_snap.active):
        app.btn_eval_holdout.label = f"Run Holdout Check ({int(eval_snap.completed)}/{int(eval_snap.total)})"
    else:
        app.btn_eval_holdout.label = f"Run Holdout Check ({holdout_mode_label}, {selector})"
    suite_phase = str(getattr(app, "_eval_suite_phase", "idle"))
    if suite_active and eval_snap is not None and bool(eval_snap.active):
        phase_label = "PPO" if suite_phase == "ppo" else "CTRL"
        app.btn_eval_suite.label = f"Run Full Evaluation ({phase_label} {int(eval_snap.completed)}/{int(eval_snap.total)})"
    else:
        app.btn_eval_suite.label = f"Run Full Evaluation (PPO + Controller, {selector})"
    app.btn_diagnostics.label = "Export Diagnostics"
    selected_tab = str(getattr(app.app_state, "right_panel_tab", "train")).strip().lower()
    if selected_tab not in ("train", "run", "debug"):
        selected_tab = "train"
    tab_palette_active = (app.theme.toggle_positive_bg, app.theme.toggle_positive_hover)
    tab_palette_idle = (app.theme.toggle_info_bg, app.theme.toggle_info_hover)
    app.btn_tab_train.label = "Train"
    app.btn_tab_train.bg, app.btn_tab_train.bg_hover = tab_palette_active if selected_tab == "train" else tab_palette_idle
    app.btn_tab_run.label = "Run"
    app.btn_tab_run.bg, app.btn_tab_run.bg_hover = tab_palette_active if selected_tab == "run" else tab_palette_idle
    app.btn_tab_debug.label = "Debug"
    app.btn_tab_debug.bg, app.btn_tab_debug.bg_hover = tab_palette_active if selected_tab == "debug" else tab_palette_idle
    left_status_lines = app.actions.build_status_lines()[:6]
    train_deaths = app.app_state.training_death_counts or {}
    training_status_lines = [
        f"Eval: {getattr(app.training.snapshot(), 'last_eval_score', None):.2f}" if getattr(app.training.snapshot(), "last_eval_score", None) is not None else "Eval: n/a",
        f"BestEval: {getattr(app.training.snapshot(), 'best_eval_score', None):.2f}" if getattr(app.training.snapshot(), "best_eval_score", None) is not None else "BestEval: n/a",
        f"Train deaths: {app._format_death_counts(train_deaths)}",
    ]
    telemetry = app.gameplay.telemetry_snapshot()
    run_status_lines = [
        f"Mode: {telemetry.current_mode}",
        f"Switch: {telemetry.last_switch_reason}",
        f"Interventions: {telemetry.interventions_total}/{telemetry.decisions_total}",
        f"Deaths: {app._format_death_counts({'wall': telemetry.deaths_wall, 'body': telemetry.deaths_body, 'starvation': telemetry.deaths_starvation, 'fill': telemetry.deaths_fill, 'other': telemetry.deaths_other})}",
    ]
    debug_status_lines = app._build_dynamic_status_lines() + app._build_settings_lines()
    panel_data = PanelRenderData(
        training_episode_scores=[int(v) for v in app.app_state.training_episode_scores],
        run_episode_scores=app._run_graph_scores(),
        training_graph_rect=pygame.Rect(app.training_graph_rect),
        run_graph_rect=pygame.Rect(app.run_graph_rect),
        training_graph_badges=app._build_training_graph_badges(),
        run_graph_badges=app._build_run_graph_badges(),
        left_status_lines=left_status_lines,
        training_status_lines=training_status_lines,
        run_status_lines=run_status_lines,
        debug_status_lines=debug_status_lines,
        training_header_y=app.training_header_y,
        training_badges_y=app.training_badges_y,
        run_header_y=app.run_header_y,
        run_badges_y=app.run_badges_y,
        selected_tab=selected_tab,
    )
    try:
        app.panel_renderer.draw(
            surface=app.surface,
            data=panel_data,
            controls=app.panel_controls,
        )
    except Exception:
        logger.exception("Panel renderer draw failed; using fallback banner")
        fallback = safe_render_text(app, "UI render fallback active", app.theme.banner_warn, small=True)
        app.surface.blit(fallback, (12, 10))
    try:
        app.game.draw(app.surface, app.font)
    except Exception:
        logger.exception("Game draw failed; using fallback board placeholder")
        rect = pygame.Rect(int(app.settings.board_offset_x), int(app.settings.board_offset_y), int(app.settings.window_px), int(app.settings.window_px))
        pygame.draw.rect(app.surface, app.theme.graph_bg, rect)
        text = safe_render_text(app, "Game render unavailable", app.theme.banner_warn, small=False)
        app.surface.blit(text, (rect.x + 12, rect.y + 12))
    draw_board_frame(app)
    draw_window_chrome(app)
    draw_runtime_banners(app, control_policy)
    draw_perf_overlay(app)
    if app.app_state.options_open:
        draw_options_window(app)
    if app.app_state.debug_overlay:
        app.gameplay.draw_debug_overlay(app.surface, app.small_font)
    if app.app_state.debug_reachable_overlay:
        app.gameplay.draw_reachable_overlay(app.surface, app.small_font)


def draw_board_frame(app) -> None:
    board_rect = pygame.Rect(
        int(app.settings.board_offset_x),
        int(app.settings.board_offset_y),
        int(app.settings.window_px),
        int(app.settings.window_px),
    )
    pygame.draw.rect(app.surface, app.theme.board_frame_border, board_rect, width=1, border_radius=6)


def draw_board_column_background(app) -> None:
    column_rect = pygame.Rect(
        int(app.settings.board_offset_x),
        0,
        int(app.settings.window_px),
        int(app.layout.window.height),
    )
    bg_surface = _board_column_background_surface(app, column_rect.width, column_rect.height)
    app.surface.blit(bg_surface, (column_rect.x, column_rect.y))


def _board_column_background_surface(app, width: int, height: int) -> pygame.Surface:
    key = (
        int(width),
        int(height),
        tuple(app.theme.panel_bg),
        tuple(app.theme.panel_bg_accent),
    )
    cached = getattr(app, "_board_column_bg_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    surf = pygame.Surface((max(1, int(width)), max(1, int(height))))
    if pygame.display.get_surface() is not None:
        surf = surf.convert()
    h = max(1, int(height))
    surf.fill(app.theme.panel_bg)
    for y in range(0, h, 4):
        t = float(y) / float(max(1, h - 1))
        shade = (
            int(app.theme.panel_bg_accent[0] * (1.0 - t) + app.theme.panel_bg[0] * t),
            int(app.theme.panel_bg_accent[1] * (1.0 - t) + app.theme.panel_bg[1] * t),
            int(app.theme.panel_bg_accent[2] * (1.0 - t) + app.theme.panel_bg[2] * t),
        )
        surf.fill(shade, (0, y, int(width), 1))
    app._board_column_bg_cache = (key, surf)
    return surf


def draw_window_chrome(app) -> None:
    w = int(app.layout.window.width)
    h = int(app.layout.window.height)
    pygame.draw.line(app.surface, app.theme.frame_outer, (0, 0), (0, h - 1), 2)
    pygame.draw.line(app.surface, app.theme.frame_outer, (w - 1, 0), (w - 1, h - 1), 2)
    pygame.draw.line(app.surface, app.theme.frame_outer, (0, h - 1), (w - 1, h - 1), 2)


def draw_runtime_banners(app, control_policy) -> None:
    banner_text = control_policy.status_banner_text
    if banner_text is None:
        return
    color = app.theme.banner_warn
    banner = safe_render_text(app, str(banner_text), color, small=True)
    x = int(app.settings.board_offset_x + 12)
    y = int(app.settings.board_offset_y + 96)
    app.surface.blit(banner, (x, y))


def draw_perf_overlay(app) -> None:
    if not app.app_state.debug_overlay:
        return
    if not app._frame_ms_samples:
        return
    samples = sorted(app._frame_ms_samples)
    avg = float(sum(samples)) / float(len(samples))
    p95 = float(samples[int(0.95 * (len(samples) - 1))])
    text = f"Frame ms avg={avg:.2f} p95={p95:.2f}"
    surf = safe_render_text(app, text, app.theme.perf_text, small=True)
    app.surface.blit(surf, (12, 8))


def safe_render_text(app, text: str, color: tuple[int, int, int], *, small: bool) -> pygame.Surface:
    try:
        font = app.small_font if small else app.font
        return font.render(str(text), True, color)
    except Exception:
        try:
            font = pygame.font.SysFont("Arial", 16 if small else 20, bold=True)
            return font.render(str(text), True, color)
        except Exception:
            # Last resort: create a surface with a colored rectangle
            surf = pygame.Surface((max(10, len(str(text)) * 8), 20 if small else 24))
            surf.fill(color)
            return surf


def draw_options_window(app) -> None:
    # Semi-transparent dark background overlay
    overlay = pygame.Surface((app.layout.window.width, app.layout.window.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 132))
    app.surface.blit(overlay, (0, 0))

    # Centered options window with fixed header/footer and responsive content grid.
    window_width = app.layout.window.width
    window_height = app.layout.window.height

    panel_width = max(560, min(int(window_width * 0.84), 980))
    panel_height = max(520, min(int(window_height * 0.86), 900))
    panel_x = (window_width - panel_width) // 2
    panel_y = (window_height - panel_height) // 2
    panel = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

    pygame.draw.rect(app.surface, app.theme.graph_bg, panel, border_radius=12)
    pygame.draw.rect(app.surface, app.theme.board_frame_border, panel, width=2, border_radius=12)

    pad = 16
    row_h = max(24, int(app.design_tokens.components.button_row_height * 0.78))
    gap = 6
    section_gap = 10
    section_title_gap = 6
    col_gap = 12
    close_h = row_h

    head = safe_render_text(app, "Options", app.theme.section_header, small=False)
    head_x = panel.x + (panel.width - head.get_width()) // 2
    app.surface.blit(head, (head_x, panel.y + pad))
    header_bottom = int(panel.y + pad + head.get_height() + 8)

    sections: list[tuple[str, list[list]]] = [
        (
            "Training",
            [
                [app.btn_adaptive_toggle],
                [app.btn_space_strategy_toggle],
                [app.btn_tail_trend_toggle],
                [app.btn_dropout_toggle],
            ],
        ),
        (
            "Visual",
            [
                [app.btn_theme_cycle, app.btn_board_bg_cycle],
                [app.btn_snake_style_cycle, app.btn_fog_cycle],
            ],
        ),
        (
            "Playback Speed",
            [
                [app.btn_speed_down, app.btn_speed_up],
            ],
        ),
        (
            "Model Checks",
            [
                [app.btn_eval_suite],
                [app.btn_eval_holdout],
                [app.btn_eval_mode_ppo, app.btn_eval_mode_controller],
            ],
        ),
        (
            "Debug & Tools",
            [
                [app.btn_debug_toggle, app.btn_reachable_toggle],
                [app.btn_diagnostics],
            ],
        ),
    ]

    total_rows = sum(len(rows) for _, rows in sections)
    total_titles = len(sections)
    min_content_h = (
        (total_rows * row_h)
        + (max(0, total_rows - total_titles) * gap)
        + (total_titles * (app.small_font.get_linesize() + section_title_gap))
        + ((total_titles - 1) * section_gap)
    )
    content_top = header_bottom
    footer_top = int(panel.bottom - pad - close_h)
    content_bottom = int(footer_top - 16)
    available_h = max(120, content_bottom - content_top)
    if min_content_h > available_h:
        scale = max(0.72, float(available_h) / float(max(1, min_content_h)))
        row_h = max(20, int(row_h * scale))
        gap = max(3, int(gap * scale))
        section_gap = max(5, int(section_gap * scale))

    full_w = int(panel.width - (pad * 2))
    half_w = int((full_w - col_gap) // 2)
    row_y = int(content_top)
    mouse = pygame.mouse.get_pos()

    for title, rows in sections:
        if row_y + app.small_font.get_linesize() > content_bottom:
            break
        title_surf = safe_render_text(app, title, app.theme.section_header, small=True)
        title_x = panel.x + (panel.width - title_surf.get_width()) // 2
        app.surface.blit(title_surf, (title_x, row_y))
        row_y += int(title_surf.get_height() + section_title_gap)
        for row in rows:
            if row_y + row_h > content_bottom:
                break
            if len(row) == 1:
                btn = row[0]
                btn.rect = pygame.Rect(panel.x + pad, row_y, full_w, row_h)
                btn.draw(app.surface, app.small_font, mouse)
            else:
                left_btn = row[0]
                right_btn = row[1]
                left_btn.rect = pygame.Rect(panel.x + pad, row_y, half_w, row_h)
                right_btn.rect = pygame.Rect(panel.x + pad + half_w + col_gap, row_y, half_w, row_h)
                left_btn.draw(app.surface, app.small_font, mouse)
                right_btn.draw(app.surface, app.small_font, mouse)
            row_y += int(row_h + gap)
        row_y += int(section_gap)

    app.btn_options_close.rect = pygame.Rect(panel.x + pad, footer_top, full_w, close_h)
    app.btn_options_close.draw(app.surface, app.small_font, mouse)
