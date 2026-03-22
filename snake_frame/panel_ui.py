"""
UI rendering components for the side panels in Snake Frame application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict

import pygame

from .graph_renderer import ScoreGraphRenderer
from .settings import Settings
from .theme import DesignTokens, ThemePalette, get_design_tokens, get_theme
from .ui import Button, NumericInput


@dataclass(frozen=True)
class PanelRenderData:
    """
    Data required for rendering the side panels.
    
    Attributes:
        training_episode_scores: List of scores from training episodes
        run_episode_scores: List of scores from run episodes
        training_graph_rect: Rectangle for training graph rendering
        run_graph_rect: Rectangle for run graph rendering
        training_graph_badges: List of badge strings for training section
        run_graph_badges: List of badge strings for run section
        run_status_lines: List of status lines to display
        settings_lines: List of settings lines to display
        training_header_y: Y position for training header
        training_badges_y: Y position for training badges
        run_header_y: Y position for run header
        run_badges_y: Y position for run badges
    """
    training_episode_scores: list[int]
    run_episode_scores: list[int]
    training_graph_rect: pygame.Rect
    run_graph_rect: pygame.Rect
    training_graph_badges: list[str]
    run_graph_badges: list[str]
    # Explicit right panel layout positions
    training_header_y: int
    training_badges_y: int
    run_header_y: int
    run_badges_y: int
    left_status_lines: list[str] = field(default_factory=list)
    training_status_lines: list[str] = field(default_factory=list)
    run_status_lines: list[str] = field(default_factory=list)
    debug_status_lines: list[str] = field(default_factory=list)
    settings_lines: list[str] = field(default_factory=list)
    selected_tab: str = "train"


@dataclass(frozen=True)
class PanelControls:
    """
    Collection of UI controls for the side panels.
    
    Attributes:
        generations_input: Input for number of training generations
        btn_train_start: Button to start training
        btn_train_stop: Button to stop training
        btn_save: Button to save model/state
        btn_load: Button to load model/state
        btn_delete: Button to delete model/state
        btn_game_start: Button to start a game/run
        btn_game_stop: Button to stop a game/run
        btn_restart: Button to restart current game
        btn_options: Button to open options panel
        btn_options_close: Button to close options panel
        btn_adaptive_toggle: Toggle for adaptive reward
        btn_space_strategy_toggle: Toggle for space strategy
        btn_tail_trend_toggle: Toggle for tail trend features
        btn_dropout_toggle: Toggle for dropout training
        btn_theme_cycle: Button to cycle through themes
        btn_board_bg_cycle: Button to cycle board background modes
        btn_snake_style_cycle: Button to cycle snake styles
        btn_fog_cycle: Button to cycle fog density
        btn_speed_down: Button to decrease live speed
        btn_speed_up: Button to increase live speed
        btn_eval_suite: Button to run evaluation suite
        btn_eval_mode_ppo: Button to set evaluation to PPO only
        btn_eval_mode_controller: Button to set evaluation to controller on
        btn_eval_holdout: Button to run holdout evaluation
        btn_debug_toggle: Toggle for debug overlay
        btn_reachable_toggle: Toggle for reachable overlay
        btn_diagnostics: Button to generate diagnostics bundle
    """
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


class SidePanelsRenderer:
    """
    Renders the side panels (left and right) for the Snake Frame application.
    
    Handles layout and rendering of UI components including graphs, controls,
    and status information in the side panels of the main application window.
    """
    def __init__(
        self,
        settings: Settings,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        """
        Initialize the side panels renderer.
        
        Args:
            settings: Application settings object
            font: Font for primary text rendering
            small_font: Font for secondary/small text rendering
        """
        self.settings = settings
        self.theme: ThemePalette = get_theme(getattr(settings, "theme_name", ""))
        compact = int(settings.window_height_px or settings.window_px) < int(get_design_tokens(settings.theme_name).spacing.graph_margin_compact_threshold)
        self.tokens: DesignTokens = get_design_tokens(getattr(settings, "theme_name", ""), compact=compact)
        self.font = font
        self.small_font = small_font
        self.graph = ScoreGraphRenderer(small_font, theme=self.theme)
        self._static_bg_cache: tuple[tuple[int, int, int, int], pygame.Surface] | None = None
        self._text_cache: OrderedDict[tuple[str, tuple[int, int, int], bool], pygame.Surface] = OrderedDict()

    def clear_caches(self) -> None:
        """
        Clear all cached surfaces to free memory.
        
        This should be called when settings change or when the renderer
        needs to refresh all cached text and background surfaces.
        """
        self._static_bg_cache = None
        self._text_cache.clear()
        self.graph.clear_cache()

    def draw(
        self,
        surface: pygame.Surface,
        data: PanelRenderData,
        controls: PanelControls,
    ) -> None:
        """
        Render the side panels onto the given surface.
        
        Args:
            surface: pygame surface to draw on
            data: PanelRenderData containing information to display
            controls: PanelControls containing interactive UI elements
        """
        left_w = int(self.settings.left_panel_px)
        board_w = int(self.settings.window_px)
        right_w = int(self.settings.right_panel_px)
        panel_h = int(self.settings.window_height_px or self.settings.window_px)
        surface.blit(self._static_background(left_w, board_w, right_w, panel_h), (0, 0))

        right_inner_width = int(max(120, data.training_graph_rect.width))
        right_x = int(data.training_graph_rect.x)
        mouse_pos = pygame.mouse.get_pos()
        controls.btn_tab_train.draw(surface, self.small_font, mouse_pos)
        controls.btn_tab_run.draw(surface, self.small_font, mouse_pos)
        controls.btn_tab_debug.draw(surface, self.small_font, mouse_pos)
        selected_tab = str(getattr(data, "selected_tab", "train")).strip().lower()
        if selected_tab not in ("train", "run", "debug"):
            selected_tab = "train"

        if selected_tab == "train":
            self._draw_section_header(surface, "Training KPIs", right_x, data.training_header_y)
            badges_end_y = self._draw_graph_badges(
                surface,
                start_x=right_x,
                start_y=data.training_badges_y,
                max_width=right_inner_width,
                badges=data.training_graph_badges,
            )
            line_h = self._line_height()
            min_text_rows = max(4, min(8, len(data.training_status_lines) + 1))
            min_text_area = int((min_text_rows * line_h) + 22)
            train_graph_rect = pygame.Rect(data.training_graph_rect)
            train_graph_rect.y = max(int(train_graph_rect.y), int(badges_end_y + 8))
            max_graph_bottom = int((self.settings.window_height_px or self.settings.window_px) - 16 - min_text_area)
            if train_graph_rect.bottom > max_graph_bottom:
                train_graph_rect.height = max(48, int(max_graph_bottom - train_graph_rect.y))
            pygame.draw.rect(surface, self.theme.graph_bg, train_graph_rect)
            self.graph.draw(
                surface,
                train_graph_rect,
                data.training_episode_scores,
                empty_message="Train PPO to build graph.",
            )
            self._draw_right_text_block(
                surface,
                title="Training Health",
                x=right_x,
                y=int(train_graph_rect.bottom + 16),
                width=right_inner_width,
                max_bottom=int(self.settings.window_height_px or self.settings.window_px) - 16,
                lines=list(data.training_status_lines),
            )
        elif selected_tab == "run":
            self._draw_section_header(surface, "Run KPIs", right_x, data.run_header_y)
            badges_end_y = self._draw_graph_badges(
                surface,
                start_x=right_x,
                start_y=data.run_badges_y,
                max_width=right_inner_width,
                badges=data.run_graph_badges,
            )
            run_graph_rect = pygame.Rect(data.run_graph_rect)
            line_h = self._line_height()
            min_text_rows = max(4, min(8, len(data.run_status_lines) + 1))
            min_text_area = int((min_text_rows * line_h) + 22)
            run_graph_rect.y = max(int(run_graph_rect.y), int(badges_end_y + 8))
            max_graph_bottom = int((self.settings.window_height_px or self.settings.window_px) - 16 - min_text_area)
            if run_graph_rect.bottom > max_graph_bottom:
                run_graph_rect.height = max(48, int(max_graph_bottom - run_graph_rect.y))
            pygame.draw.rect(surface, self.theme.graph_bg, run_graph_rect)
            self.graph.draw(
                surface,
                run_graph_rect,
                data.run_episode_scores,
                empty_message="Play/Watch runs to build graph.",
            )
            self._draw_right_text_block(
                surface,
                title="Run Summary",
                x=right_x,
                y=int(run_graph_rect.bottom + 16),
                width=right_inner_width,
                max_bottom=int(self.settings.window_height_px or self.settings.window_px) - 16,
                lines=list(data.run_status_lines),
            )
        else:
            self._draw_right_text_block(
                surface,
                title="Debug / Advanced",
                x=right_x,
                y=int(data.training_header_y),
                width=right_inner_width,
                max_bottom=int(self.settings.window_height_px or self.settings.window_px) - 16,
                lines=list(data.debug_status_lines or data.settings_lines),
            )
        
        # Left panel header should be anchored to left controls, not right-panel content.
        left_header_y = int(
            max(
                int(self.tokens.spacing.left_controls_top_padding),
                int(controls.generations_input.rect.top - self._line_height() - 8),
            )
        )
        self._draw_left_panel_sections(
            surface,
            left_header_y,
            controls,
            data.left_status_lines,
        )

    def _draw_left_panel_sections(
        self,
        surface: pygame.Surface,
        top: int,
        controls: PanelControls,
        left_status_lines: list[str],
    ) -> None:
        """
        Draw the left panel sections including controls, status, and settings.
        
        Args:
            surface: pygame surface to draw on
            top: Y coordinate to start drawing from
            controls: PanelControls containing interactive UI elements
            run_status_lines: List of status lines to display
            settings_lines: List of settings lines to display
        """
        mouse_pos = pygame.mouse.get_pos()
        panel_bottom = int(self.settings.window_height_px or self.settings.window_px) - int(self.tokens.spacing.section_gap)
        self._draw_section_header(surface, "Train Controls", 18, top)
        controls.generations_input.draw(surface, self.small_font)

        controls.btn_train_start.draw(surface, self.small_font, mouse_pos)
        controls.btn_train_stop.draw(surface, self.small_font, mouse_pos)
        controls.btn_save.draw(surface, self.small_font, mouse_pos)
        controls.btn_load.draw(surface, self.small_font, mouse_pos)
        controls.btn_delete.draw(surface, self.small_font, mouse_pos)
        controls.btn_game_start.draw(surface, self.small_font, mouse_pos)
        controls.btn_game_stop.draw(surface, self.small_font, mouse_pos)
        controls.btn_restart.draw(surface, self.small_font, mouse_pos)
        controls.btn_options.draw(surface, self.small_font, mouse_pos)

        y = controls.btn_options.rect.bottom + int(self.tokens.spacing.section_gap * 1.5)
        line_h = self._line_height()
        if y + line_h >= panel_bottom:
            return
        self._draw_divider(surface, y - 5)
        self._draw_section_header(surface, "Session", 18, y)
        y += line_h
        max_text_w = max(80, int(self.settings.left_panel_px - 36))
        for line in left_status_lines:
            if y + line_h > panel_bottom:
                return
            self._draw_key_value_line(
                surface,
                x=18,
                y=y,
                line=str(line),
                max_width=max_text_w,
                key_color=self.theme.section_header,
                value_color=self.theme.status_color,
            )
            y += line_h

    def _draw_right_text_block(
        self,
        surface: pygame.Surface,
        *,
        title: str,
        x: int,
        y: int,
        width: int,
        max_bottom: int,
        lines: list[str],
    ) -> None:
        line_h = self._line_height()
        if y + line_h >= max_bottom:
            return
        self._draw_divider_right(surface, y - 6, x, width)
        self._draw_section_header(surface, str(title), int(x), int(y))
        y += line_h
        for line in lines:
            if y + line_h > max_bottom:
                return
            self._draw_key_value_line(
                surface,
                x=int(x),
                y=int(y),
                line=str(line),
                max_width=max(80, int(width)),
                key_color=self.theme.section_header,
                value_color=self.theme.status_color,
            )
            y += line_h

    def _draw_graph_badges(
        self,
        surface: pygame.Surface,
        *,
        start_x: int,
        start_y: int,
        max_width: int,
        badges: list[str],
    ) -> int:
        """
        Draw a series of badges (small labeled rectangles) in rows.
        
        Badges are wrapped to fit within max_width and arranged in rows
        up to the maximum allowed rows.
        
        Args:
            surface: pygame surface to draw on
            start_x: X position to start drawing badges
            start_y: Y position to start drawing badges
            max_width: Maximum width available for badges
            badges: List of strings to display as badges
            
        Returns:
            Y position after the last drawn badge (for positioning subsequent elements)
        """
        if not badges:
            return int(start_y)
        rendered = [self._text(str(text), self.theme.badge_text, small=True) for text in badges]
        badge_h = max(int(self.tokens.components.badge_min_height), int(self.small_font.get_linesize() + int(self.tokens.components.badge_padding_y * 2)))
        widths = [int(label.get_width() + int(self.tokens.components.badge_padding_x * 2)) for label in rendered]

        rows: list[list[int]] = [[]]
        row_width = 0
        max_w = int(max_width)
        for idx, badge_w in enumerate(widths):
            next_w = badge_w if row_width == 0 else row_width + int(self.tokens.spacing.badge_gap_x) + badge_w
            if row_width > 0 and next_w > max_w:
                rows.append([idx])
                row_width = badge_w
                continue
            rows[-1].append(idx)
            row_width = next_w
            if len(rows) >= int(self.tokens.components.max_badge_rows):
                break

        y = int(start_y)
        for row in rows:
            x = int(start_x)
            for idx in row:
                label = rendered[idx]
                rect = pygame.Rect(x, y, widths[idx], badge_h)
                pygame.draw.rect(surface, self.theme.badge_bg, rect, border_radius=7)
                pygame.draw.rect(surface, self.theme.badge_border, rect, width=1, border_radius=7)
                label_rect = label.get_rect(center=rect.center)
                surface.blit(label, label_rect)
                x = int(rect.right + int(self.tokens.spacing.badge_gap_x))
            y += int(badge_h + int(self.tokens.spacing.badge_gap_y))
        return int(y)



    def _draw_section_header(self, surface: pygame.Surface, text: str, x: int, y: int) -> None:
        surface.blit(self._text(text, self.theme.section_header, small=False), (int(x), int(y - 2)))

    def _draw_divider(self, surface: pygame.Surface, y: int) -> None:
        x0 = int(self.tokens.spacing.panel_inner_pad_x)
        panel_right = int(self.settings.left_panel_px - self.tokens.spacing.panel_inner_pad_x)
        pygame.draw.line(surface, self.theme.divider, (x0, int(y)), (panel_right, int(y)), width=1)

    def _draw_divider_right(self, surface: pygame.Surface, y: int, x: int, width: int) -> None:
        x0 = int(max(0, x))
        x1 = int(max(x0 + 1, x + max(1, width)))
        pygame.draw.line(surface, self.theme.divider, (x0, int(y)), (x1, int(y)), width=1)

    def _line_height(self) -> int:
        return max(int(self.tokens.typography.status_line_min_height), int(self.small_font.get_linesize() + 2))

    def _static_background(self, left_w: int, board_w: int, right_w: int, panel_h: int) -> pygame.Surface:
        key = (left_w, board_w, right_w, panel_h)
        if self._static_bg_cache is not None and self._static_bg_cache[0] == key:
            return self._static_bg_cache[1]
        static = pygame.Surface((self.settings.window_width_px, panel_h), pygame.SRCALPHA)
        if pygame.display.get_surface() is not None:
            static = static.convert_alpha()
        static.fill((0, 0, 0, 0))
        pygame.draw.rect(static, self.theme.panel_bg, pygame.Rect(0, 0, left_w, panel_h))
        pygame.draw.rect(static, self.theme.panel_bg, pygame.Rect(left_w + board_w, 0, right_w, panel_h))
        for y in range(0, panel_h, 4):
            t = float(y) / float(max(1, panel_h - 1))
            shade = (
                int(self.theme.panel_bg_accent[0] * (1.0 - t) + self.theme.panel_bg[0] * t),
                int(self.theme.panel_bg_accent[1] * (1.0 - t) + self.theme.panel_bg[1] * t),
                int(self.theme.panel_bg_accent[2] * (1.0 - t) + self.theme.panel_bg[2] * t),
            )
            pygame.draw.line(static, shade, (0, y), (left_w, y), 1)
            pygame.draw.line(static, shade, (left_w + board_w, y), (left_w + board_w + right_w, y), 1)
        self._static_bg_cache = (key, static)
        return static

    def _text(self, text: str, color: tuple[int, int, int], *, small: bool) -> pygame.Surface:
        # Explicitly type the key components to satisfy type checker
        text_str: str = str(text)
        color_tuple: tuple[int, int, int] = (color[0], color[1], color[2])
        small_bool: bool = bool(small)
        key: tuple[str, tuple[int, int, int], bool] = (text_str, color_tuple, small_bool)
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached
        font = self.small_font if small else self.font
        try:
            rendered = font.render(str(text), True, color)
        except Exception:
            fallback = pygame.font.SysFont("Arial", 14 if small else 18, bold=True)
            rendered = fallback.render(str(text), True, color)
        self._text_cache[key] = rendered
        if len(self._text_cache) > 512:
            # Bounded LRU-style eviction to avoid full-cache clear frame spikes.
            while len(self._text_cache) > 448:
                self._text_cache.popitem(last=False)
        return rendered

    def _fit_text(self, text: str, color: tuple[int, int, int], max_width: int) -> pygame.Surface:
        candidate = str(text)
        rendered = self._text(candidate, color, small=True)
        if rendered.get_width() <= int(max_width):
            return rendered
        ellipsis = "..."
        while candidate:
            candidate = candidate[:-1]
            clipped = candidate.rstrip() + ellipsis
            rendered = self._text(clipped, color, small=True)
            if rendered.get_width() <= int(max_width):
                return rendered
        return self._text(ellipsis, color, small=True)

    def _draw_key_value_line(
        self,
        surface: pygame.Surface,
        *,
        x: int,
        y: int,
        line: str,
        max_width: int,
        key_color: tuple[int, int, int],
        value_color: tuple[int, int, int],
    ) -> None:
        raw = str(line)
        if ":" not in raw:
            surf = self._fit_text(raw, value_color, max_width)
            surface.blit(surf, (int(x), int(y)))
            return
        key, value = raw.split(":", 1)
        key_text = f"{key.strip()}:"
        key_surf = self._text(key_text, key_color, small=True)
        surface.blit(key_surf, (int(x), int(y)))
        value_x = int(x + key_surf.get_width() + 6)
        value_w = max(20, int(max_width - key_surf.get_width() - 6))
        value_surf = self._fit_text(value.strip(), value_color, value_w)
        surface.blit(value_surf, (value_x, int(y)))


# Backward-compatible alias for existing imports/tests.
LeftPanelRenderer = SidePanelsRenderer
