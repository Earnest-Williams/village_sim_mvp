"""wxPython GUI for running and viewing simulation output."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wx  # type: ignore[import-not-found]
import wx.stc as wxstc  # type: ignore[import-not-found]

from village_sim.core.config import SimConfig
from village_sim.orchestrator.action_model import ActionLibrary
from village_sim.sim.engine import Simulation
from village_sim.sim.metrics import SimResult
from village_sim.sim.replay import write_run_report
from village_sim.view.ascii_view import (
    ROLE_COLORS,
    RenderedMap,
    render_ascii_map,
    render_map_model,
)
from village_sim.view.stc_map import (
    GUI_DEFAULT_WORLD_SIZE,
    MAP_BACKGROUND,
    MAP_DEFAULT_FONT_POINT_SIZE,
    MAP_DEFAULT_FOREGROUND,
    MAP_STATUS_FOREGROUND,
    STC_ROLE_STYLE,
    build_stc_content,
)

MAP_UPDATE_INTERVAL_SECONDS: float = 0.05


def _build_stc_content(
    rendered_map: RenderedMap, *, left_padding_columns: int = 0
) -> tuple[str, list[tuple[int, int]]]:
    """Backward-compatible wrapper for STC content tests and callers."""
    return build_stc_content(rendered_map, left_padding_columns=left_padding_columns)


def _make_map_font(point_size: int = MAP_DEFAULT_FONT_POINT_SIZE) -> wx.Font:
    """Create a compact monospace map font with safe platform fallbacks."""
    preferred_faces: list[str] = [
        "Cascadia Mono",
        "Cascadia Code",
        "JetBrains Mono",
        "DejaVu Sans Mono",
        "Noto Sans Mono",
        "Liberation Mono",
        "Monospace",
    ]
    for face_name in preferred_faces:
        if not wx.FontEnumerator.IsValidFacename(face_name):
            continue
        font = wx.Font(
            wx.FontInfo(point_size).Family(wx.FONTFAMILY_TELETYPE).FaceName(face_name)
        )
        if font.IsOk():
            return font
    return wx.Font(wx.FontInfo(point_size).Family(wx.FONTFAMILY_TELETYPE))


@dataclass(frozen=True, slots=True)
class GuiRunOptions:
    """Validated run options collected from the wx controls."""

    config: SimConfig
    print_map: bool
    batch: int
    action_library_in: Path | None
    action_library_out: Path | None
    local_map_radius: int
    snapshot_every: int
    replay: Path | None
    update_every_tick: bool
    tick_delay_seconds: float


class VillageSimFrame(wx.Frame):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__(parent=None, title="Village Sim MVP", size=(980, 840))
        panel = wx.Panel(self)
        root_sizer = wx.BoxSizer(wx.VERTICAL)

        controls_sizer = wx.FlexGridSizer(rows=10, cols=4, vgap=8, hgap=8)
        controls_sizer.AddGrowableCol(1, 1)
        controls_sizer.AddGrowableCol(3, 1)

        self.seed_ctrl = wx.SpinCtrl(panel, min=1, max=1_000_000, initial=1)
        self.days_ctrl = wx.SpinCtrl(panel, min=1, max=365, initial=10)
        self.width_ctrl = wx.SpinCtrl(
            panel, min=8, max=256, initial=GUI_DEFAULT_WORLD_SIZE
        )
        self.height_ctrl = wx.SpinCtrl(
            panel, min=8, max=256, initial=GUI_DEFAULT_WORLD_SIZE
        )
        self.batch_ctrl = wx.SpinCtrl(panel, min=1, max=10_000, initial=1)
        self.local_map_radius_ctrl = wx.SpinCtrl(panel, min=0, max=512, initial=0)
        self.snapshot_every_ctrl = wx.SpinCtrl(panel, min=0, max=1_000_000, initial=0)
        self.speed_ctrl = wx.SpinCtrl(panel, min=0, max=1000, initial=25)
        self.map_font_size_ctrl = wx.SpinCtrl(
            panel, min=6, max=24, initial=MAP_DEFAULT_FONT_POINT_SIZE
        )
        self.print_map_ctrl = wx.CheckBox(panel, label="Render final ASCII map")
        self.print_map_ctrl.SetValue(True)
        self.discoverables_ctrl = wx.CheckBox(
            panel,
            label="Seed canonical discoverables",
        )
        self.goap_ctrl = wx.CheckBox(panel, label="Enable GOAP control")
        self.tick_update_ctrl = wx.CheckBox(panel, label="Update map every tick")
        self.action_library_in_ctrl = wx.TextCtrl(panel)
        self.action_library_out_ctrl = wx.TextCtrl(panel)
        self.replay_ctrl = wx.TextCtrl(panel)

        self._add_labeled_control(controls_sizer, panel, "Seed", self.seed_ctrl)
        self._add_labeled_control(controls_sizer, panel, "Days", self.days_ctrl)
        self._add_labeled_control(controls_sizer, panel, "Width", self.width_ctrl)
        self._add_labeled_control(controls_sizer, panel, "Height", self.height_ctrl)
        self._add_labeled_control(controls_sizer, panel, "Batch runs", self.batch_ctrl)
        self._add_labeled_control(
            controls_sizer,
            panel,
            "Local map radius",
            self.local_map_radius_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            panel,
            "Snapshot every N ticks",
            self.snapshot_every_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            panel,
            "Tick delay (ms)",
            self.speed_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            panel,
            "Map font size",
            self.map_font_size_ctrl,
        )
        controls_sizer.Add(self.print_map_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add(self.discoverables_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add(self.goap_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add(self.tick_update_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        self._add_labeled_control(
            controls_sizer,
            panel,
            "Action library in",
            self.action_library_in_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            panel,
            "Action library out",
            self.action_library_out_ctrl,
        )
        self._add_labeled_control(
            controls_sizer, panel, "Replay JSON", self.replay_ctrl
        )
        controls_sizer.Add((0, 0), 1, wx.EXPAND)

        self.run_button = wx.Button(panel, label="Run Simulation")
        self.run_button.Bind(wx.EVT_BUTTON, self.on_run)

        self.summary_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_NONE,
            size=(-1, 150),
        )
        self.summary_ctrl.SetBackgroundColour(panel.GetBackgroundColour())
        self.map_ctrl = wxstc.StyledTextCtrl(panel, style=wx.BORDER_NONE)
        self._last_rendered_map: RenderedMap | None = None
        self._configure_map_ctrl()
        self.map_font_size_ctrl.Bind(wx.EVT_SPINCTRL, self.on_map_font_size_changed)
        self.map_font_size_ctrl.Bind(wx.EVT_TEXT, self.on_map_font_size_changed)

        root_sizer.Add(controls_sizer, 0, wx.EXPAND | wx.ALL, 12)
        root_sizer.Add(self.run_button, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        root_sizer.Add(
            wx.StaticText(panel, label="Run Summary"), 0, wx.LEFT | wx.RIGHT, 12
        )
        root_sizer.Add(
            self.summary_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12
        )
        root_sizer.Add(wx.StaticText(panel, label="Map"), 0, wx.LEFT | wx.RIGHT, 12)
        root_sizer.Add(self.map_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        panel.SetSizer(root_sizer)
        self.Centre()

    @staticmethod
    def _add_labeled_control(
        controls_sizer: wx.FlexGridSizer,
        panel: wx.Panel,
        label: str,
        control: wx.Window,
    ) -> None:
        controls_sizer.Add(
            wx.StaticText(panel, label=label),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        controls_sizer.Add(control, 1, wx.EXPAND)

    def _configure_map_ctrl(self) -> None:
        map_bg = wx.Colour(MAP_BACKGROUND)
        default_fg = wx.Colour(MAP_DEFAULT_FOREGROUND)
        summary_fg = wx.Colour(MAP_STATUS_FOREGROUND)
        map_font = _make_map_font(self.map_font_size_ctrl.GetValue())

        self.map_ctrl.SetReadOnly(True)
        self.map_ctrl.SetWrapMode(wxstc.STC_WRAP_NONE)
        self.map_ctrl.SetEdgeMode(wxstc.STC_EDGE_NONE)
        self.map_ctrl.SetViewEOL(False)
        self.map_ctrl.SetViewWhiteSpace(False)
        self.map_ctrl.SetIndentationGuides(False)
        self.map_ctrl.SetUseHorizontalScrollBar(True)
        self.map_ctrl.SetUseVerticalScrollBar(True)
        self.map_ctrl.SetScrollWidthTracking(False)
        self.map_ctrl.SetUndoCollection(False)
        self.map_ctrl.SetCaretWidth(0)
        self.map_ctrl.SetCaretLineVisible(False)
        self.map_ctrl.SetZoom(0)
        self.map_ctrl.SetBackgroundColour(map_bg)
        margins_background_method: Any = getattr(
            self.map_ctrl, "SetMarginsBackground", None
        )
        if callable(margins_background_method):
            margins_background_method(map_bg)
        fold_margin_colour_method: Any = getattr(
            self.map_ctrl, "SetFoldMarginColour", None
        )
        if callable(fold_margin_colour_method):
            fold_margin_colour_method(True, map_bg)
        fold_margin_highlight_method: Any = getattr(
            self.map_ctrl, "SetFoldMarginHiColour", None
        )
        if callable(fold_margin_highlight_method):
            fold_margin_highlight_method(True, map_bg)
        self._hide_map_margins()

        self.map_ctrl.SetFont(map_font)
        self.map_ctrl.StyleSetFont(wxstc.STC_STYLE_DEFAULT, map_font)
        self.map_ctrl.StyleSetSize(
            wxstc.STC_STYLE_DEFAULT, self.map_font_size_ctrl.GetValue()
        )
        self.map_ctrl.StyleSetBackground(wxstc.STC_STYLE_DEFAULT, map_bg)
        self.map_ctrl.StyleSetForeground(wxstc.STC_STYLE_DEFAULT, default_fg)
        self.map_ctrl.StyleClearAll()
        self.map_ctrl.StyleSetFont(0, map_font)
        self.map_ctrl.StyleSetSize(0, self.map_font_size_ctrl.GetValue())
        self.map_ctrl.StyleSetBackground(0, map_bg)
        self.map_ctrl.StyleSetForeground(0, default_fg)
        self.map_ctrl.StyleSetBackground(wxstc.STC_STYLE_LINENUMBER, map_bg)
        self.map_ctrl.StyleSetForeground(wxstc.STC_STYLE_LINENUMBER, default_fg)
        for role, style_num in STC_ROLE_STYLE.items():
            role_fg = summary_fg
            if role != "summary":
                role_fg = wx.Colour(ROLE_COLORS.get(role, MAP_DEFAULT_FOREGROUND))
            self.map_ctrl.StyleSetFont(style_num, map_font)
            self.map_ctrl.StyleSetSize(style_num, self.map_font_size_ctrl.GetValue())
            self.map_ctrl.StyleSetForeground(style_num, role_fg)
            self.map_ctrl.StyleSetBackground(style_num, map_bg)

    def _hide_map_margins(self) -> None:
        margin_count_method: Any = getattr(self.map_ctrl, "GetMargins", None)
        raw_margin_count: Any = None
        if callable(margin_count_method):
            raw_margin_count = margin_count_method()
        margin_count = 5
        if isinstance(raw_margin_count, int) and raw_margin_count > 0:
            margin_count = raw_margin_count
        for margin in range(margin_count):
            self.map_ctrl.SetMarginWidth(margin, 0)
            self.map_ctrl.SetMarginSensitive(margin, False)

    def on_map_font_size_changed(self, _: wx.CommandEvent) -> None:
        self._configure_map_ctrl()
        if self._last_rendered_map is not None:
            self._set_colored_map_value(self._last_rendered_map)

    def _visible_map_columns(self) -> int:
        raw_char_width, _raw_char_height = self.map_ctrl.GetTextExtent("M")
        if not isinstance(raw_char_width, int) or raw_char_width <= 0:
            return 0
        raw_client_width: Any = self.map_ctrl.GetClientSize().width
        if not isinstance(raw_client_width, int) or raw_client_width <= 0:
            return 0
        return max(0, raw_client_width // raw_char_width)

    def _center_padding_columns(self, rendered_map: RenderedMap) -> int:
        visible_cols = self._visible_map_columns()
        map_cols = max((len(row) for row in rendered_map.rows), default=0)
        if visible_cols <= map_cols:
            return 0
        return (visible_cols - map_cols) // 2

    def _update_map_scroll_width(self, full_text: str) -> None:
        raw_char_width, _raw_char_height = self.map_ctrl.GetTextExtent("M")
        if not isinstance(raw_char_width, int) or raw_char_width <= 0:
            return
        raw_client_width: Any = self.map_ctrl.GetClientSize().width
        client_width = 0
        if isinstance(raw_client_width, int) and raw_client_width > 0:
            client_width = raw_client_width
        max_columns = max((len(line) for line in full_text.splitlines()), default=0)
        scroll_width = max(client_width, (max_columns + 1) * raw_char_width)
        self.map_ctrl.SetScrollWidth(scroll_width)

    def on_run(self, _: wx.CommandEvent) -> None:
        self.run_button.Disable()
        self.summary_ctrl.Clear()
        self._clear_map_ctrl()
        options: GuiRunOptions = self._collect_options()

        def thread_target() -> None:
            try:
                if options.batch > 1:
                    summary = self._run_batch(options)
                    wx.CallAfter(self._update_ui, summary, None)
                    return
                summary, rendered_map = self._run_single(options)
                wx.CallAfter(self._update_ui, summary, rendered_map)
            except Exception as error:
                message: str = f"Simulation failed: {error}"
                wx.CallAfter(
                    wx.MessageBox,
                    message,
                    "Error",
                    wx.OK | wx.ICON_ERROR,
                )
            finally:
                wx.CallAfter(self._enable_run_button)

        threading.Thread(target=thread_target, daemon=True).start()

    def _collect_options(self) -> GuiRunOptions:
        config = SimConfig(
            width=self.width_ctrl.GetValue(),
            height=self.height_ctrl.GetValue(),
            max_days=self.days_ctrl.GetValue(),
            seed=self.seed_ctrl.GetValue(),
            enable_initial_discoverables=self.discoverables_ctrl.GetValue(),
            enable_goap_control=self.goap_ctrl.GetValue(),
        )
        return GuiRunOptions(
            config=config,
            print_map=self.print_map_ctrl.GetValue(),
            batch=self.batch_ctrl.GetValue(),
            action_library_in=self._path_from_text_ctrl(self.action_library_in_ctrl),
            action_library_out=self._path_from_text_ctrl(self.action_library_out_ctrl),
            local_map_radius=self.local_map_radius_ctrl.GetValue(),
            snapshot_every=self.snapshot_every_ctrl.GetValue(),
            replay=self._path_from_text_ctrl(self.replay_ctrl),
            update_every_tick=self.tick_update_ctrl.GetValue(),
            tick_delay_seconds=self.speed_ctrl.GetValue() / 1000.0,
        )

    @staticmethod
    def _path_from_text_ctrl(control: wx.TextCtrl) -> Path | None:
        raw_path: str = control.GetValue().strip()
        if raw_path == "":
            return None
        return Path(raw_path)

    def _run_single(self, options: GuiRunOptions) -> tuple[str, RenderedMap | None]:
        sim = Simulation(options.config)
        if options.action_library_in is not None:
            sim.action_library = ActionLibrary.load(options.action_library_in)

        max_ticks: int = options.config.max_ticks()
        last_map_update_time: float = time.monotonic() - MAP_UPDATE_INTERVAL_SECONDS
        while sim.tick < max_ticks and sim.agent.alive:
            sim.step()
            if options.snapshot_every > 0 and sim.tick % options.snapshot_every == 0:
                sim.snapshots.append(sim.snapshot(include_ascii=True))
            if options.update_every_tick:
                now: float = time.monotonic()
                if now - last_map_update_time >= MAP_UPDATE_INTERVAL_SECONDS:
                    current_map: RenderedMap = self._render_map_model(
                        sim, options.local_map_radius
                    )
                    wx.CallAfter(self._set_map_value, current_map)
                    last_map_update_time = now
            if options.tick_delay_seconds > 0.0:
                time.sleep(options.tick_delay_seconds)

        result: SimResult = sim.result()
        if options.action_library_out is not None:
            sim.action_library.save(options.action_library_out)
        if options.replay is not None:
            write_run_report(
                options.replay,
                options.config,
                result,
                sim.events,
                sim.snapshots,
            )
        summary: str = self._format_result(
            result,
            len(sim.events),
            len(sim.snapshots),
            options.action_library_out,
            options.replay,
        )
        rendered_map: RenderedMap | None = None
        if options.print_map:
            rendered_map = self._render_map_model(sim, options.local_map_radius)
        return summary, rendered_map

    @staticmethod
    def _run_batch(options: GuiRunOptions) -> str:
        results: list[SimResult] = []
        for offset in range(options.batch):
            config = SimConfig(
                width=options.config.width,
                height=options.config.height,
                max_days=options.config.max_days,
                ticks_per_day=options.config.ticks_per_day,
                seed=options.config.seed + offset,
                enable_initial_discoverables=options.config.enable_initial_discoverables,
                enable_goap_control=options.config.enable_goap_control,
                tile_size_meters=options.config.tile_size_meters,
            )
            sim = Simulation(config)
            results.append(sim.run())

        survived_count: int = sum(1 for result in results if result.survived)
        average_days: float = sum(result.days_elapsed for result in results) / float(
            len(results)
        )
        average_distance: float = sum(
            result.distance_walked for result in results
        ) / float(len(results))
        lines: list[str] = [
            f"Batch runs: {len(results)}",
            f"Survived full duration: {survived_count}/{len(results)}",
            f"Average days elapsed: {average_days:.2f}",
            f"Average distance walked: {average_distance:.1f}",
            (
                "seed,days,survived,death,water_sites,food_sites,distance,"
                "final_cold_stress,final_temperature_c,final_feels_cold,"
                "final_is_sheltered,cold_weather_events,cold_status_events,"
                "shelter_events"
            ),
        ]
        for result in results:
            lines.append(
                f"{result.seed},{result.days_elapsed:.2f},{result.survived},"
                f"{result.death_reason},{result.remembered_water_sites},"
                f"{result.remembered_food_sites},{result.distance_walked},"
                f"{result.final_cold_stress:.2f},{result.final_temperature_c:.1f},"
                f"{result.final_feels_cold},{result.final_is_sheltered},"
                f"{result.cold_weather_events},{result.cold_status_events},"
                f"{result.shelter_events}"
            )
        if options.action_library_in is not None:
            lines.append("Action library input is ignored for batch runs.")
        if options.action_library_out is not None:
            lines.append("Action library output is ignored for batch runs.")
        if options.replay is not None:
            lines.append("Replay output is ignored for batch runs.")
        if options.snapshot_every > 0:
            lines.append("Snapshot capture is ignored for batch runs.")
        return "\n".join(lines)

    @staticmethod
    def _render_map(sim: Simulation, local_map_radius: int) -> str:
        radius: int | None = None
        if local_map_radius > 0:
            radius = local_map_radius
        return render_ascii_map(sim.world, sim.agent, radius=radius)

    @staticmethod
    def _render_map_model(sim: Simulation, local_map_radius: int) -> RenderedMap:
        radius: int | None = None
        if local_map_radius > 0:
            radius = local_map_radius
        return render_map_model(sim.world, sim.agent, radius=radius)

    def _update_ui(self, summary: str, rendered_map: RenderedMap | None) -> None:
        if not self._frame_is_alive():
            return
        try:
            self.summary_ctrl.SetValue(summary)
            if rendered_map is None:
                self._set_map_text_preserving_view("")
            else:
                self._set_colored_map_value(rendered_map)
        except wx.PyDeadObjectError:
            return

    def _set_map_value(self, rendered_map: RenderedMap) -> None:
        if not self._frame_is_alive():
            return
        try:
            self._set_colored_map_value(rendered_map)
        except wx.PyDeadObjectError:
            return

    def _set_colored_map_value(self, rendered_map: RenderedMap) -> None:
        self._last_rendered_map = rendered_map
        left_padding_columns = self._center_padding_columns(rendered_map)
        full_text, style_runs = _build_stc_content(
            rendered_map, left_padding_columns=left_padding_columns
        )
        first_line: int = self.map_ctrl.GetFirstVisibleLine()
        x_offset: int = self.map_ctrl.GetXOffset()
        self.map_ctrl.SetReadOnly(False)
        self.map_ctrl.SetText(full_text)
        self._update_map_scroll_width(full_text)
        self.map_ctrl.StartStyling(0)
        for byte_len, style_num in style_runs:
            self.map_ctrl.SetStyling(byte_len, style_num)
        self.map_ctrl.SetFirstVisibleLine(first_line)
        self.map_ctrl.SetXOffset(x_offset)
        self.map_ctrl.SetReadOnly(True)

    def _set_map_text_preserving_view(self, map_str: str) -> None:
        self._last_rendered_map = None
        first_line: int = self.map_ctrl.GetFirstVisibleLine()
        x_offset: int = self.map_ctrl.GetXOffset()
        self.map_ctrl.SetReadOnly(False)
        self.map_ctrl.SetText(map_str)
        self.map_ctrl.SetFirstVisibleLine(first_line)
        self.map_ctrl.SetXOffset(x_offset)
        self.map_ctrl.SetReadOnly(True)

    def _clear_map_ctrl(self) -> None:
        self._last_rendered_map = None
        self.map_ctrl.SetReadOnly(False)
        self.map_ctrl.ClearAll()
        self.map_ctrl.SetReadOnly(True)

    def _enable_run_button(self) -> None:
        if not self._frame_is_alive():
            return
        try:
            self.run_button.Enable()
        except wx.PyDeadObjectError:
            return

    def _frame_is_alive(self) -> bool:
        try:
            return not self.IsBeingDeleted()
        except wx.PyDeadObjectError:
            return False

    @staticmethod
    def _format_result(
        result: SimResult,
        event_count: int,
        snapshot_count: int,
        action_library_out: Path | None,
        replay: Path | None,
    ) -> str:
        lines: list[str] = [
            f"Seed: {result.seed}",
            f"Days elapsed: {result.days_elapsed:.2f}",
            f"Survived: {result.survived}",
            f"Death reason: {result.death_reason or 'n/a'}",
            (
                "Final needs: "
                f"health={result.final_health:.2f} thirst={result.final_thirst:.2f} "
                f"hunger={result.final_hunger:.2f} fatigue={result.final_fatigue:.2f} "
                f"cold_stress={result.final_cold_stress:.2f}"
            ),
            (
                "Discoveries: "
                f"water={result.water_discoveries} food={result.food_discoveries} "
                f"remembered_water={result.remembered_water_sites} "
                f"remembered_food={result.remembered_food_sites}"
            ),
            f"Distance walked: {result.distance_walked}",
            (
                "Final weather/cold: "
                f"temp_c={result.final_temperature_c:.1f} "
                f"feels_cold={result.final_feels_cold} "
                f"sheltered={result.final_is_sheltered}"
            ),
            (
                "Cold events: "
                f"weather={result.cold_weather_events} "
                f"status={result.cold_status_events} "
                f"shelter={result.shelter_events}"
            ),
            "Learning:",
            (
                "  Water memories: "
                f"{result.remembered_water_sites}, "
                f"Food memories: {result.remembered_food_sites}"
            ),
            (
                "  Memory-directed resource decisions: "
                f"{result.learning.memory_directed_resource_ticks}"
            ),
            (
                "  Exploration-directed resource decisions: "
                f"{result.learning.exploration_resource_ticks}"
            ),
            f"  Memory use ratio: {result.learning.memory_use_ratio:.2f}",
            (
                "  Reinforced memories: "
                f"water={result.learning.memory_reinforced_water} "
                f"food={result.learning.memory_reinforced_food}"
            ),
            (
                "  Failed memories: "
                f"water={result.learning.memory_failed_water} "
                f"food={result.learning.memory_failed_food}"
            ),
            f"Events logged: {event_count}",
            f"Snapshots stored: {snapshot_count}",
        ]
        if action_library_out is not None:
            lines.append(f"Wrote action library: {action_library_out}")
        if replay is not None:
            lines.append(f"Wrote replay report: {replay}")
        return "\n".join(lines)


def main() -> None:
    app = wx.App(False)
    frame = VillageSimFrame()
    frame.Show(True)
    app.MainLoop()
