"""wxPython GUI for running and viewing simulation output."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import wx
import wx.stc as wxstc

from village_sim.agent.memory import MEMORY_AGENT_ID, MEMORY_KIND
from village_sim.agent.state import ID_TO_ACTION, ID_TO_DEATH, ID_TO_GOAL, MAX_AGENTS
from village_sim.core.config import SimConfig
from village_sim.orchestrator.action_model import ActionLibrary
from village_sim.sim.engine import Simulation
from village_sim.sim.metrics import SimResult
from village_sim.sim.replay import write_run_report
from village_sim.view.ascii_view import (
    ROLE_COLORS,
    RenderedMap,
    render_agent_arrays_map_model,
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
class GuiInitialOptions:
    """Initial values used to seed wx controls from the CLI."""

    seed: int = 1
    days: int = 10
    width: int = GUI_DEFAULT_WORLD_SIZE
    height: int = GUI_DEFAULT_WORLD_SIZE
    initial_agents: int = 1
    print_map: bool = True
    batch: int = 1
    discoverables: bool = False
    goap: bool = False
    action_library_in: Path | None = None
    action_library_out: Path | None = None
    local_map_radius: int = 0
    snapshot_every: int = 0
    replay: Path | None = None


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


class VillageSimFrame(wx.Frame):
    def __init__(self, initial_options: GuiInitialOptions | None = None) -> None:
        options = (
            initial_options if initial_options is not None else GuiInitialOptions()
        )
        super().__init__(parent=None, title="Village Sim MVP", size=(980, 840))
        panel = wx.Panel(self)
        root_sizer = wx.BoxSizer(wx.VERTICAL)
        self._build_menu_bar()
        self.setup_panel = wx.Panel(panel)

        controls_sizer = wx.FlexGridSizer(rows=10, cols=4, vgap=8, hgap=8)
        controls_sizer.AddGrowableCol(1, 1)
        controls_sizer.AddGrowableCol(3, 1)

        self.seed_ctrl = wx.SpinCtrl(
            self.setup_panel, min=1, max=1_000_000, initial=options.seed
        )
        self.days_ctrl = wx.SpinCtrl(
            self.setup_panel, min=1, max=365, initial=options.days
        )
        self.agents_ctrl = wx.SpinCtrl(
            self.setup_panel, min=1, max=MAX_AGENTS, initial=options.initial_agents
        )
        self.width_ctrl = wx.SpinCtrl(
            self.setup_panel, min=8, max=256, initial=options.width
        )
        self.height_ctrl = wx.SpinCtrl(
            self.setup_panel, min=8, max=256, initial=options.height
        )
        self.batch_ctrl = wx.SpinCtrl(
            self.setup_panel, min=1, max=10_000, initial=options.batch
        )
        self.local_map_radius_ctrl = wx.SpinCtrl(
            self.setup_panel, min=0, max=512, initial=options.local_map_radius
        )
        self.snapshot_every_ctrl = wx.SpinCtrl(
            self.setup_panel, min=0, max=1_000_000, initial=options.snapshot_every
        )
        self.speed_ctrl = wx.SpinCtrl(self.setup_panel, min=0, max=1000, initial=25)
        self.map_font_size_ctrl = wx.SpinCtrl(
            self.setup_panel, min=6, max=24, initial=MAP_DEFAULT_FONT_POINT_SIZE
        )
        self.print_map_ctrl = wx.CheckBox(
            self.setup_panel, label="Render final ASCII map"
        )
        self.print_map_ctrl.SetValue(options.print_map)
        self.discoverables_ctrl = wx.CheckBox(
            self.setup_panel,
            label="Seed canonical discoverables",
        )
        self.discoverables_ctrl.SetValue(options.discoverables)
        self.goap_ctrl = wx.CheckBox(self.setup_panel, label="Enable GOAP control")
        self.goap_ctrl.SetValue(options.goap)
        self.tick_update_ctrl = wx.CheckBox(
            self.setup_panel, label="Update map every tick"
        )
        self.action_library_in_ctrl = wx.TextCtrl(self.setup_panel)
        self.action_library_out_ctrl = wx.TextCtrl(self.setup_panel)
        self.replay_ctrl = wx.TextCtrl(self.setup_panel)
        if options.action_library_in is not None:
            self.action_library_in_ctrl.SetValue(str(options.action_library_in))
        if options.action_library_out is not None:
            self.action_library_out_ctrl.SetValue(str(options.action_library_out))
        if options.replay is not None:
            self.replay_ctrl.SetValue(str(options.replay))

        self._add_labeled_control(
            controls_sizer, self.setup_panel, "Seed", self.seed_ctrl
        )
        self._add_labeled_control(
            controls_sizer, self.setup_panel, "Days", self.days_ctrl
        )
        self._add_labeled_control(
            controls_sizer, self.setup_panel, "Agents", self.agents_ctrl
        )
        self._add_labeled_control(
            controls_sizer, self.setup_panel, "Width", self.width_ctrl
        )
        self._add_labeled_control(
            controls_sizer, self.setup_panel, "Height", self.height_ctrl
        )
        self._add_labeled_control(
            controls_sizer, self.setup_panel, "Batch runs", self.batch_ctrl
        )
        self._add_labeled_control(
            controls_sizer,
            self.setup_panel,
            "Local map radius",
            self.local_map_radius_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            self.setup_panel,
            "Snapshot every N ticks",
            self.snapshot_every_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            self.setup_panel,
            "Tick delay (ms)",
            self.speed_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            self.setup_panel,
            "Map font size",
            self.map_font_size_ctrl,
        )
        controls_sizer.Add(self.print_map_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add(self.discoverables_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add(self.goap_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add(self.tick_update_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        self._add_labeled_control(
            controls_sizer,
            self.setup_panel,
            "Action library in",
            self.action_library_in_ctrl,
        )
        self._add_labeled_control(
            controls_sizer,
            self.setup_panel,
            "Action library out",
            self.action_library_out_ctrl,
        )
        self._add_labeled_control(
            controls_sizer, self.setup_panel, "Replay MessagePack", self.replay_ctrl
        )
        controls_sizer.Add((0, 0), 1, wx.EXPAND)
        self.setup_panel.SetSizer(controls_sizer)

        self.run_button = wx.Button(panel, label="Run Simulation")
        self.run_button.Bind(wx.EVT_BUTTON, self.on_run)

        self.summary_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_NONE,
            size=(-1, 150),
        )
        self.summary_ctrl.SetBackgroundColour(panel.GetBackgroundColour())
        self.selected_agent_index: int = 0
        self.selected_agent_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_NONE,
            size=(-1, 70),
        )
        self.selected_agent_ctrl.SetBackgroundColour(panel.GetBackgroundColour())
        self.roster_ctrl = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)  # type: ignore[attr-defined]
        self._configure_roster_ctrl()
        self.roster_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_roster_selected)  # type: ignore[attr-defined]
        self.map_ctrl = wxstc.StyledTextCtrl(panel, style=wx.BORDER_NONE)
        self._last_rendered_map: RenderedMap | None = None
        self._configure_map_ctrl()
        self.map_font_size_ctrl.Bind(wx.EVT_SPINCTRL, self.on_map_font_size_changed)
        self.map_font_size_ctrl.Bind(wx.EVT_TEXT, self.on_map_font_size_changed)

        root_sizer.Add(self.setup_panel, 0, wx.EXPAND | wx.ALL, 12)
        root_sizer.Add(self.run_button, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        root_sizer.Add(
            wx.StaticText(panel, label="Run Summary"), 0, wx.LEFT | wx.RIGHT, 12
        )
        root_sizer.Add(
            self.summary_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12
        )
        root_sizer.Add(
            wx.StaticText(panel, label="Selected Agent"), 0, wx.LEFT | wx.RIGHT, 12
        )
        root_sizer.Add(
            self.selected_agent_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12
        )
        root_sizer.Add(
            wx.StaticText(panel, label="Agent Roster"), 0, wx.LEFT | wx.RIGHT, 12
        )
        root_sizer.Add(
            self.roster_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12
        )
        root_sizer.Add(wx.StaticText(panel, label="Map"), 0, wx.LEFT | wx.RIGHT, 12)
        root_sizer.Add(self.map_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        panel.SetSizer(root_sizer)
        self.Centre()

    def _build_menu_bar(self) -> None:
        menu_bar = wx.MenuBar()  # type: ignore[attr-defined]
        sim_menu = wx.Menu()  # type: ignore[attr-defined]
        for label in (
            "New Simulation",
            "Start",
            "Pause",
            "Resume",
            "Stop",
            "Step One Tick",
            "Save Replay",
            "Load Replay",
        ):
            item = sim_menu.Append(wx.ID_ANY, label)  # type: ignore[attr-defined]
            if label in {"New Simulation", "Start"}:
                self.Bind(wx.EVT_MENU, self.on_run, item)  # type: ignore[attr-defined, call-arg]
        view_menu = wx.Menu()  # type: ignore[attr-defined]
        for label in (
            "View Full Map",
            "View Local Map",
            "Follow Selected Agent",
            "Show Roster",
            "Show Setup Panel",
            "Font Size",
        ):
            item = view_menu.Append(wx.ID_ANY, label)  # type: ignore[attr-defined]
            if label == "Show Setup Panel":
                self.Bind(wx.EVT_MENU, self.on_show_setup_panel, item)  # type: ignore[attr-defined, call-arg]
        menu_bar.Append(sim_menu, "Simulation")
        menu_bar.Append(view_menu, "View")
        self.SetMenuBar(menu_bar)  # type: ignore[attr-defined]

    def on_show_setup_panel(self, _: wx.CommandEvent) -> None:
        self.setup_panel.Show(not self.setup_panel.IsShown())  # type: ignore[attr-defined]
        self.Layout()  # type: ignore[attr-defined]

    def _configure_roster_ctrl(self) -> None:
        columns: tuple[str, ...] = (
            "id",
            "active",
            "x",
            "y",
            "action",
            "goal",
            "health",
            "thirst",
            "hunger",
            "fatigue",
            "cold",
            "distance",
            "water memories",
            "food memories",
            "death reason",
        )
        for column_index, label in enumerate(columns):
            self.roster_ctrl.InsertColumn(column_index, label)
            self.roster_ctrl.SetColumnWidth(column_index, 92)

    def on_roster_selected(self, event: Any) -> None:
        self.selected_agent_index = event.GetIndex()

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
        margins_background_method: object = getattr(
            self.map_ctrl, "SetMarginsBackground", None
        )
        if callable(margins_background_method):
            margins_background_method(map_bg)
        fold_margin_colour_method: object = getattr(
            self.map_ctrl, "SetFoldMarginColour", None
        )
        if callable(fold_margin_colour_method):
            fold_margin_colour_method(True, map_bg)
        fold_margin_highlight_method: object = getattr(
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
        margin_count_method: object = getattr(self.map_ctrl, "GetMargins", None)
        raw_margin_count: object = None
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
        raw_client_width: object = self.map_ctrl.GetClientSize().width
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
        raw_client_width: object = self.map_ctrl.GetClientSize().width
        client_width = 0
        if isinstance(raw_client_width, int) and raw_client_width > 0:
            client_width = raw_client_width
        max_columns = max((len(line) for line in full_text.splitlines()), default=0)
        scroll_width = max(client_width, (max_columns + 1) * raw_char_width)
        self.map_ctrl.SetScrollWidth(scroll_width)

    def on_run(self, _: wx.CommandEvent) -> None:
        self.run_button.Disable()
        self.setup_panel.Hide()  # type: ignore[attr-defined]
        self.Layout()  # type: ignore[attr-defined]
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
            initial_agents=self.agents_ctrl.GetValue(),
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
        while sim.tick < max_ticks and bool(np.any(sim.agents.active)):
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
        wx.CallAfter(self._update_roster, sim)
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
            config = replace(options.config, seed=options.config.seed + offset)
            sim = Simulation(config)
            results.append(sim.run())

        survived_count: int = sum(1 for result in results if result.survived)
        average_days: float = sum(result.days_elapsed for result in results) / float(
            len(results)
        )
        average_distance: float = sum(
            result.total_distance_walked for result in results
        ) / float(len(results))
        lines: list[str] = [
            f"Batch runs: {len(results)}",
            f"Survived full duration: {survived_count}/{len(results)}",
            f"Average days elapsed: {average_days:.2f}",
            f"Average distance walked: {average_distance:.1f}",
            (
                "initial_agents,final_active_agents,dead_agents,seed,days,survived,death,"
                "water_sites,food_sites,total_distance,avg_distance,final_cold_stress,"
                "final_temperature_c,final_feels_cold,final_is_sheltered,"
                "cold_weather_events,cold_status_events,shelter_events"
            ),
        ]
        for result in results:
            lines.append(
                f"{result.initial_agents},{result.final_active_agents},"
                f"{result.dead_agents},{result.seed},{result.days_elapsed:.2f},"
                f"{result.survived},{result.death_reason},"
                f"{result.total_water_memories},{result.total_food_memories},"
                f"{result.total_distance_walked},{result.average_distance_walked:.1f},"
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

    def _render_map_model(self, sim: Simulation, local_map_radius: int) -> RenderedMap:
        radius: int | None = None
        if local_map_radius > 0:
            radius = local_map_radius
        sim.selected_agent_index = self.selected_agent_index
        return render_agent_arrays_map_model(
            sim.world,
            sim.agents,
            sim.selected_agent_index,
            radius=radius,
            tick=sim.tick,
            day=sim.tick // sim.config.ticks_per_day,
            temperature_c=sim.current_weather.temperature_c,
            is_raining=sim.current_weather.is_raining,
            feels_cold=sim.current_weather.feels_cold,
        )

    def _update_roster(self, sim: Simulation) -> None:
        if not self._frame_is_alive():
            return
        self.roster_ctrl.DeleteAllItems()
        active_or_used = np.flatnonzero(
            sim.agents.active | (sim.agent_ids > np.int64(0))
        ).astype(np.int64)
        memory_counts = _agent_memory_counts(sim)
        for row_index, agent_index in enumerate(active_or_used):
            action = ID_TO_ACTION.get(int(sim.agents.current_action[agent_index]))
            goal = ID_TO_GOAL.get(int(sim.agents.current_goal[agent_index]))
            action_value = "unknown" if action is None else action.value
            goal_value = "unknown" if goal is None else goal.value
            reason_value = ""
            if int(sim.agents.death_reason[agent_index]) >= 0:
                death_reason = ID_TO_DEATH.get(
                    int(sim.agents.death_reason[agent_index])
                )
                reason_value = "unknown" if death_reason is None else death_reason.value
            agent_id = int(sim.agent_ids[agent_index])
            water_count, food_count = memory_counts.get(agent_id, (0, 0))
            values = (
                str(agent_id),
                str(bool(sim.agents.active[agent_index])),
                str(int(sim.agents.x[agent_index])),
                str(int(sim.agents.y[agent_index])),
                action_value,
                goal_value,
                f"{float(sim.agents.health[agent_index]):.2f}",
                f"{float(sim.agents.thirst[agent_index]):.2f}",
                f"{float(sim.agents.hunger[agent_index]):.2f}",
                f"{float(sim.agents.fatigue[agent_index]):.2f}",
                f"{float(sim.agents.cold_stress[agent_index]):.2f}",
                str(int(sim.agents.distance_walked[agent_index])),
                str(water_count),
                str(food_count),
                reason_value,
            )
            self.roster_ctrl.InsertItem(row_index, values[0])
            for column_index, value in enumerate(values[1:], start=1):
                self.roster_ctrl.SetItem(row_index, column_index, value)
        self.selected_agent_ctrl.SetValue(
            _selected_agent_detail(sim, self.selected_agent_index)
        )

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
            (
                "Population: "
                f"initial={result.initial_agents} active={result.final_active_agents} "
                f"dead={result.dead_agents}"
            ),
            f"Days elapsed: {result.days_elapsed:.2f}",
            f"Survived: {result.survived}",
            f"Death reason: {result.death_reason or 'n/a'}",
            (
                "Active averages: "
                f"health={result.active_average_health:.2f} "
                f"thirst={result.active_average_thirst:.2f} "
                f"hunger={result.active_average_hunger:.2f} "
                f"fatigue={result.active_average_fatigue:.2f} "
                f"cold={result.active_average_cold_stress:.2f}"
            ),
            (
                "Primary agent final needs: "
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
            f"Total distance walked: {result.total_distance_walked}",
            f"Average distance walked: {result.average_distance_walked:.1f}",
            (
                "Final weather/cold: "
                f"temp_c={result.final_temperature_c:.1f} "
                f"feels_cold={result.final_feels_cold} "
                f"sheltered={result.final_is_sheltered}"
            ),
            (
                "Population events: "
                f"cold={result.total_cold_events} shelter={result.total_shelter_events} "
                f"deaths_by_reason={result.deaths_by_reason}"
            ),
            "Learning:",
            (
                "  Population memories: "
                f"water={result.total_water_memories}, "
                f"food={result.total_food_memories}"
            ),
            (
                "  Memory-directed resource decisions: "
                f"{result.total_memory_directed_decisions}"
            ),
            (
                "  Exploration-directed resource decisions: "
                f"{result.total_exploration_directed_decisions}"
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


def _agent_memory_counts(sim: Simulation) -> dict[int, tuple[int, int]]:
    sim.global_memory.flush_pending()
    if sim.global_memory.frame.is_empty():
        return {}
    grouped = (
        sim.global_memory.frame.group_by([MEMORY_AGENT_ID, MEMORY_KIND])
        .len()
        .sort([MEMORY_AGENT_ID, MEMORY_KIND])
    )
    counts: dict[int, tuple[int, int]] = {}
    for values in grouped.iter_rows(named=True):
        agent_id = int(values[MEMORY_AGENT_ID])
        kind = int(values[MEMORY_KIND])
        count = int(values["len"])
        water_count, food_count = counts.get(agent_id, (0, 0))
        if kind == 1:
            water_count = count
        elif kind == 2:
            food_count = count
        counts[agent_id] = (water_count, food_count)
    return counts


def _selected_agent_detail(sim: Simulation, selected_agent_index: int) -> str:
    if not (0 <= selected_agent_index < sim.agents.count):
        return "Selected agent: none"
    if int(sim.agent_ids[selected_agent_index]) <= 0:
        return "Selected agent: none"
    action = ID_TO_ACTION.get(int(sim.agents.current_action[selected_agent_index]))
    goal = ID_TO_GOAL.get(int(sim.agents.current_goal[selected_agent_index]))
    action_value = "unknown" if action is None else action.value
    goal_value = "unknown" if goal is None else goal.value
    return (
        f"Selected agent id={int(sim.agent_ids[selected_agent_index])} "
        f"x={int(sim.agents.x[selected_agent_index])} "
        f"y={int(sim.agents.y[selected_agent_index])} goal={goal_value} "
        f"action={action_value} health={float(sim.agents.health[selected_agent_index]):.2f} "
        f"thirst={float(sim.agents.thirst[selected_agent_index]):.2f} "
        f"hunger={float(sim.agents.hunger[selected_agent_index]):.2f} "
        f"fatigue={float(sim.agents.fatigue[selected_agent_index]):.2f} "
        f"cold={float(sim.agents.cold_stress[selected_agent_index]):.2f}"
    )


def main(initial_options: GuiInitialOptions | None = None) -> None:
    app = wx.App(False)
    frame = VillageSimFrame(initial_options)
    frame.Show(True)
    app.MainLoop()
