"""wxPython GUI for running and viewing simulation output."""

from __future__ import annotations

import threading
import time

import wx  # type: ignore[import-not-found]

from village_sim.core.config import SimConfig
from village_sim.sim.engine import Simulation
from village_sim.sim.metrics import SimResult
from village_sim.view.ascii_view import render_ascii_map

MAP_UPDATE_INTERVAL_SECONDS: float = 0.05


class VillageSimFrame(wx.Frame):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__(parent=None, title="Village Sim MVP", size=(980, 760))
        panel = wx.Panel(self)
        root_sizer = wx.BoxSizer(wx.VERTICAL)

        controls_sizer = wx.FlexGridSizer(rows=3, cols=4, vgap=8, hgap=8)
        controls_sizer.AddGrowableCol(1, 1)
        controls_sizer.AddGrowableCol(3, 1)

        self.seed_ctrl = wx.SpinCtrl(panel, min=1, max=1_000_000, initial=1)
        self.days_ctrl = wx.SpinCtrl(panel, min=1, max=365, initial=10)
        self.width_ctrl = wx.SpinCtrl(panel, min=8, max=256, initial=32)
        self.height_ctrl = wx.SpinCtrl(panel, min=8, max=256, initial=32)
        self.tick_update_ctrl = wx.CheckBox(panel, label="Update map every tick")
        self.speed_ctrl = wx.SpinCtrl(panel, min=0, max=1000, initial=25)

        controls_sizer.Add(
            wx.StaticText(panel, label="Seed"), 0, wx.ALIGN_CENTER_VERTICAL
        )
        controls_sizer.Add(self.seed_ctrl, 1, wx.EXPAND)
        controls_sizer.Add(
            wx.StaticText(panel, label="Days"), 0, wx.ALIGN_CENTER_VERTICAL
        )
        controls_sizer.Add(self.days_ctrl, 1, wx.EXPAND)
        controls_sizer.Add(
            wx.StaticText(panel, label="Width"), 0, wx.ALIGN_CENTER_VERTICAL
        )
        controls_sizer.Add(self.width_ctrl, 1, wx.EXPAND)
        controls_sizer.Add(
            wx.StaticText(panel, label="Height"), 0, wx.ALIGN_CENTER_VERTICAL
        )
        controls_sizer.Add(self.height_ctrl, 1, wx.EXPAND)
        controls_sizer.Add(self.tick_update_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add((0, 0), 1, wx.EXPAND)
        controls_sizer.Add(
            wx.StaticText(panel, label="Tick delay (ms)"), 0, wx.ALIGN_CENTER_VERTICAL
        )
        controls_sizer.Add(self.speed_ctrl, 1, wx.EXPAND)

        self.run_button = wx.Button(panel, label="Run Simulation")
        self.run_button.Bind(wx.EVT_BUTTON, self.on_run)

        self.summary_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_NONE,
            size=(-1, 120),
        )
        self.summary_ctrl.SetBackgroundColour(panel.GetBackgroundColour())
        self.map_ctrl = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.TE_DONTWRAP
        )
        self.map_ctrl.SetFont(wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE)))

        root_sizer.Add(controls_sizer, 0, wx.EXPAND | wx.ALL, 12)
        root_sizer.Add(self.run_button, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        root_sizer.Add(
            wx.StaticText(panel, label="Run Summary"), 0, wx.LEFT | wx.RIGHT, 12
        )
        root_sizer.Add(
            self.summary_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12
        )
        root_sizer.Add(
            wx.StaticText(panel, label="ASCII Map"), 0, wx.LEFT | wx.RIGHT, 12
        )
        root_sizer.Add(self.map_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        panel.SetSizer(root_sizer)
        self.Centre()

    def on_run(self, _: wx.CommandEvent) -> None:
        self.run_button.Disable()
        self.summary_ctrl.Clear()
        self.map_ctrl.Clear()
        config = SimConfig(
            width=self.width_ctrl.GetValue(),
            height=self.height_ctrl.GetValue(),
            max_days=self.days_ctrl.GetValue(),
            seed=self.seed_ctrl.GetValue(),
        )
        update_every_tick: bool = self.tick_update_ctrl.GetValue()
        tick_delay_seconds: float = self.speed_ctrl.GetValue() / 1000.0

        def thread_target() -> None:
            try:
                sim = Simulation(config)
                max_ticks: int = config.max_ticks()
                last_map_update_time: float = (
                    time.monotonic() - MAP_UPDATE_INTERVAL_SECONDS
                )
                while sim.tick < max_ticks and sim.agent.alive:
                    sim.step()
                    if update_every_tick:
                        now: float = time.monotonic()
                        if now - last_map_update_time >= MAP_UPDATE_INTERVAL_SECONDS:
                            current_map: str = render_ascii_map(sim.world, sim.agent)
                            wx.CallAfter(self._set_map_value, current_map)
                            last_map_update_time = now
                    if tick_delay_seconds > 0.0:
                        time.sleep(tick_delay_seconds)
                result: SimResult = sim.result()
                summary: str = self._format_result(result, len(sim.events))
                map_str: str = render_ascii_map(sim.world, sim.agent)
                wx.CallAfter(self._update_ui, summary, map_str)
            except Exception:
                wx.CallAfter(
                    wx.MessageBox,
                    "Simulation failed. Please check your inputs and try again.",
                    "Error",
                    wx.OK | wx.ICON_ERROR,
                )
            finally:
                wx.CallAfter(self._enable_run_button)

        threading.Thread(target=thread_target, daemon=True).start()

    def _update_ui(self, summary: str, map_str: str) -> None:
        if not self._frame_is_alive():
            return
        try:
            self.summary_ctrl.SetValue(summary)
            self.map_ctrl.SetValue(map_str)
        except wx.PyDeadObjectError:
            return

    def _set_map_value(self, map_str: str) -> None:
        if not self._frame_is_alive():
            return
        try:
            self.map_ctrl.SetValue(map_str)
        except wx.PyDeadObjectError:
            return

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
    def _format_result(result: SimResult, event_count: int) -> str:
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
            f"Events logged: {event_count}",
        ]
        return "\n".join(lines)


def main() -> None:
    app = wx.App(False)
    frame = VillageSimFrame()
    frame.Show(True)
    app.MainLoop()
