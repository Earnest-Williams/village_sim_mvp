"""Population UI/reporting regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from village_sim.core.config import SimConfig
from village_sim.core.types import Position
from village_sim.sim.engine import Simulation
from village_sim.view.ascii_view import render_agent_arrays_map_model
from village_sim.world.discoverables import Discoverable, DiscoverableKind


def test_population_renderer_represents_two_active_agents() -> None:
    sim = Simulation(
        SimConfig(width=16, height=16, max_days=1, seed=7, initial_agents=2)
    )

    rendered = render_agent_arrays_map_model(
        sim.world,
        sim.agents,
        selected_agent_index=0,
        tick=sim.tick,
        day=0,
        temperature_c=sim.current_weather.temperature_c,
        is_raining=sim.current_weather.is_raining,
        feels_cold=sim.current_weather.feels_cold,
    )
    glyphs = "".join(glyph.char for row in rendered.rows for glyph in row)

    assert "active=2" in rendered.status
    assert "Selected agent:" in rendered.status
    assert glyphs.count("@") + glyphs.count("+") >= 1
    assert glyphs.count("a") + glyphs.count("+") >= 1


def test_result_exposes_population_level_summary_fields() -> None:
    sim = Simulation(
        SimConfig(width=16, height=16, max_days=1, seed=8, initial_agents=2)
    )
    sim.step()

    result = sim.result()

    assert result.initial_agents == 2
    assert result.final_active_agents == int(np.count_nonzero(sim.agents.active))
    assert result.dead_agents >= 0
    assert result.active_average_health > 0.0
    assert result.total_distance_walked >= result.distance_walked
    assert result.average_distance_walked >= 0.0
    assert isinstance(result.deaths_by_reason, dict)
    assert result.total_memory_directed_decisions >= 0
    assert result.total_exploration_directed_decisions >= 0


def test_multi_agent_shelter_mask_recovers_only_sheltered_agent() -> None:
    sim = Simulation(SimConfig(width=8, height=8, max_days=1, seed=9, initial_agents=2))
    sim.world.discoverables["test_cave"] = Discoverable(
        discoverable_id="test_cave",
        kind=DiscoverableKind.CAVE,
        x=2,
        y=2,
        visible_name="test cave",
        discovered=True,
        amount=10_000.0,
        max_amount=10_000.0,
        regrowth_per_day=0.0,
        satisfies_need="cold_stress",
        need_delta=-0.5,
        interaction_ticks=1,
    )
    sim.agents.x[0] = np.int32(2)
    sim.agents.y[0] = np.int32(3)
    sim.agents.x[1] = np.int32(7)
    sim.agents.y[1] = np.int32(7)
    sim.agents.cold_stress[:2] = np.float32(0.5)

    shelter_mask = sim.agent_sheltered_mask()
    sim._update_agent_needs(
        is_night=True,
        is_raining=False,
        is_sheltered=shelter_mask,
        is_cold_exposed=True,
    )

    assert bool(shelter_mask[0])
    assert not bool(shelter_mask[1])
    assert float(sim.agents.cold_stress[0]) < 0.5
    assert float(sim.agents.cold_stress[1]) > 0.5


def test_gui_and_lifecycle_source_use_population_paths() -> None:
    wx_source = Path("src/village_sim/view/wx_view.py").read_text()
    engine_source = Path("src/village_sim/sim/engine.py").read_text()

    assert "render_agent_arrays_map_model" in wx_source
    assert "render_map_model(sim.world, sim.agent" not in wx_source
    assert "bool(np.any(self.agents.active))" in engine_source
    assert (
        "while self.tick < self.config.max_ticks() and bool(self.agents.active[0])"
        not in engine_source
    )
