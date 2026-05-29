"""JSON replay/report writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from village_sim.core.config import SimConfig
from village_sim.sim.events import TickEvent
from village_sim.sim.metrics import SimResult
from village_sim.sim.snapshot import WorldSnapshot


def write_run_report(
    path: Path,
    config: SimConfig,
    result: SimResult,
    events: list[TickEvent],
    snapshots: list[WorldSnapshot],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "width": config.width,
            "height": config.height,
            "max_days": config.max_days,
            "ticks_per_day": config.ticks_per_day,
            "seed": config.seed,
            "day_temperature_c": config.day_temperature_c,
            "night_temperature_c": config.night_temperature_c,
            "rain_temperature_penalty_c": config.rain_temperature_penalty_c,
            "cold_temperature_threshold_c": config.cold_temperature_threshold_c,
        },
        "result": result.to_json_obj(),
        "events": [event.to_json_obj() for event in events],
        "snapshots": [snapshot.to_json_obj() for snapshot in snapshots],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
