"""MessagePack replay/report writing."""

from __future__ import annotations

import msgpack
from pathlib import Path
from typing import Any

from village_sim.core.config import SimConfig
from village_sim.sim.events import TickEvent
from village_sim.sim.metrics import SimResult
from village_sim.msgpack_codec import pack_default, unpack_object_hook
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
        "schema_version": 2,
        "config": {
            "width": config.width,
            "height": config.height,
            "max_days": config.max_days,
            "ticks_per_day": config.ticks_per_day,
            "seed": config.seed,
            "initial_agents": config.initial_agents,
            "day_temperature_c": config.day_temperature_c,
            "night_temperature_c": config.night_temperature_c,
            "rain_temperature_penalty_c": config.rain_temperature_penalty_c,
            "cold_temperature_threshold_c": config.cold_temperature_threshold_c,
        },
        "result": result,
        "events": events,
        "snapshots": snapshots,
    }
    with path.open("wb") as report_file:
        msgpack.pack(payload, report_file, default=pack_default, use_bin_type=True)


def read_run_report(path: Path) -> dict[str, Any]:
    with path.open("rb") as report_file:
        raw: object = msgpack.unpack(
            report_file, raw=False, object_hook=unpack_object_hook
        )
    if not isinstance(raw, dict):
        raise ValueError("run report file must contain a MessagePack map")
    return raw
