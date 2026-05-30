"""Command-line runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from village_sim.agent.state import MAX_AGENTS
from village_sim.core.config import SimConfig
from village_sim.orchestrator.action_model import ActionLibrary
from village_sim.sim.engine import Simulation
from village_sim.sim.metrics import SimResult
from village_sim.sim.replay import write_run_report
from village_sim.view.ascii_view import (
    render_agent_arrays_map_model,
    rendered_map_to_text,
)


def _parse_agent_count(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("agents must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("agents must be at least 1")
    if value > MAX_AGENTS:
        raise argparse.ArgumentTypeError(f"agents must be at most {MAX_AGENTS}")
    return value


def _config_from_args(
    args: argparse.Namespace, *, seed: int | None = None
) -> SimConfig:
    return SimConfig(
        width=args.width,
        height=args.height,
        max_days=args.days,
        seed=args.seed if seed is None else seed,
        initial_agents=getattr(args, "agents", 1),
        enable_initial_discoverables=args.discoverables,
        enable_goap_control=args.goap,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Village Sim MVP.")
    parser.add_argument(
        "--wx",
        action="store_true",
        help="launch the wxPython interface instead of running in the terminal",
    )
    parser.add_argument("--seed", type=int, default=1, help="deterministic RNG seed")
    parser.add_argument("--days", type=int, default=10, help="number of simulated days")
    parser.add_argument(
        "--agents",
        type=_parse_agent_count,
        default=1,
        help=f"initial active agent count, from 1 to {MAX_AGENTS}",
    )
    parser.add_argument("--width", type=int, default=32, help="world width in cells")
    parser.add_argument("--height", type=int, default=32, help="world height in cells")
    parser.add_argument(
        "--print-map",
        action="store_true",
        help="print an ASCII map at the end of the run",
    )
    parser.add_argument(
        "--discoverables",
        action="store_true",
        help="seed the live world with spring_001, berry_bush_001, and cave_001",
    )
    parser.add_argument(
        "--goap",
        action="store_true",
        help="enable bounded GOAP control for urgent discoverable needs",
    )
    parser.add_argument(
        "--action-library-in",
        type=Path,
        default=None,
        help="optional MessagePack action library to load before running",
    )
    parser.add_argument(
        "--action-library-out",
        type=Path,
        default=None,
        help="optional path to save the action library after running",
    )
    parser.add_argument(
        "--local-map-radius",
        type=int,
        default=0,
        help="when printing a map, print only this radius around the agent; 0 prints full map",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="store ASCII snapshots every N ticks in the replay MessagePack",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="optional path for MessagePack run report",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="run N seeds starting from --seed and print aggregate results",
    )
    return parser


def main() -> None:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args()

    if args.wx:
        launch_wx_interface(args)
        return

    if args.batch > 1:
        run_batch(args)
        return

    config = _config_from_args(args)
    sim = Simulation(config)
    if args.action_library_in is not None:
        sim.action_library = ActionLibrary.load(args.action_library_in)
    result: SimResult = sim.run(snapshot_every=args.snapshot_every)
    print_result(result)
    print_learning_summary(result)
    print_cold_summary(result)

    if args.print_map:
        radius: int | None = (
            None if args.local_map_radius <= 0 else args.local_map_radius
        )
        rendered = render_agent_arrays_map_model(
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
        print(rendered_map_to_text(rendered))

    if args.action_library_out is not None:
        sim.action_library.save(args.action_library_out)
        print(f"Wrote action library: {args.action_library_out}")

    if args.replay is not None:
        write_run_report(args.replay, config, result, sim.events, sim.snapshots)
        print(f"Wrote replay report: {args.replay}")


def run_batch(args: argparse.Namespace) -> None:
    results: list[SimResult] = []
    for offset in range(args.batch):
        config = _config_from_args(args, seed=args.seed + offset)
        sim = Simulation(config)
        results.append(sim.run())

    survived_count: int = sum(1 for result in results if result.survived)
    average_days: float = sum(result.days_elapsed for result in results) / float(
        len(results)
    )
    average_distance: float = sum(
        result.total_distance_walked for result in results
    ) / float(len(results))
    print(f"Batch runs: {len(results)}")
    print(f"Survived full duration: {survived_count}/{len(results)}")
    print(f"Average days elapsed: {average_days:.2f}")
    print(f"Average distance walked: {average_distance:.1f}")
    print(
        "initial_agents,final_active_agents,dead_agents,seed,days,survived,death,"
        "water_sites,food_sites,total_distance,avg_distance,final_cold_stress,final_temperature_c,"
        "final_feels_cold,final_is_sheltered,cold_weather_events,"
        "cold_status_events,shelter_events"
    )
    for result in results:
        print(
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


def print_learning_summary(result: SimResult) -> None:
    learning = result.learning
    print("Learning:")
    print(
        f"  learned water sites={learning.learned_water_sites} "
        f"food sites={learning.learned_food_sites}"
    )
    print(
        f"  remembered water selections={learning.memory_selected_water} "
        f"remembered food selections={learning.memory_selected_food}"
    )
    print(
        f"  memory reinforcements: water={learning.memory_reinforced_water} "
        f"food={learning.memory_reinforced_food}"
    )
    print(
        f"  memory failures: water={learning.memory_failed_water} "
        f"food={learning.memory_failed_food}"
    )
    print(
        "  resource decisions: "
        f"memory-directed={learning.memory_directed_resource_ticks} "
        f"explore-directed={learning.exploration_resource_ticks} "
        f"memory_use_ratio={learning.memory_use_ratio:.2f}"
    )
    if result.best_water_memory_x >= 0 and result.best_water_memory_y >= 0:
        print(
            "  water memory: "
            f"x={result.best_water_memory_x} y={result.best_water_memory_y} "
            f"confidence={result.best_water_memory_confidence:.2f} "
            f"uses={result.best_water_memory_successful_uses} "
            f"failures={result.best_water_memory_failed_uses}"
        )
    if result.best_food_memory_x >= 0 and result.best_food_memory_y >= 0:
        print(
            "  food memory: "
            f"x={result.best_food_memory_x} y={result.best_food_memory_y} "
            f"confidence={result.best_food_memory_confidence:.2f} "
            f"uses={result.best_food_memory_successful_uses} "
            f"failures={result.best_food_memory_failed_uses}"
        )


def print_cold_summary(result: SimResult) -> None:
    print(
        "Cold/weather: "
        f"temp_c={result.final_temperature_c:.1f} "
        f"feels_cold={result.final_feels_cold} "
        f"sheltered={result.final_is_sheltered} "
        f"cold_weather_events={result.cold_weather_events} "
        f"cold_status_events={result.cold_status_events} "
        f"shelter_events={result.shelter_events}"
    )


def print_result(result: SimResult) -> None:
    print(f"Seed: {result.seed}")
    print(
        f"Population: initial={result.initial_agents} "
        f"final_active={result.final_active_agents} dead={result.dead_agents}"
    )
    print(f"Days elapsed: {result.days_elapsed:.2f}")
    print(f"Survived: {result.survived}")
    if result.death_reason is not None:
        print(f"Death reason: {result.death_reason}")
    print(
        "Active averages: "
        f"health={result.active_average_health:.2f} "
        f"thirst={result.active_average_thirst:.2f} "
        f"hunger={result.active_average_hunger:.2f} "
        f"fatigue={result.active_average_fatigue:.2f} "
        f"cold_stress={result.active_average_cold_stress:.2f}"
    )
    print(
        "Primary agent final needs: "
        f"health={result.final_health:.2f} "
        f"thirst={result.final_thirst:.2f} "
        f"hunger={result.final_hunger:.2f} "
        f"fatigue={result.final_fatigue:.2f} "
        f"cold_stress={result.final_cold_stress:.2f}"
    )
    print(
        "Primary agent discoveries: "
        f"water={result.water_discoveries} food={result.food_discoveries} "
        f"remembered_water={result.remembered_water_sites} "
        f"remembered_food={result.remembered_food_sites}"
    )
    print(f"Total distance walked: {result.total_distance_walked}")
    print(f"Average distance walked: {result.average_distance_walked:.1f}")
    print(f"Deaths by reason: {result.deaths_by_reason}")


def launch_wx_interface(args: argparse.Namespace | None = None) -> None:
    try:
        from village_sim.view.wx_view import GuiInitialOptions, main as wx_main
    except ImportError as exc:
        if getattr(exc, "name", None) == "wx":
            raise SystemExit(
                'wxPython is required for --wx. Install it with: python -m pip install "village-sim-mvp[gui]"'
            ) from exc
        raise

    initial_options: GuiInitialOptions | None = None
    if args is not None:
        initial_options = GuiInitialOptions(
            seed=args.seed,
            days=args.days,
            width=args.width,
            height=args.height,
            initial_agents=args.agents,
            print_map=args.print_map,
            batch=args.batch,
            discoverables=args.discoverables,
            goap=args.goap,
            action_library_in=args.action_library_in,
            action_library_out=args.action_library_out,
            local_map_radius=args.local_map_radius,
            snapshot_every=args.snapshot_every,
            replay=args.replay,
        )
    wx_main(initial_options)


if __name__ == "__main__":
    main()
