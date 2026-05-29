# Village Sim MVP

A Python-only, headless-first survival simulation MVP for a low-graphics medieval village simulator.

This version intentionally uses only the Python standard library. It is designed to be easy to migrate later to NumPy, Numba, Mojo, Rust, Godot, Bevy, or another renderer/training system.

## What this MVP includes

- Procedural 2D grid terrain with height values, terrain kinds, water, and food.
- Simple hydrology: rain, downhill water flow, evaporation, and persistent water sources.
- Simple food ecology: terrain-biased food placement, depletion, and regrowth.
- One survival agent with thirst, hunger, fatigue, health, perception, and memory.
- Utility-style policy for drinking, eating, sleeping, exploring, and searching remembered locations.
- Resource memory with confidence and staleness.
- Headless deterministic simulation from seed and config.
- ASCII debug map.
- JSON run output with metrics, events, and optional snapshots.
- Unit tests using `unittest`.

## Run without installing

From the project root:

```bash
PYTHONPATH=src python -m village_sim --seed 1 --days 10 --width 32 --height 32 --print-map
```

Save a run report:

```bash
PYTHONPATH=src python -m village_sim --seed 1 --days 10 --width 32 --height 32 --replay runs/seed_1.json
```

Include periodic ASCII snapshots in the JSON report:

```bash
PYTHONPATH=src python -m village_sim --seed 1 --days 10 --width 32 --height 32 --snapshot-every 144 --replay runs/seed_1.json
```

Run several seeds:

```bash
PYTHONPATH=src python -m village_sim --batch 25 --days 10 --width 32 --height 32
```

## Install editable

```bash
python -m pip install -e .
village-sim --seed 1 --days 30 --print-map
```

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Migration principles

The simulation core does not depend on any renderer. The renderer consumes snapshots and has no authority over world truth.

World state uses portable scalar IDs, ints, floats, lists, dataclasses, and enums. This means individual subsystems can later be ported without changing the gameplay contract:

- `world.hydrology` can move to NumPy, Numba, Mojo, Rust, or C++.
- `agent.policy` can be replaced with evolutionary search or RL.
- `view.ascii_view` can be replaced with pygame-ce, Godot, Bevy, Unity, or a web viewer.
- `sim.replay` can move from JSON to MessagePack.

## MVP success condition

A successful MVP run is not guaranteed survival for every seed. Terrain can generate hostile starts. The target behavior is:

1. Agent gets thirsty.
2. Agent searches for water.
3. Agent drinks from visible water.
4. Agent remembers water.
5. Agent returns to remembered water later.
6. Agent gets hungry.
7. Agent finds food.
8. Agent eats and depletes local food.
9. Agent remembers food.
10. Agent returns to remembered food and searches nearby when stale or empty.
11. Agent sleeps at night unless urgent needs force movement.

## Code style

The code is intentionally explicit and data-oriented. It avoids engine object graphs, hidden global RNG calls, and renderer-owned simulation state.
