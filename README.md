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

## wxPython interface

Install GUI dependencies:

```bash
python -m pip install -e ".[gui]"
```

Launch the GUI:

```bash
PYTHONPATH=src python -m village_sim --wx
```

or

```bash
village-sim-gui
```

In the GUI, enable **Update map every tick** to animate the map and set **Tick delay (ms)** to control simulation speed.

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

## Discoverables and synthesized actions

The live simulation can optionally seed three canonical discoverables with
`--discoverables`:

```bash
PYTHONPATH=src python -m village_sim --seed 1 --days 2 --width 32 --height 32 --discoverables --print-map
```

- `spring_001` is a freshwater spring at `(12, 12)`. Exploiting it reduces
  thirst and does not deplete the spring.
- `berry_bush_001` is a berry bush at `(20, 18)`. Exploiting it reduces hunger
  and consumes one unit from the bush until daily regrowth replenishes it.
- `cave_001` is a cave at `(8, 24)`. Exploiting it reduces cold stress
  without depleting the cave.
- Discoverable sightings are stored in a separate ID-indexed discoverable memory.
- Successful or failed discoverable exploitation is recorded as a trajectory and
  fed to the orchestrator. Focused known-target travel segments can also be
  recorded when a pathfinder run reaches a remembered discoverable.
- Repeated successful trajectories synthesize in-memory instance and template
  actions, including both exploit actions and known-target travel actions.


### Reading a cold/shelter run

Use a short discoverable GOAP run with replay snapshots to see the existing
cold-weather and cave-shelter signals:

```bash
PYTHONPATH=src python -m village_sim --seed 1 --days 3 --discoverables --goap --snapshot-every 12 --replay /tmp/village_replay.json
```

Representative events and summary lines include:

```text
weather: cold night
status: agent is cold
action: seeking shelter at cave_001
action: sheltered at cave_001
Cold/weather: temp_c=...
```

Replay snapshots include world `temperature_c`, `feels_cold`, `cold_reason`,
and per-agent `is_sheltered` and `cold_status` fields.

## GOAP chaining status

Enable live GOAP control with `--goap` after seeding discoverables:

```bash
PYTHONPATH=src python -m village_sim --seed 1 --days 2 --width 32 --height 32 --discoverables --goap --print-map
```

Action libraries can be persisted and reused across runs:

```bash
PYTHONPATH=src python -m village_sim --seed 1 --days 2 --width 32 --height 32 --discoverables --goap --action-library-out /tmp/village_actions.json
PYTHONPATH=src python -m village_sim --seed 2 --days 2 --width 32 --height 32 --discoverables --goap --action-library-in /tmp/village_actions.json
```

Current supported chain:

```text
MoveToKnownDiscoverable -> ExploitDiscoverable
```

Current executor types:

- `PATHFINDER`: deterministic greedy movement toward a bound discoverable target.
- `SCRIPTED_PRIMITIVE`: currently used for discoverable exploitation.
- `RL_POLICY`: payload references remain serializable action metadata; neural RL
  policy execution is not implemented yet.

Current limitations:

- No neural RL or model-free RL training yet.
- The pathfinder is greedy and deterministic, not full A*.
- The planner is a bounded forward search, not a full production planner.
- Travel actions are still induced from controlled/repeated trajectories.
- Social teaching is not implemented yet.
- The existing utility survival policy remains the fallback when GOAP has no
  applicable plan or a plan fails.
