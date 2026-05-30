# Village Sim: Settlement & Civilization Engine

A headless, deterministic, high-throughput survival simulation designed for emergent multi-agent civilizations.

This repository has been violently refactored from a single-agent object-oriented MVP into a fully vectorized, array-backed simulation engine capable of running hundreds to tens of thousands of concurrent agents. We do not simulate individual coins or memory nodes as Python objects. We model the world as dense tensors, query memory via columnar Polars dataframes, and distribute spatial chunks across Ray Actors.

## The Paradigm Shift: What We Did

The original MVP proved the viability of our GOAP (Goal-Oriented Action Planning) action synthesis, hydrology, and survival mechanics using a single `AgentState` dataclass. But scaling to a 400-agent settlement using Python `for` loops and list comprehensions is a failure condition. The interpreter chokes on object overhead and L3 cache misses.

We executed a four-phase architectural overhaul to achieve native C-speed execution:

1. **Eradicated Object-Oriented Hot Loops:** The `AgentState` dataclass was stripped from the `step()` function. All entities are now represented as parallel 1D `numpy` arrays (`AgentArrays`). Needs decay, movement, and state transitions are executed via boolean masking and vectorized broadcasting.
2. **Global Polars Memory Bank:** Agents no longer hold individual `list[ResourceMemory]`. All 200,000+ potential resource sightings are managed by a single `GlobalMemory` Polars DataFrame. Querying the best water source or decaying memory confidence is now a C-optimized columnar operation.
3. **Dense Economic Ledger:** We do not spawn "trade contracts." The economy is a dense `numpy` matrix. Transactions are resolved atomically via advanced indexing (`np.add.at`), tracking wealth and debts instantly without iterating over agents.
4. **Vectorized Proximity & Ray Distribution:** Finding who can teach whom or who can trade with whom is solved via broadcasted Chebyshev distance matrices. For scaling beyond a single town, the engine partitions into spatial `RegionActor` instances using Ray, passing entities across borders using zero-copy `msgpack` serialization.

## What This Architecture Does For Us

* **Throughput:** We can simulate thousands of days for a 400-agent settlement in a fraction of the time it took to run 10 agents previously.
* **Curriculum Learning:** The vectorized grid allows us to run 10,000 headless simulation instances in parallel. We train a "Founder" agent via RL to execute complex 50-step mutations (chopping wood, building shelter, digging wells).
* **Cultural Transmission at Scale:** When the Founder succeeds, its `ActionKnowledgePacket` is serialized via `msgpack`. Every 90 days, new settlers are injected into the array. They learn from the Founder via our vectorized social proximity graph, propagating complex action sequences (farming, building) through the settlement without requiring every agent to learn from scratch.
* **Emergent Economies:** Because trade evaluates instantly across the entire grid via matrix math, supply and demand behaviors emerge organically based on spatial resource distribution and local agent needs.

## General Simulation Mechanics

The core environment remains data-oriented, headless, and strictly deterministic from a given seed.

### The World Grid

The terrain is a continuous 2D map backed by `np.float32` and `np.int32` arrays.

* **Hydrology:** Rain falls, flows downhill along elevation gradients, evaporates, and feeds persistent water tables.
* **Ecology:** Terrain-biased food placement, depletion from foraging, and deterministic regrowth.
* **Mutation:** Agents can actively modify the `structure_kind` and `structure_hp` arrays to build walls, plant crops, or dig wells.

### Survival & Needs

Agents must balance four critical needs: Thirst, Hunger, Fatigue, and Cold Stress.

* If any critical need reaches `1.0`, the agent dies.
* Weather is fully simulated: rain and night cycles drastically increase cold stress unless the agent is near a heat source or underneath a constructed/natural shelter.

### GOAP and Action Synthesis

The simulation uses a dual-layer decision architecture:

* **Reactive Utility:** Agents execute baseline behaviors (drink, eat, sleep) based on local needs and Polars-backed memory queries.
* **Symbolic GOAP:** When urgent needs arise, agents query a Goal-Oriented Action Planner to chain actions (e.g., `MoveToTarget -> Exploit`).
* **Trajectory Recording:** When an agent successfully interacts with the environment, its state-action trajectory is recorded, evaluated by the Orchestrator, and synthesized into reusable, serialized `msgpack` action libraries.

## System Requirements & Execution

This engine strictly enforces Python 3.10+ typing (`mypy --strict`).

**Core Dependencies:**

* `numpy` (dense arrays, environmental simulation)
* `polars` (memory modeling, relational queries)
* `msgpack` (binary serialization, IPC handoffs, action libraries)
* `ray` (distributed spatial actors, optional for N > 500)

**Run the Vectorized Simulation:**

```bash
# Run a single headless settlement (400 agents) for 365 days
PYTHONPATH=src python -m village_sim --seed 42 --days 365 --agents 400

# Serialize the Founder's knowledge base
PYTHONPATH=src python -m village_sim --train-founder --action-library-out runs/founder_knowledge.msgpack

# Run a settlement with settler injection using the Founder's knowledge
PYTHONPATH=src python -m village_sim --seed 42 --days 365 --action-library-in runs/founder_knowledge.msgpack --spawn-interval 90
```

*Note: Rendering remains decoupled. The simulation engine yields MessagePack snapshots. Viewers (ASCII, wxPython, or external Godot/Bevy clients) consume these snapshots passively without stalling the simulation hot loop.*
