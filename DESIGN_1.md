# RL-Induced GOAP Design and Initial Discoverables

## 1. Objective

The simulation should support agents that begin with primitive survival drives and minimal world knowledge.

The system should not use hand-authored GOAP actions.

Instead:

1. A Pioneer agent explores using RL, search, or exploratory policies.
2. The orchestrator records successful trajectories.
3. The orchestrator induces symbolic GOAP actions from repeated successful behavior.
4. Those synthesized actions are stored as structured data.
5. GOAP-driven Townsfolk can import those actions and use them for planning.
6. The execution payload attached to each synthesized action invokes the learned policy, path routine, or interaction controller that originally produced the successful behavior.

The core idea:

```text
RL discovers behavior.
Trajectory analysis compresses behavior.
The orchestrator induces GOAP actions.
GOAP plans with induced actions.
Policies execute the selected action.
Agents transfer actions and world facts as knowledge.
```

---

## 2. Design Boundary

### Hand-authored

The following systems are manually defined:

```text
primitive simulation rules
terrain/resource mechanics
agent needs
primitive actions
observation predicates
reward/success criteria
trajectory recording
planner implementation
promotion thresholds
```

### Learned or induced

The following are not manually authored:

```text
GOAP action names
GOAP preconditions
GOAP effects
GOAP costs
GOAP confidence values
GOAP execution payload bindings
resource exploitation actions
travel/use routines
need-satisfaction routines
```

---

## 3. Runtime Agent Layers

The agent uses three conceptual layers.

```text
Layer 1: Primitive Controller
    move_north
    move_south
    move_east
    move_west
    inspect
    drink
    eat
    sleep
    wait

Layer 2: Learned Executors
    RL policy
    local search policy
    path routine
    interaction routine

Layer 3: GOAP Planner
    selects synthesized actions based on symbolic state,
    needs, action costs, confidence, and expected effects
```

GOAP chooses what to attempt.

The execution payload handles how the action is performed.

---

## 4. Pioneer and Townsfolk Roles

### Pioneer

The Pioneer is exploration-heavy.

Responsibilities:

```text
wander
discover resources
attempt interactions
learn useful policies
generate successful trajectories
fail often
produce evidence
```

The Pioneer is not expected to be efficient at first.

### Orchestrator

The orchestrator observes trajectories and converts repeated successful patterns into symbolic action models.

Responsibilities:

```text
record trajectories
detect successful need reductions
cluster similar trajectories
infer preconditions
infer effects
fit costs
estimate confidence
attach execution payloads
promote validated actions
export knowledge packets
```

### Townsfolk

Townsfolk are planner-heavy.

Responsibilities:

```text
import learned action models
import world facts
plan using GOAP
execute action payloads
update confidence from personal experience
share knowledge with others
```

---

## 5. Core Data Flow

```text
[ Simulation Tick ]
        |
        v
[ Raw Agent/World State ]
        |
        v
[ Predicate Extractor ]
        |
        v
[ Symbolic State Snapshot ]
        |
        v
[ Trajectory Recorder ]
        |
        v
[ Orchestrator ]
        |
        v
[ Synthesized GOAP Action ]
        |
        v
[ Action Library ]
        |
        v
[ GOAP Planner ]
        |
        v
[ Execution Payload ]
        |
        v
[ Learned Policy / Executor ]
```

---

## 6. Primitive Actions

The primitive action set should remain small.

```python
from enum import StrEnum


class PrimitiveAction(StrEnum):
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    INSPECT = "inspect"
    DRINK = "drink"
    EAT = "eat"
    SLEEP = "sleep"
    WAIT = "wait"
```

These are the only actions available to the lowest-level learner.

Everything above this level is induced.

---

## 7. Symbolic State

The orchestrator should not infer actions directly from raw floats.

Raw simulation state is converted into symbolic facts.

Example symbolic state:

```json
{
  "hunger_bucket": "high",
  "thirst_bucket": "low",
  "fatigue_bucket": "medium",
  "health_low": false,
  "is_daylight": true,

  "at_discoverable": true,
  "visible_discoverable": true,

  "target_id": "berry_bush_001",
  "target_type": "berry_bush",
  "target_has_resource": true,

  "known_water": true,
  "known_food": false
}
```

The predicate vocabulary is manually defined.

The GOAP actions are not.

---

## 8. Need Buckets

Needs are stored numerically in the simulation, but exposed symbolically for induction and planning.

```python
def bucket_need(value: float) -> str:
    if value >= 0.80:
        return "critical"
    if value >= 0.60:
        return "high"
    if value >= 0.30:
        return "medium"
    return "low"
```

Example:

```text
hunger = 0.83 -> hunger_bucket = critical
thirst = 0.48 -> thirst_bucket = medium
fatigue = 0.12 -> fatigue_bucket = low
```

---

## 9. Trajectory Recording

A trajectory records state before and after each primitive action.

```python
from dataclasses import dataclass, field
from typing import Any


FactValue = bool | int | float | str
SymbolicState = dict[str, FactValue]


@dataclass(slots=True)
class NeedState:
    hunger: float
    thirst: float
    fatigue: float
    health: float


@dataclass(slots=True)
class StateSnapshot:
    tick: int
    agent_id: int
    x: int
    y: int
    needs: NeedState
    symbolic: SymbolicState


@dataclass(slots=True)
class TrajectoryStep:
    before: StateSnapshot
    action: PrimitiveAction
    after: StateSnapshot
    reward: float
    events: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Trajectory:
    trajectory_id: str
    policy_id: str
    task_name: str
    steps: list[TrajectoryStep]

    @property
    def start(self) -> StateSnapshot:
        return self.steps[0].before

    @property
    def end(self) -> StateSnapshot:
        return self.steps[-1].after

    @property
    def cost_ticks(self) -> int:
        return self.end.tick - self.start.tick
```

---

## 10. Successful Trajectory Detection

A trajectory is considered useful when it significantly improves a need without killing the agent.

Example for hunger:

```python
from dataclasses import dataclass


@dataclass(slots=True)
class TaskResult:
    success: bool
    death: bool
    timeout: bool
    main_need_delta: float


def evaluate_hunger_task(trajectory: Trajectory) -> TaskResult:
    start_hunger: float = trajectory.start.needs.hunger
    end_hunger: float = trajectory.end.needs.hunger
    delta: float = end_hunger - start_hunger

    death: bool = trajectory.end.needs.health <= 0.0
    timeout: bool = trajectory.cost_ticks > 500
    success: bool = delta <= -0.20 and not death and not timeout

    return TaskResult(
        success=success,
        death=death,
        timeout=timeout,
        main_need_delta=delta,
    )
```

Equivalent evaluators should exist for:

```text
thirst
hunger
fatigue
```

---

## 11. Trajectory Clustering

The orchestrator should not create one action per successful episode.

It should cluster similar trajectories.

Example cluster keys:

```text
hunger:berry_bush:hunger
thirst:freshwater_spring:thirst
fatigue:shelter:fatigue
```

Cluster logic:

```python
class NeedName(StrEnum):
    HUNGER = "hunger"
    THIRST = "thirst"
    FATIGUE = "fatigue"


def cluster_key_for_trajectory(trajectory: Trajectory) -> str:
    start: SymbolicState = trajectory.start.symbolic

    target_type: str = str(start.get("target_type", "none"))
    task_name: str = trajectory.task_name

    changed_needs: list[str] = []

    for need in NeedName:
        before: float = getattr(trajectory.start.needs, need.value)
        after: float = getattr(trajectory.end.needs, need.value)

        if abs(after - before) >= 0.10:
            changed_needs.append(need.value)

    return f"{task_name}:{target_type}:{','.join(changed_needs)}"
```

---

## 12. Synthesized GOAP Action Format

A synthesized action is structured data.

It should be directly parseable by the planner.

```json
{
  "schema_version": 1,
  "action_id": "action_exploit_berry_bush_001_v1",
  "display_name": "Exploit berry bush 001",
  "scope": "instance",

  "preconditions": {
    "at_discoverable": true,
    "target_type": "berry_bush",
    "target_has_resource": true
  },

  "soft_preconditions": {
    "is_daylight": 0.12
  },

  "effects": {
    "hunger_delta": {
      "mean": -0.35,
      "p10": -0.22,
      "p90": -0.45,
      "confidence": 0.91
    }
  },

  "side_effects": {
    "fatigue_delta": {
      "mean": 0.04,
      "p10": 0.01,
      "p90": 0.08,
      "confidence": 0.84
    },
    "target_resource_delta": {
      "mean": -1.0,
      "p10": -1.0,
      "p90": -1.0,
      "confidence": 1.0
    }
  },

  "cost_model": {
    "base_ticks": 6,
    "distance_weight": 0.0,
    "fatigue_weight": 1.0,
    "night_multiplier": 1.25
  },

  "confidence": {
    "trials": 205,
    "successful_trials": 184,
    "failed_trials": 21,
    "success_rate": 0.897,
    "death_rate": 0.01,
    "timeout_rate": 0.04
  },

  "execution_payload": {
    "type": "RL_POLICY",
    "policy_id": "policy_exploit_berry_bush_v1",
    "policy_version": 1,
    "target_binding": {
      "mode": "resource_id",
      "resource_id": "berry_bush_001"
    }
  }
}
```

---

## 13. Instance Actions vs Template Actions

The system should distinguish between instance knowledge and template competence.

### Instance action

```text
Exploit this exact berry bush.
```

Example:

```text
action_exploit_berry_bush_001_v1
```

### Template action

```text
Exploit any berry bush of this type.
```

Example:

```text
action_exploit_berry_bush_template_v1
```

Instance actions encode local world knowledge.

Template actions encode reusable competence.

Both are useful.

---

## 14. Inferring Preconditions

A precondition is not simply every fact true at the start.

The orchestrator should infer facts that are:

```text
common in successful runs
uncommon in failed runs
relevant to the effect
```

Example:

```python
def fact_frequency(
    trajectories: list[Trajectory],
    fact: str,
    value: FactValue,
) -> float:
    if len(trajectories) == 0:
        return 0.0

    matches: int = 0

    for trajectory in trajectories:
        if trajectory.start.symbolic.get(fact) == value:
            matches += 1

    return matches / len(trajectories)


def candidate_start_facts(
    trajectories: list[Trajectory],
) -> dict[str, set[FactValue]]:
    values: dict[str, set[FactValue]] = {}

    for trajectory in trajectories:
        for key, value in trajectory.start.symbolic.items():
            values.setdefault(key, set()).add(value)

    return values


def infer_hard_preconditions(
    successful: list[Trajectory],
    failed: list[Trajectory],
    min_success_freq: float = 0.85,
    min_failure_gap: float = 0.25,
) -> dict[str, FactValue]:
    preconditions: dict[str, FactValue] = {}
    candidates: dict[str, set[FactValue]] = candidate_start_facts(successful)

    for fact, values in candidates.items():
        for value in values:
            success_freq: float = fact_frequency(successful, fact, value)
            failure_freq: float = fact_frequency(failed, fact, value)
            gap: float = success_freq - failure_freq

            if success_freq >= min_success_freq and gap >= min_failure_gap:
                preconditions[fact] = value

    return preconditions
```

Example result:

```json
{
  "at_discoverable": true,
  "target_type": "berry_bush",
  "target_has_resource": true
}
```

---

## 15. Inferring Effects

Effects are reliable state changes observed after successful execution.

Need deltas are numeric.

```python
def percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return 0.0

    ordered: list[float] = sorted(values)
    index: int = int((len(ordered) - 1) * q)
    return ordered[index]


@dataclass(slots=True)
class EffectEstimate:
    mean: float
    p10: float
    p90: float
    confidence: float


def infer_need_effect(
    successful: list[Trajectory],
    need: NeedName,
    min_abs_delta: float = 0.10,
) -> EffectEstimate | None:
    deltas: list[float] = []

    for trajectory in successful:
        before: float = getattr(trajectory.start.needs, need.value)
        after: float = getattr(trajectory.end.needs, need.value)
        delta: float = after - before

        if abs(delta) >= min_abs_delta:
            deltas.append(delta)

    if len(deltas) == 0:
        return None

    mean: float = sum(deltas) / len(deltas)
    confidence: float = len(deltas) / len(successful)

    return EffectEstimate(
        mean=mean,
        p10=percentile(deltas, 0.10),
        p90=percentile(deltas, 0.90),
        confidence=confidence,
    )
```

Example induced effect:

```json
{
  "hunger_delta": {
    "mean": -0.35,
    "p10": -0.22,
    "p90": -0.45,
    "confidence": 0.91
  }
}
```

---

## 16. Inferring Cost

The first version can use average elapsed ticks.

```python
def average_cost(trajectories: list[Trajectory]) -> float:
    if len(trajectories) == 0:
        return 0.0

    total: int = sum(trajectory.cost_ticks for trajectory in trajectories)
    return total / len(trajectories)
```

Initial cost model:

```json
{
  "base_ticks": 6,
  "distance_weight": 0.0,
  "fatigue_weight": 1.0,
  "night_multiplier": 1.25
}
```

Later, replace this with a regression model:

```text
cost =
    base_ticks
    + distance_weight * distance
    + fatigue_weight * fatigue
    + slope_weight * slope
    + night_penalty
```

---

## 17. Promotion Criteria

An induced action should not become trusted immediately.

Initial promotion rule:

```python
def is_promotable(confidence: ActionConfidence) -> bool:
    return (
        confidence.trials >= 100
        and confidence.success_rate >= 0.85
        and confidence.death_rate <= 0.02
        and confidence.timeout_rate <= 0.10
    )
```

Action lifecycle:

```text
candidate
validated
trusted
deprecated
```

Planner behavior:

```text
candidate actions may be used by pioneers
validated actions may be used by risk-tolerant agents
trusted actions may be used by ordinary townsfolk
deprecated actions are ignored unless no alternative exists
```

---

## 18. GOAP Planner Behavior

The planner consumes only synthesized actions.

It does not know whether a human, RL policy, or other process produced them.

Planner inputs:

```text
current symbolic state
goal symbolic state
synthesized action library
```

Example goal:

```json
{
  "hunger_bucket": "low"
}
```

Planner checks:

```text
preconditions
expected effects
cost
confidence
failure risk
```

Simplified scoring:

```text
expected_action_cost =
    base_ticks
    + failure_penalty * (1.0 - success_rate)
    + side_effect_penalty
```

---

## 19. Execution Payload

The action itself does not contain code.

It contains a payload reference.

```json
{
  "type": "RL_POLICY",
  "policy_id": "policy_exploit_berry_bush_v1",
  "policy_version": 1,
  "target_binding": {
    "mode": "resource_id",
    "resource_id": "berry_bush_001"
  }
}
```

The executor resolves this into runtime behavior.

Possible executor types:

```text
RL_POLICY
PATHFINDER
COMPOSITE
SCRIPTED_PRIMITIVE
```

`SCRIPTED_PRIMITIVE` should only be used for base sim verbs such as `eat`, `drink`, or `sleep`.

Synthesized macro-actions should normally use `RL_POLICY`, `PATHFINDER`, or `COMPOSITE`.

---

## 20. Knowledge Transfer

Knowledge should be split into two packet types.

### World fact packet

```json
{
  "knowledge_type": "world_fact",
  "fact_type": "resource_location",
  "source_agent_id": "pioneer_001",
  "confidence": 0.76,
  "data": {
    "resource_id": "berry_bush_001",
    "resource_type": "berry_bush",
    "coordinates": [20, 18]
  }
}
```

### Action knowledge packet

```json
{
  "knowledge_type": "action_model",
  "source_agent_id": "pioneer_001",
  "confidence": 0.82,
  "action_id": "action_exploit_berry_bush_template_v1",
  "policy_id": "policy_exploit_berry_bush_v1"
}
```

This allows one agent to teach:

```text
where something is
what something is
how to exploit it
how reliable the action is
```

These should be independently transferable.

---

## 21. Imported Knowledge Confidence

Imported knowledge should be degraded by trust.

```text
imported_confidence =
    source_action_confidence
    * transfer_confidence
    * trust_in_source
```

Example:

```text
Bob action confidence: 0.90
Alice trust in Bob:    0.70
Transfer quality:      0.80

Alice imported confidence:
    0.90 * 0.70 * 0.80 = 0.504
```

Alice may use the action, but she should update confidence after trying it herself.

---

# Initial Discoverables

## 22. Discoverable System

A discoverable is a stable world object that can be:

```text
perceived
identified
remembered
interacted with
depleted or refreshed
used as a target binding for synthesized actions
shared as knowledge
```

Discoverables should be stored separately from terrain.

They are not just terrain cells.

They are world entities with IDs.

---

## 23. Discoverable Data Model

```python
from dataclasses import dataclass
from enum import StrEnum


class DiscoverableKind(StrEnum):
    FRESHWATER_SPRING = "freshwater_spring"
    BERRY_BUSH = "berry_bush"


@dataclass(slots=True)
class Discoverable:
    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int

    visible_name: str
    discovered: bool

    amount: float
    max_amount: float
    regrowth_per_day: float

    satisfies_need: str
    need_delta: float
    interaction_ticks: int
```

---

## 24. Initial Discoverable 1: Freshwater Spring

Purpose:

```text
test thirst reduction
test discovery
test memory
test target binding
test reliable infinite resource behavior
test synthesized thirst action
```

Definition:

```python
FRESHWATER_SPRING_001 = Discoverable(
    discoverable_id="spring_001",
    kind=DiscoverableKind.FRESHWATER_SPRING,
    x=12,
    y=12,
    visible_name="freshwater spring",
    discovered=False,

    amount=9999.0,
    max_amount=9999.0,
    regrowth_per_day=0.0,

    satisfies_need="thirst",
    need_delta=-0.65,
    interaction_ticks=3,
)
```

Expected induced instance action:

```json
{
  "action_id": "action_exploit_freshwater_spring_001_v1",
  "display_name": "Exploit freshwater spring 001",
  "scope": "instance",

  "preconditions": {
    "at_discoverable": true,
    "target_type": "freshwater_spring"
  },

  "effects": {
    "thirst_delta": {
      "mean": -0.65,
      "p10": -0.65,
      "p90": -0.65,
      "confidence": 0.99
    }
  },

  "cost_model": {
    "base_ticks": 3,
    "distance_weight": 0.0,
    "fatigue_weight": 1.0,
    "night_multiplier": 1.0
  },

  "execution_payload": {
    "type": "RL_POLICY",
    "policy_id": "policy_exploit_freshwater_spring_v1",
    "policy_version": 1,
    "target_binding": {
      "mode": "resource_id",
      "resource_id": "spring_001"
    }
  }
}
```

Expected induced template action:

```json
{
  "action_id": "action_exploit_freshwater_spring_template_v1",
  "display_name": "Exploit freshwater spring",
  "scope": "template",

  "preconditions": {
    "at_discoverable": true,
    "target_type": "freshwater_spring"
  },

  "effects": {
    "thirst_delta": {
      "mean": -0.65,
      "confidence": 0.99
    }
  },

  "execution_payload": {
    "type": "RL_POLICY",
    "policy_id": "policy_exploit_freshwater_spring_v1",
    "policy_version": 1,
    "target_binding": {
      "mode": "current_target",
      "required_type": "freshwater_spring"
    }
  }
}
```

---

## 25. Initial Discoverable 2: Berry Bush

Purpose:

```text
test hunger reduction
test depletion
test regrowth
test stale memory
test failure after depletion
test synthesized hunger action
```

Definition:

```python
BERRY_BUSH_001 = Discoverable(
    discoverable_id="berry_bush_001",
    kind=DiscoverableKind.BERRY_BUSH,
    x=20,
    y=18,
    visible_name="berry bush",
    discovered=False,

    amount=4.0,
    max_amount=4.0,
    regrowth_per_day=0.5,

    satisfies_need="hunger",
    need_delta=-0.35,
    interaction_ticks=6,
)
```

Expected induced instance action:

```json
{
  "action_id": "action_exploit_berry_bush_001_v1",
  "display_name": "Exploit berry bush 001",
  "scope": "instance",

  "preconditions": {
    "at_discoverable": true,
    "target_type": "berry_bush",
    "target_has_resource": true
  },

  "effects": {
    "hunger_delta": {
      "mean": -0.35,
      "p10": -0.22,
      "p90": -0.45,
      "confidence": 0.91
    }
  },

  "side_effects": {
    "target_resource_delta": {
      "mean": -1.0,
      "p10": -1.0,
      "p90": -1.0,
      "confidence": 1.0
    },
    "fatigue_delta": {
      "mean": 0.04,
      "p10": 0.01,
      "p90": 0.08,
      "confidence": 0.84
    }
  },

  "failure_modes": {
    "target_depleted": {
      "probability": 0.18,
      "effects": {
        "hunger_delta": 0.0,
        "memory_confidence_delta": -0.25
      }
    }
  },

  "cost_model": {
    "base_ticks": 6,
    "distance_weight": 0.0,
    "fatigue_weight": 1.0,
    "night_multiplier": 1.25
  },

  "execution_payload": {
    "type": "RL_POLICY",
    "policy_id": "policy_exploit_berry_bush_v1",
    "policy_version": 1,
    "target_binding": {
      "mode": "resource_id",
      "resource_id": "berry_bush_001"
    }
  }
}
```

Expected induced template action:

```json
{
  "action_id": "action_exploit_berry_bush_template_v1",
  "display_name": "Exploit berry bush",
  "scope": "template",

  "preconditions": {
    "at_discoverable": true,
    "target_type": "berry_bush",
    "target_has_resource": true
  },

  "effects": {
    "hunger_delta": {
      "mean": -0.35,
      "confidence": 0.91
    },
    "target_resource_delta": {
      "mean": -1.0,
      "confidence": 1.0
    }
  },

  "execution_payload": {
    "type": "RL_POLICY",
    "policy_id": "policy_exploit_berry_bush_v1",
    "policy_version": 1,
    "target_binding": {
      "mode": "current_target",
      "required_type": "berry_bush"
    }
  }
}
```

---

## 26. Discoverable Perception

Agents perceive discoverables within a vision radius.

```python
@dataclass(slots=True)
class DiscoverableObservation:
    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int
    amount: float


@dataclass(slots=True)
class Observation:
    discoverables: list[DiscoverableObservation]


def nearby_discoverables(
    world: World,
    x: int,
    y: int,
    radius: int,
) -> list[Discoverable]:
    found: list[Discoverable] = []

    for item in world.discoverables.values():
        dx: int = item.x - x
        dy: int = item.y - y

        if dx * dx + dy * dy <= radius * radius:
            found.append(item)

    return found


def perceive_discoverables(
    world: World,
    agent_x: int,
    agent_y: int,
    vision_radius: int,
) -> Observation:
    observations: list[DiscoverableObservation] = []

    for item in nearby_discoverables(world, agent_x, agent_y, vision_radius):
        observations.append(
            DiscoverableObservation(
                discoverable_id=item.discoverable_id,
                kind=item.kind,
                x=item.x,
                y=item.y,
                amount=item.amount,
            )
        )

    return Observation(discoverables=observations)
```

---

## 27. Discoverable Memory

Agents store discovered resources in memory.

```python
@dataclass(slots=True)
class DiscoverableMemory:
    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int
    last_seen_tick: int
    last_known_amount: float
    confidence: float


@dataclass(slots=True)
class AgentMemory:
    discoverables: dict[str, DiscoverableMemory]


def update_discoverable_memory(
    memory: AgentMemory,
    observation: Observation,
    tick: int,
) -> None:
    for item in observation.discoverables:
        memory.discoverables[item.discoverable_id] = DiscoverableMemory(
            discoverable_id=item.discoverable_id,
            kind=item.kind,
            x=item.x,
            y=item.y,
            last_seen_tick=tick,
            last_known_amount=item.amount,
            confidence=1.0,
        )
```

---

## 28. Discoverable Interaction

```python
@dataclass(slots=True)
class AgentNeeds:
    hunger: float
    thirst: float
    fatigue: float
    health: float


def clamp_need(value: float) -> float:
    return max(0.0, min(1.0, value))


def exploit_discoverable(
    agent_needs: AgentNeeds,
    item: Discoverable,
) -> bool:
    if item.amount <= 0.0:
        return False

    if item.satisfies_need == "thirst":
        agent_needs.thirst = clamp_need(agent_needs.thirst + item.need_delta)

    elif item.satisfies_need == "hunger":
        agent_needs.hunger = clamp_need(agent_needs.hunger + item.need_delta)

    else:
        return False

    if item.max_amount < 9999.0:
        item.amount = max(0.0, item.amount - 1.0)

    return True
```

---

## 29. Discoverable Regrowth

```python
def update_discoverables_daily(world: World) -> None:
    for item in world.discoverables.values():
        if item.regrowth_per_day <= 0.0:
            continue

        item.amount = min(
            item.max_amount,
            item.amount + item.regrowth_per_day,
        )
```

Expected behavior:

```text
freshwater spring does not deplete
berry bush depletes after use
berry bush slowly regrows
stale berry-bush memories can fail
failed memory lowers confidence
```

---

## 30. Test Fixture World

```python
@dataclass(slots=True)
class World:
    width: int
    height: int
    discoverables: dict[str, Discoverable]


def make_discoverable_test_world() -> World:
    return World(
        width=64,
        height=64,
        discoverables={
            "spring_001": Discoverable(
                discoverable_id="spring_001",
                kind=DiscoverableKind.FRESHWATER_SPRING,
                x=12,
                y=12,
                visible_name="freshwater spring",
                discovered=False,
                amount=9999.0,
                max_amount=9999.0,
                regrowth_per_day=0.0,
                satisfies_need="thirst",
                need_delta=-0.65,
                interaction_ticks=3,
            ),
            "berry_bush_001": Discoverable(
                discoverable_id="berry_bush_001",
                kind=DiscoverableKind.BERRY_BUSH,
                x=20,
                y=18,
                visible_name="berry bush",
                discovered=False,
                amount=4.0,
                max_amount=4.0,
                regrowth_per_day=0.5,
                satisfies_need="hunger",
                need_delta=-0.35,
                interaction_ticks=6,
            ),
        },
    )
```

---

## 31. Initial Tests

### Discovery test

```python
def test_agent_perceives_nearby_spring() -> None:
    world: World = make_discoverable_test_world()

    observation: Observation = perceive_discoverables(
        world=world,
        agent_x=10,
        agent_y=12,
        vision_radius=4,
    )

    ids: set[str] = {
        item.discoverable_id
        for item in observation.discoverables
    }

    assert "spring_001" in ids
```

### Memory test

```python
def test_agent_remembers_seen_discoverable() -> None:
    world: World = make_discoverable_test_world()
    memory: AgentMemory = AgentMemory(discoverables={})

    observation: Observation = perceive_discoverables(
        world=world,
        agent_x=10,
        agent_y=12,
        vision_radius=4,
    )

    update_discoverable_memory(
        memory=memory,
        observation=observation,
        tick=100,
    )

    assert "spring_001" in memory.discoverables
    assert (
        memory.discoverables["spring_001"].kind
        == DiscoverableKind.FRESHWATER_SPRING
    )
```

### Spring exploitation test

```python
def test_agent_drinks_from_spring() -> None:
    world: World = make_discoverable_test_world()
    spring: Discoverable = world.discoverables["spring_001"]

    needs: AgentNeeds = AgentNeeds(
        hunger=0.1,
        thirst=0.9,
        fatigue=0.1,
        health=1.0,
    )

    success: bool = exploit_discoverable(needs, spring)

    assert success
    assert needs.thirst < 0.9
```

### Berry depletion test

```python
def test_berry_bush_depletes() -> None:
    world: World = make_discoverable_test_world()
    bush: Discoverable = world.discoverables["berry_bush_001"]

    needs: AgentNeeds = AgentNeeds(
        hunger=0.9,
        thirst=0.1,
        fatigue=0.1,
        health=1.0,
    )

    for _ in range(4):
        assert exploit_discoverable(needs, bush)

    assert bush.amount == 0.0

    success: bool = exploit_discoverable(needs, bush)

    assert not success
```

### Berry regrowth test

```python
def test_berry_bush_regrows() -> None:
    world: World = make_discoverable_test_world()
    bush: Discoverable = world.discoverables["berry_bush_001"]
    bush.amount = 0.0

    update_discoverables_daily(world)
    update_discoverables_daily(world)

    assert bush.amount == 1.0
```

---

## 32. Initial MVP Target

The first complete test loop should prove this sequence:

```text
Pioneer explores.
Pioneer discovers spring_001.
Pioneer drinks from spring_001.
Orchestrator records successful thirst reduction.
Orchestrator induces exploit_freshwater_spring action.
Townsperson imports spring world fact and spring action.
Townsperson gets thirsty.
Townsperson GOAP planner selects travel-to-spring and exploit-spring.
Townsperson executes learned payload and drinks.

Pioneer discovers berry_bush_001.
Pioneer eats berries.
Berry bush depletes.
Orchestrator records successful hunger reduction.
Orchestrator induces exploit_berry_bush action.
Townsperson imports berry bush fact and berry action.
Townsperson gets hungry.
Townsperson GOAP planner selects travel-to-bush and exploit-bush.
If bush has food, hunger drops.
If bush is depleted, action fails and memory confidence is reduced.
```

This validates:

```text
discoverables
perception
memory
interaction
depletion
regrowth
trajectory recording
action synthesis
GOAP planning
execution payloads
knowledge transfer
confidence updates
```

---

## 33. Near-Term Implementation Order

Implement in this order:

```text
1. Discoverable entity model
2. Deterministic test world with spring_001 and berry_bush_001
3. Perception hook
4. Discoverable memory
5. Interaction logic
6. Regrowth/depletion
7. Symbolic state extraction
8. Trajectory recording around interaction events
9. Hunger/thirst task evaluators
10. Action synthesis for exploit_spring and exploit_berry_bush
11. Action library storage
12. Simple GOAP planner over synthesized actions
13. Knowledge packet import/export
14. Confidence update after action execution
```

---

## 34. Design Constraint

The system should never generate Python code at runtime.

It should generate data.

```text
Generated:
    JSON action models
    JSON world facts
    policy references
    confidence statistics

Not generated:
    Python source code
    runtime classes
    executable scripts
```

This keeps the simulation loop fast and keeps learned behavior inspectable.
