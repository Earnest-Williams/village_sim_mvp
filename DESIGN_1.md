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

---

# Implementation Plan


This document is a step-by-step implementation plan for every section of `DESIGN_1.md`.

Each step references the design section(s) it satisfies, states the target file(s), lists
the exact types/functions to add, and provides ready-to-use code samples.

---

## Orientation: What Already Exists

The following relevant code is already present and must **not** be duplicated or broken.

| Existing artefact | File | Notes |
|---|---|---|
| `ActionKind` enum | `core/types.py` | Low-level action labels; the new `PrimitiveAction` StrEnum lives alongside it |
| `AgentState` | `agent/state.py` | Contains `thirst`, `hunger`, `fatigue`, `health` as raw floats |
| `AgentMemory` / `ResourceMemory` | `agent/memory.py` | Position-indexed; the new `DiscoverableMemory` is ID-indexed, kept separate |
| `Observation` | `agent/perception.py` | Currently lists visible terrain water/food; needs a `discoverables` field added |
| `World` | `world/world.py` | Flat array terrain; needs a `discoverables` dict added |
| `Simulation` engine | `sim/engine.py` | Single-agent headless loop |
| Tests | `tests/` | `test_memory.py`, `test_simulation.py`, `test_world.py` |

The implementation order below follows §33 of the design document exactly.

---

## Step 1 — `PrimitiveAction` StrEnum

**Design section:** §6  
**Target file:** `src/village_sim/core/types.py`

Add the primitive action vocabulary alongside the existing `ActionKind` enum.  
`ActionKind` describes internal engine labels; `PrimitiveAction` is the RL vocabulary.

```python
# src/village_sim/core/types.py  (add after existing imports)

from enum import StrEnum          # Python 3.11+; or inherit from str, Enum

class PrimitiveAction(StrEnum):
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST  = "move_east"
    MOVE_WEST  = "move_west"
    INSPECT    = "inspect"
    DRINK      = "drink"
    EAT        = "eat"
    SLEEP      = "sleep"
    WAIT       = "wait"
```

**Acceptance check:** `from village_sim.core.types import PrimitiveAction; PrimitiveAction.EAT == "eat"` is `True`.

---

## Step 2 — `Discoverable` Entity Model

**Design sections:** §22, §23  
**New file:** `src/village_sim/world/discoverables.py`

Discoverables are world entities with stable IDs.  
They are **not** terrain cells and live in a separate collection.

```python
# src/village_sim/world/discoverables.py

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DiscoverableKind(StrEnum):
    FRESHWATER_SPRING = "freshwater_spring"
    BERRY_BUSH        = "berry_bush"


@dataclass(slots=True)
class Discoverable:
    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int

    visible_name: str
    discovered: bool          # True once any agent has perceived it

    amount: float             # current resource stock
    max_amount: float
    regrowth_per_day: float   # added to amount each simulated day

    satisfies_need: str       # "hunger" | "thirst"
    need_delta: float         # negative means need decreases (good)
    interaction_ticks: int    # ticks consumed by one exploitation
```

---

## Step 3 — Deterministic Test World

**Design sections:** §24, §25, §30  
**Target file:** `src/village_sim/world/discoverables.py` (append to same file)

Define the two seed discoverables and a factory for the test world.

```python
# src/village_sim/world/discoverables.py  (continued)

from dataclasses import dataclass as _dataclass   # already imported above


# ── Canonical instances (§24, §25) ────────────────────────────────────────────

def make_spring_001() -> Discoverable:
    return Discoverable(
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


def make_berry_bush_001() -> Discoverable:
    return Discoverable(
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


# ── Minimal world fixture for unit tests (§30) ────────────────────────────────

@_dataclass(slots=True)
class DiscoverableWorld:
    """Minimal test-only world that contains only discoverables (no terrain arrays)."""

    width: int
    height: int
    discoverables: dict[str, Discoverable]


def make_discoverable_test_world() -> DiscoverableWorld:
    return DiscoverableWorld(
        width=64,
        height=64,
        discoverables={
            "spring_001": make_spring_001(),
            "berry_bush_001": make_berry_bush_001(),
        },
    )
```

**Note:** The production `World` in `world/world.py` will gain a `discoverables` field in Step 6.
Until then, tests import `DiscoverableWorld` directly.

---

## Step 4 — Discoverable Perception

**Design section:** §26  
**Target file:** `src/village_sim/world/discoverables.py` (append)

```python
# src/village_sim/world/discoverables.py  (continued)

from dataclasses import dataclass as _dc2, field as _field
from typing import Protocol


class HasDiscoverables(Protocol):
    """Structural type: any world-like object with a discoverables dict."""
    discoverables: dict[str, Discoverable]


@_dc2(slots=True)
class DiscoverableObservation:
    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int
    amount: float


def nearby_discoverables(
    world: HasDiscoverables,
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
    world: HasDiscoverables,
    agent_x: int,
    agent_y: int,
    vision_radius: int,
) -> list[DiscoverableObservation]:
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
    return observations
```

**Integration:** In `agent/perception.py`, import `perceive_discoverables` and add a
`discoverables: list[DiscoverableObservation]` field to the existing `Observation` dataclass.
Call `perceive_discoverables` inside the existing `perceive()` function and store the result.

```python
# agent/perception.py — diff excerpt

from village_sim.world.discoverables import DiscoverableObservation, perceive_discoverables

@dataclass(slots=True)
class Observation:
    visible_water: list[ResourceSighting] = field(default_factory=list)
    visible_food:  list[ResourceSighting] = field(default_factory=list)
    discoverables: list[DiscoverableObservation] = field(default_factory=list)  # NEW
    is_daylight: bool = True
    is_night:    bool = False

    def all_sightings(self) -> list[ResourceSighting]:
        return [*self.visible_water, *self.visible_food]

# Inside perceive():
    # ... existing body ...
    disc_obs = perceive_discoverables(world, position.x, position.y, radius)
    return Observation(
        visible_water=visible_water,
        visible_food=visible_food,
        discoverables=disc_obs,          # NEW
        is_daylight=clock.is_daylight,
        is_night=clock.is_night,
    )
```

---

## Step 5 — Discoverable Memory

**Design section:** §27  
**Target file:** `src/village_sim/agent/memory.py` (append)

Add `DiscoverableMemory` and `update_discoverable_memory` alongside the existing
`ResourceMemory` / `AgentMemory` without modifying them.

```python
# agent/memory.py  (append at end of file)

from village_sim.world.discoverables import DiscoverableKind, DiscoverableObservation


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
class DiscoverableAgentMemory:
    """Extends agent knowledge with ID-indexed discoverable memories."""

    discoverables: dict[str, DiscoverableMemory] = field(default_factory=dict)


def update_discoverable_memory(
    memory: DiscoverableAgentMemory,
    observations: list[DiscoverableObservation],
    tick: int,
) -> None:
    """Upsert discoverable memories from the current tick's observations.

    First sighting sets confidence=1.0; subsequent sightings refresh it.
    """
    for item in observations:
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

**Integration:** Add a `discoverable_memory: DiscoverableAgentMemory` field to `AgentState`
(or keep it as a parallel object inside `Simulation`).  The simplest approach for the MVP is
to instantiate `DiscoverableAgentMemory()` alongside `AgentMemory()` in `sim/engine.py` and
pass it through `_step_agent`.

---

## Step 6 — Discoverable Interaction

**Design sections:** §28  
**Target file:** `src/village_sim/world/discoverables.py` (append)

```python
# world/discoverables.py  (continued)

@_dc2(slots=True)
class AgentNeeds:
    hunger:  float
    thirst:  float
    fatigue: float
    health:  float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def exploit_discoverable(
    agent_needs: AgentNeeds,
    item: Discoverable,
) -> bool:
    """Apply one interaction with a discoverable to the agent's needs.

    Returns True on success, False when the resource is depleted or the
    satisfies_need field is unrecognised.
    """
    if item.amount <= 0.0:
        return False

    if item.satisfies_need == "thirst":
        agent_needs.thirst = _clamp(agent_needs.thirst + item.need_delta)
    elif item.satisfies_need == "hunger":
        agent_needs.hunger = _clamp(agent_needs.hunger + item.need_delta)
    else:
        return False

    # Infinite sources (spring) never deplete.
    if item.max_amount < 9999.0:
        item.amount = max(0.0, item.amount - 1.0)

    return True
```

---

## Step 7 — Discoverable Regrowth

**Design section:** §29  
**Target file:** `src/village_sim/world/discoverables.py` (append)

```python
# world/discoverables.py  (continued)

def update_discoverables_daily(world: HasDiscoverables) -> None:
    """Replenish depletable discoverables once per simulated day."""
    for item in world.discoverables.values():
        if item.regrowth_per_day <= 0.0:
            continue
        item.amount = min(item.max_amount, item.amount + item.regrowth_per_day)
```

**Integration:** Call `update_discoverables_daily(world)` inside `World.step_environment()`
once per simulated day.  The `SimClock` already tracks `tick_of_day`; call the function when
`clock.tick_of_day == 0` (start of a new day).

---

## Step 8 — Wire `discoverables` into Production `World`

**Target file:** `src/village_sim/world/world.py`

Add an optional `discoverables` field to the existing `World` dataclass so the engine can
carry real discoverable entities.

```python
# world/world.py  — additions only

from village_sim.world.discoverables import (
    Discoverable,
    HasDiscoverables,          # the Protocol
    update_discoverables_daily,
)

@dataclass(slots=True)
class World:
    # ... existing fields ...
    discoverables: dict[str, Discoverable] = field(default_factory=dict)  # NEW
```

Update `generate_world()` to accept an optional `discoverables` dict:

```python
def generate_world(
    config: SimConfig,
    rng: random.Random,
    discoverables: dict[str, Discoverable] | None = None,
) -> World:
    # ... existing body ...
    return World(
        width=config.width,
        height=config.height,
        height_map=height_map,
        terrain=terrain,
        water=water,
        food=food,
        food_capacity=food_capacity,
        discoverables=discoverables or {},   # NEW
    )
```

Call `update_discoverables_daily(self.world)` inside `Simulation.step()` at the tick where
`clock.tick_of_day == 0`.

---

## Step 9 — Symbolic State Extraction

**Design sections:** §7, §8  
**New file:** `src/village_sim/orchestrator/symbolic.py`

```python
# src/village_sim/orchestrator/symbolic.py

from __future__ import annotations

from village_sim.agent.state import AgentState
from village_sim.agent.memory import DiscoverableAgentMemory
from village_sim.agent.perception import Observation
from village_sim.core.time import SimClock
from village_sim.world.discoverables import DiscoverableKind

FactValue    = bool | int | float | str
SymbolicState = dict[str, FactValue]


# ── Need bucketing (§8) ───────────────────────────────────────────────────────

def bucket_need(value: float) -> str:
    if value >= 0.80:
        return "critical"
    if value >= 0.60:
        return "high"
    if value >= 0.30:
        return "medium"
    return "low"


# ── Full state predicate extractor (§7) ───────────────────────────────────────

def extract_symbolic_state(
    agent: AgentState,
    observation: Observation,
    disc_memory: DiscoverableAgentMemory,
    clock: SimClock,
) -> SymbolicState:
    """Convert raw simulation state into a flat symbolic fact dictionary."""

    state: SymbolicState = {
        "hunger_bucket":  bucket_need(agent.hunger),
        "thirst_bucket":  bucket_need(agent.thirst),
        "fatigue_bucket": bucket_need(agent.fatigue),
        "health_low":     agent.health < 0.30,
        "is_daylight":    clock.is_daylight,
    }

    # Closest discoverable in current observation (§7 example predicates)
    if observation.discoverables:
        closest = min(
            observation.discoverables,
            key=lambda d: (d.x - agent.position.x) ** 2
                          + (d.y - agent.position.y) ** 2,
        )
        at_disc = (
            abs(closest.x - agent.position.x) <= 1
            and abs(closest.y - agent.position.y) <= 1
        )
        state["at_discoverable"]      = at_disc
        state["visible_discoverable"] = True
        state["target_id"]            = closest.discoverable_id
        state["target_type"]          = str(closest.kind)
        state["target_has_resource"]  = closest.amount > 0.0
    else:
        state["at_discoverable"]      = False
        state["visible_discoverable"] = False
        state["target_id"]            = "none"
        state["target_type"]          = "none"
        state["target_has_resource"]  = False

    # Memory-derived predicates
    known_water = any(
        m.kind is DiscoverableKind.FRESHWATER_SPRING
        for m in disc_memory.discoverables.values()
    )
    known_food = any(
        m.kind is DiscoverableKind.BERRY_BUSH
        for m in disc_memory.discoverables.values()
    )
    state["known_water"] = known_water
    state["known_food"]  = known_food

    return state
```

---

## Step 10 — Trajectory Recording

**Design section:** §9  
**New file:** `src/village_sim/orchestrator/trajectory.py`

```python
# src/village_sim/orchestrator/trajectory.py

from __future__ import annotations

from dataclasses import dataclass, field

from village_sim.core.types import PrimitiveAction
from village_sim.orchestrator.symbolic import FactValue, SymbolicState


@dataclass(slots=True)
class NeedState:
    hunger:  float
    thirst:  float
    fatigue: float
    health:  float


@dataclass(slots=True)
class StateSnapshot:
    tick:       int
    agent_id:   int
    x:          int
    y:          int
    needs:      NeedState
    symbolic:   SymbolicState


@dataclass(slots=True)
class TrajectoryStep:
    before: StateSnapshot
    action: PrimitiveAction
    after:  StateSnapshot
    reward: float
    events: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Trajectory:
    trajectory_id: str
    policy_id:     str
    task_name:     str
    steps:         list[TrajectoryStep]

    @property
    def start(self) -> StateSnapshot:
        return self.steps[0].before

    @property
    def end(self) -> StateSnapshot:
        return self.steps[-1].after

    @property
    def cost_ticks(self) -> int:
        return self.end.tick - self.start.tick


# ── Recorder helper ──────────────────────────────────────────────────────────

class TrajectoryRecorder:
    """Accumulates steps into a Trajectory.

    Usage inside the simulation loop:
        recorder = TrajectoryRecorder(trajectory_id="...", policy_id="...", task_name="...")
        recorder.record(before_snapshot, action, after_snapshot, reward, events)
        trajectory = recorder.finish()
    """

    def __init__(self, trajectory_id: str, policy_id: str, task_name: str) -> None:
        self.trajectory_id = trajectory_id
        self.policy_id     = policy_id
        self.task_name     = task_name
        self._steps: list[TrajectoryStep] = []

    def record(
        self,
        before: StateSnapshot,
        action: PrimitiveAction,
        after: StateSnapshot,
        reward: float,
        events: list[str] | None = None,
    ) -> None:
        self._steps.append(
            TrajectoryStep(
                before=before,
                action=action,
                after=after,
                reward=reward,
                events=events or [],
            )
        )

    def finish(self) -> Trajectory:
        if not self._steps:
            raise ValueError("Cannot finish a trajectory with no steps.")
        return Trajectory(
            trajectory_id=self.trajectory_id,
            policy_id=self.policy_id,
            task_name=self.task_name,
            steps=list(self._steps),
        )
```

**Integration note:** The simulation engine creates a `TrajectoryRecorder` at the start of
each task episode.  A `StateSnapshot` is built from `AgentState` + `extract_symbolic_state()`
before and after every primitive action.

---

## Step 11 — Task Evaluators

**Design section:** §10  
**New file:** `src/village_sim/orchestrator/evaluator.py`

```python
# src/village_sim/orchestrator/evaluator.py

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from village_sim.orchestrator.trajectory import Trajectory


class NeedName(StrEnum):
    HUNGER  = "hunger"
    THIRST  = "thirst"
    FATIGUE = "fatigue"


@dataclass(slots=True)
class TaskResult:
    success:         bool
    death:           bool
    timeout:         bool
    main_need_delta: float


def _evaluate_need_task(
    trajectory: Trajectory,
    need: NeedName,
    improvement_threshold: float = -0.20,
    max_ticks: int = 500,
) -> TaskResult:
    start_value: float = getattr(trajectory.start.needs, need.value)
    end_value:   float = getattr(trajectory.end.needs,   need.value)
    delta:       float = end_value - start_value

    death:   bool = trajectory.end.needs.health <= 0.0
    timeout: bool = trajectory.cost_ticks > max_ticks
    success: bool = delta <= improvement_threshold and not death and not timeout

    return TaskResult(
        success=success,
        death=death,
        timeout=timeout,
        main_need_delta=delta,
    )


def evaluate_hunger_task(trajectory: Trajectory) -> TaskResult:
    return _evaluate_need_task(trajectory, NeedName.HUNGER)


def evaluate_thirst_task(trajectory: Trajectory) -> TaskResult:
    return _evaluate_need_task(trajectory, NeedName.THIRST)


def evaluate_fatigue_task(trajectory: Trajectory) -> TaskResult:
    return _evaluate_need_task(trajectory, NeedName.FATIGUE)
```

---

## Step 12 — Trajectory Clustering

**Design section:** §11  
**Target file:** `src/village_sim/orchestrator/evaluator.py` (append)

```python
# orchestrator/evaluator.py  (continued)

from village_sim.orchestrator.symbolic import SymbolicState


def cluster_key_for_trajectory(trajectory: Trajectory) -> str:
    """Produce a stable cluster key for grouping similar successful trajectories.

    Format: ``<task_name>:<target_type>:<comma-joined changed needs>``
    """
    start: SymbolicState = trajectory.start.symbolic
    target_type: str     = str(start.get("target_type", "none"))
    task_name: str       = trajectory.task_name

    changed_needs: list[str] = []
    for need in NeedName:
        before: float = getattr(trajectory.start.needs, need.value)
        after:  float = getattr(trajectory.end.needs,   need.value)
        if abs(after - before) >= 0.10:
            changed_needs.append(need.value)

    return f"{task_name}:{target_type}:{','.join(changed_needs)}"
```

---

## Step 13 — Precondition and Effect Inference

**Design sections:** §14, §15, §16  
**New file:** `src/village_sim/orchestrator/induction.py`

```python
# src/village_sim/orchestrator/induction.py

from __future__ import annotations

from dataclasses import dataclass

from village_sim.orchestrator.evaluator import NeedName
from village_sim.orchestrator.symbolic import FactValue, SymbolicState
from village_sim.orchestrator.trajectory import Trajectory


# ── Precondition inference (§14) ─────────────────────────────────────────────

def fact_frequency(
    trajectories: list[Trajectory],
    fact: str,
    value: FactValue,
) -> float:
    if not trajectories:
        return 0.0
    matches = sum(
        1 for t in trajectories
        if t.start.symbolic.get(fact) == value
    )
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
    min_failure_gap:  float = 0.25,
) -> dict[str, FactValue]:
    """Return facts that are nearly always true at the start of successful runs
    but rare (or much rarer) at the start of failed runs.
    """
    preconditions: dict[str, FactValue] = {}
    candidates = candidate_start_facts(successful)

    for fact, values in candidates.items():
        for value in values:
            success_freq = fact_frequency(successful, fact, value)
            failure_freq = fact_frequency(failed,     fact, value)
            gap          = success_freq - failure_freq
            if success_freq >= min_success_freq and gap >= min_failure_gap:
                preconditions[fact] = value

    return preconditions


def infer_soft_preconditions(
    successful: list[Trajectory],
    failed: list[Trajectory],
    min_success_freq: float = 0.50,
    max_success_freq: float = 0.85,
) -> dict[str, float]:
    """Return facts that help but are not essential; value = success frequency."""
    soft: dict[str, float] = {}
    candidates = candidate_start_facts(successful)

    for fact, values in candidates.items():
        for value in values:
            freq = fact_frequency(successful, fact, value)
            if min_success_freq <= freq < max_success_freq:
                soft[f"{fact}={value}"] = round(freq, 4)

    return soft


# ── Effect inference (§15) ───────────────────────────────────────────────────

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index   = int((len(ordered) - 1) * q)
    return ordered[index]


@dataclass(slots=True)
class EffectEstimate:
    mean:       float
    p10:        float
    p90:        float
    confidence: float


def infer_need_effect(
    successful: list[Trajectory],
    need: NeedName,
    min_abs_delta: float = 0.10,
) -> EffectEstimate | None:
    """Estimate the distribution of need change across successful trajectories."""
    deltas: list[float] = []

    for t in successful:
        before: float = getattr(t.start.needs, need.value)
        after:  float = getattr(t.end.needs,   need.value)
        delta:  float = after - before
        if abs(delta) >= min_abs_delta:
            deltas.append(delta)

    if not deltas:
        return None

    mean       = sum(deltas) / len(deltas)
    confidence = len(deltas) / len(successful)

    return EffectEstimate(
        mean=round(mean, 4),
        p10=round(_percentile(deltas, 0.10), 4),
        p90=round(_percentile(deltas, 0.90), 4),
        confidence=round(confidence, 4),
    )


# ── Cost inference (§16) ─────────────────────────────────────────────────────

def average_cost(trajectories: list[Trajectory]) -> float:
    if not trajectories:
        return 0.0
    total = sum(t.cost_ticks for t in trajectories)
    return total / len(trajectories)
```

---

## Step 14 — Synthesized GOAP Action Model and Action Library

**Design sections:** §12, §13, §17  
**New file:** `src/village_sim/orchestrator/action_model.py`

```python
# src/village_sim/orchestrator/action_model.py

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import StrEnum
from pathlib import Path

from village_sim.orchestrator.induction import EffectEstimate
from village_sim.orchestrator.symbolic import FactValue


# ── Supporting types ──────────────────────────────────────────────────────────

class ActionScope(StrEnum):
    INSTANCE = "instance"
    TEMPLATE = "template"


class ActionLifecycle(StrEnum):
    CANDIDATE  = "candidate"
    VALIDATED  = "validated"
    TRUSTED    = "trusted"
    DEPRECATED = "deprecated"


class ExecutorType(StrEnum):
    RL_POLICY        = "RL_POLICY"
    PATHFINDER       = "PATHFINDER"
    COMPOSITE        = "COMPOSITE"
    SCRIPTED_PRIMITIVE = "SCRIPTED_PRIMITIVE"


@dataclass(slots=True)
class TargetBinding:
    mode: str                          # "resource_id" | "current_target"
    resource_id:   str | None = None   # used when mode == "resource_id"
    required_type: str | None = None   # used when mode == "current_target"


@dataclass(slots=True)
class ExecutionPayload:
    type:            ExecutorType
    policy_id:       str
    policy_version:  int
    target_binding:  TargetBinding


@dataclass(slots=True)
class CostModel:
    base_ticks:       float
    distance_weight:  float = 0.0
    fatigue_weight:   float = 1.0
    night_multiplier: float = 1.0


@dataclass(slots=True)
class ActionConfidence:
    trials:           int
    successful_trials: int
    failed_trials:    int
    success_rate:     float
    death_rate:       float
    timeout_rate:     float


# ── Main synthesized action (§12) ─────────────────────────────────────────────

@dataclass
class SynthesizedAction:
    schema_version:   int
    action_id:        str
    display_name:     str
    scope:            ActionScope
    lifecycle:        ActionLifecycle

    preconditions:    dict[str, FactValue]
    soft_preconditions: dict[str, float]

    effects:          dict[str, EffectEstimate]    # primary effects
    side_effects:     dict[str, EffectEstimate]    # secondary effects

    cost_model:       CostModel
    confidence:       ActionConfidence
    execution_payload: ExecutionPayload

    def to_dict(self) -> dict:
        return asdict(self)         # type: ignore[arg-type]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: dict) -> SynthesizedAction:
        """Deserialise from a plain dict (e.g. loaded from JSON)."""
        return SynthesizedAction(
            schema_version=data["schema_version"],
            action_id=data["action_id"],
            display_name=data["display_name"],
            scope=ActionScope(data["scope"]),
            lifecycle=ActionLifecycle(data["lifecycle"]),
            preconditions=data["preconditions"],
            soft_preconditions=data.get("soft_preconditions", {}),
            effects={
                k: EffectEstimate(**v)
                for k, v in data.get("effects", {}).items()
            },
            side_effects={
                k: EffectEstimate(**v)
                for k, v in data.get("side_effects", {}).items()
            },
            cost_model=CostModel(**data["cost_model"]),
            confidence=ActionConfidence(**data["confidence"]),
            execution_payload=ExecutionPayload(
                type=ExecutorType(data["execution_payload"]["type"]),
                policy_id=data["execution_payload"]["policy_id"],
                policy_version=data["execution_payload"]["policy_version"],
                target_binding=TargetBinding(
                    **data["execution_payload"]["target_binding"]
                ),
            ),
        )


# ── Promotion logic (§17) ──────────────────────────────────────────────────────

def is_promotable(confidence: ActionConfidence) -> bool:
    """True when an action has accumulated enough evidence to become trusted."""
    return (
        confidence.trials >= 100
        and confidence.success_rate >= 0.85
        and confidence.death_rate   <= 0.02
        and confidence.timeout_rate <= 0.10
    )


def promote_action(action: SynthesizedAction) -> None:
    """Advance lifecycle: candidate → validated → trusted."""
    if action.lifecycle is ActionLifecycle.CANDIDATE:
        action.lifecycle = ActionLifecycle.VALIDATED
    elif action.lifecycle is ActionLifecycle.VALIDATED:
        if is_promotable(action.confidence):
            action.lifecycle = ActionLifecycle.TRUSTED


# ── Action library (§33 step 11) ─────────────────────────────────────────────

class ActionLibrary:
    """In-memory store for synthesized actions, with JSON persistence."""

    def __init__(self) -> None:
        self._actions: dict[str, SynthesizedAction] = {}

    def add(self, action: SynthesizedAction) -> None:
        self._actions[action.action_id] = action

    def get(self, action_id: str) -> SynthesizedAction | None:
        return self._actions.get(action_id)

    def all_actions(self) -> list[SynthesizedAction]:
        return list(self._actions.values())

    def trusted_actions(self) -> list[SynthesizedAction]:
        return [
            a for a in self._actions.values()
            if a.lifecycle is ActionLifecycle.TRUSTED
        ]

    def actions_for_lifecycle(
        self, lifecycle: ActionLifecycle
    ) -> list[SynthesizedAction]:
        return [
            a for a in self._actions.values()
            if a.lifecycle is lifecycle
        ]

    def save(self, path: Path) -> None:
        """Serialise all actions to a JSON file. Generates data, not code (§34)."""
        data = [a.to_dict() for a in self._actions.values()]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ActionLibrary:
        library = cls()
        for item in json.loads(path.read_text()):
            library.add(SynthesizedAction.from_dict(item))
        return library
```

---

## Step 15 — Orchestrator: Synthesis Pipeline

**Design sections:** §4 (Orchestrator responsibilities), §10–§17  
**New file:** `src/village_sim/orchestrator/orchestrator.py`

```python
# src/village_sim/orchestrator/orchestrator.py

from __future__ import annotations

from village_sim.orchestrator.action_model import (
    ActionConfidence,
    ActionLifecycle,
    ActionScope,
    CostModel,
    EffectEstimate,
    ExecutionPayload,
    ExecutorType,
    SynthesizedAction,
    TargetBinding,
    is_promotable,
    promote_action,
)
from village_sim.orchestrator.evaluator import (
    NeedName,
    TaskResult,
    cluster_key_for_trajectory,
    evaluate_hunger_task,
    evaluate_thirst_task,
    evaluate_fatigue_task,
)
from village_sim.orchestrator.induction import (
    average_cost,
    infer_hard_preconditions,
    infer_need_effect,
    infer_soft_preconditions,
)
from village_sim.orchestrator.trajectory import Trajectory


_EVALUATORS = {
    NeedName.HUNGER:  evaluate_hunger_task,
    NeedName.THIRST:  evaluate_thirst_task,
    NeedName.FATIGUE: evaluate_fatigue_task,
}


class Orchestrator:
    """Consumes recorded trajectories and produces synthesized GOAP actions."""

    def __init__(self, schema_version: int = 1) -> None:
        self._schema_version = schema_version
        self._clusters: dict[str, list[Trajectory]] = {}   # key → successful
        self._failed:   dict[str, list[Trajectory]] = {}   # key → failed

    def record(self, trajectory: Trajectory) -> None:
        """Classify and store one completed trajectory."""
        key = cluster_key_for_trajectory(trajectory)

        # Evaluate against all need dimensions; a trajectory may satisfy more than one.
        is_success = False
        for need, evaluator in _EVALUATORS.items():
            result: TaskResult = evaluator(trajectory)
            if result.success:
                is_success = True
                break

        if is_success:
            self._clusters.setdefault(key, []).append(trajectory)
        else:
            self._failed.setdefault(key, []).append(trajectory)

    def synthesize_all(self) -> list[SynthesizedAction]:
        """Attempt to synthesize one action per cluster that has enough data."""
        actions: list[SynthesizedAction] = []
        for key, successful in self._clusters.items():
            if len(successful) < 10:
                continue    # insufficient evidence
            failed = self._failed.get(key, [])
            action = self._synthesize_cluster(key, successful, failed)
            if action is not None:
                actions.append(action)
        return actions

    def _synthesize_cluster(
        self,
        cluster_key: str,
        successful: list[Trajectory],
        failed: list[Trajectory],
    ) -> SynthesizedAction | None:
        # Derive action_id from cluster key, e.g. "hunger:berry_bush:hunger"
        parts = cluster_key.split(":")
        task_name   = parts[0] if len(parts) > 0 else "unknown"
        target_type = parts[1] if len(parts) > 1 else "none"

        # Representative trajectory for payload extraction
        rep = successful[0]
        target_id = str(rep.start.symbolic.get("target_id", "unknown"))

        action_id  = f"action_exploit_{target_id}_v1"
        policy_id  = f"policy_exploit_{target_type}_v1"

        preconditions = infer_hard_preconditions(successful, failed)
        if not preconditions:
            return None    # cannot build a meaningful action without any preconditions

        soft_preconditions = infer_soft_preconditions(successful, failed)

        # Infer effects for each need
        effects: dict[str, EffectEstimate] = {}
        for need in NeedName:
            estimate = infer_need_effect(successful, need)
            if estimate is not None:
                effects[f"{need.value}_delta"] = estimate

        if not effects:
            return None

        base_ticks = average_cost(successful)
        total      = len(successful) + len(failed)
        succ_rate  = len(successful) / total if total else 0.0
        death_ct   = sum(1 for t in failed if t.end.needs.health <= 0.0)
        timeout_ct = sum(1 for t in failed if t.cost_ticks > 500)

        # Determine scope: instance vs template (§13)
        scope = (
            ActionScope.INSTANCE
            if target_id not in ("none", "unknown")
            else ActionScope.TEMPLATE
        )
        target_binding = (
            TargetBinding(mode="resource_id", resource_id=target_id)
            if scope is ActionScope.INSTANCE
            else TargetBinding(mode="current_target", required_type=target_type)
        )

        action = SynthesizedAction(
            schema_version=self._schema_version,
            action_id=action_id,
            display_name=f"Exploit {target_id.replace('_', ' ')}",
            scope=scope,
            lifecycle=ActionLifecycle.CANDIDATE,
            preconditions=preconditions,
            soft_preconditions=soft_preconditions,
            effects=effects,
            side_effects={},
            cost_model=CostModel(
                base_ticks=round(base_ticks, 1),
                distance_weight=0.0,
                fatigue_weight=1.0,
                night_multiplier=1.25,
            ),
            confidence=ActionConfidence(
                trials=total,
                successful_trials=len(successful),
                failed_trials=len(failed),
                success_rate=round(succ_rate, 4),
                death_rate=round(death_ct / total, 4) if total else 0.0,
                timeout_rate=round(timeout_ct / total, 4) if total else 0.0,
            ),
            execution_payload=ExecutionPayload(
                type=ExecutorType.RL_POLICY,
                policy_id=policy_id,
                policy_version=1,
                target_binding=target_binding,
            ),
        )

        if is_promotable(action.confidence):
            promote_action(action)
            promote_action(action)   # candidate → validated → trusted

        return action
```

---

## Step 16 — Simple GOAP Planner

**Design section:** §18  
**New file:** `src/village_sim/goap/planner.py`

```python
# src/village_sim/goap/planner.py

from __future__ import annotations

from dataclasses import dataclass

from village_sim.orchestrator.action_model import (
    ActionLifecycle,
    SynthesizedAction,
)
from village_sim.orchestrator.symbolic import FactValue, SymbolicState


@dataclass
class PlanStep:
    action: SynthesizedAction
    expected_cost: float


def _action_applicable(
    action: SynthesizedAction,
    state: SymbolicState,
) -> bool:
    """Check hard preconditions against the current symbolic state."""
    for fact, required_value in action.preconditions.items():
        if state.get(fact) != required_value:
            return False
    return True


def _expected_cost(action: SynthesizedAction, failure_penalty: float = 50.0) -> float:
    """Simplified scoring formula from §18.

    expected_cost =
        base_ticks
        + failure_penalty * (1.0 - success_rate)
        + fatigue_weight * base_ticks * 0.1   (side-effect penalty proxy)
    """
    base = action.cost_model.base_ticks
    failure_term = failure_penalty * (1.0 - action.confidence.success_rate)
    side_effect_penalty = action.cost_model.fatigue_weight * base * 0.10
    return base + failure_term + side_effect_penalty


def _action_advances_goal(
    action: SynthesizedAction,
    goal: SymbolicState,
) -> bool:
    """True if at least one effect key-value pair moves toward the goal."""
    # For need-bucket goals, check if a negative delta is present for the relevant need.
    for goal_fact, goal_value in goal.items():
        # e.g. goal = {"hunger_bucket": "low"}, effect key = "hunger_delta" with negative mean
        need_prefix = goal_fact.replace("_bucket", "")
        delta_key   = f"{need_prefix}_delta"
        if delta_key in action.effects:
            estimate = action.effects[delta_key]
            if goal_value in ("low", "medium") and estimate.mean < 0.0:
                return True
    return False


def plan(
    state: SymbolicState,
    goal: SymbolicState,
    library: list[SynthesizedAction],
    agent_lifecycle_floor: ActionLifecycle = ActionLifecycle.TRUSTED,
) -> list[PlanStep]:
    """Return a prioritised (lowest cost first) list of applicable plan steps.

    For the MVP this is a flat best-action selector, not a full tree search.
    A full A* GOAP tree search can replace this without changing the interface.
    """
    lifecycle_order = [
        ActionLifecycle.TRUSTED,
        ActionLifecycle.VALIDATED,
        ActionLifecycle.CANDIDATE,
        ActionLifecycle.DEPRECATED,
    ]
    min_rank = lifecycle_order.index(agent_lifecycle_floor)

    candidates: list[PlanStep] = []
    for action in library:
        action_rank = lifecycle_order.index(action.lifecycle)
        if action_rank < min_rank:
            continue
        if not _action_applicable(action, state):
            continue
        if not _action_advances_goal(action, goal):
            continue
        candidates.append(PlanStep(action=action, expected_cost=_expected_cost(action)))

    candidates.sort(key=lambda ps: ps.expected_cost)
    return candidates
```

---

## Step 17 — Execution Payload Dispatcher

**Design section:** §19  
**New file:** `src/village_sim/goap/executor.py`

```python
# src/village_sim/goap/executor.py

from __future__ import annotations

from typing import Protocol

from village_sim.orchestrator.action_model import ExecutionPayload, ExecutorType


class PolicyRunner(Protocol):
    """Minimal interface a concrete policy must satisfy."""

    def run(self, target_id: str | None) -> str:
        """Execute one step of the policy; return a log message."""
        ...


class ExecutorRegistry:
    """Maps policy_id strings to live PolicyRunner objects.

    Any RL policy, pathfinder, or scripted primitive registers itself here.
    The GOAP planner only needs to call dispatch(); it never touches the policy
    implementation directly.
    """

    def __init__(self) -> None:
        self._runners: dict[str, PolicyRunner] = {}

    def register(self, policy_id: str, runner: PolicyRunner) -> None:
        self._runners[policy_id] = runner

    def dispatch(self, payload: ExecutionPayload) -> str:
        """Invoke the executor referenced by payload and return a log message."""
        if payload.type is ExecutorType.SCRIPTED_PRIMITIVE:
            # Primitives are handled inline by the engine; this branch should be
            # reached only when a synthesized action wraps one for bookkeeping.
            return f"primitive:{payload.policy_id}"

        runner = self._runners.get(payload.policy_id)
        if runner is None:
            return f"no runner registered for policy_id={payload.policy_id}"

        target_id = payload.target_binding.resource_id
        return runner.run(target_id)
```

---

## Step 18 — Knowledge Transfer

**Design sections:** §20, §21  
**New file:** `src/village_sim/goap/knowledge.py`

```python
# src/village_sim/goap/knowledge.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


# ── Packet types (§20) ────────────────────────────────────────────────────────

@dataclass
class WorldFactPacket:
    knowledge_type:  str          # always "world_fact"
    fact_type:       str          # e.g. "resource_location"
    source_agent_id: str
    confidence:      float
    data:            dict         # resource_id, resource_type, coordinates

    def to_dict(self) -> dict:
        return asdict(self)       # type: ignore[arg-type]


@dataclass
class ActionKnowledgePacket:
    knowledge_type:  str          # always "action_model"
    source_agent_id: str
    confidence:      float
    action_id:       str
    policy_id:       str

    def to_dict(self) -> dict:
        return asdict(self)       # type: ignore[arg-type]


KnowledgePacket = WorldFactPacket | ActionKnowledgePacket


# ── Confidence degradation on import (§21) ────────────────────────────────────

def imported_confidence(
    source_action_confidence: float,
    trust_in_source:          float,
    transfer_quality:         float = 1.0,
) -> float:
    """Degrade imported confidence by source trust and transfer quality.

    imported = source_confidence * transfer_quality * trust_in_source
    """
    return round(
        source_action_confidence * transfer_quality * trust_in_source,
        4,
    )


# ── Serialisation helpers ─────────────────────────────────────────────────────

def save_packets(packets: list[KnowledgePacket], path: Path) -> None:
    data = [p.to_dict() for p in packets]
    path.write_text(json.dumps(data, indent=2))


def load_packets(path: Path) -> list[dict]:
    return json.loads(path.read_text())
```

---

## Step 19 — Confidence Update After Execution

**Design section:** §21  
**Target file:** `src/village_sim/orchestrator/action_model.py` (append)

```python
# orchestrator/action_model.py  (append)

def update_confidence_after_execution(
    action: SynthesizedAction,
    success: bool,
    death: bool,
    timeout: bool,
) -> None:
    """Incrementally update a SynthesizedAction's confidence stats.

    Called each time a Townsfolk agent executes the action and reports the outcome.
    Automatically promotes the action if the updated stats meet the threshold.
    """
    c = action.confidence
    c.trials += 1
    if success:
        c.successful_trials += 1
    else:
        c.failed_trials += 1
        if death:
            # Recalculate death rate from accumulated data
            pass    # death_rate is a derived stat recalculated below
        if timeout:
            pass

    c.success_rate  = round(c.successful_trials / c.trials, 4)
    # Approximate death and timeout rates from totals
    # (exact values require storing per-outcome counts; this is close enough for MVP)
    c.death_rate    = round(
        max(0.0, c.death_rate * ((c.trials - 1) / c.trials)
            + (1.0 / c.trials if death else 0.0)),
        4,
    )
    c.timeout_rate  = round(
        max(0.0, c.timeout_rate * ((c.trials - 1) / c.trials)
            + (1.0 / c.trials if timeout else 0.0)),
        4,
    )

    promote_action(action)    # re-evaluate lifecycle after each update
```

---

## Step 20 — Unit Tests

**Design section:** §31  
**New file:** `tests/test_discoverables.py`

These tests are the acceptance criteria for Steps 2–7.

```python
# tests/test_discoverables.py

from __future__ import annotations

import unittest

from village_sim.world.discoverables import (
    AgentNeeds,
    DiscoverableAgentMemory,       # imported from agent/memory.py after Step 5
    DiscoverableKind,
    DiscoverableWorld,
    exploit_discoverable,
    make_discoverable_test_world,
    perceive_discoverables,
    update_discoverables_daily,
    update_discoverable_memory,
)


class TestDiscoverablePerception(unittest.TestCase):
    def test_agent_perceives_nearby_spring(self) -> None:
        world = make_discoverable_test_world()
        observations = perceive_discoverables(world, agent_x=10, agent_y=12, vision_radius=4)
        ids = {obs.discoverable_id for obs in observations}
        self.assertIn("spring_001", ids)

    def test_agent_does_not_perceive_distant_spring(self) -> None:
        world = make_discoverable_test_world()
        observations = perceive_discoverables(world, agent_x=0, agent_y=0, vision_radius=3)
        ids = {obs.discoverable_id for obs in observations}
        self.assertNotIn("spring_001", ids)


class TestDiscoverableMemory(unittest.TestCase):
    def test_agent_remembers_seen_discoverable(self) -> None:
        world  = make_discoverable_test_world()
        memory = DiscoverableAgentMemory()
        obs    = perceive_discoverables(world, agent_x=10, agent_y=12, vision_radius=4)
        update_discoverable_memory(memory, obs, tick=100)

        self.assertIn("spring_001", memory.discoverables)
        self.assertEqual(
            memory.discoverables["spring_001"].kind,
            DiscoverableKind.FRESHWATER_SPRING,
        )
        self.assertEqual(memory.discoverables["spring_001"].confidence, 1.0)

    def test_memory_updates_on_revisit(self) -> None:
        world  = make_discoverable_test_world()
        memory = DiscoverableAgentMemory()
        obs    = perceive_discoverables(world, agent_x=10, agent_y=12, vision_radius=4)
        update_discoverable_memory(memory, obs, tick=100)
        update_discoverable_memory(memory, obs, tick=200)

        self.assertEqual(memory.discoverables["spring_001"].last_seen_tick, 200)


class TestSpringExploitation(unittest.TestCase):
    def test_agent_drinks_from_spring(self) -> None:
        world  = make_discoverable_test_world()
        spring = world.discoverables["spring_001"]
        needs  = AgentNeeds(hunger=0.1, thirst=0.9, fatigue=0.1, health=1.0)

        success = exploit_discoverable(needs, spring)

        self.assertTrue(success)
        self.assertLess(needs.thirst, 0.9)

    def test_spring_does_not_deplete(self) -> None:
        world  = make_discoverable_test_world()
        spring = world.discoverables["spring_001"]
        needs  = AgentNeeds(hunger=0.1, thirst=0.9, fatigue=0.1, health=1.0)

        for _ in range(50):
            exploit_discoverable(needs, spring)

        self.assertEqual(spring.amount, 9999.0)


class TestBerryBushDepletion(unittest.TestCase):
    def test_berry_bush_depletes(self) -> None:
        world = make_discoverable_test_world()
        bush  = world.discoverables["berry_bush_001"]
        needs = AgentNeeds(hunger=0.9, thirst=0.1, fatigue=0.1, health=1.0)

        for _ in range(4):
            self.assertTrue(exploit_discoverable(needs, bush))

        self.assertEqual(bush.amount, 0.0)
        self.assertFalse(exploit_discoverable(needs, bush))

    def test_berry_bush_reduces_hunger(self) -> None:
        world  = make_discoverable_test_world()
        bush   = world.discoverables["berry_bush_001"]
        needs  = AgentNeeds(hunger=0.9, thirst=0.1, fatigue=0.1, health=1.0)
        before = needs.hunger

        exploit_discoverable(needs, bush)

        self.assertLess(needs.hunger, before)


class TestBerryBushRegrowth(unittest.TestCase):
    def test_berry_bush_regrows(self) -> None:
        world = make_discoverable_test_world()
        bush  = world.discoverables["berry_bush_001"]
        bush.amount = 0.0

        update_discoverables_daily(world)
        update_discoverables_daily(world)

        self.assertAlmostEqual(bush.amount, 1.0)

    def test_berry_bush_does_not_exceed_max(self) -> None:
        world = make_discoverable_test_world()
        bush  = world.discoverables["berry_bush_001"]

        for _ in range(100):
            update_discoverables_daily(world)

        self.assertEqual(bush.amount, bush.max_amount)


if __name__ == "__main__":
    unittest.main()
```

---

## Step 21 — Orchestrator Tests

**New file:** `tests/test_orchestrator.py`

```python
# tests/test_orchestrator.py

from __future__ import annotations

import unittest

from village_sim.orchestrator.evaluator import (
    NeedName,
    TaskResult,
    cluster_key_for_trajectory,
    evaluate_thirst_task,
    evaluate_hunger_task,
)
from village_sim.orchestrator.induction import (
    EffectEstimate,
    average_cost,
    fact_frequency,
    infer_hard_preconditions,
    infer_need_effect,
)
from village_sim.orchestrator.trajectory import (
    NeedState,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
)
from village_sim.core.types import PrimitiveAction


def _make_snapshot(
    tick: int,
    hunger: float,
    thirst: float,
    at_discoverable: bool = True,
    target_type: str = "freshwater_spring",
    health: float = 1.0,
) -> StateSnapshot:
    return StateSnapshot(
        tick=tick,
        agent_id=1,
        x=10,
        y=12,
        needs=NeedState(hunger=hunger, thirst=thirst, fatigue=0.1, health=health),
        symbolic={
            "at_discoverable": at_discoverable,
            "target_type": target_type,
            "target_has_resource": True,
            "target_id": f"{target_type}_001",
        },
    )


def _make_trajectory(
    hunger_start: float,
    thirst_start: float,
    hunger_end:   float,
    thirst_end:   float,
    ticks:        int = 10,
    target_type:  str = "freshwater_spring",
    task_name:    str = "thirst",
    health_end:   float = 1.0,
) -> Trajectory:
    before = _make_snapshot(0, hunger_start, thirst_start, target_type=target_type)
    after  = _make_snapshot(ticks, hunger_end,  thirst_end,  target_type=target_type,
                             health=health_end)
    return Trajectory(
        trajectory_id="traj_001",
        policy_id="policy_spring_v1",
        task_name=task_name,
        steps=[
            TrajectoryStep(
                before=before,
                action=PrimitiveAction.DRINK,
                after=after,
                reward=1.0,
            )
        ],
    )


class TestTaskEvaluators(unittest.TestCase):
    def test_thirst_success(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.15)
        result = evaluate_thirst_task(traj)
        self.assertTrue(result.success)
        self.assertFalse(result.death)

    def test_thirst_failure_small_delta(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.75)  # only -0.05 delta
        result = evaluate_thirst_task(traj)
        self.assertFalse(result.success)

    def test_thirst_failure_death(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.15, health_end=0.0)
        result = evaluate_thirst_task(traj)
        self.assertFalse(result.success)
        self.assertTrue(result.death)

    def test_hunger_success(self) -> None:
        traj = _make_trajectory(0.8, 0.1, 0.4, 0.1, target_type="berry_bush", task_name="hunger")
        result = evaluate_hunger_task(traj)
        self.assertTrue(result.success)


class TestClustering(unittest.TestCase):
    def test_cluster_key(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.15)
        key  = cluster_key_for_trajectory(traj)
        # Should contain task_name:target_type:thirst
        self.assertIn("freshwater_spring", key)
        self.assertIn("thirst", key)


class TestPreconditionInference(unittest.TestCase):
    def test_infer_hard_preconditions(self) -> None:
        successful = [
            _make_trajectory(0.1, 0.8, 0.1, 0.15) for _ in range(10)
        ]
        failed: list[Trajectory] = []
        prec = infer_hard_preconditions(successful, failed)
        self.assertIn("at_discoverable", prec)
        self.assertEqual(prec["at_discoverable"], True)

    def test_fact_frequency(self) -> None:
        trajs = [_make_trajectory(0.1, 0.8, 0.1, 0.15) for _ in range(4)]
        freq  = fact_frequency(trajs, "at_discoverable", True)
        self.assertEqual(freq, 1.0)


class TestEffectInference(unittest.TestCase):
    def test_infer_thirst_effect(self) -> None:
        trajs = [_make_trajectory(0.1, 0.8, 0.1, 0.15) for _ in range(5)]
        est   = infer_need_effect(trajs, NeedName.THIRST)
        self.assertIsNotNone(est)
        assert est is not None
        self.assertLess(est.mean, 0.0)
        self.assertGreater(est.confidence, 0.0)

    def test_average_cost(self) -> None:
        trajs = [_make_trajectory(0.1, 0.8, 0.1, 0.15, ticks=10) for _ in range(3)]
        cost  = average_cost(trajs)
        self.assertAlmostEqual(cost, 10.0)


if __name__ == "__main__":
    unittest.main()
```

---

## Step 22 — GOAP Planner Tests

**New file:** `tests/test_goap_planner.py`

```python
# tests/test_goap_planner.py

from __future__ import annotations

import unittest

from village_sim.orchestrator.action_model import (
    ActionConfidence,
    ActionLifecycle,
    ActionScope,
    CostModel,
    EffectEstimate,
    ExecutionPayload,
    ExecutorType,
    SynthesizedAction,
    TargetBinding,
)
from village_sim.goap.planner import plan


def _make_action(
    action_id: str,
    need: str = "thirst",
    lifecycle: ActionLifecycle = ActionLifecycle.TRUSTED,
    success_rate: float = 0.95,
) -> SynthesizedAction:
    return SynthesizedAction(
        schema_version=1,
        action_id=action_id,
        display_name=action_id,
        scope=ActionScope.INSTANCE,
        lifecycle=lifecycle,
        preconditions={"at_discoverable": True, "target_type": "freshwater_spring"},
        soft_preconditions={},
        effects={f"{need}_delta": EffectEstimate(mean=-0.65, p10=-0.65, p90=-0.65, confidence=0.99)},
        side_effects={},
        cost_model=CostModel(base_ticks=3, fatigue_weight=1.0),
        confidence=ActionConfidence(
            trials=200,
            successful_trials=int(200 * success_rate),
            failed_trials=int(200 * (1 - success_rate)),
            success_rate=success_rate,
            death_rate=0.0,
            timeout_rate=0.0,
        ),
        execution_payload=ExecutionPayload(
            type=ExecutorType.RL_POLICY,
            policy_id="policy_spring_v1",
            policy_version=1,
            target_binding=TargetBinding(mode="resource_id", resource_id="spring_001"),
        ),
    )


class TestGOAPPlanner(unittest.TestCase):
    def test_selects_applicable_action(self) -> None:
        state   = {"at_discoverable": True, "target_type": "freshwater_spring",
                   "thirst_bucket": "high"}
        goal    = {"thirst_bucket": "low"}
        library = [_make_action("action_spring_v1")]

        plan_steps = plan(state, goal, library)
        self.assertEqual(len(plan_steps), 1)
        self.assertEqual(plan_steps[0].action.action_id, "action_spring_v1")

    def test_rejects_non_applicable_action(self) -> None:
        # at_discoverable is False; precondition requires True
        state   = {"at_discoverable": False, "target_type": "freshwater_spring",
                   "thirst_bucket": "high"}
        goal    = {"thirst_bucket": "low"}
        library = [_make_action("action_spring_v1")]

        plan_steps = plan(state, goal, library)
        self.assertEqual(len(plan_steps), 0)

    def test_lower_cost_action_is_first(self) -> None:
        cheap     = _make_action("cheap_action", success_rate=0.99)
        expensive = _make_action("expensive_action", success_rate=0.50)
        state = {"at_discoverable": True, "target_type": "freshwater_spring",
                 "thirst_bucket": "high"}
        goal  = {"thirst_bucket": "low"}

        plan_steps = plan(state, goal, [expensive, cheap])
        self.assertEqual(plan_steps[0].action.action_id, "cheap_action")

    def test_deprecated_actions_excluded_for_trusted_agents(self) -> None:
        action  = _make_action("old_action", lifecycle=ActionLifecycle.DEPRECATED)
        state   = {"at_discoverable": True, "target_type": "freshwater_spring",
                   "thirst_bucket": "high"}
        goal    = {"thirst_bucket": "low"}
        plan_steps = plan(state, goal, [action], agent_lifecycle_floor=ActionLifecycle.TRUSTED)
        self.assertEqual(len(plan_steps), 0)


if __name__ == "__main__":
    unittest.main()
```

---

## Step 23 — Knowledge Transfer Tests

**New file:** `tests/test_knowledge.py`

```python
# tests/test_knowledge.py

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from village_sim.goap.knowledge import (
    ActionKnowledgePacket,
    WorldFactPacket,
    imported_confidence,
    load_packets,
    save_packets,
)


class TestImportedConfidence(unittest.TestCase):
    def test_confidence_formula(self) -> None:
        result = imported_confidence(
            source_action_confidence=0.90,
            trust_in_source=0.70,
            transfer_quality=0.80,
        )
        self.assertAlmostEqual(result, 0.504, places=3)

    def test_full_trust_preserves_confidence(self) -> None:
        result = imported_confidence(0.90, 1.0, 1.0)
        self.assertAlmostEqual(result, 0.90)


class TestKnowledgePacketSerialisation(unittest.TestCase):
    def test_world_fact_round_trip(self) -> None:
        packet = WorldFactPacket(
            knowledge_type="world_fact",
            fact_type="resource_location",
            source_agent_id="pioneer_001",
            confidence=0.76,
            data={"resource_id": "spring_001", "resource_type": "freshwater_spring",
                  "coordinates": [12, 12]},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_packets([packet], path)
        loaded = load_packets(path)
        self.assertEqual(loaded[0]["knowledge_type"], "world_fact")
        self.assertEqual(loaded[0]["data"]["resource_id"], "spring_001")

    def test_action_knowledge_packet(self) -> None:
        packet = ActionKnowledgePacket(
            knowledge_type="action_model",
            source_agent_id="pioneer_001",
            confidence=0.82,
            action_id="action_exploit_freshwater_spring_template_v1",
            policy_id="policy_exploit_freshwater_spring_v1",
        )
        data = packet.to_dict()
        self.assertEqual(data["knowledge_type"], "action_model")
        self.assertEqual(data["confidence"], 0.82)


if __name__ == "__main__":
    unittest.main()
```

---

## Step 24 — MVP Integration Test (Full Loop)

**Design section:** §32  
**New file:** `tests/test_mvp_loop.py`

```python
# tests/test_mvp_loop.py

"""End-to-end smoke test: Pioneer → Orchestrator → Townsfolk GOAP plan.

This test does not run the full simulation engine; it wires each subsystem
together manually to confirm the complete data flow from §32 works.
"""

from __future__ import annotations

import unittest

from village_sim.core.types import PrimitiveAction
from village_sim.goap.knowledge import ActionKnowledgePacket, WorldFactPacket, imported_confidence
from village_sim.goap.planner import plan
from village_sim.orchestrator.action_model import (
    ActionLibrary,
    ActionLifecycle,
)
from village_sim.orchestrator.evaluator import cluster_key_for_trajectory, evaluate_thirst_task
from village_sim.orchestrator.orchestrator import Orchestrator
from village_sim.orchestrator.symbolic import bucket_need, extract_symbolic_state
from village_sim.orchestrator.trajectory import NeedState, StateSnapshot, Trajectory, TrajectoryStep
from village_sim.world.discoverables import (
    AgentNeeds,
    DiscoverableAgentMemory,
    exploit_discoverable,
    make_discoverable_test_world,
    perceive_discoverables,
    update_discoverable_memory,
)


def _snapshot(tick: int, thirst: float, health: float = 1.0) -> StateSnapshot:
    return StateSnapshot(
        tick=tick,
        agent_id=1,
        x=11,
        y=12,
        needs=NeedState(hunger=0.1, thirst=thirst, fatigue=0.1, health=health),
        symbolic={
            "at_discoverable": True,
            "target_type": "freshwater_spring",
            "target_id": "spring_001",
            "target_has_resource": True,
            "thirst_bucket": bucket_need(thirst),
        },
    )


class TestMVPLoop(unittest.TestCase):
    def test_pioneer_to_townsfolk_spring_loop(self) -> None:
        # ── 1. Pioneer perceives and exploits spring ────────────────────────
        world  = make_discoverable_test_world()
        spring = world.discoverables["spring_001"]
        memory = DiscoverableAgentMemory()

        obs = perceive_discoverables(world, agent_x=10, agent_y=12, vision_radius=4)
        self.assertIn("spring_001", {o.discoverable_id for o in obs})

        update_discoverable_memory(memory, obs, tick=5)
        self.assertIn("spring_001", memory.discoverables)

        needs = AgentNeeds(hunger=0.1, thirst=0.9, fatigue=0.1, health=1.0)
        success = exploit_discoverable(needs, spring)
        self.assertTrue(success)
        self.assertLess(needs.thirst, 0.9)

        # ── 2. Orchestrator records trajectories and induces action ─────────
        orchestrator = Orchestrator()
        for _ in range(15):
            traj = Trajectory(
                trajectory_id=f"t_{_}",
                policy_id="policy_spring_v1",
                task_name="thirst",
                steps=[
                    TrajectoryStep(
                        before=_snapshot(0,  thirst=0.85),
                        action=PrimitiveAction.DRINK,
                        after =_snapshot(3,  thirst=0.20),
                        reward=1.0,
                    )
                ],
            )
            orchestrator.record(traj)

        actions = orchestrator.synthesize_all()
        self.assertGreater(len(actions), 0, "Orchestrator should produce at least one action")

        library = ActionLibrary()
        for action in actions:
            library.add(action)

        # ── 3. Townsfolk imports knowledge ──────────────────────────────────
        source_conf   = actions[0].confidence.success_rate
        townsfolk_conf = imported_confidence(source_conf, trust_in_source=0.80)
        self.assertGreater(townsfolk_conf, 0.0)

        # ── 4. Townsfolk GOAP planner selects spring action ─────────────────
        townsfolk_state = {
            "at_discoverable": True,
            "target_type": "freshwater_spring",
            "thirst_bucket": "high",
        }
        townsfolk_goal  = {"thirst_bucket": "low"}

        # Allow validated or trusted actions (townsfolk is not a pioneer)
        plan_steps = plan(
            townsfolk_state,
            townsfolk_goal,
            library.all_actions(),
            agent_lifecycle_floor=ActionLifecycle.VALIDATED,
        )
        self.assertGreater(len(plan_steps), 0, "Townsfolk should find a valid plan step")
        self.assertIn("spring", plan_steps[0].action.action_id)


if __name__ == "__main__":
    unittest.main()
```

---

## Recommended File Layout After All Steps

```
src/village_sim/
├── core/
│   └── types.py               ← Step 1: add PrimitiveAction
├── world/
│   ├── discoverables.py       ← Steps 2–7: Discoverable model, perception,
│   │                               interaction, regrowth, test world
│   └── world.py               ← Step 8: add discoverables field
├── agent/
│   ├── memory.py              ← Step 5: add DiscoverableMemory
│   └── perception.py          ← Step 4: extend Observation, wire perceive()
├── orchestrator/
│   ├── __init__.py
│   ├── symbolic.py            ← Step 9: bucket_need, extract_symbolic_state
│   ├── trajectory.py          ← Step 10: NeedState, StateSnapshot, Trajectory
│   ├── evaluator.py           ← Steps 11–12: TaskResult, evaluators, clustering
│   ├── induction.py           ← Step 13: fact_frequency, precondition/effect inference
│   ├── action_model.py        ← Step 14 + 19: SynthesizedAction, ActionLibrary,
│   │                               confidence update
│   └── orchestrator.py        ← Step 15: Orchestrator synthesis pipeline
├── goap/
│   ├── __init__.py
│   ├── planner.py             ← Step 16: plan()
│   ├── executor.py            ← Step 17: ExecutorRegistry
│   └── knowledge.py           ← Step 18: knowledge packets, confidence import
└── sim/
    └── engine.py              ← thread DiscoverableAgentMemory into _step_agent
tests/
├── test_discoverables.py      ← Step 20
├── test_orchestrator.py       ← Step 21
├── test_goap_planner.py       ← Step 22
├── test_knowledge.py          ← Step 23
└── test_mvp_loop.py           ← Step 24
```

---

## Design Constraints Checklist (§34)

All implementation steps above respect the following invariant:

| ✅ Do | ❌ Never |
|---|---|
| Generate JSON action models | Generate Python source code at runtime |
| Generate JSON world facts | Generate `exec()` or `eval()` calls |
| Store policy references as strings | Instantiate new classes dynamically |
| Store confidence statistics as data | Alter `sys.modules` or patch classes |
| Read/write `ActionLibrary` as JSON files | Emit executable scripts |

The `save()` / `load()` methods on `ActionLibrary` and the `save_packets()` helper in
`knowledge.py` fulfil this constraint by reading and writing plain JSON.

---

## Dependency Order Quick Reference

| Step | Depends on |
|---|---|
| 1 PrimitiveAction | — |
| 2 Discoverable model | — |
| 3 Test world | 2 |
| 4 Perception | 2, 3 |
| 5 Memory | 2, 4 |
| 6 Interaction | 2 |
| 7 Regrowth | 2 |
| 8 Wire into World | 2, 6, 7 |
| 9 Symbolic state | 1, 5 |
| 10 Trajectory recording | 1, 9 |
| 11 Task evaluators | 10 |
| 12 Clustering | 9, 10, 11 |
| 13 Precond/effect inference | 10 |
| 14 Action model + library | 13 |
| 15 Orchestrator pipeline | 11, 12, 13, 14 |
| 16 GOAP planner | 14 |
| 17 Executor | 14 |
| 18 Knowledge packets | 14 |
| 19 Confidence update | 14 |
| 20–24 Tests | all prior steps |
