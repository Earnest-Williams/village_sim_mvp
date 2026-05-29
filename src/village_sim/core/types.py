"""Shared value types and enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from math import sqrt


@dataclass(frozen=True, slots=True)
class Position:
    """Integer grid position."""

    x: int
    y: int

    def manhattan_to(self, other: Position) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distance_to(self, other: Position) -> float:
        dx: int = self.x - other.x
        dy: int = self.y - other.y
        return sqrt(float(dx * dx + dy * dy))


class TerrainKind(IntEnum):
    """Terrain cell classes stored as compact integers."""

    WATER = 0
    GRASS = 1
    FOREST = 2
    HILL = 3
    ROCK = 4


class ResourceKind(Enum):
    """Remembered/discovered resource kinds."""

    WATER = "water"
    FOOD = "food"


class GoalKind(Enum):
    """High-level agent goal."""

    GET_WATER = "get_water"
    GET_FOOD = "get_food"
    SLEEP = "sleep"
    EXPLORE = "explore"
    IDLE = "idle"


class ActionKind(Enum):
    """Low-level action selected for the current tick."""

    DRINK = "drink"
    EAT = "eat"
    SLEEP = "sleep"
    MOVE = "move"
    EXPLORE = "explore"
    SEARCH = "search"
    IDLE = "idle"


class PrimitiveAction(StrEnum):
    """Primitive RL action vocabulary (§6)."""

    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    MOVE_TO_TARGET = "move_to_target"
    INSPECT = "inspect"
    DRINK = "drink"
    EAT = "eat"
    SLEEP = "sleep"
    WAIT = "wait"


class DeathReason(Enum):
    """Agent death causes."""

    THIRST = "thirst"
    HUNGER = "hunger"
    EXHAUSTION = "exhaustion"
    EXPOSURE = "exposure"
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class ResourceSighting:
    """A resource visible in the current observation."""

    position: Position
    kind: ResourceKind
    amount: float


@dataclass(frozen=True, slots=True)
class MoveCandidate:
    """A possible one-tile movement candidate."""

    position: Position
    score: float
