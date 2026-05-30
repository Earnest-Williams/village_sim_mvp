"""MessagePack-based agent handoff between region actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import msgpack
import numpy as np
import polars as pl
from numpy.typing import NDArray

from village_sim.agent.memory import GlobalMemory, MEMORY_AGENT_ID, MEMORY_SCHEMA
from village_sim.agent.state import AgentArrays

AGENT_FIELD_NAMES: tuple[str, ...] = (
    "active",
    "x",
    "y",
    "thirst",
    "hunger",
    "fatigue",
    "cold_stress",
    "health",
    "awake_ticks",
    "current_goal",
    "current_action",
    "action_queue_kind",
    "action_queue_duration",
)


@dataclass(frozen=True, slots=True)
class RegionBounds:
    """Static map chunk location in global world coordinates."""

    origin_x: int
    origin_y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("region dimensions must be positive")


@dataclass(frozen=True, slots=True)
class HandoffBatch:
    """Decoded handoff payload ready for insertion into a region."""

    agent_ids: NDArray[np.int64]
    absolute_x: NDArray[np.int32]
    absolute_y: NDArray[np.int32]
    fields: dict[str, NDArray[Any]]
    memory_frame: pl.DataFrame

    @property
    def count(self) -> int:
        """Return the number of transferred agents in O(1)."""

        return int(self.agent_ids.size)


def departing_agent_mask(
    arrays: AgentArrays, bounds: RegionBounds
) -> NDArray[np.bool_]:
    """Return active agents whose local coordinates left this region."""

    return arrays.active & (
        (arrays.x < 0)
        | (arrays.y < 0)
        | (arrays.x >= bounds.width)
        | (arrays.y >= bounds.height)
    )


def group_departing_agents_by_neighbor(
    arrays: AgentArrays,
    memory: GlobalMemory,
    bounds: RegionBounds,
    agent_ids: NDArray[np.int64],
) -> dict[tuple[int, int], NDArray[np.uint8]]:
    """Detach out-of-bounds agents and return MessagePack buffers by neighbor delta."""

    if agent_ids.shape != arrays.active.shape:
        raise ValueError("agent_ids must match AgentArrays capacity")
    mask = departing_agent_mask(arrays, bounds)
    departing_indices: NDArray[np.int64] = np.where(mask)[0].astype(np.int64)
    if departing_indices.size == 0:
        return {}

    x_values: NDArray[np.int32] = arrays.x[departing_indices]
    y_values: NDArray[np.int32] = arrays.y[departing_indices]
    dx_values: NDArray[np.int8] = np.where(
        x_values < 0,
        np.int8(-1),
        np.where(x_values >= bounds.width, np.int8(1), np.int8(0)),
    ).astype(np.int8)
    dy_values: NDArray[np.int8] = np.where(
        y_values < 0,
        np.int8(-1),
        np.where(y_values >= bounds.height, np.int8(1), np.int8(0)),
    ).astype(np.int8)
    deltas: NDArray[np.int8] = np.stack((dx_values, dy_values), axis=1)
    unique_deltas: NDArray[np.int8] = np.unique(deltas, axis=0)

    departing_ids: NDArray[np.int64] = agent_ids[departing_indices]
    memory_frame: pl.DataFrame = detach_memory_rows(memory, departing_ids)
    buffers: dict[tuple[int, int], NDArray[np.uint8]] = {}
    for delta in unique_deltas:
        delta_mask: NDArray[np.bool_] = np.all(deltas == delta, axis=1)
        selected_indices: NDArray[np.int64] = departing_indices[delta_mask]
        selected_ids: NDArray[np.int64] = agent_ids[selected_indices]
        selected_memory: pl.DataFrame = memory_frame.filter(
            pl.col(MEMORY_AGENT_ID).is_in(selected_ids)
        )
        key = (int(delta[0]), int(delta[1]))
        buffers[key] = pack_agent_handoff(
            arrays=arrays,
            indices=selected_indices,
            agent_ids=selected_ids,
            bounds=bounds,
            memory_frame=selected_memory,
        )

    arrays.active[departing_indices] = False
    return buffers


def detach_memory_rows(
    memory: GlobalMemory, agent_ids: NDArray[np.int64]
) -> pl.DataFrame:
    """Serialize-and-purge memory rows for departing agents."""

    memory.flush_pending()
    if agent_ids.size == 0 or memory.frame.is_empty():
        return pl.DataFrame(schema=MEMORY_SCHEMA)
    selected = memory.frame.filter(pl.col(MEMORY_AGENT_ID).is_in(agent_ids))
    if selected.is_empty():
        return pl.DataFrame(schema=MEMORY_SCHEMA)
    memory.frame = memory.frame.filter(~pl.col(MEMORY_AGENT_ID).is_in(agent_ids))
    id_set = frozenset(int(agent_id) for agent_id in agent_ids)
    memory._pending_dict = {
        key: value
        for key, value in memory._pending_dict.items()
        if key[0] not in id_set
    }
    return selected.cast(MEMORY_SCHEMA)


def pack_agent_handoff(
    *,
    arrays: AgentArrays,
    indices: NDArray[np.int64],
    agent_ids: NDArray[np.int64],
    bounds: RegionBounds,
    memory_frame: pl.DataFrame,
) -> NDArray[np.uint8]:
    """Pack agent rows and memory rows into a flat uint8 MessagePack buffer."""

    if indices.size != agent_ids.size:
        raise ValueError("handoff indices and agent_ids must have equal length")
    absolute_x: NDArray[np.int32] = (arrays.x[indices] + bounds.origin_x).astype(
        np.int32
    )
    absolute_y: NDArray[np.int32] = (arrays.y[indices] + bounds.origin_y).astype(
        np.int32
    )
    fields: dict[str, dict[str, object]] = {}
    for field_name in AGENT_FIELD_NAMES:
        values = cast(NDArray[Any], getattr(arrays, field_name))[indices]
        fields[field_name] = _pack_array(values)

    payload: dict[str, object] = {
        "agent_ids": _pack_array(agent_ids.astype(np.int64, copy=False)),
        "absolute_x": _pack_array(absolute_x),
        "absolute_y": _pack_array(absolute_y),
        "fields": fields,
        "memory": _pack_memory_frame(memory_frame),
    }
    packed: bytes = msgpack.packb(payload, use_bin_type=True)
    return np.frombuffer(packed, dtype=np.uint8).copy()


def unpack_agent_handoff(buffer: NDArray[np.uint8]) -> HandoffBatch:
    """Decode a flat uint8 MessagePack handoff buffer."""

    payload = msgpack.unpackb(buffer.tobytes(), raw=False)
    if not isinstance(payload, dict):
        raise TypeError("handoff payload must be a map")
    agent_ids = _unpack_array(_expect_map(payload, "agent_ids"), np.int64)
    absolute_x = _unpack_array(_expect_map(payload, "absolute_x"), np.int32)
    absolute_y = _unpack_array(_expect_map(payload, "absolute_y"), np.int32)
    fields_payload = payload.get("fields")
    if not isinstance(fields_payload, dict):
        raise TypeError("handoff fields must be a map")
    fields: dict[str, NDArray[Any]] = {}
    for field_name in AGENT_FIELD_NAMES:
        field_payload = fields_payload.get(field_name)
        if not isinstance(field_payload, dict):
            raise TypeError("handoff field payload must be a map")
        fields[field_name] = _unpack_array_dynamic(field_payload)
    memory_payload = payload.get("memory")
    if not isinstance(memory_payload, dict):
        raise TypeError("handoff memory must be a map")
    memory_frame = _unpack_memory_frame(memory_payload)
    if absolute_x.size != agent_ids.size or absolute_y.size != agent_ids.size:
        raise ValueError("handoff coordinate arrays must match agent_ids")
    return HandoffBatch(
        agent_ids=agent_ids,
        absolute_x=absolute_x,
        absolute_y=absolute_y,
        fields=fields,
        memory_frame=memory_frame,
    )


def receive_agent_handoff(
    arrays: AgentArrays,
    memory: GlobalMemory,
    bounds: RegionBounds,
    agent_ids: NDArray[np.int64],
    buffer: NDArray[np.uint8],
) -> int:
    """Insert a decoded handoff into free local slots and merge its memory rows."""

    if agent_ids.shape != arrays.active.shape:
        raise ValueError("agent_ids must match AgentArrays capacity")
    batch = unpack_agent_handoff(buffer)
    if batch.count == 0:
        return 0
    free_slots: NDArray[np.int64] = np.where(~arrays.active)[0][: batch.count].astype(
        np.int64
    )
    if free_slots.size < batch.count:
        raise RuntimeError("region has insufficient free agent capacity")

    for field_name in AGENT_FIELD_NAMES:
        target = cast(NDArray[Any], getattr(arrays, field_name))
        target[free_slots] = batch.fields[field_name]
    arrays.x[free_slots] = (batch.absolute_x - bounds.origin_x).astype(np.int32)
    arrays.y[free_slots] = (batch.absolute_y - bounds.origin_y).astype(np.int32)
    arrays.active[free_slots] = True
    agent_ids[free_slots] = batch.agent_ids
    if not batch.memory_frame.is_empty():
        memory.frame = pl.concat([memory.frame, batch.memory_frame], how="vertical")
        memory.frame = memory.frame.unique(
            subset=["agent_id", "kind", "x", "y"],
            keep="last",
            maintain_order=True,
        ).cast(MEMORY_SCHEMA)
    return batch.count


def _pack_array(values: NDArray[Any]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(values)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def _unpack_array(payload: dict[Any, Any], dtype: type[np.generic]) -> NDArray[Any]:
    values = _unpack_array_dynamic(payload)
    expected_dtype = np.dtype(dtype)
    if values.dtype != expected_dtype:
        return values.astype(expected_dtype)
    return values


def _unpack_array_dynamic(payload: dict[Any, Any]) -> NDArray[Any]:
    dtype_value = payload.get("dtype")
    shape_value = payload.get("shape")
    data_value = payload.get("data")
    if not isinstance(dtype_value, str):
        raise TypeError("array dtype must be a string")
    if not isinstance(shape_value, list):
        raise TypeError("array shape must be a list")
    if not isinstance(data_value, bytes):
        raise TypeError("array data must be bytes")
    shape = tuple(_expect_int(item) for item in shape_value)
    return np.frombuffer(data_value, dtype=np.dtype(dtype_value)).copy().reshape(shape)


def _pack_memory_frame(frame: pl.DataFrame) -> dict[str, object]:
    cast_frame = frame.cast(MEMORY_SCHEMA)
    columns: dict[str, object] = {}
    for column_name in MEMORY_SCHEMA.keys():
        series = cast_frame.get_column(column_name, default=None)
        if series is None:
            values = np.empty(0, dtype=_memory_numpy_dtype(column_name))
        else:
            values = series.to_numpy().astype(_memory_numpy_dtype(column_name))
        columns[column_name] = _pack_array(values)
    return {"height": cast_frame.height, "columns": columns}


def _unpack_memory_frame(payload: dict[Any, Any]) -> pl.DataFrame:
    height_value = payload.get("height")
    columns_value = payload.get("columns")
    height = _expect_int(height_value)
    if not isinstance(columns_value, dict):
        raise TypeError("memory columns must be a map")
    data: dict[str, NDArray[Any]] = {}
    for column_name in MEMORY_SCHEMA.keys():
        column_payload = columns_value.get(column_name)
        if not isinstance(column_payload, dict):
            raise TypeError("memory column payload must be a map")
        values = _unpack_array_dynamic(column_payload)
        if values.size != height:
            raise ValueError("memory column size must match memory height")
        data[column_name] = values
    return pl.DataFrame(data, schema=MEMORY_SCHEMA)


def _memory_numpy_dtype(column_name: str) -> np.dtype[Any]:
    dtype = MEMORY_SCHEMA[column_name]
    if dtype == pl.Int8:
        return np.dtype(np.int8)
    if dtype == pl.Int32:
        return np.dtype(np.int32)
    if dtype == pl.Int64:
        return np.dtype(np.int64)
    if dtype == pl.Float32:
        return np.dtype(np.float32)
    raise TypeError("unsupported memory dtype")


def _expect_map(payload: dict[Any, Any], key: str) -> dict[Any, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError("handoff payload value must be a map")
    return value


def _expect_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("value must be an integer")
    if isinstance(value, int):
        return value
    raise TypeError("value must be an integer")
