"""Execution payload dispatcher (§19)."""

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
