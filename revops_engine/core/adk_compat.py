"""Small ADK-compatible primitives for local execution and testing.

If the real ADK package is available, this module re-exports those classes.
Otherwise it falls back to local implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:  # pragma: no cover - path used when adk is installed
    from adk import Agent, LLMAgent, Orchestrator, RemoteA2AAgent, SequentialAgent
except ImportError:  # pragma: no cover
    @dataclass
    class AgentRequest:
        args: dict[str, Any]

    class Agent:
        def __init__(self, name: str = "agent"):
            self.name = name

        def run(self, payload: Any, context: dict[str, Any] | None = None) -> Any:
            raise NotImplementedError

    class LLMAgent(Agent):
        def __init__(
            self,
            model: str,
            instructions: str,
            tools: list[str] | None = None,
            handler: Callable[[Any, dict[str, Any] | None], Any] | None = None,
            name: str = "llm-agent",
        ):
            super().__init__(name=name)
            self.model = model
            self.instructions = instructions
            self.tools = tools or []
            self._handler = handler

        def run(self, payload: Any, context: dict[str, Any] | None = None) -> Any:
            if self._handler:
                return self._handler(payload, context)
            return payload

    class SequentialAgent(Agent):
        def __init__(self, agents: list[Agent], name: str = "sequential-agent"):
            super().__init__(name=name)
            self.agents = agents
            self._callbacks: dict[str, list[Callable[[AgentRequest], tuple[bool, str | None]]]] = {}

        def add_callback(
            self,
            stage: str,
            callback: Callable[[AgentRequest], tuple[bool, str | None]],
        ) -> None:
            self._callbacks.setdefault(stage, []).append(callback)

        def run(self, payload: Any, context: dict[str, Any] | None = None) -> Any:
            request = AgentRequest(args={"data": payload, "context": context or {}})
            for callback in self._callbacks.get("before_agent", []):
                ok, error = callback(request)
                if not ok:
                    raise ValueError(error or "Callback rejected request")

            current = payload
            for agent in self.agents:
                current = agent.run(current, context=context)
            return current

    class RemoteA2AAgent(Agent):
        def __init__(self, url: str, handler: Callable[[Any], Any] | None = None, name: str = "remote-a2a"):
            super().__init__(name=name)
            self.url = url
            self._handler = handler

        def run(self, payload: Any, context: dict[str, Any] | None = None) -> Any:
            if not self._handler:
                raise RuntimeError(f"No handler configured for remote agent {self.url}")
            return self._handler(payload)

    class Orchestrator(Agent):
        def __init__(self, agents: list[Agent], model: str, instructions: str, name: str = "orchestrator"):
            super().__init__(name=name)
            self.agents = agents
            self.model = model
            self.instructions = instructions

        def on_failure(self, error: Exception) -> None:
            raise error
