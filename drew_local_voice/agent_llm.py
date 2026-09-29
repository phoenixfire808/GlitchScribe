"""Agent LLM client: wraps LocalOpenAIProvider with agentic temperature/max_tokens config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from drew_local_voice.ai.providers import (
    AIRequest,
    AIResponse,
    CancellationSignal,
    LocalOpenAIProvider,
)

if TYPE_CHECKING:
    from drew_local_voice.crew import Agent


@dataclass
class AgentLLMConfig:
    """Configuration for an agent LLM call — separate from AIRequest."""

    model: str
    temperature: float = 0.3
    max_tokens: int = 2048
    endpoint: str = "http://127.0.0.1:8080"
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


class AgentLLM:
    """Thin LLM client for agent execution.

    Wraps LocalOpenAIProvider but accepts per-call temperature, max_tokens,
    and model — suitable for agentic reasoning tasks where local models
    benefit from lower temperature than the default 0.2.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080",
        default_model: str = "qwen3-coder-local",
        default_temperature: float = 0.3,
        default_max_tokens: int = 2048,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._provider = LocalOpenAIProvider(
            endpoint=endpoint,
            model=default_model,
            timeout_seconds=timeout_seconds,
        )
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._default_endpoint = endpoint

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_temperature(self) -> float:
        return self._default_temperature

    def complete(
        self,
        agent: "Agent",
        task: str,
        context: str = "",
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 0,
        cancellation: Optional[CancellationSignal] = None,
    ) -> AIResponse:
        """Run one LLM call for an agent.

        Args:
            agent: the agent making the call (provides role/backstory/prompt)
            task: the specific task description for this call
            context: additional context text
            model: override model (default: agent's config model or crew default)
            temperature: override temperature (default: 0.3 for agentic reasoning)
            max_tokens: override max_tokens (default: 2048)
            cancellation: optional cancellation signal
        """
        # Resolve effective values
        eff_model = model.strip() or agent.config.model or self._default_model
        eff_temp = temperature if temperature > 0 else agent.config.temperature
        eff_tokens = max_tokens if max_tokens > 0 else agent.config.max_tokens

        # Build system prompt from agent
        coworkers = ()  # could be passed in if needed
        system_prompt = agent.system_prompt(task, context, coworkers)

        request = AIRequest(
            instruction=task,
            context=context or "(no additional context)",
            max_output_characters=min(eff_tokens * 4, 32_000),
            system_prompt=system_prompt,
            context_label="context",
            timeout_seconds=min(eff_tokens / 50 + 5, 120.0),  # rough estimate
        )

        return self._provider.complete(request, cancellation=cancellation)

    def complete_for_crew(
        self,
        agents: tuple["Agent", ...],
        task: str,
        context: str = "",
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 0,
        cancellation: Optional[CancellationSignal] = None,
    ) -> dict[str, AIResponse]:
        """Run one LLM call for each agent in a crew simultaneously.

        Returns a dict of agent_id -> AIResponse.
        """
        import concurrent.futures

        results: dict[str, AIResponse] = {}

        def call_for(agent: "Agent") -> tuple[str, AIResponse]:
            resp = self.complete(
                agent,
                task,
                context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                cancellation=cancellation,
            )
            return (agent.id, resp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {executor.submit(call_for, a): a for a in agents}
            for future in concurrent.futures.as_completed(futures):
                try:
                    agent_id, response = future.result()
                    results[agent_id] = response
                except Exception as exc:  # pragma: no cover — network
                    agent = futures[future]
                    from drew_local_voice.ai.providers import ProviderError  # noqa: F401

                    results[agent.id] = AIResponse(
                        text=f"Error: {exc}",
                        provider_label="error",
                        model=model or self._default_model,
                    )

        return results


# Singleton instance — created lazily
_agent_llm: Optional[AgentLLM] = None


def get_agent_llm(
    endpoint: str = "http://127.0.0.1:8080",
    model: str = "qwen3-coder-local",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> AgentLLM:
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = AgentLLM(
            endpoint=endpoint,
            default_model=model,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )
    return _agent_llm
