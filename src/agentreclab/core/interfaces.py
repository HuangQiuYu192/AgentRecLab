from typing import Any, Protocol

from agentreclab.core.schemas import (
    ExperimentContext,
    FeedbackEvent,
    LLMResult,
    RankingCase,
    RankingPrediction,
)


class LLMProvider(Protocol):
    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResult: ...


class AgentModel(Protocol):
    name: str

    def prepare(self, context: ExperimentContext) -> None: ...
    async def predict(self, case: RankingCase) -> RankingPrediction: ...
    async def observe(self, feedback: FeedbackEvent) -> None: ...
    def save_state(self, destination: str) -> None: ...
    def load_state(self, source: str) -> None: ...
