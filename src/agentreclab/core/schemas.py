from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Interaction:
    event_id: str
    user_id: str
    item_id: str
    timestamp: str | None
    review: str | None = None
    instruction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingCase:
    case_id: str
    dataset_id: str
    user_id: str
    instruction: str
    history_event_ids: tuple[str, ...]
    candidates: tuple[str, ...]
    positive_item_id: str
    evaluation_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingPrediction:
    case_id: str
    ordered_item_ids: tuple[str, ...]
    scores: dict[str, float] | None = None
    rationales: dict[str, str] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: LLMUsage | None
    provider: str
    model: str
    request_id: str | None
    raw_response_ref: str
    latency_ms: float


@dataclass(frozen=True)
class FeedbackEvent:
    user_id: str
    item_id: str
    instruction: str
    timestamp: str
    review: str | None = None
    observed_labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentContext:
    catalog: dict[str, dict]
    interactions: dict[str, Interaction]
    cases: tuple[RankingCase, ...]
    provider: Any
    state_store: Any
    tracer: Any
    protocol_mode: str
    config: dict
