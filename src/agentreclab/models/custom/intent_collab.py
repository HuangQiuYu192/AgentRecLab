from agentreclab.core.schemas import RankingPrediction
from agentreclab.models.base import BaseModel


class IntentCollab(BaseModel):
    """Injected retriever example; never forced on baseline algorithms."""

    name = "custom"

    def __init__(self, retrieval=None):
        self.retrieval = retrieval

    async def predict(self, case):
        if self.retrieval is None:
            raise ValueError("custom model requires an explicitly injected retrieval function")
        ranked = self.retrieval(case.instruction, case.candidates, self.context.catalog)
        return RankingPrediction(case.case_id, tuple(ranked))
