from agentreclab.models.base import BaseModel


class MarsAdapter(BaseModel):
    name = "mars"
    status = "experimental/not_ready"

    def prepare(self, context):
        raise NotImplementedError(
            "MARS experimental/not_ready: see docs/model_mapping/mars.md; no verified implementation"
        )
