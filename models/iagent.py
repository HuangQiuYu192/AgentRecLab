import re

from utils.base import RANK_SCHEMA, STRING, BaseModel, object_schema
from utils.schema import RankingPrediction


def clean(text):
    return re.sub(r"<.*?>", "", str(text))


class IAgent(BaseModel):
    """Native static iAgent: instruction knowledge -> candidate reranking."""

    def inputs(self, case, events):
        catalog = self.context.catalog
        memory = "".join(
            "user historical information, item title:{},item description:{} ;".format(
                catalog[e.item_id].get("title", ""),
                clean(str(catalog[e.item_id].get("description", ""))[-200:]),
            )
            for e in events
        )
        candidates = "".join(
            "item id:{}, corresponding title:{}, description:{} ;".format(
                self.item_ids[i],
                str(catalog[i].get("title", ""))[:50],
                clean(catalog[i].get("description", ""))[:20],
            )
            for i in case.candidates
        )
        return memory, candidates, [self.item_ids[i] for i in case.candidates]

    async def knowledge(self, case):
        message = "Based on the following instruction, assist me in generating relevant knowledge. Please specify the types of descriptions that the recommended items should include. Do not directly recommend specific items. "
        messages = [
            {
                "role": "assistant",
                "content": "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(
                    message, case.instruction
                ),
            }
        ]
        result = await self.ask(
            messages, object_schema(knowledge=STRING), case.case_id, "retrieval"
        )
        return messages, result["knowledge"]

    async def predict(self, case):
        memory, candidates, ids = self.inputs(case, self.history(case))
        messages, knowledge = await self.knowledge(case)
        message = (
            "Based on the information, give recommendations for the user based on the constraints. "
        )
        messages.append(
            {
                "role": "assistant",
                "content": "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Static Interest:{}, Pure Ranking List:{}".format(
                    message, candidates, knowledge, memory, ids
                ),
            }
        )
        result = await self.ask(messages, RANK_SCHEMA, case.case_id, "rerank", case.candidates)
        return self.prediction(case, result)

    def prediction(self, case, result):
        return RankingPrediction(
            case.case_id,
            tuple(self.original_ids.get(i, f"unknown:{i}") for i in result["rerank_list"]),
            diagnostics={"implementation": "native", "explanation": result["explanation"]},
        )
