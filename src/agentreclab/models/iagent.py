import re

from agentreclab.models.base import RANK_SCHEMA, STRING, BaseModel, object_schema
from agentreclab.schema import RankingPrediction


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


class I2Agent(IAgent):
    """Native dynamic i²Agent: training feedback profile + instruction-conditioned interests."""

    async def predict(self, case):
        events = self.history(case)[-15:]
        if len(events) < 2:
            raise ValueError("i2Agent requires at least two training history events")
        event = events[-2]
        catalog = self.context.catalog
        negative = case.metadata.get("warmup_negative")
        if negative is None:
            excluded = {e.item_id for e in self.history(case)} | set(case.candidates)
            negative = next((i for i in catalog if i not in excluded), None)
        if negative not in catalog or negative in {e.item_id for e in self.history(case)} | set(
            case.candidates
        ):
            raise ValueError("i2Agent requires a frozen negative outside history and candidates")
        item, other = catalog[event.item_id], catalog[negative]
        initial = [
            {
                "role": "assistant",
                "content": "{} \n. Please recommend one item for her. The first one title:{}, descrition:{}. The second one title:{}, description:{}. ".format(
                    "Here is the background of one user. ",
                    item.get("title", ""),
                    clean(str(item.get("description", ""))[-200:]),
                    other.get("title", ""),
                    clean(str(other.get("description", ""))[-200:]),
                ),
            }
        ]
        recommendation = await self.ask(
            initial, object_schema(recommend_content=STRING), case.case_id, "warmup/recommend"
        )
        initial.append(
            {
                "role": "assistant",
                "content": "The recommend content: {} \n. ".format(
                    recommendation["recommend_content"]
                ),
            }
        )
        initial.append(
            {
                "role": "assistant",
                "content": "{} \n. Great! Actually, this user choose the item with title:{} and review:{}. Can you generate the profile of this user background? Please make a detailed profile. Don’t use numerical numbering for the generated content; you can use bullet points instead.".format(
                    "", item.get("title", ""), str(event.review or "")[-200:]
                ),
            }
        )
        profile = (
            await self.ask(initial, object_schema(profile=STRING), case.case_id, "warmup/profile")
        )["profile"]
        memory, candidates, ids = self.inputs(case, events)
        messages, knowledge = await self.knowledge(case)
        message = "Based on the generated knowledge and the instruction, extract some dynamic interest information from the static memory. Moreover, based on the profile and the instruction, extract some dynamic profile information. "
        messages.append(
            {
                "role": "assistant",
                "content": "{}. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Generated Knowledge:{} Instruction:{} Historical Information:{} Profile:{}".format(
                    message, knowledge, case.instruction, memory, profile
                ),
            }
        )
        dynamic = await self.ask(
            messages,
            object_schema(dynamic_interest=STRING, dynamic_profile=STRING),
            case.case_id,
            "retrieval/dynamic",
        )
        message = (
            "Based on the information, give recommendations for the user based on the constrains. "
        )
        messages.append(
            {
                "role": "assistant",
                "content": "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{},Static Interest:{}, Static User Profile:{}, Dynamic User Profile:{}, Pure Ranking List:{}".format(
                    message,
                    candidates,
                    knowledge,
                    dynamic["dynamic_interest"],
                    memory,
                    profile,
                    dynamic["dynamic_profile"],
                    ids,
                ),
            }
        )
        result = await self.ask(messages, RANK_SCHEMA, case.case_id, "rerank", case.candidates)
        return self.prediction(case, result)
