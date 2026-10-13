from models.iagent import IAgent, clean
from utils.core.base import RANK_SCHEMA, STRING, object_schema


class I2Agent(IAgent):
    """Native dynamic i²Agent: training feedback profile + instruction-conditioned interests."""

    def workflow(self, case):
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
        recommendation = yield self.request(
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
            yield self.request(
                initial, object_schema(profile=STRING), case.case_id, "warmup/profile"
            )
        )["profile"]
        memory, candidates, ids = self.inputs(case, events)
        messages = self.knowledge_messages(case)
        knowledge = (
            yield self.request(messages, object_schema(knowledge=STRING), case.case_id, "retrieval")
        )["knowledge"]
        message = "Based on the generated knowledge and the instruction, extract some dynamic interest information from the static memory. Moreover, based on the profile and the instruction, extract some dynamic profile information. "
        messages.append(
            {
                "role": "assistant",
                "content": "{}. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Generated Knowledge:{} Instruction:{} Historical Information:{} Profile:{}".format(
                    message, knowledge, case.instruction, memory, profile
                ),
            }
        )
        dynamic = yield self.request(
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
        result = yield self.request(messages, RANK_SCHEMA, case.case_id, "rerank", case.candidates)
        return self.prediction(case, result)
