import re

from utils.core.base import RANK_SCHEMA, STRING, BaseModel, object_schema
from utils.core.schema import BudgetExceeded, RankingPrediction, ValidationError
from utils.data.dataset import permutation


def clean(text):
    return re.sub(r"<.*?>", "", str(text))


class IAgent(BaseModel):
    """Native static iAgent: instruction knowledge -> candidate reranking."""

    supports_batch = True
    handles_step_errors = True

    def request_step(self, messages, schema, case_id, phase, candidates=()):
        """Upstream's three attempts; change cache identity only for actual retries."""
        fields = schema["properties"]
        for attempt in range(3):
            request = self.request(messages, schema, case_id, phase, candidates)
            request["kwargs"]["single_attempt"] = True
            if attempt:
                request["kwargs"]["response_attempt"] = attempt
            answer = yield request
            if not isinstance(answer, Exception):
                return answer
            if isinstance(answer, BudgetExceeded) or (
                isinstance(answer, RuntimeError) and "outcome is unknown" in str(answer)
            ):
                raise answer
            self.context.tracer.record(
                {
                    "event": "step_retry",
                    "case_id": case_id,
                    "phase": phase,
                    "attempt": attempt,
                    "error_type": type(answer).__name__,
                    "error": str(answer),
                }
            )
        # Upstream proceeds to later steps with this sentinel after extraction failure.
        return {key: "Extract Error!" for key in fields}

    def finish_ranking(
        self, case, messages, memory, candidates, ids, knowledge, dynamic=None, profile=None
    ):
        result = yield from self.request_step(
            messages, RANK_SCHEMA, case.case_id, "rerank", case.candidates
        )
        policy = self.context.config.get("invalid_prediction_policy", "fail")

        def acceptable(value):
            ranking = value["rerank_list"]
            if not isinstance(ranking, list):
                return False
            if policy == "upstream":
                # Exact released-code gate: uses the test target to decide repair.
                return self.item_ids[case.positive_item_id] in ranking
            try:
                permutation(ranking, ids)
                return True
            except ValueError:
                return False

        for repair in range(1, 4):
            if acceptable(result):
                break
            previous = result["rerank_list"]
            retry = (
                "Rerank list is out of the order, you should rerank the item from the pure "
                "ranking list. The previous list:{}. Therefore, try it again according the "
                "following information."
            ).format(previous)
            if dynamic is None:
                content = "{}. \\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \\n Candidate ranking list:{},Knowledge:{},,Static Inster:{},  Please generate the reranked list from Pure Ranking List:{}. The length of the reranked list should be 10.".format(
                    retry, candidates, knowledge, memory, ids
                )
            else:
                content = "{}. \\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \\n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{},Static Inster:{}, User Profile:{}, Please generate the reranked list from Pure Ranking List:{}. The length of the reranked list should be 10.".format(
                    retry, candidates, knowledge, dynamic["dynamic_interest"], memory, profile, ids
                )
            messages.append({"role": "assistant", "content": content.replace("\\n", "\n")})
            result = yield from self.request_step(
                messages, RANK_SCHEMA, case.case_id, f"rerank/repair_{repair}", case.candidates
            )
        if not acceptable(result):
            raise ValidationError("ranking still invalid after three upstream repair rounds")
        prediction = self.prediction(case, result)
        if policy == "upstream":
            prediction.diagnostics["repair_gate"] = "upstream_target_presence"
        return prediction

    async def predict_batch(self, cases, batch):
        return await self.run_batch_steps(cases, batch)

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

    def knowledge_messages(self, case):
        message = "Based on the following instruction, assist me in generating relevant knowledge. Please specify the types of descriptions that the recommended items should include. Do not directly recommend specific items. "
        messages = [
            {
                "role": "assistant",
                "content": "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(
                    message, case.instruction
                ),
            }
        ]
        return messages

    def workflow(self, case):
        memory, candidates, ids = self.inputs(case, self.history(case))
        messages = self.knowledge_messages(case)
        knowledge = (
            yield from self.request_step(
                messages, object_schema(knowledge=STRING), case.case_id, "retrieval"
            )
        )["knowledge"]
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
        return (yield from self.finish_ranking(case, messages, memory, candidates, ids, knowledge))

    def prediction(self, case, result):
        return RankingPrediction(
            case.case_id,
            tuple(self.original_ids.get(i, f"unknown:{i}") for i in result["rerank_list"]),
            diagnostics={"implementation": "native", "explanation": result["explanation"]},
        )
