import asyncio
import logging
import re
import types

from agentreclab.core.schemas import RankingPrediction
from agentreclab.models.base import BaseModel, SyncBridge, item_mapping, load_module, upstream


class IAgentAdapter(BaseModel):
    name = "iagent"
    agent_class = "iAgent"

    def prepare(self, context):
        super().prepare(context)
        directory = upstream("iAgent")
        self.module = load_module(f"_arl_iagent_{id(self)}", directory / "model.py")
        self.module.OpenAI = lambda **kwargs: (
            None
        )  # Inject client per instance, never initialize a paid SDK.
        # TracedProvider owns backoff; upstream catch loops must not sleep or redispatch.
        self.module.time = types.SimpleNamespace(sleep=lambda seconds: None)
        # Remove label-dependent repair; runner validates the complete candidate permutation.
        self.module.cal_ndcg_hr_single = lambda *args, **kwargs: (0, 0, 0)

    async def predict(self, case):
        context = self.context
        events = [context.interactions[eid] for eid in case.history_event_ids]
        if self.agent_class == "i2Agent" and len(events) < 2:
            raise ValueError("official dynamic warmup requires at least two training interactions")
        item_map = item_mapping(context.catalog)
        inverse = {v: k for k, v in item_map.items()}
        candidates = [item_map[i] for i in case.candidates]
        negative = case.metadata.get("warmup_negative")
        if negative is None:
            negative = next(
                (
                    i
                    for i in context.catalog
                    if i not in case.candidates and i not in {e.item_id for e in events}
                ),
                None,
            )
        if negative is None:
            raise ValueError("frozen warmup negative missing")

        def clean(text):
            return re.sub(r"\<.*?\>", "", str(text))

        candidate_text = "".join(
            f"item id:{item_map[i]}, corresponding title:{str(context.catalog[i]['title'])[:50]}, description:{clean(context.catalog[i].get('description', ''))[:20]} ;"
            for i in case.candidates
        )
        task = {
            "instruction": case.instruction,
            "title": [str(context.catalog[e.item_id]["title"]) for e in events],
            "description": [str(context.catalog[e.item_id].get("description", "")) for e in events],
            "asin": [item_map[e.item_id] for e in events],
            "answer": None,
            "ranked_list_str": candidate_text,
            "pure_ranked_list": candidates,
            "reviewText": [e.review or "" for e in events],
            "neg_sample_title": context.catalog[negative]["title"],
            "neg_sample_descript": str(context.catalog[negative].get("description", "")),
        }
        bridge = SyncBridge(context.provider, asyncio.get_running_loop(), case.case_id, candidates)
        agent = getattr(self.module, self.agent_class)(task, logging.getLogger("upstream.null"))
        agent.client = bridge
        await asyncio.to_thread(agent.run)
        if bridge.error:
            raise bridge.error
        if not bridge.rankings:
            raise ValueError("official agent produced no ranking")
        ranking = tuple(inverse.get(i, f"unknown:{i}") for i in bridge.rankings[-1])
        return RankingPrediction(
            case.case_id,
            ranking,
            diagnostics={
                "implementation": "upstream_wrapper",
                "repair": "controlled_permutation_validation",
                "upstream": "e546d17848fadbcaf00e4022aa949289d7e7900b",
            },
        )
