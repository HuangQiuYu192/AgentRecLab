import asyncio
import importlib
import random
import sys
import types

from agentreclab.core.schemas import RankingPrediction
from agentreclab.data.validation import permutation
from agentreclab.models.base import BaseModel, SyncBridge, item_mapping, namespace, upstream


class MemRecAdapter(BaseModel):
    name = "memrec"

    def prepare(self, context):
        super().prepare(context)
        self.directory = upstream("MemRec")
        prefix = "_arl_memrec"
        namespace(prefix, self.directory / "src")
        namespace(prefix + ".memory", self.directory / "src" / "memory")
        namespace(prefix + ".models", self.directory / "src" / "models")
        memory = sys.modules[prefix + ".memory"]
        exports = {
            "UserItemGraph": "graph",
            "SnippetPacker": "packer",
            "MemRecManager": "manager",
            "MemoryStorage": "storage",
        }
        for symbol, module in exports.items():
            setattr(
                memory,
                symbol,
                getattr(importlib.import_module(prefix + ".memory." + module), symbol),
            )
        # Unselected MLP branch deliberately has no torch dependency.
        memory.NeighborPruner = None
        source = (self.directory / "src" / "models" / "memrec_agent.py").read_text(encoding="utf-8")
        module = types.ModuleType(prefix + ".models.agent")
        # Only relocate absolute imports; official functions/classes/prompts remain intact.
        exec(
            compile(
                source.replace("from src.", "from _arl_memrec."),
                str(self.directory / "src" / "models" / "memrec_agent.py"),
                "exec",
            ),
            module.__dict__,
        )
        self.agent_class = module.MemRecAgent
        self.graph_class = memory.UserItemGraph
        self.item_map = item_mapping(context.catalog)
        self.inverse = {v: k for k, v in self.item_map.items()}
        self.user_map = {
            uid: j
            for j, uid in enumerate(dict.fromkeys(e.user_id for e in context.interactions.values()))
        }
        self.histories = {u: [] for u in self.user_map}
        for e in context.interactions.values():
            self.histories[e.user_id].append(e)
        if context.config["model_config"].get("warmup_rounds", 1) != 1:
            raise ValueError("only audited one-round MemRec warmup is supported")
        self.dataset = types.SimpleNamespace(
            name="instructrec-" + context.cases[0].dataset_id,
            train_data={
                self.user_map[u]: [self.item_map[e.item_id] for e in history[:-1]]
                for u, history in self.histories.items()
            },
            item_metadata={self.item_map[i]: v for i, v in context.catalog.items()},
        )
        self.agent = None

    def _initialize(self, loop):
        if self.agent is not None:
            return
        bridge = SyncBridge(self.context.provider, loop, "prepare", [])
        opts = self.context.config["model_config"]
        self.agent = self.agent_class(
            self.dataset,
            bridge,
            k=opts["k"],
            tau=opts["tau_tokens"],
            n_facets=opts["n_facets"],
            mix_min_users=opts["mix_min_users"],
            mix_min_items=opts["mix_min_items"],
            fanout_cap=opts["fanout_cap"],
            temperature=self.context.config["provider"].get("temperature", 0),
            max_tokens=self.context.config["provider"].get("max_tokens", 4000),
            reranker_mode="llm",
            pruner_mode="llm_rules",
        )

    async def warmup(self):
        loop = asyncio.get_running_loop()
        self._initialize(loop)
        seed = self.context.config.get("seed", 42)
        plans = []
        # Freeze training candidates and chronological order; no access to test labels.
        for uid, history in self.histories.items():
            if len(history) < 2:
                raise ValueError("MemRec warmup requires at least two historical events")
            target = history[-1]
            pool = [i for i in self.item_map if i not in {e.item_id for e in history}]
            rng = random.Random(seed + self.user_map[uid])
            if len(pool) < 9:
                raise ValueError("insufficient catalog for training-only warmup negatives")
            candidates = [target.item_id] + rng.sample(pool, 9)
            rng.shuffle(candidates)
            plans.append(
                {
                    "user_id": uid,
                    "target_event_id": target.event_id,
                    "candidates": candidates,
                    "timestamp": target.timestamp,
                }
            )
        # Source sequence timestamps may be unavailable; retain stable row order then.
        if all(p["timestamp"] for p in plans):
            plans.sort(key=lambda p: (p["timestamp"], p["user_id"]))
        self.state["warmup_manifest"] = plans
        base_train = self.dataset.train_data
        for plan in plans:
            uid = plan["user_id"]
            cutoff = plan["timestamp"]
            if cutoff:
                from agentreclab.data.validation import instant

                # Collaborative context is cut at the training feedback boundary as well.
                self.dataset.train_data = {
                    self.user_map[u]: [
                        self.item_map[e.item_id]
                        for e in h
                        if e.timestamp and instant(e.timestamp) < instant(cutoff)
                    ]
                    for u, h in self.histories.items()
                }
                self.agent.graph = self.graph_class(self.dataset)
            ids = [self.item_map[i] for i in plan["candidates"]]
            target = self.context.interactions[plan["target_event_id"]]
            bridge = SyncBridge(self.context.provider, loop, "warmup:" + uid, ids, "warmup")
            self._set_bridge(bridge)
            ranked, details = await asyncio.to_thread(
                self.agent.rerank, self.user_map[uid], ids, target.instruction, True
            )
            if bridge.error:
                raise bridge.error
            permutation(ranked, ids)
            clicked = self.item_map[target.item_id]
            bridge.phase = "memory_update"
            await asyncio.to_thread(
                self.agent.write,
                self.user_map[uid],
                {"action": "CLICK", "item_id": clicked, "position": ranked.index(clicked)},
                details.get("facets", []),
                details.get("pruned_subgraph"),
            )
            if bridge.error:
                raise bridge.error
            self.state["completed_warmup_users"] = self.state.get("completed_warmup_users", []) + [
                uid
            ]
        self.dataset.train_data = base_train
        self.agent.graph = self.graph_class(self.dataset)
        self.state["user_profiles"] = self.agent.storage.user_profiles
        self.state["item_descriptions"] = self.agent.storage.item_descriptions
        self.state["n_updates"] = self.agent.storage.n_updates

    def _set_bridge(self, bridge):
        self.agent.llm_client = bridge
        self.agent.manager.llm = bridge
        self.agent.reranker.llm = bridge

    async def predict(self, case):
        # These bridge references are mutable: runner serializes MemRec evaluation.
        self._initialize(asyncio.get_running_loop())
        ids = [self.item_map[i] for i in case.candidates]
        bridge = SyncBridge(self.context.provider, asyncio.get_running_loop(), case.case_id, ids)
        self._set_bridge(bridge)
        ranked, details = await asyncio.to_thread(
            self.agent.rerank, self.user_map[case.user_id], ids, case.instruction, True
        )
        if bridge.error:
            raise bridge.error
        return RankingPrediction(
            case.case_id,
            tuple(self.inverse.get(i, f"unknown:{i}") for i in ranked),
            diagnostics={
                "implementation": "upstream_components",
                "neighbors": details["pruned_subgraph"],
                "upstream": "58d9031ed91c623b8034d2bf04f39aa937424c33",
                "test_observe": False,
            },
        )

    def load_state(self, source):
        super().load_state(source)
        self._initialize(asyncio.get_running_loop())
        self.agent.storage.user_profiles = {
            int(k): v for k, v in self.state["user_profiles"].items()
        }
        self.agent.storage.item_descriptions = {
            int(k): v for k, v in self.state["item_descriptions"].items()
        }
        self.agent.storage.n_updates = self.state["n_updates"]
