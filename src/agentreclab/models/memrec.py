import math
import random
from collections import Counter, defaultdict

from agentreclab.data import instant, permutation
from agentreclab.models.base import STRING, BaseModel, object_schema
from agentreclab.schema import RankingPrediction


def domain_score(neighbor, domain):
    w, m, c, s, d = (
        neighbor[k] for k in ("score", "metadata_overlap", "co_count", "memory_sim", "timestamp")
    )
    if domain == "books":
        cf = (1.8 if c > 3 else 1) * (1.5 if c > 3 and s > 0.5 else 1)
        return (
            w
            * (2.5 if m > 0.6 else 1)
            * cf
            * (math.exp(-0.004 * d) if d > 180 else 1)
            * (1 + (1.2 if neighbor["type"] == "item" else 0.8) * s)
        )
    if domain == "movietv":
        decay = 1 if d <= 60 else math.exp(-(0.018 if d <= 180 else 0.025) * d)
        memory = (1 + (1.8 if c >= 3 else 1.5) * s) * (1.5 if m > 0.6 else 1)
        return (
            w * (2.5 if c >= 3 else 1) * decay * (0.3 if d > 365 else 1) + (2.8 * m if c < 3 else 1)
        ) * memory
    if domain == "goodreads":
        social = (2 if c > 10 else 1) * (1.5 if c > 15 else 1)
        return (
            w
            * (0.7 if c > 15 else 1)
            * social
            * (1 + (1.5 if c > 10 else 1) * s)
            * (3 if m > 0.8 else 1)
            * (math.exp(-0.002 * d) if d > 365 else 1)
        )
    if domain == "yelp":
        decay = 1 if d <= 90 else math.exp(-0.028 * d) * (1 if d <= 180 else 0.5)
        return (
            w
            * (4.5 if m > 0.85 else 3.5 if m > 0.7 else 1)
            * decay
            * (1 + (2.2 if m > 0.7 else 1) * s)
            * (2 if c >= 2 else 0.5)
            * (0.2 if m < 0.4 else 1)
        )
    raise ValueError(f"unsupported MemRec domain: {domain}")


FACETS = object_schema(
    facets={
        "type": "array",
        "items": object_schema(
            facet=STRING,
            confidence={"type": "number"},
            supporting_neighbors={"type": "array", "items": STRING},
        ),
    },
    support_edges={
        "type": "array",
        "items": object_schema(**{"from": STRING, "to": STRING, "w": {"type": "number"}}),
    },
)
SCORES = object_schema(
    scores={
        "type": "array",
        "items": object_schema(
            item_id={"type": "integer"}, score={"type": "number"}, rationale=STRING
        ),
    }
)
UPDATE = object_schema(
    user_memory=STRING,
    item_memory=STRING,
    neighbor_updates={
        "type": "array",
        "items": object_schema(neighbor_id=STRING, memory_update=STRING, rationale=STRING),
    },
)


class MemRec(BaseModel):
    """Compact native MemRec; train-only memory write, read-only test inference."""

    def prepare(self, context):
        super().prepare(context)
        self.options = context.config.get("model_config", {})
        self.domain = context.cases[0].dataset_id
        self.histories = defaultdict(list)
        for event in context.interactions.values():
            self.histories[event.user_id].append(event)
        self.user_ids = {u: n for n, u in enumerate(self.histories)}
        self.original_users = {v: k for k, v in self.user_ids.items()}
        self.state = {
            "user_profiles": {},
            "item_descriptions": {
                i: str(v.get("title", "")) + " " + str(v.get("description", ""))[:300]
                for i, v in context.catalog.items()
            },
            "completed_warmup_users": [],
            "warmup_manifest": [],
            "n_updates": 0,
        }

    def graph(self, cutoff=None):
        return {
            u: [e for e in events if instant(e.timestamp) < instant(cutoff)]
            if cutoff
            else events[:-1]
            for u, events in self.histories.items()
        }

    def neighbors(self, user, graph):
        history = graph.get(user, [])
        recency = {e.item_id: (n + 1) / len(history) for n, e in enumerate(history)}
        raw = [
            {
                "type": "item",
                "id": self.item_ids[i],
                "score": r,
                "metadata_overlap": 0.5,
                "co_count": 0,
                "memory_sim": 0,
                "timestamp": (1 - r) * 365,
            }
            for i, r in sorted(
                ((e.item_id, recency[e.item_id]) for e in history), key=lambda pair: -pair[1]
            )[:100]
        ]
        users_by_item = defaultdict(list)
        for u, events in graph.items():
            for event in events:
                users_by_item[event.item_id].append(u)
        overlap = Counter(u for event in history for u in users_by_item[event.item_id] if u != user)
        raw.extend(
            {
                "type": "user",
                "id": self.user_ids[u],
                "score": count / max(1, len(history)),
                "metadata_overlap": 0,
                "co_count": count,
                "memory_sim": 0,
                "timestamp": 0,
            }
            for u, count in overlap.most_common(50)
        )
        ranked = sorted(raw, key=lambda n: -domain_score(n, self.domain))
        selected = [n for n in ranked if n["type"] == "user"][
            : self.options.get("mix_min_users", 4)
        ]
        selected += [n for n in ranked if n["type"] == "item"][
            : self.options.get("mix_min_items", 6)
        ]
        keys = {(n["type"], n["id"]) for n in selected}
        selected += [n for n in ranked if (n["type"], n["id"]) not in keys]
        return selected[: self.options.get("k", 16)]

    def pack(self, user, candidates, graph):
        summary = self.state["user_profiles"].get(
            user, "No specific memory recorded for this user yet."
        )
        allowance = self.options.get("tau_tokens", 1800) - 300 - (200 if summary else 0) - 600
        snippets, packed, used = [], [], 0
        for n in sorted(self.neighbors(user, graph), key=lambda n: -n["score"]):
            if n["type"] == "item":
                item = self.context.catalog[self.original_ids[n["id"]]]
                snippet = f"[Item-{n['id']}] {str(item.get('title', ''))[:80]}"
                if item.get("description"):
                    snippet += f" | {str(item['description'])[:120]}"
                snippet += f" (score={n['score']:.3f})"
            else:
                recent = graph[self.original_users[n["id"]]][-3:]
                titles = ", ".join(
                    str(self.context.catalog[e.item_id].get("title", ""))[:40] for e in recent
                )
                snippet = f"[User-{n['id']}] (overlap_score={n['score']:.3f})" + (
                    " - Recent: " + titles if titles else ""
                )
            snippet = snippet[:200]
            if used + len(snippet) // 4 > allowance:
                break
            used += len(snippet) // 4
            snippets.append(snippet)
            packed.append(n)
        text = "**User Memory Summary:**\n" + summary[:200] + "\n**Collaborative Neighbors:**\n"
        text += "\n".join(f"{n}. {s}" for n, s in enumerate(snippets, 1)) or "(none available)"
        text += "\n**Candidates to Rank:**\n" + "\n".join(
            f"{n}. [{self.item_ids[i]}] {str(self.context.catalog[i].get('title', ''))[:60]}"
            for n, i in enumerate(candidates, 1)
        )
        return text, packed

    async def rank(self, user, instruction, candidates, graph, case_id, phase=""):
        packed_text, packed = self.pack(user, candidates, graph)
        facets = await self.ask(
            [
                {
                    "role": "user",
                    "content": f"Infer at most {self.options.get('n_facets', 7)} preference facets for User-{self.user_ids[user]} from this training memory and collaborative evidence. Return facets with confidence and supporting neighbor IDs, and support_edges (from, to, w).\n{packed_text}",
                }
            ],
            FACETS,
            case_id,
            phase + "retrieval",
            candidates,
        )
        item_text = "\n".join(
            f"[{self.item_ids[i]}] {self.state['item_descriptions'][i][:150]}" for i in candidates
        )
        result = await self.ask(
            [
                {
                    "role": "user",
                    "content": f"Rank every candidate exactly once for this instruction: {instruction}\nPreference facets: {facets}\nCandidate memories:\n{item_text}\nReturn scores with item_id, finite numeric score, and rationale.",
                }
            ],
            SCORES,
            case_id,
            phase + "rerank",
            candidates,
        )
        scores = result["scores"]
        if any(not math.isfinite(s["score"]) for s in scores):
            raise ValueError("non-finite ranking score")
        ordered = tuple(
            self.original_ids.get(s["item_id"], f"unknown:{s['item_id']}")
            for s in sorted(scores, key=lambda s: -s["score"])
        )
        prediction = RankingPrediction(
            case_id,
            ordered,
            {
                self.original_ids.get(s["item_id"], f"unknown:{s['item_id']}"): s["score"]
                for s in scores
            },
            diagnostics={
                "implementation": "native",
                "facets": facets,
                "neighbor_count": len(packed),
            },
        )
        return prediction, packed

    async def warmup(self):
        plans = []
        for user, events in self.histories.items():
            target = events[-1]
            pool = [i for i in self.context.catalog if i not in {e.item_id for e in events}]
            if len(pool) < 9:
                raise ValueError("MemRec warmup needs nine training negatives")
            rng = random.Random(self.context.config.get("seed", 42) + self.user_ids[user])
            candidates = rng.sample(pool, 9) + [target.item_id]
            rng.shuffle(candidates)
            plans.append((user, target, candidates))
        if all(target.timestamp for _, target, _ in plans):
            plans.sort(key=lambda p: (instant(p[1].timestamp), self.user_ids[p[0]]))
        self.state["warmup_manifest"] = [
            {"user_id": u, "target_event_id": e.event_id, "candidates": ids} for u, e, ids in plans
        ]
        for user, target, candidates in plans:
            if user in self.state["completed_warmup_users"]:
                continue
            cid = "warmup:" + target.event_id
            graph = self.graph(target.timestamp)
            prediction, packed = await self.rank(
                user,
                target.instruction or "Recommend a suitable item",
                candidates,
                graph,
                cid,
                "warmup/",
            )
            permutation(prediction.ordered_item_ids, candidates)
            allowed = {f"{n['type'].title()}-{n['id']}": n for n in packed}
            result = await self.ask(
                [
                    {
                        "role": "user",
                        "content": f"Update training memories using observed training feedback only. User-{self.user_ids[user]} selected Item-{self.item_ids[target.item_id]}; review: {target.review or ''}. Previous user memory: {self.state['user_profiles'].get(user, '')}. Previous item memory: {self.state['item_descriptions'][target.item_id]}. Ranking: {prediction.ordered_item_ids}. Allowed neighbors: {list(allowed)}. Return user_memory, item_memory, and at most {self.options.get('fanout_cap', 8)} neighbor_updates (neighbor_id, memory_update, rationale).",
                    }
                ],
                UPDATE,
                cid,
                "warmup/memory_update",
                candidates,
            )
            updates = result["neighbor_updates"]
            if len(updates) > self.options.get("fanout_cap", 8) or any(
                u["neighbor_id"] not in allowed for u in updates
            ):
                raise ValueError("memory update exceeds allowed training neighbors/fanout")
            if result["user_memory"].strip():
                self.state["user_profiles"][user] = result["user_memory"].strip()
            if result["item_memory"].strip():
                self.state["item_descriptions"][target.item_id] = result["item_memory"].strip()
            for update in updates:
                neighbor = allowed[update["neighbor_id"]]
                if update["memory_update"].strip():
                    table, original = (
                        ("user_profiles", self.original_users)
                        if neighbor["type"] == "user"
                        else ("item_descriptions", self.original_ids)
                    )
                    self.state[table][original[neighbor["id"]]] = update["memory_update"].strip()
            self.state["n_updates"] += 1
            self.state["completed_warmup_users"].append(user)

    async def predict(self, case):
        prediction, _ = await self.rank(
            case.user_id, case.instruction, case.candidates, self.graph(), case.case_id
        )
        return prediction
