from datetime import datetime, timezone

from agentreclab.core.errors import ValidationError


def instant(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValidationError("timestamps require a timezone")
    return value.astimezone(timezone.utc)


def validate(cases, interactions, catalog, candidate_count=10, strict_time=True):
    seen = set()
    events = {e.event_id: e for e in interactions}
    if len(events) != len(interactions):
        raise ValidationError("duplicate event_id")
    for e in interactions:
        if e.item_id not in catalog:
            raise ValidationError(f"missing catalog item {e.item_id}")
    for c in cases:
        if c.case_id in seen:
            raise ValidationError(f"duplicate case_id {c.case_id}")
        seen.add(c.case_id)
        if len(c.candidates) != candidate_count or len(set(c.candidates)) != candidate_count:
            raise ValidationError(f"{c.case_id}: candidate count/duplicates")
        if c.candidates.count(c.positive_item_id) != 1:
            raise ValidationError(f"{c.case_id}: positive must appear exactly once")
        if any(i not in catalog for i in c.candidates):
            raise ValidationError(f"{c.case_id}: missing catalog item")
        if strict_time and c.evaluation_time is None:
            raise ValidationError("strict time requires evaluation_time")
        for eid in c.history_event_ids:
            if eid not in events:
                raise ValidationError(f"unknown history event {eid}")
            e = events[eid]
            if e.user_id != c.user_id:
                raise ValidationError("history belongs to another user")
            if eid == c.metadata.get("target_event_id"):
                raise ValidationError("target event leaked into history")
            if strict_time and (
                e.timestamp is None or instant(e.timestamp) >= instant(c.evaluation_time)
            ):
                raise ValidationError("history is not strictly before evaluation")
        # Collaborative models can see every user's train graph, not just local history.
        if strict_time and any(
            e.timestamp is None or instant(e.timestamp) >= instant(c.evaluation_time)
            for e in interactions
        ):
            raise ValidationError("global training graph contains future interactions")
    return {"n_cases": len(cases), "n_interactions": len(events), "n_items": len(catalog)}


def permutation(ids, candidates, policy="fail"):
    ids = tuple(str(i) for i in ids)
    candidates = tuple(str(i) for i in candidates)
    if len(ids) == len(candidates) and set(ids) == set(candidates):
        return ids
    if policy == "append_missing":
        if any(i not in candidates for i in ids) or len(ids) != len(set(ids)):
            raise ValidationError("append_missing cannot repair duplicate/unknown IDs")
        return ids + tuple(i for i in candidates if i not in ids)
    raise ValidationError("prediction is not a strict candidate permutation")
