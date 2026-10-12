import math

from agentreclab.data.validation import permutation


def at_rank(rank):
    if rank is not None and rank < 1:
        raise ValueError("rank is 1-based")
    result = {f"HR@{k}": float(rank is not None and rank <= k) for k in (1, 3, 5, 10)}
    result.update(
        {
            f"NDCG@{k}": 1 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0
            for k in (3, 10)
        }
    )
    result["MRR"] = 1 / rank if rank else 0.0
    return result


def evaluate_case(case, prediction):
    ids = permutation(prediction.ordered_item_ids, case.candidates)
    return at_rank(ids.index(case.positive_item_id) + 1)
