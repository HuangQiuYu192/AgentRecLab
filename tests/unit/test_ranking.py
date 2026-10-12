import math

import pytest

from agentreclab.eval.aggregate import aggregate
from agentreclab.eval.ranking import at_rank


@pytest.mark.parametrize(
    "rank,hr1,hr3,ndcg,mrr", [(1, 1, 1, 1, 1), (2, 0, 1, 1 / math.log2(3), 0.5), (4, 0, 0, 0, 0.25)]
)
def test_golden(rank, hr1, hr3, ndcg, mrr):
    value = at_rank(rank)
    assert (value["HR@1"], value["HR@3"], value["NDCG@3"], value["MRR"]) == (hr1, hr3, ndcg, mrr)


def test_failure_denominators():
    result = aggregate([{"status": "succeeded", "metrics": at_rank(1)}, {"status": "failed"}])
    assert result["exclude"]["MRR"] == 1
    assert result["count_as_failure"]["MRR"] == 0.5
    assert result["success_rate"] == 0.5
