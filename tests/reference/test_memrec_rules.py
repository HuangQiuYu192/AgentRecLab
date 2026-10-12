"""Optional numeric parity audit of native domain rules against pinned source."""

import importlib.util
import os
import random

import pytest

from agentreclab.config import ROOT
from agentreclab.models.memrec import domain_score

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENTRECLAB_REFERENCE_TESTS") != "1",
    reason="optional reference audit; no upstream required",
)


@pytest.mark.parametrize(
    "domain,classname",
    [
        ("books", "BooksRules"),
        ("movietv", "MovieTVRules"),
        ("goodreads", "GoodReadsRules"),
        ("yelp", "YelpRules"),
    ],
)
def test_numeric_domain_rules(domain, classname):
    path = ROOT / f"third_party/MemRec/src/memory/domain_rules/{domain}_rules.py"
    if not path.exists():
        pytest.skip("optional reference checkout absent")
    spec = importlib.util.spec_from_file_location("reference_" + domain, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rule = getattr(module, classname)
    rng = random.Random(42)
    for _ in range(300):
        n = dict(
            type=rng.choice(["user", "item"]),
            score=rng.random(),
            metadata_overlap=rng.random(),
            co_count=rng.randrange(30),
            memory_sim=rng.random(),
            timestamp=rng.randrange(800),
        )
        features = dict(
            neighbor_type=n["type"],
            edge_weight=n["score"],
            metadata_overlap_score=n["metadata_overlap"],
            co_interaction_count=n["co_count"],
            memory_similarity_score=n["memory_sim"],
            recency_days=n["timestamp"],
        )
        assert domain_score(n, domain) == pytest.approx(rule.apply_rules(features), rel=1e-12)
