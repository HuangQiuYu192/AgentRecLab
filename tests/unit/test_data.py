from dataclasses import replace

import pytest

from agentreclab.data.fixtures import make_fixture
from agentreclab.data.freeze import freeze
from agentreclab.data.manifest import load_dataset
from agentreclab.data.validation import permutation, validate


@pytest.fixture
def dataset(tmp_path):
    return load_dataset(make_fixture(tmp_path / "fixture"))


@pytest.mark.parametrize(
    "change",
    ["duplicate", "positive", "catalog", "future", "target", "wrong_user", "global_future"],
)
def test_reject_leakage_and_malformed(dataset, change):
    cases, events, catalog, _ = dataset
    c = cases[0]
    if change == "duplicate":
        c = replace(c, candidates=(c.candidates[0],) * 10)
    elif change == "positive":
        c = replace(c, positive_item_id="60")
    elif change == "catalog":
        catalog.pop(c.candidates[0])
    elif change == "future":
        events = (replace(events[0], timestamp="2030-01-01T00:00:00Z"), *events[1:])
    elif change == "target":
        c = replace(c, metadata={"target_event_id": c.history_event_ids[0]})
    elif change == "wrong_user":
        events = (replace(events[0], user_id="other"), *events[1:])
    elif change == "global_future":
        events = (*events[:-1], replace(events[-1], timestamp="2030-01-01T00:00:00Z"))
    with pytest.raises(ValueError):
        validate((c, *cases[1:]), events, catalog)


def test_freeze_roundtrip(tmp_path):
    source = make_fixture(tmp_path / "fixture")
    out = freeze(source, tmp_path / "frozen.json", limit=5)
    original = load_dataset(source)[0]
    assert load_dataset(out)[0] == original[:5]
    target = source.parent / "cases.parquet"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(out)


@pytest.mark.parametrize("values", [["1", "1"], ["1", "99"], []])
def test_invalid_prediction(values):
    with pytest.raises(ValueError):
        permutation(values, ("1", "2"))


def test_append_missing():
    assert permutation(["2"], ("1", "2", "3"), "append_missing") == ("2", "1", "3")
    with pytest.raises(ValueError):
        permutation(["2", "2"], ("1", "2", "3"), "append_missing")
