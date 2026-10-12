import pandas as pd
import pytest

from utils.data import convert, load_dataset


def test_sample_original_users_preserves_case_identity_and_candidates(tmp_path):
    raw, mapping = tmp_path / "books.pkl", tmp_path / "mapping.csv"
    order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 10]
    pd.DataFrame(
        [
            {
                "reviewerID": f"user-{n}",
                "asin": [20, 21, 22, 3],
                "reviewText": ["a", "b", "c", "TEST_FEEDBACK"],
                "instruction": "Read a novel",
                "ranked_lists": order,
            }
            for n in range(7)
        ],
        index=range(70, 77),
    ).to_pickle(raw)
    pd.DataFrame(
        [{"index": n, "title": f"book {n}", "description": "text"} for n in range(1, 201)]
    ).to_csv(mapping, index=False)
    config = {
        "pickle": str(raw),
        "mapping": str(mapping),
        "trusted_pickle": True,
        "user_id_column": "reviewerID",
        "max_users": 3,
        "seed": 42,
        "out": str(tmp_path / "sample"),
    }
    cases, events, catalog, manifest = load_dataset(convert("books", config))
    assert len({c.user_id for c in cases}) == 3
    assert len(cases) == 3
    assert "200" not in catalog
    assert manifest["provenance"]["max_users"] == 3
    for case in cases:
        ordinal = case.metadata["row_ordinal"]
        assert case.case_id == f"instructrec:books:{ordinal}"
        assert case.user_id == f"instructrec:books:user-{ordinal}"
        assert case.metadata["original_row_index"] == str(70 + ordinal)
        assert case.candidates == tuple(map(str, order))
    assert "TEST_FEEDBACK" not in str(events)
    repeated, _, _, _ = load_dataset(convert("books", {**config, "out": str(tmp_path / "again")}))
    assert cases == repeated
    with pytest.raises(ValueError, match="user_id_column"):
        convert("books", {**config, "user_id_column": None})
