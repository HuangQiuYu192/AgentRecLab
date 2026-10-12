from pathlib import Path

from agentreclab.core.schemas import Interaction, RankingCase
from agentreclab.data.manifest import DOMAINS, save_dataset
from agentreclab.state.snapshot import file_hash


def convert(dataset, config):
    """Read trusted original pickle; never silently resample test candidates."""
    import pandas as pd

    if dataset not in DOMAINS:
        raise ValueError(f"unsupported domain {dataset}")
    if not config.get("trusted_pickle"):
        raise ValueError(
            "pickle can execute code: set trusted_pickle only for trusted upstream data"
        )
    raw, mapping = Path(config["pickle"]), Path(config["mapping"])
    df, meta = pd.read_pickle(raw), pd.read_csv(mapping, dtype=str, keep_default_na=False)
    id_column = config.get("item_id_column", "index")
    catalog = {
        str(r[id_column]): {
            "title": str(r["title"]),
            "description": str(r.get("description", "")),
            "tags": [],
        }
        for _, r in meta.iterrows()
    }
    cases, events = [], []
    for ordinal, (index, row) in enumerate(df.iterrows()):
        namespace = f"instructrec:{dataset}"
        uid = (
            f"{namespace}:{row[config['user_id_column']]}"
            if config.get("user_id_column")
            else f"{namespace}:row:{ordinal}"
        )
        items = [str(i) for i in row["asin"]]
        if len(items) < 3:
            raise ValueError(f"row {ordinal} requires at least 3 interactions")
        history = []
        for pos, item in enumerate(items[:-1]):
            eid = f"{namespace}:{ordinal}:{pos}"
            history.append(eid)
            events.append(
                Interaction(
                    eid,
                    uid,
                    item,
                    None,
                    str(row["reviewText"][pos]),
                    metadata={"position": pos, "row_index": str(index)},
                )
            )
        if "ranked_lists" not in row:
            raise ValueError("original ranked_lists absent; explicit controlled sampling required")
        candidates = tuple(str(i) for i in row["ranked_lists"])
        negative_pool = [i for i in catalog if i not in items and i not in candidates]
        if not negative_pool:
            raise ValueError("no train warmup negative available")
        import random

        negative = random.Random(config.get("seed", 42) + ordinal).choice(negative_pool)
        cases.append(
            RankingCase(
                f"{namespace}:{ordinal}",
                dataset,
                uid,
                str(row["instruction"]),
                tuple(history),
                candidates,
                items[-1],
                metadata={
                    "original_row_index": str(index),
                    "row_ordinal": ordinal,
                    "target_event_id": f"{namespace}:{ordinal}:{len(items) - 1}",
                    "warmup_negative": negative,
                    "user_identity": "original"
                    if config.get("user_id_column")
                    else "row_namespace_original_id_unknown",
                },
            )
        )
    return save_dataset(
        config["out"],
        cases,
        events,
        catalog,
        {
            "raw_sources": {str(p): file_hash(p) for p in (raw, mapping)},
            "strict_time": False,
            "time_semantics": "sequence_only_original_timestamps_unknown",
            "candidate_source": "original",
            "seed": config.get("seed", 42),
            "filtering": "none",
            "user_id_column": config.get("user_id_column", "unknown"),
        },
    )
