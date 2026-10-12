from agentreclab.core.schemas import Interaction, RankingCase
from agentreclab.data.manifest import save_dataset


def make_fixture(out, n=10, dataset="books"):
    catalog = {
        str(i): {
            "title": f"Synthetic book {i}",
            "description": f"Fictional adventure {i}",
            "tags": ["fiction"],
        }
        for i in range(1, 61)
    }
    cases, events = [], []
    for j in range(n):
        uid = f"fixture:{dataset}:user:{j}"
        history = []
        for k in range(4):
            eid = f"{uid}:event:{k}"
            history.append(eid)
            events.append(
                Interaction(
                    eid,
                    uid,
                    str(20 + (j + k) % 20),
                    f"2020-01-0{k + 1}T00:00:00Z",
                    "Synthetic training review",
                )
            )
        candidates = tuple(str(1 + (j + k) % 15) for k in range(10))
        cases.append(
            RankingCase(
                f"fixture:{dataset}:case:{j}",
                dataset,
                uid,
                "Recommend a fictional adventure.",
                tuple(history),
                candidates,
                candidates[j % 10],
                "2020-02-01T00:00:00Z",
                {"target_event_id": f"{uid}:test", "warmup_negative": "60"},
            )
        )
    return save_dataset(
        out,
        cases,
        events,
        catalog,
        {
            "synthetic": True,
            "candidate_source": "synthetic_fixture",
            "strict_time": True,
            "seed": 42,
        },
    )
