import random
from collections import defaultdict


def user_bootstrap(rows, metric="MRR", n=1000, seed=42):
    groups = defaultdict(list)
    for r in rows:
        groups[r["user_id"]].append(r["metrics"][metric])
    if not groups:
        raise ValueError("no evaluated users")
    rng = random.Random(seed)
    users = list(groups)
    estimates = []
    for _ in range(n):
        sampled = [v for u in rng.choices(users, k=len(users)) for v in groups[u]]
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    return {
        "unit": "user",
        "metric": metric,
        "lower": estimates[int(n * 0.025)],
        "upper": estimates[min(n - 1, int(n * 0.975))],
        "seed": seed,
        "resamples": n,
    }
