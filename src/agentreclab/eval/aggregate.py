from agentreclab.eval.ranking import at_rank


def aggregate(rows):
    success = [r["metrics"] for r in rows if r["status"] == "succeeded"]
    failed = len(rows) - len(success)
    zero = at_rank(None)
    return {
        "n_evaluated": len(success),
        "n_failed": failed,
        "success_rate": len(success) / len(rows) if rows else 0.0,
        "exclude": {
            k: sum(r[k] for r in success) / len(success) if success else None for k in zero
        },
        "count_as_failure": {
            k: sum(r[k] for r in success) / len(rows) if rows else None for k in zero
        },
    }
