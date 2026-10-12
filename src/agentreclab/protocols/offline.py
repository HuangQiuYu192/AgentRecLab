def readiness(config, dataset):
    if config["protocol_mode"] == "paper_reproduction":
        raise ValueError(
            "strict paper_reproduction is not_ready: data equivalence, model snapshots and standalone differential results are unverified; use controlled_comparison with disclosed deviations"
        )
    return (
        "synthetic_mock"
        if dataset["provenance"].get("synthetic")
        else "controlled_upstream_wrapper_unverified"
    )
