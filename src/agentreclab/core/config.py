from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path):
    result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("config must be a YAML mapping")
    return result


def load_run(path, provider=None, model=None):
    config = load_yaml(path)
    config["provider"] = load_yaml(
        ROOT / "configs" / "providers" / f"{provider or config.get('provider', 'mock')}.yaml"
    )
    config["model"] = model or config.get("model", "iagent")
    config["model_config"] = load_yaml(ROOT / "configs" / "models" / f"{config['model']}.yaml")
    mode = config.get("protocol_mode", "controlled_comparison")
    if mode not in ("controlled_comparison", "paper_reproduction"):
        raise ValueError("unknown protocol mode")
    config["protocol_mode"] = mode
    if config.get("invalid_prediction_policy", "fail") not in (
        "fail",
        "one_retry",
        "append_missing",
    ):
        raise ValueError("unknown prediction policy")
    if int(config.get("concurrency", 1)) < 1:
        raise ValueError("concurrency must be positive")
    return config
