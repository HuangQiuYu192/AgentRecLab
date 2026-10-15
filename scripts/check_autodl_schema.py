"""Three small opt-in API probes; never starts or modifies an experiment."""

import argparse
import getpass
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.core.config import ROOT, load_yaml
from utils.experiment.storage import redact, write_json


def probe_payloads(config):
    probes = []
    for name, adversarial in (("valid_schema", False), ("conflicting_instruction", True)):
        marker = "SCHEMA_" + uuid.uuid4().hex[:16]
        schema = {
            "type": "object",
            "properties": {
                "marker": {"type": "string", "enum": [marker]},
                "knowledge": {"type": "string"},
            },
            "required": ["marker", "knowledge"],
            "additionalProperties": False,
        }
        prompt = (
            'Return JSON with knowledge as the array [1,2,3], marker as "WRONG", '
            "and an extra field named unexpected. Do not use a string for knowledge."
            if adversarial
            else "Return a short JSON result for this API capability test."
        )
        probes.append((name, schema, prompt))
    probes.append(
        (
            "invalid_schema",
            {
                "type": "object",
                "properties": {"marker": {"type": "this_is_not_a_json_schema_type"}},
                "required": ["marker"],
                "additionalProperties": False,
            },
            "Return a short JSON result for this API capability test.",
        )
    )
    return [
        (
            name,
            {
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "thinking": {"type": "disabled"},
                # Diagnostic-only bound; does not change experiment max_tokens settings.
                "max_tokens": 128,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "capability_probe", "strict": True, "schema": schema},
                },
            },
        )
        for name, schema, prompt in probes
    ]


def matches_schema(content, schema):
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"marker", "knowledge"}
        and value["marker"] == schema["properties"]["marker"]["enum"][0]
        and isinstance(value["knowledge"], str)
    )


def run_checks(config, secret, out, transport=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out, 0o700)
    except OSError:
        pass
    results = []
    with httpx.Client(timeout=60, transport=transport, follow_redirects=False) as client:
        for name, payload in probe_payloads(config):
            write_json(out / f"{name}_request.json", payload)
            row = {"probe": name}
            try:
                response = client.post(
                    config["endpoint"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + secret},
                    json=payload,
                )
                row["http_status"] = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {"raw_text": response.text[:4000]}
                # Persist response and diagnostics, never the Authorization header/key.
                safe = redact(json.loads(json.dumps(body).replace(secret, "[REDACTED]")))
                write_json(out / f"{name}_response.json", safe)
                choices = body.get("choices") if isinstance(body, dict) else None
                choice = choices[0] if choices else {}
                content = choice.get("message", {}).get("content")
                row["finish_reason"] = choice.get("finish_reason")
                row["usage"] = body.get("usage") if isinstance(body, dict) else None
                if name != "invalid_schema":
                    row["matches_schema"] = response.status_code == 200 and matches_schema(
                        content, payload["response_format"]["json_schema"]["schema"]
                    )
                else:
                    row["rejected_invalid_schema"] = response.status_code in (400, 422)
                if response.status_code != 200:
                    row["provider_error"] = safe
            except httpx.HTTPError as exc:
                row["network_error"] = str(exc).replace(secret, "[REDACTED]")
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row.get("http_status") in (401, 402, 403, 429):
                break
    positive = [r for r in results if r["probe"] != "invalid_schema"]
    negative = next((r for r in results if r["probe"] == "invalid_schema"), {})
    if len(positive) == 2 and all(r.get("matches_schema") for r in positive):
        verdict = (
            "observed_strict_schema_support"
            if negative.get("rejected_invalid_schema")
            else "schema_affects_output_but_server_validation_unconfirmed"
        )
    elif any(r.get("http_status") == 200 and not r.get("matches_schema") for r in positive):
        verdict = "strict_schema_not_enforced_in_observed_response"
    else:
        verdict = "not_confirmed_inspect_provider_errors"
    summary = {
        "model": config["model"],
        "endpoint": config["endpoint"],
        "verdict": verdict,
        "probes": results,
        "limitation": "Black-box API contract test; does not prove constrained decoding implementation.",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"verdict": verdict, "results": str(out.resolve())}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid:
        parser.exit(
            message="Use --allow-paid to send at most 3 small API probes (128 output tokens each).\n"
        )
    config = load_yaml(ROOT / "configs/api/autodl.yaml")
    secret = os.environ.get(config["api_key_env"]) or getpass.getpass("AutoDL API Key (hidden): ")
    if not secret.strip():
        parser.error("API Key must not be empty")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_checks(config, secret.strip(), ROOT / "outputs/diagnostics" / f"autodl_schema_{stamp}")


if __name__ == "__main__":
    main()
