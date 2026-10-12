import re

SENSITIVE = re.compile(r"api[_-]?key|authorization|secret|password|access[_-]?token", re.I)


def redact(value):
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if SENSITIVE.search(str(k)) and not str(k).endswith("_env")
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        value = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", value, flags=re.I)
        value = re.sub(
            r"(?i)(api[_-]?key|authorization|password|secret)\s*[:=]\s*[^\s,;]+",
            r"\1=[REDACTED]",
            value,
        )
        return value
    return value
