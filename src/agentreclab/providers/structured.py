def validate_schema(value, schema):
    """Validate the small JSON-schema subset used by audited upstream prompts."""
    kind = schema.get("type")
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
    }
    if kind in types and (
        not isinstance(value, types[kind])
        or (kind in ("number", "integer") and isinstance(value, bool))
    ):
        raise ValueError(f"structured response requires {kind}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            raise ValueError("structured response missing required fields")
        if schema.get("additionalProperties") is False and any(k not in properties for k in value):
            raise ValueError("structured response has unexpected fields")
        for key, spec in properties.items():
            if key in value:
                validate_schema(value[key], spec)
    if isinstance(value, list):
        for item in value:
            validate_schema(item, schema.get("items", {}))
