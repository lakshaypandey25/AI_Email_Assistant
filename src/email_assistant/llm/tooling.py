"""Convert Pydantic models to provider-friendly function-calling schemas."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def _simplify(node: Any, defs: dict[str, Any]) -> Any:
    """Inline ``$ref`` pointers and collapse ``anyOf: [X, null]`` into ``X``."""
    if isinstance(node, list):
        return [_simplify(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        name = node["$ref"].split("/")[-1]
        resolved = _simplify(copy.deepcopy(defs[name]), defs)
        extras = {k: _simplify(v, defs) for k, v in node.items() if k != "$ref"}
        return {**resolved, **extras}

    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        non_null = [opt for opt in any_of if not (isinstance(opt, dict) and opt.get("type") == "null")]
        if len(non_null) == 1 and len(any_of) == 2:
            merged = {k: v for k, v in node.items() if k != "anyOf"}
            inner = _simplify({**non_null[0], **merged}, defs)
            if isinstance(inner.get("type"), str):
                inner["type"] = [inner["type"], "null"]  # keep the field nullable: models often send null
            return inner

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title":  # schema-level title (noise); NOT a property called "title"
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {name: _simplify(sub, defs) for name, sub in value.items()}
        else:
            result[key] = _simplify(value, defs)
    return result


def pydantic_to_tool(model: type[BaseModel], name: str, description: str) -> dict[str, Any]:
    """Build an OpenAI-compatible ``tools`` entry from a Pydantic model."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    parameters = _simplify(schema, defs)
    parameters.pop("description", None)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }
