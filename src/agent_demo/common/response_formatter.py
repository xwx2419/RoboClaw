from __future__ import annotations

import json
import re
from typing import Any

JSON_OBJECT_PATTERN = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
JSON_ARRAY_PATTERN = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"


def _json_to_display_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj

    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)


def format_response_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    stripped_text = text.strip()
    try:
        return _json_to_display_text(json.loads(stripped_text))
    except (json.JSONDecodeError, ValueError):
        pass

    for pattern in (JSON_OBJECT_PATTERN, JSON_ARRAY_PATTERN):
        for match in re.finditer(pattern, text):
            json_str = match.group(0)
            try:
                formatted = _json_to_display_text(json.loads(json_str))
            except (json.JSONDecodeError, ValueError):
                continue
            text = text.replace(json_str, formatted, 1)

    return text
