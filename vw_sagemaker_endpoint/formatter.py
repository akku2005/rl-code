"""VW cb_adf request formatter for SageMaker real-time inference."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple


EMBEDDING_RE = re.compile(r"^v(\d+)$")


def _clean_token(value: Any) -> str:
    """Make a value safe enough for a VW token without changing its meaning."""
    text = str(value).strip().replace(" ", "_")
    return text.replace("|", "_").replace(":", "_")


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _embedding_sort_key(item: Tuple[str, Any]) -> int:
    match = EMBEDDING_RE.match(item[0])
    return int(match.group(1)) if match else 10**9


def build_vw_example(payload: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Build a VW multiline cb_adf example and preserve candidate action order.

    Expected output shape:
      shared |ctx v0:0.123 v1:-0.44 |user city_tier=tier1 current_cibil:720
      |action aid=A001 ch=CH001 fr=FR001 dw=DW001 of=OF001 cr=CR001 tm=TM002 cost:0.39
    """
    context = payload.get("context") or {}
    candidate_actions = payload.get("candidate_actions") or []

    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")
    if not isinstance(candidate_actions, list) or not candidate_actions:
        raise ValueError("candidate_actions must be a non-empty list")

    embedding_items = [
        (key, value)
        for key, value in context.items()
        if EMBEDDING_RE.match(str(key)) and _is_number(value)
    ]
    embedding_tokens = [
        f"{key}:{float(value):.10g}"
        for key, value in sorted(embedding_items, key=_embedding_sort_key)
    ]

    user_tokens: List[str] = []
    for key, value in context.items():
        key = str(key)
        if EMBEDDING_RE.match(key) or value is None:
            continue
        feature_name = _clean_token(key)
        if _is_number(value):
            user_tokens.append(f"{feature_name}:{float(value):.10g}")
        else:
            user_tokens.append(f"{feature_name}={_clean_token(value)}")

    shared_parts = ["shared"]
    if embedding_tokens:
        shared_parts.append("|ctx " + " ".join(embedding_tokens))
    else:
        shared_parts.append("|ctx")
    if user_tokens:
        shared_parts.append("|user " + " ".join(user_tokens))
    else:
        shared_parts.append("|user")

    lines = [" ".join(shared_parts)]
    action_ids: List[str] = []

    for index, action in enumerate(candidate_actions):
        if not isinstance(action, dict):
            raise ValueError(f"candidate_actions[{index}] must be a JSON object")

        action_id = str(action.get("action_id") or f"ACTION_{index + 1}")
        action_ids.append(action_id)

        tokens = [
            f"aid={_clean_token(action_id)}",
            f"ch={_clean_token(action.get('channel_id', ''))}",
            f"fr={_clean_token(action.get('frequency_id', ''))}",
            f"dw={_clean_token(action.get('day_id', ''))}",
            f"of={_clean_token(action.get('offer_id', ''))}",
            f"cr={_clean_token(action.get('creative_id', ''))}",
            f"tm={_clean_token(action.get('time_bucket_id', ''))}",
        ]

        channel_cost = action.get("channel_cost")
        if _is_number(channel_cost):
            tokens.append(f"cost:{float(channel_cost):.10g}")

        lines.append("|action " + " ".join(tokens))

    return "\n".join(lines), action_ids
