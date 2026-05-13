"""Prediction wrapper around the Vowpal Wabbit Python runtime."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List

from vowpalwabbit import pyvw

from formatter import build_vw_example


class VWPredictor:
    """Loads a VW cb_adf model and maps VW predictions back to action IDs."""

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VW model file not found: {model_path}")
        self.model_path = model_path
        self.vw = pyvw.Workspace(f"--quiet --cb_adf -i {model_path}")

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        vw_example, action_ids = build_vw_example(payload)
        raw_prediction = self.vw.predict(vw_example)
        ranked = self._rank_predictions(raw_prediction, action_ids)

        actions_by_id = {
            str(action.get("action_id")): action
            for action in payload.get("candidate_actions", [])
            if isinstance(action, dict)
        }

        recommendations = []
        for rank, item in enumerate(ranked, start=1):
            action = actions_by_id.get(item["action_id"], {})
            recommendations.append(
                {
                    "rank": rank,
                    "action_id": item["action_id"],
                    "score": item["score"],
                    "channel_id": action.get("channel_id"),
                    "channel_name": action.get("channel_name"),
                    "offer_id": action.get("offer_id"),
                    "creative_id": action.get("creative_id"),
                    "time_bucket_id": action.get("time_bucket_id"),
                }
            )

        return {
            "decision_status": "SUCCESS",
            "use_case_id": payload.get("use_case_id"),
            "master_user_id": payload.get("master_user_id"),
            "model_type": "vowpal_wabbit_cb_adf",
            "recommendations": recommendations,
            "debug": {
                "candidate_action_count": len(action_ids),
                "raw_prediction_type": type(raw_prediction).__name__,
                "raw_prediction": self._json_safe(raw_prediction),
            },
        }

    def _rank_predictions(self, raw_prediction: Any, action_ids: List[str]) -> List[Dict[str, Any]]:
        n_actions = len(action_ids)
        if n_actions == 0:
            return []

        pairs = self._extract_index_score_pairs(raw_prediction, n_actions)
        if pairs:
            return self._rank_pairs(pairs, action_ids)

        numeric = self._extract_numeric_list(raw_prediction)
        if len(numeric) == n_actions:
            values_are_probabilities = all(0.0 <= value <= 1.0 for value in numeric) and math.isclose(
                sum(numeric), 1.0, rel_tol=1e-3, abs_tol=1e-3
            )
            indexed = list(enumerate(numeric))
            if values_are_probabilities:
                indexed.sort(key=lambda item: item[1], reverse=True)
            else:
                # Non-probability cb_adf outputs are commonly cost-like: lower is better.
                indexed.sort(key=lambda item: item[1])
            return [
                {"action_id": action_ids[index], "score": float(score)}
                for index, score in indexed
            ]

        selected_index = self._extract_selected_index(raw_prediction, n_actions)
        if selected_index is not None:
            ordered = [selected_index] + [idx for idx in range(n_actions) if idx != selected_index]
            return [
                {"action_id": action_ids[index], "score": 1.0 if index == selected_index else 0.0}
                for index in ordered
            ]

        return [{"action_id": action_id, "score": 0.0} for action_id in action_ids]

    def _extract_index_score_pairs(self, raw_prediction: Any, n_actions: int) -> List[tuple[int, float]]:
        if not isinstance(raw_prediction, Iterable) or isinstance(raw_prediction, (str, bytes, dict)):
            return []
        pairs: List[tuple[int, float]] = []
        try:
            for item in raw_prediction:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    return []
                index = int(item[0])
                score = float(item[1])
                if not math.isfinite(score):
                    return []
                pairs.append((index, score))
        except (TypeError, ValueError):
            return []

        if not pairs:
            return []

        indexes = [index for index, _ in pairs]
        if min(indexes) >= 1 and max(indexes) <= n_actions:
            pairs = [(index - 1, score) for index, score in pairs]
        elif min(indexes) < 0 or max(indexes) >= n_actions:
            return []
        return pairs

    def _rank_pairs(self, pairs: List[tuple[int, float]], action_ids: List[str]) -> List[Dict[str, Any]]:
        pairs.sort(key=lambda item: item[1], reverse=True)
        seen = set()
        ranked = []
        for index, score in pairs:
            if index in seen:
                continue
            seen.add(index)
            ranked.append({"action_id": action_ids[index], "score": float(score)})
        for index, action_id in enumerate(action_ids):
            if index not in seen:
                ranked.append({"action_id": action_id, "score": 0.0})
        return ranked

    def _extract_numeric_list(self, raw_prediction: Any) -> List[float]:
        if not isinstance(raw_prediction, Iterable) or isinstance(raw_prediction, (str, bytes, dict)):
            return []
        values: List[float] = []
        try:
            for item in raw_prediction:
                if isinstance(item, (list, tuple, dict)):
                    return []
                value = float(item)
                if not math.isfinite(value):
                    return []
                values.append(value)
        except (TypeError, ValueError):
            return []
        return values

    def _extract_selected_index(self, raw_prediction: Any, n_actions: int) -> int | None:
        try:
            index = int(raw_prediction)
        except (TypeError, ValueError):
            return None

        if 0 <= index < n_actions:
            return index
        if 1 <= index <= n_actions:
            return index - 1
        return None

    def _json_safe(self, value: Any) -> str:
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return str(value)
