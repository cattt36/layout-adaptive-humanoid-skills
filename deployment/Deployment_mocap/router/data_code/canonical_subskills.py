"""Load and validate the frozen canonical-subskill evaluation ontology.

The ontology is deliberately separate from annotation JSON files and model
inputs.  It provides stable, action-specific relevance labels for offline
retrieval evaluation without copying evaluation targets into latent training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


CANONICAL_SUBSKILL_SCHEMA_PATH = Path(__file__).with_name(
    "canonical_subskills_v1.json"
)


class CanonicalSubskillError(ValueError):
    """Raised when the frozen subskill ontology cannot label an observation."""


def _load_schema() -> tuple[dict[str, Any], str]:
    raw = CANONICAL_SUBSKILL_SCHEMA_PATH.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != "canonical_subskills_v1":
        raise CanonicalSubskillError(
            f"Unexpected canonical-subskill schema version in "
            f"{CANONICAL_SUBSKILL_SCHEMA_PATH}"
        )
    if payload.get("evaluation_only") is not True:
        raise CanonicalSubskillError(
            "Canonical subskills must remain evaluation-only"
        )
    if payload.get("label_scope") != "within_action":
        raise CanonicalSubskillError(
            "Canonical subskills must use action-scoped labels"
        )
    if payload.get("unmapped_phase_policy") != "error":
        raise CanonicalSubskillError(
            "Unmapped gait phases must fail instead of receiving fallback labels"
        )
    actions = payload.get("actions")
    if not isinstance(actions, Mapping) or not actions:
        raise CanonicalSubskillError("Canonical-subskill schema has no actions")

    seen_ids: set[str] = set()
    for action, action_spec in actions.items():
        if not isinstance(action_spec, Mapping):
            raise CanonicalSubskillError(f"Invalid action record: {action}")
        namespace = str(action_spec.get("namespace", "")).strip()
        phase_map = action_spec.get("phase_map")
        if not namespace or not isinstance(phase_map, Mapping) or not phase_map:
            raise CanonicalSubskillError(f"Incomplete action ontology: {action}")
        for phase, record in phase_map.items():
            if not str(phase).strip() or not isinstance(record, Mapping):
                raise CanonicalSubskillError(f"Invalid phase record: {action}/{phase}")
            subskill_id = str(record.get("id", "")).strip()
            if not subskill_id.startswith(namespace + "_"):
                raise CanonicalSubskillError(
                    f"Subskill {subskill_id!r} is outside namespace {namespace!r}"
                )
            if subskill_id in seen_ids:
                raise CanonicalSubskillError(
                    f"Canonical subskill ID is not globally unique: {subskill_id}"
                )
            seen_ids.add(subskill_id)
    return payload, hashlib.sha256(raw).hexdigest()


CANONICAL_SUBSKILL_SCHEMA, CANONICAL_SUBSKILL_SCHEMA_SHA256 = _load_schema()
CANONICAL_SUBSKILL_SCHEMA_VERSION = str(
    CANONICAL_SUBSKILL_SCHEMA["schema_version"]
)
CANONICAL_SUBSKILL_ACTIONS = tuple(CANONICAL_SUBSKILL_SCHEMA["actions"])


def resolve_action(action_or_trial_id: str) -> str:
    """Resolve an exact action name or an action-prefixed trial identifier."""

    value = str(action_or_trial_id).strip()
    if value in CANONICAL_SUBSKILL_ACTIONS:
        return value
    matches = [
        action
        for action in CANONICAL_SUBSKILL_ACTIONS
        if value.startswith(action + "_")
    ]
    if len(matches) == 1:
        return matches[0]
    raise CanonicalSubskillError(
        f"Cannot resolve a mapped action from {action_or_trial_id!r}"
    )


def canonical_subskill_record(
    action_or_trial_id: str, gait_phase: str
) -> dict[str, str]:
    """Return the frozen record for one action-specific gait phase."""

    action = resolve_action(action_or_trial_id)
    phase = str(gait_phase).strip().lower()
    action_spec = CANONICAL_SUBSKILL_SCHEMA["actions"][action]
    record = action_spec["phase_map"].get(phase)
    if not isinstance(record, Mapping):
        raise CanonicalSubskillError(
            f"Unmapped gait phase {phase!r} for action {action!r}"
        )
    return {str(key): str(value) for key, value in record.items()}


def canonical_subskill_id(action_or_trial_id: str, gait_phase: str) -> str:
    """Return the stable, action-specific retrieval relevance label."""

    return canonical_subskill_record(action_or_trial_id, gait_phase)["id"]


def validate_observed_phases(
    action_or_trial_id: str, observed_phases: Iterable[str]
) -> None:
    """Fail if any observed phase is absent from the frozen ontology."""

    action = resolve_action(action_or_trial_id)
    for phase in sorted({str(value).strip().lower() for value in observed_phases}):
        canonical_subskill_id(action, phase)
