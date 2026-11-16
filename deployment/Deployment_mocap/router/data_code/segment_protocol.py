"""Shared segment-query and evaluation-label protocol.

This module intentionally contains only annotation-derived protocol helpers.  The
canonical label is used *after* latent training for a profile-independent
retrieval relevance mask; it is never fed to a latent model as an input.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from human_semantic_schema import normalize_existing_human_semantics
from robot_semantic_schema import canonical_atom

try:
    from canonical_subskills import (
        CANONICAL_SUBSKILL_SCHEMA_SHA256,
        CANONICAL_SUBSKILL_SCHEMA_VERSION,
        canonical_subskill_id,
    )
except ModuleNotFoundError:  # Supports direct import from the MOE Framework copy.
    _DATA_ROOT = Path(__file__).resolve().parents[1] / "Data"
    if str(_DATA_ROOT) not in sys.path:
        sys.path.insert(0, str(_DATA_ROOT))
    from canonical_subskills import (
        CANONICAL_SUBSKILL_SCHEMA_SHA256,
        CANONICAL_SUBSKILL_SCHEMA_VERSION,
        canonical_subskill_id,
    )

# v3 removes human/robot annotation text from the model-side query. Those
# annotations remain valid offline evaluation labels, but they are not
# observable by the deployed robot and therefore must never be query inputs.
SEGMENT_QUERY_VERSION = "segment_query_v3_observable"
VALID_HUMAN_STATES = {"metric_backed", "phase_contact_backed"}


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, child in value.items():
            if isinstance(child, bool):
                if child:
                    out.append(str(key))
            else:
                out.extend(_items(child))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for child in value:
            out.extend(_items(child))
        return out
    return [str(value)]


def _active_atoms(values: Any) -> set[str]:
    return {
        canonical_atom(item)
        for item in _items(values)
        if canonical_atom(item)
    }


def canonical_segment_atoms(segment: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a profile-independent, conservative segment-event signature.

    Robot semantic atoms are taken as explicit labels.  Human atoms are added
    only after the shared evidence-aware normalizer accepts them.  No profile
    flag is consulted, so every Z0--Z4 retrieval evaluation uses the same
    relevance definition.
    """
    atoms: set[str] = set()
    robot_metrics = segment.get("Retargeted_Robot_Metrics", {})
    if isinstance(robot_metrics, Mapping):
        robot_sem = robot_metrics.get("robot_semantics", {})
        if isinstance(robot_sem, Mapping):
            atoms.update(_active_atoms(robot_sem.get("skill_atoms")))

    human_metrics = segment.get("Human_Metrics", {})
    if isinstance(human_metrics, Mapping):
        human_sem = human_metrics.get("human_semantics", {})
        if isinstance(human_sem, Mapping):
            normalized = normalize_existing_human_semantics(human_sem, segment)
            if str(normalized.get("annotation_state", "")) in VALID_HUMAN_STATES:
                atoms.update(_active_atoms(normalized.get("skill_atoms")))
    return tuple(sorted(atoms))


def strict_segment_evidence_label(segment: Mapping[str, Any]) -> str:
    """Build the former exact phase-plus-event label for diagnostics only."""

    phase = str(segment.get("gait_phase", "")).strip().lower()
    atoms = canonical_segment_atoms(segment)
    atom_text = ",".join(atoms) if atoms else "none"
    return f"phase={phase}|events={atom_text}"


def canonical_segment_label(
    segment: Mapping[str, Any], *, action: str | None = None
) -> str:
    """Build the frozen action-specific subskill label used for retrieval.

    ``action`` is required because ThreeStepPoleRun and FiveStepPoleRun intentionally
    use separate namespaces despite sharing similarly named gait phases.
    """

    if not action:
        raise ValueError("canonical_segment_label requires an action or trial ID")
    phase = str(segment.get("gait_phase", "")).strip().lower()
    return canonical_subskill_id(action, phase)
