"""Shared schema and normalization for robot-side semantic annotations."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


ATOM_ALIASES = {
    "overhead_obstacle_clearance": "overhead_clearance",
    "overhead_obstacle_separation": "overhead_clearance",
    "overhead_clearance_management": "overhead_clearance",
    "obstacle_clearance": "overhead_clearance",
    "overhead_clearance": "overhead_clearance",
    "torso_clearance_management": "torso_clearance",
    "torso_obstacle_clearance": "torso_clearance",
    "torso_obstacle_separation": "torso_clearance",
    "torso_clearance": "torso_clearance",
    "collision_avoidance": "collision_avoidance",
    "collision_free_passage": "collision_avoidance",
    "obstacle_collision_avoidance": "collision_avoidance",
    "weight_transfer_control": "weight_transfer",
    "weight_shift": "weight_transfer",
    "load_transfer": "weight_transfer",
    "weight_transfer": "weight_transfer",
    "double_support_stabilization": "double_support_stabilization",
    "double_support_balance": "double_support_stabilization",
    "double_support_stability": "double_support_stabilization",
    "single_support_balance": "single_support_balance",
    "single_support_stability": "single_support_balance",
    "support_leg_loading": "support_loading",
    "support_chain_loading": "support_loading",
    "stance_loading": "support_loading",
    "support_load": "support_loading",
    "support_loading": "support_loading",
    "load_acceptance": "load_acceptance",
    "swing_leg_flexion": "swing_leg_flexion",
    "swing_flexion": "swing_leg_flexion",
    "deep_support_knee_flexion": "support_knee_flexion",
    "support_knee_flexion": "support_knee_flexion",
    "swing_leg_advancement": "swing_leg_advancement",
    "swing_advancement": "swing_leg_advancement",
    "forward_swing_progression": "swing_leg_advancement",
    "crouched_posture": "crouched_posture",
    "low_pelvis_posture": "crouched_posture",
    "low_body_posture": "crouched_posture",
    "compact_posture": "crouched_posture",
    "constrained_forward_progression": "forward_progression",
    "forward_progression": "forward_progression",
    "forward_advancement": "forward_progression",
    "push_off_preparation": "push_off_preparation",
    "push_off": "push_off_preparation",
    "foot_clearance_management": "foot_clearance",
    "swing_foot_clearance": "foot_clearance",
    "foot_clearance": "foot_clearance",
    "forward_torso_lean": "forward_torso_lean",
}

# These atoms are retained in the alias map for backward compatibility with
# existing JSON files, but are excluded from the active training ontology.
DISABLED_ATOMS = frozenset(
    {
        "forward_progression",
        "crouched_posture",
        "forward_torso_lean",
        "foot_clearance",
    }
)

CANONICAL_ATOMS = (
    "overhead_clearance",
    "torso_clearance",
    "collision_avoidance",
    "support_knee_flexion",
    "swing_leg_flexion",
    "swing_leg_advancement",
    "support_loading",
    "load_acceptance",
    "push_off_preparation",
    "single_support_balance",
    "double_support_stabilization",
    "weight_transfer",
)

CONTROL_OBJECTIVES = (
    "overhead_clearance_control",
    "torso_clearance_control",
    "posture_control",
    "foot_clearance_control",
    "forward_progression_control",
    "support_knee_flexion_control",
    "support_loading_control",
    "load_acceptance_control",
    "weight_transfer_control",
    "double_support_stability_control",
    "balance_control",
    "push_off_control",
    "swing_flexion_control",
    "swing_advancement_control",
    "collision_margin_control",
)

OBJECTIVE_ALIASES = {
    "overhead_clearance": "overhead_clearance_control",
    "overhead_clearance_control": "overhead_clearance_control",
    "torso_clearance": "torso_clearance_control",
    "torso_clearance_control": "torso_clearance_control",
    "posture": "posture_control",
    "posture_control": "posture_control",
    "foot_clearance": "foot_clearance_control",
    "foot_clearance_control": "foot_clearance_control",
    "forward_progression": "forward_progression_control",
    "forward_progression_control": "forward_progression_control",
    "support_knee_flexion": "support_knee_flexion_control",
    "support_knee_flexion_control": "support_knee_flexion_control",
    "support_loading": "support_loading_control",
    "support_loading_control": "support_loading_control",
    "load_acceptance": "load_acceptance_control",
    "load_acceptance_control": "load_acceptance_control",
    "weight_transfer": "weight_transfer_control",
    "weight_transfer_control": "weight_transfer_control",
    "double_support_stabilization": "double_support_stability_control",
    "double_support_stability_control": "double_support_stability_control",
    "single_support_balance": "balance_control",
    "balance": "balance_control",
    "balance_control": "balance_control",
    "push_off_preparation": "push_off_control",
    "push_off_control": "push_off_control",
    "swing_leg_flexion": "swing_flexion_control",
    "swing_flexion_control": "swing_flexion_control",
    "swing_leg_advancement": "swing_advancement_control",
    "swing_advancement_control": "swing_advancement_control",
    "collision_avoidance": "collision_margin_control",
    "collision_margin_control": "collision_margin_control",
}

TAG_FAMILIES = {
    "joint": (
        "joint_limits_satisfied",
        "joint_limit_risk",
        "joint_limits_unverified",
    ),
    "contact": (
        "contact_consistent",
        "contact_mismatch",
        "contact_unverified",
    ),
    "actuator": (
        "actuator_limits_satisfied",
        "actuator_load_risk",
        "actuator_feasibility_unverified",
    ),
    "collision": (
        "collision_free_verified",
        "collision_risk",
        "clearance_margin_limited",
        "clearance_unverified",
    ),
}

TAG_FALLBACKS = {
    "joint": "joint_limits_unverified",
    "contact": "contact_unverified",
    "actuator": "actuator_feasibility_unverified",
    "collision": "clearance_unverified",
}


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            if isinstance(child, bool):
                if child:
                    result.append(str(key))
            else:
                result.extend(_items(child))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for child in value:
            result.extend(_items(child))
        return result
    return [str(value)]


def _normalize_token(value: Any) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", token)


def canonical_atom(value: Any) -> str:
    token = _normalize_token(value)
    return ATOM_ALIASES.get(token, token)


def canonical_objective(value: Any) -> str | None:
    token = _normalize_token(value)
    if token in OBJECTIVE_ALIASES:
        return OBJECTIVE_ALIASES[token]

    # Compatibility for old free-form objective sentences. New annotations
    # should use the fixed objective vocabulary directly.
    text = str(value).strip().lower()
    if "torso" in text and "clearance" in text:
        return "torso_clearance_control"
    if "overhead" in text and "clearance" in text:
        return "overhead_clearance_control"
    if "weight" in text and "transfer" in text:
        return "weight_transfer_control"
    if "double" in text and ("support" in text or "stabil" in text):
        return "double_support_stability_control"
    if "load acceptance" in text:
        return "load_acceptance_control"
    if "support" in text and ("load" in text or "loading" in text):
        return "support_loading_control"
    if "balance" in text:
        return "balance_control"
    if "push" in text or "propuls" in text:
        return "push_off_control"
    if "swing" in text and ("flex" in text or "compact" in text):
        return "swing_flexion_control"
    if "swing" in text and ("advance" in text or "clearance" in text):
        return "swing_advancement_control"
    if "collision" in text or "clearance margin" in text:
        return "collision_margin_control"
    if "foot" in text and "clearance" in text:
        return "foot_clearance_control"
    if "forward" in text and ("progress" in text or "advance" in text):
        return "forward_progression_control"
    if "knee" in text and ("flex" in text or "bend" in text):
        return "support_knee_flexion_control"
    if "posture" in text or "pelvis" in text:
        return "posture_control"
    return None


def phase_known_absent_atoms(segment: Mapping[str, Any]) -> set[str]:
    """Return only high-confidence phase/contact-based absent atoms.

    These rules complement, rather than replace, metric-backed labels.  They
    are deliberately narrow: a flight phase cannot have a support leg, while
    a double-support phase cannot be single-support balance.  Swing-leg atoms
    are intentionally left undecided during flight because jumping motions
    can still contain substantial swing-leg flexion and advancement.
    """

    phase = _normalize_token(
        segment.get("gait_phase", segment.get("phase", ""))
    )
    human_metrics = segment.get("Human_Metrics", {})
    if not isinstance(human_metrics, Mapping):
        human_metrics = {}
    contact = human_metrics.get("contact_state", {})
    if not isinstance(contact, Mapping):
        contact = {}

    phase_type = _normalize_token(contact.get("phase_type", ""))
    transition = _normalize_token(contact.get("support_transition", ""))

    absent: set[str] = set()
    flight = (
        phase == "flight"
        or phase_type in {"flight", "flight_phase"}
        or transition in {"no_support", "flight"}
    )
    if flight:
        absent.update(
            {
                "support_knee_flexion",
                "support_loading",
                "load_acceptance",
                "push_off_preparation",
                "single_support_balance",
                "double_support_stabilization",
                "weight_transfer",
            }
        )

    double_support = (
        phase in {
            "prepared",
            "double_support",
            "double_support_right_to_left",
            "double_support_left_to_right",
            "landing",
        }
        or phase_type in {"stationary", "double_support", "landing"}
        or transition == "double_support"
    )
    if double_support and not flight:
        absent.add("single_support_balance")

    single_support = (
        phase in {"left_single_support", "right_single_support"}
        or phase_type == "single_support"
    )
    if single_support:
        absent.add("double_support_stabilization")

    take_off = phase in {"left_foot_take_off", "right_foot_take_off"}
    take_off_transition = (
        "to_toe_off" in transition
        or transition.endswith("toe_off")
        or "propulsion" in transition
    )
    if take_off or take_off_transition:
        absent.update(
            {
                "double_support_stabilization",
                "load_acceptance",
            }
        )

    stationary_prepared = (
        phase == "prepared"
        and phase_type in {"stationary", "prepared"}
    )
    if stationary_prepared:
        absent.update(
            {
                "push_off_preparation",
                "swing_leg_flexion",
                "swing_leg_advancement",
            }
        )

    return absent & set(CANONICAL_ATOMS)


def fixed_objective(value: Any) -> str | None:
    """Return an objective only when the input is already fixed-vocabulary.

    ``canonical_objective`` intentionally keeps compatibility with old
    free-form sentences. Training should not use those sentences as semantic
    supervision because they are often duplicated, vague, or hallucinated.
    """

    token = _normalize_token(value)
    if token in CONTROL_OBJECTIVES:
        return token
    return None


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize_evidence(
    value: Any,
    *,
    candidates: set[str],
) -> list[dict[str, str]]:
    """Keep only compact, auditable evidence references from an LLM response.

    Evidence is intentionally metadata rather than an additional training
    target. It helps reviewers trace a positive label back to a metric path
    without trusting free-form model explanations.
    """

    if not isinstance(value, (list, tuple)):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        atom = canonical_atom(item.get("atom", ""))
        source = str(item.get("source", "")).strip()
        observed = str(
            item.get("value", item.get("observation", item.get("claim", "")))
        ).strip()
        if atom not in candidates or not source or not observed:
            continue
        result.append(
            {
                "atom": atom,
                "source": source[:160],
                "value": observed[:320],
            }
        )
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for item in result:
        key = (item["atom"], item["source"], item["value"])
        unique[key] = item
    return list(unique.values())[:64]


def _select_one_tag(tags: set[str], family: str) -> str:
    candidates = [tag for tag in TAG_FAMILIES[family] if tag in tags]
    if not candidates:
        return TAG_FALLBACKS[family]
    # Risk/mismatch takes precedence over a satisfied tag if the model
    # accidentally emits conflicting labels for the same family.
    priority = {
        "joint_limit_risk": 0,
        "contact_mismatch": 0,
        "actuator_load_risk": 0,
        "collision_risk": 0,
        "clearance_margin_limited": 1,
        "joint_limits_satisfied": 2,
        "contact_consistent": 2,
        "actuator_limits_satisfied": 2,
        "collision_free_verified": 2,
        "joint_limits_unverified": 3,
        "contact_unverified": 3,
        "actuator_feasibility_unverified": 3,
        "clearance_unverified": 3,
    }
    return min(candidates, key=lambda item: priority.get(item, 99))


def normalize_robot_semantics(
    parsed: Mapping[str, Any],
    *,
    candidate_atoms: Iterable[str],
) -> dict[str, list[str]]:
    """Normalize one LLM response into the training annotation contract.

    ``skill_atoms``, ``unknown_atoms`` and ``known_absent_atoms`` form a
    disjoint three-way partition of the action-specific candidate vocabulary.
    Unknown is intentionally the default for omitted candidates; omission is
    never silently converted into a negative label.
    """

    candidates = {canonical_atom(item) for item in candidate_atoms}
    positive = {
        canonical_atom(item)
        for item in _items(parsed.get("skill_atoms"))
    } & candidates
    unknown = {
        canonical_atom(item)
        for item in _items(parsed.get("unknown_atoms"))
    } & candidates
    absent = {
        canonical_atom(item)
        for item in _items(parsed.get("known_absent_atoms"))
    } & candidates

    unknown -= positive
    absent -= positive
    absent -= unknown
    unknown |= candidates.difference(positive | absent | unknown)

    objectives = _unique(
        objective
        for item in _items(
            parsed.get("robot_control_objectives")
            or parsed.get("control_objectives")
        )
        if (objective := canonical_objective(item)) is not None
    )
    if not positive:
        objectives = []

    raw_tags = {
        _normalize_token(item)
        for item in _items(parsed.get("feasibility_tags"))
    }
    tags = [
        _select_one_tag(raw_tags, family)
        for family in ("joint", "contact", "actuator", "collision")
    ]

    result = {
        "skill_atoms": sorted(positive),
        "unknown_atoms": sorted(unknown),
        "known_absent_atoms": sorted(absent),
        "robot_control_objectives": objectives,
        "feasibility_tags": tags,
    }
    evidence = _normalize_evidence(parsed.get("evidence"), candidates=candidates)
    if evidence:
        result["evidence"] = evidence
    return result
