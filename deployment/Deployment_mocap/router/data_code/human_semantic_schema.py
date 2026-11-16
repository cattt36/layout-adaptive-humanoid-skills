"""Conservative schema and normalization for human semantic annotations.

Human annotations are weak supervision for Z4, not ground-truth robot
commands.  A positive atom is therefore kept only when it can be traced to
concrete ``Human_Metrics`` evidence.  Omitted atoms remain unknown unless a
small deterministic phase/contact rule can prove that they are absent.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping

from robot_semantic_schema import CANONICAL_ATOMS, canonical_atom, phase_known_absent_atoms


HUMAN_SEMANTICS_SCHEMA_VERSION = "human_semantics_v2"
HUMAN_EVIDENCE_ROOT = "Human_Metrics."
MIN_FLEXION_ROM_DEG = 5.0
POSTURAL_BALANCE_TOKENS = (
    "trunk",
    "torso",
    "pelvis",
    "center_of_mass",
    "com",
    "posture",
    "postural",
    "balance",
    "stability",
)


def _token(value: Any) -> str:
    return re.sub(
        r"_+",
        "_",
        str(value).strip().lower().replace("-", "_").replace(" ", "_"),
    )


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
        result: list[str] = []
        for child in value:
            result.extend(_items(child))
        return result
    return [str(value)]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_source(segment: Mapping[str, Any], source: str) -> Any:
    """Resolve an evidence path such as ``Human_Metrics.kinematics.foo``."""

    if not source.startswith(HUMAN_EVIDENCE_ROOT):
        return None
    current: Any = segment
    for part in source.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _source_family(source: str) -> str:
    parts = source.split(".")
    return parts[1] if len(parts) > 1 else ""


def _has_single_support_contact(segment: Mapping[str, Any]) -> bool:
    """Return whether contact metadata identifies a single-support phase."""

    phase = _token(segment.get("gait_phase", ""))
    human = segment.get("Human_Metrics", {})
    contact = human.get("contact_state", {}) if isinstance(human, Mapping) else {}
    if not isinstance(contact, Mapping):
        contact = {}
    phase_type = _token(contact.get("phase_type", ""))
    transition = _token(contact.get("support_transition", ""))
    return bool(
        phase in {"left_single_support", "right_single_support"}
        or phase in {"left_foot_take_off", "right_foot_take_off"}
        or phase_type == "single_support"
        or "single_support" in phase_type
        or "support_to_toe_off" in phase_type
        or "single_support" in transition
        or "propulsion" in transition
    )


def _has_structured_contact_state(segment: Mapping[str, Any]) -> bool:
    """Return whether contact metadata contains an explicit phase fact."""

    human = segment.get("Human_Metrics", {})
    contact = human.get("contact_state", {}) if isinstance(human, Mapping) else {}
    if not isinstance(contact, Mapping):
        return False
    return any(
        str(contact.get(key, "")).strip()
        for key in ("phase_type", "support_transition", "incoming_event", "outgoing_event")
    )


def _json_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, bool, int, float)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _is_concrete_leaf(value: Any) -> bool:
    """Return whether a value can be cited as metric evidence."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return isinstance(value, (bool, int, float, str))


def human_metric_paths(segment: Mapping[str, Any]) -> list[str]:
    """List concrete Human_Metrics leaves that may be cited as evidence."""

    human = segment.get("Human_Metrics", {})
    if not isinstance(human, Mapping):
        return []
    result: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) == "human_semantics":
                    continue
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}")
        elif _is_concrete_leaf(value):
            result.append(path)

    for namespace in (
        "contact_state",
        "step_info",
        "obstacle_relation",
        "kinematics",
        "dynamics",
    ):
        if namespace in human:
            walk(human[namespace], f"Human_Metrics.{namespace}")
    return sorted(result)


def human_metric_evidence(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten available human metric leaves for exact prompt copying."""

    human = segment.get("Human_Metrics", {})
    if not isinstance(human, Mapping):
        return {}
    result: dict[str, Any] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}")
        elif _is_concrete_leaf(value):
            result[path] = value

    for namespace in (
        "contact_state",
        "step_info",
        "obstacle_relation",
        "kinematics",
        "dynamics",
    ):
        if namespace in human:
            walk(human[namespace], f"Human_Metrics.{namespace}")
    return result


# Each tuple is a set of evidence families.  One source from every tuple is
# required for a positive label.  This prevents phase names or task goals from
# becoming fake metric evidence.
ATOM_EVIDENCE_FAMILIES: dict[str, tuple[tuple[str, ...], ...]] = {
    "overhead_clearance": (("obstacle_relation",),),
    "torso_clearance": (("obstacle_relation",),),
    "collision_avoidance": (("obstacle_relation",),),
    # Moments/forces can indicate loading, but cannot by themselves prove
    # that a knee or swing leg is flexed. Flexion atoms require kinematics.
    "support_knee_flexion": (("kinematics",),),
    "swing_leg_flexion": (("kinematics",),),
    "swing_leg_advancement": (("kinematics",),),
    "support_loading": (("contact_state",), ("dynamics",)),
    "load_acceptance": (("contact_state",), ("dynamics",)),
    "push_off_preparation": (("contact_state",), ("dynamics",)),
    # Step indices identify a step, but do not prove balance. Require a
    # mechanical or postural measurement in addition to contact state.
    "single_support_balance": (("contact_state",), ("kinematics", "dynamics")),
    "double_support_stabilization": (("contact_state",),),
    "weight_transfer": (("contact_state",), ("dynamics",)),
}


def _evidence_is_valid(
    atom: str,
    evidence: list[dict[str, str]],
    segment: Mapping[str, Any],
) -> bool:
    families = ATOM_EVIDENCE_FAMILIES.get(atom, ())
    observed_families = {
        _source_family(item["source"])
        for item in evidence
        if _resolve_source(segment, item["source"]) is not None
    }
    for group in families:
        if set(group) & observed_families:
            continue
        # Contact state is a structured annotation on the segment. It need
        # not be duplicated in the LLM evidence list when the required
        # metric evidence is present, but it must contain an explicit phase
        # or support transition.
        if "contact_state" in group and _has_structured_contact_state(segment):
            continue
        return False

    sources = [item["source"].lower() for item in evidence]
    token_rules = {
        "overhead_clearance": ("obstacle", "clearance", "distance"),
        "torso_clearance": ("obstacle", "clearance", "distance"),
        "collision_avoidance": ("obstacle", "collision", "proximity"),
        "support_knee_flexion": ("knee",),
        "swing_leg_flexion": ("swing", "knee", "hip", "ankle"),
        "swing_leg_advancement": (
            "advancement",
            "progression",
            "trajectory",
            "velocity",
            "position",
            "displacement",
            "step_length",
        ),
        "support_loading": ("support", "wrench", "force", "moment"),
        # A pure joint moment does not prove that external load was accepted.
        "load_acceptance": (
            "accept",
            "contact_force",
            "force",
            "wrench",
            "impulse",
            "load",
        ),
        "push_off_preparation": ("support", "propulsion", "moment", "force"),
        "single_support_balance": POSTURAL_BALANCE_TOKENS,
        "weight_transfer": ("support", "transition", "force", "moment"),
    }
    required_tokens = token_rules.get(atom)
    if not required_tokens:
        return True
    # Token checks are a secondary guard.  Some datasets use generic metric
    # names, so the evidence-family requirement above remains authoritative.
    if atom in {
        "support_knee_flexion",
        "swing_leg_flexion",
        "swing_leg_advancement",
        "single_support_balance",
    }:
        if atom == "single_support_balance":
            # A step index, a knee moment, or a generic "single support"
            # phase does not prove balance. Require both contact metadata and
            # an explicit postural/COM kinematic measurement.
            if not _has_single_support_contact(segment):
                return False
            return any(
                _source_family(item["source"]) == "kinematics"
                and any(token in item["source"].lower() for token in POSTURAL_BALANCE_TOKENS)
                for item in evidence
            )

        valid = any(
            any(token in source for token in required_tokens)
            for source in sources
        )
        if not valid:
            return False

        # The human metric generator stores kinematics as ROM summaries.
        # Require a non-trivial knee ROM for flexion labels; a generic
        # contact/step field must never be sufficient by itself.
        if atom in {"support_knee_flexion", "swing_leg_flexion"}:
            for item in evidence:
                source = item["source"].lower()
                if "kinematics" not in source:
                    continue
                if atom == "support_knee_flexion":
                    if "support" not in source or "knee" not in source:
                        continue
                elif not (
                    "swing" in source
                    and any(token in source for token in ("knee", "hip", "ankle"))
                ):
                    continue
                value = _resolve_source(segment, item["source"])
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    if abs(float(value)) >= MIN_FLEXION_ROM_DEG:
                        return True
            return False
        return True
    return any(
        any(token in source for token in required_tokens)
        for source in sources
    )


def _normalized_evidence(
    parsed: Mapping[str, Any],
    segment: Mapping[str, Any],
    candidates: set[str],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    raw = parsed.get("evidence", [])
    if not isinstance(raw, (list, tuple)):
        return result
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        atom = canonical_atom(item.get("atom", ""))
        source = str(item.get("source", "")).strip()
        if atom not in candidates or not source.startswith(HUMAN_EVIDENCE_ROOT):
            continue
        resolved = _resolve_source(segment, source)
        if resolved is None:
            continue
        if isinstance(resolved, (Mapping, list, tuple)):
            # Evidence must point to a concrete leaf, never an entire
            # namespace or an empty container.
            if not resolved:
                continue
            continue
        if not _is_concrete_leaf(resolved):
            continue
        # Do not trust an LLM-supplied value. Re-read the exact value from the
        # source JSON so evidence cannot contain a fabricated observation.
        observed = _json_value(resolved)
        result.append(
            {
                "atom": atom,
                "source": source[:200],
                "value": str(observed)[:320],
            }
        )
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for item in result:
        unique[(item["atom"], item["source"], item["value"])] = item
    return list(unique.values())[:64]


def _deterministic_positive(
    segment: Mapping[str, Any],
) -> tuple[set[str], list[dict[str, str]]]:
    """Return only high-confidence human labels derived from contact state."""

    human = segment.get("Human_Metrics", {})
    if not isinstance(human, Mapping):
        return set(), []
    contact = human.get("contact_state", {})
    if not isinstance(contact, Mapping):
        return set(), []

    phase = _token(segment.get("gait_phase", ""))
    phase_type = _token(contact.get("phase_type", ""))
    transition = _token(contact.get("support_transition", ""))
    is_flight = (
        phase == "flight"
        or phase_type in {"flight", "flight_phase"}
        or transition in {"flight", "no_support"}
    )
    if is_flight:
        return set(), []

    if (
        phase == "prepared"
        and phase_type in {"stationary", "prepared"}
        and transition == "double_support"
    ) or (
        phase_type == "double_support"
        and transition == "double_support"
    ):
        source = (
            "Human_Metrics.contact_state.support_transition"
            if "support_transition" in contact
            else "Human_Metrics.contact_state.phase_type"
        )
        return (
            {"double_support_stabilization"},
            [
                {
                    "atom": "double_support_stabilization",
                    "source": source,
                    "value": _json_value(_resolve_source(segment, source)),
                }
            ],
        )
    return set(), []


def _phase_policy(segment: Mapping[str, Any]) -> str:
    """Classify phases that intentionally omit human kinematics/dynamics."""

    phase = _token(segment.get("gait_phase", ""))
    human = segment.get("Human_Metrics", {})
    contact = human.get("contact_state", {}) if isinstance(human, Mapping) else {}
    if not isinstance(contact, Mapping):
        contact = {}
    phase_type = _token(contact.get("phase_type", ""))
    transition = _token(contact.get("support_transition", ""))
    if (
        phase == "flight"
        or phase_type in {"flight", "flight_phase"}
        or transition in {"flight", "no_support"}
    ):
        return "flight_empty"
    if phase == "prepared" or phase_type in {"stationary", "prepared"}:
        return "prepared_contact_only"
    return "metric_allowed"


def should_skip_human_semantic_llm(segment: Mapping[str, Any]) -> bool:
    """Return whether phase/contact rules fully determine the annotation.

    ThreeStepPoleRun/FiveStepPoleRun prepared and flight segments intentionally omit human
    kinematics/dynamics. Calling an LLM for those segments only creates
    unnecessary latency and a route for hallucinated skill atoms.
    """

    human = segment.get("Human_Metrics", {})
    if not isinstance(human, Mapping):
        human = {}
    return (
        _phase_policy(segment) in {"prepared_contact_only", "flight_empty"}
        and not human.get("kinematics")
        and not human.get("dynamics")
    )


def normalize_human_semantics(
    parsed: Mapping[str, Any],
    segment: Mapping[str, Any],
    *,
    candidate_atoms: Iterable[str] = CANONICAL_ATOMS,
) -> dict[str, Any]:
    """Normalize an LLM response into a disjoint human three-state contract.

    LLM ``known_absent_atoms`` are accepted only when the local phase/contact
    rules independently prove them absent.  This makes negative supervision
    conservative and prevents hallucinated omissions from becoming labels.
    """

    candidates = {canonical_atom(item) for item in candidate_atoms}
    raw_positive = {
        canonical_atom(item)
        for item in _items(parsed.get("skill_atoms"))
    } & candidates
    raw_absent = {
        canonical_atom(item)
        for item in _items(parsed.get("known_absent_atoms"))
    } & candidates

    evidence = _normalized_evidence(parsed, segment, candidates)
    deterministic_positive, deterministic_evidence = _deterministic_positive(segment)
    evidence.extend(deterministic_evidence)

    positive: set[str] = set(deterministic_positive)
    for atom in raw_positive:
        atom_evidence = [item for item in evidence if item["atom"] == atom]
        if _evidence_is_valid(atom, atom_evidence, segment):
            positive.add(atom)

    # The current ThreeStepPoleRun/FiveStepPoleRun human metric contract intentionally omits
    # kinematics/dynamics for prepared and flight phases.  Make that policy
    # deterministic so an LLM cannot hallucinate positive motion skills from
    # phase names, step indices, or task descriptions.
    phase_policy = _phase_policy(segment)
    if phase_policy == "flight_empty":
        positive.clear()
    elif phase_policy == "prepared_contact_only":
        positive &= deterministic_positive

    phase_absent = phase_known_absent_atoms(segment)
    # Deterministic phase/contact contradictions have priority over an LLM
    # positive claim. This is especially important for flight phases, where
    # no support-side atom should be asserted without a contradictory source.
    positive -= phase_absent
    absent = (phase_absent | (raw_absent & phase_absent)) - positive
    unknown = candidates - positive - absent

    kept_evidence = [
        item for item in evidence
        if item["atom"] in positive
    ]
    hints = _items(parsed.get("human_derived_control_hints"))
    if not positive:
        hints = []

    phase = str(segment.get("gait_phase", "unknown")).strip()
    if not positive:
        # Do not preserve a free-form caption that may describe discarded
        # hallucinated atoms.  Unknown-only segments get a neutral caption.
        caption = f"{phase} phase with no metric-backed human skill atom asserted."
    else:
        caption = str(parsed.get("short_caption_en", "") or "").strip()
        if not caption:
            caption = f"{phase} phase with metric-backed human skill evidence."

    metric_positive = positive - deterministic_positive
    if deterministic_positive and not metric_positive:
        annotation_state = "phase_contact_backed"
    elif metric_positive:
        annotation_state = "metric_backed"
    else:
        annotation_state = "unknown_or_unobserved"

    return {
        "schema_version": HUMAN_SEMANTICS_SCHEMA_VERSION,
        "short_caption_en": caption,
        "skill_atoms": sorted(positive),
        "unknown_atoms": sorted(unknown),
        "known_absent_atoms": sorted(absent),
        "human_derived_control_hints": _unique(hints)[:6],
        "evidence": kept_evidence,
        "annotation_state": annotation_state,
    }


def normalize_existing_human_semantics(
    semantics: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate old positive-only annotations without inventing evidence."""

    if not semantics:
        # Keep empty fields semantically honest while still exposing
        # deterministic phase/contact facts to the atom loss. In particular,
        # flight support atoms can be known absent even when no LLM call was
        # made; omitted swing atoms remain unknown.
        normalized = normalize_human_semantics({}, segment)
        phase = str(segment.get("gait_phase", "unknown")).strip()
        normalized["short_caption_en"] = (
            f"{phase} phase has no human semantic annotation; "
            "metric supervision is unavailable."
        )
        normalized["human_derived_control_hints"] = []
        normalized["annotation_state"] = "unannotated"
        return normalized

    parsed = dict(semantics)
    # Legacy annotations have no evidence, so only deterministic contact
    # labels survive as positive labels. Other old positives become unknown.
    return normalize_human_semantics(parsed, segment)


def human_semantic_violations(
    semantics: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> list[str]:
    """Return audit findings without changing the supplied annotation."""

    if not semantics:
        # An empty field is an explicit observation gap, not a malformed
        # closed-world annotation. ``--write-normalized`` can materialize the
        # full v2 partition when desired.
        return []

    violations: list[str] = []
    active = set(CANONICAL_ATOMS)
    positive = {canonical_atom(x) for x in _items(semantics.get("skill_atoms"))}
    unknown = {canonical_atom(x) for x in _items(semantics.get("unknown_atoms"))}
    absent = {canonical_atom(x) for x in _items(semantics.get("known_absent_atoms"))}
    if positive & unknown:
        violations.append("skill_atoms overlaps unknown_atoms")
    if positive & absent:
        violations.append("skill_atoms overlaps known_absent_atoms")
    if unknown & absent:
        violations.append("unknown_atoms overlaps known_absent_atoms")
    if (positive | unknown | absent) != active:
        violations.append("atom partition is incomplete or contains inactive atoms")
    evidence = _normalized_evidence(semantics, segment, active)
    for atom in positive:
        atom_evidence = [item for item in evidence if item["atom"] == atom]
        if not _evidence_is_valid(atom, atom_evidence, segment):
            violations.append(f"{atom} lacks valid Human_Metrics evidence")
    if str(semantics.get("schema_version", "")) != HUMAN_SEMANTICS_SCHEMA_VERSION:
        violations.append("legacy or missing schema_version")
    return violations
