#!/usr/bin/env python3
"""Validate exported motion latents without retraining the model.

The script checks:
  * finite values and latent collapse;
  * z_skill / z_sem alignment;
  * z_skill / z_quality cross-correlation;
  * train/validation/test distribution shift;
  * simple linear-probe performance for gait phases, semantic atoms,
    feasibility tags, semantic metrics and quality metrics.
  * warmup/ramp/full-semantic phases from the training history.

Only NumPy is required.

Example:
    python validate_latents.py ^
      --latent-dir runs\\motion_latent ^
      --json-dir LLM_annotation\\CrouchedTraverse

Multi-dataset example:
    python validate_latents.py ^
      --latent-dir runs\\multi_motion_latent ^
      --json-root LLM_annotation ^
      --datasets CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Reuse the exact quality-v2 implementation used during training.  The
# training module is NumPy-safe in audit/validation mode and also installs the
# sibling Data directory on sys.path when this script is run directly.
from motion_latent_z import (
    CHECKPOINT_FORMAT_VERSION,
    QUALITY_CONDITIONING_VERSION,
    QUALITY_INPUT_VERSION,
    QUALITY_V2_AXES,
    SEGMENT_QUERY_VERSION,
    SEMANTIC_FUSION_VERSION,
    TRAINING_PROTOCOL_VERSION,
    derive_quality_v2_states,
    quality_v2_targets_and_mask,
)
from robot_semantic_schema import CANONICAL_ATOMS
from robot_semantic_schema import canonical_atom as shared_canonical_atom
from robot_semantic_schema import phase_known_absent_atoms


DEFAULT_JSON_ROOT = Path(
    r"C:\Users\Administrator\Desktop\Data-HNU\LLM_annotation"
)
DEFAULT_DATASETS = ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")
DEFAULT_LATENT_ROOT = Path(r"C:\Users\Administrator\Desktop\Data-HNU\Latent")
ABLATION_PROFILES = ("Z0", "Z1", "Z2", "Z3", "Z4")
QUALITY_V2_AXES = tuple(QUALITY_V2_AXES)
QUALITY_V2_VOCAB = list(QUALITY_V2_AXES)


def infer_profile(latent_dir: Path, schema: Mapping[str, Any]) -> str:
    """Infer the ablation profile from schema.json or the output directory."""
    profile = str(schema.get("profile", "")).upper()
    if profile in ABLATION_PROFILES:
        return profile
    match = re.search(r"Z([0-4])$", latent_dir.name, flags=re.IGNORECASE)
    return match.group(0).upper() if match else "Z4"


def profile_config(profile: str) -> dict[str, bool]:
    if profile not in ABLATION_PROFILES:
        raise ValueError(f"Unknown ablation profile: {profile}")
    return {
        "robot_numeric": profile in {"Z1", "Z2", "Z3", "Z4"},
        "robot_semantics": profile in {"Z2", "Z3", "Z4"},
        "human_numeric": profile in {"Z3", "Z4"},
        "human_semantics": profile == "Z4",
    }


def profile_report_path(path: Path, profile: str) -> Path:
    """Ensure generated reports are distinguishable across ablation runs."""
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    if not re.search(r"_Z[0-4]$", stem, flags=re.IGNORECASE):
        stem = f"{stem}_{profile}"
    return path.with_name(stem + suffix)


def nested_get(obj: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    value: Any = obj
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return float(value)
    return default


def flatten_numeric(
    obj: Any,
    prefix: str = "",
    *,
    include_bool: bool = False,
) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(value, child, include_bool=include_bool))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child = f"{prefix}.{index}" if prefix else str(index)
            result.update(flatten_numeric(value, child, include_bool=include_bool))
    elif isinstance(obj, bool):
        if include_bool:
            result[prefix] = float(obj)
    elif isinstance(obj, (int, float, np.number)):
        value = float(obj)
        if math.isfinite(value):
            result[prefix] = value
    return result


def keep_metric_path(path: str) -> bool:
    lower = path.lower()
    leaf = lower.rsplit(".", 1)[-1]
    excluded = {
        "start_frame",
        "end_frame",
        "start_index",
        "end_index",
        "peak_frame",
        "sample_rate_hz",
        "source_start_s",
        "source_end_s",
    }
    if leaf in excluded or leaf.endswith("_time_s") or "peak_time_s" in leaf:
        return False
    if "step_index" in lower:
        return False
    return leaf not in {"start", "end"}


def semantic_numeric_map(
    doc: Mapping[str, Any],
    segment: Mapping[str, Any],
    profile: str,
) -> dict[str, float]:
    config = profile_config(profile)
    result: dict[str, float] = {}
    if config["human_numeric"]:
        for key, path in (
            ("trial.body_height_m", ("subject_model", "body_height_m")),
            ("trial.body_weight_kg", ("subject_model", "body_weight_kg")),
            ("segment.duration_s", ("time", "duration_s")),
        ):
            value = nested_get(doc if key.startswith("trial.") else segment, path)
            if value is not None:
                result[key] = as_float(value)
        human = nested_get(segment, ("Human_Metrics",), {}) or {}
        for namespace in (
            "contact_state",
            "step_info",
            "obstacle_relation",
            "kinematics",
            "dynamics",
        ):
            values = flatten_numeric(
                human.get(namespace, {}), f"human.{namespace}", include_bool=True
            )
            result.update(
                {key: value for key, value in values.items() if keep_metric_path(key)}
            )
    if config["robot_numeric"]:
        robot = nested_get(segment, ("Retargeted_Robot_Metrics",), {}) or {}
        values = flatten_numeric(
            robot.get("kinematics", {}), "robot.kinematics", include_bool=True
        )
        result.update(
            {key: value for key, value in values.items() if keep_metric_path(key)}
        )
    return result


def quality_numeric_map(
    segment: Mapping[str, Any], profile: str
) -> dict[str, float]:
    if not profile_config(profile)["robot_numeric"]:
        return {}
    robot = nested_get(segment, ("Retargeted_Robot_Metrics",), {}) or {}
    result: dict[str, float] = {}
    for namespace in ("dynamics", "feasibility"):
        values = flatten_numeric(
            robot.get(namespace, {}),
            f"robot_quality.{namespace}",
            include_bool=True,
        )
        result.update(
            {key: value for key, value in values.items() if keep_metric_path(key)}
        )
    alignment = nested_get(robot, ("npz_alignment",), {}) or {}
    start = as_float(alignment.get("start_index"))
    end = as_float(alignment.get("end_index"))
    whole = nested_get(robot, ("feasibility", "torque_screening", "whole_body"), {}) or {}
    joint = nested_get(robot, ("feasibility", "state_validity", "joint_limit"), {}) or {}
    collision = nested_get(robot, ("feasibility", "state_validity", "collision"), {}) or {}
    tags = nested_get(robot, ("robot_semantics", "feasibility_tags"), []) or []
    risk = (
        max(as_float(whole.get("peak_limit_ratio")) - 1.0, 0.0)
        + as_float(whole.get("violation_frame_ratio"))
        + as_float(joint.get("violation_frame_ratio"))
        + min(as_float(joint.get("max_excess")) / 0.10, 3.0)
        + min(as_float(collision.get("self_collision_frame_count")) / max(end - start, 1.0), 1.0)
        + min(as_float(collision.get("max_self_penetration_mm")) / 20.0, 2.0)
        + 0.25 * min(as_float(collision.get("max_floor_penetration_mm")) / 10.0, 2.0)
        + (0.5 if "contact_mismatch" in tags else 0.0)
    )
    result["robot_quality.derived.risk_score"] = float(risk)
    result["robot_quality.derived.feasibility_score"] = float(math.exp(-risk))
    return result


def get_phase_label(segment: Mapping[str, Any]) -> str:
    """Read phase labels from all supported annotation versions."""
    for key in ("gait_phase", "phase", "Phase", "鐩镐綅"):
        value = segment.get(key)
        if value not in (None, ""):
            return str(value).strip().lower().replace(" ", "_")
    return "unknown"


def canonical_atom(atom: str) -> str:
    """Keep the historical local API while using the shared ontology."""
    return shared_canonical_atom(atom)


def semantic_state_sets(
    semantics: Any,
    segment: Mapping[str, Any] | None = None,
) -> tuple[set[str], set[str], set[str], bool]:
    """Read both new three-state and legacy binary semantic JSON."""
    if not isinstance(semantics, Mapping):
        return set(), set(), set(), False
    active_atoms = set(CANONICAL_ATOMS)
    positive = {
        canonical_atom(str(item))
        for item in (semantics.get("skill_atoms") or [])
        if canonical_atom(str(item)) in active_atoms
    }
    has_tri_state = any(
        key in semantics for key in ("unknown_atoms", "known_absent_atoms")
    )
    if not has_tri_state:
        # Legacy annotations contain positive labels only. Every omitted
        # canonical atom is therefore unknown, not an explicit negative.
        unknown = set(CANONICAL_ATOMS).difference(positive)
        absent = set()
    else:
        unknown = {
            canonical_atom(str(item))
            for item in (semantics.get("unknown_atoms") or [])
            if canonical_atom(str(item)) in active_atoms
        }
        absent = {
            canonical_atom(str(item))
            for item in (semantics.get("known_absent_atoms") or [])
            if canonical_atom(str(item)) in active_atoms
        }
    if segment is not None:
        absent.update(phase_known_absent_atoms(segment))
    unknown -= positive
    unknown -= absent
    absent -= positive
    unknown |= active_atoms.difference(positive | absent | unknown)
    return positive, unknown, absent, True


def load_labels(
    json_dir: Path,
    *,
    dataset_id: str | None = None,
    profile: str = "Z4",
) -> dict[tuple[str, str], dict[str, Any]]:
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(json_dir.glob("**/*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        trial_id = str(doc.get("trial_id") or path.stem)
        if dataset_id and not trial_id.startswith(f"{dataset_id}_"):
            trial_id = f"{dataset_id}_{trial_id}"
        for segment in doc.get("segments", []):
            segment_id = str(segment.get("segment_id", "unknown"))
            config = profile_config(profile)
            human_semantics = (
                nested_get(
                    segment, ("Human_Metrics", "human_semantics"), {}
                )
                if config["human_semantics"]
                else {}
            )
            robot_semantics = (
                nested_get(
                    segment,
                    ("Retargeted_Robot_Metrics", "robot_semantics"),
                    {},
                )
                if config["robot_semantics"]
                else {}
            )
            human, human_unknown, human_absent, human_known = semantic_state_sets(
                human_semantics,
                segment=segment,
            )
            robot, robot_unknown, robot_absent, robot_known = semantic_state_sets(
                robot_semantics,
                segment=segment,
            )
            tags = (
                nested_get(
                    segment,
                    (
                        "Retargeted_Robot_Metrics",
                        "robot_semantics",
                        "feasibility_tags",
                    ),
                    [],
                )
                if config["robot_semantics"]
                else []
            ) or []
            atom_annotation_present = bool(human_known or robot_known)
            tag_annotation_present = bool(
                config["robot_semantics"]
                and isinstance(
                    nested_get(
                        segment,
                        (
                            "Retargeted_Robot_Metrics",
                            "robot_semantics",
                        ),
                        {},
                    ),
                    Mapping,
                )
                and "feasibility_tags"
                in nested_get(
                    segment,
                    (
                        "Retargeted_Robot_Metrics",
                        "robot_semantics",
                    ),
                    {},
                )
            )
            # Keep quality-v2 labels in exactly the same fixed axis order as
            # motion_latent_z.py.  Unknown is represented by target=-1 and
            # mask=0; it must never be treated as a negative example.
            quality_v2_targets, quality_v2_mask, quality_v2_states = (
                quality_v2_targets_and_mask(segment, profile)
            )
            tags_known_by_axis = {
                axis: bool(quality_v2_mask[index] > 0.0)
                for index, axis in enumerate(QUALITY_V2_AXES)
            }
            labels[(trial_id, segment_id)] = {
                "gait_phase": get_phase_label(segment),
                "atoms": human | robot,
                "atoms_unknown": (
                    (human_unknown | robot_unknown) - (human | robot)
                ),
                "atoms_known_absent": (
                    (human_absent | robot_absent)
                    - (human | robot)
                    - (human_unknown | robot_unknown)
                ),
                "tags": {str(item) for item in tags},
                "atoms_known": atom_annotation_present,
                # Atom-level knowledge is required for sparse semantic probes.
                # Legacy JSON contributes only its positive list; omitted
                # atoms remain unobserved instead of becoming false negatives.
                "atoms_known_atoms": (
                    (human | robot | human_absent | robot_absent)
                    - (human_unknown | robot_unknown)
                ),
                # New quality labels are known independently for every axis.
                "tags_known": tags_known_by_axis,
                "legacy_tags_known": tag_annotation_present,
                "quality_v2_targets": quality_v2_targets,
                "quality_v2_mask": quality_v2_mask,
                "quality_v2_states": quality_v2_states,
                "semantic": semantic_numeric_map(doc, segment, profile),
                "quality": quality_numeric_map(segment, profile),
            }
    if not labels:
        raise ValueError(f"No semantic labels found in {json_dir}")
    return labels


def load_multi_dataset_labels(
    json_root: Path,
    dataset_names: Sequence[str],
    profile: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load labels from several dataset directories with collision-free IDs."""
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset_name in dataset_names:
        path = json_root / dataset_name
        if not path.is_dir():
            raise NotADirectoryError(path)
        loaded = load_labels(path, dataset_id=dataset_name, profile=profile)
        overlap = set(labels).intersection(loaded)
        if overlap:
            raise ValueError(f"Duplicate label keys: {sorted(overlap)[:5]}")
        labels.update(loaded)
    if not labels:
        raise ValueError(f"No semantic labels found under {json_root}")
    return labels


def load_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "sem_keys": [],
            "quality_keys": [],
            "atom_vocab": [],
            "quality_tag_vocab": [],
            "gait_vocab": [],
        }
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_split(path: Path, labels: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    required = ("z_skill", "z_quality", "z_sem", "z_sem_deploy", "trial_id", "segment_id")
    missing_arrays = [key for key in required if key not in data]
    if missing_arrays:
        raise KeyError(
            f"{path} is missing current-protocol arrays: {', '.join(missing_arrays)}"
        )
    trial_ids = [str(x) for x in data["trial_id"]]
    segment_ids = [str(x) for x in data["segment_id"]]
    keys = list(zip(trial_ids, segment_ids))
    missing = [key for key in keys if key not in labels]
    if missing and labels:
        print(f"WARNING: {path.name}: missing labels for {len(missing)} rows")
    return {
        "name": path.stem.replace("_latents", ""),
        "data": data,
        "keys": keys,
        "labels": [labels.get(key) for key in keys],
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def latent_health(z: np.ndarray) -> dict[str, Any]:
    z = np.asarray(z, dtype=np.float64)
    finite = bool(np.isfinite(z).all())
    std = z.std(axis=0) if len(z) else np.array([])
    centered = z - z.mean(axis=0) if len(z) else z
    if len(z) > 1:
        singular = np.linalg.svd(centered, compute_uv=False)
        energy = singular * singular
        effective_rank = float((energy.sum() ** 2) / np.maximum((energy * energy).sum(), 1e-12))
    else:
        effective_rank = 0.0
    return {
        "shape": list(z.shape),
        "finite": finite,
        "mean_norm": float(np.linalg.norm(z.mean(axis=0))) if len(z) else 0.0,
        "sample_norm_mean": float(np.linalg.norm(z, axis=1).mean()) if len(z) else 0.0,
        "min_dimension_std": float(std.min()) if len(std) else 0.0,
        "max_dimension_std": float(std.max()) if len(std) else 0.0,
        "collapsed_dimensions_std_lt_0.01": int((std < 0.01).sum()),
        "effective_rank": effective_rank,
    }


def alignment_metrics(skill: np.ndarray, sem: np.ndarray) -> dict[str, float]:
    skill_sample_norm = np.linalg.norm(skill, axis=1)
    sem_sample_norm = np.linalg.norm(sem, axis=1)
    norm_ratio = sem_sample_norm / np.maximum(skill_sample_norm, 1e-8)
    cosine = (skill * sem).sum(axis=1) / (
        skill_sample_norm * sem_sample_norm + 1e-8
    )
    skill_centered = skill - skill.mean(axis=0)
    sem_centered = sem - sem.mean(axis=0)
    skill_norm = np.linalg.norm(skill_centered)
    sem_norm = np.linalg.norm(sem_centered)
    cka = float(
        (skill_centered.T @ sem_centered).sum() ** 2
        / max(
            (skill_centered.T @ skill_centered).sum()
            * (sem_centered.T @ sem_centered).sum(),
            1e-12,
        )
    )
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std()),
        "cosine_min": float(cosine.min()),
        "cosine_max": float(cosine.max()),
        "linear_cka": cka,
        "skill_sample_norm_mean": float(skill_sample_norm.mean()),
        "semantic_sample_norm_mean": float(sem_sample_norm.mean()),
        "semantic_to_skill_norm_ratio_mean": float(norm_ratio.mean()),
        "semantic_to_skill_norm_ratio_median": float(np.median(norm_ratio)),
        "absolute_log_norm_ratio_mean": float(
            np.abs(np.log(np.maximum(norm_ratio, 1e-8))).mean()
        ),
    }


def decorrelation_metrics(skill: np.ndarray, quality: np.ndarray) -> dict[str, float]:
    a = (skill - skill.mean(axis=0)) / (skill.std(axis=0) + 1e-8)
    b = (quality - quality.mean(axis=0)) / (quality.std(axis=0) + 1e-8)
    corr = a.T @ b / max(len(skill) - 1, 1)
    return {
        "mean_abs_cross_correlation": float(np.abs(corr).mean()),
        "max_abs_cross_correlation": float(np.abs(corr).max()),
        "squared_cross_covariance": float((corr * corr).mean()),
    }


def standardize_fit(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    x_mean, x_std = train_x.mean(axis=0), train_x.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x_train = (train_x - x_mean) / x_std
    x_eval = (eval_x - x_mean) / x_std
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_eval = np.column_stack((np.ones(len(x_eval)), x_eval))
    reg = alpha * np.eye(x_train.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ train_y)
    return x_eval @ weights


def binary_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    prediction = ridge_fit_predict(train_x, train_y, eval_x)
    predicted = prediction >= threshold
    actual = eval_y >= 0.5
    tp = int(np.logical_and(predicted, actual).sum())
    tn = int(np.logical_and(~predicted, ~actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "positive_rate": float(actual.mean()) if len(actual) else 0.0,
        "support": int(len(actual)),
        "n": int(len(actual)),
        "threshold": float(threshold),
    }


def _binary_scores(
    prediction: np.ndarray,
    actual: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    predicted = prediction >= threshold
    tp = int(np.logical_and(predicted, actual).sum())
    tn = int(np.logical_and(~predicted, ~actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return float(f1), float(0.5 * (tpr + tnr))


def choose_threshold_from_predictions(
    prediction: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Select one threshold from the supplied split only.

    For quality-v2, this is called with validation predictions and labels;
    test labels are never passed to this function.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(actual, dtype=bool)
    if len(actual) == 0 or actual.min() == actual.max():
        return 0.5
    finite = np.isfinite(prediction)
    prediction = prediction[finite]
    actual = actual[finite]
    if len(actual) == 0:
        return 0.5
    candidates = np.unique(
        np.clip(
            np.concatenate(
                (
                    np.linspace(0.05, 0.95, 19),
                    np.quantile(prediction, np.linspace(0.01, 0.99, 99)),
                )
            ),
            -10.0,
            10.0,
        )
    )
    best = (float("-inf"), float("-inf"), 0.5)
    for threshold in candidates:
        f1, balanced_accuracy = _binary_scores(
            prediction, actual, float(threshold)
        )
        score = (f1, balanced_accuracy)
        if score > best[:2]:
            best = (f1, balanced_accuracy, float(threshold))
    return float(best[2])


def choose_oof_threshold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    folds: int = 5,
) -> float:
    """Choose a threshold from out-of-fold train predictions only.

    Using OOF predictions avoids selecting a threshold from predictions made
    on the same samples used to fit the ridge probe.  This is important for
    sparse atoms and feasibility tags, where a threshold of 0.5 often
    predicts no positives.
    """
    n = len(train_y)
    if n < 10 or train_y.min() == train_y.max():
        return 0.5
    order = np.arange(n)
    positives = np.flatnonzero(train_y >= 0.5)
    negatives = np.flatnonzero(train_y < 0.5)
    rng = np.random.default_rng(12345)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    shuffled = np.concatenate((positives, negatives))
    rng.shuffle(shuffled)
    order = shuffled
    n_folds = min(max(int(folds), 2), n)
    oof = np.full(n, np.nan, dtype=np.float64)
    for fold in range(n_folds):
        eval_indices = order[fold::n_folds]
        train_indices = np.setdiff1d(order, eval_indices, assume_unique=False)
        if len(eval_indices) == 0 or len(train_indices) == 0:
            continue
        oof[eval_indices] = ridge_fit_predict(
            train_x[train_indices],
            train_y[train_indices],
            train_x[eval_indices],
        )
    valid = np.isfinite(oof)
    if valid.sum() < 10:
        return 0.5
    actual = train_y[valid] >= 0.5
    prediction = oof[valid]
    candidates = np.unique(
        np.concatenate(
            (
                np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                np.quantile(prediction, np.linspace(0.02, 0.98, 49)),
            )
        )
    )
    best = (float("-inf"), float("-inf"), 0.5)
    for threshold in candidates:
        f1, balanced_accuracy = _binary_scores(
            prediction, actual, float(threshold)
        )
        # F1 is primary for sparse labels; balanced accuracy breaks ties.
        score = (f1, balanced_accuracy)
        if score > best[:2]:
            best = (f1, balanced_accuracy, float(threshold))
    return float(best[2])


def average_precision_score(
    prediction: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Small NumPy-only average precision implementation."""
    actual = np.asarray(actual, dtype=bool)
    if actual.sum() == 0:
        return 0.0
    order = np.argsort(-np.asarray(prediction, dtype=np.float64), kind="mergesort")
    truth = actual[order].astype(np.float64)
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float((precision * truth).sum() / max(truth.sum(), 1.0))


def roc_auc_score(
    prediction: np.ndarray,
    actual: np.ndarray,
) -> float:
    """NumPy-only ROC AUC with average ranks for tied scores."""

    scores = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(actual, dtype=bool)
    positive_n = int(truth.sum())
    negative_n = int((~truth).sum())
    if positive_n == 0 or negative_n == 0:
        raise ValueError("ROC AUC requires both positive and negative labels")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # One-based average rank for the half-open tie group [start, end).
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    positive_rank_sum = float(ranks[truth].sum())
    mann_whitney = positive_rank_sum - positive_n * (positive_n + 1) / 2.0
    return float(mann_whitney / (positive_n * negative_n))


def run_binary_probes(
    train: dict[str, Any],
    evaluation: dict[str, Any],
    latent_key: str,
    label_key: str,
    vocabulary: Sequence[str],
    known_key: str | None = None,
    unknown_key: str | None = None,
    min_positive_support: int = 5,
    *,
    target_key: str | None = None,
    mask_key: str | None = None,
    threshold_overrides: Mapping[str, float] | None = None,
    threshold_selection: str = "oof_train",
) -> dict[str, Any]:
    """Run one-vs-rest probes while respecting per-label knowledge masks.

    ``target_key``/``mask_key`` are used by quality-v2.  In that mode each
    label is read from its fixed array position and ``mask==0`` is unknown;
    unknown rows are removed from both fitting and evaluation.  Non-finite
    targets are also treated as unknown, never as a negative label.

    ``threshold_selection='validation'`` scans thresholds on the supplied
    validation split.  ``threshold_selection='frozen_validation'`` never
    scans any labels and uses only ``threshold_overrides`` (falling back to a
    fixed 0.5 threshold when a validation threshold is unavailable).  This
    makes it impossible for a test call to select a threshold from test
    labels by accident.  The legacy ``oof_train`` mode remains available for
    non-quality diagnostic probes and old reports.
    """
    train_x = np.asarray(train["data"][latent_key], dtype=np.float64)
    eval_x = np.asarray(evaluation["data"][latent_key], dtype=np.float64)
    result: dict[str, Any] = {}
    threshold_overrides = dict(threshold_overrides or {})
    train_feature_finite = np.isfinite(train_x).all(axis=1)
    eval_feature_finite = np.isfinite(eval_x).all(axis=1)

    def array_value(
        item: Mapping[str, Any] | None,
        key: str | None,
        index: int,
        default: float,
    ) -> float:
        if item is None or key is None:
            return default
        value = item.get(key)
        try:
            array = np.asarray(value).reshape(-1)
            if index >= len(array):
                return default
            numeric = float(array[index])
            return numeric if math.isfinite(numeric) else default
        except (TypeError, ValueError, IndexError):
            return default

    def item_knows_label(
        item: Mapping[str, Any] | None, label: str, index: int
    ) -> bool:
        if target_key is not None and mask_key is not None:
            # A mask alone is not sufficient: a missing/NaN target must not be
            # converted to target=0 and counted as a known negative.
            mask_value = array_value(item, mask_key, index, 0.0)
            target_value = array_value(item, target_key, index, float("nan"))
            return mask_value > 0.0 and math.isfinite(target_value)
        if item is None:
            return False
        if known_key is None:
            return True
        value = item.get(known_key, False)
        if isinstance(value, (set, list, tuple)):
            return label in value
        if isinstance(value, Mapping):
            return bool(value.get(label, False))
        return bool(value)

    def item_is_unknown(
        item: Mapping[str, Any] | None, label: str, index: int
    ) -> bool:
        if target_key is not None and mask_key is not None:
            return not item_knows_label(item, label, index)
        return bool(
            item is not None
            and unknown_key
            and label in item.get(unknown_key, set())
        )

    def item_target(
        item: Mapping[str, Any] | None, label: str, index: int
    ) -> float:
        if target_key is not None:
            value = array_value(item, target_key, index, float("nan"))
            return 1.0 if math.isfinite(value) and value >= 0.5 else 0.0
        return float(item is not None and label in item.get(label_key, set()))

    def unavailable_result(status: str, support: Mapping[str, Any]) -> dict[str, Any]:
        # Keep metric keys present for every axis, including all-unknown and
        # constant axes, so downstream aggregation never has to special-case a
        # missing key.  None means that the metric is not defined for status.
        return {
            "status": status,
            "f1": None,
            "balanced_accuracy": None,
            "average_precision": None,
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "positive_rate": None,
            "threshold": None,
            "threshold_source": None,
            **dict(support),
        }

    for index, label in enumerate(vocabulary):
        train_known = np.array(
            [item_knows_label(item, label, index) for item in train["labels"]],
            dtype=bool,
        )
        eval_known = np.array(
            [item_knows_label(item, label, index) for item in evaluation["labels"]],
            dtype=bool,
        )
        train_unknown = np.array(
            [item_is_unknown(item, label, index) for item in train["labels"]],
            dtype=bool,
        )
        eval_unknown = np.array(
            [item_is_unknown(item, label, index) for item in evaluation["labels"]],
            dtype=bool,
        )
        train_y_all = np.array(
            [item_target(item, label, index) for item in train["labels"]],
            dtype=np.float64,
        )
        eval_y_all = np.array(
            [item_target(item, label, index) for item in evaluation["labels"]],
            dtype=np.float64,
        )

        # Label support is independent from feature support.  Invalid feature
        # rows are excluded from fitting/metrics but are reported explicitly.
        train_label_valid = train_known & ~train_unknown
        eval_label_valid = eval_known & ~eval_unknown
        train_valid = train_label_valid & train_feature_finite
        eval_valid = eval_label_valid & eval_feature_finite
        train_x_valid = train_x[train_valid]
        train_y = train_y_all[train_valid]
        eval_x_valid = eval_x[eval_valid]
        eval_y = eval_y_all[eval_valid]
        train_positive_n = int(train_y.sum())
        eval_positive_n = int(eval_y.sum())
        train_unknown_n = int((~train_label_valid).sum())
        eval_unknown_n = int((~eval_label_valid).sum())
        train_known_absent_n = int(
            train_label_valid.sum() - train_y_all[train_label_valid].sum()
        )
        eval_known_absent_n = int(
            eval_label_valid.sum() - eval_y_all[eval_label_valid].sum()
        )
        support = {
            "train_known_positive_n": train_positive_n,
            "train_known_absent_n": train_known_absent_n,
            "train_unknown_n": train_unknown_n,
            "eval_known_positive_n": eval_positive_n,
            "eval_known_absent_n": eval_known_absent_n,
            "eval_unknown_n": eval_unknown_n,
            "train_feature_invalid_n": int(
                (train_label_valid & ~train_feature_finite).sum()
            ),
            "eval_feature_invalid_n": int(
                (eval_label_valid & ~eval_feature_finite).sum()
            ),
            "support": int(eval_valid.sum()),
            "unknown": int(eval_unknown_n),
        }

        if len(eval_y) == 0:
            result[label] = unavailable_result(
                "all_unknown"
                if eval_unknown_n == len(eval_y_all)
                else "insufficient_known_labels",
                support,
            )
            continue
        if len(train_y) == 0:
            result[label] = unavailable_result("insufficient_known_labels", support)
            continue
        if train_y.min() == train_y.max() or eval_y.min() == eval_y.max():
            result[label] = unavailable_result("constant_labels", support)
            continue
        if (
            train_positive_n < min_positive_support
            or eval_positive_n < min_positive_support
        ):
            result[label] = {
                **unavailable_result("insufficient_support", support),
                "min_positive_support": int(min_positive_support),
            }
            continue

        try:
            eval_prediction = ridge_fit_predict(
                train_x_valid, train_y, eval_x_valid
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            result[label] = unavailable_result("probe_failed", support)
            continue
        finite_prediction = np.isfinite(eval_prediction)
        if not finite_prediction.any():
            result[label] = unavailable_result("no_finite_predictions", support)
            continue
        metric_eval_x = eval_x_valid
        if not finite_prediction.all():
            metric_eval_x = eval_x_valid[finite_prediction]
            eval_prediction = eval_prediction[finite_prediction]
            eval_y = eval_y[finite_prediction]
            if len(eval_y) == 0:
                result[label] = unavailable_result("no_finite_predictions", support)
                continue
            if eval_y.min() == eval_y.max():
                result[label] = unavailable_result("constant_labels", support)
                continue

        override = threshold_overrides.get(label)
        override_value = as_float(override, float("nan"))
        if override is not None and math.isfinite(override_value):
            threshold = float(override_value)
            threshold_source = "validation_override"
        elif threshold_selection == "validation":
            threshold = choose_threshold_from_predictions(
                eval_prediction, eval_y >= 0.5
            )
            threshold_source = "validation_scan"
        elif threshold_selection == "frozen_validation":
            # Test is never allowed to scan test labels or use OOF train
            # predictions as a replacement for a frozen validation choice.
            threshold = 0.5
            threshold_source = "fixed_default_no_validation_threshold"
        else:
            threshold = choose_oof_threshold(train_x_valid, train_y)
            threshold_source = "oof_train"

        try:
            calibrated = binary_probe(
                train_x_valid,
                train_y,
                metric_eval_x,
                eval_y,
                threshold=threshold,
            )
            fixed = binary_probe(
                train_x_valid,
                train_y,
                metric_eval_x,
                eval_y,
                threshold=0.5,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            result[label] = unavailable_result("probe_failed", support)
            continue
        actual = eval_y >= 0.5
        calibrated["fixed_0_5_f1"] = fixed["f1"]
        calibrated["fixed_0_5_balanced_accuracy"] = fixed["balanced_accuracy"]
        calibrated["average_precision"] = average_precision_score(
            eval_prediction, actual
        )
        calibrated["roc_auc"] = roc_auc_score(eval_prediction, actual)
        calibrated["threshold_source"] = threshold_source
        calibrated.update(support)
        calibrated["status"] = "ok"
        result[label] = calibrated

    valid = [
        item
        for item in result.values()
        if isinstance(item, Mapping)
        and item.get("status") == "ok"
        and all(
            isinstance(item.get(key), (int, float, np.number))
            and math.isfinite(float(item[key]))
            for key in ("balanced_accuracy", "f1", "average_precision", "roc_auc")
        )
    ]
    if valid:
        result["_macro"] = {
            "balanced_accuracy": float(
                np.mean([item["balanced_accuracy"] for item in valid])
            ),
            "f1": float(np.mean([item["f1"] for item in valid])),
            "precision": float(np.mean([item["precision"] for item in valid])),
            "recall": float(np.mean([item["recall"] for item in valid])),
            "average_precision": float(
                np.mean([item["average_precision"] for item in valid])
            ),
            "roc_auc": float(np.mean([item["roc_auc"] for item in valid])),
            "support": int(sum(item["support"] for item in valid)),
            "unknown": int(sum(item["unknown"] for item in valid)),
            "n_labels": len(valid),
        }
    else:
        result["_macro"] = {
            "status": "no_valid_axes",
            "balanced_accuracy": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "average_precision": None,
            "roc_auc": None,
            "support": 0,
            "unknown": int(
                sum(
                    int(item.get("unknown", 0))
                    for item in result.values()
                    if isinstance(item, Mapping)
                )
            ),
            "n_labels": 0,
        }
    return result


def run_gait_probe(
    train: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, float]:
    train_x = train["data"]["z_skill"].astype(np.float64)
    eval_x = evaluation["data"]["z_skill"].astype(np.float64)
    # Exported phase labels are available even for Z0, which intentionally has
    # no JSON supervision.
    train_labels = np.asarray(
        train["data"]["gait_phase"]
        if "gait_phase" in train["data"]
        else [
            item["gait_phase"] if item is not None else "unknown"
            for item in train["labels"]
        ]
    )
    eval_labels = np.asarray(
        evaluation["data"]["gait_phase"]
        if "gait_phase" in evaluation["data"]
        else [
            item["gait_phase"] if item is not None else "unknown"
            for item in evaluation["labels"]
        ]
    )
    classes = sorted(set(train_labels))
    means = {
        label: train_x[train_labels == label].mean(axis=0)
        for label in classes
        if np.any(train_labels == label)
    }
    predicted = np.array(
        [
            min(means, key=lambda label: float(np.linalg.norm(vector - means[label])))
            for vector in eval_x
        ]
    )
    accuracy = float((predicted == eval_labels).mean())
    per_class = []
    for label in sorted(set(eval_labels)):
        actual = eval_labels == label
        if actual.any():
            per_class.append(float((predicted[actual] == label).mean()))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class)) if per_class else 0.0,
        "n": int(len(eval_labels)),
    }


def run_metric_correlations(
    split: dict[str, Any],
    latent_key: str,
    label_group: str,
    keys: Sequence[str],
) -> dict[str, Any]:
    x = split["data"][latent_key].astype(np.float64)
    result: dict[str, Any] = {}
    for key in keys:
        values = []
        valid_rows = []
        for index, item in enumerate(split["labels"]):
            if item is not None and key in item[label_group]:
                values.append(item[label_group][key])
                valid_rows.append(index)
        if len(values) < 3:
            continue
        correlations = [
            safe_corr(x[valid_rows, dim], np.asarray(values))
            for dim in range(x.shape[1])
        ]
        correlations = [abs(value) for value in correlations if math.isfinite(value)]
        if correlations:
            result[key] = float(max(correlations))
    if result:
        result["_mean_max_abs_corr"] = float(np.mean(list(result.values())))
    return result


def distribution_shift(train: dict[str, Any], evaluation: dict[str, Any], key: str) -> dict[str, float]:
    a = train["data"][key].astype(np.float64)
    b = evaluation["data"][key].astype(np.float64)
    mean_distance = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))
    pooled_std = np.sqrt((a.var(axis=0).mean() + b.var(axis=0).mean()) / 2.0)
    return {
        "mean_distance": float(mean_distance),
        "normalized_mean_distance": float(mean_distance / max(pooled_std, 1e-8)),
    }


def validate_split(
    split: dict[str, Any],
    schema: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    data = split["data"]
    has_sem = "z_sem" in data
    has_deploy_sem = "z_sem_deploy" in data
    has_quality = "z_quality" in data
    report: dict[str, Any] = {
        "rows": int(len(data["z_skill"])),
        "latents": {
            key: latent_health(data[key])
            for key in ("z_skill", "z_sem", "z_quality")
            if key in data
        },
        "profile": profile,
    }
    if has_sem:
        report["alignment"] = alignment_metrics(
            data["z_skill"].astype(np.float64), data["z_sem"].astype(np.float64)
        )
    if has_deploy_sem:
        report["deployment_alignment"] = alignment_metrics(
            data["z_skill"].astype(np.float64),
            data["z_sem_deploy"].astype(np.float64),
        )
        report["deployment_to_teacher_alignment"] = alignment_metrics(
            data["z_sem"].astype(np.float64),
            data["z_sem_deploy"].astype(np.float64),
        )
    if has_quality:
        report["decorrelation"] = decorrelation_metrics(
            data["z_skill"].astype(np.float64), data["z_quality"].astype(np.float64)
        )
    if split["labels"] and any(item is not None for item in split["labels"]):
        if schema.get("sem_keys"):
            report["semantic_metric_correlations_from_z_skill"] = (
                run_metric_correlations(
                    split, "z_skill", "semantic", schema.get("sem_keys", [])
                )
            )
            if has_sem:
                report["semantic_metric_correlations_from_z_sem"] = (
                    run_metric_correlations(
                        split, "z_sem", "semantic", schema.get("sem_keys", [])
                    )
                )
        if has_quality and schema.get("quality_keys"):
            report["quality_metric_correlations_from_z_quality"] = (
                run_metric_correlations(
                    split, "z_quality", "quality", schema.get("quality_keys", [])
                )
            )
            report["quality_metric_correlations_from_z_skill"] = (
                run_metric_correlations(
                    split, "z_skill", "quality", schema.get("quality_keys", [])
                )
            )
    return report


def history_semantic_scale(item: Mapping[str, Any]) -> float:
    """Read semantic-loss scale from new or legacy training histories."""
    train = item.get("train", {})
    val = item.get("val", {})
    if isinstance(val, Mapping) and "semantic_scale" in val:
        return as_float(val["semantic_scale"])
    if isinstance(train, Mapping) and "semantic_scale" in train:
        return as_float(train["semantic_scale"])
    # Legacy histories only had semantic_enabled.  Treat enabled epochs as
    # full semantic training so old reports remain interpretable.
    return 1.0 if bool(item.get("semantic_enabled", False)) else 0.0


def summarize_history(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize training phases and checkpoint-relevant validation losses."""
    records = [item for item in history if isinstance(item, Mapping)]
    phase_counts = {"warmup": 0, "ramp": 0, "full_semantic": 0}
    full_records: list[Mapping[str, Any]] = []
    for item in records:
        scale = history_semantic_scale(item)
        if scale <= 0.0:
            phase_counts["warmup"] += 1
        elif scale < 1.0 - 1e-6:
            phase_counts["ramp"] += 1
        else:
            phase_counts["full_semantic"] += 1
            if isinstance(item.get("val"), Mapping):
                full_records.append(item)

    def best_record(
        items: Sequence[Mapping[str, Any]],
        *,
        use_selection_score: bool = False,
    ) -> dict[str, Any] | None:
        candidates = [
            item
            for item in items
            if isinstance(item.get("val"), Mapping)
            and math.isfinite(
                as_float(
                    item.get("selection_score")
                    if use_selection_score
                    else item["val"].get("loss"),
                    float("nan"),
                )
            )
        ]
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda item: as_float(
                item.get("selection_score")
                if use_selection_score
                else item["val"]["loss"]
            ),
        )
        return {
            "epoch": int(as_float(best.get("epoch"), 0)),
            "semantic_scale": history_semantic_scale(best),
            "train_loss": as_float(
                nested_get(best, ("train", "loss")), float("nan")
            ),
            "val_loss": as_float(
                nested_get(best, ("val", "loss")), float("nan")
            ),
            "selection_score": as_float(
                best.get("selection_score"), float("nan")
            ),
            "semantic_norm_ratio": as_float(
                nested_get(best, ("val", "semantic_norm_ratio")),
                float("nan"),
            ),
        }

    final = records[-1] if records else {}
    final_val = final.get("val", {}) if isinstance(final, Mapping) else {}
    return {
        "epochs_recorded": len(records),
        "phase_counts": phase_counts,
        "best_full_semantic_validation": best_record(full_records),
        "best_full_semantic_selection": best_record(
            full_records,
            use_selection_score=True,
        ),
        "best_overall_validation": best_record(records),
        "final": {
            "epoch": int(as_float(final.get("epoch"), 0)) if records else 0,
            "semantic_scale": history_semantic_scale(final) if records else 0.0,
            "train_loss": as_float(
                nested_get(final, ("train", "loss")), float("nan")
            )
            if records
            else float("nan"),
            "val_loss": as_float(final_val.get("loss"), float("nan"))
            if isinstance(final_val, Mapping)
            else float("nan"),
        },
    }


def print_report(report: Mapping[str, Any]) -> None:
    print("\n=== Latent health ===")
    command_interface = report.get("command_interface", {})
    if command_interface:
        print(
            "command interface: "
            f"task_context={command_interface.get('task_context')}, "
            f"query_version={command_interface.get('task_query_version')}"
        )
    for split, values in report["splits"].items():
        print(f"\n[{split}] rows={values['rows']}")
        for key, stats in values["latents"].items():
            print(
                f"{key}: shape={tuple(stats['shape'])}, finite={stats['finite']}, "
                f"std_range=({stats['min_dimension_std']:.4f}, "
                f"{stats['max_dimension_std']:.4f}), "
                f"collapsed_dims={stats['collapsed_dimensions_std_lt_0.01']}, "
                f"effective_rank={stats['effective_rank']:.2f}"
            )
        if "alignment" in values:
            alignment = values["alignment"]
            print(
                f"alignment: cosine={alignment['cosine_mean']:.4f}, "
                f"CKA={alignment['linear_cka']:.4f}, "
                "z_sem/z_skill norm="
                f"{alignment['semantic_to_skill_norm_ratio_mean']:.3f} "
                f"(median={alignment['semantic_to_skill_norm_ratio_median']:.3f})"
            )
        if "deployment_alignment" in values:
            deployment = values["deployment_alignment"]
            print(
                f"deployment alignment: cosine={deployment['cosine_mean']:.4f}, "
                "z_sem_deploy/z_skill norm="
                f"{deployment['semantic_to_skill_norm_ratio_mean']:.3f}"
            )
        if "decorrelation" in values:
            decorrelation = values["decorrelation"]
            print(
                f"decoupling: mean_abs_corr={decorrelation['mean_abs_cross_correlation']:.4f}, "
                f"max_abs_corr={decorrelation['max_abs_cross_correlation']:.4f}"
            )

    for name, value in report.get("probes", {}).items():
        print(f"\n=== {name} ===")
        print(json.dumps(value, indent=2, ensure_ascii=False))

    if report.get("training_summary"):
        print("\n=== Training history summary ===")
        summary = report["training_summary"]
        print(
            f"epochs={summary['epochs_recorded']}, "
            f"phases={summary['phase_counts']}"
        )
        best = summary["best_full_semantic_validation"]
        if best:
            print(
                "best full-semantic validation: "
                f"epoch={best['epoch']}, "
                f"semantic_scale={best['semantic_scale']:.3f}, "
                f"train={best['train_loss']:.5f}, "
                f"val={best['val_loss']:.5f}"
            )
        else:
            print("best full-semantic validation: unavailable")
        selected = summary.get("best_full_semantic_selection")
        if selected:
            norm_ratio = selected["semantic_norm_ratio"]
            norm_text = (
                f", z_sem/z_skill norm={norm_ratio:.3f}"
                if math.isfinite(norm_ratio)
                else ""
            )
            print(
                "best checkpoint selection: "
                f"epoch={selected['epoch']}, "
                f"score={selected['selection_score']:.5f}, "
                f"val={selected['val_loss']:.5f}"
                f"{norm_text}"
            )
        final = summary["final"]
        print(
            "final: "
            f"epoch={final['epoch']}, "
            f"semantic_scale={final['semantic_scale']:.3f}, "
            f"train={final['train_loss']:.5f}, "
            f"val={final['val_loss']:.5f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate exported z_skill, z_sem and z_quality latents."
    )
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=None,
        help="Latent output directory. Defaults to Latent\\Z{profile}.",
    )
    parser.add_argument(
        "--profile",
        choices=ABLATION_PROFILES,
        default=None,
        help="Ablation profile. If omitted, infer it from schema.json or latent-dir.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Legacy single-dataset JSON directory. If supplied, it takes precedence.",
    )
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names under --json-root.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report path. Defaults to latent_validation_report_Z{0..4}.json in latent-dir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.latent_dir is None:
        profile = args.profile or "Z4"
        args.latent_dir = DEFAULT_LATENT_ROOT / profile
    elif not args.latent_dir.is_dir():
        raise NotADirectoryError(args.latent_dir)

    schema = load_schema(args.latent_dir / "schema.json")
    for key, expected in (
        ("checkpoint_format_version", CHECKPOINT_FORMAT_VERSION),
        ("training_protocol_version", TRAINING_PROTOCOL_VERSION),
        ("semantic_fusion_version", SEMANTIC_FUSION_VERSION),
        ("quality_input_version", QUALITY_INPUT_VERSION),
        ("quality_conditioning_version", QUALITY_CONDITIONING_VERSION),
        ("segment_query_version", SEGMENT_QUERY_VERSION),
    ):
        if schema.get(key) != expected:
            raise ValueError(
                f"Stale schema {key}={schema.get(key)!r}; expected {expected!r}. "
                "Retrain before validating deployment readiness."
            )
    profile = args.profile or infer_profile(args.latent_dir, schema)
    config = profile_config(profile)

    # Z0 uses task_goal text but no robot/human annotation labels, so latent
    # health remains valid without loading those labels into this validator.
    needs_labels = bool(
        config["robot_numeric"]
        or config["robot_semantics"]
        or config["human_numeric"]
        or config["human_semantics"]
    )
    if not needs_labels:
        labels: dict[tuple[str, str], dict[str, Any]] = {}
        label_source = {"labels": "not_required_for_Z0"}
    elif args.json_dir is not None:
        if not args.json_dir.is_dir():
            raise NotADirectoryError(args.json_dir)
        labels = load_labels(args.json_dir, profile=profile)
        label_source = {"json_dir": str(args.json_dir)}
    else:
        if not args.json_root.is_dir():
            raise NotADirectoryError(args.json_root)
        labels = load_multi_dataset_labels(args.json_root, args.datasets, profile)
        label_source = {
            "json_root": str(args.json_root),
            "datasets": list(args.datasets),
        }

    splits = {}
    for name in ("train", "val", "test"):
        path = args.latent_dir / f"{name}_latents.npz"
        if path.exists():
            splits[name] = read_split(path, labels)
    if not splits:
        raise FileNotFoundError(
            f"No *_latents.npz files found in {args.latent_dir}"
        )

    report: dict[str, Any] = {
        "latent_dir": str(args.latent_dir),
        "profile": profile,
        "command_interface": {
            "task_context": bool(schema.get("task_context", False)),
            "task_query_version": schema.get("task_query_version"),
            "segment_query_version": schema.get("segment_query_version"),
            "training_protocol_version": schema.get("training_protocol_version"),
            "quality_input_version": schema.get("quality_input_version"),
            "quality_conditioning_version": schema.get("quality_conditioning_version"),
            "deployment_ready": (
                bool(schema.get("task_context", False))
                and schema.get("task_query_version") == "task_goal_only_v1"
                and schema.get("segment_query_version") == SEGMENT_QUERY_VERSION
                and schema.get("quality_input_version") == QUALITY_INPUT_VERSION
                and schema.get("quality_conditioning_version") == QUALITY_CONDITIONING_VERSION
            ),
        },
        **label_source,
        "splits": {
            name: validate_split(split, schema, profile)
            for name, split in splits.items()
        },
        "probes": {},
        "quality_v2": {
            "axes": list(QUALITY_V2_AXES),
            "unknown_policy": "mask_zero_excluded_from_fit_and_metrics",
            "threshold_policy": "validation_only_then_reuse_for_test",
        },
    }

    if "train" in splits:
        # Thresholds for the two quality-v2 heads are kept separately.  When
        # validating an existing directory with only a test split, an earlier
        # validation-produced file may be reused; otherwise test uses a fixed
        # 0.5 fallback and explicitly reports that no validation threshold was
        # available.  It never selects a threshold from test labels.
        thresholds_path = args.latent_dir / "quality_thresholds.json"
        frozen_quality_thresholds: dict[str, dict[str, float]] = {}
        loaded_threshold_record: dict[str, Any] | None = None
        if thresholds_path.exists():
            try:
                candidate = json.loads(
                    thresholds_path.read_text(encoding="utf-8-sig")
                )
                if isinstance(candidate, Mapping):
                    loaded_threshold_record = dict(candidate)
                    heads = candidate.get("heads")
                    if isinstance(heads, Mapping):
                        for head in ("z_quality", "z_skill"):
                            values = heads.get(head, {})
                            if isinstance(values, Mapping):
                                frozen_quality_thresholds[head] = {
                                    str(axis): float(value)
                                    for axis, value in values.items()
                                    if str(axis) in QUALITY_V2_AXES
                                    and math.isfinite(
                                        as_float(value, float("nan"))
                                    )
                                }
                    legacy = candidate.get("thresholds", {})
                    if isinstance(legacy, Mapping):
                        legacy_map = {
                            str(axis): float(value)
                            for axis, value in legacy.items()
                            if str(axis) in QUALITY_V2_AXES
                            and math.isfinite(as_float(value, float("nan")))
                        }
                        # Backward compatibility: old files had one shared
                        # map, so use it for both heads only when a head map is
                        # absent.
                        for head in ("z_quality", "z_skill"):
                            frozen_quality_thresholds.setdefault(head, {})
                            if not frozen_quality_thresholds[head]:
                                frozen_quality_thresholds[head] = dict(legacy_map)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                loaded_threshold_record = None
        if loaded_threshold_record is not None:
            report["quality_thresholds"] = loaded_threshold_record
            report["quality_thresholds_path"] = str(thresholds_path)

        for evaluation_name in ("val", "test"):
            if evaluation_name not in splits:
                continue
            evaluation = splits[evaluation_name]
            report["probes"][f"gait_phase_z_skill_train_to_{evaluation_name}"] = (
                run_gait_probe(splits["train"], evaluation)
            )
            if schema.get("atom_vocab"):
                report["probes"][
                    f"semantic_atoms_z_sem_train_to_{evaluation_name}"
                ] = run_binary_probes(
                    splits["train"],
                    evaluation,
                    "z_sem",
                    "atoms",
                    schema.get("atom_vocab", []),
                    known_key="atoms_known_atoms",
                    unknown_key="atoms_unknown",
                )
                # Keep z_skill as a diagnostic comparison; semantic retrieval
                # is evaluated primarily on z_sem.
                report["probes"][
                    f"semantic_atoms_z_skill_train_to_{evaluation_name}"
                ] = run_binary_probes(
                    splits["train"],
                    evaluation,
                    "z_skill",
                    "atoms",
                    schema.get("atom_vocab", []),
                    known_key="atoms_known_atoms",
                    unknown_key="atoms_unknown",
                )
            if "z_quality" in splits["train"]["data"]:
                for quality_latent_key, quality_name in (
                    ("z_quality", "quality_v2_z_quality"),
                    ("z_skill", "quality_v2_z_skill"),
                ):
                    if evaluation_name == "val":
                        quality_threshold_overrides = None
                        quality_threshold_selection = "validation"
                    else:
                        quality_threshold_overrides = frozen_quality_thresholds.get(
                            quality_latent_key, {}
                        )
                        quality_threshold_selection = "frozen_validation"
                    report["probes"][
                        f"{quality_name}_train_to_{evaluation_name}"
                    ] = run_binary_probes(
                        splits["train"],
                        evaluation,
                        quality_latent_key,
                        "quality_v2_targets",
                        QUALITY_V2_VOCAB,
                        target_key="quality_v2_targets",
                        mask_key="quality_v2_mask",
                        threshold_overrides=quality_threshold_overrides,
                        threshold_selection=quality_threshold_selection,
                    )

                # Preserve the historical feasibility-tag reports.  They use
                # the legacy tag vocabulary and legacy annotation-presence
                # flag; quality-v2 metrics are reported separately above.
                if schema.get("quality_tag_vocab"):
                    report["probes"][
                        f"feasibility_tags_z_quality_train_to_{evaluation_name}"
                    ] = run_binary_probes(
                        splits["train"],
                        evaluation,
                        "z_quality",
                        "tags",
                        schema.get("quality_tag_vocab", []),
                        known_key="legacy_tags_known",
                    )
                    report["probes"][
                        f"feasibility_tags_z_skill_train_to_{evaluation_name}"
                    ] = run_binary_probes(
                        splits["train"],
                        evaluation,
                        "z_skill",
                        "tags",
                        schema.get("quality_tag_vocab", []),
                        known_key="legacy_tags_known",
                    )

                if evaluation_name == "val":
                    heads: dict[str, dict[str, float]] = {}
                    for quality_latent_key, quality_name in (
                        ("z_quality", "quality_v2_z_quality"),
                        ("z_skill", "quality_v2_z_skill"),
                    ):
                        quality_probe = report["probes"].get(
                            f"{quality_name}_train_to_val", {}
                        )
                        heads[quality_latent_key] = {
                            axis: float(values["threshold"])
                            for axis, values in quality_probe.items()
                            if axis in QUALITY_V2_AXES
                            and isinstance(values, Mapping)
                            and values.get("status") == "ok"
                            and math.isfinite(
                                as_float(values.get("threshold"), float("nan"))
                            )
                        }
                    threshold_record = {
                        "version": "quality_v2",
                        "axes": list(QUALITY_V2_AXES),
                        # Legacy compatibility: this is the z_quality map.
                        "thresholds": dict(heads.get("z_quality", {})),
                        "heads": heads,
                        "selection": "validation_split_only",
                        "source_split": "val",
                        "test_used_for_selection": False,
                        "latent_dir": str(args.latent_dir),
                    }
                    report["quality_thresholds"] = threshold_record
                    report["quality_thresholds_path"] = str(thresholds_path)
                    thresholds_path.write_text(
                        json.dumps(
                            threshold_record,
                            indent=2,
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        encoding="utf-8",
                    )
                    # Freeze exactly what validation selected before entering
                    # the test branch.
                    frozen_quality_thresholds = {
                        head: dict(values) for head, values in heads.items()
                    }

            report["splits"][evaluation_name]["shift_from_train"] = {
                key: distribution_shift(splits["train"], evaluation, key)
                for key in ("z_skill", "z_sem", "z_quality")
                if key in splits["train"]["data"] and key in evaluation["data"]
            }

    history_path = args.latent_dir / "history.json"
    if history_path.exists():
        report["history"] = json.loads(
            history_path.read_text(encoding="utf-8-sig")
        )
        if isinstance(report["history"], list):
            report["training_summary"] = summarize_history(report["history"])
    print_report(report)
    report_path = profile_report_path(
        args.report
        or (args.latent_dir / "latent_validation_report.json"),
        profile,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()


