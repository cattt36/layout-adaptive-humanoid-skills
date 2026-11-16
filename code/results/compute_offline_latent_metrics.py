#!/usr/bin/env python3
"""Build paper-facing offline latent metrics for the three Data-HNU actions.

The script deliberately evaluates representation diagnostics only.  It does
not report policy success, rollout executability, or any RL-dependent metric.

Most metrics are recomputed directly from the exported NPZ files.  The
quality-v2 probe is read from the canonical ``latent_validation_report``
because its labels live in the structured annotation JSONs and its decision
thresholds must be selected on validation data.  The script audits those
reports for freshness, validation-only threshold selection, and agreement
with all metrics that can be recomputed locally.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = "1.4"
ACTIONS = ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")
SOURCE_PROFILES = tuple(f"Z{i}" for i in range(5))
QUALITY_AXES = ("torque", "joint_limit", "collision", "contact")
LATENT_KEYS = ("z_skill", "z_sem", "z_sem_deploy", "z_quality")


def find_project_root(script_path: Path) -> Path:
    """Locate Data-HNU after this script is organized below Results/<section>."""
    for candidate in script_path.parents:
        if (candidate / "MOE Framework" / "Latent").is_dir() and (
            candidate / "Results" / "8.1_Offline_Representation"
        ).is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the Data-HNU project root from {script_path}")

MAIN_FIELDS = (
    "action",
    "supervision_level",
    "source_profile",
    "test_trials",
    "test_windows",
    "phase_balanced_accuracy",
    "phase_balanced_accuracy_ci95_low",
    "phase_balanced_accuracy_ci95_high",
    "phase_train_classes",
    "phase_test_classes",
    "phase_min_test_class_windows",
    "phase_min_test_class_trials",
    "phase_label_coverage_status",
    "phase_ci_status",
    "quality_macro_average_precision",
    "quality_macro_auroc",
    "quality_macro_balanced_accuracy",
    "quality_valid_axes",
    "quality_valid_axis_names",
    "deploy_semantic_cosine",
    "skill_quality_mean_abs_correlation",
)

DIAGNOSTIC_FIELDS = (
    "action",
    "supervision_level",
    "source_profile",
    "checkpoint_epoch",
    "validation_reconstruction_loss",
    "validation_smoothness_loss",
    "z_skill_effective_rank",
    "z_skill_collapsed_dimensions",
    "z_skill_train_test_normalized_shift",
    "all_test_latents_finite",
    "semantic_macro_f1",
    "semantic_valid_axes",
)


class MetricDataError(RuntimeError):
    """Raised when source artifacts cannot support an auditable metric."""


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)
    parser = argparse.ArgumentParser(
        description=(
            "Compute the key offline representation metrics for the three "
            "Data-HNU actions and supervision levels L0-L4."
        )
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=project_root / "MOE Framework" / "Latent",
        help=(
            "MOE handoff root containing <action>/Z0...Z4 latent exports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_path.parents[1] / "outputs",
        help="Directory for CSV, JSON, and Markdown outputs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Trial-cluster bootstrap replicates for gait-phase balanced accuracy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=,
        help="Base seed used only for deterministic test-trial bootstrapping.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require_file(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def require_arrays(data: Mapping[str, np.ndarray], names: Iterable[str], path: Path) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise MetricDataError(f"{path} is missing arrays: {', '.join(missing)}")


def effective_rank(z: np.ndarray) -> float:
    values = np.asarray(z, dtype=np.float64)
    if len(values) <= 1:
        return 0.0
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular * singular
    return float((energy.sum() ** 2) / max(float((energy * energy).sum()), 1e-12))


def deployment_cosine(skill: np.ndarray, deploy_semantic: np.ndarray) -> float:
    a = np.asarray(skill, dtype=np.float64)
    b = np.asarray(deploy_semantic, dtype=np.float64)
    if a.shape != b.shape:
        raise MetricDataError(
            f"z_skill and z_sem_deploy must have equal shapes, got {a.shape} and {b.shape}"
        )
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(np.sum(a * b, axis=1) / denominator))


def mean_abs_cross_correlation(skill: np.ndarray, quality: np.ndarray) -> float:
    a = np.asarray(skill, dtype=np.float64)
    b = np.asarray(quality, dtype=np.float64)
    a = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-8)
    b = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-8)
    corr = a.T @ b / max(len(a) - 1, 1)
    return float(np.abs(corr).mean())


def normalized_mean_shift(train_z: np.ndarray, test_z: np.ndarray) -> float:
    a = np.asarray(train_z, dtype=np.float64)
    b = np.asarray(test_z, dtype=np.float64)
    distance = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    pooled_std = math.sqrt(float((a.var(axis=0).mean() + b.var(axis=0).mean()) / 2.0))
    return distance / max(pooled_std, 1e-8)


def nearest_centroid_predictions(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    train_values = np.asarray(train_x, dtype=np.float64)
    test_values = np.asarray(test_x, dtype=np.float64)
    labels = np.asarray(train_labels).astype(str)
    classes = sorted(set(labels.tolist()))
    if not classes:
        raise MetricDataError("The training split has no gait-phase labels")
    centers = {
        label: train_values[labels == label].mean(axis=0)
        for label in classes
        if np.any(labels == label)
    }
    return np.asarray(
        [
            min(
                centers,
                key=lambda label: float(np.linalg.norm(vector - centers[label])),
            )
            for vector in test_values
        ]
    )


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str] | None = None,
) -> float:
    truth = np.asarray(y_true).astype(str)
    prediction = np.asarray(y_pred).astype(str)
    recalls = []
    labels = list(classes) if classes is not None else sorted(set(truth.tolist()))
    for label in labels:
        mask = truth == label
        if not np.any(mask):
            return float("nan")
        recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true).astype(str)
    prediction = np.asarray(y_pred).astype(str)
    scores = []
    for label in sorted(set(truth.tolist())):
        actual = truth == label
        predicted = prediction == label
        tp = int(np.logical_and(actual, predicted).sum())
        fp = int(np.logical_and(~actual, predicted).sum())
        fn = int(np.logical_and(actual, ~predicted).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores)) if scores else float("nan")


def trial_cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trial_ids: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, str, int]:
    if samples <= 0:
        return None, None, "disabled", 0
    truth = np.asarray(y_true).astype(str)
    clusters = np.asarray(trial_ids).astype(str)
    unique_trials = np.unique(clusters)
    if len(unique_trials) < 2:
        return None, None, "fewer_than_two_test_trials", 0
    classes = sorted(set(truth.tolist()))
    class_trial_counts = {
        label: len(np.unique(clusters[truth == label])) for label in classes
    }
    # A trial-cluster bootstrap cannot estimate between-trial uncertainty for
    # a class observed in only one trial.  Returning a narrow interval in that
    # case would create false precision.
    if min(class_trial_counts.values(), default=0) < 2:
        return None, None, "class_observed_in_fewer_than_two_test_trials", 0
    indices = {trial: np.flatnonzero(clusters == trial) for trial in unique_trials}
    statistics = []
    for _ in range(samples):
        sampled_trials = rng.choice(unique_trials, size=len(unique_trials), replace=True)
        sampled_indices = np.concatenate([indices[trial] for trial in sampled_trials])
        statistic = balanced_accuracy(
            y_true[sampled_indices], y_pred[sampled_indices], classes=classes
        )
        if math.isfinite(statistic):
            statistics.append(statistic)
    if not statistics:
        return None, None, "no_valid_bootstrap_replicates", 0
    low, high = np.quantile(np.asarray(statistics), [0.025, 0.975])
    return float(low), float(high), "ok", len(statistics)


def stable_rng(base_seed: int, action: str, profile: str) -> np.random.Generator:
    payload = f"{base_seed}:{action}:{profile}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(offset)


def split_fingerprint(data: Mapping[str, np.ndarray]) -> str:
    fields = ("trial_id", "segment_id", "start_index", "center_index", "gait_phase")
    require_arrays(data, fields, Path("<in-memory split>"))
    rows = sorted(
        "\x1f".join(str(data[field][index]) for field in fields)
        for index in range(len(data["trial_id"]))
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(name: str, calculated: float, reported: Any, tolerance: float = 1e-6) -> None:
    expected = finite_number(reported)
    if expected is None:
        raise MetricDataError(f"Canonical report is missing finite metric: {name}")
    if not math.isclose(calculated, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise MetricDataError(
            f"Metric mismatch for {name}: recomputed={calculated:.10g}, "
            f"reported={expected:.10g}"
        )


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Load checkpoint metadata with PyTorch's restricted weights-only loader."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project environment has torch
        raise MetricDataError(
            "PyTorch is required to identify the exact exported checkpoint epoch"
        ) from exc

    torch.serialization.add_safe_globals([PosixPath, WindowsPath])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    result = {
        "epoch": int(checkpoint["epoch"]),
        "best_val_loss": finite_number(checkpoint.get("best_val_loss")),
        "selection_score": finite_number(checkpoint.get("selection_score")),
        "checkpoint_fallback": bool(checkpoint.get("checkpoint_fallback", False)),
    }
    del checkpoint
    gc.collect()
    return result


def quality_metrics(report: Mapping[str, Any], profile: str) -> dict[str, Any]:
    probe = nested(report, "probes", "quality_v2_z_quality_train_to_test")
    if not isinstance(probe, Mapping):
        if profile == "Z0":
            return {
                "average_precision": None,
                "roc_auc": None,
                "balanced_accuracy": None,
                "f1": None,
                "valid_axes": [],
                "validation_threshold_axes": [],
                "fixed_default_threshold_axes": [],
                "support": 0,
                "unknown": 0,
            }
        raise MetricDataError(f"{profile} is missing the test quality-v2 z_quality probe")

    valid_axes = [
        axis
        for axis in QUALITY_AXES
        if isinstance(probe.get(axis), Mapping) and probe[axis].get("status") == "ok"
    ]
    macro = probe.get("_macro", {})
    if not isinstance(macro, Mapping):
        macro = {}

    metrics = {
        "average_precision": finite_number(macro.get("average_precision")),
        "roc_auc": finite_number(macro.get("roc_auc")),
        "balanced_accuracy": finite_number(macro.get("balanced_accuracy")),
        "f1": finite_number(macro.get("f1")),
        "valid_axes": valid_axes,
        "validation_threshold_axes": [],
        "fixed_default_threshold_axes": [],
        "support": int(macro.get("support", 0) or 0),
        "unknown": int(macro.get("unknown", 0) or 0),
    }

    reported_count = int(macro.get("n_labels", 0) or 0)
    if reported_count != len(valid_axes):
        raise MetricDataError(
            f"{profile} quality macro says n_labels={reported_count}, "
            f"but {len(valid_axes)} axes have status='ok'"
        )

    if valid_axes:
        threshold_record = report.get("quality_thresholds", {})
        if not isinstance(threshold_record, Mapping):
            raise MetricDataError(f"{profile} has quality metrics but no threshold record")
        if threshold_record.get("test_used_for_selection") is not False:
            raise MetricDataError(
                f"{profile} quality thresholds do not prove test_used_for_selection=false"
            )
        if threshold_record.get("source_split") != "val":
            raise MetricDataError(
                f"{profile} quality thresholds were not selected on the validation split"
            )
        for axis in valid_axes:
            threshold_source = probe[axis].get("threshold_source")
            if threshold_source == "validation_override":
                metrics["validation_threshold_axes"].append(axis)
            elif threshold_source == "fixed_default_no_validation_threshold":
                threshold = finite_number(probe[axis].get("threshold"))
                if threshold is None or not math.isclose(threshold, 0.5):
                    raise MetricDataError(
                        f"{profile}/{axis} claims a fixed default threshold but it is not 0.5"
                    )
                metrics["fixed_default_threshold_axes"].append(axis)
            else:
                raise MetricDataError(
                    f"{profile}/{axis} has unapproved test threshold source "
                    f"{threshold_source!r}"
                )

        for key in ("average_precision", "roc_auc", "balanced_accuracy", "f1"):
            values = [finite_number(probe[axis].get(key)) for axis in valid_axes]
            if any(value is None for value in values):
                raise MetricDataError(f"{profile} has an invalid per-axis quality {key}")
            calculated = float(np.mean(np.asarray(values, dtype=np.float64)))
            assert_close(f"{profile}.quality_macro.{key}", calculated, macro.get(key))

    return metrics


def semantic_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    probe = nested(report, "probes", "semantic_atoms_z_sem_train_to_test")
    if not isinstance(probe, Mapping):
        return {"f1": None, "valid_axes": 0}
    macro = probe.get("_macro", {})
    if not isinstance(macro, Mapping):
        return {"f1": None, "valid_axes": 0}
    return {
        "f1": finite_number(macro.get("f1")),
        "valid_axes": int(macro.get("n_labels", 0) or 0),
    }


def history_at_checkpoint(history: Sequence[Mapping[str, Any]], epoch: int) -> Mapping[str, Any]:
    for record in history:
        if isinstance(record, Mapping) and int(record.get("epoch", -1)) == epoch:
            return record
    raise MetricDataError(f"history.json has no entry for checkpoint epoch {epoch}")


def build_row(
    latent_root: Path,
    action: str,
    profile: str,
    bootstrap_samples: int,
    seed: int,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    profile_dir = latent_root / action / profile
    if not profile_dir.is_dir():
        raise NotADirectoryError(profile_dir)

    report_path = require_file(profile_dir / f"latent_validation_report_{profile}.json")
    schema_path = require_file(profile_dir / "schema.json")
    manifest_path = require_file(profile_dir / "split_manifest.json")
    history_path = require_file(profile_dir / "history.json")
    checkpoint_path = require_file(profile_dir / "best_motion_latent.pt")
    split_paths = {
        split: require_file(profile_dir / f"{split}_latents.npz")
        for split in ("train", "val", "test")
    }

    newest_latent_time = max(path.stat().st_mtime for path in split_paths.values())
    if report_path.stat().st_mtime + 1e-6 < newest_latent_time:
        raise MetricDataError(f"{report_path} is older than its latent export")

    report = load_json(report_path)
    schema = load_json(schema_path)
    manifest = load_json(manifest_path)
    history = load_json(history_path)
    if report.get("profile") != profile or schema.get("profile") != profile:
        raise MetricDataError(f"Profile metadata mismatch under {profile_dir}")
    report_datasets = report.get("datasets")
    # Z0 intentionally skips annotation loading, so its validator has no
    # ``datasets`` field.  Its action identity is still audited below through
    # the split manifest and the exported trial IDs.
    if profile != "Z0" and report_datasets != [action]:
        raise MetricDataError(
            f"{report_path} datasets={report_datasets!r}, expected [{action!r}]"
        )
    if profile == "Z0" and report_datasets not in (None, [action]):
        raise MetricDataError(
            f"{report_path} has unexpected Z0 datasets={report_datasets!r}"
        )
    if manifest.get("split_mode") not in {"subject", "file"}:  # file = explicit trial-level held-out split
        raise MetricDataError(
            f"{manifest_path} is not a subject-disjoint or explicit held-out split manifest"
        )
    dataset_counts = manifest.get("dataset_counts", {})
    if not isinstance(dataset_counts, Mapping) or set(dataset_counts) != {action}:
        raise MetricDataError(
            f"{manifest_path} dataset identity does not match action {action!r}"
        )
    if not isinstance(history, list):
        raise MetricDataError(f"{history_path} must contain a list of epochs")

    splits = {name: load_npz(path) for name, path in split_paths.items()}
    for name, data in splits.items():
        require_arrays(
            data,
            ("z_skill", "z_sem", "z_sem_deploy", "z_quality", "trial_id", "gait_phase"),
            split_paths[name],
        )

    trial_sets = {
        name: set(np.asarray(data["trial_id"]).astype(str).tolist())
        for name, data in splits.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = trial_sets[left] & trial_sets[right]
        if overlap:
            raise MetricDataError(
                f"{action}/{profile} has trial leakage between {left} and {right}: "
                f"{sorted(overlap)}"
            )

    train = splits["train"]
    test = splits["test"]
    test_labels = np.asarray(test["gait_phase"]).astype(str)
    phase_prediction = nearest_centroid_predictions(
        train["z_skill"], train["gait_phase"], test["z_skill"]
    )
    phase_accuracy = float(np.mean(phase_prediction == test_labels))
    phase_ba = balanced_accuracy(test_labels, phase_prediction)
    phase_f1 = macro_f1(test_labels, phase_prediction)
    ci_low, ci_high, ci_status, ci_valid_replicates = trial_cluster_bootstrap_ci(
        test_labels,
        phase_prediction,
        test["trial_id"],
        bootstrap_samples,
        stable_rng(seed, action, profile),
    )
    train_phase_classes = sorted(set(np.asarray(train["gait_phase"]).astype(str).tolist()))
    test_phase_classes = sorted(set(test_labels.tolist()))
    unseen_test_phase_classes = sorted(set(test_phase_classes) - set(train_phase_classes))
    missing_test_phase_classes = sorted(set(train_phase_classes) - set(test_phase_classes))
    if unseen_test_phase_classes:
        phase_label_coverage_status = "test_contains_classes_unseen_in_train"
    elif missing_test_phase_classes:
        phase_label_coverage_status = "test_omits_training_classes"
    else:
        phase_label_coverage_status = "matched"
    phase_class_window_support = {
        label: int(np.sum(test_labels == label)) for label in test_phase_classes
    }
    test_trial_labels = np.asarray(test["trial_id"]).astype(str)
    phase_class_trial_support = {
        label: len(np.unique(test_trial_labels[test_labels == label]))
        for label in test_phase_classes
    }

    report_phase = nested(
        report, "probes", "gait_phase_z_skill_train_to_test"
    )
    if not isinstance(report_phase, Mapping):
        raise MetricDataError(f"{report_path} has no test gait-phase probe")
    assert_close(f"{action}/{profile}.phase_accuracy", phase_accuracy, report_phase.get("accuracy"))
    assert_close(
        f"{action}/{profile}.phase_balanced_accuracy",
        phase_ba,
        report_phase.get("balanced_accuracy"),
    )

    deploy_cosine = deployment_cosine(test["z_skill"], test["z_sem_deploy"])
    cross_corr = mean_abs_cross_correlation(test["z_skill"], test["z_quality"])
    rank = effective_rank(test["z_skill"])
    shift = normalized_mean_shift(train["z_skill"], test["z_skill"])
    collapsed = int(np.sum(np.std(test["z_skill"].astype(np.float64), axis=0) < 0.01))
    all_finite = bool(
        all(np.isfinite(test[key]).all() for key in LATENT_KEYS if key in test)
    )

    report_test = nested(report, "splits", "test")
    if not isinstance(report_test, Mapping):
        raise MetricDataError(f"{report_path} has no test split summary")
    assert_close(
        f"{action}/{profile}.deploy_cosine",
        deploy_cosine,
        nested(report_test, "deployment_alignment", "cosine_mean"),
    )
    assert_close(
        f"{action}/{profile}.cross_correlation",
        cross_corr,
        nested(report_test, "decorrelation", "mean_abs_cross_correlation"),
    )
    assert_close(
        f"{action}/{profile}.effective_rank",
        rank,
        nested(report_test, "latents", "z_skill", "effective_rank"),
    )
    assert_close(
        f"{action}/{profile}.normalized_shift",
        shift,
        nested(report_test, "shift_from_train", "z_skill", "normalized_mean_distance"),
    )
    reported_rows = int(report_test.get("rows", -1))
    if reported_rows != len(test["z_skill"]):
        raise MetricDataError(
            f"{action}/{profile} report rows={reported_rows}, NPZ rows={len(test['z_skill'])}"
        )

    quality = quality_metrics(report, profile)
    semantic = semantic_metrics(report)
    checkpoint = checkpoint_metadata(checkpoint_path)
    checkpoint_record = history_at_checkpoint(history, checkpoint["epoch"])
    checkpoint_val = checkpoint_record.get("val", {})
    if not isinstance(checkpoint_val, Mapping):
        raise MetricDataError(f"Invalid val history at epoch {checkpoint['epoch']}")
    if checkpoint["best_val_loss"] is not None:
        assert_close(
            f"{action}/{profile}.checkpoint_val_loss",
            checkpoint["best_val_loss"],
            checkpoint_val.get("loss"),
        )

    row = {
        "action": action,
        "supervision_level": f"L{profile[1:]}",
        "source_profile": profile,
        "split_protocol": manifest.get("split_strategy", "subject-disjoint"),
        "train_trials": len(trial_sets["train"]),
        "validation_trials": len(trial_sets["val"]),
        "test_trials": len(trial_sets["test"]),
        "train_windows": len(train["z_skill"]),
        "validation_windows": len(splits["val"]["z_skill"]),
        "test_windows": len(test["z_skill"]),
        "phase_accuracy": phase_accuracy,
        "phase_balanced_accuracy": phase_ba,
        "phase_macro_f1": phase_f1,
        "phase_balanced_accuracy_ci95_low": ci_low,
        "phase_balanced_accuracy_ci95_high": ci_high,
        "phase_train_classes": len(train_phase_classes),
        "phase_test_classes": len(test_phase_classes),
        "phase_test_class_names": ";".join(test_phase_classes),
        "phase_missing_test_class_names": ";".join(
            missing_test_phase_classes
        ),
        "phase_unseen_test_class_names": ";".join(unseen_test_phase_classes),
        "phase_label_coverage_status": phase_label_coverage_status,
        "phase_min_test_class_windows": min(phase_class_window_support.values()),
        "phase_min_test_class_trials": min(phase_class_trial_support.values()),
        "phase_class_window_support": phase_class_window_support,
        "phase_class_trial_support": phase_class_trial_support,
        "phase_ci_status": ci_status,
        "phase_ci_valid_replicates": ci_valid_replicates,
        "phase_ci_method": (
            f"percentile bootstrap over test trials ({bootstrap_samples} requested replicates)"
            if ci_status != "disabled"
            else "not_computed"
        ),
        "quality_macro_average_precision": quality["average_precision"],
        "quality_macro_auroc": quality["roc_auc"],
        "quality_macro_balanced_accuracy": quality["balanced_accuracy"],
        "quality_macro_f1": quality["f1"],
        "quality_valid_axes": len(quality["valid_axes"]),
        "quality_valid_axis_names": ";".join(quality["valid_axes"]),
        "quality_validation_threshold_axis_names": ";".join(
            quality["validation_threshold_axes"]
        ),
        "quality_fixed_default_threshold_axis_names": ";".join(
            quality["fixed_default_threshold_axes"]
        ),
        "quality_axis_universe": ";".join(QUALITY_AXES),
        "quality_axis_support_sum": quality["support"],
        "quality_axis_unknown_sum": quality["unknown"],
        "deploy_semantic_cosine": deploy_cosine,
        "skill_quality_mean_abs_correlation": cross_corr,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_fallback": checkpoint["checkpoint_fallback"],
        "validation_reconstruction_loss": finite_number(checkpoint_val.get("recon")),
        "validation_smoothness_loss": finite_number(checkpoint_val.get("smooth")),
        "z_skill_effective_rank": rank,
        "z_skill_nominal_dimension": int(test["z_skill"].shape[1]),
        "z_skill_collapsed_dimensions": collapsed,
        "z_skill_train_test_normalized_shift": shift,
        "all_test_latents_finite": all_finite,
        "semantic_macro_f1": semantic["f1"],
        "semantic_valid_axes": semantic["valid_axes"],
        "source_report": report_path.relative_to(project_root).as_posix(),
        "source_report_sha256": sha256_file(report_path),
    }
    fingerprints = {name: split_fingerprint(data) for name, data in splits.items()}
    return row, fingerprints


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def format_number(value: Any, digits: int = 3) -> str:
    number = finite_number(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def format_percent(value: Any, digits: int = 1) -> str:
    number = finite_number(value)
    return "N/A" if number is None else f"{100.0 * number:.{digits}f}%"


def phase_with_ci(row: Mapping[str, Any]) -> str:
    point = finite_number(row.get("phase_balanced_accuracy"))
    low = finite_number(row.get("phase_balanced_accuracy_ci95_low"))
    high = finite_number(row.get("phase_balanced_accuracy_ci95_high"))
    if point is None:
        return "N/A"
    coverage_note = (
        "; label mismatch"
        if row.get("phase_label_coverage_status") != "matched"
        else ""
    )
    if low is None or high is None:
        return f"{format_percent(point)} (CI N/A{coverage_note})"
    return (
        f"{format_percent(point)} [{format_percent(low)}, {format_percent(high)}]"
        f"{coverage_note}"
    )


def best_level(
    rows: Sequence[Mapping[str, Any]], key: str, maximize: bool = True
) -> tuple[str, float] | None:
    candidates = [
        (str(row["supervision_level"]), finite_number(row.get(key)))
        for row in rows
    ]
    valid = [(level, value) for level, value in candidates if value is not None]
    if not valid:
        return None
    return (max(valid, key=lambda item: item[1]) if maximize else min(valid, key=lambda item: item[1]))  # type: ignore[arg-type,return-value]


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    source_latent_root: str,
) -> str:
    source_cli = ".\\" + source_latent_root.replace("/", "\\")
    primary = [row for row in rows if row["supervision_level"] == "L4"]
    lines = [
        "# Data-HNU offline latent metrics",
        "",
        f"Source snapshot: `{source_latent_root}` (3 actions x 5 profiles).",
        "",
        "This report contains representation diagnostics available before RL. It does not "
        "measure closed-loop task success or rollout-calibrated executability.",
        "",
        "## Full-supervision L4 results",
        "",
        "| Action | Test trials / windows | Phase BA (trial-bootstrap 95% CI) | Test/train phase classes (minimum support) | Quality mAP / AUROC (valid axes) | Deploy semantic cosine | Skill-quality mean abs corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        quality = format_percent(row.get("quality_macro_average_precision"))
        quality += " / " + format_percent(row.get("quality_macro_auroc"))
        quality += f" ({row['quality_valid_axes']}/4)"
        min_windows = int(row["phase_min_test_class_windows"])
        min_trials = int(row["phase_min_test_class_trials"])
        window_label = "window" if min_windows == 1 else "windows"
        trial_label = "trial" if min_trials == 1 else "trials"
        lines.append(
            "| {action} | {trials} / {windows} | {phase} | {classes} ({min_windows} {window_label} / {min_trials} {trial_label}) | {quality} | {deploy} | {corr} |".format(
                action=row["action"],
                trials=row["test_trials"],
                windows=row["test_windows"],
                phase=phase_with_ci(row),
                classes=f"{row['phase_test_classes']}/{row['phase_train_classes']}",
                min_windows=min_windows,
                window_label=window_label,
                min_trials=min_trials,
                trial_label=trial_label,
                quality=quality,
                deploy=format_number(row.get("deploy_semantic_cosine")),
                corr=format_number(row.get("skill_quality_mean_abs_correlation")),
            )
        )

    lines.extend(
        [
            "",
        "Phase BA is the subject-held-out nearest-centroid probe on z_skill. Quality mAP "
            "and AUROC are threshold-free macro metrics of quality-v2 probes on z_quality. "
            "The parenthesized count is essential because constant or unknown axes "
            "are excluded rather than silently scored as zero.",
            "",
            "## L0-L4 offline ablation",
            "",
            "| Action | Level | Phase BA | Quality mAP / AUROC | Valid quality axes | Deploy cosine | Skill-quality mean abs corr |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        axis_names = str(row.get("quality_valid_axis_names") or "-")
        quality = format_percent(row.get("quality_macro_average_precision"))
        quality += " / " + format_percent(row.get("quality_macro_auroc"))
        lines.append(
            "| {action} | {level} | {phase} | {quality} | {axes} ({count}/4) | {deploy} | {corr} |".format(
                action=row["action"],
                level=row["supervision_level"],
                phase=format_percent(row.get("phase_balanced_accuracy")),
                quality=quality,
                axes=axis_names,
                count=row["quality_valid_axes"],
                deploy=format_number(row.get("deploy_semantic_cosine")),
                corr=format_number(row.get("skill_quality_mean_abs_correlation")),
            )
        )

    lines.extend(
        [
            "",
            "No composite score is calculated and L4 is not assumed to be optimal. Per-metric "
            "extrema are listed only to expose the current behavior:",
            "",
        ]
    )
    for action in ACTIONS:
        action_rows = [row for row in rows if row["action"] == action]
        phase = best_level(action_rows, "phase_balanced_accuracy", maximize=True)
        quality = best_level(action_rows, "quality_macro_average_precision", maximize=True)
        deploy = best_level(action_rows, "deploy_semantic_cosine", maximize=True)
        decor = best_level(action_rows, "skill_quality_mean_abs_correlation", maximize=False)
        fragments = []
        if phase:
            fragments.append(f"phase BA {phase[0]}={format_percent(phase[1])}")
        if quality:
            fragments.append(f"quality mAP {quality[0]}={format_percent(quality[1])}")
        if deploy:
            fragments.append(f"deploy cosine {deploy[0]}={format_number(deploy[1])}")
        if decor:
            fragments.append(f"lowest cross-correlation {decor[0]}={format_number(decor[1])}")
        lines.append(f"- {action}: " + "; ".join(fragments) + ".")

    lines.extend(
        [
            "",
            "## Latent health diagnostics",
            "",
            "| Action | Level | Val reconstruction | Effective rank / 16 | Collapsed dims | Train-test shift |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {action} | {level} | {recon} | {rank} / 16 | {collapsed} | {shift} |".format(
                action=row["action"],
                level=row["supervision_level"],
                recon=format_number(row.get("validation_reconstruction_loss"), 4),
                rank=format_number(row.get("z_skill_effective_rank"), 2),
                collapsed=row["z_skill_collapsed_dimensions"],
                shift=format_number(row.get("z_skill_train_test_normalized_shift")),
            )
        )

    semantic_ready = [row for row in rows if int(row.get("semantic_valid_axes", 0)) >= 2]
    incomplete_quality = [
        row
        for row in rows
        if row["supervision_level"] != "L0" and int(row["quality_valid_axes"]) < 4
    ]
    fixed_default_axes = sum(
        len(
            [
                axis
                for axis in str(row.get("quality_fixed_default_threshold_axis_names") or "").split(";")
                if axis
            ]
        )
        for row in rows
    )
    phase_ci_unavailable_actions = [
        row["action"] for row in primary if row.get("phase_ci_status") != "ok"
    ]
    phase_label_mismatch_rows = [
        row for row in primary if row.get("phase_label_coverage_status") != "matched"
    ]
    lines.extend(
        [
            "",
            "## Paper-use boundary",
            "",
            "- Use the L4 table as a compact representation sanity check; use the L0-L4 table "
            "as an offline ablation or Supplement table. These values cannot support a claim "
            "of safe execution or higher task success.",
            f"- Phase uncertainty resamples held-out test trials ({bootstrap_samples} replicates); "
            "it does not include variation across training seeds. Final paper claims still need "
            "3-5 independently trained seeds.",
            (
                "- Trial-bootstrap phase intervals are unavailable for "
                + ", ".join(phase_ci_unavailable_actions)
                + " because at least one test phase occurs in only one trial. Their point BA "
                "is reproducible but not statistically stable enough for a headline claim."
                if phase_ci_unavailable_actions
                else "- Every L4 test phase occurs in at least two trials, so all reported phase intervals are identifiable."
            ),
            (
                "- Only "
                + ", ".join(
                    str(row["action"])
                    for row in primary
                    if row.get("phase_label_coverage_status") == "matched"
                )
                + " has a matched train/test phase label universe at L4. "
                + "; ".join(
                    f"{row['action']}={row['phase_label_coverage_status']}"
                    for row in phase_label_mismatch_rows
                )
                + ". Rebuild a phase-stratified group split before using the mismatched BA values as paper evidence."
                if phase_label_mismatch_rows
                else "- All L4 train/test phase label universes match."
            ),
            f"- {len(incomplete_quality)} of the 12 supervised action-level rows have fewer than "
            "four evaluable quality axes. Always report the valid-axis count beside quality mAP/AUROC.",
            "- L0 quality is N/A by design because L0 does not load robot/human annotation labels.",
            f"- {fixed_default_axes} evaluable action-level quality axes had no usable validation "
            "threshold and therefore use the pre-specified 0.5 threshold. Their mAP remains "
            "threshold-free, but their balanced accuracy/F1 should be read with this caveat.",
            (
                f"- Semantic macro F1 is not promoted to the main table: only {len(semantic_ready)} "
                "of 15 rows have at least two evaluable semantic axes. A one-axis macro score "
                "would be misleading."
            ),
            "- The current quality labels are offline heuristic evidence, not rollout-calibrated "
            "Q_exec or p(success). AUROC/AUPRC/Brier/ECE/NLL for executability must wait for rollout labels.",
            "",
            "## Reproduce",
            "",
            "From the project root:",
            "",
            "```powershell",
            f".\\.venv\\Scripts\\python.exe .\\Results\\8.1_Offline_Representation\\scripts\\compute_offline_latent_metrics.py --latent-root '{source_cli}'",
            "```",
            "",
            "The script aborts on split leakage, stale validation reports, test-selected quality "
            "thresholds, or disagreement between recomputed and canonical metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    latent_root = args.latent_root.resolve()
    output_dir = args.output_dir.resolve()
    project_root = find_project_root(Path(__file__).resolve())
    if not latent_root.is_dir():
        raise NotADirectoryError(latent_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, dict[str, dict[str, str]]] = {}
    for action in ACTIONS:
        fingerprints[action] = {}
        for profile in SOURCE_PROFILES:
            row, profile_fingerprints = build_row(
                latent_root=latent_root,
                action=action,
                profile=profile,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                project_root=project_root,
            )
            rows.append(row)
            fingerprints[action][profile] = profile_fingerprints
            print(
                f"{action}/{profile}: phase_BA={row['phase_balanced_accuracy']:.4f}, "
                f"quality_mAP={format_number(row['quality_macro_average_precision'], 4)}, "
                f"deploy_cos={row['deploy_semantic_cosine']:.4f}"
            )

    for action in ACTIONS:
        baseline = fingerprints[action]["Z0"]
        for profile in SOURCE_PROFILES[1:]:
            if fingerprints[action][profile] != baseline:
                raise MetricDataError(
                    f"{action}: sample membership/order differs between Z0 and {profile}; "
                    "the L0-L4 comparison is not paired"
                )

    primary_rows = [row for row in rows if row["supervision_level"] == "L4"]
    write_csv(output_dir / "offline_latent_metrics.csv", rows, MAIN_FIELDS)
    write_csv(output_dir / "offline_latent_primary_L4.csv", primary_rows, MAIN_FIELDS)
    write_csv(output_dir / "offline_latent_diagnostics.csv", rows, DIAGNOSTIC_FIELDS)

    payload = {
        "schema_version": "data_hnu_offline_latent_metrics_v1",
        "script_version": SCRIPT_VERSION,
        "scope": "offline_representation_diagnostics_without_RL",
        "source_latent_root": latent_root.relative_to(project_root).as_posix(),
        "actions": list(ACTIONS),
        "supervision_levels": [f"L{i}" for i in range(5)],
        "source_profiles": list(SOURCE_PROFILES),
        "bootstrap": {
            "metric": "gait_phase_balanced_accuracy",
            "unit": "test_trial",
            "replicates": args.bootstrap_samples,
            "seed": args.seed,
            "interval": "percentile_95",
        },
        "integrity_checks": {
            "processed_all_3_actions_x_5_levels": len(rows) == 15,
            "train_validation_test_trials_disjoint": True,
            "sample_membership_identical_across_levels_within_action": True,
            "canonical_reports_newer_than_latent_exports": True,
            "recomputed_metrics_match_canonical_reports": True,
            "quality_thresholds_selected_on_validation_not_test": True,
        },
        "metric_directions": {
            "phase_balanced_accuracy": "higher_is_better",
            "quality_macro_average_precision": "higher_is_better",
            "quality_macro_auroc": "higher_is_better",
            "quality_macro_balanced_accuracy": "higher_is_better",
            "deploy_semantic_cosine": "higher_is_better",
            "skill_quality_mean_abs_correlation": "lower_is_more_disentangled",
            "validation_reconstruction_loss": "lower_is_better_within_this_preprocessing",
            "z_skill_effective_rank": "diagnostic_not_monotonic",
            "z_skill_train_test_normalized_shift": "lower_is_less_shift",
        },
        "rows": rows,
    }
    with (output_dir / "offline_latent_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    (output_dir / "offline_latent_metrics_summary.md").write_text(
        build_summary(
            rows,
            args.bootstrap_samples,
            latent_root.relative_to(project_root).as_posix(),
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {len(rows)} rows and paper-facing summaries to {output_dir}")


if __name__ == "__main__":
    main()
