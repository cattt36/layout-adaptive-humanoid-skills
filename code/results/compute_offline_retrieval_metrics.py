#!/usr/bin/env python3
"""Compute auditable offline retrieval metrics for Data-HNU latents.

Two deliberately separate protocols are reported:

1. ``skill_nn`` measures cross-subject representation structure by using each
   held-out test ``z_skill`` segment as a query into a training-only ``z_skill``
   gallery.  This is the recommended representation metric for Section 8.1.
2. ``deployment_query_to_skill`` encodes the deployment-observable segment
   query (task goal plus gait phase) and retrieves from the same training-only
   ``z_skill`` gallery.  It diagnoses the current query-to-skill interface; it
   is not silently substituted for the representation metric.

Relevance is a frozen, action-scoped canonical subskill mapped from gait phase.
The taxonomy is evaluation-only and deliberately keeps ThreeStepPoleRun and
FiveStepPoleRun separate even when their phase names are analogous.  Test queries
with no relevant item in the training gallery remain in the denominator as
misses. Their fraction is also exposed as ``gallery_label_coverage`` so
label/library coverage cannot be mistaken for model retrieval quality.

No quality filtering, feasibility bonus, test-selected threshold, or RL output
is used.  Each motion segment is one query; overlapping windows are averaged
before ranking.  Confidence intervals resample held-out test trials rather than
treating windows or segments as independent.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = "1.6"
ACTIONS = ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")
SOURCE_PROFILES = tuple(f"Z{i}" for i in range(5))
PROTOCOLS = ("skill_nn", "deployment_query_to_skill")
EXPECTED_SEGMENT_QUERY_VERSION = "segment_query_v3_observable"
EXPECTED_CANONICAL_SCHEMA = "canonical_subskills_v1"
RELEVANCE_DEFINITION = "frozen action-scoped canonical_subskill_id mapped from gait_phase"


def find_project_root(script_path: Path) -> Path:
    """Locate Data-HNU after this script is organized below Results/<section>."""
    for candidate in script_path.parents:
        if (candidate / "MOE Framework" / "Latent").is_dir() and (
            candidate / "Results" / "8.1_Offline_Representation"
        ).is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the Data-HNU project root from {script_path}")

METRIC_FIELDS = (
    "action",
    "supervision_level",
    "source_profile",
    "protocol",
    "gallery_split",
    "query_split",
    "gallery_trials",
    "query_trials",
    "gallery_subjects",
    "query_subjects",
    "subject_disjoint",
    "label_protocol_version",
    "relevance_definition",
    "gallery_segments",
    "query_segments",
    "gallery_unique_labels",
    "query_unique_labels",
    "query_labels_unseen_in_gallery",
    "queries_without_relevant_gallery_item",
    "gallery_label_coverage",
    "mean_relevant_gallery_items",
    "random_r_at_1",
    "random_r_at_5",
    "random_mrr",
    "r_at_1",
    "r_at_1_ci95_low",
    "r_at_1_ci95_high",
    "r_at_5",
    "r_at_5_ci95_low",
    "r_at_5_ci95_high",
    "mrr",
    "mrr_ci95_low",
    "mrr_ci95_high",
    "covered_query_r_at_1",
    "covered_query_r_at_5",
    "covered_query_mrr",
    "bootstrap_unit",
    "bootstrap_replicates",
)

MACRO_FIELDS = (
    "supervision_level",
    "source_profile",
    "protocol",
    "actions",
    "task_macro_gallery_label_coverage",
    "task_macro_r_at_1",
    "task_macro_r_at_5",
    "task_macro_mrr",
    "task_macro_covered_query_r_at_1",
    "task_macro_covered_query_r_at_5",
    "task_macro_covered_query_mrr",
)

QUERY_FIELDS = (
    "action",
    "supervision_level",
    "source_profile",
    "protocol",
    "trial_id",
    "segment_id",
    "gait_phase",
    "canonical_subskill_id",
    "strict_phase_event_label",
    "relevant_in_gallery",
    "relevant_gallery_items",
    "first_relevant_rank",
    "hit_at_1",
    "hit_at_5",
    "reciprocal_rank",
)


class RetrievalDataError(RuntimeError):
    """Raised when retrieval artifacts cannot support an auditable metric."""


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=project_root / "MOE Framework" / "Latent",
        help=(
            "MOE handoff root containing <action>/Z0...Z4 latent exports."
        ),
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=project_root / "LLM_annotation",
        help="Structured trial JSON root used only to derive evaluation labels.",
    )
    parser.add_argument(
        "--framework-root",
        type=Path,
        default=project_root / "MOE Framework",
        help="Runtime containing the v3 segment serializer and query encoder.",
    )
    parser.add_argument(
        "--canonical-subskills",
        type=Path,
        default=project_root / "Data" / "canonical_subskills_v1.json",
        help="Frozen evaluation-only mapping from action and gait phase to subskill ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_path.parents[1] / "outputs",
        help="Destination for retrieval CSV, JSON, and Markdown reports.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Test-trial cluster-bootstrap replicates for R@K and MRR.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=,
        help="Base seed used only for deterministic test-trial bootstrapping.",
    )
    return parser.parse_args()


def load_runtime(project_root: Path, framework_root: Path) -> tuple[Any, Any, Any]:
    """Load the exact project runtime used to serialize v3 segment queries."""

    data_root = project_root / "Data"
    for path in (data_root, framework_root):
        if not path.is_dir():
            raise NotADirectoryError(path)
    # Insert Data first and framework second so the framework's v3
    # segment_protocol wins while its schema helpers can still import Data.
    for path in (data_root, framework_root):
        text = str(path.resolve())
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)

    build_index_module = importlib.import_module("build_latent_index")
    retriever_module = importlib.import_module("retrieve_skill")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project environment supplies torch
        raise RetrievalDataError("PyTorch is required to encode deployment queries") from exc

    version = getattr(retriever_module, "SEGMENT_QUERY_VERSION", None)
    if version != EXPECTED_SEGMENT_QUERY_VERSION:
        raise RetrievalDataError(
            f"Loaded segment query protocol {version!r}; expected "
            f"{EXPECTED_SEGMENT_QUERY_VERSION!r}. Check --framework-root."
        )
    return build_index_module, retriever_module, torch


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> Any:
    require_file(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_canonical_subskills(path: Path) -> dict[str, Any]:
    """Load and validate the frozen action-scoped evaluation taxonomy."""

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise RetrievalDataError(f"Canonical subskill protocol is not an object: {path}")
    if payload.get("schema_version") != EXPECTED_CANONICAL_SCHEMA:
        raise RetrievalDataError(
            f"Canonical subskill schema mismatch in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("evaluation_only") is not True:
        raise RetrievalDataError(f"Canonical subskills must be evaluation-only: {path}")
    if payload.get("label_scope") != "within_action":
        raise RetrievalDataError(f"Canonical labels must be action-scoped: {path}")
    if payload.get("unmapped_phase_policy") != "error":
        raise RetrievalDataError(
            f"Canonical protocol must reject unmapped phases: {path}"
        )
    actions = payload.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(ACTIONS):
        raise RetrievalDataError(
            f"Canonical protocol must define exactly {list(ACTIONS)}: {path}"
        )

    action_ids: dict[str, set[str]] = {}
    all_ids: set[str] = set()
    for action in ACTIONS:
        entry = actions.get(action)
        if not isinstance(entry, Mapping):
            raise RetrievalDataError(f"Invalid canonical action entry {action}: {path}")
        namespace = str(entry.get("namespace", "")).strip()
        phase_map = entry.get("phase_map")
        if not isinstance(phase_map, Mapping) or not phase_map:
            raise RetrievalDataError(f"Missing phase map for {action}: {path}")
        if not namespace:
            raise RetrievalDataError(f"Missing namespace for {action}: {path}")
        mapped_ids: set[str] = set()
        for phase, record in phase_map.items():
            if not str(phase).strip() or not isinstance(record, Mapping):
                raise RetrievalDataError(
                    f"Invalid canonical phase record for {action}/{phase}: {path}"
                )
            subskill_id = str(record.get("id", "")).strip()
            if not subskill_id.startswith(namespace + "_"):
                raise RetrievalDataError(
                    f"Canonical ID {subskill_id!r} is outside {namespace!r}: {path}"
                )
            if subskill_id in all_ids:
                raise RetrievalDataError(
                    f"Canonical ID is not globally unique: {subskill_id}"
                )
            mapped_ids.add(subskill_id)
            all_ids.add(subskill_id)
        action_ids[action] = mapped_ids

    if action_ids["ThreeStepPoleRun"] & action_ids["FiveStepPoleRun"]:
        raise RetrievalDataError(
            "ThreeStepPoleRun and FiveStepPoleRun canonical IDs must remain disjoint"
        )
    return payload


def canonical_subskill_labels(
    index: Mapping[str, np.ndarray],
    action: str,
    protocol: Mapping[str, Any],
) -> np.ndarray:
    """Map every indexed segment phase to one frozen task-specific subskill."""

    action_entry = protocol["actions"][action]
    phase_map = {
        str(key).strip().lower(): str(value["id"])
        for key, value in action_entry["phase_map"].items()
    }
    phases = np.asarray(index["gait_phase"]).astype(str)
    normalized = [phase.strip().lower() for phase in phases]
    unmapped = sorted({phase for phase in normalized if phase not in phase_map})
    if unmapped:
        raise RetrievalDataError(
            f"Unmapped gait phases for {action}: {unmapped}. "
            "Freeze the protocol before rerunning instead of silently assigning labels."
        )
    return np.asarray([phase_map[phase] for phase in normalized])


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require_file(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rng(base_seed: int, action: str, profile: str, protocol: str) -> np.random.Generator:
    payload = f"{base_seed}:{action}:{profile}:{protocol}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(offset)


def cosine_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query_values = np.asarray(queries, dtype=np.float64)
    gallery_values = np.asarray(gallery, dtype=np.float64)
    if query_values.ndim != 2 or gallery_values.ndim != 2:
        raise RetrievalDataError("Retrieval arrays must be two-dimensional")
    if query_values.shape[1] != gallery_values.shape[1]:
        raise RetrievalDataError(
            f"Query/gallery dimensions differ: {query_values.shape} vs {gallery_values.shape}"
        )
    query_norm = np.maximum(np.linalg.norm(query_values, axis=1, keepdims=True), 1e-8)
    gallery_norm = np.maximum(np.linalg.norm(gallery_values, axis=1, keepdims=True), 1e-8)
    return (query_values / query_norm) @ (gallery_values / gallery_norm).T


def encode_segment_queries(
    index: Mapping[str, np.ndarray],
    checkpoint_path: Path,
    retriever_module: Any,
    torch: Any,
) -> np.ndarray:
    """Encode one deployment-observable task-goal + phase query per segment."""

    torch.serialization.add_safe_globals([PosixPath, WindowsPath])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, schema, _normalizers = retriever_module.build_model(checkpoint)
    cache: dict[tuple[str, str, str], np.ndarray] = {}
    vectors: list[np.ndarray] = []
    for text, phase, task_goal in zip(
        np.asarray(index["segment_query_text"]).astype(str),
        np.asarray(index["gait_phase"]).astype(str),
        np.asarray(index["task_goal"]).astype(str),
    ):
        key = (text, phase, task_goal)
        if key not in cache:
            cache[key] = retriever_module.encode_query(
                model,
                schema,
                text,
                query_mode="segment",
                phase=phase,
                task_goal=task_goal,
            )
        vectors.append(cache[key])
    result = np.stack(vectors).astype(np.float32)
    del model, checkpoint
    gc.collect()
    return result


def audit_existing_train_index(
    profile_dir: Path,
    calculated: Mapping[str, np.ndarray],
    action: str,
    profile: str,
    canonical_schema_version: str,
    canonical_schema_sha256: str,
) -> dict[str, Any]:
    """Prove that the saved deployment gallery is the same training-only index."""

    index_path = require_file(profile_dir / "latent_index_v3.npz")
    sidecar_path = require_file(profile_dir / "latent_index_v3.json")
    existing = load_npz(index_path)
    sidecar = load_json(sidecar_path)
    if sidecar.get("format") != "motion_latent_segment_index_v3":
        raise RetrievalDataError(f"Stale retrieval index format: {sidecar_path}")
    if sidecar.get("profile") != profile:
        raise RetrievalDataError(f"Index profile mismatch: {sidecar_path}")
    if sidecar.get("datasets") != [action] or not sidecar.get("action_isolated", False):
        raise RetrievalDataError(f"Index is not action-isolated for {action}: {sidecar_path}")
    if sidecar.get("segment_query_version") != EXPECTED_SEGMENT_QUERY_VERSION:
        raise RetrievalDataError(f"Index query protocol mismatch: {sidecar_path}")
    if sidecar.get("canonical_subskill_schema_version") != canonical_schema_version:
        raise RetrievalDataError(f"Index canonical schema mismatch: {sidecar_path}")
    if sidecar.get("canonical_subskill_schema_sha256") != canonical_schema_sha256:
        raise RetrievalDataError(f"Index canonical schema hash mismatch: {sidecar_path}")
    if sidecar.get("strict_segment_label_role") != "supplementary_diagnostic_only":
        raise RetrievalDataError(f"Index strict-label role mismatch: {sidecar_path}")

    for key in (
        "trial_id",
        "segment_id",
        "segment_label",
        "strict_segment_label",
        "gait_phase",
    ):
        if key not in existing or key not in calculated:
            raise RetrievalDataError(f"Index audit is missing {key}: {index_path}")
        if not np.array_equal(
            np.asarray(existing[key]).astype(str),
            np.asarray(calculated[key]).astype(str),
        ):
            raise RetrievalDataError(f"Saved/calculated train index differs for {key}: {index_path}")
    for key in ("z_skill", "z_sem_deploy"):
        if not np.allclose(existing[key], calculated[key], rtol=1e-6, atol=1e-6):
            raise RetrievalDataError(f"Saved/calculated train index differs for {key}: {index_path}")
    if int(sidecar.get("segment_count", -1)) != len(existing["trial_id"]):
        raise RetrievalDataError(f"Index metadata count mismatch: {sidecar_path}")
    return {
        "index": index_path.as_posix(),
        "index_sha256": sha256_file(index_path),
        "sidecar": sidecar_path.as_posix(),
        "sidecar_sha256": sha256_file(sidecar_path),
        "matches_rebuilt_train_index": True,
    }


def audit_split_manifest(
    profile_dir: Path,
    train_index: Mapping[str, np.ndarray],
    test_index: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Verify that retrieval uses the recorded unseen-subject split exactly."""

    manifest_path = require_file(profile_dir / "split_manifest.json")
    manifest = load_json(manifest_path)
    trial_level = manifest.get("split_mode") == "file"  # explicit trial-level held-out split (closed-loop protocol)
    if manifest.get("split_mode") not in {"subject", "loso", "file"}:
        raise RetrievalDataError(
            f"Retrieval requires a subject-aware or explicit held-out split: {manifest_path}"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise RetrievalDataError(f"Split manifest has no split records: {manifest_path}")

    def split_values(name: str, key: str) -> set[str]:
        entry = splits.get(name)
        if not isinstance(entry, Mapping):
            raise RetrievalDataError(f"Split manifest has no {name!r} entry: {manifest_path}")
        values = entry.get(key)
        if not isinstance(values, list):
            raise RetrievalDataError(
                f"Split manifest {name!r} has no {key!r} list: {manifest_path}"
            )
        return {str(value) for value in values}

    manifest_train_trials = split_values("train", "trials")
    manifest_test_trials = split_values("test", "trials")
    actual_train_trials = set(np.asarray(train_index["trial_id"]).astype(str).tolist())
    actual_test_trials = set(np.asarray(test_index["trial_id"]).astype(str).tolist())
    if manifest_train_trials != actual_train_trials:
        raise RetrievalDataError(
            f"Training latent membership differs from split manifest: {manifest_path}"
        )
    if manifest_test_trials != actual_test_trials:
        raise RetrievalDataError(
            f"Test latent membership differs from split manifest: {manifest_path}"
        )

    train_subjects = split_values("train", "subjects")
    test_subjects = split_values("test", "subjects")
    if trial_level:  # same subjects allowed, but no test trial may be in the gallery
        if manifest_train_trials & manifest_test_trials:
            raise RetrievalDataError(f"Train-test trial leakage in {manifest_path}")
        return {
            "split_manifest": manifest_path.as_posix(),
            "split_manifest_sha256": sha256_file(manifest_path),
            "split_mode": "file (trial-level held-out)",
            "gallery_subjects": sorted(train_subjects),
            "query_subjects": sorted(test_subjects),
            "subject_disjoint": False,
            "trial_disjoint": True,
        }
    overlap = train_subjects & test_subjects
    if overlap:
        raise RetrievalDataError(
            f"Train-test subject leakage in {manifest_path}: {sorted(overlap)}"
        )
    if not train_subjects or not test_subjects:
        raise RetrievalDataError(f"Empty train/test subject set: {manifest_path}")
    return {
        "split_manifest": manifest_path.as_posix(),
        "split_manifest_sha256": sha256_file(manifest_path),
        "split_mode": str(manifest.get("split_mode")),
        "gallery_subjects": sorted(train_subjects),
        "query_subjects": sorted(test_subjects),
        "subject_disjoint": True,
    }


def first_relevant_ranks(
    scores: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    gallery_trials: np.ndarray,
    gallery_segments: np.ndarray,
) -> tuple[list[int | None], list[int]]:
    """Return deterministic first-relevant ranks and relevant-item counts."""

    query_values = np.asarray(query_labels).astype(str)
    gallery_values = np.asarray(gallery_labels).astype(str)
    trial_keys = np.asarray(gallery_trials).astype(str)
    segment_keys = np.asarray(gallery_segments).astype(str)
    ranks: list[int | None] = []
    counts: list[int] = []
    for row, label in enumerate(query_values):
        relevant = gallery_values == label
        counts.append(int(relevant.sum()))
        # Score is primary; trial and segment IDs provide reproducible tie order.
        order = np.lexsort((segment_keys, trial_keys, -np.asarray(scores[row])))
        hits = np.flatnonzero(relevant[order])
        ranks.append(int(hits[0]) + 1 if len(hits) else None)
    return ranks, counts


def rank_metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {"r_at_1": 0.0, "r_at_5": 0.0, "mrr": 0.0}
    return {
        "r_at_1": float(sum(rank == 1 for rank in ranks) / n),
        "r_at_5": float(sum(rank is not None and rank <= 5 for rank in ranks) / n),
        "mrr": float(sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n),
    }


def expected_random_metrics(
    relevant_counts: Sequence[int], gallery_size: int, top_k: int = 5
) -> dict[str, float]:
    """Expected metrics for a uniformly random ranking of the same gallery.

    The expectation conditions on each query's number of relevant gallery
    items. Queries with no relevant item keep a zero contribution, matching the
    all-query retrieval protocol.
    """

    if gallery_size <= 0:
        raise RetrievalDataError("Random baseline requires a non-empty gallery")
    k = min(max(int(top_k), 1), gallery_size)
    denominator_at_k = math.comb(gallery_size, k)
    values = {"r_at_1": [], "r_at_5": [], "mrr": []}
    for raw_count in relevant_counts:
        count = int(raw_count)
        if count < 0 or count > gallery_size:
            raise RetrievalDataError(
                f"Invalid relevant-item count {count} for gallery size {gallery_size}"
            )
        if count == 0:
            for name in values:
                values[name].append(0.0)
            continue
        values["r_at_1"].append(count / gallery_size)
        miss_count = gallery_size - count
        miss_at_k = (
            math.comb(miss_count, k) / denominator_at_k
            if miss_count >= k
            else 0.0
        )
        values["r_at_5"].append(1.0 - miss_at_k)

        rank_denominator = math.comb(gallery_size, count)
        expected_rr = sum(
            (math.comb(gallery_size - rank, count - 1) / rank_denominator) / rank
            for rank in range(1, gallery_size - count + 2)
        )
        values["mrr"].append(expected_rr)
    return {
        name: float(np.mean(metric_values)) if metric_values else 0.0
        for name, metric_values in values.items()
    }


def trial_bootstrap_intervals(
    ranks: Sequence[int | None],
    trial_ids: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float | None, float | None]]:
    names = ("r_at_1", "r_at_5", "mrr")
    empty = {name: (None, None) for name in names}
    if samples <= 0:
        return empty
    trials = np.asarray(trial_ids).astype(str)
    unique_trials = np.unique(trials)
    if len(unique_trials) < 2:
        return empty
    rank_array = np.asarray(ranks, dtype=object)
    indices = {trial: np.flatnonzero(trials == trial) for trial in unique_trials}
    values = {name: [] for name in names}
    for _ in range(samples):
        sampled_trials = rng.choice(unique_trials, size=len(unique_trials), replace=True)
        sampled_indices = np.concatenate([indices[trial] for trial in sampled_trials])
        metrics = rank_metrics(rank_array[sampled_indices].tolist())
        for name in names:
            values[name].append(metrics[name])
    return {
        name: tuple(float(value) for value in np.quantile(values[name], [0.025, 0.975]))
        for name in names
    }


def evaluate_protocol(
    *,
    action: str,
    profile: str,
    protocol: str,
    query_vectors: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    train_index: Mapping[str, np.ndarray],
    test_index: Mapping[str, np.ndarray],
    split_audit: Mapping[str, Any],
    label_protocol_version: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = cosine_matrix(query_vectors, train_index["z_skill"])
    query_labels = np.asarray(query_labels).astype(str)
    gallery_labels = np.asarray(gallery_labels).astype(str)
    ranks, relevant_counts = first_relevant_ranks(
        scores,
        query_labels,
        gallery_labels,
        train_index["trial_id"],
        train_index["segment_id"],
    )
    metrics = rank_metrics(ranks)
    random_metrics = expected_random_metrics(relevant_counts, len(gallery_labels))
    covered_ranks = [rank for rank in ranks if rank is not None]
    covered_metrics = rank_metrics(covered_ranks)
    intervals = trial_bootstrap_intervals(
        ranks,
        test_index["trial_id"],
        bootstrap_samples,
        stable_rng(seed, action, profile, protocol),
    )
    covered = len(covered_ranks)
    unseen_labels = sorted(set(query_labels.tolist()) - set(gallery_labels.tolist()))
    row: dict[str, Any] = {
        "action": action,
        "supervision_level": f"L{profile[1:]}",
        "source_profile": profile,
        "protocol": protocol,
        "gallery_split": "train",
        "query_split": "test",
        "gallery_trials": len(set(np.asarray(train_index["trial_id"]).astype(str).tolist())),
        "query_trials": len(set(np.asarray(test_index["trial_id"]).astype(str).tolist())),
        "gallery_subjects": split_audit["gallery_subjects"],
        "query_subjects": split_audit["query_subjects"],
        "subject_disjoint": split_audit["subject_disjoint"],
        "label_protocol_version": label_protocol_version,
        "relevance_definition": RELEVANCE_DEFINITION,
        "gallery_segments": len(gallery_labels),
        "query_segments": len(query_labels),
        "gallery_unique_labels": len(set(gallery_labels.tolist())),
        "query_unique_labels": len(set(query_labels.tolist())),
        "query_labels_unseen_in_gallery": len(unseen_labels),
        "unseen_query_label_names": unseen_labels,
        "queries_without_relevant_gallery_item": len(ranks) - covered,
        "gallery_label_coverage": covered / max(len(ranks), 1),
        "mean_relevant_gallery_items": float(np.mean(relevant_counts)),
        "random_r_at_1": random_metrics["r_at_1"],
        "random_r_at_5": random_metrics["r_at_5"],
        "random_mrr": random_metrics["mrr"],
        "r_at_1": metrics["r_at_1"],
        "r_at_5": metrics["r_at_5"],
        "mrr": metrics["mrr"],
        "covered_query_r_at_1": covered_metrics["r_at_1"],
        "covered_query_r_at_5": covered_metrics["r_at_5"],
        "covered_query_mrr": covered_metrics["mrr"],
        "bootstrap_unit": "test_trial",
        "bootstrap_replicates": bootstrap_samples,
    }
    for name in ("r_at_1", "r_at_5", "mrr"):
        row[f"{name}_ci95_low"], row[f"{name}_ci95_high"] = intervals[name]

    query_rows = []
    for index, (rank, relevant_count) in enumerate(zip(ranks, relevant_counts)):
        query_rows.append(
            {
                "action": action,
                "supervision_level": f"L{profile[1:]}",
                "source_profile": profile,
                "protocol": protocol,
                "trial_id": str(test_index["trial_id"][index]),
                "segment_id": str(test_index["segment_id"][index]),
                "gait_phase": str(test_index["gait_phase"][index]),
                "canonical_subskill_id": str(query_labels[index]),
                "strict_phase_event_label": str(test_index["strict_segment_label"][index]),
                "relevant_in_gallery": relevant_count > 0,
                "relevant_gallery_items": relevant_count,
                "first_relevant_rank": rank,
                "hit_at_1": rank == 1,
                "hit_at_5": rank is not None and rank <= 5,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
            }
        )
    return row, query_rows


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def build_macro_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for profile in SOURCE_PROFILES:
        for protocol in PROTOCOLS:
            selected = [
                row
                for row in rows
                if row["source_profile"] == profile and row["protocol"] == protocol
            ]
            if len(selected) != len(ACTIONS):
                raise RetrievalDataError(f"Incomplete task macro for {profile}/{protocol}")
            result.append(
                {
                    "supervision_level": f"L{profile[1:]}",
                    "source_profile": profile,
                    "protocol": protocol,
                    "actions": len(selected),
                    "task_macro_gallery_label_coverage": float(
                        np.mean([row["gallery_label_coverage"] for row in selected])
                    ),
                    "task_macro_r_at_1": float(np.mean([row["r_at_1"] for row in selected])),
                    "task_macro_r_at_5": float(np.mean([row["r_at_5"] for row in selected])),
                    "task_macro_mrr": float(np.mean([row["mrr"] for row in selected])),
                    "task_macro_covered_query_r_at_1": float(
                        np.mean([row["covered_query_r_at_1"] for row in selected])
                    ),
                    "task_macro_covered_query_r_at_5": float(
                        np.mean([row["covered_query_r_at_5"] for row in selected])
                    ),
                    "task_macro_covered_query_mrr": float(
                        np.mean([row["covered_query_mrr"] for row in selected])
                    ),
                }
            )
    return result


def percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.{digits}f}%"


def metric_with_ci(row: Mapping[str, Any], name: str, *, percent_value: bool) -> str:
    value = row.get(name)
    low = row.get(f"{name}_ci95_low")
    high = row.get(f"{name}_ci95_high")
    formatter = percent if percent_value else lambda item: f"{float(item):.3f}"
    if value is None:
        return "N/A"
    if low is None or high is None:
        return formatter(value)
    return f"{formatter(value)} [{formatter(low)}, {formatter(high)}]"


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    source_latent_root: str,
) -> str:
    source_cli = ".\\" + source_latent_root.replace("/", "\\")
    l4_skill = [
        row
        for row in rows
        if row["source_profile"] == "Z4" and row["protocol"] == "skill_nn"
    ]
    l4_query = [
        row
        for row in rows
        if row["source_profile"] == "Z4"
        and row["protocol"] == "deployment_query_to_skill"
    ]
    lines = [
        "# Data-HNU offline retrieval metrics",
        "",
        f"Source snapshot: `{source_latent_root}` (3 actions x 5 profiles).",
        "",
        "The training-only gallery contains one mean latent per trial and segment. "
        "The held-out test set contributes one query per segment. Relevance uses the "
        "frozen evaluation-only canonical subskill taxonomy in Data/canonical_subskills_v1.json. "
        "Labels are scoped by action, so analogous ThreeStepPoleRun and FiveStepPoleRun phases "
        "are never cross-action matches. Queries whose label is absent from the gallery "
        "remain in the denominator as retrieval failures.",
        "",
        "## L4 cross-subject skill representation retrieval",
        "",
        "| Action | Test / train segments | Gallery label coverage | R@1 (95% CI) | R@5 (95% CI) | MRR (95% CI) | Covered-query R@1 / R@5 / MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in l4_skill:
        lines.append(
            "| {action} | {test} / {train} | {coverage} | {r1} | {r5} | {mrr} | {cr1} / {cr5} / {cmrr} |".format(
                action=row["action"],
                test=row["query_segments"],
                train=row["gallery_segments"],
                coverage=percent(row["gallery_label_coverage"]),
                r1=metric_with_ci(row, "r_at_1", percent_value=True),
                r5=metric_with_ci(row, "r_at_5", percent_value=True),
                mrr=metric_with_ci(row, "mrr", percent_value=False),
                cr1=percent(row["covered_query_r_at_1"]),
                cr5=percent(row["covered_query_r_at_5"]),
                cmrr=f"{row['covered_query_mrr']:.3f}",
            )
        )

    lines.extend(
        [
            "",
            "## L4 random-ranking reference",
            "",
            "The reference is the exact expectation of a uniformly random ordering of "
            "the same gallery, conditioned on each query's number of relevant items.",
            "",
            "| Action | Observed / random R@1 | Observed / random R@5 | Observed / random MRR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in l4_skill:
        lines.append(
            f"| {row['action']} | {percent(row['r_at_1'])} / "
            f"{percent(row['random_r_at_1'])} | {percent(row['r_at_5'])} / "
            f"{percent(row['random_r_at_5'])} | {row['mrr']:.3f} / "
            f"{row['random_mrr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "`skill_nn` uses held-out z_skill as the query and training z_skill as the "
            "gallery. It is the clean representation diagnostic recommended for Section 8.1. "
            "Its confidence interval resamples test trials and does not include retraining-seed variation "
            "or variation across unseen subjects; the current split contains one test subject per action.",
            "",
            "## Task-macro L0-L4 skill representation retrieval",
            "",
            "| Level | Mean gallery coverage | Macro R@1 | Macro R@5 | Macro MRR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in macro_rows:
        if row["protocol"] != "skill_nn":
            continue
        lines.append(
            f"| {row['supervision_level']} | "
            f"{percent(row['task_macro_gallery_label_coverage'])} | "
            f"{percent(row['task_macro_r_at_1'])} | "
            f"{percent(row['task_macro_r_at_5'])} | "
            f"{row['task_macro_mrr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## L4 deployment-query-to-skill diagnostic",
            "",
            "| Action | Unique deployment queries | Gallery label coverage | R@1 | R@5 | MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in l4_query:
        query_records = [
            item
            for item in _SUMMARY_QUERY_CACHE
            if item["action"] == row["action"]
            and item["source_profile"] == "Z4"
            and item["protocol"] == "deployment_query_to_skill"
        ]
        unique_queries = len(
            {
                (item["gait_phase"], item["trial_id"].split("_", 1)[0])
                for item in query_records
            }
        )
        lines.append(
            f"| {row['action']} | {unique_queries} | {percent(row['gallery_label_coverage'])} | "
            f"{percent(row['r_at_1'])} | {percent(row['r_at_5'])} | {row['mrr']:.3f} |"
        )

    skill_macro = [row for row in macro_rows if row["protocol"] == "skill_nn"]
    best_r1 = max(skill_macro, key=lambda row: row["task_macro_r_at_1"])
    best_r5 = max(skill_macro, key=lambda row: row["task_macro_r_at_5"])
    best_mrr = max(skill_macro, key=lambda row: row["task_macro_mrr"])
    best_levels = {
        best_r1["supervision_level"],
        best_r5["supervision_level"],
        best_mrr["supervision_level"],
    }
    best_conclusion = (
        f"{next(iter(best_levels))} has the highest point estimate for all three aggregate retrieval metrics."
        if len(best_levels) == 1
        else "No single level has the highest point estimate for all three metrics."
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"- The best task-macro skill-NN R@1 is {best_r1['supervision_level']}="
            f"{percent(best_r1['task_macro_r_at_1'])}; the best R@5 is "
            f"{best_r5['supervision_level']}={percent(best_r5['task_macro_r_at_5'])}; "
            f"the best MRR is {best_mrr['supervision_level']}={best_mrr['task_macro_mrr']:.3f}. "
            f"{best_conclusion}",
            "- Canonical subskills describe task-scoped motion function only. Quality and "
            "safety outcomes are deliberately excluded from the relevance label.",
            "- The deployment query contains task goal and gait phase. Because the current "
            "canonical taxonomy is phase-based, query-to-skill results diagnose interface "
            "alignment and are not a substitute for a held-out natural-language benchmark.",
            "- No feasibility filter or quality bonus is applied. These numbers test retrieval "
            "rather than offline safety selection, Q_exec, or closed-loop success.",
            "- Compare every metric with its matched random-ranking expectation because "
            "the number of relevant gallery segments differs substantially by action.",
            "- Current checkpoints represent one training seed. A final paper needs 3-5 "
            "independent seeds in addition to the test-trial bootstrap intervals.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            f".\\.venv\\Scripts\\python.exe .\\Results\\8.1_Offline_Representation\\scripts\\compute_offline_retrieval_metrics.py --latent-root '{source_cli}'",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


# Populated by main before build_summary; kept module-private so the Markdown
# builder can count distinct observable query forms without bloating metric rows.
_SUMMARY_QUERY_CACHE: list[dict[str, Any]] = []


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    project_root = find_project_root(Path(__file__).resolve())
    latent_root = args.latent_root.resolve()
    annotation_root = args.annotation_root.resolve()
    framework_root = args.framework_root.resolve()
    canonical_path = args.canonical_subskills.resolve()
    output_dir = args.output_dir.resolve()
    for directory in (latent_root, annotation_root, framework_root):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_protocol = load_canonical_subskills(canonical_path)
    label_protocol_version = str(canonical_protocol["schema_version"])
    canonical_sha256 = sha256_file(canonical_path)

    build_index_module, retriever_module, torch = load_runtime(
        project_root, framework_root
    )
    runtime_schema_version = str(
        getattr(build_index_module, "CANONICAL_SUBSKILL_SCHEMA_VERSION", "")
    )
    runtime_schema_sha256 = str(
        getattr(build_index_module, "CANONICAL_SUBSKILL_SCHEMA_SHA256", "")
    )
    if runtime_schema_version != label_protocol_version:
        raise RetrievalDataError(
            "Build-index runtime and evaluator use different canonical schema versions"
        )
    if runtime_schema_sha256 != canonical_sha256:
        raise RetrievalDataError(
            "Build-index runtime and evaluator use different canonical schema files"
        )
    rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    membership_fingerprints: dict[str, dict[str, str]] = {}

    for action in ACTIONS:
        provenance[action] = {}
        membership_fingerprints[action] = {}
        for profile in SOURCE_PROFILES:
            profile_dir = latent_root / action / profile
            if not profile_dir.is_dir():
                raise NotADirectoryError(profile_dir)
            train_path = require_file(profile_dir / "train_latents.npz")
            test_path = require_file(profile_dir / "test_latents.npz")
            checkpoint_path = require_file(profile_dir / "best_motion_latent.pt")
            train_index, train_metadata = build_index_module.build_index(
                [train_path], annotation_root, profile=profile
            )
            test_index, test_metadata = build_index_module.build_index(
                [test_path], annotation_root, profile=profile
            )
            split_audit = audit_split_manifest(profile_dir, train_index, test_index)
            audit = audit_existing_train_index(
                profile_dir,
                train_index,
                action,
                profile,
                label_protocol_version,
                canonical_sha256,
            )
            train_trials = set(np.asarray(train_index["trial_id"]).astype(str).tolist())
            test_trials = set(np.asarray(test_index["trial_id"]).astype(str).tolist())
            overlap = train_trials & test_trials
            if overlap:
                raise RetrievalDataError(
                    f"{action}/{profile} has train-test trial leakage: {sorted(overlap)}"
                )
            if train_metadata.get("datasets") != [action] or test_metadata.get("datasets") != [action]:
                raise RetrievalDataError(f"Action identity mismatch under {profile_dir}")

            train_labels = canonical_subskill_labels(
                train_index, action, canonical_protocol
            )
            test_labels = canonical_subskill_labels(
                test_index, action, canonical_protocol
            )
            for split_name, index_data, mapped_labels in (
                ("train", train_index, train_labels),
                ("test", test_index, test_labels),
            ):
                indexed_labels = np.asarray(index_data["segment_label"]).astype(str)
                if not np.array_equal(indexed_labels, mapped_labels.astype(str)):
                    raise RetrievalDataError(
                        f"{action}/{profile} {split_name} canonical labels differ "
                        "between the frozen ontology and built index"
                    )

            fingerprint_rows = sorted(
                f"{trial}\x1f{segment}\x1f{label}\x1f{strict}\x1f{query}"
                for trial, segment, label, strict, query in zip(
                    np.asarray(test_index["trial_id"]).astype(str),
                    np.asarray(test_index["segment_id"]).astype(str),
                    test_labels.astype(str),
                    np.asarray(test_index["strict_segment_label"]).astype(str),
                    np.asarray(test_index["segment_query_text"]).astype(str),
                )
            )
            membership_fingerprints[action][profile] = hashlib.sha256(
                "\n".join(fingerprint_rows).encode("utf-8")
            ).hexdigest()

            protocol_vectors = {
                "skill_nn": np.asarray(test_index["z_skill"], dtype=np.float32),
                "deployment_query_to_skill": encode_segment_queries(
                    test_index, checkpoint_path, retriever_module, torch
                ),
            }
            for protocol, vectors in protocol_vectors.items():
                row, details = evaluate_protocol(
                    action=action,
                    profile=profile,
                    protocol=protocol,
                    query_vectors=vectors,
                    query_labels=test_labels,
                    gallery_labels=train_labels,
                    train_index=train_index,
                    test_index=test_index,
                    split_audit=split_audit,
                    label_protocol_version=label_protocol_version,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                )
                rows.append(row)
                query_rows.extend(details)
                print(
                    f"{action}/{profile}/{protocol}: "
                    f"R@1={row['r_at_1']:.4f}, R@5={row['r_at_5']:.4f}, "
                    f"MRR={row['mrr']:.4f}, coverage={row['gallery_label_coverage']:.4f}"
                )
            provenance[action][profile] = {
                **audit,
                **split_audit,
                "checkpoint": checkpoint_path.relative_to(project_root).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "train_latents": train_path.relative_to(project_root).as_posix(),
                "train_latents_sha256": sha256_file(train_path),
                "test_latents": test_path.relative_to(project_root).as_posix(),
                "test_latents_sha256": sha256_file(test_path),
            }

    for action in ACTIONS:
        baseline = membership_fingerprints[action]["Z0"]
        for profile in SOURCE_PROFILES[1:]:
            if membership_fingerprints[action][profile] != baseline:
                raise RetrievalDataError(
                    f"{action}: retrieval query membership/labels differ between Z0 and {profile}"
                )

    macro_rows = build_macro_rows(rows)
    primary_rows = [row for row in rows if row["source_profile"] == "Z4"]
    global _SUMMARY_QUERY_CACHE
    _SUMMARY_QUERY_CACHE = query_rows

    write_csv(output_dir / "offline_retrieval_metrics.csv", rows, METRIC_FIELDS)
    write_csv(output_dir / "offline_retrieval_primary_L4.csv", primary_rows, METRIC_FIELDS)
    write_csv(output_dir / "offline_retrieval_macro_by_level.csv", macro_rows, MACRO_FIELDS)
    write_csv(output_dir / "offline_retrieval_per_query.csv", query_rows, QUERY_FIELDS)

    payload = {
        "schema_version": "data_hnu_offline_retrieval_metrics_v2",
        "script_version": SCRIPT_VERSION,
        "scope": "offline_segment_retrieval_without_RL",
        "source_latent_root": latent_root.relative_to(project_root).as_posix(),
        "source_annotation_root": annotation_root.relative_to(project_root).as_posix(),
        "framework_root": framework_root.relative_to(project_root).as_posix(),
        "canonical_subskills": {
            "path": canonical_path.relative_to(project_root).as_posix(),
            "sha256": canonical_sha256,
            "schema_version": label_protocol_version,
            "evaluation_only": True,
            "label_scope": "within_action",
            "pole_run_namespaces_disjoint": True,
        },
        "actions": list(ACTIONS),
        "source_profiles": list(SOURCE_PROFILES),
        "protocols": {
            "skill_nn": {
                "query": "held-out test segment mean z_skill",
                "gallery": "training segment mean z_skill",
                "score": "cosine_similarity",
                "paper_role": "Section 8.1 representation diagnostic",
            },
            "deployment_query_to_skill": {
                "query": "task_goal + gait_phase encoded by the checkpoint query branch",
                "gallery": "training segment mean z_skill",
                "score": "cosine_similarity",
                "paper_role": "query-interface diagnostic; not yet full language OOD evaluation",
            },
        },
        "relevance": {
            "definition": RELEVANCE_DEFINITION,
            "strict_phase_event_label_retained_for_audit_only": True,
            "uncovered_query_handling": "counted_as_zero_hit_and_zero_reciprocal_rank",
            "conditional_metrics_also_reported": True,
        },
        "bootstrap": {
            "unit": "test_trial",
            "replicates": args.bootstrap_samples,
            "seed": args.seed,
            "interval": "percentile_95",
            "does_not_include_training_seed_variation": True,
        },
        "integrity_checks": {
            "processed_all_3_actions_x_5_levels_x_2_protocols": len(rows) == 30,
            "gallery_is_train_only": True,
            "query_is_test_only": True,
            "train_test_trials_disjoint": True,
            "train_test_subjects_disjoint": True,
            "latent_membership_matches_split_manifests": True,
            "one_query_per_trial_segment": True,
            "overlapping_windows_averaged_before_retrieval": True,
            "saved_train_indexes_match_rebuilt_indexes": True,
            "sample_membership_labels_and_queries_identical_across_levels": True,
            "all_exported_phases_mapped_to_frozen_subskills": True,
            "ThreeStepPoleRun_and_FiveStepPoleRun_labels_disjoint": True,
            "quality_filter_or_bonus_used": False,
            "test_selected_hyperparameters_used": False,
        },
        "provenance": provenance,
        "macro_rows": macro_rows,
        "rows": rows,
    }
    with (output_dir / "offline_retrieval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    (output_dir / "offline_retrieval_summary.md").write_text(
        build_summary(
            rows,
            macro_rows,
            latent_root.relative_to(project_root).as_posix(),
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {len(rows)} retrieval rows to {output_dir}")


if __name__ == "__main__":
    main()
