#!/usr/bin/env python3
r"""Build a segment-level index from exported motion latents.

The training exporter writes one row per motion window.  Retrieval should not
let many overlapping windows from one segment dominate the result, so this
script averages windows by ``trial_id + segment_id`` and keeps the associated
task and feasibility metadata.

Example:
    python build_latent_index.py `
      --latent-dir C:\Users\Administrator\Desktop\Data-HNU\Latent\Z4 `
      --split train `
      --output C:\Users\Administrator\Desktop\Data-HNU\Latent\Z4\latent_index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Import the shared, profile-independent segment protocol.
LATENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = LATENT_ROOT.parent / "Data"
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))
if str(LATENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_ROOT))

from segment_protocol import (
    CANONICAL_SUBSKILL_SCHEMA_SHA256,
    CANONICAL_SUBSKILL_SCHEMA_VERSION,
    SEGMENT_QUERY_VERSION,
    canonical_segment_label,
    strict_segment_evidence_label,
)
from motion_latent_z import (
    CHECKPOINT_FORMAT_VERSION,
    QUALITY_CONDITIONING_VERSION,
    QUALITY_INPUT_VERSION,
    SEMANTIC_FUSION_VERSION,
    TRAINING_PROTOCOL_VERSION,
    _evidence_grounded_human_text,
    collect_segment_query_text,
)


def scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if value is None:
        return ""
    return str(value)


def optional_array(data: np.lib.npyio.NpzFile, key: str, length: int) -> np.ndarray:
    if key not in data:
        return np.full(length, np.nan, dtype=np.float32)
    return np.asarray(data[key])


def find_annotation_document(
    json_root: Path | None,
    trial_id: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Load one canonical annotation document for a trial."""
    if not json_root:
        return {}
    if trial_id in cache:
        return cache[trial_id]
    dataset, _, suffix = trial_id.partition("_")
    candidates = (
        json_root / dataset / f"{trial_id}.json",
        json_root / dataset / f"{suffix}.json",
    )
    candidate = next((path for path in candidates if path.is_file()), None)
    if candidate is None:
        cache[trial_id] = {}
        return {}
    try:
        document = json.loads(candidate.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        document = {}
    if not isinstance(document, dict):
        document = {}
    cache[trial_id] = document
    return document


def find_task_document(
    json_root: Path | None,
    trial_id: str,
    cache: dict[str, dict[str, str]],
    annotation_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    if not json_root:
        return {}
    document = find_annotation_document(
        json_root, trial_id, annotation_cache if annotation_cache is not None else {}
    )
    task = document.get("task", {})
    if not isinstance(task, dict):
        cache[trial_id] = {}
        return {}
    family = task.get("task_family", [])
    if isinstance(family, str):
        family = [family]
    result = {
        "task_name": scalar_text(task.get("task_name_en")),
        "task_goal": scalar_text(task.get("task_goal")),
        "task_family": "|".join(scalar_text(item) for item in family),
    }
    cache[trial_id] = result
    return result


def segment_from_document(
    document: Mapping[str, Any], segment_id: str
) -> Mapping[str, Any] | None:
    segments = document.get("segments", [])
    if not isinstance(segments, list):
        return None
    for segment in segments:
        if isinstance(segment, Mapping) and str(segment.get("segment_id", "")) == segment_id:
            return segment
    return None


def semantic_evidence_available(segment: Mapping[str, Any] | None) -> bool:
    """Return the same human-semantic coverage flag as the evaluator.

    Coverage must mean that the evidence-aware serializer actually produced
    usable human semantic content.  Checking only ``evidence`` would mark
    unsupported/empty annotations as covered and make the index provenance
    disagree with ``calculate_latent_metrics.py``.
    """
    return bool(_evidence_grounded_human_text(segment or {}))

def resolve_latent_files(args: argparse.Namespace) -> list[Path]:
    if args.latent_files:
        paths = [Path(item) for item in args.latent_files]
    else:
        paths = [
            args.latent_dir / f"{split}_latents.npz"
            for split in args.split
        ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing latent file(s): " + ", ".join(missing))
    return paths


def validate_latent_schema(latent_file: Path, profile: str | None) -> dict[str, Any]:
    """Reject latent exports that were not produced by the active v2 protocol."""
    schema_path = latent_file.parent / "schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Cannot build a {SEGMENT_QUERY_VERSION} index without {schema_path}. "
            "Retrain/export the profile first."
        )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid latent schema: {schema_path}") from exc
    version = schema.get("segment_query_version")
    if version != SEGMENT_QUERY_VERSION:
        raise ValueError(
            f"{latent_file} was exported with segment protocol {version!r}; "
            f"expected {SEGMENT_QUERY_VERSION!r}. Retrain before building an index."
        )
    for key, expected in (
        ("checkpoint_format_version", CHECKPOINT_FORMAT_VERSION),
        ("training_protocol_version", TRAINING_PROTOCOL_VERSION),
        ("semantic_fusion_version", SEMANTIC_FUSION_VERSION),
        ("quality_input_version", QUALITY_INPUT_VERSION),
        ("quality_conditioning_version", QUALITY_CONDITIONING_VERSION),
    ):
        if schema.get(key) != expected:
            raise ValueError(
                f"{schema_path} has {key}={schema.get(key)!r}; expected "
                f"{expected!r}. Retrain before building an index."
            )
    schema_profile = str(schema.get("profile", "")).upper()
    if profile and schema_profile and schema_profile != profile.upper():
        raise ValueError(
            f"Schema profile {schema_profile!r} disagrees with requested "
            f"profile {profile!r}: {schema_path}"
        )
    return schema


def build_index(
    latent_files: list[Path],
    json_root: Path | None,
    profile: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    json_cache: dict[str, dict[str, str]] = {}
    annotation_cache: dict[str, dict[str, Any]] = {}
    source_files: list[str] = []
    schema_files: list[str] = []
    schema_hashes: dict[str, str] = {}

    for latent_file in latent_files:
        schema = validate_latent_schema(latent_file, profile)
        schema_path = latent_file.parent / "schema.json"
        schema_files.append(str(schema_path))
        schema_hashes[str(schema_path)] = hashlib.sha256(
            schema_path.read_bytes()
        ).hexdigest()
        source_files.append(str(latent_file))
        with np.load(latent_file, allow_pickle=False) as data:
            required = (
                "z_skill", "z_quality", "z_sem", "z_sem_deploy",
                "trial_id", "segment_id",
            )
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"{latent_file} is missing: {', '.join(missing)}")
            count = len(data["trial_id"])
            z_skill = np.asarray(data["z_skill"], dtype=np.float32)
            z_quality = np.asarray(data["z_quality"], dtype=np.float32)
            z_sem = np.asarray(data["z_sem"], dtype=np.float32)
            z_sem_deploy = np.asarray(data["z_sem_deploy"], dtype=np.float32)
            if not (len(z_skill) == len(z_quality) == len(z_sem) == len(z_sem_deploy) == count):
                raise ValueError(f"Latent row counts disagree in {latent_file}")

            optional = {
                key: optional_array(data, key, count)
                for key in (
                    "risk_score",
                    "feasibility_score",
                    "semantic_purity",
                    "semantic_window_weight",
                )
            }
            for row in range(count):
                trial_id = scalar_text(data["trial_id"][row])
                segment_id = scalar_text(data["segment_id"][row])
                key = (trial_id, segment_id)
                if key not in groups:
                    task = {}
                    for field in ("task_name", "task_goal", "task_family"):
                        if field in data:
                            task[field] = scalar_text(data[field][row])
                    document = find_annotation_document(json_root, trial_id, annotation_cache)
                    segment = segment_from_document(document, segment_id)
                    task = {
                        **find_task_document(json_root, trial_id, json_cache, annotation_cache),
                        **task,
                    }
                    groups[key] = {
                        "skill": [],
                        "quality": [],
                        "sem": [],
                        "sem_deploy": [],
                        "risk": [],
                        "feasibility": [],
                        "purity": [],
                        "window_weight": [],
                        "task_name": task.get("task_name", ""),
                        "task_goal": task.get("task_goal", ""),
                        "task_family": task.get("task_family", ""),
                        "gait_phase": (
                            scalar_text(data["gait_phase"][row])
                            if "gait_phase" in data
                            else ""
                        ),
                        "start_index": [],
                        "center_index": [],
                        "feasibility_tags": set(),
                        "segment_label": (
                            canonical_segment_label(segment, action=trial_id)
                            if segment is not None
                            else ""
                        ),
                        "strict_segment_label": (
                            strict_segment_evidence_label(segment)
                            if segment is not None
                            else ""
                        ),
                        "segment_query_text": (
                            collect_segment_query_text(document, segment, profile=(profile or "Z4"))
                            if document and segment is not None
                            else ""
                        ),
                        "semantic_evidence": semantic_evidence_available(segment),
                    }
                item = groups[key]
                item["skill"].append(z_skill[row])
                item["quality"].append(z_quality[row])
                item["sem"].append(z_sem[row])
                item["sem_deploy"].append(z_sem_deploy[row])
                for source, target in (
                    ("risk_score", "risk"),
                    ("feasibility_score", "feasibility"),
                    ("semantic_purity", "purity"),
                    ("semantic_window_weight", "window_weight"),
                ):
                    value = float(optional[source][row])
                    if math.isfinite(value):
                        item[target].append(value)
                if "start_index" in data:
                    item["start_index"].append(int(data["start_index"][row]))
                if "center_index" in data:
                    item["center_index"].append(int(data["center_index"][row]))
                if "feasibility_tags" in data:
                    tags = scalar_text(data["feasibility_tags"][row])
                    item["feasibility_tags"].update(
                        tag for tag in tags.split("|") if tag
                    )

    if not groups:
        raise ValueError("No latent rows were found")

    ordered = sorted(groups)
    arrays: dict[str, list[Any]] = defaultdict(list)
    quality_available = 0
    for key in ordered:
        trial_id, segment_id = key
        item = groups[key]
        risk = float(np.mean(item["risk"])) if item["risk"] else float("nan")
        feasibility = (
            float(np.mean(item["feasibility"]))
            if item["feasibility"]
            else (math.exp(-risk) if math.isfinite(risk) else float("nan"))
        )
        quality_available += int(math.isfinite(feasibility))
        arrays["z_skill"].append(np.mean(item["skill"], axis=0))
        arrays["z_quality"].append(np.mean(item["quality"], axis=0))
        arrays["z_sem"].append(np.mean(item["sem"], axis=0))
        arrays["z_sem_deploy"].append(np.mean(item["sem_deploy"], axis=0))
        arrays["trial_id"].append(trial_id)
        arrays["segment_id"].append(segment_id)
        arrays["task_name"].append(item["task_name"])
        arrays["task_goal"].append(item["task_goal"])
        arrays["task_family"].append(item["task_family"])
        arrays["gait_phase"].append(item["gait_phase"])
        arrays["risk_score"].append(risk)
        arrays["feasibility_score"].append(feasibility)
        arrays["semantic_purity"].append(
            float(np.mean(item["purity"])) if item["purity"] else float("nan")
        )
        arrays["semantic_window_weight"].append(
            float(np.mean(item["window_weight"]))
            if item["window_weight"]
            else float("nan")
        )
        arrays["window_count"].append(len(item["skill"]))
        arrays["start_index"].append(min(item["start_index"], default=-1))
        arrays["center_index"].append(
            int(round(float(np.mean(item["center_index"]))))
            if item["center_index"]
            else -1
        )
        arrays["feasibility_tags"].append("|".join(sorted(item["feasibility_tags"])))
        arrays["segment_label"].append(item["segment_label"])
        arrays["strict_segment_label"].append(item["strict_segment_label"])
        arrays["segment_query_text"].append(item["segment_query_text"])
        arrays["semantic_evidence_flag"].append(int(item["semantic_evidence"]))

    result = {
        "z_skill": np.asarray(arrays["z_skill"], dtype=np.float32),
        "z_quality": np.asarray(arrays["z_quality"], dtype=np.float32),
        "z_sem": np.asarray(arrays["z_sem"], dtype=np.float32),
        "z_sem_deploy": np.asarray(arrays["z_sem_deploy"], dtype=np.float32),
        "trial_id": np.asarray(arrays["trial_id"]),
        "segment_id": np.asarray(arrays["segment_id"]),
        "task_name": np.asarray(arrays["task_name"]),
        "task_goal": np.asarray(arrays["task_goal"]),
        "task_family": np.asarray(arrays["task_family"]),
        "gait_phase": np.asarray(arrays["gait_phase"]),
        "risk_score": np.asarray(arrays["risk_score"], dtype=np.float32),
        "feasibility_score": np.asarray(
            arrays["feasibility_score"], dtype=np.float32
        ),
        "semantic_purity": np.asarray(
            arrays["semantic_purity"], dtype=np.float32
        ),
        "semantic_window_weight": np.asarray(
            arrays["semantic_window_weight"], dtype=np.float32
        ),
        "window_count": np.asarray(arrays["window_count"], dtype=np.int32),
        "start_index": np.asarray(arrays["start_index"], dtype=np.int32),
        "center_index": np.asarray(arrays["center_index"], dtype=np.int32),
        "feasibility_tags": np.asarray(arrays["feasibility_tags"]),
        "segment_label": np.asarray(arrays["segment_label"]),
        "strict_segment_label": np.asarray(arrays["strict_segment_label"]),
        "segment_query_text": np.asarray(arrays["segment_query_text"]),
        "semantic_evidence_flag": np.asarray(
            arrays["semantic_evidence_flag"], dtype=np.int8
        ),
    }
    datasets = sorted({
        trial_id.split("_", 1)[0]
        for trial_id, _segment_id in ordered
        if "_" in trial_id
    })
    metadata = {
        "format": "motion_latent_segment_index_v3",
        "segment_query_version": SEGMENT_QUERY_VERSION,
        "datasets": datasets,
        "action_isolated": len(datasets) <= 1,
        "segment_relevance": "canonical_segment_label",
        "canonical_subskill_schema_version": CANONICAL_SUBSKILL_SCHEMA_VERSION,
        "canonical_subskill_schema_sha256": CANONICAL_SUBSKILL_SCHEMA_SHA256,
        "strict_segment_label_role": "supplementary_diagnostic_only",
        "source_files": source_files,
        "schema_files": schema_files,
        "schema_sha256": schema_hashes,
        "segment_count": len(ordered),
        "quality_rows_with_feasibility": quality_available,
        "quality_available": quality_available > 0,
        "aggregation": "mean over exported windows grouped by trial_id + segment_id",
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dir", type=Path, default=Path(
        r"C:\Users\Administrator\Desktop\Data-HNU\Latent\Z4"
    ))
    parser.add_argument("--split", nargs="+", default=["train"],
                        choices=("train", "val", "test"))
    parser.add_argument("--latent-files", nargs="+", default=None)
    parser.add_argument("--profile", choices=("Z0", "Z1", "Z2", "Z3", "Z4"), default=None,
                        help="Profile provenance; validated against the latent directory name when provided.")
    parser.add_argument("--json-root", type=Path, default=Path(
        r"C:\Users\Administrator\Desktop\Data-HNU\LLM_annotation"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        r"C:\Users\Administrator\Desktop\Data-HNU\Latent\Z4\latent_index"
    ), help="Output prefix; writes <prefix>.npz and <prefix>.json.")
    args = parser.parse_args()

    files = resolve_latent_files(args)
    json_root = args.json_root if args.json_root.is_dir() else None
    inferred_profile = args.latent_dir.name.upper()
    valid_profiles = {"Z0", "Z1", "Z2", "Z3", "Z4"}
    profile = args.profile or (inferred_profile if inferred_profile in valid_profiles else None)
    if args.profile and inferred_profile in valid_profiles and args.profile != inferred_profile:
        raise ValueError(f"--profile {args.profile} disagrees with latent directory {inferred_profile}")
    index, metadata = build_index(files, json_root, profile=profile)
    if profile:
        metadata["profile"] = profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **index)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Built {metadata['segment_count']} segment entries from "
        f"{len(files)} latent file(s)."
    )
    if not metadata["quality_available"]:
        print("WARNING: no feasibility metadata found; retrieval cannot filter quality.")
    print(f"Saved: {args.output.with_suffix('.npz')}")
    print(f"Saved: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()


