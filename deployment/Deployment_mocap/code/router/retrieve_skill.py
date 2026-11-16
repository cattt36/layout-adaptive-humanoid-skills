#!/usr/bin/env python3
r"""Retrieve a motion skill latent from a segment-level latent index.

Newly retrained Z0-Z4 checkpoints share the same task_goal-only query branch
and deployment-observable segment-query serializer.
Text from ASR or a direct command is encoded by the checkpoint's shared
semantic encoder.  The base retrieval score is a motion-anchor (z_skill)
cosine similarity with a smaller task-only z_sem_deploy auxiliary term.
Exported feasibility and risk metadata remain hard safety filters.
When the checkpoint contains a quality head, z_quality is decoded into
predicted quality metrics and used as a small soft ranking bonus; it never
bypasses the explicit safety filters.

Example:
    python retrieve_skill.py `
      --index C:\Users\Administrator\Desktop\Data-HNU\Latent\segment_query\Z4\latent_index_v2.npz `
      --checkpoint C:\Users\Administrator\Desktop\Data-HNU\Latent\segment_query\Z4\best_motion_latent.pt `
      --query "Safely pass under the bridge while maintaining a low posture." `
      --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from motion_latent_z import (
    CHECKPOINT_FORMAT_VERSION,
    QUALITY_CONDITIONING_VERSION,
    QUALITY_INPUT_VERSION,
    SEMANTIC_FUSION_VERSION,
    TRAINING_PROTOCOL_VERSION,
    MotionLatentModel,
    hashed_text_vector,
    one_hot,
)
from segment_protocol import SEGMENT_QUERY_VERSION


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("This script requires PyTorch to encode the query.")


TASK_QUERY_VERSION = "task_goal_only_v1"


def cosine_rows(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a row matrix."""
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != query.shape[0]:
        raise ValueError(
            f"cosine_rows shape mismatch: query={query.shape}, rows={rows.shape}"
        )
    q_norm = max(float(np.linalg.norm(query)), 1e-8)
    row_norm = np.maximum(np.linalg.norm(rows, axis=1), 1e-8)
    return (rows @ query) / (row_norm * q_norm)


def finite_float(value: Any) -> float | None:
    """Serialize NaN/inf metadata as JSON null."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def checkpoint_value(
    checkpoint: dict[str, Any],
    schema: dict[str, Any],
    key: str,
) -> Any:
    """Read a protocol field, preferring the checkpoint root over schema."""
    value = checkpoint.get(key)
    return schema.get(key) if value is None else value


def state_shape(state: dict[str, Any], key: str) -> tuple[int, ...]:
    if key not in state:
        raise KeyError(f"Checkpoint is missing model parameter: {key}")
    shape = tuple(getattr(state[key], "shape", ()))
    if not shape:
        raise ValueError(f"Checkpoint parameter has no usable shape: {key}")
    return shape


def require_checkpoint_protocol(
    checkpoint: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Reject checkpoints that are not for the current three-branch model."""
    checkpoint_version = checkpoint_value(
        checkpoint, schema, "checkpoint_format_version"
    )
    if checkpoint_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_format_version {checkpoint_version!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION!r}."
        )

    fusion_version = checkpoint_value(
        checkpoint, schema, "semantic_fusion_version"
    )
    if fusion_version != SEMANTIC_FUSION_VERSION:
        raise ValueError(
            f"Unsupported semantic_fusion_version {fusion_version!r}; "
            f"expected {SEMANTIC_FUSION_VERSION!r} for the current "
            "task/robot/human gated model."
        )

    if schema.get("task_query_version") != TASK_QUERY_VERSION:
        raise ValueError(
            f"Checkpoint task_query_version {schema.get('task_query_version')!r} "
            f"does not match required {TASK_QUERY_VERSION!r}. Retrain before "
            "using ASR/text retrieval."
        )
    if not bool(schema.get("task_context", False)):
        raise ValueError(
            "Checkpoint was not trained with the shared task text interface. "
            "Retrain its Z0-Z4 profile with the current motion_latent script."
        )
    for key, expected in (
        ("training_protocol_version", TRAINING_PROTOCOL_VERSION),
        ("quality_input_version", QUALITY_INPUT_VERSION),
        ("quality_conditioning_version", QUALITY_CONDITIONING_VERSION),
    ):
        actual = checkpoint_value(checkpoint, schema, key)
        if actual != expected:
            raise ValueError(
                f"Unsupported {key} {actual!r}; expected {expected!r}. "
                "Retrain and rebuild the index with the current protocol."
            )


def build_model(
    checkpoint: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Reconstruct the current model and reject stale checkpoints."""
    require_torch()
    schema = checkpoint.get("schema", {})
    args = checkpoint.get("args", {})
    state = checkpoint.get("model_state")
    if not isinstance(schema, dict) or not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a usable schema/model_state")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"This retriever requires {CHECKPOINT_FORMAT_VERSION}; use a newly "
            "trained action-specific checkpoint, not the old merged model."
        )
    if checkpoint.get("semantic_fusion_version") != SEMANTIC_FUSION_VERSION:
        raise ValueError(
            "Checkpoint is not compatible with the three-branch gated semantic model."
        )
    if schema.get("semantic_fusion_version") != SEMANTIC_FUSION_VERSION:
        raise ValueError(
            f"Checkpoint schema has no {SEMANTIC_FUSION_VERSION} layout"
        )
    if not bool(schema.get("task_context", False)):
        raise ValueError("Checkpoint was not trained with the task text interface")
    if schema.get("task_query_version") != "task_goal_only_v1":
        raise ValueError("Checkpoint predates task_goal-only query training")

    def shape(key: str) -> tuple[int, ...]:
        return state_shape(state, key)

    skill_shape = shape("skill_encoder.network.0.conv1.weight")
    quality_shape = shape("quality_encoder.network.0.conv1.weight")
    task_in = shape("task_semantic_encoder.network.0.weight")
    robot_in = shape("robot_semantic_encoder.network.0.weight")
    human_in = shape("human_semantic_encoder.network.0.weight")
    quality_conditioner_in = shape("quality_conditioner.network.0.weight")
    task_out = shape("task_semantic_encoder.network.6.weight")
    fusion_in = shape("semantic_fusion.0.weight")
    quality_out = shape("quality_encoder.output.0.weight")
    task_output = shape("task_decoder.network.6.weight")
    semantic_decoder = shape("semantic_task_decoder.network.0.weight")
    local_shape = shape("local_projection.0.weight")

    latent_dim = task_out[0]
    quality_latent_dim = quality_out[0]
    channels = task_in[0]
    if robot_in[0] != channels or human_in[0] != channels:
        raise ValueError("Semantic branch hidden dimensions disagree")
    if fusion_in[1] != 3 * latent_dim + 3:
        raise ValueError("Unexpected semantic_fusion input layout")
    if quality_conditioner_in[1] != quality_latent_dim + latent_dim:
        raise ValueError("Unexpected skill-conditioned quality input layout")
    phase_dim = semantic_decoder[1] - latent_dim
    if phase_dim <= 0:
        raise ValueError("Could not infer phase dimension from checkpoint")

    model = MotionLatentModel(
        skill_input_dim=skill_shape[1],
        quality_input_dim=quality_shape[1],
        phase_dim=phase_dim,
        task_dim=task_output[0],
        task_semantic_input_dim=task_in[1],
        robot_semantic_input_dim=robot_in[1],
        human_semantic_input_dim=human_in[1],
        semantic_metric_dim=len(schema.get("sem_keys", [])),
        quality_target_dim=len(schema.get("quality_keys", [])),
        atom_dim=len(schema.get("atom_vocab", [])),
        quality_tag_dim=len(schema.get("quality_tag_vocab", [])),
        latent_dim=latent_dim,
        quality_latent_dim=quality_latent_dim,
        channels=channels,
        dropout=float(args.get("dropout", 0.0)),
        local_feature_dim=local_shape[0],
        dual_scale_skill=int(args.get("skill_local_pool_frames", 0)) > 0,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # v4 added human_atom_head; checkpoints trained before it simply do not have it
    # v5 added training-only prediction heads on z_skill (future_head / human_ctrl_head): irrelevant for encoding
    opt = ("human_atom_head.", "future_head.", "human_ctrl_head.")
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith(opt)]
    if bad:
        raise RuntimeError(f"checkpoint/model mismatch: {bad[:8]}")
    model.eval()
    normalizers = checkpoint.get("normalizers", {})
    if not isinstance(normalizers, dict):
        normalizers = {}
    return model, schema, normalizers

def encode_query(
    model: Any,
    schema: dict[str, Any],
    query: str,
    *,
    query_mode: str = "task",
    phase: str = "",
    task_goal: str = "",
) -> np.ndarray:
    """Encode a task command using the same branch serializer as training."""
    query_mode = str(query_mode).lower().strip()
    if query_mode not in {"task", "segment"}:
        raise ValueError("query_mode must be 'task' or 'segment'")
    text_dim = int(schema.get("text_dim", 128))
    sem_dim = len(schema.get("sem_keys", []))
    gait_vocab = list(schema.get("gait_vocab", []))
    gait = one_hot(phase.strip(), gait_vocab) if query_mode == "segment" else np.zeros(len(gait_vocab), dtype=np.float32)
    if query_mode == "task":
        marked_text = f"task_goal: {query}".strip()
    else:
        if schema.get("segment_query_version") != SEGMENT_QUERY_VERSION:
            raise ValueError(
                f"Checkpoint has no {SEGMENT_QUERY_VERSION} schema"
            )
        marked_text = query.strip()
        if not marked_text:
            parts = []
            if task_goal.strip():
                parts.append(f"task_goal: {task_goal.strip()}")
            if phase.strip():
                parts.append(f"segment_phase: {phase.strip()}")
            marked_text = " . ".join(parts)
    text = hashed_text_vector(marked_text, text_dim)
    task_branch = np.concatenate((text, np.asarray([float(bool(marked_text))], dtype=np.float32)))
    metric_dim = sem_dim
    robot_branch = np.zeros(2 * metric_dim + len(gait_vocab) + 1, dtype=np.float32)
    human_branch = np.zeros(2 * metric_dim + text_dim + 1, dtype=np.float32)
    if query_mode == "segment":
        robot_branch[-len(gait_vocab)-1:-1] = gait
        robot_branch[-1] = float(bool(np.any(gait)))
    with torch.no_grad():
        device = next(model.parameters()).device
        tb = torch.from_numpy(task_branch[None, :]).to(device)
        rb = torch.from_numpy(robot_branch[None, :]).to(device)
        hb = torch.from_numpy(human_branch[None, :]).to(device)
        z = model.encode_semantic(tb, rb, hb, force_task_only=(query_mode == "task"))
    return z.cpu().numpy()[0].astype(np.float32)

def inverse_quality_standardizer(
    normalized: np.ndarray,
    normalizer: dict[str, Any],
) -> np.ndarray:
    """Convert quality-head outputs back to the raw training-label scale."""
    required = ("mean", "std", "lower", "upper")
    if any(key not in normalizer for key in required):
        raise KeyError(
            "Checkpoint normalizers.quality_target is missing one of: "
            + ", ".join(required)
        )
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    lower = np.asarray(normalizer["lower"], dtype=np.float32)
    upper = np.asarray(normalizer["upper"], dtype=np.float32)
    if normalized.shape[-1] != len(mean):
        raise ValueError(
            "Quality-head output dimension does not match "
            "normalizers.quality_target."
        )
    raw = normalized * std[None, :] + mean[None, :]
    return np.clip(raw, lower[None, :], upper[None, :])


def predict_quality_scores(
    model: Any,
    schema: dict[str, Any],
    normalizers: dict[str, Any],
    z_quality: np.ndarray,
) -> dict[str, Any]:
    """Decode z_quality into feasibility/risk scores for soft ranking.

    The quality head predicts the same normalized quality targets used during
    training.  Only the derived feasibility and risk dimensions are reduced
    to scalar scores here; all decoded quality metrics are retained for the
    output audit.
    """
    result: dict[str, Any] = {
        "available": False,
        "quality_values": None,
        "feasibility": np.full(len(z_quality), np.nan, dtype=np.float32),
        "risk": np.full(len(z_quality), np.nan, dtype=np.float32),
        "score": np.full(len(z_quality), np.nan, dtype=np.float32),
        "source": "unavailable",
    }
    quality_head = getattr(model, "quality_head", None)
    quality_normalizer = normalizers.get("quality_target")
    quality_keys = list(schema.get("quality_keys", []))
    if quality_head is None or not quality_keys or not isinstance(
        quality_normalizer, dict
    ):
        return result

    expected_dim = int(quality_head.in_features)
    if z_quality.ndim != 2 or z_quality.shape[1] != expected_dim:
        raise ValueError(
            f"Index z_quality has shape {z_quality.shape}, but the checkpoint "
            f"expects ({expected_dim},)."
        )
    with torch.no_grad():
        normalized = quality_head(
            torch.from_numpy(z_quality.astype(np.float32, copy=False))
        ).cpu().numpy()
    quality_values = inverse_quality_standardizer(normalized, quality_normalizer)
    result["quality_values"] = quality_values.astype(np.float32)

    def column(name: str) -> np.ndarray | None:
        try:
            return quality_values[:, quality_keys.index(name)]
        except ValueError:
            return None

    predicted_feasibility = column("robot_quality.derived.feasibility_score")
    predicted_risk = column("robot_quality.derived.risk_score")
    if predicted_feasibility is not None:
        predicted_feasibility = np.clip(
            np.nan_to_num(predicted_feasibility, nan=0.0), 0.0, 1.0
        ).astype(np.float32)
        result["feasibility"] = predicted_feasibility
    if predicted_risk is not None:
        predicted_risk = np.maximum(
            np.nan_to_num(predicted_risk, nan=np.inf), 0.0
        ).astype(np.float32)
        result["risk"] = predicted_risk

    # Feasibility is the preferred scalar.  If a checkpoint lacks that
    # derived target, risk still gives a monotonic quality score.
    if np.isfinite(result["feasibility"]).any():
        result["score"] = result["feasibility"].copy()
        result["source"] = "quality_head.feasibility_score"
    elif np.isfinite(result["risk"]).any():
        result["score"] = np.exp(
            -np.minimum(np.nan_to_num(result["risk"], nan=np.inf), 50.0)
        ).astype(np.float32)
        result["source"] = "quality_head.risk_score"
    else:
        return result
    result["available"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--query-mode", choices=("task", "segment"), default="task",
        help="task for ASR/task retrieval; segment for offline segment-aware diagnostics.",
    )
    parser.add_argument(
        "--phase", default="",
        help="Gait phase for --query-mode segment (e.g. right_single_support).",
    )
    parser.add_argument(
        "--task-goal", default="",
        help="Optional task-level goal included in a segment query.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Require the index to contain exactly this action/dataset. "
            "Use this with action-specific indexes to prevent accidental cross-action retrieval."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-feasibility", type=float, default=None)
    parser.add_argument("--max-risk", type=float, default=None)
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.0,
        help=(
            "Optional score bonus multiplied by the exported "
            "feasibility_score after hard filtering."
        ),
    )
    parser.add_argument(
        "--predicted-quality-weight",
        type=float,
        default=0.25,
        help=(
            "Soft score bonus multiplied by feasibility predicted from "
            "z_quality. Set to 0 to disable. It never replaces hard filters."
        ),
    )
    parser.add_argument(
        "--allow-unknown-quality",
        action="store_true",
        help=(
            "Allow entries with missing quality metadata to pass the affected "
            "filter. This is unsafe for deployment and must be explicit."
        ),
    )
    parser.add_argument(
        "--disable-hard-filter",
        action="store_true",
        help=(
            "Explicitly disable feasibility/risk hard filtering for offline "
            "semantic retrieval. Do not use this for robot deployment."
        ),
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON output path; a matching NPZ is also written.")
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.index.is_file():
        raise FileNotFoundError(args.index)
    quality_requested = (
        args.min_feasibility is not None or args.max_risk is not None
    )
    if args.disable_hard_filter and quality_requested:
        raise ValueError(
            "--disable-hard-filter cannot be combined with "
            "--min-feasibility or --max-risk."
        )
    if not args.disable_hard_filter and not quality_requested:
        raise ValueError(
            "No hard quality threshold was provided. For safe retrieval, "
            "pass --min-feasibility and/or --max-risk. For offline semantic "
            "retrieval only, explicitly pass --disable-hard-filter."
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, schema, normalizers = build_model(checkpoint)
    profile = str(schema.get("profile", "")).upper()
    # Task retrieval does not require an index sidecar, but the shared index
    # loading path below performs optional metadata checks. Initialize this
    # mapping so task mode cannot reference an undefined value.
    metadata: dict[str, Any] = {}
    if args.dataset is not None and (
        not str(args.dataset).strip() or Path(str(args.dataset)).name != str(args.dataset)
    ):
        raise ValueError("--dataset must be a simple action directory name")
    if args.query_mode == "segment" and schema.get("segment_query_version") != SEGMENT_QUERY_VERSION:
        raise ValueError(
            f"Checkpoint segment protocol {schema.get('segment_query_version')!r} "
            f"does not match required {SEGMENT_QUERY_VERSION!r}. Retrain first."
        )
    sidecar = args.index.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"Index metadata sidecar is missing: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format") != "motion_latent_segment_index_v3":
        raise ValueError("Stale index format; rebuild it with build_latent_index.py")
    if metadata.get("segment_query_version") != SEGMENT_QUERY_VERSION:
        raise ValueError(
            f"Index segment protocol {metadata.get('segment_query_version')!r} "
            f"does not match {SEGMENT_QUERY_VERSION!r}. Rebuild the index."
        )
    indexed_datasets = [str(item) for item in (metadata.get("datasets") or [])]
    if indexed_datasets != [str(args.dataset)]:
        raise ValueError(
            f"Index datasets {indexed_datasets!r} do not match --dataset "
            f"{args.dataset!r}. Use an action-specific index."
        )
    if not bool(metadata.get("action_isolated", False)):
        raise ValueError("Deployment retrieval requires an action-isolated index")
    schema_files = metadata.get("schema_files") or []
    schema_hashes = metadata.get("schema_sha256") or {}
    if schema_files and schema_hashes:
        import hashlib
        for schema_file in schema_files:
            schema_path = Path(schema_file)
            if not schema_path.is_file():
                raise ValueError(
                    f"Index provenance schema is missing: {schema_path}. Rebuild index."
                )
            expected_hash = schema_hashes.get(str(schema_path))
            actual_hash = hashlib.sha256(schema_path.read_bytes()).hexdigest()
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(
                    f"Index schema hash mismatch for {schema_path}. Rebuild index."
                )
    if args.query_mode == "segment":
        if metadata.get("segment_query_version") != SEGMENT_QUERY_VERSION:
            raise ValueError(
                f"Index segment protocol {metadata.get('segment_query_version')!r} "
                f"does not match required {SEGMENT_QUERY_VERSION!r}. Rebuild index."
            )
        if metadata.get("segment_relevance") != "canonical_segment_label":
            raise ValueError(
                "Segment index does not use canonical_segment_label relevance. "
                "Rebuild it with the current build_latent_index.py."
            )
        index_profile = str(metadata.get("profile", "")).upper()
        if index_profile and index_profile != profile:
            raise ValueError(
                f"Index profile {index_profile!r} does not match checkpoint profile {profile!r}."
            )
    if profile == "Z0" and not args.disable_hard_filter:
        raise ValueError(
            "Z0 has no robot numeric quality supervision, so its exported "
            "feasibility/risk values cannot be used as a deployment safety "
            "filter. Use --disable-hard-filter only for offline retrieval, "
            "or use an external safety checker."
        )
    query_latent = encode_query(
        model, schema, args.query, query_mode=args.query_mode,
        phase=args.phase, task_goal=args.task_goal,
    )
    with np.load(args.index, allow_pickle=False) as data:
        required = ("z_skill", "z_quality", "z_sem", "trial_id", "segment_id")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError("Index is missing: " + ", ".join(missing))
        # Only the explicit task-only latent is legal for deployment.  Falling
        # back to the teacher-side z_sem silently leaks privileged annotations.
        z_skill = np.asarray(data["z_skill"], dtype=np.float32)
        if "z_sem_deploy" not in data:
            raise ValueError(
                "Index has no z_sem_deploy. Retrain/export with the current "
                "protocol and rebuild the index."
            )
        z_sem = np.asarray(data["z_sem_deploy"], dtype=np.float32)
        if z_skill.ndim != 2 or z_skill.shape[1] != query_latent.shape[0]:
            raise ValueError(
                f"Index z_skill shape {z_skill.shape} is incompatible with "
                f"checkpoint query latent shape {query_latent.shape}. Rebuild index."
            )
        if z_sem.ndim != 2 or z_sem.shape[1] != query_latent.shape[0]:
            raise ValueError(
                f"Index z_sem shape {z_sem.shape} is incompatible with "
                f"checkpoint query latent shape {query_latent.shape}. Rebuild index."
            )
        if metadata.get("segment_count") is not None and int(metadata["segment_count"]) != len(z_sem):
            raise ValueError(
                "Segment index metadata count does not match the NPZ contents. "
                "Rebuild index."
            )
        skill_similarity = cosine_rows(query_latent, z_skill)
        semantic_similarity = cosine_rows(query_latent, z_sem)
        # query_z_sem is trained against the motion anchor z_skill; use the
        # explicit task-only semantic export as a smaller auxiliary term.
        similarity = 0.70 * skill_similarity + 0.30 * semantic_similarity
        count = len(similarity)
        feasibility = (
            np.asarray(data["feasibility_score"], dtype=np.float32)
            if "feasibility_score" in data
            else np.full(count, np.nan, dtype=np.float32)
        )
        risk = (
            np.asarray(data["risk_score"], dtype=np.float32)
            if "risk_score" in data
            else np.full(count, np.nan, dtype=np.float32)
        )
        predicted_quality = predict_quality_scores(
            model,
            schema,
            normalizers,
            np.asarray(data["z_quality"], dtype=np.float32),
        )
        predicted_feasibility = predicted_quality["feasibility"]
        predicted_risk = predicted_quality["risk"]
        predicted_quality_score = predicted_quality["score"]
        if args.predicted_quality_weight and not predicted_quality["available"]:
            print(
                "WARNING: checkpoint has no usable quality_head outputs; "
                "predicted-quality ranking is disabled."
            )

        eligible = np.ones(count, dtype=bool)
        quality_metadata_known = np.isfinite(feasibility) | np.isfinite(risk)
        if args.min_feasibility is not None:
            known = np.isfinite(feasibility)
            if args.allow_unknown_quality:
                feasibility_pass = (~known) | (
                    known & (feasibility >= args.min_feasibility)
                )
            else:
                feasibility_pass = known & (feasibility >= args.min_feasibility)
            eligible &= feasibility_pass
        if args.max_risk is not None:
            known = np.isfinite(risk)
            if args.allow_unknown_quality:
                risk_pass = (~known) | (known & (risk <= args.max_risk))
            else:
                risk_pass = known & (risk <= args.max_risk)
            eligible &= risk_pass
        if quality_requested and not quality_metadata_known.any():
            if not args.allow_unknown_quality:
                raise RuntimeError(
                    "The index contains no finite feasibility/risk metadata. "
                    "Cannot apply a hard quality filter. Add valid metadata "
                    "or explicitly pass --allow-unknown-quality."
                )
            print(
                "WARNING: index contains no finite feasibility/risk metadata; "
                "all entries are treated as unknown because "
                "--allow-unknown-quality was explicitly set."
            )
        if not eligible.any():
            raise RuntimeError("No skill remains after feasibility filtering")

        score = similarity.copy()
        if args.quality_weight:
            quality_bonus = np.nan_to_num(feasibility, nan=0.0)
            score += float(args.quality_weight) * quality_bonus
        if args.predicted_quality_weight and predicted_quality["available"]:
            quality_bonus = np.nan_to_num(predicted_quality_score, nan=0.0)
            score += float(args.predicted_quality_weight) * quality_bonus
        candidate_indices = np.flatnonzero(eligible)
        candidate_indices = candidate_indices[
            np.argsort(score[candidate_indices])[::-1][:args.top_k]
        ]

        results: list[dict[str, Any]] = []
        for index in candidate_indices:
            item: dict[str, Any] = {
                "rank": len(results) + 1,
                "trial_id": str(data["trial_id"][index]),
                "segment_id": str(data["segment_id"][index]),
                "similarity": float(similarity[index]),
                "skill_similarity": float(skill_similarity[index]),
                "semantic_similarity": float(semantic_similarity[index]),
                "score": float(score[index]),
                "feasibility_score": finite_float(feasibility[index]),
                "risk_score": finite_float(risk[index]),
                "predicted_feasibility_score": finite_float(
                    predicted_feasibility[index]
                ),
                "predicted_risk_score": finite_float(predicted_risk[index]),
                "predicted_quality_score": finite_float(
                    predicted_quality_score[index]
                ),
            }
            for key in (
                "task_name",
                "task_goal",
                "task_family",
                "gait_phase",
                "feasibility_tags",
                "window_count",
                "start_index",
                "center_index",
            ):
                if key in data:
                    value = data[key][index]
                    item[key] = int(value) if np.issubdtype(
                        np.asarray(value).dtype, np.integer
                    ) else str(value)
            results.append(item)

        payload = {
            "format": "motion_skill_retrieval_v3",
            "query": args.query,
            "query_mode": args.query_mode,
            "phase": args.phase,
            "task_goal": args.task_goal,
            "dataset": args.dataset,
            "checkpoint": str(args.checkpoint),
            "index": str(args.index),
            "quality_filter": {
                "min_feasibility": args.min_feasibility,
                "max_risk": args.max_risk,
                "quality_weight": args.quality_weight,
                "predicted_quality_weight": args.predicted_quality_weight,
                "predicted_quality_source": predicted_quality["source"],
                "hard_filter_enabled": not args.disable_hard_filter,
                "allow_unknown_quality": args.allow_unknown_quality,
                "scope": "offline_reference_segment_only",
            },
            "results": results,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            selected = np.asarray(candidate_indices, dtype=np.int64)
            np.savez_compressed(
                args.output.with_suffix(".npz"),
                query_z_sem=query_latent.astype(np.float32),
                z_skill=np.asarray(data["z_skill"])[selected],
                z_quality=np.asarray(data["z_quality"])[selected],
                z_sem_deploy=np.asarray(data["z_sem_deploy"])[selected],
                trial_id=np.asarray(data["trial_id"])[selected],
                segment_id=np.asarray(data["segment_id"])[selected],
                score=score[selected].astype(np.float32),
                predicted_feasibility_score=predicted_feasibility[
                    selected
                ].astype(np.float32),
                predicted_risk_score=predicted_risk[selected].astype(np.float32),
                predicted_quality_score=predicted_quality_score[
                    selected
                ].astype(np.float32),
            )
            print(f"Saved retrieval JSON: {args.output}")
            print(f"Saved retrieval latents: {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()

