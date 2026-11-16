#!/usr/bin/env python3
"""
Motion-latent training framework for multiple HNU motion datasets.

Design goal
===========
Learn three deliberately separated representations:

1. z_skill   : motion/task content that should be shared by human semantics and
               robot motion.
2. z_sem     : the semantic encoder's estimate of z_skill.
3. z_quality : skill-conditioned feasibility / implementation quality from
               deployment-observable proprioception/contact. Simulator-only
               torque/limit/collision summaries are labels, never inputs.

Both z_skill and z_sem pass through the same task decoder.  The semantic path
uses a weaker, task-oriented reconstruction loss because segment semantics do
not uniquely determine every robot joint angle.

The script is intentionally a single file so it can be read and modified
before it is split into a production package.  It supports:

* ``--mode audit`` with only NumPy installed;
* ``--mode train`` with PyTorch installed;
* JSON files stored in one or more semantic annotation directories, optionally
  enriched by matching retargeted dynamics/feasibility JSON files;
* any number of matching ``*_Dynamics.npz`` files in one or more directories;
* subject-aware train/validation/test splitting from fixed dataset trial ranges;
* action-isolated training by default (one action per run);
* continuous skill and quality latents;
* a shared task_goal-only text/ASR query branch for Z0-Z4;
* weak supervision from numeric metrics, canonical skill atoms, captions,
  control hints, robot objectives and feasibility tags;
* checkpoint and per-window latent export.

Minimal dependencies
--------------------
    pip install numpy torch

Audit the supplied example files
--------------------------------
    python motion_latent_training_framework.py \
      --mode audit \
      --datasets CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun

Train
-----
    python motion_latent_training_framework.py \
      --mode train \
      --datasets CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun \
      --output-dir runs/multi_motion_latent \
      --epochs 200 --warmup-epochs 20

Important limitations
---------------------
The NPZ files do not contain link poses, so the default reconstruction target
uses root state, contact count and joint kinematics as a bootstrap proxy.  The
feature builder can later be extended with additional geometry features if
such data becomes available.
"""

from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Allow direct execution from the Latent directory while resolving the shared
# annotation schema stored under the sibling Data directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _PROJECT_ROOT / "Data"
if str(_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_ROOT))

import numpy as np
from human_semantic_schema import normalize_existing_human_semantics
from robot_semantic_schema import CANONICAL_ATOMS
from robot_semantic_schema import canonical_objective as shared_canonical_objective
from robot_semantic_schema import canonical_atom as shared_canonical_atom
from robot_semantic_schema import fixed_objective as shared_fixed_objective
from robot_semantic_schema import phase_known_absent_atoms
from segment_protocol import SEGMENT_QUERY_VERSION


CHECKPOINT_FORMAT_VERSION = "motion_latent_checkpoint_v4"
SEMANTIC_FUSION_VERSION = "three_branch_gated_v2"
TRAINING_PROTOCOL_VERSION = "privileged_to_deployable_v1"
QUALITY_INPUT_VERSION = "deployable_proprioception_v1"
QUALITY_CONDITIONING_VERSION = "skill_conditioned_quality_v1"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:  # Audit mode remains usable without PyTorch.
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object


# ---------------------------------------------------------------------------
# Canonical semantic ontology
# ---------------------------------------------------------------------------

# Keep this mapping small and interpretable.  It removes wording differences
# without pretending that an LLM-generated tag is ground truth.
DEFAULT_ATOM_MAP = {
    "overhead_obstacle_clearance": "overhead_clearance",
    "overhead_clearance": "overhead_clearance",
    "torso_clearance_management": "torso_clearance",
    "torso_clearance": "torso_clearance",
    "collision_avoidance": "collision_avoidance",
    "weight_transfer_control": "weight_transfer",
    "weight_transfer": "weight_transfer",
    "double_support_stabilization": "double_support_stabilization",
    "single_support_balance": "single_support_balance",
    "support_leg_loading": "support_loading",
    "support_loading": "support_loading",
    "load_acceptance": "load_acceptance",
    "swing_leg_flexion": "swing_leg_flexion",
    "deep_support_knee_flexion": "support_knee_flexion",
    "support_knee_flexion": "support_knee_flexion",
    "swing_leg_advancement": "swing_leg_advancement",
    "crouched_posture": "crouched_posture",
    "low_pelvis_posture": "crouched_posture",
    "constrained_forward_progression": "forward_progression",
    "forward_progression": "forward_progression",
    "push_off_preparation": "push_off_preparation",
    "foot_clearance_management": "foot_clearance",
    "foot_clearance": "foot_clearance",
    "forward_torso_lean": "forward_torso_lean",
}

ACTIVE_ATOM_SET = frozenset(CANONICAL_ATOMS)

# Robot semantic atoms are generated weak labels.  Keep deterministic
# phase/contact contradictions stronger than LLM-inferred absent labels.
DEFAULT_ROBOT_ATOM_POSITIVE_WEIGHT = 0.85
DEFAULT_HUMAN_ATOM_POSITIVE_WEIGHT = 0.60
DEFAULT_LLM_ABSENT_ATOM_WEIGHT = 0.25
DEFAULT_PHASE_ABSENT_ATOM_WEIGHT = 1.00
DEFAULT_HUMAN_SEMANTIC_WEIGHT = 1.00
DEFAULT_HUMAN_SEMANTIC_INPUT_SCALE = 1.00
DEFAULT_SEMANTIC_TEXT_SCALE = 1.00

# These quantities are useful teacher-side context, but directly decoding them
# from z_skill encourages the motion code to retain subject speed/scale rather
# than a subject-invariant skill identity.  They remain available to z_sem and
# to the exported metadata; only the direct z_skill regression mask excludes
# them.
SKILL_NUISANCE_SEMANTIC_KEYS = frozenset(
    {
        "segment.duration_s",
        "trial.body_height_m",
        "trial.body_weight_kg",
    }
)

DEFAULT_GAIT_PHASES = (
    "right_single_support",
    "double_support_right_to_left",
    "left_single_support",
    "double_support_left_to_right",
)

# Subject membership is defined by the dataset-specific trial ranges supplied
# for this project.  The dataset prefix is kept in the resulting group key so
# each action can be audited independently.
DEFAULT_SUBJECT_TRIAL_GROUPS = {
    "CrouchedTraverse": {
        "subject_1": frozenset({1, *range(14, 24)}),   # 001 + 014-023
        "subject_2": frozenset({2, *range(24, 34)}),   # 002 + 024-033
        "subject_3": frozenset(range(3, 14)),           # 003-013
    },
    "ThreeStepPoleRun": {
        "subject_1": frozenset(range(1, 11)),          # 001-010
        "subject_2": frozenset(range(11, 21)),         # 011-020
        "subject_4": frozenset(range(21, 33)),         # 021-032
    },
    "FiveStepPoleRun": {
        "subject_1": frozenset(range(1, 11)),          # 001-010
        "subject_2": frozenset(range(11, 22)),         # 011-021
        "subject_4": frozenset(range(22, 32)),         # 022-031
    },
}

# Default subject-aware policy.  The labels are dataset-local: subject_1 in
# one action is not assumed to be the same person as subject_1 in another
# action.  subject_1 and subject_2 form the train/validation pool; subject_3
# and subject_4 are held out as completely unseen test subjects.
DEFAULT_SUBJECT_SPLIT = {
    "subject_1": "train_val_pool",
    "subject_2": "train_val_pool",
    "subject_3": "test",
    "subject_4": "test",
}

DEFAULT_DATASETS = (
    "CrouchedTraverse",
    "ThreeStepPoleRun",
    "FiveStepPoleRun",
)

# New action directories can be loaded through ``--datasets`` without
# editing this list.  Subject-aware splitting is intentionally restricted to
# datasets whose trial-to-subject mapping is explicitly documented above.
# This prevents an unseen action from being silently assigned to artificial
# groups based only on its numeric trial suffix.
SUBJECT_CONFIGURED_DATASETS = frozenset(DEFAULT_SUBJECT_TRIAL_GROUPS)


# Ablation profiles.  All profiles use the same model, split policy,
# NPZ-derived motion stream and task-command interface.  The profile only
# controls which robot/human annotation namespaces provide extra supervision.
ABLATION_PROFILES = {
    "Z0": {
        # Z0 still pairs against JSON so every ablation uses the same trials,
        # aligned segments and random-trial split.  Robot/human annotation
        # values are masked; task_goal is retained as the shared command input.
        "requires_json": True,
        "robot_numeric": False,
        "robot_semantics": False,
        "human_numeric": False,
        "human_semantics": False,
        "task_context": True,
    },
    "Z1": {
        "requires_json": True,
        "robot_numeric": True,
        "robot_semantics": False,
        "human_numeric": False,
        "human_semantics": False,
        "task_context": True,
    },
    "Z2": {
        "requires_json": True,
        "robot_numeric": True,
        "robot_semantics": True,
        "human_numeric": False,
        "human_semantics": False,
        "task_context": True,
    },
    "Z3": {
        "requires_json": True,
        "robot_numeric": True,
        "robot_semantics": True,
        "human_numeric": True,
        "human_semantics": False,
        "task_context": True,
    },
    "Z4": {
        "requires_json": True,
        "robot_numeric": True,
        "robot_semantics": True,
        "human_numeric": True,
        "human_semantics": True,
        # Task text is common to every profile.  Z4 remains the full model
        # because it additionally receives evidence-backed human semantics.
        "task_context": True,
    },
}


def ablation_config(profile: str) -> Mapping[str, bool]:
    try:
        return ABLATION_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation profile: {profile}") from exc


# "L3-kin" ablation: HNU_HUMAN_DYNAMICS=0 removes every human FORCE / MOMENT signal from the levels that
# own human numeric data -- the Human_Metrics "dynamics" namespace (joint moments) of the semantic numeric map and the
# ground-reaction targets (grf_vertical_bw, grf_friction_demand) of the human control head.  Human kinematics, step
# info, obstacle relation and contact / touchdown / lift-off timing are kept.  Default (unset or 1): unchanged.
HUMAN_DYNAMICS = os.environ.get("HNU_HUMAN_DYNAMICS", "1") != "0"
if not HUMAN_DYNAMICS:
    print("[motion_latent_z] HNU_HUMAN_DYNAMICS=0: human dynamics (joint moments, GRF targets) removed", file=sys.stderr)
# "L3-rom" ablation: HNU_HUMAN_TIMING=0 additionally removes every human GAIT-TIMING signal -- the
# contact / touchdown / lift-off targets of the human control head, the Human_Metrics "contact_state" namespace and
# the segment duration.  Left: human joint ranges of motion, step length, obstacle relation (spatial only).
HUMAN_TIMING = os.environ.get("HNU_HUMAN_TIMING", "1") != "0"
if not HUMAN_TIMING:
    print("[motion_latent_z] HNU_HUMAN_TIMING=0: human gait timing (contact / touchdown / lift-off, duration) removed",
          file=sys.stderr)


# Conservative defaults for the command-only query branch shared by Z0-Z4.
# The weak losses make text useful for retrieval without forcing every segment
# of one action to collapse to the same task-level vector.
TASK_QUERY_TUNING_DEFAULTS: Mapping[str, Any] = {
    "lambda_query_align": 0.08,
    "lambda_query_consistency": 0.05,
    "lambda_deployment_consistency": 0.12,
    "lambda_semantic_norm": 0.05,
    "checkpoint_semantic_norm_weight": 0.03,
}


# Z4 contains the noisiest supervision sources: human metrics and human
# semantics.  Keep the additional tuning profile-local so Z0-Z3 retain their
# incremental ablation boundaries.
Z4_TUNING_DEFAULTS: Mapping[str, Any] = {
    "epochs": 180,
    "semantic_ramp_epochs": 30,
    "dropout": 0.18,
    "learning_rate": 1.5e-4,
    "weight_decay": 5.0e-4,
    "motion_noise_std": 0.01,
    "semantic_input_dropout": 0.10,
    "semantic_metric_input_dropout": 0.20,
    # Human text/labels are evidence-grounded but sparse; keep them
    # supplementary to robot-derived supervision rather than allowing them
    # to dominate z_sem. These defaults are intentionally conservative.
    "semantic_text_dropout": 0.08,
    "semantic_text_scale": 0.75,
    "human_semantic_weight": 0.35,
    "human_semantic_input_scale": 0.30,
    "human_metric_input_scale": 0.70,
    "lambda_semantic_decode": 0.10,
    "lambda_align": 0.10,
    "lambda_metric": 0.40,
    "lambda_atoms": 0.30,
    "lambda_atom_ranking": 0.12,
    "lambda_semantic_norm": 0.10,
    "lambda_semantic_covariance": 0.04,
    "lambda_skill_covariance": 0.04,
    "lambda_quality_covariance": 0.02,
    "lambda_skill_metric": 0.10,
    "lambda_skill_atoms": 0.08,
    "atom_focal_gamma": 1.0,
    "rare_sample_boost": 0.25,
    "checkpoint_semantic_f1_weight": 0.08,
    "checkpoint_rare_atom_f1_weight": 0.16,
    "checkpoint_quality_f1_weight": 0.16,
    "checkpoint_semantic_norm_weight": 0.05,
    "early_stop_patience": 32,
}


def apply_profile_defaults(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> None:
    """Apply conservative profile-specific defaults without overriding CLI flags."""
    explicit_options = {
        token.split("=", 1)[0].lstrip("-").replace("-", "_")
        for token in raw_argv
        if token.startswith("--")
    }
    defaults: dict[str, Any] = {}
    if ablation_config(args.ablation_profile)["task_context"]:
        defaults.update(TASK_QUERY_TUNING_DEFAULTS)
    # v2: identical optimisation settings for Z0-Z4 so that the ablation only varies the supervision
    # sources.  The old Z4-only tuning is kept behind HNU_Z4_TUNING=1 for reproducing old numbers.
    if args.ablation_profile == "Z4" and os.environ.get("HNU_Z4_TUNING") == "1":
        defaults.update(Z4_TUNING_DEFAULTS)
    for name, value in defaults.items():
        if name not in explicit_options:
            setattr(args, name, value)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return float(value)
    return default


def nested_get(obj: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    value: Any = obj
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def flatten_numeric(
    obj: Any,
    prefix: str = "",
    *,
    include_bool: bool = False,
) -> dict[str, float]:
    """Flatten reliable numeric leaves while retaining L/R path information.

    Some annotation exports contain summary objects such as ``min``, ``max``
    and ``mean``.  A few of those summaries are internally inconsistent.  Do
    not let an invalid mean or an inverted range become a regression target.
    Other reliable leaves in the same object remain usable.
    """
    result: dict[str, float] = {}
    if isinstance(obj, Mapping):
        def numeric_value(value: Any) -> float | None:
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float, np.number)):
                value = float(value)
                return value if math.isfinite(value) else None
            return None

        lower = numeric_value(obj.get("min"))
        upper = numeric_value(obj.get("max"))
        invalid_summary = (
            lower is not None
            and upper is not None
            and lower > upper + 1e-6
        )
        invalid_summary_leaves: set[str] = set()
        if invalid_summary:
            invalid_summary_leaves.update({"min", "max", "mean", "start", "end"})
        elif lower is not None and upper is not None:
            for summary_key in ("mean", "start", "end"):
                value = numeric_value(obj.get(summary_key))
                if value is not None and not (
                    lower - 1e-6 <= value <= upper + 1e-6
                ):
                    invalid_summary_leaves.add(summary_key)
        for key, value in obj.items():
            if str(key).lower() in invalid_summary_leaves:
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(value, child, include_bool=include_bool))
    elif isinstance(obj, list):
        # Numeric lists usually describe geometry vectors.  Keep their index.
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


def keep_metric_path(path: str, *, allow_endpoint: bool = False) -> bool:
    """Remove identifiers and timestamps that should not be regression labels."""
    lower = path.lower()
    leaf = lower.rsplit(".", 1)[-1]
    excluded_exact = {
        "start_frame",
        "end_frame",
        "start_index",
        "end_index",
        "peak_frame",
        "sample_rate_hz",
        "source_start_s",
        "source_end_s",
    }
    if leaf in excluded_exact:
        return False
    if leaf.endswith("_time_s") or "peak_time_s" in leaf:
        return False
    if "step_index" in lower:
        return False
    # The supplied human ID summaries contain endpoint/min/max inconsistencies.
    if not allow_endpoint and leaf in {"start", "end"}:
        return False
    return True


def vector_from_map(values: Mapping[str, float], keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    vector = np.zeros(len(keys), dtype=np.float32)
    mask = np.zeros(len(keys), dtype=np.float32)
    for index, key in enumerate(keys):
        if key in values and math.isfinite(float(values[key])):
            vector[index] = float(values[key])
            mask[index] = 1.0
    return vector, mask


# The original tokenizer only retained ASCII words.  That made a Chinese
# deployment command such as "???????" become an all-zero vector.
# Keep the dependency-free hashing interface, but add Chinese character
# unigrams/bigrams and a small command ontology so common ASR variants map to
# the same stable semantic tokens as the English task annotations.
TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")
COMMAND_REWRITES = (
    # English rules are deliberately before generic under/beneath cues and
    # use word boundaries, so "five poles" cannot be confused with "three poles".
    (r"\b(?:five|5)[ -]*(?:(?:step|stepping|low|consecutive|successive)[ -]*)*(?:pole|poles|obstacle|obstacles|step|steps|hurdle|hurdles|barrier|barriers|bars?)\b|\bfive[ -]+(?:consecutive|successive|in[ -]+a[ -]+row)[ -]+(?:poles|obstacles|steps|hurdles|barriers)\b", "five_step_pole_run"),
    (r"\b(?:three|3)[ -]*(?:(?:step|stepping|low|consecutive|successive)[ -]*)*(?:pole|poles|obstacle|obstacles|step|steps|hurdle|hurdles|barrier|barriers|bars?)\b|\bthree[ -]+(?:consecutive|successive|in[ -]+a[ -]+row)[ -]+(?:poles|obstacles|steps|hurdles|barriers)\b", "three_step_pole_run"),
    (r"\b(?:pass|go|move|walk|traverse|cross|crawl|squeeze)[ -]+(?:under|beneath|below)[ -]+(?:the[ -]+)?(?:low[ -]+|overhead[ -]+)?bridge\b|\b(?:under|beneath|below)[ -]+(?:the[ -]+)?(?:low[ -]+|overhead[ -]+)?bridge\b|\bbridge[ -]+(?:under|beneath|below)\b", "under_bridge"),
    (r"\u4e94\u6b65\u6869|\u4e94\u4e2a\u6869|\u4e94\u4e2a\u969c\u788d|\u4e94\u6839\u6a2a\u6746|\u4e94\u6839\u4f4e\u77ee\u6a2a\u6746", "five_step_pole_run"),
    (r"\u4e09\u6b65\u6869|\u4e09\u4e2a\u6869|\u4e09\u4e2a\u969c\u788d|\u4e09\u6839\u6a2a\u6746|\u4e09\u6839\u4f4e\u77ee\u6a2a\u6746", "three_step_pole_run"),
    (r"\u6865\u4e0b|\u6865\u5e95|\u4f4e\u77ee\u6865", "under_bridge"),
    (r"\u4f4e\u59ff\u6001|\u4f4e\u8eab\u4f4d|\u8e72\u4e0b|\u5f2f\u8170", "crouched_low_posture"),
    (r"\u8fde\u7eed\u8de8\u8fc7|\u8fde\u7eed\u8d8a\u8fc7|\u4e0d\u505c\u987f\u8de8\u8fc7|\u4e0d\u505c\u987f\u8d8a\u8fc7", "continuous_crossing"),
    (r"\u5f00\u59cb", "start_execute"),
)


def _normalize_command_text(text: str) -> str:
    normalized = str(text or "").lower().strip()
    for pattern, replacement in COMMAND_REWRITES:
        normalized = re.sub(pattern, f" {replacement} ", normalized)
    return normalized


_SENTENCE_CACHE: dict[str, np.ndarray] = {}
_SENTENCE_MODEL: dict[str, Any] = {}
SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384-d, zh + en


def sentence_text_vector(text: str, dim: int) -> np.ndarray:
    """v4: frozen pretrained multilingual sentence embedding (mean pooling, L2-normalised), cached per string."""
    key = str(text or "")
    if key in _SENTENCE_CACHE:
        return _SENTENCE_CACHE[key]
    if not key.strip():
        vec = np.zeros(dim, dtype=np.float32)
    else:
        if "tok" not in _SENTENCE_MODEL:
            from transformers import AutoModel, AutoTokenizer
            _SENTENCE_MODEL["tok"] = AutoTokenizer.from_pretrained(SENTENCE_MODEL_NAME)
            _SENTENCE_MODEL["model"] = AutoModel.from_pretrained(SENTENCE_MODEL_NAME).eval()
        import torch as _t
        tok, mdl = _SENTENCE_MODEL["tok"], _SENTENCE_MODEL["model"]
        enc = tok([key], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with _t.no_grad():
            out = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = ((out * m).sum(1) / m.sum(1).clamp_min(1.0))[0].numpy().astype(np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        if len(v) != dim:
            raise ValueError(f"--text-dim must be {len(v)} with the sentence encoder (got {dim})")
        vec = v
    _SENTENCE_CACHE[key] = vec
    return vec


def hashed_text_vector(text: str, dim: int) -> np.ndarray:
    if os.environ.get("HNU_TEXT_ENCODER", "hash") == "minilm":
        # the n-gram rewrites keep Chinese command normalisation identical for both front-ends
        return sentence_text_vector(str(text or ""), dim)
    return _hashed_text_vector(text, dim)


def _hashed_text_vector(text: str, dim: int) -> np.ndarray:
    """Dependency-free signed n-gram embedding for multilingual text.

    English words/word-bigrams and Chinese character unigrams/bigrams are
    hashed into a fixed vector.  This is not a substitute for a pretrained
    sentence encoder, but unlike the previous ASCII-only implementation it
    preserves information from Chinese ASR commands.
    """
    vector = np.zeros(dim, dtype=np.float32)
    normalized = _normalize_command_text(text)
    tokens = TOKEN_RE.findall(normalized)
    features: list[str] = list(tokens)
    features.extend(f"{a}::{b}" for a, b in zip(tokens, tokens[1:]))
    # Add contiguous Chinese bigrams as well; TOKEN_RE separates characters
    # so the language boundary is explicit and punctuation is ignored.
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for run in chinese_runs:
        features.extend(f"zh:{a}{b}" for a, b in zip(run, run[1:]))
    for token in features:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        integer = int.from_bytes(digest, "little", signed=False)
        index = integer % max(int(dim), 1)
        sign = 1.0 if ((integer >> 8) & 1) else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def build_semantic_branches(
    schema: "FeatureSchema",
    semantic_values: np.ndarray,
    semantic_mask: np.ndarray,
    gait: np.ndarray,
    task_text_vector: np.ndarray,
    task_text_present: float,
    human_text_vector: np.ndarray | None = None,
    human_text_present: float = 0.0,
    *,
    robot_present: float = 0.0,
    human_present: float = 0.0,
    human_metric_scale: float = 1.0,
    mask_annotation_channels: bool = False,
) -> dict[str, np.ndarray]:
    """Build explicitly separated task/robot/human semantic inputs.

    The old implementation concatenated task text, robot metrics, gait and
    human text into one MLP input.  That made the human annotations able to
    overwrite the task query representation and made Z4 hard to interpret.
    This serializer keeps each source in its own branch.  The final scalar in
    every branch is an availability flag; missing/unknown annotations are
    therefore not represented as negative evidence.
    """
    text_dim = int(schema.text_dim)
    metric_dim = len(schema.sem_keys)
    if len(task_text_vector) != text_dim:
        raise ValueError("task_text_vector has an unexpected dimension")
    if human_text_vector is None:
        human_text_vector = np.zeros(text_dim, dtype=np.float32)
    if len(human_text_vector) != text_dim:
        raise ValueError("human_text_vector has an unexpected dimension")
    values = np.asarray(semantic_values, dtype=np.float32)
    mask = np.asarray(semantic_mask, dtype=np.float32)
    if values.shape != (metric_dim,) or mask.shape != (metric_dim,):
        raise ValueError("semantic metric vectors do not match schema.sem_keys")
    gait = np.asarray(gait, dtype=np.float32)
    robot_values = np.zeros(metric_dim, dtype=np.float32)
    robot_mask = np.zeros(metric_dim, dtype=np.float32)
    human_values = np.zeros(metric_dim, dtype=np.float32)
    human_mask = np.zeros(metric_dim, dtype=np.float32)
    for index, key in enumerate(schema.sem_keys):
        # trial/body and segment duration are human-side context; robot.* is
        # reserved for retargeted robot kinematics.  No source is copied into
        # the other branch.
        if key.startswith("robot."):
            robot_values[index] = values[index]
            robot_mask[index] = mask[index]
        elif key.startswith(("human.", "trial.", "segment.")):
            human_values[index] = values[index]
            human_mask[index] = mask[index]
    # Human metrics are teacher-side context.  Keep their mask/availability
    # explicit, but attenuate their numeric magnitude so they cannot dominate
    # the deployable task branch during Z3/Z4 training.
    human_values *= float(np.clip(human_metric_scale, 0.0, 1.0))
    if mask_annotation_channels:
        robot_values.fill(0.0)
        robot_mask.fill(0.0)
        human_values.fill(0.0)
        human_mask.fill(0.0)
        gait = np.zeros_like(gait)
        robot_present = 0.0
        human_present = 0.0
        human_text_vector = np.zeros_like(human_text_vector)
        human_text_present = 0.0
    task_branch = np.concatenate((
        np.asarray(task_text_vector, dtype=np.float32),
        np.asarray([float(task_text_present)], dtype=np.float32),
    )).astype(np.float32)
    robot_branch = np.concatenate((
        robot_values, robot_mask, gait,
        np.asarray([float(robot_present)], dtype=np.float32),
    )).astype(np.float32)
    human_branch = np.concatenate((
        human_values, human_mask,
        np.asarray(human_text_vector, dtype=np.float32),
        # Human numeric metrics are present in Z3 even though human text is
        # intentionally absent.  Tying this flag to human_text_present made
        # the entire Z3 human branch unreachable.
        np.asarray([float(human_present)], dtype=np.float32),
    )).astype(np.float32)
    return {
        "task_branch": task_branch,
        "robot_branch": robot_branch,
        "human_branch": human_branch,
    }


def canonical_atom(atom: str) -> str:
    """Keep the historical local API while using the shared ontology."""
    return shared_canonical_atom(atom)


def text_items(value: Any) -> list[str]:
    """Convert heterogeneous LLM fields into a stable list of text items."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, Mapping):
        items: list[str] = []
        for key, child in value.items():
            if isinstance(child, (bool, int, float, np.number)) and child:
                items.append(str(key))
            elif isinstance(child, str) and child.strip():
                items.append(child.strip())
            else:
                items.extend(text_items(child))
        return items
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for child in value:
            items.extend(text_items(child))
        return items
    return [str(value)]


def mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def phase_contact_hint(gait_phase: str) -> tuple[float, float]:
    """Return expected (left_contact, right_contact), not measured contact."""
    if gait_phase == "left_single_support":
        return 1.0, 0.0
    if gait_phase == "right_single_support":
        return 0.0, 1.0
    if gait_phase.startswith("double_support"):
        return 1.0, 1.0
    # ThreeStepPoleRun/FiveStepPoleRun uses take-off and flight phases rather than gait phases.
    # A take-off phase names the foot leaving the ground, so the opposite
    # foot is the expected support foot. Prepared is double support.
    if gait_phase == "prepared":
        return 1.0, 1.0
    if gait_phase == "left_foot_take_off":
        return 0.0, 1.0
    if gait_phase == "right_foot_take_off":
        return 1.0, 0.0
    if gait_phase == "flight":
        return 0.0, 0.0
    if gait_phase == "landing":
        return 1.0, 1.0
    return 0.0, 0.0


def normalize_phase_label(value: Any) -> str:
    """Normalize phase spellings without creating duplicate vocabulary entries."""
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    label = re.sub(r"_+", "_", label)
    aliases = {
        "stationary": "prepared",
        "preparation": "prepared",
        "prepared_phase": "prepared",
        "left_support_to_toe_off": "left_foot_take_off",
        "right_support_to_toe_off": "right_foot_take_off",
        "flight_phase": "flight",
    }
    return aliases.get(label, label)


def get_phase_label(segment: Mapping[str, Any]) -> str:
    """Read the phase label used by all supported annotation versions."""
    for key in ("gait_phase", "phase", "Phase"):
        value = segment.get(key)
        if value not in (None, ""):
            return normalize_phase_label(value)
    return "unknown"


def support_sides_to_contact(
    support_sides: Sequence[str], gait_phase: str
) -> tuple[float, float]:
    """Convert JSON support_sides to expected (left, right) contact labels.

    The field is stored under robot feasibility/torque screening and is a
    segment-level support annotation.  It is used preferentially; gait phase
    is only a fallback for older JSON files.  This remains an expected/weak
    label, not an independently measured contact force.
    """
    sides = {str(side).upper() for side in support_sides}
    if sides:
        return float("L" in sides), float("R" in sides)
    return phase_contact_hint(gait_phase)


def one_hot(label: str, vocabulary: Sequence[str]) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    try:
        vector[vocabulary.index(label)] = 1.0
    except ValueError:
        pass
    return vector


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "Training requires PyTorch. Install it in your training environment "
            "with `pip install torch`, then rerun with --mode train. Audit mode "
            "does not require PyTorch."
        )


# ---------------------------------------------------------------------------
# JSON and NPZ loading
# ---------------------------------------------------------------------------


def load_json_directory(
    path: Path,
    *,
    dataset_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load semantic JSON documents from one directory."""
    documents: dict[str, dict[str, Any]] = {}

    if not path.is_dir():
        raise FileNotFoundError(path)

    for candidate in sorted(path.glob("**/*.json")):
        relative_parts = candidate.relative_to(path).parts
        if any(part.startswith("_") for part in relative_parts[:-1]):
            continue
        if candidate.name.endswith("_robot_semantics_generation_report.json"):
            continue
        doc = json.loads(candidate.read_text(encoding="utf-8-sig"))
        if not isinstance(doc, Mapping) or not isinstance(
            doc.get("segments"), list
        ):
            continue
        trial_id = str(doc.get("trial_id") or candidate.stem)
        # Keep dataset identity in the key.  This prevents collisions when
        # several datasets contain the same numeric trial suffix.
        if dataset_id and not trial_id.startswith(f"{dataset_id}_"):
            trial_id = f"{dataset_id}_{trial_id}"
        if dataset_id:
            doc = dict(doc)
            doc["_dataset_id"] = dataset_id
            doc["_source_json"] = str(candidate)
            doc["trial_id"] = trial_id
        documents[trial_id] = doc
    if not documents:
        raise ValueError(f"No JSON documents found in {path}")
    return dict(sorted(documents.items()))


def discover_dataset_names(json_root: Path, npz_root: Path) -> list[str]:
    """Return action directories containing usable JSON and NPZ files."""
    if not json_root.is_dir() or not npz_root.is_dir():
        return []
    json_names = set()
    npz_names = set()
    for item in json_root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            if any(item.glob("**/*.json")):
                json_names.add(item.name)
    for item in npz_root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            if any(item.glob("**/*.npz")):
                npz_names.add(item.name)
    return sorted(json_names.intersection(npz_names))


def _merge_nested_mappings(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge fallback retargeting fields without replacing richer annotations."""
    result = dict(fallback)
    for key, value in primary.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_nested_mappings(value, result[key])
        else:
            result[key] = value
    return result


def load_retargeted_dynamics_directory(
    path: Path,
    *,
    dataset_id: str,
) -> dict[str, dict[str, Any]]:
    """Load ``*_Dynamics.json`` files keyed by the canonical dataset trial ID."""
    documents: dict[str, dict[str, Any]] = {}
    if not path.is_dir():
        raise FileNotFoundError(path)
    for candidate in sorted(path.glob("**/*_Dynamics.json")):
        doc = json.loads(candidate.read_text(encoding="utf-8-sig"))
        stem = re.sub(r"_Dynamics$", "", candidate.stem, flags=re.IGNORECASE)
        trial_id = str(doc.get("trial_id") or stem)
        if not trial_id.startswith(f"{dataset_id}_"):
            trial_id = f"{dataset_id}_{trial_id}"
        doc = dict(doc)
        doc["_dataset_id"] = dataset_id
        doc["_source_retargeted_json"] = str(candidate)
        doc["trial_id"] = trial_id
        documents[trial_id] = doc
    return dict(sorted(documents.items()))


def discover_npz_files(npz_dir: Path, trial_ids: Iterable[str]) -> dict[str, Path]:
    candidates = sorted(npz_dir.glob("**/*.npz"))
    mapping: dict[str, Path] = {}
    for trial_id in trial_ids:
        trial_key = trial_id.lower()
        trial_number_match = re.search(r"(\d+)$", trial_key)
        trial_number = (
            int(trial_number_match.group(1)) if trial_number_match else None
        )
        exact = []
        for path in candidates:
            stem = path.stem.lower()
            # Prefer an exact conventional basename.  The boundary check
            # prevents trial 001 from accidentally matching trial 0010.
            if stem in {trial_key, f"{trial_key}_dynamics"}:
                exact.append(path)
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(trial_key)}(?![0-9])", stem):
                exact.append(path)
                continue
            # Some future exports may be named ``001_Dynamics.npz`` rather
            # than ``ActionName_001_Dynamics.npz``.  Since this function is
            # already scoped to one action directory, a unique numeric suffix
            # is safe and keeps the loader format-tolerant.
            if trial_number is not None:
                stem_without_dynamics = re.sub(
                    r"_dynamics$", "", stem, flags=re.IGNORECASE
                )
                candidate_number_match = re.search(
                    r"(\d+)$", stem_without_dynamics
                )
                if (
                    candidate_number_match is not None
                    and int(candidate_number_match.group(1)) == trial_number
                ):
                    exact.append(path)
        if len(exact) == 1:
            mapping[trial_id] = exact[0]
        elif len(exact) > 1:
            # Prefer the conventional Dynamics filename, then shortest name.
            exact.sort(key=lambda p: ("dynamics" not in p.stem.lower(), len(p.name), p.name))
            mapping[trial_id] = exact[0]
    return mapping


def load_dataset_documents(
    dataset_names: Sequence[str],
    json_root: Path,
    retargeted_json_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load semantic documents and fill missing robot fields from retargeted JSON."""
    documents: dict[str, dict[str, Any]] = {}
    for dataset_name in dataset_names:
        path = json_root / dataset_name
        loaded = load_json_directory(path, dataset_id=dataset_name)
        if retargeted_json_root is not None:
            retargeted_path = retargeted_json_root / dataset_name
            retargeted = (
                load_retargeted_dynamics_directory(
                    retargeted_path, dataset_id=dataset_name
                )
                if retargeted_path.is_dir()
                else {}
            )
            for trial_id, doc in loaded.items():
                fallback = retargeted.get(trial_id)
                if fallback is None:
                    continue
                enriched = dict(doc)
                enriched_segments = []
                fallback_segments = fallback.get("segments", {})
                for segment in doc.get("segments", []):
                    segment_id = str(segment.get("segment_id", "unknown"))
                    if isinstance(fallback_segments, Mapping):
                        fallback_segment = fallback_segments.get(segment_id, {})
                    elif isinstance(fallback_segments, list):
                        fallback_segment = next(
                            (
                                item
                                for item in fallback_segments
                                if isinstance(item, Mapping)
                                and str(item.get("segment_id", ""))
                                == segment_id
                            ),
                            {},
                        )
                    else:
                        fallback_segment = {}
                    if isinstance(fallback_segment, Mapping):
                        robot = segment.get("Retargeted_Robot_Metrics", {})
                        if not isinstance(robot, Mapping):
                            robot = {}
                        merged_robot = _merge_nested_mappings(
                            robot, fallback_segment
                        )
                        # Older semantic annotations may omit alignment while
                        # the retargeted dynamics JSON stores the valid frame
                        # interval under data_availability.
                        if "npz_alignment" not in merged_robot:
                            availability = merged_robot.get("data_availability", {})
                            if isinstance(availability, Mapping):
                                start_index = availability.get("start_index")
                                end_index = availability.get("end_index")
                                if start_index is not None and end_index is not None:
                                    merged_robot["npz_alignment"] = {
                                        "start_index": start_index,
                                        "end_index": end_index,
                                        "alignment_method": "retargeted_dynamics_json",
                                    }
                        segment = dict(segment)
                        segment["Retargeted_Robot_Metrics"] = merged_robot
                        enriched_segments.append(segment)
                enriched["segments"] = enriched_segments
                enriched["_source_retargeted_json"] = fallback.get(
                    "_source_retargeted_json"
                )
                loaded[trial_id] = enriched
        overlap = set(documents).intersection(loaded)
        if overlap:
            raise ValueError(f"Duplicate trial IDs after dataset prefixing: {sorted(overlap)}")
        documents.update(loaded)
    if not documents:
        raise ValueError(f"No JSON documents found under {json_root}")
    return dict(sorted(documents.items()))


def trial_id_from_npz_path(path: Path, dataset_id: str) -> str:
    """Build a canonical trial ID from common redirected NPZ basenames."""
    stem = re.sub(r"_Dynamics$", "", path.stem, flags=re.IGNORECASE)
    if stem.startswith(f"{dataset_id}_"):
        suffix = stem[len(dataset_id) + 1 :]
    else:
        match = re.search(r"(\d+)$", stem)
        suffix = match.group(1) if match else stem
    return f"{dataset_id}_{suffix}"


def load_npz_only_documents(
    dataset_names: Sequence[str],
    npz_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Create placeholder documents so Z0 can train from NPZ files alone."""
    documents: dict[str, dict[str, Any]] = {}
    mapping: dict[str, Path] = {}
    for dataset_id in dataset_names:
        dataset_dir = npz_root / dataset_id
        if not dataset_dir.is_dir():
            raise NotADirectoryError(dataset_dir)
        for path in sorted(dataset_dir.glob("**/*.npz")):
            trial_id = trial_id_from_npz_path(path, dataset_id)
            if trial_id in mapping:
                raise ValueError(f"Duplicate NPZ trial ID in Z0: {trial_id}")
            with np.load(path, allow_pickle=False) as npz:
                length = len(checked_array(npz, "qpos"))
                fps = as_float(npz.get("fps", 120.0), 120.0)
            # One synthetic segment covers the complete trajectory.  Z0 has no
            # JSON-derived phase or annotation, so the feature builder receives
            # only its NPZ motion stream and zero-valued supervision masks.
            documents[trial_id] = {
                "trial_id": trial_id,
                "_dataset_id": dataset_id,
                "_source_npz": str(path),
                "segments": [
                    {
                        "segment_id": "full_trial",
                        "phase": "unknown",
                        "time": {"duration_s": length / max(fps, 1e-6)},
                        "Retargeted_Robot_Metrics": {
                            "npz_alignment": {
                                "start_index": 0,
                                "end_index": length,
                            }
                        },
                    }
                ],
            }
            mapping[trial_id] = path
    if not mapping:
        raise ValueError(f"No NPZ files found under {npz_root}")
    return dict(sorted(documents.items())), dict(sorted(mapping.items()))


def collect_task_context_text(
    doc: Mapping[str, Any],
    profile: str = "Z4",
) -> str:
    """Return only deployment-available task text with stable field markers."""
    if not ablation_config(profile)["task_context"]:
        return ""
    task = mapping_or_empty(doc.get("task", {}))
    parts: list[str] = []
    for key, marker in (
        ("task_name_en", "task_name"),
        ("task_goal", "task_goal"),
        ("task_family", "task_family"),
    ):
        for item in text_items(task.get(key)):
            parts.append(f"{marker}: {item}")
    return " . ".join(parts)


def collect_task_query_text(
    doc: Mapping[str, Any],
    profile: str = "Z4",
) -> str:
    """Match deployment queries by exposing only the marked task_goal field."""
    if not ablation_config(profile)["task_context"]:
        return ""
    task = mapping_or_empty(doc.get("task", {}))
    return " . ".join(
        f"task_goal: {item}" for item in text_items(task.get("task_goal"))
    )


def _evidence_grounded_human_text(
    segment: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return concise human semantic text only when it is evidence-backed.

    The returned text is deliberately restricted to fields that can be traced
    to ``Human_Metrics`` evidence.  It is used by the common offline
    segment-query protocol; the task-level ASR interface remains unchanged.
    """
    raw = nested_get(segment, ("Human_Metrics", "human_semantics"), {})
    if not isinstance(raw, Mapping):
        return ()
    if str(raw.get("schema_version", "")) != "human_semantics_v2":
        return ()
    if str(raw.get("annotation_state", "")) not in {
        "metric_backed", "phase_contact_backed",
    }:
        return ()
    evidence = raw.get("evidence")
    if not isinstance(evidence, (list, tuple)) or not evidence:
        return ()
    atoms = tuple(sorted({
        canonical_atom(str(item))
        for item in text_items(raw.get("skill_atoms"))
        if canonical_atom(str(item)) in ACTIVE_ATOM_SET
    }))
    hints = tuple(text_items(raw.get("human_derived_control_hints")))
    captions = tuple(text_items(raw.get("short_caption_en")))
    # Atoms are canonical and compact; hints/caption provide natural language
    # context but are capped to avoid turning the query into a label dump.
    parts: list[str] = []
    if atoms:
        parts.append("human_skill_atoms: " + ", ".join(atoms))
    if hints:
        parts.append("human_control_hint: " + hints[0])
    if captions and not captions[0].lower().startswith(
        ("flight phase with no", "prepared phase with no", "unknown phase with no")
    ):
        parts.append("human_description: " + captions[0])
    return tuple(parts)


def collect_segment_query_text(
    doc: Mapping[str, Any],
    segment: Mapping[str, Any],
    profile: str = "Z4",
) -> str:
    """Build one deployment-observable segment query for every profile.

    The query contains only the received task instruction and a phase label
    that can be estimated from robot contact/state.  Human/robot annotations
    are evaluation labels or training-only privileged supervision; copying
    either into this query would leak Z4 information into Z0--Z3 and would be
    impossible to reproduce on the robot.

    ``profile`` is retained for API compatibility and deliberately does not
    alter the query.
    """
    parts: list[str] = []
    task = mapping_or_empty(doc.get("task", {}))
    goals = text_items(task.get("task_goal"))
    if goals:
        parts.append("task_goal: " + goals[0])
    phase = get_phase_label(segment)
    if phase:
        parts.append("segment_phase: " + phase)
    return " . ".join(parts)


def collect_human_semantic_text(
    segment: Mapping[str, Any],
    profile: str = "Z4",
) -> str:
    """Return only evidence-grounded human text for the semantic encoder.

    Keeping this component separate lets Z4 attenuate sparse/noisy human text
    without changing the common task/segment query protocol used by Z0--Z4.
    """
    if not ablation_config(profile)["human_semantics"]:
        return ""
    raw_human_sem = nested_get(
        segment, ("Human_Metrics", "human_semantics"), {}
    )
    if (
        not isinstance(raw_human_sem, Mapping)
        or raw_human_sem.get("schema_version") != "human_semantics_v2"
    ):
        return ""
    human_sem = mapping_or_empty(raw_human_sem)
    human_annotation_state = str(human_sem.get("annotation_state", ""))
    human_evidence = human_sem.get("evidence")
    if (
        human_annotation_state not in {"metric_backed", "phase_contact_backed"}
        or not isinstance(human_evidence, (list, tuple))
        or not human_evidence
    ):
        return ""
    parts: list[str] = []
    for key in (
        "short_caption_en",
        "long_description_en",
        "human_derived_control_hints",
        "movement_description_en",
    ):
        for item in text_items(human_sem.get(key)):
            parts.append(f"human_{key}: {item}")
    return " . ".join(parts)


def semantic_numeric_map(
    doc: Mapping[str, Any],
    segment: Mapping[str, Any],
    profile: str = "Z4",
) -> dict[str, float]:
    """Metrics allowed to shape skill content (not robot feasibility)."""
    config = ablation_config(profile)
    result: dict[str, float] = {}
    if config["human_numeric"]:
        body_height = nested_get(doc, ("subject_model", "body_height_m"))
        body_weight = nested_get(doc, ("subject_model", "body_weight_kg"))
        duration = nested_get(segment, ("time", "duration_s"))
        if body_height is not None:
            result["trial.body_height_m"] = as_float(body_height)
        if body_weight is not None:
            result["trial.body_weight_kg"] = as_float(body_weight)
        if duration is not None and HUMAN_TIMING:  # (L3-rom ablation: no segment duration)
            result["segment.duration_s"] = as_float(duration)

    if config["human_numeric"]:
        human = mapping_or_empty(nested_get(segment, ("Human_Metrics",), {}))
        for namespace in (
            "contact_state",
            "step_info",
            "obstacle_relation",
            "kinematics",
            "dynamics",
        ):
            if namespace == "dynamics" and not HUMAN_DYNAMICS:  # L3-kin ablation
                continue
            if namespace == "contact_state" and not HUMAN_TIMING:  # L3-rom ablation
                continue
            values = flatten_numeric(
                human.get(namespace, {}),
                f"human.{namespace}",
                include_bool=True,
            )
            result.update(
                {key: value for key, value in values.items() if keep_metric_path(key)}
            )
    if config["robot_numeric"]:
        robot = mapping_or_empty(
            nested_get(segment, ("Retargeted_Robot_Metrics",), {})
        )
        values = flatten_numeric(
            robot.get("kinematics", {}), "robot.kinematics", include_bool=True
        )
        result.update(
            {key: value for key, value in values.items() if keep_metric_path(key)}
        )
    return result


def quality_numeric_map(
    segment: Mapping[str, Any],
    profile: str = "Z4",
) -> dict[str, float]:
    """Continuous labels used only for z_quality and quality prediction."""
    config = ablation_config(profile)
    if not config["robot_numeric"]:
        return {}
    robot = mapping_or_empty(
        nested_get(segment, ("Retargeted_Robot_Metrics",), {})
    )
    result: dict[str, float] = {}
    # v2: "execution" = statistics of simulated expert rollouts over the segment (success rate, executed
    # torque, measured GRF, tracking error).  Labels only -- never an encoder input.
    for namespace in ("dynamics", "feasibility", "execution"):
        values = flatten_numeric(
            robot.get(namespace, {}), f"robot_quality.{namespace}", include_bool=True
        )
        result.update({key: value for key, value in values.items() if keep_metric_path(key)})
    # Shared quality axes make the heterogeneous datasets comparable. Raw
    # feasibility fields remain available alongside these derived targets.
    alignment = nested_get(robot, ("npz_alignment",), {}) or {}
    start = as_float(alignment.get("start_index"))
    end = as_float(alignment.get("end_index"))
    risk = segment_risk_score(segment, max(int(end - start), 1), profile)
    result["robot_quality.derived.risk_score"] = risk
    result["robot_quality.derived.feasibility_score"] = float(math.exp(-risk))
    whole = nested_get(robot, ("feasibility", "torque_screening", "whole_body"), {}) or {}
    joint = nested_get(robot, ("feasibility", "state_validity", "joint_limit"), {}) or {}
    collision = nested_get(robot, ("feasibility", "state_validity", "collision"), {}) or {}
    tags = feasibility_tags(segment, profile)
    # Shared axes make z_quality comparable across walking and jumping tasks.
    result["robot_quality.derived.peak_torque_ratio"] = as_float(
        whole.get("peak_limit_ratio"), 1.0
    )
    result["robot_quality.derived.torque_violation_ratio"] = as_float(
        whole.get("violation_frame_ratio")
    )
    result["robot_quality.derived.joint_limit_ratio"] = as_float(
        joint.get("violation_frame_ratio")
    )
    result["robot_quality.derived.collision_ratio"] = as_float(
        collision.get("self_collision_frame_count")
    ) / max(int(end - start), 1)
    result["robot_quality.derived.self_penetration_norm"] = min(
        as_float(collision.get("max_self_penetration_mm")) / 20.0, 2.0
    )
    result["robot_quality.derived.floor_penetration_norm"] = min(
        as_float(collision.get("max_floor_penetration_mm")) / 10.0, 2.0
    )
    result["robot_quality.derived.contact_mismatch"] = float(
        "contact_mismatch" in tags
    )
    return result


def semantic_atom_states(
    segment: Mapping[str, Any],
    doc: Mapping[str, Any] | None = None,
    profile: str = "Z4",
) -> dict[str, dict[str, Any]]:
    """Return positive/unknown/known-absent atom states by source.

    New annotations use a three-way state. Legacy annotations containing only
    ``skill_atoms`` provide positive labels only; omitted atoms are treated as
    unknown because the old generator did not distinguish absent from
    unobserved. The training code combines source states conservatively and
    masks unknown atoms per label.
    """
    config = ablation_config(profile)
    sources: dict[str, tuple[bool, Mapping[str, Any]]] = {}
    if config["human_semantics"]:
        human_metrics = segment.get("Human_Metrics")
        value = (
            human_metrics.get("human_semantics")
            if isinstance(human_metrics, Mapping)
            else None
        )
        # A missing field must not be converted into an existing empty
        # annotation. Missing human semantics provide no atom labels.
        sources["human"] = (
            isinstance(value, Mapping),
            mapping_or_empty(value),
        )
    if config["robot_semantics"]:
        robot_metrics = segment.get("Retargeted_Robot_Metrics")
        value = (
            robot_metrics.get("robot_semantics")
            if isinstance(robot_metrics, Mapping)
            else None
        )
        sources["robot"] = (
            isinstance(value, Mapping),
            mapping_or_empty(value),
        )

    result: dict[str, dict[str, set[str] | bool]] = {}
    for source, (present, semantics) in sources.items():
        if not present:
            result[source] = {
                "present": False,
                "tri_state": False,
                "legacy_closed_world": False,
                "positive": set(),
                "unknown": set(),
                "absent": set(),
            }
            continue
        if source == "human":
            # Human annotations are weak supervision. Normalize both the
            # evidence-backed v2 schema and legacy positive-only JSON before
            # they reach the atom loss. Legacy positives without evidence are
            # intentionally treated as unknown.
            semantics = mapping_or_empty(
                normalize_existing_human_semantics(semantics, segment)
            )
        positive = {
            canonical_atom(x)
            for x in text_items(semantics.get("skill_atoms"))
            if canonical_atom(x) in ACTIVE_ATOM_SET
        }
        if source == "robot":
            # A collision-free status says the retargeted robot did not
            # collide; it does not prove that the motion used an active
            # collision-avoidance skill. Require obstacle-relation evidence
            # before using that robot atom as Z4 supervision.
            robot_evidence = semantics.get("evidence", [])
            has_obstacle_evidence = any(
                isinstance(item, Mapping)
                and canonical_atom(item.get("atom", "")) == "collision_avoidance"
                and str(item.get("source", "")).startswith(
                    "Retargeted_Robot_Metrics.obstacle_relation."
                )
                for item in robot_evidence
            ) if isinstance(robot_evidence, (list, tuple)) else False
            if "collision_avoidance" in positive and not has_obstacle_evidence:
                positive.discard("collision_avoidance")
        tri_state = any(
            key in semantics for key in ("unknown_atoms", "known_absent_atoms")
        )
        if tri_state:
            unknown = {
                canonical_atom(x)
                for x in text_items(semantics.get("unknown_atoms"))
                if canonical_atom(x) in ACTIVE_ATOM_SET
            }
            absent = {
                canonical_atom(x)
                for x in text_items(semantics.get("known_absent_atoms"))
                if canonical_atom(x) in ACTIVE_ATOM_SET
            }
            unknown -= positive
            absent -= positive
            absent -= unknown
        else:
            # Old JSON has only a positive list. It is not safe to infer
            # negative labels from omissions because the old prompt did not
            # require a complete candidate partition.
            unknown = set(CANONICAL_ATOMS).difference(positive)
            absent = set()
        # High-confidence phase/contact contradictions override an omitted or
        # stale unknown label. This supplies deterministic negative
        # supervision for future actions without treating all omissions as
        # negative labels.
        phase_absent = phase_known_absent_atoms(segment)
        absent.update(phase_absent - positive)
        unknown -= absent
        unknown |= ACTIVE_ATOM_SET.difference(positive | absent | unknown)
        result[source] = {
            "present": True,
            "tri_state": tri_state,
            "legacy_closed_world": False,
            "positive": positive,
            "unknown": unknown,
            "absent": absent,
        }

    # Task text is a common command input, not an atom label source.  Keeping
    # it out of this target builder prevents Z0/Z1 from receiving semantic
    # supervision that their ablation profiles intentionally omit.
    return result


def semantic_atoms(
    segment: Mapping[str, Any],
    doc: Mapping[str, Any] | None = None,
    profile: str = "Z4",
) -> tuple[set[str], set[str]]:
    """Compatibility view returning positive human and robot atoms."""
    states = semantic_atom_states(segment, doc, profile)
    human = set(states.get("human", {}).get("positive", set()))
    robot = set(states.get("robot", {}).get("positive", set()))
    if "task" in states and not human:
        human = set(states["task"].get("positive", set()))
    return human, robot


def combined_atom_states(
    segment: Mapping[str, Any],
    doc: Mapping[str, Any] | None,
    profile: str,
) -> tuple[set[str], set[str], set[str], bool, bool]:
    """Combine available human/robot annotations into per-atom three states."""
    states = semantic_atom_states(segment, doc, profile)
    positive: set[str] = set()
    absent: set[str] = set()
    unknown: set[str] = set()
    annotated = False
    source_values = [
        value for source, value in states.items() if source != "task"
    ]
    has_tri_state = any(bool(value.get("tri_state", False)) for value in source_values)
    # Legacy binary annotations are positive-only. They must not create
    # implicit negative supervision for atoms omitted by the old generator.
    legacy_closed_world = False
    for value in source_values:
        if not bool(value.get("present", False)):
            continue
        annotated = True
        positive.update(value.get("positive", set()))
        absent.update(value.get("absent", set()))
        unknown.update(value.get("unknown", set()))
    absent -= positive
    unknown -= positive
    unknown -= absent
    return positive, unknown, absent, annotated, legacy_closed_world


# Quality v2 is intentionally independent of the legacy feasibility-tag names.
# The four axes are fixed so validate_latents.py can consume the same schema.
QUALITY_V2_AXES = ("torque", "joint_limit", "collision", "contact")
QUALITY_LABEL_VERSION = "v2"
QUALITY_V2_STATES = ("positive", "negative", "unknown")
QUALITY_V2_POSITIVE = 1.0
QUALITY_V2_NEGATIVE = 0.0
QUALITY_V2_UNKNOWN = -1.0
QUALITY_V2_DEFAULT_THRESHOLDS = {
    "torque": {"warning_peak_ratio": 0.85, "risk_peak_ratio": 1.00,
                "warning_violation_ratio": 0.01},
    # A joint-limit violation is negative only once it reaches the configured
    # warning threshold. Tiny numerical excursions are unknown rather than
    # automatically negative; this avoids making every sample a failure.
    "joint_limit": {"warning_violation_ratio": 0.01, "risk_violation_ratio": 0.01,
                     "warning_max_excess": 1e-3, "risk_max_excess": 1e-3},
    "collision": {"warning_self_penetration_mm": 1.0,
                   "risk_self_penetration_mm": 2.0,
                   "warning_collision_ratio": 0.01},
    "contact": {"warning_floor_penetration_mm": 2.0,
                 "risk_floor_penetration_mm": 10.0},
}
QUALITY_V2_LEGACY_WEAK_TAGS = {
    "torque_positive": frozenset({"actuator_limits_satisfied"}),
    "torque_negative": frozenset({"actuator_load_risk", "actuator_limits_risk"}),
    "joint_limit_positive": frozenset({"joint_limits_satisfied"}),
    "joint_limit_negative": frozenset({"joint_limit_risk"}),
    "collision_positive": frozenset({"collision_free_verified"}),
    "collision_negative": frozenset({"collision_risk"}),
    "contact_positive": frozenset({"contact_consistent"}),
    "contact_negative": frozenset({"contact_mismatch"}),
}


def feasibility_tags(segment: Mapping[str, Any], profile: str = "Z4") -> set[str]:
    if not ablation_config(profile)["robot_semantics"]:
        return set()
    tags = nested_get(
        segment,
        ("Retargeted_Robot_Metrics", "robot_semantics", "feasibility_tags"),
        [],
    ) or []
    return {str(tag) for tag in text_items(tags)}


def _quality_v2_state(value: float | None, *, positive_when_low: bool,
                      warning: float | None, risk: float | None) -> str:
    """Map a physical metric to positive/negative/unknown."""
    if value is None or not math.isfinite(float(value)):
        return "unknown"
    value = float(value)
    if positive_when_low:
        if risk is not None and value >= risk:
            return "negative"
        if warning is not None and value > warning:
            return "unknown"
        return "positive"
    if risk is not None and value > risk:
        return "negative"
    if warning is not None and value >= warning:
        return "unknown"
    return "positive"


def derive_quality_v2_states(
    segment: Mapping[str, Any],
    profile: str = "Z4",
) -> dict[str, str]:
    """Derive quality-v2 states from physical metrics, with weak tag fallback.

    State semantics: positive=1, negative=0, unknown=-1.  Unknown is retained
    explicitly and is masked out by ``quality_v2_targets_and_mask``.
    Physical indicators take precedence; legacy tags are only used when the
    corresponding physical field is unavailable and never override a metric.
    """
    if not ablation_config(profile)["robot_numeric"]:
        return {axis: "unknown" for axis in QUALITY_V2_AXES}
    robot = mapping_or_empty(nested_get(segment, ("Retargeted_Robot_Metrics",), {}))
    feasibility = mapping_or_empty(robot.get("feasibility"))
    torque = mapping_or_empty(mapping_or_empty(feasibility.get("torque_screening")).get("whole_body"))
    joint = mapping_or_empty(mapping_or_empty(feasibility.get("state_validity")).get("joint_limit"))
    collision = mapping_or_empty(mapping_or_empty(feasibility.get("state_validity")).get("collision"))
    tags = feasibility_tags(segment, profile)
    t = QUALITY_V2_DEFAULT_THRESHOLDS["torque"]
    j = QUALITY_V2_DEFAULT_THRESHOLDS["joint_limit"]
    c = QUALITY_V2_DEFAULT_THRESHOLDS["collision"]
    contact = QUALITY_V2_DEFAULT_THRESHOLDS["contact"]

    peak = as_float(torque.get("peak_limit_ratio"), float("nan"))
    violation = as_float(torque.get("violation_frame_ratio"), float("nan"))
    any_violation = torque.get("any_limit_violation")
    if math.isfinite(peak) or math.isfinite(violation) or isinstance(any_violation, bool):
        torque_state = "negative" if (
            (math.isfinite(peak) and peak >= t["risk_peak_ratio"])
            or (math.isfinite(violation) and violation >= t["warning_violation_ratio"])
            or any_violation is True
        ) else ("unknown" if (
            (math.isfinite(peak) and peak > t["warning_peak_ratio"])
            or (math.isfinite(violation) and violation > 0.0)
        ) else "positive")
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["torque_negative"]:
        torque_state = "negative"
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["torque_positive"]:
        torque_state = "positive"
    else:
        torque_state = "unknown"

    joint_ratio = as_float(joint.get("violation_frame_ratio"), float("nan"))
    joint_count = as_float(joint.get("violation_frame_count"), float("nan"))
    excess = as_float(joint.get("max_excess"), float("nan"))
    if any(math.isfinite(v) for v in (joint_ratio, joint_count, excess)):
        joint_state = "negative" if (
            (math.isfinite(joint_ratio) and joint_ratio >= j["risk_violation_ratio"])
            or (math.isfinite(joint_count) and joint_count >= 1.0
                and (not math.isfinite(joint_ratio) or joint_ratio >= j["warning_violation_ratio"]))
            or (math.isfinite(excess) and excess >= j["risk_max_excess"])
        ) else ("unknown" if (
            (math.isfinite(joint_ratio) and joint_ratio > 0.0)
            or (math.isfinite(joint_count) and joint_count > 0.0)
            or (math.isfinite(excess) and excess > 0.0)
        ) else "positive")
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["joint_limit_negative"]:
        joint_state = "negative"
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["joint_limit_positive"]:
        joint_state = "positive"
    else:
        joint_state = "unknown"

    self_frames = as_float(collision.get("self_collision_frame_count"), float("nan"))
    self_pen = as_float(collision.get("max_self_penetration_mm"), float("nan"))
    floor_pen = as_float(collision.get("max_floor_penetration_mm"), float("nan"))
    if math.isfinite(self_frames) or math.isfinite(self_pen):
        collision_state = "negative" if (
            (math.isfinite(self_frames) and self_frames > 0.0)
            or (math.isfinite(self_pen) and self_pen >= c["risk_self_penetration_mm"])
        ) else ("unknown" if (
            math.isfinite(self_pen) and self_pen > c["warning_self_penetration_mm"]
        ) else "positive")
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["collision_negative"]:
        collision_state = "negative"
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["collision_positive"]:
        collision_state = "positive"
    else:
        collision_state = "unknown"

    # Contact is deliberately conservative: floor contact is not itself a
    # failure, and no contact-consistency physical metric exists in the files.
    # Therefore only an explicit physical mismatch (if supplied) is negative;
    # otherwise use tags weakly, or remain unknown.
    mismatch = robot.get("contact_mismatch")
    if isinstance(mismatch, bool):
        contact_state = "negative" if mismatch else "positive"
    elif math.isfinite(floor_pen):
        contact_state = "negative" if floor_pen >= contact["risk_floor_penetration_mm"] else (
            "unknown" if floor_pen > contact["warning_floor_penetration_mm"] else "positive"
        )
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["contact_negative"]:
        contact_state = "negative"
    elif tags & QUALITY_V2_LEGACY_WEAK_TAGS["contact_positive"]:
        contact_state = "positive"
    else:
        contact_state = "unknown"
    return {"torque": torque_state, "joint_limit": joint_state,
            "collision": collision_state, "contact": contact_state}


def quality_v2_targets_and_mask(
    segment: Mapping[str, Any], profile: str = "Z4",
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Return fixed-axis v2 targets and masks: positive=1, negative=0, unknown=0."""
    states = derive_quality_v2_states(segment, profile)
    target = np.zeros(len(QUALITY_V2_AXES), dtype=np.float32)
    mask = np.zeros(len(QUALITY_V2_AXES), dtype=np.float32)
    for index, axis in enumerate(QUALITY_V2_AXES):
        if states[axis] == "positive":
            target[index] = QUALITY_V2_POSITIVE
            mask[index] = 1.0
        elif states[axis] == "negative":
            target[index] = QUALITY_V2_NEGATIVE
            mask[index] = 1.0
        else:
            target[index] = QUALITY_V2_UNKNOWN
            mask[index] = 0.0
    return target, mask, states


def segment_risk_score(
    segment: Mapping[str, Any],
    number_of_frames: int,
    profile: str = "Z4",
) -> float:
    """A transparent heuristic used only to downweight trajectory reconstruction.

    Bad samples remain useful for semantic and quality supervision.
    """
    if not ablation_config(profile)["robot_numeric"]:
        return 0.0
    root = nested_get(segment, ("Retargeted_Robot_Metrics", "feasibility"), {}) or {}
    whole = nested_get(root, ("torque_screening", "whole_body"), {}) or {}
    joint = nested_get(root, ("state_validity", "joint_limit"), {}) or {}
    collision = nested_get(root, ("state_validity", "collision"), {}) or {}
    peak_ratio = as_float(whole.get("peak_limit_ratio"), 1.0)
    violation_ratio = as_float(whole.get("violation_frame_ratio"))
    joint_ratio = as_float(joint.get("violation_frame_ratio"))
    max_excess = as_float(joint.get("max_excess"))
    self_frames = as_float(collision.get("self_collision_frame_count"))
    self_pen_mm = as_float(collision.get("max_self_penetration_mm"))
    floor_pen_mm = as_float(collision.get("max_floor_penetration_mm"))
    tags = feasibility_tags(segment, profile)
    contact_penalty = 0.5 if "contact_mismatch" in tags else 0.0
    risk = (
        max(peak_ratio - 1.0, 0.0)
        + violation_ratio
        + joint_ratio
        + min(max_excess / 0.10, 3.0)
        + min(self_frames / max(number_of_frames, 1), 1.0)
        + min(self_pen_mm / 20.0, 2.0)
        + 0.25 * min(floor_pen_mm / 10.0, 2.0)
        + contact_penalty
    )
    return float(risk)


@dataclass
class FeatureSchema:
    profile: str
    sem_keys: list[str]
    quality_keys: list[str]
    atom_vocab: list[str]
    quality_tag_vocab: list[str]
    gait_vocab: list[str]
    text_dim: int
    task_context: bool = False
    quality_label_version: str = QUALITY_LABEL_VERSION
    task_query_version: str = "task_goal_only_v1"
    segment_query_version: str = SEGMENT_QUERY_VERSION
    checkpoint_format_version: str = CHECKPOINT_FORMAT_VERSION
    training_protocol_version: str = TRAINING_PROTOCOL_VERSION
    semantic_fusion_version: str = SEMANTIC_FUSION_VERSION
    quality_input_version: str = QUALITY_INPUT_VERSION
    quality_conditioning_version: str = QUALITY_CONDITIONING_VERSION
    semantic_branch_layout: tuple[str, ...] = (
        "task_branch:hashed_task_text+text_present",
        "robot_branch:robot_metric_values+robot_metric_mask+gait+availability",
        "human_branch:human_metric_values+human_metric_mask+human_text+availability",
    )
    semantic_fusion_requires_new_checkpoint: bool = True

    @classmethod
    def discover(
        cls,
        docs: Mapping[str, Mapping[str, Any]],
        text_dim: int,
        profile: str = "Z4",
    ) -> "FeatureSchema":
        sem_keys: set[str] = set()
        quality_keys: set[str] = set()
        atoms: set[str] = set()
        tags: set[str] = set()
        phases: set[str] = set(DEFAULT_GAIT_PHASES)
        for doc in docs.values():
            for segment in doc.get("segments", []):
                sem_keys.update(semantic_numeric_map(doc, segment, profile))
                quality_keys.update(quality_numeric_map(segment, profile))
                atom_states = semantic_atom_states(segment, doc, profile)
                for state in atom_states.values():
                    atoms.update(
                        atom
                        for atom in state.get("positive", set())
                        if atom in ACTIVE_ATOM_SET
                    )
                    atoms.update(
                        atom
                        for atom in state.get("unknown", set())
                        if atom in ACTIVE_ATOM_SET
                    )
                    atoms.update(
                        atom
                        for atom in state.get("absent", set())
                        if atom in ACTIVE_ATOM_SET
                    )
                tags.update(feasibility_tags(segment, profile))
                phases.add(get_phase_label(segment))
        if (
            profile != "Z0"
            and (
                ablation_config(profile)["robot_semantics"]
                or ablation_config(profile)["human_semantics"]
            )
        ):
            # Keep the semantic interface fixed across runs, including atoms
            # that are currently all-unknown or absent in one dataset.
            atoms.update(CANONICAL_ATOMS)
        ordered_phases = [phase for phase in DEFAULT_GAIT_PHASES if phase in phases]
        ordered_phases += sorted(phases.difference(ordered_phases))
        return cls(
            profile=profile,
            sem_keys=sorted(sem_keys),
            quality_keys=sorted(quality_keys),
            atom_vocab=sorted(atoms),
            quality_tag_vocab=list(QUALITY_V2_AXES),
            gait_vocab=ordered_phases,
            text_dim=text_dim,
            task_context=bool(ablation_config(profile)["task_context"]),
            quality_label_version=QUALITY_LABEL_VERSION,
        )

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SegmentFeatures:
    trial_id: str
    segment_id: str
    gait_phase: str
    support_sides: tuple[str, ...]
    start_index: int
    end_index: int
    sem_values: np.ndarray
    sem_mask: np.ndarray
    atom_target: np.ndarray
    atom_mask: np.ndarray
    atom_supervision_weight: np.ndarray
    quality_values: np.ndarray
    quality_mask: np.ndarray
    quality_tags: np.ndarray
    quality_tag_mask: np.ndarray
    task_text_vector: np.ndarray
    task_text_present: float
    human_text_vector: np.ndarray
    human_text_present: float
    robot_semantic_present: float
    human_semantic_present: float
    segment_query_vector: np.ndarray
    segment_query_present: float
    reconstruction_weight: float
    risk_score: float
    feasibility_score: float
    feasibility_tags: tuple[str, ...]
    quality_v2_states: dict[str, str]
    semantic_confidence: float
    semantic_supervision_weight: float
    # v4: human-only atom labels (not overridden by robot labels); empty mask when the profile has no
    # human semantics.  Used by the human-atom head on z_skill / z_sem.
    human_atom_target: np.ndarray | None = None
    human_atom_mask: np.ndarray | None = None
    query_aug_vectors: list | None = None


@dataclass
class TrialSequence:
    trial_id: str
    npz_path: Path
    fps: float
    span_start: int
    span_end: int
    skill_frames: np.ndarray
    quality_frames: np.ndarray
    task_targets: np.ndarray
    semantic_task_weights: np.ndarray
    phase_frames: np.ndarray
    segments: list[SegmentFeatures]
    task_name: str
    task_goal: str
    task_family: tuple[str, ...]


def segment_to_features(
    doc: Mapping[str, Any],
    segment: Mapping[str, Any],
    schema: FeatureSchema,
    trial_length: int,
    profile: str = "Z4",
    robot_atom_positive_weight: float = DEFAULT_ROBOT_ATOM_POSITIVE_WEIGHT,
    human_atom_positive_weight: float = DEFAULT_HUMAN_ATOM_POSITIVE_WEIGHT,
    human_semantic_weight: float = 1.0,
    human_semantic_input_scale: float = 1.0,
    semantic_text_scale: float = 1.0,
    llm_absent_atom_weight: float = DEFAULT_LLM_ABSENT_ATOM_WEIGHT,
    phase_absent_atom_weight: float = DEFAULT_PHASE_ABSENT_ATOM_WEIGHT,
) -> SegmentFeatures | None:
    config = ablation_config(profile)
    alignment = nested_get(
        segment, ("Retargeted_Robot_Metrics", "npz_alignment"), {}
    ) or {}
    start = int(as_float(alignment.get("start_index"), -1))
    end = int(as_float(alignment.get("end_index"), -1))
    start = max(0, min(start, trial_length))
    end = max(0, min(end, trial_length))
    if end <= start:
        return None

    sem_values, sem_mask = vector_from_map(
        semantic_numeric_map(doc, segment, profile), schema.sem_keys
    )
    quality_values, quality_mask = vector_from_map(
        quality_numeric_map(segment, profile), schema.quality_keys
    )
    atom_states = semantic_atom_states(segment, doc, profile)
    human_state = atom_states.get("human", {})
    robot_state = atom_states.get("robot", {})
    human_atoms = set(human_state.get("positive", set()))
    robot_atoms = set(robot_state.get("positive", set()))
    human_absent = set(human_state.get("absent", set()))
    robot_absent = set(robot_state.get("absent", set()))
    # Robot-side labels are closer to the eventual deployment domain.  In a
    # cross-source conflict, prefer robot positives/negatives and use human
    # labels only where the robot annotation is unknown.  The old union rule
    # converted contradictory weak labels into a single positive target.
    effective_robot_atoms = set(robot_atoms)
    effective_human_atoms = human_atoms.difference(robot_absent)
    effective_robot_absent = set(robot_absent)
    effective_human_absent = human_absent.difference(robot_atoms)
    atom_target = np.zeros(len(schema.atom_vocab), dtype=np.float32)
    atom_mask = np.zeros(len(schema.atom_vocab), dtype=np.float32)
    atom_supervision_weight = np.zeros(len(schema.atom_vocab), dtype=np.float32)
    for index, atom in enumerate(schema.atom_vocab):
        # Robot semantic atoms are explicit labels. Human atoms are treated
        # as weaker supplementary evidence because they may describe a
        # slightly different abstraction of the same segment.
        if atom in effective_human_atoms:
            atom_target[index] = max(atom_target[index], 0.8)
            atom_supervision_weight[index] = max(
                atom_supervision_weight[index],
                float(human_atom_positive_weight)
                * float(np.clip(human_semantic_weight, 0.0, 1.0)),
            )
        if atom in effective_robot_atoms:
            atom_target[index] = max(atom_target[index], 1.0)
            atom_supervision_weight[index] = max(
                atom_supervision_weight[index],
                float(robot_atom_positive_weight),
            )
        if atom in effective_human_atoms or atom in effective_robot_atoms:
            atom_mask[index] = 1.0

    human_semantics = (
        nested_get(segment, ("Human_Metrics", "human_semantics"), {})
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
    robot_present = bool(robot_state.get("present", False))
    human_unknown = set(human_state.get("unknown", set()))
    robot_unknown = set(robot_state.get("unknown", set()))
    phase_absent = phase_known_absent_atoms(segment)
    for index, atom in enumerate(schema.atom_vocab):
        if atom in effective_human_atoms or atom in effective_robot_atoms:
            continue
        # An explicit known-absent label is a supervised negative. Unknown
        # labels and atoms omitted by legacy positive-only annotations are
        # deliberately left masked out.
        explicit_absent = atom in effective_human_absent or atom in effective_robot_absent
        explicit_unknown = atom in human_unknown or atom in robot_unknown
        if explicit_absent and not explicit_unknown:
            atom_mask[index] = 1.0
            confidence = 0.0
            if atom in phase_absent:
                confidence = max(confidence, float(phase_absent_atom_weight))
            if atom in (effective_human_absent - phase_absent):
                confidence = max(
                    confidence,
                    float(llm_absent_atom_weight)
                    * float(np.clip(human_semantic_weight, 0.0, 1.0)),
                )
            if atom in (effective_robot_absent - phase_absent):
                confidence = max(confidence, float(llm_absent_atom_weight))
            atom_supervision_weight[index] = np.float32(
                confidence if confidence > 0.0 else llm_absent_atom_weight
            )

    human_atom_target = np.zeros(len(schema.atom_vocab), dtype=np.float32)
    human_atom_mask = np.zeros(len(schema.atom_vocab), dtype=np.float32)
    if config["human_semantics"]:
        for index, atom in enumerate(schema.atom_vocab):
            if atom in human_atoms:
                human_atom_target[index], human_atom_mask[index] = 1.0, 1.0
            elif atom in human_absent and atom not in human_unknown:
                human_atom_mask[index] = 1.0

    tags = feasibility_tags(segment, profile)
    tag_target, tag_mask_v2, quality_v2_states = quality_v2_targets_and_mask(
        segment, profile
    )
    quality_tag_mask = tag_mask_v2
    task_text = collect_task_query_text(doc, profile)
    segment_query_text = collect_segment_query_text(doc, segment, profile)
    # Keep the deployment/query branches profile-independent. Only the Z4
    # training input attenuates the sparse human-text component.
    task_vector = hashed_text_vector(task_text, schema.text_dim)
    human_text = collect_human_semantic_text(segment, profile)
    human_vector = hashed_text_vector(human_text, schema.text_dim)
    human_scale = float(np.clip(human_semantic_input_scale, 0.0, 1.0))
    text_scale = float(np.clip(semantic_text_scale, 0.0, 1.0))
    # Keep the three semantic sources genuinely separated: task text is the
    # deployable command signal, while human text is evidence only.  The
    # attenuation must happen in the human branch itself, not only in the
    # deployable task branch.
    human_branch_vector = (text_scale * human_scale * human_vector).astype(np.float32)
    task_text_vector = task_vector
    segment_query_vector = hashed_text_vector(segment_query_text, schema.text_dim)
    # v4 query augmentation: natural-language descriptions that the profile owns, phrased like a deployment
    # command ("task_goal: ..."), used as extra inputs for the task-only query encoder during training.
    aug_texts: list[str] = []
    if config["robot_semantics"]:
        rsem = nested_get(segment, ("Retargeted_Robot_Metrics", "robot_semantics"), {}) or {}
        ratoms = [str(a).replace("_", " ") for a in text_items(rsem.get("skill_atoms"))]
        if ratoms and task_text:
            aug_texts.append(f"{task_text} . {', '.join(ratoms)}")
    if config["human_semantics"]:
        hraw = nested_get(segment, ("Human_Metrics", "human_semantics"), {}) or {}
        if isinstance(hraw, Mapping):
            for item in list(text_items(hraw.get("short_caption_en")))[:1] + list(text_items(hraw.get("human_derived_control_hints")))[:2]:
                if item and not item.lower().startswith(("flight phase with no", "prepared phase with no", "unknown phase with no")):
                    aug_texts.append(f"task_goal: {item}")
            hatoms = [str(a).replace("_", " ") for a in text_items(hraw.get("skill_atoms"))]
            if hatoms and task_text:
                aug_texts.append(f"{task_text} . {', '.join(hatoms)}")
    query_aug_vectors = [hashed_text_vector(t, schema.text_dim) for t in aug_texts]
    support_sides_raw = (
        nested_get(
            segment,
            (
                "Retargeted_Robot_Metrics",
                "feasibility",
                "torque_screening",
                "support_chain",
                "support_sides",
            ),
            [],
        )
        if config["robot_numeric"]
        else []
    ) or []
    support_sides = tuple(str(side).upper() for side in support_sides_raw)
    risk = segment_risk_score(segment, end - start, profile)
    reconstruction_weight = float(np.clip(math.exp(-0.5 * risk), 0.10, 1.0))
    human_evidence_count = sum(
        1
        for item in (
            human_semantics.get("evidence", [])
            if isinstance(human_semantics, Mapping)
            else []
        )
        if isinstance(item, Mapping)
    )
    human_v2 = (
        isinstance(human_semantics, Mapping)
        and human_semantics.get("schema_version") == "human_semantics_v2"
    )
    human_annotation_informative = bool(
        human_v2
        and human_evidence_count > 0
        and (
            bool(human_semantics.get("skill_atoms"))
            or bool(human_semantics.get("human_derived_control_hints"))
        )
    )
    semantic_source_present = bool(
        np.any(sem_mask)
        or human_annotation_informative
        or robot_present
        or np.any(quality_tag_mask)
        or task_text
        or human_text
    )
    semantic_confidence = float(
        np.clip(
            0.30
            + 0.20 * bool(
                isinstance(robot_semantics, Mapping)
                and bool(robot_semantics)
            )
            + 0.15 * bool(np.any(sem_mask))
            + 0.15 * human_annotation_informative
            + 0.10 * bool(task_text or human_text),
            0.0,
            1.0,
        )
        if semantic_source_present
        else 0.0
    )
    return SegmentFeatures(
        trial_id=str(doc.get("trial_id", "unknown")),
        segment_id=str(segment.get("segment_id", "unknown")),
        gait_phase=get_phase_label(segment),
        support_sides=support_sides,
        start_index=start,
        end_index=end,
        sem_values=sem_values,
        sem_mask=sem_mask,
        atom_target=atom_target,
        atom_mask=atom_mask,
        atom_supervision_weight=atom_supervision_weight,
        human_atom_target=human_atom_target,
        human_atom_mask=human_atom_mask,
        query_aug_vectors=query_aug_vectors,
        quality_values=quality_values,
        quality_mask=quality_mask,
        quality_tags=tag_target,
        quality_tag_mask=quality_tag_mask,
        task_text_vector=task_text_vector,
        task_text_present=float(bool(task_text)),
        human_text_vector=human_branch_vector,
        human_text_present=float(bool(human_text)),
        # Availability is source-specific. A valid robot metric must not be
        # inferred from unrelated human/trial fields.
        robot_semantic_present=float(bool(
            any(
                key.startswith("robot.") and bool(sem_mask[index])
                for index, key in enumerate(schema.sem_keys)
            )
            or robot_present
        )),
        human_semantic_present=float(bool(
            any(
                key.startswith(("human.", "trial.", "segment."))
                and bool(sem_mask[index])
                for index, key in enumerate(schema.sem_keys)
            )
            or human_text
        )),
        segment_query_vector=segment_query_vector,
        segment_query_present=float(bool(segment_query_text)),
        reconstruction_weight=reconstruction_weight,
        risk_score=float(risk),
        feasibility_score=float(math.exp(-risk)),
        feasibility_tags=tuple(sorted(tags)),
        quality_v2_states=quality_v2_states,
        semantic_confidence=semantic_confidence,
        semantic_supervision_weight=float(semantic_source_present),
    )


def quat_wxyz_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized free-joint quaternion to the first 2 matrix columns."""
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)
    w, x, y, z = (q[:, index] for index in range(4))
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    # Column-major 6D rotation representation: col_0 followed by col_1.
    return np.stack((r00, r10, r20, r01, r11, r21), axis=1).astype(np.float32)


def build_phase_frames(
    length: int,
    span_start: int,
    span_end: int,
    segments: Sequence[SegmentFeatures],
    gait_vocab: Sequence[str],
) -> np.ndarray:
    # gait one-hot, global sin/cos progress, segment-local progress, L/R hint.
    # Local progress is important for ThreeStepPoleRun/FiveStepPoleRun, where "flight" appears
    # repeatedly within one trial.
    phase_dim = len(gait_vocab) + 5
    result = np.zeros((length, phase_dim), dtype=np.float32)
    span = max(span_end - span_start, 1)
    positions = np.arange(length, dtype=np.float32)
    progress = np.clip((positions - span_start) / span, 0.0, 1.0)
    result[:, len(gait_vocab)] = np.sin(2 * np.pi * progress)
    result[:, len(gait_vocab) + 1] = np.cos(2 * np.pi * progress)
    for segment in segments:
        left, right = support_sides_to_contact(
            segment.support_sides, segment.gait_phase
        )
        sl = slice(segment.start_index, segment.end_index)
        result[sl, : len(gait_vocab)] = one_hot(segment.gait_phase, gait_vocab)
        result[sl, len(gait_vocab) + 2] = left
        result[sl, len(gait_vocab) + 3] = right
        local_span = max(segment.end_index - segment.start_index - 1, 1)
        local_progress = np.clip(
            (positions[sl] - segment.start_index) / local_span, 0.0, 1.0
        )
        result[sl, len(gait_vocab) + 4] = local_progress
    return result


def checked_array(npz: Mapping[str, np.ndarray], key: str, length: int | None = None) -> np.ndarray:
    if key not in npz:
        raise KeyError(f"NPZ is missing required array: {key}")
    value = np.asarray(npz[key])
    if length is not None and len(value) != length:
        raise ValueError(f"{key} length {len(value)} != qpos length {length}")
    return value


def build_npz_frame_features(
    npz: Mapping[str, np.ndarray],
    phase_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build motion targets and deployment-observable encoder inputs.

    Quality *labels* may use simulator inverse dynamics, penetration and
    collision checks from JSON.  Quality encoder inputs must not: those values
    are unavailable before executing a candidate on the real robot and would
    make offline quality accuracy a target-leakage artifact.  The quality
    stream is therefore limited to proprioception, measured/commanded actuator
    effort, contact and phase signals that can be reproduced online.
    """
    qpos = checked_array(npz, "qpos").astype(np.float32)
    length = len(qpos)
    qvel = checked_array(npz, "qvel", length).astype(np.float32)
    tau = checked_array(npz, "actuator_tau", length).astype(np.float32)
    limits = np.asarray(npz["actuator_torque_limits"], dtype=np.float32).reshape(-1)
    joint_names = np.asarray(npz["actuator_joint_names"]).reshape(-1)
    nu = len(joint_names)
    if qpos.shape[1] < 7 + nu or qvel.shape[1] < 6 + nu:
        raise ValueError(
            f"Expected free joint + {nu} actuated joints, got qpos={qpos.shape}, qvel={qvel.shape}"
        )
    if tau.shape[1] != nu or len(limits) != nu:
        raise ValueError("actuator_tau / torque_limits / joint_names dimensions disagree")

    rotation_6d = quat_wxyz_to_rotation_6d(qpos[:, 3:7])
    root_height = qpos[:, 2:3]  # Robot_coordinate_system says vertical is +Z.
    joint_pos = qpos[:, 7 : 7 + nu]
    root_vel = qvel[:, :6]
    joint_vel = qvel[:, 6 : 6 + nu]
    floor_count = np.asarray(npz.get("floor_contact_count", np.zeros(length)), dtype=np.float32).reshape(length, 1)
    contact_proxy = np.clip(floor_count / 2.0, 0.0, 1.0)

    # Skill encoder sees kinematics, phase and only a coarse contact proxy.
    skill_frames = np.concatenate(
        (
            root_height,
            rotation_6d,
            root_vel,
            joint_pos,
            joint_vel,
            contact_proxy,
            phase_frames,
        ),
        axis=1,
    ).astype(np.float32)

    # v2: the inverse-dynamics torque ratio used to be a quality *input* although the
    # torque quality *label* is derived from the same torques -> target leakage, and the value is not
    # measurable before execution.  It is removed; the deployable executability estimate is learned
    # separately from simulated execution rollouts (pipeline_v2/stage5).
    del tau, limits
    quality_frames = np.concatenate(
        (
            root_height,
            rotation_6d,
            root_vel,
            joint_pos,
            joint_vel,
            contact_proxy,
            phase_frames,
        ),
        axis=1,
    ).astype(np.float32)

    # Bootstrap target.  The semantic/metric heads keep this robot-specific
    # reconstruction from becoming the only structure in z_skill.
    task_targets = np.concatenate(
        (root_height, rotation_6d, root_vel, joint_pos, contact_proxy), axis=1
    ).astype(np.float32)
    # Segment-level semantics can constrain task/root/contact behavior but do
    # not uniquely determine a 29-DoF joint trajectory.  The shared semantic
    # decoder therefore receives lower supervision on exact joint positions.
    semantic_task_weights = np.concatenate(
        (
            np.ones(1, dtype=np.float32),       # root height
            np.full(6, 0.5, dtype=np.float32), # root orientation
            np.ones(6, dtype=np.float32),       # root linear/angular velocity
            np.full(nu, 0.1, dtype=np.float32),# exact joint positions
            np.ones(1, dtype=np.float32),       # contact proxy
        )
    )
    return skill_frames, quality_frames, task_targets, semantic_task_weights


def build_trial_sequence(
    doc: Mapping[str, Any],
    npz_path: Path,
    schema: FeatureSchema,
    profile: str = "Z4",
    *,
    robot_atom_positive_weight: float = DEFAULT_ROBOT_ATOM_POSITIVE_WEIGHT,
    human_atom_positive_weight: float = DEFAULT_HUMAN_ATOM_POSITIVE_WEIGHT,
    human_semantic_weight: float = 1.0,
    human_semantic_input_scale: float = 1.0,
    semantic_text_scale: float = 1.0,
    llm_absent_atom_weight: float = DEFAULT_LLM_ABSENT_ATOM_WEIGHT,
    phase_absent_atom_weight: float = DEFAULT_PHASE_ABSENT_ATOM_WEIGHT,
) -> TrialSequence:
    with np.load(npz_path, allow_pickle=False) as npz:
        qpos = checked_array(npz, "qpos")
        length = len(qpos)
        segments = [
            item
            for segment in doc.get("segments", [])
            if (
                item := segment_to_features(
                    doc,
                    segment,
                    schema,
                    length,
                    profile,
                    robot_atom_positive_weight=robot_atom_positive_weight,
                    human_atom_positive_weight=human_atom_positive_weight,
                    human_semantic_weight=human_semantic_weight,
                    human_semantic_input_scale=human_semantic_input_scale,
                    semantic_text_scale=semantic_text_scale,
                    llm_absent_atom_weight=llm_absent_atom_weight,
                    phase_absent_atom_weight=phase_absent_atom_weight,
                )
            ) is not None
        ]
        if not segments:
            raise ValueError(f"No aligned segments in {doc.get('trial_id')} / {npz_path}")
        segments.sort(key=lambda item: (item.start_index, item.end_index))
        span_start = min(segment.start_index for segment in segments)
        span_end = max(segment.end_index for segment in segments)
        phase_frames = build_phase_frames(
            length, span_start, span_end, segments, schema.gait_vocab
        )
        # Z0 uses JSON only to select the common trial/segment windows.  Phase,
        # support-contact and segment-progress features are annotation-derived
        # conditions, so zero them before they reach either encoder/decoder.
        model_phase_frames = (
            np.zeros_like(phase_frames)
            if profile == "Z0"
            else phase_frames
        )
        (
            skill_frames,
            quality_frames,
            task_targets,
            semantic_task_weights,
        ) = build_npz_frame_features(npz, model_phase_frames)
        fps = as_float(npz.get("fps", 120.0), 120.0)
    task = mapping_or_empty(doc.get("task", {}))
    task_name = " . ".join(text_items(task.get("task_name_en")))
    task_goal = " . ".join(text_items(task.get("task_goal")))
    task_family = tuple(text_items(task.get("task_family")))
    return TrialSequence(
        trial_id=str(doc.get("trial_id")),
        npz_path=npz_path,
        fps=fps,
        span_start=span_start,
        span_end=span_end,
        skill_frames=skill_frames,
        quality_frames=quality_frames,
        task_targets=task_targets,
        semantic_task_weights=semantic_task_weights,
        phase_frames=model_phase_frames,
        segments=segments,
        task_name=task_name,
        task_goal=task_goal,
        task_family=task_family,
    )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def count_summary_invariant_violations(documents: Mapping[str, Mapping[str, Any]]) -> int:
    count = 0

    def walk(value: Any) -> None:
        nonlocal count
        if isinstance(value, Mapping):
            if "min" in value and "max" in value:
                lo = value.get("min")
                hi = value.get("max")
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    for key in ("start", "end", "mean"):
                        item = value.get(key)
                        if isinstance(item, (int, float)) and not (lo - 1e-6 <= item <= hi + 1e-6):
                            count += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for doc in documents.values():
        walk(doc)
    return count


def audit_dataset(
    documents: Mapping[str, Mapping[str, Any]],
    npz_mapping: Mapping[str, Path],
    profile: str = "Z4",
) -> None:
    segments = [segment for doc in documents.values() for segment in doc.get("segments", [])]
    atom_counts: collections.Counter[str] = collections.Counter()
    atom_unknown_counts: collections.Counter[str] = collections.Counter()
    atom_absent_counts: collections.Counter[str] = collections.Counter()
    tag_counts: collections.Counter[str] = collections.Counter()
    whole_violations = support_violations = joint_violations = self_collisions = 0
    for doc in documents.values():
        for segment in doc.get("segments", []):
            atom_states = semantic_atom_states(segment, doc, profile)
            for state in atom_states.values():
                atom_counts.update(state.get("positive", set()))
                atom_unknown_counts.update(state.get("unknown", set()))
                atom_absent_counts.update(state.get("absent", set()))
            tag_counts.update(feasibility_tags(segment, profile))
            feas = nested_get(segment, ("Retargeted_Robot_Metrics", "feasibility"), {}) or {}
            if nested_get(feas, ("torque_screening", "whole_body", "any_limit_violation"), False):
                whole_violations += 1
            if nested_get(feas, ("torque_screening", "support_chain", "any_limit_violation"), False):
                support_violations += 1
            if nested_get(feas, ("state_validity", "joint_limit", "status")) == "joint_limit_violation_detected":
                joint_violations += 1
            if nested_get(feas, ("state_validity", "collision", "status")) == "self_collision_detected":
                self_collisions += 1

    print(f"=== {profile} audit ===")
    print(f"trials={len(documents)}, segments={len(segments)}")
    print(f"matched_npz={len(npz_mapping)}, missing_npz={len(documents) - len(npz_mapping)}")
    print(f"summary endpoint/min/max invariant violations={count_summary_invariant_violations(documents)}")
    print(
        "feasibility: "
        f"whole_torque_violation={whole_violations}/{len(segments)}, "
        f"support_torque_violation={support_violations}/{len(segments)}, "
        f"joint_limit_violation={joint_violations}/{len(segments)}, "
        f"self_collision={self_collisions}/{len(segments)}"
    )
    print(f"canonical_skill_atoms={sorted(atom_counts)}")
    print(f"unknown_skill_atoms={dict(sorted(atom_unknown_counts.items()))}")
    print(f"known_absent_skill_atoms={dict(sorted(atom_absent_counts.items()))}")
    print(f"feasibility_tags={sorted(tag_counts)}")

    # Show which supervision sources are actually populated per dataset.
    # Empty human_semantics fields are valid, but reduce semantic supervision.
    coverage: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    for trial_id, doc in documents.items():
        dataset_id = str(doc.get("_dataset_id") or trial_id.split("_", 1)[0])
        item = coverage[dataset_id]
        item["json_trials"] += 1
        item["matched_npz"] += int(trial_id in npz_mapping)
        for segment in doc.get("segments", []):
            item["segments"] += 1
            alignment = nested_get(
                segment, ("Retargeted_Robot_Metrics", "npz_alignment"), {}
            ) or {}
            item["aligned_segments"] += int(
                as_float(alignment.get("end_index"), -1)
                > as_float(alignment.get("start_index"), -1)
            )
            human = mapping_or_empty(segment.get("Human_Metrics"))
            robot = mapping_or_empty(segment.get("Retargeted_Robot_Metrics"))
            human_numeric = any(
                bool(flatten_numeric(human.get(namespace, {})))
                for namespace in (
                    "step_info",
                    "obstacle_relation",
                    "kinematics",
                    "dynamics",
                )
            )
            robot_numeric = any(
                bool(
                    flatten_numeric(
                        robot.get(namespace, {}),
                        include_bool=True,
                    )
                )
                for namespace in (
                    "kinematics",
                    "obstacle_relation",
                    "dynamics",
                    "feasibility",
                )
            )
            item["segments_with_human_numeric"] += int(human_numeric)
            item["segments_with_robot_numeric"] += int(robot_numeric)
            item["segments_with_human_llm_semantics"] += int(
                bool(
                    text_items(
                        nested_get(
                            segment,
                            ("Human_Metrics", "human_semantics"),
                            {},
                        )
                    )
                )
            )
            item["segments_with_robot_llm_semantics"] += int(
                bool(
                    text_items(
                        nested_get(
                            segment,
                            ("Retargeted_Robot_Metrics", "robot_semantics"),
                            {},
                        )
                    )
                )
            )
            robot_semantics = robot.get("robot_semantics", {})
            if isinstance(robot_semantics, Mapping):
                item["robot_atom_positive_labels"] += len(
                    {
                        canonical_atom(value)
                        for value in text_items(robot_semantics.get("skill_atoms"))
                        if canonical_atom(value) in ACTIVE_ATOM_SET
                    }
                )
                item["robot_atom_unknown_labels"] += len(
                    {
                        canonical_atom(value)
                        for value in text_items(robot_semantics.get("unknown_atoms"))
                        if canonical_atom(value) in ACTIVE_ATOM_SET
                    }
                )
                item["robot_atom_known_absent_labels"] += len(
                    {
                        canonical_atom(value)
                        for value in text_items(
                            robot_semantics.get("known_absent_atoms")
                        )
                        if canonical_atom(value) in ACTIVE_ATOM_SET
                    }
                )

    print("\n=== Supervision coverage ===")
    for dataset_id, item in sorted(coverage.items()):
        print(
            f"{dataset_id}: "
            f"json_trials={item['json_trials']}, "
            f"matched_npz={item['matched_npz']}, "
            f"segments={item['segments']}, "
            f"aligned={item['aligned_segments']}, "
            f"human_numeric={item['segments_with_human_numeric']}, "
            f"robot_numeric={item['segments_with_robot_numeric']}, "
            f"human_llm_semantics={item['segments_with_human_llm_semantics']}, "
            f"robot_llm_semantics={item['segments_with_robot_llm_semantics']}, "
            f"robot_atoms(+/?/-)="
            f"{item['robot_atom_positive_labels']}/"
            f"{item['robot_atom_unknown_labels']}/"
            f"{item['robot_atom_known_absent_labels']}"
        )

    print("\n=== NPZ audit ===")
    for trial_id, path in sorted(npz_mapping.items()):
        with np.load(path, allow_pickle=False) as npz:
            qpos = checked_array(npz, "qpos")
            qvel = checked_array(npz, "qvel", len(qpos))
            qacc = checked_array(npz, "qacc", len(qpos))
            tau = checked_array(npz, "actuator_tau", len(qpos))
            qfrc = checked_array(npz, "qfrc_inverse", len(qpos))
            limits = np.asarray(npz["actuator_torque_limits"])
            util = np.abs(tau) / np.maximum(limits[None, :], 1e-6)
            duplicated = qfrc.shape[1] >= 6 + tau.shape[1] and np.allclose(
                qfrc[:, 6 : 6 + tau.shape[1]], tau
            )
            grf = np.asarray(npz.get("scaled_robot_grf", np.empty((0, 3))))
            print(
                f"{trial_id}: frames={len(qpos)}, fps={as_float(npz.get('fps', 0)):.3f}, "
                f"qpos={qpos.shape}, qvel={qvel.shape}, qacc={qacc.shape}, "
                f"tau_util_max={float(util.max()):.3f}, "
                f"tau_violation_frame_ratio={float(np.mean(np.any(util > 1, axis=1))):.3f}, "
                f"qfrc_joint_equals_tau={duplicated}, robot_grf_shape={grf.shape}"
            )


# ---------------------------------------------------------------------------
# Normalization and window dataset
# ---------------------------------------------------------------------------


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    z_clip: float = 8.0

    @classmethod
    def fit(
        cls,
        arrays: Sequence[np.ndarray],
        *,
        clip_percentile: float = 1.0,
        z_clip: float = 8.0,
    ) -> "Standardizer":
        if not arrays:
            raise ValueError("Cannot fit Standardizer on no arrays")
        values = np.concatenate(arrays, axis=0).astype(np.float64)
        percentile = float(np.clip(clip_percentile, 0.0, 49.0))
        finite_values = np.where(np.isfinite(values), values, np.nan)
        lower = np.nanpercentile(finite_values, percentile, axis=0)
        upper = np.nanpercentile(finite_values, 100.0 - percentile, axis=0)
        lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.maximum(upper, lower)
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, lower, upper)
        mean = np.mean(clean, axis=0)
        std = np.std(clean, axis=0)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std[std < 1e-6] = 1.0
        return cls(
            mean.astype(np.float32),
            std.astype(np.float32),
            lower.astype(np.float32),
            upper.astype(np.float32),
            float(max(z_clip, 0.0)),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, self.lower, self.upper)
        normalized = (clean - self.mean) / self.std
        if self.z_clip > 0.0:
            normalized = np.clip(normalized, -self.z_clip, self.z_clip)
        return normalized.astype(np.float32)

    def to_jsonable(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "z_clip": self.z_clip,
        }


@dataclass
class MaskedStandardizer:
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    z_clip: float = 8.0

    @classmethod
    def fit(
        cls,
        values: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        *,
        clip_percentile: float = 1.0,
        z_clip: float = 8.0,
    ) -> "MaskedStandardizer":
        value_matrix = np.stack(values).astype(np.float64)
        mask_matrix = np.stack(masks).astype(np.float64)
        count = mask_matrix.sum(axis=0)
        percentile = float(np.clip(clip_percentile, 0.0, 49.0))
        lower = np.zeros(value_matrix.shape[1], dtype=np.float64)
        upper = np.zeros(value_matrix.shape[1], dtype=np.float64)
        for index in range(value_matrix.shape[1]):
            valid = value_matrix[:, index][
                (mask_matrix[:, index] > 0.0)
                & np.isfinite(value_matrix[:, index])
            ]
            if len(valid):
                lower[index] = np.percentile(valid, percentile)
                upper[index] = np.percentile(valid, 100.0 - percentile)
                if upper[index] < lower[index]:
                    upper[index] = lower[index]
            else:
                lower[index] = 0.0
                upper[index] = 0.0
        clean = np.nan_to_num(value_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, lower, upper)
        clean *= mask_matrix
        mean = clean.sum(axis=0) / np.maximum(count, 1.0)
        variance = (
            ((clean - mean) ** 2) * mask_matrix
        ).sum(axis=0) / np.maximum(count, 1.0)
        std = np.sqrt(variance)
        mean[count == 0] = 0.0
        std[(count == 0) | (std < 1e-6)] = 1.0
        return cls(
            mean.astype(np.float32),
            std.astype(np.float32),
            lower.astype(np.float32),
            upper.astype(np.float32),
            float(max(z_clip, 0.0)),
        )

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, self.lower, self.upper)
        normalized = (clean - self.mean) / self.std
        if self.z_clip > 0.0:
            normalized = np.clip(normalized, -self.z_clip, self.z_clip)
        return (normalized * mask).astype(np.float32)

    def to_jsonable(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "z_clip": self.z_clip,
        }


@dataclass
class Normalizers:
    skill: Standardizer
    quality_frame: Standardizer
    task: Standardizer
    semantic: MaskedStandardizer
    quality_target: MaskedStandardizer

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "skill": self.skill.to_jsonable(),
            "quality_frame": self.quality_frame.to_jsonable(),
            "task": self.task.to_jsonable(),
            "semantic": self.semantic.to_jsonable(),
            "quality_target": self.quality_target.to_jsonable(),
        }


def fit_normalizers(
    train_sequences: Sequence[TrialSequence],
    *,
    clip_percentile: float = 1.0,
    z_clip: float = 8.0,
) -> Normalizers:
    if not train_sequences:
        raise ValueError("At least one training trial is required")
    skill_arrays = [seq.skill_frames[seq.span_start : seq.span_end] for seq in train_sequences]
    quality_arrays = [seq.quality_frames[seq.span_start : seq.span_end] for seq in train_sequences]
    task_arrays = [seq.task_targets[seq.span_start : seq.span_end] for seq in train_sequences]
    segments = [segment for seq in train_sequences for segment in seq.segments]
    return Normalizers(
        skill=Standardizer.fit(
            skill_arrays,
            clip_percentile=clip_percentile,
            z_clip=z_clip,
        ),
        quality_frame=Standardizer.fit(
            quality_arrays,
            clip_percentile=clip_percentile,
            z_clip=z_clip,
        ),
        task=Standardizer.fit(
            task_arrays,
            clip_percentile=clip_percentile,
            z_clip=z_clip,
        ),
        semantic=MaskedStandardizer.fit(
            [segment.sem_values for segment in segments],
            [segment.sem_mask for segment in segments],
            clip_percentile=clip_percentile,
            z_clip=z_clip,
        ),
        quality_target=MaskedStandardizer.fit(
            [segment.quality_values for segment in segments],
            [segment.quality_mask for segment in segments],
            clip_percentile=clip_percentile,
            z_clip=z_clip,
        ),
    )


def quality_tag_positive_weights(
    train_sequences: Sequence[TrialSequence],
    tag_count: int,
) -> np.ndarray:
    """Balance rare feasibility tags using only the training split."""
    if tag_count == 0:
        return np.ones(0, dtype=np.float32)
    segments = [
        segment
        for sequence in train_sequences
        for segment in sequence.segments
    ]
    targets = np.stack([segment.quality_tags for segment in segments], axis=0)
    masks = np.stack([segment.quality_tag_mask for segment in segments], axis=0)
    known = masks > 0.0
    positives = ((targets > 0.5) & known).sum(axis=0).astype(np.float32)
    negatives = ((targets <= 0.5) & known).sum(axis=0).astype(np.float32)
    weights = negatives / np.maximum(positives, 1.0)
    # Balanced focal loss uses this as the positive-class multiplier.
    return np.clip(weights, 1.0, 8.0).astype(np.float32)


def atom_positive_weights(
    train_sequences: Sequence[TrialSequence],
    atom_count: int,
    max_weight: float = 6.0,
    single_group_max_weight: float | None = None,
) -> np.ndarray:
    """Balance rare semantic atoms using only the training split.

    Atom targets are soft because human and robot annotations have different
    confidence.  Presence is therefore counted as ``target > 0`` and the
    resulting weight is applied only to the positive BCE term.
    """
    if atom_count == 0:
        return np.ones(0, dtype=np.float32)
    segments = [
        segment
        for sequence in train_sequences
        for segment in sequence.segments
    ]
    targets = np.stack([segment.atom_target for segment in segments], axis=0)
    masks = np.stack([segment.atom_mask for segment in segments], axis=0)
    supervision = np.stack(
        [segment.atom_supervision_weight for segment in segments], axis=0
    )
    targets = targets * masks
    weighted_valid = masks * supervision
    present = ((targets > 0.0) * weighted_valid).sum(axis=0)
    absent = weighted_valid.sum(axis=0) - present
    # Plain inverse-frequency weighting is too aggressive for this dataset:
    # a handful of noisy/ambiguous annotations can dominate the atom head.
    # Square-root inverse frequency still helps rare atoms while preserving
    # useful gradients from common atoms.
    weights = np.sqrt(absent / np.maximum(present, 1.0))
    weights = np.clip(weights, 1.0, max(float(max_weight), 1.0))

    # An atom observed in only one dataset-local subject group cannot be
    # validated as a cross-subject concept.  Letting its inverse-frequency
    # weight grow too large encourages memorizing that group's motion style.
    if single_group_max_weight is not None:
        group_sets = [set() for _ in range(atom_count)]
        for sequence in train_sequences:
            group = subject_group_key(sequence.trial_id)
            for segment in sequence.segments:
                present_atoms = (
                    (segment.atom_target > 0.0) & (segment.atom_mask > 0.0)
                )
                for index in np.flatnonzero(present_atoms):
                    group_sets[int(index)].add(group)
        limited = max(float(single_group_max_weight), 1.0)
        for index, groups in enumerate(group_sets):
            if len(groups) <= 1:
                weights[index] = min(weights[index], limited)
    return weights.astype(np.float32)


@dataclass(frozen=True)
class WindowSpec:
    trial_index: int
    start_index: int
    center_index: int
    segment_index: int
    semantic_purity: float
    semantic_window_weight: float


def nearest_segment_index(segments: Sequence[SegmentFeatures], frame: int) -> int:
    for index, segment in enumerate(segments):
        if segment.start_index <= frame < segment.end_index:
            return index
    distances = [
        min(abs(frame - segment.start_index), abs(frame - (segment.end_index - 1)))
        for segment in segments
    ]
    return int(np.argmin(distances))


def window_semantic_assignment(
    sequence: TrialSequence,
    start: int,
    length: int,
    *,
    purity_threshold: float = 0.55,
    minimum_weight: float = 0.15,
    forced_segment_index: int | None = None,
) -> tuple[int, float, float]:
    """Assign the dominant segment and down-weight mixed semantic windows.

    Motion reconstruction still uses the whole window.  Segment-level
    annotations, however, are only reliable when one segment covers most of
    the window.  The returned weight is kept above a small floor so mixed
    windows remain useful without allowing one segment label to dominate them.
    """
    if not sequence.segments:
        raise ValueError(f"Trial {sequence.trial_id} has no semantic segments")
    end = start + length
    valid_start = max(0, start, sequence.span_start)
    valid_end = min(end, len(sequence.skill_frames), sequence.span_end)
    valid_length = max(valid_end - valid_start, 1)
    overlaps = np.asarray(
        [
            max(
                0,
                min(valid_end, segment.end_index)
                - max(valid_start, segment.start_index),
            )
            for segment in sequence.segments
        ],
        dtype=np.float32,
    )
    if forced_segment_index is not None:
        if not 0 <= int(forced_segment_index) < len(sequence.segments):
            raise IndexError("forced segment index is outside the trial segment list")
        segment_index = int(forced_segment_index)
        purity = float(overlaps[segment_index]) / float(valid_length)
    elif float(overlaps.max(initial=0.0)) <= 0.0:
        segment_index = nearest_segment_index(
            sequence.segments,
            start + length // 2,
        )
        purity = 0.0
    else:
        max_overlap = float(overlaps.max())
        candidates = np.flatnonzero(
            np.isclose(overlaps, max_overlap, rtol=0.0, atol=1e-6)
        )
        center = start + length // 2
        segment_index = min(
            (int(item) for item in candidates),
            key=lambda item: (
                0
                if sequence.segments[item].start_index
                <= center
                < sequence.segments[item].end_index
                else 1,
                abs(
                    center
                    - (
                        sequence.segments[item].start_index
                        + sequence.segments[item].end_index
                    )
                    // 2
                ),
            ),
        )
        purity = max_overlap / float(valid_length)

    threshold = max(float(purity_threshold), 1e-6)
    floor = float(np.clip(minimum_weight, 0.0, 1.0))
    weight = floor + (1.0 - floor) * min(max(purity, 0.0) / threshold, 1.0)
    return int(segment_index), float(np.clip(purity, 0.0, 1.0)), float(weight)


def build_window_specs(
    sequences: Sequence[TrialSequence],
    window: int,
    stride: int,
    max_windows: int | None = None,
    *,
    semantic_purity_threshold: float = 0.55,
    semantic_window_min_weight: float = 0.15,
    force_segment_centers: bool = False,
) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for trial_index, sequence in enumerate(sequences):
        span = sequence.span_end - sequence.span_start
        forced_centers = [
            ((segment.start_index + segment.end_index - 1) // 2, segment_index)
            for segment_index, segment in enumerate(sequence.segments)
        ]
        segment_center_values = {center for center, _ in forced_centers}
        regular_centers: set[int] = set()
        if span <= window:
            regular_centers.add((sequence.span_start + sequence.span_end - 1) // 2)
        else:
            regular = range(
                sequence.span_start + window // 2,
                sequence.span_end - (window - window // 2) + 1,
                stride,
            )
            regular_centers.update(regular)

        if force_segment_centers:
            # Every annotated subskill receives at least one representation,
            # even when it is shorter than the temporal context window.  A
            # phase-focused pooling mask below prevents the neighboring phase
            # frames from dominating its z_skill code.
            center_specs: list[tuple[int, int | None]] = list(forced_centers)
            center_specs.extend(
                (center, None)
                for center in regular_centers
                if center not in segment_center_values
            )
        else:
            center_specs = [
                (center, None)
                for center in segment_center_values.union(regular_centers)
            ]

        for center, forced_segment_index in sorted(
            center_specs,
            key=lambda item: (item[0], -1 if item[1] is None else item[1]),
        ):
            start = center - window // 2
            if span >= window:
                start = max(sequence.span_start, min(start, sequence.span_end - window))
            else:
                start = max(0, min(start, len(sequence.skill_frames) - window))
            (
                segment_index,
                semantic_purity,
                semantic_window_weight,
            ) = window_semantic_assignment(
                sequence,
                start,
                window,
                purity_threshold=semantic_purity_threshold,
                minimum_weight=semantic_window_min_weight,
                forced_segment_index=forced_segment_index,
            )
            specs.append(
                WindowSpec(
                    trial_index,
                    start,
                    center,
                    segment_index,
                    semantic_purity,
                    semantic_window_weight,
                )
            )
    if max_windows is not None and max_windows > 0 and len(specs) > max_windows:
        # Deterministic subsampling, preserving temporal coverage.
        indices = np.linspace(0, len(specs) - 1, max_windows).round().astype(int)
        specs = [specs[index] for index in indices]
    return specs


def slice_with_edge_padding(array: np.ndarray, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    end = start + length
    valid_start = max(start, 0)
    valid_end = min(end, len(array))
    piece = array[valid_start:valid_end]
    left = valid_start - start
    right = end - valid_end
    if left or right:
        piece = np.pad(piece, ((left, right), (0, 0)), mode="edge")
    mask = np.ones(length, dtype=np.float32)
    if left:
        mask[:left] = 0.0
    if right:
        mask[length - right :] = 0.0
    return piece.astype(np.float32), mask


_CTRL_AUX_CACHE: dict[str, Any] = {}


def ctrl_aux_meta() -> dict[str, Any]:
    """v5 control-relevant targets (stage4_latent/make_ctrl_aux.py) from HNU_CTRL_AUX_DIR; empty when unset."""
    import json as _json
    import os as _os
    d = _os.environ.get("HNU_CTRL_AUX_DIR", "")
    if not d or not _os.path.exists(_os.path.join(d, "meta.json")):
        return {"future_dim": 0, "human_dim": 0, "dir": ""}
    meta = _json.load(open(_os.path.join(d, "meta.json")))
    meta["dir"] = d
    return meta


def ctrl_aux_targets(trial_id: str, center: int) -> dict[str, np.ndarray]:
    import os as _os
    meta = ctrl_aux_meta()
    fd, hd = int(meta["future_dim"]), int(meta["human_dim"])
    out = {"future_target": np.zeros(fd, np.float32), "future_mask": np.zeros(fd, np.float32),
           "human_ctrl_target": np.zeros(hd, np.float32), "human_ctrl_mask": np.zeros(hd, np.float32)}
    if not meta["dir"]:
        return out
    if trial_id not in _CTRL_AUX_CACHE:
        p = _os.path.join(meta["dir"], f"{trial_id}.npz")
        _CTRL_AUX_CACHE[trial_id] = dict(np.load(p)) if _os.path.exists(p) else None
    t = _CTRL_AUX_CACHE[trial_id]
    if t is None:
        return out
    c = int(np.clip(center, 0, len(t["future"]) - 1))
    out["future_target"], out["future_mask"] = t["future"][c], t["future_mask"][c]
    out["human_ctrl_target"], out["human_ctrl_mask"] = t["human"][c], t["human_mask"][c]
    return out


class MotionWindowDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[TrialSequence],
        schema: FeatureSchema,
        normalizers: Normalizers,
        *,
        window: int,
        stride: int,
        max_windows: int | None = None,
        quality_tag_pos_weight: np.ndarray | None = None,
        atom_pos_weight: np.ndarray | None = None,
        semantic_purity_threshold: float = 0.55,
        semantic_window_min_weight: float = 0.15,
        human_metric_scale: float = 1.0,
        force_segment_centers: bool = False,
        phase_focused_pooling: bool = False,
        skill_local_pool_frames: int = 0,
    ) -> None:
        self.sequences = list(sequences)
        self.schema = schema
        self.normalizers = normalizers
        self.window = window
        self.phase_focused_pooling = bool(phase_focused_pooling)
        self.skill_local_pool_frames = int(max(skill_local_pool_frames, 0))
        self.human_metric_scale = float(np.clip(human_metric_scale, 0.0, 1.0))
        self.quality_tag_pos_weight = (
            np.asarray(quality_tag_pos_weight, dtype=np.float32)
            if quality_tag_pos_weight is not None
            else np.ones(len(schema.quality_tag_vocab), dtype=np.float32)
        )
        self.atom_pos_weight = (
            np.asarray(atom_pos_weight, dtype=np.float32)
            if atom_pos_weight is not None
            else np.ones(len(schema.atom_vocab), dtype=np.float32)
        )
        self.specs = build_window_specs(
            self.sequences,
            window,
            stride,
            max_windows,
            semantic_purity_threshold=semantic_purity_threshold,
            semantic_window_min_weight=semantic_window_min_weight,
            force_segment_centers=force_segment_centers,
        )
        self.segment_group_ids: dict[tuple[str, str], int] = {}
        self.trial_group_ids: dict[str, int] = {}
        self.phase_class_ids: dict[tuple[str, str], int] = {}
        for spec in self.specs:
            sequence = self.sequences[spec.trial_index]
            segment = sequence.segments[spec.segment_index]
            key = (sequence.trial_id, segment.segment_id)
            if key not in self.segment_group_ids:
                self.segment_group_ids[key] = len(self.segment_group_ids)
            if sequence.trial_id not in self.trial_group_ids:
                self.trial_group_ids[sequence.trial_id] = len(self.trial_group_ids)
            phase_key = (sequence.npz_path.parent.name, segment.gait_phase)
            if phase_key not in self.phase_class_ids:
                self.phase_class_ids[phase_key] = len(self.phase_class_ids)

    def __len__(self) -> int:
        return len(self.specs)

    def metadata(self, index: int) -> dict[str, Any]:
        spec = self.specs[index]
        sequence = self.sequences[spec.trial_index]
        segment = sequence.segments[spec.segment_index]
        return {
            "trial_id": sequence.trial_id,
            "segment_id": segment.segment_id,
            "task_name": sequence.task_name,
            "task_goal": sequence.task_goal,
            "task_family": list(sequence.task_family),
            "gait_phase": segment.gait_phase,
            "start_index": spec.start_index,
            "center_index": spec.center_index,
            "semantic_purity": spec.semantic_purity,
            "semantic_window_weight": spec.semantic_window_weight,
            "risk_score": segment.risk_score,
            "feasibility_score": segment.feasibility_score,
            "feasibility_tags": list(segment.feasibility_tags),
            "quality_v2_states": dict(segment.quality_v2_states),
        }

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.float32 | np.int64]:
        spec = self.specs[index]
        sequence = self.sequences[spec.trial_index]
        segment = sequence.segments[spec.segment_index]
        skill, seq_mask = slice_with_edge_padding(
            sequence.skill_frames, spec.start_index, self.window
        )
        quality, _ = slice_with_edge_padding(
            sequence.quality_frames, spec.start_index, self.window
        )
        task, _ = slice_with_edge_padding(
            sequence.task_targets, spec.start_index, self.window
        )
        phase, _ = slice_with_edge_padding(
            sequence.phase_frames, spec.start_index, self.window
        )
        absolute_frames = np.arange(
            spec.start_index,
            spec.start_index + self.window,
            dtype=np.int64,
        )
        if self.phase_focused_pooling:
            skill_pool_mask = (
                (absolute_frames >= segment.start_index)
                & (absolute_frames < segment.end_index)
            ).astype(np.float32)
            skill_pool_mask *= seq_mask
            if float(skill_pool_mask.sum()) < 1.0:
                skill_pool_mask = seq_mask.copy()
        else:
            skill_pool_mask = seq_mask.copy()
        # A short center crop complements the phase-wide pooling mask.  Both
        # masks see the same full 96-frame TCN context; only the final pooling
        # scale differs.  This is important for FiveStepPoleRun, where the short
        # take-off/flight transitions are easily diluted by longer segments.
        skill_local_pool_mask = skill_pool_mask.copy()
        if self.skill_local_pool_frames > 0:
            local_start = spec.center_index - self.skill_local_pool_frames // 2
            local_end = local_start + self.skill_local_pool_frames
            local_mask = (
                (absolute_frames >= local_start)
                & (absolute_frames < local_end)
            ).astype(np.float32)
            # Keep neighbouring motion in the local branch: for very short
            # phases the entry/exit dynamics carry more information than the
            # few frames strictly inside the annotation boundary.
            skill_local_pool_mask = local_mask * seq_mask
            if float(skill_local_pool_mask.sum()) < 1.0:
                skill_local_pool_mask = skill_pool_mask.copy()
        sem_norm = self.normalizers.semantic.transform(segment.sem_values, segment.sem_mask)
        skill_metric_mask = segment.sem_mask.astype(np.float32).copy()
        for metric_index, metric_key in enumerate(self.schema.sem_keys):
            if metric_key in SKILL_NUISANCE_SEMANTIC_KEYS:
                skill_metric_mask[metric_index] = 0.0
        quality_norm = self.normalizers.quality_target.transform(
            segment.quality_values, segment.quality_mask
        )
        gait = one_hot(segment.gait_phase, self.schema.gait_vocab)
        semantic_branches = build_semantic_branches(
            self.schema,
            sem_norm,
            segment.sem_mask,
            gait,
            segment.task_text_vector,
            segment.task_text_present,
            segment.human_text_vector,
            segment.human_text_present,
            robot_present=segment.robot_semantic_present,
            human_present=segment.human_semantic_present,
            human_metric_scale=self.human_metric_scale,
        )
        query_text_vector = segment.task_text_vector
        p_aug = float(getattr(self, "query_augmentation", 0.0))
        if p_aug > 0 and segment.query_aug_vectors and np.random.random() < p_aug:
            query_text_vector = segment.query_aug_vectors[np.random.randint(len(segment.query_aug_vectors))]
        task_query_branches = build_semantic_branches(
            self.schema,
            np.zeros_like(sem_norm),
            np.zeros_like(segment.sem_mask),
            np.zeros_like(gait),
            query_text_vector,
            segment.task_text_present,
            np.zeros_like(segment.human_text_vector),
            0.0,
            robot_present=0.0,
            human_present=0.0,
            mask_annotation_channels=True,
        )
        segment_query_branches = build_semantic_branches(
            self.schema,
            np.zeros_like(sem_norm),
            np.zeros_like(segment.sem_mask),
            gait,
            segment.segment_query_vector,
            segment.segment_query_present,
            np.zeros_like(segment.human_text_vector),
            0.0,
            robot_present=float(bool(np.any(gait))),
            human_present=0.0,
            mask_annotation_channels=False,
        )
        segment_group_key = (sequence.trial_id, segment.segment_id)
        return {
            "skill_seq": self.normalizers.skill.transform(skill),
            "quality_seq": self.normalizers.quality_frame.transform(quality),
            "task_target": self.normalizers.task.transform(task),
            "semantic_task_weight": sequence.semantic_task_weights.astype(np.float32),
            "phase_seq": phase.astype(np.float32),
            "seq_mask": seq_mask,
            "skill_pool_mask": skill_pool_mask.astype(np.float32),
            "skill_local_pool_mask": skill_local_pool_mask.astype(np.float32),
            "task_branch": semantic_branches["task_branch"],
            "robot_branch": semantic_branches["robot_branch"],
            "human_branch": semantic_branches["human_branch"],
            "task_query_task_branch": task_query_branches["task_branch"],
            "task_query_robot_branch": task_query_branches["robot_branch"],
            "task_query_human_branch": task_query_branches["human_branch"],
            "task_query_present": np.float32(segment.task_text_present),
            "segment_query_task_branch": segment_query_branches["task_branch"],
            "segment_query_robot_branch": segment_query_branches["robot_branch"],
            "segment_query_human_branch": segment_query_branches["human_branch"],
            "segment_query_present": np.float32(segment.segment_query_present),
            "segment_group_index": np.int64(self.segment_group_ids[segment_group_key]),
            "segment_order_index": np.int64(spec.segment_index),
            "trial_group_index": np.int64(
                self.trial_group_ids[sequence.trial_id]
            ),
            "phase_class_index": np.int64(
                self.phase_class_ids[
                    (sequence.npz_path.parent.name, segment.gait_phase)
                ]
            ),
            "action_index": np.int64(
                DEFAULT_DATASETS.index(sequence.npz_path.parent.name)
                if sequence.npz_path.parent.name in DEFAULT_DATASETS else -1
            ),
            "semantic_metric_target": sem_norm,
            "semantic_metric_mask": segment.sem_mask.astype(np.float32),
            "skill_metric_mask": skill_metric_mask,
            "atom_target": segment.atom_target.astype(np.float32),
            "atom_mask": segment.atom_mask.astype(np.float32),
            "atom_supervision_weight": segment.atom_supervision_weight.astype(
                np.float32
            ),
            "human_atom_target": (segment.human_atom_target if segment.human_atom_target is not None
                                  else np.zeros_like(segment.atom_target)).astype(np.float32),
            "human_atom_mask": (segment.human_atom_mask if segment.human_atom_mask is not None
                                else np.zeros_like(segment.atom_mask)).astype(np.float32),
            "quality_target": quality_norm,
            "quality_mask": segment.quality_mask.astype(np.float32),
            "quality_tags": segment.quality_tags.astype(np.float32),
            "quality_tag_mask": segment.quality_tag_mask.astype(np.float32),
            "quality_tag_pos_weight": self.quality_tag_pos_weight,
            "atom_pos_weight": self.atom_pos_weight,
            "reconstruction_weight": np.float32(segment.reconstruction_weight),
            "semantic_confidence": np.float32(segment.semantic_confidence),
            "semantic_supervision_weight": np.float32(
                segment.semantic_supervision_weight
            ),
            "semantic_purity": np.float32(spec.semantic_purity),
            "semantic_window_weight": np.float32(spec.semantic_window_weight),
            "sample_index": np.int64(index),
            **ctrl_aux_targets(sequence.trial_id, spec.center_index),
        }


def build_atom_sample_weights(
    dataset: MotionWindowDataset,
    atom_pos_weight: np.ndarray,
    boost: float,
    max_weight: float = 3.0,
    dataset_balance_boost: float = 0.0,
    coverage_boost: float = 0.0,
    trial_balance_boost: float = 0.0,
    phase_balance_boost: float = 0.0,
    segment_balance_power: float = 0.0,
) -> np.ndarray | None:
    """Return capped per-window weights for balanced semantic sampling.

    The sampler is based on training windows only.  It combines controlled
    rare-atom oversampling with a mild inverse-sqrt action-frequency factor.
    The latter prevents the longest action family from dominating the shared
    encoder while keeping the original distribution mostly intact.  Coverage
    balancing is intentionally applied only when an atom appears in multiple
    dataset-local subject groups; a single-group atom is not oversampled into
    a false sense of cross-subject generalization.
    """
    if (
        len(dataset) == 0
        or (
            boost <= 0.0
            and dataset_balance_boost <= 0.0
            and coverage_boost <= 0.0
            and trial_balance_boost <= 0.0
            and phase_balance_boost <= 0.0
            and segment_balance_power <= 0.0
        )
    ):
        return None
    weights = np.ones(len(dataset), dtype=np.float64)
    if boost > 0.0:
        positive_weights = np.asarray(atom_pos_weight, dtype=np.float64)
        max_excess = max(float(positive_weights.max(initial=1.0) - 1.0), 1.0)
    else:
        positive_weights = np.ones(len(atom_pos_weight), dtype=np.float64)
        max_excess = 1.0

    dataset_counts: collections.Counter[str] = collections.Counter()
    trial_counts: collections.Counter[str] = collections.Counter()
    phase_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    segment_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    dataset_names: list[str] = []
    phase_names: list[tuple[str, str]] = []
    segment_names: list[tuple[str, str]] = []
    group_names: list[str] = []
    atom_group_counts: collections.Counter[tuple[int, str]] = collections.Counter()
    atom_groups: list[set[str]] = [set() for _ in range(len(atom_pos_weight))]
    for spec in dataset.specs:
        sequence = dataset.sequences[spec.trial_index]
        segment = sequence.segments[spec.segment_index]
        dataset_name = sequence.npz_path.parent.name
        dataset_names.append(dataset_name)
        dataset_counts[dataset_name] += 1
        phase_name = (dataset_name, segment.gait_phase)
        phase_names.append(phase_name)
        phase_counts[phase_name] += 1
        trial_id = sequence.trial_id
        trial_counts[trial_id] += 1
        segment_name = (trial_id, segment.segment_id)
        segment_names.append(segment_name)
        segment_counts[segment_name] += 1
        group = subject_group_key(sequence.trial_id)
        group_names.append(group)
        present = (segment.atom_target > 0.0) & (segment.atom_mask > 0.0)
        for atom_index in np.flatnonzero(present):
            atom_index = int(atom_index)
            atom_group_counts[(atom_index, group)] += 1
            atom_groups[atom_index].add(group)
    max_dataset_count = max(dataset_counts.values(), default=1)
    max_group_count_by_atom = {
        atom_index: max(
            (
                count
                for (candidate, _), count in atom_group_counts.items()
                if candidate == atom_index
            ),
            default=1,
        )
        for atom_index in range(len(atom_pos_weight))
    }

    for index, spec in enumerate(dataset.specs):
        segment = dataset.sequences[spec.trial_index].segments[spec.segment_index]
        if boost > 0.0:
            present = (segment.atom_target > 0.0) & (segment.atom_mask > 0.0)
            if present.any():
                excess = np.maximum(positive_weights[present] - 1.0, 0.0)
                # A window containing one genuinely rare atom should not be
                # diluted by several common atoms annotated in the same
                # segment. Keep a small mean contribution for stability.
                max_rarity = float(excess.max(initial=0.0) / max_excess)
                mean_rarity = float(excess.mean() / max_excess)
                rarity = 0.75 * max_rarity + 0.25 * mean_rarity
            else:
                rarity = 0.0
            weights[index] *= 1.0 + float(boost) * np.clip(rarity, 0.0, 1.0)
        if coverage_boost > 0.0:
            group = group_names[index]
            factors = []
            for atom_index in np.flatnonzero(
                (segment.atom_target > 0.0) & (segment.atom_mask > 0.0)
            ):
                atom_index = int(atom_index)
                # Coverage balancing is useful only when the atom has more
                # than one group in training data.
                if len(atom_groups[atom_index]) <= 1:
                    continue
                group_count = atom_group_counts[(atom_index, group)]
                max_group_count = max_group_count_by_atom[atom_index]
                inverse_sqrt = math.sqrt(max_group_count / max(group_count, 1))
                factors.append(np.clip(inverse_sqrt - 1.0, 0.0, 1.0))
            if factors:
                weights[index] *= 1.0 + float(coverage_boost) * float(np.mean(factors))
        if dataset_balance_boost > 0.0:
            dataset_name = dataset_names[index]
            count = max(dataset_counts[dataset_name], 1)
            inverse_sqrt = math.sqrt(max_dataset_count / count)
            weights[index] *= 1.0 + float(dataset_balance_boost) * (
                inverse_sqrt - 1.0
            )
        if trial_balance_boost > 0.0:
            trial_id = dataset.sequences[spec.trial_index].trial_id
            count = max(trial_counts[trial_id], 1)
            inverse_sqrt = 1.0 / math.sqrt(count)
            # Normalize against the median trial size so the adjustment is
            # mild and does not erase the useful motion-window distribution.
            median_count = float(np.median(list(trial_counts.values())))
            relative = math.sqrt(max(median_count, 1.0) / count)
            weights[index] *= 1.0 + float(trial_balance_boost) * (
                np.clip(relative - 1.0, -0.5, 1.0)
            )
        if phase_balance_boost > 0.0:
            phase_name = phase_names[index]
            phase_count = max(phase_counts[phase_name], 1)
            max_phase_count = max(phase_counts.values(), default=phase_count)
            relative = math.sqrt(max_phase_count / phase_count)
            weights[index] *= 1.0 + float(phase_balance_boost) * (
                np.clip(relative - 1.0, 0.0, 2.0)
            )
        if segment_balance_power > 0.0:
            # Interpolate between the natural window distribution (power=0)
            # and equal expected exposure per annotated segment (power=1).
            # Unlike phase balancing, this does not require phase semantics
            # and therefore preserves the meaning of the Z0 control.
            segment_count = max(segment_counts[segment_names[index]], 1)
            median_segment_count = float(np.median(list(segment_counts.values())))
            weights[index] *= (
                max(median_segment_count, 1.0) / float(segment_count)
            ) ** float(segment_balance_power)
        weights[index] = min(max_weight, weights[index])

    # Keep the sampler close to the original number of samples per epoch.
    # Only relative weights matter, but mean normalization makes diagnostics
    # easier to compare across runs.
    weights /= max(float(weights.mean()), 1e-8)
    weights = np.clip(weights, 1.0 / max(max_weight, 1.0), max_weight)
    return weights.astype(np.float64)


def atom_coverage_report(
    sequences: Sequence[TrialSequence],
    atom_vocab: Sequence[str],
) -> dict[str, Any]:
    """Summarize atom coverage by action and dataset-local subject group."""
    report: dict[str, Any] = {}
    for atom_index, atom in enumerate(atom_vocab):
        datasets: set[str] = set()
        groups: set[str] = set()
        trials: set[str] = set()
        segments = 0
        for sequence in sequences:
            group = subject_group_key(sequence.trial_id)
            dataset = sequence.npz_path.parent.name
            for segment in sequence.segments:
                present = (
                    segment.atom_mask[atom_index] > 0.0
                    and segment.atom_target[atom_index] > 0.0
                )
                if present:
                    segments += 1
                    datasets.add(dataset)
                    groups.add(group)
                    trials.add(sequence.trial_id)
        report[atom] = {
            "segments": segments,
            "trials": len(trials),
            "datasets": sorted(datasets),
            "subject_groups": sorted(groups),
            "n_subject_groups": len(groups),
            "cross_group_supported": len(groups) >= 2,
        }
    return report


def subject_group_key(
    trial_id: str,
    doc: Mapping[str, Any] | None = None,
    group_size: int = 10,
    subject_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Return a dataset-local subject group from fixed trial ranges.

    ``subject_info`` is retained only for call compatibility with older
    revisions. It is deliberately ignored for assignment.
    """
    del subject_info
    dataset_id = str(
        (doc or {}).get("_dataset_id")
        or str(trial_id).split("_", 1)[0]
    )

    match = re.search(r"(\d+)$", str(trial_id))
    if match is not None:
        number = int(match.group(1))
        dataset_groups = DEFAULT_SUBJECT_TRIAL_GROUPS.get(dataset_id, {})
        for subject, trial_numbers in dataset_groups.items():
            if number in trial_numbers:
                return f"{dataset_id}:{subject}"
    if doc is not None:
        for path in (
            ("subject_id",),
            ("subject",),
            ("subject_model", "subject_id"),
        ):
            value = nested_get(doc, path)
            if value not in (None, ""):
                return f"{dataset_id}:subject:{value}"

    if match is None:
        return f"{dataset_id}:trial_group:{trial_id}"
    number = int(match.group(1))
    return f"{dataset_id}:subject_group:{max((number - 1) // max(group_size, 1), 0) + 1:02d}"


def subject_groups(
    trial_ids: Sequence[str],
    docs: Mapping[str, Mapping[str, Any]],
    group_size: int,
    subject_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for trial_id in sorted(trial_ids):
        groups[
            subject_group_key(
                trial_id, docs.get(trial_id), group_size, subject_info
            )
        ].append(
            trial_id
        )
    return dict(sorted(groups.items()))


def choose_group_subsets(
    groups: Mapping[str, Sequence[str]],
    target_val: int,
    target_test: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Choose disjoint subject groups whose trial counts match the targets."""
    keys = list(groups)
    if not keys:
        return [], []
    rng = random.Random(seed)
    tie_break = {key: rng.random() for key in keys}
    best: tuple[float, float, float, tuple[str, ...], tuple[str, ...]] | None = None

    def score(val_keys: Sequence[str], test_keys: Sequence[str]) -> tuple[float, float, float]:
        val_count = sum(len(groups[key]) for key in val_keys)
        test_count = sum(len(groups[key]) for key in test_keys)
        # Prefer accurate counts, then avoid an unnecessarily small split.
        error = abs(val_count - target_val) + abs(test_count - target_test)
        small_split_penalty = float(
            (target_val > 0 and val_count == 0) or (target_test > 0 and test_count == 0)
        )
        random_tie = sum(tie_break[key] for key in [*val_keys, *test_keys])
        return float(error), small_split_penalty, random_tie

    if len(keys) <= 16:
        subsets: list[tuple[tuple[str, ...], int]] = []
        for size in range(1 << len(keys)):
            selected = tuple(
                keys[index] for index in range(len(keys)) if size & (1 << index)
            )
            subsets.append((selected, sum(len(groups[key]) for key in selected)))
        subsets.sort(key=lambda item: abs(item[1] - target_val))
        val_candidates = subsets[: min(len(subsets), 128)]
        subsets.sort(key=lambda item: abs(item[1] - target_test))
        test_candidates = subsets[: min(len(subsets), 128)]
        for val_keys, _ in val_candidates:
            val_set = set(val_keys)
            for test_keys, _ in test_candidates:
                if val_set.intersection(test_keys):
                    continue
                current = (*score(val_keys, test_keys), val_keys, test_keys)
                if best is None or current < best:
                    best = current
    else:
        shuffled = sorted(keys, key=lambda key: (len(groups[key]), tie_break[key]))
        val_keys: list[str] = []
        test_keys: list[str] = []
        val_count = test_count = 0
        for key in reversed(shuffled):
            if val_count < target_val and val_count <= test_count:
                val_keys.append(key)
                val_count += len(groups[key])
            elif test_count < target_test:
                test_keys.append(key)
                test_count += len(groups[key])
        best = (*score(val_keys, test_keys), tuple(val_keys), tuple(test_keys))

    if best is None:
        return [], []
    return list(best[-2]), list(best[-1])


class _SplitContext:
    loso_test_subject: str = ""


_SPLIT_CONTEXT = _SplitContext()


def split_trial_ids(
    trial_ids: Sequence[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    *,
    docs: Mapping[str, Mapping[str, Any]] | None = None,
    subject_info: Mapping[str, Mapping[str, Any]] | None = None,
    split_mode: str = "subject",
) -> tuple[list[str], list[str], list[str]]:
    """Split trials reproducibly without subject leakage.

    For the supplied datasets the default policy holds out subject_3 and
    subject_4 for testing.  The remaining subject_1/subject_2 trials are
    split at trial level into train and validation, stratified by dataset.
    This preserves an unseen-subject test while avoiding an unnecessarily
    small training set.  Unknown datasets retain the ratio-based group
    fallback.
    """
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    if split_mode == "file":
        # explicit split (trial-level held-out protocol): HNU_LATENT_SPLIT_JSON = {"train": [...], "val": [...],
        # "test": [...]} of trial ids; trials not listed (not used by the closed-loop evaluation) go to train
        spec = json.load(open(os.environ["HNU_LATENT_SPLIT_JSON"], encoding="utf-8"))
        val_set, test_set = set(spec["val"]), set(spec["test"])
        val_ids = sorted(t for t in trial_ids if t in val_set)
        test_ids = sorted(t for t in trial_ids if t in test_set)
        train_ids = sorted(t for t in trial_ids if t not in val_set and t not in test_set)
        return train_ids, val_ids, test_ids

    if split_mode == "loso":
        if not subject_info and docs is None:
            docs = {}
        if not subject_info and docs is None:
            raise ValueError("LOSO requires trial documents for subject mapping")
        requested_subject = str(getattr(_SPLIT_CONTEXT, "loso_test_subject", "") or "")
        # The caller passes the selected subject through the module-level
        # context because this function is kept backwards-compatible with the
        # existing three-return-value API.
        if not requested_subject:
            raise ValueError("--loso-test-subject is required with --split-mode loso")
        groups = subject_groups(trial_ids, docs or {}, 10, subject_info)
        dataset_ids = {str((docs or {}).get(t, {}).get("_dataset_id") or str(t).split("_", 1)[0]) for t in trial_ids}
        test_keys = [key for key in groups if key.rsplit(":", 1)[-1] == requested_subject]
        if len(dataset_ids) != 1:
            raise ValueError("LOSO training must receive exactly one action dataset per run")
        if not test_keys:
            available = sorted(key.rsplit(":", 1)[-1] for key in groups)
            raise ValueError(f"LOSO subject {requested_subject!r} is not present; available={available}")
        test_ids = sorted(sum((groups[key] for key in test_keys), []))
        train_val_ids = sorted(
            sum((groups[key] for key in groups if key not in set(test_keys)), [])
        )
        if len(train_val_ids) < 2:
            raise ValueError("LOSO requires at least two non-test trials")
        rng = random.Random(seed)
        candidates = list(train_val_ids)
        rng.shuffle(candidates)
        target_val = int(round(len(candidates) * val_ratio))
        target_val = max(1, target_val) if val_ratio > 0 else 0
        target_val = min(target_val, max(0, len(candidates) - 1))
        # Ensure each remaining subject contributes to validation whenever
        # enough trials are available; this is trial-level splitting, while
        # the held-out test set remains a complete unseen subject.
        pool_keys = [key for key in groups if key not in set(test_keys)]
        selected: list[str] = []
        if target_val >= len(pool_keys):
            for key in sorted(pool_keys):
                if groups[key]:
                    selected.append(sorted(groups[key])[0])
        else:
            for key in sorted(pool_keys):
                if groups[key] and len(selected) < target_val:
                    selected.append(sorted(groups[key])[0])
        selected_set = set(selected[:target_val])
        for trial_id in candidates:
            if len(selected_set) >= target_val:
                break
            selected_set.add(trial_id)
        val_ids = sorted(selected_set)
        train_ids = sorted(item for item in train_val_ids if item not in selected_set)
        return train_ids, val_ids, test_ids

    if split_mode == "random_trial":
        shuffled = list(trial_ids)
        random.Random(seed).shuffle(shuffled)
        count = len(shuffled)
        n_val = int(round(count * val_ratio))
        n_test = int(round(count * test_ratio))
        while n_val + n_test >= count and (n_val > 0 or n_test > 0):
            if n_test > 0:
                n_test -= 1
            else:
                n_val -= 1
        val_ids = shuffled[:n_val]
        test_ids = shuffled[n_val : n_val + n_test]
        train_ids = shuffled[n_val + n_test :]
        return sorted(train_ids), sorted(val_ids), sorted(test_ids)

    docs = docs or {}
    groups = subject_groups(trial_ids, docs, 10, None)
    known_group_count = 0
    train_val_pool: list[str] = []
    test_ids: list[str] = []
    unknown_groups: dict[str, list[str]] = {}
    for group_key, group_trials in groups.items():
        subject_name = group_key.rsplit(":", 1)[-1]
        split_name = DEFAULT_SUBJECT_SPLIT.get(subject_name)
        if split_name == "train_val_pool":
            train_val_pool.extend(group_trials)
            known_group_count += 1
        elif split_name == "test":
            test_ids.extend(group_trials)
            known_group_count += 1
        else:
            unknown_groups[group_key] = list(group_trials)

    # The configured datasets use the fixed subject ranges above.  Split the
    # non-test pool independently per dataset so each action contributes to
    # both optimization and model selection.
    if known_group_count == len(groups) and train_val_pool and test_ids:
        by_dataset: dict[str, list[str]] = collections.defaultdict(list)
        for trial_id in train_val_pool:
            dataset_id = str(
                (docs.get(trial_id) or {}).get("_dataset_id")
                or str(trial_id).split("_", 1)[0]
            )
            by_dataset[dataset_id].append(trial_id)

        rng = random.Random(seed)
        val_ids: list[str] = []
        train_ids: list[str] = []
        for dataset_id in sorted(by_dataset):
            candidates = sorted(by_dataset[dataset_id])
            rng.shuffle(candidates)
            target_val = int(round(len(candidates) * val_ratio))
            if val_ratio > 0:
                target_val = max(1, target_val)
            target_val = min(target_val, max(0, len(candidates) - 1))

            # Keep both known training subjects represented in validation when
            # the dataset has enough trials for that constraint.
            if target_val >= 2:
                dataset_groups = {
                    subject: [
                        trial_id
                        for trial_id in candidates
                        if subject_group_key(trial_id, docs.get(trial_id)) == (
                            f"{dataset_id}:{subject}"
                        )
                    ]
                    for subject in ("subject_1", "subject_2")
                }
                selected: list[str] = []
                for subject in ("subject_1", "subject_2"):
                    if dataset_groups[subject]:
                        selected.append(dataset_groups[subject][0])
                selected_set = set(selected[:target_val])
                for trial_id in candidates:
                    if len(selected_set) >= target_val:
                        break
                    selected_set.add(trial_id)
                dataset_val = selected_set
            else:
                dataset_val = set(candidates[:target_val])

            val_ids.extend(sorted(dataset_val))
            train_ids.extend(
                trial_id for trial_id in candidates if trial_id not in dataset_val
            )
        return sorted(train_ids), sorted(val_ids), sorted(test_ids)

    count = len(trial_ids)
    target_val = max(1, int(round(count * val_ratio))) if val_ratio > 0 else 0
    target_test = max(1, int(round(count * test_ratio))) if test_ratio > 0 else 0
    while target_val + target_test >= count and (target_val > 0 or target_test > 0):
        if target_test > 0:
            target_test -= 1
        else:
            target_val -= 1
    val_groups, test_groups = choose_group_subsets(
        groups, target_val, target_test, seed
    )
    held_out = set(val_groups).union(test_groups)
    train_groups = [key for key in groups if key not in held_out]
    val_ids = sum((groups[key] for key in val_groups), [])
    test_ids = sum((groups[key] for key in test_groups), [])
    train_ids = sum((groups[key] for key in train_groups), [])
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


# ---------------------------------------------------------------------------
# PyTorch model and losses
# ---------------------------------------------------------------------------


if torch is not None:

    def valid_group_count(channels: int, preferred: int = 8) -> int:
        """Choose the largest GroupNorm group count that divides channels."""
        for groups in range(min(preferred, channels), 0, -1):
            if channels % groups == 0:
                return groups
        return 1

    class TCNBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
            super().__init__()
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
            )
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
            )
            groups = valid_group_count(out_channels)
            self.norm1 = nn.GroupNorm(groups, out_channels)
            self.norm2 = nn.GroupNorm(groups, out_channels)
            self.dropout = nn.Dropout(dropout)
            self.residual = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )

        def forward(self, x: Any) -> Any:
            residual = self.residual(x)
            x = self.dropout(F.silu(self.norm1(self.conv1(x))))
            x = self.dropout(F.silu(self.norm2(self.conv2(x))))
            return x + residual


    class TemporalEncoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            channels: int,
            output_dim: int,
            dropout: float,
            *,
            dual_scale_pooling: bool = False,
        ):
            super().__init__()
            blocks = []
            in_channels = input_dim
            for dilation in (1, 2, 4, 8):
                blocks.append(TCNBlock(in_channels, channels, dilation, dropout))
                in_channels = channels
            self.network = nn.Sequential(*blocks)
            self.dual_scale_pooling = bool(dual_scale_pooling)
            pooled_dim = channels * (2 if self.dual_scale_pooling else 1)
            self.output = nn.Sequential(
                nn.Linear(pooled_dim, output_dim), nn.LayerNorm(output_dim)
            )

        def forward(
            self,
            sequence: Any,
            mask: Any,
            *,
            pool_mask: Any | None = None,
            local_pool_mask: Any | None = None,
            return_sequence: bool = False,
        ) -> Any:
            hidden = self.network(sequence.transpose(1, 2)).transpose(1, 2)
            weights = (mask if pool_mask is None else pool_mask).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            if self.dual_scale_pooling:
                local_weights = (
                    weights
                    if local_pool_mask is None
                    else local_pool_mask.unsqueeze(-1)
                )
                local_pooled = (hidden * local_weights).sum(dim=1) / (
                    local_weights.sum(dim=1).clamp_min(1.0)
                )
                pooled = torch.cat((pooled, local_pooled), dim=-1)
            latent = self.output(pooled)
            if return_sequence:
                return latent, hidden
            return latent


    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: Any) -> Any:
            return self.network(x)


    class MotionLatentModel(nn.Module):
        def __init__(
            self,
            *,
            skill_input_dim: int,
            quality_input_dim: int,
            phase_dim: int,
            task_dim: int,
            task_semantic_input_dim: int,
            robot_semantic_input_dim: int,
            human_semantic_input_dim: int,
            semantic_metric_dim: int,
            quality_target_dim: int,
            atom_dim: int,
            quality_tag_dim: int,
            latent_dim: int,
            quality_latent_dim: int,
            channels: int,
            dropout: float,
            local_feature_dim: int,
            dual_scale_skill: bool = False,
        ) -> None:
            super().__init__()
            for name, value in (
                ("task_semantic_input_dim", task_semantic_input_dim),
                ("robot_semantic_input_dim", robot_semantic_input_dim),
                ("human_semantic_input_dim", human_semantic_input_dim),
            ):
                if value is None or int(value) < 1:
                    raise ValueError(f"{name} must be positive for the current three-branch model")
            self.skill_encoder = TemporalEncoder(
                skill_input_dim,
                channels,
                latent_dim,
                dropout,
                dual_scale_pooling=dual_scale_skill,
            )
            self.quality_encoder = TemporalEncoder(
                quality_input_dim, channels, quality_latent_dim, dropout
            )
            # Executability is conditional: the same robot/scene state may be
            # safe for one skill and unsafe for another.  Quality losses are
            # stopped at z_skill so they cannot turn the skill code into a
            # disguised risk code.
            self.quality_conditioner = MLP(
                quality_latent_dim + latent_dim,
                channels,
                quality_latent_dim,
                dropout,
            )
            # Semantic information is intentionally source-separated.  The
            # task branch is the deployable command interface; robot and human
            # branches are optional evidence channels with explicit presence
            # flags.  A gated fusion prevents sparse/noisy human annotations
            # from overwriting the task representation in Z4.
            self.task_semantic_encoder = MLP(
                task_semantic_input_dim, channels, latent_dim, dropout
            )
            self.robot_semantic_encoder = MLP(
                robot_semantic_input_dim, channels, latent_dim, dropout
            )
            self.human_semantic_encoder = MLP(
                human_semantic_input_dim, channels, latent_dim, dropout
            )
            self.semantic_fusion = nn.Sequential(
                nn.Linear(3 * latent_dim + 3, channels),
                nn.LayerNorm(channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, 3),
            )
            self.local_feature_dim = int(max(local_feature_dim, 1))
            self.local_projection = nn.Sequential(
                nn.Linear(channels, self.local_feature_dim),
                nn.LayerNorm(self.local_feature_dim),
                nn.SiLU(),
            )
            self.task_decoder = MLP(
                latent_dim + self.local_feature_dim + phase_dim,
                channels,
                task_dim,
                dropout,
            )
            # Semantic annotations describe the segment, not every local
            # frame. Keep this decoder separate so semantic reconstruction
            # cannot accidentally consume motion features unavailable to z_sem.
            self.semantic_task_decoder = MLP(
                latent_dim + phase_dim,
                channels,
                task_dim,
                dropout,
            )
            self.semantic_metric_dim = semantic_metric_dim
            self.atom_dim = atom_dim
            self.quality_target_dim = quality_target_dim
            self.quality_tag_dim = quality_tag_dim
            self.semantic_metric_head = (
                nn.Linear(latent_dim, semantic_metric_dim)
                if semantic_metric_dim
                else None
            )
            self.atom_head = (
                nn.Linear(latent_dim, atom_dim) if atom_dim else None
            )
            # v4: separate head for human-only atom labels (shared by z_skill and z_sem).
            self.human_atom_head = (
                nn.Linear(latent_dim, atom_dim) if atom_dim else None
            )
            self.quality_head = (
                nn.Linear(quality_latent_dim, quality_target_dim)
                if quality_target_dim
                else None
            )
            self.quality_tag_head = (
                nn.Linear(quality_latent_dim, quality_tag_dim)
                if quality_tag_dim
                else None
            )
            # v5: control-relevant prediction heads on z_skill (future robot motion; human contact timing / GRF)
            aux = ctrl_aux_meta()
            self.future_head = (
                nn.Sequential(nn.Linear(latent_dim, channels), nn.SiLU(), nn.Linear(channels, int(aux["future_dim"])))
                if int(aux["future_dim"]) else None
            )
            self.human_ctrl_head = (
                nn.Sequential(nn.Linear(latent_dim, channels), nn.SiLU(), nn.Linear(channels, int(aux["human_dim"])))
                if int(aux["human_dim"]) else None
            )

        def encode_semantic(
            self,
            task_branch: Any,
            robot_branch: Any,
            human_branch: Any,
            *,
            force_task_only: bool = False,
        ) -> Any:
            task_z = self.task_semantic_encoder(task_branch)
            robot_z = self.robot_semantic_encoder(robot_branch)
            human_z = self.human_semantic_encoder(human_branch)
            # Availability flags are the last scalar in each serialized branch.
            availability = torch.stack(
                (task_branch[:, -1], robot_branch[:, -1], human_branch[:, -1]), dim=-1
            )
            logits = self.semantic_fusion(
                torch.cat((task_z, robot_z, human_z, availability), dim=-1)
            )
            # Task text is always active. Query-time task-only encoding can
            # explicitly disable both annotation branches.
            if force_task_only:
                active = torch.zeros_like(availability)
                active[:, 0] = 1.0
            else:
                active = torch.stack(
                    (torch.ones_like(availability[:, 0]),
                     availability[:, 1].clamp(0.0, 1.0),
                     availability[:, 2].clamp(0.0, 1.0)), dim=-1
                )
            logits = logits.masked_fill(active <= 0.0, -1e4)
            weights = torch.softmax(logits, dim=-1)
            return (
                weights[:, :1] * task_z
                + weights[:, 1:2] * robot_z
                + weights[:, 2:3] * human_z
            )

        def encode_quality(
            self,
            quality_sequence: Any,
            sequence_mask: Any,
            z_skill: Any,
        ) -> Any:
            """Estimate skill-conditioned quality from online-observable state."""
            state_quality = self.quality_encoder(quality_sequence, sequence_mask)
            return self.quality_conditioner(
                torch.cat((state_quality, z_skill.detach()), dim=-1)
            )

        def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
            z_skill, skill_hidden = self.skill_encoder(
                batch["skill_seq"],
                batch["seq_mask"],
                pool_mask=batch.get("skill_pool_mask"),
                local_pool_mask=batch.get("skill_local_pool_mask"),
                return_sequence=True,
            )
            z_quality = self.encode_quality(
                batch["quality_seq"], batch["seq_mask"], z_skill
            )
            z_sem = self.encode_semantic(
                batch["task_branch"], batch["robot_branch"], batch["human_branch"]
            )
            query_z_sem = self.encode_semantic(
                batch["task_query_task_branch"],
                batch["task_query_robot_branch"],
                batch["task_query_human_branch"],
                force_task_only=True,
            )
            segment_query_z_sem = self.encode_semantic(
                batch["segment_query_task_branch"],
                batch["segment_query_robot_branch"],
                batch["segment_query_human_branch"],
            )
            steps = batch["phase_seq"].shape[1]
            latent_sequence = z_skill[:, None, :].expand(-1, steps, -1)
            local_sequence = self.local_projection(skill_hidden)
            task_prediction = self.task_decoder(
                torch.cat(
                    (latent_sequence, local_sequence, batch["phase_seq"]),
                    dim=-1,
                )
            )
            semantic_latent_sequence = z_sem[:, None, :].expand(-1, steps, -1)
            semantic_task_prediction = self.semantic_task_decoder(
                torch.cat((semantic_latent_sequence, batch["phase_seq"]), dim=-1)
            )
            return {
                "z_skill": z_skill,
                "z_quality": z_quality,
                "z_sem": z_sem,
                "query_z_sem": query_z_sem,
                "segment_query_z_sem": segment_query_z_sem,
                "task_prediction": task_prediction,
                "semantic_task_prediction": semantic_task_prediction,
                "semantic_metric_prediction": (
                    self.semantic_metric_head(z_sem)
                    if self.semantic_metric_head is not None
                    else z_sem.new_zeros((z_sem.shape[0], 0))
                ),
                "skill_metric_prediction": (
                    self.semantic_metric_head(z_skill)
                    if self.semantic_metric_head is not None
                    else z_skill.new_zeros((z_skill.shape[0], 0))
                ),
                "atom_logits": (
                    self.atom_head(z_sem)
                    if self.atom_head is not None
                    else z_sem.new_zeros((z_sem.shape[0], 0))
                ),
                "skill_atom_logits": (
                    self.atom_head(z_skill)
                    if self.atom_head is not None
                    else z_skill.new_zeros((z_skill.shape[0], 0))
                ),
                "skill_human_atom_logits": (
                    self.human_atom_head(z_skill)
                    if self.human_atom_head is not None
                    else z_skill.new_zeros((z_skill.shape[0], 0))
                ),
                "sem_human_atom_logits": (
                    self.human_atom_head(z_sem)
                    if self.human_atom_head is not None
                    else z_sem.new_zeros((z_sem.shape[0], 0))
                ),
                "quality_prediction": (
                    self.quality_head(z_quality)
                    if self.quality_head is not None
                    else z_quality.new_zeros((z_quality.shape[0], 0))
                ),
                "quality_tag_logits": (
                    self.quality_tag_head(z_quality)
                    if self.quality_tag_head is not None
                    else z_quality.new_zeros((z_quality.shape[0], 0))
                ),
                "future_prediction": (
                    self.future_head(z_skill) if getattr(self, "future_head", None) is not None
                    else z_skill.new_zeros((z_skill.shape[0], 0))
                ),
                "human_ctrl_prediction": (
                    self.human_ctrl_head(z_skill) if getattr(self, "human_ctrl_head", None) is not None
                    else z_skill.new_zeros((z_skill.shape[0], 0))
                ),
            }


def masked_mse(prediction: Any, target: Any, mask: Any) -> Any:
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def weighted_masked_mse(
    prediction: Any,
    target: Any,
    mask: Any,
    sample_weight: Any | None = None,
) -> Any:
    """Masked MSE with optional per-window confidence weights."""
    error = (prediction - target) ** 2
    if sample_weight is not None:
        error = error * sample_weight[:, None, ...]
        mask = mask * sample_weight[:, None, ...]
    denominator = mask.sum().clamp_min(1.0)
    return (error * mask).sum() / denominator


def latent_cross_covariance_loss(z_skill: Any, z_quality: Any) -> Any:
    if z_skill.shape[0] < 2:
        return z_skill.new_zeros(())
    a = (z_skill - z_skill.mean(dim=0)) / (z_skill.std(dim=0) + 1e-4)
    b = (z_quality - z_quality.mean(dim=0)) / (z_quality.std(dim=0) + 1e-4)
    covariance = a.T @ b / max(z_skill.shape[0] - 1, 1)
    return covariance.square().mean()


def latent_variance_loss(z: Any, target_std: float = 0.5) -> Any:
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(target_std - std).square().mean()


def latent_covariance_loss(z: Any) -> Any:
    """Penalize correlated latent dimensions without forcing a fixed basis."""
    if z.shape[0] < 2 or z.shape[-1] < 2:
        return z.new_zeros(())
    standardized = (z - z.mean(dim=0, keepdim=True)) / (
        z.std(dim=0, unbiased=False, keepdim=True) + 1e-4
    )
    correlation = standardized.T @ standardized / max(z.shape[0], 1)
    diagonal = torch.diagonal(correlation)
    off_diagonal = correlation - torch.diag(diagonal)
    return off_diagonal.square().mean()


def latent_norm_alignment_loss(
    reference: Any,
    target: Any,
    sample_weight: Any | None = None,
) -> tuple[Any, Any]:
    """Match latent scales using a symmetric, robust log-norm objective."""
    reference_norm = reference.detach().norm(dim=-1).clamp_min(1e-4)
    target_norm = target.norm(dim=-1).clamp_min(1e-4)
    log_ratio = torch.log(target_norm / reference_norm)
    per_sample = F.smooth_l1_loss(
        log_ratio,
        torch.zeros_like(log_ratio),
        reduction="none",
    )
    if sample_weight is None:
        loss = per_sample.mean()
        mean_log_ratio = log_ratio.mean()
    else:
        weight = sample_weight.clamp_min(0.0)
        denominator = weight.sum().clamp_min(1.0)
        loss = (per_sample * weight).sum() / denominator
        mean_log_ratio = (log_ratio * weight).sum() / denominator
    # The geometric ratio is less sensitive to a few unusually large windows.
    ratio = torch.exp(mean_log_ratio.detach())
    return loss, ratio


def binary_margin_ranking_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    margin: float = 0.5,
    sample_weight: Any | None = None,
) -> Any:
    """Separate annotated positives from negatives for sparse labels.

    This complements BCE for rare labels.  It is intentionally computed from
    per-label batch means, so it adds little noise and does not change the
    latent interface or decoder dimensions.
    """
    if logits.shape[-1] == 0:
        return logits.new_zeros(())
    losses = []
    for index in range(logits.shape[-1]):
        known = mask[:, index] > 0.0
        positive = known & (targets[:, index] >= 0.5)
        negative = known & (targets[:, index] < 0.5)
        if torch.any(positive) and torch.any(negative):
            if sample_weight is None:
                positive_mean = logits[positive, index].mean()
                negative_mean = logits[negative, index].mean()
            else:
                if sample_weight.ndim == 1:
                    positive_weights = sample_weight[positive].clamp_min(1e-4)
                    negative_weights = sample_weight[negative].clamp_min(1e-4)
                else:
                    positive_weights = sample_weight[positive, index].clamp_min(1e-4)
                    negative_weights = sample_weight[negative, index].clamp_min(1e-4)
                positive_mean = (
                    logits[positive, index] * positive_weights
                ).sum() / positive_weights.sum()
                negative_mean = (
                    logits[negative, index] * negative_weights
                ).sum() / negative_weights.sum()
            losses.append(
                F.relu(
                    logits.new_tensor(float(margin))
                    - positive_mean
                    + negative_mean
                )
            )
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def segment_query_contrastive_loss(
    query: Any,
    target: Any,
    group_ids: Any,
    present: Any,
    temperature: float = 0.10,
) -> Any:
    """Contrastive alignment with same-segment positives and hard negatives.

    All windows belonging to the same segment are positives for one another;
    windows from other segments in the batch are negatives.  The implementation
    is fully vectorized to avoid the per-sample Python loop used previously.
    """
    valid = present > 0.0
    if int(valid.sum().item()) < 2:
        return query.new_zeros(())

    q = F.normalize(query[valid], dim=-1)
    t = F.normalize(target[valid].detach(), dim=-1)
    groups = group_ids[valid]
    logits = (q @ t.T) / max(float(temperature), 1e-3)

    # Same-segment windows are positives.  Entries outside the positive set
    # are masked from the numerator; every valid window remains in the
    # denominator, so other segments act as negatives.
    positive_mask = groups[:, None].eq(groups[None, :])
    positive_logsum = torch.logsumexp(
        logits.masked_fill(~positive_mask, float("-inf")), dim=1
    )
    all_logsum = torch.logsumexp(logits, dim=1)
    has_negative = (~positive_mask).any(dim=1)
    if not torch.any(has_negative):
        return query.new_zeros(())
    return (-(positive_logsum - all_logsum)[has_negative]).mean()


def phase_supervised_contrastive_loss(
    latent: Any,
    phase_class_ids: Any,
    segment_group_ids: Any,
    sample_weight: Any,
    temperature: float = 0.10,
) -> Any:
    """Cluster the same action/phase across different trial segments.

    Windows from the same exact segment are removed from both numerator and
    denominator.  This prevents long segments from creating trivial positive
    pairs and makes the objective focus on cross-trial, subject-transferable
    phase structure.  Mixed windows contribute in proportion to their
    semantic-purity weight.
    """
    valid = sample_weight > 0.0
    if int(valid.sum().item()) < 3:
        return latent.new_zeros(())

    z = F.normalize(latent[valid], dim=-1)
    labels = phase_class_ids[valid]
    groups = segment_group_ids[valid]
    weights = sample_weight[valid].clamp_min(0.0)
    logits = (z @ z.T) / max(float(temperature), 1e-3)

    different_segment = ~groups[:, None].eq(groups[None, :])
    positive_mask = labels[:, None].eq(labels[None, :]) & different_segment
    candidate_mask = different_segment
    has_positive = positive_mask.any(dim=1)
    has_negative = (
        candidate_mask & ~labels[:, None].eq(labels[None, :])
    ).any(dim=1)
    usable = has_positive & has_negative
    if not torch.any(usable):
        return latent.new_zeros(())

    positive_logsum = torch.logsumexp(
        logits.masked_fill(~positive_mask, float("-inf")), dim=1
    )
    all_logsum = torch.logsumexp(
        logits.masked_fill(~candidate_mask, float("-inf")), dim=1
    )
    per_anchor = -(positive_logsum[usable] - all_logsum[usable])
    anchor_weights = weights[usable]
    return (per_anchor * anchor_weights).sum() / anchor_weights.sum().clamp_min(1.0)


def segment_prototype_boundary_loss(
    latent: Any,
    segment_group_ids: Any,
    trial_group_ids: Any,
    segment_order_ids: Any,
    sample_weight: Any,
    max_adjacent_cosine: float = 0.70,
) -> tuple[Any, Any]:
    """Compact exact segments and separate temporally adjacent segments.

    This objective deliberately uses only segment identity and temporal order,
    not gait-phase class labels.  It can therefore be applied unchanged to
    Z0--Z4 without leaking privileged semantics into the motion-only control.
    Segment prototypes are computed inside each batch; adjacent prototypes
    from the same trial are penalized only when they are too similar.
    """
    valid = sample_weight > 0.0
    if int(valid.sum().item()) < 2:
        zero = latent.new_zeros(())
        return zero, zero

    z = F.normalize(latent[valid], dim=-1)
    groups = segment_group_ids[valid]
    trials = trial_group_ids[valid]
    orders = segment_order_ids[valid]
    weights = sample_weight[valid].clamp_min(0.0)
    unique_groups, inverse = torch.unique(groups, sorted=True, return_inverse=True)
    group_count = int(unique_groups.numel())
    if group_count == 0:
        zero = latent.new_zeros(())
        return zero, zero

    prototype_sum = z.new_zeros((group_count, z.shape[-1]))
    prototype_sum.index_add_(0, inverse, z * weights[:, None])
    prototype_weight = z.new_zeros((group_count,))
    prototype_weight.index_add_(0, inverse, weights)
    prototypes = F.normalize(
        prototype_sum / prototype_weight[:, None].clamp_min(1e-6), dim=-1
    )

    member_count = z.new_zeros((group_count,))
    member_count.index_add_(0, inverse, torch.ones_like(weights))
    repeated = member_count[inverse] > 1.0
    if torch.any(repeated):
        compact_error = 1.0 - F.cosine_similarity(z, prototypes[inverse], dim=-1)
        compact_weight = weights * repeated.to(weights.dtype)
        compactness = (compact_error * compact_weight).sum() / (
            compact_weight.sum().clamp_min(1.0)
        )
    else:
        compactness = latent.new_zeros(())

    if group_count < 2:
        return compactness, latent.new_zeros(())
    # Every segment group belongs to exactly one trial/order pair.  Selecting
    # the first member avoids phase labels and is stable under sampler repeats.
    prototype_trials = torch.stack(
        [trials[torch.nonzero(inverse == i, as_tuple=False)[0, 0]] for i in range(group_count)]
    )
    prototype_orders = torch.stack(
        [orders[torch.nonzero(inverse == i, as_tuple=False)[0, 0]] for i in range(group_count)]
    )
    upper = torch.triu(
        torch.ones((group_count, group_count), dtype=torch.bool, device=z.device),
        diagonal=1,
    )
    adjacent = (
        prototype_trials[:, None].eq(prototype_trials[None, :])
        & (prototype_orders[:, None] - prototype_orders[None, :]).abs().eq(1)
        & upper
    )
    if not torch.any(adjacent):
        return compactness, latent.new_zeros(())
    cosine = prototypes @ prototypes.T
    boundary = F.relu(cosine[adjacent] - float(max_adjacent_cosine)).mean()
    return compactness, boundary

def compute_losses(
    batch: Mapping[str, Any],
    output: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    semantic_enabled: bool,
    semantic_scale: float = 1.0,
) -> tuple[Any, dict[str, float]]:
    zero = output["z_skill"].new_zeros(())
    frame_error = F.smooth_l1_loss(
        output["task_prediction"], batch["task_target"], reduction="none"
    ).mean(dim=-1)
    per_sample_recon = (
        frame_error * batch["seq_mask"]
    ).sum(dim=1) / batch["seq_mask"].sum(dim=1).clamp_min(1.0)
    recon = (
        per_sample_recon * batch["reconstruction_weight"]
    ).sum() / batch["reconstruction_weight"].sum().clamp_min(1.0)

    prediction_delta = output["task_prediction"][:, 1:] - output["task_prediction"][:, :-1]
    target_delta = batch["task_target"][:, 1:] - batch["task_target"][:, :-1]
    delta_mask = batch["seq_mask"][:, 1:] * batch["seq_mask"][:, :-1]
    smooth = (
        (prediction_delta - target_delta).abs().mean(dim=-1) * delta_mask
    ).sum() / delta_mask.sum().clamp_min(1.0)

    segment_weight = batch.get("semantic_window_weight")
    quality = (
        weighted_masked_mse(
            output["quality_prediction"],
            batch["quality_target"],
            batch["quality_mask"],
            segment_weight,
        )
        if output["quality_prediction"].shape[-1]
        else zero
    )
    if output["quality_tag_logits"].shape[-1]:
        tag_pos_weight = (
            batch["quality_tag_pos_weight"][0]
            if batch["quality_tag_pos_weight"].ndim == 2
            else batch["quality_tag_pos_weight"]
        )
        tag_target = batch["quality_tags"].clamp(0.0, 1.0)
        tag_mask = batch["quality_tag_mask"]
        if hasattr(tag_pos_weight, "to"):
            tag_pos_weight = tag_pos_weight.to(output["quality_tag_logits"].device)
        else:
            tag_pos_weight = output["quality_tag_logits"].new_tensor(tag_pos_weight)
        tag_prob = torch.sigmoid(output["quality_tag_logits"])
        ce = F.binary_cross_entropy_with_logits(
            output["quality_tag_logits"], tag_target, reduction="none"
        )
        # Balanced focal: inverse-frequency positive weight + focal focusing.
        alpha = float(np.clip(getattr(args, "quality_focal_alpha", 0.75), 0.0, 1.0))
        gamma = max(float(getattr(args, "quality_focal_gamma", 2.0)), 0.0)
        negative_weight = max(float(getattr(args, "quality_negative_weight", 0.5)), 0.0)
        class_weight = torch.where(tag_target >= 0.5, tag_pos_weight, torch.full_like(tag_pos_weight, negative_weight))
        focal_weight = torch.where(tag_target >= 0.5, alpha, 1.0 - alpha)
        focal_weight = focal_weight * (1.0 - torch.where(tag_target >= 0.5, tag_prob, 1.0 - tag_prob)).pow(gamma)
        tag_raw = ce * class_weight * focal_weight * tag_mask
        if segment_weight is not None:
            tag_raw = tag_raw * segment_weight[:, None]
            tag_mask = tag_mask * segment_weight[:, None]
        tag_counts = tag_mask.sum(dim=0)
        tag_means = tag_raw.sum(dim=0) / tag_counts.clamp_min(1.0)
        known_tag_means = tag_means[tag_counts > 0.0]
        # Z0 intentionally masks all quality supervision.  Taking the mean of
        # an empty tensor would produce NaN and poison the whole objective.
        quality_tags = known_tag_means.mean() if known_tag_means.numel() else zero
    else:
        quality_tags = zero
    quality_tag_ranking = (
        binary_margin_ranking_loss(
            output["quality_tag_logits"],
            batch["quality_tags"],
            batch["quality_tag_mask"],
            args.quality_tag_ranking_margin,
            segment_weight,
        )
        if output["quality_tag_logits"].shape[-1]
        else zero
    )
    decorrelation = latent_cross_covariance_loss(
        output["z_skill"], output["z_quality"]
    )
    variance = latent_variance_loss(output["z_skill"])
    z_sem = output["z_sem"]
    variance = variance + latent_variance_loss(z_sem)
    skill_covariance = latent_covariance_loss(output["z_skill"])
    quality_covariance = latent_covariance_loss(output["z_quality"])
    semantic_covariance = latent_covariance_loss(z_sem)
    segment_compactness, segment_boundary = segment_prototype_boundary_loss(
        output["z_skill"],
        batch["segment_group_index"],
        batch["trial_group_index"],
        batch["segment_order_index"],
        batch["reconstruction_weight"],
        args.segment_boundary_max_cosine,
    )

    align = metric = atoms = atom_ranking = semantic_decode = zero
    skill_metric = skill_atoms = skill_atom_ranking = zero
    human_skill_atoms = human_sem_atoms = zero
    semantic_norm = semantic_norm_ratio = zero
    query_align = query_consistency = deployment_consistency = zero
    segment_query_align = segment_query_contrastive = zero
    phase_contrastive = zero
    if semantic_enabled:
        semantic_weight = (
            segment_weight
            if segment_weight is not None
            else torch.ones_like(batch["semantic_confidence"])
        )
        if args.lambda_semantic_norm > 0.0:
            semantic_norm_weight = (
                batch["semantic_confidence"]
                * batch["semantic_supervision_weight"]
                * semantic_weight
            )
            semantic_norm, semantic_norm_ratio = latent_norm_alignment_loss(
                output["z_skill"],
                output["z_sem"],
                semantic_norm_weight,
            )
        query_weight = batch["task_query_present"] * semantic_weight
        query_denominator = query_weight.sum().clamp_min(1.0)
        query_skill_cosine = F.cosine_similarity(
            output["z_skill"].detach(), output["query_z_sem"], dim=-1
        )
        query_align = (
            F.relu(args.query_align_margin - query_skill_cosine) * query_weight
        ).sum() / query_denominator
        query_full_cosine = F.cosine_similarity(
            output["z_sem"].detach(), output["query_z_sem"], dim=-1
        )
        query_consistency = (
            (1.0 - query_full_cosine) * query_weight
        ).sum() / query_denominator
        deployment_cosine = F.cosine_similarity(
            output["z_sem"], output["query_z_sem"], dim=-1
        )
        deployment_consistency = (
            (1.0 - deployment_cosine) * query_weight
        ).sum() / query_denominator
        segment_query = output["segment_query_z_sem"]
        segment_query_weight = batch["segment_query_present"] * semantic_weight
        segment_query_denominator = segment_query_weight.sum().clamp_min(1.0)
        segment_query_cosine = F.cosine_similarity(
            output["z_skill"].detach(), segment_query, dim=-1
        )
        segment_query_align = (
            F.relu(args.segment_query_align_margin - segment_query_cosine)
            * segment_query_weight
        ).sum() / segment_query_denominator
        segment_query_contrastive = segment_query_contrastive_loss(
            segment_query,
            output["z_skill"],
            batch["segment_group_index"],
            segment_query_weight,
            args.segment_query_temperature,
        )
        if args.lambda_phase_contrastive > 0.0:
            phase_contrastive = phase_supervised_contrastive_loss(
                output["z_skill"],
                batch["phase_class_index"],
                batch["segment_group_index"],
                semantic_weight,
                args.phase_contrastive_temperature,
            )
        # The motion encoder is the anchor.  Noisy semantic labels should
        # train z_sem to approximate z_skill, but should not move z_skill
        # through the alignment objective.
        cosine = F.cosine_similarity(
            output["z_skill"].detach(), output["z_sem"], dim=-1
        )
        align = (
            F.relu(args.semantic_align_margin - cosine)
            * batch["semantic_confidence"]
            * (
                segment_weight
                if segment_weight is not None
                else torch.ones_like(batch["semantic_confidence"])
            )
        ).sum() / (
            batch["semantic_confidence"]
            * (
                segment_weight
                if segment_weight is not None
                else torch.ones_like(batch["semantic_confidence"])
            )
        ).sum().clamp_min(1.0)
        metric = weighted_masked_mse(
            output["semantic_metric_prediction"],
            batch["semantic_metric_target"],
            batch["semantic_metric_mask"],
            semantic_weight,
        )
        # The exported/distilled representation is z_skill.  A small direct
        # privileged-label objective makes the Z1--Z4 ablations actually
        # affect that representation; the former stop-gradient-only design
        # trained z_sem but left z_skill almost unchanged by semantic data.
        skill_metric = weighted_masked_mse(
            output["skill_metric_prediction"],
            batch["semantic_metric_target"],
            batch.get("skill_metric_mask", batch["semantic_metric_mask"]),
            semantic_weight,
        )
        atom_target = batch["atom_target"]
        atom_smoothing = float(
            np.clip(getattr(args, "atom_label_smoothing", 0.0), 0.0, 0.25)
        )
        atom_loss_target = (
            atom_target * (1.0 - atom_smoothing) + 0.5 * atom_smoothing
        )
        atom_pos_weight = (
            batch["atom_pos_weight"][0]
            if batch["atom_pos_weight"].ndim == 2
            else batch["atom_pos_weight"]
        )
        gamma = max(float(getattr(args, "atom_focal_gamma", 0.0)), 0.0)
        atom_weight = torch.where(
            atom_target > 0,
            batch["atom_supervision_weight"],
            args.atom_negative_weight * batch["atom_supervision_weight"],
        )
        atom_weight = atom_weight * batch["atom_mask"] * semantic_weight[:, None]
        atom_counts = (batch["atom_mask"] * semantic_weight[:, None]).sum(dim=0)
        atom_positive_counts = (
            (
                (atom_target > 0.0)
                & (batch["atom_mask"] > 0.0)
            ).to(atom_counts.dtype)
            * semantic_weight[:, None]
        ).sum(dim=0)
        # A sparse atom that has no positives in the current batch contributes
        # almost only negative evidence. Repeating that over many batches can
        # overwhelm the few positive updates and collapse rare-atom recall.
        informative_atoms = (atom_counts > 0.0) & (atom_positive_counts > 0.0)

        def atom_objective(logits: Any) -> tuple[Any, Any]:
            # Weight only the positive term so rare atoms receive more
            # gradient. Missing atoms are unknown and remain masked.
            probability = torch.sigmoid(logits)
            positive_term = (
                -atom_pos_weight
                * atom_loss_target
                * (1.0 - probability).pow(gamma)
                * F.logsigmoid(logits)
            )
            negative_term = (
                -(1.0 - atom_loss_target)
                * probability.pow(gamma)
                * F.logsigmoid(-logits)
            )
            raw = positive_term + negative_term
            means = (raw * atom_weight).sum(dim=0) / atom_counts.clamp_min(1.0)
            objective = (
                means[informative_atoms].mean()
                if torch.any(informative_atoms)
                else zero
            )
            ranking = binary_margin_ranking_loss(
                logits,
                atom_target,
                batch["atom_mask"],
                args.atom_ranking_margin,
                batch["atom_supervision_weight"] * semantic_weight[:, None],
            )
            return objective, ranking

        atoms, atom_ranking = atom_objective(output["atom_logits"])
        skill_atoms, skill_atom_ranking = atom_objective(
            output["skill_atom_logits"]
        )
        # v4: human-only atoms, masked BCE (no override by robot labels)
        if "human_atom_mask" in batch and output["skill_human_atom_logits"].shape[-1]:
            hm = batch["human_atom_mask"] * semantic_weight[:, None]
            ht = batch["human_atom_target"]

            def human_bce(logits: Any) -> Any:
                raw = F.binary_cross_entropy_with_logits(logits, ht, reduction="none")
                return (raw * hm).sum() / hm.sum().clamp_min(1.0)

            human_skill_atoms = human_bce(output["skill_human_atom_logits"])
            human_sem_atoms = human_bce(output["sem_human_atom_logits"])

        semantic_frame_error = F.smooth_l1_loss(
            output["semantic_task_prediction"],
            batch["task_target"],
            reduction="none",
        )
        dimension_weight = batch["semantic_task_weight"][:, None, :]
        semantic_frame_error = (
            semantic_frame_error * dimension_weight
        ).sum(dim=-1) / dimension_weight.sum(dim=-1).clamp_min(1.0)
        semantic_per_sample = (
            semantic_frame_error * batch["seq_mask"]
        ).sum(dim=1) / batch["seq_mask"].sum(dim=1).clamp_min(1.0)
        semantic_sample_weight = (
            batch["semantic_confidence"]
            * batch["semantic_supervision_weight"]
            * semantic_weight
            * batch["reconstruction_weight"]
        )
        semantic_decode = (
            semantic_per_sample * semantic_sample_weight
        ).sum() / semantic_sample_weight.sum().clamp_min(1.0)

    total = (
        args.lambda_recon * recon
        + args.lambda_smooth * smooth
        + args.lambda_quality * (quality + quality_tags)
        + args.lambda_quality_tag_ranking * quality_tag_ranking
        + args.lambda_decorrelation * decorrelation
        + args.lambda_variance * variance
        + args.lambda_skill_covariance * skill_covariance
        + args.lambda_quality_covariance * quality_covariance
        + args.lambda_semantic_covariance * semantic_covariance
        + args.lambda_segment_compactness * segment_compactness
        + args.lambda_segment_boundary * segment_boundary
    )
    if semantic_enabled:
        total = total + (
            semantic_scale
            * (
                args.lambda_align * align
                + args.lambda_metric * metric
                + args.lambda_atoms * atoms
                + args.lambda_atom_ranking * atom_ranking
                + args.lambda_skill_metric * skill_metric
                + args.lambda_skill_atoms * skill_atoms
                + args.lambda_skill_atoms * 0.25 * skill_atom_ranking
                + getattr(args, "lambda_human_skill_atoms", 0.0) * human_skill_atoms
                + getattr(args, "lambda_human_sem_atoms", 0.0) * human_sem_atoms
                + args.lambda_semantic_decode * semantic_decode
                + args.lambda_semantic_norm * semantic_norm
                + args.lambda_query_align * query_align
                + args.lambda_query_consistency * query_consistency
                + args.lambda_deployment_consistency * deployment_consistency
                + args.lambda_segment_query_align * segment_query_align
                + args.lambda_segment_query_contrastive * segment_query_contrastive
                + args.lambda_phase_contrastive * phase_contrastive
            )
        )
    # v4b: deployable routing objective for every level -- the task-only query z_sem (task_goal or, with query
    # augmentation, a profile-owned description) must retrieve z_skill windows of the same action
    # (supervised contrastive over the batch, temperature 0.1).
    query_task = zero
    lam_qt = float(getattr(args, "lambda_query_task", 0.0))
    if lam_qt > 0 and "action_index" in batch:
        act = batch["action_index"]
        ok = act >= 0
        if ok.sum() > 1 and len(torch.unique(act[ok])) > 1:
            q = F.normalize(output["query_z_sem"][ok], dim=-1)
            zk = output["z_skill"][ok]
            if getattr(args, "query_task_stopgrad", 0):
                # v4d: same principle as every other alignment term -- the motion encoder is the anchor,
                # language may not move the control latent z_skill
                zk = zk.detach()
            k = F.normalize(zk, dim=-1)
            logits = q @ k.T / 0.1
            pos = (act[ok][:, None] == act[ok][None, :]).float()
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            query_task = -((log_prob * pos).sum(1) / pos.sum(1).clamp_min(1.0)).mean()
            total = total + lam_qt * query_task
    # v5: z_skill must predict how the robot will move next (all levels: robot reference itself) and, where the
    # level has human numeric data (L3/L4), when the feet touch down / lift off and how the ground is loaded.
    future_loss, human_ctrl_loss = zero, zero
    lam_f = float(getattr(args, "lambda_future", 0.0))
    if lam_f > 0 and output["future_prediction"].shape[-1]:
        future_loss = masked_mse(output["future_prediction"], batch["future_target"], batch["future_mask"])
        total = total + lam_f * future_loss
    lam_h = float(getattr(args, "lambda_human_ctrl", 0.0))
    if not ablation_config(args.ablation_profile)["human_numeric"]:
        lam_h = 0.0  # same rule as every other human signal: only levels that own human numeric data
    if lam_h > 0 and output["human_ctrl_prediction"].shape[-1]:
        hmask = batch["human_ctrl_mask"]
        if not HUMAN_DYNAMICS or not HUMAN_TIMING:  # L3-kin: no GRF targets; L3-rom: no timing targets either
            keys = ctrl_aux_meta().get("human_keys", [])
            drop = lambda k: (k.startswith("grf") and not HUMAN_DYNAMICS) or (  # noqa: E731
                k.startswith(("contact", "t_next")) and not HUMAN_TIMING)
            keep = torch.tensor([0.0 if drop(k) else 1.0 for k in keys], dtype=hmask.dtype, device=hmask.device)
            hmask = hmask * keep
        human_ctrl_loss = masked_mse(output["human_ctrl_prediction"], batch["human_ctrl_target"], hmask)
        total = total + lam_h * human_ctrl_loss
    values = {
        "loss": float(total.detach()),
        "future": float(future_loss.detach()),
        "human_ctrl": float(human_ctrl_loss.detach()),
        "query_task": float(query_task.detach()),
        "recon": float(recon.detach()),
        "smooth": float(smooth.detach()),
        "quality": float(quality.detach()),
        "quality_tags": float(quality_tags.detach()),
        "quality_tag_ranking": float(quality_tag_ranking.detach()),
        "decorrelation": float(decorrelation.detach()),
        "variance": float(variance.detach()),
        "skill_covariance": float(skill_covariance.detach()),
        "quality_covariance": float(quality_covariance.detach()),
        "semantic_covariance": float(semantic_covariance.detach()),
        "segment_compactness": float(segment_compactness.detach()),
        "segment_boundary": float(segment_boundary.detach()),
        "align": float(align.detach()),
        "metric": float(metric.detach()),
        "skill_metric": float(skill_metric.detach()),
        "atoms": float(atoms.detach()),
        "skill_atoms": float(skill_atoms.detach()),
        "skill_atom_ranking": float(skill_atom_ranking.detach()),
        "human_skill_atoms": float(human_skill_atoms.detach()),
        "human_sem_atoms": float(human_sem_atoms.detach()),
        "atom_ranking": float(atom_ranking.detach()),
        "semantic_decode": float(semantic_decode.detach()),
        "semantic_norm": float(semantic_norm.detach()),
        "semantic_norm_ratio": float(semantic_norm_ratio.detach()),
        "query_align": float(query_align.detach()),
        "query_consistency": float(query_consistency.detach()),
        "deployment_consistency": float(deployment_consistency.detach()),
        "segment_query_align": float(segment_query_align.detach()),
        "segment_query_contrastive": float(segment_query_contrastive.detach()),
        "phase_contrastive": float(phase_contrastive.detach()),
        "semantic_scale": float(semantic_scale if semantic_enabled else 0.0),
    }
    return total, values


def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype != torch.int64:
                value = value.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
            else:
                value = value.to(device=device, non_blocking=True)
        result[key] = value
    return result


def aggregate_metrics(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {
        key: float(np.mean([item[key] for item in items]))
        for key in items[0]
    }


def masked_macro_f1(
    logits: Any,
    targets: Any,
    mask: Any,
    threshold: float = 0.5,
) -> Any:
    """Macro F1 for annotated binary labels, ignoring unknown entries."""
    if logits.shape[-1] == 0:
        return logits.new_zeros(())
    predicted = torch.sigmoid(logits) >= threshold
    actual = targets >= 0.5
    valid = mask > 0.0
    f1_values = []
    for index in range(logits.shape[-1]):
        keep = valid[:, index]
        if not torch.any(keep):
            continue
        pred = predicted[keep, index]
        truth = actual[keep, index]
        tp = (pred & truth).sum().float()
        fp = (pred & ~truth).sum().float()
        fn = (~pred & truth).sum().float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1_values.append(2.0 * precision * recall / (precision + recall).clamp_min(1e-6))
    return (
        torch.stack(f1_values).mean()
        if f1_values
        else logits.new_zeros(())
    )


def masked_f1_counts(
    logits: Any,
    targets: Any,
    mask: Any,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-label TP/FP/FN/known counts for global epoch metrics."""
    if logits.shape[-1] == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty
    predicted = torch.sigmoid(logits) >= threshold
    actual = targets >= 0.5
    valid = mask > 0.0
    tp = (predicted & actual & valid).sum(dim=0).detach().cpu().numpy()
    fp = (predicted & ~actual & valid).sum(dim=0).detach().cpu().numpy()
    fn = (~predicted & actual & valid).sum(dim=0).detach().cpu().numpy()
    known = valid.sum(dim=0).detach().cpu().numpy()
    return (
        np.asarray(tp, dtype=np.float64),
        np.asarray(fp, dtype=np.float64),
        np.asarray(fn, dtype=np.float64),
        np.asarray(known, dtype=np.float64),
    )


def macro_f1_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    known: np.ndarray,
    *,
    max_positive_count: float | None = None,
    min_positive_count: float = 1.0,
    min_negative_count: float = 0.0,
) -> float:
    """Compute macro F1 from aggregated counts.

    Axes without sufficient two-sided support are ignored. This avoids
    selecting a checkpoint from a validation fold where sparse/unknown
    quality labels contain only positives or only negatives.
    """
    positives = tp + fn
    negatives = np.maximum(known - positives, 0.0)
    valid = (known > 0.0) & (positives >= float(min_positive_count))
    if min_negative_count > 0.0:
        valid &= negatives >= float(min_negative_count)
    if max_positive_count is not None:
        valid &= positives <= float(max_positive_count)
    if not np.any(valid):
        return 0.0
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
    return float(f1[valid].mean())
def update_ema_model(ema_model: Any, model: Any, decay: float) -> None:
    """Update a frozen exponential moving average of model parameters."""
    decay = float(np.clip(decay, 0.0, 0.999999))
    with torch.no_grad():
        for ema_value, value in zip(
            ema_model.state_dict().values(), model.state_dict().values()
        ):
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def semantic_loss_scale(epoch_index: int, args: argparse.Namespace) -> float:
    """Ramp semantic supervision after warmup to avoid an abrupt objective jump."""
    if epoch_index < args.warmup_epochs:
        return 0.0
    if args.semantic_ramp_epochs <= 0:
        return 1.0
    progress = (epoch_index - args.warmup_epochs + 1) / args.semantic_ramp_epochs
    return float(np.clip(progress, 0.0, 1.0))


def segment_phase_retrieval_metrics(
    latents: np.ndarray,
    phase_ids: np.ndarray,
    segment_ids: np.ndarray,
    trial_ids: np.ndarray,
) -> dict[str, float]:
    """Cross-trial cosine retrieval on segment-mean validation latents."""
    if len(latents) == 0:
        return {
            "phase_retrieval_r1": 0.0,
            "phase_retrieval_mrr": 0.0,
            "phase_retrieval_coverage": 0.0,
        }
    unique_segments = np.unique(segment_ids)
    segment_latents: list[np.ndarray] = []
    segment_phases: list[int] = []
    segment_trials: list[int] = []
    for segment_id in unique_segments:
        keep = segment_ids == segment_id
        segment_latents.append(np.asarray(latents[keep]).mean(axis=0))
        segment_phases.append(int(np.asarray(phase_ids[keep])[0]))
        segment_trials.append(int(np.asarray(trial_ids[keep])[0]))
    values = np.asarray(segment_latents, dtype=np.float64)
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-9)
    phases = np.asarray(segment_phases, dtype=np.int64)
    trials = np.asarray(segment_trials, dtype=np.int64)
    similarities = values @ values.T
    ranks: list[int] = []
    for query_index in range(len(values)):
        gallery = np.flatnonzero(trials != trials[query_index])
        relevant = phases[gallery] == phases[query_index]
        if not np.any(relevant):
            continue
        order = gallery[np.argsort(-similarities[query_index, gallery], kind="stable")]
        match_positions = np.flatnonzero(phases[order] == phases[query_index])
        if len(match_positions):
            ranks.append(int(match_positions[0]) + 1)
    coverage = len(ranks) / max(len(values), 1)
    if not ranks:
        return {
            "phase_retrieval_r1": 0.0,
            "phase_retrieval_mrr": 0.0,
            "phase_retrieval_coverage": float(coverage),
        }
    rank_values = np.asarray(ranks, dtype=np.float64)
    return {
        "phase_retrieval_r1": float(np.mean(rank_values <= 1.0)),
        "phase_retrieval_mrr": float(np.mean(1.0 / rank_values)),
        "phase_retrieval_coverage": float(coverage),
    }


def run_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    args: argparse.Namespace,
    *,
    training: bool,
    semantic_enabled: bool,
    semantic_scale: float = 1.0,
    ema_model: Any | None = None,
) -> dict[str, float]:
    model.train(training)
    records: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    atom_counts: list[np.ndarray] = []
    skill_atom_counts: list[np.ndarray] = []
    tag_counts: list[np.ndarray] = []
    retrieval_latents: list[np.ndarray] = []
    retrieval_phases: list[np.ndarray] = []
    retrieval_segments: list[np.ndarray] = []
    retrieval_trials: list[np.ndarray] = []
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            noise_std = (
                float(getattr(args, "motion_noise_std", 0.0))
                if training
                else 0.0
            )
            if noise_std > 0.0:
                batch["skill_seq"] = batch["skill_seq"] + noise_std * torch.randn_like(
                    batch["skill_seq"]
                )
                batch["quality_seq"] = batch["quality_seq"] + noise_std * torch.randn_like(
                    batch["quality_seq"]
                )
            if training and semantic_enabled:
                # Whole-source dropout simulates loss of privileged annotation
                # channels.  Do not alter task/segment query branches: those
                # already contain only deployment-observable information.
                source_drop_p = float(np.clip(
                    getattr(args, "semantic_input_dropout", 0.0), 0.0, 0.95
                ))
                if source_drop_p > 0.0:
                    for branch_key in ("robot_branch", "human_branch"):
                        values = batch.get(branch_key)
                        if values is None:
                            continue
                        keep = (
                            torch.rand(values.shape[0], 1, device=values.device)
                            >= source_drop_p
                        ).to(values.dtype)
                        batch[branch_key] = values * keep

                # Metric values and masks must be dropped as pairs.  The old
                # elementwise F.dropout independently corrupted masks and
                # rescaled retained standardized values.
                metric_drop_p = float(np.clip(
                    getattr(args, "semantic_metric_input_dropout", 0.0),
                    0.0,
                    0.95,
                ))
                metric_dim = int(getattr(model, "semantic_metric_dim", 0))
                if metric_drop_p > 0.0 and metric_dim > 0:
                    for branch_key in ("robot_branch", "human_branch"):
                        values = batch.get(branch_key)
                        if values is None:
                            continue
                        keep = (
                            torch.rand(
                                values.shape[0], metric_dim,
                                device=values.device,
                            ) >= metric_drop_p
                        ).to(values.dtype)
                        values = values.clone()
                        values[:, :metric_dim] *= keep
                        values[:, metric_dim:2 * metric_dim] *= keep
                        batch[branch_key] = values

                # Human text is one modality, so drop it per sample without
                # inverse-probability rescaling. Numeric human metrics remain.
                text_drop_p = float(np.clip(
                    getattr(args, "semantic_text_dropout", 0.0), 0.0, 0.95
                ))
                if text_drop_p > 0.0 and "human_branch" in batch:
                    values = batch["human_branch"].clone()
                    text_dim = int(getattr(args, "text_dim", 0))
                    text_start = values.shape[-1] - 1 - text_dim
                    if text_dim > 0 and text_start >= 2 * metric_dim:
                        keep = (
                            torch.rand(values.shape[0], 1, device=values.device)
                            >= text_drop_p
                        ).to(values.dtype)
                        values[:, text_start:text_start + text_dim] *= keep
                        batch["human_branch"] = values
            output = model(batch)

            if not training:
                retrieval_latents.append(output["z_skill"].detach().cpu().numpy())
                retrieval_phases.append(
                    batch["phase_class_index"].detach().cpu().numpy()
                )
                retrieval_segments.append(
                    batch["segment_group_index"].detach().cpu().numpy()
                )
                retrieval_trials.append(
                    batch["trial_group_index"].detach().cpu().numpy()
                )

            loss, values = compute_losses(
                batch,
                output,
                args,
                semantic_enabled=semantic_enabled,
                semantic_scale=semantic_scale,
            )
            if semantic_enabled:
                atom_counts.append(
                    np.stack(masked_f1_counts(
                        output["atom_logits"],
                        batch["atom_target"],
                        batch["atom_mask"],
                    ))
                )
                skill_atom_counts.append(
                    np.stack(masked_f1_counts(
                        output["skill_atom_logits"],
                        batch["atom_target"],
                        batch["atom_mask"],
                    ))
                )
                tag_counts.append(
                    np.stack(masked_f1_counts(
                        output["quality_tag_logits"],
                        batch["quality_tags"],
                        batch["quality_tag_mask"],
                    ))
                )
                values["atom_f1"] = float(
                    masked_macro_f1(
                        output["atom_logits"],
                        batch["atom_target"],
                        batch["atom_mask"],
                    ).detach()
                )
                values["skill_atom_f1"] = float(
                    masked_macro_f1(
                        output["skill_atom_logits"],
                        batch["atom_target"],
                        batch["atom_mask"],
                    ).detach()
                )
                values["quality_tag_f1"] = float(
                    masked_macro_f1(
                        output["quality_tag_logits"],
                        batch["quality_tags"],
                        batch["quality_tag_mask"],
                    ).detach()
                )
            else:
                values["atom_f1"] = 0.0
                values["skill_atom_f1"] = 0.0
                values["quality_tag_f1"] = 0.0
                values["rare_atom_f1"] = 0.0
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if ema_model is not None and args.ema_decay > 0.0:
                    update_ema_model(ema_model, model, args.ema_decay)
            records.append(values)
    result = aggregate_metrics(records)
    if semantic_enabled and atom_counts:
        atom_tp, atom_fp, atom_fn, atom_known = np.sum(
            np.stack(atom_counts), axis=0
        )
        result["atom_f1"] = macro_f1_from_counts(
            atom_tp, atom_fp, atom_fn, atom_known
        )
        result["rare_atom_f1"] = macro_f1_from_counts(
            atom_tp,
            atom_fp,
            atom_fn,
            atom_known,
            max_positive_count=args.rare_atom_positive_threshold,
        )
    if semantic_enabled and skill_atom_counts:
        skill_tp, skill_fp, skill_fn, skill_known = np.sum(
            np.stack(skill_atom_counts), axis=0
        )
        result["skill_atom_f1"] = macro_f1_from_counts(
            skill_tp, skill_fp, skill_fn, skill_known
        )
    if semantic_enabled and tag_counts:
        tag_tp, tag_fp, tag_fn, tag_known = np.sum(
            np.stack(tag_counts), axis=0
        )
        tag_positive = tag_tp + tag_fn
        tag_negative = np.maximum(tag_known - tag_positive, 0.0)
        # F1 participates in checkpoint selection only when at least one
        # quality axis has sufficient known positives and negatives.
        quality_valid_axes = (tag_positive >= 2.0) & (tag_negative >= 2.0)
        result["quality_tag_f1"] = macro_f1_from_counts(
            tag_tp, tag_fp, tag_fn, tag_known,
            min_positive_count=2.0,
            min_negative_count=2.0,
        )
        result["quality_tag_selection_valid"] = float(np.any(quality_valid_axes))
        result["quality_tag_evaluable_axes"] = float(quality_valid_axes.sum())
        result["quality_tag_positive_support"] = float(tag_positive[quality_valid_axes].sum())
        result["quality_tag_negative_support"] = float(tag_negative[quality_valid_axes].sum())
    else:
        result["quality_tag_selection_valid"] = 0.0
        result["quality_tag_evaluable_axes"] = 0.0
        result["quality_tag_positive_support"] = 0.0
        result["quality_tag_negative_support"] = 0.0

    if retrieval_latents:
        result.update(
            segment_phase_retrieval_metrics(
                np.concatenate(retrieval_latents, axis=0),
                np.concatenate(retrieval_phases, axis=0),
                np.concatenate(retrieval_segments, axis=0),
                np.concatenate(retrieval_trials, axis=0),
            )
        )
    else:
        result.update(
            {
                "phase_retrieval_r1": 0.0,
                "phase_retrieval_mrr": 0.0,
                "phase_retrieval_coverage": 0.0,
            }
        )

    return result


def export_latents(
    model: Any,
    dataset: MotionWindowDataset,
    output_path: Path,
    device: Any,
    batch_size: int,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    arrays: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    sample_indices: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            sample_indices.extend(batch["sample_index"].numpy().tolist())
            moved = move_batch_to_device(batch, device)
            output = model(moved)
            for key in ("z_skill", "z_quality", "z_sem", "query_z_sem"):
                arrays[key].append(output[key].cpu().numpy())
    metadata = [dataset.metadata(index) for index in sample_indices]
    np.savez_compressed(
        output_path,
        z_skill=np.concatenate(arrays["z_skill"], axis=0),
        z_quality=np.concatenate(arrays["z_quality"], axis=0),
        z_sem=np.concatenate(arrays["z_sem"], axis=0),
        # Explicit deployment representation: encoded from task text only,
        # with robot/human annotation channels masked.  Keeping this separate
        # prevents downstream code from accidentally indexing teacher-side
        # semantics that the robot cannot observe.
        z_sem_deploy=np.concatenate(arrays["query_z_sem"], axis=0),
        trial_id=np.array([item["trial_id"] for item in metadata]),
        segment_id=np.array([item["segment_id"] for item in metadata]),
        task_name=np.array([item["task_name"] for item in metadata]),
        task_goal=np.array([item["task_goal"] for item in metadata]),
        task_family=np.array(
            ["|".join(item["task_family"]) for item in metadata]
        ),
        gait_phase=np.array([item["gait_phase"] for item in metadata]),
        start_index=np.array([item["start_index"] for item in metadata]),
        center_index=np.array([item["center_index"] for item in metadata]),
        semantic_purity=np.array(
            [item["semantic_purity"] for item in metadata],
            dtype=np.float32,
        ),
        semantic_window_weight=np.array(
            [item["semantic_window_weight"] for item in metadata],
            dtype=np.float32,
        ),
        risk_score=np.array(
            [item["risk_score"] for item in metadata],
            dtype=np.float32,
        ),
        feasibility_score=np.array(
            [item["feasibility_score"] for item in metadata],
            dtype=np.float32,
        ),
        feasibility_tags=np.array(
            ["|".join(item["feasibility_tags"]) for item in metadata]
        ),
    )


# ---------------------------------------------------------------------------
# End-to-end preparation and training
# ---------------------------------------------------------------------------


def prepare_data(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, Any]],
    FeatureSchema,
    list[TrialSequence],
    list[TrialSequence],
    list[TrialSequence],
    Normalizers,
    dict[str, dict[str, Any]],
]:
    if not args.legacy_single_dataset:
        docs = load_dataset_documents(
            args.datasets,
            args.json_root,
            args.retargeted_json_root,
        )
        npz_mapping: dict[str, Path] = {}
        for dataset_name in args.datasets:
            dataset_docs = {
                trial_id: doc
                for trial_id, doc in docs.items()
                if doc.get("_dataset_id") == dataset_name
            }
            dataset_npz_mapping = discover_npz_files(
                args.npz_root / dataset_name, dataset_docs
            )
            missing_npz = sorted(set(dataset_docs).difference(dataset_npz_mapping))
            if missing_npz:
                message = (
                    f"{dataset_name}: {len(missing_npz)} JSON trial(s) have no "
                    f"matching NPZ, examples={missing_npz[:5]}"
                )
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"WARNING: {message}", file=sys.stderr)
            npz_mapping.update(dataset_npz_mapping)
    else:
        docs = load_json_directory(args.json_dir)
        npz_mapping = discover_npz_files(args.npz_dir, docs)
    audit_dataset(docs, npz_mapping, args.ablation_profile)
    if not npz_mapping:
        raise ValueError("No NPZ file matched a JSON trial_id")
    schema = FeatureSchema.discover(docs, args.text_dim, args.ablation_profile)
    sequences: dict[str, TrialSequence] = {}
    for trial_id, path in npz_mapping.items():
        try:
            sequences[trial_id] = build_trial_sequence(
                docs[trial_id],
                path,
                schema,
                args.ablation_profile,
                robot_atom_positive_weight=args.robot_atom_positive_weight,
                human_atom_positive_weight=args.human_atom_positive_weight,
                human_semantic_weight=args.human_semantic_weight,
                human_semantic_input_scale=args.human_semantic_input_scale,
                semantic_text_scale=args.semantic_text_scale,
                llm_absent_atom_weight=args.llm_absent_atom_weight,
                phase_absent_atom_weight=args.phase_absent_atom_weight,
            )
        except Exception as exc:
            if args.strict:
                raise
            print(f"WARNING: skipping {trial_id}: {exc}", file=sys.stderr)
    if not sequences:
        raise ValueError("No valid trial sequence remained after validation")

    # Split assignment is defined by DEFAULT_SUBJECT_TRIAL_GROUPS and does
    # not depend on Subject_info files.
    subject_info: dict[str, dict[str, Any]] = {}
    _SPLIT_CONTEXT.loso_test_subject = str(args.loso_test_subject or "")
    train_ids, val_ids, test_ids = split_trial_ids(
        list(sequences),
        args.seed,
        args.val_ratio,
        args.test_ratio,
        docs=docs,
        subject_info=subject_info,
        split_mode=args.split_mode,
    )
    write_split_manifest(
        args, train_ids, val_ids, test_ids, docs, subject_info
    )
    train = [sequences[item] for item in train_ids]
    val = [sequences[item] for item in val_ids]
    test = [sequences[item] for item in test_ids]
    normalizers = fit_normalizers(train)
    print("\n=== Prepared data ===")
    print(f"ablation profile: {args.ablation_profile}")
    if args.ablation_profile == "Z0":
        print(
            "Z0 pairing: robot/human numeric, semantic, quality and phase "
            "inputs are masked; shared task_goal query input is retained"
        )
    if args.ablation_profile == "Z4":
        print(
            "Z4 semantic fusion: evidence-backed human_semantics + task_goal "
            "enter the shared z_sem text input"
        )
    print(
        "task query: task_goal-only input -> shared semantic encoder -> "
        "query_z_sem alignment"
    )
    print(f"split trials: train={train_ids}, val={val_ids}, test={test_ids}")
    print(
        f"dimensions: skill_frame={train[0].skill_frames.shape[1]}, "
        f"quality_frame={train[0].quality_frames.shape[1]}, "
        f"task_target={train[0].task_targets.shape[1]}, "
        f"phase={train[0].phase_frames.shape[1]}, semantic_metrics={len(schema.sem_keys)}, "
        f"quality_metrics={len(schema.quality_keys)}, atoms={len(schema.atom_vocab)}, "
        f"quality_tags={len(schema.quality_tag_vocab)}"
    )
    return docs, schema, train, val, test, normalizers, subject_info


def train_model(args: argparse.Namespace) -> None:
    require_torch()
    seed_everything(args.seed)
    _, schema, train_seq, val_seq, test_seq, normalizers, _ = prepare_data(args)
    quality_tag_pos_weight = quality_tag_positive_weights(
        train_seq, len(schema.quality_tag_vocab)
    )
    atom_pos_weight = atom_positive_weights(
        train_seq,
        len(schema.atom_vocab),
        args.atom_positive_max_weight,
        args.single_group_atom_max_weight,
    )
    coverage_report = atom_coverage_report(train_seq, schema.atom_vocab)
    train_dataset = MotionWindowDataset(
        train_seq,
        schema,
        normalizers,
        window=args.window,
        stride=args.stride,
        max_windows=args.max_windows,
        quality_tag_pos_weight=quality_tag_pos_weight,
        atom_pos_weight=atom_pos_weight,
        semantic_purity_threshold=args.semantic_purity_threshold,
        semantic_window_min_weight=args.semantic_window_min_weight,
        human_metric_scale=args.human_metric_input_scale,
        force_segment_centers=args.force_segment_centers,
        phase_focused_pooling=args.phase_focused_pooling,
        skill_local_pool_frames=args.skill_local_pool_frames,
    )
    train_dataset.query_augmentation = float(getattr(args, "query_augmentation", 0.0))
    val_dataset = (
        MotionWindowDataset(
            val_seq,
            schema,
            normalizers,
            window=args.window,
            stride=args.stride,
            max_windows=args.max_windows,
            quality_tag_pos_weight=quality_tag_pos_weight,
            atom_pos_weight=atom_pos_weight,
            semantic_purity_threshold=args.semantic_purity_threshold,
            semantic_window_min_weight=args.semantic_window_min_weight,
            human_metric_scale=args.human_metric_input_scale,
            force_segment_centers=args.force_segment_centers,
            phase_focused_pooling=args.phase_focused_pooling,
            skill_local_pool_frames=args.skill_local_pool_frames,
        )
        if val_seq
        else None
    )
    test_dataset = (
        MotionWindowDataset(
            test_seq,
            schema,
            normalizers,
            window=args.window,
            stride=args.stride,
            max_windows=args.max_windows,
            quality_tag_pos_weight=quality_tag_pos_weight,
            atom_pos_weight=atom_pos_weight,
            semantic_purity_threshold=args.semantic_purity_threshold,
            semantic_window_min_weight=args.semantic_window_min_weight,
            human_metric_scale=args.human_metric_input_scale,
            force_segment_centers=args.force_segment_centers,
            phase_focused_pooling=args.phase_focused_pooling,
            skill_local_pool_frames=args.skill_local_pool_frames,
        )
        if test_seq
        else None
    )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset has no windows")
    print(
        f"windows: train={len(train_dataset)}, "
        f"val={len(val_dataset) if val_dataset else 0}, "
        f"test={len(test_dataset) if test_dataset else 0}"
    )

    sample = train_dataset[0]
    model = MotionLatentModel(
        skill_input_dim=int(sample["skill_seq"].shape[-1]),
        quality_input_dim=int(sample["quality_seq"].shape[-1]),
        phase_dim=int(sample["phase_seq"].shape[-1]),
        task_dim=int(sample["task_target"].shape[-1]),
        task_semantic_input_dim=int(sample["task_branch"].shape[-1]),
        robot_semantic_input_dim=int(sample["robot_branch"].shape[-1]),
        human_semantic_input_dim=int(sample["human_branch"].shape[-1]),
        semantic_metric_dim=len(schema.sem_keys),
        quality_target_dim=len(schema.quality_keys),
        atom_dim=len(schema.atom_vocab),
        quality_tag_dim=len(schema.quality_tag_vocab),
        latent_dim=args.latent_dim,
        quality_latent_dim=args.quality_latent_dim,
        channels=args.channels,
        dropout=args.dropout,
        local_feature_dim=args.local_feature_dim,
        dual_scale_skill=args.skill_local_pool_frames > 0,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled "
            "PyTorch build and verify torch.cuda.is_available()."
        )
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model.to(device)
    ema_model = copy.deepcopy(model).to(device)
    # Start EMA from the same parameters as the trainable model.  This avoids
    # evaluating a mixture of random initialization and early updates.
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=(
            WeightedRandomSampler(
                torch.as_tensor(
                    sampler_weights,
                    dtype=torch.double,
                ),
                num_samples=len(train_dataset),
                replacement=True,
            )
            if (sampler_weights := build_atom_sample_weights(
                train_dataset,
                atom_pos_weight,
                args.rare_sample_boost,
                args.max_sampler_weight,
                args.dataset_balance_boost,
                args.coverage_sample_boost,
                args.trial_balance_boost,
                args.phase_balance_boost,
                args.segment_balance_power,
            ))
            is not None
            else None
        ),
        shuffle=False
        if (
            args.rare_sample_boost > 0.0
            or args.dataset_balance_boost > 0.0
            or args.coverage_sample_boost > 0.0
            or args.trial_balance_boost > 0.0
            or args.phase_balance_boost > 0.0
            or args.segment_balance_power > 0.0
        )
        else True,
        num_workers=args.num_workers,
        drop_last=len(train_dataset) >= args.batch_size,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "atom_coverage_train.json").write_text(
        json.dumps(coverage_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "atom coverage: "
        + ", ".join(
            f"{atom}={item['n_subject_groups']}groups"
            for atom, item in coverage_report.items()
        )
    )
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    patience = 0
    best_path = args.output_dir / "best_motion_latent.pt"

    print(f"model parameters={sum(p.numel() for p in model.parameters()):,}, device={device}")
    for epoch in range(args.epochs):
        semantic_scale = semantic_loss_scale(epoch, args)
        semantic_enabled = (
            ablation_config(args.ablation_profile)["task_context"]
            and semantic_scale > 0.0
        )
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            training=True,
            semantic_enabled=semantic_enabled,
            semantic_scale=semantic_scale,
            ema_model=ema_model,
        )
        val_metrics = (
            run_epoch(
                ema_model if args.ema_decay > 0.0 else model,
                val_loader,
                optimizer=None,
                device=device,
                args=args,
                training=False,
                semantic_enabled=semantic_enabled,
                semantic_scale=semantic_scale,
            )
            if val_loader
            else train_metrics
        )
        # Warmup and full training optimize different objectives.  Do not
        # compare their losses when selecting the checkpoint, otherwise the
        # exported latents may come from before semantic alignment is enabled.
        score = val_metrics["loss"]
        if semantic_scale >= 1.0:
            score += args.checkpoint_semantic_f1_weight * (
                1.0 - val_metrics["atom_f1"]
            )
            score += args.checkpoint_rare_atom_f1_weight * (
                1.0 - val_metrics.get("rare_atom_f1", 0.0)
            )
            score += args.checkpoint_skill_semantic_f1_weight * (
                1.0 - val_metrics.get("skill_atom_f1", 0.0)
            )
            if (
                args.checkpoint_quality_f1_weight > 0.0
                and val_metrics.get("quality_tag_selection_valid", 0.0) > 0.0
            ):
                score += args.checkpoint_quality_f1_weight * (
                    1.0 - val_metrics["quality_tag_f1"]
                )
            norm_ratio = max(val_metrics.get("semantic_norm_ratio", 1.0), 1e-8)
            score += args.checkpoint_semantic_norm_weight * abs(
                math.log(norm_ratio)
            )
            if val_metrics.get("phase_retrieval_coverage", 0.0) > 0.0:
                retrieval_score = 0.5 * (
                    val_metrics.get("phase_retrieval_r1", 0.0)
                    + val_metrics.get("phase_retrieval_mrr", 0.0)
                )
                score += args.checkpoint_phase_retrieval_weight * (
                    1.0 - retrieval_score
                )
        record = {
            "epoch": epoch + 1,
            "semantic_enabled": semantic_enabled,
            "selection_score": float(score),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch + 1:04d} train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} recon={val_metrics['recon']:.5f} "
            f"smooth={val_metrics['smooth']:.5f} "
            f"quality={val_metrics['quality']:.5f} "
            f"quality_tags={val_metrics['quality_tags']:.5f} "
            f"decor={val_metrics['decorrelation']:.5f} "
            f"variance={val_metrics['variance']:.5f} "
            f"sem_cov={val_metrics['semantic_covariance']:.5f} "
            f"sem_decode={val_metrics['semantic_decode']:.5f} "
            f"sem_norm={val_metrics['semantic_norm']:.5f} "
            f"sem_ratio={val_metrics['semantic_norm_ratio']:.3f} "
            f"align={val_metrics['align']:.5f} "
            f"query_align={val_metrics['query_align']:.5f} "
            f"query_cons={val_metrics['query_consistency']:.5f} "
            f"seg_q_align={val_metrics['segment_query_align']:.5f} "
            f"seg_q_con={val_metrics['segment_query_contrastive']:.5f} "
            f"phase_con={val_metrics['phase_contrastive']:.5f} "
            f"seg_comp={val_metrics['segment_compactness']:.5f} "
            f"seg_boundary={val_metrics['segment_boundary']:.5f} "
            f"phase_r1={val_metrics.get('phase_retrieval_r1', 0.0):.3f} "
            f"phase_mrr={val_metrics.get('phase_retrieval_mrr', 0.0):.3f} "
            f"metric={val_metrics['metric']:.5f} "
            f"atoms={val_metrics['atoms']:.5f} "
            f"atom_f1={val_metrics['atom_f1']:.5f} "
            f"skill_atom_f1={val_metrics.get('skill_atom_f1', 0.0):.5f} "
            f"rare_atom_f1={val_metrics.get('rare_atom_f1', 0.0):.5f} "
            f"tag_f1={val_metrics['quality_tag_f1']:.5f} "
            f"select={score:.5f} "
            f"sem_scale={semantic_scale:.3f}"
        )
        # Compare checkpoints only after the semantic objective has reached
        # its full weight; warmup/ramp losses are not directly comparable.
        checkpoint_ready = semantic_scale >= 1.0
        # A short smoke test (or an interrupted/custom run) may finish before
        # the semantic ramp reaches 1.0.  Keep normal checkpoint selection
        # unchanged, but allow the final model to be exported as a fallback so
        # the run does not fail with ``No checkpoint was saved``.
        fallback_checkpoint = (
            epoch == args.epochs - 1
            and not checkpoint_ready
            and not best_path.exists()
        )
        if checkpoint_ready and val_loader:
            # The learning-rate scheduler follows the raw validation loss;
            # checkpoint selection may additionally include semantic F1.
            scheduler.step(val_metrics["loss"])
        if (checkpoint_ready and score < best_score - args.min_delta) or fallback_checkpoint:
            # Keep a finite selection score in custom short runs; the fallback
            # checkpoint is only a smoke-test safety net.
            best_score = float(score) if math.isfinite(float(score)) else best_score
            patience = 0
            selected_model = ema_model if args.ema_decay > 0.0 else model
            torch.save(
                {
                    "model_state": selected_model.state_dict(),
                    "raw_model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "checkpoint_fallback": bool(fallback_checkpoint),
                    "best_val_loss": val_metrics["loss"],
                    "selection_score": best_score,
                    "val_atom_f1": val_metrics["atom_f1"],
                    "val_rare_atom_f1": val_metrics.get("rare_atom_f1", 0.0),
                    "val_skill_atom_f1": val_metrics.get("skill_atom_f1", 0.0),
                    "val_quality_tag_f1": val_metrics["quality_tag_f1"],
                    "quality_tag_selection_valid": bool(val_metrics.get("quality_tag_selection_valid", 0.0) > 0.0),
                    "quality_tag_evaluable_axes": val_metrics.get("quality_tag_evaluable_axes", 0.0),
                    "quality_tag_positive_support": val_metrics.get("quality_tag_positive_support", 0.0),
                    "quality_tag_negative_support": val_metrics.get("quality_tag_negative_support", 0.0),
                    "val_semantic_norm_loss": val_metrics["semantic_norm"],
                    "val_semantic_norm_ratio": val_metrics["semantic_norm_ratio"],
                    "val_query_align_loss": val_metrics["query_align"],
                    "val_query_consistency_loss": val_metrics[
                        "query_consistency"
                    ],
                    "val_segment_query_align_loss": val_metrics[
                        "segment_query_align"
                    ],
                    "val_segment_query_contrastive_loss": val_metrics[
                        "segment_query_contrastive"
                    ],
                    "val_phase_contrastive_loss": val_metrics[
                        "phase_contrastive"
                    ],
                    "val_phase_retrieval_r1": val_metrics.get(
                        "phase_retrieval_r1", 0.0
                    ),
                    "val_phase_retrieval_mrr": val_metrics.get(
                        "phase_retrieval_mrr", 0.0
                    ),
                    "val_phase_retrieval_coverage": val_metrics.get(
                        "phase_retrieval_coverage", 0.0
                    ),
                    "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                    "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                    "semantic_fusion_version": SEMANTIC_FUSION_VERSION,
                    "quality_input_version": QUALITY_INPUT_VERSION,
                    "quality_conditioning_version": QUALITY_CONDITIONING_VERSION,
                    "schema": schema.to_jsonable(),
                    "normalizers": normalizers.to_jsonable(),
                    "args": vars(args),
                },
                best_path,
            )
        elif checkpoint_ready:
            patience += 1
            if epoch >= args.warmup_epochs and patience >= args.early_stop_patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    if not best_path.exists():
        # Defensive fallback for short runs.  Full training (the supplied
        # protocol uses 180 epochs, 20 warmup + 30 semantic ramp) still uses
        # only full-semantic epochs for model selection.  This branch only
        # prevents a custom smoke test from losing all exports.
        if not history:
            raise RuntimeError(
                "No checkpoint was saved because training produced no epochs. "
                "Set --epochs to a positive integer."
            )
        last = history[-1]
        selected_model = ema_model if args.ema_decay > 0.0 else model
        fallback_score = float(last.get("selection_score", last["val"]["loss"]))
        torch.save(
            {
                "model_state": selected_model.state_dict(),
                "raw_model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": int(last["epoch"]),
                "checkpoint_fallback": True,
                "best_val_loss": float(last["val"]["loss"]),
                "selection_score": fallback_score,
                "val_atom_f1": float(last["val"].get("atom_f1", 0.0)),
                "val_rare_atom_f1": float(last["val"].get("rare_atom_f1", 0.0)),
                "val_skill_atom_f1": float(last["val"].get("skill_atom_f1", 0.0)),
                "val_quality_tag_f1": float(last["val"].get("quality_tag_f1", 0.0)),
                "quality_tag_selection_valid": bool(last["val"].get("quality_tag_selection_valid", 0.0) > 0.0),
                "quality_tag_evaluable_axes": float(last["val"].get("quality_tag_evaluable_axes", 0.0)),
                "quality_tag_positive_support": float(last["val"].get("quality_tag_positive_support", 0.0)),
                "quality_tag_negative_support": float(last["val"].get("quality_tag_negative_support", 0.0)),
                "val_semantic_norm_loss": float(last["val"].get("semantic_norm", 0.0)),
                "val_semantic_norm_ratio": float(last["val"].get("semantic_norm_ratio", 1.0)),
                "val_query_align_loss": float(last["val"].get("query_align", 0.0)),
                "val_query_consistency_loss": float(last["val"].get("query_consistency", 0.0)),
                "val_segment_query_align_loss": float(last["val"].get("segment_query_align", 0.0)),
                "val_segment_query_contrastive_loss": float(last["val"].get("segment_query_contrastive", 0.0)),
                "val_phase_contrastive_loss": float(last["val"].get("phase_contrastive", 0.0)),
                "val_phase_retrieval_r1": float(last["val"].get("phase_retrieval_r1", 0.0)),
                "val_phase_retrieval_mrr": float(last["val"].get("phase_retrieval_mrr", 0.0)),
                "val_phase_retrieval_coverage": float(last["val"].get("phase_retrieval_coverage", 0.0)),
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                "semantic_fusion_version": SEMANTIC_FUSION_VERSION,
                "quality_input_version": QUALITY_INPUT_VERSION,
                "quality_conditioning_version": QUALITY_CONDITIONING_VERSION,
                "schema": schema.to_jsonable(),
                "normalizers": normalizers.to_jsonable(),
                "args": vars(args),
            },
            best_path,
        )
        print(
            "WARNING: semantic ramp did not complete before the run ended; "
            "saved the final model as a fallback checkpoint. For a fully "
            "semantic-aligned model, use --epochs > --warmup-epochs + "
            "--semantic-ramp-epochs."
        )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "schema.json").write_text(
        json.dumps(schema.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    export_latents(
        model,
        train_dataset,
        args.output_dir / "train_latents.npz",
        device,
        args.batch_size,
    )
    if val_dataset:
        export_latents(
            model,
            val_dataset,
            args.output_dir / "val_latents.npz",
            device,
            args.batch_size,
        )
    if test_dataset:
        export_latents(
            model,
            test_dataset,
            args.output_dir / "test_latents.npz",
            device,
            args.batch_size,
        )
    print(f"Saved best checkpoint and latent exports to {args.output_dir}")


def write_split_manifest(
    args: argparse.Namespace,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    docs: Mapping[str, Mapping[str, Any]],
    subject_info: Mapping[str, Mapping[str, Any]],
) -> None:
    """Persist the exact split and the source of subject assignments."""
    all_trial_ids = list(dict.fromkeys([*train_ids, *val_ids, *test_ids]))
    report_group_size = 10
    groups = subject_groups(
        all_trial_ids,
        docs,
        report_group_size,
        subject_info,
    )
    split_subjects = {
        "train": sorted(
            {
                subject_group_key(
                    item, docs.get(item), report_group_size, subject_info
                )
                for item in train_ids
            }
        ),
        "val": sorted(
            {
                subject_group_key(
                    item, docs.get(item), report_group_size, subject_info
                )
                for item in val_ids
            }
        ),
        "test": sorted(
            {
                subject_group_key(
                    item, docs.get(item), report_group_size, subject_info
                )
                for item in test_ids
            }
        ),
    }
    phase_counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for trial_id in all_trial_ids:
        doc = docs.get(trial_id, {})
        dataset_id = str(doc.get("_dataset_id") or trial_id.split("_", 1)[0])
        for segment in doc.get("segments", []):
            phase_counts[dataset_id][get_phase_label(segment)] += 1
    split_dataset_counts: dict[str, dict[str, int]] = {}
    selected_datasets = (
        args.datasets if not args.legacy_single_dataset else [args.json_dir.name]
    )
    for dataset in selected_datasets:
        split_dataset_counts[dataset] = {
            split_name: sum(
                1
                for trial_id in split_ids
                if str(trial_id).startswith(f"{dataset}_")
            )
            for split_name, split_ids in (
                ("train", train_ids),
                ("val", val_ids),
                ("test", test_ids),
            )
        }
    manifest = {
        "split_mode": args.split_mode,
        "split_strategy": (
            (
                "leave-one-subject-out: one complete subject is test; "
                "remaining subjects are split at trial level into train/val"
                if args.split_mode == "loso"
                else "fixed unseen-subject test; stratified trial-level validation "
                     "within the non-test subject pool"
            )
            if args.split_mode in {"subject", "loso"}
            else "random trial split"
        ),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "loso_test_subject": args.loso_test_subject,
        "subject_groups": groups,
        "phase_counts": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(phase_counts.items())
        },
        "dataset_counts": {
            dataset: sum(
                1
                for trial_id in all_trial_ids
                if str(trial_id).startswith(f"{dataset}_")
            )
            for dataset in selected_datasets
        },
        "split_dataset_counts": split_dataset_counts,
        "subject_assignment": "fixed_dataset_trial_ranges",
        "subject_trial_groups": {
            dataset: {
                subject: sorted(trials)
                for subject, trials in DEFAULT_SUBJECT_TRIAL_GROUPS.get(
                    dataset, {}
                ).items()
            }
            for dataset in selected_datasets
        },
        "subject_split_policy": DEFAULT_SUBJECT_SPLIT,
        "splits": {
            "train": {"subjects": split_subjects["train"], "trials": list(train_ids)},
            "val": {"subjects": split_subjects["val"], "trials": list(val_ids)},
            "test": {"subjects": split_subjects["test"], "trials": list(test_ids)},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def audit_and_prepare(args: argparse.Namespace) -> None:
    docs, schema, train, val, test, normalizers, _ = prepare_data(args)
    quality_tag_pos_weight = quality_tag_positive_weights(
        train, len(schema.quality_tag_vocab)
    )
    atom_pos_weight = atom_positive_weights(
        train,
        len(schema.atom_vocab),
        args.atom_positive_max_weight,
    )
    datasets = {
        "train": MotionWindowDataset(
            train,
            schema,
            normalizers,
            window=args.window,
            stride=args.stride,
            max_windows=args.max_windows,
            quality_tag_pos_weight=quality_tag_pos_weight,
            atom_pos_weight=atom_pos_weight,
            semantic_purity_threshold=args.semantic_purity_threshold,
            semantic_window_min_weight=args.semantic_window_min_weight,
            human_metric_scale=args.human_metric_input_scale,
            force_segment_centers=args.force_segment_centers,
            phase_focused_pooling=args.phase_focused_pooling,
        ),
        "val": MotionWindowDataset(
            val,
            schema,
            normalizers,
            window=args.window,
            stride=args.stride,
            max_windows=args.max_windows,
            quality_tag_pos_weight=quality_tag_pos_weight,
            atom_pos_weight=atom_pos_weight,
            semantic_purity_threshold=args.semantic_purity_threshold,
            semantic_window_min_weight=args.semantic_window_min_weight,
            human_metric_scale=args.human_metric_input_scale,
            force_segment_centers=args.force_segment_centers,
            phase_focused_pooling=args.phase_focused_pooling,
        ) if val else None,
        "test": MotionWindowDataset(
            test,
            schema,
            normalizers,
            window=args.window,
            stride=args.stride,
            max_windows=args.max_windows,
            quality_tag_pos_weight=quality_tag_pos_weight,
            atom_pos_weight=atom_pos_weight,
            semantic_purity_threshold=args.semantic_purity_threshold,
            semantic_window_min_weight=args.semantic_window_min_weight,
            human_metric_scale=args.human_metric_input_scale,
            force_segment_centers=args.force_segment_centers,
            phase_focused_pooling=args.phase_focused_pooling,
        ) if test else None,
    }
    print("\n=== Window smoke test (NumPy) ===")
    for name, dataset in datasets.items():
        if dataset is None:
            continue
        print(f"{name}: windows={len(dataset)}")
        if len(dataset):
            sample = dataset[0]
            shapes = {
                key: tuple(value.shape)
                for key, value in sample.items()
                if isinstance(value, np.ndarray)
            }
            print(f"  first={dataset.metadata(0)}")
            print(f"  tensor_shapes={shapes}")
            for key, value in sample.items():
                if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
                    raise ValueError(f"Non-finite value in {name}.{key}")
    print("Audit and NumPy data-pipeline smoke test passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a semantics-aligned, feasibility-disentangled motion latent."
    )
    parser.add_argument(
        "--mode",
        choices=("audit", "train"),
        default="audit",
    )
    parser.add_argument(
        "--ablation-profile",
        choices=tuple(ABLATION_PROFILES),
        default="Z4",
        help="Input ablation profile. Z4 is the full model.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\LLM_annotation\CrouchedTraverse"),
        help="Legacy single-dataset JSON directory.",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\Retargeted\Data\CrouchedTraverse"),
        help="Legacy single-dataset redirected Dynamics NPZ directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Train exactly one action/dataset. This is the recommended mode for "
            "producing action-specific Z0-Z4 latents; the name must exist under "
            "--json-root and --npz-root."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Dataset directory names. By default exactly one action is required; "
            "multiple actions are rejected unless --allow-mixed-datasets is set."
        ),
    )
    parser.add_argument(
        "--allow-mixed-datasets",
        action="store_true",
        help=(
            "Explicitly allow the legacy behavior of merging multiple actions "
            "into one training run. Do not use this when generating separate "
            "action-specific z_latents."
        ),
    )
    parser.add_argument(
        "--auto-discover-datasets",
        action="store_true",
        help=(
            "Use every action directory present in both --json-root and "
            "--npz-root. This is useful after adding new motion datasets."
        ),
    )
    parser.add_argument(
        "--legacy-single-dataset",
        action="store_true",
        help="Use the legacy single --json-dir/--npz-dir pair instead of --datasets.",
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\LLM_annotation"),
        help="Root containing one subdirectory per dataset.",
    )
    parser.add_argument(
        "--npz-root",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\Retargeted\Data"),
        help="Root containing one subdirectory per dataset.",
    )
    parser.add_argument(
        "--retargeted-json-root",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\Retargeted\Dynamics"),
        help="Root containing per-dataset retargeted dynamics/feasibility JSON.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("subject", "random_trial", "loso", "file"),
        default="subject",
        help=("subject holds out the configured test subject; loso holds out exactly "
              "one subject selected by --loso-test-subject; random_trial preserves "
              "the old behavior; file reads an explicit split (HNU_LATENT_SPLIT_JSON)."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\Latent\Z4"),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--loso-test-subject",
        type=str,
        default=None,
        help=("Subject name to leave out for --split-mode loso, e.g. "
              "subject_1, subject_2, subject_3 or subject_4."),
    )
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument(
        "--force-segment-centers",
        action="store_true",
        help=(
            "Keep one explicitly assigned window for every annotated segment, "
            "including phases shorter than the context window."
        ),
    )
    parser.add_argument(
        "--phase-focused-pooling",
        action="store_true",
        help=(
            "Pool z_skill only over frames belonging to the assigned segment "
            "while retaining the full temporal window as encoder context."
        ),
    )
    parser.add_argument(
        "--skill-local-pool-frames",
        type=int,
        default=0,
        help=(
            "If positive, fuse phase-wide z_skill pooling with a centered "
            "local crop of this many frames. Zero preserves the legacy encoder."
        ),
    )
    parser.add_argument(
        "--semantic-purity-threshold",
        type=float,
        default=0.55,
        help=(
            "Window coverage ratio at which segment-level semantic supervision "
            "reaches full weight."
        ),
    )
    parser.add_argument(
        "--semantic-window-min-weight",
        type=float,
        default=0.15,
        help=(
            "Minimum semantic/quality supervision weight for mixed windows; "
            "reconstruction remains unchanged."
        ),
    )
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--text-encoder", choices=("hash", "minilm"), default="hash",
                        help="v4: text front-end; minilm = frozen multilingual sentence encoder (use --text-dim 384)")
    parser.add_argument("--lambda-query-task", type=float, default=0.0,
                        help="v4b: supervised contrastive routing loss between task-only query z_sem and z_skill (same action = positive)")
    parser.add_argument("--query-task-stopgrad", type=int, default=0,
                        help="v4d: 1 = stop-gradient on z_skill in the routing loss (z_skill is the anchor)")
    parser.add_argument("--query-augmentation", type=float, default=0.0,
                        help="v4: probability of replacing the task-only query text by a profile-owned description")
    parser.add_argument("--max-windows", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--semantic-ramp-epochs",
        type=int,
        default=20,
        help="Epochs used to ramp semantic-loss weights from 0 to full strength.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument(
        "--quality-latent-dim",
        type=int,
        default=12,
        help=(
            "Capacity for feasibility/quality representation; 12 preserves "
            "more variation across the three mixed action datasets."
        ),
    )
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument(
        "--local-feature-dim",
        type=int,
        default=16,
        help=(
            "Low-dimensional local temporal feature used only by the motion "
            "task decoder; exported latent dimensions stay unchanged."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "Exponential moving-average decay for validation/checkpointing. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Validation epochs without improvement before reducing learning rate.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--checkpoint-semantic-f1-weight",
        type=float,
        default=0.0,
        help="Extra checkpoint-selection weight for validation atom macro F1.",
    )
    parser.add_argument(
        "--checkpoint-rare-atom-f1-weight",
        type=float,
        default=0.0,
        help="Extra checkpoint-selection weight for validation rare-atom macro F1.",
    )
    parser.add_argument(
        "--checkpoint-skill-semantic-f1-weight",
        type=float,
        default=0.0,
        help="Checkpoint-selection weight for semantic atoms decoded from z_skill.",
    )
    parser.add_argument(
        "--checkpoint-quality-f1-weight",
        type=float,
        default=0.0,
        help="Extra checkpoint-selection weight for validation feasibility-tag macro F1.",
    )
    parser.add_argument(
        "--checkpoint-semantic-norm-weight",
        type=float,
        default=0.0,
        help=(
            "Extra checkpoint-selection weight for the absolute log scale "
            "ratio between z_sem and z_skill."
        ),
    )
    parser.add_argument(
        "--checkpoint-phase-retrieval-weight",
        type=float,
        default=0.0,
        help=(
            "Checkpoint-selection weight for cross-trial validation phase "
            "retrieval. The held-out test subject is never consulted."
        ),
    )

    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-semantic-decode", type=float, default=0.20)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-align", type=float, default=0.20)
    parser.add_argument(
        "--lambda-query-align",
        type=float,
        default=0.0,
        help=(
            "Weak task_goal-only query alignment to stop-gradient z_skill. "
            "Profile defaults enable it for the shared command interface."
        ),
    )
    parser.add_argument(
        "--lambda-query-consistency",
        type=float,
        default=0.0,
        help=(
            "Weak consistency loss from the task_goal-only query latent to "
            "the stop-gradient full z_sem latent."
        ),
    )
    parser.add_argument(
        "--lambda-deployment-consistency",
        type=float,
        default=0.0,
        help="Symmetric consistency between full fusion and task-only query.",
    )
    parser.add_argument(
        "--lambda-segment-query-align",
        type=float,
        default=0.12,
        help="Alignment of the segment query latent to stop-gradient z_skill.",
    )
    parser.add_argument(
        "--lambda-segment-query-contrastive",
        type=float,
        default=0.10,
        help="In-batch contrastive loss for segment-aware queries.",
    )
    parser.add_argument(
        "--segment-query-align-margin",
        type=float,
        default=0.75,
        help="Cosine target for segment query versus z_skill.",
    )
    parser.add_argument(
        "--segment-query-temperature",
        type=float,
        default=0.10,
        help="Temperature for segment-query contrastive training.",
    )
    parser.add_argument(
        "--lambda-phase-contrastive",
        type=float,
        default=0.0,
        help=(
            "Supervised contrastive weight that aligns the same action/phase "
            "across different trial segments in z_skill."
        ),
    )
    parser.add_argument(
        "--phase-contrastive-temperature",
        type=float,
        default=0.10,
        help="Temperature for cross-trial phase contrastive training.",
    )
    parser.add_argument(
        "--lambda-segment-compactness",
        type=float,
        default=0.0,
        help="Weight for within-segment window-to-prototype consistency.",
    )
    parser.add_argument(
        "--lambda-segment-boundary",
        type=float,
        default=0.0,
        help="Weight for separating temporally adjacent segment prototypes.",
    )
    parser.add_argument(
        "--segment-boundary-max-cosine",
        type=float,
        default=0.70,
        help="Maximum unpenalized cosine for adjacent segment prototypes.",
    )
    parser.add_argument("--lambda-metric", type=float, default=0.50)
    parser.add_argument("--lambda-atoms", type=float, default=0.20)
    parser.add_argument("--lambda-quality", type=float, default=0.50)
    parser.add_argument(
        "--lambda-quality-tag-ranking",
        type=float,
        default=0.03,
        help="Ranking loss that separates positive and negative feasibility tags.",
    )
    parser.add_argument(
        "--lambda-atom-ranking",
        type=float,
        default=0.05,
        help="Ranking loss that separates positive and negative semantic atoms.",
    )
    parser.add_argument("--lambda-decorrelation", type=float, default=0.05)
    parser.add_argument("--lambda-variance", type=float, default=0.05)
    parser.add_argument(
        "--lambda-semantic-covariance",
        type=float,
        default=0.02,
        help="Penalty on off-diagonal covariance inside z_sem.",
    )
    parser.add_argument(
        "--lambda-skill-covariance",
        type=float,
        default=0.03,
        help="Penalty preventing correlated/collapsed dimensions in z_skill.",
    )
    parser.add_argument(
        "--lambda-quality-covariance",
        type=float,
        default=0.02,
        help="Penalty preventing correlated/collapsed dimensions in z_quality.",
    )
    parser.add_argument(
        "--lambda-skill-metric",
        type=float,
        default=0.0,
        help="Low-weight privileged numeric supervision transferred into z_skill.",
    )
    parser.add_argument("--lambda-future", type=float, default=0.0,
                        help="v5: z_skill predicts the robot's future joints / feet / pelvis (0.25-1 s; HNU_CTRL_AUX_DIR).")
    parser.add_argument("--lambda-human-ctrl", type=float, default=0.0,
                        help="v5: z_skill predicts human contact timing and GRF (only profiles with human numeric data).")
    parser.add_argument(
        "--lambda-skill-atoms",
        type=float,
        default=0.0,
        help="Low-weight privileged semantic-atom supervision transferred into z_skill.",
    )
    parser.add_argument("--lambda-human-skill-atoms", type=float, default=0.0,
                        help="v4: human-only atom labels predicted from z_skill (separate head, no robot override).")
    parser.add_argument("--lambda-human-sem-atoms", type=float, default=0.0,
                        help="v4: human-only atom labels predicted from z_sem.")
    parser.add_argument(
        "--lambda-semantic-norm",
        type=float,
        default=0.0,
        help=(
            "Loss matching z_sem sample norms to stop-gradient z_skill "
            "sample norms for task-context profiles."
        ),
    )
    parser.add_argument(
        "--query-align-margin",
        type=float,
        default=0.65,
        help=(
            "Cosine target for task_goal-only query_z_sem versus z_skill. "
            "It is lower than the full-semantic alignment target because a "
            "single task command covers multiple motion segments."
        ),
    )
    parser.add_argument(
        "--semantic-align-margin",
        type=float,
        default=0.85,
        help=(
            "Cosine similarity target for z_skill/z_sem. Alignment is "
            "penalized only below this margin."
        ),
    )
    parser.add_argument("--atom-negative-weight", type=float, default=0.20)
    parser.add_argument("--atom-positive-max-weight", type=float, default=6.0)
    parser.add_argument(
        "--single-group-atom-max-weight",
        type=float,
        default=2.0,
        help=(
            "Maximum positive loss weight for atoms observed in only one "
            "dataset-local subject group."
        ),
    )
    parser.add_argument(
        "--atom-label-smoothing",
        type=float,
        default=0.0,
        help="Binary-label smoothing used only by the atom loss.",
    )
    parser.add_argument(
        "--quality-tag-label-smoothing",
        type=float,
        default=0.0,
        help="Deprecated for quality v2; retained for CLI compatibility.",
    )
    parser.add_argument("--quality-label-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--quality-focal-gamma", type=float, default=2.0)
    parser.add_argument("--quality-focal-alpha", type=float, default=0.75)
    parser.add_argument("--quality-negative-weight", type=float, default=0.5,
                        help="Multiplier for known negative quality-v2 labels.")
    parser.add_argument(
        "--atom-focal-gamma",
        type=float,
        default=0.0,
        help="Focal exponent for atom BCE; 0 disables focal modulation.",
    )
    parser.add_argument(
        "--atom-ranking-margin",
        type=float,
        default=0.50,
        help="Logit margin used by the sparse semantic-atom ranking loss.",
    )
    parser.add_argument(
        "--quality-tag-ranking-margin",
        type=float,
        default=0.50,
        help="Logit margin used by the feasibility-tag ranking loss.",
    )
    parser.add_argument(
        "--rare-atom-positive-threshold",
        type=float,
        default=128.0,
        help=(
            "Validation labels with at most this many positive windows are "
            "included in rare_atom_f1."
        ),
    )
    parser.add_argument(
        "--rare-sample-boost",
        type=float,
        default=0.0,
        help="Controlled rare-atom window oversampling strength; 0 disables it.",
    )
    parser.add_argument(
        "--dataset-balance-boost",
        type=float,
        default=0.0,
        help=(
            "Mild inverse-sqrt balancing across action directories in the "
            "training sampler; 0 disables it."
        ),
    )
    parser.add_argument("--max-sampler-weight", type=float, default=3.0)
    parser.add_argument(
        "--coverage-sample-boost",
        type=float,
        default=0.0,
        help=(
            "Additional sampling for atoms underrepresented across subject "
            "groups; single-group atoms are excluded."
        ),
    )
    parser.add_argument(
        "--trial-balance-boost",
        type=float,
        default=0.0,
        help=(
            "Inverse-sqrt balancing across trials so long trials do not "
            "dominate the shared encoder."
        ),
    )
    parser.add_argument(
        "--phase-balance-boost",
        type=float,
        default=0.0,
        help=(
            "Inverse-sqrt sampling boost for short/rare gait phases within "
            "each action."
        ),
    )
    parser.add_argument(
        "--segment-balance-power",
        type=float,
        default=0.0,
        help=(
            "Power-law balancing across annotated segments: 0 keeps the "
            "window distribution and 1 gives equal expected segment exposure."
        ),
    )
    parser.add_argument(
        "--semantic-text-dropout",
        type=float,
        default=0.0,
        help=(
            "Training-only dropout applied to the text slice of semantic_input; "
            "does not alter task/segment query construction."
        ),
    )
    parser.add_argument(
        "--semantic-text-scale",
        type=float,
        default=DEFAULT_SEMANTIC_TEXT_SCALE,
        help="Scale of evidence-grounded human text in the Z4 semantic input.",
    )
    parser.add_argument(
        "--human-semantic-weight",
        type=float,
        default=DEFAULT_HUMAN_SEMANTIC_WEIGHT,
        help="Confidence multiplier for human semantic atom supervision.",
    )
    parser.add_argument(
        "--human-semantic-input-scale",
        type=float,
        default=DEFAULT_HUMAN_SEMANTIC_INPUT_SCALE,
        help="Additional scale for human text in the Z4 semantic input.",
    )
    parser.add_argument(
        "--human-metric-input-scale",
        type=float,
        default=1.0,
        help="Scale for human numeric teacher inputs in Z3/Z4 (separate from text).",
    )
    parser.add_argument(
        "--motion-noise-std",
        type=float,
        default=0.0,
        help="Training-only Gaussian noise in standardized motion streams.",
    )
    parser.add_argument(
        "--semantic-input-dropout",
        type=float,
        default=0.0,
        help=(
            "Training-only feature dropout on the semantic encoder input. "
            "Useful when semantic JSON labels are sparse or action-specific."
        ),
    )
    parser.add_argument(
        "--semantic-metric-input-dropout",
        type=float,
        default=0.0,
        help=(
            "Training-only dropout applied jointly to semantic metric values "
            "and their masks before the semantic metric head."
        ),
    )
    parser.add_argument(
        "--robot-atom-positive-weight",
        type=float,
        default=DEFAULT_ROBOT_ATOM_POSITIVE_WEIGHT,
        help="Base confidence for positive robot LLM atom labels.",
    )
    parser.add_argument(
        "--human-atom-positive-weight",
        type=float,
        default=DEFAULT_HUMAN_ATOM_POSITIVE_WEIGHT,
        help="Base confidence for positive human atom labels.",
    )
    parser.add_argument(
        "--llm-absent-atom-weight",
        type=float,
        default=DEFAULT_LLM_ABSENT_ATOM_WEIGHT,
        help="Confidence for LLM-inferred known_absent_atoms.",
    )
    parser.add_argument(
        "--phase-absent-atom-weight",
        type=float,
        default=DEFAULT_PHASE_ABSENT_ATOM_WEIGHT,
        help="Confidence for deterministic phase/contact absent atoms.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    config = ablation_config(args.ablation_profile)
    for name in (
        "robot_atom_positive_weight",
        "human_atom_positive_weight",
        "llm_absent_atom_weight",
        "phase_absent_atom_weight",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.ablation_profile != "Z4" and args.output_dir == Path(
        r"C:\Users\Administrator\Desktop\Data-HNU\Latent\Z4"
    ):
        args.output_dir = Path(
            rf"C:\Users\Administrator\Desktop\Data-HNU\Latent\{args.ablation_profile}"
        )
    if args.dataset is not None and args.datasets is not None:
        raise ValueError("Use either --dataset or --datasets, not both")
    if args.dataset is not None:
        args.datasets = [str(args.dataset)]
    elif args.datasets is None:
        if args.auto_discover_datasets:
            if config["requires_json"]:
                args.datasets = discover_dataset_names(args.json_root, args.npz_root)
            else:
                args.datasets = sorted(
                    item.name
                    for item in args.npz_root.iterdir()
                    if item.is_dir()
                    and not item.name.startswith(".")
                    and any(item.glob("**/*.npz"))
                )
        else:
            # Safe default: one action per run. The old three-action default is
            # intentionally no longer used because it mixes unrelated motions.
            args.datasets = ["CrouchedTraverse"]
    if not args.datasets:
        raise ValueError("At least one dataset must be supplied")
    if len(args.datasets) > 1 and not args.allow_mixed_datasets:
        raise ValueError(
            "Training one latent space from multiple actions is disabled by "
            "default because it mixes action-specific JSON semantics. Supply "
            "exactly one --dataset (recommended), or pass --allow-mixed-datasets only "
            "for the legacy mixed mode."
        )
    invalid_names = [
        dataset
        for dataset in args.datasets
        if not dataset or Path(dataset).name != dataset
    ]
    if invalid_names:
        raise ValueError(
            "Dataset names must be simple directory names: "
            + ", ".join(invalid_names)
        )
    if args.legacy_single_dataset:
        if config["requires_json"] and not args.json_dir.is_dir():
            raise NotADirectoryError(args.json_dir)
        if not args.npz_dir.is_dir():
            raise NotADirectoryError(args.npz_dir)
    else:
        if config["requires_json"] and not args.json_root.is_dir():
            raise NotADirectoryError(args.json_root)
        if not args.npz_root.is_dir():
            raise NotADirectoryError(args.npz_root)
        missing = [
            dataset
            for dataset in args.datasets
            if not (args.npz_root / dataset).is_dir()
            or (
                config["requires_json"]
                and not (args.json_root / dataset).is_dir()
            )
        ]
        if missing:
            raise NotADirectoryError(
                f"Missing JSON or NPZ dataset directories: {', '.join(missing)}"
            )
    if (
        config["requires_json"]
        and not args.legacy_single_dataset
        and args.retargeted_json_root is not None
        and not args.retargeted_json_root.is_dir()
    ):
        print(
            f"WARNING: optional retargeted dynamics JSON root not found: "
            f"{args.retargeted_json_root}. Continuing with annotation JSON + NPZ.",
            file=sys.stderr,
        )
    if args.split_mode in {"subject", "loso"}:
        unmapped = sorted(
            set(args.datasets).difference(SUBJECT_CONFIGURED_DATASETS)
        )
        if unmapped:
            raise ValueError(
                "Subject-aware splitting is not configured for these dataset(s): "
                + ", ".join(unmapped)
                + ". Use --split-mode random_trial for a new action until its "
                "trial-to-subject mapping is added to "
                "DEFAULT_SUBJECT_TRIAL_GROUPS in motion_latent_z.py."
            )
    if args.split_mode == "loso" and not args.loso_test_subject:
        raise ValueError("--loso-test-subject is required when --split-mode loso")
    if args.split_mode == "loso" and args.loso_test_subject not in {"subject_1", "subject_2", "subject_3", "subject_4"}:
        raise ValueError("--loso-test-subject must be subject_1, subject_2, subject_3 or subject_4")

    if args.window < 8 or args.stride < 1:
        raise ValueError("--window must be >= 8 and --stride >= 1")
    if not 0 <= args.skill_local_pool_frames <= args.window:
        raise ValueError("--skill-local-pool-frames must be in [0, --window]")
    if not 0.0 < args.semantic_purity_threshold <= 1.0:
        raise ValueError("--semantic-purity-threshold must be in (0, 1]")
    if not 0.0 <= args.semantic_window_min_weight <= 1.0:
        raise ValueError("--semantic-window-min-weight must be in [0, 1]")
    if args.local_feature_dim < 1:
        raise ValueError("--local-feature-dim must be >= 1")
    if args.quality_label_version != "v2":
        raise ValueError("Only --quality-label-version v2 is supported by this file")
    if args.quality_focal_gamma < 0.0:
        raise ValueError("--quality-focal-gamma must be non-negative")
    if not 0.0 <= args.quality_focal_alpha <= 1.0:
        raise ValueError("--quality-focal-alpha must be in [0, 1]")
    if args.quality_negative_weight < 0.0:
        raise ValueError("--quality-negative-weight must be non-negative")
    if args.latent_dim < 2 or args.quality_latent_dim < 1:
        raise ValueError("Latent dimensions are too small")
    if (
        args.checkpoint_semantic_f1_weight < 0.0
        or args.checkpoint_rare_atom_f1_weight < 0.0
        or args.checkpoint_skill_semantic_f1_weight < 0.0
        or args.checkpoint_quality_f1_weight < 0.0
        or args.checkpoint_semantic_norm_weight < 0.0
        or args.checkpoint_phase_retrieval_weight < 0.0
    ):
        raise ValueError("Checkpoint selection weights must be non-negative")
    for name in (
        "lambda_semantic_norm",
        "lambda_query_align",
        "lambda_query_consistency",
        "lambda_deployment_consistency",
        "lambda_segment_query_align",
        "lambda_segment_query_contrastive",
        "lambda_phase_contrastive",
        "lambda_segment_compactness",
        "lambda_segment_boundary",
        "lambda_skill_covariance",
        "lambda_quality_covariance",
        "lambda_skill_metric",
        "lambda_skill_atoms",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.warmup_epochs < 0 or args.semantic_ramp_epochs < 0:
        raise ValueError("--warmup-epochs and --semantic-ramp-epochs must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    for name in (
        "atom_label_smoothing",
        "quality_tag_label_smoothing",
        "coverage_sample_boost",
        "trial_balance_boost",
        "phase_balance_boost",
        "rare_atom_positive_threshold",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.0 <= args.segment_balance_power <= 1.0:
        raise ValueError("--segment-balance-power must be in [0, 1]")
    if args.single_group_atom_max_weight < 1.0:
        raise ValueError("--single-group-atom-max-weight must be >= 1")
    if not 0.0 <= args.semantic_align_margin <= 1.0:
        raise ValueError("--semantic-align-margin must be in [0, 1]")
    if not 0.0 <= args.query_align_margin <= 1.0:
        raise ValueError("--query-align-margin must be in [0, 1]")
    if not 0.0 <= args.segment_query_align_margin <= 1.0:
        raise ValueError("--segment-query-align-margin must be in [0, 1]")
    if not -1.0 <= args.segment_boundary_max_cosine <= 1.0:
        raise ValueError("--segment-boundary-max-cosine must be in [-1, 1]")
    if not 0.0 <= args.semantic_text_dropout < 1.0:
        raise ValueError("--semantic-text-dropout must be in [0, 1)")
    if not 0.0 <= args.semantic_text_scale <= 1.0:
        raise ValueError("--semantic-text-scale must be in [0, 1]")
    if not 0.0 <= args.human_semantic_weight <= 1.0:
        raise ValueError("--human-semantic-weight must be in [0, 1]")
    if not 0.0 <= args.human_semantic_input_scale <= 1.0:
        raise ValueError("--human-semantic-input-scale must be in [0, 1]")
    if not 0.0 <= args.human_metric_input_scale <= 1.0:
        raise ValueError("--human-metric-input-scale must be in [0, 1]")
    if not 0.0 <= args.semantic_input_dropout < 1.0:
        raise ValueError("--semantic-input-dropout must be in [0, 1)")
    if not 0.0 <= args.semantic_metric_input_dropout < 1.0:
        raise ValueError("--semantic-metric-input-dropout must be in [0, 1)")
    if args.segment_query_temperature <= 0.0:
        raise ValueError("--segment-query-temperature must be > 0")
    if args.phase_contrastive_temperature <= 0.0:
        raise ValueError("--phase-contrastive-temperature must be > 0")


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    apply_profile_defaults(args, raw_argv)
    validate_args(args)
    os.environ["HNU_TEXT_ENCODER"] = args.text_encoder  # read by hashed_text_vector everywhere in this process
    if args.mode == "audit":
        audit_and_prepare(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()




