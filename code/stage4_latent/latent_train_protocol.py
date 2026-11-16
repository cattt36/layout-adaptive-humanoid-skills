"""Shared fair training protocol for the five latent ablations.

The ablation variable is the annotation source enabled by ``--ablation-profile``;
windowing, split, model capacity, optimizer, and segment-query objectives are
kept fixed across Z0--Z4.  Profile-specific semantic/quality weights are added
by each wrapper only when they correspond to an available supervision head.
"""

from __future__ import annotations

from collections.abc import Sequence

# These are intentionally CLI tokens so a wrapper can still be overridden by
# user-provided arguments appended after this list.
COMMON_TRAIN_ARGS = [
    "--mode", "train",
    "--epochs", "180",
    "--warmup-epochs", "20",
    "--semantic-ramp-epochs", "30",
    "--batch-size", "64",
    "--latent-dim", "16",
    "--quality-latent-dim", "12",
    "--channels", "128",
    "--local-feature-dim", "16",
    "--dropout", "0.18",
    "--learning-rate", "1.5e-4",
    "--weight-decay", "5e-4",
    "--grad-clip", "1.0",
    "--ema-decay", "0.995",
    "--early-stop-patience", "32",
    "--lr-patience", "6",
    "--lr-factor", "0.5",
    "--min-lr", "1e-6",
    "--window", "96",
    "--stride", "12",
    "--semantic-purity-threshold", "0.55",
    "--semantic-window-min-weight", "0.15",
    "--text-dim", "128",
    # The same task and segment query protocol is used in every profile.
    "--lambda-query-align", "0.08",
    "--lambda-query-consistency", "0.05",
    "--lambda-deployment-consistency", "0.12",
    "--query-align-margin", "0.65",
    "--lambda-segment-query-align", "0.12",
    "--lambda-segment-query-contrastive", "0.10",
    "--segment-query-align-margin", "0.75",
    "--segment-query-temperature", "0.10",
    # Shared motion/quality/disentanglement objective.
    "--lambda-recon", "1.0",
    "--lambda-smooth", "0.05",
    "--lambda-quality", "0.50",
    "--lambda-quality-tag-ranking", "0.03",
    "--lambda-decorrelation", "0.06",
    "--lambda-variance", "0.06",
    "--lambda-skill-covariance", "0.04",
    "--lambda-quality-covariance", "0.02",
    "--lambda-semantic-covariance", "0.02",
    "--motion-noise-std", "0.008",
    "--semantic-input-dropout", "0.08",
    "--semantic-metric-input-dropout", "0.20",
    "--dataset-balance-boost", "0.45",
    "--trial-balance-boost", "0.25",
    "--max-sampler-weight", "1.80",
    "--quality-label-version", "v2",
    "--quality-focal-gamma", "2.0",
    "--quality-focal-alpha", "0.75",
    "--quality-negative-weight", "0.5",
]

# Only the supervision-specific weights differ.  The representations and
# exported interfaces remain identical, which makes downstream distillation
# comparisons interpretable.
PROFILE_ARGS = {
    "Z0": [
        "--lambda-semantic-decode", "0.08",
        "--lambda-align", "0.08",
        "--lambda-metric", "0.0",
        "--lambda-atoms", "0.0",
        "--lambda-atom-ranking", "0.0",
        "--lambda-skill-metric", "0.0",
        "--lambda-skill-atoms", "0.0",
    ],
    "Z1": [
        "--lambda-semantic-decode", "0.08",
        "--lambda-align", "0.08",
        "--lambda-metric", "0.35",
        "--lambda-atoms", "0.0",
        "--lambda-atom-ranking", "0.0",
        "--lambda-skill-metric", "0.08",
        "--lambda-skill-atoms", "0.0",
        "--checkpoint-quality-f1-weight", "0.10",
    ],
    "Z2": [
        "--lambda-semantic-decode", "0.08",
        "--lambda-align", "0.06",
        "--lambda-metric", "0.30",
        "--lambda-atoms", "0.30",
        "--lambda-atom-ranking", "0.08",
        "--lambda-skill-metric", "0.08",
        "--lambda-skill-atoms", "0.06",
        "--lambda-semantic-norm", "0.06",
        "--rare-sample-boost", "0.15",
        "--coverage-sample-boost", "0.20",
        "--checkpoint-semantic-f1-weight", "0.08",
        "--checkpoint-rare-atom-f1-weight", "0.16",
        "--checkpoint-quality-f1-weight", "0.04",
        "--robot-atom-positive-weight", "0.85",
        "--llm-absent-atom-weight", "0.25",
        "--phase-absent-atom-weight", "1.0",
    ],
    "Z3": [
        # Keep the deployable task-only branch close to the motion teacher
        # while human metrics remain a supplementary source.
        "--lambda-query-align", "0.16",
        "--lambda-query-consistency", "0.12",
        "--lambda-deployment-consistency", "0.24",
        "--human-metric-input-scale", "0.70",
        "--lambda-semantic-decode", "0.08",
        "--lambda-align", "0.06",
        "--lambda-metric", "0.35",
        "--lambda-atoms", "0.30",
        "--lambda-atom-ranking", "0.08",
        "--lambda-skill-metric", "0.10",
        "--lambda-skill-atoms", "0.06",
        "--lambda-semantic-norm", "0.06",
        "--rare-sample-boost", "0.15",
        "--coverage-sample-boost", "0.20",
        "--checkpoint-semantic-f1-weight", "0.08",
        "--checkpoint-rare-atom-f1-weight", "0.16",
        "--checkpoint-quality-f1-weight", "0.06",
        "--checkpoint-skill-semantic-f1-weight", "0.06",
        "--robot-atom-positive-weight", "0.85",
        "--llm-absent-atom-weight", "0.25",
        "--phase-absent-atom-weight", "1.0",
    ],
    "Z4": [
        # Keep effective Z4 tuning explicit and reproducible in the shared protocol.
        # Stronger task-only distillation is essential because robot deployment
        # cannot provide the human/robot annotation branches used by the teacher.
        "--lambda-query-align", "0.24",
        "--lambda-query-consistency", "0.18",
        "--lambda-deployment-consistency", "0.35",
        "--lambda-segment-query-align", "0.12",
        "--lambda-segment-query-contrastive", "0.10",
        "--segment-query-temperature", "0.10",
        "--semantic-text-dropout", "0.08",
        "--semantic-text-scale", "0.75",
        "--human-semantic-weight", "0.35",
        "--human-semantic-input-scale", "0.30",
        "--human-metric-input-scale", "0.70",
        "--lambda-semantic-decode", "0.10",
        "--lambda-align", "0.10",
        "--lambda-metric", "0.40",
        "--lambda-atoms", "0.30",
        "--lambda-atom-ranking", "0.12",
        "--lambda-skill-metric", "0.10",
        "--lambda-skill-atoms", "0.08",
        "--lambda-semantic-norm", "0.10",
        "--rare-sample-boost", "0.25",
        "--coverage-sample-boost", "0.20",
        "--checkpoint-semantic-f1-weight", "0.08",
        "--checkpoint-rare-atom-f1-weight", "0.16",
        "--checkpoint-quality-f1-weight", "0.16",
        "--checkpoint-skill-semantic-f1-weight", "0.08",
        "--checkpoint-semantic-norm-weight", "0.05",
        "--robot-atom-positive-weight", "0.85",
        "--human-atom-positive-weight", "0.85",
        "--llm-absent-atom-weight", "0.25",
        "--phase-absent-atom-weight", "1.0",
    ],
}


# ThreeStepPoleRun and FiveStepPoleRun contain short take-off/flight phases that can be
# swallowed by a 96-frame context window.  These action-specific settings
# preserve the full context for the encoder, but pool z_skill over the
# annotated phase itself and guarantee that every annotated segment
# contributes at least one training and evaluation window.  Keeping the
# settings action-scoped avoids silently changing the already validated
# CrouchedTraverse expert.
ACTION_COMMON_ARGS = {
    "ThreeStepPoleRun": [
        "--force-segment-centers",
        "--phase-focused-pooling",
        # Validation-only retrieval is used as a secondary checkpoint signal;
        # the held-out test subject never participates in model selection.
        "--checkpoint-phase-retrieval-weight", "0.25",
    ],
    "FiveStepPoleRun": [
        "--force-segment-centers",
        "--phase-focused-pooling",
        # Rescue v1: sample annotated segments rather than raw windows, fuse
        # phase-wide and 32-frame transition context, compact windows from the
        # same segment, and separate only temporally adjacent segments.  These
        # structural signals are shared by Z0--Z4 and do not inject gait-phase
        # semantics into the motion-only control.
        "--segment-balance-power", "1.0",
        "--max-sampler-weight", "8.0",
        "--skill-local-pool-frames", "32",
        "--lambda-segment-compactness", "0.06",
        "--lambda-segment-boundary", "0.06",
        "--segment-boundary-max-cosine", "0.70",
        # FiveStepPoleRun has five short flight intervals per trial.  Validation
        # retrieval is therefore a useful secondary signal, while the held-out
        # test subjects remain excluded from checkpoint selection.
        "--checkpoint-phase-retrieval-weight", "0.25",
    ],
}


# ThreeStepPoleRun needs an explicit phase-contrastive term from Z2 onward to keep
# its canonical subskills separated.  FiveStepPoleRun was tested with the same
# mechanism, but it did not improve either validation retrieval interface and
# reduced z_skill effective rank.  Its rescue therefore uses segment identity
# and temporal adjacency only; no privileged phase labels are added to the
# motion-only (Z0) or human-only (Z1) controls.
ACTION_PROFILE_ARGS = {
    ("ThreeStepPoleRun", "Z2"): [
        "--lambda-phase-contrastive", "0.10",
        "--phase-contrastive-temperature", "0.10",
        "--phase-balance-boost", "0.80",
    ],
    ("ThreeStepPoleRun", "Z3"): [
        "--lambda-phase-contrastive", "0.10",
        "--phase-contrastive-temperature", "0.10",
        "--phase-balance-boost", "0.80",
        # Subject anthropometrics and segment duration remain available to
        # z_sem, but human numeric supervision is kept supplementary for
        # z_skill so it cannot dominate the phase structure.
        "--human-metric-input-scale", "0.50",
        "--lambda-skill-metric", "0.05",
    ],
    ("ThreeStepPoleRun", "Z4"): [
        "--lambda-phase-contrastive", "0.10",
        "--phase-contrastive-temperature", "0.10",
        "--phase-balance-boost", "0.80",
        "--human-metric-input-scale", "0.50",
        "--lambda-skill-metric", "0.05",
        "--lambda-skill-atoms", "0.06",
    ],
}


def _dataset_from_cli(argv: Sequence[str]) -> str | None:
    """Return the single action requested by a wrapper invocation."""

    for index, token in enumerate(argv):
        if token == "--dataset" and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith("--dataset="):
            return token.split("=", 1)[1]
    return None


def resolved_train_args(profile: str, argv: Sequence[str]) -> list[str]:
    """Compose shared, profile, action-specific, then user CLI overrides."""

    user_args = list(argv)
    dataset = _dataset_from_cli(user_args)
    return [
        *COMMON_TRAIN_ARGS,
        *PROFILE_ARGS[profile],
        *ACTION_COMMON_ARGS.get(dataset or "", []),
        *ACTION_PROFILE_ARGS.get((dataset or "", profile), []),
        *user_args,
    ]


