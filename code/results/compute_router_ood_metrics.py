#!/usr/bin/env python3
"""Recompute the non-RL language-routing metrics required by Results 8.4.

The evaluator uses the frozen Router checkpoint and exactly the same text
normalizer, ontology and rejection policy as ``MOE Framework``.  It reports
the learned classifier separately from the deployed ontology-assisted router
so rule-assisted performance cannot be mistaken for learned generalization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def find_project_root(script_path: Path) -> Path:
    """Locate Data-HNU after this script is organized below Results/<section>."""
    for candidate in script_path.parents:
        if (candidate / "MOE Framework" / "Router").is_dir() and (
            candidate / "Results"
        ).is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the Data-HNU project root from {script_path}")


SCRIPT_PATH = Path(__file__).resolve()
CHAPTER_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = find_project_root(SCRIPT_PATH)
DEFAULT_FRAMEWORK_ROOT = PROJECT_ROOT / "MOE Framework"
DEFAULT_BENCHMARK = CHAPTER_ROOT / "inputs" / "router_ood_benchmark_v1.json"
DEFAULT_OUTPUT_DIR = CHAPTER_ROOT / "outputs"
REJECT = "__REJECT__"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def classification_metrics(
    expected: list[str], predicted: list[str | None], actions: tuple[str, ...]
) -> dict[str, Any]:
    if not expected:
        return {
            "n": 0,
            "accuracy": None,
            "accuracy_ci95_wilson": None,
            "macro_f1": None,
            "per_class": {},
            "confusion": {},
        }
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    confusion = {
        truth: {pred: 0 for pred in (*actions, REJECT)} for truth in actions
    }
    for truth, pred in zip(expected, predicted):
        confusion[truth][pred if pred in actions else REJECT] += 1
    for action in actions:
        tp = sum(t == action and p == action for t, p in zip(expected, predicted))
        fp = sum(t != action and p == action for t, p in zip(expected, predicted))
        fn = sum(t == action and p != action for t, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[action] = {
            "support": sum(t == action for t in expected),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(t == p for t, p in zip(expected, predicted))
    return {
        "n": len(expected),
        "correct": correct,
        "accuracy": correct / len(expected),
        "accuracy_ci95_wilson": wilson_interval(correct, len(expected)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion": confusion,
    }


def reciprocal_rank(order: list[str], expected: str) -> float:
    return 1.0 / (order.index(expected) + 1)


def validate_benchmark(payload: dict[str, Any], actions: tuple[str, ...]) -> list[dict[str, Any]]:
    if payload.get("format") != "router_ood_benchmark_v1":
        raise ValueError("Unexpected benchmark format")
    if payload.get("frozen_before_first_scoring") is not True:
        raise ValueError("Benchmark must be frozen before scoring")
    cases = payload.get("cases")
    if not isinstance(cases, list) or payload.get("case_count") != len(cases):
        raise ValueError("Benchmark case_count mismatch")
    ids = [str(case.get("id", "")) for case in cases]
    queries = [str(case.get("query", "")) for case in cases]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Benchmark case IDs must be nonempty and unique")
    if not all(queries) or len(queries) != len(set(queries)):
        raise ValueError("Benchmark queries must be nonempty and unique")
    valid_conditions = set(payload["construction"]["resolvable_conditions"])
    reject_conditions = set(payload["construction"]["rejection_conditions"])
    for case in cases:
        condition = case.get("condition")
        expected = case.get("expected_action")
        abstain = case.get("should_abstain")
        if condition in valid_conditions:
            if abstain is not False or expected not in actions:
                raise ValueError(f"Invalid resolvable case: {case.get('id')}")
        elif condition in reject_conditions:
            if abstain is not True or expected is not None:
                raise ValueError(f"Invalid rejection case: {case.get('id')}")
        else:
            raise ValueError(f"Unknown condition in {case.get('id')}: {condition}")
    for condition in valid_conditions:
        subset = [case for case in cases if case["condition"] == condition]
        counts = Counter(case["expected_action"] for case in subset)
        if len(set(counts.values())) != 1 or set(counts) != set(actions):
            raise ValueError(f"Resolvable condition is not action-balanced: {condition}")
    return cases


def policy_decision(
    row: dict[str, Any], allow_learned_fallback: bool, min_confidence: float, min_margin: float
) -> tuple[bool, str | None]:
    explicit_route = row["explicit_cue"] in row["actions"]
    accepted = bool(
        explicit_route
        or (
            allow_learned_fallback
            and row["routed_top_probability"] >= min_confidence
            and row["routed_top_margin"] >= min_margin
        )
    )
    return accepted, row["routed_prediction"] if accepted else None


def system_metrics(
    rows: list[dict[str, Any]],
    actions: tuple[str, ...],
    allow_learned_fallback: bool,
    min_confidence: float,
    min_margin: float,
) -> dict[str, Any]:
    valid = [row for row in rows if not row["should_abstain"]]
    invalid = [row for row in rows if row["should_abstain"]]
    decisions = {
        row["id"]: policy_decision(row, allow_learned_fallback, min_confidence, min_margin)
        for row in rows
    }
    valid_predictions = [decisions[row["id"]][1] for row in valid]
    valid_expected = [row["expected_action"] for row in valid]
    routed_classification = classification_metrics(valid_expected, valid_predictions, actions)
    learned_classification = classification_metrics(
        valid_expected, [row["learned_prediction"] for row in valid], actions
    )
    valid_accepted = sum(decisions[row["id"]][0] for row in valid)
    valid_correct = sum(
        decisions[row["id"]][0] and decisions[row["id"]][1] == row["expected_action"]
        for row in valid
    )
    valid_wrong = sum(
        decisions[row["id"]][0] and decisions[row["id"]][1] != row["expected_action"]
        for row in valid
    )
    invalid_rejected = sum(not decisions[row["id"]][0] for row in invalid)
    invalid_activated = len(invalid) - invalid_rejected
    wrong_activation = valid_wrong + invalid_activated
    correct_decisions = valid_correct + invalid_rejected
    routed_mrr = (
        float(np.mean([reciprocal_rank(row["routed_order"], row["expected_action"]) for row in valid]))
        if valid
        else None
    )
    learned_mrr = (
        float(np.mean([reciprocal_rank(row["learned_order"], row["expected_action"]) for row in valid]))
        if valid
        else None
    )
    result: dict[str, Any] = {
        "policy": {
            "allow_learned_fallback": allow_learned_fallback,
            "min_confidence": min_confidence,
            "min_margin": min_margin,
            "explicit_ontology_routes_bypass_thresholds": True,
        },
        "total_n": len(rows),
        "resolvable_n": len(valid),
        "rejection_required_n": len(invalid),
        "routing_accuracy": routed_classification["accuracy"],
        "routing_accuracy_ci95_wilson": routed_classification["accuracy_ci95_wilson"],
        "routing_macro_f1": routed_classification["macro_f1"],
        "routing_mrr": routed_mrr,
        "learned_top1_accuracy": learned_classification["accuracy"],
        "learned_macro_f1": learned_classification["macro_f1"],
        "learned_mrr": learned_mrr,
        "valid_coverage": valid_accepted / len(valid) if valid else None,
        "valid_coverage_ci95_wilson": wilson_interval(valid_accepted, len(valid)),
        "valid_false_rejection_rate": (len(valid) - valid_accepted) / len(valid) if valid else None,
        "accepted_valid_accuracy": valid_correct / valid_accepted if valid_accepted else None,
        "abstention_accuracy": invalid_rejected / len(invalid) if invalid else None,
        "abstention_accuracy_ci95_wilson": wilson_interval(invalid_rejected, len(invalid)),
        "wrong_action_activation_rate": wrong_activation / len(rows) if rows else None,
        "wrong_action_activation_count": wrong_activation,
        "valid_wrong_action_activation_rate": valid_wrong / len(valid) if valid else None,
        "invalid_activation_rate": invalid_activated / len(invalid) if invalid else None,
        "safe_decision_accuracy": correct_decisions / len(rows) if rows else None,
        "safe_decision_accuracy_ci95_wilson": wilson_interval(correct_decisions, len(rows)),
        "ontology_cue_rate_valid": (
            sum(row["explicit_cue"] in actions for row in valid) / len(valid) if valid else None
        ),
        "routed_classification": routed_classification,
        "learned_classification": learned_classification,
    }
    return result


def infer_rows(
    cases: Iterable[dict[str, Any]],
    model: torch.nn.Module,
    actions: tuple[str, ...],
    text_dim: int,
    device: torch.device,
    normalize: Any,
    vectorize: Any,
    cue_fn: Any,
) -> list[dict[str, Any]]:
    cases = list(cases)
    vectors = np.stack(
        [vectorize(normalize(str(case["query"])), text_dim) for case in cases]
    ).astype(np.float32)
    with torch.no_grad():
        _, logits = model(torch.from_numpy(vectors).to(device))
        learned_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    rows: list[dict[str, Any]] = []
    for case, raw in zip(cases, learned_probabilities):
        learned_order_idx = np.argsort(raw)[::-1]
        cue = cue_fn(str(case["query"]))
        routed = raw.copy()
        if cue in actions:
            cue_index = actions.index(cue)
            routed *= 0.10
            routed[cue_index] = 0.90 + 0.10 * routed[cue_index]
        routed_order_idx = np.argsort(routed)[::-1]
        top = int(routed_order_idx[0])
        second = int(routed_order_idx[1])
        row = dict(case)
        row.update(
            {
                "actions": list(actions),
                "normalized_query": normalize(str(case["query"])),
                "explicit_cue": cue,
                "learned_prediction": actions[int(learned_order_idx[0])],
                "learned_order": [actions[int(index)] for index in learned_order_idx],
                "learned_probabilities": {
                    action: float(raw[i]) for i, action in enumerate(actions)
                },
                "routed_prediction": actions[top],
                "routed_order": [actions[int(index)] for index in routed_order_idx],
                "routed_probabilities": {
                    action: float(routed[i]) for i, action in enumerate(actions)
                },
                "routed_top_probability": float(routed[top]),
                "routed_top_margin": float(routed[top] - routed[second]),
            }
        )
        rows.append(row)
    return rows


def metric_row(
    condition: str,
    rows: list[dict[str, Any]],
    actions: tuple[str, ...],
    allow_fallback: bool,
    min_confidence: float,
    min_margin: float,
) -> dict[str, Any]:
    metrics = system_metrics(rows, actions, allow_fallback, min_confidence, min_margin)
    return {
        "condition": condition,
        "total_n": metrics["total_n"],
        "resolvable_n": metrics["resolvable_n"],
        "rejection_required_n": metrics["rejection_required_n"],
        "learned_top1_accuracy": metrics["learned_top1_accuracy"],
        "learned_macro_f1": metrics["learned_macro_f1"],
        "learned_mrr": metrics["learned_mrr"],
        "routing_accuracy": metrics["routing_accuracy"],
        "routing_macro_f1": metrics["routing_macro_f1"],
        "routing_mrr": metrics["routing_mrr"],
        "valid_coverage": metrics["valid_coverage"],
        "accepted_valid_accuracy": metrics["accepted_valid_accuracy"],
        "abstention_accuracy": metrics["abstention_accuracy"],
        "wrong_action_activation_rate": metrics["wrong_action_activation_rate"],
        "valid_wrong_action_activation_rate": metrics["valid_wrong_action_activation_rate"],
        "invalid_activation_rate": metrics["invalid_activation_rate"],
        "safe_decision_accuracy": metrics["safe_decision_accuracy"],
        "ontology_cue_rate_valid": metrics["ontology_cue_rate_valid"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def make_summary(report: dict[str, Any], condition_rows: list[dict[str, Any]]) -> str:
    default = report["development_benchmark"]["default_deployment_policy"]
    fallback = report["development_benchmark"]["learned_fallback_policy"]
    validation = report["existing_checkpoint_evidence"]["group_disjoint_validation"]
    legacy = report["existing_checkpoint_evidence"]["legacy_explicit_query_set"]
    lines = [
        "# Section 8.4 non-RL routing results",
        "",
        "The current files support language routing and rejection evaluation only. "
        "Environment-OOD success and instruction-to-execution success require a "
        "closed-loop policy, scene observations, and rollouts and are not reported here.",
        "",
        "## Existing checkpoint evidence",
        "",
        "| Set | N | Learned accuracy | Learned Macro-F1 | Learned MRR | Full routed Macro-F1 | Full routed MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Normalized-group-disjoint checkpoint validation | {validation['total_n']} | {fmt(validation['learned_top1_accuracy'])} | {fmt(validation['learned_macro_f1'])} | {fmt(validation['learned_mrr'])} | {fmt(validation['routing_macro_f1'])} | {fmt(validation['routing_mrr'])} |",
        f"| Legacy explicit bilingual queries | {legacy['total_n']} | {fmt(legacy['learned_top1_accuracy'])} | {fmt(legacy['learned_macro_f1'])} | {fmt(legacy['learned_mrr'])} | {fmt(legacy['routing_macro_f1'])} | {fmt(legacy['routing_mrr'])} |",
        "",
        "The 13-item validation split selected the checkpoint and is not an untouched "
        "test set. The legacy queries all contain explicit ontology cues; therefore the "
        "full routed score is deployment behavior rather than independent learned-model evidence.",
        "",
        "## Frozen development OOD benchmark",
        "",
        "The benchmark contains 48 resolvable commands (four per action in each of "
        "four language conditions) and 24 rejection-required commands. It was frozen "
        "before the first scoring run and was not used to change the Router or thresholds.",
        "",
        "| Condition | N | Learned Macro-F1 | Learned MRR | Routed Macro-F1 | Routed MRR | Valid coverage | Abstention accuracy | Wrong activation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in condition_rows:
        if row["condition"] in {"all_benchmark", "all_resolvable", "all_rejection_required"}:
            continue
        lines.append(
            f"| {row['condition']} | {row['total_n']} | {fmt(row['learned_macro_f1'])} | "
            f"{fmt(row['learned_mrr'])} | {fmt(row['routing_macro_f1'])} | "
            f"{fmt(row['routing_mrr'])} | {fmt(row['valid_coverage'])} | "
            f"{fmt(row['abstention_accuracy'])} | {fmt(row['wrong_action_activation_rate'])} |"
        )
    lines.extend(
        [
            "",
            "### Default safe policy",
            "",
            f"On the 48 resolvable OOD commands, routing Macro-F1 is "
            f"{default['routing_macro_f1']:.3f}, routing MRR is {default['routing_mrr']:.3f}, "
            f"and valid-command coverage is {default['valid_coverage']:.3f}. Among the "
            f"24 ambiguous/OOS commands, abstention accuracy is "
            f"{default['abstention_accuracy']:.3f}. The wrong-action activation rate "
            f"over all 72 commands is {default['wrong_action_activation_rate']:.3f}.",
            "",
            "### Learned-fallback diagnostic",
            "",
            f"Enabling the existing confidence/margin fallback changes valid coverage to "
            f"{fallback['valid_coverage']:.3f}, routing Macro-F1 to "
            f"{fallback['routing_macro_f1']:.3f}, abstention accuracy to "
            f"{fallback['abstention_accuracy']:.3f}, and wrong-action activation to "
            f"{fallback['wrong_action_activation_rate']:.3f}. This is a diagnostic only; "
            "the benchmark was not used to select a new deployment threshold.",
            "",
            "## Metric definitions",
            "",
            "- Routing Macro-F1 treats a rejected resolvable command as a false negative.",
            "- Routing MRR uses the ontology-assisted ranking even when the safe policy rejects execution.",
            "- Abstention accuracy is the fraction of ambiguous/OOS commands that are rejected.",
            "- Wrong-action activation counts an accepted wrong expert for a resolvable command, or any accepted expert for an ambiguous/OOS command, divided by all commands.",
            "- Valid coverage is the fraction of resolvable commands for which an expert is activated.",
            "",
            "## Interpretation boundary",
            "",
            "These are development diagnostics from a small hand-authored command set. "
            "The ASR condition uses synthetic transcript corruption rather than recorded "
            "audio. Paper-final claims require independently collected utterances, a "
            "pre-registered split, real ASR transcripts at multiple noise levels, and no "
            "threshold tuning on the reported test set. No result here is an environment-"
            "OOD or instruction-to-execution success rate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", type=Path, default=DEFAULT_FRAMEWORK_ROOT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if not 0.0 <= args.min_confidence <= 1.0 or not 0.0 <= args.min_margin <= 1.0:
        raise ValueError("Thresholds must be within [0, 1]")

    framework_root = args.framework_root.resolve()
    benchmark_path = args.benchmark.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = framework_root / "Router" / "moe_router_v3" / "joint_semantic_router.pt"
    manifest_path = framework_root / "Router" / "moe_router_v3" / "router_manifest.json"
    legacy_path = framework_root / "Router" / "moe_router_v3" / "instruction_routing_validation.json"
    for path in (checkpoint_path, manifest_path, legacy_path, benchmark_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if str(framework_root) not in sys.path:
        sys.path.insert(0, str(framework_root))
    from motion_latent_z import (  # pylint: disable=import-outside-toplevel
        CHECKPOINT_FORMAT_VERSION,
        _normalize_command_text,
        hashed_text_vector,
    )
    from train_joint_semantic_router import (  # pylint: disable=import-outside-toplevel
        DEFAULT_ACTIONS,
        Router,
        explicit_action_cue,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actions = tuple(checkpoint.get("actions", DEFAULT_ACTIONS))
    if actions != tuple(DEFAULT_ACTIONS):
        raise ValueError(f"Unexpected Router actions/order: {actions}")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    model = Router(
        int(checkpoint["text_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["latent_dim"]),
        len(actions),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    expert_status = {}
    for action in actions:
        expert = framework_root / "Latent" / action / "Z4" / "best_motion_latent.pt"
        if not expert.is_file():
            raise FileNotFoundError(expert)
        protocol = torch.load(expert, map_location="cpu", weights_only=False).get(
            "checkpoint_format_version"
        )
        expert_status[action] = {
            "relative_checkpoint": expert.relative_to(framework_root).as_posix(),
            "protocol": protocol,
            "ready": protocol == CHECKPOINT_FORMAT_VERSION,
        }
    if not all(item["ready"] for item in expert_status.values()):
        raise RuntimeError(f"One or more Z4 experts are incompatible: {expert_status}")

    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_cases = validate_benchmark(benchmark_payload, actions)
    benchmark_rows = infer_rows(
        benchmark_cases,
        model,
        actions,
        int(checkpoint["text_dim"]),
        device,
        _normalize_command_text,
        hashed_text_vector,
        explicit_action_cue,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_cases = [
        {
            "id": f"manifest_{i:03d}",
            "condition": "router_manifest",
            "language": "mixed",
            "query": text,
            "expected_action": source,
            "should_abstain": False,
        }
        for i, (text, source) in enumerate(zip(manifest["texts"], manifest["sources"]))
    ]
    manifest_rows = infer_rows(
        manifest_cases,
        model,
        actions,
        int(checkpoint["text_dim"]),
        device,
        _normalize_command_text,
        hashed_text_vector,
        explicit_action_cue,
    )
    train_indices = set(int(i) for i in manifest["train_indices"])
    val_indices = set(int(i) for i in manifest["val_indices"])
    if train_indices & val_indices or train_indices | val_indices != set(range(len(manifest_rows))):
        raise ValueError("Invalid Router manifest split")
    train_rows = [row for i, row in enumerate(manifest_rows) if i in train_indices]
    val_rows = [row for i, row in enumerate(manifest_rows) if i in val_indices]

    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_cases = [
        {
            "id": f"legacy_{i:03d}",
            "condition": "legacy_explicit_query",
            "language": "mixed",
            "query": row["query"],
            "expected_action": row["expected_action"],
            "should_abstain": False,
        }
        for i, row in enumerate(legacy["rows"])
    ]
    legacy_rows = infer_rows(
        legacy_cases,
        model,
        actions,
        int(checkpoint["text_dim"]),
        device,
        _normalize_command_text,
        hashed_text_vector,
        explicit_action_cue,
    )

    default_metrics = system_metrics(
        benchmark_rows, actions, False, args.min_confidence, args.min_margin
    )
    fallback_metrics = system_metrics(
        benchmark_rows, actions, True, args.min_confidence, args.min_margin
    )
    train_metrics = system_metrics(train_rows, actions, False, args.min_confidence, args.min_margin)
    val_metrics = system_metrics(val_rows, actions, False, args.min_confidence, args.min_margin)
    legacy_metrics = system_metrics(
        legacy_rows, actions, False, args.min_confidence, args.min_margin
    )

    conditions = [
        "held_out_paraphrase",
        "bilingual_code_switch",
        "synthetic_asr_noise",
        "implicit_description",
        "ambiguous",
        "out_of_scope",
    ]
    condition_rows = [
        metric_row(
            condition,
            [row for row in benchmark_rows if row["condition"] == condition],
            actions,
            False,
            args.min_confidence,
            args.min_margin,
        )
        for condition in conditions
    ]
    condition_rows.extend(
        [
            metric_row(
                "all_resolvable",
                [row for row in benchmark_rows if not row["should_abstain"]],
                actions,
                False,
                args.min_confidence,
                args.min_margin,
            ),
            metric_row(
                "all_rejection_required",
                [row for row in benchmark_rows if row["should_abstain"]],
                actions,
                False,
                args.min_confidence,
                args.min_margin,
            ),
            metric_row(
                "all_benchmark",
                benchmark_rows,
                actions,
                False,
                args.min_confidence,
                args.min_margin,
            ),
        ]
    )

    threshold_rows: list[dict[str, Any]] = []
    threshold_rows.append(
        {
            "policy": "explicit_only_default",
            "min_confidence": args.min_confidence,
            "min_margin": args.min_margin,
            **{
                key: default_metrics[key]
                for key in (
                    "routing_accuracy",
                    "routing_macro_f1",
                    "routing_mrr",
                    "valid_coverage",
                    "abstention_accuracy",
                    "wrong_action_activation_rate",
                    "safe_decision_accuracy",
                )
            },
        }
    )
    for confidence in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        for margin in (0.05, 0.15, 0.25, 0.35):
            metrics = system_metrics(benchmark_rows, actions, True, confidence, margin)
            threshold_rows.append(
                {
                    "policy": "learned_fallback_diagnostic",
                    "min_confidence": confidence,
                    "min_margin": margin,
                    **{
                        key: metrics[key]
                        for key in (
                            "routing_accuracy",
                            "routing_macro_f1",
                            "routing_mrr",
                            "valid_coverage",
                            "abstention_accuracy",
                            "wrong_action_activation_rate",
                            "safe_decision_accuracy",
                        )
                    },
                }
            )

    all_manifest_texts = set(manifest["texts"])
    all_manifest_normalized = set(manifest["normalized_texts"])
    exact_overlap = sum(row["query"] in all_manifest_texts for row in benchmark_rows)
    normalized_overlap = sum(row["normalized_query"] in all_manifest_normalized for row in benchmark_rows)
    ontology_changes = sum(
        row["learned_prediction"] != row["routed_prediction"]
        for row in benchmark_rows
        if not row["should_abstain"]
    )
    ontology_corrections = sum(
        row["learned_prediction"] != row["expected_action"]
        and row["routed_prediction"] == row["expected_action"]
        for row in benchmark_rows
        if not row["should_abstain"]
    )
    ontology_harms = sum(
        row["learned_prediction"] == row["expected_action"]
        and row["routed_prediction"] != row["expected_action"]
        for row in benchmark_rows
        if not row["should_abstain"]
    )

    report = {
        "format": "router_ood_metrics_v1",
        "paper_section": "8.4 Language and Environment OOD Generalization",
        "scope": "non-RL language routing only",
        "actions": list(actions),
        "source": {
            "framework_root": "MOE Framework",
            "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": sha256(checkpoint_path),
            "router_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "router_manifest_sha256": sha256(manifest_path),
            "legacy_validation": legacy_path.relative_to(PROJECT_ROOT).as_posix(),
            "legacy_validation_sha256": sha256(legacy_path),
            "benchmark": benchmark_path.relative_to(PROJECT_ROOT).as_posix(),
            "benchmark_sha256": sha256(benchmark_path),
            "expert_status": expert_status,
        },
        "protocol": {
            "default_deployment_policy": "Explicit ontology cues activate an expert; cue-free learned routes are rejected.",
            "diagnostic_fallback_policy": "Cue-free routes are accepted only when top probability and margin pass the fixed CLI defaults.",
            "min_confidence": args.min_confidence,
            "min_margin": args.min_margin,
            "ranking": "Three action probabilities, descending; ontology cue applies the same 0.90/0.10 reweighting as route_moe_instruction.py.",
            "metric_definitions": {
                "routing_macro_f1": "Macro-F1 over the three resolvable actions; rejected valid commands count as false negatives.",
                "routing_mrr": "Mean reciprocal rank of the true action in the ontology-assisted ranking, irrespective of acceptance.",
                "abstention_accuracy": "Rejected ambiguous/OOS commands divided by all rejection-required commands.",
                "wrong_action_activation_rate": "Accepted wrong expert on a valid command plus any accepted expert on an ambiguous/OOS command, divided by all evaluated commands.",
                "valid_coverage": "Accepted resolvable commands divided by all resolvable commands.",
                "safe_decision_accuracy": "Correct accepted valid routes plus correct invalid rejections, divided by all commands."
            },
        },
        "split_audit": {
            "manifest_train_n": len(train_rows),
            "manifest_validation_n": len(val_rows),
            "manifest_normalized_group_overlap": manifest.get("normalized_group_overlap", []),
            "benchmark_exact_text_overlap_with_manifest": exact_overlap,
            "benchmark_normalized_text_overlap_with_manifest": normalized_overlap,
            "benchmark_frozen_before_first_scoring": benchmark_payload["frozen_before_first_scoring"],
            "benchmark_used_for_training_or_threshold_selection": False,
        },
        "existing_checkpoint_evidence": {
            "seen_training_commands": train_metrics,
            "group_disjoint_validation": val_metrics,
            "legacy_explicit_query_set": legacy_metrics,
        },
        "development_benchmark": {
            "case_count": len(benchmark_rows),
            "condition_counts": dict(Counter(row["condition"] for row in benchmark_rows)),
            "default_deployment_policy": default_metrics,
            "learned_fallback_policy": fallback_metrics,
            "ontology_effect_on_resolvable": {
                "changed_top1_count": ontology_changes,
                "corrected_learned_errors": ontology_corrections,
                "introduced_errors": ontology_harms,
            },
            "by_condition": condition_rows,
        },
        "not_computable_without_rl_or_scene_rollouts": [
            "environment-OOD success",
            "instruction-to-execution success",
            "closed-loop dangerous failure",
            "scene-conditioned retrieval/executability performance",
        ],
        "limitations": benchmark_payload["construction"]["limitations"]
        + [
            "The checkpoint-validation split was used for model selection and is not a final test set.",
            "The legacy query set contains explicit ontology cues, so its full routed score is not independent evidence for the learned classifier.",
            "Wilson intervals quantify finite command counts only and do not cover benchmark-authoring or model-retraining uncertainty.",
        ],
        "per_query": benchmark_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "router_ood_metrics.json"
    by_condition_path = output_dir / "router_ood_by_condition.csv"
    per_query_path = output_dir / "router_ood_per_query.csv"
    sweep_path = output_dir / "router_threshold_sweep.csv"
    summary_path = output_dir / "router_ood_summary.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(by_condition_path, condition_rows)
    write_csv(sweep_path, threshold_rows)

    per_query_csv: list[dict[str, Any]] = []
    for row in benchmark_rows:
        default_accepted, default_selected = policy_decision(
            row, False, args.min_confidence, args.min_margin
        )
        fallback_accepted, fallback_selected = policy_decision(
            row, True, args.min_confidence, args.min_margin
        )
        expected = row["expected_action"]
        per_query_csv.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "language": row.get("language", ""),
                "noise_level": row.get("noise_level", ""),
                "query": row["query"],
                "expected_action": expected or "REJECT",
                "explicit_cue": row["explicit_cue"] or "",
                "learned_prediction": row["learned_prediction"],
                "learned_reciprocal_rank": (
                    reciprocal_rank(row["learned_order"], expected) if expected else ""
                ),
                "routed_prediction_before_rejection": row["routed_prediction"],
                "routed_reciprocal_rank": (
                    reciprocal_rank(row["routed_order"], expected) if expected else ""
                ),
                "top_probability": row["routed_top_probability"],
                "top_margin": row["routed_top_margin"],
                "default_accepted": default_accepted,
                "default_selected_action": default_selected or "REJECT",
                "default_correct_decision": (
                    (not default_accepted)
                    if row["should_abstain"]
                    else (default_accepted and default_selected == expected)
                ),
                "fallback_accepted": fallback_accepted,
                "fallback_selected_action": fallback_selected or "REJECT",
                "fallback_correct_decision": (
                    (not fallback_accepted)
                    if row["should_abstain"]
                    else (fallback_accepted and fallback_selected == expected)
                ),
            }
        )
    write_csv(per_query_path, per_query_csv)
    summary_path.write_text(make_summary(report, condition_rows), encoding="utf-8")

    print(
        json.dumps(
            {
                "outputs": [
                    path.relative_to(PROJECT_ROOT).as_posix()
                    for path in (json_path, by_condition_path, per_query_path, sweep_path, summary_path)
                ],
                "checkpoint_validation": {
                    "n": val_metrics["total_n"],
                    "learned_macro_f1": val_metrics["learned_macro_f1"],
                    "learned_mrr": val_metrics["learned_mrr"],
                },
                "legacy_explicit_queries": {
                    "n": legacy_metrics["total_n"],
                    "learned_macro_f1": legacy_metrics["learned_macro_f1"],
                    "routed_macro_f1": legacy_metrics["routing_macro_f1"],
                },
                "development_ood_default": {
                    key: default_metrics[key]
                    for key in (
                        "resolvable_n",
                        "rejection_required_n",
                        "routing_macro_f1",
                        "routing_mrr",
                        "valid_coverage",
                        "abstention_accuracy",
                        "wrong_action_activation_rate",
                        "safe_decision_accuracy",
                    )
                },
                "development_ood_fallback": {
                    key: fallback_metrics[key]
                    for key in (
                        "routing_macro_f1",
                        "routing_mrr",
                        "valid_coverage",
                        "abstention_accuracy",
                        "wrong_action_activation_rate",
                        "safe_decision_accuracy",
                    )
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
