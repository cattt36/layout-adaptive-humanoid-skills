#!/usr/bin/env python3
"""Evaluate learned-only and ontology-assisted instruction routing.

The model-only metric is the primary metric.  Ontology-assisted routing is
reported separately because explicit bilingual cues can override a model
mistake and therefore must not be presented as learned generalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

LATENT_ROOT = Path(__file__).resolve().parent
if str(LATENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_ROOT))

from motion_latent_z import _normalize_command_text, hashed_text_vector
from train_joint_semantic_router import DEFAULT_ACTIONS, Router, explicit_action_cue

CASES = {
    "CrouchedTraverse": (
        "pass beneath the bridge while crouching",
        "Duck beneath the bridge and keep your head low.",
        "Crouch and move below the low bridge.",
        "Please pass under the overhead bridge without touching it.",
        "\u4ece\u6865\u4e0b\u4f4e\u59ff\u901a\u8fc7",
        "\u5f2f\u8170\u7a7f\u8fc7\u4f4e\u77ee\u6865",
    ),
    "ThreeStepPoleRun": (
        "clear three barriers in a row",
        "Clear the three poles one after another.",
        "Sprint across 3 low hurdles.",
        "Run over three rails in succession.",
        "There are three barriers ahead; cross them continuously.",
        "\u5f00\u59cb\u8de8\u8fc7\u4e09\u6b65\u6869",
        "\u8fde\u7eed\u8d8a\u8fc7\u4e09\u4e2a\u969c\u788d",
    ),
    "FiveStepPoleRun": (
        "sprint over the five low obstacles",
        "Clear the five poles one after another.",
        "Sprint across 5 low hurdles.",
        "Run over five beams in succession.",
        "There are five barriers ahead; cross them continuously.",
        "\u5f00\u59cb\u8de8\u8fc7\u4e94\u6b65\u6869",
        "\u8fde\u7eed\u8d8a\u8fc7\u4e94\u4e2a\u969c\u788d",
    ),
}

# These queries intentionally avoid the explicit ontology cue.  They are a
# diagnostic stress set for the learned router, not a replacement for a held-out
# test set (the wording is hand-authored and therefore auxiliary).
CUE_FREE_CASES = {
    "CrouchedTraverse": ("Duck beneath the low overhead structure and keep your head down.",),
    "ThreeStepPoleRun": ("Run across the shorter sequence of low rails one after another.",),
    "FiveStepPoleRun": ("Run across the longer sequence of low rails one after another.",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=LATENT_ROOT / "moe_router_v3" / "joint_semantic_router.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LATENT_ROOT / "moe_router_v3" / "instruction_routing_validation.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actions = tuple(checkpoint.get("actions", DEFAULT_ACTIONS))
    model = Router(
        int(checkpoint["text_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["latent_dim"]),
        len(actions),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: list[dict[str, object]] = []
    cue_free_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for expected, queries in CASES.items():
            for query in queries:
                normalized = _normalize_command_text(query)
                vector = hashed_text_vector(normalized, int(checkpoint["text_dim"]))
                tensor = torch.from_numpy(vector[None, :]).to(device)
                _, logits = model(tensor)
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                learned = actions[int(np.argmax(probabilities))]
                cue = explicit_action_cue(query)
                routed = cue if cue in actions else learned
                rows.append(
                    {
                        "query": query,
                        "normalized_query": normalized,
                        "expected_action": expected,
                        "learned_prediction": learned,
                        "routed_prediction": routed,
                        "routing_basis": "explicit_command_ontology" if cue in actions else "learned_router_fallback",
                        "correct": routed == expected,
                        "probabilities": {
                            action: float(probabilities[i]) for i, action in enumerate(actions)
                        },
                    }
                )

    with torch.no_grad():
        for expected, queries in CUE_FREE_CASES.items():
            for query in queries:
                normalized = _normalize_command_text(query)
                vector = hashed_text_vector(normalized, int(checkpoint["text_dim"]))
                _, logits = model(torch.from_numpy(vector[None, :]).to(device))
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                learned = actions[int(np.argmax(probabilities))]
                cue = explicit_action_cue(query)
                cue_free_rows.append({
                    "query": query, "expected_action": expected,
                    "learned_prediction": learned, "explicit_cue": cue,
                    "correct": learned == expected,
                    "probabilities": {a: float(probabilities[i]) for i, a in enumerate(actions)},
                })

    correct = sum(bool(row["correct"]) for row in rows)
    learned_correct = sum(row["learned_prediction"] == row["expected_action"] for row in rows)
    cue_free_correct = sum(bool(row["correct"]) for row in cue_free_rows)
    report = {
        "format": "instruction_routing_validation_v2",
        "legacy_format": "instruction_routing_validation_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "actions": list(actions),
        "total": len(rows),
        # Keep legacy keys, but make the model-only metric explicit and primary.
        "correct": learned_correct,
        "accuracy": learned_correct / max(len(rows), 1),
        "learned_correct": learned_correct,
        "learned_accuracy": learned_correct / max(len(rows), 1),
        "ontology_routed_correct": correct,
        "ontology_routed_accuracy": correct / max(len(rows), 1),
        "explicit_cue_count": sum(bool(row["routing_basis"] == "explicit_command_ontology") for row in rows),
        "learned_only": {
            "correct": learned_correct,
            "total": len(rows),
            "accuracy": learned_correct / max(len(rows), 1),
        },
        "ontology_assisted": {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / max(len(rows), 1),
            "override_count": sum(bool(row["routing_basis"] == "explicit_command_ontology") for row in rows),
        },
        "metric_primary": "learned_only",
        "evaluation_note": "The legacy CASES contain explicit cues; ontology_routed_accuracy is deployment behavior, not an independent learned-test score.",
        "rows": rows,
        "cue_free_stress": {
            "total": len(cue_free_rows),
            "correct": cue_free_correct,
            "accuracy": cue_free_correct / max(len(cue_free_rows), 1),
            "rows": cue_free_rows,
            "note": "Hand-authored diagnostic stress set; do not use for checkpoint or threshold selection.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"instruction routing: {correct}/{len(rows)} "
        f"({report['ontology_routed_accuracy']:.3f}); learned-only: "
        f"{learned_correct}/{len(rows)} ({report['learned_accuracy']:.3f}); cue-free stress: "
        f"{cue_free_correct}/{len(cue_free_rows)} ({report['cue_free_stress']['accuracy']:.3f})"
    )
    print(f"report: {args.output}")
    if learned_correct != len(rows):
        for row in rows:
            if row["learned_prediction"] != row["expected_action"]:
                print(
                    f"LEARNED FAIL expected={row['expected_action']} "
                    f"learned={row['learned_prediction']} routed={row['routed_prediction']} query={row['query']!r}"
                )
    if learned_correct != len(rows):
        # Keep the command useful in CI: a learned-only regression is a real
        # failure even when the ontology fallback masks it in deployment.
        raise SystemExit(1)


if __name__ == "__main__":
    main()

