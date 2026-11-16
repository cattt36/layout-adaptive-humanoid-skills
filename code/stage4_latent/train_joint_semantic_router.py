#!/usr/bin/env python3
"""Train a shared instruction -> action semantic router for the MOE student.

The three action JSON directories are read independently and receive distinct
labels.  Their JSON contents are never merged into one action annotation.  The
router learns a shared ``z_sem`` command embedding and an action classifier:

    instruction -> shared z_sem -> action_id -> action-specific Z0..Z4 expert

This is intentionally dependency-light.  It uses the same multilingual
character/word n-gram hashing front-end as ``motion_latent_z.py`` and a small
PyTorch MLP, so Chinese ASR commands are supported without downloading a large
language model.
"""
from __future__ import annotations

import argparse
import re
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

LATENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LATENT_ROOT.parent
DEFAULT_JSON_ROOT = PROJECT_ROOT / "LLM_annotation"
DEFAULT_OUTPUT_DIR = LATENT_ROOT / "moe_router_v3"
if str(LATENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_ROOT))
from motion_latent_z import hashed_text_vector, _normalize_command_text

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Training the shared router requires PyTorch") from exc

DEFAULT_ACTIONS = ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")
ACTION_DESCRIPTIONS = {
    "CrouchedTraverse": {
        "canonical": "Crouched traverse under a bridge. Safely pass below the bridge with a low posture and avoid contact with the obstacle.",
        "aliases": (
            "pass under the bridge", "go beneath the bridge",
            "move under a low bridge", "traverse below the bridge",
            "crouch and pass under the bridge", "walk under the bridge without touching it",
            "duck beneath a low overhead structure", "keep your head down below an overhead structure",
            "crouch below the low overhead obstacle",
            "\u4ece\u6865\u4e0b\u901a\u8fc7", "\u4f4e\u59ff\u901a\u8fc7\u6865\u4e0b", "\u5f2f\u8170\u7a7f\u8fc7\u6865\u4e0b",
            "\u5f00\u59cb\u4ece\u6865\u4e0b\u901a\u8fc7", "\u8e72\u4e0b\u7a7f\u8fc7\u4f4e\u77ee\u6865",
        ),
    },
    "ThreeStepPoleRun": {
        "canonical": "Three-Step Pole Run. Run across three consecutive low poles with continuous alternating foot strikes.",
        "aliases": (
            "run over three consecutive poles", "cross three poles continuously",
            "traverse three low poles", "run across three stepping poles",
            "cross three obstacles in a row", "three pole sprint",
            "run across the shorter sequence of low rails", "clear the short low-rail sequence",
            "\u8de8\u8fc7\u4e09\u6b65\u6869", "\u8dd1\u8fc7\u4e09\u6b65\u6869", "\u8fde\u7eed\u8de8\u8fc7\u4e09\u4e2a\u6869",
            "\u5f00\u59cb\u8de8\u8fc7\u4e09\u6b65\u6869", "\u8d8a\u8fc7\u4e09\u4e2a\u4f4e\u77ee\u969c\u788d",
        ),
    },
    "FiveStepPoleRun": {
        "canonical": "Five-Step Pole Run. Run across five consecutive low poles with continuous alternating foot strikes.",
        "aliases": (
            "run over five consecutive poles", "cross five poles continuously",
            "traverse five low poles", "run across five stepping poles",
            "cross five obstacles in a row", "five pole sprint",
            "run across the longer sequence of low rails", "clear the long low-rail sequence",
            "\u8de8\u8fc7\u4e94\u6b65\u6869", "\u8dd1\u8fc7\u4e94\u6b65\u6869", "\u8fde\u7eed\u8de8\u8fc7\u4e94\u4e2a\u6869",
            "\u5f00\u59cb\u8de8\u8fc7\u4e94\u6b65\u6869", "\u8d8a\u8fc7\u4e94\u4e2a\u4f4e\u77ee\u969c\u788d",
        ),
    },
}


class Router(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int, latent_dim: int, n_actions: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim),
        )
        self.action_head = nn.Linear(latent_dim, n_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.action_head(z)
        return z, logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_task_docs(json_root: Path, action: str) -> list[dict[str, Any]]:
    directory = json_root / action
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    docs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if "report" in path.name.lower() or path.name.startswith("_"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("task"), dict):
            docs.append(doc)
    if not docs:
        raise ValueError(f"No task JSON documents found in {directory}")
    return docs


def document_text(doc: dict[str, Any]) -> str:
    task = doc.get("task", {})
    family = task.get("task_family", [])
    if isinstance(family, str):
        family = [family]
    fields = [
        f"task_name: {task.get('task_name_en', '')}",
        f"task_name_zh: {task.get('task_name_zh', '')}",
        f"task_goal: {task.get('task_goal', '')}",
        f"task_description: {task.get('description', '')}",
        "task_family: " + " . ".join(str(x) for x in family),
    ]
    return " . ".join(item for item in fields if item.split(": ", 1)[-1].strip()).strip()


def build_samples(json_root: Path, actions: tuple[str, ...], text_dim: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    texts: list[str] = []
    sources: list[str] = []
    for action_id, action in enumerate(actions):
        docs = load_task_docs(json_root, action)
        # Use every independent document as provenance, but deduplicate identical
        # task text so the training set is not accidentally weighted by trial count.
        seen: set[str] = set()
        candidates: list[str] = []
        for doc in docs:
            text = document_text(doc)
            if text and text not in seen:
                candidates.append(text); seen.add(text)
        candidates.append(ACTION_DESCRIPTIONS[action]["canonical"])
        candidates.extend(ACTION_DESCRIPTIONS[action]["aliases"])
        # Add compositional English commands in several word orders. These are
        # held out by the evaluation routine below only as surface variants,
        # while the learned model sees diverse lexical realizations.
        if action == "CrouchedTraverse":
            candidates.extend((
                "start and pass underneath the bridge in a crouch",
                "keep low while going under the bridge",
                "safely go below the overhead obstacle",
            ))
        elif action == "ThreeStepPoleRun":
            candidates.extend((
                "start running and clear three poles",
                "quickly step over three poles one after another",
                "perform a continuous three pole crossing",
            ))
        else:
            candidates.extend((
                "start running and clear five poles",
                "quickly step over five poles one after another",
                "perform a continuous five pole crossing",
            ))
        # Deduplicate after augmentation.
        candidates = list(dict.fromkeys(candidates))
        for text in candidates:
            vectors.append(hashed_text_vector(_normalize_command_text(text), text_dim))
            labels.append(action_id); texts.append(text); sources.append(action)
    return np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64), texts, sources


def stratified_group_split(
    labels: np.ndarray,
    seed: int,
    val_ratio: float,
    groups: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by label while keeping normalized-text groups intact.

    A group is one normalized command.  Splitting individual aliases allowed
    equivalent text (for example ``under the bridge``) to land in both train
    and validation, which made the validation score optimistic.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = [str(i) for i in range(len(labels))]
    if len(groups) != len(labels):
        raise ValueError("groups must have one entry per label")
    train: list[int] = []; val: list[int] = []
    for label in sorted(set(labels.tolist())):
        class_indices = np.flatnonzero(labels == label).tolist()
        grouped: dict[str, list[int]] = {}
        for index in class_indices:
            grouped.setdefault(groups[index], []).append(index)
        group_keys = list(grouped)
        rng.shuffle(group_keys)
        target = max(1, int(round(len(class_indices) * val_ratio)))
        if len(group_keys) > 1:
            target = min(target, len(class_indices) - 1)
        selected: list[str] = []
        count = 0
        for key in group_keys:
            if count >= target and selected:
                break
            selected.append(key)
            count += len(grouped[key])
        val_set = {i for key in selected for i in grouped[key]}
        val.extend(sorted(val_set))
        train.extend(i for i in class_indices if i not in val_set)
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)


# Backward-compatible alias for callers that imported the old helper.
stratified_split = stratified_group_split


def explicit_action_cue(query: str) -> str | None:
    """Return an unambiguous action label for common bilingual commands.

    The shared normalizer maps both Chinese and English action phrases to one
    of three canonical ontology tokens.  Count-bearing pole cues deliberately
    take priority over bridge-related words, preventing close three/five-pole
    commands from being sent to the wrong MOE expert.
    """
    normalized = _normalize_command_text(query)
    if "five_step_pole_run" in normalized:
        return "FiveStepPoleRun"
    if "three_step_pole_run" in normalized:
        return "ThreeStepPoleRun"
    if "under_bridge" in normalized:
        return "CrouchedTraverse"

    sequence_object = r"(?:pole|rail|bar|obstacle|barrier|beam)s?"
    if re.search(rf"\b(?:short|shorter)\b.*\b{sequence_object}\b", normalized):
        return "ThreeStepPoleRun"
    if re.search(rf"\b(?:long|longer)\b.*\b{sequence_object}\b", normalized):
        return "FiveStepPoleRun"

    obstacle = r"(?:pole|poles|obstacle|obstacles|step|steps|hurdle|hurdles|barrier|barriers|bar|bars|rail|rails|beam|beams)"
    if re.search(rf"\b(?:five|5)\b.*\b{obstacle}\b", normalized) or re.search(rf"\b{obstacle}\b.*\b(?:five|5)\b", normalized):
        return "FiveStepPoleRun"
    if re.search(rf"\b(?:three|3)\b.*\b{obstacle}\b", normalized) or re.search(rf"\b{obstacle}\b.*\b(?:three|3)\b", normalized):
        return "ThreeStepPoleRun"
    overhead = r"(?:bridge|overhead\s+(?:structure|obstacle)|low\s+(?:structure|beam))"
    if re.search(rf"\b{overhead}\b", normalized) and re.search(r"\b(?:under|underneath|beneath|below|crouch|crouched|duck|ducking|low)\b", normalized):
        return "CrouchedTraverse"
    return None


def evaluate_queries(model: Router, actions: tuple[str, ...], text_dim: int, device: torch.device) -> dict[str, Any]:
    queries = {
        "CrouchedTraverse": (
            "pass beneath the bridge while crouching", "go under the overhead bridge without collision",
            "start and pass under the low bridge", "\u5f00\u59cb\u8e72\u4e0b\u4ece\u6865\u4e0b\u901a\u8fc7",
        ),
        "ThreeStepPoleRun": (
            "clear three poles in succession", "sprint over the three low obstacles",
            "start running and cross three poles", "\u5f00\u59cb\u8fde\u7eed\u8de8\u8fc7\u4e09\u6b65\u6869",
        ),
        "FiveStepPoleRun": (
            "clear five poles in succession", "sprint over the five low obstacles",
            "start running and cross five poles", "\u5f00\u59cb\u8fde\u7eed\u8de8\u8fc7\u4e94\u6b65\u6869",
        ),
    }
    rows = []
    learned_correct = ontology_correct = total = 0
    model.eval()
    with torch.no_grad():
        for expected in actions:
            for query in queries[expected]:
                x = torch.from_numpy(hashed_text_vector(_normalize_command_text(query), text_dim)[None, :]).to(device)
                _, logits = model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                learned_pred = actions[int(np.argmax(probs))]
                cue = explicit_action_cue(query)
                routed_pred = cue if cue in actions else learned_pred
                learned_correct += int(learned_pred == expected)
                ontology_correct += int(routed_pred == expected)
                total += 1
                rows.append({"query": query, "expected_action": expected, "learned_prediction": learned_pred, "routed_prediction": routed_pred, "explicit_cue": cue, "learned_correct": bool(learned_pred == expected), "routed_correct": bool(routed_pred == expected), "probabilities": {a: float(probs[i]) for i, a in enumerate(actions)}})
    learned_accuracy = float(learned_correct / max(total, 1))
    routed_accuracy = float(ontology_correct / max(total, 1))
    return {
        # ``accuracy`` is now the model-only metric; rule-assisted routing is
        # reported separately so it cannot be mistaken for learned skill.
        "accuracy": learned_accuracy,
        "learned_accuracy": learned_accuracy,
        "routed_accuracy": routed_accuracy,
        "ontology_accuracy": routed_accuracy,
        "correct": learned_correct,
        "learned_correct": learned_correct,
        "routed_correct": ontology_correct,
        "total": total,
        "ontology_override": True,
        "metric_primary": "learned_only",
        "evaluation_note": "Queries contain explicit bilingual cues; routed_accuracy is auxiliary deployment performance, not an independent test score.",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--actions", nargs="+", default=list(DEFAULT_ACTIONS))
    # A wider hash space materially reduces bilingual n-gram collisions while
    # keeping the router tiny enough for CPU deployment.
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    actions = tuple(args.actions)
    if tuple(actions) != DEFAULT_ACTIONS:
        # Keep output order explicit and stable for the MOE expert table.
        if set(actions) != set(DEFAULT_ACTIONS):
            raise ValueError(f"actions must be exactly {DEFAULT_ACTIONS}")
    seed_everything(args.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(device_name)
    x, y, texts, sources = build_samples(args.json_root, actions, args.text_dim)
    normalized_groups = [_normalize_command_text(text) for text in texts]
    train_idx, val_idx = stratified_split(y, args.seed, args.val_ratio, normalized_groups)
    train_groups = {normalized_groups[i] for i in train_idx.tolist()}
    val_groups = {normalized_groups[i] for i in val_idx.tolist()}
    leaked_groups = sorted(train_groups & val_groups)
    if leaked_groups:
        raise RuntimeError(f"normalized command leakage across train/val: {leaked_groups[:3]}")
    model = Router(args.text_dim, args.hidden_dim, args.latent_dim, len(actions)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    tx = torch.from_numpy(x[train_idx]).to(device); ty = torch.from_numpy(y[train_idx]).to(device)
    vx = torch.from_numpy(x[val_idx]).to(device); vy = torch.from_numpy(y[val_idx]).to(device)
    best: dict[str, Any] | None = None
    best_acc = -1.0
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        z, logits = model(tx)
        ce = F.cross_entropy(logits, ty)
        zn = F.normalize(z, dim=-1)
        sim = (zn @ zn.T) / max(args.temperature, 1e-4)
        same = ty[:, None].eq(ty[None, :])
        eye = torch.eye(len(ty), device=device, dtype=torch.bool)
        positive = same & ~eye
        contrastive = z.new_zeros(())
        if positive.any():
            # Exclude self from both numerator and denominator.  A batch can
            # contain one sample for a class, so only rows with a positive
            # same-action partner contribute.
            pos_logsum = torch.logsumexp(sim.masked_fill(~positive, float("-inf")), dim=1)
            all_logsum = torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1)
            valid = positive.any(dim=1)
            contrastive = (-(pos_logsum[valid] - all_logsum[valid])).mean()
        class_prototypes = []
        for class_id in range(len(actions)):
            class_prototypes.append(z[ty == class_id].mean(dim=0))
        proto = F.normalize(torch.stack(class_prototypes), dim=-1)
        proto_logits = F.normalize(z, dim=-1) @ proto.T / max(args.temperature, 1e-4)
        proto_ce = F.cross_entropy(proto_logits, ty)
        loss = ce + 0.15 * contrastive + 0.35 * proto_ce
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            vz, vlogits = model(vx)
            pred = vlogits.argmax(dim=1)
            acc = float((pred == vy).float().mean().item())
            val_loss = float(F.cross_entropy(vlogits, vy).item())
        if acc > best_acc or (acc == best_acc and float(loss.detach()) < float((best or {}).get("loss", 1e9))):
            best_acc = acc
            best = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": epoch + 1, "loss": float(loss.detach()), "val_loss": val_loss}
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch={epoch+1:04d} loss={float(loss.detach()):.5f} val_loss={val_loss:.5f} val_acc={acc:.3f}")
    assert best is not None
    model.load_state_dict(best["model_state"]); model.eval()
    with torch.no_grad():
        z_all, logits_all = model(torch.from_numpy(x).to(device))
        z_all = z_all.cpu().numpy().astype(np.float32)
        probs = torch.softmax(logits_all, dim=-1).cpu().numpy().astype(np.float32)
    prototypes = np.stack([z_all[y == i].mean(axis=0) for i in range(len(actions))]).astype(np.float32)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    query_eval = evaluate_queries(model, actions, args.text_dim, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "joint_semantic_router_v1", "actions": list(actions),
        "text_dim": args.text_dim, "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
        "model_state": best["model_state"], "epoch": best["epoch"],
        "best_val_accuracy": best_acc, "seed": args.seed,
        "front_end": "motion_latent_z.hashed_text_vector.multilingual_ngrams_v2",
        "query_evaluation": query_eval,
        "split_protocol": "normalized_text_group_stratified_train_val_v1",
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "normalized_group_overlap": leaked_groups,
        "selection_metric": "val_learned_accuracy",
    }
    torch.save(checkpoint, args.output_dir / "joint_semantic_router.pt")
    np.savez_compressed(args.output_dir / "router_samples.npz", z_sem=z_all, action_id=y, probability=probs, text=np.asarray(texts), source=np.asarray(sources))
    (args.output_dir / "action_prototypes.json").write_text(json.dumps({"format": "joint_action_prototypes_v1", "actions": list(actions), "prototypes": prototypes.tolist(), "canonical_queries": {a: ACTION_DESCRIPTIONS[a]["canonical"] for a in actions}}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "router_manifest.json").write_text(json.dumps({
        "format": "joint_semantic_router_v1",
        "actions": list(actions), "n_samples": len(y),
        "train_samples": len(train_idx), "val_samples": len(val_idx),
        "best_val_accuracy": best_acc,
        "query_evaluation": query_eval,
        "metric_primary": "learned_only",
        "split_protocol": "normalized_text_group_stratified_train_val_v1",
        "train_indices": train_idx.tolist(), "val_indices": val_idx.tolist(),
        "normalized_group_overlap": leaked_groups,
        "selection_metric": "val_learned_accuracy",
        "texts": texts, "normalized_texts": normalized_groups, "sources": sources,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved shared router to {args.output_dir}")
    print(f"samples={len(y)} train={len(train_idx)} val={len(val_idx)} best_val_acc={best_acc:.3f} query_learned_acc={query_eval['learned_accuracy']:.3f} query_routed_acc={query_eval['routed_accuracy']:.3f} device={device}")

if __name__ == "__main__":
    main()





