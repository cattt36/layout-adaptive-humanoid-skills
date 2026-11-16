#!/usr/bin/env python3
"""Audit or conservatively migrate human_semantics without an API call."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from human_semantic_schema import (
    human_semantic_violations,
    normalize_existing_human_semantics,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def audit_trial(path: Path, write_normalized: bool) -> dict[str, Any]:
    trial = read_json(path)
    segments = trial.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"segments is not a list: {path}")

    positive_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    violations: list[str] = []
    normalized_count = 0

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        human_metrics = segment.get("Human_Metrics")
        if not isinstance(human_metrics, dict):
            continue
        raw = human_metrics.get("human_semantics", {})
        if not isinstance(raw, dict):
            raw = {}
        normalized = normalize_existing_human_semantics(raw, segment)
        state_counts[str(normalized.get("annotation_state", "unknown"))] += 1
        for atom in normalized.get("skill_atoms", []):
            positive_counts[str(atom)] += 1
        for problem in human_semantic_violations(raw, segment):
            violations.append(
                f"{segment.get('segment_id', '?')}: {problem}"
            )
        if write_normalized:
            human_metrics["human_semantics"] = normalized
            normalized_count += 1

    if write_normalized:
        write_json(path, trial)

    return {
        "trial_id": trial.get("trial_id", path.stem),
        "path": str(path),
        "segments": len(segments),
        "normalized_segments": normalized_count,
        "positive_atom_counts": dict(sorted(positive_counts.items())),
        "annotation_state_counts": dict(sorted(state_counts.items())),
        "violation_count": len(violations),
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", nargs="+", required=True)
    parser.add_argument(
        "--write-normalized",
        action="store_true",
        help="Overwrite the supplied JSON files after creating a .bak copy.",
    )
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for raw_path in args.json:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if args.write_normalized:
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_bytes(path.read_bytes())
        reports.append(audit_trial(path, args.write_normalized))
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
