"""Shared, resumable human-semantic annotation generation utilities."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from human_semantic_schema import (
    HUMAN_SEMANTICS_SCHEMA_VERSION,
    human_metric_evidence,
    human_metric_paths,
    human_semantic_violations,
    normalize_human_semantics,
    should_skip_human_semantic_llm,
)
from robot_semantic_generation_utils import (
    LLMRequestError,
    RequestGate,
    TruncatedLLMResponseError,
    backup_json,
    openai_compatible_chat,
    write_json_atomic,
)


HUMAN_SYSTEM_PROMPT = """
Annotate human motion segments for weak supervision of morphology-invariant
humanoid motion learning.

For a single segment, return exactly one JSON object with only:
{
  "short_caption_en": string,
  "skill_atoms": [string],
  "unknown_atoms": [string],
  "known_absent_atoms": [string],
  "human_derived_control_hints": [string],
  "evidence": [{"atom": string, "source": string}]
}

Allowed atoms:
overhead_clearance, torso_clearance, collision_avoidance,
support_knee_flexion, swing_leg_flexion, swing_leg_advancement,
support_loading, load_acceptance, push_off_preparation,
single_support_balance, double_support_stabilization, weight_transfer.

Evidence rules:
1. Use only explicit Human_Metrics evidence. task_goal and gait_phase are
   context, never evidence.
2. Copy evidence.source exactly from allowed_metric_paths. Every positive
   skill_atoms item needs at least one concrete evidence path.
3. Use the smallest directly supported positive set. If a metric namespace is
   empty, its atoms are unknown, not absent.
4. support_knee_flexion and swing_leg_flexion require human kinematics.
5. swing_leg_advancement requires a human kinematic trajectory or equivalent
   kinematic evidence; step length alone is insufficient.
6. support_loading, load_acceptance and push_off_preparation require contact
   plus human dynamics or another explicit loading/propulsion metric.
7. single_support_balance requires single-support contact plus explicit
   postural or COM-related kinematics.
8. double_support_stabilization may use explicit contact_state evidence.
9. Clearance and collision atoms require explicit human obstacle/proximity/
   clearance evidence. Do not infer them from task names or robot metrics.
10. Do not put an atom in known_absent_atoms unless phase/contact evidence
    makes its absence certain. The local postprocessor is authoritative.
11. In prepared and flight phases with empty human kinematics/dynamics, do
    not invent skill atoms. The local postprocessor will produce the
    deterministic phase/contact result.
12. Keep hints concise and grounded in selected evidence. Return [] when no
    positive atom is selected.
13. Return valid English JSON only. No markdown, comments or extra keys.
""".strip()


def _load_dotenv_like(base_dir: Path) -> None:
    for candidate in (
        base_dir / ".env",
        base_dir / ".env.local",
        base_dir / "openai_api_key.txt",
    ):
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".txt":
            value = candidate.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
            if value:
                os.environ.setdefault("OPENAI_API_KEY", value)
            continue
        for line in candidate.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'"),
            )


def _load_environment() -> None:
    _load_dotenv_like(Path.cwd())
    _load_dotenv_like(Path(__file__).resolve().parent)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line
            for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    start = cleaned.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def build_human_context(
    segment: Mapping[str, Any],
    task_goal: Optional[str],
) -> dict[str, Any]:
    """Build a compact context without sending duplicate nested metrics."""

    return {
        "segment_id": segment.get("segment_id"),
        "gait_phase": segment.get("gait_phase"),
        "task_goal": task_goal,
        "allowed_metric_paths": human_metric_paths(segment),
        "metric_evidence": human_metric_evidence(segment),
    }


def parse_human_response(
    content: str,
    segment: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = _parse_json_object(content)
    return normalize_human_semantics(parsed, segment)


def _existing_is_valid(
    semantics: Any,
    segment: Mapping[str, Any],
) -> bool:
    if not isinstance(semantics, Mapping):
        return False
    if str(semantics.get("schema_version", "")) != HUMAN_SEMANTICS_SCHEMA_VERSION:
        return False
    return not human_semantic_violations(semantics, segment)


def _annotation_from_mapping(
    value: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> dict[str, Any]:
    return normalize_human_semantics(dict(value), segment)


def _call_single_with_retry(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    segment: Mapping[str, Any],
    task_goal: Optional[str],
    retries: int,
    request_timeout: float,
    retry_backoff: float,
    request_gate: RequestGate,
    max_output_tokens: int,
) -> dict[str, Any]:
    user_prompt = json.dumps(
        build_human_context(segment, task_goal),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    output_budget = max(128, int(max_output_tokens))

    for attempt in range(1, retries + 1):
        try:
            request_gate.wait()
            messages = list(base_messages)
            if last_error is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return the complete JSON object again. Correct "
                            "only the previous formatting or validation error. "
                            "Do not add explanations. Previous error: "
                            f"{last_error}"
                        ),
                    }
                )
            response = openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=0.0,
                request_timeout=request_timeout,
                max_output_tokens=output_budget,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("response.choices is missing or empty")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise ValueError("response.choices[0] is not an object")
            finish_reason = str(choice.get("finish_reason", "")).lower()
            if finish_reason in {"length", "max_tokens"}:
                raise TruncatedLLMResponseError(
                    f"endpoint finish_reason={finish_reason}"
                )
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise ValueError("response.choices[0].message is missing")
            content = message.get("content") or "{}"
            if not isinstance(content, str):
                raise ValueError("response message content is not a string")
            return parse_human_response(content, segment)
        except LLMRequestError as error:
            last_error = error
            if attempt >= retries or not error.retryable:
                break
            delay = max(
                error.retry_after or 0.0,
                min(45.0, retry_backoff * (2 ** (attempt - 1))),
            )
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying human segment {segment.get('segment_id')} "
                f"(attempt {attempt + 1}/{retries}) after {error}; "
                f"waiting {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            last_error = error
            if attempt >= retries:
                break
            if isinstance(error, (json.JSONDecodeError, TruncatedLLMResponseError)):
                output_budget = min(4096, max(1000, output_budget * 2))
            delay = min(20.0, retry_backoff * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying human segment {segment.get('segment_id')} after "
                f"{type(error).__name__}; waiting {delay:.1f}s; "
                f"output_budget={output_budget}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to generate human semantic annotation after {retries} "
        f"attempts: {last_error}"
    ) from last_error


def _call_batch_with_retry(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    items: list[tuple[int, Mapping[str, Any]]],
    task_goal: Optional[str],
    retries: int,
    request_timeout: float,
    retry_backoff: float,
    request_gate: RequestGate,
    max_output_tokens: int,
) -> dict[int, dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for request_index, (_, segment) in enumerate(items):
        context = build_human_context(segment, None)
        context["request_index"] = request_index
        contexts.append(context)
    user_prompt = json.dumps(
        {"task_goal": task_goal, "segments": contexts},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    batch_contract = """

Batch mode:
- The input contains a segments list.
- Return exactly {"annotations":[...]}.
- Return exactly one annotation for each input segment.
- Each annotation must contain request_index, short_caption_en,
  skill_atoms, unknown_atoms, known_absent_atoms,
  human_derived_control_hints and evidence.
- Evidence items must contain atom and source; local code reads the exact
  observed value from the source JSON.
- Preserve every integer request_index exactly.
""".strip()
    base_messages = [
        {"role": "system", "content": system_prompt + "\n\n" + batch_contract},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    output_budget = max(
        max(128, int(max_output_tokens)),
        min(4096, 260 * len(items)),
    )

    for attempt in range(1, retries + 1):
        try:
            request_gate.wait()
            messages = list(base_messages)
            if last_error is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return the complete batch JSON again. Correct "
                            "only the previous formatting or validation error. "
                            "Do not add explanations. Previous error: "
                            f"{last_error}"
                        ),
                    }
                )
            response = openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=0.0,
                request_timeout=request_timeout,
                max_output_tokens=output_budget,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("response.choices is missing or empty")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise ValueError("response.choices[0] is not an object")
            finish_reason = str(choice.get("finish_reason", "")).lower()
            if finish_reason in {"length", "max_tokens"}:
                raise TruncatedLLMResponseError(
                    f"endpoint finish_reason={finish_reason}"
                )
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise ValueError("response.choices[0].message is missing")
            content = message.get("content") or "{}"
            if not isinstance(content, str):
                raise ValueError("response message content is not a string")
            parsed = _parse_json_object(content)
            annotations = parsed.get("annotations")
            if not isinstance(annotations, list):
                raise ValueError("batch response annotations must be a list")
            if len(annotations) != len(items):
                raise ValueError(
                    "batch response count mismatch: "
                    f"expected {len(items)}, received {len(annotations)}"
                )

            by_index: dict[int, Mapping[str, Any]] = {}
            for annotation in annotations:
                if not isinstance(annotation, Mapping):
                    raise ValueError("batch annotation must be an object")
                request_index = annotation.get("request_index")
                if isinstance(request_index, bool) or not isinstance(
                    request_index,
                    int,
                ):
                    raise ValueError("request_index must be an integer")
                if request_index in by_index:
                    raise ValueError(
                        f"duplicate request_index: {request_index}"
                    )
                by_index[request_index] = annotation

            results: dict[int, dict[str, Any]] = {}
            for request_index, (segment_index, segment) in enumerate(items):
                annotation = by_index.get(request_index)
                if annotation is None:
                    raise ValueError(
                        f"missing request_index: {request_index}"
                    )
                payload = {
                    key: value
                    for key, value in annotation.items()
                    if key != "request_index"
                }
                results[segment_index] = _annotation_from_mapping(
                    payload,
                    segment,
                )
            return results
        except LLMRequestError as error:
            last_error = error
            if attempt >= retries or not error.retryable:
                break
            delay = max(
                error.retry_after or 0.0,
                min(45.0, retry_backoff * (2 ** (attempt - 1))),
            )
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying human batch ({len(items)} segments; "
                f"attempt {attempt + 1}/{retries}) after {error}; "
                f"waiting {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            last_error = error
            if attempt >= retries:
                break
            if isinstance(error, (json.JSONDecodeError, TruncatedLLMResponseError)):
                output_budget = min(4096, max(1400, output_budget * 2))
            delay = min(20.0, retry_backoff * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying human batch ({len(items)} segments) after "
                f"{type(error).__name__}; waiting {delay:.1f}s; "
                f"output_budget={output_budget}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to generate human semantic batch after {retries} attempts: "
        f"{last_error}"
    ) from last_error


def _generate_batch_resilient(
    *,
    items: list[tuple[int, Mapping[str, Any]]],
    **kwargs: Any,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Split a bad batch so one malformed response does not lose all work."""

    try:
        if len(items) == 1:
            index, segment = items[0]
            return {
                index: _call_single_with_retry(
                    segment=segment,
                    **kwargs,
                )
            }, {}
        return _call_batch_with_retry(items=items, **kwargs), {}
    except RuntimeError as error:
        if len(items) == 1:
            return {}, {items[0][0]: str(error)}
        midpoint = len(items) // 2
        left_results, left_failures = _generate_batch_resilient(
            items=items[:midpoint],
            **kwargs,
        )
        right_results, right_failures = _generate_batch_resilient(
            items=items[midpoint:],
            **kwargs,
        )
        left_results.update(right_results)
        left_failures.update(right_failures)
        return left_results, left_failures


def _stats_to_json(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    for key, value in list(result.items()):
        if isinstance(value, Counter):
            result[key] = dict(sorted(value.items()))
    result["failed_segment_count"] = len(result.get("failed_segments", []))
    result["validation_violation_count"] = len(
        result.get("validation_violations", [])
    )
    return result


def _write_report(
    report_path: Path,
    stats: dict[str, Any],
    backup_dir: Path | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _stats_to_json(stats)
    payload["backup_dir"] = str(backup_dir) if backup_dir else None
    write_json_atomic(report_path, payload)


def _new_stats(
    action: str,
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "annotation_version": HUMAN_SEMANTICS_SCHEMA_VERSION,
        "action": action,
        "model": model,
        "dry_run": bool(dry_run),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials_seen": 0,
        "trials_written": 0,
        "segments_seen": 0,
        "segments_annotated": 0,
        "segments_local": 0,
        "segments_skipped_existing": 0,
        "segments_skipped_missing_metrics": 0,
        "segments_skipped_invalid": 0,
        "segments_failed": 0,
        "segments_cleared_for_regeneration": 0,
        "positive_atom_counts": Counter(),
        "unknown_atom_counts": Counter(),
        "known_absent_atom_counts": Counter(),
        "annotation_state_counts": Counter(),
        "positive_without_evidence": Counter(),
        "failed_segments": [],
        "validation_violations": [],
        "last_checkpoint": None,
        "last_checkpoint_utc": None,
    }


def _record_annotation(
    stats: dict[str, Any],
    semantics: Mapping[str, Any],
) -> None:
    stats["positive_atom_counts"].update(
        str(item) for item in semantics.get("skill_atoms", [])
    )
    stats["unknown_atom_counts"].update(
        str(item) for item in semantics.get("unknown_atoms", [])
    )
    stats["known_absent_atom_counts"].update(
        str(item)
        for item in semantics.get("known_absent_atoms", [])
    )
    stats["annotation_state_counts"].update(
        [str(semantics.get("annotation_state", "unknown"))]
    )
    evidence_atoms = {
        str(item.get("atom"))
        for item in semantics.get("evidence", [])
        if isinstance(item, Mapping)
    }
    stats["positive_without_evidence"].update(
        str(atom)
        for atom in semantics.get("skill_atoms", [])
        if str(atom) not in evidence_atoms
    )


def _prepare_full_regeneration(
    trial_paths: list[Path],
    backup_dir: Path | None,
    stats: dict[str, Any],
    report_path: Path,
) -> set[Path]:
    prepared: set[Path] = set()
    for json_path in trial_paths:
        if not json_path.exists():
            continue
        trial = _read_json(json_path)
        segments = trial.get("segments", [])
        if not isinstance(segments, list):
            continue
        cleared = 0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            human_metrics = segment.get("Human_Metrics")
            if not isinstance(human_metrics, dict):
                continue
            if "human_semantics" in human_metrics:
                human_metrics.pop("human_semantics", None)
                cleared += 1
        if not cleared:
            continue
        if backup_dir is not None:
            backup_json(json_path, backup_dir)
        write_json_atomic(json_path, trial)
        prepared.add(json_path)
        stats["segments_cleared_for_regeneration"] += cleared
        stats["last_checkpoint"] = f"{json_path.name}:cleared"
        stats["last_checkpoint_utc"] = datetime.now(
            timezone.utc,
        ).isoformat()
        _write_report(report_path, stats, backup_dir)
        print(
            f"Prepared {json_path.name}: cleared {cleared} old "
            "human_semantic annotation(s)",
            flush=True,
        )
    return prepared


def _process_trial(
    *,
    json_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    retries: int,
    skip_failed: bool,
    workers: int,
    batch_size: int,
    dry_run: bool,
    backup_dir: Path | None,
    stats: dict[str, Any],
    request_timeout: float,
    retry_backoff: float,
    request_gate: RequestGate,
    skip_valid: bool,
    max_output_tokens: int,
    report_path: Path,
    checkpoint_every: int,
) -> None:
    trial = _read_json(json_path)
    stats["trials_seen"] += 1
    task_goal = (trial.get("task") or {}).get("task_goal")
    segments = trial.get("segments", [])
    if not isinstance(segments, list) or not segments:
        print(f"No segments: {json_path.name}", flush=True)
        return

    pending: list[tuple[int, Mapping[str, Any]]] = []
    local_results: dict[int, dict[str, Any]] = {}
    for index, segment in enumerate(segments):
        stats["segments_seen"] += 1
        if not isinstance(segment, dict):
            stats["segments_skipped_invalid"] += 1
            continue
        human_metrics = segment.get("Human_Metrics")
        if not isinstance(human_metrics, dict):
            stats["segments_skipped_missing_metrics"] += 1
            continue
        existing = human_metrics.get("human_semantics")
        if skip_valid and _existing_is_valid(existing, segment):
            stats["segments_skipped_existing"] += 1
            continue
        if not skip_valid:
            human_metrics.pop("human_semantics", None)
        if should_skip_human_semantic_llm(segment):
            local_results[index] = normalize_human_semantics({}, segment)
        else:
            pending.append((index, segment))

    total_pending = len(pending) + len(local_results)
    print(
        f"Pending {json_path.name}: {total_pending} segment(s), "
        f"api={len(pending)}, local={len(local_results)}, "
        f"batch_size={batch_size}, workers={max(1, workers)}",
        flush=True,
    )

    if dry_run:
        if local_results:
            print(
                f"Dry-run {json_path.name}: {len(local_results)} segment(s) "
                "would be filled by deterministic phase/contact rules",
                flush=True,
            )
        if pending:
            print(
                f"Dry-run {json_path.name}: {len(pending)} segment(s) "
                "would call the LLM; no API request was sent",
                flush=True,
            )
        if not total_pending:
            print(
                f"Dry-run {json_path.name}: no pending segment",
                flush=True,
            )
        return

    backup_created = False
    dirty_since_checkpoint = 0
    completed = 0
    last_segment_id: str | int | None = None
    trial_written = False

    def checkpoint(force: bool = False) -> None:
        nonlocal backup_created, dirty_since_checkpoint, trial_written
        if dry_run or dirty_since_checkpoint <= 0:
            return
        if not force and dirty_since_checkpoint < checkpoint_every:
            return
        if backup_dir is not None and not backup_created:
            backup_json(json_path, backup_dir)
            backup_created = True
        write_json_atomic(json_path, trial)
        if not trial_written:
            stats["trials_written"] += 1
            trial_written = True
        stats["last_checkpoint"] = f"{json_path.name}:{last_segment_id}"
        stats["last_checkpoint_utc"] = datetime.now(
            timezone.utc,
        ).isoformat()
        _write_report(report_path, stats, backup_dir)
        dirty_since_checkpoint = 0

    def apply_result(index: int, semantics: Mapping[str, Any]) -> None:
        nonlocal completed, dirty_since_checkpoint, last_segment_id
        human_metrics = segments[index].get("Human_Metrics")
        if not isinstance(human_metrics, dict):
            return
        human_metrics["human_semantics"] = dict(semantics)
        stats["segments_annotated"] += 1
        _record_annotation(stats, semantics)
        completed += 1
        dirty_since_checkpoint += 1
        last_segment_id = segments[index].get("segment_id", index)

    for index, semantics in sorted(local_results.items()):
        apply_result(index, semantics)
        stats["segments_local"] += 1
        checkpoint()
        print(
            f"Generated {json_path.name}:{segments[index].get('segment_id', index)} "
            f"({completed}/{total_pending}, local phase rule)",
            flush=True,
        )

    if not pending:
        checkpoint(force=True)
        print(f"Updated {json_path.name}: no API segments remain", flush=True)
        return

    batches = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]
    batch_iter = iter(batches)
    future_map: dict[Any, tuple[list[tuple[int, Mapping[str, Any]]], float]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> None:
        try:
            batch = next(batch_iter)
        except StopIteration:
            return
        future = executor.submit(
            _generate_batch_resilient,
            items=batch,
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            task_goal=task_goal,
            retries=retries,
            request_timeout=request_timeout,
            retry_backoff=retry_backoff,
            request_gate=request_gate,
            max_output_tokens=max_output_tokens,
        )
        future_map[future] = (batch, time.monotonic())

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _ in range(min(max(1, workers), len(batches))):
            submit_next(executor)
        while future_map:
            done, _ = wait(
                tuple(future_map),
                timeout=30.0,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                now = time.monotonic()
                longest = max(
                    now - started
                    for _, started in future_map.values()
                )
                print(
                    f"Waiting {json_path.name}: "
                    f"{len(future_map)} active batch(es), "
                    f"longest {longest:.0f}s",
                    flush=True,
                )
                continue
            for future in done:
                batch, started = future_map.pop(future)
                elapsed = time.monotonic() - started
                try:
                    results, failures = future.result()
                except RuntimeError as error:
                    results = {}
                    failures = {index: str(error) for index, _ in batch}
                for index, semantics in sorted(results.items()):
                    apply_result(index, semantics)
                    checkpoint()
                if results:
                    print(
                        f"Progress {json_path.name}: "
                        f"{completed}/{total_pending} completed "
                        f"(+{len(results)}, {elapsed:.1f}s)",
                        flush=True,
                    )
                for index, error_message in sorted(failures.items()):
                    segment_id = segments[index].get("segment_id", index)
                    stats["segments_failed"] += 1
                    failure = {
                        "trial_id": str(
                            trial.get("trial_id", json_path.stem)
                        ),
                        "segment_id": str(segment_id),
                        "error": error_message,
                    }
                    stats["failed_segments"].append(failure)
                    print(
                        f"Skip {json_path.name}:{segment_id} after retries -> "
                        f"{error_message}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not skip_failed:
                        checkpoint(force=True)
                        raise RuntimeError(error_message)
                submit_next(executor)

    checkpoint(force=True)
    print(
        f"Updated {json_path.name}: {completed}/{total_pending} completed",
        flush=True,
    )


def run_generation_cli(
    *,
    action: str,
    json_prefix: str,
    default_json_dir: str,
    default_model: str,
    default_base_url: str,
    default_api_key: str,
    default_report: str,
    system_prompt: str = HUMAN_SYSTEM_PROMPT,
) -> None:
    """CLI shared by all action-specific human semantic generators."""

    _load_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", "--json-dir", default=default_json_dir)
    parser.add_argument("--trial", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", default_model),
    )
    parser.add_argument(
        "--base_url",
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", default_base_url),
    )
    parser.add_argument(
        "--api_key",
        "--api-key",
        default=default_api_key or os.environ.get("OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("OPENAI_RETRIES", "3")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("OPENAI_WORKERS", "4")),
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=int(os.environ.get("OPENAI_BATCH_SIZE", "4")),
    )
    parser.add_argument(
        "--request-timeout",
        "--request_timeout",
        type=float,
        default=float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "180")),
    )
    parser.add_argument(
        "--retry-backoff",
        "--retry_backoff",
        type=float,
        default=float(os.environ.get("OPENAI_RETRY_BACKOFF", "2")),
    )
    parser.add_argument(
        "--min-interval",
        "--min_interval",
        type=float,
        default=float(os.environ.get("OPENAI_MIN_INTERVAL", "0.15")),
    )
    parser.add_argument(
        "--max-output-tokens",
        "--max_output_tokens",
        type=int,
        default=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "900")),
    )
    parser.add_argument(
        "--checkpoint-every",
        "--checkpoint_every",
        type=int,
        default=int(os.environ.get("OPENAI_CHECKPOINT_EVERY", "1")),
    )
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--report", type=Path, default=Path(default_report))
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument(
        "--fail-fast",
        dest="skip_failed",
        action="store_false",
        help="Stop when a segment still fails after retries.",
    )
    parser.add_argument(
        "--skip-failed",
        "--skip_failed",
        dest="skip_failed",
        action="store_true",
        help="Continue after a segment fails after retries.",
    )
    parser.set_defaults(skip_failed=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force",
        "--regenerate-all",
        dest="skip_valid",
        action="store_false",
        help="Regenerate all eligible segments.",
    )
    mode.add_argument(
        "--resume",
        "--skip-valid",
        "--skip_valid",
        dest="skip_valid",
        action="store_true",
        help="Skip existing valid human_semantics and continue failures.",
    )
    parser.set_defaults(skip_valid=False)
    args = parser.parse_args()

    if args.retries < 1 or args.workers < 1 or args.batch_size < 1:
        parser.error("retries, workers and batch-size must be at least 1")
    if args.request_timeout <= 0:
        parser.error("request-timeout must be positive")
    if args.retry_backoff < 0 or args.min_interval < 0:
        parser.error("retry-backoff and min-interval must be non-negative")
    if args.max_output_tokens < 128:
        parser.error("max-output-tokens must be at least 128")
    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be at least 1")
    if not args.dry_run and not args.api_key:
        parser.error(
            "OPENAI_API_KEY is missing. Set it in the environment or pass "
            "--api_key."
        )

    json_dir = Path(args.json_dir)
    if args.trial:
        trial_paths = [
            json_dir / (
                trial if trial.lower().endswith(".json") else trial + ".json"
            )
            for trial in args.trial
        ]
    else:
        trial_paths = [
            path
            for path in sorted(json_dir.glob(f"{json_prefix}_*.json"))
            if path.stem.removeprefix(f"{json_prefix}_").isdigit()
        ]
        if args.limit:
            trial_paths = trial_paths[: args.limit]

    print(
        "Generation mode: "
        + (
            "resume (skip valid annotations)"
            if args.skip_valid
            else "regenerate all eligible segments"
        ),
        flush=True,
    )
    print(
        f"Trials: {len(trial_paths)}; workers={args.workers}; "
        f"batch_size={args.batch_size}; timeout={args.request_timeout}s",
        flush=True,
    )
    if not trial_paths:
        print(f"No matching trial files in {json_dir}", flush=True)
        return

    backup_dir = None
    if not args.no_backup:
        backup_dir = args.backup_dir or (json_dir / "_human_semantics_backup")
    stats = _new_stats(action, args.model, args.dry_run)
    request_gate = RequestGate(args.min_interval)

    if not args.skip_valid and not args.dry_run:
        _prepare_full_regeneration(
            trial_paths,
            backup_dir,
            stats,
            args.report,
        )

    for json_path in trial_paths:
        if not json_path.exists():
            print(f"Missing: {json_path}", flush=True)
            continue
        _process_trial(
            json_path=json_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            system_prompt=system_prompt,
            retries=args.retries,
            skip_failed=args.skip_failed,
            workers=args.workers,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            backup_dir=backup_dir,
            stats=stats,
            request_timeout=args.request_timeout,
            retry_backoff=args.retry_backoff,
            request_gate=request_gate,
            skip_valid=args.skip_valid,
            max_output_tokens=args.max_output_tokens,
            report_path=args.report,
            checkpoint_every=args.checkpoint_every,
        )

    _write_report(args.report, stats, backup_dir)
    print(f"Report: {args.report}", flush=True)
