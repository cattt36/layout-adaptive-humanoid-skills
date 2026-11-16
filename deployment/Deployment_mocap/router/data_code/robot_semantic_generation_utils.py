"""Utilities shared by the action-specific robot semantic generators."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
import argparse
import json
import os
import random
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib import request
from urllib.error import HTTPError, URLError

from robot_semantic_schema import (
    CANONICAL_ATOMS,
    CONTROL_OBJECTIVES,
    canonical_atom,
    canonical_objective,
    fixed_objective,
    normalize_robot_semantics,
    phase_known_absent_atoms,
)


class LLMRequestError(RuntimeError):
    """A classified failure from an OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.status_code = status_code


class TruncatedLLMResponseError(ValueError):
    """The endpoint stopped before returning a complete JSON object."""


_REQUEST_VARIANT_CACHE: dict[tuple[str, str], int] = {}
_REQUEST_VARIANT_CACHE_LOCK = threading.Lock()


def _retry_after_seconds(error: HTTPError) -> float | None:
    value = error.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, min(120.0, float(value)))
    except (TypeError, ValueError):
        return None


def _http_error_body(error: HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if len(body) > 1000:
        body = body[:1000] + "..."
    return body


def openai_compatible_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    request_timeout: float = 300.0,
    max_output_tokens: int | None = 1200,
) -> dict[str, Any]:
    """Call a chat-completions-compatible endpoint with useful error details."""

    url = base_url.rstrip("/") + "/chat/completions"
    base_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    token_limit = (
        max(128, int(max_output_tokens))
        if max_output_tokens is not None
        else None
    )

    # Compatible gateways differ on output-limit names and JSON mode support.
    # Try a small ordered set and cache the first successful variant.
    model_key = model.lower()
    preferred_token_key = (
        "max_completion_tokens"
        if model_key.startswith(("gpt-5", "o1", "o3", "o4"))
        else "max_tokens"
    )
    if model_key.startswith(("gpt-5", "o1", "o3", "o4")):
        base_payload["reasoning_effort"] = "low"
    alternate_token_key = (
        "max_tokens"
        if preferred_token_key == "max_completion_tokens"
        else "max_completion_tokens"
    )
    variants: list[dict[str, Any]] = []
    for token_key, include_response_format, include_reasoning in (
        (preferred_token_key, True, True),
        (preferred_token_key, False, True),
        (preferred_token_key, True, False),
        (preferred_token_key, False, False),
        (alternate_token_key, True, False),
        (alternate_token_key, False, False),
    ):
        payload_variant = dict(base_payload)
        if not include_response_format:
            payload_variant.pop("response_format", None)
        if not include_reasoning:
            payload_variant.pop("reasoning_effort", None)
        if token_limit is not None:
            payload_variant[token_key] = token_limit
        variants.append(payload_variant)
    payload_variant = dict(base_payload)
    payload_variant.pop("response_format", None)
    payload_variant.pop("temperature", None)
    payload_variant.pop("reasoning_effort", None)
    if token_limit is not None:
        payload_variant[preferred_token_key] = token_limit
    variants.append(payload_variant)
    payload_variant = dict(base_payload)
    payload_variant.pop("response_format", None)
    payload_variant.pop("temperature", None)
    payload_variant.pop("max_tokens", None)
    payload_variant.pop("max_completion_tokens", None)
    variants.append(payload_variant)

    cache_key = (base_url.rstrip("/"), model)
    with _REQUEST_VARIANT_CACHE_LOCK:
        cached_index = _REQUEST_VARIANT_CACHE.get(cache_key)
    start_index = cached_index if cached_index is not None else 0

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    last_error: LLMRequestError | None = None
    for variant_index in range(start_index, len(variants)):
        request_payload = variants[variant_index]
        req = request.Request(
            url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=request_timeout) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as error:
            status = int(error.code)
            body = _http_error_body(error)
            retryable = status in {408, 409, 425, 429} or status >= 500
            detail = f"HTTP {status}"
            if body:
                detail += f": {body}"
            current_error = LLMRequestError(
                detail,
                retryable=retryable,
                retry_after=_retry_after_seconds(error),
                status_code=status,
            )
            last_error = current_error
            if _is_request_compatibility_error(current_error):
                print(
                    "Request parameters rejected; trying a compatible "
                    "fallback payload.",
                    file=sys.stderr,
                )
                continue
            raise current_error from error
        except (
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            OSError,
        ) as error:
            raise LLMRequestError(
                f"{type(error).__name__}: {error}",
                retryable=True,
            ) from error

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LLMRequestError(
                f"Invalid JSON response from endpoint: {error}",
                retryable=True,
            ) from error
        if not isinstance(parsed, dict):
            raise LLMRequestError(
                "Endpoint response is not a JSON object",
                retryable=True,
            )
        with _REQUEST_VARIANT_CACHE_LOCK:
            _REQUEST_VARIANT_CACHE[cache_key] = variant_index
        return parsed

    if last_error is not None:
        raise last_error
    raise LLMRequestError("No compatible request payload was available")


def _is_request_compatibility_error(error: LLMRequestError) -> bool:
    if error.status_code not in {400, 404, 422}:
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "unsupported",
            "unrecognized",
            "unknown parameter",
            "extra inputs",
            "invalid parameter",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "json_object",
            "temperature",
            "reasoning_effort",
        )
    )


class RequestGate:
    """Serialize request starts enough to avoid bursting an API proxy."""

    def __init__(self, min_interval: float = 0.0) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self.min_interval
        if delay:
            time.sleep(delay)


def write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file and replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_dotenv_like(base_dir: Path) -> None:
    for candidate in (
        base_dir / ".env",
        base_dir / ".env.local",
        base_dir / "openai_api_key.txt",
    ):
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".txt":
            value = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            if value:
                os.environ.setdefault("OPENAI_API_KEY", value)
            continue
        for line in candidate.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(
                key.strip(), value.strip().strip('"').strip("'")
            )


def _load_generation_environment() -> None:
    _load_dotenv_like(Path.cwd())
    _load_dotenv_like(Path(__file__).resolve().parent)


def _generate_with_prompt(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    build_context: Callable[[Mapping[str, Any], Optional[str]], Mapping[str, Any]],
    parse_response: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    segment: Mapping[str, Any],
    task_goal: Optional[str],
    retries: int,
    request_timeout: float,
    retry_backoff: float,
    request_gate: RequestGate,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    user_prompt = json.dumps(
        build_context(segment, task_goal),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    output_token_budget = max_output_tokens

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
                            "only the previous validation or formatting error. "
                            "Do not add explanations. "
                            f"Previous error: {last_error}"
                        ),
                    }
                )
            response = openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                request_timeout=request_timeout,
                max_output_tokens=output_token_budget,
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
            return parse_response(content, segment)
        except LLMRequestError as error:
            last_error = error
            if attempt >= retries or not error.retryable:
                break
            delay = max(
                error.retry_after or 0.0,
                min(60.0, retry_backoff * (2 ** (attempt - 1))),
            )
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying segment {segment.get('segment_id')} "
                f"(attempt {attempt + 1}/{retries}) after {error}; "
                f"waiting {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            last_error = error
            if attempt >= retries:
                break
            if isinstance(
                error,
                (json.JSONDecodeError, TruncatedLLMResponseError),
            ):
                current_budget = output_token_budget or 640
                output_token_budget = min(
                    4096,
                    max(1200, current_budget * 2),
                )
            delay = min(30.0, retry_backoff * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying segment {segment.get('segment_id')} after "
                f"{type(error).__name__}: waiting {delay:.1f}s"
                + (
                    f", output budget={output_token_budget}"
                    if output_token_budget is not None
                    else ""
                ),
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to generate robot semantic annotation after {retries} "
        f"attempts: {last_error}"
    )


def _generate_batch_with_prompt(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    build_context: Callable[[Mapping[str, Any], Optional[str]], Mapping[str, Any]],
    parse_response: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    items: list[tuple[int, Mapping[str, Any]]],
    task_goal: Optional[str],
    retries: int,
    request_timeout: float,
    retry_backoff: float,
    request_gate: RequestGate,
    max_output_tokens: int | None,
) -> dict[int, dict[str, Any]]:
    """Generate several segment annotations in one network request."""

    contexts: list[dict[str, Any]] = []
    for request_index, (_, segment) in enumerate(items):
        context = dict(build_context(segment, task_goal))
        context.pop("task_goal", None)
        context["request_index"] = request_index
        contexts.append(context)
    user_prompt = json.dumps(
        {"task_goal": task_goal, "segments": contexts},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    batch_contract = """

Batch mode:
- The user input contains a segments list.
- Return {"annotations":[...]} with exactly one object per input segment.
- Each object must contain request_index, skill_atoms, known_absent_atoms and
  evidence. Each evidence item needs only atom and source; local code fills the
  exact metric value.
- Preserve input order and copy each integer request_index exactly.
""".rstrip()
    base_messages = [
        {"role": "system", "content": system_prompt + batch_contract},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    output_token_budget = max(
        max_output_tokens or 1200,
        min(4096, 320 * len(items)),
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
                            "Return the complete batch JSON again. Correct only "
                            "the previous validation or formatting error. Do not "
                            "add explanations. "
                            f"Previous error: {last_error}"
                        ),
                    }
                )
            response = openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                request_timeout=request_timeout,
                max_output_tokens=output_token_budget,
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

            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(
                    line
                    for line in cleaned.splitlines()
                    if not line.strip().startswith("```")
                ).strip()
            start_index = cleaned.find("{")
            if start_index == -1:
                raise json.JSONDecodeError("No JSON object found", cleaned, 0)
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start_index:])
            if not isinstance(parsed, Mapping):
                raise ValueError("batch response must be a JSON object")
            annotations = parsed.get("annotations")
            if not isinstance(annotations, list):
                raise ValueError("batch response annotations must be a list")
            if len(annotations) != len(items):
                raise ValueError(
                    "batch response count mismatch: "
                    f"expected {len(items)}, received {len(annotations)}"
                )

            by_request_index: dict[int, Mapping[str, Any]] = {}
            for annotation in annotations:
                if not isinstance(annotation, Mapping):
                    raise ValueError("batch annotation must be a JSON object")
                request_index = annotation.get("request_index")
                if isinstance(request_index, bool) or not isinstance(
                    request_index, int
                ):
                    raise ValueError(
                        "batch annotation request_index must be an integer"
                    )
                if request_index in by_request_index:
                    raise ValueError(
                        f"duplicate batch request_index: {request_index}"
                    )
                by_request_index[request_index] = annotation

            result: dict[int, dict[str, Any]] = {}
            for request_index, (segment_index, segment) in enumerate(items):
                annotation = by_request_index.get(request_index)
                if annotation is None:
                    raise ValueError(
                        f"missing batch request_index: {request_index}"
                    )
                result[segment_index] = parse_response(
                    json.dumps(
                        {
                            key: value
                            for key, value in annotation.items()
                            if key != "request_index"
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    segment,
                )
            return result
        except LLMRequestError as error:
            last_error = error
            if attempt >= retries or not error.retryable:
                break
            delay = max(
                error.retry_after or 0.0,
                min(60.0, retry_backoff * (2 ** (attempt - 1))),
            )
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying batch of {len(items)} segments "
                f"(attempt {attempt + 1}/{retries}) after {error}; "
                f"waiting {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            last_error = error
            if attempt >= retries:
                break
            if isinstance(
                error,
                (json.JSONDecodeError, TruncatedLLMResponseError),
            ):
                output_token_budget = min(
                    4096,
                    max(1600, output_token_budget * 2),
                )
            delay = min(30.0, retry_backoff * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, min(1.0, delay * 0.2))
            print(
                f"Retrying batch of {len(items)} segments after "
                f"{type(error).__name__}: waiting {delay:.1f}s, "
                f"output budget={output_token_budget}",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to generate robot semantic batch after {retries} "
        f"attempts: {last_error}"
    ) from last_error


def _generate_batch_resilient(
    *,
    items: list[tuple[int, Mapping[str, Any]]],
    **kwargs: Any,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Split a failed batch until only independently failing segments remain."""

    try:
        if len(items) == 1:
            segment_index, segment = items[0]
            semantics = _generate_with_prompt(
                segment=segment,
                **kwargs,
            )
            return {segment_index: semantics}, {}
        return _generate_batch_with_prompt(items=items, **kwargs), {}
    except RuntimeError as error:
        if len(items) == 1:
            return {}, {items[0][0]: str(error)}
        if isinstance(error.__cause__, LLMRequestError):
            return {}, {segment_index: str(error) for segment_index, _ in items}
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


def _audit_trial(
    *,
    json_path: Path,
    stats: dict[str, Any],
) -> None:
    trial = json.loads(json_path.read_text(encoding="utf-8-sig"))
    stats["trials_seen"] += 1
    segments = trial.get("segments", [])
    if not isinstance(segments, list):
        return
    for index, segment in enumerate(segments):
        stats["segments_seen"] += 1
        if not isinstance(segment, Mapping):
            stats["segments_skipped_invalid"] += 1
            continue
        robot_metrics = segment.get("Retargeted_Robot_Metrics")
        if not isinstance(robot_metrics, Mapping) or not robot_metrics:
            stats["segments_skipped_missing_metrics"] += 1
            continue
        semantics = robot_metrics.get("robot_semantics")
        if not isinstance(semantics, Mapping):
            stats["segments_skipped_invalid"] += 1
            record_validation(
                stats,
                trial_id=str(trial.get("trial_id", json_path.stem)),
                segment_id=str(segment.get("segment_id", index)),
                violations=["robot_semantics is missing"],
            )
            continue
        violations = semantic_validation_violations(semantics, segment)
        record_validation(
            stats,
            trial_id=str(trial.get("trial_id", json_path.stem)),
            segment_id=str(segment.get("segment_id", index)),
            violations=violations,
        )
        stats["segments_annotated"] += 1
        record_annotation(
            stats,
            semantics,
            trial_id=str(trial.get("trial_id", json_path.stem)),
            segment_id=str(segment.get("segment_id", index)),
        )
    print(f"Audited: {json_path.name}")


def _process_generation_trial(
    *,
    json_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    build_context: Callable[[Mapping[str, Any], Optional[str]], Mapping[str, Any]],
    parse_response: Callable[[str, Mapping[str, Any]], dict[str, Any]],
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
    max_output_tokens: int | None,
    progress_report: Path | None,
    checkpoint_every: int,
    backup_prepared: bool = False,
) -> None:
    trial = json.loads(json_path.read_text(encoding="utf-8-sig"))
    stats["trials_seen"] += 1
    task_goal = (trial.get("task") or {}).get("task_goal")
    segments = trial.get("segments", [])
    if not isinstance(segments, list) or not segments:
        print(f"No segments: {json_path.name}")
        return

    pending: list[tuple[int, Mapping[str, Any]]] = []
    repaired_existing = 0
    cleared_existing = 0
    for index, segment in enumerate(segments):
        stats["segments_seen"] += 1
        if not isinstance(segment, Mapping):
            stats["segments_skipped_invalid"] += 1
            continue
        robot_metrics = segment.get("Retargeted_Robot_Metrics")
        if not isinstance(robot_metrics, dict) or not robot_metrics:
            stats["segments_skipped_missing_metrics"] += 1
            continue
        if skip_valid:
            existing = robot_metrics.get("robot_semantics")
            if isinstance(existing, Mapping):
                if not semantic_validation_violations(existing, segment):
                    stats.setdefault("segments_skipped_existing", 0)
                    stats["segments_skipped_existing"] += 1
                    continue
                repaired = repair_existing_semantics(existing, segment)
                if repaired is not None:
                    robot_metrics["robot_semantics"] = repaired
                    repaired_existing += 1
                    stats.setdefault("segments_repaired_existing", 0)
                    stats["segments_repaired_existing"] += 1
                    stats.setdefault("segments_skipped_existing", 0)
                    stats["segments_skipped_existing"] += 1
                    continue
        elif "robot_semantics" in robot_metrics:
            # A full regeneration must not silently retain an old annotation
            # when this segment later fails. Removing it from the in-memory
            # trial also makes --resume retry exactly those failed segments
            # after the first successful checkpoint writes the trial.
            robot_metrics.pop("robot_semantics", None)
            cleared_existing += 1
            stats.setdefault("segments_cleared_for_regeneration", 0)
            stats["segments_cleared_for_regeneration"] += 1
        pending.append((index, segment))

    backup_created = backup_prepared
    trial_written = False
    if cleared_existing and not dry_run:
        if backup_dir is not None:
            backup_json(json_path, backup_dir)
            backup_created = True
        write_json_atomic(json_path, trial)
        stats["trials_written"] += 1
        trial_written = True
        stats["last_checkpoint"] = (
            f"{json_path.name}:cleared_for_regeneration"
        )
        stats["last_checkpoint_utc"] = datetime.now(timezone.utc).isoformat()
        if progress_report is not None:
            write_report(
                progress_report,
                stats,
                backup_dir=backup_dir,
            )
        print(
            f"Cleared {cleared_existing} existing annotation(s) before "
            f"regeneration: {json_path.name}"
        )
    if repaired_existing and not dry_run:
        if backup_dir is not None:
            backup_json(json_path, backup_dir)
            backup_created = True
        write_json_atomic(json_path, trial)
        stats["trials_written"] += 1
        trial_written = True
        if progress_report is not None:
            write_report(
                progress_report,
                stats,
                backup_dir=backup_dir,
            )
        print(
            f"Locally repaired {repaired_existing} existing annotation(s): "
            f"{json_path.name}"
        )

    if not pending:
        print(f"Skipped (no pending segments): {json_path.name}")
        return
    if cleared_existing and dry_run:
        print(
            f"Would regenerate {cleared_existing} existing annotation(s): "
            f"{json_path.name}"
        )

    pending_batches = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]
    pending_iter = iter(pending_batches)
    future_map: dict[
        Any,
        tuple[list[tuple[int, Mapping[str, Any]]], float],
    ] = {}
    completed_pending = 0
    total_pending = len(pending)
    dirty_since_checkpoint = 0
    last_completed_index: int | None = None

    def checkpoint(*, force: bool = False) -> None:
        """Persist completed annotations without writing the trial too often."""

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
        stats["last_checkpoint"] = (
            f"{json_path.name}:"
            f"{segments[last_completed_index].get('segment_id', last_completed_index)}"
            if last_completed_index is not None
            else json_path.name
        )
        stats["last_checkpoint_utc"] = datetime.now(timezone.utc).isoformat()
        if progress_report is not None:
            write_report(
                progress_report,
                stats,
                backup_dir=backup_dir,
            )
        dirty_since_checkpoint = 0

    def submit_next(executor: ThreadPoolExecutor) -> None:
        try:
            batch = next(pending_iter)
        except StopIteration:
            return
        future = executor.submit(
            _generate_batch_resilient,
            items=batch,
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            build_context=build_context,
            parse_response=parse_response,
            task_goal=task_goal,
            retries=retries,
            request_timeout=request_timeout,
            retry_backoff=retry_backoff,
            request_gate=request_gate,
            max_output_tokens=max_output_tokens,
        )
        future_map[future] = (batch, time.monotonic())

    print(
        f"Pending: {json_path.name}: {total_pending} segment(s) in "
        f"{len(pending_batches)} request batch(es), "
        f"batch_size={batch_size}, workers={max(1, workers)}"
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _ in range(min(max(1, workers), len(pending_batches))):
            submit_next(executor)
        while future_map:
            done, _ = wait(
                tuple(future_map),
                timeout=30.0,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                now = time.monotonic()
                longest_wait = max(
                    now - submitted_at
                    for _, submitted_at in future_map.values()
                )
                print(
                    f"Waiting: {json_path.name}: "
                    f"{len(future_map)} active batch(es), "
                    f"longest {longest_wait:.0f}s"
                )
                continue
            for future in done:
                batch, submitted_at = future_map.pop(future)
                elapsed = time.monotonic() - submitted_at
                try:
                    results, failures = future.result()
                except RuntimeError as error:
                    if not skip_failed:
                        checkpoint(force=True)
                        raise
                    failures = {
                        index: str(error)
                        for index, _ in batch
                    }
                    results = {}
                completed_in_batch = 0
                for index, semantics in sorted(results.items()):
                    segment = segments[index]
                    robot_metrics = segment.get("Retargeted_Robot_Metrics")
                    if not isinstance(robot_metrics, dict):
                        continue
                    robot_metrics["robot_semantics"] = semantics
                    stats["segments_annotated"] += 1
                    completed_pending += 1
                    completed_in_batch += 1
                    record_annotation(
                        stats,
                        semantics,
                        trial_id=str(
                            trial.get("trial_id", json_path.stem)
                        ),
                        segment_id=str(
                            segment.get("segment_id", index)
                        ),
                    )
                    if not dry_run:
                        dirty_since_checkpoint += 1
                        last_completed_index = index
                if completed_in_batch:
                    wrote_checkpoint = (
                        not dry_run
                        and dirty_since_checkpoint >= checkpoint_every
                    )
                    checkpoint()
                    last_index = max(results)
                    last_segment = segments[last_index]
                    print(
                        f"{'Checked' if dry_run else 'Checkpointed' if wrote_checkpoint else 'Generated'}: "
                        f"{json_path.name}:"
                        f"{last_segment.get('segment_id', last_index)} "
                        f"(+{completed_in_batch}, "
                        f"{completed_pending}/{total_pending}, "
                        f"{elapsed:.1f}s)"
                    )
                for index, error_message in sorted(failures.items()):
                    segment = segments[index]
                    stats["segments_failed"] += 1
                    if not skip_failed:
                        checkpoint(force=True)
                        raise RuntimeError(error_message)
                    print(
                        f"Skip {json_path.name}:"
                        f"{segment.get('segment_id', index)} after retries -> "
                        f"{error_message}",
                        file=sys.stderr,
                    )
                submit_next(executor)
    # Always persist the final partial batch, including a batch smaller than
    # checkpoint_every. This keeps interruption loss bounded but guarantees
    # that a completed trial is fully written.
    checkpoint(force=True)
    print(f"{'Checked' if dry_run else 'Updated'}: {json_path.name}")


def _prepare_trials_for_full_regeneration(
    *,
    trial_paths: list[Path],
    backup_dir: Path | None,
    stats: dict[str, Any],
    progress_report: Path | None,
) -> set[Path]:
    """Back up and clear all old labels before a resumable full run."""

    prepared_paths: set[Path] = set()
    for json_path in trial_paths:
        if not json_path.exists():
            continue
        trial = json.loads(json_path.read_text(encoding="utf-8-sig"))
        segments = trial.get("segments", [])
        if not isinstance(segments, list):
            continue

        cleared = 0
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            robot_metrics = segment.get("Retargeted_Robot_Metrics")
            if not isinstance(robot_metrics, dict) or not robot_metrics:
                continue
            if "robot_semantics" in robot_metrics:
                robot_metrics.pop("robot_semantics", None)
                cleared += 1

        if not cleared:
            continue
        if backup_dir is not None:
            backup_json(json_path, backup_dir)
        write_json_atomic(json_path, trial)
        prepared_paths.add(json_path)
        stats["segments_cleared_for_regeneration"] += cleared
        stats["trials_prepared_for_regeneration"] += 1
        stats["last_checkpoint"] = (
            f"{json_path.name}:cleared_for_regeneration"
        )
        stats["last_checkpoint_utc"] = datetime.now(timezone.utc).isoformat()
        if progress_report is not None:
            write_report(
                progress_report,
                stats,
                backup_dir=backup_dir,
            )
        print(
            f"Prepared for regeneration: {json_path.name} "
            f"({cleared} old annotation(s) backed up and cleared)"
        )
    return prepared_paths


def run_generation_cli(
    *,
    action: str,
    json_prefix: str,
    default_json_dir: str,
    default_model: str,
    default_base_url: str,
    default_api_key: str,
    default_report: str,
    system_prompt: str,
    build_context: Callable[[Mapping[str, Any], Optional[str]], Mapping[str, Any]],
    parse_response: Callable[[str, Mapping[str, Any]], dict[str, Any]],
) -> None:
    """Run an action-specific generator with bounded, resumable requests."""

    # Load dotenv-style configuration before constructing argparse defaults so
    # OPENAI_API_KEY, OPENAI_MODEL and OPENAI_BASE_URL all take effect.
    _load_generation_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", "--json-dir", default=default_json_dir)
    parser.add_argument(
        "--trial",
        nargs="*",
        default=None,
        help=f"Trial ids such as {json_prefix}_001.",
    )
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
        "--skip_failed",
        "--skip-failed",
        dest="skip_failed",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--fail-fast",
        dest="skip_failed",
        action="store_false",
        help="Stop if one segment still fails after retries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("OPENAI_WORKERS", "2")),
        help="Concurrent API requests per trial.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=int(os.environ.get("OPENAI_BATCH_SIZE", "3")),
        help=(
            "Segments per request. Failed validation batches are split "
            "automatically down to single segments."
        ),
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
        default=float(os.environ.get("OPENAI_MIN_INTERVAL", "0.25")),
        help="Minimum seconds between starting requests.",
    )
    parser.add_argument(
        "--max-output-tokens",
        "--max_output_tokens",
        type=int,
        default=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1200")),
        help=(
            "Initial completion-token budget. JSON truncation automatically "
            "raises it on the next retry."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        "--checkpoint_every",
        type=int,
        default=int(os.environ.get("OPENAI_CHECKPOINT_EVERY", "1")),
        help="Write the updated trial after this many completed segments.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline-audit", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--report", type=Path, default=Path(default_report))
    generation_mode = parser.add_mutually_exclusive_group()
    generation_mode.add_argument(
        "--force",
        "--regenerate-all",
        dest="skip_valid",
        action="store_false",
        help=(
            "Regenerate every eligible segment, including existing valid "
            "annotations. This is the default."
        ),
    )
    generation_mode.add_argument(
        "--resume",
        "--skip-valid",
        "--skip_valid",
        dest="skip_valid",
        action="store_true",
        help=(
            "Resume by skipping existing annotations that pass local "
            "validation."
        ),
    )
    parser.set_defaults(skip_valid=False)
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.retry_backoff < 0 or args.min_interval < 0:
        parser.error("--retry-backoff and --min-interval must be non-negative")
    if args.max_output_tokens < 128:
        parser.error("--max-output-tokens must be at least 128")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")

    # Keep an explicit --api-key value authoritative while retaining the
    # environment-backed fallback loaded before argparse.
    args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not args.offline_audit and not args.api_key:
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
        )
    )
    if trial_paths:
        print(
            f"Trials: {trial_paths[0].name} -> {trial_paths[-1].name} "
            f"({len(trial_paths)} file(s))"
        )

    backup_dir = None
    if not args.no_backup:
        backup_dir = args.backup_dir or (json_dir / "_robot_semantics_backup")
    stats = new_stats(
        action,
        dry_run=args.dry_run or args.offline_audit,
        model=args.model if not args.offline_audit else "offline-audit",
    )
    request_gate = RequestGate(args.min_interval)
    prepared_paths: set[Path] = set()
    if not args.skip_valid and not args.dry_run and not args.offline_audit:
        prepared_paths = _prepare_trials_for_full_regeneration(
            trial_paths=trial_paths,
            backup_dir=backup_dir,
            stats=stats,
            progress_report=args.report,
        )
        if prepared_paths:
            print(
                "Full-regeneration preparation complete. If generation is "
                "interrupted, rerun with --resume."
            )

    for json_path in trial_paths:
        if not json_path.exists():
            print(f"Missing: {json_path}")
            continue
        if args.offline_audit:
            _audit_trial(json_path=json_path, stats=stats)
            continue
        _process_generation_trial(
            json_path=json_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            system_prompt=system_prompt,
            build_context=build_context,
            parse_response=parse_response,
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
            progress_report=args.report,
            checkpoint_every=args.checkpoint_every,
            backup_prepared=json_path in prepared_paths,
        )
    write_report(args.report, stats, backup_dir=backup_dir)
    print(f"Report: {args.report}")


ATOM_EVIDENCE_GUIDE = """
Canonical atom evidence guide:
- overhead_clearance: use a numeric overhead/bridge clearance path in
  Retargeted_Robot_Metrics.obstacle_relation. Task name and phase are not
  enough.
- torso_clearance: use a numeric torso/chest-to-obstacle clearance path in
  Retargeted_Robot_Metrics.obstacle_relation.
- collision_avoidance: use explicit collision status, collision count,
  penetration, obstacle distance, or clearance evidence. A collision_risk tag
  alone is feasibility evidence, not proof of the skill atom.
- support_knee_flexion: use support-knee kinematics or a clearly named
  support-knee moment/trajectory.
- swing_leg_flexion: use swing-knee/hip/ankle kinematics or trajectory.
- swing_leg_advancement: use swing-limb trajectory, advancement, or an
  unambiguous take-off/flight event plus kinematic evidence.
- support_loading: use support-chain force/torque or explicit support loading
  evidence. Do not infer it from phase alone.
- load_acceptance: use contact transition and force/torque evidence.
- push_off_preparation: use an explicit take-off phase and preferably
  support-chain or actuator dynamics evidence.
- single_support_balance: use support-side/contact and balance, force,
  wrench, or support-chain evidence.
- double_support_stabilization: use double-support contact/support evidence
  and, when available, force or wrench evidence.
- weight_transfer: use an explicit support transition together with contact,
  support-side, force, or torque evidence.

Evidence source rules:
- Every positive atom needs at least one evidence item.
- `source` must be a concrete path beginning with
  `Retargeted_Robot_Metrics.` or the exact string `gait_phase`.
- The value must quote an observed value, status, ratio, or event from the
  supplied input. Never invent thresholds or values.
- Do not cite `task_goal`, `action_context`, `npz_alignment`, or an entire
  namespace as evidence.
"""


STRICT_EVIDENCE_POLICY = """
Evidence requirements:
1. Every positive skill atom must have at least one evidence item. The
   evidence item must cite a concrete JSON path under
   Retargeted_Robot_Metrics, or the exact gait_phase only for
   push_off_preparation during an explicit left_foot_take_off or
   right_foot_take_off phase. Include the observed value or state.
   Valid source examples are
   `Retargeted_Robot_Metrics.kinematics.<metric>`,
   `Retargeted_Robot_Metrics.dynamics.<metric>`,
   `Retargeted_Robot_Metrics.feasibility.<metric>`,
   `Retargeted_Robot_Metrics.obstacle_relation.<metric>`, and
   `gait_phase`. Do not cite an entire namespace without a metric path.
   The user input contains `allowed_metric_paths`, which is the complete
   whitelist of concrete metric paths present in this segment. For every
   metric-backed evidence item, `source` must be copied exactly from that
   whitelist. Never construct, shorten, rename, or infer a path that is not
   listed there. If no suitable path is listed, leave the atom out of
   skill_atoms and known_absent_atoms; the local parser will mark it unknown.
   `gait_phase` is not a metric path and may be used only under the
   push_off_preparation exception above.
2. Do not use task name, task goal, action name, or generic common sense as
   evidence. They may narrow the candidate set but cannot prove an atom.
3. A missing or empty metric namespace is unknown. Never convert missing data
   into known_absent, safe, collision_free, or contact_consistent.
4. Separate skill from feasibility. A skill atom describes what the motion is
   doing; feasibility_tags describe whether the retargeted execution is safe.
   A risk tag does not prove a skill atom, and a skill atom does not prove
   feasibility.
5. Use numeric evidence when available: cite the metric path and a compact
   value such as min/max/mean, ratio, status, or event name. Do not invent
   thresholds or values.
6. The three atom lists must be disjoint. Use known_absent only when the
   supplied evidence clearly contradicts the atom. You do not need to
   enumerate all unknown atoms; any candidate omitted from skill_atoms and
   known_absent_atoms will be marked unknown by the local parser.
7. Keep skill_atoms minimal. Do not output synonyms, free-form objectives,
   phase names, or every atom that could be associated with the task.
"""

EVIDENCE_ROOTS = (
    "Retargeted_Robot_Metrics.kinematics.",
    "Retargeted_Robot_Metrics.dynamics.",
    "Retargeted_Robot_Metrics.feasibility.",
    "Retargeted_Robot_Metrics.obstacle_relation.",
)


def compact_robot_semantic_prompt(
    *,
    action_context: str,
    candidate_atoms: tuple[str, ...] | list[str],
) -> str:
    """Build the concise output contract shared by all action generators."""

    ontology = ", ".join(candidate_atoms)
    return f"""
Annotate one retargeted humanoid-robot motion segment.

Return exactly one JSON object with only these fields:
{{
  "skill_atoms": [string],
  "known_absent_atoms": [string],
  "evidence": [
    {{"atom": string, "source": string, "value": string}}
  ]
}}

Allowed skill atoms:
{ontology}

Rules:
1. Select the smallest set of skill_atoms directly supported by the supplied
   metric_evidence. Every positive atom needs at least one evidence item.
2. Copy evidence.source exactly from a metric_evidence key. Use a short
   observed value in evidence.value. Do not invent paths, values or thresholds.
3. The only non-metric evidence allowed is source="gait_phase" for
   push_off_preparation during left_foot_take_off or right_foot_take_off.
4. Use known_absent_atoms only when a supplied metric explicitly contradicts
   an atom. Omit undecidable atoms; local code marks them unknown.
5. Missing or empty metrics mean unknown, never absent or safe. Task names and
   general knowledge are context, not evidence.
6. Clearance atoms require obstacle-distance/clearance metrics. Collision
   avoidance requires collision or obstacle metrics. Limb-motion atoms
   require relevant kinematics. Loading, balance and transfer require
   dynamics or support-chain metrics.
7. Do not return unknown_atoms, control objectives, feasibility tags,
   explanations, markdown or extra keys. Local deterministic code creates
   those fields.

Action context:
{action_context.strip()}
""".strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def metric_paths_for_prompt(segment: Mapping[str, Any]) -> list[str]:
    """List concrete metric leaves the model is allowed to cite.

    The list is generated from the actual segment instead of being hard-coded,
    so missing kinematics or obstacle metrics remain visibly unavailable.
    """

    result: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}")
        elif isinstance(value, (bool, int, float, str)):
            result.append(path)

    metrics = robot_metrics_for_prompt(segment)
    for namespace in ("kinematics", "dynamics", "feasibility", "obstacle_relation"):
        if namespace in metrics:
            walk(metrics[namespace], f"Retargeted_Robot_Metrics.{namespace}")
    return sorted(result)


_PROMPT_IGNORED_LEAVES = {
    "min_time_s",
    "max_time_s",
    "peak_time_s",
    "peak_frame",
}

_PROMPT_DYNAMICS_IGNORED_LEAVES = {
    "min",
}


def metric_evidence_for_prompt(
    segment: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten semantic metric leaves so each path and value is sent once."""

    result: dict[str, Any] = {}

    def compact_value(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 6)
        return value

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in _PROMPT_IGNORED_LEAVES:
                    continue
                if (
                    path.startswith("Retargeted_Robot_Metrics.dynamics.")
                    and str(key) in _PROMPT_DYNAMICS_IGNORED_LEAVES
                ):
                    continue
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}")
        elif isinstance(value, (bool, int, float, str)):
            result[path] = compact_value(value)

    metrics = robot_metrics_for_prompt(segment)
    for namespace in ("kinematics", "dynamics", "obstacle_relation"):
        if namespace in metrics:
            walk(metrics[namespace], f"Retargeted_Robot_Metrics.{namespace}")

    feasibility = _mapping(metrics.get("feasibility"))
    torque = _mapping(feasibility.get("torque_screening"))
    support = _mapping(torque.get("support_chain"))
    if support:
        support = {
            key: support[key]
            for key in (
                "label",
                "support_sides",
                "peak_limit_ratio",
                "violation_frame_ratio",
                "any_limit_violation",
            )
            if key in support
        }
        walk(
            support,
            "Retargeted_Robot_Metrics.feasibility."
            "torque_screening.support_chain",
        )
    validity = _mapping(feasibility.get("state_validity"))
    collision = _mapping(validity.get("collision"))
    if collision:
        collision = {
            key: collision[key]
            for key in (
                "status",
                "floor_contact_frame_count",
                "max_floor_penetration_mm",
                "self_collision_frame_count",
                "max_self_penetration_mm",
            )
            if key in collision
        }
        walk(
            collision,
            "Retargeted_Robot_Metrics.feasibility."
            "state_validity.collision",
        )
    return dict(sorted(result.items()))


def _has_nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def derive_feasibility_tags(
    segment: Mapping[str, Any],
    *,
    fallback: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Derive feasibility tags from explicit retargeting metrics.

    Feasibility is not a semantic interpretation task.  When the exporter
    provides a clear status/label, using it directly prevents LLM wording
    differences from becoming training-label noise.  The fallback is used only
    when a family is genuinely unavailable.
    """

    robot = _mapping(segment.get("Retargeted_Robot_Metrics"))
    feasibility = _mapping(robot.get("feasibility"))
    torque = _mapping(feasibility.get("torque_screening"))
    whole = _mapping(torque.get("whole_body"))
    support = _mapping(torque.get("support_chain"))
    validity = _mapping(feasibility.get("state_validity"))
    joint = _mapping(validity.get("joint_limit"))
    collision = _mapping(validity.get("collision"))
    joint_status = str(joint.get("status", "")).lower()
    if "violation" in joint_status or bool(joint.get("any_limit_violation")):
        joint_tag = "joint_limit_risk"
    elif joint_status in {"within_limits", "within_limit", "ok", "clear"}:
        joint_tag = "joint_limits_satisfied"
    else:
        joint_tag = "joint_limits_unverified"

    whole_label = str(whole.get("label", "")).lower()
    if "exceed" in whole_label or bool(whole.get("any_limit_violation")):
        actuator_tag = "actuator_load_risk"
    elif "near" in whole_label:
        actuator_tag = "actuator_load_risk"
    elif "within" in whole_label or whole_label in {"ok", "clear"}:
        actuator_tag = "actuator_limits_satisfied"
    else:
        actuator_tag = "actuator_feasibility_unverified"

    collision_status = str(collision.get("status", "")).lower()
    if "collision" in collision_status or "penetrat" in collision_status:
        collision_tag = "collision_risk"
    elif collision_status in {"clear", "within_limits", "ok"}:
        collision_tag = "collision_free_verified"
    else:
        collision_tag = "clearance_unverified"

    # support_chain is an explicit support-side/torque assessment when it was
    # evaluated. It is not inferred from the action name or phase.
    support_sides = support.get("support_sides")
    support_label = str(support.get("label", "")).lower()
    if (
        isinstance(support_sides, (list, tuple))
        and support_sides
        and "not_evaluated" not in support_label
    ):
        contact_tag = "contact_consistent"
    elif "mismatch" in support_label:
        contact_tag = "contact_mismatch"
    else:
        contact_tag = "contact_unverified"

    return [joint_tag, contact_tag, actuator_tag, collision_tag]


def objective_violations(semantics: Mapping[str, Any]) -> list[str]:
    """Reject free-form control objectives in newly generated annotations."""

    violations: list[str] = []
    raw = semantics.get("robot_control_objectives")
    if raw is None:
        raw = semantics.get("control_objectives", [])
    if isinstance(raw, str):
        raw = [raw]
    for item in raw or []:
        token = str(item).strip()
        if token not in CONTROL_OBJECTIVES:
            violations.append(f"invalid control objective: {token}")
    return violations


def derive_control_objectives(skill_atoms: Any) -> list[str]:
    """Map canonical skill atoms to the fixed objective vocabulary locally."""

    positives = {canonical_atom(item) for item in skill_atoms or []}
    objectives: list[str] = []
    for atom in CANONICAL_ATOMS:
        if atom not in positives:
            continue
        objective = canonical_objective(atom)
        if objective is not None and objective not in objectives:
            objectives.append(objective)
    return objectives


def robot_metrics_for_prompt(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fresh robot measurements for an annotation request.

    Existing ``robot_semantics`` is deliberately excluded. Sending a previous
    model answer back to the model creates label leakage and makes a prompt
    revision ineffective. NPZ alignment is also bookkeeping, not evidence.
    """

    value = segment.get("Retargeted_Robot_Metrics", {})
    if not isinstance(value, Mapping):
        return {}
    allowed = ("kinematics", "dynamics", "feasibility", "obstacle_relation")
    return {
        key: value[key]
        for key in allowed
        if key in value
        and isinstance(value[key], (Mapping, list, tuple, bool, int, float))
    }


def _resolve_metric_path(segment: Mapping[str, Any], source: str) -> Any:
    if source == "gait_phase":
        return segment.get("gait_phase", segment.get("phase"))
    prefix = "Retargeted_Robot_Metrics."
    if not source.startswith(prefix):
        return None
    value: Any = segment.get("Retargeted_Robot_Metrics", {})
    for key in source[len(prefix) :].split("."):
        if isinstance(value, Mapping):
            if key not in value:
                return None
            value = value[key]
        elif isinstance(value, (list, tuple)):
            try:
                index = int(key)
            except (TypeError, ValueError):
                return None
            if index < 0 or index >= len(value):
                return None
            value = value[index]
        else:
            return None
    return value


def evidence_violations(
    semantics: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> list[str]:
    """Check that evidence paths point to values in the supplied segment."""

    violations: list[str] = []
    active_atoms = set(CANONICAL_ATOMS)
    positive = {
        canonical_atom(item)
        for item in semantics.get("skill_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    phase = str(segment.get("gait_phase", segment.get("phase", ""))).strip()
    allowed_paths = set(metric_paths_for_prompt(segment))
    for item in semantics.get("evidence", []):
        if not isinstance(item, Mapping):
            continue
        atom = canonical_atom(item.get("atom", ""))
        if atom not in active_atoms:
            continue
        source = str(item.get("source", "")).strip()
        resolved = _resolve_metric_path(segment, source)
        if atom not in positive:
            continue
        allowed_source = source == "gait_phase" or source.startswith(EVIDENCE_ROOTS)
        if not allowed_source:
            violations.append(f"evidence source is not allowed: {source}")
            continue
        if source == "gait_phase" and (
            atom != "push_off_preparation"
            or phase not in {"left_foot_take_off", "right_foot_take_off"}
        ):
            violations.append(
                f"gait_phase cannot be evidence for {atom} in phase {phase}"
            )
            continue
        if source != "gait_phase" and source not in allowed_paths:
            violations.append(
                f"evidence source is not present in allowed_metric_paths: {source}"
            )
            continue
        if resolved is None:
            violations.append(
                f"evidence path does not exist for {atom}: {source}"
            )
            continue
        if isinstance(resolved, Mapping) and not resolved:
            violations.append(
                f"evidence path is empty for {atom}: {source}"
            )
        elif isinstance(resolved, Mapping):
            violations.append(
                f"evidence path must point to a metric leaf for {atom}: {source}"
            )
        if isinstance(resolved, (list, tuple)) and not resolved:
            violations.append(
                f"evidence path is empty for {atom}: {source}"
            )
        elif isinstance(resolved, (list, tuple)):
            violations.append(
                f"evidence path must point to a metric leaf for {atom}: {source}"
            )
    return violations


def atom_evidence_violations(
    semantics: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> list[str]:
    """Check that each positive atom uses a plausible metric family.

    This is deliberately a family-level check, not a numeric threshold
    detector. It prevents a feasibility status or a phase name from being
    reused as proof of an unrelated motion atom.
    """

    active_atoms = set(CANONICAL_ATOMS)
    evidence_by_atom: dict[str, list[str]] = {}
    for item in semantics.get("evidence", []):
        if not isinstance(item, Mapping):
            continue
        atom = canonical_atom(item.get("atom", ""))
        if atom not in active_atoms:
            continue
        evidence_by_atom.setdefault(atom, []).append(
            str(item.get("source", "")).strip()
        )

    positive = {
        canonical_atom(item)
        for item in semantics.get("skill_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    violations: list[str] = []

    def has_prefix(atom: str, prefixes: tuple[str, ...]) -> bool:
        return any(
            source.startswith(prefix_item)
            for source in evidence_by_atom.get(atom, [])
            for prefix_item in prefixes
        )

    def has_token(atom: str, tokens: tuple[str, ...]) -> bool:
        return any(
            any(token in source.lower() for token in tokens)
            for source in evidence_by_atom.get(atom, [])
        )

    for atom in sorted(positive):
        if atom in {"overhead_clearance", "torso_clearance"}:
            ok = has_prefix(atom, ("Retargeted_Robot_Metrics.obstacle_relation.",))
        elif atom == "collision_avoidance":
            ok = has_prefix(
                atom,
                (
                    "Retargeted_Robot_Metrics.feasibility.",
                    "Retargeted_Robot_Metrics.obstacle_relation.",
                ),
            )
        elif atom in {
            "support_knee_flexion",
            "swing_leg_flexion",
            "swing_leg_advancement",
        }:
            ok = has_prefix(
                atom,
                (
                    "Retargeted_Robot_Metrics.kinematics.",
                    "Retargeted_Robot_Metrics.dynamics.",
                    "Retargeted_Robot_Metrics.obstacle_relation.",
                ),
            )
        elif atom in {
            "support_loading",
            "load_acceptance",
            "single_support_balance",
            "double_support_stabilization",
            "weight_transfer",
        }:
            ok = has_prefix(
                atom,
                (
                    "Retargeted_Robot_Metrics.dynamics.",
                    "Retargeted_Robot_Metrics.feasibility.torque_screening.support_chain.",
                ),
            )
        elif atom == "push_off_preparation":
            ok = "gait_phase" in evidence_by_atom.get(atom, []) or has_prefix(
                atom,
                (
                    "Retargeted_Robot_Metrics.dynamics.",
                    "Retargeted_Robot_Metrics.feasibility.torque_screening.support_chain.",
                ),
            )
        else:
            ok = bool(evidence_by_atom.get(atom))
        if not ok:
            violations.append(
                f"evidence family is implausible for positive atom: {atom}"
            )
    return violations


def feasibility_tag_mismatches(
    semantics: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> list[str]:
    """Report feasibility labels that disagree with deterministic metrics."""

    observed = [str(item) for item in semantics.get("feasibility_tags", [])]
    expected = derive_feasibility_tags(segment)
    if observed == expected:
        return []
    return [
        "feasibility_tags disagree with deterministic metrics: "
        f"observed={observed}, expected={expected}"
    ]


def semantic_validation_violations(
    semantics: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> list[str]:
    """Validate an existing annotation without contacting an LLM."""

    violations = annotation_violations(semantics)
    violations.extend(evidence_violations(semantics, segment))
    violations.extend(atom_evidence_violations(semantics, segment))
    violations.extend(objective_violations(semantics))
    violations.extend(feasibility_tag_mismatches(semantics, segment))
    return violations


def record_validation(
    stats: dict[str, Any],
    *,
    trial_id: str,
    segment_id: str,
    violations: list[str],
) -> None:
    if not violations:
        return
    stats["validation_violations"].append(
        {
            "trial_id": trial_id,
            "segment_id": segment_id,
            "violations": list(violations),
        }
    )


def new_stats(action: str, *, dry_run: bool, model: str) -> dict[str, Any]:
    return {
        "annotation_version": "robot_semantics_v2",
        "action": action,
        "model": model,
        "dry_run": bool(dry_run),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials_seen": 0,
        "trials_written": 0,
        "segments_seen": 0,
        "segments_annotated": 0,
        "segments_skipped_missing_metrics": 0,
        "segments_skipped_invalid": 0,
        "segments_failed": 0,
        "segments_repaired_existing": 0,
        "segments_cleared_for_regeneration": 0,
        "trials_prepared_for_regeneration": 0,
        "positive_atom_counts": Counter(),
        "unknown_atom_counts": Counter(),
        "known_absent_atom_counts": Counter(),
        "feasibility_tag_counts": Counter(),
        "partition_violations": [],
        "positive_without_evidence": Counter(),
        "evidence_path_violations": [],
        "validation_violations": [],
        "positive_evidence_counts": Counter(),
    }


def record_annotation(
    stats: dict[str, Any],
    semantics: Mapping[str, Any],
    *,
    trial_id: str,
    segment_id: str,
) -> None:
    active_atoms = set(CANONICAL_ATOMS)
    positive = {
        canonical_atom(item)
        for item in semantics.get("skill_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    unknown = {
        canonical_atom(item)
        for item in semantics.get("unknown_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    absent = {
        canonical_atom(item)
        for item in semantics.get("known_absent_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    candidates = active_atoms
    overlap = (positive & unknown) | (positive & absent) | (unknown & absent)
    covered = positive | unknown | absent
    if overlap or covered != candidates:
        stats["partition_violations"].append(
            {
                "trial_id": trial_id,
                "segment_id": segment_id,
                "overlap": sorted(overlap),
                "missing": sorted(candidates - covered),
                "extra": sorted(covered - candidates),
            }
        )

    stats["positive_atom_counts"].update(sorted(positive))
    stats["unknown_atom_counts"].update(sorted(unknown))
    stats["known_absent_atom_counts"].update(sorted(absent))
    stats["feasibility_tag_counts"].update(
        str(item) for item in semantics.get("feasibility_tags", [])
    )

    evidence_atoms = {
        canonical_atom(item.get("atom"))
        for item in semantics.get("evidence", [])
        if isinstance(item, Mapping)
        and canonical_atom(item.get("atom")) in active_atoms
    }
    stats["positive_without_evidence"].update(
        sorted(positive - evidence_atoms)
    )
    stats["positive_evidence_counts"].update(
        atom for atom in positive if atom in evidence_atoms
    )


def annotation_violations(
    semantics: Mapping[str, Any],
    *,
    require_evidence: bool = True,
) -> list[str]:
    active_atoms = set(CANONICAL_ATOMS)
    positive = {
        canonical_atom(item)
        for item in semantics.get("skill_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    unknown = {
        canonical_atom(item)
        for item in semantics.get("unknown_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    absent = {
        canonical_atom(item)
        for item in semantics.get("known_absent_atoms", [])
        if canonical_atom(item) in active_atoms
    }
    candidates = active_atoms
    violations: list[str] = []
    if positive & unknown:
        violations.append("skill_atoms overlaps unknown_atoms")
    if positive & absent:
        violations.append("skill_atoms overlaps known_absent_atoms")
    if unknown & absent:
        violations.append("unknown_atoms overlaps known_absent_atoms")
    covered = positive | unknown | absent
    if candidates - covered:
        violations.append(
            "missing atom partition: " + ", ".join(sorted(candidates - covered))
        )
    if covered - candidates:
        violations.append(
            "unknown atom names: " + ", ".join(sorted(covered - candidates))
        )
    if len(semantics.get("feasibility_tags", [])) != 4:
        violations.append("feasibility_tags must contain exactly four tags")
    if require_evidence and positive:
        evidence_atoms = {
            canonical_atom(item.get("atom"))
            for item in semantics.get("evidence", [])
            if isinstance(item, Mapping)
            and canonical_atom(item.get("atom")) in active_atoms
        }
        missing_evidence = positive - evidence_atoms
        if missing_evidence:
            violations.append(
                "positive atoms without evidence: "
                + ", ".join(sorted(missing_evidence))
            )
    evidence_atoms = set()
    for item in semantics.get("evidence", []):
        if not isinstance(item, Mapping):
            violations.append("evidence items must be objects")
            continue
        atom = canonical_atom(item.get("atom", ""))
        if atom not in active_atoms:
            continue
        source = str(item.get("source", "")).strip()
        value = str(item.get("value", "")).strip()
        evidence_atoms.add(atom)
        if atom not in positive:
            violations.append(f"evidence atom is not positive: {atom}")
        if not source or not value:
            violations.append(f"evidence for {atom} must include source and value")
        elif source != "gait_phase" and not source.startswith(EVIDENCE_ROOTS):
            violations.append(f"invalid evidence source for {atom}: {source}")
        elif source != "gait_phase" and source.count(".") < 2:
            violations.append(f"evidence source is not a concrete metric path: {source}")
        if source in {
            "Retargeted_Robot_Metrics.npz_alignment",
            "Retargeted_Robot_Metrics.task_goal",
        } or "npz_alignment" in source:
            violations.append(f"bookkeeping path cannot be evidence: {source}")
    return violations


def normalize_generated_semantics(
    parsed: Mapping[str, Any],
    segment: Mapping[str, Any],
    *,
    candidate_atoms: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Normalize one new LLM response and enforce the label contract.

    Objective vocabulary and evidence are checked against the raw response
    before compatibility normalization. Feasibility tags are deterministic
    labels from the supplied retargeting metrics, so the LLM output is
    intentionally overwritten for that field.
    """

    required = {"skill_atoms", "known_absent_atoms", "evidence"}
    missing = sorted(required.difference(parsed))
    if missing:
        raise ValueError(
            "LLM JSON is missing required fields: " + ", ".join(missing)
        )

    parsed_with_values = dict(parsed)
    evidence_with_values: list[dict[str, Any]] = []
    for item in parsed.get("evidence", []):
        if not isinstance(item, Mapping):
            evidence_with_values.append(dict(item) if isinstance(item, dict) else item)
            continue
        evidence_item = dict(item)
        source = str(evidence_item.get("source", "")).strip()
        observed = evidence_item.get(
            "value",
            evidence_item.get("observation", evidence_item.get("claim")),
        )
        if (observed is None or str(observed).strip() == "") and source:
            resolved = _resolve_metric_path(segment, source)
            if isinstance(resolved, (bool, int, float, str)):
                evidence_item["value"] = json.dumps(
                    resolved,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        evidence_with_values.append(evidence_item)
    parsed_with_values["evidence"] = evidence_with_values

    normalized = normalize_robot_semantics(
        parsed_with_values,
        candidate_atoms=candidate_atoms,
    )
    phase_absent = phase_known_absent_atoms(segment)
    positive = set(normalized.get("skill_atoms", []))
    normalized["known_absent_atoms"] = sorted(
        (set(normalized.get("known_absent_atoms", [])) | phase_absent)
        - positive
    )
    normalized["unknown_atoms"] = sorted(
        set(normalized.get("unknown_atoms", []))
        - set(normalized["known_absent_atoms"])
    )
    normalized["robot_control_objectives"] = derive_control_objectives(
        normalized["skill_atoms"]
    )
    normalized["feasibility_tags"] = derive_feasibility_tags(segment)
    violations = annotation_violations(normalized)
    violations.extend(evidence_violations(normalized, segment))
    violations.extend(atom_evidence_violations(normalized, segment))
    if violations:
        raise ValueError("; ".join(violations))
    return normalized


def repair_existing_semantics(
    semantics: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Conservatively migrate an older annotation without another API call.

    Only positive atoms whose existing evidence passes the current path and
    evidence-family checks are retained. Unsupported positives and all old
    negative claims become unknown, avoiding accidental false supervision.
    """

    normalized = normalize_robot_semantics(
        semantics,
        candidate_atoms=CANONICAL_ATOMS,
    )
    phase_absent = phase_known_absent_atoms(segment)
    evidence = [
        item
        for item in normalized.get("evidence", [])
        if isinstance(item, Mapping)
    ]
    retained_atoms: list[str] = []
    retained_evidence: list[dict[str, str]] = []
    all_atoms = set(CANONICAL_ATOMS)

    for atom in normalized.get("skill_atoms", []):
        atom = canonical_atom(atom)
        atom_evidence = [
            dict(item)
            for item in evidence
            if canonical_atom(item.get("atom", "")) == atom
        ]
        candidate = {
            "skill_atoms": [atom],
            "unknown_atoms": sorted(all_atoms - {atom}),
            "known_absent_atoms": [],
            "robot_control_objectives": derive_control_objectives([atom]),
            "feasibility_tags": derive_feasibility_tags(segment),
            "evidence": atom_evidence,
        }
        violations = annotation_violations(candidate)
        violations.extend(evidence_violations(candidate, segment))
        violations.extend(atom_evidence_violations(candidate, segment))
        if not violations:
            retained_atoms.append(atom)
            retained_evidence.extend(atom_evidence)

    if normalized.get("skill_atoms") and not retained_atoms:
        return None

    repaired = {
        "skill_atoms": sorted(set(retained_atoms)),
        "unknown_atoms": sorted(
            all_atoms - set(retained_atoms) - (phase_absent - set(retained_atoms))
        ),
        "known_absent_atoms": sorted(phase_absent - set(retained_atoms)),
        "robot_control_objectives": derive_control_objectives(retained_atoms),
        "feasibility_tags": derive_feasibility_tags(segment),
    }
    if retained_evidence:
        repaired["evidence"] = retained_evidence

    if semantic_validation_violations(repaired, segment):
        return None
    return repaired


def backup_json(path: Path, backup_dir: Path) -> Path:
    """Create a timestamped backup before modifying an annotation JSON."""

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = backup_dir / f"{path.stem}.{timestamp}.json"
    shutil.copy2(path, target)
    return target


def jsonable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    for key in (
        "positive_atom_counts",
        "unknown_atom_counts",
        "known_absent_atom_counts",
        "feasibility_tag_counts",
        "positive_without_evidence",
        "positive_evidence_counts",
    ):
        result[key] = dict(sorted(stats[key].items()))
    result["partition_violation_count"] = len(stats["partition_violations"])
    result["validation_violation_count"] = len(stats["validation_violations"])
    return result


def write_report(
    report_path: Path,
    stats: dict[str, Any],
    *,
    backup_dir: Path | None = None,
) -> None:
    payload = jsonable_stats(stats)
    payload["backup_dir"] = str(backup_dir) if backup_dir else None
    write_json_atomic(report_path, payload)
