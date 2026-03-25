#!/usr/bin/env python3
"""Aggregate raw Tempo stress artifacts into a CSV and markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Any

QUERY_CLASSES = ["generic", "span_name", "svc_field", "dot_svc_field"]


def _parse_stage_run(file_name: str) -> tuple[str, int] | None:
    m = re.match(r"(?P<stage>[a-z0-9_]+)_run(?P<run>\d+)\.json$", file_name)
    if not m:
        return None
    return m.group("stage"), int(m.group("run"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _oom_or_restart(inspect_payload: list[dict[str, Any]]) -> tuple[bool, int]:
    oom = False
    restarts = 0
    for service in inspect_payload:
        state = service.get("State") or {}
        if state.get("OOMKilled"):
            oom = True
        restart_count = service.get("RestartCount")
        if isinstance(restart_count, int):
            restarts += restart_count
    return oom, restarts


def _probe_fields(probe: dict[str, Any]) -> tuple[int, float]:
    http_code = int(probe.get("http_code", 0) or 0)
    latency = float(probe.get("time_total", 0.0) or 0.0)
    return http_code, latency


def _criteria_failed(row: dict[str, Any]) -> bool:
    if float(row["failure_rate"]) > 0.05:
        return True
    if int(row["export_failures"]) > 0 or int(row["failed_spans"]) > 0:
        return True
    if float(row["p95_seconds"]) > 10.0:
        return True
    if int(row["tempo_http_code"]) >= 500 or int(row["grafana_http_code"]) >= 500:
        return True
    if float(row["tempo_query_seconds"]) > 10.0 or float(row["grafana_query_seconds"]) > 10.0:
        return True
    if int(row["tempo_search_probe_failures"]) > 0 or int(row["grafana_search_probe_failures"]) > 0:
        return True
    if float(row["tempo_search_probe_p95_seconds"]) > 10.0 or float(row["grafana_search_probe_p95_seconds"]) > 10.0:
        return True
    if int(row.get("search_probe_samples", 0) or 0) > 0:
        if not bool(row.get("tempo_fresh_within_slo", True)) or not bool(row.get("grafana_fresh_within_slo", True)):
            return True
    return False


def _failure_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if float(row["failure_rate"]) > 0.05:
        reasons.append("request_failure_rate")
    if int(row["export_failures"]) > 0 or int(row["failed_spans"]) > 0:
        reasons.append("otel_export_failure")
    if float(row["p95_seconds"]) > 10.0:
        reasons.append("app_latency_p95")
    if int(row["tempo_http_code"]) >= 500 or int(row["grafana_http_code"]) >= 500:
        reasons.append("query_http_error")
    if float(row["tempo_query_seconds"]) > 10.0 or float(row["grafana_query_seconds"]) > 10.0:
        reasons.append("query_latency")
    if bool(row["oom_killed"]) or int(row["restart_count"]) > 0:
        reasons.append("container_instability")
    if int(row.get("tempo_search_probe_failures", 0)) > 0 or int(row.get("grafana_search_probe_failures", 0)) > 0:
        reasons.append("search_probe_errors")
    if float(row.get("tempo_search_probe_p95_seconds", 0.0)) > 10.0 or float(row.get("grafana_search_probe_p95_seconds", 0.0)) > 10.0:
        reasons.append("search_probe_latency")
    if int(row.get("search_probe_samples", 0) or 0) > 0:
        if not bool(row.get("tempo_fresh_within_slo", True)) or not bool(row.get("grafana_fresh_within_slo", True)):
            reasons.append("trace_freshness")
    return reasons


def _stable_stage_order() -> list[str]:
    return ["baseline", "medium", "heavy", "extreme", "overload"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tempo stress CSV/report from raw artifacts")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    rows: list[dict[str, Any]] = []

    for metrics_file in sorted(raw_dir.glob("*_run*.json")):
        if metrics_file.name.endswith("_query_checks.json"):
            continue
        if metrics_file.name.endswith("_inspect.json"):
            continue
        if metrics_file.name.endswith("_docker_stats.json"):
            continue

        parsed = _parse_stage_run(metrics_file.name)
        if parsed is None:
            continue
        stage, run = parsed

        metrics = _load_json(metrics_file)
        query_checks_path = raw_dir / f"{stage}_run{run}_query_checks.json"
        inspect_path = raw_dir / f"{stage}_run{run}_inspect.json"

        query_checks = _load_json(query_checks_path) if query_checks_path.exists() else {}
        inspect_payload = _load_json(inspect_path) if inspect_path.exists() else []
        oom_killed, restart_count = _oom_or_restart(inspect_payload if isinstance(inspect_payload, list) else [])

        tempo_http, tempo_seconds = _probe_fields((query_checks.get("tempo_probe") or {}))
        grafana_http, grafana_seconds = _probe_fields((query_checks.get("grafana_probe") or {}))
        search_probe = query_checks.get("search_probe") or {}
        search_probe_classes = search_probe.get("classes") or {}

        attempts = int(metrics.get("attempts", 0) or 0)
        failures = int(metrics.get("failures", 0) or 0)
        failure_rate = (failures / attempts) if attempts else 0.0

        otel_stats = metrics.get("otel_export_stats") or {}
        synthetic_only = bool(
            metrics.get("synthetic_only")
            if metrics.get("synthetic_only") is not None
            else (query_checks.get("synthetic_only") if query_checks else stage != "live_sanity")
        )

        row = {
            "stage": stage,
            "run": run,
            "synthetic_only": synthetic_only,
            "trace_target_bytes": int(
                metrics.get("trace_target_bytes", 0)
                or query_checks.get("trace_target_bytes", 0)
                or 0
            ),
            "trace_span_target_bytes": int(query_checks.get("trace_span_target_bytes", 0) or 0),
            "trace_span_pause_ms": int(query_checks.get("trace_span_pause_ms", 0) or 0),
            "trace_span_pause_every": int(query_checks.get("trace_span_pause_every", 0) or 0),
            "questions": int(metrics.get("questions", 0) or 0),
            "concurrency": int(metrics.get("concurrency", 0) or 0),
            "payload_profile": str(metrics.get("payload_profile", "")),
            "attempts": attempts,
            "failures": failures,
            "failure_rate": round(failure_rate, 6),
            "p95_seconds": float((metrics.get("latency_summary") or {}).get("p95_seconds", 0.0) or 0.0),
            "tempo_http_code": tempo_http,
            "tempo_query_seconds": tempo_seconds,
            "grafana_http_code": grafana_http,
            "grafana_query_seconds": grafana_seconds,
            "search_probe_samples": int(search_probe.get("samples", 0) or 0),
            "tempo_search_probe_failures": int(search_probe.get("tempo_failures", 0) or 0),
            "grafana_search_probe_failures": int(search_probe.get("grafana_failures", 0) or 0),
            "tempo_search_probe_p95_seconds": float(search_probe.get("tempo_p95_seconds", 0.0) or 0.0),
            "grafana_search_probe_p95_seconds": float(search_probe.get("grafana_p95_seconds", 0.0) or 0.0),
            "freshness_slo_ms": int(search_probe.get("freshness_slo_ms", 0) or 0),
            "latest_emitted_at_unix_ms": int(search_probe.get("latest_emitted_at_unix_ms", 0) or 0),
            "tempo_first_hit_delay_ms": int(search_probe.get("tempo_first_hit_delay_ms", -1) or -1),
            "grafana_first_hit_delay_ms": int(search_probe.get("grafana_first_hit_delay_ms", -1) or -1),
            "tempo_fresh_within_slo": bool(search_probe.get("tempo_fresh_within_slo", False)),
            "grafana_fresh_within_slo": bool(search_probe.get("grafana_fresh_within_slo", False)),
            "export_failures": int(otel_stats.get("export_failures", 0) or 0),
            "failed_spans": int(otel_stats.get("failed_spans", 0) or 0),
            "estimated_retryable_failures": int(otel_stats.get("estimated_retryable_failures", 0) or 0),
            "oom_killed": bool(oom_killed),
            "restart_count": int(restart_count),
        }
        for query_class in QUERY_CLASSES:
            class_payload = search_probe_classes.get(query_class) or {}
            row[f"{query_class}_probe_samples"] = int(class_payload.get("samples", 0) or 0)
            row[f"tempo_{query_class}_probe_failures"] = int(class_payload.get("tempo_failures", 0) or 0)
            row[f"grafana_{query_class}_probe_failures"] = int(class_payload.get("grafana_failures", 0) or 0)
            row[f"tempo_{query_class}_probe_p95_seconds"] = float(class_payload.get("tempo_p95_seconds", 0.0) or 0.0)
            row[f"grafana_{query_class}_probe_p95_seconds"] = float(class_payload.get("grafana_p95_seconds", 0.0) or 0.0)
            row[f"tempo_{query_class}_first_hit_delay_ms"] = int(class_payload.get("tempo_first_hit_delay_ms", -1) or -1)
            row[f"grafana_{query_class}_first_hit_delay_ms"] = int(class_payload.get("grafana_first_hit_delay_ms", -1) or -1)
        row["criteria_failed"] = _criteria_failed(row)
        row["failure_reasons"] = ",".join(_failure_reasons(row))
        rows.append(row)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stage",
        "run",
        "synthetic_only",
        "trace_target_bytes",
        "trace_span_target_bytes",
        "trace_span_pause_ms",
        "trace_span_pause_every",
        "questions",
        "concurrency",
        "payload_profile",
        "attempts",
        "failures",
        "failure_rate",
        "p95_seconds",
        "tempo_http_code",
        "tempo_query_seconds",
        "grafana_http_code",
        "grafana_query_seconds",
        "search_probe_samples",
        "tempo_search_probe_failures",
        "grafana_search_probe_failures",
        "tempo_search_probe_p95_seconds",
        "grafana_search_probe_p95_seconds",
        "freshness_slo_ms",
        "latest_emitted_at_unix_ms",
        "tempo_first_hit_delay_ms",
        "grafana_first_hit_delay_ms",
        "tempo_fresh_within_slo",
        "grafana_fresh_within_slo",
        "export_failures",
        "failed_spans",
        "estimated_retryable_failures",
        "oom_killed",
        "restart_count",
        "criteria_failed",
        "failure_reasons",
    ]
    for query_class in QUERY_CLASSES:
        fieldnames.extend(
            [
                f"{query_class}_probe_samples",
                f"tempo_{query_class}_probe_failures",
                f"grafana_{query_class}_probe_failures",
                f"tempo_{query_class}_probe_p95_seconds",
                f"grafana_{query_class}_probe_p95_seconds",
                f"tempo_{query_class}_first_hit_delay_ms",
                f"grafana_{query_class}_first_hit_delay_ms",
            ]
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    stage_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage_rows.setdefault(str(row["stage"]), []).append(row)

    stage_summary_lines: list[str] = []
    breakpoint_stage = "None observed"
    breakpoint_detail = "No unstable stage in synthetic ramp"
    dominant_failure_modes = "None"

    for stage in _stable_stage_order():
        items = stage_rows.get(stage, [])
        if not items:
            continue
        med_p95 = median([float(item["p95_seconds"]) for item in items])
        worst_p95 = max(float(item["p95_seconds"]) for item in items)
        fail_count = sum(1 for item in items if bool(item["criteria_failed"]))
        stage_summary_lines.append(
            f"| {stage} | {len(items)} | {med_p95:.3f} | {worst_p95:.3f} | {fail_count} |"
        )
        if breakpoint_stage == "None observed" and len(items) >= 3 and fail_count >= 2:
            breakpoint_stage = stage

    if breakpoint_stage != "None observed":
        breakpoint_detail = "First synthetic stage with >=2 failed runs (out of 3+ runs)"
        mode_counts: dict[str, int] = {}
        for item in stage_rows.get(breakpoint_stage, []):
            if not bool(item["criteria_failed"]):
                continue
            for reason in str(item.get("failure_reasons", "")).split(","):
                reason = reason.strip()
                if not reason:
                    continue
                mode_counts[reason] = mode_counts.get(reason, 0) + 1
        if mode_counts:
            ranked = sorted(mode_counts.items(), key=lambda pair: pair[1], reverse=True)
            dominant_failure_modes = ", ".join(f"{name} ({count})" for name, count in ranked[:3])

    synthetic_rows = [row for row in rows if bool(row.get("synthetic_only"))]
    live_rows = [row for row in rows if not bool(row.get("synthetic_only"))]

    def _slice_summary(items: list[dict[str, Any]]) -> str:
        if not items:
            return "n/a"
        fail_count = sum(1 for item in items if bool(item["criteria_failed"]))
        fail_rate = fail_count / len(items)
        med_p95 = median([float(item["p95_seconds"]) for item in items])
        max_trace_target = max(int(item.get("trace_target_bytes", 0) or 0) for item in items)
        max_search_p95 = max(
            max(
                float(item.get("tempo_search_probe_p95_seconds", 0.0) or 0.0),
                float(item.get("grafana_search_probe_p95_seconds", 0.0) or 0.0),
            )
            for item in items
        )
        return (
            f"runs={len(items)}, failed={fail_count}, fail_rate={fail_rate:.2%}, "
            f"median_p95={med_p95:.3f}s, max_trace_target_bytes={max_trace_target}, "
            f"max_search_probe_p95_seconds={max_search_p95:.3f}"
        )

    synthetic_summary = _slice_summary(synthetic_rows)
    live_summary = _slice_summary(live_rows)

    class_summary_lines: list[str] = []
    for query_class in QUERY_CLASSES:
        class_tempo_p95_vals = [
            float(row.get(f"tempo_{query_class}_probe_p95_seconds", 0.0) or 0.0)
            for row in rows
            if int(row.get(f"{query_class}_probe_samples", 0) or 0) > 0
        ]
        class_grafana_p95_vals = [
            float(row.get(f"grafana_{query_class}_probe_p95_seconds", 0.0) or 0.0)
            for row in rows
            if int(row.get(f"{query_class}_probe_samples", 0) or 0) > 0
        ]
        total_samples = sum(int(row.get(f"{query_class}_probe_samples", 0) or 0) for row in rows)
        total_failures = sum(
            int(row.get(f"tempo_{query_class}_probe_failures", 0) or 0)
            + int(row.get(f"grafana_{query_class}_probe_failures", 0) or 0)
            for row in rows
        )
        tempo_delay_vals = [
            int(row.get(f"tempo_{query_class}_first_hit_delay_ms", -1) or -1)
            for row in rows
            if int(row.get(f"tempo_{query_class}_first_hit_delay_ms", -1) or -1) >= 0
        ]
        grafana_delay_vals = [
            int(row.get(f"grafana_{query_class}_first_hit_delay_ms", -1) or -1)
            for row in rows
            if int(row.get(f"grafana_{query_class}_first_hit_delay_ms", -1) or -1) >= 0
        ]
        if total_samples == 0:
            class_summary_lines.append(
                f"| {query_class} | 0 | 0 | 0.000 | 0.000 | n/a | n/a |"
            )
            continue
        class_summary_lines.append(
            # Delay is based on first successful non-empty search response after latest trace emit.
            # Use n/a if no hit was observed.
            f"| {query_class} | {total_samples} | {total_failures} | "
            f"{(max(class_tempo_p95_vals) if class_tempo_p95_vals else 0.0):.3f} | "
            f"{(max(class_grafana_p95_vals) if class_grafana_p95_vals else 0.0):.3f} | "
            f"{(str(min(tempo_delay_vals)) if tempo_delay_vals else 'n/a')} | "
            f"{(str(min(grafana_delay_vals)) if grafana_delay_vals else 'n/a')} |"
        )

    env_header = {}
    env_path = raw_dir / "environment.json"
    if env_path.exists():
        env_header = _load_json(env_path)

    timestamp_utc = str(env_header.get("timestamp_utc", "n/a"))
    git_sha = str(env_header.get("git_sha", "n/a"))
    cpu = str(env_header.get("cpu", "n/a"))
    memory_bytes = int(env_header.get("memory_bytes", 0) or 0)
    memory_gib = (memory_bytes / (1024**3)) if memory_bytes > 0 else 0.0
    trace_backend = str(env_header.get("trace_backend", "n/a"))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = "\n".join(
        [
            "# Tempo Stress Report",
            "",
            "## Summary",
            "",
            f"- Breakpoint stage: **{breakpoint_stage}**",
            f"- Breakpoint rule: {breakpoint_detail}",
            f"- Dominant failure mode(s) at breakpoint: {dominant_failure_modes}",
            f"- Raw artifact directory: `{raw_dir}`",
            f"- Aggregated CSV: `{csv_path}`",
            "",
            "## Environment Header",
            "",
            f"- Timestamp (UTC): `{timestamp_utc}`",
            f"- Git SHA: `{git_sha}`",
            f"- Trace backend: `{trace_backend}`",
            f"- CPU: `{cpu}`",
            f"- Memory: `{memory_gib:.2f} GiB`",
            "",
            "## Stage Medians/Worst",
            "",
            "| Stage | Runs | Median p95 (s) | Worst p95 (s) | Runs Failing Criteria |",
            "|---|---:|---:|---:|---:|",
            *stage_summary_lines,
            "",
            "## Breakpoint Analysis",
            "",
            f"- Synthetic ramp summary: {synthetic_summary}",
            f"- Live sanity summary: {live_summary}",
            "",
            "## Query Class Latency",
            "",
            "| Query Class | Samples | Total Probe Failures | Tempo p95 Max (s) | Grafana p95 Max (s) | Tempo first-hit delay min (ms) | Grafana first-hit delay min (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *class_summary_lines,
            "",
            "## Failure Criteria",
            "",
            "A run is marked failed when any of the following is true:",
            "- request failure rate > 5%",
            "- OTEL export failures or failed spans detected",
            "- p95 question latency > 10s",
            "- Tempo or Grafana query endpoint returned >= 500",
            "- Tempo or Grafana query probe took > 10s",
            "- Latest emitted trace was not queryable within freshness SLO",
            "",
            "## Notes",
            "",
            "- `scripts/run_tempo_stress.sh` updates raw artifacts and regenerates this report.",
            "- Use `TRACE_RUN_TAG` in raw JSON payloads to correlate traces in Tempo UI/API.",
        ]
    )

    report_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
