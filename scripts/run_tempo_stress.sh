#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="infra/tempo/docker-compose.yml"
RAW_DIR="artifacts/raw"
REPORT_PATH="docs/tempo_stress_report.md"
CSV_PATH="artifacts/tempo_stress_results.csv"
ENV_PATH="$RAW_DIR/environment.json"

mkdir -p "$RAW_DIR"

OTEL_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}"
OTEL_PROTOCOL="${OTEL_EXPORTER_OTLP_PROTOCOL:-http/protobuf}"
REPEATS="${TEMPO_STAGE_REPEATS:-3}"
SYNTHETIC_ONLY="${TEMPO_SYNTHETIC_ONLY:-1}"
INCLUDE_OVERLOAD="${TEMPO_INCLUDE_OVERLOAD:-1}"
RUN_LIVE_SANITY="${TEMPO_RUN_LIVE_SANITY:-1}"
TRACE_BACKEND_VALUE="${TRACE_BACKEND:-otlp}"
STAGE_FILTER="${TEMPO_STAGE_FILTER:-}"
TRACE_TARGET_BYTES="${TEMPO_TRACE_TARGET_BYTES:-0}"
TRACE_SIZE_RAMP_BYTES="${TEMPO_TRACE_SIZE_RAMP_BYTES:-}"
TRACE_SIZE_RAMP_INDEX=0
NEXT_TRACE_TARGET_BYTES=""
TRACE_SPAN_TARGET_BYTES="${TEMPO_TRACE_SPAN_TARGET_BYTES:-6291456}"
TRACE_SPAN_PAUSE_MS="${TEMPO_TRACE_SPAN_PAUSE_MS:-0}"
TRACE_SPAN_PAUSE_EVERY="${TEMPO_TRACE_SPAN_PAUSE_EVERY:-0}"
SEARCH_PROBE_REQUESTS="${TEMPO_SEARCH_PROBE_REQUESTS:-20}"
SEARCH_PROBE_DELAY_MS="${TEMPO_SEARCH_PROBE_DELAY_MS:-100}"
QUERY_CLASSES="${TEMPO_QUERY_CLASSES:-generic,span_name,svc_field,dot_svc_field}"
FRESHNESS_SLO_MS="${TEMPO_QUERY_FRESHNESS_SLO_MS:-15000}"
OTEL_BSP_MAX_QUEUE_SIZE="${TEMPO_OTEL_BSP_MAX_QUEUE_SIZE:-8192}"
OTEL_BSP_MAX_EXPORT_BATCH_SIZE="${TEMPO_OTEL_BSP_MAX_EXPORT_BATCH_SIZE:-128}"
OTEL_BSP_SCHEDULE_DELAY="${TEMPO_OTEL_BSP_SCHEDULE_DELAY:-500}"
OTEL_BSP_EXPORT_TIMEOUT="${TEMPO_OTEL_BSP_EXPORT_TIMEOUT:-30000}"
TEMPO_INGEST_RATE_LIMIT_BYTES="${TEMPO_INGEST_RATE_LIMIT_BYTES:-5000000}"
TEMPO_INGEST_BURST_SIZE_BYTES="${TEMPO_INGEST_BURST_SIZE_BYTES:-20000000}"
TEMPO_MAX_BYTES_PER_TRACE="${TEMPO_MAX_BYTES_PER_TRACE:-10737418240}"
TEMPO_SEARCH_DURATION_SLO_SECONDS="${TEMPO_SEARCH_DURATION_SLO_SECONDS:-30}"
TEMPO_SEARCH_THROUGHPUT_BYTES_SLO="${TEMPO_SEARCH_THROUGHPUT_BYTES_SLO:-1073741824}"
TEMPO_CONFIG_TEMPLATE="$ROOT_DIR/infra/tempo/tempo.dynamic.yaml.tpl"
TEMPO_CONFIG_RENDERED="$ROOT_DIR/infra/tempo/tempo.generated.yaml"

function json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}

function compose_up_wait() {
  export TEMPO_CONFIG_PATH="$TEMPO_CONFIG_RENDERED"
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" up -d --force-recreate
    return 0
  fi
  docker compose -f "$COMPOSE_FILE" up -d --wait --force-recreate
}

function render_tempo_config() {
  if [[ ! -f "$TEMPO_CONFIG_TEMPLATE" ]]; then
    cp "$ROOT_DIR/infra/tempo/tempo.yaml" "$TEMPO_CONFIG_RENDERED"
    return 0
  fi

python3 - "$TEMPO_CONFIG_TEMPLATE" "$TEMPO_CONFIG_RENDERED" <<'PY'
import os
import pathlib
import sys

template = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
rendered = template
for key in (
    "TEMPO_INGEST_RATE_LIMIT_BYTES",
    "TEMPO_INGEST_BURST_SIZE_BYTES",
    "TEMPO_MAX_BYTES_PER_TRACE",
    "TEMPO_SEARCH_DURATION_SLO_SECONDS",
    "TEMPO_SEARCH_THROUGHPUT_BYTES_SLO",
):
    rendered = rendered.replace(f"__{key}__", str(os.environ[key]))

pathlib.Path(sys.argv[2]).write_text(rendered, encoding="utf-8")
PY
}

function compose_ps_q() {
  local service="$1"
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" ps -q "$service"
    return 0
  fi
  docker compose -f "$COMPOSE_FILE" ps -q "$service"
}

function json_http_probe() {
  local url="$1"
  local auth="${2:-}"
  if [[ -n "$auth" ]]; then
    curl -sS -u "$auth" -o /tmp/tempo_probe_body.txt -w '{"http_code":%{http_code},"time_total":%{time_total}}' "$url" || echo '{"http_code":0,"time_total":0.0}'
  else
    curl -sS -o /tmp/tempo_probe_body.txt -w '{"http_code":%{http_code},"time_total":%{time_total}}' "$url" || echo '{"http_code":0,"time_total":0.0}'
  fi
}

function json_http_probe_traceql() {
  local url="$1"
  local auth="$2"
  local traceql="$3"
  local body_file
  body_file="$(mktemp)"
  local probe_json='{"http_code":0,"time_total":0.0}'

  if [[ -z "$traceql" ]]; then
    if [[ -n "$auth" ]]; then
      probe_json="$(curl -sS -u "$auth" -o "$body_file" -w '{"http_code":%{http_code},"time_total":%{time_total}}' "$url" || echo '{"http_code":0,"time_total":0.0}')"
    else
      probe_json="$(curl -sS -o "$body_file" -w '{"http_code":%{http_code},"time_total":%{time_total}}' "$url" || echo '{"http_code":0,"time_total":0.0}')"
    fi
    python3 - "$probe_json" "$body_file" <<'PY'
import json
import pathlib
import sys
import time

probe_raw = sys.argv[1]
body_path = pathlib.Path(sys.argv[2])
out = {"http_code": 0, "time_total": 0.0, "trace_count": 0, "latest_start_unix_ns": 0, "sampled_at_unix_ms": int(time.time() * 1000)}
try:
    parsed = json.loads(probe_raw)
    out["http_code"] = int(parsed.get("http_code", 0) or 0)
    out["time_total"] = float(parsed.get("time_total", 0.0) or 0.0)
except Exception:
    pass
try:
    body = json.loads(body_path.read_text(encoding="utf-8"))
except Exception:
    body = None

trace_count = 0
latest_start_ns = 0
def walk(value):
    global trace_count, latest_start_ns
    if isinstance(value, dict):
        trace_id = value.get("traceID") or value.get("traceId")
        if trace_id:
            trace_count += 1
            raw_start = value.get("startTimeUnixNano") or value.get("start_time_unix_nano") or value.get("start")
            try:
                candidate = int(raw_start)
            except Exception:
                candidate = 0
            if candidate > latest_start_ns:
                latest_start_ns = candidate
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

walk(body)
out["trace_count"] = trace_count
out["latest_start_unix_ns"] = latest_start_ns
print(json.dumps(out))
PY
    rm -f "$body_file"
    return 0
  fi

  if [[ -n "$auth" ]]; then
    probe_json="$(curl -sS -u "$auth" --get \
      --data-urlencode "q=$traceql" \
      --data "limit=20" \
      -o "$body_file" \
      -w '{"http_code":%{http_code},"time_total":%{time_total}}' \
      "$url" || echo '{"http_code":0,"time_total":0.0}')"
  else
    probe_json="$(curl -sS --get \
      --data-urlencode "q=$traceql" \
      --data "limit=20" \
      -o "$body_file" \
      -w '{"http_code":%{http_code},"time_total":%{time_total}}' \
      "$url" || echo '{"http_code":0,"time_total":0.0}')"
  fi
  python3 - "$probe_json" "$body_file" <<'PY'
import json
import pathlib
import sys
import time

probe_raw = sys.argv[1]
body_path = pathlib.Path(sys.argv[2])
out = {"http_code": 0, "time_total": 0.0, "trace_count": 0, "latest_start_unix_ns": 0, "sampled_at_unix_ms": int(time.time() * 1000)}
try:
    parsed = json.loads(probe_raw)
    out["http_code"] = int(parsed.get("http_code", 0) or 0)
    out["time_total"] = float(parsed.get("time_total", 0.0) or 0.0)
except Exception:
    pass
try:
    body = json.loads(body_path.read_text(encoding="utf-8"))
except Exception:
    body = None

trace_count = 0
latest_start_ns = 0
def walk(value):
    global trace_count, latest_start_ns
    if isinstance(value, dict):
        trace_id = value.get("traceID") or value.get("traceId")
        if trace_id:
            trace_count += 1
            raw_start = value.get("startTimeUnixNano") or value.get("start_time_unix_nano") or value.get("start")
            try:
                candidate = int(raw_start)
            except Exception:
                candidate = 0
            if candidate > latest_start_ns:
                latest_start_ns = candidate
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

walk(body)
out["trace_count"] = trace_count
out["latest_start_unix_ns"] = latest_start_ns
print(json.dumps(out))
PY
  rm -f "$body_file"
}

function traceql_for_class() {
  local class_name="$1"
  local run_tag="$2"
  local run_filter=".stress_run_tag=\"$run_tag\""
  case "$class_name" in
    generic)
      echo "{$run_filter}"
      ;;
    span_name)
      echo "{name=\"synthetic_step_000\" && $run_filter}"
      ;;
    svc_field)
      echo "{resource.service.name=\"pydantic-supervisor\" && $run_filter}"
      ;;
    dot_svc_field)
      echo "{.service.name=\"pydantic-supervisor\" && $run_filter}"
      ;;
    *)
      echo "{$run_filter}"
      ;;
  esac
}

function wait_for_http_200() {
  local name="$1"
  local url="$2"
  local auth="${3:-}"
  local max_attempts="${4:-30}"
  local sleep_seconds="${5:-2}"

  for attempt in $(seq 1 "$max_attempts"); do
    local probe
    probe="$(json_http_probe "$url" "$auth")"
    local code
    code="$(echo "$probe" | python3 -c 'import json,sys; print(int((json.load(sys.stdin).get("http_code") or 0)))' 2>/dev/null || echo 0)"
    if [[ "$code" == "200" ]]; then
      echo "   ✓ $name healthy (attempt $attempt)"
      return 0
    fi
    echo "   ... waiting for $name (attempt $attempt/$max_attempts, http=$code)"
    sleep "$sleep_seconds"
  done

  echo "ERROR: $name did not become healthy in time"
  return 1
}

function capture_environment_header() {
  local now_utc
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  local git_sha
  git_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  local host_uname
  host_uname="$(uname -a | json_escape)"
  local docker_version
  docker_version="$(docker version --format '{{json .}}' 2>/dev/null || echo '{}')"
  local cpu_info
  cpu_info="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null || echo unknown)"
  local mem_bytes
  mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || free -b 2>/dev/null | awk '/Mem:/ {print $2}' || echo 0)"

  cat > "$ENV_PATH" <<EOF
{
  "timestamp_utc": "$now_utc",
  "git_sha": "$git_sha",
  "host_uname": $host_uname,
  "cpu": $(echo "$cpu_info" | json_escape),
  "memory_bytes": $mem_bytes,
  "docker_version": $docker_version,
  "trace_backend": "$TRACE_BACKEND_VALUE",
  "otlp_endpoint": "$OTEL_ENDPOINT",
  "otlp_protocol": "$OTEL_PROTOCOL",
  "stage_repeats": $REPEATS,
  "synthetic_only_core_ramp": $SYNTHETIC_ONLY,
  "include_overload_stage": $INCLUDE_OVERLOAD,
  "run_live_sanity_stage": $RUN_LIVE_SANITY
  ,"stage_filter": $(echo "$STAGE_FILTER" | json_escape)
  ,"trace_target_bytes": $TRACE_TARGET_BYTES
  ,"trace_size_ramp_bytes": $(echo "$TRACE_SIZE_RAMP_BYTES" | json_escape)
  ,"trace_span_target_bytes": $TRACE_SPAN_TARGET_BYTES
  ,"trace_span_pause_ms": $TRACE_SPAN_PAUSE_MS
  ,"trace_span_pause_every": $TRACE_SPAN_PAUSE_EVERY
  ,"search_probe_requests": $SEARCH_PROBE_REQUESTS
  ,"search_probe_delay_ms": $SEARCH_PROBE_DELAY_MS
  ,"query_freshness_slo_ms": $FRESHNESS_SLO_MS
  ,"query_classes": $(echo "$QUERY_CLASSES" | json_escape)
  ,"otel_bsp_max_queue_size": $OTEL_BSP_MAX_QUEUE_SIZE
  ,"otel_bsp_max_export_batch_size": $OTEL_BSP_MAX_EXPORT_BATCH_SIZE
  ,"otel_bsp_schedule_delay": $OTEL_BSP_SCHEDULE_DELAY
  ,"otel_bsp_export_timeout": $OTEL_BSP_EXPORT_TIMEOUT
  ,"tempo_ingest_rate_limit_bytes": $TEMPO_INGEST_RATE_LIMIT_BYTES
  ,"tempo_ingest_burst_size_bytes": $TEMPO_INGEST_BURST_SIZE_BYTES
  ,"tempo_max_bytes_per_trace": $TEMPO_MAX_BYTES_PER_TRACE
  ,"tempo_search_duration_slo_seconds": $TEMPO_SEARCH_DURATION_SLO_SECONDS
  ,"tempo_search_throughput_bytes_slo": $TEMPO_SEARCH_THROUGHPUT_BYTES_SLO
  ,"tempo_config_path": $(echo "$TEMPO_CONFIG_RENDERED" | json_escape)
}
EOF
}

function next_trace_target_bytes() {
  local fallback="$1"
  NEXT_TRACE_TARGET_BYTES="$fallback"
  if [[ -z "$TRACE_SIZE_RAMP_BYTES" ]]; then
    NEXT_TRACE_TARGET_BYTES="$fallback"
    return 0
  fi

  local -a ramp_values=()
  IFS=',' read -r -a ramp_values <<< "$TRACE_SIZE_RAMP_BYTES"
  if [[ "${#ramp_values[@]}" -eq 0 ]]; then
    NEXT_TRACE_TARGET_BYTES="$fallback"
    return 0
  fi

  local idx=$((TRACE_SIZE_RAMP_INDEX % ${#ramp_values[@]}))
  TRACE_SIZE_RAMP_INDEX=$((TRACE_SIZE_RAMP_INDEX + 1))
  local candidate="${ramp_values[$idx]}"
  candidate="${candidate//[[:space:]]/}"
  if [[ "$candidate" =~ ^[0-9]+$ ]]; then
    NEXT_TRACE_TARGET_BYTES="$candidate"
  else
    NEXT_TRACE_TARGET_BYTES="$fallback"
  fi
}

function run_stage() {
  local stage_name="$1"
  local questions="$2"
  local concurrency="$3"
  local payload_profile="$4"
  local inject_flag="$5"
  local synthetic_mode="$6"

  local inject_arg="--no-inject-large-attributes"
  if [[ "$inject_flag" == "true" ]]; then
    inject_arg="--inject-large-attributes"
  fi

  local synthetic_arg="--no-synthetic-only"
  if [[ "$synthetic_mode" == "1" ]]; then
    synthetic_arg="--synthetic-only"
  fi

  for run_index in $(seq 1 "$REPEATS"); do
    local effective_trace_target_bytes
    next_trace_target_bytes "$TRACE_TARGET_BYTES"
    effective_trace_target_bytes="$NEXT_TRACE_TARGET_BYTES"
    run_tag="${stage_name}-run${run_index}-$(date +%s)"
    metrics_file="$RAW_DIR/${stage_name}_run${run_index}.json"
    docker_stats_file="$RAW_DIR/${stage_name}_run${run_index}_docker_stats.json"
    inspect_file="$RAW_DIR/${stage_name}_run${run_index}_inspect.json"
    query_file="$RAW_DIR/${stage_name}_run${run_index}_query_checks.json"
    query_probe_file="$RAW_DIR/${stage_name}_run${run_index}_query_probe_samples.ndjson"
    trace_manifest_file="$RAW_DIR/${stage_name}_run${run_index}_trace_manifest.json"

    echo "==> Stage=$stage_name run=$run_index questions=$questions concurrency=$concurrency payload=$payload_profile inject=$inject_flag synthetic=$synthetic_mode trace_target_bytes=$effective_trace_target_bytes span_target_bytes=$TRACE_SPAN_TARGET_BYTES pause_ms=$TRACE_SPAN_PAUSE_MS pause_every=$TRACE_SPAN_PAUSE_EVERY"

    TRACE_BACKEND="$TRACE_BACKEND_VALUE" \
    TRACE_RUN_TAG="$run_tag" \
    TRACE_PAYLOAD_PROFILE="$payload_profile" \
    TRACE_SPAN_TARGET_BYTES="$TRACE_SPAN_TARGET_BYTES" \
    TRACE_SPAN_PAUSE_MS="$TRACE_SPAN_PAUSE_MS" \
    TRACE_SPAN_PAUSE_EVERY="$TRACE_SPAN_PAUSE_EVERY" \
    OTEL_BSP_MAX_QUEUE_SIZE="$OTEL_BSP_MAX_QUEUE_SIZE" \
    OTEL_BSP_MAX_EXPORT_BATCH_SIZE="$OTEL_BSP_MAX_EXPORT_BATCH_SIZE" \
    OTEL_BSP_SCHEDULE_DELAY="$OTEL_BSP_SCHEDULE_DELAY" \
    OTEL_BSP_EXPORT_TIMEOUT="$OTEL_BSP_EXPORT_TIMEOUT" \
    OTEL_EXPORTER_OTLP_ENDPOINT="$OTEL_ENDPOINT" \
    OTEL_EXPORTER_OTLP_PROTOCOL="$OTEL_PROTOCOL" \
    python3 scripts/run_queries.py \
      --question-source bank \
      --questions "$questions" \
      --rounds 1 \
      --concurrency "$concurrency" \
      --payload-profile "$payload_profile" \
      "$inject_arg" \
      "$synthetic_arg" \
      --trace-target-bytes "$effective_trace_target_bytes" \
      --trace-run-tag "$run_tag" \
      --metrics-output "$metrics_file" \
      --trace-manifest-output "$trace_manifest_file" \
      --no-quota-preflight

    local tempo_cid
    local grafana_cid
    local collector_cid
    tempo_cid="$(compose_ps_q tempo)"
    grafana_cid="$(compose_ps_q grafana)"
    collector_cid="$(compose_ps_q otel-collector)"
    docker stats --no-stream --format '{{json .}}' "$tempo_cid" "$grafana_cid" "$collector_cid" > "$docker_stats_file" || true
    docker inspect "$tempo_cid" "$grafana_cid" "$collector_cid" > "$inspect_file" || true

    tempo_probe="$(json_http_probe 'http://localhost:3200/api/search?limit=20')"
    grafana_probe="$(json_http_probe 'http://localhost:3000/api/datasources/proxy/uid/tempo/api/search?limit=20' 'admin:admin')"
    collector_probe="$(json_http_probe 'http://localhost:13133/')"

    : > "$query_probe_file"
    local -a probe_classes=()
    IFS=',' read -r -a probe_classes <<< "$QUERY_CLASSES"
    for class_name in "${probe_classes[@]}"; do
      class_name="${class_name//[[:space:]]/}"
      if [[ -z "$class_name" ]]; then
        continue
      fi
      traceql="$(traceql_for_class "$class_name" "$run_tag")"
      for _ in $(seq 1 "$SEARCH_PROBE_REQUESTS"); do
        tempo_search_probe="$(json_http_probe_traceql 'http://localhost:3200/api/search' '' "$traceql")"
        grafana_search_probe="$(json_http_probe_traceql 'http://localhost:3000/api/datasources/proxy/uid/tempo/api/search' 'admin:admin' "$traceql")"
        cat >> "$query_probe_file" <<EOF
{"class": "$class_name", "tempo": $tempo_search_probe, "grafana": $grafana_search_probe}
EOF
        if [[ "$SEARCH_PROBE_DELAY_MS" -gt 0 ]]; then
          sleep "$(python3 -c "print($SEARCH_PROBE_DELAY_MS/1000)")"
        fi
      done
    done

    query_probe_summary="$(python3 - "$query_probe_file" "$trace_manifest_file" "$FRESHNESS_SLO_MS" <<'PY'
import json
import sys
from statistics import median
from collections import defaultdict

path = sys.argv[1]
manifest_path = sys.argv[2]
freshness_slo_ms = int(sys.argv[3])
tempo_latencies = []
grafana_latencies = []
tempo_failures = 0
grafana_failures = 0
samples = 0
class_data: dict[str, dict[str, list[float] | int]] = defaultdict(
    lambda: {
        "tempo_latencies": [],
        "grafana_latencies": [],
        "tempo_failures": 0,
        "grafana_failures": 0,
        "samples": 0,
        "tempo_first_hit_delay_ms": None,
        "grafana_first_hit_delay_ms": None,
    }
)

latest_emitted_ms = 0
try:
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    traces = manifest.get("traces") or []
    latest_emitted_ms = max((int((item or {}).get("emitted_at_unix_ms", 0) or 0) for item in traces), default=0)
except Exception:
    latest_emitted_ms = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        samples += 1
        row = json.loads(line)
        class_name = str(row.get("class") or "generic")
        tempo = row.get("tempo", {})
        grafana = row.get("grafana", {})
        tempo_code = int(tempo.get("http_code", 0) or 0)
        grafana_code = int(grafana.get("http_code", 0) or 0)
        tempo_latency = float(tempo.get("time_total", 0.0) or 0.0)
        grafana_latency = float(grafana.get("time_total", 0.0) or 0.0)
        tempo_trace_count = int(tempo.get("trace_count", 0) or 0)
        grafana_trace_count = int(grafana.get("trace_count", 0) or 0)
        tempo_sampled_at_ms = int(tempo.get("sampled_at_unix_ms", 0) or 0)
        grafana_sampled_at_ms = int(grafana.get("sampled_at_unix_ms", 0) or 0)
        tempo_latencies.append(tempo_latency)
        grafana_latencies.append(grafana_latency)
        class_entry = class_data[class_name]
        class_entry["samples"] = int(class_entry["samples"]) + 1
        class_entry["tempo_latencies"].append(tempo_latency)
        class_entry["grafana_latencies"].append(grafana_latency)
        if latest_emitted_ms > 0 and tempo_trace_count > 0 and tempo_sampled_at_ms >= latest_emitted_ms:
            delay = tempo_sampled_at_ms - latest_emitted_ms
            current = class_entry.get("tempo_first_hit_delay_ms")
            class_entry["tempo_first_hit_delay_ms"] = delay if current is None else min(int(current), delay)
        if latest_emitted_ms > 0 and grafana_trace_count > 0 and grafana_sampled_at_ms >= latest_emitted_ms:
            delay = grafana_sampled_at_ms - latest_emitted_ms
            current = class_entry.get("grafana_first_hit_delay_ms")
            class_entry["grafana_first_hit_delay_ms"] = delay if current is None else min(int(current), delay)
        if tempo_code < 200 or tempo_code >= 300:
            tempo_failures += 1
            class_entry["tempo_failures"] = int(class_entry["tempo_failures"]) + 1
        if grafana_code < 200 or grafana_code >= 300:
            grafana_failures += 1
            class_entry["grafana_failures"] = int(class_entry["grafana_failures"]) + 1

def p95(values):
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, int((len(vals) - 1) * 0.95))
    return vals[idx]

class_summary: dict[str, dict[str, float | int]] = {}
for class_name, payload in class_data.items():
    t_lat = payload["tempo_latencies"]
    g_lat = payload["grafana_latencies"]
    class_summary[class_name] = {
        "samples": int(payload["samples"]),
        "tempo_failures": int(payload["tempo_failures"]),
        "grafana_failures": int(payload["grafana_failures"]),
        "tempo_p95_seconds": p95(t_lat),
        "grafana_p95_seconds": p95(g_lat),
        "tempo_median_seconds": median(t_lat) if t_lat else 0.0,
        "grafana_median_seconds": median(g_lat) if g_lat else 0.0,
        "tempo_first_hit_delay_ms": payload.get("tempo_first_hit_delay_ms"),
        "grafana_first_hit_delay_ms": payload.get("grafana_first_hit_delay_ms"),
        "tempo_fresh_within_slo": bool(
            payload.get("tempo_first_hit_delay_ms") is not None
            and int(payload.get("tempo_first_hit_delay_ms")) <= freshness_slo_ms
        ),
        "grafana_fresh_within_slo": bool(
            payload.get("grafana_first_hit_delay_ms") is not None
            and int(payload.get("grafana_first_hit_delay_ms")) <= freshness_slo_ms
        ),
    }

tempo_first_hit_delays = [
    int((class_summary.get(name) or {}).get("tempo_first_hit_delay_ms"))
    for name in class_summary
    if (class_summary.get(name) or {}).get("tempo_first_hit_delay_ms") is not None
]
grafana_first_hit_delays = [
    int((class_summary.get(name) or {}).get("grafana_first_hit_delay_ms"))
    for name in class_summary
    if (class_summary.get(name) or {}).get("grafana_first_hit_delay_ms") is not None
]

summary = {
    "samples": samples,
    "tempo_failures": tempo_failures,
    "grafana_failures": grafana_failures,
    "tempo_p95_seconds": p95(tempo_latencies),
    "grafana_p95_seconds": p95(grafana_latencies),
    "tempo_median_seconds": median(tempo_latencies) if tempo_latencies else 0.0,
    "grafana_median_seconds": median(grafana_latencies) if grafana_latencies else 0.0,
    "freshness_slo_ms": freshness_slo_ms,
    "latest_emitted_at_unix_ms": latest_emitted_ms,
    "tempo_first_hit_delay_ms": min(tempo_first_hit_delays) if tempo_first_hit_delays else None,
    "grafana_first_hit_delay_ms": min(grafana_first_hit_delays) if grafana_first_hit_delays else None,
    "tempo_fresh_within_slo": bool(tempo_first_hit_delays) and min(tempo_first_hit_delays) <= freshness_slo_ms,
    "grafana_fresh_within_slo": bool(grafana_first_hit_delays) and min(grafana_first_hit_delays) <= freshness_slo_ms,
    "classes": class_summary,
}
print(json.dumps(summary))
PY
)"

    cat > "$query_file" <<EOF
{
  "stage": "$stage_name",
  "run": $run_index,
  "synthetic_only": $synthetic_mode,
  "trace_target_bytes": $effective_trace_target_bytes,
  "trace_span_target_bytes": $TRACE_SPAN_TARGET_BYTES,
  "trace_span_pause_ms": $TRACE_SPAN_PAUSE_MS,
  "trace_span_pause_every": $TRACE_SPAN_PAUSE_EVERY,
  "trace_run_tag": "$run_tag",
  "trace_manifest_file": "$trace_manifest_file",
  "tempo_probe": $tempo_probe,
  "grafana_probe": $grafana_probe,
  "collector_probe": $collector_probe,
  "search_probe": $query_probe_summary
}
EOF
  done
}

echo "==> Starting local Tempo + Grafana + OTel Collector stack"
echo "==> Tempo limits: ingest_rate=${TEMPO_INGEST_RATE_LIMIT_BYTES}B/s burst=${TEMPO_INGEST_BURST_SIZE_BYTES}B max_trace_bytes=${TEMPO_MAX_BYTES_PER_TRACE} search_slo=${TEMPO_SEARCH_DURATION_SLO_SECONDS}s throughput_slo=${TEMPO_SEARCH_THROUGHPUT_BYTES_SLO}B"
render_tempo_config
compose_up_wait
capture_environment_header

echo "==> Preflight: verifying stack health endpoints"
wait_for_http_200 "Tempo search endpoint" "http://localhost:3200/api/search?limit=20"
wait_for_http_200 "Grafana Tempo datasource proxy" "http://localhost:3000/api/datasources/proxy/uid/tempo/api/search?limit=20" "admin:admin"
wait_for_http_200 "OTel collector health endpoint" "http://localhost:13133/"

# stage_name questions concurrency payload_profile inject_large_attributes
STAGES=(
  "baseline 10 1 baseline false"
  "medium 40 4 large true"
  "heavy 80 8 large true"
  "extreme 120 12 xlarge true"
)

if [[ "$INCLUDE_OVERLOAD" == "1" ]]; then
  STAGES+=("overload 180 16 xlarge true")
fi

for stage_def in "${STAGES[@]}"; do
  read -r stage_name questions concurrency payload_profile inject_flag <<< "$stage_def"
  if [[ -n "$STAGE_FILTER" ]] && [[ ",$STAGE_FILTER," != *",$stage_name,"* ]]; then
    echo "==> Skipping stage=$stage_name due to TEMPO_STAGE_FILTER=$STAGE_FILTER"
    continue
  fi
  run_stage "$stage_name" "$questions" "$concurrency" "$payload_profile" "$inject_flag" "$SYNTHETIC_ONLY"
done

if [[ "$RUN_LIVE_SANITY" == "1" ]]; then
  if [[ -n "$STAGE_FILTER" ]] && [[ ",$STAGE_FILTER," != *",live_sanity,"* ]]; then
    echo "==> Skipping stage=live_sanity due to TEMPO_STAGE_FILTER=$STAGE_FILTER"
  else
  echo "==> Running live sanity slice after synthetic ramp"
  run_stage "live_sanity" "20" "2" "large" "true" "0"
  fi
fi

python3 scripts/build_tempo_stress_report.py \
  --raw-dir "$RAW_DIR" \
  --csv "$CSV_PATH" \
  --report "$REPORT_PATH"

echo "==> Completed Tempo stress run"
echo "CSV: $CSV_PATH"
echo "Report: $REPORT_PATH"
echo "Environment header: $ENV_PATH"
