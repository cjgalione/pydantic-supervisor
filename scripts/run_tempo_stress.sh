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
TRACE_SPAN_TARGET_BYTES="${TEMPO_TRACE_SPAN_TARGET_BYTES:-8192}"
TRACE_SPAN_PAUSE_MS="${TEMPO_TRACE_SPAN_PAUSE_MS:-0}"
TRACE_SPAN_PAUSE_EVERY="${TEMPO_TRACE_SPAN_PAUSE_EVERY:-0}"
SEARCH_PROBE_REQUESTS="${TEMPO_SEARCH_PROBE_REQUESTS:-20}"
SEARCH_PROBE_DELAY_MS="${TEMPO_SEARCH_PROBE_DELAY_MS:-100}"
OTEL_BSP_MAX_QUEUE_SIZE="${TEMPO_OTEL_BSP_MAX_QUEUE_SIZE:-8192}"
OTEL_BSP_MAX_EXPORT_BATCH_SIZE="${TEMPO_OTEL_BSP_MAX_EXPORT_BATCH_SIZE:-128}"
OTEL_BSP_SCHEDULE_DELAY="${TEMPO_OTEL_BSP_SCHEDULE_DELAY:-500}"
OTEL_BSP_EXPORT_TIMEOUT="${TEMPO_OTEL_BSP_EXPORT_TIMEOUT:-30000}"

function json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}

function compose_up_wait() {
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" up -d
    return 0
  fi
  docker compose -f "$COMPOSE_FILE" up -d --wait
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
  ,"otel_bsp_max_queue_size": $OTEL_BSP_MAX_QUEUE_SIZE
  ,"otel_bsp_max_export_batch_size": $OTEL_BSP_MAX_EXPORT_BATCH_SIZE
  ,"otel_bsp_schedule_delay": $OTEL_BSP_SCHEDULE_DELAY
  ,"otel_bsp_export_timeout": $OTEL_BSP_EXPORT_TIMEOUT
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
    for _ in $(seq 1 "$SEARCH_PROBE_REQUESTS"); do
      tempo_search_probe="$(json_http_probe 'http://localhost:3200/api/search?limit=20')"
      grafana_search_probe="$(json_http_probe 'http://localhost:3000/api/datasources/proxy/uid/tempo/api/search?limit=20' 'admin:admin')"
      cat >> "$query_probe_file" <<EOF
{"tempo": $tempo_search_probe, "grafana": $grafana_search_probe}
EOF
      if [[ "$SEARCH_PROBE_DELAY_MS" -gt 0 ]]; then
        sleep "$(python3 -c "print($SEARCH_PROBE_DELAY_MS/1000)")"
      fi
    done

    query_probe_summary="$(python3 - "$query_probe_file" <<'PY'
import json
import sys
from statistics import median

path = sys.argv[1]
tempo_latencies = []
grafana_latencies = []
tempo_failures = 0
grafana_failures = 0
samples = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        samples += 1
        row = json.loads(line)
        tempo = row.get("tempo", {})
        grafana = row.get("grafana", {})
        tempo_code = int(tempo.get("http_code", 0) or 0)
        grafana_code = int(grafana.get("http_code", 0) or 0)
        tempo_latencies.append(float(tempo.get("time_total", 0.0) or 0.0))
        grafana_latencies.append(float(grafana.get("time_total", 0.0) or 0.0))
        if tempo_code < 200 or tempo_code >= 300:
            tempo_failures += 1
        if grafana_code < 200 or grafana_code >= 300:
            grafana_failures += 1

def p95(values):
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, int((len(vals) - 1) * 0.95))
    return vals[idx]

summary = {
    "samples": samples,
    "tempo_failures": tempo_failures,
    "grafana_failures": grafana_failures,
    "tempo_p95_seconds": p95(tempo_latencies),
    "grafana_p95_seconds": p95(grafana_latencies),
    "tempo_median_seconds": median(tempo_latencies) if tempo_latencies else 0.0,
    "grafana_median_seconds": median(grafana_latencies) if grafana_latencies else 0.0,
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
  "tempo_probe": $tempo_probe,
  "grafana_probe": $grafana_probe,
  "collector_probe": $collector_probe,
  "search_probe": $query_probe_summary
}
EOF
  done
}

echo "==> Starting local Tempo + Grafana + OTel Collector stack"
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
