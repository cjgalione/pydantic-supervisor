#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="infra/tempo/docker-compose.yml"
TEMPO_CONFIG_TEMPLATE="$ROOT_DIR/infra/tempo/tempo.dynamic.yaml.tpl"
TEMPO_CONFIG_RENDERED="$ROOT_DIR/infra/tempo/tempo.generated.yaml"
RAW_DIR="${TEMPO_BREAKPOINT_RAW_DIR:-artifacts/raw/breakpoint}"

mkdir -p "$RAW_DIR"

TRACE_BACKEND_VALUE="${TRACE_BACKEND:-otlp}"
OTEL_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}"
OTEL_PROTOCOL="${OTEL_EXPORTER_OTLP_PROTOCOL:-http/protobuf}"

# Tempo limits
TEMPO_INGEST_RATE_LIMIT_BYTES="${TEMPO_INGEST_RATE_LIMIT_BYTES:-100000000}"
TEMPO_INGEST_BURST_SIZE_BYTES="${TEMPO_INGEST_BURST_SIZE_BYTES:-400000000}"
TEMPO_MAX_BYTES_PER_TRACE="${TEMPO_MAX_BYTES_PER_TRACE:-10737418240}"
TEMPO_SEARCH_DURATION_SLO_SECONDS="${TEMPO_SEARCH_DURATION_SLO_SECONDS:-120}"
TEMPO_SEARCH_THROUGHPUT_BYTES_SLO="${TEMPO_SEARCH_THROUGHPUT_BYTES_SLO:-4294967296}"

# Breakpoint loop settings
START_SPAN_TARGET_BYTES="${TEMPO_BREAKPOINT_START_SPAN_TARGET_BYTES:-10485760}" # 10 MiB
SPAN_GROWTH_FACTOR="${TEMPO_BREAKPOINT_SPAN_GROWTH_FACTOR:-2}"
STEP_INTERVAL_SECONDS="${TEMPO_BREAKPOINT_STEP_INTERVAL_SECONDS:-60}"
MAX_STEPS="${TEMPO_BREAKPOINT_MAX_STEPS:-12}"
MIN_TRACE_TARGET_BYTES="${TEMPO_BREAKPOINT_MIN_TRACE_TARGET_BYTES:-1073741824}" # 1 GiB
SPANS_PER_TRACE="${TEMPO_BREAKPOINT_SPANS_PER_TRACE:-200}"
QUESTIONS_PER_STEP="${TEMPO_BREAKPOINT_QUESTIONS_PER_STEP:-1}"
CONCURRENCY_PER_STEP="${TEMPO_BREAKPOINT_CONCURRENCY_PER_STEP:-1}"
SEARCH_LIMIT="${TEMPO_BREAKPOINT_SEARCH_LIMIT:-20}"

OTEL_BSP_MAX_QUEUE_SIZE="${TEMPO_OTEL_BSP_MAX_QUEUE_SIZE:-32768}"
OTEL_BSP_MAX_EXPORT_BATCH_SIZE="${TEMPO_OTEL_BSP_MAX_EXPORT_BATCH_SIZE:-256}"
OTEL_BSP_SCHEDULE_DELAY="${TEMPO_OTEL_BSP_SCHEDULE_DELAY:-500}"
OTEL_BSP_EXPORT_TIMEOUT="${TEMPO_OTEL_BSP_EXPORT_TIMEOUT:-60000}"

TRACE_SPAN_PAUSE_MS="${TEMPO_TRACE_SPAN_PAUSE_MS:-10}"
TRACE_SPAN_PAUSE_EVERY="${TEMPO_TRACE_SPAN_PAUSE_EVERY:-100}"
TRACE_SPAN_ATTRIBUTE_LIMIT="${TRACE_SPAN_ATTRIBUTE_LIMIT:-50000}"
TRACE_SPAN_ATTRIBUTE_VALUE_LIMIT="${TRACE_SPAN_ATTRIBUTE_VALUE_LIMIT:-131072}"

function compose_up_wait() {
  export TEMPO_CONFIG_PATH="$TEMPO_CONFIG_RENDERED"
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" up -d
    return 0
  fi
  docker compose -f "$COMPOSE_FILE" up -d --wait
}

function render_tempo_config() {
  python3 - \
    "$TEMPO_CONFIG_TEMPLATE" \
    "$TEMPO_CONFIG_RENDERED" \
    "$TEMPO_INGEST_RATE_LIMIT_BYTES" \
    "$TEMPO_INGEST_BURST_SIZE_BYTES" \
    "$TEMPO_MAX_BYTES_PER_TRACE" \
    "$TEMPO_SEARCH_DURATION_SLO_SECONDS" \
    "$TEMPO_SEARCH_THROUGHPUT_BYTES_SLO" <<'PY'
import pathlib
import sys

template = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
rendered = template
keys = (
    "TEMPO_INGEST_RATE_LIMIT_BYTES",
    "TEMPO_INGEST_BURST_SIZE_BYTES",
    "TEMPO_MAX_BYTES_PER_TRACE",
    "TEMPO_SEARCH_DURATION_SLO_SECONDS",
    "TEMPO_SEARCH_THROUGHPUT_BYTES_SLO",
)
values = sys.argv[3:3 + len(keys)]
for key, value in zip(keys, values):
    rendered = rendered.replace(f"__{key}__", str(value))
pathlib.Path(sys.argv[2]).write_text(rendered, encoding="utf-8")
PY
}

function wait_for_http_200() {
  local name="$1"
  local url="$2"
  local auth="${3:-}"
  local max_attempts="${4:-40}"
  local sleep_seconds="${5:-2}"
  for attempt in $(seq 1 "$max_attempts"); do
    local code=0
    if [[ -n "$auth" ]]; then
      code="$(curl -sS -u "$auth" -o /dev/null -w '%{http_code}' "$url" || echo 0)"
    else
      code="$(curl -sS -o /dev/null -w '%{http_code}' "$url" || echo 0)"
    fi
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

function json_http_probe_traceql() {
  local url="$1"
  local auth="$2"
  local traceql="$3"
  local limit="$4"
  local body_file
  body_file="$(mktemp)"
  local probe_json='{"http_code":0,"time_total":0.0}'
  if [[ -n "$auth" ]]; then
    probe_json="$(curl -sS -u "$auth" --get \
      --data-urlencode "q=$traceql" \
      --data "limit=$limit" \
      -o "$body_file" \
      -w '{"http_code":%{http_code},"time_total":%{time_total}}' \
      "$url" || echo '{"http_code":0,"time_total":0.0}')"
  else
    probe_json="$(curl -sS --get \
      --data-urlencode "q=$traceql" \
      --data "limit=$limit" \
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
out = {"http_code": 0, "time_total": 0.0, "trace_count": 0, "sampled_at_unix_ms": int(time.time() * 1000)}
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
def walk(value):
    global trace_count
    if isinstance(value, dict):
        trace_id = value.get("traceID") or value.get("traceId")
        if trace_id:
            trace_count += 1
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

walk(body)
out["trace_count"] = trace_count
print(json.dumps(out))
PY
  rm -f "$body_file"
}

echo "==> Rendering Tempo config with max_bytes_per_trace=${TEMPO_MAX_BYTES_PER_TRACE}"
render_tempo_config
echo "==> Ensuring Tempo stack is running"
compose_up_wait
echo "==> Waiting for stack health"
wait_for_http_200 "Tempo search endpoint" "http://localhost:3200/api/search?limit=20"
wait_for_http_200 "Grafana Tempo datasource proxy" "http://localhost:3000/api/datasources/proxy/uid/tempo/api/search?limit=20" "admin:admin"

echo "==> Starting breakpoint loop"
echo "==> start_span_target_bytes=${START_SPAN_TARGET_BYTES} spans_per_trace=${SPANS_PER_TRACE} min_trace_target_bytes=${MIN_TRACE_TARGET_BYTES} max_steps=${MAX_STEPS}"

span_target_bytes="$START_SPAN_TARGET_BYTES"
for step in $(seq 1 "$MAX_STEPS"); do
  step_started="$(date +%s)"
  run_tag="breakpoint-step${step}-$(date +%s)"

  computed_trace_target=$((span_target_bytes * SPANS_PER_TRACE))
  trace_target_bytes="$computed_trace_target"
  if [[ "$trace_target_bytes" -lt "$MIN_TRACE_TARGET_BYTES" ]]; then
    trace_target_bytes="$MIN_TRACE_TARGET_BYTES"
  fi

  metrics_file="$RAW_DIR/step${step}.json"
  manifest_file="$RAW_DIR/step${step}_trace_manifest.json"
  probe_file="$RAW_DIR/step${step}_search_probe.json"

  echo "==> step=${step} run_tag=${run_tag} span_target_bytes=${span_target_bytes} trace_target_bytes=${trace_target_bytes}"

  set +e
  TRACE_BACKEND="$TRACE_BACKEND_VALUE" \
  TRACE_RUN_TAG="$run_tag" \
  TRACE_SPAN_TARGET_BYTES="$span_target_bytes" \
  TRACE_SYNTHETIC_MIN_SPANS="$SPANS_PER_TRACE" \
  TRACE_SPAN_PAUSE_MS="$TRACE_SPAN_PAUSE_MS" \
  TRACE_SPAN_PAUSE_EVERY="$TRACE_SPAN_PAUSE_EVERY" \
  TRACE_SPAN_ATTRIBUTE_LIMIT="$TRACE_SPAN_ATTRIBUTE_LIMIT" \
  TRACE_SPAN_ATTRIBUTE_VALUE_LIMIT="$TRACE_SPAN_ATTRIBUTE_VALUE_LIMIT" \
  OTEL_BSP_MAX_QUEUE_SIZE="$OTEL_BSP_MAX_QUEUE_SIZE" \
  OTEL_BSP_MAX_EXPORT_BATCH_SIZE="$OTEL_BSP_MAX_EXPORT_BATCH_SIZE" \
  OTEL_BSP_SCHEDULE_DELAY="$OTEL_BSP_SCHEDULE_DELAY" \
  OTEL_BSP_EXPORT_TIMEOUT="$OTEL_BSP_EXPORT_TIMEOUT" \
  OTEL_EXPORTER_OTLP_ENDPOINT="$OTEL_ENDPOINT" \
  OTEL_EXPORTER_OTLP_PROTOCOL="$OTEL_PROTOCOL" \
  python3 scripts/run_queries.py \
    --question-source bank \
    --questions "$QUESTIONS_PER_STEP" \
    --rounds 1 \
    --concurrency "$CONCURRENCY_PER_STEP" \
    --payload-profile xlarge \
    --inject-large-attributes \
    --synthetic-only \
    --trace-target-bytes "$trace_target_bytes" \
    --trace-run-tag "$run_tag" \
    --metrics-output "$metrics_file" \
    --trace-manifest-output "$manifest_file" \
    --no-quota-preflight
  run_exit=$?
  set -e

  run_filter=".stress_run_tag=\"$run_tag\""
  generic_q="{$run_filter}"
  span_name_q="{name=\"synthetic_step_000\" && $run_filter}"
  service_q="{resource.service.name=\"pydantic-supervisor\" && $run_filter}"
  dot_service_q="{.service.name=\"pydantic-supervisor\" && $run_filter}"

  tempo_generic="$(json_http_probe_traceql 'http://localhost:3200/api/search' '' "$generic_q" "$SEARCH_LIMIT")"
  tempo_span_name="$(json_http_probe_traceql 'http://localhost:3200/api/search' '' "$span_name_q" "$SEARCH_LIMIT")"
  tempo_service="$(json_http_probe_traceql 'http://localhost:3200/api/search' '' "$service_q" "$SEARCH_LIMIT")"
  tempo_dot_service="$(json_http_probe_traceql 'http://localhost:3200/api/search' '' "$dot_service_q" "$SEARCH_LIMIT")"
  grafana_generic="$(json_http_probe_traceql 'http://localhost:3000/api/datasources/proxy/uid/tempo/api/search' 'admin:admin' "$generic_q" "$SEARCH_LIMIT")"

  cat > "$probe_file" <<EOF
{
  "step": $step,
  "run_tag": "$run_tag",
  "span_target_bytes": $span_target_bytes,
  "trace_target_bytes": $trace_target_bytes,
  "tempo_generic": $tempo_generic,
  "tempo_span_name": $tempo_span_name,
  "tempo_service": $tempo_service,
  "tempo_dot_service": $tempo_dot_service,
  "grafana_generic": $grafana_generic
}
EOF

  read -r failures export_failures failed_spans <<< "$(python3 - "$metrics_file" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, "r", encoding="utf-8")) if p else {}
o = d.get("otel_export_stats") or {}
print(int(d.get("failures", 0) or 0), int(o.get("export_failures", 0) or 0), int(o.get("failed_spans", 0) or 0))
PY
)"

  read -r tempo_code tempo_hits <<< "$(python3 - "$probe_file" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], "r", encoding="utf-8"))
g = d.get("tempo_generic") or {}
print(int(g.get("http_code", 0) or 0), int(g.get("trace_count", 0) or 0))
PY
)"

  echo "   step=${step} run_exit=${run_exit} failures=${failures} export_failures=${export_failures} failed_spans=${failed_spans} tempo_generic_code=${tempo_code} tempo_generic_hits=${tempo_hits}"

  if [[ "$run_exit" -ne 0 ]]; then
    echo "==> STOP: run_queries exited non-zero at step=${step}"
    break
  fi
  if [[ "$failures" -gt 0 || "$export_failures" -gt 0 || "$failed_spans" -gt 0 ]]; then
    echo "==> STOP: failures detected at step=${step}"
    break
  fi
  if [[ "$tempo_code" -ne 200 || "$tempo_hits" -le 0 ]]; then
    echo "==> STOP: search verification failed at step=${step}"
    break
  fi

  step_elapsed=$(( $(date +%s) - step_started ))
  sleep_seconds=$(( STEP_INTERVAL_SECONDS - step_elapsed ))
  if [[ "$sleep_seconds" -gt 0 ]]; then
    echo "   sleeping ${sleep_seconds}s before next ramp"
    sleep "$sleep_seconds"
  fi

  span_target_bytes=$(( span_target_bytes * SPAN_GROWTH_FACTOR ))
done

echo "==> Breakpoint loop finished"
echo "==> Results are in $RAW_DIR"
