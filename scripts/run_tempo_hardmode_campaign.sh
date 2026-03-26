#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="infra/tempo/docker-compose.yml"
TEMPO_CONFIG_TEMPLATE="$ROOT_DIR/infra/tempo/tempo.dynamic.yaml.tpl"
TEMPO_CONFIG_RENDERED="$ROOT_DIR/infra/tempo/tempo.generated.yaml"
RAW_DIR="${TEMPO_HARDMODE_RAW_DIR:-artifacts/raw/hardmode_$(date +%s)}"
mkdir -p "$RAW_DIR"

# Tempo limits / runtime posture
TEMPO_INGEST_RATE_LIMIT_BYTES="${TEMPO_INGEST_RATE_LIMIT_BYTES:-100000000}"
TEMPO_INGEST_BURST_SIZE_BYTES="${TEMPO_INGEST_BURST_SIZE_BYTES:-400000000}"
TEMPO_MAX_BYTES_PER_TRACE="${TEMPO_MAX_BYTES_PER_TRACE:-10737418240}"
TEMPO_MAX_ATTRIBUTE_BYTES="${TEMPO_MAX_ATTRIBUTE_BYTES:-104857600}"
TEMPO_GRPC_MAX_MSG_SIZE="${TEMPO_GRPC_MAX_MSG_SIZE:-2147483647}"
TEMPO_INTERNAL_GRPC_MAX_MSG_SIZE="${TEMPO_INTERNAL_GRPC_MAX_MSG_SIZE:-$TEMPO_GRPC_MAX_MSG_SIZE}"
TEMPO_OTLP_GRPC_MAX_RECV_MSG_SIZE_MIB="${TEMPO_OTLP_GRPC_MAX_RECV_MSG_SIZE_MIB:-1024}"
TEMPO_SEARCH_DURATION_SLO_SECONDS="${TEMPO_SEARCH_DURATION_SLO_SECONDS:-120}"
TEMPO_SEARCH_THROUGHPUT_BYTES_SLO="${TEMPO_SEARCH_THROUGHPUT_BYTES_SLO:-4294967296}"

# Campaign shape
STAGE_SEQUENCE_BYTES="${TEMPO_HARDMODE_STAGE_SEQUENCE_BYTES:-102400,512000,1048576,10485760,52428800,104857600}"
POST_PLATEAU_STEPS="${TEMPO_HARDMODE_POST_PLATEAU_STEPS:-12}"
MAX_STEPS="${TEMPO_HARDMODE_MAX_STEPS:-0}"
STEP_INTERVAL_SECONDS="${TEMPO_HARDMODE_STEP_INTERVAL_SECONDS:-60}"
START_SPANS_PER_TRACE="${TEMPO_HARDMODE_START_SPANS_PER_TRACE:-100}"
SPANS_PER_TRACE_INCREMENT="${TEMPO_HARDMODE_SPANS_PER_TRACE_INCREMENT:-10}"
TRACE_COUNT_PER_STEP="${TEMPO_HARDMODE_TRACE_COUNT_PER_STEP:-1}"
CHUNK_SIZE_BYTES="${TEMPO_HARDMODE_CHUNK_SIZE_BYTES:-1048576}"
PAYLOAD_SEED="${TEMPO_HARDMODE_PAYLOAD_SEED:-tempo-hardmode-v1}"

# Replay / validation controls
OTLP_ENDPOINT="${TEMPO_HARDMODE_OTLP_ENDPOINT:-localhost:4317}"
TEMPO_BASE_URL="${TEMPO_HARDMODE_TEMPO_BASE_URL:-http://localhost:3200}"
GRAFANA_BASE_URL="${TEMPO_HARDMODE_GRAFANA_BASE_URL:-http://localhost:3000/api/datasources/proxy/uid/tempo}"
GRAFANA_AUTH="${TEMPO_HARDMODE_GRAFANA_AUTH:-admin:admin}"
REPLAY_CONCURRENCY="${TEMPO_HARDMODE_REPLAY_CONCURRENCY:-4}"
REPLAY_SPANS_PER_REQUEST_SMALL="${TEMPO_HARDMODE_REPLAY_SPANS_PER_REQUEST_SMALL:-10}"
REPLAY_LARGE_SPAN_THRESHOLD_BYTES="${TEMPO_HARDMODE_REPLAY_LARGE_SPAN_THRESHOLD_BYTES:-52428800}"
REPLAY_MAX_MESSAGE_BYTES="${TEMPO_HARDMODE_REPLAY_MAX_MESSAGE_BYTES:-2147483647}"
VALIDATION_QUERY_TIMEOUT_SECONDS="${TEMPO_HARDMODE_VALIDATION_QUERY_TIMEOUT_SECONDS:-300}"
VALIDATION_POLL_INTERVAL_SECONDS="${TEMPO_HARDMODE_VALIDATION_POLL_INTERVAL_SECONDS:-2}"

SEARCH_LIMIT="${TEMPO_HARDMODE_SEARCH_LIMIT:-20}"
SEARCH_PROBE_REQUESTS="${TEMPO_HARDMODE_SEARCH_PROBE_REQUESTS:-10}"
SEARCH_PROBE_DELAY_MS="${TEMPO_HARDMODE_SEARCH_PROBE_DELAY_MS:-100}"

function compose_up_wait() {
  export TEMPO_CONFIG_PATH="$TEMPO_CONFIG_RENDERED"
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" up -d --force-recreate
    return 0
  fi
  docker compose -f "$COMPOSE_FILE" up -d --wait --force-recreate
}

function render_tempo_config() {
  python3 - \
    "$TEMPO_CONFIG_TEMPLATE" \
    "$TEMPO_CONFIG_RENDERED" \
    "$TEMPO_INGEST_RATE_LIMIT_BYTES" \
    "$TEMPO_INGEST_BURST_SIZE_BYTES" \
    "$TEMPO_MAX_BYTES_PER_TRACE" \
    "$TEMPO_MAX_ATTRIBUTE_BYTES" \
    "$TEMPO_GRPC_MAX_MSG_SIZE" \
    "$TEMPO_INTERNAL_GRPC_MAX_MSG_SIZE" \
    "$TEMPO_OTLP_GRPC_MAX_RECV_MSG_SIZE_MIB" \
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
    "TEMPO_MAX_ATTRIBUTE_BYTES",
    "TEMPO_GRPC_MAX_MSG_SIZE",
    "TEMPO_INTERNAL_GRPC_MAX_MSG_SIZE",
    "TEMPO_OTLP_GRPC_MAX_RECV_MSG_SIZE_MIB",
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

function run_search_probes() {
  local run_tag="$1"
  local out_file="$2"
  python3 - "$run_tag" "$out_file" "$SEARCH_LIMIT" "$SEARCH_PROBE_REQUESTS" "$SEARCH_PROBE_DELAY_MS" "$TEMPO_BASE_URL" "$GRAFANA_BASE_URL" "$GRAFANA_AUTH" <<'PY'
import json
import pathlib
import statistics
import sys
import time

from src.tempo_hardmode_otlp import traceql_search_probe

run_tag, out_file, search_limit, probe_requests, probe_delay_ms, tempo_base_url, grafana_base_url, grafana_auth = sys.argv[1:]
search_limit = int(search_limit)
probe_requests = int(probe_requests)
probe_delay_ms = int(probe_delay_ms)

queries = {
    "generic": f'{{.stress_run_tag="{run_tag}"}}',
    "span_name": f'{{name="hardmode_span_0000" && .stress_run_tag="{run_tag}"}}',
    "svc_field": f'{{resource.service.name="pydantic-supervisor-hardmode" && .stress_run_tag="{run_tag}"}}',
    "dot_svc_field": f'{{.service.name="pydantic-supervisor-hardmode" && .stress_run_tag="{run_tag}"}}',
}

summary = {"run_tag": run_tag, "classes": {}}
for cls, traceql in queries.items():
    tempo_samples = []
    grafana_samples = []
    for _ in range(probe_requests):
        tempo_samples.append(
            traceql_search_probe(
                base_url=tempo_base_url,
                traceql=traceql,
                limit=search_limit,
                timeout_seconds=10.0,
            )
        )
        grafana_samples.append(
            traceql_search_probe(
                base_url=grafana_base_url,
                traceql=traceql,
                limit=search_limit,
                timeout_seconds=10.0,
                basic_auth=grafana_auth,
            )
        )
        if probe_delay_ms > 0:
            time.sleep(probe_delay_ms / 1000.0)

    tempo_lat = [float(x["time_total_seconds"]) for x in tempo_samples]
    grafana_lat = [float(x["time_total_seconds"]) for x in grafana_samples]
    tempo_failures = sum(1 for x in tempo_samples if int(x["http_code"]) != 200 or int(x["trace_count"]) <= 0)
    grafana_failures = sum(1 for x in grafana_samples if int(x["http_code"]) != 200 or int(x["trace_count"]) <= 0)
    summary["classes"][cls] = {
        "samples": probe_requests,
        "tempo_failures": tempo_failures,
        "grafana_failures": grafana_failures,
        "tempo_p95_seconds": statistics.quantiles(tempo_lat, n=20)[-1] if len(tempo_lat) > 1 else (tempo_lat[0] if tempo_lat else 0.0),
        "grafana_p95_seconds": statistics.quantiles(grafana_lat, n=20)[-1] if len(grafana_lat) > 1 else (grafana_lat[0] if grafana_lat else 0.0),
    }

summary["overall_failure_count"] = sum(v["tempo_failures"] + v["grafana_failures"] for v in summary["classes"].values())
path = pathlib.Path(out_file)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary))
PY
}

echo "==> Rendering Tempo config for hard-mode campaign"
render_tempo_config
echo "==> Ensuring Tempo stack is running"
compose_up_wait
echo "==> Waiting for stack health"
wait_for_http_200 "Tempo search endpoint" "${TEMPO_BASE_URL}/api/search?limit=20"
wait_for_http_200 "Grafana Tempo datasource proxy" "${GRAFANA_BASE_URL}/api/search?limit=20" "$GRAFANA_AUTH"

cp "$TEMPO_CONFIG_RENDERED" "$RAW_DIR/tempo_generated_config.yaml"
curl -sS "${TEMPO_BASE_URL}/status/config" > "$RAW_DIR/tempo_active_config_step0.yaml"

IFS=',' read -r -a _seq_tokens <<< "$STAGE_SEQUENCE_BYTES"
declare -a SEQ=()
for tok in "${_seq_tokens[@]}"; do
  tok="${tok//[[:space:]]/}"
  if [[ "$tok" =~ ^[0-9]+$ ]] && [[ "$tok" -gt 0 ]]; then
    SEQ+=("$tok")
  fi
done
if [[ "${#SEQ[@]}" -eq 0 ]]; then
  echo "ERROR: stage sequence is empty"
  exit 1
fi

last_span_bytes="${SEQ[$(( ${#SEQ[@]} - 1 ))]}"
default_total_steps=$(( ${#SEQ[@]} + POST_PLATEAU_STEPS ))
if [[ "$MAX_STEPS" -gt 0 ]] && [[ "$MAX_STEPS" -lt "$default_total_steps" ]]; then
  total_steps="$MAX_STEPS"
else
  total_steps="$default_total_steps"
fi

spans_per_trace="$START_SPANS_PER_TRACE"
echo "==> Starting hard-mode campaign raw_dir=$RAW_DIR total_steps=$total_steps"
for step in $(seq 1 "$total_steps"); do
  step_started_epoch="$(date +%s)"
  run_tag="hardmode-step${step}-$(date +%s)"
  stage_name="step${step}"

  if [[ "$step" -le "${#SEQ[@]}" ]]; then
    span_payload_bytes="${SEQ[$((step - 1))]}"
  else
    span_payload_bytes="$last_span_bytes"
    spans_per_trace=$((spans_per_trace + SPANS_PER_TRACE_INCREMENT))
  fi

  if [[ "$span_payload_bytes" -ge "$REPLAY_LARGE_SPAN_THRESHOLD_BYTES" ]]; then
    spans_per_request=1
  else
    spans_per_request="$REPLAY_SPANS_PER_REQUEST_SMALL"
  fi

  dataset_manifest="$RAW_DIR/${stage_name}_dataset_manifest.json"
  replay_results="$RAW_DIR/${stage_name}_replay_results.json"
  validation_output="$RAW_DIR/${stage_name}_validation.json"
  search_probe_output="$RAW_DIR/${stage_name}_search_probe.json"
  step_config_output="$RAW_DIR/${stage_name}_tempo_active_config.yaml"

  echo "==> ramp step=${step} run_tag=${run_tag} span_payload_bytes=${span_payload_bytes} spans_per_trace=${spans_per_trace} trace_count=${TRACE_COUNT_PER_STEP} spans_per_request=${spans_per_request}"
  curl -sS "${TEMPO_BASE_URL}/status/config" > "$step_config_output"

  python3 scripts/generate_otlp_dataset.py \
    --output-dir "$RAW_DIR" \
    --stage-name "$stage_name" \
    --run-tag "$run_tag" \
    --trace-count "$TRACE_COUNT_PER_STEP" \
    --spans-per-trace "$spans_per_trace" \
    --span-payload-bytes "$span_payload_bytes" \
    --chunk-size-bytes "$CHUNK_SIZE_BYTES" \
    --payload-seed "$PAYLOAD_SEED" \
    --manifest-path "$dataset_manifest"

  set +e
  python3 scripts/replay_otlp_dataset.py \
    --manifest "$dataset_manifest" \
    --endpoint "$OTLP_ENDPOINT" \
    --timeout-seconds 120 \
    --concurrency "$REPLAY_CONCURRENCY" \
    --spans-per-request "$spans_per_request" \
    --max-message-bytes "$REPLAY_MAX_MESSAGE_BYTES" \
    --compression none \
    --results-output "$replay_results"
  replay_exit=$?
  set -e
  if [[ "$replay_exit" -ne 0 ]]; then
    echo "==> STOP: replay transport failure at step=${step}"
    break
  fi

  set +e
  python3 scripts/validate_tempo_trace_integrity.py \
    --manifest "$replay_results" \
    --tempo-base-url "$TEMPO_BASE_URL" \
    --grafana-base-url "$GRAFANA_BASE_URL" \
    --grafana-auth "$GRAFANA_AUTH" \
    --query-timeout-seconds "$VALIDATION_QUERY_TIMEOUT_SECONDS" \
    --poll-interval-seconds "$VALIDATION_POLL_INTERVAL_SECONDS" \
    --output "$validation_output"
  validate_exit=$?
  set -e

  run_search_probes "$run_tag" "$search_probe_output"

  if [[ "$validate_exit" -ne 0 ]]; then
    echo "==> STOP: integrity mismatch at step=${step}"
    break
  fi

  step_elapsed=$(( $(date +%s) - step_started_epoch ))
  sleep_seconds=$(( STEP_INTERVAL_SECONDS - step_elapsed ))
  if [[ "$sleep_seconds" -gt 0 ]]; then
    echo "   sleeping ${sleep_seconds}s before next ramp"
    sleep "$sleep_seconds"
  fi
done

echo "==> Hard-mode campaign finished"
echo "==> Results are in $RAW_DIR"
