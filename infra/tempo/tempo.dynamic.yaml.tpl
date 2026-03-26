server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318

ingester:
  max_block_duration: 5m

compactor:
  compaction:
    block_retention: 24h

storage:
  trace:
    backend: local
    local:
      path: /var/tempo/traces

query_frontend:
  search:
    duration_slo: __TEMPO_SEARCH_DURATION_SLO_SECONDS__s
    throughput_bytes_slo: __TEMPO_SEARCH_THROUGHPUT_BYTES_SLO__

metrics_generator:
  registry:
    external_labels:
      source: pydantic-supervisor-stress

overrides:
  defaults:
    ingestion:
      burst_size_bytes: __TEMPO_INGEST_BURST_SIZE_BYTES__
      rate_limit_bytes: __TEMPO_INGEST_RATE_LIMIT_BYTES__
    global:
      max_bytes_per_trace: __TEMPO_MAX_BYTES_PER_TRACE__
