server:
  http_listen_port: 3200
  grpc_server_max_recv_msg_size: __TEMPO_GRPC_MAX_MSG_SIZE__
  grpc_server_max_send_msg_size: __TEMPO_GRPC_MAX_MSG_SIZE__

distributor:
  max_attribute_bytes: __TEMPO_MAX_ATTRIBUTE_BYTES__
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
          max_recv_msg_size_mib: __TEMPO_OTLP_GRPC_MAX_RECV_MSG_SIZE_MIB__
        http:
          endpoint: 0.0.0.0:4318

ingester:
  max_block_duration: 5m

ingester_client:
  grpc_client_config:
    max_recv_msg_size: __TEMPO_INTERNAL_GRPC_MAX_MSG_SIZE__
    max_send_msg_size: __TEMPO_INTERNAL_GRPC_MAX_MSG_SIZE__

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

querier:
  frontend_worker:
    grpc_client_config:
      max_recv_msg_size: __TEMPO_GRPC_MAX_MSG_SIZE__
      max_send_msg_size: __TEMPO_GRPC_MAX_MSG_SIZE__

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
