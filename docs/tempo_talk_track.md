# Tempo Talk Track: Why OSS Tempo Struggles With Agent Traces

## Setup Parity

This demo uses:
- Open source Tempo + open source Grafana (local single-machine deployment)
- OTLP trace ingestion from a realistic multi-agent workload (`pydantic-supervisor`)
- Synthetic payload amplification to mimic long agent trajectories (large tool-call/message payloads)

The goal is not to benchmark peak hardware limits; it is to validate whether a standard OSS Tempo deployment can reliably support trajectory-style AI traces under practical load.

## What We Measure

- Ingest reliability: application success/failure rate and OTEL exporter failures/failed spans
- Query reliability: Tempo search endpoint latency + HTTP status
- UI-adjacent reliability: Grafana datasource proxy query latency + HTTP status
- Resource stress signals: OOM kill and restart count on Tempo/Grafana/collector containers

## Decision Criteria

Tempo is considered not holding up when any sustained condition appears at realistic agent load:
- >5% failed requests from workload runner
- any exporter/failed span events
- p95 per-query runtime >10s
- repeated Tempo/Grafana query errors or severe latency spikes

## Why Agent Traces Are Different

Compared with traditional infra traces, agent traces often combine:
- deeper span trees from multi-agent handoffs and tool orchestration
- large text attributes (prompts, tool arguments, model outputs)
- high cardinality run metadata (model variants, route tags, customer/session tags)

This is primarily a data-shape and queryability challenge, not a charting/UI preference issue.

## Practical Implication For Customers

If the target user workflow is “inspect complete trajectories and search across them quickly,” failures usually appear first in query and retrieval behavior long before raw ingestion cost becomes the only concern.

Use `docs/tempo_stress_report.md` and `artifacts/tempo_stress_results.csv` as the evidence bundle for customer/commercial conversations.
