# Tempo Stress Report

## Summary

- Breakpoint stage: **None observed**
- Breakpoint rule: No unstable stage in synthetic ramp
- Dominant failure mode(s) at breakpoint: None
- Raw artifact directory: `artifacts/raw`
- Aggregated CSV: `artifacts/tempo_stress_results.csv`

## Environment Header

- Timestamp (UTC): `2026-03-25T19:20:05Z`
- Git SHA: `2cdd94ff1dee81d73260b5f58f51f96e44c59f99`
- Trace backend: `otlp`
- CPU: `Apple M4`
- Memory: `16.00 GiB`

## Stage Medians/Worst

| Stage | Runs | Median p95 (s) | Worst p95 (s) | Runs Failing Criteria |
|---|---:|---:|---:|---:|
| baseline | 1 | 0.000 | 0.000 | 0 |

## Breakpoint Analysis

- Synthetic ramp summary: runs=5, failed=4, fail_rate=80.00%, median_p95=0.000s, max_trace_target_bytes=1073741824
- Live sanity summary: n/a

## Failure Criteria

A run is marked failed when any of the following is true:
- request failure rate > 5%
- OTEL export failures or failed spans detected
- p95 question latency > 10s
- Tempo or Grafana query endpoint returned >= 500
- Tempo or Grafana query probe took > 10s

## Notes

- `scripts/run_tempo_stress.sh` updates raw artifacts and regenerates this report.
- Use `TRACE_RUN_TAG` in raw JSON payloads to correlate traces in Tempo UI/API.
