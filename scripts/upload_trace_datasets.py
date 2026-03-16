"""Upload the curated trace datasets to Braintrust datasets."""

from __future__ import annotations

import json
from pathlib import Path

from braintrust import init_dataset
from dotenv import load_dotenv

PROJECT_NAME = "pydantic-supervisor"
ORG_NAME = "Braintrust Demos"

DATASETS = {
    "Supervisor Trace Dataset": Path("datasets/supervisor_trace_dataset.jsonl"),
    "Research Trace Dataset": Path("datasets/research_trace_dataset.jsonl"),
    "Math Trace Dataset": Path("datasets/math_trace_dataset.jsonl"),
}


def _row_id(row: dict) -> str:
    metadata = row.get("metadata") or {}
    source_span_id = metadata.get("source_span_id")
    if source_span_id:
        return str(source_span_id)
    return json.dumps(row.get("input"), sort_keys=True)


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def upload_dataset(name: str, path: Path) -> None:
    rows = _load_rows(path)
    dataset = init_dataset(project=PROJECT_NAME, name=name, org_name=ORG_NAME)
    existing = {str(item["id"]) for item in dataset.fetch()}
    desired = {_row_id(row) for row in rows}

    for row in rows:
        row_id = _row_id(row)
        if row_id in existing:
            dataset.update(
                id=row_id,
                input=row.get("input"),
                expected=row.get("expected"),
                metadata=row.get("metadata"),
            )
        else:
            dataset.insert(
                id=row_id,
                input=row.get("input"),
                expected=row.get("expected"),
                metadata=row.get("metadata"),
            )

    for row_id in sorted(existing - desired):
        dataset.delete(row_id)

    dataset.flush()
    print(f"uploaded {len(rows)} rows to {name}")


def main() -> None:
    load_dotenv()
    for name, path in DATASETS.items():
        upload_dataset(name, path)


if __name__ == "__main__":
    main()
