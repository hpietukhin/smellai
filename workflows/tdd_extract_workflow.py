#!/usr/bin/env python3
"""Technical Debt Dataset v2 extraction workflow.

This workflow materialises three reproducible artifacts:
- a schema snapshot markdown file
- a summary JSON report
- a transition JSONL sample or full extract

Usage:
    uv run workflows/tdd_extract_workflow.py \
        --db-path /path/to/td_V2.db \
        --project org.apache:commons-io \
        --limit 10
"""

from __future__ import annotations

from pathlib import Path

from smellai_datasets.td_v2 import (
    extract_transitions,
    write_schema_markdown,
    write_summary_json,
    write_transitions_jsonl,
)


def main(
    db_path: str,
    project: str | None = None,
    limit: int | None = 10,
    output: str = "outputs/td_v2_transitions_sample.jsonl",
    summary: str = "outputs/td_v2_summary.json",
    schema: str = "outputs/td_v2_schema.md",
) -> int:
    """Run the DB-first TDD v2 extraction pipeline locally."""
    db = Path(db_path).expanduser().resolve()
    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")

    schema_path = Path(schema)
    summary_path = Path(summary)
    output_path = Path(output)

    write_schema_markdown(schema_path, db)
    transitions, report = extract_transitions(db, project_id=project, limit=limit)
    write_transitions_jsonl(output_path, transitions)

    report.update(
        {
            "schema_path": str(schema_path),
            "summary_path": str(summary_path),
            "output_path": str(output_path),
            "project": project,
            "limit": limit,
        }
    )
    write_summary_json(summary_path, report)

    print(f"Schema written to {schema_path}")
    print(f"Summary written to {summary_path}")
    print(f"Transitions written to {output_path}")
    print(
        "Extracted "
        f"{report['transition_count']} transitions "
        f"(validation errors: {report['validation_error_count']})"
    )
    return 0


if __name__ == "__main__":
    import fire

    raise SystemExit(fire.Fire(main))
