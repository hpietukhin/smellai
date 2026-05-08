#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import orjson


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Sousa script outputs to minimal case manifest")
    p.add_argument("--input", required=True, help="Path to JSON output from detect_* script")
    p.add_argument("--heuristic", required=True, choices=["element", "commit", "range"])
    p.add_argument("--project", required=True, help="Project label for case_id")
    p.add_argument("--top-n", type=int, default=3, help="How many refs/elements to use in key")
    p.add_argument("--output", required=True, help="Output JSON path")

    p.add_argument("--log-mlflow", action="store_true", help="Also log manifest source via mlflow.data")
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="planner-eval-case-sources")
    p.add_argument("--run-name", default="case-source-convert")
    return p.parse_args()


def _stable_short_hash(parts: list[str]) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


def _batch_key(batch: list[dict], heuristic: str, top_n: int) -> str:
    assert batch, "batch must be non-empty"

    if heuristic == "element":
        elems = sorted({str(r["element"]).strip() for r in batch if "element" in r and str(r["element"]).strip()})
        if not elems:
            raise ValueError("element heuristic requires non-empty 'element' fields")
        return ",".join(elems[:top_n])

    if heuristic == "commit":
        commits = sorted({str(r["commit_id"]).strip() for r in batch if "commit_id" in r and str(r["commit_id"]).strip()})
        if not commits:
            raise ValueError("commit heuristic requires non-empty 'commit_id' fields")
        return ",".join(commits[:top_n])

    # range
    elems = sorted({str(r.get("element", "")).strip() for r in batch if str(r.get("element", "")).strip()})
    if elems:
        return ",".join(elems[:top_n])
    refs = sorted({str(r["ref_id"]).strip() for r in batch if "ref_id" in r and str(r["ref_id"]).strip()})
    if refs:
        return ",".join(refs[:top_n])
    raise ValueError("range heuristic requires non-empty 'element' or 'ref_id' fields")


def _to_cases(raw: list, heuristic: str, project: str, top_n: int) -> list[dict]:
    cases: list[dict] = []
    for i, batch in enumerate(raw):
        if not isinstance(batch, list):
            raise TypeError(f"batch at index {i} must be a list")
        key = _batch_key(batch, heuristic=heuristic, top_n=top_n)
        cid = f"{heuristic}:{project}:{_stable_short_hash([key, str(i)])}"
        cases.append(
            {
                "case_id": cid,
                "heuristic": heuristic,
                "project": project,
                "case_key": key,
                "size": len(batch),
                "raw_index": i,
            }
        )
    return cases


def _maybe_log_mlflow(cases: list[dict], args: argparse.Namespace) -> None:
    import mlflow
    import pandas as pd
    from mlflow.data import from_pandas
    from mlflow.utils.file_utils import local_file_uri_to_path

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    df = pd.DataFrame(cases)
    with mlflow.start_run(run_name=args.run_name):
        ds = from_pandas(df, source=local_file_uri_to_path(Path(args.input).resolve().as_uri()), name="case_manifest_source")
        mlflow.log_input(ds, context="evaluation")
        mlflow.log_metric("cases.count", float(len(cases)))


def main() -> int:
    args = _parse_args()

    raw = orjson.loads(Path(args.input).read_bytes())
    assert isinstance(raw, list), "Input JSON must be a list of batches"

    cases = _to_cases(raw, heuristic=args.heuristic, project=args.project, top_n=args.top_n)
    out = {"cases": cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_bytes(orjson.dumps(out, option=orjson.OPT_INDENT_2))

    if args.log_mlflow:
        _maybe_log_mlflow(cases, args)

    print(f"wrote {len(cases)} cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
