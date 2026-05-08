# How to evaluate (Composite Refactorings 2020)

Troubleshooting and runtime pitfalls: see `evals/troubleshoot.md`.

## 0) Runnable eval policy

Runnable evals use only these **5 safe Maven repos**:

- `JUnit4`
- `Lyra`
- `OkHttp`
- `PhiCode Philib`
- `Tap4j`

A case only enters a batch run after baseline verification on its exact `start_commit_hash`:

- checkout succeeds
- eval patches apply
- baseline Java build/tests pass with the same command family used by the full workflow

So there is no separate subset gate anymore.

---

## 1) Main files

### `batch-list.json`
The only batch-run input.

Contains only runnable cases with:
- `project`
- `repo_url`
- `start_commit_hash`
- `elements`
- `baseline_verification.status == "passed"`

### `candidate-report.json`
Audit/debug artifact from generation.

Contains:
- accepted cases
- rejected cases
- rejection reasons like `checkout_fail`, `patch_fail`, `build_fail`, `test_fail`

---

## 2) Generate a batch list

Use the generator instead of hand-writing JSON:

```bash
uv run python evals/generate_batch_list.py \
  --projects "JUnit4,Lyra,OkHttp,PhiCode Philib,Tap4j" \
  --heuristic range \
  --limit-per-project 5 \
  --output-batch-list outputs/evals/safe_maven_range_batch_list.json \
  --output-report outputs/evals/safe_maven_range_candidate_report.json
```

What it does:
1. mines range-based start situations from Neo4j
2. keeps only non-neutral smell trajectories
3. removes outliers
4. verifies each candidate baseline on `start_commit_hash`
5. writes only runnable cases into the batch list

---

## 3) Run the full workflow over a batch list

`workflows/composite_workflow_full.py batch`

```bash
uv run python workflows/composite_workflow_full.py batch \
  --config evals/config/full_eval_openrouter_minimax_file_only.json
```

The config now points at a `batch_list`, not a manifest/subset pair.

Example workflow config shape:

```json
{
  "workflow": {
    "batch_list": "outputs/evals/safe_maven_range_batch_list.json",
    "experiment": "composite_workflow_full",
    "planner": "befs",
    "detector_backend": "organic",
    "locality": "none",
    "model": "openrouter/minimax/minimax-m2.7",
    "concurrency": 2,
    "max_steps": 5,
    "max_no_progress": 2,
    "retry_budget": 1,
    "timeout": 300
  }
}
```

If `run_name_prefix` is omitted, batch runs get a searchable auto-prefix like:

```text
full-20260505-batch-safe_maven_range_batch_list-planner-befs-det-organic-loc-none-model-openrouter_minimax_minimax-m2.7-steps-5-case-1
```

You can still set `run_name_prefix` explicitly for ad-hoc labels.

---

## 4) Single-case full run

If you want to run one verified case directly:

```bash
uv run python workflows/composite_workflow_full.py \
  --project "JUnit4" \
  --repo-url "https://github.com/junit-team/junit4.git" \
  --elements '["junit.tests.framework.AssertTest.testAssertNaNEqualsFails"]' \
  --start-commit-hash "d3b3a19c78435ef6b0d1c8832bcdb1a8d5ed6a4e" \
  --model "claude-sonnet-4-5-20250929"
```

---

## 5) Practical policy

- Primary methodology: `range-based`
- Use generated batch lists, not hand-written manifests
- Keep full-workflow refactoring scope at `file`
- Use the candidate report to inspect why potential cases were rejected
