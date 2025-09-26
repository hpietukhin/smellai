<!--
Sync Impact Report
Version change: 1.0.0 → 1.1.0
Modified principles: II. Baseline Parity and Comparisons → explicit Dockerized SonarQube baseline requirement
Added sections: None
Removed sections: None
Templates requiring updates:
 - ✅ .specify/templates/plan-template.md (version reference updated to v1.1.0)
 - ✅ .specify/templates/spec-template.md (no outdated references)
 - ✅ .specify/templates/tasks-template.md (no outdated references)
 - ⚠  .specify/templates/commands/* (directory not present; no action)
Follow-up TODOs: Consider adding `infra/sonarqube/docker-compose.yml` during implementation
-->

# SmellAI Thesis Lab Constitution

## Core Principles

### I. Reproducible Experimentation (NON-NEGOTIABLE)
- All experiments MUST run via importable LangGraph pipelines in `src/` at a fixed
  Git commit (log full SHA).
- Each run MUST be tracked in Weights & Biases (W&B), logging: run ID, Git SHA,
  dataset ID/version, DB snapshot timestamp, model name/version, prompts/system
  messages, temperature/top_p, seeds, cost/tokens, and metrics.
- Global randomness seeds MUST be set and recorded; any nondeterminism settings
  MUST be logged.
- Produced artifacts (refactored sources, diffs, smell labels, reports) MUST be
  saved as W&B artifacts with immutable versioning.

Rationale: Ensures results are independently reproducible and auditable.

### II. Baseline Parity and Comparisons
- Classical detectors (SonarQube, PMD or equivalents) MUST run on the same
  datasets as LLM approaches.
- Reports MUST include per‑smell metrics (precision/recall/F1), refactoring
  success rates, runtime, and quality proxies using identical definitions across
  methods.
- LLM ablations (e.g., zero‑shot, few‑shot, tool‑augmented) SHOULD be included
  when relevant and logged as separate runs.
- Where sample sizes permit, statistical significance or confidence intervals
  SHOULD be reported.

Implementation requirement: SonarQube MUST be provided as a local Dockerized
service (pinned image tag) for reproducible baselines. The configuration MUST
avoid committing secrets, support a default local port, and allow optional
persistent volumes. SonarQube version and configuration MUST be logged in W&B
alongside baseline results.

Rationale: Enables fair, measurable comparison between LLM and classical tools.

### III. Correctness and Safety of Refactorings
- Behavior preservation MUST be validated: build/lint pass and `pytest` suites
  run for the target projects (or synthetic tests when available).
- Pre/post gates MUST include: successful execution, no new critical smells of
  the targeted categories, idempotency checks where applicable, and bounded diff
  size.
- Execution MUST be sandboxed; experiments MUST avoid network side‑effects and
  comply with licenses of any code corpora.

Rationale: Prevents harmful changes and ensures refactorings are safe to adopt.

### IV. Transparent LLM Configuration and Prompting
- Prompts, system messages, tool definitions, LangGraph graph version, and step
  traces MUST be persisted per run.
- Generation settings (temperature, top_p, max_tokens) and retry/guardrail
  policies MUST be recorded.
- Token usage and cost MUST be captured for all LLM calls.

Rationale: Supports interpretability, debugging, and cost accounting.

### V. Project Structure and Data Governance
- Notebooks for experiments MUST reside in `experiments/` and import pipelines
  from `src/` only.
- Pipelines and utilities MUST live in `src/` as importable modules with
  configuration separated from code.
- Datasets MUST be accessed via read‑only connectors (e.g., MySQL) with schema
  versioning and migration history tracked; dataset metadata (source, license,
  split, checksum) MUST be recorded.
- Any sensitive data MUST be anonymized or excluded; only licensed data MAY be
  used.

Rationale: Keeps the repository organized, safe, and compliant.

## Additional Constraints & Tech Stack

- Language: Python 3.11+
- Core: LangGraph for orchestration; W&B for tracking/evaluation; MySQL for
  dataset storage; integration with SonarQube and PMD for baselines.
- Baseline Runtime: SonarQube via Docker (compose or single-container) with
  pinned version; local access default on port 9000; credentials/tokens provided
  via environment.
- Testing: `pytest`; static checks via linters/formatters.
- Structure: `src/` (pipelines, data access, evaluation), `experiments/`
  (Jupyter notebooks), `tests/` (unit/integration), DB migration scripts as
  applicable.
- Config: Use environment variables and `.env` (not committed) or a config file
  in `src/` with clear precedence; never hard‑code secrets.

## Development Workflow & Quality Gates

1. Every PR introducing a new experiment MUST include:
   - A link to the W&B run(s) and artifact versions
   - The Git SHA used for the run
   - Constitution Check acknowledgment (list any deviations and rationale)
2. Before merge, the following MUST pass:
   - Reproducibility gate (Principle I)
   - Baseline parity checks (Principle II) where applicable, including a Docker
     SonarQube analysis run against the same dataset snapshot and export of
     baseline metrics
   - Correctness/safety gates (Principle III)
   - LLM transparency artifacts present (Principle IV)
3. Notebook hygiene:
   - Notebooks MUST be deterministic, with execution order cleaned and outputs
     cleared unless they serve as minimal evidence; long outputs go to W&B
     artifacts.
4. Releases and milestones:
   - Tag important result sets with Git tags and W&B tags referencing the
     Constitution version in effect.

## Governance

- This Constitution supersedes other guidelines for this repository’s research
  workflow.
- Amendments require a PR describing the change, its rationale, migration
  impact, and updates to dependent templates.
- Versioning policy: Semantic Versioning
  - MAJOR: Backward‑incompatible principle removal or redefinition
  - MINOR: New principle/section or materially expanded guidance
  - PATCH: Clarifications/wording/typos
- Compliance review: Plan/spec/tasks templates include a "Constitution Check"
  gate; reviewers MUST verify compliance before approval. A periodic audit MAY
  be conducted before major milestones.

**Version**: 1.1.0 | **Ratified**: 2025-09-26 | **Last Amended**: 2025-09-26