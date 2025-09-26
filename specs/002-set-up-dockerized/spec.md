# Feature Specification: Dockerized Local SonarQube Baseline Integration

**Feature Branch**: `002-set-up-dockerized`  
**Created**: 2025-09-26  
**Status**: Draft  
**Input**: User description: 
"Set up Dockerized local SonarQube as baseline: provide compose/single-container options, pinned version, default port 9000, env-based credentials; add CLI/notebook snippets to run analysis on datasets used by LLM pipelines; log SonarQube version/config and export baseline metrics to W&B."

## Execution Flow (main)
```
1. Parse user description from Input
2. Extract key concepts from description
3. Identify ambiguities and mark them
4. Fill User Scenarios & Testing
5. Generate Functional Requirements
6. Identify Key Entities
7. Run Review Checklist
```

---

## Clarifications
### Session 2025-09-26
- Q: Which SonarQube image tag should we pin for Docker? → A: sonarqube:10.6.0-community
- Q: How will datasets be analyzed in SonarQube? → A: Clone full Java projects from dataset context; analyze smelly and refactored versions; fetch issues via API; export subset metrics and full JSON artifact.
- Q: How should we locate and materialize project sources per dataset sample? → A: Git URL + commit SHA (smelly/refactored).

## User Scenarios & Testing (mandatory)

### Primary User Story
As a researcher, I can run a local Dockerized SonarQube and analyze the same
code datasets used by LLM pipelines, then export baseline metrics to W&B for
fair comparison.

### Acceptance Scenarios
1. SonarQube starts locally via Docker (compose or single container) with a
   pinned image tag and default port 9000.
2. Credentials are passed via environment variables; no secrets are committed.
3. A CLI/notebook helper triggers analysis for a dataset snapshot and captures
   SonarQube version, quality profile, and key metrics.
4. Baseline metrics are exported to W&B (linked to the same dataset artifact).
5. The run is reproducible on another machine following README steps.

### Edge Cases
- Port 9000 unavailable: choose next free port or fail with clear remediation.
- Cold start time: health-check with bounded retry and timeout.
- Missing license/commercial plugins: stick to OSS features only.

## Requirements (mandatory)

### Functional Requirements
- **FR-001**: Provide Docker configuration (compose and single-container) for
  SonarQube pinned to `sonarqube:10.6.0-community`.
- **FR-002**: Default local port 9000 and health-check with retry/timeout.
- **FR-003**: Credentials via environment variables and `.env` (untracked).
- **FR-004**: CLI/notebook helper to trigger analysis for a given project path
  or dataset snapshot and wait for completion.
- **FR-005**: For each dataset pair (smelly code, refactoring), clone the full
  Java project(s) referenced by the sample, run SonarQube analysis on the
  smelly and refactored versions, and capture results.
- **FR-005a**: Each dataset pair MUST specify `repo_url` and `commit_sha` for
  both smelly and refactored versions; analysis MUST checkout those exact SHAs.
- **FR-006**: Export baseline metrics (issues by type/severity, hotspots, scan
  duration, quality gate status, triggered rules) to W&B and link to the same
  dataset artifact; also upload the full raw JSON/issues export as a W&B
  artifact.
- **FR-006**: Log SonarQube version/config (quality profile, project key) in W&B.
- **FR-007**: Document run steps in README and example notebook cells.
- **FR-008**: Keep scope to OSS SonarQube (no paid plugins).

### Key Entities
- **SonarRuntime**: container image, tag, port, health status.
- **SonarProject**: key, profile, ruleset, analysis params.
- **BaselineMetrics**: issues by type/severity, hotspots, timings, version,
  quality gate status, triggered rules; plus reference to raw JSON artifact.
- **DatasetPair**: smelly and refactored code references and their associated
  project source location(s) and commit SHAs (smelly/refactored), including
  `repo_url`, `commit_sha_smelly`, `commit_sha_refactored`.

---

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details beyond necessary runtime constraints
- [ ] User-focused outcomes clear
- [ ] Mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements testable and measurable
- [ ] Scope and dependencies clear

---

## Ambiguities
- [RESOLVED] SonarQube image tag pinned to `sonarqube:10.6.0-community` (OSS).
- [RESOLVED] Project sources located via Git URL + commit SHA for smelly and
  refactored versions.
