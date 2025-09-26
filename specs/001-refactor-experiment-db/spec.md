# Feature Specification: Refactor Experiment DB Pipeline into Importable, Extensible Module

**Feature Branch**: `001-refactor-experiment-db`  
**Created**: 2025-09-26  
**Status**: Draft  
**Input**: User description: 
"Refactor experiment DB pipeline to be imported through extensible Python module from src; move DB access out of may25/experiment1.ipynb; provide pluggable DB connectors (start with MySQL), clean configuration, and importable LangGraph pipeline used by notebooks in experiments/. Ensure W&B tracking and reproducibility gates per Constitution v1.0.0."

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors, actions, data, constraints
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies  
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## Clarifications
### Session 2025-09-26
- Q: Which output format should the pipeline return to notebooks? → A: Pandas DataFrame(s) for classes/refactorings.
- Q: What minimal columns must each DataFrame include? → A: Connector-specific; MySQL follows current notebook-defined schema.
- Q: How should we name and organize W&B dataset artifacts? → A: `datasets/{dataset_name}:{version}`; connector in metadata only.
- Q: Which W&B project should we use? → A: mt

## User Scenarios & Testing (mandatory)

### Primary User Story
As a research author, I can import an experiment pipeline from `src/` into a
notebook in `experiments/`, select a database connector (starting with MySQL)
via configuration, and run the pipeline to fetch prepared datasets (classes and
their refactorings) so that I can execute LLM-based detection/refactoring
experiments with W&B tracking and reproducible metadata.

### Acceptance Scenarios
1. **Given** a notebook in `experiments/` and valid env/config for MySQL,
   **When** I `import` the pipeline from `src/` and invoke the data fetch step,
   **Then** I receive dataset objects/iterables for classes and refactorings and
   a W&B run is created logging Git SHA, dataset ID/version, connector name and
   connection timestamp.
2. **Given** misconfigured/missing credentials, **When** I run the notebook,
   **Then** a clear configuration error is raised without partial writes, and
   the failure is recorded with a reproducible error artifact (without leaking
   secrets).
3. **Given** a second connector implementation (e.g., PostgreSQL) added under
   the same interface, **When** I switch the connector via configuration only,
   **Then** the notebook runs without code changes and logs the new connector
   name in W&B metadata.
4. **Given** Constitution v1.0.0 gates, **When** I run the pipeline,
   **Then** required reproducibility fields (seeds, model name, prompts if used)
   are logged and accessible from the W&B run.

### Edge Cases
- No network/DB unreachable: the run fails fast with a single, user-readable
  error and no partial artifacts; retry policy is documented.
- Empty datasets: pipeline returns empty iterables with explicit reason in
  metadata; downstream steps handle gracefully.
- Very large datasets: connector supports batched/streamed iteration; notebook
  example demonstrates pagination parameters.
- Secret management: no secrets appear in notebook cells or logs; redaction in
  error messages is enforced.

## Requirements (mandatory)

### Functional Requirements
- **FR-001**: Provide an importable experiment pipeline in `src/` that notebooks
  in `experiments/` can invoke to fetch datasets and run steps.
- **FR-002**: Remove direct DB access logic from `may25/experiment1.ipynb` and
  future notebooks; notebooks MUST import functionality from `src/` only.
- **FR-003**: Support a pluggable connector interface for relational stores;
  initial implementation MUST include MySQL.
- **FR-004**: Connector selection MUST be controlled by configuration (env vars
  and/or config file), without notebook code changes.
- **FR-005**: Configuration MUST keep secrets out of version control and
  notebook outputs; failures MUST not leak secrets.
- **FR-006**: Each run MUST be tracked in W&B with Git SHA, dataset ID/version,
  connector name/version, seeds, model settings (if used), prompts/system
  messages (if used), token/cost metrics.
- **FR-007**: Reproducibility requirements from the Constitution v1.0.0 MUST be
  satisfied and verifiable from the W&B run.
- **FR-008**: Dataset access MUST capture schema version and source metadata;
  returned objects MUST include dataset identifiers for audit.
- **FR-009**: Provide clear error messages for configuration/connection issues;
  errors MUST be surfaced to the notebook with actionable remediation hints.
- **FR-010**: Expose batched/streamed reading for large datasets with stable
  ordering guarantees.
- **FR-011**: The refactor MUST not change current experiment results when run
  with equivalent inputs (behavior-preserving).
- **FR-012**: Provide a minimal example notebook cell showing how to import and
  run the pipeline with configuration (documentation-only, not implementation).
- **FR-013**: Logging MUST avoid PII/secrets; connector logs MUST be opt-in at
  a configurable verbosity level.
- **FR-014**: Extensibility: adding a new connector MUST require implementing a
  single documented interface and registering it via configuration only.
- **FR-015**: Compatibility: pipeline MUST remain usable for evaluation against
  classical tools (SonarQube/PMD) using the same datasets.
- **FR-016**: Pipeline outputs MUST be Pandas DataFrame(s) for classes and
  refactorings returned to notebooks, with stable column names documented in
  connector docs.
- **FR-017**: Each connector MUST expose a discoverable schema (e.g., `schema()`
  function) and publish a `schema.json` with the dataset W&B Artifact; the
  MySQL connector MUST mirror the column set currently used in the notebook.
- **FR-018**: W&B dataset artifacts MUST follow the naming pattern
  `datasets/{dataset_name}:{version}` and MUST include the connector identity in
  artifact metadata (not the name).
- **FR-019**: All runs and dataset artifacts MUST be logged under the W&B
  project `mt` unless explicitly overridden by configuration.

### Key Entities (include if feature involves data)
- **Dataset**: Logical collection of classes and corresponding refactorings;
  attributes include dataset_id, schema_version, split, checksum, source.
- **Connector**: Abstraction for data access to relational stores; attributes
  include name, version, config_keys, capabilities (streaming, pagination).
- **ExperimentRunMetadata**: Reproducibility and tracking info; attributes
  include git_sha, seeds, llm_model/version (if used), prompts, token/costs,
  start/end timestamps, artifact references.
- **ExperimentConfig**: User-supplied configuration; attributes include
  connector type, connection parameters (secret-managed), batch size, limits.
- **SchemaDescriptor**: Connector-specific DataFrame schema description
  (column names, types, primary identifiers); persisted alongside dataset
  artifacts and retrievable via connector `schema()`.

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous  
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

---

## Ambiguities
- [RESOLVED] Dataset identity/versioning/snapshotting: use Weights & Biases
  Artifacts (unique artifact names, automatic versioning v0..vn, immutable
  snapshotting with hashing). Pipelines MUST log and consume datasets via W&B
  Artifacts.
- [RESOLVED] LLM provider and defaults: provider set via LiteLLM. Use the
  current default generation parameters from `may25/experiment1.ipynb` for
  temperature and related settings unless explicitly overridden by config.
- [NEEDS CLARIFICATION] Exact DB schema(s) and table/column names for classes
  and refactorings (to be documented in connector config/docs).
- [DEFERRED] Minimal error redaction policy for secrets in logs (out of scope for
  this feature; to be addressed in a future governance update).
