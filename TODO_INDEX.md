# TODO Index

**Version**: 1.0
**Date**: 2026-01-12
**Source**: TECHNICAL_SPECIFICATION.md v1.1

This document provides a complete index of all TODO items referenced in the codebase, traceable back to the technical specification.

---

## How to Use This Index

Each TODO in the source code follows this format:
```python
# TODO SPEC-XXX: <brief description>
# <additional context>
# <priority> priority.
# (See TECHNICAL_SPECIFICATION.md §Y.Z)
```

Use this index to:
1. Find all TODOs by priority, category, or module
2. Trace TODOs back to specification sections
3. Track implementation progress
4. Plan development sprints

---

## Summary Statistics

- **Total TODOs**: 18
- **HIGH Priority**: 4 (22%)
- **MEDIUM Priority**: 7 (39%)
- **LOW Priority**: 7 (39%)
- **Files Modified**: 9

---

## Complete TODO List

| ID | Priority | Category | File | Line Area | Description |
|----|----------|----------|------|-----------|-------------|
| SPEC-001 | MEDIUM | Agent Feature | agents/java_test/agent.py | Module docstring | Implement test generation capabilities for methods without test coverage |
| SPEC-002 | MEDIUM | Agent Feature | agents/java_test/agent.py | Module docstring | Implement behavior preservation checks beyond test execution |
| SPEC-003 | LOW | Architecture | agents/rminer_eval/agent.py | RMinerEvalState class | Add simple persistence mechanism for long-running workflows |
| SPEC-004 | LOW | Documentation | agents/rminer_eval/agent.py | map_refactorings() | Document when sonar_issues context is included vs excluded |
| SPEC-005 | LOW | Documentation | agents/rminer_eval/agent.py | map_refactorings() | Document when dependency_analysis context is included vs excluded |
| SPEC-006 | LOW | Documentation | agents/rminer_eval/agent.py | map_refactorings() | Add reference link to exact prompt structure datamodels |
| SPEC-007 | HIGH | Core Feature | agents/rminer_eval/agent.py | map_refactorings() | Implement token counting and truncation strategy for large files |
| SPEC-008 | HIGH | Data Integration | mlflow_utils/datasets/rminer_factory.py | RMinerDatasetFactory class | Implement adapter for new dataset format |
| SPEC-009 | MEDIUM | Documentation | agents/dependency_analysis/agent.py | DEPENDENCY_RULES | Create comprehensive map with detailed citations |
| SPEC-010 | HIGH | Core Feature | scripts/prioritize_smells.py | calculate_priorities() | Implement cycle detection mechanism and max-step limit |
| SPEC-011 | LOW | Research | scripts/prioritize_smells.py | calculate_priorities() | Investigate Airflow capabilities for cyclic dependencies |
| SPEC-012 | LOW | Verification | sonarqube/commit_scan.py | SEVERITY_MAP | Verify mapping table location and document |
| SPEC-013 | MEDIUM | Data Integration | mlflow_utils/datasets/rminer_factory.py | RMinerDatasetFactory class | Verify handling of multi-file refactorings |
| SPEC-014 | HIGH | Data Integration | mlflow_utils/datasets/rminer_factory.py | RMinerDatasetFactory class | Implement adapter for new dataset format (duplicate of SPEC-008) |
| SPEC-015 | MEDIUM | Performance | mlflow_utils/runner.py | EvaluationRunner class | Investigate parallel evaluation by breaking datasets into chunks |
| SPEC-016 | MEDIUM | Infrastructure | mlflow_utils/server.py | MLflowServer class | Implement proper concurrency handling for server management |
| SPEC-017 | LOW | Verification | rminer/rminer_utils.py | Module docstring | Verify manifest format matches RefactoringMiner 2.0 output |
| SPEC-018 | MEDIUM | Performance | mlflow_utils/runner.py | EvaluationRunner class | Investigate parallel evaluation capabilities (duplicate of SPEC-015) |

---

## TODOs by Priority

### HIGH Priority (4 TODOs)

These require immediate attention as they affect core functionality.

| ID | File | Description | Spec Section |
|----|------|-------------|--------------|
| SPEC-007 | agents/rminer_eval/agent.py:96 | Token counting/truncation for large files | §4.3 |
| SPEC-008 | mlflow_utils/datasets/rminer_factory.py:9 | Dataset format adapter | §4.3 |
| SPEC-010 | scripts/prioritize_smells.py:127 | Cycle detection in prioritization | §4.4 |
| SPEC-014 | mlflow_utils/datasets/rminer_factory.py:15 | Dataset format adapter (duplicate) | §5.5 |

### MEDIUM Priority (7 TODOs)

Important for completeness but not blocking core functionality.

| ID | File | Description | Spec Section |
|----|------|-------------|--------------|
| SPEC-001 | agents/java_test/agent.py:7 | Test generation agent | §3.2 |
| SPEC-002 | agents/java_test/agent.py:13 | Behavior preservation checks | §3.2 |
| SPEC-009 | agents/dependency_analysis/agent.py:36 | Dependency rules citations | §4.4 |
| SPEC-013 | mlflow_utils/datasets/rminer_factory.py:19 | Multi-file refactoring handling | §5.5 |
| SPEC-015 | mlflow_utils/runner.py:10 | Parallel evaluation | §6.3 |
| SPEC-016 | mlflow_utils/server.py:12 | MLflow concurrency | §6.3 |
| SPEC-018 | mlflow_utils/runner.py:16 | Parallel evaluation (duplicate) | §8.2 |

### LOW Priority (7 TODOs)

Nice to have; documentation and verification tasks.

| ID | File | Description | Spec Section |
|----|------|-------------|--------------|
| SPEC-003 | agents/rminer_eval/agent.py:43 | State persistence | §3.4 |
| SPEC-004 | agents/rminer_eval/agent.py:114 | Document sonar_issues inclusion | §4.3 |
| SPEC-005 | agents/rminer_eval/agent.py:118 | Document dependency_analysis inclusion | §4.3 |
| SPEC-006 | agents/rminer_eval/agent.py:122 | Datamodel reference link | §4.3 |
| SPEC-011 | scripts/prioritize_smells.py:133 | Airflow investigation | §4.4 |
| SPEC-012 | sonarqube/commit_scan.py:43 | Severity map verification | §5.1 |
| SPEC-017 | rminer/rminer_utils.py:15 | Manifest format verification | §6.5 |

---

## TODOs by Category

### Core Feature (2 TODOs)
- SPEC-007: Token counting/truncation (HIGH)
- SPEC-010: Cycle detection (HIGH)

### Data Integration (3 TODOs)
- SPEC-008: Dataset adapter (HIGH)
- SPEC-013: Multi-file refactorings (MEDIUM)
- SPEC-014: Dataset adapter duplicate (HIGH)

### Agent Feature (2 TODOs)
- SPEC-001: Test generation (MEDIUM)
- SPEC-002: Behavior preservation (MEDIUM)

### Documentation (4 TODOs)
- SPEC-004: sonar_issues context (LOW)
- SPEC-005: dependency_analysis context (LOW)
- SPEC-006: Datamodel reference (LOW)
- SPEC-009: Dependency rules citations (MEDIUM)

### Performance (2 TODOs)
- SPEC-015: Parallel evaluation (MEDIUM)
- SPEC-018: Parallel evaluation duplicate (MEDIUM)

### Infrastructure (1 TODO)
- SPEC-016: MLflow concurrency (MEDIUM)

### Architecture (1 TODO)
- SPEC-003: State persistence (LOW)

### Research (1 TODO)
- SPEC-011: Airflow investigation (LOW)

### Verification (2 TODOs)
- SPEC-012: Severity map (LOW)
- SPEC-017: Manifest format (LOW)

---

## TODOs by Module

### agents/rminer_eval/agent.py (5 TODOs)
1. SPEC-003 (LOW): State persistence
2. SPEC-004 (LOW): Document sonar_issues context
3. SPEC-005 (LOW): Document dependency_analysis context
4. SPEC-006 (LOW): Datamodel reference link
5. SPEC-007 (HIGH): Token counting/truncation

### mlflow_utils/ (4 TODOs)
1. SPEC-008 (HIGH): Dataset adapter - rminer_factory.py
2. SPEC-013 (MEDIUM): Multi-file refactorings - rminer_factory.py
3. SPEC-014 (HIGH): Dataset adapter duplicate - rminer_factory.py
4. SPEC-015 (MEDIUM): Parallel evaluation - runner.py
5. SPEC-016 (MEDIUM): MLflow concurrency - server.py
6. SPEC-018 (MEDIUM): Parallel evaluation duplicate - runner.py

### agents/java_test/agent.py (2 TODOs)
1. SPEC-001 (MEDIUM): Test generation
2. SPEC-002 (MEDIUM): Behavior preservation

### scripts/prioritize_smells.py (2 TODOs)
1. SPEC-010 (HIGH): Cycle detection
2. SPEC-011 (LOW): Airflow investigation

### agents/dependency_analysis/agent.py (1 TODO)
1. SPEC-009 (MEDIUM): Dependency rules citations

### sonarqube/commit_scan.py (1 TODO)
1. SPEC-012 (LOW): Severity map verification

### rminer/rminer_utils.py (1 TODO)
1. SPEC-017 (LOW): Manifest format verification

---

## Duplicate TODOs

Some TODOs appear multiple times in the specification, referring to the same implementation task:

| Primary ID | Duplicate ID | Task | Status |
|------------|--------------|------|--------|
| SPEC-008 | SPEC-014 | Dataset format adapter | Not implemented |
| SPEC-015 | SPEC-018 | Parallel evaluation | Not implemented |

**Strategy**: Both IDs reference the same location in code, noting they are duplicates.

---

## Implementation Progress

| Status | Count | Percentage |
|--------|-------|------------|
| Not Started | 18 | 100% |
| In Progress | 0 | 0% |
| Completed | 0 | 0% |

**Last Updated**: 2026-01-12

---

## Recommended Implementation Order

### Sprint 1: Core Functionality (HIGH Priority)
1. **SPEC-007**: Token counting/truncation - Critical for handling large files
2. **SPEC-010**: Cycle detection - Prevents infinite loops in prioritization
3. **SPEC-008/SPEC-014**: Dataset adapter - Enables new dataset formats

### Sprint 2: Extended Features (MEDIUM Priority)
4. **SPEC-015/SPEC-018**: Parallel evaluation - Performance improvement
5. **SPEC-016**: MLflow concurrency - Supports parallel evaluation
6. **SPEC-009**: Dependency rules citations - Improves research credibility
7. **SPEC-013**: Multi-file refactorings - Extends dataset support

### Sprint 3: Agent Extensions (MEDIUM Priority)
8. **SPEC-001**: Test generation - New agent capability
9. **SPEC-002**: Behavior preservation - Enhanced verification

### Sprint 4: Documentation & Verification (LOW Priority)
10. **SPEC-004**: Document sonar_issues context
11. **SPEC-005**: Document dependency_analysis context
12. **SPEC-006**: Datamodel reference links
13. **SPEC-012**: Verify severity mapping
14. **SPEC-017**: Verify manifest format

### Future Research (LOW Priority)
15. **SPEC-003**: State persistence - Only if long-running workflows become necessary
16. **SPEC-011**: Airflow investigation - Exploratory research task

---

## Notes

- All TODOs are now traceable from source code to specification
- Use `grep "TODO SPEC-" **/*.py` to find all TODOs
- Each TODO references the exact specification section for context
- Duplicates are marked to avoid redundant work
- Priority levels match those defined in TECHNICAL_SPECIFICATION.md §TODO

---

## Maintenance

This index should be updated when:
- TODOs are completed (update progress section)
- New TODOs are added to the specification
- TODO priorities change
- Implementation order is revised

**Maintainer**: Keep this file in sync with TODO_SUMMARY.md and TECHNICAL_SPECIFICATION.md
