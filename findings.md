# Findings: TODO Extraction and Traceability Mapping

**Task**: Extract TODOs from TECHNICAL_SPECIFICATION.md and add traceable TODO comments to source code
**Date**: 2026-01-12

---

## Phase 1: TODOs Extracted from TECHNICAL_SPECIFICATION.md

### Summary
- **Total TODOs found**: 18
- **Source**: TECHNICAL_SPECIFICATION.md v1.1
- **Extraction method**: Grep search + manual review

### Complete TODO List with Assigned IDs

| ID | Line | Section | Description | Priority | Category |
|----|------|---------|-------------|----------|----------|
| SPEC-001 | 118 | §3.2 | Implement test generation capabilities for methods without test coverage | MEDIUM | Agent Feature |
| SPEC-002 | 136 | §3.2 | Implement behavior preservation checks beyond test execution | MEDIUM | Agent Feature |
| SPEC-003 | 175 | §3.4 | Add simple persistence mechanism for long-running workflows | LOW | Architecture |
| SPEC-004 | 327 | §4.3 | Document when sonar_issues context is included vs excluded | LOW | Documentation |
| SPEC-005 | 328 | §4.3 | Document when dependency_analysis context is included vs excluded | LOW | Documentation |
| SPEC-006 | 338 | §4.3 | Add reference link to exact prompt structure datamodels in code | LOW | Documentation |
| SPEC-007 | 348 | §4.3 | Implement token counting and truncation strategy for large files (God Classes) | HIGH | Core Feature |
| SPEC-008 | 364 | §4.3 | Implement adapter for new dataset format (specification to be provided) | HIGH | Data Integration |
| SPEC-009 | 389 | §4.4 | Create comprehensive map of dependency rules with detailed citations | MEDIUM | Documentation |
| SPEC-010 | 406 | §4.4 | Implement cycle detection mechanism and max-step limit to prevent infinite loops | HIGH | Core Feature |
| SPEC-011 | 407 | §4.4 | Investigate Airflow capabilities for handling problematic cyclic dependency situations | LOW | Research |
| SPEC-012 | 603 | §5.1 | Verify severity mapping table exists in codebase and document exact location | LOW | Verification |
| SPEC-013 | 814 | §5.5 | Verify how new dataset handles refactorings spanning multiple files | MEDIUM | Data Integration |
| SPEC-014 | 816 | §5.5 | Implement adapter for new dataset format | HIGH | Data Integration |
| SPEC-015 | 1009 | §6.3 | Investigate parallel evaluation by breaking datasets into chunks | MEDIUM | Performance |
| SPEC-016 | 1010 | §6.3 | Implement proper concurrency handling for MLflow server management | MEDIUM | Infrastructure |
| SPEC-017 | 1066 | §6.5 | Verify manifest format matches raw RefactoringMiner 2.0 output | LOW | Verification |
| SPEC-018 | 1201 | §8.2 | Investigate parallel evaluation capabilities and test concurrency handling | MEDIUM | Performance |

### Priority Breakdown
- **HIGH**: 4 TODOs (SPEC-007, SPEC-008, SPEC-010, SPEC-014)
- **MEDIUM**: 7 TODOs (SPEC-001, SPEC-002, SPEC-009, SPEC-013, SPEC-015, SPEC-016, SPEC-018)
- **LOW**: 7 TODOs (SPEC-003, SPEC-004, SPEC-005, SPEC-006, SPEC-011, SPEC-012, SPEC-017)

### Category Breakdown
- **Core Feature**: 2 (SPEC-007, SPEC-010)
- **Data Integration**: 3 (SPEC-008, SPEC-013, SPEC-014)
- **Agent Feature**: 2 (SPEC-001, SPEC-002)
- **Documentation**: 4 (SPEC-004, SPEC-005, SPEC-006, SPEC-009)
- **Performance**: 2 (SPEC-015, SPEC-018)
- **Infrastructure**: 1 (SPEC-016)
- **Architecture**: 1 (SPEC-003)
- **Research**: 1 (SPEC-011)
- **Verification**: 2 (SPEC-012, SPEC-017)

---

## Phase 2: Existing TODOs in Source Code

### Summary
- **Total existing TODOs in source code**: 0
- **Finding**: No TODO, FIXME, or XXX comments found in any Python files
- **Implication**: All 18 specification TODOs need to be added to appropriate source files

---

## Phase 3: TODO Mapping (Spec → Code Location)

### Mapping Table

| TODO ID | Target File | Location Context | Rationale |
|---------|-------------|------------------|-----------|
| SPEC-001 | agents/java_test/agent.py | Module level or Agent 3 placeholder file | Test generation is mentioned as Agent 3 |
| SPEC-002 | agents/java_test/agent.py | After test execution logic | Verification agent reuses Agent 2 |
| SPEC-003 | agents/rminer_eval/agent.py | Near state definition | LangGraph state persistence note |
| SPEC-004 | agents/rminer_eval/agent.py | Prompt construction logic | Context inclusion decision |
| SPEC-005 | agents/rminer_eval/agent.py | Prompt construction logic | Context inclusion decision |
| SPEC-006 | agents/rminer_eval/agent.py | State definition or prompt construction | Datamodel reference |
| SPEC-007 | agents/rminer_eval/agent.py | Prompt construction or input validation | Token limit handling |
| SPEC-008 | mlflow_utils/datasets/rminer_factory.py | Dataset creation logic | New format adapter |
| SPEC-009 | agents/dependency_analysis/agent.py | DEPENDENCY_RULES definition | Citation documentation |
| SPEC-010 | scripts/prioritize_smells.py | Priority calculation loop | Cycle detection |
| SPEC-011 | scripts/prioritize_smells.py | Near cycle detection logic | Airflow investigation |
| SPEC-012 | sonarqube/commit_scan.py | Severity mapping section | Verify mapping table |
| SPEC-013 | mlflow_utils/datasets/rminer_factory.py | pair_id generation logic | Multi-file handling |
| SPEC-014 | mlflow_utils/datasets/rminer_factory.py | Dataset creation logic | Duplicate of SPEC-008 |
| SPEC-015 | mlflow_utils/runner.py | Evaluation execution | Parallel processing |
| SPEC-016 | mlflow_utils/server.py | Server management | Concurrency handling |
| SPEC-017 | rminer/rminer_utils.py | Manifest parsing | Format verification |
| SPEC-018 | mlflow_utils/runner.py | Evaluation execution | Duplicate of SPEC-015 |

### Notes
- **Duplicates identified**: SPEC-008/SPEC-014 (dataset adapter), SPEC-015/SPEC-018 (parallel evaluation)
- **Strategy**: Will add both references but note they refer to the same implementation task

---

## Phase 4: Implementation Status

### Summary
**Status**: ✅ COMPLETE
**Date**: 2026-01-12
**TODOs Added**: 18 / 18 (100%)

### Files Modified

| File | TODOs Added | Line Numbers |
|------|-------------|--------------|
| agents/rminer_eval/agent.py | 5 | 43, 96, 114, 118, 122 |
| agents/java_test/agent.py | 2 | 7, 13 |
| agents/dependency_analysis/agent.py | 1 | 36 |
| mlflow_utils/datasets/rminer_factory.py | 3 | 9, 15, 19 |
| scripts/prioritize_smells.py | 2 | 127, 133 |
| sonarqube/commit_scan.py | 1 | 43 |
| mlflow_utils/runner.py | 2 | 10, 16 |
| mlflow_utils/server.py | 1 | 12 |
| rminer/rminer_utils.py | 1 | 15 |
| **Total** | **18** | **9 files** |

### TODO Format

All TODOs follow this consistent format:
```python
# TODO SPEC-XXX: <brief description>
# <additional context>
# <priority> priority.
# (See TECHNICAL_SPECIFICATION.md §Y.Z)
```

### Verification

All 18 TODOs from TECHNICAL_SPECIFICATION.md have been:
- ✅ Assigned unique integer IDs (SPEC-001 through SPEC-018)
- ✅ Placed in appropriate source files
- ✅ Positioned at relevant code locations
- ✅ Referenced back to specification sections
- ✅ Categorized by priority (HIGH/MEDIUM/LOW)
- ✅ Documented in TODO_INDEX.md

### Key Deliverables

1. **TODO_INDEX.md** - Master document with:
   - Complete TODO list with IDs
   - Priority breakdown (4 HIGH, 7 MEDIUM, 7 LOW)
   - Category breakdown (9 categories)
   - Module breakdown (9 files)
   - Duplicate identification (SPEC-008/014, SPEC-015/018)
   - Recommended implementation order (4 sprints)
   - Progress tracking section

2. **Updated Source Files** - 9 files with traceable TODOs
3. **This Findings Document** - Complete analysis and mapping

### Statistics

- **Coverage**: 100% of specification TODOs added to source code
- **Traceability**: All TODOs link back to spec sections
- **Consistency**: All TODOs follow project conventions
- **Files touched**: 9 Python files across 5 modules
- **Average TODOs per file**: 2.0
- **Most TODOs in single file**: 5 (agents/rminer_eval/agent.py)
