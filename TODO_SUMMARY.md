# TODO Summary: Technical Specification Updates

Generated from TECHNICAL_SPECIFICATION.md version 1.1 (2026-01-12)

This document lists all TODO items identified during the ambiguity clarification process. Items are organized by priority and component.

---

## Critical TODOs (Impact on Core Functionality)

### 1. Data & Evaluation
**Line 364**: Implement adapter for new dataset format (specification to be provided)
- **Impact**: Blocks use of new dataset
- **Dependency**: Need dataset specification from user
- **Location**: Section 4.3 - Refactoring mapping

**Line 816**: Implement adapter for new format
- **Impact**: Same as above, duplicate reference
- **Location**: Section 5.5 - MLflow GenAI contract

### 2. Metrics & Scoring
**Line 603**: Verify severity mapping table exists in codebase and document exact location
- **Impact**: Need to confirm SEVERITY_MAP implementation
- **Action**: Search codebase for existing mapping
- **Location**: Section 5.1 - Code smell detection contract

### 3. Algorithm Improvements
**Line 406**: Implement cycle detection mechanism and max-step limit to prevent infinite loops
- **Impact**: Critical for preventing infinite loops in prioritization
- **Urgency**: High
- **Location**: Section 4.4 - Dependency-aware prioritization

---

## High Priority TODOs (Research Quality)

### 4. Agent Architecture
**Line 118**: Implement test generation capabilities for methods without test coverage
- **Impact**: Agent 3 is placeholder
- **Urgency**: Medium (not primary research focus)
- **Location**: Section 3.2 - Agent architecture

**Line 136**: Implement behavior preservation checks beyond test execution
- **Impact**: Agent 6 functionality limited to test execution
- **Urgency**: High (validates refactoring safety)
- **Location**: Section 3.2 - Agent 6

### 5. Documentation & Citations
**Line 389**: Create comprehensive map of dependency rules with detailed citations
- **Impact**: Research credibility, reproducibility
- **Action**: Expand DEPENDENCY_RULES with Markovič & Polášek citations
- **Urgency**: High (thesis requirement)
- **Location**: Section 4.4 - Dependency-aware prioritization

**Line 327**: Document when optional SonarQube issues are included vs excluded
- **Impact**: Understanding of context usage
- **Action**: Check code for conditional logic
- **Location**: Section 4.3 - Refactoring mapping

**Line 328**: Document when optional dependency analysis is included vs excluded
- **Impact**: Same as above
- **Action**: Check code for conditional logic
- **Location**: Section 4.3 - Refactoring mapping

**Line 338**: Add reference link to datamodel for prompt structure
- **Impact**: Implementation clarity
- **Action**: Link to relevant code file
- **Location**: Section 4.3 - Refactoring mapping

---

## Medium Priority TODOs (Optimization & Future Work)

### 6. Performance & Scalability
**Line 1009**: Investigate parallel evaluation by breaking datasets into chunks
- **Impact**: Performance improvement potential
- **Urgency**: Medium (optimization, not critical)
- **Location**: Section 6.3 - MLflow integration

**Line 1010**: Implement proper concurrency handling for MLflow server management
- **Impact**: Enables parallel evaluation
- **Dependency**: Requires investigation first (line 1009)
- **Location**: Section 6.3 - MLflow integration

**Line 1201**: Investigate parallel evaluation capabilities and test concurrency handling
- **Impact**: Same as 1009/1010
- **Location**: Section 8.2 - Performance

### 7. Context & Token Management
**Line 348**: Implement token counting and truncation strategy for large files
- **Impact**: Prevents API failures on large files ("God Classes")
- **Urgency**: Medium (nice-to-have, not critical if datasets are clean)
- **Location**: Section 4.3 - Refactoring mapping

**Line 175**: Add simple persistence mechanism for long-running workflows
- **Impact**: Low (current workflow clears context frequently)
- **Priority**: Low
- **Location**: Section 3.4 - Design patterns

---

## Low Priority TODOs (Investigation & Verification)

### 8. Data Format Verification
**Line 814**: Verify how new dataset handles multi-file refactorings
- **Impact**: Understanding of dataset structure
- **Action**: Inspect dataset once specification provided
- **Location**: Section 5.5 - MLflow GenAI contract

**Line 1066**: Verify RefactoringMiner manifest format matches raw output
- **Impact**: Understanding of data pipeline
- **Action**: Compare manifest files to RefactoringMiner 2.0 output
- **Location**: Section 6.5 - RefactoringMiner data integration

### 9. Infrastructure Investigation
**Line 407**: Investigate Airflow capabilities for handling cyclic dependency situations
- **Impact**: Error handling strategy for outliers
- **Dependency**: Requires cycle detection implementation (line 406)
- **Location**: Section 4.4 - Dependency-aware prioritization

---

## TODO Count by Category

| Category | Count | Priority |
|----------|-------|----------|
| Data & Evaluation | 3 | Critical |
| Algorithm | 1 | Critical |
| Agent Architecture | 2 | High |
| Documentation | 4 | High |
| Performance | 3 | Medium |
| Context Management | 2 | Medium |
| Verification | 3 | Low |
| **TOTAL** | **18** | - |

---

## Recommended Implementation Order

1. **Immediate** (before continuing development):
   - Verify severity mapping table (line 603)
   - Implement cycle detection (line 406)
   - Implement behavior preservation checks (line 136)

2. **Short-term** (thesis requirements):
   - Create comprehensive dependency rules map (line 389)
   - Document optional context usage (lines 327, 328)

3. **When new dataset arrives**:
   - Implement dataset adapter (lines 364, 816)
   - Verify multi-file handling (line 814)
   - Verify RefactoringMiner format (line 1066)

4. **Optimization phase** (after core functionality works):
   - Token counting/truncation (line 348)
   - Parallel evaluation investigation (lines 1009, 1010, 1201)

5. **Future work** (low priority):
   - Test generation (line 118)
   - State persistence (line 175)
   - Airflow investigation (line 407)

---

## Notes

- Some TODOs are duplicates or closely related (e.g., 364/816, 1009/1010/1201)
- Many TODOs are blocked by external dependencies (new dataset specification)
- Focus on critical path: cycle detection → behavior preservation → documentation
