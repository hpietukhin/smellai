# Findings and decisions

## Requirements
- Analyze TECHNICAL_SPECIFICATION.md against codebase
- Use ast-grep to find inconsistencies
- Add pytest tests for data contracts
- Identify and remove ineffective/verbose code

## Research Findings

### From TECHNICAL_SPECIFICATION.md (Section 5 - Data Contracts)
Key data models specified:
1. **SmellDetection** - Code smell detection output (section 5.1)
2. **TestResult** / **TestRunSummary** - Java test analysis (section 5.2)
3. **RMinerEvalState** - Refactoring mapping agent state (section 5.3)
4. **RefactoringMapping** / **RefactoringMappingOutput** - LLM output (section 5.3)
5. **DependencyAnalysis** - Dependency analysis output (section 5.4)
6. **Refactoring** / **RefactoringLocation** / **RMinerCommit** - RefactoringMiner models (section 5.6)
7. **RefactoringStats** - Statistical summary (section 5.6)

### Code Locations from Spec
- `models/` - Pydantic data models (RefactoringMiner structures)
- `agents/rminer_eval/` - Refactoring mapping agent
- `agents/java_test/` - Test analysis agent
- `agents/dependency_analysis/` - Dependency analysis logic

### Codebase Exploration Results (31 Pydantic + 7 dataclasses)

**CRITICAL INCONSISTENCIES FOUND:**

1. **DiffHunk duplication** - Defined in 3 places:
   - `datasets/models.py` (Pydantic)
   - `rminer/create_rminer_dataset.py` (dataclass)
   - `rminer/rminer_utils.py` (dataclass)

2. **RefactoringMapping mismatch** - Spec says `hunk_start_line`/`hunk_end_line` but code uses `hunk_index`:
   - Spec (5.3): `hunk_start_line: int`, `hunk_end_line: int`
   - Code: `hunk_index: int`

3. **SmellDetection model MISSING** - Spec defines it in 5.1 but not found in code

4. **TestResult/TestRunSummary** - Spec shows Pydantic but code uses dataclasses

**Models found by location:**
- `datasets/models.py`: 18 Pydantic models (new dataset framework)
- `models/refactoring.py`: 5 Pydantic models (RefactoringMiner)
- `agents/dependency_analysis/agent.py`: 1 Pydantic model
- `agents/rminer_eval/agent.py`: 2 Pydantic models
- `sonarqube/tool.py`: 1 Pydantic model
- Various dataclasses in agent tools and utilities

### Additional findings from ast-grep analysis

**Duplication issues:**
1. **DiffHunk** defined 3 times with nearly identical fields:
   - `datasets/models.py:73` (Pydantic)
   - `rminer/create_rminer_dataset.py:42` (dataclass)
   - `rminer/rminer_utils.py:73` (dataclass)

2. **parse_refactoring_info** function duplicated:
   - `rminer/create_rminer_dataset.py:108`
   - `rminer/rminer_utils.py:619`

**Code smell violations (CLAUDE.md policy violations):**
1. `except Exception:` with no logging/re-raise (violates "no silent except"):
   - `agents/rminer_eval/agent.py:89` - silently falls back to non-structured model
   - `agents/rminer_eval/agent.py:180` - silently sets mappings_data = []
   - `agents/tools/java_test_tools.py:214` - silently continues on malformed XML
   - `agents/tools/java_test_tools.py:281` - silently continues on malformed XML

**Spec vs code mismatches:**
1. `RefactoringMapping.hunk_index` (code) vs `hunk_start_line`/`hunk_end_line` (spec 5.3)
2. `SmellDetection` model defined in spec 5.1 but missing from code
3. `TestResult`/`TestRunSummary` spec shows Pydantic but code uses dataclasses

## Changes made

### Fixed CLAUDE.md policy violations
1. **agents/tools/java_test_tools.py**:
   - Added logging import
   - Changed `except Exception:` to `except ET.ParseError as e:` with logging (lines 217, 284)

2. **agents/rminer_eval/agent.py**:
   - Added logging import
   - Changed `except Exception:` to `except (NotImplementedError, AttributeError) as e:` with logging (line 92)
   - Changed `except Exception:` to `except json.JSONDecodeError as e:` with logging (line 184)

### Created new tests
- **tests/test_data_contracts.py**: 40 comprehensive tests for all Pydantic models
  - Tests required fields and validation
  - Tests defaults and optional fields
  - Tests serialization/deserialization round-trips
  - Tests JSON compatibility

## Recommendations for future work

### Code duplication to consolidate
1. **DiffHunk** (3 definitions) - Consolidate to `datasets/models.py` as canonical:
   - Remove from `rminer/create_rminer_dataset.py`
   - Remove from `rminer/rminer_utils.py`
   - Update imports

2. **parse_diff_hunks** (2 definitions with different signatures):
   - `rminer/rminer_utils.py:96` - `parse_diff_hunks(diff_text: str)` - parses text
   - `rminer/create_rminer_dataset.py:57` - `parse_diff_hunks(before_file, after_file)` - runs git diff
   - Rename one to clarify: e.g., `parse_diff_text()` vs `compute_diff_hunks()`

3. **parse_refactoring_info** duplicated in both rminer modules

### Spec vs code mismatches to address
1. Update spec OR code for `RefactoringMapping.hunk_index` vs `hunk_start_line`/`hunk_end_line`
2. Add missing `SmellDetection` Pydantic model from spec 5.1
3. Consider converting `TestResult`/`TestRunSummary` from dataclasses to Pydantic

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use planning-with-files | Complex multi-step task requiring organized approach |
| Fix silent except blocks | CLAUDE.md requires logging and specific exception types |
| Document duplications only | CLAUDE.md requires coordination before removing others' work |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| DiffHunk defined 3 times | Documented for future consolidation |
| Silent except Exception: | Fixed with specific exceptions and logging |

## Resources
- TECHNICAL_SPECIFICATION.md - Main specification document
- `models/` - Pydantic models location
- `agents/` - Agent implementations

## Visual/Browser Findings
- N/A (no browser operations yet)

---
*Update this file after every 2 view/browser/search operations*
