# Task Plan: Add Traceable TODOs to Source Code

## Goal
1. Extract all TODOs and gaps from TECHNICAL_SPECIFICATION.md
2. Verify which TODOs are already marked in source code
3. Add missing TODOs to relevant source files with integer identifiers (SPEC-001, SPEC-002, etc.)
4. Create mapping document linking TODO IDs back to TECHNICAL_SPECIFICATION.md sections

## Context
- TECHNICAL_SPECIFICATION.md is version 1.1 (recently updated on 2026-01-12)
- Previous task completed: All 34 ambiguities in spec were addressed
- Current branch: revised
- Working directory: /Users/havriil.pietukhin/PycharmProjects/smellai3/smellai

## Success Criteria
- [ ] Complete mapping between actual code structure and specification claims
- [ ] All misalignments or gaps identified
- [ ] Comprehensive quality evaluation (navigability, usability, unambiguity)
- [ ] README updated with source-of-truth statement
- [ ] Actionable recommendations for improvements

## Phases

### Phase 1: Extract TODOs from TECHNICAL_SPECIFICATION.md [complete]
**Status**: complete ✅
**Goal**: Create comprehensive list of all TODOs mentioned in the specification

**Actions**:
- [ ] Systematically read TECHNICAL_SPECIFICATION.md
- [ ] Extract all TODO statements with section references
- [ ] Extract all "future work" mentions
- [ ] Extract all placeholder implementations (e.g., Agent 3)
- [ ] Assign unique integer IDs (SPEC-001, SPEC-002, etc.)
- [ ] Document severity/priority for each TODO

**Deliverables**: Complete TODO list in findings.md with IDs and section references

---

### Phase 2: Audit Existing TODOs in Source Code [complete]
**Status**: complete ✅
**Goal**: Find all existing TODO comments in source code

**Actions**:
- [ ] Search for TODO comments in all Python files
- [ ] Catalog location, description, and context
- [ ] Check if existing TODOs reference specification
- [ ] Identify which spec TODOs are already marked in code

**Deliverables**: Inventory of existing code TODOs in findings.md

---

### Phase 3: Create TODO Mapping [complete]
**Status**: complete ✅
**Goal**: Match specification TODOs to source code locations

**Actions**:
- [ ] For each spec TODO, identify relevant source file(s)
- [ ] Determine exact location for TODO comment (function, class, module level)
- [ ] Create mapping: SPEC-ID → file:line → spec section
- [ ] Prioritize TODOs that are actionable vs informational

**Deliverables**: TODO mapping document

---

### Phase 4: Add TODO Comments to Source Code [complete]
**Status**: complete ✅
**Goal**: Insert traceable TODO comments into source files

**Actions**:
- [ ] Add TODO comments with format: `# TODO SPEC-XXX: <description> (See TECHNICAL_SPECIFICATION.md §Y.Z)`
- [ ] Ensure comments are placed at relevant code locations
- [ ] Avoid adding TODOs for already-completed features
- [ ] Follow project conventions from CLAUDE.md

**Deliverables**: Updated source files with traceable TODOs

---

### Phase 5: Create TODO Index Document [complete]
**Status**: complete ✅
**Goal**: Create master document linking all TODOs

**Actions**:
- [ ] Create TODO_INDEX.md with table format
- [ ] Include: ID, Description, Spec Section, File Location, Priority, Status
- [ ] Sort by priority and module
- [ ] Add "How to Use" section explaining the numbering system

**Deliverables**: TODO_INDEX.md

---

### Phase 6: Verification & Summary [complete]
**Status**: complete ✅
**Goal**: Verify all TODOs are properly linked

**Actions**:
- [ ] Verify all spec TODOs have corresponding code TODOs
- [ ] Verify all code TODOs reference correct spec sections
- [ ] Check for any duplicate or conflicting TODOs
- [ ] Write summary report

**Deliverables**: Summary of work completed

---

## Decisions Log

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Reuse planning files from previous session | Previous task complete, can repurpose | Faster startup |
| Use planning-with-files pattern | Complex multi-phase analysis task | Better organization |
| Use Explore agent for Phase 1 | Systematic codebase discovery | Comprehensive mapping |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| - | - | - |

## Notes
- TECHNICAL_SPECIFICATION.md: Version 1.1, 1425 lines, updated 2026-01-12
- README.md: 249 lines, no mention of technical spec
- Previous session: Successfully addressed 34 ambiguities in spec
