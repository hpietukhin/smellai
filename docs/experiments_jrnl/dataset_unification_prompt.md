# Prompt: Dataset Unification Analysis for Smell-Aware Refactoring Pipeline

## Your Task

I am writing a master's thesis on **LLM-based code smell detection and refactoring using multi-agent orchestration**. I use three research datasets (described below) and a unified evaluation pipeline (described below). I need you to:

1. **Find information** about each of the three datasets (papers, documentation, GitHub repos, structure, intended use cases). Cite sources.
2. **Analyze** whether all three datasets can be used within **one unified experimental approach** as described in my pipeline below.
3. **Identify gaps**: what information is missing from each dataset that the pipeline requires, and how to bridge those gaps.
4. **Propose a concrete mapping** of each dataset to the pipeline stages.

---

## My Three Datasets

### Dataset 1: SWE-Refactor

A benchmark of 1,099 pure (behavior-preserving) Java method-level refactorings across 18 open-source projects.

**Source format**: ZIP archive containing `pure_refactoring_data.json` + experimental results from GPT-4o baseline.

**Key fields per record**:
- `uniqueId`, `commitId`, `projectName` — identifiers
- `type` — refactoring type: Extract Method (441), Move Method (410), Extract And Move Method (142), Inline Method (71), Move And Rename Method (21), Move And Inline Method (14)
- `sourceCodeBeforeRefactoring` / `sourceCodeAfterRefactoring` — method body before/after
- `sourceCodeBeforeForWhole` / `sourceCodeAfterForWhole` — full file before/after
- `filePathBefore` / `filePathAfter` — file paths (may differ for Move)
- `diffSourceCode` — unified diff with line numbers and +/- markers (present in all 1099 records)
- `diffLocations` — list of `{filePath, startLine, endLine}` (present in all 1099 records)
- `compileJDK` (8/11/17/21), `compileCommand` (mvn/gradle), `compileResultBefore`/`compileResultCurrent` — build info
- `hasTestC` — test coverage flag (almost always null, only 1 record = True)
- `coverageInfo` — JaCoCo coverage: {INSTRUCTION, LINE, COMPLEXITY, METHOD} with missed/covered
- `isPureRefactoring` — always True (all 1099)
- `purityCheckResultList` — purity verification details

**What it gives the pipeline**: before/after code + diff at method and file level, build configuration, ground truth refactoring type. Does NOT contain smell information.

### Dataset 2: RMiner 2.0 Oracle (RefactoringMiner Benchmark)

The ground truth oracle used to evaluate RefactoringMiner tool accuracy. Two variants:
- **Java Benchmark 1** (`data.json`): List of commits with refactorings, each annotated with validation status (TP/FP) and detection tools.
- **Java Benchmark 2** (`tse-dataset/`): Extended dataset from TSE paper.

**Source**: https://github.com/tsantalis/RefactoringMiner (src/test/resources/oracle/)

**Key fields per record** (after flattening commits → refactorings):
- `commit_sha`, `repository`, `author`, `time`
- `refactoring_type`, `description` — type and textual description of the refactoring
- `validation` — TP (true positive) or FP (false positive)
- `detectionTools` — which tools detected this refactoring

**What it gives the pipeline**: commit-level refactoring ground truth across many repositories, validated by experts. Does NOT contain source code (only references to commits) or smell information.

### Dataset 3: Technical Debt Dataset v2.0.1 (TDD) by Lenarduzzi et al.

Longitudinal dataset tracking SonarQube issues across 31 Java projects over their full commit history (master branch).

**Source**: SQLite database (`td_V2.db`, 1.47 GB)

**Key tables and sizes**:
- `SONAR_ISSUES` — 1,024,614 rows: SonarQube issues with rule, severity, component, status, creation/close analysis keys
- `SONAR_ANALYSIS` — 67,550 rows: maps analysis_key → git revision (commit hash)
- `REFACTORING_MINER` — 362,253 rows: refactorings detected by RMiner per commit
- `GIT_COMMITS` / `GIT_COMMITS_CHANGES` — commit metadata and file changes

**SonarQube rules use legacy `squid:` prefix** (not `java:`): `squid:S138` (Long Method), `squid:S1541` (Complex Method), `squid:MethodCyclomaticComplexity`, `squid:S1067` (Conditional Complexity), `squid:S106` (Print Statements), etc.

**Key capability**: By joining SONAR_ISSUES with SONAR_ANALYSIS, you can get `creationCommitHash` and `closeCommitHash` for each issue — providing ground truth for when smells appeared and were resolved. Combined with REFACTORING_MINER table, you can correlate smell resolution with specific refactoring types.

**What it gives the pipeline**: smell lifecycle data (S₀ → commit → S₁ transitions), refactoring-smell correlations, longitudinal smell evolution. Does NOT contain source code.

---

## My Pipeline (conf.tex)

The system evaluates how well AI agents can prioritize and execute code smell refactorings. The core experimental loop:

```
For each evaluated commit:
  1. Reconstruct pre-refactoring smell set S₀ (via SonarQube scan of parent commit)
  2. Build dependency graph between smells (positive/negative dependencies)
  3. Apply planner (Greedy or BeFS) to produce ordered plan π
  4. Compare π[0] against developer's actual first refactoring
  5. Execute refactoring via LLM agent
  6. Verify: compile + test → S₁
  7. Measure: smells resolved, smells created, behavior preservation
```

### Stages A–J

| Stage | Description |
|-------|-------------|
| A | Load source code, detect build system (Maven/Gradle) |
| B | Check test coverage |
| C | Generate missing tests via LLM (planned) |
| D | Run test suite (baseline) |
| E | Build smell dependency graph using DEPENDENCY_RULES (8 smell types, positive/negative edges) |
| F | SonarQube scan → S₀ (pre-refactoring smell set) |
| G | Developer selects smells to address |
| H | Planner (Greedy/BeFS) → plan π using PZ formula |
| I | LLM refactoring execution (chain-of-thought) |
| J | Verify: compile + test, rollback on failure |

### PZ Prioritization Formula

```
PZ_i = severity(s_i) + Σ(weight × positive_dependencies) - Σ(weight × negative_dependencies)
```

### Dependency Model (8 SonarQube smell types)

| Smell Type | SonarQube Rule | Positive Dependencies (resolves) | Negative Dependencies (introduces) |
|---|---|---|---|
| Long Method | java:S138 | Feature Envy, Dup. Code, Long Param List | Long Method, Long Param List |
| Complex Method | java:S1541 | Feature Envy, Dup. Code, Long Param List | Long Method, Long Param List |
| Long Parameter List | java:S107 | Data Clumps | Data Class |
| God Class | java:S1200 | Feature Envy, Data Clumps | Long Method, Data Class, Inappropriate Intimacy |
| Large Class | java:S110 | Feature Envy, Data Clumps | Long Method, Data Class |
| Duplicated Conditions | java:S1871 | Divergent Change | Large Class |
| Conditional Complexity | java:S1067 | Feature Envy, Dup. Code | Long Method |
| Print Statements | java:S106 | Needless Part | Data Class, Lazy Class |

### Evaluation Metrics

- **plan_efficiency** η = steps / smells_resolved
- **negative_dependency_rate** ρ = new_smells / refactorings_executed
- **compile_and_test_pass_rate** — behavioral preservation
- **first_action_match** — does π[0] match developer's actual refactoring?

---

## What I Need From You

1. **For each dataset**: Find the original paper/documentation, confirm the structure I described, note any corrections or additional capabilities I missed.

2. **Unified approach feasibility**: Can all three datasets serve this pipeline? Specifically:
   - SWE-Refactor: has code but no smells — can I reconstruct S₀ via SonarQube scan?
   - RMiner: has refactoring types but no code/smells — what role can it play?
   - TDD: has smell lifecycle but no code — can it validate the dependency model and PZ formula?

3. **Concrete proposal**: For each dataset, which pipeline stages (A–J) can it cover, what data it provides, and what must be reconstructed/approximated.

4. **Risks and limitations**: What assumptions am I making that might not hold? What biases exist in each dataset?
