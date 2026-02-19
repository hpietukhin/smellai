I am working on a master's thesis project that evaluates LLM-based code smell detection and refactoring using a multi-agent system. I need to find publicly available datasets that contain **code smells linked to their corresponding refactoring operations (or sequences of refactorings)** on Java source code. The dataset should be suitable for evaluating an automated pipeline that:

1. Detects code smells in Java projects via SonarQube
2. Plans an optimal **sequence of refactorings** using dependency-aware prioritization (based on Markovič & Polášek's positive/negative smell dependency theory)
3. Executes refactorings via LLM agents
4. Verifies behavior preservation via compilation and test execution

### What I already have

I already use these datasets:

- **RefactoringMiner 2.0 Oracle** (Tsantalis, Ketkar & Dig, IEEE TSE 2022) — 549 commits, 188 projects, 15,562 refactorings (11,514 true positives), 101 unique refactoring types. Provides before/after file pairs and refactoring metadata. Strong coverage of: Extract Method (1,033 TP), Move Class (855 TP), Move Method (266 TP), Extract Class (108 TP), Extract Superclass (72 TP), Inline Method (120 TP).

- **SWE-Refactor** (2025) — 1,099 pure refactoring records from 18 Java projects. Each record includes: target method code, full class before/after, class hierarchy, caller/callee graphs, build configuration, JaCoCo test coverage, compilation and test results. Covers: Extract Method (441), Move Method (410), Extract And Move Method (142), Inline Method (71).

- **DACOS** (Nandani, Saad & Sharma, MSR 2023) — referenced but not yet integrated. Tracks 4 smell types in MySQL: Complex Method, Insufficient Modularization, Long Parameter List, Multifaceted Abstraction.

### The 8 code smell types I detect (via SonarQube)

| Smell Type | SonarQube Rule | Severity | Typical Refactoring |
|---|---|---|---|
| Long Method | java:S138 | HIGH | Extract Method |
| Complex Method | java:S1541 | HIGH | Extract Method / Decompose Conditional |
| Conditional Complexity | java:S1067 | MEDIUM | Replace Conditional with Polymorphism / Extract Method |
| Long Parameter List | java:S107 | MEDIUM | Introduce Parameter Object |
| God Class | java:S1200 | HIGH | Extract Class / Move Method |
| Large Class | java:S110 | HIGH | Extract Class / Extract Superclass |
| Duplicated Conditions | java:S1871 | MEDIUM | Consolidate Conditional Expression |
| Print Statements | java:S106 | LOW | Replace with Logger |

### What I am missing — the gaps

My current datasets cover 6 of 8 smell types well, but have gaps for these refactoring types:

1. **Introduce Parameter Object** — needed for Long Parameter List smells. Neither RMiner oracle nor SWE-Refactor has significant coverage. The RMiner oracle has "Merge Parameter" (28 TP) which partially covers this.
2. **Consolidate Conditional Expression** — needed for Duplicated Conditions smells. No ground truth in either dataset.
3. **Decompose Conditional** — needed for Conditional Complexity smells. Currently using Extract Method as a workaround.
4. **Replace with Logger** — needed for Print Statements smells. Low priority but still a gap.

### What I need from a new dataset

I am looking for datasets that have **any combination** of the following properties:

#### Must-have (at least one):
- **Code smells explicitly annotated** on Java source code (not just refactorings — actual smell labels like "Long Method", "God Class", "Feature Envy", etc.)
- **Refactoring operations linked to specific code smells** — i.e., "this refactoring was applied to fix this smell"
- **Sequences/chains of refactorings** on the same codebase — showing how multiple refactorings are applied in order to resolve a cluster of related smells
- **Before/after source code** for each refactoring step (method-level or file-level)

#### Highly desirable:
- **Java source code** (my entire pipeline is Java-specific: Maven/Gradle builds, SonarQube rules, JDK switching)
- Coverage of **Introduce Parameter Object**, **Consolidate Conditional**, or **Decompose Conditional** refactoring types
- **Smell dependency information** — which smells are related, which refactorings resolve or create other smells
- **Compilation and test information** — whether the code compiles and tests pass after refactoring
- Multiple refactorings per commit or per file (not just single isolated refactorings)
- **Ground truth annotations** by developers or validated by tools like RefactoringMiner

#### Nice-to-have:
- **Smell co-occurrence data** — which smells appear together in the same class/method
- **Code metrics** (cyclomatic complexity, LOC, CBO, WMC, LCOM, RFC) before and after refactoring
- **Git history** — access to the actual repositories and commits
- Mapping between code smells and SonarQube rules
- Available as structured data (JSON, CSV, SQL) rather than only as raw repository mining output
- Published at a reputable venue (ICSE, FSE, ASE, MSR, EMSE, TSE, TOSEM, JSS, IST)

### Specific datasets and resources to investigate

Please search for and evaluate the following (and suggest any others you find):

1. **Qualitas Corpus** — large collection of Java systems used in empirical studies. Does it have smell annotations?
2. **Landfill dataset** (Palomba et al.) — code smell detection and evolution dataset
3. **MLCQ** (Machine Learning for Code Quality) — crowd-sourced code smell annotations
4. **Technical Debt Dataset** (TDD) by Lenarduzzi et al.
5. **Ptidej / JDeodorant** datasets — tool-generated smell detection results on open-source projects
6. **Bavota et al.** datasets on code smell and refactoring co-evolution
7. **ref-Dataset** (Liu et al., 2025) — 100 pure atomic refactorings from real Java projects
8. **Extended community corpus** (Pomian et al., 2024) — 1,752 Extract Method instances
9. **RefactorBench** (Gautam et al., 2025) — Python-based but may have transferable methodology
10. **SEART GitHub Search / GHS** for mining Java repositories with specific refactoring patterns
11. Any datasets from **Fowler's refactoring catalog** mapped to real code
12. Any datasets linking **SonarQube/PMD/Checkstyle** detections to subsequent refactoring commits
13. **SmellRefactoring datasets** — any dataset explicitly linking smell detection to refactoring application
14. Datasets from Sharma, Tsantalis, Di Penta, Bavota, Palomba, or other code smell researchers

### What to return for each dataset found

For each dataset, please provide:

1. **Name and citation** (authors, venue, year)
2. **URL/availability** (GitHub repo, Zenodo, institutional page, etc.)
3. **Size** (number of projects, classes, smells, refactorings)
4. **Content summary** — what exactly is in the dataset (smells? refactorings? before/after code? metrics?)
5. **Smell types covered** (if applicable)
6. **Refactoring types covered** (if applicable)
7. **Language** (Java preferred, but note others)
8. **Format** (JSON, CSV, SQL, raw repos, etc.)
9. **Gap coverage** — does it help fill the gaps listed above (Introduce Parameter Object, Consolidate Conditional, Decompose Conditional)?
10. **Integration effort** — how much work to adapt it to my MLflow GenAI format (`{inputs, expectations, tags}`)
11. **Limitations** — what's missing or problematic about this dataset for my use case

### Additional context for better search

- My system uses **dependency-aware prioritization** where refactoring order matters: positive dependencies (refactoring A resolves smell B) and negative dependencies (refactoring A creates smell C)
- I need to evaluate a **best-first search planner** that explores multiple refactoring paths and picks the sequence with minimal negative side effects
- The key research question is: "Can LLM agents plan and execute optimal refactoring sequences that account for smell dependencies and cascading effects?"
- I already have strong coverage for **Extract Method** and **Move Method**. The biggest gaps are in parameter-level and conditional-level refactorings
- The dataset must support **automated evaluation** — I need to programmatically compare LLM output against ground truth
- I am particularly interested in datasets where **multiple smells coexist** in the same class and **multiple refactorings** are applied to resolve them, because this is where sequencing and dependency analysis matter most

### Output format

Please organize your response as:
1. **Top recommendations** (3-5 datasets most likely to help)
2. **Additional datasets worth investigating** (with brief notes)
3. **Datasets that won't work** (and why, to save investigation time)
4. **Suggested mining strategies** if no perfect dataset exists (e.g., how to create the missing data by combining SonarQube scans with RefactoringMiner results on specific repositories)
