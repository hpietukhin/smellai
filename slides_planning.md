---
theme: default
background: https://cover.sli.dev
title: Best-First Search Planning for Smell Refactoring
info: Introducing planning into the agentic refactoring workflow - Best-first search approach for optimal refactoring sequences
class: text-center
canvasWidth: 1280
aspectRatio: '16/10'
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Best-First Search Planning for Smell Refactoring

Introducing planning into the multi-agent refactoring workflow

Havriil Pietukhin

<style>
  .mermaid svg {
    max-height: 480px !important;
    width: auto !important;
  }
  .slidev-code {
    font-size: 0.82em !important;
    line-height: 1.35 !important;
  }
</style>

---

# Current approach: Greedy prioritization

The system currently uses a **greedy algorithm** to select the next smell to refactor.

$$PZ_i = Severity_i + \sum_{j \in \text{PositiveDeps}(i)} w_{\text{impact}}$$

**Algorithm:**
1. Compute PZ for all detected smells
2. Pick the smell with **maximum PZ**
3. Refactor it
4. Remove from graph, recalculate
5. Repeat

**Problem:** Greedy is **myopic** - it commits to each choice without foreseeing cascading effects.

---

# Why greedy falls short


<div class="grid grid-cols-2 gap-8">
<div>

**Scenario: God Class + Long Methods**

A God Class (PZ=7) is picked first. Refactoring it via Extract Class may:

- (+) Resolve Feature Envy, Data Clumps
- (-) **Create** new Long Methods in extracted classes
- (-) **Create** Inappropriate Intimacy

The newly created smells force additional unplanned refactorings.

</div>
<div>

```mermaid
graph TD
  A["🔴 God Class<br/>PZ=7"] -->|Extract Class| B["✅ Feature Envy resolved"]
  A -->|Extract Class| C["❌ New Long Method"]
  A -->|Extract Class| D["❌ Inappropriate Intimacy"]
  C -->|Extract Method| E["⚠️ New Long Param List"]
  style A fill:#ff6b6b,color:#fff
  style B fill:#51cf66,color:#fff
  style C fill:#ff6b6b,color:#fff
  style D fill:#ff6b6b,color:#fff
  style E fill:#ffa94d,color:#fff
```

</div>
</div>

---

# Proposed: Best-first search planning

Model the refactoring process as a **search problem** over the space of smell configurations.

| Component | Definition |
|-----------|-----------|
| **State** | Current set of smells S = {s₁, s₂, ..., sₙ} with severities |
| **Initial state** | S₀ = all smells detected by SonarQube |
| **Actions** | Apply refactoring r(sᵢ) to address smell sᵢ |
| **Transition** | S' = (S \ resolved(sᵢ)) ∪ created(sᵢ) |
| **Heuristic** | h(S) = Σ severity(s) - estimated_positive_impact |
| **Goal** | Minimize total remaining smell impact |

**Key advantage:** Can explore multiple paths and **backtrack** when negative dependencies make a path worse.

---

# Best-first search: how it works


```mermaid
graph TD
  S0["S₀: {God Class, Long Method,<br/>Complex Method, Dup Conditions}<br/>h=17"] -->|"Refactor God Class"| S1a["S₁a: {Long Method, Complex Method,<br/>Dup Conditions, <b>NEW: Inapp. Intimacy</b>}<br/>h=14"]
  S0 -->|"Refactor Long Method"| S1b["S₁b: {God Class, Complex Method,<br/>Dup Conditions}<br/><i>+resolved: Dup Code, Comments</i><br/>h=11"]
  S0 -->|"Refactor Complex Method"| S1c["S₁c: {God Class, Long Method,<br/>Dup Conditions}<br/>h=13"]
  S1b -->|"Refactor God Class"| S2["S₂: {Dup Conditions}<br/>h=4"]
  S2 -->|"Refactor Dup Conditions"| S3["S₃: {} <br/>h=0 ✅"]
  style S0 fill:#4c6ef5,color:#fff
  style S1b fill:#51cf66,color:#fff
  style S2 fill:#51cf66,color:#fff
  style S3 fill:#37b24d,color:#fff
  style S1a fill:#ffa94d,color:#fff
  style S1c fill:#ffa94d,color:#fff
```

Priority queue expands the node with **lowest h(S)** first.

Path: Long Method → God Class → Dup Conditions (h: 17→11→4→0)

---

# Greedy vs best-first: side by side

<div class="grid grid-cols-2 gap-4">
<div>

### Greedy path
```
Step 1: God Class (PZ=7)
  → creates Inapp. Intimacy
  → creates Long Method

Step 2: Long Method (PZ=6)
  → creates Long Param List

Step 3: Complex Method (PZ=6)
  → creates Long Param List

Step 4: Dup Conditions (PZ=4)
Step 5: Inapp. Intimacy (PZ=2)
Step 6: Long Param List x2 (PZ=2)
─────────────────────
Total steps: 7
New smells created: 3
```

</div>
<div>

### Best-first path
```
Step 1: Long Method (PZ=6)
  → resolves Dup Code, Comments
  → positive cascade!

Step 2: God Class (PZ=7)
  → fewer methods to extract
  → reduced negative impact

Step 3: Complex Method (PZ=6)
Step 4: Dup Conditions (PZ=4)
─────────────────────
Total steps: 4
New smells created: 0
```

</div>
</div>

Best-first search finds a shorter path by **avoiding paths that create new smells**.

---

# The heuristic function

The quality of best-first search depends on the **heuristic function** h(S).

$$h(S) = \sum_{s \in S} \text{severity}(s) - \alpha \cdot \text{positive\_deps}(s) + \beta \cdot \text{negative\_deps}(s)$$

Where:
- $\text{severity}(s)$: HIGH=3, MEDIUM=2, LOW=1
- $\text{positive\_deps}(s)$: count of smells that refactoring s would also resolve
- $\text{negative\_deps}(s)$: count of smells that refactoring s might create
- $\alpha, \beta$: tunable weights (default: $\alpha=2, \beta=1.5$)

**Properties:**
- **Admissible** if $\alpha$ is not too aggressive (never overestimates benefit)
- **Consistent** since positive/negative dependency counts are monotonic
- Reduces to greedy PZ when search depth = 1

---

# What smells do we detect?

Our SonarQube integration detects **8 smell types**:

| Smell Type | SonarQube Rule | Severity | Typical Refactoring |
|-----------|---------------|----------|-------------------|
| Long Method | java:S138 | HIGH | **Extract Method** |
| Complex Method | java:S1541 | HIGH | **Extract Method** / Decompose Conditional |
| Conditional Complexity | java:S1067 | MEDIUM | Replace Conditional / **Extract Method** |
| Long Parameter List | java:S107 | MEDIUM | Introduce Parameter Object |
| God Class | java:S1200 | HIGH | Extract Class / **Move Method** |
| Large Class | java:S110 | HIGH | Extract Class |
| Duplicated Conditions | java:S1871 | MEDIUM | Consolidate Conditional |
| Print Statements | java:S106 | LOW | Replace with Logger |

Bold = available in our datasets.

---

# What refactorings do the datasets have?

<div class="grid grid-cols-2 gap-4">
<div>

### RMiner oracle (549 commits, 188 repos)
15,562 refactorings (11,514 TP), **101 unique types**.

| Refactoring | Total | TP |
|------------|------:|---:|
| Rename Method | 1,161 | 400 |
| **Move Class** | 1,115 | 855 |
| **Extract Method** | 1,096 | 1,033 |
| **Move Method** | 1,067 | 266 |
| Change Variable Type | 910 | 811 |
| **Extract Variable** | 325 | 325 |
| **Extract And Move Method** | 295 | 127 |
| **Inline Method** | 149 | 120 |
| **Extract Class** | 108 | 108 |
| **Extract Superclass** | 80 | 72 |
| + 91 more types | ... | ... |

</div>
<div>

### SWE-Refactor dataset (1,099 records, 18 projects)
| Refactoring | Count |
|------------|------:|
| **Extract Method** | 441 |
| **Move Method** | 410 |
| Extract And Move Method | 142 |
| **Inline Method** | 71 |
| Move And Rename Method | 21 |
| Move And Inline Method | 14 |

Top projects: Guava (300), PMD (125), JUnit5 (105), Commons-IO (93), Checkstyle (91), Hibernate (63+89).

</div>
</div>

---

# Gap analysis: smells vs available refactorings


```mermaid
graph LR
  LM["Long Method"] --> EM["Extract Method ✅"]
  CM["Complex Method"] --> EM
  CC["Cond. Complexity"] --> EM
  CC --> DCond["Decompose Cond. ❌"]
  GC["God Class"] --> MM["Move Method ✅"]
  GC --> EC["Extract Class ✅ RMiner"]
  LC["Large Class"] --> EC
  LPL["Long Param List"] --> IPO["Intro Param Object ❌"]
  DC["Dup. Conditions"] --> CCP["Consolidate Cond. ❌"]
  PS["Print Statements"] --> RL["Replace Logger ❌"]

  style EM fill:#51cf66,color:#fff
  style MM fill:#51cf66,color:#fff
  style EC fill:#a9e34b,color:#333
  style DCond fill:#ff6b6b,color:#fff
  style IPO fill:#ff6b6b,color:#fff
  style CCP fill:#ff6b6b,color:#fff
  style RL fill:#ff6b6b,color:#fff
```

---

# Coverage summary

<div class="grid grid-cols-2 gap-8">
<div>

### Covered (green path)
- **Long Method** → Extract Method (SWE-Refactor: 441, RMiner: 1,033 TP)
- **Complex Method** → Extract Method (same datasets)
- **God Class** → Move Method (SWE-Refactor: 410) + Extract Class (RMiner: 108 TP)
- **Large Class** → Extract Class (RMiner: 108 TP) + Extract Superclass (72 TP)
- **Conditional Complexity** → Extract Method (workaround)

Both datasets together cover most method-level and some class-level refactorings.

</div>
<div>

### NOT covered (red path)
- **Long Parameter List** → Introduce Parameter Object (missing from both datasets)
- **Duplicated Conditions** → Consolidate Conditional (missing)
- **Print Statements** → Replace with Logger (missing, low priority)

Note: Extract Class is now covered by the full RMiner oracle (108 TP instances), significantly closing the gap for God Class / Large Class.

</div>
</div>

**Conclusion:** SWE-Refactor covers method-level refactorings. The full RMiner oracle adds **Extract Class (108 TP)**, **Extract Superclass (72 TP)**, and **91 more types**. Together they cover **6 of 8** smell types.

---

# Real Example 1: HttpJobExecutor (SWE-Refactor)

<style scoped>
pre { font-size: 0.72em; line-height: 1.4; }
</style>

From **shardingsphere-elasticjob** project, commit `1bdc817c`. Two refactorings on the same file.

<div class="grid grid-cols-2 gap-4">
<div>

**Smells in `HttpJobExecutor.process()`:**
- **Long Method** (java:S138) — 40+ lines, HTTP setup + request + response parsing all inline
- **Feature Envy** — `isWriteMethod(method)` takes a String param from `HttpParam` and checks it; belongs in `HttpParam` itself

**Two refactorings applied in this commit:**
1. Extract Method: `getConnectionInputStream()` from `process()`
2. Move Method: `isWriteMethod()` from HttpJobExecutor to HttpParam

</div>
<div>

```java
// BEFORE: process() — everything inline
public void process(ElasticJob elasticJob,
    JobConfiguration jobConfig, ...) {
  HttpParam httpParam = getHttpParam(jobConfig.getProps());
  HttpURLConnection connection = null;
  try {
    URL url = new URL(httpParam.getUrl());
    connection = (HttpURLConnection) url.openConnection();
    connection.setRequestMethod(httpParam.getMethod());
    connection.setDoOutput(true);
    connection.setConnectTimeout(httpParam.getConnectTimeout());
    connection.setReadTimeout(httpParam.getReadTimeout());
    // ... setup headers, connect
    if (isWriteMethod(httpParam.getMethod())  // Feature Envy
        && !Strings.isNullOrEmpty(data)) {
      // write data ...
    }
    int code = connection.getResponseCode();
    InputStream resultInputStream;
    if (isRequestSucceed(code)) {           // inline
      resultInputStream = connection.getInputStream();
    } else {
      log.warn("HTTP job {} response {}", ...);
      resultInputStream = connection.getErrorStream();
    }
    // ... read response, log result
  } catch (IOException ex) { throw ...; }
}
private boolean isWriteMethod(String method) {
  return Arrays.asList("POST","PUT","DELETE")
    .contains(method.toUpperCase());
}
```

</div>
</div>

---

# HttpJobExecutor: Best-First Search Plan


```mermaid
graph LR
  S0["S₀: process()<br/>{Long Method, Feature Envy}<br/>h = 3+2 = 5"] -->|"① Extract Method<br/>getConnectionInputStream()"| S1["S₁: {Feature Envy}<br/>✅ Long Method resolved<br/>h = 2"]

  S1 -->|"② Move Method<br/>isWriteMethod() → HttpParam"| S2["S₂: {} ✅<br/>✅ Feature Envy resolved<br/>h = 0"]

  style S0 fill:#ff6b6b,color:#fff
  style S1 fill:#ffa94d,color:#fff
  style S2 fill:#51cf66,color:#fff
```

Both steps use refactorings available in SWE-Refactor: **Extract Method** (441 records) and **Move Method** (410 records).

**Why this order?** Extract Method first makes `process()` shorter, and the remaining `isWriteMethod()` becomes clearly misplaced — it only accesses `HttpParam` fields. The planner avoids the reverse (moving first would move the long method's complexity into HttpParam).

---

# HttpJobExecutor: Step 1 - Extract Method (real code)

<style scoped>
pre { font-size: 0.68em; line-height: 1.35; }
</style>

<div class="grid grid-cols-2 gap-4">
<div>

**BEFORE** — response handling inline in process():
```java
int code = connection.getResponseCode();
InputStream resultInputStream;
if (isRequestSucceed(code)) {
  resultInputStream = connection.getInputStream();
} else {
  log.warn("HTTP job {} executed with response code {}",
    jobConfig.getJobName(), code);
  resultInputStream = connection.getErrorStream();
}
StringBuilder result = new StringBuilder();
try (BufferedReader bufferedReader = new BufferedReader(
    new InputStreamReader(resultInputStream,
      StandardCharsets.UTF_8))) {
  String line;
  while (null != (line = bufferedReader.readLine())) {
    result.append(line);
  }
}
```

</div>
<div>

**AFTER** — extracted getConnectionInputStream():
```java
int responseCode = connection.getResponseCode();
StringBuilder result = new StringBuilder();
try (
    InputStream inputStream = getConnectionInputStream(
      jobConfig.getJobName(), connection, responseCode);
    BufferedReader bufferedReader = new BufferedReader(
      new InputStreamReader(inputStream,
        StandardCharsets.UTF_8))) {
  String line;
  while (null != (line = bufferedReader.readLine())) {
    result.append(line);
  }
}

private InputStream getConnectionInputStream(
    String jobName, HttpURLConnection connection,
    int code) throws IOException {
  if (isRequestSucceed(code)) {
    return connection.getInputStream();
  }
  log.warn("HTTP job {} executed with response code {}",
    jobName, code);
  return connection.getErrorStream();
}
```

**Resolves:** Long Method -- `process()` is now shorter and delegates I/O handling.

</div>
</div>

---

# HttpJobExecutor: Step 2 - Move Method (real code)

<style scoped>
pre { font-size: 0.72em; line-height: 1.4; }
</style>

<div class="grid grid-cols-2 gap-4">
<div>

**BEFORE** — `isWriteMethod()` lives in HttpJobExecutor:
```java
// In HttpJobExecutor:
private boolean isWriteMethod(final String method) {
  return Arrays.asList("POST", "PUT", "DELETE")
    .contains(method.toUpperCase());
}

// Called as:
if (isWriteMethod(httpParam.getMethod())
    && !Strings.isNullOrEmpty(data)) {
  try (OutputStream outputStream =
      connection.getOutputStream()) {
    outputStream.write(
      data.getBytes(StandardCharsets.UTF_8));
  }
}
```

Feature Envy: the method only accesses `HttpParam`'s method field, not any `HttpJobExecutor` state.

</div>
<div>

**AFTER** — moved to HttpParam where it belongs:
```java
// In HttpParam (new location):
/**
 * Is write method.
 * @return write method or not
 */
public boolean isWriteMethod() {
  return Arrays.asList("POST", "PUT", "DELETE")
    .contains(method.toUpperCase());
}

// Called in HttpJobExecutor as:
if (httpParam.isWriteMethod()
    && !Strings.isNullOrEmpty(data)) {
  try (OutputStream outputStream =
      connection.getOutputStream()) {
    outputStream.write(
      data.getBytes(StandardCharsets.UTF_8));
  }
}
```

**Resolves:** Feature Envy -- method now lives with the data it operates on.

Parameter `String method` removed; uses own field `this.method`.

</div>
</div>

---

# HttpJobExecutor: SWE-Refactor evaluation data

<style scoped>
pre { font-size: 0.72em; line-height: 1.4; }
</style>

This commit (`1bdc817c6a`) is a **real record** in SWE-Refactor with compile and test metadata.

<div class="grid grid-cols-2 gap-4">
<div>

**Dataset record fields (per refactoring):**
- `sourceCodeBeforeRefactoring` / `After` -- exact method-level diff
- `sourceCodeBeforeForWhole` / `After` -- full file context
- `compileResultBefore` / `compileResultCurrent` -- compilation status
- `compileJDK` -- required Java version
- `hasTestC` -- whether tests cover this code
- `coverageInfo` -- line-level test coverage
- `callInfo` -- caller/callee graph for the refactored method
- `invokedMethodSet` -- methods invoked by the refactored code
- `diffSourceCode` -- unified diff format

**Evaluation criteria (from the paper):**
1. Code compiles after refactoring
2. All existing tests pass
3. RefactoringMiner detects the expected refactoring type in the LLM output

</div>
<div>

**What the agent must produce for Step 1:**
```java
// Extract Method: getConnectionInputStream()
// Input: the full HttpJobExecutor.java file
// Expected output: new private method extracted
//   from process() that handles InputStream selection

// The SWE-Refactor record provides:
// - Exact before/after code for comparison
// - Compilation command: mvn clean compile
// - Test command: mvn test
// - JDK version: 8
```

**What the agent must produce for Step 2:**
```java
// Move Method: isWriteMethod() → HttpParam
// Input: HttpJobExecutor.java + HttpParam.java
// Expected output:
//   1. Remove isWriteMethod(String) from Executor
//   2. Add isWriteMethod() to HttpParam (no param)
//   3. Update caller: httpParam.isWriteMethod()

// Ground truth available in dataset for both steps
```

**DeepSeek-V3 success rate on similar Extract Method: 68%**
**GPT-4o-mini success rate on similar Move Method: 42%**

</div>
</div>

---

# Real Example 2: checkstyle UnusedLocalVariableCheck (SWE-Refactor)

<style scoped>
pre { font-size: 0.68em; line-height: 1.35; }
</style>

From **checkstyle** project, commit `1ca66693`. **5 refactorings** across 2 files -- Duplicated Code + God Class.

<div class="grid grid-cols-2 gap-4">
<div>

**Smells detected:**
- **Duplicated Code** — `extractQualifiedName()` is copy-pasted in both `UnusedLocalVariableCheck` and `FinalClassCheck` (source comments say "Duplicated, until issue #11201")
- **God Class** — `UnusedLocalVariableCheck` contains utility methods (`getShortNameOfAnonInnerClass`, `getQualifiedTypeDeclarationName`) unrelated to its core responsibility
- **Long Method** — `getTheNearestClass()` has inline lambda logic

**5 refactorings applied in this commit:**
1. Move Method: `extractQualifiedName()` from UnusedLocalVariableCheck
2. Move Method: `extractQualifiedName()` from FinalClassCheck
3. Move Method: `getQualifiedTypeDeclarationName()` from UnusedLocalVariableCheck
4. Move Method: `getShortNameOfAnonInnerClass()` from UnusedLocalVariableCheck
5. Extract Method: `getTypeDeclarationNameMatchingCountDiff()` from `getTheNearestClass()`

</div>
<div>

**The duplicated method (real source code):**
```java
// In UnusedLocalVariableCheck.java:
/**
 * ...
 * Duplicated, until
 * <a>https://github.com/checkstyle/checkstyle/issues/11201</a>
 */
private static String extractQualifiedName(DetailAST ast) {
  return FullIdent.createFullIdent(ast).getText();
}

// In FinalClassCheck.java — IDENTICAL:
private static String extractQualifiedName(DetailAST ast) {
  return FullIdent.createFullIdent(ast).getText();
}
```

**Also duplicated:**
```java
// In UnusedLocalVariableCheck.java:
/**
 * Duplicated, until issue #11201
 */
private static String getQualifiedTypeDeclarationName(
    String packageName,
    String outerClassQualifiedName,
    String className) {
  // ... 15 lines of qualified name construction
}
```

</div>
</div>

---

# checkstyle: Best-First Search Plan


```mermaid
graph LR
  S0["S₀: UnusedLocalVarCheck<br/>{Dup Code x2, God Class, Long Method}<br/>h = 2+2+3+3 = 10"] -->|"① Move Methods<br/>extractQualifiedName() x2<br/>getQualifiedTypeDecl()"| S1["S₁: {God Class↓, Long Method}<br/>✅ Dup Code resolved<br/>h = 4"]

  S1 -->|"② Move Method<br/>getShortNameOfAnonInner()"| S2["S₂: {Long Method}<br/>✅ God Class resolved<br/>h = 3"]

  S2 -->|"③ Extract Method<br/>getTypeDeclNameMatchingDiff()"| S3["S₃: {} ✅<br/>✅ Long Method resolved<br/>h = 0"]

  style S0 fill:#ff6b6b,color:#fff
  style S1 fill:#ffa94d,color:#fff
  style S2 fill:#ffd43b,color:#333
  style S3 fill:#51cf66,color:#fff
```

All steps use **Move Method** and **Extract Method** -- fully evaluable with SWE-Refactor ground truth.

**Why this order?** Fixing duplicated code first (Move Methods to shared utility) is the highest priority because it affects multiple files. The planner's heuristic scores Dup Code resolution highest due to cross-file positive dependencies.

---

# checkstyle: Step 1 - Move Duplicated Methods (real code)

<style scoped>
pre { font-size: 0.68em; line-height: 1.35; }
</style>

<div class="grid grid-cols-2 gap-4">
<div>

**BEFORE** — same method in 2 different check classes:
```java
// UnusedLocalVariableCheck.java:
/**
 * Duplicated, until
 * <a>github.com/checkstyle/checkstyle/issues/11201</a>
 */
private static String extractQualifiedName(
    DetailAST ast) {
  return FullIdent.createFullIdent(ast).getText();
}

// FinalClassCheck.java — identical copy:
/**
 * Get name of class in ast.
 */
private static String extractQualifiedName(
    DetailAST ast) {
  return FullIdent.createFullIdent(ast).getText();
}
```

Both `private` — each class has its own copy.

</div>
<div>

**AFTER** — moved to shared utility (CheckUtil):
```java
// CheckUtil.java (shared location):
/**
 * Get name of package and super class of anon inner class
 * by concatenating identifier values under DOT.
 */
public static String extractQualifiedName(
    DetailAST ast) {
  return FullIdent.createFullIdent(ast).getText();
}

// UnusedLocalVariableCheck now calls:
CheckUtil.extractQualifiedName(firstChild);

// FinalClassCheck now calls:
CheckUtil.extractQualifiedName(ast);
```

**Resolves:** Duplicated Code -- single source of truth.

`private` → `public` access modifier change enables sharing.

Same pattern for `getQualifiedTypeDeclarationName()` (15 lines, also marked "Duplicated" in source).

</div>
</div>

---

# Real Dataset: EduStepicConnector (RMiner)

<style scoped>
pre { font-size: 0.72em; line-height: 1.4; }
</style>

From **RMiner dataset** — real Extract Method from JetBrains IntelliJ (commit `7ed3f27`).

<div class="grid grid-cols-2 gap-4">
<div>

**BEFORE** — getCourses() has inline logic:
```java
@NotNull
public static List<CourseInfo> getCourses() {
  try {
    List<CourseInfo> result = new ArrayList<>();
    final List<CourseInfo> courseInfos =
      getFromStepic("courses", CoursesContainer.class)
        .courses;
    for (CourseInfo info : courseInfos) {
      final String courseType = info.getType();
      if (StringUtil.isEmptyOrSpaces(courseType))
        continue;
      final List<String> typeLanguage =
        StringUtil.split(courseType, " ");
      if (typeLanguage.size() == 2
          && PYCHARM_PREFIX.equals(typeLanguage.get(0))) {
        result.add(info);
      }
    }
    return result;
  } catch (IOException e) {
    LOG.error("Cannot load course list "
      + e.getMessage());
  }
  return Collections.emptyList();
}
```

Single-page fetch only. No pagination support.

</div>
<div>

**AFTER** — Extract Method + pagination:
```java
@NotNull
public static List<CourseInfo> getCourses() {
  try {
    List<CourseInfo> result = new ArrayList<>();
    int pageNumber = 0;
    boolean hasNext =
      addCoursesFromStepic(result, pageNumber);
    while (hasNext) {
      pageNumber += 1;
      hasNext =
        addCoursesFromStepic(result, pageNumber);
    }
    return result;
  } catch (IOException e) {
    LOG.error("Cannot load course list "
      + e.getMessage());
  }
  return Collections.emptyList();
}

private static boolean addCoursesFromStepic(
    List<CourseInfo> result, int pageNumber)
    throws IOException {
  final String url = pageNumber == 0
    ? "courses"
    : "courses?page=" + String.valueOf(pageNumber);
  final CoursesContainer coursesContainer =
    getFromStepic(url, CoursesContainer.class);
  final List<CourseInfo> courseInfos =
    coursesContainer.courses;
  for (CourseInfo info : courseInfos) { /* filter */ }
  return coursesContainer.meta.containsKey("has_next")
    && coursesContainer.meta.get("has_next") == TRUE;
}
```

**Refactoring:** Extract Method + Extract Variable

</div>
</div>

---

# Real Dataset: ScrollableToolbarPopupMenu (RMiner)

<style scoped>
pre { font-size: 0.78em; line-height: 1.4; }
</style>

From **RMiner dataset** — real Extract Method from RStudio (commit `cb49e43`).

<div class="grid grid-cols-2 gap-4">
<div>

**BEFORE** — inline style manipulation:
```java
@Override
protected Widget wrapMenuBar(ToolbarMenuBar menuBar) {
  scrollPanel_ = new ScrollPanel(menuBar);
  scrollPanel_.addStyleName(
    ThemeStyles.INSTANCE.scrollableMenuBar());
  scrollPanel_.getElement().getStyle()
    .setOverflowY(Overflow.AUTO);
  scrollPanel_.getElement().getStyle()
    .setOverflowX(Overflow.HIDDEN);
  scrollPanel_.getElement().getStyle()
    .setProperty("maxHeight", getMaxHeight() + "px");
  return scrollPanel_;
}
```

Height setting is embedded in widget creation — can't be changed later.

</div>
<div>

**AFTER** — extracted setMaxHeight():
```java
@Override
protected Widget wrapMenuBar(ToolbarMenuBar menuBar) {
  scrollPanel_ = new ScrollPanel(menuBar);
  scrollPanel_.addStyleName(
    ThemeStyles.INSTANCE.scrollableMenuBar());
  scrollPanel_.getElement().getStyle()
    .setOverflowY(Overflow.AUTO);
  scrollPanel_.getElement().getStyle()
    .setOverflowX(Overflow.HIDDEN);
  setMaxHeight(getMaxHeight());
  return scrollPanel_;
}

protected void setMaxHeight(int maxHeight) {
  scrollPanel_.getElement().getStyle()
    .setProperty("maxHeight", maxHeight + "px");
}
```

Now subclasses can dynamically adjust height.

</div>
</div>

**This is exactly what the LLM agent must produce** — matching this ground truth is how we evaluate.

---

# Test Generation for Behavior Preservation

<style scoped>
pre { font-size: 0.72em; line-height: 1.4; }
</style>

Agent A7 generates tests **before** refactoring, Agent A6 runs them **after** each step.

<div class="grid grid-cols-2 gap-4">
<div>

**For EduStepicConnector (Extract Method):**
```java
@Test
void getCoursesReturnsFilteredPycharmCourses() {
  // Setup: mock getFromStepic to return test data
  CourseInfo pycharm = new CourseInfo();
  pycharm.setType("pycharm python");
  CourseInfo other = new CourseInfo();
  other.setType("java basics");

  // Before refactoring: single page
  List<CourseInfo> courses = getCourses();
  assertFalse(courses.isEmpty());
  assertTrue(courses.stream()
    .allMatch(c -> c.getType()
      .startsWith("pycharm")));
}

@Test
void getCoursesReturnsEmptyOnIOException() {
  // Mock: getFromStepic throws IOException
  List<CourseInfo> courses = getCourses();
  assertTrue(courses.isEmpty());
}
```

After refactoring, **same tests must still pass** with the new pagination-enabled `addCoursesFromStepic()`.

</div>
<div>

**For HttpJobExecutor (Move Method):**
```java
@Test
void processExecutesGetRequest() {
  // Verify GET request works before refactoring
  HttpJobExecutor executor = new HttpJobExecutor();
  JobConfiguration config = createConfig(
    "http://example.com/api", "GET", null);
  // After Move Method: isWriteMethod() is on HttpParam
  // but process() behavior must remain identical
  executor.process(null, config, null, shardingCtx);
  // Verify no output stream opened for GET
}

@Test
void processExecutesPostWithBody() {
  JobConfiguration config = createConfig(
    "http://example.com/api", "POST",
    "{\"key\": \"value\"}");
  // After refactoring, POST must still send body
  executor.process(null, config, null, shardingCtx);
  // Verify output stream received data
}

@Test
void processHandlesErrorResponse() {
  // After Extract Method: getConnectionInputStream()
  // must handle error codes the same way
  // Verify warn log + error stream on 500
}
```

</div>
</div>

---

# Real Example 3: languagetool UkrainianTagger (RMiner oracle)

<style scoped>
pre { font-size: 0.72em; line-height: 1.4; }
</style>

From **languagetool** project, commit `bec15926de`. **9 TP refactorings** -- God Class decomposition.

<div class="grid grid-cols-2 gap-4">
<div>

**Smells in `UkrainianTagger`:**
- **God Class** (java:S1200) — tagger does compound tagging, pos-tag helpers, attribute storage, regex patterns
- **Long Method** (java:S138) — `guessCompoundTag()` has complex word analysis logic
- **Duplicated Code** — regex patterns and maps shared across concerns

**9 refactorings applied (all TP in oracle):**
1. **Extract Class**: `CompoundTagger` from `UkrainianTagger`
2. **Extract Method**: `doGuessCompoundTag()` from `guessCompoundTag()` in CompoundTagger
3. **Move Attribute**: `VIDMINKY_MAP` → `PosTagHelper`
4. **Extract Attribute**: `NUM_REGEX` in `PosTagHelper`
5. **Extract Attribute**: `CONJ_REGEX` in `PosTagHelper`
6. **Extract Variable**: `leftConj` in CompoundTagger
7. Change Method Access Modifier
8. Change Attribute Access Modifier
9. Rename Method

</div>
<div>

**Search plan for this scenario:**

```mermaid
graph LR
  S0["S₀: UkrainianTagger<br/>{God Class, LM, Dup}<br/>h = 8"] -->|"① Extract Class<br/>CompoundTagger"| S1["S₁: {LM in new class,<br/>scattered attrs}<br/>h = 5"]
  S1 -->|"② Extract Method<br/>doGuessCompound()"| S2["S₂: {scattered attrs}<br/>h = 2"]
  S2 -->|"③ Move/Extract Attrs<br/>→ PosTagHelper"| S3["S₃: {} ✅<br/>h = 0"]

  style S0 fill:#ff6b6b,color:#fff
  style S1 fill:#ffa94d,color:#fff
  style S2 fill:#ffd43b,color:#333
  style S3 fill:#51cf66,color:#fff
```

**Extract Class (108 TP in RMiner oracle)** is the key refactoring here. Without the full oracle data, this example would have no ground truth.

Step 1 resolves God Class. Step 2 resolves Long Method in the extracted class. Step 3 groups related attributes.

</div>
</div>

---

# Real Example 4: ParameterAssignmentCheck (SWE-Refactor)

<style scoped>
pre { font-size: 0.68em; line-height: 1.35; }
</style>

From **checkstyle**, commit `ebfc50d227`. **3 refactorings** on the **same file** -- Extract Method + 2 Inline Methods.

<div class="grid grid-cols-2 gap-4">
<div>

**Extract Method -- reusable parameter visitor:**
```java
// BEFORE: visitMethodParameters() has inline logic
private void visitMethodParameters(DetailAST ast) {
  DetailAST parameterDefAST =
    ast.findFirstToken(TokenTypes.PARAMETER_DEF);
  while (parameterDefAST != null) {
    if (parameterDefAST.getType() == TokenTypes.PARAMETER_DEF
        && !CheckUtil.isReceiverParameter(parameterDefAST)) {
      final DetailAST param =
        parameterDefAST.findFirstToken(TokenTypes.IDENT);
      parameterNames.add(param.getText());
    }
    parameterDefAST = parameterDefAST.getNextSibling();
  }
}

// AFTER: extracted visitParameters() for reuse
private void visitMethodParameters(DetailAST ast) {
  visitParameters(ast);  // delegation
}
private void visitParameters(DetailAST parametersAst) {
  // ... same loop logic, now reusable for lambdas
}
```

</div>
<div>

**Inline Method -- remove unnecessary delegation:**
```java
// BEFORE: trivial wrapper method
private void leaveMethodDef() {
  parameterNames = parameterNamesStack.pop();
}
// Called from leaveToken() via switch

// AFTER: inlined into leaveToken()
@Override
public void leaveToken(DetailAST ast) {
  final int type = ast.getType();
  if (TokenUtil.isOfType(type,
      TokenTypes.CTOR_DEF, TokenTypes.METHOD_DEF)
      || type == TokenTypes.LAMBDA
      && ast.getParent().getType()
        != TokenTypes.SWITCH_RULE) {
    parameterNames = parameterNamesStack.pop();
  }
}
```

**Same-file, opposite directions:** Extract to create reusable method, Inline to remove trivial wrapper. The planner must recognize that these don't conflict.

</div>
</div>

---

# Dependency rules powering the search

The search uses **Markovič & Polášek dependency rules** as its transition model.

<div class="grid grid-cols-2 gap-4 text-sm">
<div>

### Positive dependencies (resolves)
| Refactoring target | May also resolve |
|---|---|
| Long Method | Switch Statement, Feature Envy, Duplicated Code, Divergent Change, Comments, Long Param List |
| Complex Method | (same as Long Method) |
| Conditional Complexity | (same as Long Method) |
| Long Param List | Long Param List, Data Clumps |
| Large Class / God Class | Data Clumps, Feature Envy, Bad Class Content |
| Duplicated Conditions | Divergent Change, Shotgun Surgery |

</div>
<div>

### Negative dependencies (creates)
| Refactoring target | May create |
|---|---|
| Long Method | Long Method, Long Param List |
| Complex Method | Long Method, Long Param List |
| Long Param List | Data Class |
| Large Class / God Class | Long Method, Data Class, Inappropriate Intimacy, Message Chains |
| Duplicated Conditions | Large Class, Bad Inheritance |

</div>
</div>

---

# Integration into the agentic workflow


```mermaid
graph LR
  A0["A0: Test Coverage<br/>(setup)"] --> A1["A1: Smell Detection<br/>(SonarQube)"]
  A1 --> PLAN["🆕 PLANNER<br/>Best-First Search"]
  PLAN --> A5["A5: Refactoring<br/>(LLM execution)"]
  A5 --> A6["A6: Verification<br/>(test execution)"]
  A6 -->|"tests pass"| CHECK{"More smells<br/>in plan?"}
  A6 -->|"tests fail"| PLAN
  CHECK -->|"yes"| A5
  CHECK -->|"no"| END["END"]

  style PLAN fill:#4c6ef5,color:#fff,stroke:#364fc7,stroke-width:3px
  style A0 fill:#868e96,color:#fff
  style A1 fill:#868e96,color:#fff
  style A5 fill:#868e96,color:#fff
  style A6 fill:#868e96,color:#fff
  style END fill:#51cf66,color:#fff
```

**The planner replaces the greedy algorithm** in Agent 4 (Smell Prioritization).

Key change: Instead of a single priority queue, the planner produces an **ordered plan** — a sequence of (smell, refactoring) pairs — optimized for minimal total negative impact.

---

# Planner: replanning on failure


When verification fails, the planner **replans** from the current state.

```mermaid
graph LR
  START(( )) --> Detect["Detect<br/>(SonarQube)"]
  Detect -->|"smells found"| Plan["Plan<br/>(Best-First)"]
  Plan -->|"plan ready"| Refactor["Refactor<br/>(LLM)"]
  Refactor --> Verify["Verify<br/>(tests)"]
  Verify -->|"pass"| Next{"More in<br/>plan?"}
  Next -->|"yes"| Refactor
  Next -->|"no"| DONE(("Done"))
  Verify -->|"fail"| Replan["Replan"]
  Replan -->|"re-scan"| Detect

  style Plan fill:#4c6ef5,color:#fff,stroke-width:3px
  style DONE fill:#51cf66,color:#fff
  style Replan fill:#ff6b6b,color:#fff
```

**Replan triggers:**
1. Test failure after refactoring (rollback + rescan + replan)
2. New smells detected that weren't predicted by dependency rules
3. Predicted positive dependencies didn't materialize

---

# What we CAN evaluate today

Given current dataset coverage, the planner can be evaluated for these smell combinations:

| Scenario | Smells Involved | Refactoring | Dataset |
|----------|----------------|-------------|---------|
| Long Method cluster | Long Method + Complex Method + Cond. Complexity | Extract Method chain | SWE-Refactor (441) |
| God Class decomposition | God Class + Feature Envy | Move Method series | SWE-Refactor (410) |
| Class-level decomposition | God Class + Large Class | Extract Class | RMiner oracle (108 TP) |
| Method extraction cascade | Long Method resolves Dup Code | Extract Method | RMiner (1,033 TP) + SWE-Refactor |
| Compound refactoring | Long Method + Feature Envy | Extract And Move Method | SWE-Refactor (142) + RMiner (127 TP) |
| Inline dead code | Middle Man / unnecessary delegation | Inline Method | SWE-Refactor (71) + RMiner (120 TP) |
| Inheritance refactoring | Large Class with shared behavior | Extract Superclass | RMiner oracle (72 TP) |

These cover **6 of 8** detected smell types with ground truth.

---

# What we CANNOT evaluate today

| Missing Refactoring | Needed For | Impact |
|---------------------|-----------|--------|
| **Introduce Parameter Object** | Long Parameter List | No ground truth for parameter grouping |
| **Consolidate Conditional** | Duplicated Conditions | No ground truth for condition merging |
| **Decompose Conditional** | Conditional Complexity | Must use Extract Method as workaround |
| **Replace with Logger** | Print Statements | Trivial refactoring, low priority |

**Previously missing, now covered by RMiner oracle:**
- Extract Class (108 TP) — for God Class / Large Class decomposition
- Extract Superclass (80 total, 72 TP) — for inheritance-based decomposition
- Merge Parameter (28 TP) — partial coverage for Long Parameter List

---

# Proposed evaluation strategy

<div class="grid grid-cols-2 gap-8">
<div>

### Phase 1: Method-level planning
*Using existing datasets*

- Evaluate best-first vs greedy on **Long Method / Complex Method / Conditional Complexity** clusters
- Use SWE-Refactor Extract Method ground truth
- Measure: steps needed, new smells created, compile+test success rate

</div>
<div>

### Phase 2: Class-level planning
*Using RMiner oracle (108 Extract Class, 72 Extract Superclass TP)*

- Evaluate best-first vs greedy on **God Class / Large Class** scenarios
- Use RMiner oracle Extract Class ground truth
- Remaining gaps (Introduce Parameter Object, Consolidate Conditional): evaluate via SonarQube rescan

</div>
</div>

**Metrics for planning quality:**
- Plan efficiency: steps executed / smells resolved
- Negative dependency rate: new smells created / refactorings applied
- Backtrack rate: replans triggered / total executions

---

# End-to-end walkthrough: HttpJobExecutor


```mermaid
sequenceDiagram
  participant SQ as A1: SonarQube
  participant PL as Planner (BFS)
  participant LLM as A5: LLM Refactoring
  participant TEST as A6: Test Runner
  participant TG as A7: Test Generator

  SQ->>PL: {Long Method in process(), Feature Envy in isWriteMethod()}
  TG->>TEST: Generate tests for process() before any changes
  PL->>LLM: Plan step 1: Extract getConnectionInputStream() from process()
  LLM->>TEST: mvn test (shardingsphere-elasticjob)
  TEST-->>PL: All tests pass
  PL->>LLM: Plan step 2: Move isWriteMethod() to HttpParam
  LLM->>TEST: mvn test
  TEST-->>PL: All tests pass
  PL->>SQ: Rescan → no smells remaining
  Note over PL: Plan complete. 2 smells resolved in 2 steps.
```

---

# Real scenario: IntelliJ AbstractExternalFilter (RMiner)

<style scoped>
pre { font-size: 0.75em; line-height: 1.4; }
</style>

Commit `7a4dab88` applies **7 refactoring types** across 4 files. The agent must decide what to fix first.

<div class="grid grid-cols-2 gap-4">
<div>

**Smells detected in `AbstractExternalFilter.java`:**

| Smell | Location | Severity |
|-------|----------|----------|
| Long Method | `doBuildFromStream()` ~120 lines | HIGH |
| Complex Method | nested loops + conditionals in same method | HIGH |
| Data Clumps | `Trinity<Pattern, Pattern, Boolean>` used as return/param everywhere | MEDIUM |

**Multiple valid refactoring paths:**

- **Path A:** Extract Method on `doBuildFromStream()` first (reduces Long Method + Complex Method)
- **Path B:** Extract Class `ParseSettings` from `Trinity` first (resolves Data Clumps across hierarchy)
- **Path C:** Inline Method + Change Return Type first (type cleanup before structural changes)

</div>
<div>

**What the developers actually did (ground truth):**

```java
// BEFORE: opaque tuple
protected Trinity<Pattern, Pattern, Boolean>
    getParseSettings(@NotNull String url) {
  return Trinity.create(defined_pattern,
    defined_pattern2, false);
}
// Callers use settings.first, settings.second, settings.third

// AFTER: named class extracted
protected ParseSettings
    getParseSettings(@NotNull String url) {
  return new ParseSettings(defined_pattern,
    defined_pattern2, false);
}
// Callers use settings.startPattern, settings.endPattern

// New class (Extract Class):
static class ParseSettings {
  final Pattern startPattern;
  final Pattern endPattern;
  final boolean useDt;
  ParseSettings(Pattern start, Pattern end,
                boolean useDt) { ... }
}
```

Developers chose **Path B** — fixing Data Clumps first improved readability before tackling the Long Method.

</div>
</div>

---

# Real scenario: IntelliJ Debugger (RMiner)

<style scoped>
pre { font-size: 0.75em; line-height: 1.4; }
</style>

Commit `76552001` refactors **9 files** in the debugger subsystem. Agent faces a cross-file choice.

<div class="grid grid-cols-2 gap-4">
<div>

**Smells spanning multiple files:**

| Smell | Files Affected | Severity |
|-------|---------------|----------|
| Long Parameter List | `DebuggerSession`, `BreakpointManager`, `DebugProcessImpl` | HIGH |
| Primitive Obsession | same 3 files — `(Document, int)` instead of `XSourcePosition` | MEDIUM |
| Extract Method needed | `RemappedSourcePosition.getLine()`, `.getOffset()` | MEDIUM |

**The choice:** fix parameters first, or extract methods first?

- **Path A:** Merge Parameter `(Document, int)` → `XSourcePosition` across all callers first, then extract methods
- **Path B:** Extract Method `checkRemap()` in `RemappedSourcePosition` first, then fix parameter lists

</div>
<div>

**BEFORE — Long Parameter List:**
```java
// DebuggerSession.java
public void runToCursor(
    Document document,    // primitive pair
    int line,             // instead of position object
    final boolean ignoreBreakpoints) { ... }

// BreakpointManager.java
public RunToCursorBreakpoint addRunToCursorBreakpoint(
    Document document,    // same primitive pair
    int lineIndex,        // duplicated across files
    final boolean ignoreBreakpoints) { ... }
```

**AFTER — Merge Parameter (Introduce Parameter Object):**
```java
// DebuggerSession.java
public void runToCursor(
    @NotNull XSourcePosition position,
    final boolean ignoreBreakpoints) { ... }

// BreakpointManager.java
public RunToCursorBreakpoint addRunToCursorBreakpoint(
    @NotNull XSourcePosition position,
    final boolean ignoreBreakpoints) { ... }
```

Developers chose **Path A** — parameter cleanup first, because it touched the public API and cascaded to all callers.

</div>
</div>

---

# How the planner calculates the best first step


Concrete calculation for the **IntelliJ AbstractExternalFilter** scenario.

**Initial state S0:** {Long Method (H=3), Complex Method (H=3), Data Clumps (M=2)}

$$h(S) = \sum \text{severity}(s) - \alpha \cdot |\text{positive\_deps}| + \beta \cdot |\text{negative\_deps}|$$

| Action | Resolves | Creates | h(S') calculation | h(S') |
|--------|----------|---------|-------------------|-------|
| **Extract Method on doBuildFromStream()** | Long Method, Complex Method (+resolves: Dup Code, Comments via positive deps) | may create Long Param List | (2) - 2(2) + 1.5(1) | **-0.5** |
| **Extract Class ParseSettings** | Data Clumps (+resolves: Feature Envy via God Class positive deps) | may create Data Class | (3+3) - 2(1) + 1.5(1) | **5.5** |
| **Change Return Type only** | none directly | none | (3+3+2) - 0 + 0 | **8.0** |

**Planner picks:** Extract Method (h = -0.5, lowest), because it resolves the most smells with fewest negative effects.

But developers chose Extract Class first. Why? Because **cross-file impact** matters — the `Trinity` type appeared in the entire class hierarchy. The heuristic can be improved by adding a **scope multiplier** for changes that cascade across files:

$$h'(S) = h(S) - \gamma \cdot \text{files\_affected}(s)$$

---

# SWE-Refactor: compound refactoring choices

<style scoped>
pre { font-size: 0.75em; line-height: 1.4; }
</style>

SWE-Refactor contains **3 compound types** where the agent must decompose a multi-step refactoring.

<div class="grid grid-cols-2 gap-4">
<div>

**Extract And Move Method** — two operations in one:

The agent detects Long Method + Feature Envy in the same method. Two valid orderings:

| Order | Step 1 | Step 2 |
|-------|--------|--------|
| A | Extract Method (reduce length) | Move Method (fix envy) |
| B | Move Method (fix envy first) | Extract in target class |

The planner evaluates both:
- Order A: after extraction, the shorter method may no longer exhibit Feature Envy (positive cascade), making Move unnecessary
- Order B: moving the long method brings the smell into a new class, still needs extraction

**Planner chooses Order A** — Extract first has higher chance of resolving both smells in fewer steps.

</div>
<div>

**Move And Inline Method** — opposing forces:

The agent detects Feature Envy + unnecessary delegation. The method should be moved to where it belongs, then inlined into its only caller.

| Order | Step 1 | Step 2 | Risk |
|-------|--------|--------|------|
| A | Move Method | Inline Method | Safe: move first, then simplify |
| B | Inline Method | Move (now larger) | Risky: inlining may create Long Method in wrong class |

The planner's heuristic penalizes Order B:

```
Order A: h = severity(FE) + severity(MM)
         - alpha * 2 (both resolved) + 0
Order B: h = severity(FE) + severity(MM)
         - alpha * 1 + beta * 1 (Long Method risk)
```

Order A always wins because it avoids creating new smells.

</div>
</div>
