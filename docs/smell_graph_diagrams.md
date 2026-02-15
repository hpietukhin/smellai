# SmellAI System - Mermaid Diagrams

## 1. Smell Dependency Graph

```mermaid
graph TD
    subgraph "EXISTING SMELLS"
        GOD["🔴 GOD CLASS<br/>OrderProcessor.java<br/>1500 LOC, 45 methods<br/>PZ = 7"]
        LM["🟡 LONG METHOD<br/>processOrder():45<br/>85 lines<br/>PZ = 4"]
        FE["🟡 FEATURE ENVY<br/>updateInventory():120<br/>12 Customer accesses<br/>PZ = 2"]
        DC["🔴 DATA CLUMPS<br/>customer, addr, region<br/>passed together 8x<br/>PZ = 3"]
        LP["🟢 LONG PARAM LIST<br/>generate(8 args)<br/>ReportGenerator:88<br/>PZ = 1"]
    end

    subgraph "POTENTIAL NEW SMELLS"
        MM["⚠️ MIDDLE MAN<br/>ShippingManager<br/>only delegates"]
        DCL["⚠️ DATA CLASS<br/>ShippingConfig<br/>no behavior"]
        DCL2["⚠️ DATA CLASS<br/>ReportConfig<br/>only getters"]
        SG["⚠️ SPECULATIVE GEN.<br/>over-extracted<br/>tiny methods"]
    end

    GOD -->|"positive<br/>+2"| LM
    GOD -->|"positive<br/>+2"| FE
    GOD -->|"positive<br/>+2"| DC
    LM -->|"positive<br/>+2"| LP

    FE -.->|"negative<br/>-1"| MM
    FE -.->|"negative<br/>-1"| DCL
    LP -.->|"negative<br/>-1"| DCL2
    LM -.->|"negative<br/>-1"| SG

    style GOD fill:#ff6b6b,stroke:#333,stroke-width:2px,color:#fff
    style LM fill:#ffd93d,stroke:#333,stroke-width:2px
    style FE fill:#ffd93d,stroke:#333,stroke-width:2px
    style DC fill:#ff6b6b,stroke:#333,stroke-width:2px,color:#fff
    style LP fill:#6bcb77,stroke:#333,stroke-width:2px

    style MM fill:#ffe066,stroke:#f00,stroke-width:2px,stroke-dasharray: 5 5
    style DCL fill:#ffe066,stroke:#f00,stroke-width:2px,stroke-dasharray: 5 5
    style DCL2 fill:#ffe066,stroke:#f00,stroke-width:2px,stroke-dasharray: 5 5
    style SG fill:#ffe066,stroke:#f00,stroke-width:2px,stroke-dasharray: 5 5
```

---

## 2. Agent Workflow (Composite Mode)

```mermaid
flowchart TD
    START([START]) --> A0

    subgraph SETUP["A0: Setup"]
        A0[Clone repo<br/>Checkout commit<br/>Detect build system]
    end

    subgraph DETECT["A1: Smell Detection"]
        A1[SonarQube scan<br/>→ List of SmellEvents]
    end

    subgraph PRIORITIZE["A3: Prioritization"]
        A3[Build dependency graph<br/>Calculate PZ scores<br/>Sort by priority]
    end

    A0 --> A1 --> A3

    A3 --> LOOP

    subgraph LOOP["Refactoring Loop"]
        SELECT{More smells<br/>to process?}
        
        subgraph GENERATE["A5: Generate"]
            A5[LLM generates<br/>refactored code]
        end

        subgraph VERIFY["A6: Verify"]
            COMPILE[Compile project]
            TEST[Run tests]
            RESCAN[Re-scan smells]
            COMPARE[Compare before/after]
        end

        SELECT -->|Yes| A5
        A5 --> COMPILE
        COMPILE -->|Success| TEST
        COMPILE -->|Fail| RETRY1{Retry?}
        TEST -->|Pass| RESCAN
        TEST -->|Fail| RETRY2{Retry?}
        RESCAN --> COMPARE
        COMPARE --> LOG[Log RefactoringAttempt<br/>+ TestRun]
        LOG --> SELECT

        RETRY1 -->|"Yes (max 3)"| A5
        RETRY1 -->|No| SKIP1[Skip smell]
        RETRY2 -->|"Yes (max 3)"| A5
        RETRY2 -->|No| SKIP2[Skip smell]
        SKIP1 --> SELECT
        SKIP2 --> SELECT
    end

    SELECT -->|No| FINISH([END])

    style START fill:#4ecdc4,stroke:#333,stroke-width:2px
    style FINISH fill:#4ecdc4,stroke:#333,stroke-width:2px
    style A5 fill:#74b9ff,stroke:#333,stroke-width:2px
    style COMPILE fill:#a29bfe,stroke:#333,stroke-width:2px
    style TEST fill:#a29bfe,stroke:#333,stroke-width:2px
    style RESCAN fill:#a29bfe,stroke:#333,stroke-width:2px
```

---

## 3. PZ Score Calculation

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        SEV["Severity<br/>HIGH=3, MED=2, LOW=1"]
        POS["Positive Dependencies<br/>smells fixed together"]
        NEG["Negative Dependencies<br/>smells may be created"]
    end

    subgraph CALC["Calculation"]
        FORMULA["PZ = Severity<br/>+ Σ(positive × 2)<br/>- Σ(negative × 1)"]
    end

    subgraph EXAMPLE["Example: God Class"]
        EX["Severity = 3 (HIGH)<br/>+ Long Method (+2)<br/>+ Feature Envy (+2)<br/>- Lazy Class risk (-1)<br/>─────────────<br/>PZ = 6"]
    end

    SEV --> FORMULA
    POS --> FORMULA
    NEG --> FORMULA
    FORMULA --> EX

    style FORMULA fill:#74b9ff,stroke:#333,stroke-width:2px
    style EX fill:#00b894,stroke:#333,stroke-width:2px,color:#fff
```

---

## 4. Verification Pipeline

```mermaid
flowchart LR
    subgraph L1["Level 1: Compile"]
        C1[mvn compile<br/>gradle build]
        C1 -->|Fail| E1["Error feedback:<br/>'cannot find symbol'"]
    end

    subgraph L2["Level 2: Test"]
        T1[mvn test<br/>gradle test]
        T1 -->|Fail| E2["Test feedback:<br/>'expected X but was Y'"]
    end

    subgraph L3["Level 3: Re-scan"]
        S1[SonarQube scan]
        S1 --> CMP[Compare smells]
    end

    subgraph L4["Level 4: Metrics"]
        M1["resolved: 2<br/>created: 0<br/>net: +2"]
    end

    C1 -->|Success| T1
    T1 -->|Pass| S1
    CMP --> M1

    E1 -->|Retry| C1
    E2 -->|Retry| C1

    style C1 fill:#a29bfe,stroke:#333
    style T1 fill:#74b9ff,stroke:#333
    style S1 fill:#00b894,stroke:#333,color:#fff
    style M1 fill:#fdcb6e,stroke:#333
```

---

## 5. Negative Dependency Cases

```mermaid
flowchart TD
    subgraph CASE1["Case 1: Extract Method → Speculative Generality"]
        LM1["Long Method<br/>processOrder() 85 lines"]
        LM1 -->|"Extract<br/>Method"| AFTER1["processOrder()<br/>calls step1..step5()"]
        AFTER1 -.->|"creates"| SG1["⚠️ Speculative Generality<br/>5 tiny methods, used once"]
    end

    subgraph CASE2["Case 2: Move Method → Middle Man"]
        FE1["Feature Envy<br/>accesses Customer 12x"]
        FE1 -->|"Move<br/>Method"| AFTER2["Delegates to<br/>ShippingManager"]
        AFTER2 -.->|"creates"| MM1["⚠️ Middle Man<br/>only delegates, no logic"]
    end

    subgraph CASE3["Case 3: Introduce Parameter Object → Data Class"]
        LP1["Long Parameter List<br/>generate(8 args)"]
        LP1 -->|"Introduce<br/>Param Object"| AFTER3["generate(ReportConfig)"]
        AFTER3 -.->|"creates"| DC1["⚠️ Data Class<br/>only getters, no behavior"]
    end

    subgraph CASE4["Case 4: Extract Class → Lazy Class"]
        GC1["God Class<br/>1500 lines, 45 methods"]
        GC1 -->|"Extract<br/>Class"| AFTER4["6 small classes<br/>1-2 methods each"]
        AFTER4 -.->|"creates"| LC1["⚠️ Lazy Classes<br/>too small, no cohesion"]
    end

    style LM1 fill:#ffd93d
    style FE1 fill:#ffd93d
    style LP1 fill:#6bcb77
    style GC1 fill:#ff6b6b,color:#fff

    style SG1 fill:#ffe066,stroke:#f00,stroke-dasharray: 5 5
    style MM1 fill:#ffe066,stroke:#f00,stroke-dasharray: 5 5
    style DC1 fill:#ffe066,stroke:#f00,stroke-dasharray: 5 5
    style LC1 fill:#ffe066,stroke:#f00,stroke-dasharray: 5 5
```

---

## 6. Data Flow Architecture

```mermaid
flowchart TB
    subgraph AGENTS["Multi-Agent System (LangGraph)"]
        A0["A0: Setup"]
        A1["A1: Detect Smells"]
        A3["A3: Prioritize"]
        A5["A5: Generate Code"]
        A6["A6: Verify"]
    end

    subgraph TOOLS["External Tools"]
        SONAR["SonarQube"]
        MVN["Maven/Gradle"]
        GIT["Git"]
        LLM["LLM (GPT/Claude)"]
    end

    subgraph DB["Analytics DB (SQLite)"]
        SE["SmellEvent"]
        RA["RefactoringAttempt"]
        TR["TestRun"]
        TC["ToolCall"]
        SD["SmellDependency"]
    end

    subgraph VIS["Visualizer (NiceGUI)"]
        GRAPH["Dependency Graph"]
        TIMELINE["Execution Timeline"]
        METRICS["Quality Metrics"]
        DIFF["Code Diff Viewer"]
        TESTS["Test Results"]
    end

    A0 --> GIT
    A1 --> SONAR
    A3 --> SD
    A5 --> LLM
    A6 --> MVN

    A1 --> SE
    A6 --> RA
    A6 --> TR
    A5 --> TC
    A6 --> TC

    SE --> GRAPH
    RA --> TIMELINE
    TR --> TESTS
    RA --> DIFF
    SE --> METRICS
    SD --> GRAPH

    style AGENTS fill:#e8f4f8,stroke:#333
    style TOOLS fill:#f0e8f8,stroke:#333
    style DB fill:#f8f4e8,stroke:#333
    style VIS fill:#e8f8e8,stroke:#333
```

---

## 7. Demo Scenario Timeline

```mermaid
gantt
    title Demo Session: demo-2cf71c99
    dateFormat X
    axisFormat %s

    section Iteration 1
    Extract Method (Long Method)     :done, i1, 0, 10
    Compile                          :done, c1, 10, 12
    Run Tests (18/18 ✅)             :done, t1, 12, 15
    Re-scan Smells                   :done, s1, 15, 17

    section Iteration 2
    Move Method (Feature Envy)       :done, i2, 17, 27
    Compile                          :done, c2, 27, 29
    Run Tests (17/18 ❌)             :crit, t2, 29, 32
    Retry                            :done, r2, 32, 40
    Run Tests (18/18 ✅)             :done, t2b, 40, 43

    section Iteration 3
    Introduce Param Object           :done, i3, 43, 53
    Compile                          :done, c3, 53, 55
    Run Tests (18/18 ✅)             :done, t3, 55, 58
    Re-scan Smells                   :done, s3, 58, 60
```

---

## 8. Metrics Dashboard

```mermaid
pie title Smell Resolution
    "Resolved" : 4
    "Remaining" : 1
```

```mermaid
pie title Test Results (Iteration 2)
    "Passed" : 17
    "Failed" : 1
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 | HIGH severity smell |
| 🟡 | MEDIUM severity smell |
| 🟢 | LOW severity smell |
| ⚠️ | Potential new smell (risk) |
| `───▶` | Positive dependency (fixes together) |
| `- - ▶` | Negative dependency (may create) |
| ✅ | Success |
| ❌ | Failure |
