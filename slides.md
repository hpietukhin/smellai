---
theme: default
background: https://cover.sli.dev
title: Multi-Agent System for Code Smell Detection and Refactoring
info: |
  ## System design and implementation
  Multi-agent architecture for code smell detection, prioritization, and refactoring
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Refactoring software systems using AI  

Supervisor: doc. Ing. Ivan Polášek, PhD  
Autor: Havriil Pietukhin



---

# Related work 1: Markovič & Polášek (2016)

**Paper:** "Towards Rule Based Refactoring"

**Core contribution:** Framework for understanding code smell dependencies

**Key concepts:**
* **Positive dependencies:** Refactoring smell A resolves smell B
* **Negative dependencies:** Refactoring smell A introduces smell B

**Smell groups classification:**
* Bad Size (Large Class, Long Method, Long Parameter List)
* Bad Location (Feature Envy, Duplicated Code, Switch Statement)
* Bad Class Content (Data Class, Lazy Class)
* Bad Inheritance, Needless Part, Attribute Problem, Bad Communication

**Implementation:** Rule-based expert system using Jess (Java Expert System Shell)

**Application in this work:** Theoretical foundation for dependency graph and prioritization algorithm


---

<style scoped>
.slidev-layout {
  transform: scale(0.85);
  transform-origin: top left;
}
</style>

# Related work 2: RefactoringMiner 2.0 (2020)

**Paper:** Tsantalis et al., IEEE Transactions on Software Engineering 2020

**Core contribution:** Accurate refactoring detection tool without similarity thresholds

**Technical approach:**
* AST-based statement matching with specialized heuristics
* Detects 40 refactoring types (Extract Method, Move Class, Rename Variable, etc.)
* Processes git commits to identify applied refactorings

**Evaluation oracle:**
* 7,226 validated refactoring instances
* 536 commits from 185 open-source projects
* Multiple tools and expert validation

**Performance:**
* Precision: 99.6%, Recall: 94%
* 2.6× faster than competing tools

**Application in this work:** Ground truth dataset for evaluation and refactoring pattern analysis



---


<style scoped>
.slidev-layout {
  transform: scale(0.85);
  transform-origin: top left;
}
</style>

# Related work 3: Code Foundation Models Survey (2025)

**Paper:** "From Code Foundation Models to Agents and Applications" - BUAA-SKLCCSE et al.

**Scope:** Comprehensive survey of LLMs for code intelligence

**Key topics:**
* **Evolution:** GPT-4, Claude 4, DeepSeek-Coder, Qwen-Coder, StarCoder
* **Training pipeline:** Data curation → Pre-training → Supervised fine-tuning → RL
* **Agentic systems:** Autonomous coding agents (SWE-Bench, multi-step problem solving)
* **Evaluation:** Statement/function/class-level tasks, repository-level benchmarks

**Performance trajectory:** From <10% to >95% on HumanEval benchmark

---

<style scoped>
.slidev-layout {
  transform: scale(0.85);
  transform-origin: top left;
}
</style>

## Autonomous Agents and Refactoring

**Repository-Level Development Agents:**

* **CodePlan** - Structured planning for compilation, execution, and debugging
  - Extends debugging loop with automatic tool calls and explicit execution reasoning
  - Multi-round verbalized feedback substantially improves robustness
  - Coordinates compile-test-debug cycles with strategic decision making

* **OpenHands** - Event-stream architecture for agent-environment interaction
  - Docker sandbox ensures secure code execution
  - Diverse execution environments: bash terminal, Jupyter IPython, playwright browser
  - Enhances refinement by diversifying repair strategies instead of repeating fixes
  - Multi-agent collaboration with specialized sub-agents

---

<style scoped>
.slidev-layout {
  transform: scale(0.85);
  transform-origin: top left;
}
</style>


## Tool-Integrated and Memory-Enhanced Agents

**HyperAgent** - Four-role team architecture
* Planner: Strategic task decomposition
* Navigator: Repository and codebase navigation
* Editor: Precision code modifications
* Executor: Test execution and validation
* State-of-the-art performance on RepoExec by allocating different models to different functions

**CodeAct** - Extends actions to executable Python code
* Integrated Python interpreter for dynamic code generation
* Multi-turn interaction with control flow features
* Self-debugging capabilities and intermediate result storage

**OpenCode** - Controllable and extensible intelligent programming environment
* Plan Agent: Analysis and planning
* Build Agent: Executing modifications
* General Agent: Auxiliary queries
* Decouples reasoning from action while ensuring safety

---

<style scoped>
.slidev-layout {
  transform: scale(0.85);
  transform-origin: top left;
}
</style>


## Dependency-Aware and RAG Systems

**Commit0** - Interactive environment for complete library development
* Test-driven development workflow
* Highlights difficulty of maintaining cross-file consistency
* Agents must understand import graphs and API relations

**RepoCoder & CodeChain** - Explicit repository dependency modeling
* Generate modules in dependency-sorted order
* Retrieve relevant snippets for each component
* Shortens context length by orders of magnitude while retaining coherence

 Scaling to repository level requires not just larger context windows but persistent memory and tool-aware interaction

---

<style scoped>
.slidev-layout {
  transform: scale(0.85);
  transform-origin: top left;
}
</style>

## Future Trends

**From General to Specialized Code Intelligence:**
* Domain-specific optimization yields substantial gains
* Dedicated coding assistants outperform general LLMs on complex tasks

**Agentic Training and Complex Scenario Mastery:**
* Models trained to operate autonomously across multi-step scenarios
* Not just writing code, but understanding project contexts, navigating codebases
* Execute iterative debugging, collaborate with developers through extended interactions

**Scaling Laws and Scientific Development:**
* Principled understanding of how performance scales with parameters, data, compute
* Data-driven optimization of scaling trade-offs unique to programming domains
* Mixture-of-experts architectures optimized for code tasks

**Application in this work:** 
Context for LLM-based refactoring agents, methodologies for evaluating multi-agent systems, combining static analysis with agentic reasoning, repository-level understanding of code smell dependencies


---

# Our approach

A multi-agent architecture where specialized AI agents collaborate to analyze code quality, detect smells, and execute refactorings.

**Key components:**

* 🔍 SonarQube for initial code smell detection using rule-based static analysis
* 🤖 LLM-based agents built using LangGraph framework for intelligent analysis
* 📊 MLflow for experiment tracking, evaluation, and dataset management
* 🔗 Dependency analysis engine modeling positive and negative smell relationships
* 📈 Priority calculation system for optimal refactoring sequences


---

# Architecture principles

**Pipeline approach** where each agent has a specific responsibility

* Modular development - agents can be improved independently
* Shared state communication through LangGraph
* Continuous validation and verification
* Comprehensive experiment tracking

**Six specialized agents working in sequence:**
1. Code smell detection
2. Test existence verification
3. Test generation (placeholder)
4. Smell prioritization
5. Refactoring execution
6. Test execution and verification


---

# Agent 1: Code smell detection

Performs static analysis using SonarQube to identify code smells in Java projects.

**Detected smells (mapped to SonarQube rules):**
* Long Method (java:S138) - Methods exceeding 60 lines
* Complex Method (java:S1541) - High cyclomatic complexity
* Conditional Complexity (java:S1067) - Deeply nested conditionals
* Long Parameter List (java:S107) - More than 7 parameters
* God Class (java:S1200) - Excessive coupling
* Large Class (java:S110) - Excessive lines of code
* Duplicated Conditions (java:S1871) - Identical branches
* Print Statements (java:S106) - Direct use of System.out/err

**Output:** Severity levels (HIGH, MEDIUM, LOW) with line-level locations


---

# Agent 2: Test existence verification

Verifies whether unit tests exist for methods containing code smells.

**Capabilities:**
* 🔧 Detects build system (Maven or Gradle)
* ✅ Executes test suite using detected build system
* 📝 Analyzes test results (passed/failed)
* 🎯 Maps test coverage to specific smelly methods
* 🚨 Reports gaps in test coverage

**Implementation:** LangGraph-based agent with LLM-powered tools for build configuration detection and test result parsing

Location: `agents/java_test/agent.py`


---

# Agent 3: Test generation

**Current status:** Placeholder for extensibility

While test generation is important, it is not the primary focus of this research.

**Research emphasis:**
* Code smell detection
* Prioritization
* Refactoring

The architecture includes this agent to demonstrate the extensibility of the multi-agent system for future work.


---

# Agent 4: Smell prioritization

Determines the optimal sequence for refactoring detected smells.

**Critical component** implementing theoretical framework based on smell dependencies and impact analysis.


---

# Theoretical foundation

Based on the work of Markovič and Polášek on code smell dependencies.

**Key insight:** Code smells do not exist in isolation

**Positive dependencies (PZ):**
Refactoring smell A may resolve smell B

*Example:* Extracting a Long Method often resolves:
- Duplicated Code
- Switch Statements
- Complex Method

**Negative dependencies (NZ):**
Refactoring smell A may introduce smell B

*Example:* Applying "Introduce Parameter Object" to fix Long Parameter List may create a Data Class


---

# Priority calculation algorithm

**Priority score (PZ) for each smell instance:**

$$PZ_i = Severity_i + \sum_{j \in \text{PositiveDeps}(i)} w_{\text{impact}}$$

Where:
- $Severity_i$ is the base severity score: HIGH = 3, MEDIUM = 2, LOW = 1
- $\text{PositiveDeps}(i)$ is the set of smells that refactoring smell $i$ would help resolve
- $w_{\text{impact}} = 2$ is the weight assigned to each positive impact relationship

**Strategy:** Greedy algorithm that iteratively selects the smell with maximum PZ, removes it from the graph, and recalculates scores.

**Result:** Prioritizes smells with both high severity and high potential to resolve other smells.


---

# Dependency rules (1/2)

**Long Method / Complex Method / Conditional Complexity**
* ✅ Positive: May resolve Switch Statement, Feature Envy, Duplicated Code, Divergent Change, Comments, Long Parameter List
* ❌ Negative: May create Long Method, Long Parameter List

**Long Parameter List**
* ✅ Positive: May resolve Long Parameter List, Data Clumps
* ❌ Negative: May create Data Class

**Large Class / God Class**
* ✅ Positive: May resolve Data Clumps, Feature Envy, Bad Class Content
* ❌ Negative: May create Long Method, Data Class, Inappropriate Intimacy, Message Chains


---

# Dependency rules (2/2)

**Duplicated Conditions**
* ✅ Positive: May resolve Divergent Change, Shotgun Surgery
* ❌ Negative: May create Large Class, Bad Inheritance

**Print Statements**
* ✅ Positive: May resolve Needless Part
* ❌ Negative: May create Data Class, Lazy Class

**Implementation:** Centralized rule base in `agents/dependency_analysis/agent.py`


---

# Visualization: Priority graph

<img src="/smell_priority_graph.png" class="h-120" />

**Nodes:** Smell instances sized by PZ score, different shapes for different smell types
**Green edges:** Positive dependencies (solving relationships)
**Red edges:** Negative dependencies (risk of creating new smells)


---

# Graph extensibility

The dependency graph is designed to accept additional dimensions and attributes:

**Current attributes:**
* 📊 Node size - Priority score (PZ = Severity + Positive Impact)
* 🔷 Node shape - Smell type visualization
* 🎨 Node color - Severity level (High=red, Medium=orange, Low=green)
* ➡️ Edge color - Dependency type (Green=positive, Red=negative)

**Future enrichment possibilities:**
* ⚖️ Smell weight/priority - Additional weighting based on code metrics (WMC, CBO, RFC, LCOM)
* 🏷️ Edge labels - Specific refactoring techniques that enable dependencies
* 📍 Node clustering - Group smells by file/package/module
* ⏱️ Temporal dimension - Track how smells evolve over time
* 🎯 Confidence scores - Indicate certainty of dependency relationships


---

# Visualization: File-level dependencies

<img src="/smell_deps_OrderProcessor.png" class="h-110" />

File-level view showing how refactoring one smell (e.g., Long Method) can cascade to resolve multiple related smells (Duplicated Code, Switch Statement, Complex Method).


---

# Visualization: ReportGenerator dependencies

<img src="/smell_deps_ReportGenerator.png" class="h-110" />

Example of God Class with multiple interconnected smells and complex dependency relationships.


---

# Agent 5: Refactoring execution

Performs actual code refactoring based on the prioritized smell sequence.

**Refactoring techniques applied:**
* Long Method → Extract Method refactoring
* Large Class/God Class → Extract Class refactoring
* Long Parameter List → Introduce Parameter Object
* Duplicated Code → Extract Method and remove duplication
* Complex Method → Decompose conditional logic

**Implementation:**
* Uses LLM capabilities to understand code context
* Generates semantically correct refactorings
* Preserves code functionality while improving structure
* Generates intermediate code versions after each step


---

# Agent 6: Test execution and verification

Validates that refactorings preserve program behavior by executing tests.

**Responsibilities:**
* ✅ Runs complete test suite after each refactoring
* 🚨 Reports test failures and errors with detailed information
* ↩️ Enables rollback if tests fail
* 🛡️ Ensures code quality improvements do not introduce regressions

**Implementation:** Reuses test execution capabilities from Agent 2, but focuses on verification rather than coverage analysis.


---

# Evaluation framework

**MLflow for comprehensive experiment tracking**

* 📦 Dataset management: Creating and versioning datasets from RefactoringMiner data
* 📊 Experiment tracking: Recording model configurations, parameters, and metrics
* 🔄 Result comparison: Comparing different approaches (vanilla LLM, RAG-enhanced, hybrid)
* 🔁 Reproducibility: Ensuring experiments can be reproduced with exact configurations

**Implementation:**
* `scripts/create_rminer_dataset.py` - Dataset creation
* `scripts/manage_datasets.py` - Dataset inspection
* `agent_workflows/rminer_eval.py` - Evaluation pipeline


---

# Ground truth dataset

**RefactoringMiner 2.0 dataset** (Tsantalis et al., IEEE TSE 2022)

**Properties:**
* 📦 Size: 547 commits from 188 open-source projects
* 📅 Time range: June 8, 2015 - August 7, 2015
* 📝 Content: Real refactorings by developers
  - Before and after code versions
  - Refactoring type annotations (Extract Method, Move Method, etc.)
  - Detailed location information (class, method, line numbers)
  - Diff hunks showing exact code changes

**Purpose:** Ground truth for evaluating the refactoring mapping agent


---

# Test data: Smell co-occurrence

Manually crafted test cases demonstrating smell co-occurrence.

Location: `tests/test_data/smell_cooccurrence/`

**Test files:**
* `OrderProcessor.java` - Positive dependencies in "bad size" category
* `CustomerDataService.java` - Negative dependencies (trade-offs)
* `ReportGenerator.java` - God Class with multiple interconnected smells
* `PaymentValidator.java` - Duplicated conditions and dependencies
* `ConfigurationManager.java` - Complex interplay of multiple smells

Each file annotated in `smells_manifest.json` with locations, severities, and descriptions.


---

# Code metrics and impact analysis

**Standard object-oriented metrics:**

* WMC (Weighted Methods per Class) - Sum of cyclomatic complexities
* CBO (Coupling Between Objects) - Number of coupled classes
* RFC (Response For a Class) - Number of potentially executed methods
* LCOM (Lack of Cohesion of Methods) - Method/attribute relationships

**Validation examples:**
* Extracting a Long Method should reduce WMC
* Splitting a God Class should reduce CBO and RFC
* Resolving Feature Envy should improve LCOM


---

# Dependency validation

**Positive dependencies validation:**

After refactoring a smell, check if predicted positive dependencies were realized:
* If Long Method is refactored → Does Duplicated Code decrease?
* If Large Class is split → Does Feature Envy reduce?

**Negative dependencies monitoring:**

Monitor if refactorings introduce predicted negative dependencies:
* After "Introduce Parameter Object" → Is a Data Class created?
* After splitting Large Class → Do Long Methods appear in extracted classes?

**Feedback loop:** Enables the system to learn which dependency rules are accurate and adjust priorities accordingly.


---

# Implementation technologies

**Core stack:**

* 🐍 Python 3.11+ - Core implementation language
* 🔗 LangGraph - Multi-agent workflows with state management
* 🦜 LangChain - LLM orchestration and tool integration
* ☁️ LiteLLM - Unified interface for multiple LLM providers (OpenAI, Anthropic, Google)
* 📊 MLflow - Experiment tracking and dataset management
* 🔍 SonarQube - Static code analysis engine
* 📈 NetworkX - Graph analysis and dependency modeling
* 📉 Matplotlib - Visualization of dependency graphs
* ✅ Pydantic - Data validation and structured outputs


---

# Workflow execution

Complete analysis and refactoring workflow:

1. **Initial scan:** Agent 1 analyzes commit using SonarQube, identifies smells
2. **Test verification:** Agent 2 checks test coverage for affected methods
3. **Dependency analysis:** Agent 4 analyzes smell dependencies and calculates PZ scores
4. **Priority determination:** Agent 4 generates refactoring sequence (max PZ strategy)
5. **Visualization:** System generates dependency graphs and priority visualizations
6. **Refactoring execution:** Agent 5 applies refactorings in priority order
7. **Verification:** Agent 6 runs tests after each refactoring
8. **Metrics evaluation:** System calculates code quality metrics before and after
9. **Result logging:** MLflow records all steps, decisions, and outcomes


---

# Workflow benefits

**Systematic approach ensures:**

* ✅ Refactorings applied in optimal order
* 🔄 Continuous validation at each step
* 📊 Detailed tracking for research analysis
* 🛡️ Program behavior preservation
* 📈 Measurable code quality improvements
* 🔬 Reproducible experiments


---

# Example: Priority sequence

<div class="text-sm">

| Order | PZ | Smell Type | Location |
|-------|-----|------------|----------|
| 1 | 7 | God Class | ReportGenerator.java |
| 2 | 6 | Long Method | OrderProcessor.java:processOrder() |
| 3 | 6 | Complex Method | OrderProcessor.java:processOrder() |
| 4 | 6 | Conditional Complexity | OrderProcessor.java:processOrder() |
| 5 | 6 | Large Class | ReportGenerator.java |
| 6 | 6 | Long Method | ConfigurationManager.java:loadConfiguration() |

</div>

**Observation:** Highest priority smells have both high severity AND high potential to resolve other smells.


---

# Summary

Multi-agent system for code smell detection, prioritization, and refactoring

**Key contributions:**
* Multi-agent architecture with 6 specialized agents
* Dependency-based prioritization algorithm
* Visualization of smell relationships
* Comprehensive evaluation framework
* Integration of LLMs with static analysis

**Documentation:** docs/SYSTEM_DESIGN_SUMMARY.md

**Workflow scripts:** workflows/
