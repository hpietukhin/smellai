# Architecture Document: LLM-Based Code Smell Detection

**Version**: 1.0  
**Date**: 2025-10-18  
**Status**: Draft  
**Project**: Master's Thesis - Code Smell Detection Using LLMs

## 1. System Overview

### 1.1 Purpose
CLI tool for evaluating LLM-based code smell detection against ground truth annotations from the DACOS dataset. The system compares LLM detection accuracy using rubric-based evaluation with an LLM-as-judge approach.

### 1.2 Phase 1 Scope (2-month prototype)
- Single LangGraph agent: LLM-as-judge for smell detection evaluation
- MLflow for tracing, experiment tracking, and evaluation metrics
- Ground truth from MySQL DACOS dataset
- SonarQube baseline (separate process, not integrated in eval pipeline)
- Focus: Java code smell detection quality assessment

### 1.3 Out of Scope (Phase 1)
- Multi-agent workflows (coordinator, test generation, refactoring)
- Automated refactoring
- Real-time SonarQube integration in eval pipeline
- Advanced MLflow features (model registry, deployment)

## 2. System Components

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Tracking                          │
│  (experiment tracking, tracing, metrics logging)            │
└──────────────────────┬──────────────────────────────────────┘
                       │ (automatic tracing)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           LLM-as-Judge Agent                        │   │
│  │  - Receives: file content + LLM detections +        │   │
│  │             ground truth                            │   │
│  │  - Evaluates: precision, recall, smell-level scores │   │
│  │  - Returns: structured evaluation result            │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────┬─────────────────────────────────┬───────────────┘
            │                                 │
            ▼                                 ▼
┌───────────────────────┐         ┌──────────────────────────┐
│   MySQL Connector     │         │  Git Operations          │
│  - Read DACOS samples │         │  - Sparse/shallow clone  │
│  - Fetch ground truth │         │  - Checkout commit SHA   │
│  - Query by filters   │         │  - Cleanup after eval    │
└───────────────────────┘         └──────────────────────────┘
            │                                 │
            ▼                                 ▼
┌───────────────────────┐         ┌──────────────────────────┐
│  DACOS MySQL DB       │         │  Git Repositories        │
│  (tagman5.sample)     │         │  (cloned at commit SHA)  │
└───────────────────────┘         └──────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Separate Baseline Process                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │  SonarQube (Docker)                                │    │
│  │  - Bash script: clone → scan → export JSON        │    │
│  │  - Runs independently for baseline comparison      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 Supporting Components                       │
│  ┌─────────────────────┐    ┌────────────────────────┐    │
│  │  DeepLake Vector DB │    │  LLM Detector Module   │    │
│  │  (smell knowledge)  │    │  (RAG-based detection) │    │
│  └─────────────────────┘    └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Descriptions

#### 2.2.1 LangGraph Pipeline
**Purpose**: Core evaluation workflow
**Implementation**: Single-agent graph with linear flow
**Tracing**: Automatic MLflow tracing via `mlflow.langchain.autolog()`
**State Schema**:
```python
class EvaluationState(TypedDict):
    sample_id: int
    file_path: str
    file_content: str
    ground_truth: List[SmellAnnotation]
    llm_detections: List[SmellDetection]
    evaluation_result: EvaluationResult
    error: Optional[str]
```

**Agent: LLM-as-Judge**
- Uses rubric-based evaluation (EXCELLENT/GOOD/ACCEPTABLE/POOR/INCORRECT)
- Compares LLM detections against ground truth
- Calculates precision, recall, per-smell scores
- Returns structured evaluation result

**Evaluation Matching Criteria** (approximate location):
- **EXCELLENT**: Correct smell type + exact file + correct class/method name
- **GOOD**: Correct smell type + exact file + approximate location (same class, different method)
- **ACCEPTABLE**: Correct smell type + correct file (location unclear)
- **POOR**: Wrong smell type but correct general area
- **INCORRECT**: Wrong smell type and wrong location

**Note**: Since ground truth lacks line numbers, judge evaluates based on:
- File path match (from sample.path_to_file)
- Class name match (from class_metrics.type_name or method_metrics.type_name)
- Method name match if applicable (from method_metrics.method_name)

### 2.2.3 MySQL Connector
**Purpose**: DACOS dataset access  
**Database**: MySQL (localhost, database: dacos)  
**Dataset Reference**: Nandani, H., Saad, M., & Sharma, T. (2023). DACOS—A Manually Annotated Dataset of Code Smells. MSR 2023. [arXiv:2303.08729]  

**Schema Overview**:
```sql
-- Core tables
tagman5.sample          -- Code samples with metadata
tagman5.annotation      -- Ground truth smell labels (binary flags)
tagman5.smell           -- Smell type definitions
tagman5.class_metrics   -- Designite class-level metrics
tagman5.method_metrics  -- Designite method-level metrics

-- Key relationships
sample.id → annotation.sample_id (1:1)
sample.designite_id → class_metrics.id OR method_metrics.id (based on sample.is_class)
sample.smells → smell.id (lookup)
```

**sample table**:
```sql
id: bigint PK
designite_id: bigint (FK to metrics tables)
has_smell: bit(1)
is_class: bit(1)
path_to_file: varchar(255)
project_name: varchar(255)
sample_constraints: int
smells: varchar(255)  -- smell IDs or names
```

**annotation table** (ground truth):
```sql
id: bigint PK
sample_id: bigint FK
is_smell: bit(1)      -- general smell flag
iscm: bit(1)          -- Complex Method
isim: bit(1)          -- Insufficient Modularization
islp: bit(1)          -- Long Parameter List
isma: bit(1)          -- Multifaceted Abstraction
```

**Note**: DACOS tracks 4 specific smell types via binary flags: Complex Method (iscm), Insufficient Modularization (isim), Long Parameter List (islp), and Multifaceted Abstraction (isma). For Phase 1, evaluation focuses on these 4 types. Additional smell types may exist in the `smell` table but won't have ground truth annotations.

**smell table** (type definitions):
```sql
id: bigint PK
name: varchar(255)
description: varchar(255)
is_design_smell: bit(1)
```

**Operations**:
- Query samples by filters (smell type, project, is_class)
- Join with annotation table for ground truth labels
- Join with smell table for type names
- Join with metrics tables for context (LOC, CC, etc.)
- Read-only access (no writes)

#### 2.2.4 Git Operations
**Purpose**: Source code retrieval at specific commits  
**Strategy**: Sparse/shallow clones to save time and space
```bash
# Sparse-checkout for specific file paths
git clone --depth 1 --filter=blob:none --sparse <repo_url>
git checkout <commit_sha>
git sparse-checkout set <file_path>
```

**Cleanup**: Remove cloned repos after evaluation batch completes

#### 2.2.5 LLM Detector Module
**Purpose**: Code smell detection using LLM with RAG
**Components**:
- DeepLake vector database (smell knowledge from smells repo, persistent local storage)
- LiteLLM with Cerebras provider (detection LLM)
- Retrieval-augmented generation for smell definitions
- Structured output (Pydantic models)

**Detection Process**:
1. Retrieve relevant smell documentation from vector DB
2. Analyze file content with RAG context
3. Return structured detections (type, location, severity, description)

**Note**: DeepLake configured to use in-memory storage (`mem://deeplake/smells`) by default for development speed. Persistent storage at `./data/deeplake/smells` available for production use.

#### 2.2.6 SonarQube Baseline (Separate Process)
**Purpose**: Classical tool baseline for comparison
**Deployment**: Docker container (sonarqube:10.6.0-community)
**Implementation**: Python script (`infra/sonarqube/baseline_scan.py`)
**Configuration**:
- Port: 9000
- Credentials: environment variables (SONAR_URL, SONAR_TOKEN from .env)
- Quality profile: default Java profile
- Scanner: Docker image `sonarsource/sonar-scanner-cli`

**Python Script Flow** (`baseline_scan.py`):
```python
1. Clone repository at commit before cutoff date (2024-01-01)
2. Run sonar-scanner via Docker container
3. Poll SonarQube API for analysis completion
4. Fetch issues via REST API (/api/issues/search)
5. Normalize issues to SmellDetection format
6. Save baseline JSON artifacts to eval_results/sonarqube_baseline/
```

**Rule Mapping**:
- java:S1541 → Complex Method
- java:S138 → Long Method
- java:S107 → Long Parameter List
- java:S1067 → Conditional Complexity
- java:S1200 → God Class
- java:S110 → Large Class

**Output**: JSON files per project with normalized smell detections for baseline comparison

## 3. Data Flow

### 3.1 End-to-End Evaluation Flow

```
[Evaluation Script/CLI]
    ↓ (invoke for each sample)
[LangGraph Pipeline Entry]
    ↓ (automatically traced by MLflow)
┌─────────────────────────────────────────┐
│ 1. Fetch Sample Metadata                │
│    - Query MySQL for sample record      │
│    - Extract: project_name, path,       │
│      commit SHA, ground truth smells    │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 2. Clone Repository                     │
│    - Sparse checkout at commit SHA      │
│    - Extract file content               │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 3. Run LLM Detection                    │
│    - Load smell knowledge (DeepLake)    │
│    - RAG-enhanced detection             │
│    - Return structured detections       │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 4. LLM-as-Judge Evaluation              │
│    - Compare detections vs ground truth │
│    - Apply rubric scoring               │
│    - Calculate precision/recall         │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 5. Return Evaluation Result             │
│    - Structured JSON output             │
│    - Per-smell scores                   │
│    - Aggregate metrics                  │
│    - Logged to MLflow                   │
└────────────┬────────────────────────────┘
             ↓
[MLflow Tracking]
    ↓ (store trace + metrics)
[MLflow UI / Analysis]
```

### 3.2 State Transitions (LangGraph)

```
START
  ↓
fetch_sample_node
  ↓ (sample metadata)
clone_repo_node
  ↓ (file content)
detect_smells_node
  ↓ (LLM detections)
judge_evaluation_node
  ↓ (evaluation result)
END
```

**Error Handling**: Each node can transition to error state, which outputs partial result with error message.

## 4. Key Entities

### 4.1 Domain Models

```python
# Core entities (Pydantic models)

class SmellAnnotation(BaseModel):
    """Ground truth from DACOS annotation table"""
    smell_type: str  # e.g., "Complex Method", "Long Parameter List"
    is_present: bool  # from annotation flags (iscm, isim, islp, isma)
    
    # Location info (derived from metrics tables or Designite output)
    package_name: Optional[str]
    type_name: Optional[str]  # class name
    method_name: Optional[str]  # if method-level smell
    
    # Metrics context (from class_metrics or method_metrics)
    loc: Optional[int]  # lines of code
    cc: Optional[int]  # cyclomatic complexity (methods only)
    pc: Optional[int]  # parameter count (methods only)

class SmellDetection(BaseModel):
    """LLM detection result"""
    smell_type: str
    location: str
    description: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    refactoring_suggestion: str
    confidence: Optional[float]

class SmellEvaluation(BaseModel):
    """Per-smell evaluation by judge"""
    detected_smell: str
    location: str
    ground_truth_match: Optional[str]
    score: Literal["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR", "INCORRECT"]
    justification: str

class EvaluationResult(BaseModel):
    """Final evaluation output"""
    sample_id: int
    file_path: str
    overall_score: float  # 0-5 scale
    precision: float
    recall: float
    f1_score: float
    evaluations: List[SmellEvaluation]
    summary: str
    timestamp: str
    git_sha: str  # commit SHA analyzed
    
class DACOSSample(BaseModel):
    """Database record with joined annotation data"""
    # From sample table
    id: int
    designite_id: int
    has_smell: bool
    is_class: bool
    path_to_file: str
    project_name: str
    sample_constraints: int
    smells: str  # smell type identifier
    
    # From annotation table (ground truth)
    iscm: bool  # Complex Method
    isim: bool  # Insufficient Modularization
    islp: bool  # Long Parameter List
    isma: bool  # Multifaceted Abstraction
    
    # From smell table
    smell_name: str
    smell_description: str
    
    # Derived/external fields (not in DB)
    repo_url: str  # derived from project_name
    commit_sha: str  # source TBD: external mapping or sample_constraints
    
    @property
    def ground_truth_smells(self) -> List[str]:
        """Extract active smell flags as list"""
        smells = []
        if self.iscm:
            smells.append("Complex Method")
        if self.isim:
            smells.append("Insufficient Modularization")
        if self.islp:
            smells.append("Long Parameter List")
        if self.isma:
            smells.append("Multifaceted Abstraction")
        return smells
```

### 4.2 Configuration Models

```python
**Note**: Pipeline configuration is managed via environment variables (.env file) and function parameters. No dedicated configuration module exists in current implementation. Key configuration points:

- **LLM Model**: `cerebras/llama3.1-8b` (via LiteLLM)
- **Embedding Model**: `models/text-embedding-004` (Google Generative AI)
- **Vector DB Path**: `mem://deeplake/smells` (in-memory, default) or `./data/deeplake/smells` (persistent)
- **MySQL Connection**: MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD (from .env)
- **MLflow**: MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME (from .env, defaults to `./mlruns` and `code-smell-evaluation`)
- **SonarQube**: SONAR_URL, SONAR_TOKEN (from .env)
- **API Keys**: CEREBRAS_API_KEY, GOOGLE_API_KEY (from .env)
```

## 5. Integration Points

### 5.1 Evaluation Entry Point

**Interface**: Direct function call to LangGraph pipeline

**Entry Point**:
```python
def run_evaluation(sample_id: int) -> Dict[str, Any]:
    """
    Entry point for running evaluations.

    Args:
        sample_id: DACOS sample ID

    Returns:
        Evaluation result as dict (serializable to JSON)

    Notes:
        - Automatically traced by MLflow
        - Can be called from CLI, notebooks, or batch scripts
    """
    pass
```

**Usage Examples**:
```python
# Single evaluation
result = run_evaluation(sample_id=12345)

# Batch evaluation with MLflow tracking
for sample_id in sample_ids:
    result = run_evaluation(sample_id)
    # Each run automatically traced in MLflow

# From CLI
python -m src.pipelines.evaluation_pipeline 12345
```

### 5.2 MySQL Schema Mapping

**Sample Query**:
```sql
SELECT 
    s.id,
    s.designite_id,
    s.has_smell,
    s.is_class,
    s.path_to_file,
    s.project_name,
    s.sample_constraints,
    s.smells,
    a.iscm,
    a.isim,
    a.islp,
    a.isma,
    sm.name AS smell_name,
    sm.description AS smell_description
FROM tagman5.sample s
LEFT JOIN tagman5.annotation a ON s.id = a.sample_id
LEFT JOIN tagman5.smell sm ON s.smells = sm.id
WHERE s.has_smell = 1
  AND s.project_name = 'alibaba_arthas'
LIMIT 100;
```

**Ground Truth Extraction**:
- Annotation flags (iscm, isim, islp, isma) indicate which smells are present
- Multiple flags can be true for a single sample
- Smell type names from `smell.name` column

### 5.3 LangGraph State Management

**State Persistence**: In-memory only (no checkpointing for Phase 1)

**State Updates**:
```python
# Node signature
def fetch_sample_node(state: EvaluationState) -> EvaluationState:
    # Fetch from MySQL
    sample = db.query_sample(state['sample_id'])
    return {
        **state,
        'file_path': sample.path_to_file,
        'ground_truth': sample.smell_annotations,
    }

def clone_repo_node(state: EvaluationState) -> EvaluationState:
    # Clone and read file
    content = git_ops.clone_and_read(
        repo_url=extract_repo_url(state['file_path']),
        commit_sha=state['commit_sha'],
        file_path=state['file_path']
    )
    return {
        **state,
        'file_content': content,
    }
```

### 5.4 MLflow Integration

**Purpose**: Automatic tracing and experiment tracking

**Usage Pattern**:
```python
# Initialize once at module load
import mlflow

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("code-smell-evaluation")
mlflow.langchain.autolog()

# Automatic tracing on every run_evaluation() call
result = run_evaluation(sample_id=12345)

# Optional: Log additional metrics
mlflow.log_metrics({
    "precision": result["precision"],
    "recall": result["recall"],
    "f1_score": result["f1_score"]
})

# View in UI
# mlflow ui --backend-store-uri ./mlruns
```

## 6. Non-Functional Requirements

### 6.1 Performance
- **Target**: Evaluate 100 samples in <30 minutes (sequential)
- **Bottleneck**: Git clone operations
- **Mitigation**: Sparse/shallow clones, cleanup after batches

### 6.2 Reproducibility
**Constitution Compliance**:
- Log Git SHA for each analyzed commit
- Log LLM model version, temperature, seeds
- Store full prompts and system messages
- Record MLflow experiment config
- Version LangGraph graph definition
- MLflow traces for complete execution history

**Logging Requirements**:
```json
{
  "run_metadata": {
    "git_sha": "abc123...",
    "timestamp": "2025-10-18T10:30:00Z",
    "pipeline_version": "1.0.0",
    "langgraph_version": "0.2.x",
    "mlflow_version": "3.x",
    "mlflow_run_id": "abc123...",
    "llm_model": "cerebras/llama3.1-8b",
    "llm_provider": "LiteLLM",
    "temperature": 0.0,
    "seed": 42
  },
  "evaluation_results": [...]
}
```

**MLflow Tracing Benefits**:
- Automatic logging of all LangGraph node inputs/outputs
- Execution timeline and performance metrics
- Searchable trace history across experiments
- UI for visualizing execution flow
- Comparison across different evaluation runs

### 6.3 Correctness
**Validation Gates**:
- File exists and is readable at specified commit
- LLM detection returns valid Pydantic-structured output
- Evaluation scores in valid range (0-5)
- Pydantic model validation on all inputs/outputs

**Error Recovery**:
- Each pipeline node handles errors gracefully
- Errors propagate through state but don't halt pipeline
- Failed nodes return error state, subsequent nodes create error responses
- Full error context included in final evaluation result
- No silent failures - all errors logged and included in output

### 6.4 Security
- MySQL credentials via environment variables (.env)
- No secrets in code or Git
- SonarQube token in .env
- Read-only database access

## 7. Technology Stack Summary

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Language | Python | 3.11+ | Type hints required |
| Orchestration | LangGraph | Latest | Single-agent workflow |
| Tracking & Tracing | MLflow | 3.0+ | Experiment tracking, observability |
| Database | MySQL | 8.0+ | DACOS dataset |
| Vector Store | DeepLake | <4.0.0 | Persistent local storage |
| LLM | LiteLLM + Cerebras | llama3.1-8b | Detection & judging |
| Embeddings | Google | text-embedding-004 | RAG retrieval |
| Baseline Tool | SonarQube | 10.6.0-community | Docker container |
| Git | GitPython | Latest | Sparse clone support |
| Testing | pytest | Latest | Unit/integration tests |
| Code Quality | ruff, mypy | Latest | Linting, type checking |
| Package Manager | uv | Latest | Dependency management |

## 8. Deployment Architecture

### 8.1 Local Development Setup
```
├── MySQL (localhost:3306) - DACOS database
├── SonarQube (Docker, localhost:9000) - baseline tool
├── MLflow (./mlruns) - tracing and experiment tracking
├── DeepLake (./data/deeplake/) - smell knowledge (persistent)
├── Git clones (/tmp/smell-eval-clones) - temporary
└── Python environment (venv/uv) - pipeline execution
```

### 8.2 File Structure
```
project/
├── src/
│   ├── pipelines/
│   │   ├── evaluation_pipeline.py  # Main LangGraph pipeline
│   │   └── nodes.py                # Individual graph nodes
│   ├── data/
│   │   ├── mysql_connector.py      # DACOS access
│   │   └── git_ops.py              # Clone/checkout logic
│   ├── agents/
│   │   ├── detector.py             # LLM smell detector
│   │   └── judge.py                # LLM-as-judge evaluator
│   └── models/
│       └── entities.py             # Pydantic models
├── pipeline_reference/              # Reference implementation
├── tests/
│   ├── unit/
│   └── integration/
├── infra/
│   └── sonarqube/
│       └── baseline_scan.py        # Python script for SonarQube baseline
├── eval_results/                   # Evaluation outputs (git-ignored)
├── docs/                           # Documentation
│   ├── architecture.md
│   ├── tech_stack.md
│   ├── tasks.md
│   └── sonarqube_smells.md
├── .env                            # Environment variables (git-ignored)
├── .env.example                    # Template for secrets
├── pyproject.toml                  # Dependencies (uv format)
├── uv.lock                         # Locked dependencies
├── CLAUDE.md                       # Project instructions for AI agents
└── README.md                       # Setup instructions
```

## 9. Future Extensions (Post-Phase 1)

### 9.1 Multi-Agent Workflow
- Coordinator node (star topology)
- Test checker agent
- Test generator agent
- Prioritizer agent
- Refactorer agent
- Test runner agent

### 9.2 Advanced Features
- W&B integration for tracking
- Real-time SonarQube integration in pipeline
- Refactoring execution and validation
- Multiple LLM provider support
- Batch processing optimizations

## 10. Open Questions & Assumptions

### 10.1 Resolved Design Decisions

**1. Repo URL derivation**: 
```python
# project_name uses "_" instead of "/"
# Example: "alibaba_arthas" → "https://github.com/alibaba/arthas"
def derive_repo_url(project_name: str) -> str:
    org, repo = project_name.split("_", 1)
    return f"https://github.com/{org}/{repo}"
```

**2. Commit SHA selection**:
- DACOS dataset published: January 2023 (Zenodo v2: 2023-01-24)
- Use commit immediately before this date: `--before="2023-01-24"`
- For each project, find latest commit before dataset publication
- Command: `git log --before="2023-01-24" --max-count=1 --format=%H`

**Note on smell types**:
- DACOS annotation table tracks 4 design smell types:
  - Complex Method (iscm)
  - Insufficient Modularization (isim)
  - Long Parameter List (islp)
  - Multifaceted Abstraction (isma)
- Focus evaluation on these 4 types present in annotation table

**3. Smell location strategy**:
- NO Designite reruns
- Use existing metrics tables for location context:
  - `class_metrics.type_name` for class-level smells
  - `method_metrics.method_name` for method-level smells
  - Package names from both tables
- Accept approximate location matching in evaluation
- Judge evaluates based on: correct file + correct class/method name (not exact line numbers)

**4. Smell severity**:
- Not tracked in annotation table
- Derive from metrics using thresholds:
  - Complex Method: CC > 10 = HIGH, CC > 5 = MEDIUM, else LOW
  - Long Parameter List: PC > 5 = HIGH, PC > 3 = MEDIUM, else LOW
  - Default to MEDIUM if metrics unavailable

### 10.2 Remaining Questions for Tech Stack Doc
1. **Sample constraints field**: What does `sample_constraints: int` represent?
   - Dataset split indicator (train/test/validation)?
   - Quality filter or confidence score?
   - Subjectivity threshold indicator?

### 10.2 Resolved (from schema)
- ✓ Ground truth location: `annotation` table with binary flags
- ✓ Smell type mapping: `smell` table with id/name/description
- ✓ Multiple smells per sample: annotation flags can be combined
- ✓ Class vs method distinction: `is_class` flag in sample table

### 10.3 Current Assumptions
- Ground truth annotations in `annotation` table (binary flags for 4 smell types)
- Multiple smells can be present per sample (flags can overlap)
- Smell types tracked:
  - **iscm**: Complex Method
  - **isim**: Insufficient Modularization (Multifaceted Abstraction in DACOS paper)
  - **islp**: Long Parameter List
  - **isma**: Magic Number (custom addition, not in original DACOS paper)
- Location details from metrics tables (package/type/method names) - approximate matching acceptable
- Repo URL: `project_name` with "_" replaced by "/" (e.g., "alibaba_arthas" → "github.com/alibaba/arthas")
- Commit SHA: Latest commit before 2023-01-24 (DACOS publication date)
- File paths relative to repo root
- Severity derived from metrics (CC, PC thresholds) or defaults to MEDIUM
- No Designite reruns - use existing metrics only

## 11. Validation Criteria

**Architecture is complete when**:
- [ ] All components identified and described
- [ ] Data flow documented end-to-end
- [ ] Integration points defined with contracts
- [ ] Key entities modeled (Pydantic schemas)
- [ ] Non-functional requirements specified
- [ ] Open questions documented for next phase
- [ ] Reviewable by project supervisor