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
- Promptfoo-driven evaluation framework
- Ground truth from MySQL DACOS dataset
- SonarQube baseline (separate process, not integrated in eval pipeline)
- Local JSON output (no W&B)
- Focus: Java code smell detection quality assessment

### 1.3 Out of Scope (Phase 1)
- Multi-agent workflows (coordinator, test generation, refactoring)
- W&B tracking
- Automated refactoring
- Real-time SonarQube integration in eval pipeline

## 2. System Components

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Promptfoo CLI                          │
│  (orchestrates evaluation runs, provides test cases)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
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

#### 2.2.1 Promptfoo CLI
**Purpose**: Evaluation orchestration framework  
**Responsibilities**:
- Load test cases (DACOS sample IDs)
- Invoke LangGraph pipeline for each test case
- Collect and aggregate evaluation results
- Generate evaluation reports (JSON)

**Configuration**: `promptfoo.config.yaml`
- Provider: LangGraph pipeline endpoint
- Test cases: DACOS sample filters/IDs
- Output: JSON results directory

#### 2.2.2 LangGraph Pipeline
**Purpose**: Core evaluation workflow  
**Implementation**: Single-agent graph with linear flow  
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
isma: bit(1)          -- Magic Number
```

**Note**: DACOS tracks 4 specific smell types via binary flags. For Phase 1, evaluation focuses on these 4 types. Additional smell types may exist in the `smell` table but won't have ground truth annotations.

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

**Note**: DeepLake uses persistent local storage at `./data/deeplake/` for reproducibility

#### 2.2.6 SonarQube Baseline (Separate Process)
**Purpose**: Classical tool baseline for comparison  
**Deployment**: Docker container (sonarqube:10.6.0-community)  
**Configuration**:
- Port: 9000
- Credentials: environment variables (.env)
- Quality profile: default Java profile

**Bash Script Flow**:
```bash
1. Start SonarQube container
2. For each DACOS sample:
   - Clone repo at commit SHA
   - Run sonar-scanner
   - Export issues via REST API (/api/issues/search)
3. Save baseline JSON artifacts
4. Stop container
```

**Output**: Separate JSON files for baseline comparison

## 3. Data Flow

### 3.1 End-to-End Evaluation Flow

```
[Promptfoo Config] 
    ↓ (sample IDs/filters)
[Promptfoo CLI]
    ↓ (invoke for each test case)
[LangGraph Pipeline Entry]
    ↓
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
└────────────┬────────────────────────────┘
             ↓
[Promptfoo CLI]
    ↓ (collect results)
[JSON Output File]
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
    isma: bool  # Magic Number
    
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
            smells.append("Magic Number")
        return smells
```

### 4.2 Configuration Models

```python
class PipelineConfig(BaseModel):
    """Pipeline configuration"""
    llm_model: str = "cerebras/llama3.1-8b"  # Cerebras via LiteLLM
    embedding_model: str = "text-embedding-004"
    vector_db_path: str = "./data/deeplake/smells"  # Persistent local storage
    mysql_host: str = "localhost"
    mysql_database: str = "dacos"
    mysql_user: str  # from env
    mysql_password: str  # from env
    temp_clone_dir: str = "/tmp/smell-eval-clones"
    max_file_size_kb: int = 500  # skip files larger than this
    
class PromptfooConfig(BaseModel):
    """Promptfoo configuration"""
    provider: str = "langgraph"
    test_cases_query: str  # SQL query for DACOS samples
    output_dir: str = "eval_results"
    max_concurrency: int = 1  # sequential for prototype
```

## 5. Integration Points

### 5.1 Promptfoo → LangGraph

**Interface**: Function call from Promptfoo to LangGraph pipeline

`promptfoo.config.yaml`:
```yaml
providers:
  - id: langgraph
    config:
      module: src.pipelines.evaluation_pipeline
      function: run_evaluation
      
prompts:
  - 'Evaluate sample: {{sample_id}}'
  
tests:
  # Generated from DACOS query
  - vars:
      sample_id: 1234
  - vars:
      sample_id: 5678
```

**Contract**:
```python
def run_evaluation(sample_id: int) -> Dict[str, Any]:
    """
    Entry point called by Promptfoo.
    
    Args:
        sample_id: DACOS sample ID
        
    Returns:
        Evaluation result as dict (serializable to JSON)
    """
    pass
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

### 5.4 Context7 Integration

**Purpose**: Fetch up-to-date LangGraph and Promptfoo documentation

**Usage Pattern**:
```python
# Before implementing pipeline, fetch docs
context7.get_library_docs(
    library_id='/langchain-ai/langgraph',
    topic='single agent workflows'
)

context7.get_library_docs(
    library_id='/promptfoo/promptfoo',
    topic='evaluating python functions'
)
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
- Record Promptfoo run config
- Version LangGraph graph definition

**Logging Requirements**:
```json
{
  "run_metadata": {
    "git_sha": "abc123...",
    "timestamp": "2025-10-18T10:30:00Z",
    "pipeline_version": "1.0.0",
    "langgraph_version": "0.2.x",
    "llm_model": "cerebras/llama3.1-8b",
    "llm_provider": "LiteLLM",
    "temperature": 0.0,
    "seed": 42
  },
  "evaluation_results": [...]
}
```

### 6.3 Correctness
**Validation Gates**:
- File exists and is readable
- File size within limits (<500KB)
- LLM detection returns valid structure
- Evaluation scores in valid range (0-5)

**Error Recovery**:
- Skip samples that fail validation
- Log errors but continue batch
- Output partial results

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
| Evaluation | Promptfoo | Latest | CLI-driven |
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
│   ├── models/
│   │   └── entities.py             # Pydantic models
│   └── config/
│       └── settings.py             # Configuration loading
├── experiments/
│   └── notebooks/
│       └── prototype_eval.ipynb    # Original notebook
├── tests/
│   ├── unit/
│   └── integration/
├── infra/
│   ├── sonarqube/
│   │   ├── docker-compose.yml
│   │   └── analyze_baseline.sh    # SonarQube batch script
│   └── mysql/
│       └── schema.sql              # DACOS schema reference
├── eval_results/                   # Promptfoo outputs
├── promptfoo.config.yaml           # Evaluation config
├── .env.example                    # Template for secrets
├── pyproject.toml                  # Dependencies
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
- DACOS paper mentions 3 smells: Multifaceted Abstraction (isim), Complex Method (iscm), Long Parameter List (islp)
- Your database has 4th flag: Magic Number (isma) - may be custom addition
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