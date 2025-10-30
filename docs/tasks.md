# Tasks Checklist: LLM-Based Code Smell Detection

**Version**: 1.0  
**Date**: 2025-10-18  
**Status**: Draft  
**Target**: Junior developer (2-4 hours per task, 100-150 lines of code)

## Task Organization

Tasks are grouped by component and ordered by dependency. Complete tasks in order within each phase.

**Notation**:
- [ ] Not started
- [x] Complete
- Task ID format: `PHASE-COMPONENT-NUMBER` (e.g., P1-INFRA-001)

---

## Phase 1: Infrastructure Setup

### INFRA: SonarQube Docker Setup

#### Task P1-INFRA-001: Create SonarQube Docker Compose file
**Estimated time**: 2 hours  
**Estimated LOC**: 50 lines

**Description**: Create a Docker Compose configuration file for SonarQube 10.6.0-community that runs on port 9000 with persistent volumes.

**Steps**:
1. Create directory: `infra/sonarqube/`
2. Create file: `infra/sonarqube/docker-compose.yml`
3. Define SonarQube service with:
   - Image: `sonarqube:10.6.0-community`
   - Port mapping: `9000:9000`
   - Three volumes: data, extensions, logs
4. Add environment variable: `SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true`
5. Test: Run `docker-compose up` and verify SonarQube starts

**Acceptance criteria**:
- [ ] docker-compose.yml file exists in correct location
- [ ] SonarQube starts without errors
- [ ] Web UI accessible at http://localhost:9000
- [ ] Default admin credentials work (admin/admin)

**Dependencies**: None

---

#### Task P1-INFRA-002: Create SonarQube startup script
**Estimated time**: 2 hours  
**Estimated LOC**: 80 lines

**Description**: Write a bash script that starts SonarQube, waits for it to be ready, and verifies health.

**Steps**:
1. Create file: `infra/sonarqube/start_sonarqube.sh`
2. Make it executable: `chmod +x start_sonarqube.sh`
3. Script should:
   - Start docker-compose in detached mode
   - Wait for port 9000 to be open (use `nc` or `curl`)
   - Retry up to 10 times with 10-second intervals
   - Check health endpoint: `http://localhost:9000/api/system/health`
   - Print success/failure message
4. Test: Run script and verify it waits for SonarQube

**Acceptance criteria**:
- [ ] Script starts SonarQube successfully
- [ ] Script waits until SonarQube is healthy
- [ ] Script exits with code 0 on success, 1 on failure
- [ ] Script prints clear status messages

**Dependencies**: P1-INFRA-001

---

#### Task P1-INFRA-003: Create SonarQube shutdown script
**Estimated time**: 1 hour  
**Estimated LOC**: 30 lines

**Description**: Write a bash script that cleanly stops SonarQube.

**Steps**:
1. Create file: `infra/sonarqube/stop_sonarqube.sh`
2. Make it executable
3. Script should:
   - Run `docker-compose down`
   - Print confirmation message
4. Test: Start SonarQube, then run stop script

**Acceptance criteria**:
- [ ] Script stops SonarQube cleanly
- [ ] No orphaned containers remain
- [ ] Script prints confirmation

**Dependencies**: P1-INFRA-001

---

### CONFIG: Project Configuration

#### Task P1-CONFIG-001: Create .env.example template
**Estimated time**: 1 hour  
**Estimated LOC**: 30 lines

**Description**: Create a template file showing all required environment variables with example values.

**Steps**:
1. Create file: `.env.example` in project root
2. Add variables with descriptions:
   - `CEREBRAS_API_KEY=your_key_here`
   - `GOOGLE_API_KEY=your_key_here` (for embeddings)
   - `MYSQL_HOST=localhost`
   - `MYSQL_PORT=3306`
   - `MYSQL_DATABASE=dacos`
   - `MYSQL_USER=dacos_reader`
   - `MYSQL_PASSWORD=your_password`
   - `SONAR_URL=http://localhost:9000`
   - `SONAR_TOKEN=your_token`
3. Add comments explaining each variable

**Acceptance criteria**:
- [ ] File exists with all required variables
- [ ] Each variable has a comment explaining its purpose
- [ ] Example values are clear placeholders (not real credentials)

**Dependencies**: None

---

#### Task P1-CONFIG-002: Create pyproject.toml with dependencies
**Estimated time**: 3 hours
**Estimated LOC**: 100 lines

**Description**: Create pyproject.toml configuration file with all project dependencies listed in Tech Stack document for use with uv package manager.

**Steps**:
1. Create file: `pyproject.toml` in project root
2. Add `[project]` section with project metadata (name, version, description, requires-python)
3. Add `dependencies` array with production dependencies:
   - Python version constraint: `>=3.11,<3.13"`
   - langgraph>=0.2.0, langchain>=0.3.0, langchain-community>=0.3.0
   - litellm>=1.0.0 (for Cerebras integration)
   - langchain-google-genai>=2.0.0 (for embeddings)
   - deeplake<4.0.0, mysql-connector-python>=9.0.0, GitPython>=3.1.0
   - pydantic>=2.0.0, pydantic-settings>=2.0.0, python-dotenv>=1.0.0
   - requests>=2.31.0, pandas>=2.2.0
4. Add `[project.optional-dependencies]` with dev tools:
   - pytest>=8.0.0, pytest-cov>=4.1.0, pytest-asyncio>=0.23.0
   - ruff>=0.5.0, mypy>=1.10.0, ipython>=8.20.0, jupyter>=1.0.0
5. Test: Run `uv pip install -e .` to verify dependencies resolve

**Acceptance criteria**:
- [ ] pyproject.toml has correct structure for uv/pip
- [ ] All dependencies from Tech Stack included with correct versions
- [ ] `uv pip install -e .` succeeds without conflicts

**Dependencies**: None

---

#### Task P1-CONFIG-003: Create .gitignore file
**Estimated time**: 1 hour  
**Estimated LOC**: 60 lines

**Description**: Create .gitignore file to exclude generated files, secrets, and build artifacts.

**Steps**:
1. Create file: `.gitignore` in project root
2. Add patterns for:
   - Python artifacts: `__pycache__/`, `*.pyc`, `*.egg-info/`
   - Virtual environments: `venv/`, `.venv/`
   - Secrets: `.env`, `.env.local`
   - Generated data: `eval_results/`, `data/deeplake/`
   - IDE files: `.vscode/`, `.idea/`
   - Test artifacts: `.coverage`, `.pytest_cache/`
3. Test: Create a .env file and verify git status doesn't show it

**Acceptance criteria**:
- [ ] .gitignore includes all patterns from Tech Stack document
- [ ] Test .env file is ignored by git
- [ ] __pycache__ directories are ignored

**Dependencies**: None

---

## Phase 2: Data Layer

### MODELS: Pydantic Models

#### Task P2-MODELS-001: Create SmellAnnotation model
**Estimated time**: 2 hours  
**Estimated LOC**: 40 lines

**Description**: Create Pydantic model for ground truth smell annotations from DACOS database.

**Steps**:
1. Create file: `src/models/entities.py`
2. Import: `from pydantic import BaseModel, Field`
3. Import: `from typing import Optional`
4. Create class `SmellAnnotation` with fields:
   - `smell_type: str` - smell name (e.g., "Complex Method")
   - `is_present: bool` - from annotation flags
   - `package_name: Optional[str]`
   - `type_name: Optional[str]` - class name
   - `method_name: Optional[str]`
   - `loc: Optional[int]` - lines of code
   - `cc: Optional[int]` - cyclomatic complexity
   - `pc: Optional[int]` - parameter count
5. Test: Create instance and convert to JSON

**Acceptance criteria**:
- [ ] Model defined with all fields from Architecture doc
- [ ] Can create instance: `SmellAnnotation(smell_type="Complex Method", is_present=True)`
- [ ] Can serialize to JSON: `.model_dump_json()`
- [ ] Type hints work correctly

**Dependencies**: None

---

#### Task P2-MODELS-002: Create SmellDetection model
**Estimated time**: 2 hours  
**Estimated LOC**: 50 lines

**Description**: Create Pydantic model for LLM-detected code smells.

**Steps**:
1. In file: `src/models/entities.py`
2. Import: `from typing import Literal, Optional`
3. Create class `SmellDetection` with fields:
   - `smell_type: str`
   - `location: str` - description of where smell was found
   - `description: str`
   - `severity: Literal["LOW", "MEDIUM", "HIGH"]`
   - `refactoring_suggestion: str`
   - `confidence: Optional[float]` - range 0.0 to 1.0
4. Add field validators where needed
5. Test: Create instance with all fields

**Acceptance criteria**:
- [ ] Model has all required fields
- [ ] Severity only accepts LOW/MEDIUM/HIGH
- [ ] Confidence validated to be between 0 and 1
- [ ] Can serialize to JSON

**Dependencies**: P2-MODELS-001

---

#### Task P2-MODELS-003: Create EvaluationResult model
**Estimated time**: 3 hours  
**Estimated LOC**: 80 lines

**Description**: Create Pydantic models for evaluation results (per-smell and overall).

**Steps**:
1. In file: `src/models/entities.py`
2. Import: `from typing import List, Literal`
3. Create enum: `EvaluationScore` with values: EXCELLENT, GOOD, ACCEPTABLE, POOR, INCORRECT
4. Create class `SmellEvaluation` with fields:
   - `detected_smell: str`
   - `location: str`
   - `ground_truth_match: Optional[str]`
   - `score: EvaluationScore`
   - `justification: str`
5. Create class `EvaluationResult` with fields:
   - `sample_id: int`
   - `file_path: str`
   - `overall_score: float` - 0 to 5
   - `precision: float`
   - `recall: float`
   - `f1_score: float`
   - `evaluations: List[SmellEvaluation]`
   - `summary: str`
   - `timestamp: str`
   - `git_sha: str`
6. Test: Create nested structure

**Acceptance criteria**:
- [ ] Both models defined correctly
- [ ] EvaluationScore enum works
- [ ] Can create nested structure with list of SmellEvaluation
- [ ] JSON serialization works

**Dependencies**: P2-MODELS-002

---

#### Task P2-MODELS-004: Create DACOSSample model
**Estimated time**: 3 hours  
**Estimated LOC**: 100 lines

**Description**: Create Pydantic model for DACOS database records with annotations.

**Steps**:
1. In file: `src/models/entities.py`
2. Create class `DACOSSample` with fields:
   - Sample table fields: `id`, `designite_id`, `has_smell`, `is_class`, `path_to_file`, `project_name`, `sample_constraints`, `smells`
   - Annotation fields: `iscm`, `isim`, `islp`, `isma` (all bool)
   - Smell info: `smell_name`, `smell_description`
   - Derived: `repo_url`, `commit_sha`
3. Add property method `ground_truth_smells()` that returns list of active smells
4. Add method to convert annotation flags to SmellAnnotation objects
5. Test: Create instance and call ground_truth_smells()

**Acceptance criteria**:
- [ ] Model matches DACOS schema exactly
- [ ] ground_truth_smells() returns correct list
- [ ] Can convert flags to SmellAnnotation list
- [ ] All fields properly typed

**Dependencies**: P2-MODELS-001

---

### DATABASE: MySQL Connector

#### Task P2-DB-001: Create MySQL connection pool
**Estimated time**: 3 hours  
**Estimated LOC**: 80 lines

**Description**: Create a module that establishes a connection pool to MySQL using environment variables.

**Steps**:
1. Create file: `src/data/mysql_connector.py`
2. Import: `import mysql.connector`, `from mysql.connector import pooling`
3. Import: `import os`, `from dotenv import load_dotenv`
4. Call `load_dotenv()` at module level
5. Create function `get_connection_pool()` that:
   - Reads env vars: MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD
   - Creates connection pool with 5 connections
   - Returns pool object
6. Create function `get_connection()` that gets connection from pool
7. Test: Call get_connection() and execute simple query

**Acceptance criteria**:
- [ ] Connection pool created successfully
- [ ] Can get connection from pool
- [ ] Environment variables loaded correctly
- [ ] Handles missing env vars with clear error

**Dependencies**: P1-CONFIG-001, P1-CONFIG-002

---

#### Task P2-DB-002: Create function to fetch sample by ID
**Estimated time**: 3 hours  
**Estimated LOC**: 100 lines

**Description**: Write a function that fetches a complete DACOS sample record with annotations.

**Steps**:
1. In file: `src/data/mysql_connector.py`
2. Import: `from src.models.entities import DACOSSample`
3. Create function `fetch_sample_by_id(sample_id: int) -> DACOSSample`:
   - Get connection from pool
   - Execute SQL with JOINs:
     ```sql
     SELECT s.*, a.iscm, a.isim, a.islp, a.isma, sm.name, sm.description
     FROM tagman5.sample s
     LEFT JOIN tagman5.annotation a ON s.id = a.sample_id
     LEFT JOIN tagman5.smell sm ON s.smells = sm.id
     WHERE s.id = ?
     ```
   - Fetch one row
   - Convert row to DACOSSample object
   - Close connection
   - Return sample
4. Handle case where sample not found (return None)
5. Test: Fetch a known sample ID

**Acceptance criteria**:
- [ ] Function returns DACOSSample object for valid ID
- [ ] Returns None for invalid ID
- [ ] All fields populated correctly
- [ ] Connection properly closed

**Dependencies**: P2-DB-001, P2-MODELS-004

---

#### Task P2-DB-003: Create function to fetch samples by filters
**Estimated time**: 3 hours  
**Estimated LOC**: 120 lines

**Description**: Write a function that fetches multiple samples based on filter criteria.

**Steps**:
1. In file: `src/data/mysql_connector.py`
2. Import: `from typing import List, Optional`
3. Create function `fetch_samples(project_name: Optional[str] = None, has_smell: Optional[bool] = None, limit: int = 100) -> List[DACOSSample]`:
   - Build SQL query dynamically based on filters
   - Use parameterized queries (prevent SQL injection)
   - Execute query
   - Fetch all rows
   - Convert each row to DACOSSample
   - Return list
4. Test: Fetch samples with different filter combinations

**Acceptance criteria**:
- [ ] Returns list of DACOSSample objects
- [ ] Filters work correctly (project_name, has_smell)
- [ ] Limit parameter respected
- [ ] Empty list returned when no matches
- [ ] SQL injection prevented (parameterized queries)

**Dependencies**: P2-DB-002

---

### GIT: Repository Operations

#### Task P2-GIT-001: Create function to derive repo URL from project name
**Estimated time**: 1 hour  
**Estimated LOC**: 30 lines

**Description**: Write a simple function that converts project_name to GitHub repo URL.

**Steps**:
1. Create file: `src/data/git_ops.py`
2. Create function `derive_repo_url(project_name: str) -> str`:
   - Split project_name on "_" character
   - First part is org, second part is repo
   - Return f"https://github.com/{org}/{repo}"
3. Test with examples:
   - "alibaba_arthas" → "https://github.com/alibaba/arthas"
   - "watabou_pixel-dungeon" → "https://github.com/watabou/pixel-dungeon"

**Acceptance criteria**:
- [ ] Function correctly splits on underscore
- [ ] Returns valid GitHub URL format
- [ ] Works for test examples

**Dependencies**: None

---

#### Task P2-GIT-002: Create function to get commit SHA before date
**Estimated time**: 3 hours  
**Estimated LOC**: 80 lines

**Description**: Write a function that finds the latest commit before a specific date.

**Steps**:
1. In file: `src/data/git_ops.py`
2. Import: `from git import Repo`, `import tempfile`, `import shutil`
3. Create function `get_commit_before_date(repo_url: str, before_date: str = "2023-01-24") -> str`:
   - Create temporary directory
   - Clone repository (shallow, depth=1000)
   - Run git log command: `git log --before="{before_date}" --max-count=1 --format=%H`
   - Extract commit SHA from output
   - Clean up temp directory
   - Return commit SHA
4. Handle errors (repo doesn't exist, no commits before date)
5. Test with alibaba/arthas repo

**Acceptance criteria**:
- [ ] Returns valid commit SHA
- [ ] Works with "2023-01-24" cutoff date
- [ ] Cleans up temporary files
- [ ] Handles errors gracefully

**Dependencies**: P2-GIT-001

---

#### Task P2-GIT-003: Create function for sparse file checkout
**Estimated time**: 4 hours  
**Estimated LOC**: 130 lines

**Description**: Write a function that clones a repo, checks out a specific commit, and extracts one file.

**Steps**:
1. In file: `src/data/git_ops.py`
2. Import: `from pathlib import Path`
3. Create function `clone_and_read_file(repo_url: str, commit_sha: str, file_path: str) -> str`:
   - Create temporary directory
   - Clone with sparse checkout:
     - `git clone --depth 1 --filter=blob:none --sparse <url> <dir>`
     - `git checkout <commit_sha>`
     - `git sparse-checkout set <file_path>`
   - Read file content
   - Clean up temp directory
   - Return file content as string
4. Handle file not found, invalid commit, etc.
5. Test with a known file from alibaba/arthas

**Acceptance criteria**:
- [ ] Clones only necessary data (sparse/shallow)
- [ ] Checks out correct commit
- [ ] Returns file content as string
- [ ] Cleans up temporary directory
- [ ] Handles errors with clear messages

**Dependencies**: P2-GIT-002

---

## Phase 3: LLM Components

### VECTOR: Vector Database

#### Task P3-VECTOR-001: Create function to load smell documents
**Estimated time**: 3 hours  
**Estimated LOC**: 100 lines

**Description**: Write a function that loads smell documentation from markdown files and returns Document objects.

**Steps**:
1. Create file: `src/data/vector_db.py`
2. Import: `from glob import glob`, `from langchain.schema import Document`
3. Import: `from langchain_text_splitters import MarkdownHeaderTextSplitter`
4. Create function `load_smell_documents(smell_files: List[str]) -> List[Document]`:
   - Define headers: `[("#", "Title"), ("##", "Section"), ("###", "Subsection")]`
   - For each file:
     - Read file content
     - Skip frontmatter (if present)
     - Split by headers using MarkdownHeaderTextSplitter
     - Add source file to metadata
   - Return list of Document objects
5. Test with smell markdown files

**Acceptance criteria**:
- [ ] Loads markdown files correctly
- [ ] Splits by headers
- [ ] Metadata includes source file
- [ ] Returns list of Document objects

**Dependencies**: None

---

#### Task P3-VECTOR-002: Create function to initialize DeepLake vector DB
**Estimated time**: 3 hours  
**Estimated LOC**: 80 lines

**Description**: Write a function that creates a DeepLake vector database from smell documents.

**Steps**:
1. In file: `src/data/vector_db.py`
2. Import: `from langchain.vectorstores import DeepLake`
3. Import: `from langchain_google_genai import GoogleGenerativeAIEmbeddings`
4. Create function `create_smell_vector_db(documents: List[Document], dataset_path: str = "./data/deeplake/smells") -> DeepLake`:
   - Create embeddings object: `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`
   - Create DeepLake from documents with dataset_path
   - Return DeepLake object
5. Test: Create DB from sample documents and verify persistent storage

**Acceptance criteria**:
- [ ] Creates DeepLake vector DB successfully
- [ ] Documents embedded correctly
- [ ] Returns DeepLake object
- [ ] Persists to local directory (./data/deeplake/)

**Dependencies**: P3-VECTOR-001

---

#### Task P3-VECTOR-003: Create function to get retriever
**Estimated time**: 2 hours  
**Estimated LOC**: 40 lines

**Description**: Write a function that configures and returns a retriever from the vector DB.

**Steps**:
1. In file: `src/data/vector_db.py`
2. Create function `get_retriever(vector_db: DeepLake, k: int = 20)`:
   - Get retriever from vector_db: `vector_db.as_retriever()`
   - Configure: `retriever.search_kwargs['distance_metric'] = 'cos'`
   - Configure: `retriever.search_kwargs['k'] = k`
   - Return retriever
3. Test: Retrieve documents for a query

**Acceptance criteria**:
- [ ] Returns configured retriever
- [ ] Cosine distance metric set
- [ ] k parameter works
- [ ] Can retrieve relevant documents

**Dependencies**: P3-VECTOR-002

---

### LLM: Detector Agent

#### Task P3-LLM-001: Create function to initialize Cerebras LLM via LiteLLM
**Estimated time**: 2 hours
**Estimated LOC**: 50 lines

**Description**: Write a function that creates and configures a Cerebras LLM client using LiteLLM's unified API.

**Steps**:
1. Create file: `src/agents/detector.py`
2. Import: `from litellm import completion`
3. Import: `import os`
4. Create function `get_llm_completion(messages: list, temperature: float = 0.0, max_tokens: int = 4096, **kwargs)`:
   - Check CEREBRAS_API_KEY exists in environment
   - Call `completion()` with parameters:
     - model="cerebras/llama3.1-8b"
     - messages=messages
     - temperature (default 0.0)
     - max_tokens (default 4096)
     - api_key=os.environ.get("CEREBRAS_API_KEY")
     - Additional kwargs passed through
   - Return completion response
5. Test: Call function with simple message list and verify response

**Acceptance criteria**:
- [ ] Creates Cerebras LLM client via LiteLLM
- [ ] Checks for CEREBRAS_API_KEY
- [ ] Parameters configurable
- [ ] Can invoke LLM successfully

**Dependencies**: P1-CONFIG-001

**Note**: LiteLLM provides OpenAI-compatible API for Cerebras models with model prefix `cerebras/`

---

#### Task P3-LLM-002: Create detection prompt template
**Estimated time**: 3 hours  
**Estimated LOC**: 120 lines

**Description**: Create a prompt template for code smell detection with structured output.

**Steps**:
1. In file: `src/agents/detector.py`
2. Import: `from langchain.prompts import PromptTemplate`
3. Import: `from langchain.output_parsers import PydanticOutputParser`
4. Import: `from src.models.entities import SmellDetection`
5. Import: `from typing import List`
6. Create Pydantic model for detection results with list of SmellDetection
7. Create prompt template string:
   - System instructions: "You are an expert code smell detector"
   - Context section: "{context}" (from RAG)
   - Code section: "{code}"
   - Output format instructions: "{format_instructions}"
8. Create PydanticOutputParser for structured output
9. Create PromptTemplate with input variables: ["context", "code"]
10. Test: Format prompt with sample data

**Acceptance criteria**:
- [ ] Prompt template created
- [ ] Includes context and code placeholders
- [ ] Output parser configured for structured response
- [ ] Can format prompt with variables

**Dependencies**: P3-LLM-001, P2-MODELS-002

---

#### Task P3-LLM-003: Create smell detection function
**Estimated time**: 4 hours  
**Estimated LOC**: 100 lines

**Description**: Write a function that detects code smells using RAG and LLM.

**Steps**:
1. In file: `src/agents/detector.py`
2. Create function `detect_smells(code: str, retriever, llm) -> List[SmellDetection]`:
   - Retrieve relevant smell documentation: `retriever.get_relevant_documents(code)`
   - Format context from retrieved docs
   - Format prompt with context and code
   - Invoke LLM with prompt
   - Parse response using output parser
   - Return list of SmellDetection objects
3. Handle parsing errors (fallback to empty list)
4. Test with sample Java code

**Acceptance criteria**:
- [ ] Retrieves relevant documentation
- [ ] Formats prompt correctly
- [ ] Invokes LLM successfully
- [ ] Parses structured response
- [ ] Returns list of SmellDetection

**Dependencies**: P3-LLM-002, P3-VECTOR-003

---

### JUDGE: Evaluation Agent

#### Task P3-JUDGE-001: Create evaluation prompt template
**Estimated time**: 4 hours  
**Estimated LOC**: 150 lines

**Description**: Create a prompt template for LLM-as-judge evaluation with rubric.

**Steps**:
1. Create file: `src/agents/judge.py`
2. Import: `from langchain.prompts import PromptTemplate`
3. Import: `from langchain.output_parsers import PydanticOutputParser`
4. Import: `from src.models.entities import EvaluationResult`
5. Create long prompt template with:
   - Role: "You are an expert evaluator for code smell detection"
   - Rubric: EXCELLENT (5), GOOD (4), ACCEPTABLE (3), POOR (2), INCORRECT (1)
   - Matching criteria for approximate locations
   - Input placeholders: {ground_truth}, {detected_smells}
   - Output format instructions
6. Create PydanticOutputParser for EvaluationResult
7. Create PromptTemplate
8. Test: Format with sample data

**Acceptance criteria**:
- [ ] Prompt includes complete rubric
- [ ] Matching criteria explained clearly
- [ ] Output parser configured
- [ ] Can format with ground truth and detections

**Dependencies**: P2-MODELS-003

---

#### Task P3-JUDGE-002: Create evaluation function
**Estimated time**: 3 hours  
**Estimated LOC**: 100 lines

**Description**: Write a function that evaluates detection quality using LLM-as-judge.

**Steps**:
1. In file: `src/agents/judge.py`
2. Import: `import json`
3. Create function `evaluate_detections(ground_truth: List[SmellAnnotation], detected_smells: List[SmellDetection], llm) -> EvaluationResult`:
   - Convert inputs to JSON strings
   - Format prompt with ground truth and detections
   - Invoke LLM
   - Parse response to EvaluationResult
   - Return result
4. Handle parsing errors gracefully
5. Test with sample ground truth and detections

**Acceptance criteria**:
- [ ] Formats inputs correctly
- [ ] Invokes LLM successfully
- [ ] Parses structured evaluation response
- [ ] Returns EvaluationResult object
- [ ] Handles errors without crashing

**Dependencies**: P3-JUDGE-001, P3-LLM-001

---

## Phase 4: Pipeline Integration

### PIPELINE: LangGraph Pipeline

#### Task P4-PIPELINE-001: Create LangGraph state definition
**Estimated time**: 2 hours  
**Estimated LOC**: 60 lines

**Description**: Define the state schema for the LangGraph evaluation pipeline.

**Steps**:
1. Create file: `src/pipelines/nodes.py`
2. Import: `from typing import TypedDict, List, Optional`
3. Import: `from src.models.entities import SmellAnnotation, SmellDetection, EvaluationResult`
4. Create class `EvaluationState(TypedDict)`:
   - `sample_id: int`
   - `file_path: str`
   - `file_content: str`
   - `ground_truth: List[SmellAnnotation]`
   - `llm_detections: List[SmellDetection]`
   - `evaluation_result: Optional[EvaluationResult]`
   - `error: Optional[str]`
5. Test: Create instance with required fields

**Acceptance criteria**:
- [ ] State class defined with TypedDict
- [ ] All fields from Architecture doc included
- [ ] Type hints correct
- [ ] Can create state dict

**Dependencies**: P2-MODELS-001, P2-MODELS-002, P2-MODELS-003

---

#### Task P4-PIPELINE-002: Create fetch_sample node
**Estimated time**: 3 hours  
**Estimated LOC**: 80 lines

**Description**: Create a LangGraph node function that fetches sample data from MySQL.

**Steps**:
1. In file: `src/pipelines/nodes.py`
2. Import: `from src.data.mysql_connector import fetch_sample_by_id`
3. Import: `from src.data.git_ops import derive_repo_url, get_commit_before_date`
4. Create function `fetch_sample_node(state: EvaluationState) -> EvaluationState`:
   - Get sample_id from state
   - Fetch sample from database
   - If sample not found, set error and return
   - Derive repo_url from project_name
   - Get commit_sha before cutoff date
   - Update state with file_path, ground_truth
   - Return updated state
5. Test: Call with mock state

**Acceptance criteria**:
- [ ] Fetches sample from database
- [ ] Derives repo URL correctly
- [ ] Gets commit SHA
- [ ] Updates state correctly
- [ ] Handles errors (sets error field)

**Dependencies**: P4-PIPELINE-001, P2-DB-002, P2-GIT-001, P2-GIT-002

---

#### Task P4-PIPELINE-003: Create clone_repo node
**Estimated time**: 3 hours  
**Estimated LOC**: 70 lines

**Description**: Create a LangGraph node that clones the repository and reads the target file.

**Steps**:
1. In file: `src/pipelines/nodes.py`
2. Import: `from src.data.git_ops import clone_and_read_file`
3. Create function `clone_repo_node(state: EvaluationState) -> EvaluationState`:
   - Get repo_url, commit_sha, file_path from state
   - Call clone_and_read_file()
   - If error, set state error and return
   - Update state with file_content
   - Return updated state
4. Test: Call with mock state containing valid repo info

**Acceptance criteria**:
- [ ] Clones repository successfully
- [ ] Reads file content
- [ ] Updates state with content
- [ ] Handles errors gracefully

**Dependencies**: P4-PIPELINE-002, P2-GIT-003

---

#### Task P4-PIPELINE-004: Create detect_smells node
**Estimated time**: 3 hours  
**Estimated LOC**: 80 lines

**Description**: Create a LangGraph node that detects code smells using the LLM detector.

**Steps**:
1. In file: `src/pipelines/nodes.py`
2. Import: `from src.agents.detector import detect_smells, get_llm`
3. Import: `from src.data.vector_db import get_retriever`
4. At module level, initialize LLM and retriever (will be reused)
5. Create function `detect_smells_node(state: EvaluationState) -> EvaluationState`:
   - Get file_content from state
   - Call detect_smells() with content, retriever, llm
   - Update state with llm_detections
   - Return updated state
6. Test: Call with state containing file content

**Acceptance criteria**:
- [ ] Detects smells successfully
- [ ] Returns list of SmellDetection objects
- [ ] Updates state correctly
- [ ] Handles LLM errors

**Dependencies**: P4-PIPELINE-003, P3-LLM-003, P3-VECTOR-003

---

#### Task P4-PIPELINE-005: Create judge_evaluation node
**Estimated time**: 3 hours  
**Estimated LOC**: 70 lines

**Description**: Create a LangGraph node that evaluates detection quality using LLM-as-judge.

**Steps**:
1. In file: `src/pipelines/nodes.py`
2. Import: `from src.agents.judge import evaluate_detections`
3. Create function `judge_evaluation_node(state: EvaluationState) -> EvaluationState`:
   - Get ground_truth and llm_detections from state
   - Call evaluate_detections()
   - Update state with evaluation_result
   - Return updated state
4. Test: Call with state containing ground truth and detections

**Acceptance criteria**:
- [ ] Evaluates detections successfully
- [ ] Returns EvaluationResult object
- [ ] Updates state correctly
- [ ] Handles evaluation errors

**Dependencies**: P4-PIPELINE-004, P3-JUDGE-002

---

#### Task P4-PIPELINE-006: Create LangGraph pipeline
**Estimated time**: 4 hours  
**Estimated LOC**: 100 lines

**Description**: Assemble all nodes into a LangGraph StateGraph pipeline.

**Steps**:
1. Create file: `src/pipelines/evaluation_pipeline.py`
2. Import: `from langgraph.graph import StateGraph, END`
3. Import all node functions from `src.pipelines.nodes`
4. Import: `from src.pipelines.nodes import EvaluationState`
5. Create function `create_evaluation_graph()`:
   - Create StateGraph with EvaluationState
   - Add nodes: fetch_sample, clone_repo, detect_smells, judge_evaluation
   - Add edges in sequence: START → fetch → clone → detect → judge → END
   - Compile graph
   - Return compiled graph
6. Test: Create graph and verify it compiles

**Acceptance criteria**:
- [ ] Graph created with all nodes
- [ ] Nodes connected in correct sequence
- [ ] Graph compiles without errors
- [ ] Returns compiled graph

**Dependencies**: P4-PIPELINE-005

---

#### Task P4-PIPELINE-007: Create pipeline entry point function
**Estimated time**: 3 hours
**Estimated LOC**: 80 lines

**Description**: Create the main function to run evaluations with MLflow tracing.

**Steps**:
1. In file: `src/pipelines/evaluation_pipeline.py`
2. Import: `from typing import Dict, Any`
3. Import: `import json`
4. Create function `run_evaluation(sample_id: int) -> Dict[str, Any]`:
   - Create initial state with sample_id
   - Get graph from create_evaluation_graph()
   - Invoke graph with initial state
   - Get final state
   - If error in state, return error dict
   - Convert evaluation_result to dict
   - Return result
5. Test: Call with a sample ID

**Acceptance criteria**:
- [ ] Function accepts sample_id
- [ ] Runs complete pipeline
- [ ] Returns evaluation result as dict
- [ ] Returns error dict on failure
- [ ] JSON-serializable output

**Dependencies**: P4-PIPELINE-006

---

## Phase 5: Evaluation Framework

### MLFLOW: Batch Evaluation

#### Task P5-MLFLOW-001: Create batch evaluation script
**Estimated time**: 3 hours
**Estimated LOC**: 100 lines

**Description**: Write a script that runs batch evaluations with MLflow tracking.

**Steps**:
1. Create file: `scripts/run_batch_evaluation.py`
2. Import: `from src.data.mysql_connector import fetch_samples`
3. Import: `from src.pipelines.evaluation_pipeline import run_evaluation`
4. Import: `import mlflow`
5. Create function `run_batch_evaluation(limit: int = 10, project_name: str = None)`:
   - Fetch samples with has_smell=True
   - Initialize MLflow experiment
   - For each sample, call run_evaluation()
   - Collect results and aggregate metrics
   - Log batch metrics to MLflow
6. Add main block with CLI arguments
7. Test: Run script and verify MLflow tracking

**Acceptance criteria**:
- [ ] Fetches samples from database
- [ ] Runs evaluation for each sample
- [ ] Each run automatically traced in MLflow
- [ ] Batch metrics logged to MLflow
- [ ] Progress printed to console

**Dependencies**: P4-PIPELINE-007, P2-DB-003

---

#### Task P5-MLFLOW-002: Create MLflow results export script
**Estimated time**: 2 hours
**Estimated LOC**: 80 lines

**Description**: Write a script to export MLflow results to JSON/CSV for analysis.

**Steps**:
1. Create file: `scripts/export_mlflow_results.py`
2. Import: `import mlflow`, `import pandas as pd`
3. Create function `export_results(experiment_name: str, output_format: str = "json")`:
   - Connect to MLflow tracking
   - Load experiment runs
   - Extract evaluation metrics
   - Export to JSON or CSV
4. Add main block with CLI arguments
5. Test: Run script and verify export

**Acceptance criteria**:
- [ ] Loads experiment runs from MLflow
- [ ] Extracts all evaluation metrics
- [ ] Exports to JSON or CSV
- [ ] Handles missing experiments gracefully

**Dependencies**: P5-MLFLOW-001

---

#### Task P5-MLFLOW-003: Create evaluation analysis notebook
**Estimated time**: 3 hours
**Estimated LOC**: 150 lines

**Description**: Create Jupyter notebook for analyzing MLflow evaluation results.

**Steps**:
1. Create file: `experiments/notebooks/analyze_results.ipynb`
2. Add cells for:
   - Loading MLflow experiment data
   - Computing aggregate statistics
   - Visualizing precision/recall/F1 distributions
   - Comparing across different runs
   - Analyzing per-smell-type performance
3. Add markdown explanations
4. Test: Run all cells successfully

**Acceptance criteria**:
- [ ] Loads data from MLflow
- [ ] Computes summary statistics
- [ ] Creates visualizations
- [ ] All cells execute without errors

**Dependencies**: P5-MLFLOW-001

---

## Phase 6: Configuration & Documentation

### CONFIG: Settings Module

#### Task P6-CONFIG-001: Create Settings class
**Estimated time**: 3 hours  
**Estimated LOC**: 100 lines

**Description**: Create a Pydantic Settings class for configuration management.

**Steps**:
1. Create file: `src/config/settings.py`
2. Import: `from pydantic_settings import BaseSettings`
3. Import: `from functools import lru_cache`
4. Create class `Settings(BaseSettings)` with fields:
   - LLM: google_api_key, llm_model, llm_temperature, llm_max_tokens
   - Database: mysql_host, mysql_port, mysql_database, mysql_user, mysql_password
   - Paths: temp_clone_dir, vector_db_path, eval_results_dir
   - Limits: max_file_size_kb, max_samples_per_batch
   - Git: commit_cutoff_date
5. Set model_config: env_file=".env", case_sensitive=False
6. Create cached function `get_settings() -> Settings`
7. Test: Load settings from .env

**Acceptance criteria**:
- [ ] Settings class loads from .env
- [ ] All fields have correct types
- [ ] Caching works (singleton pattern)
- [ ] Validation errors are clear

**Dependencies**: P1-CONFIG-001

---

#### Task P6-CONFIG-002: Create environment verification script
**Estimated time**: 3 hours  
**Estimated LOC**: 120 lines

**Description**: Create a script that verifies all required environment variables and connections.

**Steps**:
1. Create file: `scripts/verify_env.py`
2. Import: `from src.config.settings import get_settings`
3. Import: `from src.data.mysql_connector import get_connection`
4. Import: `import sys`
5. Create function `verify_environment()`:
   - Try to load settings (will fail if vars missing)
   - Test MySQL connection
   - Test Google API key (simple LLM call)
   - Check if required directories exist
   - Print success or failure for each check
   - Return True if all pass, False otherwise
6. Add main block
7. Test: Run with correct and incorrect .env

**Acceptance criteria**:
- [ ] Checks all required environment variables
- [ ] Tests MySQL connection
- [ ] Tests Google API access
- [ ] Prints clear status for each check
- [ ] Exits with correct code

**Dependencies**: P6-CONFIG-001, P2-DB-001

---

### DOCS: Setup Instructions

#### Task P6-DOCS-001: Create README.md
**Estimated time**: 3 hours  
**Estimated LOC**: 150 lines (markdown)

**Description**: Create main README with project overview and quick start guide.

**Steps**:
1. Create file: `README.md` in project root
2. Add sections:
   - Project title and description
   - Prerequisites (Python 3.11+, MySQL, Docker, Node.js)
   - Quick start steps:
     1. Clone repo
     2. Install Poetry
     3. Run `poetry install`
     4. Copy .env.example to .env and fill in
     5. Start SonarQube
     6. Run verification script
     7. Export test cases
     8. Run evaluation
   - Link to full documentation in docs/
3. Keep it concise (quick start only)

**Acceptance criteria**:
- [ ] README exists with clear structure
- [ ] Quick start steps are complete
- [ ] Prerequisites listed
- [ ] Links to detailed docs

**Dependencies**: None

---

## Phase 7: Testing

### TESTS: Unit Tests

#### Task P7-TESTS-001: Create test for derive_repo_url
**Estimated time**: 2 hours  
**Estimated LOC**: 60 lines

**Description**: Write unit tests for the repo URL derivation function.

**Steps**:
1. Create file: `tests/unit/test_git_ops.py`
2. Import: `from src.data.git_ops import derive_repo_url`
3. Import: `import pytest`
4. Write test function `test_derive_repo_url()`:
   - Test case: "alibaba_arthas" → "https://github.com/alibaba/arthas"
   - Test case: "watabou_pixel-dungeon" → "https://github.com/watabou/pixel-dungeon"
   - Test edge case: single word (should fail)
5. Run: `pytest tests/unit/test_git_ops.py`

**Acceptance criteria**:
- [ ] Test file created
- [ ] All test cases pass
- [ ] Edge case handled
- [ ] Can run with pytest

**Dependencies**: P2-GIT-001

---

#### Task P7-TESTS-002: Create test for MySQL connection
**Estimated time**: 2 hours  
**Estimated LOC**: 70 lines

**Description**: Write integration test for MySQL connection (requires database).

**Steps**:
1. Create file: `tests/integration/test_mysql_connector.py`
2. Import: `from src.data.mysql_connector import get_connection, fetch_sample_by_id`
3. Import: `pytest`
4. Write test function `test_connection()`:
   - Get connection
   - Execute simple query: `SELECT 1`
   - Assert result is 1
5. Write test function `test_fetch_sample()`:
   - Fetch a known sample ID
   - Assert result is not None
   - Check fields are populated
6. Run: `pytest tests/integration/test_mysql_connector.py`

**Acceptance criteria**:
- [ ] Test connects to database
- [ ] Query executes successfully
- [ ] Fetch sample works
- [ ] Tests pass when database available

**Dependencies**: P2-DB-002

---

#### Task P7-TESTS-003: Create test for Pydantic models
**Estimated time**: 2 hours  
**Estimated LOC**: 80 lines

**Description**: Write unit tests for all Pydantic models.

**Steps**:
1. Create file: `tests/unit/test_models.py`
2. Import all models from `src.models.entities`
3. Write test for SmellAnnotation:
   - Create instance with valid data
   - Test JSON serialization
4. Write test for SmellDetection:
   - Test severity validation
   - Test confidence range validation
5. Write test for EvaluationResult:
   - Test nested structure with list
6. Run: `pytest tests/unit/test_models.py`

**Acceptance criteria**:
- [ ] Tests for all main models
- [ ] Validation tested
- [ ] JSON serialization tested
- [ ] All tests pass

**Dependencies**: P2-MODELS-004

---

#### Task P7-TESTS-004: Create mock data fixtures
**Estimated time**: 3 hours  
**Estimated LOC**: 150 lines

**Description**: Create pytest fixtures with mock data for testing.

**Steps**:
1. Create file: `tests/conftest.py`
2. Import: `import pytest`
3. Import models
4. Create fixture `sample_code()`: Returns sample Java code string
5. Create fixture `mock_smell_detection()`: Returns SmellDetection object
6. Create fixture `mock_ground_truth()`: Returns list of SmellAnnotation
7. Create fixture `mock_dacos_sample()`: Returns DACOSSample object
8. Test: Use fixture in a simple test

**Acceptance criteria**:
- [ ] Fixtures defined in conftest.py
- [ ] Fixtures return valid mock data
- [ ] Can be used in other test files
- [ ] Data is realistic

**Dependencies**: P2-MODELS-004

---

## Summary

**Total Tasks**: 54  
**Estimated Total Time**: 125-135 hours  
**Phases**: 7  
**Components**: 11

### Completion Checklist

**Phase 1 - Infrastructure**: [ ] (8 tasks)  
**Phase 2 - Data Layer**: [ ] (11 tasks)  
**Phase 3 - LLM Components**: [ ] (9 tasks)  
**Phase 4 - Pipeline Integration**: [ ] (7 tasks)  
**Phase 5 - Evaluation Framework**: [ ] (3 tasks - MLflow batch evaluation)  
**Phase 6 - Configuration & Documentation**: [ ] (3 tasks)  
**Phase 7 - Testing**: [ ] (4 tasks)

### Critical Path

1. P1-CONFIG-001 → P1-CONFIG-002 (Dependencies setup)
2. P2-MODELS-001 → P2-MODELS-004 (Data models)
3. P2-DB-001 → P2-DB-003 (Database access)
4. P2-GIT-001 → P2-GIT-003 (Git operations)
5. P3-VECTOR-001 → P3-VECTOR-003 (Vector DB)
6. P3-LLM-001 → P3-LLM-003 (Detector)
7. P3-JUDGE-001 → P3-JUDGE-002 (Judge)
8. P4-PIPELINE-001 → P4-PIPELINE-007 (Pipeline with MLflow tracing)
9. P5-MLFLOW-001 → P5-MLFLOW-003 (Batch evaluation and analysis)

### Notes for agent

1. **Read documentation first**: Check Architecture and Tech Stack docs before starting
1a. When implementing pipeline with sonarqube always double-check the referential implementation in pipeline_reference/pipeline.py
2. **One task at a time**: Complete and test each task before moving to next
3. **Test immediately**: Run tests after completing each task
4. **Ask questions**: If task is ambiguous, ask for clarification
5. **Commit frequently**: Commit after each completed task
6. **Check dependencies**: Ensure prerequisite tasks are complete
7. **Use type hints**: Python 3.11+ requires proper type annotations
8. **Handle errors**: Every function should handle potential errors gracefully