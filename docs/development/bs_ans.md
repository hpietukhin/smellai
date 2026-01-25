 ---
  Critical architecture questions (answer first - determines everything else)

  1. Dataset and research focus

  Q1.1: Is DACOS dataset used anywhere in the current codebase?
  - If yes: where and for what purpose? - now we dont use it but it is still important to access all datasets in a unified manner via Adapter pattern.
  - If no: should we delete all DACOS references from docs?
  - i dont think so, but just mention that we will use SWE-refactor instead

  Q1.2: Is the primary dataset RefactoringMiner 2.0?
  - Confirm: All evaluation uses RefactoringMiner JSON manifests?
  Ну вообще-то evaluation also need to use just some generic dataset records from model that is created via adapter in datasets/. If eval currently uses raw RefactoringMiner JSON manifests, we should hide this behind some abstraction layer.

  Q1.3: What is the actual research question being evaluated?
  - A) LLM smell detection accuracy (DACOS-based)?
  - B) LLM refactoring mapping accuracy (RefactoringMiner-based)?
  - C) Both?
  A multiagent system is used to evaluate the accuracy of LLM smell detection and refactoring suggestions.

  2. Agent architecture

  Q2.1: How many agents are actually implemented and working?
  - List each: name, location, status (working/placeholder/planned)
  

  Q2.2: Is there a multi-agent orchestration workflow that connects all 6 agents?
  - If yes: which file implements it?
  - If no: are agents currently run independently?
  They are run in sequence:
  4. agentic AI with LangGraph: A0: existuju testy kvoli refactoringu? Ak nie zavolaj A7. A1: vyhlada pachy cez Sonar, A2: spyta sa developera, ktore riesit, A3: usporiada ich, A4: vezme ten usporiadany zoznam a pripravi podla bodu 3 prompty pre jednotlive pachy alebo vyhlada v DB promptov, A5 v cykle bude refaktorovat, A6: skontroluje ci dobre zrefaktoroval a zdrojak stale funguje, ak nenajde test, poziada A7 o vygenerovanie testu

  Q2.3: Agent naming confusion - clarify:
  - Is "Agent 2: Test verification" the same as "Agent 6: Test execution"?
  - well, it can be the same agent, but i dont know, for now let it be several agents
  - Is "A0" a separate agent or just a step in the workflow?
  - this will be a separate agent that will look into branch coverage of code (it must deduce how to obtain this information) and then generate missing tests if they are not present.
  - Should we use "Agent 1-6" or "A0-A6" consistently?
  - use A0-A6

  3. MySQL vs JSON data storage

  Q3.1: Is MySQL used anywhere in the current codebase?
  - If no: delete all MySQL references from docs?
  - MySQL used to obtain DACOS dataset

  Q3.2: Where is ground truth data actually stored?
  - RefactoringMiner JSON files?
  - MLflow database (SQLite)?
  - Something else?
  - well i dont know, try to find out. but for DACOS it was in MySQL, but i dont remember whether i processed it once again.

  ---
  Implementation status questions

  4. Which components are real vs planned?

  Q4.1: DeepLake vector database:
  - Is it used in any current workflow?
  - If not, is it planned or should we remove from docs?
  - Deeplake is only planned.

  Q4.2: LLM Detector Module (tech_stack.md:195):
  - Does this module exist?
  - If not, what's the actual smell detection mechanism?
  - Currently we work in this workflow - datasets must provide location of smell and type of smell. Sonarqube integration will detect smells, but only certain types of smells are supported. The smells mapping is in RULE_NAME_MAP structure.

  Q4.3: Test generation agent (Agent 3):
  - Confirmed placeholder?
  - Any implementation planned or out of scope?
  Yes, we need implement it. but i dont know whether it is implemented or not.

  Q4.4: Sparse checkout for Git operations:
  - TECHNICAL_SPECIFICATION.md:1024 says "removed from design"
  - architecture.md:179-184 describes sparse checkout in detail
  - Which is correct?
  - we should completely abandon the idea of sparse checkout.

  ---
  Configuration and environment questions

  5. Required environment variables

  Q5.1: Which env vars are REQUIRED for basic workflow?
  - OPENAI_API_KEY or ANTHROPIC_API_KEY? (one or both?)
  - SONAR_TOKEN?
  - Others?
  - currently cerebras and openai

  Q5.2: Which env vars are OPTIONAL?
  - MLFLOW_TRACKING_URI?
  - WANDB_API_KEY?
  - we give up wandb completely, leaving only MLFLOW_TRACKING_URI

  Q5.3: Which env vars in README.md are wrong/obsolete?
  - CLASSES_CSV_PATH - delete?
  - REFACTORINGS_CSV_PATH - delete?
  - WANDB_PROJECT - keep or delete?
  - wandb wont be used, mlflow instead.

  Q5.4: W&B (Weights & Biases) integration:
  - Is W&B actually used in the project?
  - If yes: for what purpose?
  - If no: why is it in README.md lines 5-18?
It wont be used, mlflow instead.

  ---
  File structure questions

  6. Correct file paths

  Q6.1: Main evaluation workflow location:
  - Is it workflows/rminer_eval_workflow.py?
  - Or something else?
  - Yes, but i want them to be interchangeable. So i can use it with my exsisting MLFLow workflow.

  Q6.2: Do these directories exist?
  - src/ directory? (tech_stack.md says yes, exploration says no)
  - src/pipelines/?
  - src/agents/detector.py and src/agents/judge.py?
  - You can always check. src/ directory is absent now.

  Q6.3: Are there any "src/" imports in the actual code?
  - If yes: where?
  - If no: tech_stack.md structure is completely wrong?
  - i dont need src/

  ---
  Workflow and data flow questions

  7. End-to-end workflows

  Q7.1: What is the simplest working end-to-end workflow?
  - Start: what input?
  - Steps: which commands in order?
  - End: what output?
    I am working on this workflow:
  4. agentic AI with LangGraph: A0: existuju testy kvoli refactoringu? Ak nie zavolaj A7. A1: vyhlada pachy cez Sonar, A2: spyta sa developera, ktore riesit, A3: usporiada ich, A4: vezme ten usporiadany zoznam a pripravi podla bodu 3 prompty pre jednotlive pachy alebo vyhlada v DB promptov, A5 v cykle bude refaktorovat, A6: skontroluje ci dobre zrefaktoroval a zdrojak stale funguje, ak nenajde test, poziada A7 o vygenerovanie testu
  Also i need it to be modular and interchargable tools for my workflow facilitated with good interfaces.
  Q7.2: How does SonarQube data reach prioritization?
  - Trace: SonarQube scan → ? → dependency analysis agent
  - Concrete file/function names for each step?
  - I dont know

  Q7.3: Agent execution order:
  - Are agents run sequentially or independently?
  - If sequential: what's the exact order?
  - If independent: how are results combined?
- 4. agentic AI with LangGraph: A0: existuju testy kvoli refactoringu? Ak nie zavolaj A7. A1: vyhlada pachy cez Sonar, A2: spyta sa developera, ktore riesit, A3: usporiada ich, A4: vezme ten usporiadany zoznam a pripravi podla bodu 3 prompty pre jednotlive pachy alebo vyhlada v DB promptov, A5 v cykle bude refaktorovat, A6: skontroluje ci dobre zrefaktoroval a zdrojak stale funguje, ak nenajde test, poziada A7 o vygenerovanie testu
  

  ---
  Terminology questions

  8. Standardization

  Q8.1: Refactoring tool naming - pick one:
  - "RefactoringMiner" (formal)
  - "RMiner" (abbreviated)
  - "rminer" (directory/file naming)
- RefactoringMiner

  Q8.2: Code issues - pick one:
  - "Code Smell"
  - "Smell"
  - "Issue"
- Code Smell

  Q8.3: Code changes - pick one:
  - "Hunk"
  - "Diff"
  - "Change"
  - "Fragment"
- hunk

  Q8.4: Analysis process - pick one:
  - "Evaluation"
  - "Assessment"
  - "Analysis"
- Analysis

  ---
  Documentation strategy questions

  9. Which docs to keep/update/delete?

  Q9.1: architecture.md status:
  - A) Delete entirely (outdated DACOS design)
  - B) Mark as "archived/historical"
  - C) Rewrite to match current system
  - D) Keep as-is but add big warning at top
  - rewrite but say that DACOS is only 1 possible dataset

  Q9.2: TECHNICAL_SPECIFICATION.md:
  - Is this the single source of truth going forward?
  - Should all other docs defer to it?
  - yes it must contain actual info

  Q9.3: tech_stack.md:
  - Rewrite file structure section completely?
  - Or delete and merge into TECHNICAL_SPECIFICATION.md?
  - delete and merge into TECHNICAL_SPECIFICATION.md

  Q9.4: Multiple "getting started" docs:
  - README.md has quickstart
  - README_RMINER.md has quickstart
  - Should we consolidate into one canonical tutorial?
  - because datasets will be interchargeable, incorporate it into README.md

  ---
  Prioritization and scope questions

  10. TODOs and limitations

  Q10.1: The 35+ TODOs in TECHNICAL_SPECIFICATION.md:
  - Which are blockers for current use?
  - Which are future work only?
  - Should we add priority labels?
  - I dont know, we should discuss them

  Q10.2: Missing features status:
  - Cycle detection (line 406): needed now or later?
  - Token counting (line 348): needed now or later?
  - New dataset adapter (line 365): planned or speculative?
  - later for first 2, planned for third

  ---
  Testing and validation questions

  11. Current working state

  Q11.1: Has anyone successfully run the complete evaluation pipeline recently?
  - If yes: which exact commands did they use?
  - Can we use that as the "hello world" tutorial?
  - yes, see docs/react_agent_mlflow.md for instructions

  Q11.2: Which features have been tested and verified working?
  - SonarQube integration? - yes
  - RefactoringMiner evaluation? - yes
  - Dependency analysis? - not tested, dont know if ran
  - Java test agent? - not tested, dont know if ran
  - MLflow tracking? - yes

  Q11.3: What's the largest dataset successfully evaluated?
  - 5 samples? 100 samples? More?
  - dont know, does not matter

  ---
  Documentation audience questions

  12. Target readers

  Q12.1: Primary documentation audience:
  - A) Thesis committee (explain research)
  - B) Future maintainers (onboarding)
  - C) External researchers (reproduce experiments)
  - D) All of the above?
  - me and my coding agents. count them as a juniors that need strict guidance, because they have poor techical taste and like to produce some reimplementation of the codebase. so we will need to provide them with some code paths and relevant commands

  Q12.2: Should docs prioritize:
  - A) Conceptual understanding (why/how it works)
  - B) Practical usage (copy-paste commands)
  - C) Both equally?
  - both
