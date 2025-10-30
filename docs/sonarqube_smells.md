# SonarQube Local Workflow & Baseline Consensus

This consensus consolidates the local workflow (previous SonarQube smells doc) and baseline design (infra/sonarqube/SONARQUBE_PIPELINE_DESIGN.md). It defines two distinct usages:

1. Baseline (evaluation-only, isolated) – automated batch scanning of selected samples for comparison with LLM detection.
2. Ad‑hoc / developer workflow – manual local scanning to pull raw issues and map to internal smell taxonomy.

## Separation of Concerns
- Baseline results live in: eval_results/sonarqube_baseline/
- Do not inject baseline data into the main LangGraph pipeline directly.
- LLM pipeline may optionally consume normalized issue JSON for enrichment experiments, but this is a secondary step.

## Container Startup
```bash
docker compose -f infra/sonarqube/docker-compose.yml up -d
# optional health check
curl -u admin:admin http://localhost:9000/api/system/health
```

## Authentication
Create a user token (My Account → Security). Export:
```bash
export SONAR_TOKEN="<token>"
```
Use basic auth (`-u $SONAR_TOKEN:`) for API calls or `-Dsonar.login` for scanner.

## Project Scan (Manual Workflow)
Create sonar-project.properties at repo root:
```
sonar.projectKey=<key>
sonar.projectName=<name>
sonar.sources=.
sonar.host.url=http://localhost:9000
sonar.login=${SONAR_TOKEN}
```
Run:
```bash
sonar-scanner
```
Poll CE task for completion (see baseline script below).

## Target Java Code Smell Rule Set (Consensus)
| Conceptual Smell | Sonar Rule | Rationale |
| ---------------- | ---------- | --------- |
| Complex Method | java:S1541 | High cyclomatic complexity |
| Long Method | java:S138 | Excessive LOC |
| Long Parameter List | java:S107 | Many parameters reduce readability |
| Conditional Complexity | java:S1067 | Too many conditions branches |
| God Class (approx) | java:S1200 | Too many responsibilities / members |
| Large Class (alt heuristic) | java:S110 | Too many methods |
| Duplicated Conditions | java:S1871 | Repetition indicates poorer design |
| Print Statements / Poor Logging | java:S106 | Noisy console output |

(Feature Envy requires heuristic / LLM reasoning – not directly mapped.)

## Severity Mapping (Internal)
BLOCKER, CRITICAL -> HIGH
MAJOR -> MEDIUM
MINOR, INFO -> LOW

## API Retrieval & Filtering (General Pattern)
```bash
curl -u $SONAR_TOKEN: "http://localhost:9000/api/issues/search?componentKeys=$PROJECT_KEY&types=CODE_SMELL&rules=java:S1541,java:S138,java:S107,java:S1067,java:S1200&ps=500" > issues.json
```
Add facets or statuses if needed:
`&statuses=OPEN,CONFIRMED&facets=rules`.

## Normalization Schema
```json
{
  "smell_type": "Complex Method",
  "location": "src/main/java/Example.java:45 (methodName)",
  "severity": "HIGH",
  "description": "Method has cyclomatic complexity of 15",
  "refactoring_suggestion": "Reduce complexity via extraction",
  "confidence": 1.0,
  "rule": "java:S1541"
}
```

## Baseline Automation Script
A Python script (infra/sonarqube/baseline_scan.py) will:
1. Resolve latest commit BEFORE a cutoff date (e.g. before 2024-01-01 for experiment).
2. Clone repository at that commit to /tmp/scan_<project_key>.
3. Generate sonar-project.properties.
4. Run sonar-scanner (unless --dry-run).
5. Poll CE task until SUCCESS or timeout.
6. Fetch issues for selected rule set.
7. Map severities & rules to internal smell taxonomy.
8. Emit JSON to eval_results/sonarqube_baseline/<project_key>.json.

Supports: `--dry-run` (clone + commit resolution only) for environments lacking scanner.

## Expanding Higher-Level Smells
Aggregate multiple rules + metrics (LOC, number of methods) to approximate God Class or Feature Envy, then optionally feed into LLM for reasoning-based confirmation/refinement.

## Troubleshooting Quicklist
- Empty issues: confirm project key matches sonar.projectKey and analysis complete.
- Missing rules: verify language plugin loaded; restart container.
- Authentication failures: token format `-u TOKEN:` (blank password).
- Long queue times: inspect `api/ce/activity` for backlog.

## Minimal Steps Recap
1. Start Docker SonarQube.
2. Export SONAR_TOKEN.
3. Run baseline_scan.py (optionally --dry-run first).
4. Inspect output JSON in eval_results/sonarqube_baseline/.
5. Compare with LLM detections.

## References
- SonarQube Web API: http://localhost:9000/web_api
- Java Rules: https://rules.sonarsource.com/java/type/Code%20Smell/
- Baseline design: infra/sonarqube/SONARQUBE_PIPELINE_DESIGN.md
