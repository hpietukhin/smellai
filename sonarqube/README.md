# SonarQube Integration

This folder contains tools for scanning code repositories with SonarQube to detect code smells and quality issues.

## Current Setup

### SonarQube Server Status
- **Running**: Yes (Docker container)
- **Container**: `smellai-sonarqube`
- **Image**: `sonarqube:10.6.0-community`
- **URL**: http://localhost:9000
- **Port**: 9000
- **Status**: UP
- **Created**: 2025-10-30
- **Uptime**: Currently up for 10 hours

### Architecture

```
┌─────────────────────────────────────────┐
│  Your Code (commit_scan.py, etc.)      │
│  ↓                                       │
│  sonar-scanner (local CLI)             │
│  ↓                                       │
│  SonarQube Server (localhost:9000)     │
│  ↓                                       │
│  PostgreSQL Database (embedded/H2)      │
└─────────────────────────────────────────┘
```

**How it works:**
1. **sonar-scanner CLI** (installed locally) scans your code
2. Downloads language analyzers from SonarQube Server
3. Analyzes code according to configured rules
4. Uploads results to SonarQube Server
5. Server processes and stores issues in database
6. View results in web UI at http://localhost:9000

### Storage Volumes
- `sonarqube_data`: Analysis data and database files
- `sonarqube_extensions`: Plugins and extensions
- `sonarqube_logs`: Server logs

**Note**: PostgreSQL container from docker-compose.yml is NOT running. SonarQube is using embedded database or different configuration.

## Files

### Core Scripts
- **commit_scan.py**: Scan specific commits or files at specific commits
- **baseline_scan.py**: Scan baseline/main branch
- **scan_commit_tool.py**: Tool wrapper for commit scanning

### Configuration
- **docker-compose.yml**: SonarQube + PostgreSQL setup (currently only SonarQube container running)
- **sonarqube_server.sh**: Server management script

### Analysis Configuration
- **RULE_NAME_MAP**: Maps SonarQube rule IDs to human-readable smell types
  - `java:S1541` → Complex Method
  - `java:S138` → Long Method
  - `java:S107` → Long Parameter List
  - `java:S1067` → Conditional Complexity
  - `java:S1200` → God Class
  - `java:S110` → Large Class
  - `java:S1871` → Duplicated Conditions
  - `java:S106` → Print Statements

## Usage

### Scan Commit with SonarQube

```bash
python -m sonarqube.commit_scan \
  --repo https://github.com/org/repo \
  --commit abc123 \
  --sonar-url http://localhost:9000 \
  --sonar-token YOUR_TOKEN
```

**Requirements:**
- `sonar-scanner` CLI must be installed locally and available in your PATH
- SonarQube Server must be running at the specified URL
- Valid SonarQube authentication token

**How it works:**
- Creates `sonar-project.properties` in the cloned repo
- Runs `sonar-scanner` command directly on your system
- Connects to SonarQube Server to download analyzers
- Uploads analysis results to server

### Scan Specific Files
```bash
python -m sonarqube.commit_scan \
  --repo https://github.com/org/repo \
  --commit abc123 \
  --file src/Main.java \
  --file src/Utils.java \
  --sonar-url http://localhost:9000 \
  --sonar-token YOUR_TOKEN
```

### With Caching
```bash
python -m sonarqube.commit_scan \
  --repo https://github.com/org/repo \
  --commit abc123 \
  --sonar-url http://localhost:9000 \
  --sonar-token YOUR_TOKEN \
  --cache-dir ./sonar_cache
```

### Baseline Scan
```bash
python -m sonarqube.baseline_scan \
  https://github.com/org/repo \
  --output ./eval_results
```

## Managing SonarQube Server

### Start Server
```bash
cd sonarqube
docker-compose up -d
```

### Stop Server
```bash
docker-compose down
```

### View Logs
```bash
docker logs smellai-sonarqube
```

### Check Status
```bash
# Check container
docker ps | grep sonarqube

# Check API
curl http://localhost:9000/api/system/status
```

### Access Web UI
Open http://localhost:9000 in your browser

Default credentials (change on first login):
- Username: `admin`
- Password: `admin`

## Environment Variables

Create a `.env` file in the project root:
```bash
SONAR_TOKEN=your_token_here
SONAR_URL=http://localhost:9000
```

Generate token: http://localhost:9000/account/security → Generate Tokens

## How Analysis Works

1. **Clone & Checkout**: Creates temp directory, clones repo, checks out commit
2. **Create Properties File**: Generates `sonar-project.properties` with:
   - Project key (derived from repo URL + commit SHA)
   - SonarQube Server URL
   - Authentication token
   - Source directories
3. **Scanner Execution**: Runs `sonar-scanner` CLI command
4. **Scanner Bootstrapping**:
   - Connects to SonarQube Server at `localhost:9000`
   - Downloads scanner engine (if not cached)
   - Detects project languages (Java, Python, etc.)
   - Downloads language analyzers (if not cached)
5. **Code Analysis**:
   - Scans source files according to quality profile
   - Detects issues based on configured rules
   - Generates analysis report
6. **Upload Results**: Sends report to SonarQube Server
7. **Background Processing**:
   - Server queues analysis for processing
   - Compute engine processes report
   - Calculates metrics and quality gate
   - Stores issues in database
8. **Polling**: Script polls until analysis completes (timeout: 600s)
9. **Fetch Issues**: Retrieves issues via REST API for requested files
10. **Normalize & Cache**: Converts to simplified format and caches results

## Cache Structure

When using `--cache-dir`:
- Single file: `{commit_sha}_{file_path}.json`
- Full scan: `{commit_sha}_full.json`

Example:
```
cache/
├── abc12345_src_Main.java.json
├── abc12345_src_Utils.java.json
└── abc12345_full.json
```

## Project Key Format

Each scan creates a unique project in SonarQube:
```
{org}_{repo}_{short_sha}
```

Example: `apache_commons-lang_abc12345`

## Dependencies

### Python
- `requests`: API calls to SonarQube
- `python-dotenv`: Environment variable management

### External Tools (Required)
- **sonar-scanner CLI**: Must be installed locally and in PATH
- **Git**: For cloning and checking out commits

### SonarQube Components
- **SonarQube Server**: Code analysis platform (running at localhost:9000)
- **Scanner Engine**: Downloaded at analysis time from server
- **Language Analyzers**: Java, Python, etc. (downloaded as needed from server)

### Installing sonar-scanner CLI

**macOS:**
```bash
brew install sonar-scanner
```

**Linux:**
Download from https://docs.sonarqube.org/latest/analysis/scan/sonarscanner/
Extract and add to PATH

**Verify installation:**
```bash
sonar-scanner --version
```

## Troubleshooting

### SonarQube not responding
```bash
docker logs smellai-sonarqube
docker restart smellai-sonarqube
```

### Scanner timeout
Increase timeout in `poll_analysis_completion()` (default: 600s)

### Scanner not found
```bash
# Verify sonar-scanner is installed
which sonar-scanner

# If not found, install it
# macOS
brew install sonar-scanner

# Linux
# Download from https://docs.sonarqube.org/latest/analysis/scan/sonarscanner/
# Extract and add bin/ directory to PATH
```

### Scanner fails to connect
- Verify SonarQube Server is running: `curl http://localhost:9000/api/system/status`
- Check SONAR_TOKEN is set correctly
- Check firewall/network settings

### Cache issues
Clear cache directory:
```bash
rm -rf ./sonar_cache
```

## Notes

- **SonarLint**: You have SonarLint VSCode extension installed (processes running)
  - Different from SonarQube Server
  - Provides real-time analysis in editor
  - Can connect to SonarQube Server for shared rules

- **Scanner Installation**:
  - Must have `sonar-scanner` CLI installed locally
  - Scanner is lightweight - it downloads analyzers from server at runtime
  - No need for Docker - direct CLI execution

- **Analysis Time**:
  - First scan takes longer (downloads analyzers)
  - Subsequent scans use cache for faster execution
  - Scanner cache typically stored in `~/.sonar/cache`

- **Rate Limiting**: Be cautious with many scans in short time

- **Local Scanner Benefits**:
  - Faster startup (no Docker overhead)
  - Simpler debugging
  - Direct access to logs and temp files
  - One less dependency to manage
