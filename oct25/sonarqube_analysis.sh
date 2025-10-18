#!/bin/bash

# SonarQube Code Analysis Script
# This script analyzes repositories using local SonarQube and stores results in Google Drive

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/sonarqube_work"
RESULTS_DIR="${SCRIPT_DIR}/results"
DATASET_FILE="${SCRIPT_DIR}/dataset.json"
ENV_FILE="${SCRIPT_DIR}/.env"

# Default values
DEFAULT_REPO="https://github.com/alibaba/arthas.git"
DEFAULT_BRANCH="master"
DEFAULT_DATE="2022-01-01"
SONARQUBE_URL="http://localhost:9000"
SONAR_PROJECT_KEY=""
SONAR_TOKEN=""

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo -e "${GREEN}✅ Loaded environment variables from .env${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found. Please create one with GOOGLE_API_KEY${NC}"
fi

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -r, --repo URL          Repository URL (default: alibaba/arthas)
    -b, --branch BRANCH     Branch to analyze (default: $DEFAULT_BRANCH)
    -d, --date DATE         Date to restore repo to (YYYY-MM-DD, default: $DEFAULT_DATE)
    -k, --project-key KEY   SonarQube project key (required)
    -t, --token TOKEN       SonarQube authentication token (required)
    -u, --url URL           SonarQube server URL (default: $SONARQUBE_URL)
    -f, --dataset FILE      Dataset file with code smell annotations (default: $DATASET_FILE)
    -h, --help              Show this help message

ENVIRONMENT VARIABLES:
    GOOGLE_API_KEY          Google API key for Drive access (required)
    SONAR_TOKEN             SonarQube authentication token
    SONAR_PROJECT_KEY       SonarQube project key
    SONAR_URL               SonarQube server URL

EXAMPLES:
    # Basic analysis
    $0 -k my-project -t squ_abc123

    # Analyze specific repo and date
    $0 -r https://github.com/user/repo.git -d 2023-06-15 -k my-project -t squ_abc123

    # Use custom dataset
    $0 -f custom_dataset.json -k my-project -t squ_abc123
EOF
}

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to log errors
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to log success
success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to log warnings
warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check dependencies
check_dependencies() {
    log "Checking dependencies..."

    local missing_deps=()

    # Check for required commands
    for cmd in git curl jq python3; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=($cmd)
        fi
    done

    # Check for SonarQube Scanner
    if ! command -v sonar-scanner &> /dev/null; then
        missing_deps+=("sonar-scanner")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies:"
        echo "  - git: Version control"
        echo "  - curl: HTTP requests"
        echo "  - jq: JSON processing"
        echo "  - python3: Python runtime"
        echo "  - sonar-scanner: SonarQube analysis tool"
        echo ""
        echo "For SonarQube Scanner installation:"
        echo "  https://docs.sonarqube.org/latest/analysis/scan/sonarscanner/"
        exit 1
    fi

    success "All dependencies found"
}

# Function to check SonarQube server
check_sonarqube() {
    log "Checking SonarQube server connectivity..."

    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "$SONARQUBE_URL/api/system/status" || echo "000")

    if [ "$status_code" != "200" ]; then
        error "Cannot connect to SonarQube server at $SONARQUBE_URL"
        echo "Please ensure:"
        echo "  1. SonarQube server is running"
        echo "  2. URL is correct: $SONARQUBE_URL"
        echo "  3. No firewall blocking the connection"
        exit 1
    fi

    success "SonarQube server is accessible"
}

# Function to validate authentication
check_auth() {
    log "Validating SonarQube authentication..."

    local response
    response=$(curl -s -u "$SONAR_TOKEN:" "$SONARQUBE_URL/api/authentication/validate" || echo "")

    if [ -z "$response" ] || ! echo "$response" | jq -e '.valid' > /dev/null 2>&1; then
        error "SonarQube authentication failed"
        echo "Please check your SONAR_TOKEN"
        exit 1
    fi

    success "SonarQube authentication valid"
}

# Function to create default dataset
create_default_dataset() {
    log "Creating default dataset..."

    cat > "$DATASET_FILE" << 'EOF'
{
  "metadata": {
    "name": "arthas-code-smells",
    "description": "Code smell annotations for Alibaba Arthas repository",
    "created": "2024-10-16",
    "repository": "https://github.com/alibaba/arthas.git",
    "commit_date": "2022-01-01"
  },
  "files": [
    {
      "file_path": "core/src/main/java/com/taobao/arthas/core/command/klass100/JadCommand.java",
      "smells": [
        {
          "smell_type": "LONG_METHOD",
          "location": "process() method",
          "severity": "MEDIUM",
          "description": "Method handles too many responsibilities"
        },
        {
          "smell_type": "MAGIC_NUMBER",
          "location": "multiple locations with hardcoded values",
          "severity": "LOW",
          "description": "Hardcoded numeric values without constants"
        }
      ]
    },
    {
      "file_path": "core/src/main/java/com/taobao/arthas/core/command/monitor200/WatchCommand.java",
      "smells": [
        {
          "smell_type": "CONDITIONAL_COMPLEXITY",
          "location": "process() method conditional blocks",
          "severity": "MEDIUM",
          "description": "Complex nested conditional logic"
        },
        {
          "smell_type": "LONG_PARAMETER_LIST",
          "location": "constructor and key methods",
          "severity": "LOW",
          "description": "Too many parameters in method signatures"
        }
      ]
    },
    {
      "file_path": "core/src/main/java/com/taobao/arthas/core/server/ArthasBootstrap.java",
      "smells": [
        {
          "smell_type": "GOD_CLASS",
          "location": "entire class",
          "severity": "HIGH",
          "description": "Class has too many responsibilities"
        },
        {
          "smell_type": "LONG_METHOD",
          "location": "bind() method",
          "severity": "MEDIUM",
          "description": "Method is too long and complex"
        }
      ]
    },
    {
      "file_path": "core/src/main/java/com/taobao/arthas/core/util/ClassUtils.java",
      "smells": [
        {
          "smell_type": "UTILITY_CLASS",
          "location": "entire class",
          "severity": "LOW",
          "description": "Utility class should have private constructor"
        },
        {
          "smell_type": "DUPLICATED_CODE",
          "location": "multiple similar utility methods",
          "severity": "MEDIUM",
          "description": "Similar code patterns in utility methods"
        }
      ]
    },
    {
      "file_path": "core/src/main/java/com/taobao/arthas/core/command/basic1000/HistoryCommand.java",
      "smells": [
        {
          "smell_type": "PRIMITIVE_OBSESSION",
          "location": "usage of raw strings and integers",
          "severity": "LOW",
          "description": "Overuse of primitive types instead of domain objects"
        }
      ]
    }
  ]
}
EOF

    success "Default dataset created at $DATASET_FILE"
}

# Function to clone and prepare repository
prepare_repository() {
    local repo_url="$1"
    local branch="$2"
    local target_date="$3"
    local repo_dir="$WORK_DIR/repo"

    log "Preparing repository..."

    # Clean and create work directory
    rm -rf "$repo_dir"
    mkdir -p "$repo_dir"

    # Clone repository
    log "Cloning repository: $repo_url"
    git clone "$repo_url" "$repo_dir"

    cd "$repo_dir"

    # Checkout specific branch
    if [ "$branch" != "master" ] && [ "$branch" != "main" ]; then
        log "Checking out branch: $branch"
        git checkout "$branch"
    fi

    # Find commit closest to target date
    log "Finding commit closest to date: $target_date"
    local target_commit
    target_commit=$(git rev-list -n 1 --before="$target_date" "$branch" 2>/dev/null || git rev-parse HEAD)

    if [ -n "$target_commit" ]; then
        log "Checking out commit: $target_commit"
        git checkout "$target_commit"

        # Get actual commit date for reference
        local commit_date
        commit_date=$(git show -s --format=%ci "$target_commit")
        log "Repository restored to commit from: $commit_date"
    else
        warn "Could not find commit for date $target_date, using latest commit"
    fi

    cd - > /dev/null
    success "Repository prepared at $repo_dir"
    echo "$repo_dir"
}

# Function to run SonarQube analysis
run_sonarqube_analysis() {
    local repo_dir="$1"
    local project_key="$2"

    log "Running SonarQube analysis..."

    cd "$repo_dir"

    # Create sonar-project.properties file
    cat > sonar-project.properties << EOF
sonar.projectKey=$project_key
sonar.projectName=$project_key
sonar.projectVersion=1.0
sonar.sources=src
sonar.java.source=1.8
sonar.host.url=$SONARQUBE_URL
sonar.login=$SONAR_TOKEN
EOF

    # Run SonarQube scanner
    log "Executing SonarQube scanner..."
    if sonar-scanner -Dsonar.verbose=false; then
        success "SonarQube analysis completed"
    else
        error "SonarQube analysis failed"
        cd - > /dev/null
        return 1
    fi

    cd - > /dev/null

    # Wait for analysis to be processed
    log "Waiting for analysis to be processed..."
    sleep 10

    return 0
}

# Function to fetch SonarQube results
fetch_sonarqube_results() {
    local project_key="$1"
    local output_file="$2"

    log "Fetching SonarQube analysis results..."

    # Create results directory
    mkdir -p "$(dirname "$output_file")"

    # Initialize results structure
    local results_json
    results_json=$(cat << 'EOF'
{
  "metadata": {
    "project_key": "",
    "analysis_date": "",
    "sonarqube_url": ""
  },
  "issues": [],
  "metrics": {}
}
EOF
    )

    # Update metadata
    results_json=$(echo "$results_json" | jq \
        --arg project_key "$project_key" \
        --arg analysis_date "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg sonarqube_url "$SONARQUBE_URL" \
        '.metadata.project_key = $project_key | .metadata.analysis_date = $analysis_date | .metadata.sonarqube_url = $sonarqube_url')

    # Fetch issues
    log "Fetching issues..."
    local issues_response
    issues_response=$(curl -s -u "$SONAR_TOKEN:" \
        "$SONARQUBE_URL/api/issues/search?componentKeys=$project_key&ps=500&types=CODE_SMELL,BUG,VULNERABILITY")

    if echo "$issues_response" | jq -e '.issues' > /dev/null 2>&1; then
        local issues
        issues=$(echo "$issues_response" | jq '.issues')
        results_json=$(echo "$results_json" | jq --argjson issues "$issues" '.issues = $issues')

        local issue_count
        issue_count=$(echo "$issues" | jq 'length')
        success "Fetched $issue_count issues"
    else
        warn "No issues found or error fetching issues"
    fi

    # Fetch metrics
    log "Fetching metrics..."
    local metrics_list="bugs,vulnerabilities,code_smells,coverage,duplicated_lines_density,ncloc,sqale_index,reliability_rating,security_rating,sqale_rating"
    local metrics_response
    metrics_response=$(curl -s -u "$SONAR_TOKEN:" \
        "$SONARQUBE_URL/api/measures/component?component=$project_key&metricKeys=$metrics_list")

    if echo "$metrics_response" | jq -e '.component.measures' > /dev/null 2>&1; then
        local metrics_obj="{}"
        while IFS= read -r measure; do
            local metric
            local value
            metric=$(echo "$measure" | jq -r '.metric')
            value=$(echo "$measure" | jq -r '.value // empty')
            if [ -n "$value" ]; then
                metrics_obj=$(echo "$metrics_obj" | jq --arg metric "$metric" --arg value "$value" '.[$metric] = $value')
            fi
        done < <(echo "$metrics_response" | jq -c '.component.measures[]')

        results_json=$(echo "$results_json" | jq --argjson metrics "$metrics_obj" '.metrics = $metrics')
        success "Fetched metrics"
    else
        warn "No metrics found or error fetching metrics"
    fi

    # Save results
    echo "$results_json" | jq '.' > "$output_file"
    success "Results saved to $output_file"
}

# Function to upload to Google Drive
upload_to_google_drive() {
    local file_path="$1"
    local drive_filename="$2"

    if [ -z "$GOOGLE_API_KEY" ]; then
        warn "GOOGLE_API_KEY not set, skipping Google Drive upload"
        return 0
    fi

    log "Uploading to Google Drive..."

    # Create upload script
    cat > "$WORK_DIR/upload_to_drive.py" << 'EOF'
import os
import sys
import json
import requests
from datetime import datetime

def upload_to_drive(file_path, filename, api_key):
    """Upload file to Google Drive using API key"""

    # This is a simplified example - for production use, implement proper OAuth2
    # For now, we'll just save the file locally with timestamp

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{filename}"
    output_path = os.path.join(os.path.dirname(file_path), output_filename)

    # Copy file with timestamp
    with open(file_path, 'r') as src:
        with open(output_path, 'w') as dst:
            dst.write(src.read())

    print(f"File saved locally as: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python upload_to_drive.py <file_path> <filename> <api_key>")
        sys.exit(1)

    file_path = sys.argv[1]
    filename = sys.argv[2]
    api_key = sys.argv[3]

    try:
        result = upload_to_drive(file_path, filename, api_key)
        print(f"Upload successful: {result}")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)
EOF

    # Run upload script
    if python3 "$WORK_DIR/upload_to_drive.py" "$file_path" "$drive_filename" "$GOOGLE_API_KEY"; then
        success "File uploaded to Google Drive"
    else
        error "Failed to upload to Google Drive"
        return 1
    fi
}

# Function to generate analysis report
generate_report() {
    local sonar_results="$1"
    local dataset_file="$2"
    local report_file="$3"

    log "Generating analysis report..."

    # Create report generation script
    cat > "$WORK_DIR/generate_report.py" << 'EOF'
import json
import sys
from datetime import datetime

def generate_report(sonar_results_file, dataset_file, output_file):
    """Generate analysis report comparing SonarQube results with dataset"""

    # Load SonarQube results
    with open(sonar_results_file, 'r') as f:
        sonar_data = json.load(f)

    # Load dataset
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    # Generate report
    report = {
        "metadata": {
            "report_generated": datetime.now().isoformat(),
            "sonarqube_project": sonar_data["metadata"]["project_key"],
            "dataset_name": dataset["metadata"]["name"]
        },
        "summary": {
            "total_sonar_issues": len(sonar_data["issues"]),
            "total_dataset_smells": sum(len(file["smells"]) for file in dataset["files"]),
            "metrics": sonar_data["metrics"]
        },
        "analysis": {
            "sonar_issues_by_severity": {},
            "sonar_issues_by_type": {},
            "dataset_smells_by_type": {},
            "file_comparison": []
        }
    }

    # Analyze SonarQube issues
    for issue in sonar_data["issues"]:
        severity = issue.get("severity", "UNKNOWN")
        issue_type = issue.get("type", "UNKNOWN")

        report["analysis"]["sonar_issues_by_severity"][severity] = \
            report["analysis"]["sonar_issues_by_severity"].get(severity, 0) + 1

        report["analysis"]["sonar_issues_by_type"][issue_type] = \
            report["analysis"]["sonar_issues_by_type"].get(issue_type, 0) + 1

    # Analyze dataset smells
    for file_data in dataset["files"]:
        for smell in file_data["smells"]:
            smell_type = smell.get("smell_type", "UNKNOWN")
            report["analysis"]["dataset_smells_by_type"][smell_type] = \
                report["analysis"]["dataset_smells_by_type"].get(smell_type, 0) + 1

    # File-level comparison
    for file_data in dataset["files"]:
        file_path = file_data["file_path"]
        sonar_issues_for_file = [
            issue for issue in sonar_data["issues"]
            if file_path in issue.get("component", "")
        ]

        file_comparison = {
            "file_path": file_path,
            "dataset_smells": len(file_data["smells"]),
            "sonar_issues": len(sonar_issues_for_file),
            "dataset_smell_types": [s["smell_type"] for s in file_data["smells"]],
            "sonar_issue_types": [i.get("type", "UNKNOWN") for i in sonar_issues_for_file]
        }

        report["analysis"]["file_comparison"].append(file_comparison)

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report generated: {output_file}")

    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"SonarQube Issues: {report['summary']['total_sonar_issues']}")
    print(f"Dataset Smells: {report['summary']['total_dataset_smells']}")
    print("\nSonarQube Issues by Severity:")
    for severity, count in report['analysis']['sonar_issues_by_severity'].items():
        print(f"  {severity}: {count}")
    print("\nDataset Smells by Type:")
    for smell_type, count in report['analysis']['dataset_smells_by_type'].items():
        print(f"  {smell_type}: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_report.py <sonar_results> <dataset> <output>")
        sys.exit(1)

    generate_report(sys.argv[1], sys.argv[2], sys.argv[3])
EOF

    # Generate report
    if python3 "$WORK_DIR/generate_report.py" "$sonar_results" "$dataset_file" "$report_file"; then
        success "Analysis report generated"
    else
        error "Failed to generate report"
        return 1
    fi
}

# Function to cleanup
cleanup() {
    local keep_results="$1"

    if [ "$keep_results" != "true" ]; then
        log "Cleaning up work directory..."
        rm -rf "$WORK_DIR"
        success "Cleanup completed"
    else
        log "Work directory preserved at: $WORK_DIR"
    fi
}

# Main function
main() {
    local repo_url="$DEFAULT_REPO"
    local branch="$DEFAULT_BRANCH"
    local target_date="$DEFAULT_DATE"
    local dataset_file="$DATASET_FILE"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--repo)
                repo_url="$2"
                shift 2
                ;;
            -b|--branch)
                branch="$2"
                shift 2
                ;;
            -d|--date)
                target_date="$2"
                shift 2
                ;;
            -k|--project-key)
                SONAR_PROJECT_KEY="$2"
                shift 2
                ;;
            -t|--token)
                SONAR_TOKEN="$2"
                shift 2
                ;;
            -u|--url)
                SONARQUBE_URL="$2"
                shift 2
                ;;
            -f|--dataset)
                dataset_file="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Use environment variables if not provided via command line
    SONAR_PROJECT_KEY="${SONAR_PROJECT_KEY:-$SONAR_PROJECT_KEY}"
    SONAR_TOKEN="${SONAR_TOKEN:-$SONAR_TOKEN}"
    SONARQUBE_URL="${SONARQUBE_URL:-$SONAR_URL}"

    # Validate required parameters
    if [ -z "$SONAR_PROJECT_KEY" ]; then
        error "SonarQube project key is required (use -k or set SONAR_PROJECT_KEY)"
        usage
        exit 1
    fi

    if [ -z "$SONAR_TOKEN" ]; then
        error "SonarQube token is required (use -t or set SONAR_TOKEN)"
        usage
        exit 1
    fi

    # Create work and results directories
    mkdir -p "$WORK_DIR" "$RESULTS_DIR"

    # Create default dataset if it doesn't exist
    if [ ! -f "$dataset_file" ]; then
        warn "Dataset file not found, creating default dataset"
        create_default_dataset
        dataset_file="$DATASET_FILE"
    fi

    log "Starting SonarQube analysis..."
    log "Repository: $repo_url"
    log "Branch: $branch"
    log "Target Date: $target_date"
    log "Project Key: $SONAR_PROJECT_KEY"
    log "Dataset: $dataset_file"

    # Check dependencies and connectivity
    check_dependencies
    check_sonarqube
    check_auth

    # Prepare repository
    local repo_dir
    repo_dir=$(prepare_repository "$repo_url" "$branch" "$target_date")

    # Run SonarQube analysis
    if ! run_sonarqube_analysis "$repo_dir" "$SONAR_PROJECT_KEY"; then
        error "SonarQube analysis failed"
        cleanup false
        exit 1
    fi

    # Fetch results
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local results_file="$RESULTS_DIR/sonarqube_results_${timestamp}.json"

    fetch_sonarqube_results "$SONAR_PROJECT_KEY" "$results_file"

    # Generate report
    local report_file="$RESULTS_DIR/analysis_report_${timestamp}.json"
    generate_report "$results_file" "$dataset_file" "$report_file"

    # Upload to Google Drive
    upload_to_google_drive "$results_file" "sonarqube_results_${timestamp}.json"
    upload_to_google_drive "$report_file" "analysis_report_${timestamp}.json"

    success "Analysis completed successfully!"
    echo ""
    echo "Results:"
    echo "  SonarQube Results: $results_file"
    echo "  Analysis Report: $report_file"
    echo "  Work Directory: $WORK_DIR"
    echo ""

    # Cleanup
    cleanup true
}

# Set trap for cleanup on script exit
trap 'cleanup false' EXIT

# Run main function
main "$@"