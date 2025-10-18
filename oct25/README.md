# SonarQube Code Analysis Shell Script

This shell script automates the process of analyzing code repositories using local SonarQube and stores results in Google Drive. It's designed to work with datasets containing code smell annotations and compare them with SonarQube findings.

## Features

- 🔄 **Repository Management**: Clones repos and restores them to specific dates
- 🔍 **SonarQube Integration**: Runs local SonarQube analysis and fetches results via REST API
- 📊 **Analysis Comparison**: Compares SonarQube findings with manual code smell annotations
- ☁️ **Google Drive Integration**: Automatically uploads results to Google Drive
- 📁 **Dataset Support**: Works with structured JSON datasets of code smell annotations
- 🎯 **Arthas Default**: Pre-configured for the Alibaba Arthas repository analysis

## Prerequisites

### Dependencies
- **git**: Version control
- **curl**: HTTP requests
- **jq**: JSON processing
- **python3**: Python runtime
- **sonar-scanner**: SonarQube analysis tool

### SonarQube Setup
1. **Local SonarQube server** running on `http://localhost:9000`
2. **Authentication token** from SonarQube
3. **Project** created in SonarQube

### Installation

#### SonarQube Scanner
```bash
# Download and install SonarQube Scanner
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.8.0.2856-linux.zip
unzip sonar-scanner-cli-4.8.0.2856-linux.zip
sudo mv sonar-scanner-4.8.0.2856-linux /opt/sonar-scanner
sudo ln -s /opt/sonar-scanner/bin/sonar-scanner /usr/local/bin/sonar-scanner
```

#### Other Dependencies (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install git curl jq python3
```

#### Other Dependencies (macOS)
```bash
brew install git curl jq python3
```

## Configuration

### Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here
SONAR_TOKEN=your_sonarqube_token_here

# Optional (defaults provided)
SONAR_URL=http://localhost:9000
SONAR_PROJECT_KEY=arthas-analysis
```

### Getting SonarQube Token

1. Login to your SonarQube instance
2. Go to **User > My Account > Security**
3. Generate a new token
4. Copy the token to your `.env` file

### Getting Google API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Google Drive API
3. Create credentials (API Key)
4. Copy the key to your `.env` file

## Usage

### Basic Usage

```bash
# Analyze with default settings (arthas repo)
./sonarqube_analysis.sh -k arthas-project -t squ_abc123456789
```

### Advanced Usage

```bash
# Analyze specific repository and date
./sonarqube_analysis.sh \
  -r https://github.com/user/repo.git \
  -d 2023-06-15 \
  -b main \
  -k my-project \
  -t squ_abc123456789

# Use custom dataset
./sonarqube_analysis.sh \
  -f custom_dataset.json \
  -k my-project \
  -t squ_abc123456789
```

### Command Line Options

```
-r, --repo URL          Repository URL (default: alibaba/arthas)
-b, --branch BRANCH     Branch to analyze (default: master)
-d, --date DATE         Date to restore repo to (YYYY-MM-DD, default: 2022-01-01)
-k, --project-key KEY   SonarQube project key (required)
-t, --token TOKEN       SonarQube authentication token (required)
-u, --url URL           SonarQube server URL (default: http://localhost:9000)
-f, --dataset FILE      Dataset file with code smell annotations
-h, --help              Show help message
```

## Dataset Format

The script uses JSON datasets with code smell annotations. Here's the format:

```json
{
  "metadata": {
    "name": "arthas-code-smells",
    "description": "Code smell annotations for Alibaba Arthas",
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
          "smell_type": "GOD_CLASS",
          "location": "entire class",
          "severity": "HIGH",
          "description": "Class has too many responsibilities"
        }
      ]
    }
  ]
}
```

### Supported Code Smell Types

Based on the research repository used:
- `LONG_METHOD`
- `GOD_CLASS`
- `CONDITIONAL_COMPLEXITY`
- `DUPLICATED_CODE`
- `MAGIC_NUMBER`
- `LONG_PARAMETER_LIST`
- `UTILITY_CLASS`
- `PRIMITIVE_OBSESSION`
- And many more...

## Output

### Generated Files

The script generates several output files:

1. **SonarQube Results** (`sonarqube_results_TIMESTAMP.json`)
   - Raw SonarQube API responses
   - Issues, metrics, and metadata

2. **Analysis Report** (`analysis_report_TIMESTAMP.json`)
   - Comparison between SonarQube and dataset
   - Summary statistics
   - File-level analysis

3. **Google Drive Uploads**
   - Both files uploaded with timestamps
   - Accessible via your Google Drive

### Sample Output Structure

```json
{
  "metadata": {
    "project_key": "my-project",
    "analysis_date": "2024-10-16T14:30:00Z",
    "sonarqube_url": "http://localhost:9000"
  },
  "issues": [
    {
      "key": "issue-key-123",
      "rule": "java:S1142",
      "severity": "MAJOR",
      "component": "src/main/java/MyClass.java",
      "line": 45,
      "message": "Methods should not have too many parameters",
      "type": "CODE_SMELL"
    }
  ],
  "metrics": {
    "bugs": "0",
    "vulnerabilities": "2",
    "code_smells": "15",
    "coverage": "78.5",
    "ncloc": "1250"
  }
}
```

## Workflow

1. **Setup**: Check dependencies and SonarQube connectivity
2. **Repository Preparation**: Clone repo and checkout specific date/commit
3. **SonarQube Analysis**: Run scanner and upload results
4. **Data Collection**: Fetch issues and metrics via REST API
5. **Report Generation**: Compare with dataset annotations
6. **Upload**: Store results in Google Drive
7. **Cleanup**: Remove temporary files (optional)

## Examples

### Analyze Arthas (Default)

```bash
# Setup environment
cp .env.example .env
# Edit .env with your tokens

# Run analysis
./sonarqube_analysis.sh -k arthas-2022 -t squ_your_token_here
```

### Analyze Custom Repository

```bash
# Create custom dataset
cat > my_dataset.json << 'EOF'
{
  "metadata": {
    "name": "my-project-smells",
    "repository": "https://github.com/myuser/myproject.git"
  },
  "files": [
    {
      "file_path": "src/main/java/com/example/Service.java",
      "smells": [
        {
          "smell_type": "GOD_CLASS",
          "location": "entire class",
          "severity": "HIGH"
        },
        {
          "smell_type": "LONG_METHOD",
          "location": "processRequest() method",
          "severity": "MEDIUM"
        }
      ]
    }
  ]
}
EOF

# Run analysis
./sonarqube_analysis.sh \
  -r https://github.com/myuser/myproject.git \
  -d 2023-12-01 \
  -f my_dataset.json \
  -k my-project \
  -t squ_your_token_here
```

## Troubleshooting

### Common Issues

1. **SonarQube Connection Failed**
   ```bash
   # Check if SonarQube is running
   curl http://localhost:9000/api/system/status

   # Check firewall/ports
   sudo netstat -tlnp | grep 9000
   ```

2. **Authentication Failed**
   ```bash
   # Test token manually
   curl -u "your_token:" http://localhost:9000/api/authentication/validate
   ```

3. **Scanner Not Found**
   ```bash
   # Check if sonar-scanner is in PATH
   which sonar-scanner

   # Check installation
   sonar-scanner --version
   ```

4. **Git Issues**
   ```bash
   # Check git access
   git clone https://github.com/alibaba/arthas.git test-clone
   rm -rf test-clone
   ```

### Error Codes

- **Exit 1**: Missing dependencies or configuration
- **Exit 2**: SonarQube connectivity issues
- **Exit 3**: Repository access problems
- **Exit 4**: Analysis execution failed

## Contributing

To extend the script:

1. **Add new analysis types**: Modify the metrics fetching section
2. **Support new repositories**: Update the dataset format
3. **Add new output formats**: Extend the report generation
4. **Improve Google Drive integration**: Implement full OAuth2 flow

## License

This script is provided as-is for research and educational purposes. Please ensure compliance with SonarQube and Google Drive API terms of service.

## References

- [SonarQube REST API Documentation](https://docs.sonarqube.org/latest/extend/web-api/)
- [SonarQube Scanner Documentation](https://docs.sonarqube.org/latest/analysis/scan/sonarscanner/)
- [Google Drive API Documentation](https://developers.google.com/drive/api)
- [Alibaba Arthas Repository](https://github.com/alibaba/arthas)
- [Code Smells Research Repository](https://github.com/Luzkan/smells)