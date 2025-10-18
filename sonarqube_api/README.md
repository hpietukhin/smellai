# SonarQube FastAPI Bridge

A FastAPI application that serves as a bridge between your local SonarQube instance and Google Colab, allowing you to access SonarQube analysis results through a simple REST API.

## Features

- **Project Management**: List all projects in your SonarQube instance
- **Issue Analysis**: Get detailed issues for any project with severity filtering
- **Metrics Collection**: Retrieve comprehensive metrics (bugs, vulnerabilities, code smells, coverage, etc.)
- **Complete Analysis**: Get both issues and metrics in a single request
- **CORS Enabled**: Works seamlessly with Google Colab
- **Easy Configuration**: Configure SonarQube connection via API or environment variables

## Quick Start

### 1. Installation

```bash
cd sonarqube_api
pip install -r requirements.txt
```

### 2. Configuration

Set environment variables (optional):
```bash
export SONAR_URL="http://localhost:9000"
export SONAR_TOKEN="your_sonarqube_token"
```

Or configure via API after starting the server.

### 3. Run the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### 4. Configure SonarQube Connection (if not using env vars)

```bash
curl -X POST "http://localhost:8000/configure" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "http://localhost:9000",
       "token": "your_sonarqube_token"
     }'
```

## API Endpoints

### Root
- `GET /` - API information and available endpoints

### Configuration
- `POST /configure` - Configure SonarQube connection

### Projects
- `GET /projects` - List all projects
- `GET /projects/{project_key}/issues` - Get project issues
- `GET /projects/{project_key}/metrics` - Get project metrics
- `GET /projects/{project_key}/analysis` - Get complete analysis (issues + metrics)

### Health
- `GET /health` - Health check

## API Documentation

Once the server is running, visit:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Examples

### List Projects
```bash
curl "http://localhost:8000/projects"
```

### Get Project Issues
```bash
# All issues
curl "http://localhost:8000/projects/my-project/issues"

# Filter by severity
curl "http://localhost:8000/projects/my-project/issues?severity=MAJOR"
```

### Get Project Metrics
```bash
curl "http://localhost:8000/projects/my-project/metrics"
```

### Get Complete Analysis
```bash
curl "http://localhost:8000/projects/my-project/analysis"
```

## Google Colab Integration

See `colab_client_example.py` for a complete example of how to use this API from Google Colab.

## SonarQube Token

To get your SonarQube token:
1. Log into your SonarQube instance
2. Go to User > My Account > Security
3. Generate a new token
4. Use this token in the configuration

## Production Deployment

For production use:
1. Update CORS origins in `main.py` to specific domains
2. Use environment variables for configuration
3. Consider using a reverse proxy (nginx)
4. Enable HTTPS
5. Add proper logging and monitoring