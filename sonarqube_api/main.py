"""
FastAPI application for accessing SonarQube analysis results.
Serves as a bridge between local SonarQube instance and Google Colab.
"""

import os
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# Pydantic models for API responses
class SonarQubeConfig(BaseModel):
    url: str = Field(default="http://localhost:9000", description="SonarQube server URL")
    token: str = Field(description="SonarQube authentication token")


class ProjectInfo(BaseModel):
    key: str
    name: str
    qualifier: str
    lastAnalysisDate: Optional[str] = None


class Issue(BaseModel):
    key: str
    rule: str
    severity: str
    component: str
    project: str
    line: Optional[int] = None
    message: str
    type: str
    creationDate: str
    status: str


class Metric(BaseModel):
    metric: str
    value: str
    component: str


class AnalysisResult(BaseModel):
    project: str
    issues: List[Issue]
    metrics: List[Metric]
    summary: Dict[str, Any]


# FastAPI app initialization
app = FastAPI(
    title="SonarQube API Bridge",
    description="FastAPI bridge for accessing SonarQube analysis results from Google Colab",
    version="1.0.0"
)

# Add CORS middleware for Colab access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual Colab domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SonarQubeClient:
    """Client for interacting with SonarQube API"""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/')
        self.token = token
        self.auth_header = self._create_auth_header()

    def _create_auth_header(self) -> str:
        """Create basic authentication header"""
        auth_string = f"{self.token}:"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {encoded_auth}"

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to SonarQube API"""
        headers = {"Authorization": self.auth_header}
        url = f"{self.url}/api/{endpoint}"

        try:
            response = requests.get(url, headers=headers, params=params or {})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"SonarQube API error: {str(e)}")

    def get_projects(self) -> List[ProjectInfo]:
        """Get list of all projects"""
        data = self._make_request("projects/search")
        return [ProjectInfo(**project) for project in data.get("components", [])]

    def get_project_issues(self, project_key: str, severity: Optional[str] = None) -> List[Issue]:
        """Get issues for a specific project"""
        params = {"componentKeys": project_key, "ps": 500}
        if severity:
            params["severities"] = severity

        data = self._make_request("issues/search", params)
        return [Issue(**issue) for issue in data.get("issues", [])]

    def get_project_metrics(self, project_key: str) -> List[Metric]:
        """Get metrics for a specific project"""
        metric_keys = [
            "bugs", "vulnerabilities", "code_smells", "coverage",
            "duplicated_lines_density", "ncloc", "sqale_index",
            "reliability_rating", "security_rating", "sqale_rating"
        ]

        params = {
            "component": project_key,
            "metricKeys": ",".join(metric_keys)
        }

        data = self._make_request("measures/component", params)
        component = data.get("component", {})
        measures = component.get("measures", [])

        return [
            Metric(
                metric=measure["metric"],
                value=measure.get("value", "0"),
                component=project_key
            )
            for measure in measures
        ]


# Global SonarQube client (will be configured via environment or endpoint)
sonar_client: Optional[SonarQubeClient] = None


def get_sonar_client() -> SonarQubeClient:
    """Dependency to get configured SonarQube client"""
    global sonar_client
    if sonar_client is None:
        # Try to get configuration from environment
        sonar_url = os.getenv("SONAR_URL", "http://localhost:9000")
        sonar_token = os.getenv("SONAR_TOKEN")

        if not sonar_token:
            raise HTTPException(
                status_code=500,
                detail="SonarQube not configured. Use /configure endpoint or set SONAR_TOKEN environment variable"
            )

        sonar_client = SonarQubeClient(sonar_url, sonar_token)

    return sonar_client


@app.post("/configure")
async def configure_sonarqube(config: SonarQubeConfig):
    """Configure SonarQube connection"""
    global sonar_client
    try:
        sonar_client = SonarQubeClient(config.url, config.token)
        # Test connection
        sonar_client._make_request("system/status")
        return {"message": "SonarQube configured successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SonarQube API Bridge",
        "version": "1.0.0",
        "endpoints": {
            "configure": "POST /configure - Configure SonarQube connection",
            "projects": "GET /projects - List all projects",
            "project_issues": "GET /projects/{project_key}/issues - Get project issues",
            "project_metrics": "GET /projects/{project_key}/metrics - Get project metrics",
            "project_analysis": "GET /projects/{project_key}/analysis - Get complete analysis"
        }
    }


@app.get("/projects", response_model=List[ProjectInfo])
async def get_projects(client: SonarQubeClient = Depends(get_sonar_client)):
    """Get list of all projects in SonarQube"""
    return client.get_projects()


@app.get("/projects/{project_key}/issues", response_model=List[Issue])
async def get_project_issues(
    project_key: str,
    severity: Optional[str] = Query(None, description="Filter by severity: INFO, MINOR, MAJOR, CRITICAL, BLOCKER"),
    client: SonarQubeClient = Depends(get_sonar_client)
):
    """Get issues for a specific project"""
    return client.get_project_issues(project_key, severity)


@app.get("/projects/{project_key}/metrics", response_model=List[Metric])
async def get_project_metrics(
    project_key: str,
    client: SonarQubeClient = Depends(get_sonar_client)
):
    """Get metrics for a specific project"""
    return client.get_project_metrics(project_key)


@app.get("/projects/{project_key}/analysis", response_model=AnalysisResult)
async def get_project_analysis(
    project_key: str,
    include_issues: bool = Query(True, description="Include issues in response"),
    severity_filter: Optional[str] = Query(None, description="Filter issues by severity"),
    client: SonarQubeClient = Depends(get_sonar_client)
):
    """Get complete analysis for a project (issues + metrics)"""

    # Get metrics
    metrics = client.get_project_metrics(project_key)

    # Get issues (optionally)
    issues = []
    if include_issues:
        issues = client.get_project_issues(project_key, severity_filter)

    # Create summary from metrics
    summary = {}
    for metric in metrics:
        summary[metric.metric] = metric.value

    # Add issue summary
    if issues:
        issue_summary = {}
        for issue in issues:
            severity = issue.severity
            issue_summary[severity] = issue_summary.get(severity, 0) + 1
        summary["issues_by_severity"] = issue_summary
        summary["total_issues"] = len(issues)

    return AnalysisResult(
        project=project_key,
        issues=issues,
        metrics=metrics,
        summary=summary
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sonarqube_configured": sonar_client is not None
    }


if __name__ == "__main__":
    # Load configuration from environment if available
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )