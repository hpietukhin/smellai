"""
Comprehensive tests for SonarQube FastAPI Bridge
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import json
import os
import sys

# Add the current directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, SonarQubeClient, get_sonar_client


class TestSonarQubeClient:
    """Test the SonarQubeClient class"""

    def test_create_auth_header(self):
        """Test authentication header creation"""
        client = SonarQubeClient("http://localhost:9000", "test_token")
        expected = "dGVzdF90b2tlbjo="  # base64 of "test_token:"
        assert client.auth_header == f"Basic {expected}"

    @patch('requests.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request"""
        mock_response = Mock()
        mock_response.json.return_value = {"key": "value"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = SonarQubeClient("http://localhost:9000", "test_token")
        result = client._make_request("test/endpoint")

        assert result == {"key": "value"}
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_make_request_failure(self, mock_get):
        """Test API request failure"""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        client = SonarQubeClient("http://localhost:9000", "test_token")

        with pytest.raises(Exception):  # Should raise HTTPException
            client._make_request("test/endpoint")

    @patch('main.SonarQubeClient._make_request')
    def test_get_projects(self, mock_request):
        """Test getting projects list"""
        mock_request.return_value = {
            "components": [
                {"key": "project1", "name": "Project 1", "qualifier": "TRK"},
                {"key": "project2", "name": "Project 2", "qualifier": "TRK"}
            ]
        }

        client = SonarQubeClient("http://localhost:9000", "test_token")
        projects = client.get_projects()

        assert len(projects) == 2
        assert projects[0].key == "project1"
        assert projects[1].key == "project2"

    @patch('main.SonarQubeClient._make_request')
    def test_get_project_issues(self, mock_request):
        """Test getting project issues"""
        mock_request.return_value = {
            "issues": [
                {
                    "key": "issue1",
                    "rule": "java:S1234",
                    "severity": "MAJOR",
                    "component": "project1:src/Main.java",
                    "project": "project1",
                    "line": 10,
                    "message": "Test issue",
                    "type": "CODE_SMELL",
                    "creationDate": "2024-01-01T00:00:00Z",
                    "status": "OPEN"
                }
            ]
        }

        client = SonarQubeClient("http://localhost:9000", "test_token")
        issues = client.get_project_issues("project1")

        assert len(issues) == 1
        assert issues[0].key == "issue1"
        assert issues[0].severity == "MAJOR"

    @patch('main.SonarQubeClient._make_request')
    def test_get_project_metrics(self, mock_request):
        """Test getting project metrics"""
        mock_request.return_value = {
            "component": {
                "measures": [
                    {"metric": "bugs", "value": "5"},
                    {"metric": "vulnerabilities", "value": "0"},
                    {"metric": "code_smells", "value": "15"}
                ]
            }
        }

        client = SonarQubeClient("http://localhost:9000", "test_token")
        metrics = client.get_project_metrics("project1")

        assert len(metrics) == 3
        assert metrics[0].metric == "bugs"
        assert metrics[0].value == "5"


class TestAPIEndpoints:
    """Test the FastAPI endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "sonarqube_configured" in data

    @patch('main.sonar_client', None)
    def test_configure_endpoint_success(self):
        """Test successful configuration"""
        with patch('main.SonarQubeClient') as mock_client_class:
            mock_client = Mock()
            mock_client._make_request.return_value = {"status": "UP"}
            mock_client_class.return_value = mock_client

            config_data = {
                "url": "http://localhost:9000",
                "token": "test_token"
            }

            response = self.client.post("/configure", json=config_data)
            assert response.status_code == 200
            assert response.json()["message"] == "SonarQube configured successfully"

    def test_configure_endpoint_failure(self):
        """Test configuration failure"""
        with patch('main.SonarQubeClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            config_data = {
                "url": "http://invalid:9000",
                "token": "invalid_token"
            }

            response = self.client.post("/configure", json=config_data)
            assert response.status_code == 400

    @patch('main.get_sonar_client')
    def test_get_projects_endpoint(self, mock_get_client):
        """Test projects endpoint"""
        from main import ProjectInfo

        mock_client = Mock()
        mock_client.get_projects.return_value = [
            ProjectInfo(key="project1", name="Project 1", qualifier="TRK", lastAnalysisDate=None)
        ]
        mock_get_client.return_value = mock_client

        response = self.client.get("/projects")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["key"] == "project1"

    @patch('main.get_sonar_client')
    def test_get_project_issues_endpoint(self, mock_get_client):
        """Test project issues endpoint"""
        from main import Issue

        mock_client = Mock()
        mock_client.get_project_issues.return_value = [
            Issue(
                key="issue1",
                rule="java:S1234",
                severity="MAJOR",
                component="project1:src/Main.java",
                project="project1",
                line=10,
                message="Test issue",
                type="CODE_SMELL",
                creationDate="2024-01-01T00:00:00Z",
                status="OPEN"
            )
        ]
        mock_get_client.return_value = mock_client

        response = self.client.get("/projects/project1/issues")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["key"] == "issue1"

    @patch('main.get_sonar_client')
    def test_get_project_metrics_endpoint(self, mock_get_client):
        """Test project metrics endpoint"""
        from main import Metric

        mock_client = Mock()
        mock_client.get_project_metrics.return_value = [
            Metric(metric="bugs", value="5", component="project1")
        ]
        mock_get_client.return_value = mock_client

        response = self.client.get("/projects/project1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["metric"] == "bugs"

    @patch('main.get_sonar_client')
    def test_get_project_analysis_endpoint(self, mock_get_client):
        """Test complete project analysis endpoint"""
        from main import Metric, Issue

        mock_client = Mock()

        # Mock metrics
        mock_client.get_project_metrics.return_value = [
            Metric(metric="bugs", value="5", component="project1"),
            Metric(metric="code_smells", value="10", component="project1")
        ]

        # Mock issues
        mock_client.get_project_issues.return_value = [
            Issue(
                key="issue1",
                rule="java:S1234",
                severity="MAJOR",
                component="project1:src/Main.java",
                project="project1",
                line=10,
                message="Test issue",
                type="CODE_SMELL",
                creationDate="2024-01-01T00:00:00Z",
                status="OPEN"
            )
        ]

        mock_get_client.return_value = mock_client

        response = self.client.get("/projects/project1/analysis")
        assert response.status_code == 200
        data = response.json()

        assert data["project"] == "project1"
        assert len(data["issues"]) == 1
        assert len(data["metrics"]) == 2
        assert data["summary"]["bugs"] == "5"
        assert data["summary"]["total_issues"] == 1

    def test_unconfigured_client_error(self):
        """Test error when SonarQube client is not configured"""
        with patch('main.sonar_client', None), \
             patch('os.getenv', return_value=None):

            response = self.client.get("/projects")
            assert response.status_code == 500


class TestIntegration:
    """Integration tests that test the full workflow"""

    @pytest.fixture
    def mock_sonar_api_responses(self):
        """Mock SonarQube API responses for integration testing"""
        return {
            "system/status": {"status": "UP"},
            "projects/search": {
                "components": [
                    {
                        "key": "test-project",
                        "name": "Test Project",
                        "qualifier": "TRK",
                        "lastAnalysisDate": "2024-01-01T12:00:00Z"
                    }
                ]
            },
            "issues/search": {
                "issues": [
                    {
                        "key": "test-issue-1",
                        "rule": "java:S1234",
                        "severity": "MAJOR",
                        "component": "test-project:src/Main.java",
                        "project": "test-project",
                        "line": 15,
                        "message": "Remove this unused import",
                        "type": "CODE_SMELL",
                        "creationDate": "2024-01-01T10:00:00Z",
                        "status": "OPEN"
                    }
                ]
            },
            "measures/component": {
                "component": {
                    "measures": [
                        {"metric": "bugs", "value": "2"},
                        {"metric": "vulnerabilities", "value": "0"},
                        {"metric": "code_smells", "value": "8"},
                        {"metric": "coverage", "value": "85.5"},
                        {"metric": "ncloc", "value": "1250"}
                    ]
                }
            }
        }

    @patch('requests.get')
    def test_full_workflow(self, mock_get, mock_sonar_api_responses):
        """Test the complete workflow from configuration to analysis"""
        def mock_request_side_effect(url, **kwargs):
            # Extract endpoint from URL
            endpoint = url.split('/api/')[-1].split('?')[0]

            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_sonar_api_responses.get(
                endpoint, {}
            )
            return mock_response

        mock_get.side_effect = mock_request_side_effect

        client = TestClient(app)

        # 1. Test configuration
        config_response = client.post("/configure", json={
            "url": "http://localhost:9000",
            "token": "test_token"
        })
        assert config_response.status_code == 200

        # 2. Test getting projects
        projects_response = client.get("/projects")
        assert projects_response.status_code == 200
        projects = projects_response.json()
        assert len(projects) == 1
        assert projects[0]["key"] == "test-project"

        # 3. Test getting issues
        issues_response = client.get("/projects/test-project/issues")
        assert issues_response.status_code == 200
        issues = issues_response.json()
        assert len(issues) == 1
        assert issues[0]["severity"] == "MAJOR"

        # 4. Test getting metrics
        metrics_response = client.get("/projects/test-project/metrics")
        assert metrics_response.status_code == 200
        metrics = metrics_response.json()
        assert len(metrics) == 5

        # 5. Test complete analysis
        analysis_response = client.get("/projects/test-project/analysis")
        assert analysis_response.status_code == 200
        analysis = analysis_response.json()

        assert analysis["project"] == "test-project"
        assert len(analysis["issues"]) == 1
        assert len(analysis["metrics"]) == 5
        assert analysis["summary"]["bugs"] == "2"
        assert analysis["summary"]["total_issues"] == 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])