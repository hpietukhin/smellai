"""
Example client code for testing the SonarQube FastAPI Bridge
Demonstrates how to interact with the API from Python (including Google Colab)
"""

import requests
import json
import time
from typing import Optional, Dict, Any, List


class SonarQubeAPIClient:
    """Python client for the SonarQube FastAPI Bridge"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client

        Args:
            base_url: Base URL of the FastAPI bridge server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response content: {e.response.text}")
            raise

    def configure_sonarqube(self, sonar_url: str, sonar_token: str) -> bool:
        """
        Configure SonarQube connection

        Args:
            sonar_url: SonarQube server URL
            sonar_token: SonarQube authentication token

        Returns:
            True if configuration successful
        """
        try:
            config_data = {
                "url": sonar_url,
                "token": sonar_token
            }

            response = self._request("POST", "/configure", json=config_data)
            print(f"✓ {response['message']}")
            return True

        except Exception as e:
            print(f"✗ Configuration failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        return self._request("GET", "/health")

    def get_api_info(self) -> Dict[str, Any]:
        """Get API information"""
        return self._request("GET", "/")

    def get_projects(self) -> List[Dict[str, Any]]:
        """Get list of all projects"""
        return self._request("GET", "/projects")

    def get_project_issues(self, project_key: str, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get issues for a specific project

        Args:
            project_key: SonarQube project key
            severity: Optional severity filter (INFO, MINOR, MAJOR, CRITICAL, BLOCKER)
        """
        endpoint = f"/projects/{project_key}/issues"
        params = {}
        if severity:
            params["severity"] = severity

        return self._request("GET", endpoint, params=params)

    def get_project_metrics(self, project_key: str) -> List[Dict[str, Any]]:
        """Get metrics for a specific project"""
        endpoint = f"/projects/{project_key}/metrics"
        return self._request("GET", endpoint)

    def get_project_analysis(self, project_key: str, include_issues: bool = True,
                           severity_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get complete analysis for a project

        Args:
            project_key: SonarQube project key
            include_issues: Whether to include issues in the response
            severity_filter: Optional severity filter for issues
        """
        endpoint = f"/projects/{project_key}/analysis"
        params = {
            "include_issues": include_issues
        }
        if severity_filter:
            params["severity_filter"] = severity_filter

        return self._request("GET", endpoint, params=params)


def demo_workflow():
    """Demonstrate the complete workflow"""
    print("🧪 SonarQube API Bridge - Demo Workflow")
    print("=" * 50)

    # Initialize client
    client = SonarQubeAPIClient()

    # 1. Check health
    print("\n1. Health Check")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"SonarQube configured: {health['sonarqube_configured']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # 2. Get API info
    print("\n2. API Information")
    try:
        api_info = client.get_api_info()
        print(f"API: {api_info['message']} v{api_info['version']}")
    except Exception as e:
        print(f"❌ Failed to get API info: {e}")
        return

    # 3. Configure SonarQube (if not already configured)
    print("\n3. SonarQube Configuration")
    health = client.health_check()
    if not health.get('sonarqube_configured', False):
        print("SonarQube not configured. Please configure manually or set environment variables.")
        print("Example:")
        print("  client.configure_sonarqube('http://localhost:9000', 'your_token_here')")
        return
    else:
        print("✓ SonarQube already configured")

    # 4. Get projects
    print("\n4. Available Projects")
    try:
        projects = client.get_projects()
        if projects:
            print(f"Found {len(projects)} projects:")
            for project in projects[:5]:  # Show first 5
                print(f"  - {project['key']}: {project['name']}")

            # Use first project for demo
            demo_project = projects[0]['key']
            print(f"\nUsing '{demo_project}' for detailed analysis...")

        else:
            print("No projects found. Make sure you have projects in SonarQube.")
            return

    except Exception as e:
        print(f"❌ Failed to get projects: {e}")
        return

    # 5. Get project issues
    print(f"\n5. Issues for Project '{demo_project}'")
    try:
        issues = client.get_project_issues(demo_project)
        print(f"Found {len(issues)} total issues")

        # Show issues by severity
        if issues:
            severity_counts = {}
            for issue in issues:
                severity = issue['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            print("Issues by severity:")
            for severity, count in sorted(severity_counts.items()):
                print(f"  {severity}: {count}")

            # Show a few example issues
            print(f"\nExample issues (showing first 3):")
            for issue in issues[:3]:
                print(f"  - {issue['rule']}: {issue['message'][:50]}...")

    except Exception as e:
        print(f"❌ Failed to get issues: {e}")

    # 6. Get project metrics
    print(f"\n6. Metrics for Project '{demo_project}'")
    try:
        metrics = client.get_project_metrics(demo_project)
        print(f"Available metrics: {len(metrics)}")

        # Show key metrics
        key_metrics = ["bugs", "vulnerabilities", "code_smells", "coverage", "ncloc"]
        print("Key metrics:")
        for metric in metrics:
            if metric['metric'] in key_metrics:
                value = metric['value']
                if metric['metric'] == 'coverage':
                    print(f"  {metric['metric']}: {value}%")
                else:
                    print(f"  {metric['metric']}: {value}")

    except Exception as e:
        print(f"❌ Failed to get metrics: {e}")

    # 7. Get complete analysis
    print(f"\n7. Complete Analysis for Project '{demo_project}'")
    try:
        analysis = client.get_project_analysis(demo_project)
        print(f"Project: {analysis['project']}")
        print(f"Total issues: {len(analysis['issues'])}")
        print(f"Total metrics: {len(analysis['metrics'])}")

        # Show summary
        summary = analysis['summary']
        print("\nSummary:")
        if 'total_issues' in summary:
            print(f"  Total issues: {summary['total_issues']}")
        if 'issues_by_severity' in summary:
            print("  Issues by severity:")
            for severity, count in summary['issues_by_severity'].items():
                print(f"    {severity}: {count}")

        # Show key metrics from summary
        key_metrics = ["bugs", "vulnerabilities", "code_smells", "coverage"]
        print("  Key metrics:")
        for metric in key_metrics:
            if metric in summary:
                value = summary[metric]
                if metric == 'coverage':
                    print(f"    {metric}: {value}%")
                else:
                    print(f"    {metric}: {value}")

    except Exception as e:
        print(f"❌ Failed to get complete analysis: {e}")

    print("\n" + "=" * 50)
    print("✅ Demo completed!")


def test_server_connection(base_url: str = "http://localhost:8000") -> bool:
    """Test if the server is running and accessible"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Server is running at {base_url}")
            return True
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server at {base_url}: {e}")
        return False


def wait_for_server(base_url: str = "http://localhost:8000", timeout: int = 30) -> bool:
    """Wait for server to become available"""
    print(f"Waiting for server at {base_url}...")

    for i in range(timeout):
        if test_server_connection(base_url):
            return True

        time.sleep(1)
        if i % 5 == 0 and i > 0:
            print(f"Still waiting... ({i}/{timeout}s)")

    print(f"✗ Server not available after {timeout} seconds")
    return False


if __name__ == "__main__":
    print("🚀 SonarQube API Bridge - Client Example")
    print("=" * 50)

    # Test connection first
    if not test_server_connection():
        print("\n⚠️  Server is not running. To start the server:")
        print("1. cd sonarqube_api")
        print("2. python start_server.py")
        print("\nOr run the server manually:")
        print("python -m uvicorn main:app --reload")
        exit(1)

    # Run the demo
    demo_workflow()

    print("\n💡 Usage in Google Colab:")
    print("""
# Install required packages
!pip install requests

# Use the client
from client_example import SonarQubeAPIClient

# Initialize client (replace with your server URL)
client = SonarQubeAPIClient("http://your-server:8000")

# Configure SonarQube
client.configure_sonarqube("http://localhost:9000", "your_token")

# Get projects and analyze
projects = client.get_projects()
analysis = client.get_project_analysis("your-project-key")
""")