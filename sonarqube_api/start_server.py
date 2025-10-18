#!/usr/bin/env python3
"""
Startup script for SonarQube FastAPI Bridge
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import requests
        import pydantic
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_sonar_connection(url="http://localhost:9000"):
    """Check if SonarQube is running"""
    try:
        response = requests.get(f"{url}/api/system/status", timeout=5)
        if response.status_code == 200:
            print(f"✓ SonarQube is running at {url}")
            return True
    except requests.exceptions.RequestException:
        pass

    print(f"⚠ SonarQube not accessible at {url}")
    print("Make sure SonarQube is running or configure the correct URL")
    return False

def wait_for_server(url="http://localhost:8000", timeout=30):
    """Wait for the FastAPI server to start"""
    print(f"Waiting for server to start at {url}...")

    for i in range(timeout):
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ Server is running at {url}")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)
        if i % 5 == 0:
            print(f"Still waiting... ({i}/{timeout}s)")

    print(f"✗ Server failed to start within {timeout} seconds")
    return False

def main():
    """Main startup function"""
    print("🚀 Starting SonarQube FastAPI Bridge...")
    print("=" * 50)

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check SonarQube connection
    sonar_url = os.getenv("SONAR_URL", "http://localhost:9000")
    check_sonar_connection(sonar_url)

    # Get configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"\nStarting server on {host}:{port}")
    print(f"SonarQube URL: {sonar_url}")

    if os.getenv("SONAR_TOKEN"):
        print("✓ SONAR_TOKEN is configured")
    else:
        print("⚠ SONAR_TOKEN not set - you'll need to configure via /configure endpoint")

    print("\nAPI Documentation will be available at:")
    print(f"  - Swagger UI: http://{host}:{port}/docs")
    print(f"  - ReDoc: http://{host}:{port}/redoc")
    print(f"  - Health Check: http://{host}:{port}/health")

    print("\n" + "=" * 50)

    try:
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", host,
            "--port", str(port),
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"✗ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()