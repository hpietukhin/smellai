"""SonarQube integration for code smell detection."""

try:
    from smellai.sonarqube.tool import scan_commit_smells

    __all__ = ["scan_commit_smells"]
except ImportError:
    # Tool dependencies may not be installed
    __all__ = []
