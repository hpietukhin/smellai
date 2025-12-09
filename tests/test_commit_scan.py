import os

import pytest
from sonarqube import commit_scan
from sonarqube.commit_scan import derive_project_key, normalize_issue


def test_derive_project_key_from_https_url() -> None:
    key = derive_project_key(
        "https://github.com/Org/Repo.git", "5e764b98eafb4eb5af3cdf8b6c83efc1"
    )
    assert key == "org_repo_5e764b98"


def test_derive_project_key_from_ssh_url() -> None:
    key = derive_project_key("git@github.com:Some-Org/Another.Repo", "abcdef1234567890")
    assert key == "some-org_another.repo_abcdef12"


def test_derive_project_key_invalid_url() -> None:
    with pytest.raises(ValueError):
        derive_project_key("https://example.com/not-github/repo.git", "abc123")


def test_normalize_issue_known_rule() -> None:
    issue = {
        "rule": "java:S1541",
        "severity": "CRITICAL",
        "line": 42,
        "message": "Dummy message",
    }
    normalized = normalize_issue(issue)

    assert normalized["smell_type"] == "Complex Method"
    assert normalized["severity"] == "HIGH"
    assert normalized["line"] == 42
    assert normalized["raw_severity"] == "CRITICAL"


def test_normalize_issue_unknown_rule_defaults() -> None:
    issue = {
        "rule": "java:S9999",
        "severity": "UNKNOWN",
        "line": None,
        "message": "Another message",
    }
    normalized = normalize_issue(issue)

    assert normalized["smell_type"] == "java:S9999"
    assert normalized["severity"] == "LOW"
    assert normalized["rule"] == "java:S9999"


# @pytest.mark.skipif(
#     not os.getenv("SONAR_TOKEN") or os.getenv("SKIP_SONAR_TESTS") == "1",
#     reason="Requires SonarQube server running and SONAR_TOKEN environment variable",
# )
@pytest.mark.skip("TOO SLOW WAITING FOR SONARQUBE SETUP")
def test_real_repo_scan():
    repo_url = "https://github.com/frohoff/ysoserial.git"
    commit_sha = "218bcff"
    sonar_token = os.getenv("SONAR_TOKEN", "")
    issues_by_file = commit_scan.scan_commit(
        repo_url=repo_url,
        commit_sha=commit_sha,
        sonar_url="http://localhost:9000",
        sonar_token=sonar_token,
        cache_dir=None,
    )

    # Verify the function returns a dictionary
    assert isinstance(issues_by_file, dict)

    # Verify all values are lists
    for file_path, issues in issues_by_file.items():
        assert isinstance(file_path, str)
        assert isinstance(issues, list)

        # Verify each issue has the expected structure
        for issue in issues:
            assert "smell_type" in issue
            assert "line" in issue
            assert "severity" in issue
            assert "message" in issue
            assert "rule" in issue
            assert "raw_severity" in issue

            # Verify severity is normalized
            assert issue["severity"] in ["HIGH", "MEDIUM", "LOW"]
