# repo_utils

Git repository operations: clone, checkout, test execution, and temporary directory management.

## Key Files

- **\_\_init\_\_.py** - Core functions: `clone_repository`, `checkout_repo`, `get_branch`, `temp_repo_context`
- **operations.py** - Enhanced operations: `clone_and_checkout`, `checkout_commit`, `get_previous_commit`, `find_project_root`
- **test_execution.py** - `run_tests_enhanced`, `get_build_command`, `has_wrapper` (Maven/Gradle support)
- **errors.py** - `RepoDontExistError`, `NoGithubTokenFoundError`

## Usage

```python
from repo_utils import clone_repository, checkout_repo, temp_repo_context

# Clone and checkout
repo_name = clone_repository("https://github.com/user/repo.git")
checkout_repo(Path(f"repos/{repo_name}"), branch="main")

# Temporary repo with automatic cleanup
with temp_repo_context() as temp_dir:
    clone_repository("https://github.com/user/repo.git", target_dir=temp_dir)
```

```python
from repo_utils import clone_and_checkout, run_tests_enhanced

# Clone + checkout commit in one call
repo, project_root = clone_and_checkout(repo_url, commit_sha, target_dir)

# Run tests with wrapper/system tool detection
result = run_tests_enhanced(project_root, custom_command="mvn clean test")
```

## Notes

- All git operations use GitPython; functions degrade gracefully if `git` is not installed (via `@require_git_import` decorator)
- Temporary directories are created under `temp/` and safety-checked before deletion
