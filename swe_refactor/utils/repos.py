"""Project repository URL mappings for SWE-Refactor dataset."""

# Mapping from project name to GitHub repository URL
PROJECT_REPOS = {
    "checkstyle": "https://github.com/checkstyle/checkstyle.git",
    "commons-lang": "https://github.com/apache/commons-lang.git",
    "commons-io": "https://github.com/apache/commons-io.git",
    "hibernate-orm": "https://github.com/hibernate/hibernate-orm.git",
    "hibernate-search": "https://github.com/hibernate/hibernate-search.git",
    "javaparser": "https://github.com/javaparser/javaparser.git",
    "junit4": "https://github.com/junit-team/junit4.git",
    "junit5": "https://github.com/junit-team/junit5.git",
    "mockito": "https://github.com/mockito/mockito.git",
    "pmd": "https://github.com/pmd/pmd.git",
}


def get_repo_url(project_name: str) -> str:
    """Get repository URL for project name.

    Args:
        project_name: Name of the project (e.g., "checkstyle")

    Returns:
        GitHub clone URL

    Raises:
        KeyError: If project name not found
    """
    return PROJECT_REPOS[project_name]
