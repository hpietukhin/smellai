"""Shared SonarQube constants used across agents and scanning scripts."""

SEVERITY_MAP: dict[str, str] = {
    "BLOCKER": "HIGH",
    "CRITICAL": "HIGH",
    "MAJOR": "MEDIUM",
    "MINOR": "LOW",
    "INFO": "LOW",
}

RULE_NAME_MAP = {
    "java:S1541": "Complex Method",
    "java:S138": "Long Method",
    "java:S107": "Long Parameter List",
    "java:S1067": "Conditional Complexity",
    "java:S1200": "God Class",
    "java:S110": "Large Class",
    "java:S1871": "Duplicated Conditions",
    "java:S106": "Print Statements",
}
