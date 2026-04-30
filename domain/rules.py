"""Canonical smell dependency rules used across agents and workflows.

These rules encode positive and negative relationships between smell types
and serve as the shared source of truth for dependency analysis and graph
construction.
"""

DEPENDENCY_RULES = {
    "Long Method": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Complex Method": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Conditional Complexity": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Long Parameter List": {
        "positive": ["Long Parameter List", "Data Clumps"],
        "negative": ["Data Class"],
    },
    "Large Class": {
        "positive": ["Data Clumps", "Feature Envy", "Bad Class Content"],
        "negative": [
            "Long Method",
            "Data Class",
            "Inappropriate Intimacy",
            "Message Chains",
        ],
    },
    "God Class": {
        "positive": ["Data Clumps", "Feature Envy", "Bad Class Content"],
        "negative": [
            "Long Method",
            "Data Class",
            "Inappropriate Intimacy",
            "Message Chains",
        ],
    },
    "Duplicated Conditions": {
        "positive": ["Divergent Change", "Shotgun Surgery"],
        "negative": ["Large Class", "Bad Inheritance"],
    },
    "Print Statements": {
        "positive": ["Needless Part"],
        "negative": ["Data Class", "Lazy Class"],
    },
}

__all__ = ["DEPENDENCY_RULES"]
