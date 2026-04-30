"""Canonical smell dependency rules used across agents and workflows.

These rules encode positive and negative relationships between smell types
and serve as the shared source of truth for dependency analysis and graph
construction.
"""

LONG_METHOD = "Long Method"
COMPLEX_METHOD = "Complex Method"
CONDITIONAL_COMPLEXITY = "Conditional Complexity"
SWITCH_STATEMENT = "Switch Statement"
FEATURE_ENVY = "Feature Envy"
DUPLICATED_CODE = "Duplicated Code"
DIVERGENT_CHANGE = "Divergent Change"
COMMENTS = "Comments"
LONG_PARAMETER_LIST = "Long Parameter List"
DATA_CLUMPS = "Data Clumps"
DATA_CLASS = "Data Class"
LARGE_CLASS = "Large Class"
GOD_CLASS = "God Class"
BAD_CLASS_CONTENT = "Bad Class Content"
INAPPROPRIATE_INTIMACY = "Inappropriate Intimacy"
MESSAGE_CHAINS = "Message Chains"
DUPLICATED_CONDITIONS = "Duplicated Conditions"
SHOTGUN_SURGERY = "Shotgun Surgery"
BAD_INHERITANCE = "Bad Inheritance"
PRINT_STATEMENTS = "Print Statements"
NEEDLESS_PART = "Needless Part"
LAZY_CLASS = "Lazy Class"

METHOD_REFACTORING_POSITIVES = [
    SWITCH_STATEMENT,
    FEATURE_ENVY,
    DUPLICATED_CODE,
    DIVERGENT_CHANGE,
    COMMENTS,
    LONG_PARAMETER_LIST,
]
METHOD_REFACTORING_NEGATIVES = [LONG_METHOD, LONG_PARAMETER_LIST]
CLASS_REFACTORING_POSITIVES = [DATA_CLUMPS, FEATURE_ENVY, BAD_CLASS_CONTENT]
CLASS_REFACTORING_NEGATIVES = [
    LONG_METHOD,
    DATA_CLASS,
    INAPPROPRIATE_INTIMACY,
    MESSAGE_CHAINS,
]

DEPENDENCY_RULES = {
    LONG_METHOD: {
        "positive": METHOD_REFACTORING_POSITIVES,
        "negative": METHOD_REFACTORING_NEGATIVES,
    },
    COMPLEX_METHOD: {
        "positive": METHOD_REFACTORING_POSITIVES,
        "negative": METHOD_REFACTORING_NEGATIVES,
    },
    CONDITIONAL_COMPLEXITY: {
        "positive": METHOD_REFACTORING_POSITIVES,
        "negative": METHOD_REFACTORING_NEGATIVES,
    },
    LONG_PARAMETER_LIST: {
        "positive": [LONG_PARAMETER_LIST, DATA_CLUMPS],
        "negative": [DATA_CLASS],
    },
    LARGE_CLASS: {
        "positive": CLASS_REFACTORING_POSITIVES,
        "negative": CLASS_REFACTORING_NEGATIVES,
    },
    GOD_CLASS: {
        "positive": CLASS_REFACTORING_POSITIVES,
        "negative": CLASS_REFACTORING_NEGATIVES,
    },
    DUPLICATED_CONDITIONS: {
        "positive": [DIVERGENT_CHANGE, SHOTGUN_SURGERY],
        "negative": [LARGE_CLASS, BAD_INHERITANCE],
    },
    PRINT_STATEMENTS: {
        "positive": [NEEDLESS_PART],
        "negative": [DATA_CLASS, LAZY_CLASS],
    },
}

__all__ = [
    "DEPENDENCY_RULES",
    "LONG_METHOD",
    "COMPLEX_METHOD",
    "LONG_PARAMETER_LIST",
    "DATA_CLUMPS",
    "DATA_CLASS",
    "LARGE_CLASS",
    "GOD_CLASS",
]
