"""Canonical smell dependency rules used across agents and workflows.

Source basis
------------
* Fowler, Beck et al. (1999) — 22 canonical code smells, 91 refactoring ops
* Markovič & Polášek (2016, FIIT STU diploma) — dependency graph (ch. 5),
  refactoring-operation frequency rankings (Fig. 22–29), 7 smell superclasses
* Composite Refactorings 2020 dataset (Palomba et al.) — 19 OO-metrics smell types

Structure
---------
Constants       : one Python name per canonical smell string
Superclasses    : Markovic ch.5.1 — 7 groups for coarse reasoning
DEPENDENCY_RULES: positive = "resolving A tends to also resolve B"
                  negative = "resolving A with its typical operation may CREATE B"
                  (derived from Fig. 20 / Fig. 21 in Markovic; key text examples:
                   Middle Man → creates Message Chains;
                   Long Parameter List → can create Data Class)
REFACTORING_CATALOGUE: smell → ranked [(refactoring_op, frequency_rank)] 1=most common
DATASET_SMELL_TYPE_MAP: CamelCase Neo4j types → canonical strings
DATASET_DEFAULT_SEVERITY: severity for dataset smells (no SonarQube value available)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Smell-type constants — Fowler's 22
# ---------------------------------------------------------------------------

LONG_METHOD                 = "Long Method"
LARGE_CLASS                 = "Large Class"
LONG_PARAMETER_LIST         = "Long Parameter List"
DUPLICATED_CODE             = "Duplicated Code"
DIVERGENT_CHANGE            = "Divergent Change"
SHOTGUN_SURGERY             = "Shotgun Surgery"
FEATURE_ENVY                = "Feature Envy"
DATA_CLUMPS                 = "Data Clumps"
PRIMITIVE_OBSESSION         = "Primitive Obsession"
SWITCH_STATEMENT            = "Switch Statement"
PARALLEL_INHERITANCE        = "Parallel Inheritance Hierarchies"
LAZY_CLASS                  = "Lazy Class"
SPECULATIVE_GENERALITY      = "Speculative Generality"
TEMPORARY_FIELD             = "Temporary Field"
MESSAGE_CHAINS              = "Message Chains"
MIDDLE_MAN                  = "Middle Man"
INAPPROPRIATE_INTIMACY      = "Inappropriate Intimacy"
ALTERNATIVE_CLASSES         = "Alternative Classes with Different Interfaces"
INCOMPLETE_LIBRARY_CLASS    = "Incomplete Library Class"
DATA_CLASS                  = "Data Class"
REFUSED_BEQUEST             = "Refused Bequest"
COMMENTS                    = "Comments"

# ---------------------------------------------------------------------------
# Smell-type constants — spec / SmellAI internal (some overlap with Fowler)
# ---------------------------------------------------------------------------

COMPLEX_METHOD              = "Complex Method"
CONDITIONAL_COMPLEXITY      = "Conditional Complexity"
GOD_CLASS                   = "God Class"
BAD_CLASS_CONTENT           = "Bad Class Content"
BAD_INHERITANCE             = "Bad Inheritance"        # Markovic superclass as smell
NEEDLESS_PART               = "Needless Part"          # Markovic superclass as smell
DUPLICATED_CONDITIONS       = "Duplicated Conditions"
PRINT_STATEMENTS            = "Print Statements"

# ---------------------------------------------------------------------------
# Smell-type constants — Composite Refactorings 2020 dataset (19 types)
# ---------------------------------------------------------------------------

COMPLEX_CLASS               = "Complex Class"
SPAGHETTI_CODE              = "Spaghetti Code"
CLASS_DATA_SHOULD_BE_PRIVATE = "Class Data Should Be Private"
BRAIN_METHOD                = "Brain Method"
BRAIN_CLASS                 = "Brain Class"
INTENSIVE_COUPLING          = "Intensive Coupling"
DISPERSED_COUPLING          = "Dispersed Coupling"

# ---------------------------------------------------------------------------
# Markovic ch.5.1 — 7 smell superclasses (Table 8)
#
# A smell can belong to multiple groups (multiple inheritance allowed).
# Use for coarse graph search and priority estimation.
# ---------------------------------------------------------------------------

SMELL_GROUPS: dict[str, list[str]] = {
    # Problém predstavuje veľkosť metódy, triedy a podobne
    "Bad Size": [
        LARGE_CLASS, LONG_METHOD, LONG_PARAMETER_LIST,
        COMPLEX_METHOD, CONDITIONAL_COMPLEXITY, GOD_CLASS,
        COMPLEX_CLASS, BRAIN_CLASS, BRAIN_METHOD, SPAGHETTI_CODE,
    ],
    # Daná časť kódu by sa nemala nachádzať tam, kde sa aktuálne nachádza
    "Bad Location": [
        FEATURE_ENVY, COMMENTS, DUPLICATED_CODE, DIVERGENT_CHANGE,
        SHOTGUN_SURGERY, SWITCH_STATEMENT, DUPLICATED_CONDITIONS,
        DISPERSED_COUPLING, INTENSIVE_COUPLING,
    ],
    # Trieda neobsahuje funkcionalitu, ktorú by mala obsahovať
    "Bad Class Content": [
        DATA_CLASS, LAZY_CLASS, FEATURE_ENVY, LARGE_CLASS,
        BAD_CLASS_CONTENT, CLASS_DATA_SHOULD_BE_PRIVATE,
        INCOMPLETE_LIBRARY_CLASS,
    ],
    # Rôzne problémy s dedením
    "Bad Inheritance": [
        ALTERNATIVE_CLASSES, PARALLEL_INHERITANCE, REFUSED_BEQUEST,
        BAD_INHERITANCE,
    ],
    # V kóde sa nachádzajú časti, ktoré je možné odstrániť
    "Needless Part": [
        COMMENTS, DUPLICATED_CODE, REFUSED_BEQUEST, SPECULATIVE_GENERALITY,
        LAZY_CLASS, NEEDLESS_PART, PRINT_STATEMENTS,
    ],
    # Problém so spôsobom práce s atribútmi triedy
    "Attribute Problem": [
        DATA_CLUMPS, TEMPORARY_FIELD, PRIMITIVE_OBSESSION,
    ],
    # Zlá, prípadne komplikovaná komunikácia medzi triedami
    "Bad Communication": [
        INAPPROPRIATE_INTIMACY, MIDDLE_MAN, MESSAGE_CHAINS,
        INTENSIVE_COUPLING,
    ],
}

# Reverse index: smell → list of group names it belongs to
SMELL_GROUP_MEMBERSHIP: dict[str, list[str]] = {}
for _grp, _members in SMELL_GROUPS.items():
    for _s in _members:
        SMELL_GROUP_MEMBERSHIP.setdefault(_s, []).append(_grp)

# ---------------------------------------------------------------------------
# DEPENDENCY_RULES
#
# "positive": resolving this smell tends to simultaneously resolve the listed smells
# "negative": resolving this smell with its typical refactoring may CREATE the listed smells
#
# Sources: Markovic Fig. 20 (positive) & Fig. 21 (negative).
# Explicit text examples (§2.5 Markovič):
#   - "After Middle Man removal, Message Chains typically emerge"
#   - "Long Parameter List → may create Data Class (Introduce Parameter Object)"
#   - "Long Method + Long Parameter List: fix Long Method first (Extract Method eliminates
#      parameters too)"
# ---------------------------------------------------------------------------

DEPENDENCY_RULES: dict[str, dict[str, list[str]]] = {

    # --- Bad Size -----------------------------------------------------------

    LONG_METHOD: {
        "positive": [
            DUPLICATED_CODE,       # Extract Method removes duplication
            COMMENTS,              # self-documenting after extraction
            FEATURE_ENVY,          # co-located code gets properly placed
            DIVERGENT_CHANGE,      # separated concerns after split
            SWITCH_STATEMENT,      # can apply polymorphism via extract
            LONG_PARAMETER_LIST,   # smaller method needs fewer params
            COMPLEX_METHOD,        # closely correlated
            CONDITIONAL_COMPLEXITY,
            BRAIN_METHOD,          # dataset alias
        ],
        "negative": [
            LONG_PARAMETER_LIST,   # Extract Method may need more parameters passed in
            MESSAGE_CHAINS,        # new delegation calls can chain
        ],
    },

    LARGE_CLASS: {
        "positive": [
            DATA_CLUMPS,           # Extract Class groups field clusters
            FEATURE_ENVY,          # moved methods go to right class
            INAPPROPRIATE_INTIMACY,# separation reduces cross-class access
            DUPLICATED_CODE,       # consolidation eliminates copies
            BAD_CLASS_CONTENT,
            GOD_CLASS,             # closely correlated
        ],
        "negative": [
            SHOTGUN_SURGERY,       # Extract Class creates new change points
            MESSAGE_CHAINS,        # delegation after extraction creates chains
            DATA_CLASS,            # extracted class may start as data-only
        ],
    },

    LONG_PARAMETER_LIST: {
        "positive": [
            DATA_CLUMPS,           # clumped params and param lists co-occur
            PRIMITIVE_OBSESSION,   # parameter objects replace primitives
            TEMPORARY_FIELD,
        ],
        "negative": [
            DATA_CLASS,            # Introduce Parameter Object → new data class
        ],
    },

    GOD_CLASS: {
        "positive": [
            DATA_CLUMPS, FEATURE_ENVY, BAD_CLASS_CONTENT,
            INAPPROPRIATE_INTIMACY, DUPLICATED_CODE, LARGE_CLASS,
            BRAIN_CLASS,
        ],
        "negative": [
            SHOTGUN_SURGERY, MESSAGE_CHAINS, DATA_CLASS,
        ],
    },

    COMPLEX_METHOD: {
        "positive": [
            SWITCH_STATEMENT, FEATURE_ENVY, DUPLICATED_CODE,
            DIVERGENT_CHANGE, COMMENTS, LONG_PARAMETER_LIST,
            LONG_METHOD, CONDITIONAL_COMPLEXITY, BRAIN_METHOD,
        ],
        "negative": [
            LONG_PARAMETER_LIST, MESSAGE_CHAINS,
        ],
    },

    CONDITIONAL_COMPLEXITY: {
        "positive": [
            SWITCH_STATEMENT, DUPLICATED_CODE, COMMENTS,
            LONG_METHOD, COMPLEX_METHOD, BRAIN_METHOD,
        ],
        "negative": [
            LARGE_CLASS,    # Replace Conditional with Polymorphism adds classes
        ],
    },

    BRAIN_CLASS: {   # dataset alias for God Class / Large Class
        "positive": [GOD_CLASS, LARGE_CLASS, DATA_CLUMPS, FEATURE_ENVY],
        "negative": [SHOTGUN_SURGERY, MESSAGE_CHAINS, DATA_CLASS],
    },

    BRAIN_METHOD: {  # dataset alias for Long Method
        "positive": [LONG_METHOD, DUPLICATED_CODE, COMMENTS, FEATURE_ENVY],
        "negative": [LONG_PARAMETER_LIST, MESSAGE_CHAINS],
    },

    COMPLEX_CLASS: {  # dataset alias for Large Class / Complex Method
        "positive": [LARGE_CLASS, COMPLEX_METHOD, DUPLICATED_CODE],
        "negative": [SHOTGUN_SURGERY, MESSAGE_CHAINS],
    },

    SPAGHETTI_CODE: {
        "positive": [LONG_METHOD, COMPLEX_METHOD, DUPLICATED_CODE, DIVERGENT_CHANGE],
        "negative": [LONG_PARAMETER_LIST, MESSAGE_CHAINS, SHOTGUN_SURGERY],
    },

    # --- Bad Location -------------------------------------------------------

    FEATURE_ENVY: {
        "positive": [
            MESSAGE_CHAINS,        # moved method uses local calls
            INAPPROPRIATE_INTIMACY,# reduced cross-class access
            LONG_METHOD,           # extraction cleans up the source class
            DATA_CLASS,            # moved behaviour activates data class
        ],
        "negative": [
            SHOTGUN_SURGERY,       # Move Method = change propagates
            DIVERGENT_CHANGE,      # method may carry mixed concerns
        ],
    },

    DUPLICATED_CODE: {
        "positive": [
            LONG_METHOD,           # Extract Method removes duplicated block
            LARGE_CLASS,           # consolidation reduces class size
            DIVERGENT_CHANGE,
            PARALLEL_INHERITANCE,
        ],
        "negative": [
            LONG_PARAMETER_LIST,   # Pull Up Method may add parameters
            SHOTGUN_SURGERY,       # pulling up creates wider impact
        ],
    },

    DIVERGENT_CHANGE: {
        "positive": [
            SHOTGUN_SURGERY,       # Extract Class consolidates related changes
            LARGE_CLASS,
            FEATURE_ENVY,
        ],
        "negative": [
            SHOTGUN_SURGERY,       # wrong Extract Class split increases scatter
        ],
    },

    SHOTGUN_SURGERY: {
        "positive": [
            DIVERGENT_CHANGE,      # Inline Class consolidates
            INAPPROPRIATE_INTIMACY,
            FEATURE_ENVY,
        ],
        "negative": [
            LARGE_CLASS,           # Inline Class grows the target
        ],
    },

    SWITCH_STATEMENT: {
        "positive": [
            DUPLICATED_CODE,       # polymorphism removes type-checking duplication
            FEATURE_ENVY,          # behaviour moves to the right class
            DIVERGENT_CHANGE,
        ],
        "negative": [
            LARGE_CLASS,           # new subclass hierarchy can bloat
            PARALLEL_INHERITANCE,
        ],
    },

    COMMENTS: {
        "positive": [
            LONG_METHOD,           # Extract Method → self-documenting code
            NEEDLESS_PART,
        ],
        "negative": [],            # rarely creates smells
    },

    DISPERSED_COUPLING: {   # dataset — methods in class call many disparate classes
        "positive": [SHOTGUN_SURGERY, FEATURE_ENVY, INAPPROPRIATE_INTIMACY],
        "negative": [LARGE_CLASS, MESSAGE_CHAINS],
    },

    INTENSIVE_COUPLING: {   # dataset — single method calls many methods of one class
        "positive": [FEATURE_ENVY, MESSAGE_CHAINS, INAPPROPRIATE_INTIMACY],
        "negative": [SHOTGUN_SURGERY, DIVERGENT_CHANGE],
    },

    # --- Bad Inheritance ----------------------------------------------------

    REFUSED_BEQUEST: {
        "positive": [
            INAPPROPRIATE_INTIMACY,  # Replace Inheritance with Delegation
            ALTERNATIVE_CLASSES,
        ],
        "negative": [
            MESSAGE_CHAINS,          # delegation chains
            FEATURE_ENVY,            # delegation accesses parent's data
        ],
    },

    PARALLEL_INHERITANCE: {
        "positive": [
            SHOTGUN_SURGERY,
            DUPLICATED_CODE,
        ],
        "negative": [
            LARGE_CLASS,             # consolidating hierarchies bloats class
        ],
    },

    ALTERNATIVE_CLASSES: {
        "positive": [
            DUPLICATED_CODE,
        ],
        "negative": [],
    },

    BAD_INHERITANCE: {   # spec / dataset proxy for the Bad Inheritance group
        "positive": [REFUSED_BEQUEST, INAPPROPRIATE_INTIMACY],
        "negative": [MESSAGE_CHAINS, FEATURE_ENVY, DATA_CLASS],
    },

    # --- Needless Part ------------------------------------------------------

    SPECULATIVE_GENERALITY: {
        "positive": [
            LAZY_CLASS,
            NEEDLESS_PART,
            COMMENTS,
        ],
        "negative": [],
    },

    LAZY_CLASS: {
        "positive": [NEEDLESS_PART],
        "negative": [],
    },

    NEEDLESS_PART: {
        "positive": [COMMENTS, LAZY_CLASS, DUPLICATED_CODE],
        "negative": [],
    },

    PRINT_STATEMENTS: {
        "positive": [NEEDLESS_PART],
        "negative": [DATA_CLASS, LAZY_CLASS],
    },

    # --- Attribute Problem --------------------------------------------------

    DATA_CLUMPS: {
        "positive": [
            LONG_PARAMETER_LIST,   # Introduce Parameter Object
            PRIMITIVE_OBSESSION,
            TEMPORARY_FIELD,
        ],
        "negative": [
            DATA_CLASS,            # Extract Class for clumps = data-only class
        ],
    },

    TEMPORARY_FIELD: {
        "positive": [
            DATA_CLUMPS,
            LARGE_CLASS,           # removes dead fields from class
        ],
        "negative": [
            DATA_CLASS,            # Introduce Null Object needs new class
        ],
    },

    PRIMITIVE_OBSESSION: {
        "positive": [
            DATA_CLUMPS,
            LONG_PARAMETER_LIST,
        ],
        "negative": [
            DATA_CLASS,            # Replace Data Value with Object
            LAZY_CLASS,            # introduced wrapper may be thin
        ],
    },

    CLASS_DATA_SHOULD_BE_PRIVATE: {   # dataset
        "positive": [PRIMITIVE_OBSESSION, DATA_CLUMPS],
        "negative": [DATA_CLASS],
    },

    # --- Bad Class Content --------------------------------------------------

    DATA_CLASS: {
        "positive": [
            FEATURE_ENVY,          # moved behaviour activates class
            BAD_CLASS_CONTENT,
        ],
        "negative": [
            INAPPROPRIATE_INTIMACY,  # added methods may over-access others
            MESSAGE_CHAINS,
        ],
    },

    BAD_CLASS_CONTENT: {
        "positive": [DATA_CLASS, FEATURE_ENVY, LAZY_CLASS],
        "negative": [SHOTGUN_SURGERY],
    },

    INCOMPLETE_LIBRARY_CLASS: {
        "positive": [DUPLICATED_CODE, FEATURE_ENVY],
        "negative": [DATA_CLASS],
    },

    # --- Bad Communication --------------------------------------------------

    INAPPROPRIATE_INTIMACY: {
        "positive": [
            FEATURE_ENVY,          # exposing what was internal
            MESSAGE_CHAINS,        # Hide Delegate  ← see also negative
            DIVERGENT_CHANGE,
        ],
        "negative": [
            MESSAGE_CHAINS,        # Hide Delegate creates delegation chains
            SHOTGUN_SURGERY,
        ],
    },

    MIDDLE_MAN: {
        # Text §2.5 explicitly: "After Middle Man removal, Message Chains typically emerge"
        "positive": [
            INAPPROPRIATE_INTIMACY,  # direct call reduces indirection
        ],
        "negative": [
            MESSAGE_CHAINS,        # Remove Middle Man exposes call chain
            INAPPROPRIATE_INTIMACY,# direct access can create intimacy
        ],
    },

    MESSAGE_CHAINS: {
        "positive": [
            MIDDLE_MAN,            # Hide Delegate introduces a middle man
            INAPPROPRIATE_INTIMACY,
        ],
        "negative": [
            MIDDLE_MAN,
            DATA_CLASS,
        ],
    },

    # --- SmellAI / spec additions -------------------------------------------

    DUPLICATED_CONDITIONS: {
        "positive": [DIVERGENT_CHANGE, SHOTGUN_SURGERY],
        "negative": [LARGE_CLASS, BAD_INHERITANCE],
    },
}

# Convenience shorthands re-exported for legacy callers
METHOD_REFACTORING_POSITIVES = DEPENDENCY_RULES[LONG_METHOD]["positive"]
METHOD_REFACTORING_NEGATIVES = DEPENDENCY_RULES[LONG_METHOD]["negative"]
CLASS_REFACTORING_POSITIVES  = DEPENDENCY_RULES[LARGE_CLASS]["positive"]
CLASS_REFACTORING_NEGATIVES  = DEPENDENCY_RULES[LARGE_CLASS]["negative"]

# ---------------------------------------------------------------------------
# REFACTORING_CATALOGUE
#
# smell_type → [(refactoring_op, frequency_rank)]
# rank 1 = most commonly applied, 5 = least common
# Sources: Fowler (1999) ch. 3; Markovic Fig. 22-29 frequency rankings.
# Composite Refactorings 2020 uses several RefactoringMiner labels that are
# naming variants of Fowler/Markovic operations.  Keep those labels here (not
# in evaluation code) so dataset refactorings compare directly to planner
# actions:
#   - Attribute == Field: Move/Pull Up/Push Down Attribute
#   - Move Class / Rename Class are dataset extensions absent from Markovic's
#     91-operation list, added for coverage of RefactoringMiner vocabulary.
# ---------------------------------------------------------------------------

REFACTORING_CATALOGUE: dict[str, list[tuple[str, int]]] = {

    # Fowler's 22 ----------------------------------------------------------------
    LONG_METHOD: [
        ("Extract Method", 1),
        ("Replace Temp with Query", 2),
        ("Introduce Parameter Object", 3),
        ("Preserve Whole Object", 4),
        ("Replace Method with Method Object", 5),
    ],
    LARGE_CLASS: [
        ("Extract Class", 1),
        ("Extract Subclass", 2),
        ("Extract Interface", 3),
        ("Move Class", 4),
        ("Rename Class", 5),
    ],
    LONG_PARAMETER_LIST: [
        ("Introduce Parameter Object", 1),
        ("Preserve Whole Object", 2),
        ("Replace Parameter with Method", 3),
    ],
    DUPLICATED_CODE: [
        ("Extract Method", 1),
        ("Extract Class", 2),
        ("Pull Up Method", 3),
        ("Form Template Method", 4),
    ],
    DIVERGENT_CHANGE: [
        ("Extract Class", 1),
        ("Move Class", 2),
    ],
    SHOTGUN_SURGERY: [
        ("Move Method", 1),
        ("Move Field", 2),
        ("Move Attribute", 2),
        ("Inline Class", 3),
        ("Move Class", 4),
    ],
    FEATURE_ENVY: [
        ("Move Method", 1),
        ("Extract Method", 2),
        ("Move Field", 3),
        ("Move Attribute", 3),
        ("Move Class", 4),
    ],
    DATA_CLUMPS: [
        ("Extract Class", 1),
        ("Introduce Parameter Object", 2),
        ("Preserve Whole Object", 3),
    ],
    PRIMITIVE_OBSESSION: [
        ("Replace Data Value with Object", 1),
        ("Replace Type Code with Class", 2),
        ("Extract Class", 3),
        ("Introduce Parameter Object", 4),
        ("Replace Array with Object", 5),
    ],
    SWITCH_STATEMENT: [
        ("Replace Conditional with Polymorphism", 1),
        ("Extract Method", 2),
        ("Move Method", 3),
        ("Replace Type Code with Subclasses", 4),
        ("Replace Type Code with State/Strategy", 5),
    ],
    PARALLEL_INHERITANCE: [
        ("Move Method", 1),
        ("Move Field", 2),
        ("Move Attribute", 2),
        ("Pull Up Method", 3),
        ("Pull Up Field", 4),
        ("Pull Up Attribute", 4),
    ],
    LAZY_CLASS: [
        ("Inline Class", 1),
        ("Collapse Hierarchy", 2),
    ],
    SPECULATIVE_GENERALITY: [
        ("Collapse Hierarchy", 1),
        ("Inline Class", 2),
        ("Remove Parameter", 3),
        ("Rename Method", 4),
        ("Rename Class", 5),
    ],
    TEMPORARY_FIELD: [
        ("Extract Class", 1),
        ("Introduce Null Object", 2),
    ],
    MESSAGE_CHAINS: [
        ("Hide Delegate", 1),
        ("Extract Method", 2),
    ],
    MIDDLE_MAN: [
        ("Remove Middle Man", 1),
        ("Inline Method", 2),
        ("Replace Delegation with Inheritance", 3),
    ],
    INAPPROPRIATE_INTIMACY: [
        ("Move Method", 1),
        ("Move Field", 2),
        ("Move Attribute", 2),
        ("Change Bidirectional Association to Unidirectional", 3),
        ("Replace Inheritance with Delegation", 4),
        ("Hide Delegate", 5),
    ],
    ALTERNATIVE_CLASSES: [
        ("Rename Method", 1),
        ("Rename Class", 1),
        ("Move Method", 2),
        ("Move Class", 2),
    ],
    INCOMPLETE_LIBRARY_CLASS: [
        ("Introduce Foreign Method", 1),
        ("Introduce Local Extension", 2),
    ],
    DATA_CLASS: [
        ("Move Method", 1),
        ("Encapsulate Field", 2),
        ("Encapsulate Collection", 3),
        ("Remove Setting Method", 4),
    ],
    REFUSED_BEQUEST: [
        ("Push Down Method", 1),
        ("Move Method", 1),      # dataset: moved inherited behaviour to proper owner
        ("Push Down Field", 2),
        ("Push Down Attribute", 2),
        ("Move Attribute", 2),   # dataset: Attribute == Field
        ("Replace Inheritance with Delegation", 3),
    ],
    COMMENTS: [
        ("Extract Method", 1),
        ("Rename Method", 2),
        ("Introduce Assertion", 3),
    ],

    # Spec / SmellAI internal ------------------------------------------------
    COMPLEX_METHOD: [
        ("Extract Method", 1),
        ("Decompose Conditional", 2),
    ],
    CONDITIONAL_COMPLEXITY: [
        ("Extract Method", 1),
        ("Replace Nested Conditional with Guard Clauses", 2),
        ("Replace Conditional with Polymorphism", 3),
    ],
    GOD_CLASS: [
        ("Extract Class", 1),
        ("Move Method", 2),
        ("Extract Method", 3),
        ("Move Field", 4),
        ("Move Attribute", 4),
        ("Move Class", 5),
    ],
    BAD_CLASS_CONTENT: [
        ("Move Method", 1),
        ("Extract Class", 2),
        ("Encapsulate Field", 3),
    ],
    BAD_INHERITANCE: [
        ("Replace Inheritance with Delegation", 1),
        ("Push Down Method", 2),
        ("Push Down Field", 2),
        ("Push Down Attribute", 2),
        ("Extract Superclass", 3),
        ("Pull Up Method", 4),
        ("Pull Up Field", 5),
        ("Pull Up Attribute", 5),
    ],
    NEEDLESS_PART: [
        ("Inline Class", 1),
        ("Collapse Hierarchy", 2),
        ("Remove Parameter", 3),
    ],
    DUPLICATED_CONDITIONS: [
        ("Consolidate Conditional Expression", 1),
        ("Consolidate Duplicate Conditional Fragments", 2),
    ],
    PRINT_STATEMENTS: [
        ("Replace with Logger", 1),
        ("Remove Control Flag", 2),
    ],

    # Composite Refactorings 2020 dataset types ------------------------------
    COMPLEX_CLASS: [
        ("Extract Method", 1),
        ("Extract Class", 2),
        ("Move Method", 3),
    ],
    SPAGHETTI_CODE: [
        ("Extract Method", 1),
        ("Move Method", 2),
        ("Extract Class", 3),
        ("Replace Method with Method Object", 4),
    ],
    BRAIN_METHOD: [
        ("Extract Method", 1),
        ("Replace Method with Method Object", 2),
        ("Decompose Conditional", 3),
    ],
    BRAIN_CLASS: [
        ("Extract Class", 1),
        ("Extract Method", 2),
        ("Move Method", 3),
        ("Move Field", 4),
        ("Move Attribute", 4),
        ("Move Class", 5),
    ],
    INTENSIVE_COUPLING: [
        ("Move Method", 1),
        ("Extract Method", 2),
        ("Replace Method with Method Object", 3),
        ("Hide Delegate", 4),
    ],
    DISPERSED_COUPLING: [
        ("Move Method", 1),
        ("Inline Class", 2),
        ("Extract Class", 3),
    ],
    CLASS_DATA_SHOULD_BE_PRIVATE: [
        ("Encapsulate Field", 1),
        ("Self Encapsulate Field", 2),
        ("Remove Setting Method", 3),
    ],
}

# ---------------------------------------------------------------------------
# Dataset smell type normalisation (Composite Refactorings 2020, 19 types)
# ---------------------------------------------------------------------------

DATASET_SMELL_TYPE_MAP: dict[str, str] = {
    # CamelCase Neo4j name → canonical string
    "FeatureEnvy":              FEATURE_ENVY,
    "LongMethod":               LONG_METHOD,
    "GodClass":                 GOD_CLASS,
    "LargeClass":               LARGE_CLASS,
    "LongParameterList":        LONG_PARAMETER_LIST,
    "DataClass":                DATA_CLASS,
    "LazyClass":                LAZY_CLASS,
    "ShotgunSurgery":           SHOTGUN_SURGERY,
    "MessageChain":             MESSAGE_CHAINS,
    "ComplexClass":             COMPLEX_CLASS,
    "SpaghettiCode":            SPAGHETTI_CODE,
    "SpeculativeGenerality":    SPECULATIVE_GENERALITY,
    "ClassDataShouldBePrivate": CLASS_DATA_SHOULD_BE_PRIVATE,
    "BrainMethod":              BRAIN_METHOD,
    "BrainClass":               BRAIN_CLASS,
    "IntensiveCoupling":        INTENSIVE_COUPLING,
    "DispersedCoupling":        DISPERSED_COUPLING,
    "RefusedBequest":           REFUSED_BEQUEST,
    "DivergentChange":          DIVERGENT_CHANGE,
}

# ---------------------------------------------------------------------------
# Default severity for dataset smell types (no SonarQube severity available)
# Based on typical impact: class-level God/Brain = HIGH, most method-level = MEDIUM
# ---------------------------------------------------------------------------

DATASET_DEFAULT_SEVERITY: dict[str, str] = {
    # HIGH — systemic structural issues
    LONG_METHOD:                "HIGH",
    GOD_CLASS:                  "HIGH",
    LARGE_CLASS:                "HIGH",
    COMPLEX_CLASS:              "HIGH",
    BRAIN_CLASS:                "HIGH",
    BRAIN_METHOD:               "HIGH",
    SPAGHETTI_CODE:             "HIGH",
    COMPLEX_METHOD:             "HIGH",
    CONDITIONAL_COMPLEXITY:     "HIGH",
    # MEDIUM — localised but impactful
    FEATURE_ENVY:               "MEDIUM",
    DATA_CLASS:                 "MEDIUM",
    LONG_PARAMETER_LIST:        "MEDIUM",
    SHOTGUN_SURGERY:            "MEDIUM",
    DIVERGENT_CHANGE:           "MEDIUM",
    MESSAGE_CHAINS:             "MEDIUM",
    INAPPROPRIATE_INTIMACY:     "MEDIUM",
    INTENSIVE_COUPLING:         "MEDIUM",
    DISPERSED_COUPLING:         "MEDIUM",
    DUPLICATED_CODE:            "MEDIUM",
    DUPLICATED_CONDITIONS:      "MEDIUM",
    BAD_CLASS_CONTENT:          "MEDIUM",
    BAD_INHERITANCE:            "MEDIUM",
    DATA_CLUMPS:                "MEDIUM",
    PRIMITIVE_OBSESSION:        "MEDIUM",
    SWITCH_STATEMENT:           "MEDIUM",
    TEMPORARY_FIELD:            "MEDIUM",
    PARALLEL_INHERITANCE:       "MEDIUM",
    MIDDLE_MAN:                 "MEDIUM",
    REFUSED_BEQUEST:            "MEDIUM",
    # LOW — cosmetic / minor
    LAZY_CLASS:                 "LOW",
    SPECULATIVE_GENERALITY:     "LOW",
    CLASS_DATA_SHOULD_BE_PRIVATE: "LOW",
    ALTERNATIVE_CLASSES:        "LOW",
    COMMENTS:                   "LOW",
    PRINT_STATEMENTS:           "LOW",
    NEEDLESS_PART:              "LOW",
    INCOMPLETE_LIBRARY_CLASS:   "LOW",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def normalize_dataset_smell_type(raw_type: str) -> str:
    """Convert dataset CamelCase smell type to canonical name."""
    return DATASET_SMELL_TYPE_MAP.get(raw_type, raw_type)


def get_refactoring_types(smell_type: str) -> list[str]:
    """Return ranked refactoring type names for a smell (most common first)."""
    return [rt for rt, _ in REFACTORING_CATALOGUE.get(smell_type, [])]


def get_default_severity(smell_type: str) -> str:
    """Return default severity for a smell type. Falls back to MEDIUM."""
    return DATASET_DEFAULT_SEVERITY.get(smell_type, "MEDIUM")


def get_smell_groups(smell_type: str) -> list[str]:
    """Return the Markovic superclass group names for a smell type."""
    return SMELL_GROUP_MEMBERSHIP.get(smell_type, [])


def smells_in_group(group_name: str) -> list[str]:
    """Return all smell types in a Markovic superclass group."""
    return SMELL_GROUPS.get(group_name, [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # rules
    "DEPENDENCY_RULES",
    "REFACTORING_CATALOGUE",
    "DATASET_SMELL_TYPE_MAP",
    "DATASET_DEFAULT_SEVERITY",
    "SMELL_GROUPS",
    "SMELL_GROUP_MEMBERSHIP",
    # helpers
    "normalize_dataset_smell_type",
    "get_refactoring_types",
    "get_default_severity",
    "get_smell_groups",
    "smells_in_group",
    # Fowler's 22
    "LONG_METHOD",
    "LARGE_CLASS",
    "LONG_PARAMETER_LIST",
    "DUPLICATED_CODE",
    "DIVERGENT_CHANGE",
    "SHOTGUN_SURGERY",
    "FEATURE_ENVY",
    "DATA_CLUMPS",
    "PRIMITIVE_OBSESSION",
    "SWITCH_STATEMENT",
    "PARALLEL_INHERITANCE",
    "LAZY_CLASS",
    "SPECULATIVE_GENERALITY",
    "TEMPORARY_FIELD",
    "MESSAGE_CHAINS",
    "MIDDLE_MAN",
    "INAPPROPRIATE_INTIMACY",
    "ALTERNATIVE_CLASSES",
    "INCOMPLETE_LIBRARY_CLASS",
    "DATA_CLASS",
    "REFUSED_BEQUEST",
    "COMMENTS",
    # spec / internal
    "COMPLEX_METHOD",
    "CONDITIONAL_COMPLEXITY",
    "GOD_CLASS",
    "BAD_CLASS_CONTENT",
    "BAD_INHERITANCE",
    "NEEDLESS_PART",
    "DUPLICATED_CONDITIONS",
    "PRINT_STATEMENTS",
    # dataset (Composite Refactorings 2020)
    "COMPLEX_CLASS",
    "SPAGHETTI_CODE",
    "CLASS_DATA_SHOULD_BE_PRIVATE",
    "BRAIN_METHOD",
    "BRAIN_CLASS",
    "INTENSIVE_COUPLING",
    "DISPERSED_COUPLING",
]
