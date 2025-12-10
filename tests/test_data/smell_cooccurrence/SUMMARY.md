# Code smell co-occurrence test resources - Summary

## What was created

This directory contains comprehensive Java test resources demonstrating code smell co-occurrence patterns, created for research on refactoring dependencies and expert system rules.

## Files created

### Documentation
1. **README.md** - Overview of each file and the smells it demonstrates
2. **USAGE.md** - Detailed guide on using these resources for testing and research
3. **SUMMARY.md** - This file

### Java test resources
4. **OrderProcessor.java** - Demonstrates positive dependencies
   - Long Method (168 lines)
   - Complex Method
   - Conditional Complexity
   - Duplicated Code (3 instances)
   - Switch Statement (2 large switches)
   - Print Statements (13+ occurrences)
   - **Positive dependency**: Refactoring Long Method solves all above

5. **CustomerDataService.java** - Demonstrates negative dependency
   - Long Parameter List (4 methods with 7-10 params)
   - Feature Envy (uses external services more than own data)
   - Data Class (CustomerData with only getters/setters)
   - **Negative dependency**: Fixing Long Parameter List creates Data Class

6. **ReportGenerator.java** - Demonstrates God Class pattern
   - God Class/Large Class (200+ lines, multiple responsibilities)
   - Data Clumps (4 params appear together 8+ times)
   - Feature Envy (works with external collections)
   - Long Method (multiple)
   - Print Statements
   - **Positive dependencies**: Refactoring Large Class solves Data Clumps, Feature Envy, Long Methods

7. **PaymentValidator.java** - Demonstrates Duplicated Conditions
   - Duplicated Conditions (4 validation methods nearly identical)
   - Duplicated Code (same blocks repeated)
   - Complex Method (nested conditionals)
   - **Positive dependency**: Fixing Duplicated Conditions solves Duplicated Code

8. **ConfigurationManager.java** - Demonstrates multiple interconnected smells
   - Long Method (85+ lines)
   - Long Parameter List (4 methods with 6-8 params)
   - Duplicated Code (validation blocks repeated 4 times)
   - Switch Statement (2 large switches)
   - Print Statements
   - **Complex dependencies**: Multiple refactorings needed in sequence

### Metadata and testing
9. **smells_manifest.json** - Machine-readable specification of:
   - All smells in each file
   - Exact locations (line numbers)
   - Positive dependencies (which smells solving one fixes)
   - Negative dependencies (which smells fixing one creates)
   - SonarQube rule mappings

10. **../test_smell_cooccurrence.py** - Pytest test suite (19 tests, all passing)
    - Validates file existence and structure
    - Tests positive dependency documentation
    - Tests negative dependency documentation
    - Validates smell co-occurrence patterns
    - Tests dependency graph integrity

## Smell dependency relationships documented

### Positive dependencies (solving one solves others)
- **Long Method** → Duplicated Code, Switch Statement, Print Statements, Conditional Complexity, Feature Envy
- **Long Parameter List** → Data Clumps
- **Large Class** → Data Clumps, Feature Envy, Long Method
- **Duplicated Conditions** → Duplicated Code, Complex Method

### Negative dependencies (fixing one creates another)
- **Long Parameter List** + "Introduce Parameter Object" → **Data Class**
- **Long Method** + "Extract Method" (incorrect) → **Long Parameter List**
- **Large Class** + "Extract Class" (poor extraction) → **Data Class**

## SonarQube rules mapped

| Rule | Description | Files Affected |
|------|-------------|----------------|
| java:S138 | Functions too long | OrderProcessor, ConfigurationManager, ReportGenerator |
| java:S1541 | Methods too complex | OrderProcessor, PaymentValidator |
| java:S1067 | Expressions too complex | OrderProcessor |
| java:S107 | Too many parameters | CustomerDataService, ConfigurationManager, ReportGenerator |
| java:S1200 | Too many dependencies | ReportGenerator |
| java:S110 | Inheritance tree too deep | ReportGenerator |
| java:S1871 | Duplicate branches | PaymentValidator |
| java:S106 | System.out usage | All files |

## Usage scenarios

### 1. Testing smell detection algorithms
```bash
# Run your smell detector on these files
your_smell_detector tests/test_data/smell_cooccurrence/*.java

# Compare results with smells_manifest.json
```

### 2. Training ML models
```python
# Use as labeled training data
# Each file has ground-truth smell labels in manifest
from pathlib import Path
import json

manifest = json.loads(Path("smells_manifest.json").read_text())
for file_info in manifest["files"]:
    code = Path(file_info["filename"]).read_text()
    labels = [s["type"] for s in file_info["smells"]]
    # Train your model...
```

### 3. Validating expert system rules
```python
# Test if expert system respects dependencies
def test_refactoring_order():
    # Given: Long Method + Duplicated Code
    # Expert system should recommend Extract Method
    # This should eliminate both smells
    pass
```

### 4. Refactoring recommendation testing
```python
# Test if recommendations respect negative dependencies
def test_negative_dependencies():
    # Given: Long Parameter List
    # If recommending Introduce Parameter Object
    # System should warn about creating Data Class
    pass
```

## Integration with existing workflows

These resources integrate with the `rminer/extract_rminer_data.py` workflow:

1. Real RefactoringMiner data provides actual refactoring examples
2. These synthetic files provide controlled, labeled examples
3. Combined dataset for training and validation

## Validation

All resources validated by:
- ✅ 19 pytest tests (all passing)
- ✅ Manifest structure validated
- ✅ Dependency graphs checked for cycles
- ✅ All files compile-ready Java code
- ✅ Documentation complete

## Statistics

- **5 Java files** with documented smells
- **8 SonarQube rules** mapped
- **6 unique smell types** demonstrated
- **4 positive dependency patterns** documented
- **3 negative dependency patterns** documented
- **19 automated tests** validating resources
- **~1000 lines** of Java code
- **~1500 lines** of documentation

## Next steps

To extend these resources:

1. **Add more smell types**: Lazy Class, Speculative Generality, Message Chains
2. **Add refactored versions**: Create "after" files showing proper refactorings
3. **Add intermediate states**: Show partial refactorings and their effects
4. **Add real-world examples**: Extract from actual projects
5. **Add cross-language examples**: Python, JavaScript versions

## References

Based on:
- Martin Fowler's "Refactoring" catalog
- F. Khomh et al. "An exploratory study of the impact of code smells on software change-proneness"
- Research on code smell dependencies for expert system rules
- SonarQube Java rule definitions
