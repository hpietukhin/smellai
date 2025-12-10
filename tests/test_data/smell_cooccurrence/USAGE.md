# Using the code smell co-occurrence test resources

## Purpose

These Java files demonstrate real-world code smell co-occurrence patterns. They are designed to:

1. Test code smell detection algorithms
2. Validate co-occurrence pattern recognition
3. Train machine learning models on smell dependencies
4. Test expert system refactoring rules
5. Validate refactoring recommendation engines

## File structure

```
smell_cooccurrence/
├── README.md                    # Overview of smells in each file
├── USAGE.md                     # This file
├── smells_manifest.json         # Programmatic description of smells
├── OrderProcessor.java          # Long Method + dependencies
├── CustomerDataService.java     # Long Parameter List → Data Class
├── ReportGenerator.java         # God Class + dependencies
├── PaymentValidator.java        # Duplicated Conditions
└── ConfigurationManager.java    # Multiple interconnected smells
```

## Testing with SonarQube

To scan these files with SonarQube:

```bash
# 1. Create a temporary Java project structure
mkdir -p temp_java_project/src/com/example/smells
cp *.java temp_java_project/src/com/example/smells/

# 2. Create sonar-project.properties
cat > temp_java_project/sonar-project.properties <<EOF
sonar.projectKey=smell-cooccurrence-test
sonar.projectName=Code Smell Co-occurrence Test
sonar.projectVersion=1.0
sonar.sources=src
sonar.java.source=11
sonar.sourceEncoding=UTF-8
EOF

# 3. Run SonarScanner
cd temp_java_project
sonar-scanner

# 4. Review results at http://localhost:9000
```

## Expected SonarQube detections

### OrderProcessor.java
- `java:S138` - Method has too many lines (Long Method)
- `java:S1541` - Cognitive complexity too high
- `java:S1067` - Expression complexity too high
- `java:S106` - System.out.println usage (13+ occurrences)
- Duplicated code blocks

### CustomerDataService.java
- `java:S107` - Too many parameters (4 methods)
- Feature Envy (manual inspection required)
- Data Class pattern (CustomerData)

### ReportGenerator.java
- `java:S1200` - Too many dependencies
- `java:S138` - Long methods
- `java:S106` - Print statements
- Data Clumps (manual or custom rule)

### PaymentValidator.java
- `java:S1871` - Identical implementations in different branches
- High duplication percentage

### ConfigurationManager.java
- `java:S138` - Long Method
- `java:S107` - Long Parameter List
- `java:S106` - Print statements
- High duplication percentage

## Using with Python smell detection

Example script to parse and validate:

```python
import json
from pathlib import Path

# Load the manifest
manifest_path = Path("smells_manifest.json")
manifest = json.loads(manifest_path.read_text())

# Iterate over files and expected smells
for file_info in manifest["files"]:
    filename = file_info["filename"]
    java_file = Path(filename)

    print(f"\\nAnalyzing {filename}:")
    print(f"  Expected smells: {len(file_info['smells'])}")

    for smell in file_info["smells"]:
        print(f"    - {smell['type']} ({smell.get('rule', 'manual')})")

    if "positive_dependencies" in file_info:
        print(f"  Positive dependencies: {len(file_info['positive_dependencies'])}")

    if "negative_dependencies" in file_info:
        print(f"  Negative dependencies: {len(file_info['negative_dependencies'])}")
```

## Research applications

### 1. Dependency graph construction

Use the `smell_dependencies` section in `smells_manifest.json` to build a directed graph:

```python
import networkx as nx

# Build positive dependency graph
G_positive = nx.DiGraph()
for source, targets in manifest["smell_dependencies"]["positive"].items():
    for target in targets:
        G_positive.add_edge(source, target, type="solves")

# Build negative dependency graph
G_negative = nx.DiGraph()
for item in manifest["smell_dependencies"]["negative"].values():
    for dep in item:
        if isinstance(dep, dict):
            G_negative.add_edge(
                dep.get("refactoring"),
                dep.get("creates"),
                type="creates"
            )
```

### 2. Expert system rule validation

Test if your expert system correctly identifies refactoring order:

```python
def test_refactoring_order(expert_system):
    # Given: OrderProcessor.java with Long Method + Duplicated Code
    # Expected: System should recommend Extract Method first
    # Result: Should eliminate both smells

    smells = ["Long Method", "Duplicated Code", "Switch Statement"]
    recommendation = expert_system.suggest_refactoring(smells)

    assert recommendation == "Extract Method"
    assert expert_system.will_solve(recommendation, "Duplicated Code")
    assert expert_system.will_solve(recommendation, "Switch Statement")
```

### 3. Machine learning training data

Convert to training examples:

```python
# Example: Train a classifier to predict smell co-occurrence
import pandas as pd

def create_training_data():
    examples = []

    for file_info in manifest["files"]:
        smell_types = [s["type"] for s in file_info["smells"]]

        # Create binary features for each smell type
        features = {
            "has_long_method": "Long Method" in smell_types,
            "has_long_params": "Long Parameter List" in smell_types,
            "has_duplicated": "Duplicated Code" in smell_types,
            # ... more features
        }

        # Target: co-occurrence patterns
        targets = {
            "cooccurs_duplicated_switch":
                ("Duplicated Code" in smell_types and
                 "Switch Statement" in smell_types)
        }

        examples.append({**features, **targets})

    return pd.DataFrame(examples)
```

## Integration with RefactoringMiner workflow

These files can be integrated with the `rminer/extract_rminer_data.py` workflow:

```python
# Add to your workflow to include synthetic examples
from pathlib import Path

def load_smell_cooccurrence_examples():
    """Load test resources as training examples."""
    base_path = Path("tests/test_data/smell_cooccurrence")
    manifest = json.loads((base_path / "smells_manifest.json").read_text())

    examples = []
    for file_info in manifest["files"]:
        java_file = base_path / file_info["filename"]
        code = java_file.read_text()

        examples.append({
            "filename": file_info["filename"],
            "code": code,
            "smells": file_info["smells"],
            "dependencies": file_info.get("positive_dependencies", [])
        })

    return examples
```

## Validation checklist

When using these files for testing, verify:

- [ ] All mentioned smells are detected
- [ ] Co-occurrence patterns match manifest
- [ ] Positive dependencies are recognized
- [ ] Negative dependencies are recognized
- [ ] Refactoring recommendations respect dependencies
- [ ] Expert system orders refactorings correctly

## Citation

If using these resources in research, please document:
- Source: SmellAI project test resources
- Purpose: Code smell co-occurrence validation
- Files: 5 Java classes with documented smells
- Dependencies: Positive and negative relationships documented
