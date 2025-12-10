# Code smell co-occurrence test resources

This directory contains Java code examples demonstrating how code smells co-occur, based on positive and negative dependencies between smells.

## Test files

### 1. OrderProcessor.java - Bad size category (positive dependencies)

**Smells demonstrated:**
- **Long Method** (java:S138): `processOrder()` exceeds 60 lines
- **Complex Method** (java:S1541): Multiple nested conditions and loops
- **Conditional Complexity** (java:S1067): Deep nested if-else chains
- **Duplicated Code**: Similar validation logic repeated multiple times
- **Switch Statement**: Large switch-case structure
- **Print Statements** (java:S106): Debug System.out.println left in code

**Positive dependencies shown:**
- Refactoring the Long Method would simultaneously solve:
  - Duplicated Code (validation logic appears 3+ times)
  - Switch Statement (can be extracted)
  - Print Statements (scattered debug statements)
  - Conditional Complexity (nested conditions can be extracted)

### 2. CustomerDataService.java - Long parameter list (negative dependency)

**Smells demonstrated:**
- **Long Parameter List** (java:S107): Methods with 8+ parameters
- **Feature Envy**: Methods accessing more data from other classes than their own
- **Data Class**: `CustomerData` class (in same file) - shows negative dependency

**Negative dependency shown:**
- Refactoring Long Parameter List using "Introduce Parameter Object" creates `CustomerData`
- `CustomerData` becomes a **Data Class** - a class with only getters/setters and no behavior
- This demonstrates the trade-off: fixing Long Parameter List can create Data Class smell

### 3. ReportGenerator.java - God class/large class

**Smells demonstrated:**
- **God Class** (java:S1200): Class exceeds 200 lines with multiple responsibilities
- **Large Class** (java:S110): Too many methods and fields
- **Data Clumps**: Same group of parameters appears together multiple times
- **Feature Envy**: Methods manipulating data from other objects more than own data
- **Long Method**: Individual methods are also too long
- **Print Statements**: Debug output scattered throughout

**Positive dependencies shown:**
- Refactoring Large Class would solve:
  - Data Clumps (extract related data into new classes)
  - Feature Envy (move methods closer to data they use)
  - Some Long Methods (distributed across extracted classes)

### 4. PaymentValidator.java - Duplicated conditions

**Smells demonstrated:**
- **Duplicated Conditions** (java:S1871): Same conditional logic repeated
- **Duplicated Code**: Similar validation blocks
- **Complex Method**: Due to repeated conditionals

**Positive dependencies shown:**
- Fixing Duplicated Conditions would solve:
  - Duplicated Code
  - Reduce overall complexity

### 5. ConfigurationManager.java - Multiple interconnected smells

**Smells demonstrated:**
- **Long Method**: `loadConfiguration()` exceeds 60 lines
- **Long Parameter List**: Several methods with 8+ parameters
- **Duplicated Code**: Similar error handling repeated
- **Print Statements**: Debug output
- **Switch Statement**: Multiple large switch blocks

**Demonstrates:**
- Complex interplay of multiple smells
- How fixing one can cascade to fix others

## Usage

These files can be used to:

1. Test code smell detection algorithms
2. Validate co-occurrence detection
3. Test refactoring recommendation systems
4. Build training data for ML models learning smell dependencies
5. Validate expert system rules for refactoring order

## Mapping to SonarQube rules

The smells map to these SonarQube Java rules:
- java:S138 - Functions should not have too many lines of code (Long Method)
- java:S1541 - Methods should not be too complex (Complex Method)
- java:S1067 - Expressions should not be too complex (Conditional Complexity)
- java:S107 - Methods should not have too many parameters (Long Parameter List)
- java:S1200 - Classes should not be coupled to too many other classes (God Class)
- java:S110 - Inheritance tree should not be too deep (Large Class indicator)
- java:S1871 - Two branches should not have the same implementation (Duplicated Conditions)
- java:S106 - Standard outputs should not be used directly (Print Statements)
