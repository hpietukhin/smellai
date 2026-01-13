"""Pytest tests for data contracts and Pydantic models.

Tests cover:
1. Model instantiation and validation
2. Serialization/deserialization (model_dump/model_validate)
3. Field constraints and defaults
4. Consistency between spec and implementation
"""

import pytest
from pydantic import ValidationError


class TestDatasetsModels:
    """Tests for datasets/models.py Pydantic models."""

    def test_record_inputs_required_fields(self):
        """RecordInputs requires pair_id, code_before, refactoring_type."""
        from datasets.models import RecordInputs

        with pytest.raises(ValidationError) as exc_info:
            RecordInputs()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "pair_id" in missing_fields
        assert "code_before" in missing_fields
        assert "refactoring_type" in missing_fields

    def test_record_inputs_valid(self):
        """RecordInputs accepts valid data."""
        from datasets.models import RecordInputs

        record = RecordInputs(
            pair_id="test-001",
            code_before="public class Foo {}",
            refactoring_type="Extract Method",
        )
        assert record.pair_id == "test-001"
        assert record.code_before == "public class Foo {}"
        assert record.refactoring_type == "Extract Method"
        assert record.context == {}

    def test_record_inputs_with_context(self):
        """RecordInputs accepts optional context dict."""
        from datasets.models import RecordInputs

        record = RecordInputs(
            pair_id="test-001",
            code_before="public class Foo {}",
            refactoring_type="Extract Method",
            context={"sonar_issues": [{"rule": "S1234"}]},
        )
        assert record.context == {"sonar_issues": [{"rule": "S1234"}]}

    def test_record_expectations_required_fields(self):
        """RecordExpectations requires code_after."""
        from datasets.models import RecordExpectations

        with pytest.raises(ValidationError) as exc_info:
            RecordExpectations()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "code_after" in missing_fields

    def test_record_expectations_defaults(self):
        """RecordExpectations provides sensible defaults."""
        from datasets.models import RecordExpectations

        record = RecordExpectations(code_after="public class Foo { void bar() {} }")
        assert record.diff_hunks == []
        assert record.metadata == {}

    def test_record_tags_required_fields(self):
        """RecordTags requires dataset_source."""
        from datasets.models import RecordTags

        with pytest.raises(ValidationError) as exc_info:
            RecordTags()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "dataset_source" in missing_fields

    def test_record_tags_defaults(self):
        """RecordTags provides defaults for optional fields."""
        from datasets.models import RecordTags

        record = RecordTags(dataset_source="rminer")
        assert record.repository == ""
        assert record.commit_sha == ""
        assert record.dataset_source == "rminer"

    def test_diff_hunk_required_fields(self):
        """DiffHunk requires position fields."""
        from datasets.models import DiffHunk

        with pytest.raises(ValidationError) as exc_info:
            DiffHunk()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "old_start" in missing_fields
        assert "old_count" in missing_fields
        assert "new_start" in missing_fields
        assert "new_count" in missing_fields

    def test_diff_hunk_valid(self):
        """DiffHunk accepts valid data."""
        from datasets.models import DiffHunk

        hunk = DiffHunk(
            old_start=10,
            old_count=5,
            new_start=10,
            new_count=8,
        )
        assert hunk.old_start == 10
        assert hunk.old_count == 5
        assert hunk.new_start == 10
        assert hunk.new_count == 8
        assert hunk.removed_lines == []
        assert hunk.added_lines == []
        assert hunk.context_lines == []

    def test_diff_hunk_with_lines(self):
        """DiffHunk stores line content."""
        from datasets.models import DiffHunk

        hunk = DiffHunk(
            old_start=10,
            old_count=3,
            new_start=10,
            new_count=4,
            removed_lines=["    int x = 1;"],
            added_lines=["    int x = 1;", "    int y = 2;"],
            context_lines=["public void foo() {", "}"],
        )
        assert len(hunk.removed_lines) == 1
        assert len(hunk.added_lines) == 2
        assert len(hunk.context_lines) == 2

    def test_diff_hunk_serialization(self):
        """DiffHunk serializes and deserializes correctly."""
        from datasets.models import DiffHunk

        hunk = DiffHunk(
            old_start=10,
            old_count=3,
            new_start=10,
            new_count=4,
            removed_lines=["old line"],
            added_lines=["new line"],
        )

        data = hunk.model_dump()
        assert data["old_start"] == 10
        assert data["removed_lines"] == ["old line"]

        restored = DiffHunk.model_validate(data)
        assert restored == hunk

    def test_rminer_expectations_required_fields(self):
        """RMinerExpectations requires all refactoring metadata."""
        from datasets.models import RMinerExpectations

        with pytest.raises(ValidationError) as exc_info:
            RMinerExpectations()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "num_refactorings" in missing_fields
        assert "num_hunks" in missing_fields
        assert "diff_hunks" in missing_fields
        assert "refactoring_types" in missing_fields
        assert "refactoring_descriptions" in missing_fields
        assert "file_path" in missing_fields

    def test_rminer_expectations_valid(self):
        """RMinerExpectations accepts valid data with nested DiffHunk."""
        from datasets.models import DiffHunk, RMinerExpectations

        hunk = DiffHunk(old_start=10, old_count=5, new_start=10, new_count=8)
        expectations = RMinerExpectations(
            num_refactorings=1,
            num_hunks=1,
            diff_hunks=[hunk],
            refactoring_types=["Extract Method"],
            refactoring_descriptions=["Extract Method foo from bar"],
            file_path="src/Main.java",
        )
        assert expectations.num_refactorings == 1
        assert len(expectations.diff_hunks) == 1
        assert isinstance(expectations.diff_hunks[0], DiffHunk)

    def test_rminer_record_structure(self):
        """RMinerRecord has correct nested structure."""
        from datasets.models import DiffHunk, RMinerExpectations, RMinerRecord, RMinerTags

        hunk = DiffHunk(old_start=10, old_count=5, new_start=10, new_count=8)
        expectations = RMinerExpectations(
            num_refactorings=1,
            num_hunks=1,
            diff_hunks=[hunk],
            refactoring_types=["Extract Method"],
            refactoring_descriptions=["Extract Method foo from bar"],
            file_path="src/Main.java",
        )
        tags = RMinerTags(
            repository="https://github.com/test/repo",
            commit_sha="abc123",
            status="modified",
        )
        record = RMinerRecord(
            inputs={"pair_id": "test-001", "sonar_issues": []},
            expectations=expectations,
            tags=tags,
        )
        assert record.inputs["pair_id"] == "test-001"
        assert record.expectations.file_path == "src/Main.java"
        assert record.tags.repository == "https://github.com/test/repo"


class TestRefactoringModels:
    """Tests for models/refactoring.py Pydantic models."""

    def test_refactoring_location_required_fields(self):
        """RefactoringLocation requires position fields."""
        from models.refactoring import RefactoringLocation

        with pytest.raises(ValidationError) as exc_info:
            RefactoringLocation()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "file_path" in missing_fields
        assert "start_line" in missing_fields
        assert "end_line" in missing_fields

    def test_refactoring_location_valid(self):
        """RefactoringLocation accepts valid data."""
        from models.refactoring import RefactoringLocation

        location = RefactoringLocation(
            file_path="src/Main.java",
            start_line=10,
            end_line=20,
        )
        assert location.file_path == "src/Main.java"
        assert location.start_line == 10
        assert location.end_line == 20
        assert location.start_column is None
        assert location.end_column is None
        assert location.code_element is None

    def test_refactoring_location_with_optional_fields(self):
        """RefactoringLocation accepts optional column and element fields."""
        from models.refactoring import RefactoringLocation

        location = RefactoringLocation(
            file_path="src/Main.java",
            start_line=10,
            end_line=20,
            start_column=5,
            end_column=30,
            code_element="calculateTotal()",
        )
        assert location.start_column == 5
        assert location.end_column == 30
        assert location.code_element == "calculateTotal()"

    def test_refactoring_required_fields(self):
        """Refactoring requires type and description."""
        from models.refactoring import Refactoring

        with pytest.raises(ValidationError) as exc_info:
            Refactoring()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "type" in missing_fields
        assert "description" in missing_fields

    def test_refactoring_valid(self):
        """Refactoring accepts valid data."""
        from models.refactoring import Refactoring

        refactoring = Refactoring(
            type="Extract Method",
            description="Extract Method private validate(User user) : boolean",
        )
        assert refactoring.type == "Extract Method"
        assert "validate" in refactoring.description
        assert refactoring.validation is None
        assert refactoring.left_side_locations == []
        assert refactoring.right_side_locations == []

    def test_refactoring_with_locations(self):
        """Refactoring accepts location lists."""
        from models.refactoring import Refactoring, RefactoringLocation

        left_loc = RefactoringLocation(
            file_path="src/Before.java",
            start_line=10,
            end_line=30,
        )
        right_loc = RefactoringLocation(
            file_path="src/After.java",
            start_line=10,
            end_line=15,
        )

        refactoring = Refactoring(
            type="Extract Method",
            description="Extract Method private validate(User user) : boolean",
            validation="TP",
            left_side_locations=[left_loc],
            right_side_locations=[right_loc],
        )
        assert len(refactoring.left_side_locations) == 1
        assert len(refactoring.right_side_locations) == 1
        assert refactoring.validation == "TP"

    def test_rminer_commit_required_fields(self):
        """RMinerCommit requires all identification fields."""
        from models.refactoring import RMinerCommit

        with pytest.raises(ValidationError) as exc_info:
            RMinerCommit()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "id" in missing_fields
        assert "repository" in missing_fields
        assert "sha1" in missing_fields
        assert "url" in missing_fields
        assert "author" in missing_fields
        assert "time" in missing_fields

    def test_rminer_commit_valid(self):
        """RMinerCommit accepts valid data."""
        from models.refactoring import RMinerCommit

        commit = RMinerCommit(
            id=12345,
            repository="https://github.com/test/repo.git",
            sha1="abc123def456",
            url="https://github.com/test/repo/commit/abc123def456",
            author="Test Author",
            time="2026-01-13T10:00:00",
            refactorings=[],
        )
        assert commit.id == 12345
        assert commit.repository == "https://github.com/test/repo.git"
        assert commit.sha1 == "abc123def456"
        assert commit.refactorings == []

    def test_rminer_commit_with_refactorings(self):
        """RMinerCommit contains nested Refactoring objects."""
        from models.refactoring import Refactoring, RMinerCommit

        refactoring = Refactoring(
            type="Extract Method",
            description="Extract Method private foo() : void",
        )
        commit = RMinerCommit(
            id=12345,
            repository="https://github.com/test/repo.git",
            sha1="abc123def456",
            url="https://github.com/test/repo/commit/abc123def456",
            author="Test Author",
            time="2026-01-13T10:00:00",
            refactorings=[refactoring],
        )
        assert len(commit.refactorings) == 1
        assert commit.refactorings[0].type == "Extract Method"

    def test_refactoring_stats_valid(self):
        """RefactoringStats accepts valid statistics."""
        from models.refactoring import RefactoringStats

        stats = RefactoringStats(
            total_commits=100,
            total_repositories=20,
            total_refactorings=250,
            refactoring_type_counts={"Extract Method": 150, "Rename Method": 100},
            validation_counts={"TP": 200, "FP": 50},
            top_repositories=[{"repository": "repo1", "commit_count": 10}],
            clusters_found=5,
            clusters_detail=[],
        )
        assert stats.total_commits == 100
        assert stats.total_refactorings == 250
        assert stats.refactoring_type_counts["Extract Method"] == 150


class TestAgentModels:
    """Tests for agent-related models."""

    def test_refactoring_mapping_required_fields(self):
        """RefactoringMapping requires index and reasoning fields."""
        from agents.rminer_eval.agent import RefactoringMapping

        with pytest.raises(ValidationError) as exc_info:
            RefactoringMapping()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "refactoring_index" in missing_fields
        assert "hunk_index" in missing_fields
        assert "reasoning" in missing_fields

    def test_refactoring_mapping_valid(self):
        """RefactoringMapping accepts valid data."""
        from agents.rminer_eval.agent import RefactoringMapping

        mapping = RefactoringMapping(
            refactoring_index=0,
            hunk_index=1,
            reasoning="The extract method refactoring matches hunk 1 because...",
        )
        assert mapping.refactoring_index == 0
        assert mapping.hunk_index == 1
        assert "extract method" in mapping.reasoning.lower()

    def test_refactoring_mapping_output_valid(self):
        """RefactoringMappingOutput contains analysis and mappings."""
        from agents.rminer_eval.agent import RefactoringMapping, RefactoringMappingOutput

        mapping = RefactoringMapping(
            refactoring_index=0,
            hunk_index=0,
            reasoning="Direct correspondence",
        )
        output = RefactoringMappingOutput(
            analysis="Single extract method refactoring identified",
            mappings=[mapping],
        )
        assert "extract method" in output.analysis.lower()
        assert len(output.mappings) == 1

    def test_dependency_analysis_required_fields(self):
        """DependencyAnalysis requires smell type and rule id."""
        from agents.dependency_analysis.agent import DependencyAnalysis

        with pytest.raises(ValidationError) as exc_info:
            DependencyAnalysis()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "smell_type" in missing_fields
        assert "rule_id" in missing_fields

    def test_dependency_analysis_valid(self):
        """DependencyAnalysis accepts valid dependency data."""
        from agents.dependency_analysis.agent import DependencyAnalysis

        analysis = DependencyAnalysis(
            smell_type="Long Method",
            rule_id="java:S138",
            positive_dependencies=["Switch Statement", "Feature Envy"],
            negative_dependencies=["Long Parameter List"],
        )
        assert analysis.smell_type == "Long Method"
        assert analysis.rule_id == "java:S138"
        assert "Switch Statement" in analysis.positive_dependencies
        assert "Long Parameter List" in analysis.negative_dependencies


class TestSWERefactorModels:
    """Tests for SWE-Refactor specific models."""

    def test_class_hierarchy_defaults(self):
        """ClassHierarchy provides sensible defaults."""
        from datasets.models import ClassHierarchy

        hierarchy = ClassHierarchy()
        assert hierarchy.superclass is None
        assert hierarchy.subclasses == []
        assert hierarchy.interfaces == []

    def test_class_hierarchy_with_data(self):
        """ClassHierarchy stores inheritance information."""
        from datasets.models import ClassHierarchy

        hierarchy = ClassHierarchy(
            superclass="AbstractService",
            subclasses=["UserService", "AdminService"],
            interfaces=["Serializable", "Comparable"],
        )
        assert hierarchy.superclass == "AbstractService"
        assert len(hierarchy.subclasses) == 2
        assert "Serializable" in hierarchy.interfaces

    def test_method_signature_required_fields(self):
        """MethodSignature requires name."""
        from datasets.models import MethodSignature

        with pytest.raises(ValidationError) as exc_info:
            MethodSignature()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "name" in missing_fields

    def test_method_signature_valid(self):
        """MethodSignature stores method information."""
        from datasets.models import MethodSignature

        signature = MethodSignature(
            name="calculateTotal",
            parameters=["int", "double"],
            return_type="double",
        )
        assert signature.name == "calculateTotal"
        assert len(signature.parameters) == 2
        assert signature.return_type == "double"

    def test_build_configuration_required_fields(self):
        """BuildConfiguration requires all fields."""
        from datasets.models import BuildConfiguration

        with pytest.raises(ValidationError) as exc_info:
            BuildConfiguration()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "commit_id" in missing_fields
        assert "jdk_version" in missing_fields
        assert "build_command" in missing_fields

    def test_build_configuration_valid(self):
        """BuildConfiguration accepts valid build data."""
        from datasets.models import BuildConfiguration

        config = BuildConfiguration(
            commit_id="abc123",
            jdk_version=17,
            build_command="mvn clean package -DskipTests",
        )
        assert config.commit_id == "abc123"
        assert config.jdk_version == 17
        assert "mvn" in config.build_command

    def test_test_coverage_defaults(self):
        """TestCoverage provides zero defaults."""
        from datasets.models import TestCoverage

        coverage = TestCoverage()
        assert coverage.branch_coverage == 0.0
        assert coverage.instruction_coverage == 0.0
        assert coverage.line_coverage == 0.0
        assert coverage.complexity_coverage == 0.0
        assert coverage.method_coverage == 0.0

    def test_test_coverage_with_values(self):
        """TestCoverage stores coverage metrics."""
        from datasets.models import TestCoverage

        coverage = TestCoverage(
            branch_coverage=75.5,
            instruction_coverage=80.2,
            line_coverage=82.1,
            complexity_coverage=70.0,
            method_coverage=90.0,
        )
        assert coverage.branch_coverage == 75.5
        assert coverage.line_coverage == 82.1


class TestModelSerialization:
    """Tests for model serialization round-trips."""

    def test_dataset_record_roundtrip(self):
        """DatasetRecord serializes and deserializes correctly."""
        from datasets.models import (
            DatasetRecord,
            RecordExpectations,
            RecordInputs,
            RecordTags,
        )

        record = DatasetRecord(
            inputs=RecordInputs(
                pair_id="test-001",
                code_before="public class Foo {}",
                refactoring_type="Extract Method",
            ),
            expectations=RecordExpectations(
                code_after="public class Foo { void bar() {} }",
            ),
            tags=RecordTags(
                dataset_source="rminer",
                repository="https://github.com/test/repo",
                commit_sha="abc123",
            ),
        )

        data = record.model_dump()
        restored = DatasetRecord.model_validate(data)

        assert restored.inputs.pair_id == record.inputs.pair_id
        assert restored.expectations.code_after == record.expectations.code_after
        assert restored.tags.dataset_source == record.tags.dataset_source

    def test_rminer_commit_roundtrip(self):
        """RMinerCommit serializes and deserializes correctly."""
        from models.refactoring import Refactoring, RefactoringLocation, RMinerCommit

        location = RefactoringLocation(
            file_path="src/Main.java",
            start_line=10,
            end_line=20,
            code_element="calculateTotal()",
        )
        refactoring = Refactoring(
            type="Extract Method",
            description="Extract Method calculateTotal",
            validation="TP",
            left_side_locations=[location],
        )
        commit = RMinerCommit(
            id=12345,
            repository="https://github.com/test/repo.git",
            sha1="abc123",
            url="https://github.com/test/repo/commit/abc123",
            author="Test Author",
            time="2026-01-13T10:00:00",
            refactorings=[refactoring],
        )

        data = commit.model_dump()
        restored = RMinerCommit.model_validate(data)

        assert restored.id == commit.id
        assert restored.sha1 == commit.sha1
        assert len(restored.refactorings) == 1
        assert restored.refactorings[0].type == "Extract Method"
        assert len(restored.refactorings[0].left_side_locations) == 1
        assert restored.refactorings[0].left_side_locations[0].code_element == "calculateTotal()"

    def test_json_compatibility(self):
        """Models produce JSON-compatible output."""
        import json

        from datasets.models import DiffHunk

        hunk = DiffHunk(
            old_start=10,
            old_count=5,
            new_start=10,
            new_count=8,
            removed_lines=["old line"],
            added_lines=["new line"],
        )

        json_str = json.dumps(hunk.model_dump())
        parsed = json.loads(json_str)
        restored = DiffHunk.model_validate(parsed)

        assert restored.old_start == hunk.old_start
        assert restored.removed_lines == hunk.removed_lines
