"""Pytest tests for data contracts — model serialization round-trips."""

class TestModelSerialization:
    """Tests for model serialization round-trips."""

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
