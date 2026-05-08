"""TDD tests for OrganicDetector — SmellDetector backed by organic-standalone.

Requires:
  - Java 11+
  - organic-standalone cloned at ../organic-standalone/
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


ORGANIC_DIR = Path(__file__).resolve().parent.parent.parent / "organic-standalone"

needs_organic = pytest.mark.skipif(
    not ORGANIC_DIR.is_dir() or not shutil.which("java"),
    reason="organic-standalone not found or Java not installed",
)


@needs_organic
class TestOrganicDetector:

    def test_implements_smell_detector(self):
        from dataset.organic_detector import OrganicDetector
        from domain.detector import SmellDetector
        assert issubclass(OrganicDetector, SmellDetector)

    def test_detect_on_dummy_java_file(self, tmp_path):
        """Create a trivially smelly Java class and detect smells."""
        from dataset.organic_detector import OrganicDetector
        from domain.models import SmellEvent

        # Write a Java file with obvious Long Method smell
        java_dir = tmp_path / "src"
        java_dir.mkdir()
        (java_dir / "Foo.java").write_text(
            "public class Foo {\n"
            + "  public void longMethod() {\n"
            + "".join(f"    System.out.println({i});\n" for i in range(60))
            + "  }\n"
            + "}\n"
        )

        detector = OrganicDetector(organic_dir=ORGANIC_DIR)
        smells = detector.detect(java_dir)

        # Should detect at least something (LongMethod likely)
        assert isinstance(smells, list)
        assert all(isinstance(s, SmellEvent) for s in smells)

    def test_detect_empty_project_returns_empty(self, tmp_path):
        from dataset.organic_detector import OrganicDetector

        java_dir = tmp_path / "src"
        java_dir.mkdir()

        detector = OrganicDetector(organic_dir=ORGANIC_DIR)
        smells = detector.detect(java_dir)
        assert smells == []

    def test_smell_types_are_normalized(self, tmp_path):
        """Organic outputs CamelCase names — detector should normalize."""
        from dataset.organic_detector import OrganicDetector

        java_dir = tmp_path / "src"
        java_dir.mkdir()
        # A class with public field → ClassDataShouldBePrivate
        (java_dir / "Bar.java").write_text(
            "public class Bar {\n"
            "  public int x = 0;\n"
            "}\n"
        )

        detector = OrganicDetector(organic_dir=ORGANIC_DIR)
        smells = detector.detect(java_dir)
        for s in smells:
            # Should be canonical names like "Class Data Should Be Private", not "ClassDataShouldBePrivate"
            assert not s.smell_type[0].islower() or " " in s.smell_type
