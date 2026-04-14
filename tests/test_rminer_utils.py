import textwrap

from rminer.diff_hunk import DiffHunk
from rminer.rminer_utils import parse_diff_hunks


DOCKERFILE_PATCH = textwrap.dedent(
    """@@ -14,7 +14,7 @@ COPY src ./src
 RUN mvn clean package -DskipTests
 RUN mv target/ysoserial-*all*.jar target/ysoserial.jar
 
-FROM openjdk:8-jdk-alpine
+FROM eclipse-temurin:8-jdk-alpine
 
 WORKDIR /app
"""
)


def test_parse_diff_hunks_real_repo_patch() -> None:
    """Ensure parsing real commit diff hunks retains structure."""
    hunks = parse_diff_hunks(DOCKERFILE_PATCH)

    assert len(hunks) == 1

    hunk = hunks[0]
    assert isinstance(hunk, DiffHunk)
    assert hunk.old_start == 14
    assert hunk.old_count == 7
    assert hunk.new_start == 14
    assert hunk.new_count == 7
    assert hunk.removed_lines == ["FROM openjdk:8-jdk-alpine"]
    assert hunk.added_lines == ["FROM eclipse-temurin:8-jdk-alpine"]
    assert "RUN mvn clean package -DskipTests" in hunk.context_lines
    assert (
        "RUN mv target/ysoserial-*all*.jar target/ysoserial.jar" in hunk.context_lines
    )

    hunk_dict = hunk.model_dump()
    assert hunk_dict["old_start"] == 14
    assert hunk_dict["added_lines"] == ["FROM eclipse-temurin:8-jdk-alpine"]
