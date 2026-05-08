"""SmellDetector backed by organic-standalone (Opus/PUC-Rio).

Runs Organic via Gradle subprocess, parses JSON output.
Detects 17 smell types — the same rules used to create the
Composite Refactorings 2020 dataset.

Usage:
    detector = OrganicDetector(organic_dir=Path("../organic-standalone"))
    smells = detector.detect(Path("/path/to/java/source"))
"""
from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import tempfile
import time
from pathlib import Path

from domain.detector import DetectorExecutionError, DetectorUnavailableError, SmellDetector
from domain.models import SmellEvent
from domain.rules import normalize_dataset_smell_type, get_default_severity

LOGGER = logging.getLogger(__name__)


class OrganicDetector(SmellDetector):
    """Detect smells via organic-standalone Gradle CLI."""

    def __init__(
        self,
        organic_dir: Path | None = None,
        timeout: int = 2,
    ) -> None:
        self._organic_dir = organic_dir or Path(__file__).resolve().parent.parent.parent / "organic-standalone"
        self._timeout = timeout

    def detect(self, project_path: Path) -> list[SmellEvent]:
        if not self._organic_dir.is_dir():
            raise DetectorUnavailableError(
                f"organic-standalone not found at {self._organic_dir}"
            )

        gradlew = self._organic_dir / "gradlew"
        if not gradlew.exists():
            raise DetectorUnavailableError(f"gradlew not found in {self._organic_dir}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = Path(tmp.name)

        # Use SDKMAN Java 17 if available (Gradle 7.4 needs ≤17)
        env = None
        sdkman_java = Path.home() / ".sdkman/candidates/java/17.0.14-amzn"
        if sdkman_java.is_dir():
            env = {**os.environ, "JAVA_HOME": str(sdkman_java)}

        cmd = [
            str(gradlew), "run",
            f"--args=-sf '{output_path}' -src '{project_path}'",
        ]

        try:
            LOGGER.info("Running Organic on %s", project_path)
            if os.environ.get("ORGANIC_STREAM_LOGS", ""):
                LOGGER.info("Streaming Organic stdout/stderr (ORGANIC_STREAM_LOGS=1)")
                proc = subprocess.Popen(
                    cmd,
                    cwd=self._organic_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                assert proc.stdout is not None
                tail: list[str] = []
                deadline = time.monotonic() + self._timeout
                fd = proc.stdout.fileno()
                while proc.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        proc.kill()
                        raise DetectorExecutionError(f"Organic timed out after {self._timeout}s")
                    ready, _, _ = select.select([fd], [], [], min(1.0, remaining))
                    if ready:
                        line = proc.stdout.readline()
                        if line:
                            print(f"[organic] {line}", end="", flush=True)
                            tail.append(line.rstrip())
                            tail = tail[-25:]

                for line in proc.stdout:
                    print(f"[organic] {line}", end="", flush=True)
                    tail.append(line.rstrip())
                    tail = tail[-25:]
                returncode = proc.wait()

                if returncode != 0:
                    msg = "\n".join(tail[-10:])
                    LOGGER.error("Organic failed: %s", msg[:500])
                    raise DetectorExecutionError(
                        f"Organic exited with code {returncode}: {msg[:500]}"
                    )
            else:
                result = subprocess.run(
                    cmd,
                    cwd=self._organic_dir,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env=env,
                )

                if result.returncode != 0:
                    LOGGER.error("Organic failed: %s", result.stderr[:500])
                    raise DetectorExecutionError(
                        f"Organic exited with code {result.returncode}: {result.stderr[:200]}"
                    )

            if not output_path.exists() or output_path.stat().st_size == 0:
                return []

            raw = json.loads(output_path.read_text())
            return _parse_organic_json(raw)

        except subprocess.TimeoutExpired as e:
            raise DetectorExecutionError(f"Organic timed out after {self._timeout}s") from e
        except json.JSONDecodeError as e:
            raise DetectorExecutionError("Organic output is not valid JSON") from e
        finally:
            output_path.unlink(missing_ok=True)


def _parse_organic_json(raw: list[dict]) -> list[SmellEvent]:
    """Parse Organic JSON output into SmellEvents.

    Organic JSON format:
    [
      {
        "fullyQualifiedName": "com.example.Foo",
        "smells": [{"name": "GodClass", "reason": "...", "startingLine": 1, "endingLine": 100}],
        "methods": [
          {
            "fullyQualifiedName": "com.example.Foo.bar",
            "smells": [{"name": "LongMethod", ...}]
          }
        ]
      }
    ]
    """
    events: list[SmellEvent] = []
    seen: set[str] = set()

    for class_entry in raw:
        class_fqn = class_entry.get("fullyQualifiedName", "")
        file_path = class_fqn.replace(".", "/") + ".java" if class_fqn else ""

        # Class-level smells
        for smell in class_entry.get("smells", []):
            event = _smell_to_event(smell, file_path, class_fqn)
            if event and event.smell_id not in seen:
                seen.add(event.smell_id)
                events.append(event)

        # Method-level smells
        for method in class_entry.get("methods", []):
            method_fqn = method.get("fullyQualifiedName", "")
            for smell in method.get("smells", []):
                event = _smell_to_event(smell, file_path, class_fqn, method_fqn)
                if event and event.smell_id not in seen:
                    seen.add(event.smell_id)
                    events.append(event)

    return events


def _smell_to_event(
    smell: dict,
    file_path: str,
    class_name: str,
    method_fqn: str | None = None,
) -> SmellEvent | None:
    raw_name = smell.get("name", "")
    if not raw_name:
        return None

    canonical = normalize_dataset_smell_type(raw_name)
    line = smell.get("startingLine", 0) or 0

    return SmellEvent(
        smell_id=f"{canonical}:{file_path}:{line}",
        smell_type=canonical,
        severity=get_default_severity(canonical),
        file_path=file_path,
        line_number=line,
        class_name=class_name or None,
        method_signature=method_fqn.split(".")[-1] if method_fqn and "." in method_fqn else None,
        end_line=smell.get("endingLine"),
        detection_reason=smell.get("reason"),
    )
