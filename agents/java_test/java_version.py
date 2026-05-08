"""Maven/SDKMAN Java-version helpers shared by Java agents and workflows."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from functools import cache, lru_cache
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Iterator
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as ET
from langchain_core.prompts import PromptTemplate

LOGGER = logging.getLogger(__name__)

SDKMAN_ENV_TIMEOUT_SECONDS = 2
MAVEN_EFFECTIVE_POM_TIMEOUT_SECONDS = 2
JAVA_VERSION_PROMPT = PromptTemplate.from_template("Determine Java version for project: {project_info}")


def parse_bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "on", "y"}


def maven_binary() -> str:
    """Resolve the Maven binary to execute, preferring mvnd when available."""
    override = os.getenv("JAVA_TEST_MAVEN_BINARY", "auto").strip().lower()
    if override in {"", "auto"}:
        if shutil.which("mvnd"):
            return "mvnd"
        return "mvn"
    if override in {"mvn", "mvn.exe"}:
        return "mvn"
    if override in {"mvnd", "mvnd.exe"}:
        if shutil.which("mvnd"):
            return "mvnd"
        if shutil.which("mvn"):
            return "mvn"
        return override
    return override


def _is_test_goal_argument(arg: str) -> bool:
    if not arg.startswith("-"):
        return arg == "test" or arg.endswith(":test")
    return False


def _extract_surefire_system_properties() -> tuple[str, ...]:
    if not parse_bool_env(os.getenv("JAVA_TEST_MAVEN_PARALLEL_TESTS"), default=True):
        return ()

    defaults: list[tuple[str, str]] = [
        ("surefire.parallel", os.getenv("JAVA_TEST_SUREFIRE_PARALLEL", "classes").strip()),
        ("surefire.threadCount", os.getenv("JAVA_TEST_SUREFIRE_THREAD_COUNT", "2").strip()),
        ("surefire.forkCount", os.getenv("JAVA_TEST_SUREFIRE_FORK_COUNT", "1C").strip()),
        ("surefire.reuseForks", os.getenv("JAVA_TEST_SUREFIRE_REUSE_FORKS", "true").strip()),
    ]
    return tuple(f"-D{key}={value}" for key, value in defaults if value)


def _add_missing_props(argv: list[str], props: tuple[str, ...]) -> list[str]:
    existing = {arg.split("=", 1)[0] for arg in argv if arg.startswith("-D") and "=" in arg}
    insert_at = 1
    while insert_at < len(argv) and argv[insert_at].startswith("-"):
        insert_at += 1

    for prop in reversed(props):
        key = prop.split("=", 1)[0]
        if key not in existing:
            argv.insert(insert_at, prop)
    return argv


def enhance_maven_command(command: str, *, allow_test_optimizations: bool = True) -> str:
    tokens = shlex.split(command)
    if not tokens:
        return command

    if tokens[0] not in {"mvn", "mvn.exe", "mvnd", "mvnd.exe"}:
        return command

    tokens[0] = maven_binary()

    if allow_test_optimizations:
        goals = [arg for arg in tokens[1:] if not arg.startswith("-")]
        if any(_is_test_goal_argument(goal) for goal in goals):
            tokens = _add_missing_props(tokens, _extract_surefire_system_properties())

    return shlex.join(tokens)


def with_sdkman_maven_setup(command: str) -> str:
    """Wrap an eval command with SDKMAN Java 8 + Maven 3.6.3 setup."""
    enhanced = enhance_maven_command(command)
    return "\n".join(
        [
            "set -o pipefail",
            'if [ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]; then',
            '  source "$HOME/.sdkman/bin/sdkman-init.sh"',
            "  sdk use java 8.0.442-amzn >/dev/null",
            "  sdk use maven 3.6.3 >/dev/null",
            "fi",
            enhanced,
        ]
    )


@lru_cache(maxsize=1)
def sdkman_cached_env() -> MappingProxyType[str, str]:
    """Resolve SDKMAN-adjusted Java/Maven env once as an immutable mapping."""
    sdkman_init = Path.home() / ".sdkman" / "bin" / "sdkman-init.sh"
    if not sdkman_init.exists():
        return MappingProxyType({})

    command = "\n".join(
        [
            "set -o pipefail",
            'if [ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]; then',
            '  source "$HOME/.sdkman/bin/sdkman-init.sh"',
            "  sdk use java 8.0.442-amzn >/dev/null",
            "  sdk use maven 3.6.3 >/dev/null",
            "  env",
            "fi",
        ]
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=SDKMAN_ENV_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return MappingProxyType({})
    if result.returncode != 0 or not result.stdout.strip():
        return MappingProxyType({})

    wanted = {"JAVA_HOME", "M2_HOME", "MAVEN_HOME", "PATH"}
    env = {
        key: value
        for line in result.stdout.splitlines()
        for key, separator, value in [line.partition("=")]
        if separator and key in wanted and value
    }
    return MappingProxyType(env)


def sdkman_maven_command_env() -> dict[str, str]:
    """Return an executable environment preconfigured for Java 8 + Maven 3.6.3."""
    env = os.environ.copy()
    env.update(sdkman_cached_env())
    return env


def _clean_maven_evaluate_output(stdout: str) -> str | None:
    """Return a Maven help:evaluate scalar value, or None for unset/noisy output."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    candidates = [line for line in lines if not line.startswith("[")]
    value = candidates[-1] if candidates else lines[-1]
    invalid_markers = (
        "null object or invalid expression",
        "invalid expression",
        "not found",
        "${",
    )
    lowered = value.lower()
    if any(marker in lowered for marker in invalid_markers):
        return None
    return value


def _xml_local_name(tag: str) -> str:
    """Return the local XML tag name, ignoring Maven POM namespaces."""
    return tag.rsplit("}", 1)[-1]


def _child_text_by_local_name(parent: Element, child_name: str) -> str | None:
    for child in list(parent):
        if _xml_local_name(child.tag) != child_name:
            continue
        text = child.text.strip() if child.text is not None else ""
        return text or None
    return None


def _pom_property_candidates(root: Element) -> Iterator[str]:
    """Yield source/release values from Maven <properties> blocks."""
    for properties in root.iter():
        if _xml_local_name(properties.tag) != "properties":
            continue
        for key in ("maven.compiler.release", "maven.compiler.source", "maven.compiler.target"):
            value = _child_text_by_local_name(properties, key)
            if value is not None:
                yield value


def _pom_compiler_plugin_candidates(root: Element) -> Iterator[str]:
    """Yield source/release values from maven-compiler-plugin configuration."""
    for plugin in root.iter():
        if _xml_local_name(plugin.tag) != "plugin":
            continue
        artifact_id = _child_text_by_local_name(plugin, "artifactId")
        if artifact_id != "maven-compiler-plugin":
            continue
        for configuration in plugin.iter():
            if _xml_local_name(configuration.tag) != "configuration":
                continue
            for key in ("release", "source", "target"):
                value = _child_text_by_local_name(configuration, key)
                if value is not None:
                    yield value


def _is_resolved_maven_value(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return bool(lowered) and "${" not in lowered and "invalid expression" not in lowered


def _compiler_source_from_pom_file(pom_path: Path) -> str | None:
    try:
        root = ET.parse(pom_path).getroot()
        if root is None:
            return None
    except (ET.ParseError, OSError, UnicodeDecodeError) as exc:
        LOGGER.warning("Could not parse %s for Java source level: %s", pom_path, exc)
        return None
    value = next(chain(_pom_property_candidates(root), _pom_compiler_plugin_candidates(root)), None)
    return value if _is_resolved_maven_value(value) else None


def _pom_has_parent(pom_path: Path) -> bool:
    try:
        root = ET.parse(pom_path).getroot()
        if root is None:
            return False
    except (ET.ParseError, OSError, UnicodeDecodeError) as exc:
        LOGGER.warning("Could not parse %s for Maven parent detection: %s", pom_path, exc)
        return False
    return any(_xml_local_name(child.tag) == "parent" for child in list(root))


def _detect_maven_compiler_source_from_pom(project: Path) -> str | None:
    """Fallback source-level detection from the checkout pom.xml."""
    return _compiler_source_from_pom_file(project / "pom.xml")


def _maven_evaluate_property(project: Path, expression: str, timeout: int) -> str | None:
    command = with_sdkman_maven_setup(f"mvn -q help:evaluate -Dexpression={expression} -DforceStdout")
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning("Maven property detection failed for %s: %s", expression, exc)
        return None
    if result.returncode != 0:
        return None
    return _clean_maven_evaluate_output(result.stdout)


def _detect_maven_compiler_source_from_effective_pom(project: Path, timeout: int) -> str | None:
    """Detect source level from Maven's effective POM, including parents/profiles."""
    with tempfile.NamedTemporaryFile(prefix="effective-pom-", suffix=".xml", delete=True) as output_file:
        output_path = Path(output_file.name)
        command = with_sdkman_maven_setup(f"mvn help:effective-pom -Doutput={shlex.quote(str(output_path))}")
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=max(timeout, MAVEN_EFFECTIVE_POM_TIMEOUT_SECONDS),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOGGER.warning("Maven effective-pom detection failed: %s", exc)
            return None
        if result.returncode != 0 or not output_path.exists():
            return None
        return _compiler_source_from_pom_file(output_path)


@cache
def detect_maven_compiler_source(project_path: str, timeout: int = 2) -> str | None:
    """Detect Java source language level for Maven projects."""
    project = Path(project_path).resolve()
    pom_path = project / "pom.xml"
    if not pom_path.exists():
        return None

    pom_detected = _detect_maven_compiler_source_from_pom(project)
    if pom_detected is not None and not _pom_has_parent(pom_path):
        return pom_detected

    maven_expressions = ("maven.compiler.source", "maven.compiler.release", "maven.compiler.target")
    detected = next(
        (value for expression in maven_expressions if (value := _maven_evaluate_property(project, expression, timeout)) is not None),
        None,
    )
    return detected or _detect_maven_compiler_source_from_effective_pom(project, timeout) or pom_detected


def _format_java_version_prompt(project_info: str) -> str:
    return str(JAVA_VERSION_PROMPT.format(project_info=project_info))


def java_version_prompt_context(project_path: str) -> str:
    """Build deterministic Java-version context for repair/refactor prompts."""
    compiler_source = detect_maven_compiler_source(project_path)
    if compiler_source is None:
        return _format_java_version_prompt("maven.compiler.source could not be determined from Maven help:evaluate.")
    return "\n".join(
        [
            _format_java_version_prompt(f"Maven reports maven.compiler.source={compiler_source}."),
            f"Java compiler source level: {compiler_source}.",
            f"Do not use Java language features newer than source level {compiler_source}.",
        ]
    )


def reset_java_version_caches() -> None:
    """Clear Java-version/SDKMAN caches for tests and deterministic ad-hoc runs."""
    detect_maven_compiler_source.cache_clear()
    sdkman_cached_env.cache_clear()
