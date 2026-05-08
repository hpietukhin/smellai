"""Read-only accessor over Composite Refactorings 2020 Neo4j graph.

Uses py2neo. Dataset is never modified.

Graph schema (from probe):
  - (Element)-[:COMMITTED_IN]->(Commit)     Element is per-commit snapshot
  - (Element)-[:AFFECTED_BY]->(Smell)        Smell attached to element snapshot
  - (Refactoring)-[:STARTED_AT]->(Commit)    Parent commit (before refactoring)
  - (Refactoring)-[:ENDED_AT]->(Commit)      Child commit (after refactoring)
  - (Refactoring)-[:CHANGED]->(Element)      Element at STARTED_AT commit
  - (Refactoring)-[:PRODUCED]->(Element)     Element created by refactoring
  - (Commit)-[:BELONGS_TO]->(Project)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from py2neo import Graph

from domain.models import SmellEvent
from domain.rules import normalize_dataset_smell_type, get_default_severity


NEO4J_URI = "http://localhost:7474"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "boil2.eat"


@dataclass
class RefactoringRecord:
    """A single refactoring from the dataset graph."""
    ref_type: str
    hash_id: str
    commit_hash: str
    commit_order: int
    classification: str = ""
    degradation_level: str = ""
    changed_elements: list[str] = field(default_factory=list)
    produced_elements: list[str] = field(default_factory=list)


class DatasetGraph:
    """Read-only py2neo accessor over the Composite Refactorings 2020 Neo4j."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ) -> None:
        self._graph = Graph(uri, auth=(user, password))

    def is_available(self) -> bool:
        try:
            self._graph.run("RETURN 1").evaluate()
            return True
        except Exception:
            return False

    def smell_state(
        self,
        elements: set[str],
        commit_hash: str,
    ) -> list[SmellEvent]:
        """Smell snapshot for given element names at a given commit.

        Finds Element nodes COMMITTED_IN that commit, with matching names,
        then follows AFFECTED_BY to Smell nodes.
        """
        if not elements:
            return []

        rows = self._graph.run(
            """
            MATCH (e:Element)-[:COMMITTED_IN]->(c:Commit {hash: $commit}),
                  (e)-[:AFFECTED_BY]->(s:Smell)
            WHERE e.name IN $elements
            RETURN DISTINCT
                   s.hash_id AS smell_hash,
                   s.type    AS smell_type,
                   s.reason  AS reason,
                   s.starting_line AS starting_line,
                   s.ending_line   AS ending_line,
                   e.name AS element_name,
                   e.path AS element_path
            """,
            commit=commit_hash,
            elements=list(elements),
        ).data()

        seen: set[str] = set()
        result: list[SmellEvent] = []
        for row in rows:
            smell_hash = row["smell_hash"] or ""
            if smell_hash in seen:
                continue
            seen.add(smell_hash)

            raw_type = row["smell_type"] or ""
            canonical = normalize_dataset_smell_type(raw_type)
            element_name = row["element_name"] or ""
            element_path = row["element_path"] or ""

            # Extract class name from element FQN
            class_name = _class_name_from_element(element_name)

            result.append(SmellEvent(
                smell_id=smell_hash or f"{canonical}:{element_path}:{row.get('starting_line', 0)}",
                smell_type=canonical,
                severity=get_default_severity(canonical),
                file_path=element_path,
                line_number=row.get("starting_line") or 0,
                class_name=class_name or None,
                project=None,
                commit_hash=commit_hash,
                end_line=row.get("ending_line"),
                detection_reason=row.get("reason") or None,
            ))
        return result

    def refactorings_at_start(self, commit_hash: str) -> list[RefactoringRecord]:
        """All refactorings whose STARTED_AT is this commit."""
        rows = self._graph.run(
            """
            MATCH (r:Refactoring)-[:STARTED_AT]->(c:Commit {hash: $commit})
            RETURN r.type AS ref_type,
                   r.hash_id AS hash_id,
                   r.classification AS classification,
                   r.degradation_level AS degradation_level,
                   c.hash AS commit_hash,
                   c.order AS commit_order
            """,
            commit=commit_hash,
        ).data()

        return [
            RefactoringRecord(
                ref_type=r["ref_type"] or "",
                hash_id=r["hash_id"] or "",
                commit_hash=r["commit_hash"] or "",
                commit_order=r["commit_order"] or 0,
                classification=r["classification"] or "",
                degradation_level=r["degradation_level"] or "",
            )
            for r in rows
        ]


    def composite_refactoring(
        self,
        elements: set[str],
        project: str,
        *,
        max_steps: int = 50,
    ) -> list[CompositeStep]:
        """Extract a range-based composite refactoring for a set of elements.

        Finds all commits where these elements have refactorings (CHANGED/PRODUCED),
        orders by commit_order, and returns smell state + refactorings at each step.
        Includes both STARTED_AT (before) and ENDED_AT (after) commits.
        """
        if not elements:
            return []

        # Find all commits where refactorings touch these elements
        # Two queries: CHANGED and PRODUCED (Neo4j 3.1 lacks EXISTS subquery)
        all_rows = []
        for rel in ("CHANGED", "PRODUCED"):
            rows = self._graph.run(
                f"""
                MATCH (r:Refactoring)-[:{rel}]->(e:Element),
                      (r)-[:STARTED_AT]->(cs:Commit)-[:BELONGS_TO]->(p:Project {{name: $project}}),
                      (r)-[:ENDED_AT]->(ce:Commit)
                WHERE e.name IN $elements
                RETURN DISTINCT cs.hash AS start_hash, cs.order AS start_order,
                                ce.hash AS end_hash, ce.order AS end_order
                ORDER BY cs.order
                LIMIT $limit
                """,
                project=project,
                elements=list(elements),
                limit=max_steps,
            ).data()
            all_rows.extend(rows)
        rows = all_rows

        if not rows:
            return []

        # Collect unique commits (both start and end) ordered
        commit_map: dict[str, int] = {}  # hash → order
        for row in rows:
            commit_map[row["start_hash"]] = row["start_order"]
            commit_map[row["end_hash"]] = row["end_order"]

        ordered_commits = sorted(commit_map.items(), key=lambda x: x[1])

        # Bulk-load refactorings at these commits that touch the requested
        # elements, but retain *all* CHANGED/PRODUCED elements for those
        # refactorings.  This preserves developer intent for operations like
        # Extract Class, where the produced class is part of the after-state
        # even if the user supplied only the original class as input scope.
        all_commits = [h for h, _ in ordered_commits]
        ref_rows = self._graph.run(
            """
            MATCH (r:Refactoring)-[:STARTED_AT]->(c:Commit)
            WHERE c.hash IN $commits
            OPTIONAL MATCH (r)-[:CHANGED]->(ec:Element)
            WITH r, c, collect(DISTINCT ec.name) AS changed_names
            OPTIONAL MATCH (r)-[:PRODUCED]->(ep:Element)
            WITH r, c, changed_names,
                 collect(DISTINCT ep.name) AS produced_names,
                 changed_names + collect(DISTINCT ep.name) AS touched
            WHERE any(name IN touched WHERE name IN $elements)
            RETURN c.hash AS commit_hash,
                   r.type AS ref_type,
                   r.hash_id AS hash_id,
                   r.classification AS classification,
                   r.degradation_level AS degradation_level,
                   c.order AS commit_order,
                   changed_names,
                   produced_names
            """,
            commits=all_commits,
            elements=list(elements),
        ).data()

        # Group refactorings by commit
        from collections import defaultdict
        refs_by_commit: dict[str, list[RefactoringRecord]] = defaultdict(list)
        for r in ref_rows:
            refs_by_commit[r["commit_hash"]].append(RefactoringRecord(
                ref_type=r["ref_type"] or "",
                hash_id=r["hash_id"] or "",
                commit_hash=r["commit_hash"] or "",
                commit_order=r["commit_order"] or 0,
                classification=r["classification"] or "",
                degradation_level=r["degradation_level"] or "",
                changed_elements=[n for n in (r["changed_names"] or []) if n],
                produced_elements=[n for n in (r["produced_names"] or []) if n],
            ))

        # Build smell-state scope: requested elements plus all elements that the
        # developer actually changed/produced in this range.
        smell_scope = set(elements)
        for r in ref_rows:
            smell_scope.update(n for n in (r["changed_names"] or []) if n)
            smell_scope.update(n for n in (r["produced_names"] or []) if n)

        # Build steps
        steps: list[CompositeStep] = []
        for commit_hash, commit_order in ordered_commits:
            smells = self.smell_state(smell_scope, commit_hash)
            steps.append(CompositeStep(
                commit_hash=commit_hash,
                commit_order=commit_order,
                smells=smells,
                refactorings=refs_by_commit.get(commit_hash, []),
            ))

        return steps[:max_steps]


@dataclass
class CompositeStep:
    """One commit in a composite refactoring: smell state + refactorings applied."""
    commit_hash: str
    commit_order: int
    smells: list[SmellEvent]
    refactorings: list[RefactoringRecord]


def _class_name_from_element(element_name: str) -> str:
    """Best-effort class FQN from dataset element FQN."""
    if not element_name:
        return ""
    paren = element_name.find("(")
    base = element_name[:paren] if paren != -1 else element_name
    dot = base.rfind(".")
    if dot == -1:
        return base
    tail = base[dot + 1:]
    if tail and tail[0].islower():
        return base[:dot]
    return base
