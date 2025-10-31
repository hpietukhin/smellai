"""
MySQL database connector for DACOS dataset.

This module provides connection pooling and data access functions for the DACOS
(Detection And Correction Of Smells) database containing ground truth code smell
annotations.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from mysql.connector import Error as MySQLError
from mysql.connector import pooling

from src.models.entities import DACOSSample

# Load environment variables
load_dotenv()

# Global connection pool (lazy initialization)
_connection_pool: Optional[pooling.MySQLConnectionPool] = None


_SMELL_FLAG_COLUMNS: Sequence[Tuple[str, str]] = (
    ("Complex Method", "a.iscm"),
    ("Long Parameter List", "a.islp"),
    ("Multifaceted Abstraction", "a.isma"),
)


def _bit_to_bool(value: Any) -> bool:
    """Convert MySQL BIT column values into Python booleans.

    mysql-connector returns BIT columns as byte-like objects (e.g. ``b"\x01"``),
    so a direct ``bool(value)`` would incorrectly evaluate to ``True`` even
    when the bit is ``0``. This helper normalises the various shapes into a
    proper boolean.
    """

    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, memoryview):
        return _bit_to_bool(value.tobytes())

    if isinstance(value, (bytes, bytearray)):
        return any(value)

    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def get_connection_pool() -> pooling.MySQLConnectionPool:
    """
    Get or create MySQL connection pool.

    Creates a connection pool with 5 connections on first call, returns cached
    pool on subsequent calls.

    Returns:
        MySQLConnectionPool instance

    Raises:
        ValueError: If required environment variables are missing
        MySQLError: If connection pool creation fails
    """
    global _connection_pool

    if _connection_pool is not None:
        return _connection_pool

    # Read environment variables
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_port = os.getenv("MYSQL_PORT", "3306")
    mysql_database = os.getenv("MYSQL_DATABASE")
    mysql_user = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")
    mysql_pool_size = int(os.getenv("MYSQL_POOL_SIZE", "5"))

    # Validate required variables
    missing_vars = []
    if not mysql_host:
        missing_vars.append("MYSQL_HOST")
    if not mysql_database:
        missing_vars.append("MYSQL_DATABASE")
    if not mysql_user:
        missing_vars.append("MYSQL_USER")
    if mysql_password is None:
        missing_vars.append("MYSQL_PASSWORD")

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            "Please set them in your .env file."
        )

    try:
        # Create connection pool
        _connection_pool = pooling.MySQLConnectionPool(
            pool_name="dacos_pool",
            pool_size=mysql_pool_size,
            pool_reset_session=True,
            host=mysql_host,
            port=int(mysql_port),
            database=mysql_database,
            user=mysql_user,
            password=mysql_password,
            autocommit=True,
        )

        print(
            f"✓ MySQL connection pool created (size={mysql_pool_size}): {mysql_user}@{mysql_host}:{mysql_port}/{mysql_database}"
        )
        return _connection_pool

    except MySQLError as e:
        raise MySQLError(
            f"Failed to create MySQL connection pool: {e}. "
            "Please check your database credentials and ensure MySQL is running."
        ) from e


def get_connection():
    """
    Get a connection from the pool.

    Returns:
        MySQL connection object

    Raises:
        MySQLError: If unable to get connection from pool
    """
    pool = get_connection_pool()
    try:
        return pool.get_connection()
    except MySQLError as e:
        raise MySQLError(f"Failed to get connection from pool: {e}") from e


def _row_to_sample(row: Dict[str, Any]) -> DACOSSample:
    """Convert a SQL row into a ``DACOSSample`` with normalised types."""

    sample_constraints = row.get("sample_constraints")
    if sample_constraints is not None:
        sample_constraints = str(sample_constraints)

    return DACOSSample(
        id=row["id"],
        designite_id=row.get("designite_id"),
        has_smell=_bit_to_bool(row.get("has_smell")),
        is_class=_bit_to_bool(row.get("is_class")),
        path_to_file=row["path_to_file"],
        project_name=row["project_name"],
        sample_constraints=sample_constraints,
        smells=row.get("smells"),
        iscm=_bit_to_bool(row.get("iscm")),
        isim=_bit_to_bool(row.get("isim")),
        islp=_bit_to_bool(row.get("islp")),
        isma=_bit_to_bool(row.get("isma")),
        smell_name=row.get("smell_name"),
        smell_description=row.get("smell_description"),
        repo_url=row.get("repo_url"),
        commit_sha=row.get("commit_sha"),
    )


def fetch_sample_by_id(sample_id: int) -> Optional[DACOSSample]:
    """
    Fetch a complete DACOS sample record with annotations by ID.

    Executes a JOIN query to get sample data with smell annotations and
    smell type information.

    Args:
        sample_id: The sample ID to fetch

    Returns:
        DACOSSample object if found, None otherwise

    Raises:
        MySQLError: If database query fails
    """
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        # SQL query with JOINs to get complete sample data
        query = """
        SELECT DISTINCT
            s.id,
            s.designite_id,
            s.has_smell,
            s.is_class,
            s.path_to_file,
            s.project_name,
            s.sample_constraints,
            s.smells,
            a.iscm,
            a.isim,
            a.islp,
            a.isma,
            sm.name AS smell_name,
            sm.description AS smell_description
        FROM tagman5.sample s
        LEFT JOIN tagman5.annotation a ON s.id = a.sample_id
        LEFT JOIN tagman5.smell sm
            ON (
                (
                    s.smells IS NOT NULL
                    AND s.smells <> ''
                    AND FIND_IN_SET(
                        CAST(sm.id AS CHAR),
                        REPLACE(s.smells, ' ', '')
                    )
                )
                OR s.smells = sm.name
            )
        WHERE s.id = %s
        """

        cursor.execute(query, (sample_id,))
        rows = cursor.fetchall()

        if not rows:
            return None

        # Some samples have multiple smell rows; prefer the first non-null smell info
        primary = rows[0]
        smell_name = primary.get("smell_name")
        smell_description = primary.get("smell_description")

        if smell_name is None:
            for candidate in rows[1:]:
                if candidate.get("smell_name"):
                    smell_name = candidate.get("smell_name")
                    smell_description = candidate.get("smell_description")
                    break

        # Aggregate smell flags across all rows to account for multi-annotation records.
        aggregated = dict(primary)
        aggregated["smell_name"] = smell_name
        aggregated["smell_description"] = smell_description
        aggregated["iscm"] = any(_bit_to_bool(row.get("iscm")) for row in rows)
        aggregated["isim"] = any(_bit_to_bool(row.get("isim")) for row in rows)
        aggregated["islp"] = any(_bit_to_bool(row.get("islp")) for row in rows)
        aggregated["isma"] = any(_bit_to_bool(row.get("isma")) for row in rows)

        return _row_to_sample(aggregated)

    except MySQLError as e:
        raise MySQLError(f"Failed to fetch sample {sample_id}: {e}") from e

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def fetch_samples(
    project_name: Optional[str] = None,
    has_smell: Optional[bool] = None,
    limit: int = 100,
) -> List[DACOSSample]:
    """
    Fetch multiple samples based on filter criteria.

    Builds a dynamic SQL query based on provided filters and returns
    matching samples with annotations.

    Args:
        project_name: Filter by project name (e.g., "alibaba_arthas")
        has_smell: Filter by whether sample contains smells
        limit: Maximum number of samples to return (default: 100)

    Returns:
        List of DACOSSample objects (empty list if no matches)

    Raises:
        MySQLError: If database query fails
    """
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        # Build base query
        query = """
        SELECT
            s.id,
            s.designite_id,
            s.has_smell,
            s.is_class,
            s.path_to_file,
            s.project_name,
            s.sample_constraints,
            s.smells,
            a.iscm,
            a.isim,
            a.islp,
            a.isma,
            sm.name AS smell_name,
            sm.description AS smell_description
        FROM tagman5.sample s
        LEFT JOIN tagman5.annotation a ON s.id = a.sample_id
        LEFT JOIN tagman5.smell sm
            ON (
                (
                    s.smells IS NOT NULL
                    AND s.smells <> ''
                    AND FIND_IN_SET(
                        CAST(sm.id AS CHAR),
                        REPLACE(s.smells, ' ', '')
                    )
                )
                OR s.smells = sm.name
            )
        WHERE 1=1
        """

        # Build WHERE clauses and parameters
        params = []

        if project_name is not None:
            query += " AND s.project_name = %s"
            params.append(project_name)

        if has_smell is not None:
            query += " AND s.has_smell = %s"
            params.append(1 if has_smell else 0)

        # Add limit
        query += " LIMIT %s"
        params.append(limit)

        # Execute query
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        # Convert rows to DACOSSample objects
        samples = [_row_to_sample(row) for row in rows]

        return samples

    except MySQLError as e:
        raise MySQLError(f"Failed to fetch samples: {e}") from e

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _collect_balanced_sample_ids(per_smell: int) -> Dict[str, List[int]]:
    """Return a mapping from smell labels to sample identifiers."""

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        query_template = """
        SELECT DISTINCT s.id
        FROM tagman5.sample s
        JOIN tagman5.annotation a ON s.id = a.sample_id
        WHERE {condition}
        ORDER BY s.id
        LIMIT %s
        """

        smell_to_ids: Dict[str, List[int]] = {}
        for smell_label, flag_column in _SMELL_FLAG_COLUMNS:
            cursor.execute(
                query_template.format(condition=f"{flag_column} = 1"), (per_smell,)
            )
            rows = cursor.fetchall()
            ids = [row[0] if not isinstance(row, dict) else row["id"] for row in rows]
            smell_to_ids[smell_label] = ids

        return smell_to_ids

    except MySQLError as e:
        raise MySQLError(
            f"Failed to fetch balanced smell sample identifiers: {e}"
        ) from e

    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _hydrate_samples_with_labels(
    smell_to_ids: Dict[str, List[int]]
) -> List[Tuple[DACOSSample, str]]:
    """Fetch full sample objects for the supplied identifier mapping."""

    samples: List[Tuple[DACOSSample, str]] = []
    for smell_label, sample_ids in smell_to_ids.items():
        for sample_id in sample_ids:
            sample = fetch_sample_by_id(sample_id)
            if sample is None:
                continue
            samples.append((sample, smell_label))
    return samples


def fetch_balanced_smell_samples(per_smell: int = 5) -> List[Tuple[DACOSSample, str]]:
    """Return up to ``per_smell`` samples per smell, tagged with the target smell name.

    A sample that exhibits multiple smells may appear more than once, each time paired
    with a different smell label.
    """

    if per_smell <= 0:
        return []

    smell_to_ids = _collect_balanced_sample_ids(per_smell)
    return _hydrate_samples_with_labels(smell_to_ids)


def fetch_samples_dataframe(
    smell_ids: Optional[List[int]] = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Fetch sample rows as a pandas DataFrame.

    Mirrors the exploratory query used in the reference notebook while
    reusing the shared connection pool. Optional ``smell_ids`` filtering
    aligns with ``WHERE smells IN (...)`` from the original snippet.

    Args:
        smell_ids: Optional list of smell IDs to filter (``smells`` column).
        limit: Maximum number of rows to return.

    Returns:
        DataFrame with the requested sample rows (empty if none match).

    Raises:
        MySQLError: If database access or query execution fails.
    """

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = (
            "SELECT id, designite_id, has_smell, is_class, path_to_file, "
            "project_name, sample_constraints, smells "
            "FROM tagman5.sample"
        )

        clauses: List[str] = []
        params: List[object] = []

        if smell_ids:
            placeholders = ", ".join(["%s"] * len(smell_ids))
            clauses.append(f"smells IN ({placeholders})")
            params.extend(smell_ids)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " LIMIT %s"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        df = pd.DataFrame(rows)

        if not df.empty:
            for column in ("has_smell", "is_class"):
                if column in df.columns:
                    df[column] = df[column].apply(_bit_to_bool)

            if "sample_constraints" in df.columns:
                df["sample_constraints"] = pd.to_numeric(
                    df["sample_constraints"], errors="coerce"
                )

        return df

    except MySQLError as e:
        raise MySQLError(f"Failed to fetch samples dataframe: {e}") from e

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def test_connection() -> bool:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"✗ MySQL connectivity failed: {e}")
        return False


# Async wrappers for non-blocking database operations


async def fetch_sample_by_id_async(sample_id: int) -> Optional[DACOSSample]:
    """Async wrapper for fetch_sample_by_id that runs in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_sample_by_id, sample_id)


async def fetch_samples_async(
    project_name: Optional[str] = None,
    has_smell: Optional[bool] = None,
    limit: int = 100,
) -> List[DACOSSample]:
    """Async wrapper for fetch_samples that runs in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: fetch_samples(
            project_name=project_name, has_smell=has_smell, limit=limit
        ),
    )


async def fetch_samples_dataframe_async(
    smell_ids: Optional[List[int]] = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Async wrapper for fetch_samples_dataframe that runs in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: fetch_samples_dataframe(smell_ids=smell_ids, limit=limit)
    )


async def fetch_balanced_smell_samples_async(
    per_smell: int = 5,
) -> List[Tuple[DACOSSample, str]]:
    """Async wrapper for fetch_balanced_smell_samples that runs in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_balanced_smell_samples, per_smell)
