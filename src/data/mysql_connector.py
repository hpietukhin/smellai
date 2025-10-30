"""
MySQL database connector for DACOS dataset.

This module provides connection pooling and data access functions for the DACOS
(Detection And Correction Of Smells) database containing ground truth code smell
annotations.
"""

import os
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from mysql.connector import Error as MySQLError
from mysql.connector import pooling

from src.models.entities import DACOSSample

# Load environment variables
load_dotenv()

# Global connection pool (lazy initialization)
_connection_pool: Optional[pooling.MySQLConnectionPool] = None


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
        LEFT JOIN tagman5.smell sm ON s.smells = sm.id
        WHERE s.id = %s
        """

        cursor.execute(query, (sample_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        # Convert row to DACOSSample
        # Handle NULL values for boolean fields
        sample = DACOSSample(
            id=row["id"],
            designite_id=row.get("designite_id"),
            has_smell=bool(row["has_smell"]),
            is_class=bool(row["is_class"]),
            path_to_file=row["path_to_file"],
            project_name=row["project_name"],
            sample_constraints=row.get("sample_constraints"),
            smells=row.get("smells"),
            iscm=bool(row.get("iscm", False)),
            isim=bool(row.get("isim", False)),
            islp=bool(row.get("islp", False)),
            isma=bool(row.get("isma", False)),
            smell_name=row.get("smell_name"),
            smell_description=row.get("smell_description"),
        )

        return sample

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
        LEFT JOIN tagman5.smell sm ON s.smells = sm.id
        WHERE 1=1
        """

        # Build WHERE clauses and parameters
        params = []

        if project_name is not None:
            query += " AND s.project_name = %s"
            params.append(project_name)

        if has_smell is not None:
            query += " AND s.has_smell = %s"
            params.append(has_smell)

        # Add limit
        query += " LIMIT %s"
        params.append(limit)

        # Execute query
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        # Convert rows to DACOSSample objects
        samples = []
        for row in rows:
            sample = DACOSSample(
                id=row["id"],
                designite_id=row.get("designite_id"),
                has_smell=bool(row["has_smell"]),
                is_class=bool(row["is_class"]),
                path_to_file=row["path_to_file"],
                project_name=row["project_name"],
                sample_constraints=row.get("sample_constraints"),
                smells=row.get("smells"),
                iscm=bool(row.get("iscm", False)),
                isim=bool(row.get("isim", False)),
                islp=bool(row.get("islp", False)),
                isma=bool(row.get("isma", False)),
                smell_name=row.get("smell_name"),
                smell_description=row.get("smell_description"),
            )
            samples.append(sample)

        return samples

    except MySQLError as e:
        raise MySQLError(f"Failed to fetch samples: {e}") from e

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


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

        try:
            connection = get_connection()

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

            df = pd.read_sql(query, connection, params=params)

            if not df.empty:
                df["has_smell"] = df["has_smell"].astype(bool)
                df["is_class"] = df["is_class"].astype(bool)

            return df

        except MySQLError as e:
            raise MySQLError(f"Failed to fetch samples dataframe: {e}") from e

        finally:
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
