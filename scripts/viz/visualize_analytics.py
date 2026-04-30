#!/usr/bin/env python3
"""Visualization script for composite refactoring analytics database.

Usage:
    uv run python scripts/visualize_analytics.py analytics.db
    uv run python scripts/visualize_analytics.py analytics.db --session <session_id>
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NO_REFACTORING_ATTEMPTS = "No refactoring attempts found"

sns.set_theme(style="whitegrid")


def plot_smell_evolution(db_path: str, session_id: str, output_dir: Path):
    """Plot smell counts over iterations for a specific session."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      iteration,
      action as status,
      COUNT(*) as count
    FROM smell_events
    WHERE session_id = ?
    GROUP BY iteration, action
    ORDER BY iteration, action
    """

    df = pd.read_sql_query(query, conn, params=(session_id,))
    conn.close()

    if df.empty:
        print(f"No smell events found for session {session_id}")
        return

    # Pivot for easier plotting
    pivot_df = df.pivot_table(
        index="iteration", columns="status", values="count", fill_value=0
    )

    plt.figure(figsize=(12, 6))
    for status in pivot_df.columns:
        plt.plot(
            pivot_df.index,
            pivot_df[status],
            marker="o",
            label=status.capitalize(),
            linewidth=2,
        )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Smell Count", fontsize=12)
    plt.title(
        f"Smell Evolution - Session {session_id[:8]}", fontsize=14, fontweight="bold"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"smell_evolution_{session_id[:8]}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_refactoring_outcomes(db_path: str, output_dir: Path):
    """Plot refactoring outcomes across all sessions."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      session_id,
      outcome,
      COUNT(*) as count
    FROM refactoring_attempts
    GROUP BY session_id, outcome
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(NO_REFACTORING_ATTEMPTS)
        return

    # Pivot for stacked bar chart
    pivot_df = df.pivot_table(
        index="session_id", columns="outcome", values="count", fill_value=0
    )

    plt.figure(figsize=(14, 6))
    pivot_df.plot(
        kind="bar", stacked=True, ax=plt.gca(), color=["#2ecc71", "#e74c3c", "#f39c12"]
    )
    plt.xlabel("Session", fontsize=12)
    plt.ylabel("Refactoring Attempts", fontsize=12)
    plt.title("Refactoring Outcomes by Session", fontsize=14, fontweight="bold")
    plt.legend(title="Outcome")
    plt.xticks(
        range(len(pivot_df)),
        [sid[:8] for sid in pivot_df.index],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()

    output_path = output_dir / "refactoring_outcomes_by_session.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_success_rate(db_path: str, output_dir: Path):
    """Plot success rate across all sessions."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      session_id,
      SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
      COUNT(*) as total_attempts
    FROM refactoring_attempts
    GROUP BY session_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(NO_REFACTORING_ATTEMPTS)
        return

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(df)), df["success_rate"], color="#3498db")
    plt.xlabel("Session", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.title("Refactoring Success Rate by Session", fontsize=14, fontweight="bold")
    plt.axhline(y=50, color="r", linestyle="--", label="50% Baseline", linewidth=2)
    plt.legend()
    plt.xticks(
        range(len(df)), [sid[:8] for sid in df["session_id"]], rotation=45, ha="right"
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df["success_rate"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylim(0, 110)
    plt.tight_layout()

    output_path = output_dir / "success_rate_by_session.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_token_usage(db_path: str, output_dir: Path, session_id: str | None = None):
    """Plot token usage by node."""
    conn = sqlite3.connect(db_path)

    if session_id:
        query = """
        SELECT 
          node_name,
          SUM(total_tokens) as total_tokens,
          SUM(prompt_tokens) as prompt_tokens,
          SUM(completion_tokens) as completion_tokens
        FROM token_usage
        WHERE session_id = ?
        GROUP BY node_name
        """
        df = pd.read_sql_query(query, conn, params=(session_id,))
        title_suffix = f" - Session {session_id[:8]}"
        filename = f"token_usage_{session_id[:8]}.png"
    else:
        query = """
        SELECT 
          node_name,
          SUM(total_tokens) as total_tokens,
          SUM(prompt_tokens) as prompt_tokens,
          SUM(completion_tokens) as completion_tokens
        FROM token_usage
        GROUP BY node_name
        """
        df = pd.read_sql_query(query, conn)
        title_suffix = " - All Sessions"
        filename = "token_usage_distribution.png"

    conn.close()

    if df.empty:
        print(
            f"No token usage data found{' for session ' + session_id if session_id else ''}"
        )
        return

    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    ax1.pie(
        df["total_tokens"],
        labels=df["node_name"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax1.set_title(
        f"Token Distribution by Node{title_suffix}", fontsize=12, fontweight="bold"
    )

    # Stacked bar chart
    df_plot = df.set_index("node_name")[["prompt_tokens", "completion_tokens"]]
    df_plot.plot(kind="bar", stacked=True, ax=ax2, color=["#3498db", "#e74c3c"])
    ax2.set_xlabel("Node", fontsize=11)
    ax2.set_ylabel("Tokens", fontsize=11)
    ax2.set_title(f"Token Breakdown{title_suffix}", fontsize=12, fontweight="bold")
    ax2.legend(["Prompt", "Completion"])
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_smell_resolution_rate(db_path: str, output_dir: Path):
    """Plot smell resolution vs creation rate."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      session_id,
      SUM(smells_resolved) as total_resolved,
      SUM(smells_created) as total_created
    FROM refactoring_attempts
    WHERE outcome = 'success'
    GROUP BY session_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No successful refactoring attempts found")
        return

    df["net_improvement"] = df["total_resolved"] - df["total_created"]

    plt.figure(figsize=(14, 6))
    x = range(len(df))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        df["total_resolved"],
        width,
        label="Resolved",
        color="#2ecc71",
    )
    plt.bar(
        [i + width / 2 for i in x],
        df["total_created"],
        width,
        label="Created",
        color="#e74c3c",
    )

    plt.xlabel("Session", fontsize=12)
    plt.ylabel("Smell Count", fontsize=12)
    plt.title(
        "Smells Resolved vs Created (Successful Refactorings)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend()
    plt.xticks(x, [sid[:8] for sid in df["session_id"]], rotation=45, ha="right")
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    plt.tight_layout()

    output_path = output_dir / "smell_resolution_rate.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_iteration_distribution(db_path: str, output_dir: Path):
    """Plot distribution of iterations across sessions."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      session_id,
      MAX(iteration) + 1 as iterations
    FROM refactoring_attempts
    GROUP BY session_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(NO_REFACTORING_ATTEMPTS)
        return

    plt.figure(figsize=(10, 6))
    plt.hist(
        df["iterations"],
        bins=range(1, df["iterations"].max() + 2),
        edgecolor="black",
        color="#9b59b6",
    )
    plt.xlabel("Number of Iterations", fontsize=12)
    plt.ylabel("Session Count", fontsize=12)
    plt.title("Distribution of Refactoring Iterations", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    output_path = output_dir / "iteration_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")


def print_summary_stats(db_path: str):
    """Print summary statistics."""
    conn = sqlite3.connect(db_path)

    # Session count
    cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM refactoring_attempts")
    session_count = cursor.fetchone()[0]

    # Total attempts
    cursor = conn.execute("SELECT COUNT(*) FROM refactoring_attempts")
    total_attempts = cursor.fetchone()[0]

    # Success rate
    cursor = conn.execute("""
        SELECT 
          SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
        FROM refactoring_attempts
    """)
    success_rate = cursor.fetchone()[0]

    # Total smells
    cursor = conn.execute(
        "SELECT SUM(smells_resolved), SUM(smells_created) FROM refactoring_attempts"
    )
    resolved, created = cursor.fetchone()

    # Total tokens
    cursor = conn.execute("SELECT SUM(total_tokens) FROM token_usage")
    total_tokens = cursor.fetchone()[0] or 0

    conn.close()

    print("\n" + "=" * 60)
    print("ANALYTICS SUMMARY")
    print("=" * 60)
    print(f"Total Sessions: {session_count}")
    print(f"Total Refactoring Attempts: {total_attempts}")
    print(f"Overall Success Rate: {success_rate:.2f}%")
    print(f"Total Smells Resolved: {resolved or 0}")
    print(f"Total Smells Created: {created or 0}")
    print(f"Net Smell Reduction: {(resolved or 0) - (created or 0)}")
    print(f"Total Tokens Used: {total_tokens:,}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize composite refactoring analytics"
    )
    parser.add_argument("db_path", help="Path to analytics database")
    parser.add_argument("--session", help="Specific session ID to visualize")
    parser.add_argument(
        "--output-dir", default="./visualizations", help="Output directory for plots"
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations from {db_path}...")
    print(f"Output directory: {output_dir}\n")

    # Print summary stats
    print_summary_stats(str(db_path))

    # Generate plots
    if args.session:
        # Session-specific plots
        plot_smell_evolution(str(db_path), args.session, output_dir)
        plot_token_usage(str(db_path), output_dir, args.session)
    else:
        # Aggregate plots
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT DISTINCT session_id FROM smell_events LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            # Generate smell evolution for first session as example
            plot_smell_evolution(str(db_path), row[0], output_dir)

        plot_refactoring_outcomes(str(db_path), output_dir)
        plot_success_rate(str(db_path), output_dir)
        plot_token_usage(str(db_path), output_dir)
        plot_smell_resolution_rate(str(db_path), output_dir)
        plot_iteration_distribution(str(db_path), output_dir)

    print(f"\n✅ All visualizations saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
