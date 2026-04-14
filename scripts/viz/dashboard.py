#!/usr/bin/env python3
"""Interactive Streamlit dashboard for composite refactoring analytics.

Usage:
    uv run streamlit run scripts/dashboard.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Composite Refactoring Analytics", page_icon="🔍", layout="wide"
)


@st.cache_data
def load_sessions(db_path: str) -> pd.DataFrame:
    """Load session overview data."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      ra.session_id,
      COUNT(DISTINCT ra.iteration) as iterations,
      SUM(CASE WHEN ra.outcome = 'success' THEN 1 ELSE 0 END) as successful_refactorings,
      SUM(CASE WHEN ra.outcome = 'compile_failed' THEN 1 ELSE 0 END) as compile_failures,
      SUM(CASE WHEN ra.outcome = 'test_failed' THEN 1 ELSE 0 END) as test_failures,
      SUM(ra.smells_resolved) as total_resolved,
      SUM(ra.smells_created) as total_created,
      COALESCE(SUM(tu.total_tokens), 0) as total_tokens
    FROM refactoring_attempts ra
    LEFT JOIN token_usage tu ON ra.session_id = tu.session_id
    GROUP BY ra.session_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["net_improvement"] = df["total_resolved"] - df["total_created"]
    df["success_rate"] = (
        df["successful_refactorings"]
        / (df["successful_refactorings"] + df["compile_failures"] + df["test_failures"])
    ) * 100

    return df


@st.cache_data
def load_smell_events(db_path: str, session_id: str) -> pd.DataFrame:
    """Load smell events for a specific session."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      iteration,
      smell_type,
      location,
      severity,
      status,
      timestamp
    FROM smell_events
    WHERE session_id = ?
    ORDER BY iteration, timestamp
    """

    df = pd.read_sql_query(query, conn, params=(session_id,))
    conn.close()

    return df


@st.cache_data
def load_refactoring_attempts(db_path: str, session_id: str) -> pd.DataFrame:
    """Load refactoring attempts for a specific session."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      iteration,
      smell_id,
      refactoring_type,
      outcome,
      retries,
      smells_resolved,
      smells_created,
      timestamp
    FROM refactoring_attempts
    WHERE session_id = ?
    ORDER BY iteration
    """

    df = pd.read_sql_query(query, conn, params=(session_id,))
    conn.close()

    return df


@st.cache_data
def load_token_usage(db_path: str, session_id: str) -> pd.DataFrame:
    """Load token usage for a specific session."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
      iteration,
      node_name,
      prompt_tokens,
      completion_tokens,
      total_tokens,
      model,
      timestamp
    FROM token_usage
    WHERE session_id = ?
    ORDER BY timestamp
    """

    df = pd.read_sql_query(query, conn, params=(session_id,))
    conn.close()

    return df


def main():
    st.title("🔍 Composite Refactoring Analytics Dashboard")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    db_path = st.sidebar.text_input("Database Path", value="analytics.db")

    if not Path(db_path).exists():
        st.error(f"Database not found: {db_path}")
        st.info(
            "Please run a composite refactoring workflow to generate analytics data."
        )
        return

    # Load sessions
    try:
        sessions_df = load_sessions(db_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if sessions_df.empty:
        st.warning("No analytics data found in database.")
        return

    # Overview section
    st.header("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sessions", len(sessions_df))

    with col2:
        avg_success = sessions_df["success_rate"].mean()
        st.metric("Avg Success Rate", f"{avg_success:.1f}%")

    with col3:
        total_resolved = sessions_df["total_resolved"].sum()
        st.metric("Total Smells Resolved", int(total_resolved))

    with col4:
        total_tokens = sessions_df["total_tokens"].sum()
        st.metric("Total Tokens Used", f"{int(total_tokens):,}")

    # Sessions table
    st.subheader("Sessions")

    display_df = sessions_df.copy()
    display_df["session_id"] = display_df["session_id"].str[:8]
    display_df = display_df.rename(
        columns={
            "session_id": "Session",
            "iterations": "Iterations",
            "successful_refactorings": "Successes",
            "compile_failures": "Compile Fails",
            "test_failures": "Test Fails",
            "total_resolved": "Resolved",
            "total_created": "Created",
            "net_improvement": "Net Improvement",
            "total_tokens": "Tokens",
            "success_rate": "Success Rate (%)",
        }
    )

    st.dataframe(
        display_df.style.background_gradient(
            subset=["Success Rate (%)"], cmap="RdYlGn"
        ).format({"Success Rate (%)": "{:.1f}", "Tokens": "{:,}"}),
        use_container_width=True,
    )

    # Session selector
    st.sidebar.header("Session Details")
    session_options = {
        f"{sid[:8]} ({row['iterations']} iter)": sid
        for sid, row in sessions_df.set_index("session_id").iterrows()
    }

    if not session_options:
        st.warning("No sessions available")
        return

    selected_label = st.sidebar.selectbox(
        "Select Session", list(session_options.keys())
    )
    selected_session = session_options[selected_label]

    # Session details
    st.header(f"📝 Session Details: {selected_session[:8]}")

    # Load session data
    smell_events_df = load_smell_events(db_path, selected_session)
    refactoring_attempts_df = load_refactoring_attempts(db_path, selected_session)
    token_usage_df = load_token_usage(db_path, selected_session)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Smells", "🔄 Refactorings", "💰 Tokens", "📊 Statistics"]
    )

    with tab1:
        st.subheader("Smell Evolution")

        if not smell_events_df.empty:
            # Smell counts by iteration
            smell_counts = (
                smell_events_df.groupby(["iteration", "status"])
                .size()
                .reset_index(name="count")
            )

            fig = px.line(
                smell_counts,
                x="iteration",
                y="count",
                color="status",
                markers=True,
                title="Smell Count by Iteration",
                labels={
                    "count": "Smell Count",
                    "iteration": "Iteration",
                    "status": "Status",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Smell types distribution
            col1, col2 = st.columns(2)

            with col1:
                smell_types = smell_events_df["smell_type"].value_counts()
                fig = px.pie(
                    values=smell_types.values,
                    names=smell_types.index,
                    title="Smell Types Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                severity_counts = smell_events_df["severity"].value_counts()
                fig = px.bar(
                    x=severity_counts.index,
                    y=severity_counts.values,
                    title="Smell Severity Distribution",
                    labels={"x": "Severity", "y": "Count"},
                    color=severity_counts.index,
                    color_discrete_map={
                        "HIGH": "red",
                        "MEDIUM": "orange",
                        "LOW": "yellow",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed smell events table
            st.subheader("Smell Events Details")
            st.dataframe(smell_events_df, use_container_width=True)
        else:
            st.info("No smell events recorded for this session")

    with tab2:
        st.subheader("Refactoring Attempts")

        if not refactoring_attempts_df.empty:
            # Outcome by iteration
            fig = px.bar(
                refactoring_attempts_df,
                x="iteration",
                color="outcome",
                title="Refactoring Outcomes by Iteration",
                labels={"iteration": "Iteration"},
                color_discrete_map={
                    "success": "#2ecc71",
                    "compile_failed": "#e74c3c",
                    "test_failed": "#f39c12",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Smell resolution progress
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=refactoring_attempts_df["iteration"],
                    y=refactoring_attempts_df["smells_resolved"].cumsum(),
                    mode="lines+markers",
                    name="Resolved",
                    line=dict(color="green", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=refactoring_attempts_df["iteration"],
                    y=refactoring_attempts_df["smells_created"].cumsum(),
                    mode="lines+markers",
                    name="Created",
                    line=dict(color="red", width=3),
                )
            )
            fig.update_layout(
                title="Cumulative Smell Changes",
                xaxis_title="Iteration",
                yaxis_title="Cumulative Count",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Refactoring types
            refactoring_types = refactoring_attempts_df[
                "refactoring_type"
            ].value_counts()
            fig = px.bar(
                x=refactoring_types.index,
                y=refactoring_types.values,
                title="Refactoring Types Applied",
                labels={"x": "Refactoring Type", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed attempts table
            st.subheader("Refactoring Attempts Details")
            st.dataframe(refactoring_attempts_df, use_container_width=True)
        else:
            st.info("No refactoring attempts recorded for this session")

    with tab3:
        st.subheader("Token Usage Analysis")

        if not token_usage_df.empty:
            # Total tokens by node
            node_tokens = (
                token_usage_df.groupby("node_name")["total_tokens"].sum().reset_index()
            )

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    node_tokens,
                    values="total_tokens",
                    names="node_name",
                    title="Token Distribution by Node",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Prompt vs completion tokens
                token_breakdown = (
                    token_usage_df.groupby("node_name")
                    .agg({"prompt_tokens": "sum", "completion_tokens": "sum"})
                    .reset_index()
                )

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=token_breakdown["node_name"],
                        y=token_breakdown["prompt_tokens"],
                        name="Prompt",
                        marker_color="#3498db",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=token_breakdown["node_name"],
                        y=token_breakdown["completion_tokens"],
                        name="Completion",
                        marker_color="#e74c3c",
                    )
                )
                fig.update_layout(
                    title="Token Breakdown by Node",
                    barmode="stack",
                    xaxis_title="Node",
                    yaxis_title="Tokens",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Token usage over iterations
            iteration_tokens = (
                token_usage_df.groupby("iteration")["total_tokens"].sum().reset_index()
            )
            fig = px.line(
                iteration_tokens,
                x="iteration",
                y="total_tokens",
                markers=True,
                title="Token Usage by Iteration",
                labels={"total_tokens": "Total Tokens", "iteration": "Iteration"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Model usage
            if "model" in token_usage_df.columns:
                model_usage = (
                    token_usage_df.groupby("model")["total_tokens"].sum().reset_index()
                )
                st.subheader("Model Usage")
                st.dataframe(model_usage, use_container_width=True)

            # Detailed token usage table
            st.subheader("Token Usage Details")
            st.dataframe(token_usage_df, use_container_width=True)
        else:
            st.info("No token usage data recorded for this session")

    with tab4:
        st.subheader("Session Statistics")

        session_stats = sessions_df[sessions_df["session_id"] == selected_session].iloc[
            0
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Iterations", int(session_stats["iterations"]))
            st.metric(
                "Successful Refactorings", int(session_stats["successful_refactorings"])
            )

        with col2:
            st.metric("Compile Failures", int(session_stats["compile_failures"]))
            st.metric("Test Failures", int(session_stats["test_failures"]))

        with col3:
            st.metric("Success Rate", f"{session_stats['success_rate']:.1f}%")
            st.metric(
                "Net Improvement",
                int(session_stats["net_improvement"]),
                delta=int(session_stats["net_improvement"]),
            )

        st.divider()

        st.subheader("Smell Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Smells Resolved", int(session_stats["total_resolved"]))

        with col2:
            st.metric("Smells Created", int(session_stats["total_created"]))

        with col3:
            st.metric("Total Tokens", f"{int(session_stats['total_tokens']):,}")

        # Efficiency metrics
        if session_stats["total_resolved"] > 0:
            tokens_per_smell = (
                session_stats["total_tokens"] / session_stats["total_resolved"]
            )
            st.metric("Tokens per Resolved Smell", f"{tokens_per_smell:,.0f}")


if __name__ == "__main__":
    main()
