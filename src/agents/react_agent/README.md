# DACOS LangGraph Agent

This module wraps the LangGraph-based ReAct agent that analyses DACOS
samples. It consumes MySQL-backed metadata via `src.data.mysql_connector`
and produces structured `CodeAnalysisResult` payloads defined in
`schema.py`.

To run the graph with the LangGraph CLI:

1. Ensure the MySQL credentials and optional Git settings live in your
   `.env` file.
2. Install the project dependencies (`poetry install` or `pip install -e .`).
3. Use `langgraph dev` or `langgraph run agent` from the repository root.
