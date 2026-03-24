"""Tests for workflows/common.py shared helpers."""

from unittest.mock import MagicMock

from workflows.common import save_agent_graph, print_eval_results


class TestSaveAgentGraph:
    def test_saves_png_on_success(self, tmp_path):
        agent = MagicMock()
        agent.get_graph.return_value.draw_mermaid_png.return_value = b"PNG_DATA"
        output = str(tmp_path / "graph.png")

        save_agent_graph(agent, output)

        with open(output, "rb") as f:
            assert f.read() == b"PNG_DATA"

    def test_prints_path_on_success(self, tmp_path, capsys):
        agent = MagicMock()
        agent.get_graph.return_value.draw_mermaid_png.return_value = b"X"
        output = str(tmp_path / "graph.png")

        save_agent_graph(agent, output)

        assert f"Graph saved to {output}" in capsys.readouterr().out

    def test_prints_error_on_failure(self, capsys):
        agent = MagicMock()
        agent.get_graph.return_value.draw_mermaid_png.side_effect = RuntimeError("boom")

        save_agent_graph(agent, "/tmp/noop.png")

        out = capsys.readouterr().out
        assert "Failed to draw graph" in out
        assert "boom" in out

    def test_prints_grandalf_hint_on_failure(self, capsys):
        agent = MagicMock()
        agent.get_graph.return_value.draw_mermaid_png.side_effect = Exception("missing")

        save_agent_graph(agent, "/tmp/noop.png")

        assert "grandalf" in capsys.readouterr().out


class TestPrintEvalResults:
    def _make_results(self, run_id="abc123", metrics=None, experiment_id="exp1"):
        r = MagicMock()
        r.run_id = run_id
        r.metrics = metrics or {"accuracy": 0.9, "count": 10}
        r.experiment_id = experiment_id
        return r

    def test_prints_header(self, capsys):
        print_eval_results(self._make_results())
        out = capsys.readouterr().out
        assert "EVALUATION RESULTS" in out

    def test_prints_float_metrics_with_4_decimals(self, capsys):
        print_eval_results(self._make_results(metrics={"score": 0.12345}))
        assert "0.1235" in capsys.readouterr().out

    def test_prints_non_float_metrics_as_is(self, capsys):
        print_eval_results(self._make_results(metrics={"count": 42}))
        assert "42" in capsys.readouterr().out

    def test_prints_run_id(self, capsys):
        print_eval_results(self._make_results(run_id="run99"))
        assert "run99" in capsys.readouterr().out

    def test_prints_url_when_http_tracking_uri(self, capsys):
        results = self._make_results(run_id="r1", experiment_id="e1")
        print_eval_results(results, tracking_uri="http://localhost:5000")
        assert "http://localhost:5000" in capsys.readouterr().out

    def test_no_url_when_na_run_id(self, capsys):
        results = self._make_results(run_id="N/A")
        print_eval_results(results, tracking_uri="http://localhost:5000")
        # URL line should not appear (run_id is N/A)
        out = capsys.readouterr().out
        assert "/#/experiments/" not in out

    def test_no_url_without_tracking_uri(self, capsys):
        print_eval_results(self._make_results(run_id="r1"))
        assert "/#/experiments/" not in capsys.readouterr().out
