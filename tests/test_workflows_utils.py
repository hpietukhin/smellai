"""Tests for workflows/utils.py shared utilities."""

import json
import logging
from unittest.mock import patch

import pytest

from workflows.utils import configure_logging, load_manifest, save_matplotlib_graph


class TestConfigureLogging:
    def test_sets_up_root_logger(self):
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_idempotent(self):
        # Calling twice should not raise
        configure_logging()
        configure_logging()


class TestLoadManifest:
    def test_loads_list_format(self, tmp_path):
        data = [{"id": 1}, {"id": 2}]
        f = tmp_path / "manifest.json"
        f.write_text(json.dumps(data))

        result = load_manifest(f)
        assert result == data

    def test_loads_dict_with_pairs_key(self, tmp_path):
        pairs = [{"id": "a"}, {"id": "b"}]
        f = tmp_path / "manifest.json"
        f.write_text(json.dumps({"pairs": pairs, "meta": "x"}))

        result = load_manifest(f)
        assert result == pairs

    def test_loads_plain_dict(self, tmp_path):
        data = {"files": [{"filename": "A.java"}], "smell_dependencies": {}}
        f = tmp_path / "manifest.json"
        f.write_text(json.dumps(data))

        result = load_manifest(f)
        assert result == data

    def test_exits_when_file_missing(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(SystemExit):
            load_manifest(missing)


class TestSaveMatplotlibGraph:
    def test_calls_plt_operations(self):
        """Verify save_matplotlib_graph delegates to matplotlib correctly."""
        with (
            patch("matplotlib.pyplot.axis") as m_axis,
            patch("matplotlib.pyplot.tight_layout") as m_tight,
            patch("matplotlib.pyplot.savefig") as m_savefig,
            patch("matplotlib.pyplot.close") as m_close,
        ):
            save_matplotlib_graph("out.png")

        m_axis.assert_called_once_with("off")
        m_tight.assert_called_once()
        m_savefig.assert_called_once_with("out.png")
        m_close.assert_called_once()
