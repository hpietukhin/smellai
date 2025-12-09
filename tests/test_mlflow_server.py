import signal
import subprocess
from unittest.mock import Mock, patch, MagicMock
import pytest
import requests
from mlflow_utils.server import MLflowServer


class TestMLflowServer:
    @pytest.fixture
    def server(self):
        return MLflowServer(port=5000)

    @patch("subprocess.check_output")
    def test_is_running_found(self, mock_check_output, server):
        mock_check_output.return_value = b"12345\n"
        assert server.is_running() == 12345
        mock_check_output.assert_called_once()

    @patch("subprocess.check_output")
    def test_is_running_multiple_pids(self, mock_check_output, server):
        # Test the fix for multiple PIDs
        mock_check_output.return_value = b"12345\n67890\n"
        assert server.is_running() == 12345

    @patch("subprocess.check_output")
    def test_is_running_not_found_empty(self, mock_check_output, server):
        mock_check_output.return_value = b""
        assert server.is_running() is None

    @patch("subprocess.check_output")
    def test_is_running_error(self, mock_check_output, server):
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "lsof")
        assert server.is_running() is None

    @patch("subprocess.check_output")
    def test_is_running_no_lsof(self, mock_check_output, server):
        mock_check_output.side_effect = FileNotFoundError
        assert server.is_running() is None

    @patch("mlflow_utils.server.MLflowServer.is_running")
    def test_start_already_running(self, mock_is_running, server):
        mock_is_running.return_value = 12345
        pid = server.start()
        assert pid == 12345

    @patch("subprocess.Popen")
    @patch("mlflow_utils.server.MLflowServer.is_running")
    def test_start_new_background(self, mock_is_running, mock_popen, server):
        mock_is_running.return_value = None
        mock_process = Mock()
        mock_process.pid = 54321
        mock_popen.return_value = mock_process

        # Mock open to avoid creating actual log files
        with patch("builtins.open", new_callable=MagicMock):
            pid = server.start(background=True)

        assert pid == 54321
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "mlflow" in args
        assert "ui" in args
        assert "--port" in args
        assert "5000" in args

    @patch("subprocess.run")
    @patch("mlflow_utils.server.MLflowServer.is_running")
    def test_start_foreground(self, mock_is_running, mock_run, server):
        mock_is_running.return_value = None
        server.start(background=False)
        mock_run.assert_called_once()

    @patch("os.kill")
    @patch("mlflow_utils.server.MLflowServer.is_running")
    def test_stop_running(self, mock_is_running, mock_kill, server):
        mock_is_running.return_value = 12345
        server.stop()
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    @patch("os.kill")
    @patch("mlflow_utils.server.MLflowServer.is_running")
    def test_stop_not_running(self, mock_is_running, mock_kill, server):
        mock_is_running.return_value = None
        server.stop()
        mock_kill.assert_not_called()

    @patch("requests.get")
    def test_wait_for_ready_success(self, mock_get, server):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert server.wait_for_ready(timeout=1) is True

    @patch("requests.get")
    def test_wait_for_ready_retry_then_success(self, mock_get, server):
        mock_response = Mock()
        mock_response.status_code = 200
        # Fail once then succeed
        mock_get.side_effect = [requests.ConnectionError, mock_response]

        # We need to mock time.sleep to speed up test
        with patch("time.sleep"):
            assert server.wait_for_ready(timeout=5) is True

        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_wait_for_ready_timeout(self, mock_get, server):
        mock_get.side_effect = requests.ConnectionError

        with patch("time.sleep"):
            assert server.wait_for_ready(timeout=0.1) is False
