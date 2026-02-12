"""Tests for SQLite connection retry logic with exponential backoff."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from giflab.storage import GifLabStorage


@pytest.fixture
def storage(tmp_path):
    """Create a GifLabStorage instance with a temporary database."""
    db_path = tmp_path / "test.db"
    return GifLabStorage(db_path)


class TestConnectRetryLogic:
    """Tests for _connect() retry with exponential backoff."""

    def test_connect_succeeds_on_first_attempt(self, storage):
        """Normal connection should work without retries."""
        with storage._connect() as conn:
            assert conn is not None
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1

    @patch("giflab.storage.time.sleep")
    @patch("giflab.storage.sqlite3.connect")
    def test_connect_retries_on_locked_database(
        self, mock_connect, mock_sleep, storage
    ):
        """Connection should retry with backoff when database is locked."""
        good_conn = MagicMock(spec=sqlite3.Connection)
        good_conn.execute = MagicMock()

        mock_connect.side_effect = [
            sqlite3.OperationalError("database is locked"),
            sqlite3.OperationalError("database is locked"),
            good_conn,
        ]

        with storage._connect() as conn:
            assert conn is good_conn

        assert mock_connect.call_count == 3
        assert mock_sleep.call_count == 2
        # Verify exponential backoff delays: 0.1s, 0.2s
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)

    @patch("giflab.storage.time.sleep")
    @patch("giflab.storage.sqlite3.connect")
    def test_connect_raises_after_all_retries_exhausted(
        self, mock_connect, mock_sleep, storage
    ):
        """Should raise RuntimeError after 5 failed attempts."""
        mock_connect.side_effect = sqlite3.OperationalError("database is locked")

        with pytest.raises(
            RuntimeError, match="Database locked after 5 retry attempts"
        ):
            with storage._connect():
                pass  # pragma: no cover

        assert mock_connect.call_count == 5
        assert mock_sleep.call_count == 4
        # Verify all backoff delays: 0.1, 0.2, 0.4, 0.8
        expected_delays = [0.1, 0.2, 0.4, 0.8]
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @patch("giflab.storage.time.sleep")
    @patch("giflab.storage.sqlite3.connect")
    def test_connect_reraises_non_lock_operational_error(
        self, mock_connect, mock_sleep, storage
    ):
        """Non-lock OperationalError should propagate immediately."""
        mock_connect.side_effect = sqlite3.OperationalError(
            "unable to open database file"
        )

        with pytest.raises(sqlite3.OperationalError, match="unable to open"):
            with storage._connect():
                pass  # pragma: no cover

        assert mock_connect.call_count == 1
        mock_sleep.assert_not_called()

    @patch("giflab.storage.time.sleep")
    @patch("giflab.storage.sqlite3.connect")
    def test_connect_exponential_backoff_values(
        self, mock_connect, mock_sleep, storage
    ):
        """Verify exact exponential backoff: 0.1, 0.2, 0.4, 0.8, then fail."""
        mock_connect.side_effect = sqlite3.OperationalError("database is locked")

        with pytest.raises(RuntimeError):
            with storage._connect():
                pass  # pragma: no cover

        expected_delays = [0.1, 0.2, 0.4, 0.8]
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @patch("giflab.storage.sqlite3.connect")
    def test_connect_passes_timeout_30(self, mock_connect, storage):
        """Connection should use timeout=30."""
        good_conn = MagicMock(spec=sqlite3.Connection)
        good_conn.execute = MagicMock()
        mock_connect.return_value = good_conn

        with storage._connect() as conn:
            assert conn is good_conn

        mock_connect.assert_called_once_with(storage.db_path, timeout=30)

    @patch("giflab.storage.time.sleep")
    @patch("giflab.storage.sqlite3.connect")
    def test_connect_succeeds_on_last_attempt(self, mock_connect, mock_sleep, storage):
        """Connection should succeed even on the 5th (last) attempt."""
        good_conn = MagicMock(spec=sqlite3.Connection)
        good_conn.execute = MagicMock()

        mock_connect.side_effect = [
            sqlite3.OperationalError("database is locked"),
            sqlite3.OperationalError("database is locked"),
            sqlite3.OperationalError("database is locked"),
            sqlite3.OperationalError("database is locked"),
            good_conn,
        ]

        with storage._connect() as conn:
            assert conn is good_conn

        assert mock_connect.call_count == 5
        assert mock_sleep.call_count == 4

    def test_connect_closes_connection_on_exit(self, storage):
        """Connection should be properly closed after context manager exits."""
        with storage._connect() as conn:
            # Connection is open and usable
            conn.execute("SELECT 1")

        # After exiting context, connection should be closed
        with pytest.raises(Exception):
            conn.execute("SELECT 1")
